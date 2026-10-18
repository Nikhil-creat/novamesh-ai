"""PyTorch CNN vision pipeline: async frame buffer -> ResNet-50 512-d embeddings, saliency boxes, anomaly score."""
from __future__ import annotations

import asyncio
import io
import logging
import time
from dataclasses import dataclass
from urllib.parse import urlparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T
from torchvision.models import ResNet50_Weights, resnet50

from src.config import STREAM_VISION, Settings
from src.core import metrics
from src.core.redis_bus import RedisBus
from src.schemas import BoundingBox, VisionAnalysis, VisionFrameEvent

log = logging.getLogger(__name__)
Image.MAX_IMAGE_PIXELS = 50_000_000
EMBED_DIM = 512
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "BMP"}


class VisionError(Exception):
    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


class FeatureExtractor(nn.Module):
    """ResNet-50 trunk -> global pool (2048) -> fixed Gaussian random projection (JL) -> L2-normalised 512-d."""

    def __init__(self, pretrained: bool) -> None:
        super().__init__()
        base = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        self.trunk = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool, base.layer1, base.layer2, base.layer3, base.layer4)
        gen = torch.Generator().manual_seed(1337)
        self.register_buffer("proj", torch.randn(EMBED_DIM, 2048, generator=gen) / (2048**0.5))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        fmap = self.trunk(x)
        pooled = fmap.mean(dim=(2, 3))
        return F.normalize(pooled @ self.proj.T, dim=1), fmap  # type: ignore[operator]


class FeatureBaseline:
    """Online (Welford) diagonal-Gaussian baseline of 'normal' scenes; anomaly = scaled mean squared z-score."""

    def __init__(self, warmup: int, scale: float = 2.0) -> None:
        self.n = 0
        self.mean = np.zeros(EMBED_DIM)
        self.m2 = np.zeros(EMBED_DIM)
        self.warmup, self.scale = warmup, scale

    @property
    def ready(self) -> bool:
        return self.n >= self.warmup

    def score(self, f: np.ndarray) -> float:
        if not self.ready:
            return 0.0
        var = self.m2 / max(self.n - 1, 1)
        z2 = float(np.mean((f - self.mean) ** 2 / (var + 1e-8)))
        return float(1.0 - np.exp(-max(z2 - 1.0, 0.0) / self.scale))

    def update(self, f: np.ndarray) -> None:
        self.n += 1
        d = f - self.mean
        self.mean += d / self.n
        self.m2 += d * (f - self.mean)


def boxes_from_saliency(sal: np.ndarray, thr: float = 0.6, max_boxes: int = 5) -> list[BoundingBox]:
    """Connected components over the thresholded activation map -> normalised boxes (regions of interest)."""
    h, w = sal.shape
    mask, seen = sal >= thr, np.zeros(sal.shape, dtype=bool)
    out: list[BoundingBox] = []
    for i in range(h):
        for j in range(w):
            if not mask[i, j] or seen[i, j]:
                continue
            stack, cells = [(i, j)], []
            seen[i, j] = True
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            ys, xs = [c[0] for c in cells], [c[1] for c in cells]
            out.append(BoundingBox(
                x1=min(xs) / w, y1=min(ys) / h, x2=(max(xs) + 1) / w, y2=(max(ys) + 1) / h,
                score=float(np.mean([sal[c] for c in cells])),
            ))
    out.sort(key=lambda b: b.score, reverse=True)
    return out[:max_boxes]


@dataclass(slots=True)
class _Job:
    data: bytes
    source_id: str
    future: asyncio.Future[VisionAnalysis]
    enqueued: float


class VisionEngine:
    MODEL_NAME = "resnet50-proj512"

    def __init__(self, settings: Settings, bus: RedisBus) -> None:
        self._s, self._bus = settings, bus
        self._queue: asyncio.Queue[_Job] = asyncio.Queue(maxsize=settings.vision_queue_size)
        self._worker: asyncio.Task[None] | None = None
        self._model: FeatureExtractor | None = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        size = settings.vision_input_size
        self._tf = T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.baseline = FeatureBaseline(settings.vision_warmup_frames)
        self.paused = False
        self.quarantined: set[str] = set()
        self.frames = 0
        self.anomalies = 0
        self.last_latency_ms = 0.0
        self._ewma_latency = 0.0

    async def start(self) -> None:
        self._model = await asyncio.to_thread(self._load)
        self._worker = asyncio.create_task(self._run(), name="vision-worker")
        log.info("vision engine ready on %s", self._device)

    def _load(self) -> FeatureExtractor:
        model = FeatureExtractor(self._s.vision_pretrained).eval().to(self._device)
        with torch.inference_mode():  # warm-up pass
            model(torch.zeros(1, 3, self._s.vision_input_size, self._s.vision_input_size, device=self._device))
        return model

    async def stop(self) -> None:
        if self._worker:
            self._worker.cancel()
            await asyncio.gather(self._worker, return_exceptions=True)
        while not self._queue.empty():
            job = self._queue.get_nowait()
            if not job.future.done():
                job.future.set_exception(VisionError("engine shutting down", 503))

    # ---------------------------------------------------------- controls
    def pause(self) -> None:
        self.paused = True

    def resume(self) -> None:
        self.paused = False

    def quarantine(self, source_id: str) -> None:
        self.quarantined.add(source_id)

    def stats(self) -> dict[str, float | int | bool]:
        return {
            "frames": self.frames, "anomalies": self.anomalies, "queue_depth": self._queue.qsize(),
            "last_latency_ms": round(self.last_latency_ms, 2), "ewma_latency_ms": round(self._ewma_latency, 2),
            "baseline_frames": self.baseline.n, "baseline_ready": self.baseline.ready,
            "paused": self.paused, "quarantined_sources": len(self.quarantined),
        }

    # ---------------------------------------------------------- ingestion
    async def analyze(self, data: bytes, source_id: str) -> VisionAnalysis:
        if self.paused:
            raise VisionError("vision ingest is paused by an administrator", 503)
        if source_id in self.quarantined:
            raise VisionError(f"source '{source_id}' is quarantined", 403)
        fut: asyncio.Future[VisionAnalysis] = asyncio.get_running_loop().create_future()
        try:
            self._queue.put_nowait(_Job(data, source_id, fut, time.perf_counter()))
        except asyncio.QueueFull as exc:
            metrics.VISION_FRAMES.labels("dropped").inc()
            raise VisionError("vision queue saturated, retry shortly", 429) from exc
        metrics.VISION_QUEUE.set(self._queue.qsize())
        return await fut

    async def ingest_rtsp(self, url: str, source_id: str, max_frames: int = 5) -> list[VisionAnalysis]:
        parsed = urlparse(url)
        if parsed.scheme not in {"rtsp", "rtsps"} or not parsed.hostname:
            raise VisionError("only rtsp:// or rtsps:// URLs are accepted")
        allowed = self._s.rtsp_hosts
        if allowed and parsed.hostname.lower() not in allowed:
            raise VisionError("RTSP host is not in RTSP_ALLOWED_HOSTS", 403)
        frames = await asyncio.to_thread(self._grab_rtsp, url, max(1, min(max_frames, 30)))
        if not frames:
            raise VisionError("no frames could be read from the stream", 502)
        return [await self.analyze(f, source_id) for f in frames]

    @staticmethod
    def _grab_rtsp(url: str, n: int, stride: int = 5) -> list[bytes]:
        import cv2

        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG, [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000])
        frames: list[bytes] = []
        try:
            if not cap.isOpened():
                return []
            idx, deadline = 0, time.monotonic() + 30
            while len(frames) < n and time.monotonic() < deadline:
                ok, frame = cap.read()
                if not ok:
                    break
                idx += 1
                if idx % stride:
                    continue
                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                if ok:
                    frames.append(buf.tobytes())
        finally:
            cap.release()
        return frames

    # ---------------------------------------------------------- inference
    def _infer(self, data: bytes) -> tuple[list[float], list[BoundingBox], float, float]:
        assert self._model is not None
        t0 = time.perf_counter()
        probe = Image.open(io.BytesIO(data))
        if probe.format not in ALLOWED_FORMATS:
            raise VisionError(f"unsupported image format {probe.format}", 415)
        img = Image.open(io.BytesIO(data)).convert("RGB")
        x = self._tf(img).unsqueeze(0).to(self._device)
        with torch.inference_mode():
            feat, fmap = self._model(x)
            sal = fmap[0].norm(dim=0)
            sal = (sal / sal.max().clamp_min(1e-8)).cpu().numpy()
        f = feat[0].cpu().numpy().astype(np.float64)
        score = self.baseline.score(f)
        if score < self._s.vision_anomaly_threshold:
            self.baseline.update(f)  # only learn from normal frames (anti-poisoning)
        return [round(float(v), 5) for v in f], boxes_from_saliency(sal), score, (time.perf_counter() - t0) * 1000

    async def _run(self) -> None:
        while True:
            job = await self._queue.get()
            metrics.VISION_QUEUE.set(self._queue.qsize())
            try:
                emb, boxes, score, infer_ms = await asyncio.to_thread(self._infer, job.data)
                total_ms = (time.perf_counter() - job.enqueued) * 1000
                peak = boxes[0].score if boxes else 0.0
                event = VisionFrameEvent(
                    source_id=job.source_id, anomaly_score=round(score, 4), confidence=round(0.7 * score + 0.3 * peak * score, 4),
                    boxes=boxes, embedding=emb, model_name=self.MODEL_NAME, latency_ms=round(total_ms, 2),
                )
                anomalous = score >= self._s.vision_anomaly_threshold
                published = False
                self.frames += 1
                self.last_latency_ms = total_ms
                self._ewma_latency = total_ms if self._ewma_latency == 0 else 0.9 * self._ewma_latency + 0.1 * total_ms
                metrics.VISION_LATENCY.observe(total_ms)
                metrics.VISION_SCORE.observe(score)
                metrics.VISION_FRAMES.labels("anomalous" if anomalous else "normal").inc()
                if anomalous:
                    self.anomalies += 1
                    await self._bus.publish(STREAM_VISION, event)
                    published = True
                await self._bus.publish_telemetry("cnn_latency", {
                    "source_id": job.source_id, "latency_ms": round(total_ms, 2), "infer_ms": round(infer_ms, 2),
                    "anomaly_score": event.anomaly_score, "anomalous": anomalous,
                })
                if not job.future.done():
                    job.future.set_result(VisionAnalysis(event=event, anomalous=anomalous, published=published))
            except asyncio.CancelledError:
                raise
            except VisionError as exc:
                metrics.VISION_FRAMES.labels("rejected").inc()
                if not job.future.done():
                    job.future.set_exception(exc)
            except Exception as exc:  # noqa: BLE001 - never kill the worker on a bad frame
                log.exception("frame inference failed")
                metrics.VISION_FRAMES.labels("error").inc()
                if not job.future.done():
                    job.future.set_exception(VisionError(f"could not process image: {type(exc).__name__}", 422))
