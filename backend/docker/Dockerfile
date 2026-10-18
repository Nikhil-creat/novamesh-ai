# syntax=docker/dockerfile:1.7
FROM python:3.12-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /build
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt && python -m spacy download en_core_web_sm

FROM python:3.12-slim AS runtime
ENV PATH="/opt/venv/bin:$PATH" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HF_HOME=/home/novamesh/.cache/huggingface TORCH_HOME=/home/novamesh/.cache/torch
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --uid 10001 novamesh \
 && mkdir -p /home/novamesh/.cache && chown -R novamesh:novamesh /home/novamesh
COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY --chown=novamesh:novamesh src ./src
USER novamesh
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=120s --retries=5 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz',timeout=3).status==200 else 1)"
# Single worker: consensus state, LangGraph checkpoints and the vision baseline are in-process.
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--loop", "uvloop", "--http", "httptools", "--workers", "1"]
