"""JWT issuance/verification and FastAPI RBAC dependencies."""
from __future__ import annotations

import hmac
import time
from collections.abc import Callable
from typing import Annotated

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.config import Settings
from src.schemas import Role, TokenClaims


class AuthError(Exception):
    pass


class AuthService:
    def __init__(self, settings: Settings) -> None:
        self._s = settings
        self._secret = settings.jwt_secret.get_secret_value()

    def issue(self, subject: str, role: Role, ttl_minutes: int | None = None) -> tuple[str, int]:
        ttl = (ttl_minutes or self._s.jwt_ttl_minutes) * 60
        now = int(time.time())
        token = jwt.encode(
            {"sub": subject, "role": role.value, "iat": now, "exp": now + ttl, "iss": self._s.jwt_issuer, "aud": self._s.jwt_audience},
            self._secret, algorithm=self._s.jwt_algorithm,
        )
        return token, ttl

    def verify(self, token: str) -> TokenClaims:
        try:
            data = jwt.decode(
                token, self._secret, algorithms=[self._s.jwt_algorithm], audience=self._s.jwt_audience,
                issuer=self._s.jwt_issuer, options={"require": ["exp", "sub", "iss", "aud"]},
            )
            return TokenClaims(sub=data["sub"], role=Role(data["role"]), exp=data["exp"])
        except (jwt.PyJWTError, ValueError, KeyError) as exc:
            raise AuthError("invalid or expired token") from exc

    def check_admin_credentials(self, username: str, password: str) -> bool:
        ok_user = hmac.compare_digest(username.encode(), self._s.bootstrap_admin_user.encode())
        ok_pw = hmac.compare_digest(password.encode(), self._s.bootstrap_admin_password.get_secret_value().encode())
        return ok_user and ok_pw


_bearer = HTTPBearer(auto_error=False)


async def current_claims(request: Request, creds: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)]) -> TokenClaims:
    if creds is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing bearer token", headers={"WWW-Authenticate": "Bearer"})
    try:
        return request.app.state.container.auth.verify(creds.credentials)
    except AuthError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, str(exc), headers={"WWW-Authenticate": "Bearer"}) from exc


def require(min_role: Role) -> Callable[..., TokenClaims]:
    async def dep(claims: Annotated[TokenClaims, Depends(current_claims)]) -> TokenClaims:
        if claims.role.rank < min_role.rank:
            raise HTTPException(status.HTTP_403_FORBIDDEN, f"requires role >= {min_role.value}")
        return claims

    return dep
