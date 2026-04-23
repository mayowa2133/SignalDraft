from __future__ import annotations

import hmac

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.utils.config import settings

bearer_scheme = HTTPBearer(auto_error=False)


def require_api_token(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> None:
    if not settings.backend_auth_enabled:
        return

    token = credentials.credentials if credentials and credentials.scheme.lower() == "bearer" else ""
    if token and hmac.compare_digest(token, settings.api_token):
        return

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail={
            "code": "invalid_api_token",
            "message": "A valid API bearer token is required for this endpoint.",
        },
        headers={"WWW-Authenticate": "Bearer"},
    )

