import time

from fastapi import Depends, Header, Request
from jose import jwt
from redis.exceptions import RedisError

from ..core.config import Settings, get_settings
from ..core.errors import AppError
from ..schemas.domain import AuthContext


def _dev_auth(x_tenant_id: str | None, x_user_id: str | None, x_groups: str) -> AuthContext:
    if not x_tenant_id or not x_user_id:
        raise AppError(401, "AUTH_REQUIRED", "X-Tenant-ID and X-User-ID are required")
    return AuthContext(
        subject=x_user_id,
        tenant_id=x_tenant_id,
        groups=frozenset(filter(None, (item.strip() for item in x_groups.split(",")))),
    )


async def _cached_jwks(request: Request, settings: Settings) -> dict:
    if not getattr(request.app.state, "jwks", None):
        response = await request.app.state.http.get(settings.oidc_jwks_url)
        response.raise_for_status()
        request.app.state.jwks = response.json()
    return request.app.state.jwks


async def _oidc_auth(
    request: Request, authorization: str | None, settings: Settings
) -> AuthContext:
    if not authorization or not authorization.startswith("Bearer ") or not settings.oidc_jwks_url:
        raise AppError(401, "AUTH_REQUIRED", "A valid bearer token is required")
    try:
        claims = jwt.decode(
            authorization.removeprefix("Bearer "),
            await _cached_jwks(request, settings),
            algorithms=["RS256"],
            audience=settings.oidc_audience,
            issuer=settings.oidc_issuer,
        )
        return AuthContext(
            subject=claims["sub"],
            tenant_id=claims["tenant_id"],
            groups=frozenset(claims.get("groups", [])),
        )
    except Exception as exc:
        raise AppError(401, "INVALID_TOKEN", "The bearer token is invalid") from exc


async def auth_context(
    request: Request,
    authorization: str | None = Header(default=None),
    x_tenant_id: str | None = Header(default=None),
    x_user_id: str | None = Header(default=None),
    x_groups: str = Header(default=""),
    settings: Settings = Depends(get_settings),
) -> AuthContext:
    if settings.dev_auth:
        auth = _dev_auth(x_tenant_id, x_user_id, x_groups)
    else:
        auth = await _oidc_auth(request, authorization, settings)
    await enforce_rate_limit(request, auth, settings)
    return auth


async def enforce_rate_limit(request: Request, auth: AuthContext, settings: Settings) -> None:
    redis = getattr(request.app.state, "redis", None)
    if not redis:
        return
    key = f"rate:{auth.tenant_id}:{auth.subject}:{int(time.time() // 60)}"
    try:
        pipeline = redis.pipeline()
        pipeline.incr(key)
        pipeline.expire(key, 120)
        count = (await pipeline.execute())[0]
    except (AttributeError, RedisError):
        return
    if count > settings.requests_per_minute:
        raise AppError(429, "RATE_LIMITED", "Request limit exceeded; retry next minute")
