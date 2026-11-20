"""FastAPI security middleware: safe Origin/CORS validation, ownership auth, and TTL rate limiter.

Fixes:
- BUG-13: Centralized ownership authorization (`load_owned_run`) on all mutating routes.
- BUG-20: Safe Origin parsing returning HTTP 403 (never 500) on `Origin: null` or malformed URLs.
- Hardening: Bounded, TTL-pruned in-memory sliding-window rate limiter with zero unbounded growth.
"""

from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Callable, Sequence
from urllib.parse import urlparse

from fastapi import Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from gtm_videogen.core.config import get_settings
from gtm_videogen.core.run_store import RunStore
from gtm_videogen.models.schemas import RunState

ACTOR_HEADER = "X-Actor-Id"
DEFAULT_ACTOR = "local"
MAX_RUN_ID_LENGTH = 128
DEFAULT_LOCAL_HOSTS: frozenset[str] = frozenset({"localhost", "127.0.0.1", "::1", "testserver"})

_CORS_ALLOW_METHODS = "GET, POST, PUT, PATCH, DELETE, OPTIONS"
_CORS_ALLOW_HEADERS = "Content-Type, Authorization, X-Actor-Id, Range"
_CORS_EXPOSE_HEADERS = "Content-Range, Accept-Ranges, Content-Length, Retry-After"


def _has_control_or_whitespace(value: str) -> bool:
    return any(ord(ch) <= 0x20 or ord(ch) == 0x7F for ch in value)


def _normalize_origin(raw: str) -> str | None:
    """Safely parse and normalize an HTTP(S) Origin header. Returns None if malformed."""
    if not raw or raw.strip().lower() == "null":
        return None
    if _has_control_or_whitespace(raw):
        return None

    try:
        parsed = urlparse(raw)
        port = parsed.port
    except ValueError:
        return None

    scheme = parsed.scheme.lower()
    if scheme not in ("http", "https"):
        return None
    if not parsed.netloc or not parsed.hostname:
        return None
    if parsed.username is not None or parsed.password is not None:
        return None
    if parsed.path != "" or parsed.params or parsed.query or parsed.fragment:
        return None

    host = parsed.hostname.lower()
    host_fmt = f"[{host}]" if ":" in host else host
    default_port = 80 if scheme == "http" else 443
    if port is not None and port != default_port:
        return f"{scheme}://{host_fmt}:{port}"
    return f"{scheme}://{host_fmt}"


def validate_origin(
    origin: str,
    *,
    allowed_origins: Sequence[str] | None = None,
) -> tuple[bool, str, str | None]:
    """Validate an Origin header against syntax rules and allowlist (BUG-20).

    Returns `(is_valid, reason, normalized_origin)`.
    """
    if not isinstance(origin, str) or not origin:
        return False, "Empty Origin header is forbidden", None
    if origin.strip().lower() == "null":
        return False, "Origin 'null' is forbidden", None

    normalized = _normalize_origin(origin)
    if normalized is None:
        return False, "Malformed Origin header", None

    if allowed_origins is not None:
        if "*" in allowed_origins:
            return True, "ok", normalized
        normalized_allowlist = {
            norm for item in allowed_origins if (norm := _normalize_origin(item)) is not None
        }
        if normalized in normalized_allowlist:
            return True, "ok", normalized
        return False, f"Origin {origin!r} is not in allowed origins", None

    # Default policy: allow configured Settings origins + local development/test hosts
    configured = get_settings().allowed_origins_list()
    normalized_configured = {
        norm for item in configured if (norm := _normalize_origin(item)) is not None
    }
    if normalized in normalized_configured:
        return True, "ok", normalized

    parsed = urlparse(normalized)
    if parsed.hostname and parsed.hostname.lower() in DEFAULT_LOCAL_HOSTS:
        return True, "ok", normalized

    return False, f"Origin {origin!r} is not allowed", None


class OriginSecurityMiddleware(BaseHTTPMiddleware):
    """Rejects malformed / `Origin: null` / disallowed Origins with HTTP 403 (BUG-20)."""

    def __init__(
        self,
        app,  # type: ignore[no-untyped-def]
        *,
        allowed_origins: Sequence[str] | None = None,
    ) -> None:
        super().__init__(app)
        self._allowed_origins = list(allowed_origins) if allowed_origins is not None else None

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        origin = request.headers.get("origin")
        if origin is None:
            return await call_next(request)

        valid, reason, _ = validate_origin(origin, allowed_origins=self._allowed_origins)
        if not valid and self._allowed_origins is None:
            normalized = _normalize_origin(origin)
            if normalized is not None:
                origin_netloc = urlparse(normalized).netloc.lower()
                req_hosts = {
                    h.strip().lower()
                    for hdr in (
                        request.headers.get("host", ""),
                        request.headers.get("x-forwarded-host", ""),
                    )
                    for h in hdr.split(",")
                    if h.strip()
                }
                if origin_netloc and origin_netloc in req_hosts:
                    valid = True
        if not valid:
            return JSONResponse(status_code=403, content={"detail": reason})

        cors_headers = {
            "Access-Control-Allow-Origin": origin,
            "Vary": "Origin",
            "Access-Control-Allow-Methods": _CORS_ALLOW_METHODS,
            "Access-Control-Allow-Headers": _CORS_ALLOW_HEADERS,
            "Access-Control-Expose-Headers": _CORS_EXPOSE_HEADERS,
        }

        if request.method == "OPTIONS":
            return Response(status_code=204, headers=cors_headers)

        response = await call_next(request)
        for key, value in cors_headers.items():
            response.headers[key] = value
        return response


class TTLRateLimiter:
    """Bounded, TTL-pruned in-memory sliding-window rate limiter."""

    def __init__(
        self,
        *,
        max_requests: int = 120,
        window_sec: float = 60.0,
        max_keys: int = 10_000,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if window_sec <= 0:
            raise ValueError("window_sec must be > 0")
        if max_keys < 1:
            raise ValueError("max_keys must be >= 1")
        self.max_requests = max_requests
        self.window_sec = window_sec
        self.max_keys = max_keys
        self._clock = clock
        self._buckets: dict[str, deque[float]] = {}
        self._last_prune: float = self._clock()

    @property
    def active_keys(self) -> int:
        return len(self._buckets)

    def prune(self, now: float | None = None, *, target_size: int | None = None) -> int:
        """Evict all expired timestamps and empty/stale keys; enforce max_keys bound."""
        ts = self._clock() if now is None else now
        cutoff = ts - self.window_sec
        removed = 0

        for key in list(self._buckets.keys()):
            bucket = self._buckets[key]
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()
            if not bucket:
                del self._buckets[key]
                removed += 1

        limit = self.max_keys if target_size is None else max(0, min(target_size, self.max_keys))
        if len(self._buckets) > limit:
            # Evict keys with the oldest most-recent activity first
            excess = len(self._buckets) - limit
            oldest_keys = sorted(
                self._buckets.keys(),
                key=lambda k: self._buckets[k][-1] if self._buckets[k] else 0.0,
            )[:excess]
            for key in oldest_keys:
                del self._buckets[key]
                removed += 1

        self._last_prune = ts
        return removed

    def check(self, key: str) -> tuple[bool, int]:
        """Record a request for `key` if allowed. Returns `(allowed, retry_after_sec)`."""
        if self.max_requests <= 0:
            return True, 0

        now = self._clock()
        is_new_key = key not in self._buckets
        if (now - self._last_prune) >= self.window_sec or (
            is_new_key and len(self._buckets) >= self.max_keys
        ):
            target = self.max_keys - 1 if is_new_key else self.max_keys
            self.prune(now, target_size=target)

        cutoff = now - self.window_sec
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = deque()
            self._buckets[key] = bucket
        else:
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()

        if len(bucket) >= self.max_requests:
            oldest = bucket[0]
            retry_after = max(1, math.ceil(self.window_sec - (now - oldest)))
            return False, retry_after

        bucket.append(now)
        return True, 0


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Enforces TTLRateLimiter per actor or client IP, returning HTTP 429 + Retry-After."""

    def __init__(
        self,
        app,  # type: ignore[no-untyped-def]
        *,
        limiter: TTLRateLimiter | None = None,
    ) -> None:
        super().__init__(app)
        self._limiter = limiter

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        if self._limiter is None or self._limiter.max_requests <= 0:
            return await call_next(request)

        actor = request.headers.get(ACTOR_HEADER, "").strip()
        client_host = request.client.host if request.client else "unknown"
        key = f"actor:{actor}" if actor else f"ip:{client_host}"

        allowed, retry_after = self._limiter.check(key)
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Please retry later."},
                headers={"Retry-After": str(retry_after)},
            )
        return await call_next(request)


def get_actor(
    x_actor_id: str | None = Header(default=None, alias=ACTOR_HEADER),
) -> str:
    """Extract actor identity from `X-Actor-Id` header (defaults to `'local'`)."""
    if x_actor_id is not None and x_actor_id.strip():
        return x_actor_id.strip()
    return DEFAULT_ACTOR


def load_owned_run(run_id: str, run_store: RunStore, actor: str) -> RunState:
    """Load RunState and verify `run.created_by == actor` (BUG-13).

    Raises HTTP 400/404 if the run_id is invalid or not found, or HTTP 403 if actor does not own it.
    """
    if not run_id or len(run_id) > MAX_RUN_ID_LENGTH:
        raise HTTPException(status_code=400, detail="Invalid run_id")
    try:
        run_state = run_store.load(run_id)
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=404, detail="Invalid run_id") from exc

    if run_state is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found")

    if run_state.created_by != actor:
        raise HTTPException(
            status_code=403,
            detail=f"Forbidden: actor {actor!r} does not own run {run_id!r}",
        )
    return run_state
