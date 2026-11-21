"""Resilient async HTTP client with timeouts, retries, SSRF guards, and cancellation.

Fixes BUG-14:
- Explicit connect/read/write/pool timeouts on every request.
- Strictly calls response.raise_for_status() BEFORE response.json().
- Exponential backoff with jitter on 429/5xx/network errors.
- Propagates CancellationToken to abort in-flight requests and retry waits immediately.
"""

from __future__ import annotations

import asyncio
import random
from typing import Any

import httpx

from gtm_videogen.core.job_registry import CancellationToken
from gtm_videogen.core.ssrf import MAX_REDIRECT_HOPS, assert_public_redirect, assert_public_url

RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
REDIRECT_STATUS_CODES = frozenset({301, 302, 303, 307, 308})
_SENSITIVE_REDIRECT_HEADERS = frozenset({"authorization", "proxy-authorization", "cookie"})


def _origin_tuple(url: str) -> tuple[str, str, int | None]:
    """Return canonical (scheme, host, effective_port) origin tuple for a URL."""
    parsed = httpx.URL(url)
    scheme = parsed.scheme.lower()
    host = (parsed.host or "").lower()
    port = parsed.port
    if port is None:
        if scheme == "https":
            port = 443
        elif scheme == "http":
            port = 80
    return (scheme, host, port)


class ResilientHttpClient:
    """Async httpx wrapper with timeouts, SSRF checks, retries, and cancellation."""

    def __init__(
        self,
        *,
        connect_timeout: float = 15.0,
        read_timeout: float = 90.0,
        write_timeout: float = 30.0,
        pool_timeout: float = 15.0,
        max_retries: int = 3,
        backoff_base_sec: float = 0.5,
        backoff_max_sec: float = 8.0,
        transport: httpx.AsyncBaseTransport | None = None,
        resolve_dns_for_ssrf: bool = True,
    ) -> None:
        self._timeout = httpx.Timeout(
            connect=connect_timeout,
            read=read_timeout,
            write=write_timeout,
            pool=pool_timeout,
        )
        self._max_retries = max_retries
        self._backoff_base_sec = backoff_base_sec
        self._backoff_max_sec = backoff_max_sec
        self._resolve_dns_for_ssrf = resolve_dns_for_ssrf
        self._client = httpx.AsyncClient(
            timeout=self._timeout,
            follow_redirects=False,
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> ResilientHttpClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
        json: Any | None = None,
        content: bytes | str | None = None,
        public_only: bool = False,
        max_redirects: int = MAX_REDIRECT_HOPS,
        cancel: CancellationToken | None = None,
    ) -> httpx.Response:
        """Execute an HTTP request with retry backoff, optional SSRF checks, and cancellation."""
        if public_only:
            url = assert_public_url(url, resolve_dns=self._resolve_dns_for_ssrf)

        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            if cancel is not None:
                cancel.raise_if_cancelled()

            try:
                response = await self._send_with_redirects(
                    method=method,
                    url=url,
                    headers=headers,
                    params=params,
                    json=json,
                    content=content,
                    public_only=public_only,
                    max_redirects=max_redirects,
                    cancel=cancel,
                )
            except asyncio.CancelledError:
                raise
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_exc = exc
                if attempt >= self._max_retries:
                    raise
                await self._sleep_backoff(attempt, cancel=cancel)
                continue

            if response.status_code in RETRYABLE_STATUS_CODES and attempt < self._max_retries:
                retry_after = self._parse_retry_after(response)
                await self._sleep_backoff(attempt, retry_after=retry_after, cancel=cancel)
                continue

            if response.is_error:
                body_snippet = ""
                try:
                    err_json = response.json()
                    if isinstance(err_json, dict):
                        detail = err_json.get("detail") or err_json.get("message") or err_json.get("error")
                        if detail:
                            body_snippet = str(detail)[:300]
                except Exception:
                    pass
                if not body_snippet:
                    body_snippet = (response.text or "").strip()[:300]
                if body_snippet:
                    raise httpx.HTTPStatusError(
                        f"HTTP {response.status_code} {response.reason_phrase} for {response.url}: {body_snippet}",
                        request=response.request,
                        response=response,
                    )
                response.raise_for_status()
            return response

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("ResilientHttpClient exhausted retries without a response")

    async def _send_with_redirects(
        self,
        *,
        method: str,
        url: str,
        headers: dict[str, str] | None,
        params: dict[str, Any] | None,
        json: Any | None,
        content: bytes | str | None,
        public_only: bool,
        max_redirects: int,
        cancel: CancellationToken | None,
    ) -> httpx.Response:
        current_url = url
        current_method = method
        current_headers = dict(headers) if headers is not None else None
        current_params = params
        current_json = json
        current_content = content

        for hop in range(max_redirects + 1):
            if cancel is not None:
                cancel.raise_if_cancelled()

            prev_origin = _origin_tuple(current_url)
            req_coro = self._client.request(
                current_method,
                current_url,
                headers=current_headers,
                params=current_params,
                json=current_json,
                content=current_content,
            )
            response = await self._await_cancellable(req_coro, cancel=cancel)

            if response.status_code not in REDIRECT_STATUS_CODES:
                return response

            location = response.headers.get("location")
            if not location:
                return response

            next_hop = hop + 1
            if public_only:
                current_url = assert_public_redirect(
                    current_url,
                    location,
                    hop=next_hop,
                    max_hops=max_redirects,
                    resolve_dns=self._resolve_dns_for_ssrf,
                )
            else:
                if next_hop > max_redirects:
                    raise httpx.TooManyRedirects(
                        f"Exceeded maximum redirects ({max_redirects})",
                        request=response.request,
                    )
                current_url = str(response.url.join(location))

            if _origin_tuple(current_url) != prev_origin and current_headers is not None:
                current_headers = {
                    k: v
                    for k, v in current_headers.items()
                    if k.lower() not in _SENSITIVE_REDIRECT_HEADERS
                }

            if response.status_code == 303:
                current_method = "GET"
                current_json = None
                current_content = None
            current_params = None

        return response

    async def get_json(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
        public_only: bool = False,
        cancel: CancellationToken | None = None,
    ) -> Any:
        """GET a URL, calling raise_for_status() strictly before .json()."""
        response = await self.request(
            "GET",
            url,
            headers=headers,
            params=params,
            public_only=public_only,
            cancel=cancel,
        )
        return response.json()

    async def post_json(
        self,
        url: str,
        *,
        json: Any | None = None,
        headers: dict[str, str] | None = None,
        public_only: bool = False,
        cancel: CancellationToken | None = None,
    ) -> Any:
        """POST JSON to a URL, calling raise_for_status() strictly before .json()."""
        response = await self.request(
            "POST",
            url,
            headers=headers,
            json=json,
            public_only=public_only,
            cancel=cancel,
        )
        return response.json()

    async def get_bytes(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        public_only: bool = False,
        max_bytes: int | None = None,
        cancel: CancellationToken | None = None,
    ) -> bytes:
        """Download binary content with optional max_bytes size cap."""
        response = await self.request(
            "GET",
            url,
            headers=headers,
            public_only=public_only,
            cancel=cancel,
        )
        data = response.content
        if max_bytes is not None and len(data) > max_bytes:
            raise ValueError(f"Response size {len(data)} exceeds limit of {max_bytes} bytes")
        return data

    async def get_text(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        public_only: bool = False,
        max_bytes: int = 2 * 1024 * 1024,
        cancel: CancellationToken | None = None,
    ) -> str:
        """Download text content with SSRF and size guardrails."""
        response = await self.request(
            "GET",
            url,
            headers=headers,
            public_only=public_only,
            cancel=cancel,
        )
        if len(response.content) > max_bytes:
            raise ValueError(
                f"Response size {len(response.content)} exceeds limit of {max_bytes} bytes"
            )
        return response.text

    @staticmethod
    async def _await_cancellable(
        coro: Any,
        *,
        cancel: CancellationToken | None,
    ) -> httpx.Response:
        if cancel is None:
            return await coro

        cancel.raise_if_cancelled()
        req_task: asyncio.Task[httpx.Response] = asyncio.create_task(coro)
        cancel_task: asyncio.Task[None] = asyncio.create_task(cancel.wait())
        try:
            done, _ = await asyncio.wait(
                {req_task, cancel_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if cancel_task in done:
                req_task.cancel()
                await asyncio.gather(req_task, return_exceptions=True)
                cancel.raise_if_cancelled()
            return req_task.result()
        finally:
            if not cancel_task.done():
                cancel_task.cancel()
                await asyncio.gather(cancel_task, return_exceptions=True)

    async def _sleep_backoff(
        self,
        attempt: int,
        *,
        retry_after: float | None = None,
        cancel: CancellationToken | None = None,
    ) -> None:
        if retry_after is not None and retry_after > 0:
            delay = min(retry_after, self._backoff_max_sec)
        else:
            exp = min(self._backoff_base_sec * (2**attempt), self._backoff_max_sec)
            jitter = random.uniform(0.0, exp * 0.25)
            delay = exp + jitter

        if cancel is None:
            await asyncio.sleep(delay)
            return

        cancel.raise_if_cancelled()
        try:
            await asyncio.wait_for(cancel.wait(), timeout=delay)
            cancel.raise_if_cancelled()
        except TimeoutError:
            return

    @staticmethod
    def _parse_retry_after(response: httpx.Response) -> float | None:
        raw = response.headers.get("retry-after")
        if not raw:
            return None
        try:
            return float(raw)
        except ValueError:
            return None
