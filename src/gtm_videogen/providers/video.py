"""Multi-provider async video generation client (fal, openrouter, higgsfield).

Fixes & Speedups:
- BUG-05: Checks CancellationToken before submit, during every poll, and before download;
  releases CostMeter reservations on cancellation or error.
- BUG-07: Honors requested aspect ratio ('16:9', '9:16', '1:1', '4:5') first in every
  provider payload (never hardcodes '9:16').
- BUG-12: Reserves before submit, settles with min_seconds billing floor on success,
  and releases on error/cancellation via CostMeter.
- BUG-14: Uses ResilientHttpClient (timeouts, raise_for_status(), 429/5xx retries).
- P4: Content-addressed clipCache hit returns immediately with $0 cost and zero HTTP calls.
- P6: Progressive polling backoff (2.0 -> 3.0 -> 5.0s).
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, Literal

from gtm_videogen.core.cache_store import ContentCache
from gtm_videogen.core.config import Settings, get_settings
from gtm_videogen.core.cost_meter import CostMeter, current_cost_meter
from gtm_videogen.core.http_client import ResilientHttpClient
from gtm_videogen.core.job_registry import CancellationToken
from gtm_videogen.core.media_store import MediaStore
from gtm_videogen.models.schemas import RenderedClip, Shot

AspectRatio = Literal["16:9", "9:16", "1:1", "4:5"]
VideoBackend = Literal["fal", "openrouter", "higgsfield"]

VIDEO_POLL_SCHEDULE: tuple[float, ...] = (2.0, 3.0, 5.0)

ASPECT_DIMENSIONS: dict[AspectRatio, tuple[int, int]] = {
    "16:9": (1280, 720),
    "9:16": (720, 1280),
    "1:1": (720, 720),
    "4:5": (720, 900),
}

_TERMINAL_SUCCESS = frozenset({"completed", "succeeded", "success", "done", "ok"})
_TERMINAL_FAILURE = frozenset({"failed", "error", "cancelled", "canceled", "rejected"})


class VideoProviderError(RuntimeError):
    """Raised when a video provider returns an error or invalid response."""


def select_supported_aspect(
    requested: AspectRatio,
    supported: Sequence[str] = ("16:9", "9:16", "1:1", "4:5"),
) -> str:
    """Prioritize the requested aspect ratio first (fixes BUG-07)."""
    if requested in supported:
        return requested
    # Fallback order preserves orientation rather than hardcoding 9:16
    fallback_map: dict[str, tuple[str, ...]] = {
        "16:9": ("16:9", "1:1", "4:5", "9:16"),
        "9:16": ("9:16", "4:5", "1:1", "16:9"),
        "4:5": ("4:5", "9:16", "1:1", "16:9"),
        "1:1": ("1:1", "4:5", "16:9", "9:16"),
    }
    for candidate in fallback_map.get(requested, ("16:9", "1:1", "9:16")):
        if candidate in supported:
            return candidate
    return supported[0] if supported else requested


def _is_valid_image_ref(value: str | None) -> bool:
    return bool(value and value.startswith(("http://", "https://", "data:image/")))


def build_fal_video_payload(
    shot: Shot,
    aspect: AspectRatio,
    fps: int,
    model: str = "",
    *,
    raw_prompt: str | None = None,
    generate_audio: bool = False,
) -> dict[str, Any]:
    """Build fal.ai video queue submission payload honoring requested aspect ratio."""
    from gtm_videogen.models.prompt_compiler import compile_video_prompt

    compiled_prompt = (raw_prompt or "").strip() or compile_video_prompt(shot)
    low_model = model.strip().lower()
    if "kling-video/v3" in low_model or "kling-video/o3" in low_model:
        dur_str = str(max(3, min(15, int(round(float(shot.duration_sec))))))
        payload = {
            "prompt": compiled_prompt[:2450],
            "duration": dur_str,
            "generate_audio": bool(generate_audio),
        }
        if shot.negative_prompt:
            payload["negative_prompt"] = shot.negative_prompt[:500]
        has_start = bool(
            shot.engine == "image_to_video" and _is_valid_image_ref(shot.reference_image)
        )
        if has_start:
            payload["start_image_url"] = shot.reference_image
            if _is_valid_image_ref(shot.end_reference_image):
                payload["end_image_url"] = shot.end_reference_image
        else:
            payload["aspect_ratio"] = select_supported_aspect(aspect, ("16:9", "9:16", "1:1"))
        return payload

    if "kling/v4" in low_model:
        dur_int = max(3, min(20, int(round(float(shot.duration_sec)))))
        payload = {
            "prompt": compiled_prompt[:2450],
            "duration": dur_int,
            "resolution": "720p",
        }
        has_start = bool(
            shot.engine == "image_to_video" and _is_valid_image_ref(shot.reference_image)
        )
        if has_start:
            payload["start_image_url"] = shot.reference_image
            # Full Kling v4 Pro I2V (non-flash) supports Start -> End keyframe bridging!
            if "flash" not in low_model and _is_valid_image_ref(shot.end_reference_image):
                payload["end_image_url"] = shot.end_reference_image
        else:
            payload["aspect_ratio"] = select_supported_aspect(aspect)
        return payload

    if "veo3" in low_model:
        dur_val = float(shot.duration_sec)
        veo_dur = "4s" if dur_val <= 5.0 else ("6s" if dur_val <= 7.0 else "8s")
        payload = {
            "prompt": compiled_prompt,
            "aspect_ratio": select_supported_aspect(aspect),
            "duration": veo_dur,
            "resolution": "720p",
            "generate_audio": bool(generate_audio),
            "auto_fix": True,
        }
        if shot.negative_prompt:
            payload["negative_prompt"] = shot.negative_prompt
        if shot.engine == "image_to_video" and _is_valid_image_ref(shot.reference_image):
            payload["image_url"] = shot.reference_image
        return payload

    payload = {
        "prompt": compiled_prompt,
        "aspect_ratio": select_supported_aspect(aspect),
        "duration": str(int(round(max(1.0, shot.duration_sec)))),
        "fps": fps,
    }
    if shot.negative_prompt:
        payload["negative_prompt"] = shot.negative_prompt
    if shot.reference_image:
        payload["image_url"] = shot.reference_image
    return payload


def build_openrouter_video_payload(
    shot: Shot,
    aspect: AspectRatio,
    fps: int,
    model: str,
) -> dict[str, Any]:
    """Build OpenRouter /api/v1/videos payload honoring requested aspect ratio (fixes BUG-07)."""
    from gtm_videogen.models.prompt_compiler import compile_video_prompt

    chosen_aspect = select_supported_aspect(aspect)
    payload: dict[str, Any] = {
        "model": model,
        "prompt": compile_video_prompt(shot),
        "aspect_ratio": chosen_aspect,
        "duration": round(float(shot.duration_sec), 2),
        "fps": fps,
    }
    if shot.negative_prompt:
        payload["negative_prompt"] = shot.negative_prompt
    if shot.reference_image:
        payload["image_url"] = shot.reference_image
    return payload


def build_higgsfield_video_payload(
    shot: Shot,
    aspect: AspectRatio,
    fps: int,
    model: str,
) -> dict[str, Any]:
    """Build Higgsfield REST payload honoring requested aspect ratio (fixes BUG-07)."""
    from gtm_videogen.models.prompt_compiler import compile_video_prompt

    chosen_aspect = select_supported_aspect(aspect)
    width, height = ASPECT_DIMENSIONS[aspect]
    payload: dict[str, Any] = {
        "model": model,
        "prompt": compile_video_prompt(shot),
        "aspect_ratio": chosen_aspect,
        "width": width,
        "height": height,
        "duration": round(float(shot.duration_sec), 2),
        "fps": fps,
    }
    if shot.negative_prompt:
        payload["negative_prompt"] = shot.negative_prompt
    if shot.reference_image:
        payload["image_url"] = shot.reference_image
    return payload


def _extract_video_url(data: Any) -> str | None:
    if not isinstance(data, dict):
        return None
    video = data.get("video")
    if isinstance(video, dict) and isinstance(video.get("url"), str) and video["url"]:
        return video["url"]
    if isinstance(video, list) and video:
        first = video[0]
        if isinstance(first, dict) and isinstance(first.get("url"), str) and first["url"]:
            return first["url"]
    for key in ("video_url", "url"):
        val = data.get(key)
        if isinstance(val, str) and val.startswith(("http://", "https://")):
            return val
    output = data.get("output")
    if isinstance(output, dict):
        nested = _extract_video_url(output)
        if nested:
            return nested
    result = data.get("result")
    if isinstance(result, dict):
        nested = _extract_video_url(result)
        if nested:
            return nested
    items = data.get("data")
    if isinstance(items, list) and items:
        first = items[0]
        if isinstance(first, dict):
            nested = _extract_video_url(first)
            if nested:
                return nested
    return None


async def _cancellable_sleep(
    delay: float,
    *,
    cancel: CancellationToken | None,
    sleep_fn: Callable[[float], Awaitable[None]] | None = None,
) -> None:
    if cancel is not None:
        cancel.raise_if_cancelled()
    if sleep_fn is not None:
        await sleep_fn(delay)
        if cancel is not None:
            cancel.raise_if_cancelled()
        return
    if cancel is None:
        await asyncio.sleep(delay)
        return
    try:
        await asyncio.wait_for(cancel.wait(), timeout=delay)
        cancel.raise_if_cancelled()
    except TimeoutError:
        return


class VideoProvider:
    """Async multi-provider video generator with caching, cost metering, and progressive polling."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        http_client: ResilientHttpClient | None = None,
        media_store: MediaStore | None = None,
        clip_cache: ContentCache | None = None,
        cost_meter: CostMeter | None = None,
        default_provider: VideoBackend = "fal",
        usd_per_sec: float = 0.07,
        min_seconds: float = 5.0,
        poll_schedule: Sequence[float] = VIDEO_POLL_SCHEDULE,
        max_poll_attempts: int = 120,
        sleep_fn: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._http_client = http_client
        self._media_store = media_store or MediaStore(self.settings.data_dir / "media")
        self._clip_cache = clip_cache
        self._cost_meter = cost_meter
        self._default_provider: VideoBackend = default_provider
        self._usd_per_sec = float(usd_per_sec)
        self._min_seconds = float(min_seconds)
        self._poll_schedule = tuple(float(s) for s in poll_schedule) or VIDEO_POLL_SCHEDULE
        self._max_poll_attempts = max(1, max_poll_attempts)
        self._sleep_fn = sleep_fn

    def _compute_billed_usd(self, duration_sec: float) -> float:
        billed_sec = max(float(duration_sec), self._min_seconds)
        return round(billed_sec * self._usd_per_sec, 6)

    def _cache_key(
        self,
        *,
        provider: str,
        model: str,
        shot: Shot,
        aspect: AspectRatio,
        fps: int,
        raw_prompt: str | None = None,
        generate_audio: bool = False,
    ) -> str:
        assert self._clip_cache is not None
        return self._clip_cache.make_key(
            {
                "provider": provider,
                "model": model,
                "prompt": (raw_prompt or "").strip() or shot.visual_prompt,
                "negative_prompt": shot.negative_prompt,
                "camera_move": shot.camera_move,
                "camera_angle": shot.camera_angle,
                "aspect": aspect,
                "fps": fps,
                "duration_sec": round(float(shot.duration_sec), 3),
                "reference_image": (shot.reference_image or "")[:128],
                "end_reference_image": (shot.end_reference_image or "")[:128],
                "generate_audio": bool(generate_audio),
            }
        )

    async def generate_clip(
        self,
        shot: Shot,
        aspect: AspectRatio,
        fps: int,
        cancel: CancellationToken | None = None,
        provider: str | None = None,
        model: str | None = None,
        *,
        raw_prompt: str | None = None,
        generate_audio: bool = False,
    ) -> RenderedClip:
        """Generate (or retrieve cached) video clip for a shot."""
        if cancel is not None:
            cancel.raise_if_cancelled()

        chosen_provider = (provider or self._default_provider).strip().lower()
        if chosen_provider not in ("fal", "openrouter", "higgsfield"):
            raise VideoProviderError(f"Unsupported video provider: {chosen_provider!r}")

        chosen_model = (model or self.settings.video_model).strip()
        width, height = ASPECT_DIMENSIONS[aspect]

        # P4: Check content-addressed clipCache before reserving budget or calling network.
        cache_key: str | None = None
        if self._clip_cache is not None:
            cache_key = self._cache_key(
                provider=chosen_provider,
                model=chosen_model,
                shot=shot,
                aspect=aspect,
                fps=fps,
                raw_prompt=raw_prompt,
                generate_audio=generate_audio,
            )
            hit = self._clip_cache.get(cache_key)
            if hit is not None and hit.artifact_path is not None:
                return RenderedClip(
                    shot_id=shot.id,
                    path=str(hit.artifact_path),
                    duration_sec=float(hit.metadata.get("duration_sec", shot.duration_sec)),
                    fps=int(hit.metadata.get("fps", fps)),
                    width=int(hit.metadata.get("width", width)),
                    height=int(hit.metadata.get("height", height)),
                    cache_hit=True,
                    usd=0.0,
                )

        # BUG-12: Reserve estimated spend (with min_seconds floor) before submitting.
        meter = self._cost_meter or current_cost_meter()
        estimated_usd = self._compute_billed_usd(shot.duration_sec)
        reservation_id: str | None = None
        if meter is not None:
            reservation_id = meter.reserve(
                f"video:{shot.id}",
                estimated_usd,
                stage="render_video",
                provider=chosen_provider,
                model=chosen_model,
            )

        client = self._http_client
        owns_client = client is None
        if client is None:
            client = ResilientHttpClient(
                connect_timeout=self.settings.http_connect_timeout_sec,
                read_timeout=self.settings.http_read_timeout_sec,
                write_timeout=self.settings.http_write_timeout_sec,
                pool_timeout=self.settings.http_pool_timeout_sec,
                max_retries=self.settings.http_max_retries,
            )

        try:
            if cancel is not None:
                cancel.raise_if_cancelled()

            if chosen_provider == "fal":
                video_url, reported_usd = await self._run_fal(
                    client,
                    shot,
                    aspect,
                    fps,
                    chosen_model,
                    cancel,
                    raw_prompt=raw_prompt,
                    generate_audio=generate_audio,
                )
            elif chosen_provider == "openrouter":
                video_url, reported_usd = await self._run_openrouter(
                    client, shot, aspect, fps, chosen_model, cancel
                )
            else:
                video_url, reported_usd = await self._run_higgsfield(
                    client, shot, aspect, fps, chosen_model, cancel
                )

            if cancel is not None:
                cancel.raise_if_cancelled()

            video_bytes = await client.get_bytes(video_url, cancel=cancel)
            if not video_bytes:
                raise VideoProviderError(
                    f"Empty video payload downloaded from {chosen_provider}: {video_url}"
                )

            _, saved_path = self._media_store.save_bytes(
                video_bytes,
                ext=".mp4",
                subdir="clips",
            )

            actual_usd = reported_usd if reported_usd is not None else estimated_usd
            if meter is not None and reservation_id is not None:
                meter.settle(reservation_id, actual_usd)
                reservation_id = None

            if self._clip_cache is not None and cache_key is not None:
                self._clip_cache.put(
                    cache_key,
                    artifact_path=saved_path,
                    metadata={
                        "shot_id": shot.id,
                        "provider": chosen_provider,
                        "model": chosen_model,
                        "aspect": aspect,
                        "fps": fps,
                        "width": width,
                        "height": height,
                        "duration_sec": float(shot.duration_sec),
                    },
                )

            return RenderedClip(
                shot_id=shot.id,
                path=str(saved_path),
                duration_sec=float(shot.duration_sec),
                fps=fps,
                width=width,
                height=height,
                cache_hit=False,
                usd=actual_usd,
            )
        except BaseException:
            if meter is not None and reservation_id is not None:
                meter.release(reservation_id)
            raise
        finally:
            if owns_client:
                await client.aclose()

    async def _poll_until_complete(
        self,
        *,
        client: ResilientHttpClient,
        status_url: str,
        headers: dict[str, str],
        cancel: CancellationToken | None,
        provider_name: str,
    ) -> dict[str, Any]:
        """Poll a provider status URL with P6 progressive backoff (2.0 -> 3.0 -> 5.0s)."""
        for attempt in range(self._max_poll_attempts):
            if cancel is not None:
                cancel.raise_if_cancelled()

            delay = self._poll_schedule[min(attempt, len(self._poll_schedule) - 1)]
            await _cancellable_sleep(delay, cancel=cancel, sleep_fn=self._sleep_fn)

            status_data = await client.get_json(status_url, headers=headers, cancel=cancel)
            if not isinstance(status_data, dict):
                raise VideoProviderError(
                    f"{provider_name} returned non-object status payload: {status_data!r}"
                )

            raw_status = str(status_data.get("status", "")).strip().lower()
            if raw_status in _TERMINAL_SUCCESS:
                return status_data
            if raw_status in _TERMINAL_FAILURE:
                err_msg = status_data.get("error") or status_data.get("message") or raw_status
                raise VideoProviderError(f"{provider_name} video generation failed: {err_msg}")

        raise VideoProviderError(
            f"{provider_name} video generation timed out after {self._max_poll_attempts} polls"
        )

    async def _run_fal(
        self,
        client: ResilientHttpClient,
        shot: Shot,
        aspect: AspectRatio,
        fps: int,
        model: str,
        cancel: CancellationToken | None,
        *,
        raw_prompt: str | None = None,
        generate_audio: bool = False,
    ) -> tuple[str, float | None]:
        headers = {"Content-Type": "application/json"}
        if self.settings.fal_key:
            headers["Authorization"] = f"Key {self.settings.fal_key}"

        target_model = model.lstrip("/")
        # fal gates non-flash kling/v4 (playground-only -> HTTP 403 on queue.fal.run);
        # route cleanly to the live public Kling 3.0 Pro 1080p API (Start->End + Native Audio).
        if target_model == "kling/v4/text-to-video":
            target_model = "fal-ai/kling-video/v3/pro/text-to-video"
        elif target_model == "kling/v4/image-to-video":
            target_model = "fal-ai/kling-video/v3/pro/image-to-video"

        has_ref = bool(
            shot.engine == "image_to_video" and _is_valid_image_ref(shot.reference_image)
        )
        if has_ref and target_model.endswith("/text-to-video"):
            target_model = target_model[: -len("/text-to-video")] + "/image-to-video"
        elif has_ref and target_model in ("fal-ai/veo3", "fal-ai/veo3/fast"):
            target_model = "fal-ai/veo3/image-to-video"

        submit_url = f"https://queue.fal.run/{target_model}"
        payload = build_fal_video_payload(
            shot,
            aspect,
            fps,
            model=target_model,
            raw_prompt=raw_prompt,
            generate_audio=generate_audio,
        )
        submitted = await client.post_json(submit_url, json=payload, headers=headers, cancel=cancel)
        if not isinstance(submitted, dict):
            raise VideoProviderError(f"fal returned non-object submit payload: {submitted!r}")

        direct_url = _extract_video_url(submitted)
        if direct_url:
            return direct_url, self._parse_reported_cost(submitted)

        request_id = submitted.get("request_id")
        status_url = submitted.get("status_url") or (
            f"https://queue.fal.run/{target_model}/requests/{request_id}/status"
            if request_id
            else None
        )
        response_url = submitted.get("response_url") or (
            f"https://queue.fal.run/{target_model}/requests/{request_id}"
            if request_id
            else None
        )
        if not status_url or not response_url:
            raise VideoProviderError(
                f"fal submit response missing status/response URL: {submitted}"
            )

        status_data = await self._poll_until_complete(
            client=client,
            status_url=str(status_url),
            headers=headers,
            cancel=cancel,
            provider_name="fal",
        )
        result_url = str(status_data.get("response_url") or response_url)
        result_data = await client.get_json(result_url, headers=headers, cancel=cancel)
        video_url = _extract_video_url(result_data)
        if not video_url:
            raise VideoProviderError(f"fal result missing video URL: {result_data!r}")
        return video_url, self._parse_reported_cost(result_data)

    async def _run_openrouter(
        self,
        client: ResilientHttpClient,
        shot: Shot,
        aspect: AspectRatio,
        fps: int,
        model: str,
        cancel: CancellationToken | None,
    ) -> tuple[str, float | None]:
        headers = {"Content-Type": "application/json"}
        if self.settings.openrouter_api_key:
            headers["Authorization"] = f"Bearer {self.settings.openrouter_api_key}"

        submit_url = "https://openrouter.ai/api/v1/videos"
        payload = build_openrouter_video_payload(shot, aspect, fps, model)
        submitted = await client.post_json(submit_url, json=payload, headers=headers, cancel=cancel)
        if not isinstance(submitted, dict):
            raise VideoProviderError(
                f"openrouter returned non-object submit payload: {submitted!r}"
            )

        direct_url = _extract_video_url(submitted)
        if direct_url:
            return direct_url, self._parse_reported_cost(submitted)

        job_id = submitted.get("id") or submitted.get("job_id")
        status_url = submitted.get("status_url") or (
            f"https://openrouter.ai/api/v1/videos/{job_id}" if job_id else None
        )
        if not status_url:
            raise VideoProviderError(f"openrouter submit missing job id/status_url: {submitted}")

        completed = await self._poll_until_complete(
            client=client,
            status_url=str(status_url),
            headers=headers,
            cancel=cancel,
            provider_name="openrouter",
        )
        video_url = _extract_video_url(completed)
        if not video_url:
            raise VideoProviderError(f"openrouter completed payload missing video URL: {completed}")
        return video_url, self._parse_reported_cost(completed)

    async def _run_higgsfield(
        self,
        client: ResilientHttpClient,
        shot: Shot,
        aspect: AspectRatio,
        fps: int,
        model: str,
        cancel: CancellationToken | None,
    ) -> tuple[str, float | None]:
        headers = {"Content-Type": "application/json"}
        if self.settings.hf_api_key:
            headers["Authorization"] = f"Bearer {self.settings.hf_api_key}"
        elif self.settings.hf_api_key_id and self.settings.hf_api_key_secret:
            headers["hf-api-key"] = self.settings.hf_api_key_id
            headers["hf-api-secret"] = self.settings.hf_api_key_secret

        submit_url = "https://api.higgsfield.ai/v1/videos"
        payload = build_higgsfield_video_payload(shot, aspect, fps, model)
        submitted = await client.post_json(submit_url, json=payload, headers=headers, cancel=cancel)
        if not isinstance(submitted, dict):
            raise VideoProviderError(
                f"higgsfield returned non-object submit payload: {submitted!r}"
            )

        direct_url = _extract_video_url(submitted)
        if direct_url:
            return direct_url, self._parse_reported_cost(submitted)

        job_id = submitted.get("id") or submitted.get("job_id") or submitted.get("request_id")
        status_url = submitted.get("status_url") or (
            f"https://api.higgsfield.ai/v1/videos/{job_id}" if job_id else None
        )
        if not status_url:
            raise VideoProviderError(f"higgsfield submit missing job id/status_url: {submitted}")

        completed = await self._poll_until_complete(
            client=client,
            status_url=str(status_url),
            headers=headers,
            cancel=cancel,
            provider_name="higgsfield",
        )
        video_url = _extract_video_url(completed)
        if not video_url:
            raise VideoProviderError(f"higgsfield completed payload missing video URL: {completed}")
        return video_url, self._parse_reported_cost(completed)

    @staticmethod
    def _parse_reported_cost(data: Any) -> float | None:
        if not isinstance(data, dict):
            return None
        for key in ("cost_usd", "usd", "total_cost"):
            val = data.get(key)
            if (
                isinstance(val, (int, float))
                and not isinstance(val, bool)
                and math.isfinite(val)
                and val >= 0
            ):
                return float(val)
        return None
