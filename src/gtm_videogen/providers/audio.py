"""Multi-model async audio provider (TTS, SFX fallback chain, Music) with typed payloads.

Fixes & Speedups:
- BUG-04: Typed per-model Pydantic v2 payload builders for TTS (ElevenLabs v3, Chatterbox,
  Minimax), SFX (Mirelo -> ElevenLabs -> Sonilo explicit fallback chain; NEVER deprecated
  fal-ai/mmaudio-v2), and Music (Stable Audio 2.5, ElevenLabs Music v2.5, ACE-Step).
- BUG-05: Checks CancellationToken before submit, during every poll, and before download.
- BUG-12: Reserves/settles/releases every billable call via CostMeter.
- BUG-14: Uses ResilientHttpClient (timeouts, raise_for_status(), 429/5xx retries).
- P4: Content-addressed audioCache hit returns immediately with $0 cost and zero HTTP calls.
- P6: Progressive polling backoff (0.5 -> 1.0 -> 1.5 -> 2.0 -> 3.0s).
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from gtm_videogen.core.cache_store import ContentCache
from gtm_videogen.core.config import Settings, get_settings
from gtm_videogen.core.cost_meter import BudgetExceeded, CostMeter, current_cost_meter
from gtm_videogen.core.http_client import ResilientHttpClient
from gtm_videogen.core.job_registry import CancellationToken
from gtm_videogen.core.media_store import MediaStore
from gtm_videogen.models.schemas import MusicCue, RenderedAudio, SfxCue, VoiceCue
from gtm_videogen.rules.speakability import spoken_form

AUDIO_POLL_SCHEDULE: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 3.0)

DEFAULT_VOICE_MODEL = "fal-ai/elevenlabs/tts/eleven-v3"
DEFAULT_MUSIC_MODEL = "fal-ai/stable-audio-25/text-to-audio"

SFX_FALLBACK_CHAIN: tuple[str, ...] = (
    "mirelo-ai/sfx2.0/text-to-audio",
    "elevenlabs/sound-effects/v3",
    "sonilo/v1.1/text-to-sound-effects",
)

_DEPRECATED_SFX_MODELS = frozenset({"fal-ai/mmaudio-v2", "mmaudio-v2"})

_TERMINAL_SUCCESS = frozenset({"completed", "succeeded", "success", "done", "ok"})
_TERMINAL_FAILURE = frozenset({"failed", "error", "cancelled", "canceled", "rejected"})

_ALL_CAPS_WORD_RE = re.compile(r"\b([A-Z]{4,})\b")
_KEEP_ACRONYMS = frozenset({"USDA", "NASA", "NCAA", "ESPN", "ASAP", "FAQS", "HTTP", "HTTPS"})


class AudioProviderError(RuntimeError):
    """Raised when an audio provider fails or receives an unsupported model."""


# ---------------------------------------------------------------------------
# Typed Per-Model Pydantic v2 Payload Builders (BUG-04)
# ---------------------------------------------------------------------------


class ElevenLabsTTSPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    voice: str = Field(min_length=1)
    stability: float = Field(0.5, ge=0.0, le=1.0)


class ChatterboxTTSPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    exaggeration: float = Field(0.5, ge=0.0, le=1.0)
    cfg: float = Field(0.5, ge=0.0, le=1.0)
    audio_url: str | None = None


class MinimaxTTSPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    voice_id: str = Field(min_length=1)
    output_format: Literal["url"] = "url"


class MireloSFXPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text_prompt: str = Field(min_length=1)
    duration: float = Field(ge=1.0, le=10.0)


class ElevenLabsSFXPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    duration_seconds: float = Field(ge=0.5, le=22.0)


class SoniloSFXPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1)
    duration: float = Field(ge=1.0, le=15.0)


class StableAudioMusicPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1)
    seconds_total: int = Field(ge=3, le=190)


class ElevenLabsMusicPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1)
    music_length_ms: int = Field(ge=3000, le=300_000)
    force_instrumental: bool = True


class AceStepMusicPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tags: str = Field(min_length=1)
    duration: float = Field(ge=5.0, le=240.0)
    lyrics: str = "[instrumental]"


def speakable(text: str, pronunciations: dict[str, str] | None = None) -> str:
    """Normalize text for TTS: apply custom brand pronunciations, tame ALL-CAPS, expand symbols."""
    if not text or not text.strip():
        return ""

    out = text.strip()
    if pronunciations:
        for term in sorted(pronunciations, key=len, reverse=True):
            replacement = pronunciations[term]
            if not term.strip() or not replacement.strip():
                continue
            pattern = re.compile(rf"\b{re.escape(term.strip())}\b", re.IGNORECASE)
            repl = replacement.strip()
            out = pattern.sub(lambda _, r=repl: r, out)

    def _tame_caps(match: re.Match[str]) -> str:
        word = match.group(1)
        if word in _KEEP_ACRONYMS:
            return word
        return word.capitalize()

    out = _ALL_CAPS_WORD_RE.sub(_tame_caps, out)
    expanded = spoken_form(out)
    # Guard against M1 spoken_form() currency regex eating trailing space before non-scale words
    expanded = re.sub(r"\bdollars(?=[a-zA-Z])", "dollars ", expanded)
    return re.sub(r"\s+", " ", expanded).strip()


def build_tts_payload(
    model: str,
    text: str,
    voice_id: str,
    *,
    pronunciations: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build a strictly typed TTS payload for the requested fal.ai TTS model (BUG-04)."""
    low = model.strip().lower()
    normalized_text = speakable(text, pronunciations) or text.strip()

    if "chatterbox" in low:
        payload = ChatterboxTTSPayload(text=normalized_text)
        return payload.model_dump(exclude_none=True)
    if "minimax" in low:
        payload = MinimaxTTSPayload(
            text=normalized_text,
            voice_id=voice_id or "Wise_Woman",
            output_format="url",
        )
        return payload.model_dump(exclude_none=True)

    # Default: fal-ai/elevenlabs/tts/eleven-v3
    payload = ElevenLabsTTSPayload(
        text=normalized_text,
        voice=voice_id or "Aria",
        stability=0.5,
    )
    return payload.model_dump(exclude_none=True)


def build_sfx_payload(model: str, prompt: str, duration_sec: float) -> dict[str, Any]:
    """Build a strictly typed SFX payload for Mirelo, ElevenLabs SFX v3, or Sonilo (BUG-04)."""
    low = model.strip().lower()
    if low in _DEPRECATED_SFX_MODELS or "mmaudio-v2" in low:
        raise AudioProviderError(
            f"Deprecated video-to-audio model {model!r} cannot be used for text-to-audio SFX"
        )

    cleaned_prompt = prompt.strip()
    if "mirelo" in low:
        clamped = min(10.0, max(1.0, float(duration_sec)))
        return MireloSFXPayload(text_prompt=cleaned_prompt, duration=clamped).model_dump()
    if "elevenlabs" in low:
        clamped = min(22.0, max(0.5, float(duration_sec)))
        return ElevenLabsSFXPayload(text=cleaned_prompt, duration_seconds=clamped).model_dump()
    if "sonilo" in low:
        clamped = min(15.0, max(1.0, float(duration_sec)))
        return SoniloSFXPayload(prompt=cleaned_prompt, duration=clamped).model_dump()

    clamped = min(10.0, max(1.0, float(duration_sec)))
    return MireloSFXPayload(text_prompt=cleaned_prompt, duration=clamped).model_dump()


def build_music_payload(model: str, cue: MusicCue) -> dict[str, Any]:
    """Build a strictly typed Music payload for Stable Audio 2.5, ElevenLabs v2.5, or ACE-Step."""
    low = model.strip().lower()
    cleaned_prompt = cue.prompt.strip()
    duration = float(cue.duration_sec)

    if "elevenlabs" in low and "music" in low:
        ms = int(round(min(300_000.0, max(3000.0, duration * 1000.0))))
        return ElevenLabsMusicPayload(
            prompt=cleaned_prompt,
            music_length_ms=ms,
            force_instrumental=True,
        ).model_dump()

    if "ace-step" in low:
        tags_str = ", ".join(t.strip() for t in cue.tags if t.strip()) or cleaned_prompt
        clamped = min(240.0, max(5.0, duration))
        return AceStepMusicPayload(
            tags=tags_str,
            duration=clamped,
            lyrics="[instrumental]",
        ).model_dump()

    # Default: fal-ai/stable-audio-25/text-to-audio
    secs = int(round(min(190.0, max(3.0, duration))))
    return StableAudioMusicPayload(prompt=cleaned_prompt, seconds_total=secs).model_dump()


def extract_audio_url(result: Any) -> str | None:
    """Extract audio URL supporting audio.url, audio[0].url, audio_file.url, and top-level url."""
    if not isinstance(result, dict):
        return None

    audio = result.get("audio")
    if isinstance(audio, dict) and isinstance(audio.get("url"), str) and audio["url"]:
        return audio["url"]
    if isinstance(audio, list) and audio:
        first = audio[0]
        if isinstance(first, dict) and isinstance(first.get("url"), str) and first["url"]:
            return first["url"]
        if isinstance(first, str) and first.startswith(("http://", "https://")):
            return first

    audio_file = result.get("audio_file")
    if (
        isinstance(audio_file, dict)
        and isinstance(audio_file.get("url"), str)
        and audio_file["url"]
    ):
        return audio_file["url"]

    for key in ("audio_url", "url"):
        val = result.get(key)
        if isinstance(val, str) and val.startswith(("http://", "https://")):
            return val

    for nested_key in ("output", "result", "data"):
        nested = result.get(nested_key)
        if isinstance(nested, dict):
            found = extract_audio_url(nested)
            if found:
                return found

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


class AudioProvider:
    """Async TTS, SFX (with fallback chain), and Music provider."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        http_client: ResilientHttpClient | None = None,
        media_store: MediaStore | None = None,
        audio_cache: ContentCache | None = None,
        cost_meter: CostMeter | None = None,
        sfx_fallback_chain: Sequence[str] = SFX_FALLBACK_CHAIN,
        poll_schedule: Sequence[float] = AUDIO_POLL_SCHEDULE,
        max_poll_attempts: int = 120,
        sleep_fn: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._http_client = http_client
        self._media_store = media_store or MediaStore(self.settings.data_dir / "media")
        self._audio_cache = audio_cache
        self._cost_meter = cost_meter
        self._sfx_fallback_chain = tuple(sfx_fallback_chain) or SFX_FALLBACK_CHAIN
        self._poll_schedule = tuple(float(s) for s in poll_schedule) or AUDIO_POLL_SCHEDULE
        self._max_poll_attempts = max(1, max_poll_attempts)
        self._sleep_fn = sleep_fn

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.settings.fal_key:
            headers["Authorization"] = f"Key {self.settings.fal_key}"
        return headers

    async def generate_voice(
        self,
        cue: VoiceCue,
        voice_id: str,
        cancel: CancellationToken | None = None,
        brand_pronunciations: dict[str, str] | None = None,
        model: str | None = None,
    ) -> RenderedAudio:
        """Synthesize voiceover for a VoiceCue with speakable() normalization and audioCache."""
        if cancel is not None:
            cancel.raise_if_cancelled()

        chosen_model = (model or self.settings.voice_model or DEFAULT_VOICE_MODEL).strip()
        merged_pronunciations = dict(brand_pronunciations or {})
        merged_pronunciations.update(cue.pronunciation or {})
        payload = build_tts_payload(
            chosen_model,
            cue.text,
            voice_id,
            pronunciations=merged_pronunciations,
        )
        est_usd = round(max(0.01, (len(cue.text.strip()) / 1000.0) * 0.10), 6)

        return await self._generate_single(
            kind="vo",
            cue_id=cue.shot_id,
            model=chosen_model,
            payload=payload,
            duration_sec=float(cue.duration_sec),
            est_usd=est_usd,
            cache_payload={
                "kind": "vo",
                "model": chosen_model,
                "voice_id": voice_id,
                "payload": payload,
            },
            ext=".mp3",
            cancel=cancel,
        )

    async def generate_sfx(
        self,
        cue: SfxCue,
        cancel: CancellationToken | None = None,
        model: str | None = None,
    ) -> RenderedAudio:
        """Generate SFX using the explicit Mirelo -> ElevenLabs -> Sonilo fallback chain."""
        if cancel is not None:
            cancel.raise_if_cancelled()

        chain: list[str] = []
        primary = (model or self.settings.sfx_model or SFX_FALLBACK_CHAIN[0]).strip()
        if primary:
            chain.append(primary)
        for fallback in self._sfx_fallback_chain:
            if fallback not in chain:
                chain.append(fallback)

        errors: list[str] = []
        for candidate_model in chain:
            if cancel is not None:
                cancel.raise_if_cancelled()

            payload = build_sfx_payload(candidate_model, cue.prompt, cue.duration_sec)
            est_usd = self._estimate_sfx_cost(candidate_model, cue.duration_sec)
            try:
                return await self._generate_single(
                    kind="sfx",
                    cue_id=cue.shot_id or f"sfx_{cue.at_sec:.2f}",
                    model=candidate_model,
                    payload=payload,
                    duration_sec=float(cue.duration_sec),
                    est_usd=est_usd,
                    cache_payload={
                        "kind": "sfx",
                        "model": candidate_model,
                        "payload": payload,
                    },
                    ext=".wav",
                    cancel=cancel,
                )
            except (asyncio.CancelledError, BudgetExceeded):
                raise
            except Exception as exc:
                errors.append(f"{candidate_model}: {exc}")
                continue

        raise AudioProviderError(
            f"All SFX models failed ({len(chain)} tried): {'; '.join(errors)}"
        )

    async def generate_music(
        self,
        cue: MusicCue,
        cancel: CancellationToken | None = None,
        model: str | None = None,
    ) -> RenderedAudio:
        """Generate background music bed using per-model typed payload schema."""
        if cancel is not None:
            cancel.raise_if_cancelled()

        chosen_model = (model or self.settings.music_model or DEFAULT_MUSIC_MODEL).strip()
        payload = build_music_payload(chosen_model, cue)
        est_usd = 0.20

        return await self._generate_single(
            kind="music",
            cue_id="music_bed",
            model=chosen_model,
            payload=payload,
            duration_sec=float(cue.duration_sec),
            est_usd=est_usd,
            cache_payload={
                "kind": "music",
                "model": chosen_model,
                "payload": payload,
            },
            ext=".wav",
            cancel=cancel,
        )

    @staticmethod
    def _estimate_sfx_cost(model: str, duration_sec: float) -> float:
        low = model.lower()
        if "elevenlabs" in low:
            return round(max(0.005, float(duration_sec) * 0.002), 6)
        if "sonilo" in low:
            return round(max(0.005, float(duration_sec) * 0.0018), 6)
        return 0.02

    async def _generate_single(
        self,
        *,
        kind: Literal["vo", "sfx", "music"],
        cue_id: str,
        model: str,
        payload: dict[str, Any],
        duration_sec: float,
        est_usd: float,
        cache_payload: dict[str, Any],
        ext: str,
        cancel: CancellationToken | None,
    ) -> RenderedAudio:
        if cancel is not None:
            cancel.raise_if_cancelled()

        # P4: Check content-addressed audioCache before reserving budget or calling network.
        cache_key: str | None = None
        if self._audio_cache is not None:
            cache_key = self._audio_cache.make_key(cache_payload)
            hit = self._audio_cache.get(cache_key)
            if hit is not None and hit.artifact_path is not None:
                return RenderedAudio(
                    kind=kind,
                    cue_id=cue_id,
                    path=str(hit.artifact_path),
                    duration_sec=float(hit.metadata.get("duration_sec", duration_sec)),
                    cache_hit=True,
                    usd=0.0,
                )

        # BUG-12: Reserve spend before submit.
        meter = self._cost_meter or current_cost_meter()
        reservation_id: str | None = None
        if meter is not None:
            reservation_id = meter.reserve(
                f"audio:{kind}:{cue_id}",
                est_usd,
                stage=f"render_{kind}",
                provider="fal",
                model=model,
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
            headers = self._headers()
            submit_url = f"https://queue.fal.run/{model.lstrip('/')}"
            submitted = await client.post_json(
                submit_url,
                json=payload,
                headers=headers,
                cancel=cancel,
            )
            if not isinstance(submitted, dict):
                raise AudioProviderError(f"{model} returned non-object payload: {submitted!r}")

            audio_url = extract_audio_url(submitted)
            if audio_url is None:
                request_id = submitted.get("request_id")
                status_url = submitted.get("status_url") or (
                    f"https://queue.fal.run/{model.lstrip('/')}/requests/{request_id}/status"
                    if request_id
                    else None
                )
                response_url = submitted.get("response_url") or (
                    f"https://queue.fal.run/{model.lstrip('/')}/requests/{request_id}"
                    if request_id
                    else None
                )
                if not status_url or not response_url:
                    raise AudioProviderError(
                        f"{model} submit response missing audio URL and status_url: {submitted}"
                    )

                status_data = await self._poll_until_complete(
                    client=client,
                    status_url=str(status_url),
                    headers=headers,
                    cancel=cancel,
                    model_name=model,
                )
                final_url = str(status_data.get("response_url") or response_url)
                result_data = await client.get_json(final_url, headers=headers, cancel=cancel)
                audio_url = extract_audio_url(result_data)
                if not audio_url:
                    raise AudioProviderError(
                        f"{model} completed result missing audio URL: {result_data!r}"
                    )

            if cancel is not None:
                cancel.raise_if_cancelled()

            audio_bytes = await client.get_bytes(audio_url, cancel=cancel)
            if not audio_bytes:
                raise AudioProviderError(f"Empty audio downloaded from {model}: {audio_url}")

            _, saved_path = self._media_store.save_bytes(
                audio_bytes,
                ext=ext,
                subdir=f"audio_{kind}",
            )

            if meter is not None and reservation_id is not None:
                meter.settle(reservation_id, est_usd)
                reservation_id = None

            if self._audio_cache is not None and cache_key is not None:
                self._audio_cache.put(
                    cache_key,
                    artifact_path=saved_path,
                    metadata={
                        "kind": kind,
                        "cue_id": cue_id,
                        "model": model,
                        "duration_sec": duration_sec,
                    },
                )

            return RenderedAudio(
                kind=kind,
                cue_id=cue_id,
                path=str(saved_path),
                duration_sec=duration_sec,
                cache_hit=False,
                usd=est_usd,
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
        model_name: str,
    ) -> dict[str, Any]:
        """Poll fal.ai status with P6 progressive backoff (0.5 -> 1.0 -> 1.5 -> 2.0 -> 3.0s)."""
        for attempt in range(self._max_poll_attempts):
            if cancel is not None:
                cancel.raise_if_cancelled()

            delay = self._poll_schedule[min(attempt, len(self._poll_schedule) - 1)]
            await _cancellable_sleep(delay, cancel=cancel, sleep_fn=self._sleep_fn)

            status_data = await client.get_json(status_url, headers=headers, cancel=cancel)
            if not isinstance(status_data, dict):
                raise AudioProviderError(
                    f"{model_name} returned non-object status payload: {status_data!r}"
                )

            raw_status = str(status_data.get("status", "")).strip().lower()
            if raw_status in _TERMINAL_SUCCESS:
                return status_data
            if raw_status in _TERMINAL_FAILURE:
                err_msg = status_data.get("error") or status_data.get("message") or raw_status
                raise AudioProviderError(f"{model_name} audio generation failed: {err_msg}")

        raise AudioProviderError(
            f"{model_name} audio generation timed out after {self._max_poll_attempts} polls"
        )
