"""Deterministic offline mock providers for zero-API-key runs and pytest suites.

Includes:
- ScriptedChatModel(BaseChatModel): genuine LangChain chat model supporting sync/async
  invocation, .bind_tools(), and .with_structured_output() for deepagents & RubricMiddleware.
- MockBrandScraper: deterministic offline BrandScraper writing real synthetic PNG product photos.
- MockVideoProvider: generates REAL synthetic MP4 clips via FFmpeg lavfi (aspect/fps/duration).
- MockAudioProvider: generates REAL synthetic WAV audio via FFmpeg sine/anullsrc.
- MockWhisperProvider: deterministic offline Whisper transcription stub.
- MockProviderBundle: batteries-included bundle for `--mock` pipeline runs.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from PIL import Image, ImageDraw
from pydantic import BaseModel, Field, PrivateAttr

from gtm_videogen.core.cache_store import CacheStore, ContentCache, compute_cache_key
from gtm_videogen.core.config import Settings, get_settings
from gtm_videogen.core.cost_meter import CostMeter, current_cost_meter
from gtm_videogen.core.job_registry import CancellationToken
from gtm_videogen.core.media_store import MediaStore
from gtm_videogen.models.schemas import (
    MediaRef,
    MusicCue,
    RenderedAudio,
    RenderedClip,
    ResearchBundle,
    SfxCue,
    Shot,
    VoiceCue,
)
from gtm_videogen.providers.audio import AudioProvider
from gtm_videogen.providers.llm_router import LLMRouter
from gtm_videogen.providers.scraper import BrandScraper
from gtm_videogen.providers.video import AspectRatio, VideoProvider, VideoProviderError
from gtm_videogen.providers.whisper import WhisperProvider

MOCK_ASPECT_DIMENSIONS: dict[AspectRatio, tuple[int, int]] = {
    "16:9": (640, 360),
    "9:16": (360, 640),
    "1:1": (480, 480),
    "4:5": (480, 600),
}


@dataclass
class _ScriptedState:
    cursor: int = 0
    calls: list[list[BaseMessage]] = field(default_factory=list)


def _coerce_to_ai_message(raw: Any) -> AIMessage:
    if isinstance(raw, AIMessage):
        return raw
    if isinstance(raw, BaseModel):
        return AIMessage(content=raw.model_dump_json())
    if isinstance(raw, dict):
        if "tool_calls" in raw:
            return AIMessage(
                content=str(raw.get("content", "")),
                tool_calls=raw["tool_calls"],
            )
        return AIMessage(content=json.dumps(raw))
    return AIMessage(content=str(raw))


class ScriptedChatModel(BaseChatModel):
    """Genuine BaseChatModel fake supporting .bind_tools() and .with_structured_output()."""

    responses: list[Any] = Field(default_factory=list)
    responder: Callable[[list[BaseMessage]], Any] | None = None
    bound_tools: list[Any] = Field(default_factory=list)
    model_name: str = "scripted-mock-chat"

    _state: _ScriptedState = PrivateAttr(default_factory=_ScriptedState)

    @property
    def _llm_type(self) -> str:
        return "scripted-chat-model"

    @property
    def call_history(self) -> list[list[BaseMessage]]:
        return self._state.calls

    @property
    def call_count(self) -> int:
        return len(self._state.calls)

    def _next_message(self, messages: list[BaseMessage]) -> AIMessage:
        self._state.calls.append(list(messages))
        if self.responder is not None:
            raw = self.responder(messages)
            return _coerce_to_ai_message(raw)

        if not self.responses:
            return AIMessage(content="ok")

        idx = min(self._state.cursor, len(self.responses) - 1)
        self._state.cursor += 1
        item = self.responses[idx]
        if callable(item):
            item = item(messages)
        return _coerce_to_ai_message(item)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        msg = self._next_message(messages)
        return ChatResult(generations=[ChatGeneration(message=msg)])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        msg = self._next_message(messages)
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | Any],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        clone = ScriptedChatModel(
            responses=self.responses,
            responder=self.responder,
            bound_tools=list(tools),
            model_name=self.model_name,
        )
        # Share mutable cursor and call history so tool-bound invocations advance in sync.
        clone._state = self._state
        return clone

    def with_structured_output(
        self,
        schema: dict[str, Any] | type,
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, Any]:
        model_ref = self

        def _parse_message(msg: AIMessage) -> Any:
            raw_obj: Any
            if msg.tool_calls:
                raw_obj = msg.tool_calls[0].get("args", {})
            else:
                text = msg.content if isinstance(msg.content, str) else json.dumps(msg.content)
                text = text.strip()
                if text.startswith("```"):
                    lines = text.splitlines()
                    if len(lines) >= 3:
                        text = "\n".join(lines[1:-1]).strip()
                raw_obj = json.loads(text)

            if isinstance(schema, type) and issubclass(schema, BaseModel):
                parsed: Any = schema.model_validate(raw_obj)
            else:
                parsed = raw_obj

            if include_raw:
                return {"raw": msg, "parsed": parsed, "parsing_error": None}
            return parsed

        def _sync_invoke(inp: LanguageModelInput) -> Any:
            msg = model_ref.invoke(inp)
            return _parse_message(msg)

        async def _async_invoke(inp: LanguageModelInput) -> Any:
            msg = await model_ref.ainvoke(inp)
            return _parse_message(msg)

        return RunnableLambda(_sync_invoke, afunc=_async_invoke)


def _create_synthetic_png(dest_path: Path, color: tuple[int, int, int], label: str) -> Path:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (320, 320), color=color)
    draw = ImageDraw.Draw(img)
    draw.rectangle([20, 20, 300, 300], outline=(255, 255, 255), width=4)
    draw.text((40, 145), label[:24], fill=(255, 255, 255))
    img.save(dest_path, format="PNG")
    return dest_path


class MockBrandScraper(BrandScraper):
    """Deterministic offline BrandScraper returning a realistic ResearchBundle."""

    def __init__(self, *, media_dir: Path | None = None) -> None:
        super().__init__(http_client=None, resolve_dns_for_ssrf=False)
        self._media_dir = Path(media_dir).resolve() if media_dir else None

    async def scrape_brand(
        self,
        brand_url: str,
        inspiration_url: str | None = None,
        cancel: CancellationToken | None = None,
        brand_name: str | None = None,
    ) -> ResearchBundle:
        if cancel is not None:
            cancel.raise_if_cancelled()

        name = (brand_name or "Demo Brand").strip()
        colors = ["#1D3557", "#E63946", "#F1FAEE", "#457B9D"]
        photos: list[MediaRef] = []

        if self._media_dir is not None:
            p1 = _create_synthetic_png(
                self._media_dir / "mock_product_1.png",
                (29, 53, 87),
                f"{name} Hero",
            )
            p2 = _create_synthetic_png(
                self._media_dir / "mock_product_2.png",
                (69, 123, 157),
                f"{name} Detail",
            )
            photos = [
                MediaRef(
                    id="photo_1",
                    path=str(p1),
                    mime_type="image/png",
                    alt=f"{name} product hero shot",
                    width=320,
                    height=320,
                    source="mock",
                ),
                MediaRef(
                    id="photo_2",
                    path=str(p2),
                    mime_type="image/png",
                    alt=f"{name} product detail shot",
                    width=320,
                    height=320,
                    source="mock",
                ),
            ]
        else:
            photos = [
                MediaRef(
                    id="photo_1",
                    url=f"{brand_url.rstrip('/')}/images/hero.png",
                    mime_type="image/png",
                    alt=f"{name} product hero shot",
                    source="mock",
                )
            ]

        insp_notes = (
            [f"Reference pacing and composition inspired by {inspiration_url}"]
            if inspiration_url
            else []
        )

        return ResearchBundle(
            brand_url=brand_url,
            brand_name=name,
            verified_claims=[
                f"{name} ships nationwide in 48 hours",
                f"{name} is crafted in small batches",
            ],
            product_facts=[
                f"{name} flagship product built for daily use",
                "Backed by a 30-day satisfaction return policy",
            ],
            proof_points=["Rated 4.9 stars by verified customers"],
            voc_quotes=["Clean design and noticeable quality right out of the box."],
            brand_colors=colors,
            product_photos=photos,
            inspiration_notes=insp_notes,
        )


async def _run_ffmpeg(
    cmd: list[str],
    *,
    cancel: CancellationToken | None,
) -> None:
    if cancel is not None:
        cancel.raise_if_cancelled()

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    if cancel is not None:
        cancel.track_process(proc)
    try:
        _, stderr_bytes = await proc.communicate()
    finally:
        if cancel is not None:
            cancel.untrack_process(proc)

    if cancel is not None:
        cancel.raise_if_cancelled()

    if proc.returncode != 0:
        err = stderr_bytes.decode("utf-8", errors="replace").strip()
        raise VideoProviderError(f"FFmpeg synthetic render failed (exit {proc.returncode}): {err}")


class MockVideoProvider(VideoProvider):
    """Generates REAL synthetic MP4 clips via FFmpeg lavfi at exact aspect, fps, and duration."""

    def __init__(
        self,
        *,
        media_store: MediaStore | None = None,
        clip_cache: ContentCache | None = None,
        cost_meter: CostMeter | None = None,
        settings: Settings | None = None,
        mock_usd: float = 0.0,
        dimensions: dict[AspectRatio, tuple[int, int]] | None = None,
    ) -> None:
        super().__init__(
            settings=settings,
            media_store=media_store,
            clip_cache=clip_cache,
            cost_meter=cost_meter,
            usd_per_sec=0.0,
            min_seconds=0.0,
        )
        self._mock_usd = max(0.0, float(mock_usd))
        self._dimensions = dict(dimensions or MOCK_ASPECT_DIMENSIONS)

    def _cache_key(
        self,
        *,
        provider: str,
        model: str,
        shot: Shot,
        aspect: AspectRatio,
        fps: int,
    ) -> str:
        payload = {
            "provider": provider,
            "model": model,
            "prompt": shot.visual_prompt,
            "negative_prompt": shot.negative_prompt,
            "camera_move": shot.camera_move,
            "camera_angle": shot.camera_angle,
            "aspect": aspect,
            "fps": fps,
            "duration_sec": round(float(shot.duration_sec), 3),
            "reference_image": shot.reference_image or "",
        }
        if self._clip_cache is not None:
            return self._clip_cache.make_key(payload)
        return compute_cache_key("mock_video_path", payload)

    async def generate_clip(
        self,
        shot: Shot,
        aspect: AspectRatio,
        fps: int,
        cancel: CancellationToken | None = None,
        provider: str | None = None,
        model: str | None = None,
    ) -> RenderedClip:
        if cancel is not None:
            cancel.raise_if_cancelled()

        width, height = self._dimensions[aspect]
        chosen_provider = provider or "mock"
        chosen_model = model or "mock-lavfi-mp4"

        cache_key = self._cache_key(
            provider=chosen_provider,
            model=chosen_model,
            shot=shot,
            aspect=aspect,
            fps=fps,
        )
        if self._clip_cache is not None:
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

        meter = self._cost_meter or current_cost_meter()
        res_id: str | None = None
        if meter is not None:
            res_id = meter.reserve(
                f"mock_video:{shot.id}",
                self._mock_usd,
                stage="render_video",
                provider=chosen_provider,
                model=chosen_model,
            )

        out_dir = self._media_store.base_dir / "mock_clips"
        out_dir.mkdir(parents=True, exist_ok=True)
        suffix = cache_key[:12]
        out_path = out_dir / f"{shot.id}_{width}x{height}_{fps}fps_{suffix}.mp4"

        dur_str = f"{float(shot.duration_sec):.3f}"
        cmd = [
            self.settings.ffmpeg_bin,
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size={width}x{height}:rate={fps}:duration={dur_str}",
            "-f",
            "lavfi",
            "-i",
            f"anullsrc=r=48000:cl=stereo:d={dur_str}",
            "-t",
            dur_str,
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-r",
            str(fps),
            "-c:a",
            "aac",
            "-shortest",
            str(out_path),
        ]

        try:
            await _run_ffmpeg(cmd, cancel=cancel)
            if meter is not None and res_id is not None:
                meter.settle(res_id, self._mock_usd)
                res_id = None
        except BaseException:
            if meter is not None and res_id is not None:
                meter.release(res_id)
            raise

        if self._clip_cache is not None and cache_key is not None:
            self._clip_cache.put(
                cache_key,
                artifact_path=out_path,
                metadata={
                    "shot_id": shot.id,
                    "width": width,
                    "height": height,
                    "fps": fps,
                    "duration_sec": float(shot.duration_sec),
                },
            )

        return RenderedClip(
            shot_id=shot.id,
            path=str(out_path),
            duration_sec=float(shot.duration_sec),
            fps=fps,
            width=width,
            height=height,
            cache_hit=False,
            usd=self._mock_usd,
        )


class MockAudioProvider(AudioProvider):
    """Generates REAL synthetic WAV audio via FFmpeg sine/anullsrc."""

    def __init__(
        self,
        *,
        media_store: MediaStore | None = None,
        audio_cache: ContentCache | None = None,
        cost_meter: CostMeter | None = None,
        settings: Settings | None = None,
        mock_usd: float = 0.0,
    ) -> None:
        super().__init__(
            settings=settings,
            media_store=media_store,
            audio_cache=audio_cache,
            cost_meter=cost_meter,
        )
        self._mock_usd = max(0.0, float(mock_usd))

    async def _synthesize_wav(
        self,
        *,
        kind: Literal["vo", "sfx", "music"],
        cue_id: str,
        duration_sec: float,
        frequency_hz: int | None,
        cache_payload: dict[str, Any],
        cancel: CancellationToken | None,
    ) -> RenderedAudio:
        if cancel is not None:
            cancel.raise_if_cancelled()

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

        meter = self._cost_meter or current_cost_meter()
        res_id: str | None = None
        if meter is not None:
            res_id = meter.reserve(
                f"mock_audio:{kind}:{cue_id}",
                self._mock_usd,
                stage=f"render_{kind}",
                provider="mock",
                model="mock-lavfi-wav",
            )

        out_dir = self._media_store.base_dir / f"mock_audio_{kind}"
        out_dir.mkdir(parents=True, exist_ok=True)
        safe_id = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in cue_id)
        suffix = (
            cache_key[:12]
            if cache_key is not None
            else compute_cache_key("mock_audio_path", cache_payload)[:12]
        )
        out_path = out_dir / f"{safe_id}_{suffix}.wav"

        dur_str = f"{max(0.1, float(duration_sec)):.3f}"
        if frequency_hz is None:
            lavfi_src = f"anullsrc=r=44100:cl=stereo:d={dur_str}"
        else:
            lavfi_src = f"sine=frequency={frequency_hz}:sample_rate=44100:duration={dur_str}"

        cmd = [
            self.settings.ffmpeg_bin,
            "-y",
            "-f",
            "lavfi",
            "-i",
            lavfi_src,
            "-t",
            dur_str,
            "-ac",
            "2",
            "-c:a",
            "pcm_s16le",
            str(out_path),
        ]

        try:
            await _run_ffmpeg(cmd, cancel=cancel)
            if meter is not None and res_id is not None:
                meter.settle(res_id, self._mock_usd)
                res_id = None
        except BaseException:
            if meter is not None and res_id is not None:
                meter.release(res_id)
            raise

        if self._audio_cache is not None and cache_key is not None:
            self._audio_cache.put(
                cache_key,
                artifact_path=out_path,
                metadata={"kind": kind, "cue_id": cue_id, "duration_sec": float(duration_sec)},
            )

        return RenderedAudio(
            kind=kind,
            cue_id=cue_id,
            path=str(out_path),
            duration_sec=float(duration_sec),
            cache_hit=False,
            usd=self._mock_usd,
        )

    async def generate_voice(
        self,
        cue: VoiceCue,
        voice_id: str,
        cancel: CancellationToken | None = None,
        brand_pronunciations: dict[str, str] | None = None,
        model: str | None = None,
    ) -> RenderedAudio:
        return await self._synthesize_wav(
            kind="vo",
            cue_id=cue.shot_id,
            duration_sec=float(cue.duration_sec),
            frequency_hz=440,
            cache_payload={
                "kind": "vo",
                "voice_id": voice_id,
                "text": cue.text,
                "dur": float(cue.duration_sec),
            },
            cancel=cancel,
        )

    async def generate_sfx(
        self,
        cue: SfxCue,
        cancel: CancellationToken | None = None,
        model: str | None = None,
    ) -> RenderedAudio:
        return await self._synthesize_wav(
            kind="sfx",
            cue_id=cue.shot_id or f"sfx_{cue.at_sec:.2f}",
            duration_sec=float(cue.duration_sec),
            frequency_hz=660,
            cache_payload={"kind": "sfx", "prompt": cue.prompt, "dur": cue.duration_sec},
            cancel=cancel,
        )

    async def generate_music(
        self,
        cue: MusicCue,
        cancel: CancellationToken | None = None,
        model: str | None = None,
    ) -> RenderedAudio:
        return await self._synthesize_wav(
            kind="music",
            cue_id="music_bed",
            duration_sec=float(cue.duration_sec),
            frequency_hz=220,
            cache_payload={"kind": "music", "prompt": cue.prompt, "dur": cue.duration_sec},
            cancel=cancel,
        )


class MockWhisperProvider(WhisperProvider):
    """Deterministic offline WhisperProvider returning scripted transcripts."""

    def __init__(
        self,
        *,
        transcripts: Sequence[str] | None = None,
        default_transcript: str = "Mock voiceover transcript.",
    ) -> None:
        super().__init__()
        self._transcripts = list(transcripts or [])
        self._default_transcript = default_transcript
        self._cursor = 0

    async def transcribe(
        self,
        audio_path: Path,
        vocabulary: list[str] | None = None,
        cancel: CancellationToken | None = None,
    ) -> str:
        if cancel is not None:
            cancel.raise_if_cancelled()
        if not Path(audio_path).is_file():
            raise FileNotFoundError(f"Audio file does not exist: {audio_path}")
        if self._transcripts:
            idx = min(self._cursor, len(self._transcripts) - 1)
            self._cursor += 1
            return self._transcripts[idx]
        return self._default_transcript


@dataclass
class MockProviderBundle:
    """Zero-API-key provider bundle for `--mock` pipeline runs and offline testing."""

    chat_model: ScriptedChatModel
    scraper: MockBrandScraper
    video: MockVideoProvider
    audio: MockAudioProvider
    whisper: MockWhisperProvider
    llm_router: LLMRouter
    media_store: MediaStore
    cache_store: CacheStore

    @classmethod
    def create(
        cls,
        work_dir: Path,
        *,
        responses: Sequence[Any] | None = None,
        responder: Callable[[list[BaseMessage]], Any] | None = None,
        transcripts: Sequence[str] | None = None,
        cost_meter: CostMeter | None = None,
        settings: Settings | None = None,
    ) -> MockProviderBundle:
        root = Path(work_dir).resolve()
        media_dir = root / "media"
        cache_dir = root / "cache"
        media_store = MediaStore(media_dir, resolve_dns_for_ssrf=False)
        cache_store = CacheStore(cache_dir)
        cfg = settings or get_settings()

        return cls(
            chat_model=ScriptedChatModel(
                responses=list(responses or []),
                responder=responder,
            ),
            scraper=MockBrandScraper(media_dir=media_dir / "scraped"),
            video=MockVideoProvider(
                media_store=media_store,
                clip_cache=cache_store.clip_cache,
                cost_meter=cost_meter,
                settings=cfg,
            ),
            audio=MockAudioProvider(
                media_store=media_store,
                audio_cache=cache_store.audio_cache,
                cost_meter=cost_meter,
                settings=cfg,
            ),
            whisper=MockWhisperProvider(transcripts=transcripts),
            llm_router=LLMRouter(settings=cfg),
            media_store=media_store,
            cache_store=cache_store,
        )
