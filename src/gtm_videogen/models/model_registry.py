"""Declarative Dynamic Model Registry & schema-driven payload builder for Stitch.

Allows adding or updating future fal.ai / OpenRouter video, image, and LLM models
without Python code changes via:
  1. Built-in verified catalog (Kling 3.0 Pro 1080p, Kling 3.0 Native 4K, Kling 4.0 Flash,
     Google Veo 3 Fast / Flagship, Instant $0 Animatic, Flux Schnell/Kontext, OpenRouter LLMs).
  2. `STITCH_EXTRA_MODELS_JSON` environment variable (optional JSON list of ModelSpec dicts).
  3. Persistent user-added custom models in `<data_dir>/stitch/custom_models.json` via UI/API.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

ModelModality = Literal["video", "image", "llm"]
DurationFormat = Literal["int", "string", "seconds_suffix"]
PromptStyle = Literal["structured_json", "cinematic_prose"]


class VideoModelSpec(BaseModel):
    """Capability-driven contract for any video generation endpoint."""

    model_config = ConfigDict(extra="ignore")

    id: str = Field(..., min_length=2)
    i2v_id: str | None = None
    label: str = Field(..., min_length=2)
    tier: str = "master"  # smart | draft | master | 4k | audio | mock | custom
    provider: Literal["fal", "gemini", "openrouter", "mock"] = "fal"
    resolution: str = "1080p"
    prompt_style: PromptStyle = "structured_json"
    durations_sec: list[int] = Field(default_factory=lambda: [5, 8, 10, 15])
    default_duration_sec: int = 10
    duration_format: DurationFormat = "string"
    omit_duration: bool = False
    max_prompt_chars: int = 2400
    supports_start_frame: bool = True
    supports_end_frame: bool = False
    supports_elements: bool = False
    supports_native_audio: bool = False
    supports_negative_prompt: bool = True
    start_image_param: str = "start_image_url"
    end_image_param: str | None = "end_image_url"
    audio_param: str | None = "generate_audio"
    aspect_ratios: list[str] = Field(default_factory=lambda: ["16:9", "9:16", "1:1"])
    omit_aspect: bool = False
    omit_aspect_on_i2v: bool = True
    est_usd_per_sec: float = 0.112
    est_usd_per_sec_audio: float | None = None
    flat_usd_per_clip: float | None = None
    extra_params: dict[str, Any] = Field(default_factory=dict)
    fallback_id: str | None = "fal-ai/kling-video/v2.5-turbo/pro/text-to-video"
    custom: bool = False
    badge: str = ""

    def resolve_endpoint(self, *, has_start_frame: bool) -> str:
        """Pick T2V vs I2V endpoint slug cleanly."""
        if has_start_frame and self.i2v_id:
            return self.i2v_id.lstrip("/")
        if has_start_frame and self.id.endswith("/text-to-video"):
            return (self.id[: -len("/text-to-video")] + "/image-to-video").lstrip("/")
        return self.id.lstrip("/")

    def snap_duration(self, requested_sec: int | float) -> int:
        """Snap requested duration to the closest supported duration for this model."""
        if not self.durations_sec:
            return max(3, int(round(float(requested_sec))))
        target = float(requested_sec)
        return min(self.durations_sec, key=lambda d: abs(d - target))

    def format_duration(self, sec: int) -> int | str:
        snapped = self.snap_duration(sec)
        if self.duration_format == "int":
            return int(snapped)
        if self.duration_format == "seconds_suffix":
            return f"{snapped}s"
        return str(snapped)

    def estimate_clip_usd(self, duration_sec: int | float, *, with_audio: bool = False) -> float:
        """Estimate USD cost for a single rendered clip honoring silent vs audio & flat-rate pricing."""
        if self.provider == "mock" or self.est_usd_per_sec <= 0.0 and self.flat_usd_per_clip is None:
            return 0.0
        if self.flat_usd_per_clip is not None:
            return round(float(self.flat_usd_per_clip), 4)
        snapped = float(self.snap_duration(duration_sec))
        rate = (
            float(self.est_usd_per_sec_audio)
            if (with_audio and self.supports_native_audio and self.est_usd_per_sec_audio is not None)
            else float(self.est_usd_per_sec)
        )
        return round(snapped * rate, 4)


BUILTIN_VIDEO_MODELS: list[VideoModelSpec] = [
    VideoModelSpec(
        id="auto",
        i2v_id="auto",
        label="Smart Multi-Model Router (Best Quality / $)",
        tier="smart",
        provider="fal",
        resolution="1080p",
        prompt_style="cinematic_prose",
        durations_sec=[5, 6, 8, 10, 15],
        default_duration_sec=8,
        duration_format="string",
        max_prompt_chars=2400,
        supports_start_frame=True,
        supports_end_frame=True,
        supports_elements=True,
        supports_native_audio=True,
        est_usd_per_sec=0.078,
        fallback_id="fal-ai/kling-video/v2.5-turbo/pro/text-to-video",
        badge="Recommended · ~75% Cheaper",
    ),
    VideoModelSpec(
        id="fal-ai/kling-video/v2.5-turbo/pro/text-to-video",
        i2v_id="fal-ai/kling-video/v2.5-turbo/pro/image-to-video",
        label="Kling 2.5 Turbo Pro (1080p Value)",
        tier="draft",
        provider="fal",
        resolution="1080p",
        prompt_style="cinematic_prose",
        durations_sec=[5, 10],
        default_duration_sec=5,
        duration_format="string",
        max_prompt_chars=2200,
        supports_start_frame=True,
        supports_end_frame=False,
        supports_elements=False,
        supports_native_audio=False,
        start_image_param="image_url",
        end_image_param=None,
        audio_param=None,
        omit_aspect_on_i2v=True,
        est_usd_per_sec=0.07,
        fallback_id="fal-ai/bytedance/seedance/v1/lite/text-to-video",
        badge="1080p Value · $0.07/s",
    ),
    VideoModelSpec(
        id="fal-ai/bytedance/seedance/v1/lite/text-to-video",
        i2v_id="fal-ai/bytedance/seedance/v1/lite/image-to-video",
        label="ByteDance Seedance Lite (Fast Test Draft)",
        tier="draft",
        provider="fal",
        resolution="720p",
        prompt_style="cinematic_prose",
        durations_sec=[5, 8, 10],
        default_duration_sec=5,
        duration_format="string",
        max_prompt_chars=1800,
        supports_start_frame=True,
        supports_end_frame=True,
        supports_elements=False,
        supports_native_audio=False,
        supports_negative_prompt=False,
        start_image_param="image_url",
        end_image_param="end_image_url",
        audio_param=None,
        omit_aspect_on_i2v=False,
        extra_params={"resolution": "720p"},
        est_usd_per_sec=0.036,
        fallback_id="fal-ai/wan-25-preview/text-to-video",
        badge="$0.18 per 5s Test",
    ),
    VideoModelSpec(
        id="fal-ai/wan-25-preview/text-to-video",
        i2v_id="fal-ai/wan-25-preview/image-to-video",
        label="Alibaba Wan 2.5 (Budget Motion)",
        tier="draft",
        provider="fal",
        resolution="720p",
        prompt_style="cinematic_prose",
        durations_sec=[5, 10],
        default_duration_sec=5,
        duration_format="string",
        max_prompt_chars=1800,
        supports_start_frame=True,
        supports_end_frame=False,
        supports_elements=False,
        supports_native_audio=False,
        start_image_param="image_url",
        end_image_param=None,
        audio_param=None,
        omit_aspect_on_i2v=True,
        est_usd_per_sec=0.05,
        fallback_id="fal-ai/bytedance/seedance/v1/lite/text-to-video",
        badge="Budget · $0.05/s",
    ),
    VideoModelSpec(
        id="fal-ai/minimax/hailuo-2.3/pro/text-to-video",
        i2v_id="fal-ai/minimax/hailuo-2.3/pro/image-to-video",
        label="MiniMax Hailuo 2.3 Pro (1080p Flat $0.49)",
        tier="master",
        provider="fal",
        resolution="1080p",
        prompt_style="cinematic_prose",
        durations_sec=[6, 8],
        default_duration_sec=6,
        duration_format="int",
        omit_duration=True,
        omit_aspect=True,
        max_prompt_chars=2000,
        supports_start_frame=True,
        supports_end_frame=False,
        supports_elements=False,
        supports_native_audio=False,
        supports_negative_prompt=False,
        start_image_param="image_url",
        end_image_param=None,
        audio_param=None,
        extra_params={"prompt_optimizer": True},
        est_usd_per_sec=0.075,
        flat_usd_per_clip=0.49,
        fallback_id="fal-ai/kling-video/v2.5-turbo/pro/text-to-video",
        badge="1080p Physics · $0.49 Flat",
    ),
    VideoModelSpec(
        id="fal-ai/kling-video/v3/standard/text-to-video",
        i2v_id="fal-ai/kling-video/v3/standard/image-to-video",
        label="Kling 3.0 Standard (Chained + @Elements)",
        tier="master",
        provider="fal",
        resolution="1080p",
        prompt_style="structured_json",
        durations_sec=[5, 8, 10, 15],
        default_duration_sec=8,
        duration_format="string",
        max_prompt_chars=2400,
        supports_start_frame=True,
        supports_end_frame=True,
        supports_elements=True,
        supports_native_audio=True,
        start_image_param="start_image_url",
        end_image_param="end_image_url",
        audio_param="generate_audio",
        omit_aspect_on_i2v=True,
        est_usd_per_sec=0.084,
        est_usd_per_sec_audio=0.126,
        fallback_id="fal-ai/kling-video/v2.5-turbo/pro/text-to-video",
        badge="Best 60s Continuity · $0.084/s",
    ),
    VideoModelSpec(
        id="fal-ai/kling-video/v3/pro/text-to-video",
        i2v_id="fal-ai/kling-video/v3/pro/image-to-video",
        label="Kling 3.0 Pro Cinema",
        tier="master",
        provider="fal",
        resolution="1080p",
        prompt_style="structured_json",
        durations_sec=[5, 8, 10, 15],
        default_duration_sec=10,
        duration_format="string",
        max_prompt_chars=2400,
        supports_start_frame=True,
        supports_end_frame=True,
        supports_elements=True,
        supports_native_audio=True,
        start_image_param="start_image_url",
        end_image_param="end_image_url",
        audio_param="generate_audio",
        omit_aspect_on_i2v=True,
        est_usd_per_sec=0.112,
        est_usd_per_sec_audio=0.168,
        fallback_id="fal-ai/kling-video/v3/standard/text-to-video",
        badge="Hero Cinema · $0.112/s",
    ),
    VideoModelSpec(
        id="kling/v4/flash/text-to-video",
        i2v_id="kling/v4/flash/image-to-video",
        label="Kling 4.0 Flash (Fast Draft)",
        tier="draft",
        provider="fal",
        resolution="720p",
        prompt_style="structured_json",
        durations_sec=[5, 8, 10, 15],
        default_duration_sec=10,
        duration_format="int",
        max_prompt_chars=2400,
        supports_start_frame=True,
        supports_end_frame=False,
        supports_elements=False,
        supports_native_audio=False,
        supports_negative_prompt=False,
        start_image_param="start_image_url",
        end_image_param=None,
        audio_param=None,
        omit_aspect_on_i2v=True,
        est_usd_per_sec=0.094,
        extra_params={"resolution": "720p"},
        fallback_id="fal-ai/kling-video/v2.5-turbo/pro/text-to-video",
        badge="Fast Draft",
    ),
    VideoModelSpec(
        id="fal-ai/kling-video/v3/4k/image-to-video",
        i2v_id="fal-ai/kling-video/v3/4k/image-to-video",
        label="Kling 3.0 Native 4K",
        tier="4k",
        provider="fal",
        resolution="4K",
        prompt_style="structured_json",
        durations_sec=[5, 8, 10, 15],
        default_duration_sec=10,
        duration_format="string",
        max_prompt_chars=2400,
        supports_start_frame=True,
        supports_end_frame=True,
        supports_elements=True,
        supports_native_audio=True,
        start_image_param="start_image_url",
        end_image_param="end_image_url",
        audio_param="generate_audio",
        omit_aspect_on_i2v=True,
        est_usd_per_sec=0.42,
        fallback_id="fal-ai/kling-video/v3/pro/text-to-video",
        badge="Native 4K",
    ),
    VideoModelSpec(
        id="fal-ai/veo3/fast",
        i2v_id="fal-ai/veo3/image-to-video",
        label="Google Veo 3 Fast",
        tier="audio",
        provider="fal",
        resolution="720p",
        prompt_style="cinematic_prose",
        durations_sec=[4, 6, 8],
        default_duration_sec=8,
        duration_format="seconds_suffix",
        max_prompt_chars=2200,
        supports_start_frame=True,
        supports_end_frame=False,
        supports_elements=False,
        supports_native_audio=True,
        start_image_param="image_url",
        end_image_param=None,
        audio_param="generate_audio",
        omit_aspect_on_i2v=False,
        est_usd_per_sec=0.15,
        extra_params={"resolution": "720p", "auto_fix": True},
        fallback_id="fal-ai/kling-video/v3/standard/text-to-video",
        badge="Native ASMR Audio",
    ),
    VideoModelSpec(
        id="fal-ai/veo3",
        i2v_id="fal-ai/veo3/image-to-video",
        label="Google Veo 3 Flagship",
        tier="master",
        provider="fal",
        resolution="1080p",
        prompt_style="cinematic_prose",
        durations_sec=[4, 6, 8],
        default_duration_sec=8,
        duration_format="seconds_suffix",
        max_prompt_chars=2200,
        supports_start_frame=True,
        supports_end_frame=False,
        supports_elements=False,
        supports_native_audio=True,
        start_image_param="image_url",
        end_image_param=None,
        audio_param="generate_audio",
        omit_aspect_on_i2v=False,
        est_usd_per_sec=0.30,
        extra_params={"resolution": "720p", "auto_fix": True},
        fallback_id="fal-ai/veo3/fast",
        badge="Studio Flagship",
    ),
    VideoModelSpec(
        id="gemini-omni-1.2-flash",
        i2v_id="gemini-omni-1.2-flash",
        label="Google Gemini Omni 1.2 (Gemini CLI)",
        tier="master",
        provider="gemini",
        resolution="1080p",
        prompt_style="cinematic_prose",
        durations_sec=[5, 8, 10],
        default_duration_sec=8,
        duration_format="int",
        max_prompt_chars=2800,
        supports_start_frame=True,
        supports_end_frame=False,
        supports_elements=False,
        supports_native_audio=True,
        est_usd_per_sec=0.12,
        fallback_id="fal-ai/veo3/fast",
        badge="Gemini CLI Native",
    ),
    VideoModelSpec(
        id="mock",
        i2v_id="mock",
        label="Studio Animatic ($0 Offline)",
        tier="mock",
        provider="mock",
        resolution="1080p",
        prompt_style="structured_json",
        durations_sec=[5, 8, 10, 15],
        default_duration_sec=10,
        duration_format="int",
        max_prompt_chars=4000,
        supports_start_frame=True,
        supports_end_frame=True,
        supports_elements=True,
        supports_native_audio=False,
        est_usd_per_sec=0.0,
        fallback_id=None,
        badge="$0 Preview",
    ),
]

BUILTIN_LLM_MODELS: list[dict[str, str]] = [
    {"id": "", "label": "Server Default Flagship LLM (Auto: OpenRouter / Gemini CLI)"},
    {"id": "gemini-cli/gemini-4-hydrogen", "label": "Gemini 4 Hydrogen (Gemini CLI Native)"},
    {"id": "google/gemini-2.5-pro", "label": "Gemini 2.5 Pro (Director Craft)"},
    {"id": "anthropic/claude-sonnet-4", "label": "Claude Sonnet 4 (Story & Beats)"},
    {"id": "openai/gpt-4o", "label": "GPT-4o (Structured JSON)"},
    {"id": "google/gemini-2.5-flash", "label": "Gemini 2.5 Flash (Fast & Low-Cost)"},
]


def _is_image_ref(val: str | None) -> bool:
    return bool(val and val.startswith(("http://", "https://", "data:image/")))


def build_registry_fal_payload(
    spec: VideoModelSpec,
    *,
    prompt: str,
    duration_sec: int | float,
    aspect_ratio: str = "16:9",
    start_image: str | None = None,
    end_image: str | None = None,
    negative_prompt: str | None = None,
    generate_audio: bool = False,
    element_image_urls: list[str] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Build (endpoint_slug, payload_dict) dynamically from VideoModelSpec capabilities."""
    has_start = bool(spec.supports_start_frame and _is_image_ref(start_image))
    endpoint = spec.resolve_endpoint(has_start_frame=has_start)
    safe_prompt = prompt.strip()[: max(200, spec.max_prompt_chars)]

    payload: dict[str, Any] = {
        "prompt": safe_prompt,
    }
    if not spec.omit_duration:
        payload["duration"] = spec.format_duration(int(round(float(duration_sec))))
    if spec.extra_params:
        payload.update(spec.extra_params)

    if spec.audio_param:
        payload[spec.audio_param] = bool(generate_audio and spec.supports_native_audio)

    if (
        spec.supports_negative_prompt
        and negative_prompt
        and negative_prompt.strip()
        and "kling/v4" not in endpoint.lower()
    ):
        payload["negative_prompt"] = negative_prompt.strip()[:500]

    if has_start:
        payload[spec.start_image_param] = start_image
        if spec.supports_end_frame and spec.end_image_param and _is_image_ref(end_image):
            payload[spec.end_image_param] = end_image
        if not spec.omit_aspect and not spec.omit_aspect_on_i2v:
            chosen_aspect = (
                aspect_ratio if aspect_ratio in spec.aspect_ratios else spec.aspect_ratios[0]
            )
            payload["aspect_ratio"] = chosen_aspect
    elif not spec.omit_aspect:
        chosen_aspect = (
            aspect_ratio if aspect_ratio in spec.aspect_ratios else spec.aspect_ratios[0]
        )
        payload["aspect_ratio"] = chosen_aspect

    if spec.supports_elements and element_image_urls:
        valid_els = [u for u in element_image_urls if _is_image_ref(u)][:3]
        if valid_els:
            payload["elements"] = [{"frontal_image_url": u} for u in valid_els]

    return endpoint, payload


class ModelRegistryStore:
    """Loads built-in, env-configured, and user-added custom models dynamically."""

    def __init__(self, stitch_dir: Path | str) -> None:
        self.stitch_dir = Path(stitch_dir).resolve()
        self.stitch_dir.mkdir(parents=True, exist_ok=True)
        self.custom_file = self.stitch_dir / "custom_models.json"

    def _load_custom_models(self) -> list[VideoModelSpec]:
        out: list[VideoModelSpec] = []
        raw_env = os.getenv("STITCH_EXTRA_MODELS_JSON", "").strip()
        if raw_env:
            try:
                items = json.loads(raw_env)
                if isinstance(items, list):
                    for item in items:
                        if isinstance(item, dict):
                            spec = VideoModelSpec.model_validate({**item, "custom": True})
                            out.append(spec)
            except Exception:
                pass
        if self.custom_file.is_file():
            try:
                items = json.loads(self.custom_file.read_text(encoding="utf-8"))
                if isinstance(items, list):
                    for item in items:
                        if isinstance(item, dict):
                            spec = VideoModelSpec.model_validate({**item, "custom": True})
                            out.append(spec)
            except Exception:
                pass
        return out

    def list_video_models(self) -> list[VideoModelSpec]:
        by_id: dict[str, VideoModelSpec] = {m.id: m for m in BUILTIN_VIDEO_MODELS}
        for cm in self._load_custom_models():
            by_id[cm.id] = cm
        # Put custom models cleanly right after the default master or preserve order
        ordered: list[VideoModelSpec] = []
        seen: set[str] = set()
        for m in BUILTIN_VIDEO_MODELS:
            ordered.append(by_id[m.id])
            seen.add(m.id)
        for cm in self._load_custom_models():
            if cm.id not in seen:
                ordered.insert(1, cm)
                seen.add(cm.id)
        return ordered

    def get_video_model(self, model_id: str) -> VideoModelSpec:
        clean = (model_id or "").strip().lstrip("/")
        # Safe alias for legacy playground-gated kling/v4 non-flash IDs
        if clean in ("kling/v4/text-to-video", "kling/v4/image-to-video"):
            clean = "fal-ai/kling-video/v3/pro/text-to-video"
        for m in self.list_video_models():
            if m.id == clean or (m.i2v_id and m.i2v_id == clean):
                return m
        # Fallback generic dynamic spec if user passed a brand-new endpoint directly
        if clean and clean != "mock":
            return VideoModelSpec(
                id=clean,
                label=clean.split("/")[-1],
                tier="custom",
                provider="fal",
                custom=True,
            )
        return BUILTIN_VIDEO_MODELS[0]

    def add_custom_model(self, spec: VideoModelSpec) -> VideoModelSpec:
        spec = spec.model_copy(update={"custom": True, "id": spec.id.strip().lstrip("/")})
        existing = [m for m in self._read_file_models() if m.id != spec.id]
        existing.append(spec)
        self._write_file_models(existing)
        return spec

    def delete_custom_model(self, model_id: str) -> bool:
        clean = model_id.strip().lstrip("/")
        existing = self._read_file_models()
        remaining = [m for m in existing if m.id != clean]
        if len(remaining) == len(existing):
            return False
        self._write_file_models(remaining)
        return True

    def _read_file_models(self) -> list[VideoModelSpec]:
        if not self.custom_file.is_file():
            return []
        try:
            raw = json.loads(self.custom_file.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                return [VideoModelSpec.model_validate(x) for x in raw if isinstance(x, dict)]
        except Exception:
            return []
        return []

    def _write_file_models(self, specs: list[VideoModelSpec]) -> None:
        payload = json.dumps([s.model_dump() for s in specs], indent=2)
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=self.stitch_dir,
            prefix=".models.",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            tmp.write(payload)
            tmp_path = Path(tmp.name)
        os.replace(tmp_path, self.custom_file)


def select_optimal_scene_model(
    registry: ModelRegistryStore,
    *,
    project_model_id: str,
    scene_model_id: str | None = None,
    narrative_role: str = "",
    duration_sec: int = 8,
    has_elements: bool = False,
    has_end_frame: bool = False,
    needs_native_audio: bool = False,
    aspect_ratio: str = "16:9",
) -> VideoModelSpec:
    """Resolve the optimal concrete VideoModelSpec for one Act (honors per-act overrides & 'auto' router)."""
    sc_override = (scene_model_id or "").strip()
    if sc_override and sc_override != "auto":
        return registry.get_video_model(sc_override)

    proj_choice = (project_model_id or "").strip()
    if proj_choice and proj_choice != "auto":
        return registry.get_video_model(proj_choice)

    # Smart Multi-Model Router ('auto'): pick best price/performance + continuity engine per Act!
    role = (narrative_role or "").strip().lower()
    if needs_native_audio and duration_sec <= 8 and not has_elements and not has_end_frame:
        return registry.get_video_model("fal-ai/veo3/fast")

    if has_elements or has_end_frame or duration_sec >= 12:
        # Acts needing @Elements identity lock or Start->End bridging:
        # Use Kling 3.0 Pro ($0.112/s silent) on Hero Hook, and Kling 3.0 Standard ($0.084/s silent) elsewhere!
        if role == "hook":
            return registry.get_video_model("fal-ai/kling-video/v3/pro/text-to-video")
        return registry.get_video_model("fal-ai/kling-video/v3/standard/text-to-video")

    # Standard 5s-10s cinema acts without @Elements lock:
    if aspect_ratio == "16:9" and 6 <= duration_sec <= 8:
        # Hailuo 2.3 Pro delivers crisp 1080p cinema physics at a flat $0.49 per clip!
        return registry.get_video_model("fal-ai/minimax/hailuo-2.3/pro/text-to-video")
    if duration_sec <= 5:
        # Crisp 1080p Kling 2.5 Turbo Pro at $0.07/sec ($0.35 for 5s!)
        return registry.get_video_model("fal-ai/kling-video/v2.5-turbo/pro/text-to-video")
    if role in ("hook", "climax_cta"):
        return registry.get_video_model("fal-ai/kling-video/v3/standard/text-to-video")
    return registry.get_video_model("fal-ai/kling-video/v2.5-turbo/pro/text-to-video")


__all__ = [
    "BUILTIN_LLM_MODELS",
    "BUILTIN_VIDEO_MODELS",
    "ModelRegistryStore",
    "PromptStyle",
    "VideoModelSpec",
    "build_registry_fal_payload",
    "select_optimal_scene_model",
]
