"""Human-in-the-Loop (HITL) step-by-step Studio API endpoints.

Supports the 5-step interactive workflow:
  1. Brand & Goal  -> POST /api/runs/{run_id}/step/research (Agent 1 only)
  2. Concept/Bible -> PATCH /api/runs/{run_id}/concept      (select pitch, edit Bible/facts)
  3. Shot Sheet    -> POST /api/runs/{run_id}/step/direct   (Agent 2 only)
                   -> PATCH /api/runs/{run_id}/sheet        (edit shots + live script-gate check)
  4. Previews      -> POST /api/runs/{run_id}/step/previews (1-4 A/B model stacks side-by-side)
                   -> GET  /api/runs/{run_id}/versions/{version_id}/video (HTTP 206 streaming)
                   -> POST /api/runs/{run_id}/versions/{version_id}/pick  (pick winning preview)
  5. Final Cut     -> POST /api/runs/{run_id}/step/final    (render winner at full final quality)
  Budget Cap       -> PATCH /api/runs/{run_id}/cap          (adjust USD budget cap live)
"""

from __future__ import annotations

import asyncio
import base64
import copy
import io
import math
import mimetypes
import re
import struct
import wave
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from gtm_videogen.agents.agent1_research_bible import run_agent1_research_bible
from gtm_videogen.agents.agent2_director import (
    _build_script_from_sheet,
    _score_craft_dimensions,
    run_agent2_director,
)
from gtm_videogen.agents.agent3_producer import run_agent3_producer
from gtm_videogen.agents.pipeline import PipelineRunner
from gtm_videogen.api.middleware import MAX_RUN_ID_LENGTH, load_owned_run
from gtm_videogen.api.routes import (
    TERMINAL_EVENT_TYPES,
    ActorDep,
    ServicesDep,
    _iter_file_range,
    _parse_byte_range,
    make_sse_event,
)
from gtm_videogen.core.cache_store import ContentCache
from gtm_videogen.core.cost_meter import CostMeter
from gtm_videogen.core.job_registry import CancellationToken
from gtm_videogen.core.media_store import MediaStore
from gtm_videogen.media.vision_critic import (
    CRAFT_ARCHETYPES,
    VisionQualityCritic,
    analyze_reference_moodboard,
)
from gtm_videogen.models.schemas import (
    CostEntry,
    CreativeBible,
    CreativeFingerprint,
    HumanScore,
    MediaRef,
    MusicCue,
    RunState,
    Script,
    ShotSheet,
    SoundPlan,
    Version,
    VoiceCue,
)
from gtm_videogen.providers.audio import AudioProvider
from gtm_videogen.providers.storyboard import StoryboardGenerator
from gtm_videogen.rules.creative_history import CreativeHistoryStore
from gtm_videogen.rules.script_gate import evaluate_script_gate

studio_router = APIRouter(prefix="/api")

# Streamlined, high-clarity preview stacks
PREVIEW_STACK_PRESETS: list[dict[str, Any]] = [
    {
        "id": "hybrid_motion",
        "label": "Launch Film Hybrid (Recommended)",
        "description": (
            "Anthropic/OpenAI style: continuity-locked Kling v4 Flash hero shots "
            "+ $0 kinetic typography, product/UI bezel, and brand logo screens."
        ),
        "video_model": "kling/v4/flash/text-to-video",
        "voice_model": "fal-ai/elevenlabs/tts/eleven-v3",
        "hybrid_motion_cards": True,
        "use_keyframes": True,
        "mock": False,
        "est_per_ai_shot_usd": 0.35,
    },
    {
        "id": "kling_flash",
        "label": "Full Cinema AI (Keyframe-Anchored)",
        "description": (
            "Animates every approved Step 4 keyframe via Kling v4 Flash I2V ($0.094/s) "
            "+ ElevenLabs v3 voiceover."
        ),
        "video_model": "kling/v4/flash/text-to-video",
        "voice_model": "fal-ai/elevenlabs/tts/eleven-v3",
        "hybrid_motion_cards": False,
        "use_keyframes": True,
        "mock": False,
        "est_per_ai_shot_usd": 0.35,
    },
    {
        "id": "mock_pacing",
        "label": "Instant Pacing Animatic ($0)",
        "description": (
            "Zero-cost FFmpeg timing preview to verify pacing, voice cadence, "
            "and shot rhythm before spending."
        ),
        "video_model": "mock-lavfi-mp4",
        "voice_model": "mock-lavfi-wav",
        "hybrid_motion_cards": False,
        "use_keyframes": False,
        "mock": True,
        "est_per_ai_shot_usd": 0.0,
    },
]

# Upgrade 4: Flagship Final Cut mastering engines selectable in Step 6
FLAGSHIP_MASTER_MODELS: list[dict[str, Any]] = [
    {
        "id": "",
        "label": "Same as Winning Preview (Kling v4 Flash)",
        "description": "Fast, cost-efficient master using the exact model stack you approved.",
    },
    {
        "id": "kling/v4/text-to-video",
        "label": "Kling v4 Pro Cinema (Start→End Frame Bridged)",
        "description": "High-craft Kling v4 Pro with Start→End keyframe bridging across adjacent shots.",
    },
    {
        "id": "fal-ai/veo3/fast",
        "label": "Google Veo 3 Fast (Hero Realism)",
        "description": "Google Veo 3 Fast for photorealistic lighting, physics, and keyframe animation.",
    },
    {
        "id": "fal-ai/veo3",
        "label": "Google Veo 3 Flagship (Studio Master)",
        "description": "Top-tier Google Veo 3 flagship rendering for showcase hero launches.",
    },
]


def _resolve_brand_assets(run_state: RunState) -> tuple[Path | None, list[Path]]:
    """Return (brand_logo_path, product_image_paths) for existing uploaded files on disk."""
    logo: Path | None = None
    if run_state.brand_logo_path:
        lp = Path(run_state.brand_logo_path)
        if lp.is_file():
            logo = lp
    products: list[Path] = []
    for ref in run_state.uploaded_assets:
        if ref.path:
            pp = Path(ref.path)
            if pp.is_file():
                products.append(pp)
    return logo, products


def _wire_adjacent_end_keyframes(sheet: ShotSheet) -> None:
    """Populate Start->End keyframe links between adjacent approved AI shots for Kling v4 Pro."""
    shots = sheet.shots
    for idx, sh in enumerate(shots):
        if (
            sh.engine in ("ai_video", "image_to_video")
            and sh.keyframe_approved
            and sh.keyframe_url
            and sh.keyframe_url.startswith(("http://", "https://"))
        ):
            sh.engine = "image_to_video"
            sh.reference_image = sh.keyframe_url
            if idx + 1 < len(shots):
                nxt = shots[idx + 1]
                if (
                    nxt.engine in ("ai_video", "image_to_video")
                    and nxt.keyframe_approved
                    and nxt.keyframe_url
                    and nxt.keyframe_url.startswith(("http://", "https://"))
                    and nxt.keyframe_url != sh.keyframe_url
                ):
                    sh.end_reference_image = nxt.keyframe_url


def _synthesize_mock_audition_wav(kind: str, label: str = "") -> bytes:
    """Generate a clean 2.2s studio chime WAV for zero-cost offline voice/music auditions."""
    sr = 22050
    dur = 2.2
    n = int(sr * dur)
    base_hz = 260.0 if kind == "music" else 340.0
    if "george" in label.lower():
        base_hz = 210.0
    elif "river" in label.lower():
        base_hz = 290.0
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        frames = bytearray()
        for i in range(n):
            t = i / sr
            env = min(1.0, t / 0.08) * max(0.0, 1.0 - max(0.0, (t - (dur - 0.35)) / 0.35))
            sig = 0.55 * math.sin(2.0 * math.pi * base_hz * t) + 0.25 * math.sin(
                2.0 * math.pi * (base_hz * 1.5) * t
            )
            sample = int(max(-32000, min(32000, sig * env * 12000)))
            frames.extend(struct.pack("<h", sample))
        wf.writeframes(bytes(frames))
    return buf.getvalue()


class ConceptPatchRequest(BaseModel):
    """Human-in-the-loop edits to concept selection, CreativeBible, or verified facts."""

    model_config = ConfigDict(extra="forbid")

    selected_concept_id: str | None = None
    concept_thesis: str | None = None
    arc_name: str | None = None
    world: str | None = None
    lighting_style: str | None = None
    color_palette: list[str] | None = None
    camera_style: str | None = None
    font_family: str | None = None
    audio_strategy: str | None = None
    anti_cliche_list: list[str] | None = None
    facts: list[str] | None = None
    offer: str | None = None
    must_say: list[str] | None = None
    user_direction: str | None = None
    creative_bible: CreativeBible | None = None


class SheetPatchRequest(BaseModel):
    """Human-in-the-loop edits to ShotSheet (and optional SoundPlan/Script)."""

    model_config = ConfigDict(extra="forbid")

    shot_sheet: ShotSheet
    sound_plan: SoundPlan | None = None
    script: Script | None = None


class KeyframeRegenRequest(BaseModel):
    """Regenerate or approve a single shot's visual storyboard keyframe."""

    model_config = ConfigDict(extra="forbid")

    visual_prompt: str | None = None
    approved: bool | None = None
    regenerate: bool = True


class PreviewStackSpec(BaseModel):
    """Single preview option / model stack chosen for A/B comparison."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=64)
    label: str = Field(min_length=1, max_length=120)
    video_model: str = "kling/v4/flash/text-to-video"
    voice_model: str = "fal-ai/elevenlabs/tts/eleven-v3"
    hybrid_motion_cards: bool = False
    use_keyframes: bool = True
    mock: bool = False


class PreviewsRequest(BaseModel):
    """Launch 1-4 preview stacks for side-by-side comparison."""

    model_config = ConfigDict(extra="forbid")

    stacks: list[PreviewStackSpec] = Field(min_length=1, max_length=4)


class FinalRenderRequest(BaseModel):
    """Optional override for Final Cut (including Step 6 Flagship Master video engine)."""

    model_config = ConfigDict(extra="forbid")

    version_id: str | None = None
    mock: bool | None = None
    video_model_override: str | None = None


class CapPatchRequest(BaseModel):
    """Live USD budget cap adjustment."""

    model_config = ConfigDict(extra="forbid")

    cap_usd: float = Field(ge=0.0, le=500.0)


class ReiterateRequest(BaseModel):
    """Targeted quality reiteration on weak or user-flagged shots."""

    model_config = ConfigDict(extra="forbid")

    shot_ids: list[str] | None = None
    version_id: str | None = None
    notes: str = ""
    regenerate_keyframes: bool = True


class AssetUploadRequest(BaseModel):
    """Upload a real brand logo or product/UI image as a base64 data URL (max ~8MB)."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["logo", "product"] = "product"
    filename: str = Field(min_length=1, max_length=120)
    data_url: str = Field(min_length=20)
    alt: str = ""


class AuditionRequest(BaseModel):
    """Request a short cached voice or music audition clip for Step 3."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["voice", "music"] = "voice"
    voice_id: str = "Rachel"
    text: str = "Built for teams that ship with quiet precision."
    prompt: str = "Minimal warm analog synth pulse, clean modern product launch music"
    mock: bool = False


class ReferenceTasteRequest(BaseModel):
    """Apply a Craft Archetype preset and/or analyze an uploaded reference moodboard image."""

    model_config = ConfigDict(extra="forbid")

    craft_archetype: str | None = None
    reference_image_data_url: str | None = None
    notes: str = ""


class AutoPolishRequest(BaseModel):
    """Autonomous Auto-Polish loop targeting >= target_score (up to 2 rounds within USD cap)."""

    model_config = ConfigDict(extra="forbid")

    version_id: str | None = None
    target_score: float = Field(8.5, ge=5.0, le=10.0)
    max_rounds: int = Field(2, ge=1, le=2)
    notes: str = ""


def _estimate_stack_usd(sheet: ShotSheet | None, stack: dict[str, Any]) -> float:
    if stack.get("mock"):
        return 0.0
    shots = sheet.shots if sheet else []
    n_shots = len(shots) if shots else 4
    if stack.get("hybrid_motion_cards"):
        ai_shots = max(1, (n_shots + 1) // 2)
    else:
        ai_shots = sum(1 for s in shots if s.engine in ("ai_video", "image_to_video")) or n_shots
    per_shot = float(stack.get("est_per_ai_shot_usd", 0.35))
    audio_est = 0.01 * n_shots
    return round(ai_shots * per_shot + audio_est, 2)


def _apply_hybrid_motion_cards(sheet: ShotSheet) -> ShotSheet:
    """Return a copy where closing/supporting shots use $0 motion_card while hero beats stay AI."""
    cloned = sheet.model_copy(deep=True)
    total = len(cloned.shots)
    for idx, shot in enumerate(cloned.shots):
        if idx == total - 1 or (total >= 4 and idx == total - 2):
            shot.engine = "motion_card"
            if not shot.on_screen_text:
                shot.on_screen_text = (
                    cloned.end_card_title if idx == total - 1 else (shot.vo_line or shot.beat)
                )
    return cloned


def _merge_cost_ledger(run_state: RunState, delta_entries: list[CostEntry]) -> None:
    """Append newly incurred CostEntries into run_state.cost_ledger without losing prior spend."""
    existing_keys = {
        (e.timestamp.isoformat(), e.stage, e.what, e.actual_usd)
        for e in run_state.cost_ledger.entries
    }
    for entry in delta_entries:
        key = (entry.timestamp.isoformat(), entry.stage, entry.what, entry.actual_usd)
        if key not in existing_keys:
            run_state.cost_ledger.entries.append(entry)
            existing_keys.add(key)
    run_state.cost_ledger.actual_usd = round(
        sum(e.actual_usd for e in run_state.cost_ledger.entries), 6
    )


def _build_step_providers(
    services: Any,
    run_state: RunState,
    work_dir: Path,
    meter: CostMeter,
    *,
    force_mock: bool = False,
    video_model_override: str | None = None,
    voice_model_override: str | None = None,
) -> Any:
    """Build live or mock provider bundle honoring per-version stack model overrides."""
    settings_copy = copy.deepcopy(services.settings)
    if video_model_override and not force_mock:
        settings_copy.video_model = video_model_override
    if voice_model_override and not force_mock:
        settings_copy.voice_model = voice_model_override

    state_for_bundle = run_state.model_copy(deep=True)
    if force_mock:
        state_for_bundle.input.mock = True

    runner = PipelineRunner(
        work_dir=work_dir,
        settings=settings_copy,
        history_store=CreativeHistoryStore(services.data_dir / "creative_history.json"),
    )
    return runner._resolve_providers(
        run_state=state_for_bundle,
        run_dir=work_dir,
        cost_meter=meter,
    )


@studio_router.get("/studio/presets")
async def list_preview_presets(
    services: ServicesDep,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Return available multi-option Preview Version Stacks with dynamic sheet cost estimates."""
    sheet: ShotSheet | None = None
    if run_id:
        try:
            st = services.run_store.load(run_id)
            sheet = st.shot_sheet if st else None
        except (ValueError, OSError):
            sheet = None

    presets = []
    for p in PREVIEW_STACK_PRESETS:
        item = dict(p)
        item["estimated_usd"] = _estimate_stack_usd(sheet, p)
        presets.append(item)
    return {
        "presets": presets,
        "flagship_models": FLAGSHIP_MASTER_MODELS,
        "craft_archetypes": CRAFT_ARCHETYPES,
    }


@studio_router.post("/studio/audition")
async def audition_audio(
    payload: AuditionRequest,
    services: ServicesDep,
) -> Response:
    """Generate or replay a cached 3-4s voice or music audition clip for Step 3."""
    use_mock = payload.mock or not services.settings.fal_key.strip()
    if use_mock:
        label = payload.voice_id if payload.kind == "voice" else payload.prompt
        wav = _synthesize_mock_audition_wav(payload.kind, label=label)
        return Response(content=wav, media_type="audio/wav")

    aud_dir = services.data_dir / "cache" / "auditions"
    aud_dir.mkdir(parents=True, exist_ok=True)
    try:
        provider = AudioProvider(
            settings=services.settings,
            media_store=MediaStore(aud_dir / "media"),
            audio_cache=ContentCache(aud_dir / "cache", "auditions"),
        )
        if payload.kind == "voice":
            sample_text = (payload.text or "").strip()[:180] or (
                "Built for teams that ship with quiet precision."
            )
            cue = VoiceCue(
                shot_id="audition",
                at_sec=0.0,
                duration_sec=3.5,
                text=sample_text,
            )
            rendered = await provider.generate_voice(cue, voice_id=payload.voice_id or "Rachel")
        else:
            prompt = (payload.prompt or "").strip()[:240] or (
                "Minimal warm analog synth pulse, modern product launch film score"
            )
            cue_m = MusicCue(prompt=prompt, duration_sec=4.0)
            rendered = await provider.generate_music(cue_m)
    except Exception:
        wav = _synthesize_mock_audition_wav(payload.kind, label=payload.voice_id)
        return Response(content=wav, media_type="audio/wav")

    path = Path(rendered.path)
    if not path.is_file():
        wav = _synthesize_mock_audition_wav(payload.kind, label=payload.voice_id)
        return Response(content=wav, media_type="audio/wav")
    media_type = mimetypes.guess_type(path.name)[0] or (
        "audio/mpeg" if path.suffix.lower() == ".mp3" else "audio/wav"
    )
    return Response(content=path.read_bytes(), media_type=media_type)


@studio_router.patch("/runs/{run_id}/cap")
async def update_run_cap(
    run_id: str,
    payload: CapPatchRequest,
    actor: ActorDep,
    services: ServicesDep,
) -> dict[str, Any]:
    """Update a run's USD spend cap live."""
    run_state = load_owned_run(run_id, services.run_store, actor)
    async with services.run_lock(run_id):
        run_state = services.run_store.load(run_id) or run_state
        run_state.input.cap_usd = payload.cap_usd
        run_state.cost_ledger.cap_usd = payload.cap_usd
        services.run_store.save(run_state)
    return run_state.model_dump(mode="json")


@studio_router.post("/runs/{run_id}/step/research", status_code=202)
async def step_research(
    run_id: str,
    actor: ActorDep,
    services: ServicesDep,
) -> dict[str, Any]:
    """Step 1 -> 2: Run Agent 1 (Brand Research, 4 Concept Pitches, CreativeBible) ONLY."""
    run_state = load_owned_run(run_id, services.run_store, actor)
    async with services.run_lock(run_id):
        if services.job_registry.is_run_active(run_id):
            raise HTTPException(status_code=409, detail=f"Run {run_id!r} already has an active job")

        run_state = services.run_store.load(run_id) or run_state
        gen = services.next_generation(run_id)
        run_state.status = "working"
        run_state.error = None
        services.event_broker.reset_run(run_id)

        async def _on_event(ev: dict[str, Any]) -> None:
            if not services.is_current_generation(run_id, gen):
                return
            if ev.get("type") not in TERMINAL_EVENT_TYPES:
                services.event_broker.publish(run_id, ev)

        async def _job(cancel_token: CancellationToken) -> RunState | None:
            services.event_broker.publish(
                run_id,
                make_sse_event(
                    "stage_started",
                    run_id,
                    stage="agent1_research_bible",
                    data={"step": "research"},
                ),
            )
            try:
                run_dir = services.data_dir / "work" / run_id
                run_dir.mkdir(parents=True, exist_ok=True)
                meter = CostMeter(run_id=run_id, cap_usd=run_state.input.cap_usd)
                providers = _build_step_providers(
                    services, run_state, run_dir, meter, force_mock=run_state.input.mock
                )
                history = CreativeHistoryStore(services.data_dir / "creative_history.json")
                a1 = await run_agent1_research_bible(
                    run_input=run_state.input,
                    providers=providers,
                    history_store=history,
                    work_dir=run_dir,
                    run_id=run_id,
                    event_cb=_on_event,
                    cancel=cancel_token,
                )
                if not services.is_current_generation(run_id, gen):
                    return None
                run_state.research = a1.research
                run_state.concepts = a1.concepts
                run_state.creative_bible = a1.creative_bible
                run_state.selected_concept_id = a1.concepts[0].id if a1.concepts else None
                run_state.rubric_evaluations["agent1_research_bible"] = a1.rubric_evaluations
                run_state.step = "concept"
                run_state.status = "idle"
                services.run_store.save(run_state)
                services.event_broker.publish(
                    run_id,
                    make_sse_event(
                        "stage_completed",
                        run_id,
                        stage="agent1_research_bible",
                        data={"concepts": len(a1.concepts), "step": "concept"},
                    ),
                )
                services.event_broker.publish(
                    run_id,
                    make_sse_event(
                        "run_completed",
                        run_id,
                        stage="agent1_research_bible",
                        data={"status": "idle", "step": "concept"},
                    ),
                )
                return run_state
            except asyncio.CancelledError:
                run_state.status = "cancelled"
                services.run_store.save(run_state)
                services.event_broker.publish(
                    run_id, make_sse_event("run_cancelled", run_id, stage="agent1_research_bible")
                )
                raise
            except Exception as exc:
                run_state.status = "failed"
                run_state.error = str(exc)
                services.run_store.save(run_state)
                services.event_broker.publish(
                    run_id,
                    make_sse_event(
                        "run_failed",
                        run_id,
                        stage="agent1_research_bible",
                        data={"error": str(exc)},
                    ),
                )
                raise

        job = await services.job_registry.submit(run_id, _job, stage="research")
        services.run_store.save(run_state)
    return {"run_id": run_id, "job": job.model_dump(mode="json")}


@studio_router.patch("/runs/{run_id}/concept")
async def edit_concept_and_bible(
    run_id: str,
    payload: ConceptPatchRequest,
    actor: ActorDep,
    services: ServicesDep,
) -> dict[str, Any]:
    """Step 2 HITL: Pick a concept pitch and/or edit CreativeBible and verified brand facts."""
    run_state = load_owned_run(run_id, services.run_store, actor)
    async with services.run_lock(run_id):
        run_state = services.run_store.load(run_id) or run_state
        if run_state.creative_bible is None:
            raise HTTPException(
                status_code=400, detail="Run Step 1 (Research & Concepts) before editing concept."
            )

        if payload.creative_bible is not None:
            run_state.creative_bible = payload.creative_bible

        bible = run_state.creative_bible
        if payload.selected_concept_id is not None:
            matched = next(
                (c for c in run_state.concepts if c.id == payload.selected_concept_id), None
            )
            if matched is None:
                raise HTTPException(
                    status_code=404, detail=f"Concept {payload.selected_concept_id!r} not found"
                )
            run_state.selected_concept_id = matched.id
            if payload.concept_thesis is None:
                bible.concept_thesis = matched.thesis
            if payload.arc_name is None:
                bible.narrative_structure.arc_name = matched.narrative_arc
            if payload.world is None:
                bible.visual_language.world = matched.visual_world
            if payload.audio_strategy is None:
                bible.sonic_identity.audio_strategy = matched.sonic_strategy

        if payload.concept_thesis is not None and payload.concept_thesis.strip():
            bible.concept_thesis = payload.concept_thesis.strip()
        if payload.arc_name is not None and payload.arc_name.strip():
            bible.narrative_structure.arc_name = payload.arc_name.strip()
        if payload.world is not None and payload.world.strip():
            bible.visual_language.world = payload.world.strip()
        if payload.lighting_style is not None and payload.lighting_style.strip():
            bible.visual_language.lighting_style = payload.lighting_style.strip()
        if payload.color_palette is not None:
            bible.visual_language.color_palette = [
                c.strip() for c in payload.color_palette if c.strip()
            ]
        if payload.camera_style is not None and payload.camera_style.strip():
            bible.camera_grammar.dominant_style = payload.camera_style.strip()
        if payload.font_family is not None and payload.font_family.strip():
            bible.typography_and_layout.font_family = payload.font_family.strip()
        if payload.audio_strategy is not None and payload.audio_strategy.strip():
            bible.sonic_identity.audio_strategy = payload.audio_strategy.strip()
        if payload.anti_cliche_list is not None:
            bible.anti_cliche_list = [x.strip() for x in payload.anti_cliche_list if x.strip()]

        if payload.facts is not None:
            cleaned_facts = [f.strip() for f in payload.facts if f.strip()]
            run_state.input.facts = cleaned_facts
            if run_state.research is not None:
                for fact in cleaned_facts:
                    if fact not in run_state.research.product_facts:
                        run_state.research.product_facts.insert(0, fact)
        if payload.offer is not None:
            run_state.input.offer = payload.offer.strip() or None
        if payload.must_say is not None:
            run_state.input.must_say = [m.strip() for m in payload.must_say if m.strip()]
        if payload.user_direction is not None:
            run_state.input.user_direction = payload.user_direction.strip() or None

        run_state.step = "concept"
        services.run_store.save(run_state)
    return run_state.model_dump(mode="json")


@studio_router.post("/runs/{run_id}/step/direct", status_code=202)
async def step_direct_sheet(
    run_id: str,
    actor: ActorDep,
    services: ServicesDep,
) -> dict[str, Any]:
    """Step 2 -> 3: Run Agent 2 (Script, ShotSheet & SoundPlan Director) ONLY."""
    run_state = load_owned_run(run_id, services.run_store, actor)
    if run_state.research is None or run_state.creative_bible is None:
        raise HTTPException(
            status_code=400,
            detail="Complete Step 1 (Research & Concept) before directing the sheet.",
        )

    async with services.run_lock(run_id):
        if services.job_registry.is_run_active(run_id):
            raise HTTPException(status_code=409, detail=f"Run {run_id!r} already has an active job")

        run_state = services.run_store.load(run_id) or run_state
        gen = services.next_generation(run_id)
        run_state.status = "working"
        run_state.error = None
        services.event_broker.reset_run(run_id)

        async def _on_event(ev: dict[str, Any]) -> None:
            if not services.is_current_generation(run_id, gen):
                return
            if ev.get("type") not in TERMINAL_EVENT_TYPES:
                services.event_broker.publish(run_id, ev)

        async def _job(cancel_token: CancellationToken) -> RunState | None:
            services.event_broker.publish(
                run_id,
                make_sse_event(
                    "stage_started", run_id, stage="agent2_director", data={"step": "sheet"}
                ),
            )
            try:
                assert run_state.research is not None and run_state.creative_bible is not None
                run_dir = services.data_dir / "work" / run_id
                run_dir.mkdir(parents=True, exist_ok=True)
                meter = CostMeter(run_id=run_id, cap_usd=run_state.input.cap_usd)
                providers = _build_step_providers(
                    services, run_state, run_dir, meter, force_mock=run_state.input.mock
                )
                history = CreativeHistoryStore(services.data_dir / "creative_history.json")
                a2 = await run_agent2_director(
                    run_input=run_state.input,
                    research=run_state.research,
                    creative_bible=run_state.creative_bible,
                    providers=providers,
                    history_store=history,
                    work_dir=run_dir,
                    run_id=run_id,
                    event_cb=_on_event,
                    cancel=cancel_token,
                )
                if not services.is_current_generation(run_id, gen):
                    return None
                run_state.script = a2.script
                run_state.shot_sheet = a2.shot_sheet
                run_state.sound_plan = a2.sound_plan
                run_state.qa_report = a2.qa_report
                run_state.rubric_evaluations["agent2_director"] = a2.rubric_evaluations
                run_state.step = "sheet"
                run_state.status = "idle"
                services.run_store.save(run_state)
                services.event_broker.publish(
                    run_id,
                    make_sse_event(
                        "stage_completed",
                        run_id,
                        stage="agent2_director",
                        data={"shots": len(a2.shot_sheet.shots), "step": "sheet"},
                    ),
                )
                services.event_broker.publish(
                    run_id,
                    make_sse_event(
                        "run_completed",
                        run_id,
                        stage="agent2_director",
                        data={"status": "idle", "step": "sheet"},
                    ),
                )
                return run_state
            except asyncio.CancelledError:
                run_state.status = "cancelled"
                services.run_store.save(run_state)
                services.event_broker.publish(
                    run_id, make_sse_event("run_cancelled", run_id, stage="agent2_director")
                )
                raise
            except Exception as exc:
                run_state.status = "failed"
                run_state.error = str(exc)
                services.run_store.save(run_state)
                services.event_broker.publish(
                    run_id,
                    make_sse_event(
                        "run_failed", run_id, stage="agent2_director", data={"error": str(exc)}
                    ),
                )
                raise

        job = await services.job_registry.submit(run_id, _job, stage="direct")
        services.run_store.save(run_state)
    return {"run_id": run_id, "job": job.model_dump(mode="json")}


@studio_router.patch("/runs/{run_id}/sheet")
async def edit_shot_sheet(
    run_id: str,
    payload: SheetPatchRequest,
    actor: ActorDep,
    services: ServicesDep,
) -> dict[str, Any]:
    """Step 3 HITL: Save human shot-by-shot edits and run immediate deterministic script gate."""
    run_state = load_owned_run(run_id, services.run_store, actor)
    async with services.run_lock(run_id):
        run_state = services.run_store.load(run_id) or run_state
        if run_state.research is None or run_state.creative_bible is None:
            raise HTTPException(
                status_code=400, detail="Run Research & Concept before editing the shot sheet."
            )

        sheet = payload.shot_sheet
        for idx, shot in enumerate(sheet.shots):
            shot.index = idx

        script = payload.script or _build_script_from_sheet(sheet, run_state.research)

        sound_plan = payload.sound_plan or run_state.sound_plan or SoundPlan()
        # Keep vo_cues synchronized with human-edited shot VO lines & durations
        if sheet.audio_mode in ("vo", "both"):
            new_vo_cues: list[VoiceCue] = []
            cursor = 0.0
            for shot in sheet.shots:
                if shot.vo_line and shot.vo_line.strip():
                    new_vo_cues.append(
                        VoiceCue(
                            shot_id=shot.id,
                            at_sec=round(cursor, 3),
                            duration_sec=shot.duration_sec,
                            text=shot.vo_line.strip(),
                            delivery_note=run_state.creative_bible.sonic_identity.voice_casting.delivery_style,
                        )
                    )
                cursor += shot.duration_sec
            sound_plan.vo_cues = new_vo_cues

        gate = evaluate_script_gate(
            sheet,
            run_state.research,
            script=script,
            has_product_photos=bool(run_state.research.product_photos),
            extra_facts=run_state.input.facts,
            offer=run_state.input.offer,
            must_say=run_state.input.must_say,
            gate_override=False,
        )
        orig, coh = _score_craft_dimensions(sheet, run_state.creative_bible)
        gate.originality_score = orig
        gate.bible_coherence_score = coh

        run_state.shot_sheet = sheet
        run_state.script = script
        run_state.sound_plan = sound_plan
        run_state.qa_report = gate
        run_state.step = "sheet"
        services.run_store.save(run_state)
    return run_state.model_dump(mode="json")


@studio_router.post("/runs/{run_id}/step/previews", status_code=202)
async def step_render_previews(
    run_id: str,
    payload: PreviewsRequest,
    actor: ActorDep,
    services: ServicesDep,
) -> dict[str, Any]:
    """Step 3 -> 4: Render 1-4 preview option stacks side-by-side into run.versions."""
    run_state = load_owned_run(run_id, services.run_store, actor)
    if run_state.shot_sheet is None or run_state.creative_bible is None:
        raise HTTPException(
            status_code=400, detail="Complete Step 3 (Shot Sheet) before rendering previews."
        )

    async with services.run_lock(run_id):
        if services.job_registry.is_run_active(run_id):
            raise HTTPException(status_code=409, detail=f"Run {run_id!r} already has an active job")

        run_state = services.run_store.load(run_id) or run_state
        gen = services.next_generation(run_id)
        run_state.status = "working"
        run_state.step = "previews"
        run_state.error = None
        services.event_broker.reset_run(run_id)

        async def _on_event(ev: dict[str, Any]) -> None:
            if not services.is_current_generation(run_id, gen):
                return
            if ev.get("type") not in TERMINAL_EVENT_TYPES:
                services.event_broker.publish(run_id, ev)

        async def _job(cancel_token: CancellationToken) -> RunState | None:
            services.event_broker.publish(
                run_id,
                make_sse_event(
                    "stage_started",
                    run_id,
                    stage="previews",
                    data={
                        "count": len(payload.stacks),
                        "labels": [s.label for s in payload.stacks],
                    },
                ),
            )
            try:
                assert run_state.shot_sheet is not None and run_state.creative_bible is not None
                for idx, stack in enumerate(payload.stacks):
                    cancel_token.raise_if_cancelled()
                    ver_id = f"v_{stack.id}_{uuid4().hex[:6]}"
                    ver_dir = services.data_dir / "work" / run_id / "versions" / ver_id
                    ver_dir.mkdir(parents=True, exist_ok=True)

                    ver_sheet = (
                        _apply_hybrid_motion_cards(run_state.shot_sheet)
                        if stack.hybrid_motion_cards
                        else run_state.shot_sheet.model_copy(deep=True)
                    )
                    if stack.use_keyframes and not (stack.mock or run_state.input.mock):
                        _wire_adjacent_end_keyframes(ver_sheet)

                    ver_input = run_state.input.model_copy(deep=True)
                    ver_input.render_mode = "preview"
                    ver_input.mock = bool(stack.mock or run_state.input.mock)

                    remaining_cap = max(
                        0.0, run_state.cost_ledger.cap_usd - run_state.cost_ledger.actual_usd
                    )
                    meter = CostMeter(
                        run_id=f"{run_id}_{ver_id}",
                        cap_usd=remaining_cap
                        if not ver_input.mock
                        else run_state.cost_ledger.cap_usd,
                        ledger_path=ver_dir / "spend_ledger.jsonl",
                    )
                    providers = _build_step_providers(
                        services,
                        run_state,
                        ver_dir,
                        meter,
                        force_mock=ver_input.mock,
                        video_model_override=stack.video_model,
                        voice_model_override=stack.voice_model,
                    )

                    services.event_broker.publish(
                        run_id,
                        make_sse_event(
                            "shot_progress",
                            run_id,
                            stage="previews",
                            data={
                                "version_id": ver_id,
                                "label": stack.label,
                                "index": idx + 1,
                                "total": len(payload.stacks),
                                "status": "rendering",
                            },
                        ),
                    )

                    brand_logo, product_imgs = _resolve_brand_assets(run_state)
                    try:
                        a3 = await run_agent3_producer(
                            run_id=f"{run_id}_{ver_id}",
                            run_input=ver_input,
                            shot_sheet=ver_sheet,
                            sound_plan=run_state.sound_plan,
                            creative_bible=run_state.creative_bible,
                            providers=providers,
                            cost_meter=meter,
                            work_dir=ver_dir,
                            director_qa=run_state.qa_report,
                            event_cb=_on_event,
                            cancel=cancel_token,
                            max_iterations=1,
                            raise_on_failure=False,
                            brand_logo_path=brand_logo,
                            product_image_paths=product_imgs,
                        )
                        _merge_cost_ledger(run_state, a3.cost_ledger.entries)
                        scorecard = None
                        if a3.render_manifest.stitched_path:
                            critic = VisionQualityCritic(settings=services.settings)
                            scorecard = await critic.score_video(
                                a3.render_manifest.stitched_path,
                                ver_sheet,
                                run_state.creative_bible,
                                qa=a3.qa_report,
                                mock=ver_input.mock,
                                cancel=cancel_token,
                            )
                        ver = Version(
                            version_id=ver_id,
                            label=stack.label,
                            tier="preview",
                            status="done" if a3.render_manifest.stitched_path else "failed",
                            stack=stack.model_dump(mode="json"),
                            usd=round(a3.cost_ledger.actual_usd, 4),
                            creative_bible=run_state.creative_bible,
                            shot_sheet=ver_sheet,
                            sound_plan=run_state.sound_plan,
                            render_manifest=a3.render_manifest,
                            qa_report=a3.qa_report,
                            quality_scorecard=scorecard,
                            warnings=a3.render_manifest.warnings,
                            error=a3.render_manifest.error,
                        )
                    except asyncio.CancelledError:
                        raise
                    except Exception as ver_exc:
                        ver = Version(
                            version_id=ver_id,
                            label=stack.label,
                            tier="preview",
                            status="failed",
                            stack=stack.model_dump(mode="json"),
                            usd=round(meter.spent_usd, 4),
                            shot_sheet=ver_sheet,
                            error=str(ver_exc),
                        )

                    run_state.versions.append(ver)
                    services.run_store.save(run_state)
                    services.event_broker.publish(
                        run_id,
                        make_sse_event(
                            "cost_update",
                            run_id,
                            stage="previews",
                            data={
                                "actual_usd": run_state.cost_ledger.actual_usd,
                                "cap_usd": run_state.cost_ledger.cap_usd,
                                "version_id": ver_id,
                            },
                        ),
                    )

                if not services.is_current_generation(run_id, gen):
                    return None
                run_state.step = "previews"
                run_state.status = "idle"
                services.run_store.save(run_state)
                services.event_broker.publish(
                    run_id,
                    make_sse_event(
                        "stage_completed",
                        run_id,
                        stage="previews",
                        data={"versions": len(run_state.versions), "step": "previews"},
                    ),
                )
                services.event_broker.publish(
                    run_id,
                    make_sse_event(
                        "run_completed",
                        run_id,
                        stage="previews",
                        data={"status": "idle", "step": "previews"},
                    ),
                )
                return run_state
            except asyncio.CancelledError:
                run_state.status = "cancelled"
                services.run_store.save(run_state)
                services.event_broker.publish(
                    run_id, make_sse_event("run_cancelled", run_id, stage="previews")
                )
                raise
            except Exception as exc:
                run_state.status = "failed"
                run_state.error = str(exc)
                services.run_store.save(run_state)
                services.event_broker.publish(
                    run_id,
                    make_sse_event(
                        "run_failed", run_id, stage="previews", data={"error": str(exc)}
                    ),
                )
                raise

        job = await services.job_registry.submit(run_id, _job, stage="previews")
        services.run_store.save(run_state)
    return {"run_id": run_id, "job": job.model_dump(mode="json")}


@studio_router.get("/runs/{run_id}/versions/{version_id}/video")
async def serve_version_video(
    run_id: str,
    version_id: str,
    request: Request,
    services: ServicesDep,
) -> Response:
    """Stream a specific preview or final version MP4 with HTTP 206 Range support."""
    if not run_id or len(run_id) > MAX_RUN_ID_LENGTH:
        raise HTTPException(status_code=400, detail="Invalid run_id")
    state = services.run_store.load(run_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found")

    version = next((v for v in state.versions if v.version_id == version_id), None)
    if version is None or not version.render_manifest or not version.render_manifest.stitched_path:
        raise HTTPException(status_code=404, detail=f"Version {version_id!r} video not found")

    path = Path(version.render_manifest.stitched_path).resolve()
    try:
        path.relative_to(services.data_dir.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="Video path outside data directory") from exc
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Rendered version file missing on disk")

    file_size = path.stat().st_size
    media_type = mimetypes.guess_type(path.name)[0] or "video/mp4"
    range_header = request.headers.get("range")
    if not range_header:
        return StreamingResponse(
            _iter_file_range(path, 0, file_size),
            status_code=200,
            media_type=media_type,
            headers={
                "Accept-Ranges": "bytes",
                "Content-Length": str(file_size),
                "Content-Disposition": f'inline; filename="{run_id}_{version_id}.mp4"',
            },
        )
    parsed = _parse_byte_range(range_header, file_size)
    if parsed is None:
        return Response(
            status_code=416,
            headers={"Content-Range": f"bytes */{file_size}", "Accept-Ranges": "bytes"},
        )
    start, end = parsed
    content_length = end - start + 1
    return StreamingResponse(
        _iter_file_range(path, start, content_length),
        status_code=206,
        media_type=media_type,
        headers={
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(content_length),
        },
    )


@studio_router.post("/runs/{run_id}/versions/{version_id}/pick")
async def pick_winning_version(
    run_id: str,
    version_id: str,
    actor: ActorDep,
    services: ServicesDep,
) -> dict[str, Any]:
    """Step 4 HITL: Pick the winning preview version before rendering the Final Cut."""
    run_state = load_owned_run(run_id, services.run_store, actor)
    async with services.run_lock(run_id):
        run_state = services.run_store.load(run_id) or run_state
        matched = next((v for v in run_state.versions if v.version_id == version_id), None)
        if matched is None:
            raise HTTPException(status_code=404, detail=f"Version {version_id!r} not found")
        run_state.winning_version_id = version_id
        for v in run_state.versions:
            v.is_winner = v.version_id == version_id
        run_state.step = "final"
        services.run_store.save(run_state)
    return run_state.model_dump(mode="json")


@studio_router.post("/runs/{run_id}/step/final", status_code=202)
async def step_render_final(
    run_id: str,
    actor: ActorDep,
    services: ServicesDep,
    payload: FinalRenderRequest | None = None,
) -> dict[str, Any]:
    """Step 5: Render the winning preview stack at full Final quality into run.render_manifest."""
    run_state = load_owned_run(run_id, services.run_store, actor)
    if run_state.shot_sheet is None or run_state.creative_bible is None:
        raise HTTPException(
            status_code=400, detail="Complete Shot Sheet and Previews before rendering Final Cut."
        )

    async with services.run_lock(run_id):
        if services.job_registry.is_run_active(run_id):
            raise HTTPException(status_code=409, detail=f"Run {run_id!r} already has an active job")

        run_state = services.run_store.load(run_id) or run_state
        gen = services.next_generation(run_id)
        run_state.status = "working"
        run_state.step = "final"
        run_state.error = None
        services.event_broker.reset_run(run_id)

        chosen_id = (payload.version_id if payload else None) or run_state.winning_version_id
        winner = next((v for v in run_state.versions if v.version_id == chosen_id), None)
        if winner is None and run_state.versions:
            winner = next((v for v in reversed(run_state.versions) if v.status == "done"), None)

        async def _on_event(ev: dict[str, Any]) -> None:
            if not services.is_current_generation(run_id, gen):
                return
            if ev.get("type") not in TERMINAL_EVENT_TYPES:
                services.event_broker.publish(run_id, ev)

        async def _job(cancel_token: CancellationToken) -> RunState | None:
            services.event_broker.publish(
                run_id,
                make_sse_event(
                    "stage_started",
                    run_id,
                    stage="final",
                    data={"winning_version_id": winner.version_id if winner else None},
                ),
            )
            try:
                assert run_state.shot_sheet is not None and run_state.creative_bible is not None
                final_dir = services.data_dir / "work" / run_id / "final"
                final_dir.mkdir(parents=True, exist_ok=True)

                stack = dict(winner.stack) if winner else {}
                final_sheet = (
                    winner.shot_sheet.model_copy(deep=True)
                    if winner and winner.shot_sheet
                    else run_state.shot_sheet.model_copy(deep=True)
                )
                final_input = run_state.input.model_copy(deep=True)
                final_input.render_mode = "final"
                if payload and payload.mock is not None:
                    final_input.mock = payload.mock
                elif stack.get("mock") is not None:
                    final_input.mock = bool(run_state.input.mock or stack.get("mock"))

                chosen_video_model = (
                    payload.video_model_override.strip()
                    if payload
                    and payload.video_model_override
                    and payload.video_model_override.strip()
                    else stack.get("video_model")
                )
                if chosen_video_model:
                    stack["video_model"] = chosen_video_model

                if not final_input.mock:
                    _wire_adjacent_end_keyframes(final_sheet)

                remaining_cap = max(
                    0.0, run_state.cost_ledger.cap_usd - run_state.cost_ledger.actual_usd
                )
                meter = CostMeter(
                    run_id=f"{run_id}_final",
                    cap_usd=remaining_cap
                    if not final_input.mock
                    else run_state.cost_ledger.cap_usd,
                    ledger_path=final_dir / "spend_ledger.jsonl",
                )
                providers = _build_step_providers(
                    services,
                    run_state,
                    final_dir,
                    meter,
                    force_mock=final_input.mock,
                    video_model_override=chosen_video_model,
                    voice_model_override=stack.get("voice_model"),
                )

                brand_logo, product_imgs = _resolve_brand_assets(run_state)
                a3 = await run_agent3_producer(
                    run_id=run_id,
                    run_input=final_input,
                    shot_sheet=final_sheet,
                    sound_plan=run_state.sound_plan,
                    creative_bible=run_state.creative_bible,
                    providers=providers,
                    cost_meter=meter,
                    work_dir=final_dir,
                    director_qa=run_state.qa_report,
                    event_cb=_on_event,
                    cancel=cancel_token,
                    max_iterations=2,
                    raise_on_failure=False,
                    brand_logo_path=brand_logo,
                    product_image_paths=product_imgs,
                )
                if not services.is_current_generation(run_id, gen):
                    return None

                _merge_cost_ledger(run_state, a3.cost_ledger.entries)
                final_scorecard = None
                if a3.render_manifest.stitched_path:
                    critic = VisionQualityCritic(settings=services.settings)
                    final_scorecard = await critic.score_video(
                        a3.render_manifest.stitched_path,
                        final_sheet,
                        run_state.creative_bible,
                        qa=a3.qa_report,
                        mock=final_input.mock,
                        cancel=cancel_token,
                    )
                run_state.render_manifest = a3.render_manifest
                run_state.qa_report = a3.qa_report
                run_state.quality_scorecard = final_scorecard
                run_state.rubric_evaluations["agent3_producer"] = a3.rubric_evaluations
                model_tag = (
                    f" · {chosen_video_model}" if payload and payload.video_model_override else ""
                )
                run_state.versions.append(
                    Version(
                        version_id=f"final_{uuid4().hex[:6]}",
                        label=f"Final Cut ({winner.label if winner else 'Default'}{model_tag})",
                        tier="final",
                        status="done" if a3.render_manifest.stitched_path else "failed",
                        stack=stack,
                        usd=round(a3.cost_ledger.actual_usd, 4),
                        is_winner=True,
                        creative_bible=run_state.creative_bible,
                        shot_sheet=final_sheet,
                        sound_plan=run_state.sound_plan,
                        render_manifest=a3.render_manifest,
                        qa_report=a3.qa_report,
                        quality_scorecard=final_scorecard,
                    )
                )
                run_state.step = "final"
                run_state.status = "done"
                services.run_store.save(run_state)
                services.event_broker.publish(
                    run_id,
                    make_sse_event(
                        "stage_completed",
                        run_id,
                        stage="final",
                        data={
                            "stitched_path": a3.render_manifest.stitched_path,
                            "overall_score": (
                                final_scorecard.overall_score if final_scorecard else None
                            ),
                        },
                    ),
                )
                services.event_broker.publish(
                    run_id,
                    make_sse_event(
                        "run_completed",
                        run_id,
                        stage="final",
                        data={"status": "done", "step": "final"},
                    ),
                )
                return run_state
            except asyncio.CancelledError:
                run_state.status = "cancelled"
                services.run_store.save(run_state)
                services.event_broker.publish(
                    run_id, make_sse_event("run_cancelled", run_id, stage="final")
                )
                raise
            except Exception as exc:
                run_state.status = "failed"
                run_state.error = str(exc)
                services.run_store.save(run_state)
                services.event_broker.publish(
                    run_id,
                    make_sse_event("run_failed", run_id, stage="final", data={"error": str(exc)}),
                )
                raise

        job = await services.job_registry.submit(run_id, _job, stage="final")
        services.run_store.save(run_state)
    return {"run_id": run_id, "job": job.model_dump(mode="json")}


@studio_router.post("/runs/{run_id}/step/storyboards", status_code=202)
async def step_generate_storyboards(
    run_id: str,
    actor: ActorDep,
    services: ServicesDep,
) -> dict[str, Any]:
    """Step 3 -> 4: Generate low-cost visual keyframes ($0.005/still) for every shot."""
    run_state = load_owned_run(run_id, services.run_store, actor)
    if run_state.shot_sheet is None or run_state.creative_bible is None:
        raise HTTPException(
            status_code=400,
            detail="Complete Step 3 (Shot Sheet) before generating visual storyboards.",
        )

    async with services.run_lock(run_id):
        if services.job_registry.is_run_active(run_id):
            raise HTTPException(status_code=409, detail=f"Run {run_id!r} already has an active job")

        run_state = services.run_store.load(run_id) or run_state
        gen = services.next_generation(run_id)
        run_state.status = "working"
        run_state.error = None
        services.event_broker.reset_run(run_id)

        async def _job(cancel_token: CancellationToken) -> RunState | None:
            services.event_broker.publish(
                run_id,
                make_sse_event(
                    "stage_started", run_id, stage="storyboard", data={"step": "storyboard"}
                ),
            )
            try:
                assert run_state.shot_sheet is not None and run_state.creative_bible is not None
                kf_dir = services.data_dir / "work" / run_id / "keyframes"
                remaining = max(
                    0.0, run_state.cost_ledger.cap_usd - run_state.cost_ledger.actual_usd
                )
                meter = CostMeter(
                    run_id=f"{run_id}_kf",
                    cap_usd=remaining
                    if not run_state.input.mock
                    else run_state.cost_ledger.cap_usd,
                )
                generator = StoryboardGenerator(settings=services.settings, cost_meter=meter)
                brand_logo, product_imgs = _resolve_brand_assets(run_state)
                results = await generator.generate_all_keyframes(
                    run_state.shot_sheet,
                    run_state.creative_bible,
                    kf_dir,
                    mock=run_state.input.mock,
                    logo_path=brand_logo,
                    product_images=product_imgs,
                    cancel=cancel_token,
                )
                if not services.is_current_generation(run_id, gen):
                    return None
                by_id = {sid: (p, url) for sid, p, url, _ in results}
                for shot in run_state.shot_sheet.shots:
                    if shot.id in by_id:
                        p, url = by_id[shot.id]
                        shot.keyframe_path = str(p)
                        shot.keyframe_url = url
                        shot.keyframe_approved = True
                        if url and shot.engine == "ai_video":
                            shot.reference_image = url

                _merge_cost_ledger(run_state, meter.to_ledger().entries)
                run_state.step = "storyboard"
                run_state.status = "idle"
                services.run_store.save(run_state)
                services.event_broker.publish(
                    run_id,
                    make_sse_event(
                        "stage_completed",
                        run_id,
                        stage="storyboard",
                        data={"keyframes": len(results), "step": "storyboard"},
                    ),
                )
                services.event_broker.publish(
                    run_id,
                    make_sse_event(
                        "run_completed",
                        run_id,
                        stage="storyboard",
                        data={"status": "idle", "step": "storyboard"},
                    ),
                )
                return run_state
            except asyncio.CancelledError:
                run_state.status = "cancelled"
                services.run_store.save(run_state)
                services.event_broker.publish(
                    run_id, make_sse_event("run_cancelled", run_id, stage="storyboard")
                )
                raise
            except Exception as exc:
                run_state.status = "failed"
                run_state.error = str(exc)
                services.run_store.save(run_state)
                services.event_broker.publish(
                    run_id,
                    make_sse_event(
                        "run_failed", run_id, stage="storyboard", data={"error": str(exc)}
                    ),
                )
                raise

        job = await services.job_registry.submit(run_id, _job, stage="storyboard")
        services.run_store.save(run_state)
    return {"run_id": run_id, "job": job.model_dump(mode="json")}


@studio_router.post("/runs/{run_id}/shots/{shot_id}/keyframe")
async def regenerate_shot_keyframe(
    run_id: str,
    shot_id: str,
    payload: KeyframeRegenRequest,
    actor: ActorDep,
    services: ServicesDep,
) -> dict[str, Any]:
    """HITL Step 4: Regenerate or approve/unapprove a single shot's visual keyframe."""
    run_state = load_owned_run(run_id, services.run_store, actor)
    async with services.run_lock(run_id):
        run_state = services.run_store.load(run_id) or run_state
        if run_state.shot_sheet is None or run_state.creative_bible is None:
            raise HTTPException(status_code=400, detail="Run has no shot_sheet yet.")
        shot = next((s for s in run_state.shot_sheet.shots if s.id == shot_id), None)
        if shot is None:
            raise HTTPException(status_code=404, detail=f"Shot {shot_id!r} not found")

        if payload.visual_prompt is not None and payload.visual_prompt.strip():
            shot.visual_prompt = payload.visual_prompt.strip()

        if payload.regenerate:
            kf_dir = services.data_dir / "work" / run_id / "keyframes"
            meter = CostMeter(
                run_id=f"{run_id}_kf_{shot_id}",
                cap_usd=max(0.0, run_state.cost_ledger.cap_usd - run_state.cost_ledger.actual_usd)
                if not run_state.input.mock
                else run_state.cost_ledger.cap_usd,
            )
            gen = StoryboardGenerator(settings=services.settings, cost_meter=meter)
            brand_logo, product_imgs = _resolve_brand_assets(run_state)
            first_ai = next(
                (s for s in run_state.shot_sheet.shots if s.engine != "motion_card"), None
            )
            anchor_url = (
                first_ai.keyframe_url
                if first_ai is not None and first_ai.id != shot.id and first_ai.keyframe_url
                else None
            )
            media_img = product_imgs[shot.index % len(product_imgs)] if product_imgs else None
            p, url, _ = await gen.generate_shot_keyframe(
                shot,
                run_state.shot_sheet,
                run_state.creative_bible,
                kf_dir,
                mock=run_state.input.mock,
                anchor_image_url=anchor_url,
                logo_path=brand_logo,
                media_image_path=media_img,
            )
            shot.keyframe_path = str(p)
            shot.keyframe_url = url
            if url:
                shot.reference_image = url
            _merge_cost_ledger(run_state, meter.to_ledger().entries)

        if payload.approved is not None:
            shot.keyframe_approved = payload.approved
        elif payload.regenerate:
            shot.keyframe_approved = True

        services.run_store.save(run_state)
    return run_state.model_dump(mode="json")


@studio_router.get("/runs/{run_id}/shots/{shot_id}/keyframe")
async def serve_shot_keyframe(
    run_id: str,
    shot_id: str,
    services: ServicesDep,
) -> Response:
    """Serve the generated keyframe JPEG for a shot."""
    if not run_id or len(run_id) > MAX_RUN_ID_LENGTH:
        raise HTTPException(status_code=400, detail="Invalid run_id")
    state = services.run_store.load(run_id)
    if state is None or state.shot_sheet is None:
        raise HTTPException(status_code=404, detail="Run or shot sheet not found")
    shot = next((s for s in state.shot_sheet.shots if s.id == shot_id), None)
    if shot is None or not shot.keyframe_path:
        raise HTTPException(status_code=404, detail=f"Keyframe for {shot_id!r} not found")
    path = Path(shot.keyframe_path).resolve()
    try:
        path.relative_to(services.data_dir.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="Keyframe path outside data directory") from exc
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Keyframe file missing on disk")
    return Response(content=path.read_bytes(), media_type="image/jpeg")


@studio_router.post("/runs/{run_id}/score")
async def submit_human_score(
    run_id: str,
    payload: HumanScore,
    actor: ActorDep,
    services: ServicesDep,
) -> dict[str, Any]:
    """Capture human 1-5 star rating + critique tags and feed brand taste into CreativeHistory."""
    run_state = load_owned_run(run_id, services.run_store, actor)
    async with services.run_lock(run_id):
        run_state = services.run_store.load(run_id) or run_state
        run_state.human_scores.append(payload)
        if payload.target_version_id:
            ver = next(
                (v for v in run_state.versions if v.version_id == payload.target_version_id),
                None,
            )
            if ver is not None:
                ver.human_score = payload

        # Persist human approval/rejection signal into per-brand CreativeHistory taste memory
        if run_state.creative_bible is not None:
            history = CreativeHistoryStore(services.data_dir / "creative_history.json")
            first_hook = (
                run_state.shot_sheet.shots[0].vo_line
                if run_state.shot_sheet and run_state.shot_sheet.shots
                else run_state.creative_bible.concept_thesis[:80]
            )
            fp = CreativeFingerprint(
                run_id=run_id,
                brand_url=run_state.input.brand_url,
                arc_name=run_state.creative_bible.narrative_structure.arc_name,
                hook_opening=first_hook or run_state.creative_bible.concept_thesis[:80],
                visual_world=run_state.creative_bible.visual_language.world,
                lighting_style=run_state.creative_bible.visual_language.lighting_style,
                camera_style=run_state.creative_bible.camera_grammar.dominant_style,
                voice_id=run_state.sound_plan.voice_id if run_state.sound_plan else None,
                outcome="approved" if payload.rating_overall >= 4 else "rejected",
            )
            history.record(fp)

        services.run_store.save(run_state)
    return run_state.model_dump(mode="json")


@studio_router.post("/runs/{run_id}/reiterate", status_code=202)
async def reiterate_weak_shots(
    run_id: str,
    payload: ReiterateRequest,
    actor: ActorDep,
    services: ServicesDep,
) -> dict[str, Any]:
    """Targeted Quality Reiteration: refine & re-render ONLY weak/flagged shots, then re-score."""
    run_state = load_owned_run(run_id, services.run_store, actor)
    if run_state.shot_sheet is None or run_state.creative_bible is None:
        raise HTTPException(status_code=400, detail="Run has no shot_sheet to reiterate.")

    async with services.run_lock(run_id):
        if services.job_registry.is_run_active(run_id):
            raise HTTPException(status_code=409, detail=f"Run {run_id!r} already has an active job")

        run_state = services.run_store.load(run_id) or run_state
        gen = services.next_generation(run_id)
        run_state.status = "working"
        run_state.error = None
        services.event_broker.reset_run(run_id)

        target_ver = (
            next((v for v in run_state.versions if v.version_id == payload.version_id), None)
            if payload.version_id
            else (next((v for v in reversed(run_state.versions) if v.status == "done"), None))
        )
        scorecard = (
            target_ver.quality_scorecard
            if target_ver and target_ver.quality_scorecard
            else run_state.quality_scorecard
        )

        # Pick explicit shot_ids or auto-select weak shots from Vision Critic / Media QA
        chosen_ids = list(payload.shot_ids or [])
        if not chosen_ids and scorecard:
            chosen_ids = [
                sc.shot_id
                for sc in scorecard.shot_scores
                if sc.verdict == "reiterate" or sc.score < 8.0
            ]
        if not chosen_ids and run_state.qa_report and run_state.qa_report.failed_shot_ids:
            chosen_ids = list(run_state.qa_report.failed_shot_ids)
        if not chosen_ids and run_state.shot_sheet.shots:
            chosen_ids = [run_state.shot_sheet.shots[0].id]

        async def _on_event(ev: dict[str, Any]) -> None:
            if not services.is_current_generation(run_id, gen):
                return
            if ev.get("type") not in TERMINAL_EVENT_TYPES:
                services.event_broker.publish(run_id, ev)

        async def _job(cancel_token: CancellationToken) -> RunState | None:
            services.event_broker.publish(
                run_id,
                make_sse_event(
                    "stage_started",
                    run_id,
                    stage="reiterate",
                    data={"shot_ids": chosen_ids, "notes": payload.notes},
                ),
            )
            try:
                assert run_state.shot_sheet is not None and run_state.creative_bible is not None
                base_sheet = (
                    target_ver.shot_sheet.model_copy(deep=True)
                    if target_ver and target_ver.shot_sheet
                    else run_state.shot_sheet.model_copy(deep=True)
                )
                fixes = (
                    {sc.shot_id: sc.suggested_prompt_fix for sc in scorecard.shot_scores}
                    if scorecard
                    else {}
                )
                for sh in base_sheet.shots:
                    if sh.id in chosen_ids:
                        if fixes.get(sh.id):
                            sh.visual_prompt = fixes[sh.id]
                        if payload.notes.strip():
                            sh.visual_prompt = (
                                f"{sh.visual_prompt.strip()}; Director revision: "
                                f"{payload.notes.strip()}"
                            )

                iter_id = f"iter_{uuid4().hex[:6]}"
                iter_dir = services.data_dir / "work" / run_id / "versions" / iter_id
                iter_dir.mkdir(parents=True, exist_ok=True)

                stack = target_ver.stack if target_ver else {}
                iter_input = run_state.input.model_copy(deep=True)
                if stack.get("mock") is not None:
                    iter_input.mock = bool(run_state.input.mock or stack.get("mock"))

                remaining = max(
                    0.0, run_state.cost_ledger.cap_usd - run_state.cost_ledger.actual_usd
                )
                meter = CostMeter(
                    run_id=f"{run_id}_{iter_id}",
                    cap_usd=remaining if not iter_input.mock else run_state.cost_ledger.cap_usd,
                    ledger_path=iter_dir / "spend_ledger.jsonl",
                )

                brand_logo, product_imgs = _resolve_brand_assets(run_state)
                first_ai = next((s for s in base_sheet.shots if s.engine != "motion_card"), None)

                # Optionally regenerate keyframes for reiterated shots first (anchored to Shot 1)
                if payload.regenerate_keyframes:
                    kf_gen = StoryboardGenerator(settings=services.settings, cost_meter=meter)
                    kf_dir = services.data_dir / "work" / run_id / "keyframes"
                    for sh in base_sheet.shots:
                        if sh.id in chosen_ids:
                            anchor_url = (
                                first_ai.keyframe_url
                                if first_ai is not None
                                and first_ai.id != sh.id
                                and first_ai.keyframe_url
                                else None
                            )
                            media_img = (
                                product_imgs[sh.index % len(product_imgs)] if product_imgs else None
                            )
                            p, url, _ = await kf_gen.generate_shot_keyframe(
                                sh,
                                base_sheet,
                                run_state.creative_bible,
                                kf_dir,
                                mock=iter_input.mock,
                                anchor_image_url=anchor_url,
                                logo_path=brand_logo,
                                media_image_path=media_img,
                                cancel=cancel_token,
                            )
                            sh.keyframe_path = str(p)
                            sh.keyframe_url = url
                            if url and not iter_input.mock and sh.engine == "ai_video":
                                sh.engine = "image_to_video"
                                sh.reference_image = url

                # Reuse untouched rendered MP4 clips from target version ($0 & 0s for kept shots!)
                existing_clips = (
                    {c.shot_id: c for c in target_ver.render_manifest.clips}
                    if target_ver
                    and target_ver.render_manifest
                    and target_ver.render_manifest.clips
                    else None
                )

                providers = _build_step_providers(
                    services,
                    run_state,
                    iter_dir,
                    meter,
                    force_mock=iter_input.mock,
                    video_model_override=stack.get("video_model"),
                    voice_model_override=stack.get("voice_model"),
                )
                a3 = await run_agent3_producer(
                    run_id=f"{run_id}_{iter_id}",
                    run_input=iter_input,
                    shot_sheet=base_sheet,
                    sound_plan=run_state.sound_plan,
                    creative_bible=run_state.creative_bible,
                    providers=providers,
                    cost_meter=meter,
                    work_dir=iter_dir,
                    director_qa=run_state.qa_report,
                    event_cb=_on_event,
                    cancel=cancel_token,
                    max_iterations=1,
                    raise_on_failure=False,
                    existing_clips=existing_clips,
                    repair_shot_ids=set(chosen_ids),
                    brand_logo_path=brand_logo,
                    product_image_paths=product_imgs,
                )
                if not services.is_current_generation(run_id, gen):
                    return None

                _merge_cost_ledger(run_state, a3.cost_ledger.entries)
                new_scorecard = None
                if a3.render_manifest.stitched_path:
                    critic = VisionQualityCritic(settings=services.settings)
                    new_scorecard = await critic.score_video(
                        a3.render_manifest.stitched_path,
                        base_sheet,
                        run_state.creative_bible,
                        qa=a3.qa_report,
                        mock=iter_input.mock,
                        cancel=cancel_token,
                    )

                run_state.shot_sheet = base_sheet
                run_state.versions.append(
                    Version(
                        version_id=iter_id,
                        label=f"Reiterated ({', '.join(chosen_ids)})",
                        tier="preview",
                        status="done" if a3.render_manifest.stitched_path else "failed",
                        stack=stack,
                        usd=round(a3.cost_ledger.actual_usd, 4),
                        creative_bible=run_state.creative_bible,
                        shot_sheet=base_sheet,
                        sound_plan=run_state.sound_plan,
                        render_manifest=a3.render_manifest,
                        qa_report=a3.qa_report,
                        quality_scorecard=new_scorecard,
                    )
                )
                run_state.quality_scorecard = new_scorecard
                run_state.status = "idle"
                services.run_store.save(run_state)
                services.event_broker.publish(
                    run_id,
                    make_sse_event(
                        "stage_completed",
                        run_id,
                        stage="reiterate",
                        data={
                            "version_id": iter_id,
                            "shot_ids": chosen_ids,
                            "overall_score": (
                                new_scorecard.overall_score if new_scorecard else None
                            ),
                        },
                    ),
                )
                services.event_broker.publish(
                    run_id,
                    make_sse_event(
                        "run_completed",
                        run_id,
                        stage="reiterate",
                        data={"status": "idle", "version_id": iter_id},
                    ),
                )
                return run_state
            except asyncio.CancelledError:
                run_state.status = "cancelled"
                services.run_store.save(run_state)
                services.event_broker.publish(
                    run_id, make_sse_event("run_cancelled", run_id, stage="reiterate")
                )
                raise
            except Exception as exc:
                run_state.status = "failed"
                run_state.error = str(exc)
                services.run_store.save(run_state)
                services.event_broker.publish(
                    run_id,
                    make_sse_event(
                        "run_failed", run_id, stage="reiterate", data={"error": str(exc)}
                    ),
                )
                raise

        job = await services.job_registry.submit(run_id, _job, stage="reiterate")
        services.run_store.save(run_state)
    return {"run_id": run_id, "shot_ids": chosen_ids, "job": job.model_dump(mode="json")}


_ALLOWED_IMG_MIMES: dict[str, str] = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/webp": ".webp",
}
_DATA_URL_RE = re.compile(r"^data:(image/[a-zA-Z0-9.+-]+);base64,(.+)$", re.DOTALL)
_MAX_ASSET_BYTES = 8 * 1024 * 1024


@studio_router.post("/runs/{run_id}/assets")
async def upload_brand_asset(
    run_id: str,
    payload: AssetUploadRequest,
    actor: ActorDep,
    services: ServicesDep,
) -> dict[str, Any]:
    """Upload a real brand logo or product/UI image (base64 data URL) for a run."""
    run_state = load_owned_run(run_id, services.run_store, actor)
    match = _DATA_URL_RE.match(payload.data_url.strip())
    if not match:
        raise HTTPException(
            status_code=400, detail="Asset must be a valid base64 image data URL (PNG/JPG/WebP)."
        )
    mime = match.group(1).lower()
    ext = _ALLOWED_IMG_MIMES.get(mime)
    if ext is None:
        raise HTTPException(
            status_code=400, detail=f"Unsupported image type {mime!r}; use PNG, JPEG, or WebP."
        )
    try:
        raw = base64.b64decode(match.group(2), validate=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid base64 image payload.") from exc
    if not raw or len(raw) > _MAX_ASSET_BYTES:
        raise HTTPException(status_code=400, detail="Image must be between 1 byte and 8 MB.")

    safe_stem = re.sub(r"[^a-zA-Z0-9_-]+", "_", Path(payload.filename).stem)[:40] or payload.kind
    out_name = f"{payload.kind}_{safe_stem}_{uuid4().hex[:6]}{ext}"

    async with services.run_lock(run_id):
        run_state = services.run_store.load(run_id) or run_state
        assets_dir = services.data_dir / "work" / run_id / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)
        out_path = (assets_dir / out_name).resolve()
        out_path.write_bytes(raw)

        ref = MediaRef(
            id=out_name,
            url=f"/api/runs/{run_id}/assets/{out_name}",
            path=str(out_path),
            mime_type=mime,
            alt=payload.alt or payload.filename,
            source=f"uploaded_{payload.kind}",
        )
        if payload.kind == "logo":
            run_state.brand_logo_path = str(out_path)
        else:
            run_state.uploaded_assets.append(ref)
            if run_state.research is not None:
                run_state.research.product_photos.insert(0, ref)

        services.run_store.save(run_state)
    return {
        "run_id": run_id,
        "asset": ref.model_dump(mode="json"),
        "run": run_state.model_dump(mode="json"),
    }


@studio_router.get("/runs/{run_id}/assets/{filename}")
async def serve_brand_asset(
    run_id: str,
    filename: str,
    services: ServicesDep,
) -> Response:
    """Serve an uploaded brand logo or product/UI image safely."""
    if not run_id or len(run_id) > MAX_RUN_ID_LENGTH or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid run_id or filename")
    assets_dir = (services.data_dir / "work" / run_id / "assets").resolve()
    target = (assets_dir / filename).resolve()
    try:
        target.relative_to(assets_dir)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="Asset path traversal blocked") from exc
    if not target.is_file():
        raise HTTPException(status_code=404, detail="Asset not found")
    media_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
    return Response(content=target.read_bytes(), media_type=media_type)


@studio_router.post("/runs/{run_id}/reference")
async def apply_reference_taste(
    run_id: str,
    payload: ReferenceTasteRequest,
    actor: ActorDep,
    services: ServicesDep,
) -> dict[str, Any]:
    """Upgrade 5: Apply a Craft Archetype preset and/or Gemini-analyzed moodboard taste brief."""
    run_state = load_owned_run(run_id, services.run_store, actor)

    archetype_key = (payload.craft_archetype or "").strip()
    if archetype_key and archetype_key not in CRAFT_ARCHETYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown craft_archetype {archetype_key!r}; valid: {sorted(CRAFT_ARCHETYPES)}",
        )

    taste_parts: list[str] = []
    if archetype_key:
        arch = CRAFT_ARCHETYPES[archetype_key]
        taste_parts.append(
            f"Craft Archetype ({arch['label']} — {arch['references']}): {arch['brief']}"
        )

    moodboard_notes = ""
    if payload.reference_image_data_url and payload.reference_image_data_url.strip():
        data_url = payload.reference_image_data_url.strip()
        if not _DATA_URL_RE.match(data_url):
            raise HTTPException(
                status_code=400,
                detail="Reference moodboard must be a valid base64 image data URL.",
            )
        moodboard_notes = await analyze_reference_moodboard(
            data_url,
            settings=services.settings,
            mock=run_state.input.mock,
        )
        taste_parts.append(moodboard_notes)

    if payload.notes and payload.notes.strip():
        taste_parts.append(payload.notes.strip())

    combined_taste = "\n".join(taste_parts).strip()

    async with services.run_lock(run_id):
        run_state = services.run_store.load(run_id) or run_state
        if archetype_key:
            run_state.craft_archetype = archetype_key
        if combined_taste:
            run_state.reference_taste_notes = combined_taste
            existing_dir = (run_state.input.user_direction or "").strip()
            if combined_taste not in existing_dir:
                run_state.input.user_direction = (
                    f"{existing_dir}\n\n{combined_taste}".strip()
                    if existing_dir
                    else combined_taste
                )
        services.run_store.save(run_state)

    return {
        "run_id": run_id,
        "craft_archetype": run_state.craft_archetype,
        "reference_taste_notes": run_state.reference_taste_notes,
        "run": run_state.model_dump(mode="json"),
    }


@studio_router.post("/runs/{run_id}/auto-polish", status_code=202)
async def auto_polish_run(
    run_id: str,
    payload: AutoPolishRequest,
    actor: ActorDep,
    services: ServicesDep,
) -> dict[str, Any]:
    """Upgrade 5: Autonomous multi-round polish loop targeting overall_score >= target_score."""
    run_state = load_owned_run(run_id, services.run_store, actor)
    if run_state.shot_sheet is None or run_state.creative_bible is None:
        raise HTTPException(status_code=400, detail="Run has no shot_sheet to auto-polish.")

    async with services.run_lock(run_id):
        if services.job_registry.is_run_active(run_id):
            raise HTTPException(status_code=409, detail=f"Run {run_id!r} already has an active job")

        run_state = services.run_store.load(run_id) or run_state
        gen = services.next_generation(run_id)
        run_state.status = "working"
        run_state.error = None
        services.event_broker.reset_run(run_id)

        done_previews = [v for v in run_state.versions if v.status == "done"]
        start_ver = (
            next((v for v in run_state.versions if v.version_id == payload.version_id), None)
            if payload.version_id
            else (done_previews[-1] if done_previews else None)
        )
        if start_ver is None:
            raise HTTPException(
                status_code=400,
                detail="Render at least one preview before launching Auto-Polish.",
            )

        async def _on_event(ev: dict[str, Any]) -> None:
            if not services.is_current_generation(run_id, gen):
                return
            if ev.get("type") not in TERMINAL_EVENT_TYPES:
                services.event_broker.publish(run_id, ev)

        async def _job(cancel_token: CancellationToken) -> RunState | None:
            services.event_broker.publish(
                run_id,
                make_sse_event(
                    "stage_started",
                    run_id,
                    stage="auto_polish",
                    data={
                        "target_score": payload.target_score,
                        "max_rounds": payload.max_rounds,
                        "start_version_id": start_ver.version_id,
                    },
                ),
            )
            try:
                assert run_state.shot_sheet is not None and run_state.creative_bible is not None
                current_ver = start_ver
                best_ver = start_ver
                best_score = (
                    float(start_ver.quality_scorecard.overall_score)
                    if start_ver.quality_scorecard
                    else 0.0
                )
                brand_logo, product_imgs = _resolve_brand_assets(run_state)
                rounds_completed = 0

                for round_idx in range(1, payload.max_rounds + 1):
                    cancel_token.raise_if_cancelled()
                    if best_score >= payload.target_score:
                        break

                    stack = dict(current_ver.stack or {})
                    iter_input = run_state.input.model_copy(deep=True)
                    if stack.get("mock") is not None:
                        iter_input.mock = bool(run_state.input.mock or stack.get("mock"))

                    remaining = max(
                        0.0, run_state.cost_ledger.cap_usd - run_state.cost_ledger.actual_usd
                    )
                    if not iter_input.mock and remaining < 0.25:
                        services.event_broker.publish(
                            run_id,
                            make_sse_event(
                                "warning",
                                run_id,
                                stage="auto_polish",
                                data={
                                    "message": f"Stopped before round {round_idx}: USD cap reached."
                                },
                            ),
                        )
                        break

                    base_sheet = (
                        current_ver.shot_sheet.model_copy(deep=True)
                        if current_ver.shot_sheet
                        else run_state.shot_sheet.model_copy(deep=True)
                    )
                    sc = current_ver.quality_scorecard
                    weak_ids: list[str] = []
                    fixes: dict[str, str] = {}
                    if sc and sc.shot_scores:
                        fixes = {s.shot_id: s.suggested_prompt_fix for s in sc.shot_scores}
                        weak_ids = [
                            s.shot_id
                            for s in sc.shot_scores
                            if s.verdict == "reiterate" or float(s.score) < payload.target_score
                        ]
                        if not weak_ids:
                            lowest = min(sc.shot_scores, key=lambda s: float(s.score))
                            weak_ids = [lowest.shot_id]
                    if not weak_ids and base_sheet.shots:
                        weak_ids = [base_sheet.shots[0].id]

                    for sh in base_sheet.shots:
                        if sh.id in weak_ids:
                            if fixes.get(sh.id):
                                sh.visual_prompt = fixes[sh.id]
                            else:
                                sh.visual_prompt = (
                                    f"{sh.visual_prompt.strip()}; elevate lighting contrast, "
                                    "crisp framing, and deliberate camera motion."
                                )
                            if payload.notes.strip():
                                sh.visual_prompt = (
                                    f"{sh.visual_prompt.strip()}; {payload.notes.strip()}"
                                )

                    iter_id = f"polish_r{round_idx}_{uuid4().hex[:6]}"
                    iter_dir = services.data_dir / "work" / run_id / "versions" / iter_id
                    iter_dir.mkdir(parents=True, exist_ok=True)
                    meter = CostMeter(
                        run_id=f"{run_id}_{iter_id}",
                        cap_usd=remaining if not iter_input.mock else run_state.cost_ledger.cap_usd,
                        ledger_path=iter_dir / "spend_ledger.jsonl",
                    )

                    services.event_broker.publish(
                        run_id,
                        make_sse_event(
                            "shot_progress",
                            run_id,
                            stage="auto_polish",
                            data={
                                "label": f"Auto-Polish Round {round_idx}/{payload.max_rounds} ({', '.join(weak_ids)})",
                                "round": round_idx,
                                "shot_ids": weak_ids,
                            },
                        ),
                    )

                    # Regenerate keyframes for polished shots anchored to Shot 1
                    first_ai = next(
                        (s for s in base_sheet.shots if s.engine != "motion_card"), None
                    )
                    kf_gen = StoryboardGenerator(settings=services.settings, cost_meter=meter)
                    kf_dir = services.data_dir / "work" / run_id / "keyframes"
                    for sh in base_sheet.shots:
                        if sh.id in weak_ids:
                            anchor_url = (
                                first_ai.keyframe_url
                                if first_ai is not None
                                and first_ai.id != sh.id
                                and first_ai.keyframe_url
                                else None
                            )
                            media_img = (
                                product_imgs[sh.index % len(product_imgs)] if product_imgs else None
                            )
                            p, url, _ = await kf_gen.generate_shot_keyframe(
                                sh,
                                base_sheet,
                                run_state.creative_bible,
                                kf_dir,
                                mock=iter_input.mock,
                                anchor_image_url=anchor_url,
                                logo_path=brand_logo,
                                media_image_path=media_img,
                                cancel=cancel_token,
                            )
                            sh.keyframe_path = str(p)
                            sh.keyframe_url = url
                    if not iter_input.mock:
                        _wire_adjacent_end_keyframes(base_sheet)

                    existing_clips = (
                        {c.shot_id: c for c in current_ver.render_manifest.clips}
                        if current_ver.render_manifest and current_ver.render_manifest.clips
                        else None
                    )
                    providers = _build_step_providers(
                        services,
                        run_state,
                        iter_dir,
                        meter,
                        force_mock=iter_input.mock,
                        video_model_override=stack.get("video_model"),
                        voice_model_override=stack.get("voice_model"),
                    )
                    a3 = await run_agent3_producer(
                        run_id=f"{run_id}_{iter_id}",
                        run_input=iter_input,
                        shot_sheet=base_sheet,
                        sound_plan=run_state.sound_plan,
                        creative_bible=run_state.creative_bible,
                        providers=providers,
                        cost_meter=meter,
                        work_dir=iter_dir,
                        director_qa=run_state.qa_report,
                        event_cb=_on_event,
                        cancel=cancel_token,
                        max_iterations=1,
                        raise_on_failure=False,
                        existing_clips=existing_clips,
                        repair_shot_ids=set(weak_ids),
                        brand_logo_path=brand_logo,
                        product_image_paths=product_imgs,
                    )
                    if not services.is_current_generation(run_id, gen):
                        return None

                    _merge_cost_ledger(run_state, a3.cost_ledger.entries)
                    new_sc = None
                    if a3.render_manifest.stitched_path:
                        critic = VisionQualityCritic(settings=services.settings)
                        new_sc = await critic.score_video(
                            a3.render_manifest.stitched_path,
                            base_sheet,
                            run_state.creative_bible,
                            qa=a3.qa_report,
                            mock=iter_input.mock,
                            cancel=cancel_token,
                        )
                    new_ver = Version(
                        version_id=iter_id,
                        label=f"Auto-Polished R{round_idx} ({', '.join(weak_ids)})",
                        tier="preview",
                        status="done" if a3.render_manifest.stitched_path else "failed",
                        stack=stack,
                        usd=round(a3.cost_ledger.actual_usd, 4),
                        creative_bible=run_state.creative_bible,
                        shot_sheet=base_sheet,
                        sound_plan=run_state.sound_plan,
                        render_manifest=a3.render_manifest,
                        qa_report=a3.qa_report,
                        quality_scorecard=new_sc,
                    )
                    run_state.versions.append(new_ver)
                    rounds_completed = round_idx
                    if new_ver.status == "done":
                        current_ver = new_ver
                        new_score = float(new_sc.overall_score) if new_sc else 0.0
                        if new_score >= best_score:
                            best_score = new_score
                            best_ver = new_ver
                            run_state.shot_sheet = base_sheet
                    services.run_store.save(run_state)

                # Auto-select highest-scoring cut as winning preview
                run_state.winning_version_id = best_ver.version_id
                for v in run_state.versions:
                    v.is_winner = v.version_id == best_ver.version_id
                run_state.quality_scorecard = best_ver.quality_scorecard
                run_state.step = "previews"
                run_state.status = "idle"
                services.run_store.save(run_state)
                services.event_broker.publish(
                    run_id,
                    make_sse_event(
                        "stage_completed",
                        run_id,
                        stage="auto_polish",
                        data={
                            "rounds_completed": rounds_completed,
                            "best_version_id": best_ver.version_id,
                            "best_score": best_score,
                            "target_score": payload.target_score,
                        },
                    ),
                )
                services.event_broker.publish(
                    run_id,
                    make_sse_event(
                        "run_completed",
                        run_id,
                        stage="auto_polish",
                        data={"status": "idle", "best_version_id": best_ver.version_id},
                    ),
                )
                return run_state
            except asyncio.CancelledError:
                run_state.status = "cancelled"
                services.run_store.save(run_state)
                services.event_broker.publish(
                    run_id, make_sse_event("run_cancelled", run_id, stage="auto_polish")
                )
                raise
            except Exception as exc:
                run_state.status = "failed"
                run_state.error = str(exc)
                services.run_store.save(run_state)
                services.event_broker.publish(
                    run_id,
                    make_sse_event(
                        "run_failed", run_id, stage="auto_polish", data={"error": str(exc)}
                    ),
                )
                raise

        job = await services.job_registry.submit(run_id, _job, stage="auto_polish")
        services.run_store.save(run_state)
    return {
        "run_id": run_id,
        "target_score": payload.target_score,
        "max_rounds": payload.max_rounds,
        "job": job.model_dump(mode="json"),
    }
