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
import copy
import mimetypes
from pathlib import Path
from typing import Any
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
from gtm_videogen.core.cost_meter import CostMeter
from gtm_videogen.core.job_registry import CancellationToken
from gtm_videogen.models.schemas import (
    CostEntry,
    CreativeBible,
    RunState,
    Script,
    ShotSheet,
    SoundPlan,
    Version,
    VoiceCue,
)
from gtm_videogen.rules.creative_history import CreativeHistoryStore
from gtm_videogen.rules.script_gate import evaluate_script_gate

studio_router = APIRouter(prefix="/api")

# Curated multi-option preview stacks (inspired by GTM_VideoGen sheet-studio)
PREVIEW_STACK_PRESETS: list[dict[str, Any]] = [
    {
        "id": "kling_flash",
        "label": "Kling v4 Flash",
        "description": (
            "Fast live AI video ($0.094/s) + ElevenLabs v3 voiceover. "
            "Best speed/cost balance for previews."
        ),
        "video_model": "kling/v4/flash/text-to-video",
        "voice_model": "fal-ai/elevenlabs/tts/eleven-v3",
        "hybrid_motion_cards": False,
        "mock": False,
        "est_per_ai_shot_usd": 0.35,
    },
    {
        "id": "hybrid_motion",
        "label": "Hybrid Hero + Motion Cards",
        "description": (
            "Renders opening/hero beats with Kling v4 Flash and "
            "closing/supporting beats as $0 kinetic typography cards."
        ),
        "video_model": "kling/v4/flash/text-to-video",
        "voice_model": "fal-ai/elevenlabs/tts/eleven-v3",
        "hybrid_motion_cards": True,
        "mock": False,
        "est_per_ai_shot_usd": 0.35,
    },
    {
        "id": "kling_cinema",
        "label": "Kling v4 Cinema",
        "description": (
            "Full Kling v4 text/image-to-video model for maximum texture and cinematic motion."
        ),
        "video_model": "kling/v4/text-to-video",
        "voice_model": "fal-ai/elevenlabs/tts/eleven-v3",
        "hybrid_motion_cards": False,
        "mock": False,
        "est_per_ai_shot_usd": 0.55,
    },
    {
        "id": "mock_pacing",
        "label": "Instant Pacing Animatic ($0)",
        "description": (
            "Zero-cost FFmpeg synthetic preview to verify timing, pacing, "
            "voice cadence, and captions before spending."
        ),
        "video_model": "mock-lavfi-mp4",
        "voice_model": "mock-lavfi-wav",
        "hybrid_motion_cards": False,
        "mock": True,
        "est_per_ai_shot_usd": 0.0,
    },
]


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


class PreviewStackSpec(BaseModel):
    """Single preview option / model stack chosen for Step 4 A/B comparison."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=64)
    label: str = Field(min_length=1, max_length=120)
    video_model: str = "kling/v4/flash/text-to-video"
    voice_model: str = "fal-ai/elevenlabs/tts/eleven-v3"
    hybrid_motion_cards: bool = False
    mock: bool = False


class PreviewsRequest(BaseModel):
    """Launch 1-4 preview stacks for side-by-side comparison."""

    model_config = ConfigDict(extra="forbid")

    stacks: list[PreviewStackSpec] = Field(min_length=1, max_length=4)


class FinalRenderRequest(BaseModel):
    """Optional override for Step 5 Final Cut."""

    model_config = ConfigDict(extra="forbid")

    version_id: str | None = None
    mock: bool | None = None


class CapPatchRequest(BaseModel):
    """Live USD budget cap adjustment."""

    model_config = ConfigDict(extra="forbid")

    cap_usd: float = Field(ge=0.0, le=500.0)


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
    return {"presets": presets}


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
                        )
                        _merge_cost_ledger(run_state, a3.cost_ledger.entries)
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
                            usd=round(meter.actual_usd, 4),
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

                stack = winner.stack if winner else {}
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
                    final_input.mock = bool(stack.get("mock"))

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
                    video_model_override=stack.get("video_model"),
                    voice_model_override=stack.get("voice_model"),
                )

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
                )
                if not services.is_current_generation(run_id, gen):
                    return None

                _merge_cost_ledger(run_state, a3.cost_ledger.entries)
                run_state.render_manifest = a3.render_manifest
                run_state.qa_report = a3.qa_report
                run_state.rubric_evaluations["agent3_producer"] = a3.rubric_evaluations
                run_state.versions.append(
                    Version(
                        version_id=f"final_{uuid4().hex[:6]}",
                        label=f"Final Cut ({winner.label if winner else 'Default'})",
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
                        data={"stitched_path": a3.render_manifest.stitched_path},
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
