"""FastAPI REST & HTTP 206 Media Streaming Endpoints for Launch Cut Cinema Studio."""

from __future__ import annotations

import asyncio
import base64
import mimetypes
import re
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from gtm_videogen.agents.cinema_producer import (
    autopilot_cinema_project,
    build_leica_reel_animatic,
    estimate_cinema_shoot_usd,
    generate_single_shot_keyframes,
    generate_vault_references,
    remaster_cinema_cut,
    render_cinema_shot_take,
    shoot_and_mix_cinema_film,
)
from gtm_videogen.agents.cinema_writers_room import (
    apply_cinema_taste_or_note,
    compile_cinema_master_script,
    pitch_cinema_treatments,
    run_table_read_timing_pass,
)
from gtm_videogen.api.routes import ServicesDep, _iter_file_range, _parse_byte_range
from gtm_videogen.models.cinema_compiler import (
    autofix_cinema_project_gate,
    evaluate_story_rubric,
    to_hydrated_shot_prompt_dict,
    validate_cinema_project,
)
from gtm_videogen.models.cinema_project import (
    CinemaAspect,
    CinemaProject,
    CinemaProjectStore,
    CinemaScene,
    CinemaShot,
    StoryArchetype,
    VibeDials,
    WorkflowMode,
    WorldVault,
)
from gtm_videogen.models.model_registry import (
    BUILTIN_LLM_MODELS,
    ModelRegistryStore,
)
from gtm_videogen.rules.cinema_taste import (
    CINEMA_MOOD_CATALOG,
    STORY_ARCHETYPE_CATALOG,
    TASTE_REACTION_CHIPS,
    TasteProfileStore,
)

cinema_router = APIRouter(prefix="/api/cinema", tags=["cinema"])

_SAFE_SEGMENT_RE = re.compile(r"^[A-Za-z0-9._-]{1,120}$")
_ACTIVE_CINEMA_TASKS: dict[str, asyncio.Task[Any]] = {}


def _get_cinema_store(services: ServicesDep) -> CinemaProjectStore:
    return CinemaProjectStore(services.data_dir / "cinema")


def _get_taste_store(services: ServicesDep) -> TasteProfileStore:
    return TasteProfileStore(services.data_dir / "cinema")


def _get_registry(services: ServicesDep) -> ModelRegistryStore:
    return ModelRegistryStore(services.data_dir / "stitch")


def _actor_from_request(request: Request) -> str:
    raw = (request.headers.get("X-Actor-Id") or "default").strip()
    return re.sub(r"[^A-Za-z0-9._:-]", "", raw)[:64] or "default"


def _decode_data_url_image(data_url: str) -> tuple[bytes, str]:
    raw = data_url.strip()
    ext = ".jpg"
    if raw.startswith("data:"):
        hdr, _, b64 = raw.partition(",")
        low = hdr.lower()
        if "png" in low:
            ext = ".png"
        elif "webp" in low:
            ext = ".webp"
        raw = b64
    try:
        return base64.b64decode(raw, validate=False), ext
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image payload: {exc}") from exc


def _serialize_cinema_project(
    project: CinemaProject,
    store: CinemaProjectStore,
    registry: ModelRegistryStore,
    *,
    settings: Any,
) -> dict[str, Any]:
    """Serialize CinemaProject with media URLs, hydrated shot JSON previews, and Script Gate report."""
    pdir = store.project_dir(project.project_id)
    model_spec = registry.get_video_model(project.video_model_id)
    gate = validate_cinema_project(project, model_spec)
    project.estimated_shoot_usd = estimate_cinema_shoot_usd(project, settings=settings)

    def _murl(raw_path: str | None) -> str | None:
        if not raw_path:
            return None
        p = Path(raw_path)
        if not p.is_file():
            return None
        try:
            rel = p.resolve().relative_to(pdir.resolve())
            parts = rel.parts
            if len(parts) == 2:
                return f"/api/cinema/projects/{project.project_id}/media/{parts[0]}/{parts[1]}"
        except Exception:
            pass
        return None

    doc = project.model_dump()
    doc["total_duration_sec"] = project.total_duration_sec()
    doc["shot_count"] = len(project.all_shots())
    doc["leica_reel_url"] = _murl(project.leica_reel_path)
    doc["master_video_url"] = _murl(project.master_video_path)
    doc["master_score_url"] = _murl(project.master_score_path)
    doc["script_gate"] = gate

    for tr in doc.get("treatments", []):
        tr["look_test_url"] = _murl(tr.get("look_test_image_path"))

    vault_doc = doc.get("vault", {})
    for ch in vault_doc.get("characters", []):
        ch["turnaround_url"] = _murl(ch.get("turnaround_sheet_path"))
        ch["frontal_url"] = _murl(ch.get("frontal_crop_path"))
        ch["three_quarter_url"] = _murl(ch.get("three_quarter_crop_path"))
        ch["full_body_url"] = _murl(ch.get("full_body_crop_path"))

    for loc in vault_doc.get("locations", []):
        loc["primary_wall_url"] = _murl(loc.get("primary_wall_plate_path"))
        loc["reverse_wall_url"] = _murl(loc.get("reverse_wall_plate_path"))
        loc["room_tone_url"] = _murl(loc.get("room_tone_audio_path"))

    for pr in vault_doc.get("props", []):
        pr["turnaround_url"] = _murl(pr.get("turnaround_sheet_path"))
        pr["frontal_url"] = _murl(pr.get("frontal_crop_path"))

    for sc_idx, sc_doc in enumerate(doc.get("scenes", [])):
        sc_obj = project.scenes[sc_idx]
        sc_doc["duration_sec"] = sc_obj.duration_sec()
        for sh_idx, sh_doc in enumerate(sc_doc.get("shots", [])):
            sh_obj = sc_obj.shots[sh_idx]
            sh_doc["ref_strip_url"] = _murl(sh_doc.get("ref_strip_path"))
            sh_doc["start_keyframe_media_url"] = _murl(sh_doc.get("start_keyframe_path"))
            sh_doc["end_keyframe_media_url"] = _murl(sh_doc.get("end_keyframe_path"))
            if sh_doc.get("dialogue"):
                sh_doc["dialogue"]["audio_url"] = _murl(
                    sh_doc["dialogue"].get("synthesized_audio_path")
                )
            for tk in sh_doc.get("takes", []):
                tk["clip_url"] = _murl(tk.get("clip_path"))
                tk["rendered_end_frame_url"] = _murl(tk.get("rendered_end_frame_path"))
            win = sh_obj.winning_take()
            sh_doc["winning_clip_url"] = _murl(win.clip_path) if win else None
            sh_doc["hydrated_prompt_json"] = to_hydrated_shot_prompt_dict(
                sh_obj, sc_obj, project.vault, supports_elements=model_spec.supports_elements
            )

    return doc


# ---------------------------------------------------------------------------
# Request Payloads
# ---------------------------------------------------------------------------


class CreateCinemaProjectRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    idea: str = Field(..., min_length=3, max_length=1200)
    aspect_ratio: CinemaAspect = "16:9"
    target_duration_sec: int = Field(default=90, ge=20, le=210)
    workflow_mode: WorkflowMode = "guided"
    mood_presets: list[str] = Field(default_factory=lambda: ["cinema_a24"])
    story_archetype: StoryArchetype = "auto"
    user_vibe_note: str = ""
    dials: VibeDials | None = None
    source_url: str | None = None
    brand_name: str | None = None
    brand_tagline: str | None = None
    reference_images_b64: list[str] = Field(default_factory=list)
    video_model_id: str = "auto"
    llm_model_id: str = ""
    budget_cap_usd: float = Field(default=15.0, ge=0.0, le=250.0)
    mock: bool = False
    wait: bool = False


class CompileScriptRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    treatment_id: str | None = None
    mock: bool = False
    wait: bool = True


class TasteReactionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    note_or_chip: str = Field(..., min_length=1, max_length=600)
    target_type: Literal["project", "treatment", "vault", "scene", "shot", "cut"] = "project"
    target_id: str = ""
    polarity: Literal["love", "adjust", "reject"] = "adjust"
    mock: bool = False


class PatchCinemaProjectRequest(BaseModel):
    """2-Way Live JSON Inspector & Project Settings Patch."""

    model_config = ConfigDict(extra="ignore")

    title: str | None = None
    video_model_id: str | None = None
    budget_cap_usd: float | None = None
    enable_lipsync: bool | None = None
    enable_subtitles: bool | None = None
    cinema_letterbox: bool | None = None
    film_grain: bool | None = None
    vault: dict[str, Any] | None = None
    scenes: list[dict[str, Any]] | None = None
    shot_patch: dict[str, Any] | None = None
    winning_take_selection: dict[str, int] | None = None
    autofix_gate: bool = False
    resynth_dialogue: bool = False
    mock: bool = False


class BuildAnimaticRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    force_vault: bool = False
    mock: bool = False
    wait: bool = False


class ShotKeyframeRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    which: Literal["start", "end", "both"] = "both"
    tweak_note: str | None = None
    mock: bool = False


class ShootShotRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    engine_override: str | None = None
    mock: bool = False


class ProduceFilmRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    only_missing_takes: bool = True
    mock: bool = False
    wait: bool = False


class UploadReferenceRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    data_url: str = Field(..., min_length=16)
    target_kind: Literal["project", "character", "prop", "logo"] = "project"
    target_id: str = ""


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@cinema_router.get("/catalog")
async def get_cinema_catalog(request: Request, services: ServicesDep) -> dict[str, Any]:
    actor = _actor_from_request(request)
    reg = _get_registry(services)
    t_store = _get_taste_store(services)
    prof = t_store.get_profile(actor)
    has_live_video = bool(
        services.settings.fal_key.strip() or services.settings.effective_gemini_key()
    )
    return {
        "moods": CINEMA_MOOD_CATALOG,
        "archetypes": STORY_ARCHETYPE_CATALOG,
        "reaction_chips": TASTE_REACTION_CHIPS,
        "video_models": [m.model_dump() for m in reg.list_video_models()],
        "llm_models": BUILTIN_LLM_MODELS,
        "taste_profile": prof.model_dump(),
        "live_keys_configured": has_live_video,
    }


@cinema_router.get("/projects")
async def list_cinema_projects(services: ServicesDep) -> dict[str, Any]:
    store = _get_cinema_store(services)
    items = store.list_all(limit=25)
    return {
        "projects": [
            {
                "project_id": p.project_id,
                "title": p.title,
                "idea": p.idea,
                "status": p.status,
                "aspect_ratio": p.aspect_ratio,
                "total_duration_sec": p.total_duration_sec(),
                "shot_count": len(p.all_shots()),
                "actual_usd": p.actual_usd,
                "updated_at": p.updated_at,
            }
            for p in items
        ]
    }


@cinema_router.post("/projects")
async def create_cinema_project(
    body: CreateCinemaProjectRequest,
    request: Request,
    services: ServicesDep,
) -> dict[str, Any]:
    actor = _actor_from_request(request)
    store = _get_cinema_store(services)
    t_store = _get_taste_store(services)
    reg = _get_registry(services)

    pid = f"cin_{uuid4().hex[:10]}"
    pdir = store.project_dir(pid)
    uploads_dir = store.media_dir(pid, "uploads")

    saved_refs: list[str] = []
    for idx, b64_img in enumerate(body.reference_images_b64[:4]):
        raw_bytes, ext = _decode_data_url_image(b64_img)
        dest = uploads_dir / f"ref_{idx + 1}{ext}"
        dest.write_bytes(raw_bytes)
        saved_refs.append(str(dest))

    project = await pitch_cinema_treatments(
        idea=body.idea,
        aspect_ratio=body.aspect_ratio,
        target_duration_sec=body.target_duration_sec,
        workflow_mode=body.workflow_mode,
        mood_presets=body.mood_presets,
        story_archetype=body.story_archetype,
        user_vibe_note=body.user_vibe_note,
        dials=body.dials,
        source_url=body.source_url,
        brand_name=body.brand_name,
        brand_tagline=body.brand_tagline,
        uploaded_reference_paths=saved_refs,
        video_model_id=body.video_model_id,
        llm_model_id=body.llm_model_id,
        budget_cap_usd=body.budget_cap_usd,
        actor_id=actor,
        project_id=pid,
        media_dir=pdir,
        taste_store=t_store,
        settings=services.settings,
        mock=body.mock,
    )
    store.save(project)

    if body.workflow_mode == "autopilot":
        if body.wait:
            await autopilot_cinema_project(
                project,
                pdir,
                actor_id=actor,
                settings=services.settings,
                mock=body.mock,
                on_progress=lambda p: store.save(p),
            )
            store.save(project)
        else:
            project.status = "shooting"
            project.progress_stage = "Autopilot · Compiling Vault & Script"
            project.status_message = "1-Click Autopilot running..."
            store.save(project)

            async def _bg_autopilot() -> None:
                try:
                    await autopilot_cinema_project(
                        project,
                        pdir,
                        actor_id=actor,
                        settings=services.settings,
                        mock=body.mock,
                        on_progress=lambda p: store.save(p),
                    )
                    store.save(project)
                except Exception as exc:
                    project.status = "failed"
                    project.error = str(exc)
                    project.status_message = f"Autopilot failed: {exc}"
                    store.save(project)
                finally:
                    _ACTIVE_CINEMA_TASKS.pop(pid, None)

            _ACTIVE_CINEMA_TASKS[pid] = asyncio.create_task(_bg_autopilot())

    return _serialize_cinema_project(project, store, reg, settings=services.settings)


@cinema_router.get("/projects/{project_id}")
async def get_cinema_project(project_id: str, services: ServicesDep) -> dict[str, Any]:
    store = _get_cinema_store(services)
    reg = _get_registry(services)
    project = store.load(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Cinema project not found")
    return _serialize_cinema_project(project, store, reg, settings=services.settings)


@cinema_router.post("/projects/{project_id}/compile")
async def compile_project_script(
    project_id: str,
    body: CompileScriptRequest,
    request: Request,
    services: ServicesDep,
) -> dict[str, Any]:
    actor = _actor_from_request(request)
    store = _get_cinema_store(services)
    t_store = _get_taste_store(services)
    reg = _get_registry(services)
    project = store.load(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Cinema project not found")
    pdir = store.project_dir(project_id)

    if body.wait:
        await compile_cinema_master_script(
            project,
            pdir,
            treatment_id=body.treatment_id,
            actor_id=actor,
            taste_store=t_store,
            settings=services.settings,
            mock=body.mock,
            on_progress=lambda p: store.save(p),
        )
        store.save(project)
    else:
        project.status = "compiling"
        project.progress_pct = 20
        project.status_message = "Compiling WorldVault, Scene Spine & Table-Read timings..."
        store.save(project)

        async def _bg_compile() -> None:
            try:
                await compile_cinema_master_script(
                    project,
                    pdir,
                    treatment_id=body.treatment_id,
                    actor_id=actor,
                    taste_store=t_store,
                    settings=services.settings,
                    mock=body.mock,
                    on_progress=lambda p: store.save(p),
                )
                store.save(project)
            except Exception as exc:
                project.status = "failed"
                project.error = str(exc)
                project.status_message = f"Script compilation failed: {exc}"
                store.save(project)
            finally:
                _ACTIVE_CINEMA_TASKS.pop(project_id, None)

        _ACTIVE_CINEMA_TASKS[project_id] = asyncio.create_task(_bg_compile())

    return _serialize_cinema_project(project, store, reg, settings=services.settings)


@cinema_router.patch("/projects/{project_id}")
async def patch_cinema_project(
    project_id: str,
    body: PatchCinemaProjectRequest,
    services: ServicesDep,
) -> dict[str, Any]:
    """2-Way Live JSON Inspector & Project Settings Sync."""
    store = _get_cinema_store(services)
    reg = _get_registry(services)
    project = store.load(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Cinema project not found")
    pdir = store.project_dir(project_id)

    if body.title is not None:
        project.title = body.title.strip()[:90] or project.title
    if body.video_model_id is not None:
        project.video_model_id = body.video_model_id.strip() or "auto"
    if body.budget_cap_usd is not None:
        project.budget_cap_usd = max(0.0, float(body.budget_cap_usd))
    if body.enable_lipsync is not None:
        project.enable_lipsync = body.enable_lipsync
    if body.enable_subtitles is not None:
        project.enable_subtitles = body.enable_subtitles
    if body.cinema_letterbox is not None:
        project.cinema_letterbox = body.cinema_letterbox
    if body.film_grain is not None:
        project.film_grain = body.film_grain

    try:
        if body.vault is not None:
            project.vault = WorldVault.model_validate(body.vault)
        if body.scenes is not None:
            project.scenes = [CinemaScene.model_validate(s) for s in body.scenes]
        if body.shot_patch is not None:
            target_sid = str(body.shot_patch.get("shot_id") or "")
            found = project.find_shot(target_sid)
            if found is None:
                raise HTTPException(status_code=404, detail=f"Shot '{target_sid}' not found")
            sc_obj, sh_obj, _ = found
            merged = {
                **sh_obj.model_dump(),
                **body.shot_patch,
                "shot_id": sh_obj.shot_id,
                "scene_id": sc_obj.scene_id,
            }
            updated_shot = CinemaShot.model_validate(merged)
            for idx_s, s_item in enumerate(sc_obj.shots):
                if s_item.shot_id == sh_obj.shot_id:
                    sc_obj.shots[idx_s] = updated_shot
                    break
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid Cinema JSON schema: {exc}") from exc

    if body.winning_take_selection:
        for sid, tid in body.winning_take_selection.items():
            found = project.find_shot(sid)
            if found and any(t.take_id == int(tid) for t in found[1].takes):
                found[1].winning_take_id = int(tid)

    if body.resynth_dialogue:
        await run_table_read_timing_pass(
            project, pdir / "audio", settings=services.settings, mock=body.mock
        )
    elif body.autofix_gate:
        autofix_cinema_project_gate(project)
    else:
        project.story_rubric = evaluate_story_rubric(project)

    store.save(project)
    return _serialize_cinema_project(project, store, reg, settings=services.settings)


@cinema_router.post("/projects/{project_id}/react")
async def react_to_cinema_project(
    project_id: str,
    body: TasteReactionRequest,
    request: Request,
    services: ServicesDep,
) -> dict[str, Any]:
    actor = _actor_from_request(request)
    store = _get_cinema_store(services)
    t_store = _get_taste_store(services)
    reg = _get_registry(services)
    project = store.load(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Cinema project not found")
    pdir = store.project_dir(project_id)

    await apply_cinema_taste_or_note(
        project,
        pdir,
        note_or_chip=body.note_or_chip,
        target_type=body.target_type,
        target_id=body.target_id,
        polarity=body.polarity,
        actor_id=actor,
        taste_store=t_store,
        settings=services.settings,
        mock=body.mock,
    )
    store.save(project)
    return _serialize_cinema_project(project, store, reg, settings=services.settings)


@cinema_router.post("/projects/{project_id}/animatic")
async def build_project_animatic(
    project_id: str,
    body: BuildAnimaticRequest,
    services: ServicesDep,
) -> dict[str, Any]:
    store = _get_cinema_store(services)
    reg = _get_registry(services)
    project = store.load(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Cinema project not found")
    pdir = store.project_dir(project_id)

    if body.wait:
        await build_leica_reel_animatic(
            project,
            pdir,
            force_vault=body.force_vault,
            settings=services.settings,
            mock=body.mock,
            on_progress=lambda p: store.save(p),
        )
        store.save(project)
    else:
        project.status = "building_animatic"
        project.progress_pct = 45
        project.status_message = "Building Vault sheets, Start/End keyframes & Voiced Leica Reel..."
        store.save(project)

        async def _bg_animatic() -> None:
            try:
                await build_leica_reel_animatic(
                    project,
                    pdir,
                    force_vault=body.force_vault,
                    settings=services.settings,
                    mock=body.mock,
                    on_progress=lambda p: store.save(p),
                )
                store.save(project)
            except Exception as exc:
                project.status = "failed"
                project.error = str(exc)
                project.status_message = f"Animatic build failed: {exc}"
                store.save(project)
            finally:
                _ACTIVE_CINEMA_TASKS.pop(project_id, None)

        _ACTIVE_CINEMA_TASKS[project_id] = asyncio.create_task(_bg_animatic())

    return _serialize_cinema_project(project, store, reg, settings=services.settings)


@cinema_router.post("/projects/{project_id}/shots/{shot_id}/keyframes")
async def regenerate_shot_keyframes(
    project_id: str,
    shot_id: str,
    body: ShotKeyframeRequest,
    services: ServicesDep,
) -> dict[str, Any]:
    store = _get_cinema_store(services)
    reg = _get_registry(services)
    project = store.load(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Cinema project not found")
    pdir = store.project_dir(project_id)

    # Ensure Vault turnaround crops exist first
    if project.vault.characters and not project.vault.characters[0].frontal_crop_path:
        await generate_vault_references(project, pdir, settings=services.settings, mock=body.mock)

    found = project.find_shot(shot_id)
    if found is None:
        raise HTTPException(status_code=404, detail=f"Shot '{shot_id}' not found")
    _, _, flat_idx = found
    all_shots = project.all_shots()
    prev_sh = all_shots[flat_idx - 1] if flat_idx > 0 else None

    await generate_single_shot_keyframes(
        project,
        shot_id,
        pdir,
        which=body.which,
        tweak_note=body.tweak_note,
        prev_shot=prev_sh,
        settings=services.settings,
        mock=body.mock,
    )
    store.save(project)
    return _serialize_cinema_project(project, store, reg, settings=services.settings)


@cinema_router.post("/projects/{project_id}/shots/{shot_id}/shoot")
async def shoot_single_shot(
    project_id: str,
    shot_id: str,
    body: ShootShotRequest,
    services: ServicesDep,
) -> dict[str, Any]:
    store = _get_cinema_store(services)
    reg = _get_registry(services)
    project = store.load(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Cinema project not found")
    pdir = store.project_dir(project_id)

    found = project.find_shot(shot_id)
    if found is None:
        raise HTTPException(status_code=404, detail=f"Shot '{shot_id}' not found")
    _, _, flat_idx = found
    all_shots = project.all_shots()
    prev_win = all_shots[flat_idx - 1].winning_take() if flat_idx > 0 else None
    prev_clip = (
        Path(prev_win.clip_path) if (prev_win and Path(prev_win.clip_path).is_file()) else None
    )

    await render_cinema_shot_take(
        project,
        shot_id,
        pdir,
        engine_override=body.engine_override,
        prev_rendered_clip_path=prev_clip,
        settings=services.settings,
        mock=body.mock,
    )
    store.save(project)
    return _serialize_cinema_project(project, store, reg, settings=services.settings)


@cinema_router.post("/projects/{project_id}/produce")
async def produce_cinema_film(
    project_id: str,
    body: ProduceFilmRequest,
    services: ServicesDep,
) -> dict[str, Any]:
    store = _get_cinema_store(services)
    reg = _get_registry(services)
    project = store.load(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Cinema project not found")
    pdir = store.project_dir(project_id)

    if body.wait:
        await shoot_and_mix_cinema_film(
            project,
            pdir,
            only_missing_takes=body.only_missing_takes,
            settings=services.settings,
            mock=body.mock,
            on_progress=lambda p: store.save(p),
        )
        store.save(project)
    else:
        project.status = "shooting"
        project.progress_pct = 74
        project.status_message = "Rolling Principal Photography & Cinema Soundstage..."
        store.save(project)

        async def _bg_shoot() -> None:
            try:
                await shoot_and_mix_cinema_film(
                    project,
                    pdir,
                    only_missing_takes=body.only_missing_takes,
                    settings=services.settings,
                    mock=body.mock,
                    on_progress=lambda p: store.save(p),
                )
                store.save(project)
            except Exception as exc:
                project.status = "failed"
                project.error = str(exc)
                project.status_message = f"Production failed: {exc}"
                store.save(project)
            finally:
                _ACTIVE_CINEMA_TASKS.pop(project_id, None)

        _ACTIVE_CINEMA_TASKS[project_id] = asyncio.create_task(_bg_shoot())

    return _serialize_cinema_project(project, store, reg, settings=services.settings)


@cinema_router.post("/projects/{project_id}/remaster")
async def remaster_project_film(
    project_id: str,
    body: ProduceFilmRequest,
    services: ServicesDep,
) -> dict[str, Any]:
    store = _get_cinema_store(services)
    reg = _get_registry(services)
    project = store.load(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Cinema project not found")
    pdir = store.project_dir(project_id)

    await remaster_cinema_cut(
        project,
        pdir,
        settings=services.settings,
        mock=body.mock,
        on_progress=lambda p: store.save(p),
    )
    store.save(project)
    return _serialize_cinema_project(project, store, reg, settings=services.settings)


@cinema_router.post("/projects/{project_id}/upload-reference")
async def upload_project_reference(
    project_id: str,
    body: UploadReferenceRequest,
    services: ServicesDep,
) -> dict[str, Any]:
    store = _get_cinema_store(services)
    reg = _get_registry(services)
    project = store.load(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Cinema project not found")
    uploads_dir = store.media_dir(project_id, "uploads")

    raw_bytes, ext = _decode_data_url_image(body.data_url)
    dest = uploads_dir / f"{body.target_kind}_{uuid4().hex[:6]}{ext}"
    dest.write_bytes(raw_bytes)

    if body.target_kind == "logo":
        project.brand_logo_path = str(dest)
    elif body.target_kind == "character":
        ch = project.vault.get_character(body.target_id)
        if ch:
            ch.uploaded_ref_path = str(dest)
            ch.turnaround_sheet_path = None
    elif body.target_kind == "prop":
        pr = project.vault.get_prop(body.target_id)
        if pr:
            pr.uploaded_ref_path = str(dest)
            pr.turnaround_sheet_path = None
    else:
        project.uploaded_reference_paths.append(str(dest))

    store.save(project)
    return _serialize_cinema_project(project, store, reg, settings=services.settings)


@cinema_router.get("/projects/{project_id}/media/{subdir}/{filename}")
async def serve_cinema_media(
    project_id: str,
    subdir: str,
    filename: str,
    request: Request,
    services: ServicesDep,
) -> Response:
    """Serve project media with path-traversal guard and HTTP 206 Partial Content Range streaming."""
    if not _SAFE_SEGMENT_RE.match(subdir) or not _SAFE_SEGMENT_RE.match(filename):
        raise HTTPException(status_code=400, detail="Invalid media path segment")
    store = _get_cinema_store(services)
    pdir = store.project_dir(project_id).resolve()
    target = (pdir / subdir / filename).resolve()
    try:
        target.relative_to(pdir)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="Path traversal blocked") from exc

    if not target.is_file():
        raise HTTPException(status_code=404, detail="Media asset not found")

    mime, _ = mimetypes.guess_type(str(target))
    media_type = mime or "application/octet-stream"
    file_size = target.stat().st_size
    range_header = request.headers.get("range") or request.headers.get("Range")

    if not range_header:
        return FileResponse(
            target,
            media_type=media_type,
            headers={"Accept-Ranges": "bytes", "Cache-Control": "no-cache"},
        )

    parsed_range = _parse_byte_range(range_header, file_size)
    if parsed_range is None:
        return Response(
            status_code=416,
            headers={"Content-Range": f"bytes */{file_size}"},
        )
    start, end = parsed_range
    content_length = end - start + 1
    return StreamingResponse(
        _iter_file_range(target, start, content_length),
        status_code=206,
        media_type=media_type,
        headers={
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(content_length),
            "Cache-Control": "no-cache",
        },
    )


__all__ = ["cinema_router"]
