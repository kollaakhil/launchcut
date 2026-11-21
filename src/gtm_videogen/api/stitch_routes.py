"""FastAPI REST endpoints for Stitch (Concept C · Tactile Director's Pinboard)."""

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

from gtm_videogen.agents.stitch_engine import (
    StitchProjectStore,
    compile_stitch_project,
    generate_all_keyframes,
    generate_scene_keyframe,
    pitch_three_concepts,
    render_all_scenes,
    render_single_scene_take,
    stitch_master_cut,
)
from gtm_videogen.api.routes import ServicesDep, _iter_file_range, _parse_byte_range
from gtm_videogen.models.model_registry import (
    BUILTIN_LLM_MODELS,
    ModelRegistryStore,
    VideoModelSpec,
)
from gtm_videogen.models.stitch_project import (
    AudioMode,
    SceneKind,
    SeamTransition,
    StitchAspect,
    StitchBeat,
    StitchElement,
    StitchProject,
    StitchScene,
    apply_deterministic_script_autofix,
    get_stitch_catalog,
    validate_stitch_project,
)

stitch_router = APIRouter(prefix="/api/stitch", tags=["stitch"])

_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9._-]{1,120}$")
_ACTIVE_STITCH_TASKS: dict[str, asyncio.Task[Any]] = {}


def _gallery_dir() -> Path:
    # Works both from repo root and editable installs
    here = Path(__file__).resolve()
    for parent in (here.parents[3], here.parents[2], Path.cwd()):
        cand = parent / "assets" / "stitch_gallery"
        if cand.is_dir():
            return cand
    return Path.cwd() / "assets" / "stitch_gallery"


def _get_store(services: ServicesDep) -> StitchProjectStore:
    return StitchProjectStore(services.data_dir / "stitch")


def _get_registry(services: ServicesDep) -> ModelRegistryStore:
    return ModelRegistryStore(services.data_dir / "stitch")


def _decode_data_url(data_url: str, fallback_ext: str = ".png") -> tuple[bytes, str]:
    raw = data_url.strip()
    ext = fallback_ext
    if raw.startswith("data:"):
        header, _, b64_part = raw.partition(",")
        low_h = header.lower()
        if "jpeg" in low_h or "jpg" in low_h:
            ext = ".jpg"
        elif "webp" in low_h:
            ext = ".webp"
        elif "png" in low_h:
            ext = ".png"
        raw = b64_part
    try:
        return base64.b64decode(raw, validate=False), ext
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image payload: {exc}") from exc


def _serialize_project(project: StitchProject, registry: ModelRegistryStore) -> dict[str, Any]:
    """Return full StitchProject JSON with media URLs, Script Gate report, model specs & compiled JSON."""
    model_spec = registry.get_video_model(project.video_model_id)
    gate = validate_stitch_project(project, model_spec)
    data = project.model_dump()
    pid = project.project_id

    def _murl(path_str: str | None) -> str | None:
        if not path_str:
            return None
        p = Path(path_str)
        if not p.is_file():
            return None
        return f"/api/stitch/projects/{pid}/media/{p.name}"

    data["master_video_url"] = _murl(project.master_video_path)
    data["brand_logo_url"] = _murl(project.brand_logo_path)
    data["total_duration_sec"] = project.total_duration_sec()

    est_usd = 0.0
    for el_dict, el_obj in zip(data.get("elements", []), project.elements, strict=False):
        el_dict["image_url"] = _murl(el_obj.image_path)

    for idx, (sc_dict, sc_obj) in enumerate(zip(data.get("scenes", []), project.scenes, strict=False)):
        sc_dict["start_keyframe_media_url"] = _murl(sc_obj.start_keyframe_path)
        sc_dict["end_keyframe_media_url"] = _murl(sc_obj.end_keyframe_path)
        compiled_str = project.compile_scene_model_prompt(idx, max_chars=model_spec.max_prompt_chars)
        sc_dict["compiled_prompt_dict"] = project.to_scene_prompt_dict(idx)
        sc_dict["compiled_prompt_chars"] = len(compiled_str)
        sc_dict["max_prompt_chars"] = model_spec.max_prompt_chars
        if sc_obj.kind == "ai_sequence":
            est_usd += float(model_spec.snap_duration(sc_obj.duration_sec)) * float(
                model_spec.est_usd_per_sec
            )
        for t_dict, t_obj in zip(sc_dict.get("takes", []), sc_obj.takes, strict=False):
            t_dict["clip_url"] = _murl(t_obj.clip_path)
        win = sc_obj.winning_take()
        sc_dict["winning_clip_url"] = _murl(win.clip_path) if win else None

    data["estimated_usd"] = round(est_usd, 2)
    data["script_gate"] = gate
    data["active_model"] = model_spec.model_dump()
    return data


# ---------------------------------------------------------------------------
# Request Schemas
# ---------------------------------------------------------------------------


class PitchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    idea: str = Field(default="", max_length=1500)
    source_url: str | None = None
    llm_model_id: str = ""
    mock: bool = False


class CreateProjectRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    idea: str = Field(..., min_length=2, max_length=1500)
    target_duration_sec: int = Field(default=15, ge=5, le=60)
    aspect_ratio: StitchAspect = "16:9"
    video_model_id: str = "fal-ai/kling-video/v3/pro/text-to-video"
    llm_model_id: str = ""
    world_preset: str = "miniature_diorama"
    camera_preset: str = "macro_tilt_shift"
    lighting_preset: str = "volumetric_gold"
    motion_preset: str = "asmr_tactile"
    source_url: str | None = None
    recipe_id: str | None = None
    brand_name: str | None = None
    brand_tagline: str | None = None
    mock: bool = False


class PatchProjectRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: str | None = None
    style: str | None = None
    main_subject: str | None = None
    aspect_ratio: StitchAspect | None = None
    video_model_id: str | None = None
    llm_model_id: str | None = None
    audio_direction: str | None = None
    audio_mode: AudioMode | None = None
    seam_transition: SeamTransition | None = None
    voice_name: str | None = None
    add_brand_outro: bool | None = None
    brand_name: str | None = None
    brand_tagline: str | None = None
    negative_constraints: list[str] | None = None
    scenes: list[dict[str, Any]] | None = None


class AddSceneRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = "New Scene"
    duration_sec: int = Field(default=10, ge=4, le=15)
    kind: SceneKind = "ai_sequence"
    continues_from_prev: bool = True


class UploadElementRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    slot: int = Field(..., ge=1, le=3)
    role: Literal["product", "character", "style"] = "product"
    label: str = Field(default="Hero Reference", min_length=1, max_length=64)
    filename: str = "reference.png"
    data_url: str = Field(..., min_length=16)


class UploadLogoRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filename: str = "logo.png"
    data_url: str = Field(..., min_length=16)


class KeyframeActionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    which: Literal["start", "end"] = "start"
    tweak_note: str | None = None
    upload_data_url: str | None = None
    mock: bool = False


class AllKeyframesRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mock: bool = False


class RenderSceneRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    engine_id: str | None = None
    mock: bool = False
    sync: bool = False


class SelectTakeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    take_id: int = Field(..., ge=1)


class StitchMasterRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    audio_mode: AudioMode | None = None
    seam_transition: SeamTransition | None = None
    voice_name: str | None = None
    add_brand_outro: bool | None = None
    brand_name: str | None = None
    brand_tagline: str | None = None
    mock: bool = False


# ---------------------------------------------------------------------------
# Catalog, Gallery & Dynamic Model Registry
# ---------------------------------------------------------------------------


@stitch_router.get("/catalog")
async def get_catalog(services: ServicesDep) -> dict[str, Any]:
    reg = _get_registry(services)
    cat = get_stitch_catalog()
    cat["video_models"] = [m.model_dump() for m in reg.list_video_models()]
    cat["llm_models"] = BUILTIN_LLM_MODELS
    cat["live_video_ready"] = bool(services.settings.fal_key.strip())
    cat["live_llm_ready"] = bool(services.settings.openrouter_api_key.strip())
    return cat


@stitch_router.get("/gallery/{filename}")
async def serve_gallery_image(filename: str) -> FileResponse:
    if not _SAFE_NAME_RE.match(filename):
        raise HTTPException(status_code=400, detail="Invalid gallery filename")
    target = (_gallery_dir() / filename).resolve()
    if not target.is_file() or target.parent != _gallery_dir().resolve():
        raise HTTPException(status_code=404, detail="Gallery image not found")
    return FileResponse(target, headers={"Cache-Control": "public, max-age=86400"})


@stitch_router.get("/models")
async def list_models(services: ServicesDep) -> dict[str, Any]:
    reg = _get_registry(services)
    return {
        "video_models": [m.model_dump() for m in reg.list_video_models()],
        "llm_models": BUILTIN_LLM_MODELS,
    }


@stitch_router.post("/models")
async def add_custom_model(payload: VideoModelSpec, services: ServicesDep) -> dict[str, Any]:
    reg = _get_registry(services)
    saved = reg.add_custom_model(payload)
    return {
        "model": saved.model_dump(),
        "video_models": [m.model_dump() for m in reg.list_video_models()],
    }


@stitch_router.delete("/models/{model_id:path}")
async def remove_custom_model(model_id: str, services: ServicesDep) -> dict[str, Any]:
    reg = _get_registry(services)
    removed = reg.delete_custom_model(model_id)
    if not removed:
        raise HTTPException(status_code=404, detail="Custom model not found (built-in models cannot be deleted)")
    return {
        "deleted": model_id,
        "video_models": [m.model_dump() for m in reg.list_video_models()],
    }


# ---------------------------------------------------------------------------
# Stage 1: Pitch & Project Creation
# ---------------------------------------------------------------------------


@stitch_router.post("/pitch")
async def pitch_concepts(payload: PitchRequest, services: ServicesDep) -> dict[str, Any]:
    if not payload.idea.strip() and not (payload.source_url or "").strip():
        raise HTTPException(status_code=400, detail="Provide an idea or a product URL to pitch.")
    return await pitch_three_concepts(
        idea=payload.idea,
        source_url=payload.source_url,
        llm_model_id=payload.llm_model_id,
        settings=services.settings,
        mock=payload.mock,
    )


@stitch_router.get("/projects")
async def list_projects(services: ServicesDep) -> dict[str, Any]:
    store = _get_store(services)
    items = store.list_all(limit=25)
    return {
        "projects": [
            {
                "project_id": p.project_id,
                "title": p.title,
                "total_duration_sec": p.total_duration_sec(),
                "scene_count": len(p.scenes),
                "aspect_ratio": p.aspect_ratio,
                "status": p.status,
                "updated_at": p.updated_at,
                "has_master": bool(p.master_video_path and Path(p.master_video_path).is_file()),
            }
            for p in items
        ]
    }


@stitch_router.post("/projects")
async def create_project(payload: CreateProjectRequest, services: ServicesDep) -> dict[str, Any]:
    store = _get_store(services)
    reg = _get_registry(services)
    project = await compile_stitch_project(
        idea=payload.idea,
        target_duration_sec=payload.target_duration_sec,
        aspect_ratio=payload.aspect_ratio,
        video_model_id=payload.video_model_id,
        llm_model_id=payload.llm_model_id,
        world_preset=payload.world_preset,
        camera_preset=payload.camera_preset,
        lighting_preset=payload.lighting_preset,
        motion_preset=payload.motion_preset,
        source_url=payload.source_url,
        recipe_id=payload.recipe_id,
        brand_name=payload.brand_name,
        brand_tagline=payload.brand_tagline,
        registry_store=reg,
        settings=services.settings,
        mock=payload.mock,
    )
    store.save(project)
    return _serialize_project(project, reg)


@stitch_router.get("/projects/{project_id}")
async def get_project(project_id: str, services: ServicesDep) -> dict[str, Any]:
    store = _get_store(services)
    reg = _get_registry(services)
    project = store.load(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Stitch project not found")
    return _serialize_project(project, reg)


# ---------------------------------------------------------------------------
# Stage 2: Pinboard Editing, Scenes CRUD, Script Gate Auto-Fix
# ---------------------------------------------------------------------------


@stitch_router.patch("/projects/{project_id}")
async def patch_project(
    project_id: str, payload: PatchProjectRequest, services: ServicesDep
) -> dict[str, Any]:
    store = _get_store(services)
    reg = _get_registry(services)
    project = store.load(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Stitch project not found")

    for field in (
        "title",
        "style",
        "main_subject",
        "aspect_ratio",
        "video_model_id",
        "llm_model_id",
        "audio_direction",
        "audio_mode",
        "seam_transition",
        "voice_name",
        "add_brand_outro",
        "brand_name",
        "brand_tagline",
        "negative_constraints",
    ):
        val = getattr(payload, field)
        if val is not None:
            setattr(project, field, val)

    if payload.scenes is not None and len(payload.scenes) > 0:
        existing_by_id = {s.scene_id: s for s in project.scenes}
        updated_scenes: list[StitchScene] = []
        for idx, raw_sc in enumerate(payload.scenes[:6]):
            sc_id = str(raw_sc.get("scene_id") or f"sc_{idx + 1}_{uuid4().hex[:6]}")
            base = existing_by_id.get(sc_id)
            if base is not None:
                # Preserve media paths and take history unless explicitly overwritten
                merged = base.model_dump()
                for k in (
                    "title",
                    "duration_sec",
                    "kind",
                    "continues_from_prev",
                    "subjects",
                    "materials",
                    "camera_spec",
                    "colors",
                    "vo_line",
                    "end_frame",
                    "beats",
                ):
                    if k in raw_sc and raw_sc[k] is not None:
                        merged[k] = raw_sc[k]
                merged["index"] = idx
                if idx == 0:
                    merged["continues_from_prev"] = False
                updated_scenes.append(StitchScene.model_validate(merged))
            else:
                raw_copy = dict(raw_sc)
                raw_copy["scene_id"] = sc_id
                raw_copy["index"] = idx
                if idx == 0:
                    raw_copy["continues_from_prev"] = False
                updated_scenes.append(StitchScene.model_validate(raw_copy))
        project.scenes = updated_scenes
        project.target_duration_sec = project.total_duration_sec()

    store.save(project)
    return _serialize_project(project, reg)


@stitch_router.post("/projects/{project_id}/autofix")
async def autofix_project(project_id: str, services: ServicesDep) -> dict[str, Any]:
    store = _get_store(services)
    reg = _get_registry(services)
    project = store.load(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Stitch project not found")
    spec = reg.get_video_model(project.video_model_id)
    apply_deterministic_script_autofix(project, spec)
    project.target_duration_sec = project.total_duration_sec()
    store.save(project)
    return _serialize_project(project, reg)


@stitch_router.post("/projects/{project_id}/scenes")
async def add_scene(
    project_id: str, payload: AddSceneRequest, services: ServicesDep
) -> dict[str, Any]:
    store = _get_store(services)
    reg = _get_registry(services)
    project = store.load(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Stitch project not found")
    if len(project.scenes) >= 6:
        raise HTTPException(status_code=400, detail="Maximum 6 scenes per Stitch project.")

    idx = len(project.scenes)
    prev = project.scenes[-1] if project.scenes else None
    new_sc = StitchScene(
        scene_id=f"sc_{idx + 1}_{uuid4().hex[:6]}",
        index=idx,
        title=payload.title.strip() or f"Scene {idx + 1} · Next Beat",
        duration_sec=payload.duration_sec,
        kind=payload.kind,
        continues_from_prev=(payload.continues_from_prev and idx > 0),
        subjects=list(prev.subjects) if prev else [project.main_subject or "hero centerpiece"],
        materials=list(prev.materials) if prev else ["aged oak wood", "optical glass"],
        camera_spec=prev.camera_spec if prev else "85mm studio cinema · smooth continuous push-in",
        colors=list(prev.colors) if prev else ["#191512", "#E54D3A", "#F6E7D4"],
        end_frame="Locked-off hero composition with balanced negative space",
        beats=[
            StitchBeat(
                time=f"0-{max(2, payload.duration_sec // 2)}s",
                scene=(
                    f"Continues seamlessly from Scene {idx}'s locked frame ({prev.end_frame[:80]})."
                    if prev and payload.continues_from_prev
                    else "Opening studio composition."
                ),
                action="Fluid physical motion advances the scene with warm natural light.",
                camera="Smooth continuous cinematic glide.",
                sound="Crisp tactile ambient foley.",
            ),
            StitchBeat(
                time=f"{max(2, payload.duration_sec // 2)}-{payload.duration_sec}s",
                scene="Poised hero tableau settling into sharp focus.",
                action="Motion gently decelerates into a calm, luminous hold.",
                camera="Controlled slow lock-off.",
                sound="Warm harmonic bloom.",
            ),
        ],
    )
    project.scenes.append(new_sc)
    project.target_duration_sec = project.total_duration_sec()
    store.save(project)
    return _serialize_project(project, reg)


@stitch_router.delete("/projects/{project_id}/scenes/{scene_id}")
async def delete_scene(
    project_id: str, scene_id: str, services: ServicesDep
) -> dict[str, Any]:
    store = _get_store(services)
    reg = _get_registry(services)
    project = store.load(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Stitch project not found")
    if len(project.scenes) <= 1:
        raise HTTPException(status_code=400, detail="A Stitch film needs at least 1 Scene.")
    remaining = [s for s in project.scenes if s.scene_id != scene_id]
    if len(remaining) == len(project.scenes):
        raise HTTPException(status_code=404, detail="Scene not found")
    for idx, sc in enumerate(remaining):
        sc.index = idx
        if idx == 0:
            sc.continues_from_prev = False
    project.scenes = remaining
    project.target_duration_sec = project.total_duration_sec()
    store.save(project)
    return _serialize_project(project, reg)


# ---------------------------------------------------------------------------
# Stage 3: Reference Elements (@Element1..3), Editable Keyframes, Per-Act Takes
# ---------------------------------------------------------------------------


@stitch_router.post("/projects/{project_id}/elements")
async def upload_element(
    project_id: str, payload: UploadElementRequest, services: ServicesDep
) -> dict[str, Any]:
    store = _get_store(services)
    reg = _get_registry(services)
    project = store.load(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Stitch project not found")

    raw_bytes, ext = _decode_data_url(payload.data_url, ".png")
    media_dir = store.media_dir(project_id)
    out_file = media_dir / f"element_{payload.slot}_{uuid4().hex[:5]}{ext}"
    out_file.write_bytes(raw_bytes)

    el = StitchElement(
        slot=payload.slot,
        tag=f"@Element{payload.slot}",
        role=payload.role,
        label=payload.label.strip(),
        image_path=str(out_file),
        filename=payload.filename[:80],
    )
    project.elements = [e for e in project.elements if e.slot != payload.slot]
    project.elements.append(el)
    project.elements.sort(key=lambda e: e.slot)
    project.status_message = f"Locked {el.tag} ({el.label}) into Reference Tray."
    store.save(project)
    return _serialize_project(project, reg)


@stitch_router.delete("/projects/{project_id}/elements/{slot}")
async def delete_element(
    project_id: str, slot: int, services: ServicesDep
) -> dict[str, Any]:
    store = _get_store(services)
    reg = _get_registry(services)
    project = store.load(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Stitch project not found")
    project.elements = [e for e in project.elements if e.slot != slot]
    store.save(project)
    return _serialize_project(project, reg)


@stitch_router.post("/projects/{project_id}/logo")
async def upload_logo(
    project_id: str, payload: UploadLogoRequest, services: ServicesDep
) -> dict[str, Any]:
    store = _get_store(services)
    reg = _get_registry(services)
    project = store.load(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Stitch project not found")
    raw_bytes, ext = _decode_data_url(payload.data_url, ".png")
    media_dir = store.media_dir(project_id)
    out_file = media_dir / f"brand_logo_{uuid4().hex[:5]}{ext}"
    out_file.write_bytes(raw_bytes)
    project.brand_logo_path = str(out_file)
    project.add_brand_outro = True
    store.save(project)
    return _serialize_project(project, reg)


@stitch_router.post("/projects/{project_id}/keyframes")
async def generate_project_keyframes(
    project_id: str, payload: AllKeyframesRequest, services: ServicesDep
) -> dict[str, Any]:
    store = _get_store(services)
    reg = _get_registry(services)
    project = store.load(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Stitch project not found")
    await generate_all_keyframes(
        project,
        store.media_dir(project_id),
        settings=services.settings,
        mock=payload.mock,
    )
    store.save(project)
    return _serialize_project(project, reg)


@stitch_router.post("/projects/{project_id}/scenes/{scene_id}/keyframe")
async def update_scene_keyframe(
    project_id: str,
    scene_id: str,
    payload: KeyframeActionRequest,
    services: ServicesDep,
) -> dict[str, Any]:
    store = _get_store(services)
    reg = _get_registry(services)
    project = store.load(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Stitch project not found")
    idx = next((i for i, s in enumerate(project.scenes) if s.scene_id == scene_id), -1)
    if idx < 0:
        raise HTTPException(status_code=404, detail="Scene not found")

    if payload.upload_data_url:
        raw_bytes, ext = _decode_data_url(payload.upload_data_url, ".jpg")
        media_dir = store.media_dir(project_id)
        out_file = media_dir / f"{scene_id}_{payload.which}_upload_{uuid4().hex[:5]}{ext}"
        out_file.write_bytes(raw_bytes)
        if payload.which == "end":
            project.scenes[idx].end_keyframe_path = str(out_file)
            project.scenes[idx].end_keyframe_url = None
        else:
            project.scenes[idx].start_keyframe_path = str(out_file)
            project.scenes[idx].start_keyframe_url = None
        project.status_message = f"Uploaded custom {payload.which} frame for {project.scenes[idx].title}."
    else:
        await generate_scene_keyframe(
            project,
            idx,
            store.media_dir(project_id),
            which=payload.which,
            tweak_note=payload.tweak_note,
            settings=services.settings,
            mock=payload.mock,
        )
    store.save(project)
    return _serialize_project(project, reg)


@stitch_router.post("/projects/{project_id}/scenes/{scene_id}/render")
async def render_scene_endpoint(
    project_id: str,
    scene_id: str,
    payload: RenderSceneRequest,
    services: ServicesDep,
) -> dict[str, Any]:
    store = _get_store(services)
    reg = _get_registry(services)
    project = store.load(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Stitch project not found")
    idx = next((i for i, s in enumerate(project.scenes) if s.scene_id == scene_id), -1)
    if idx < 0:
        raise HTTPException(status_code=404, detail="Scene not found")

    def _on_progress(updated: StitchProject) -> None:
        store.save(updated)

    if payload.sync:
        await render_single_scene_take(
            project,
            idx,
            store.media_dir(project_id),
            engine_id=payload.engine_id,
            registry_store=reg,
            settings=services.settings,
            media_store=services.media_store,
            mock=payload.mock,
            on_progress=_on_progress,
        )
        store.save(project)
        return _serialize_project(project, reg)

    task_key = f"{project_id}:{scene_id}"
    existing = _ACTIVE_STITCH_TASKS.get(task_key)
    if existing and not existing.done():
        return _serialize_project(project, reg)

    project.scenes[idx].render_status = "rendering"
    project.scenes[idx].render_error = None
    project.scenes[idx].render_message = "Queued for rendering..."
    project.status = "shooting"
    project.status_message = f"Rendering {project.scenes[idx].title}..."
    store.save(project)

    async def _bg_render_one() -> None:
        try:
            fresh = store.load(project_id) or project
            await render_single_scene_take(
                fresh,
                idx,
                store.media_dir(project_id),
                engine_id=payload.engine_id,
                registry_store=reg,
                settings=services.settings,
                media_store=services.media_store,
                mock=payload.mock,
                on_progress=_on_progress,
            )
            store.save(fresh)
        except Exception as exc:
            fresh = store.load(project_id)
            if fresh and idx < len(fresh.scenes):
                fresh.scenes[idx].render_status = "failed"
                fresh.scenes[idx].render_error = str(exc)
                fresh.scenes[idx].render_message = f"Failed: {exc}"
                fresh.error = str(exc)
                fresh.status_message = f"{fresh.scenes[idx].title} failed: {exc}"
                store.save(fresh)
        finally:
            _ACTIVE_STITCH_TASKS.pop(task_key, None)

    _ACTIVE_STITCH_TASKS[task_key] = asyncio.create_task(_bg_render_one())
    return _serialize_project(project, reg)


@stitch_router.post("/projects/{project_id}/scenes/{scene_id}/select-take")
async def select_winning_take(
    project_id: str,
    scene_id: str,
    payload: SelectTakeRequest,
    services: ServicesDep,
) -> dict[str, Any]:
    store = _get_store(services)
    reg = _get_registry(services)
    project = store.load(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Stitch project not found")
    sc = next((s for s in project.scenes if s.scene_id == scene_id), None)
    if sc is None:
        raise HTTPException(status_code=404, detail="Scene not found")
    if not any(t.take_id == payload.take_id for t in sc.takes):
        raise HTTPException(status_code=404, detail="Take not found")
    sc.winning_take_id = payload.take_id
    project.status_message = f"Selected Take {payload.take_id} as hero take for {sc.title}."
    store.save(project)
    return _serialize_project(project, reg)


@stitch_router.post("/projects/{project_id}/render-all")
async def render_all_endpoint(
    project_id: str, payload: RenderSceneRequest, services: ServicesDep
) -> dict[str, Any]:
    store = _get_store(services)
    reg = _get_registry(services)
    project = store.load(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Stitch project not found")

    def _on_progress(updated: StitchProject) -> None:
        store.save(updated)

    if payload.sync:
        await render_all_scenes(
            project,
            store.media_dir(project_id),
            engine_id=payload.engine_id,
            registry_store=reg,
            settings=services.settings,
            media_store=services.media_store,
            mock=payload.mock,
            on_progress=_on_progress,
        )
        store.save(project)
        return _serialize_project(project, reg)

    task_key = f"{project_id}:all"
    existing = _ACTIVE_STITCH_TASKS.get(task_key)
    if existing and not existing.done():
        return _serialize_project(project, reg)

    project.status = "shooting"
    project.error = None
    project.status_message = f"Starting chained render across {len(project.scenes)} Scene(s)..."
    store.save(project)

    async def _bg_render_all() -> None:
        try:
            fresh = store.load(project_id) or project
            await render_all_scenes(
                fresh,
                store.media_dir(project_id),
                engine_id=payload.engine_id,
                registry_store=reg,
                settings=services.settings,
                media_store=services.media_store,
                mock=payload.mock,
                on_progress=_on_progress,
            )
            store.save(fresh)
        except Exception as exc:
            fresh = store.load(project_id)
            if fresh:
                fresh.status = "failed"
                fresh.error = str(exc)
                fresh.status_message = f"Render failed: {exc}"
                store.save(fresh)
        finally:
            _ACTIVE_STITCH_TASKS.pop(task_key, None)

    _ACTIVE_STITCH_TASKS[task_key] = asyncio.create_task(_bg_render_all())
    return _serialize_project(project, reg)


# ---------------------------------------------------------------------------
# Stage 4: Screening Room Master Stitch & HTTP 206 Range Streaming
# ---------------------------------------------------------------------------


@stitch_router.post("/projects/{project_id}/stitch")
async def stitch_master_endpoint(
    project_id: str, payload: StitchMasterRequest, services: ServicesDep
) -> dict[str, Any]:
    store = _get_store(services)
    reg = _get_registry(services)
    project = store.load(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Stitch project not found")

    for field in (
        "audio_mode",
        "seam_transition",
        "voice_name",
        "add_brand_outro",
        "brand_name",
        "brand_tagline",
    ):
        val = getattr(payload, field)
        if val is not None:
            setattr(project, field, val)

    try:
        await stitch_master_cut(
            project,
            store.media_dir(project_id),
            settings=services.settings,
            mock=payload.mock,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    store.save(project)
    return _serialize_project(project, reg)


@stitch_router.get("/projects/{project_id}/media/{filename}")
async def stream_stitch_media(
    project_id: str,
    filename: str,
    request: Request,
    services: ServicesDep,
) -> Response:
    if not _SAFE_NAME_RE.match(filename):
        raise HTTPException(status_code=400, detail="Invalid media filename")
    store = _get_store(services)
    media_dir = store.media_dir(project_id).resolve()
    target = (media_dir / filename).resolve()
    if not target.is_file() or target.parent != media_dir:
        raise HTTPException(status_code=404, detail="Media file not found")

    file_size = target.stat().st_size
    media_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
    range_header = request.headers.get("range")

    if range_header:
        parsed = _parse_byte_range(range_header, file_size)
        if parsed is None:
            return Response(
                status_code=416,
                headers={"Content-Range": f"bytes */{file_size}", "Accept-Ranges": "bytes"},
            )
        start, end = parsed
        length = end - start + 1
        return StreamingResponse(
            _iter_file_range(target, start, length),
            status_code=206,
            media_type=media_type,
            headers={
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(length),
                "Cache-Control": "no-cache",
            },
        )

    return StreamingResponse(
        _iter_file_range(target, 0, file_size),
        status_code=200,
        media_type=media_type,
        headers={
            "Accept-Ranges": "bytes",
            "Content-Length": str(file_size),
            "Cache-Control": "no-cache",
        },
    )


__all__ = ["stitch_router"]
