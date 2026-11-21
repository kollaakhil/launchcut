"""FastAPI endpoints for the Guided Structured JSON Director (10s / 20s / 30s)."""

from __future__ import annotations

import asyncio
import base64
import mimetypes
import re
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from gtm_videogen.agents.director_studio import (
    DirectorBlueprintStore,
    estimate_blueprint_cost,
    generate_blueprint_keyframes,
    generate_director_blueprint,
    refine_director_blueprint,
    render_director_blueprint,
)
from gtm_videogen.api.routes import ServicesDep, _iter_file_range, _parse_byte_range
from gtm_videogen.models.director_blueprint import (
    DirectorAspect,
    DirectorBlueprint,
    DirectorContinuityMode,
    DirectorDuration,
    DirectorFormatMode,
    get_director_presets_payload,
)

director_router = APIRouter(prefix="/api/director", tags=["director"])

_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9._-]{1,120}$")
_ACTIVE_RENDER_TASKS: dict[str, asyncio.Task[Any]] = {}


def _get_store(services: ServicesDep) -> DirectorBlueprintStore:
    return DirectorBlueprintStore(services.data_dir / "director")


def _serialize_blueprint(bp: DirectorBlueprint) -> dict[str, Any]:
    """Return blueprint JSON augmented with ready-to-render media URLs and compiled prompt."""
    data = bp.model_dump()
    bp_id = bp.blueprint_id

    def _media_url(path_str: str | None) -> str | None:
        if not path_str:
            return None
        p = Path(path_str)
        if not p.is_file():
            return None
        return f"/api/director/blueprints/{bp_id}/media/{p.name}"

    data["final_video_url"] = _media_url(bp.final_video_path)
    data["brand_logo_url"] = _media_url(bp.brand_logo_path)
    data["product_image_url"] = _media_url(bp.product_image_path)

    for act_dict, act_obj in zip(data.get("acts", []), bp.acts, strict=False):
        act_dict["start_keyframe_media_url"] = _media_url(act_obj.start_keyframe_path)
        act_dict["end_keyframe_media_url"] = _media_url(act_obj.end_keyframe_path)
        act_dict["rendered_clip_url"] = _media_url(act_obj.rendered_clip_path)

    data["compiled_master_json"] = bp.to_master_prompt_dict()
    data["compiled_act_jsons"] = [bp.to_act_prompt_dict(i) for i in range(len(bp.acts))]
    data["estimated_usd"] = estimate_blueprint_cost(bp)
    return data


class CreateBlueprintRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    idea: str = Field(..., min_length=2, max_length=1500)
    duration_sec: DirectorDuration = 10
    format_mode: DirectorFormatMode = "continuous"
    continuity_mode: DirectorContinuityMode = "last_frame_chain"
    aspect_ratio: DirectorAspect = "16:9"
    engine: str = "fal-ai/kling-video/v3/pro/text-to-video"
    native_audio: bool = True
    add_brand_endcard: bool = False
    world_preset: str = "miniature_diorama"
    camera_preset: str = "macro_tilt_shift"
    lighting_preset: str = "volumetric_gold"
    motion_preset: str = "asmr_tactile"
    recipe_id: str | None = None
    brand_name: str | None = None
    brand_tagline: str | None = None
    brand_url: str | None = None
    mock: bool = False


class RefineBlueprintRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instruction: str = Field(..., min_length=2, max_length=1000)
    act_number: int | None = Field(default=None, ge=1, le=3)
    beat_index: int | None = Field(default=None, ge=0, le=10)
    mock: bool = False


class PatchBlueprintRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: str | None = None
    style: str | None = None
    main_subject: str | None = None
    engine: str | None = None
    aspect_ratio: DirectorAspect | None = None
    continuity_mode: DirectorContinuityMode | None = None
    native_audio: bool | None = None
    add_brand_endcard: bool | None = None
    brand_name: str | None = None
    brand_tagline: str | None = None
    brand_url: str | None = None
    audio_direction: str | None = None
    camera: dict[str, Any] | None = None
    environment: dict[str, Any] | None = None
    lighting: dict[str, Any] | None = None
    subject_details: dict[str, Any] | None = None
    motion: dict[str, Any] | None = None
    acts: list[dict[str, Any]] | None = None
    negative_constraints: list[str] | None = None


class UploadAssetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["product", "logo"] = "product"
    filename: str = "asset.png"
    data_base64: str = Field(..., min_length=10)


class KeyframeGenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mock: bool = False


class RenderBlueprintRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    engine: str | None = None
    continuity_mode: DirectorContinuityMode | None = None
    native_audio: bool | None = None
    add_brand_endcard: bool | None = None
    mock: bool = False
    sync: bool = False


@director_router.get("/presets")
async def get_director_presets(services: ServicesDep) -> dict[str, Any]:
    """Return zero-jargon visual direction cards, concept recipes, refinement chips & engines."""
    payload = get_director_presets_payload()
    payload["live_ready"] = bool(
        services.settings.openrouter_api_key.strip() and services.settings.fal_key.strip()
    )
    payload["has_fal_key"] = bool(services.settings.fal_key.strip())
    payload["has_openrouter_key"] = bool(services.settings.openrouter_api_key.strip())
    return payload


@director_router.get("/blueprints")
async def list_director_blueprints(services: ServicesDep) -> dict[str, Any]:
    store = _get_store(services)
    items = store.list_all(limit=30)
    return {
        "blueprints": [
            {
                "blueprint_id": b.blueprint_id,
                "title": b.title,
                "duration_sec": b.duration_sec,
                "format_mode": b.format_mode,
                "engine": b.engine,
                "status": b.status,
                "updated_at": b.updated_at,
                "has_video": bool(b.final_video_path and Path(b.final_video_path).is_file()),
            }
            for b in items
        ]
    }


@director_router.post("/blueprints", status_code=201)
async def create_director_blueprint(
    req: CreateBlueprintRequest,
    services: ServicesDep,
) -> dict[str, Any]:
    store = _get_store(services)
    bp = await generate_director_blueprint(
        idea=req.idea,
        duration_sec=req.duration_sec,
        format_mode=req.format_mode,
        continuity_mode=req.continuity_mode,
        aspect_ratio=req.aspect_ratio,
        engine=req.engine,
        native_audio=req.native_audio,
        add_brand_endcard=req.add_brand_endcard,
        world_preset=req.world_preset,
        camera_preset=req.camera_preset,
        lighting_preset=req.lighting_preset,
        motion_preset=req.motion_preset,
        recipe_id=req.recipe_id,
        brand_name=req.brand_name,
        brand_tagline=req.brand_tagline,
        brand_url=req.brand_url,
        settings=services.settings,
        mock=req.mock,
    )
    store.save(bp)
    return _serialize_blueprint(bp)


@director_router.get("/blueprints/{blueprint_id}")
async def get_director_blueprint(
    blueprint_id: str,
    services: ServicesDep,
) -> dict[str, Any]:
    store = _get_store(services)
    bp = store.load(blueprint_id)
    if bp is None:
        raise HTTPException(status_code=404, detail="Blueprint not found")
    return _serialize_blueprint(bp)


@director_router.patch("/blueprints/{blueprint_id}")
async def patch_director_blueprint(
    blueprint_id: str,
    req: PatchBlueprintRequest,
    services: ServicesDep,
) -> dict[str, Any]:
    store = _get_store(services)
    bp = store.load(blueprint_id)
    if bp is None:
        raise HTTPException(status_code=404, detail="Blueprint not found")

    raw = bp.model_dump()
    updates = req.model_dump(exclude_none=True)
    for k, v in updates.items():
        if k in ("camera", "environment", "lighting", "subject_details", "motion") and isinstance(
            v, dict
        ):
            raw[k] = {**(raw.get(k) or {}), **v}
        else:
            raw[k] = v

    updated = DirectorBlueprint.model_validate(raw)
    updated.estimated_usd = estimate_blueprint_cost(updated)
    store.save(updated)
    return _serialize_blueprint(updated)


@director_router.post("/blueprints/{blueprint_id}/refine")
async def refine_blueprint_endpoint(
    blueprint_id: str,
    req: RefineBlueprintRequest,
    services: ServicesDep,
) -> dict[str, Any]:
    store = _get_store(services)
    bp = store.load(blueprint_id)
    if bp is None:
        raise HTTPException(status_code=404, detail="Blueprint not found")

    refined = await refine_director_blueprint(
        bp,
        instruction=req.instruction,
        act_number=req.act_number,
        beat_index=req.beat_index,
        settings=services.settings,
        mock=req.mock,
    )
    store.save(refined)
    return _serialize_blueprint(refined)


@director_router.post("/blueprints/{blueprint_id}/upload")
async def upload_blueprint_asset(
    blueprint_id: str,
    req: UploadAssetRequest,
    services: ServicesDep,
) -> dict[str, Any]:
    store = _get_store(services)
    bp = store.load(blueprint_id)
    if bp is None:
        raise HTTPException(status_code=404, detail="Blueprint not found")

    raw_b64 = req.data_base64.strip()
    if "," in raw_b64 and raw_b64.startswith("data:"):
        raw_b64 = raw_b64.split(",", 1)[1]
    try:
        blob = base64.b64decode(raw_b64, validate=False)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64 payload: {exc}") from exc
    if not blob or len(blob) > 12 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Asset must be between 1 byte and 12 MB")

    ext = Path(req.filename).suffix.lower()
    if ext not in (".png", ".jpg", ".jpeg", ".webp"):
        ext = ".png"
    media_dir = store.media_dir(blueprint_id)
    out_file = media_dir / f"uploaded_{req.kind}{ext}"
    out_file.write_bytes(blob)

    if req.kind == "logo":
        bp.brand_logo_path = str(out_file)
    else:
        bp.product_image_path = str(out_file)
    store.save(bp)
    return _serialize_blueprint(bp)


@director_router.post("/blueprints/{blueprint_id}/keyframes")
async def generate_keyframes_endpoint(
    blueprint_id: str,
    req: KeyframeGenerateRequest,
    services: ServicesDep,
) -> dict[str, Any]:
    store = _get_store(services)
    bp = store.load(blueprint_id)
    if bp is None:
        raise HTTPException(status_code=404, detail="Blueprint not found")

    media_dir = store.media_dir(blueprint_id)
    updated = await generate_blueprint_keyframes(
        bp,
        media_dir,
        settings=services.settings,
        mock=req.mock,
    )
    store.save(updated)
    return _serialize_blueprint(updated)


@director_router.post("/blueprints/{blueprint_id}/render", status_code=202)
async def render_blueprint_endpoint(
    blueprint_id: str,
    req: RenderBlueprintRequest,
    services: ServicesDep,
) -> dict[str, Any]:
    store = _get_store(services)
    bp = store.load(blueprint_id)
    if bp is None:
        raise HTTPException(status_code=404, detail="Blueprint not found")

    existing = _ACTIVE_RENDER_TASKS.get(blueprint_id)
    if existing is not None and not existing.done():
        raise HTTPException(status_code=409, detail="Render already in progress for this blueprint")

    if req.engine is not None:
        bp.engine = req.engine
    if req.continuity_mode is not None:
        bp.continuity_mode = req.continuity_mode
    if req.native_audio is not None:
        bp.native_audio = req.native_audio
    if req.add_brand_endcard is not None:
        bp.add_brand_endcard = req.add_brand_endcard

    bp.estimated_usd = estimate_blueprint_cost(bp)
    bp.status = "rendering"
    bp.error = None
    bp.status_message = "Queued for rendering..."
    store.save(bp)

    media_dir = store.media_dir(blueprint_id)

    def _save_progress(state: DirectorBlueprint) -> None:
        store.save(state)

    async def _run_job() -> None:
        try:
            done_bp = await render_director_blueprint(
                bp,
                media_dir,
                settings=services.settings,
                media_store=services.media_store,
                mock=req.mock,
                on_progress=_save_progress,
            )
            store.save(done_bp)
        except Exception as exc:
            bp.status = "failed"
            bp.error = str(exc)
            bp.status_message = f"Render failed: {exc}"
            store.save(bp)
        finally:
            _ACTIVE_RENDER_TASKS.pop(blueprint_id, None)

    if req.sync:
        await _run_job()
        final_bp = store.load(blueprint_id) or bp
        return _serialize_blueprint(final_bp)

    task = asyncio.create_task(_run_job(), name=f"director_render_{blueprint_id}")
    _ACTIVE_RENDER_TASKS[blueprint_id] = task
    return _serialize_blueprint(bp)


@director_router.get("/blueprints/{blueprint_id}/media/{filename}")
async def serve_blueprint_media(
    blueprint_id: str,
    filename: str,
    request: Request,
    services: ServicesDep,
) -> Response:
    if not _SAFE_NAME_RE.match(filename) or ".." in filename or "/" in filename:
        raise HTTPException(status_code=400, detail="Invalid media filename")

    store = _get_store(services)
    media_dir = store.media_dir(blueprint_id).resolve()
    target = (media_dir / filename).resolve()
    try:
        target.relative_to(media_dir)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Path outside blueprint media root") from exc

    if not target.is_file():
        raise HTTPException(status_code=404, detail="Media file not found")

    size = target.stat().st_size
    content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
    range_header = request.headers.get("range")
    if range_header:
        parsed = _parse_byte_range(range_header, size)
        if parsed is None:
            return Response(
                status_code=416,
                headers={"Content-Range": f"bytes */{size}", "Accept-Ranges": "bytes"},
            )
        start, end = parsed
        length = end - start + 1
        return StreamingResponse(
            _iter_file_range(target, start, length),
            status_code=206,
            media_type=content_type,
            headers={
                "Content-Range": f"bytes {start}-{end}/{size}",
                "Content-Length": str(length),
                "Accept-Ranges": "bytes",
                "Cache-Control": "no-cache",
            },
        )

    return StreamingResponse(
        _iter_file_range(target, 0, size) if size > 0 else iter(()),
        status_code=200,
        media_type=content_type,
        headers={
            "Content-Length": str(size),
            "Accept-Ranges": "bytes",
            "Cache-Control": "no-cache",
        },
    )


__all__ = ["director_router"]
