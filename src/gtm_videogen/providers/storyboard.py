"""Keyframe-first visual storyboard generator ($0.003/still via Flux Schnell before video).

Generates a crisp visual keyframe still for every shot so the human can review, refine,
or regenerate visuals for pennies before committing to AI video rendering ($0.35+/clip).
Approved keyframes automatically anchor `image_to_video` (`start_image_url`) for stronger
visual continuity and product accuracy.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from gtm_videogen.core.config import Settings, get_settings
from gtm_videogen.core.cost_meter import CostMeter
from gtm_videogen.core.http_client import ResilientHttpClient
from gtm_videogen.core.job_registry import CancellationToken
from gtm_videogen.media.graphics_card import GraphicsCardRenderer, _pick_card_colors
from gtm_videogen.models.schemas import CreativeBible, Shot, ShotSheet

FLUX_UNIT_USD = 0.005

_ASPECT_FLUX_SIZE: dict[str, str] = {
    "16:9": "landscape_16_9",
    "9:16": "portrait_16_9",
    "1:1": "square_hd",
    "4:5": "portrait_4_3",
}


def _render_mock_still(
    shot: Shot,
    sheet: ShotSheet,
    bible: CreativeBible,
    out_path: Path,
) -> Path:
    """Render a clean labeled storyboard frame offline ($0) via GraphicsCardRenderer."""
    renderer = GraphicsCardRenderer()
    subtitle = shot.vo_line or shot.visual_prompt[:90]
    img = renderer._compose_card_image(
        text=f"{shot.id.upper()} · {shot.beat}",
        width=960,
        height=540,
        bible=bible,
        subtitle=subtitle,
        footer=f"{shot.engine} · {shot.duration_sec:.2f}s · {shot.camera_move or 'glide'}",
        card_kind="title",
        theme=shot.theme or sheet.theme,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="JPEG", quality=88)
    return out_path


def _render_motion_card_still(
    shot: Shot,
    sheet: ShotSheet,
    bible: CreativeBible,
    out_path: Path,
) -> Path:
    """Render the exact editorial typography still for a motion_card shot ($0)."""
    renderer = GraphicsCardRenderer()
    is_end = shot.index == len(sheet.shots) - 1
    card_text = (
        shot.on_screen_text
        or (sheet.end_card_title if is_end else None)
        or (shot.vo_line if is_end and shot.vo_line else None)
        or shot.beat
    )
    img = renderer._compose_card_image(
        text=card_text,
        width=1280,
        height=720,
        bible=bible,
        subtitle=sheet.end_card_subtitle if is_end else None,
        footer=sheet.end_card_footer if is_end else None,
        card_kind="end_card" if is_end else "title",
        theme=shot.theme or sheet.theme,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="JPEG", quality=90)
    return out_path


class StoryboardGenerator:
    """Generate per-shot visual keyframes via fal-ai/flux/schnell or local Pillow renderer."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        cost_meter: CostMeter | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.cost_meter = cost_meter

    async def generate_shot_keyframe(
        self,
        shot: Shot,
        sheet: ShotSheet,
        bible: CreativeBible,
        out_dir: Path,
        *,
        mock: bool = False,
        cancel: CancellationToken | None = None,
    ) -> tuple[Path, str | None, float]:
        """Generate a keyframe JPEG for a single shot. Returns (local_path, remote_url, usd)."""
        if cancel is not None:
            cancel.raise_if_cancelled()

        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{shot.id}_keyframe.jpg"

        if shot.engine == "motion_card":
            _render_motion_card_still(shot, sheet, bible, out_path)
            return out_path, None, 0.0

        if mock or not self.settings.fal_key.strip():
            _render_mock_still(shot, sheet, bible, out_path)
            return out_path, None, 0.0

        # Live Flux Schnell generation ($0.005/still, ~1.5s)
        model = (self.settings.image_model or "fal-ai/flux/schnell").lstrip("/")
        res_id: str | None = None
        if self.cost_meter is not None:
            res_id = self.cost_meter.reserve(
                f"keyframe:{shot.id}",
                FLUX_UNIT_USD,
                stage="storyboard",
                provider="fal",
                model=model,
            )

        style_suffix = (
            f" Visual world: {bible.visual_language.world}. "
            f"Lighting: {bible.visual_language.lighting_style}. "
            f"Ultra-sharp editorial commercial photography, no text or watermarks."
        )
        prompt = f"{shot.visual_prompt.strip()}.{style_suffix}"
        payload: dict[str, Any] = {
            "prompt": prompt,
            "image_size": _ASPECT_FLUX_SIZE.get(sheet.aspect, "landscape_16_9"),
            "num_inference_steps": 4,
            "num_images": 1,
            "output_format": "jpeg",
        }
        headers = {
            "Authorization": f"Key {self.settings.fal_key.strip()}",
            "Content-Type": "application/json",
        }

        async with ResilientHttpClient(
            connect_timeout=self.settings.http_connect_timeout_sec,
            read_timeout=60.0,
            write_timeout=30.0,
            pool_timeout=15.0,
            max_retries=2,
        ) as client:
            try:
                data = await client.post_json(
                    f"https://fal.run/{model}",
                    json=payload,
                    headers=headers,
                    cancel=cancel,
                )
                images = data.get("images") if isinstance(data, dict) else None
                remote_url = (
                    images[0].get("url")
                    if isinstance(images, list) and images and isinstance(images[0], dict)
                    else None
                )
                if not remote_url:
                    raise RuntimeError(f"Flux Schnell response missing image URL: {data}")

                img_bytes = await client.get_bytes(remote_url, cancel=cancel)
                out_path.write_bytes(img_bytes)
                if self.cost_meter is not None and res_id is not None:
                    self.cost_meter.settle(res_id, FLUX_UNIT_USD)
                return out_path, str(remote_url), FLUX_UNIT_USD
            except BaseException:
                if self.cost_meter is not None and res_id is not None:
                    self.cost_meter.release(res_id)
                raise

    async def generate_all_keyframes(
        self,
        sheet: ShotSheet,
        bible: CreativeBible,
        out_dir: Path,
        *,
        mock: bool = False,
        cancel: CancellationToken | None = None,
    ) -> list[tuple[str, Path, str | None, float]]:
        """Generate keyframes for all shots in bounded parallel batches (concurrency=4)."""
        sem = asyncio.Semaphore(4)

        async def _one(shot: Shot) -> tuple[str, Path, str | None, float]:
            async with sem:
                p, url, usd = await self.generate_shot_keyframe(
                    shot, sheet, bible, out_dir, mock=mock, cancel=cancel
                )
                return shot.id, p, url, usd

        return list(await asyncio.gather(*(_one(s) for s in sheet.shots)))


__all__ = ["FLUX_UNIT_USD", "StoryboardGenerator", "_pick_card_colors", "Image", "ImageDraw"]
