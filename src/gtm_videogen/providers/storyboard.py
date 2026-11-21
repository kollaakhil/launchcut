"""Keyframe-first visual storyboard generator with Reference-Locked Continuity.

- Shot 1 generates the Master Look Anchor via fal-ai/flux/schnell ($0.005/still).
- Shots 2..N anchor to Shot 1's CDN URL via fal-ai/flux-kontext/dev (with automatic
  Flux Schnell fallback) so lighting, color temperature, lens, and product world stay
  visually locked across every shot in the film.
- `motion_card` shots render exact $0 studio typography / real logo lockup stills.
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
KONTEXT_UNIT_USD = 0.025
KONTEXT_MODEL = "fal-ai/flux-kontext/dev"

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
    *,
    logo_path: Path | None = None,
    media_image_path: Path | None = None,
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
        logo_path=logo_path if shot.index == len(sheet.shots) - 1 else None,
        media_image_path=media_image_path if shot.index != len(sheet.shots) - 1 else None,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="JPEG", quality=88)
    return out_path


def _render_motion_card_still(
    shot: Shot,
    sheet: ShotSheet,
    bible: CreativeBible,
    out_path: Path,
    *,
    logo_path: Path | None = None,
    media_image_path: Path | None = None,
) -> Path:
    """Render the exact editorial studio / logo lockup still for a motion_card shot ($0)."""
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
        logo_path=logo_path if is_end else None,
        media_image_path=media_image_path if not is_end else None,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="JPEG", quality=90)
    return out_path


class StoryboardGenerator:
    """Generate continuity-locked visual keyframes via Flux Schnell + Flux Kontext."""

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
        anchor_image_url: str | None = None,
        logo_path: Path | None = None,
        media_image_path: Path | None = None,
        cancel: CancellationToken | None = None,
    ) -> tuple[Path, str | None, float]:
        """Generate a keyframe JPEG for a single shot. Returns (local_path, remote_url, usd)."""
        if cancel is not None:
            cancel.raise_if_cancelled()

        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{shot.id}_keyframe.jpg"

        if shot.engine == "motion_card":
            _render_motion_card_still(
                shot,
                sheet,
                bible,
                out_path,
                logo_path=logo_path,
                media_image_path=media_image_path,
            )
            return out_path, None, 0.0

        if mock or not self.settings.fal_key.strip():
            _render_mock_still(
                shot,
                sheet,
                bible,
                out_path,
                logo_path=logo_path,
                media_image_path=media_image_path,
            )
            return out_path, None, 0.0

        headers = {
            "Authorization": f"Key {self.settings.fal_key.strip()}",
            "Content-Type": "application/json",
        }
        palette_str = ", ".join(bible.visual_language.color_palette[:4])
        style_suffix = (
            f" Visual world: {bible.visual_language.world}. "
            f"Lighting: {bible.visual_language.lighting_style}. "
            f"Color palette: {palette_str}. "
            "Ultra-sharp editorial commercial photography, zero garbled text or watermarks."
        )

        # Reference-locked continuity via Flux Kontext when an anchor frame URL is provided
        use_kontext = bool(anchor_image_url and anchor_image_url.startswith(("http://", "https://")))
        if use_kontext:
            try:
                return await self._generate_with_kontext(
                    shot=shot,
                    anchor_url=anchor_image_url or "",
                    style_suffix=style_suffix,
                    headers=headers,
                    out_path=out_path,
                    cancel=cancel,
                )
            except Exception:
                # Graceful fallback to Flux Schnell if Kontext fails or budget rejects
                pass

        return await self._generate_with_schnell(
            shot=shot,
            sheet=sheet,
            style_suffix=style_suffix,
            headers=headers,
            out_path=out_path,
            cancel=cancel,
        )

    async def _generate_with_kontext(
        self,
        *,
        shot: Shot,
        anchor_url: str,
        style_suffix: str,
        headers: dict[str, str],
        out_path: Path,
        cancel: CancellationToken | None,
    ) -> tuple[Path, str | None, float]:
        res_id: str | None = None
        if self.cost_meter is not None:
            res_id = self.cost_meter.reserve(
                f"keyframe_kontext:{shot.id}",
                KONTEXT_UNIT_USD,
                stage="storyboard",
                provider="fal",
                model=KONTEXT_MODEL,
            )
        prompt = (
            "Preserve the exact same lighting temperature, studio environment, material finish, "
            "and cinematic lens aesthetic as the reference frame while staging this next shot: "
            f"{shot.visual_prompt.strip()}.{style_suffix}"
        )
        payload: dict[str, Any] = {
            "prompt": prompt,
            "image_url": anchor_url,
            "num_inference_steps": 24,
            "guidance_scale": 3.5,
            "num_images": 1,
            "output_format": "jpeg",
            "resolution_mode": "match_input",
        }
        async with ResilientHttpClient(
            connect_timeout=self.settings.http_connect_timeout_sec,
            read_timeout=75.0,
            write_timeout=30.0,
            pool_timeout=15.0,
            max_retries=1,
        ) as client:
            try:
                data = await client.post_json(
                    f"https://fal.run/{KONTEXT_MODEL}",
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
                    raise RuntimeError(f"Flux Kontext response missing image URL: {data}")
                img_bytes = await client.get_bytes(remote_url, cancel=cancel)
                out_path.write_bytes(img_bytes)
                if self.cost_meter is not None and res_id is not None:
                    self.cost_meter.settle(res_id, KONTEXT_UNIT_USD)
                return out_path, str(remote_url), KONTEXT_UNIT_USD
            except BaseException:
                if self.cost_meter is not None and res_id is not None:
                    self.cost_meter.release(res_id)
                raise

    async def _generate_with_schnell(
        self,
        *,
        shot: Shot,
        sheet: ShotSheet,
        style_suffix: str,
        headers: dict[str, str],
        out_path: Path,
        cancel: CancellationToken | None,
    ) -> tuple[Path, str | None, float]:
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

        prompt = f"{shot.visual_prompt.strip()}.{style_suffix}"
        payload: dict[str, Any] = {
            "prompt": prompt,
            "image_size": _ASPECT_FLUX_SIZE.get(sheet.aspect, "landscape_16_9"),
            "num_inference_steps": 4,
            "num_images": 1,
            "output_format": "jpeg",
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
        logo_path: Path | None = None,
        product_images: list[Path] | None = None,
        cancel: CancellationToken | None = None,
    ) -> list[tuple[str, Path, str | None, float]]:
        """Generate Shot 1 Master Look Anchor, then continuity-locked Shots 2..N in parallel."""
        ordered = sorted(sheet.shots, key=lambda s: s.index)
        if not ordered:
            return []

        results: dict[str, tuple[str, Path, str | None, float]] = {}
        anchor_url: str | None = None
        first_ai = next((s for s in ordered if s.engine != "motion_card"), None)
        prod_imgs = [p for p in (product_images or []) if p.is_file()]

        # Step 1: Generate the Master Look Anchor (first AI shot) so remaining shots lock to it
        if first_ai is not None:
            p0, url0, usd0 = await self.generate_shot_keyframe(
                first_ai,
                sheet,
                bible,
                out_dir,
                mock=mock,
                logo_path=logo_path,
                media_image_path=prod_imgs[0] if prod_imgs else None,
                cancel=cancel,
            )
            results[first_ai.id] = (first_ai.id, p0, url0, usd0)
            anchor_url = url0

        # Step 2: Generate all remaining shots in bounded parallel anchored to Shot 1
        sem = asyncio.Semaphore(4)
        remaining = [s for s in ordered if s.id not in results]

        async def _one(idx: int, shot: Shot) -> tuple[str, Path, str | None, float]:
            async with sem:
                media_img = prod_imgs[idx % len(prod_imgs)] if prod_imgs else None
                p, url, usd = await self.generate_shot_keyframe(
                    shot,
                    sheet,
                    bible,
                    out_dir,
                    mock=mock,
                    anchor_image_url=anchor_url if shot.engine != "motion_card" else None,
                    logo_path=logo_path,
                    media_image_path=media_img,
                    cancel=cancel,
                )
                return shot.id, p, url, usd

        rest = await asyncio.gather(*(_one(i, s) for i, s in enumerate(remaining)))
        for item in rest:
            results[item[0]] = item

        return [results[s.id] for s in ordered]


__all__ = [
    "FLUX_UNIT_USD",
    "KONTEXT_MODEL",
    "KONTEXT_UNIT_USD",
    "StoryboardGenerator",
    "_pick_card_colors",
    "Image",
    "ImageDraw",
]
