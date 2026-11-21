"""Bespoke Pillow + FFmpeg motion graphics card renderer (title, end, lower-third, metric).

Addresses:
- F12 / Pillar 4: Driven by CreativeBible.typography_and_layout & color_palette.
- BUG-08 & BUG-09: Explicitly binds width, height, fps, and duration_sec on every render.
- P5: Memoizes PIL font loading per process via functools.lru_cache.
- Zero hardcoded house styles (no fixed yellow pills, tilt angles, or OS font paths).
"""

from __future__ import annotations

import os
import shutil
import tempfile
import textwrap
from functools import lru_cache
from pathlib import Path
from typing import Literal

from PIL import Image, ImageDraw, ImageFont

from gtm_videogen.core.config import get_settings
from gtm_videogen.core.job_registry import CancellationToken
from gtm_videogen.media._subprocess import probe_media, run_subprocess
from gtm_videogen.media.captions import (
    ensure_wcag_contrast,
    parse_hex_color,
    relative_luminance,
    resolve_font_path,
)
from gtm_videogen.media.ffmpeg_stitcher import _move_output_atomically
from gtm_videogen.models.schemas import CreativeBible, GraphicsTheme, RenderedClip

CardKind = Literal["title", "end_card", "lower_third", "metric"]


@lru_cache(maxsize=128)
def _load_pil_font(font_path_str: str, size: int) -> ImageFont.FreeTypeFont:
    """Memoized TrueType font loader (P5)."""
    return ImageFont.truetype(font_path_str, size=max(10, size))


def _pick_card_colors(
    bible: CreativeBible,
    theme: GraphicsTheme | None = None,
) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]:
    """Return (bg_rgb, text_rgb, accent_rgb) with guaranteed WCAG >= 4.5:1 contrast."""
    palette: list[str] = []
    for raw in bible.visual_language.color_palette:
        try:
            parse_hex_color(raw)
            palette.append(raw.strip())
        except ValueError:
            continue

    if theme is not None:
        bg_hex = theme.background_color
        fg_hex = theme.text_color
        accent_hex = theme.accent_color
        clamp_bg = theme.clamp_background_luminance
    else:
        bg_hex = (
            min(palette, key=lambda c: relative_luminance(parse_hex_color(c)))
            if palette
            else "#111827"
        )
        fg_hex = "#FFFFFF"
        accent_hex = (
            max(palette, key=lambda c: relative_luminance(parse_hex_color(c)))
            if len(palette) > 1
            else (palette[0] if palette else "#3B82F6")
        )
        clamp_bg = False

    fg_checked, bg_checked = ensure_wcag_contrast(
        fg_hex,
        bg_hex,
        min_ratio=4.5,
        clamp_background_luminance=clamp_bg,
    )
    return (
        parse_hex_color(bg_checked),
        parse_hex_color(fg_checked),
        parse_hex_color(accent_hex),
    )


def _wrap_lines(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    max_width: int,
) -> list[str]:
    words = text.strip().split()
    if not words:
        return []
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        bbox = draw.textbbox((0, 0), candidate, font=font)
        if (bbox[2] - bbox[0]) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)

    # Fallback hard-wrap if a single token exceeds max_width
    final_lines: list[str] = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        if (bbox[2] - bbox[0]) <= max_width or len(line) <= 8:
            final_lines.append(line)
        else:
            est_chars = max(8, int(len(line) * (max_width / max(1, bbox[2] - bbox[0]))))
            final_lines.extend(textwrap.wrap(line, width=est_chars))
    return final_lines


class GraphicsCardRenderer:
    """Renders bespoke motion-graphics cards via Pillow + FFmpeg with exact video specs."""

    def __init__(
        self,
        *,
        ffmpeg_bin: str | None = None,
        ffprobe_bin: str | None = None,
        fonts_dir: Path | None = None,
    ) -> None:
        settings = get_settings()
        self.ffmpeg_bin = ffmpeg_bin or settings.ffmpeg_bin
        self.ffprobe_bin = ffprobe_bin or settings.ffprobe_bin
        self.fonts_dir = str(fonts_dir.resolve()) if fonts_dir else None

    def _compose_card_image(
        self,
        *,
        text: str,
        width: int,
        height: int,
        bible: CreativeBible,
        subtitle: str | None = None,
        footer: str | None = None,
        card_kind: CardKind = "title",
        theme: GraphicsTheme | None = None,
        logo_path: Path | None = None,
        media_image_path: Path | None = None,
    ) -> Image.Image:
        bg_rgb, text_rgb, accent_rgb = _pick_card_colors(bible, theme)
        typo = bible.typography_and_layout
        font_family = typo.font_family or (theme.font_family if theme else "Roboto")

        bold_path = str(resolve_font_path(font_family, bold=True, fonts_dir=self.fonts_dir))
        reg_path = str(resolve_font_path(font_family, bold=False, fonts_dir=self.fonts_dir))

        base_dim = min(width, height)
        if card_kind == "metric":
            title_size = max(36, int(base_dim * 0.13))
            sub_size = max(22, int(base_dim * 0.052))
        elif card_kind == "lower_third":
            title_size = max(28, int(base_dim * 0.068))
            sub_size = max(20, int(base_dim * 0.044))
        else:
            title_size = max(32, int(base_dim * 0.088))
            sub_size = max(20, int(base_dim * 0.046))
        footer_size = max(16, int(base_dim * 0.034))

        title_font = _load_pil_font(bold_path, title_size)
        sub_font = _load_pil_font(reg_path, sub_size)
        footer_font = _load_pil_font(reg_path, footer_size)

        # Editorial radial studio lighting backdrop tinted subtly with brand accent
        grad_w, grad_h = 160, 90
        grad_small = Image.new("RGB", (grad_w, grad_h), color=bg_rgb)
        gpx = grad_small.load()
        cx, cy = grad_w * 0.5, grad_h * 0.42
        max_r = (grad_w * grad_w + grad_h * grad_h) ** 0.5 * 0.65
        for gy in range(grad_h):
            for gx in range(grad_w):
                dist = ((gx - cx) ** 2 + (gy - cy) ** 2) ** 0.5
                t = max(0.0, 1.0 - dist / max_r)
                glow = (t * t) * 0.14
                vig = (1.0 - t) * 0.10
                r_c = int(max(0, min(255, bg_rgb[0] * (1.0 - vig) + accent_rgb[0] * glow)))
                g_c = int(max(0, min(255, bg_rgb[1] * (1.0 - vig) + accent_rgb[1] * glow)))
                b_c = int(max(0, min(255, bg_rgb[2] * (1.0 - vig) + accent_rgb[2] * glow)))
                gpx[gx, gy] = (r_c, g_c, b_c)
        img = grad_small.resize((width, height), resample=Image.Resampling.BILINEAR)
        draw = ImageDraw.Draw(img)

        # Subtle editorial framing hairline
        frame_pad = max(18, int(min(width, height) * 0.045))
        border_rgb = tuple(int(bg_rgb[i] * 0.78 + text_rgb[i] * 0.22) for i in range(3))
        draw.rectangle(
            [frame_pad, frame_pad, width - frame_pad, height - frame_pad],
            outline=border_rgb,
            width=max(1, int(min(width, height) * 0.002)),
        )

        # Optional real uploaded product/UI screenshot showcase (floating dark studio bezel)
        if media_image_path is not None and Path(media_image_path).is_file():
            try:
                media_img = Image.open(media_image_path).convert("RGBA")
                max_mw = int(width * 0.56)
                max_mh = int(height * 0.54)
                media_img.thumbnail((max_mw, max_mh), Image.Resampling.LANCZOS)
                mw, mh = media_img.size
                mx = (width - mw) // 2
                my = int(height * 0.11)
                bezel = max(6, int(base_dim * 0.012))
                draw.rectangle(
                    [mx - bezel, my - bezel, mx + mw + bezel, my + mh + bezel],
                    fill=tuple(max(0, int(c * 0.55)) for c in bg_rgb),
                    outline=border_rgb,
                    width=max(1, int(base_dim * 0.0025)),
                )
                img.paste(media_img, (mx, my), media_img)
                # Render clean editorial caption underneath the framed product/UI shot
                cap_font = _load_pil_font(bold_path, max(24, int(base_dim * 0.056)))
                cap_lines = _wrap_lines(draw, text, cap_font, int(width * 0.80))
                cy_txt = my + mh + bezel + int(height * 0.045)
                for line in cap_lines[:2]:
                    bbox = draw.textbbox((0, 0), line, font=cap_font)
                    lw = bbox[2] - bbox[0]
                    draw.text(((width - lw) // 2, cy_txt), line, fill=text_rgb, font=cap_font)
                    cy_txt += int(cap_font.size * 1.25)
                return img
            except Exception:
                pass

        # Optional real uploaded brand logo (composited cleanly above headline on end cards)
        logo_img: Image.Image | None = None
        if logo_path is not None and Path(logo_path).is_file():
            try:
                loaded_logo = Image.open(logo_path).convert("RGBA")
                loaded_logo.thumbnail(
                    (int(width * 0.34), int(base_dim * 0.16)), Image.Resampling.LANCZOS
                )
                logo_img = loaded_logo
            except Exception:
                logo_img = None

        margin_x = max(32, int(width * 0.09))
        max_text_w = max(100, width - 2 * margin_x)
        layout = (typo.end_card_layout or "centered_minimal").lower()

        title_lines = _wrap_lines(draw, text, title_font, max_text_w)
        sub_lines = _wrap_lines(draw, subtitle, sub_font, max_text_w) if subtitle else []

        title_line_h = int(title_size * 1.24)
        sub_line_h = int(sub_size * 1.35)
        accent_h = max(4, int(base_dim * 0.008))

        logo_h = (logo_img.height + int(base_dim * 0.05)) if logo_img is not None else 0
        block_h = logo_h + len(title_lines) * title_line_h
        if sub_lines:
            block_h += int(base_dim * 0.04) + len(sub_lines) * sub_line_h

        if card_kind == "lower_third":
            band_h = block_h + int(height * 0.10)
            band_top = max(int(height * 0.55), height - band_h - int(height * 0.07))
            draw.rectangle(
                [margin_x, band_top, margin_x + int(width * 0.18), band_top + accent_h],
                fill=accent_rgb,
            )
            cursor_y = band_top + accent_h + int(height * 0.03)
            align_left = True
        elif ("split" in layout or "left" in layout) and logo_img is None:
            rule_w = max(6, int(base_dim * 0.012))
            cursor_y = max(40, (height - block_h) // 2)
            draw.rectangle(
                [margin_x, cursor_y, margin_x + rule_w, cursor_y + block_h],
                fill=accent_rgb,
            )
            margin_x = margin_x + rule_w + int(width * 0.03)
            max_text_w = max(100, width - margin_x - int(width * 0.08))
            title_lines = _wrap_lines(draw, text, title_font, max_text_w)
            sub_lines = _wrap_lines(draw, subtitle, sub_font, max_text_w) if subtitle else []
            align_left = True
        else:
            cursor_y = max(40, (height - block_h) // 2)
            if logo_img is not None:
                lx = (width - logo_img.width) // 2
                img.paste(logo_img, (lx, cursor_y), logo_img)
                cursor_y += logo_img.height + int(base_dim * 0.035)
                rule_w = int(width * 0.12)
                rule_x = (width - rule_w) // 2
                draw.rectangle(
                    [rule_x, cursor_y, rule_x + rule_w, cursor_y + accent_h], fill=accent_rgb
                )
                cursor_y += accent_h + int(base_dim * 0.035)
            else:
                rule_w = int(width * 0.14)
                rule_x = (width - rule_w) // 2
                rule_y = max(20, cursor_y - int(base_dim * 0.045))
                draw.rectangle(
                    [rule_x, rule_y, rule_x + rule_w, rule_y + accent_h], fill=accent_rgb
                )
            align_left = False

        for line in title_lines:
            bbox = draw.textbbox((0, 0), line, font=title_font)
            line_w = bbox[2] - bbox[0]
            x = margin_x if align_left else (width - line_w) // 2
            draw.text((x, cursor_y), line, fill=text_rgb, font=title_font)
            cursor_y += title_line_h

        if sub_lines:
            cursor_y += int(base_dim * 0.035)
            for line in sub_lines:
                bbox = draw.textbbox((0, 0), line, font=sub_font)
                line_w = bbox[2] - bbox[0]
                x = margin_x if align_left else (width - line_w) // 2
                draw.text((x, cursor_y), line, fill=text_rgb, font=sub_font)
                cursor_y += sub_line_h

        if footer and footer.strip():
            f_text = footer.strip()
            bbox = draw.textbbox((0, 0), f_text, font=footer_font)
            fw = bbox[2] - bbox[0]
            fx = margin_x if align_left else (width - fw) // 2
            fy = height - int(height * 0.08) - footer_size
            draw.text((fx, fy), f_text, fill=text_rgb, font=footer_font)

        return img

    async def render_card(
        self,
        text: str,
        width: int,
        height: int,
        fps: int,
        duration_sec: float,
        bible: CreativeBible,
        output_path: Path | str,
        *,
        subtitle: str | None = None,
        footer: str | None = None,
        card_kind: CardKind = "title",
        shot_id: str = "card",
        theme: GraphicsTheme | None = None,
        logo_path: Path | None = None,
        media_image_path: Path | None = None,
        word_timings: list[dict[str, object]] | None = None,
        kinetic: bool = True,
        cancel: CancellationToken | None = None,
    ) -> RenderedClip:
        """Render a motion-graphics card to an H.264 MP4 with exact width/height/fps/duration."""
        if width < 16 or height < 16:
            raise ValueError(f"Invalid card dimensions: {width}x{height}")
        if fps <= 0:
            raise ValueError(f"Invalid fps: {fps}")
        if duration_sec <= 0.0:
            raise ValueError(f"Invalid duration_sec: {duration_sec}")

        # Ensure even dimensions for yuv420p
        even_w = width if width % 2 == 0 else width + 1
        even_h = height if height % 2 == 0 else height + 1

        out = Path(output_path).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)

        if kinetic:
            try:
                from gtm_videogen.media.kinetic_motion import render_kinetic_card_mp4

                bg_rgb, text_rgb, accent_rgb = _pick_card_colors(bible, theme)
                return await render_kinetic_card_mp4(
                    text=text,
                    width=even_w,
                    height=even_h,
                    fps=fps,
                    duration_sec=duration_sec,
                    bible=bible,
                    output_path=out,
                    bg_rgb=bg_rgb,
                    text_rgb=text_rgb,
                    accent_rgb=accent_rgb,
                    ffmpeg_bin=self.ffmpeg_bin,
                    ffprobe_bin=self.ffprobe_bin,
                    fonts_dir=self.fonts_dir,
                    subtitle=subtitle,
                    footer=footer,
                    card_kind=card_kind,
                    shot_id=shot_id,
                    theme=theme,
                    logo_path=logo_path,
                    media_image_path=media_image_path,
                    word_timings=word_timings,
                    cancel=cancel,
                )
            except Exception:
                # Fall back to single-image zoompan if kinetic pipe ever hits an OS error
                pass

        tmp_dir = Path(tempfile.mkdtemp(prefix="gtm_card_"))
        try:
            png_path = tmp_dir / "card.png"
            tmp_mp4 = tmp_dir / "card.mp4"

            img = self._compose_card_image(
                text=text,
                width=even_w,
                height=even_h,
                bible=bible,
                subtitle=subtitle,
                footer=footer,
                card_kind=card_kind,
                theme=theme,
                logo_path=logo_path,
                media_image_path=media_image_path,
            )
            img.save(png_path, format="PNG")

            total_frames = max(1, int(round(duration_sec * fps)))
            zoom_step = 0.045
            vf_parts = [
                (
                    f"zoompan=z='1+{zoom_step:.3f}*(on/{total_frames})'"
                    f":x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
                    f":d={total_frames}:s={even_w}x{even_h}:fps={fps}"
                ),
                "format=yuv420p",
            ]

            cmd = [
                self.ffmpeg_bin,
                "-y",
                "-i",
                str(png_path),
                "-f",
                "lavfi",
                "-t",
                f"{duration_sec:.4f}",
                "-i",
                "anullsrc=r=44100:cl=stereo",
                "-t",
                f"{duration_sec:.4f}",
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-vf",
                ",".join(vf_parts),
                "-threads",
                str(max(1, int(os.getenv("GTM_FFMPEG_THREADS", "2") or "2"))),
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "18",
                "-profile:v",
                "high",
                "-level:v",
                "4.1",
                "-pix_fmt",
                "yuv420p",
                "-color_range",
                "tv",
                "-colorspace",
                "bt709",
                "-color_primaries",
                "bt709",
                "-color_trc",
                "bt709",
                "-video_track_timescale",
                "90000",
                "-r",
                str(fps),
                "-c:a",
                "aac",
                "-ar",
                "44100",
                "-ac",
                "2",
                "-b:a",
                "128k",
                "-shortest",
                "-movflags",
                "+faststart",
                str(tmp_mp4),
            ]
            await run_subprocess(cmd, cancel=cancel, timeout_sec=60.0)
            _move_output_atomically(tmp_mp4, out)

            probe = await probe_media(out, ffprobe_bin=self.ffprobe_bin, cancel=cancel)
            return RenderedClip(
                shot_id=shot_id,
                path=str(out),
                duration_sec=probe.duration_sec or duration_sec,
                fps=round(probe.fps) if probe.fps > 0 else fps,
                width=probe.width or even_w,
                height=probe.height or even_h,
                cache_hit=False,
                usd=0.0,
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
