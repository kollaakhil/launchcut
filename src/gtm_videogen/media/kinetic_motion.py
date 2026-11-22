"""Procedural 24/30fps Kinetic Motion Graphics Engine ($0/shot, Render-512MB-safe).

Streams Pillow-composited RGB24 animation frames directly to FFmpeg stdin (`-threads 1`),
delivering Anthropic/OpenAI/Linear-style motion design with zero browser OOM overhead:
  1. Masked Word-by-Word Kinetic Typography (spring ease-out slide-up, optionally word-synced).
  2. Floating Dark-Bezel Product/UI Showcase with animated spotlight sweep & precision cursor.
  3. Animated Brand Logo Lockup with expanding accent rule and staggered subtitle reveal.

Always starts on the illuminated radial-studio backdrop from frame 0 (`blackdetect == 0.00s`).
"""

from __future__ import annotations

import asyncio
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Literal

from PIL import Image, ImageDraw, ImageFont

from gtm_videogen.core.job_registry import CancellationToken
from gtm_videogen.media._subprocess import probe_media
from gtm_videogen.media.captions import resolve_font_path
from gtm_videogen.media.ffmpeg_stitcher import _move_output_atomically
from gtm_videogen.models.schemas import CreativeBible, GraphicsTheme, RenderedClip

CardKind = Literal["title", "end_card", "lower_third", "metric"]


def _ease_out_cubic(t: float) -> float:
    clamped = max(0.0, min(1.0, t))
    inv = 1.0 - clamped
    return 1.0 - inv * inv * inv


def _ease_out_quint(t: float) -> float:
    clamped = max(0.0, min(1.0, t))
    inv = 1.0 - clamped
    return 1.0 - inv**5


def _blend_rgb(
    bg: tuple[int, int, int], fg: tuple[int, int, int], alpha: float
) -> tuple[int, int, int]:
    a = max(0.0, min(1.0, alpha))
    return (
        int(bg[0] * (1.0 - a) + fg[0] * a),
        int(bg[1] * (1.0 - a) + fg[1] * a),
        int(bg[2] * (1.0 - a) + fg[2] * a),
    )


def _make_radial_backdrop(
    width: int,
    height: int,
    bg_rgb: tuple[int, int, int],
    accent_rgb: tuple[int, int, int],
    text_rgb: tuple[int, int, int],
) -> Image.Image:
    """Create the illuminated editorial radial-studio backdrop + hairline frame once."""
    grad_w, grad_h = 160, 90
    grad_small = Image.new("RGB", (grad_w, grad_h), color=bg_rgb)
    gpx = grad_small.load()
    cx, cy = grad_w * 0.5, grad_h * 0.42
    max_r = (grad_w * grad_w + grad_h * grad_h) ** 0.5 * 0.65
    for gy in range(grad_h):
        for gx in range(grad_w):
            dist = ((gx - cx) ** 2 + (gy - cy) ** 2) ** 0.5
            t = max(0.0, 1.0 - dist / max_r)
            glow = (t * t) * 0.16
            vig = (1.0 - t) * 0.10
            r_c = int(max(0, min(255, bg_rgb[0] * (1.0 - vig) + accent_rgb[0] * glow)))
            g_c = int(max(0, min(255, bg_rgb[1] * (1.0 - vig) + accent_rgb[1] * glow)))
            b_c = int(max(0, min(255, bg_rgb[2] * (1.0 - vig) + accent_rgb[2] * glow)))
            gpx[gx, gy] = (r_c, g_c, b_c)
    img = grad_small.resize((width, height), resample=Image.Resampling.BILINEAR)
    draw = ImageDraw.Draw(img)
    frame_pad = max(18, int(min(width, height) * 0.045))
    border_rgb = _blend_rgb(bg_rgb, text_rgb, 0.22)
    draw.rectangle(
        [frame_pad, frame_pad, width - frame_pad, height - frame_pad],
        outline=border_rgb,
        width=max(1, int(min(width, height) * 0.002)),
    )
    return img


def _wrap_words_to_lines(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    max_width: int,
) -> list[list[str]]:
    words = text.split()
    if not words:
        return []
    lines: list[list[str]] = []
    cur: list[str] = []
    for w in words:
        trial = " ".join([*cur, w])
        bbox = draw.textbbox((0, 0), trial, font=font)
        if (bbox[2] - bbox[0]) <= max_width or not cur:
            cur.append(w)
        else:
            lines.append(cur)
            cur = [w]
    if cur:
        lines.append(cur)
    return lines


def _draw_cursor(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    size: int,
    accent_rgb: tuple[int, int, int],
    click_t: float,
) -> None:
    """Draw a sharp minimalist UI pointer + optional spring click ripple."""
    if 0.0 < click_t < 1.0:
        r = int(size * (0.4 + 1.8 * _ease_out_cubic(click_t)))
        ring_w = max(1, int(size * 0.14 * (1.0 - click_t)))
        draw.ellipse([x - r, y - r, x + r, y + r], outline=accent_rgb, width=ring_w)

    pts = [
        (x, y),
        (x, y + int(size * 1.15)),
        (x + int(size * 0.34), y + int(size * 0.86)),
        (x + int(size * 0.60), y + int(size * 1.32)),
        (x + int(size * 0.78), y + int(size * 1.22)),
        (x + int(size * 0.52), y + int(size * 0.76)),
        (x + int(size * 0.92), y + int(size * 0.72)),
    ]
    draw.polygon(pts, fill=(248, 248, 250), outline=(18, 20, 24))


async def render_kinetic_card_mp4(
    *,
    text: str,
    width: int,
    height: int,
    fps: int,
    duration_sec: float,
    bible: CreativeBible,
    output_path: Path,
    bg_rgb: tuple[int, int, int],
    text_rgb: tuple[int, int, int],
    accent_rgb: tuple[int, int, int],
    ffmpeg_bin: str = "ffmpeg",
    ffprobe_bin: str = "ffprobe",
    fonts_dir: Path | None = None,
    subtitle: str | None = None,
    footer: str | None = None,
    card_kind: CardKind = "title",
    shot_id: str = "card",
    theme: GraphicsTheme | None = None,
    logo_path: Path | None = None,
    media_image_path: Path | None = None,
    word_timings: list[dict[str, Any]] | None = None,
    cancel: CancellationToken | None = None,
) -> RenderedClip:
    """Render a multi-frame kinetic motion-graphics MP4 piped straight into FFmpeg stdin."""
    if cancel is not None:
        cancel.raise_if_cancelled()

    even_w = width if width % 2 == 0 else width + 1
    even_h = height if height % 2 == 0 else height + 1
    total_frames = max(1, int(round(duration_sec * fps)))
    base_dim = min(even_w, even_h)

    typo = bible.typography_and_layout
    font_family = typo.font_family or (theme.font_family if theme else "Roboto")
    bold_path = str(resolve_font_path(font_family, bold=True, fonts_dir=fonts_dir))
    reg_path = str(resolve_font_path(font_family, bold=False, fonts_dir=fonts_dir))

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

    title_font = ImageFont.truetype(bold_path, title_size)
    sub_font = ImageFont.truetype(reg_path, sub_size)
    footer_font = ImageFont.truetype(reg_path, footer_size)

    backdrop = _make_radial_backdrop(even_w, even_h, bg_rgb, accent_rgb, text_rgb)
    border_rgb = _blend_rgb(bg_rgb, text_rgb, 0.24)
    bezel_fill = tuple(max(0, int(c * 0.55)) for c in bg_rgb)

    # Pre-load optional brand logo / product UI screenshot once
    logo_img: Image.Image | None = None
    if logo_path is not None and Path(logo_path).is_file():
        try:
            loaded = Image.open(logo_path).convert("RGBA")
            loaded.thumbnail((int(even_w * 0.34), int(base_dim * 0.16)), Image.Resampling.LANCZOS)
            logo_img = loaded
        except Exception:
            logo_img = None

    media_img: Image.Image | None = None
    if media_image_path is not None and Path(media_image_path).is_file():
        try:
            loaded_m = Image.open(media_image_path).convert("RGBA")
            loaded_m.thumbnail((int(even_w * 0.58), int(even_h * 0.54)), Image.Resampling.LANCZOS)
            media_img = loaded_m
        except Exception:
            media_img = None

    # Pre-compute text layout metrics once
    probe_draw = ImageDraw.Draw(backdrop)
    margin_x = max(32, int(even_w * 0.09))
    max_text_w = max(100, even_w - 2 * margin_x)
    title_word_lines = _wrap_words_to_lines(probe_draw, text, title_font, max_text_w)
    sub_word_lines = (
        _wrap_words_to_lines(probe_draw, subtitle, sub_font, max_text_w) if subtitle else []
    )
    space_w = int(probe_draw.textbbox((0, 0), " ", font=title_font)[2])
    title_line_h = int(title_size * 1.24)
    sub_line_h = int(sub_size * 1.35)
    accent_h = max(4, int(base_dim * 0.008))

    # Build per-word start times (synced to Whisper word timestamps when provided!)
    word_starts: list[float] = []
    flat_words = [w for line in title_word_lines for w in line]
    if word_timings:
        for idx in range(len(flat_words)):
            if idx < len(word_timings) and "start" in word_timings[idx]:
                word_starts.append(max(0.0, float(word_timings[idx]["start"])))
            else:
                word_starts.append(0.06 + idx * 0.085)
    else:
        stagger = min(0.09, max(0.04, (duration_sec * 0.32) / max(1, len(flat_words))))
        word_starts = [0.04 + i * stagger for i in range(len(flat_words))]

    # Launch FFmpeg rawvideo encoder once and stream RGB24 frames cleanly
    out = Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="gtm_kcard_"))
    tmp_mp4 = tmp_dir / "kinetic.mp4"
    ff_threads = str(max(1, int(os.getenv("GTM_FFMPEG_THREADS", "2") or "2")))

    cmd = [
        ffmpeg_bin,
        "-y",
        "-threads",
        ff_threads,
        "-filter_threads",
        ff_threads,
        "-filter_complex_threads",
        ff_threads,
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{even_w}x{even_h}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
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
        "-threads",
        ff_threads,
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast" if (ff_threads == "1" or even_w <= 860) else "veryfast",
        "-crf",
        "19",
        *(
            ["-x264-params", "rc-lookahead=10:ref=1:bframes=0:sync-lookahead=0:threads=1"]
            if ff_threads == "1"
            else []
        ),
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
        "-b:a",
        "128k",
        "-ar",
        "44100",
        "-ac",
        "2",
        "-movflags",
        "+faststart",
        str(tmp_mp4),
    ]

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        assert proc.stdin is not None

        # Render animation frames and stream to FFmpeg stdin
        cap_font = ImageFont.truetype(bold_path, max(24, int(base_dim * 0.056)))
        cap_lines = (
            _wrap_words_to_lines(probe_draw, text, cap_font, int(even_w * 0.80))
            if media_img is not None
            else []
        )

        for f_idx in range(total_frames):
            if cancel is not None and f_idx % 8 == 0:
                cancel.raise_if_cancelled()
            t_sec = f_idx / float(fps)
            p_total = f_idx / float(max(1, total_frames - 1))

            frame = backdrop.copy()
            draw = ImageDraw.Draw(frame)
            # Subtle continuous top hairline progress shimmer so cards stay alive & never trip freezedetect
            inset = max(14, int(base_dim * 0.03))
            sweep_x0 = inset
            sweep_x1 = inset + max(8, int((even_w - 2 * inset) * max(0.04, p_total)))
            draw.line(
                [(sweep_x0, inset), (sweep_x1, inset)],
                fill=_blend_rgb(bg_rgb, accent_rgb, 0.55),
                width=max(2, int(base_dim * 0.004)),
            )

            if media_img is not None:
                # MODE 2: Floating Dark-Bezel Product/UI Showcase + Spotlight Sweep + Cursor
                mw, mh = media_img.size
                enter = _ease_out_cubic(t_sec / 0.48)
                drift_y = int((1.0 - enter) * (base_dim * 0.05) - p_total * (base_dim * 0.015))
                mx = (even_w - mw) // 2
                my = int(even_h * 0.11) + drift_y
                bezel = max(6, int(base_dim * 0.012))

                # Soft offset shadow + dark bezel
                shadow_off = max(6, int(base_dim * 0.014))
                draw.rectangle(
                    [
                        mx - bezel + shadow_off,
                        my - bezel + shadow_off,
                        mx + mw + bezel + shadow_off,
                        my + mh + bezel + shadow_off,
                    ],
                    fill=tuple(max(0, int(c * 0.35)) for c in bg_rgb),
                )
                draw.rectangle(
                    [mx - bezel, my - bezel, mx + mw + bezel, my + mh + bezel],
                    fill=bezel_fill,
                    outline=border_rgb,
                    width=max(1, int(base_dim * 0.0025)),
                )
                frame.paste(media_img, (mx, my), media_img)

                # Subtle animated top accent bar along bezel
                bar_w = int((mw + 2 * bezel) * _ease_out_quint(min(1.0, (t_sec + 0.05) / 0.65)))
                draw.rectangle(
                    [
                        mx - bezel,
                        my - bezel,
                        mx - bezel + bar_w,
                        my - bezel + max(3, accent_h // 2),
                    ],
                    fill=accent_rgb,
                )

                # Precision cursor gliding across the focal UI area + click ripple
                cur_p = _ease_out_cubic(
                    max(0.0, min(1.0, (t_sec - 0.30) / max(0.6, duration_sec * 0.45)))
                )
                cx0, cy0 = mx + int(mw * 0.22), my + int(mh * 0.72)
                cx1, cy1 = mx + int(mw * 0.66), my + int(mh * 0.44)
                cur_x = int(cx0 + (cx1 - cx0) * cur_p)
                cur_y = int(cy0 + (cy1 - cy0) * cur_p - math.sin(cur_p * math.pi) * (mh * 0.08))
                click_start = 0.30 + max(0.6, duration_sec * 0.45)
                click_t = (t_sec - click_start) / 0.42 if t_sec >= click_start else 0.0
                _draw_cursor(
                    draw, cur_x, cur_y, max(14, int(base_dim * 0.026)), accent_rgb, click_t
                )

                # Kinetic caption underneath the UI frame
                cy_txt = my + mh + bezel + int(even_h * 0.045)
                cap_alpha = _ease_out_cubic((t_sec - 0.14) / 0.36)
                cap_color = _blend_rgb(bg_rgb, text_rgb, max(0.25, cap_alpha))
                cap_dy = int((1.0 - cap_alpha) * (cap_font.size * 0.45))
                for line_words in cap_lines[:2]:
                    line_str = " ".join(line_words)
                    bbox = draw.textbbox((0, 0), line_str, font=cap_font)
                    lw = bbox[2] - bbox[0]
                    draw.text(
                        ((even_w - lw) // 2, cy_txt + cap_dy),
                        line_str,
                        fill=cap_color,
                        font=cap_font,
                    )
                    cy_txt += int(cap_font.size * 1.25)
            else:
                # MODE 1 & 3: Masked Word-by-Word Kinetic Typography & Animated Brand Logo Lockup
                logo_h = (logo_img.height + int(base_dim * 0.055)) if logo_img is not None else 0
                block_h = logo_h + len(title_word_lines) * title_line_h
                if sub_word_lines:
                    block_h += int(base_dim * 0.04) + len(sub_word_lines) * sub_line_h

                cursor_y = max(40, (even_h - block_h) // 2)
                if card_kind == "lower_third":
                    cursor_y = max(int(even_h * 0.58), even_h - block_h - int(even_h * 0.12))

                # Animated Brand Logo (spring scale + lift)
                if logo_img is not None:
                    lp = _ease_out_cubic(min(1.0, (t_sec + 0.05) / 0.42))
                    scale = 0.88 + 0.12 * lp
                    sw = max(8, int(logo_img.width * scale))
                    sh = max(8, int(logo_img.height * scale))
                    scaled_logo = logo_img.resize((sw, sh), Image.Resampling.BILINEAR)
                    lx = (even_w - sw) // 2
                    ly = cursor_y + int((1.0 - lp) * (base_dim * 0.025))
                    frame.paste(scaled_logo, (lx, ly), scaled_logo)
                    cursor_y += logo_img.height + int(base_dim * 0.032)

                # Expanding Brand Accent Hairline Rule
                rule_p = _ease_out_quint(min(1.0, (t_sec + 0.04) / 0.48))
                full_rule_w = int(even_w * (0.12 if logo_img is not None else 0.15))
                cur_rule_w = max(8, int(full_rule_w * rule_p))
                if card_kind == "lower_third":
                    draw.rectangle(
                        [
                            margin_x,
                            cursor_y - int(base_dim * 0.03),
                            margin_x + cur_rule_w,
                            cursor_y - int(base_dim * 0.03) + accent_h,
                        ],
                        fill=accent_rgb,
                    )
                else:
                    rx = (even_w - cur_rule_w) // 2
                    ry = (
                        cursor_y
                        if logo_img is not None
                        else max(20, cursor_y - int(base_dim * 0.045))
                    )
                    draw.rectangle([rx, ry, rx + cur_rule_w, ry + accent_h], fill=accent_rgb)
                    if logo_img is not None:
                        cursor_y += accent_h + int(base_dim * 0.032)

                # Staggered Word-by-Word Masked Spring Reveal
                w_idx = 0
                for line_words in title_word_lines:
                    line_str = " ".join(line_words)
                    line_w = int(draw.textbbox((0, 0), line_str, font=title_font)[2])
                    x = margin_x if card_kind == "lower_third" else (even_w - line_w) // 2
                    for w in line_words:
                        wb = draw.textbbox((0, 0), w, font=title_font)
                        ww = wb[2] - wb[0]
                        ws = word_starts[w_idx] if w_idx < len(word_starts) else 0.04
                        wp = _ease_out_cubic((t_sec - ws) / 0.32)
                        # Always visible at >= 0.18 alpha on frame 0 if early word, then snaps cleanly to 1.0
                        w_alpha = (
                            1.0
                            if t_sec >= ws + 0.32
                            else (wp if t_sec >= ws else (0.18 if w_idx == 0 else 0.0))
                        )
                        if w_alpha > 0.01:
                            dy = int((1.0 - wp) * (title_size * 0.38))
                            col = _blend_rgb(bg_rgb, text_rgb, w_alpha)
                            draw.text((x, cursor_y + dy), w, fill=col, font=title_font)
                        x += ww + space_w
                        w_idx += 1
                    cursor_y += title_line_h

                # Staggered Subtitle Reveal
                if sub_word_lines:
                    cursor_y += int(base_dim * 0.032)
                    sub_start = min(0.55, (word_starts[-1] if word_starts else 0.15) + 0.12)
                    sp = _ease_out_cubic((t_sec - sub_start) / 0.38)
                    if sp > 0.02:
                        s_col = _blend_rgb(bg_rgb, text_rgb, sp)
                        s_dy = int((1.0 - sp) * (sub_size * 0.35))
                        for s_words in sub_word_lines:
                            s_str = " ".join(s_words)
                            sw_px = int(draw.textbbox((0, 0), s_str, font=sub_font)[2])
                            sx = margin_x if card_kind == "lower_third" else (even_w - sw_px) // 2
                            draw.text((sx, cursor_y + s_dy), s_str, fill=s_col, font=sub_font)
                            cursor_y += sub_line_h

                if footer and footer.strip():
                    f_str = footer.strip()
                    fw_px = int(draw.textbbox((0, 0), f_str, font=footer_font)[2])
                    fx = margin_x if card_kind == "lower_third" else (even_w - fw_px) // 2
                    fy = even_h - int(even_h * 0.08) - footer_size
                    draw.text(
                        (fx, fy), f_str, fill=_blend_rgb(bg_rgb, text_rgb, 0.78), font=footer_font
                    )

            # Smooth 2.8% sub-pixel lens push-in across duration so cards feel cinematic & never freeze
            zoom_k = 1.0 + 0.028 * p_total
            vw_k = even_w / zoom_k
            vh_k = even_h / zoom_k
            ox_k = (even_w - vw_k) * 0.5 + math.sin(t_sec * 0.9) * (even_w * 0.003)
            oy_k = (even_h - vh_k) * 0.5 + math.cos(t_sec * 0.7) * (even_h * 0.0025)
            ox_k = max(0.0, min(even_w - vw_k, ox_k))
            oy_k = max(0.0, min(even_h - vh_k, oy_k))
            frame = frame.transform(
                (even_w, even_h),
                Image.Transform.EXTENT,
                (ox_k, oy_k, ox_k + vw_k, oy_k + vh_k),
                resample=Image.Resampling.BILINEAR,
            )
            raw_rgb = frame.tobytes()
            proc.stdin.write(raw_rgb)
            if f_idx % 4 == 0:
                await proc.stdin.drain()

        proc.stdin.close()
        _, stderr_bytes = await proc.communicate()
        if proc.returncode != 0:
            err_txt = stderr_bytes.decode("utf-8", errors="replace")[-600:]
            raise RuntimeError(f"Kinetic motion FFmpeg failed ({proc.returncode}): {err_txt}")

        _move_output_atomically(tmp_mp4, out)
        probe = await probe_media(out, ffprobe_bin=ffprobe_bin, cancel=cancel)
        return RenderedClip(
            shot_id=shot_id,
            engine="motion_card",
            path=str(out),
            duration_sec=round(probe.duration_sec or duration_sec, 4),
            width=probe.width or even_w,
            height=probe.height or even_h,
            fps=round(probe.fps or float(fps), 3),
            cache_hit=False,
            usd=0.0,
        )
    finally:
        import shutil

        shutil.rmtree(tmp_dir, ignore_errors=True)
