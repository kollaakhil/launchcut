"""Launch-Grade 24fps Motion Graphics Engine & RGBA Overlay Renderer ($0/shot, Render-512MB safe).

Delivers OpenAI / Anthropic / Gemini-style crisp UI choreography, animated benchmark charts,
bento feature grids, terminal code reveals, split comparisons, and kinetic editorial typography
by streaming Pillow-composited RGB24 frames directly to FFmpeg stdin (`-threads 1`).

Zero headless browser overhead, guaranteed non-black frame 0, and bundled open-license fonts.
"""

from __future__ import annotations

import asyncio
import math
import os
import struct
import tempfile
import wave
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from gtm_videogen.core.job_registry import CancellationToken
from gtm_videogen.media._subprocess import probe_media
from gtm_videogen.media.captions import resolve_font_path
from gtm_videogen.media.ffmpeg_stitcher import _move_output_atomically
from gtm_videogen.media.kinetic_motion import (
    _blend_rgb,
    _draw_cursor,
    _ease_out_cubic,
    _ease_out_quint,
    _make_radial_backdrop,
    _wrap_words_to_lines,
)
from gtm_videogen.models.stitch_project import (
    MotionGraphicSpec,
    MotionThemeMode,
    SceneOverlaySpec,
)


def _theme_palette(
    theme_mode: MotionThemeMode,
    accent_rgb: tuple[int, int, int] | None = None,
) -> dict[str, tuple[int, int, int]]:
    """Return cohesive editorial/glass/light surface palettes."""
    if theme_mode == "warm_editorial":
        acc = accent_rgb or (217, 119, 87)  # Anthropic terracotta
        return {
            "bg": (245, 241, 234),
            "surface": (253, 251, 247),
            "surface_alt": (237, 232, 223),
            "border": (218, 210, 198),
            "text": (25, 24, 22),
            "muted": (110, 103, 95),
            "accent": acc,
            "accent_fg": (255, 255, 255),
            "ok": (28, 128, 86),
        }
    if theme_mode == "clean_light":
        acc = accent_rgb or (37, 99, 235)
        return {
            "bg": (248, 250, 252),
            "surface": (255, 255, 255),
            "surface_alt": (241, 245, 249),
            "border": (220, 228, 238),
            "text": (15, 23, 42),
            "muted": (100, 116, 139),
            "accent": acc,
            "accent_fg": (255, 255, 255),
            "ok": (16, 163, 127),
        }
    # dark_glass (OpenAI / Gemini keynote aesthetic)
    acc = accent_rgb or (78, 140, 255)
    return {
        "bg": (12, 14, 20),
        "surface": (22, 26, 36),
        "surface_alt": (30, 35, 48),
        "border": (56, 65, 84),
        "text": (244, 247, 252),
        "muted": (148, 160, 180),
        "accent": acc,
        "accent_fg": (255, 255, 255),
        "ok": (34, 197, 94),
    }


def _load_fonts(
    fonts_dir: Path | None,
    theme_mode: MotionThemeMode,
    h: int,
) -> dict[str, ImageFont.FreeTypeFont | ImageFont.ImageFont]:
    """Load bundled fonts scaled to output height."""
    display_family = "Liberation Serif" if theme_mode == "warm_editorial" else "Roboto"
    display_bold = resolve_font_path(display_family, bold=True, fonts_dir=fonts_dir)
    sans_reg = resolve_font_path("Roboto", bold=False, fonts_dir=fonts_dir)
    sans_bold = resolve_font_path("Roboto", bold=True, fonts_dir=fonts_dir)
    mono_reg = resolve_font_path("Liberation Mono", bold=False, fonts_dir=fonts_dir)
    mono_bold = resolve_font_path("Liberation Mono", bold=True, fonts_dir=fonts_dir)

    def _ft(p: Path, sz: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        try:
            return ImageFont.truetype(str(p), max(12, sz))
        except Exception:
            return ImageFont.load_default()

    return {
        "hero": _ft(display_bold, int(h * 0.082)),
        "title": _ft(display_bold, int(h * 0.056)),
        "sub": _ft(sans_reg, int(h * 0.031)),
        "body": _ft(sans_reg, int(h * 0.026)),
        "body_bold": _ft(sans_bold, int(h * 0.026)),
        "label": _ft(sans_bold, int(h * 0.020)),
        "mono": _ft(mono_reg, int(h * 0.025)),
        "mono_bold": _ft(mono_bold, int(h * 0.032)),
        "metric": _ft(sans_bold, int(h * 0.058)),
    }


def _draw_rounded_card(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    *,
    radius: int,
    fill: tuple[int, int, int],
    outline: tuple[int, int, int] | None = None,
    width: int = 2,
) -> None:
    draw.rounded_rectangle(
        list(box),
        radius=max(4, radius),
        fill=fill,
        outline=outline,
        width=max(1, width),
    )


def _draw_pill(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    *,
    bg: tuple[int, int, int],
    fg: tuple[int, int, int],
    pad_x: int = 16,
    pad_y: int = 7,
) -> tuple[int, int]:
    bb = draw.textbbox((0, 0), text, font=font)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    w, h = tw + pad_x * 2, th + pad_y * 2
    draw.rounded_rectangle([x, y, x + w, y + h], radius=h // 2, fill=bg)
    draw.text((x + pad_x, y + pad_y - bb[1] // 2), text, font=font, fill=fg)
    return w, h


def _draw_window_chrome(
    draw: ImageDraw.ImageDraw,
    x0: int,
    y0: int,
    x1: int,
    bar_h: int,
    *,
    pal: dict[str, tuple[int, int, int]],
    title: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> None:
    dot_r = max(4, bar_h // 5)
    cy = y0 + bar_h // 2
    for i, col in enumerate(((239, 107, 94), (242, 186, 73), (86, 194, 108))):
        cx = x0 + int(bar_h * 0.55) + i * int(dot_r * 3.1)
        draw.ellipse([cx - dot_r, cy - dot_r, cx + dot_r, cy + dot_r], fill=col)
    if title:
        bb = draw.textbbox((0, 0), title, font=font)
        tw = bb[2] - bb[0]
        draw.text(((x0 + x1 - tw) // 2, cy - (bb[3] - bb[1]) // 2 - 1), title, font=font, fill=pal["muted"])
    draw.line([(x0, y0 + bar_h), (x1, y0 + bar_h)], fill=pal["border"], width=1)


def _synthesize_ui_foley_wav(duration_sec: float, out_path: Path) -> Path:
    """Synthesize clean, subtle UI foley (soft keystroke ticks + warm harmonic chime, $0)."""
    sr = 24000
    n = max(sr, int(sr * duration_sec))
    frames = bytearray()
    chime_t = min(duration_sec * 0.42, 2.2)
    for i in range(n):
        t = i / sr
        fade_in = min(1.0, t / 0.25)
        fade_out = min(1.0, max(0.0, (duration_sec - t) / 0.5))
        # Subtle warm studio air pad
        pad = (
            math.sin(2.0 * math.pi * 174.61 * t) * 0.35
            + math.sin(2.0 * math.pi * 261.63 * t) * 0.25
        ) * 0.14
        # Soft harmonic chime on hero reveal
        chime = 0.0
        dt = t - chime_t
        if 0.0 <= dt <= 1.2:
            env = math.exp(-dt * 4.2)
            chime = (
                math.sin(2.0 * math.pi * 523.25 * dt) * 0.5
                + math.sin(2.0 * math.pi * 659.25 * dt) * 0.35
            ) * env * 0.28
        sample = (pad + chime) * fade_in * fade_out
        val = int(max(-32767, min(32767, sample * 16000.0)))
        frames.extend(struct.pack("<h", val))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(bytes(frames))
    return out_path


def _find_perspective_coeffs(
    src_quad: list[tuple[float, float]],
    dst_quad: list[tuple[float, float]],
) -> tuple[float, ...]:
    """Pure-Python 8x8 Gaussian elimination for PIL.Image.Transform.PERSPECTIVE (zero NumPy)."""
    matrix: list[list[float]] = []
    for (sx, sy), (dx, dy) in zip(src_quad, dst_quad, strict=True):
        matrix.append([dx, dy, 1.0, 0.0, 0.0, 0.0, -sx * dx, -sx * dy, sx])
        matrix.append([0.0, 0.0, 0.0, dx, dy, 1.0, -sy * dx, -sy * dy, sy])
    n = 8
    for i in range(n):
        pivot = max(range(i, n), key=lambda r: abs(matrix[r][i]))
        matrix[i], matrix[pivot] = matrix[pivot], matrix[i]
        div = matrix[i][i] or 1e-12
        for j in range(i, n + 1):
            matrix[i][j] /= div
        for r in range(n):
            if r != i:
                factor = matrix[r][i]
                for j in range(i, n + 1):
                    matrix[r][j] -= factor * matrix[i][j]
    return tuple(matrix[i][n] for i in range(n))


def _apply_living_camera_polish(
    frame: Image.Image,
    pal: dict[str, tuple[int, int, int]],
    t_sec: float,
    duration_sec: float,
) -> Image.Image:
    """Apply smooth sub-pixel continuous camera drift + live progress bar so holds never freeze."""
    w, h = frame.size
    p = max(0.0, min(1.0, t_sec / max(0.25, duration_sec)))
    # Smooth 3.0% continuous push-in + subtle floating pan using float EXTENT transform
    zoom = 1.0 + 0.030 * p
    view_w = w / zoom
    view_h = h / zoom
    pan_x = math.sin(t_sec * 0.95) * (w * 0.004)
    pan_y = math.cos(t_sec * 0.75) * (h * 0.003)
    cx = w * 0.5 + pan_x
    cy = h * 0.5 + pan_y
    x0 = max(0.0, min(w - view_w, cx - view_w * 0.5))
    y0 = max(0.0, min(h - view_h, cy - view_h * 0.5))
    drifted = frame.transform(
        (w, h),
        Image.Transform.EXTENT,
        (x0, y0, x0 + view_w, y0 + view_h),
        resample=Image.Resampling.BILINEAR,
    )
    draw = ImageDraw.Draw(drifted)
    bar_h = max(3, int(h * 0.0045))
    prog_w = max(8, int(w * max(0.03, p)))
    draw.rectangle([0, h - bar_h, prog_w, h], fill=pal["accent"])
    return drifted


def render_motion_template_frame(
    *,
    spec: MotionGraphicSpec,
    backdrop: Image.Image,
    fonts: dict[str, ImageFont.FreeTypeFont | ImageFont.ImageFont],
    pal: dict[str, tuple[int, int, int]],
    t_sec: float,
    duration_sec: float,
    brand_name: str = "",
    screenshot_img: Image.Image | None = None,
) -> Image.Image:
    """Render a single crisp aspect-aware frame with continuous living camera polish."""
    raw = _render_motion_template_frame_raw(
        spec=spec,
        backdrop=backdrop,
        fonts=fonts,
        pal=pal,
        t_sec=t_sec,
        duration_sec=duration_sec,
        brand_name=brand_name,
        screenshot_img=screenshot_img,
    )
    return _apply_living_camera_polish(raw, pal, t_sec, duration_sec)


def _render_motion_template_frame_raw(
    *,
    spec: MotionGraphicSpec,
    backdrop: Image.Image,
    fonts: dict[str, ImageFont.FreeTypeFont | ImageFont.ImageFont],
    pal: dict[str, tuple[int, int, int]],
    t_sec: float,
    duration_sec: float,
    brand_name: str = "",
    screenshot_img: Image.Image | None = None,
) -> Image.Image:
    """Render a single crisp aspect-aware frame at timestamp `t_sec` for any of the 10 launch templates."""
    frame = backdrop.copy()
    draw = ImageDraw.Draw(frame)
    w, h = frame.size
    is_portrait = h > w
    p = max(0.0, min(1.0, t_sec / max(0.25, duration_sec)))
    tpl = spec.template_id
    eyebrow = (spec.eyebrow or brand_name or "INTRODUCING").upper()[:36]
    headline = (spec.headline or brand_name or "Built for what comes next")[:90]
    subheadline = (spec.subheadline or "")[:130]

    if tpl == "device_showcase_25d":
        # Screen-Studio 2.5D tilted MacBook (16:9/1:1) or iPhone (9:16) showcase + cursor zoom
        pad_x = int(w * 0.07)
        _draw_pill(draw, pad_x, int(h * 0.06), eyebrow, fonts["label"], bg=_blend_rgb(pal["bg"], pal["accent"], 0.18), fg=pal["accent"])
        draw.text((pad_x, int(h * 0.115)), headline[:48], font=fonts["title"], fill=pal["text"])

        # Build crisp flat device card on RGBA canvas, then apply 2.5D perspective tilt + zoom
        dw = int(w * (0.76 if is_portrait else 0.72))
        dh = int(h * (0.64 if is_portrait else 0.64))
        dev = Image.new("RGBA", (dw, dh), (0, 0, 0, 0))
        dd = ImageDraw.Draw(dev)
        bezel_r = 38 if is_portrait else 22
        dd.rounded_rectangle([0, 0, dw - 1, dh - 1], radius=bezel_r, fill=(*pal["surface"], 255), outline=(*pal["accent"], 190), width=3)

        bar_h = max(30, int(dh * 0.08))
        if is_portrait:
            # iPhone Dynamic Island pill
            iw = int(dw * 0.28)
            dd.rounded_rectangle([(dw - iw) // 2, 14, (dw + iw) // 2, 30], radius=8, fill=(10, 12, 16, 255))
        else:
            _draw_window_chrome(dd, 4, 4, dw - 4, bar_h, pal=pal, title=(brand_name or "app").lower() + ".app", font=fonts["label"])

        inner_box = (14, bar_h + 10, dw - 14, dh - 14)
        inner_w = max(64, inner_box[2] - inner_box[0])
        inner_h = max(64, inner_box[3] - inner_box[1])

        if screenshot_img is not None:
            # Smooth Screen-Studio zoom (1.0x -> 1.28x) centered slightly toward active focal area
            zoom = 1.0 + 0.26 * _ease_out_quint(max(0.0, (p - 0.18) / 0.65))
            sw, sh = screenshot_img.size
            crop_w = max(64, int(sw / zoom))
            crop_h = max(64, int(sh / zoom))
            cx = int(sw * (0.5 + 0.06 * math.sin(p * math.pi)))
            cy = int(sh * (0.45 + 0.04 * math.cos(p * math.pi)))
            x0 = max(0, min(sw - crop_w, cx - crop_w // 2))
            y0 = max(0, min(sh - crop_h, cy - crop_h // 2))
            cropped = screenshot_img.crop((x0, y0, x0 + crop_w, y0 + crop_h)).resize((inner_w, inner_h), Image.Resampling.LANCZOS)
            dev.paste(cropped.convert("RGBA"), (inner_box[0], inner_box[1]))
        else:
            # Crisp synthetic interactive SaaS dashboard UI when no screenshot uploaded
            dd.rounded_rectangle(list(inner_box), radius=12, fill=(*pal["bg"], 255))
            dd.text((inner_box[0] + 24, inner_box[1] + 24), (brand_name or "Workspace")[:28], font=fonts["body_bold"], fill=(*pal["text"], 255))
            dd.text((inner_box[0] + 24, inner_box[1] + 60), (subheadline or headline)[:52], font=fonts["body"], fill=(*pal["muted"], 255))
            for ri, ln in enumerate(spec.response_lines[:3] or ["Live Brand Intelligence", "Automated 24fps Rendering", "Instant Multi-Format Export"]):
                ry = inner_box[1] + int(inner_h * 0.36) + ri * int(inner_h * 0.18)
                dd.rounded_rectangle([inner_box[0] + 24, ry, inner_box[2] - 24, ry + int(inner_h * 0.13)], radius=10, fill=(*pal["surface_alt"], 255), outline=(*pal["border"], 255), width=1)
                dd.text((inner_box[0] + 42, ry + 12), f"✦  {ln[:44]}", font=fonts["body_bold"], fill=(*pal["text"], 255))

        # Animated precision cursor sweeping onto device UI
        cur_e = _ease_out_cubic(min(1.0, max(0.0, (p - 0.12) / 0.45)))
        cx_dev = int(dw * (0.22 + 0.44 * cur_e))
        cy_dev = int(dh * (0.76 - 0.28 * cur_e))
        click_t = max(0.0, (p - 0.52) / 0.18) if p >= 0.52 else 0.0
        _draw_cursor(dd, cx_dev, cy_dev, max(16, int(dh * 0.045)), pal["accent"], click_t)

        # 2.5D Perspective tilt gently leveling as camera zooms in
        tilt_amt = (1.0 - _ease_out_quint(min(1.0, p / 0.75))) * (0.045 if is_portrait else 0.06)
        dx_t = int(dw * tilt_amt)
        dy_t = int(dh * tilt_amt * 0.65)
        src_q = [(0.0, 0.0), (float(dw), 0.0), (float(dw), float(dh)), (0.0, float(dh))]
        dst_q = [(float(dx_t), float(dy_t)), (float(dw - dx_t // 2), 0.0), (float(dw), float(dh)), (0.0, float(dh - dy_t))]
        coeffs = _find_perspective_coeffs(src_q, dst_q)
        warped = dev.transform((dw, dh), Image.Transform.PERSPECTIVE, coeffs, resample=Image.Resampling.BICUBIC)
        ox = (w - dw) // 2
        oy = int(h * 0.23)
        frame_rgba = frame.convert("RGBA")
        frame_rgba.alpha_composite(warped, (ox, oy))
        return frame_rgba.convert("RGB")

    if tpl == "node_flow_diagram":
        # Stripe-style living architecture node-and-connector pipeline with glowing packets
        pad_x = int(w * 0.08)
        _draw_pill(draw, pad_x, int(h * 0.09), eyebrow, fonts["label"], bg=_blend_rgb(pal["bg"], pal["accent"], 0.16), fg=pal["accent"])
        draw.text((pad_x, int(h * 0.155)), headline[:52], font=fonts["title"], fill=pal["text"])
        if subheadline:
            draw.text((pad_x, int(h * 0.225)), subheadline[:78], font=fonts["sub"], fill=pal["muted"])

        nodes = [str(m.get("label") or m.get("display") or "") for m in (spec.metrics[:3] if spec.metrics else []) if m]
        if len(nodes) < 3:
            nodes = [
                f"{brand_name or 'Brand'} Input",
                "Intelligent Engine",
                "Verified Output",
            ]
        active_node = int(p * 3.0) % 3
        if is_portrait:
            # Vertical stacked architecture for 9:16
            nw = int(w * 0.76)
            nh = int(h * 0.13)
            nx = (w - nw) // 2
            ys = [int(h * 0.34), int(h * 0.54), int(h * 0.74)]
            for i in range(2):
                y_top, y_bot = ys[i] + nh, ys[i + 1]
                draw.line([(w // 2, y_top), (w // 2, y_bot)], fill=pal["border"], width=4)
                pkt_p = (t_sec * 1.4 - i * 0.4) % 1.0
                py = int(y_top + (y_bot - y_top) * pkt_p)
                draw.ellipse([w // 2 - 7, py - 7, w // 2 + 7, py + 7], fill=pal["accent"])
            for i, lbl in enumerate(nodes[:3]):
                is_act = i == active_node
                _draw_rounded_card(draw, (nx, ys[i], nx + nw, ys[i] + nh), radius=18, fill=pal["surface"], outline=pal["accent"] if is_act else pal["border"], width=3 if is_act else 2)
                _draw_pill(draw, nx + 20, ys[i] + 18, f"0{i + 1}", fonts["label"], bg=pal["accent"] if is_act else pal["surface_alt"], fg=pal["accent_fg"] if is_act else pal["muted"])
                draw.text((nx + 78, ys[i] + nh // 2 - 14), lbl[:28], font=fonts["body_bold"], fill=pal["text"])
        else:
            # Horizontal 3-node Stripe pipeline for 16:9 / 1:1
            nw = int(w * 0.24)
            nh = int(h * 0.28)
            cy = int(h * 0.56)
            xs = [pad_x, (w - nw) // 2, w - pad_x - nw]
            for i in range(2):
                x_left, x_right = xs[i] + nw, xs[i + 1]
                draw.line([(x_left, cy), (x_right, cy)], fill=pal["border"], width=4)
                pkt_p = (t_sec * 1.35 - i * 0.35) % 1.0
                px = int(x_left + (x_right - x_left) * pkt_p)
                draw.ellipse([px - 8, cy - 8, px + 8, cy + 8], fill=pal["accent"])
            for i, lbl in enumerate(nodes[:3]):
                ne = _ease_out_quint((t_sec - 0.15 - i * 0.18) / 0.6)
                y_sh = int((1.0 - ne) * 28)
                is_act = i == active_node
                _draw_rounded_card(draw, (xs[i], cy - nh // 2 + y_sh, xs[i] + nw, cy + nh // 2 + y_sh), radius=20, fill=pal["surface"], outline=pal["accent"] if is_act else pal["border"], width=3 if is_act else 2)
                _draw_pill(draw, xs[i] + 20, cy - nh // 2 + y_sh + 20, f"STEP 0{i + 1}", fonts["label"], bg=_blend_rgb(pal["surface"], pal["accent"], 0.2), fg=pal["accent"])
                lns = _wrap_words_to_lines(draw, lbl, fonts["body_bold"], nw - 40)[:2]
                for li, lw in enumerate(lns):
                    draw.text((xs[i] + 20, cy - 6 + y_sh + li * 30), " ".join(lw), font=fonts["body_bold"], fill=pal["text"])
        return frame

    if tpl == "customer_quote_proof":
        # Editorial social proof quote card + animated ROI metric pill
        pad_x = int(w * 0.10)
        card_box = (pad_x, int(h * 0.16), w - pad_x, int(h * 0.84))
        _draw_rounded_card(draw, card_box, radius=24, fill=pal["surface"], outline=pal["border"], width=2)
        _draw_pill(draw, pad_x + 34, int(h * 0.21), eyebrow, fonts["label"], bg=_blend_rgb(pal["surface"], pal["accent"], 0.18), fg=pal["accent"])
        quote_txt = f"“{(spec.prompt_text or headline)[:110]}”"
        q_lines = _wrap_words_to_lines(draw, quote_txt, fonts["title"], (w - pad_x * 2) - 68)[:4]
        qe = _ease_out_quint(min(1.0, (t_sec - 0.15) / 0.7))
        for li, lw in enumerate(q_lines):
            draw.text(
                (pad_x + 34, int(h * 0.31 + (1.0 - qe) * 18) + li * int(h * 0.075)),
                " ".join(lw),
                font=fonts["title"],
                fill=_blend_rgb(pal["surface"], pal["text"], max(0.15, qe)),
            )
        attr = subheadline or f"Verified Customer · {brand_name or 'Enterprise Team'}"
        draw.text((pad_x + 34, int(h * 0.72)), f"— {attr[:62]}", font=fonts["body_bold"], fill=pal["muted"])
        stat_disp = str(spec.metrics[0].get("display") if spec.metrics else "+73% ROI")[:16]
        if p > 0.35:
            _draw_pill(draw, w - pad_x - 210, int(h * 0.70), stat_disp, fonts["body_bold"], bg=pal["accent"], fg=pal["accent_fg"], pad_x=20, pad_y=10)
        return frame

    if tpl == "kinetic_hook_reel":
        # High-contrast 1.5s pattern-interrupt word slam reveal for 9:16 social feeds
        words = (headline or "STOP SCROLLING").split()[:8]
        line_h = int(h * (0.095 if is_portrait else 0.12))
        lines = _wrap_words_to_lines(draw, " ".join(words), fonts["hero"], int(w * 0.82))[:4]
        total_h = len(lines) * line_h
        start_y = (h - total_h) // 2 - int(h * 0.04)
        w_idx = 0
        for li, lw in enumerate(lines):
            full = " ".join(lw)
            lbb = draw.textbbox((0, 0), full, font=fonts["hero"])
            cx = (w - (lbb[2] - lbb[0])) // 2
            y = start_y + li * line_h
            for word in lw:
                appear_t = 0.08 + w_idx * 0.13
                if t_sec >= appear_t:
                    we = _ease_out_quint((t_sec - appear_t) / 0.24)
                    wbb = draw.textbbox((0, 0), word + " ", font=fonts["hero"])
                    if w_idx == len(words) - 1:
                        # Accent slam pill on final punch word
                        draw.rounded_rectangle([cx - 12, y - 4, cx + (wbb[2] - wbb[0]) + 4, y + int(line_h * 0.84)], radius=14, fill=pal["accent"])
                        draw.text((cx, y + int((1.0 - we) * 16)), word, font=fonts["hero"], fill=pal["accent_fg"])
                    else:
                        draw.text((cx, y + int((1.0 - we) * 20)), word, font=fonts["hero"], fill=pal["text"])
                    cx += wbb[2] - wbb[0]
                w_idx += 1
        if subheadline and p > 0.45:
            _draw_pill(draw, (w - min(int(w * 0.76), 420)) // 2, start_y + total_h + 28, subheadline[:38], fonts["body_bold"], bg=pal["surface"], fg=pal["text"])
        return frame

    if tpl == "kinetic_headline":
        # Anthropic-style warm editorial masked word reveal
        ey_e = _ease_out_cubic(min(1.0, t_sec / 0.45))
        pill_bg = _blend_rgb(pal["bg"], pal["accent"], 0.16)
        bb = draw.textbbox((0, 0), eyebrow, font=fonts["label"])
        pw = (bb[2] - bb[0]) + 32
        _draw_pill(
            draw,
            (w - pw) // 2,
            int(h * 0.24 - (1.0 - ey_e) * 14),
            eyebrow,
            fonts["label"],
            bg=pill_bg,
            fg=pal["accent"],
        )
        lines = _wrap_words_to_lines(draw, headline, fonts["hero"], int(w * 0.76))[:3]
        line_h = int(h * 0.105)
        start_y = int(h * 0.36)
        word_idx = 0
        for li, words in enumerate(lines):
            full_line = " ".join(words)
            lbb = draw.textbbox((0, 0), full_line, font=fonts["hero"])
            cur_x = (w - (lbb[2] - lbb[0])) // 2
            y_base = start_y + li * line_h
            for word in words:
                wt = _ease_out_quint((t_sec - 0.12 - word_idx * 0.08) / 0.55)
                col = _blend_rgb(pal["bg"], pal["text"], max(0.15, wt))
                y_off = int((1.0 - wt) * (h * 0.04))
                draw.text((cur_x, y_base + y_off), word, font=fonts["hero"], fill=col)
                wbb = draw.textbbox((0, 0), word + " ", font=fonts["hero"])
                cur_x += wbb[2] - wbb[0]
                word_idx += 1
        # Expanding accent rule
        rule_e = _ease_out_quint((t_sec - 0.55) / 0.65)
        rule_w = int(w * 0.18 * rule_e)
        rule_y = start_y + len(lines) * line_h + int(h * 0.03)
        if rule_w > 2:
            draw.rounded_rectangle(
                [(w - rule_w) // 2, rule_y, (w + rule_w) // 2, rule_y + 4],
                radius=2,
                fill=pal["accent"],
            )
        if subheadline:
            sub_e = _ease_out_cubic((t_sec - 0.7) / 0.6)
            sub_col = _blend_rgb(pal["bg"], pal["muted"], sub_e)
            s_lines = _wrap_words_to_lines(draw, subheadline, fonts["sub"], int(w * 0.68))[:2]
            for si, sw in enumerate(s_lines):
                st = " ".join(sw)
                sbb = draw.textbbox((0, 0), st, font=fonts["sub"])
                draw.text(((w - (sbb[2] - sbb[0])) // 2, rule_y + 24 + si * int(h * 0.045)), st, font=fonts["sub"], fill=sub_col)
        return frame

    if tpl == "benchmark_bars":
        # Gemini / OpenAI animated horizontal benchmark bar chart
        pad_x = int(w * 0.10)
        head_e = _ease_out_cubic(min(1.0, t_sec / 0.45))
        _draw_pill(draw, pad_x, int(h * 0.11), eyebrow, fonts["label"], bg=_blend_rgb(pal["bg"], pal["accent"], 0.16), fg=pal["accent"])
        draw.text((pad_x, int(h * 0.175 + (1.0 - head_e) * 12)), headline[:54], font=fonts["title"], fill=pal["text"])
        if subheadline:
            draw.text((pad_x, int(h * 0.25)), subheadline[:80], font=fonts["sub"], fill=pal["muted"])

        metrics = spec.metrics[:4] if spec.metrics else [
            {"label": brand_name or "Flagship v2", "value": 94.2, "display": "94.2%", "highlight": True},
            {"label": "Previous Gen", "value": 71.5, "display": "71.5%", "highlight": False},
            {"label": "Industry Avg", "value": 58.0, "display": "58.0%", "highlight": False},
        ]
        max_val = max((float(m.get("value", 100) or 100) for m in metrics), default=100.0) or 100.0
        card_x0, card_x1 = pad_x, w - pad_x
        card_y0, card_y1 = int(h * 0.32), int(h * 0.86)
        _draw_rounded_card(draw, (card_x0, card_y0, card_x1, card_y1), radius=22, fill=pal["surface"], outline=pal["border"], width=2)

        row_h = (card_y1 - card_y0 - 48) // max(1, len(metrics))
        bar_left = card_x0 + int((card_x1 - card_x0) * 0.26)
        bar_right_max = card_x1 - int((card_x1 - card_x0) * 0.15)
        bar_h = max(18, min(42, int(row_h * 0.38)))
        for idx, m in enumerate(metrics):
            ry = card_y0 + 28 + idx * row_h
            bar_e = _ease_out_quint((t_sec - 0.25 - idx * 0.14) / 1.05)
            val = float(m.get("value", 50) or 50)
            is_hi = bool(m.get("highlight", idx == 0))
            lbl = str(m.get("label") or f"Model {idx + 1}")[:22]
            draw.text((card_x0 + 30, ry + bar_h // 4), lbl, font=fonts["body_bold"] if is_hi else fonts["body"], fill=pal["text"] if is_hi else pal["muted"])
            # Track
            draw.rounded_rectangle([bar_left, ry, bar_right_max, ry + bar_h], radius=bar_h // 2, fill=pal["surface_alt"])
            # Animated fill
            fill_w = max(bar_h, int((bar_right_max - bar_left) * (val / max_val) * max(0.04, bar_e)))
            fill_col = pal["accent"] if is_hi else _blend_rgb(pal["surface_alt"], pal["muted"], 0.45)
            draw.rounded_rectangle([bar_left, ry, bar_left + fill_w, ry + bar_h], radius=bar_h // 2, fill=fill_col)
            # Counter roll-up
            disp = str(m.get("display") or f"{val:.1f}")
            cur_num = val * bar_e
            if "%" in disp:
                num_txt = f"{cur_num:.1f}%"
            elif "x" in disp.lower():
                num_txt = f"{cur_num:.1f}×"
            else:
                num_txt = disp if bar_e > 0.92 else f"{int(cur_num)}"
            draw.text((bar_right_max + 18, ry + bar_h // 5), num_txt, font=fonts["body_bold"], fill=pal["accent"] if is_hi else pal["muted"])
        return frame

    if tpl == "bento_feature_grid":
        # 3-card glassmorphic bento grid with staggered spring pop + spotlight border
        pad_x = int(w * 0.08)
        _draw_pill(draw, pad_x, int(h * 0.10), eyebrow, fonts["label"], bg=_blend_rgb(pal["bg"], pal["accent"], 0.16), fg=pal["accent"])
        draw.text((pad_x, int(h * 0.165)), headline[:54], font=fonts["title"], fill=pal["text"])
        cards = spec.metrics[:3] if spec.metrics else [
            {"display": "10×", "label": "Instant Execution", "sub": "Built for real-time speed"},
            {"display": "99.9%", "label": "Precision Accuracy", "sub": "Verified across every step"},
            {"display": "Zero", "label": "Setup Friction", "sub": "Ready from your first prompt"},
        ]
        while len(cards) < 3:
            cards.append({"display": f"0{len(cards) + 1}", "label": "Flagship Craft", "sub": subheadline[:36] or "Engineered for scale"})
        gap = int((h if is_portrait else w) * 0.025)
        active_idx = int((t_sec / max(0.5, duration_sec)) * 3) % 3
        if is_portrait:
            ch = int((h * 0.62 - gap * 2) // 3)
            cw = w - pad_x * 2
            for i, c in enumerate(cards[:3]):
                ce = _ease_out_quint((t_sec - 0.12 - i * 0.16) / 0.6)
                cy0 = int(h * 0.25) + i * (ch + gap) + int((1.0 - ce) * 28)
                is_act = i == active_idx
                _draw_rounded_card(draw, (pad_x, cy0, pad_x + cw, cy0 + ch), radius=20, fill=pal["surface"], outline=pal["accent"] if is_act else pal["border"], width=3 if is_act else 2)
                disp = str(c.get("display") or c.get("value") or f"0{i + 1}")[:12]
                lbl = str(c.get("label") or f"Feature {i + 1}")[:28]
                sub = str(c.get("sub") or "")[:54]
                draw.text((pad_x + 24, cy0 + 20), disp, font=fonts["title"], fill=pal["accent"] if is_act else pal["text"])
                draw.text((pad_x + 24, cy0 + int(ch * 0.48)), lbl, font=fonts["body_bold"], fill=pal["text"])
                if sub:
                    draw.text((pad_x + 24, cy0 + int(ch * 0.72)), sub, font=fonts["body"], fill=pal["muted"])
        else:
            cw = (w - pad_x * 2 - gap * 2) // 3
            cy0, cy1 = int(h * 0.30), int(h * 0.84)
            for i, c in enumerate(cards[:3]):
                ce = _ease_out_quint((t_sec - 0.15 - i * 0.16) / 0.65)
                cx0 = pad_x + i * (cw + gap)
                y_shift = int((1.0 - ce) * (h * 0.07))
                is_act = i == active_idx
                border_c = pal["accent"] if is_act else pal["border"]
                _draw_rounded_card(draw, (cx0, cy0 + y_shift, cx0 + cw, cy1 + y_shift), radius=22, fill=pal["surface"], outline=border_c, width=3 if is_act else 2)
                disp = str(c.get("display") or c.get("value") or f"0{i + 1}")[:12]
                lbl = str(c.get("label") or f"Feature {i + 1}")[:28]
                sub = str(c.get("sub") or "")[:64]
                draw.text((cx0 + 26, cy0 + y_shift + 34), disp, font=fonts["metric"], fill=pal["accent"] if is_act else pal["text"])
                draw.text((cx0 + 26, cy0 + y_shift + int((cy1 - cy0) * 0.42)), lbl, font=fonts["body_bold"], fill=pal["text"])
                if sub:
                    s_lines = _wrap_words_to_lines(draw, sub, fonts["body"], cw - 52)[:3]
                    for si, sw in enumerate(s_lines):
                        draw.text((cx0 + 26, cy0 + y_shift + int((cy1 - cy0) * 0.56) + si * int(h * 0.036)), " ".join(sw), font=fonts["body"], fill=pal["muted"])
        return frame

    if tpl == "terminal_code":
        # Dark macOS-bezel terminal typing command + green status badge
        pad_x = int(w * 0.10)
        win_x0, win_y0 = pad_x, int(h * 0.14)
        win_x1, win_y1 = w - pad_x, int(h * 0.86)
        bar_h = max(36, int(h * 0.06))
        _draw_rounded_card(draw, (win_x0, win_y0, win_x1, win_y1), radius=20, fill=pal["surface"], outline=pal["border"], width=2)
        _draw_window_chrome(draw, win_x0, win_y0, win_x1, bar_h, pal=pal, title=f"{(brand_name or 'launch').lower()} — terminal", font=fonts["label"])

        cmd = spec.prompt_text or f"npx {(brand_name or 'launchcut').lower().replace(' ', '-')} deploy --production"
        out_lines = spec.response_lines[:4] if spec.response_lines else [
            f"✓ Connected to {brand_name or 'Cloud Engine'}",
            "✓ Compiled production pipeline in 140ms",
            "✓ Ready — live globally at 60fps",
        ]
        # Type command over first 45% of duration
        type_p = max(0.0, min(1.0, (t_sec - 0.15) / max(0.8, duration_sec * 0.42)))
        shown_cmd = cmd[: max(1, int(len(cmd) * type_p))]
        caret = "▌" if int(t_sec * 3.5) % 2 == 0 and type_p < 1.0 else ""
        ty = win_y0 + bar_h + 36
        draw.text((win_x0 + 34, ty), "$ ", font=fonts["mono_bold"], fill=pal["accent"])
        draw.text((win_x0 + 68, ty), shown_cmd + caret, font=fonts["mono_bold"], fill=pal["text"])

        # Reveal output lines sequentially after command completes
        out_start = duration_sec * 0.46
        for li, line in enumerate(out_lines):
            le = _ease_out_cubic((t_sec - out_start - li * 0.28) / 0.35)
            if le <= 0.02:
                continue
            ly = ty + int(h * 0.09) + li * int(h * 0.065)
            col = pal["ok"] if line.strip().startswith("✓") else pal["muted"]
            draw.text((win_x0 + 34, ly + int((1.0 - le) * 8)), line[:74], font=fonts["mono"], fill=_blend_rgb(pal["surface"], col, le))

        if p > 0.72:
            badge_e = _ease_out_quint((p - 0.72) / 0.20)
            _draw_pill(
                draw,
                win_x0 + 34,
                win_y1 - int(h * 0.10) + int((1.0 - badge_e) * 10),
                f"● {headline[:40]}",
                fonts["label"],
                bg=_blend_rgb(pal["surface"], pal["ok"], 0.22),
                fg=pal["ok"],
            )
        return frame

    if tpl == "split_comparison":
        # Before vs. After animated divider wipe
        pad_x = int(w * 0.07)
        draw.text((pad_x, int(h * 0.09)), headline[:56], font=fonts["title"], fill=pal["text"])
        y0, y1 = int(h * 0.22), int(h * 0.88)
        mid = w // 2
        gap = 18
        # Left card (Before)
        _draw_rounded_card(draw, (pad_x, y0, mid - gap // 2, y1), radius=22, fill=pal["surface_alt"], outline=pal["border"], width=2)
        _draw_pill(draw, pad_x + 26, y0 + 26, "BEFORE", fonts["label"], bg=pal["border"], fg=pal["muted"])
        before_lines = spec.response_lines[:3] if len(spec.response_lines) >= 2 else [
            "Manual multi-tool handoffs",
            "Slow turnaround & static templates",
            "Inconsistent brand look",
        ]
        for i, ln in enumerate(before_lines[:3]):
            draw.text((pad_x + 26, y0 + int((y1 - y0) * 0.26) + i * int(h * 0.09)), f"✕  {ln[:34]}", font=fonts["body"], fill=pal["muted"])

        # Right card (After — slides/wipes in with accent border)
        re_e = _ease_out_quint((t_sec - 0.35) / 0.75)
        rx0 = mid + gap // 2 + int((1.0 - re_e) * 48)
        _draw_rounded_card(draw, (rx0, y0, w - pad_x, y1), radius=22, fill=pal["surface"], outline=pal["accent"], width=3)
        _draw_pill(draw, rx0 + 26, y0 + 26, (brand_name or "AFTER").upper()[:20], fonts["label"], bg=pal["accent"], fg=pal["accent_fg"])
        after_lines = [str(m.get("label") or m.get("display") or "") for m in spec.metrics[:3] if m] or [
            "Unified instant workflow",
            "Studio-grade brand consistency",
            "Ship in minutes, not weeks",
        ]
        for i, ln in enumerate(after_lines[:3]):
            draw.text((rx0 + 26, y0 + int((y1 - y0) * 0.26) + i * int(h * 0.09)), f"✓  {ln[:34]}", font=fonts["body_bold"], fill=pal["text"])
        return frame

    # Default: prompt_ui_demo (OpenAI / Gemini live prompt -> cursor click -> response card / screenshot)
    pad_x = int(w * 0.09)
    win_x0, win_y0 = pad_x, int(h * 0.11)
    win_x1, win_y1 = w - pad_x, int(h * 0.89)
    bar_h = max(34, int(h * 0.055))
    _draw_rounded_card(draw, (win_x0, win_y0, win_x1, win_y1), radius=22, fill=pal["surface"], outline=pal["border"], width=2)
    _draw_window_chrome(draw, win_x0, win_y0, win_x1, bar_h, pal=pal, title=brand_name or eyebrow, font=fonts["label"])

    # Prompt input pill
    px0, py0 = win_x0 + 36, win_y0 + bar_h + 28
    px1, py1 = win_x1 - 36, py0 + int(h * 0.10)
    _draw_rounded_card(draw, (px0, py0, px1, py1), radius=16, fill=pal["surface_alt"], outline=pal["accent"] if p < 0.5 else pal["border"], width=2)
    prompt_txt = spec.prompt_text or headline
    type_p = max(0.0, min(1.0, (t_sec - 0.1) / max(0.7, duration_sec * 0.36)))
    shown_prompt = prompt_txt[: max(1, int(len(prompt_txt) * type_p))]
    caret = "|" if int(t_sec * 4) % 2 == 0 and type_p < 1.0 else ""
    draw.text((px0 + 22, py0 + (py1 - py0) // 3), shown_prompt[:62] + caret, font=fonts["body_bold"], fill=pal["text"])

    # Run button inside prompt bar
    btn_w, btn_h = int(w * 0.10), int((py1 - py0) * 0.64)
    bx0, by0 = px1 - btn_w - 12, py0 + (py1 - py0 - btn_h) // 2
    draw.rounded_rectangle([bx0, by0, bx0 + btn_w, by0 + btn_h], radius=btn_h // 2, fill=pal["accent"])
    draw.text((bx0 + btn_w // 4, by0 + btn_h // 4), "Run ✦", font=fonts["label"], fill=pal["accent_fg"])

    # Response area (either uploaded screenshot or staggered answer lines)
    resp_y0 = py1 + 24
    resp_y1 = win_y1 - 28
    resp_e = _ease_out_quint((t_sec - duration_sec * 0.42) / 0.65)
    if resp_e > 0.02:
        ry_shift = int((1.0 - resp_e) * 24)
        _draw_rounded_card(draw, (px0, resp_y0 + ry_shift, px1, resp_y1), radius=16, fill=pal["bg"], outline=pal["border"], width=1)
        if screenshot_img is not None:
            inner_w = max(64, (px1 - px0) - 24)
            inner_h = max(64, (resp_y1 - resp_y0 - ry_shift) - 24)
            resized = screenshot_img.copy()
            resized.thumbnail((inner_w, inner_h), Image.Resampling.LANCZOS)
            ox = px0 + ((px1 - px0) - resized.width) // 2
            oy = resp_y0 + ry_shift + ((resp_y1 - resp_y0 - ry_shift) - resized.height) // 2
            frame.paste(resized, (ox, oy))
        else:
            resp_lines = spec.response_lines[:4] if spec.response_lines else [
                headline[:68],
                subheadline[:68] if subheadline else f"Powered by {brand_name or 'next-gen intelligence'} — live in seconds.",
                "✓ Zero manual setup · Studio-grade output ready to ship",
            ]
            for li, ln in enumerate(resp_lines):
                le = _ease_out_cubic((t_sec - duration_sec * 0.44 - li * 0.22) / 0.35)
                if le <= 0.02:
                    continue
                col = pal["ok"] if ln.strip().startswith("✓") else (pal["text"] if li == 0 else pal["muted"])
                draw.text(
                    (px0 + 26, resp_y0 + ry_shift + 24 + li * int(h * 0.068)),
                    ln[:72],
                    font=fonts["body_bold"] if li == 0 else fonts["body"],
                    fill=_blend_rgb(pal["bg"], col, le),
                )

    # Animated precision cursor clicking Run right around t = 38%
    if 0.22 <= p <= 0.56:
        cur_p = (p - 0.22) / 0.22
        ce = _ease_out_cubic(min(1.0, cur_p))
        cx = int(win_x0 + (bx0 + btn_w // 2 - win_x0) * ce)
        cy = int(win_y1 - 40 + (by0 + btn_h // 2 - (win_y1 - 40)) * ce)
        click_t = max(0.0, (p - 0.40) / 0.14) if p >= 0.40 else 0.0
        _draw_cursor(draw, cx, cy, max(16, int(h * 0.032)), pal["accent"], click_t)

    return frame


async def render_motion_template_mp4(
    *,
    spec: MotionGraphicSpec,
    output_path: Path,
    width: int = 1280,
    height: int = 720,
    fps: int = 24,
    duration_sec: float = 5.0,
    brand_name: str = "",
    accent_rgb: tuple[int, int, int] | None = None,
    ffmpeg_bin: str = "ffmpeg",
    ffprobe_bin: str = "ffprobe",
    fonts_dir: Path | None = None,
    cancel: CancellationToken | None = None,
) -> Path:
    """Render any MotionGraphicSpec template directly to H.264 MP4 + subtle UI audio ($0)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dur = max(2.0, min(20.0, float(duration_sec)))
    total_frames = max(int(fps * dur), fps * 2)
    pal = _theme_palette(spec.theme_mode, accent_rgb=accent_rgb)

    # Render Pillow on <=1280px canvas and let FFmpeg SIMD bicubic scale to full 1080p (3x faster, <512MB RAM)
    max_dim = max(width, height)
    if max_dim > 1280:
        scale_r = 1280.0 / float(max_dim)
        rw = max(2, (int(round(width * scale_r)) // 2) * 2)
        rh = max(2, (int(round(height * scale_r)) // 2) * 2)
    else:
        rw, rh = width, height

    backdrop = _make_radial_backdrop(rw, rh, pal["bg"], pal["accent"], pal["text"])
    fonts = _load_fonts(fonts_dir, spec.theme_mode, rh)

    screenshot_img: Image.Image | None = None
    if spec.screenshot_path and Path(spec.screenshot_path).is_file():
        try:
            screenshot_img = Image.open(spec.screenshot_path).convert("RGB")
        except Exception:
            screenshot_img = None

    raw_t = os.getenv("GTM_FFMPEG_THREADS", "1").strip()
    eff_threads = max(1, int(raw_t)) if raw_t.isdigit() else 1
    vf_args = ["-vf", f"scale={width}:{height}:flags=bicubic"] if (rw != width or rh != height) else []

    tmp_dir = Path(tempfile.mkdtemp(prefix="motion_tpl_"))
    try:
        foley_wav = _synthesize_ui_foley_wav(dur, tmp_dir / "ui_foley.wav")
        tmp_mp4 = tmp_dir / "motion.mp4"
        cmd = [
            ffmpeg_bin,
            "-y",
            "-threads",
            str(eff_threads),
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{rw}x{rh}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-i",
            str(foley_wav),
            *vf_args,
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "19",
            "-pix_fmt",
            "yuv420p",
            "-r",
            str(fps),
            "-c:a",
            "aac",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-b:a",
            "128k",
            "-shortest",
            "-movflags",
            "+faststart",
            str(tmp_mp4),
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        assert proc.stdin is not None
        try:
            for fi in range(total_frames):
                if cancel is not None:
                    cancel.raise_if_cancelled()
                frame = render_motion_template_frame(
                    spec=spec,
                    backdrop=backdrop,
                    fonts=fonts,
                    pal=pal,
                    t_sec=fi / fps,
                    duration_sec=dur,
                    brand_name=brand_name,
                    screenshot_img=screenshot_img,
                )
                proc.stdin.write(frame.tobytes())
                del frame
                if fi % 4 == 0:
                    await proc.stdin.drain()
                    await asyncio.sleep(0)
            proc.stdin.close()
        except Exception:
            proc.kill()
            raise
        _, stderr_b = await proc.communicate()
        if proc.returncode != 0 or not tmp_mp4.is_file():
            err_tail = (stderr_b or b"").decode("utf-8", errors="ignore")[-400:]
            raise RuntimeError(f"Motion template FFmpeg render failed: {err_tail}")
        _move_output_atomically(tmp_mp4, output_path)
        await probe_media(output_path, ffprobe_bin=ffprobe_bin, cancel=cancel)
        return output_path
    finally:
        import shutil

        shutil.rmtree(tmp_dir, ignore_errors=True)


def render_overlay_png(
    *,
    overlay: SceneOverlaySpec,
    width: int,
    height: int,
    output_path: Path,
    accent_rgb: tuple[int, int, int] = (229, 77, 58),
    fonts_dir: Path | None = None,
) -> Path:
    """Render a 2x-supersampled transparent RGBA overlay PNG (lower-third, stat pill, callout)."""
    scale = 2
    cw, ch = width * scale, height * scale
    canvas = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    sans_bold = resolve_font_path("Roboto", bold=True, fonts_dir=fonts_dir)
    sans_reg = resolve_font_path("Roboto", bold=False, fonts_dir=fonts_dir)
    try:
        f_pri = ImageFont.truetype(str(sans_bold), max(22, int(ch * 0.032)))
        f_sec = ImageFont.truetype(str(sans_reg), max(18, int(ch * 0.023)))
    except Exception:
        f_pri = ImageFont.load_default()
        f_sec = ImageFont.load_default()

    pri = (overlay.primary_text or "Featured").strip()[:48]
    sec = (overlay.secondary_text or "").strip()[:60]
    pbb = draw.textbbox((0, 0), pri, font=f_pri)
    pw, ph = pbb[2] - pbb[0], pbb[3] - pbb[1]
    sbb = draw.textbbox((0, 0), sec, font=f_sec) if sec else (0, 0, 0, 0)
    sw, sh = (sbb[2] - sbb[0], sbb[3] - sbb[1]) if sec else (0, 0)

    pad_x = int(cw * 0.022)
    pad_y = int(ch * 0.018)
    accent_bar_w = int(cw * 0.006) if overlay.kind == "lower_third" else 0
    box_w = max(pw, sw) + pad_x * 2 + (accent_bar_w + 18 if accent_bar_w else 0)
    box_h = ph + (sh + 10 if sec else 0) + pad_y * 2

    margin_x = int(cw * 0.045)
    margin_y = int(ch * 0.065)
    if overlay.position == "top_right":
        x0, y0 = cw - margin_x - box_w, margin_y
    elif overlay.position == "bottom_center":
        x0, y0 = (cw - box_w) // 2, ch - margin_y - box_h
    else:
        x0, y0 = margin_x, ch - margin_y - box_h

    radius = box_h // 2 if overlay.kind == "stat_pill" else int(ch * 0.018)
    bg_rgba = (18, 20, 26, 224)
    border_rgba = (*accent_rgb, 210) if overlay.kind in ("stat_pill", "feature_callout") else (255, 255, 255, 42)
    draw.rounded_rectangle([x0, y0, x0 + box_w, y0 + box_h], radius=radius, fill=bg_rgba, outline=border_rgba, width=3)

    tx = x0 + pad_x
    if accent_bar_w:
        draw.rounded_rectangle(
            [x0 + pad_x // 2, y0 + pad_y // 2, x0 + pad_x // 2 + accent_bar_w, y0 + box_h - pad_y // 2],
            radius=accent_bar_w // 2,
            fill=(*accent_rgb, 255),
        )
        tx = x0 + pad_x // 2 + accent_bar_w + 18

    draw.text((tx, y0 + pad_y - pbb[1] // 2), pri, font=f_pri, fill=(250, 250, 252, 255))
    if sec:
        draw.text((tx, y0 + pad_y + ph + 8 - sbb[1] // 2), sec, font=f_sec, fill=(195, 202, 214, 240))

    final = canvas.resize((width, height), resample=Image.Resampling.LANCZOS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final.save(output_path, format="PNG")
    return output_path
