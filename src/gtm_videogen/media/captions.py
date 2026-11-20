"""Dynamic ASS v4.00+ caption generation, WCAG 4.5:1 contrast, and bundled font resolution.

Addresses:
- F12: Driven by CreativeBible.typography_and_layout & 5 bundled open-license fonts.
- BUG-24: Proper :fontsdir= and FFmpeg filter path escaping; zero hardcoded OS font paths.
- Pillar 4: Zero hardcoded house styles; enforces ONE_TEXT_LAYER and WCAG 4.5:1 contrast
  while preserving intentional dark brand palettes.
- P5: Memoized font resolution via functools.lru_cache.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

from gtm_videogen.core.config import get_settings
from gtm_videogen.media._subprocess import aspect_to_dimensions
from gtm_videogen.models.schemas import CreativeBible, GraphicsTheme, ShotSheet

ONE_TEXT_LAYER = True

# 5 bundled open-license font families in assets/fonts/
BUNDLED_FONTS: dict[str, tuple[str, str]] = {
    "Roboto": ("Roboto-Regular.ttf", "Roboto-Bold.ttf"),
    "Roboto Condensed": ("RobotoCondensed-Regular.ttf", "RobotoCondensed-Bold.ttf"),
    "Liberation Serif": ("LiberationSerif-Regular.ttf", "LiberationSerif-Bold.ttf"),
    "Liberation Mono": ("LiberationMono-Regular.ttf", "LiberationMono-Bold.ttf"),
    "Liberation Sans": ("LiberationSans-Regular.ttf", "LiberationSans-Bold.ttf"),
}

# Semantic aliases mapping stylistic font requests onto the 5 bundled open-license families
_FONT_ALIASES: dict[str, str] = {
    "roboto": "Roboto",
    "plus jakarta sans": "Roboto",
    "inter": "Roboto",
    "sans": "Roboto",
    "geometric sans": "Roboto",
    "roboto condensed": "Roboto Condensed",
    "anton": "Roboto Condensed",
    "oswald": "Roboto Condensed",
    "condensed": "Roboto Condensed",
    "display": "Roboto Condensed",
    "liberation serif": "Liberation Serif",
    "instrument serif": "Liberation Serif",
    "serif": "Liberation Serif",
    "editorial serif": "Liberation Serif",
    "liberation mono": "Liberation Mono",
    "jetbrains mono": "Liberation Mono",
    "mono": "Liberation Mono",
    "monospace": "Liberation Mono",
    "liberation sans": "Liberation Sans",
    "caveat": "Liberation Sans",
    "humanist sans": "Liberation Sans",
    "neutral sans": "Liberation Sans",
}

_HEX_RE = re.compile(r"^#?([0-9a-fA-F]{6}|[0-9a-fA-F]{3})$")


@dataclass(frozen=True)
class CaptionCue:
    """Single timed text cue on the video timeline."""

    start_sec: float
    end_sec: float
    text: str
    layer_kind: Literal["caption", "overlay"] = "caption"
    shot_id: str = ""


def default_fonts_dir() -> Path:
    """Return the bundled assets/fonts directory."""
    settings_dir = get_settings().fonts_dir
    if settings_dir.exists():
        return settings_dir.resolve()
    repo_candidate = Path(__file__).resolve().parents[3] / "assets" / "fonts"
    return repo_candidate.resolve()


@lru_cache(maxsize=64)
def resolve_font_family(family: str | None) -> str:
    """Normalize a requested font family name to one of the 5 bundled open-license families."""
    if not family:
        return "Roboto"
    cleaned = " ".join(family.strip().lower().split())
    if cleaned in _FONT_ALIASES:
        return _FONT_ALIASES[cleaned]
    for key, canonical in _FONT_ALIASES.items():
        if key in cleaned:
            return canonical
    return "Roboto"


@lru_cache(maxsize=128)
def resolve_font_path(
    family: str | None,
    *,
    bold: bool = False,
    fonts_dir: str | None = None,
) -> Path:
    """Resolve a bundled TrueType font file path (memoized per process — P5)."""
    canonical = resolve_font_family(family)
    regular_file, bold_file = BUNDLED_FONTS[canonical]
    filename = bold_file if bold else regular_file
    base_dir = Path(fonts_dir).resolve() if fonts_dir else default_fonts_dir()
    candidate = base_dir / filename
    if candidate.exists():
        return candidate
    # Fallback to regular variant in the same family, then Roboto-Regular.ttf
    fallback = base_dir / regular_file
    if fallback.exists():
        return fallback
    roboto = base_dir / BUNDLED_FONTS["Roboto"][0]
    if roboto.exists():
        return roboto
    raise FileNotFoundError(f"Bundled font {filename!r} not found in {base_dir}")


def parse_hex_color(color: str) -> tuple[int, int, int]:
    """Parse a 3- or 6-digit hex color string into an (R, G, B) tuple."""
    match = _HEX_RE.match(color.strip())
    if not match:
        raise ValueError(f"Invalid hex color: {color!r}")
    hex_str = match.group(1)
    if len(hex_str) == 3:
        hex_str = "".join(ch * 2 for ch in hex_str)
    return (int(hex_str[0:2], 16), int(hex_str[2:4], 16), int(hex_str[4:6], 16))


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    r, g, b = (max(0, min(255, c)) for c in rgb)
    return f"#{r:02X}{g:02X}{b:02X}"


def relative_luminance(rgb: tuple[int, int, int]) -> float:
    """Compute WCAG 2.1 relative luminance for an sRGB (R, G, B) tuple."""

    def _linearize(channel: int) -> float:
        srgb = channel / 255.0
        return srgb / 12.92 if srgb <= 0.03928 else ((srgb + 0.055) / 1.055) ** 2.4

    r, g, b = rgb
    return 0.2126 * _linearize(r) + 0.7152 * _linearize(g) + 0.0722 * _linearize(b)


def contrast_ratio(fg_hex: str, bg_hex: str) -> float:
    """Return WCAG 2.1 contrast ratio (>= 1.0, up to 21.0) between two hex colors."""
    l1 = relative_luminance(parse_hex_color(fg_hex))
    l2 = relative_luminance(parse_hex_color(bg_hex))
    lighter, darker = max(l1, l2), min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)


def ensure_wcag_contrast(
    fg_hex: str,
    bg_hex: str,
    *,
    min_ratio: float = 4.5,
    clamp_background_luminance: bool = False,
) -> tuple[str, str]:
    """Ensure fg/bg meets WCAG contrast >= min_ratio while preserving dark brand palettes.

    By default (`clamp_background_luminance=False`), the background color is preserved
    whenever a high-contrast foreground (#FFFFFF or #111111) achieves >= min_ratio.
    Only mid-tone backgrounds where neither pure white nor near-black reaches min_ratio
    are nudged darker/lighter to guarantee readability.
    """
    fg_rgb = parse_hex_color(fg_hex)
    bg_rgb = parse_hex_color(bg_hex)
    fg_norm = _rgb_to_hex(fg_rgb)
    bg_norm = _rgb_to_hex(bg_rgb)

    if clamp_background_luminance and relative_luminance(bg_rgb) < 0.05:
        bg_rgb = tuple(min(255, int(c * 1.4 + 28)) for c in bg_rgb)  # type: ignore[assignment]
        bg_norm = _rgb_to_hex(bg_rgb)

    if contrast_ratio(fg_norm, bg_norm) >= min_ratio:
        return fg_norm, bg_norm

    white_ratio = contrast_ratio("#FFFFFF", bg_norm)
    dark_ratio = contrast_ratio("#111111", bg_norm)

    if white_ratio >= min_ratio or dark_ratio >= min_ratio:
        best_fg = "#FFFFFF" if white_ratio >= dark_ratio else "#111111"
        return best_fg, bg_norm

    # Mid-tone background where neither #FFFFFF nor #111111 reaches 4.5:1:
    # Darken the background slightly and use #FFFFFF so brand hue is preserved.
    adj_bg = list(bg_rgb)
    for _ in range(20):
        adj_bg = [max(0, int(c * 0.85)) for c in adj_bg]
        candidate_bg = _rgb_to_hex((adj_bg[0], adj_bg[1], adj_bg[2]))
        if contrast_ratio("#FFFFFF", candidate_bg) >= min_ratio:
            return "#FFFFFF", candidate_bg

    return "#FFFFFF", "#111111"


def hex_to_ass_color(hex_color: str, *, alpha: int = 0) -> str:
    """Convert #RRGGBB + alpha (0=opaque, 255=transparent) to ASS &HAABBGGRR."""
    r, g, b = parse_hex_color(hex_color)
    a = max(0, min(255, alpha))
    return f"&H{a:02X}{b:02X}{g:02X}{r:02X}"


def escape_ffmpeg_filter_path(path: Path | str) -> str:
    """Escape a filesystem path for FFmpeg's ass=/subtitles= filter syntax (BUG-24, BUG-C1).

    FFmpeg filtergraphs apply two levels of parsing:
    - Level 1 (avfilter_graph_parse2) parses filtergraph tokens (`\\`, `'`, `[`, `]`, `,`, `;`).
    - Level 2 (av_opt_set_from_string) parses filter option arguments (`\\`, `:`, `'`).
    Escaping Level 2 first and then Level 1 ensures colons, single quotes, spaces, brackets,
    commas, and semicolons in both `ass_path` and `fonts_dir` pass cleanly to libass.
    """
    raw = str(Path(path).resolve())
    # Level 2: filter argument escaping (\, :, ')
    escaped = raw.replace("\\", "\\\\")
    escaped = escaped.replace(":", "\\:")
    escaped = escaped.replace("'", "\\'")
    # Level 1: filtergraph escaping (\, ', [, ], ,, ;)
    escaped = escaped.replace("\\", "\\\\")
    escaped = escaped.replace("'", "\\'")
    escaped = escaped.replace("[", "\\[")
    escaped = escaped.replace("]", "\\]")
    escaped = escaped.replace(",", "\\,")
    escaped = escaped.replace(";", "\\;")
    return escaped


def build_ass_filter(ass_path: Path | str, fonts_dir: Path | str | None = None) -> str:
    """Build an FFmpeg ass=<path>:fontsdir=<dir> filter string with safe escaping (BUG-24)."""
    fdir = Path(fonts_dir) if fonts_dir is not None else default_fonts_dir()
    escaped_ass = escape_ffmpeg_filter_path(ass_path)
    escaped_fonts = escape_ffmpeg_filter_path(fdir)
    return f"ass={escaped_ass}:fontsdir={escaped_fonts}"


def _format_ass_time(seconds: float) -> str:
    clamped = max(0.0, seconds)
    total_cs = round(clamped * 100)
    cs = total_cs % 100
    total_s = total_cs // 100
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _escape_ass_text(text: str) -> str:
    cleaned = " ".join(text.strip().split())
    return cleaned.replace("\\", "\\\u200b").replace("{", "\\{").replace("}", "\\}")


def enforce_one_text_layer(cues: list[CaptionCue]) -> list[CaptionCue]:
    """Enforce ONE_TEXT_LAYER so no two text layers overlap simultaneously.

    Overlay cues (on_screen_text) take visual precedence over spoken caption cues during
    overlapping windows so captions and overlays never stack on top of each other.
    """
    valid = [c for c in cues if c.text.strip() and c.end_sec > c.start_sec + 0.02]
    overlays = sorted(
        [c for c in valid if c.layer_kind == "overlay"],
        key=lambda c: (c.start_sec, c.end_sec),
    )
    captions = sorted(
        [c for c in valid if c.layer_kind == "caption"],
        key=lambda c: (c.start_sec, c.end_sec),
    )

    # First ensure overlays themselves do not overlap
    resolved_overlays: list[CaptionCue] = []
    for ov in overlays:
        if resolved_overlays and ov.start_sec < resolved_overlays[-1].end_sec:
            prev = resolved_overlays[-1]
            if ov.end_sec <= prev.end_sec:
                continue
            ov = CaptionCue(
                start_sec=prev.end_sec,
                end_sec=ov.end_sec,
                text=ov.text,
                layer_kind="overlay",
                shot_id=ov.shot_id,
            )
        if ov.end_sec > ov.start_sec + 0.04:
            resolved_overlays.append(ov)

    # Subtract overlay intervals from caption intervals
    resolved_captions: list[CaptionCue] = []
    for cap in captions:
        segments: list[tuple[float, float]] = [(cap.start_sec, cap.end_sec)]
        for ov in resolved_overlays:
            next_segments: list[tuple[float, float]] = []
            for seg_start, seg_end in segments:
                if ov.end_sec <= seg_start or ov.start_sec >= seg_end:
                    next_segments.append((seg_start, seg_end))
                else:
                    if ov.start_sec > seg_start + 0.04:
                        next_segments.append((seg_start, ov.start_sec))
                    if ov.end_sec < seg_end - 0.04:
                        next_segments.append((ov.end_sec, seg_end))
            segments = next_segments

        for seg_start, seg_end in segments:
            if resolved_captions and seg_start < resolved_captions[-1].end_sec:
                seg_start = resolved_captions[-1].end_sec
            if seg_end > seg_start + 0.04:
                resolved_captions.append(
                    CaptionCue(
                        start_sec=seg_start,
                        end_sec=seg_end,
                        text=cap.text,
                        layer_kind="caption",
                        shot_id=cap.shot_id,
                    )
                )

    combined = resolved_overlays + resolved_captions
    combined.sort(key=lambda c: (c.start_sec, c.end_sec))
    return combined


def extract_cues_from_sheet(
    sheet: ShotSheet,
    *,
    include_vo_captions: bool = True,
) -> list[CaptionCue]:
    """Derive timed overlay and optional VO caption cues from a ShotSheet."""
    cues: list[CaptionCue] = []
    cursor = 0.0
    for shot in sorted(sheet.shots, key=lambda s: s.index):
        start = cursor
        end = cursor + shot.duration_sec
        cursor = end

        if shot.on_screen_text and shot.on_screen_text.strip():
            cues.append(
                CaptionCue(
                    start_sec=start,
                    end_sec=end,
                    text=shot.on_screen_text.strip(),
                    layer_kind="overlay",
                    shot_id=shot.id,
                )
            )
        if include_vo_captions and shot.vo_line and shot.vo_line.strip():
            cues.append(
                CaptionCue(
                    start_sec=start,
                    end_sec=end,
                    text=shot.vo_line.strip(),
                    layer_kind="caption",
                    shot_id=shot.id,
                )
            )
    return enforce_one_text_layer(cues) if ONE_TEXT_LAYER else cues


def _resolve_palette_colors(
    bible: CreativeBible,
    theme: GraphicsTheme | None = None,
) -> tuple[str, str, str]:
    """Resolve (text_hex, box_bg_hex, accent_hex) from CreativeBible + optional theme."""
    palette = [c for c in bible.visual_language.color_palette if _HEX_RE.match(c.strip())]

    bg_candidate = theme.background_color if theme else (palette[0] if palette else "#111827")
    if palette and theme is None:
        # Prefer the darkest palette color as the caption/overlay box background
        bg_candidate = min(palette, key=lambda c: relative_luminance(parse_hex_color(c)))

    fg_candidate = theme.text_color if theme else "#FFFFFF"
    accent_candidate = (
        theme.accent_color
        if theme
        else (palette[1] if len(palette) > 1 else (palette[0] if palette else "#3B82F6"))
    )

    clamp_bg = theme.clamp_background_luminance if theme else False
    fg_hex, bg_hex = ensure_wcag_contrast(
        fg_candidate,
        bg_candidate,
        min_ratio=4.5,
        clamp_background_luminance=clamp_bg,
    )
    return fg_hex, bg_hex, accent_candidate


def generate_ass_subtitles(
    sheet: ShotSheet,
    bible: CreativeBible,
    output_path: Path | str,
    *,
    width: int | None = None,
    height: int | None = None,
    cues: list[CaptionCue] | None = None,
    include_vo_captions: bool = True,
) -> Path:
    """Generate an ASS v4.00+ subtitle file driven by CreativeBible.typography_and_layout."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if width is None or height is None:
        w, h = aspect_to_dimensions(sheet.aspect, preview=False)
    else:
        w, h = width, height

    typo = bible.typography_and_layout
    font_family = resolve_font_family(typo.font_family or sheet.theme.font_family)
    fg_hex, bg_hex, _accent_hex = _resolve_palette_colors(bible, sheet.theme)

    primary_ass = hex_to_ass_color(fg_hex, alpha=0)
    # Opaque box background (BorderStyle=3) with subtle transparency (alpha=24) for legibility
    box_ass = hex_to_ass_color(bg_hex, alpha=24)
    outline_ass = hex_to_ass_color(bg_hex, alpha=0)

    treatment = (typo.caption_treatment or "clean_bottom").lower()
    if "top" in treatment:
        cap_alignment = 8
        margin_v = max(36, int(h * 0.07))
    elif "center" in treatment:
        cap_alignment = 5
        margin_v = max(24, int(h * 0.05))
    else:
        cap_alignment = 2
        margin_v = max(48, int(h * 0.08))

    overlay_alignment = 5 if "center" in treatment else 2
    cap_font_size = max(28, int(min(w, h) * 0.048))
    overlay_font_size = max(34, int(min(w, h) * 0.058))
    margin_lr = max(32, int(w * 0.06))

    resolved_cues = (
        enforce_one_text_layer(cues)
        if cues is not None
        else extract_cues_from_sheet(sheet, include_vo_captions=include_vo_captions)
    )

    lines = [
        "[Script Info]",
        "Title: GTM VideoGen Captions",
        "ScriptType: v4.00+",
        f"PlayResX: {w}",
        f"PlayResY: {h}",
        "WrapStyle: 2",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        (
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
            "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
            "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
            "Alignment, MarginL, MarginR, MarginV, Encoding"
        ),
        (
            f"Style: cap,{font_family},{cap_font_size},{primary_ass},{primary_ass},"
            f"{box_ass},{outline_ass},-1,0,0,0,100,100,0,0,3,10,0,"
            f"{cap_alignment},{margin_lr},{margin_lr},{margin_v},1"
        ),
        (
            f"Style: overlay,{font_family},{overlay_font_size},{primary_ass},{primary_ass},"
            f"{box_ass},{outline_ass},-1,0,0,0,100,100,0,0,3,14,0,"
            f"{overlay_alignment},{margin_lr},{margin_lr},{margin_v},1"
        ),
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    for cue in resolved_cues:
        style = "overlay" if cue.layer_kind == "overlay" else "cap"
        start_ts = _format_ass_time(cue.start_sec)
        end_ts = _format_ass_time(cue.end_sec)
        escaped_text = _escape_ass_text(cue.text)
        lines.append(f"Dialogue: 0,{start_ts},{end_ts},{style},,0,0,0,,{escaped_text}")

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


generate_ass_captions = generate_ass_subtitles
