"""Stitch Director Engine: Pitching, Multi-Scene Script Compilation, Per-Act Takes & Master Stitch.

Powers all 4 stages of Stitch (Concept C · Tactile Director's Pinboard):
  1. Discover & Pitch: optional SSRF-safe URL scrape + 3 visual concept pitches.
  2. Pinboard & Script Gate: 1-4 chained Scenes (10s -> 60s) with structured subjects,
     materials, camera_spec, colors, beats, and locked end frames.
  3. Shoot & Takes: editable/uploadable Start & End anchor keyframes, @Element1..3
     identity lock, non-linear single-Act rendering with Take History, and pixel-exact
     Last-Frame Chaining (Act N final frame -> Act N+1 start frame).
  4. Screening Room & Master Stitch: <5s low-RAM FFmpeg master cut with 3 seam transitions
     (invisible_chain / micro_dissolve / audio_riser), 3 audio modes (native_only /
     native_plus_score / vo_plus_ducked_score), and optional +2.5s Kinetic Brand Outro.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import shutil
import struct
import tempfile
import wave
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from gtm_videogen.agents.director_studio import (
    _generate_flux_image,
    _hex_to_rgb,
    _normalize_act_clip,
    _render_mock_keyframe_jpeg,
    _strip_json_fences,
    _synthesize_ambient_cinematic_wav,
    encode_image_data_uri,
    extract_last_frame_jpeg,
)
from gtm_videogen.core.config import Settings, get_settings
from gtm_videogen.core.cost_meter import CostMeter
from gtm_videogen.core.http_client import ResilientHttpClient
from gtm_videogen.core.job_registry import CancellationToken
from gtm_videogen.core.media_store import MediaStore
from gtm_videogen.media._subprocess import aspect_to_dimensions, probe_media, run_subprocess
from gtm_videogen.media.audio_sync import detect_audio_beats
from gtm_videogen.media.captions import resolve_font_path
from gtm_videogen.media.ffmpeg_stitcher import _move_output_atomically, _pinned_h264_flags
from gtm_videogen.media.kinetic_motion import render_kinetic_card_mp4
from gtm_videogen.media.motion_templates import render_motion_template_mp4, render_overlay_png
from gtm_videogen.media.user_clip_processor import normalize_user_clip_take
from gtm_videogen.models.director_blueprint import (
    CAMERA_PRESETS,
    LIGHTING_PRESETS,
    MOTION_PRESETS,
    WORLD_PRESETS,
    CameraSpec,
    EnvironmentSpec,
    LightingSpec,
    MotionSpec,
    SubjectDetailsSpec,
)
from gtm_videogen.models.model_registry import ModelRegistryStore, VideoModelSpec
from gtm_videogen.models.schemas import (
    BeatSpec,
    CameraGrammar,
    CreativeBible,
    NarrativeStructure,
    SonicIdentity,
    TypographyAndLayout,
    VisualLanguage,
    VoiceCue,
)
from gtm_videogen.models.stitch_project import (
    STITCH_GALLERY_RECIPES,
    MotionGraphicSpec,
    MotionTemplateId,
    MotionThemeMode,
    SceneKind,
    SceneOverlaySpec,
    StitchAspect,
    StitchBeat,
    StitchElement,
    StitchProject,
    StitchScene,
    StitchTake,
    UserClipSpec,
    validate_stitch_project,
)
from gtm_videogen.providers.audio import AudioProvider
from gtm_videogen.providers.scraper import BrandScraper
from gtm_videogen.providers.transcription import transcribe_video_words
from gtm_videogen.providers.url_capture import capture_url_viewport_png, download_public_image
from gtm_videogen.providers.video import VideoProvider


class StitchProjectStore:
    """Atomic filesystem JSON store for Stitch projects under `<data_dir>/stitch`."""

    def __init__(self, root_dir: Path | str) -> None:
        self.root_dir = Path(root_dir).resolve()
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def project_dir(self, project_id: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9_-]", "", project_id)
        if not safe:
            raise ValueError("Invalid project_id")
        return self.root_dir / safe

    def media_dir(self, project_id: str) -> Path:
        p = self.project_dir(project_id) / "media"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def save(self, project: StitchProject) -> StitchProject:
        project.updated_at = datetime.now(UTC).isoformat()
        pdir = self.project_dir(project.project_id)
        pdir.mkdir(parents=True, exist_ok=True)
        target = pdir / "project.json"
        payload = project.model_dump_json(indent=2)
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=pdir,
            prefix=".project.",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            tmp.write(payload)
            tmp_path = Path(tmp.name)
        os.replace(tmp_path, target)
        return project

    def load(self, project_id: str) -> StitchProject | None:
        target = self.project_dir(project_id) / "project.json"
        if not target.is_file():
            return None
        return StitchProject.model_validate_json(target.read_text(encoding="utf-8"))

    def list_all(self, limit: int = 30) -> list[StitchProject]:
        items: list[StitchProject] = []
        if not self.root_dir.exists():
            return items
        for child in self.root_dir.iterdir():
            pfile = child / "project.json"
            if pfile.is_file():
                try:
                    items.append(
                        StitchProject.model_validate_json(pfile.read_text(encoding="utf-8"))
                    )
                except Exception:
                    continue
        items.sort(key=lambda p: p.updated_at, reverse=True)
        return items[:limit]


async def _call_openrouter_json(
    *,
    system_prompt: str,
    user_prompt: str,
    settings: Settings,
    llm_model_id: str = "",
    max_tokens: int = 6000,
    temperature: float = 0.65,
    cancel: CancellationToken | None = None,
) -> dict[str, Any] | None:
    """Call OpenRouter chat completions and parse a JSON object cleanly."""
    api_key = settings.openrouter_api_key.strip()
    if not api_key:
        return None
    base_url = "https://openrouter.ai/api/v1"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://launchcut.onrender.com",
        "X-Title": "Stitch Director Studio",
    }
    candidates: list[str] = []
    for m in (llm_model_id.strip(), settings.flagship_model, settings.text_model):
        if m and m not in candidates:
            candidates.append(m)

    async with ResilientHttpClient(
        connect_timeout=settings.http_connect_timeout_sec,
        read_timeout=65.0,
        write_timeout=20.0,
        pool_timeout=10.0,
        max_retries=1,
    ) as client:
        for model_name in candidates:
            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "response_format": {"type": "json_object"},
            }
            try:
                data = await client.post_json(
                    f"{base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                    cancel=cancel,
                )
                choices = data.get("choices") if isinstance(data, dict) else None
                if not isinstance(choices, list) or not choices:
                    continue
                content = str(choices[0].get("message", {}).get("content", "") or "")
                cleaned = _strip_json_fences(content)
                if not cleaned:
                    continue
                parsed = json.loads(cleaned)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                continue
    return None


async def scrape_optional_url(
    url: str | None,
    *,
    mock: bool = False,
    cancel: CancellationToken | None = None,
) -> dict[str, Any]:
    """Safely scrape a product/brand URL to enrich Stitch pitches & scripts."""
    clean = (url or "").strip()
    if not clean or mock:
        return {}
    if not clean.startswith(("http://", "https://")):
        clean = "https://" + clean
    try:
        scraper = BrandScraper(max_photos=4, max_product_pages=1)
        bundle = await asyncio.wait_for(
            scraper.scrape(clean, cancel=cancel),
            timeout=12.0,
        )
        return {
            "brand_name": bundle.brand_name or "",
            "tagline": bundle.tagline or "",
            "facts": list(bundle.facts[:6]),
            "colors": list(bundle.colors[:4]),
            "photos": [p.url for p in bundle.photos[:3]],
            "summary": " | ".join(
                p for p in [bundle.brand_name, bundle.tagline, *bundle.facts[:3]] if p
            )[:500],
        }
    except Exception:
        return {}


def _deterministic_pitches(idea: str, scraped: dict[str, Any]) -> list[dict[str, Any]]:
    clean = idea.strip() or (
        scraped.get("summary") or "A tactile miniature world awakening in warm morning sunlight"
    )
    brand_prefix = f"{scraped['brand_name']} — " if scraped.get("brand_name") else ""
    return [
        {
            "id": "pitch_diorama",
            "title": f"{brand_prefix}Tactile Tabletop Awakening",
            "angle": "ASMR Macro Diorama",
            "logline": (
                f"Ultra-photorealistic 90mm tilt-shift macro opening on {clean[:110]}, "
                "evolving in one unbroken liquid-gold tactile reveal."
            ),
            "idea": (
                f"{clean}. Shot as an intimate 90mm macro miniature diorama on warm aged oak, "
                "with volumetric golden shafts awakening tactile materials into a locked hero reveal."
            ),
            "world_preset": "miniature_diorama",
            "camera_preset": "macro_tilt_shift",
            "lighting_preset": "volumetric_gold",
            "motion_preset": "asmr_tactile",
            "palette": ["#191512", "#E54D3A", "#F6E7D4"],
        },
        {
            "id": "pitch_architectural",
            "title": f"{brand_prefix}Architectural Daylight Study",
            "angle": "Minimalist Editorial Film",
            "logline": (
                f"Crisp sunlit Roman travertine and brushed aluminum geometry celebrating {clean[:100]} "
                "with slow sculptural camera orbit."
            ),
            "idea": (
                f"{clean}. Filmed in an airy sunlit gallery on travertine stone and warm linen, "
                "with a smooth 360-degree sculptural orbit and razor-sharp optical glass reflections."
            ),
            "world_preset": "architectural_luxury",
            "camera_preset": "continuous_orbit",
            "lighting_preset": "crisp_daylight",
            "motion_preset": "epic_slowmo",
            "palette": ["#F5EFE6", "#1F1D1A", "#C98D49"],
        },
        {
            "id": "pitch_origami",
            "title": f"{brand_prefix}Living Paper-Craft Morph",
            "angle": "Continuous Physical Morph",
            "logline": (
                f"Handcrafted cotton paper folds and transforms autonomously around {clean[:100]}, "
                "pulling back into a luminous warm hero tableau."
            ),
            "idea": (
                f"{clean}. Begins on a flat textured 300gsm cotton sheet that folds autonomously "
                "like living origami into a glowing three-dimensional hero scene in golden hour light."
            ),
            "world_preset": "paper_origami",
            "camera_preset": "pullback_reveal",
            "lighting_preset": "golden_hour",
            "motion_preset": "seamless_morph",
            "palette": ["#231B15", "#E89B42", "#F8F3EA"],
        },
    ]


async def pitch_three_concepts(
    *,
    idea: str,
    source_url: str | None = None,
    llm_model_id: str = "",
    settings: Settings | None = None,
    mock: bool = False,
    cancel: CancellationToken | None = None,
) -> dict[str, Any]:
    """Generate 3 distinct, production-ready film concepts from a 1-line idea and optional URL."""
    resolved = settings or get_settings()
    scraped = await scrape_optional_url(source_url, mock=mock, cancel=cancel)
    clean_idea = idea.strip() or scraped.get("summary") or "A cinematic visual story"

    if not mock and resolved.openrouter_api_key.strip():
        sys_prompt = (
            "You are a world-class commercial film director. Given a creator's rough idea "
            "(and optional scraped product context), pitch EXACTLY 3 distinct, visual, "
            "filmable director treatments. Never use buzzwords or readable text on screen. "
            "Return ONLY a JSON object with key 'pitches' containing 3 objects, each having: "
            "id, title (3-6 words), angle (2-4 words), logline (1 vivid sentence), "
            "idea (2-3 concrete visual sentences), world_preset (one of: miniature_diorama, "
            "paper_origami, surreal_glass, architectural_luxury, cyberpunk_neon, nature_epic), "
            "camera_preset (one of: macro_tilt_shift, continuous_orbit, fpv_glide, pullback_reveal, overhead_topdown), "
            "lighting_preset (one of: volumetric_gold, crisp_daylight, golden_hour, neon_rain, moody_chiaroscuro), "
            "motion_preset (one of: asmr_tactile, seamless_morph, epic_slowmo, energetic_whip), "
            "palette (array of 3 hex colors)."
        )
        usr_prompt = json.dumps(
            {
                "idea": clean_idea,
                "scraped_context": scraped.get("summary", ""),
                "brand_name": scraped.get("brand_name", ""),
            },
            indent=2,
        )
        doc = await _call_openrouter_json(
            system_prompt=sys_prompt,
            user_prompt=usr_prompt,
            settings=resolved,
            llm_model_id=llm_model_id,
            max_tokens=2200,
            temperature=0.8,
            cancel=cancel,
        )
        raw_pitches = doc.get("pitches") if isinstance(doc, dict) else None
        if isinstance(raw_pitches, list) and len(raw_pitches) >= 3:
            normalized: list[dict[str, Any]] = []
            for i, p in enumerate(raw_pitches[:3]):
                if not isinstance(p, dict):
                    continue
                wp = str(p.get("world_preset", "miniature_diorama"))
                cp = str(p.get("camera_preset", "macro_tilt_shift"))
                lp = str(p.get("lighting_preset", "volumetric_gold"))
                mp = str(p.get("motion_preset", "asmr_tactile"))
                normalized.append(
                    {
                        "id": str(p.get("id") or f"pitch_{i + 1}"),
                        "title": str(p.get("title") or f"Concept {i + 1}")[:64],
                        "angle": str(p.get("angle") or "Director Treatment")[:36],
                        "logline": str(p.get("logline") or clean_idea)[:220],
                        "idea": str(p.get("idea") or clean_idea)[:500],
                        "world_preset": wp if wp in WORLD_PRESETS else "miniature_diorama",
                        "camera_preset": cp if cp in CAMERA_PRESETS else "macro_tilt_shift",
                        "lighting_preset": lp if lp in LIGHTING_PRESETS else "volumetric_gold",
                        "motion_preset": mp if mp in MOTION_PRESETS else "asmr_tactile",
                        "palette": (
                            [str(c) for c in p.get("palette", [])[:3]]
                            if isinstance(p.get("palette"), list) and len(p.get("palette")) >= 3
                            else ["#191512", "#E54D3A", "#F6E7D4"]
                        ),
                    }
                )
            if len(normalized) == 3:
                return {"pitches": normalized, "scraped": scraped}

    return {"pitches": _deterministic_pitches(clean_idea, scraped), "scraped": scraped}


def _plan_scene_durations(target_sec: int, model_spec: VideoModelSpec) -> list[int]:
    """Split target film duration (10s -> 60s) into 1-4 clean Scene durations supported by the model."""
    allowed = sorted(model_spec.durations_sec or [5, 8, 10, 15])
    max_single = max(allowed)
    target = max(5, min(60, int(target_sec)))

    if target <= max_single:
        return [model_spec.snap_duration(target)]

    # Pick 2, 3, or 4 scenes so total stays close to target
    count = max(2, min(4, math.ceil(target / float(min(15, max_single)))))
    per = model_spec.snap_duration(target / float(count))
    return [per] * count


def _make_beats_for_duration(
    duration_sec: int,
    *,
    scene_title: str,
    idea: str,
    subjects: list[str],
    materials: list[str],
    camera_spec: str,
    continues: bool,
    prev_end_frame: str,
) -> list[StitchBeat]:
    """Create 3-4 well-paced 2-4s beats for a single Scene."""
    n_beats = 3 if duration_sec <= 8 else (4 if duration_sec <= 12 else 5)
    step = duration_sec / float(n_beats)
    subj_str = ", ".join(subjects) or "the hero centerpiece"
    mat_str = ", ".join(materials) or "tactile physical surfaces"
    beats: list[StitchBeat] = []
    for i in range(n_beats):
        t0 = int(round(i * step))
        t1 = int(round((i + 1) * step))
        if i == n_beats - 1:
            t1 = duration_sec
        tr = f"{t0}-{t1}s"
        if i == 0:
            if continues and prev_end_frame:
                sc_txt = (
                    f"Continues seamlessly from the previous scene's locked frame ({prev_end_frame[:95]}). "
                    f"{subj_str} catches fresh light across {mat_str}."
                )
            else:
                sc_txt = f"Opening composition of {scene_title}: {subj_str} resting on {mat_str} with natural depth."
            act_txt = f"Subtle micro-motion begins as warm light glides across {mat_str}, initiating the {idea[:85]} story."
            cam_txt = f"Smooth opening {camera_spec}, zero jitter."
            snd_txt = "Soft room tone and crisp tactile contact foley."
        elif i == n_beats - 1:
            sc_txt = f"Final hero composition holding {subj_str} cleanly centered with balanced negative space."
            act_txt = "Motion gently decelerates into a poised, luminous hero hold with floating dust motes."
            cam_txt = "Controlled slow deceleration into a rock-steady locked-off frame."
            snd_txt = "Warm harmonic swell fading smoothly into crisp silence."
        else:
            sc_txt = f"Dynamic evolution of {subj_str} showcasing intricate {mat_str} grain and golden caustics."
            act_txt = f"Fluid physical transformation advances {idea[:90]} with authentic momentum and reflections."
            cam_txt = f"Continuous fluid glide ({camera_spec}) tracking the focal action."
            snd_txt = "Layered ASMR material texture and subtle resonant whoosh."
        beats.append(
            StitchBeat(time=tr, scene=sc_txt, action=act_txt, camera=cam_txt, sound=snd_txt)
        )
    return beats


def _find_gallery_recipe(recipe_id: str | None) -> dict[str, Any] | None:
    if not recipe_id:
        return None
    for rec in STITCH_GALLERY_RECIPES:
        if rec.get("id") == recipe_id:
            return rec
    return None


def _default_graphic_spec(
    template_id: MotionTemplateId,
    *,
    brand: str,
    idea: str,
    scraped: dict[str, Any],
    world_id: str,
) -> MotionGraphicSpec:
    """Build a brand-tailored MotionGraphicSpec for any of the 10 launch templates."""
    if world_id == "stripe_living_flow":
        theme_mode: MotionThemeMode = "clean_light"
    elif world_id in ("anthropic_editorial", "notion_playful", "airbnb_human_story"):
        theme_mode = "warm_editorial"
    else:
        theme_mode = "dark_glass"
    hero = brand or "LaunchCut"
    tagline = str(scraped.get("tagline") or "").strip()
    facts = [str(f).strip() for f in (scraped.get("facts") or []) if str(f).strip()]
    clean_idea = idea.strip()

    if template_id == "device_showcase_25d":
        return MotionGraphicSpec(
            template_id="device_showcase_25d",
            theme_mode=theme_mode,
            eyebrow=f"{hero.upper()} LIVE TOUR",
            headline=tagline[:60] or f"Inside {hero}",
            subheadline=clean_idea[:90],
            response_lines=[
                facts[0][:44] if len(facts) > 0 else f"Live {hero} interface in 2.5D motion",
                facts[1][:44] if len(facts) > 1 else "Smooth cursor-tracked focus zoom",
                facts[2][:44] if len(facts) > 2 else "Built for instant product clarity",
            ],
        )
    if template_id == "node_flow_diagram":
        return MotionGraphicSpec(
            template_id="node_flow_diagram",
            theme_mode=theme_mode,
            eyebrow=f"{hero.upper()} ARCHITECTURE",
            headline=f"How {hero} works end-to-end",
            subheadline=tagline[:86] or clean_idea[:86],
            metrics=[
                {"label": f"{hero} Input"},
                {"label": facts[0][:24] if facts else "Automated Engine"},
                {"label": "Production Output"},
            ],
        )
    if template_id == "customer_quote_proof":
        return MotionGraphicSpec(
            template_id="customer_quote_proof",
            theme_mode=theme_mode,
            eyebrow="VERIFIED CUSTOMER PROOF",
            headline=f"{hero} transformed how our team ships.",
            prompt_text=facts[0][:100] if facts else f"Switching to {hero} cut our turnaround time from weeks to minutes.",
            subheadline=f"Head of Product · {hero} Customer",
            metrics=[{"display": "+73% Faster"}],
        )
    if template_id == "kinetic_hook_reel":
        hook_words = " ".join((clean_idea or f"Meet {hero} built for speed").split()[:6]).upper()
        return MotionGraphicSpec(
            template_id="kinetic_hook_reel",
            theme_mode=theme_mode,
            eyebrow=hero.upper(),
            headline=hook_words[:56],
            subheadline=tagline[:42] or f"Discover {hero}",
        )
    if template_id == "kinetic_headline":
        return MotionGraphicSpec(
            template_id="kinetic_headline",
            theme_mode=theme_mode,
            eyebrow=f"INTRODUCING {hero.upper()}",
            headline=tagline[:72] or clean_idea[:72] or f"Meet {hero}",
            subheadline=facts[0][:110] if facts else clean_idea[:110],
        )
    if template_id == "benchmark_bars":
        return MotionGraphicSpec(
            template_id="benchmark_bars",
            theme_mode=theme_mode,
            eyebrow="PERFORMANCE BENCHMARK",
            headline=f"{hero} leads on speed & quality",
            subheadline=facts[0][:90] if facts else clean_idea[:90],
            metrics=[
                {"label": hero[:20], "value": 96.4, "display": "96.4%", "highlight": True},
                {"label": "Previous Gen", "value": 72.1, "display": "72.1%", "highlight": False},
                {"label": "Legacy Flow", "value": 48.5, "display": "48.5%", "highlight": False},
            ],
        )
    if template_id == "bento_feature_grid":
        f1 = facts[0][:38] if len(facts) > 0 else "Instant brand adaptation"
        f2 = facts[1][:38] if len(facts) > 1 else "Studio-grade visual lock"
        f3 = facts[2][:38] if len(facts) > 2 else "Ready to ship in minutes"
        return MotionGraphicSpec(
            template_id="bento_feature_grid",
            theme_mode=theme_mode,
            eyebrow=f"WHY {hero.upper()}",
            headline=f"Everything {hero} unlocks",
            subheadline=tagline[:90] or clean_idea[:90],
            metrics=[
                {"display": "10×", "label": "Faster Launch", "sub": f1},
                {"display": "4K", "label": "Crisp Precision", "sub": f2},
                {"display": "1-Click", "label": "Ready to Ship", "sub": f3},
            ],
        )
    if template_id == "terminal_code":
        slug = re.sub(r"[^a-z0-9]+", "-", hero.lower()).strip("-") or "app"
        return MotionGraphicSpec(
            template_id="terminal_code",
            theme_mode="dark_glass",
            eyebrow="DEVELOPER EXPERIENCE",
            headline=f"{hero} live in one command",
            prompt_text=f"npx {slug} launch --theme studio",
            response_lines=[
                f"✓ Authenticated {hero} workspace",
                f"✓ {facts[0][:62]}" if facts else f"✓ Compiled {clean_idea[:52]}",
                "✓ Deployed globally — ready in 120ms",
            ],
        )
    if template_id == "split_comparison":
        return MotionGraphicSpec(
            template_id="split_comparison",
            theme_mode=theme_mode,
            eyebrow="THE SHIFT",
            headline=f"Before vs. With {hero}",
            response_lines=[
                "Fragmented manual workflows",
                "Days of back-and-forth edits",
                "Generic off-brand output",
            ],
            metrics=[
                {"label": facts[0][:34] if facts else f"Unified {hero} pipeline"},
                {"label": facts[1][:34] if len(facts) > 1 else "Crisp on-brand motion design"},
                {"label": "Ship launch-ready in minutes"},
            ],
        )
    # prompt_ui_demo
    return MotionGraphicSpec(
        template_id="prompt_ui_demo",
        theme_mode=theme_mode,
        eyebrow=f"{hero.upper()} LIVE DEMO",
        headline=tagline[:68] or f"{hero} in action",
        subheadline=clean_idea[:90],
        prompt_text=clean_idea[:64] or f"Launch {hero} for our flagship release",
        response_lines=[
            tagline[:68] or f"{hero}: {clean_idea[:54]}",
            f"✓ {facts[0][:64]}" if facts else "✓ Instant visual choreography generated",
            f"✓ {facts[1][:64]}" if len(facts) > 1 else "✓ Studio-grade output ready to publish",
        ],
    )


def _build_deterministic_scenes(
    *,
    idea: str,
    durations: list[int],
    recipe_id: str | None,
    world_data: dict[str, Any],
    camera_data: dict[str, Any],
    lighting_data: dict[str, Any],
    scraped: dict[str, Any],
) -> tuple[str, str, list[StitchScene]]:
    """Build rich, structured, editable StitchScenes offline or as fallback."""
    clean_idea = idea.strip() or "A tactile miniature world awakening on an aged oak table"
    brand = str(scraped.get("brand_name") or "").strip()
    palette = list(lighting_data["lighting"].get("color_palette") or ["#191512", "#E8A838", "#F6E7D4"])
    if scraped.get("colors"):
        palette = list(scraped["colors"][:3])
    while len(palette) < 3:
        palette.append("#F6E7D4")

    mats_raw = [m.strip() for m in str(world_data["environment"].get("materials", "")).split(",") if m.strip()]
    base_mats = mats_raw[:3] or ["aged oak wood", "optical borosilicate glass", "brushed brass"]
    base_cam = f"{camera_data['camera'].get('lens_mm', 85)}mm {camera_data['camera'].get('style', 'macro cinema')} · {camera_data['camera'].get('movement', 'smooth continuous glide')}"

    is_jar_sample = (
        recipe_id == "miniature_jar_world"
        and not brand
        and (not idea.strip() or "jar" in clean_idea.lower() or "miniature" in clean_idea.lower())
    )
    if is_jar_sample:
        title = "Miniature World in a Jar"
        main_subject = "a tiny living miniature city awakened by glowing liquid gold inside a glass jar"
        arc_specs = [
            (
                "Scene 1 · Awakening Drop",
                ["giant realistic human hand", "glass pipette with liquid gold", "frozen gray miniature city"],
                ["glowing molten gold", "weathered stone streets", "crystal pipette glass"],
                "90mm macro tilt-shift · extreme close-up slowly pushing in",
                "Glowing liquid gold ripples outward across the waking miniature avenue",
                "One drop of gold wakes a silent world.",
            ),
            (
                "Scene 2 · Folding Horizon",
                ["living miniature town", "glowing amber rooftops", "tiny illuminated trains"],
                ["aged oak desk", "warm amber lantern light", "handcrafted miniature timber"],
                "90mm macro tilt-shift · smooth continuous orbital arc",
                "The unfolding miniature valley settles into a radiant glowing tabletop village",
                "Every street folds upward into living warmth.",
            ),
            (
                "Scene 3 · Sealed in Glass",
                ["crystal apothecary glass jar", "human fingers closing cork lid", "glowing miniature world"],
                ["borosilicate glass", "natural cork stopper", "golden volumetric dust motes"],
                "85mm studio cinema · sweeping continuous pull-back reveal",
                "Locked-off hero shot of the illuminated living world sealed inside the crystal jar",
                "A complete universe, sealed in glass.",
            ),
            (
                "Scene 4 · Quiet Afterglow",
                ["glowing glass jar hero centerpiece", "sunlit artisan workshop"],
                ["warm oak tabletop", "optical glass caustics", "soft floating dust"],
                "85mm locked-off macro · gentle 2% breathing push",
                "Serene locked hero still with glowing amber caustics across the oak desk",
                "Crafted to hold its own light.",
            ),
        ]
    else:
        world_label = str(world_data.get("label") or "Cinematic").strip()
        short_title = " ".join(clean_idea.split()[:5]).rstrip(".,;:")
        title = f"{brand}: {short_title}" if brand else (short_title.title() or "Untitled Film")
        main_subject = (
            f"{brand} — {clean_idea[:110]} (reimagined in {world_label})"
            if brand
            else f"{clean_idea[:120]} ({world_label})"
        )
        hero_ref = brand or "hero centerpiece"
        arc_specs = [
            (
                f"Scene 1 · {world_label} Hook",
                [hero_ref, f"{base_mats[0]} sculpted foreground"],
                base_mats,
                base_cam,
                f"{hero_ref} emerging cleanly inside a {world_label.lower()} world ({clean_idea[:65]})",
                f"Start with the quiet craft of {brand or 'every detail'}.",
            ),
            (
                f"Scene 2 · {world_label} Transformation",
                [hero_ref, f"living {base_mats[0]} & light interplay"],
                [*base_mats[:2], "volumetric golden caustics"],
                "65mm cinema glide · smooth continuous sculptural orbit",
                f"Mid-arc {world_label.lower()} peak showcasing {hero_ref} with luminous depth",
                "Light moves across every surface with effortless precision.",
            ),
            (
                "Scene 3 · Hero Pull-Back Reveal",
                [brand or "flagship hero centerpiece", f"complete {world_label.lower()} tableau"],
                [*base_mats[:2], base_mats[-1]],
                "85mm cinema lens · sweeping continuous pull-back into hero lock",
                f"Locked-off hero composition framing {hero_ref} in {world_label.lower()} craft with balanced negative space",
                "Pull back to see the entire story lock into place.",
            ),
            (
                "Scene 4 · Signature Hold",
                [hero_ref],
                base_mats,
                "85mm locked-off cinema frame · subtle breathing focus",
                f"Unforgettable {world_label.lower()} hero finish with glowing ambient depth",
                f"{brand or 'Designed'} to stay with you.",
            ),
        ]

    recipe = _find_gallery_recipe(recipe_id)
    hybrid_plan: list[str] = list(recipe.get("hybrid_plan") or []) if recipe else []
    world_id = str(world_data.get("id") or "miniature_diorama")

    scenes: list[StitchScene] = []
    for idx, dur in enumerate(durations):
        spec = arc_specs[min(idx, len(arc_specs) - 1)]
        sc_title, subjs, mats, cam_s, end_f, vo_l = spec
        if idx >= len(arc_specs):
            sc_title = f"Scene {idx + 1} · Continuation"

        sc_kind: SceneKind = "ai_sequence"
        graphic_spec: MotionGraphicSpec | None = None
        user_clip: UserClipSpec | None = None
        overlays: list[SceneOverlaySpec] = []

        if idx < len(hybrid_plan):
            token = hybrid_plan[idx]
            if token.startswith("motion_graphic:"):
                tpl_id = token.split(":", 1)[1]
                sc_kind = "motion_graphic"
                graphic_spec = _default_graphic_spec(
                    tpl_id,  # type: ignore[arg-type]
                    brand=brand,
                    idea=clean_idea,
                    scraped=scraped,
                    world_id=world_id,
                )
                sc_title = f"Act {idx + 1} · {graphic_spec.template_id.replace('_', ' ').title()}"
            elif token == "user_clip":
                sc_kind = "user_clip"
                user_clip = UserClipSpec(filename="", raw_video_path="", source_duration_sec=float(dur))
                sc_title = f"Act {idx + 1} · Founder / Product Video"
                if brand:
                    overlays.append(
                        SceneOverlaySpec(
                            overlay_id=f"ov_{idx + 1}",
                            kind="lower_third",
                            primary_text=brand,
                            secondary_text=str(scraped.get("tagline") or "Product Launch")[:48],
                            position="bottom_left",
                            start_sec=0.5,
                            end_sec=min(float(dur) - 0.5, 4.5),
                        )
                    )

        continues = idx > 0 and sc_kind == "ai_sequence" and scenes[idx - 1].kind == "ai_sequence"
        prev_end = scenes[idx - 1].end_frame if continues else ""
        beats = _make_beats_for_duration(
            dur,
            scene_title=sc_title,
            idea=clean_idea,
            subjects=list(subjs),
            materials=list(mats),
            camera_spec=cam_s,
            continues=continues,
            prev_end_frame=prev_end,
        )
        scenes.append(
            StitchScene(
                scene_id=f"sc_{idx + 1}_{uuid4().hex[:6]}",
                index=idx,
                title=sc_title,
                duration_sec=dur,
                kind=sc_kind,
                continues_from_prev=continues,
                subjects=list(subjs),
                materials=list(mats),
                camera_spec=cam_s,
                colors=list(palette[:3]),
                vo_line=vo_l,
                end_frame=end_f,
                beats=beats,
                graphic_spec=graphic_spec,
                user_clip=user_clip,
                overlays=overlays,
            )
        )
    return title[:72], main_subject[:220], scenes


async def compile_stitch_project(
    *,
    idea: str,
    target_duration_sec: int = 15,
    aspect_ratio: StitchAspect = "16:9",
    video_model_id: str = "fal-ai/kling-video/v3/pro/text-to-video",
    llm_model_id: str = "",
    world_preset: str = "miniature_diorama",
    camera_preset: str = "macro_tilt_shift",
    lighting_preset: str = "volumetric_gold",
    motion_preset: str = "asmr_tactile",
    source_url: str | None = None,
    recipe_id: str | None = None,
    brand_name: str | None = None,
    brand_tagline: str | None = None,
    registry_store: ModelRegistryStore | None = None,
    settings: Settings | None = None,
    media_dir: Path | None = None,
    mock: bool = False,
    cancel: CancellationToken | None = None,
) -> StitchProject:
    """Compile a user idea + optional URL into a structured multi-Scene StitchProject."""
    resolved = settings or get_settings()
    model_spec = (
        registry_store.get_video_model(video_model_id)
        if registry_store is not None
        else ModelRegistryStore(resolved.data_dir / "stitch").get_video_model(video_model_id)
    )
    world_data = WORLD_PRESETS.get(world_preset, WORLD_PRESETS["miniature_diorama"])
    camera_data = CAMERA_PRESETS.get(camera_preset, CAMERA_PRESETS["macro_tilt_shift"])
    lighting_data = LIGHTING_PRESETS.get(lighting_preset, LIGHTING_PRESETS["volumetric_gold"])
    motion_data = MOTION_PRESETS.get(motion_preset, MOTION_PRESETS["asmr_tactile"])

    scraped = await scrape_optional_url(source_url, mock=mock, cancel=cancel)
    eff_brand = (brand_name or "").strip() or scraped.get("brand_name") or None
    eff_tagline = (brand_tagline or "").strip() or scraped.get("tagline") or None
    durations = _plan_scene_durations(target_duration_sec, model_spec)

    title, main_subject, scenes = _build_deterministic_scenes(
        idea=idea,
        durations=durations,
        recipe_id=recipe_id,
        world_data=world_data,
        camera_data=camera_data,
        lighting_data=lighting_data,
        scraped=scraped,
    )

    project_id = f"st_{uuid4().hex[:10]}"
    lighting_spec = LightingSpec.model_validate(lighting_data["lighting"])
    if scraped.get("colors"):
        lighting_spec.color_palette = list(scraped["colors"][:3])

    project = StitchProject(
        project_id=project_id,
        title=title,
        idea=idea.strip() or main_subject,
        source_url=(source_url or "").strip() or None,
        scraped_context=str(scraped.get("summary") or ""),
        recipe_id=recipe_id,
        aspect_ratio=aspect_ratio,
        target_duration_sec=sum(durations),
        video_model_id=model_spec.id,
        llm_model_id=llm_model_id.strip(),
        world_preset=world_preset if world_preset in WORLD_PRESETS else "miniature_diorama",
        camera_preset=camera_preset if camera_preset in CAMERA_PRESETS else "macro_tilt_shift",
        lighting_preset=lighting_preset if lighting_preset in LIGHTING_PRESETS else "volumetric_gold",
        motion_preset=motion_preset if motion_preset in MOTION_PRESETS else "asmr_tactile",
        style=str(world_data.get("style", "ultra-photorealistic cinematic film")),
        main_subject=main_subject,
        camera=CameraSpec.model_validate(camera_data["camera"]),
        environment=EnvironmentSpec.model_validate(world_data["environment"]),
        lighting=lighting_spec,
        subject_details=SubjectDetailsSpec.model_validate(world_data.get("subject_details") or {}),
        motion=MotionSpec.model_validate(motion_data["motion"]),
        scenes=scenes,
        brand_name=eff_brand,
        brand_tagline=eff_tagline,
        add_brand_outro=bool(eff_brand),
        enable_captions=(recipe_id == "viral_vertical_reel"),
        status="draft",
        status_message=f"Drafted {len(scenes)} Scene(s) ({sum(durations)}s). Review & edit on your Pinboard.",
    )

    # Auto-Brand-Kit Injection: capture live URL viewport + scraped hero photo into @Element1 & 2.5D UI scenes
    if source_url and source_url.strip():
        kit_dir = media_dir if media_dir is not None else (resolved.data_dir / "stitch" / project_id / "media")
        kit_parts: list[str] = []
        if scraped.get("colors"):
            kit_parts.append("Palette")
        if eff_brand:
            kit_parts.append("Outro")
        if not mock:
            kit_dir.mkdir(parents=True, exist_ok=True)
            cap_task = capture_url_viewport_png(source_url, kit_dir / "brand_viewport.png", cancel=cancel)
            photos = list(scraped.get("photos") or [])
            hero_task = (
                download_public_image(photos[0], kit_dir / "brand_hero.jpg", cancel=cancel)
                if photos
                else asyncio.sleep(0, result=None)
            )
            viewport_path, hero_path = await asyncio.gather(cap_task, hero_task)
            ui_img = viewport_path or hero_path
            if ui_img and Path(ui_img).is_file():
                kit_parts.append("Live UI")
                for sc in project.scenes:
                    if sc.graphic_spec and sc.graphic_spec.template_id in ("device_showcase_25d", "prompt_ui_demo"):
                        sc.graphic_spec.screenshot_path = str(ui_img)
            el_img = hero_path or viewport_path
            if el_img and Path(el_img).is_file():
                project.elements = [
                    StitchElement(
                        slot=1,
                        tag="@Element1",
                        role="product",
                        label=(eff_brand or "Brand Hero")[:48],
                        image_path=str(el_img),
                        filename=Path(el_img).name,
                    )
                ]
                kit_parts.append("@Element1")
        if kit_parts:
            project.brand_kit_loaded = True
            project.brand_kit_summary = " · ".join(kit_parts)

    # Upgrade with live OpenRouter LLM choreography when live & API key configured
    if not mock and resolved.openrouter_api_key.strip():
        theme_recipe = _find_gallery_recipe(recipe_id)
        sys_prompt = (
            "You are a Master Film Director crafting a structured multi-scene Stitch Script "
            "for AI video models (Kling 3.0 Pro / Veo 3). Every scene has concrete physical "
            "subjects, materials, camera movement, 3-color hex palette, a locked end_frame, "
            "and 3-4 timed beats (time, scene, action, camera, sound). "
            "THEME ADAPTATION RULE: `world_theme` and `theme_reference_example` define the "
            "visual style, tactile craft, materials, lighting, and cinematography direction ONLY. "
            "You MUST reimagine and transform the USER'S `idea`, `brand_name`, and "
            "`scraped_brand_context` INTO that visual theme — the story, product, and hero "
            "centerpiece MUST belong to the user's brand/idea, NEVER copy an unrelated subject "
            "from `theme_reference_example` unless the user explicitly asked for that example. "
            "RULES: Never include readable text, neon signs, billboards, or watermarks in scene "
            "descriptions. For Scene 2+, Beat 1 MUST explicitly start with 'Continues seamlessly "
            "from Scene N's locked frame...' so Last-Frame Chaining is invisible. "
            "Return ONLY a JSON object with keys: title, main_subject, style, audio_direction, "
            "and scenes (array of exact requested length, each with title, subjects (3 strings), "
            "materials (3 strings), camera_spec, colors (3 hex), vo_line (<=14 words), end_frame, "
            "beats (array of {time, scene, action, camera, sound}))."
        )
        usr_payload: dict[str, Any] = {
            "idea": project.idea,
            "brand_name": eff_brand or "",
            "brand_tagline": eff_tagline or "",
            "scraped_brand_context": project.scraped_context,
            "scraped_facts": list(scraped.get("facts") or [])[:6],
            "scraped_colors": list(scraped.get("colors") or [])[:4],
            "num_scenes": len(durations),
            "scene_durations_sec": durations,
            "aspect_ratio": aspect_ratio,
            "world_theme": {
                "label": world_data.get("label", ""),
                "tagline": world_data.get("tagline", ""),
                "style": project.style,
                "environment": world_data.get("environment", {}),
                "subject_details": world_data.get("subject_details", {}),
            },
            "camera_preset": camera_data["label"],
            "lighting_preset": lighting_data["label"],
            "motion_preset": motion_data["label"],
        }
        if theme_recipe and (eff_brand or project.idea.strip() != str(theme_recipe.get("idea") or "").strip()):
            usr_payload["theme_reference_example"] = {
                "title": theme_recipe.get("title", ""),
                "badge": theme_recipe.get("badge", ""),
                "aesthetic_reference_only": theme_recipe.get("idea", ""),
            }
        usr_prompt = json.dumps(usr_payload, indent=2)
        llm_doc = await _call_openrouter_json(
            system_prompt=sys_prompt,
            user_prompt=usr_prompt,
            settings=resolved,
            llm_model_id=llm_model_id,
            max_tokens=9000 if len(durations) > 1 else 5500,
            temperature=0.68,
            cancel=cancel,
        )
        if llm_doc and isinstance(llm_doc.get("scenes"), list) and llm_doc["scenes"]:
            if isinstance(llm_doc.get("title"), str) and llm_doc["title"].strip():
                project.title = llm_doc["title"].strip()[:72]
            if isinstance(llm_doc.get("main_subject"), str) and llm_doc["main_subject"].strip():
                project.main_subject = llm_doc["main_subject"].strip()[:240]
            if isinstance(llm_doc.get("style"), str) and llm_doc["style"].strip():
                project.style = llm_doc["style"].strip()[:240]
            if isinstance(llm_doc.get("audio_direction"), str) and llm_doc["audio_direction"].strip():
                project.audio_direction = llm_doc["audio_direction"].strip()[:320]

            merged_scenes: list[StitchScene] = []
            for idx, _dur in enumerate(durations):
                base_sc = project.scenes[idx]
                raw_sc = llm_doc["scenes"][idx] if idx < len(llm_doc["scenes"]) else {}
                if not isinstance(raw_sc, dict):
                    merged_scenes.append(base_sc)
                    continue
                raw_beats = raw_sc.get("beats") if isinstance(raw_sc.get("beats"), list) else []
                parsed_beats: list[StitchBeat] = []
                for bi, rb in enumerate(raw_beats[:6]):
                    if isinstance(rb, dict) and (rb.get("scene") or rb.get("action")):
                        parsed_beats.append(
                            StitchBeat(
                                time=str(rb.get("time") or f"{bi * 2}-{(bi + 1) * 2}s"),
                                scene=str(rb.get("scene") or ""),
                                action=str(rb.get("action") or ""),
                                camera=str(rb.get("camera") or raw_sc.get("camera_spec") or base_sc.camera_spec),
                                sound=str(rb.get("sound") or ""),
                            )
                        )
                subjs = (
                    [str(x).strip() for x in raw_sc["subjects"] if str(x).strip()][:5]
                    if isinstance(raw_sc.get("subjects"), list)
                    else base_sc.subjects
                )
                mats = (
                    [str(x).strip() for x in raw_sc["materials"] if str(x).strip()][:5]
                    if isinstance(raw_sc.get("materials"), list)
                    else base_sc.materials
                )
                cols = (
                    [str(x).strip() for x in raw_sc["colors"] if str(x).strip()][:4]
                    if isinstance(raw_sc.get("colors"), list) and len(raw_sc["colors"]) >= 2
                    else base_sc.colors
                )
                merged_scenes.append(
                    base_sc.model_copy(
                        update={
                            "title": str(raw_sc.get("title") or base_sc.title)[:64],
                            "subjects": subjs or base_sc.subjects,
                            "materials": mats or base_sc.materials,
                            "camera_spec": str(raw_sc.get("camera_spec") or base_sc.camera_spec)[:180],
                            "colors": cols,
                            "vo_line": str(raw_sc.get("vo_line") or base_sc.vo_line)[:180],
                            "end_frame": str(raw_sc.get("end_frame") or base_sc.end_frame)[:240],
                            "beats": parsed_beats or base_sc.beats,
                        }
                    )
                )
            project.scenes = merged_scenes
            project.status_message = (
                f"Directed {len(project.scenes)} Scene(s) ({project.total_duration_sec()}s) with live AI."
            )

    gate = validate_stitch_project(project, model_spec)
    if gate["valid"]:
        project.status = "validated"
    return project


def _synthetic_bible_from_project(project: StitchProject) -> CreativeBible:
    palette = list(project.lighting.color_palette or ["#191512", "#E54D3A", "#F6E7D4"])
    while len(palette) < 3:
        palette.append("#F6E7D4")
    return CreativeBible(
        concept_thesis=project.main_subject or project.title,
        narrative_structure=NarrativeStructure(
            arc_name="stitch_film",
            beats=[
                BeatSpec(
                    name="hero",
                    purpose="visual centerpiece",
                    duration_hint_sec=float(project.total_duration_sec() or 10),
                )
            ],
        ),
        visual_language=VisualLanguage(
            world=project.environment.world,
            lighting_style=project.lighting.style,
            color_palette=palette[:4],
            texture_and_grade=project.style,
        ),
        camera_grammar=CameraGrammar(
            dominant_style=project.camera.style,
            allowed_moves=[project.camera.movement],
            lens_range_mm=f"{project.camera.lens_mm}mm",
        ),
        typography_and_layout=TypographyAndLayout(
            font_family="Roboto",
            end_card_layout="centered_minimal",
        ),
        sonic_identity=SonicIdentity(
            audio_strategy=project.audio_direction or "tactile ambient foley",
        ),
    )


async def generate_scene_keyframe(
    project: StitchProject,
    scene_idx: int,
    media_dir: Path,
    *,
    which: str = "start",
    tweak_note: str | None = None,
    settings: Settings | None = None,
    mock: bool = False,
    cancel: CancellationToken | None = None,
) -> StitchProject:
    """Generate or regenerate ONE Start or End anchor still for a Scene."""
    resolved = settings or get_settings()
    use_mock = mock or project.video_model_id == "mock" or not resolved.fal_key.strip()
    media_dir.mkdir(parents=True, exist_ok=True)
    idx = max(0, min(scene_idx, len(project.scenes) - 1))
    sc = project.scenes[idx]
    is_end = which.lower() == "end"

    out_file = media_dir / f"{sc.scene_id}_{'end' if is_end else 'start'}_{uuid4().hex[:5]}.jpg"
    prompt = project.compile_scene_keyframe_prompt(idx, is_end=is_end, tweak_note=tweak_note)

    # Pick continuity anchor URL for Flux Kontext if available
    anchor_url: str | None = None
    if is_end and sc.start_keyframe_url:
        anchor_url = sc.start_keyframe_url
    elif not is_end and idx > 0 and project.scenes[idx - 1].end_keyframe_url:
        anchor_url = project.scenes[idx - 1].end_keyframe_url
    elif project.scenes and project.scenes[0].start_keyframe_url:
        anchor_url = project.scenes[0].start_keyframe_url

    if use_mock or sc.kind == "kinetic_ui":
        _render_mock_keyframe_jpeg(
            title=f"{sc.title} · {'End' if is_end else 'Start'} Frame",
            subtitle=(
                f"{sc.end_frame}" if is_end else (sc.beats[0].scene if sc.beats else sc.title)
            )
            + (f" [{tweak_note.strip()}]" if tweak_note and tweak_note.strip() else ""),
            footer=f"{sc.camera_spec[:54]} · {project.aspect_ratio}",
            aspect=project.aspect_ratio,
            palette=sc.colors or project.lighting.color_palette,
            out_path=out_file,
            fonts_dir=resolved.fonts_dir,
        )
        remote_url = None
    else:
        _, remote_url, usd = await _generate_flux_image(
            prompt=prompt,
            aspect=project.aspect_ratio,
            out_path=out_file,
            settings=resolved,
            anchor_url=anchor_url,
            cancel=cancel,
        )
        project.actual_usd = round(project.actual_usd + usd, 4)

    if is_end:
        sc.end_keyframe_path = str(out_file)
        sc.end_keyframe_url = remote_url
        # If next scene continues from this one and has no custom uploaded start yet, sync it
        if idx + 1 < len(project.scenes) and project.scenes[idx + 1].continues_from_prev:
            project.scenes[idx + 1].start_keyframe_path = str(out_file)
            project.scenes[idx + 1].start_keyframe_url = remote_url
    else:
        sc.start_keyframe_path = str(out_file)
        sc.start_keyframe_url = remote_url

    project.status_message = f"Updated {'End' if is_end else 'Start'} anchor still for {sc.title}."
    return project


async def generate_all_keyframes(
    project: StitchProject,
    media_dir: Path,
    *,
    settings: Settings | None = None,
    mock: bool = False,
    cancel: CancellationToken | None = None,
) -> StitchProject:
    """Generate continuity-locked Start & End anchor stills across all Scenes."""
    for idx, sc in enumerate(project.scenes):
        if cancel is not None:
            cancel.raise_if_cancelled()
        if idx > 0 and sc.continues_from_prev and project.scenes[idx - 1].end_keyframe_path:
            sc.start_keyframe_path = project.scenes[idx - 1].end_keyframe_path
            sc.start_keyframe_url = project.scenes[idx - 1].end_keyframe_url
        else:
            await generate_scene_keyframe(
                project, idx, media_dir, which="start", settings=settings, mock=mock, cancel=cancel
            )
        await generate_scene_keyframe(
            project, idx, media_dir, which="end", settings=settings, mock=mock, cancel=cancel
        )
    project.status_message = f"Generated continuity-locked Start & End stills for {len(project.scenes)} Scene(s)."
    return project


async def render_single_scene_take(
    project: StitchProject,
    scene_idx: int,
    media_dir: Path,
    *,
    engine_id: str | None = None,
    registry_store: ModelRegistryStore | None = None,
    settings: Settings | None = None,
    media_store: MediaStore | None = None,
    mock: bool = False,
    on_progress: Any | None = None,
    cancel: CancellationToken | None = None,
) -> StitchProject:
    """Render ONE Scene/Act as a versioned Take with automatic Last-Frame Chaining & @Elements."""
    resolved = settings or get_settings()
    reg = registry_store or ModelRegistryStore(resolved.data_dir / "stitch")
    chosen_engine = (engine_id or project.video_model_id or "").strip()
    spec = reg.get_video_model(chosen_engine)
    use_mock = mock or spec.provider == "mock" or not resolved.fal_key.strip()

    idx = max(0, min(scene_idx, len(project.scenes) - 1))
    sc = project.scenes[idx]
    media_dir.mkdir(parents=True, exist_ok=True)
    fps = 24
    width, height = aspect_to_dimensions(project.aspect_ratio)
    bible = _synthetic_bible_from_project(project)
    palette = sc.colors or project.lighting.color_palette or ["#191512", "#E54D3A", "#F6E7D4"]
    bg_rgb = _hex_to_rgb(palette[0], (25, 21, 18))
    accent_rgb = _hex_to_rgb(palette[1] if len(palette) > 1 else "#E54D3A", (229, 77, 58))
    text_rgb = _hex_to_rgb(palette[2] if len(palette) > 2 else "#F6E7D4", (246, 231, 212))

    async def _report(msg: str) -> None:
        sc.render_message = msg
        project.status_message = msg
        if on_progress is not None:
            res = on_progress(project)
            if asyncio.iscoroutine(res):
                await res

    sc.render_status = "rendering"
    sc.render_error = None
    project.status = "shooting"
    take_num = (max((t.take_id for t in sc.takes), default=0)) + 1
    out_clip = media_dir / f"{sc.scene_id}_take{take_num}.mp4"
    chained_from_prev = False

    try:
        # 1. Check Last-Frame Chaining from previous Scene's winning take!
        chained_start_uri: str | None = None
        if idx > 0 and sc.continues_from_prev:
            prev_win = project.scenes[idx - 1].winning_take()
            if prev_win and Path(prev_win.clip_path).is_file():
                await _report(f"Extracting final pixel frame from Scene {idx} for seamless chain...")
                chain_jpg = media_dir / f"{sc.scene_id}_chained_start_t{take_num}.jpg"
                await extract_last_frame_jpeg(
                    Path(prev_win.clip_path),
                    chain_jpg,
                    ffmpeg_bin=resolved.ffmpeg_bin,
                    cancel=cancel,
                )
                sc.start_keyframe_path = str(chain_jpg)
                sc.start_keyframe_url = None
                chained_start_uri = encode_image_data_uri(chain_jpg)
                chained_from_prev = True

        # 2A. User / Founder Video Clip Take ($0 — trimmed, framed & loudnormed) when a clip is uploaded
        has_uploaded_clip = (
            sc.kind == "user_clip"
            and sc.user_clip is not None
            and bool(sc.user_clip.raw_video_path)
            and Path(sc.user_clip.raw_video_path).is_file()
        )
        if has_uploaded_clip:
            assert sc.user_clip is not None
            await _report(f"Normalizing {sc.title} ({sc.user_clip.framing}, loudnorm)...")
            _, norm_dur = await normalize_user_clip_take(
                clip_spec=sc.user_clip,
                output_path=out_clip,
                width=width,
                height=height,
                fps=fps,
                ffmpeg_bin=resolved.ffmpeg_bin,
                ffprobe_bin=resolved.ffprobe_bin,
                cancel=cancel,
            )
            sc.duration_sec = max(3, min(60, int(round(norm_dur))))
            take = StitchTake(
                take_id=take_num,
                engine_id="user_clip",
                engine_label=f"User Video ({sc.user_clip.framing})",
                clip_path=str(out_clip),
                duration_sec=float(norm_dur),
                usd=0.0,
                chained_from_prev=False,
            )
        # 2B. Launch Motion Graphics Template ($0 — crisp 24fps UI / Charts / Code / Editorial)
        elif sc.kind == "motion_graphic" or (sc.kind == "kinetic_ui" and sc.graphic_spec is not None):
            gspec = sc.graphic_spec or _default_graphic_spec(
                "prompt_ui_demo",
                brand=project.brand_name or project.title,
                idea=project.idea,
                scraped={},
                world_id=project.world_preset,
            )
            label = f"Motion Graphic · {gspec.template_id.replace('_', ' ').title()} ($0)"
            await _report(f"Rendering {sc.title} · Take {take_num} ({label})...")
            await render_motion_template_mp4(
                spec=gspec,
                output_path=out_clip,
                width=width,
                height=height,
                fps=fps,
                duration_sec=float(sc.duration_sec),
                brand_name=project.brand_name or project.title,
                accent_rgb=accent_rgb,
                ffmpeg_bin=resolved.ffmpeg_bin,
                ffprobe_bin=resolved.ffprobe_bin,
                fonts_dir=resolved.fonts_dir,
                cancel=cancel,
            )
            take = StitchTake(
                take_id=take_num,
                engine_id=f"motion_graphic:{gspec.template_id}",
                engine_label=label,
                clip_path=str(out_clip),
                duration_sec=float(sc.duration_sec),
                usd=0.0,
                chained_from_prev=False,
            )
        # 2C. Legacy $0 Kinetic UI Scene OR $0 Offline Animatic Take
        elif sc.kind == "kinetic_ui" or use_mock:
            label = "Kinetic UI ($0)" if sc.kind == "kinetic_ui" else "Studio Animatic ($0)"
            await _report(f"Rendering {sc.title} · Take {take_num} ({label})...")
            hero_img: Path | None = None
            if sc.start_keyframe_path and Path(sc.start_keyframe_path).is_file():
                hero_img = Path(sc.start_keyframe_path)
            elif project.elements and Path(project.elements[0].image_path).is_file():
                hero_img = Path(project.elements[0].image_path)

            await render_kinetic_card_mp4(
                text=sc.title[:48],
                subtitle=(sc.vo_line or (sc.beats[0].scene if sc.beats else sc.end_frame))[:110],
                footer=f"{sc.camera_spec[:48]} · Take {take_num}" + (" · Pixel-Chained" if chained_from_prev else ""),
                width=width,
                height=height,
                fps=fps,
                duration_sec=float(sc.duration_sec),
                bible=bible,
                output_path=out_clip,
                bg_rgb=bg_rgb,
                text_rgb=text_rgb,
                accent_rgb=accent_rgb,
                ffmpeg_bin=resolved.ffmpeg_bin,
                ffprobe_bin=resolved.ffprobe_bin,
                fonts_dir=resolved.fonts_dir,
                card_kind="title",
                shot_id=f"{sc.scene_id}_t{take_num}",
                media_image_path=hero_img,
                cancel=cancel,
            )
            take = StitchTake(
                take_id=take_num,
                engine_id="kinetic_ui" if sc.kind == "kinetic_ui" else "mock",
                engine_label=label,
                clip_path=str(out_clip),
                duration_sec=float(sc.duration_sec),
                usd=0.0,
                chained_from_prev=chained_from_prev,
            )
        else:
            # 3. Live Dynamic-Registry AI Video Render (Kling 3.0 Pro, Kling 4.0 Flash, Native 4K, Veo 3, or Custom!)
            await _report(
                f"Shooting {sc.title} · Take {take_num} via {spec.label} ({spec.snap_duration(sc.duration_sec)}s)..."
            )
            start_ref: str | None = chained_start_uri
            if not start_ref:
                if sc.start_keyframe_url and sc.start_keyframe_url.startswith(("http://", "https://")):
                    start_ref = sc.start_keyframe_url
                elif sc.start_keyframe_path and Path(sc.start_keyframe_path).is_file():
                    start_ref = encode_image_data_uri(Path(sc.start_keyframe_path))

            end_ref: str | None = None
            if spec.supports_end_frame:
                if sc.end_keyframe_url and sc.end_keyframe_url.startswith(("http://", "https://")):
                    end_ref = sc.end_keyframe_url
                elif sc.end_keyframe_path and Path(sc.end_keyframe_path).is_file():
                    end_ref = encode_image_data_uri(Path(sc.end_keyframe_path))

            element_uris: list[str] = []
            if spec.supports_elements and project.elements:
                for el in sorted(project.elements, key=lambda e: e.slot)[:3]:
                    if Path(el.image_path).is_file():
                        element_uris.append(encode_image_data_uri(Path(el.image_path), max_width=1024))

            compiled_json = project.compile_scene_model_prompt(idx, max_chars=spec.max_prompt_chars)
            neg_str = ", ".join(project.negative_constraints)
            cost_meter = CostMeter(
                run_id=f"{project.project_id}_{sc.scene_id}_t{take_num}",
                cap_usd=max(12.0, spec.est_usd_per_sec * sc.duration_sec * 3.0 + 4.0),
            )
            provider = VideoProvider(
                settings=resolved,
                media_store=media_store or MediaStore(media_dir / "store"),
                cost_meter=cost_meter,
                usd_per_sec=spec.est_usd_per_sec,
            )

            try:
                rendered = await provider.generate_from_registry(
                    spec=spec,
                    shot_id=f"{sc.scene_id}_t{take_num}",
                    prompt=compiled_json,
                    duration_sec=spec.snap_duration(sc.duration_sec),
                    aspect=project.aspect_ratio,
                    fps=fps,
                    start_image=start_ref,
                    end_image=end_ref,
                    negative_prompt=neg_str,
                    generate_audio=(project.audio_mode != "vo_plus_ducked_score"),
                    element_image_urls=element_uris,
                    cancel=cancel,
                )
            except Exception as primary_exc:
                # If a fallback model is configured and distinct, try once cleanly unless it's a balance/auth error
                err_low = str(primary_exc).lower()
                is_billing = any(k in err_low for k in ("balance", "exhausted", "unauthorized", "forbidden", "401", "403"))
                if spec.fallback_id and spec.fallback_id != spec.id and not is_billing:
                    fb_spec = reg.get_video_model(spec.fallback_id)
                    await _report(f"{spec.label} unavailable — seamlessly switching to {fb_spec.label}...")
                    rendered = await provider.generate_from_registry(
                        spec=fb_spec,
                        shot_id=f"{sc.scene_id}_t{take_num}_fb",
                        prompt=project.compile_scene_model_prompt(idx, max_chars=fb_spec.max_prompt_chars),
                        duration_sec=fb_spec.snap_duration(sc.duration_sec),
                        aspect=project.aspect_ratio,
                        fps=fps,
                        start_image=start_ref,
                        end_image=end_ref if fb_spec.supports_end_frame else None,
                        negative_prompt=neg_str,
                        generate_audio=(project.audio_mode != "vo_plus_ducked_score"),
                        element_image_urls=element_uris if fb_spec.supports_elements else None,
                        cancel=cancel,
                    )
                    spec = fb_spec
                else:
                    raise

            shutil.copyfile(rendered.path, out_clip)
            probe = await probe_media(out_clip, ffprobe_bin=resolved.ffprobe_bin, cancel=cancel)
            take = StitchTake(
                take_id=take_num,
                engine_id=spec.id,
                engine_label=spec.label,
                clip_path=str(out_clip),
                duration_sec=round(float(probe.duration_sec or sc.duration_sec), 2),
                usd=round(float(rendered.usd), 4),
                chained_from_prev=chained_from_prev,
            )
            project.actual_usd = round(project.actual_usd + float(rendered.usd), 4)

        sc.takes.append(take)
        sc.winning_take_id = take.take_id
        sc.render_status = "done"
        sc.render_message = f"Take {take.take_id} ready ({take.engine_label})."
        project.status_message = f"{sc.title} · Take {take.take_id} ready."
        if not any(s.render_status == "rendering" for s in project.scenes):
            all_shot = all(s.winning_take() is not None for s in project.scenes)
            project.status = "done" if all_shot else "validated"
        return project
    except Exception as exc:
        sc.render_status = "failed"
        sc.render_error = str(exc)
        sc.render_message = f"Render failed: {exc}"
        project.error = str(exc)
        project.status_message = f"{sc.title} failed: {exc}"
        if not any(s.render_status == "rendering" for s in project.scenes):
            project.status = "failed"
        raise


async def render_all_scenes(
    project: StitchProject,
    media_dir: Path,
    *,
    engine_id: str | None = None,
    registry_store: ModelRegistryStore | None = None,
    settings: Settings | None = None,
    media_store: MediaStore | None = None,
    mock: bool = False,
    on_progress: Any | None = None,
    cancel: CancellationToken | None = None,
) -> StitchProject:
    """Sequentially render all Scenes (Act 1 -> Act N) so Last-Frame Chaining flows cleanly."""
    for idx in range(len(project.scenes)):
        if cancel is not None:
            cancel.raise_if_cancelled()
        await render_single_scene_take(
            project,
            idx,
            media_dir,
            engine_id=engine_id,
            registry_store=registry_store,
            settings=settings,
            media_store=media_store,
            mock=mock,
            on_progress=on_progress,
            cancel=cancel,
        )
    return await stitch_master_cut(
        project,
        media_dir,
        settings=settings,
        mock=mock,
        on_progress=on_progress,
        cancel=cancel,
    )


def _synthesize_riser_bed_wav(
    duration_sec: float,
    seam_times: list[float],
    out_path: Path,
) -> Path:
    """Synthesize warm ambient pad with subtle harmonic risers at each Scene seam ($0)."""
    sr = 24000
    n = max(sr, int(sr * duration_sec))
    frames = bytearray()
    freqs = (130.81, 196.00, 261.63, 329.63)
    for i in range(n):
        t = i / sr
        fade_in = min(1.0, t / 0.7)
        fade_out = min(1.0, max(0.0, (duration_sec - t) / 0.9))
        riser = 0.0
        for st in seam_times:
            dist = t - st
            if -0.8 <= dist <= 0.35:
                # Subtle harmonic swell peaking right on the cut
                riser = max(riser, math.exp(-((dist / 0.32) ** 2)) * 0.45)
        env = fade_in * fade_out * (0.75 + 0.12 * math.sin(2.0 * math.pi * 0.25 * t) + riser)
        sample = sum(math.sin(2.0 * math.pi * f * t) for f in freqs) / len(freqs)
        if riser > 0.05:
            sample += 0.25 * riser * math.sin(2.0 * math.pi * (392.0 + 120.0 * riser) * t)
        val = int(max(-32767, min(32767, sample * env * 4600.0)))
        frames.extend(struct.pack("<h", val))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(bytes(frames))
    return out_path


async def plan_aroll_cutaways(
    project: StitchProject,
    *,
    settings: Settings | None = None,
    mock: bool = False,
    cancel: CancellationToken | None = None,
) -> StitchProject:
    """Transcribe Founder A-Roll (if needed) and plan timed Motion Graphic & AI B-Roll cutaways."""
    if not project.aroll_clip_path or not Path(project.aroll_clip_path).is_file():
        raise RuntimeError("Upload a Founder A-Roll video first before planning cutaways.")
    resolved = settings or get_settings()
    if not project.aroll_transcript or project.aroll_duration_sec <= 0:
        _, words, dur = await transcribe_video_words(
            Path(project.aroll_clip_path),
            fallback_text=project.idea or project.title,
            settings=resolved,
            mock=mock,
            cancel=cancel,
        )
        project.aroll_transcript = words
        project.aroll_duration_sec = dur

    dur = max(4.0, float(project.aroll_duration_sec or 15.0))
    brand = project.brand_name or project.title
    # Plan 2 well-paced cutaway windows leaving founder on camera for opening & closing hooks
    w1_start = round(max(1.2, dur * 0.22), 2)
    w1_end = round(min(dur - 1.5, w1_start + min(5.0, max(2.5, dur * 0.28))), 2)
    w2_start = round(max(w1_end + 1.2, dur * 0.60), 2)
    w2_end = round(min(dur - 1.0, w2_start + min(5.0, max(2.2, dur * 0.24))), 2)

    def _snippet(t0: float, t1: float) -> str:
        ws = [
            str(w.get("word") or "")
            for w in project.aroll_transcript
            if float(w.get("start", 0.0)) >= t0 - 0.2 and float(w.get("end", 0.0)) <= t1 + 0.3
        ]
        return " ".join(ws).strip()

    s1_txt = _snippet(w1_start, w1_end) or project.idea[:80]
    s2_txt = _snippet(w2_start, w2_end) or f"{brand} product vision"

    cutaway_scenes: list[StitchScene] = []
    if w1_end - w1_start >= 1.5:
        g1 = _default_graphic_spec(
            "prompt_ui_demo",
            brand=brand,
            idea=s1_txt,
            scraped={"tagline": project.brand_tagline or "", "facts": [s1_txt]},
            world_id=project.world_preset,
        )
        cutaway_scenes.append(
            StitchScene(
                scene_id=f"sc_cut1_{uuid4().hex[:6]}",
                index=0,
                title=f"Cutaway 1 · UI Proof ({w1_start:.1f}s–{w1_end:.1f}s)",
                duration_sec=max(3, int(round(w1_end - w1_start))),
                kind="motion_graphic",
                continues_from_prev=False,
                subjects=[brand, "live product interface"],
                materials=["dark frosted glass", "crisp vector UI"],
                camera_spec="Locked studio UI choreography",
                colors=list(project.lighting.color_palette[:3] or ["#0C0E14", "#4E8CFF", "#F4F7FC"]),
                vo_line=s1_txt[:140],
                end_frame="Clean product UI proof locked in frame",
                beats=_make_beats_for_duration(
                    max(3, int(round(w1_end - w1_start))),
                    scene_title="UI Proof Cutaway",
                    idea=s1_txt,
                    subjects=[brand],
                    materials=["frosted glass", "vector UI"],
                    camera_spec="Locked studio UI choreography",
                    continues=False,
                    prev_end_frame="",
                ),
                graphic_spec=g1,
                aroll_window=(w1_start, w1_end),
            )
        )

    if w2_end - w2_start >= 1.5:
        g2 = _default_graphic_spec(
            "benchmark_bars",
            brand=brand,
            idea=s2_txt,
            scraped={"tagline": project.brand_tagline or "", "facts": [s2_txt]},
            world_id=project.world_preset,
        )
        cutaway_scenes.append(
            StitchScene(
                scene_id=f"sc_cut2_{uuid4().hex[:6]}",
                index=len(cutaway_scenes),
                title=f"Cutaway 2 · Impact Proof ({w2_start:.1f}s–{w2_end:.1f}s)",
                duration_sec=max(3, int(round(w2_end - w2_start))),
                kind="motion_graphic",
                continues_from_prev=False,
                subjects=[brand, "animated performance benchmark"],
                materials=["editorial glass", "luminous accent bars"],
                camera_spec="Smooth studio data reveal",
                colors=list(project.lighting.color_palette[:3] or ["#0C0E14", "#E54D3A", "#F4F7FC"]),
                vo_line=s2_txt[:140],
                end_frame="Benchmark proof locked with brand highlight",
                beats=_make_beats_for_duration(
                    max(3, int(round(w2_end - w2_start))),
                    scene_title="Benchmark Cutaway",
                    idea=s2_txt,
                    subjects=[brand],
                    materials=["editorial glass"],
                    camera_spec="Smooth studio data reveal",
                    continues=False,
                    prev_end_frame="",
                ),
                graphic_spec=g2,
                aroll_window=(w2_start, w2_end),
            )
        )

    if cutaway_scenes:
        project.scenes = cutaway_scenes
    project.timeline_mode = "aroll_with_cutaways"
    project.status_message = (
        f"Planned {len(cutaway_scenes)} timed cutaway(s) over your {dur:.1f}s Founder A-Roll voice track."
    )
    return project


async def _apply_scene_overlays(
    *,
    clip_path: Path,
    overlays: list[SceneOverlaySpec],
    output_path: Path,
    width: int,
    height: int,
    fps: int,
    accent_rgb: tuple[int, int, int],
    ffmpeg_bin: str,
    fonts_dir: Path | None,
    threads: int,
    tmp_dir: Path,
    cancel: CancellationToken | None = None,
) -> Path:
    """Composite transparent RGBA overlay PNGs onto a single scene clip using FFmpeg."""
    if not overlays:
        return clip_path
    inputs: list[str] = ["-i", str(clip_path)]
    fc_parts: list[str] = []
    cur = "0:v"
    for idx, ov in enumerate(overlays[:4]):
        png_p = render_overlay_png(
            overlay=ov,
            width=width,
            height=height,
            output_path=tmp_dir / f"ov_{idx}_{ov.overlay_id}.png",
            accent_rgb=accent_rgb,
            fonts_dir=fonts_dir,
        )
        inputs.extend(["-loop", "1", "-i", str(png_p)])
        nxt = f"vov{idx}"
        t0 = max(0.0, float(ov.start_sec))
        t1 = max(t0 + 0.4, float(ov.end_sec))
        fc_parts.append(
            f"[{cur}][{idx + 1}:v]overlay=0:0:shortest=1:enable='between(t,{t0:.2f},{t1:.2f})'[{nxt}]"
        )
        cur = nxt

    cmd = [
        ffmpeg_bin,
        "-y",
        "-threads",
        str(threads),
        *inputs,
        "-filter_complex",
        ";".join(fc_parts),
        "-map",
        f"[{cur}]",
        "-map",
        "0:a?",
        *_pinned_h264_flags(fps, threads=threads),
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    await run_subprocess(cmd, timeout_sec=120.0, cancel=cancel, check=True)
    return output_path


def _ass_ts(sec: float) -> str:
    sec = max(0.0, float(sec))
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    cs = int(round((sec - math.floor(sec)) * 100))
    if cs >= 100:
        cs = 99
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _write_project_captions_ass(
    *,
    project: StitchProject,
    clip_durations: list[float],
    width: int,
    height: int,
    out_path: Path,
) -> Path | None:
    """Write a clean ASS v4.00+ subtitle file from A-roll word timestamps or scene VO lines."""
    cues: list[tuple[float, float, str]] = []
    if project.aroll_transcript:
        # Group into crisp 4-5 word subtitle phrases
        chunk: list[dict[str, Any]] = []
        for w in project.aroll_transcript:
            chunk.append(w)
            if len(chunk) >= 5:
                t0 = float(chunk[0].get("start", 0.0))
                t1 = max(t0 + 0.35, float(chunk[-1].get("end", t0 + 1.2)))
                txt = " ".join(str(x.get("word") or "") for x in chunk).strip()
                if txt:
                    cues.append((t0, t1, txt))
                chunk = []
        if chunk:
            t0 = float(chunk[0].get("start", 0.0))
            t1 = max(t0 + 0.35, float(chunk[-1].get("end", t0 + 1.2)))
            txt = " ".join(str(x.get("word") or "") for x in chunk).strip()
            if txt:
                cues.append((t0, t1, txt))
    else:
        cursor = 0.0
        for sc, dur in zip(project.scenes, clip_durations, strict=False):
            line = (sc.vo_line or "").strip()
            if line:
                cues.append((cursor + 0.2, max(cursor + 0.8, cursor + dur - 0.25), line))
            cursor += dur

    if not cues:
        return None

    font_name = "Liberation Serif" if project.caption_style == "editorial_serif" else "Roboto"
    font_size = max(22, int(height * 0.044))
    margin_v = max(28, int(height * 0.06))
    primary_col = "&H00F4F7FC"
    back_col = "&H80101218"
    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {width}\n"
        f"PlayResY: {height}\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{font_name},{font_size},{primary_col},&H0038A8E8,&H0012141A,{back_col},"
        f"1,0,0,0,100,100,0,0,3,2,0,2,48,48,{margin_v},1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    lines = [header]
    for t0, t1, txt in cues:
        safe = txt.replace("\n", " ").replace("{", "").replace("}", "")
        lines.append(f"Dialogue: 0,{_ass_ts(t0)},{_ass_ts(t1)},Default,,0,0,0,,{safe}\n")
    out_path.write_text("".join(lines), encoding="utf-8")
    return out_path


async def stitch_master_cut(
    project: StitchProject,
    media_dir: Path,
    *,
    settings: Settings | None = None,
    mock: bool = False,
    on_progress: Any | None = None,
    cancel: CancellationToken | None = None,
) -> StitchProject:
    """Assemble all winning Takes (or Founder A-Roll + Cutaways) + Overlays + Captions + Brand Outro."""
    resolved = settings or get_settings()
    media_dir.mkdir(parents=True, exist_ok=True)
    fps = 24
    width, height = aspect_to_dimensions(project.aspect_ratio)
    raw_t = os.getenv("GTM_FFMPEG_THREADS", "2").strip()
    eff_threads = max(1, int(raw_t)) if raw_t.isdigit() else 2

    async def _report(msg: str) -> None:
        project.status_message = msg
        if on_progress is not None:
            res = on_progress(project)
            if asyncio.iscoroutine(res):
                await res

    palette = project.lighting.color_palette or ["#191512", "#E54D3A", "#F6E7D4"]
    accent_rgb = _hex_to_rgb(palette[1] if len(palette) > 1 else "#E54D3A", (229, 77, 58))

    use_aroll_mode = (
        project.timeline_mode == "aroll_with_cutaways"
        and bool(project.aroll_clip_path)
        and Path(str(project.aroll_clip_path)).is_file()
    )

    winning_pairs: list[tuple[StitchScene | None, Path]] = []
    for sc in project.scenes:
        win = sc.winning_take()
        if win and Path(win.clip_path).is_file():
            winning_pairs.append((sc, Path(win.clip_path)))
    if not winning_pairs and not use_aroll_mode:
        raise RuntimeError("Render at least one Scene take before stitching the Master Cut.")

    project.status = "stitching"
    project.error = None
    await _report(
        "Stitching Founder A-Roll + Cutaways into Master Cut..."
        if use_aroll_mode
        else f"Stitching {len(winning_pairs)} Scene(s) into Master Cut..."
    )

    tmp_dir = Path(tempfile.mkdtemp(prefix="stitch_master_"))
    aroll_audio_spine: Path | None = None
    try:
        norm_paths: list[Path] = []
        clip_durations: list[float] = []
        any_native_audio = False
        preserve_native = project.audio_mode in ("native_only", "native_plus_score")

        if use_aroll_mode:
            aroll_src = Path(str(project.aroll_clip_path))
            aroll_probe = await probe_media(aroll_src, ffprobe_bin=resolved.ffprobe_bin, cancel=cancel)
            aroll_dur = max(2.0, float(aroll_probe.duration_sec or project.aroll_duration_sec or 10.0))

            # Normalize full A-roll once to target dimensions/fps & extract continuous voice spine
            aroll_norm = tmp_dir / "aroll_full_norm.mp4"
            await normalize_user_clip_take(
                clip_spec=UserClipSpec(
                    raw_video_path=str(aroll_src),
                    filename=project.aroll_filename or aroll_src.name,
                    source_duration_sec=aroll_dur,
                    trim_start_sec=0.0,
                    trim_end_sec=aroll_dur,
                    framing="cover_crop",
                    keep_original_audio=True,
                ),
                output_path=aroll_norm,
                width=width,
                height=height,
                fps=fps,
                ffmpeg_bin=resolved.ffmpeg_bin,
                ffprobe_bin=resolved.ffprobe_bin,
                cancel=cancel,
            )
            aroll_audio_spine = tmp_dir / "aroll_voice_spine.wav"
            await run_subprocess(
                [
                    resolved.ffmpeg_bin,
                    "-y",
                    "-threads",
                    str(eff_threads),
                    "-i",
                    str(aroll_norm),
                    "-vn",
                    "-ar",
                    "48000",
                    "-ac",
                    "2",
                    str(aroll_audio_spine),
                ],
                timeout_sec=90.0,
                cancel=cancel,
                check=True,
            )
            any_native_audio = True

            # Build ordered non-overlapping visual intervals: A-roll + cutaways
            cutaways = sorted(
                [(sc, cp) for sc, cp in winning_pairs if sc is not None and sc.aroll_window is not None],
                key=lambda item: float(item[0].aroll_window[0]),  # type: ignore[index]
            )
            cursor = 0.0
            seg_idx = 0
            for sc_cut, cp_cut in cutaways:
                assert sc_cut is not None and sc_cut.aroll_window is not None
                t0 = max(cursor, min(aroll_dur, float(sc_cut.aroll_window[0])))
                t1 = max(t0, min(aroll_dur, float(sc_cut.aroll_window[1])))
                if t0 - cursor >= 0.25:
                    sl = tmp_dir / f"aroll_seg_{seg_idx:02d}.mp4"
                    await run_subprocess(
                        [
                            resolved.ffmpeg_bin,
                            "-y",
                            "-threads",
                            str(eff_threads),
                            "-ss",
                            f"{cursor:.3f}",
                            "-t",
                            f"{(t0 - cursor):.3f}",
                            "-i",
                            str(aroll_norm),
                            *_pinned_h264_flags(fps, threads=eff_threads),
                            "-c:a",
                            "aac",
                            "-ar",
                            "48000",
                            "-ac",
                            "2",
                            "-b:a",
                            "160k",
                            str(sl),
                        ],
                        timeout_sec=90.0,
                        cancel=cancel,
                        check=True,
                    )
                    norm_paths.append(sl)
                    clip_durations.append(t0 - cursor)
                    seg_idx += 1

                if t1 - t0 >= 0.25:
                    cut_norm = tmp_dir / f"cut_seg_{seg_idx:02d}.mp4"
                    await _normalize_act_clip(
                        cp_cut,
                        cut_norm,
                        width=width,
                        height=height,
                        fps=fps,
                        has_audio=False,
                        ffmpeg_bin=resolved.ffmpeg_bin,
                        cancel=cancel,
                    )
                    if sc_cut.overlays:
                        cut_ov = tmp_dir / f"cut_seg_ov_{seg_idx:02d}.mp4"
                        cut_norm = await _apply_scene_overlays(
                            clip_path=cut_norm,
                            overlays=sc_cut.overlays,
                            output_path=cut_ov,
                            width=width,
                            height=height,
                            fps=fps,
                            accent_rgb=accent_rgb,
                            ffmpeg_bin=resolved.ffmpeg_bin,
                            fonts_dir=resolved.fonts_dir,
                            threads=eff_threads,
                            tmp_dir=tmp_dir,
                            cancel=cancel,
                        )
                    # Fit cutaway visual to exact [t0, t1] window
                    cut_fit = tmp_dir / f"cut_fit_{seg_idx:02d}.mp4"
                    await run_subprocess(
                        [
                            resolved.ffmpeg_bin,
                            "-y",
                            "-threads",
                            str(eff_threads),
                            "-stream_loop",
                            "-1",
                            "-i",
                            str(cut_norm),
                            "-t",
                            f"{(t1 - t0):.3f}",
                            *_pinned_h264_flags(fps, threads=eff_threads),
                            "-c:a",
                            "aac",
                            "-ar",
                            "48000",
                            "-ac",
                            "2",
                            "-b:a",
                            "160k",
                            str(cut_fit),
                        ],
                        timeout_sec=90.0,
                        cancel=cancel,
                        check=True,
                    )
                    norm_paths.append(cut_fit)
                    clip_durations.append(t1 - t0)
                    seg_idx += 1
                    cursor = t1

            if aroll_dur - cursor >= 0.25 or not norm_paths:
                tail_dur = max(0.5, aroll_dur - cursor)
                sl = tmp_dir / f"aroll_seg_{seg_idx:02d}.mp4"
                await run_subprocess(
                    [
                        resolved.ffmpeg_bin,
                        "-y",
                        "-threads",
                        str(eff_threads),
                        "-ss",
                        f"{cursor:.3f}",
                        "-t",
                        f"{tail_dur:.3f}",
                        "-i",
                        str(aroll_norm),
                        *_pinned_h264_flags(fps, threads=eff_threads),
                        "-c:a",
                        "aac",
                        "-ar",
                        "48000",
                        "-ac",
                        "2",
                        "-b:a",
                        "160k",
                        str(sl),
                    ],
                    timeout_sec=90.0,
                    cancel=cancel,
                    check=True,
                )
                norm_paths.append(sl)
                clip_durations.append(tail_dur)
        else:
            # Sequential Acts mode
            if project.add_brand_outro and (project.brand_name or project.brand_logo_path):
                outro_mp4 = tmp_dir / "brand_outro.mp4"
                bible = _synthetic_bible_from_project(project)
                logo_p = (
                    Path(project.brand_logo_path)
                    if project.brand_logo_path and Path(project.brand_logo_path).is_file()
                    else None
                )
                await render_kinetic_card_mp4(
                    text=(project.brand_name or project.title)[:42],
                    subtitle=(project.brand_tagline or "Directed with Stitch")[:80],
                    footer=(project.source_url or "")[:64],
                    width=width,
                    height=height,
                    fps=fps,
                    duration_sec=2.5,
                    bible=bible,
                    output_path=outro_mp4,
                    bg_rgb=_hex_to_rgb(palette[0], (25, 21, 18)),
                    accent_rgb=accent_rgb,
                    text_rgb=_hex_to_rgb(palette[2] if len(palette) > 2 else "#F6E7D4", (246, 231, 212)),
                    ffmpeg_bin=resolved.ffmpeg_bin,
                    ffprobe_bin=resolved.ffprobe_bin,
                    fonts_dir=resolved.fonts_dir,
                    card_kind="end_card",
                    shot_id="brand_outro",
                    logo_path=logo_p,
                    cancel=cancel,
                )
                winning_pairs.append((None, outro_mp4))

            for idx, (sc_obj, clip_p) in enumerate(winning_pairs):
                probe = await probe_media(clip_p, ffprobe_bin=resolved.ffprobe_bin, cancel=cancel)
                has_a = bool(probe.has_audio and preserve_native)
                if has_a:
                    any_native_audio = True
                dur_s = float(probe.duration_sec or 8.0)
                clip_durations.append(dur_s)
                norm_p = tmp_dir / f"norm_{idx:02d}.mp4"
                await _normalize_act_clip(
                    clip_p,
                    norm_p,
                    width=width,
                    height=height,
                    fps=fps,
                    has_audio=has_a,
                    ffmpeg_bin=resolved.ffmpeg_bin,
                    cancel=cancel,
                )
                if sc_obj is not None and sc_obj.overlays:
                    ov_p = tmp_dir / f"norm_ov_{idx:02d}.mp4"
                    norm_p = await _apply_scene_overlays(
                        clip_path=norm_p,
                        overlays=sc_obj.overlays,
                        output_path=ov_p,
                        width=width,
                        height=height,
                        fps=fps,
                        accent_rgb=accent_rgb,
                        ffmpeg_bin=resolved.ffmpeg_bin,
                        fonts_dir=resolved.fonts_dir,
                        threads=eff_threads,
                        tmp_dir=tmp_dir,
                        cancel=cancel,
                    )
                norm_paths.append(norm_p)

        assembled_mp4 = tmp_dir / "assembled.mp4"
        if len(norm_paths) == 1:
            shutil.copyfile(norm_paths[0], assembled_mp4)
        elif project.seam_transition == "micro_dissolve" and len(norm_paths) <= 5:
            # Pairwise or chained 0.35s xfade + acrossfade (Render-512MB safe with threads=1)
            xf = 0.35
            inputs: list[str] = []
            for p in norm_paths:
                inputs.extend(["-i", str(p)])
            vf_parts: list[str] = []
            af_parts: list[str] = []
            cur_v = "0:v"
            cur_a = "0:a"
            offset = max(0.2, clip_durations[0] - xf)
            for i in range(1, len(norm_paths)):
                nxt_v = f"v{i}"
                nxt_a = f"a{i}"
                vf_parts.append(
                    f"[{cur_v}][{i}:v]xfade=transition=fade:duration={xf:.2f}:offset={offset:.2f}[{nxt_v}]"
                )
                af_parts.append(f"[{cur_a}][{i}:a]acrossfade=d={xf:.2f}[{nxt_a}]")
                cur_v = nxt_v
                cur_a = nxt_a
                offset = max(0.2, offset + clip_durations[i] - xf)
            fc = ";".join([*vf_parts, *af_parts])
            xf_cmd = [
                resolved.ffmpeg_bin,
                "-y",
                "-threads",
                str(eff_threads),
                *inputs,
                "-filter_complex",
                fc,
                "-map",
                f"[{cur_v}]",
                "-map",
                f"[{cur_a}]",
                *_pinned_h264_flags(fps, threads=eff_threads),
                "-c:a",
                "aac",
                "-ar",
                "48000",
                "-ac",
                "2",
                "-b:a",
                "160k",
                "-movflags",
                "+faststart",
                str(assembled_mp4),
            ]
            rc_xf, _, _ = await run_subprocess(xf_cmd, timeout_sec=180.0, cancel=cancel, check=False)
            if rc_xf != 0 or not assembled_mp4.is_file():
                # Clean fallback to zero-overhead concat
                list_txt = tmp_dir / "list.txt"
                lines = [f"file '{str(p.resolve())}'" for p in norm_paths]
                list_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
                await run_subprocess(
                    [
                        resolved.ffmpeg_bin,
                        "-y",
                        "-threads",
                        str(eff_threads),
                        "-f",
                        "concat",
                        "-safe",
                        "0",
                        "-i",
                        str(list_txt),
                        "-c",
                        "copy",
                        "-movflags",
                        "+faststart",
                        str(assembled_mp4),
                    ],
                    timeout_sec=120.0,
                    cancel=cancel,
                    check=True,
                )
        else:
            list_txt = tmp_dir / "list.txt"
            lines = [f"file '{str(p.resolve())}'" for p in norm_paths]
            list_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
            await run_subprocess(
                [
                    resolved.ffmpeg_bin,
                    "-y",
                    "-threads",
                    str(eff_threads),
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    str(list_txt),
                    "-c",
                    "copy",
                    "-movflags",
                    "+faststart",
                    str(assembled_mp4),
                ],
                timeout_sec=120.0,
                cancel=cancel,
                check=True,
            )

        # Compute seam timestamps for audio_riser bed
        seam_times: list[float] = []
        acc = 0.0
        for d in clip_durations[:-1]:
            acc += d
            seam_times.append(acc)
        total_dur = max(3.0, sum(clip_durations))

        # Audio Mastering (3 Soundstage Modes)
        final_tmp = tmp_dir / "master_final.mp4"
        vo_path: Path | None = None
        if (
            project.audio_mode == "vo_plus_ducked_score"
            and not mock
            and resolved.fal_key.strip()
        ):
            vo_script = " ".join(s.vo_line.strip() for s in project.scenes if s.vo_line.strip())
            if vo_script:
                try:
                    ap = AudioProvider(
                        settings=resolved,
                        media_store=MediaStore(media_dir / "store"),
                    )
                    rendered_vo = await ap.generate_voice(
                        VoiceCue(
                            shot_id="stitch_vo",
                            text=vo_script[:420],
                            voice=project.voice_name or "Rachel",
                        ),
                        cancel=cancel,
                    )
                    if Path(rendered_vo.path).is_file():
                        vo_path = Path(rendered_vo.path)
                        project.actual_usd = round(project.actual_usd + float(rendered_vo.usd), 4)
                except Exception:
                    vo_path = None

        bed_wav = (
            _synthesize_riser_bed_wav(total_dur, seam_times, tmp_dir / "bed.wav")
            if project.seam_transition == "audio_riser"
            else _synthesize_ambient_cinematic_wav(total_dur, tmp_dir / "bed.wav")
        )

        if aroll_audio_spine is not None and aroll_audio_spine.is_file():
            # Continuous Founder A-Roll voice spine never cuts across visual cutaways
            if project.audio_mode == "native_only":
                mix_cmd = [
                    resolved.ffmpeg_bin,
                    "-y",
                    "-threads",
                    str(eff_threads),
                    "-i",
                    str(assembled_mp4),
                    "-i",
                    str(aroll_audio_spine),
                    "-map",
                    "0:v:0",
                    "-map",
                    "1:a:0",
                    "-c:v",
                    "copy",
                    "-af",
                    "loudnorm=I=-15:TP=-1.5:LRA=11",
                    "-c:a",
                    "aac",
                    "-ar",
                    "48000",
                    "-ac",
                    "2",
                    "-b:a",
                    "192k",
                    "-shortest",
                    "-movflags",
                    "+faststart",
                    str(final_tmp),
                ]
            else:
                mix_cmd = [
                    resolved.ffmpeg_bin,
                    "-y",
                    "-threads",
                    str(eff_threads),
                    "-i",
                    str(assembled_mp4),
                    "-i",
                    str(aroll_audio_spine),
                    "-i",
                    str(bed_wav),
                    "-filter_complex",
                    "[1:a]volume=1.05[vo];[2:a]volume=0.20[bed];[vo][bed]amix=inputs=2:duration=first:dropout_transition=2,loudnorm=I=-14:TP=-1.5:LRA=11[aout]",
                    "-map",
                    "0:v:0",
                    "-map",
                    "[aout]",
                    "-c:v",
                    "copy",
                    "-c:a",
                    "aac",
                    "-ar",
                    "48000",
                    "-ac",
                    "2",
                    "-b:a",
                    "192k",
                    "-shortest",
                    "-movflags",
                    "+faststart",
                    str(final_tmp),
                ]
        elif vo_path is not None:
            mix_cmd = [
                resolved.ffmpeg_bin,
                "-y",
                "-threads",
                str(eff_threads),
                "-i",
                str(assembled_mp4),
                "-i",
                str(vo_path),
                "-i",
                str(bed_wav),
                "-filter_complex",
                "[1:a]volume=1.05[vo];[2:a]volume=0.22[bed];[vo][bed]amix=inputs=2:duration=longest:dropout_transition=2,loudnorm=I=-14:TP=-1.5:LRA=11[aout]",
                "-map",
                "0:v:0",
                "-map",
                "[aout]",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-ar",
                "48000",
                "-ac",
                "2",
                "-b:a",
                "192k",
                "-shortest",
                "-movflags",
                "+faststart",
                str(final_tmp),
            ]
        elif project.audio_mode == "native_only" and any_native_audio:
            mix_cmd = [
                resolved.ffmpeg_bin,
                "-y",
                "-threads",
                str(eff_threads),
                "-i",
                str(assembled_mp4),
                "-c:v",
                "copy",
                "-af",
                "loudnorm=I=-14:TP=-1.5:LRA=11",
                "-c:a",
                "aac",
                "-ar",
                "48000",
                "-ac",
                "2",
                "-b:a",
                "192k",
                "-movflags",
                "+faststart",
                str(final_tmp),
            ]
        elif project.audio_mode == "native_plus_score" and any_native_audio:
            mix_cmd = [
                resolved.ffmpeg_bin,
                "-y",
                "-threads",
                str(eff_threads),
                "-i",
                str(assembled_mp4),
                "-i",
                str(bed_wav),
                "-filter_complex",
                "[0:a]volume=0.95[nat];[1:a]volume=0.30[bed];[nat][bed]amix=inputs=2:duration=first:dropout_transition=2,loudnorm=I=-14:TP=-1.5:LRA=11[aout]",
                "-map",
                "0:v:0",
                "-map",
                "[aout]",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-ar",
                "48000",
                "-ac",
                "2",
                "-b:a",
                "192k",
                "-shortest",
                "-movflags",
                "+faststart",
                str(final_tmp),
            ]
        else:
            mix_cmd = [
                resolved.ffmpeg_bin,
                "-y",
                "-threads",
                str(eff_threads),
                "-i",
                str(assembled_mp4),
                "-i",
                str(bed_wav),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "copy",
                "-af",
                "loudnorm=I=-15:TP=-1.5:LRA=11",
                "-c:a",
                "aac",
                "-ar",
                "48000",
                "-ac",
                "2",
                "-b:a",
                "192k",
                "-shortest",
                "-movflags",
                "+faststart",
                str(final_tmp),
            ]

        rc_mix, _, _ = await run_subprocess(mix_cmd, timeout_sec=180.0, cancel=cancel, check=False)
        if rc_mix != 0 or not final_tmp.is_file():
            shutil.copyfile(assembled_mp4, final_tmp)

        # Optional ASS Captions burn-in (bundled open-license fonts, Render-512MB safe)
        if project.enable_captions:
            ass_path = _write_project_captions_ass(
                project=project,
                clip_durations=clip_durations,
                width=width,
                height=height,
                out_path=tmp_dir / "captions.ass",
            )
            if ass_path is not None and ass_path.is_file():
                cap_tmp = tmp_dir / "master_captioned.mp4"
                fonts_dir_resolved = resolved.fonts_dir
                if not fonts_dir_resolved.exists():
                    fonts_dir_resolved = resolve_font_path("Roboto", bold=True).parent
                esc_ass = str(ass_path.resolve()).replace("\\", "/").replace(":", "\\:")
                esc_fonts = str(fonts_dir_resolved.resolve()).replace("\\", "/").replace(":", "\\:")
                rc_cap, _, _ = await run_subprocess(
                    [
                        resolved.ffmpeg_bin,
                        "-y",
                        "-threads",
                        str(eff_threads),
                        "-i",
                        str(final_tmp),
                        "-vf",
                        f"ass='{esc_ass}':fontsdir='{esc_fonts}'",
                        *_pinned_h264_flags(fps, threads=eff_threads),
                        "-c:a",
                        "copy",
                        "-movflags",
                        "+faststart",
                        str(cap_tmp),
                    ],
                    timeout_sec=180.0,
                    cancel=cancel,
                    check=False,
                )
                if rc_cap == 0 and cap_tmp.is_file():
                    final_tmp = cap_tmp

        master_out = media_dir / f"master_{uuid4().hex[:6]}.mp4"
        _move_output_atomically(final_tmp, master_out)
        project.master_video_path = str(master_out)
        project.campaign_pack_paths[project.aspect_ratio] = str(master_out)
        project.status = "done"
        project.status_message = (
            f"Master Cut ready ({len(norm_paths)} segment(s), {project.seam_transition.replace('_', ' ')})."
        )
        return project
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


async def sync_vo_pacing(
    project: StitchProject,
    media_dir: Path,
    *,
    settings: Settings | None = None,
    mock: bool = False,
    cancel: CancellationToken | None = None,
) -> StitchProject:
    """Measure per-scene VO duration (ElevenLabs or 2.6 wps cadence) and snap motion_graphic durations."""
    resolved = settings or get_settings()
    media_dir.mkdir(parents=True, exist_ok=True)
    synced = 0
    for sc in project.scenes:
        line = (sc.vo_line or "").strip()
        if not line:
            continue
        words_cnt = max(1, len(line.split()))
        est_dur = max(2.2, words_cnt / 2.6)
        if not mock and resolved.fal_key.strip():
            try:
                ap = AudioProvider(settings=resolved, media_store=MediaStore(media_dir / "store"))
                rendered = await ap.generate_voice(
                    VoiceCue(shot_id=f"{sc.scene_id}_vo", text=line[:240], voice=project.voice_name or "Rachel"),
                    cancel=cancel,
                )
                if Path(rendered.path).is_file():
                    vo_p = media_dir / f"{sc.scene_id}_vo.mp3"
                    shutil.copyfile(rendered.path, vo_p)
                    sc.vo_audio_path = str(vo_p)
                    probe = await probe_media(vo_p, ffprobe_bin=resolved.ffprobe_bin, cancel=cancel)
                    est_dur = max(2.0, float(probe.duration_sec or est_dur))
                    project.actual_usd = round(project.actual_usd + float(rendered.usd), 4)
            except Exception:
                pass
        if sc.kind in ("motion_graphic", "kinetic_ui"):
            sc.duration_sec = max(3, min(20, int(math.ceil(est_dur + 0.8))))
        synced += 1
    project.target_duration_sec = project.total_duration_sec()
    project.status_message = f"VO-synced pacing across {synced} scene(s) ({project.total_duration_sec()}s total)."
    return project


async def export_campaign_pack(
    project: StitchProject,
    media_dir: Path,
    *,
    settings: Settings | None = None,
    mock: bool = False,
    cancel: CancellationToken | None = None,
) -> StitchProject:
    """Export 3-format Campaign Pack (16:9, 9:16, 1:1) with native reflowed motion graphics at $0 AI cost."""
    resolved = settings or get_settings()
    media_dir.mkdir(parents=True, exist_ok=True)
    orig_aspect = project.aspect_ratio
    orig_master = project.master_video_path
    orig_takes = [(sc.scene_id, list(sc.takes), sc.winning_take_id) for sc in project.scenes]
    palette = project.lighting.color_palette or ["#191512", "#E54D3A", "#F6E7D4"]
    accent_rgb = _hex_to_rgb(palette[1] if len(palette) > 1 else "#E54D3A", (229, 77, 58))

    pack_paths: dict[str, str] = {}
    try:
        for aspect in ("16:9", "9:16", "1:1"):
            project.aspect_ratio = aspect  # type: ignore[assignment]
            w, h = aspect_to_dimensions(aspect)
            # Natively re-render $0 motion_graphic scenes at each aspect ratio so text/bento/devices never crop
            for idx, sc in enumerate(project.scenes):
                if sc.kind == "motion_graphic" or (sc.kind == "kinetic_ui" and sc.graphic_spec is not None):
                    gspec = sc.graphic_spec or _default_graphic_spec(
                        "prompt_ui_demo",
                        brand=project.brand_name or project.title,
                        idea=project.idea,
                        scraped={},
                        world_id=project.world_preset,
                    )
                    tag = aspect.replace(":", "x")
                    out_clip = media_dir / f"{sc.scene_id}_pack_{tag}.mp4"
                    await render_motion_template_mp4(
                        spec=gspec,
                        output_path=out_clip,
                        width=w,
                        height=h,
                        fps=24,
                        duration_sec=float(sc.duration_sec),
                        brand_name=project.brand_name or project.title,
                        accent_rgb=accent_rgb,
                        ffmpeg_bin=resolved.ffmpeg_bin,
                        ffprobe_bin=resolved.ffprobe_bin,
                        fonts_dir=resolved.fonts_dir,
                        cancel=cancel,
                    )
                    sc.takes = [
                        StitchTake(
                            take_id=1,
                            engine_id=f"motion_graphic:{gspec.template_id}",
                            engine_label=f"Campaign {aspect} ($0)",
                            clip_path=str(out_clip),
                            duration_sec=float(sc.duration_sec),
                            usd=0.0,
                        )
                    ]
                    sc.winning_take_id = 1
                else:
                    # Restore original takes for AI / user clips (reframed automatically by stitch_master_cut at $0)
                    _, saved_takes, saved_win = orig_takes[idx]
                    sc.takes = list(saved_takes)
                    sc.winning_take_id = saved_win

            await stitch_master_cut(project, media_dir, settings=resolved, mock=mock, cancel=cancel)
            if project.master_video_path:
                # Also probe beats on the primary 16:9 master for telemetry
                if aspect == "16:9":
                    await detect_audio_beats(project.master_video_path, ffmpeg_bin=resolved.ffmpeg_bin, cancel=cancel)
                pack_paths[aspect] = project.master_video_path
    finally:
        # Restore original aspect ratio, takes, and primary master
        project.aspect_ratio = orig_aspect
        for idx, sc in enumerate(project.scenes):
            _, saved_takes, saved_win = orig_takes[idx]
            sc.takes = list(saved_takes)
            sc.winning_take_id = saved_win
        project.campaign_pack_paths = pack_paths
        project.master_video_path = pack_paths.get(orig_aspect) or orig_master
        project.status = "done"
        project.status_message = "3-Format Campaign Pack ready (16:9 · 9:16 · 1:1)!"
    return project


__all__ = [
    "StitchProjectStore",
    "compile_stitch_project",
    "export_campaign_pack",
    "generate_all_keyframes",
    "generate_scene_keyframe",
    "pitch_three_concepts",
    "plan_aroll_cutaways",
    "render_all_scenes",
    "render_single_scene_take",
    "scrape_optional_url",
    "stitch_master_cut",
    "sync_vo_pacing",
]
