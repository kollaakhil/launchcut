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
from gtm_videogen.media.ffmpeg_stitcher import _move_output_atomically, _pinned_h264_flags
from gtm_videogen.media.kinetic_motion import render_kinetic_card_mp4
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
    StitchAspect,
    StitchBeat,
    StitchProject,
    StitchScene,
    StitchTake,
    validate_stitch_project,
)
from gtm_videogen.providers.audio import AudioProvider
from gtm_videogen.providers.scraper import BrandScraper
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

    if recipe_id == "miniature_jar_world":
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
        short_title = " ".join(clean_idea.split()[:5]).rstrip(".,;:")
        title = f"{brand}: {short_title}" if brand else (short_title.title() or "Untitled Film")
        main_subject = f"{brand} — {clean_idea[:120]}" if brand else clean_idea[:130]
        arc_specs = [
            (
                "Scene 1 · Tactile Hook",
                [brand or "hero centerpiece", "sculpted foreground detail"],
                base_mats,
                base_cam,
                f"Hero subject illuminated cleanly in frame with warm depth ({clean_idea[:75]})",
                f"Start with the quiet craft of {brand or 'every detail'}.",
            ),
            (
                "Scene 2 · World Transformation",
                [brand or "hero centerpiece", "dynamic light & material interplay"],
                [*base_mats[:2], "volumetric golden caustics"],
                "65mm cinema glide · smooth continuous sculptural orbit",
                "Mid-arc hero peak with luminous reflections and clean silhouette",
                "Light moves across every surface with effortless precision.",
            ),
            (
                "Scene 3 · Hero Pull-Back Reveal",
                [brand or "flagship hero centerpiece", "complete sunlit studio tableau"],
                [*base_mats[:2], "warm natural stone"],
                "85mm cinema lens · sweeping continuous pull-back into hero lock",
                f"Locked-off hero composition framing {brand or 'the centerpiece'} with balanced negative space",
                "Pull back to see the entire story lock into place.",
            ),
            (
                "Scene 4 · Signature Hold",
                [brand or "hero centerpiece"],
                base_mats,
                "85mm locked-off cinema frame · subtle breathing focus",
                "Unforgettable still hero finish with glowing ambient depth",
                f"{brand or 'Designed'} to stay with you.",
            ),
        ]

    scenes: list[StitchScene] = []
    for idx, dur in enumerate(durations):
        spec = arc_specs[min(idx, len(arc_specs) - 1)]
        sc_title, subjs, mats, cam_s, end_f, vo_l = spec
        if idx >= len(arc_specs):
            sc_title = f"Scene {idx + 1} · Continuation"
        continues = idx > 0
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
                kind="ai_sequence",
                continues_from_prev=continues,
                subjects=list(subjs),
                materials=list(mats),
                camera_spec=cam_s,
                colors=list(palette[:3]),
                vo_line=vo_l,
                end_frame=end_f,
                beats=beats,
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

    project = StitchProject(
        project_id=f"st_{uuid4().hex[:10]}",
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
        lighting=LightingSpec.model_validate(lighting_data["lighting"]),
        subject_details=SubjectDetailsSpec(),
        motion=MotionSpec.model_validate(motion_data["motion"]),
        scenes=scenes,
        brand_name=eff_brand,
        brand_tagline=eff_tagline,
        add_brand_outro=bool(eff_brand),
        status="draft",
        status_message=f"Drafted {len(scenes)} Scene(s) ({sum(durations)}s). Review & edit on your Pinboard.",
    )

    # Upgrade with live OpenRouter LLM choreography when live & API key configured
    if not mock and resolved.openrouter_api_key.strip():
        sys_prompt = (
            "You are a Master Film Director crafting a structured multi-scene Stitch Script "
            "for AI video models (Kling 3.0 Pro / Veo 3). Every scene has concrete physical "
            "subjects, materials, camera movement, 3-color hex palette, a locked end_frame, "
            "and 3-4 timed beats (time, scene, action, camera, sound). "
            "RULES: Never include readable text, neon signs, billboards, or watermarks in scene "
            "descriptions. For Scene 2+, Beat 1 MUST explicitly start with 'Continues seamlessly "
            "from Scene N's locked frame...' so Last-Frame Chaining is invisible. "
            "Return ONLY a JSON object with keys: title, main_subject, style, audio_direction, "
            "and scenes (array of exact requested length, each with title, subjects (3 strings), "
            "materials (3 strings), camera_spec, colors (3 hex), vo_line (<=14 words), end_frame, "
            "beats (array of {time, scene, action, camera, sound}))."
        )
        usr_prompt = json.dumps(
            {
                "idea": project.idea,
                "scraped_brand_context": project.scraped_context,
                "brand_name": eff_brand or "",
                "num_scenes": len(durations),
                "scene_durations_sec": durations,
                "aspect_ratio": aspect_ratio,
                "world_style": project.style,
                "camera_preset": camera_data["label"],
                "lighting_preset": lighting_data["label"],
                "motion_preset": motion_data["label"],
            },
            indent=2,
        )
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

        # 2. $0 Kinetic UI Scene OR $0 Offline Animatic Take
        if sc.kind == "kinetic_ui" or use_mock:
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
        return project
    except Exception as exc:
        sc.render_status = "failed"
        sc.render_error = str(exc)
        sc.render_message = f"Render failed: {exc}"
        project.error = str(exc)
        project.status_message = f"{sc.title} failed: {exc}"
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


async def stitch_master_cut(
    project: StitchProject,
    media_dir: Path,
    *,
    settings: Settings | None = None,
    mock: bool = False,
    on_progress: Any | None = None,
    cancel: CancellationToken | None = None,
) -> StitchProject:
    """Assemble all winning Takes + Seam Transitions + Audio Mode + Optional Brand Outro (<5s)."""
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

    winning_clips: list[Path] = []
    for sc in project.scenes:
        win = sc.winning_take()
        if win and Path(win.clip_path).is_file():
            winning_clips.append(Path(win.clip_path))
    if not winning_clips:
        raise RuntimeError("Render at least one Scene take before stitching the Master Cut.")

    project.status = "stitching"
    project.error = None
    await _report(f"Stitching {len(winning_clips)} Scene(s) into Master Cut...")

    tmp_dir = Path(tempfile.mkdtemp(prefix="stitch_master_"))
    try:
        # Optional +2.5s Kinetic Brand Outro
        if project.add_brand_outro and (project.brand_name or project.brand_logo_path):
            outro_mp4 = tmp_dir / "brand_outro.mp4"
            bible = _synthetic_bible_from_project(project)
            palette = project.lighting.color_palette or ["#191512", "#E54D3A", "#F6E7D4"]
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
                accent_rgb=_hex_to_rgb(palette[1] if len(palette) > 1 else "#E54D3A", (229, 77, 58)),
                text_rgb=_hex_to_rgb(palette[2] if len(palette) > 2 else "#F6E7D4", (246, 231, 212)),
                ffmpeg_bin=resolved.ffmpeg_bin,
                ffprobe_bin=resolved.ffprobe_bin,
                fonts_dir=resolved.fonts_dir,
                card_kind="end_card",
                shot_id="brand_outro",
                logo_path=logo_p,
                cancel=cancel,
            )
            winning_clips.append(outro_mp4)

        norm_paths: list[Path] = []
        clip_durations: list[float] = []
        any_native_audio = False
        preserve_native = project.audio_mode in ("native_only", "native_plus_score")

        for idx, clip_p in enumerate(winning_clips):
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

        if vo_path is not None:
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

        master_out = media_dir / f"master_{uuid4().hex[:6]}.mp4"
        _move_output_atomically(final_tmp, master_out)
        project.master_video_path = str(master_out)
        project.status = "done"
        project.status_message = (
            f"Master Cut ready ({len(winning_clips)} clip(s), {project.seam_transition.replace('_', ' ')})."
        )
        return project
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


__all__ = [
    "StitchProjectStore",
    "compile_stitch_project",
    "generate_all_keyframes",
    "generate_scene_keyframe",
    "pitch_three_concepts",
    "render_all_scenes",
    "render_single_scene_take",
    "scrape_optional_url",
    "stitch_master_cut",
]
