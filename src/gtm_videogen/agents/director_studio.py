"""Guided Structured JSON Director service (10s / 20s / 30s Chained Acts).

Turns casual user ideas + clickable visual direction cards into hyper-detailed,
model-native structured JSON video blueprints, supports 1-click plain-English
refinements, continuity-locked Flux Schnell/Kontext anchor keyframes, Last-Frame
Chaining (pixel-exact Act 1 -> Act 2 -> Act 3 handoff), Parallel Start->End
Bridging, Veo 3 native audio, and Render-512MB-safe FFmpeg final assembly.
"""

from __future__ import annotations

import asyncio
import base64
import io
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

from PIL import Image, ImageDraw, ImageFont

from gtm_videogen.core.config import Settings, get_settings
from gtm_videogen.core.cost_meter import CostMeter
from gtm_videogen.core.http_client import ResilientHttpClient
from gtm_videogen.core.job_registry import CancellationToken
from gtm_videogen.core.media_store import MediaStore
from gtm_videogen.media._subprocess import aspect_to_dimensions, probe_media, run_subprocess
from gtm_videogen.media.captions import resolve_font_path
from gtm_videogen.media.ffmpeg_stitcher import _move_output_atomically, _pinned_h264_flags
from gtm_videogen.media.kinetic_motion import render_kinetic_card_mp4
from gtm_videogen.models.director_blueprint import (
    CAMERA_PRESETS,
    DIRECTOR_ENGINES,
    LIGHTING_PRESETS,
    MOTION_PRESETS,
    WORLD_PRESETS,
    CameraSpec,
    DirectorAct,
    DirectorAspect,
    DirectorBlueprint,
    DirectorContinuityMode,
    DirectorDuration,
    DirectorFormatMode,
    EnvironmentSpec,
    FinalFrameSpec,
    LightingSpec,
    MotionSpec,
    SubjectDetailsSpec,
    TimelineBeat,
)
from gtm_videogen.models.schemas import (
    BeatSpec,
    CameraGrammar,
    CreativeBible,
    NarrativeStructure,
    Shot,
    SonicIdentity,
    TypographyAndLayout,
    VisualLanguage,
)
from gtm_videogen.providers.video import VideoProvider

FLUX_UNIT_USD = 0.005
KONTEXT_UNIT_USD = 0.025
KONTEXT_MODEL = "fal-ai/flux-kontext/dev"

_ASPECT_FLUX_SIZE: dict[str, str] = {
    "16:9": "landscape_16_9",
    "9:16": "portrait_16_9",
    "1:1": "square_hd",
}


class DirectorBlueprintStore:
    """Atomic filesystem JSON store for Director Blueprints under `<data_dir>/director`."""

    def __init__(self, root_dir: Path | str) -> None:
        self.root_dir = Path(root_dir).resolve()
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def blueprint_dir(self, blueprint_id: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9_-]", "", blueprint_id)
        if not safe:
            raise ValueError("Invalid blueprint_id")
        return self.root_dir / safe

    def media_dir(self, blueprint_id: str) -> Path:
        p = self.blueprint_dir(blueprint_id) / "media"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def save(self, blueprint: DirectorBlueprint) -> DirectorBlueprint:
        blueprint.updated_at = datetime.now(UTC).isoformat()
        bp_dir = self.blueprint_dir(blueprint.blueprint_id)
        bp_dir.mkdir(parents=True, exist_ok=True)
        target = bp_dir / "blueprint.json"
        payload = blueprint.model_dump_json(indent=2)
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=bp_dir,
            prefix=".blueprint.",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            tmp.write(payload)
            tmp_path = Path(tmp.name)
        os.replace(tmp_path, target)
        return blueprint

    def load(self, blueprint_id: str) -> DirectorBlueprint | None:
        target = self.blueprint_dir(blueprint_id) / "blueprint.json"
        if not target.is_file():
            return None
        return DirectorBlueprint.model_validate_json(target.read_text(encoding="utf-8"))

    def list_all(self, limit: int = 30) -> list[DirectorBlueprint]:
        items: list[DirectorBlueprint] = []
        if not self.root_dir.exists():
            return items
        for child in self.root_dir.iterdir():
            bp_file = child / "blueprint.json"
            if bp_file.is_file():
                try:
                    items.append(
                        DirectorBlueprint.model_validate_json(bp_file.read_text(encoding="utf-8"))
                    )
                except Exception:
                    continue
        items.sort(key=lambda b: b.updated_at, reverse=True)
        return items[:limit]


def _engine_meta(engine_id: str) -> dict[str, Any]:
    for item in DIRECTOR_ENGINES:
        if item["id"] == engine_id:
            return item
    return DIRECTOR_ENGINES[0]


def estimate_blueprint_cost(blueprint: DirectorBlueprint) -> float:
    """Estimate USD render cost based on active AI Acts and chosen engine."""
    meta = _engine_meta(blueprint.engine)
    per_act = float(meta.get("est_usd_per_act", 0.0))
    ai_acts = sum(1 for a in blueprint.acts if a.mode == "cinematic_sequence")
    return round(per_act * ai_acts, 2)


def _strip_json_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _hex_to_rgb(hex_str: str, fallback: tuple[int, int, int]) -> tuple[int, int, int]:
    cleaned = (hex_str or "").strip().lstrip("#")
    if len(cleaned) == 6:
        try:
            return (int(cleaned[0:2], 16), int(cleaned[2:4], 16), int(cleaned[4:6], 16))
        except ValueError:
            pass
    return fallback


def _synthetic_bible_from_blueprint(bp: DirectorBlueprint) -> CreativeBible:
    """Build a lightweight CreativeBible so KineticMotionRenderer inherits the blueprint palette."""
    palette = list(bp.lighting.color_palette or ["#111318", "#E2A336", "#F5E9D6"])
    while len(palette) < 3:
        palette.append("#F5E9D6")
    return CreativeBible(
        concept_thesis=bp.main_subject or bp.title,
        narrative_structure=NarrativeStructure(
            arc_name="hero_sequence",
            beats=[
                BeatSpec(
                    name="hero",
                    purpose="visual hero reveal",
                    duration_hint_sec=float(bp.duration_sec),
                )
            ],
        ),
        visual_language=VisualLanguage(
            world=bp.environment.world,
            lighting_style=bp.lighting.style,
            color_palette=palette[:4],
            texture_and_grade=bp.style,
        ),
        camera_grammar=CameraGrammar(
            dominant_style=bp.camera.style,
            allowed_moves=[bp.camera.movement],
            lens_range_mm=f"{bp.camera.lens_mm}mm",
        ),
        typography_and_layout=TypographyAndLayout(
            font_family="Roboto",
            end_card_layout="centered_minimal",
        ),
        sonic_identity=SonicIdentity(
            audio_strategy=bp.audio_direction or "tactile ambient foley",
        ),
    )


def _build_deterministic_acts(
    *,
    idea: str,
    duration_sec: DirectorDuration,
    format_mode: DirectorFormatMode,
    world_data: dict[str, Any],
    camera_data: dict[str, Any],
    lighting_data: dict[str, Any],
    motion_data: dict[str, Any],
    recipe_id: str | None,
    brand_name: str | None,
    brand_tagline: str | None,
    act_dur: int,
) -> tuple[str, str, list[DirectorAct]]:
    """Produce a rich, cinema-grade structured multi-act choreography offline/deterministically."""
    num_acts = 1 if duration_sec == 10 else (2 if duration_sec == 20 else 3)
    clean_idea = idea.strip() or "a tiny living miniature world awakening on a wooden workshop table"
    brand = (brand_name or "").strip()
    tagline = (brand_tagline or "").strip()
    surface = world_data["environment"]["surface"]
    world_desc = world_data["environment"]["world"]
    cam_style = camera_data["camera"]["style"]

    # Special flagship fidelity for the Miniature World in a Jar diorama recipe
    if recipe_id == "miniature_jar_world" and not brand:
        title = "Miniature World in a Jar"
        main_subject = "a tiny living miniature world manipulated by a giant realistic human hand"
        act1_beats = [
            TimelineBeat(
                time="0-2s",
                scene="Macro close-up of a dull gray miniature city street. Tiny people are frozen mid-walk and miniature cars are completely stopped. A giant realistic human hand enters from the right holding a small glass dropper filled with glowing liquid gold.",
                action="One glowing golden drop falls from the dropper in slow motion and strikes the miniature street, creating a subtle soft shockwave.",
                camera="Extreme macro close-up with shallow depth of field.",
                sound="Soft glass pipette click followed by a resonant slow-motion liquid drop impact.",
            ),
            TimelineBeat(
                time="2-4s",
                scene="The golden liquid spreads across the miniature street like living light.",
                action="Gray buildings rapidly bloom into warm colorful shops. Tiny neon signs flicker on. Miniature cars begin moving naturally and tiny people come alive, laughing, walking, and interacting.",
                camera="Slow cinematic push-in following the spreading golden light.",
                sound="Warm crystalline shimmer blooming outward with gentle miniature street ambience.",
            ),
            TimelineBeat(
                time="4-6s",
                scene="The giant hand gently tilts the wooden table.",
                action="The miniature city street rolls and folds like a physical paper map, transforming seamlessly into a tiny mountain village. Pine trees pop up one by one from the ground. A miniature river naturally carves through the valley with realistic flowing-water physics.",
                camera="Smooth tracking movement following the transformation.",
                sound="Tactile heavy paper folding rustle, soft wooden pops of trees, crisp babbling water.",
            ),
            TimelineBeat(
                time="6-8s",
                scene="The completed miniature mountain village glows beneath a cloudy sky.",
                action="The giant fingertip slowly drags across the sky above the village. Dark storm clouds peel away as a warm glowing sunset spills across the mountains. Tiny fireflies appear and miniature lanterns gradually illuminate rooftops.",
                camera="Gentle cinematic upward movement transitioning from the fingertip to the glowing sunset.",
                sound="Soft airy sweep as clouds part, delicate firefly chimes and warm lantern crackle.",
            ),
            TimelineBeat(
                time="8-10s",
                scene="The camera rapidly pulls backward in one continuous smooth cinematic movement.",
                action="The entire miniature world is revealed to be contained inside a transparent glass jar sitting on the wooden workshop table. The giant realistic hand places and screws the lid onto the jar while the tiny village lights continue glowing warmly inside.",
                camera="Fast but smooth pull-back ending on a locked-off hero shot of the glowing glass jar.",
                sound="Smooth airy whoosh pull-back, crisp glass-on-metal lid twist, warm resonant hold.",
            ),
        ]
        acts: list[DirectorAct] = [
            DirectorAct(
                act_number=1,
                title="Act 1 · Golden Awakening & Jar Reveal" if num_acts == 1 else "Act 1 · Golden Awakening",
                time_range=f"0-{act_dur}s",
                duration_sec=act_dur,
                mode="cinematic_sequence",
                continues_from_prev=False,
                timeline=act1_beats[:4] if act_dur <= 8 else act1_beats,
                final_frame=FinalFrameSpec(
                    subject="glowing miniature mountain village inside a transparent glass jar",
                    composition="hero macro shot on the wooden workshop table",
                    hold="brief cinematic hold",
                ),
            )
        ]
        if num_acts >= 2:
            acts[0].timeline[-1] = TimelineBeat(
                time="8-10s",
                scene="Golden lanterns and fireflies pulse warmly across the miniature mountain valley.",
                action="A shimmering river of golden light spills over the mountain ridge, cascading toward an untouched wooden workbench horizon.",
                camera="Smooth sweeping crane tracking the cascading golden light stream.",
                sound="Flowing water swell blending into a warm crystalline riser.",
            )
            acts[0].final_frame = FinalFrameSpec(
                subject="golden glowing miniature valley spilling light toward the wooden table edge",
                composition="intimate macro tracking composition ready to expand",
                hold="seamless motion momentum into Act 2",
            )
            acts.append(
                DirectorAct(
                    act_number=2,
                    title="Act 2 · Living World Expansion",
                    time_range=f"{act_dur}-{act_dur * 2}s",
                    duration_sec=act_dur,
                    mode="cinematic_sequence",
                    continues_from_prev=True,
                    timeline=[
                        TimelineBeat(
                            time="0-2s",
                            scene="Continues directly from the glowing valley ridge where golden light flows across the table grain.",
                            action="Miniature brass bridges and tiny sunlit harbors assemble piece by piece along the riverbanks.",
                            camera="Smooth low-angle macro glide skimming over the glowing water.",
                            sound="Delicate brass clicks and gentle lapping water.",
                        ),
                        TimelineBeat(
                            time="2-4s",
                            scene="A tiny clockwork windmill and miniature sailboats appear on the sparkling harbor.",
                            action="The giant hand softly blows a breath of warm mist; windmill sails spin and tiny sails billow forward smoothly.",
                            camera="Gentle orbital arc around the sunlit miniature harbor.",
                            sound="Soft breeze exhale, tiny creaking wooden sails, delicate water ripples.",
                        ),
                        TimelineBeat(
                            time="4-6s",
                            scene="Twilight descends across the handcrafted valley as tiny windows ignite one by one.",
                            action="Golden tram cars glide silently across miniature stone viaducts, casting warm reflections on the water below.",
                            camera="Slow rising crane revealing the interconnected city, mountain, and harbor.",
                            sound="Soft distant bell chime and tranquil ambient warmth.",
                        ),
                        TimelineBeat(
                            time="6-8s",
                            scene="The entire glowing miniature landscape shimmers beneath drifting volumetric dust motes.",
                            action="Curved transparent glass walls gently descend from above around the perimeter of the glowing valley.",
                            camera="Steady controlled pull-back framing the curved glass refraction.",
                            sound="Pure resonant glass hum and soft airy shimmer.",
                        ),
                        TimelineBeat(
                            time="8-10s",
                            scene="Warm reflections curve across the crystal-clear glass enclosure surrounding the living diorama.",
                            action="The miniature world settles into serene glowing equilibrium inside the glass cylinder.",
                            camera="Smooth pullback holding crisp edge refractions on the glass.",
                            sound="Warm harmonic swell settling softly.",
                        ),
                    ][:4 if act_dur <= 8 else 5],
                    final_frame=FinalFrameSpec(
                        subject="complete illuminated miniature world framed inside crystal-clear glass walls",
                        composition="balanced mid-macro hero framing on the wooden table",
                        hold="seamless motion handoff into Act 3" if num_acts == 3 else "brief cinematic hold",
                    ),
                )
            )
        if num_acts == 3:
            acts.append(
                DirectorAct(
                    act_number=3,
                    title="Act 3 · Glass Jar Reveal & Final Seal",
                    time_range=f"{act_dur * 2}-{act_dur * 3}s",
                    duration_sec=act_dur,
                    mode="cinematic_sequence",
                    continues_from_prev=True,
                    timeline=[
                        TimelineBeat(
                            time="0-2s",
                            scene="Continues seamlessly on the crystal glass enclosure glowing warmly on the workshop table.",
                            action="The giant realistic hand reaches into frame holding a weathered brushed-brass jar lid.",
                            camera="Smooth backward dolly opening up the workshop environment.",
                            sound="Soft metallic touch and quiet workshop room tone.",
                        ),
                        TimelineBeat(
                            time="2-4s",
                            scene="The brushed-brass lid aligns precisely over the rim of the glowing glass jar.",
                            action="The fingers gently screw the lid closed with tactile precision; warm golden fireflies swirl upward inside.",
                            camera="Slow deliberate push-out holding shallow bokeh on the hand and jar threads.",
                            sound="Crisp satisfying glass-and-metal thread twist.",
                        ),
                        TimelineBeat(
                            time="4-6s",
                            scene="The sealed jar glows like a captured constellation on the real wooden workshop table.",
                            action="The giant hand releases the lid and withdraws smoothly out of frame as volumetric sunbeams sweep across the glass.",
                            camera="Silky continuous pull-back revealing warm bokeh workshop tools in the distance.",
                            sound="Gentle resonant glass settle and expanding warm harmonic chord.",
                        ),
                        TimelineBeat(
                            time="6-8s",
                            scene="Inside the jar, tiny lanterns flicker softly and the miniature river catches golden specular light.",
                            action="Golden volumetric light rays kiss the curved glass surface, casting intricate warm caustics across the oak wood grain.",
                            camera="Micro-slow heroic orbit settling dead-center on the jar.",
                            sound="Serene crystalline shimmer fading cleanly.",
                        ),
                        TimelineBeat(
                            time="8-10s",
                            scene="Locked-off hero portrait of the glowing miniature world sealed safely inside the glass jar.",
                            action="Subtle drifting dust motes and tiny twinkling village lights breathe quietly during a prestigious final hold.",
                            camera="Locked-off hero macro shot with creamy cinematic bokeh.",
                            sound="Warm resonant low-end fade to clean silence.",
                        ),
                    ][:4 if act_dur <= 8 else 5],
                    final_frame=FinalFrameSpec(
                        subject="glowing miniature world sealed inside a transparent glass jar",
                        composition="locked-off hero macro shot on the wooden workshop table",
                        hold="brief cinematic hold",
                    ),
                )
            )
        return title, main_subject, acts

    # General bespoke expansion for ANY user idea + product/brand (10s / 20s / 30s)
    short_title = brand or " ".join(clean_idea.split()[:6]).rstrip(".,!?").title() or "Hero Cinema Sequence"
    hero_label = brand if brand else "the hero centerpiece"
    main_subject = f"{clean_idea.rstrip('.')} centered around {hero_label}"

    def _beats_for_act(act_num: int) -> list[TimelineBeat]:
        if act_num == 1:
            beats = [
                TimelineBeat(
                    time="0-2s",
                    scene=f"Extreme opening macro framing inside {world_desc} resting on {surface}. The subject ({clean_idea[:110]}) begins in poised, tactile stillness.",
                    action="A first catalytic spark of light and physical motion enters frame in silky slow motion, awakening the textures.",
                    camera=f"Extreme macro opening ({cam_style}) with razor-shallow depth of field.",
                    sound="Quiet tactile room tone broken by a crisp catalytic click and soft resonant shimmer.",
                ),
                TimelineBeat(
                    time="2-4s",
                    scene="Luminous highlights race smoothly across micro-textures and physical edges.",
                    action=f"Materials respond with believable physics as {hero_label} begins emerging with unmistakable craft and precision.",
                    camera="Slow deliberate push-in tracking the wave of illumination.",
                    sound="Warm rising harmonic hum layered with subtle physical material rustle.",
                ),
                TimelineBeat(
                    time="4-6s",
                    scene="Mid-transformation hero perspective revealing deeper architectural geometry and depth.",
                    action="Key structural elements unfold, align, and lock into motion with effortless fluid continuity.",
                    camera="Smooth continuous orbital tracking following the physical transformation.",
                    sound="Satisfying tactile mechanical glide and resonant fluid swell.",
                ),
                TimelineBeat(
                    time="6-8s",
                    scene="Volumetric light rays cut cleanly across the evolving world as atmospheric haze catches specular reflections.",
                    action="Secondary ambient details ignite softly around the hero form, creating rich foreground-to-background depth.",
                    camera="Gentle upward crane tilt gliding toward the illuminated crest.",
                    sound="Airy volumetric sweep and crystalline high-frequency sparkle.",
                ),
                TimelineBeat(
                    time="8-10s",
                    scene=(
                        f"Sweeping continuous pull-back revealing the complete {hero_label} resting cleanly on {surface}."
                        if num_acts == 1
                        else f"Dynamic bridging momentum carrying the glowing transformation smoothly across {surface}."
                    ),
                    action=(
                        "All motion resolves into a balanced, sculptural hero silhouette with subtle living highlights."
                        if num_acts == 1
                        else "Light and camera momentum sweep forward seamlessly, teeing up the next Act's expansion."
                    ),
                    camera=(
                        "Fast, silky pull-back ending on a locked-off hero composition."
                        if num_acts == 1
                        else "Continuous tracking glide maintaining exact lens focus and momentum."
                    ),
                    sound="Confident warm sub-bass swell resolving cleanly.",
                ),
            ]
        elif act_num == 2:
            beats = [
                TimelineBeat(
                    time="0-2s",
                    scene=f"Seamlessly continues from Act 1's exact final frame on {surface}, preserving identical lighting and lens grain.",
                    action=f"The world opens outward in fluid choreography, revealing intricate functional craft inside {hero_label}.",
                    camera="Unbroken forward glide weaving smoothly past foreground bokeh into hero detail.",
                    sound="Seamless acoustic continuation with crisp tactile micro-clicks.",
                ),
                TimelineBeat(
                    time="2-4s",
                    scene="Intimate close-up on the signature material interaction and core mechanism in live motion.",
                    action="Precision components and luminous reflections dance in hypnotic, ASMR-grade physical harmony.",
                    camera="Controlled precision macro orbit locking tack-sharp focus on specular bevels.",
                    sound="Silky mechanical precision and resonant glass/metal harmonics.",
                ),
                TimelineBeat(
                    time="4-6s",
                    scene="Broader contextual perspective showing how the surrounding environment responds to the hero's energy.",
                    action="Warm volumetric light spills across the tabletop surface, sculpting crisp caustics and organic shadows.",
                    camera="Smooth sweeping crane arc elevating gently above the scene.",
                    sound="Expanding cinematic air and warm tactile swell.",
                ),
                TimelineBeat(
                    time="6-8s",
                    scene="Peak visual complexity: every material layer, reflection, and micro-particle aligns in effortless symmetry.",
                    action="Kinetic elements converge cleanly toward center frame as excess motion distills into pure clarity.",
                    camera="Steadying gimbal glide aligning symmetrically with the hero axis.",
                    sound="Rising tonal shimmer building anticipation toward the climax.",
                ),
                TimelineBeat(
                    time="8-10s",
                    scene=f"Poised climax framing of {hero_label} bathed in signature {lighting_data['lighting']['style']}.",
                    action="Subtle glowing energy settles smoothly across the hero silhouette, preparing the grand final reveal.",
                    camera="Deliberate poised pull-back transitioning smoothly into Act 3." if num_acts == 3 else "Smooth heroic pull-back settling into locked-off final composition.",
                    sound="Warm resonant crest holding clean studio clarity.",
                ),
            ]
        else:
            beats = [
                TimelineBeat(
                    time="0-2s",
                    scene=f"Continues seamlessly from Act 2's exact final frame, carrying identical lighting and textures on {surface}.",
                    action=f"The final tactile finishing gesture completes {hero_label} with unmistakable physical presence.",
                    camera="Silky backward dolly expanding the frame with prestigious cinema weight.",
                    sound="Crisp tactile lock-in click followed by expansive airy bloom.",
                ),
                TimelineBeat(
                    time="2-4s",
                    scene="Full sculptural silhouette of the hero centerpiece catching sculpted rim highlights and volumetric rays.",
                    action="Gentle atmospheric particles drift through the warm beam as reflections settle across polished surfaces.",
                    camera="Slow, confident heroic arc around the finished subject.",
                    sound="Rich warm harmonic chord swelling effortlessly.",
                ),
                TimelineBeat(
                    time="4-6s",
                    scene=f"Grand contextual reveal placing {hero_label} harmoniously within {world_desc}.",
                    action="Natural caustics and soft directional shadows anchor the object with unmistakable photographic realism.",
                    camera="Smooth unbroken pull-back balancing generous negative space around the hero.",
                    sound="Deep resonant cinema warmth with subtle tactile air.",
                ),
                TimelineBeat(
                    time="6-8s",
                    scene="Iconic centered hero frame illuminated by balanced key and rim light.",
                    action="Micro-highlights breathe gently along the edges as the world reaches serene, unforgettable stillness.",
                    camera="Imperceptible slow settling push-out into exact lock-off.",
                    sound="Crystalline shimmer tail softening into pure room tone.",
                ),
                TimelineBeat(
                    time="8-10s",
                    scene=f"Locked-off iconic final hero shot of {hero_label} on {surface}.",
                    action="Subtle living specular glow and floating dust motes maintain photographic life during a confident hold.",
                    camera="Locked-off hero macro shot with creamy studio bokeh.",
                    sound="Warm low-end resolution fading cleanly to silence.",
                ),
            ]
        return beats[:4] if act_dur <= 8 else beats

    acts_out: list[DirectorAct] = []
    act_titles = [
        "Act 1 · Hook & Visual Spark",
        "Act 2 · Metamorphosis & Craft",
        "Act 3 · Grand Hero Reveal",
    ]
    for i in range(num_acts):
        act_num = i + 1
        is_hybrid_ui = format_mode == "hybrid_launch" and num_acts == 3 and act_num == 2
        t_start = i * act_dur
        t_end = (i + 1) * act_dur
        if is_hybrid_ui:
            acts_out.append(
                DirectorAct(
                    act_number=act_num,
                    title="Act 2 · Real Product / UI Showcase ($0 Kinetic)",
                    time_range=f"{t_start}-{t_end}s",
                    duration_sec=act_dur,
                    mode="kinetic_ui_showcase",
                    continues_from_prev=False,
                    on_screen_headline=f"{brand or short_title} in Action",
                    on_screen_subline=tagline or "Crafted for speed, clarity, and effortless control.",
                    timeline=[
                        TimelineBeat(
                            time="0-4s",
                            scene="Illuminated dark studio bezel glides smoothly into center frame showcasing the real product interface.",
                            action="Staggered spring-eased kinetic headline reveals cleanly above the floating UI showcase.",
                            camera="Precision editorial studio framing with subtle radial glow.",
                            sound="Clean muted sub-whoosh and soft tactile click.",
                        ),
                        TimelineBeat(
                            time="4-10s",
                            scene="Spotlight sweep glides diagonally across the product screen as a precision cursor highlights key value.",
                            action="Cursor glides to the hero action with a crisp golden click ripple before handing back to Act 3.",
                            camera="Locked studio composition with continuous specular shimmer.",
                            sound="Crisp glass click and warm transition riser into Act 3.",
                        ),
                    ],
                    final_frame=FinalFrameSpec(
                        subject=f"{brand or short_title} floating studio UI showcase",
                        composition="centered editorial bezel with clean negative space",
                        hold="crisp transition into Act 3 hero reveal",
                    ),
                )
            )
        else:
            acts_out.append(
                DirectorAct(
                    act_number=act_num,
                    title=act_titles[i],
                    time_range=f"{t_start}-{t_end}s",
                    duration_sec=act_dur,
                    mode="cinematic_sequence",
                    continues_from_prev=(act_num > 1 and not (format_mode == "hybrid_launch" and act_num == 3)),
                    timeline=_beats_for_act(act_num),
                    final_frame=FinalFrameSpec(
                        subject=f"{hero_label} in {world_desc}",
                        composition=f"locked-off hero composition on {surface}",
                        hold="brief cinematic hold" if act_num == num_acts else f"seamless handoff into Act {act_num + 1}",
                    ),
                )
            )

    return short_title, main_subject, acts_out


async def _call_openrouter_json(
    *,
    system_prompt: str,
    user_prompt: str,
    settings: Settings,
    max_tokens: int = 4500,
    temperature: float = 0.75,
    cancel: CancellationToken | None = None,
) -> dict[str, Any] | None:
    """Invoke OpenRouter chat completions expecting a strict JSON object."""
    api_key = settings.openrouter_api_key.strip()
    if not api_key:
        return None
    model = (settings.flagship_text_model or settings.fast_text_model or "openai/gpt-4o-mini").strip()
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://launchcut.onrender.com",
        "X-Title": "LaunchCut JSON Director",
    }
    async with ResilientHttpClient(
        connect_timeout=settings.http_connect_timeout_sec,
        read_timeout=75.0,
        write_timeout=30.0,
        pool_timeout=15.0,
        max_retries=1,
    ) as client:
        data = await client.post_json(
            f"{settings.openrouter_base_url.rstrip('/')}/chat/completions",
            json=payload,
            headers=headers,
            cancel=cancel,
        )
    choices = data.get("choices") if isinstance(data, dict) else None
    if not isinstance(choices, list) or not choices:
        return None
    msg = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = msg.get("content") if isinstance(msg, dict) else None
    if not isinstance(content, str) or not content.strip():
        return None
    parsed = json.loads(_strip_json_fences(content))
    return parsed if isinstance(parsed, dict) else None


async def generate_director_blueprint(
    *,
    idea: str,
    duration_sec: DirectorDuration = 10,
    format_mode: DirectorFormatMode = "continuous",
    continuity_mode: DirectorContinuityMode = "last_frame_chain",
    aspect_ratio: DirectorAspect = "16:9",
    engine: str = "fal-ai/kling-video/v3/pro/text-to-video",
    native_audio: bool = True,
    add_brand_endcard: bool = False,
    world_preset: str = "miniature_diorama",
    camera_preset: str = "macro_tilt_shift",
    lighting_preset: str = "volumetric_gold",
    motion_preset: str = "asmr_tactile",
    recipe_id: str | None = None,
    brand_name: str | None = None,
    brand_tagline: str | None = None,
    brand_url: str | None = None,
    settings: Settings | None = None,
    mock: bool = False,
    cancel: CancellationToken | None = None,
) -> DirectorBlueprint:
    """Compile casual user intent + visual preset cards into a hyper-detailed JSON Blueprint."""
    resolved_settings = settings or get_settings()
    world_data = WORLD_PRESETS.get(world_preset, WORLD_PRESETS["miniature_diorama"])
    camera_data = CAMERA_PRESETS.get(camera_preset, CAMERA_PRESETS["macro_tilt_shift"])
    lighting_data = LIGHTING_PRESETS.get(lighting_preset, LIGHTING_PRESETS["volumetric_gold"])
    motion_data = MOTION_PRESETS.get(motion_preset, MOTION_PRESETS["asmr_tactile"])

    eng_meta = _engine_meta(engine)
    act_dur = int(eng_meta.get("act_duration_sec", 10))
    if duration_sec == 10 and act_dur == 8 or duration_sec in (20, 30) and act_dur == 8:
        act_dur = 8

    # Start with rock-solid deterministic structure as baseline + fallback
    base_title, base_subject, base_acts = _build_deterministic_acts(
        idea=idea,
        duration_sec=duration_sec,
        format_mode=format_mode,
        world_data=world_data,
        camera_data=camera_data,
        lighting_data=lighting_data,
        motion_data=motion_data,
        recipe_id=recipe_id,
        brand_name=brand_name,
        brand_tagline=brand_tagline,
        act_dur=act_dur,
    )

    bp = DirectorBlueprint(
        blueprint_id=f"bp_{uuid4().hex[:10]}",
        title=base_title,
        duration=f"{duration_sec} seconds",
        duration_sec=duration_sec,
        format_mode=format_mode,
        continuity_mode=continuity_mode,
        aspect_ratio=aspect_ratio,
        resolution="4K",
        engine=engine,
        native_audio=native_audio,
        add_brand_endcard=add_brand_endcard,
        brand_name=(brand_name or "").strip() or None,
        brand_tagline=(brand_tagline or "").strip() or None,
        brand_url=(brand_url or "").strip() or None,
        user_idea=idea.strip(),
        recipe_id=recipe_id,
        world_preset=world_preset,
        camera_preset=camera_preset,
        lighting_preset=lighting_preset,
        motion_preset=motion_preset,
        style=world_data["style"],
        main_subject=base_subject,
        camera=CameraSpec(**camera_data["camera"]),
        environment=EnvironmentSpec(**world_data["environment"]),
        lighting=LightingSpec(**lighting_data["lighting"]),
        subject_details=SubjectDetailsSpec(**world_data["subject_details"]),
        motion=MotionSpec(**motion_data["motion"]),
        audio_direction=motion_data["audio_direction"],
        acts=base_acts,
    )

    # Upgrade with live flagship LLM choreography when OpenRouter is configured and not mock
    if not mock and resolved_settings.openrouter_api_key.strip() and idea.strip():
        num_acts = len(bp.acts)
        beats_per_act = 4 if act_dur <= 8 else 5
        sys_prompt = (
            "You are an Academy-Award-caliber Commercial Director & Cinematographer specializing in "
            "hyper-detailed structured JSON video prompts for Kling v4 Pro and Google Veo 3.\n"
            "Transform the user's simple idea and visual selections into breathtaking, physically grounded, "
            "2-second choreographed timelines with zero garbled text, zero cliches, and exact tactile realism.\n"
            "Return ONLY a valid JSON object matching this schema:\n"
            "{\n"
            '  "title": "3-6 word evocative title",\n'
            '  "style": "ultra-photorealistic ...",\n'
            '  "main_subject": "clear single sentence hero subject",\n'
            '  "camera": {"style": "...", "depth_of_field": "...", "focus": "...", "movement": "...", "lens_mm": 85},\n'
            '  "environment": {"surface": "...", "world": "...", "materials": "...", "atmosphere": "..."},\n'
            '  "lighting": {"style": "...", "effects": ["...", "..."], "color_palette": ["#hex1", "#hex2", "#hex3", "#hex4"]},\n'
            '  "subject_details": {"appearance": "...", "textures": "...", "interaction": "...", "scale": "..."},\n'
            '  "motion": {"style": "...", "physics": "...", "feel": "..."},\n'
            '  "audio_direction": "specific ASMR/foley/acoustic soundscape",\n'
            '  "acts": [\n'
            "    {\n"
            '      "act_number": 1,\n'
            '      "title": "Act 1 · ...",\n'
            '      "timeline": [\n'
            '        {"time": "0-2s", "scene": "...", "action": "...", "camera": "...", "sound": "..."}\n'
            "      ],\n"
            '      "final_frame": {"subject": "...", "composition": "...", "hold": "brief cinematic hold"}\n'
            "    }\n"
            "  ]\n"
            "}\n"
            "RULES:\n"
            "- Write vivid, concrete physical details (skin pores, wood grain, liquid surface tension, brass bevels, glass caustics).\n"
            "- Never ask the video model to render readable signs, words, or typography.\n"
            "- When there are 2 or 3 Acts, Act N+1 beat 0-2s MUST physically continue from Act N's final_frame with zero jump cut.\n"
            f"- Produce exactly {num_acts} act(s), each with {beats_per_act} sequential 2-second timeline beats."
        )
        usr_prompt = json.dumps(
            {
                "user_idea": idea.strip(),
                "brand_name": bp.brand_name,
                "brand_tagline": bp.brand_tagline,
                "duration_sec": duration_sec,
                "format_mode": format_mode,
                "num_acts": num_acts,
                "act_duration_sec": act_dur,
                "selected_visual_cards": {
                    "world": world_data["label"],
                    "world_seed": world_data,
                    "camera": camera_data["label"],
                    "camera_seed": camera_data["camera"],
                    "lighting": lighting_data["label"],
                    "lighting_seed": lighting_data["lighting"],
                    "motion": motion_data["label"],
                    "motion_seed": motion_data["motion"],
                },
            },
            indent=2,
        )
        try:
            llm_doc = await _call_openrouter_json(
                system_prompt=sys_prompt,
                user_prompt=usr_prompt,
                settings=resolved_settings,
                max_tokens=5000 if num_acts > 1 else 3000,
                temperature=0.78,
                cancel=cancel,
            )
            if llm_doc:
                _merge_llm_doc_into_blueprint(bp, llm_doc, act_dur=act_dur)
        except Exception:
            # Graceful fallback to our rich deterministic blueprint
            pass

    bp.estimated_usd = estimate_blueprint_cost(bp)
    return bp


def _merge_llm_doc_into_blueprint(
    bp: DirectorBlueprint,
    doc: dict[str, Any],
    *,
    act_dur: int,
) -> None:
    """Safely overlay validated LLM-generated Director JSON onto an existing blueprint."""
    if isinstance(doc.get("title"), str) and doc["title"].strip():
        bp.title = doc["title"].strip()[:90]
    if isinstance(doc.get("style"), str) and doc["style"].strip():
        bp.style = doc["style"].strip()
    if isinstance(doc.get("main_subject"), str) and doc["main_subject"].strip():
        bp.main_subject = doc["main_subject"].strip()
    if isinstance(doc.get("audio_direction"), str) and doc["audio_direction"].strip():
        bp.audio_direction = doc["audio_direction"].strip()
    if isinstance(doc.get("camera"), dict):
        bp.camera = CameraSpec.model_validate({**bp.camera.model_dump(), **doc["camera"]})
    if isinstance(doc.get("environment"), dict):
        bp.environment = EnvironmentSpec.model_validate(
            {**bp.environment.model_dump(), **doc["environment"]}
        )
    if isinstance(doc.get("lighting"), dict):
        bp.lighting = LightingSpec.model_validate({**bp.lighting.model_dump(), **doc["lighting"]})
    if isinstance(doc.get("subject_details"), dict):
        bp.subject_details = SubjectDetailsSpec.model_validate(
            {**bp.subject_details.model_dump(), **doc["subject_details"]}
        )
    if isinstance(doc.get("motion"), dict):
        bp.motion = MotionSpec.model_validate({**bp.motion.model_dump(), **doc["motion"]})

    raw_acts = doc.get("acts")
    if isinstance(raw_acts, list) and raw_acts:
        for idx, existing_act in enumerate(bp.acts):
            if existing_act.mode == "kinetic_ui_showcase":
                continue
            if idx >= len(raw_acts) or not isinstance(raw_acts[idx], dict):
                continue
            item = raw_acts[idx]
            if isinstance(item.get("title"), str) and item["title"].strip():
                existing_act.title = item["title"].strip()[:80]
            raw_tl = item.get("timeline")
            if isinstance(raw_tl, list) and raw_tl:
                parsed_beats: list[TimelineBeat] = []
                for b_idx, raw_b in enumerate(raw_tl[:5]):
                    if isinstance(raw_b, dict) and raw_b.get("scene") and raw_b.get("action"):
                        parsed_beats.append(
                            TimelineBeat(
                                time=str(raw_b.get("time") or f"{b_idx * 2}-{b_idx * 2 + 2}s"),
                                scene=str(raw_b["scene"]).strip(),
                                action=str(raw_b["action"]).strip(),
                                camera=str(raw_b.get("camera") or bp.camera.movement).strip(),
                                sound=str(raw_b.get("sound") or "").strip(),
                            )
                        )
                if parsed_beats:
                    existing_act.timeline = parsed_beats
            if isinstance(item.get("final_frame"), dict):
                existing_act.final_frame = FinalFrameSpec.model_validate(
                    {**existing_act.final_frame.model_dump(), **item["final_frame"]}
                )
            existing_act.duration_sec = act_dur


async def refine_director_blueprint(
    bp: DirectorBlueprint,
    *,
    instruction: str,
    act_number: int | None = None,
    beat_index: int | None = None,
    settings: Settings | None = None,
    mock: bool = False,
    cancel: CancellationToken | None = None,
) -> DirectorBlueprint:
    """Apply a plain-English tweak chip or custom note globally or to a specific Act/Beat."""
    clean_instr = instruction.strip()
    if not clean_instr:
        return bp
    resolved_settings = settings or get_settings()

    if not mock and resolved_settings.openrouter_api_key.strip():
        scope = "entire blueprint"
        if act_number is not None and beat_index is not None:
            scope = f"Act {act_number}, timeline beat index {beat_index} (preserving global invariants)"
        elif act_number is not None:
            scope = f"Act {act_number} (preserving global invariants and continuity with adjacent Acts)"
        sys_prompt = (
            "You are a Master Commercial Cinematographer refining an existing Structured JSON Video Blueprint. "
            "Apply the director's note cleanly with concrete, photorealistic visual physics. "
            "Return ONLY the updated JSON object with keys: title, style, main_subject, camera, environment, "
            "lighting, subject_details, motion, audio_direction, acts (preserving exact act_number and timeline length)."
        )
        usr_prompt = json.dumps(
            {
                "scope": scope,
                "director_note": clean_instr,
                "current_blueprint": bp.to_master_prompt_dict(),
            },
            indent=2,
        )
        try:
            llm_doc = await _call_openrouter_json(
                system_prompt=sys_prompt,
                user_prompt=usr_prompt,
                settings=resolved_settings,
                max_tokens=5000 if len(bp.acts) > 1 else 3000,
                temperature=0.55,
                cancel=cancel,
            )
            if llm_doc:
                # Handle single-act flat response if LLM returned top-level timeline
                if "acts" not in llm_doc and "timeline" in llm_doc and len(bp.acts) == 1:
                    llm_doc["acts"] = [
                        {
                            "act_number": 1,
                            "title": bp.acts[0].title,
                            "timeline": llm_doc["timeline"],
                            "final_frame": llm_doc.get("final_frame") or bp.acts[0].final_frame.model_dump(),
                        }
                    ]
                _merge_llm_doc_into_blueprint(
                    bp, llm_doc, act_dur=bp.acts[0].duration_sec if bp.acts else 10
                )
                bp.status_message = f"Refined: {clean_instr[:80]}"
                return bp
        except Exception:
            pass

    # Deterministic fallback refinement (instant, zero-cost, always works offline)
    low = clean_instr.lower()
    if act_number is not None and 1 <= act_number <= len(bp.acts):
        target_act = bp.acts[act_number - 1]
        if beat_index is not None and 0 <= beat_index < len(target_act.timeline):
            b = target_act.timeline[beat_index]
            b.action = f"{b.action.rstrip('.')}. Refined direction: {clean_instr}."
            if "camera" in low or "macro" in low or "pull" in low or "orbit" in low:
                b.camera = f"{b.camera.rstrip('.')}, incorporating {clean_instr}."
        else:
            if target_act.timeline:
                target_act.timeline[-1].action = (
                    f"{target_act.timeline[-1].action.rstrip('.')}. {clean_instr}."
                )
            target_act.final_frame.composition = (
                f"{target_act.final_frame.composition.rstrip('.')} ({clean_instr[:60]})"
            )
    else:
        if "macro" in low or "bokeh" in low:
            bp.camera.depth_of_field = "ultra-shallow macro"
            bp.camera.focus = "razor-sharp extreme macro focus with creamy realistic optical bokeh"
            bp.camera.lens_mm = 100
        if "volumetric" in low or "light" in low:
            bp.lighting.style = "dramatic cinematic volumetric lighting"
            if "sculpted golden volumetric god-rays" not in bp.lighting.effects:
                bp.lighting.effects.insert(0, "sculpted golden volumetric god-rays")
        if "asmr" in low or "slow" in low or "tactile" in low:
            bp.motion.style = "hypnotic ultra-smooth slow-motion tactile choreography"
            bp.motion.feel = "deeply satisfying ASMR physical precision"
        if ("pull-back" in low or "pullback" in low or "reveal" in low) and bp.acts and bp.acts[-1].timeline:
            last_b = bp.acts[-1].timeline[-1]
            last_b.camera = "One continuous sweeping pull-back reveal ending on a locked-off hero shot."
            bp.acts[-1].final_frame.composition = "locked-off grand hero pull-back composition with balanced negative space"
        if ("product" in low or "hero" in low) and bp.acts:
            hero_name = bp.brand_name or "the flagship hero product"
            bp.acts[-1].final_frame.subject = f"{hero_name} gleaming as the unmistakable final hero centerpiece"
        zero_text_rule = "strictly zero readable text, signs, or letters anywhere in frame"
        if "zero" in low and "text" in low and zero_text_rule not in bp.negative_constraints:
            bp.negative_constraints.insert(0, zero_text_rule)
        if "warm" in low or "sunset" in low or "gold" in low:
            bp.lighting.style = "warm golden-hour sunset volumetric lighting"
            bp.lighting.color_palette = ["#1B120C", "#F2A73B", "#FFE7BE", "#4A3A30"]
        if "audio" in low or "foley" in low:
            bp.audio_direction = (
                f"{bp.audio_direction.rstrip('.')}. Layered micro-foley on every 2-second beat: "
                "tactile material resonance, fluid texture, crisp glass/metal ticks, warm harmonic bloom."
            )
            for a in bp.acts:
                for b in a.timeline:
                    if not b.sound:
                        b.sound = "Crisp tactile micro-foley synchronized to the physical motion."
        # Always reflect custom non-preset global instructions cleanly
        if not any(
            k in low
            for k in ("macro", "bokeh", "volumetric", "light", "asmr", "slow", "pull-back", "pullback", "product", "text", "warm", "sunset", "audio", "foley")
        ):
            bp.style = f"{bp.style.rstrip(',')}, {clean_instr}"

    bp.status_message = f"Refined: {clean_instr[:80]}"
    return bp


def _render_mock_keyframe_jpeg(
    *,
    title: str,
    subtitle: str,
    footer: str,
    aspect: DirectorAspect,
    palette: list[str],
    out_path: Path,
    fonts_dir: Path | None = None,
) -> Path:
    """Render a clean labeled storyboard anchor JPEG offline ($0)."""
    w, h = (960, 540) if aspect == "16:9" else ((540, 960) if aspect == "9:16" else (720, 720))
    bg = _hex_to_rgb(palette[0] if palette else "#111318", (17, 19, 24))
    accent = _hex_to_rgb(palette[1] if len(palette) > 1 else "#E2A336", (226, 163, 54))
    fg = _hex_to_rgb(palette[2] if len(palette) > 2 else "#F5E9D6", (245, 233, 214))

    img = Image.new("RGB", (w, h), bg)
    draw = ImageDraw.Draw(img)
    # Soft studio radial vignette bands
    for step in range(8, 0, -1):
        frac = step / 8.0
        col = (
            int(bg[0] + (accent[0] - bg[0]) * (1 - frac) * 0.14),
            int(bg[1] + (accent[1] - bg[1]) * (1 - frac) * 0.14),
            int(bg[2] + (accent[2] - bg[2]) * (1 - frac) * 0.14),
        )
        rx = int(w * 0.45 * frac)
        ry = int(h * 0.45 * frac)
        draw.ellipse([w // 2 - rx, h // 2 - ry, w // 2 + rx, h // 2 + ry], fill=col)

    pad = 28
    draw.rectangle([pad, pad, w - pad, h - pad], outline=accent, width=2)

    bold_p = str(resolve_font_path("Roboto", bold=True, fonts_dir=fonts_dir))
    reg_p = str(resolve_font_path("Roboto", bold=False, fonts_dir=fonts_dir))
    f_title = ImageFont.truetype(bold_p, max(24, int(min(w, h) * 0.06)))
    f_sub = ImageFont.truetype(reg_p, max(16, int(min(w, h) * 0.036)))
    f_foot = ImageFont.truetype(reg_p, max(14, int(min(w, h) * 0.028)))

    draw.text((pad + 26, pad + 26), title[:48], fill=accent, font=f_title)
    # Word wrap subtitle cleanly
    words = subtitle.split()
    lines: list[str] = []
    cur: list[str] = []
    max_w = w - (pad + 26) * 2
    for word in words:
        trial = " ".join([*cur, word])
        if draw.textbbox((0, 0), trial, font=f_sub)[2] <= max_w or not cur:
            cur.append(word)
        else:
            lines.append(" ".join(cur))
            cur = [word]
    if cur:
        lines.append(" ".join(cur))

    y = pad + 92
    for line in lines[:6]:
        draw.text((pad + 26, y), line, fill=fg, font=f_sub)
        y += int(min(w, h) * 0.052)

    draw.text((pad + 26, h - pad - 44), footer[:80], fill=fg, font=f_foot)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="JPEG", quality=88)
    return out_path


async def _generate_flux_image(
    *,
    prompt: str,
    aspect: DirectorAspect,
    out_path: Path,
    settings: Settings,
    anchor_url: str | None = None,
    cancel: CancellationToken | None = None,
) -> tuple[Path, str | None, float]:
    """Generate a Flux Schnell or Flux Kontext continuity-locked keyframe via fal.run."""
    headers = {
        "Authorization": f"Key {settings.fal_key.strip()}",
        "Content-Type": "application/json",
    }
    async with ResilientHttpClient(
        connect_timeout=settings.http_connect_timeout_sec,
        read_timeout=75.0,
        write_timeout=30.0,
        pool_timeout=15.0,
        max_retries=1,
    ) as client:
        if anchor_url and anchor_url.startswith(("http://", "https://")):
            try:
                kontext_payload = {
                    "prompt": (
                        "Preserve the exact physical world, lighting temperature, lens bokeh, "
                        f"surface grain, and textures from the reference frame: {prompt}"
                    ),
                    "image_url": anchor_url,
                    "num_inference_steps": 24,
                    "guidance_scale": 3.5,
                    "num_images": 1,
                    "output_format": "jpeg",
                    "resolution_mode": "match_input",
                }
                data = await client.post_json(
                    f"https://fal.run/{KONTEXT_MODEL}",
                    json=kontext_payload,
                    headers=headers,
                    cancel=cancel,
                )
                images = data.get("images") if isinstance(data, dict) else None
                remote_url = images[0].get("url") if isinstance(images, list) and images else None
                if remote_url:
                    img_bytes = await client.get_bytes(remote_url, cancel=cancel)
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    out_path.write_bytes(img_bytes)
                    return out_path, str(remote_url), KONTEXT_UNIT_USD
            except Exception:
                pass

        model = (settings.image_model or "fal-ai/flux/schnell").lstrip("/")
        schnell_payload = {
            "prompt": prompt,
            "image_size": _ASPECT_FLUX_SIZE.get(aspect, "landscape_16_9"),
            "num_inference_steps": 4,
            "num_images": 1,
            "output_format": "jpeg",
        }
        data = await client.post_json(
            f"https://fal.run/{model}",
            json=schnell_payload,
            headers=headers,
            cancel=cancel,
        )
        images = data.get("images") if isinstance(data, dict) else None
        remote_url = images[0].get("url") if isinstance(images, list) and images else None
        if not remote_url:
            raise RuntimeError(f"Flux keyframe missing image URL: {data}")
        img_bytes = await client.get_bytes(remote_url, cancel=cancel)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(img_bytes)
        return out_path, str(remote_url), FLUX_UNIT_USD


async def generate_blueprint_keyframes(
    bp: DirectorBlueprint,
    media_dir: Path,
    *,
    settings: Settings | None = None,
    mock: bool = False,
    cancel: CancellationToken | None = None,
) -> DirectorBlueprint:
    """Generate continuity-locked Start & End anchor keyframes across all Acts."""
    resolved = settings or get_settings()
    use_mock = mock or bp.engine == "mock" or not resolved.fal_key.strip()
    media_dir.mkdir(parents=True, exist_ok=True)
    master_anchor_url: str | None = None
    total_usd = 0.0

    for idx, act in enumerate(bp.acts):
        if cancel is not None:
            cancel.raise_if_cancelled()

        # Start keyframe: Act 1 generates fresh; chained Act 2/3 inherits previous Act's end frame!
        if idx > 0 and act.continues_from_prev and bp.acts[idx - 1].end_keyframe_path:
            act.start_keyframe_path = bp.acts[idx - 1].end_keyframe_path
            act.start_keyframe_url = bp.acts[idx - 1].end_keyframe_url
        else:
            start_file = media_dir / f"act{act.act_number}_start.jpg"
            start_prompt = bp.compile_keyframe_prompt(idx, is_end=False)
            if use_mock or act.mode == "kinetic_ui_showcase":
                _render_mock_keyframe_jpeg(
                    title=f"{act.title} · Start ({act.time_range.split('-')[0]}s)",
                    subtitle=(act.timeline[0].scene if act.timeline else bp.main_subject),
                    footer=f"{bp.camera.style} · {bp.lighting.style}",
                    aspect=bp.aspect_ratio,
                    palette=bp.lighting.color_palette,
                    out_path=start_file,
                    fonts_dir=resolved.fonts_dir,
                )
                act.start_keyframe_path = str(start_file)
                act.start_keyframe_url = None
            else:
                path, url, usd = await _generate_flux_image(
                    prompt=start_prompt,
                    aspect=bp.aspect_ratio,
                    out_path=start_file,
                    settings=resolved,
                    anchor_url=master_anchor_url,
                    cancel=cancel,
                )
                act.start_keyframe_path = str(path)
                act.start_keyframe_url = url
                total_usd += usd
                if master_anchor_url is None and url:
                    master_anchor_url = url

        # End keyframe for every Act (anchored via Flux Kontext to master_anchor_url!)
        end_file = media_dir / f"act{act.act_number}_end.jpg"
        end_prompt = bp.compile_keyframe_prompt(idx, is_end=True)
        if use_mock or act.mode == "kinetic_ui_showcase":
            _render_mock_keyframe_jpeg(
                title=f"{act.title} · End Frame",
                subtitle=f"{act.final_frame.subject} — {act.final_frame.composition}",
                footer=f"Final Frame Anchor · {bp.aspect_ratio}",
                aspect=bp.aspect_ratio,
                palette=bp.lighting.color_palette,
                out_path=end_file,
                fonts_dir=resolved.fonts_dir,
            )
            act.end_keyframe_path = str(end_file)
            act.end_keyframe_url = None
        else:
            path, url, usd = await _generate_flux_image(
                prompt=end_prompt,
                aspect=bp.aspect_ratio,
                out_path=end_file,
                settings=resolved,
                anchor_url=master_anchor_url,
                cancel=cancel,
            )
            act.end_keyframe_path = str(path)
            act.end_keyframe_url = url
            total_usd += usd
            if master_anchor_url is None and url:
                master_anchor_url = url

    bp.actual_usd = round(bp.actual_usd + total_usd, 4)
    bp.status = "keyframes_ready"
    bp.status_message = (
        f"Generated continuity-locked anchor frames for {len(bp.acts)} Act(s)."
    )
    return bp


def encode_image_data_uri(img_path: Path, *, max_width: int = 1280, quality: int = 86) -> str:
    """Resize/re-encode a local JPEG/PNG into a compact data:image/jpeg;base64 URI for fal I2V."""
    img = Image.open(img_path).convert("RGB")
    if img.width > max_width:
        ratio = max_width / float(img.width)
        img = img.resize((max_width, max(2, int(round(img.height * ratio)))), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


async def extract_last_frame_jpeg(
    video_path: Path,
    out_path: Path,
    *,
    ffmpeg_bin: str = "ffmpeg",
    cancel: CancellationToken | None = None,
) -> Path:
    """Extract the exact final pixel frame from an Act MP4 for pixel-seamless Last-Frame Chaining."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_bin,
        "-y",
        "-threads",
        "1",
        "-sseof",
        "-0.08",
        "-i",
        str(video_path),
        "-vframes",
        "1",
        "-q:v",
        "3",
        str(out_path),
    ]
    rc, _, _ = await run_subprocess(cmd, timeout_sec=45.0, cancel=cancel, check=False)
    if rc != 0 or not out_path.is_file():
        # Fallback to frame 0 if clip is ultra-short
        cmd_fb = [
            ffmpeg_bin,
            "-y",
            "-threads",
            "1",
            "-i",
            str(video_path),
            "-vframes",
            "1",
            "-q:v",
            "3",
            str(out_path),
        ]
        rc2, _, stderr2 = await run_subprocess(cmd_fb, timeout_sec=45.0, cancel=cancel, check=False)
        if rc2 != 0 or not out_path.is_file():
            err_txt = stderr2.decode("utf-8", errors="replace")[-300:]
            raise RuntimeError(f"Failed to extract last frame from {video_path}: {err_txt}")
    return out_path


def _synthesize_ambient_cinematic_wav(duration_sec: float, out_path: Path) -> Path:
    """Synthesize a clean, warm low-level cinematic harmonic bed WAV ($0, zero network)."""
    sr = 24000
    n = max(sr, int(sr * duration_sec))
    frames = bytearray()
    freqs = (130.81, 196.00, 261.63, 329.63)  # C3 warm ambient fifth + octave pad
    for i in range(n):
        t = i / sr
        fade_in = min(1.0, t / 0.8)
        fade_out = min(1.0, max(0.0, (duration_sec - t) / 1.0))
        env = fade_in * fade_out * (0.85 + 0.15 * math.sin(2.0 * math.pi * 0.22 * t))
        sample = sum(math.sin(2.0 * math.pi * f * t) for f in freqs) / len(freqs)
        val = int(max(-32767, min(32767, sample * env * 4800.0)))
        frames.extend(struct.pack("<h", val))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(bytes(frames))
    return out_path


async def _normalize_act_clip(
    src_path: Path,
    dst_path: Path,
    *,
    width: int,
    height: int,
    fps: int,
    has_audio: bool,
    ffmpeg_bin: str = "ffmpeg",
    cancel: CancellationToken | None = None,
) -> Path:
    """Normalize an Act clip to uniform H.264/bt709/48kHz stereo using Render-512MB-safe flags."""
    raw_t = os.getenv("GTM_FFMPEG_THREADS", "2").strip()
    eff_threads = max(1, int(raw_t)) if raw_t.isdigit() else 2
    vf = (
        f"scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},setsar=1,fps={fps},format=yuv420p"
    )
    cmd: list[str] = [
        ffmpeg_bin,
        "-y",
        "-threads",
        str(eff_threads),
        "-i",
        str(src_path),
    ]
    if not has_audio:
        cmd.extend(["-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000"])

    cmd.extend(["-vf", vf])
    cmd.extend(_pinned_h264_flags(fps, threads=eff_threads))
    if has_audio:
        cmd.extend(["-map", "0:v:0", "-map", "0:a:0", "-c:a", "aac", "-ar", "48000", "-ac", "2", "-b:a", "160k"])
    else:
        cmd.extend(["-map", "0:v:0", "-map", "1:a:0", "-shortest", "-c:a", "aac", "-ar", "48000", "-ac", "2", "-b:a", "160k"])
    cmd.extend(["-movflags", "+faststart", str(dst_path)])

    rc, _, stderr_b = await run_subprocess(cmd, timeout_sec=240.0, cancel=cancel, check=False)
    if rc != 0 or not dst_path.is_file():
        err_txt = stderr_b.decode("utf-8", errors="replace")[-500:]
        raise RuntimeError(f"FFmpeg normalize failed for {src_path}: {err_txt}")
    return dst_path


async def _stitch_act_clips(
    *,
    clip_paths: list[Path],
    output_path: Path,
    aspect: DirectorAspect,
    fps: int,
    preserve_native_audio: bool,
    ffmpeg_bin: str = "ffmpeg",
    ffprobe_bin: str = "ffprobe",
    cancel: CancellationToken | None = None,
) -> Path:
    """Normalize, concatenate, and audio-master all Act clips with Render-safe memory bounds."""
    width, height = aspect_to_dimensions(aspect)
    raw_t = os.getenv("GTM_FFMPEG_THREADS", "2").strip()
    eff_threads = max(1, int(raw_t)) if raw_t.isdigit() else 2

    tmp_dir = Path(tempfile.mkdtemp(prefix="director_stitch_"))
    try:
        norm_paths: list[Path] = []
        any_native_audio = False
        total_dur = 0.0
        for idx, clip_p in enumerate(clip_paths):
            probe = await probe_media(clip_p, ffprobe_bin=ffprobe_bin, cancel=cancel)
            has_a = bool(probe.has_audio and preserve_native_audio)
            if has_a:
                any_native_audio = True
            total_dur += float(probe.duration_sec or 8.0)
            norm_p = tmp_dir / f"norm_{idx:02d}.mp4"
            await _normalize_act_clip(
                clip_p,
                norm_p,
                width=width,
                height=height,
                fps=fps,
                has_audio=has_a,
                ffmpeg_bin=ffmpeg_bin,
                cancel=cancel,
            )
            norm_paths.append(norm_p)

        concat_mp4 = tmp_dir / "concat.mp4"
        if len(norm_paths) == 1:
            shutil.copyfile(norm_paths[0], concat_mp4)
        else:
            list_txt = tmp_dir / "list.txt"
            lines = [
                f"file '{str(p.resolve()).replace(chr(39), chr(39) + chr(92) + chr(39) + chr(39))}'"
                for p in norm_paths
            ]
            list_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
            concat_cmd = [
                ffmpeg_bin,
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
                str(concat_mp4),
            ]
            rc, _, stderr_b = await run_subprocess(
                concat_cmd, timeout_sec=180.0, cancel=cancel, check=False
            )
            if rc != 0 or not concat_mp4.is_file():
                err_txt = stderr_b.decode("utf-8", errors="replace")[-500:]
                raise RuntimeError(f"FFmpeg concat failed: {err_txt}")

        final_tmp = tmp_dir / "final_master.mp4"
        if any_native_audio:
            # Keep Veo 3 native audio cleanly with gentle loudnorm mastering
            mix_cmd = [
                ffmpeg_bin,
                "-y",
                "-threads",
                str(eff_threads),
                "-i",
                str(concat_mp4),
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
        else:
            # Add warm subtle ambient sound bed so silent Kling/mock clips sound rich and finished
            bed_wav = _synthesize_ambient_cinematic_wav(max(3.0, total_dur), tmp_dir / "bed.wav")
            mix_cmd = [
                ffmpeg_bin,
                "-y",
                "-threads",
                str(eff_threads),
                "-i",
                str(concat_mp4),
                "-i",
                str(bed_wav),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "copy",
                "-af",
                "loudnorm=I=-16:TP=-1.5:LRA=11",
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

        mix_rc, _, _ = await run_subprocess(mix_cmd, timeout_sec=180.0, cancel=cancel, check=False)
        if mix_rc != 0 or not final_tmp.is_file():
            # Fallback to concat_mp4 if audio filter fails
            shutil.copyfile(concat_mp4, final_tmp)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        _move_output_atomically(final_tmp, output_path)
        return output_path
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


async def render_director_blueprint(
    bp: DirectorBlueprint,
    media_dir: Path,
    *,
    settings: Settings | None = None,
    media_store: MediaStore | None = None,
    mock: bool = False,
    on_progress: Any | None = None,
    cancel: CancellationToken | None = None,
) -> DirectorBlueprint:
    """Render a 10s, 20s, or 30s Director Blueprint end-to-end with seamless Act continuity."""
    resolved = settings or get_settings()
    use_mock = mock or bp.engine == "mock" or not resolved.fal_key.strip()
    media_dir.mkdir(parents=True, exist_ok=True)
    fps = 24
    width, height = aspect_to_dimensions(bp.aspect_ratio)
    bible = _synthetic_bible_from_blueprint(bp)
    palette = bp.lighting.color_palette or ["#111318", "#E2A336", "#F5E9D6"]
    bg_rgb = _hex_to_rgb(palette[0], (17, 19, 24))
    accent_rgb = _hex_to_rgb(palette[1] if len(palette) > 1 else "#E2A336", (226, 163, 54))
    text_rgb = _hex_to_rgb(palette[2] if len(palette) > 2 else "#F5E9D6", (245, 233, 214))

    async def _report(msg: str) -> None:
        bp.status_message = msg
        if on_progress is not None:
            res = on_progress(bp)
            if asyncio.iscoroutine(res):
                await res

    bp.status = "rendering"
    bp.error = None
    await _report(f"Starting {bp.duration_sec}s render ({len(bp.acts)} Act(s))...")

    cost_meter = CostMeter(
        run_id=bp.blueprint_id,
        cap_usd=max(10.0, estimate_blueprint_cost(bp) + 5.0),
    )
    provider = VideoProvider(
        settings=resolved,
        media_store=media_store or MediaStore(media_dir / "store"),
        cost_meter=cost_meter,
        usd_per_sec=0.11 if "veo3" not in bp.engine else 0.15,
    )

    low_eng = bp.engine.lower()
    supports_native_audio = "veo3" in low_eng or "kling-video/v3" in low_eng or "kling-video/o3" in low_eng
    use_native_audio = bool(supports_native_audio and bp.native_audio and not use_mock)

    async def _render_single_act(
        idx: int,
        act: DirectorAct,
        chained_start_uri: str | None = None,
    ) -> Path:
        out_clip = media_dir / f"act{act.act_number}_clip.mp4"
        if act.mode == "kinetic_ui_showcase":
            await _report(f"Rendering {act.title} (Kinetic Studio UI)...")
            prod_img = Path(bp.product_image_path) if bp.product_image_path and Path(bp.product_image_path).is_file() else None
            await render_kinetic_card_mp4(
                text=act.on_screen_headline or f"{bp.brand_name or bp.title}",
                subtitle=act.on_screen_subline or bp.brand_tagline or bp.style[:70],
                footer=bp.brand_url or "",
                width=width,
                height=height,
                fps=fps,
                duration_sec=float(act.duration_sec),
                bible=bible,
                output_path=out_clip,
                bg_rgb=bg_rgb,
                text_rgb=text_rgb,
                accent_rgb=accent_rgb,
                ffmpeg_bin=resolved.ffmpeg_bin,
                ffprobe_bin=resolved.ffprobe_bin,
                fonts_dir=resolved.fonts_dir,
                card_kind="title",
                shot_id=f"act{act.act_number}",
                media_image_path=prod_img,
                cancel=cancel,
            )
            act.rendered_clip_path = str(out_clip)
            act.rendered_duration_sec = float(act.duration_sec)
            return out_clip

        if use_mock:
            await _report(f"Rendering {act.title} ($0 Studio Animatic)...")
            beat_summary = " → ".join(b.time for b in act.timeline) or act.time_range
            prod_img = (
                Path(bp.product_image_path)
                if idx == len(bp.acts) - 1 and bp.product_image_path and Path(bp.product_image_path).is_file()
                else None
            )
            await render_kinetic_card_mp4(
                text=act.title,
                subtitle=act.final_frame.subject[:90],
                footer=f"{bp.style[:44]} · {beat_summary}",
                width=width,
                height=height,
                fps=fps,
                duration_sec=float(min(4.0, act.duration_sec)) if mock else float(act.duration_sec),
                bible=bible,
                output_path=out_clip,
                bg_rgb=bg_rgb,
                text_rgb=text_rgb,
                accent_rgb=accent_rgb,
                ffmpeg_bin=resolved.ffmpeg_bin,
                ffprobe_bin=resolved.ffprobe_bin,
                fonts_dir=resolved.fonts_dir,
                card_kind="title",
                shot_id=f"act{act.act_number}",
                media_image_path=prod_img,
                cancel=cancel,
            )
            act.rendered_clip_path = str(out_clip)
            act.rendered_duration_sec = float(min(4.0, act.duration_sec)) if mock else float(act.duration_sec)
            return out_clip

        # Live flagship video generation (Kling v4 Pro 10s / Kling Flash / Veo 3 8s Native Audio)
        await _report(f"Generating {act.title} via {bp.engine}...")
        raw_json_prompt = bp.compile_act_model_prompt(idx)

        start_ref = chained_start_uri or act.start_keyframe_url
        if not start_ref and act.start_keyframe_path and Path(act.start_keyframe_path).is_file():
            start_ref = encode_image_data_uri(Path(act.start_keyframe_path))

        end_ref = act.end_keyframe_url
        if not end_ref and act.end_keyframe_path and Path(act.end_keyframe_path).is_file():
            end_ref = encode_image_data_uri(Path(act.end_keyframe_path))
        # If user uploaded a real product photo and this is the final Act, offer it as End Frame lock if no end keyframe exists
        if (
            not end_ref
            and idx == len(bp.acts) - 1
            and bp.product_image_path
            and Path(bp.product_image_path).is_file()
        ):
            end_ref = encode_image_data_uri(Path(bp.product_image_path))

        shot = Shot(
            id=f"act{act.act_number}",
            index=idx,
            beat=act.title,
            duration_sec=float(act.duration_sec),
            engine="image_to_video" if start_ref else "ai_video",
            visual_prompt=act.timeline[0].scene if act.timeline else bp.main_subject,
            negative_prompt=", ".join(bp.negative_constraints[:4]),
            camera_move=bp.camera.movement,
            camera_angle=bp.camera.style,
            reference_image=start_ref,
            end_reference_image=end_ref if bp.continuity_mode == "parallel_bridge" or idx == len(bp.acts) - 1 else None,
        )
        rendered = await provider.generate_clip(
            shot=shot,
            aspect=bp.aspect_ratio,
            fps=fps,
            cancel=cancel,
            provider="fal",
            model=bp.engine,
            raw_prompt=raw_json_prompt,
            generate_audio=use_native_audio,
        )
        shutil.copyfile(Path(rendered.path), out_clip)
        act.rendered_clip_path = str(out_clip)
        act.rendered_duration_sec = float(rendered.duration_sec)
        bp.actual_usd = round(bp.actual_usd + float(rendered.usd or 0.0), 4)
        return out_clip

    rendered_paths: list[Path] = []
    if bp.continuity_mode == "parallel_bridge" and len(bp.acts) > 1 and not use_mock:
        # Parallel Start->End Bridged Acts (3x faster when keyframes are pre-approved!)
        await _report(f"Rendering {len(bp.acts)} keyframe-bridged Acts in parallel...")
        sem = asyncio.Semaphore(max(1, resolved.max_concurrent_jobs))

        async def _bounded(i: int, a: DirectorAct) -> Path:
            async with sem:
                return await _render_single_act(i, a, None)

        rendered_paths = list(
            await asyncio.gather(*[_bounded(i, a) for i, a in enumerate(bp.acts)])
        )
    else:
        # Sequential Last-Frame Chaining (pixel-exact Act 1 -> Act 2 -> Act 3 continuity!)
        prev_last_frame_uri: str | None = None
        for idx, act in enumerate(bp.acts):
            start_uri = prev_last_frame_uri if (idx > 0 and act.continues_from_prev) else None
            clip_path = await _render_single_act(idx, act, chained_start_uri=start_uri)
            rendered_paths.append(clip_path)
            # Extract exact final pixel frame to feed seamlessly into the next Act!
            if idx + 1 < len(bp.acts) and bp.acts[idx + 1].continues_from_prev and not use_mock:
                await _report(f"Extracting exact last frame of Act {act.act_number} for Act {act.act_number + 1} handoff...")
                chain_jpg = media_dir / f"act{act.act_number}_chain_last.jpg"
                await extract_last_frame_jpeg(
                    clip_path,
                    chain_jpg,
                    ffmpeg_bin=resolved.ffmpeg_bin,
                    cancel=cancel,
                )
                prev_last_frame_uri = encode_image_data_uri(chain_jpg)
            else:
                prev_last_frame_uri = None

    # Optional +2.5s Kinetic Brand Logo End-Card
    if bp.add_brand_endcard and (bp.brand_name or bp.brand_logo_path):
        await _report("Rendering Kinetic Brand Logo End-Card...")
        endcard_mp4 = media_dir / "brand_endcard.mp4"
        logo_p = Path(bp.brand_logo_path) if bp.brand_logo_path and Path(bp.brand_logo_path).is_file() else None
        await render_kinetic_card_mp4(
            text=bp.brand_name or bp.title,
            subtitle=bp.brand_tagline or "",
            footer=bp.brand_url or "",
            width=width,
            height=height,
            fps=fps,
            duration_sec=2.5,
            bible=bible,
            output_path=endcard_mp4,
            bg_rgb=bg_rgb,
            text_rgb=text_rgb,
            accent_rgb=accent_rgb,
            ffmpeg_bin=resolved.ffmpeg_bin,
            ffprobe_bin=resolved.ffprobe_bin,
            fonts_dir=resolved.fonts_dir,
            card_kind="end_card",
            shot_id="endcard",
            logo_path=logo_p,
            cancel=cancel,
        )
        rendered_paths.append(endcard_mp4)

    await _report("Mastering unified final cut & audio mix...")
    final_mp4 = media_dir / "final_cut.mp4"
    await _stitch_act_clips(
        clip_paths=rendered_paths,
        output_path=final_mp4,
        aspect=bp.aspect_ratio,
        fps=fps,
        preserve_native_audio=use_native_audio,
        ffmpeg_bin=resolved.ffmpeg_bin,
        ffprobe_bin=resolved.ffprobe_bin,
        cancel=cancel,
    )

    bp.final_video_path = str(final_mp4)
    bp.status = "done"
    bp.status_message = f"Final {bp.duration_sec}s cut mastered and ready."
    return bp


__all__ = [
    "DirectorBlueprintStore",
    "encode_image_data_uri",
    "estimate_blueprint_cost",
    "extract_last_frame_jpeg",
    "generate_blueprint_keyframes",
    "generate_director_blueprint",
    "refine_director_blueprint",
    "render_director_blueprint",
]
