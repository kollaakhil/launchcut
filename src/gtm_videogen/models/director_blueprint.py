"""Structured JSON Director Blueprint models & zero-jargon visual preset catalog.

Supports 10s (1 Act), 20s (2 Chained Acts), and 30s (3 Chained Acts) films compiled
into hyper-detailed, model-native structured JSON video prompts with locked visual
invariants (style, camera, environment, lighting, subject_details, motion) and
2-second choreographed timelines.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

DirectorDuration = Literal[10, 20, 30]
DirectorAspect = Literal["16:9", "9:16", "1:1"]
DirectorFormatMode = Literal["continuous", "hybrid_launch"]
DirectorContinuityMode = Literal["last_frame_chain", "parallel_bridge"]
DirectorActMode = Literal["cinematic_sequence", "kinetic_ui_showcase"]


class TimelineBeat(BaseModel):
    """One 2-second choreographed micro-beat inside a Director Act."""

    model_config = ConfigDict(extra="ignore")

    time: str = Field(..., min_length=1, description="Beat window e.g. '0-2s', '2-4s'")
    scene: str = Field(..., min_length=1, description="Exact visual framing & state at beat start")
    action: str = Field(..., min_length=1, description="Continuous physical action & transformation")
    camera: str = Field(..., min_length=1, description="Camera trajectory & depth-of-field change")
    sound: str = Field(
        default="",
        description="Optional synchronized ASMR/foley/acoustic cue for this 2s window",
    )


class CameraSpec(BaseModel):
    """Global camera & optical invariants shared across all Acts."""

    model_config = ConfigDict(extra="ignore")

    style: str = "macro tilt-shift photography"
    depth_of_field: str = "very shallow"
    focus: str = "precise miniature-scale focus with realistic bokeh"
    movement: str = "smooth cinematic camera motion"
    lens_mm: int = 85


class EnvironmentSpec(BaseModel):
    """Global physical world, surface, and material invariants."""

    model_config = ConfigDict(extra="ignore")

    surface: str = "real wooden workshop table"
    world: str = "handcrafted physical miniature model"
    materials: str = "believable physical textures, realistic imperfections, authentic scale"
    atmosphere: str = "subtle volumetric dust motes and warm studio depth"


class LightingSpec(BaseModel):
    """Global lighting setup and atmospheric light effects."""

    model_config = ConfigDict(extra="ignore")

    style: str = "cinematic volumetric lighting"
    effects: list[str] = Field(
        default_factory=lambda: [
            "warm golden glow",
            "soft volumetric light rays",
            "realistic reflections",
            "natural shadows",
            "subtle atmospheric haze",
        ]
    )
    color_palette: list[str] = Field(
        default_factory=lambda: ["#111318", "#E2A336", "#F5E9D6", "#2B3A4A"]
    )


class SubjectDetailsSpec(BaseModel):
    """Tactile micro-detail & physical scale invariants for the hero subject/hand/product."""

    model_config = ConfigDict(extra="ignore")

    appearance: str = "photorealistic hero subject with tactile studio precision"
    textures: str = "visible micro-grain, realistic reflections, authentic surface imperfections"
    interaction: str = "physically believable contact, weight, and light bounce with the environment"
    scale: str = "clear foreground-to-background spatial hierarchy"


class MotionSpec(BaseModel):
    """Motion curve, physics fidelity, and tactile feel."""

    model_config = ConfigDict(extra="ignore")

    style: str = "smooth, satisfying, seamless transitions"
    physics: str = "realistic fluid, material, and inertia physics"
    feel: str = "ASMR-like satisfying motion"


class FinalFrameSpec(BaseModel):
    """Exact hero composition held at the end of an Act (or entire film)."""

    model_config = ConfigDict(extra="ignore")

    subject: str = "glowing hero subject centered in frame"
    composition: str = "locked-off hero macro shot with balanced negative space"
    hold: str = "brief cinematic hold"


class DirectorAct(BaseModel):
    """One 8–10s Act inside a 10s, 20s, or 30s Structured JSON Director Blueprint."""

    model_config = ConfigDict(extra="ignore")

    act_number: int = Field(..., ge=1, le=3)
    title: str = "Act 1 · Hook & Spark"
    time_range: str = "0-10s"
    duration_sec: int = Field(default=10, ge=4, le=12)
    mode: DirectorActMode = "cinematic_sequence"
    continues_from_prev: bool = False
    timeline: list[TimelineBeat] = Field(default_factory=list)
    final_frame: FinalFrameSpec = Field(default_factory=FinalFrameSpec)
    on_screen_headline: str | None = None
    on_screen_subline: str | None = None
    start_keyframe_prompt: str | None = None
    end_keyframe_prompt: str | None = None
    start_keyframe_path: str | None = None
    start_keyframe_url: str | None = None
    end_keyframe_path: str | None = None
    end_keyframe_url: str | None = None
    rendered_clip_path: str | None = None
    rendered_duration_sec: float | None = None


class DirectorBlueprint(BaseModel):
    """Complete Guided Structured JSON Director Blueprint supporting 10s, 20s, and 30s films."""

    model_config = ConfigDict(extra="ignore")

    blueprint_id: str
    title: str = "Untitled Hero Sequence"
    type: str = "video_prompt"
    duration: str = "10 seconds"
    duration_sec: DirectorDuration = 10
    format_mode: DirectorFormatMode = "continuous"
    continuity_mode: DirectorContinuityMode = "last_frame_chain"
    aspect_ratio: DirectorAspect = "16:9"
    resolution: str = "4K"
    engine: str = "kling/v4/text-to-video"
    native_audio: bool = True
    add_brand_endcard: bool = False
    brand_name: str | None = None
    brand_tagline: str | None = None
    brand_url: str | None = None
    brand_logo_path: str | None = None
    product_image_path: str | None = None

    # Guided visual selections (saved so users can see & re-spin their choices)
    user_idea: str = ""
    recipe_id: str | None = None
    world_preset: str = "miniature_diorama"
    camera_preset: str = "macro_tilt_shift"
    lighting_preset: str = "volumetric_gold"
    motion_preset: str = "asmr_tactile"

    # Global visual invariants (identical across all Acts)
    style: str = "ultra-photorealistic cinematic miniature diorama"
    main_subject: str = "a tiny living miniature world manipulated by a giant realistic human hand"
    camera: CameraSpec = Field(default_factory=CameraSpec)
    environment: EnvironmentSpec = Field(default_factory=EnvironmentSpec)
    lighting: LightingSpec = Field(default_factory=LightingSpec)
    subject_details: SubjectDetailsSpec = Field(default_factory=SubjectDetailsSpec)
    motion: MotionSpec = Field(default_factory=MotionSpec)
    audio_direction: str = (
        "Tactile ASMR foley: crisp glass clinks, subtle liquid shimmer, gentle ambient wind, "
        "warm resonant sub-bass swell on hero reveal."
    )
    negative_constraints: list[str] = Field(
        default_factory=lambda: [
            "no garbled text or watermarks",
            "no sudden jump cuts or flicker",
            "no plastic CGI sheen or unnatural morphing",
            "no inconsistent lighting or warped hands",
        ]
    )

    acts: list[DirectorAct] = Field(default_factory=list)

    # Render state & telemetry
    status: str = "draft"  # draft | keyframes_ready | rendering | done | failed
    status_message: str = "Blueprint ready for review."
    error: str | None = None
    final_video_path: str | None = None
    estimated_usd: float = 0.0
    actual_usd: float = 0.0
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_act_prompt_dict(self, act_index: int = 0) -> dict[str, Any]:
        """Export a single Act as the crisp, model-native structured JSON prompt dict."""
        if not self.acts:
            raise ValueError("Blueprint has no acts")
        idx = max(0, min(act_index, len(self.acts) - 1))
        act = self.acts[idx]
        doc: dict[str, Any] = {
            "type": self.type,
            "duration": f"{act.duration_sec} seconds",
            "aspect_ratio": self.aspect_ratio,
            "resolution": self.resolution,
            "style": self.style,
            "camera": self.camera.model_dump(),
            "environment": self.environment.model_dump(),
            "main_subject": self.main_subject,
            "timeline": [
                {
                    "time": b.time,
                    "scene": b.scene,
                    "action": b.action,
                    "camera": b.camera,
                    **({"sound": b.sound} if b.sound else {}),
                }
                for b in act.timeline
            ],
            "lighting": self.lighting.model_dump(),
            "subject_details": self.subject_details.model_dump(),
            "motion": self.motion.model_dump(),
            "final_frame": act.final_frame.model_dump(),
        }
        if self.native_audio and self.audio_direction:
            doc["audio"] = self.audio_direction
        if act.continues_from_prev and idx > 0:
            prev_final = self.acts[idx - 1].final_frame
            doc["continuity_from_previous_act"] = (
                f"Seamlessly continues from the exact final frame of Act {idx}: "
                f"{prev_final.subject} ({prev_final.composition}). Match exact lighting, "
                "surface textures, scale, and camera momentum with zero jump cut."
            )
        if self.negative_constraints:
            doc["avoid"] = self.negative_constraints
        return doc

    def to_master_prompt_dict(self) -> dict[str, Any]:
        """Export the full Blueprint JSON (single-act flat format for 10s, multi-act for 20s/30s)."""
        if len(self.acts) == 1:
            return self.to_act_prompt_dict(0)
        return {
            "type": self.type,
            "title": self.title,
            "duration": f"{self.duration_sec} seconds",
            "format_mode": self.format_mode,
            "continuity_mode": self.continuity_mode,
            "aspect_ratio": self.aspect_ratio,
            "resolution": self.resolution,
            "style": self.style,
            "camera": self.camera.model_dump(),
            "environment": self.environment.model_dump(),
            "main_subject": self.main_subject,
            "lighting": self.lighting.model_dump(),
            "subject_details": self.subject_details.model_dump(),
            "motion": self.motion.model_dump(),
            "audio": self.audio_direction,
            "acts": [
                {
                    "act": act.act_number,
                    "title": act.title,
                    "time_range": act.time_range,
                    "duration": f"{act.duration_sec} seconds",
                    "mode": act.mode,
                    "continues_from_prev": act.continues_from_prev,
                    "timeline": [
                        {
                            "time": b.time,
                            "scene": b.scene,
                            "action": b.action,
                            "camera": b.camera,
                            **({"sound": b.sound} if b.sound else {}),
                        }
                        for b in act.timeline
                    ],
                    "final_frame": act.final_frame.model_dump(),
                }
                for act in self.acts
            ],
            "avoid": self.negative_constraints,
        }

    def compile_act_model_prompt(self, act_index: int = 0) -> str:
        """Compile an Act into a clean, compact JSON string ready for Kling v4 / Veo 3."""
        return json.dumps(self.to_act_prompt_dict(act_index), ensure_ascii=False, indent=2)

    def compile_keyframe_prompt(self, act_index: int = 0, *, is_end: bool = False) -> str:
        """Compile a static photographic Flux Schnell / Kontext prompt for Act start or end frame."""
        act = self.acts[max(0, min(act_index, len(self.acts) - 1))]
        if is_end:
            if act.end_keyframe_prompt:
                base = act.end_keyframe_prompt
            else:
                base = (
                    f"{act.final_frame.subject}. {act.final_frame.composition}."
                )
        else:
            if act.start_keyframe_prompt:
                base = act.start_keyframe_prompt
            elif act.timeline:
                base = act.timeline[0].scene
            else:
                base = self.main_subject

        effects = ", ".join(self.lighting.effects[:4])
        parts = [
            f"{self.style}, {base.rstrip('.')}",
            f"World & materials: {self.environment.world} on {self.environment.surface}; "
            f"{self.environment.materials}",
            f"Subject details: {self.subject_details.appearance}, {self.subject_details.textures}",
            f"Camera: {self.camera.style}, {self.camera.lens_mm}mm lens, "
            f"{self.camera.depth_of_field} depth of field, {self.camera.focus}",
            f"Lighting: {self.lighting.style} with {effects}",
            "Ultra-photorealistic 4K cinema still frame, zero garbled text, no watermarks, no artifacts",
        ]
        return ". ".join(p.strip().rstrip(".") for p in parts if p.strip()) + "."


# ---------------------------------------------------------------------------
# Guided Zero-Jargon Visual Presets & Concept Recipes
# ---------------------------------------------------------------------------

WORLD_PRESETS: dict[str, dict[str, Any]] = {
    "miniature_diorama": {
        "id": "miniature_diorama",
        "label": "Miniature Diorama",
        "tagline": "Handcrafted tiny living world & tilt-shift magic",
        "style": "ultra-photorealistic cinematic miniature diorama",
        "environment": {
            "surface": "real aged wooden workshop table with fine grain",
            "world": "handcrafted physical miniature diorama model",
            "materials": "believable miniature textures, painted wood, brass, matte clay, realistic imperfections, authentic scale",
            "atmosphere": "soft workshop dust motes caught in warm directional sunbeams",
        },
        "subject_details": {
            "appearance": "photorealistic human hand interacting with a hyper-detailed miniature world",
            "textures": "visible skin pores, fine wrinkles, tiny handcrafted architectural imperfections",
            "interaction": "physically believable contact, gentle weight, and micro-scale ripples",
            "scale": "clearly gigantic human scale juxtaposed against intricate miniature world",
        },
    },
    "macro_studio": {
        "id": "macro_studio",
        "label": "Macro Luxury Studio",
        "tagline": "Apple/Rimowa-grade product craft & tactile materials",
        "style": "ultra-photorealistic luxury studio product commercial",
        "environment": {
            "surface": "honed matte obsidian stone plinth with subtle specular sheen",
            "world": "architectural dark-studio tabletop set with precision rim reflections",
            "materials": "CNC-machined anodized aluminum, sapphire glass, brushed titanium, tactile micro-knurling",
            "atmosphere": "clean optical clarity with delicate mist and controlled specular bloom",
        },
        "subject_details": {
            "appearance": "precision-engineered flagship product photographed at studio hero quality",
            "textures": "machined bevels, microscopic brushed grain, crisp glass refraction",
            "interaction": "clean physical reflections and razor-sharp contact shadows on stone",
            "scale": "hero macro dominance filling the frame with authoritative geometry",
        },
    },
    "architectural_luxury": {
        "id": "architectural_luxury",
        "label": "Architectural Editorial",
        "tagline": "Sunlit travertine, oak, linen & calm prestige cinema",
        "style": "ultra-photorealistic architectural editorial cinema",
        "environment": {
            "surface": "warm Roman travertine slab beside natural pale white oak",
            "world": "minimalist daylight architectural loft with sculpted shadows",
            "materials": "raw limestone, brushed brass, textured ceramic, optical glass, soft shadow gradients",
            "atmosphere": "quiet morning light streaming through tall louvered windows",
        },
        "subject_details": {
            "appearance": "sculptural industrial design hero object anchored in sunlit architecture",
            "textures": "natural stone pores, subtle ceramic glaze, soft sunlit dust",
            "interaction": "crisp caustics and soft natural ambient bounce across travertine",
            "scale": "refined architectural proportion with generous editorial negative space",
        },
    },
    "surreal_glass": {
        "id": "surreal_glass",
        "label": "Surreal Liquid & Glass",
        "tagline": "Molten gold, prism refraction & sculptural metamorphosis",
        "style": "ultra-photorealistic surreal optical glass & liquid sculpture cinema",
        "environment": {
            "surface": "seamless dark reflective water plane and frosted quartz horizon",
            "world": "elemental optical stage where liquid light and blown glass converge",
            "materials": "molten liquid gold, crystal-clear borosilicate glass, iridescent caustics, suspended droplets",
            "atmosphere": "luminous dispersion rays and fine glowing micro-particles",
        },
        "subject_details": {
            "appearance": "living sculptural liquid-metal and prism glass transformation",
            "textures": "high-viscosity fluid ripples, chromatic glass dispersion, mirror highlights",
            "interaction": "realistic surface tension, buoyant droplets, and glowing caustic refractions",
            "scale": "monumental macro sculpture suspended in cinematic space",
        },
    },
    "cinematic_live": {
        "id": "cinematic_live",
        "label": "Cinematic Live-Action",
        "tagline": "35mm anamorphic human storytelling & natural realism",
        "style": "ultra-photorealistic 35mm anamorphic live-action cinema",
        "environment": {
            "surface": "real tactile urban and natural locations with lived-in depth",
            "world": "grounded photorealistic cinema world shot on location",
            "materials": "authentic fabrics, brushed metal, rain-slick asphalt, warm interior wood",
            "atmosphere": "natural atmospheric depth, subtle 35mm film grain, organic bokeh",
        },
        "subject_details": {
            "appearance": "authentic human presence and tactile hero object in real motion",
            "textures": "natural skin tone, woven fabric grain, genuine specular highlights",
            "interaction": "grounded real-world physics, natural hand movement, believable weight",
            "scale": "cinematic human-eye perspective transitioning into intimate detail",
        },
    },
    "paper_origami": {
        "id": "paper_origami",
        "label": "Tactile Paper & Craft",
        "tagline": "Heavy cotton paper folding seamlessly into 3D worlds",
        "style": "ultra-photorealistic tactile architectural papercraft & kinetic sculpture",
        "environment": {
            "surface": "architect drafting table in warm walnut with brass ruler edge",
            "world": "living pop-up architectural world crafted from layered cotton paper",
            "materials": "heavy 300gsm textured cotton paper, crisp deckled edges, gold foil inlays, fine shadow folds",
            "atmosphere": "warm studio tabletop glow with delicate backlit paper translucency",
        },
        "subject_details": {
            "appearance": "intricate origami folds unfolding autonomously into living 3D forms",
            "textures": "visible cotton paper fibers, crisp crease lines, warm translucent backlighting",
            "interaction": "precise mechanical folding physics, crisp paper tension, soft tabletop shadows",
            "scale": "tabletop kinetic diorama expanding seamlessly into full depth",
        },
    },
}

CAMERA_PRESETS: dict[str, dict[str, Any]] = {
    "macro_tilt_shift": {
        "id": "macro_tilt_shift",
        "label": "Macro Tilt-Shift",
        "tagline": "Shallow bokeh, miniature scale focus & smooth glide",
        "camera": {
            "style": "macro tilt-shift photography",
            "depth_of_field": "very shallow",
            "focus": "precise miniature-scale focus with creamy realistic bokeh",
            "movement": "smooth continuous cinematic glide ending in dramatic pull-back",
            "lens_mm": 90,
        },
    },
    "continuous_orbit": {
        "id": "continuous_orbit",
        "label": "Orbital Hero Sweep",
        "tagline": "360° precision turntable dolly with crisp product focus",
        "camera": {
            "style": "precision robotic cinema dolly",
            "depth_of_field": "shallow to medium",
            "focus": "razor-sharp tack focus locked onto hero edges and specular highlights",
            "movement": "fluid 180-degree orbital sweep with subtle elevation crane",
            "lens_mm": 65,
        },
    },
    "pullback_reveal": {
        "id": "pullback_reveal",
        "label": "Macro → Pull-Back Reveal",
        "tagline": "Starts microscopic, pulls back to reveal the whole world",
        "camera": {
            "style": "cinema probe lens to wide anamorphic reveal",
            "depth_of_field": "ultra-shallow macro opening into crisp deep hero focus",
            "focus": "seamless rack focus from micro-texture to full hero silhouette",
            "movement": "intimate slow push-in followed by one continuous smooth pull-back reveal",
            "lens_mm": 50,
        },
    },
    "fpv_glide": {
        "id": "fpv_glide",
        "label": "Seamless One-Take Glide",
        "tagline": "Weightless continuous floating lens moving through space",
        "camera": {
            "style": "stabilized one-take cinema gimbal",
            "depth_of_field": "cinematic medium-shallow",
            "focus": "continuous forward focus tracking evolving focal planes",
            "movement": "unbroken forward glide weaving through scenes into a locked hero frame",
            "lens_mm": 35,
        },
    },
    "overhead_topdown": {
        "id": "overhead_topdown",
        "label": "Overhead Architectural",
        "tagline": "Symmetrical top-down tabletop choreography tilting to hero",
        "camera": {
            "style": "precision overhead cinema rig",
            "depth_of_field": "crisp controlled planar focus",
            "focus": "razor-sharp edge-to-edge surface detail transitioning to hero portrait",
            "movement": "symmetrical top-down crane descending and tilting smoothly into eye-level hero",
            "lens_mm": 50,
        },
    },
}

LIGHTING_PRESETS: dict[str, dict[str, Any]] = {
    "volumetric_gold": {
        "id": "volumetric_gold",
        "label": "Volumetric Gold",
        "tagline": "Warm glowing rays, rich shadows & magical depth",
        "lighting": {
            "style": "cinematic volumetric lighting",
            "effects": [
                "warm golden glow",
                "soft volumetric light rays",
                "realistic reflections",
                "natural shadows",
                "subtle atmospheric haze",
            ],
            "color_palette": ["#141210", "#E8A838", "#FFF1D6", "#2E3A48"],
        },
    },
    "chiaroscuro_studio": {
        "id": "chiaroscuro_studio",
        "label": "Obsidian Rim Studio",
        "tagline": "Sculpted key light, sleek edge highlights, zero clutter",
        "lighting": {
            "style": "high-contrast sculptural studio rim lighting",
            "effects": [
                "razor-sharp silver specular rim highlights",
                "velvety controlled shadow falloff",
                "subtle overhead softbox gradient",
                "authentic metallic micro-reflections",
            ],
            "color_palette": ["#0B0C10", "#F3F4F6", "#D9A441", "#252932"],
        },
    },
    "crisp_daylight": {
        "id": "crisp_daylight",
        "label": "Architectural Daylight",
        "tagline": "Clean sunlit clarity, natural caustics & warm stone tones",
        "lighting": {
            "style": "natural diffused architectural daylight",
            "effects": [
                "crisp window-blinds shadow geometry",
                "luminous glass caustics",
                "soft warm bounce fill",
                "true-to-life color rendition",
            ],
            "color_palette": ["#F5F1EA", "#1D1E22", "#C9965B", "#8FA3B0"],
        },
    },
    "neon_rain": {
        "id": "neon_rain",
        "label": "Cyber Neon & Rain",
        "tagline": "Electric cyan/amber reflections on wet obsidian surfaces",
        "lighting": {
            "style": "atmospheric nocturnal neon cinema lighting",
            "effects": [
                "electric cyan and warm amber wet reflections",
                "rain-mist volumetric halation",
                "deep moody teal shadows",
                "crisp specular highlights on glass",
            ],
            "color_palette": ["#080D14", "#22D3EE", "#F59E0B", "#1E293B"],
        },
    },
    "golden_hour": {
        "id": "golden_hour",
        "label": "Golden Hour Sunset",
        "tagline": "Long amber shadows, warm lens bloom & organic magic",
        "lighting": {
            "style": "natural late-afternoon golden hour backlight",
            "effects": [
                "low-angle amber sun flare",
                "long dramatic natural shadows",
                "warm rim glow on edges and mist",
                "soft pastel horizon gradient",
            ],
            "color_palette": ["#1E1510", "#F59E42", "#FFE6B8", "#4A5A70"],
        },
    },
}

MOTION_PRESETS: dict[str, dict[str, Any]] = {
    "asmr_tactile": {
        "id": "asmr_tactile",
        "label": "ASMR Satisfying",
        "tagline": "Silky, tactile, hypnotic physics & crisp micro-foley",
        "motion": {
            "style": "smooth, satisfying, seamless transitions",
            "physics": "realistic miniature-scale and fluid physics",
            "feel": "ASMR-like satisfying motion",
        },
        "audio_direction": (
            "Intimate tactile ASMR foley: crisp glass clinks, soft liquid drop resonance, "
            "gentle rustle of physical materials, warm ambient air, subtle resonant swell on reveal."
        ),
    },
    "epic_slowmo": {
        "id": "epic_slowmo",
        "label": "Cinema Slow-Mo",
        "tagline": "120fps prestige weight, suspended droplets & orchestral swell",
        "motion": {
            "style": "high-speed 120fps cinema slow motion easing into deliberate hero glide",
            "physics": "hyper-detailed fluid dynamics, inertia, and micro-particle suspension",
            "feel": "monumental, prestigious, breathtaking cinema weight",
        },
        "audio_direction": (
            "Deep cinematic soundscape: sub-bass pulse, crystalline high-speed shimmer, "
            "clean tactile impact hit, expansive warm harmonic swell at climax."
        ),
    },
    "seamless_morph": {
        "id": "seamless_morph",
        "label": "Seamless Metamorphosis",
        "tagline": "Continuous physical folding, blooming & world transforms",
        "motion": {
            "style": "unbroken physical transformation with fluid choreography",
            "physics": "mechanically plausible unfolding, blooming, and material transitions",
            "feel": "effortless wonder and continuous visual momentum",
        },
        "audio_direction": (
            "Organic mechanical and elemental transitions: crisp paper/wood folds, soft chime bloom, "
            "flowing water and breeze rising smoothly into a locked resonant chord."
        ),
    },
    "snappy_kinetic": {
        "id": "snappy_kinetic",
        "label": "Precision Kinetic",
        "tagline": "Crisp acceleration curves, exact snap-ins & modern tech energy",
        "motion": {
            "style": "precision speed-ramped camera motion with crisp ease-out locks",
            "physics": "magnetic precision assembly, crisp mechanical detents, zero wobble",
            "feel": "hyper-confident modern flagship launch energy",
        },
        "audio_direction": (
            "Precision industrial sound design: soft magnetic clicks, crisp optical shutter ticks, "
            "clean tonal riser, confident muted sub-impact on hero lockup."
        ),
    },
}

CONCEPT_RECIPES: list[dict[str, Any]] = [
    {
        "id": "miniature_jar_world",
        "title": "Miniature World in a Jar",
        "badge": "Viral Diorama",
        "idea": (
            "A giant human hand drops glowing liquid gold onto a frozen gray miniature city, "
            "bringing streets to life, folding the map into a glowing mountain village, "
            "and pulling back to reveal the entire living world sealed inside a glass jar."
        ),
        "world_preset": "miniature_diorama",
        "camera_preset": "macro_tilt_shift",
        "lighting_preset": "volumetric_gold",
        "motion_preset": "asmr_tactile",
    },
    {
        "id": "liquid_gold_product",
        "title": "Liquid Gold Product Alchemy",
        "badge": "Luxury Launch",
        "idea": (
            "Raw elemental ingredients melt into swirling molten gold and optical glass on obsidian stone, "
            "sculpting themselves smoothly in mid-air into our flagship product with a crisp rim-lit hero reveal."
        ),
        "world_preset": "surreal_glass",
        "camera_preset": "continuous_orbit",
        "lighting_preset": "chiaroscuro_studio",
        "motion_preset": "epic_slowmo",
    },
    {
        "id": "exploded_hardware",
        "title": "Exploded Precision Assembly",
        "badge": "Hardware Flagship",
        "idea": (
            "Microscopic machined aluminum, optical lenses, and sapphire glass components float in dark studio space, "
            "snapping together in hypnotic magnetic slow motion into the finished flagship device."
        ),
        "world_preset": "macro_studio",
        "camera_preset": "continuous_orbit",
        "lighting_preset": "chiaroscuro_studio",
        "motion_preset": "snappy_kinetic",
    },
    {
        "id": "paper_city_fold",
        "title": "Origami World Unfolding",
        "badge": "Craft & Storytelling",
        "idea": (
            "A flat blueprint sheet on a warm walnut drafting table folds itself like living origami into a sunlit "
            "architectural miniature city, illuminating from within before revealing the brand centerpiece."
        ),
        "world_preset": "paper_origami",
        "camera_preset": "overhead_topdown",
        "lighting_preset": "crisp_daylight",
        "motion_preset": "seamless_morph",
    },
    {
        "id": "saas_prism_clarity",
        "title": "Chaos to Glass Clarity",
        "badge": "SaaS / AI Launch",
        "idea": (
            "Tangled dark geometric blocks and scattered glowing light filaments rush through an optical glass prism, "
            "aligning effortlessly into a serene glowing crystal monolith and crisp product reveal."
        ),
        "world_preset": "surreal_glass",
        "camera_preset": "pullback_reveal",
        "lighting_preset": "neon_rain",
        "motion_preset": "snappy_kinetic",
    },
    {
        "id": "coffee_ritual_macro",
        "title": "Tactile ASMR Ritual",
        "badge": "Consumer / D2C",
        "idea": (
            "Ultra-macro slow motion of roasted beans, velvety water bloom, and golden crema swirling in sunlit "
            "travertine architecture, pulling back to a serene morning hero tabletop moment."
        ),
        "world_preset": "architectural_luxury",
        "camera_preset": "pullback_reveal",
        "lighting_preset": "golden_hour",
        "motion_preset": "asmr_tactile",
    },
]

REFINEMENT_CHIPS: list[dict[str, str]] = [
    {"id": "closer_macro", "label": "Closer macro & bokeh", "instruction": "Tighten camera to extreme macro with creamier shallow depth-of-field bokeh and visible surface micro-textures."},
    {"id": "dramatic_light", "label": "More volumetric light", "instruction": "Amplify dramatic volumetric light rays, golden rim highlights, and atmospheric depth across every beat."},
    {"id": "slower_asmr", "label": "Slower ASMR physics", "instruction": "Slow down transitions into hypnotic, tactile ASMR-like fluid and mechanical physics with richer micro-foley cues."},
    {"id": "pullback_ending", "label": "Dramatic pull-back reveal", "instruction": "Make the final 2 seconds a continuous sweeping pull-back reveal that reframes the entire world into a locked-off hero composition."},
    {"id": "hero_product_lock", "label": "Make product the final hero", "instruction": "Ensure the final beat and final_frame culminate unmistakably on the hero product centerpiece with balanced negative space."},
    {"id": "pure_visual", "label": "Zero AI text / pure visual", "instruction": "Remove any text, signs, or lettering from generated scenes so the story is 100% pure visual cinema with zero garbled text."},
    {"id": "warmer_sunset", "label": "Warmer golden palette", "instruction": "Shift lighting and color grade toward warm golden-hour sunset tones, amber glows, and inviting shadows."},
    {"id": "richer_audio", "label": "Richer foley & audio", "instruction": "Enrich every 2-second beat with specific tactile foley and acoustic transitions for native Veo 3 audio generation."},
]

DIRECTOR_ENGINES: list[dict[str, Any]] = [
    {
        "id": "kling/v4/text-to-video",
        "label": "Kling v4 Pro Cinema (10s · Start→End Bridge)",
        "act_duration_sec": 10,
        "supports_native_audio": False,
        "est_usd_per_act": 1.10,
        "badge": "Best for 10s–30s Chained",
    },
    {
        "id": "fal-ai/veo3/fast",
        "label": "Google Veo 3 Fast (8s · Native ASMR/Foley Audio)",
        "act_duration_sec": 8,
        "supports_native_audio": True,
        "est_usd_per_act": 1.20,
        "badge": "Native Audio",
    },
    {
        "id": "fal-ai/veo3",
        "label": "Google Veo 3 Flagship (8s · Studio Master + Audio)",
        "act_duration_sec": 8,
        "supports_native_audio": True,
        "est_usd_per_act": 2.40,
        "badge": "Flagship",
    },
    {
        "id": "kling/v4/flash/text-to-video",
        "label": "Kling v4 Flash (10s · Fast Draft)",
        "act_duration_sec": 10,
        "supports_native_audio": False,
        "est_usd_per_act": 0.94,
        "badge": "Fast Draft",
    },
    {
        "id": "mock",
        "label": "Instant Studio Animatic ($0 Offline Preview)",
        "act_duration_sec": 10,
        "supports_native_audio": False,
        "est_usd_per_act": 0.0,
        "badge": "$0 Preview",
    },
]


def get_director_presets_payload() -> dict[str, Any]:
    """Return the complete visual preset catalog for the Guided Director UI."""
    return {
        "recipes": CONCEPT_RECIPES,
        "worlds": list(WORLD_PRESETS.values()),
        "cameras": list(CAMERA_PRESETS.values()),
        "lightings": list(LIGHTING_PRESETS.values()),
        "motions": list(MOTION_PRESETS.values()),
        "refinement_chips": REFINEMENT_CHIPS,
        "engines": DIRECTOR_ENGINES,
        "durations": [
            {"sec": 10, "label": "10s · 1 Hero Act", "acts": 1},
            {"sec": 20, "label": "20s · 2 Chained Acts", "acts": 2},
            {"sec": 30, "label": "30s · 3 Chained Acts", "acts": 3},
        ],
    }


__all__ = [
    "CAMERA_PRESETS",
    "CONCEPT_RECIPES",
    "DIRECTOR_ENGINES",
    "LIGHTING_PRESETS",
    "MOTION_PRESETS",
    "REFINEMENT_CHIPS",
    "WORLD_PRESETS",
    "CameraSpec",
    "DirectorAct",
    "DirectorBlueprint",
    "EnvironmentSpec",
    "FinalFrameSpec",
    "LightingSpec",
    "MotionSpec",
    "SubjectDetailsSpec",
    "TimelineBeat",
    "get_director_presets_payload",
]
