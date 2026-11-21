"""Structured JSON Director Blueprint models & zero-jargon visual preset catalog.

Supports 10s (1 Act), 20s (2 Chained Acts), and 30s (3 Chained Acts) films compiled
into hyper-detailed, model-native structured JSON video prompts with locked visual
invariants (style, camera, environment, lighting, subject_details, motion) and
2-second choreographed timelines.
"""

from __future__ import annotations

import copy
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
    engine: str = "fal-ai/kling-video/v3/pro/text-to-video"
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

    def compile_act_model_prompt(self, act_index: int = 0, *, max_chars: int = 2400) -> str:
        """Compile an Act into valid compact JSON strictly within provider char limits (<= 2400)."""
        doc = self.to_act_prompt_dict(act_index)
        compact = json.dumps(doc, ensure_ascii=False, separators=(",", ":"))
        if len(compact) <= max_chars:
            return compact

        # Stage 1: drop verbose secondary top-level keys first (keeps style/camera/env/lighting/timeline/final_frame)
        trimmed = copy.deepcopy(doc)
        for drop_key in ("avoid", "subject_details", "audio"):
            trimmed.pop(drop_key, None)
            compact = json.dumps(trimmed, ensure_ascii=False, separators=(",", ":"))
            if len(compact) <= max_chars:
                return compact

        # Stage 2: drop optional color_palette/atmosphere/per-beat sound when over budget
        if isinstance(trimmed.get("lighting"), dict):
            trimmed["lighting"].pop("color_palette", None)
        if isinstance(trimmed.get("environment"), dict):
            trimmed["environment"].pop("atmosphere", None)
        if isinstance(trimmed.get("timeline"), list):
            for beat in trimmed["timeline"]:
                if isinstance(beat, dict):
                    beat.pop("sound", None)
        compact = json.dumps(trimmed, ensure_ascii=False, separators=(",", ":"))
        if len(compact) <= max_chars:
            return compact

        # Stage 3: proportionally condense beat strings while preserving 100% valid JSON & all 5 beats
        tl = trimmed.get("timeline")
        if isinstance(tl, list) and tl:
            for cap in (150, 120, 95, 75, 60):
                for beat in tl:
                    if isinstance(beat, dict):
                        for k in ("scene", "action", "camera"):
                            val = beat.get(k)
                            if isinstance(val, str) and len(val) > cap:
                                beat[k] = val[: cap - 1].rstrip() + "…"
                compact = json.dumps(trimmed, ensure_ascii=False, separators=(",", ":"))
                if len(compact) <= max_chars:
                    return compact

        # Stage 4: compact remaining global strings if still slightly over
        for sec in ("environment", "camera", "lighting", "motion", "final_frame"):
            obj = trimmed.get(sec)
            if isinstance(obj, dict):
                for k, v in list(obj.items()):
                    if isinstance(v, str) and len(v) > 70:
                        obj[k] = v[:69].rstrip() + "…"
                    elif isinstance(v, list) and len(v) > 3:
                        obj[k] = v[:3]
        compact = json.dumps(trimmed, ensure_ascii=False, separators=(",", ":"))
        return compact if len(compact) <= max_chars else compact[:max_chars]

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
    "anthropic_editorial": {
        "id": "anthropic_editorial",
        "label": "Anthropic Warm Keynote",
        "tagline": "Warm alabaster editorial typography, tactile craft & calm human intelligence",
        "style": "warm editorial studio cinema & minimalist kinetic typography launch film",
        "environment": {
            "surface": "warm alabaster architectural plaster and matte travertine studio plinth",
            "world": "calm sunlit modern design studio balancing human warmth and quiet precision",
            "materials": "warm cotton rag paper, brushed terracotta anodize, matte limestone, soft linen",
            "atmosphere": "clean northern window daylight with gentle architectural shadow falloff",
        },
        "subject_details": {
            "appearance": "thoughtful editorial product showcase paired with calm kinetic typography",
            "textures": "matte mineral paper, fine anodized aluminum, natural daylight skin & fabric tones",
            "interaction": "unhurried deliberate motion, tactile page-like transitions, quiet focus",
            "scale": "intimate editorial framing transitioning seamlessly into crisp UI clarity",
        },
    },
    "openai_product_drop": {
        "id": "openai_product_drop",
        "label": "OpenAI Minimal Product Drop",
        "tagline": "High-contrast monochrome glassmorphic UI, live prompt choreography & crisp realism",
        "style": "ultra-clean minimalist monochrome product launch & live interface cinema",
        "environment": {
            "surface": "matte obsidian glass and seamless soft-lit studio cyclorama",
            "world": "minimalist flagship product stage highlighting effortless human-AI interaction",
            "materials": "frosted optical glass, precision bead-blasted aluminum, crisp monochrome surfaces",
            "atmosphere": "high-contrast softbox studio clarity with subtle emerald & white edge light",
        },
        "subject_details": {
            "appearance": "iconic minimalist hero product and ultra-crisp interactive interface choreography",
            "textures": "micro-textured matte glass, razor-sharp UI geometry, clean specular rim highlights",
            "interaction": "snappy spring-physics reveals, instant visual feedback, effortless precision",
            "scale": "clean centered hero framing with generous architectural negative space",
        },
    },
    "gemini_multimodal": {
        "id": "gemini_multimodal",
        "label": "Gemini Multimodal Launch",
        "tagline": "Deep cosmos navy, luminous prism gradients, fast multimodal UI & benchmark proof",
        "style": "kinetic luminous prism glass & high-energy multimodal tech keynote film",
        "environment": {
            "surface": "deep midnight obsidian glass with luminous sapphire and violet refractions",
            "world": "futuristic luminous optical stage uniting real-world imagery, code, and benchmarks",
            "materials": "optical prism crystal, glowing sapphire caustics, dark smoked glass, brushed titanium",
            "atmosphere": "radiant volumetric blue-violet light beams and crisp energetic micro-glints",
        },
        "subject_details": {
            "appearance": "dynamic multimodal showcase blending optical refraction with rapid UI & chart proof",
            "textures": "iridescent glass dispersion, crisp luminous UI gradients, polished dark metal",
            "interaction": "energetic beat-synced transitions, fluid optical morphs, bold benchmark roll-ups",
            "scale": "expansive widescreen keynote framing with bold visual rhythm",
        },
    },
    "founder_keynote_hybrid": {
        "id": "founder_keynote_hybrid",
        "label": "Founder + Product Launch",
        "tagline": "Founder talking-head A-roll seamlessly interwoven with crisp UI graphics & AI B-roll",
        "style": "authentic founder-led product launch combining real footage, motion graphics & AI B-roll",
        "environment": {
            "surface": "warm modern founder studio desk with soft depth-of-field backdrop",
            "world": "authentic founder storytelling anchored by crisp on-screen UI proof and visual metaphors",
            "materials": "natural studio lighting, warm wood, clean dark-glass UI frames, subtle brass accents",
            "atmosphere": "natural documentary-grade key light with clean editorial polish",
        },
        "subject_details": {
            "appearance": "authentic founder presence paired with sharp UI callouts and tactile B-roll cutaways",
            "textures": "natural 35mm lens depth, crisp vector motion graphics, tactile studio surfaces",
            "interaction": "continuous founder voiceover with J/L-cut visual cutaways on key product moments",
            "scale": "conversational medium close-up alternating with full-bleed UI and hero B-roll",
        },
    },
    "apple_titanium": {
        "id": "apple_titanium",
        "label": "Apple Titanium Keynote",
        "tagline": "Pitch-black cyclorama, razor rim light, orbital hardware macro & bold centered stats",
        "style": "ultra-clean obsidian hardware keynote cinema with razor-thin rim lighting",
        "environment": {
            "surface": "seamless matte obsidian black cyclorama with subtle brushed titanium reflection",
            "world": "iconic flagship hardware keynote stage with precision spotlight drama",
            "materials": "grade-5 brushed titanium, sapphire optical glass, matte black anodized aluminum",
            "atmosphere": "pure dark studio negative space pierced by crisp silver-white rim light",
        },
        "subject_details": {
            "appearance": "sculptural flagship product emerging cleanly out of dramatic dark shadow",
            "textures": "micro-blasted titanium grain, mirror chamfers, razor-sharp display glass",
            "interaction": "slow deliberate orbital precision with bold stat reveal cuts",
            "scale": "monumental macro detail pulling back to iconic centered silhouette",
        },
    },
    "linear_dark_craft": {
        "id": "linear_dark_craft",
        "label": "Linear / Vercel Dark Craft",
        "tagline": "Obsidian dot-grid, indigo edge glow, 2.5D UI zoom & snappy keyboard-first SaaS polish",
        "style": "high-craft dark-mode developer software launch with luminous indigo edge glow",
        "environment": {
            "surface": "deep obsidian architectural grid with subtle violet-indigo ambient gradient",
            "world": "precision software studio showcasing fast, tactile developer craftsmanship",
            "materials": "smoked glass panels, hairline 1px illuminated borders, matte dark slate",
            "atmosphere": "crisp dark-mode contrast with restrained electric indigo glow",
        },
        "subject_details": {
            "appearance": "razor-sharp interactive interface and terminal choreography in 2.5D motion",
            "textures": "clean vector geometry, frosted glass layers, crisp monospace code",
            "interaction": "spring-physics zoom-ins, instant command execution, zero wasted frames",
            "scale": "widescreen interface stage zooming smoothly onto active interactions",
        },
    },
    "stripe_living_flow": {
        "id": "stripe_living_flow",
        "label": "Stripe Living Architecture",
        "tagline": "Crisp light canvas, animated node-and-connector flows, live metrics & fintech clarity",
        "style": "editorial light-mode fintech architecture with animated data pipelines and crisp clarity",
        "environment": {
            "surface": "clean architectural white and cool slate canvas with subtle mesh gradient",
            "world": "modern financial infrastructure stage turning complex systems into visual clarity",
            "materials": "frosted acrylic cards, luminous indigo/emerald connectors, fine hairline grids",
            "atmosphere": "bright editorial daylight clarity with crisp drop shadows",
        },
        "subject_details": {
            "appearance": "living architectural node diagrams, live payloads, and smooth metric roll-ups",
            "textures": "clean vector surfaces, crisp geometric typography, subtle glass refraction",
            "interaction": "flowing animated data packets linking nodes into instant proof",
            "scale": "balanced widescreen infographic stage with generous breathing room",
        },
    },
    "notion_playful": {
        "id": "notion_playful",
        "label": "Notion / Figma Editorial",
        "tagline": "Warm cream paper canvas, crisp ink borders, bouncy cursor choreography & human warmth",
        "style": "playful tactile editorial software film on warm paper canvas with ink-crisp framing",
        "environment": {
            "surface": "warm cream heavy stationery paper with clean editorial margins",
            "world": "inviting creative workspace combining structured clarity and human charm",
            "materials": "warm cotton paper, matte charcoal ink lines, soft terracotta & sage accents",
            "atmosphere": "bright friendly daylight studio feel with tactile warmth",
        },
        "subject_details": {
            "appearance": "expressive editorial typography paired with friendly live UI walkthroughs",
            "textures": "crisp ink outlines, warm off-white cards, organic serif + sans contrast",
            "interaction": "playful cursor clicks, springy card pops, smooth before-and-after wipes",
            "scale": "inviting desktop canvas framing real workflows cleanly",
        },
    },
    "screen_studio_tour": {
        "id": "screen_studio_tour",
        "label": "Screen-Studio Product Tour",
        "tagline": "2.5D tilted browser/device walkthroughs with smooth cursor zooms & zero fluff",
        "style": "sleek 2.5D perspective software walkthrough with smooth cursor-tracked camera zooms",
        "environment": {
            "surface": "deep subtle studio gradient backdrop framing a floating hardware display",
            "world": "focused product-first demo stage engineered for Product Hunt and launch day",
            "materials": "anodized display bezel, crisp anti-reflective screen glass, soft shadow bloom",
            "atmosphere": "clean focused studio spotlight centering 100% attention on the live product",
        },
        "subject_details": {
            "appearance": "real product UI tilted in smooth 2.5D perspective with live cursor focus",
            "textures": "pixel-crisp interface detail, subtle glass specular highlight, clean typography",
            "interaction": "cinematic zoom-in onto key UI action followed by crisp feature proof",
            "scale": "hero device framing transitioning smoothly between full view and zoomed detail",
        },
    },
    "airbnb_human_story": {
        "id": "airbnb_human_story",
        "label": "Airbnb Warm Human Story",
        "tagline": "35mm sunlit human storytelling paired with warm editorial overlays & social proof",
        "style": "warm 35mm anamorphic human lifestyle cinema paired with understated editorial UI",
        "environment": {
            "surface": "sunlit lived-in architectural interiors and natural outdoor golden-hour scenes",
            "world": "authentic human-first brand story grounding technology in everyday joy",
            "materials": "warm natural timber, woven linen, sunlight through glass, authentic skin tones",
            "atmosphere": "golden-hour natural light with gentle 35mm organic bokeh",
        },
        "subject_details": {
            "appearance": "relatable human moments seamlessly bridging into clean product & quote proof",
            "textures": "natural film grain, warm organic fabrics, clean minimal editorial typography",
            "interaction": "emotional problem-to-relief pacing with smooth cross-dissolves",
            "scale": "intimate human-eye perspective opening into warm lifestyle tableau",
        },
    },
    "viral_vertical_reel": {
        "id": "viral_vertical_reel",
        "label": "9:16 Hook-First Growth Reel",
        "tagline": "Scroll-stopping 1.5s vertical hook, mobile device showcase, bold captions & punchy CTA",
        "style": "high-retention 9:16 vertical social ad with pattern-interrupt kinetic hook & bold captions",
        "environment": {
            "surface": "high-contrast dark studio vertical canvas engineered for mobile feeds",
            "world": "fast-paced vertical growth video optimized for Reels, TikTok, and YouTube Shorts",
            "materials": "bold high-contrast typography, sleek mobile glass frame, luminous accent pills",
            "atmosphere": "energetic punchy contrast engineered for instant thumb-stop clarity",
        },
        "subject_details": {
            "appearance": "oversized hook headline snapping into mobile product proof and rapid stats",
            "textures": "crisp high-legibility mobile UI, bold color blocks, smooth kinetic motion",
            "interaction": "fast 3-second rhythm with zero dead air and safe-zone aligned captions",
            "scale": "full-bleed 9:16 vertical framing with mobile-native stacked layouts",
        },
    },
    "enterprise_case_study": {
        "id": "enterprise_case_study",
        "label": "Customer Proof & ROI Cut",
        "tagline": "Executive metric hook, verified customer quote card & animated benchmark ROI proof",
        "style": "authoritative B2B executive case study with crisp quote cards and verified ROI charts",
        "environment": {
            "surface": "refined deep navy and slate executive presentation canvas",
            "world": "high-trust enterprise GTM stage turning customer results into compelling video proof",
            "materials": "matte architectural slate, subtle gold/emerald stat accents, crisp editorial serif",
            "atmosphere": "calm boardroom authority with razor-sharp data legibility",
        },
        "subject_details": {
            "appearance": "verified customer quote, bold ROI percentage reveal, and comparative bar proof",
            "textures": "clean editorial cards, fine hairline dividers, high-contrast tabular numerals",
            "interaction": "deliberate executive pacing building from headline metric to social proof",
            "scale": "clean widescreen keynote framing built for LinkedIn and sales decks",
        },
    },
    "a24_indie_35mm": {
        "id": "a24_indie_35mm",
        "label": "A24 Indie 35mm",
        "tagline": "Kodak Vision3 500T grain, warm halation, intimate handheld 35mm emotion",
        "style": "intimate A24-style 35mm photochemical cinema shot on Kodak Vision3 500T film",
        "environment": {
            "surface": "lived-in warm interiors, dusk-lit windows, textured plaster and worn wood",
            "world": "poetic grounded real-world setting rich with quiet human atmosphere",
            "materials": "organic 35mm silver halide grain, soft amber halation around tungsten lamps, natural linen",
            "atmosphere": "nostalgic twilight warmth, shallow anamorphic bokeh, dust motes in window beams",
        },
        "subject_details": {
            "appearance": "authentic expressive human presence and tactile hero details in natural light",
            "textures": "real skin pores, soft woven fabrics, warm practical light rolloff",
            "interaction": "subtle breathing handheld intimacy, unhurried emotional beats",
            "scale": "establishing atmospheric wide into intimate close-up detail",
        },
    },
    "imax_70mm_epic": {
        "id": "imax_70mm_epic",
        "label": "IMAX 70mm Epic",
        "tagline": "Monumental Villeneuve/Nolan scale, sweeping vistas & colossal camera push-ins",
        "style": "monumental 65mm IMAX theatrical cinema with awe-inspiring architectural & natural scale",
        "environment": {
            "surface": "vast windswept monolithic landscapes, brutalist stone, reflective dark water",
            "world": "colossal cinematic world evoking awe, scale, and frontier ambition",
            "materials": "weathered basalt, brushed titanium monoliths, atmospheric mist, volumetric sun shafts",
            "atmosphere": "towering clouds, suspended dust clouds, dramatic low-angle god rays",
        },
        "subject_details": {
            "appearance": "lone silhouette or monumental hero form framed against vast scale",
            "textures": "razor-sharp 70mm large-format clarity, fine mineral grit, misty depth",
            "interaction": "slow inexorable aerial & crane push-ins with heavy physical presence",
            "scale": "vast epic wide vistas converging onto an iconic centerpiece",
        },
    },
    "luxury_maison_noir": {
        "id": "luxury_maison_noir",
        "label": "Luxury Maison Noir",
        "tagline": "Chiaroscuro shadow drama, liquid silk & gold speculars, 120fps phantom elegance",
        "style": "haute-couture Parisian luxury commercial cinema with chiaroscuro lighting & 120fps slow-motion",
        "environment": {
            "surface": "polished black Nero Marquina marble, liquid obsidian mirror reflections",
            "world": "intimate high-fashion luxury atelier sculpted by shadow and warm gold light",
            "materials": "heavy liquid silk, 24k brushed gold, cut crystal glass, rich velvet shadow",
            "atmosphere": "dramatic chiaroscuro key light, smoky amber caustics, zero clutter",
        },
        "subject_details": {
            "appearance": "sculptural luxury centerpiece catching slow-moving golden light ribbons",
            "textures": "micro-polished metal, flowing silk drape, crystalline refractive sparkle",
            "interaction": "weightless 120fps phantom slow-motion with hypnotic fluid grace",
            "scale": "sensual macro close-ups revealing iconic silhouette",
        },
    },
    "blade_runner_anamorphic": {
        "id": "blade_runner_anamorphic",
        "label": "Neon Rain Anamorphic",
        "tagline": "Rain-slicked nocturnal streets, blue anamorphic lens flares & teal-amber contrast",
        "style": "neo-noir anamorphic sci-fi cinema with wet asphalt reflections and volumetric rain",
        "environment": {
            "surface": "rain-slicked dark asphalt mirroring glowing cyan and amber light",
            "world": "moody nocturnal metropolis thick with steam, rain, and atmospheric depth",
            "materials": "wet dark concrete, raindrop-beaded glass, brushed gunmetal, glowing mist",
            "atmosphere": "horizontal blue anamorphic flares, volumetric street steam, teal-and-amber contrast",
        },
        "subject_details": {
            "appearance": "striking silhouette and hero form illuminated by nocturnal rain reflections",
            "textures": "beading rainwater, wet leather/metal sheen, rich atmospheric haze",
            "interaction": "raindrops rippling in slow motion across surfaces as camera glides forward",
            "scale": "moody widescreen street depth pushing into razor-sharp hero focus",
        },
    },
    "wes_pastel_symmetry": {
        "id": "wes_pastel_symmetry",
        "label": "Symmetrical Pastel Cinema",
        "tagline": "Dead-center Wes Anderson symmetry, whip-pans & warm storybook palettes",
        "style": "dead-center symmetrical planimetric 35mm cinema with curated pastel production design",
        "environment": {
            "surface": "hand-crafted vintage parlor wood, pastel mint walls, brass bellhop trim",
            "world": "whimsical meticulously art-directed diorama-like live-action world",
            "materials": "matte pastel lacquer, embossed leather, warm brass hardware, crisp woven tweed",
            "atmosphere": "even warm sunlit studio clarity with storybook color harmony",
        },
        "subject_details": {
            "appearance": "perfectly centered hero subject framed with geometric architectural precision",
            "textures": "tactile vintage craftsmanship, rich matte colors, crisp shadow edges",
            "interaction": "precise mechanical choreography, crisp lateral dolly & snap-zoom framing",
            "scale": "locked symmetrical tableau transitioning via clean axial push-in",
        },
    },
    "kinetic_sports_grit": {
        "id": "kinetic_sports_grit",
        "label": "16mm Kinetic Grit & Sweat",
        "tagline": "High-contrast 16mm grain, stadium rim light & explosive athletic energy",
        "style": "raw 16mm Kodak film commercial with crisp shutter energy, sweat & dramatic rim light",
        "environment": {
            "surface": "wet concrete tunnel, rain-drenched track, dramatic dark training arena",
            "world": "high-intensity athletic proving ground charged with adrenaline and focus",
            "materials": "raw 16mm film grain, glistening sweat droplets, textured rubber, chalk dust",
            "atmosphere": "harsh overhead sodium/LED stadium backlight cutting through cold mist",
        },
        "subject_details": {
            "appearance": "explosive human motion and high-performance gear under peak tension",
            "textures": "suspended water & chalk particles, taut technical fabric, raw film halation",
            "interaction": "dynamic low-angle tracking shots synced to breath and footstrike impact",
            "scale": "visceral close-up intensity erupting into full-stride hero power",
        },
    },
    "painted_anime_cinema": {
        "id": "painted_anime_cinema",
        "label": "Hand-Painted Anime Film",
        "tagline": "Lush gouache skies, wind-swept landscapes & warm Ghibli/Shinkai light beams",
        "style": "theatrical hand-painted 2D cel animation feature film with luminous gouache backgrounds",
        "environment": {
            "surface": "sun-dappled emerald meadow, wooden hillside veranda, towering cumulus clouds",
            "world": "breathtaking hand-painted animated world alive with wind, light, and wonder",
            "materials": "watercolor & gouache brushwork, warm cel-shaded sunlight, fluttering leaves",
            "atmosphere": "golden-hour sunbeams, drifting cherry petals, shimmering summer sky",
        },
        "subject_details": {
            "appearance": "expressive hand-drawn hero subject with fluid wind-swept motion on twos",
            "textures": "painterly cloud strokes, warm rim-lit linework, sparkling water reflections",
            "interaction": "gentle breeze rippling across grass and fabric into a sweeping horizon reveal",
            "scale": "intimate quiet moment expanding into a panoramic painted sky",
        },
    },
    "natgeo_natural_world": {
        "id": "natgeo_natural_world",
        "label": "Natural World Documentary",
        "tagline": "Golden-hour telephoto clarity, alpine mist & pristine Earth cinematography",
        "style": "BBC/NatGeo flagship 8K natural history cinema shot on long telephoto cinema glass",
        "environment": {
            "surface": "dewy alpine moss, glacier-fed crystal river, ancient sunlit redwood canopy",
            "world": "untouched natural wilderness captured at peak golden-hour magic",
            "materials": "crystalline morning dew, textured bark, volcanic stone, drifting valley fog",
            "atmosphere": "crisp dawn mist pierced by slant golden sunlight and natural serenity",
        },
        "subject_details": {
            "appearance": "pristine natural world and authentic outdoor craft in organic harmony",
            "textures": "micro-droplets on foliage, rippling crystal water, crisp feather/stone detail",
            "interaction": "patient stabilized gimbal glide revealing untouched natural grandeur",
            "scale": "microscopic morning dew pulling back to a sweeping mountain horizon",
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
        "id": "fal-ai/kling-video/v3/pro/text-to-video",
        "label": "Kling 3.0 Pro Cinema (10s · 1080p · Start→End + Native Audio)",
        "act_duration_sec": 10,
        "supports_native_audio": True,
        "est_usd_per_act": 1.68,
        "badge": "Best for 10s–30s Chained",
    },
    {
        "id": "kling/v4/flash/text-to-video",
        "label": "Kling 4.0 Flash (10s · 720p · Fast & Cost-Efficient)",
        "act_duration_sec": 10,
        "supports_native_audio": False,
        "est_usd_per_act": 0.94,
        "badge": "Fast Draft",
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
