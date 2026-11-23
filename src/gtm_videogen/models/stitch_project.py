"""Stitch Project schemas, Script Gate Validator, and Model-Native JSON Compilers.

Powers the 4-stage Stitch workflow:
  1. Discover & Pitch (Idea / URL / Curated Gallery Recipes / 3 Concept Pitches)
  2. Pinboard & Script Gate (Visual Scene Flow Diagram + Live Validation + Auto-Fix)
  3. Shoot & Takes (Reference Elements @Element1..3 + Editable Keyframes + Per-Act Takes)
  4. Screening Room (Unified Soundstage + Seam Transitions + Instant <5s Master Stitch)
"""

from __future__ import annotations

import copy
import json
import re
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

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
from gtm_videogen.models.model_registry import VideoModelSpec

StitchAspect = Literal["16:9", "9:16", "1:1"]
SceneKind = Literal["ai_sequence", "kinetic_ui", "motion_graphic", "user_clip"]
AudioMode = Literal["native_only", "native_plus_score", "vo_plus_ducked_score"]
SeamTransition = Literal["invisible_chain", "micro_dissolve", "audio_riser"]
MotionTemplateId = Literal[
    "kinetic_headline",
    "prompt_ui_demo",
    "benchmark_bars",
    "bento_feature_grid",
    "terminal_code",
    "split_comparison",
    "device_showcase_25d",
    "node_flow_diagram",
    "customer_quote_proof",
    "kinetic_hook_reel",
]
MotionThemeMode = Literal["dark_glass", "warm_editorial", "clean_light"]
UserClipFraming = Literal["cover_crop", "contain_blur", "device_bezel"]
OverlayKind = Literal["lower_third", "stat_pill", "feature_callout", "brand_watermark"]
OverlayPosition = Literal["bottom_left", "top_right", "bottom_center"]
TimelineMode = Literal["sequential_acts", "aroll_with_cutaways"]
CaptionStyle = Literal["minimal_bottom", "karaoke_highlight", "editorial_serif"]

MOTION_TEMPLATE_CATALOG: list[dict[str, str]] = [
    {
        "id": "kinetic_headline",
        "label": "Kinetic Editorial Headline",
        "tagline": "Anthropic-style warm masked typography with spring reveal",
    },
    {
        "id": "device_showcase_25d",
        "label": "2.5D Screen-Studio UI Zoom",
        "tagline": "3D-tilted MacBook/iPhone product walkthrough with smooth cursor zoom",
    },
    {
        "id": "prompt_ui_demo",
        "label": "Live Prompt & UI Demo",
        "tagline": "OpenAI/Gemini-style typing prompt bar, cursor click & response card",
    },
    {
        "id": "node_flow_diagram",
        "label": "Living Architecture Node Flow",
        "tagline": "Stripe-style animated connected pipeline nodes with glowing data packets",
    },
    {
        "id": "benchmark_bars",
        "label": "Animated Benchmark Chart",
        "tagline": "Animated horizontal bar chart with rolling counters & hero highlight",
    },
    {
        "id": "bento_feature_grid",
        "label": "Bento Feature Grid",
        "tagline": "3 glassmorphic capability cards staggering in (auto-stacks in 9:16)",
    },
    {
        "id": "terminal_code",
        "label": "Terminal / SDK Code Reveal",
        "tagline": "Dark macOS terminal typing a real command/SDK call + green deploy badge",
    },
    {
        "id": "split_comparison",
        "label": "Before vs. After Split",
        "tagline": "Crisp split-screen comparison wipe (auto-stacks in 9:16)",
    },
    {
        "id": "customer_quote_proof",
        "label": "Customer Quote & ROI Proof",
        "tagline": "Editorial testimonial card with attribution & animated ROI metric badge",
    },
    {
        "id": "kinetic_hook_reel",
        "label": "Pattern-Interrupt Social Hook",
        "tagline": "High-contrast 1.5s word slam reveal engineered for Reels, Shorts & TikTok",
    },
]

_TEXT_IN_SCENE_RE = re.compile(
    r"\b(neon sign|billboard|sign reading|text reading|words reading|logo text|banner saying|written words|readable text)\b",
    re.IGNORECASE,
)


class StitchBeat(BaseModel):
    """A 2-5 second choreographed beat inside a Stitch Scene."""

    model_config = ConfigDict(extra="ignore")

    time: str = "0-5s"
    scene: str = ""
    action: str = ""
    camera: str = ""
    sound: str = ""


class MotionGraphicSpec(BaseModel):
    """Parameterized spec for crisp 24fps procedural motion graphics ($0/shot, Render-512MB safe)."""

    model_config = ConfigDict(extra="ignore")

    template_id: MotionTemplateId = "prompt_ui_demo"
    theme_mode: MotionThemeMode = "dark_glass"
    eyebrow: str = ""
    headline: str = ""
    subheadline: str = ""
    prompt_text: str = ""
    response_lines: list[str] = Field(default_factory=list)
    metrics: list[dict[str, Any]] = Field(default_factory=list)
    screenshot_path: str | None = None


class UserClipSpec(BaseModel):
    """Uploaded founder video or screen recording with trim, framing, and audio controls."""

    model_config = ConfigDict(extra="ignore")

    raw_video_path: str = ""
    filename: str = "clip.mp4"
    source_duration_sec: float = 10.0
    trim_start_sec: float = Field(default=0.0, ge=0.0)
    trim_end_sec: float | None = None
    framing: UserClipFraming = "cover_crop"
    keep_original_audio: bool = True
    transcript_words: list[dict[str, Any]] = Field(default_factory=list)


class SceneOverlaySpec(BaseModel):
    """Transparent RGBA graphic overlay composited over a scene via FFmpeg."""

    model_config = ConfigDict(extra="ignore")

    overlay_id: str = "ov_1"
    kind: OverlayKind = "lower_third"
    primary_text: str = ""
    secondary_text: str = ""
    position: OverlayPosition = "bottom_left"
    start_sec: float = Field(default=0.4, ge=0.0)
    end_sec: float = Field(default=4.2, ge=0.5)


class StitchTake(BaseModel):
    """One versioned video render take for a single Stitch Scene/Act."""

    model_config = ConfigDict(extra="ignore")

    take_id: int = Field(..., ge=1)
    engine_id: str
    engine_label: str
    clip_path: str
    duration_sec: float = 10.0
    usd: float = 0.0
    chained_from_prev: bool = False
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class StitchElement(BaseModel):
    """Uploaded visual identity reference (@Element1..3) locked across 0s -> 60s."""

    model_config = ConfigDict(extra="ignore")

    slot: int = Field(..., ge=1, le=3)
    tag: str = "@Element1"
    role: Literal["product", "character", "style"] = "product"
    label: str = "Hero Reference"
    image_path: str
    filename: str = "element.png"


class StitchScene(BaseModel):
    """One Scene/Act (3s -> 60s) in the Visual Scene Flow Diagram."""

    model_config = ConfigDict(extra="ignore")

    scene_id: str
    index: int = 0
    title: str = "Scene 1 · Opening Hook"
    narrative_role: str = ""  # hook | problem | turning_point | product_reveal | deep_dive | proof | climax_cta
    duration_sec: int = Field(default=10, ge=3, le=60)
    kind: SceneKind = "ai_sequence"
    continues_from_prev: bool = False
    video_model_id: str | None = None
    enable_native_audio: bool = False

    # Structured surgical controls shown on the Pinboard & Slide-Over Drawer
    subjects: list[str] = Field(default_factory=list)
    materials: list[str] = Field(default_factory=list)
    camera_spec: str = "90mm macro tilt-shift · smooth continuous push-in"
    colors: list[str] = Field(default_factory=lambda: ["#141210", "#E8A838", "#F4E8D1"])
    vo_line: str = ""
    vo_audio_path: str | None = None
    end_frame: str = "Locked-off hero composition with balanced negative space"
    beats: list[StitchBeat] = Field(default_factory=list)

    # Hybrid Launch Controls (Motion Graphics, User Clips, Overlays & A-Roll Cutaway Window)
    graphic_spec: MotionGraphicSpec | None = None
    user_clip: UserClipSpec | None = None
    overlays: list[SceneOverlaySpec] = Field(default_factory=list)
    aroll_window: tuple[float, float] | None = None

    # Editable / Uploadable Start & End Anchor Keyframes
    start_keyframe_path: str | None = None
    start_keyframe_url: str | None = None
    end_keyframe_path: str | None = None
    end_keyframe_url: str | None = None

    # Non-Linear Take History
    takes: list[StitchTake] = Field(default_factory=list)
    winning_take_id: int | None = None
    render_status: str = "idle"  # idle | rendering | done | failed
    render_message: str = ""
    render_error: str | None = None

    def winning_take(self) -> StitchTake | None:
        if not self.takes:
            return None
        if self.winning_take_id is not None:
            for t in self.takes:
                if t.take_id == self.winning_take_id:
                    return t
        return self.takes[-1]


class StitchProject(BaseModel):
    """Complete Stitch film project (10s -> 60s) with Global World Lock, Scenes, Elements & Soundstage."""

    model_config = ConfigDict(extra="ignore")

    project_id: str
    title: str = "Untitled Film"
    idea: str = ""
    source_url: str | None = None
    scraped_context: str = ""
    recipe_id: str | None = None
    brand_kit_loaded: bool = False
    brand_kit_summary: str = ""

    aspect_ratio: StitchAspect = "16:9"
    target_duration_sec: int = 15
    video_model_id: str = "auto"
    llm_model_id: str = ""
    master_anchor_path: str | None = None
    master_anchor_url: str | None = None

    # Visual Preset IDs + Global World Lock (shared across all Scenes)
    world_preset: str = "miniature_diorama"
    camera_preset: str = "macro_tilt_shift"
    lighting_preset: str = "volumetric_gold"
    motion_preset: str = "asmr_tactile"

    style: str = "ultra-photorealistic cinematic miniature diorama"
    main_subject: str = ""
    camera: CameraSpec = Field(default_factory=CameraSpec)
    environment: EnvironmentSpec = Field(default_factory=EnvironmentSpec)
    lighting: LightingSpec = Field(default_factory=LightingSpec)
    subject_details: SubjectDetailsSpec = Field(default_factory=SubjectDetailsSpec)
    motion: MotionSpec = Field(default_factory=MotionSpec)
    audio_direction: str = (
        "Intimate tactile ASMR foley: crisp glass clinks, liquid drop resonance, "
        "subtle room air, and warm harmonic swell on hero reveal."
    )
    negative_constraints: list[str] = Field(
        default_factory=lambda: [
            "strictly zero readable text, signs, or watermarks",
            "no sudden jump cuts or visual flicker",
            "no plastic CGI sheen or warped hands",
        ]
    )

    elements: list[StitchElement] = Field(default_factory=list)
    scenes: list[StitchScene] = Field(default_factory=list)

    # Hybrid Timeline: Founder A-Roll + Cutaways & Captions
    timeline_mode: TimelineMode = "sequential_acts"
    aroll_clip_path: str | None = None
    aroll_filename: str | None = None
    aroll_duration_sec: float = 0.0
    aroll_transcript: list[dict[str, Any]] = Field(default_factory=list)
    enable_captions: bool = False
    caption_style: CaptionStyle = "minimal_bottom"

    # Soundstage & Master Finishing Controls
    audio_mode: AudioMode = "native_plus_score"
    seam_transition: SeamTransition = "invisible_chain"
    voice_name: str = "Rachel"
    add_brand_outro: bool = False
    brand_name: str | None = None
    brand_tagline: str | None = None
    brand_logo_path: str | None = None

    # Master Cut, Automated Video QA Scorecard & Multi-Format Campaign Pack Telemetry
    cinema_letterbox: bool = False
    film_grain: bool = False
    master_video_path: str | None = None
    qa_report: dict[str, Any] | None = None
    campaign_pack_paths: dict[str, str] = Field(default_factory=dict)
    status: str = "draft"  # draft | validated | shooting | stitching | done | failed
    status_message: str = "Ready to review on your Pinboard."
    error: str | None = None
    actual_usd: float = 0.0
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())

    def total_duration_sec(self) -> int:
        return sum(int(s.duration_sec) for s in self.scenes)

    def to_scene_prompt_dict(self, scene_idx: int) -> dict[str, Any]:
        """Export one Scene as a crisp, model-native structured JSON prompt dictionary."""
        idx = max(0, min(scene_idx, len(self.scenes) - 1))
        sc = self.scenes[idx]
        doc: dict[str, Any] = {
            "type": "video_prompt",
            "duration": f"{sc.duration_sec} seconds",
            "aspect_ratio": self.aspect_ratio,
            "resolution": "4K",
            "style": self.style,
            "camera": {
                **self.camera.model_dump(),
                "scene_move": sc.camera_spec,
            },
            "environment": {
                **self.environment.model_dump(),
                "scene_materials": ", ".join(sc.materials) if sc.materials else self.environment.materials,
            },
            "main_subject": ", ".join(sc.subjects) if sc.subjects else (self.main_subject or sc.title),
            "timeline": [
                {
                    "time": b.time,
                    "scene": b.scene,
                    "action": b.action,
                    "camera": b.camera,
                    **({"sound": b.sound} if b.sound else {}),
                }
                for b in sc.beats
            ],
            "lighting": {
                **self.lighting.model_dump(),
                "color_palette": sc.colors or self.lighting.color_palette,
            },
            "motion": self.motion.model_dump(),
            "final_frame": {
                "subject": sc.end_frame,
                "composition": "locked-off hero composition with balanced negative space",
                "hold": "brief cinematic hold",
            },
        }
        if self.elements:
            doc["reference_elements"] = [
                f"{e.tag} ({e.role}: {e.label})" for e in self.elements
            ]
        if self.audio_mode != "vo_plus_ducked_score" and self.audio_direction:
            doc["audio"] = self.audio_direction
        if sc.continues_from_prev and idx > 0:
            prev = self.scenes[idx - 1]
            doc["continuity_from_previous_scene"] = (
                f"Continues pixel-exact from Scene {idx}'s final frame ({prev.end_frame}). "
                "Match exact lighting, surface grain, scale, and camera momentum with zero cut."
            )
        elif idx > 0 and sc.kind == "ai_sequence":
            prior_ai = [s for s in self.scenes[:idx] if s.kind == "ai_sequence"]
            if prior_ai:
                anchor_sc = prior_ai[0]
                doc["world_continuity_lock"] = (
                    f"Same 60-second film world as Act {anchor_sc.index + 1}: preserve exact hero identity "
                    f"({self.main_subject or ', '.join(anchor_sc.subjects[:2])}), architectural surfaces, "
                    f"lighting grade ({self.lighting.style}), and color palette ({', '.join(sc.colors[:3])})."
                )
        if self.negative_constraints:
            doc["avoid"] = self.negative_constraints
        return doc

    def compile_scene_cinematic_prose(self, scene_idx: int, *, max_chars: int = 2000) -> str:
        """Compile natural-language cinematography prose tailored for Veo, Hailuo, Seedance, Wan & Kling 2.5."""
        idx = max(0, min(scene_idx, len(self.scenes) - 1))
        sc = self.scenes[idx]
        subjects_txt = ", ".join(sc.subjects) if sc.subjects else (self.main_subject or sc.title)
        mats_txt = ", ".join(sc.materials) if sc.materials else self.environment.materials
        beat_actions = [
            f"{b.scene.rstrip('.')}: {b.action.rstrip('.')}"
            if b.scene and b.action
            else (b.action or b.scene).rstrip(".")
            for b in sc.beats[:3]
            if (b.scene or b.action)
        ]
        motion_story = ". ".join(x for x in beat_actions if x)
        effects_txt = ", ".join(self.lighting.effects[:3])
        palette_txt = ", ".join((sc.colors or self.lighting.color_palette)[:3])

        continuity_prefix = ""
        if sc.continues_from_prev and idx > 0:
            prev = self.scenes[idx - 1]
            continuity_prefix = (
                f"Seamless continuous shot from previous frame ({prev.end_frame.rstrip('.')}), "
                "maintaining identical lighting, scale, and camera momentum. "
            )
        elif idx > 0:
            prior_ai = [s for s in self.scenes[:idx] if s.kind == "ai_sequence"]
            if prior_ai:
                continuity_prefix = (
                    f"Same consistent visual world and hero subject ({self.main_subject or subjects_txt}) "
                    "established in the opening scene, matching exact color grade and lens character. "
                )

        el_hint = ""
        if self.elements:
            el_labels = ", ".join(f"{e.label} ({e.role})" for e in self.elements[:2])
            el_hint = f"Hero visual anchor: {el_labels}. "

        segments = [
            f"{continuity_prefix}{self.style}.",
            f"{el_hint}Hero subject: {subjects_txt}, crafted with tactile {mats_txt} in {self.environment.world} ({self.environment.surface}).",
            f"Action & motion: {motion_story}." if motion_story else "",
            f"Final composition settles into: {sc.end_frame.rstrip('.')}." if sc.end_frame else "",
            f"Cinematography: {sc.camera_spec or self.camera.style}, {self.camera.lens_mm}mm cinema prime, {self.camera.depth_of_field} depth of field, {self.motion.feel}.",
            f"Lighting & color grade: {self.lighting.style}"
            + (f" with {effects_txt}" if effects_txt else "")
            + (f", palette {palette_txt}." if palette_txt else "."),
            f"Subtle natural sound atmosphere: {self.audio_direction.rstrip('.')}."
            if sc.enable_native_audio and self.audio_direction
            else "",
            "Ultra-photorealistic 4K cinema grade, coherent physical motion, strictly zero readable text, signs, UI labels, or watermarks.",
        ]
        prose = " ".join(s.strip() for s in segments if s and s.strip())
        prose = re.sub(r"\s+", " ", prose).strip()
        if len(prose) <= max_chars:
            return prose
        return prose[: max_chars - 1].rstrip() + "."

    def compile_scene_for_spec(self, scene_idx: int, spec: VideoModelSpec) -> str:
        """Compile prompt tailored to `spec.prompt_style` ('cinematic_prose' vs 'structured_json')."""
        if getattr(spec, "prompt_style", "structured_json") == "cinematic_prose":
            return self.compile_scene_cinematic_prose(scene_idx, max_chars=spec.max_prompt_chars)
        return self.compile_scene_model_prompt(scene_idx, max_chars=spec.max_prompt_chars)

    def compile_scene_model_prompt(self, scene_idx: int, *, max_chars: int = 2400) -> str:
        """Compile one Scene into valid compact JSON strictly <= max_chars."""
        doc = self.to_scene_prompt_dict(scene_idx)
        compact = json.dumps(doc, ensure_ascii=False, separators=(",", ":"))
        if len(compact) <= max_chars:
            return compact

        trimmed = copy.deepcopy(doc)
        for drop_key in ("avoid", "audio", "reference_elements", "world_continuity_lock"):
            trimmed.pop(drop_key, None)
            compact = json.dumps(trimmed, ensure_ascii=False, separators=(",", ":"))
            if len(compact) <= max_chars:
                return compact

        if isinstance(trimmed.get("lighting"), dict):
            trimmed["lighting"].pop("color_palette", None)
        if isinstance(trimmed.get("environment"), dict):
            trimmed["environment"].pop("atmosphere", None)
        if isinstance(trimmed.get("timeline"), list):
            for b in trimmed["timeline"]:
                if isinstance(b, dict):
                    b.pop("sound", None)
        compact = json.dumps(trimmed, ensure_ascii=False, separators=(",", ":"))
        if len(compact) <= max_chars:
            return compact

        tl = trimmed.get("timeline")
        if isinstance(tl, list) and tl:
            for cap in (150, 115, 85, 65):
                for b in tl:
                    if isinstance(b, dict):
                        for k in ("scene", "action", "camera"):
                            val = b.get(k)
                            if isinstance(val, str) and len(val) > cap:
                                b[k] = val[: cap - 1].rstrip() + "…"
                compact = json.dumps(trimmed, ensure_ascii=False, separators=(",", ":"))
                if len(compact) <= max_chars:
                    return compact

        for sec in ("environment", "camera", "lighting", "motion", "final_frame"):
            obj = trimmed.get(sec)
            if isinstance(obj, dict):
                for k, v in list(obj.items()):
                    if isinstance(v, str) and len(v) > 65:
                        obj[k] = v[:64].rstrip() + "…"
                    elif isinstance(v, list) and len(v) > 3:
                        obj[k] = v[:3]
        compact = json.dumps(trimmed, ensure_ascii=False, separators=(",", ":"))
        return compact if len(compact) <= max_chars else compact[:max_chars]

    def compile_scene_keyframe_prompt(
        self,
        scene_idx: int,
        *,
        is_end: bool = False,
        tweak_note: str | None = None,
    ) -> str:
        """Compile a static photographic Flux Schnell / Kontext prompt for Scene Start or End."""
        sc = self.scenes[max(0, min(scene_idx, len(self.scenes) - 1))]
        if is_end:
            base = sc.end_frame or sc.title
        elif sc.beats:
            base = sc.beats[0].scene
        else:
            base = sc.title
        subjects_txt = ", ".join(sc.subjects) if sc.subjects else self.main_subject
        mats_txt = ", ".join(sc.materials) if sc.materials else self.environment.materials
        effects = ", ".join(self.lighting.effects[:4])
        parts = [
            f"{self.style}, {base.rstrip('.')}",
            f"Consistent hero identity: {self.main_subject}" if self.main_subject else "",
            f"Subjects: {subjects_txt}" if subjects_txt else "",
            f"Materials & surface: {mats_txt} on {self.environment.surface}",
            f"Camera: {sc.camera_spec or self.camera.style}, {self.camera.lens_mm}mm lens, {self.camera.depth_of_field} depth of field",
            f"Lighting: {self.lighting.style} with {effects}",
            f"Director tweak: {tweak_note.strip()}" if tweak_note and tweak_note.strip() else "",
            "Ultra-photorealistic 4K commercial cinema still frame, zero readable text, no watermarks",
        ]
        return ". ".join(p.strip().rstrip(".") for p in parts if p.strip()) + "."


def validate_stitch_project(
    project: StitchProject,
    model_spec: VideoModelSpec,
) -> dict[str, Any]:
    """Run instant deterministic Script Gate checks across all Scenes before rendering."""
    scene_reports: list[dict[str, Any]] = []
    total_warnings = 0

    for idx, sc in enumerate(project.scenes):
        issues: list[dict[str, str]] = []
        compiled = project.compile_scene_for_spec(idx, model_spec)
        raw_len = len(json.dumps(project.to_scene_prompt_dict(idx), ensure_ascii=False, separators=(",", ":")))

        # Voiceover speakability check across all scene kinds (target <= 3.1 words/sec)
        vo_words = [w for w in sc.vo_line.strip().split() if w]
        if vo_words and len(vo_words) / float(max(3, sc.duration_sec)) > 3.1:
            wps = round(len(vo_words) / float(max(3, sc.duration_sec)), 1)
            issues.append({
                "code": "vo_too_fast",
                "message": f"Narration rate ({wps} words/sec) exceeds natural broadcast pace (≤ 3.0 wps) — trim copy or extend scene duration.",
            })

        if sc.kind == "ai_sequence":
            if not sc.subjects:
                issues.append({"code": "missing_subjects", "message": "Add at least 1 character or subject so the video model knows what to focus on."})
            if not sc.materials:
                issues.append({"code": "missing_materials", "message": "Add concrete physical materials/textures (e.g. oak wood, brushed aluminum, glass)."})
            if not sc.camera_spec.strip():
                issues.append({"code": "missing_camera", "message": "Specify a camera position or movement for this scene."})
            if not sc.beats:
                issues.append({"code": "missing_beats", "message": "Scene needs at least 2 timed beats."})

            blob = " ".join([sc.end_frame, *(f"{b.scene} {b.action}" for b in sc.beats)])
            m = _TEXT_IN_SCENE_RE.search(blob)
            if m:
                issues.append({
                    "code": "garbled_text_risk",
                    "message": f"Anti-Garbled-Text Guard: mentions '{m.group(0)}'. AI video models garble readable words — replace with pure visual detail.",
                })

            if (
                model_spec.id != "auto"
                and not model_spec.omit_duration
                and not sc.video_model_id
                and model_spec.durations_sec
                and sc.duration_sec not in model_spec.durations_sec
            ):
                issues.append({
                    "code": "duration_mismatch",
                    "message": f"Scene duration {sc.duration_sec}s will snap to closest {model_spec.label} duration ({model_spec.durations_sec}).",
                })

            if idx > 0 and sc.continues_from_prev and sc.beats:
                first_txt = f"{sc.beats[0].scene} {sc.beats[0].action}".lower()
                if not any(w in first_txt for w in ("continue", "same", "seamless", "from ", "previous", "exact")):
                    issues.append({
                        "code": "continuity_anchor",
                        "message": f"Beat 1 should explicitly continue from Scene {idx}'s locked end frame for a seamless pixel-chained handoff.",
                    })

            if (
                idx >= 2
                and len(project.scenes) >= 4
                and project.scenes[idx - 1].kind == "ai_sequence"
                and project.scenes[idx - 2].kind == "ai_sequence"
                and sc.camera_spec.strip().lower()
                == project.scenes[idx - 1].camera_spec.strip().lower()
                == project.scenes[idx - 2].camera_spec.strip().lower()
            ):
                issues.append({
                    "code": "monotonous_camera",
                    "message": "3 consecutive AI scenes share identical camera framing — vary camera movement to keep 60s retention high.",
                })

        total_warnings += len(issues)
        scene_reports.append({
            "scene_id": sc.scene_id,
            "index": idx,
            "valid": len(issues) == 0,
            "issues": issues,
            "compiled_chars": len(compiled),
            "raw_chars": raw_len,
            "max_chars": model_spec.max_prompt_chars,
        })

    return {
        "valid": total_warnings == 0,
        "total_warnings": total_warnings,
        "scenes": scene_reports,
    }


_CAMERA_VARIATION_CYCLE: list[str] = [
    "90mm macro tilt-shift · extreme close-up slowly pushing in",
    "65mm cinema glide · smooth continuous lateral dolly",
    "85mm sculptural lens · 180-degree continuous orbital arc",
    "50mm anamorphic prime · sweeping continuous pull-back reveal",
    "85mm studio cinema · poised hero lock-off with subtle breathing focus",
]


def apply_deterministic_script_autofix(
    project: StitchProject,
    model_spec: VideoModelSpec,
) -> StitchProject:
    """Fix common Script Gate warnings cleanly in 1 click without losing user intent."""
    for idx, sc in enumerate(project.scenes):
        # Trim rushed narration to <= 2.8 words/sec across any scene kind
        vo_words = [w for w in sc.vo_line.strip().split() if w]
        max_words = max(7, int(max(3, sc.duration_sec) * 2.8))
        if len(vo_words) > max_words and len(vo_words) / float(max(3, sc.duration_sec)) > 3.1:
            trimmed_vo = " ".join(vo_words[:max_words]).rstrip(",;:-")
            if not trimmed_vo.endswith((".", "!", "?")):
                trimmed_vo += "."
            sc.vo_line = trimmed_vo

        if sc.kind != "ai_sequence":
            continue
        if not sc.subjects:
            sc.subjects = [project.main_subject or sc.title]
        if not sc.materials:
            sc.materials = [m.strip() for m in project.environment.materials.split(",")[:3] if m.strip()] or ["aged oak wood", "optical glass", "brushed metal"]
        if not sc.camera_spec.strip():
            sc.camera_spec = f"{project.camera.lens_mm}mm {project.camera.style} · {project.camera.movement}"

        # Clean readable-text risks
        sc.end_frame = _TEXT_IN_SCENE_RE.sub("warm glowing studio caustics", sc.end_frame)
        for b in sc.beats:
            b.scene = _TEXT_IN_SCENE_RE.sub("warm glowing light reflections", b.scene)
            b.action = _TEXT_IN_SCENE_RE.sub("subtle luminous highlights", b.action)

        # Snap duration cleanly if concrete model doesn't support it
        if (
            model_spec.id != "auto"
            and not model_spec.omit_duration
            and not sc.video_model_id
            and model_spec.durations_sec
            and sc.duration_sec not in model_spec.durations_sec
        ):
            sc.duration_sec = model_spec.snap_duration(sc.duration_sec)

        # Diversify monotonous 3-in-a-row camera moves
        if (
            idx >= 2
            and len(project.scenes) >= 4
            and project.scenes[idx - 1].kind == "ai_sequence"
            and project.scenes[idx - 2].kind == "ai_sequence"
            and sc.camera_spec.strip().lower()
            == project.scenes[idx - 1].camera_spec.strip().lower()
            == project.scenes[idx - 2].camera_spec.strip().lower()
        ):
            sc.camera_spec = _CAMERA_VARIATION_CYCLE[idx % len(_CAMERA_VARIATION_CYCLE)]

        # Anchor continuity on Scene 2+
        if idx > 0 and sc.continues_from_prev and sc.beats:
            first_txt = f"{sc.beats[0].scene} {sc.beats[0].action}".lower()
            if not any(w in first_txt for w in ("continue", "same", "seamless", "from ", "previous", "exact")):
                prev_end = project.scenes[idx - 1].end_frame
                sc.beats[0].scene = f"Continues seamlessly from Scene {idx}'s locked frame ({prev_end[:80]}). {sc.beats[0].scene}"

    project.status = "validated"
    project.status_message = "All scenes auto-sharpened and verified by Script Gate."
    return project


STITCH_GALLERY_RECIPES: list[dict[str, Any]] = [
    {
        "id": "miniature_jar_world",
        "category": "diorama",
        "badge": "Viral Diorama · 10s–30s",
        "title": "Miniature World in a Jar",
        "lens_tag": "90mm Macro",
        "idea": (
            "A giant photorealistic human hand drops glowing liquid gold onto a frozen gray miniature city, "
            "awakening tiny streets, folding the map into a glowing mountain village, and pulling back "
            "to seal the living world inside a glass jar."
        ),
        "image_url": "/api/stitch/gallery/jar_open.jpg",
        "end_image_url": "/api/stitch/gallery/jar_end.jpg",
        "world_preset": "miniature_diorama",
        "camera_preset": "macro_tilt_shift",
        "lighting_preset": "volumetric_gold",
        "motion_preset": "asmr_tactile",
    },
    {
        "id": "anodized_hardware_silence",
        "category": "hardware",
        "badge": "Hardware Launch · 15s–30s",
        "title": "Anodized Silence on Travertine",
        "lens_tag": "85mm Studio",
        "idea": (
            "Matte-black anodized aluminum flagship headphones rest on sunlit Roman travertine as golden "
            "light caustics silence chaotic urban energy into serene acoustic clarity."
        ),
        "image_url": "/api/stitch/gallery/headphones.jpg",
        "end_image_url": "/api/stitch/gallery/headphones.jpg",
        "world_preset": "architectural_luxury",
        "camera_preset": "continuous_orbit",
        "lighting_preset": "crisp_daylight",
        "motion_preset": "epic_slowmo",
    },
    {
        "id": "cotton_origami_city",
        "category": "craft",
        "badge": "Paper Craft · 15s–45s",
        "title": "Living Cotton Origami City",
        "lens_tag": "Tilt-Shift",
        "idea": (
            "A flat 300gsm cotton drafting sheet on a warm sunlit walnut table folds autonomously like "
            "living origami into an intricate architectural miniature city glowing with warm gold windows."
        ),
        "image_url": "/api/stitch/gallery/origami.jpg",
        "end_image_url": "/api/stitch/gallery/origami.jpg",
        "world_preset": "paper_origami",
        "camera_preset": "overhead_topdown",
        "lighting_preset": "golden_hour",
        "motion_preset": "seamless_morph",
    },
    {
        "id": "golden_crema_ritual",
        "category": "asmr",
        "badge": "ASMR Macro · 15s",
        "title": "Golden Crema 120fps Pour",
        "lens_tag": "120fps Macro",
        "idea": (
            "High-speed 120fps macro of velvety golden espresso crema pouring into crystal borosilicate glass "
            "on warm pale oak in golden hour sunlight, suspended droplets catching amber reflections."
        ),
        "image_url": "/api/stitch/gallery/espresso.jpg",
        "end_image_url": "/api/stitch/gallery/espresso.jpg",
        "world_preset": "architectural_luxury",
        "camera_preset": "pullback_reveal",
        "lighting_preset": "golden_hour",
        "motion_preset": "asmr_tactile",
    },
    {
        "id": "liquid_gold_awakening",
        "category": "diorama",
        "badge": "World Morph · 30s",
        "title": "Liquid Gold Street Awakening",
        "lens_tag": "Volumetric",
        "idea": (
            "Molten liquid gold pours down a tiny handcrafted balsa-wood street, igniting warm amber windows "
            "and miniature brass lanterns across an expanding living tabletop world."
        ),
        "image_url": "/api/stitch/gallery/jar_mid.jpg",
        "end_image_url": "/api/stitch/gallery/jar_end.jpg",
        "world_preset": "miniature_diorama",
        "camera_preset": "fpv_glide",
        "lighting_preset": "volumetric_gold",
        "motion_preset": "seamless_morph",
    },
    {
        "id": "sealed_constellation",
        "category": "hardware",
        "badge": "Hero Reveal · 15s–60s",
        "title": "Sealed Constellation Reveal",
        "lens_tag": "4K Ready",
        "idea": (
            "Intimate warm volumetric rays dance through crystal borosilicate glass and brushed brass, "
            "pulling back smoothly to lock onto an unforgettable glowing hero centerpiece."
        ),
        "image_url": "/api/stitch/gallery/jar_end.jpg",
        "end_image_url": "/api/stitch/gallery/jar_end.jpg",
        "world_preset": "surreal_glass",
        "camera_preset": "pullback_reveal",
        "lighting_preset": "volumetric_gold",
        "motion_preset": "epic_slowmo",
    },
    {
        "id": "anthropic_editorial",
        "category": "launch",
        "badge": "AI Launch · Hybrid",
        "title": "Anthropic Warm Keynote",
        "lens_tag": "Editorial UI",
        "idea": (
            "Warm alabaster editorial typography introduces your product, transitioning smoothly "
            "into a crisp live prompt reasoning demo and a calm sunlit studio metaphor."
        ),
        "image_url": "/api/stitch/gallery/launch_anthropic.jpg",
        "end_image_url": "/api/stitch/gallery/launch_anthropic.jpg",
        "world_preset": "anthropic_editorial",
        "camera_preset": "macro_tilt_shift",
        "lighting_preset": "crisp_daylight",
        "motion_preset": "seamless_morph",
        "hybrid_plan": ["motion_graphic:kinetic_headline", "motion_graphic:prompt_ui_demo", "ai_sequence"],
    },
    {
        "id": "openai_product_drop",
        "category": "launch",
        "badge": "Product Drop · Hybrid",
        "title": "OpenAI Minimal Drop",
        "lens_tag": "Live UI + 35mm",
        "idea": (
            "High-contrast monochrome glassmorphic prompt box streams a live answer, cuts to an "
            "authentic human moment, and locks onto a crisp bento grid of flagship capabilities."
        ),
        "image_url": "/api/stitch/gallery/launch_openai.jpg",
        "end_image_url": "/api/stitch/gallery/launch_openai.jpg",
        "world_preset": "openai_product_drop",
        "camera_preset": "continuous_orbit",
        "lighting_preset": "crisp_daylight",
        "motion_preset": "epic_slowmo",
        "hybrid_plan": ["motion_graphic:prompt_ui_demo", "ai_sequence", "motion_graphic:bento_feature_grid"],
    },
    {
        "id": "gemini_multimodal",
        "category": "launch",
        "badge": "Multimodal · Benchmarks",
        "title": "Gemini Multimodal Launch",
        "lens_tag": "Prism + Charts",
        "idea": (
            "Deep cosmos navy prism glass opens into a rapid before-and-after multimodal split "
            "comparison and animated benchmark bars proving flagship speed and intelligence."
        ),
        "image_url": "/api/stitch/gallery/launch_gemini.jpg",
        "end_image_url": "/api/stitch/gallery/launch_gemini.jpg",
        "world_preset": "gemini_multimodal",
        "camera_preset": "fpv_glide",
        "lighting_preset": "volumetric_gold",
        "motion_preset": "seamless_morph",
        "hybrid_plan": ["ai_sequence", "motion_graphic:split_comparison", "motion_graphic:benchmark_bars"],
    },
    {
        "id": "founder_keynote_hybrid",
        "category": "launch",
        "badge": "Founder + Demo · Hybrid",
        "title": "Founder + Product Demo",
        "lens_tag": "A-Roll + UI",
        "idea": (
            "Upload your own founder talking-head or screen demo as Act 1, then weave in crisp "
            "product UI graphics, terminal code reveals, and cinematic AI B-roll cutaways."
        ),
        "image_url": "/api/stitch/gallery/launch_founder.jpg",
        "end_image_url": "/api/stitch/gallery/launch_founder.jpg",
        "world_preset": "founder_keynote_hybrid",
        "camera_preset": "pullback_reveal",
        "lighting_preset": "golden_hour",
        "motion_preset": "seamless_morph",
        "hybrid_plan": ["user_clip", "motion_graphic:terminal_code", "ai_sequence"],
    },
    {
        "id": "apple_titanium",
        "category": "hardware",
        "badge": "Keynote · Flagship",
        "title": "Apple Titanium Keynote",
        "lens_tag": "Obsidian Rim",
        "idea": (
            "Pitch-black cyclorama with razor-thin rim lighting reveals your flagship product, "
            "gliding into a 2.5D hardware showcase and bold centered keynote performance stats."
        ),
        "image_url": "/api/stitch/gallery/theme_apple.jpg",
        "end_image_url": "/api/stitch/gallery/theme_apple.jpg",
        "world_preset": "apple_titanium",
        "camera_preset": "continuous_orbit",
        "lighting_preset": "crisp_daylight",
        "motion_preset": "epic_slowmo",
        "hybrid_plan": ["ai_sequence", "motion_graphic:device_showcase_25d", "motion_graphic:bento_feature_grid"],
    },
    {
        "id": "linear_dark_craft",
        "category": "saas",
        "badge": "Dark SaaS · DevTools",
        "title": "Linear / Vercel Dark Craft",
        "lens_tag": "2.5D UI + CLI",
        "idea": (
            "Obsidian grid typography snaps into a 2.5D perspective software walkthrough with "
            "indigo edge glow and a live terminal SDK deployment reveal."
        ),
        "image_url": "/api/stitch/gallery/theme_linear.jpg",
        "end_image_url": "/api/stitch/gallery/theme_linear.jpg",
        "world_preset": "linear_dark_craft",
        "camera_preset": "macro_tilt_shift",
        "lighting_preset": "crisp_daylight",
        "motion_preset": "seamless_morph",
        "hybrid_plan": ["motion_graphic:kinetic_headline", "motion_graphic:device_showcase_25d", "motion_graphic:terminal_code"],
    },
    {
        "id": "stripe_living_flow",
        "category": "saas",
        "badge": "Fintech · Architecture",
        "title": "Stripe Living Architecture",
        "lens_tag": "Node Flow",
        "idea": (
            "Crisp light editorial typography introduces your platform, flowing into an animated "
            "living node-and-connector architecture diagram and rolling benchmark proof."
        ),
        "image_url": "/api/stitch/gallery/theme_stripe.jpg",
        "end_image_url": "/api/stitch/gallery/theme_stripe.jpg",
        "world_preset": "stripe_living_flow",
        "camera_preset": "pullback_reveal",
        "lighting_preset": "crisp_daylight",
        "motion_preset": "seamless_morph",
        "hybrid_plan": ["motion_graphic:kinetic_headline", "motion_graphic:node_flow_diagram", "motion_graphic:benchmark_bars"],
    },
    {
        "id": "notion_playful",
        "category": "saas",
        "badge": "Editorial · Productivity",
        "title": "Notion / Figma Editorial",
        "lens_tag": "Warm Paper UI",
        "idea": (
            "Warm stationery-paper editorial typography flows into a friendly live prompt tour "
            "and a crisp before-and-after workflow comparison."
        ),
        "image_url": "/api/stitch/gallery/theme_notion.jpg",
        "end_image_url": "/api/stitch/gallery/theme_notion.jpg",
        "world_preset": "notion_playful",
        "camera_preset": "overhead_topdown",
        "lighting_preset": "golden_hour",
        "motion_preset": "seamless_morph",
        "hybrid_plan": ["motion_graphic:kinetic_headline", "motion_graphic:prompt_ui_demo", "motion_graphic:split_comparison"],
    },
    {
        "id": "screen_studio_tour",
        "category": "saas",
        "badge": "Product Hunt · 2.5D",
        "title": "Screen-Studio Product Tour",
        "lens_tag": "2.5D Zoom",
        "idea": (
            "No-fluff product tour framing your live website and app inside a 2.5D tilted display "
            "with cursor-tracked camera zooms and a crisp capability bento grid."
        ),
        "image_url": "/api/stitch/gallery/theme_screenstudio.jpg",
        "end_image_url": "/api/stitch/gallery/theme_screenstudio.jpg",
        "world_preset": "screen_studio_tour",
        "camera_preset": "macro_tilt_shift",
        "lighting_preset": "crisp_daylight",
        "motion_preset": "seamless_morph",
        "hybrid_plan": ["motion_graphic:device_showcase_25d", "motion_graphic:prompt_ui_demo", "motion_graphic:bento_feature_grid"],
    },
    {
        "id": "airbnb_human_story",
        "category": "launch",
        "badge": "Brand Film · 35mm",
        "title": "Airbnb Warm Human Story",
        "lens_tag": "35mm + Proof",
        "idea": (
            "Sunlit 35mm human storytelling bridges naturally into your live product experience "
            "and closes with a warm customer quote and brand outro."
        ),
        "image_url": "/api/stitch/gallery/theme_airbnb.jpg",
        "end_image_url": "/api/stitch/gallery/theme_airbnb.jpg",
        "world_preset": "airbnb_human_story",
        "camera_preset": "pullback_reveal",
        "lighting_preset": "golden_hour",
        "motion_preset": "epic_slowmo",
        "hybrid_plan": ["ai_sequence", "motion_graphic:device_showcase_25d", "motion_graphic:customer_quote_proof"],
    },
    {
        "id": "viral_vertical_reel",
        "category": "social",
        "badge": "9:16 Social · Hook",
        "title": "9:16 Hook-First Growth Reel",
        "lens_tag": "9:16 Vertical",
        "default_aspect": "9:16",
        "idea": (
            "Scroll-stopping 1.5-second vertical kinetic hook slams into a mobile iPhone UI "
            "showcase and high-contrast benchmark proof built for Reels, TikTok, and Shorts."
        ),
        "image_url": "/api/stitch/gallery/theme_reel.jpg",
        "end_image_url": "/api/stitch/gallery/theme_reel.jpg",
        "world_preset": "viral_vertical_reel",
        "camera_preset": "fpv_glide",
        "lighting_preset": "crisp_daylight",
        "motion_preset": "asmr_tactile",
        "hybrid_plan": ["motion_graphic:kinetic_hook_reel", "motion_graphic:device_showcase_25d", "motion_graphic:benchmark_bars"],
    },
    {
        "id": "enterprise_case_study",
        "category": "saas",
        "badge": "Enterprise GTM · ROI",
        "title": "Customer Proof & ROI Cut",
        "lens_tag": "Quote + Chart",
        "idea": (
            "Authoritative executive metric hook leading into a verified customer testimonial "
            "quote card and animated benchmark bars proving measurable ROI."
        ),
        "image_url": "/api/stitch/gallery/theme_casestudy.jpg",
        "end_image_url": "/api/stitch/gallery/theme_casestudy.jpg",
        "world_preset": "enterprise_case_study",
        "camera_preset": "pullback_reveal",
        "lighting_preset": "crisp_daylight",
        "motion_preset": "epic_slowmo",
        "hybrid_plan": ["motion_graphic:kinetic_headline", "motion_graphic:customer_quote_proof", "motion_graphic:benchmark_bars"],
    },
    {
        "id": "a24_indie_35mm",
        "category": "cinema",
        "badge": "35mm Film · Emotional",
        "title": "A24 Indie 35mm Manifesto",
        "lens_tag": "Kodak 500T",
        "idea": (
            "Intimate Kodak Vision3 500T 35mm grain, warm tungsten halation, and poetic human "
            "moments unfolding in a single nostalgic cinematic story."
        ),
        "image_url": "/api/stitch/gallery/cinema_a24.jpg",
        "end_image_url": "/api/stitch/gallery/cinema_a24.jpg",
        "world_preset": "a24_indie_35mm",
        "camera_preset": "pullback_reveal",
        "lighting_preset": "golden_hour",
        "motion_preset": "epic_slowmo",
    },
    {
        "id": "imax_70mm_epic",
        "category": "cinema",
        "badge": "70mm IMAX · Trailer",
        "title": "IMAX 70mm Frontier Epic",
        "lens_tag": "65mm IMAX",
        "idea": (
            "Monumental Villeneuve-scale vistas, sweeping atmospheric mist, and slow colossal "
            "camera push-ins converging onto an iconic centerpiece."
        ),
        "image_url": "/api/stitch/gallery/cinema_imax.jpg",
        "end_image_url": "/api/stitch/gallery/cinema_imax.jpg",
        "world_preset": "imax_70mm_epic",
        "camera_preset": "fpv_glide",
        "lighting_preset": "volumetric_gold",
        "motion_preset": "epic_slowmo",
    },
    {
        "id": "luxury_maison_noir",
        "category": "cinema",
        "badge": "Luxury · 120fps Noir",
        "title": "Luxury Maison Chiaroscuro",
        "lens_tag": "120fps Macro",
        "idea": (
            "Parisian haute-couture chiaroscuro shadow drama with liquid silk, 24k gold "
            "specular ribbons, and hypnotic 120fps phantom slow-motion."
        ),
        "image_url": "/api/stitch/gallery/cinema_noir.jpg",
        "end_image_url": "/api/stitch/gallery/cinema_noir.jpg",
        "world_preset": "luxury_maison_noir",
        "camera_preset": "continuous_orbit",
        "lighting_preset": "volumetric_gold",
        "motion_preset": "epic_slowmo",
    },
    {
        "id": "blade_runner_anamorphic",
        "category": "cinema",
        "badge": "Anamorphic · Neo-Noir",
        "title": "Neon Rain Anamorphic",
        "lens_tag": "2.39:1 Flare",
        "idea": (
            "Rain-slicked nocturnal streets mirroring cyan and amber light with horizontal "
            "blue anamorphic lens flares and volumetric steam."
        ),
        "image_url": "/api/stitch/gallery/cinema_neon.jpg",
        "end_image_url": "/api/stitch/gallery/cinema_neon.jpg",
        "world_preset": "blade_runner_anamorphic",
        "camera_preset": "fpv_glide",
        "lighting_preset": "volumetric_gold",
        "motion_preset": "seamless_morph",
    },
    {
        "id": "wes_pastel_symmetry",
        "category": "cinema",
        "badge": "35mm · Symmetrical",
        "title": "Symmetrical Pastel Storybook",
        "lens_tag": "Dead-Center",
        "idea": (
            "Dead-center planimetric symmetry, curated warm pastel production design, and "
            "crisp mechanical dolly framing with whimsical charm."
        ),
        "image_url": "/api/stitch/gallery/cinema_pastel.jpg",
        "end_image_url": "/api/stitch/gallery/cinema_pastel.jpg",
        "world_preset": "wes_pastel_symmetry",
        "camera_preset": "overhead_topdown",
        "lighting_preset": "crisp_daylight",
        "motion_preset": "asmr_tactile",
    },
    {
        "id": "kinetic_sports_grit",
        "category": "cinema",
        "badge": "16mm · High-Energy",
        "title": "16mm Kinetic Grit & Sweat",
        "lens_tag": "16mm Film",
        "idea": (
            "Raw 16mm film grain, harsh stadium rim lighting cutting through cold mist, and "
            "explosive athletic momentum."
        ),
        "image_url": "/api/stitch/gallery/cinema_grit.jpg",
        "end_image_url": "/api/stitch/gallery/cinema_grit.jpg",
        "world_preset": "kinetic_sports_grit",
        "camera_preset": "fpv_glide",
        "lighting_preset": "crisp_daylight",
        "motion_preset": "epic_slowmo",
    },
    {
        "id": "painted_anime_cinema",
        "category": "cinema",
        "badge": "2D Cel · Gouache",
        "title": "Hand-Painted Anime Cinema",
        "lens_tag": "Painted Sky",
        "idea": (
            "Lush hand-painted gouache clouds, wind sweeping across sunlit emerald grass, "
            "and nostalgic theatrical 2D animation warmth."
        ),
        "image_url": "/api/stitch/gallery/cinema_anime.jpg",
        "end_image_url": "/api/stitch/gallery/cinema_anime.jpg",
        "world_preset": "painted_anime_cinema",
        "camera_preset": "pullback_reveal",
        "lighting_preset": "golden_hour",
        "motion_preset": "seamless_morph",
    },
    {
        "id": "natgeo_natural_world",
        "category": "cinema",
        "badge": "8K Earth · Telephoto",
        "title": "Natural World Documentary",
        "lens_tag": "Telephoto",
        "idea": (
            "Crystalline morning dew, alpine valley fog pierced by golden-hour sunbeams, and "
            "patient stabilized telephoto Earth cinematography."
        ),
        "image_url": "/api/stitch/gallery/cinema_natgeo.jpg",
        "end_image_url": "/api/stitch/gallery/cinema_natgeo.jpg",
        "world_preset": "natgeo_natural_world",
        "camera_preset": "macro_tilt_shift",
        "lighting_preset": "golden_hour",
        "motion_preset": "epic_slowmo",
    },
]


def get_stitch_catalog() -> dict[str, Any]:
    """Return curated visual gallery themes, motion templates, and world/camera/lighting/motion presets."""
    return {
        "gallery": STITCH_GALLERY_RECIPES,
        "motion_templates": MOTION_TEMPLATE_CATALOG,
        "worlds": list(WORLD_PRESETS.values()),
        "cameras": list(CAMERA_PRESETS.values()),
        "lightings": list(LIGHTING_PRESETS.values()),
        "motions": list(MOTION_PRESETS.values()),
    }


__all__ = [
    "STITCH_GALLERY_RECIPES",
    "AudioMode",
    "SceneKind",
    "SeamTransition",
    "StitchAspect",
    "StitchBeat",
    "StitchElement",
    "StitchProject",
    "StitchScene",
    "StitchTake",
    "apply_deterministic_script_autofix",
    "get_stitch_catalog",
    "validate_stitch_project",
]
