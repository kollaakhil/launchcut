"""Cinema Studio 3-Layer Master JSON Schemas (WorldVault -> CinemaScene -> CinemaShot).

Encodes the "You Bring the Taste, Launch Cut Brings the Film Crew" architecture:
  - Layer 1 (`WorldVault`): Immutable Character, Location (Dual-Wall A/B Set Plates),
    and Stateful Prop registries with 3-panel turnaround crops & Taste Signature.
  - Layer 2 (`CinemaScene`): `Therefore/But` causal story turns, tension curve (1-10),
    Pixar Color Script beats, and Scene Entry -> Exit world states.
  - Layer 3 (`CinemaShot`): Unambiguous spatial blocking (`foreground`/`midground`/`background`),
    textures, active prop states, camera & 180-deg eyeline vectors, `start_frame`,
    `motion_delta`, `end_frame`, 3-mode `anchor_strategy`, and Table-Read timed dialogue.
"""

from __future__ import annotations

import os
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from gtm_videogen.models.stitch_project import MotionGraphicSpec

CinemaAspect = Literal["16:9", "9:16", "1:1"]
WorkflowMode = Literal["guided", "autopilot"]

StoryArchetype = Literal[
    "auto",
    "mystery_reveal",
    "high_stakes_thriller",
    "intimate_character_arc",
    "founder_origin_story",
    "visual_poem",
]

CausalLink = Literal["opening_hook", "but", "therefore", "meanwhile"]
ShotRenderEngine = Literal["ai_cinema", "insert_ui_plate", "title_card"]

ShotScale = Literal[
    "extreme_wide",
    "wide",
    "medium_wide",
    "medium",
    "medium_close_up",
    "close_up",
    "extreme_close_up_insert",
    "over_the_shoulder",
]

AnchorStrategy = Literal[
    "start_and_end_locked",
    "chained_from_prev_to_end",
    "start_and_elements_forward",
]

DialogueDeliveryMode = Literal[
    "on_screen_speaking",
    "off_screen_reaction",
    "over_the_shoulder",
    "narrator_voiceover",
]

FacingWallZone = Literal[
    "A_north_primary",
    "B_south_reverse",
    "west_side",
    "east_side",
]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Taste-to-Craft Models & Persistent Director Taste Profile
# ---------------------------------------------------------------------------


class VibeDials(BaseModel):
    """Four human-language aesthetic dials (0.0 -> 1.0) replacing camera jargon."""

    model_config = ConfigDict(extra="ignore")

    scale: float = Field(
        default=0.45,
        ge=0.0,
        le=1.0,
        description="0.0 = Intimate & Handheld, 1.0 = Grand & Architectural",
    )
    pacing: float = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
        description="0.0 = Slow-Burn & Contemplative, 1.0 = Tight & Kinetic",
    )
    mood_contrast: float = Field(
        default=0.60,
        ge=0.0,
        le=1.0,
        description="0.0 = Warm & Sunlit, 1.0 = Moody & High-Contrast",
    )
    dialogue_density: float = Field(
        default=0.45,
        ge=0.0,
        le=1.0,
        description="0.0 = Pure Visual / Minimal Words, 1.0 = Dialogue & Character Driven",
    )


class CompiledCraftRules(BaseModel):
    """Technical cinematography, lighting, and editorial rules compiled from human taste."""

    model_config = ConfigDict(extra="ignore")

    lens_package: str = (
        "vintage 35mm, 50mm, and 85mm cinema primes, organic oval bokeh, subtle halation"
    )
    lighting_philosophy: str = (
        "motivated natural key light with shaped shadow falloff and tactile rim separation"
    )
    film_texture: str = "fine organic 35mm film grain, authentic skin pores, lived-in surface patina, zero CGI plastic sheen"
    coverage_grammar: str = "alternate architectural wide shots with tactile macro prop inserts and eye-level reaction close-ups"
    editorial_style: str = (
        "0.4s J/L-cut dialogue bridges, continuous room-tone atmosphere, restrained emotional score"
    )
    target_avg_shot_sec: float = Field(default=6.0, ge=3.5, le=10.5)
    max_wpm: int = Field(default=125, ge=50, le=175)


class TasteSignature(BaseModel):
    """Active aesthetic DNA for a Cinema Project."""

    model_config = ConfigDict(extra="ignore")

    mood_presets: list[str] = Field(default_factory=lambda: ["cinema_a24"])
    story_archetype: StoryArchetype = "auto"
    user_vibe_note: str = ""
    dials: VibeDials = Field(default_factory=VibeDials)
    compiled_craft_rules: CompiledCraftRules = Field(default_factory=CompiledCraftRules)


class TasteReactionEvent(BaseModel):
    """Human reaction chip or plain-English direction recorded during a project."""

    model_config = ConfigDict(extra="ignore")

    reaction_id: str = Field(default_factory=lambda: f"rx_{uuid4().hex[:8]}")
    target_type: Literal["treatment", "vault", "scene", "shot", "cut"] = "shot"
    target_id: str = ""
    chip_or_note: str = ""
    polarity: Literal["love", "adjust", "reject"] = "adjust"
    created_at: str = Field(default_factory=_utc_now)


class DirectorTasteProfile(BaseModel):
    """Persistent cross-project taste memory learning from hearted/adjusted frames & cuts."""

    model_config = ConfigDict(extra="ignore")

    actor_id: str = "default"
    preferred_mood_presets: dict[str, int] = Field(default_factory=dict)
    preferred_archetypes: dict[str, int] = Field(default_factory=dict)
    dial_ema: VibeDials = Field(default_factory=VibeDials)
    love_notes: list[str] = Field(default_factory=list)
    avoid_notes: list[str] = Field(
        default_factory=lambda: [
            "avoid cheesy commercial announcer clichés",
            "avoid on-the-nose exposition where physical action can show emotion",
        ]
    )
    reactions_count: int = 0
    updated_at: str = Field(default_factory=_utc_now)


# ---------------------------------------------------------------------------
# Layer 1: WorldVault (Immutable Entities, Multi-Angle Crops & A/B Set Walls)
# ---------------------------------------------------------------------------


class CharacterVoiceCasting(BaseModel):
    """Locked TTS voice binding so a character sounds identical across 20+ shots."""

    model_config = ConfigDict(extra="ignore")

    provider: str = "fal-ai/elevenlabs/tts/eleven-v3"
    voice_id: str = "v_helena_f_mature_neutral"
    voice_name: str = "Helena"
    delivery_baseline: str = "measured, grounded cinema realism"


class CharacterAsset(BaseModel):
    """Canonical character identity, wardrobe, voice, and 3-panel turnaround crops."""

    model_config = ConfigDict(extra="ignore")

    id: str = Field(..., min_length=2, description="Canonical ID e.g. 'CHAR_MARA'")
    element_tag: str = "@Element1"
    name: str = "Protagonist"
    role: Literal["protagonist", "deuteragonist", "antagonist", "supporting", "narrator"] = (
        "protagonist"
    )
    identity_lock: str = Field(
        ...,
        min_length=5,
        description="Exact age, facial structure, eyes, hair, skin micro-texture, marks",
    )
    wardrobe_lock: str = Field(
        ...,
        min_length=5,
        description="Exact garment silhouette, fabric weave, color, wear, and accessories",
    )
    posture_and_mannerisms: str = "grounded natural physical presence"
    voice_casting: CharacterVoiceCasting = Field(default_factory=CharacterVoiceCasting)

    uploaded_ref_path: str | None = None
    turnaround_sheet_path: str | None = None
    turnaround_sheet_url: str | None = None
    frontal_crop_path: str | None = None
    frontal_crop_url: str | None = None
    three_quarter_crop_path: str | None = None
    full_body_crop_path: str | None = None


class CardinalZones(BaseModel):
    """Architectural layout of a location so reverse angles never hallucinate a different room."""

    model_config = ConfigDict(extra="ignore")

    A_north_primary: str = Field(
        ...,
        min_length=5,
        description="Primary hero background wall/view when looking toward Zone A",
    )
    B_south_reverse: str = Field(
        ...,
        min_length=5,
        description="Opposite wall/view seen on 180-degree reverse-angle coverage shots",
    )
    west_side: str = ""
    east_side: str = ""


class LocationAsset(BaseModel):
    """Canonical set architecture, A/B wall plates, surface textures, and continuous room tone."""

    model_config = ConfigDict(extra="ignore")

    id: str = Field(..., min_length=2, description="Canonical ID e.g. 'LOC_OBSERVATORY'")
    name: str = "Primary Set"
    time_of_day_and_weather: str = "night, subtle rain"
    cardinal_zones: CardinalZones
    axis_180_rule: str = (
        "Camera stays on East/South-East side of line of action to preserve eyeline matching"
    )
    surface_textures: list[str] = Field(default_factory=list)
    ambient_particles: str = "subtle volumetric dust motes in directional light"
    room_tone_sfx_prompt: str = "quiet interior room air with subtle low ambient resonance"

    room_tone_audio_path: str | None = None
    primary_wall_plate_path: str | None = None
    primary_wall_plate_url: str | None = None
    reverse_wall_plate_path: str | None = None
    reverse_wall_plate_url: str | None = None


class PropAsset(BaseModel):
    """Stateful narrative or brand-hero prop with chronological state definitions."""

    model_config = ConfigDict(extra="ignore")

    id: str = Field(..., min_length=2, description="Canonical ID e.g. 'PROP_QUARTZ_CORE'")
    element_tag: str = ""
    name: str = "Hero Prop"
    is_brand_hero: bool = False
    geometry_and_scale: str = "palm-sized tactile hero object with precise physical proportions"
    surface_textures: str = "brushed metal, optical glass, authentic micro-imperfections"
    states: dict[str, str] = Field(
        default_factory=lambda: {"default": "intact resting state with clean tactile finish"}
    )
    default_state: str = "default"
    setup_scene_id: str | None = None
    payoff_scene_id: str | None = None

    uploaded_ref_path: str | None = None
    turnaround_sheet_path: str | None = None
    turnaround_sheet_url: str | None = None
    frontal_crop_path: str | None = None
    frontal_crop_url: str | None = None

    def describe_state(self, state_key: str | None = None) -> str:
        key = (state_key or self.default_state or "default").strip()
        if key in self.states:
            return self.states[key]
        if self.states:
            return next(iter(self.states.values()))
        return "authentic physical finish"


class ColorScriptBeat(BaseModel):
    """Pixar-style per-scene color & lighting progression across the film arc."""

    model_config = ConfigDict(extra="ignore")

    scene_id: str
    palette_hex: list[str] = Field(default_factory=lambda: ["#12161D", "#D98A2B", "#F2E6D0"])
    lighting_arc: str = "motivated directional cinema lighting"
    emotional_note: str = ""


class WorldVault(BaseModel):
    """Complete production design & casting vault shared across all scenes and shots."""

    model_config = ConfigDict(extra="ignore")

    vault_id: str = Field(default_factory=lambda: f"vault_{uuid4().hex[:8]}")
    taste_signature: TasteSignature = Field(default_factory=TasteSignature)
    characters: list[CharacterAsset] = Field(default_factory=list)
    locations: list[LocationAsset] = Field(default_factory=list)
    props: list[PropAsset] = Field(default_factory=list)
    color_script: list[ColorScriptBeat] = Field(default_factory=list)
    score_direction: str = "Restrained cinematic chamber score: felt piano and low warm cello pulse rising toward climax"
    negative_constraints: list[str] = Field(
        default_factory=lambda: [
            "strictly zero readable text, signs, UI labels, or watermarks in AI cinema frames",
            "no plastic CGI sheen, warped hands, or sudden lighting flicker",
            "no unmotivated jump cuts or broken eyelines",
        ]
    )

    def get_character(self, char_id: str | None) -> CharacterAsset | None:
        if not char_id:
            return None
        clean = char_id.strip()
        for ch in self.characters:
            if ch.id == clean or ch.name.lower() == clean.lower():
                return ch
        return None

    def get_location(self, loc_id: str | None) -> LocationAsset | None:
        if not loc_id:
            return None
        clean = loc_id.strip()
        for loc in self.locations:
            if loc.id == clean or loc.name.lower() == clean.lower():
                return loc
        return self.locations[0] if self.locations else None

    def get_prop(self, prop_id: str | None) -> PropAsset | None:
        if not prop_id:
            return None
        clean = prop_id.strip()
        for pr in self.props:
            if pr.id == clean or pr.name.lower() == clean.lower():
                return pr
        return None


# ---------------------------------------------------------------------------
# Layer 2 & 3: CinemaScene & CinemaShot Instruction Graph
# ---------------------------------------------------------------------------


class SceneWorldState(BaseModel):
    """Explicit entry/exit world state linking Scene N cleanly into Scene N+1."""

    model_config = ConfigDict(extra="ignore")

    characters_present: list[str] = Field(default_factory=list)
    character_positions: dict[str, str] = Field(default_factory=dict)
    prop_states: dict[str, str] = Field(default_factory=dict)


class ShotCameraSpec(BaseModel):
    """Per-shot optical, framing, movement, wall-direction, and eyeline lock."""

    model_config = ConfigDict(extra="ignore")

    facing_wall_zone: FacingWallZone = "A_north_primary"
    lens_mm: int = Field(default=50, ge=14, le=200)
    t_stop: str = "T2.0 shallow depth of field"
    angle: str = "eye-level three-quarter framing"
    movement: str = "slow 10cm dolly push-in"
    eyeline_vector: str = "looking_screen_right"


class SpatialZonesSpec(BaseModel):
    """Explicit 3-depth staging so diffusion models know exact object coordinates."""

    model_config = ConfigDict(extra="ignore")

    foreground: str = ""
    midground: str = ""
    background: str = ""


class ShotPropInstance(BaseModel):
    """Active prop placement, state, and physical interaction inside a single shot."""

    model_config = ConfigDict(extra="ignore")

    prop_id: str
    state: str = "default"
    screen_zone: str = "midground"
    interaction: str = ""


class ShotLightingSpec(BaseModel):
    """Per-shot motivated key, fill/rim, and active hex color palette."""

    model_config = ConfigDict(extra="ignore")

    key: str = "motivated directional cinema key light"
    fill_and_rim: str = "subtle cool rim separation and natural shadow falloff"
    palette_hex: list[str] = Field(default_factory=lambda: ["#12161D", "#D98A2B", "#F2E6D0"])


class FrameStateSpec(BaseModel):
    """Static photographic composition contract for either `start_frame` or `end_frame`."""

    model_config = ConfigDict(extra="ignore")

    composition: str = Field(..., min_length=4)
    character_pose: str = ""
    focal_plane: str = "sharp focus on primary subject with natural optical bokeh"
    prop_states: dict[str, str] = Field(default_factory=dict)


class MotionDeltaBeat(BaseModel):
    """Timed physical choreography bridging `start_frame` -> `end_frame`."""

    model_config = ConfigDict(extra="ignore")

    window: str = "0.0-3.0s"
    action: str = Field(..., min_length=3)
    camera_delta: str = ""


class DialogueCueSpec(BaseModel):
    """Character-bound dialogue line timed via the pre-shoot Table Read pass."""

    model_config = ConfigDict(extra="ignore")

    speaker_id: str = "NARRATOR"
    line: str = Field(..., min_length=1)
    delivery_mode: DialogueDeliveryMode = "off_screen_reaction"
    emotion: str = "grounded, understated cinema realism"
    pre_pause_sec: float = Field(default=0.5, ge=0.0, le=5.0)
    synthesized_audio_path: str | None = None
    actual_speech_sec: float | None = None


class ShotSoundStems(BaseModel):
    """Per-shot acoustic foley prompt and Walter Murch J-cut / L-cut audio overlaps."""

    model_config = ConfigDict(extra="ignore")

    foley_prompt: str = ""
    foley_audio_path: str | None = None
    j_cut_lead_sec: float = Field(default=0.0, ge=0.0, le=1.5)
    l_cut_trail_sec: float = Field(default=0.0, ge=0.0, le=1.5)


class CinemaTake(BaseModel):
    """Versioned rendered video take (with optional close-up lip-sync pass)."""

    model_config = ConfigDict(extra="ignore")

    take_id: int = Field(..., ge=1)
    engine_id: str
    engine_label: str
    clip_path: str
    lipsynced: bool = False
    duration_sec: float = 6.0
    usd: float = 0.0
    anchored_start: bool = False
    anchored_end: bool = False
    rendered_end_frame_path: str | None = None
    created_at: str = Field(default_factory=_utc_now)


class CinemaShot(BaseModel):
    """Complete self-contained JSON instruction unit for one shot in a CinemaScene."""

    model_config = ConfigDict(extra="ignore")

    shot_id: str
    scene_id: str
    index_in_scene: int = 0
    duration_sec: int = Field(default=6, ge=3, le=15)
    render_engine: ShotRenderEngine = "ai_cinema"
    shot_scale: ShotScale = "medium"
    coverage_role: str = "coverage"
    anchor_strategy: AnchorStrategy = "start_and_end_locked"
    video_model_id: str | None = None

    characters_in_shot: list[str] = Field(default_factory=list)
    camera: ShotCameraSpec = Field(default_factory=ShotCameraSpec)
    spatial_zones: SpatialZonesSpec = Field(default_factory=SpatialZonesSpec)
    textures_and_atmosphere: list[str] = Field(default_factory=list)
    props_in_shot: list[ShotPropInstance] = Field(default_factory=list)
    lighting: ShotLightingSpec = Field(default_factory=ShotLightingSpec)

    start_frame: FrameStateSpec
    motion_delta: list[MotionDeltaBeat] = Field(default_factory=list)
    end_frame: FrameStateSpec

    dialogue: DialogueCueSpec | None = None
    sound_stems: ShotSoundStems = Field(default_factory=ShotSoundStems)

    # Optional crisp procedural plates for software/brand UI inserts or cinema title cards
    ui_plate_spec: MotionGraphicSpec | None = None
    title_card_headline: str | None = None
    title_card_subline: str | None = None

    # Conditioning & Storyboard Keyframe Artifacts
    ref_strip_path: str | None = None
    start_keyframe_path: str | None = None
    start_keyframe_url: str | None = None
    end_keyframe_path: str | None = None
    end_keyframe_url: str | None = None
    continuity_qa_score: float | None = None
    continuity_qa_note: str = ""

    # Non-Linear Take History & Taste Signals
    takes: list[CinemaTake] = Field(default_factory=list)
    winning_take_id: int | None = None
    render_status: str = "idle"  # idle | rendering | done | failed
    render_message: str = ""
    render_error: str | None = None
    hearted: bool = False

    def winning_take(self) -> CinemaTake | None:
        if not self.takes:
            return None
        if self.winning_take_id is not None:
            for tk in self.takes:
                if tk.take_id == self.winning_take_id:
                    return tk
        return self.takes[-1]


class CinemaScene(BaseModel):
    """One narrative scene in a single location with entry/exit states & causal story link."""

    model_config = ConfigDict(extra="ignore")

    scene_id: str
    scene_number: int = Field(..., ge=1)
    title: str = "Scene 1"
    slugline: str = "INT. STUDIO - NIGHT"
    location_id: str = "LOC_01"
    causal_link_from_prev: CausalLink = "opening_hook"
    dramatic_turn: str = ""
    emotional_shift: str = ""
    tension_level: int = Field(default=5, ge=1, le=10)
    target_duration_sec: int = Field(default=24, ge=6, le=90)
    color_palette_hex: list[str] = Field(default_factory=lambda: ["#12161D", "#D98A2B", "#F2E6D0"])
    lighting_arc: str = "motivated cinema key light"
    scene_entry_state: SceneWorldState = Field(default_factory=SceneWorldState)
    scene_exit_state: SceneWorldState = Field(default_factory=SceneWorldState)
    shots: list[CinemaShot] = Field(default_factory=list)

    def duration_sec(self) -> int:
        if not self.shots:
            return int(self.target_duration_sec)
        return sum(int(sh.duration_sec) for sh in self.shots)


class TreatmentPitch(BaseModel):
    """Visual + narrative Director Treatment pitched during Stage 1."""

    model_config = ConfigDict(extra="ignore")

    treatment_id: str
    title: str
    archetype: StoryArchetype = "intimate_character_arc"
    logline_hook: str
    story_arc_summary: str
    visual_and_sonic_vibe: str
    mood_preset: str = "cinema_a24"
    dials: VibeDials = Field(default_factory=VibeDials)
    palette_hex: list[str] = Field(default_factory=lambda: ["#12161D", "#D98A2B", "#F2E6D0"])
    sample_look_prompt: str = ""
    look_test_image_path: str | None = None
    look_test_image_url: str | None = None
    rubric_score: float = 9.0


class StoryRubricReport(BaseModel):
    """Cinema Story & Continuity Rubric report evaluating narrative propulsion & craft."""

    model_config = ConfigDict(extra="ignore")

    overall_score: float = 8.8
    causality_score: float = 9.0
    subtext_and_restraint_score: float = 8.8
    setup_payoff_score: float = 8.8
    visual_variety_score: float = 8.8
    passed: bool = True
    strengths: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class CinemaProject(BaseModel):
    """Complete 1-3 minute Cinema Studio film project."""

    model_config = ConfigDict(extra="ignore")

    project_id: str
    title: str = "Untitled Film"
    idea: str = ""
    source_url: str | None = None
    scraped_context: str = ""
    brand_name: str | None = None
    brand_tagline: str | None = None
    brand_logo_path: str | None = None
    uploaded_reference_paths: list[str] = Field(default_factory=list)

    aspect_ratio: CinemaAspect = "16:9"
    target_duration_sec: int = Field(default=90, ge=20, le=210)
    workflow_mode: WorkflowMode = "guided"
    video_model_id: str = "auto"
    llm_model_id: str = ""
    budget_cap_usd: float = Field(default=15.0, ge=0.0)
    enable_lipsync: bool = True
    enable_subtitles: bool = False
    cinema_letterbox: bool = True
    film_grain: bool = True

    treatments: list[TreatmentPitch] = Field(default_factory=list)
    selected_treatment_id: str | None = None
    vault: WorldVault = Field(default_factory=WorldVault)
    scenes: list[CinemaScene] = Field(default_factory=list)
    story_rubric: StoryRubricReport | None = None

    leica_reel_path: str | None = None
    master_score_path: str | None = None
    master_video_path: str | None = None
    qa_report: dict[str, Any] | None = None
    reaction_history: list[TasteReactionEvent] = Field(default_factory=list)

    status: str = "treatments_ready"
    status_message: str = "Choose a Director Treatment or build your Master Script."
    progress_pct: int = 0
    progress_stage: str = ""
    error: str | None = None
    estimated_shoot_usd: float = 0.0
    actual_usd: float = 0.0
    created_at: str = Field(default_factory=_utc_now)
    updated_at: str = Field(default_factory=_utc_now)

    def all_shots(self) -> list[CinemaShot]:
        return [sh for sc in self.scenes for sh in sc.shots]

    def total_duration_sec(self) -> int:
        shots = self.all_shots()
        if not shots:
            return int(self.target_duration_sec)
        return sum(int(sh.duration_sec) for sh in shots)

    def find_shot(self, shot_id: str) -> tuple[CinemaScene, CinemaShot, int] | None:
        flat_idx = 0
        for sc in self.scenes:
            for sh in sc.shots:
                if sh.shot_id == shot_id:
                    return sc, sh, flat_idx
                flat_idx += 1
        return None


# ---------------------------------------------------------------------------
# Atomic Filesystem Store (`<data_dir>/cinema/<project_id>/project.json`)
# ---------------------------------------------------------------------------


class CinemaProjectStore:
    """Atomic filesystem persistence for CinemaProject and media subdirectories."""

    def __init__(self, root_dir: Path | str) -> None:
        self.root_dir = Path(root_dir).resolve()
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def project_dir(self, project_id: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9_-]", "", project_id)
        if not safe:
            raise ValueError("Invalid cinema project_id")
        return self.root_dir / safe

    def media_dir(self, project_id: str, subdir: str = "media") -> Path:
        safe_sub = re.sub(r"[^A-Za-z0-9_-]", "", subdir) or "media"
        d = self.project_dir(project_id) / safe_sub
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save(self, project: CinemaProject) -> CinemaProject:
        project.updated_at = _utc_now()
        pdir = self.project_dir(project.project_id)
        pdir.mkdir(parents=True, exist_ok=True)
        target = pdir / "project.json"
        payload = project.model_dump_json(indent=2)
        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=pdir,
                prefix=".cinema_proj_",
                suffix=".tmp",
                delete=False,
            ) as tmp:
                tmp.write(payload)
                tmp.flush()
                os.fsync(tmp.fileno())
                tmp_path = Path(tmp.name)
            os.replace(tmp_path, target)
            tmp_path = None
        finally:
            if tmp_path is not None and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
        return project

    def load(self, project_id: str) -> CinemaProject | None:
        target = self.project_dir(project_id) / "project.json"
        if not target.is_file():
            return None
        return CinemaProject.model_validate_json(target.read_text(encoding="utf-8"))

    def list_all(self, limit: int = 30) -> list[CinemaProject]:
        items: list[CinemaProject] = []
        if not self.root_dir.exists():
            return items
        for child in self.root_dir.iterdir():
            pfile = child / "project.json"
            if pfile.is_file():
                try:
                    items.append(
                        CinemaProject.model_validate_json(pfile.read_text(encoding="utf-8"))
                    )
                except Exception:
                    continue
        items.sort(key=lambda p: p.updated_at, reverse=True)
        return items[:limit]


__all__ = [
    "AnchorStrategy",
    "CardinalZones",
    "CausalLink",
    "CharacterAsset",
    "CharacterVoiceCasting",
    "CinemaAspect",
    "CinemaProject",
    "CinemaProjectStore",
    "CinemaScene",
    "CinemaShot",
    "CinemaTake",
    "ColorScriptBeat",
    "CompiledCraftRules",
    "DialogueCueSpec",
    "DialogueDeliveryMode",
    "DirectorTasteProfile",
    "FacingWallZone",
    "FrameStateSpec",
    "LocationAsset",
    "MotionDeltaBeat",
    "PropAsset",
    "SceneWorldState",
    "ShotCameraSpec",
    "ShotLightingSpec",
    "ShotPropInstance",
    "ShotRenderEngine",
    "ShotScale",
    "ShotSoundStems",
    "SpatialZonesSpec",
    "StoryArchetype",
    "StoryRubricReport",
    "TasteReactionEvent",
    "TasteSignature",
    "TreatmentPitch",
    "VibeDials",
    "WorkflowMode",
    "WorldVault",
]
