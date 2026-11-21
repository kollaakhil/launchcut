"""Strict Pydantic v2 data models for gtm-videogen-deepagents."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _utc_now() -> datetime:
    return datetime.now(UTC)


class RunInput(BaseModel):
    """Strict user request schema for creating a video generation run (fixes BUG-03)."""

    model_config = ConfigDict(extra="forbid")

    brand_name: str = Field(min_length=1, max_length=200)
    brand_url: str = Field(min_length=1, max_length=2048)
    goal: str = Field(min_length=1, max_length=2000)
    offer: str | None = None
    facts: list[str] = Field(default_factory=list)
    must_say: list[str] = Field(default_factory=list)
    target_duration_sec: float = Field(15.0, ge=6.0, le=60.0)
    aspect: Literal["16:9", "9:16", "1:1", "4:5"] = "16:9"
    fps: Literal[24, 30] = 24
    audio_mode: Literal["vo", "foley", "both"] = "vo"
    inspiration_url: str | None = None
    user_direction: str | None = None
    cap_usd: float = Field(5.0, ge=0.0, le=500.0)
    num_concepts: int = Field(4, ge=1, le=8)
    render_mode: Literal["preview", "final"] = "preview"
    mock: bool = False

    @field_validator("brand_name", "goal")
    @classmethod
    def _validate_non_blank_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Field must not be empty or whitespace-only")
        return stripped

    @field_validator("brand_url")
    @classmethod
    def _validate_brand_url(cls, value: str) -> str:
        stripped = value.strip()
        parsed = urlparse(stripped)
        if parsed.scheme.lower() not in ("http", "https") or not parsed.hostname:
            raise ValueError("brand_url must be a valid http:// or https:// URL with a hostname")
        return stripped

    @field_validator("inspiration_url")
    @classmethod
    def _validate_inspiration_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            return None
        parsed = urlparse(stripped)
        if parsed.scheme.lower() not in ("http", "https") or not parsed.hostname:
            raise ValueError(
                "inspiration_url must be a valid http:// or https:// URL with a hostname"
            )
        return stripped


class MediaRef(BaseModel):
    """Reference to an image, audio, or video asset (path or URL)."""

    id: str
    url: str | None = None
    path: str | None = None
    mime_type: str = "image/jpeg"
    alt: str = ""
    width: int | None = None
    height: int | None = None
    source: str = ""


class ResearchBundle(BaseModel):
    """Verified brand research, facts, proof points, and scraped product photos."""

    brand_url: str
    brand_name: str
    verified_claims: list[str] = Field(default_factory=list)
    product_facts: list[str] = Field(default_factory=list)
    proof_points: list[str] = Field(default_factory=list)
    voc_quotes: list[str] = Field(default_factory=list)
    brand_colors: list[str] = Field(default_factory=list)
    product_photos: list[MediaRef] = Field(default_factory=list)
    inspiration_notes: list[str] = Field(default_factory=list)
    banned_phrases: list[str] = Field(default_factory=list)
    scraped_at: datetime = Field(default_factory=_utc_now)


class CreativeFingerprint(BaseModel):
    """Compact creative signature persisted per brand to prevent cross-run repetition."""

    run_id: str
    brand_url: str
    created_at: datetime = Field(default_factory=_utc_now)
    arc_name: str
    hook_opening: str
    visual_world: str
    lighting_style: str
    camera_style: str
    music_genre: str | None = None
    voice_id: str | None = None
    outcome: Literal["approved", "rejected", "completed", "failed"] = "completed"


class CreativeHistory(BaseModel):
    """Rolling window of the most recent CreativeFingerprints for a brand."""

    brand_url: str
    entries: list[CreativeFingerprint] = Field(default_factory=list)


class ConceptPitch(BaseModel):
    """Structurally orthogonal concept pitch generated during divergent ideation."""

    id: str
    title: str
    thesis: str
    narrative_arc: str
    visual_world: str
    sonic_strategy: str
    emotional_register: str
    diversity_score: float = Field(0.0, ge=0.0)


class BeatSpec(BaseModel):
    """Single narrative beat inside a bespoke CreativeBible."""

    name: str
    purpose: str
    duration_hint_sec: float = Field(gt=0.0)
    emotional_shift: str = ""


class NarrativeStructure(BaseModel):
    """Dynamic brief-driven story arc (replaces hardcoded story patterns)."""

    arc_name: str
    beats: list[BeatSpec] = Field(min_length=1)
    ending_strategy: Literal["cta", "loop", "punchline", "logo_linger"] = "cta"


class VisualLanguage(BaseModel):
    """Visual world, lighting, color palette, and film grade."""

    world: str
    lighting_style: str
    color_palette: list[str] = Field(default_factory=list)
    texture_and_grade: str = ""


class CameraGrammar(BaseModel):
    """Camera movement language, lens range, and editorial pacing."""

    dominant_style: str
    allowed_moves: list[str] = Field(default_factory=list)
    lens_range_mm: str = "24-85mm"
    pacing_curve: str = ""


class TypographyAndLayout(BaseModel):
    """Typography and motion graphics direction grounded in bundled open-license fonts."""

    font_family: str = "Roboto"
    caption_treatment: str = "clean_bottom"
    end_card_layout: str = "centered_minimal"
    motion_feel: str = "crisp"


class VoiceCasting(BaseModel):
    """Full voice persona specification for attribute-matched voice selection (BUG-17)."""

    gender: Literal["female", "male", "neutral", "any"] = "any"
    age: Literal["young", "adult", "mature", "any"] = "any"
    accent: str = "neutral"
    energy: Literal["calm", "warm", "conversational", "energetic", "authoritative"] = (
        "conversational"
    )
    pace: Literal["slow", "measured", "natural", "brisk"] = "natural"
    delivery_style: str = ""


class MusicBrief(BaseModel):
    """Bespoke music direction synthesized from the creative brief."""

    genre: str
    mood: str
    instrumentation: list[str] = Field(default_factory=list)
    tempo_feel: str = "medium"
    energy_curve: str = ""
    prompt: str = ""
    tags: list[str] = Field(default_factory=list)


class SonicIdentity(BaseModel):
    """Holistic sound strategy covering VO, SFX, music, and transitions."""

    audio_strategy: str
    music_brief: MusicBrief | None = None
    voice_casting: VoiceCasting = Field(default_factory=VoiceCasting)
    transition_style: str = "clean_cuts"


class CreativeBible(BaseModel):
    """Runtime brief-driven creative direction (zero hardcoded templates)."""

    concept_thesis: str
    narrative_structure: NarrativeStructure
    visual_language: VisualLanguage
    camera_grammar: CameraGrammar
    typography_and_layout: TypographyAndLayout = Field(default_factory=TypographyAndLayout)
    sonic_identity: SonicIdentity
    anti_cliche_list: list[str] = Field(default_factory=list)


class ScriptLine(BaseModel):
    """Single spoken line bound to a shot."""

    shot_id: str
    text: str
    word_count: int = Field(0, ge=0)
    estimated_sec: float = Field(0.0, ge=0.0)
    delivery_note: str = ""


class ClaimCitation(BaseModel):
    """Citation mapping a claim in the script to verified research."""

    claim_text: str
    source_index: int | None = None
    source_fact: str = ""


class Script(BaseModel):
    """Spoken script with per-shot lines and grounded claim citations."""

    full_vo_text: str = ""
    lines: list[ScriptLine] = Field(default_factory=list)
    claims_used: list[ClaimCitation] = Field(default_factory=list)


class GraphicsTheme(BaseModel):
    """Visual theme for captions and motion graphics cards."""

    font_family: str = "Roboto"
    primary_color: str = "#FFFFFF"
    accent_color: str = "#3B82F6"
    background_color: str = "#111827"
    text_color: str = "#FFFFFF"
    caption_treatment: str = "clean_bottom"
    clamp_background_luminance: bool = False


class Shot(BaseModel):
    """Individual shot specification with open beat and expressive renderer fields."""

    id: str
    index: int = Field(ge=0)
    beat: str
    duration_sec: float = Field(gt=0.0)
    engine: Literal["ai_video", "image_to_video", "motion_card", "blender", "manim"] = "ai_video"
    visual_prompt: str
    negative_prompt: str = ""
    camera_move: str = ""
    camera_angle: str = ""
    lens_mm: int | None = None
    transition: Literal[
        "cut", "fade_black", "crossfade", "j_cut", "whip", "match_cut", "smash_cut"
    ] = "cut"
    environment: str | None = None
    lighting_rig: str | None = None
    motion_curve: str | None = None
    theme: GraphicsTheme | None = None
    product_hero: bool = False
    on_screen_text: str | None = None
    vo_line: str | None = None
    sfx_note: str | None = None
    reference_image: str | None = None


class ShotSheet(BaseModel):
    """Ordered sequence of shots with exact fps, aspect, and duration budget."""

    fps: Literal[24, 30] = 24
    aspect: Literal["16:9", "9:16", "1:1", "4:5"] = "16:9"
    target_duration_sec: float = Field(gt=0.0)
    audio_mode: Literal["vo", "foley", "both"] = "both"
    theme: GraphicsTheme = Field(default_factory=GraphicsTheme)
    shots: list[Shot] = Field(min_length=1)
    end_card_title: str | None = None
    end_card_subtitle: str | None = None
    end_card_footer: str | None = None


class VoiceCue(BaseModel):
    """Voiceover synthesis cue aligned to a shot timestamp."""

    shot_id: str
    at_sec: float = Field(ge=0.0)
    duration_sec: float = Field(gt=0.0)
    text: str
    delivery_note: str = ""
    pronunciation: dict[str, str] = Field(default_factory=dict)


class SfxCue(BaseModel):
    """Sound effect cue placed at a specific timeline offset."""

    shot_id: str | None = None
    at_sec: float = Field(ge=0.0)
    duration_sec: float = Field(gt=0.0)
    prompt: str
    gain_db: float = -6.0


class MusicCue(BaseModel):
    """Background music bed cue with ducking configuration."""

    prompt: str
    tags: list[str] = Field(default_factory=list)
    duration_sec: float = Field(gt=0.0)
    gain_db: float = -16.0
    duck_db: float = -10.0


class SoundPlan(BaseModel):
    """Bespoke sound design plan for a shot sheet."""

    voice_id: str = ""
    voice_provider: str = "fal"
    voice_casting: VoiceCasting = Field(default_factory=VoiceCasting)
    vo_cues: list[VoiceCue] = Field(default_factory=list)
    sfx_cues: list[SfxCue] = Field(default_factory=list)
    music_cue: MusicCue | None = None
    silence_gaps: list[tuple[float, float]] = Field(default_factory=list)


class RenderedClip(BaseModel):
    """Metadata for a rendered video clip stored on disk."""

    shot_id: str
    path: str
    duration_sec: float = Field(gt=0.0)
    fps: int = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    cache_hit: bool = False
    usd: float = Field(0.0, ge=0.0)


class RenderedAudio(BaseModel):
    """Metadata for a rendered audio asset stored on disk."""

    kind: Literal["vo", "sfx", "music"]
    cue_id: str = ""
    path: str
    duration_sec: float = Field(0.0, ge=0.0)
    cache_hit: bool = False
    usd: float = Field(0.0, ge=0.0)


class RenderManifest(BaseModel):
    """Complete record of rendered clips, audio assets, stitched output, and warnings."""

    run_id: str
    version_id: str = "v1"
    clips: list[RenderedClip] = Field(default_factory=list)
    audio_assets: list[RenderedAudio] = Field(default_factory=list)
    ass_caption_path: str | None = None
    stitched_path: str | None = None
    poster_path: str | None = None
    timings_ms: dict[str, int] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None


class CostEntry(BaseModel):
    """Single metered provider spend entry."""

    timestamp: datetime = Field(default_factory=_utc_now)
    stage: str
    provider: str
    model: str
    what: str
    estimated_usd: float = Field(0.0, ge=0.0)
    actual_usd: float = Field(0.0, ge=0.0)
    cache_hit: bool = False


class CostLedger(BaseModel):
    """Per-run USD budget ledger."""

    run_id: str
    cap_usd: float = Field(5.0, ge=0.0)
    estimated_usd: float = Field(0.0, ge=0.0)
    actual_usd: float = Field(0.0, ge=0.0)
    entries: list[CostEntry] = Field(default_factory=list)


class QAIssue(BaseModel):
    """Structured blocking error or warning from script rules or media QA."""

    code: str
    severity: Literal["blocking", "warning"]
    message: str
    shot_id: str | None = None
    overridable: bool = False
    details: dict[str, Any] = Field(default_factory=dict)


class OCRCheck(BaseModel):
    """Result of Tesseract OCR verification on an on-screen text cue."""

    shot_id: str
    expected_text: str
    observed_text: str
    passed: bool
    at_sec: float = 0.0
    similarity: float = Field(0.0, ge=0.0, le=1.0)


class SyncCheck(BaseModel):
    """Result of Whisper/STT voiceover sync verification."""

    shot_id: str
    expected_vo: str
    transcribed_vo: str
    passed: bool
    similarity: float = Field(0.0, ge=0.0, le=1.0)


class QAReport(BaseModel):
    """Unified QA report covering pure script rules, media checks, and creative scores."""

    passed: bool
    blocking_errors: list[QAIssue] = Field(default_factory=list)
    warnings: list[QAIssue] = Field(default_factory=list)
    originality_score: float = Field(0.0, ge=0.0, le=10.0)
    bible_coherence_score: float = Field(0.0, ge=0.0, le=10.0)
    loudness_lufs: float | None = None
    peak_dbtp: float | None = None
    ocr_checks: list[OCRCheck] = Field(default_factory=list)
    sync_checks: list[SyncCheck] = Field(default_factory=list)
    ffprobe_summary: dict[str, Any] = Field(default_factory=dict)
    failed_shot_ids: list[str] = Field(default_factory=list)
    score_caps_applied: list[str] = Field(default_factory=list)


class ChatOptions(BaseModel):
    """Stage-aware LLM sampling options filtered per provider capability matrix."""

    temperature: float | None = Field(None, ge=0.0, le=2.0)
    top_p: float | None = Field(None, gt=0.0, le=1.0)
    max_tokens: int | None = Field(None, gt=0)
    presence_penalty: float | None = Field(None, ge=-2.0, le=2.0)
    frequency_penalty: float | None = Field(None, ge=-2.0, le=2.0)
    reasoning_effort: Literal["low", "medium", "high"] | None = None
    json_schema: dict[str, Any] | None = None


class Version(BaseModel):
    """Individual creative/render version supporting A/B Creative Bibles and preview stacks."""

    version_id: str
    label: str = ""
    tier: Literal["preview", "final"] = "preview"
    status: Literal["queued", "working", "done", "failed", "cancelled"] = "done"
    stack: dict[str, Any] = Field(default_factory=dict)
    usd: float = Field(0.0, ge=0.0)
    is_winner: bool = False
    creative_bible: CreativeBible | None = None
    shot_sheet: ShotSheet | None = None
    sound_plan: SoundPlan | None = None
    render_manifest: RenderManifest | None = None
    qa_report: QAReport | None = None
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None


class RunState(BaseModel):
    """Full persisted state for a video generation run."""

    run_id: str
    created_by: str = "local"
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)
    status: Literal["idle", "working", "done", "cancelled", "failed", "interrupted"] = "idle"
    step: Literal["brand", "concept", "sheet", "previews", "final"] = "brand"
    selected_concept_id: str | None = None
    winning_version_id: str | None = None
    input: RunInput
    research: ResearchBundle | None = None
    concepts: list[ConceptPitch] = Field(default_factory=list)
    creative_bible: CreativeBible | None = None
    script: Script | None = None
    shot_sheet: ShotSheet | None = None
    sound_plan: SoundPlan | None = None
    render_manifest: RenderManifest | None = None
    versions: list[Version] = Field(default_factory=list)
    cost_ledger: CostLedger
    qa_report: QAReport | None = None
    rubric_evaluations: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    repair_result: dict[str, Any] | None = None
    gate_override: bool = False
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None


class JobStatusDTO(BaseModel):
    """Strictly JSON-serializable job status projection (fixes BUG-02)."""

    model_config = ConfigDict(extra="forbid")

    id: str
    run_id: str
    version_id: str | None = None
    shot_id: str | None = None
    stage: str = "pipeline"
    state: Literal["queued", "running", "done", "cancelled", "failed"] = "queued"
    priority: int = 0
    enqueued_at: datetime = Field(default_factory=_utc_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None
    position: int | None = None
    duration_ms: int | None = None
    wait_ms: int | None = None
