"""Shared fixtures, marker registration, and real FFmpeg/ffprobe helpers for E2E tests."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from gtm_videogen.models.schemas import (
    BeatSpec,
    CameraGrammar,
    CostLedger,
    CreativeBible,
    GraphicsTheme,
    MediaRef,
    MusicBrief,
    MusicCue,
    NarrativeStructure,
    ResearchBundle,
    RunInput,
    RunState,
    SfxCue,
    Shot,
    ShotSheet,
    SonicIdentity,
    SoundPlan,
    TypographyAndLayout,
    VisualLanguage,
    VoiceCasting,
    VoiceCue,
)


def pytest_configure(config: pytest.Config) -> None:
    """Register E2E custom markers without modifying pyproject.toml."""
    markers = [
        "e2e: End-to-end opaque-box test suite",
        "tier1: Tier 1 isolated happy-path feature tests",
        "tier2: Tier 2 boundary, corner, error-rejection, and bug regression tests",
        "tier3: Tier 3 pairwise cross-feature combination tests",
        "tier4: Tier 4 realistic end-to-end application workload tests",
        "m1: E2E tests runnable immediately against M1 modules (models/core/rules)",
    ]
    for marker in markers:
        config.addinivalue_line("markers", marker)


def repo_root_path() -> Path:
    """Return the repository root directory."""
    return Path(__file__).resolve().parents[2]


def _bin_path(name: str) -> str:
    found = shutil.which(name)
    if not found:
        raise RuntimeError(f"Required binary {name!r} not found on PATH")
    return found


def ffprobe_inspect(path: Path) -> dict[str, Any]:
    """Run real ffprobe on a media file and return parsed JSON metadata."""
    ffprobe = _bin_path("ffprobe")
    proc = subprocess.run(
        [
            ffprobe,
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(proc.stdout)


def parse_frame_rate(rate_str: str) -> float:
    """Parse an ffprobe r_frame_rate string like '30/1' or '24000/1001' into float."""
    if "/" in rate_str:
        num_s, den_s = rate_str.split("/", 1)
        den = float(den_s)
        return float(num_s) / den if den != 0 else 0.0
    return float(rate_str)


def make_synthetic_clip(
    path: Path,
    *,
    width: int = 640,
    height: int = 360,
    fps: int = 24,
    duration_sec: float = 2.0,
    color: str = "0x264653",
    text: str | None = None,
    include_audio: bool = False,
    audio_freq_hz: int = 440,
) -> Path:
    """Generate a real H.264 MP4 synthetic clip via FFmpeg for opaque-box testing."""
    ffmpeg = _bin_path("ffmpeg")
    path.parent.mkdir(parents=True, exist_ok=True)
    vf_parts = ["format=yuv420p"]
    if text:
        escaped = text.replace(":", r"\:").replace("'", r"\'")
        font_file = repo_root_path() / "assets" / "fonts" / "Roboto-Bold.ttf"
        font_opt = f":fontfile='{font_file}'" if font_file.exists() else ""
        vf_parts.insert(
            0,
            f"drawtext=text='{escaped}'{font_opt}:fontcolor=white:fontsize=48"
            ":x=(w-text_w)/2:y=(h-text_h)/2",
        )
    cmd = [
        ffmpeg,
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"color=c={color}:s={width}x{height}:r={fps}:d={duration_sec}",
    ]
    if include_audio:
        cmd.extend(
            [
                "-f",
                "lavfi",
                "-i",
                f"sine=frequency={audio_freq_hz}:sample_rate=44100:duration={duration_sec}",
            ]
        )
    cmd.extend(
        [
            "-vf",
            ",".join(vf_parts),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-r",
            str(fps),
        ]
    )
    if include_audio:
        cmd.extend(["-c:a", "aac", "-b:a", "128k", "-shortest"])
    cmd.extend(["-t", str(duration_sec), str(path)])
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    return path


def make_synthetic_wav(
    path: Path,
    *,
    duration_sec: float = 2.0,
    freq_hz: int = 440,
    sample_rate: int = 44100,
) -> Path:
    """Generate a real PCM WAV audio file via FFmpeg."""
    ffmpeg = _bin_path("ffmpeg")
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency={freq_hz}:sample_rate={sample_rate}:duration={duration_sec}",
        "-c:a",
        "pcm_s16le",
        str(path),
    ]
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    return path


def assert_valid_sse_event(
    event: dict[str, Any], *, expected_run_id: str | None = None
) -> None:
    """Validate that a parsed SSE JSON payload matches PROJECT.md's SSE event contract."""
    assert isinstance(event, dict)
    assert isinstance(event.get("type"), str) and len(event["type"]) > 0
    assert isinstance(event.get("run_id"), str) and len(event["run_id"]) > 0
    if expected_run_id is not None:
        assert event["run_id"] == expected_run_id
    assert isinstance(event.get("timestamp"), str) and len(event["timestamp"]) > 0
    assert event.get("stage") is None or isinstance(event.get("stage"), str)
    assert isinstance(event.get("data"), dict)


def make_run_input(**overrides: Any) -> RunInput:
    """Build a valid opaque-box RunInput with optional field overrides."""
    defaults: dict[str, Any] = {
        "brand_name": "Kestrel Roasters",
        "brand_url": "https://kestrelroasters.example.com",
        "goal": "Convert specialty coffee drinkers to a bi-weekly single-origin subscription",
        "offer": "Save 25% on your first two bags",
        "facts": [
            "Roasted to order in Seattle every Tuesday",
            "100% direct-trade Colombian and Ethiopian lots",
            "Compostable bags shipped within 24 hours of roast",
        ],
        "must_say": ["Roasted to order in Seattle"],
        "target_duration_sec": 12.0,
        "aspect": "16:9",
        "fps": 24,
        "audio_mode": "both",
        "cap_usd": 5.0,
        "num_concepts": 4,
        "render_mode": "preview",
        "mock": True,
    }
    defaults.update(overrides)
    return RunInput(**defaults)


def make_research_bundle(**overrides: Any) -> ResearchBundle:
    """Build a realistic ResearchBundle for opaque-box testing."""
    defaults: dict[str, Any] = {
        "brand_url": "https://kestrelroasters.example.com",
        "brand_name": "Kestrel Roasters",
        "verified_claims": [
            "Roasted to order in Seattle every Tuesday",
            "100% direct-trade Colombian and Ethiopian lots",
            "Save 25% on your first two bags",
            "Compostable bags shipped within 24 hours of roast",
        ],
        "product_facts": [
            "Washed Huila micro-lot with blood orange and cocoa tasting notes",
            'Not verified - do not use "doubles your IQ"',
        ],
        "proof_points": ["4.9-star average across 840 subscriber reviews"],
        "voc_quotes": ["First subscription that actually arrives smelling like fresh roast."],
        "brand_colors": ["#2B2D42", "#D90429", "#EDF2F4"],
        "product_photos": [
            MediaRef(
                id="kestrel_bag_1",
                url="https://kestrelroasters.example.com/images/huila-bag.jpg",
                alt="Kestrel Roasters Huila coffee bag",
                width=1200,
                height=1200,
                source="og:image",
            )
        ],
        "banned_phrases": ["guaranteed miracle"],
    }
    defaults.update(overrides)
    return ResearchBundle(**defaults)


def make_creative_bible(**overrides: Any) -> CreativeBible:
    """Build a bespoke CreativeBible with bundled font and rich sensory specs."""
    defaults: dict[str, Any] = {
        "concept_thesis": (
            "Contrast stale supermarket shelves with steam rising from Tuesday-roasted beans."
        ),
        "narrative_structure": NarrativeStructure(
            arc_name="Roast-Day Ritual",
            beats=[
                BeatSpec(name="hook", purpose="Immediate sensory contrast", duration_hint_sec=3.0),
                BeatSpec(name="proof", purpose="Seattle roast craft", duration_hint_sec=4.0),
                BeatSpec(name="payoff", purpose="Pour-over bloom", duration_hint_sec=2.5),
                BeatSpec(name="cta", purpose="Clear trial offer", duration_hint_sec=2.5),
            ],
            ending_strategy="cta",
        ),
        "visual_language": VisualLanguage(
            world="Warm Pacific Northwest roastery counter in natural overcast light",
            lighting_style="Soft diffused window light with crisp specular highlights",
            color_palette=["#2B2D42", "#D90429", "#EDF2F4"],
            texture_and_grade="Natural filmic contrast, clean shadows",
        ),
        "camera_grammar": CameraGrammar(
            dominant_style="Steady macro glides with decisive cuts",
            allowed_moves=["slow push", "top-down pan", "locked frame"],
            lens_range_mm="35-85mm",
            pacing_curve="Brisk opening into calm pour-over reveal",
        ),
        "typography_and_layout": TypographyAndLayout(
            font_family="Roboto",
            caption_treatment="clean_bottom",
            end_card_layout="centered_minimal",
            motion_feel="crisp",
        ),
        "sonic_identity": SonicIdentity(
            audio_strategy="Grounded conversational VO with warm upright bass and grinder foley",
            music_brief=MusicBrief(
                genre="warm minimalist jazz",
                mood="focused and inviting",
                instrumentation=["upright bass", "brushed snare"],
                tempo_feel="medium",
                prompt="Warm minimalist jazz upright bass and brushed snare instrumental",
                tags=["jazz", "warm", "instrumental"],
            ),
            voice_casting=VoiceCasting(
                gender="female",
                age="adult",
                accent="us",
                energy="warm",
                pace="natural",
            ),
            transition_style="natural_foley",
        ),
        "anti_cliche_list": [
            "No generic morning alarm clock groan",
            "No neon cyberpunk backdrops",
        ],
    }
    defaults.update(overrides)
    return CreativeBible(**defaults)


def make_shot_sheet(
    *,
    fps: int = 24,
    aspect: str = "16:9",
    target_duration_sec: float = 12.0,
    audio_mode: str = "both",
    shots: list[Shot] | None = None,
) -> ShotSheet:
    """Build a valid ShotSheet matching target duration and fps."""
    default_shots = shots or [
        Shot(
            id="s1",
            index=0,
            beat="hook",
            duration_sec=3.0,
            engine="ai_video",
            visual_prompt=(
                "Macro shot of freshly roasted coffee beans tumbling into a cooling tray."
            ),
            camera_move="slow push",
            vo_line="Fresh coffee should not sit in a warehouse.",
            sfx_note="Warm roasted beans tumbling",
        ),
        Shot(
            id="s2",
            index=1,
            beat="proof",
            duration_sec=4.0,
            engine="ai_video",
            visual_prompt="Roaster inspecting golden beans beside a steel drum in overcast light.",
            camera_move="top-down pan",
            vo_line="Kestrel is roasted to order in Seattle every Tuesday.",
            sfx_note="Subtle roaster hum",
        ),
        Shot(
            id="s3",
            index=2,
            beat="payoff",
            duration_sec=2.5,
            engine="image_to_video",
            visual_prompt="Pour-over bloom rising in a glass carafe beside the kraft coffee bag.",
            camera_move="locked frame",
            product_hero=True,
            vo_line="Shipped within 24 hours.",
            on_screen_text="Roasted To Order",
        ),
        Shot(
            id="s4",
            index=3,
            beat="cta",
            duration_sec=2.5,
            engine="motion_card",
            visual_prompt="Minimalist cream end card framing the coffee bag silhouette.",
            camera_move="locked frame",
            vo_line="Save 25% on two bags.",
            on_screen_text="25% Off First 2 Bags",
        ),
    ]
    return ShotSheet(
        fps=fps,  # type: ignore[arg-type]
        aspect=aspect,  # type: ignore[arg-type]
        target_duration_sec=target_duration_sec,
        audio_mode=audio_mode,  # type: ignore[arg-type]
        theme=GraphicsTheme(font_family="Roboto"),
        shots=default_shots,
        end_card_title="Kestrel Roasters",
        end_card_subtitle="Save 25% on your first two bags",
    )


def make_sound_plan(**overrides: Any) -> SoundPlan:
    """Build a realistic SoundPlan with VO, SFX, and music cues."""
    defaults: dict[str, Any] = {
        "voice_id": "v_aria_f_adult_us",
        "voice_provider": "fal",
        "voice_casting": VoiceCasting(
            gender="female",
            age="adult",
            accent="us",
            energy="warm",
            pace="natural",
        ),
        "vo_cues": [
            VoiceCue(
                shot_id="s1",
                at_sec=0.0,
                duration_sec=3.0,
                text="Fresh coffee should not sit in a warehouse.",
            ),
            VoiceCue(
                shot_id="s2",
                at_sec=3.0,
                duration_sec=4.0,
                text="Kestrel is roasted to order in Seattle every Tuesday.",
            ),
        ],
        "sfx_cues": [
            SfxCue(
                shot_id="s1",
                at_sec=0.2,
                duration_sec=1.5,
                prompt="Roasted coffee beans tumbling gently into a metal tray",
                gain_db=-8.0,
            ),
        ],
        "music_cue": MusicCue(
            prompt="Warm minimalist jazz upright bass and brushed snare instrumental",
            tags=["jazz", "warm", "instrumental"],
            duration_sec=12.0,
            gain_db=-16.0,
            duck_db=-10.0,
        ),
        "silence_gaps": [],
    }
    defaults.update(overrides)
    return SoundPlan(**defaults)


def make_run_state(
    *,
    run_id: str = "e2e_run_001",
    created_by: str = "e2e_user",
    run_input: RunInput | None = None,
    **overrides: Any,
) -> RunState:
    """Build a valid RunState with a matching CostLedger."""
    inp = run_input or make_run_input()
    defaults: dict[str, Any] = {
        "run_id": run_id,
        "created_by": created_by,
        "status": "idle",
        "input": inp,
        "cost_ledger": CostLedger(run_id=run_id, cap_usd=inp.cap_usd),
    }
    defaults.update(overrides)
    return RunState(**defaults)


@pytest.fixture
def e2e_repo_root() -> Path:
    return repo_root_path()


@pytest.fixture
def e2e_run_input() -> RunInput:
    return make_run_input()


@pytest.fixture
def e2e_research() -> ResearchBundle:
    return make_research_bundle()


@pytest.fixture
def e2e_bible() -> CreativeBible:
    return make_creative_bible()


@pytest.fixture
def e2e_shot_sheet() -> ShotSheet:
    return make_shot_sheet()


@pytest.fixture
def e2e_sound_plan() -> SoundPlan:
    return make_sound_plan()


@pytest.fixture
def e2e_run_state(e2e_run_input: RunInput) -> RunState:
    return make_run_state(run_input=e2e_run_input)
