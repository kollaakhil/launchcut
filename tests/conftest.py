"""Shared offline pytest fixtures for gtm-videogen-deepagents."""

from __future__ import annotations

from pathlib import Path

import pytest

from gtm_videogen.models.schemas import (
    BeatSpec,
    CameraGrammar,
    ConceptPitch,
    CostLedger,
    CreativeBible,
    GraphicsTheme,
    MediaRef,
    MusicBrief,
    NarrativeStructure,
    ResearchBundle,
    RunInput,
    RunState,
    Shot,
    ShotSheet,
    SonicIdentity,
    SoundPlan,
    TypographyAndLayout,
    VisualLanguage,
    VoiceCasting,
    VoiceCue,
)


@pytest.fixture
def sample_run_input() -> RunInput:
    return RunInput(
        brand_name="Solara Botanics",
        brand_url="https://solarabotanics.example.com",
        goal="Drive first-time trial orders for cold-pressed ginger tonic",
        offer="Save 20% on your first 6-pack",
        facts=[
            "Cold-pressed in small batches in Portland",
            "Only 4g of organic cane sugar per bottle",
            "Ships nationwide in 48 hours",
        ],
        target_duration_sec=15.0,
        aspect="16:9",
        fps=24,
        audio_mode="both",
        cap_usd=5.0,
        mock=True,
    )


@pytest.fixture
def sample_research() -> ResearchBundle:
    return ResearchBundle(
        brand_url="https://solarabotanics.example.com",
        brand_name="Solara Botanics",
        verified_claims=[
            "Only 4g of organic cane sugar per bottle",
            "Cold-pressed in small batches in Portland",
            "20% off first 6-pack with code FIRST20",
        ],
        product_facts=[
            "Real Peruvian ginger root and Meyer lemon",
            'Not verified - do not use "cures migraines"',
        ],
        proof_points=["Rated 4.9 stars across 1,200 verified buyers"],
        voc_quotes=["Crisp ginger bite without the syrupy crash."],
        brand_colors=["#F4A261", "#264653", "#FAF3DD"],
        product_photos=[
            MediaRef(
                id="photo_1",
                url="https://solarabotanics.example.com/images/bottle.jpg",
                alt="Solara Botanics glass bottle",
            )
        ],
        banned_phrases=["miracle cure"],
    )


@pytest.fixture
def sample_creative_bible() -> CreativeBible:
    return CreativeBible(
        concept_thesis="Morning sunlight ritual contrasting fresh ginger craft with syrupy sodas.",
        narrative_structure=NarrativeStructure(
            arc_name="Tactile Craft Reveal",
            beats=[
                BeatSpec(
                    name="sensory_hook",
                    purpose="Grab attention with texture",
                    duration_hint_sec=3.0,
                ),
                BeatSpec(
                    name="craft_proof", purpose="Show real ingredients", duration_hint_sec=5.0
                ),
                BeatSpec(name="payoff", purpose="Product tasting moment", duration_hint_sec=4.0),
                BeatSpec(name="offer_close", purpose="Clear trial offer", duration_hint_sec=3.0),
            ],
            ending_strategy="cta",
        ),
        visual_language=VisualLanguage(
            world="Sunlit kitchen counter with warm terracotta tiles",
            lighting_style="Natural morning sidelight",
            color_palette=["#F4A261", "#264653", "#FAF3DD"],
            texture_and_grade="35mm warm organic grain",
        ),
        camera_grammar=CameraGrammar(
            dominant_style="Handheld macro with crisp cuts",
            allowed_moves=["micro push", "lateral slide", "static lock"],
            lens_range_mm="35-85mm",
            pacing_curve="Brisk opener settling into steady reveal",
        ),
        typography_and_layout=TypographyAndLayout(
            font_family="Liberation Serif",
            caption_treatment="clean_bottom",
            end_card_layout="editorial_split",
            motion_feel="measured",
        ),
        sonic_identity=SonicIdentity(
            audio_strategy=(
                "Warm conversational VO over subtle acoustic guitar and natural kitchen foley"
            ),
            music_brief=MusicBrief(
                genre="indie acoustic",
                mood="bright and grounded",
                instrumentation=["acoustic guitar", "soft shaker"],
                tempo_feel="medium",
                prompt="Warm sunlit indie acoustic guitar bed with light percussion",
                tags=["acoustic", "warm", "instrumental"],
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
        anti_cliche_list=[
            "No unsolicited dark studio voids",
            "No generic 'Are you tired of' openers",
        ],
    )


@pytest.fixture
def sample_shot_sheet() -> ShotSheet:
    return ShotSheet(
        fps=24,
        aspect="16:9",
        target_duration_sec=15.0,
        audio_mode="both",
        theme=GraphicsTheme(font_family="Liberation Serif"),
        shots=[
            Shot(
                id="s1",
                index=0,
                beat="sensory_hook",
                duration_sec=3.0,
                engine="ai_video",
                visual_prompt="Macro sunlight hitting fresh sliced ginger root on ceramic board.",
                camera_move="lateral slide",
                vo_line="Fresh ginger hits different.",
                sfx_note="Crisp knife slice on ceramic",
            ),
            Shot(
                id="s2",
                index=1,
                beat="craft_proof",
                duration_sec=5.0,
                engine="ai_video",
                visual_prompt=(
                    "Golden cold-pressed juice streaming into a glass carafe in sunlit kitchen."
                ),
                camera_move="micro push",
                vo_line="Cold-pressed in Portland with four grams of sugar.",
                sfx_note="Gentle liquid stream",
            ),
            Shot(
                id="s3",
                index=2,
                beat="payoff",
                duration_sec=4.0,
                engine="image_to_video",
                visual_prompt=(
                    "Chilled glass bottle resting beside sliced Meyer lemon on terracotta tile."
                ),
                camera_move="static lock",
                product_hero=True,
                vo_line="Real Peruvian ginger and bright lemon.",
                on_screen_text="4g Organic Sugar",
            ),
            Shot(
                id="s4",
                index=3,
                beat="offer_close",
                duration_sec=3.0,
                engine="motion_card",
                visual_prompt="Warm cream editorial card with product silhouette.",
                camera_move="static lock",
                vo_line="Save 20% on your first six-pack.",
                on_screen_text="20% Off First 6-Pack",
            ),
        ],
        end_card_title="Solara Botanics",
        end_card_subtitle="Save 20% on your first 6-pack",
    )


@pytest.fixture
def sample_sound_plan() -> SoundPlan:
    return SoundPlan(
        voice_id="v_aria_f_adult_us",
        voice_provider="fal",
        vo_cues=[
            VoiceCue(
                shot_id="s1", at_sec=0.0, duration_sec=3.0, text="Fresh ginger hits different."
            ),
            VoiceCue(
                shot_id="s2",
                at_sec=3.0,
                duration_sec=5.0,
                text="Cold-pressed in Portland with four grams of sugar.",
            ),
        ],
    )


@pytest.fixture
def sample_concepts() -> list[ConceptPitch]:
    return [
        ConceptPitch(
            id="c1",
            title="Sunlit Kitchen Craft",
            thesis="Warm tactile morning craft.",
            narrative_arc="sensory hook to ingredient proof to taste payoff",
            visual_world="sunlit terracotta kitchen with morning sidelight",
            sonic_strategy="warm acoustic guitar and kitchen foley",
            emotional_register="grounded warmth and curiosity",
        ),
        ConceptPitch(
            id="c2",
            title="Founder Dispatch",
            thesis="Candid founder documentary.",
            narrative_arc="candid confession to workshop tour to invitation",
            visual_world="documentary portland press room with stainless steel",
            sonic_strategy="dry intimate voiceover with room tone only",
            emotional_register="honest transparency and pride",
        ),
        ConceptPitch(
            id="c3",
            title="Midday Slump Contrast",
            thesis="Playful afternoon energy shift.",
            narrative_arc="before after contrast with witty punchline",
            visual_world="crisp minimalist desk and sunlit balcony",
            sonic_strategy="percussive pizzicato strings and brisk cuts",
            emotional_register="witty relief and bright energy",
        ),
        ConceptPitch(
            id="c4",
            title="Kitchen Craft Duplicate",
            thesis="Another sunlit kitchen craft angle.",
            narrative_arc="sensory hook to ingredient proof to taste payoff",
            visual_world="sunlit terracotta kitchen with morning sidelight",
            sonic_strategy="warm acoustic guitar and kitchen foley",
            emotional_register="grounded warmth and curiosity",
        ),
    ]


@pytest.fixture
def sample_run_state(sample_run_input: RunInput) -> RunState:
    return RunState(
        run_id="run_test_001",
        created_by="tester",
        status="idle",
        input=sample_run_input,
        cost_ledger=CostLedger(run_id="run_test_001", cap_usd=sample_run_input.cap_usd),
    )


@pytest.fixture
def repo_src_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "src" / "gtm_videogen"
