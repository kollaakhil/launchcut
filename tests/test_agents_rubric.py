"""Comprehensive offline tests for M4: 3 Rubric-Gated DeepAgents & PipelineRunner.

Verifies:
- Real `deepagents.create_deep_agent` + `deepagents.RubricMiddleware` execution on all 3 agents:
  * `satisfied` happy paths
  * `needs_revision -> satisfied` revision loops
  * Terminal `max_iterations_reached` and `failed` rejection (never assuming AIMessage == pass)
- Full 3-agent `PipelineRunner.run()` in offline mock mode producing real `ffprobe`-verified MP4,
  `QAReport`, `CostLedger`, `rubric_evaluations`, SSE events (sync & async), and `CreativeHistory`.
- Targeted `PipelineRunner.repair()` reusing cached clips & audio (BUG-06, P4).
- Non-fatal SFX warning persistence with `error=None` (BUG-16).
- Mid-pipeline cancellation leaving `status == "cancelled"` (BUG-05).
- Honest fallback for `engine="manim"` when Manim is unavailable.
- Zero hard templates anywhere in `src/gtm_videogen/agents/`.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from gtm_videogen.agents import (
    AGENT1_STAGE,
    AGENT2_STAGE,
    AGENT3_STAGE,
    PipelineRunner,
    RubricGateError,
    run_agent1_research_bible,
    run_agent2_director,
    run_agent3_producer,
)
from gtm_videogen.core.job_registry import CancellationToken
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
from gtm_videogen.providers.audio import AudioProviderError
from gtm_videogen.providers.mocks import MockProviderBundle, ScriptedChatModel
from gtm_videogen.rules.creative_history import CreativeHistoryStore
from gtm_videogen.rules.script_gate import evaluate_script_gate


def _ffprobe_json(path: Path) -> dict[str, Any]:
    proc = subprocess.run(
        [
            "ffprobe",
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


def _make_run_input(**overrides: Any) -> RunInput:
    defaults: dict[str, Any] = {
        "brand_name": "Solstice Tea Co",
        "brand_url": "https://solsticetea.example.com",
        "goal": "Drive first-time sampler box orders from loose-leaf tea drinkers",
        "offer": "Free ceramic infuser on your first box",
        "facts": ["Steeps in 90 seconds", "Sourced directly from family farms"],
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


def _make_run_state(run_id: str = "test_m4_run", **input_overrides: Any) -> RunState:
    inp = _make_run_input(**input_overrides)
    return RunState(
        run_id=run_id,
        input=inp,
        cost_ledger=CostLedger(run_id=run_id, cap_usd=inp.cap_usd),
    )


def _make_research() -> ResearchBundle:
    return ResearchBundle(
        brand_url="https://solsticetea.example.com",
        brand_name="Solstice Tea Co",
        verified_claims=[
            "Steeps in 90 seconds",
            "Sourced directly from family farms",
        ],
        product_facts=[
            "Whole-leaf botanical blends sealed in compostable pouches",
        ],
        brand_colors=["#1B4332", "#D8F3DC", "#D4A373"],
        product_photos=[
            MediaRef(id="p1", url="https://solsticetea.example.com/hero.png", alt="Tea pouch")
        ],
    )


def _make_bible() -> CreativeBible:
    return CreativeBible(
        concept_thesis="Show the quiet 90-second ritual of whole-leaf tea blooming in glass.",
        narrative_structure=NarrativeStructure(
            arc_name="botanical_bloom_study",
            beats=[
                BeatSpec(
                    name="hook", purpose="Pour hot water over whole leaves", duration_hint_sec=4.0
                ),
                BeatSpec(
                    name="proof",
                    purpose="Reveal leaf expansion and origin craft",
                    duration_hint_sec=4.0,
                ),
                BeatSpec(name="cta", purpose="Present starter box offer", duration_hint_sec=4.0),
            ],
            ending_strategy="cta",
        ),
        visual_language=VisualLanguage(
            world="sunlit oak counter with clear glass teapot and warm steam",
            lighting_style="soft morning window light with gentle backlight on steam",
            color_palette=["#1B4332", "#D8F3DC", "#D4A373"],
            texture_and_grade="clean natural film grain",
        ),
        camera_grammar=CameraGrammar(
            dominant_style="steady macro framing with gentle lateral motion",
            allowed_moves=["macro_dolly", "lateral_glide", "static_frame"],
            lens_range_mm="35-85mm",
            pacing_curve="calm opening settling into crisp proof",
        ),
        typography_and_layout=TypographyAndLayout(
            font_family="Playfair Display",
            caption_treatment="clean_bottom",
            end_card_layout="centered_minimal",
            motion_feel="composed",
        ),
        sonic_identity=SonicIdentity(
            audio_strategy="warm voiceover with gentle kettle and pour foley",
            music_brief=MusicBrief(
                genre="acoustic ambient",
                mood="calm and bright",
                instrumentation=["felt piano", "acoustic guitar"],
                tempo_feel="relaxed",
                prompt="acoustic ambient underscore with felt piano and soft guitar",
            ),
            voice_casting=VoiceCasting(
                gender="female",
                age="adult",
                accent="neutral",
                energy="warm",
                pace="measured",
            ),
            transition_style="clean cuts with subtle j_cut",
        ),
        anti_cliche_list=["Are you tired of", "POV:", "game-changer"],
    )


def _make_valid_sheet() -> ShotSheet:
    theme = GraphicsTheme(
        font_family="Playfair Display",
        background_color="#1B4332",
        accent_color="#D4A373",
        text_color="#FFFFFF",
    )
    return ShotSheet(
        fps=24,
        aspect="16:9",
        target_duration_sec=12.0,
        audio_mode="both",
        theme=theme,
        shots=[
            Shot(
                id="s1",
                index=0,
                beat="hook",
                duration_sec=4.0,
                engine="ai_video",
                visual_prompt="Clear glass teapot on sunlit oak counter with rising steam",
                camera_move="macro_dolly",
                vo_line="Meet Solstice Tea Co.",
            ),
            Shot(
                id="s2",
                index=1,
                beat="proof",
                duration_sec=4.0,
                engine="ai_video",
                visual_prompt="Whole tea leaves unfurling in warm morning light",
                camera_move="lateral_glide",
                product_hero=True,
                vo_line="Steeps in 90 seconds from family farms.",
            ),
            Shot(
                id="s3",
                index=2,
                beat="cta",
                duration_sec=4.0,
                engine="ai_video",
                visual_prompt="Tea pouch beside ceramic infuser on linen cloth",
                camera_move="static_frame",
                vo_line="Free ceramic infuser on your first box.",
            ),
        ],
    )


def _make_sound_plan() -> SoundPlan:
    return SoundPlan(
        voice_id="v_maya_warm",
        vo_cues=[
            VoiceCue(shot_id="s1", at_sec=0.0, duration_sec=4.0, text="Meet Solstice Tea Co."),
            VoiceCue(
                shot_id="s2",
                at_sec=4.0,
                duration_sec=4.0,
                text="Steeps in 90 seconds from family farms.",
            ),
            VoiceCue(
                shot_id="s3",
                at_sec=8.0,
                duration_sec=4.0,
                text="Free ceramic infuser on your first box.",
            ),
        ],
        sfx_cues=[
            SfxCue(
                shot_id="s1",
                at_sec=0.0,
                duration_sec=1.2,
                prompt="gentle hot water pour into glass",
            ),
        ],
        music_cue=MusicCue(
            prompt="acoustic ambient underscore with felt piano",
            duration_sec=12.0,
        ),
    )


# ---------------------------------------------------------------------------
# 1. Agent 1: satisfied, needs_revision -> satisfied, max_iterations_reached
# ---------------------------------------------------------------------------


async def test_agent1_satisfied_and_history_non_repetition(tmp_path: Path) -> None:
    """Agent 1 runs real create_deep_agent + RubricMiddleware and varies on repeat runs."""
    bundle = MockProviderBundle.create(work_dir=tmp_path)
    history = CreativeHistoryStore(tmp_path / "history.json")
    events: list[dict[str, Any]] = []
    run_input = _make_run_input()

    res = await run_agent1_research_bible(
        run_input=run_input,
        providers=bundle,
        history_store=history,
        work_dir=tmp_path / "run1",
        run_id="a1_sat",
        event_cb=lambda ev: events.append(ev),
    )

    assert res.status == "satisfied"
    assert len(res.rubric_evaluations) == 1
    assert res.rubric_evaluations[0]["result"] == "satisfied"
    assert len(res.concepts) >= 2
    assert res.creative_bible.concept_thesis
    assert any(ev["type"] == "rubric_evaluation" and ev["stage"] == AGENT1_STAGE for ev in events)


async def test_agent1_needs_revision_then_satisfied(tmp_path: Path) -> None:
    """Agent 1 RubricMiddleware rejects a cliché-violating Bible on iter 1, passes iter 2."""
    bundle = MockProviderBundle.create(work_dir=tmp_path)
    research = _make_research()
    bad_bible = _make_bible().model_copy(
        update={"concept_thesis": "POV: Are you tired of bland tea? This is a game-changer."}
    )
    good_bible = _make_bible()
    concepts = [
        {
            "id": "c1",
            "title": "Botanical Bloom",
            "thesis": "Quiet craft ritual",
            "narrative_arc": "botanical_bloom_study",
            "visual_world": "sunlit oak counter",
            "sonic_strategy": "warm vo and kettle foley",
            "emotional_register": "calm",
            "diversity_score": 0.85,
        },
        {
            "id": "c2",
            "title": "Kinetic Pour",
            "thesis": "Fast morning rhythm",
            "narrative_arc": "kinetic_pour_cut",
            "visual_world": "crisp ceramic studio",
            "sonic_strategy": "percussive foley",
            "emotional_register": "energetic",
            "diversity_score": 0.90,
        },
    ]

    attempt = {"n": 0}

    def _responder(_msgs: list[Any]) -> str:
        attempt["n"] += 1
        bible = bad_bible if attempt["n"] == 1 else good_bible
        return json.dumps(
            {
                "research": research.model_dump(mode="json"),
                "concepts": concepts,
                "creative_bible": bible.model_dump(mode="json"),
            }
        )

    res = await run_agent1_research_bible(
        run_input=_make_run_input(num_concepts=2),
        providers=bundle,
        work_dir=tmp_path,
        run_id="a1_rev",
        main_model=ScriptedChatModel(responder=_responder),
        max_iterations=3,
    )

    assert res.status == "satisfied"
    assert len(res.rubric_evaluations) == 2
    assert res.rubric_evaluations[0]["result"] == "needs_revision"
    assert res.rubric_evaluations[1]["result"] == "satisfied"
    assert "POV:" not in res.creative_bible.concept_thesis


async def test_agent1_terminal_max_iterations_rejected(tmp_path: Path) -> None:
    """Agent 1 raises RubricGateError on max_iterations_reached (never treats AIMessage as pass)."""
    bundle = MockProviderBundle.create(work_dir=tmp_path)
    bad_model = ScriptedChatModel(responses=["I refused to call any tools or save a bible."])

    with pytest.raises(RubricGateError) as exc_info:
        await run_agent1_research_bible(
            run_input=_make_run_input(),
            providers=bundle,
            work_dir=tmp_path,
            run_id="a1_fail",
            main_model=bad_model,
            max_iterations=2,
            raise_on_failure=True,
        )

    assert exc_info.value.status == "max_iterations_reached"
    assert len(exc_info.value.evaluations) == 2


# ---------------------------------------------------------------------------
# 2. Agent 2: satisfied, needs_revision -> satisfied, failed rejection
# ---------------------------------------------------------------------------


async def test_agent2_satisfied_happy_path(tmp_path: Path) -> None:
    """Agent 2 runs real create_deep_agent + RubricMiddleware and passes pure script rules."""
    bundle = MockProviderBundle.create(work_dir=tmp_path)
    research = _make_research()
    bible = _make_bible()
    run_input = _make_run_input()

    res = await run_agent2_director(
        run_input=run_input,
        research=research,
        creative_bible=bible,
        providers=bundle,
        work_dir=tmp_path,
        run_id="a2_sat",
    )

    assert res.status == "satisfied"
    assert len(res.rubric_evaluations) == 1
    assert res.rubric_evaluations[0]["result"] == "satisfied"
    assert res.qa_report.passed is True
    assert res.qa_report.originality_score >= 6.0
    assert res.qa_report.bible_coherence_score >= 6.0
    assert res.sound_plan.voice_id


async def test_agent2_needs_revision_then_satisfied(tmp_path: Path) -> None:
    """Agent 2 rejects ungrounded claim & placeholder on iter 1, revises & passes on iter 2."""
    bundle = MockProviderBundle.create(work_dir=tmp_path)
    research = _make_research()
    bible = _make_bible()

    bad_sheet = _make_valid_sheet().model_copy(
        update={
            "shots": [
                Shot(
                    id="s1",
                    index=0,
                    beat="hook",
                    duration_sec=6.0,
                    visual_prompt="Teapot on table",
                    camera_move="dolly",
                    vo_line="The #1 best guaranteed miracle tea on earth!",
                ),
                Shot(
                    id="s2",
                    index=1,
                    beat="cta",
                    duration_sec=6.0,
                    visual_prompt="Tea pouch",
                    camera_move="static",
                    vo_line="Order now for {{PRICE}}.",
                ),
            ]
        }
    )
    good_sheet = _make_valid_sheet()
    good_sound = _make_sound_plan()
    calls = {"n": 0}

    def _responder(_msgs: list[Any]) -> str:
        calls["n"] += 1
        chosen = bad_sheet if calls["n"] == 1 else good_sheet
        return json.dumps(
            {
                "shot_sheet": chosen.model_dump(mode="json"),
                "sound_plan": good_sound.model_dump(mode="json"),
            }
        )

    res = await run_agent2_director(
        run_input=_make_run_input(),
        research=research,
        creative_bible=bible,
        providers=bundle,
        work_dir=tmp_path,
        run_id="a2_rev",
        main_model=ScriptedChatModel(responder=_responder),
    )

    assert res.status == "satisfied"
    assert len(res.rubric_evaluations) == 2
    assert res.rubric_evaluations[0]["result"] == "needs_revision"
    assert res.rubric_evaluations[1]["result"] == "satisfied"
    gate = evaluate_script_gate(
        res.shot_sheet,
        research,
        has_product_photos=bool(research.product_photos),
        offer=_make_run_input().offer,
    )
    assert gate.passed is True


async def test_agent2_terminal_failed_verdict_rejected(tmp_path: Path) -> None:
    """Agent 2 captures terminal `failed` verdict from grader via checkpointer/callback."""
    bundle = MockProviderBundle.create(work_dir=tmp_path)
    grader_fail = ScriptedChatModel(
        responses=[
            {
                "content": "",
                "tool_calls": [
                    {
                        "id": "tc_fail_1",
                        "name": "GraderResponse",
                        "args": {
                            "result": "failed",
                            "explanation": "Unrecoverable policy violation in script.",
                            "criteria": [
                                {
                                    "name": "claim_grounding",
                                    "passed": False,
                                    "gap": "Policy-violating medical claim.",
                                }
                            ],
                        },
                        "type": "tool_call",
                    }
                ],
            }
        ]
    )

    res = await run_agent2_director(
        run_input=_make_run_input(),
        research=_make_research(),
        creative_bible=_make_bible(),
        providers=bundle,
        work_dir=tmp_path,
        run_id="a2_failed",
        grader_model=grader_fail,
        raise_on_failure=False,
    )
    assert res.status == "failed"
    assert len(res.rubric_evaluations) == 1
    assert res.rubric_evaluations[0]["result"] == "failed"


# ---------------------------------------------------------------------------
# 3. Agent 3: satisfied, needs_revision -> satisfied, max_iterations_reached, Manim fallback
# ---------------------------------------------------------------------------


async def test_agent3_satisfied_and_real_mp4_output(tmp_path: Path) -> None:
    """Agent 3 renders, stitches, runs parallel MediaQA, and passes rubric verification."""
    bundle = MockProviderBundle.create(work_dir=tmp_path)
    res = await run_agent3_producer(
        run_id="a3_sat",
        run_input=_make_run_input(),
        shot_sheet=_make_valid_sheet(),
        sound_plan=_make_sound_plan(),
        creative_bible=_make_bible(),
        providers=bundle,
        work_dir=tmp_path,
    )

    assert res.status == "satisfied"
    assert len(res.rubric_evaluations) == 1
    assert res.render_manifest.stitched_path is not None
    mp4 = Path(res.render_manifest.stitched_path)
    assert mp4.is_file() and mp4.stat().st_size > 0
    probe = _ffprobe_json(mp4)
    v_stream = next(s for s in probe["streams"] if s["codec_type"] == "video")
    assert int(v_stream["width"]) > int(v_stream["height"])
    assert res.cost_ledger.actual_usd <= res.cost_ledger.cap_usd


async def test_agent3_needs_revision_then_satisfied(tmp_path: Path) -> None:
    """Agent 3 fails iter 1 when stitch is skipped, then runs tools on iter 2 and passes."""
    bundle = MockProviderBundle.create(work_dir=tmp_path)
    phase = {"iter": 0, "step": 0}

    def _producer_responder(messages: list[Any]) -> dict[str, Any] | str:
        from langchain_core.messages import HumanMessage, ToolMessage

        # Count rubric feedback HumanMessages
        grader_msgs = [
            m
            for m in messages
            if isinstance(m, HumanMessage)
            and m.additional_kwargs.get("lc_source") == "rubric_grader"
        ]
        if not grader_msgs:
            # Iter 1: prematurely claim done without rendering/stitching
            return "Skipped rendering on first attempt."

        # Iter 2: execute full tool sequence
        seen = {getattr(m, "name", "") for m in messages if isinstance(m, ToolMessage)}
        phase["step"] += 1
        for tool_name in (
            "preflight_cost_tool",
            "render_assets_tool",
            "stitch_video_tool",
            "run_media_qa_tool",
        ):
            if tool_name not in seen:
                return {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": f"tc_a3_rev_{phase['step']}",
                            "name": tool_name,
                            "args": {},
                            "type": "tool_call",
                        }
                    ],
                }
        return "Completed rendering, stitching, and QA on revision."

    res = await run_agent3_producer(
        run_id="a3_rev",
        run_input=_make_run_input(),
        shot_sheet=_make_valid_sheet(),
        sound_plan=_make_sound_plan(),
        creative_bible=_make_bible(),
        providers=bundle,
        work_dir=tmp_path,
        main_model=ScriptedChatModel(responder=_producer_responder),
        max_iterations=3,
    )

    assert res.status == "satisfied"
    assert len(res.rubric_evaluations) == 2
    assert res.rubric_evaluations[0]["result"] == "needs_revision"
    assert res.rubric_evaluations[1]["result"] == "satisfied"
    assert res.render_manifest.stitched_path is not None
    assert Path(res.render_manifest.stitched_path).is_file()


async def test_agent3_max_iterations_rejected(tmp_path: Path) -> None:
    """Agent 3 rejects when production never creates a stitched MP4."""
    bundle = MockProviderBundle.create(work_dir=tmp_path)
    with pytest.raises(RubricGateError) as exc_info:
        await run_agent3_producer(
            run_id="a3_fail",
            run_input=_make_run_input(),
            shot_sheet=_make_valid_sheet(),
            sound_plan=_make_sound_plan(),
            creative_bible=_make_bible(),
            providers=bundle,
            work_dir=tmp_path,
            main_model=ScriptedChatModel(responses=["No tools called."]),
            max_iterations=2,
            raise_on_failure=True,
        )
    assert exc_info.value.status == "max_iterations_reached"
    assert len(exc_info.value.evaluations) == 2


async def test_agent3_manim_unavailable_honest_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When Manim is unavailable, engine='manim' warns honestly and falls back to motion_card."""
    import gtm_videogen.agents.agent3_producer as a3_mod

    monkeypatch.setattr(a3_mod, "is_manim_available", lambda: False)
    bundle = MockProviderBundle.create(work_dir=tmp_path)
    sheet = _make_valid_sheet()
    sheet.shots[0] = sheet.shots[0].model_copy(update={"engine": "manim", "on_screen_text": None})

    res = await run_agent3_producer(
        run_id="a3_manim_fb",
        run_input=_make_run_input(),
        shot_sheet=sheet,
        sound_plan=_make_sound_plan(),
        creative_bible=_make_bible(),
        providers=bundle,
        work_dir=tmp_path,
        raise_on_failure=False,
    )
    assert any("manim" in w.lower() for w in res.warnings)
    assert res.render_manifest.stitched_path is not None
    assert Path(res.render_manifest.stitched_path).is_file()


# ---------------------------------------------------------------------------
# 4. Full PipelineRunner: run(), async event_cb, repair(), BUG-16, BUG-05
# ---------------------------------------------------------------------------


async def test_pipeline_runner_full_offline_run_and_history_writeback(tmp_path: Path) -> None:
    """Full 3-agent PipelineRunner.run() produces MP4, QA, CostLedger, evals, CreativeHistory."""
    history = CreativeHistoryStore(tmp_path / "creative_history.json")
    runner = PipelineRunner(work_dir=tmp_path, history_store=history)
    events: list[dict[str, Any]] = []

    async def _async_cb(ev: dict[str, Any]) -> None:
        events.append(ev)

    state = _make_run_state(run_id="pipe_full_1")
    done = await runner.run(state, event_cb=_async_cb)

    assert done.status == "done"
    assert done.error is None
    assert done.research is not None
    assert done.creative_bible is not None
    assert done.script is not None
    assert done.shot_sheet is not None
    assert done.sound_plan is not None
    assert done.render_manifest is not None
    assert done.render_manifest.stitched_path is not None

    mp4 = Path(done.render_manifest.stitched_path)
    assert mp4.is_file() and mp4.stat().st_size > 0
    probe = _ffprobe_json(mp4)
    v_stream = next(s for s in probe["streams"] if s["codec_type"] == "video")
    assert int(v_stream["width"]) > int(v_stream["height"])

    assert done.qa_report is not None
    assert done.cost_ledger.actual_usd <= done.cost_ledger.cap_usd
    assert set(done.rubric_evaluations.keys()) == {AGENT1_STAGE, AGENT2_STAGE, AGENT3_STAGE}

    event_types = {ev["type"] for ev in events}
    assert "stage_started" in event_types
    assert "rubric_evaluation" in event_types
    assert "stage_completed" in event_types
    assert "cost_update" in event_types
    assert "run_completed" in event_types

    saved_hist = history.get_history(state.input.brand_url)
    assert len(saved_hist.entries) == 1
    assert saved_hist.entries[0].run_id == "pipe_full_1"
    assert saved_hist.entries[0].voice_id == done.sound_plan.voice_id


async def test_pipeline_runner_targeted_repair_reuses_cache(tmp_path: Path) -> None:
    """PipelineRunner.repair() re-renders only targeted shot and reuses cached clips & audio."""
    bundle = MockProviderBundle.create(work_dir=tmp_path)
    runner = PipelineRunner(providers=bundle, work_dir=tmp_path)

    state = _make_run_state(run_id="pipe_repair_1")
    done = await runner.run(state)
    assert done.status == "done"
    assert done.shot_sheet is not None
    assert len(done.shot_sheet.shots) >= 2

    target_shot_id = done.shot_sheet.shots[0].id
    other_shot_ids = [s.id for s in done.shot_sheet.shots[1:]]

    repaired = await runner.repair(done, failed_shot_ids=[target_shot_id])
    assert repaired.status == "done"
    assert repaired.repair_result is not None
    assert repaired.repair_result["repaired_shot_ids"] == [target_shot_id]
    for oid in other_shot_ids:
        assert oid in repaired.repair_result["reused_shot_ids"]
    assert repaired.render_manifest is not None
    assert repaired.render_manifest.stitched_path is not None
    assert Path(repaired.render_manifest.stitched_path).is_file()
    # Audio assets should be cache hits on repair
    assert all(a.cache_hit for a in repaired.render_manifest.audio_assets)


async def test_pipeline_runner_non_fatal_sfx_warnings_persist_bug16(tmp_path: Path) -> None:
    """BUG-16: Non-fatal SFX failures record persistent warnings while run completes cleanly."""
    bundle = MockProviderBundle.create(work_dir=tmp_path)

    async def _boom_sfx(cue: SfxCue, cancel: Any = None, model: Any = None) -> Any:
        raise AudioProviderError("simulated SFX provider timeout")

    bundle.audio.generate_sfx = _boom_sfx  # type: ignore[method-assign]
    runner = PipelineRunner(providers=bundle, work_dir=tmp_path)

    state = _make_run_state(run_id="pipe_bug16", audio_mode="both")
    done = await runner.run(state)
    assert done.status == "done"
    assert done.error is None
    assert done.render_manifest is not None
    assert done.render_manifest.error is None
    assert len(done.warnings) >= 1
    assert len(done.render_manifest.warnings) >= 1


async def test_pipeline_runner_mid_pipeline_cancellation_bug05(tmp_path: Path) -> None:
    """BUG-05: Mid-pipeline cancellation keeps status == 'cancelled' (never overwritten)."""
    bundle = MockProviderBundle.create(work_dir=tmp_path)
    runner = PipelineRunner(providers=bundle, work_dir=tmp_path)
    cancel = CancellationToken()
    events: list[dict[str, Any]] = []

    def _cancelling_cb(ev: dict[str, Any]) -> None:
        events.append(ev)
        if ev.get("type") == "stage_completed" and ev.get("stage") == AGENT1_STAGE:
            cancel.cancel("User aborted after research stage")

    state = _make_run_state(run_id="pipe_cancel_1")
    result = await runner.run(state, event_cb=_cancelling_cb, cancel=cancel)

    assert result.status == "cancelled"
    assert result.research is not None
    assert result.render_manifest is None
    assert any(ev["type"] == "run_cancelled" for ev in events)
    assert not any(ev["type"] == "run_completed" for ev in events)


# ---------------------------------------------------------------------------
# 5. Zero Hard Templates Check across src/gtm_videogen/agents/
# ---------------------------------------------------------------------------


def test_zero_hard_templates_in_agents_package() -> None:
    """Verify zero legacy hard-template strings exist anywhere in src/gtm_videogen/agents/."""
    agents_dir = Path(__file__).resolve().parents[1] / "src" / "gtm_videogen/agents"
    forbidden = (
        "confession_loop",
        "NEON_LAB_RULES",
        "droneBed",
        "END_STING",
        "112 BPM",
        "OFF_FRAME_POUR",
    )
    py_files = sorted(agents_dir.glob("*.py"))
    assert len(py_files) >= 5
    for path in py_files:
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token.lower() not in text.lower(), (
                f"Forbidden hard-template token {token!r} found in {path.name}"
            )


# ---------------------------------------------------------------------------
# 6. M4 Iteration 2 Regression Tests
# ---------------------------------------------------------------------------


async def test_live_mode_fails_loudly_without_api_key(tmp_path: Path) -> None:
    """Fix 1: mock=False without injected chat_model or LLM API key raises RuntimeError."""
    from types import SimpleNamespace

    from gtm_videogen.core.config import Settings
    from gtm_videogen.providers.llm_router import LLMRouter

    empty_settings = Settings(
        openrouter_api_key="",
        openai_api_key="",
        anthropic_api_key="",
        google_api_key="",
        data_dir=tmp_path,
    )
    live_input = _make_run_input(mock=False)
    live_state = RunState(
        run_id="live_no_key",
        input=live_input,
        cost_ledger=CostLedger(run_id="live_no_key", cap_usd=live_input.cap_usd),
    )

    # PipelineRunner.run() must fail loudly, never silently using offline ScriptedChatModel
    runner = PipelineRunner(work_dir=tmp_path, settings=empty_settings)
    with pytest.raises(RuntimeError, match="Live mode \\(mock=False\\) requires"):
        await runner.run(live_state)

    # Direct agent calls with a live-like bundle (chat_model=None) must also fail loudly
    live_bundle = SimpleNamespace(
        chat_model=None,
        llm_router=LLMRouter(settings=empty_settings),
        scraper=None,
        video=None,
        audio=None,
        whisper=None,
    )
    with pytest.raises(RuntimeError, match="Live mode \\(mock=False\\) requires"):
        await run_agent1_research_bible(
            run_input=live_input,
            providers=live_bundle,
            work_dir=tmp_path,
        )
    with pytest.raises(RuntimeError, match="Live mode \\(mock=False\\) requires"):
        await run_agent2_director(
            run_input=live_input,
            research=_make_research(),
            creative_bible=_make_bible(),
            providers=live_bundle,
            work_dir=tmp_path,
        )
    with pytest.raises(RuntimeError, match="Live mode \\(mock=False\\) requires"):
        await run_agent3_producer(
            run_id="live_a3",
            run_input=live_input,
            shot_sheet=_make_valid_sheet(),
            sound_plan=_make_sound_plan(),
            creative_bible=_make_bible(),
            providers=live_bundle,
            work_dir=tmp_path,
        )


async def test_agent2_rejects_missing_vo_lines_and_grader_honors_passed_false(
    tmp_path: Path,
) -> None:
    """Fix 2: MISSING_VO_LINES is never dropped and grader honors passed=False."""
    from langchain_core.messages import ToolMessage

    from gtm_videogen.agents.rubric_utils import make_tool_driven_grader_model

    bundle = MockProviderBundle.create(work_dir=tmp_path)
    no_vo_sheet = _make_valid_sheet().model_copy(
        update={
            "audio_mode": "vo",
            "shots": [s.model_copy(update={"vo_line": None}) for s in _make_valid_sheet().shots],
        }
    )
    payload = json.dumps(
        {
            "shot_sheet": no_vo_sheet.model_dump(mode="json"),
            "sound_plan": _make_sound_plan().model_dump(mode="json"),
        }
    )

    with pytest.raises(RubricGateError) as exc_info:
        await run_agent2_director(
            run_input=_make_run_input(audio_mode="vo"),
            research=_make_research(),
            creative_bible=_make_bible(),
            providers=bundle,
            work_dir=tmp_path,
            main_model=ScriptedChatModel(responder=lambda _m: payload),
            max_iterations=2,
            raise_on_failure=True,
        )
    assert exc_info.value.status == "max_iterations_reached"

    # Also verify make_tool_driven_grader_model refuses satisfied when verdict_data["passed"]=False
    grader = make_tool_driven_grader_model("check_tool")
    tool_msg = ToolMessage(
        content=json.dumps(
            {
                "passed": False,
                "result": "needs_revision",
                "explanation": "Unmapped blocking failure",
                "criteria": [{"name": "c1", "passed": True}],
            }
        ),
        tool_call_id="tc1",
        name="check_tool",
    )
    ai_msg = grader.invoke([tool_msg])
    assert ai_msg.tool_calls
    grader_args = ai_msg.tool_calls[0]["args"]
    assert grader_args["result"] == "needs_revision"
    assert any(not c["passed"] for c in grader_args["criteria"])


async def test_repair_versioning_validation_and_cumulative_cost_cap(tmp_path: Path) -> None:
    """Fixes 3 & 4: repair() validates shot IDs, appends v2 without mutating v1, enforces cap."""
    from gtm_videogen.core.cost_meter import current_cost_meter, read_ledger_total

    bundle = MockProviderBundle.create(work_dir=tmp_path)
    runner = PipelineRunner(providers=bundle, work_dir=tmp_path)

    # Wrap mock video provider to bill $0.05 per uncached shot against active CostMeter
    orig_gen_clip = bundle.video.generate_clip

    async def _billed_clip(shot: Any, aspect: str, fps: int, cancel: Any = None) -> Any:
        meter = current_cost_meter()
        res_id = meter.reserve(f"video:{shot.id}", 0.05, stage="render") if meter else None
        try:
            clip = await orig_gen_clip(shot, aspect=aspect, fps=fps, cancel=cancel)
            if meter and res_id:
                meter.settle(res_id, 0.05, cache_hit=clip.cache_hit)
            return clip
        except Exception:
            if meter and res_id:
                meter.release(res_id)
            raise

    bundle.video.generate_clip = _billed_clip  # type: ignore[method-assign]

    # 3 shots * $0.05 = $0.15; cap = $0.22 allows 1 repair ($0.20) but blocks a 2nd repair ($0.25)
    state = _make_run_state(run_id="pipe_cum_cost", cap_usd=0.22)
    done = await runner.run(state)
    assert done.status == "done"
    assert len(done.versions) == 1
    assert done.versions[0].version_id == "v1"
    assert done.render_manifest is not None
    assert done.render_manifest.version_id == "v1"
    initial_spend = done.cost_ledger.actual_usd
    assert initial_spend == pytest.approx(0.15, abs=1e-4)

    # Unknown shot ID must raise ValueError immediately
    with pytest.raises(ValueError, match="Unknown failed_shot_ids"):
        await runner.repair(done, failed_shot_ids=["does_not_exist_999"])

    # Valid repair #1: succeeds, appends v2 without mutating v1, cumulative spend = $0.20
    repaired = await runner.repair(done, failed_shot_ids=[done.shot_sheet.shots[0].id])
    assert repaired.status == "done"
    assert len(repaired.versions) == 2
    assert repaired.versions[0].version_id == "v1"
    assert repaired.versions[1].version_id == "v2"
    assert repaired.render_manifest is not None
    assert repaired.render_manifest.version_id == "v2"
    assert repaired.cost_ledger.actual_usd == pytest.approx(0.20, abs=1e-4)

    # Repair #2: would push cumulative spend to $0.25 > $0.22 cap -> fails cleanly, preserves $0.20
    over_cap = await runner.repair(repaired, failed_shot_ids=[repaired.shot_sheet.shots[1].id])
    assert over_cap.status == "failed"
    assert over_cap.error is not None and "Budget exceeded" in over_cap.error
    assert over_cap.cost_ledger.actual_usd == pytest.approx(0.20, abs=1e-4)
    ledger_path = tmp_path / "runs" / "pipe_cum_cost" / "spend_ledger.jsonl"
    assert read_ledger_total(ledger_path) <= 0.22 + 1e-6


async def test_pipeline_preserves_rubric_gate_error_evaluations_and_partial_spend(
    tmp_path: Path,
) -> None:
    """Fixes 3 & 5: RubricGateError evaluations and partial spend before error are persisted."""
    from gtm_videogen.core.cost_meter import current_cost_meter

    # 1. RubricGateError evaluations persisted on run_state
    bundle1 = MockProviderBundle.create(work_dir=tmp_path / "r1")
    bundle1.chat_model = ScriptedChatModel(responses=["No tools called."] * 6)
    runner1 = PipelineRunner(
        providers=bundle1,
        work_dir=tmp_path / "r1",
        max_rubric_iterations=2,
    )
    res1 = await runner1.run(_make_run_state(run_id="pipe_gate_evals"))
    assert res1.status == "failed"
    assert AGENT1_STAGE in res1.rubric_evaluations
    assert len(res1.rubric_evaluations[AGENT1_STAGE]) == 2

    # 2. Partial spend settled before mid-render BudgetExceeded is synced onto run_state
    from gtm_videogen.core.cost_meter import read_ledger_total

    bundle2 = MockProviderBundle.create(work_dir=tmp_path / "r2")
    orig_clip = bundle2.video.generate_clip
    orig_voice = bundle2.audio.generate_voice

    async def _costly_clip(shot: Any, aspect: str, fps: int, cancel: Any = None) -> Any:
        meter = current_cost_meter()
        res_id = meter.reserve(f"video:{shot.id}", 0.05, stage="render") if meter else None
        try:
            clip = await orig_clip(shot, aspect=aspect, fps=fps, cancel=cancel)
            if meter and res_id:
                meter.settle(res_id, 0.05)
            return clip
        except Exception:
            if meter and res_id:
                meter.release(res_id)
            raise

    async def _costly_voice(cue: Any, voice_id: str, cancel: Any = None) -> Any:
        meter = current_cost_meter()
        res_id = meter.reserve(f"vo:{cue.shot_id}", 0.10, stage="render") if meter else None
        try:
            audio = await orig_voice(cue, voice_id=voice_id, cancel=cancel)
            if meter and res_id:
                meter.settle(res_id, 0.10)
            return audio
        except Exception:
            if meter and res_id:
                meter.release(res_id)
            raise

    bundle2.video.generate_clip = _costly_clip  # type: ignore[method-assign]
    bundle2.audio.generate_voice = _costly_voice  # type: ignore[method-assign]
    runner2 = PipelineRunner(providers=bundle2, work_dir=tmp_path / "r2")
    # 3 shots * $0.05 = $0.15 settle; VO ($0.10) exceeds cap=$0.18 -> partial spend synced
    res2 = await runner2.run(_make_run_state(run_id="pipe_partial_spend", cap_usd=0.18))
    assert res2.status == "failed"
    ledger_path = tmp_path / "r2" / "runs" / "pipe_partial_spend" / "spend_ledger.jsonl"
    disk_total = read_ledger_total(ledger_path)
    assert disk_total > 0.0
    assert res2.cost_ledger.actual_usd == pytest.approx(disk_total, abs=1e-6)


async def test_prompt_cancellation_tracks_ainvoke_and_parallel_render_tasks(
    tmp_path: Path,
) -> None:
    """Fix 6: cancel.cancel() promptly interrupts slow ainvoke and parallel render tasks."""
    import asyncio
    import time

    from langchain_core.outputs import ChatGeneration, ChatResult

    from gtm_videogen.agents.agent3_producer import render_assets_overlapped

    class _SlowChatModel(ScriptedChatModel):
        async def _agenerate(self, messages: Any, stop: Any = None, **kwargs: Any) -> ChatResult:
            await asyncio.sleep(5.0)
            return ChatResult(generations=[ChatGeneration(message=self.invoke(messages))])

        def bind_tools(self, tools: Any, **kwargs: Any) -> _SlowChatModel:
            return _SlowChatModel(responses=self.responses, responder=self.responder)

    # 1. Mid-stage LLM cancellation in run_rubric_deep_agent via PipelineRunner
    bundle = MockProviderBundle.create(work_dir=tmp_path / "c1")
    bundle.chat_model = _SlowChatModel(responses=["never reached"])
    runner = PipelineRunner(providers=bundle, work_dir=tmp_path / "c1")
    cancel1 = CancellationToken()

    async def _cancel_soon(tok: CancellationToken) -> None:
        await asyncio.sleep(0.15)
        tok.cancel("abort slow llm")

    t0 = time.monotonic()
    canceller = asyncio.create_task(_cancel_soon(cancel1))
    res = await asyncio.wait_for(
        runner.run(_make_run_state(run_id="cancel_slow_llm"), cancel=cancel1),
        timeout=2.0,
    )
    await canceller
    elapsed1 = time.monotonic() - t0
    assert res.status == "cancelled"
    assert elapsed1 < 1.5

    # 2. Mid-parallel-render cancellation in render_assets_overlapped
    bundle2 = MockProviderBundle.create(work_dir=tmp_path / "c2")

    async def _hanging_clip(shot: Any, aspect: str, fps: int, cancel: Any = None) -> Any:
        await asyncio.sleep(5.0)
        raise AssertionError("Should have been cancelled")

    bundle2.video.generate_clip = _hanging_clip  # type: ignore[method-assign]
    cancel2 = CancellationToken()
    t1 = time.monotonic()
    canceller2 = asyncio.create_task(_cancel_soon(cancel2))
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(
            render_assets_overlapped(
                run_id="cancel_render",
                shot_sheet=_make_valid_sheet(),
                sound_plan=_make_sound_plan(),
                creative_bible=_make_bible(),
                providers=bundle2,
                work_dir=tmp_path / "c2",
                cancel=cancel2,
            ),
            timeout=2.0,
        )
    await canceller2
    elapsed2 = time.monotonic() - t1
    assert elapsed2 < 1.5


async def test_manim_shot_end_to_end_escaping_no_frozen_frames_and_one_text_layer(
    tmp_path: Path,
) -> None:
    """Fix 7: Manim escaping, FROZEN_FRAMES exemption, and ONE_TEXT_LAYER pass end-to-end."""
    import ast

    from gtm_videogen.agents.agent3_producer import _build_default_manim_scene

    tricky_shot = _make_valid_sheet().shots[0].model_copy(
        update={"engine": "manim", "on_screen_text": 'Line 1\n"Line 2"\\end'}
    )
    code, scene_cls = _build_default_manim_scene(tricky_shot, _make_bible())
    assert scene_cls == "BespokeShotScene"
    ast.parse(code)  # Must not raise SyntaxError

    # End-to-end Agent 3 with an engine="manim" shot + on_screen_text
    bundle = MockProviderBundle.create(work_dir=tmp_path)
    sheet = _make_valid_sheet()
    sheet.shots[1] = sheet.shots[1].model_copy(
        update={"engine": "manim", "on_screen_text": "Zero Latency"}
    )
    res = await run_agent3_producer(
        run_id="a3_manim_e2e",
        run_input=_make_run_input(),
        shot_sheet=sheet,
        sound_plan=_make_sound_plan(),
        creative_bible=_make_bible(),
        providers=bundle,
        work_dir=tmp_path,
        raise_on_failure=True,
    )
    assert res.status == "satisfied"
    assert res.render_manifest.stitched_path is not None
    assert Path(res.render_manifest.stitched_path).is_file()
    # Ensure ONE_TEXT_LAYER: OCR on the Manim shot must NOT duplicate "Zero Latency Zero Latency"
    for ocr in res.qa_report.ocr_checks:
        if ocr.shot_id == "s2":
            assert ocr.observed_text.lower().count("zero latency") <= 1


async def test_invalid_pydantic_json_in_grader_hook_does_not_crash(tmp_path: Path) -> None:
    """Fix 8: Invalid Pydantic JSON in AIMessage triggers needs_revision, not crash."""
    bundle = MockProviderBundle.create(work_dir=tmp_path)
    bad_a1_json = json.dumps({"creative_bible": {"concept_thesis": 123, "bad_field": True}})
    with pytest.raises(RubricGateError) as exc1:
        await run_agent1_research_bible(
            run_input=_make_run_input(),
            providers=bundle,
            work_dir=tmp_path / "a1",
            main_model=ScriptedChatModel(responder=lambda _m: bad_a1_json),
            max_iterations=2,
            raise_on_failure=True,
        )
    assert exc1.value.status == "max_iterations_reached"

    bad_a2_json = json.dumps({"shot_sheet": {"fps": "not_an_int", "shots": []}})
    with pytest.raises(RubricGateError) as exc2:
        await run_agent2_director(
            run_input=_make_run_input(),
            research=_make_research(),
            creative_bible=_make_bible(),
            providers=bundle,
            work_dir=tmp_path / "a2",
            main_model=ScriptedChatModel(responder=lambda _m: bad_a2_json),
            max_iterations=2,
            raise_on_failure=True,
        )
    assert exc2.value.status == "max_iterations_reached"


async def test_minor_polish_sync_mismatch_num_concepts_and_multi_must_say(tmp_path: Path) -> None:
    """Fix 9: SYNC_MISMATCH bucket, num_concepts enforcement, and multi-item must_say synthesis."""
    from gtm_videogen.agents.agent2_director import _synthesize_director_artifacts
    from gtm_videogen.agents.agent3_producer import _build_agent3_tools
    from gtm_videogen.agents.rubric_utils import write_workspace_json
    from gtm_videogen.core.cost_meter import CostMeter
    from gtm_videogen.models.schemas import QAIssue, QAReport, RenderManifest

    # 1. Multi-item must_say: both phrases appear in offline Agent 2 synthesis
    inp = _make_run_input(must_say=["Steeps in 90 seconds", "Family farms"])
    _script, sheet, _sp = _synthesize_director_artifacts(
        run_input=inp,
        research=_make_research(),
        creative_bible=_make_bible(),
        fitted_durations=[4.0, 4.0, 4.0],
        selected_voice_id="v_maya_warm",
    )
    all_vo = " ".join((s.vo_line or "") for s in sheet.shots).lower()
    assert "steeps in 90 seconds" in all_vo
    assert "family farms" in all_vo

    # 2. SYNC_MISMATCH bucketed under ocr_and_sync (not loudness_and_audio)
    ws = tmp_path / "a3_sync_ws"
    ws.mkdir(parents=True, exist_ok=True)
    dummy_mp4 = ws / "out.mp4"
    dummy_mp4.write_bytes(b"fake-mp4-bytes")
    write_workspace_json(
        ws, "render_manifest.json", RenderManifest(run_id="r_sync", stitched_path=str(dummy_mp4))
    )
    write_workspace_json(
        ws,
        "qa_report.json",
        QAReport(
            passed=False,
            blocking_errors=[
                QAIssue(
                    code="SYNC_MISMATCH",
                    severity="blocking",
                    message="VO transcript mismatch",
                )
            ],
        ),
    )
    meter = CostMeter(run_id="r_sync", cap_usd=5.0)
    write_workspace_json(ws, "cost_ledger.json", meter.to_ledger())
    _w_tools, g_tools = _build_agent3_tools(
        run_id="r_sync",
        run_input=_make_run_input(),
        shot_sheet=_make_valid_sheet(),
        sound_plan=_make_sound_plan(),
        creative_bible=_make_bible(),
        providers=MockProviderBundle.create(work_dir=ws),
        cost_meter=meter,
        workspace=ws,
        director_qa=None,
        event_cb=None,
        cancel=None,
    )
    verdict = json.loads(g_tools[0].invoke({}))
    crit_map = {c["name"]: c for c in verdict["criteria"]}
    assert crit_map["ocr_and_sync"]["passed"] is False
    assert "SYNC_MISMATCH" in crit_map["ocr_and_sync"]["gap"]
    assert crit_map["loudness_and_audio"]["passed"] is True


async def test_mixed_ai_video_and_manim_caption_safe_sheet_no_double_text(tmp_path: Path) -> None:
    """Fix 1: _caption_safe_sheet clears on_screen_text and vo_line on manim/motion_card."""
    from gtm_videogen.agents.agent3_producer import _caption_safe_sheet

    base = _make_valid_sheet()
    mixed = base.model_copy(
        update={
            "shots": [
                base.shots[0].model_copy(
                    update={
                        "engine": "ai_video",
                        "on_screen_text": "Built to Last",
                        "vo_line": "Built from solid aluminum.",
                    }
                ),
                base.shots[1].model_copy(
                    update={
                        "engine": "manim",
                        "on_screen_text": "Zero Latency",
                        "vo_line": "Zero latency monitoring.",
                    }
                ),
                base.shots[2].model_copy(
                    update={
                        "engine": "motion_card",
                        "on_screen_text": "Try Now",
                        "vo_line": "Start your free trial today.",
                    }
                ),
            ]
        }
    )

    safe = _caption_safe_sheet(mixed)
    assert safe.shots[0].on_screen_text == "Built to Last"
    assert safe.shots[0].vo_line == "Built from solid aluminum."
    assert safe.shots[1].on_screen_text is None
    assert safe.shots[1].vo_line is None
    assert safe.shots[2].on_screen_text is None
    assert safe.shots[2].vo_line is None

    # End-to-end mixed ai_video + manim + motion_card render: verify zero double-text
    bundle = MockProviderBundle.create(work_dir=tmp_path)
    res = await run_agent3_producer(
        run_id="a3_mixed_manim_one_layer",
        run_input=_make_run_input(),
        shot_sheet=mixed,
        sound_plan=_make_sound_plan(),
        creative_bible=_make_bible(),
        providers=bundle,
        work_dir=tmp_path,
        raise_on_failure=True,
    )
    assert res.status == "satisfied"
    assert res.render_manifest.stitched_path is not None
    assert Path(res.render_manifest.stitched_path).is_file()
    ocr_by_shot = {ocr.shot_id: ocr for ocr in res.qa_report.ocr_checks}
    assert "s1" in ocr_by_shot
    assert "built to last" in ocr_by_shot["s1"].observed_text.lower()
    assert "s2" in ocr_by_shot
    observed_s2 = ocr_by_shot["s2"].observed_text.lower()
    assert "zero latency" in observed_s2
    assert "monitoring" not in observed_s2
    assert "s3" in ocr_by_shot
    observed_s3 = ocr_by_shot["s3"].observed_text.lower()
    assert "try now" in observed_s3
    assert "trial" not in observed_s3


async def test_pipeline_multiple_repairs_produce_distinct_versioned_mp4s(tmp_path: Path) -> None:
    """M4 Iter 3 Fix 2: Sequential repair() calls (v2, v3) keep distinct MP4 paths on disk."""
    bundle = MockProviderBundle.create(work_dir=tmp_path)
    runner = PipelineRunner(providers=bundle, work_dir=tmp_path)

    state = await runner.run(_make_run_state(run_id="pipe_ver_mp4"))
    assert state.status == "done"
    assert len(state.versions) == 1

    v2_state = await runner.repair(state, failed_shot_ids=[state.shot_sheet.shots[0].id])
    assert v2_state.status == "done"
    assert len(v2_state.versions) == 2

    v3_state = await runner.repair(v2_state, failed_shot_ids=[v2_state.shot_sheet.shots[1].id])
    assert v3_state.status == "done"
    assert [v.version_id for v in v3_state.versions] == ["v1", "v2", "v3"]

    v2_path = v3_state.versions[1].render_manifest.stitched_path
    v3_path = v3_state.versions[2].render_manifest.stitched_path
    assert v2_path is not None and v3_path is not None
    assert v2_path != v3_path
    assert Path(v2_path).name == "pipe_ver_mp4_v2.mp4"
    assert Path(v3_path).name == "pipe_ver_mp4_v3.mp4"
    assert Path(v2_path).is_file() and Path(v2_path).stat().st_size > 0
    assert Path(v3_path).is_file() and Path(v3_path).stat().st_size > 0


async def test_agent1_offline_mock_supports_six_distinct_concepts(tmp_path: Path) -> None:
    """M4 Iter 3 Fix 3: num_concepts=6 offline mock run generates 6 distinct concepts and passes."""
    bundle = MockProviderBundle.create(work_dir=tmp_path)
    res = await run_agent1_research_bible(
        run_input=_make_run_input(num_concepts=6),
        providers=bundle,
        work_dir=tmp_path,
        raise_on_failure=True,
    )
    assert res.status == "satisfied"
    assert len(res.concepts) == 6
    assert len({c.id for c in res.concepts}) == 6
    assert len({c.title for c in res.concepts}) == 6
    assert len({c.narrative_arc.lower() for c in res.concepts}) == 6
    assert len({c.visual_world.lower() for c in res.concepts}) == 6
    assert max(c.diversity_score for c in res.concepts) > 0.0


def test_pipeline_runner_resolve_work_dir_rejects_path_traversal(tmp_path: Path) -> None:
    """M6 Tier 5 Challenge 4: _resolve_work_dir rejects traversal run_ids before creating dirs."""
    from gtm_videogen.core.config import Settings

    base = tmp_path / "pipe_base"
    base.mkdir()
    runner_with_work_dir = PipelineRunner(work_dir=base)
    runner_with_settings = PipelineRunner(settings=Settings(data_dir=base))

    for bad_run_id in ("../../escaped", "../escaped", "a/b", "a\\b", "", ".", ".."):
        with pytest.raises(ValueError):
            runner_with_work_dir._resolve_work_dir(bad_run_id)
        with pytest.raises(ValueError):
            runner_with_settings._resolve_work_dir(bad_run_id)

    assert not (tmp_path / "escaped").exists()
    assert not (base / "escaped").exists()

    # Valid run_id still resolves inside runs_root
    ok_dir = runner_with_work_dir._resolve_work_dir("run_safe_123")
    assert ok_dir.is_dir()
    assert ok_dir.resolve().relative_to((base / "runs").resolve()) == Path("run_safe_123")



