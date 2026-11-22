"""End-to-end tests for Gemini CLI integration, 60s 7-Act Story Engine, and 0-100 Video QA Loop."""

from __future__ import annotations

from pathlib import Path

import pytest

from gtm_videogen.core.config import Settings
from gtm_videogen.models.model_registry import ModelRegistryStore
from gtm_videogen.providers.gemini_cli import (
    extract_json_object,
    gemini_analyze_media_json,
    gemini_generate_image,
    gemini_generate_json,
    gemini_generate_tts,
    gemini_generate_video,
    resolve_gemini_cli_bin,
)


@pytest.mark.asyncio
async def test_loop1_gemini_cli_dry_run_contracts_and_registry(tmp_path: Path) -> None:
    """Verify google-gemini/gemini-api-cli wrapper contracts via real --dry-run JSON previews."""
    settings = Settings(data_dir=tmp_path / "data", gemini_api_key="test_gemini_key")
    reg = ModelRegistryStore(tmp_path / "stitch")
    models = reg.list_video_models()
    gem_spec = reg.get_video_model("gemini-omni-1.2-flash")
    assert gem_spec.provider == "gemini"
    assert any(m.id == "gemini-omni-1.2-flash" for m in models)
    assert settings.effective_gemini_key() == "test_gemini_key"

    # JSON extraction handles fenced and prose-wrapped JSON cleanly
    parsed = extract_json_object("Here is the output:\n```json\n{\"banger_score\": 94}\n```")
    assert parsed == {"banger_score": 94}

    cli_bin = resolve_gemini_cli_bin(settings)
    if cli_bin is None:
        pytest.skip("gemini-api binary not installed in test environment")

    gen_preview = await gemini_generate_json(
        system_prompt="Master Director",
        user_prompt="Create 7 acts",
        model="gemini-cli/gemini-4-hydrogen",
        settings=settings,
        dry_run=True,
    )
    assert isinstance(gen_preview, dict)
    assert gen_preview.get("dry_run") is True
    assert gen_preview["request"]["body"]["model"] == "gemini-4-hydrogen"
    assert gen_preview["request"]["body"]["stream"] is False

    sample_file = tmp_path / "sample.txt"
    sample_file.write_text("launchcut frame telemetry", encoding="utf-8")
    ana_preview = await gemini_analyze_media_json(
        sample_file,
        question="Score this video",
        system_prompt="QA Critic",
        settings=settings,
        dry_run=True,
    )
    assert isinstance(ana_preview, dict)
    assert ana_preview.get("dry_run") is True

    vid_preview = await gemini_generate_video(
        prompt="Smooth macro camera push-in",
        out_path=tmp_path / "out.mp4",
        aspect_ratio="16:9",
        resolution="1080p",
        settings=settings,
        dry_run=True,
    )
    assert isinstance(vid_preview, dict)
    assert vid_preview["request"]["body"]["model"] == "gemini-omni-1.2-flash"

    img_preview = await gemini_generate_image(
        prompt="Hero product still",
        out_path=tmp_path / "out.jpg",
        aspect_ratio="16:9",
        settings=settings,
        dry_run=True,
    )
    assert isinstance(img_preview, dict)
    assert img_preview["request"]["body"]["model"] == "gemini-nano-banana-2.1"

    tts_preview = await gemini_generate_tts(
        text="Welcome to LaunchCut",
        out_path=tmp_path / "vo.wav",
        voice="Rachel",
        style="confident warm tone",
        settings=settings,
        dry_run=True,
    )
    assert isinstance(tts_preview, dict)
    assert tts_preview["request"]["body"]["model"] == "gemini-3.8-flash-tts"


@pytest.mark.asyncio
async def test_loop2_60s_seven_act_story_rhythm_and_script_gate(tmp_path: Path) -> None:
    """Verify 60s hybrid & cinema compilation hits 58-62s across 6-8 acts with distinct templates & VO gate."""
    from gtm_videogen.agents.stitch_engine import compile_stitch_project
    from gtm_videogen.models.stitch_project import (
        apply_deterministic_script_autofix,
        validate_stitch_project,
    )

    settings = Settings(data_dir=tmp_path / "data")
    reg = ModelRegistryStore(tmp_path / "stitch")

    # 1. 60s Hybrid SaaS Launch (Linear Dark Craft) -> 7 acts, 60s exact, zero duplicate templates!
    hybrid_proj = await compile_stitch_project(
        idea="Autonomous AI code review that catches production bugs in 200ms",
        target_duration_sec=60,
        recipe_id="linear_dark_craft",
        brand_name="VeloceAI",
        registry_store=reg,
        settings=settings,
        mock=True,
    )
    assert 58 <= hybrid_proj.total_duration_sec() <= 62
    assert len(hybrid_proj.scenes) == 7
    assert hybrid_proj.scenes[0].narrative_role == "hook"
    assert hybrid_proj.scenes[-1].narrative_role == "climax_cta"

    mg_templates = [
        s.graphic_spec.template_id
        for s in hybrid_proj.scenes
        if s.kind == "motion_graphic" and s.graphic_spec is not None
    ]
    ai_acts = [s for s in hybrid_proj.scenes if s.kind == "ai_sequence"]
    assert len(mg_templates) == 5
    assert len(set(mg_templates)) == 5, f"Expected 5 distinct motion templates, got {mg_templates}"
    assert len(ai_acts) == 2

    # 2. 60s Pure Cinema on an 8s-max model (Veo 3 Fast) -> 8 acts = 60s exact (not capped at 32s!)
    cinema_proj = await compile_stitch_project(
        idea="Lone explorer crossing golden desert dunes toward a monolith",
        target_duration_sec=60,
        video_model_id="fal-ai/veo3/fast",
        recipe_id="imax_70mm_epic",
        registry_store=reg,
        settings=settings,
        mock=True,
    )
    assert cinema_proj.total_duration_sec() == 60
    assert len(cinema_proj.scenes) == 8

    # 3. Script Gate catches rushed VO (>3.1 wps) & monotonous camera, and Auto-Fix repairs both!
    veo_spec = reg.get_video_model("fal-ai/veo3/fast")
    cinema_proj.scenes[0].vo_line = " ".join(["rapid"] * 35)  # 35 words in 6s = 5.8 wps
    cinema_proj.scenes[0].camera_spec = "85mm static shot"
    cinema_proj.scenes[1].camera_spec = "85mm static shot"
    cinema_proj.scenes[2].camera_spec = "85mm static shot"

    gate_before = validate_stitch_project(cinema_proj, veo_spec)
    codes_before = {
        issue["code"]
        for sc_rep in gate_before["scenes"]
        for issue in sc_rep["issues"]
    }
    assert "vo_too_fast" in codes_before
    assert "monotonous_camera" in codes_before

    apply_deterministic_script_autofix(cinema_proj, veo_spec)
    gate_after = validate_stitch_project(cinema_proj, veo_spec)
    assert gate_after["valid"] is True, f"Remaining warnings: {gate_after}"


@pytest.mark.asyncio
async def test_loop3_and_loop4_full_60s_autopilot_banger_qa_and_zero_freeze(tmp_path: Path) -> None:
    """Produce a real 60s 7-Act hybrid video via 1-Click /autopilot and verify >=86/100 QA & zero freezes."""
    from fastapi.testclient import TestClient

    from gtm_videogen.api.app import create_app

    settings = Settings(data_dir=tmp_path / "data")
    app = create_app(settings=settings)
    with TestClient(app) as client:
        resp = client.post(
            "/api/stitch/autopilot",
            json={
                "idea": "Global multi-currency developer billing in three lines of code",
                "target_duration_sec": 60,
                "aspect_ratio": "16:9",
                "recipe_id": "stripe_living_flow",
                "brand_name": "NovaPay",
                "brand_tagline": "Programmable revenue infrastructure",
                "mock": True,
                "sync": True,
            },
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["status"] == "done"
        assert data["master_video_url"]
        assert len(data["scenes"]) == 7

        qa = data["qa_report"]
        assert isinstance(qa, dict)
        assert qa["passed"] is True, f"Unexpected QA failure: {qa}"
        assert qa["banger_score"] >= 86, f"Expected >=86/100 banger score, got {qa}"
        assert qa["grade"] in ("A", "A+")

        tel = qa["telemetry"]
        assert tel["duration_sec"] >= 57.0
        assert tel["black_segments"] == [], f"Unexpected black segments: {tel['black_segments']}"
        assert tel["freeze_segments"] == [], f"Unexpected freeze segments: {tel['freeze_segments']}"
        assert tel["loudness_lufs"] is not None
        assert -21.0 <= tel["loudness_lufs"] <= -10.0


@pytest.mark.asyncio
async def test_loop3_flawed_black_silent_video_caught_by_qa_and_healed_by_autofix(
    tmp_path: Path,
) -> None:
    """Inject a dead-black silent clip into Act 1, verify QA catches & caps score, then verify /qa-autofix heals it."""
    from fastapi.testclient import TestClient

    from gtm_videogen.agents.stitch_engine import StitchProjectStore
    from gtm_videogen.api.app import create_app
    from gtm_videogen.media._subprocess import run_subprocess

    settings = Settings(data_dir=tmp_path / "data")
    app = create_app(settings=settings)
    with TestClient(app) as client:
        c_resp = client.post(
            "/api/stitch/projects",
            json={
                "idea": "Realtime cloud observability dashboard",
                "target_duration_sec": 15,
                "recipe_id": "linear_dark_craft",
                "brand_name": "PulseOps",
                "mock": True,
            },
        )
        assert c_resp.status_code == 200
        proj_data = c_resp.json()
        pid = proj_data["project_id"]

        # Render all acts cleanly first
        r_resp = client.post(
            f"/api/stitch/projects/{pid}/render-all",
            json={"mock": True, "sync": True},
        )
        assert r_resp.status_code == 200

        # Sabotage Act 1's winning take with a 5s pure-black + silent MP4 and disable brand outro
        store = StitchProjectStore(settings.data_dir / "stitch")
        proj = store.load(pid)
        assert proj is not None
        bad_mp4 = store.media_dir(pid) / "sabotaged_black.mp4"
        await run_subprocess(
            [
                settings.ffmpeg_bin,
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=1280x720:r=24:d=5",
                "-f",
                "lavfi",
                "-i",
                "anullsrc=r=48000:cl=stereo",
                "-t",
                "5",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                str(bad_mp4),
            ],
            check=True,
        )
        proj.scenes[0].takes[-1].clip_path = str(bad_mp4)
        proj.add_brand_outro = False
        store.save(proj)

        # Stitch with native_only so audio stays silent and Act 1 is pure black
        st_resp = client.post(
            f"/api/stitch/projects/{pid}/stitch",
            json={"audio_mode": "native_only", "mock": True},
        )
        assert st_resp.status_code == 200
        bad_qa = st_resp.json()["qa_report"]
        assert bad_qa["passed"] is False
        assert bad_qa["banger_score"] <= 68, f"Expected capped score <=68, got {bad_qa}"
        assert bad_qa["scene_scores"][0]["status"] == "fail"
        assert any("black frames" in msg for msg in bad_qa["blocking_issues"])

        # Now run 1-click Self-Healing QA Auto-Fix!
        fix_resp = client.post(
            f"/api/stitch/projects/{pid}/qa-autofix",
            json={"mock": True},
        )
        assert fix_resp.status_code == 200, fix_resp.text
        healed_qa = fix_resp.json()["qa_report"]
        assert healed_qa["passed"] is True, f"Expected healed video to pass QA, got {healed_qa}"
        assert healed_qa["banger_score"] >= 86
        assert healed_qa["scene_scores"][0]["status"] == "pass"



def test_loop5_intuitive_ui_autopilot_and_qa_scorecard_markup(tmp_path: Path) -> None:
    """Verify Stage 1 1-Click Auto-Produce + Stage 4 QA Scorecard UI & Gemini CLI model catalog."""
    from fastapi.testclient import TestClient

    from gtm_videogen.api.app import create_app

    settings = Settings(data_dir=tmp_path / "data")
    app = create_app(settings=settings)
    with TestClient(app) as client:
        r = client.get("/")
        assert r.status_code == 200
        html = r.text
        assert 'id="btn-autopilot"' in html
        assert "autoProduceFromSpark()" in html
        assert "60s (1-Min Banger Film)" in html
        assert 'id="qa-scorecard-box"' in html
        assert "renderQAScorecard" in html
        assert "runQAAutofixUI()" in html
        assert "runDeepQAUI()" in html

        cat = client.get("/api/stitch/catalog").json()
        assert any(m["id"] == "gemini-omni-1.2-flash" for m in cat["video_models"])
        assert any("gemini-cli/" in m["id"] for m in cat["llm_models"])
