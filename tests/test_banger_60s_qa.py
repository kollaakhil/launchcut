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
    """Verify 2-Screen Intent Studio UI (Brief Composer + Co-Pilot + QA Scorecard) & Gemini CLI catalog."""
    from fastapi.testclient import TestClient

    from gtm_videogen.api.app import create_app

    settings = Settings(data_dir=tmp_path / "data")
    app = create_app(settings=settings)
    with TestClient(app) as client:
        r = client.get("/")
        assert r.status_code == 200
        html = r.text
        assert 'id="nav-studio"' in html
        assert 'id="brief-composer"' in html
        assert 'id="composer-files"' in html
        assert 'id="spark-audio-intent"' in html
        assert 'id="btn-autopilot"' in html
        assert "autoProduceFromSpark()" in html
        assert "60s (1-Min Banger Film)" in html
        assert 'id="copilot-input"' in html
        assert "applyDirectorCopilotUI()" in html
        assert 'id="studio-acts-strip"' in html
        assert 'id="qa-scorecard-box"' in html
        assert "renderQAScorecard" in html
        assert "runQAAutofixUI()" in html
        assert "runDeepQAUI()" in html

        cat = client.get("/api/stitch/catalog").json()
        assert any(m["id"] == "gemini-omni-1.2-flash" for m in cat["video_models"])
        assert any("gemini-cli/" in m["id"] for m in cat["llm_models"])


def test_2screen_intent_studio_copilot_and_smart_attach(tmp_path: Path) -> None:
    """Verify multi-line creative brief, smart asset routing, and surgical Director Co-Pilot edits."""
    import io

    from fastapi.testclient import TestClient
    from PIL import Image

    from gtm_videogen.api.app import create_app

    def _png_bytes(color: tuple[int, int, int]) -> bytes:
        img = Image.new("RGB", (320, 180), color)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    settings = Settings(data_dir=tmp_path / "data")
    app = create_app(settings=settings)
    with TestClient(app) as client:
        # 1. Create project from a multi-line creative brief + upfront audio/caption intent
        brief = (
            "Launch VeloceAI autonomous PR reviewer:\n"
            "- Catches production race conditions in 180ms\n"
            "- Show crisp UI demo and benchmark proof\n"
            "- Confident editorial tone"
        )
        c_resp = client.post(
            "/api/stitch/projects",
            json={
                "idea": brief,
                "target_duration_sec": 15,
                "recipe_id": "linear_dark_craft",
                "brand_name": "VeloceAI",
                "audio_mode": "vo_plus_ducked_score",
                "enable_captions": True,
                "mock": True,
            },
        )
        assert c_resp.status_code == 200, c_resp.text
        proj = c_resp.json()
        pid = proj["project_id"]
        assert proj["audio_mode"] == "vo_plus_ducked_score"
        assert proj["enable_captions"] is True
        assert len(proj["scenes"]) >= 2

        # 2. Smart-attach a UI screenshot -> locks @Element1 and attaches to UI showcase act
        att_ui = client.post(
            f"/api/stitch/projects/{pid}/smart-attach",
            files={"file": ("dashboard_ui.png", _png_bytes((24, 34, 52)), "image/png")},
            data={"role": "auto"},
        )
        assert att_ui.status_code == 200, att_ui.text
        proj_after_ui = att_ui.json()
        assert len(proj_after_ui["elements"]) == 1
        assert proj_after_ui["elements"][0]["tag"] == "@Element1"
        assert any(
            (s.get("graphic_spec") or {}).get("screenshot_url")
            for s in proj_after_ui["scenes"]
        )

        # 3. Smart-attach a brand logo -> automatically enables Brand Logo Outro
        att_logo = client.post(
            f"/api/stitch/projects/{pid}/smart-attach",
            files={"file": ("veloce_logo.png", _png_bytes((229, 77, 58)), "image/png")},
            data={"role": "auto"},
        )
        assert att_logo.status_code == 200, att_logo.text
        assert att_logo.json()["add_brand_outro"] is True

        # 4. Render all acts & stitch initial Master Cut
        r_all = client.post(
            f"/api/stitch/projects/{pid}/render-all",
            json={"mock": True, "sync": True},
        )
        assert r_all.status_code == 200
        st = client.post(f"/api/stitch/projects/{pid}/stitch", json={"mock": True})
        assert st.status_code == 200
        assert st.json()["master_video_url"]
        act1_takes_before = len(st.json()["scenes"][0]["takes"])
        act2_takes_before = len(st.json()["scenes"][1]["takes"])

        # 5. Surgical Director Co-Pilot edit: transform Act 2 into a benchmark chart + cinema letterbox
        edit_resp = client.post(
            f"/api/stitch/projects/{pid}/direct-edit",
            json={
                "instruction": 'Turn Act 2 into an animated benchmark chart saying "Ten times faster than manual review" and add 2.39:1 cinema letterbox',
                "auto_render": True,
                "mock": True,
            },
        )
        assert edit_resp.status_code == 200, edit_resp.text
        edited = edit_resp.json()
        assert edited["cinema_letterbox"] is True
        act2 = edited["scenes"][1]
        assert act2["kind"] == "motion_graphic"
        assert act2["graphic_spec"]["template_id"] == "benchmark_bars"
        assert "Ten times faster" in act2["vo_line"]
        # Surgical re-render: Act 1 was NOT re-shot, while Act 2 got a fresh take and Master Cut re-stitched!
        assert len(edited["scenes"][0]["takes"]) == act1_takes_before
        assert len(act2["takes"]) == act2_takes_before + 1
        assert edited["master_video_url"]
        assert edited["qa_report"]["passed"] is True
        assert edited["qa_report"]["banger_score"] >= 86


def test_multimodel_cost_reducer_prose_adapter_and_60s_continuity(tmp_path: Path) -> None:
    """Verify multi-model Smart Router, >75% 15s test cost drop, prose vs JSON adapter & 60s world continuity."""
    from fastapi.testclient import TestClient

    from gtm_videogen.api.app import create_app
    from gtm_videogen.models.model_registry import (
        ModelRegistryStore,
        build_registry_fal_payload,
    )

    reg = ModelRegistryStore(tmp_path / "reg")

    # 1. Verify verified fal.ai schemas & silent-default cost savings
    kling3 = reg.get_video_model("fal-ai/kling-video/v3/pro/text-to-video")
    assert kling3.estimate_clip_usd(15, with_audio=False) == 1.68  # 33% cheaper than $2.52!
    assert kling3.estimate_clip_usd(15, with_audio=True) == 2.52
    k3_ep, k3_payload = build_registry_fal_payload(kling3, prompt='{"test":1}', duration_sec=10, aspect_ratio="16:9")
    assert k3_ep == "fal-ai/kling-video/v3/pro/text-to-video"
    assert k3_payload["generate_audio"] is False

    hailuo = reg.get_video_model("fal-ai/minimax/hailuo-2.3/pro/text-to-video")
    assert hailuo.estimate_clip_usd(8, with_audio=False) == 0.49
    hl_ep, hl_payload = build_registry_fal_payload(
        hailuo,
        prompt="Cinematic 50mm push-in",
        duration_sec=8,
        aspect_ratio="16:9",
        start_image="https://example.com/start.jpg",
        negative_prompt="blurry",
    )
    assert hl_ep == "fal-ai/minimax/hailuo-2.3/pro/image-to-video"
    assert hl_payload["image_url"] == "https://example.com/start.jpg"
    assert "duration" not in hl_payload
    assert "aspect_ratio" not in hl_payload
    assert "negative_prompt" not in hl_payload

    seedance = reg.get_video_model("fal-ai/bytedance/seedance/v1/lite/text-to-video")
    assert seedance.estimate_clip_usd(5, with_audio=False) == 0.18

    settings = Settings(data_dir=tmp_path / "data")
    app = create_app(settings=settings)
    with TestClient(app) as client:
        # UI markup sanity for Quality Tier Pill + Live Cost Estimate + Preview Stills + Per-Act Engine
        html = client.get("/").text
        assert 'id="spark-quality-tier"' in html
        assert 'id="live-cost-pill"' in html
        assert "updateLiveCostEstimate()" in html
        assert 'id="btn-filmstrip-stills"' in html
        assert 'id="ed-scene-model"' in html
        assert 'id="ed-native-audio"' in html

        # 2. Default 15s SaaS launch run (no recipe_id!) -> Smart Router + Auto-Hybrid drops cost from $2.52 to <= $0.55!
        c15 = client.post(
            "/api/stitch/projects",
            json={
                "idea": "Launch HyperQueue serverless Postgres workflow engine for AI agents",
                "target_duration_sec": 15,
                "mock": True,
            },
        )
        assert c15.status_code == 200, c15.text
        p15 = c15.json()
        assert p15["video_model_id"] == "auto"
        assert any(s["kind"] == "motion_graphic" for s in p15["scenes"])
        assert p15["estimated_usd"] <= 0.55, f"Expected <=$0.55 on 15s default run, got {p15['estimated_usd']}"

        # 3. 60s 7-Act Hybrid Film -> verify Cross-Graphic Continuity Lock & Model-Specific Prose vs JSON
        c60 = client.post(
            "/api/stitch/projects",
            json={
                "idea": "Introduce NovaDeploy autonomous cloud rollout platform with zero-downtime benchmarks",
                "target_duration_sec": 60,
                "brand_name": "NovaDeploy",
                "mock": True,
            },
        )
        assert c60.status_code == 200, c60.text
        p60 = c60.json()
        pid60 = p60["project_id"]
        ai_acts = [(i, s) for i, s in enumerate(p60["scenes"]) if s["kind"] == "ai_sequence"]
        assert len(ai_acts) >= 2

        # Later AI act separated by Motion Graphics MUST carry world_continuity_lock!
        later_idx, later_sc = ai_acts[-1]
        if later_idx > 0 and not later_sc["continues_from_prev"]:
            assert "world_continuity_lock" in later_sc["compiled_prompt_dict"]

        # Override Act 1 to Kling 3.0 Pro (Structured JSON) and last AI Act to Seedance Lite (Natural Prose)
        scenes_patch = list(p60["scenes"])
        first_ai_idx = ai_acts[0][0]
        scenes_patch[first_ai_idx]["video_model_id"] = "fal-ai/kling-video/v3/pro/text-to-video"
        scenes_patch[later_idx]["video_model_id"] = "fal-ai/bytedance/seedance/v1/lite/text-to-video"
        pr = client.patch(f"/api/stitch/projects/{pid60}", json={"scenes": scenes_patch})
        assert pr.status_code == 200, pr.text
        patched60 = pr.json()
        assert patched60["scenes"][first_ai_idx]["compiled_prompt_str"].lstrip().startswith("{")
        assert not patched60["scenes"][later_idx]["compiled_prompt_str"].lstrip().startswith("{")
        assert patched60["scenes"][later_idx]["resolved_model_id"] == "fal-ai/bytedance/seedance/v1/lite/text-to-video"

        # Plain-English Director Co-Pilot switch to Fast Draft mode across film
        de = client.post(
            f"/api/stitch/projects/{pid60}/direct-edit",
            json={"instruction": "Switch to fast draft budget mode", "auto_render": False, "mock": True},
        )
        assert de.status_code == 200, de.text
        assert de.json()["video_model_id"] == "fal-ai/bytedance/seedance/v1/lite/text-to-video"

        # Render all 7 acts (parallel $0 motion graphics + sequential AI acts) & verify Master Cut QA!
        ra = client.post(
            f"/api/stitch/projects/{pid60}/render-all",
            json={"mock": True, "sync": True},
        )
        assert ra.status_code == 200, ra.text
        done60 = ra.json()
        assert done60["master_video_url"]
        assert done60["qa_report"]["passed"] is True
        assert done60["qa_report"]["banger_score"] >= 86


@pytest.mark.asyncio
async def test_genie4_registry_and_real_progress_bar_stages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify Google Genie 4 world provider helpers, session guard, 0->100% progress stages & single stitch."""
    import base64

    from fastapi.testclient import TestClient
    from PIL import Image

    import gtm_videogen.agents.stitch_engine as se
    from gtm_videogen.api.app import create_app
    from gtm_videogen.models.model_registry import ModelRegistryStore
    from gtm_videogen.providers.genie_world import (
        GENIE4_MODEL_ID,
        GENIE4_USD_PER_SEC,
        compile_genie4_world_prompt,
        prepare_genie4_reference_jpeg,
    )

    # 1. Registry + alias resolution for Google DeepMind Genie 4
    reg = ModelRegistryStore(tmp_path / "reg_genie")
    g4 = reg.get_video_model("genie-4")
    assert g4.id == GENIE4_MODEL_ID == "models/genie-4"
    assert g4.provider == "genie"
    assert g4.est_usd_per_sec == GENIE4_USD_PER_SEC
    assert g4.supports_native_audio is True

    # 2. <44 KB 16:9 JPEG reference seeder + spatial world prompt adapter
    big_png = tmp_path / "hero_large.png"
    Image.new("RGB", (1920, 1920), (42, 88, 140)).save(big_png, format="PNG")
    ref_b64 = prepare_genie4_reference_jpeg(big_png)
    assert ref_b64 is not None
    raw_jpg = base64.b64decode(ref_b64)
    assert len(raw_jpg) <= 44_000

    world_prompt = compile_genie4_world_prompt(
        "Warm studio desk, cut to glowing brass dial with orchestral soundtrack",
        camera_spec="35mm gentle push-in",
        audio_direction="soft mechanical click and room tone",
    )
    assert "cut to" not in world_prompt.lower()
    assert "spatial" in world_prompt.lower() or "environment" in world_prompt.lower()

    # 3. API catalog, health, UI markup & /genie/session key guard
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    settings = Settings(data_dir=tmp_path / "data_genie", gemini_api_key="", google_api_key="")
    app = create_app(settings=settings)
    with TestClient(app) as client:
        html = client.get("/").text
        assert 'value="models/genie-4"' in html
        assert "serverDriven: true" in html
        assert "!proj.master_video_url" in html
        assert 'id="models-key-status"' in html

        health = client.get("/api/health").json()
        assert "gemini" in health["api_keys"]
        assert health["api_keys"]["gemini"] is False

        cat = client.get("/api/stitch/catalog").json()
        assert cat["live_gemini_ready"] is False
        assert any(m["id"] == "models/genie-4" for m in cat["video_models"])

        # Private browser X-Gemini-Api-Key header activates live_gemini_ready per-request without mutating global env!
        cat_hdr = client.get(
            "/api/stitch/catalog",
            headers={"X-Gemini-Api-Key": "AQ.test_dummy_key_for_header_isolation"},
        ).json()
        assert cat_hdr["live_gemini_ready"] is True
        assert client.get("/api/stitch/catalog").json()["live_gemini_ready"] is False
        assert 'id="browser-gemini-key-input"' in html

        gs = client.post("/api/stitch/genie/session")
        assert gs.status_code == 400
        assert "GEMINI_API_KEY" in gs.json()["detail"]

    # 4. Track real server-driven progress_pct / progress_stage & verify stitch_master_cut runs ONCE in autopilot!
    proj = await se.compile_stitch_project(
        idea="Launch OrbitSQL zero-latency distributed database",
        target_duration_sec=15,
        brand_name="OrbitSQL",
        registry_store=reg,
        settings=settings,
        mock=True,
    )
    # Test Director Co-Pilot switching to Genie 4
    await se.apply_director_intent_edit(
        proj,
        tmp_path / "media",
        instruction="Use Genie 4 live world for this film",
        auto_render=False,
        registry_store=reg,
        settings=settings,
        mock=True,
    )
    assert proj.video_model_id == "models/genie-4"

    seen_pcts: list[int] = []
    seen_stages: list[str] = []
    stitch_calls = 0
    orig_stitch = se.stitch_master_cut

    async def _counted_stitch(*args, **kwargs):
        nonlocal stitch_calls
        stitch_calls += 1
        return await orig_stitch(*args, **kwargs)

    monkeypatch.setattr(se, "stitch_master_cut", _counted_stitch)

    def _on_prog(p) -> None:
        seen_pcts.append(int(p.progress_pct))
        if p.progress_stage:
            seen_stages.append(p.progress_stage)

    await se.autopilot_produce_project(
        proj,
        tmp_path / "media_auto",
        registry_store=reg,
        settings=settings,
        mock=True,
        on_progress=_on_prog,
    )
    assert stitch_calls == 1, f"Expected stitch_master_cut to run once, ran {stitch_calls} times!"
    assert proj.progress_pct == 100
    assert proj.progress_stage == "Complete"
    assert any(82 <= p <= 96 for p in seen_pcts), f"Expected stitching progress stages in {seen_pcts}"
    assert 98 in seen_pcts, f"Expected 98% QA stage in {seen_pcts}"
    assert any("Normalizing Clips" in st for st in seen_stages)
    assert any("Soundstage" in st for st in seen_stages)
    assert any("Video QA" in st for st in seen_stages)


