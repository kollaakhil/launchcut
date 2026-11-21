"""Integration & unit tests for Hybrid Launch Videos (Motion Graphics, User Clips, A-Roll Cutaways & Overlays)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from gtm_videogen.api.app import create_app
from gtm_videogen.core.config import Settings
from gtm_videogen.media.motion_templates import render_motion_template_mp4, render_overlay_png
from gtm_videogen.models.stitch_project import MotionGraphicSpec, SceneOverlaySpec


def _make_client(tmp_path: Path) -> TestClient:
    settings = Settings(
        data_dir=tmp_path,
        fal_key="",
        openrouter_api_key="",
    )
    app = create_app(data_dir=tmp_path, settings=settings, rate_limit_requests=0)
    return TestClient(app)


def _make_synthetic_video(out_path: Path, duration_sec: float = 4.0) -> Path:
    """Create a tiny valid H.264 + AAC test clip using FFmpeg."""
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=size=320x180:rate=24:duration={duration_sec}",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={duration_sec}",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-shortest",
            str(out_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return out_path


@pytest.mark.asyncio
async def test_all_six_motion_templates_and_rgba_overlay(tmp_path: Path) -> None:
    templates = [
        "kinetic_headline",
        "prompt_ui_demo",
        "benchmark_bars",
        "bento_feature_grid",
        "terminal_code",
        "split_comparison",
    ]
    for tpl in templates:
        out_mp4 = tmp_path / f"{tpl}.mp4"
        spec = MotionGraphicSpec(
            template_id=tpl,  # type: ignore[arg-type]
            theme_mode="warm_editorial" if tpl == "kinetic_headline" else "dark_glass",
            eyebrow="LAUNCH TEST",
            headline=f"Headline for {tpl}",
            subheadline="Crisp 24fps procedural motion design",
            prompt_text="launchcut deploy --prod",
            response_lines=["✓ Compiled in 120ms", "✓ Studio ready"],
        )
        res = await render_motion_template_mp4(
            spec=spec,
            output_path=out_mp4,
            width=480,
            height=270,
            fps=12,
            duration_sec=2.0,
            brand_name="LaunchCut",
        )
        assert res.is_file()
        assert res.stat().st_size > 4_000

    ov_png = render_overlay_png(
        overlay=SceneOverlaySpec(
            overlay_id="ov_test",
            kind="lower_third",
            primary_text="Akhil Kolla",
            secondary_text="Founder, LaunchCut",
            position="bottom_left",
            start_sec=0.2,
            end_sec=2.0,
        ),
        width=640,
        height=360,
        output_path=tmp_path / "ov.png",
    )
    assert ov_png.is_file()
    assert ov_png.stat().st_size > 1_000


def test_hybrid_launch_user_clip_aroll_cutaways_and_captions_e2e(tmp_path: Path) -> None:
    client = _make_client(tmp_path)
    sample_vid = _make_synthetic_video(tmp_path / "founder.mp4", duration_sec=6.0)

    # 1. Create a hybrid Anthropic Warm Keynote project for a custom brand
    create_r = client.post(
        "/api/stitch/projects",
        json={
            "idea": "Introducing our instant zero-latency cloud database for AI agents",
            "brand_name": "NeonPulse",
            "world_preset": "anthropic_editorial",
            "recipe_id": "anthropic_editorial",
            "target_duration_sec": 15,
            "mock": True,
        },
    )
    assert create_r.status_code == 200
    proj = create_r.json()
    pid = proj["project_id"]
    assert proj["scenes"][0]["kind"] == "motion_graphic"
    assert proj["scenes"][0]["graphic_spec"]["template_id"] == "kinetic_headline"

    # 2. Add a user_clip Act, stream-upload a video via multipart, adjust framing & render it
    add_r = client.post(
        f"/api/stitch/projects/{pid}/scenes",
        json={"title": "Founder Intro", "duration_sec": 4, "kind": "user_clip", "continues_from_prev": False},
    )
    assert add_r.status_code == 200
    uc_sid = add_r.json()["scenes"][-1]["scene_id"]

    with sample_vid.open("rb") as fh:
        up_r = client.post(
            f"/api/stitch/projects/{pid}/scenes/{uc_sid}/upload-clip",
            files={"file": ("founder.mp4", fh, "video/mp4")},
        )
    assert up_r.status_code == 200
    assert up_r.json()["scenes"][-1]["user_clip"]["clip_url"]

    trim_r = client.patch(
        f"/api/stitch/projects/{pid}/scenes/{uc_sid}/clip-settings",
        json={"trim_start_sec": 0.5, "trim_end_sec": 3.5, "framing": "device_bezel", "keep_original_audio": True},
    )
    assert trim_r.status_code == 200
    assert trim_r.json()["scenes"][-1]["user_clip"]["framing"] == "device_bezel"

    ren_uc = client.post(
        f"/api/stitch/projects/{pid}/scenes/{uc_sid}/render",
        json={"engine_id": "mock", "mock": True, "sync": True},
    )
    assert ren_uc.status_code == 200
    assert ren_uc.json()["scenes"][-1]["winning_clip_url"]

    # 3. Upload Founder A-Roll, transcribe, auto-plan timed cutaways, render cutaways, and stitch with Captions!
    with sample_vid.open("rb") as fh:
        aroll_r = client.post(
            f"/api/stitch/projects/{pid}/upload-aroll",
            files={"file": ("founder_keynote.mp4", fh, "video/mp4")},
        )
    assert aroll_r.status_code == 200
    assert aroll_r.json()["aroll_video_url"]

    tr_r = client.post(f"/api/stitch/projects/{pid}/transcribe-aroll", json={"mock": True})
    assert tr_r.status_code == 200
    assert len(tr_r.json()["aroll_transcript"]) >= 5

    cut_r = client.post(f"/api/stitch/projects/{pid}/auto-cutaways", json={"mock": True})
    assert cut_r.status_code == 200
    cut_proj = cut_r.json()
    assert cut_proj["timeline_mode"] == "aroll_with_cutaways"
    assert len(cut_proj["scenes"]) >= 1
    assert cut_proj["scenes"][0]["aroll_window"] is not None

    # Render first cutaway motion graphic ($0)
    cut_sid = cut_proj["scenes"][0]["scene_id"]
    ren_cut = client.post(
        f"/api/stitch/projects/{pid}/scenes/{cut_sid}/render",
        json={"engine_id": "mock", "mock": True, "sync": True},
    )
    assert ren_cut.status_code == 200
    assert ren_cut.json()["scenes"][0]["winning_clip_url"]

    # Stitch continuous Founder A-Roll + Cutaway + ASS Captions
    st_r = client.post(
        f"/api/stitch/projects/{pid}/stitch",
        json={
            "timeline_mode": "aroll_with_cutaways",
            "audio_mode": "native_plus_score",
            "enable_captions": True,
            "caption_style": "editorial_serif",
            "mock": True,
        },
    )
    assert st_r.status_code == 200
    master_url = st_r.json()["master_video_url"]
    assert master_url

    # Verify HTTP 206 streaming works on the hybrid A-roll master
    rng = client.get(master_url, headers={"Range": "bytes=0-1023"})
    assert rng.status_code == 206
    assert len(rng.content) == 1024
