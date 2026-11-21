"""Comprehensive unit + integration tests for the 5 LaunchCut v2 Upgrades."""

from __future__ import annotations

import asyncio
import base64
import io
from pathlib import Path

import httpx
import pytest
from PIL import Image

from gtm_videogen.api.app import create_app
from gtm_videogen.api.studio_steps import _wire_adjacent_end_keyframes
from gtm_videogen.media.audio_sync import (
    build_transition_sfx_cues,
    detect_audio_beats,
    snap_to_nearest_beat,
)
from gtm_videogen.media.ffmpeg_stitcher import FFmpegStitcher
from gtm_videogen.media.kinetic_motion import render_kinetic_card_mp4
from gtm_videogen.media.media_qa import MediaQA, probe_media
from gtm_videogen.models.schemas import (
    CreativeBible,
    MusicCue,
    RenderedAudio,
    RenderedClip,
    Shot,
    ShotSheet,
    SoundPlan,
    VoiceCue,
)
from gtm_videogen.providers.video import build_fal_video_payload


def _make_png_file(path: Path, size: tuple[int, int], color: tuple[int, int, int, int]) -> Path:
    img = Image.new("RGBA", size, color)
    img.save(path, format="PNG")
    return path


def _make_png_data_url(
    size: tuple[int, int] = (64, 64), color: tuple[int, int, int] = (30, 80, 190)
) -> str:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


async def _wait_idle(client: httpx.AsyncClient, run_id: str, timeout: float = 30.0) -> dict:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        r = await client.get(f"/api/runs/{run_id}")
        assert r.status_code == 200
        data = r.json()
        if data["status"] != "working":
            return data
        await asyncio.sleep(0.08)
    raise TimeoutError(f"Run {run_id} did not settle within {timeout}s")


# ---------------------------------------------------------------------------
# Upgrade 1: Real Kinetic Motion Graphics Engine ($0/shot)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_kinetic_motion_engine_all_three_modes(
    tmp_path: Path, sample_creative_bible: CreativeBible
) -> None:
    qa = MediaQA()
    logo_png = _make_png_file(tmp_path / "logo.png", (180, 72), (226, 163, 54, 255))
    ui_png = _make_png_file(tmp_path / "ui.png", (400, 250), (36, 48, 72, 255))

    # Mode A: Kinetic word-by-word typography with word_timings
    typo_out = tmp_path / "kinetic_typo.mp4"
    await render_kinetic_card_mp4(
        text="Ship faster with precision",
        subtitle="Built for modern product teams",
        output_path=typo_out,
        duration_sec=1.5,
        width=480,
        height=270,
        fps=24,
        bible=sample_creative_bible,
        bg_rgb=(17, 24, 39),
        text_rgb=(255, 255, 255),
        accent_rgb=(226, 163, 54),
        word_timings=[
            {"word": "Ship", "start": 0.0, "end": 0.25},
            {"word": "faster", "start": 0.25, "end": 0.55},
            {"word": "with", "start": 0.55, "end": 0.8},
            {"word": "precision", "start": 0.8, "end": 1.2},
        ],
    )
    p1 = await probe_media(typo_out)
    assert p1.has_video and p1.width == 480 and p1.height == 270
    assert abs(p1.duration_sec - 1.5) < 0.25
    bf1 = await qa.detect_black_and_freeze(typo_out)
    assert bf1["black_segments"] == []

    # Mode B: Dark-bezel product/UI showcase with animated cursor & click ripple
    ui_out = tmp_path / "kinetic_ui.mp4"
    await render_kinetic_card_mp4(
        text="Instant triage",
        subtitle="Keyboard-first workflows",
        output_path=ui_out,
        duration_sec=1.5,
        width=480,
        height=270,
        fps=24,
        bible=sample_creative_bible,
        bg_rgb=(17, 24, 39),
        text_rgb=(255, 255, 255),
        accent_rgb=(59, 130, 246),
        media_image_path=ui_png,
    )
    p2 = await probe_media(ui_out)
    assert p2.has_video and p2.width == 480 and p2.height == 270
    bf2 = await qa.detect_black_and_freeze(ui_out)
    assert bf2["black_segments"] == []

    # Mode C: Animated Brand Logo Lockup
    logo_out = tmp_path / "kinetic_logo.mp4"
    await render_kinetic_card_mp4(
        text="LaunchCut",
        subtitle="launchcut.ai",
        output_path=logo_out,
        duration_sec=1.5,
        width=480,
        height=270,
        fps=24,
        bible=sample_creative_bible,
        bg_rgb=(17, 24, 39),
        text_rgb=(255, 255, 255),
        accent_rgb=(226, 163, 54),
        logo_path=logo_png,
        card_kind="end_card",
    )
    p3 = await probe_media(logo_out)
    assert p3.has_video and p3.width == 480 and p3.height == 270
    bf3 = await qa.detect_black_and_freeze(logo_out)
    assert bf3["black_segments"] == []


# ---------------------------------------------------------------------------
# Upgrade 2: Beat Detection, SFX Snapping & Transition SFX Cues
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_audio_sync_beats_and_transition_sfx(tmp_path: Path) -> None:
    # Snap helper unit check
    beats = [0.5, 1.0, 1.5, 2.0]
    assert snap_to_nearest_beat(0.92, beats, max_shift_sec=0.18) == 1.0
    assert snap_to_nearest_beat(0.75, beats, max_shift_sec=0.10) == 0.75

    # Transition SFX generator
    sheet = ShotSheet(
        target_duration_sec=9.0,
        shots=[
            Shot(
                id="s1",
                index=0,
                beat="hook",
                duration_sec=3.0,
                engine="ai_video",
                visual_prompt="a",
            ),
            Shot(
                id="s2",
                index=1,
                beat="proof",
                duration_sec=3.0,
                engine="motion_card",
                visual_prompt="b",
            ),
            Shot(
                id="s3",
                index=2,
                beat="cta",
                duration_sec=3.0,
                engine="motion_card",
                visual_prompt="c",
            ),
        ],
    )
    cues = build_transition_sfx_cues(sheet, existing_sfx=[])
    assert len(cues) >= 2
    assert any(abs(c.at_sec - 2.65) < 0.35 for c in cues)

    # Generate rhythmic pulsed WAV and verify beat detection runs cleanly
    wav_path = tmp_path / "pulse.wav"
    import math
    import struct
    import wave

    sr = 16000
    n = sr * 3
    frames = bytearray()
    for i in range(n):
        t = i / sr
        beat_phase = t % 0.5
        amp = 28000 if beat_phase < 0.06 else 1200
        val = int(amp * math.sin(2.0 * math.pi * 180.0 * t))
        frames.extend(struct.pack("<h", val))
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(bytes(frames))

    detected = await detect_audio_beats(wav_path)
    assert isinstance(detected, list)
    assert len(detected) >= 2


# ---------------------------------------------------------------------------
# Upgrade 3: J-Cut VO Lead-In & Unified Cinema Finishing Grade
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stitcher_j_cut_and_cinema_finishing_grade(
    tmp_path: Path, sample_creative_bible: CreativeBible
) -> None:
    # Build two short kinetic cards as clips
    c1 = tmp_path / "c1.mp4"
    c2 = tmp_path / "c2.mp4"
    await render_kinetic_card_mp4(
        text="Shot One",
        output_path=c1,
        duration_sec=2.0,
        width=480,
        height=270,
        fps=24,
        bible=sample_creative_bible,
        bg_rgb=(17, 24, 39),
        text_rgb=(255, 255, 255),
        accent_rgb=(226, 163, 54),
    )
    await render_kinetic_card_mp4(
        text="Shot Two",
        output_path=c2,
        duration_sec=2.0,
        width=480,
        height=270,
        fps=24,
        bible=sample_creative_bible,
        bg_rgb=(17, 24, 39),
        text_rgb=(255, 255, 255),
        accent_rgb=(226, 163, 54),
    )

    # Synth VO wav
    vo1 = tmp_path / "vo1.wav"
    vo2 = tmp_path / "vo2.wav"
    from gtm_videogen.api.studio_steps import _synthesize_mock_audition_wav

    vo1.write_bytes(_synthesize_mock_audition_wav("voice", "Rachel"))
    vo2.write_bytes(_synthesize_mock_audition_wav("voice", "George"))

    sheet = ShotSheet(
        target_duration_sec=4.0,
        fps=24,
        aspect="16:9",
        shots=[
            Shot(
                id="s1",
                index=0,
                beat="hook",
                duration_sec=2.0,
                engine="motion_card",
                visual_prompt="a",
                transition="cut",
            ),
            Shot(
                id="s2",
                index=1,
                beat="payoff",
                duration_sec=2.0,
                engine="motion_card",
                visual_prompt="b",
                transition="j_cut",
            ),
        ],
    )
    sound_plan = SoundPlan(
        vo_cues=[
            VoiceCue(shot_id="s1", at_sec=0.0, duration_sec=1.5, text="Shot one line"),
            VoiceCue(shot_id="s2", at_sec=2.0, duration_sec=1.5, text="Shot two line"),
        ],
        sfx_cues=[],
        music_cue=MusicCue(prompt="minimal synth", duration_sec=4.0),
    )
    out = tmp_path / "final_graded.mp4"
    stitcher = FFmpegStitcher()
    manifest = await stitcher.stitch(
        clips=[
            RenderedClip(
                shot_id="s1", path=str(c1), duration_sec=2.0, fps=24, width=480, height=270
            ),
            RenderedClip(
                shot_id="s2", path=str(c2), duration_sec=2.0, fps=24, width=480, height=270
            ),
        ],
        sound_plan=sound_plan,
        audio_assets=[
            RenderedAudio(kind="vo", cue_id="s1", path=str(vo1), duration_sec=1.5),
            RenderedAudio(kind="vo", cue_id="s2", path=str(vo2), duration_sec=1.5),
        ],
        sheet=sheet,
        bible=sample_creative_bible,
        output_path=out,
        width=480,
        height=270,
        preview=False,
    )
    assert manifest.error is None
    assert out.is_file()
    probe = await probe_media(out)
    assert probe.has_video and probe.has_audio
    assert abs(probe.duration_sec - 4.0) < 0.25


# ---------------------------------------------------------------------------
# Upgrade 4: Kling v4 Pro Start->End Bridging + Google Veo 3 Payloads
# ---------------------------------------------------------------------------


def test_video_payloads_kling_start_end_and_veo3() -> None:
    # Kling v4 Pro I2V Start->End bridging
    shot_bridge = Shot(
        id="s1",
        index=0,
        beat="hook",
        duration_sec=4.0,
        engine="image_to_video",
        visual_prompt="Smooth studio dolly around espresso machine",
        reference_image="https://cdn.example.com/kf1.jpg",
        end_reference_image="https://cdn.example.com/kf2.jpg",
    )
    p_pro = build_fal_video_payload(
        shot_bridge, aspect="16:9", fps=24, model="kling/v4/image-to-video"
    )
    assert p_pro["start_image_url"] == "https://cdn.example.com/kf1.jpg"
    assert p_pro["end_image_url"] == "https://cdn.example.com/kf2.jpg"

    # Kling v4 Flash does NOT emit end_image_url
    p_flash = build_fal_video_payload(
        shot_bridge, aspect="16:9", fps=24, model="kling/v4/flash/image-to-video"
    )
    assert p_flash["start_image_url"] == "https://cdn.example.com/kf1.jpg"
    assert "end_image_url" not in p_flash

    # Google Veo 3 T2V + I2V payload
    shot_t2v = Shot(
        id="s2",
        index=1,
        beat="hero",
        duration_sec=5.5,
        engine="ai_video",
        visual_prompt="Cinematic sunlit product reveal",
    )
    p_veo_t2v = build_fal_video_payload(shot_t2v, aspect="16:9", fps=24, model="fal-ai/veo3")
    assert p_veo_t2v["duration"] == "6s"
    assert p_veo_t2v["generate_audio"] is False
    assert p_veo_t2v["auto_fix"] is True

    p_veo_i2v = build_fal_video_payload(
        shot_bridge, aspect="16:9", fps=24, model="fal-ai/veo3/image-to-video"
    )
    assert p_veo_i2v["image_url"] == "https://cdn.example.com/kf1.jpg"
    assert p_veo_i2v["duration"] == "4s"

    # Adjacent keyframe bridging helper
    sheet = ShotSheet(
        target_duration_sec=8.0,
        shots=[
            Shot(
                id="s1",
                index=0,
                beat="hook",
                duration_sec=4.0,
                engine="ai_video",
                visual_prompt="a",
                keyframe_url="https://cdn.example.com/kf1.jpg",
                keyframe_approved=True,
            ),
            Shot(
                id="s2",
                index=1,
                beat="proof",
                duration_sec=4.0,
                engine="ai_video",
                visual_prompt="b",
                keyframe_url="https://cdn.example.com/kf2.jpg",
                keyframe_approved=True,
            ),
        ],
    )
    _wire_adjacent_end_keyframes(sheet)
    assert sheet.shots[0].engine == "image_to_video"
    assert sheet.shots[0].reference_image == "https://cdn.example.com/kf1.jpg"
    assert sheet.shots[0].end_reference_image == "https://cdn.example.com/kf2.jpg"


# ---------------------------------------------------------------------------
# Upgrade 2, 4, 5 Studio API Integration: Audition, Reference Taste, Auto-Polish, Flagship Override
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_studio_v2_endpoints_end_to_end(tmp_path: Path) -> None:
    app = create_app(data_dir=tmp_path)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver", headers={"X-Actor-Id": "v2tester"}
    ) as client:
        # Health features flag
        h = await client.get("/api/health")
        assert h.status_code == 200
        assert h.json()["features"]["upgrades_v2"] is True

        # Presets expose flagship_models & craft_archetypes
        pr = await client.get("/api/studio/presets")
        assert pr.status_code == 200
        pr_json = pr.json()
        assert len(pr_json["flagship_models"]) >= 4
        assert "kinetic_saas" in pr_json["craft_archetypes"]

        # Upgrade 2: Voice & Music Audition (mock)
        aud_v = await client.post(
            "/api/studio/audition",
            json={
                "kind": "voice",
                "voice_id": "George",
                "text": "Precision engineered.",
                "mock": True,
            },
        )
        assert aud_v.status_code == 200
        assert aud_v.headers["content-type"].startswith("audio/")
        assert aud_v.content[:4] == b"RIFF"

        aud_m = await client.post(
            "/api/studio/audition",
            json={"kind": "music", "prompt": "Minimal synth", "mock": True},
        )
        assert aud_m.status_code == 200
        assert aud_m.content[:4] == b"RIFF"

        # Create mock run
        cr = await client.post(
            "/api/runs",
            json={
                "brand_name": "Kestrel Roasters",
                "brand_url": "https://kestrelroasters.example.com",
                "goal": "Drive single-origin coffee subscriptions.",
                "target_duration_sec": 9,
                "aspect": "16:9",
                "fps": 24,
                "audio_mode": "both",
                "cap_usd": 5.0,
                "mock": True,
                "facts": ["Small-batch roasted within 24 hours"],
            },
        )
        assert cr.status_code == 201
        run_id = cr.json()["run_id"]

        # Upgrade 5: Apply Craft Archetype + Reference Moodboard image
        ref = await client.post(
            f"/api/runs/{run_id}/reference",
            json={
                "craft_archetype": "quiet_editorial",
                "reference_image_data_url": _make_png_data_url(),
                "notes": "Keep transitions unhurried.",
            },
        )
        assert ref.status_code == 200
        ref_data = ref.json()
        assert ref_data["craft_archetype"] == "quiet_editorial"
        assert "Quiet Editorial" in ref_data["reference_taste_notes"]
        assert "Quiet Editorial" in (ref_data["run"]["input"]["user_direction"] or "")

        # Step 1 Research -> Step 2 Direct -> Step 5 Preview
        assert (await client.post(f"/api/runs/{run_id}/step/research")).status_code == 202
        await _wait_idle(client, run_id)
        assert (await client.post(f"/api/runs/{run_id}/step/direct")).status_code == 202
        await _wait_idle(client, run_id)

        pv = await client.post(
            f"/api/runs/{run_id}/step/previews",
            json={
                "stacks": [
                    {
                        "id": "mock_pacing",
                        "label": "Pacing ($0)",
                        "video_model": "mock-lavfi-mp4",
                        "voice_model": "mock-lavfi-wav",
                        "mock": True,
                    }
                ]
            },
        )
        assert pv.status_code == 202
        st_pv = await _wait_idle(client, run_id)
        assert len(st_pv["versions"]) == 1
        v0_id = st_pv["versions"][0]["version_id"]

        # Upgrade 5: Autonomous Auto-Polish Loop
        ap = await client.post(
            f"/api/runs/{run_id}/auto-polish",
            json={
                "version_id": v0_id,
                "target_score": 9.8,
                "max_rounds": 1,
                "notes": "Sharper opening",
            },
        )
        assert ap.status_code == 202
        st_ap = await _wait_idle(client, run_id, timeout=45.0)
        assert len(st_ap["versions"]) >= 2
        assert st_ap["winning_version_id"] is not None

        # Upgrade 4: Step 6 Final Cut with video_model_override
        fin = await client.post(
            f"/api/runs/{run_id}/step/final",
            json={
                "version_id": st_ap["winning_version_id"],
                "mock": True,
                "video_model_override": "fal-ai/veo3/fast",
            },
        )
        assert fin.status_code == 202
        st_fin = await _wait_idle(client, run_id, timeout=45.0)
        assert st_fin["status"] == "done"
        assert st_fin["render_manifest"]["stitched_path"]
        assert st_fin["versions"][-1]["stack"]["video_model"] == "fal-ai/veo3/fast"
        assert "fal-ai/veo3/fast" in st_fin["versions"][-1]["label"]


def test_modality_prompt_compilers(sample_creative_bible: CreativeBible) -> None:
    from gtm_videogen.models.prompt_compiler import (
        compile_image_prompt,
        compile_sfx_prompt,
        compile_video_prompt,
    )

    detailed = Shot(
        id="s1",
        index=0,
        beat="hero_reveal",
        duration_sec=4.0,
        engine="image_to_video",
        visual_prompt="Anodized aluminum kettle resting on dark slate countertop",
        still_prompt="Rule-of-thirds composition, kettle on right third, negative space on left",
        motion_prompt="Slow 8-degree orbital push-in as delicate steam curls upward from spout",
        subject_and_materials="Matte black powder-coated steel body with walnut handle",
        environment="Minimalist sunlit architectural kitchen with honed basalt slab",
        lighting_rig="45-degree diffused north window key light + warm 3200K rim glint",
        lens_mm=85,
        camera_move="slow 8-degree orbital push-in",
        camera_angle="three_quarter_close_up",
        motion_curve="ease-in-out silky glide",
        color_grade="5400K neutral highlights, deep obsidian shadows, warm amber specular",
        sfx_note="Crisp metallic lid click at 0.3s followed by gentle simmering kettle hiss",
        vo_delivery="Close-mic calm editorial authority",
    )

    img_p = compile_image_prompt(detailed, sample_creative_bible, kontext_anchor=True)
    assert "Rule-of-thirds composition" in img_p
    assert "85mm cinema prime lens" in img_p
    assert "45-degree diffused north window" in img_p
    assert "Matte black powder-coated steel" in img_p

    vid_p = compile_video_prompt(detailed, sample_creative_bible)
    assert "Slow 8-degree orbital push-in" in vid_p
    assert "ease-in-out silky glide" in vid_p
    assert "85mm lens" in vid_p
    assert "no morphing" in vid_p

    sfx_p = compile_sfx_prompt(detailed)
    assert sfx_p is not None and "Crisp metallic lid click" in sfx_p
    assert "honed basalt slab" in sfx_p

