"""Unit & end-to-end tests for the Guided Structured JSON Director (10s / 20s / 30s)."""

from __future__ import annotations

import base64
import io
from pathlib import Path

import httpx
import pytest
from PIL import Image

from gtm_videogen.agents.director_studio import (
    encode_image_data_uri,
    extract_last_frame_jpeg,
    generate_director_blueprint,
    refine_director_blueprint,
)
from gtm_videogen.api.app import create_app
from gtm_videogen.models.schemas import Shot
from gtm_videogen.providers.video import build_fal_video_payload


def test_fal_payload_supports_json_raw_prompt_native_audio_and_data_uri() -> None:
    """Kling v4 Pro accepts data:image/ Last-Frame Chaining URIs; Veo 3 honors generate_audio."""
    data_uri = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/"
    shot = Shot(
        id="act2",
        index=1,
        beat="Act 2",
        duration_sec=10.0,
        engine="image_to_video",
        visual_prompt="Fallback prompt",
        reference_image=data_uri,
        end_reference_image=data_uri,
    )
    kling_payload = build_fal_video_payload(
        shot,
        "16:9",
        24,
        model="kling/v4/image-to-video",
        raw_prompt='{"type": "video_prompt", "duration": "10 seconds"}',
    )
    assert kling_payload["prompt"] == '{"type": "video_prompt", "duration": "10 seconds"}'
    assert kling_payload["duration"] == 10
    assert kling_payload["start_image_url"] == data_uri
    assert kling_payload["end_image_url"] == data_uri

    veo_payload = build_fal_video_payload(
        shot,
        "16:9",
        24,
        model="fal-ai/veo3/image-to-video",
        raw_prompt='{"type": "video_prompt", "duration": "8 seconds"}',
        generate_audio=True,
    )
    assert veo_payload["duration"] == "8s"
    assert veo_payload["generate_audio"] is True
    assert veo_payload["image_url"] == data_uri

    kling3_payload = build_fal_video_payload(
        shot,
        "16:9",
        24,
        model="fal-ai/kling-video/v3/pro/image-to-video",
        raw_prompt='{"type": "video_prompt", "duration": "10 seconds"}',
        generate_audio=True,
    )
    assert kling3_payload["duration"] == "10"
    assert kling3_payload["generate_audio"] is True
    assert kling3_payload["start_image_url"] == data_uri
    assert kling3_payload["end_image_url"] == data_uri


@pytest.mark.asyncio
async def test_director_blueprint_generation_10s_20s_30s_and_refine() -> None:
    """Verify 10s (1 Act), 20s (2 Chained Acts), and 30s Hybrid Launch JSON compilation."""
    bp10 = await generate_director_blueprint(
        idea="Miniature living city awakening in a glass jar",
        duration_sec=10,
        recipe_id="miniature_jar_world",
        mock=True,
    )
    assert len(bp10.acts) == 1
    assert len(bp10.acts[0].timeline) == 5
    master10 = bp10.to_master_prompt_dict()
    assert master10["type"] == "video_prompt"
    assert master10["duration"] == "10 seconds"
    assert master10["resolution"] == "4K"
    assert "camera" in master10 and "environment" in master10 and "lighting" in master10
    assert "final_frame" in master10 and len(master10["timeline"]) == 5

    # Refine globally + single beat
    await refine_director_blueprint(
        bp10,
        instruction="Closer macro & shallow bokeh with zero readable text",
        mock=True,
    )
    assert bp10.camera.lens_mm == 100
    assert any("zero readable text" in c.lower() for c in bp10.negative_constraints)

    await refine_director_blueprint(
        bp10,
        instruction="Add glowing firefly trails",
        act_number=1,
        beat_index=2,
        mock=True,
    )
    assert "glowing firefly trails" in bp10.acts[0].timeline[2].action

    # 30s Hybrid Launch (Act 1 Hero -> Act 2 Kinetic UI -> Act 3 Hero Reveal)
    bp30 = await generate_director_blueprint(
        idea="Dark obsidian espresso grinder sculpting golden crema",
        duration_sec=30,
        format_mode="hybrid_launch",
        brand_name="Fellow Opus",
        brand_tagline="Precision grind, every morning.",
        mock=True,
    )
    assert len(bp30.acts) == 3
    assert bp30.acts[0].mode == "cinematic_sequence"
    assert bp30.acts[1].mode == "kinetic_ui_showcase"
    assert bp30.acts[2].mode == "cinematic_sequence"
    master30 = bp30.to_master_prompt_dict()
    assert master30["duration"] == "30 seconds"
    assert len(master30["acts"]) == 3


@pytest.mark.asyncio
async def test_director_api_end_to_end_30s_render_and_last_frame_extract(tmp_path: Path) -> None:
    """Full API lifecycle: presets -> 30s create -> refine -> upload -> keyframes -> render -> 206 stream."""
    app = create_app(data_dir=tmp_path)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        # 1. Presets
        p_resp = await client.get("/api/director/presets")
        assert p_resp.status_code == 200
        presets = p_resp.json()
        assert len(presets["recipes"]) >= 6
        assert len(presets["worlds"]) >= 5
        assert len(presets["refinement_chips"]) >= 6

        # 2. Create 30s Hybrid Launch blueprint
        create_resp = await client.post(
            "/api/director/blueprints",
            json={
                "idea": "Handcrafted miniature coffee village turning into a gleaming obsidian espresso machine",
                "duration_sec": 30,
                "format_mode": "hybrid_launch",
                "continuity_mode": "last_frame_chain",
                "aspect_ratio": "16:9",
                "engine": "mock",
                "brand_name": "Kestrel",
                "brand_tagline": "Brewed to cinema craft.",
                "add_brand_endcard": True,
                "mock": True,
            },
        )
        assert create_resp.status_code == 201
        bp = create_resp.json()
        bp_id = bp["blueprint_id"]
        assert len(bp["acts"]) == 3
        assert len(bp["compiled_act_jsons"]) == 3

        # 3. Refine via chip
        ref_resp = await client.post(
            f"/api/director/blueprints/{bp_id}/refine",
            json={"instruction": "More volumetric light and warmer sunset glow", "mock": True},
        )
        assert ref_resp.status_code == 200
        assert "volumetric" in ref_resp.json()["lighting"]["style"].lower()

        # 4. Upload small PNG product image + logo
        img_buf = io.BytesIO()
        Image.new("RGB", (120, 80), (226, 163, 54)).save(img_buf, format="PNG")
        b64_png = "data:image/png;base64," + base64.b64encode(img_buf.getvalue()).decode("ascii")
        up_resp = await client.post(
            f"/api/director/blueprints/{bp_id}/upload",
            json={"kind": "product", "filename": "hero.png", "data_base64": b64_png},
        )
        assert up_resp.status_code == 200
        assert up_resp.json()["product_image_url"]

        # 5. Generate continuity keyframes
        kf_resp = await client.post(
            f"/api/director/blueprints/{bp_id}/keyframes",
            json={"mock": True},
        )
        assert kf_resp.status_code == 200
        kf_bp = kf_resp.json()
        assert kf_bp["status"] == "keyframes_ready"
        kf_url = kf_bp["acts"][0]["start_keyframe_media_url"]
        assert kf_url
        img_resp = await client.get(kf_url)
        assert img_resp.status_code == 200
        assert len(img_resp.content) > 1000

        # 6. Render 30s (3 Acts + Brand End-Card) synchronously in offline mode
        ren_resp = await client.post(
            f"/api/director/blueprints/{bp_id}/render",
            json={"engine": "mock", "mock": True, "sync": True},
        )
        assert ren_resp.status_code == 202
        done_bp = ren_resp.json()
        assert done_bp["status"] == "done", f"Render failed: {done_bp.get('error')}"
        video_url = done_bp["final_video_url"]
        assert video_url

        # 7. Verify HTTP 206 Range streaming works cleanly
        range_resp = await client.get(video_url, headers={"Range": "bytes=0-1023"})
        assert range_resp.status_code == 206
        assert len(range_resp.content) == 1024

        # 8. Verify Last-Frame Extraction & Data-URI Encoding on the actual rendered MP4
        final_mp4_path = Path(done_bp["final_video_path"])
        extracted = await extract_last_frame_jpeg(final_mp4_path, tmp_path / "last_frame.jpg")
        assert extracted.is_file() and extracted.stat().st_size > 500
        uri = encode_image_data_uri(extracted)
        assert uri.startswith("data:image/jpeg;base64,") and len(uri) > 500

        # 9. Verify listing endpoint returns our blueprint
        list_resp = await client.get("/api/director/blueprints")
        assert list_resp.status_code == 200
        assert any(b["blueprint_id"] == bp_id and b["has_video"] for b in list_resp.json()["blueprints"])
