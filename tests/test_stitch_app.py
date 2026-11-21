"""Comprehensive unit & integration tests for Stitch (Concept C · Tactile Director's Pinboard)."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from gtm_videogen.api.app import create_app
from gtm_videogen.core.config import Settings
from gtm_videogen.models.model_registry import (
    ModelRegistryStore,
    VideoModelSpec,
    build_registry_fal_payload,
)


def _tiny_png_data_url(color: tuple[int, int, int] = (229, 77, 58)) -> str:
    img = Image.new("RGB", (64, 64), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _make_client(tmp_path: Path) -> TestClient:
    settings = Settings(
        data_dir=tmp_path,
        fal_key="",
        openrouter_api_key="",
    )
    app = create_app(data_dir=tmp_path, settings=settings, rate_limit_requests=0)
    return TestClient(app)


def test_dynamic_model_registry_and_payload_builder(tmp_path: Path) -> None:
    store = ModelRegistryStore(tmp_path / "stitch")
    models = store.list_video_models()
    assert any(m.id == "fal-ai/kling-video/v3/pro/text-to-video" for m in models)

    # Register a future custom model with zero Python changes
    future_spec = VideoModelSpec(
        id="fal-ai/future-cinema/v5/text-to-video",
        i2v_id="fal-ai/future-cinema/v5/image-to-video",
        label="Future Cinema v5",
        durations_sec=[6, 12],
        default_duration_sec=12,
        duration_format="seconds_suffix",
        max_prompt_chars=1800,
        supports_start_frame=True,
        supports_end_frame=True,
        supports_elements=True,
        supports_native_audio=True,
        est_usd_per_sec=0.22,
    )
    store.add_custom_model(future_spec)
    loaded = store.get_video_model("fal-ai/future-cinema/v5/text-to-video")
    assert loaded.custom is True
    assert loaded.snap_duration(10) == 12

    endpoint, payload = build_registry_fal_payload(
        loaded,
        prompt="A" * 2500,
        duration_sec=10,
        aspect_ratio="16:9",
        start_image="https://example.com/start.jpg",
        end_image="https://example.com/end.jpg",
        generate_audio=True,
        element_image_urls=["data:image/jpeg;base64,abc", "https://example.com/el2.png"],
    )
    assert endpoint == "fal-ai/future-cinema/v5/image-to-video"
    assert payload["duration"] == "12s"
    assert len(payload["prompt"]) == 1800
    assert payload["start_image_url"] == "https://example.com/start.jpg"
    assert payload["end_image_url"] == "https://example.com/end.jpg"
    assert payload["generate_audio"] is True
    assert len(payload["elements"]) == 2

    assert store.delete_custom_model("fal-ai/future-cinema/v5/text-to-video") is True
    assert store.delete_custom_model("fal-ai/kling-video/v3/pro/text-to-video") is False


def test_root_serves_stitch_ui_and_gallery_photos(tmp_path: Path) -> None:
    client = _make_client(tmp_path)
    r = client.get("/")
    assert r.status_code == 200
    assert "Stitch" in r.text
    assert "Tactile Director's Pinboard" in r.text

    cat = client.get("/api/stitch/catalog")
    assert cat.status_code == 200
    body = cat.json()
    assert len(body["gallery"]) >= 18
    assert len(body["motion_templates"]) >= 10
    assert len(body["video_models"]) >= 6

    img_r = client.get("/api/stitch/gallery/jar_open.jpg")
    assert img_r.status_code == 200
    assert len(img_r.content) > 10_000


def test_stitch_full_four_stage_workflow_and_http_206(tmp_path: Path) -> None:
    client = _make_client(tmp_path)

    # 1. Dynamic Model Registry REST CRUD
    add_m = client.post(
        "/api/stitch/models",
        json={
            "id": "fal-ai/test-studio/v1/text-to-video",
            "label": "Test Studio v1",
            "durations_sec": [5, 10],
            "default_duration_sec": 5,
            "duration_format": "string",
            "max_prompt_chars": 2000,
        },
    )
    assert add_m.status_code == 200
    assert any(m["id"] == "fal-ai/test-studio/v1/text-to-video" for m in add_m.json()["video_models"])
    del_m = client.delete("/api/stitch/models/fal-ai/test-studio/v1/text-to-video")
    assert del_m.status_code == 200

    # 2. Stage 1: Pitch 3 Visual Concepts
    pitch_r = client.post(
        "/api/stitch/pitch",
        json={"idea": "Anodized coffee kettle steaming in morning sunlight", "mock": True},
    )
    assert pitch_r.status_code == 200
    pitches = pitch_r.json()["pitches"]
    assert len(pitches) == 3

    # 3. Create 20s 2-Scene Chained Project
    create_r = client.post(
        "/api/stitch/projects",
        json={
            "idea": "Liquid gold awakens a miniature tabletop wooden city inside a glass jar",
            "target_duration_sec": 20,
            "aspect_ratio": "16:9",
            "video_model_id": "fal-ai/kling-video/v3/pro/text-to-video",
            "recipe_id": "miniature_jar_world",
            "mock": True,
        },
    )
    assert create_r.status_code == 200
    proj = create_r.json()
    pid = proj["project_id"]
    assert len(proj["scenes"]) == 2
    assert proj["scenes"][1]["continues_from_prev"] is True
    for sc in proj["scenes"]:
        assert sc["compiled_prompt_chars"] <= sc["max_prompt_chars"]
        assert isinstance(sc["compiled_prompt_dict"], dict)

    # 4. Stage 2: Trigger Script Gate warning (garbled text + broken continuity) & Auto-Fix
    scenes_mod = json.loads(json.dumps(proj["scenes"]))
    scenes_mod[0]["end_frame"] = "A glowing neon sign reading HELLO WORLD on the street"
    scenes_mod[1]["beats"][0]["scene"] = "A completely new alley appears"
    patch_r = client.patch(f"/api/stitch/projects/{pid}", json={"scenes": scenes_mod})
    assert patch_r.status_code == 200
    patched = patch_r.json()
    assert patched["script_gate"]["valid"] is False
    assert patched["script_gate"]["total_warnings"] >= 2

    fix_r = client.post(f"/api/stitch/projects/{pid}/autofix")
    assert fix_r.status_code == 200
    fixed = fix_r.json()
    assert fixed["script_gate"]["valid"] is True
    assert fixed["script_gate"]["total_warnings"] == 0

    # Add a 3rd Scene, then delete it back to 2 Scenes
    add_sc = client.post(
        f"/api/stitch/projects/{pid}/scenes",
        json={"title": "Scene 3 · Epilogue", "duration_sec": 5, "kind": "kinetic_ui", "continues_from_prev": True},
    )
    assert add_sc.status_code == 200
    assert len(add_sc.json()["scenes"]) == 3
    sc3_id = add_sc.json()["scenes"][2]["scene_id"]
    del_sc = client.delete(f"/api/stitch/projects/{pid}/scenes/{sc3_id}")
    assert del_sc.status_code == 200
    assert len(del_sc.json()["scenes"]) == 2

    # 5. Stage 3: Upload @Element1, generate all anchor stills, tweak 1 still, upload 1 custom still
    el_r = client.post(
        f"/api/stitch/projects/{pid}/elements",
        json={
            "slot": 1,
            "role": "product",
            "label": "Glass Apothecary Jar",
            "filename": "jar.png",
            "data_url": _tiny_png_data_url(),
        },
    )
    assert el_r.status_code == 200
    assert len(el_r.json()["elements"]) == 1
    assert el_r.json()["elements"][0]["tag"] == "@Element1"

    kf_all = client.post(f"/api/stitch/projects/{pid}/keyframes", json={"mock": True})
    assert kf_all.status_code == 200
    s1_id = kf_all.json()["scenes"][0]["scene_id"]
    s2_id = kf_all.json()["scenes"][1]["scene_id"]
    assert kf_all.json()["scenes"][0]["start_keyframe_media_url"]
    assert kf_all.json()["scenes"][1]["end_keyframe_media_url"]

    kf_tweak = client.post(
        f"/api/stitch/projects/{pid}/scenes/{s1_id}/keyframe",
        json={"which": "start", "tweak_note": "warmer golden rim light", "mock": True},
    )
    assert kf_tweak.status_code == 200

    kf_up = client.post(
        f"/api/stitch/projects/{pid}/scenes/{s1_id}/keyframe",
        json={"which": "end", "upload_data_url": _tiny_png_data_url((40, 120, 90))},
    )
    assert kf_up.status_code == 200

    # Render Scene 1 Take 1 + Take 2, switch winning take back to Take 1
    r1_t1 = client.post(
        f"/api/stitch/projects/{pid}/scenes/{s1_id}/render",
        json={"engine_id": "mock", "mock": True, "sync": True},
    )
    assert r1_t1.status_code == 200
    assert len(r1_t1.json()["scenes"][0]["takes"]) == 1

    r1_t2 = client.post(
        f"/api/stitch/projects/{pid}/scenes/{s1_id}/render",
        json={"engine_id": "mock", "mock": True, "sync": True},
    )
    assert r1_t2.status_code == 200
    assert len(r1_t2.json()["scenes"][0]["takes"]) == 2
    assert r1_t2.json()["scenes"][0]["winning_take_id"] == 2

    sel_t1 = client.post(
        f"/api/stitch/projects/{pid}/scenes/{s1_id}/select-take",
        json={"take_id": 1},
    )
    assert sel_t1.status_code == 200
    assert sel_t1.json()["scenes"][0]["winning_take_id"] == 1

    # Render Scene 2 — MUST automatically pixel-chain from Scene 1's winning Take 1!
    r2_t1 = client.post(
        f"/api/stitch/projects/{pid}/scenes/{s2_id}/render",
        json={"engine_id": "mock", "mock": True, "sync": True},
    )
    assert r2_t1.status_code == 200
    assert r2_t1.json()["scenes"][1]["takes"][0]["chained_from_prev"] is True

    # 6. Stage 4: Stitch Master Cut with micro_dissolve + Brand Logo Outro & verify HTTP 206 streaming
    logo_r = client.post(
        f"/api/stitch/projects/{pid}/logo",
        json={"filename": "logo.png", "data_url": _tiny_png_data_url((255, 255, 255))},
    )
    assert logo_r.status_code == 200

    stitch_r = client.post(
        f"/api/stitch/projects/{pid}/stitch",
        json={
            "audio_mode": "native_plus_score",
            "seam_transition": "micro_dissolve",
            "add_brand_outro": True,
            "brand_name": "Stitch Cinema",
            "brand_tagline": "Directed frame by frame",
            "mock": True,
        },
    )
    assert stitch_r.status_code == 200
    master_url = stitch_r.json()["master_video_url"]
    assert master_url

    # Verify HTTP 206 Byte-Range streaming returns exact 1024 bytes
    rng = client.get(master_url, headers={"Range": "bytes=0-1023"})
    assert rng.status_code == 206
    assert len(rng.content) == 1024
    assert rng.headers["content-range"].startswith("bytes 0-1023/")


def test_theme_adapts_user_brand_idea_and_progress_bar_markup(tmp_path: Path) -> None:
    client = _make_client(tmp_path)

    # UI exposes non-destructive theme selector & live progress bar
    html = client.get("/").text
    assert "selectGalleryTheme(" in html
    assert "status-progress-fill" in html
    assert "active-theme-pill" in html

    # Applying PaperCraft theme to a custom brand + idea preserves user idea in PaperCraft world
    res = client.post(
        "/api/stitch/projects",
        json={
            "idea": "Unfolding our ultralight carbon running shoe for marathon launch",
            "brand_name": "AeroStride",
            "world_preset": "paper_origami",
            "camera_preset": "overhead_topdown",
            "lighting_preset": "golden_hour",
            "motion_preset": "seamless_morph",
            "recipe_id": "cotton_origami_city",
            "target_duration_sec": 15,
            "mock": True,
        },
    )
    assert res.status_code == 200
    proj = res.json()
    assert "AeroStride" in proj["brand_name"]
    assert "running shoe" in proj["idea"].lower()
    assert "papercraft" in proj["style"].lower()
    assert "cotton paper" in proj["environment"]["materials"].lower()
    assert "origami" in proj["subject_details"]["appearance"].lower()
    assert "Tactile Paper & Craft" in proj["scenes"][0]["title"]

    # Selecting miniature_jar_world theme with a custom idea does NOT overwrite with the jar sample
    res2 = client.post(
        "/api/stitch/projects",
        json={
            "idea": "Solar powered espresso maker on a mountain camp table",
            "brand_name": "PeakBrew",
            "world_preset": "miniature_diorama",
            "recipe_id": "miniature_jar_world",
            "target_duration_sec": 15,
            "mock": True,
        },
    )
    assert res2.status_code == 200
    proj2 = res2.json()
    assert "espresso" in proj2["idea"].lower()
    assert proj2["title"] != "Miniature World in a Jar"

