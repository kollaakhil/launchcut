"""Comprehensive unit & end-to-end tests for Launch Cut Cinema Studio (v2 Rethought)."""

from __future__ import annotations

import base64
import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from gtm_videogen.agents.cinema_writers_room import _synthesize_mock_dialogue_wav
from gtm_videogen.api.app import create_app
from gtm_videogen.core.config import Settings
from gtm_videogen.media._subprocess import probe_media
from gtm_videogen.models.cinema_compiler import (
    choose_smart_anchor_strategy,
    compile_end_keyframe_prompt,
    compile_shot_video_prompt,
    compile_start_keyframe_prompt,
    ensure_kling_element_dimensions,
    hydrate_entities_in_text,
    slice_three_panel_turnaround,
)
from gtm_videogen.models.cinema_project import (
    CardinalZones,
    CharacterAsset,
    CinemaScene,
    CinemaShot,
    DialogueCueSpec,
    FrameStateSpec,
    LocationAsset,
    PropAsset,
    ShotCameraSpec,
    TasteReactionEvent,
    WorldVault,
)
from gtm_videogen.models.model_registry import (
    ModelRegistryStore,
    build_registry_fal_payload,
)
from gtm_videogen.providers.lipsync import build_duration_locked_speech_wav
from gtm_videogen.rules.cinema_taste import (
    TasteProfileStore,
    compile_taste_signature,
)


def _tiny_jpeg_data_url(color: tuple[int, int, int] = (212, 140, 48)) -> str:
    img = Image.new("RGB", (96, 96), color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _make_client(tmp_path: Path) -> TestClient:
    settings = Settings(
        data_dir=tmp_path,
        fal_key="",
        openrouter_api_key="",
    )
    app = create_app(data_dir=tmp_path, settings=settings, rate_limit_requests=0)
    return TestClient(app)


def test_taste_translator_and_persistent_taste_memory(tmp_path: Path) -> None:
    store = TasteProfileStore(tmp_path / "cinema")
    sig = compile_taste_signature(
        mood_presets=["cinema_noir"],
        user_vibe_note="quiet intimate rainy night, slow contemplative pace",
    )
    assert sig.dials.mood_contrast >= 0.85
    assert sig.dials.scale < 0.48
    assert sig.dials.pacing < 0.56
    assert "anamorphic" in sig.compiled_craft_rules.lens_package.lower()

    ev = TasteReactionEvent(
        target_type="shot",
        target_id="SC01_SH01",
        chip_or_note="warmer_intimate",
        polarity="adjust",
    )
    prof = store.record_reaction(actor_id="director_1", event=ev, active_signature=sig)
    assert prof.reactions_count == 1
    reloaded = store.get_profile("director_1")
    assert reloaded.reactions_count == 1


def test_entity_hydration_turnaround_slicing_and_kling_elements(tmp_path: Path) -> None:
    vault = WorldVault(
        characters=[
            CharacterAsset(
                id="CHAR_MARA",
                name="Mara Vance",
                identity_lock="38yo woman with hazel eyes and eyebrow scar",
                wardrobe_lock="olive waxed field jacket over charcoal knit",
            )
        ],
        locations=[
            LocationAsset(
                id="LOC_LAB",
                name="Coastal Observatory",
                cardinal_zones=CardinalZones(
                    A_north_primary="rain-lashed storm glass above walnut desk",
                    B_south_reverse="green steel tape cabinets and iron bulkhead door",
                ),
            )
        ],
        props=[
            PropAsset(
                id="PROP_CORE",
                name="Quartz Cylinder",
                geometry_and_scale="12cm beveled optical quartz cylinder",
                surface_textures="knurled oxidized brass and clear quartz",
                states={
                    "dormant": "cold unlit glass",
                    "resonating": "glowing warm 2800K amber filament",
                },
                default_state="dormant",
            )
        ],
    )
    scene = CinemaScene(
        scene_id="SC_01",
        scene_number=1,
        location_id="LOC_LAB",
    )
    shot = CinemaShot(
        shot_id="SC01_SH01",
        scene_id="SC_01",
        shot_scale="close_up",
        characters_in_shot=["CHAR_MARA"],
        camera=ShotCameraSpec(
            facing_wall_zone="B_south_reverse",
            lens_mm=85,
            movement="subtle 5cm breathing hold",
        ),
        start_frame=FrameStateSpec(
            composition="Close-up on CHAR_MARA holding PROP_CORE inside LOC_LAB",
            prop_states={"PROP_CORE": "resonating"},
        ),
        end_frame=FrameStateSpec(
            composition="Tight close-up on CHAR_MARA's eyes illuminated by PROP_CORE",
            prop_states={"PROP_CORE": "resonating"},
        ),
        dialogue=DialogueCueSpec(
            speaker_id="CHAR_MARA",
            line="It waited forty years for tonight.",
            delivery_mode="on_screen_speaking",
        ),
    )

    # 1. Verify Trap #7: zero raw IDs leak into compiled prompts
    hydrated = hydrate_entities_in_text(
        "CHAR_MARA inspects PROP_CORE inside LOC_LAB",
        vault,
        {"PROP_CORE": "resonating"},
        element_tag_map={"CHAR_MARA": "@Element1"},
    )
    assert "CHAR_MARA" not in hydrated
    assert "PROP_CORE" not in hydrated
    assert "LOC_LAB" not in hydrated
    assert "@Element1" in hydrated
    assert "2800K amber filament" in hydrated

    # 2. Verify Trap #5: B_south_reverse wall is injected on reverse-angle shots
    start_p = compile_start_keyframe_prompt(shot, scene, vault, has_ref_strip=True)
    assert "green steel tape cabinets" in start_p
    end_p = compile_end_keyframe_prompt(shot, scene, vault, anchored_to_start=True)
    assert "Preserve the EXACT same person" in end_p

    # 3. Verify Trap #3: subtle speaking close-up avoids synthetic FLF2V puppet morph
    assert choose_smart_anchor_strategy(shot) == "start_and_elements_forward"

    # 4. Verify Trap #6: 3-panel turnaround slicing & Kling v3 @Elements bounds
    tall_skinny = Image.new("RGB", (120, 900), (30, 35, 42))
    safe_img = ensure_kling_element_dimensions(tall_skinny)
    w, h = safe_img.size
    assert min(w, h) >= 300 and max(w, h) <= 2048
    assert 0.4 <= (w / float(h)) <= 2.5

    sheet = Image.new("RGB", (1280, 720), (24, 28, 34))
    sheet_p = tmp_path / "sheet.jpg"
    sheet.save(sheet_p, format="JPEG")
    c1, c2, c3 = slice_three_panel_turnaround(sheet_p, tmp_path / "crops", "CHAR_MARA")
    assert c1.is_file() and c2.is_file() and c3.is_file()

    reg = ModelRegistryStore(tmp_path / "stitch")
    kling_spec = reg.get_video_model("fal-ai/kling-video/v3/pro/text-to-video")
    vid_prompt = compile_shot_video_prompt(shot, scene, vault, kling_spec)
    assert len(vid_prompt) <= kling_spec.max_prompt_chars

    _, payload = build_registry_fal_payload(
        kling_spec,
        prompt=vid_prompt,
        duration_sec=6,
        start_image="https://example.com/s.jpg",
        element_bundles=[
            {
                "frontal_image_url": "https://example.com/front.jpg",
                "reference_image_urls": [
                    "https://example.com/3q.jpg",
                    "https://example.com/full.jpg",
                ],
            }
        ],
    )
    assert payload["elements"][0]["frontal_image_url"] == "https://example.com/front.jpg"
    assert len(payload["elements"][0]["reference_image_urls"]) == 2


@pytest.mark.asyncio
async def test_silence_padded_lipsync_stem_matches_exact_clip_duration(tmp_path: Path) -> None:
    raw_speech = tmp_path / "line.wav"
    _synthesize_mock_dialogue_wav("Listen closely to the signal in the glass.", raw_speech)
    padded = tmp_path / "padded_stem.wav"

    await build_duration_locked_speech_wav(
        speech_audio_path=raw_speech,
        target_duration_sec=6.0,
        pre_pause_sec=0.8,
        out_wav_path=padded,
    )
    pr = await probe_media(padded)
    assert padded.is_file()
    assert abs(float(pr.duration_sec or 0.0) - 6.0) <= 0.08


def test_cinema_guided_workflow_2way_json_and_soundstage(tmp_path: Path) -> None:
    client = _make_client(tmp_path)

    # UI & Catalog
    ui_r = client.get("/cinema")
    assert ui_r.status_code == 200
    assert "Launch Cut Cinema" in ui_r.text
    assert "Bring one line and your taste" in ui_r.text

    stitch_r = client.get("/")
    assert stitch_r.status_code == 200
    assert "/cinema" in stitch_r.text

    cat_r = client.get("/api/cinema/catalog")
    assert cat_r.status_code == 200
    assert len(cat_r.json()["moods"]) >= 8

    # 1. Stage 1: Pitch 3 Treatments + Look-Test Stills
    create_r = client.post(
        "/api/cinema/projects",
        json={
            "idea": "At 2 AM in a rain-lashed coastal atelier, an engineer hears a warm harmonic inside a dormant glass cylinder",
            "target_duration_sec": 45,
            "aspect_ratio": "16:9",
            "workflow_mode": "guided",
            "mood_presets": ["cinema_a24", "cinema_noir"],
            "brand_name": "Aether Audio",
            "reference_images_b64": [_tiny_jpeg_data_url()],
            "mock": True,
        },
    )
    assert create_r.status_code == 200
    proj = create_r.json()
    pid = proj["project_id"]
    assert len(proj["treatments"]) == 3
    assert proj["treatments"][0]["look_test_url"] is not None

    # Verify Look-Test image serves cleanly
    look_r = client.get(proj["treatments"][0]["look_test_url"])
    assert look_r.status_code == 200
    assert len(look_r.content) > 2000

    # 2. Stage 2: Compile WorldVault + Parallel Scenes + Table-Read TTS Timings
    comp_r = client.post(
        f"/api/cinema/projects/{pid}/compile",
        json={"treatment_id": proj["treatments"][0]["treatment_id"], "mock": True, "wait": True},
    )
    assert comp_r.status_code == 200
    compiled = comp_r.json()
    assert compiled["status"] == "script_ready"
    assert len(compiled["scenes"]) >= 3
    assert compiled["shot_count"] >= 9
    assert compiled["story_rubric"]["overall_score"] >= 8.0

    # Verify Table-Read synthesized dialogue WAVs & hydrated JSON exist
    first_shot = compiled["scenes"][0]["shots"][0]
    assert "CHAR_PROTAG" not in str(first_shot["hydrated_prompt_json"])

    # 3. Producer Taste Reaction + 2-Way JSON Inspector Patch
    react_r = client.post(
        f"/api/cinema/projects/{pid}/react",
        json={
            "note_or_chip": "warmer_intimate",
            "target_type": "project",
            "polarity": "adjust",
            "mock": True,
        },
    )
    assert react_r.status_code == 200

    patch_r = client.patch(
        f"/api/cinema/projects/{pid}",
        json={
            "enable_subtitles": True,
            "shot_patch": {
                "shot_id": first_shot["shot_id"],
                "duration_sec": 6,
            },
        },
    )
    assert patch_r.status_code == 200
    assert patch_r.json()["enable_subtitles"] is True

    # 4. Stage 3: Build Vault Turnarounds + Dual-Wall Plates + Start/End Keyframes + Voiced Leica Reel
    anim_r = client.post(
        f"/api/cinema/projects/{pid}/animatic",
        json={"mock": True, "wait": True},
    )
    assert anim_r.status_code == 200
    anim = anim_r.json()
    assert anim["status"] == "animatic_ready"
    assert anim["leica_reel_url"] is not None
    assert anim["vault"]["characters"][0]["frontal_url"] is not None
    assert anim["vault"]["locations"][0]["primary_wall_url"] is not None
    assert anim["vault"]["locations"][0]["reverse_wall_url"] is not None

    # 5. Stage 4: Shoot All Takes + PCM Soundstage Master Mix + HTTP 206 Range Streaming
    prod_r = client.post(
        f"/api/cinema/projects/{pid}/produce",
        json={"mock": True, "wait": True},
    )
    assert prod_r.status_code == 200
    done = prod_r.json()
    assert done["status"] == "done"
    assert done["master_video_url"] is not None
    assert done["qa_report"]["overall_score"] >= 7.5

    range_r = client.get(
        done["master_video_url"],
        headers={"Range": "bytes=0-1023"},
    )
    assert range_r.status_code == 206
    assert len(range_r.content) == 1024


def test_cinema_one_click_autopilot_film(tmp_path: Path) -> None:
    client = _make_client(tmp_path)
    r = client.post(
        "/api/cinema/projects",
        json={
            "idea": "A watchmaker discovers a pocket watch that pauses falling rain for three seconds",
            "target_duration_sec": 35,
            "workflow_mode": "autopilot",
            "mood_presets": ["cinema_natgeo"],
            "mock": True,
            "wait": True,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "done"
    assert body["leica_reel_url"] is not None
    assert body["master_video_url"] is not None
    assert Path(body["master_video_path"]).is_file()


def test_script_gate_diagnostics_and_autofix(tmp_path: Path) -> None:
    from gtm_videogen.models.cinema_compiler import (
        autofix_cinema_project_gate,
        validate_cinema_project,
    )
    from gtm_videogen.models.cinema_project import (
        CinemaProject,
        MotionDeltaBeat,
        SceneWorldState,
    )

    vault = WorldVault(
        characters=[
            CharacterAsset(
                id="CHAR_A",
                name="Alex",
                identity_lock="30yo architect with dark hair",
                wardrobe_lock="black wool trench coat",
            )
        ],
        locations=[
            LocationAsset(
                id="LOC_1",
                name="Train Station",
                cardinal_zones=CardinalZones(
                    A_north_primary="rainy platform and iron arches",
                    B_south_reverse="wooden ticket booth and warm lamps",
                ),
            )
        ],
        props=[
            PropAsset(
                id="PROP_KEY",
                name="Brass Key",
                states={"intact": "clean brass key", "bent": "bent scratched brass key"},
                default_state="intact",
            )
        ],
    )

    # Deliberately trigger every Script Gate diagnostic
    bad_shots = [
        CinemaShot(
            shot_id="SC01_SH01",
            scene_id="SC_01",
            index_in_scene=0,
            duration_sec=3,
            shot_scale="medium",
            characters_in_shot=["CHAR_UNKNOWN"],
            start_frame=FrameStateSpec(
                composition="Medium shot with a bright neon sign reading WELCOME on wall"
            ),
            motion_delta=[MotionDeltaBeat(window="0-3s", action="Camera holds")],
            end_frame=FrameStateSpec(composition="Medium shot holds"),
            dialogue=DialogueCueSpec(
                speaker_id="CHAR_A",
                line="This dialogue line is deliberately far too long to fit inside a tiny three second shot window without overflowing.",
                delivery_mode="on_screen_speaking",
                pre_pause_sec=0.8,
            ),
        ),
        CinemaShot(
            shot_id="SC01_SH02",
            scene_id="SC_01",
            index_in_scene=1,
            duration_sec=5,
            shot_scale="medium",
            start_frame=FrameStateSpec(composition="Medium shot of CHAR_A"),
            end_frame=FrameStateSpec(composition="Medium shot settles"),
        ),
        CinemaShot(
            shot_id="SC01_SH03",
            scene_id="SC_01",
            index_in_scene=2,
            duration_sec=5,
            shot_scale="medium",
            start_frame=FrameStateSpec(composition="Third consecutive medium shot"),
            end_frame=FrameStateSpec(composition="Medium shot settles"),
        ),
    ]
    sc1 = CinemaScene(
        scene_id="SC_01",
        scene_number=1,
        location_id="LOC_1",
        scene_entry_state=SceneWorldState(prop_states={"PROP_KEY": "nonexistent_state"}),
        shots=bad_shots,
    )
    sc2 = CinemaScene(
        scene_id="SC_02",
        scene_number=2,
        location_id="LOC_1",
        causal_link_from_prev="opening_hook",  # invalid on Scene 2!
        shots=[
            CinemaShot(
                shot_id="SC02_SH01",
                scene_id="SC_02",
                index_in_scene=0,
                duration_sec=5,
                shot_scale="wide",
                start_frame=FrameStateSpec(composition="Wide shot of platform"),
                end_frame=FrameStateSpec(composition="Wide shot holds"),
            )
        ],
    )
    proj = CinemaProject(project_id="cin_gate_test", vault=vault, scenes=[sc1, sc2])
    reg = ModelRegistryStore(tmp_path / "stitch")
    spec = reg.get_video_model("fal-ai/kling-video/v3/pro/text-to-video")

    gate_before = validate_cinema_project(proj, spec)
    codes = {i["code"] for i in gate_before["issues"]}
    assert "weak_causality" in codes
    assert "unknown_prop_state" in codes
    assert "monotonous_shot_scale" in codes
    assert "unknown_character_id" in codes
    assert "garbled_text_risk" in codes
    assert "dialogue_overflow" in codes

    # Autofix must heal dialogue duration overflow & break monotonous 3-in-a-row shot scales
    autofix_cinema_project_gate(proj)
    assert proj.scenes[0].shots[0].duration_sec >= 8
    assert proj.scenes[0].shots[2].shot_scale != "medium"
    gate_after = validate_cinema_project(proj, spec)
    after_codes = {i["code"] for i in gate_after["issues"]}
    assert "dialogue_overflow" not in after_codes
    assert "monotonous_shot_scale" not in after_codes


def test_cinema_security_validation_uploads_reshoots_and_wardrobe_ripple(tmp_path: Path) -> None:
    client = _make_client(tmp_path)

    # 1. Create & compile project
    create_r = client.post(
        "/api/cinema/projects",
        json={
            "idea": "A midnight baker discovers a sourdough starter that glows softly before sunrise",
            "target_duration_sec": 30,
            "aspect_ratio": "9:16",
            "workflow_mode": "guided",
            "mood_presets": ["cinema_pastel"],
            "mock": True,
        },
    )
    assert create_r.status_code == 200
    pid = create_r.json()["project_id"]

    comp_r = client.post(
        f"/api/cinema/projects/{pid}/compile",
        json={"mock": True, "wait": True},
    )
    assert comp_r.status_code == 200
    compiled = comp_r.json()
    first_shot_id = compiled["scenes"][0]["shots"][0]["shot_id"]
    second_shot_id = compiled["scenes"][0]["shots"][1]["shot_id"]

    # 2. Negative & security tests: 404, 422 schema error, path traversal, 416 bad Range
    assert client.get("/api/cinema/projects/does_not_exist").status_code == 404
    assert (
        client.patch(
            f"/api/cinema/projects/{pid}",
            json={"shot_patch": {"shot_id": "NO_SUCH_SHOT", "duration_sec": 5}},
        ).status_code
        == 404
    )
    assert (
        client.patch(
            f"/api/cinema/projects/{pid}",
            json={"scenes": [{"scene_id": "BAD", "scene_number": 0}]},
        ).status_code
        == 422
    )
    # Path traversal segments must be rejected
    assert client.get(f"/api/cinema/projects/{pid}/media/..%2F..%2Fetc/passwd").status_code in (
        400,
        403,
        404,
    )
    assert client.get(f"/api/cinema/projects/{pid}/media/vault/nonexistent.jpg").status_code == 404

    # 3. Upload custom reference photo for character + logo
    up_char = client.post(
        f"/api/cinema/projects/{pid}/upload-reference",
        json={
            "data_url": _tiny_jpeg_data_url((180, 70, 55)),
            "target_kind": "character",
            "target_id": compiled["vault"]["characters"][0]["id"],
        },
    )
    assert up_char.status_code == 200
    assert up_char.json()["vault"]["characters"][0]["uploaded_ref_path"] is not None

    # 4. Plain-English Wardrobe Ripple Edit updates WorldVault & propagates to hydrated prompt
    ripple_r = client.post(
        f"/api/cinema/projects/{pid}/react",
        json={
            "note_or_chip": "Give her a crimson linen trench coat",
            "target_type": "project",
            "polarity": "adjust",
            "mock": True,
        },
    )
    assert ripple_r.status_code == 200
    rippled = ripple_r.json()
    assert "crimson linen trench coat" in rippled["vault"]["characters"][0]["wardrobe_lock"]
    assert "crimson linen trench coat" in str(
        rippled["scenes"][0]["shots"][0]["hydrated_prompt_json"]
    )

    # 5. Single-shot keyframe regen (start, end, both) + hybrid engines (insert_ui_plate & title_card)
    kf_r = client.post(
        f"/api/cinema/projects/{pid}/shots/{first_shot_id}/keyframes",
        json={"which": "both", "tweak_note": "warmer flour dust in morning ray", "mock": True},
    )
    assert kf_r.status_code == 200
    sh0 = kf_r.json()["scenes"][0]["shots"][0]
    assert sh0["start_keyframe_media_url"] is not None
    assert sh0["end_keyframe_media_url"] is not None

    # Switch shot 2 to insert_ui_plate and shot 3 to title_card, then shoot individually!
    third_shot_id = compiled["scenes"][0]["shots"][2]["shot_id"]
    for sid, eng in ((second_shot_id, "insert_ui_plate"), (third_shot_id, "title_card")):
        p_eng = client.patch(
            f"/api/cinema/projects/{pid}",
            json={
                "shot_patch": {
                    "shot_id": sid,
                    "render_engine": eng,
                    "title_card_headline": "Sunrise Batch",
                }
            },
        )
        assert p_eng.status_code == 200

    # Shoot take 1 & take 2 for shot 1, plus hybrid UI plate and title card
    t1 = client.post(f"/api/cinema/projects/{pid}/shots/{first_shot_id}/shoot", json={"mock": True})
    assert t1.status_code == 200
    t2 = client.post(f"/api/cinema/projects/{pid}/shots/{first_shot_id}/shoot", json={"mock": True})
    assert t2.status_code == 200
    assert len(t2.json()["scenes"][0]["shots"][0]["takes"]) == 2

    # Switch winning take back to Take 1 via PATCH
    pick_r = client.patch(
        f"/api/cinema/projects/{pid}",
        json={"winning_take_selection": {first_shot_id: 1}},
    )
    assert pick_r.status_code == 200
    assert pick_r.json()["scenes"][0]["shots"][0]["winning_take_id"] == 1

    assert (
        client.post(f"/api/cinema/projects/{pid}/shots/{second_shot_id}/shoot", json={"mock": True}).status_code
        == 200
    )
    assert (
        client.post(f"/api/cinema/projects/{pid}/shots/{third_shot_id}/shoot", json={"mock": True}).status_code
        == 200
    )

    # 6. Remaster 9:16 vertical film from the 3 hybrid takes and verify dimensions & 416 Range
    rem_r = client.post(f"/api/cinema/projects/{pid}/remaster", json={"mock": True})
    assert rem_r.status_code == 200
    rem = rem_r.json()
    assert rem["status"] == "done"
    assert rem["qa_report"]["width"] == 1080
    assert rem["qa_report"]["height"] == 1920
    assert rem["qa_report"]["has_audio"] is True

    # Invalid byte range must return HTTP 416
    bad_range = client.get(rem["master_video_url"], headers={"Range": "bytes=999999999-9999999999"})
    assert bad_range.status_code == 416


def test_title_extraction_and_lenient_llm_shot_normalizer() -> None:
    from gtm_videogen.agents.cinema_writers_room import (
        _extract_subject_noun,
        _normalize_llm_shot_dict,
    )
    from gtm_videogen.models.cinema_project import FrameStateSpec

    idea = (
        "Title: The Brass Bell\n"
        "Genre: Mass Action / Drama\n"
        "Logline: When a ruthless local gang tries to shut down a quiet, rain-drenched tea stall, "
        "they learn the hard way why the old man behind the counter never leaves his town."
    )
    assert _extract_subject_noun(idea) == "The Brass Bell"

    vault = WorldVault(
        characters=[
            CharacterAsset(
                id="CHAR_PROTAG",
                name="Velayudham",
                identity_lock="60yo South Indian tea stall owner",
                wardrobe_lock="faded cotton shirt and lungi",
            )
        ],
        locations=[
            LocationAsset(
                id="LOC_1",
                name="Velayudham Tea Stall",
                cardinal_zones=CardinalZones(
                    A_north_primary="steaming copper samovar and wooden counter",
                    B_south_reverse="rain-drenched street and flickering streetlights",
                ),
            )
        ],
        props=[
            PropAsset(
                id="PROP_HERO",
                name="Brass Bell",
                states={"still": "hanging silent", "ringing": "vibrating sharply"},
            )
        ],
    )
    fb_shot = CinemaShot(
        shot_id="SC_01_SH01",
        scene_id="SC_01",
        location_id="LOC_1",
        duration_sec=6,
        start_frame=FrameStateSpec(composition="fallback start frame"),
        end_frame=FrameStateSpec(composition="fallback end frame"),
    )
    scene = CinemaScene(
        scene_id="SC_01",
        scene_number=1,
        title="Monsoon Tea",
        slugline="INT. TEA STALL - NIGHT",
        location_id="LOC_1",
        shots=[fb_shot],
    )

    raw_llm_shot = {
        "shot_id": "SC_01_SH01",
        "shot_scale": "MCU",
        "duration_sec": "7",
        "camera": {
            "lens_mm": "65mm",
            "facing_wall_zone": "A",
            "movement": "slow push-in through monsoon rain",
        },
        "start_frame": "Velayudham wipes a brass tumbler behind the steaming tea stall counter.",
        "end_frame": "Velayudham stops and looks up as headlights cut across the wet street.",
        "motion_delta": [
            "0-2s: Steam curls over the copper samovar",
            "2-5s: He sets the tumbler down calmly",
        ],
        "dialogue": {
            "speaker_id": "CHAR_PROTAG",
            "line": "Tea is fresh. Leave the machetes outside.",
            "delivery_mode": "close_up",
        },
        "active_entities": ["CHAR_PROTAG", "PROP_HERO"],
    }
    shot = _normalize_llm_shot_dict(raw_llm_shot, fb_shot, scene, vault, 1)
    assert shot.shot_scale == "medium_close_up"
    assert shot.camera.lens_mm == 65
    assert shot.camera.facing_wall_zone == "A_north_primary"
    assert shot.duration_sec == 7
    assert shot.dialogue is not None
    assert shot.dialogue.delivery_mode == "on_screen_speaking"
    assert "Tea is fresh" in shot.dialogue.line
    assert len(shot.motion_delta) == 2


def test_quick_storyboard_one_click_guided(tmp_path: Path) -> None:
    client = _make_client(tmp_path)
    r = client.post(
        "/api/cinema/projects",
        json={
            "idea": "Title: The Brass Bell\nA quiet rain-drenched South Indian tea stall owner confronts a local gang.",
            "target_duration_sec": 35,
            "workflow_mode": "guided",
            "quick_storyboard": True,
            "mood_presets": ["cinema_noir"],
            "mock": True,
            "wait": True,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "animatic_ready"
    assert "The Brass Bell" in body["title"]
    assert body["leica_reel_url"] is not None
    assert len(body["scenes"]) >= 3
    assert body["vault"]["characters"][0]["frontal_url"] is not None
    assert body["scenes"][0]["shots"][0]["start_keyframe_media_url"] is not None


@pytest.mark.asyncio
async def test_partial_render_resilience_preserves_completed_shots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gtm_videogen.agents import cinema_producer
    from gtm_videogen.agents.cinema_writers_room import (
        compile_cinema_master_script,
        pitch_cinema_treatments,
    )
    from gtm_videogen.core.config import Settings

    settings = Settings(data_dir=tmp_path / "data")
    pdir = tmp_path / "proj"
    pdir.mkdir(parents=True, exist_ok=True)

    project = await pitch_cinema_treatments(
        idea="Title: Rain Tea\nA tea stall owner faces down local thugs in a monsoon.",
        target_duration_sec=24,
        project_id="cin_partial_test",
        media_dir=pdir,
        settings=settings,
        mock=True,
    )
    await compile_cinema_master_script(project, pdir, settings=settings, mock=True)
    # Trim to 1 scene (3 shots) so the partial-render E2E test runs in seconds
    project.scenes = project.scenes[:1]
    await cinema_producer.generate_vault_references(project, pdir, settings=settings, mock=True)
    await cinema_producer.build_leica_reel_animatic(project, pdir, settings=settings, mock=True)

    orig_render = cinema_producer.render_cinema_shot_take
    call_count = {"n": 0}

    async def _flaky_render(proj, shot_id, project_dir, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("HTTP 403 Exhausted balance")
        return await orig_render(proj, shot_id, project_dir, **kwargs)

    monkeypatch.setattr(cinema_producer, "render_cinema_shot_take", _flaky_render)

    await cinema_producer.shoot_and_mix_cinema_film(project, pdir, settings=settings, mock=True)

    assert project.status == "done"
    assert project.master_video_path is not None
    assert Path(project.master_video_path).is_file()
    assert "1 shot(s) skipped" in project.status_message


