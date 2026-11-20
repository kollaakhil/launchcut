"""Tier 2 E2E Tests: Boundary, corner, negative/error-rejection, and BUG-01..BUG-28 regressions."""

from __future__ import annotations

import asyncio
import json
import math
import time
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from gtm_videogen.core.cost_meter import (
    BudgetExceeded,
    CostMeter,
    estimate_run_cost,
    read_ledger_total,
)
from gtm_videogen.core.http_client import ResilientHttpClient
from gtm_videogen.core.job_registry import CancellationToken, JobRegistry
from gtm_videogen.core.media_store import MediaStore, MediaStoreError
from gtm_videogen.core.run_store import RunStore
from gtm_videogen.core.ssrf import SSRFError, assert_public_url, validate_redirect_chain
from gtm_videogen.models.schemas import (
    CreativeFingerprint,
    CreativeHistory,
    MusicCue,
    RenderedAudio,
    RenderedClip,
    RunInput,
    SfxCue,
    Shot,
    SoundPlan,
    VoiceCasting,
    VoiceCue,
)
from gtm_videogen.rules.claim_grounding import check_claim_grounding
from gtm_videogen.rules.script_gate import evaluate_script_gate
from gtm_videogen.rules.timing import fit_timings
from gtm_videogen.rules.voice_casting import select_voice

from .conftest import (
    ffprobe_inspect,
    make_creative_bible,
    make_research_bundle,
    make_run_input,
    make_run_state,
    make_shot_sheet,
    make_sound_plan,
    make_synthetic_clip,
    make_synthetic_wav,
    parse_frame_rate,
)

pytestmark = [pytest.mark.e2e, pytest.mark.tier2]


# ---------------------------------------------------------------------------
# M1 Boundary & Bug Regression Tests (BUG-01, 02, 03, 05, 11, 12, 14, 15,
# 17, 18, 19, 21, 22-core, 23, 25, 26, SSRF)
# ---------------------------------------------------------------------------


@pytest.mark.m1
async def test_bug01_job_registry_slot_recovery_no_deadlock() -> None:
    """BUG-01: Concurrency slots recover after success, failure, and cancellation (0 deadlock)."""
    registry = JobRegistry(max_concurrent=2)
    started = asyncio.Event()

    async def _ok(tok: CancellationToken) -> None:
        await asyncio.sleep(0.01)

    async def _fail(tok: CancellationToken) -> None:
        await asyncio.sleep(0.01)
        raise RuntimeError("boom")

    async def _hang(tok: CancellationToken) -> None:
        started.set()
        await tok.wait()
        tok.raise_if_cancelled()

    j_ok = await registry.submit("r1", _ok)
    j_fail = await registry.submit("r2", _fail)
    await registry.wait_for_job(j_ok.id)
    await registry.wait_for_job(j_fail.id)
    assert registry.active_count == 0

    j_cancel = await registry.submit("r3", _hang)
    await asyncio.wait_for(started.wait(), timeout=2.0)
    await registry.cancel_job(j_cancel.id, reason="slot-recovery-test")
    res_cancel = await registry.wait_for_job(j_cancel.id)
    assert res_cancel is not None and res_cancel.state == "cancelled"
    assert registry.active_count == 0

    # Subsequent jobs must run without deadlock
    j_after = await registry.submit("r4", _ok)
    res_after = await registry.wait_for_job(j_after.id)
    assert res_after is not None and res_after.state == "done"
    assert registry.active_count == 0


@pytest.mark.m1
async def test_bug02_job_status_dto_strictly_serializable_while_running() -> None:
    """BUG-02: JobStatusDTO.model_dump_json() succeeds while jobs are actively executing."""
    registry = JobRegistry(max_concurrent=2)
    release = asyncio.Event()
    entered = asyncio.Event()

    async def _running_worker(tok: CancellationToken) -> None:
        entered.set()
        await release.wait()

    job = await registry.submit("run_live", _running_worker, stage="render", shot_id="s1")
    await asyncio.wait_for(entered.wait(), timeout=2.0)

    live_jobs = registry.list_jobs()
    assert len(live_jobs) == 1
    dumped = live_jobs[0].model_dump_json()
    parsed = json.loads(dumped)
    assert parsed["id"] == job.id
    assert parsed["state"] == "running"

    release.set()
    await registry.wait_for_job(job.id)


@pytest.mark.m1
def test_bug03_run_input_validation_boundaries() -> None:
    """BUG-03: RunInput rejects extra fields, accepts audio_mode='both', rejects bad bounds/URLs."""
    valid = make_run_input(audio_mode="both")
    assert valid.audio_mode == "both"

    with pytest.raises(ValidationError):
        RunInput(**{**valid.model_dump(), "patternId": "confession_loop"})  # type: ignore[arg-type]

    with pytest.raises(ValidationError):
        make_run_input(target_duration_sec=2.0)

    with pytest.raises(ValidationError):
        make_run_input(target_duration_sec=120.0)

    with pytest.raises(ValidationError):
        make_run_input(cap_usd=-1.0)

    with pytest.raises(ValidationError):
        make_run_input(brand_url="https://")

    with pytest.raises(ValidationError):
        make_run_input(brand_url="ftp://example.com")


@pytest.mark.m1
async def test_bug05_four_layer_cancellation_never_overwritten_and_rerun_succeeds() -> None:
    """BUG-05: Cancelled job stays 'cancelled' (never 'done') and subsequent re-run succeeds."""
    registry = JobRegistry(max_concurrent=2)
    entered = asyncio.Event()
    callback_fired = {"value": False}

    async def _cancellable_run(tok: CancellationToken) -> None:
        tok.register_callback(lambda: callback_fired.__setitem__("value", True))
        entered.set()
        await tok.wait()
        # Even if worker finishes normally after cancel, registry must preserve "cancelled"

    job = await registry.submit("run_c5", _cancellable_run, stage="render")
    await asyncio.wait_for(entered.wait(), timeout=2.0)
    cancelled_count = await registry.cancel_run("run_c5", reason="user clicked stop")
    assert cancelled_count >= 1
    res = await registry.wait_for_job(job.id)
    assert res is not None
    assert res.state == "cancelled"
    assert callback_fired["value"] is True
    assert not registry.is_run_active("run_c5")

    # Re-run of same run_id must succeed cleanly
    async def _rerun(tok: CancellationToken) -> None:
        tok.raise_if_cancelled()

    job2 = await registry.submit("run_c5", _rerun, stage="render")
    res2 = await registry.wait_for_job(job2.id)
    assert res2 is not None and res2.state == "done"


@pytest.mark.m1
async def test_bug11_media_store_rejects_malformed_data_url_and_private_ip(tmp_path: Path) -> None:
    """BUG-11: MediaStore rejects malformed data: URLs & private-IP https:// URLs cleanly."""
    store = MediaStore(tmp_path / "media", resolve_dns_for_ssrf=False)

    with pytest.raises(MediaStoreError):
        await store.resolve_to_path("data:image/png;base64")  # missing comma

    with pytest.raises(MediaStoreError):
        await store.resolve_to_path("data:image/png;base64,!!!not-valid-base64!!!")

    with pytest.raises(SSRFError):
        await store.resolve_to_path("https://127.0.0.1/secret.png")


@pytest.mark.m1
def test_bug12_bug15_bug25_cost_meter_cap_preflight_floor_and_missing_ledger(
    tmp_path: Path,
) -> None:
    """BUG-12, BUG-15, BUG-25: Cap enforcement, min_seconds+audio preflight, missing ledger=0.0."""
    # BUG-25: Missing ledger file returns 0.0 cleanly
    assert read_ledger_total(tmp_path / "does_not_exist.jsonl") == 0.0

    # BUG-12: Exceeding cap raises BudgetExceeded
    meter = CostMeter(run_id="r_cap", cap_usd=1.0)
    r1 = meter.reserve("shot_1", 0.60, stage="render", provider="fal", model="kling")
    meter.settle(r1, 0.60)
    with pytest.raises(BudgetExceeded):
        meter.reserve("shot_2", 0.50, stage="render", provider="fal", model="kling")

    # BUG-15: Preflight estimate includes video_min_seconds floor AND audio (VO/SFX/music)
    short_sheet = make_shot_sheet(
        target_duration_sec=4.0,
        shots=[
            Shot(id="s1", index=0, beat="hook", duration_sec=2.0, visual_prompt="Bean close-up"),
            Shot(id="s2", index=1, beat="cta", duration_sec=2.0, visual_prompt="Bag close-up"),
        ],
    )
    sound_plan = make_sound_plan()
    est = estimate_run_cost(
        shot_sheet=short_sheet,
        sound_plan=sound_plan,
        video_usd_per_sec=0.10,
        video_min_seconds=5.0,
    )
    # 2 video shots x max(2.0, 5.0)s x $0.10 = $1.00
    assert est["video_usd"] == pytest.approx(1.0, abs=0.01)
    assert est["voice_usd"] > 0.0
    assert est["sfx_usd"] > 0.0
    assert est["music_usd"] > 0.0


@pytest.mark.m1
async def test_bug14_resilient_http_raises_on_404_500_and_honors_cancel() -> None:
    """BUG-14: ResilientHttpClient raises on HTTP 404/500 before JSON parse & aborts on cancel."""

    def _handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/missing":
            return httpx.Response(404, text="<html>not json</html>")
        if request.url.path == "/error":
            return httpx.Response(500, text="internal failure")
        return httpx.Response(200, json={"ok": True})

    client = ResilientHttpClient(
        max_retries=1,
        backoff_base_sec=0.01,
        backoff_max_sec=0.02,
        transport=httpx.MockTransport(_handler),
        resolve_dns_for_ssrf=False,
    )
    try:
        with pytest.raises(httpx.HTTPStatusError):
            await client.get_json("https://93.184.216.34/missing")

        with pytest.raises(httpx.HTTPStatusError):
            await client.get_json("https://93.184.216.34/error")

        tok = CancellationToken()
        tok.cancel("aborted before request")
        with pytest.raises(asyncio.CancelledError):
            await client.get_json("https://93.184.216.34/ok", cancel=tok)
    finally:
        await client.aclose()


@pytest.mark.m1
def test_bug17_select_voice_matches_attributes_and_avoids_history_repeat() -> None:
    """BUG-17: select_voice matches gender/age/accent/energy/pace and avoids history repeats."""
    casting = VoiceCasting(
        gender="male",
        age="mature",
        accent="uk",
        energy="authoritative",
        pace="measured",
    )
    first = select_voice(casting)
    assert first.gender == "male"
    assert first.age == "mature"
    assert first.accent == "uk"
    assert first.energy == "authoritative"
    assert first.pace == "measured"

    history = CreativeHistory(
        brand_url="https://kestrelroasters.example.com",
        entries=[
            CreativeFingerprint(
                run_id="prev",
                brand_url="https://kestrelroasters.example.com",
                arc_name="A",
                hook_opening="H",
                visual_world="W",
                lighting_style="L",
                camera_style="C",
                voice_id=first.voice_id,
            )
        ],
    )
    second = select_voice(casting, brand_history=history)
    assert second.voice_id != first.voice_id


@pytest.mark.m1
def test_bug18_script_gate_blocks_product_hero_without_product_photos() -> None:
    """BUG-18: evaluate_script_gate blocks product_hero=True when product photos are absent."""
    research = make_research_bundle(product_photos=[])
    sheet = make_shot_sheet()
    assert any(s.product_hero for s in sheet.shots)

    report = evaluate_script_gate(sheet, research, has_product_photos=False)
    assert report.passed is False
    assert any(
        "photo" in err.code.lower() or "product" in err.message.lower()
        for err in report.blocking_errors
    )


@pytest.mark.m1
def test_bug19_run_store_heals_orphaned_working_run_to_interrupted(tmp_path: Path) -> None:
    """BUG-19: Orphaned 'working' run absent from JobRegistry heals to 'interrupted' + warning."""
    registry = JobRegistry(max_concurrent=2)
    store = RunStore(tmp_path / "runs", job_registry=registry)

    orphan = make_run_state(run_id="run_orphan_19", status="working")
    store.save(orphan)

    healed = store.load("run_orphan_19")
    assert healed is not None
    assert healed.status == "interrupted"
    assert len(healed.warnings) >= 1


@pytest.mark.m1
def test_bug21_fit_timings_never_nan_handles_impossible_and_enforces_bounds() -> None:
    """BUG-21: fit_timings never returns NaN/Inf, handles impossible targets, honors [min, max]."""
    # Normal fit within [1.5, 8.0]
    normal = fit_timings([1.0, 20.0, 3.0], target_sec=12.0, min_shot=1.5, max_shot=8.0)
    assert all(math.isfinite(x) for x in normal)
    assert all(1.5 - 1e-6 <= x <= 8.0 + 1e-6 for x in normal)
    assert sum(normal) == pytest.approx(12.0, abs=0.05)

    # Impossible target: 10 shots * 1.5s = 15s > target 6s (must not NaN/Inf or crash)
    impossible = fit_timings([2.0] * 10, target_sec=6.0, min_shot=1.5, max_shot=8.0)
    assert len(impossible) == 10
    assert all(math.isfinite(x) and x > 0.0 for x in impossible)


@pytest.mark.m1
async def test_bug22_job_registry_other_active_jobs_count_accuracy() -> None:
    """BUG-22 (core): other_active_jobs_count(exclude_run_id=...) stays accurate under filtering."""
    registry = JobRegistry(max_concurrent=4)
    started_a = asyncio.Event()
    started_b = asyncio.Event()
    release = asyncio.Event()

    async def _hold(ev: asyncio.Event, tok: CancellationToken) -> None:
        ev.set()
        await release.wait()

    ja = await registry.submit("run_A", lambda tok: _hold(started_a, tok))
    jb = await registry.submit("run_B", lambda tok: _hold(started_b, tok))
    await asyncio.wait_for(asyncio.gather(started_a.wait(), started_b.wait()), timeout=2.0)

    assert len(registry.list_jobs(run_id="run_A")) == 1
    assert registry.other_active_jobs_count(exclude_run_id="run_A") == 1
    assert registry.other_active_jobs_count(exclude_run_id="run_C") == 2

    release.set()
    await registry.wait_for_job(ja.id)
    await registry.wait_for_job(jb.id)
    assert registry.other_active_jobs_count(exclude_run_id="run_A") == 0


@pytest.mark.m1
def test_bug23_claim_grounding_and_placeholders_non_overridable() -> None:
    """BUG-23: Unverified claims, superlatives, banned phrases & placeholders stay blocking."""
    research = make_research_bundle()

    bad_text = (
        "The #1 best coffee on earth! Guaranteed miracle for [BRAND] at {{PRICE}} "
        "with 99% caffeine boost."
    )
    issues = check_claim_grounding(bad_text, research)
    assert len(issues) >= 3
    assert all(issue.severity == "blocking" and issue.overridable is False for issue in issues)

    bad_sheet = make_shot_sheet(
        shots=[
            Shot(
                id="s1",
                index=0,
                beat="hook",
                duration_sec=4.0,
                visual_prompt="Coffee cup",
                vo_line="The #1 best guaranteed miracle for [BRAND]!",
            ),
            Shot(
                id="s2",
                index=1,
                beat="cta",
                duration_sec=4.0,
                visual_prompt="Coffee bag",
                vo_line="Order now at {{PRICE}}.",
            ),
        ],
        target_duration_sec=8.0,
    )
    report = evaluate_script_gate(
        bad_sheet,
        research,
        has_product_photos=True,
        gate_override=True,
    )
    assert report.passed is False
    assert len(report.blocking_errors) >= 1


@pytest.mark.m1
def test_bug26_atomic_write_cleanup_leaves_zero_temp_files_on_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """BUG-26: Atomic write failure in MediaStore / RunStore cleans up temp files cleanly."""
    media_dir = tmp_path / "media"
    store = MediaStore(media_dir, resolve_dns_for_ssrf=False)

    def _fail_move(*args: object, **kwargs: object) -> None:
        raise OSError("simulated cross-device failure")

    monkeypatch.setattr("shutil.move", _fail_move)
    with pytest.raises(OSError):
        store.save_bytes(b"payload", ext=".bin")

    leaked = list(media_dir.rglob("*.tmp"))
    assert leaked == []


@pytest.mark.m1
@pytest.mark.parametrize(
    "bad_url",
    [
        "http://127.0.0.1/admin",
        "http://169.254.169.254/latest/meta-data/",
        "http://10.0.0.1/internal",
        "http://[::1]/loopback",
        "http://localhost:8080/api",
        "https://93.184.216.34:8443/nonstandard",
    ],
)
def test_ssrf_blocks_private_ips_localhost_and_non_standard_ports(bad_url: str) -> None:
    """SSRF boundary: blocks private IPs, loopback, metadata IP, localhost, and non-80/443 ports."""
    with pytest.raises(SSRFError):
        assert_public_url(bad_url, resolve_dns=False)


@pytest.mark.m1
def test_ssrf_blocks_excessive_redirect_hops() -> None:
    """SSRF boundary: blocks redirect chains exceeding 3 hops."""
    chain = [
        "https://93.184.216.34/a",
        "https://93.184.216.34/b",
        "https://93.184.216.34/c",
        "https://93.184.216.34/d",
        "https://93.184.216.34/e",
    ]
    with pytest.raises(SSRFError):
        validate_redirect_chain(chain, max_hops=3, resolve_dns=False)


# ---------------------------------------------------------------------------
# M2 - M5 Boundary & Bug Regression Tests (imported inside test bodies)
# ---------------------------------------------------------------------------


async def test_bug04_audio_payloads_and_sfx_fallback_chain(tmp_path: Path) -> None:
    """BUG-04: Per-model audio payload schemas + SFX fallback chain (mirelo->elevenlabs->sonilo)."""
    from gtm_videogen.core.config import Settings
    from gtm_videogen.providers.audio import (
        AudioProvider,
        build_music_payload,
        build_tts_payload,
    )

    # Part A: Verify per-model typed payload schemas
    music_cue = MusicCue(prompt="Warm upright bass", tags=["jazz", "warm"], duration_sec=10.0)
    el_music = build_music_payload("elevenlabs/music", music_cue)
    assert el_music["music_length_ms"] == 10000
    acestep_music = build_music_payload("ace-step/music", music_cue)
    assert "tags" in acestep_music and "duration" in acestep_music and "lyrics" in acestep_music

    chatterbox_tts = build_tts_payload("chatterbox/tts", "Save 25% today", "v_aria")
    assert "stability" not in chatterbox_tts
    assert "voice" not in chatterbox_tts
    assert "percent" in chatterbox_tts["text"].lower()

    # Part B: Live AudioProvider calls verifying outbound payloads + SFX fallback chain
    wav_bytes = make_synthetic_wav(tmp_path / "sample.wav", duration_sec=1.0).read_bytes()
    post_log: list[tuple[str, dict]] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        url_str = str(request.url)
        if request.method == "GET" and url_str.endswith(".wav"):
            return httpx.Response(200, content=wav_bytes, headers={"content-type": "audio/wav"})
        body = json.loads(request.content.decode("utf-8")) if request.content else {}
        post_log.append((url_str, body))
        if "mirelo" in url_str.lower():
            return httpx.Response(503, text="primary sfx unavailable")
        return httpx.Response(200, json={"audio": {"url": "https://93.184.216.34/out.wav"}})

    async def _no_sleep(_sec: float) -> None:
        return None

    client = ResilientHttpClient(
        max_retries=0,
        transport=httpx.MockTransport(_handler),
        resolve_dns_for_ssrf=False,
    )
    media_store = MediaStore(tmp_path / "media", resolve_dns_for_ssrf=False)
    settings = Settings(fal_key="test-fal-key")
    try:
        provider = AudioProvider(
            settings=settings,
            http_client=client,
            media_store=media_store,
            sleep_fn=_no_sleep,
        )
        # Voice with Chatterbox model sends no ElevenLabs-only fields
        vo_res = await provider.generate_voice(
            VoiceCue(shot_id="s1", at_sec=0.0, duration_sec=1.5, text="Save 25%"),
            voice_id="v_aria",
            model="chatterbox/tts",
        )
        assert vo_res.kind == "vo" and Path(vo_res.path).exists()
        assert "stability" not in post_log[-1][1]

        # Music with ElevenLabs model sends music_length_ms
        music_res = await provider.generate_music(music_cue, model="elevenlabs/music")
        assert music_res.kind == "music" and Path(music_res.path).exists()
        assert post_log[-1][1]["music_length_ms"] == 10000

        # SFX fallback: primary mirelo returns 503 -> secondary elevenlabs succeeds
        sfx_start_idx = len(post_log)
        sfx_cue = SfxCue(shot_id="s1", at_sec=0.0, duration_sec=1.5, prompt="Coffee beans pouring")
        sfx_res = await provider.generate_sfx(sfx_cue)
        sfx_posts = [url for url, _ in post_log[sfx_start_idx:]]
        assert sfx_res.kind == "sfx"
        assert Path(sfx_res.path).exists()
        assert len(sfx_posts) >= 2
        assert "mirelo" in sfx_posts[0].lower()
        assert "elevenlabs" in sfx_posts[1].lower()
        probe = ffprobe_inspect(Path(sfx_res.path))
        assert any(s.get("codec_type") == "audio" for s in probe["streams"])
    finally:
        await client.aclose()


async def test_bug06_async_repair_persists_repair_result(tmp_path: Path) -> None:
    """BUG-06: POST /api/runs/{id}/repair executes asynchronously & persists repair_result."""
    from gtm_videogen.api.app import create_app

    app = create_app(data_dir=tmp_path)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        created = await client.post("/api/runs", json=make_run_input(mock=True).model_dump())
        assert created.status_code in (200, 201)
        run_id = created.json()["run_id"]

        start_res = await client.post(f"/api/runs/{run_id}/start")
        assert start_res.status_code in (200, 202)

        # Poll until initial run completes
        start_state = None
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            r = await client.get(f"/api/runs/{run_id}")
            body = r.json()
            if body["status"] in ("done", "failed", "cancelled"):
                start_state = body
                break
            await asyncio.sleep(0.1)
        assert start_state is not None and start_state["status"] == "done", (
            f"Initial run did not complete: {start_state}"
        )

        repair_res = await client.post(
            f"/api/runs/{run_id}/repair", json={"failed_shot_ids": ["s1"]}
        )
        assert repair_res.status_code in (200, 202)

        final_state = None
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            r = await client.get(f"/api/runs/{run_id}")
            body = r.json()
            if body.get("repair_result") is not None:
                final_state = body
                break
            await asyncio.sleep(0.1)
        assert final_state is not None
        assert final_state["repair_result"] is not None


async def test_bug07_video_provider_honors_aspect_16_9(tmp_path: Path) -> None:
    """BUG-07: VideoProvider honors requested aspect ('16:9' and '9:16') in payload & dimensions."""
    from gtm_videogen.core.config import Settings
    from gtm_videogen.providers.mocks import MockProviderBundle
    from gtm_videogen.providers.video import VideoProvider

    # Part A: Real VideoProvider outbound HTTP payload + downloaded clip dimensions
    mp4_16_9 = make_synthetic_clip(
        tmp_path / "src_16_9.mp4", width=640, height=360, fps=24, duration_sec=2.0
    ).read_bytes()
    mp4_9_16 = make_synthetic_clip(
        tmp_path / "src_9_16.mp4", width=360, height=640, fps=24, duration_sec=2.0
    ).read_bytes()
    sent_payloads: list[dict] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        url_str = str(request.url)
        if request.method == "GET" and url_str.endswith("16_9.mp4"):
            return httpx.Response(200, content=mp4_16_9, headers={"content-type": "video/mp4"})
        if request.method == "GET" and url_str.endswith("9_16.mp4"):
            return httpx.Response(200, content=mp4_9_16, headers={"content-type": "video/mp4"})
        payload = json.loads(request.content.decode("utf-8")) if request.content else {}
        sent_payloads.append(payload)
        suffix = "16_9.mp4" if payload.get("aspect_ratio") == "16:9" else "9_16.mp4"
        return httpx.Response(200, json={"video": {"url": f"https://93.184.216.34/{suffix}"}})

    async def _no_sleep(_sec: float) -> None:
        return None

    client = ResilientHttpClient(
        max_retries=0,
        transport=httpx.MockTransport(_handler),
        resolve_dns_for_ssrf=False,
    )
    media_store = MediaStore(tmp_path / "media", resolve_dns_for_ssrf=False)
    try:
        provider = VideoProvider(
            settings=Settings(fal_key="test-fal-key"),
            http_client=client,
            media_store=media_store,
            sleep_fn=_no_sleep,
        )
        shot = make_shot_sheet(aspect="16:9").shots[0]
        clip_16_9 = await provider.generate_clip(shot, aspect="16:9", fps=24)
        clip_9_16 = await provider.generate_clip(shot, aspect="9:16", fps=24)

        assert sent_payloads[0]["aspect_ratio"] == "16:9"
        assert sent_payloads[1]["aspect_ratio"] == "9:16"
        assert clip_16_9.width > clip_16_9.height
        assert clip_9_16.height > clip_9_16.width
    finally:
        await client.aclose()

    # Part B: MockProviderBundle also honors both 16:9 and 9:16 in rendered MP4s
    bundle = MockProviderBundle.create(work_dir=tmp_path / "mock")
    for aspect in ("16:9", "9:16"):
        m_clip = await bundle.video.generate_clip(shot, aspect=aspect, fps=24)
        probe = ffprobe_inspect(Path(m_clip.path))
        v_stream = next(s for s in probe["streams"] if s.get("codec_type") == "video")
        w, h = int(v_stream["width"]), int(v_stream["height"])
        if aspect == "16:9":
            assert w > h
        else:
            assert h > w


async def test_bug08_graphics_card_respects_exact_dimensions_fps_and_duration(
    tmp_path: Path,
) -> None:
    """BUG-08: GraphicsCardRenderer honors exact 1080x1920 @ 30fps, 2.5s."""
    from gtm_videogen.media.graphics_card import GraphicsCardRenderer

    bible = make_creative_bible()
    renderer = GraphicsCardRenderer()
    out = tmp_path / "portrait_card.mp4"
    clip = await renderer.render_card(
        text="Kestrel Roasters",
        width=1080,
        height=1920,
        fps=30,
        duration_sec=2.5,
        bible=bible,
        output_path=out,
    )
    probe = ffprobe_inspect(Path(clip.path))
    v_stream = next(s for s in probe["streams"] if s.get("codec_type") == "video")
    assert int(v_stream["width"]) == 1080
    assert int(v_stream["height"]) == 1920
    assert parse_frame_rate(v_stream["r_frame_rate"]) == pytest.approx(30.0, abs=0.5)
    assert float(probe["format"]["duration"]) == pytest.approx(2.5, abs=0.35)


async def test_bug09_ffmpeg_stitcher_preserves_30fps_no_judder(tmp_path: Path) -> None:
    """BUG-09: FFmpegStitcher preserves sheet.fps=30 without 24fps judder."""
    from gtm_videogen.media.ffmpeg_stitcher import FFmpegStitcher

    c1 = make_synthetic_clip(tmp_path / "c1.mp4", width=640, height=360, fps=30, duration_sec=2.0)
    c2 = make_synthetic_clip(tmp_path / "c2.mp4", width=640, height=360, fps=30, duration_sec=2.0)
    sheet = make_shot_sheet(
        fps=30,
        target_duration_sec=4.0,
        shots=[
            Shot(id="s1", index=0, beat="hook", duration_sec=2.0, visual_prompt="Shot 1"),
            Shot(id="s2", index=1, beat="cta", duration_sec=2.0, visual_prompt="Shot 2"),
        ],
    )
    clips = [
        RenderedClip(shot_id="s1", path=str(c1), duration_sec=2.0, fps=30, width=640, height=360),
        RenderedClip(shot_id="s2", path=str(c2), duration_sec=2.0, fps=30, width=640, height=360),
    ]
    stitcher = FFmpegStitcher()
    manifest = await stitcher.stitch(
        clips=clips,
        sound_plan=None,
        audio_assets=[],
        sheet=sheet,
        bible=make_creative_bible(),
        output_path=tmp_path / "out_30fps.mp4",
    )
    assert manifest.stitched_path is not None
    probe = ffprobe_inspect(Path(manifest.stitched_path))
    v_stream = next(s for s in probe["streams"] if s.get("codec_type") == "video")
    assert parse_frame_rate(v_stream["r_frame_rate"]) == pytest.approx(30.0, abs=0.5)


async def test_bug10_ffmpeg_stitcher_handles_none_and_empty_audio_cues(tmp_path: Path) -> None:
    """BUG-10: FFmpegStitcher succeeds with sound_plan=None and empty/filtered audio cues."""
    from gtm_videogen.media.ffmpeg_stitcher import FFmpegStitcher

    c1 = make_synthetic_clip(tmp_path / "c1.mp4", width=640, height=360, fps=24, duration_sec=2.0)
    sheet = make_shot_sheet(
        fps=24,
        target_duration_sec=2.0,
        shots=[Shot(id="s1", index=0, beat="hook", duration_sec=2.0, visual_prompt="Shot 1")],
    )
    clips = [
        RenderedClip(shot_id="s1", path=str(c1), duration_sec=2.0, fps=24, width=640, height=360),
    ]
    stitcher = FFmpegStitcher()

    # Case 1: sound_plan is None
    m1 = await stitcher.stitch(
        clips=clips,
        sound_plan=None,
        audio_assets=[],
        sheet=sheet,
        bible=make_creative_bible(),
        output_path=tmp_path / "no_audio_none.mp4",
    )
    assert m1.stitched_path is not None and Path(m1.stitched_path).exists()

    # Case 2: sound_plan has REAL non-empty VO & SFX cues that fall inside silence_gaps
    vo_wav = make_synthetic_wav(tmp_path / "filtered_vo.wav", duration_sec=1.0, freq_hz=440)
    sfx_wav = make_synthetic_wav(tmp_path / "filtered_sfx.wav", duration_sec=0.8, freq_hz=880)
    silent_plan = SoundPlan(
        vo_cues=[VoiceCue(shot_id="s1", at_sec=0.2, duration_sec=1.0, text="Filtered VO")],
        sfx_cues=[SfxCue(shot_id="s1", at_sec=0.5, duration_sec=0.8, prompt="Filtered SFX")],
        music_cue=None,
        silence_gaps=[(0.0, 2.0)],
    )
    filtered_assets = [
        RenderedAudio(kind="vo", cue_id="s1", path=str(vo_wav), duration_sec=1.0),
        RenderedAudio(kind="sfx", cue_id="s1", path=str(sfx_wav), duration_sec=0.8),
    ]
    m2 = await stitcher.stitch(
        clips=clips,
        sound_plan=silent_plan,
        audio_assets=filtered_assets,
        sheet=sheet,
        bible=make_creative_bible(),
        output_path=tmp_path / "no_audio_filtered.mp4",
    )
    assert m2.stitched_path is not None and Path(m2.stitched_path).exists()
    probe2 = ffprobe_inspect(Path(m2.stitched_path))
    assert any(s.get("codec_type") == "video" for s in probe2["streams"])


async def test_bug13_fastapi_ownership_rejects_non_owner_403(tmp_path: Path) -> None:
    """BUG-13: FastAPI mutating /api/runs/{id}/* rejects non-owner actor with HTTP 403."""
    from gtm_videogen.api.app import create_app

    app = create_app(data_dir=tmp_path)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        created = await client.post(
            "/api/runs",
            json=make_run_input(mock=True).model_dump(),
            headers={"X-Actor-Id": "alice"},
        )
        assert created.status_code in (200, 201)
        run_id = created.json()["run_id"]

        forbidden = await client.post(
            f"/api/runs/{run_id}/cancel",
            headers={"X-Actor-Id": "bob"},
        )
        assert forbidden.status_code == 403


async def test_bug16_non_fatal_warnings_persist_when_error_is_none(tmp_path: Path) -> None:
    """BUG-16: Non-fatal SFX errors record warnings that persist when run succeeds (error=None)."""
    from gtm_videogen.agents.pipeline import PipelineRunner
    from gtm_videogen.providers.audio import AudioProviderError
    from gtm_videogen.providers.mocks import MockProviderBundle

    bundle = MockProviderBundle.create(work_dir=tmp_path)

    async def _failing_sfx(cue: SfxCue, cancel: object = None, model: object = None) -> object:
        raise AudioProviderError("simulated non-fatal SFX outage")

    bundle.audio.generate_sfx = _failing_sfx  # type: ignore[method-assign]
    runner = PipelineRunner(providers=bundle, work_dir=tmp_path)
    run_state = make_run_state(
        run_id="r_bug16_warn",
        run_input=make_run_input(audio_mode="both", mock=True),
    )
    completed = await runner.run(run_state)
    assert completed.status == "done"
    assert completed.error is None
    assert completed.render_manifest is not None
    assert completed.render_manifest.error is None
    all_warnings = list(completed.warnings) + list(completed.render_manifest.warnings)
    assert len(all_warnings) >= 1


async def test_bug20_fastapi_origin_middleware_returns_403_never_500(tmp_path: Path) -> None:
    """BUG-20: Origin middleware returns HTTP 403 (never 500) on Origin: null & malformed Origin."""
    from gtm_videogen.api.app import create_app

    app = create_app(data_dir=tmp_path)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        for bad_origin in ("null", "not-a-valid-origin://[bad", "http://evil.example.org"):
            resp = await client.post(
                "/api/runs",
                json=make_run_input(mock=True).model_dump(),
                headers={"Origin": bad_origin},
            )
            assert resp.status_code == 403, (
                f"Expected 403 for Origin={bad_origin!r}, got {resp.status_code}"
            )


async def test_bug22_fastapi_jobs_endpoint_includes_other_active_jobs_count(
    tmp_path: Path,
) -> None:
    """BUG-22 (API): GET /api/jobs?run_id=... counts active jobs belonging to other runs."""
    from gtm_videogen.api.app import create_app

    job_registry = JobRegistry(max_concurrent=4)
    started = asyncio.Event()
    release = asyncio.Event()

    async def _other_run_worker(tok: CancellationToken) -> None:
        started.set()
        await release.wait()

    other_job = await job_registry.submit("run_other_active", _other_run_worker, stage="render")
    await asyncio.wait_for(started.wait(), timeout=2.0)

    app = create_app(data_dir=tmp_path, job_registry=job_registry)
    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            resp = await client.get("/api/jobs", params={"run_id": "run_scoped_1"})
            assert resp.status_code == 200
            body = resp.json()
            assert body["jobs"] == []
            assert body["other_active_jobs_count"] == 1

            release.set()
            await job_registry.wait_for_job(other_job.id)

            resp_after = await client.get("/api/jobs", params={"run_id": "run_scoped_1"})
            assert resp_after.status_code == 200
            assert resp_after.json()["other_active_jobs_count"] == 0
    finally:
        release.set()
        await job_registry.wait_for_job(other_job.id)


async def test_bug24_fontsdir_escaping_with_spaces_and_media_qa_ocr(tmp_path: Path) -> None:
    """BUG-24: FFmpeg :fontsdir= escaping works with spaces/special chars & MediaQA runs OCR."""
    from gtm_videogen.media.captions import escape_ffmpeg_filter_path
    from gtm_videogen.media.media_qa import MediaQA

    spaced_dir = tmp_path / "fonts dir:test"
    spaced_dir.mkdir(parents=True, exist_ok=True)
    escaped = escape_ffmpeg_filter_path(spaced_dir)
    assert r"\:" in escaped or ":" not in escaped

    clip = make_synthetic_clip(
        tmp_path / "ocr_bug24.mp4",
        width=1280,
        height=720,
        fps=24,
        duration_sec=2.0,
        color="0x111827",
        text="DIRECT TRADE",
    )
    sheet = make_shot_sheet(
        target_duration_sec=2.0,
        shots=[
            Shot(
                id="s1",
                index=0,
                beat="cta",
                duration_sec=2.0,
                visual_prompt="Card",
                on_screen_text="DIRECT TRADE",
            )
        ],
    )
    qa = MediaQA()
    report = await qa.evaluate(video_path=clip, sheet=sheet, run_ocr=True)
    assert len(report.ocr_checks) >= 1
    assert any(c.passed for c in report.ocr_checks)


def test_bug27_llm_router_routes_sound_design_to_text_model() -> None:
    """BUG-27: LLMRouter routes sound_design stage to text_model, never vision_model."""
    from gtm_videogen.core.config import Settings
    from gtm_videogen.providers.llm_router import LLMRouter

    settings = Settings(text_model="text-only-model-v1", vision_model="vision-only-model-v9")
    router = LLMRouter(settings=settings)
    chosen = router.resolve_model("sound_design")
    assert chosen == "text-only-model-v1"
    assert chosen != "vision-only-model-v9"


def test_bug28_cli_doctor_reports_binaries_and_fonts() -> None:
    """BUG-28: `gtm-videogen doctor` reports real ffmpeg/ffprobe/tesseract/font checks."""
    from typer.testing import CliRunner

    from gtm_videogen.cli.main import app

    res = CliRunner().invoke(app, ["doctor"])
    assert res.exit_code == 0, res.output
    lower = res.output.lower()
    assert "ffmpeg" in lower
    assert "ffprobe" in lower
    assert "tesseract" in lower
    assert "font" in lower


@pytest.mark.parametrize(
    "unsafe_code",
    [
        "import os\nos.system('echo pwned')\n",
        "import subprocess\nsubprocess.run(['ls'])\n",
        "with open('/etc/passwd') as f:\n    data = f.read()\n",
        "import socket\nsocket.create_connection(('127.0.0.1', 80))\n",
    ],
)
def test_f21_manim_ast_sandbox_rejects_unsafe_code(unsafe_code: str) -> None:
    """F21 Manim AST Sandbox: blocks os.system, subprocess, open('/etc/passwd'), and socket."""
    from gtm_videogen.media.manim_renderer import ManimSandboxError, validate_manim_code

    with pytest.raises((ManimSandboxError, ValueError)):
        validate_manim_code(unsafe_code)
