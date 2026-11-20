"""Comprehensive offline unit tests for M1: models, core infrastructure, and pure script rules."""

from __future__ import annotations

import asyncio
import base64
import json
import math
import re
import sys
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from gtm_videogen.core import (
    BudgetExceeded,
    CacheStore,
    CancellationToken,
    CorruptRunError,
    CostMeter,
    JobRegistry,
    MediaStore,
    MediaStoreError,
    ResilientHttpClient,
    RunStore,
    Settings,
    SSRFError,
    assert_public_url,
    current_cost_meter,
    estimate_run_cost,
    get_logger,
    log_context,
    parse_data_url,
    read_ledger_total,
    use_cost_meter,
    validate_redirect_chain,
)
from gtm_videogen.models import (
    ChatOptions,
    ClaimCitation,
    CostEntry,
    CreativeFingerprint,
    JobStatusDTO,
    MusicCue,
    OCRCheck,
    QAReport,
    RenderedAudio,
    RenderedClip,
    RenderManifest,
    ResearchBundle,
    RunInput,
    RunState,
    Script,
    ScriptLine,
    SfxCue,
    ShotSheet,
    SoundPlan,
    SyncCheck,
    Version,
    VoiceCasting,
)
from gtm_videogen.rules import (
    CreativeHistoryStore,
    beat_fit_problem,
    check_claim_grounding,
    diversity_scores,
    estimate_spoken_duration,
    evaluate_script_gate,
    extract_claim_tokens,
    find_placeholders,
    fit_timings,
    format_previously_explored,
    normalize_brand_url,
    rendered_text_problem,
    select_diverse,
    select_voice,
    spoken_form,
)

# ---------------------------------------------------------------------------
# 1. Models & BUG-03 Strict Validation
# ---------------------------------------------------------------------------


def test_run_input_forbids_extra_fields_and_supports_all_audio_modes() -> None:
    """BUG-03: unknown fields raise ValidationError; audio_mode defaults to 'vo'."""
    default_req = RunInput(
        brand_name="Acme",
        brand_url="https://acme.example.com",
        goal="Increase trial signups",
    )
    assert default_req.audio_mode == "vo"

    for mode in ("vo", "foley", "both"):
        req = RunInput(
            brand_name="Acme",
            brand_url="https://acme.example.com",
            goal="Increase trial signups",
            audio_mode=mode,  # type: ignore[arg-type]
        )
        assert req.audio_mode == mode

    with pytest.raises(ValidationError):
        RunInput(
            brand_name="Acme",
            brand_url="https://acme.example.com",
            goal="Increase trial signups",
            unknown_field="should_fail",  # type: ignore[call-arg]
        )

    with pytest.raises(ValidationError):
        RunInput(
            brand_name="Acme",
            brand_url="ftp://acme.example.com",
            goal="Bad scheme",
        )

    # Hostless URLs and whitespace-only brand_name/goal must be rejected
    for bad_url in ("https://", "http:///"):
        with pytest.raises(ValidationError):
            RunInput(brand_name="Acme", brand_url=bad_url, goal="Valid goal")
        with pytest.raises(ValidationError):
            RunInput(
                brand_name="Acme",
                brand_url="https://acme.example.com",
                goal="Valid goal",
                inspiration_url=bad_url,
            )

    with pytest.raises(ValidationError):
        RunInput(brand_name="   ", brand_url="https://acme.example.com", goal="Valid goal")

    with pytest.raises(ValidationError):
        RunInput(brand_name="Acme", brand_url="https://acme.example.com", goal="   \t\n  ")


def test_all_schema_models_round_trip_json(
    sample_run_state: RunState,
    sample_research: ResearchBundle,
    sample_creative_bible,
    sample_shot_sheet: ShotSheet,
    sample_sound_plan: SoundPlan,
    sample_concepts,
) -> None:
    """Verify full RunState + nested schemas serialize cleanly to and from JSON."""
    script = Script(
        full_vo_text="Fresh ginger hits different.",
        lines=[
            ScriptLine(
                shot_id="s1", text="Fresh ginger hits different.", word_count=4, estimated_sec=1.5
            )
        ],
        claims_used=[
            ClaimCitation(
                claim_text="4g sugar", source_index=0, source_fact="Only 4g of organic cane sugar"
            )
        ],
    )
    manifest = RenderManifest(
        run_id=sample_run_state.run_id,
        clips=[
            RenderedClip(
                shot_id="s1", path="/tmp/s1.mp4", duration_sec=3.0, fps=24, width=1920, height=1080
            )
        ],
        audio_assets=[RenderedAudio(kind="vo", cue_id="s1", path="/tmp/s1.wav", duration_sec=2.1)],
    )
    qa = QAReport(
        passed=True,
        originality_score=8.5,
        bible_coherence_score=9.0,
        loudness_lufs=-14.2,
        ocr_checks=[
            OCRCheck(
                shot_id="s3",
                expected_text="4g Sugar",
                observed_text="4g Sugar",
                passed=True,
                similarity=1.0,
            )
        ],
        sync_checks=[
            SyncCheck(
                shot_id="s1",
                expected_vo="Fresh ginger",
                transcribed_vo="Fresh ginger",
                passed=True,
                similarity=1.0,
            )
        ],
    )
    version = Version(
        version_id="v1",
        label="Concept A",
        creative_bible=sample_creative_bible,
        shot_sheet=sample_shot_sheet,
        sound_plan=sample_sound_plan,
        render_manifest=manifest,
        qa_report=qa,
    )
    sample_run_state.research = sample_research
    sample_run_state.concepts = sample_concepts
    sample_run_state.creative_bible = sample_creative_bible
    sample_run_state.script = script
    sample_run_state.shot_sheet = sample_shot_sheet
    sample_run_state.sound_plan = sample_sound_plan
    sample_run_state.render_manifest = manifest
    sample_run_state.versions = [version]
    sample_run_state.qa_report = qa

    dumped = sample_run_state.model_dump_json()
    restored = RunState.model_validate_json(dumped)
    assert restored.run_id == sample_run_state.run_id
    assert restored.versions[0].qa_report is not None
    assert restored.versions[0].qa_report.originality_score == 8.5

    opts = ChatOptions(temperature=0.75, top_p=0.95, max_tokens=2048, presence_penalty=0.2)
    assert ChatOptions.model_validate_json(opts.model_dump_json()).temperature == 0.75


# ---------------------------------------------------------------------------
# 2. Core Config & Structured Logging
# ---------------------------------------------------------------------------


def test_settings_and_structured_logging(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    settings = Settings(
        GTM_DATA_DIR=tmp_path / "data",
        GTM_RUNS_DIR=tmp_path / "runs",
        GTM_CACHE_DIR=tmp_path / "cache",
    )
    settings.ensure_dirs()
    assert settings.runs_dir.is_dir()
    assert settings.cache_dir.is_dir()

    logger = get_logger("gtm_videogen.test_structured")
    with log_context(run_id="run_42", version_id="v1", shot_id="s2", stage="render"):
        logger.info("clip rendered", extra={"duration_ms": 345})

    captured = capsys.readouterr().err.strip().splitlines()[-1]
    parsed = json.loads(captured)
    assert parsed["run_id"] == "run_42"
    assert parsed["shot_id"] == "s2"
    assert parsed["stage"] == "render"
    assert parsed["duration_ms"] == 345


# ---------------------------------------------------------------------------
# 3. SSRF Protection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_url",
    [
        "http://127.0.0.1/admin",
        "http://localhost/secret",
        "http://sub.localhost/test",
        "http://10.0.0.5/internal",
        "http://172.16.0.1/internal",
        "http://192.168.1.10/router",
        "http://169.254.169.254/latest/meta-data/",
        "http://[::1]/loopback",
        "http://[fe80::1]/linklocal",
        "http://[::ffff:127.0.0.1]/mapped",
        "http://[::ffff:10.0.0.1]/mapped-private",
        "https://93.184.216.34:8080/nonstandard",
        "file:///etc/passwd",
        "ftp://93.184.216.34/file",
        "https://user:pass@93.184.216.34/",
        "http://127.1/",
        "http://2130706433/",
        "http://0x7f000001/",
        "http://0177.0.0.1/",
    ],
)
def test_ssrf_blocks_private_loopback_ipv6_ports_and_schemes(bad_url: str) -> None:
    with pytest.raises(SSRFError):
        assert_public_url(bad_url, resolve_dns=False)


def test_ssrf_allows_public_ip_and_validates_redirect_chains() -> None:
    assert (
        assert_public_url("https://93.184.216.34/page", resolve_dns=False)
        == "https://93.184.216.34/page"
    )

    ok_chain = [
        "https://93.184.216.34/a",
        "/b",
        "https://93.184.216.34/c",
        "https://93.184.216.34/d",
    ]
    assert len(validate_redirect_chain(ok_chain, max_hops=3, resolve_dns=False)) == 4

    # Too many hops (>3)
    with pytest.raises(SSRFError):
        validate_redirect_chain(ok_chain + ["/e"], max_hops=3, resolve_dns=False)

    # Redirect to private IP
    with pytest.raises(SSRFError):
        validate_redirect_chain(
            ["https://93.184.216.34/start", "http://127.0.0.1/internal"],
            resolve_dns=False,
        )


@pytest.mark.parametrize(
    "homograph_url",
    [
        "http://127\u30020\u30020\u30021/",
        "http://169\u3002254\u3002169\u3002254/",
        "http://127\uff0e0\uff0e0\uff0e1/",
        "http://127\uff610\uff610\uff611/",
        "http://localhost\u3002/",
        "http://sub\u3002localhost\u3002/",
        "http://\uff11\uff12\uff17\uff0e\uff10\uff0e\uff10\uff0e\uff11/",
    ],
)
def test_ssrf_blocks_idna_dot_homographs_without_dns(homograph_url: str) -> None:
    """BUG-C1-I2-01: Unicode dot-homographs and fullwidth digits must be blocked without DNS."""
    with pytest.raises(SSRFError):
        assert_public_url(homograph_url, resolve_dns=False)


def test_ssrf_allows_valid_public_idn_without_dns() -> None:
    assert (
        assert_public_url("https://m\u00fcnchen.de/path", resolve_dns=False)
        == "https://m\u00fcnchen.de/path"
    )


@pytest.mark.parametrize(
    "malformed_url",
    [
        "http://127.0.0.1\x00.example.com/",
        "http://[::1/",
        "http://example.com\r\nX-Injected: 1/",
        "http://example.com:99999/",
        "http://local host/",
        "http://a..b/",
        "http://\u3002/",
    ],
)
def test_ssrf_raises_ssrf_error_not_raw_value_error(malformed_url: str) -> None:
    """BUG-C1-I2-02: null bytes, control chars, and malformed IPv6 must raise SSRFError."""
    with pytest.raises(SSRFError):
        assert_public_url(malformed_url, resolve_dns=False)

    with pytest.raises(SSRFError):
        validate_redirect_chain(["https://93.184.216.34/start", malformed_url], resolve_dns=False)


# ---------------------------------------------------------------------------
# 4. JobRegistry & 4-Layer Cancellation (BUG-01, BUG-02, BUG-05, BUG-19, BUG-22)
# ---------------------------------------------------------------------------


async def test_job_registry_concurrent_slot_release_and_dto_serialization() -> None:
    """BUG-01/02/22: set-based slots release cleanly, DTOs serialize, global count works."""
    registry = JobRegistry(max_concurrent=2)
    started_gate = asyncio.Event()
    release_gate = asyncio.Event()
    running_count = 0

    async def worker(token: CancellationToken) -> str:
        nonlocal running_count
        running_count += 1
        if running_count >= 2:
            started_gate.set()
        await release_gate.wait()
        token.raise_if_cancelled()
        return "ok"

    j1 = await registry.submit("run_a", worker, stage="render", shot_id="s1")
    j2 = await registry.submit("run_b", worker, stage="render", shot_id="s2")
    j3 = await registry.submit("run_c", worker, stage="render", shot_id="s3")

    await asyncio.wait_for(started_gate.wait(), timeout=2.0)

    # BUG-22: global other-active-jobs count works while filtering
    assert registry.active_count == 2
    assert registry.other_active_jobs_count(exclude_run_id="run_a") == 2
    assert registry.other_active_jobs_count(exclude_job_id=j1.id) == 2

    # BUG-02: strictly JSON-serializable DTOs while jobs are live
    dtos = registry.list_jobs()
    assert len(dtos) == 3
    for dto in dtos:
        assert isinstance(dto, JobStatusDTO)
        parsed = json.loads(dto.model_dump_json())
        assert "id" in parsed and "state" in parsed

    release_gate.set()
    await asyncio.gather(
        registry.wait_for_job(j1.id),
        registry.wait_for_job(j2.id),
        registry.wait_for_job(j3.id),
    )

    # BUG-01: all slots released; subsequent job runs immediately without deadlock
    assert registry.active_count == 0
    j4 = await registry.submit("run_d", lambda _tok: asyncio.sleep(0.01))
    done_j4 = await registry.wait_for_job(j4.id)
    assert done_j4 is not None and done_j4.state == "done"


async def test_four_layer_cancellation_never_overwrites_cancelled_with_done() -> None:
    """BUG-05: CancellationToken cancels callbacks/tasks/subprocesses; state stays cancelled."""
    registry = JobRegistry(max_concurrent=2)
    callback_fired = False
    started = asyncio.Event()

    async def stubborn_job(token: CancellationToken) -> str:
        nonlocal callback_fired
        token.register_callback(lambda: _set_flag())
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-c", "import time; time.sleep(30)"
        )
        token.track_process(proc)
        started.set()
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            # Even if a worker swallows CancelledError and returns normally,
            # JobRegistry MUST NOT overwrite 'cancelled' with 'done' (BUG-05)!
            return "finished_anyway"
        finally:
            await proc.wait()
            token.untrack_process(proc)
        return "finished"

    def _set_flag() -> None:
        nonlocal callback_fired
        callback_fired = True

    dto = await registry.submit("run_cancel", stubborn_job)
    await asyncio.wait_for(started.wait(), timeout=2.0)

    cancelled_count = await registry.cancel_run("run_cancel", reason="User stopped")
    assert cancelled_count == 1
    assert callback_fired is True

    final_dto = await registry.wait_for_job(dto.id)
    assert final_dto is not None
    assert final_dto.state == "cancelled"
    assert registry.active_count == 0


# ---------------------------------------------------------------------------
# 5. ResilientHttpClient (BUG-14)
# ---------------------------------------------------------------------------


async def test_http_client_retries_502_and_raises_before_json() -> None:
    """BUG-14: retries 502 HTML without JSONDecodeError; raises HTTPStatusError if exhausted."""
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            return httpx.Response(502, text="<html>Bad Gateway</html>", request=request)
        return httpx.Response(200, json={"ok": True}, request=request)

    async with ResilientHttpClient(
        max_retries=3,
        backoff_base_sec=0.01,
        transport=httpx.MockTransport(handler),
    ) as client:
        data = await client.get_json("https://api.example.com/status")
        assert data == {"ok": True}
        assert attempts == 3

    # Exhausted 503 HTML must raise HTTPStatusError (NOT json.JSONDecodeError)
    def always_503(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="<html>Service Unavailable</html>", request=request)

    async with ResilientHttpClient(
        max_retries=1,
        backoff_base_sec=0.01,
        transport=httpx.MockTransport(always_503),
    ) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await client.get_json("https://api.example.com/down")


async def test_http_client_cancellation_aborts_retry_wait() -> None:
    """BUG-14 & BUG-05: CancellationToken aborts in-flight backoff sleep immediately."""

    def always_429(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429, headers={"Retry-After": "30"}, text="rate limited", request=request
        )

    token = CancellationToken()

    async with ResilientHttpClient(
        max_retries=3,
        backoff_base_sec=5.0,
        transport=httpx.MockTransport(always_429),
    ) as client:
        task = asyncio.create_task(client.get_json("https://api.example.com/rate", cancel=token))
        await asyncio.sleep(0.05)
        token.cancel("abort now")
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1.0)


# ---------------------------------------------------------------------------
# 6. CostMeter & Preflight Estimation (BUG-12, BUG-15, BUG-25)
# ---------------------------------------------------------------------------


def test_cost_meter_reserve_settle_release_budget_cap_and_missing_ledger(
    tmp_path: Path,
    sample_shot_sheet: ShotSheet,
    sample_sound_plan: SoundPlan,
) -> None:
    """BUG-12, BUG-15, BUG-25."""
    ledger_file = tmp_path / "missing_ledger.jsonl"
    # BUG-25: missing file returns 0.0 cleanly
    assert read_ledger_total(ledger_file) == 0.0

    meter = CostMeter("run_cost", cap_usd=1.0, ledger_path=ledger_file)
    with use_cost_meter(meter):
        assert current_cost_meter() is meter
        r1 = meter.reserve("shot_1", 0.60, stage="render", provider="fal", model="kling")
        # Second reservation of 0.50 must fail atomically before spending (0.60 + 0.50 > 1.0)
        with pytest.raises(BudgetExceeded):
            meter.reserve("shot_2", 0.50, stage="render", provider="fal", model="kling")

        # Settle r1 for actual 0.45 -> remaining is 0.55
        entry = meter.settle(r1, 0.45)
        assert isinstance(entry, CostEntry)
        assert meter.spent_usd == 0.45
        assert meter.reserved_usd == 0.0

        # Now 0.50 succeeds; test release on error
        r2 = meter.reserve("shot_2", 0.50, stage="render", provider="fal", model="kling")
        assert meter.state == "warn"  # 0.95 / 1.0 >= 0.8
        meter.release(r2)
        assert meter.reserved_usd == 0.0
        assert meter.remaining_usd == 0.55

    assert current_cost_meter() is None
    assert read_ledger_total(ledger_file) == pytest.approx(0.45)

    # BUG-15: preflight includes LLM, image, video (with min_seconds floor), voice, SFX, music
    sample_sound_plan.sfx_cues = [
        SfxCue(shot_id="s1", at_sec=0.0, duration_sec=1.0, prompt="slice")
    ]
    sample_sound_plan.music_cue = MusicCue(prompt="acoustic bed", duration_sec=15.0)
    est = estimate_run_cost(
        shot_sheet=sample_shot_sheet,
        sound_plan=sample_sound_plan,
        image_count=2,
        video_usd_per_sec=0.10,
        video_min_seconds=5.0,
    )
    # 3 video shots (3s->5s, 5s->5s, 4s->5s; s4 motion_card=0s) => 15s * $0.10 = $1.50
    assert est["video_usd"] == pytest.approx(1.50)
    assert est["voice_usd"] > 0
    assert est["sfx_usd"] > 0
    assert est["music_usd"] > 0
    assert est["image_usd"] > 0
    assert est["llm_usd"] > 0
    assert est["total_usd"] == pytest.approx(
        est["llm_usd"]
        + est["image_usd"]
        + est["video_usd"]
        + est["voice_usd"]
        + est["sfx_usd"]
        + est["music_usd"],
        abs=1e-3,
    )


# ---------------------------------------------------------------------------
# 7. MediaStore & CacheStore (BUG-11, BUG-26, P4, P7)
# ---------------------------------------------------------------------------


async def test_media_store_resolves_all_ref_types_and_guards_malformed_data_urls(
    tmp_path: Path,
) -> None:
    """BUG-11, BUG-26, P7."""
    png_bytes = b"\x89PNG\r\n\x1a\nfake_png_payload"

    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=png_bytes, request=request)

    async with ResilientHttpClient(
        transport=httpx.MockTransport(mock_handler),
        resolve_dns_for_ssrf=False,
    ) as http_client:
        store = MediaStore(tmp_path / "media", http_client=http_client, resolve_dns_for_ssrf=False)

        # Save + resolve media:// and /api/media/...
        rel_id, saved = store.save_bytes(png_bytes, ext=".png", subdir="clips")
        assert saved.is_file()
        assert await store.resolve_to_path(f"media://{rel_id}") == saved
        assert await store.resolve_to_path(f"/api/media/{rel_id}") == saved
        assert await store.resolve_to_path(saved) == saved

        # Resolve valid data: URL
        b64 = base64.b64encode(png_bytes).decode("ascii")
        resolved_data = await store.resolve_to_path(f"data:image/png;base64,{b64}")
        assert resolved_data.read_bytes() == png_bytes

        # Resolve public https:// URL
        resolved_http = await store.resolve_to_path("https://93.184.216.34/pic.png")
        assert resolved_http.read_bytes() == png_bytes

        # Reject private URL via SSRF
        with pytest.raises(SSRFError):
            await store.resolve_to_path("http://127.0.0.1/secret.png")

        # Reject path traversal
        with pytest.raises(MediaStoreError):
            await store.resolve_to_path("media://../../etc/passwd")

    # BUG-26: strict malformed data: URL rejection
    for bad_data in (
        "data:image/png",
        "data:image/png;base64,",
        "data:image/png;base64,!!!not_base64!!!",
        "not_a_data_url",
    ):
        with pytest.raises(MediaStoreError):
            parse_data_url(bad_data)


def test_cache_store_verifies_artifact_existence_and_non_empty_size(tmp_path: Path) -> None:
    """P4: cache hit requires artifact file to exist on disk and be >0 bytes."""
    caches = CacheStore(tmp_path / "caches")
    clip_file = tmp_path / "clip.mp4"
    clip_file.write_bytes(b"mp4-binary-data")

    key = caches.clip_cache.make_key({"shot": "s1", "model": "kling", "aspect": "16:9"})
    caches.clip_cache.put(key, artifact_path=clip_file, metadata={"usd": 0.35})

    hit = caches.clip_cache.get(key)
    assert hit is not None
    assert hit.artifact_path == clip_file.resolve()

    # Delete underlying file -> MUST NOT return a stale hit
    clip_file.unlink()
    assert caches.clip_cache.get(key) is None

    # Empty (0-byte) file -> MUST NOT be accepted or returned
    empty_audio = tmp_path / "empty.wav"
    empty_audio.write_bytes(b"")
    akey = caches.audio_cache.make_key({"cue": "vo1"})
    with pytest.raises(ValueError):
        caches.audio_cache.put(akey, artifact_path=empty_audio)

    # Non-file judgeCache works without artifact_path
    jkey = caches.judge_cache.make_key({"prompt": "score"})
    caches.judge_cache.put(jkey, metadata={"score": 9})
    jhit = caches.judge_cache.get(jkey)
    assert jhit is not None and jhit.metadata["score"] == 9


# ---------------------------------------------------------------------------
# 8. RunStore Atomic Persistence, Corrupt Recovery & Orphan Healing (BUG-19)
# ---------------------------------------------------------------------------


def test_run_store_heals_orphaned_working_runs_and_quarantines_corrupt_json(
    tmp_path: Path,
    sample_run_state: RunState,
) -> None:
    """BUG-19: stale 'working' run without active job is healed to 'interrupted' on load."""
    registry = JobRegistry()
    store = RunStore(tmp_path / "runs", job_registry=registry)

    sample_run_state.status = "working"
    store.save(sample_run_state)

    loaded = store.load(sample_run_state.run_id)
    assert loaded is not None
    assert loaded.status == "interrupted"
    assert any("orphaned" in w.lower() for w in loaded.warnings)

    # Corrupt JSON file gets quarantined instead of crashing
    corrupt_path = tmp_path / "runs" / "run_corrupt.json"
    corrupt_path.write_text("{broken json!!!", encoding="utf-8")
    assert store.load("run_corrupt", recover_corrupt=True) is None
    assert not corrupt_path.exists()
    assert len(list((tmp_path / "runs").glob("run_corrupt.corrupt.*.json"))) == 1

    # With recover_corrupt=False raises CorruptRunError
    corrupt_path2 = tmp_path / "runs" / "run_corrupt2.json"
    corrupt_path2.write_text("{broken", encoding="utf-8")
    with pytest.raises(CorruptRunError):
        store.load("run_corrupt2", recover_corrupt=False)


# ---------------------------------------------------------------------------
# 9. Pure Script Rules (BUG-17, BUG-18, BUG-21, BUG-23, Pillar 3/6)
# ---------------------------------------------------------------------------


def test_fit_timings_enforces_min_max_and_never_produces_nan() -> None:
    """BUG-21: enforces BOTH min_shot and max_shot, guards infeasible bounds & NaN weights."""
    # Normal redistribution enforcing max_shot=8.0 and min_shot=1.5
    fitted = fit_timings([0.2, 20.0, 2.0, 2.0], target_sec=15.0, min_shot=1.5, max_shot=8.0)
    assert len(fitted) == 4
    assert sum(fitted) == pytest.approx(15.0, abs=1e-6)
    assert all(1.5 - 1e-6 <= d <= 8.0 + 1e-6 for d in fitted)

    # Infeasible min_shot: 10 shots * 1.5s = 15s > 10s -> even scaling (1.0s each), NO NaN!
    too_many = fit_timings([1.0] * 10, target_sec=10.0, min_shot=1.5, max_shot=8.0)
    assert all(math.isfinite(d) and d > 0 for d in too_many)
    assert sum(too_many) == pytest.approx(10.0, abs=1e-6)

    # Infeasible max_shot: 2 shots * 8.0s = 16s < 20s -> even scaling (10.0s each)
    too_few = fit_timings([2.0, 3.0], target_sec=20.0, min_shot=1.5, max_shot=8.0)
    assert sum(too_few) == pytest.approx(20.0, abs=1e-6)

    # Zero, negative, NaN, inf weights sanitized cleanly
    dirty = fit_timings([0.0, -5.0, float("nan"), float("inf")], target_sec=12.0)
    assert all(math.isfinite(d) and 1.5 <= d <= 8.0 for d in dirty)
    assert sum(dirty) == pytest.approx(12.0, abs=1e-6)


def test_speakability_and_spoken_form() -> None:
    expanded = spoken_form("Save 20% at acme.ai for $29 today!")
    assert "twenty percent" in expanded
    assert "acme dot a i" in expanded
    assert "twenty nine dollars" in expanded

    dur = estimate_spoken_duration("Fresh ginger hits different.")
    assert 0.8 < dur < 3.0
    assert beat_fit_problem("Fresh ginger hits different.", 3.0) is None

    crowded = (
        "This extraordinarily verbose sentence tries to cram far too many technical specifications "
        "and twenty five different product claims into a single two second visual cut."
    )
    assert beat_fit_problem(crowded, 2.0) is not None


def test_claim_grounding_and_anti_garbled_text(sample_research: ResearchBundle) -> None:
    tokens = extract_claim_tokens(
        "Ranked #1 with 99% faster results and $49 savings, 100% guaranteed!"
    )
    assert any("#1" in t for t in tokens)
    assert any("99%" in t for t in tokens)
    assert any("$49" in t for t in tokens)

    assert "[BRAND]" in find_placeholders("Try [BRAND] today — TBD!")
    assert "TBD" in find_placeholders("Try [BRAND] today — TBD!")

    # Grounded claim passes
    ok_issues = check_claim_grounding(
        "Only 4g of organic cane sugar per bottle. Save 20% on your first 6-pack.",
        sample_research,
        offer="Save 20% on your first 6-pack",
    )
    assert ok_issues == []

    # Ungrounded claim + banned phrase + superlative + placeholder -> all non-overridable blocking
    bad_issues = check_claim_grounding(
        "The fastest #1 miracle cure with 95% off at [BRAND] that cures migraines!",
        sample_research,
    )
    codes = {i.code for i in bad_issues}
    assert "UNGROUNDED_CLAIM" in codes
    assert "BANNED_PHRASE" in codes
    assert "UNSUPPORTED_SUPERLATIVE" in codes
    assert "UNRESOLVED_PLACEHOLDER" in codes
    assert all(not i.overridable for i in bad_issues)

    # Anti-garbled-text
    assert (
        rendered_text_problem(
            "Close-up of a phone screen showing UI dashboard and text reading 'Buy Now'"
        )
        is not None
    )
    assert (
        rendered_text_problem(
            "Sunlit ceramic cup of ginger tonic on warm wood table, shallow depth of field."
        )
        is None
    )


def test_script_gate_bug18_and_bug23(
    sample_shot_sheet: ShotSheet,
    sample_research: ResearchBundle,
) -> None:
    """BUG-18 (product_hero without photos) and BUG-23 (non-overridable vs overridable)."""
    # Clean sheet with product photos passes
    ok_report = evaluate_script_gate(
        sample_shot_sheet,
        sample_research,
        offer="Save 20% on your first 6-pack",
    )
    assert ok_report.passed is True
    assert ok_report.blocking_errors == []

    # BUG-18: product_hero=True with no product photos fails (non-overridable)
    no_photo_research = sample_research.model_copy(update={"product_photos": []})
    photo_report = evaluate_script_gate(
        sample_shot_sheet,
        no_photo_research,
        offer="Save 20% on your first 6-pack",
        gate_override=True,
    )
    assert photo_report.passed is False
    assert any(
        e.code == "PRODUCT_HERO_WITHOUT_PHOTOS" and not e.overridable
        for e in photo_report.blocking_errors
    )

    # BUG-23: gate_override=True downgrades overridable issues (e.g. missing camera_move)
    # but NEVER bypasses ungrounded claims!
    mutated = sample_shot_sheet.model_copy(deep=True)
    mutated.shots[0].camera_move = ""  # overridable
    mutated.shots[0].vo_line = "Save 99% with the world's best tonic."  # non-overridable

    overridden_report = evaluate_script_gate(
        mutated,
        sample_research,
        offer="Save 20% on your first 6-pack",
        gate_override=True,
    )
    assert overridden_report.passed is False
    assert any(w.code == "MISSING_CAMERA_SPEC" for w in overridden_report.warnings)
    assert any(
        e.code == "UNGROUNDED_CLAIM" and not e.overridable
        for e in overridden_report.blocking_errors
    )


def test_voice_casting_attribute_matching_and_history_avoidance() -> None:
    """BUG-17: matches gender, age, accent, energy, pace and avoids CreativeHistory repeats."""
    casting = VoiceCasting(
        gender="male",
        age="mature",
        accent="uk",
        energy="authoritative",
        pace="measured",
    )
    first = select_voice(casting)
    assert first.voice_id == "v_arthur_m_mature_uk"

    history = [
        CreativeFingerprint(
            run_id="r1",
            brand_url="https://acme.example.com",
            arc_name="Prestige Documentary",
            hook_opening="Close-up on craft",
            visual_world="Workshop",
            lighting_style="Dramatic",
            camera_style="Static",
            voice_id=first.voice_id,
        )
    ]
    second = select_voice(casting, brand_history=history)
    assert second.voice_id != first.voice_id


def test_creative_history_persistence_cap_10_and_diversity_scoring(
    tmp_path: Path,
    sample_concepts,
) -> None:
    """Pillar 3 & Pillar 6: history caps at 10 per URL; diversity separates orthogonal concepts."""
    assert normalize_brand_url("HTTPS://www.Acme.COM/shop/") == "acme.com/shop"

    store = CreativeHistoryStore(tmp_path / "creative_history.json")
    for idx in range(13):
        store.record(
            CreativeFingerprint(
                run_id=f"run_{idx}",
                brand_url="https://www.SolaraBotanics.example.com/",
                arc_name=f"Arc {idx}",
                hook_opening=f"Hook {idx}",
                visual_world=f"World {idx}",
                lighting_style="Natural",
                camera_style="Handheld",
                voice_id=f"voice_{idx}",
            )
        )

    hist = store.get_history("https://solarabotanics.example.com")
    assert len(hist.entries) == 10
    assert hist.entries[0].run_id == "run_3"
    assert hist.entries[-1].run_id == "run_12"

    prompt_block = format_previously_explored(hist)
    assert "Previously Explored (Do Not Repeat)" in prompt_block
    assert "Arc 12" in prompt_block

    # Diversity scoring: c1 and c4 are duplicates, so selecting 3 picks c1/c4 once + c2 + c3
    scores = diversity_scores(sample_concepts)
    assert len(scores) == 4
    assert scores[1] > scores[0]  # c2 is more orthogonal than duplicated c1
    selected = select_diverse(sample_concepts, 3)
    selected_ids = {c.id for c in selected}
    assert {"c2", "c3"}.issubset(selected_ids)
    assert not ({"c1", "c4"}.issubset(selected_ids))


# ---------------------------------------------------------------------------
# 10. Zero Hard-Coded Legacy Templates Guardrail
# ---------------------------------------------------------------------------


def test_zero_legacy_hardcoded_templates_in_src(repo_src_dir: Path) -> None:
    """Verify zero legacy hard-coded creative template identifiers exist anywhere in src/."""
    banned_pattern = re.compile(
        r"confession_loop|NEON_LAB_RULES|droneBed|END_STING|112 BPM|OFF_FRAME_POUR|cola pours",
        re.IGNORECASE,
    )
    matches: list[str] = []
    for py_file in sorted(repo_src_dir.rglob("*.py")):
        content = py_file.read_text(encoding="utf-8")
        if banned_pattern.search(content):
            matches.append(str(py_file))
    assert matches == [], f"Legacy template strings found in src/: {matches}"


# ---------------------------------------------------------------------------
# 11. Iteration 2 Regression Tests (All 11 Reviewer & Challenger Defects)
# ---------------------------------------------------------------------------


def test_atomic_write_temp_cleanup_and_corrupt_binary_or_nondict_recovery(
    tmp_path: Path,
    sample_run_state: RunState,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fix 2: no temp file leaks on write error; binary/non-dict JSON & dotfiles handled safely."""
    # ContentCache.put temp cleanup on JSON serialization failure
    caches = CacheStore(tmp_path / "cache")
    clip_file = tmp_path / "a.mp4"
    clip_file.write_bytes(b"video-bytes")
    with pytest.raises(TypeError):
        caches.clip_cache.put("k", artifact_path=clip_file, metadata={"p": Path("/tmp/x")})
    assert list(caches.clip_cache.root_dir.glob(".tmp_cache_*")) == []

    # MediaStore.save_bytes temp cleanup on write failure
    media = MediaStore(tmp_path / "media")
    with pytest.raises(TypeError):
        media.save_bytes("not-bytes", ext=".bin")  # type: ignore[arg-type]
    assert list((tmp_path / "media" / "assets").glob(".tmp_media_*")) == []

    # RunStore.save temp cleanup on serialization failure
    run_store = RunStore(tmp_path / "runs")
    monkeypatch.setattr(
        RunState,
        "model_dump_json",
        lambda self, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    with pytest.raises(RuntimeError, match="boom"):
        run_store.save(sample_run_state)
    assert list((tmp_path / "runs").glob(".tmp_*")) == []

    # ContentCache.get recovers from corrupt binary bytes and non-dict JSON
    bin_key = "corrupt_bin"
    (caches.judge_cache.root_dir / f"{bin_key}.json").write_bytes(b"\xff\xfe\x00\x01")
    assert caches.judge_cache.get(bin_key) is None
    assert not (caches.judge_cache.root_dir / f"{bin_key}.json").exists()

    for idx, payload in enumerate(("[]", "null", '"just-a-string"')):
        k = f"nondict_{idx}"
        (caches.judge_cache.root_dir / f"{k}.json").write_text(payload, encoding="utf-8")
        assert caches.judge_cache.get(k) is None
        assert not (caches.judge_cache.root_dir / f"{k}.json").exists()

    # RunStore.load and list_runs recover from binary bytes, non-dict JSON, and skip dotfiles
    bad_bin = tmp_path / "runs" / "bad_bin.json"
    bad_bin.write_bytes(b"\xff\xfe\x00\x01")
    assert run_store.load("bad_bin", recover_corrupt=True) is None
    assert not bad_bin.exists()

    bad_list = tmp_path / "runs" / "bad_list.json"
    bad_list.write_text("[]", encoding="utf-8")
    assert run_store.load("bad_list", recover_corrupt=True) is None
    assert not bad_list.exists()

    leftover_tmp = tmp_path / "runs" / ".tmp_aborted_123.json"
    leftover_tmp.write_text("{incomplete", encoding="utf-8")
    assert run_store.list_runs() == []
    # Dotfile must be skipped untouched (not quarantined)
    assert leftover_tmp.exists()
    assert list((tmp_path / "runs").glob(".tmp_aborted_123.corrupt.*.json")) == []


def test_cost_meter_rejects_non_finite_values() -> None:
    """Fix 3: NaN, inf, and -inf raise ValueError in cap_usd, reserve(), and settle()."""
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            CostMeter("r", cap_usd=bad)

    meter = CostMeter("r", cap_usd=10.0)
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            meter.reserve("shot", bad)

    res_id = meter.reserve("shot", 1.0)
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            meter.settle(res_id, bad)

    # Reservation must remain intact after rejected settle()
    entry = meter.settle(res_id, 0.75)
    assert entry.actual_usd == 0.75
    assert meter.spent_usd == 0.75


async def test_job_registry_rejects_duplicate_job_id() -> None:
    """Fix 4: submit() raises ValueError on duplicate job_id without overwriting active job."""
    registry = JobRegistry(max_concurrent=2)
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow_worker(_token: CancellationToken) -> str:
        started.set()
        await release.wait()
        return "done"

    dto = await registry.submit("run_1", slow_worker, job_id="dup_job_1")
    await asyncio.wait_for(started.wait(), timeout=2.0)

    with pytest.raises(ValueError, match="Duplicate job_id"):
        await registry.submit("run_2", slow_worker, job_id="dup_job_1")

    # Original job must still be tracked and complete cleanly
    live = registry.get_job(dto.id)
    assert live is not None and live.run_id == "run_1"
    release.set()
    final = await registry.wait_for_job(dto.id)
    assert final is not None and final.state == "done" and registry.active_count == 0


def test_fit_timings_simultaneous_min_max_clamping_water_filling() -> None:
    """Fix 6: bounded water-filling satisfies [min_shot, max_shot] and sum==target_sec."""
    cases = [
        ([0.01] * 10 + [100.0, 100.0], 20.0),
        ([1000.0, 1000.0] + [0.001] * 8, 18.0),
        ([100.0] * 10 + [0.01, 0.01], 85.0),
    ]
    for weights, target in cases:
        res = fit_timings(weights, target, min_shot=1.5, max_shot=8.0)
        assert len(res) == len(weights)
        assert all(1.5 - 1e-9 <= x <= 8.0 + 1e-9 for x in res), f"Out of bounds: {res}"
        assert math.isclose(sum(res), target, abs_tol=1e-6), f"Sum mismatch: {sum(res)} != {target}"


def test_claim_grounding_word_boundaries_and_standalone_guaranteed() -> None:
    """Fix 7: no scale-suffix false positives, catch standalone guaranteed, no substring bypass."""
    assert extract_claim_tokens("Ready in 5 minutes for 100 kids with 3 books.") == []

    empty = ResearchBundle(brand_url="https://acme.ai", brand_name="Acme")
    guaranteed_issues = check_claim_grounding("Faster onboarding, guaranteed.", empty)
    assert any(i.code == "UNGROUNDED_CLAIM" for i in guaranteed_issues)

    research = ResearchBundle(
        brand_url="https://acme.ai",
        brand_name="Acme",
        verified_claims=[
            "Costs $499/yr.",
            "Ranked #10.",
            "150% battery.",
            "Asbestos-free housing.",
        ],
    )
    for ungrounded_copy in (
        "Starts at $49!",
        "Voted #1!",
        "Save 50%!",
        "The best mug ever.",
    ):
        issues = check_claim_grounding(ungrounded_copy, research)
        assert issues != [], f"Expected blocking issue for {ungrounded_copy!r}"
        assert all(not i.overridable for i in issues)


def test_speakability_multipliers_subdomains_and_non_finite_duration() -> None:
    """Fix 8: 10x -> ten times, multi-label subdomains expand all dots, NaN/inf duration flagged."""
    from gtm_videogen.rules.speakability import count_text_syllables

    assert spoken_form("10x") == "ten times"
    assert count_text_syllables("10x") == count_text_syllables("ten times")
    assert count_text_syllables("10x") >= 2
    assert spoken_form("app.example.io") == "app dot example dot i o"
    assert spoken_form("shop.sub.acme.com") == "shop dot sub dot acme dot com"

    for bad_dur in (float("nan"), float("inf"), float("-inf"), 0.0, -2.0):
        problem = beat_fit_problem("Hello world", bad_dur)
        assert problem is not None, f"Expected problem for duration={bad_dur!r}"


def test_anti_garbled_text_catches_baked_in_text_without_false_positives() -> None:
    """Fix 9: catches dashboard screenshots, neon sign with words, quoted clothing text."""
    bad_prompts = [
        "Close-up of a dashboard screenshot with readable metrics and revenue charts.",
        "A moody alley with a glowing neon sign with words in the window.",
        'A runner wearing a cotton t-shirt with "RUN FASTER" across the chest.',
    ]
    for prompt in bad_prompts:
        assert rendered_text_problem(prompt) is not None, f"Missed baked-in text: {prompt!r}"

    clean_prompts = [
        "Wide tracking shot of a runner on a misty coastal trail at golden hour, 35mm lens.",
        "Close-up of hands pouring amber tonic over ice in a sunlit kitchen.",
    ]
    for prompt in clean_prompts:
        assert rendered_text_problem(prompt) is None, f"False positive on clean prompt: {prompt!r}"


def test_creative_history_recovers_from_corrupt_entries_and_null_values(tmp_path: Path) -> None:
    """Fix 10: schema-corrupt entries and null/non-list values recover cleanly."""
    store_path = tmp_path / "h.json"
    store_path.write_text(
        json.dumps({"acme.com": [{"bad_key": 123}], "nullbrand.com": None}),
        encoding="utf-8",
    )
    store = CreativeHistoryStore(store_path)

    assert store.get_history("https://acme.com").entries == []
    assert store.get_history("https://nullbrand.com").entries == []

    fp = CreativeFingerprint(
        run_id="r1",
        brand_url="https://nullbrand.com",
        arc_name="Minimalist Craft",
        hook_opening="Macro texture",
        visual_world="Studio",
        lighting_style="Soft",
        camera_style="Locked",
    )
    recorded = store.record(fp)
    assert len(recorded.entries) == 1
    assert recorded.entries[0].run_id == "r1"


def test_voice_casting_strictly_prefers_unused_voices() -> None:
    """Fix 11: unused voices always take precedence over recently used voices."""
    from gtm_videogen.rules.voice_casting import VOICE_CATALOG

    casting = VoiceCasting(
        gender="male",
        age="young",
        accent="us",
        energy="energetic",
        pace="brisk",
        delivery_style="punchy fast",
    )
    v1 = select_voice(casting)
    fp1 = CreativeFingerprint(
        run_id="r1",
        brand_url="https://acme.ai",
        arc_name="a",
        hook_opening="h",
        visual_world="w",
        lighting_style="l",
        camera_style="c",
        voice_id=v1.voice_id,
    )
    fp2 = fp1.model_copy(update={"run_id": "r2", "voice_id": "v_aria_f_adult_us"})
    v2 = select_voice(casting, [fp1, fp2])
    assert v2.voice_id not in {v1.voice_id, "v_aria_f_adult_us"}

    # Fallback when all catalog voices have been used still selects a valid voice
    all_used = [
        fp1.model_copy(update={"run_id": f"r_{i}", "voice_id": p.voice_id})
        for i, p in enumerate(VOICE_CATALOG)
    ]
    v_fallback = select_voice(casting, all_used)
    assert v_fallback.voice_id in {p.voice_id for p in VOICE_CATALOG}


async def test_http_client_strips_sensitive_headers_on_cross_origin_redirect() -> None:
    """M6 Challenge 2: Strip Authorization, Proxy-Authorization, Cookie on cross-origin 302."""
    seen_requests: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen_requests.append(req)
        if req.url == httpx.URL("https://api.example.com/cross"):
            return httpx.Response(302, headers={"Location": "https://evil.example.org/steal"})
        if req.url == httpx.URL("https://api.example.com/same"):
            return httpx.Response(302, headers={"Location": "https://api.example.com:443/final"})
        if req.url == httpx.URL("https://api.example.com/downgrade"):
            return httpx.Response(302, headers={"Location": "http://api.example.com/plain"})
        return httpx.Response(200, json={"ok": True})

    caller_headers = {
        "Authorization": "Bearer TOP_SECRET",
        "Proxy-Authorization": "Basic cHJveHk6c2VjcmV0",
        "Cookie": "sid=xyz123",
        "X-Trace-Id": "trace-42",
    }

    async with ResilientHttpClient(
        transport=httpx.MockTransport(handler),
        max_retries=0,
        resolve_dns_for_ssrf=False,
    ) as client:
        # 1. Cross-origin redirect MUST strip sensitive headers, keep benign headers
        seen_requests.clear()
        res_cross = await client.get_json(
            "https://api.example.com/cross",
            headers=caller_headers,
        )
        assert res_cross == {"ok": True}
        assert len(seen_requests) == 2
        second_headers = {k.lower(): v for k, v in seen_requests[1].headers.items()}
        assert "authorization" not in second_headers
        assert "proxy-authorization" not in second_headers
        assert "cookie" not in second_headers
        assert second_headers.get("x-trace-id") == "trace-42"
        # Caller dict must not be mutated
        assert caller_headers["Authorization"] == "Bearer TOP_SECRET"

        # 2. Same-origin redirect (including explicit default port 443) MUST preserve them
        seen_requests.clear()
        res_same = await client.get_json(
            "https://api.example.com/same",
            headers=caller_headers,
        )
        assert res_same == {"ok": True}
        assert len(seen_requests) == 2
        same_headers = {k.lower(): v for k, v in seen_requests[1].headers.items()}
        assert same_headers.get("authorization") == "Bearer TOP_SECRET"
        assert same_headers.get("proxy-authorization") == "Basic cHJveHk6c2VjcmV0"
        assert same_headers.get("cookie") == "sid=xyz123"

        # 3. Scheme downgrade (https -> http) changes origin and MUST strip sensitive headers
        seen_requests.clear()
        res_down = await client.get_json(
            "https://api.example.com/downgrade",
            headers=caller_headers,
        )
        assert res_down == {"ok": True}
        assert len(seen_requests) == 2
        down_headers = {k.lower(): v for k, v in seen_requests[1].headers.items()}
        assert "authorization" not in down_headers
        assert "proxy-authorization" not in down_headers
        assert "cookie" not in down_headers


def test_cache_store_rejects_path_traversal_keys(tmp_path: Path) -> None:
    """M6 Tier 5 Challenge 3: ContentCache rejects traversal keys without touching outside files."""
    caches = CacheStore(tmp_path / "caches")
    victim = tmp_path / "caches" / "victim.json"
    victim.write_text('{"protected": true}', encoding="utf-8")

    artifact = tmp_path / "clip.mp4"
    artifact.write_bytes(b"mp4-bytes")

    for bad_key in ("../victim", "a/b", "a\\b", "", ".", "..", "bad key!"):
        with pytest.raises(ValueError):
            caches.judge_cache.get(bad_key)
        with pytest.raises(ValueError):
            caches.judge_cache.put(bad_key, metadata={"pwned": True})
        with pytest.raises(ValueError):
            caches.judge_cache.invalidate(bad_key)
        with pytest.raises(ValueError):
            caches.clip_cache.put(bad_key, artifact_path=artifact, metadata={"pwned": True})

    # Outside victim file must remain completely untouched
    assert victim.exists()
    assert json.loads(victim.read_text(encoding="utf-8")) == {"protected": True}

