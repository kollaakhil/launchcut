"""Comprehensive FastAPI SSE server, security middleware, and Typer CLI tests (Milestone M5)."""

from __future__ import annotations

import asyncio
import inspect
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

from gtm_videogen.api.app import create_app
from gtm_videogen.api.middleware import TTLRateLimiter, validate_origin
from gtm_videogen.api.routes import PipelineRunnerProtocol, make_sse_event
from gtm_videogen.cli.main import app as cli_app
from gtm_videogen.core.job_registry import CancellationToken, JobRegistry
from gtm_videogen.core.media_store import MediaStore
from gtm_videogen.core.run_store import RunStore
from gtm_videogen.models.schemas import (
    CostEntry,
    CostLedger,
    QAReport,
    RenderManifest,
    RunInput,
    RunState,
)


def _sample_run_input(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "brand_name": "Kestrel Roasters",
        "brand_url": "https://kestrelroasters.example.com",
        "goal": "Drive specialty coffee subscription trials",
        "offer": "Save 25% on your first two bags",
        "facts": ["Roasted to order in Seattle every Tuesday"],
        "target_duration_sec": 12.0,
        "aspect": "16:9",
        "fps": 24,
        "audio_mode": "both",
        "cap_usd": 5.0,
        "mock": True,
    }
    data.update(overrides)
    return data


class _StubPipelineRunner:
    """Protocol-compliant async pipeline runner for isolated M5 API/CLI testing."""

    def __init__(
        self,
        *,
        work_dir: Path,
        hold_event: asyncio.Event | None = None,
        entered_event: asyncio.Event | None = None,
        qa_passed: bool = True,
        raise_error: str | None = None,
    ) -> None:
        self.work_dir = work_dir
        self.hold_event = hold_event
        self.entered_event = entered_event
        self.qa_passed = qa_passed
        self.raise_error = raise_error
        self.repair_calls: list[list[str] | None] = []

    async def _emit(self, event_cb: Any, event: dict[str, Any]) -> None:
        if event_cb is None:
            return
        res = event_cb(event)
        if inspect.isawaitable(res):
            await res

    async def run(
        self,
        run_state: RunState,
        *,
        event_cb: Any = None,
        cancel: CancellationToken | None = None,
    ) -> RunState:
        if self.entered_event is not None:
            self.entered_event.set()

        await self._emit(
            event_cb,
            make_sse_event("stage_started", run_state.run_id, stage="agent1_research"),
        )
        await self._emit(
            event_cb,
            make_sse_event(
                "rubric_evaluation",
                run_state.run_id,
                stage="agent2_director",
                data={"iteration": 1, "status": "satisfied"},
            ),
        )
        await self._emit(
            event_cb,
            make_sse_event(
                "shot_progress",
                run_state.run_id,
                stage="agent3_producer",
                data={"shot_id": "s1", "pct": 100},
            ),
        )
        await self._emit(
            event_cb,
            make_sse_event(
                "cost_update",
                run_state.run_id,
                stage="agent3_producer",
                data={"actual_usd": 0.12, "cap_usd": run_state.input.cap_usd},
            ),
        )
        await self._emit(
            event_cb,
            make_sse_event(
                "warning",
                run_state.run_id,
                stage="agent3_producer",
                data={"message": "Non-fatal SFX fallback used"},
            ),
        )

        if self.hold_event is not None:
            if cancel is not None:
                hold_task = asyncio.create_task(self.hold_event.wait())
                cancel_task = asyncio.create_task(cancel.wait())
                done, pending = await asyncio.wait(
                    {hold_task, cancel_task}, return_when=asyncio.FIRST_COMPLETED
                )
                for p in pending:
                    p.cancel()
                if cancel.is_cancelled:
                    cancel.raise_if_cancelled()
            else:
                await self.hold_event.wait()

        if cancel is not None:
            cancel.raise_if_cancelled()

        if self.raise_error:
            raise RuntimeError(self.raise_error)

        out_mp4 = self.work_dir / f"{run_state.run_id}_final.mp4"
        out_mp4.parent.mkdir(parents=True, exist_ok=True)
        out_mp4.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"0" * 512)

        run_state.render_manifest = RenderManifest(
            run_id=run_state.run_id,
            stitched_path=str(out_mp4),
            warnings=["Non-fatal SFX fallback used"],
        )
        run_state.cost_ledger.actual_usd = 0.12
        run_state.cost_ledger.estimated_usd = 0.15
        run_state.cost_ledger.entries.append(
            CostEntry(
                stage="render",
                provider="mock",
                model="mock-video",
                what="shot_s1",
                estimated_usd=0.15,
                actual_usd=0.12,
                cache_hit=False,
            )
        )
        run_state.qa_report = QAReport(
            passed=self.qa_passed,
            originality_score=8.5,
            bible_coherence_score=9.0,
            loudness_lufs=-14.2,
        )
        run_state.status = "done"
        await self._emit(
            event_cb,
            make_sse_event("stage_completed", run_state.run_id, stage="agent3_producer"),
        )
        await self._emit(
            event_cb,
            make_sse_event(
                "run_completed",
                run_state.run_id,
                stage="pipeline",
                data={"stitched_path": str(out_mp4)},
            ),
        )
        return run_state

    async def repair(
        self,
        run_state: RunState,
        *,
        failed_shot_ids: list[str] | None = None,
        event_cb: Any = None,
        cancel: CancellationToken | None = None,
    ) -> RunState:
        self.repair_calls.append(failed_shot_ids)
        if cancel is not None:
            cancel.raise_if_cancelled()
        run_state.repair_result = {
            "repaired_shot_ids": list(failed_shot_ids or []),
            "status": "repaired",
        }
        run_state.status = "done"
        await self._emit(
            event_cb,
            make_sse_event(
                "run_completed",
                run_state.run_id,
                stage="repair",
                data={"repair_result": run_state.repair_result},
            ),
        )
        return run_state


def test_stub_satisfies_pipeline_runner_protocol(tmp_path: Path) -> None:
    runner = _StubPipelineRunner(work_dir=tmp_path)
    assert isinstance(runner, PipelineRunnerProtocol)


async def test_health_endpoint_reports_live_tools(tmp_path: Path) -> None:
    app = create_app(data_dir=tmp_path)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        resp = await client.get("/api/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] in ("ok", "degraded")
        assert isinstance(body["tools"]["ffmpeg"], bool)
        assert isinstance(body["tools"]["ffprobe"], bool)
        assert isinstance(body["tools"]["tesseract"], bool)
        assert isinstance(body["tools"]["manim"], bool)
        assert "manim_version" in body["manim"]


async def test_runs_crud_and_bug03_extra_forbid(tmp_path: Path) -> None:
    app = create_app(data_dir=tmp_path)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        # BUG-03: unknown field must fail fast with 422
        bad = await client.post(
            "/api/runs",
            json={**_sample_run_input(), "unknown_extra_field": "nope"},
        )
        assert bad.status_code == 422

        # Valid creation with audio_mode="both"
        created = await client.post(
            "/api/runs",
            json=_sample_run_input(audio_mode="both"),
            headers={"X-Actor-Id": "alice"},
        )
        assert created.status_code == 201
        run_data = created.json()
        run_id = run_data["run_id"]
        assert run_data["created_by"] == "alice"
        assert run_data["input"]["audio_mode"] == "both"
        assert run_data["status"] == "idle"

        # List runs
        listed = await client.get("/api/runs")
        assert listed.status_code == 200
        assert any(r["run_id"] == run_id for r in listed.json()["runs"])

        # Get single run
        fetched = await client.get(f"/api/runs/{run_id}")
        assert fetched.status_code == 200
        assert fetched.json()["run_id"] == run_id

        # Missing run -> 404
        missing = await client.get("/api/runs/does_not_exist")
        assert missing.status_code == 404


async def test_bug20_origin_middleware_403_never_500_and_preflight(tmp_path: Path) -> None:
    app = create_app(data_dir=tmp_path)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        for bad_origin in (
            "null",
            "NULL",
            "not-a-valid-origin://[bad",
            "http://evil.example.org",
            "ftp://localhost:3000",
            "http://localhost:99999",
            "http://user:pass@localhost:3000",
            "http://localhost:3000/path",
            "http://local host:3000",
        ):
            resp = await client.post(
                "/api/runs",
                json=_sample_run_input(),
                headers={"Origin": bad_origin},
            )
            assert resp.status_code == 403, (
                f"Expected 403 for {bad_origin!r}, got {resp.status_code}"
            )
            assert "detail" in resp.json()

        # Valid local origin succeeds with CORS headers
        ok = await client.get("/api/health", headers={"Origin": "http://localhost:3000"})
        assert ok.status_code == 200
        assert ok.headers["access-control-allow-origin"] == "http://localhost:3000"
        assert ok.headers["vary"] == "Origin"

        # OPTIONS preflight returns 204
        preflight = await client.options(
            "/api/runs",
            headers={"Origin": "http://localhost:3000"},
        )
        assert preflight.status_code == 204
        assert preflight.headers["access-control-allow-origin"] == "http://localhost:3000"


def test_validate_origin_custom_allowlist() -> None:
    ok, _, _ = validate_origin(
        "https://studio.example.com",
        allowed_origins=["https://studio.example.com"],
    )
    assert ok is True
    denied, _, _ = validate_origin(
        "https://other.example.com",
        allowed_origins=["https://studio.example.com"],
    )
    assert denied is False
    # Even with wildcard, Origin: null is rejected
    null_ok, _, _ = validate_origin("null", allowed_origins=["*"])
    assert null_ok is False


async def test_bug13_ownership_enforced_on_mutating_routes(tmp_path: Path) -> None:
    runner = _StubPipelineRunner(work_dir=tmp_path)
    app = create_app(data_dir=tmp_path, runner=runner)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        created = await client.post(
            "/api/runs",
            json=_sample_run_input(),
            headers={"X-Actor-Id": "alice"},
        )
        run_id = created.json()["run_id"]

        for endpoint in ("start", "cancel", "repair"):
            resp = await client.post(
                f"/api/runs/{run_id}/{endpoint}",
                headers={"X-Actor-Id": "bob"},
            )
            assert resp.status_code == 403

        # Missing run -> 404
        not_found = await client.post(
            "/api/runs/missing_id/cancel",
            headers={"X-Actor-Id": "alice"},
        )
        assert not_found.status_code == 404


async def test_sse_streaming_full_lifecycle_and_queue_cleanup(tmp_path: Path) -> None:
    runner = _StubPipelineRunner(work_dir=tmp_path)
    app = create_app(data_dir=tmp_path, runner=runner)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        created = await client.post("/api/runs", json=_sample_run_input())
        run_id = created.json()["run_id"]

        started = await client.post(f"/api/runs/{run_id}/start")
        assert started.status_code == 202
        assert started.json()["run_id"] == run_id

        events: list[dict[str, Any]] = []
        event_names: list[str] = []
        async with client.stream("GET", f"/api/runs/{run_id}/events") as sse:
            assert sse.status_code == 200
            assert "text/event-stream" in sse.headers["content-type"]
            async for line in sse.aiter_lines():
                if line.startswith("event:"):
                    event_names.append(line.split(":", 1)[1].strip())
                elif line.startswith("data:"):
                    payload = json.loads(line.split(":", 1)[1].strip())
                    events.append(payload)
                    if payload["type"] == "run_completed":
                        break

        seen_types = {e["type"] for e in events}
        assert {
            "stage_started",
            "rubric_evaluation",
            "shot_progress",
            "cost_update",
            "warning",
            "stage_completed",
            "run_completed",
        }.issubset(seen_types)
        assert "run_completed" in event_names
        for ev in events:
            assert ev["run_id"] == run_id
            assert isinstance(ev["timestamp"], str) and ev["timestamp"]
            assert isinstance(ev["data"], dict)

        # Subscriber queue must be cleaned up on stream termination
        assert app.state.services.event_broker.subscriber_count(run_id) == 0

        final_run = (await client.get(f"/api/runs/{run_id}")).json()
        assert final_run["status"] == "done"
        assert final_run["qa_report"]["passed"] is True


async def test_bug05_cancel_endpoint_and_rerun(tmp_path: Path) -> None:
    entered = asyncio.Event()
    hold = asyncio.Event()
    runner = _StubPipelineRunner(work_dir=tmp_path, hold_event=hold, entered_event=entered)
    app = create_app(data_dir=tmp_path, runner=runner)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        created = await client.post(
            "/api/runs",
            json=_sample_run_input(),
            headers={"X-Actor-Id": "alice"},
        )
        run_id = created.json()["run_id"]

        started = await client.post(
            f"/api/runs/{run_id}/start",
            headers={"X-Actor-Id": "alice"},
        )
        assert started.status_code == 202
        job_id = started.json()["job"]["id"]

        await asyncio.wait_for(entered.wait(), timeout=2.0)

        cancelled = await client.post(
            f"/api/runs/{run_id}/cancel",
            headers={"X-Actor-Id": "alice"},
        )
        assert cancelled.status_code == 200
        assert cancelled.json()["status"] == "cancelled"
        assert cancelled.json()["cancelled_jobs"] >= 1

        job_dto = await app.state.services.job_registry.wait_for_job(job_id)
        assert job_dto is not None and job_dto.state == "cancelled"

        state = (await client.get(f"/api/runs/{run_id}")).json()
        assert state["status"] == "cancelled"

        # SSE stream replays run_cancelled and terminates cleanly
        seen: list[str] = []
        async with client.stream("GET", f"/api/runs/{run_id}/events") as sse:
            async for line in sse.aiter_lines():
                if line.startswith("data:"):
                    ev = json.loads(line.split(":", 1)[1].strip())
                    seen.append(ev["type"])
                    if ev["type"] == "run_cancelled":
                        break
        assert "run_cancelled" in seen

        # Subsequent re-start succeeds cleanly (no stuck 'already working' 409)
        hold.set()
        restarted = await client.post(
            f"/api/runs/{run_id}/start",
            headers={"X-Actor-Id": "alice"},
        )
        assert restarted.status_code == 202
        await app.state.services.job_registry.wait_for_job(restarted.json()["job"]["id"])
        assert (await client.get(f"/api/runs/{run_id}")).json()["status"] == "done"


async def test_bug06_async_repair_persists_repair_result(tmp_path: Path) -> None:
    runner = _StubPipelineRunner(work_dir=tmp_path)
    app = create_app(data_dir=tmp_path, runner=runner)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        created = await client.post("/api/runs", json=_sample_run_input())
        run_id = created.json()["run_id"]

        # Extra field rejected with 422
        bad_repair = await client.post(
            f"/api/runs/{run_id}/repair",
            json={"failed_shot_ids": ["s1"], "bogus": True},
        )
        assert bad_repair.status_code == 422

        repair_resp = await client.post(
            f"/api/runs/{run_id}/repair",
            json={"failed_shot_ids": ["s1", "s3"]},
        )
        assert repair_resp.status_code == 202
        job_id = repair_resp.json()["job"]["id"]
        await app.state.services.job_registry.wait_for_job(job_id)

        state = (await client.get(f"/api/runs/{run_id}")).json()
        assert state["status"] == "done"
        assert state["repair_result"] == {
            "repaired_shot_ids": ["s1", "s3"],
            "status": "repaired",
        }
        assert runner.repair_calls == [["s1", "s3"]]


async def test_bug02_bug22_jobs_endpoint_serializable_and_other_active_count(
    tmp_path: Path,
) -> None:
    registry = JobRegistry(max_concurrent=4)
    started = asyncio.Event()
    release = asyncio.Event()

    async def _worker(tok: CancellationToken) -> None:
        started.set()
        await release.wait()

    job = await registry.submit("run_other", _worker, stage="render", shot_id="s1")
    await asyncio.wait_for(started.wait(), timeout=2.0)

    app = create_app(data_dir=tmp_path, job_registry=registry)
    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            all_resp = await client.get("/api/jobs")
            assert all_resp.status_code == 200
            all_body = all_resp.json()
            assert len(all_body["jobs"]) == 1
            assert all_body["jobs"][0]["state"] == "running"

            scoped = await client.get("/api/jobs", params={"run_id": "run_mine"})
            assert scoped.status_code == 200
            scoped_body = scoped.json()
            assert scoped_body["jobs"] == []
            assert scoped_body["other_active_jobs_count"] == 1
    finally:
        release.set()
        await registry.wait_for_job(job.id)


async def test_bug19_orphan_working_run_heals_on_get(tmp_path: Path) -> None:
    registry = JobRegistry(max_concurrent=2)
    run_store = RunStore(tmp_path / "runs", job_registry=registry)
    orphan = RunState(
        run_id="run_orphan_api",
        created_by="local",
        status="working",
        input=RunInput(**_sample_run_input()),
        cost_ledger=CostLedger(run_id="run_orphan_api", cap_usd=5.0),
    )
    run_store.save(orphan)

    app = create_app(data_dir=tmp_path, run_store=run_store, job_registry=registry)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        resp = await client.get("/api/runs/run_orphan_api")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "interrupted"
        assert len(body["warnings"]) >= 1


async def test_media_range_206_416_and_traversal_protection(tmp_path: Path) -> None:
    media_store = MediaStore(tmp_path / "media", resolve_dns_for_ssrf=False)
    sample = bytes(range(256)) * 8  # 2048 bytes
    media_id, _ = media_store.save_bytes(sample, ext=".mp4")

    app = create_app(data_dir=tmp_path, media_store=media_store)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        # Full 200 response
        full = await client.get(f"/api/media/{media_id}")
        assert full.status_code == 200
        assert full.content == sample
        assert full.headers["accept-ranges"] == "bytes"
        assert full.headers["content-length"] == str(len(sample))

        # Standard 206 byte range
        r1 = await client.get(f"/api/media/{media_id}", headers={"Range": "bytes=0-63"})
        assert r1.status_code == 206
        assert r1.content == sample[:64]
        assert r1.headers["content-range"] == f"bytes 0-63/{len(sample)}"
        assert r1.headers["content-length"] == "64"

        # Open-ended range
        r2 = await client.get(f"/api/media/{media_id}", headers={"Range": "bytes=2000-"})
        assert r2.status_code == 206
        assert r2.content == sample[2000:]
        assert r2.headers["content-range"] == f"bytes 2000-2047/{len(sample)}"

        # Suffix range
        r3 = await client.get(f"/api/media/{media_id}", headers={"Range": "bytes=-32"})
        assert r3.status_code == 206
        assert r3.content == sample[-32:]
        assert r3.headers["content-range"] == f"bytes 2016-2047/{len(sample)}"

        # Out-of-range -> 416
        r416 = await client.get(f"/api/media/{media_id}", headers={"Range": "bytes=9999-10000"})
        assert r416.status_code == 416
        assert r416.headers["content-range"] == f"bytes */{len(sample)}"

        # Reversed range -> 416
        r_rev = await client.get(f"/api/media/{media_id}", headers={"Range": "bytes=100-50"})
        assert r_rev.status_code == 416

        # Path traversal blocked
        trav = await client.get("/api/media/..%2F..%2Fetc%2Fpasswd")
        assert trav.status_code in (400, 404)

        # Missing file -> 404
        missing = await client.get("/api/media/nonexistent-id.mp4")
        assert missing.status_code == 404


async def test_ttl_rate_limiter_pruning_and_429_middleware(tmp_path: Path) -> None:
    now = [100.0]
    limiter = TTLRateLimiter(
        max_requests=2,
        window_sec=10.0,
        max_keys=3,
        clock=lambda: now[0],
    )
    assert limiter.check("k1") == (True, 0)
    assert limiter.check("k1") == (True, 0)
    allowed, retry_after = limiter.check("k1")
    assert allowed is False
    assert retry_after >= 1

    # Advance past TTL -> old key pruned and memory bounded
    now[0] = 115.0
    assert limiter.check("k2") == (True, 0)
    assert limiter.active_keys == 1

    # Max keys bound strictly enforced (BUG-C1-03)
    for idx in range(10):
        limiter.check(f"burst_{idx}")
        assert limiter.active_keys <= 3

    # Middleware returns 429 + Retry-After
    app = create_app(data_dir=tmp_path, rate_limit_requests=2, rate_limit_window_sec=60.0)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        assert (await client.get("/api/health", headers={"X-Actor-Id": "u1"})).status_code == 200
        assert (await client.get("/api/health", headers={"X-Actor-Id": "u1"})).status_code == 200
        throttled = await client.get("/api/health", headers={"X-Actor-Id": "u1"})
        assert throttled.status_code == 429
        assert int(throttled.headers["retry-after"]) >= 1


def test_cli_doctor_reports_tools_manim_fonts_and_keys() -> None:
    res = CliRunner().invoke(cli_app, ["doctor"])
    assert res.exit_code == 0, res.output
    lower = res.output.lower()
    assert "ffmpeg" in lower
    assert "ffprobe" in lower
    assert "tesseract" in lower
    assert "manim" in lower
    assert "font" in lower
    assert "offline mode" in lower


def test_cli_run_mock_and_qa_failure_exit_codes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import gtm_videogen.cli.main as cli_mod

    # Passing run -> exit 0, prints MP4, QAReport, CostLedger, writes run_*.json
    monkeypatch.setattr(
        cli_mod,
        "_build_pipeline_runner",
        lambda *, run_state, work_dir: _StubPipelineRunner(work_dir=work_dir, qa_passed=True),
    )
    out_ok = tmp_path / "ok_run"
    res_ok = CliRunner().invoke(
        cli_app,
        [
            "run",
            "--brand",
            "Kestrel Roasters",
            "--url",
            "https://kestrelroasters.example.com",
            "--goal",
            "Drive coffee subscription trials",
            "--duration",
            "12",
            "--aspect",
            "16:9",
            "--cap-usd",
            "5.0",
            "--output-dir",
            str(out_ok),
            "--mock",
        ],
    )
    assert res_ok.exit_code == 0, res_ok.output
    assert "QAReport" in res_ok.output
    assert "CostLedger" in res_ok.output
    assert list(out_ok.glob("*.mp4"))
    json_files = list(out_ok.glob("run_*.json"))
    assert len(json_files) == 1
    parsed = json.loads(json_files[0].read_text(encoding="utf-8"))
    assert parsed["qa_report"]["passed"] is True
    assert parsed["cost_ledger"]["actual_usd"] == pytest.approx(0.12)

    # Failing QA -> exit 1
    monkeypatch.setattr(
        cli_mod,
        "_build_pipeline_runner",
        lambda *, run_state, work_dir: _StubPipelineRunner(work_dir=work_dir, qa_passed=False),
    )
    out_fail = tmp_path / "fail_run"
    res_fail = CliRunner().invoke(
        cli_app,
        [
            "run",
            "--brand-name",
            "Kestrel Roasters",
            "--brand-url",
            "https://kestrelroasters.example.com",
            "--goal",
            "Drive coffee subscription trials",
            "--output-dir",
            str(out_fail),
            "--mock",
        ],
    )
    assert res_fail.exit_code == 1


def test_cli_serve_invokes_uvicorn(monkeypatch: pytest.MonkeyPatch) -> None:
    import gtm_videogen.cli.main as cli_mod

    captured: dict[str, Any] = {}

    def _fake_uvicorn_run(target: str, *, host: str, port: int, reload: bool) -> None:
        captured["target"] = target
        captured["host"] = host
        captured["port"] = port
        captured["reload"] = reload

    monkeypatch.setattr(cli_mod.uvicorn, "run", _fake_uvicorn_run)
    res = CliRunner().invoke(cli_app, ["serve", "--host", "127.0.0.1", "--port", "9123"])
    assert res.exit_code == 0, res.output
    assert captured == {
        "target": "gtm_videogen.api.app:app",
        "host": "127.0.0.1",
        "port": 9123,
        "reload": False,
    }


async def test_bug_c1_01_overlong_run_id_never_500(tmp_path: Path) -> None:
    app = create_app(data_dir=tmp_path)
    transport = httpx.ASGITransport(app=app)
    overlong_id = "a" * 500
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        responses = [
            await client.get(f"/api/runs/{overlong_id}"),
            await client.post(f"/api/runs/{overlong_id}/start"),
            await client.post(f"/api/runs/{overlong_id}/cancel"),
            await client.post(f"/api/runs/{overlong_id}/repair"),
            await client.get(f"/api/runs/{overlong_id}/events"),
        ]
        for resp in responses:
            assert resp.status_code in (400, 404), f"Unexpected status {resp.status_code}"
            assert resp.status_code != 500


async def test_bug_c1_02_overlong_media_id_and_symlink_no_path_leak(tmp_path: Path) -> None:
    media_root = tmp_path / "media"
    media_store = MediaStore(media_root, resolve_dns_for_ssrf=False)

    # Create a secret file outside MediaStore root and a symlink inside pointing to it
    secret_file = tmp_path / "secret.txt"
    secret_file.write_text("top-secret-data", encoding="utf-8")
    escape_link = media_root / "escape_link.txt"
    escape_link.symlink_to(secret_file)

    app = create_app(data_dir=tmp_path, media_store=media_store)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        # Overlong ASCII segment -> 400 (never 500)
        long_resp = await client.get(f"/api/media/{'a' * 500}.bin")
        assert long_resp.status_code in (400, 404)
        assert long_resp.status_code != 500

        # Overlong UTF-8 byte-length segment (>255 bytes) -> 400 (never 500)
        utf8_resp = await client.get(f"/api/media/{'é' * 140}.bin")
        assert utf8_resp.status_code in (400, 404)
        assert utf8_resp.status_code != 500

        # Symlink escape -> 400 with generic detail and zero filesystem path disclosure
        sym_resp = await client.get("/api/media/escape_link.txt")
        assert sym_resp.status_code == 400
        assert sym_resp.json() == {"detail": "Unsafe media_id"}
        assert str(tmp_path) not in sym_resp.text
        assert "secret.txt" not in sym_resp.text


async def test_bug_c1_03_rate_limiter_strict_max_keys_bound(tmp_path: Path) -> None:
    now = [100.0]
    limiter = TTLRateLimiter(
        max_requests=2,
        window_sec=10.0,
        max_keys=5,
        clock=lambda: now[0],
    )
    for idx in range(20):
        limiter.check(f"key_{idx}")
        assert limiter.active_keys <= 5

    # Boundary case: max_keys=1
    tiny = TTLRateLimiter(max_requests=2, window_sec=10.0, max_keys=1, clock=lambda: now[0])
    for idx in range(5):
        tiny.check(f"single_{idx}")
        assert tiny.active_keys <= 1

    # HTTP middleware enforcement
    app = create_app(
        data_dir=tmp_path,
        rate_limit_requests=10,
        rate_limit_window_sec=60.0,
        rate_limit_max_keys=3,
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        for idx in range(10):
            resp = await client.get("/api/health", headers={"X-Actor-Id": f"actor_{idx}"})
            assert resp.status_code == 200
            assert app.state.rate_limiter.active_keys <= 3


async def test_origin_trailing_slash_rejected_rfc6454(tmp_path: Path) -> None:
    valid, _, _ = validate_origin("http://localhost:3000/")
    assert valid is False

    app = create_app(data_dir=tmp_path)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        get_resp = await client.get("/api/health", headers={"Origin": "http://localhost:3000/"})
        assert get_resp.status_code == 403

        opt_resp = await client.options("/api/health", headers={"Origin": "http://localhost:3000/"})
        assert opt_resp.status_code == 403


async def test_repair_job_non_cancelled_exception_during_cancellation_preserves_cancelled(
    tmp_path: Path,
) -> None:
    entered = asyncio.Event()

    class _AbortingRepairRunner(_StubPipelineRunner):
        async def repair(
            self,
            run_state: RunState,
            *,
            failed_shot_ids: list[str] | None = None,
            event_cb: Any = None,
            cancel: CancellationToken | None = None,
        ) -> RunState:
            entered.set()
            if cancel is not None:
                await cancel.wait()
            # Simulate a downstream library raising a generic exception on abort
            raise RuntimeError("transport closed due to cancellation")

    runner = _AbortingRepairRunner(work_dir=tmp_path)
    app = create_app(data_dir=tmp_path, runner=runner)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        created = await client.post(
            "/api/runs",
            json=_sample_run_input(),
            headers={"X-Actor-Id": "alice"},
        )
        run_id = created.json()["run_id"]

        repair_resp = await client.post(
            f"/api/runs/{run_id}/repair",
            json={"failed_shot_ids": ["s1"]},
            headers={"X-Actor-Id": "alice"},
        )
        assert repair_resp.status_code == 202
        job_id = repair_resp.json()["job"]["id"]

        await asyncio.wait_for(entered.wait(), timeout=2.0)

        cancel_resp = await client.post(
            f"/api/runs/{run_id}/cancel",
            headers={"X-Actor-Id": "alice"},
        )
        assert cancel_resp.status_code == 200
        await app.state.services.job_registry.wait_for_job(job_id)

        state = (await client.get(f"/api/runs/{run_id}")).json()
        assert state["status"] == "cancelled"

        seen: list[str] = []
        async with client.stream("GET", f"/api/runs/{run_id}/events") as sse:
            async for line in sse.aiter_lines():
                if line.startswith("data:"):
                    ev = json.loads(line.split(":", 1)[1].strip())
                    seen.append(ev["type"])
                    if ev["type"] in ("run_cancelled", "run_failed", "run_completed"):
                        break
        assert "run_cancelled" in seen
        assert "run_failed" not in seen


async def test_bug_c2_01_cancel_then_immediate_restart_does_not_clobber_new_job(
    tmp_path: Path,
) -> None:
    """Regression test for BUG-C2-01: slow Job 1 teardown after cancel must not clobber Job 2."""
    job1_entered = asyncio.Event()
    job1_in_cleanup = asyncio.Event()
    job1_release_cleanup = asyncio.Event()
    job2_entered = asyncio.Event()
    job2_release = asyncio.Event()
    call_count = 0

    class _SlowTeardownRunner(_StubPipelineRunner):
        async def run(
            self,
            run_state: RunState,
            *,
            event_cb: Any = None,
            cancel: CancellationToken | None = None,
        ) -> RunState:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                job1_entered.set()
                try:
                    if cancel is not None:
                        await cancel.wait()
                        cancel.raise_if_cancelled()
                except asyncio.CancelledError:
                    job1_in_cleanup.set()
                    # Simulate slow async teardown (e.g., terminating FFmpeg / HTTP cleanup)
                    await job1_release_cleanup.wait()
                    await self._emit(
                        event_cb,
                        make_sse_event(
                            "warning",
                            run_state.run_id,
                            stage="pipeline",
                            data={"message": "stale-job1-event"},
                        ),
                    )
                    raise
                return run_state

            # Job 2
            job2_entered.set()
            await self._emit(
                event_cb,
                make_sse_event(
                    "shot_progress",
                    run_state.run_id,
                    stage="agent3_producer",
                    data={"shot_id": "s1", "pct": 50},
                ),
            )
            await job2_release.wait()
            run_state.status = "done"
            await self._emit(
                event_cb,
                make_sse_event(
                    "run_completed",
                    run_state.run_id,
                    stage="pipeline",
                    data={"status": "done"},
                ),
            )
            return run_state

    runner = _SlowTeardownRunner(work_dir=tmp_path)
    app = create_app(data_dir=tmp_path, runner=runner)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        created = await client.post(
            "/api/runs",
            json=_sample_run_input(),
            headers={"X-Actor-Id": "alice"},
        )
        assert created.status_code == 201
        run_id = created.json()["run_id"]

        # Start Job 1 and wait until it is executing
        start1 = await client.post(
            f"/api/runs/{run_id}/start",
            headers={"X-Actor-Id": "alice"},
        )
        assert start1.status_code == 202
        job1_id = start1.json()["job"]["id"]
        await asyncio.wait_for(job1_entered.wait(), timeout=2.0)

        # Cancel Job 1 -> Job 1 enters slow async cleanup
        cancel_resp = await client.post(
            f"/api/runs/{run_id}/cancel",
            headers={"X-Actor-Id": "alice"},
        )
        assert cancel_resp.status_code == 200
        await asyncio.wait_for(job1_in_cleanup.wait(), timeout=2.0)

        # Immediately re-start Job 2 while Job 1 is still tearing down
        start2 = await client.post(
            f"/api/runs/{run_id}/start",
            headers={"X-Actor-Id": "alice"},
        )
        assert start2.status_code == 202
        job2_id = start2.json()["job"]["id"]
        await asyncio.wait_for(job2_entered.wait(), timeout=2.0)

        # Allow Job 1 to finish its teardown while Job 2 is still running
        job1_release_cleanup.set()
        job1_dto = await app.state.services.job_registry.wait_for_job(job1_id)
        assert job1_dto is not None and job1_dto.state == "cancelled"

        # Mid-flight invariant: Job 2 must still be "working" with NO stale terminal event
        mid_state = (await client.get(f"/api/runs/{run_id}")).json()
        assert mid_state["status"] == "working"
        assert app.state.services.event_broker.has_terminal_event(run_id) is False

        # Let Job 2 complete
        job2_release.set()
        job2_dto = await app.state.services.job_registry.wait_for_job(job2_id)
        assert job2_dto is not None and job2_dto.state == "done"

        # Final persisted RunState must be "done"
        final_state = (await client.get(f"/api/runs/{run_id}")).json()
        assert final_state["status"] == "done"

        # SSE stream must deliver run_completed and NEVER contain stale run_cancelled / job1 event
        events: list[dict[str, Any]] = []
        async with client.stream("GET", f"/api/runs/{run_id}/events") as sse:
            assert sse.status_code == 200
            async for line in sse.aiter_lines():
                if line.startswith("data:"):
                    ev = json.loads(line.split(":", 1)[1].strip())
                    events.append(ev)
                    if ev["type"] in ("run_completed", "run_cancelled", "run_failed"):
                        break

        event_types = [ev["type"] for ev in events]
        assert "run_completed" in event_types
        assert "run_cancelled" not in event_types
        assert all(ev.get("data", {}).get("message") != "stale-job1-event" for ev in events)


async def test_concurrent_start_repair_start_strictly_single_202_and_no_lock_leak(
    tmp_path: Path,
) -> None:
    """Regression test: concurrent [POST /start, POST /repair, POST /start] yields 1x202 + 2x409."""

    class _YieldingRunner(_StubPipelineRunner):
        async def run(
            self,
            run_state: RunState,
            *,
            event_cb: Any = None,
            cancel: CancellationToken | None = None,
        ) -> RunState:
            await asyncio.sleep(0.01)
            return await super().run(run_state, event_cb=event_cb, cancel=cancel)

        async def repair(
            self,
            run_state: RunState,
            *,
            failed_shot_ids: list[str] | None = None,
            event_cb: Any = None,
            cancel: CancellationToken | None = None,
        ) -> RunState:
            await asyncio.sleep(0.01)
            return await super().repair(
                run_state,
                failed_shot_ids=failed_shot_ids,
                event_cb=event_cb,
                cancel=cancel,
            )

    runner = _YieldingRunner(work_dir=tmp_path)
    app = create_app(data_dir=tmp_path, runner=runner, rate_limit_requests=0)
    transport = httpx.ASGITransport(app=app)
    headers = {"X-Actor-Id": "alice"}

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        # Unknown run_ids must NOT allocate entries in services._run_locks
        for missing_id in ("missing_a", "missing_b", "missing_c"):
            for endpoint in ("start", "repair", "cancel"):
                resp = await client.post(f"/api/runs/{missing_id}/{endpoint}", headers=headers)
                assert resp.status_code == 404
        assert app.state.services._run_locks == {}

        created = await client.post("/api/runs", json=_sample_run_input(), headers=headers)
        assert created.status_code == 201
        run_id = created.json()["run_id"]

        for iteration in range(20):
            r_start1, r_repair, r_start2 = await asyncio.gather(
                client.post(f"/api/runs/{run_id}/start", headers=headers),
                client.post(
                    f"/api/runs/{run_id}/repair",
                    json={"failed_shot_ids": [f"s_{iteration}"]},
                    headers=headers,
                ),
                client.post(f"/api/runs/{run_id}/start", headers=headers),
            )
            responses = (r_start1, r_repair, r_start2)
            codes = [r.status_code for r in responses]
            assert codes.count(202) == 1, f"iter {iteration}: expected 1x202, got {codes}"
            assert codes.count(409) == 2, f"iter {iteration}: expected 2x409, got {codes}"

            accepted = next(r for r in responses if r.status_code == 202)
            job_id = accepted.json()["job"]["id"]
            job_dto = await app.state.services.job_registry.wait_for_job(job_id)
            assert job_dto is not None and job_dto.state == "done"

        assert list(app.state.services._run_locks.keys()) == [run_id]
        final_state = (await client.get(f"/api/runs/{run_id}")).json()
        assert final_state["status"] == "done"


async def test_contended_job_registry_lock_does_not_false_orphan_heal_start_or_repair(
    tmp_path: Path,
) -> None:
    """Regression (Challenger 2 Bug 1): contended job_registry._lock during start/repair
    must never cause concurrent GET to orphan-heal a live run to 'interrupted'.
    """

    class _DelayedRunner(_StubPipelineRunner):
        async def run(
            self,
            run_state: RunState,
            *,
            event_cb: Any = None,
            cancel: CancellationToken | None = None,
        ) -> RunState:
            await asyncio.sleep(0.05)
            return await super().run(run_state, event_cb=event_cb, cancel=cancel)

        async def repair(
            self,
            run_state: RunState,
            *,
            failed_shot_ids: list[str] | None = None,
            event_cb: Any = None,
            cancel: CancellationToken | None = None,
        ) -> RunState:
            await asyncio.sleep(0.05)
            return await super().repair(
                run_state,
                failed_shot_ids=failed_shot_ids,
                event_cb=event_cb,
                cancel=cancel,
            )

    runner = _DelayedRunner(work_dir=tmp_path)
    app = create_app(data_dir=tmp_path, runner=runner, rate_limit_requests=0)
    reg = app.state.services.job_registry
    transport = httpx.ASGITransport(app=app)
    headers = {"X-Actor-Id": "alice"}

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        created = await client.post("/api/runs", json=_sample_run_input(), headers=headers)
        assert created.status_code == 201
        run_id = created.json()["run_id"]

        # 1) Contended POST /start
        async with reg._lock:
            start_task = asyncio.create_task(
                client.post(f"/api/runs/{run_id}/start", headers=headers)
            )
            await asyncio.sleep(0.02)
            mid_get = await client.get(f"/api/runs/{run_id}")
            assert mid_get.status_code == 200
            mid_body = mid_get.json()
            assert mid_body["status"] != "interrupted", f"False orphan heal on start: {mid_body}"
            assert mid_body["warnings"] == []

        start_resp = await start_task
        assert start_resp.status_code == 202
        start_job = await reg.wait_for_job(start_resp.json()["job"]["id"])
        assert start_job is not None and start_job.state == "done"

        # 2) Contended POST /repair
        async with reg._lock:
            repair_task = asyncio.create_task(
                client.post(
                    f"/api/runs/{run_id}/repair",
                    json={"failed_shot_ids": ["s1"]},
                    headers=headers,
                )
            )
            await asyncio.sleep(0.02)
            mid_get_rep = await client.get(f"/api/runs/{run_id}")
            assert mid_get_rep.status_code == 200
            mid_rep_body = mid_get_rep.json()
            assert mid_rep_body["status"] != "interrupted", (
                f"False orphan heal on repair: {mid_rep_body}"
            )
            assert mid_rep_body["warnings"] == []

        repair_resp = await repair_task
        assert repair_resp.status_code == 202
        repair_job = await reg.wait_for_job(repair_resp.json()["job"]["id"])
        assert repair_job is not None and repair_job.state == "done"

        final_body = (await client.get(f"/api/runs/{run_id}")).json()
        assert final_body["status"] == "done"
        assert final_body["warnings"] == []


async def test_sse_events_on_interrupted_run_emits_terminal_event_and_closes(
    tmp_path: Path,
) -> None:
    """Regression (Challenger 2 Bug 2): GET /events on an 'interrupted' run emits terminal SSE
    event and closes promptly without hanging.
    """
    app = create_app(data_dir=tmp_path)
    orphan = RunState(
        run_id="run_interrupted_sse",
        owner_id="alice",
        status="working",
        input=RunInput.model_validate(_sample_run_input()),
        cost_ledger=CostLedger(run_id="run_interrupted_sse", cap_usd=5.0),
    )
    app.state.services.run_store.save(orphan)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        # Loading heals orphaned 'working' run to 'interrupted'
        get_resp = await client.get("/api/runs/run_interrupted_sse")
        assert get_resp.status_code == 200
        assert get_resp.json()["status"] == "interrupted"

        events: list[dict[str, Any]] = []
        async with asyncio.timeout(2.0):
            async with client.stream("GET", "/api/runs/run_interrupted_sse/events") as sse:
                assert sse.status_code == 200
                async for line in sse.aiter_lines():
                    if line.startswith("data:"):
                        events.append(json.loads(line.split(":", 1)[1].strip()))

        assert len(events) == 1
        assert events[0]["type"] == "run_failed"
        assert events[0]["data"]["status"] == "interrupted"
        assert events[0]["data"]["error"]


async def test_serve_media_missing_unsafe_or_escapes_filenames_return_404_not_400(
    tmp_path: Path,
) -> None:
    """Regression (Challenger 2 Bug 3): missing filenames containing 'unsafe' or 'escapes'
    must return 404 while path traversal and symlink escapes still return 400.
    """
    media_root = tmp_path / "media"
    media_store = MediaStore(media_root, resolve_dns_for_ssrf=False)

    secret_file = tmp_path / "secret.txt"
    secret_file.write_text("secret", encoding="utf-8")
    escape_link = media_root / "escape_link.txt"
    escape_link.symlink_to(secret_file)

    app = create_app(data_dir=tmp_path, media_store=media_store)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        r_unsafe = await client.get("/api/media/assets/unsafe_clip.mp4")
        assert r_unsafe.status_code == 404
        assert r_unsafe.json() == {"detail": "Media not found"}

        r_escapes = await client.get("/api/media/assets/escapes_demo.mp4")
        assert r_escapes.status_code == 404
        assert r_escapes.json() == {"detail": "Media not found"}

        r_traversal = await client.get("/api/media/..%2F..%2Fetc%2Fpasswd")
        assert r_traversal.status_code == 400
        assert r_traversal.json() == {"detail": "Unsafe media_id"}

        r_symlink = await client.get("/api/media/escape_link.txt")
        assert r_symlink.status_code == 400
        assert r_symlink.json() == {"detail": "Unsafe media_id"}


async def test_failed_repair_does_not_fabricate_completed_repair_result(
    tmp_path: Path,
) -> None:
    """Regression (Reviewer 1 Finding 3): failed repair returning status='failed' and
    repair_result=None must NOT fabricate repair_result={'status': 'completed'}.
    """

    class _FailingRepairRunner(_StubPipelineRunner):
        async def repair(
            self,
            run_state: RunState,
            *,
            failed_shot_ids: list[str] | None = None,
            event_cb: Any = None,
            cancel: CancellationToken | None = None,
        ) -> RunState:
            run_state.status = "failed"
            run_state.error = "simulated stitch failure"
            run_state.repair_result = None
            return run_state

    runner = _FailingRepairRunner(work_dir=tmp_path)
    app = create_app(data_dir=tmp_path, runner=runner)
    transport = httpx.ASGITransport(app=app)
    headers = {"X-Actor-Id": "alice"}

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        created = await client.post("/api/runs", json=_sample_run_input(), headers=headers)
        assert created.status_code == 201
        run_id = created.json()["run_id"]

        rep = await client.post(
            f"/api/runs/{run_id}/repair",
            json={"failed_shot_ids": ["s1"]},
            headers=headers,
        )
        assert rep.status_code == 202
        job_id = rep.json()["job"]["id"]
        await app.state.services.job_registry.wait_for_job(job_id)

        state = (await client.get(f"/api/runs/{run_id}")).json()
        assert state["status"] == "failed"
        assert state["error"] == "simulated stitch failure"
        assert state["repair_result"] is None

        events: list[dict[str, Any]] = []
        async with asyncio.timeout(2.0):
            async with client.stream("GET", f"/api/runs/{run_id}/events") as sse:
                assert sse.status_code == 200
                async for line in sse.aiter_lines():
                    if line.startswith("data:"):
                        ev = json.loads(line.split(":", 1)[1].strip())
                        events.append(ev)
                        if ev["type"] in ("run_completed", "run_cancelled", "run_failed"):
                            break

        event_types = [ev["type"] for ev in events]
        assert "run_failed" in event_types
        assert "run_completed" not in event_types



