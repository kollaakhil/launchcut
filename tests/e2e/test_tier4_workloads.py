"""Tier 4 E2E Tests: Realistic end-to-end application workloads & repo-wide invariants."""

from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path

import httpx
import pytest

from gtm_videogen.core.media_store import MediaStore
from gtm_videogen.rules.creative_history import CreativeHistoryStore

from .conftest import (
    assert_valid_sse_event,
    ffprobe_inspect,
    make_run_input,
    make_run_state,
    repo_root_path,
)

pytestmark = [pytest.mark.e2e, pytest.mark.tier4]

LEGACY_TEMPLATE_PATTERN = re.compile(
    r"confession_loop|NEON_LAB_RULES|droneBed|END_STING|112 BPM|OFF_FRAME_POUR|cola pours",
    re.IGNORECASE,
)


@pytest.mark.m1
def test_repo_wide_zero_hard_templates_invariant() -> None:
    """Repo-wide negative invariant: zero legacy hard-template identifiers in src/gtm_videogen/."""
    src_root = repo_root_path() / "src" / "gtm_videogen"
    assert src_root.is_dir(), f"Missing source tree: {src_root}"

    violations: list[str] = []
    for py_file in sorted(src_root.rglob("*.py")):
        text = py_file.read_text(encoding="utf-8")
        for match in LEGACY_TEMPLATE_PATTERN.finditer(text):
            rel = py_file.relative_to(src_root)
            violations.append(f"{rel}: matched {match.group(0)!r}")

    assert violations == [], f"Legacy hard-template identifiers found in src/: {violations}"


def test_full_offline_cli_workload(tmp_path: Path) -> None:
    """Workload: `gtm-videogen run --mock` produces ffprobe-verified MP4, QAReport & CostLedger."""
    from typer.testing import CliRunner

    from gtm_videogen.cli.main import app

    out_dir = tmp_path / "cli_run"
    out_dir.mkdir(parents=True, exist_ok=True)
    result = CliRunner().invoke(
        app,
        [
            "run",
            "--brand-name",
            "Kestrel Roasters",
            "--brand-url",
            "https://kestrelroasters.example.com",
            "--goal",
            "Drive coffee subscription trials",
            "--cap-usd",
            "5.0",
            "--output-dir",
            str(out_dir),
            "--mock",
        ],
    )
    assert result.exit_code == 0, result.output

    mp4_files = list(out_dir.rglob("*.mp4"))
    assert len(mp4_files) >= 1, f"Expected rendered MP4 in {out_dir}, output was:\n{result.output}"
    final_mp4 = max(mp4_files, key=lambda p: p.stat().st_size)
    probe = ffprobe_inspect(final_mp4)
    v_streams = [s for s in probe["streams"] if s.get("codec_type") == "video"]
    a_streams = [s for s in probe["streams"] if s.get("codec_type") == "audio"]
    assert len(v_streams) >= 1
    assert len(a_streams) >= 1

    # Verify persisted run state JSON includes QAReport and CostLedger within cap_usd
    json_files = [p for p in out_dir.rglob("*.json") if "run" in p.name.lower()]
    assert len(json_files) >= 1
    parsed = json.loads(json_files[0].read_text(encoding="utf-8"))
    assert parsed.get("qa_report") is not None
    ledger = parsed.get("cost_ledger")
    assert isinstance(ledger, dict)
    assert isinstance(ledger.get("actual_usd"), (int, float))
    assert isinstance(ledger.get("cap_usd"), (int, float))
    assert 0.0 <= ledger["actual_usd"] <= ledger["cap_usd"]


async def test_full_fastapi_lifecycle_workload(tmp_path: Path) -> None:
    """Workload: FastAPI create -> start -> SSE -> HTTP 206 Range -> async /repair."""
    from gtm_videogen.api.app import create_app

    media_store = MediaStore(tmp_path / "media", resolve_dns_for_ssrf=False)
    app = create_app(data_dir=tmp_path, media_store=media_store)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        created = await client.post("/api/runs", json=make_run_input(mock=True).model_dump())
        assert created.status_code in (200, 201)
        run_id = created.json()["run_id"]

        started = await client.post(f"/api/runs/{run_id}/start")
        assert started.status_code in (200, 202)

        completed_seen = False
        async with client.stream("GET", f"/api/runs/{run_id}/events") as sse:
            assert sse.status_code == 200
            async for line in sse.aiter_lines():
                if not line.startswith("data:"):
                    continue
                payload_str = line.split(":", 1)[1].strip()
                if not payload_str:
                    continue
                event = json.loads(payload_str)
                assert_valid_sse_event(event, expected_run_id=run_id)
                if event["type"] == "run_completed":
                    completed_seen = True
                    break
        assert completed_seen is True

        run_body = (await client.get(f"/api/runs/{run_id}")).json()
        assert run_body["status"] == "done"
        stitched_path = Path(run_body["render_manifest"]["stitched_path"])
        assert stitched_path.exists()

        media_id, _ = media_store.save_from_path(stitched_path)
        expected_bytes = stitched_path.read_bytes()
        ranged = await client.get(f"/api/media/{media_id}", headers={"Range": "bytes=0-2047"})
        assert ranged.status_code == 206
        assert len(ranged.content) == 2048
        assert ranged.content == expected_bytes[:2048]
        content_range = ranged.headers.get("content-range")
        assert content_range is not None
        assert content_range.startswith(f"bytes 0-2047/{len(expected_bytes)}")

        repair_resp = await client.post(
            f"/api/runs/{run_id}/repair",
            json={"failed_shot_ids": [run_body["shot_sheet"]["shots"][0]["id"]]},
        )
        assert repair_resp.status_code in (200, 202)

        repaired_body = None
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            r = await client.get(f"/api/runs/{run_id}")
            body = r.json()
            if body.get("repair_result") is not None:
                repaired_body = body
                break
            await asyncio.sleep(0.1)
        assert repaired_body is not None
        assert repaired_body["repair_result"] is not None


async def test_multi_aspect_pipeline_workload_16_9_and_9_16(tmp_path: Path) -> None:
    """Workload: PipelineRunner renders both 16:9 and 9:16 with ffprobe-verified aspect ratios."""
    from gtm_videogen.agents.pipeline import PipelineRunner
    from gtm_videogen.providers.mocks import MockProviderBundle

    for aspect in ("16:9", "9:16"):
        sub_dir = tmp_path / aspect.replace(":", "x")
        bundle = MockProviderBundle.create(work_dir=sub_dir)
        runner = PipelineRunner(providers=bundle, work_dir=sub_dir)
        run_state = make_run_state(
            run_id=f"aspect_{aspect.replace(':', '_')}",
            run_input=make_run_input(aspect=aspect, mock=True),
        )
        done = await runner.run(run_state)
        assert done.status == "done"
        assert done.render_manifest is not None
        assert done.render_manifest.stitched_path is not None

        probe = ffprobe_inspect(Path(done.render_manifest.stitched_path))
        v_stream = next(s for s in probe["streams"] if s.get("codec_type") == "video")
        w, h = int(v_stream["width"]), int(v_stream["height"])
        if aspect == "16:9":
            assert w > h, f"Expected landscape 16:9, got {w}x{h}"
        else:
            assert h > w, f"Expected portrait 9:16, got {w}x{h}"


async def test_creative_ab_multi_version_workload(tmp_path: Path) -> None:
    """Workload: 2 consecutive runs for same brand record both fingerprints & vary choices."""
    from gtm_videogen.agents.pipeline import PipelineRunner
    from gtm_videogen.providers.mocks import MockProviderBundle

    history_path = tmp_path / "creative_history.json"
    history_store = CreativeHistoryStore(history_path)
    brand_url = "https://kestrelroasters.example.com"

    bundle1 = MockProviderBundle.create(work_dir=tmp_path / "v1")
    runner1 = PipelineRunner(
        providers=bundle1,
        work_dir=tmp_path / "v1",
        history_store=history_store,
    )
    done1 = await runner1.run(
        make_run_state(run_id="ab_v1", run_input=make_run_input(brand_url=brand_url, mock=True))
    )
    assert done1.status == "done"

    bundle2 = MockProviderBundle.create(work_dir=tmp_path / "v2")
    runner2 = PipelineRunner(
        providers=bundle2,
        work_dir=tmp_path / "v2",
        history_store=history_store,
    )
    done2 = await runner2.run(
        make_run_state(run_id="ab_v2", run_input=make_run_input(brand_url=brand_url, mock=True))
    )
    assert done2.status == "done"

    history = history_store.get_history(brand_url)
    assert len(history.entries) >= 2
    assert done1.sound_plan is not None and done1.sound_plan.voice_id
    assert done2.sound_plan is not None and done2.sound_plan.voice_id
    assert done2.sound_plan.voice_id != done1.sound_plan.voice_id
