"""Tests for the 5-step Human-in-the-Loop (HITL) Studio API endpoints."""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest

from gtm_videogen.api.app import create_app


async def _wait_until_not_working(
    client: httpx.AsyncClient,
    run_id: str,
    *,
    timeout_sec: float = 45.0,
) -> dict:
    deadline = asyncio.get_running_loop().time() + timeout_sec
    while asyncio.get_running_loop().time() < deadline:
        resp = await client.get(f"/api/runs/{run_id}")
        assert resp.status_code == 200
        body = resp.json()
        if body["status"] != "working":
            return body
        await asyncio.sleep(0.15)
    raise AssertionError(f"Run {run_id} did not settle within {timeout_sec}s")


@pytest.mark.asyncio
async def test_hitl_5_step_studio_workflow_offline(tmp_path: Path) -> None:
    """Exercise Brand -> Concept -> Shot Sheet -> Multi-Option Previews -> Final Cut end-to-end."""
    app = create_app(data_dir=tmp_path)
    headers = {"X-Actor-Id": "hitl_tester"}
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver", headers=headers
    ) as client:
        # Presets catalog
        presets_resp = await client.get("/api/studio/presets")
        assert presets_resp.status_code == 200
        presets = presets_resp.json()["presets"]
        assert len(presets) >= 4

        # Step 1: Create run & trigger research ONLY
        create_resp = await client.post(
            "/api/runs",
            json={
                "brand_name": "Kestrel Roasters",
                "brand_url": "https://kestrelroasters.example.com",
                "goal": "Drive coffee subscription trials with grounded sourcing proof",
                "target_duration_sec": 12.0,
                "aspect": "16:9",
                "fps": 24,
                "audio_mode": "both",
                "cap_usd": 5.0,
                "mock": True,
            },
        )
        assert create_resp.status_code == 201
        run_id = create_resp.json()["run_id"]

        # Editable USD budget cap
        cap_resp = await client.patch(f"/api/runs/{run_id}/cap", json={"cap_usd": 8.5})
        assert cap_resp.status_code == 200
        assert cap_resp.json()["cost_ledger"]["cap_usd"] == 8.5

        res1 = await client.post(f"/api/runs/{run_id}/step/research")
        assert res1.status_code == 202
        state1 = await _wait_until_not_working(client, run_id)
        assert state1["step"] == "concept"
        assert state1["creative_bible"] is not None
        assert len(state1["concepts"]) >= 2

        # Step 2 HITL: Pick 2nd concept & edit CreativeBible thesis
        second_cid = state1["concepts"][1]["id"]
        patch_concept = await client.patch(
            f"/api/runs/{run_id}/concept",
            json={
                "selected_concept_id": second_cid,
                "concept_thesis": "Custom human-approved thesis for Kestrel Roasters.",
            },
        )
        assert patch_concept.status_code == 200
        assert patch_concept.json()["selected_concept_id"] == second_cid
        assert (
            patch_concept.json()["creative_bible"]["concept_thesis"]
            == "Custom human-approved thesis for Kestrel Roasters."
        )

        # Step 2 -> 3: Direct Script & Shot Sheet ONLY
        res2 = await client.post(f"/api/runs/{run_id}/step/direct")
        assert res2.status_code == 202
        state2 = await _wait_until_not_working(client, run_id)
        assert state2["step"] == "sheet"
        assert state2["shot_sheet"] is not None
        assert len(state2["shot_sheet"]["shots"]) >= 1

        # Step 3 HITL: Edit ShotSheet & run live pure-script-rules validation
        edited_sheet = dict(state2["shot_sheet"])
        edited_sheet["end_card_title"] = "Kestrel Roasters"
        patch_sheet = await client.patch(
            f"/api/runs/{run_id}/sheet",
            json={"shot_sheet": edited_sheet},
        )
        assert patch_sheet.status_code == 200
        assert patch_sheet.json()["shot_sheet"]["end_card_title"] == "Kestrel Roasters"
        assert patch_sheet.json()["qa_report"] is not None

        # Step 4: Render 2 Preview stacks side-by-side
        res4 = await client.post(
            f"/api/runs/{run_id}/step/previews",
            json={
                "stacks": [
                    {"id": "mock_a", "label": "Pacing A", "mock": True},
                    {
                        "id": "mock_b",
                        "label": "Hybrid B",
                        "hybrid_motion_cards": True,
                        "mock": True,
                    },
                ]
            },
        )
        assert res4.status_code == 202
        state4 = await _wait_until_not_working(client, run_id)
        assert state4["step"] == "previews"
        assert len(state4["versions"]) == 2
        v0 = state4["versions"][0]["version_id"]

        # Stream preview video with HTTP 206 Range
        vid_resp = await client.get(
            f"/api/runs/{run_id}/versions/{v0}/video",
            headers={"Range": "bytes=0-1023"},
        )
        assert vid_resp.status_code == 206
        assert len(vid_resp.content) == 1024

        # Pick Winner
        pick_resp = await client.post(f"/api/runs/{run_id}/versions/{v0}/pick")
        assert pick_resp.status_code == 200
        assert pick_resp.json()["winning_version_id"] == v0

        # Step 5: Render Final Cut
        res5 = await client.post(f"/api/runs/{run_id}/step/final", json={"version_id": v0})
        assert res5.status_code == 202
        state5 = await _wait_until_not_working(client, run_id)
        assert state5["status"] == "done"
        assert state5["step"] == "final"
        assert state5["render_manifest"]["stitched_path"]

        final_vid = await client.get(f"/api/runs/{run_id}/video")
        assert final_vid.status_code == 200
        assert len(final_vid.content) > 1000
