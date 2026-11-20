"""Tier 3 E2E Tests: Pairwise cross-feature interactions across M1-M5."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from gtm_videogen.core.cache_store import CacheStore
from gtm_videogen.core.cost_meter import CostMeter
from gtm_videogen.core.media_store import MediaStore
from gtm_videogen.models.schemas import CreativeFingerprint, Shot, VoiceCasting
from gtm_videogen.rules.creative_history import CreativeHistoryStore, format_previously_explored
from gtm_videogen.rules.script_gate import evaluate_script_gate
from gtm_videogen.rules.voice_casting import select_voice

from .conftest import (
    assert_valid_sse_event,
    make_creative_bible,
    make_research_bundle,
    make_run_input,
    make_run_state,
    make_shot_sheet,
    make_sound_plan,
)

pytestmark = [pytest.mark.e2e, pytest.mark.tier3]


@pytest.mark.m1
def test_m1_combo_history_voice_casting_prompt_injection_and_script_gate(tmp_path: Path) -> None:
    """M1 combo: CreativeHistoryStore + select_voice + format_previously_explored + script_gate."""
    store = CreativeHistoryStore(tmp_path / "creative_history.json")
    research = make_research_bundle()
    casting = VoiceCasting(
        gender="female", age="adult", accent="us", energy="warm", pace="natural"
    )

    # Run 1: empty history -> pick initial voice & validate grounded script
    hist_before = store.get_history(research.brand_url)
    assert hist_before.entries == []
    voice_run1 = select_voice(casting, brand_history=hist_before)

    sheet_run1 = make_shot_sheet()
    gate1 = evaluate_script_gate(
        sheet_run1,
        research,
        has_product_photos=True,
        offer="Save 25% on your first two bags",
        must_say=["Roasted to order in Seattle"],
    )
    assert gate1.passed is True

    # Record Run 1 creative fingerprint
    store.record(
        CreativeFingerprint(
            run_id="run_combo_1",
            brand_url=research.brand_url,
            arc_name="Roast-Day Ritual",
            hook_opening="Fresh coffee should not sit in a warehouse.",
            visual_world="Warm roastery counter",
            lighting_style="Diffused window light",
            camera_style="Steady macro glides",
            music_genre="warm minimalist jazz",
            voice_id=voice_run1.voice_id,
            outcome="completed",
        )
    )

    # Run 2: history prompt includes prior run and select_voice avoids repeating Run 1 voice
    hist_run2 = store.get_history(research.brand_url)
    assert len(hist_run2.entries) == 1
    injected = format_previously_explored(hist_run2)
    assert "Roast-Day Ritual" in injected
    assert "Warm roastery counter" in injected

    voice_run2 = select_voice(casting, brand_history=hist_run2)
    assert voice_run2.voice_id != voice_run1.voice_id


@pytest.mark.m1
async def test_m1_combo_cost_meter_cache_store_and_media_store_zero_dollar_hit(
    tmp_path: Path,
) -> None:
    """M1 combo: CostMeter + CacheStore + MediaStore reuses media:// artifact at $0.00."""
    media = MediaStore(tmp_path / "media", resolve_dns_for_ssrf=False)
    caches = CacheStore(tmp_path / "caches")
    meter = CostMeter(run_id="combo_cost_cache", cap_usd=1.0, ledger_path=tmp_path / "spend.jsonl")

    payload_spec = {"shot_id": "s1", "prompt": "Macro coffee beans", "aspect": "16:9", "fps": 24}
    cache_key = caches.clip_cache.make_key(payload_spec)

    # First render: cache miss -> reserve & settle $0.45, persist in MediaStore + CacheStore
    assert caches.clip_cache.get(cache_key) is None
    res1 = meter.reserve("clip_s1", 0.50, stage="render", provider="fal", model="kling")
    media_id, saved_path = media.save_bytes(b"synthetic-mp4-content", ext=".mp4")
    caches.clip_cache.put(cache_key, artifact_path=saved_path, metadata={"media_id": media_id})
    meter.settle(res1, 0.45, cache_hit=False)
    assert meter.spent_usd == pytest.approx(0.45)

    # Second render: cache hit -> resolves media:// path and settles at $0.00 (preserving budget)
    hit = caches.clip_cache.get(cache_key)
    assert hit is not None
    resolved = await media.resolve_to_path(f"media://{hit.metadata['media_id']}")
    assert resolved == saved_path

    res2 = meter.reserve("clip_s1_cached", 0.0, stage="render", provider="cache", model="clip")
    entry2 = meter.settle(res2, 0.0, cache_hit=True)
    assert entry2.cache_hit is True
    assert meter.spent_usd == pytest.approx(0.45)
    assert meter.remaining_usd == pytest.approx(0.55)


async def test_m2_m4_combo_history_agent1_non_repetition_and_scraper_grounding(
    tmp_path: Path,
) -> None:
    """M2+M4 combo: CreativeHistory -> Agent 1 non-repetition + select_voice + BrandScraper."""
    from gtm_videogen.agents.agent1_research_bible import run_agent1_research_bible

    from gtm_videogen.providers.mocks import MockProviderBundle

    bundle = MockProviderBundle.create(work_dir=tmp_path)
    history_store = CreativeHistoryStore(tmp_path / "history.json")
    run_input = make_run_input(mock=True)

    res1 = await run_agent1_research_bible(
        run_input=run_input,
        providers=bundle,
        history_store=history_store,
        work_dir=tmp_path / "r1",
    )
    assert res1.research is not None
    assert len(res1.research.verified_claims) + len(res1.research.product_facts) >= 1
    assert len(res1.research.product_photos) >= 1
    assert res1.creative_bible is not None
    voice1 = select_voice(res1.creative_bible.sonic_identity.voice_casting)

    history_store.record(
        CreativeFingerprint(
            run_id="r1",
            brand_url=run_input.brand_url,
            arc_name=res1.creative_bible.narrative_structure.arc_name,
            hook_opening=res1.creative_bible.concept_thesis,
            visual_world=res1.creative_bible.visual_language.world,
            lighting_style=res1.creative_bible.visual_language.lighting_style,
            camera_style=res1.creative_bible.camera_grammar.dominant_style,
            voice_id=voice1.voice_id,
        )
    )

    res2 = await run_agent1_research_bible(
        run_input=run_input,
        providers=bundle,
        history_store=history_store,
        work_dir=tmp_path / "r2",
    )
    assert res2.creative_bible is not None
    bible_varied = (
        res2.creative_bible.concept_thesis != res1.creative_bible.concept_thesis
        or res2.creative_bible.narrative_structure.arc_name
        != res1.creative_bible.narrative_structure.arc_name
        or res2.creative_bible.visual_language.world != res1.creative_bible.visual_language.world
    )
    assert bible_varied, "Second Agent 1 run must vary its CreativeBible given CreativeHistory"

    voice2 = select_voice(
        res2.creative_bible.sonic_identity.voice_casting,
        brand_history=history_store.get_history(run_input.brand_url),
    )
    assert voice2.voice_id != voice1.voice_id


async def test_m1_m4_combo_agent2_rubric_revision_loop_fixes_ungrounded_claim(
    tmp_path: Path,
) -> None:
    """M1+M4 combo: Agent 2 RubricMiddleware fails iter 1 on ungrounded claim -> revises iter 2."""
    from gtm_videogen.agents.agent2_director import run_agent2_director

    from gtm_videogen.providers.mocks import MockProviderBundle

    research = make_research_bundle()
    bible = make_creative_bible()
    run_input = make_run_input(mock=True)

    bad_sheet = make_shot_sheet(
        shots=[
            Shot(
                id="s1",
                index=0,
                beat="hook",
                duration_sec=6.0,
                visual_prompt="Coffee cup",
                vo_line="The #1 best guaranteed miracle coffee on earth!",
            ),
            Shot(
                id="s2",
                index=1,
                beat="cta",
                duration_sec=6.0,
                visual_prompt="Coffee bag",
                vo_line="Order now at {{PRICE}}.",
            ),
        ],
    )
    good_sheet = make_shot_sheet()
    good_sound = make_sound_plan()
    draft_calls = {"count": 0}

    def _director_responder(_messages: list) -> str:
        draft_calls["count"] += 1
        chosen_sheet = bad_sheet if draft_calls["count"] == 1 else good_sheet
        return json.dumps(
            {
                "shot_sheet": chosen_sheet.model_dump(mode="json"),
                "sound_plan": good_sound.model_dump(mode="json"),
            }
        )

    bundle = MockProviderBundle.create(work_dir=tmp_path, responder=_director_responder)
    result = await run_agent2_director(
        run_input=run_input,
        research=research,
        creative_bible=bible,
        providers=bundle,
        work_dir=tmp_path,
    )
    assert result.shot_sheet is not None
    assert result.sound_plan is not None
    assert len(result.rubric_evaluations) >= 2
    final_gate = evaluate_script_gate(
        result.shot_sheet,
        research,
        has_product_photos=bool(research.product_photos),
        offer=run_input.offer,
        must_say=run_input.must_say,
    )
    assert final_gate.passed is True


async def test_m2_m3_m4_combo_agent3_cache_reuse_stitch_qa_and_targeted_repair(
    tmp_path: Path,
) -> None:
    """M2+M3+M4 combo: Agent 3 overlapped render + cache reuse + stitch + QA + repair()."""
    from gtm_videogen.agents.pipeline import PipelineRunner

    from gtm_videogen.providers.mocks import MockProviderBundle

    bundle = MockProviderBundle.create(work_dir=tmp_path)
    runner = PipelineRunner(providers=bundle, work_dir=tmp_path)

    run1 = make_run_state(run_id="combo_p1", run_input=make_run_input(mock=True))
    done1 = await runner.run(run1)
    assert done1.status == "done"
    assert done1.render_manifest is not None
    assert done1.render_manifest.stitched_path is not None
    assert Path(done1.render_manifest.stitched_path).exists()
    assert done1.qa_report is not None

    # Targeted single-shot repair on the first shot
    first_shot_id = done1.shot_sheet.shots[0].id if done1.shot_sheet else "s1"
    repaired = await runner.repair(done1, failed_shot_ids=[first_shot_id])
    assert repaired.repair_result is not None
    assert repaired.render_manifest is not None
    assert repaired.render_manifest.stitched_path is not None
    assert Path(repaired.render_manifest.stitched_path).exists()


async def test_m4_m5_combo_fastapi_sse_lifecycle_and_http_206_range(tmp_path: Path) -> None:
    """M4+M5 combo: FastAPI create -> start -> SSE events -> HTTP 206 Range media playback."""
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

        parsed_events: list[dict] = []
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
                parsed_events.append(event)
                if event["type"] == "run_completed":
                    break

        seen_types = {ev["type"] for ev in parsed_events}
        assert "run_completed" in seen_types
        assert "stage_started" in seen_types

        final_run = (await client.get(f"/api/runs/{run_id}")).json()
        assert final_run["status"] == "done"
        stitched_path = Path(final_run["render_manifest"]["stitched_path"])
        assert stitched_path.exists()

        media_id, _ = media_store.save_from_path(stitched_path)
        expected_bytes = stitched_path.read_bytes()
        ranged = await client.get(
            f"/api/media/{media_id}",
            headers={"Range": "bytes=0-1023"},
        )
        assert ranged.status_code == 206
        assert len(ranged.content) == 1024
        assert ranged.content == expected_bytes[:1024]
        content_range = ranged.headers.get("content-range")
        assert content_range is not None
        assert content_range.startswith(f"bytes 0-1023/{len(expected_bytes)}")
        await asyncio.sleep(0)
