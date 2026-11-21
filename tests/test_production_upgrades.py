"""Tests for Production-Grade Upgrades: 18 Themes, 10 Aspect-Aware Templates (2.5D), VO Pacing & 3-Format Campaign Pack."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from gtm_videogen.api.app import create_app
from gtm_videogen.core.config import Settings
from gtm_videogen.media.motion_templates import render_motion_template_mp4
from gtm_videogen.models.stitch_project import MotionGraphicSpec


def _make_client(tmp_path: Path) -> TestClient:
    settings = Settings(
        data_dir=tmp_path,
        fal_key="",
        openrouter_api_key="",
    )
    app = create_app(data_dir=tmp_path, settings=settings, rate_limit_requests=0)
    return TestClient(app)


@pytest.mark.asyncio
async def test_new_agency_templates_in_16x9_and_9x16_portrait(tmp_path: Path) -> None:
    new_templates = [
        "device_showcase_25d",
        "node_flow_diagram",
        "customer_quote_proof",
        "kinetic_hook_reel",
    ]
    for tpl in new_templates:
        for tag, (w, h) in (("16x9", (480, 270)), ("9x16", (270, 480))):
            out_mp4 = tmp_path / f"{tpl}_{tag}.mp4"
            spec = MotionGraphicSpec(
                template_id=tpl,  # type: ignore[arg-type]
                theme_mode="clean_light" if tpl == "node_flow_diagram" else "dark_glass",
                eyebrow="LAUNCHCUT PRO",
                headline="Ship studio launch films in minutes",
                subheadline="Automated 2.5D UI & multi-format export",
                prompt_text="Cut our launch production from 3 weeks to 15 minutes.",
                metrics=[{"label": "Brand URL"}, {"label": "Motion Engine"}, {"label": "Campaign Pack"}],
            )
            res = await render_motion_template_mp4(
                spec=spec,
                output_path=out_mp4,
                width=w,
                height=h,
                fps=12,
                duration_sec=2.0,
                brand_name="LaunchCut",
            )
            assert res.is_file()
            assert res.stat().st_size > 3_500


def test_catalog_18_themes_vo_sync_and_three_format_campaign_pack(tmp_path: Path) -> None:
    client = _make_client(tmp_path)

    # 1. Verify 18 curated themes + 10 motion templates + new thumbnails
    cat = client.get("/api/stitch/catalog").json()
    assert len(cat["gallery"]) == 18
    assert len(cat["motion_templates"]) == 10
    for thumb in ("theme_apple.jpg", "theme_linear.jpg", "theme_stripe.jpg", "theme_reel.jpg"):
        r = client.get(f"/api/stitch/gallery/{thumb}")
        assert r.status_code == 200
        assert len(r.content) > 5_000

    # 2. Create a Stripe Living Architecture project (all motion_graphic hybrid plan)
    create_r = client.post(
        "/api/stitch/projects",
        json={
            "idea": "Instant global treasury API with real-time settlement",
            "brand_name": "FlowPay",
            "world_preset": "stripe_living_flow",
            "recipe_id": "stripe_living_flow",
            "target_duration_sec": 10,
            "mock": True,
        },
    )
    assert create_r.status_code == 200
    proj = create_r.json()
    pid = proj["project_id"]
    assert proj["scenes"][0]["kind"] == "motion_graphic"

    # 3. Sync VO Pacing ($0 cadence)
    vo_r = client.post(f"/api/stitch/projects/{pid}/sync-vo", json={"mock": True})
    assert vo_r.status_code == 200
    assert "VO-synced" in vo_r.json()["status_message"]

    # 4. Export 3-Format Campaign Pack (16:9 + 9:16 + 1:1) with native reflow at $0 AI cost
    pack_r = client.post(
        f"/api/stitch/projects/{pid}/campaign-pack",
        json={"audio_mode": "native_plus_score", "enable_captions": True, "mock": True},
    )
    assert pack_r.status_code == 200
    pack_urls = pack_r.json()["campaign_pack_urls"]
    assert set(pack_urls.keys()) == {"16:9", "9:16", "1:1"}
    for asp, url in pack_urls.items():
        rng = client.get(url, headers={"Range": "bytes=0-511"})
        assert rng.status_code == 206, f"Failed 206 stream for {asp}"
        assert len(rng.content) == 512
