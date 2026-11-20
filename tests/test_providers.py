"""Comprehensive offline tests for M2 provider clients, scraper, router, and mocks."""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from pydantic import BaseModel

from gtm_videogen.core.cache_store import CacheStore
from gtm_videogen.core.config import Settings
from gtm_videogen.core.cost_meter import CostMeter
from gtm_videogen.core.http_client import ResilientHttpClient
from gtm_videogen.core.job_registry import CancellationToken
from gtm_videogen.core.media_store import MediaStore
from gtm_videogen.core.ssrf import SSRFError
from gtm_videogen.models.schemas import ChatOptions, MusicCue, SfxCue, Shot, VoiceCue
from gtm_videogen.providers.audio import (
    SFX_FALLBACK_CHAIN,
    AudioProvider,
    AudioProviderError,
    build_music_payload,
    build_sfx_payload,
    build_tts_payload,
    extract_audio_url,
    speakable,
)
from gtm_videogen.providers.llm_router import (
    STAGE_PROFILES,
    LLMRouter,
    estimate_llm_cost,
    record_llm_cost,
)
from gtm_videogen.providers.mocks import (
    MOCK_ASPECT_DIMENSIONS,
    MockAudioProvider,
    MockBrandScraper,
    MockProviderBundle,
    MockVideoProvider,
    MockWhisperProvider,
    ScriptedChatModel,
)
from gtm_videogen.providers.scraper import BrandScraper
from gtm_videogen.providers.speech_canon import (
    _align_brand_terms,
    asr_vocabulary,
    canonicalize_speech,
    levenshtein_distance,
    same_canon,
)
from gtm_videogen.providers.video import (
    VideoProvider,
    build_fal_video_payload,
    build_higgsfield_video_payload,
    build_openrouter_video_payload,
)
from gtm_videogen.providers.whisper import WhisperError, WhisperProvider


def _ffprobe_json(path: Path) -> dict[str, Any]:
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_streams",
            "-show_format",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(proc.stdout)


# ===========================================================================
# 1. F8 — LLMRouter & Provider Capability-Filtering Matrix
# ===========================================================================


class TestLLMRouter:
    def test_all_11_stage_profiles_present_and_bug27_sound_design_uses_text_model(self) -> None:
        expected_stages = {
            "research",
            "claims_check",
            "qa",
            "shot_check",
            "creative_bible",
            "concept_divergence",
            "draft_hooks",
            "script_write",
            "sheet_compile",
            "repair",
            "sound_design",
        }
        assert set(STAGE_PROFILES.keys()) == expected_stages

        # BUG-27: sound_design MUST use text_model, NEVER vision_model
        assert STAGE_PROFILES["sound_design"].model_slot == "text_model"
        assert STAGE_PROFILES["qa"].model_slot == "vision_model"
        assert STAGE_PROFILES["shot_check"].model_slot == "vision_model"

        cfg = Settings(
            fast_model="google/gemini-2.5-flash",
            flagship_model="anthropic/claude-opus-4",
            text_model="anthropic/claude-sonnet-4",
            vision_model="google/gemini-2.5-vision",
        )
        router = LLMRouter(settings=cfg)
        assert router.resolve_model("sound_design") == "anthropic/claude-sonnet-4"
        assert router.resolve_model("soundDesign") == "anthropic/claude-sonnet-4"
        assert router.resolve_model("sound_design") != cfg.vision_model
        assert router.resolve_model("qa") == "google/gemini-2.5-vision"
        assert router.resolve_model("research") == "google/gemini-2.5-flash"
        assert router.resolve_model("creative_bible") == "anthropic/claude-opus-4"
        assert router.resolve_model("creativeBible") == "anthropic/claude-opus-4"
        assert router.resolve_model("script_write", model_override="openai/gpt-4.1") == (
            "openai/gpt-4.1"
        )

    def test_reasoning_models_omit_sampling_params(self) -> None:
        router = LLMRouter()
        for model in ("o1", "openai/o3-mini", "openai/o4-mini", "deepseek/deepseek-r1"):
            opts = router.build_chat_options(
                model,
                "concept_divergence",
                ChatOptions(
                    temperature=1.0,
                    top_p=0.95,
                    presence_penalty=0.5,
                    frequency_penalty=0.3,
                    reasoning_effort="high",
                ),
            )
            assert "temperature" not in opts
            assert "top_p" not in opts
            assert "presence_penalty" not in opts
            assert "frequency_penalty" not in opts
            assert opts["reasoning_effort"] == "high"
            assert "max_tokens" in opts

        # Setting reasoning_effort on a non-o-series model also omits sampling params
        opts_effort = router.build_chat_options(
            "openai/gpt-5",
            "script_write",
            ChatOptions(reasoning_effort="medium"),
        )
        assert "temperature" not in opts_effort
        assert "top_p" not in opts_effort
        assert opts_effort["reasoning_effort"] == "medium"

    def test_anthropic_omits_penalties_and_sends_only_one_of_temperature_or_top_p(self) -> None:
        router = LLMRouter()
        for model in ("anthropic/claude-sonnet-4", "claude-3-7-sonnet-20250219"):
            opts = router.build_chat_options(
                model,
                "concept_divergence",
                ChatOptions(
                    temperature=0.8,
                    top_p=0.9,
                    presence_penalty=0.5,
                    frequency_penalty=0.3,
                ),
            )
            assert "presence_penalty" not in opts
            assert "frequency_penalty" not in opts
            # Prefers temperature and omits top_p when both are set
            assert opts["temperature"] == 0.8
            assert "top_p" not in opts

    def test_json_schema_native_vs_json_object_fallback(self) -> None:
        router = LLMRouter()
        schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}

        for capable in (
            "openai/gpt-4o",
            "google/gemini-2.5-flash",
            "anthropic/claude-sonnet-4",
            "openai/o3-mini",
        ):
            opts = router.build_chat_options(
                capable,
                "sheet_compile",
                ChatOptions(json_schema=schema),
            )
            assert opts["response_format"]["type"] == "json_schema"
            assert opts["response_format"]["json_schema"]["schema"] == schema

        fallback_opts = router.build_chat_options(
            "meta-llama/llama-3.1-8b-instruct",
            "sheet_compile",
            ChatOptions(json_schema=schema),
        )
        assert fallback_opts["response_format"] == {"type": "json_object"}
        assert all(v is not None for v in fallback_opts.values())

    def test_tiered_llm_cost_estimation_and_cost_meter_recording(self) -> None:
        assert estimate_llm_cost("research") == 0.02
        assert estimate_llm_cost("creative_bible") == 0.10
        assert estimate_llm_cost("fast", prompt_tokens=100_000, completion_tokens=50_000) == round(
            0.1 * 0.30 + 0.05 * 1.20, 6
        )
        assert estimate_llm_cost("flagship", prompt_tokens=100_000, completion_tokens=50_000) == (
            round(0.1 * 3.00 + 0.05 * 15.00, 6)
        )

        meter = CostMeter("run_llm", cap_usd=2.0)
        entry = record_llm_cost(
            meter,
            stage="sound_design",
            model="anthropic/claude-sonnet-4",
            prompt_tokens=10_000,
            completion_tokens=2_000,
        )
        assert entry is not None
        assert entry.stage == "sound_design"
        assert meter.spent_usd == pytest.approx(0.06, abs=1e-6)
        assert meter.reserved_usd == 0.0

    def test_write_sheet_maps_to_script_write_and_multiprefix_model_detection(self) -> None:
        router = LLMRouter()
        for alias in ("writeSheet", "write_sheet"):
            profile = router.get_profile(alias)
            assert profile.stage == "script_write"
            assert profile.tier == "flagship"
            assert profile.temperature == 0.75
            assert profile.presence_penalty == 0.2

        schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}

        # Multi-prefix Anthropic model via OpenRouter must apply Anthropic filtering
        anth_opts = router.build_chat_options(
            "openrouter/anthropic/claude-3-5-sonnet",
            "script_write",
            ChatOptions(
                temperature=0.7,
                top_p=0.9,
                presence_penalty=0.2,
                frequency_penalty=0.1,
                json_schema=schema,
            ),
        )
        assert anth_opts["temperature"] == 0.7
        assert "top_p" not in anth_opts
        assert "presence_penalty" not in anth_opts
        assert "frequency_penalty" not in anth_opts
        assert anth_opts["response_format"] == {"type": "json_object"}

        # Multi-prefix reasoning model via OpenRouter must strip sampling params & keep json_schema
        o3_opts = router.build_chat_options(
            "openrouter/openai/o3-mini",
            "script_write",
            ChatOptions(temperature=0.7, top_p=0.9, json_schema=schema),
        )
        assert "temperature" not in o3_opts
        assert "top_p" not in o3_opts
        assert o3_opts["response_format"]["type"] == "json_schema"


# ===========================================================================
# 2. F9 — Parallel SSRF-Safe BrandScraper
# ===========================================================================


class TestBrandScraper:
    async def test_ssrf_blocks_private_brand_and_inspiration_urls(self) -> None:
        scraper = BrandScraper(resolve_dns_for_ssrf=False)
        with pytest.raises(SSRFError):
            await scraper.scrape_brand("http://127.0.0.1/admin")
        with pytest.raises(SSRFError):
            await scraper.scrape_brand(
                "https://93.184.216.34",
                inspiration_url="http://169.254.169.254/latest/meta-data",
            )

    async def test_single_homepage_fetch_parallel_inspiration_and_extractions(self) -> None:
        call_counts: dict[str, int] = {}
        brand_started = asyncio.Event()
        insp_started = asyncio.Event()
        parallel_overlap = False

        homepage_html = """
        <!doctype html>
        <html>
        <head>
          <title>Solara Botanics | Cold-Pressed Ginger</title>
          <meta name="description"
                content="Cold-pressed ginger tonic crafted in Portland with 4g sugar.">
          <meta name="theme-color" content="#f4a261">
          <meta property="og:site_name" content="Solara Botanics">
          <meta property="og:image" content="https://93.184.216.34/img/og-hero.jpg">
          <style>:root { --brand: #264653; --cream: #faf3dd; --short: #abc; }</style>
          <script type="application/ld+json">
            {
              "@context": "https://schema.org",
              "@graph": [
                {
                  "@type": "Product",
                  "name": "Solara Ginger 6-Pack",
                  "description": "Real Peruvian ginger root and Meyer lemon.",
                  "image": [
                    {"@type": "ImageObject", "url": "https://93.184.216.34/img/jsonld-1.png"},
                    "https://93.184.216.34/img/pixel-tracker.gif",
                    "http://127.0.0.1/secret.png"
                  ],
                  "offers": {"price": "24.00", "priceCurrency": "USD"}
                }
              ]
            }
          </script>
        </head>
        <body style="background:#FAF3DD">
          <h1>Crisp Ginger Tonic Without The Crash</h1>
          <p>Small-batch cold-pressed in Portland and shipped nationwide in 48 hours.</p>
          <a href="/products/ginger-tonic">Shop Ginger Tonic</a>
          <a href="/shop/variety-pack">Variety Pack</a>
        </body>
        </html>
        """

        shopify_json = {
            "products": [
                {
                    "title": "Solara Citrus Zing",
                    "body_html": "<p>Organic Meyer lemon and fresh ginger root.</p>",
                    "images": [
                        {"src": "https://93.184.216.34/img/shopify-1.webp"},
                        {"src": "https://93.184.216.34/img/icon.svg"},
                    ],
                }
            ]
        }

        product_page_html = """
        <html><head>
          <meta property="og:image" content="https://93.184.216.34/img/product-page-1.jpg">
        </head><body>
          <img src="https://93.184.216.34/img/product-page-extra.jpg" alt="Extra shot">
          <p>Zero artificial sweeteners and recyclable glass bottles.</p>
        </body></html>
        """

        insp_html = """
        <html><head><title>Editorial Sunlit Spot</title></head>
        <body><h1>Fast tactile cuts in morning light</h1></body></html>
        """

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal parallel_overlap
            url = str(request.url)
            call_counts[url] = call_counts.get(url, 0) + 1

            if url == "https://93.184.216.34/":
                brand_started.set()
                try:
                    await asyncio.wait_for(insp_started.wait(), timeout=0.5)
                    parallel_overlap = True
                except TimeoutError:
                    pass
                return httpx.Response(200, text=homepage_html)
            if url == "https://93.184.216.35/spot":
                insp_started.set()
                try:
                    await asyncio.wait_for(brand_started.wait(), timeout=0.5)
                    parallel_overlap = True
                except TimeoutError:
                    pass
                return httpx.Response(200, text=insp_html)
            if url == "https://93.184.216.34/products.json":
                return httpx.Response(200, json=shopify_json)
            if "/products/" in url or "/shop/" in url:
                return httpx.Response(200, text=product_page_html)
            return httpx.Response(404, text="Not found")

        client = ResilientHttpClient(
            transport=httpx.MockTransport(handler),
            resolve_dns_for_ssrf=False,
        )
        scraper = BrandScraper(http_client=client, resolve_dns_for_ssrf=False)
        try:
            bundle = await scraper.scrape_brand(
                "https://93.184.216.34/",
                inspiration_url="https://93.184.216.35/spot",
            )
        finally:
            await client.aclose()

        # Homepage fetched ONCE (no double-fetch!)
        assert call_counts["https://93.184.216.34/"] == 1
        # P8 parallel brand + inspiration execution verified
        assert parallel_overlap is True

        assert bundle.brand_name == "Solara Botanics"
        assert "#F4A261" in bundle.brand_colors
        assert "#264653" in bundle.brand_colors
        assert "#FAF3DD" in bundle.brand_colors
        assert "#AABBCC" in bundle.brand_colors

        # Photos deduplicated, SSRF/SVG/tracker filtered, capped at 4
        assert len(bundle.product_photos) == 4
        photo_urls = [p.url for p in bundle.product_photos]
        assert "https://93.184.216.34/img/og-hero.jpg" in photo_urls
        assert "https://93.184.216.34/img/jsonld-1.png" in photo_urls
        assert "https://93.184.216.34/img/shopify-1.webp" in photo_urls
        assert not any("127.0.0.1" in (u or "") for u in photo_urls)
        assert not any(".svg" in (u or "") for u in photo_urls)
        assert not any("pixel-tracker" in (u or "") for u in photo_urls)

        assert any("Editorial Sunlit Spot" in n for n in bundle.inspiration_notes)
        assert len(bundle.product_facts) > 0


# ===========================================================================
# 3. F10 — VideoProvider (BUG-07, P6, BUG-05, BUG-12, BUG-14, P4)
# ===========================================================================


class TestVideoProvider:
    @pytest.mark.parametrize("aspect", ["16:9", "9:16", "1:1", "4:5"])
    @pytest.mark.parametrize("provider_name", ["fal", "openrouter", "higgsfield"])
    async def test_bug07_aspect_ratio_honored_across_all_providers(
        self,
        tmp_path: Path,
        aspect: Any,
        provider_name: str,
    ) -> None:
        captured_payloads: list[dict[str, Any]] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "POST":
                captured_payloads.append(json.loads(request.content))
                return httpx.Response(
                    200,
                    json={"video": {"url": "https://cdn.example.com/out.mp4"}},
                )
            if str(request.url) == "https://cdn.example.com/out.mp4":
                return httpx.Response(200, content=b"FAKE_MP4_BYTES")
            return httpx.Response(404)

        client = ResilientHttpClient(transport=httpx.MockTransport(handler))
        vp = VideoProvider(
            http_client=client,
            media_store=MediaStore(tmp_path / "media"),
            settings=Settings(fal_key="k", openrouter_api_key="k", hf_api_key="k"),
        )
        shot = Shot(
            id="s1",
            index=0,
            beat="hook",
            duration_sec=3.0,
            visual_prompt="Morning sunlight over ceramic table",
        )
        try:
            clip = await vp.generate_clip(shot, aspect=aspect, fps=24, provider=provider_name)
        finally:
            await client.aclose()

        assert len(captured_payloads) == 1
        assert captured_payloads[0]["aspect_ratio"] == aspect
        assert Path(clip.path).is_file()
        assert clip.fps == 24

    def test_payload_builders_never_hardcode_9_16(self) -> None:
        shot = Shot(id="s1", index=0, beat="hook", duration_sec=4.0, visual_prompt="Wide landscape")
        assert build_fal_video_payload(shot, "16:9", 24)["aspect_ratio"] == "16:9"
        assert build_openrouter_video_payload(shot, "16:9", 24, "m")["aspect_ratio"] == "16:9"
        assert build_higgsfield_video_payload(shot, "4:5", 30, "m")["aspect_ratio"] == "4:5"

    async def test_p6_backoff_bug12_min_seconds_bug14_retry_and_p4_clip_cache(
        self,
        tmp_path: Path,
    ) -> None:
        sleeps: list[float] = []
        poll_count = 0
        submit_attempts = 0
        network_calls = 0

        async def fake_sleep(sec: float) -> None:
            sleeps.append(sec)

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal poll_count, submit_attempts, network_calls
            network_calls += 1
            url = str(request.url)

            if request.method == "POST":
                submit_attempts += 1
                if submit_attempts == 1:
                    # BUG-14: HTML 502 must trigger raise_for_status + retry (no JSONDecodeError)
                    return httpx.Response(502, text="<html>Bad Gateway</html>")
                return httpx.Response(
                    200,
                    json={
                        "request_id": "req_123",
                        "status_url": "https://queue.fal.run/m/requests/req_123/status",
                        "response_url": "https://queue.fal.run/m/requests/req_123",
                    },
                )
            if url.endswith("/status"):
                poll_count += 1
                if poll_count < 4:
                    return httpx.Response(200, json={"status": "IN_PROGRESS"})
                return httpx.Response(200, json={"status": "COMPLETED"})
            if url.endswith("/requests/req_123"):
                return httpx.Response(
                    200,
                    json={"video": {"url": "https://cdn.example.com/clip.mp4"}},
                )
            if url == "https://cdn.example.com/clip.mp4":
                return httpx.Response(200, content=b"MP4_DATA_12345")
            return httpx.Response(404)

        client = ResilientHttpClient(
            transport=httpx.MockTransport(handler),
            backoff_base_sec=0.001,
            backoff_max_sec=0.005,
        )
        cache_store = CacheStore(tmp_path / "cache")
        meter = CostMeter("run_vid", cap_usd=5.0)
        vp = VideoProvider(
            http_client=client,
            media_store=MediaStore(tmp_path / "media"),
            clip_cache=cache_store.clip_cache,
            cost_meter=meter,
            usd_per_sec=0.07,
            min_seconds=5.0,
            sleep_fn=fake_sleep,
        )
        shot = Shot(
            id="s_short",
            index=0,
            beat="hook",
            duration_sec=2.0,  # Below 5.0s min_seconds floor
            visual_prompt="Crisp sunlit studio shot",
        )
        try:
            clip1 = await vp.generate_clip(shot, aspect="16:9", fps=24)
            calls_after_first = network_calls
            clip2 = await vp.generate_clip(shot, aspect="16:9", fps=24)
        finally:
            await client.aclose()

        # P6 progressive backoff verified: 2.0 -> 3.0 -> 5.0 -> 5.0
        assert sleeps == [2.0, 3.0, 5.0, 5.0]
        # BUG-14 retry recovered after 502 HTML
        assert submit_attempts == 2
        # BUG-12 min_seconds billing floor (max(2.0, 5.0) * 0.07 = 0.35)
        assert clip1.usd == pytest.approx(0.35, abs=1e-6)
        assert meter.spent_usd == pytest.approx(0.35, abs=1e-6)
        assert meter.reserved_usd == 0.0
        # P4 clipCache hit avoids network and cost
        assert clip2.cache_hit is True
        assert clip2.usd == 0.0
        assert clip2.path == clip1.path
        assert network_calls == calls_after_first

    async def test_bug05_cancellation_mid_poll_releases_cost_reservation(
        self,
        tmp_path: Path,
    ) -> None:
        token = CancellationToken()
        polls = 0

        async def fake_sleep(_: float) -> None:
            if polls >= 1:
                token.cancel("User aborted render")

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal polls
            if request.method == "POST":
                return httpx.Response(
                    200,
                    json={
                        "request_id": "req_cancel",
                        "status_url": "https://queue.fal.run/m/requests/req_cancel/status",
                        "response_url": "https://queue.fal.run/m/requests/req_cancel",
                    },
                )
            polls += 1
            return httpx.Response(200, json={"status": "IN_PROGRESS"})

        client = ResilientHttpClient(transport=httpx.MockTransport(handler))
        meter = CostMeter("run_cancel", cap_usd=5.0)
        vp = VideoProvider(
            http_client=client,
            media_store=MediaStore(tmp_path / "media"),
            cost_meter=meter,
            sleep_fn=fake_sleep,
        )
        shot = Shot(id="s1", index=0, beat="hook", duration_sec=4.0, visual_prompt="Cancel test")
        try:
            with pytest.raises(asyncio.CancelledError):
                await vp.generate_clip(shot, aspect="16:9", fps=24, cancel=token)
        finally:
            await client.aclose()

        # CostMeter reservation must be completely released
        assert meter.reserved_usd == 0.0
        assert meter.spent_usd == 0.0

    async def test_parse_reported_cost_rejects_bool_and_non_finite_floats(
        self,
        tmp_path: Path,
    ) -> None:
        assert VideoProvider._parse_reported_cost({"usd": False}) is None
        assert VideoProvider._parse_reported_cost({"usd": True}) is None
        assert VideoProvider._parse_reported_cost({"cost_usd": float("inf")}) is None
        assert VideoProvider._parse_reported_cost({"total_cost": float("-inf")}) is None
        assert VideoProvider._parse_reported_cost({"usd": float("nan")}) is None
        assert VideoProvider._parse_reported_cost({"usd": -1.0}) is None
        assert VideoProvider._parse_reported_cost({"usd": 0.0}) == 0.0
        assert VideoProvider._parse_reported_cost({"cost_usd": 0.42}) == 0.42

        async def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "POST":
                return httpx.Response(
                    200,
                    json={"video": {"url": "https://cdn.example.com/clip.mp4"}, "usd": False},
                )
            return httpx.Response(200, content=b"MP4_BYTES")

        client = ResilientHttpClient(transport=httpx.MockTransport(handler))
        meter = CostMeter("run_bool_cost", cap_usd=5.0)
        vp = VideoProvider(
            http_client=client,
            media_store=MediaStore(tmp_path / "media"),
            cost_meter=meter,
            usd_per_sec=0.07,
            min_seconds=5.0,
        )
        shot = Shot(id="s_bool", index=0, beat="hook", duration_sec=2.0, visual_prompt="Bool test")
        try:
            clip = await vp.generate_clip(shot, aspect="16:9", fps=24)
        finally:
            await client.aclose()

        assert clip.usd == pytest.approx(0.35, abs=1e-6)
        assert meter.spent_usd == pytest.approx(0.35, abs=1e-6)
        assert meter.reserved_usd == 0.0


# ===========================================================================
# 4. F10 — AudioProvider (BUG-04, P6, P4, BUG-12)
# ===========================================================================


class TestAudioProvider:
    def test_bug04_typed_payload_builders_and_speakable(self) -> None:
        assert "so-LAH-rah" in speakable(
            "SOLARA saves 20% at solara.io",
            pronunciations={"SOLARA": "so-LAH-rah"},
        )
        assert "twenty percent" in speakable("SOLARA saves 20% at solara.io")
        assert "solara dot i o" in speakable("SOLARA saves 20% at solara.io")
        # Backslashes in pronunciation replacements must be treated literally
        assert r"so\1lah" in speakable("Try SOLARA", pronunciations={"SOLARA": r"so\1lah"})

        # TTS payloads
        el = build_tts_payload(
            "fal-ai/elevenlabs/tts/eleven-v3",
            "Save $20 today",
            "Aria",
        )
        assert el == {"text": "Save twenty dollars today", "voice": "Aria", "stability": 0.5}

        cb = build_tts_payload("fal-ai/chatterbox/text-to-speech", "Hello world", "Aria")
        assert "voice" not in cb
        assert "stability" not in cb
        assert cb["text"] == "Hello world"

        mm = build_tts_payload("fal-ai/minimax/speech-02-hd", "Hello world", "v_minimax")
        assert mm["output_format"] == "url"
        assert mm["voice_id"] == "v_minimax"

        # Music payloads
        cue = MusicCue(
            prompt="Warm indie acoustic guitar",
            tags=["acoustic", "warm"],
            duration_sec=15.0,
        )
        sa = build_music_payload("fal-ai/stable-audio-25/text-to-audio", cue)
        assert sa == {"prompt": "Warm indie acoustic guitar", "seconds_total": 15}

        el_m = build_music_payload("elevenlabs/music/v2.5", cue)
        assert el_m == {
            "prompt": "Warm indie acoustic guitar",
            "music_length_ms": 15000,
            "force_instrumental": True,
        }

        ace = build_music_payload("fal-ai/ace-step", cue)
        assert ace == {
            "tags": "acoustic, warm",
            "duration": 15.0,
            "lyrics": "[instrumental]",
        }

        # SFX payloads & deprecated mmaudio-v2 rejection
        assert "fal-ai/mmaudio-v2" not in SFX_FALLBACK_CHAIN
        assert SFX_FALLBACK_CHAIN == (
            "mirelo-ai/sfx2.0/text-to-audio",
            "elevenlabs/sound-effects/v3",
            "sonilo/v1.1/text-to-sound-effects",
        )
        with pytest.raises(AudioProviderError, match="mmaudio-v2"):
            build_sfx_payload("fal-ai/mmaudio-v2", "whoosh", 2.0)

        assert build_sfx_payload("mirelo-ai/sfx2.0/text-to-audio", "pour", 2.0) == {
            "text_prompt": "pour",
            "duration": 2.0,
        }
        assert build_sfx_payload("elevenlabs/sound-effects/v3", "pour", 2.0) == {
            "text": "pour",
            "duration_seconds": 2.0,
        }
        assert build_sfx_payload("sonilo/v1.1/text-to-sound-effects", "pour", 2.0) == {
            "prompt": "pour",
            "duration": 2.0,
        }

    def test_extract_audio_url_handles_dict_and_array_shapes(self) -> None:
        assert extract_audio_url({"audio": {"url": "https://a.test/1.wav"}}) == "https://a.test/1.wav"
        assert extract_audio_url({"audio": [{"url": "https://a.test/2.wav"}]}) == (
            "https://a.test/2.wav"
        )
        assert extract_audio_url({"audio_file": {"url": "https://a.test/3.wav"}}) == (
            "https://a.test/3.wav"
        )

    async def test_sfx_fallback_chain_mirelo_to_elevenlabs_to_sonilo(self, tmp_path: Path) -> None:
        tried_models: list[str] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            if request.method == "POST":
                model_part = url.split("queue.fal.run/", 1)[1]
                tried_models.append(model_part)
                if "mirelo" in model_part:
                    return httpx.Response(500, text="mirelo down")
                if "elevenlabs" in model_part:
                    return httpx.Response(200, json={"status": "FAILED", "error": "quota"})
                if "sonilo" in model_part:
                    # Array shape audio[0].url
                    return httpx.Response(
                        200,
                        json={"audio": [{"url": "https://cdn.example.com/sonilo.wav"}]},
                    )
            if url == "https://cdn.example.com/sonilo.wav":
                return httpx.Response(200, content=b"RIFF_SONILO_WAV")
            return httpx.Response(404)

        client = ResilientHttpClient(
            transport=httpx.MockTransport(handler),
            max_retries=0,
        )
        meter = CostMeter("run_sfx", cap_usd=2.0)
        ap = AudioProvider(
            http_client=client,
            media_store=MediaStore(tmp_path / "media"),
            cost_meter=meter,
        )
        cue = SfxCue(shot_id="s1", at_sec=0.5, duration_sec=2.0, prompt="Crisp glass clink")
        try:
            rendered = await ap.generate_sfx(cue)
        finally:
            await client.aclose()

        assert tried_models == [
            "mirelo-ai/sfx2.0/text-to-audio",
            "elevenlabs/sound-effects/v3",
            "sonilo/v1.1/text-to-sound-effects",
        ]
        assert rendered.kind == "sfx"
        assert Path(rendered.path).is_file()
        assert meter.reserved_usd == 0.0
        assert meter.spent_usd > 0.0

    async def test_audio_p6_backoff_and_p4_audio_cache_hit(self, tmp_path: Path) -> None:
        sleeps: list[float] = []
        polls = 0
        network_calls = 0

        async def fake_sleep(sec: float) -> None:
            sleeps.append(sec)

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal polls, network_calls
            network_calls += 1
            url = str(request.url)
            if request.method == "POST":
                return httpx.Response(
                    200,
                    json={
                        "request_id": "a_req",
                        "status_url": "https://queue.fal.run/v/requests/a_req/status",
                        "response_url": "https://queue.fal.run/v/requests/a_req",
                    },
                )
            if url.endswith("/status"):
                polls += 1
                if polls < 5:
                    return httpx.Response(200, json={"status": "IN_QUEUE"})
                return httpx.Response(200, json={"status": "COMPLETED"})
            if url.endswith("/requests/a_req"):
                return httpx.Response(
                    200,
                    json={"audio": {"url": "https://cdn.example.com/vo.mp3"}},
                )
            if url == "https://cdn.example.com/vo.mp3":
                return httpx.Response(200, content=b"ID3_MP3_BYTES")
            return httpx.Response(404)

        client = ResilientHttpClient(transport=httpx.MockTransport(handler))
        cache_store = CacheStore(tmp_path / "cache")
        meter = CostMeter("run_aud", cap_usd=2.0)
        ap = AudioProvider(
            http_client=client,
            media_store=MediaStore(tmp_path / "media"),
            audio_cache=cache_store.audio_cache,
            cost_meter=meter,
            sleep_fn=fake_sleep,
        )
        cue = VoiceCue(shot_id="s1", at_sec=0.0, duration_sec=3.0, text="Fresh ginger hits.")
        try:
            a1 = await ap.generate_voice(cue, voice_id="Aria")
            calls_after_first = network_calls
            spent_after_first = meter.spent_usd
            a2 = await ap.generate_voice(cue, voice_id="Aria")
        finally:
            await client.aclose()

        # P6 audio backoff: 0.5 -> 1.0 -> 1.5 -> 2.0 -> 3.0
        assert sleeps == [0.5, 1.0, 1.5, 2.0, 3.0]
        # P4 audioCache hit avoids network and cost
        assert a1.cache_hit is False
        assert a2.cache_hit is True
        assert a2.usd == 0.0
        assert a2.path == a1.path
        assert network_calls == calls_after_first
        assert meter.spent_usd == spent_after_first


# ===========================================================================
# 5. F10 — speech_canon & WhisperProvider
# ===========================================================================


class TestSpeechCanonAndWhisper:
    def test_speech_canon_vocabulary_levenshtein_and_brand_matching(self) -> None:
        vocab = asr_vocabulary(
            "Solara Botanics",
            facts=["Order at solarabotanics.io for 20% off", "Crafted in Portland"],
            extra_terms=["Meyer lemon"],
        )
        assert "Solara Botanics" in vocab
        assert "solarabotanics.io" in vocab
        assert "Portland" in vocab
        assert "Meyer lemon" in vocab

        assert levenshtein_distance("solara", "solara") == 0
        assert levenshtein_distance("solara", "salara") == 1
        assert canonicalize_speech("Save $20 today!") == "save twenty dollars today"

        assert same_canon("Save $20 at solara.io", "save twenty dollars at solara dot i o")
        assert same_canon(
            "Try Solara Botanics today",
            "try so la ra botanics today",
            brand_terms=["Solara", "Solara Botanics"],
        )
        assert not same_canon(
            "Fresh ginger hits different",
            "completely unrelated sentence about cars",
        )

    def test_same_canon_rejects_factual_mismatches_and_preserves_words_before_brand(self) -> None:
        # Strict default (max_edit_ratio=0.0): factual/numeric/negation errors MUST fail
        assert not same_canon("Only $49.99 today", "Only $99.99 today")
        assert not same_canon("Get 2x faster builds", "Get 5x faster builds")
        assert not same_canon("Ships in 48 hours nationwide", "Ships in 98 hours nationwide")
        assert not same_canon("Do not miss out on this deal", "Do miss out on this deal")

        # 1-token lookahead prevents swallowing preceding ordinary word ('at')
        assert _align_brand_terms("shop at drinkolipop", ["drinkolipop"]) == "shop at drinkolipop"
        # Dropping 'at' in actual transcript must fail because expected is kept verbatim
        assert not same_canon(
            "shop at drinkolipop",
            "shop drinkolipop",
            brand_terms=["drinkolipop"],
        )
        # Split brand tokens in actual transcript still align cleanly while preserving 'at'
        assert same_canon(
            "shop at drinkolipop",
            "shop at drink olipop",
            brand_terms=["drinkolipop"],
        )

    def test_short_brand_alignment_and_same_canon_preserves_common_words(self) -> None:
        assert _align_brand_terms("try a new ark browser", ["Arc"]) == "try a new arc browser"
        assert same_canon("Try a new Arc browser", "Try a new Ark browser", brand_terms=["Arc"])
        assert _align_brand_terms("we love our own car", ["Oura"]) == "we love our own car"
        assert same_canon("We love Oura ring", "We love Aura ring", brand_terms=["Oura"])
        assert _align_brand_terms("try aura or a band", ["Oura"]) == "try oura or a band"
        assert same_canon("Try Oura or a band", "Try Aura or a band", brand_terms=["Oura"]) is True
        assert (
            same_canon("Buy a ring Oura band", "Buy a ring or a band", brand_terms=["Oura"])
            is False
        )

    async def test_whisper_explicit_local_model_fails_loudly_without_cloud_fallback(
        self,
        tmp_path: Path,
    ) -> None:
        audio_file = tmp_path / "sample.wav"
        audio_file.write_bytes(b"RIFF_DUMMY")

        # Even with FAL_KEY set, explicit missing WHISPER_MODEL_PATH MUST raise WhisperError!
        wp = WhisperProvider(
            whisper_model_path=tmp_path / "nonexistent_ggml.bin",
            fal_key="fal_secret_should_not_be_used",
        )
        with pytest.raises(WhisperError, match="WHISPER_MODEL_PATH"):
            await wp.transcribe(audio_file)

        # Model exists, but binary missing -> also fails loudly with no fallback
        fake_model = tmp_path / "ggml-base.bin"
        fake_model.write_bytes(b"MODEL")
        wp_missing_bin = WhisperProvider(
            whisper_model_path=fake_model,
            whisper_bin="nonexistent-whisper-cli-xyz",
            fal_key="fal_secret_should_not_be_used",
        )
        with pytest.raises(WhisperError, match="not found"):
            await wp_missing_bin.transcribe(audio_file)

    async def test_whisper_local_cli_invocation_passes_vocabulary(self, tmp_path: Path) -> None:
        audio_file = tmp_path / "sample.wav"
        audio_file.write_bytes(b"RIFF_DUMMY")
        fake_model = tmp_path / "ggml-base.bin"
        fake_model.write_bytes(b"MODEL")
        fake_bin = tmp_path / "whisper-cli"
        fake_bin.write_text("#!/bin/sh\n", encoding="utf-8")

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"Solara ginger tonic.\n", b""))

        wp = WhisperProvider(
            whisper_model_path=fake_model,
            whisper_bin=str(fake_bin),
        )
        with patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
            text = await wp.transcribe(audio_file, vocabulary=["Solara", "Botanics"])

        assert text == "Solara ginger tonic."
        called_args = mock_exec.call_args.args
        assert "--prompt" in called_args
        assert "Solara, Botanics" in called_args

    async def test_whisper_fal_cloud_path_and_unconfigured_error(self, tmp_path: Path) -> None:
        audio_file = tmp_path / "sample.wav"
        audio_file.write_bytes(b"RIFF_WAV_DATA")

        # Unconfigured raises WhisperError
        wp_none = WhisperProvider(settings=Settings(whisper_model_path="", fal_key=""))
        with pytest.raises(WhisperError, match="not configured"):
            await wp_none.transcribe(audio_file)

        # fal.ai Whisper path
        async def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "POST":
                body = json.loads(request.content)
                assert body["audio_url"].startswith("data:audio/wav;base64,")
                assert body["prompt"] == "Solara"
                return httpx.Response(200, json={"text": "Fresh ginger hits different."})
            return httpx.Response(404)

        client = ResilientHttpClient(transport=httpx.MockTransport(handler))
        meter = CostMeter("run_whisper", cap_usd=1.0)
        wp_fal = WhisperProvider(
            http_client=client,
            cost_meter=meter,
            whisper_model_path="",
            fal_key="test_fal_key",
        )
        try:
            transcript = await wp_fal.transcribe(audio_file, vocabulary=["Solara"])
        finally:
            await client.aclose()

        assert transcript == "Fresh ginger hits different."
        assert meter.spent_usd == pytest.approx(0.002, abs=1e-6)


# ===========================================================================
# 6. F10 — Offline Mocks (ScriptedChatModel, Real FFmpeg MP4 & WAV, Bundle)
# ===========================================================================


class _StructuredSample(BaseModel):
    headline: str
    score: int


class TestMocks:
    async def test_scripted_chat_model_sync_async_tools_and_structured_output(self) -> None:
        model = ScriptedChatModel(
            responses=[
                "hello world",
                AIMessage(content="async reply"),
                {"headline": "Crisp Ginger", "score": 9},
            ]
        )
        r1 = model.invoke([HumanMessage(content="hi")])
        assert r1.content == "hello world"

        r2 = await model.ainvoke([HumanMessage(content="hi again")])
        assert r2.content == "async reply"

        structured = model.with_structured_output(_StructuredSample)
        parsed = await structured.ainvoke([HumanMessage(content="give json")])
        assert isinstance(parsed, _StructuredSample)
        assert parsed.headline == "Crisp Ginger"
        assert parsed.score == 9

        # bind_tools shares cursor & works with tool definitions
        @tool
        def ping(msg: str) -> str:
            """Echo ping."""
            return f"pong:{msg}"

        tool_model = ScriptedChatModel(
            responses=[
                {
                    "content": "",
                    "tool_calls": [
                        {"id": "tc_1", "name": "ping", "args": {"msg": "ok"}, "type": "tool_call"}
                    ],
                },
                "done",
            ]
        )
        bound = tool_model.bind_tools([ping])
        msg_tc = bound.invoke([HumanMessage(content="call ping")])
        assert len(msg_tc.tool_calls) == 1
        assert msg_tc.tool_calls[0]["name"] == "ping"
        msg_done = bound.invoke([HumanMessage(content="finish")])
        assert msg_done.content == "done"
        assert tool_model.call_count == 2

    def test_scripted_chat_model_compatible_with_deepagents_and_rubric(self) -> None:
        from deepagents import RubricMiddleware, create_deep_agent

        evaluations: list[dict[str, Any]] = []
        main_llm = ScriptedChatModel(responses=["Completed brand concept draft."])
        grader_llm = ScriptedChatModel(
            responses=[
                {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "tc_grader_1",
                            "name": "GraderResponse",
                            "args": {
                                "result": "satisfied",
                                "explanation": "Draft meets rubric criteria.",
                                "criteria": [
                                    {"name": "Produces a brand concept draft", "passed": True}
                                ],
                            },
                            "type": "tool_call",
                        }
                    ],
                }
            ]
        )
        middleware = RubricMiddleware(
            model=grader_llm,
            system_prompt="Grade the draft.",
            max_iterations=1,
            on_evaluation=lambda ev: evaluations.append(dict(ev)),
        )
        agent = create_deep_agent(model=main_llm, middleware=[middleware])
        result = agent.invoke(
            {
                "messages": [HumanMessage(content="Draft a concept")],
                "rubric": "Produces a brand concept draft.",
            }
        )
        assert "messages" in result
        assert main_llm.call_count == 1
        assert grader_llm.call_count == 1
        assert len(evaluations) == 1
        assert evaluations[0]["result"] == "satisfied"

    @pytest.mark.parametrize("aspect", ["16:9", "9:16", "1:1", "4:5"])
    async def test_mock_video_provider_generates_real_mp4_verified_with_ffprobe(
        self,
        tmp_path: Path,
        aspect: Any,
    ) -> None:
        cache_store = CacheStore(tmp_path / "cache")
        vp = MockVideoProvider(
            media_store=MediaStore(tmp_path / "media"),
            clip_cache=cache_store.clip_cache,
        )
        shot = Shot(
            id=f"s_{aspect.replace(':', '_')}",
            index=0,
            beat="hook",
            duration_sec=1.0,
            visual_prompt="Synthetic clip test",
        )
        clip = await vp.generate_clip(shot, aspect=aspect, fps=24)
        clip_path = Path(clip.path)
        assert clip_path.is_file()
        assert clip_path.stat().st_size > 0

        info = _ffprobe_json(clip_path)
        v_streams = [s for s in info["streams"] if s["codec_type"] == "video"]
        assert len(v_streams) == 1
        expected_w, expected_h = MOCK_ASPECT_DIMENSIONS[aspect]
        assert int(v_streams[0]["width"]) == expected_w
        assert int(v_streams[0]["height"]) == expected_h
        assert v_streams[0]["r_frame_rate"] == "24/1"
        duration = float(info["format"]["duration"])
        assert abs(duration - 1.0) <= 0.25

        # Second call hits clipCache
        clip_cached = await vp.generate_clip(shot, aspect=aspect, fps=24)
        assert clip_cached.cache_hit is True
        assert clip_cached.path == clip.path

    async def test_mock_audio_provider_generates_real_wav_verified_with_ffprobe(
        self,
        tmp_path: Path,
    ) -> None:
        ap = MockAudioProvider(media_store=MediaStore(tmp_path / "media"))
        vo = await ap.generate_voice(
            VoiceCue(shot_id="s1", at_sec=0.0, duration_sec=0.8, text="Hello world"),
            voice_id="Aria",
        )
        sfx = await ap.generate_sfx(
            SfxCue(shot_id="s1", at_sec=0.1, duration_sec=0.5, prompt="Snap"),
        )
        music = await ap.generate_music(
            MusicCue(prompt="Acoustic bed", duration_sec=1.0),
        )

        for rendered, expected_dur in ((vo, 0.8), (sfx, 0.5), (music, 1.0)):
            p = Path(rendered.path)
            assert p.is_file()
            info = _ffprobe_json(p)
            a_streams = [s for s in info["streams"] if s["codec_type"] == "audio"]
            assert len(a_streams) == 1
            dur = float(info["format"]["duration"])
            assert abs(dur - expected_dur) <= 0.15

    async def test_mock_brand_scraper_whisper_and_bundle(self, tmp_path: Path) -> None:
        bundle = MockProviderBundle.create(
            tmp_path,
            responses=["ok"],
            transcripts=["Fresh ginger hits different."],
        )
        research = await bundle.scraper.scrape_brand(
            "https://solarabotanics.example.com",
            brand_name="Solara Botanics",
        )
        assert research.brand_name == "Solara Botanics"
        assert len(research.product_photos) == 2
        assert Path(research.product_photos[0].path or "").is_file()

        wav_path = tmp_path / "dummy.wav"
        wav_path.write_bytes(b"RIFF")
        assert await bundle.whisper.transcribe(wav_path) == "Fresh ginger hits different."
        assert isinstance(bundle.scraper, MockBrandScraper)
        assert isinstance(bundle.whisper, MockWhisperProvider)

    async def test_mock_concurrent_sfx_shot_repair_and_voice_duration_isolation(
        self,
        tmp_path: Path,
    ) -> None:
        cache_store = CacheStore(tmp_path / "cache")
        media_store = MediaStore(tmp_path / "media")

        # 1. Concurrent multiple SfxCues on the same shot_id produce distinct valid WAVs
        ap = MockAudioProvider(media_store=media_store, audio_cache=cache_store.audio_cache)
        sfx_a = SfxCue(shot_id="s1", at_sec=0.1, duration_sec=0.4, prompt="Crisp glass clink")
        sfx_b = SfxCue(shot_id="s1", at_sec=0.6, duration_sec=0.5, prompt="Fizzy citrus pour")
        res_a, res_b = await asyncio.gather(ap.generate_sfx(sfx_a), ap.generate_sfx(sfx_b))

        assert res_a.path != res_b.path
        for rendered in (res_a, res_b):
            p = Path(rendered.path)
            assert p.is_file() and p.stat().st_size > 0
            info = _ffprobe_json(p)
            assert any(s["codec_type"] == "audio" for s in info["streams"])

        # 2. Voice cues with identical text but different duration_sec do not alias on cache hits
        vo_short = await ap.generate_voice(
            VoiceCue(shot_id="s1", at_sec=0.0, duration_sec=0.4, text="Same line"),
            voice_id="Aria",
        )
        vo_long = await ap.generate_voice(
            VoiceCue(shot_id="s1", at_sec=0.0, duration_sec=1.0, text="Same line"),
            voice_id="Aria",
        )
        assert vo_long.cache_hit is False
        assert vo_short.path != vo_long.path
        assert abs(float(_ffprobe_json(Path(vo_short.path))["format"]["duration"]) - 0.4) <= 0.15
        assert abs(float(_ffprobe_json(Path(vo_long.path))["format"]["duration"]) - 1.0) <= 0.15

        # 3. Shot repair re-render with modified visual_prompt produces distinct MP4 without clobber
        vp = MockVideoProvider(media_store=media_store, clip_cache=cache_store.clip_cache)
        shot_v1 = Shot(
            id="s_repair",
            index=0,
            beat="hook",
            duration_sec=0.8,
            visual_prompt="Sunlit table version one",
        )
        shot_v2 = Shot(
            id="s_repair",
            index=0,
            beat="hook",
            duration_sec=0.8,
            visual_prompt="Sunlit table version two repaired",
        )
        clip_v1 = await vp.generate_clip(shot_v1, aspect="16:9", fps=24)
        v1_bytes = Path(clip_v1.path).read_bytes()
        clip_v2 = await vp.generate_clip(shot_v2, aspect="16:9", fps=24)

        assert clip_v2.cache_hit is False
        assert clip_v1.path != clip_v2.path
        assert Path(clip_v1.path).is_file()
        assert Path(clip_v1.path).read_bytes() == v1_bytes
        assert Path(clip_v2.path).is_file() and Path(clip_v2.path).stat().st_size > 0

    async def test_mock_video_and_audio_providers_work_with_cache_disabled(
        self,
        tmp_path: Path,
    ) -> None:
        media_store = MediaStore(tmp_path / "media")

        vp = MockVideoProvider(media_store=media_store, clip_cache=None)
        shot_a = Shot(
            id="s_nocache",
            index=0,
            beat="hook",
            duration_sec=0.8,
            visual_prompt="Morning light on glass bottle",
        )
        shot_b = Shot(
            id="s_nocache",
            index=0,
            beat="hook",
            duration_sec=0.8,
            visual_prompt="Evening studio rim light on glass bottle",
        )
        clip_a = await vp.generate_clip(shot_a, aspect="9:16", fps=30)
        clip_b = await vp.generate_clip(shot_b, aspect="9:16", fps=30)

        assert clip_a.cache_hit is False
        assert clip_b.cache_hit is False
        assert clip_a.path != clip_b.path
        for clip in (clip_a, clip_b):
            p = Path(clip.path)
            assert p.is_file() and p.stat().st_size > 0
            info = _ffprobe_json(p)
            assert any(s["codec_type"] == "video" for s in info["streams"])

        ap = MockAudioProvider(media_store=media_store, audio_cache=None)
        vo_a = await ap.generate_voice(
            VoiceCue(shot_id="s_nocache", at_sec=0.0, duration_sec=0.5, text="First line"),
            voice_id="Aria",
        )
        vo_b = await ap.generate_voice(
            VoiceCue(shot_id="s_nocache", at_sec=0.0, duration_sec=0.5, text="Second line"),
            voice_id="Aria",
        )
        assert vo_a.cache_hit is False
        assert vo_b.cache_hit is False
        assert vo_a.path != vo_b.path
        for rendered in (vo_a, vo_b):
            p = Path(rendered.path)
            assert p.is_file() and p.stat().st_size > 0
            info = _ffprobe_json(p)
            assert any(s["codec_type"] == "audio" for s in info["streams"])

    async def test_mock_video_camera_move_and_angle_produce_distinct_clips(
        self,
        tmp_path: Path,
    ) -> None:
        media_store = MediaStore(tmp_path / "media")
        shot_base = Shot(
            id="s_cam",
            index=0,
            beat="hook",
            duration_sec=0.8,
            visual_prompt="Sunlit glass bottle on ceramic table",
            camera_move="static",
            camera_angle="eye level",
        )
        shot_move = shot_base.model_copy(update={"camera_move": "slow dolly in"})
        shot_angle = shot_base.model_copy(update={"camera_angle": "low angle hero"})

        # Uncached: distinct MP4 filenames on the same shot.id
        vp_nocache = MockVideoProvider(media_store=media_store, clip_cache=None)
        c_base = await vp_nocache.generate_clip(shot_base, aspect="16:9", fps=24)
        c_move = await vp_nocache.generate_clip(shot_move, aspect="16:9", fps=24)
        c_angle = await vp_nocache.generate_clip(shot_angle, aspect="16:9", fps=24)
        assert len({c_base.path, c_move.path, c_angle.path}) == 3
        for clip in (c_base, c_move, c_angle):
            assert Path(clip.path).is_file() and Path(clip.path).stat().st_size > 0

        # Cached: changing camera_move or camera_angle causes a cache miss and distinct artifact
        cache_store = CacheStore(tmp_path / "cache")
        vp_cached = MockVideoProvider(
            media_store=MediaStore(tmp_path / "media_cached"),
            clip_cache=cache_store.clip_cache,
        )
        cc_base = await vp_cached.generate_clip(shot_base, aspect="16:9", fps=24)
        cc_base_hit = await vp_cached.generate_clip(shot_base, aspect="16:9", fps=24)
        cc_move = await vp_cached.generate_clip(shot_move, aspect="16:9", fps=24)
        cc_angle = await vp_cached.generate_clip(shot_angle, aspect="16:9", fps=24)
        assert cc_base.cache_hit is False
        assert cc_base_hit.cache_hit is True
        assert cc_base_hit.path == cc_base.path
        assert cc_move.cache_hit is False
        assert cc_angle.cache_hit is False
        assert len({cc_base.path, cc_move.path, cc_angle.path}) == 3
