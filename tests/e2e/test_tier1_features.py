"""Tier 1 E2E Tests: Isolated happy-path coverage for every feature F1-F18 & F21."""

from __future__ import annotations

import asyncio
import base64
import math
from pathlib import Path

import httpx
import pytest

from gtm_videogen.core.cache_store import CacheStore
from gtm_videogen.core.config import Settings, get_settings
from gtm_videogen.core.cost_meter import CostMeter, estimate_run_cost
from gtm_videogen.core.http_client import ResilientHttpClient
from gtm_videogen.core.job_registry import CancellationToken, JobRegistry
from gtm_videogen.core.media_store import MediaStore
from gtm_videogen.core.run_store import RunStore
from gtm_videogen.core.ssrf import assert_public_url
from gtm_videogen.models.schemas import (
    ConceptPitch,
    CostLedger,
    CreativeBible,
    CreativeFingerprint,
    JobStatusDTO,
    QAReport,
    RenderedAudio,
    RenderedClip,
    RenderManifest,
    ResearchBundle,
    RunInput,
    RunState,
    Shot,
    ShotSheet,
    SoundPlan,
    VoiceCasting,
)
from gtm_videogen.rules.anti_garbled_text import rendered_text_problem
from gtm_videogen.rules.claim_grounding import check_claim_grounding
from gtm_videogen.rules.creative_history import CreativeHistoryStore
from gtm_videogen.rules.diversity import diversity_scores
from gtm_videogen.rules.script_gate import evaluate_script_gate
from gtm_videogen.rules.speakability import spoken_form
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
    repo_root_path,
)

pytestmark = [pytest.mark.e2e, pytest.mark.tier1]


# ---------------------------------------------------------------------------
# M1 Features: F1 - F7
# ---------------------------------------------------------------------------


@pytest.mark.m1
def test_f1_bundled_fonts_and_settings_load(tmp_path: Path) -> None:
    """F1: 5 open-license font families (10 TTFs) exist in assets/fonts/ & Settings loads."""
    fonts_dir = repo_root_path() / "assets" / "fonts"
    assert fonts_dir.is_dir(), f"Missing bundled fonts directory: {fonts_dir}"

    expected_ttfs = {
        "Roboto-Regular.ttf",
        "Roboto-Bold.ttf",
        "RobotoCondensed-Regular.ttf",
        "RobotoCondensed-Bold.ttf",
        "LiberationSerif-Regular.ttf",
        "LiberationSerif-Bold.ttf",
        "LiberationMono-Regular.ttf",
        "LiberationMono-Bold.ttf",
        "LiberationSans-Regular.ttf",
        "LiberationSans-Bold.ttf",
    }
    actual_ttfs = {p.name for p in fonts_dir.glob("*.ttf")}
    assert expected_ttfs.issubset(actual_ttfs), (
        f"Missing bundled font files: {expected_ttfs - actual_ttfs}"
    )

    settings = Settings(
        data_dir=tmp_path / "data",
        runs_dir=tmp_path / "runs",
        cache_dir=tmp_path / "cache",
        fonts_dir=fonts_dir,
    )
    settings.ensure_dirs()
    assert settings.data_dir.is_dir()
    assert settings.runs_dir.is_dir()
    assert settings.cache_dir.is_dir()
    assert get_settings().max_concurrent_jobs >= 1


@pytest.mark.m1
def test_f2_schema_json_round_trip() -> None:
    """F2: All core Pydantic v2 schemas round-trip cleanly through JSON serialization."""
    run_input = make_run_input()
    research = make_research_bundle()
    bible = make_creative_bible()
    sheet = make_shot_sheet()
    sound_plan = make_sound_plan()
    manifest = RenderManifest(
        run_id="r_f2",
        clips=[
            RenderedClip(
                shot_id="s1",
                path="/tmp/s1.mp4",
                duration_sec=3.0,
                fps=24,
                width=1280,
                height=720,
            )
        ],
        audio_assets=[
            RenderedAudio(kind="vo", cue_id="s1", path="/tmp/s1.wav", duration_sec=2.8)
        ],
        stitched_path="/tmp/final.mp4",
    )
    ledger = CostLedger(run_id="r_f2", cap_usd=5.0, estimated_usd=1.5, actual_usd=1.2)
    qa = QAReport(passed=True, originality_score=8.5, bible_coherence_score=9.0)
    run_state = make_run_state(
        run_id="r_f2",
        run_input=run_input,
        research=research,
        creative_bible=bible,
        shot_sheet=sheet,
        sound_plan=sound_plan,
        render_manifest=manifest,
        cost_ledger=ledger,
        qa_report=qa,
    )
    job_dto = JobStatusDTO(id="job_f2", run_id="r_f2", stage="render", state="running")

    models: list[tuple[type, object]] = [
        (RunInput, run_input),
        (ResearchBundle, research),
        (CreativeBible, bible),
        (ShotSheet, sheet),
        (SoundPlan, sound_plan),
        (RenderManifest, manifest),
        (CostLedger, ledger),
        (QAReport, qa),
        (RunState, run_state),
        (JobStatusDTO, job_dto),
    ]
    for model_cls, instance in models:
        dumped = instance.model_dump_json()  # type: ignore[attr-defined]
        reloaded = model_cls.model_validate_json(dumped)
        assert reloaded == instance


@pytest.mark.m1
async def test_f3_job_registry_cancellation_and_run_store(tmp_path: Path) -> None:
    """F3: JobRegistry runs concurrent jobs, releases slots, CancellationToken & RunStore."""
    registry = JobRegistry(max_concurrent=2)
    completed: list[str] = []

    async def _worker(tag: str, token: CancellationToken) -> None:
        token.raise_if_cancelled()
        await asyncio.sleep(0.02)
        completed.append(tag)

    j1 = await registry.submit("run_a", lambda tok: _worker("a", tok), stage="research")
    j2 = await registry.submit("run_b", lambda tok: _worker("b", tok), stage="director")
    res1 = await registry.wait_for_job(j1.id)
    res2 = await registry.wait_for_job(j2.id)

    assert res1 is not None and res1.state == "done"
    assert res2 is not None and res2.state == "done"
    assert set(completed) == {"a", "b"}
    assert registry.active_count == 0

    store = RunStore(tmp_path / "runs", job_registry=registry)
    state = make_run_state(run_id="run_f3", status="done")
    saved_path = store.save(state)
    assert saved_path.exists()
    loaded = store.load("run_f3")
    assert loaded is not None
    assert loaded.run_id == "run_f3"
    assert loaded.status == "done"


@pytest.mark.m1
async def test_f4_resilient_http_retry_and_public_url() -> None:
    """F4: ResilientHttpClient retries 503 then parses JSON; assert_public_url allows HTTPS."""
    url = assert_public_url("https://93.184.216.34/api/ping", resolve_dns=False)
    assert url.startswith("https://")

    attempts = {"count": 0}

    def _handler(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        if attempts["count"] == 1:
            return httpx.Response(503, text="Service Unavailable")
        return httpx.Response(200, json={"status": "ok", "attempt": attempts["count"]})

    client = ResilientHttpClient(
        max_retries=2,
        backoff_base_sec=0.01,
        backoff_max_sec=0.02,
        transport=httpx.MockTransport(_handler),
        resolve_dns_for_ssrf=False,
    )
    try:
        payload = await client.get_json("https://93.184.216.34/api/ping", public_only=True)
        assert payload == {"status": "ok", "attempt": 2}
        assert attempts["count"] == 2
    finally:
        await client.aclose()


@pytest.mark.m1
def test_f5_cost_meter_lifecycle_and_preflight_estimate(tmp_path: Path) -> None:
    """F5: CostMeter reserve->settle lifecycle and estimate_run_cost() breakdown."""
    ledger_path = tmp_path / "spend.jsonl"
    meter = CostMeter(run_id="run_f5", cap_usd=5.0, ledger_path=ledger_path)

    res_id = meter.reserve("clip_s1", 0.50, stage="render", provider="fal", model="kling")
    assert meter.reserved_usd == pytest.approx(0.50)
    entry = meter.settle(res_id, 0.35)
    assert entry.actual_usd == pytest.approx(0.35)
    assert meter.spent_usd == pytest.approx(0.35)
    assert meter.reserved_usd == pytest.approx(0.0)
    assert meter.state == "ok"

    sheet = make_shot_sheet()
    sound_plan = make_sound_plan()
    breakdown = estimate_run_cost(shot_sheet=sheet, sound_plan=sound_plan)
    assert breakdown["total_usd"] > 0.0
    assert breakdown["video_usd"] > 0.0
    assert breakdown["voice_usd"] > 0.0
    assert breakdown["sfx_usd"] > 0.0
    assert breakdown["music_usd"] > 0.0


@pytest.mark.m1
async def test_f6_media_store_resolution_and_cache_store(tmp_path: Path) -> None:
    """F6: MediaStore save/lookup/resolve_to_path (media://, /api/media/, data:, local) & cache."""
    store = MediaStore(tmp_path / "media", resolve_dns_for_ssrf=False)
    media_id, saved_path = store.save_bytes(b"hello-media-bytes", ext=".bin")
    assert store.lookup(media_id) == saved_path

    via_media_scheme = await store.resolve_to_path(f"media://{media_id}")
    via_api_path = await store.resolve_to_path(f"/api/media/{media_id}")
    via_local_path = await store.resolve_to_path(str(saved_path))
    b64 = base64.b64encode(b"inline-png-bytes").decode("ascii")
    via_data_url = await store.resolve_to_path(f"data:image/png;base64,{b64}")

    assert via_media_scheme == saved_path
    assert via_api_path == saved_path
    assert via_local_path == saved_path
    assert via_data_url.read_bytes() == b"inline-png-bytes"

    caches = CacheStore(tmp_path / "caches")
    key = caches.clip_cache.make_key({"shot": "s1", "prompt": "coffee beans"})
    assert caches.clip_cache.get(key) is None
    caches.clip_cache.put(key, artifact_path=saved_path, metadata={"fps": 24})
    hit = caches.clip_cache.get(key)
    assert hit is not None
    assert hit.artifact_path == saved_path
    assert hit.metadata["fps"] == 24


@pytest.mark.m1
def test_f7_pure_script_rules_happy_path(tmp_path: Path) -> None:
    """F7: Pure script rules evaluate grounded script, timing, voice, diversity & history."""
    research = make_research_bundle()
    sheet = make_shot_sheet()

    issues = check_claim_grounding(
        "Roasted to order in Seattle every Tuesday with 25% off your first two bags.",
        research,
    )
    assert issues == []

    assert "percent" in spoken_form("Save 25% today").lower()
    timings = fit_timings(sheet.shots, target_sec=12.0)
    assert sum(timings) == pytest.approx(12.0, abs=0.05)
    assert rendered_text_problem("Macro pour-over bloom in natural window light") is None

    concepts = [
        ConceptPitch(
            id="c1",
            title="Roast Day",
            thesis="Fresh beans",
            narrative_arc="hook proof payoff",
            visual_world="warm wooden roastery",
            sonic_strategy="upright bass and foley",
            emotional_register="warm grounded",
        ),
        ConceptPitch(
            id="c2",
            title="Lab Precision",
            thesis="Extraction craft",
            narrative_arc="contrast reveal cta",
            visual_world="bright ceramic pour bar",
            sonic_strategy="dry intimate voiceover",
            emotional_register="crisp curious",
        ),
    ]
    scores = diversity_scores(concepts)
    assert len(scores) == 2 and all(s > 0.0 for s in scores)

    history_store = CreativeHistoryStore(tmp_path / "history.json")
    hist = history_store.record(
        CreativeFingerprint(
            run_id="r0",
            brand_url=research.brand_url,
            arc_name="Old Arc",
            hook_opening="Old hook",
            visual_world="Old world",
            lighting_style="Old light",
            camera_style="Old cam",
            voice_id="v_aria_f_adult_us",
        )
    )
    chosen_voice = select_voice(
        VoiceCasting(gender="female", age="adult", accent="us", energy="warm", pace="natural"),
        brand_history=hist,
    )
    assert chosen_voice.voice_id != "v_aria_f_adult_us"

    gate_report = evaluate_script_gate(
        sheet,
        research,
        has_product_photos=True,
        offer="Save 25% on your first two bags",
        must_say=["Roasted to order in Seattle"],
    )
    assert gate_report.passed is True
    assert gate_report.blocking_errors == []


# ---------------------------------------------------------------------------
# M2 Features: F8 - F10 (imported inside test bodies for progressive collection)
# ---------------------------------------------------------------------------


def test_f8_llm_router_capability_matrix_and_stage_routing() -> None:
    """F8: LLMRouter filters reasoning/Anthropic params and routes sound_design to text_model."""
    from gtm_videogen.models.schemas import ChatOptions
    from gtm_videogen.providers.llm_router import LLMRouter

    router = LLMRouter()

    reasoning_opts = router.build_chat_options(
        model="o3-mini",
        stage="script_write",
        overrides=ChatOptions(temperature=0.8, top_p=0.9, max_tokens=1024),
    )
    assert "temperature" not in reasoning_opts
    assert "top_p" not in reasoning_opts
    assert reasoning_opts["max_tokens"] == 1024

    anthropic_opts = router.build_chat_options(
        model="claude-3-7-sonnet",
        stage="creative_bible",
        overrides=ChatOptions(temperature=0.7, top_p=0.9, max_tokens=2048),
    )
    assert anthropic_opts["max_tokens"] == 2048
    assert anthropic_opts["temperature"] == pytest.approx(0.7)
    assert "top_p" not in anthropic_opts
    assert "presence_penalty" not in anthropic_opts
    assert "frequency_penalty" not in anthropic_opts

    sound_model = router.resolve_model("sound_design")
    assert sound_model == router.settings.text_model
    sound_opts = router.build_chat_options(model=sound_model, stage="sound_design")
    assert "max_tokens" in sound_opts and sound_opts["max_tokens"] > 0


async def test_f9_brand_scraper_extracts_claims_and_photos() -> None:
    """F9: BrandScraper extracts og:image, JSON-LD, and claims via mocked HTTP transport."""
    from gtm_videogen.providers.scraper import BrandScraper

    html = """
    <html>
      <head>
        <title>Kestrel Roasters — Direct-Trade Seattle Coffee</title>
        <meta property="og:image" content="https://93.184.216.34/img/bag.jpg" />
        <meta name="description" content="Roasted to order in Seattle every Tuesday." />
        <script type="application/ld+json">
          {
            "@context": "https://schema.org",
            "@type": "Product",
            "name": "Huila Single-Origin",
            "description": "100% direct-trade Colombian coffee shipped within 24 hours.",
            "image": "https://93.184.216.34/img/huila.jpg"
          }
        </script>
      </head>
      <body>
        <h1>Kestrel Roasters</h1>
        <p>Roasted to order in Seattle every Tuesday. Save 25% on your first two bags.</p>
      </body>
    </html>
    """

    def _handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/products.json":
            return httpx.Response(404, text="Not Found")
        return httpx.Response(200, text=html, headers={"content-type": "text/html"})

    client = ResilientHttpClient(
        transport=httpx.MockTransport(_handler),
        resolve_dns_for_ssrf=False,
    )
    try:
        scraper = BrandScraper(http_client=client, resolve_dns_for_ssrf=False)
        bundle = await scraper.scrape_brand("https://93.184.216.34/")
        assert isinstance(bundle, ResearchBundle)
        assert len(bundle.product_photos) >= 1
        combined_text = " ".join(
            bundle.verified_claims + bundle.product_facts + bundle.proof_points
        )
        assert "Seattle" in combined_text or "Kestrel" in bundle.brand_name
    finally:
        await client.aclose()


async def test_f10_mock_provider_bundle_generates_real_media_assets(tmp_path: Path) -> None:
    """F10: MockProviderBundle produces real ffprobe-verified video clips, WAVs, and transcripts."""
    from gtm_videogen.providers.mocks import MockProviderBundle

    bundle = MockProviderBundle.create(work_dir=tmp_path)
    sheet = make_shot_sheet()
    sound_plan = make_sound_plan()

    clip = await bundle.video.generate_clip(sheet.shots[0], aspect="16:9", fps=24)
    assert Path(clip.path).exists()
    video_probe = ffprobe_inspect(Path(clip.path))
    assert any(s.get("codec_type") == "video" for s in video_probe["streams"])

    vo = await bundle.audio.generate_voice(sound_plan.vo_cues[0], sound_plan.voice_id)
    sfx = await bundle.audio.generate_sfx(sound_plan.sfx_cues[0])
    assert sound_plan.music_cue is not None
    music = await bundle.audio.generate_music(sound_plan.music_cue)

    for audio in (vo, sfx, music):
        assert Path(audio.path).exists()
        audio_probe = ffprobe_inspect(Path(audio.path))
        assert any(s.get("codec_type") == "audio" for s in audio_probe["streams"])

    transcript = await bundle.whisper.transcribe(Path(vo.path))
    assert isinstance(transcript, str) and len(transcript.strip()) > 0


# ---------------------------------------------------------------------------
# M3 Features: F11 - F13 & F21
# ---------------------------------------------------------------------------


async def test_f11_ffmpeg_stitcher_single_pass_with_audio(tmp_path: Path) -> None:
    """F11: FFmpegStitcher stitches 2 synthetic clips + VO + music into ffprobe-verified MP4."""
    from gtm_videogen.media.ffmpeg_stitcher import FFmpegStitcher

    clip1_path = make_synthetic_clip(
        tmp_path / "c1.mp4", width=640, height=360, fps=24, duration_sec=2.0
    )
    clip2_path = make_synthetic_clip(
        tmp_path / "c2.mp4", width=640, height=360, fps=24, duration_sec=2.0
    )
    vo_path = make_synthetic_wav(tmp_path / "vo.wav", duration_sec=1.8, freq_hz=440)
    music_path = make_synthetic_wav(tmp_path / "music.wav", duration_sec=4.0, freq_hz=220)

    sheet = make_shot_sheet(
        fps=24,
        aspect="16:9",
        target_duration_sec=4.0,
        shots=[
            Shot(
                id="s1",
                index=0,
                beat="hook",
                duration_sec=2.0,
                visual_prompt="Coffee beans",
                vo_line="Fresh coffee.",
            ),
            Shot(
                id="s2",
                index=1,
                beat="cta",
                duration_sec=2.0,
                visual_prompt="Coffee bag",
                vo_line="Save 25%.",
            ),
        ],
    )
    bible = make_creative_bible()
    sound_plan = make_sound_plan()
    clips = [
        RenderedClip(
            shot_id="s1", path=str(clip1_path), duration_sec=2.0, fps=24, width=640, height=360
        ),
        RenderedClip(
            shot_id="s2", path=str(clip2_path), duration_sec=2.0, fps=24, width=640, height=360
        ),
    ]
    audio_assets = [
        RenderedAudio(kind="vo", cue_id="s1", path=str(vo_path), duration_sec=1.8),
        RenderedAudio(kind="music", cue_id="bg", path=str(music_path), duration_sec=4.0),
    ]

    output_path = tmp_path / "stitched.mp4"
    stitcher = FFmpegStitcher()
    manifest = await stitcher.stitch(
        clips=clips,
        sound_plan=sound_plan,
        audio_assets=audio_assets,
        sheet=sheet,
        bible=bible,
        output_path=output_path,
    )
    assert manifest.stitched_path is not None
    assert Path(manifest.stitched_path).exists()

    probe = ffprobe_inspect(Path(manifest.stitched_path))
    v_streams = [s for s in probe["streams"] if s.get("codec_type") == "video"]
    a_streams = [s for s in probe["streams"] if s.get("codec_type") == "audio"]
    assert len(v_streams) == 1
    assert len(a_streams) == 1
    assert parse_frame_rate(v_streams[0]["r_frame_rate"]) == pytest.approx(24.0, abs=0.5)


async def test_f12_graphics_card_and_ass_captions(tmp_path: Path) -> None:
    """F12: GraphicsCardRenderer renders exact WxH@fps card & ASS captions use bundled font."""
    from gtm_videogen.media.captions import generate_ass_subtitles
    from gtm_videogen.media.graphics_card import GraphicsCardRenderer

    bible = make_creative_bible()
    card_path = tmp_path / "end_card.mp4"
    renderer = GraphicsCardRenderer()
    clip = await renderer.render_card(
        text="Save 25% First 2 Bags",
        width=1280,
        height=720,
        fps=30,
        duration_sec=2.0,
        bible=bible,
        output_path=card_path,
    )
    assert Path(clip.path).exists()
    probe = ffprobe_inspect(Path(clip.path))
    v_stream = next(s for s in probe["streams"] if s.get("codec_type") == "video")
    assert int(v_stream["width"]) == 1280
    assert int(v_stream["height"]) == 720
    assert parse_frame_rate(v_stream["r_frame_rate"]) == pytest.approx(30.0, abs=0.5)
    assert float(probe["format"]["duration"]) == pytest.approx(2.0, abs=0.3)

    sheet = make_shot_sheet()
    ass_path = tmp_path / "captions.ass"
    written = generate_ass_subtitles(sheet, bible, ass_path)
    ass_text = Path(written).read_text(encoding="utf-8")
    assert bible.typography_and_layout.font_family in ass_text
    assert "[Events]" in ass_text


async def test_f13_media_qa_streams_lufs_and_ocr(tmp_path: Path) -> None:
    """F13: MediaQA verifies ffprobe streams, ebur128 LUFS loudness, and Tesseract OCR."""
    from gtm_videogen.media.media_qa import MediaQA

    video_path = make_synthetic_clip(
        tmp_path / "ocr_test.mp4",
        width=1280,
        height=720,
        fps=24,
        duration_sec=2.0,
        color="0x111827",
        text="ROASTED FRESH",
        include_audio=True,
    )
    sheet = make_shot_sheet(
        fps=24,
        aspect="16:9",
        target_duration_sec=2.0,
        shots=[
            Shot(
                id="s1",
                index=0,
                beat="cta",
                duration_sec=2.0,
                visual_prompt="Title card",
                on_screen_text="ROASTED FRESH",
            )
        ],
    )
    qa = MediaQA()
    report = await qa.evaluate(video_path=video_path, sheet=sheet, run_ocr=True)
    assert isinstance(report, QAReport)
    assert report.ffprobe_summary
    assert report.loudness_lufs is not None
    assert math.isfinite(report.loudness_lufs)
    assert len(report.ocr_checks) >= 1
    assert any(chk.passed for chk in report.ocr_checks)


async def test_f21_manim_renderer_availability_and_honest_fallback(tmp_path: Path) -> None:
    """F21: is_manim_available() reports truthfully; absent manim returns honest fallback error."""
    from gtm_videogen.media.manim_renderer import (
        ManimRenderer,
        ManimRenderError,
        ManimUnavailableError,
        is_manim_available,
    )

    available = is_manim_available()
    assert isinstance(available, bool)

    scene_code = (
        "from manim import FadeIn, Scene, Text\n"
        "class SimpleScene(Scene):\n"
        "    def construct(self):\n"
        "        self.play(FadeIn(Text('Kestrel')), run_time=0.5)\n"
        "        self.wait(0.5)\n"
    )
    out_path = tmp_path / "manim_out.mp4"
    renderer = ManimRenderer()
    if not available:
        with pytest.raises((ManimUnavailableError, ManimRenderError)):
            await renderer.render_scene(
                code=scene_code,
                scene_class="SimpleScene",
                output_path=out_path,
                quality="preview",
            )
    else:
        rendered = await renderer.render_scene(
            code=scene_code,
            scene_class="SimpleScene",
            output_path=out_path,
            quality="preview",
        )
        assert Path(rendered.path).exists()
        probe = ffprobe_inspect(Path(rendered.path))
        assert any(s.get("codec_type") == "video" for s in probe["streams"])


# ---------------------------------------------------------------------------
# M4 Features: F14 - F16 (3 Rubric-Gated DeepAgents + PipelineRunner)
# ---------------------------------------------------------------------------


async def test_f14_to_f16_pipeline_runner_with_mock_bundle(tmp_path: Path) -> None:
    """F14-F16: PipelineRunner executes all 3 rubric-gated DeepAgents offline to completion."""
    from gtm_videogen.agents.pipeline import PipelineRunner

    from gtm_videogen.providers.mocks import MockProviderBundle

    bundle = MockProviderBundle.create(work_dir=tmp_path)
    runner = PipelineRunner(providers=bundle, work_dir=tmp_path)
    events: list[dict] = []

    run_state = make_run_state(run_id="e2e_f14_f16", run_input=make_run_input(mock=True))
    completed = await runner.run(run_state, event_cb=lambda ev: events.append(ev))

    assert completed.status == "done"
    assert completed.research is not None
    assert completed.creative_bible is not None
    assert completed.shot_sheet is not None
    assert completed.render_manifest is not None
    assert completed.render_manifest.stitched_path is not None
    assert Path(completed.render_manifest.stitched_path).exists()
    assert completed.qa_report is not None
    assert len(completed.rubric_evaluations) >= 1
    assert any(ev.get("type") == "run_completed" for ev in events)


# ---------------------------------------------------------------------------
# M5 Features: F17 - F18 (FastAPI ASGI Server & Typer CLI)
# ---------------------------------------------------------------------------


async def test_f17_fastapi_core_endpoints_and_range_playback(tmp_path: Path) -> None:
    """F17: FastAPI /api/health, /api/runs, /api/jobs, and /api/media/{id} HTTP 206 Range."""
    from gtm_videogen.api.app import create_app

    media_store = MediaStore(tmp_path / "media", resolve_dns_for_ssrf=False)
    sample_bytes = b"0123456789abcdef" * 128
    media_id, _ = media_store.save_bytes(sample_bytes, ext=".mp4")

    app = create_app(data_dir=tmp_path, media_store=media_store)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        health = await client.get("/api/health")
        assert health.status_code == 200

        payload = make_run_input(mock=True).model_dump()
        created = await client.post("/api/runs", json=payload)
        assert created.status_code in (200, 201)
        run_data = created.json()
        run_id = run_data["run_id"]

        fetched = await client.get(f"/api/runs/{run_id}")
        assert fetched.status_code == 200
        assert fetched.json()["run_id"] == run_id

        jobs = await client.get("/api/jobs")
        assert jobs.status_code == 200

        missing = await client.get("/api/media/nonexistent-media-id")
        assert missing.status_code == 404

        ranged = await client.get(
            f"/api/media/{media_id}",
            headers={"Range": "bytes=0-63"},
        )
        assert ranged.status_code == 206
        assert len(ranged.content) == 64
        assert ranged.content == sample_bytes[:64]
        content_range = ranged.headers.get("content-range")
        assert content_range is not None
        assert content_range.startswith(f"bytes 0-63/{len(sample_bytes)}")


def test_f18_cli_doctor_and_mock_run(tmp_path: Path) -> None:
    """F18: Typer CLI `doctor` and `run --mock` execute cleanly via CliRunner."""
    from gtm_videogen.cli.main import app
    from typer.testing import CliRunner

    cli = CliRunner()
    doc_res = cli.invoke(app, ["doctor"])
    assert doc_res.exit_code == 0, doc_res.output
    assert "ffmpeg" in doc_res.output.lower()

    run_res = cli.invoke(
        app,
        [
            "run",
            "--brand-name",
            "Kestrel Roasters",
            "--brand-url",
            "https://kestrelroasters.example.com",
            "--goal",
            "Drive coffee subscription trials",
            "--output-dir",
            str(tmp_path),
            "--mock",
        ],
    )
    assert run_res.exit_code == 0, run_res.output
