# gtm-videogen-deepagents

A Python 3.12 (`uv`) rewrite of the GTM video generation pipeline powered by **3 rubric-gated LangChain DeepAgents** (`deepagents==0.7.41` + `RubricMiddleware`), a single-pass FFmpeg media engine, an optional sandboxed **Manim Community** motion-graphics renderer (`$0/shot`), resilient async multi-provider clients, a FastAPI SSE server, and a Typer CLI — built with a **strict zero-hard-templates policy** everywhere except **pure script rules**.

---

## Architecture

```text
models/     Pydantic v2 strict schemas (RunInput extra="forbid", CreativeBible, ShotSheet, SoundPlan, QAReport, CostLedger, RunState, JobStatusDTO)
   ↑
core/       Settings, structured logging, SSRF guard, ResilientHttpClient, ContextVar CostMeter, path-based MediaStore, ContentCache, unified JobRegistry, atomic RunStore
   ↑
rules/      Pure script rules ONLY (claim grounding, speakability budget, exact timing bisection, anti-garbled-text, diversity scoring, attribute-matched voice casting, CreativeHistory)
   ↑
providers/  Stage-aware LLMRouter + provider capability matrix, SSRF-safe BrandScraper, multi-provider Video & Audio adapters + fallback chains, Whisper + speech_canon, offline MockProviderBundle
   ↑
media/      Single-pass FFmpegStitcher, CreativeBible-driven ASS captions & GraphicsCardRenderer, parallel MediaQA (ffprobe, blackdetect/freezedetect, ebur128 LUFS, Tesseract OCR), sandboxed ManimRenderer
   ↑
agents/     3 Rubric-Gated DeepAgents (create_deep_agent + RubricMiddleware) & PipelineRunner:
              1. Research & CreativeBible Agent (Brand scraping, 4 orthogonal concepts, bespoke CreativeBible, CreativeHistory anti-repetition)
              2. Script & ShotSheet Director Agent (Script, dynamic ShotSheet, bespoke SoundPlan; graded by pure script rules + originality/bible_coherence)
              3. Render, Stitch & Final QA Producer Agent (Overlapped render + audio, single-pass stitch, parallel QA, targeted single-shot repair, USD cap enforcement)
   ↑
api/ & cli/ FastAPI SSE server (HTTP 206 Range media serving, Origin 403 middleware, ownership auth, TTL rate limiter) & Typer CLI (`run`, `serve`, `doctor`)
```

### Zero Hard Templates vs. Pure Script Rules

- **Zero Hard Templates**: No hardcoded story patterns, fixed beat locks, hardcoded color/font house styles, fixed camera moves, hardcoded 3D props, or synthetic audio stings. Every run synthesizes a bespoke `CreativeBible` (narrative arc, visual world, camera grammar, typography from 5 bundled open-license font families, sonic identity, and brief-specific anti-cliché list) informed by cross-run `CreativeHistory` anti-repetition memory.
- **Pure Script Rules Only (Blocking Rubric Gates)**:
  1. **Claim Grounding**: Every factual/numeric claim must be backed by verified research facts; unsupported superlatives, banned phrases, and unresolved placeholders are non-overridable blocking errors.
  2. **Speakability & Timing Physics**: Spoken-form syllable budget per shot and exact bounded water-filling timing (`fit_timings`) summing to `target_duration_sec` with zero `NaN`s.
  3. **Production & Readability Validity**: Complete camera/audio spec, `product_hero` requires scraped product photos, anti-garbled-text guardrail, and WCAG 4.5:1 text contrast.

---

## 28 Legacy Bugs Fixed & 10 Performance Speedups

### Key Reliability & Security Fixes (28/28)
- **Concurrency & Cancellation (BUG-01, BUG-02, BUG-05, BUG-19, BUG-22)**: Set-based `JobRegistry` slot tracking with `asyncio.Semaphore` + `try/finally`, strictly JSON-serializable `JobStatusDTO` + `other_active_jobs_count`, 4-layer `CancellationToken` that never overwrites `cancelled` with `done`, and automatic orphan `"working"` run healing on load.
- **Validation & Security (BUG-03, BUG-11, BUG-13, BUG-14, BUG-20)**: `RunInput(extra="forbid")` returning HTTP 422 on unknown fields (`audio_mode` supports `"vo" | "foley" | "both"`), SSRF protection blocking private/loopback/metadata IPs and >3 redirects, centralized path-based `MediaStore`, `X-Actor-Id` ownership authorization (`403`), `Origin: null`/malformed Origin rejection (`403`, never `500`), TTL-pruned in-memory rate limiter (`429`), and explicit HTTP timeouts + `raise_for_status()` before `.json()`.
- **Providers, Audio, Media & Cost Control (BUG-04, BUG-06..BUG-10, BUG-12, BUG-15..BUG-18, BUG-21, BUG-23..BUG-28)**: Typed per-model audio payloads + SFX fallback chain (`mirelo -> elevenlabs -> sonilo`), aspect ratio honoring (`16:9`, `9:16`, `1:1`, `4:5`) across video providers, exact `width`/`height`/`fps`/`duration_sec` on graphics cards and stitcher, empty-audio FFmpeg filtergraph guard, full preflight cost estimation (including audio + video `min_seconds` floor) + `ContextVar`-scoped `CostMeter`, non-fatal `warnings` separated from fatal `error`, attribute-matched `select_voice()`, `sound_design` routed to `text_model`, cross-platform binary discovery + bundled fonts + `:fontsdir=` escaping, and async `/repair` persisting `repair_result`.

### 10 Performance Speedups (P1–P10)
1. **P1**: Bounded parallel FFmpeg clip normalization with pinned H.264 High@4.1 / `bt709` / `sheet.fps` SPS metadata.
2. **P2**: Stream-copy concatenation (`-f concat -c copy`) + single-pass audio mix, optional ASS caption burn, and `-t` trim.
3. **P3**: Overlapped audio generation alongside video rendering.
4. **P4**: Content-addressed persistent `clipCache`, `audioCache`, and `judgeCache` (verified on disk before hit).
5. **P5**: Process-level memoization (`@lru_cache`) of bundled font discovery and graphics setup.
6. **P6**: Progressive provider polling backoff (`0.5s -> 3.0s` audio; `2.0s -> 5.0s` video).
7. **P7**: Path-based `MediaStore` + HTTP `206 Partial Content` streaming (zero multi-MB base64 RAM churn).
8. **P8**: Parallel brand + inspiration scraping with single homepage HTML fetch reused for product photos.
9. **P9**: Clean extensible engine routing (`ai_video`, `image_to_video`, `motion_card`, `manim`, optional `blender`).
10. **P10**: Bounded parallel Media QA (`ffprobe`, `blackdetect`/`freezedetect`, `ebur128` LUFS, Tesseract OCR).

---

## Installation

Requires **Python 3.12+**, [`uv`](https://docs.astral.sh/uv/), **FFmpeg** (`ffmpeg` & `ffprobe`), and optionally **Tesseract OCR** (`tesseract`).

```bash
# Core install (DeepAgents, FastAPI, Typer, FFmpeg/Pillow/Tesseract wrappers)
uv sync

# Verify local environment and bundled fonts
uv run gtm-videogen doctor
```

### Optional Manim Community Renderer (`engine="manim"`, `$0/shot`)

`manim` is an optional extra for bespoke, `CreativeBible`-driven kinetic typography, charts, explainers, and dynamic end cards rendered locally in an AST-sandboxed subprocess.

> **Honest Linux Host Dependency Note:** On PyPI, `pycairo` and `manimpango` currently publish **no Linux binary wheels** and compile from source against system C headers. On Debian/Ubuntu Linux, install the native build libraries before syncing the `manim` extra:
> ```bash
> sudo apt-get install -y libcairo2-dev libpango1.0-dev pkg-config gcc ffmpeg
> uv sync --extra manim
> ```
> Non-LaTeX rendering (`Text`, `MarkupText`, vector shapes/charts) works without TeX; host-file loaders like `SVGMobject`, `ImageMobject`, and `add_sound` are intentionally blocked by the AST sandbox. LaTeX mobjects (`Tex`/`MathTex`) additionally require `dvisvgm` and a TeX distribution; the AST sandbox automatically blocks LaTeX classes when `dvisvgm` is absent.

---

## Honest Offline vs. Live Usage

| Capability | Offline (`--mock`) | Live Mode (API Keys Required) |
|---|---|---|
| 3-Agent `deepagents` + `RubricMiddleware` loop | Real `create_deep_agent` + `RubricMiddleware` using `ScriptedChatModel` | Requires `OPENROUTER_API_KEY` (or `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `GOOGLE_API_KEY`) |
| Video & Audio Rendering | Real synthetic FFmpeg `lavfi` clips & WAVs | Requires `FAL_KEY` (and/or `OPENROUTER_API_KEY`, `HF_API_KEY`) |
| Single-Pass FFmpeg Stitcher, ASS Captions, Graphics Cards | **100% Real** local FFmpeg + Pillow + 5 bundled fonts | **100% Real** |
| Deterministic Media QA (`ffprobe`, LUFS, OCR) | **100% Real** FFmpeg + Tesseract OCR | **100% Real** (+ optional Whisper/vision QA) |
| Manim Community Motion Graphics (`engine="manim"`) | **100% Real** when `manim` extra & Cairo/Pango are installed (`$0/shot`) | **100% Real** (`$0/shot`) |

Copy `.env.example` to `.env` and populate your provider keys for live runs.

---

## CLI Usage (`gtm-videogen`)

```bash
# Check local binaries (ffmpeg, ffprobe, tesseract, manim), bundled fonts, and API keys
uv run gtm-videogen doctor

# Run the full 3-agent pipeline offline with deterministic mocks + real FFmpeg
uv run gtm-videogen run \
  --brand-name "Kestrel Roasters" \
  --brand-url "https://kestrelroasters.example.com" \
  --goal "Drive single-origin coffee subscription trials" \
  --duration 15 \
  --aspect 16:9 \
  --cap-usd 5.0 \
  --output-dir ./output/demo \
  --mock

# Start the FastAPI SSE server
uv run gtm-videogen serve --host 127.0.0.1 --port 8000
```

---

## FastAPI SSE Server Endpoints

| Method & Path | Description |
|---|---|
| `GET /api/health` | Live tool availability (`ffmpeg`, `ffprobe`, `tesseract`, `manim` diagnostics) |
| `POST /api/runs` | Create run (`RunInput` with `extra="forbid"` -> HTTP 422 on unknown fields); records `X-Actor-Id` owner |
| `GET /api/runs` | List persisted runs (auto-heals orphaned `"working"` runs) |
| `GET /api/runs/{run_id}` | Fetch `RunState` (auto-heals orphaned `"working"` runs) |
| `POST /api/runs/{run_id}/start` | Owner-only (`X-Actor-Id`); launches async pipeline job via `JobRegistry` (`202`) |
| `GET /api/runs/{run_id}/events` | SSE (`text/event-stream`) streaming `stage_started`, `rubric_evaluation`, `shot_progress`, `cost_update`, `warning`, `stage_completed`, `run_completed`, `run_cancelled`, `run_failed` |
| `POST /api/runs/{run_id}/cancel` | Owner-only; triggers 4-layer cancellation via `JobRegistry` and emits `run_cancelled` |
| `POST /api/runs/{run_id}/repair` | Owner-only; triggers async targeted repair (`failed_shot_ids`) persisting `repair_result` (`202`) |
| `GET /api/jobs` | Returns strictly serializable `JobStatusDTO` list + `other_active_jobs_count` (supports `?run_id=`) |
| `GET /api/media/{media_id}` | Streams media assets with HTTP `206 Partial Content` (`Range`), `416 Range Not Satisfiable`, and path-traversal protection |

### Security Middleware
- **Origin/CORS Validation**: Rejects `Origin: null`, malformed URLs, and disallowed origins with HTTP `403` JSON (never `500`).
- **Ownership Authorization**: Enforces `run.created_by == X-Actor-Id` on `/start`, `/cancel`, and `/repair` (`403` on mismatch, `404` if missing).
- **TTL-Pruned Rate Limiter**: Sliding-window per-actor/IP rate limiter with automatic TTL eviction and bounded key count (`429` + `Retry-After`).

---

## Testing

```bash
# Run the full offline test suite
uv run pytest -v

# Run API & CLI tests only
uv run pytest tests/test_api_cli.py -v
```

---

## Repository Status (Honest Git Note)

This repository is maintained locally (`git init -b main`) at `/home/antigravity/kollaakhil/gtm-videogen-deepagents` with intended GitHub remote target `https://github.com/kollaakhil/GTMTickner.git` (branch `main`). Because `gh repo create` in this environment previously returned `GraphQL: Resource not accessible by integration (createRepository)`, pushing to `https://github.com/kollaakhil/GTMTickner.git` requires valid push credentials:

```bash
git remote add origin https://github.com/kollaakhil/GTMTickner.git
git push -u origin main
```
