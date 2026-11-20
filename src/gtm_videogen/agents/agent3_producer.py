"""Agent 3 (F16): Render, Stitch & Final QA Producer Rubric-Gated DeepAgent.

Owns:
- Pillar 1 (qa / shotCheck fast tier)
- Pillar 4 (open renderer/audio/caption execution incl. optional engine="manim")
- Pillar 5 (parallel deterministic Media QA + targeted single-shot repair loop)
- Pillar 6 (CreativeHistory fingerprint writeback on completion)
- P3 (overlapped video + audio generation)
- P4 (clipCache & audioCache reuse)
- P10 (parallel MediaQA evaluation)
- BUG-05 (CancellationToken propagation across render/stitch/QA)
- BUG-06 (targeted single-shot repair leveraging caches)
- BUG-15 (preflight estimate_run_cost including audio + min_seconds)
- BUG-16 (non-fatal SFX warnings preserved separately from fatal error)
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.tools import BaseTool, tool

from gtm_videogen.agents.rubric_utils import (
    EventCallback,
    emit_event,
    is_default_unscripted_mock_model,
    is_offline_provider_bundle,
    make_tool_driven_grader_model,
    read_workspace_json,
    resolve_live_chat_model,
    run_rubric_deep_agent,
    write_workspace_json,
)
from gtm_videogen.core.cost_meter import BudgetExceeded, CostMeter, estimate_run_cost
from gtm_videogen.core.job_registry import CancellationToken
from gtm_videogen.media._subprocess import aspect_to_dimensions
from gtm_videogen.media.ffmpeg_stitcher import FFmpegStitcher
from gtm_videogen.media.graphics_card import GraphicsCardRenderer
from gtm_videogen.media.manim_renderer import (
    ManimRenderer,
    ManimRenderError,
    is_manim_available,
)
from gtm_videogen.media.media_qa import MediaQA
from gtm_videogen.models.schemas import (
    CostLedger,
    CreativeBible,
    QAReport,
    RenderedAudio,
    RenderedClip,
    RenderManifest,
    RunInput,
    Shot,
    ShotSheet,
    SoundPlan,
)
from gtm_videogen.providers.mocks import ScriptedChatModel

AGENT3_STAGE = "agent3_producer"

AGENT3_SYSTEM_PROMPT = """You are the Render, Stitch & Final QA Producer DeepAgent.
Execute preflight cost checks, overlap video and audio rendering with cache reuse,
stitch the final video in a single pass with CreativeBible-driven captions, run parallel
deterministic Media QA, and perform targeted single-shot repairs when needed.

Always use your tools in sequence:
1. preflight_cost_tool to verify budget before paid rendering.
2. render_assets_tool to render video shots and audio assets concurrently.
3. stitch_video_tool to assemble the final MP4.
4. run_media_qa_tool to evaluate MP4 validity, loudness, OCR, and sync.
5. repair_shots_tool if any specific shots fail QA.
"""

AGENT3_GRADER_SYSTEM_PROMPT = """You are the Final Production QA & Budget Rubric Grader.
Always invoke `inspect_qa_and_budget_tool` to inspect `render_manifest.json`, `qa_report.json`,
and `cost_ledger.json` directly before emitting your structured GraderResponse verdict.
"""

AGENT3_RUBRIC = """Evaluate the final rendered video, QA report, and CostLedger:
1. valid_mp4_output: Stitched MP4 exists on disk with valid video matching aspect/fps/duration.
2. loudness_and_audio: No blocking container, black-frame, or non-card freeze-frame errors.
3. ocr_and_sync: On-screen text OCR and voiceover sync checks have no blocking errors.
4. budget_compliance: cost_ledger.actual_usd <= cost_ledger.cap_usd.
"""


@dataclass
class Agent3Result:
    """Output of Agent 3: RenderManifest, QAReport, CostLedger, warnings, and rubric history."""

    render_manifest: RenderManifest
    qa_report: QAReport
    cost_ledger: CostLedger
    rubric_evaluations: list[dict[str, Any]] = field(default_factory=list)
    status: str = "satisfied"
    warnings: list[str] = field(default_factory=list)
    workspace_dir: Path | None = None


def _caption_safe_sheet(shot_sheet: ShotSheet) -> ShotSheet:
    """Clear text/caption fields on motion_card/manim shots for ASS burn-in (ONE_TEXT_LAYER)."""
    updated_shots = [
        (
            s.model_copy(update={"on_screen_text": None, "vo_line": None})
            if s.engine in ("motion_card", "manim")
            else s
        )
        for s in shot_sheet.shots
    ]
    return shot_sheet.model_copy(update={"shots": updated_shots})


def _build_default_manim_scene(shot: Shot, bible: CreativeBible) -> tuple[str, str]:
    """Build a minimal CreativeBible-driven Manim Scene when a shot uses engine='manim'."""
    palette = bible.visual_language.color_palette or ["#111827", "#FFFFFF", "#3B82F6"]
    bg_hex = palette[0]
    fg_hex = "#FFFFFF"
    label = (shot.on_screen_text or shot.beat or "Scene")[:60]
    hold_sec = max(0.2, round(float(shot.duration_sec) - 0.5, 2))
    scene_class = "BespokeShotScene"
    code = (
        "from manim import Scene, Text, Write\n\n"
        f"class {scene_class}(Scene):\n"
        "    def construct(self):\n"
        f"        self.camera.background_color = {json.dumps(bg_hex)}\n"
        f"        title = Text({json.dumps(label)}, color={json.dumps(fg_hex)}, font_size=44)\n"
        "        self.play(Write(title), run_time=0.5)\n"
        f"        self.wait({hold_sec})\n"
    )
    return code, scene_class


async def _render_single_shot(
    *,
    shot: Shot,
    sheet: ShotSheet,
    bible: CreativeBible,
    providers: Any,
    graphics_renderer: GraphicsCardRenderer,
    manim_renderer: ManimRenderer,
    clips_dir: Path,
    render_mode: str,
    warnings: list[str],
    cancel: CancellationToken | None,
    bypass_cache: bool = False,
) -> RenderedClip:
    """Render one shot per its engine (ai_video, image_to_video, motion_card, manim)."""
    if cancel is not None:
        cancel.raise_if_cancelled()

    width, height = aspect_to_dimensions(sheet.aspect)
    clips_dir.mkdir(parents=True, exist_ok=True)

    if shot.engine == "motion_card":
        out_path = clips_dir / f"{shot.id}_card.mp4"
        is_end = shot.index == len(sheet.shots) - 1
        card_text = (
            shot.on_screen_text
            or sheet.end_card_title
            or (shot.vo_line if is_end and shot.vo_line else None)
            or shot.beat
        )
        return await graphics_renderer.render_card(
            text=card_text,
            width=width,
            height=height,
            fps=sheet.fps,
            duration_sec=float(shot.duration_sec),
            bible=bible,
            output_path=out_path,
            subtitle=sheet.end_card_subtitle,
            footer=sheet.end_card_footer,
            card_kind="end_card" if is_end else "title",
            shot_id=shot.id,
            theme=shot.theme or sheet.theme,
            cancel=cancel,
        )

    if shot.engine == "manim":
        out_path = clips_dir / f"{shot.id}_manim.mp4"
        if is_manim_available():
            try:
                code, scene_cls = _build_default_manim_scene(shot, bible)
                return await manim_renderer.render_scene(
                    code=code,
                    scene_class=scene_cls,
                    output_path=out_path,
                    aspect=sheet.aspect,
                    fps=sheet.fps,
                    quality="final" if render_mode == "final" else "preview",
                    shot_id=shot.id,
                    static_duration_sec=float(shot.duration_sec),
                    cancel=cancel,
                )
            except (ManimRenderError, Exception) as exc:
                warnings.append(
                    f"Manim render failed for shot {shot.id} ({exc}); falling back to motion_card."
                )
        else:
            warnings.append(
                f"Manim unavailable on host for shot {shot.id}; falling back to motion_card."
            )
        return await graphics_renderer.render_card(
            text=shot.on_screen_text or shot.beat,
            width=width,
            height=height,
            fps=sheet.fps,
            duration_sec=float(shot.duration_sec),
            bible=bible,
            output_path=clips_dir / f"{shot.id}_fallback_card.mp4",
            shot_id=shot.id,
            theme=shot.theme or sheet.theme,
            cancel=cancel,
        )

    # ai_video / image_to_video / blender -> VideoProvider
    video_provider = providers.video
    effective_shot = shot
    if bypass_cache:
        effective_shot = shot.model_copy(
            update={"visual_prompt": f"{shot.visual_prompt} [repair:{int(time.time() * 1000)}]"}
        )
    return await video_provider.generate_clip(
        effective_shot,
        aspect=sheet.aspect,
        fps=sheet.fps,
        cancel=cancel,
    )


async def render_assets_overlapped(
    *,
    run_id: str,
    shot_sheet: ShotSheet,
    sound_plan: SoundPlan | None,
    creative_bible: CreativeBible,
    providers: Any,
    work_dir: Path,
    render_mode: str = "preview",
    event_cb: EventCallback | None = None,
    cancel: CancellationToken | None = None,
    existing_clips: dict[str, RenderedClip] | None = None,
    repair_shot_ids: set[str] | None = None,
) -> tuple[list[RenderedClip], list[RenderedAudio], list[str]]:
    """Overlap video rendering with audio generation (P3, P4, BUG-05, BUG-16)."""
    if cancel is not None:
        cancel.raise_if_cancelled()

    warnings: list[str] = []
    graphics_renderer = GraphicsCardRenderer()
    manim_renderer = ManimRenderer()
    clips_dir = work_dir / "clips"
    ordered_shots = sorted(shot_sheet.shots, key=lambda s: s.index)

    first_video_ok = asyncio.Event()
    clip_sem = asyncio.Semaphore(4)

    async def _render_shot_task(shot: Shot) -> RenderedClip:
        async with clip_sem:
            if cancel is not None:
                cancel.raise_if_cancelled()

            # Reuse existing clip if doing targeted repair and this shot is NOT targeted
            if (
                existing_clips is not None
                and repair_shot_ids is not None
                and shot.id not in repair_shot_ids
                and shot.id in existing_clips
                and Path(existing_clips[shot.id].path).is_file()
            ):
                reused = existing_clips[shot.id].model_copy(update={"cache_hit": True, "usd": 0.0})
                first_video_ok.set()
                await emit_event(
                    event_cb,
                    event_type="shot_progress",
                    run_id=run_id,
                    stage=AGENT3_STAGE,
                    data={"shot_id": shot.id, "status": "reused", "cache_hit": True},
                )
                return reused

            bypass = bool(repair_shot_ids and shot.id in repair_shot_ids)
            clip = await _render_single_shot(
                shot=shot,
                sheet=shot_sheet,
                bible=creative_bible,
                providers=providers,
                graphics_renderer=graphics_renderer,
                manim_renderer=manim_renderer,
                clips_dir=clips_dir,
                render_mode=render_mode,
                warnings=warnings,
                cancel=cancel,
                bypass_cache=bypass,
            )
            first_video_ok.set()
            await emit_event(
                event_cb,
                event_type="shot_progress",
                run_id=run_id,
                stage=AGENT3_STAGE,
                data={
                    "shot_id": shot.id,
                    "status": "rendered",
                    "cache_hit": clip.cache_hit,
                    "usd": clip.usd,
                },
            )
            return clip

    async def _render_audio_task() -> list[RenderedAudio]:
        if sound_plan is None:
            return []
        # Gate audio spend until >= 1 video clip succeeds (P3)
        await first_video_ok.wait()
        if cancel is not None:
            cancel.raise_if_cancelled()

        audio_provider = providers.audio
        voice_id = sound_plan.voice_id or "v_maya_warm"
        audio_assets: list[RenderedAudio] = []

        async def _gen_vo(cue: Any) -> RenderedAudio | None:
            if cancel is not None:
                cancel.raise_if_cancelled()
            return await audio_provider.generate_voice(cue, voice_id=voice_id, cancel=cancel)

        async def _gen_sfx(cue: Any) -> RenderedAudio | None:
            if cancel is not None:
                cancel.raise_if_cancelled()
            try:
                return await audio_provider.generate_sfx(cue, cancel=cancel)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # BUG-16: Non-fatal SFX failures record persistent warnings without failing run
                msg = f"Non-fatal SFX failure for {cue.shot_id or cue.at_sec}: {exc}"
                warnings.append(msg)
                await emit_event(
                    event_cb,
                    event_type="warning",
                    run_id=run_id,
                    stage=AGENT3_STAGE,
                    data={"warning": msg},
                )
                return None

        async def _gen_music(cue: Any) -> RenderedAudio | None:
            if cancel is not None:
                cancel.raise_if_cancelled()
            try:
                return await audio_provider.generate_music(cue, cancel=cancel)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                msg = f"Non-fatal music generation failure: {exc}"
                warnings.append(msg)
                await emit_event(
                    event_cb,
                    event_type="warning",
                    run_id=run_id,
                    stage=AGENT3_STAGE,
                    data={"warning": msg},
                )
                return None

        coros: list[Any] = []
        for vc in sound_plan.vo_cues:
            coros.append(_gen_vo(vc))
        for sc in sound_plan.sfx_cues:
            coros.append(_gen_sfx(sc))
        if sound_plan.music_cue is not None:
            coros.append(_gen_music(sound_plan.music_cue))

        if not coros:
            return []

        results = await asyncio.gather(*coros)
        for item in results:
            if item is not None:
                audio_assets.append(item)
        return audio_assets

    if not ordered_shots:
        first_video_ok.set()
    video_tasks = [asyncio.create_task(_render_shot_task(s)) for s in ordered_shots]
    audio_task = asyncio.create_task(_render_audio_task())
    if cancel is not None:
        for t in video_tasks:
            cancel.track_task(t)
        cancel.track_task(audio_task)

    try:
        clips = list(await asyncio.gather(*video_tasks))
        audio_assets = await audio_task
    except BaseException:
        for t in (*video_tasks, audio_task):
            if not t.done():
                t.cancel()
        await asyncio.gather(*video_tasks, audio_task, return_exceptions=True)
        raise

    return clips, audio_assets, warnings


def _build_agent3_tools(
    *,
    run_id: str,
    run_input: RunInput,
    shot_sheet: ShotSheet,
    sound_plan: SoundPlan | None,
    creative_bible: CreativeBible,
    providers: Any,
    cost_meter: CostMeter,
    workspace: Path,
    director_qa: QAReport | None,
    event_cb: EventCallback | None,
    cancel: CancellationToken | None,
) -> tuple[list[BaseTool], list[BaseTool]]:
    """Create Agent 3 Producer worker tools and its deterministic QA/budget grader tool."""
    stitcher = FFmpegStitcher()
    media_qa = MediaQA()
    state_bag: dict[str, Any] = {
        "clips": [],
        "audio_assets": [],
        "warnings": [],
        "estimated_usd": 0.0,
    }

    @tool("preflight_cost_tool")
    def preflight_cost_tool() -> str:
        """Estimate full run USD cost (incl. audio + min_seconds) vs cap_usd (BUG-15)."""
        est = estimate_run_cost(shot_sheet=shot_sheet, sound_plan=sound_plan)
        state_bag["estimated_usd"] = est["total_usd"]
        budget_ok = True if run_input.mock else (est["total_usd"] <= cost_meter.cap_usd + 1e-9)
        payload = {
            **est,
            "cap_usd": cost_meter.cap_usd,
            "budget_ok": budget_ok,
            "mock": run_input.mock,
        }
        write_workspace_json(workspace, "preflight_cost.json", payload)
        if not budget_ok:
            raise BudgetExceeded(
                label="preflight_estimate",
                cap_usd=cost_meter.cap_usd,
                spent_usd=cost_meter.spent_usd,
                reserved_usd=cost_meter.reserved_usd,
                requested_usd=est["total_usd"],
            )
        return json.dumps(payload)

    @tool("render_assets_tool")
    async def render_assets_tool() -> str:
        """Render video clips and audio assets concurrently with cache reuse (P3/P4/BUG-05/16)."""
        t0 = time.monotonic()
        clips, audio_assets, warnings = await render_assets_overlapped(
            run_id=run_id,
            shot_sheet=shot_sheet,
            sound_plan=sound_plan,
            creative_bible=creative_bible,
            providers=providers,
            work_dir=workspace,
            render_mode=run_input.render_mode,
            event_cb=event_cb,
            cancel=cancel,
        )
        state_bag["clips"] = clips
        state_bag["audio_assets"] = audio_assets
        state_bag["warnings"] = list(warnings)

        manifest = RenderManifest(
            run_id=run_id,
            clips=clips,
            audio_assets=audio_assets,
            timings_ms={"render_assets_ms": int((time.monotonic() - t0) * 1000)},
            warnings=list(warnings),
            error=None,
        )
        ledger = cost_meter.to_ledger(estimated_usd=state_bag["estimated_usd"])
        write_workspace_json(workspace, "render_manifest.json", manifest)
        write_workspace_json(workspace, "cost_ledger.json", ledger)
        return json.dumps(
            {
                "status": "ok",
                "clips_count": len(clips),
                "audio_assets_count": len(audio_assets),
                "warnings_count": len(warnings),
                "actual_usd": ledger.actual_usd,
            }
        )

    @tool("stitch_video_tool")
    async def stitch_video_tool() -> str:
        """Stitch rendered clips and audio in a single FFmpeg pass with optional ASS captions."""
        if cancel is not None:
            cancel.raise_if_cancelled()

        clips: list[RenderedClip] = state_bag["clips"]
        audio_assets: list[RenderedAudio] = state_bag["audio_assets"]
        prior_warnings: list[str] = list(state_bag["warnings"])

        if not clips:
            raw_m = read_workspace_json(workspace, "render_manifest.json")
            if raw_m:
                m_obj = RenderManifest.model_validate(raw_m)
                clips = m_obj.clips
                audio_assets = m_obj.audio_assets
                prior_warnings = list(m_obj.warnings)

        output_path = workspace / f"{run_id}_final.mp4"
        burn_captions = any(
            s.on_screen_text
            and s.on_screen_text.strip()
            and s.engine not in ("motion_card", "manim")
            for s in shot_sheet.shots
        )
        stitch_sheet = _caption_safe_sheet(shot_sheet) if burn_captions else shot_sheet
        stitch_w = clips[0].width if (run_input.mock and clips) else None
        stitch_h = clips[0].height if (run_input.mock and clips) else None
        manifest = await stitcher.stitch(
            clips=clips,
            sound_plan=sound_plan,
            audio_assets=audio_assets,
            sheet=stitch_sheet,
            bible=creative_bible,
            output_path=output_path,
            cancel=cancel,
            run_id=run_id,
            width=stitch_w,
            height=stitch_h,
            burn_captions=burn_captions,
        )
        # BUG-16: Preserve non-fatal warnings from render phase alongside stitcher warnings
        merged_warnings = list(dict.fromkeys([*prior_warnings, *manifest.warnings]))
        manifest.warnings = merged_warnings
        manifest.error = None
        state_bag["warnings"] = merged_warnings

        write_workspace_json(workspace, "render_manifest.json", manifest)
        return json.dumps(
            {
                "status": "ok",
                "stitched_path": manifest.stitched_path,
                "warnings_count": len(manifest.warnings),
            }
        )

    @tool("run_media_qa_tool")
    async def run_media_qa_tool() -> str:
        """Run parallel deterministic MediaQA (ffprobe, black/freeze, ebur128 LUFS, OCR, sync)."""
        if cancel is not None:
            cancel.raise_if_cancelled()

        raw_m = read_workspace_json(workspace, "render_manifest.json")
        if not raw_m or not raw_m.get("stitched_path"):
            return json.dumps({"status": "error", "message": "stitched_path missing"})

        stitched_path = Path(raw_m["stitched_path"])
        expected_vo = (
            " ".join(s.vo_line.strip() for s in shot_sheet.shots if s.vo_line and s.vo_line.strip())
            or None
        )
        whisper = getattr(providers, "whisper", None)

        qa = await media_qa.evaluate(
            stitched_path,
            shot_sheet,
            expected_vo=expected_vo,
            run_ocr=True,
            whisper=whisper,
            cancel=cancel,
        )
        if director_qa is not None:
            qa.originality_score = director_qa.originality_score
            qa.bible_coherence_score = director_qa.bible_coherence_score
        else:
            qa.originality_score = qa.originality_score or 8.0
            qa.bible_coherence_score = qa.bible_coherence_score or 8.0

        ledger = cost_meter.to_ledger(estimated_usd=state_bag["estimated_usd"])
        write_workspace_json(workspace, "qa_report.json", qa)
        write_workspace_json(workspace, "cost_ledger.json", ledger)
        return json.dumps(
            {
                "status": "ok",
                "passed": qa.passed,
                "blocking_errors": len(qa.blocking_errors),
                "warnings": len(qa.warnings),
                "loudness_lufs": qa.loudness_lufs,
                "failed_shot_ids": qa.failed_shot_ids,
            }
        )

    @tool("repair_shots_tool")
    async def repair_shots_tool(failed_shot_ids_json: str = "[]") -> str:
        """Re-render only targeted failed shots, reusing cached clips & audio (BUG-06)."""
        if cancel is not None:
            cancel.raise_if_cancelled()

        target_ids = set(json.loads(failed_shot_ids_json))
        raw_m = read_workspace_json(workspace, "render_manifest.json")
        existing_map: dict[str, RenderedClip] = {}
        if raw_m:
            m_obj = RenderManifest.model_validate(raw_m)
            existing_map = {c.shot_id: c for c in m_obj.clips}

        clips, audio_assets, warnings = await render_assets_overlapped(
            run_id=run_id,
            shot_sheet=shot_sheet,
            sound_plan=sound_plan,
            creative_bible=creative_bible,
            providers=providers,
            work_dir=workspace,
            render_mode=run_input.render_mode,
            event_cb=event_cb,
            cancel=cancel,
            existing_clips=existing_map,
            repair_shot_ids=target_ids,
        )
        state_bag["clips"] = clips
        state_bag["audio_assets"] = audio_assets
        merged_warnings = list(dict.fromkeys([*state_bag["warnings"], *warnings]))
        state_bag["warnings"] = merged_warnings

        output_path = workspace / f"{run_id}_repaired.mp4"
        burn_captions = any(
            s.on_screen_text
            and s.on_screen_text.strip()
            and s.engine not in ("motion_card", "manim")
            for s in shot_sheet.shots
        )
        stitch_sheet = _caption_safe_sheet(shot_sheet) if burn_captions else shot_sheet
        stitch_w = clips[0].width if (run_input.mock and clips) else None
        stitch_h = clips[0].height if (run_input.mock and clips) else None
        manifest = await stitcher.stitch(
            clips=clips,
            sound_plan=sound_plan,
            audio_assets=audio_assets,
            sheet=stitch_sheet,
            bible=creative_bible,
            output_path=output_path,
            cancel=cancel,
            run_id=run_id,
            width=stitch_w,
            height=stitch_h,
            burn_captions=burn_captions,
        )
        manifest.warnings = list(dict.fromkeys([*merged_warnings, *manifest.warnings]))
        manifest.error = None
        write_workspace_json(workspace, "render_manifest.json", manifest)

        expected_vo = (
            " ".join(s.vo_line.strip() for s in shot_sheet.shots if s.vo_line and s.vo_line.strip())
            or None
        )
        qa = await media_qa.evaluate(
            Path(manifest.stitched_path or output_path),
            shot_sheet,
            expected_vo=expected_vo,
            run_ocr=True,
            whisper=getattr(providers, "whisper", None),
            cancel=cancel,
        )
        if director_qa is not None:
            qa.originality_score = director_qa.originality_score
            qa.bible_coherence_score = director_qa.bible_coherence_score
        ledger = cost_meter.to_ledger(estimated_usd=state_bag["estimated_usd"])
        write_workspace_json(workspace, "qa_report.json", qa)
        write_workspace_json(workspace, "cost_ledger.json", ledger)
        return json.dumps(
            {
                "status": "ok",
                "repaired_shot_ids": sorted(target_ids),
                "qa_passed": qa.passed,
                "stitched_path": manifest.stitched_path,
            }
        )

    @tool("inspect_qa_and_budget_tool")
    def inspect_qa_and_budget_tool() -> str:
        """Deterministically inspect render_manifest.json, qa_report.json, and cost_ledger.json."""
        raw_m = read_workspace_json(workspace, "render_manifest.json")
        raw_qa = read_workspace_json(workspace, "qa_report.json")
        raw_ledger = read_workspace_json(workspace, "cost_ledger.json")

        criteria: list[dict[str, Any]] = []

        # 1. Valid MP4 output
        stitched_ok = False
        if raw_m and raw_m.get("stitched_path") and Path(raw_m["stitched_path"]).is_file():
            stitched_ok = Path(raw_m["stitched_path"]).stat().st_size > 0
        if not stitched_ok:
            criteria.append(
                {
                    "name": "valid_mp4_output",
                    "passed": False,
                    "gap": "Stitched MP4 file is missing or empty.",
                }
            )
        else:
            criteria.append({"name": "valid_mp4_output", "passed": True})

        # 2 & 3. QA checks (loudness/audio/container + OCR/sync)
        if raw_qa is None:
            criteria.append(
                {
                    "name": "loudness_and_audio",
                    "passed": False,
                    "gap": "qa_report.json is missing; run run_media_qa_tool.",
                }
            )
            criteria.append(
                {
                    "name": "ocr_and_sync",
                    "passed": False,
                    "gap": "qa_report.json is missing; run run_media_qa_tool.",
                }
            )
        else:
            qa = QAReport.model_validate(raw_qa)
            static_graphic_ids = {
                s.id for s in shot_sheet.shots if s.engine in ("motion_card", "manim")
            }
            ocr_sync_codes = {"OCR_MISMATCH", "SYNC_MISMATCH"}
            ocr_blocking = [i for i in qa.blocking_errors if i.code in ocr_sync_codes]
            other_blocking = [
                i
                for i in qa.blocking_errors
                if i.code not in ocr_sync_codes
                and not (i.code == "FROZEN_FRAMES" and i.shot_id in static_graphic_ids)
            ]

            if other_blocking:
                criteria.append(
                    {
                        "name": "loudness_and_audio",
                        "passed": False,
                        "gap": "; ".join(f"[{i.code}] {i.message}" for i in other_blocking),
                    }
                )
            else:
                criteria.append({"name": "loudness_and_audio", "passed": True})

            if ocr_blocking:
                criteria.append(
                    {
                        "name": "ocr_and_sync",
                        "passed": False,
                        "gap": "; ".join(f"[{i.code}] {i.message}" for i in ocr_blocking),
                    }
                )
            else:
                criteria.append({"name": "ocr_and_sync", "passed": True})

        # 4. Budget compliance
        if raw_ledger is None:
            ledger = cost_meter.to_ledger(estimated_usd=state_bag["estimated_usd"])
        else:
            ledger = CostLedger.model_validate(raw_ledger)

        if ledger.actual_usd > ledger.cap_usd + 1e-9:
            criteria.append(
                {
                    "name": "budget_compliance",
                    "passed": False,
                    "gap": (
                        f"Actual spend ${ledger.actual_usd:.4f} exceeds cap ${ledger.cap_usd:.4f}."
                    ),
                }
            )
        else:
            criteria.append({"name": "budget_compliance", "passed": True})

        all_passed = all(c["passed"] for c in criteria)
        failed_names = [c["name"] for c in criteria if not c["passed"]]
        return json.dumps(
            {
                "passed": all_passed,
                "result": "satisfied" if all_passed else "needs_revision",
                "explanation": (
                    "Final MP4, Media QA, and USD budget cap all passed."
                    if all_passed
                    else f"Production QA failed criteria: {', '.join(failed_names)}."
                ),
                "criteria": criteria,
            }
        )

    worker_tools: list[BaseTool] = [
        preflight_cost_tool,
        render_assets_tool,
        stitch_video_tool,
        run_media_qa_tool,
        repair_shots_tool,
    ]
    grader_tools: list[BaseTool] = [inspect_qa_and_budget_tool]
    return worker_tools, grader_tools


def _make_offline_agent3_model(workspace: Path) -> ScriptedChatModel:
    """Create a genuine tool-calling offline ScriptedChatModel for Agent 3."""
    step = {"idx": 0}

    def _responder(messages: list[BaseMessage]) -> dict[str, Any] | str:
        seen_tools = {getattr(m, "name", "") for m in messages if isinstance(m, ToolMessage)}
        step["idx"] += 1

        if "preflight_cost_tool" not in seen_tools:
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": f"tc_a3_pre_{step['idx']}",
                        "name": "preflight_cost_tool",
                        "args": {},
                        "type": "tool_call",
                    }
                ],
            }

        if "render_assets_tool" not in seen_tools:
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": f"tc_a3_render_{step['idx']}",
                        "name": "render_assets_tool",
                        "args": {},
                        "type": "tool_call",
                    }
                ],
            }

        if "stitch_video_tool" not in seen_tools:
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": f"tc_a3_stitch_{step['idx']}",
                        "name": "stitch_video_tool",
                        "args": {},
                        "type": "tool_call",
                    }
                ],
            }

        if "run_media_qa_tool" not in seen_tools:
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": f"tc_a3_qa_{step['idx']}",
                        "name": "run_media_qa_tool",
                        "args": {},
                        "type": "tool_call",
                    }
                ],
            }

        # If a rubric revision requested repair on failed shots, invoke repair_shots_tool once
        raw_qa = read_workspace_json(workspace, "qa_report.json")
        if (
            raw_qa
            and not raw_qa.get("passed", True)
            and raw_qa.get("failed_shot_ids")
            and "repair_shots_tool" not in seen_tools
        ):
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": f"tc_a3_repair_{step['idx']}",
                        "name": "repair_shots_tool",
                        "args": {
                            "failed_shot_ids_json": json.dumps(raw_qa["failed_shot_ids"]),
                        },
                        "type": "tool_call",
                    }
                ],
            }

        return (
            "Completed overlapped asset rendering, single-pass FFmpeg stitch, "
            "and deterministic Media QA."
        )

    return ScriptedChatModel(responder=_responder, model_name="agent3-offline-producer")


async def run_agent3_producer(
    *,
    run_id: str,
    run_input: RunInput,
    shot_sheet: ShotSheet,
    sound_plan: SoundPlan | None,
    creative_bible: CreativeBible,
    providers: Any,
    cost_meter: CostMeter | None = None,
    director_qa: QAReport | None = None,
    work_dir: Path | None = None,
    event_cb: EventCallback | None = None,
    cancel: CancellationToken | None = None,
    main_model: BaseChatModel | str | None = None,
    grader_model: BaseChatModel | str | None = None,
    max_iterations: int = 3,
    raise_on_failure: bool = True,
) -> Agent3Result:
    """Run Agent 3 (Render, Stitch & Final QA Producer DeepAgent) with RubricMiddleware gating."""
    if cancel is not None:
        cancel.raise_if_cancelled()

    base_dir = (
        Path(work_dir).resolve()
        if work_dir is not None
        else Path(tempfile.mkdtemp(prefix="gtm_a3_"))
    )
    workspace = base_dir / "agent3_workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    meter = cost_meter or CostMeter(
        run_id=run_id,
        cap_usd=run_input.cap_usd,
        ledger_path=workspace / "spend_ledger.jsonl",
    )

    worker_tools, grader_tools = _build_agent3_tools(
        run_id=run_id,
        run_input=run_input,
        shot_sheet=shot_sheet,
        sound_plan=sound_plan,
        creative_bible=creative_bible,
        providers=providers,
        cost_meter=meter,
        workspace=workspace,
        director_qa=director_qa,
        event_cb=event_cb,
        cancel=cancel,
    )

    is_offline_mode = run_input.mock or is_offline_provider_bundle(providers)

    resolved_main = main_model
    if resolved_main is None:
        provider_chat = getattr(providers, "chat_model", None)
        if provider_chat is not None and not is_default_unscripted_mock_model(provider_chat):
            resolved_main = provider_chat
        elif is_offline_mode:
            resolved_main = _make_offline_agent3_model(workspace)
        else:
            resolved_main = resolve_live_chat_model("repair", providers=providers)

    if grader_model is not None:
        resolved_grader = grader_model
    elif is_offline_mode or isinstance(resolved_main, ScriptedChatModel):
        resolved_grader = make_tool_driven_grader_model(
            "inspect_qa_and_budget_tool",
            model_name="agent3-rubric-grader",
        )
    else:
        resolved_grader = resolve_live_chat_model("qa", providers=providers)

    prompt = (
        f"Produce final video for run '{run_id}' ({len(shot_sheet.shots)} shots, "
        f"{shot_sheet.aspect} @ {shot_sheet.fps}fps, target={shot_sheet.target_duration_sec}s, "
        f"cap=${meter.cap_usd:.2f})."
    )

    outcome = await run_rubric_deep_agent(
        stage=AGENT3_STAGE,
        run_id=run_id,
        prompt=prompt,
        rubric=AGENT3_RUBRIC,
        main_model=resolved_main,
        grader_model=resolved_grader,
        system_prompt=AGENT3_SYSTEM_PROMPT,
        grader_system_prompt=AGENT3_GRADER_SYSTEM_PROMPT,
        tools=worker_tools,
        grader_tools=grader_tools,
        max_iterations=max_iterations,
        event_cb=event_cb,
        cancel=cancel,
        raise_on_failure=raise_on_failure,
    )

    raw_m = read_workspace_json(workspace, "render_manifest.json")
    raw_qa = read_workspace_json(workspace, "qa_report.json")
    raw_ledger = read_workspace_json(workspace, "cost_ledger.json")

    manifest = (
        RenderManifest.model_validate(raw_m) if raw_m is not None else RenderManifest(run_id=run_id)
    )
    qa_report = (
        QAReport.model_validate(raw_qa) if raw_qa is not None else QAReport(passed=outcome.passed)
    )
    cost_ledger = (
        CostLedger.model_validate(raw_ledger) if raw_ledger is not None else meter.to_ledger()
    )

    return Agent3Result(
        render_manifest=manifest,
        qa_report=qa_report,
        cost_ledger=cost_ledger,
        rubric_evaluations=outcome.evaluations,
        status=outcome.status,
        warnings=list(manifest.warnings),
        workspace_dir=workspace,
    )


__all__ = [
    "AGENT3_RUBRIC",
    "AGENT3_STAGE",
    "Agent3Result",
    "render_assets_overlapped",
    "run_agent3_producer",
]
