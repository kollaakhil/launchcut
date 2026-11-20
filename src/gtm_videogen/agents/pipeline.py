"""Three-agent rubric-gated PipelineRunner orchestrator.

Orchestrates:
- Agent 1: Research & CreativeBible DeepAgent (F14)
- Agent 2: Script & ShotSheet Director DeepAgent (F15)
- Agent 3: Render, Stitch & Final QA Producer DeepAgent (F16)
- Targeted single-shot repair() reusing clipCache & audioCache (BUG-06, P4)
- 4-layer cancellation propagation never overwriting 'cancelled' with 'done' (BUG-05)
- CreativeHistory fingerprint persistence on completion (Pillar 6)
- Canonical SSE event streaming matching PROJECT.md contract
"""

from __future__ import annotations

import asyncio
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gtm_videogen.agents.agent1_research_bible import AGENT1_STAGE, run_agent1_research_bible
from gtm_videogen.agents.agent2_director import AGENT2_STAGE, run_agent2_director
from gtm_videogen.agents.agent3_producer import (
    AGENT3_STAGE,
    _caption_safe_sheet,
    render_assets_overlapped,
    run_agent3_producer,
)
from gtm_videogen.agents.rubric_utils import (
    EventCallback,
    RubricGateError,
    emit_event,
    ensure_live_llm_configured,
)
from gtm_videogen.core.cache_store import CacheStore
from gtm_videogen.core.config import Settings, get_settings
from gtm_videogen.core.cost_meter import CostMeter, read_ledger_total, use_cost_meter
from gtm_videogen.core.job_registry import CancellationToken
from gtm_videogen.core.logging import get_logger
from gtm_videogen.core.media_store import MediaStore
from gtm_videogen.core.run_store import RunStore
from gtm_videogen.media.ffmpeg_stitcher import FFmpegStitcher
from gtm_videogen.media.media_qa import MediaQA
from gtm_videogen.models.schemas import (
    CostEntry,
    CreativeFingerprint,
    RenderedClip,
    RunState,
    Version,
)
from gtm_videogen.providers.audio import AudioProvider
from gtm_videogen.providers.llm_router import LLMRouter
from gtm_videogen.providers.mocks import MockProviderBundle
from gtm_videogen.providers.scraper import BrandScraper
from gtm_videogen.providers.video import VideoProvider
from gtm_videogen.providers.whisper import WhisperProvider
from gtm_videogen.rules.creative_history import CreativeHistoryStore

logger = get_logger("gtm_videogen.agents.pipeline")


@dataclass
class _LiveProviderBundle:
    """Container assembling live providers when run_state.input.mock is False."""

    scraper: BrandScraper
    video: VideoProvider
    audio: AudioProvider
    whisper: WhisperProvider
    llm_router: LLMRouter
    media_store: MediaStore
    cache_store: CacheStore
    chat_model: Any = None


class PipelineRunner:
    """Orchestrates the 3 rubric-gated DeepAgents and targeted repair loop."""

    def __init__(
        self,
        *,
        providers: Any | None = None,
        work_dir: Path | None = None,
        history_store: CreativeHistoryStore | None = None,
        run_store: RunStore | None = None,
        settings: Settings | None = None,
        max_rubric_iterations: int = 3,
    ) -> None:
        self.settings = settings or get_settings()
        self.providers = providers
        self.work_dir = Path(work_dir).resolve() if work_dir is not None else None
        self.history_store = history_store
        self.run_store = run_store
        self.max_rubric_iterations = max(1, int(max_rubric_iterations))

    def _resolve_work_dir(self, run_id: str) -> Path:
        if not run_id or "/" in run_id or "\\" in run_id or ".." in run_id or run_id == ".":
            raise ValueError(f"Unsafe run_id: {run_id!r}")
        if self.work_dir is not None:
            runs_root = (self.work_dir / "runs").resolve()
            run_dir = (runs_root / run_id).resolve()
            try:
                run_dir.relative_to(runs_root)
            except ValueError as exc:
                raise ValueError(f"run_id escapes runs directory: {run_id!r}") from exc
            if run_dir == runs_root:
                raise ValueError(f"Unsafe run_id: {run_id!r}")
            run_dir.mkdir(parents=True, exist_ok=True)
            return run_dir
        runs_root = (self.settings.data_dir / "runs").resolve()
        base = (runs_root / run_id).resolve()
        try:
            base.relative_to(runs_root)
        except ValueError as exc:
            raise ValueError(f"run_id escapes runs directory: {run_id!r}") from exc
        if base == runs_root:
            raise ValueError(f"Unsafe run_id: {run_id!r}")
        try:
            base.mkdir(parents=True, exist_ok=True)
            return base
        except OSError:
            return Path(tempfile.mkdtemp(prefix=f"gtm_run_{run_id}_"))

    def _resolve_history_store(self, run_dir: Path) -> CreativeHistoryStore:
        if self.history_store is not None:
            return self.history_store
        root = self.work_dir if self.work_dir is not None else run_dir.parent
        return CreativeHistoryStore(root / "creative_history.json")

    def _resolve_providers(
        self,
        *,
        run_state: RunState,
        run_dir: Path,
        cost_meter: CostMeter,
        require_llm: bool = False,
    ) -> Any:
        if self.providers is not None:
            return self.providers

        if run_state.input.mock:
            return MockProviderBundle.create(
                work_dir=run_dir,
                cost_meter=cost_meter,
                settings=self.settings,
            )

        if require_llm:
            ensure_live_llm_configured(self.settings)

        media_store = MediaStore(run_dir / "media")
        cache_store = CacheStore(run_dir / "cache")
        return _LiveProviderBundle(
            scraper=BrandScraper(),
            video=VideoProvider(
                settings=self.settings,
                media_store=media_store,
                clip_cache=cache_store.clip_cache,
                cost_meter=cost_meter,
            ),
            audio=AudioProvider(
                settings=self.settings,
                media_store=media_store,
                audio_cache=cache_store.audio_cache,
                cost_meter=cost_meter,
            ),
            whisper=WhisperProvider(settings=self.settings, cost_meter=cost_meter),
            llm_router=LLMRouter(settings=self.settings),
            media_store=media_store,
            cache_store=cache_store,
        )

    @staticmethod
    def _init_repair_cost_meter(run_state: RunState, ledger_path: Path) -> CostMeter:
        """Initialize a CostMeter for repair() preserving prior spend and ledger entries."""
        cost_meter = CostMeter(
            run_id=run_state.run_id,
            cap_usd=run_state.input.cap_usd,
            ledger_path=ledger_path,
        )
        prior_entries: list[CostEntry] = []
        if ledger_path.exists():
            with ledger_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    prior_entries.append(CostEntry.model_validate_json(stripped))
        elif run_state.cost_ledger.entries:
            prior_entries = list(run_state.cost_ledger.entries)

        disk_total = read_ledger_total(ledger_path) if ledger_path.exists() else 0.0
        entries_total = round(sum(e.actual_usd for e in prior_entries), 6)
        prior_spent = max(disk_total, entries_total, float(run_state.cost_ledger.actual_usd))

        cost_meter._entries = prior_entries
        cost_meter._spent_usd = prior_spent
        return cost_meter

    def _save_state(self, run_state: RunState) -> None:
        if self.run_store is not None:
            self.run_store.save(run_state)

    async def _handle_cancelled(
        self,
        run_state: RunState,
        *,
        stage: str | None,
        cancel: CancellationToken | None,
        event_cb: EventCallback | None,
        cost_meter: CostMeter | None = None,
    ) -> RunState:
        if cost_meter is not None:
            run_state.cost_ledger = cost_meter.to_ledger()
        run_state.status = "cancelled"
        run_state.error = cancel.reason if cancel and cancel.reason else "Run cancelled"
        self._save_state(run_state)
        await emit_event(
            event_cb,
            event_type="run_cancelled",
            run_id=run_state.run_id,
            stage=stage,
            data={"reason": run_state.error},
        )
        return run_state

    async def run(
        self,
        run_state: RunState,
        *,
        event_cb: EventCallback | None = None,
        cancel: CancellationToken | None = None,
    ) -> RunState:
        """Execute Agent 1 -> Agent 2 -> Agent 3 end-to-end with rubric gating and SSE events."""
        current_stage: str | None = None
        if cancel is not None and cancel.is_cancelled:
            return await self._handle_cancelled(
                run_state, stage=None, cancel=cancel, event_cb=event_cb
            )

        run_dir = self._resolve_work_dir(run_state.run_id)
        history_store = self._resolve_history_store(run_dir)
        cost_meter = CostMeter(
            run_id=run_state.run_id,
            cap_usd=run_state.input.cap_usd,
            ledger_path=run_dir / "spend_ledger.jsonl",
        )
        providers = self._resolve_providers(
            run_state=run_state,
            run_dir=run_dir,
            cost_meter=cost_meter,
            require_llm=True,
        )

        run_state.status = "working"
        run_state.error = None
        self._save_state(run_state)

        try:
            with use_cost_meter(cost_meter):
                # ------------------------------------------------------------------
                # Stage 1: Agent 1 — Research & CreativeBible
                # ------------------------------------------------------------------
                if cancel is not None and cancel.is_cancelled:
                    return await self._handle_cancelled(
                        run_state,
                        stage=AGENT1_STAGE,
                        cancel=cancel,
                        event_cb=event_cb,
                        cost_meter=cost_meter,
                    )

                current_stage = AGENT1_STAGE
                await emit_event(
                    event_cb,
                    event_type="stage_started",
                    run_id=run_state.run_id,
                    stage=AGENT1_STAGE,
                    data={"brand_name": run_state.input.brand_name},
                )
                a1 = await run_agent1_research_bible(
                    run_input=run_state.input,
                    providers=providers,
                    history_store=history_store,
                    work_dir=run_dir,
                    run_id=run_state.run_id,
                    event_cb=event_cb,
                    cancel=cancel,
                    max_iterations=self.max_rubric_iterations,
                )
                run_state.research = a1.research
                run_state.concepts = a1.concepts
                run_state.creative_bible = a1.creative_bible
                run_state.rubric_evaluations[AGENT1_STAGE] = a1.rubric_evaluations
                self._save_state(run_state)
                await emit_event(
                    event_cb,
                    event_type="stage_completed",
                    run_id=run_state.run_id,
                    stage=AGENT1_STAGE,
                    data={
                        "status": a1.status,
                        "arc_name": a1.creative_bible.narrative_structure.arc_name,
                        "concepts_count": len(a1.concepts),
                    },
                )

                # ------------------------------------------------------------------
                # Stage 2: Agent 2 — Script & ShotSheet Director
                # ------------------------------------------------------------------
                if cancel is not None and cancel.is_cancelled:
                    return await self._handle_cancelled(
                        run_state,
                        stage=AGENT2_STAGE,
                        cancel=cancel,
                        event_cb=event_cb,
                        cost_meter=cost_meter,
                    )

                current_stage = AGENT2_STAGE
                await emit_event(
                    event_cb,
                    event_type="stage_started",
                    run_id=run_state.run_id,
                    stage=AGENT2_STAGE,
                    data={"target_duration_sec": run_state.input.target_duration_sec},
                )
                a2 = await run_agent2_director(
                    run_input=run_state.input,
                    research=a1.research,
                    creative_bible=a1.creative_bible,
                    providers=providers,
                    history_store=history_store,
                    work_dir=run_dir,
                    run_id=run_state.run_id,
                    event_cb=event_cb,
                    cancel=cancel,
                    max_iterations=self.max_rubric_iterations,
                )
                run_state.script = a2.script
                run_state.shot_sheet = a2.shot_sheet
                run_state.sound_plan = a2.sound_plan
                run_state.rubric_evaluations[AGENT2_STAGE] = a2.rubric_evaluations
                self._save_state(run_state)
                await emit_event(
                    event_cb,
                    event_type="stage_completed",
                    run_id=run_state.run_id,
                    stage=AGENT2_STAGE,
                    data={
                        "status": a2.status,
                        "shot_count": len(a2.shot_sheet.shots),
                        "voice_id": a2.sound_plan.voice_id,
                        "originality_score": a2.qa_report.originality_score,
                        "bible_coherence_score": a2.qa_report.bible_coherence_score,
                    },
                )

                # ------------------------------------------------------------------
                # Stage 3: Agent 3 — Render, Stitch & Final QA Producer
                # ------------------------------------------------------------------
                if cancel is not None and cancel.is_cancelled:
                    return await self._handle_cancelled(
                        run_state,
                        stage=AGENT3_STAGE,
                        cancel=cancel,
                        event_cb=event_cb,
                        cost_meter=cost_meter,
                    )

                current_stage = AGENT3_STAGE
                await emit_event(
                    event_cb,
                    event_type="stage_started",
                    run_id=run_state.run_id,
                    stage=AGENT3_STAGE,
                    data={"shot_count": len(a2.shot_sheet.shots)},
                )
                a3 = await run_agent3_producer(
                    run_id=run_state.run_id,
                    run_input=run_state.input,
                    shot_sheet=a2.shot_sheet,
                    sound_plan=a2.sound_plan,
                    creative_bible=a1.creative_bible,
                    providers=providers,
                    cost_meter=cost_meter,
                    director_qa=a2.qa_report,
                    work_dir=run_dir,
                    event_cb=event_cb,
                    cancel=cancel,
                    max_iterations=self.max_rubric_iterations,
                )
                run_state.render_manifest = a3.render_manifest
                run_state.qa_report = a3.qa_report
                run_state.cost_ledger = a3.cost_ledger
                run_state.rubric_evaluations[AGENT3_STAGE] = a3.rubric_evaluations
                run_state.warnings = list(dict.fromkeys([*run_state.warnings, *a3.warnings]))

                await emit_event(
                    event_cb,
                    event_type="cost_update",
                    run_id=run_state.run_id,
                    stage=AGENT3_STAGE,
                    data={
                        "estimated_usd": a3.cost_ledger.estimated_usd,
                        "actual_usd": a3.cost_ledger.actual_usd,
                        "cap_usd": a3.cost_ledger.cap_usd,
                    },
                )
                await emit_event(
                    event_cb,
                    event_type="stage_completed",
                    run_id=run_state.run_id,
                    stage=AGENT3_STAGE,
                    data={
                        "status": a3.status,
                        "stitched_path": a3.render_manifest.stitched_path,
                        "qa_passed": a3.qa_report.passed,
                    },
                )

                # BUG-05: Never overwrite 'cancelled' with 'done'
                if (cancel is not None and cancel.is_cancelled) or run_state.status == "cancelled":
                    return await self._handle_cancelled(
                        run_state,
                        stage=AGENT3_STAGE,
                        cancel=cancel,
                        event_cb=event_cb,
                        cost_meter=cost_meter,
                    )

                # Pillar 6: Persist CreativeFingerprint to CreativeHistory on completion
                music_genre = (
                    a1.creative_bible.sonic_identity.music_brief.genre
                    if a1.creative_bible.sonic_identity.music_brief is not None
                    else None
                )
                history_store.record(
                    CreativeFingerprint(
                        run_id=run_state.run_id,
                        brand_url=run_state.input.brand_url,
                        arc_name=a1.creative_bible.narrative_structure.arc_name,
                        hook_opening=a1.creative_bible.concept_thesis,
                        visual_world=a1.creative_bible.visual_language.world,
                        lighting_style=a1.creative_bible.visual_language.lighting_style,
                        camera_style=a1.creative_bible.camera_grammar.dominant_style,
                        music_genre=music_genre,
                        voice_id=a2.sound_plan.voice_id or None,
                        outcome="completed",
                    )
                )

                version_id = f"v{len(run_state.versions) + 1}"
                a3.render_manifest.version_id = version_id
                run_state.versions.append(
                    Version(
                        version_id=version_id,
                        label=a1.creative_bible.narrative_structure.arc_name,
                        creative_bible=a1.creative_bible,
                        shot_sheet=a2.shot_sheet,
                        sound_plan=a2.sound_plan,
                        render_manifest=a3.render_manifest,
                        qa_report=a3.qa_report,
                        warnings=list(a3.warnings),
                        error=None,
                    )
                )

                run_state.status = "done"
                run_state.error = None
                self._save_state(run_state)

                await emit_event(
                    event_cb,
                    event_type="run_completed",
                    run_id=run_state.run_id,
                    stage=None,
                    data={
                        "status": "done",
                        "stitched_path": a3.render_manifest.stitched_path,
                        "actual_usd": a3.cost_ledger.actual_usd,
                        "warnings_count": len(run_state.warnings),
                    },
                )
                return run_state

        except asyncio.CancelledError:
            return await self._handle_cancelled(
                run_state,
                stage=current_stage,
                cancel=cancel,
                event_cb=event_cb,
                cost_meter=cost_meter,
            )
        except Exception as exc:
            logger.exception("Pipeline run %s failed at stage %s", run_state.run_id, current_stage)
            if isinstance(exc, RubricGateError):
                stage_key = exc.stage or current_stage or "unknown"
                run_state.rubric_evaluations[stage_key] = list(exc.evaluations)
            run_state.cost_ledger = cost_meter.to_ledger()
            if (cancel is not None and cancel.is_cancelled) or run_state.status == "cancelled":
                return await self._handle_cancelled(
                    run_state,
                    stage=current_stage,
                    cancel=cancel,
                    event_cb=event_cb,
                    cost_meter=cost_meter,
                )
            run_state.status = "failed"
            run_state.error = str(exc)
            self._save_state(run_state)
            await emit_event(
                event_cb,
                event_type="run_failed",
                run_id=run_state.run_id,
                stage=current_stage,
                data={"error": str(exc)},
            )
            return run_state

    async def repair(
        self,
        run_state: RunState,
        *,
        failed_shot_ids: list[str] | None = None,
        event_cb: EventCallback | None = None,
        cancel: CancellationToken | None = None,
    ) -> RunState:
        """Targeted single-shot/audio repair reusing cached clips & audio (BUG-06, P4)."""
        if cancel is not None and cancel.is_cancelled:
            return await self._handle_cancelled(
                run_state, stage="repair", cancel=cancel, event_cb=event_cb
            )

        if run_state.shot_sheet is None or run_state.creative_bible is None:
            raise ValueError("Cannot repair a run before shot_sheet and creative_bible exist.")

        valid_shot_ids = {s.id for s in run_state.shot_sheet.shots}
        if failed_shot_ids is not None:
            unknown_ids = [sid for sid in failed_shot_ids if sid not in valid_shot_ids]
            if unknown_ids:
                raise ValueError(
                    f"Unknown failed_shot_ids {unknown_ids!r}; "
                    f"valid shot IDs are {sorted(valid_shot_ids)!r}."
                )
            if not failed_shot_ids:
                raise ValueError("failed_shot_ids must not be empty when explicitly provided.")
            target_ids = list(dict.fromkeys(failed_shot_ids))
        else:
            qa_failed = [
                sid
                for sid in (run_state.qa_report.failed_shot_ids if run_state.qa_report else [])
                if sid in valid_shot_ids
            ]
            target_ids = qa_failed or [run_state.shot_sheet.shots[0].id]
        target_set = set(target_ids)

        run_dir = self._resolve_work_dir(run_state.run_id)
        repair_dir = run_dir / "repair_workspace"
        repair_dir.mkdir(parents=True, exist_ok=True)

        cost_meter = self._init_repair_cost_meter(run_state, run_dir / "spend_ledger.jsonl")
        providers = self._resolve_providers(
            run_state=run_state,
            run_dir=run_dir,
            cost_meter=cost_meter,
            require_llm=False,
        )

        existing_clips: dict[str, RenderedClip] = {}
        if run_state.render_manifest is not None:
            existing_clips = {c.shot_id: c for c in run_state.render_manifest.clips}

        run_state.status = "working"
        run_state.error = None
        self._save_state(run_state)

        await emit_event(
            event_cb,
            event_type="stage_started",
            run_id=run_state.run_id,
            stage="repair",
            data={"failed_shot_ids": target_ids},
        )

        try:
            with use_cost_meter(cost_meter):
                clips, audio_assets, warnings = await render_assets_overlapped(
                    run_id=run_state.run_id,
                    shot_sheet=run_state.shot_sheet,
                    sound_plan=run_state.sound_plan,
                    creative_bible=run_state.creative_bible,
                    providers=providers,
                    work_dir=repair_dir,
                    render_mode=run_state.input.render_mode,
                    event_cb=event_cb,
                    cancel=cancel,
                    existing_clips=existing_clips,
                    repair_shot_ids=target_set,
                )

                if cancel is not None and cancel.is_cancelled:
                    return await self._handle_cancelled(
                        run_state,
                        stage="repair",
                        cancel=cancel,
                        event_cb=event_cb,
                        cost_meter=cost_meter,
                    )

                stitcher = FFmpegStitcher()
                next_version_id = f"v{len(run_state.versions) + 1}"
                output_path = repair_dir / f"{run_state.run_id}_{next_version_id}.mp4"
                burn_captions = any(
                    s.on_screen_text
                    and s.on_screen_text.strip()
                    and s.engine not in ("motion_card", "manim")
                    for s in run_state.shot_sheet.shots
                )
                stitch_sheet = (
                    _caption_safe_sheet(run_state.shot_sheet)
                    if burn_captions
                    else run_state.shot_sheet
                )
                stitch_w = clips[0].width if (run_state.input.mock and clips) else None
                stitch_h = clips[0].height if (run_state.input.mock and clips) else None
                manifest = await stitcher.stitch(
                    clips=clips,
                    sound_plan=run_state.sound_plan,
                    audio_assets=audio_assets,
                    sheet=stitch_sheet,
                    bible=run_state.creative_bible,
                    output_path=output_path,
                    cancel=cancel,
                    run_id=run_state.run_id,
                    width=stitch_w,
                    height=stitch_h,
                    burn_captions=burn_captions,
                )
                merged_warnings = list(dict.fromkeys([*warnings, *manifest.warnings]))
                manifest.warnings = merged_warnings
                manifest.error = None

                expected_vo = (
                    " ".join(
                        s.vo_line.strip()
                        for s in run_state.shot_sheet.shots
                        if s.vo_line and s.vo_line.strip()
                    )
                    or None
                )
                qa = await MediaQA().evaluate(
                    Path(manifest.stitched_path or output_path),
                    run_state.shot_sheet,
                    expected_vo=expected_vo,
                    run_ocr=True,
                    whisper=getattr(providers, "whisper", None),
                    cancel=cancel,
                )
                if run_state.qa_report is not None:
                    qa.originality_score = run_state.qa_report.originality_score
                    qa.bible_coherence_score = run_state.qa_report.bible_coherence_score

                if (cancel is not None and cancel.is_cancelled) or run_state.status == "cancelled":
                    return await self._handle_cancelled(
                        run_state,
                        stage="repair",
                        cancel=cancel,
                        event_cb=event_cb,
                        cost_meter=cost_meter,
                    )

                version_id = next_version_id
                manifest.version_id = version_id
                run_state.versions.append(
                    Version(
                        version_id=version_id,
                        label=f"repair:{','.join(target_ids)}",
                        creative_bible=run_state.creative_bible,
                        shot_sheet=run_state.shot_sheet,
                        sound_plan=run_state.sound_plan,
                        render_manifest=manifest,
                        qa_report=qa,
                        warnings=list(merged_warnings),
                        error=None,
                    )
                )

                reused_ids = [c.shot_id for c in clips if c.cache_hit]
                run_state.render_manifest = manifest
                run_state.qa_report = qa
                run_state.cost_ledger = cost_meter.to_ledger()
                run_state.warnings = list(dict.fromkeys([*run_state.warnings, *merged_warnings]))
                run_state.repair_result = {
                    "version_id": version_id,
                    "repaired_shot_ids": target_ids,
                    "reused_shot_ids": reused_ids,
                    "qa_passed": qa.passed,
                    "stitched_path": manifest.stitched_path,
                    "warnings": merged_warnings,
                }
                run_state.status = "done"
                run_state.error = None
                self._save_state(run_state)

                await emit_event(
                    event_cb,
                    event_type="stage_completed",
                    run_id=run_state.run_id,
                    stage="repair",
                    data=run_state.repair_result,
                )
                await emit_event(
                    event_cb,
                    event_type="run_completed",
                    run_id=run_state.run_id,
                    stage="repair",
                    data={
                        "status": "done",
                        "stitched_path": manifest.stitched_path,
                        "repair_result": run_state.repair_result,
                    },
                )
                return run_state

        except asyncio.CancelledError:
            return await self._handle_cancelled(
                run_state,
                stage="repair",
                cancel=cancel,
                event_cb=event_cb,
                cost_meter=cost_meter,
            )
        except Exception as exc:
            logger.exception("Pipeline repair %s failed", run_state.run_id)
            run_state.cost_ledger = cost_meter.to_ledger()
            if (cancel is not None and cancel.is_cancelled) or run_state.status == "cancelled":
                return await self._handle_cancelled(
                    run_state,
                    stage="repair",
                    cancel=cancel,
                    event_cb=event_cb,
                    cost_meter=cost_meter,
                )
            run_state.status = "failed"
            run_state.error = str(exc)
            self._save_state(run_state)
            await emit_event(
                event_cb,
                event_type="run_failed",
                run_id=run_state.run_id,
                stage="repair",
                data={"error": str(exc)},
            )
            return run_state


__all__ = ["PipelineRunner"]
