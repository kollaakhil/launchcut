"""FastAPI routes, SSE event broker, HTTP Range media streaming, and PipelineRunnerProtocol.

Addresses F17 and fixes:
- BUG-02 / BUG-22: Strictly serializable JobStatusDTO + `other_active_jobs_count` on GET /api/jobs.
- BUG-03: Strict Pydantic `RunInput` (`extra="forbid"`) rejecting unknown fields with HTTP 422.
- BUG-05: 4-layer cancellation via unified `JobRegistry` + `run_cancelled` SSE emission.
- BUG-06: Consistent async job semantics for POST /api/runs/{id}/repair persisting `repair_result`.
- BUG-13: Ownership authorization (`load_owned_run`) on all mutating `/api/runs/{id}/*` routes.
- BUG-19: Automatic orphan `'working'` run healing via `RunStore` + `JobRegistry`.
- P7: Path-based streaming media delivery with RFC 7233 HTTP 206 Range / 416 support.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import mimetypes
import re
import shutil
from collections import OrderedDict
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Protocol, runtime_checkable
from uuid import uuid4

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict

from gtm_videogen.api.middleware import MAX_RUN_ID_LENGTH, get_actor, load_owned_run
from gtm_videogen.core.config import Settings
from gtm_videogen.core.job_registry import CancellationToken, JobRegistry
from gtm_videogen.core.media_store import MediaStore, MediaStoreError
from gtm_videogen.core.run_store import RunStore
from gtm_videogen.media.manim_renderer import manim_diagnostics
from gtm_videogen.models.schemas import CostLedger, RunInput, RunState

EventCallback = Callable[[dict[str, Any]], Awaitable[None] | None]

TERMINAL_EVENT_TYPES: frozenset[str] = frozenset({"run_completed", "run_cancelled", "run_failed"})
_RANGE_RE = re.compile(r"^bytes=(\d*)-(\d*)$")
_CHUNK_SIZE = 65_536
_MAX_MEDIA_ID_LENGTH = 1024
_MAX_MEDIA_SEGMENT_LENGTH = 200


@runtime_checkable
class PipelineRunnerProtocol(Protocol):
    """Protocol decoupling the API/CLI layer from `gtm_videogen.agents.pipeline.PipelineRunner`."""

    async def run(
        self,
        run_state: RunState,
        *,
        event_cb: EventCallback | None = None,
        cancel: CancellationToken | None = None,
    ) -> RunState: ...

    async def repair(
        self,
        run_state: RunState,
        *,
        failed_shot_ids: list[str] | None = None,
        event_cb: EventCallback | None = None,
        cancel: CancellationToken | None = None,
    ) -> RunState: ...


class RepairRequest(BaseModel):
    """Optional request payload for POST /api/runs/{run_id}/repair."""

    model_config = ConfigDict(extra="forbid")

    failed_shot_ids: list[str] | None = None


def make_sse_event(
    event_type: str,
    run_id: str,
    *,
    stage: str | None = None,
    data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Construct a spec-compliant SSE event payload."""
    return {
        "type": event_type,
        "run_id": run_id,
        "timestamp": datetime.now(UTC).isoformat(),
        "stage": stage,
        "data": data if isinstance(data, dict) else {},
    }


def _normalize_event(raw: dict[str, Any], run_id: str) -> dict[str, Any]:
    """Ensure every emitted event conforms to PROJECT.md's SSE payload shape."""
    ev_type = str(raw.get("type") or "warning")
    ts = raw.get("timestamp")
    if not isinstance(ts, str) or not ts:
        ts = datetime.now(UTC).isoformat()
    stage = raw.get("stage")
    if stage is not None and not isinstance(stage, str):
        stage = str(stage)
    data = raw.get("data")
    if not isinstance(data, dict):
        data = {}
    return {
        "type": ev_type,
        "run_id": str(raw.get("run_id") or run_id),
        "timestamp": ts,
        "stage": stage,
        "data": data,
    }


class RunEventBroker:
    """In-memory per-run SSE pub/sub broker with bounded replay history.

    Buffering history prevents a race where a fast run emits `stage_started` or
    `run_completed` before the client opens `GET /api/runs/{run_id}/events`.
    """

    def __init__(self, *, max_runs: int = 500, max_events_per_run: int = 500) -> None:
        self._max_runs = max_runs
        self._max_events_per_run = max_events_per_run
        self._history: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
        self._subscribers: dict[str, set[asyncio.Queue[dict[str, Any]]]] = {}

    def reset_run(self, run_id: str) -> None:
        """Clear buffered event history before launching a fresh start/repair job."""
        self._history[run_id] = []
        self._history.move_to_end(run_id)
        self._evict_old_runs()

    def has_history(self, run_id: str) -> bool:
        return bool(self._history.get(run_id))

    def has_terminal_event(self, run_id: str) -> bool:
        return any(ev.get("type") in TERMINAL_EVENT_TYPES for ev in self._history.get(run_id, []))

    def subscriber_count(self, run_id: str) -> int:
        return len(self._subscribers.get(run_id, set()))

    def publish(self, run_id: str, event: dict[str, Any]) -> dict[str, Any]:
        """Record an event in run history and broadcast to all live subscribers."""
        normalized = _normalize_event(event, run_id)
        history = self._history.get(run_id)
        if history is None:
            history = []
            self._history[run_id] = history
        self._history.move_to_end(run_id)
        history.append(normalized)
        if len(history) > self._max_events_per_run:
            del history[: len(history) - self._max_events_per_run]
        self._evict_old_runs()

        for queue in list(self._subscribers.get(run_id, set())):
            try:
                queue.put_nowait(normalized)
            except asyncio.QueueFull:
                # Drop oldest unread item so slow subscribers never stall the pipeline
                try:
                    queue.get_nowait()
                    queue.put_nowait(normalized)
                except asyncio.QueueEmpty:
                    pass
        return normalized

    async def subscribe(
        self,
        run_id: str,
        *,
        request: Request | None = None,
        heartbeat_sec: float = 15.0,
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield buffered history and then live events until a terminal event or disconnect."""
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=self._max_events_per_run)
        subs = self._subscribers.setdefault(run_id, set())
        subs.add(queue)
        try:
            snapshot = list(self._history.get(run_id, []))
            replayed_count = len(snapshot)
            for ev in snapshot:
                yield ev
                if ev.get("type") in TERMINAL_EVENT_TYPES:
                    return

            # Drain any events queued during replay that were already in snapshot
            # (asyncio is single-threaded so snapshot and queue registration are atomic
            # before the first yield, meaning anything in `queue` arrived after snapshot).
            _ = replayed_count
            while True:
                if request is not None and await request.is_disconnected():
                    return
                try:
                    ev = await asyncio.wait_for(queue.get(), timeout=heartbeat_sec)
                except TimeoutError:
                    continue
                yield ev
                if ev.get("type") in TERMINAL_EVENT_TYPES:
                    return
        finally:
            subs.discard(queue)
            if not subs:
                self._subscribers.pop(run_id, None)

    def _evict_old_runs(self) -> None:
        while len(self._history) > self._max_runs:
            oldest_run_id, _ = next(iter(self._history.items()))
            if oldest_run_id in self._subscribers:
                # Keep active subscriber runs at the end
                self._history.move_to_end(oldest_run_id)
                break
            self._history.popitem(last=False)


@dataclass
class AppServices:
    """Dependency container attached to `app.state.services`."""

    settings: Settings
    data_dir: Path
    run_store: RunStore
    media_store: MediaStore
    job_registry: JobRegistry
    event_broker: RunEventBroker
    runner: PipelineRunnerProtocol | None = None
    _run_generations: dict[str, int] = field(default_factory=dict, repr=False)
    _run_locks: dict[str, asyncio.Lock] = field(default_factory=dict, repr=False)

    def run_lock(self, run_id: str) -> asyncio.Lock:
        """Return the per-run asyncio.Lock for serializing launch/cancel transitions."""
        lock = self._run_locks.get(run_id)
        if lock is None:
            lock = asyncio.Lock()
            self._run_locks[run_id] = lock
        return lock

    def current_generation(self, run_id: str) -> int:
        """Return the current monotonic generation/epoch for `run_id`."""
        return self._run_generations.get(run_id, 0)

    def next_generation(self, run_id: str) -> int:
        """Advance and return the monotonic generation/epoch for `run_id`."""
        gen = self._run_generations.get(run_id, 0) + 1
        self._run_generations[run_id] = gen
        return gen

    def is_current_generation(self, run_id: str, generation: int) -> bool:
        """Return True iff `generation` is still the active generation for `run_id`."""
        return self._run_generations.get(run_id, 0) == generation

    def resolve_runner(self, run_state: RunState) -> PipelineRunnerProtocol:
        """Return injected runner or lazy-import `PipelineRunner` for live/mock execution."""
        if self.runner is not None:
            return self.runner

        from gtm_videogen.agents.pipeline import PipelineRunner
        from gtm_videogen.rules.creative_history import CreativeHistoryStore

        run_work_dir = self.data_dir / "work" / run_state.run_id
        run_work_dir.mkdir(parents=True, exist_ok=True)
        history_store = CreativeHistoryStore(self.data_dir / "creative_history.json")

        if run_state.input.mock:
            from gtm_videogen.providers.mocks import MockProviderBundle

            bundle = MockProviderBundle.create(work_dir=run_work_dir)
            return PipelineRunner(
                providers=bundle,
                work_dir=run_work_dir,
                history_store=history_store,
            )

        return PipelineRunner(
            work_dir=run_work_dir,
            history_store=history_store,
        )


def _get_services(request: Request) -> AppServices:
    return request.app.state.services  # type: ignore[no-any-return]


def _parse_byte_range(range_header: str, file_size: int) -> tuple[int, int] | None:
    """Parse a single RFC 7233 `bytes=...` Range header. Returns `(start, end)` or None (416)."""
    if file_size <= 0:
        return None
    header = range_header.strip()
    match = _RANGE_RE.match(header)
    if not match:
        return None

    start_s, end_s = match.group(1), match.group(2)
    if not start_s and not end_s:
        return None

    if not start_s:
        # Suffix-byte-range-spec: bytes=-<suffix_length>
        suffix = int(end_s)
        if suffix <= 0:
            return None
        start = max(0, file_size - suffix)
        end = file_size - 1
        return start, end

    start = int(start_s)
    if start >= file_size:
        return None

    end = file_size - 1 if not end_s else min(int(end_s), file_size - 1)

    if start > end:
        return None
    return start, end


def _iter_file_range(path: Path, start: int, length: int) -> AsyncIterator[bytes]:
    async def _gen() -> AsyncIterator[bytes]:
        remaining = length
        with path.open("rb") as fh:
            fh.seek(start)
            while remaining > 0:
                chunk = fh.read(min(_CHUNK_SIZE, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    return _gen()


ServicesDep = Annotated[AppServices, Depends(_get_services)]
ActorDep = Annotated[str, Depends(get_actor)]

router = APIRouter(prefix="/api")


@router.get("/health")
async def health_check(services: ServicesDep) -> dict[str, Any]:
    """Report live local tool availability (`ffmpeg`, `ffprobe`, `tesseract`, `manim`)."""
    settings = services.settings
    ffmpeg_ok = shutil.which(settings.ffmpeg_bin) is not None
    ffprobe_ok = shutil.which(settings.ffprobe_bin) is not None
    tesseract_ok = shutil.which(settings.tesseract_bin) is not None
    manim_info = manim_diagnostics()

    llm_ready = any(
        bool(k)
        for k in (
            settings.openrouter_api_key,
            settings.openai_api_key,
            settings.anthropic_api_key,
            settings.google_api_key,
        )
    )
    fal_ready = bool(settings.fal_key)

    return {
        "status": "ok" if (ffmpeg_ok and ffprobe_ok) else "degraded",
        "tools": {
            "ffmpeg": ffmpeg_ok,
            "ffprobe": ffprobe_ok,
            "tesseract": tesseract_ok,
            "manim": bool(manim_info.get("available", False)),
        },
        "api_keys": {
            "llm": llm_ready,
            "fal": fal_ready,
            "live_ready": llm_ready and fal_ready,
        },
        "manim": manim_info,
    }


@router.post("/runs", status_code=201)
async def create_run(
    run_input: RunInput,
    actor: ActorDep,
    services: ServicesDep,
) -> dict[str, Any]:
    """Create and persist a new RunState with strict Pydantic validation (BUG-03)."""
    run_id = f"run_{uuid4().hex[:12]}"
    state = RunState(
        run_id=run_id,
        created_by=actor,
        status="idle",
        input=run_input,
        cost_ledger=CostLedger(run_id=run_id, cap_usd=run_input.cap_usd),
    )
    services.run_store.save(state)
    return state.model_dump(mode="json")


@router.get("/runs")
async def list_runs(
    services: ServicesDep,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> dict[str, Any]:
    """List persisted runs (auto-healing orphaned `'working'` states per BUG-19)."""
    runs = services.run_store.list_runs(limit=limit)
    return {"runs": [r.model_dump(mode="json") for r in runs]}


@router.get("/runs/{run_id}")
async def get_run(
    run_id: str,
    services: ServicesDep,
) -> dict[str, Any]:
    """Fetch a single RunState (auto-heals orphaned `'working'` runs per BUG-19)."""
    if not run_id or len(run_id) > MAX_RUN_ID_LENGTH:
        raise HTTPException(status_code=400, detail="Invalid run_id")
    try:
        state = services.run_store.load(run_id)
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=404, detail="Invalid run_id") from exc

    if state is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found")
    return state.model_dump(mode="json")


@router.post("/runs/{run_id}/start", status_code=202)
async def start_run(
    run_id: str,
    actor: ActorDep,
    services: ServicesDep,
) -> dict[str, Any]:
    """Launch the async 3-agent pipeline for a run via `JobRegistry`."""
    run_state = load_owned_run(run_id, services.run_store, actor)
    async with services.run_lock(run_id):
        if services.job_registry.is_run_active(run_id):
            raise HTTPException(
                status_code=409, detail=f"Run {run_id!r} already has an active job"
            )

        run_state = services.run_store.load(run_id) or run_state
        gen = services.next_generation(run_id)
        run_state.status = "working"
        run_state.error = None
        services.event_broker.reset_run(run_id)

        deferred_terminal: list[dict[str, Any]] = []

        async def _on_event(ev: dict[str, Any]) -> None:
            if not services.is_current_generation(run_id, gen):
                return
            if ev.get("type") in TERMINAL_EVENT_TYPES:
                deferred_terminal.append(ev)
                return
            services.event_broker.publish(run_id, ev)

        async def _run_job(cancel_token: CancellationToken) -> RunState | None:
            if not services.is_current_generation(run_id, gen):
                return None
            services.event_broker.publish(
                run_id,
                make_sse_event("stage_started", run_id, stage="pipeline", data={"action": "start"}),
            )
            try:
                active_runner = services.resolve_runner(run_state)
                result = await active_runner.run(
                    run_state,
                    event_cb=_on_event,
                    cancel=cancel_token,
                )
                if not services.is_current_generation(run_id, gen):
                    return None
                if cancel_token.is_cancelled:
                    result.status = "cancelled"
                    services.run_store.save(result)
                    if not services.event_broker.has_terminal_event(run_id):
                        services.event_broker.publish(
                            run_id,
                            make_sse_event(
                                "run_cancelled",
                                run_id,
                                stage="pipeline",
                                data={"reason": cancel_token.reason or "Cancelled"},
                            ),
                        )
                    return result

                if result.status == "working":
                    result.status = "done"
                services.run_store.save(result)
                if not services.event_broker.has_terminal_event(run_id):
                    if deferred_terminal:
                        services.event_broker.publish(run_id, deferred_terminal[-1])
                    else:
                        terminal_type = (
                            "run_failed" if result.status == "failed" else "run_completed"
                        )
                        services.event_broker.publish(
                            run_id,
                            make_sse_event(
                                terminal_type,
                                run_id,
                                stage="pipeline",
                                data={"status": result.status, "error": result.error},
                            ),
                        )
                return result
            except asyncio.CancelledError:
                if not services.is_current_generation(run_id, gen):
                    raise
                run_state.status = "cancelled"
                services.run_store.save(run_state)
                if not services.event_broker.has_terminal_event(run_id):
                    services.event_broker.publish(
                        run_id,
                        make_sse_event(
                            "run_cancelled",
                            run_id,
                            stage="pipeline",
                            data={"reason": cancel_token.reason or "Cancelled"},
                        ),
                    )
                raise
            except Exception as exc:
                if not services.is_current_generation(run_id, gen):
                    return None
                if cancel_token.is_cancelled:
                    run_state.status = "cancelled"
                    services.run_store.save(run_state)
                    if not services.event_broker.has_terminal_event(run_id):
                        services.event_broker.publish(
                            run_id,
                            make_sse_event(
                                "run_cancelled",
                                run_id,
                                stage="pipeline",
                                data={"reason": cancel_token.reason or "Cancelled"},
                            ),
                        )
                    return run_state

                run_state.status = "failed"
                run_state.error = str(exc)
                services.run_store.save(run_state)
                if not services.event_broker.has_terminal_event(run_id):
                    services.event_broker.publish(
                        run_id,
                        make_sse_event(
                            "run_failed",
                            run_id,
                            stage="pipeline",
                            data={"error": str(exc)},
                        ),
                    )
                raise

        job = await services.job_registry.submit(run_id, _run_job, stage="pipeline")
        services.run_store.save(run_state)
    return {"run_id": run_id, "job": job.model_dump(mode="json")}


@router.get("/runs/{run_id}/events")
async def stream_run_events(
    run_id: str,
    request: Request,
    services: ServicesDep,
) -> StreamingResponse:
    """Stream live and replayed SSE events for `run_id` (`text/event-stream`)."""
    if not run_id or len(run_id) > MAX_RUN_ID_LENGTH:
        raise HTTPException(status_code=400, detail="Invalid run_id")
    try:
        state = services.run_store.load(run_id)
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=404, detail="Invalid run_id") from exc

    if state is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found")

    async def _sse_generator() -> AsyncIterator[str]:
        # If run is already terminal on disk and no broker history exists, emit terminal event
        if (
            not services.event_broker.has_history(run_id)
            and not services.job_registry.is_run_active(run_id)
            and state.status in ("done", "cancelled", "failed", "interrupted")
        ):
            terminal_map = {
                "done": "run_completed",
                "cancelled": "run_cancelled",
                "failed": "run_failed",
                "interrupted": "run_failed",
            }
            error_msg = (
                state.error or "Run was interrupted"
                if state.status == "interrupted"
                else state.error
            )
            ev = make_sse_event(
                terminal_map[state.status],
                run_id,
                stage="pipeline",
                data={"status": state.status, "error": error_msg},
            )
            yield f"event: {ev['type']}\ndata: {json.dumps(ev)}\n\n"
            return

        async for ev in services.event_broker.subscribe(run_id, request=request):
            yield f"event: {ev['type']}\ndata: {json.dumps(ev)}\n\n"

    return StreamingResponse(
        _sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/runs/{run_id}/cancel")
async def cancel_run(
    run_id: str,
    actor: ActorDep,
    services: ServicesDep,
) -> dict[str, Any]:
    """Trigger 4-layer cancellation for `run_id` via `JobRegistry` (BUG-05, BUG-13)."""
    run_state = load_owned_run(run_id, services.run_store, actor)
    async with services.run_lock(run_id):
        run_state = services.run_store.load(run_id) or run_state
        gen = services.current_generation(run_id)
        cancelled_jobs = await services.job_registry.cancel_run(
            run_id, reason="Run cancelled by user"
        )
        if not services.is_current_generation(run_id, gen):
            return {
                "run_id": run_id,
                "status": "cancelled",
                "cancelled_jobs": cancelled_jobs,
            }
        run_state.status = "cancelled"
        services.run_store.save(run_state)
        if not services.event_broker.has_terminal_event(run_id):
            services.event_broker.publish(
                run_id,
                make_sse_event(
                    "run_cancelled",
                    run_id,
                    stage="pipeline",
                    data={"cancelled_jobs": cancelled_jobs},
                ),
            )
    return {
        "run_id": run_id,
        "status": "cancelled",
        "cancelled_jobs": cancelled_jobs,
    }


@router.post("/runs/{run_id}/repair", status_code=202)
async def repair_run(
    run_id: str,
    actor: ActorDep,
    services: ServicesDep,
    payload: Annotated[RepairRequest | None, Body()] = None,
) -> dict[str, Any]:
    """Launch async targeted single-shot repair via `JobRegistry` (BUG-06, BUG-13)."""
    run_state = load_owned_run(run_id, services.run_store, actor)
    failed_shot_ids = payload.failed_shot_ids if payload is not None else None
    async with services.run_lock(run_id):
        if services.job_registry.is_run_active(run_id):
            raise HTTPException(
                status_code=409, detail=f"Run {run_id!r} already has an active job"
            )

        run_state = services.run_store.load(run_id) or run_state
        gen = services.next_generation(run_id)
        run_state.status = "working"
        run_state.error = None
        services.event_broker.reset_run(run_id)

        deferred_terminal: list[dict[str, Any]] = []

        async def _on_event(ev: dict[str, Any]) -> None:
            if not services.is_current_generation(run_id, gen):
                return
            if ev.get("type") in TERMINAL_EVENT_TYPES:
                deferred_terminal.append(ev)
                return
            services.event_broker.publish(run_id, ev)

        async def _repair_job(cancel_token: CancellationToken) -> RunState | None:
            if not services.is_current_generation(run_id, gen):
                return None
            services.event_broker.publish(
                run_id,
                make_sse_event(
                    "stage_started",
                    run_id,
                    stage="repair",
                    data={"failed_shot_ids": failed_shot_ids or []},
                ),
            )
            try:
                active_runner = services.resolve_runner(run_state)
                res = active_runner.repair(
                    run_state,
                    failed_shot_ids=failed_shot_ids,
                    event_cb=_on_event,
                    cancel=cancel_token,
                )
                repaired = await res if inspect.isawaitable(res) else res

                if not services.is_current_generation(run_id, gen):
                    return None

                if cancel_token.is_cancelled:
                    repaired.status = "cancelled"
                    services.run_store.save(repaired)
                    if not services.event_broker.has_terminal_event(run_id):
                        services.event_broker.publish(
                            run_id,
                            make_sse_event(
                                "run_cancelled",
                                run_id,
                                stage="repair",
                                data={"reason": cancel_token.reason or "Cancelled"},
                            ),
                        )
                    return repaired

                if repaired.repair_result is None and repaired.status != "failed":
                    repaired.repair_result = {
                        "repaired_shot_ids": list(failed_shot_ids or []),
                        "status": "completed",
                    }
                if repaired.status == "working":
                    repaired.status = "done"
                services.run_store.save(repaired)
                if not services.event_broker.has_terminal_event(run_id):
                    if deferred_terminal:
                        services.event_broker.publish(run_id, deferred_terminal[-1])
                    else:
                        terminal_type = (
                            "run_failed" if repaired.status == "failed" else "run_completed"
                        )
                        services.event_broker.publish(
                            run_id,
                            make_sse_event(
                                terminal_type,
                                run_id,
                                stage="repair",
                                data={"repair_result": repaired.repair_result},
                            ),
                        )
                return repaired
            except asyncio.CancelledError:
                if not services.is_current_generation(run_id, gen):
                    raise
                run_state.status = "cancelled"
                services.run_store.save(run_state)
                if not services.event_broker.has_terminal_event(run_id):
                    services.event_broker.publish(
                        run_id,
                        make_sse_event(
                            "run_cancelled",
                            run_id,
                            stage="repair",
                            data={"reason": cancel_token.reason or "Cancelled"},
                        ),
                    )
                raise
            except Exception as exc:
                if not services.is_current_generation(run_id, gen):
                    return None
                if cancel_token.is_cancelled:
                    run_state.status = "cancelled"
                    services.run_store.save(run_state)
                    if not services.event_broker.has_terminal_event(run_id):
                        services.event_broker.publish(
                            run_id,
                            make_sse_event(
                                "run_cancelled",
                                run_id,
                                stage="repair",
                                data={"reason": cancel_token.reason or "Cancelled"},
                            ),
                        )
                    return run_state

                run_state.status = "failed"
                run_state.error = str(exc)
                services.run_store.save(run_state)
                if not services.event_broker.has_terminal_event(run_id):
                    services.event_broker.publish(
                        run_id,
                        make_sse_event(
                            "run_failed",
                            run_id,
                            stage="repair",
                            data={"error": str(exc)},
                        ),
                    )
                raise

        job = await services.job_registry.submit(run_id, _repair_job, stage="repair")
        services.run_store.save(run_state)
    return {"run_id": run_id, "job": job.model_dump(mode="json")}


@router.get("/jobs")
async def list_jobs(
    services: ServicesDep,
    run_id: Annotated[str | None, Query()] = None,
) -> dict[str, Any]:
    """Return strictly serializable `JobStatusDTO` list + `other_active_jobs_count` (BUG-02/22)."""
    jobs = services.job_registry.list_jobs(run_id=run_id)
    other_active = services.job_registry.other_active_jobs_count(exclude_run_id=run_id)
    return {
        "jobs": [job.model_dump(mode="json") for job in jobs],
        "other_active_jobs_count": other_active,
    }


@router.get("/media/{media_id:path}")
async def serve_media(
    media_id: str,
    request: Request,
    services: ServicesDep,
) -> Response:
    """Serve media assets with HTTP 206 Partial Content Range support and traversal protection."""
    parts = Path(media_id).parts if media_id else ()
    if (
        not media_id
        or len(media_id) > _MAX_MEDIA_ID_LENGTH
        or "\x00" in media_id
        or "\\" in media_id
        or ".." in parts
        or any(len(part.encode("utf-8")) > _MAX_MEDIA_SEGMENT_LENGTH for part in parts)
    ):
        raise HTTPException(status_code=400, detail="Unsafe media_id")

    try:
        path = services.media_store.lookup(media_id)
        file_size = path.stat().st_size
    except MediaStoreError as exc:
        msg = str(exc).lower()
        if msg.startswith("unsafe media reference:") or msg.startswith("path escapes"):
            raise HTTPException(status_code=400, detail="Unsafe media_id") from exc
        raise HTTPException(status_code=404, detail="Media not found") from exc
    except OSError as exc:
        raise HTTPException(status_code=400, detail="Unsafe media_id") from exc

    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    range_header = request.headers.get("range")

    if not range_header:
        return StreamingResponse(
            _iter_file_range(path, 0, file_size),
            status_code=200,
            media_type=media_type,
            headers={
                "Accept-Ranges": "bytes",
                "Content-Length": str(file_size),
            },
        )

    parsed_range = _parse_byte_range(range_header, file_size)
    if parsed_range is None:
        return Response(
            status_code=416,
            headers={
                "Content-Range": f"bytes */{file_size}",
                "Accept-Ranges": "bytes",
            },
        )

    start, end = parsed_range
    content_length = end - start + 1
    return StreamingResponse(
        _iter_file_range(path, start, content_length),
        status_code=206,
        media_type=media_type,
        headers={
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(content_length),
        },
    )


@router.get("/runs/{run_id}/video")
async def serve_run_video(
    run_id: str,
    request: Request,
    services: ServicesDep,
) -> Response:
    """Stream the stitched MP4 output for a completed run with HTTP 206 Range support."""
    if not run_id or len(run_id) > MAX_RUN_ID_LENGTH:
        raise HTTPException(status_code=400, detail="Invalid run_id")
    try:
        state = services.run_store.load(run_id)
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=404, detail="Invalid run_id") from exc

    if state is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found")

    stitched = state.render_manifest.stitched_path if state.render_manifest else None
    if not stitched:
        raise HTTPException(status_code=404, detail="Run has no rendered video yet")

    path = Path(stitched).resolve()
    try:
        path.relative_to(services.data_dir.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="Video path outside data directory") from exc

    if not path.is_file():
        raise HTTPException(status_code=404, detail="Rendered video file not found on disk")

    file_size = path.stat().st_size
    media_type = mimetypes.guess_type(path.name)[0] or "video/mp4"
    range_header = request.headers.get("range")

    if not range_header:
        return StreamingResponse(
            _iter_file_range(path, 0, file_size),
            status_code=200,
            media_type=media_type,
            headers={
                "Accept-Ranges": "bytes",
                "Content-Length": str(file_size),
                "Content-Disposition": f'inline; filename="{run_id}.mp4"',
            },
        )

    parsed_range = _parse_byte_range(range_header, file_size)
    if parsed_range is None:
        return Response(
            status_code=416,
            headers={
                "Content-Range": f"bytes */{file_size}",
                "Accept-Ranges": "bytes",
            },
        )

    start, end = parsed_range
    content_length = end - start + 1
    return StreamingResponse(
        _iter_file_range(path, start, content_length),
        status_code=206,
        media_type=media_type,
        headers={
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(content_length),
        },
    )
