"""Unified JobRegistry with set-based slot tracking and 4-layer cancellation.

Fixes:
- BUG-01: Set-based active job tracking with asyncio.Semaphore + guaranteed try/finally release.
- BUG-02: Strictly serializable JobStatusDTO projection (never leaks Task/Event/Process).
- BUG-05: 4-layer cancellation (Event + callbacks + task/process tracking + immutable cancelled).
- BUG-19: Orphaned 'working' run healing when run_id is absent from active registry.
- BUG-22: Global other_active_jobs_count reporting alongside run-scoped jobs.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from gtm_videogen.models.schemas import JobStatusDTO, RunState

JobState = Literal["queued", "running", "done", "cancelled", "failed"]


def _utc_now() -> datetime:
    return datetime.now(UTC)


class CancellationToken:
    """4-layer cooperative and active cancellation token."""

    def __init__(self) -> None:
        self._event = asyncio.Event()
        self._reason: str | None = None
        self._callbacks: list[Callable[[], Any]] = []
        self._tasks: set[asyncio.Task[Any]] = set()
        self._processes: set[asyncio.subprocess.Process] = set()

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> str | None:
        return self._reason

    def raise_if_cancelled(self) -> None:
        if self._event.is_set():
            raise asyncio.CancelledError(self._reason or "Operation cancelled")

    async def wait(self) -> None:
        await self._event.wait()

    def register_callback(self, callback: Callable[[], Any]) -> None:
        """Register a hook invoked immediately upon cancellation."""
        if self._event.is_set():
            with contextlib.suppress(Exception):
                callback()
            return
        self._callbacks.append(callback)

    def track_task(self, task: asyncio.Task[Any]) -> None:
        """Track an asyncio.Task so cancel() cancels it immediately."""
        if self._event.is_set():
            task.cancel()
            return
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def track_process(self, proc: asyncio.subprocess.Process) -> None:
        """Track a subprocess so cancel() terminates it immediately."""
        if self._event.is_set():
            self._terminate_process(proc)
            return
        self._processes.add(proc)

    def untrack_process(self, proc: asyncio.subprocess.Process) -> None:
        self._processes.discard(proc)

    def cancel(self, reason: str = "Cancelled by user") -> None:
        """Trigger cancellation across Event, callbacks, tracked tasks, and subprocesses."""
        if self._event.is_set():
            return
        self._reason = reason
        self._event.set()

        for cb in list(self._callbacks):
            with contextlib.suppress(Exception):
                cb()

        for task in list(self._tasks):
            if not task.done():
                task.cancel()

        for proc in list(self._processes):
            self._terminate_process(proc)

    @staticmethod
    def _terminate_process(proc: asyncio.subprocess.Process) -> None:
        if proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                proc.terminate()


@dataclass
class _JobRecord:
    """Internal mutable job handle (NEVER serialized directly — fixes BUG-02)."""

    id: str
    run_id: str
    stage: str
    version_id: str | None = None
    shot_id: str | None = None
    priority: int = 0
    state: JobState = "queued"
    enqueued_at: datetime = field(default_factory=_utc_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None
    cancel_token: CancellationToken = field(default_factory=CancellationToken)
    task: asyncio.Task[Any] | None = None

    def to_dto(self, *, position: int | None = None) -> JobStatusDTO:
        duration_ms: int | None = None
        if self.started_at is not None:
            end_ts = self.finished_at or _utc_now()
            duration_ms = max(0, int((end_ts - self.started_at).total_seconds() * 1000))

        wait_ms: int | None = None
        if self.started_at is not None:
            wait_ms = max(0, int((self.started_at - self.enqueued_at).total_seconds() * 1000))

        return JobStatusDTO(
            id=self.id,
            run_id=self.run_id,
            version_id=self.version_id,
            shot_id=self.shot_id,
            stage=self.stage,
            state=self.state,
            priority=self.priority,
            enqueued_at=self.enqueued_at,
            started_at=self.started_at,
            finished_at=self.finished_at,
            error=self.error,
            position=position,
            duration_ms=duration_ms,
            wait_ms=wait_ms,
        )


class JobRegistry:
    """Unified concurrency-bounded job tracker with set-based slot release."""

    def __init__(self, max_concurrent: int = 4) -> None:
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be >= 1")
        self._max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._lock = asyncio.Lock()
        self._jobs: dict[str, _JobRecord] = {}
        self._active_ids: set[str] = set()
        self._queued_order: list[str] = []

    @property
    def max_concurrent(self) -> int:
        return self._max_concurrent

    @property
    def active_count(self) -> int:
        return len(self._active_ids)

    def is_run_active(self, run_id: str) -> bool:
        """Return True if any queued or running job belongs to run_id."""
        for rec in self._jobs.values():
            if rec.run_id == run_id and rec.state in ("queued", "running"):
                return True
        return False

    def other_active_jobs_count(
        self,
        *,
        exclude_run_id: str | None = None,
        exclude_job_id: str | None = None,
    ) -> int:
        """Count active (queued or running) jobs excluding a given run_id or job_id (BUG-22)."""
        count = 0
        for rec in self._jobs.values():
            if rec.state not in ("queued", "running"):
                continue
            if exclude_run_id is not None and rec.run_id == exclude_run_id:
                continue
            if exclude_job_id is not None and rec.id == exclude_job_id:
                continue
            count += 1
        return count

    async def submit(
        self,
        run_id: str,
        coro_factory: Callable[[CancellationToken], Awaitable[Any]],
        *,
        stage: str = "pipeline",
        version_id: str | None = None,
        shot_id: str | None = None,
        priority: int = 0,
        job_id: str | None = None,
    ) -> JobStatusDTO:
        """Enqueue and run an async job with bounded concurrency and guaranteed slot release."""
        jid = job_id or f"job_{uuid4().hex[:12]}"
        token = CancellationToken()
        record = _JobRecord(
            id=jid,
            run_id=run_id,
            stage=stage,
            version_id=version_id,
            shot_id=shot_id,
            priority=priority,
            cancel_token=token,
        )

        async with self._lock:
            if jid in self._jobs:
                raise ValueError(f"Duplicate job_id: {jid!r}")
            self._jobs[jid] = record
            self._queued_order.append(jid)

        task = asyncio.create_task(self._execute(record, coro_factory), name=f"gtm-job-{jid}")
        record.task = task
        token.track_task(task)
        return self._dto_for(record)

    async def _execute(
        self,
        record: _JobRecord,
        coro_factory: Callable[[CancellationToken], Awaitable[Any]],
    ) -> Any:
        acquired = False
        try:
            if record.cancel_token.is_cancelled:
                self._mark_terminal(record, "cancelled", error=record.cancel_token.reason)
                return None

            await self._semaphore.acquire()
            acquired = True

            async with self._lock:
                if record.id in self._queued_order:
                    self._queued_order.remove(record.id)

                if record.cancel_token.is_cancelled or record.state == "cancelled":
                    self._mark_terminal(record, "cancelled", error=record.cancel_token.reason)
                    return None

                self._active_ids.add(record.id)
                record.state = "running"
                record.started_at = _utc_now()

            result = await coro_factory(record.cancel_token)

            async with self._lock:
                if record.cancel_token.is_cancelled or record.state == "cancelled":
                    self._mark_terminal(record, "cancelled", error=record.cancel_token.reason)
                else:
                    self._mark_terminal(record, "done")
            return result

        except asyncio.CancelledError:
            async with self._lock:
                self._mark_terminal(
                    record,
                    "cancelled",
                    error=record.cancel_token.reason or "Cancelled",
                )
            return None
        except Exception as exc:
            async with self._lock:
                if record.cancel_token.is_cancelled or record.state == "cancelled":
                    self._mark_terminal(record, "cancelled", error=record.cancel_token.reason)
                else:
                    self._mark_terminal(record, "failed", error=str(exc))
            return None
        finally:
            async with self._lock:
                if record.id in self._queued_order:
                    self._queued_order.remove(record.id)
                if record.id in self._active_ids:
                    self._active_ids.discard(record.id)
            if acquired:
                self._semaphore.release()

    @staticmethod
    def _mark_terminal(record: _JobRecord, state: JobState, *, error: str | None = None) -> None:
        """Transition job to terminal state; NEVER overwrite cancelled with done (BUG-05)."""
        if record.state == "cancelled" and state == "done":
            return
        record.state = state
        if error is not None:
            record.error = error
        if record.finished_at is None:
            record.finished_at = _utc_now()

    async def cancel_job(self, job_id: str, reason: str = "Cancelled by user") -> bool:
        """Cancel a single job by ID."""
        async with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                return False
            if record.state in ("done", "failed", "cancelled"):
                return record.state == "cancelled"
            record.state = "cancelled"
            record.error = reason
            if record.finished_at is None:
                record.finished_at = _utc_now()
            if record.id in self._queued_order:
                self._queued_order.remove(record.id)

        record.cancel_token.cancel(reason=reason)
        return True

    async def cancel_run(self, run_id: str, reason: str = "Run cancelled by user") -> int:
        """Cancel all queued/running jobs belonging to run_id."""
        matching: list[_JobRecord] = []
        async with self._lock:
            for record in self._jobs.values():
                if record.run_id == run_id and record.state in ("queued", "running"):
                    record.state = "cancelled"
                    record.error = reason
                    if record.finished_at is None:
                        record.finished_at = _utc_now()
                    if record.id in self._queued_order:
                        self._queued_order.remove(record.id)
                    matching.append(record)

        for record in matching:
            record.cancel_token.cancel(reason=reason)
        return len(matching)

    async def wait_for_job(self, job_id: str) -> JobStatusDTO | None:
        """Await completion of a submitted job and return its final DTO."""
        record = self._jobs.get(job_id)
        if record is None:
            return None
        if record.task is not None:
            await asyncio.gather(record.task, return_exceptions=True)
        return self._dto_for(record)

    def get_job(self, job_id: str) -> JobStatusDTO | None:
        record = self._jobs.get(job_id)
        if record is None:
            return None
        return self._dto_for(record)

    def get_token(self, job_id: str) -> CancellationToken | None:
        record = self._jobs.get(job_id)
        return record.cancel_token if record is not None else None

    def list_jobs(self, *, run_id: str | None = None) -> list[JobStatusDTO]:
        """Return strictly JSON-serializable JobStatusDTO projections (BUG-02)."""
        records = list(self._jobs.values())
        if run_id is not None:
            records = [r for r in records if r.run_id == run_id]
        return [self._dto_for(r) for r in records]

    def heal_orphaned_run(self, run_state: RunState) -> bool:
        """Heal a stale 'working' RunState when no active job exists in memory (BUG-19)."""
        if run_state.status != "working":
            return False
        if self.is_run_active(run_state.run_id):
            return False
        run_state.status = "interrupted"
        msg = "Recovered orphaned 'working' run after process restart (no active job in registry)."
        if msg not in run_state.warnings:
            run_state.warnings.append(msg)
        run_state.updated_at = _utc_now()
        return True

    def _dto_for(self, record: _JobRecord) -> JobStatusDTO:
        pos: int | None = None
        if record.state == "queued" and record.id in self._queued_order:
            pos = self._queued_order.index(record.id)
        return record.to_dto(position=pos)
