"""Atomic JSON persistence for RunState with corrupt-file quarantine and orphan healing.

Fixes BUG-19: Automatically heals stale 'working' runs whose run_id is absent from
the active JobRegistry on load.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from gtm_videogen.core.job_registry import JobRegistry
from gtm_videogen.models.schemas import RunState


class CorruptRunError(ValueError):
    """Raised when a persisted run JSON file is corrupt and has been quarantined."""

    def __init__(self, run_id: str, quarantine_path: Path, cause: Exception) -> None:
        self.run_id = run_id
        self.quarantine_path = quarantine_path
        self.cause = cause
        super().__init__(
            f"Run {run_id!r} JSON was corrupt and quarantined to {quarantine_path}: {cause}"
        )


class RunStore:
    """Atomic filesystem JSON store for RunState objects."""

    def __init__(
        self,
        runs_dir: Path,
        *,
        job_registry: JobRegistry | None = None,
    ) -> None:
        self.runs_dir = Path(runs_dir).resolve()
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self._job_registry = job_registry

    def _run_path(self, run_id: str) -> Path:
        if not run_id or "/" in run_id or "\\" in run_id or ".." in run_id:
            raise ValueError(f"Unsafe run_id: {run_id!r}")
        return self.runs_dir / f"{run_id}.json"

    def save(self, run_state: RunState) -> Path:
        """Persist RunState atomically via temp file + fsync + os.replace."""
        run_state.updated_at = datetime.now(UTC)
        dest = self._run_path(run_state.run_id)
        dest.parent.mkdir(parents=True, exist_ok=True)

        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=dest.parent,
                prefix=f".tmp_{run_state.run_id}_",
                suffix=".json",
                delete=False,
            ) as tmp:
                tmp_path = Path(tmp.name)
                tmp.write(run_state.model_dump_json(indent=2))
                tmp.flush()
                os.fsync(tmp.fileno())
            os.replace(tmp_path, dest)
            tmp_path = None
        finally:
            if tmp_path is not None and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

        return dest

    def load(self, run_id: str, *, recover_corrupt: bool = True) -> RunState | None:
        """Load RunState, quarantining corrupt JSON and healing orphaned 'working' runs."""
        path = self._run_path(run_id)
        if not path.is_file():
            return None

        try:
            raw = path.read_text(encoding="utf-8")
            state = RunState.model_validate_json(raw)
        except (
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            ValidationError,
            ValueError,
            TypeError,
        ) as exc:
            quarantined = self._quarantine(path)
            if recover_corrupt:
                return None
            raise CorruptRunError(run_id, quarantined, exc) from exc

        if self._job_registry is not None and self._job_registry.heal_orphaned_run(state):
            self.save(state)
        elif self._job_registry is None and state.status == "working":
            # No registry attached: cannot assume an active in-memory job exists on cold load
            state.status = "interrupted"
            msg = "Recovered orphaned 'working' run on load."
            if msg not in state.warnings:
                state.warnings.append(msg)
            self.save(state)

        return state

    def list_runs(self, *, limit: int = 50) -> list[RunState]:
        """List persisted runs sorted by updated_at descending, skipping quarantined files."""
        runs: list[RunState] = []
        for candidate in sorted(self.runs_dir.glob("*.json")):
            if candidate.name.startswith(".") or ".corrupt." in candidate.name:
                continue
            run_id = candidate.stem
            loaded = self.load(run_id, recover_corrupt=True)
            if loaded is not None:
                runs.append(loaded)

        runs.sort(key=lambda r: r.updated_at, reverse=True)
        return runs[:limit]

    def delete(self, run_id: str) -> bool:
        path = self._run_path(run_id)
        if not path.is_file():
            return False
        path.unlink()
        return True

    def _quarantine(self, path: Path) -> Path:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        quarantine_path = path.with_name(f"{path.stem}.corrupt.{stamp}.json")
        if path.exists():
            os.replace(path, quarantine_path)
        return quarantine_path
