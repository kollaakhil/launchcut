"""ContextVar-scoped CostMeter, preflight cost estimation, and safe spend ledger I/O.

Fixes:
- BUG-12: Every billable call reserves/settles through CostMeter.
- BUG-15: Preflight estimate includes tiered LLM + image + video (with min_seconds floor)
  + voiceover + SFX + music.
- BUG-25: Missing ledger file returns 0.0 cleanly without swallowing real I/O errors.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from uuid import uuid4

from gtm_videogen.models.schemas import CostEntry, CostLedger, ShotSheet, SoundPlan

_CURRENT_METER: ContextVar[CostMeter | None] = ContextVar("gtm_videogen_cost_meter", default=None)


class BudgetExceeded(RuntimeError):
    """Raised before spending when a reservation would exceed cap_usd."""

    def __init__(
        self,
        *,
        label: str,
        cap_usd: float,
        spent_usd: float,
        reserved_usd: float,
        requested_usd: float,
    ) -> None:
        self.label = label
        self.cap_usd = cap_usd
        self.spent_usd = spent_usd
        self.reserved_usd = reserved_usd
        self.requested_usd = requested_usd
        super().__init__(
            f"Budget exceeded for {label!r}: spent=${spent_usd:.4f} + "
            f"reserved=${reserved_usd:.4f} + requested=${requested_usd:.4f} > cap=${cap_usd:.4f}"
        )


@dataclass
class _Reservation:
    reservation_id: str
    label: str
    stage: str
    provider: str
    model: str
    estimated_usd: float


class CostMeter:
    """Synchronous atomic budget tracker safe across concurrent asyncio tasks."""

    def __init__(
        self,
        run_id: str,
        cap_usd: float,
        *,
        ledger_path: Path | None = None,
    ) -> None:
        cap_val = float(cap_usd)
        if not math.isfinite(cap_val) or cap_val < 0:
            raise ValueError(f"cap_usd must be a finite number >= 0 (got {cap_usd!r})")
        self.run_id = run_id
        self.cap_usd = cap_val
        self.ledger_path = ledger_path
        self._spent_usd: float = 0.0
        self._reservations: dict[str, _Reservation] = {}
        self._entries: list[CostEntry] = []

    @property
    def spent_usd(self) -> float:
        return round(self._spent_usd, 6)

    @property
    def actual_usd(self) -> float:
        return self.spent_usd

    @property
    def reserved_usd(self) -> float:
        return round(sum(r.estimated_usd for r in self._reservations.values()), 6)

    @property
    def remaining_usd(self) -> float:
        return round(max(0.0, self.cap_usd - self._spent_usd - self.reserved_usd), 6)

    @property
    def fraction(self) -> float:
        if self.cap_usd <= 0:
            return 1.0 if (self._spent_usd + self.reserved_usd) > 0 else 0.0
        return (self._spent_usd + self.reserved_usd) / self.cap_usd

    @property
    def state(self) -> Literal["ok", "warn", "blocked"]:
        frac = self.fraction
        if frac >= 1.0:
            return "blocked"
        if frac >= 0.8:
            return "warn"
        return "ok"

    def reserve(
        self,
        label: str,
        estimate_usd: float,
        *,
        stage: str = "render",
        provider: str = "unknown",
        model: str = "unknown",
    ) -> str:
        """Synchronously reserve estimated spend before launching a paid call."""
        raw_est = float(estimate_usd)
        if not math.isfinite(raw_est):
            raise ValueError(f"estimate_usd must be finite (got {estimate_usd!r})")
        est = max(0.0, raw_est)
        committed = self._spent_usd + sum(r.estimated_usd for r in self._reservations.values())
        if committed + est > self.cap_usd + 1e-9:
            raise BudgetExceeded(
                label=label,
                cap_usd=self.cap_usd,
                spent_usd=self._spent_usd,
                reserved_usd=self.reserved_usd,
                requested_usd=est,
            )

        res_id = f"res_{uuid4().hex[:12]}"
        self._reservations[res_id] = _Reservation(
            reservation_id=res_id,
            label=label,
            stage=stage,
            provider=provider,
            model=model,
            estimated_usd=est,
        )
        return res_id

    def settle(
        self,
        reservation_id: str,
        actual_usd: float,
        *,
        cache_hit: bool = False,
    ) -> CostEntry:
        """Settle a prior reservation with actual billed cost and record ledger entry."""
        raw_actual = float(actual_usd)
        if not math.isfinite(raw_actual):
            raise ValueError(f"actual_usd must be finite (got {actual_usd!r})")

        res = self._reservations.pop(reservation_id, None)
        if res is None:
            raise KeyError(f"Unknown or already settled reservation: {reservation_id}")

        actual = 0.0 if cache_hit else max(0.0, raw_actual)
        self._spent_usd += actual
        entry = CostEntry(
            stage=res.stage,
            provider=res.provider,
            model=res.model,
            what=res.label,
            estimated_usd=res.estimated_usd,
            actual_usd=actual,
            cache_hit=cache_hit,
        )
        self._entries.append(entry)
        if self.ledger_path is not None:
            append_ledger_entry(self.ledger_path, entry)
        return entry

    def release(self, reservation_id: str) -> None:
        """Release an unused reservation (e.g. on provider error or cancellation)."""
        self._reservations.pop(reservation_id, None)

    def to_ledger(self, *, estimated_usd: float | None = None) -> CostLedger:
        total_est = (
            float(estimated_usd)
            if estimated_usd is not None
            else round(sum(e.estimated_usd for e in self._entries) + self.reserved_usd, 6)
        )
        return CostLedger(
            run_id=self.run_id,
            cap_usd=self.cap_usd,
            estimated_usd=total_est,
            actual_usd=self.spent_usd,
            entries=list(self._entries),
        )


def current_cost_meter() -> CostMeter | None:
    """Return the active ContextVar-scoped CostMeter, if any."""
    return _CURRENT_METER.get()


@contextmanager
def use_cost_meter(meter: CostMeter) -> Iterator[CostMeter]:
    """Bind a CostMeter into the current asyncio ContextVar scope."""
    token = _CURRENT_METER.set(meter)
    try:
        yield meter
    finally:
        _CURRENT_METER.reset(token)


def read_ledger_total(ledger_path: Path) -> float:
    """Read total actual_usd from a JSONL spend ledger; returns 0.0 if file missing (BUG-25)."""
    if not ledger_path.exists():
        return 0.0
    total = 0.0
    with ledger_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            total += float(record.get("actual_usd", 0.0))
    return round(total, 6)


def append_ledger_entry(ledger_path: Path, entry: CostEntry) -> None:
    """Append a CostEntry as a JSONL line, creating parent directories if needed."""
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a", encoding="utf-8") as fh:
        fh.write(entry.model_dump_json() + "\n")


def estimate_run_cost(
    *,
    shot_sheet: ShotSheet | None = None,
    sound_plan: SoundPlan | None = None,
    fast_llm_calls: int = 4,
    flagship_llm_calls: int = 3,
    image_count: int = 0,
    video_usd_per_sec: float = 0.07,
    video_min_seconds: float = 5.0,
    image_unit_usd: float = 0.03,
    fast_llm_usd: float = 0.02,
    flagship_llm_usd: float = 0.10,
    voice_usd_per_1k_chars: float = 0.10,
    sfx_unit_usd: float = 0.02,
    music_unit_usd: float = 0.20,
) -> dict[str, float]:
    """Preflight USD cost estimate including tiered LLM, image, video (with min_seconds),
    voiceover, SFX, and music (fixes BUG-15).
    """
    llm_usd = fast_llm_calls * fast_llm_usd + flagship_llm_calls * flagship_llm_usd
    img_usd = max(0, image_count) * image_unit_usd

    video_usd = 0.0
    if shot_sheet is not None:
        for shot in shot_sheet.shots:
            if shot.engine in ("ai_video", "image_to_video"):
                billed_sec = max(float(shot.duration_sec), float(video_min_seconds))
                video_usd += billed_sec * video_usd_per_sec

    voice_usd = 0.0
    sfx_usd = 0.0
    music_usd = 0.0
    if sound_plan is not None:
        total_chars = sum(len(cue.text.strip()) for cue in sound_plan.vo_cues if cue.text.strip())
        if total_chars > 0:
            voice_usd = max(0.01, (total_chars / 1000.0) * voice_usd_per_1k_chars)
        sfx_usd = len(sound_plan.sfx_cues) * sfx_unit_usd
        if sound_plan.music_cue is not None:
            music_usd = music_unit_usd
    elif shot_sheet is not None:
        # Fallback audio estimate directly from ShotSheet when SoundPlan not compiled yet
        if shot_sheet.audio_mode in ("vo", "both"):
            vo_chars = sum(
                len((shot.vo_line or "").strip())
                for shot in shot_sheet.shots
                if (shot.vo_line or "").strip()
            )
            if vo_chars > 0:
                voice_usd = max(0.01, (vo_chars / 1000.0) * voice_usd_per_1k_chars)
        if shot_sheet.audio_mode in ("foley", "both"):
            sfx_count = sum(1 for shot in shot_sheet.shots if (shot.sfx_note or "").strip())
            sfx_usd = sfx_count * sfx_unit_usd
            music_usd = music_unit_usd

    total_usd = llm_usd + img_usd + video_usd + voice_usd + sfx_usd + music_usd
    return {
        "llm_usd": round(llm_usd, 4),
        "image_usd": round(img_usd, 4),
        "video_usd": round(video_usd, 4),
        "voice_usd": round(voice_usd, 4),
        "sfx_usd": round(sfx_usd, 4),
        "music_usd": round(music_usd, 4),
        "total_usd": round(total_usd, 4),
    }
