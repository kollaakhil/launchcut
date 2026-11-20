"""Exact shot timing redistribution with [MIN_SHOT, MAX_SHOT] bounds and NaN guards.

Fixes BUG-21:
- Guards len(shots) * min_shot > target_sec via even scaling (never divides by zero / NaN).
- Guards len(shots) * max_shot < target_sec via even scaling.
- Enforces BOTH min_shot and max_shot iteratively during redistribution.
- Sanitizes zero, negative, NaN, or infinite weights.
- Guarantees finite positive floats summing to target_sec within floating-point epsilon.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from gtm_videogen.models.schemas import Shot

MIN_SHOT_SEC = 1.5
MAX_SHOT_SEC = 8.0


def _extract_weight(item: float | int | Shot) -> float:
    raw = item.duration_sec if isinstance(item, Shot) else float(item)
    if not math.isfinite(raw) or raw <= 0.0:
        return 1.0
    return raw


def fit_timings(
    shots: Sequence[float | int | Shot],
    target_sec: float,
    *,
    min_shot: float = MIN_SHOT_SEC,
    max_shot: float = MAX_SHOT_SEC,
) -> list[float]:
    """Fit shot durations to sum to target_sec while enforcing [min_shot, max_shot]."""
    n = len(shots)
    if n == 0:
        return []

    if not math.isfinite(target_sec) or target_sec <= 0.0:
        raise ValueError(f"target_sec must be a finite positive number (got {target_sec!r})")
    if not math.isfinite(min_shot) or min_shot <= 0.0:
        raise ValueError(f"min_shot must be a finite positive number (got {min_shot!r})")
    if not math.isfinite(max_shot) or max_shot < min_shot:
        raise ValueError(f"max_shot must be >= min_shot (got {max_shot!r} < {min_shot!r})")

    # Feasibility guards: if bounds cannot sum to target_sec, scale evenly without NaN
    if n * min_shot >= target_sec:
        even = target_sec / n
        return [even] * n

    if n * max_shot <= target_sec:
        even = target_sec / n
        return [even] * n

    weights = [_extract_weight(s) for s in shots]

    # Exact bounded water-filling via bisection on scalar lam > 0:
    # f(lam) = sum(min(max_shot, max(min_shot, lam * w)) for w in weights)
    # is continuous and monotonically non-decreasing with
    # f(0) = n * min_shot < target_sec < n * max_shot = f(hi).
    lo = 0.0
    hi = max_shot / min(weights)
    if not math.isfinite(hi):
        hi = 1.0
        while sum(min(max_shot, max(min_shot, hi * w)) for w in weights) < target_sec:
            hi *= 2.0

    for _ in range(100):
        mid = (lo + hi) * 0.5
        total = sum(min(max_shot, max(min_shot, mid * w)) for w in weights)
        if total < target_sec:
            lo = mid
        else:
            hi = mid

    lam = (lo + hi) * 0.5
    result = [min(max_shot, max(min_shot, lam * w)) for w in weights]

    # Distribute any tiny floating-point residual strictly within [min_shot, max_shot]
    residual = target_sec - sum(result)
    if abs(residual) > 1e-12:
        for i in range(n):
            if residual > 0.0:
                slack = max_shot - result[i]
                delta = min(residual, slack)
            else:
                slack = result[i] - min_shot
                delta = -min(-residual, slack)
            result[i] += delta
            residual -= delta
            if abs(residual) <= 1e-12:
                break

    return result
