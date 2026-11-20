"""Structural orthogonality and diversity scoring across ConceptPitches (Pillar 3)."""

from __future__ import annotations

import re
from collections.abc import Sequence

from gtm_videogen.models.schemas import ConceptPitch

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "of",
        "in",
        "to",
        "with",
        "for",
        "on",
        "at",
        "by",
        "from",
        "as",
        "is",
        "are",
    }
)


def _tokens(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS and len(t) > 1}


def _axis_distance(a: str, b: str) -> float:
    na = a.strip().lower()
    nb = b.strip().lower()
    if na == nb:
        return 0.0
    ta = _tokens(na)
    tb = _tokens(nb)
    if not ta and not tb:
        return 0.0
    union = ta | tb
    if not union:
        return 0.0
    return 1.0 - (len(ta & tb) / len(union))


def pairwise_distance(a: ConceptPitch, b: ConceptPitch) -> float:
    """Compute structural distance [0.0, 1.0] across 4 orthogonal creative axes."""
    axes = (
        _axis_distance(a.narrative_arc, b.narrative_arc),
        _axis_distance(a.visual_world, b.visual_world),
        _axis_distance(a.sonic_strategy, b.sonic_strategy),
        _axis_distance(a.emotional_register, b.emotional_register),
    )
    return sum(axes) / len(axes)


def diversity_scores(concepts: Sequence[ConceptPitch]) -> list[float]:
    """Compute mean structural orthogonality score [0.0, 1.0] for each concept."""
    n = len(concepts)
    if n == 0:
        return []
    if n == 1:
        return [1.0]

    scores: list[float] = []
    for i, c_i in enumerate(concepts):
        dists = [pairwise_distance(c_i, c_j) for j, c_j in enumerate(concepts) if i != j]
        scores.append(round(sum(dists) / len(dists), 4))
    return scores


def select_diverse(concepts: Sequence[ConceptPitch], k: int) -> list[ConceptPitch]:
    """Select k maximally orthogonal concepts using greedy farthest-point sampling."""
    if k <= 0 or not concepts:
        return []

    base_scores = diversity_scores(concepts)
    if k >= len(concepts):
        return [
            c.model_copy(update={"diversity_score": score})
            for c, score in zip(concepts, base_scores, strict=True)
        ]

    # Start with the concept that has the highest overall diversity score
    first_idx = max(range(len(concepts)), key=lambda i: base_scores[i])
    selected_indices = [first_idx]

    while len(selected_indices) < k:
        best_idx = -1
        best_min_dist = -1.0
        for idx in range(len(concepts)):
            if idx in selected_indices:
                continue
            min_dist = min(
                pairwise_distance(concepts[idx], concepts[s_idx]) for s_idx in selected_indices
            )
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_idx = idx
        if best_idx == -1:
            break
        selected_indices.append(best_idx)

    chosen = [concepts[i] for i in selected_indices]
    chosen_scores = diversity_scores(chosen)
    return [
        c.model_copy(update={"diversity_score": score})
        for c, score in zip(chosen, chosen_scores, strict=True)
    ]
