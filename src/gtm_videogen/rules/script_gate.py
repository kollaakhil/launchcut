"""Pure script rules evaluator (ZERO hard templates).

Enforces ONLY pure script rules & production validity:
- Factual & claim grounding, banned phrases, superlatives, placeholders (non-overridable)
- Speakability syllable budget per shot
- Exact timing math & [MIN_SHOT, MAX_SHOT] bounds
- Schema completeness (visual_prompt, camera_move, audio spec)
- product_hero requires has_product_photos (BUG-18, non-overridable)
- Anti-garbled-text guardrail on AI video prompts
- Craft rule b2: one idea per shot (warning)
- Distinguishes non-overridable blocking errors from overridable warnings (BUG-23)
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence

from gtm_videogen.models.schemas import QAIssue, QAReport, ResearchBundle, Script, ShotSheet
from gtm_videogen.rules.anti_garbled_text import rendered_text_problem
from gtm_videogen.rules.claim_grounding import check_claim_grounding
from gtm_videogen.rules.speakability import beat_fit_problem
from gtm_videogen.rules.timing import MAX_SHOT_SEC, MIN_SHOT_SEC

_TIMING_TOLERANCE_SEC = 0.25
_SENTENCE_BREAK_RE = re.compile(r"[.!?]+(?:\s+|$)")


def evaluate_script_gate(
    sheet: ShotSheet,
    research: ResearchBundle,
    *,
    script: Script | None = None,
    has_product_photos: bool | None = None,
    extra_facts: Sequence[str] = (),
    offer: str | None = None,
    must_say: Sequence[str] = (),
    gate_override: bool = False,
    min_shot: float = MIN_SHOT_SEC,
    max_shot: float = MAX_SHOT_SEC,
) -> QAReport:
    """Evaluate a ShotSheet (and optional Script) strictly against pure script rules."""
    blocking: list[QAIssue] = []
    warnings: list[QAIssue] = []
    failed_shots: list[str] = []

    photos_available = (
        bool(research.product_photos) if has_product_photos is None else bool(has_product_photos)
    )

    def _add_blocking(issue: QAIssue) -> None:
        if gate_override and issue.overridable:
            warnings.append(
                issue.model_copy(
                    update={
                        "severity": "warning",
                        "message": f"[OVERRIDDEN] {issue.message}",
                    }
                )
            )
        else:
            blocking.append(issue)
            if issue.shot_id and issue.shot_id not in failed_shots:
                failed_shots.append(issue.shot_id)

    # 1. Exact timing math (non-overridable)
    durations = [s.duration_sec for s in sheet.shots]
    if any(not math.isfinite(d) or d <= 0 for d in durations):
        _add_blocking(
            QAIssue(
                code="INVALID_SHOT_DURATION",
                severity="blocking",
                message="All shot durations must be finite positive numbers.",
                overridable=False,
            )
        )
    else:
        total_sec = sum(durations)
        if abs(total_sec - sheet.target_duration_sec) > _TIMING_TOLERANCE_SEC:
            _add_blocking(
                QAIssue(
                    code="TIMING_SUM_MISMATCH",
                    severity="blocking",
                    message=(
                        f"Shot durations sum to {total_sec:.2f}s, which differs from "
                        f"target_duration_sec={sheet.target_duration_sec:.2f}s by more than "
                        f"{_TIMING_TOLERANCE_SEC:.2f}s."
                    ),
                    overridable=False,
                    details={"sum_sec": total_sec, "target_sec": sheet.target_duration_sec},
                )
            )

        # Enforce [min_shot, max_shot] when feasible for the number of shots
        n_shots = len(sheet.shots)
        min_feasible = n_shots * min_shot <= sheet.target_duration_sec + 1e-6
        max_feasible = n_shots * max_shot >= sheet.target_duration_sec - 1e-6
        for shot in sheet.shots:
            if min_feasible and shot.duration_sec < min_shot - 1e-6:
                _add_blocking(
                    QAIssue(
                        code="SHOT_TOO_SHORT",
                        severity="blocking",
                        message=(
                            f"Shot {shot.id} duration {shot.duration_sec:.2f}s is below "
                            f"minimum {min_shot:.2f}s."
                        ),
                        shot_id=shot.id,
                        overridable=False,
                    )
                )
            if max_feasible and shot.duration_sec > max_shot + 1e-6:
                _add_blocking(
                    QAIssue(
                        code="SHOT_TOO_LONG",
                        severity="blocking",
                        message=(
                            f"Shot {shot.id} duration {shot.duration_sec:.2f}s exceeds "
                            f"maximum {max_shot:.2f}s."
                        ),
                        shot_id=shot.id,
                        overridable=False,
                    )
                )

    # 2. Per-shot checks
    all_copy_parts: list[str] = []
    has_any_vo = False
    has_any_sfx = False

    for shot in sheet.shots:
        # 2a. Schema completeness
        if not shot.visual_prompt.strip():
            _add_blocking(
                QAIssue(
                    code="MISSING_VISUAL_PROMPT",
                    severity="blocking",
                    message=f"Shot {shot.id} is missing a concrete visual_prompt.",
                    shot_id=shot.id,
                    overridable=False,
                )
            )

        if shot.engine in ("ai_video", "image_to_video") and not shot.camera_move.strip():
            _add_blocking(
                QAIssue(
                    code="MISSING_CAMERA_SPEC",
                    severity="blocking",
                    message=f"Shot {shot.id} is missing camera_move specification.",
                    shot_id=shot.id,
                    overridable=True,
                )
            )

        # 2b. product_hero requires has_product_photos (BUG-18, non-overridable)
        if shot.product_hero and not photos_available:
            _add_blocking(
                QAIssue(
                    code="PRODUCT_HERO_WITHOUT_PHOTOS",
                    severity="blocking",
                    message=(
                        f"Shot {shot.id} is marked product_hero=True, but no product photos "
                        "are available in ResearchBundle."
                    ),
                    shot_id=shot.id,
                    overridable=False,
                )
            )

        # 2c. Anti-garbled-text guardrail on AI video engines
        if shot.engine in ("ai_video", "image_to_video"):
            text_prob = rendered_text_problem(shot.visual_prompt)
            if text_prob:
                _add_blocking(
                    QAIssue(
                        code="GARBLED_TEXT_RISK",
                        severity="blocking",
                        message=text_prob,
                        shot_id=shot.id,
                        overridable=True,
                    )
                )

        # 2d. VO speakability + claim grounding
        if shot.vo_line and shot.vo_line.strip():
            has_any_vo = True
            all_copy_parts.append(shot.vo_line)
            fit_err = beat_fit_problem(shot.vo_line, shot.duration_sec)
            if fit_err:
                _add_blocking(
                    QAIssue(
                        code="SPEAKABILITY_OVERFLOW",
                        severity="blocking",
                        message=fit_err,
                        shot_id=shot.id,
                        overridable=True,
                    )
                )
            for claim_issue in check_claim_grounding(
                shot.vo_line,
                research,
                extra_facts=extra_facts,
                offer=offer,
                shot_id=shot.id,
            ):
                _add_blocking(claim_issue)

            # b2: one idea per shot warning if VO packs >3 sentences into a single shot
            vo_sentences = [s for s in _SENTENCE_BREAK_RE.split(shot.vo_line.strip()) if s.strip()]
            if len(vo_sentences) > 3:
                warnings.append(
                    QAIssue(
                        code="B2_CROWDED_SHOT_VO",
                        severity="warning",
                        message=(
                            f"Shot {shot.id} packs {len(vo_sentences)} sentences into one shot "
                            "(craft rule b2: keep one clear idea per shot)."
                        ),
                        shot_id=shot.id,
                        overridable=True,
                    )
                )

        # 2e. On-screen text claim grounding + b2 readability warning
        if shot.on_screen_text and shot.on_screen_text.strip():
            all_copy_parts.append(shot.on_screen_text)
            for claim_issue in check_claim_grounding(
                shot.on_screen_text,
                research,
                extra_facts=extra_facts,
                offer=offer,
                shot_id=shot.id,
            ):
                _add_blocking(claim_issue)

            words = shot.on_screen_text.split()
            if len(words) > 10:
                warnings.append(
                    QAIssue(
                        code="B2_CROWDED_ON_SCREEN_TEXT",
                        severity="warning",
                        message=(
                            f"Shot {shot.id} on_screen_text has {len(words)} words (>10); "
                            "keep one scannable idea per shot."
                        ),
                        shot_id=shot.id,
                        overridable=True,
                    )
                )

        if shot.sfx_note and shot.sfx_note.strip():
            has_any_sfx = True

    # 3. Audio mode consistency
    if sheet.audio_mode in ("vo", "both") and not has_any_vo:
        _add_blocking(
            QAIssue(
                code="MISSING_VO_LINES",
                severity="blocking",
                message=f"audio_mode={sheet.audio_mode!r} requires at least one shot with vo_line.",
                overridable=True,
            )
        )
    if sheet.audio_mode == "foley" and not has_any_sfx:
        warnings.append(
            QAIssue(
                code="NO_SFX_NOTES_IN_FOLEY_MODE",
                severity="warning",
                message="audio_mode='foley' has no sfx_note cues specified on shots.",
                overridable=True,
            )
        )

    # 4. End card & full script claim grounding (non-overridable)
    for field_name, value in (
        ("end_card_title", sheet.end_card_title),
        ("end_card_subtitle", sheet.end_card_subtitle),
        ("end_card_footer", sheet.end_card_footer),
    ):
        if value and value.strip():
            all_copy_parts.append(value)
            for claim_issue in check_claim_grounding(
                value,
                research,
                extra_facts=extra_facts,
                offer=offer,
                shot_id=field_name,
            ):
                _add_blocking(claim_issue)

    if script is not None and script.full_vo_text.strip():
        all_copy_parts.append(script.full_vo_text)
        for claim_issue in check_claim_grounding(
            script.full_vo_text,
            research,
            extra_facts=extra_facts,
            offer=offer,
        ):
            # Deduplicate identical claim issues already flagged on individual shots
            if not any(
                b.code == claim_issue.code and b.details == claim_issue.details for b in blocking
            ):
                _add_blocking(claim_issue)

    # 5. Verbatim must_say terms (non-overridable)
    combined_copy_lower = " ".join(all_copy_parts).lower()
    for required_phrase in must_say:
        cleaned = required_phrase.strip()
        if cleaned and cleaned.lower() not in combined_copy_lower:
            _add_blocking(
                QAIssue(
                    code="MISSING_MUST_SAY",
                    severity="blocking",
                    message=f"Required verbatim must_say phrase missing from script: {cleaned!r}",
                    overridable=False,
                    details={"must_say": cleaned},
                )
            )

    return QAReport(
        passed=len(blocking) == 0,
        blocking_errors=blocking,
        warnings=warnings,
        failed_shot_ids=failed_shots,
    )
