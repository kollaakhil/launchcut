"""Multimodal Vision Quality Critic for rendered GTM launch videos.

Samples real frames from each shot in the stitched MP4 and evaluates visual craft,
brand fidelity, shot-to-shot continuity, audio-video sync, and hook impact using
`GTM_VISION_MODEL` (Gemini 3.8 Flash on OpenRouter) — returning a structured
`VideoQualityScorecard` with actionable per-shot `suggested_prompt_fix` instructions
for the Studio's targeted Reiterate loop.
"""

from __future__ import annotations

import base64
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from gtm_videogen.core.config import Settings, get_settings
from gtm_videogen.core.http_client import ResilientHttpClient
from gtm_videogen.core.job_registry import CancellationToken
from gtm_videogen.media._subprocess import run_subprocess
from gtm_videogen.models.schemas import (
    CreativeBible,
    QAReport,
    ShotScore,
    ShotSheet,
    VideoQualityScorecard,
)


def _clamp10(val: Any, default: float = 7.5) -> float:
    try:
        return round(max(0.0, min(10.0, float(val))), 2)
    except (TypeError, ValueError):
        return default


def _deterministic_scorecard(
    sheet: ShotSheet,
    qa: QAReport | None,
    *,
    reason: str = "deterministic-telemetry",
) -> VideoQualityScorecard:
    """Compute a grounded 0-10 scorecard from deterministic Media QA when offline/fallback."""
    orig = float(qa.originality_score) if qa and qa.originality_score > 0 else 8.2
    coh = float(qa.bible_coherence_score) if qa and qa.bible_coherence_score > 0 else 8.4
    failed = set(qa.failed_shot_ids) if qa else set()

    # Audio sync score from Whisper sync_checks + LUFS score
    loud_score = (
        float(qa.ffprobe_summary.get("loudness_score", 9.0)) if qa and qa.ffprobe_summary else 9.0
    )
    if qa and qa.sync_checks:
        avg_sim = sum(c.similarity for c in qa.sync_checks) / len(qa.sync_checks)
        audio_sync = round(min(10.0, max(4.0, avg_sim * 10.0 * 0.65 + loud_score * 0.35)), 2)
    else:
        audio_sync = round(min(10.0, loud_score), 2)

    shot_scores: list[ShotScore] = []
    for s in sheet.shots:
        is_bad = s.id in failed
        sc = 6.2 if is_bad else round(min(9.5, (orig + coh) / 2.0), 2)
        shot_scores.append(
            ShotScore(
                shot_id=s.id,
                score=sc,
                verdict="reiterate" if is_bad else "keep",
                critique=(
                    "Technical QA flagged black/freeze frame on this shot."
                    if is_bad
                    else "Clean technical framing and pacing."
                ),
                suggested_prompt_fix=(
                    f"{s.visual_prompt.strip()}; add continuous smooth camera motion from frame 1."
                    if is_bad
                    else ""
                ),
            )
        )

    visual_craft = round(sum(x.score for x in shot_scores) / max(1, len(shot_scores)), 2)
    brand_fidelity = round(min(10.0, coh), 2)
    continuity = round(min(10.0, max(5.5, coh - (0.8 if failed else 0.0))), 2)
    hook_impact = round(min(10.0, orig), 2)
    overall = round(
        (visual_craft + brand_fidelity + continuity + audio_sync + hook_impact) / 5.0, 2
    )
    return VideoQualityScorecard(
        overall_score=overall,
        visual_craft=visual_craft,
        brand_fidelity=brand_fidelity,
        continuity=continuity,
        audio_sync=audio_sync,
        hook_impact=hook_impact,
        summary=(
            f"Overall {overall}/10 ({len(failed)} shot(s) flagged for reiteration; "
            f"loudness={qa.loudness_lufs if qa else 'n/a'} LUFS)."
        ),
        shot_scores=shot_scores,
        model=reason,
    )


class VisionQualityCritic:
    """Multimodal video quality critic that watches sampled shot frames from stitched MP4s."""

    def __init__(self, *, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    async def _sample_shot_frames(
        self,
        video_path: Path,
        sheet: ShotSheet,
        *,
        cancel: CancellationToken | None = None,
    ) -> list[tuple[str, str]]:
        """Extract 1 midpoint JPEG frame per shot (640px wide) as base64 data URLs."""
        tmp_dir = Path(tempfile.mkdtemp(prefix="gtm_critic_"))
        frames: list[tuple[str, str]] = []
        try:
            cursor = 0.0
            for shot in sheet.shots:
                mid = max(0.05, cursor + shot.duration_sec * 0.5)
                out_jpg = tmp_dir / f"{shot.id}.jpg"
                cmd = [
                    self.settings.ffmpeg_bin,
                    "-y",
                    "-ss",
                    f"{mid:.2f}",
                    "-i",
                    str(video_path),
                    "-frames:v",
                    "1",
                    "-q:v",
                    "4",
                    "-vf",
                    "scale=640:-2",
                    str(out_jpg),
                ]
                rc, _, _ = await run_subprocess(cmd, cancel=cancel, timeout_sec=30.0, check=False)
                if rc == 0 and out_jpg.is_file():
                    b64 = base64.b64encode(out_jpg.read_bytes()).decode("ascii")
                    frames.append((shot.id, f"data:image/jpeg;base64,{b64}"))
                cursor += shot.duration_sec
            return frames
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    async def score_video(
        self,
        video_path: Path | str,
        sheet: ShotSheet,
        bible: CreativeBible,
        qa: QAReport | None = None,
        *,
        mock: bool = False,
        cancel: CancellationToken | None = None,
    ) -> VideoQualityScorecard:
        """Evaluate the rendered video across 5 quality dimensions + per-shot critiques."""
        resolved = Path(video_path).resolve()
        if mock or not resolved.is_file() or not self.settings.openrouter_api_key.strip():
            return _deterministic_scorecard(sheet, qa)

        frames = await self._sample_shot_frames(resolved, sheet, cancel=cancel)
        if not frames:
            return _deterministic_scorecard(sheet, qa, reason="frame-extract-fallback")

        vision_model = self.settings.vision_model or "google/gemini-3.8-flash"
        sync_summary = (
            [
                {
                    "expected": c.expected_vo,
                    "transcribed": c.transcribed_vo,
                    "similarity": c.similarity,
                }
                for c in qa.sync_checks
            ]
            if qa and qa.sync_checks
            else []
        )
        qa_brief = {
            "loudness_lufs": qa.loudness_lufs if qa else None,
            "peak_dbtp": qa.peak_dbtp if qa else None,
            "failed_shot_ids": qa.failed_shot_ids if qa else [],
            "blocking_errors": [e.message for e in (qa.blocking_errors if qa else [])][:5],
            "sync_checks": sync_summary,
        }
        shots_brief = [
            {
                "id": s.id,
                "beat": s.beat,
                "duration_sec": s.duration_sec,
                "engine": s.engine,
                "visual_prompt": s.visual_prompt,
                "vo_line": s.vo_line,
                "on_screen_text": s.on_screen_text,
            }
            for s in sheet.shots
        ]

        content_parts: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "You are a senior commercial film director and QA critic grading a GTM launch "
                    "video against Anthropic/OpenAI/Apple launch-film standards.\n"
                    f"CreativeBible Thesis: {bible.concept_thesis}\n"
                    f"Visual World: {bible.visual_language.world}\n"
                    f"Lighting: {bible.visual_language.lighting_style}\n"
                    f"ShotSheet: {json.dumps(shots_brief)}\n"
                    f"Technical QA: {json.dumps(qa_brief)}\n\n"
                    "Inspect each shot frame attached below in order. "
                    "Return ONLY valid JSON with:\n"
                    "{\n"
                    '  "overall_score": 0-10,\n'
                    '  "visual_craft": 0-10,\n'
                    '  "brand_fidelity": 0-10,\n'
                    '  "continuity": 0-10,\n'
                    '  "audio_sync": 0-10,\n'
                    '  "hook_impact": 0-10,\n'
                    '  "summary": "concise 2-sentence director critique",\n'
                    '  "shot_scores": [\n'
                    '    {"shot_id": "s1", "score": 0-10, "verdict": "keep"|"reiterate", '
                    '"critique": "specific visual observation", '
                    '"suggested_prompt_fix": "concrete prompt fix if score < 8.0"}\n'
                    "  ]\n"
                    "}"
                ),
            }
        ]
        for sid, data_url in frames:
            content_parts.append({"type": "text", "text": f"Frame for shot {sid}:"})
            content_parts.append({"type": "image_url", "image_url": {"url": data_url}})

        headers = {
            "Authorization": f"Bearer {self.settings.openrouter_api_key.strip()}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": vision_model,
            "temperature": 0.2,
            "max_tokens": 1200,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": content_parts}],
        }

        try:
            async with ResilientHttpClient(
                connect_timeout=15.0,
                read_timeout=60.0,
                write_timeout=30.0,
                pool_timeout=15.0,
                max_retries=1,
            ) as client:
                resp = await client.post_json(
                    "https://openrouter.ai/api/v1/chat/completions",
                    json=payload,
                    headers=headers,
                    cancel=cancel,
                )
            raw_text = resp["choices"][0]["message"]["content"]
            parsed = json.loads(raw_text)
            raw_shots = parsed.get("shot_scores") or []
            shot_scores = [
                ShotScore(
                    shot_id=str(item.get("shot_id") or ""),
                    score=_clamp10(item.get("score"), 7.5),
                    verdict=(
                        "reiterate"
                        if str(item.get("verdict", "")).lower() == "reiterate"
                        or _clamp10(item.get("score"), 7.5) < 7.8
                        else "keep"
                    ),
                    critique=str(item.get("critique") or ""),
                    suggested_prompt_fix=str(item.get("suggested_prompt_fix") or ""),
                )
                for item in raw_shots
                if isinstance(item, dict) and item.get("shot_id")
            ]
            return VideoQualityScorecard(
                overall_score=_clamp10(parsed.get("overall_score"), 7.5),
                visual_craft=_clamp10(parsed.get("visual_craft"), 7.5),
                brand_fidelity=_clamp10(parsed.get("brand_fidelity"), 7.5),
                continuity=_clamp10(parsed.get("continuity"), 7.5),
                audio_sync=_clamp10(parsed.get("audio_sync"), 7.5),
                hook_impact=_clamp10(parsed.get("hook_impact"), 7.5),
                summary=str(parsed.get("summary") or ""),
                shot_scores=shot_scores,
                model=vision_model,
            )
        except Exception:
            return _deterministic_scorecard(sheet, qa, reason="vision-critic-fallback")


__all__ = ["VisionQualityCritic", "_deterministic_scorecard"]
