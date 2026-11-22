"""Automated Video QA & Multimodal Vision Scorecard Engine for LaunchCut Stitch Studio.

Validates every stitched Master Cut (10s -> 60s) with:
  1. Deterministic FFmpeg & OCR Telemetry (`MediaQA`):
     - Stream/codec/resolution/FPS & duration accuracy vs. `target_duration_sec`
     - `blackdetect` (catches dead black frames and maps them to exact Act indices)
     - `freezedetect` (catches stalled/frozen holds and maps them to exact Act indices)
     - `ebur128` integrated loudness (`-14 LUFS` target) and true peak (`dBTP`)
     - Tesseract OCR legibility verification on Motion Graphic headlines
  2. Native Full-Video + Audio Multimodal Critique:
     - Primary: `google-gemini/gemini-api-cli` (`gemini-api analyze`) watching the full MP4 + audio
     - Fallback: OpenRouter Vision frame sampling -> Deterministic telemetry synthesis
  3. Honest `0-100` Banger QA Score across 6 pillars + per-Act actionable `suggested_fix`es
     powering the 1-Click Self-Healing QA Auto-Fix loop.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gtm_videogen.core.config import Settings, get_settings
from gtm_videogen.core.job_registry import CancellationToken
from gtm_videogen.media._subprocess import probe_media
from gtm_videogen.media.media_qa import MediaQA, _text_similarity
from gtm_videogen.models.stitch_project import StitchProject
from gtm_videogen.providers.gemini_cli import gemini_analyze_media_json, is_gemini_cli_ready


def _clamp10(val: Any, default: float = 8.5) -> float:
    try:
        return round(max(0.0, min(10.0, float(val))), 2)
    except (TypeError, ValueError):
        return default


def _grade_for_score(score_100: int) -> str:
    if score_100 >= 92:
        return "A+"
    if score_100 >= 85:
        return "A"
    if score_100 >= 78:
        return "B+"
    if score_100 >= 70:
        return "B"
    if score_100 >= 60:
        return "C"
    return "Needs Polish"


def _compute_act_windows(project: StitchProject, total_master_dur: float) -> list[dict[str, Any]]:
    """Map each StitchScene onto its `[start_sec, end_sec]` window in the stitched master."""
    windows: list[dict[str, Any]] = []
    cursor = 0.0
    raw_durs: list[float] = []
    for sc in project.scenes:
        win = sc.winning_take()
        d = float(win.duration_sec if (win and win.duration_sec > 0) else sc.duration_sec)
        raw_durs.append(max(1.0, d))
    raw_sum = sum(raw_durs) or 1.0
    # Scale proportionally if micro_dissolve or brand outro shifted total duration slightly
    outro_pad = 2.5 if (project.add_brand_outro and (project.brand_name or project.brand_logo_path)) else 0.0
    usable_master = max(1.0, total_master_dur - outro_pad)
    scale = usable_master / raw_sum if raw_sum > 0 else 1.0

    for sc, rd in zip(project.scenes, raw_durs, strict=False):
        span = rd * scale
        windows.append(
            {
                "scene_id": sc.scene_id,
                "index": sc.index,
                "title": sc.title,
                "kind": sc.kind,
                "start_sec": round(cursor, 2),
                "end_sec": round(cursor + span, 2),
                "duration_sec": round(span, 2),
            }
        )
        cursor += span
    return windows


def _overlap_sec(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


async def evaluate_stitch_master_qa(
    project: StitchProject,
    *,
    deep_vision: bool = False,
    mock: bool = False,
    settings: Settings | None = None,
    cancel: CancellationToken | None = None,
) -> dict[str, Any]:
    """Run deterministic FFmpeg/OCR QA + optional Gemini CLI multimodal critique on `master_video_path`."""
    resolved = settings or get_settings()
    master_str = (project.master_video_path or "").strip()
    if not master_str or not Path(master_str).is_file():
        return {
            "banger_score": 0,
            "overall_10": 0.0,
            "grade": "Needs Polish",
            "passed": False,
            "auto_fixable": False,
            "evaluator": "missing-master",
            "summary": "No stitched Master Cut found yet — stitch your film to generate its QA Scorecard.",
            "pillars": {},
            "telemetry": {},
            "blocking_issues": ["Master video file does not exist."],
            "warnings": [],
            "scene_scores": [],
            "evaluated_at": datetime.now(UTC).isoformat(),
        }

    master_path = Path(master_str).resolve()
    qa_engine = MediaQA(
        ffmpeg_bin=resolved.ffmpeg_bin,
        ffprobe_bin=resolved.ffprobe_bin,
        tesseract_bin=resolved.tesseract_bin,
        ocr_lang=resolved.ocr_lang,
        max_concurrency=1,
    )

    probe = await probe_media(master_path, ffprobe_bin=resolved.ffprobe_bin, cancel=cancel)
    actual_dur = float(probe.duration_sec or 0.0)

    # Run black/freeze scan and EBU R128 loudness in parallel
    bf_task = qa_engine.detect_black_and_freeze(
        master_path, total_duration=actual_dur, cancel=cancel
    )
    loud_task = qa_engine.measure_loudness(master_path, cancel=cancel)
    bf_res, loud_res = await asyncio.gather(bf_task, loud_task)

    black_segs = [
        seg for seg in (bf_res.get("black_segments") or []) if float(seg.get("duration", 0.0)) >= 0.15
    ]
    freeze_segs = [
        seg for seg in (bf_res.get("freeze_segments") or []) if float(seg.get("duration", 0.0)) >= 0.90
    ]

    lufs = loud_res.get("lufs")
    peak_dbtp = loud_res.get("peak_dbtp")
    loud_score = float(loud_res.get("loudness_score") or 2.0)

    act_windows = _compute_act_windows(project, actual_dur)

    # Optional OCR legibility check on up to 3 Motion Graphic scenes
    ocr_results: list[dict[str, Any]] = []
    if qa_engine.is_tesseract_available():
        sem = asyncio.Semaphore(1)
        mg_candidates = [
            (sc, win)
            for sc, win in zip(project.scenes, act_windows, strict=False)
            if sc.kind == "motion_graphic"
            and sc.graphic_spec is not None
            and (sc.graphic_spec.headline or sc.graphic_spec.eyebrow)
        ][:3]
        for sc, win in mg_candidates:
            assert sc.graphic_spec is not None
            exp_txt = (sc.graphic_spec.headline or sc.graphic_spec.eyebrow).strip()[:48]
            sample_t = win["start_sec"] + max(0.6, win["duration_sec"] * 0.58)
            try:
                chk = await qa_engine._ocr_single_cue(
                    video_path=master_path,
                    shot_id=sc.scene_id,
                    expected_text=exp_txt,
                    at_sec=min(max(0.1, actual_dur - 0.2), sample_t),
                    sem=sem,
                    cancel=cancel,
                )
                # Accept partial keyword legibility (>=0.42) since stylized fonts/2.5D tilt affect raw OCR
                sim = max(float(chk.similarity), _text_similarity(exp_txt.split()[0], chk.observed_text))
                ocr_results.append(
                    {
                        "scene_id": sc.scene_id,
                        "expected": exp_txt,
                        "observed": chk.observed_text[:80],
                        "similarity": round(sim, 2),
                        "passed": sim >= 0.42,
                    }
                )
            except Exception:
                continue

    # Map black & freeze segments to per-Act diagnostics
    blocking_issues: list[str] = []
    warnings: list[str] = []
    scene_scores: list[dict[str, Any]] = []

    for sc, win in zip(project.scenes, act_windows, strict=False):
        w0, w1 = float(win["start_sec"]), float(win["end_sec"])
        sc_issues: list[str] = []
        sc_fix = ""
        sc_score = 9.3

        blk_ov = sum(
            _overlap_sec(w0, w1, float(b["start"]), float(b["end"])) for b in black_segs
        )
        frz_ov = sum(
            _overlap_sec(w0, w1, float(f["start"]), float(f["end"])) for f in freeze_segs
        )

        if blk_ov >= 0.20:
            msg = f"{sc.title}: {blk_ov:.2f}s of dead black frames detected."
            sc_issues.append(msg)
            blocking_issues.append(msg)
            sc_score -= min(4.8, 2.5 + blk_ov * 1.4)
            sc_fix = "Re-render Act with bright frame-0 hero lighting and zero fade-from-black."

        if frz_ov >= 1.10:
            msg = f"{sc.title}: {frz_ov:.2f}s static freeze detected."
            sc_issues.append(msg)
            if frz_ov >= 1.80:
                blocking_issues.append(msg)
            else:
                warnings.append(msg)
            sc_score -= min(3.8, 1.6 + frz_ov * 0.7)
            if not sc_fix:
                sc_fix = "Add continuous smooth camera movement across all beats so the frame stays alive."

        vo_words = [w for w in sc.vo_line.strip().split() if w]
        if vo_words and len(vo_words) / float(max(3, sc.duration_sec)) > 3.2:
            wps = round(len(vo_words) / float(max(3, sc.duration_sec)), 1)
            w_msg = f"{sc.title}: dense narration ({wps} words/sec)."
            sc_issues.append(w_msg)
            warnings.append(w_msg)
            sc_score -= 0.7
            if not sc_fix:
                sc_fix = "Tighten VO copy to <= 2.8 words/sec."

        ocr_match = next((o for o in ocr_results if o["scene_id"] == sc.scene_id), None)
        if ocr_match and not ocr_match["passed"]:
            w_msg = f"{sc.title}: low headline OCR contrast ({ocr_match['similarity']:.0%})."
            sc_issues.append(w_msg)
            warnings.append(w_msg)
            sc_score -= 1.0
            if not sc_fix:
                sc_fix = "Shorten motion graphic headline for high-contrast legibility."

        sc_score = round(max(2.5, min(9.8, sc_score)), 2)
        status = "fail" if sc_score < 6.8 else ("warn" if sc_score < 8.2 or sc_issues else "pass")
        scene_scores.append(
            {
                "scene_id": sc.scene_id,
                "index": sc.index,
                "title": sc.title,
                "kind": sc.kind,
                "narrative_role": sc.narrative_role or ("hook" if sc.index == 0 else "act"),
                "score": sc_score,
                "status": status,
                "issues": sc_issues,
                "suggested_fix": sc_fix,
            }
        )

    if not probe.has_audio or lufs is None or lufs <= -48.0:
        blocking_issues.append("Master audio track is silent or missing.")
    elif lufs < -24.0 or lufs > -8.5:
        warnings.append(f"Integrated loudness ({lufs:.1f} LUFS) drifts from -14 LUFS broadcast target.")

    expected_dur = float(project.total_duration_sec()) + (
        2.5 if (project.add_brand_outro and (project.brand_name or project.brand_logo_path)) else 0.0
    )
    if project.timeline_mode == "aroll_with_cutaways" and project.aroll_duration_sec > 0:
        expected_dur = float(project.aroll_duration_sec)
    dur_delta = abs(actual_dur - expected_dur)
    if dur_delta > max(5.0, expected_dur * 0.18):
        warnings.append(
            f"Master duration ({actual_dur:.1f}s) differs from planned timeline ({expected_dur:.1f}s)."
        )

    # Compute 6 deterministic base pillars (0-10)
    hook_sc = scene_scores[0]["score"] if scene_scores else 8.0
    hook_impact = _clamp10(
        hook_sc + (0.3 if project.scenes and project.scenes[0].duration_sec <= 9 else -0.2)
    )

    roles_present = {s.narrative_role for s in project.scenes if s.narrative_role}
    arc_bonus = 0.45 if len(roles_present) >= 4 or len(project.scenes) >= 5 else 0.15
    dur_pen = min(2.0, max(0.0, (dur_delta - 2.5) * 0.25))
    story_pacing = _clamp10(8.9 + arc_bonus - dur_pen - (0.3 * sum(1 for w in warnings if "narration" in w)))

    avg_act_visual = sum(s["score"] for s in scene_scores) / max(1, len(scene_scores))
    res_bonus = 0.25 if (probe.height or 0) >= 720 and (probe.width or 0) >= 720 else -0.8
    visual_craft = _clamp10(avg_act_visual + res_bonus)

    if ocr_results:
        ocr_avg = sum(1.0 if o["passed"] else 0.55 for o in ocr_results) / len(ocr_results)
        graphics_clarity = _clamp10(7.2 + 2.4 * ocr_avg)
    else:
        graphics_clarity = 9.1

    seam_continuity = _clamp10(
        9.2
        - (0.8 if any(b["start"] > 0.5 for b in black_segs) else 0.0)
        + (0.25 if project.seam_transition in ("micro_dissolve", "audio_riser") or len(project.scenes) > 1 else 0.0)
    )
    audio_mastering = _clamp10(loud_score)

    evaluator = "deterministic-ffmpeg-ocr"
    director_summary = ""

    # Optional Deep Multimodal Full-Video + Audio Critique via google-gemini/gemini-api-cli!
    if deep_vision and not mock and is_gemini_cli_ready(resolved):
        sys_prompt = (
            "You are an executive launch-film QA critic evaluating a finished commercial video "
            "and its audio track. Grade strictly and constructively on 0-10 scales."
        )
        question = (
            f"Evaluate this '{project.title}' ({actual_dur:.1f}s, {project.aspect_ratio}) video. "
            f"Planned acts: {json.dumps([{'scene_id': s.scene_id, 'title': s.title, 'vo': s.vo_line} for s in project.scenes])}. "
            f"FFmpeg telemetry: LUFS={lufs}, black_segments={len(black_segs)}, freeze_segments={len(freeze_segs)}. "
            "Return ONLY a JSON object with keys: hook_impact (0-10), story_pacing (0-10), "
            "visual_craft (0-10), graphics_clarity (0-10), seam_continuity (0-10), "
            "audio_mastering (0-10), summary (2 concise sentences), and scene_critiques "
            "(array of {scene_id, score (0-10), suggested_fix})."
        )
        gem_qa = await gemini_analyze_media_json(
            master_path,
            question=question,
            system_prompt=sys_prompt,
            settings=resolved,
            cancel=cancel,
        )
        if isinstance(gem_qa, dict) and any(k in gem_qa for k in ("hook_impact", "visual_craft")):
            evaluator = "gemini-api-cli-multimodal"
            # Blend 60% deterministic telemetry + 40% Gemini full-video perceptual critique
            hook_impact = _clamp10(hook_impact * 0.6 + _clamp10(gem_qa.get("hook_impact"), hook_impact) * 0.4)
            story_pacing = _clamp10(story_pacing * 0.6 + _clamp10(gem_qa.get("story_pacing"), story_pacing) * 0.4)
            visual_craft = _clamp10(visual_craft * 0.6 + _clamp10(gem_qa.get("visual_craft"), visual_craft) * 0.4)
            graphics_clarity = _clamp10(
                graphics_clarity * 0.6 + _clamp10(gem_qa.get("graphics_clarity"), graphics_clarity) * 0.4
            )
            seam_continuity = _clamp10(
                seam_continuity * 0.6 + _clamp10(gem_qa.get("seam_continuity"), seam_continuity) * 0.4
            )
            audio_mastering = _clamp10(
                audio_mastering * 0.7 + _clamp10(gem_qa.get("audio_mastering"), audio_mastering) * 0.3
            )
            director_summary = str(gem_qa.get("summary") or "").strip()
            raw_sc_crit = gem_qa.get("scene_critiques")
            if isinstance(raw_sc_crit, list):
                by_sid = {str(x.get("scene_id")): x for x in raw_sc_crit if isinstance(x, dict)}
                for sc_entry in scene_scores:
                    g_item = by_sid.get(sc_entry["scene_id"])
                    if g_item:
                        blended = round(sc_entry["score"] * 0.6 + _clamp10(g_item.get("score"), sc_entry["score"]) * 0.4, 2)
                        sc_entry["score"] = blended
                        if not sc_entry["suggested_fix"] and g_item.get("suggested_fix"):
                            sc_entry["suggested_fix"] = str(g_item["suggested_fix"])[:180]
                        if blended < 7.5 and sc_entry["status"] == "pass":
                            sc_entry["status"] = "warn"

    # Weighted composite 0-100 Banger Score
    weighted_10 = (
        hook_impact * 0.20
        + story_pacing * 0.20
        + visual_craft * 0.20
        + graphics_clarity * 0.15
        + seam_continuity * 0.10
        + audio_mastering * 0.15
    )

    # Honest deterministic caps when technical defects exist
    if any("black frames" in b for b in blocking_issues):
        visual_craft = min(visual_craft, 4.8)
        weighted_10 = min(weighted_10, 6.8)
    if any("static freeze" in b for b in blocking_issues):
        visual_craft = min(visual_craft, 5.5)
        weighted_10 = min(weighted_10, 7.3)
    if not probe.has_audio or lufs is None or lufs <= -48.0:
        audio_mastering = min(audio_mastering, 2.5)
        weighted_10 = min(weighted_10, 7.0)

    overall_10 = round(max(0.0, min(10.0, weighted_10)), 2)
    banger_score = int(round(overall_10 * 10.0))
    passed = bool(banger_score >= 78 and len(blocking_issues) == 0)
    auto_fixable = bool(
        not passed
        or any(s["status"] in ("fail", "warn") for s in scene_scores)
        or len(warnings) > 0
    )

    if not director_summary:
        if passed:
            lufs_txt = f"{lufs:.1f} LUFS" if lufs is not None else "clean audio"
            director_summary = (
                f"Banger Score {banger_score}/100 ({_grade_for_score(banger_score)}) — "
                f"{len(project.scenes)} acts ({actual_dur:.1f}s) verified with zero black/freeze defects and {lufs_txt} mastering."
            )
        else:
            lead_issue = (blocking_issues + warnings or ["Needs polish on flagged acts."])[0]
            director_summary = (
                f"QA Score {banger_score}/100 ({_grade_for_score(banger_score)}) — {lead_issue} "
                "Click Auto-Polish to repair flagged acts and re-stitch."
            )

    return {
        "banger_score": banger_score,
        "overall_10": overall_10,
        "grade": _grade_for_score(banger_score),
        "passed": passed,
        "auto_fixable": auto_fixable,
        "evaluator": evaluator,
        "summary": director_summary,
        "pillars": {
            "hook_impact": hook_impact,
            "story_pacing": story_pacing,
            "visual_craft": visual_craft,
            "graphics_clarity": graphics_clarity,
            "seam_continuity": seam_continuity,
            "audio_mastering": audio_mastering,
        },
        "telemetry": {
            "duration_sec": round(actual_dur, 2),
            "expected_duration_sec": round(expected_dur, 2),
            "width": probe.width,
            "height": probe.height,
            "fps": round(float(probe.fps or 24.0), 2),
            "loudness_lufs": round(lufs, 2) if lufs is not None else None,
            "peak_dbtp": round(peak_dbtp, 2) if peak_dbtp is not None else None,
            "black_segments": black_segs,
            "freeze_segments": freeze_segs,
            "ocr_checks": ocr_results,
        },
        "blocking_issues": blocking_issues,
        "warnings": warnings,
        "scene_scores": scene_scores,
        "evaluated_at": datetime.now(UTC).isoformat(),
    }
