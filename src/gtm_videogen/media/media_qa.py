"""Parallel deterministic Media QA: ffprobe, blackdetect/freezedetect, ebur128 LUFS, OCR, frames.

Addresses:
- F13 / P10: Bounded parallel QA (concurrency = 4), resolving video_path once per evaluation.
- BUG-24: Verifies frame count >= expected frames and wires shot.on_screen_text into live
  Tesseract OCR verification with binarized frame crops.
- Deterministic score caps: OCR fail -> cap 6; loudness < 7 -> cap 7; sync < 7 -> cap 7.
"""

from __future__ import annotations

import asyncio
import difflib
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

from gtm_videogen.core.config import get_settings
from gtm_videogen.core.job_registry import CancellationToken
from gtm_videogen.media._subprocess import MediaProbe, probe_media, run_subprocess
from gtm_videogen.models.schemas import (
    OCRCheck,
    QAIssue,
    QAReport,
    ShotSheet,
    SyncCheck,
)

_BLACK_START_RE = re.compile(
    r"black_start:([0-9.]+)\s+black_end:([0-9.]+)\s+black_duration:([0-9.]+)"
)
_FREEZE_DUR_RE = re.compile(r"freeze_duration:\s*([0-9.]+)")
_FREEZE_START_RE = re.compile(r"freeze_start:\s*([0-9.]+)")
_FREEZE_END_RE = re.compile(r"freeze_end:\s*([0-9.]+)")
_EBUR_SUMMARY_I_RE = re.compile(r"Integrated loudness:.*?I:\s*(-?[0-9.]+)\s*LUFS", re.DOTALL)
_EBUR_SUMMARY_PEAK_RE = re.compile(r"True peak:.*?Peak:\s*(-?[0-9.]+)\s*dBFS", re.DOTALL)
_SCDET_TIME_RE = re.compile(r"lavfi\.scd\.time:\s*([0-9.]+)")

_ASPECT_RATIOS: dict[str, float] = {
    "16:9": 16.0 / 9.0,
    "9:16": 9.0 / 16.0,
    "1:1": 1.0,
    "4:5": 4.0 / 5.0,
}


def _normalize_text(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9\s%$]+", " ", text.lower())
    return " ".join(cleaned.split())


def _text_similarity(expected: str, observed: str) -> float:
    exp_n = _normalize_text(expected)
    obs_n = _normalize_text(observed)
    if not exp_n:
        return 1.0
    if not obs_n:
        return 0.0
    if exp_n in obs_n:
        return 1.0
    seq_ratio = difflib.SequenceMatcher(None, exp_n, obs_n).ratio()
    exp_tokens = set(exp_n.split())
    obs_tokens = set(obs_n.split())
    token_overlap = len(exp_tokens & obs_tokens) / max(1, len(exp_tokens))
    return max(seq_ratio, token_overlap)


def score_loudness(lufs: float | None, peak_dbtp: float | None) -> float:
    """Compute a 0–10 loudness quality score targeting -14 LUFS and peak <= -1.0 dBTP."""
    if lufs is None or lufs <= -60.0:
        return 2.0
    score = 10.0
    # Outside acceptable broadcast/social export window [-30, -9] LUFS
    if lufs < -30.0 or lufs > -9.0:
        score = min(score, 5.0)
    else:
        diff = abs(lufs - (-14.0))
        if diff > 2.5:
            score = max(6.0, 10.0 - (diff - 2.5) * 0.4)
    if peak_dbtp is not None:
        if peak_dbtp > -0.5:
            score = min(score, 5.5)
        elif peak_dbtp > -1.0:
            score = min(score, 7.5)
    return round(max(0.0, min(10.0, score)), 2)


def apply_score_caps(raw_score: float, score_caps_applied: list[str]) -> float:
    """Apply deterministic QA score caps (OCR fail -> 6, loudness < 7 -> 7, sync < 7 -> 7)."""
    capped = raw_score
    for cap in score_caps_applied:
        if "cap_6" in cap:
            capped = min(capped, 6.0)
        elif "cap_7" in cap:
            capped = min(capped, 7.0)
    return round(max(0.0, min(10.0, capped)), 2)


class MediaQA:
    """Parallel deterministic media QA engine (ffprobe, black/freeze, LUFS, OCR, frames)."""

    def __init__(
        self,
        *,
        ffmpeg_bin: str | None = None,
        ffprobe_bin: str | None = None,
        tesseract_bin: str | None = None,
        ocr_lang: str | None = None,
        max_concurrency: int = 4,
    ) -> None:
        settings = get_settings()
        self.ffmpeg_bin = ffmpeg_bin or settings.ffmpeg_bin
        self.ffprobe_bin = ffprobe_bin or settings.ffprobe_bin
        self.tesseract_bin = tesseract_bin or settings.tesseract_bin
        self.ocr_lang = ocr_lang or settings.ocr_lang
        self.max_concurrency = max(1, max_concurrency)

    def is_tesseract_available(self) -> bool:
        return shutil.which(self.tesseract_bin) is not None

    async def detect_black_and_freeze(
        self,
        video_path: Path,
        *,
        total_duration: float | None = None,
        cancel: CancellationToken | None = None,
    ) -> dict[str, Any]:
        """Run blackdetect (d=0.06, pix_th=0.035) and freezedetect (n=-55dB, d=0.35)."""
        cmd = [
            self.ffmpeg_bin,
            "-xerror",
            "-i",
            str(video_path),
            "-vf",
            "blackdetect=d=0.06:pix_th=0.035,freezedetect=n=-55dB:d=0.35",
            "-f",
            "null",
            "-",
        ]
        _, _, stderr_b = await run_subprocess(cmd, cancel=cancel, timeout_sec=120.0)
        stderr = stderr_b.decode("utf-8", errors="replace")

        black_segments = [
            {
                "start": float(m.group(1)),
                "end": float(m.group(2)),
                "duration": float(m.group(3)),
            }
            for m in _BLACK_START_RE.finditer(stderr)
        ]

        freeze_starts = [float(m.group(1)) for m in _FREEZE_START_RE.finditer(stderr)]
        freeze_ends = [float(m.group(1)) for m in _FREEZE_END_RE.finditer(stderr)]
        freeze_durs = [float(m.group(1)) for m in _FREEZE_DUR_RE.finditer(stderr)]
        freeze_segments: list[dict[str, float]] = []
        for idx, start in enumerate(freeze_starts):
            if idx < len(freeze_durs):
                dur = freeze_durs[idx]
                end = freeze_ends[idx] if idx < len(freeze_ends) else start + dur
            else:
                # FFmpeg 5.x emits freeze_start without duration/end when freeze runs to EOF
                end = (
                    total_duration
                    if (total_duration is not None and total_duration > start)
                    else start + 0.35
                )
                dur = max(0.35, end - start)
            freeze_segments.append(
                {
                    "start": start,
                    "end": end,
                    "duration": round(dur, 4),
                }
            )

        return {
            "black_segments": black_segments,
            "freeze_segments": freeze_segments,
        }

    async def measure_loudness(
        self,
        video_path: Path,
        *,
        cancel: CancellationToken | None = None,
    ) -> dict[str, float | None]:
        """Measure integrated loudness (LUFS) and true peak (dBTP) via FFmpeg ebur128."""
        cmd = [
            self.ffmpeg_bin,
            "-i",
            str(video_path),
            "-vn",
            "-af",
            "ebur128=peak=true:framelog=verbose",
            "-f",
            "null",
            "-",
        ]
        rc, _, stderr_b = await run_subprocess(cmd, cancel=cancel, timeout_sec=120.0, check=False)
        stderr = stderr_b.decode("utf-8", errors="replace")
        if rc != 0 and "Integrated loudness" not in stderr:
            return {"lufs": None, "peak_dbtp": None, "loudness_score": 2.0}

        lufs_m = _EBUR_SUMMARY_I_RE.search(stderr)
        peak_m = _EBUR_SUMMARY_PEAK_RE.search(stderr)
        lufs = float(lufs_m.group(1)) if lufs_m else None
        peak = float(peak_m.group(1)) if peak_m else None
        return {
            "lufs": lufs,
            "peak_dbtp": peak,
            "loudness_score": score_loudness(lufs, peak),
        }

    async def _run_tesseract_on_png(
        self,
        png_bytes: bytes,
        *,
        psm: int = 6,
        cancel: CancellationToken | None = None,
    ) -> str:
        cmd = [
            self.tesseract_bin,
            "stdin",
            "stdout",
            "-l",
            self.ocr_lang,
            "--psm",
            str(psm),
        ]
        rc, stdout_b, _ = await run_subprocess(
            cmd,
            cancel=cancel,
            timeout_sec=30.0,
            stdin_bytes=png_bytes,
            check=False,
        )
        if rc != 0:
            return ""
        return " ".join(stdout_b.decode("utf-8", errors="replace").split())

    async def _ocr_single_cue(
        self,
        *,
        video_path: Path,
        shot_id: str,
        expected_text: str,
        at_sec: float,
        sem: asyncio.Semaphore,
        cancel: CancellationToken | None,
    ) -> OCRCheck:
        async with sem:
            if cancel is not None:
                cancel.raise_if_cancelled()

            # Pass 1: Binarized high-contrast crop (bright text -> black on white)
            vf_binarized = "scale=1200:-1:flags=lanczos,lutyuv=y='if(gt(val,185),0,255)'"
            cmd_bin = [
                self.ffmpeg_bin,
                "-ss",
                f"{max(0.0, at_sec):.3f}",
                "-i",
                str(video_path),
                "-frames:v",
                "1",
                "-vf",
                vf_binarized,
                "-f",
                "image2pipe",
                "-vcodec",
                "png",
                "pipe:1",
            ]
            _, png_bin, _ = await run_subprocess(cmd_bin, cancel=cancel, timeout_sec=30.0)
            obs_bin = await self._run_tesseract_on_png(png_bin, psm=6, cancel=cancel)
            sim_bin = _text_similarity(expected_text, obs_bin)

            best_obs = obs_bin
            best_sim = sim_bin

            # Fallback pass for dark-on-light brand palettes if binarized threshold missed dark text
            if best_sim < 0.75:
                vf_gray = "scale=1200:-1:flags=lanczos,format=gray"
                cmd_gray = [
                    self.ffmpeg_bin,
                    "-ss",
                    f"{max(0.0, at_sec):.3f}",
                    "-i",
                    str(video_path),
                    "-frames:v",
                    "1",
                    "-vf",
                    vf_gray,
                    "-f",
                    "image2pipe",
                    "-vcodec",
                    "png",
                    "pipe:1",
                ]
                _, png_gray, _ = await run_subprocess(cmd_gray, cancel=cancel, timeout_sec=30.0)
                obs_gray = await self._run_tesseract_on_png(png_gray, psm=6, cancel=cancel)
                sim_gray = _text_similarity(expected_text, obs_gray)
                if sim_gray > best_sim:
                    best_obs = obs_gray
                    best_sim = sim_gray

            return OCRCheck(
                shot_id=shot_id,
                expected_text=expected_text,
                observed_text=best_obs,
                passed=best_sim >= 0.72,
                at_sec=round(at_sec, 3),
                similarity=round(best_sim, 3),
            )

    async def check_ocr(
        self,
        video_path: Path,
        sheet: ShotSheet,
        *,
        cancel: CancellationToken | None = None,
    ) -> list[OCRCheck]:
        """Run bounded parallel Tesseract OCR checks on all shots with on_screen_text."""
        if not self.is_tesseract_available():
            return []

        sem = asyncio.Semaphore(self.max_concurrency)
        tasks = []
        cursor = 0.0
        for shot in sorted(sheet.shots, key=lambda s: s.index):
            mid = cursor + (shot.duration_sec * 0.5)
            cursor += shot.duration_sec
            if shot.on_screen_text and shot.on_screen_text.strip():
                tasks.append(
                    self._ocr_single_cue(
                        video_path=video_path,
                        shot_id=shot.id,
                        expected_text=shot.on_screen_text.strip(),
                        at_sec=mid,
                        sem=sem,
                        cancel=cancel,
                    )
                )

        if not tasks:
            return []
        return list(await asyncio.gather(*tasks))

    async def check_sync(
        self,
        video_path: Path,
        sheet: ShotSheet,
        expected_vo: str | None,
        whisper: Any | None,
        *,
        cancel: CancellationToken | None = None,
    ) -> list[SyncCheck]:
        """Verify voiceover sync via Whisper provider when available."""
        if whisper is None:
            return []

        expected_text = (expected_vo or "").strip()
        if not expected_text:
            vo_parts = [s.vo_line.strip() for s in sheet.shots if s.vo_line and s.vo_line.strip()]
            expected_text = " ".join(vo_parts)
        if not expected_text:
            return []

        tmp_dir = Path(tempfile.mkdtemp(prefix="gtm_qa_sync_"))
        try:
            wav_path = tmp_dir / "audio_16k.wav"
            cmd = [
                self.ffmpeg_bin,
                "-y",
                "-i",
                str(video_path),
                "-vn",
                "-ar",
                "16000",
                "-ac",
                "1",
                "-c:a",
                "pcm_s16le",
                str(wav_path),
            ]
            rc, _, _ = await run_subprocess(cmd, cancel=cancel, timeout_sec=60.0, check=False)
            if rc != 0 or not wav_path.exists():
                return [
                    SyncCheck(
                        shot_id="full",
                        expected_vo=expected_text,
                        transcribed_vo="",
                        passed=False,
                        similarity=0.0,
                    )
                ]

            transcribed = await whisper.transcribe(wav_path)
            sim = _text_similarity(expected_text, transcribed or "")
            return [
                SyncCheck(
                    shot_id="full",
                    expected_vo=expected_text,
                    transcribed_vo=transcribed or "",
                    passed=sim >= 0.70,
                    similarity=round(sim, 3),
                )
            ]
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    async def poster_frame(
        self,
        video_path: Path | str,
        output_path: Path | str,
        *,
        at_sec: float = 0.5,
        cancel: CancellationToken | None = None,
    ) -> Path:
        """Extract a high-quality JPEG poster frame from the video."""
        resolved = Path(video_path).resolve()
        out = Path(output_path).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        probe = await probe_media(resolved, ffprobe_bin=self.ffprobe_bin, cancel=cancel)
        ts = min(max(0.0, at_sec), max(0.0, probe.duration_sec - 0.05))

        cmd = [
            self.ffmpeg_bin,
            "-y",
            "-ss",
            f"{ts:.3f}",
            "-i",
            str(resolved),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(out),
        ]
        await run_subprocess(cmd, cancel=cancel, timeout_sec=30.0)
        return out

    @staticmethod
    def _video_duration_sec(probe: MediaProbe) -> float:
        """Return the video stream's duration (falling back to nb_frames/fps or container)."""
        streams = probe.raw.get("streams", []) if isinstance(probe.raw, dict) else []
        v_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
        if v_stream:
            cand = v_stream.get("duration")
            if cand is not None:
                try:
                    val = float(cand)
                    if val > 0:
                        return val
                except (TypeError, ValueError):
                    pass
            if probe.nb_frames and probe.fps > 0:
                return probe.nb_frames / probe.fps
        return probe.duration_sec

    async def evidence_frames(
        self,
        video_path: Path | str,
        output_dir: Path | str,
        *,
        shot_start: float = 0.0,
        shot_end: float | None = None,
        detect_cuts: bool = True,
        cancel: CancellationToken | None = None,
    ) -> list[Path]:
        """Extract evidence frames across [0.04, 0.25, 0.5, 0.75, 0.96] and scene cuts (P10)."""
        resolved = Path(video_path).resolve()
        out_dir = Path(output_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        probe = await probe_media(resolved, ffprobe_bin=self.ffprobe_bin, cancel=cancel)
        vid_dur = self._video_duration_sec(probe)
        frame_margin = max(0.05, 1.5 / max(probe.fps, 1.0))
        max_seek = max(0.0, vid_dur - frame_margin)

        end = vid_dur if shot_end is None else min(shot_end, vid_dur)
        span = max(0.0, end - shot_start)

        timestamps = [shot_start + span * frac for frac in (0.04, 0.25, 0.5, 0.75, 0.96)]

        if detect_cuts:
            sc_cmd = [
                self.ffmpeg_bin,
                "-i",
                str(resolved),
                "-vf",
                "scdet=t=14",
                "-f",
                "null",
                "-",
            ]
            _, _, stderr_b = await run_subprocess(
                sc_cmd, cancel=cancel, timeout_sec=60.0, check=False
            )
            stderr = stderr_b.decode("utf-8", errors="replace")
            for m in _SCDET_TIME_RE.finditer(stderr):
                cut_t = float(m.group(1))
                if shot_start <= cut_t <= end:
                    timestamps.append(max(shot_start, cut_t - 0.07))
                    timestamps.append(min(max_seek, cut_t + 0.07))

        # Clamp all timestamps within safe video seek bounds and deduplicate within 0.05s
        sorted_ts = sorted(max(0.0, min(t, max_seek)) for t in timestamps)
        deduped: list[float] = []
        for ts in sorted_ts:
            if not deduped or abs(ts - deduped[-1]) >= 0.05:
                deduped.append(ts)

        sem = asyncio.Semaphore(self.max_concurrency)

        async def _extract(idx: int, ts: float) -> Path | None:
            async with sem:
                frame_path = out_dir / f"evidence_{idx:03d}_{int(ts * 1000)}ms.png"
                cmd = [
                    self.ffmpeg_bin,
                    "-y",
                    "-ss",
                    f"{ts:.3f}",
                    "-i",
                    str(resolved),
                    "-frames:v",
                    "1",
                    str(frame_path),
                ]
                await run_subprocess(cmd, cancel=cancel, timeout_sec=30.0)
                if frame_path.exists() and frame_path.stat().st_size > 0:
                    return frame_path
                # Fallback to 0.0s if FFmpeg exited 0 without writing a frame past last PTS
                if ts > 0.0:
                    retry_cmd = [
                        self.ffmpeg_bin,
                        "-y",
                        "-ss",
                        "0.000",
                        "-i",
                        str(resolved),
                        "-frames:v",
                        "1",
                        str(frame_path),
                    ]
                    await run_subprocess(retry_cmd, cancel=cancel, timeout_sec=30.0)
                    if frame_path.exists() and frame_path.stat().st_size > 0:
                        return frame_path
                return None

        paths = await asyncio.gather(*(_extract(i, ts) for i, ts in enumerate(deduped)))
        return [p for p in paths if p is not None]

    async def evaluate(
        self,
        video_path: Path | str,
        sheet: ShotSheet,
        expected_vo: str | None = None,
        run_ocr: bool = True,
        whisper: Any | None = None,
        cancel: CancellationToken | None = None,
    ) -> QAReport:
        """Run parallel deterministic QA checks and return a QAReport with score caps."""
        # P10: resolve video_path once
        resolved = Path(video_path).resolve()
        probe: MediaProbe = await probe_media(resolved, ffprobe_bin=self.ffprobe_bin, cancel=cancel)

        blocking_errors: list[QAIssue] = []
        warnings: list[QAIssue] = []
        failed_shot_ids: list[str] = []
        score_caps_applied: list[str] = []

        # 1. ffprobe stream, resolution, aspect, fps, frame-count & duration checks
        if not probe.has_video or probe.width <= 0 or probe.height <= 0:
            blocking_errors.append(
                QAIssue(
                    code="NO_VIDEO_STREAM",
                    severity="blocking",
                    message="Rendered output has no valid video stream.",
                )
            )
        else:
            expected_ratio = _ASPECT_RATIOS.get(sheet.aspect)
            actual_ratio = probe.width / probe.height
            if expected_ratio is not None and abs(actual_ratio - expected_ratio) > 0.02:
                blocking_errors.append(
                    QAIssue(
                        code="ASPECT_MISMATCH",
                        severity="blocking",
                        message=(
                            f"Video aspect {probe.width}x{probe.height} ({actual_ratio:.3f}) "
                            f"does not match target {sheet.aspect} ({expected_ratio:.3f})."
                        ),
                    )
                )

            if abs(probe.fps - float(sheet.fps)) > 0.5:
                blocking_errors.append(
                    QAIssue(
                        code="FPS_MISMATCH",
                        severity="blocking",
                        message=(
                            f"Video fps {probe.fps:.2f} does not match sheet.fps={sheet.fps}."
                        ),
                    )
                )

            dur_diff = abs(probe.duration_sec - sheet.target_duration_sec)
            if dur_diff > 0.25:
                blocking_errors.append(
                    QAIssue(
                        code="DURATION_MISMATCH",
                        severity="blocking",
                        message=(
                            f"Video duration {probe.duration_sec:.2f}s differs from target "
                            f"{sheet.target_duration_sec:.2f}s by {dur_diff:.2f}s."
                        ),
                    )
                )

            expected_frames = max(1, int(sheet.target_duration_sec * sheet.fps * 0.90))
            if probe.nb_frames is not None and probe.nb_frames < expected_frames:
                blocking_errors.append(
                    QAIssue(
                        code="INCOMPLETE_FRAMES",
                        severity="blocking",
                        message=(
                            f"Rendered frame count {probe.nb_frames} is below expected minimum "
                            f"{expected_frames}."
                        ),
                    )
                )

        # 2. Run black/freeze detect, ebur128 loudness, OCR, and optional Whisper sync in parallel
        bf_task = asyncio.create_task(
            self.detect_black_and_freeze(resolved, total_duration=probe.duration_sec, cancel=cancel)
        )
        loud_task = (
            asyncio.create_task(self.measure_loudness(resolved, cancel=cancel))
            if probe.has_audio
            else None
        )
        ocr_task = (
            asyncio.create_task(self.check_ocr(resolved, sheet, cancel=cancel)) if run_ocr else None
        )
        sync_task = (
            asyncio.create_task(
                self.check_sync(resolved, sheet, expected_vo, whisper, cancel=cancel)
            )
            if whisper is not None
            else None
        )

        sibling_tasks = [t for t in (bf_task, loud_task, ocr_task, sync_task) if t is not None]
        try:
            bf_res = await bf_task
            loud_res = (
                await loud_task
                if loud_task is not None
                else {"lufs": None, "peak_dbtp": None, "loudness_score": 2.0}
            )
            ocr_checks: list[OCRCheck] = await ocr_task if ocr_task is not None else []
            sync_checks: list[SyncCheck] = await sync_task if sync_task is not None else []
        except BaseException:
            for t in sibling_tasks:
                if not t.done():
                    t.cancel()
            await asyncio.gather(*sibling_tasks, return_exceptions=True)
            raise

        # Map black/freeze defects to issues & affected shots
        shot_windows: list[tuple[str, float, float]] = []
        cursor = 0.0
        for s in sorted(sheet.shots, key=lambda x: x.index):
            shot_windows.append((s.id, cursor, cursor + s.duration_sec))
            cursor += s.duration_sec

        def _shots_in_range(start: float, end: float) -> list[str]:
            return [sid for sid, ws, we in shot_windows if we > start and ws < end]

        for seg in bf_res["black_segments"]:
            hit_shots = _shots_in_range(seg["start"], seg["end"])
            for sid in hit_shots:
                if sid not in failed_shot_ids:
                    failed_shot_ids.append(sid)
            blocking_errors.append(
                QAIssue(
                    code="BLACK_FRAMES",
                    severity="blocking",
                    message=(
                        f"Detected black segment {seg['start']:.2f}s–{seg['end']:.2f}s "
                        f"({seg['duration']:.2f}s)."
                    ),
                    shot_id=hit_shots[0] if hit_shots else None,
                    details=seg,
                )
            )

        for seg in bf_res["freeze_segments"]:
            hit_shots = _shots_in_range(seg["start"], seg["end"])
            for sid in hit_shots:
                if sid not in failed_shot_ids:
                    failed_shot_ids.append(sid)
            blocking_errors.append(
                QAIssue(
                    code="FROZEN_FRAMES",
                    severity="blocking",
                    message=(
                        f"Detected frozen segment {seg['start']:.2f}s–{seg['end']:.2f}s "
                        f"({seg['duration']:.2f}s)."
                    ),
                    shot_id=hit_shots[0] if hit_shots else None,
                    details=seg,
                )
            )

        # Loudness evaluation + cap 7
        lufs = loud_res["lufs"]
        peak_dbtp = loud_res["peak_dbtp"]
        loudness_sc = float(loud_res["loudness_score"] or 0.0)
        if loudness_sc < 7.0:
            score_caps_applied.append("loudness_cap_7")
            warnings.append(
                QAIssue(
                    code="LOUDNESS_OUT_OF_RANGE",
                    severity="warning",
                    message=(
                        f"Audio loudness score {loudness_sc:.1f}/10 "
                        f"(LUFS={lufs}, peak={peak_dbtp} dBTP)."
                    ),
                )
            )

        # OCR evaluation + cap 6
        for check in ocr_checks:
            if not check.passed:
                if "ocr_fail_cap_6" not in score_caps_applied:
                    score_caps_applied.append("ocr_fail_cap_6")
                if check.shot_id not in failed_shot_ids:
                    failed_shot_ids.append(check.shot_id)
                blocking_errors.append(
                    QAIssue(
                        code="OCR_MISMATCH",
                        severity="blocking",
                        message=(
                            f"Shot {check.shot_id} on-screen text OCR failed: "
                            f"expected={check.expected_text!r}, observed={check.observed_text!r} "
                            f"(similarity={check.similarity:.2f})."
                        ),
                        shot_id=check.shot_id,
                    )
                )

        # Sync evaluation + cap 7
        for sc in sync_checks:
            if not sc.passed:
                if "sync_cap_7" not in score_caps_applied:
                    score_caps_applied.append("sync_cap_7")
                warnings.append(
                    QAIssue(
                        code="SYNC_MISMATCH",
                        severity="warning",
                        message=(
                            f"Voiceover sync similarity {sc.similarity:.2f} below threshold "
                            f"(expected={sc.expected_vo!r}, transcribed={sc.transcribed_vo!r})."
                        ),
                        shot_id=sc.shot_id,
                    )
                )

        return QAReport(
            passed=len(blocking_errors) == 0,
            blocking_errors=blocking_errors,
            warnings=warnings,
            loudness_lufs=lufs,
            peak_dbtp=peak_dbtp,
            ocr_checks=ocr_checks,
            sync_checks=sync_checks,
            ffprobe_summary={
                "duration_sec": probe.duration_sec,
                "width": probe.width,
                "height": probe.height,
                "fps": probe.fps,
                "has_video": probe.has_video,
                "has_audio": probe.has_audio,
                "video_codec": probe.video_codec,
                "audio_codec": probe.audio_codec,
                "nb_frames": probe.nb_frames,
                "black_segments": bf_res["black_segments"],
                "freeze_segments": bf_res["freeze_segments"],
                "loudness_score": loudness_sc,
            },
            failed_shot_ids=failed_shot_ids,
            score_caps_applied=score_caps_applied,
        )
