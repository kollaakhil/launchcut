"""Single-pass FFmpeg media stitcher with parallel clip normalization and stream-copy concat.

Addresses:
- F11 / P1: Bounded parallel clip normalization with identical H.264 SPS & bt709 color metadata.
- P2: Stream-copy concat (`-f concat -safe 0 -i list.txt -c:v copy -c:a copy`) + single-pass
  audio mix, optional ASS caption burn, and `-t` trim (retaining `-c:v copy` when ASS is absent).
- BUG-05: Full CancellationToken propagation terminating FFmpeg subprocesses cleanly.
- BUG-09: Threads `sheet.fps` consistently through normalization and final output.
- BUG-10: Empty-audio guard preventing malformed filtergraphs (`undefinedanull`).
- BUG-26: Atomic `shutil.move` and deterministic temp-dir cleanup in `finally`.
"""

from __future__ import annotations

import asyncio
import errno
import os
import shutil
import tempfile
import time
from pathlib import Path

from gtm_videogen.core.config import get_settings
from gtm_videogen.core.job_registry import CancellationToken
from gtm_videogen.media._subprocess import (
    aspect_to_dimensions,
    probe_media,
    run_subprocess,
)
from gtm_videogen.media.captions import (
    build_ass_filter,
    extract_cues_from_sheet,
    generate_ass_subtitles,
)
from gtm_videogen.models.schemas import (
    CreativeBible,
    RenderedAudio,
    RenderedClip,
    RenderManifest,
    ShotSheet,
    SoundPlan,
)

SHORTAGE_TPAD_MIN_SEC = 0.12
SHORTAGE_MAX_SEC = 1.0
MAX_ATEMPO = 1.5


class ClipShortageError(ValueError):
    """Raised when a rendered clip is shorter than its planned duration by > 1.0s."""


def _pinned_h264_flags(fps: int, *, threads: int | None = None) -> list[str]:
    eff_threads = threads
    if eff_threads is None:
        raw_t = os.getenv("GTM_FFMPEG_THREADS", "2").strip()
        eff_threads = max(1, int(raw_t)) if raw_t.isdigit() else 2
    flags = [
        "-threads",
        str(eff_threads),
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "20",
        "-profile:v",
        "high",
        "-level:v",
        "4.1",
        "-pix_fmt",
        "yuv420p",
        "-color_range",
        "tv",
        "-colorspace",
        "bt709",
        "-color_primaries",
        "bt709",
        "-color_trc",
        "bt709",
        "-video_track_timescale",
        "90000",
        "-r",
        str(fps),
    ]
    return flags


def _in_silence_gap(at_sec: float, silence_gaps: list[tuple[float, float]]) -> bool:
    return any(start <= at_sec <= end for start, end in silence_gaps)


def _concat_list_escape(path: Path) -> str:
    return str(path.resolve()).replace("'", "'\\''")


def _move_output_atomically(src: Path, dst: Path) -> None:
    """Move src to dst atomically, handling EXDEV across mounts without calling copystat."""
    try:
        os.replace(src, dst)
        return
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise

    staged: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=dst.parent,
            prefix=f".{dst.name}.",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            staged = Path(tmp.name)
        shutil.copyfile(src, staged)
        os.replace(staged, dst)
        staged = None
        src.unlink(missing_ok=True)
    finally:
        if staged is not None and staged.exists():
            staged.unlink(missing_ok=True)


class FFmpegStitcher:
    """Three-stage single-pass FFmpeg stitcher (Parallel Normalize → Stream-Copy Concat → Mix)."""

    def __init__(
        self,
        *,
        ffmpeg_bin: str | None = None,
        ffprobe_bin: str | None = None,
        fonts_dir: Path | None = None,
        max_workers: int | None = None,
    ) -> None:
        settings = get_settings()
        self.ffmpeg_bin = ffmpeg_bin or settings.ffmpeg_bin
        self.ffprobe_bin = ffprobe_bin or settings.ffprobe_bin
        self.fonts_dir = fonts_dir
        raw_w = os.getenv("GTM_FFMPEG_WORKERS", "").strip()
        env_workers = max(1, int(raw_w)) if raw_w.isdigit() else None
        self.max_workers = max_workers or env_workers or min(2, os.cpu_count() or 2)

    async def _normalize_single_clip(
        self,
        *,
        index: int,
        clip: RenderedClip,
        planned_duration: float,
        width: int,
        height: int,
        fps: int,
        out_path: Path,
        sem: asyncio.Semaphore,
        cancel: CancellationToken | None,
    ) -> tuple[int, Path, list[str]]:
        warnings: list[str] = []
        async with sem:
            if cancel is not None:
                cancel.raise_if_cancelled()

            src_path = Path(clip.path).resolve()
            probe = await probe_media(src_path, ffprobe_bin=self.ffprobe_bin, cancel=cancel)
            if not probe.has_video:
                raise ValueError(f"Clip {clip.shot_id!r} ({src_path}) has no video stream")

            actual_dur = probe.duration_sec
            shortage = planned_duration - actual_dur
            if shortage > SHORTAGE_MAX_SEC:
                raise ClipShortageError(
                    f"Clip {clip.shot_id!r} is too short: actual={actual_dur:.3f}s, "
                    f"planned={planned_duration:.3f}s "
                    f"(shortage={shortage:.3f}s > {SHORTAGE_MAX_SEC}s)"
                )

            vf_parts = [
                f"scale={width}:{height}:force_original_aspect_ratio=increase",
                f"crop={width}:{height}",
                "setsar=1",
            ]
            if shortage > SHORTAGE_TPAD_MIN_SEC:
                vf_parts.append(f"tpad=stop_mode=clone:stop_duration={shortage:.4f}")
                warnings.append(
                    f"Extended short clip {clip.shot_id} by {shortage:.2f}s via tpad clone."
                )
            vf_parts.extend([f"fps={fps}", "format=yuv420p"])

            # AI video generators (e.g. Kling) often ease in over the first 0.35-0.45s.
            # When the raw clip has ample slack above planned_duration, trim the initial ease-in
            # so cuts open directly on live motion without tripping freezedetect.
            lead_in_sec = 0.0
            slack = actual_dur - planned_duration
            if slack >= 0.65:
                lead_in_sec = round(min(0.40, slack * 0.45), 3)

            cmd = [self.ffmpeg_bin, "-y"]
            if lead_in_sec > 0.0:
                cmd.extend(["-ss", f"{lead_in_sec:.3f}"])
            cmd.extend(["-i", str(src_path)])
            if probe.has_audio:
                cmd.extend(
                    [
                        "-map",
                        "0:v:0",
                        "-map",
                        "0:a:0",
                        "-vf",
                        ",".join(vf_parts),
                        *_pinned_h264_flags(fps, threads=2),
                        "-af",
                        f"aresample=44100,aformat=channel_layouts=stereo,apad=whole_dur={planned_duration:.4f}",
                        "-c:a",
                        "aac",
                        "-ar",
                        "44100",
                        "-ac",
                        "2",
                        "-b:a",
                        "160k",
                        "-t",
                        f"{planned_duration:.4f}",
                        str(out_path),
                    ]
                )
            else:
                cmd.extend(
                    [
                        "-f",
                        "lavfi",
                        "-i",
                        "anullsrc=r=44100:cl=stereo",
                        "-map",
                        "0:v:0",
                        "-map",
                        "1:a:0",
                        "-vf",
                        ",".join(vf_parts),
                        *_pinned_h264_flags(fps, threads=2),
                        "-af",
                        f"apad=whole_dur={planned_duration:.4f}",
                        "-c:a",
                        "aac",
                        "-ar",
                        "44100",
                        "-ac",
                        "2",
                        "-b:a",
                        "160k",
                        "-t",
                        f"{planned_duration:.4f}",
                        str(out_path),
                    ]
                )

            await run_subprocess(cmd, cancel=cancel, timeout_sec=120.0)
            return index, out_path, warnings

    async def _prepare_voice_wav(
        self,
        *,
        src_path: Path,
        target_dur: float,
        out_wav: Path,
        cancel: CancellationToken | None,
    ) -> Path:
        """Pre-pass voice clip to 44.1kHz stereo WAV with atempo capped at 1.5x."""
        probe = await probe_media(src_path, ffprobe_bin=self.ffprobe_bin, cancel=cancel)
        af_parts = ["aresample=44100", "aformat=channel_layouts=stereo"]
        if probe.duration_sec > 0 and target_dur > 0 and probe.duration_sec > target_dur + 0.05:
            tempo = min(MAX_ATEMPO, probe.duration_sec / target_dur)
            if tempo > 1.01:
                af_parts.insert(0, f"atempo={tempo:.4f}")

        cmd = [
            self.ffmpeg_bin,
            "-y",
            "-i",
            str(src_path),
            "-vn",
            "-af",
            ",".join(af_parts),
            "-ar",
            "44100",
            "-ac",
            "2",
            "-c:a",
            "pcm_s16le",
            str(out_wav),
        ]
        await run_subprocess(cmd, cancel=cancel, timeout_sec=60.0)
        return out_wav

    async def stitch(
        self,
        clips: list[RenderedClip],
        sound_plan: SoundPlan | None,
        audio_assets: list[RenderedAudio],
        sheet: ShotSheet,
        bible: CreativeBible,
        output_path: Path | str,
        cancel: CancellationToken | None = None,
        *,
        run_id: str = "run",
        version_id: str = "v1",
        width: int | None = None,
        height: int | None = None,
        burn_captions: bool = False,
        ass_path: Path | None = None,
        preview: bool = False,
    ) -> RenderManifest:
        """Normalize clips in parallel, stream-copy concat, and run single-pass mix/ASS/trim."""
        if not clips:
            raise ValueError("FFmpegStitcher.stitch() requires at least one clip")
        if cancel is not None:
            cancel.raise_if_cancelled()

        t_total_start = time.monotonic()
        timings_ms: dict[str, int] = {}
        warnings: list[str] = []

        if width is None or height is None:
            target_w, target_h = aspect_to_dimensions(sheet.aspect, preview=preview)
        else:
            target_w = width if width % 2 == 0 else width + 1
            target_h = height if height % 2 == 0 else height + 1

        fps = int(sheet.fps)
        target_sec = float(sheet.target_duration_sec)
        out_final = Path(output_path).resolve()
        out_final.parent.mkdir(parents=True, exist_ok=True)

        # Map shot_id -> planned duration from sheet (fallback to clip.duration_sec)
        shot_dur_by_id = {s.id: s.duration_sec for s in sheet.shots}
        sorted_shots = sorted(sheet.shots, key=lambda s: s.index)

        tmp_dir = Path(tempfile.mkdtemp(prefix="gtm_stitch_"))
        generated_ass_path: Path | None = ass_path
        try:
            # --- PASS 1: Parallel Clip Normalization (P1) ---
            t_p1 = time.monotonic()
            sem = asyncio.Semaphore(max(1, self.max_workers))
            norm_tasks = []
            for idx, clip in enumerate(clips):
                planned = shot_dur_by_id.get(
                    clip.shot_id,
                    sorted_shots[idx].duration_sec
                    if idx < len(sorted_shots)
                    else clip.duration_sec,
                )
                norm_out = tmp_dir / f"norm_{idx:04d}.mp4"
                norm_tasks.append(
                    self._normalize_single_clip(
                        index=idx,
                        clip=clip,
                        planned_duration=planned,
                        width=target_w,
                        height=target_h,
                        fps=fps,
                        out_path=norm_out,
                        sem=sem,
                        cancel=cancel,
                    )
                )

            norm_results = await asyncio.gather(*norm_tasks)
            norm_results.sort(key=lambda item: item[0])
            norm_paths: list[Path] = []
            for _, norm_path, clip_warnings in norm_results:
                norm_paths.append(norm_path)
                warnings.extend(clip_warnings)
            timings_ms["pass1_normalize_ms"] = int((time.monotonic() - t_p1) * 1000)

            if cancel is not None:
                cancel.raise_if_cancelled()

            # --- PASS 2: Stream-Copy Concat (P2) ---
            t_p2 = time.monotonic()
            concat_list = tmp_dir / "concat_list.txt"
            concat_list.write_text(
                "\n".join(f"file '{_concat_list_escape(p)}'" for p in norm_paths) + "\n",
                encoding="utf-8",
            )
            concat_mp4 = tmp_dir / "concat.mp4"
            concat_cmd = [
                self.ffmpeg_bin,
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_list),
                "-c:v",
                "copy",
                "-c:a",
                "copy",
                str(concat_mp4),
            ]
            await run_subprocess(concat_cmd, cancel=cancel, timeout_sec=60.0)
            timings_ms["pass2_concat_ms"] = int((time.monotonic() - t_p2) * 1000)

            if cancel is not None:
                cancel.raise_if_cancelled()

            # Optional ASS generation if requested
            if generated_ass_path is None and burn_captions:
                cues = extract_cues_from_sheet(sheet, include_vo_captions=True)
                if cues:
                    ass_out = out_final.with_suffix(".ass")
                    generated_ass_path = generate_ass_subtitles(
                        sheet=sheet,
                        bible=bible,
                        output_path=ass_out,
                        width=target_w,
                        height=target_h,
                        cues=cues,
                    )

            # --- PASS 3: Single-Pass Audio Mix + Optional ASS Burn + Trim (P2, BUG-10) ---
            t_p3 = time.monotonic()
            pass3_out = tmp_dir / "final.mp4"

            vo_assets = [a for a in audio_assets if a.kind == "vo" and Path(a.path).exists()]
            sfx_assets = [a for a in audio_assets if a.kind == "sfx" and Path(a.path).exists()]
            music_assets = [a for a in audio_assets if a.kind == "music" and Path(a.path).exists()]

            silence_gaps = sound_plan.silence_gaps if sound_plan else []
            vo_cues = sound_plan.vo_cues if sound_plan else []
            sfx_cues = sound_plan.sfx_cues if sound_plan else []
            music_cue = sound_plan.music_cue if sound_plan else None

            # Prepare voice pre-pass WAVs (atempo capped at 1.5x)
            prepared_vo: list[tuple[Path, float]] = []
            vo_cue_by_id = {c.shot_id: c for c in vo_cues}
            for idx, va in enumerate(vo_assets):
                cue = vo_cue_by_id.get(va.cue_id) or (vo_cues[idx] if idx < len(vo_cues) else None)
                at_sec = cue.at_sec if cue else 0.0
                if at_sec > target_sec - 0.15:
                    continue
                cue_dur = cue.duration_sec if cue else max(0.5, target_sec - at_sec)
                wav_path = tmp_dir / f"vo_{idx:03d}.wav"
                await self._prepare_voice_wav(
                    src_path=Path(va.path).resolve(),
                    target_dur=cue_dur,
                    out_wav=wav_path,
                    cancel=cancel,
                )
                prepared_vo.append((wav_path, at_sec))

            # Filter SFX cues against silence_gaps and end-of-timeline cutoff (BUG-10)
            prepared_sfx: list[tuple[Path, float, float]] = []
            sfx_cue_by_id = {c.shot_id: c for c in sfx_cues if c.shot_id}
            for idx, sa in enumerate(sfx_assets):
                cue = sfx_cue_by_id.get(sa.cue_id) or (
                    sfx_cues[idx] if idx < len(sfx_cues) else None
                )
                at_sec = cue.at_sec if cue else 0.0
                gain_db = cue.gain_db if cue else -6.0
                if at_sec > target_sec - 0.2:
                    warnings.append(
                        f"Skipped SFX cue {sa.cue_id or idx} at {at_sec:.2f}s past timeline end."
                    )
                    continue
                if _in_silence_gap(at_sec, silence_gaps):
                    warnings.append(
                        f"Skipped SFX cue {sa.cue_id or idx} at {at_sec:.2f}s inside silence gap."
                    )
                    continue
                prepared_sfx.append((Path(sa.path).resolve(), at_sec, gain_db))

            music_path = Path(music_assets[0].path).resolve() if music_assets else None
            has_audio_mix = bool(prepared_vo or prepared_sfx or music_path)

            cmd = [self.ffmpeg_bin, "-y", "-i", str(concat_mp4)]
            input_idx = 1

            if not has_audio_mix:
                # BUG-10 GUARD: No external audio cues -> NEVER emit malformed filtergraph!
                cmd.extend(["-map", "0:v:0", "-map", "0:a:0?"])
                if generated_ass_path is not None:
                    cmd.extend(
                        [
                            "-vf",
                            build_ass_filter(generated_ass_path, self.fonts_dir),
                            *_pinned_h264_flags(fps),
                        ]
                    )
                else:
                    cmd.extend(["-c:v", "copy"])
                cmd.extend(
                    [
                        "-c:a",
                        "copy",
                        "-t",
                        f"{target_sec:.4f}",
                        "-movflags",
                        "+faststart",
                        str(pass3_out),
                    ]
                )
            else:
                filter_parts: list[str] = []
                fg_labels: list[str] = []

                for i, (wav_path, at_sec) in enumerate(prepared_vo):
                    cmd.extend(["-i", str(wav_path)])
                    delay_ms = max(0, int(round(at_sec * 1000)))
                    lbl = f"[vo{i}]"
                    filter_parts.append(
                        f"[{input_idx}:a]aresample=44100,aformat=channel_layouts=stereo,"
                        f"adelay={delay_ms}|{delay_ms}{lbl}"
                    )
                    fg_labels.append(lbl)
                    input_idx += 1

                for i, (sfx_path, at_sec, gain_db) in enumerate(prepared_sfx):
                    cmd.extend(["-i", str(sfx_path)])
                    delay_ms = max(0, int(round(at_sec * 1000)))
                    lbl = f"[sfx{i}]"
                    filter_parts.append(
                        f"[{input_idx}:a]aresample=44100,aformat=channel_layouts=stereo,"
                        f"volume={gain_db:.2f}dB,adelay={delay_ms}|{delay_ms}{lbl}"
                    )
                    fg_labels.append(lbl)
                    input_idx += 1

                music_label: str | None = None
                if music_path is not None:
                    cmd.extend(["-stream_loop", "-1", "-i", str(music_path)])
                    music_gain_db = music_cue.gain_db if music_cue else -16.0
                    fade_st = max(0.0, target_sec - 1.0)
                    fade_d = min(1.0, target_sec)
                    music_label = "[mu]"
                    filter_parts.append(
                        f"[{input_idx}:a]aresample=44100,aformat=channel_layouts=stereo,"
                        f"volume={music_gain_db:.2f}dB,"
                        f"afade=t=out:st={fade_st:.3f}:d={fade_d:.3f}{music_label}"
                    )
                    input_idx += 1

                # Combine foreground and music with sidechain ducking when both are present
                if fg_labels and music_label is not None:
                    if len(fg_labels) == 1:
                        filter_parts.append(f"{fg_labels[0]}asplit=2[fgm][fgsc]")
                    else:
                        joined_fg = "".join(fg_labels)
                        filter_parts.append(
                            f"{joined_fg}amix=inputs={len(fg_labels)}:duration=longest:"
                            f"normalize=0,asplit=2[fgm][fgsc]"
                        )
                    filter_parts.append(
                        f"{music_label}[fgsc]sidechaincompress=threshold=0.02:ratio=10:"
                        f"attack=15:release=350:makeup=1[mud]"
                    )
                    filter_parts.append(
                        "[fgm][mud]amix=inputs=2:duration=longest:normalize=0[pre_norm]"
                    )
                elif fg_labels:
                    if len(fg_labels) == 1:
                        filter_parts.append(f"{fg_labels[0]}anull[pre_norm]")
                    else:
                        joined_fg = "".join(fg_labels)
                        filter_parts.append(
                            f"{joined_fg}amix=inputs={len(fg_labels)}:duration=longest:"
                            f"normalize=0[pre_norm]"
                        )
                elif music_label is not None:
                    filter_parts.append(f"{music_label}anull[pre_norm]")

                # Final loudness normalization & limiting
                filter_parts.append(
                    f"[pre_norm]apad=whole_dur={target_sec:.4f},"
                    f"loudnorm=I=-14:LRA=7:TP=-1.5,"
                    f"aresample=44100,aformat=sample_fmts=dbl:channel_layouts=stereo,"
                    f"alimiter=limit=0.79:level=disabled[aout]"
                )

                cmd.extend(["-filter_complex", ";".join(filter_parts)])
                cmd.extend(["-map", "0:v:0", "-map", "[aout]"])

                if generated_ass_path is not None:
                    cmd.extend(
                        [
                            "-vf",
                            build_ass_filter(generated_ass_path, self.fonts_dir),
                            *_pinned_h264_flags(fps),
                        ]
                    )
                else:
                    cmd.extend(["-c:v", "copy"])

                cmd.extend(
                    [
                        "-c:a",
                        "aac",
                        "-ar",
                        "44100",
                        "-ac",
                        "2",
                        "-b:a",
                        "160k",
                        "-t",
                        f"{target_sec:.4f}",
                        "-movflags",
                        "+faststart",
                        str(pass3_out),
                    ]
                )

            await run_subprocess(cmd, cancel=cancel, timeout_sec=180.0)
            _move_output_atomically(pass3_out, out_final)
            timings_ms["pass3_mix_ms"] = int((time.monotonic() - t_p3) * 1000)
            timings_ms["total_ms"] = int((time.monotonic() - t_total_start) * 1000)

            return RenderManifest(
                run_id=run_id,
                version_id=version_id,
                clips=clips,
                audio_assets=audio_assets,
                ass_caption_path=str(generated_ass_path) if generated_ass_path else None,
                stitched_path=str(out_final),
                timings_ms=timings_ms,
                warnings=warnings,
                error=None,
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
