"""User / Founder Video Clip Normalizer (Trim, Aspect Framing, Loudnorm & Last-Frame Export).

Processes user-uploaded `.mp4` / `.mov` / `.webm` founder talking-head videos or screen
recordings into Stitch-native 24fps H.264 takes (`-threads 1`, Render-512MB safe):
  - `cover_crop`: smart center fill to target aspect ratio (ideal for talking-head A-roll)
  - `contain_blur`: blurred ambient background fill (ideal for vertical phone clips in 16:9)
  - `device_bezel`: floating framed screen recording over a clean studio dark backdrop
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from gtm_videogen.core.job_registry import CancellationToken
from gtm_videogen.media._subprocess import probe_media, run_subprocess
from gtm_videogen.media.ffmpeg_stitcher import _move_output_atomically, _pinned_h264_flags
from gtm_videogen.models.stitch_project import UserClipSpec


async def normalize_user_clip_take(
    *,
    clip_spec: UserClipSpec,
    output_path: Path,
    width: int = 1280,
    height: int = 720,
    fps: int = 24,
    ffmpeg_bin: str = "ffmpeg",
    ffprobe_bin: str = "ffprobe",
    cancel: CancellationToken | None = None,
) -> tuple[Path, float]:
    """Trim, frame, loudness-normalize, and encode a user video into a Stitch Take MP4."""
    src = Path(clip_spec.raw_video_path)
    if not src.is_file():
        raise FileNotFoundError(f"Uploaded video clip not found: {src}")

    probe = await probe_media(src, ffprobe_bin=ffprobe_bin, cancel=cancel)
    src_dur = max(0.5, float(probe.duration_sec or clip_spec.source_duration_sec or 5.0))
    t_start = max(0.0, min(float(clip_spec.trim_start_sec or 0.0), max(0.0, src_dur - 0.5)))
    raw_end = float(clip_spec.trim_end_sec) if clip_spec.trim_end_sec is not None else src_dur
    t_end = max(t_start + 0.5, min(raw_end, src_dur))
    clip_dur = max(0.5, t_end - t_start)

    raw_t = os.getenv("GTM_FFMPEG_THREADS", "1").strip()
    eff_threads = max(1, int(raw_t)) if raw_t.isdigit() else 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="user_clip_"))
    try:
        tmp_mp4 = tmp_dir / "normalized_user_clip.mp4"
        keep_audio = bool(clip_spec.keep_original_audio and probe.has_audio)

        if clip_spec.framing == "contain_blur":
            vf = (
                f"[0:v]split=2[bg][fg];"
                f"[bg]scale={width}:{height}:force_original_aspect_ratio=increase,"
                f"crop={width}:{height},boxblur=20:2[bgb];"
                f"[fg]scale={width}:{height}:force_original_aspect_ratio=decrease[fgs];"
                f"[bgb][fgs]overlay=(W-w)/2:(H-h)/2,fps={fps},format=yuv420p[vout]"
            )
            use_complex = True
        elif clip_spec.framing == "device_bezel":
            inner_w = int(width * 0.86)
            inner_h = int(height * 0.82)
            # Even dimensions for H.264
            inner_w -= inner_w % 2
            inner_h -= inner_h % 2
            vf = (
                f"color=c=#11131A:s={width}x{height}:r={fps}[base];"
                f"[0:v]scale={inner_w}:{inner_h}:force_original_aspect_ratio=decrease,"
                f"pad={inner_w}:{inner_h}:(ow-iw)/2:(oh-ih)/2:color=#1C202B[win];"
                f"[base][win]overlay=(W-w)/2:(H-h)/2:shortest=1,fps={fps},format=yuv420p[vout]"
            )
            use_complex = True
        else:
            # cover_crop (smart center fill)
            vf = (
                f"scale={width}:{height}:force_original_aspect_ratio=increase,"
                f"crop={width}:{height},fps={fps},format=yuv420p"
            )
            use_complex = False

        cmd: list[str] = [
            ffmpeg_bin,
            "-y",
            "-threads",
            str(eff_threads),
            "-ss",
            f"{t_start:.3f}",
            "-t",
            f"{clip_dur:.3f}",
            "-i",
            str(src),
        ]
        if not keep_audio:
            cmd.extend(["-f", "lavfi", "-t", f"{clip_dur:.3f}", "-i", "anullsrc=r=48000:cl=stereo"])

        if use_complex:
            cmd.extend(["-filter_complex", vf, "-map", "[vout]"])
        else:
            cmd.extend(["-vf", vf, "-map", "0:v:0"])

        if keep_audio:
            cmd.extend([
                "-map",
                "0:a:0",
                "-af",
                "loudnorm=I=-16:TP=-1.5:LRA=11",
            ])
        else:
            cmd.extend(["-map", "1:a:0"])

        cmd.extend([
            *_pinned_h264_flags(fps, threads=eff_threads),
            "-c:a",
            "aac",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-b:a",
            "160k",
            "-shortest",
            "-movflags",
            "+faststart",
            str(tmp_mp4),
        ])
        await run_subprocess(cmd, timeout_sec=180.0, cancel=cancel, check=True)
        _move_output_atomically(tmp_mp4, output_path)
        out_probe = await probe_media(output_path, ffprobe_bin=ffprobe_bin, cancel=cancel)
        return output_path, float(out_probe.duration_sec or clip_dur)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
