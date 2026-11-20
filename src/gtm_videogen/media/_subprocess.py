"""Shared async subprocess, FFprobe, and resolution helpers for the media engine."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gtm_videogen.core.job_registry import CancellationToken

ASPECT_DIMENSIONS_FINAL: dict[str, tuple[int, int]] = {
    "16:9": (1920, 1080),
    "9:16": (1080, 1920),
    "1:1": (1080, 1080),
    "4:5": (1080, 1350),
}

ASPECT_DIMENSIONS_PREVIEW: dict[str, tuple[int, int]] = {
    "16:9": (854, 480),
    "9:16": (480, 854),
    "1:1": (480, 480),
    "4:5": (480, 600),
}


class MediaProcessError(RuntimeError):
    """Raised when an FFmpeg/FFprobe/Tesseract subprocess exits non-zero."""

    def __init__(self, cmd: list[str], returncode: int, stderr: str) -> None:
        self.cmd = cmd
        self.returncode = returncode
        self.stderr = stderr
        excerpt = stderr.strip()[-800:] if stderr else ""
        super().__init__(f"Command {cmd[0]!r} failed (exit {returncode}): {excerpt}")


@dataclass(frozen=True)
class MediaProbe:
    """Structured metadata extracted via ffprobe."""

    path: Path
    duration_sec: float
    width: int
    height: int
    fps: float
    has_video: bool
    has_audio: bool
    video_codec: str | None
    audio_codec: str | None
    pix_fmt: str | None
    nb_frames: int | None
    raw: dict[str, Any]


def aspect_to_dimensions(aspect: str, *, preview: bool = False) -> tuple[int, int]:
    """Resolve an aspect ratio string to even (width, height) pixel dimensions."""
    table = ASPECT_DIMENSIONS_PREVIEW if preview else ASPECT_DIMENSIONS_FINAL
    if aspect in table:
        return table[aspect]
    raise ValueError(f"Unsupported aspect ratio: {aspect!r}. Expected one of {sorted(table)}")


def _kill_process_group(proc: asyncio.subprocess.Process, sig: int) -> None:
    if proc.returncode is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError, OSError):
        with contextlib.suppress(ProcessLookupError, OSError):
            if sig == signal.SIGKILL:
                proc.kill()
            else:
                proc.terminate()


async def _terminate_and_reap(proc: asyncio.subprocess.Process) -> None:
    """Gracefully terminate, then SIGKILL process group if needed, and await exit."""
    if proc.returncode is not None:
        return
    _kill_process_group(proc, signal.SIGTERM)
    try:
        await asyncio.wait_for(proc.wait(), timeout=2.0)
    except TimeoutError:
        _kill_process_group(proc, signal.SIGKILL)
        with contextlib.suppress(Exception):
            await asyncio.wait_for(proc.wait(), timeout=2.0)


async def run_subprocess(
    cmd: list[str],
    *,
    cancel: CancellationToken | None = None,
    timeout_sec: float | None = None,
    cwd: Path | str | None = None,
    env: dict[str, str] | None = None,
    stdin_bytes: bytes | None = None,
    check: bool = True,
) -> tuple[int, bytes, bytes]:
    """Execute a subprocess with CancellationToken and process-group timeout enforcement."""
    if cancel is not None:
        cancel.raise_if_cancelled()

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE if stdin_bytes is not None else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        start_new_session=True,
    )

    if cancel is not None:
        cancel.track_process(proc)

    comm_task = asyncio.create_task(proc.communicate(input=stdin_bytes))
    cancel_task: asyncio.Task[None] | None = None
    if cancel is not None:
        cancel_task = asyncio.create_task(cancel.wait())

    try:
        wait_tasks: set[asyncio.Task[Any]] = {comm_task}
        if cancel_task is not None:
            wait_tasks.add(cancel_task)

        done, pending = await asyncio.wait(
            wait_tasks,
            timeout=timeout_sec,
            return_when=asyncio.FIRST_COMPLETED,
        )

        if not done:
            # Timeout reached
            await _terminate_and_reap(proc)
            comm_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await comm_task
            raise TimeoutError(f"Subprocess {cmd[0]!r} timed out after {timeout_sec}s")

        if cancel_task is not None and cancel_task in done:
            await _terminate_and_reap(proc)
            comm_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await comm_task
            raise asyncio.CancelledError(
                (cancel.reason if cancel else None) or "Subprocess cancelled"
            )

        stdout, stderr = comm_task.result()
        returncode = proc.returncode if proc.returncode is not None else 0

    except asyncio.CancelledError:
        await _terminate_and_reap(proc)
        if not comm_task.done():
            comm_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await comm_task
        raise
    finally:
        if cancel_task is not None and not cancel_task.done():
            cancel_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await cancel_task
        if cancel is not None:
            cancel.untrack_process(proc)

    if check and returncode != 0:
        raise MediaProcessError(cmd, returncode, stderr.decode("utf-8", errors="replace"))

    return returncode, stdout, stderr


def _parse_fps(rate_str: str | None) -> float:
    if not rate_str or rate_str in ("0/0", "N/A"):
        return 0.0
    if "/" in rate_str:
        num_s, den_s = rate_str.split("/", 1)
        try:
            num = float(num_s)
            den = float(den_s)
            return num / den if den > 0 else 0.0
        except ValueError:
            return 0.0
    try:
        return float(rate_str)
    except ValueError:
        return 0.0


async def probe_media(
    path: Path | str,
    *,
    ffprobe_bin: str = "ffprobe",
    cancel: CancellationToken | None = None,
    timeout_sec: float = 30.0,
) -> MediaProbe:
    """Run ffprobe on a media file and return structured stream & container metadata."""
    resolved = Path(path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Media file not found: {resolved}")

    cmd = [
        ffprobe_bin,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(resolved),
    ]
    _, stdout, _ = await run_subprocess(cmd, cancel=cancel, timeout_sec=timeout_sec)
    data = json.loads(stdout.decode("utf-8"))

    streams: list[dict[str, Any]] = data.get("streams", [])
    fmt: dict[str, Any] = data.get("format", {})

    v_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    a_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)

    duration_sec = 0.0
    for candidate in (
        fmt.get("duration"),
        v_stream.get("duration") if v_stream else None,
        a_stream.get("duration") if a_stream else None,
    ):
        if candidate is not None:
            try:
                val = float(candidate)
                if val > 0:
                    duration_sec = val
                    break
            except (TypeError, ValueError):
                continue

    width = int(v_stream.get("width", 0)) if v_stream else 0
    height = int(v_stream.get("height", 0)) if v_stream else 0

    fps = 0.0
    if v_stream:
        fps = _parse_fps(v_stream.get("avg_frame_rate")) or _parse_fps(v_stream.get("r_frame_rate"))

    nb_frames: int | None = None
    if v_stream and v_stream.get("nb_frames") not in (None, "N/A"):
        with contextlib.suppress(ValueError, TypeError):
            nb_frames = int(v_stream["nb_frames"])
    if nb_frames is None and fps > 0 and duration_sec > 0:
        nb_frames = max(1, round(fps * duration_sec))

    return MediaProbe(
        path=resolved,
        duration_sec=duration_sec,
        width=width,
        height=height,
        fps=fps,
        has_video=v_stream is not None,
        has_audio=a_stream is not None,
        video_codec=v_stream.get("codec_name") if v_stream else None,
        audio_codec=a_stream.get("codec_name") if a_stream else None,
        pix_fmt=v_stream.get("pix_fmt") if v_stream else None,
        nb_frames=nb_frames,
        raw=data,
    )
