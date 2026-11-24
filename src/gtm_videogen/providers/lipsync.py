"""Selective Close-Up Lip-Sync Provider with Silence-Padded Duration Lock.

Solves Trap #2 & Trap #3:
  - Only invoked on `delivery_mode == "on_screen_speaking"` close-up/MCU shots.
  - Builds a sample-exact duration-matched WAV (`[pre_pause silence] + [speech] + [post_hold silence]`)
    matching `probe_media(video).duration_sec` so lip-sync models NEVER truncate the clip or
    animate the character's mouth during silent pre/post reaction beats.
  - Fallback chain: `fal-ai/sync-lipsync/v2/pro` -> `fal-ai/sync-lipsync/v2` -> `fal-ai/latentsync`.
"""

from __future__ import annotations

import asyncio
import base64
import shutil
import wave
from pathlib import Path
from typing import Any

from gtm_videogen.core.config import Settings, get_settings
from gtm_videogen.core.http_client import ResilientHttpClient
from gtm_videogen.core.job_registry import CancellationToken
from gtm_videogen.media._subprocess import probe_media, run_subprocess

LIPSYNC_FALLBACK_CHAIN: tuple[tuple[str, float], ...] = (
    ("fal-ai/sync-lipsync/v2/pro", 0.08),
    ("fal-ai/sync-lipsync/v2", 0.05),
    ("fal-ai/latentsync", 0.04),
)

_POLL_SCHEDULE: tuple[float, ...] = (2.0, 3.0, 5.0)
_SUCCESS_STATES = frozenset({"completed", "succeeded", "success", "done", "ok"})
_FAIL_STATES = frozenset({"failed", "error", "cancelled", "canceled", "rejected"})


async def build_duration_locked_speech_wav(
    *,
    speech_audio_path: Path,
    target_duration_sec: float,
    pre_pause_sec: float,
    out_wav_path: Path,
    ffmpeg_bin: str = "ffmpeg",
    cancel: CancellationToken | None = None,
) -> Path:
    """Decode `speech_audio_path` to 24kHz mono 16-bit PCM and pad pre/post silence to exact `target_duration_sec`."""
    sr = 24000
    raw_pcm_path = out_wav_path.with_suffix(".raw.pcm")
    out_wav_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        await run_subprocess(
            [
                ffmpeg_bin,
                "-y",
                "-i",
                str(speech_audio_path),
                "-vn",
                "-ac",
                "1",
                "-ar",
                str(sr),
                "-f",
                "s16le",
                str(raw_pcm_path),
            ],
            timeout_sec=30.0,
            cancel=cancel,
            check=True,
        )
        speech_bytes = raw_pcm_path.read_bytes() if raw_pcm_path.is_file() else b""
    finally:
        raw_pcm_path.unlink(missing_ok=True)

    bytes_per_sample = 2
    total_target_samples = max(sr, int(round(target_duration_sec * sr)))
    pre_samples = max(0, min(total_target_samples // 2, int(round(pre_pause_sec * sr))))
    speech_samples = len(speech_bytes) // bytes_per_sample

    max_speech_samples = max(0, total_target_samples - pre_samples)
    if speech_samples > max_speech_samples:
        speech_bytes = speech_bytes[: max_speech_samples * bytes_per_sample]
        speech_samples = max_speech_samples

    post_samples = max(0, total_target_samples - pre_samples - speech_samples)
    full_pcm = (b"\x00\x00" * pre_samples) + speech_bytes + (b"\x00\x00" * post_samples)

    with wave.open(str(out_wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(bytes_per_sample)
        wf.setframerate(sr)
        wf.writeframes(full_pcm)
    return out_wav_path


def _encode_file_data_uri(path: Path, mime: str) -> str:
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _extract_video_url(data: Any) -> str | None:
    if not isinstance(data, dict):
        return None
    vid = data.get("video")
    if isinstance(vid, dict) and isinstance(vid.get("url"), str):
        return vid["url"]
    if isinstance(data.get("video_url"), str):
        return str(data["video_url"])
    out = data.get("output")
    if isinstance(out, dict) and isinstance(out.get("url"), str):
        return out["url"]
    return None


async def apply_selective_lipsync(
    *,
    video_path: Path,
    speech_audio_path: Path,
    pre_pause_sec: float,
    out_video_path: Path,
    settings: Settings | None = None,
    mock: bool = False,
    cancel: CancellationToken | None = None,
) -> tuple[Path, float, bool]:
    """Apply duration-locked lip-sync to a rendered close-up clip. Returns `(path, usd, applied)`."""
    resolved = settings or get_settings()
    probe = await probe_media(video_path, ffprobe_bin=resolved.ffprobe_bin, cancel=cancel)
    clip_dur = max(2.0, float(probe.duration_sec or 6.0))

    padded_wav = out_video_path.with_name(f"{out_video_path.stem}_lipsync_stem.wav")
    await build_duration_locked_speech_wav(
        speech_audio_path=speech_audio_path,
        target_duration_sec=clip_dur,
        pre_pause_sec=pre_pause_sec,
        out_wav_path=padded_wav,
        ffmpeg_bin=resolved.ffmpeg_bin,
        cancel=cancel,
    )

    if mock or not resolved.fal_key.strip():
        # Offline / Mock mode: preserve exact visual frames ($0) while verifying padded stem integrity
        out_video_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(video_path, out_video_path)
        return out_video_path, 0.0, False

    headers = {
        "Authorization": f"Key {resolved.fal_key.strip()}",
        "Content-Type": "application/json",
    }
    video_uri = _encode_file_data_uri(video_path, "video/mp4")
    audio_uri = _encode_file_data_uri(padded_wav, "audio/wav")

    async with ResilientHttpClient(
        connect_timeout=resolved.http_connect_timeout_sec,
        read_timeout=90.0,
        write_timeout=45.0,
        pool_timeout=15.0,
        max_retries=1,
    ) as client:
        for endpoint, usd_per_sec in LIPSYNC_FALLBACK_CHAIN:
            if cancel is not None:
                cancel.raise_if_cancelled()
            payload: dict[str, Any] = {
                "video_url": video_uri,
                "audio_url": audio_uri,
            }
            if "sync-lipsync" in endpoint:
                payload["sync_mode"] = "remap"
            try:
                submitted = await client.post_json(
                    f"https://queue.fal.run/{endpoint}",
                    json=payload,
                    headers=headers,
                    cancel=cancel,
                )
                if not isinstance(submitted, dict):
                    continue
                vid_url = _extract_video_url(submitted)
                if not vid_url:
                    req_id = submitted.get("request_id")
                    status_url = submitted.get("status_url") or (
                        f"https://queue.fal.run/{endpoint}/requests/{req_id}/status"
                        if req_id
                        else None
                    )
                    resp_url = submitted.get("response_url") or (
                        f"https://queue.fal.run/{endpoint}/requests/{req_id}" if req_id else None
                    )
                    if not status_url or not resp_url:
                        continue
                    for attempt in range(90):
                        if cancel is not None:
                            cancel.raise_if_cancelled()
                        await asyncio.sleep(_POLL_SCHEDULE[min(attempt, len(_POLL_SCHEDULE) - 1)])
                        st = await client.get_json(str(status_url), headers=headers, cancel=cancel)
                        if not isinstance(st, dict):
                            break
                        state = str(st.get("status") or "").strip().lower()
                        if state in _SUCCESS_STATES:
                            res_data = await client.get_json(
                                str(st.get("response_url") or resp_url),
                                headers=headers,
                                cancel=cancel,
                            )
                            vid_url = _extract_video_url(res_data)
                            break
                        if state in _FAIL_STATES:
                            break
                if vid_url:
                    vid_bytes = await client.get_bytes(vid_url, cancel=cancel)
                    if vid_bytes:
                        out_video_path.parent.mkdir(parents=True, exist_ok=True)
                        out_video_path.write_bytes(vid_bytes)
                        est_usd = round(clip_dur * usd_per_sec, 4)
                        return out_video_path, est_usd, True
            except Exception:
                continue

    # Graceful non-destructive fallback: never fail a film shoot if lip-sync endpoint is busy
    out_video_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(video_path, out_video_path)
    return out_video_path, 0.0, False


__all__ = [
    "LIPSYNC_FALLBACK_CHAIN",
    "apply_selective_lipsync",
    "build_duration_locked_speech_wav",
]
