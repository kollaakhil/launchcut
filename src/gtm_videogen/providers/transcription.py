"""Zero-RAM Cloud Speech-to-Text with Word-Level Timestamps (fal.ai Whisper + Offline Mock).

Extracts a tiny 16kHz mono audio track from an uploaded founder video via FFmpeg and
transcribes word-level timestamps (`[{"word", "start", "end"}]`) via fal.ai `fal-ai/wizper`
(no local PyTorch/Whisper weights -> safe for 512MB Render containers).
"""

from __future__ import annotations

import base64
import shutil
import tempfile
from pathlib import Path
from typing import Any

from gtm_videogen.core.config import Settings, get_settings
from gtm_videogen.core.http_client import ResilientHttpClient
from gtm_videogen.core.job_registry import CancellationToken
from gtm_videogen.media._subprocess import probe_media, run_subprocess
from gtm_videogen.providers.gemini_cli import gemini_transcribe_media, is_gemini_cli_ready


def _deterministic_mock_words(
    fallback_text: str,
    duration_sec: float,
) -> list[dict[str, Any]]:
    """Build evenly spaced word timestamps for $0 Preview / offline testing."""
    clean = (fallback_text or "").strip()
    if not clean:
        clean = (
            "Today we are launching a faster way to build and ship studio grade "
            "product films with real footage crisp motion graphics and AI visuals"
        )
    raw_words = [w.strip() for w in clean.split() if w.strip()]
    if not raw_words:
        raw_words = ["Introducing", "our", "new", "launch"]
    dur = max(1.0, float(duration_sec or 10.0))
    step = dur / len(raw_words)
    out: list[dict[str, Any]] = []
    for idx, w in enumerate(raw_words):
        t0 = round(idx * step, 2)
        t1 = round(min(dur, (idx + 0.92) * step), 2)
        out.append({"word": w, "start": t0, "end": max(t0 + 0.08, t1)})
    return out


async def transcribe_video_words(
    video_path: Path,
    *,
    fallback_text: str = "",
    settings: Settings | None = None,
    mock: bool = False,
    cancel: CancellationToken | None = None,
) -> tuple[str, list[dict[str, Any]], float]:
    """Return `(full_text, word_timestamps, duration_sec)` for an uploaded founder video."""
    resolved = settings or get_settings()
    probe = await probe_media(video_path, ffprobe_bin=resolved.ffprobe_bin, cancel=cancel)
    duration_sec = max(1.0, float(probe.duration_sec or 10.0))

    if mock or not probe.has_audio:
        words = _deterministic_mock_words(fallback_text, duration_sec)
        return " ".join(w["word"] for w in words), words, duration_sec

    if not resolved.fal_key.strip():
        if is_gemini_cli_ready(resolved):
            gem_res = await gemini_transcribe_media(
                video_path,
                duration_sec=duration_sec,
                settings=resolved,
                cancel=cancel,
            )
            if gem_res is not None:
                return gem_res[0], gem_res[1], duration_sec
        words = _deterministic_mock_words(fallback_text, duration_sec)
        return " ".join(w["word"] for w in words), words, duration_sec

    tmp_dir = Path(tempfile.mkdtemp(prefix="stitch_stt_"))
    try:
        audio_mp3 = tmp_dir / "speech_16k.mp3"
        rc, _, _ = await run_subprocess(
            [
                resolved.ffmpeg_bin,
                "-y",
                "-threads",
                "1",
                "-i",
                str(video_path),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-b:a",
                "64k",
                str(audio_mp3),
            ],
            timeout_sec=60.0,
            cancel=cancel,
            check=False,
        )
        if rc != 0 or not audio_mp3.is_file():
            words = _deterministic_mock_words(fallback_text, duration_sec)
            return " ".join(w["word"] for w in words), words, duration_sec

        b64 = base64.b64encode(audio_mp3.read_bytes()).decode("ascii")
        audio_data_uri = f"data:audio/mpeg;base64,{b64}"

        headers = {
            "Authorization": f"Key {resolved.fal_key.strip()}",
            "Content-Type": "application/json",
        }
        async with ResilientHttpClient(
            connect_timeout=resolved.http_connect_timeout_sec,
            read_timeout=90.0,
            write_timeout=30.0,
            pool_timeout=10.0,
            max_retries=1,
        ) as client:
            try:
                data = await client.post_json(
                    "https://fal.run/fal-ai/wizper",
                    json={
                        "audio_url": audio_data_uri,
                        "task": "transcribe",
                        "chunk_level": "word",
                        "version": "3",
                    },
                    headers=headers,
                    cancel=cancel,
                )
            except Exception:
                words = _deterministic_mock_words(fallback_text, duration_sec)
                return " ".join(w["word"] for w in words), words, duration_sec

        chunks = data.get("chunks") if isinstance(data, dict) else None
        parsed_words: list[dict[str, Any]] = []
        if isinstance(chunks, list):
            for ch in chunks:
                if not isinstance(ch, dict):
                    continue
                txt = str(ch.get("text") or "").strip()
                ts = ch.get("timestamp")
                if not txt:
                    continue
                if isinstance(ts, list) and len(ts) >= 2:
                    t0 = float(ts[0] or 0.0)
                    t1 = float(ts[1] or (t0 + 0.25))
                else:
                    t0 = parsed_words[-1]["end"] if parsed_words else 0.0
                    t1 = t0 + 0.25
                parsed_words.append({"word": txt, "start": round(t0, 2), "end": round(max(t0 + 0.05, t1), 2)})

        if not parsed_words:
            full_t = str(data.get("text") or fallback_text or "") if isinstance(data, dict) else fallback_text
            parsed_words = _deterministic_mock_words(full_t, duration_sec)

        full_text = " ".join(w["word"] for w in parsed_words)
        return full_text, parsed_words, duration_sec
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
