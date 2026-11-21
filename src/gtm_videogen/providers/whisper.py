"""Whisper transcription provider supporting local whisper-cli and fal.ai Whisper.

Key contract:
- If WHISPER_MODEL_PATH is explicitly configured, runs local whisper-cli ONLY.
  If the model file or binary is missing/broken, raises WhisperError loudly
  (NO silent fallback to cloud!).
- Otherwise, if FAL_KEY is configured, submits audio to fal.ai Whisper (fal-ai/whisper)
  via ResilientHttpClient with CostMeter and progressive polling backoff.
- Otherwise raises WhisperError explaining neither local nor cloud Whisper is configured.
"""

from __future__ import annotations

import asyncio
import base64
import shutil
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any

from gtm_videogen.core.config import Settings, get_settings
from gtm_videogen.core.cost_meter import CostMeter, current_cost_meter
from gtm_videogen.core.http_client import ResilientHttpClient
from gtm_videogen.core.job_registry import CancellationToken

WHISPER_POLL_SCHEDULE: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 3.0)
FAL_WHISPER_MODEL = "fal-ai/whisper"
FAL_WHISPER_UNIT_USD = 0.002

_TERMINAL_SUCCESS = frozenset({"completed", "succeeded", "success", "done", "ok"})
_TERMINAL_FAILURE = frozenset({"failed", "error", "cancelled", "canceled", "rejected"})


class WhisperError(RuntimeError):
    """Raised when local or cloud Whisper transcription fails or is misconfigured."""


def _mime_for_audio(audio_path: Path) -> str:
    ext = audio_path.suffix.lower()
    if ext == ".mp3":
        return "audio/mpeg"
    if ext in (".m4a", ".mp4"):
        return "audio/mp4"
    if ext == ".ogg":
        return "audio/ogg"
    return "audio/wav"


def _extract_transcript_text(data: Any) -> str:
    if not isinstance(data, dict):
        return ""
    if isinstance(data.get("text"), str) and data["text"].strip():
        return data["text"].strip()
    chunks = data.get("chunks")
    if isinstance(chunks, list):
        parts = [
            str(chunk.get("text", "")).strip()
            for chunk in chunks
            if isinstance(chunk, dict) and str(chunk.get("text", "")).strip()
        ]
        if parts:
            return " ".join(parts)
    for nested_key in ("output", "result"):
        nested = data.get(nested_key)
        if isinstance(nested, dict):
            found = _extract_transcript_text(nested)
            if found:
                return found
    return ""


async def _cancellable_sleep(
    delay: float,
    *,
    cancel: CancellationToken | None,
    sleep_fn: Callable[[float], Awaitable[None]] | None = None,
) -> None:
    if cancel is not None:
        cancel.raise_if_cancelled()
    if sleep_fn is not None:
        await sleep_fn(delay)
        if cancel is not None:
            cancel.raise_if_cancelled()
        return
    if cancel is None:
        await asyncio.sleep(delay)
        return
    try:
        await asyncio.wait_for(cancel.wait(), timeout=delay)
        cancel.raise_if_cancelled()
    except TimeoutError:
        return


class WhisperProvider:
    """Speech-to-text transcription via local whisper-cli or fal.ai Whisper."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        http_client: ResilientHttpClient | None = None,
        cost_meter: CostMeter | None = None,
        whisper_model_path: str | Path | None = None,
        whisper_bin: str | None = None,
        whisper_lang: str | None = None,
        fal_key: str | None = None,
        poll_schedule: Sequence[float] = WHISPER_POLL_SCHEDULE,
        max_poll_attempts: int = 120,
        sleep_fn: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._http_client = http_client
        self._cost_meter = cost_meter
        self._whisper_model_path = (
            str(whisper_model_path) if whisper_model_path is not None else None
        )
        self._whisper_bin = whisper_bin
        self._whisper_lang = whisper_lang
        self._fal_key = fal_key
        self._poll_schedule = tuple(float(s) for s in poll_schedule) or WHISPER_POLL_SCHEDULE
        self._max_poll_attempts = max(1, max_poll_attempts)
        self._sleep_fn = sleep_fn

    def _effective_model_path(self) -> str:
        raw = (
            self._whisper_model_path
            if self._whisper_model_path is not None
            else self.settings.whisper_model_path
        )
        return raw.strip()

    def _effective_fal_key(self) -> str:
        raw = self._fal_key if self._fal_key is not None else self.settings.fal_key
        return raw.strip()

    def _effective_lang(self) -> str:
        raw = self._whisper_lang if self._whisper_lang is not None else self.settings.whisper_lang
        return raw.strip() or "en"

    async def transcribe(
        self,
        audio_path: Path,
        vocabulary: list[str] | None = None,
        cancel: CancellationToken | None = None,
    ) -> str:
        """Transcribe an audio file using local whisper-cli (when configured) or fal.ai Whisper."""
        if cancel is not None:
            cancel.raise_if_cancelled()

        path = Path(audio_path)
        if not path.is_file():
            raise WhisperError(f"Audio file does not exist: {path}")

        model_path_str = self._effective_model_path()
        if model_path_str:
            # Explicit local configuration: fail loudly if missing/broken (NO silent fallback).
            return await self._transcribe_local(
                audio_path=path,
                model_path=Path(model_path_str),
                vocabulary=vocabulary,
                cancel=cancel,
            )

        fal_key = self._effective_fal_key()
        if fal_key:
            return await self._transcribe_fal(
                audio_path=path,
                fal_key=fal_key,
                vocabulary=vocabulary,
                cancel=cancel,
            )

        raise WhisperError(
            "Whisper is not configured: set WHISPER_MODEL_PATH for local whisper-cli "
            "or FAL_KEY for fal.ai Whisper."
        )

    async def _transcribe_local(
        self,
        *,
        audio_path: Path,
        model_path: Path,
        vocabulary: list[str] | None,
        cancel: CancellationToken | None,
    ) -> str:
        if not model_path.is_file():
            raise WhisperError(f"Configured WHISPER_MODEL_PATH does not exist: {model_path}")

        explicit_bin = (
            self._whisper_bin if self._whisper_bin is not None else self.settings.whisper_bin
        ).strip()
        if explicit_bin:
            resolved_bin = shutil.which(explicit_bin) or (
                explicit_bin if Path(explicit_bin).is_file() else None
            )
        else:
            resolved_bin = shutil.which("whisper-cli") or shutil.which("whisper")

        if not resolved_bin:
            raise WhisperError(
                f"WHISPER_MODEL_PATH is configured ({model_path}), but local whisper binary "
                f"({explicit_bin or 'whisper-cli'}) was not found."
            )

        cmd: list[str] = [
            resolved_bin,
            "-m",
            str(model_path),
            "-f",
            str(audio_path),
            "-l",
            self._effective_lang(),
            "--no-timestamps",
        ]
        if vocabulary:
            prompt_str = ", ".join(term.strip() for term in vocabulary if term and term.strip())
            if prompt_str:
                cmd.extend(["--prompt", prompt_str])

        if cancel is not None:
            cancel.raise_if_cancelled()

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as exc:
            raise WhisperError(
                f"Failed to start local whisper-cli ({resolved_bin}): {exc}"
            ) from exc

        if cancel is not None:
            cancel.track_process(proc)
        try:
            stdout_bytes, stderr_bytes = await proc.communicate()
        finally:
            if cancel is not None:
                cancel.untrack_process(proc)

        if cancel is not None:
            cancel.raise_if_cancelled()

        if proc.returncode != 0:
            err_text = stderr_bytes.decode("utf-8", errors="replace").strip()
            raise WhisperError(
                f"Local whisper-cli failed with exit code {proc.returncode}: {err_text}"
            )

        return stdout_bytes.decode("utf-8", errors="replace").strip()

    async def _transcribe_fal(
        self,
        *,
        audio_path: Path,
        fal_key: str,
        vocabulary: list[str] | None,
        cancel: CancellationToken | None,
    ) -> str:
        meter = self._cost_meter or current_cost_meter()
        reservation_id: str | None = None
        if meter is not None:
            reservation_id = meter.reserve(
                f"whisper:{audio_path.name}",
                FAL_WHISPER_UNIT_USD,
                stage="qa_whisper",
                provider="fal",
                model=FAL_WHISPER_MODEL,
            )

        client = self._http_client
        owns_client = client is None
        if client is None:
            client = ResilientHttpClient(
                connect_timeout=self.settings.http_connect_timeout_sec,
                read_timeout=self.settings.http_read_timeout_sec,
                write_timeout=self.settings.http_write_timeout_sec,
                pool_timeout=self.settings.http_pool_timeout_sec,
                max_retries=self.settings.http_max_retries,
            )

        try:
            send_path = audio_path
            tmp_mp3: Path | None = None
            if audio_path.suffix.lower() in (".mp4", ".mov", ".mkv", ".webm"):
                ffmpeg_bin = shutil.which("ffmpeg")
                if ffmpeg_bin:
                    import tempfile

                    tmp_fd, tmp_name = tempfile.mkstemp(prefix="gtm_whisper_", suffix=".mp3")
                    import os

                    os.close(tmp_fd)
                    tmp_mp3 = Path(tmp_name)
                    proc = await asyncio.create_subprocess_exec(
                        ffmpeg_bin,
                        "-y",
                        "-i",
                        str(audio_path),
                        "-vn",
                        "-ac",
                        "1",
                        "-ar",
                        "16000",
                        "-b:a",
                        "64k",
                        str(tmp_mp3),
                        stdout=asyncio.subprocess.DEVNULL,
                        stderr=asyncio.subprocess.DEVNULL,
                    )
                    await proc.communicate()
                    if proc.returncode == 0 and tmp_mp3.is_file() and tmp_mp3.stat().st_size > 0:
                        send_path = tmp_mp3

            try:
                raw_bytes = send_path.read_bytes()
                b64 = base64.b64encode(raw_bytes).decode("ascii")
                data_url = f"data:{_mime_for_audio(send_path)};base64,{b64}"
            finally:
                if tmp_mp3 is not None and tmp_mp3.exists():
                    tmp_mp3.unlink(missing_ok=True)

            lang = self._effective_lang()
            payload: dict[str, Any] = {
                "audio_url": data_url,
                "task": "transcribe",
                "chunk_level": "segment",
                "language": lang.lower()[:2],
            }
            if vocabulary:
                prompt_str = ", ".join(term.strip() for term in vocabulary if term and term.strip())
                if prompt_str:
                    payload["prompt"] = prompt_str

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Key {fal_key}",
            }
            submit_url = f"https://queue.fal.run/{FAL_WHISPER_MODEL}"
            submitted = await client.post_json(
                submit_url,
                json=payload,
                headers=headers,
                cancel=cancel,
            )
            if not isinstance(submitted, dict):
                raise WhisperError(f"fal.ai Whisper returned non-object payload: {submitted!r}")

            text = _extract_transcript_text(submitted)
            if not text and ("status_url" in submitted or "request_id" in submitted):
                request_id = submitted.get("request_id")
                status_url = submitted.get("status_url") or (
                    f"https://queue.fal.run/{FAL_WHISPER_MODEL}/requests/{request_id}/status"
                )
                response_url = submitted.get("response_url") or (
                    f"https://queue.fal.run/{FAL_WHISPER_MODEL}/requests/{request_id}"
                )
                status_data = await self._poll_until_complete(
                    client=client,
                    status_url=str(status_url),
                    headers=headers,
                    cancel=cancel,
                )
                final_url = str(status_data.get("response_url") or response_url)
                result_data = await client.get_json(final_url, headers=headers, cancel=cancel)
                text = _extract_transcript_text(result_data)

            if meter is not None and reservation_id is not None:
                meter.settle(reservation_id, FAL_WHISPER_UNIT_USD)
                reservation_id = None

            return text
        except BaseException:
            if meter is not None and reservation_id is not None:
                meter.release(reservation_id)
            raise
        finally:
            if owns_client:
                await client.aclose()

    async def _poll_until_complete(
        self,
        *,
        client: ResilientHttpClient,
        status_url: str,
        headers: dict[str, str],
        cancel: CancellationToken | None,
    ) -> dict[str, Any]:
        for attempt in range(self._max_poll_attempts):
            if cancel is not None:
                cancel.raise_if_cancelled()

            delay = self._poll_schedule[min(attempt, len(self._poll_schedule) - 1)]
            await _cancellable_sleep(delay, cancel=cancel, sleep_fn=self._sleep_fn)

            status_data = await client.get_json(status_url, headers=headers, cancel=cancel)
            if not isinstance(status_data, dict):
                raise WhisperError(f"fal.ai Whisper non-object status payload: {status_data!r}")

            raw_status = str(status_data.get("status", "")).strip().lower()
            if raw_status in _TERMINAL_SUCCESS:
                return status_data
            if raw_status in _TERMINAL_FAILURE:
                err_msg = status_data.get("error") or status_data.get("message") or raw_status
                raise WhisperError(f"fal.ai Whisper transcription failed: {err_msg}")

        raise WhisperError(
            f"fal.ai Whisper transcription timed out after {self._max_poll_attempts} polls"
        )

    async def transcribe_words(
        self,
        audio_path: Path | str,
        *,
        cancel: CancellationToken | None = None,
    ) -> list[dict[str, Any]]:
        """Return word-level [{word, start, end}] timestamps via fal-ai/whisper (best-effort)."""
        path = Path(audio_path)
        fal_key = self.settings.fal_key.strip()
        if not path.is_file() or not fal_key:
            return []
        try:
            raw_bytes = path.read_bytes()
            b64 = base64.b64encode(raw_bytes).decode("ascii")
            data_url = f"data:{_mime_for_audio(path)};base64,{b64}"
            headers = {"Content-Type": "application/json", "Authorization": f"Key {fal_key}"}
            payload = {
                "audio_url": data_url,
                "task": "transcribe",
                "chunk_level": "word",
                "language": self._effective_lang().lower()[:2],
            }
            async with ResilientHttpClient(
                connect_timeout=self.settings.http_connect_timeout_sec,
                read_timeout=45.0,
                write_timeout=20.0,
                pool_timeout=10.0,
                max_retries=1,
            ) as client:
                data = await client.post_json(
                    f"https://fal.run/{FAL_WHISPER_MODEL}",
                    json=payload,
                    headers=headers,
                    cancel=cancel,
                )
            chunks = data.get("chunks") if isinstance(data, dict) else None
            out: list[dict[str, Any]] = []
            if isinstance(chunks, list):
                for ch in chunks:
                    if not isinstance(ch, dict):
                        continue
                    w = str(ch.get("text", "")).strip()
                    ts = ch.get("timestamp")
                    if w and isinstance(ts, list) and len(ts) >= 2 and ts[0] is not None:
                        out.append(
                            {
                                "word": w,
                                "start": float(ts[0]),
                                "end": float(ts[1] if ts[1] is not None else ts[0] + 0.25),
                            }
                        )
            return out
        except Exception:
            return []
