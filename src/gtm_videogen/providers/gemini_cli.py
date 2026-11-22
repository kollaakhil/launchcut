"""Google Gemini API CLI (`google-gemini/gemini-api-cli`) subprocess provider.

Wraps the zero-idle-RAM static Go binary `gemini-api` (`v0.5.0+`) for:
  1. Native Full-Video + Audio Multimodal QA (`gemini-api analyze` + `files upload --wait`)
  2. Structured Story & Pitch JSON generation (`gemini-api generate --stream=false`)
  3. Conversational/Chained Video generation (`gemini-api video` -> `gemini-omni-1.2-flash`)
  4. Anchor Keyframe Image generation (`gemini-api image` -> `gemini-nano-banana-2.1`)
  5. Expressive Voiceover TTS (`gemini-api tts` -> `gemini-3.8-flash-tts`) & Music (`lyria-3.5`)
  6. Audio/Video Transcription (`gemini-api transcribe --format json`)
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

from gtm_videogen.core.config import Settings, get_settings
from gtm_videogen.core.job_registry import CancellationToken
from gtm_videogen.media._subprocess import run_subprocess

_INLINE_MEDIA_LIMIT_BYTES = 13_500_000  # CLI keeps inline base64 payloads < 20 MB (~14 MB raw)
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


def resolve_gemini_cli_bin(settings: Settings | None = None) -> str | None:
    """Locate the `gemini-api` executable on PATH or standard install locations."""
    resolved = settings or get_settings()
    configured = (resolved.gemini_cli_bin or "gemini-api").strip()
    found = shutil.which(configured)
    if found:
        return found
    for candidate in (
        Path.home() / ".local" / "bin" / "gemini-api",
        Path("/usr/local/bin/gemini-api"),
        Path("/usr/bin/gemini-api"),
    ):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def is_gemini_cli_ready(settings: Settings | None = None, *, allow_without_key: bool = False) -> bool:
    """Return True when `gemini-api` binary is installed and an API key is configured."""
    resolved = settings or get_settings()
    if resolve_gemini_cli_bin(resolved) is None:
        return False
    if allow_without_key:
        return True
    return bool(resolved.effective_gemini_key())


def extract_json_object(raw_text: str) -> dict[str, Any] | None:
    """Extract a JSON dictionary from raw model text or fenced markdown output."""
    text = (raw_text or "").strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    fence_m = _JSON_FENCE_RE.search(text)
    if fence_m:
        try:
            parsed = json.loads(fence_m.group(1))
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last > first:
        try:
            parsed = json.loads(text[first : last + 1])
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    return None


def _base_cli_flags(api_key: str, *, dry_run: bool = False) -> list[str]:
    flags = ["--no-interactive", "--no-keyring", "--color", "never"]
    if api_key:
        flags.extend(["--api-key", api_key])
    if dry_run:
        flags.extend(["--dry-run", "-o", "json"])
    return flags


def _normalize_gemini_model_name(model_id: str | None) -> str:
    """Normalize UI/OpenRouter model IDs into native Gemini API model IDs."""
    raw = (model_id or "").strip()
    if not raw:
        return ""
    for prefix in ("gemini-cli/", "google/"):
        if raw.startswith(prefix):
            raw = raw[len(prefix) :]
    # Only forward model names that belong to Gemini/Gemma/Lyria/Nano-Banana families
    low = raw.lower()
    if any(low.startswith(p) for p in ("gemini-", "gemma-", "lyria-", "nano-banana", "veo-")):
        return raw
    return ""


async def gemini_generate_json(
    *,
    system_prompt: str,
    user_prompt: str,
    model: str = "",
    settings: Settings | None = None,
    dry_run: bool = False,
    timeout_sec: float = 90.0,
    cancel: CancellationToken | None = None,
) -> dict[str, Any] | None:
    """Call `gemini-api generate --stream=false` and parse a structured JSON object."""
    resolved = settings or get_settings()
    cli_bin = resolve_gemini_cli_bin(resolved)
    api_key = "dry_run_key" if dry_run else resolved.effective_gemini_key()
    if not cli_bin or not api_key:
        return None

    full_prompt = (
        f"{user_prompt.strip()}\n\n"
        "IMPORTANT: Respond ONLY with a single valid JSON object and no markdown prose."
    )
    cmd = [
        cli_bin,
        "generate",
        full_prompt,
        "--system",
        system_prompt,
        "--stream=false",
        *_base_cli_flags(api_key, dry_run=dry_run),
    ]
    norm_model = _normalize_gemini_model_name(model)
    if norm_model:
        cmd.extend(["-m", norm_model])

    rc, stdout_b, _ = await run_subprocess(
        cmd,
        timeout_sec=timeout_sec,
        cancel=cancel,
        check=False,
    )
    if rc != 0:
        return None
    return extract_json_object(stdout_b.decode("utf-8", errors="replace"))


async def gemini_upload_file(
    media_path: Path,
    *,
    settings: Settings | None = None,
    timeout_sec: float = 180.0,
    cancel: CancellationToken | None = None,
) -> str | None:
    """Upload a video/audio file (>14 MB) via `gemini-api files upload --wait` -> `files/<id>`."""
    resolved = settings or get_settings()
    cli_bin = resolve_gemini_cli_bin(resolved)
    api_key = resolved.effective_gemini_key()
    if not cli_bin or not api_key or not media_path.is_file():
        return None

    cmd = [
        cli_bin,
        "files",
        "upload",
        str(media_path),
        "--wait",
        *_base_cli_flags(api_key),
    ]
    rc, stdout_b, _ = await run_subprocess(
        cmd,
        timeout_sec=timeout_sec,
        cancel=cancel,
        check=False,
    )
    if rc != 0:
        return None
    for line in stdout_b.decode("utf-8", errors="replace").splitlines():
        tok = line.strip()
        if tok.startswith("files/"):
            return tok
    return None


async def gemini_delete_file(
    file_ref: str,
    *,
    settings: Settings | None = None,
) -> None:
    """Best-effort cleanup of an uploaded `files/<id>` asset."""
    resolved = settings or get_settings()
    cli_bin = resolve_gemini_cli_bin(resolved)
    api_key = resolved.effective_gemini_key()
    if not cli_bin or not api_key or not file_ref.startswith("files/"):
        return
    await run_subprocess(
        [cli_bin, "files", "delete", file_ref, *_base_cli_flags(api_key)],
        timeout_sec=25.0,
        check=False,
    )


async def gemini_analyze_media_json(
    media_path: Path,
    *,
    question: str,
    system_prompt: str = "",
    model: str = "",
    settings: Settings | None = None,
    dry_run: bool = False,
    timeout_sec: float = 150.0,
    cancel: CancellationToken | None = None,
) -> dict[str, Any] | None:
    """Analyze a full video/audio/image file via `gemini-api analyze` (auto-uploads if >13.5 MB)."""
    resolved = settings or get_settings()
    cli_bin = resolve_gemini_cli_bin(resolved)
    api_key = "dry_run_key" if dry_run else resolved.effective_gemini_key()
    if not cli_bin or not api_key or not media_path.is_file():
        return None

    uploaded_ref: str | None = None
    try:
        input_arg = str(media_path)
        if not dry_run and media_path.stat().st_size > _INLINE_MEDIA_LIMIT_BYTES:
            uploaded_ref = await gemini_upload_file(
                media_path,
                settings=resolved,
                cancel=cancel,
            )
            if not uploaded_ref:
                return None
            input_arg = uploaded_ref

        cmd = [
            cli_bin,
            "analyze",
            question,
            "-i",
            input_arg,
            *_base_cli_flags(api_key, dry_run=dry_run),
        ]
        if system_prompt.strip():
            cmd.extend(["--system", system_prompt.strip()])
        norm_model = _normalize_gemini_model_name(model)
        if norm_model:
            cmd.extend(["-m", norm_model])

        rc, stdout_b, _ = await run_subprocess(
            cmd,
            timeout_sec=timeout_sec,
            cancel=cancel,
            check=False,
        )
        if rc != 0:
            return None
        return extract_json_object(stdout_b.decode("utf-8", errors="replace"))
    finally:
        if uploaded_ref:
            await gemini_delete_file(uploaded_ref, settings=resolved)


async def gemini_generate_video(
    *,
    prompt: str,
    out_path: Path,
    aspect_ratio: str = "16:9",
    resolution: str = "1080p",
    start_image_path: Path | None = None,
    model: str = "gemini-omni-1.2-flash",
    settings: Settings | None = None,
    dry_run: bool = False,
    timeout_sec: float = 360.0,
    cancel: CancellationToken | None = None,
) -> Path | dict[str, Any] | None:
    """Generate a video clip via `gemini-api video` (supports `-i start_image.jpg` chaining)."""
    resolved = settings or get_settings()
    cli_bin = resolve_gemini_cli_bin(resolved)
    api_key = "dry_run_key" if dry_run else resolved.effective_gemini_key()
    if not cli_bin or not api_key:
        return None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    norm_res = resolution.lower() if resolution.lower() in ("360p", "720p", "1080p", "4k") else "1080p"
    cmd = [
        cli_bin,
        "video",
        prompt,
        "--aspect-ratio",
        aspect_ratio,
        "--resolution",
        norm_res,
        "--out",
        str(out_path),
        *_base_cli_flags(api_key, dry_run=dry_run),
    ]
    norm_model = _normalize_gemini_model_name(model) or "gemini-omni-1.2-flash"
    cmd.extend(["-m", norm_model])
    if start_image_path is not None and start_image_path.is_file():
        cmd.extend(["-i", str(start_image_path)])

    rc, stdout_b, _ = await run_subprocess(
        cmd,
        timeout_sec=timeout_sec,
        cancel=cancel,
        check=False,
    )
    if rc != 0:
        return None
    if dry_run:
        return extract_json_object(stdout_b.decode("utf-8", errors="replace"))
    if out_path.is_file() and out_path.stat().st_size > 0:
        return out_path
    # Check if CLI printed a generated path on stdout
    for line in stdout_b.decode("utf-8", errors="replace").splitlines():
        cand = Path(line.strip())
        if cand.is_file() and cand.stat().st_size > 0:
            shutil.move(str(cand), str(out_path))
            return out_path
    return None


async def gemini_generate_image(
    *,
    prompt: str,
    out_path: Path,
    aspect_ratio: str = "16:9",
    reference_image_path: Path | None = None,
    model: str = "gemini-nano-banana-2.1",
    settings: Settings | None = None,
    dry_run: bool = False,
    timeout_sec: float = 90.0,
    cancel: CancellationToken | None = None,
) -> Path | dict[str, Any] | None:
    """Generate a keyframe still via `gemini-api image`."""
    resolved = settings or get_settings()
    cli_bin = resolve_gemini_cli_bin(resolved)
    api_key = "dry_run_key" if dry_run else resolved.effective_gemini_key()
    if not cli_bin or not api_key:
        return None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        cli_bin,
        "image",
        prompt,
        "--aspect-ratio",
        aspect_ratio,
        "--image-size",
        "2K",
        "--out",
        str(out_path),
        *_base_cli_flags(api_key, dry_run=dry_run),
    ]
    norm_model = _normalize_gemini_model_name(model) or "gemini-nano-banana-2.1"
    cmd.extend(["-m", norm_model])
    if reference_image_path is not None and reference_image_path.is_file():
        cmd.extend(["-i", str(reference_image_path)])

    rc, stdout_b, _ = await run_subprocess(
        cmd,
        timeout_sec=timeout_sec,
        cancel=cancel,
        check=False,
    )
    if rc != 0:
        return None
    if dry_run:
        return extract_json_object(stdout_b.decode("utf-8", errors="replace"))
    if out_path.is_file() and out_path.stat().st_size > 0:
        return out_path
    for line in stdout_b.decode("utf-8", errors="replace").splitlines():
        cand = Path(line.strip())
        if cand.is_file() and cand.stat().st_size > 0:
            shutil.move(str(cand), str(out_path))
            return out_path
    return None


async def gemini_generate_tts(
    *,
    text: str,
    out_path: Path,
    voice: str = "Kore",
    style: str = "",
    settings: Settings | None = None,
    dry_run: bool = False,
    timeout_sec: float = 90.0,
    cancel: CancellationToken | None = None,
) -> Path | dict[str, Any] | None:
    """Generate expressive narration via `gemini-api tts` (`gemini-3.8-flash-tts`)."""
    resolved = settings or get_settings()
    cli_bin = resolve_gemini_cli_bin(resolved)
    api_key = "dry_run_key" if dry_run else resolved.effective_gemini_key()
    if not cli_bin or not api_key or not text.strip():
        return None

    voice_map = {
        "rachel": "Kore",
        "adam": "Puck",
        "bella": "Zephyr",
        "josh": "Charon",
    }
    mapped_voice = voice_map.get(voice.strip().lower(), voice.strip() or "Kore")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        cli_bin,
        "tts",
        text.strip(),
        "--voice",
        mapped_voice,
        "--out",
        str(out_path),
        *_base_cli_flags(api_key, dry_run=dry_run),
    ]
    if style.strip():
        cmd.extend(["--style", style.strip()])

    rc, stdout_b, _ = await run_subprocess(
        cmd,
        timeout_sec=timeout_sec,
        cancel=cancel,
        check=False,
    )
    if rc != 0:
        return None
    if dry_run:
        return extract_json_object(stdout_b.decode("utf-8", errors="replace"))
    if out_path.is_file() and out_path.stat().st_size > 0:
        return out_path
    for line in stdout_b.decode("utf-8", errors="replace").splitlines():
        cand = Path(line.strip())
        if cand.is_file() and cand.stat().st_size > 0:
            shutil.move(str(cand), str(out_path))
            return out_path
    return None


async def gemini_transcribe_media(
    media_path: Path,
    *,
    duration_sec: float = 10.0,
    settings: Settings | None = None,
    timeout_sec: float = 120.0,
    cancel: CancellationToken | None = None,
) -> tuple[str, list[dict[str, Any]]] | None:
    """Transcribe audio/video via `gemini-api transcribe --format json`."""
    resolved = settings or get_settings()
    cli_bin = resolve_gemini_cli_bin(resolved)
    api_key = resolved.effective_gemini_key()
    if not cli_bin or not api_key or not media_path.is_file():
        return None

    tmp_dir = Path(tempfile.mkdtemp(prefix="gemini_stt_"))
    uploaded_ref: str | None = None
    try:
        input_arg = str(media_path)
        if media_path.stat().st_size > _INLINE_MEDIA_LIMIT_BYTES:
            uploaded_ref = await gemini_upload_file(media_path, settings=resolved, cancel=cancel)
            if not uploaded_ref:
                return None
            input_arg = uploaded_ref

        out_json = tmp_dir / "transcript.json"
        cmd = [
            cli_bin,
            "transcribe",
            "-i",
            input_arg,
            "--format",
            "json",
            "--no-speakers",
            "--out",
            str(out_json),
            *_base_cli_flags(api_key),
        ]
        rc, stdout_b, _ = await run_subprocess(
            cmd,
            timeout_sec=timeout_sec,
            cancel=cancel,
            check=False,
        )
        if rc != 0:
            return None
        target = out_json
        if not target.is_file():
            for line in stdout_b.decode("utf-8", errors="replace").splitlines():
                cand = Path(line.strip())
                if cand.is_file():
                    target = cand
                    break
        if not target.is_file():
            return None

        raw_data = json.loads(target.read_text(encoding="utf-8", errors="replace"))
        words: list[dict[str, Any]] = []
        full_text = ""
        if isinstance(raw_data, dict):
            full_text = str(raw_data.get("text") or raw_data.get("transcript") or "").strip()
            segments = raw_data.get("segments") or raw_data.get("words") or []
            if isinstance(segments, list):
                for seg in segments:
                    if not isinstance(seg, dict):
                        continue
                    txt = str(seg.get("word") or seg.get("text") or "").strip()
                    if not txt:
                        continue
                    t0 = float(seg.get("start") or seg.get("start_sec") or 0.0)
                    t1 = float(seg.get("end") or seg.get("end_sec") or (t0 + 0.3))
                    for tok in txt.split():
                        words.append({"word": tok, "start": round(t0, 2), "end": round(max(t0 + 0.08, t1), 2)})
        if not words and full_text:
            toks = [w for w in full_text.split() if w]
            step = max(1.0, duration_sec) / max(1, len(toks))
            for i, tok in enumerate(toks):
                t0 = round(i * step, 2)
                words.append({"word": tok, "start": t0, "end": round(t0 + step * 0.9, 2)})
        if not words:
            return None
        return (" ".join(w["word"] for w in words), words)
    except Exception:
        return None
    finally:
        if uploaded_ref:
            await gemini_delete_file(uploaded_ref, settings=resolved)
        shutil.rmtree(tmp_dir, ignore_errors=True)
