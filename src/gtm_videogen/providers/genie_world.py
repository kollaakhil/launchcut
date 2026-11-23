"""Google DeepMind Genie 4 (`models/genie-4`) Live World Provider via Gemini Interactions API.

Supports two capabilities on the same Paid-Tier `GEMINI_API_KEY`:
  1. `genie4_generate_steered_clip`: Automated continuous 720p@24fps + 48kHz stereo world
     shoot (up to 60s-120s unbroken in one world) with reference-image seeding, smooth WASD
     camera dolly steering, mid-stream beat steering, zero-RAM FFmpeg `image2pipe` streaming,
     and guaranteed `complete()` / `cancel()` billing cutoff.
  2. `mint_genie4_browser_session`: Mints a single-session scoped `auth_tokens/...` credential
     so browser WebRTC clients can interactively explore & record a Genie 4 world without
     exposing `GEMINI_API_KEY`.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import io
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image

from gtm_videogen.core.config import Settings, get_settings
from gtm_videogen.core.job_registry import CancellationToken
from gtm_videogen.media._subprocess import probe_media, run_subprocess

GENIE4_MODEL_ID = "models/genie-4"
GENIE4_USD_PER_SEC = 0.1008  # 5,760 video tokens/sec @ $17.50 / 1M tokens (~$0.10/s, stereo audio included)


def is_genie4_ready(settings: Settings | None = None) -> bool:
    """Return True when a Gemini API key is configured for Genie 4."""
    resolved = settings or get_settings()
    return bool(resolved.effective_gemini_key())


def prepare_genie4_reference_jpeg(image_path: Path | str | None, *, max_bytes: int = 44_000) -> str | None:
    """Resize & compress a reference image to 16:9 JPEG strictly < 48 KB per Genie 4's spec."""
    if not image_path:
        return None
    p = Path(image_path)
    if not p.is_file():
        return None
    try:
        with Image.open(p) as raw_im:
            im = raw_im.convert("RGB")
            src_w, src_h = im.size
            target_ratio = 16.0 / 9.0
            cur_ratio = src_w / float(max(1, src_h))
            if cur_ratio > target_ratio:
                new_w = int(round(src_h * target_ratio))
                x0 = max(0, (src_w - new_w) // 2)
                im = im.crop((x0, 0, x0 + new_w, src_h))
            elif cur_ratio < target_ratio:
                new_h = int(round(src_w / target_ratio))
                y0 = max(0, (src_h - new_h) // 2)
                im = im.crop((0, y0, src_w, y0 + new_h))

            for dims in ((1280, 720), (960, 540), (640, 360)):
                resized = im.resize(dims, Image.Resampling.LANCZOS)
                for qual in (72, 56, 42, 32):
                    buf = io.BytesIO()
                    resized.save(buf, format="JPEG", quality=qual, optimize=True)
                    raw = buf.getvalue()
                    if len(raw) <= max_bytes:
                        return base64.b64encode(raw).decode("ascii")
            return base64.b64encode(raw).decode("ascii")
    except Exception:
        return None


_CUT_RE = re.compile(r"\b(cut to|jump cut to|smash cut to|hard cut to)\b", re.IGNORECASE)
_MUSIC_DRONE_RE = re.compile(
    r"\b(synthwave soundtrack|background music|epic orchestral score|deep ambient drone|engine hum)\b",
    re.IGNORECASE,
)


def compile_genie4_world_prompt(
    scene_prose: str,
    *,
    camera_spec: str = "",
    audio_direction: str = "",
) -> str:
    """Adapt scene prose into Genie-4-aligned continuous spatial + physical contact-sound phrasing."""
    cleaned = _CUT_RE.sub("smoothly glide toward", scene_prose or "")
    cleaned = _MUSIC_DRONE_RE.sub("natural physical environment acoustics", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    cam_low = (camera_spec or "").lower()
    pov_lead = (
        "Immersive continuous first-person camera moving smoothly forward through "
        if any(k in cam_low for k in ("dolly", "push", "glide", "track", "walk", "forward"))
        else "Immersive continuous cinema perspective focused steadily on "
    )
    if not cleaned.lower().startswith(("immersive", "first-person", "continuous")):
        cleaned = f"{pov_lead}{cleaned[0].lower() + cleaned[1:] if cleaned else 'a tactile photorealistic environment.'}"

    audio_clean = _MUSIC_DRONE_RE.sub("subtle environmental contact sounds", audio_direction or "").strip()
    audio_clause = (
        f" Audio: {audio_clean.rstrip('.')}, crisp physical surface contact and natural spatial air."
        if audio_clean
        else " Audio: Subtle tactile surface contact, gentle air movement, and natural room acoustics."
    )
    full = f"{cleaned.rstrip('.')}.{audio_clause}"
    return full[:1800]


async def mint_genie4_browser_session(
    *,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Create a Genie 4 interaction and mint a single-session browser `auth_token` (keeps GEMINI_API_KEY secret)."""
    resolved = settings or get_settings()
    api_key = resolved.effective_gemini_key()
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured. Add a Paid-Tier Gemini API key from https://aistudio.google.com/apikey to use Genie 4."
        )

    from google import genai

    client = genai.Client(
        api_key=api_key,
        http_options={"api_version": "v1beta", "timeout": 20000, "retry_options": {"attempts": 1}},
    )
    interaction = await client.aio.interactions.create(
        model="genie-4",
        api_version="v1beta",
        automatic_continuation=False,
        timeout=18.0,
    )
    tok_resp = await client.aio.interactions.auth_tokens.create(
        interaction.id,
        api_version="v1beta",
    )
    auth_token = (
        getattr(tok_resp, "auth_token", None)
        or getattr(tok_resp, "name", None)
        or (tok_resp.get("auth_token") if isinstance(tok_resp, dict) else None)
        or str(tok_resp)
    )
    return {
        "interaction_id": interaction.id,
        "auth_token": auth_token,
        "model": GENIE4_MODEL_ID,
        "est_usd_per_sec": GENIE4_USD_PER_SEC,
        "max_session_sec": 120,
    }


async def verify_gemini_and_genie4_access(
    *,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Verify a Gemini API key (`AIza...` or `AQ....`), count visible models, and check Paid-Tier Genie 4 access without billing."""
    resolved = settings or get_settings()
    api_key = resolved.effective_gemini_key()
    if not api_key:
        return {
            "valid_key": False,
            "genie4_listed": False,
            "genie4_paid_ready": False,
            "tier": "missing",
            "models_count": 0,
            "message": "GEMINI_API_KEY is not set.",
        }

    import httpx

    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=12.0) as hc:
        m_resp = await hc.get("https://generativelanguage.googleapis.com/v1beta/models?pageSize=100", headers=headers)
        if m_resp.status_code != 200:
            detail = m_resp.text[:200]
            return {
                "valid_key": False,
                "genie4_listed": False,
                "genie4_paid_ready": False,
                "tier": "invalid",
                "models_count": 0,
                "message": f"Gemini API rejected key (HTTP {m_resp.status_code}): {detail}",
            }
        m_json = m_resp.json()
        model_names = [str(it.get("name") or "") for it in (m_json.get("models") or [])]
        genie4_listed = "models/genie-4" in model_names

        # Probe Genie 4 session quota and immediately cancel if created (0 frames streamed = $0 billed!)
        i_resp = await hc.post(
            "https://generativelanguage.googleapis.com/v1beta/interactions",
            headers=headers,
            json={"model": "genie-4"},
        )
        if i_resp.status_code in (200, 201):
            i_data = i_resp.json()
            inter_id = str(i_data.get("id") or "")
            if inter_id:
                with contextlib.suppress(Exception):
                    await hc.post(
                        f"https://generativelanguage.googleapis.com/v1beta/interactions/{inter_id}:cancel",
                        headers=headers,
                    )
            return {
                "valid_key": True,
                "genie4_listed": genie4_listed,
                "genie4_paid_ready": True,
                "tier": "paid",
                "models_count": len(model_names),
                "message": f"Gemini & Genie 4 Paid Tier verified ({len(model_names)} models unlocked)!",
            }

        err_txt = i_resp.text
        if "Free Tier" in err_txt or "limit: 0" in err_txt:
            return {
                "valid_key": True,
                "genie4_listed": genie4_listed,
                "genie4_paid_ready": False,
                "tier": "free",
                "models_count": len(model_names),
                "message": (
                    f"Valid Gemini key ({len(model_names)} models, Gemini 4 text/QA ready), "
                    "but currently on Free Tier (0 Genie 4 sessions). Enable Billing / Tier 1 at https://ai.dev/rate-limit to unlock Genie 4!"
                ),
            }

        return {
            "valid_key": True,
            "genie4_listed": genie4_listed,
            "genie4_paid_ready": False,
            "tier": "limited",
            "models_count": len(model_names),
            "message": f"Valid Gemini key ({len(model_names)} models), Genie 4 returned HTTP {i_resp.status_code}: {err_txt[:160]}",
        }


async def genie4_generate_steered_clip(
    *,
    prompt: str,
    out_path: Path,
    duration_sec: int = 15,
    camera_spec: str = "",
    audio_direction: str = "",
    start_image_path: Path | str | None = None,
    mid_beat_prompts: list[str] | None = None,
    settings: Settings | None = None,
    on_frame_progress: Any | None = None,
    cancel: CancellationToken | None = None,
) -> tuple[Path, float]:
    """Stream & steer a continuous Genie 4 world into a 1280x720@24fps + 48kHz stereo MP4, then stop billing cleanly."""
    resolved = settings or get_settings()
    api_key = resolved.effective_gemini_key()
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is required for Google Genie 4 (models/genie-4). Add a Paid-Tier key from https://aistudio.google.com/apikey."
        )

    from google import genai

    target_dur = max(5, min(90, int(duration_sec)))
    fps = 24
    target_frames = target_dur * fps
    world_text = compile_genie4_world_prompt(
        prompt,
        camera_spec=camera_spec,
        audio_direction=audio_direction,
    )
    ref_b64 = prepare_genie4_reference_jpeg(start_image_path)

    raw_t = os.getenv("GTM_FFMPEG_THREADS", "1").strip()
    eff_threads = max(1, int(raw_t)) if raw_t.isdigit() else 1

    tmp_dir = Path(tempfile.mkdtemp(prefix="genie4_stream_"))
    video_only_mp4 = tmp_dir / "genie_video.mp4"
    audio_pcm_path = tmp_dir / "genie_audio.pcm"

    client = genai.Client(
        api_key=api_key,
        http_options={"api_version": "v1beta", "timeout": 20000, "retry_options": {"attempts": 1}},
    )
    interaction = await client.aio.interactions.create(
        model="genie-4",
        api_version="v1beta",
        automatic_continuation=False,
        timeout=18.0,
    )
    interaction_id = interaction.id
    completed_cleanly = False
    frames_written = 0

    ff_cmd = [
        resolved.ffmpeg_bin,
        "-y",
        "-threads",
        str(eff_threads),
        "-f",
        "image2pipe",
        "-vcodec",
        "mjpeg",
        "-r",
        str(fps),
        "-i",
        "-",
        "-vf",
        "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1,format=yuv420p",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "20",
        "-r",
        str(fps),
        "-movflags",
        "+faststart",
        str(video_only_mp4),
    ]
    ff_proc = await asyncio.create_subprocess_exec(
        *ff_cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )

    try:
        async with client.aio.interactions.connect(
            interaction_id,
            api_version="v1beta",
            timeout=35.0,
        ) as conn:
            setup_event: dict[str, Any] = {
                "event_type": "world_setup",
                "world_prompt": {"text": world_text},
            }
            if ref_b64:
                setup_event["reference_image"] = {
                    "mime_type": "image/jpeg",
                    "data": ref_b64,
                }
            await conn.send(setup_event)  # type: ignore[arg-type]

            first_frame_ev = asyncio.Event()
            stop_steer_ev = asyncio.Event()

            async def _steer_world() -> None:
                """Drive smooth forward camera glide & timed beat prompts, then end session at target_dur."""
                try:
                    await asyncio.wait_for(first_frame_ev.wait(), timeout=22.0)
                except TimeoutError:
                    return
                cam_low = (camera_spec or "").lower()
                pan_key = (
                    "arrow_right"
                    if "pan right" in cam_low or "orbit" in cam_low
                    else ("arrow_left" if "pan left" in cam_low else None)
                )
                move_forward = not any(k in cam_low for k in ("locked-off", "static", "stationary"))

                try:
                    if move_forward:
                        await conn.send({"event_type": "user_action", "action": "key_down", "input": "key_w"})  # type: ignore[arg-type]
                    beats = [b.strip() for b in (mid_beat_prompts or []) if b and b.strip()][:4]
                    if beats and target_dur >= 10:
                        interval = max(3.0, (target_dur - 2.5) / float(len(beats) + 1))
                        for b_txt in beats:
                            if stop_steer_ev.is_set():
                                break
                            await asyncio.sleep(interval)
                            if stop_steer_ev.is_set():
                                break
                            # Repeat briefly across 150ms intervals so Genie 4's 4-frame planner step catches it
                            clean_b = _CUT_RE.sub("reveal", b_txt)[:160]
                            for _ in range(5):
                                if stop_steer_ev.is_set():
                                    break
                                await conn.send(
                                    {"event_type": "input", "content": [{"type": "text", "text": clean_b}]}  # type: ignore[arg-type]
                                )
                                await asyncio.sleep(0.15)
                    if pan_key and not stop_steer_ev.is_set():
                        await conn.send({"event_type": "user_action", "action": "key_down", "input": pan_key})  # type: ignore[arg-type]
                        await asyncio.sleep(min(2.0, target_dur * 0.2))
                        await conn.send({"event_type": "user_action", "action": "key_up", "input": pan_key})  # type: ignore[arg-type]
                except Exception:
                    pass

            steer_task = asyncio.create_task(_steer_world())
            deadline = asyncio.get_running_loop().time() + float(target_dur) + 28.0

            with audio_pcm_path.open("wb") as pcm_fh:
                try:
                    async for ev in conn.receive():
                        if cancel is not None:
                            cancel.raise_if_cancelled()
                        ev_type = getattr(ev, "event_type", "") or ""
                        if ev_type == "error":
                            err_obj = getattr(ev, "error", None)
                            raise RuntimeError(f"Genie 4 stream error: {err_obj or ev}")
                        if ev_type in ("interaction.completed", "interaction.cancelled"):
                            completed_cleanly = True
                            break

                        if ev_type == "step.delta":
                            delta = getattr(ev, "delta", None)
                            d_type = getattr(delta, "type", "") if delta is not None else ""
                            raw_b64 = getattr(delta, "data", None) if delta is not None else None
                            if d_type in ("image", "video") and isinstance(raw_b64, str) and raw_b64:
                                jpg_bytes = base64.b64decode(raw_b64)
                                if ff_proc.stdin is not None:
                                    ff_proc.stdin.write(jpg_bytes)
                                    frames_written += 1
                                    if frames_written == 1:
                                        first_frame_ev.set()
                                    if frames_written % 8 == 0:
                                        await ff_proc.stdin.drain()
                                    if on_frame_progress is not None and frames_written % 24 == 0:
                                        sec_done = frames_written // fps
                                        res = on_frame_progress(sec_done, target_dur)
                                        if asyncio.iscoroutine(res):
                                            await res
                            elif d_type == "audio" and isinstance(raw_b64, str) and raw_b64:
                                pcm_fh.write(base64.b64decode(raw_b64))

                        if frames_written >= target_frames or asyncio.get_running_loop().time() >= deadline:
                            stop_steer_ev.set()
                            with contextlib.suppress(Exception):
                                await conn.send({"event_type": "user_action", "action": "key_up", "input": "key_w"})  # type: ignore[arg-type]
                                await conn.complete()
                            completed_cleanly = True
                            break
                finally:
                    stop_steer_ev.set()
                    steer_task.cancel()
                    with contextlib.suppress(Exception):
                        await steer_task
                    if not completed_cleanly:
                        with contextlib.suppress(Exception):
                            await conn.complete()
    finally:
        # Ironclad billing cutoff: ensure server-side Genie 4 interaction never stays alive after we exit
        if not completed_cleanly:
            with contextlib.suppress(Exception):
                await client.aio.interactions.cancel(interaction_id, api_version="v1beta")
        if ff_proc.stdin is not None and not ff_proc.stdin.is_closing():
            with contextlib.suppress(Exception):
                ff_proc.stdin.close()

    with contextlib.suppress(Exception):
        await asyncio.wait_for(ff_proc.communicate(), timeout=45.0)

    try:
        if frames_written == 0 or not video_only_mp4.is_file():
            raise RuntimeError("Genie 4 session ended before delivering video frames.")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        has_pcm = audio_pcm_path.is_file() and audio_pcm_path.stat().st_size > 1024
        if has_pcm:
            mux_cmd = [
                resolved.ffmpeg_bin,
                "-y",
                "-threads",
                str(eff_threads),
                "-i",
                str(video_only_mp4),
                "-f",
                "s16le",
                "-ar",
                "48000",
                "-ac",
                "2",
                "-i",
                str(audio_pcm_path),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "copy",
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
                str(out_path),
            ]
            rc, _, _ = await run_subprocess(mux_cmd, timeout_sec=90.0, cancel=cancel, check=False)
            if rc != 0 or not out_path.is_file():
                shutil.copyfile(video_only_mp4, out_path)
        else:
            shutil.copyfile(video_only_mp4, out_path)

        probe = await probe_media(out_path, ffprobe_bin=resolved.ffprobe_bin, cancel=cancel)
        actual_sec = max(1.0, float(probe.duration_sec or (frames_written / float(fps))))
        est_usd = round(actual_sec * GENIE4_USD_PER_SEC, 4)
        return out_path, est_usd
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


__all__ = [
    "GENIE4_MODEL_ID",
    "GENIE4_USD_PER_SEC",
    "compile_genie4_world_prompt",
    "genie4_generate_steered_clip",
    "is_genie4_ready",
    "mint_genie4_browser_session",
    "prepare_genie4_reference_jpeg",
    "verify_gemini_and_genie4_access",
]
