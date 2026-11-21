"""Audio-Visual Sync & Beat Detection Engine (zero heavy ML/librosa dependencies).

Provides:
- `detect_audio_beats`: pure-Python + FFmpeg 16kHz PCM RMS energy-flux onset detector
  returning beat timestamps (`±25ms`) for beat-locked cutting & transition SFX placement.
- `snap_to_nearest_beat`: snaps a cut or SFX cue onto the nearest musical transient.
- `build_transition_sfx_cues`: enriches sparse SoundPlans with subtle bespoke transition
  risers and tactile clicks timed to exact shot boundaries / musical beats.
"""

from __future__ import annotations

import asyncio
import math
import shutil
import struct
from pathlib import Path

from gtm_videogen.core.job_registry import CancellationToken
from gtm_videogen.models.schemas import SfxCue, ShotSheet


async def detect_audio_beats(
    audio_path: Path | str,
    *,
    ffmpeg_bin: str | None = None,
    min_gap_sec: float = 0.35,
    cancel: CancellationToken | None = None,
) -> list[float]:
    """Detect rhythmic beat/onset timestamps (seconds) via FFmpeg 16kHz mono PCM flux."""
    if cancel is not None:
        cancel.raise_if_cancelled()
    path = Path(audio_path)
    if not path.is_file():
        return []
    ff = ffmpeg_bin or shutil.which("ffmpeg") or "ffmpeg"

    proc = await asyncio.create_subprocess_exec(
        ff,
        "-v",
        "error",
        "-i",
        str(path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "s16le",
        "pipe:1",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    pcm, _ = await proc.communicate()
    if proc.returncode != 0 or len(pcm) < 6400:
        return []

    sample_rate = 16000
    hop_samples = 400  # 25ms windows
    total_samples = len(pcm) // 2
    energies: list[float] = []
    for off in range(0, total_samples - hop_samples, hop_samples):
        chunk = struct.unpack_from(f"<{hop_samples}h", pcm, off * 2)
        rms = math.sqrt(sum(s * s for s in chunk) / hop_samples)
        energies.append(rms)

    if len(energies) < 8:
        return []

    # Positive spectral/energy flux (attack transients)
    flux = [max(0.0, energies[i] - energies[i - 1]) for i in range(1, len(energies))]
    mean_flux = sum(flux) / len(flux)
    var_flux = sum((x - mean_flux) ** 2 for x in flux) / len(flux)
    threshold = mean_flux + 1.15 * math.sqrt(var_flux)
    if threshold <= 10.0:
        return []

    beats: list[float] = []
    last_t = -min_gap_sec
    for i in range(1, len(flux) - 1):
        val = flux[i]
        if val >= threshold and val >= flux[i - 1] and val >= flux[i + 1]:
            t_sec = round((i + 1) * (hop_samples / sample_rate), 3)
            if t_sec - last_t >= min_gap_sec:
                beats.append(t_sec)
                last_t = t_sec
    return beats


def snap_to_nearest_beat(
    t_sec: float,
    beats: list[float],
    *,
    max_shift_sec: float = 0.22,
) -> float:
    """Snap `t_sec` to the nearest beat timestamp within `max_shift_sec`."""
    if not beats:
        return round(max(0.0, t_sec), 3)
    best = min(beats, key=lambda b: abs(b - t_sec))
    if abs(best - t_sec) <= max_shift_sec:
        return round(max(0.0, best), 3)
    return round(max(0.0, t_sec), 3)


def build_transition_sfx_cues(
    sheet: ShotSheet,
    *,
    beats: list[float] | None = None,
    existing_sfx: list[SfxCue] | None = None,
) -> list[SfxCue]:
    """Return existing SFX cues enriched with subtle transition risers/clicks on shot cuts."""
    cues = list(existing_sfx or [])
    if len(sheet.shots) < 2:
        return cues

    ordered = sorted(sheet.shots, key=lambda s: s.index)
    cursor = 0.0
    beat_list = beats or []
    for idx, shot in enumerate(ordered[:-1]):
        cursor += float(shot.duration_sec)
        next_shot = ordered[idx + 1]
        # Skip if an SFX cue already sits within 0.5s of this transition
        if any(abs(c.at_sec - cursor) < 0.5 for c in cues):
            continue
        is_hero_next = idx == 0 or next_shot.product_hero
        at_t = snap_to_nearest_beat(max(0.15, cursor - (0.35 if is_hero_next else 0.12)), beat_list)
        if at_t >= sheet.target_duration_sec - 0.35:
            continue
        if is_hero_next:
            prompt = (
                "Subtle warm cinematic sub-bass riser swell transitioning cleanly into a reveal"
            )
            dur = 0.8
            gain = -9.5
        elif next_shot.engine == "motion_card":
            prompt = "Crisp tactile minimalist studio click and soft air shimmer"
            dur = 0.5
            gain = -10.5
        else:
            prompt = "Gentle soft air whoosh transition, clean editorial cinema mix"
            dur = 0.6
            gain = -11.0
        cues.append(
            SfxCue(
                at_sec=at_t,
                duration_sec=dur,
                prompt=prompt,
                gain_db=gain,
                shot_id=next_shot.id,
            )
        )
    return cues
