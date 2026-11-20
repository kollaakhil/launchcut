"""Real offline FFmpeg, ASS captions, GraphicsCardRenderer, and MediaQA tests (M3)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from gtm_videogen.core.job_registry import CancellationToken
from gtm_videogen.media._subprocess import probe_media, run_subprocess
from gtm_videogen.media.captions import (
    CaptionCue,
    _escape_ass_text,
    build_ass_filter,
    contrast_ratio,
    enforce_one_text_layer,
    ensure_wcag_contrast,
    escape_ffmpeg_filter_path,
    generate_ass_subtitles,
    resolve_font_path,
)
from gtm_videogen.media.ffmpeg_stitcher import ClipShortageError, FFmpegStitcher
from gtm_videogen.media.graphics_card import GraphicsCardRenderer
from gtm_videogen.media.media_qa import MediaQA, apply_score_caps
from gtm_videogen.models.schemas import (
    CreativeBible,
    GraphicsTheme,
    MusicCue,
    RenderedAudio,
    RenderedClip,
    SfxCue,
    Shot,
    ShotSheet,
    SoundPlan,
    VoiceCue,
)


async def _make_synth_clip(
    out_path: Path,
    *,
    width: int = 320,
    height: int = 180,
    fps: int = 24,
    duration: float = 2.0,
    with_audio: bool = True,
    freq: int = 440,
    vf_extra: str | None = None,
    silent_audio: bool = False,
) -> Path:
    """Create a real synthetic MP4 clip via FFmpeg lavfi."""
    vf = f"testsrc2=size={width}x{height}:rate={fps}:duration={duration}"
    cmd = ["ffmpeg", "-y", "-f", "lavfi", "-i", vf]
    if with_audio:
        audio_src = (
            "anullsrc=r=44100:cl=stereo"
            if silent_audio
            else f"sine=frequency={freq}:sample_rate=44100:duration={duration}"
        )
        cmd.extend(["-f", "lavfi", "-i", audio_src])
        cmd.extend(["-map", "0:v:0", "-map", "1:a:0"])
    else:
        cmd.extend(["-map", "0:v:0"])

    if vf_extra:
        cmd.extend(["-vf", vf_extra])

    cmd.extend(
        [
            "-t",
            f"{duration:.3f}",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-r",
            str(fps),
        ]
    )
    if with_audio:
        cmd.extend(["-c:a", "aac", "-ar", "44100", "-ac", "2", "-b:a", "128k"])
    cmd.append(str(out_path))
    await run_subprocess(cmd, timeout_sec=60.0)
    return out_path


async def _make_synth_wav(out_path: Path, *, duration: float = 1.5, freq: int = 520) -> Path:
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency={freq}:sample_rate=44100:duration={duration}",
        "-ar",
        "44100",
        "-ac",
        "2",
        "-c:a",
        "pcm_s16le",
        str(out_path),
    ]
    await run_subprocess(cmd, timeout_sec=30.0)
    return out_path


def _mini_sheet(
    *,
    fps: int = 24,
    aspect: str = "16:9",
    durs: tuple[float, ...] = (2.0, 2.0),
    on_screen_texts: tuple[str | None, ...] = (None, None),
    vo_lines: tuple[str | None, ...] = (None, None),
) -> ShotSheet:
    shots = []
    for i, dur in enumerate(durs):
        shots.append(
            Shot(
                id=f"s{i + 1}",
                index=i,
                beat=f"beat_{i + 1}",
                duration_sec=dur,
                engine="ai_video",
                visual_prompt=f"Synthetic scene {i + 1}",
                camera_move="static",
                on_screen_text=on_screen_texts[i] if i < len(on_screen_texts) else None,
                vo_line=vo_lines[i] if i < len(vo_lines) else None,
            )
        )
    return ShotSheet(
        fps=fps,  # type: ignore[arg-type]
        aspect=aspect,  # type: ignore[arg-type]
        target_duration_sec=sum(durs),
        audio_mode="both",
        theme=GraphicsTheme(font_family="Roboto"),
        shots=shots,
    )


# ---------------------------------------------------------------------------
# 1. Captions, WCAG Contrast, ONE_TEXT_LAYER, and Font Escaping (F12 / BUG-24)
# ---------------------------------------------------------------------------


def test_wcag_contrast_preserves_dark_brand_palette() -> None:
    fg, bg = ensure_wcag_contrast("#333333", "#101820", min_ratio=4.5)
    assert bg == "#101820"  # Dark brand palette preserved
    assert contrast_ratio(fg, bg) >= 4.5

    # Light background picks dark text
    fg2, bg2 = ensure_wcag_contrast("#EEEEEE", "#FAF3DD", min_ratio=4.5)
    assert bg2 == "#FAF3DD"
    assert fg2 == "#111111"
    assert contrast_ratio(fg2, bg2) >= 4.5


def test_one_text_layer_prevents_overlapping_captions_and_overlays() -> None:
    cues = [
        CaptionCue(start_sec=0.0, end_sec=4.0, text="Spoken caption", layer_kind="caption"),
        CaptionCue(start_sec=1.0, end_sec=2.5, text="Hero Headline", layer_kind="overlay"),
    ]
    resolved = enforce_one_text_layer(cues)
    # Verify no two resolved cues overlap in time
    for prev, curr in zip(resolved, resolved[1:], strict=False):
        assert prev.end_sec <= curr.start_sec + 1e-6
    assert any(c.layer_kind == "overlay" and c.text == "Hero Headline" for c in resolved)


def test_ffmpeg_ass_filter_escaping_and_bundled_fonts(tmp_path: Path) -> None:
    tricky = tmp_path / "dir:with:colons" / "sub's[1].ass"
    escaped = escape_ffmpeg_filter_path(tricky)
    assert "\\:" in escaped
    assert "\\'" in escaped
    assert "\\[" in escaped

    flt = build_ass_filter(tricky)
    assert flt.startswith("ass=")
    assert ":fontsdir=" in flt

    for fam in (
        "Roboto",
        "Roboto Condensed",
        "Liberation Serif",
        "Liberation Mono",
        "Liberation Sans",
    ):
        p = resolve_font_path(fam, bold=True)
        assert p.exists()
        assert p.suffix == ".ttf"

    # BUG-C3: literal backslash neutralized with zero-width space instead of \\
    escaped_text = _escape_ass_text(r"ALPHA {BETA}\NGAMMA")
    assert "\\\u200bN" in escaped_text
    assert "\\\\N" not in escaped_text
    assert "\\{BETA\\}" in escaped_text


@pytest.mark.asyncio
async def test_ffmpeg_ass_burn_live_with_colons_and_quotes_in_paths(
    tmp_path: Path, sample_creative_bible: CreativeBible
) -> None:
    """BUG-C1: Live FFmpeg ASS burn succeeds with colons, quotes, brackets, commas, semicolons."""
    tricky_ass_dir = tmp_path / "dir:colon [1] 'quote',comma;semi"
    tricky_ass_dir.mkdir(parents=True)
    ass_file = tricky_ass_dir / "cap:a'b [2],c;d.ass"

    tricky_fonts_dir = tmp_path / "fonts:dir [3] 'q',x;y"
    tricky_fonts_dir.mkdir(parents=True)

    sheet = _mini_sheet(fps=24, aspect="16:9", durs=(1.0,), on_screen_texts=("LIVE BURN OK",))
    generate_ass_subtitles(
        sheet=sheet,
        bible=sample_creative_bible,
        output_path=ass_file,
        width=320,
        height=180,
    )
    vf = build_ass_filter(ass_file, fonts_dir=tricky_fonts_dir)
    out_mp4 = tmp_path / "tricky_ass_out.mp4"
    await run_subprocess(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=#111827:s=320x180:r=24:d=1.0",
            "-vf",
            vf,
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            str(out_mp4),
        ],
        timeout_sec=30.0,
    )
    assert out_mp4.exists()
    probe = await probe_media(out_mp4)
    assert probe.has_video and probe.width == 320 and probe.height == 180


# ---------------------------------------------------------------------------
# 2. FFmpegStitcher: Parallel Normalize, 24 vs 30fps, tpad, BUG-10, Audio Mix
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stitcher_24fps_and_30fps_parity_with_audio_mix(
    tmp_path: Path, sample_creative_bible: CreativeBible
) -> None:
    stitcher = FFmpegStitcher()

    for target_fps in (24, 30):
        c1_path = await _make_synth_clip(tmp_path / f"c1_{target_fps}.mp4", fps=30, duration=2.0)
        # Video-only clip (no audio) to verify anullsrc + apad normalization
        c2_path = await _make_synth_clip(
            tmp_path / f"c2_{target_fps}.mp4", fps=24, duration=2.0, with_audio=False
        )
        vo_wav = await _make_synth_wav(tmp_path / f"vo_{target_fps}.wav", duration=1.8, freq=480)
        sfx_wav = await _make_synth_wav(tmp_path / f"sfx_{target_fps}.wav", duration=0.4, freq=880)
        mus_wav = await _make_synth_wav(tmp_path / f"mus_{target_fps}.wav", duration=2.0, freq=220)

        sheet = _mini_sheet(fps=target_fps, aspect="16:9", durs=(2.0, 2.0))
        clips = [
            RenderedClip(
                shot_id="s1", path=str(c1_path), duration_sec=2.0, fps=30, width=320, height=180
            ),
            RenderedClip(
                shot_id="s2", path=str(c2_path), duration_sec=2.0, fps=24, width=320, height=180
            ),
        ]
        sound_plan = SoundPlan(
            vo_cues=[VoiceCue(shot_id="s1", at_sec=0.1, duration_sec=1.5, text="Hello world")],
            sfx_cues=[
                SfxCue(shot_id="s2", at_sec=2.0, duration_sec=0.4, prompt="chime", gain_db=-8.0)
            ],
            music_cue=MusicCue(prompt="warm bed", duration_sec=4.0, gain_db=-16.0),
        )
        audio_assets = [
            RenderedAudio(kind="vo", cue_id="s1", path=str(vo_wav), duration_sec=1.8),
            RenderedAudio(kind="sfx", cue_id="s2", path=str(sfx_wav), duration_sec=0.4),
            RenderedAudio(kind="music", cue_id="m1", path=str(mus_wav), duration_sec=2.0),
        ]

        out_mp4 = tmp_path / f"stitched_{target_fps}.mp4"
        manifest = await stitcher.stitch(
            clips=clips,
            sound_plan=sound_plan,
            audio_assets=audio_assets,
            sheet=sheet,
            bible=sample_creative_bible,
            output_path=out_mp4,
            width=480,
            height=270,
        )

        assert manifest.error is None
        assert out_mp4.exists()
        probe = await probe_media(out_mp4)
        assert probe.has_video and probe.has_audio
        assert probe.width == 480 and probe.height == 270
        assert abs(probe.fps - target_fps) < 0.2
        assert abs(probe.duration_sec - 4.0) < 0.2


@pytest.mark.asyncio
async def test_stitcher_tpad_short_clip_and_shortage_error(
    tmp_path: Path, sample_creative_bible: CreativeBible
) -> None:
    stitcher = FFmpegStitcher()

    # Shortage = 0.5s (0.12 < 0.5 <= 1.0) -> succeeds via tpad
    short_ok = await _make_synth_clip(tmp_path / "short_ok.mp4", fps=24, duration=1.5)
    sheet_ok = _mini_sheet(fps=24, durs=(2.0,))
    out_ok = tmp_path / "out_tpad.mp4"
    manifest = await stitcher.stitch(
        clips=[
            RenderedClip(
                shot_id="s1", path=str(short_ok), duration_sec=1.5, fps=24, width=320, height=180
            )
        ],
        sound_plan=None,
        audio_assets=[],
        sheet=sheet_ok,
        bible=sample_creative_bible,
        output_path=out_ok,
        width=320,
        height=180,
    )
    assert out_ok.exists()
    assert any("tpad" in w for w in manifest.warnings)
    probe = await probe_media(out_ok)
    assert abs(probe.duration_sec - 2.0) < 0.2

    # Shortage = 1.5s (> 1.0s) -> raises ClipShortageError
    too_short = await _make_synth_clip(tmp_path / "too_short.mp4", fps=24, duration=1.0)
    sheet_bad = _mini_sheet(fps=24, durs=(2.5,))
    with pytest.raises(ClipShortageError):
        await stitcher.stitch(
            clips=[
                RenderedClip(
                    shot_id="s1",
                    path=str(too_short),
                    duration_sec=1.0,
                    fps=24,
                    width=320,
                    height=180,
                )
            ],
            sound_plan=None,
            audio_assets=[],
            sheet=sheet_bad,
            bible=sample_creative_bible,
            output_path=tmp_path / "out_fail.mp4",
            width=320,
            height=180,
        )


@pytest.mark.asyncio
async def test_stitcher_bug10_empty_audio_guard(
    tmp_path: Path, sample_creative_bible: CreativeBible
) -> None:
    """No VO, no music, and all SFX filtered out by silence_gaps/cutoff -> valid MP4, no crash."""
    stitcher = FFmpegStitcher()
    c1 = await _make_synth_clip(tmp_path / "c1_empty.mp4", fps=24, duration=2.0)
    sfx = await _make_synth_wav(tmp_path / "sfx_filtered.wav", duration=0.3)

    sheet = _mini_sheet(fps=24, durs=(2.0,))
    sound_plan = SoundPlan(
        vo_cues=[],
        sfx_cues=[SfxCue(shot_id="s1", at_sec=0.5, duration_sec=0.3, prompt="whoosh")],
        music_cue=None,
        silence_gaps=[(0.0, 1.0)],  # Filters out the only SFX cue!
    )
    out_mp4 = tmp_path / "bug10_guard.mp4"
    manifest = await stitcher.stitch(
        clips=[
            RenderedClip(
                shot_id="s1", path=str(c1), duration_sec=2.0, fps=24, width=320, height=180
            )
        ],
        sound_plan=sound_plan,
        audio_assets=[RenderedAudio(kind="sfx", cue_id="s1", path=str(sfx), duration_sec=0.3)],
        sheet=sheet,
        bible=sample_creative_bible,
        output_path=out_mp4,
        width=320,
        height=180,
    )
    assert manifest.error is None
    assert out_mp4.exists()
    assert any("silence gap" in w for w in manifest.warnings)
    probe = await probe_media(out_mp4)
    assert probe.has_video
    assert abs(probe.duration_sec - 2.0) < 0.2


@pytest.mark.asyncio
async def test_stitcher_cancellation_terminates_ffmpeg(
    tmp_path: Path, sample_creative_bible: CreativeBible
) -> None:
    """BUG-05: CancellationToken cleanly cancels an in-flight FFmpeg stitch."""
    stitcher = FFmpegStitcher()
    c1 = await _make_synth_clip(tmp_path / "c_cancel.mp4", fps=24, duration=2.0)
    sheet = _mini_sheet(fps=24, durs=(2.0,))
    token = CancellationToken()
    token.cancel("User aborted stitch")

    with pytest.raises(asyncio.CancelledError):
        await stitcher.stitch(
            clips=[
                RenderedClip(
                    shot_id="s1", path=str(c1), duration_sec=2.0, fps=24, width=320, height=180
                )
            ],
            sound_plan=None,
            audio_assets=[],
            sheet=sheet,
            bible=sample_creative_bible,
            output_path=tmp_path / "cancelled.mp4",
            cancel=token,
            width=320,
            height=180,
        )


# ---------------------------------------------------------------------------
# 3. Single-Pass ASS Caption Burn + Tesseract OCR Round-Trip (F12 / F13 / BUG-24)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ass_caption_burn_and_tesseract_ocr_roundtrip(
    tmp_path: Path, sample_creative_bible: CreativeBible
) -> None:
    qa = MediaQA()
    if not qa.is_tesseract_available():
        pytest.skip("tesseract binary not installed")

    stitcher = FFmpegStitcher()
    # Solid dark background clip so burned ASS caption has crystal-clear contrast for OCR
    bg_path = tmp_path / "dark_bg.mp4"
    await run_subprocess(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=#111827:s=640x360:r=24:d=2.0",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=44100:duration=2.0",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(bg_path),
        ],
        timeout_sec=30.0,
    )

    sheet = _mini_sheet(
        fps=24,
        aspect="16:9",
        durs=(2.0,),
        on_screen_texts=("FRESH GINGER TONIC",),
    )
    ass_file = generate_ass_subtitles(
        sheet=sheet,
        bible=sample_creative_bible,
        output_path=tmp_path / "captions.ass",
        width=640,
        height=360,
    )
    assert ass_file.exists()
    assert "ScriptType: v4.00+" in ass_file.read_text(encoding="utf-8")

    out_mp4 = tmp_path / "captioned.mp4"
    await stitcher.stitch(
        clips=[
            RenderedClip(
                shot_id="s1", path=str(bg_path), duration_sec=2.0, fps=24, width=640, height=360
            )
        ],
        sound_plan=None,
        audio_assets=[],
        sheet=sheet,
        bible=sample_creative_bible,
        output_path=out_mp4,
        width=640,
        height=360,
        ass_path=ass_file,
    )

    checks = await qa.check_ocr(out_mp4, sheet)
    assert len(checks) == 1
    assert checks[0].passed, (
        f"OCR failed: observed={checks[0].observed_text!r} sim={checks[0].similarity}"
    )


# ---------------------------------------------------------------------------
# 4. MediaQA: Clean vs Defective (Black, Freeze, Silent Loudness) & Frames
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_media_qa_clean_vs_defective_videos_and_frames(tmp_path: Path) -> None:
    qa = MediaQA()

    # A. Clean video with -14 LUFS audio
    clean_raw = await _make_synth_clip(
        tmp_path / "clean_raw.mp4", width=480, height=270, fps=24, duration=2.0
    )
    clean_mp4 = tmp_path / "clean_loudnorm.mp4"
    await run_subprocess(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(clean_raw),
            "-c:v",
            "copy",
            "-af",
            "loudnorm=I=-14:LRA=7:TP=-1.5",
            "-c:a",
            "aac",
            "-ar",
            "44100",
            str(clean_mp4),
        ],
        timeout_sec=30.0,
    )

    sheet = _mini_sheet(fps=24, aspect="16:9", durs=(2.0,))
    clean_report = await qa.evaluate(clean_mp4, sheet, run_ocr=False)
    assert clean_report.passed, f"Expected clean video to pass: {clean_report.blocking_errors}"
    assert clean_report.loudness_lufs is not None
    assert -20.0 <= clean_report.loudness_lufs <= -9.0

    # Poster & evidence frames
    poster = await qa.poster_frame(clean_mp4, tmp_path / "poster.jpg", at_sec=0.5)
    assert poster.exists() and poster.stat().st_size > 500
    ev_frames = await qa.evidence_frames(clean_mp4, tmp_path / "evidence", detect_cuts=False)
    assert len(ev_frames) >= 4
    assert all(p.exists() and p.stat().st_size > 500 for p in ev_frames)

    # B. Intentionally defective video with black frames
    black_mp4 = await _make_synth_clip(
        tmp_path / "black_defect.mp4",
        width=480,
        height=270,
        fps=24,
        duration=2.0,
        vf_extra="drawbox=color=black:t=fill:enable='between(t,0.6,1.1)'",
    )
    black_report = await qa.evaluate(black_mp4, sheet, run_ocr=False)
    assert not black_report.passed
    assert any(err.code == "BLACK_FRAMES" for err in black_report.blocking_errors)
    assert "s1" in black_report.failed_shot_ids

    # C. Intentionally defective video with frozen frames + silent audio (loudness cap 7)
    freeze_mp4 = await _make_synth_clip(
        tmp_path / "freeze_defect.mp4",
        width=480,
        height=270,
        fps=24,
        duration=2.0,
        vf_extra="trim=duration=0.5,tpad=stop_mode=clone:stop_duration=1.5",
        silent_audio=True,
    )
    freeze_report = await qa.evaluate(freeze_mp4, sheet, run_ocr=False)
    assert not freeze_report.passed
    assert any(err.code == "FROZEN_FRAMES" for err in freeze_report.blocking_errors)
    assert "loudness_cap_7" in freeze_report.score_caps_applied

    # Score cap helper verification
    assert apply_score_caps(9.5, ["loudness_cap_7"]) == 7.0
    assert apply_score_caps(9.5, ["loudness_cap_7", "ocr_fail_cap_6"]) == 6.0


# ---------------------------------------------------------------------------
# 5. GraphicsCardRenderer: Exact width/height/fps/duration_sec (BUG-08/09)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_graphics_card_renderer_exact_dimensions_fps_duration_and_ocr(
    tmp_path: Path, sample_creative_bible: CreativeBible
) -> None:
    renderer = GraphicsCardRenderer()

    # Landscape 24fps 1.5s card
    card1_path = tmp_path / "title_card.mp4"
    clip1 = await renderer.render_card(
        text="SOLARA BOTANICS",
        subtitle="Cold-Pressed Ginger",
        footer="solarabotanics.example.com",
        width=640,
        height=360,
        fps=24,
        duration_sec=1.5,
        bible=sample_creative_bible,
        output_path=card1_path,
        card_kind="title",
        shot_id="card_1",
    )
    probe1 = await probe_media(card1_path)
    assert clip1.width == 640 and clip1.height == 360
    assert probe1.width == 640 and probe1.height == 360
    assert abs(probe1.fps - 24.0) < 0.2
    assert abs(probe1.duration_sec - 1.5) < 0.15

    # Portrait 30fps 2.0s lower-third card
    card2_path = tmp_path / "lower_third.mp4"
    clip2 = await renderer.render_card(
        text="4G ORGANIC SUGAR",
        width=480,
        height=854,
        fps=30,
        duration_sec=2.0,
        bible=sample_creative_bible,
        output_path=card2_path,
        card_kind="lower_third",
        shot_id="card_2",
    )
    probe2 = await probe_media(card2_path)
    assert clip2.width == 480 and clip2.height == 854
    assert probe2.width == 480 and probe2.height == 854
    assert abs(probe2.fps - 30.0) < 0.2
    assert abs(probe2.duration_sec - 2.0) < 0.15

    # OCR verification on rendered card
    qa = MediaQA()
    if qa.is_tesseract_available():
        card_sheet = _mini_sheet(
            fps=24, aspect="16:9", durs=(1.5,), on_screen_texts=("SOLARA BOTANICS",)
        )
        ocr_res = await qa.check_ocr(card1_path, card_sheet)
        assert len(ocr_res) == 1
        assert ocr_res[0].passed, f"Card OCR failed: {ocr_res[0]}"


@pytest.mark.asyncio
@pytest.mark.parametrize("dur", [1.0, 0.5, 0.08])
async def test_media_qa_evidence_frames_short_clips_all_exist(
    tmp_path: Path, dur: float
) -> None:
    """BUG-C2: evidence_frames() must never return non-existent PNG paths on <=1.0s 24fps clips."""
    qa = MediaQA()
    clip = await _make_synth_clip(
        tmp_path / f"short_{dur}s.mp4",
        width=320,
        height=180,
        fps=24,
        duration=dur,
        with_audio=True,
    )
    frames = await qa.evidence_frames(clip, tmp_path / f"ev_{dur}", detect_cuts=True)
    assert len(frames) >= 1
    for p in frames:
        assert p.exists(), f"Missing evidence frame on {dur}s clip: {p}"
        assert p.stat().st_size > 0, f"Empty evidence frame on {dur}s clip: {p}"


@pytest.mark.asyncio
async def test_stitcher_handles_cross_device_exdev_without_copystat_permission_error(
    tmp_path: Path,
    sample_creative_bible: CreativeBible,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """M6 Finding 2: stitch() succeeds on cross-device EXDEV + copystat PermissionError."""
    import errno
    import os
    import shutil

    stitcher = FFmpegStitcher()
    c1 = await _make_synth_clip(tmp_path / "c1_exdev.mp4", fps=24, duration=1.5)
    sheet = _mini_sheet(fps=24, durs=(1.5,))
    out_dir = tmp_path / "other_mount"
    out_dir.mkdir()
    out_mp4 = out_dir / "stitched_exdev.mp4"

    real_replace = os.replace
    exdev_triggered = False

    def fake_rename(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        raise OSError(errno.EXDEV, "Invalid cross-device link", str(src), str(dst))

    def fake_replace(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        nonlocal exdev_triggered
        if Path(src).resolve().parent != Path(dst).resolve().parent:
            exdev_triggered = True
            raise OSError(errno.EXDEV, "Invalid cross-device link", str(src), str(dst))
        real_replace(src, dst)

    def fake_copystat(*_args: object, **_kwargs: object) -> None:
        raise PermissionError(errno.EPERM, "Operation not permitted")

    monkeypatch.setattr(os, "rename", fake_rename)
    monkeypatch.setattr(os, "replace", fake_replace)
    monkeypatch.setattr(shutil, "copystat", fake_copystat)

    manifest = await stitcher.stitch(
        clips=[
            RenderedClip(
                shot_id="s1", path=str(c1), duration_sec=1.5, fps=24, width=320, height=180
            )
        ],
        sound_plan=None,
        audio_assets=[],
        sheet=sheet,
        bible=sample_creative_bible,
        output_path=out_mp4,
        width=320,
        height=180,
    )

    assert exdev_triggered, "Expected cross-device EXDEV fallback path to be exercised"
    assert manifest.error is None
    assert out_mp4.is_file() and out_mp4.stat().st_size > 0
    assert list(out_dir.glob("*.tmp")) == []
    probe = await probe_media(out_mp4)
    assert probe.has_video
    assert abs(probe.duration_sec - 1.5) < 0.2


@pytest.mark.asyncio
async def test_graphics_card_renderer_handles_cross_device_exdev_without_copystat_permission_error(
    tmp_path: Path,
    sample_creative_bible: CreativeBible,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """M6 Iter 2 Challenge 2: render_card() succeeds on cross-device EXDEV + copystat EPERM."""
    import errno
    import os
    import shutil

    renderer = GraphicsCardRenderer()
    out_dir = tmp_path / "other_mount_card"
    out_dir.mkdir()
    out_mp4 = out_dir / "card_exdev.mp4"

    real_replace = os.replace
    exdev_triggered = False

    def fake_rename(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        raise OSError(errno.EXDEV, "Invalid cross-device link", str(src), str(dst))

    def fake_replace(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        nonlocal exdev_triggered
        if Path(src).resolve().parent != Path(dst).resolve().parent:
            exdev_triggered = True
            raise OSError(errno.EXDEV, "Invalid cross-device link", str(src), str(dst))
        real_replace(src, dst)

    def fake_copystat(*_args: object, **_kwargs: object) -> None:
        raise PermissionError(errno.EPERM, "Operation not permitted")

    monkeypatch.setattr(os, "rename", fake_rename)
    monkeypatch.setattr(os, "replace", fake_replace)
    monkeypatch.setattr(shutil, "copystat", fake_copystat)

    clip = await renderer.render_card(
        text="SOLARA EXDEV",
        subtitle="Cross-Mount Atomic Move",
        width=640,
        height=360,
        fps=24,
        duration_sec=1.5,
        bible=sample_creative_bible,
        output_path=out_mp4,
        card_kind="title",
        shot_id="card_exdev",
    )

    assert exdev_triggered, "Expected cross-device EXDEV fallback path to be exercised"
    assert clip.path == str(out_mp4)
    assert out_mp4.is_file() and out_mp4.stat().st_size > 0
    assert list(out_dir.glob("*.tmp")) == []
    probe = await probe_media(out_mp4)
    assert probe.has_video
    assert probe.width == 640 and probe.height == 360
    assert abs(probe.duration_sec - 1.5) < 0.2


