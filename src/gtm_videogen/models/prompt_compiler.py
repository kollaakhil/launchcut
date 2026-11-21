"""Modality-specific prompt compilers turning detailed Shot specs into exact prompts.

Image models (Flux Schnell / Flux Kontext), video models (Kling v4 / Google Veo 3),
and audio models (Mirelo / ElevenLabs SFX / ElevenLabs v3 TTS) respond best to
distinct, non-overlapping instructions:
  - Image models want static photographic geometry (composition, lens mm, lighting rig,
    subject materials, environment, color grade) with zero temporal motion verbs.
  - Video models want temporal choreography (camera move trajectory, motion curve,
    subject micro-motion & physics from start to end frame, lighting continuity).
  - Audio/SFX models want acoustic texture, material interaction, room tone, and timing.
"""

from __future__ import annotations

from gtm_videogen.models.schemas import CreativeBible, Shot


def compile_image_prompt(
    shot: Shot,
    bible: CreativeBible | None = None,
    *,
    kontext_anchor: bool = False,
) -> str:
    """Compile a hyper-precise photographic prompt for Flux Schnell / Flux Kontext."""
    base = (shot.still_prompt or "").strip() or shot.visual_prompt.strip()
    parts: list[str] = []

    if kontext_anchor:
        parts.append(
            "Preserve the exact lighting temperature, studio environment, material finish, "
            "and lens character from the reference frame while staging this next shot:"
        )

    parts.append(base.rstrip("."))

    if shot.subject_and_materials and shot.subject_and_materials.strip():
        parts.append(f"Subject & materials: {shot.subject_and_materials.strip().rstrip('.')}")

    if shot.environment and shot.environment.strip():
        parts.append(f"Environment: {shot.environment.strip().rstrip('.')}")
    elif bible and bible.visual_language.world:
        parts.append(f"Environment: {bible.visual_language.world.strip().rstrip('.')}")

    cam_bits: list[str] = []
    if shot.camera_angle and shot.camera_angle.strip():
        cam_bits.append(f"{shot.camera_angle.strip()} framing")
    if shot.lens_mm:
        cam_bits.append(f"{shot.lens_mm}mm cinema prime lens")
    if cam_bits:
        parts.append("Camera: " + ", ".join(cam_bits))

    if shot.lighting_rig and shot.lighting_rig.strip():
        parts.append(f"Lighting rig: {shot.lighting_rig.strip().rstrip('.')}")
    elif bible and bible.visual_language.lighting_style:
        parts.append(f"Lighting: {bible.visual_language.lighting_style.strip().rstrip('.')}")

    if shot.color_grade and shot.color_grade.strip():
        parts.append(f"Color grade: {shot.color_grade.strip().rstrip('.')}")
    elif bible and bible.visual_language.color_palette:
        palette_str = ", ".join(bible.visual_language.color_palette[:4])
        parts.append(f"Color palette: {palette_str}")

    parts.append(
        "Ultra-sharp commercial still photography, clean depth of field, "
        "zero garbled text, watermarks, or artifacts"
    )
    return ". ".join(p for p in parts if p).strip() + "."


def compile_video_prompt(
    shot: Shot,
    bible: CreativeBible | None = None,
) -> str:
    """Compile a physics- and camera-explicit motion prompt for Kling v4 & Google Veo 3."""
    base = (shot.motion_prompt or "").strip()
    visual = shot.visual_prompt.strip()
    parts: list[str] = []

    if base:
        parts.append(base.rstrip("."))
        # Include concise visual scene anchor if not already covered by motion_prompt
        if visual and visual[:40].lower() not in base.lower():
            parts.append(f"Scene: {visual.rstrip('.')}")
    else:
        parts.append(visual.rstrip("."))

    if shot.subject_and_materials and shot.subject_and_materials.strip():
        parts.append(f"Subject fidelity: {shot.subject_and_materials.strip().rstrip('.')}")

    cam_bits: list[str] = []
    if shot.camera_move and shot.camera_move.strip():
        cam_bits.append(shot.camera_move.strip())
    if shot.camera_angle and shot.camera_angle.strip():
        cam_bits.append(f"{shot.camera_angle.strip()} angle")
    if shot.lens_mm:
        cam_bits.append(f"{shot.lens_mm}mm lens")
    if shot.motion_curve and shot.motion_curve.strip():
        cam_bits.append(f"pacing: {shot.motion_curve.strip()}")
    if cam_bits:
        parts.append("Camera choreography: " + ", ".join(cam_bits))

    if shot.lighting_rig and shot.lighting_rig.strip():
        parts.append(f"Lighting: {shot.lighting_rig.strip().rstrip('.')}")
    elif bible and bible.visual_language.lighting_style:
        parts.append(f"Lighting: {bible.visual_language.lighting_style.strip().rstrip('.')}")

    if shot.color_grade and shot.color_grade.strip():
        parts.append(f"Grade: {shot.color_grade.strip().rstrip('.')}")

    if shot.environment and shot.environment.strip():
        parts.append(f"Setting: {shot.environment.strip().rstrip('.')}")

    parts.append("Continuous natural physical motion from frame 1, stable geometry, no morphing")
    return ". ".join(p for p in parts if p).strip() + "."


def compile_sfx_prompt(shot: Shot) -> str | None:
    """Compile a clean acoustic foley prompt from shot.sfx_note + shot.environment."""
    note = (shot.sfx_note or "").strip()
    if not note:
        return None
    env = (shot.environment or "").strip()
    if env and env[:24].lower() not in note.lower():
        return f"{note.rstrip('.')}, natural acoustic space matching {env.rstrip('.')}, clean studio mix."
    return f"{note.rstrip('.')}, crisp high-fidelity studio foley, no music."


__all__ = [
    "compile_image_prompt",
    "compile_sfx_prompt",
    "compile_video_prompt",
]
