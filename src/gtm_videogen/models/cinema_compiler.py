"""Deterministic Entity Hydration, Modality Slice Compilers, and Cinema Script Gate.

Slices the 3-layer Master JSON (`WorldVault` -> `CinemaScene` -> `CinemaShot`) into
exact, non-overlapping instructions for:
  1. 3-Panel Character/Prop Turnarounds & Dual-Wall A/B Location Set Plates
  2. Static Start Keyframe Stills (Composite Multi-Ref conditioned, zero motion verbs)
  3. Kontext Delta End Keyframe Stills (locked to Start Keyframe identity & room)
  4. Video Models (`structured_json` for Kling 3.0 + `@Elements`; `cinematic_prose` for Veo/Seedance/Hailuo)
  5. Cinema Script Gate & Story Rubric evaluation
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

from PIL import Image

from gtm_videogen.models.cinema_project import (
    CharacterAsset,
    CinemaProject,
    CinemaScene,
    CinemaShot,
    LocationAsset,
    PropAsset,
    StoryRubricReport,
    WorldVault,
)
from gtm_videogen.models.model_registry import VideoModelSpec

_TEXT_IN_AI_SHOT_RE = re.compile(
    r"\b(neon sign reading|billboard reading|sign saying|text reading|words on screen|readable text|logo text|subtitle reading)\b",
    re.IGNORECASE,
)

_CHEESY_CLICHE_RE = re.compile(
    r"\b(in a world where|redefine the future|synergy|game[- ]changer|unlock your potential|next[- ]generation solution|revolutionize)\b",
    re.IGNORECASE,
)


def resolve_shot_prop_states(
    shot: CinemaShot, scene: CinemaScene, vault: WorldVault
) -> dict[str, str]:
    """Merge scene entry prop states -> shot start/prop overrides into active prop state map."""
    merged: dict[str, str] = {}
    for pr in vault.props:
        merged[pr.id] = pr.default_state
    merged.update(scene.scene_entry_state.prop_states)
    merged.update(shot.start_frame.prop_states)
    for inst in shot.props_in_shot:
        if inst.prop_id and inst.state:
            merged[inst.prop_id] = inst.state
    return merged


def hydrate_entities_in_text(
    raw_text: str,
    vault: WorldVault,
    prop_states: dict[str, str] | None = None,
    *,
    element_tag_map: dict[str, str] | None = None,
    concise: bool = False,
) -> str:
    """Replace `CHAR_*`, `PROP_*`, and `LOC_*` IDs inline with locked physical descriptions."""
    if not raw_text:
        return ""
    out = raw_text
    states = prop_states or {}
    tags = element_tag_map or {}

    # Sort longest IDs first so prefix collisions never happen
    for ch in sorted(vault.characters, key=lambda c: len(c.id), reverse=True):
        if ch.id in out:
            tag_suffix = f" ({tags[ch.id]})" if ch.id in tags else ""
            if concise:
                desc = f"{ch.name}{tag_suffix} ({ch.identity_lock.split(',')[0].strip()}, wearing {ch.wardrobe_lock.split(',')[0].strip()})"
            else:
                desc = f"{ch.name}{tag_suffix} ({ch.identity_lock}; wearing {ch.wardrobe_lock})"
            out = out.replace(ch.id, desc)

    for pr in sorted(vault.props, key=lambda p: len(p.id), reverse=True):
        if pr.id in out:
            tag_suffix = f" ({tags[pr.id]})" if pr.id in tags else ""
            st_desc = pr.describe_state(states.get(pr.id))
            if concise:
                desc = f"{pr.name}{tag_suffix} ({st_desc})"
            else:
                desc = f"{pr.name}{tag_suffix} ({pr.geometry_and_scale}, {pr.surface_textures}; state: {st_desc})"
            out = out.replace(pr.id, desc)

    for loc in sorted(vault.locations, key=lambda loc_item: len(loc_item.id), reverse=True):
        if loc.id in out:
            desc = f"{loc.name} ({loc.time_of_day_and_weather})"
            out = out.replace(loc.id, desc)

    return re.sub(r"\s+", " ", out).strip()


# ---------------------------------------------------------------------------
# Vault Turnaround Sheets & Dual-Wall Location Set Plate Compilers
# ---------------------------------------------------------------------------


def compile_character_turnaround_prompt(char: CharacterAsset, vault: WorldVault) -> str:
    """Compile a 16:9 3-panel character turnaround sheet prompt for consistent crops."""
    craft = vault.taste_signature.compiled_craft_rules
    is_anime = "cinema_anime" in vault.taste_signature.mood_presets
    medium_style = (
        "Theatrical hand-painted anime character model sheet"
        if is_anime
        else f"Photorealistic 35mm cinema costume & casting turnaround sheet ({craft.film_texture})"
    )
    return (
        f"{medium_style}, clean three-panel horizontal triptych on neutral warm-slate studio backdrop, "
        f"showing the EXACT same person in three views: "
        f"[Left third: medium close-up frontal portrait looking at camera] "
        f"[Center third: waist-up three-quarter angle portrait] "
        f"[Right third: full-body standing posture head-to-toe]. "
        f"Character identity: {char.identity_lock}. "
        f"Exact wardrobe across all three panels: {char.wardrobe_lock}. "
        f"Posture: {char.posture_and_mannerisms}. "
        "Even soft studio key lighting, ultra-consistent face and clothing across all 3 panels, "
        "strictly zero text, labels, borders, or watermarks."
    )


def compile_prop_turnaround_prompt(prop: PropAsset, vault: WorldVault) -> str:
    """Compile a 16:9 3-panel macro prop study sheet prompt."""
    craft = vault.taste_signature.compiled_craft_rules
    default_desc = prop.describe_state(prop.default_state)
    alt_states = [v for k, v in prop.states.items() if k != prop.default_state]
    active_desc = alt_states[0] if alt_states else default_desc
    return (
        f"Photorealistic macro product & cinema prop turnaround triptych ({craft.film_texture}) on dark neutral surface: "
        f"[Left third: straight-on frontal hero view of {prop.name}] "
        f"[Center third: 45-degree angled tactile detail view] "
        f"[Right third: active state showing {active_desc}]. "
        f"Geometry & scale: {prop.geometry_and_scale}. "
        f"Materials & surface texture: {prop.surface_textures}. "
        "Crisp studio rim lighting, zero text, zero labels, zero watermarks."
    )


def compile_location_wall_prompt(
    loc: LocationAsset,
    vault: WorldVault,
    *,
    reverse: bool = False,
) -> str:
    """Compile either the Primary (A-Side) or Reverse (B-Side) architectural set plate (empty of people)."""
    craft = vault.taste_signature.compiled_craft_rules
    wall_desc = (
        loc.cardinal_zones.B_south_reverse if reverse else loc.cardinal_zones.A_north_primary
    )
    side_notes = ", ".join(
        s for s in (loc.cardinal_zones.west_side, loc.cardinal_zones.east_side) if s
    )
    textures = ", ".join(loc.surface_textures[:4])
    reverse_prefix = (
        "180-degree reverse-angle architectural view of the exact same room, preserving identical lighting temperature, color grade, floor/wall materials, and atmosphere: "
        if reverse
        else ""
    )
    return (
        f"{reverse_prefix}Empty cinema set establishing plate of {loc.name} ({loc.time_of_day_and_weather}). "
        f"Primary view in frame: {wall_desc}. "
        + (f"Peripheral architecture: {side_notes}. " if side_notes else "")
        + (f"Surface textures: {textures}. " if textures else "")
        + f"Atmosphere: {loc.ambient_particles}. "
        + f"Optics & lighting: {craft.lens_package}, {craft.lighting_philosophy}, {craft.film_texture}. "
        + "Empty set plate with NO people or characters in frame, zero readable text or watermarks."
    )


def slice_three_panel_turnaround(
    sheet_path: Path,
    out_dir: Path,
    prefix: str,
) -> tuple[Path, Path, Path]:
    """Slice a 16:9 3-panel turnaround sheet into 3 clean, Kling-aspect-safe portrait crops."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(sheet_path) as im:
        rgb = im.convert("RGB")
        w, h = rgb.size
        third = max(1, w // 3)
        # Trim 2% inner gutter so panel seams never bleed into crops
        pad_x = max(2, int(third * 0.02))
        boxes = [
            (0, 0, max(10, third - pad_x), h),
            (min(w - 10, third + pad_x), 0, max(third + 12, 2 * third - pad_x), h),
            (min(w - 10, 2 * third + pad_x), 0, w, h),
        ]
        names = ("frontal", "3q", "full")
        saved: list[Path] = []
        for name, box in zip(names, boxes, strict=True):
            crop = rgb.crop(box)
            safe_crop = ensure_kling_element_dimensions(crop)
            dest = out_dir / f"{prefix}_{name}.jpg"
            safe_crop.save(dest, format="JPEG", quality=92)
            saved.append(dest)
    return saved[0], saved[1], saved[2]


def ensure_kling_element_dimensions(img: Image.Image) -> Image.Image:
    """Guarantee image satisfies Kling v3 @Elements bounds (300-2048px side, 0.4-2.5 aspect) via edge-color pad."""
    rgb = img.convert("RGB")
    w, h = rgb.size
    aspect = w / float(max(1, h))
    # Sample average edge color for seamless padding if needed
    edge_px = rgb.getpixel((0, 0))
    if aspect < 0.42:
        new_w = int(round(h * 0.45))
        canvas = Image.new("RGB", (new_w, h), edge_px)
        canvas.paste(rgb, ((new_w - w) // 2, 0))
        rgb = canvas
    elif aspect > 2.45:
        new_h = int(round(w / 2.4))
        canvas = Image.new("RGB", (w, new_h), edge_px)
        canvas.paste(rgb, (0, (new_h - h) // 2))
        rgb = canvas

    w, h = rgb.size
    min_side = min(w, h)
    max_side = max(w, h)
    if min_side < 320:
        scale = 320.0 / float(max(1, min_side))
        rgb = rgb.resize((int(round(w * scale)), int(round(h * scale))), Image.Resampling.LANCZOS)
    elif max_side > 2000:
        scale = 2000.0 / float(max_side)
        rgb = rgb.resize((int(round(w * scale)), int(round(h * scale))), Image.Resampling.LANCZOS)
    return rgb


def compose_scene_reference_strip(
    panel_paths: list[Path],
    out_path: Path,
    *,
    target_height: int = 720,
) -> Path | None:
    """Composite up to 3 reference images ([Wall Plate | Character | Prop]) into 1 conditioning strip."""
    valid = [p for p in panel_paths if p and p.is_file()][:3]
    if not valid:
        return None
    if len(valid) == 1:
        with Image.open(valid[0]) as im:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            im.convert("RGB").save(out_path, format="JPEG", quality=90)
        return out_path

    resized: list[Image.Image] = []
    for p in valid:
        with Image.open(p) as im:
            rgb = im.convert("RGB")
            w, h = rgb.size
            new_w = max(240, int(round(w * (target_height / float(max(1, h))))))
            resized.append(rgb.resize((min(960, new_w), target_height), Image.Resampling.LANCZOS))

    gap = 8
    total_w = sum(im.size[0] for im in resized) + gap * (len(resized) - 1)
    strip = Image.new("RGB", (total_w, target_height), (18, 20, 24))
    cursor_x = 0
    for im in resized:
        strip.paste(im, (cursor_x, 0))
        cursor_x += im.size[0] + gap

    out_path.parent.mkdir(parents=True, exist_ok=True)
    strip.save(out_path, format="JPEG", quality=90)
    return out_path


# ---------------------------------------------------------------------------
# Start & End Keyframe Slice Compilers (Static Geometry Only)
# ---------------------------------------------------------------------------


def _resolve_wall_description(shot: CinemaShot, loc: LocationAsset | None) -> str:
    if loc is None:
        return ""
    zone = shot.camera.facing_wall_zone
    if zone == "B_south_reverse":
        return loc.cardinal_zones.B_south_reverse
    if zone == "west_side" and loc.cardinal_zones.west_side:
        return loc.cardinal_zones.west_side
    if zone == "east_side" and loc.cardinal_zones.east_side:
        return loc.cardinal_zones.east_side
    return loc.cardinal_zones.A_north_primary


def compile_start_keyframe_prompt(
    shot: CinemaShot,
    scene: CinemaScene,
    vault: WorldVault,
    *,
    has_ref_strip: bool = False,
    tweak_note: str | None = None,
) -> str:
    """Compile static photographic Start Frame prompt (zero motion verbs, zero end-frame bleed)."""
    craft = vault.taste_signature.compiled_craft_rules
    loc = vault.get_location(scene.location_id)
    prop_states = resolve_shot_prop_states(shot, scene, vault)

    comp = hydrate_entities_in_text(shot.start_frame.composition, vault, prop_states)
    pose = hydrate_entities_in_text(
        shot.start_frame.character_pose, vault, prop_states, concise=True
    )
    fg = hydrate_entities_in_text(shot.spatial_zones.foreground, vault, prop_states, concise=True)
    mg = hydrate_entities_in_text(shot.spatial_zones.midground, vault, prop_states, concise=True)
    bg = hydrate_entities_in_text(shot.spatial_zones.background, vault, prop_states, concise=True)
    wall_txt = _resolve_wall_description(shot, loc)
    textures_txt = ", ".join(shot.textures_and_atmosphere[:4])

    parts: list[str] = []
    if has_ref_strip:
        parts.append(
            "Using the exact room architecture, character facial identity, wardrobe fabric, and prop geometry "
            "from the reference images, compose one unified seamless single-frame photograph (NOT a split screen or collage)"
        )

    parts.append(f"Shot scale ({shot.shot_scale.replace('_', ' ')}): {comp.rstrip('.')}")
    if pose:
        parts.append(f"Pose & expression: {pose.rstrip('.')}")

    spatial_bits = [
        f"Foreground: {fg}" if fg else "",
        f"Midground: {mg}" if mg else "",
        f"Background ({shot.camera.facing_wall_zone}): {bg or wall_txt}"
        if (bg or wall_txt)
        else "",
    ]
    spatial_joined = "; ".join(b for b in spatial_bits if b)
    if spatial_joined:
        parts.append(f"Spatial blocking: {spatial_joined}")

    if textures_txt:
        parts.append(f"Tactile surfaces & atmosphere: {textures_txt}")

    parts.append(
        f"Camera optics: {shot.camera.lens_mm}mm cinema prime, {shot.camera.t_stop}, "
        f"{shot.camera.angle}, eyeline {shot.camera.eyeline_vector.replace('_', ' ')}, "
        f"focal plane: {shot.start_frame.focal_plane}"
    )
    parts.append(
        f"Lighting & palette: {shot.lighting.key}; {shot.lighting.fill_and_rim}; "
        f"palette {', '.join((shot.lighting.palette_hex or scene.color_palette_hex)[:3])}"
    )
    parts.append(f"Film stock & craft: {craft.film_texture}")
    if tweak_note and tweak_note.strip():
        parts.append(f"Director note: {tweak_note.strip()}")
    parts.append(
        "Single full-bleed cinema still frame, ultra-authentic textures, strictly zero readable text, signs, or watermarks"
    )
    return ". ".join(p.strip().rstrip(".") for p in parts if p and p.strip()) + "."


def compile_end_keyframe_prompt(
    shot: CinemaShot,
    scene: CinemaScene,
    vault: WorldVault,
    *,
    anchored_to_start: bool = True,
    tweak_note: str | None = None,
) -> str:
    """Compile Kontext Start->End delta prompt preserving exact identity, set, and lighting."""
    end_states = resolve_shot_prop_states(shot, scene, vault)
    end_states.update(shot.end_frame.prop_states)

    comp = hydrate_entities_in_text(shot.end_frame.composition, vault, end_states)
    pose = hydrate_entities_in_text(shot.end_frame.character_pose, vault, end_states, concise=True)

    parts: list[str] = []
    if anchored_to_start:
        parts.append(
            "Preserve the EXACT same person, facial structure, wardrobe weave, room architecture, "
            "surface grain, and lighting color temperature from the reference start frame, "
            "updating only the physical pose, camera framing, and focus to this ending state"
        )
    else:
        craft = vault.taste_signature.compiled_craft_rules
        parts.append(f"Cinema still frame ({craft.film_texture})")

    parts.append(f"Ending composition ({shot.shot_scale.replace('_', ' ')}): {comp.rstrip('.')}")
    if pose:
        parts.append(f"Ending pose & expression: {pose.rstrip('.')}")
    if shot.end_frame.focal_plane:
        parts.append(f"Ending focus: {shot.end_frame.focal_plane.rstrip('.')}")
    parts.append(f"Lighting continuity: {shot.lighting.key}")
    if tweak_note and tweak_note.strip():
        parts.append(f"Director note: {tweak_note.strip()}")
    parts.append("Single full-bleed cinema still frame, zero readable text, zero watermarks")
    return ". ".join(p.strip().rstrip(".") for p in parts if p and p.strip()) + "."


# ---------------------------------------------------------------------------
# Model-Native Video Prompt Compiler (`structured_json` & `cinematic_prose`)
# ---------------------------------------------------------------------------


def resolve_shot_element_bindings(
    shot: CinemaShot,
    vault: WorldVault,
    *,
    supports_elements: bool = True,
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """Bind up to 3 active Characters/Props in the shot to `@Element1..3` with multi-angle crops."""
    if not supports_elements:
        return {}, []

    tag_map: dict[str, str] = {}
    bundles: list[dict[str, Any]] = []
    slot = 1

    # Priority 1: Characters visible in shot
    for cid in shot.characters_in_shot:
        if slot > 3:
            break
        ch = vault.get_character(cid)
        if ch is None:
            continue
        frontal = ch.frontal_crop_path or ch.uploaded_ref_path or ch.turnaround_sheet_path
        if frontal and Path(frontal).is_file():
            refs = [
                p
                for p in (ch.three_quarter_crop_path, ch.full_body_crop_path, frontal)
                if p and Path(p).is_file()
            ]
            tag = f"@Element{slot}"
            tag_map[ch.id] = tag
            bundles.append(
                {
                    "tag": tag,
                    "entity_id": ch.id,
                    "label": ch.name,
                    "frontal_path": frontal,
                    "reference_paths": refs[:3] or [frontal],
                }
            )
            slot += 1

    # Priority 2: Active hero props in shot
    for p_inst in shot.props_in_shot:
        if slot > 3:
            break
        pr = vault.get_prop(p_inst.prop_id)
        if pr is None:
            continue
        frontal = pr.frontal_crop_path or pr.uploaded_ref_path or pr.turnaround_sheet_path
        if frontal and Path(frontal).is_file():
            tag = f"@Element{slot}"
            tag_map[pr.id] = tag
            bundles.append(
                {
                    "tag": tag,
                    "entity_id": pr.id,
                    "label": pr.name,
                    "frontal_path": frontal,
                    "reference_paths": [frontal],
                }
            )
            slot += 1

    return tag_map, bundles


def to_hydrated_shot_prompt_dict(
    shot: CinemaShot,
    scene: CinemaScene,
    vault: WorldVault,
    *,
    supports_elements: bool = True,
) -> dict[str, Any]:
    """Build the hydrated model-native JSON prompt dictionary for a single `CinemaShot`."""
    craft = vault.taste_signature.compiled_craft_rules
    loc = vault.get_location(scene.location_id)
    prop_states = resolve_shot_prop_states(shot, scene, vault)
    tag_map, bundles = resolve_shot_element_bindings(
        shot, vault, supports_elements=supports_elements
    )

    wall_txt = _resolve_wall_description(shot, loc)
    doc: dict[str, Any] = {
        "type": "cinema_shot_prompt",
        "shot_id": shot.shot_id,
        "duration": f"{shot.duration_sec} seconds",
        "shot_scale": shot.shot_scale,
        "anchor_strategy": shot.anchor_strategy,
        "style_and_stock": craft.film_texture,
        "camera": {
            "lens_mm": shot.camera.lens_mm,
            "aperture": shot.camera.t_stop,
            "angle": shot.camera.angle,
            "movement": shot.camera.movement,
            "eyeline": shot.camera.eyeline_vector,
        },
        "environment_and_blocking": {
            "location": f"{loc.name} ({loc.time_of_day_and_weather})" if loc else scene.slugline,
            "facing_wall": wall_txt,
            "foreground": hydrate_entities_in_text(
                shot.spatial_zones.foreground,
                vault,
                prop_states,
                element_tag_map=tag_map,
                concise=True,
            ),
            "midground": hydrate_entities_in_text(
                shot.spatial_zones.midground,
                vault,
                prop_states,
                element_tag_map=tag_map,
                concise=True,
            ),
            "background": hydrate_entities_in_text(
                shot.spatial_zones.background,
                vault,
                prop_states,
                element_tag_map=tag_map,
                concise=True,
            ),
            "textures": shot.textures_and_atmosphere[:4],
        },
        "lighting": {
            "key": shot.lighting.key,
            "fill_and_rim": shot.lighting.fill_and_rim,
            "palette": (shot.lighting.palette_hex or scene.color_palette_hex)[:3],
        },
        "start_frame": {
            "composition": hydrate_entities_in_text(
                shot.start_frame.composition,
                vault,
                prop_states,
                element_tag_map=tag_map,
                concise=True,
            ),
            "pose": hydrate_entities_in_text(
                shot.start_frame.character_pose,
                vault,
                prop_states,
                element_tag_map=tag_map,
                concise=True,
            ),
        },
        "timeline": [
            {
                "window": b.window,
                "action": hydrate_entities_in_text(
                    b.action, vault, prop_states, element_tag_map=tag_map, concise=True
                ),
                **({"camera": b.camera_delta} if b.camera_delta else {}),
            }
            for b in shot.motion_delta
        ],
        "end_frame": {
            "composition": hydrate_entities_in_text(
                shot.end_frame.composition,
                vault,
                {**prop_states, **shot.end_frame.prop_states},
                element_tag_map=tag_map,
                concise=True,
            ),
            "pose": hydrate_entities_in_text(
                shot.end_frame.character_pose,
                vault,
                {**prop_states, **shot.end_frame.prop_states},
                element_tag_map=tag_map,
                concise=True,
            ),
        },
    }
    if bundles:
        doc["elements"] = [f"{b['tag']}: {b['label']}" for b in bundles]
    if vault.negative_constraints:
        doc["avoid"] = vault.negative_constraints[:3]
    return doc


def compile_shot_video_prompt(
    shot: CinemaShot,
    scene: CinemaScene,
    vault: WorldVault,
    spec: VideoModelSpec,
) -> str:
    """Compile hydrated shot instructions tailored to `spec.prompt_style` within `spec.max_prompt_chars`."""
    max_chars = max(300, int(spec.max_prompt_chars))
    if getattr(spec, "prompt_style", "structured_json") == "cinematic_prose":
        return _compile_shot_prose(
            shot,
            scene,
            vault,
            supports_elements=spec.supports_elements,
            max_chars=max_chars,
        )

    doc = to_hydrated_shot_prompt_dict(shot, scene, vault, supports_elements=spec.supports_elements)
    compact = json.dumps(doc, ensure_ascii=False, separators=(",", ":"))
    if len(compact) <= max_chars:
        return compact

    trimmed = copy.deepcopy(doc)
    for drop_k in ("avoid", "style_and_stock", "shot_id"):
        trimmed.pop(drop_k, None)
        compact = json.dumps(trimmed, ensure_ascii=False, separators=(",", ":"))
        if len(compact) <= max_chars:
            return compact

    env = trimmed.get("environment_and_blocking")
    if isinstance(env, dict):
        env.pop("facing_wall", None)
        env["textures"] = (env.get("textures") or [])[:2]
    compact = json.dumps(trimmed, ensure_ascii=False, separators=(",", ":"))
    if len(compact) <= max_chars:
        return compact

    for cap in (140, 100, 72):
        for section_key in ("start_frame", "end_frame", "environment_and_blocking", "lighting"):
            sec_obj = trimmed.get(section_key)
            if isinstance(sec_obj, dict):
                for k, v in list(sec_obj.items()):
                    if isinstance(v, str) and len(v) > cap:
                        sec_obj[k] = v[: cap - 1].rstrip() + "…"
        tl = trimmed.get("timeline")
        if isinstance(tl, list):
            for beat in tl:
                if (
                    isinstance(beat, dict)
                    and isinstance(beat.get("action"), str)
                    and len(beat["action"]) > cap
                ):
                    beat["action"] = beat["action"][: cap - 1].rstrip() + "…"
        compact = json.dumps(trimmed, ensure_ascii=False, separators=(",", ":"))
        if len(compact) <= max_chars:
            return compact

    return _compile_shot_prose(
        shot,
        scene,
        vault,
        supports_elements=spec.supports_elements,
        max_chars=max_chars,
    )


def _compile_shot_prose(
    shot: CinemaShot,
    scene: CinemaScene,
    vault: WorldVault,
    *,
    supports_elements: bool,
    max_chars: int,
) -> str:
    craft = vault.taste_signature.compiled_craft_rules
    loc = vault.get_location(scene.location_id)
    prop_states = resolve_shot_prop_states(shot, scene, vault)
    tag_map, _ = resolve_shot_element_bindings(shot, vault, supports_elements=supports_elements)

    start_txt = hydrate_entities_in_text(
        shot.start_frame.composition, vault, prop_states, element_tag_map=tag_map, concise=True
    )
    actions = ". ".join(
        hydrate_entities_in_text(
            b.action, vault, prop_states, element_tag_map=tag_map, concise=True
        ).rstrip(".")
        for b in shot.motion_delta[:4]
        if b.action
    )
    end_txt = hydrate_entities_in_text(
        shot.end_frame.composition,
        vault,
        {**prop_states, **shot.end_frame.prop_states},
        element_tag_map=tag_map,
        concise=True,
    )
    textures = ", ".join(shot.textures_and_atmosphere[:3])
    loc_label = f"{loc.name} ({loc.time_of_day_and_weather})" if loc else scene.slugline

    segments = [
        f"{shot.shot_scale.replace('_', ' ').title()} in {loc_label}: {start_txt.rstrip('.')}.",
        f"Choreography: {actions}." if actions else "",
        f"Settles into final frame: {end_txt.rstrip('.')}." if end_txt else "",
        f"Camera: {shot.camera.lens_mm}mm prime, {shot.camera.t_stop}, {shot.camera.angle}, {shot.camera.movement}.",
        f"Textures & light: {textures}; {shot.lighting.key}."
        if textures
        else f"Light: {shot.lighting.key}.",
        f"{craft.film_texture}, natural physical inertia, strictly zero readable text or watermarks.",
    ]
    prose = re.sub(r"\s+", " ", " ".join(s for s in segments if s)).strip()
    return prose if len(prose) <= max_chars else prose[: max_chars - 1].rstrip() + "."


# ---------------------------------------------------------------------------
# Cinema Script Gate & Story Rubric Evaluator
# ---------------------------------------------------------------------------


def choose_smart_anchor_strategy(shot: CinemaShot, is_chained_from_prev: bool = False) -> str:
    """Apply Trap #3 rule: avoid synthetic FLF2V puppet-morphing on subtle speaking close-ups."""
    if is_chained_from_prev:
        return "chained_from_prev_to_end"
    has_speaking_cu = (
        shot.dialogue is not None
        and shot.dialogue.delivery_mode == "on_screen_speaking"
        and shot.shot_scale in ("close_up", "medium_close_up")
    )
    low_move = shot.camera.movement.lower()
    is_subtle_camera = any(
        k in low_move
        for k in ("locked", "static", "subtle", "gentle", "5cm", "8cm", "10cm", "12cm", "breathing")
    ) and not any(k in low_move for k in ("whip", "crane", "orbit", "180", "sweep", "pull-back"))
    if has_speaking_cu and is_subtle_camera:
        return "start_and_elements_forward"
    return "start_and_end_locked"


def validate_cinema_project(
    project: CinemaProject,
    model_spec: VideoModelSpec,
) -> dict[str, Any]:
    """Deterministic Script & Continuity Gate across all Scenes and Shots."""
    issues: list[dict[str, Any]] = []
    shot_reports: list[dict[str, Any]] = []

    known_props = {p.id: p for p in project.vault.props}
    known_chars = {c.id for c in project.vault.characters}

    # 1. Scene causality & prop state continuity check
    running_prop_states: dict[str, str] = {p.id: p.default_state for p in project.vault.props}
    for s_idx, sc in enumerate(project.scenes):
        if s_idx > 0 and sc.causal_link_from_prev not in ("but", "therefore", "meanwhile"):
            issues.append(
                {
                    "scope": sc.scene_id,
                    "code": "weak_causality",
                    "message": f"{sc.title} should connect via 'but' or 'therefore' to drive story momentum.",
                }
            )
        for pid, st_key in sc.scene_entry_state.prop_states.items():
            if pid in known_props and st_key not in known_props[pid].states:
                issues.append(
                    {
                        "scope": sc.scene_id,
                        "code": "unknown_prop_state",
                        "message": f"Prop {pid} state '{st_key}' is not defined in WorldVault ({list(known_props[pid].states)}).",
                    }
                )
        running_prop_states.update(sc.scene_exit_state.prop_states)

        # Shot-scale monotony check inside scene
        scales = [sh.shot_scale for sh in sc.shots if sh.render_engine == "ai_cinema"]
        for i in range(2, len(scales)):
            if (
                scales[i] == scales[i - 1] == scales[i - 2]
                and project.vault.taste_signature.dials.pacing >= 0.25
            ):
                issues.append(
                    {
                        "scope": sc.shots[i].shot_id,
                        "code": "monotonous_shot_scale",
                        "message": f"3 consecutive shots use '{scales[i]}' framing — vary shot scale (e.g. insert or close-up) for cinema rhythm.",
                    }
                )

        for sh in sc.shots:
            sh_issues: list[dict[str, str]] = []
            compiled = compile_shot_video_prompt(sh, sc, project.vault, model_spec)

            for cid in sh.characters_in_shot:
                if known_chars and cid not in known_chars:
                    sh_issues.append(
                        {
                            "code": "unknown_character_id",
                            "message": f"Character '{cid}' is not registered in WorldVault.",
                        }
                    )

            if sh.render_engine == "ai_cinema":
                blob = f"{sh.start_frame.composition} {sh.end_frame.composition} " + " ".join(
                    b.action for b in sh.motion_delta
                )
                m = _TEXT_IN_AI_SHOT_RE.search(blob)
                if m:
                    sh_issues.append(
                        {
                            "code": "garbled_text_risk",
                            "message": f"Mentions '{m.group(0)}' in AI cinema shot — switch shot to 'insert_ui_plate' or 'title_card' for 100% sharp text.",
                        }
                    )

            if sh.dialogue is not None and sh.dialogue.line.strip():
                if sh.dialogue.actual_speech_sec is not None:
                    min_needed = sh.dialogue.pre_pause_sec + sh.dialogue.actual_speech_sec + 0.35
                else:
                    words = len(sh.dialogue.line.strip().split())
                    min_needed = sh.dialogue.pre_pause_sec + (words / 2.6) + 0.35
                if min_needed > float(sh.duration_sec) + 0.15:
                    sh_issues.append(
                        {
                            "code": "dialogue_overflow",
                            "message": f"Dialogue needs ~{min_needed:.1f}s (including breath pause) vs {sh.duration_sec}s shot duration.",
                        }
                    )

            for item in sh_issues:
                issues.append({"scope": sh.shot_id, **item})

            shot_reports.append(
                {
                    "shot_id": sh.shot_id,
                    "scene_id": sc.scene_id,
                    "valid": len(sh_issues) == 0,
                    "issues": sh_issues,
                    "compiled_chars": len(compiled),
                    "max_chars": model_spec.max_prompt_chars,
                }
            )

    return {
        "valid": len(issues) == 0,
        "total_issues": len(issues),
        "issues": issues,
        "shots": shot_reports,
    }


def evaluate_story_rubric(project: CinemaProject) -> StoryRubricReport:
    """Grade narrative propulsion (`Therefore/But`), restraint, prop payoffs, and coverage variety."""
    strengths: list[str] = []
    warnings: list[str] = []

    if not project.scenes:
        return StoryRubricReport(
            overall_score=0.0,
            causality_score=0.0,
            subtext_and_restraint_score=0.0,
            setup_payoff_score=0.0,
            visual_variety_score=0.0,
            passed=False,
            warnings=["Project has no compiled scenes yet."],
        )

    # 1. Causality score
    follow_scenes = project.scenes[1:]
    strong_links = sum(
        1
        for sc in follow_scenes
        if sc.causal_link_from_prev in ("but", "therefore") and len(sc.dramatic_turn.strip()) >= 12
    )
    causality = (
        9.4 if not follow_scenes else round(6.5 + 3.0 * (strong_links / len(follow_scenes)), 2)
    )
    if causality >= 8.8:
        strengths.append("Strong 'Therefore / But' causal story spine across scenes.")
    else:
        warnings.append("Strengthen causal 'BUT / THEREFORE' turns between scenes.")

    # 2. Subtext & anti-cliché restraint
    all_lines = " ".join(
        sh.dialogue.line for sh in project.all_shots() if sh.dialogue and sh.dialogue.line
    )
    cliche_hits = _CHEESY_CLICHE_RE.findall(all_lines)
    total_words = len(all_lines.split())
    dur_min = max(0.5, project.total_duration_sec() / 60.0)
    wpm = total_words / dur_min
    budget_wpm = project.vault.taste_signature.compiled_craft_rules.max_wpm

    restraint = 9.3
    if cliche_hits:
        restraint -= min(3.0, len(cliche_hits) * 1.2)
        warnings.append(
            f"Detected commercial clichés ({', '.join(set(cliche_hits))}); rewrite with grounded human subtext."
        )
    if wpm > budget_wpm + 20:
        restraint -= 1.2
        warnings.append(
            f"Dialogue density ({int(wpm)} wpm) exceeds taste target ({budget_wpm} wpm); let visuals breathe."
        )
    else:
        strengths.append(
            f"Restrained cinema dialogue pacing ({int(wpm)} wpm) leaves room for visual acting and room tone."
        )
    restraint = round(max(4.0, restraint), 2)

    # 3. Prop Setup -> Payoff tracking
    used_props = {
        inst.prop_id for sh in project.all_shots() for inst in sh.props_in_shot if inst.prop_id
    }
    if project.vault.props:
        ratio = len(used_props) / float(len(project.vault.props))
        payoff_score = round(min(9.6, 6.8 + 2.8 * ratio), 2)
        if ratio >= 0.8:
            strengths.append("Vault hero props are actively staged and transformed across shots.")
    else:
        payoff_score = 7.5
        warnings.append(
            "Add at least one physical hero prop in WorldVault to anchor tactile close-ups."
        )

    # 4. Shot-scale & lens coverage variety
    distinct_scales = {sh.shot_scale for sh in project.all_shots()}
    variety = round(min(9.6, 5.8 + 0.85 * len(distinct_scales)), 2)
    if len(distinct_scales) >= 4:
        strengths.append(f"Rich camera coverage ({len(distinct_scales)} distinct shot scales).")
    else:
        warnings.append(
            "Increase shot-scale contrast (mix wide establishing, medium, close-up, and macro prop inserts)."
        )

    overall = round((causality + restraint + payoff_score + variety) / 4.0, 2)
    return StoryRubricReport(
        overall_score=overall,
        causality_score=causality,
        subtext_and_restraint_score=restraint,
        setup_payoff_score=payoff_score,
        visual_variety_score=variety,
        passed=overall >= 7.5,
        strengths=strengths,
        warnings=warnings,
    )


def autofix_cinema_project_gate(project: CinemaProject) -> CinemaProject:
    """Deterministically heal dialogue duration overflows and speaking close-up anchor strategies."""
    scale_cycle: list[Any] = [
        "wide",
        "medium",
        "close_up",
        "extreme_close_up_insert",
        "over_the_shoulder",
        "medium_close_up",
    ]
    for sc in project.scenes:
        for idx, sh in enumerate(sc.shots):
            # 1. Expand duration if dialogue speech + pause exceeds current shot duration
            if sh.dialogue is not None and sh.dialogue.line.strip():
                speech_s = sh.dialogue.actual_speech_sec
                if speech_s is None:
                    speech_s = round(len(sh.dialogue.line.strip().split()) / 2.6, 2)
                needed = sh.dialogue.pre_pause_sec + speech_s + 0.55
                if needed > float(sh.duration_sec):
                    for snap_d in (5, 6, 8, 10, 12, 15):
                        if snap_d >= needed:
                            sh.duration_sec = snap_d
                            break
                    else:
                        sh.duration_sec = 15

            # 2. Break 3-in-a-row identical shot scales
            if (
                idx >= 2
                and sh.render_engine == "ai_cinema"
                and sh.shot_scale == sc.shots[idx - 1].shot_scale == sc.shots[idx - 2].shot_scale
            ):
                for candidate in scale_cycle:
                    if candidate != sh.shot_scale:
                        sh.shot_scale = candidate
                        break

            # 3. Apply Trap #3 smart anchor strategy on subtle speaking close-ups
            chained = sh.anchor_strategy == "chained_from_prev_to_end"
            sh.anchor_strategy = choose_smart_anchor_strategy(sh, is_chained_from_prev=chained)  # type: ignore[assignment]

    project.story_rubric = evaluate_story_rubric(project)
    return project


__all__ = [
    "autofix_cinema_project_gate",
    "choose_smart_anchor_strategy",
    "compile_character_turnaround_prompt",
    "compile_end_keyframe_prompt",
    "compile_location_wall_prompt",
    "compile_prop_turnaround_prompt",
    "compile_shot_video_prompt",
    "compile_start_keyframe_prompt",
    "compose_scene_reference_strip",
    "ensure_kling_element_dimensions",
    "evaluate_story_rubric",
    "hydrate_entities_in_text",
    "resolve_shot_element_bindings",
    "resolve_shot_prop_states",
    "slice_three_panel_turnaround",
    "to_hydrated_shot_prompt_dict",
    "validate_cinema_project",
]
