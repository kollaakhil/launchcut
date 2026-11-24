"""Taste-to-Craft Translator & Persistent DirectorTasteProfile Store.

Translates non-expert human taste signals (visual Cinema Mood Cards, plain-English
vibe notes, 4 human-language dials, and 1-click Producer Reaction Chips) into
concrete cinematography, lighting, color-script, and Walter Murch editorial rules.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gtm_videogen.models.cinema_project import (
    CompiledCraftRules,
    DirectorTasteProfile,
    StoryArchetype,
    TasteReactionEvent,
    TasteSignature,
    VibeDials,
)

CINEMA_MOOD_CATALOG: list[dict[str, Any]] = [
    {
        "id": "cinema_a24",
        "label": "Intimate A24 Naturalism",
        "tagline": "Quiet human tension, soft window light, 35mm grain, zero melodrama",
        "gallery_image": "/api/stitch/gallery/cinema_a24.jpg",
        "palette_hex": ["#14191F", "#C9894B", "#EAE0D0"],
        "dials": {"scale": 0.30, "pacing": 0.38, "mood_contrast": 0.65, "dialogue_density": 0.50},
        "lens_hint": "vintage 35mm & 65mm prime lenses, soft halation, intimate eye-level framing",
        "lighting_hint": "motivated practical lamps and overcast window key light with rich shadow roll-off",
        "texture_hint": "organic Kodak Vision3 500T 35mm grain, authentic skin pores, lived-in surfaces",
        "score_hint": "sparse felt piano, bowed cello harmonics, and tactile room silence",
    },
    {
        "id": "cinema_imax",
        "label": "Architectural IMAX Scale",
        "tagline": "Monumental geometry, sweeping vistas, crisp anamorphic depth, resonant bass",
        "gallery_image": "/api/stitch/gallery/cinema_imax.jpg",
        "palette_hex": ["#0D141C", "#D4943A", "#DCE5EC"],
        "dials": {"scale": 0.90, "pacing": 0.52, "mood_contrast": 0.72, "dialogue_density": 0.35},
        "lens_hint": "large-format 28mm & 50mm IMAX panavision optics, deep architectural lines",
        "lighting_hint": "epic volumetric shafts, crisp directional sunlight or stark monolithic contrast",
        "texture_hint": "ultra-clean 65mm large-format clarity, weathered concrete, steel, atmospheric mist",
        "score_hint": "deep orchestral brass swells, sub-bass pulse, and ticking rhythmic tension",
    },
    {
        "id": "cinema_noir",
        "label": "Neo-Noir Rain & Shadow",
        "tagline": "Wet asphalt reflections, amber tungsten vs. midnight cyan, razor-sharp intrigue",
        "gallery_image": "/api/stitch/gallery/cinema_noir.jpg",
        "palette_hex": ["#0A0E16", "#E08B2E", "#4A7C96"],
        "dials": {"scale": 0.48, "pacing": 0.56, "mood_contrast": 0.92, "dialogue_density": 0.52},
        "lens_hint": "40mm & 75mm anamorphic primes with horizontal amber streak bokeh",
        "lighting_hint": "high-contrast 6:1 chiaroscuro, sodium-vapor rim light against cold rain shadow",
        "texture_hint": "rain-slicked glass, damp wool, brushed brass, cigarette-haze atmosphere",
        "score_hint": "brooding analog synth bass, muted trumpet air, and rain-on-glass foley",
    },
    {
        "id": "cinema_pastel",
        "label": "Warm Editorial Symmetry",
        "tagline": "Poised center compositions, sunlit tactile warmth, wry human charm",
        "gallery_image": "/api/stitch/gallery/cinema_pastel.jpg",
        "palette_hex": ["#2A2320", "#E26D5A", "#F6EAD7"],
        "dials": {"scale": 0.45, "pacing": 0.62, "mood_contrast": 0.22, "dialogue_density": 0.62},
        "lens_hint": "40mm rectilinear prime, deadpan symmetrical lock-offs and crisp whip-pans",
        "lighting_hint": "warm golden-hour bounce fill, soft pastel color separation, luminous skin tones",
        "texture_hint": "tactile paper, warm walnut, woven linen, crisp 35mm daylight film stock",
        "score_hint": "plucked pizzicato strings, warm woodwinds, and crisp percussive foley",
    },
    {
        "id": "cinema_grit",
        "label": "Raw Handheld Vérité",
        "tagline": "Unfiltered documentary urgency, breathing camera, real sweat & texture",
        "gallery_image": "/api/stitch/gallery/cinema_grit.jpg",
        "palette_hex": ["#161513", "#B86B35", "#DFD8CC"],
        "dials": {"scale": 0.22, "pacing": 0.72, "mood_contrast": 0.68, "dialogue_density": 0.58},
        "lens_hint": "handheld 35mm & 85mm cinema zoom, tight over-the-shoulder proximity",
        "lighting_hint": "available fluorescent/daylight practicals, raw unpolished contrast",
        "texture_hint": "16mm Kodak grain, raw metal, scuffed leather, authentic hand micro-tremor",
        "score_hint": "minimalist percussive pulse and raw diegetic street/room sound",
    },
    {
        "id": "cinema_natgeo",
        "label": "Tactile Macro Craft",
        "tagline": "Sensory close-ups, golden dust motes, artisan physics, breathtaking clarity",
        "gallery_image": "/api/stitch/gallery/cinema_natgeo.jpg",
        "palette_hex": ["#13110E", "#E39B32", "#F4E8D4"],
        "dials": {"scale": 0.28, "pacing": 0.46, "mood_contrast": 0.50, "dialogue_density": 0.35},
        "lens_hint": "90mm macro prime & 50mm cinema lens, razor-thin T1.8 focal plane",
        "lighting_hint": "warm directional studio backlight catching steam, dust, and material grain",
        "texture_hint": "micro-scratched metal, condensation droplets, raw wood fibers, optical glass",
        "score_hint": "hyper-detailed ASMR foley paired with warm acoustic guitar and cello",
    },
    {
        "id": "cinema_neon",
        "label": "Kinetic Cyber Pulse",
        "tagline": "Sleek nocturnal momentum, electric emissive light, propulsive modern thriller",
        "gallery_image": "/api/stitch/gallery/cinema_neon.jpg",
        "palette_hex": ["#090D16", "#25A6D9", "#F06449"],
        "dials": {"scale": 0.62, "pacing": 0.82, "mood_contrast": 0.85, "dialogue_density": 0.45},
        "lens_hint": "35mm & 50mm Master Anamorphic, low-slung gimbal tracking and tight push-ins",
        "lighting_hint": "bi-color emissive practicals, wet specular highlights, deep obsidian blacks",
        "texture_hint": "anodized aluminum, rain-beaded carbon glass, atmospheric night mist",
        "score_hint": "driving modular analog arpeggio, crisp sub-hits, and precision UI/tech foley",
    },
    {
        "id": "cinema_anime",
        "label": "Hand-Painted Anime Cinema",
        "tagline": "Makoto Shinkai luminous skies, emotional wind, expressive cel-shaded warmth",
        "gallery_image": "/api/stitch/gallery/cinema_anime.jpg",
        "palette_hex": ["#182238", "#F28B50", "#FFF1DF"],
        "dials": {"scale": 0.68, "pacing": 0.55, "mood_contrast": 0.48, "dialogue_density": 0.55},
        "lens_hint": "expressive theatrical anime multi-plane depth, dramatic low-angle sky framing",
        "lighting_hint": "luminous twilight rim blooms, dappled sunbeams, painterly atmospheric glow",
        "texture_hint": "hand-painted gouache backgrounds, crisp cel linework, wind-swept fabric motion",
        "score_hint": "emotional lyrical piano melody building into sweeping cinematic strings",
    },
]

STORY_ARCHETYPE_CATALOG: list[dict[str, Any]] = [
    {
        "id": "auto",
        "label": "Showrunner's Choice (3 Contrasting Angles)",
        "description": "Pitch 3 distinct dramatic angles tailored to your idea and taste.",
    },
    {
        "id": "mystery_reveal",
        "label": "Quiet Mystery & Revelation",
        "description": "An anomaly interrupts ordinary life -> investigation -> unexpected truth.",
    },
    {
        "id": "high_stakes_thriller",
        "label": "Ticking-Clock Tension",
        "description": "Immediate pressure -> escalating obstacle ('BUT') -> decisive turn ('THEREFORE').",
    },
    {
        "id": "intimate_character_arc",
        "label": "Human Turning Point",
        "description": "Internal conflict shown through small physical details, silence, and earned change.",
    },
    {
        "id": "founder_origin_story",
        "label": "Obsession & Craft Story (Brand / Product)",
        "description": "Real frustration -> late-night breakthrough -> tangible human impact (zero ad clichés).",
    },
    {
        "id": "visual_poem",
        "label": "Sensory Visual Poem",
        "description": "Minimal dialogue; pure visual metaphor, rhythm, texture, and sound.",
    },
]

TASTE_REACTION_CHIPS: list[dict[str, Any]] = [
    {
        "id": "love_this",
        "label": "♥ Love this vibe",
        "polarity": "love",
        "dial_delta": {},
        "instruction": "Lock in this exact visual tone, pacing restraint, and lighting mood.",
    },
    {
        "id": "less_cheesy",
        "label": "Less commercial / more real",
        "polarity": "adjust",
        "dial_delta": {"dialogue_density": -0.15, "mood_contrast": 0.08},
        "instruction": "Strip all trailer clichés and on-the-nose lines; replace spoken explanation with quiet physical action and lived-in realism.",
    },
    {
        "id": "more_tension",
        "label": "Build more tension",
        "polarity": "adjust",
        "dial_delta": {"mood_contrast": 0.15, "pacing": 0.12},
        "instruction": "Raise dramatic stakes, tighten shot transitions, deepen shadow contrast, and sharpen the 'BUT' obstacle.",
    },
    {
        "id": "warmer_intimate",
        "label": "Warmer & more intimate",
        "polarity": "adjust",
        "dial_delta": {"scale": -0.18, "mood_contrast": -0.14},
        "instruction": "Move the camera closer (65mm-85mm eye-level close-ups), warm key light toward golden tungsten, and linger on human micro-expressions.",
    },
    {
        "id": "slower_atmosphere",
        "label": "Let it breathe (slower)",
        "polarity": "adjust",
        "dial_delta": {"pacing": -0.20, "dialogue_density": -0.12},
        "instruction": "Extend atmospheric holds, trim excess words, and let room tone and visual texture carry the beat.",
    },
    {
        "id": "tighter_pacing",
        "label": "Faster & more kinetic",
        "polarity": "adjust",
        "dial_delta": {"pacing": 0.22},
        "instruction": "Shorten shot durations, add dynamic camera momentum, and cut on active physical motion.",
    },
    {
        "id": "grand_scale",
        "label": "Bigger cinematic scale",
        "polarity": "adjust",
        "dial_delta": {"scale": 0.25},
        "instruction": "Widen architectural framing (28mm-35mm anamorphic), emphasize environmental depth, and elevate sonic scale.",
    },
]


def _clamp01(val: float) -> float:
    return round(max(0.0, min(1.0, float(val))), 3)


def get_mood_preset(mood_id: str | None) -> dict[str, Any]:
    clean = (mood_id or "").strip().lower()
    for item in CINEMA_MOOD_CATALOG:
        if item["id"] == clean:
            return item
    return CINEMA_MOOD_CATALOG[0]


def infer_dials_from_vibe_note(dials: VibeDials, note: str) -> VibeDials:
    """Nudge human dials gently when plain-English vibe notes contain clear signals."""
    low = (note or "").strip().lower()
    if not low:
        return dials
    scale = dials.scale
    pacing = dials.pacing
    contrast = dials.mood_contrast
    dialogue = dials.dialogue_density

    if any(
        k in low for k in ("intimate", "close", "handheld", "personal", "quiet", "cafe", "room")
    ):
        scale = _clamp01(scale - 0.14)
    if any(k in low for k in ("epic", "imax", "grand", "vast", "monumental", "sweeping")):
        scale = _clamp01(scale + 0.20)
    if any(
        k in low for k in ("slow", "contemplative", "lingering", "calm", "meditative", "breathe")
    ):
        pacing = _clamp01(pacing - 0.16)
    if any(k in low for k in ("fast", "kinetic", "urgent", "heist", "adrenaline", "propulsive")):
        pacing = _clamp01(pacing + 0.18)
    if any(k in low for k in ("dark", "noir", "moody", "shadow", "night", "rain", "tense")):
        contrast = _clamp01(contrast + 0.15)
    if any(k in low for k in ("warm", "sunlit", "bright", "pastel", "morning", "hopeful")):
        contrast = _clamp01(contrast - 0.15)
    if any(
        k in low
        for k in ("silent", "no dialogue", "visual only", "minimal words", "show don't tell")
    ):
        dialogue = _clamp01(dialogue - 0.22)
    if any(k in low for k in ("witty", "conversation", "dialogue", "monologue", "sorkin")):
        dialogue = _clamp01(dialogue + 0.18)

    return VibeDials(
        scale=scale,
        pacing=pacing,
        mood_contrast=contrast,
        dialogue_density=dialogue,
    )


def compile_taste_signature(
    *,
    mood_presets: list[str] | None = None,
    story_archetype: StoryArchetype = "auto",
    user_vibe_note: str = "",
    dials_override: VibeDials | None = None,
    taste_profile: DirectorTasteProfile | None = None,
) -> TasteSignature:
    """Deterministically compile human taste choices into technical film-crew rules."""
    chosen_moods = [m for m in (mood_presets or []) if m]
    if not chosen_moods and taste_profile and taste_profile.preferred_mood_presets:
        top_mood = max(
            taste_profile.preferred_mood_presets.items(),
            key=lambda kv: kv[1],
        )[0]
        chosen_moods = [top_mood]
    if not chosen_moods:
        chosen_moods = ["cinema_a24"]

    primary = get_mood_preset(chosen_moods[0])
    secondary = get_mood_preset(chosen_moods[1]) if len(chosen_moods) > 1 else None

    if dials_override is not None:
        base_dials = dials_override.model_copy(deep=True)
    else:
        pd = primary["dials"]
        if secondary is not None:
            sd = secondary["dials"]
            base_dials = VibeDials(
                scale=_clamp01(pd["scale"] * 0.65 + sd["scale"] * 0.35),
                pacing=_clamp01(pd["pacing"] * 0.65 + sd["pacing"] * 0.35),
                mood_contrast=_clamp01(pd["mood_contrast"] * 0.65 + sd["mood_contrast"] * 0.35),
                dialogue_density=_clamp01(
                    pd["dialogue_density"] * 0.65 + sd["dialogue_density"] * 0.35
                ),
            )
        else:
            base_dials = VibeDials(**pd)

        # Blend gently (20%) with persistent DirectorTasteProfile EMA when available
        if taste_profile and taste_profile.reactions_count > 0:
            ema = taste_profile.dial_ema
            base_dials = VibeDials(
                scale=_clamp01(base_dials.scale * 0.8 + ema.scale * 0.2),
                pacing=_clamp01(base_dials.pacing * 0.8 + ema.pacing * 0.2),
                mood_contrast=_clamp01(base_dials.mood_contrast * 0.8 + ema.mood_contrast * 0.2),
                dialogue_density=_clamp01(
                    base_dials.dialogue_density * 0.8 + ema.dialogue_density * 0.2
                ),
            )

    tuned_dials = infer_dials_from_vibe_note(base_dials, user_vibe_note)

    # Translate dials -> concrete optical, lighting, coverage & rhythm specs
    if tuned_dials.scale <= 0.35:
        lens_pkg = (
            f"intimate 50mm & 85mm close-focus primes ({primary['lens_hint']}), shallow T1.8 bokeh"
        )
        coverage = "prioritize intimate close-ups, tactile hand/prop macro inserts, and over-the-shoulder proximity"
    elif tuned_dials.scale >= 0.72:
        lens_pkg = f"sweeping 28mm & 40mm wide anamorphic optics ({primary['lens_hint']}), deep spatial geometry"
        coverage = (
            "open scenes with grand architectural wides balanced against sharp emotional close-ups"
        )
    else:
        lens_pkg = primary["lens_hint"]
        coverage = "classic narrative coverage: establishing wide -> medium two-shot -> tactile prop insert -> eye-line close-up"

    if tuned_dials.mood_contrast >= 0.72:
        light_phil = f"high-contrast chiaroscuro ({primary['lighting_hint']}), deep shadows, crisp rim separation"
    elif tuned_dials.mood_contrast <= 0.32:
        light_phil = f"luminous low-contrast natural fill ({primary['lighting_hint']}), warm highlight roll-off"
    else:
        light_phil = primary["lighting_hint"]

    # Average shot duration: 8.5s at pacing=0.0 -> 4.2s at pacing=1.0
    avg_shot = round(8.5 - 4.3 * tuned_dials.pacing, 1)
    # Spoken word budget: 60 wpm at dialogue=0.0 -> 148 wpm at dialogue=1.0
    max_wpm = int(round(60 + 88 * tuned_dials.dialogue_density))

    if tuned_dials.pacing <= 0.38:
        edit_style = "patient lingering holds, 0.5s J/L-cut dialogue bridges, unhurried room-tone breathing space"
    elif tuned_dials.pacing >= 0.70:
        edit_style = (
            "propulsive match-on-action cuts, crisp 0.25s J-cuts, escalating rhythmic momentum"
        )
    else:
        edit_style = "balanced Walter Murch emotional cuts with 0.4s J/L-cut audio overlaps and continuous room tone"

    texture = primary["texture_hint"]
    if secondary is not None:
        texture = f"{primary['texture_hint']}; subtle influence of {secondary['label'].lower()}"

    craft = CompiledCraftRules(
        lens_package=lens_pkg,
        lighting_philosophy=light_phil,
        film_texture=texture,
        coverage_grammar=coverage,
        editorial_style=edit_style,
        target_avg_shot_sec=avg_shot,
        max_wpm=max_wpm,
    )

    return TasteSignature(
        mood_presets=chosen_moods,
        story_archetype=story_archetype,
        user_vibe_note=user_vibe_note.strip(),
        dials=tuned_dials,
        compiled_craft_rules=craft,
    )


def find_reaction_chip(chip_or_note: str) -> dict[str, Any] | None:
    clean = (chip_or_note or "").strip().lower()
    for ch in TASTE_REACTION_CHIPS:
        if ch["id"] == clean or ch["label"].lower() == clean:
            return ch
    return None


class TasteProfileStore:
    """Atomic JSON store persisting `DirectorTasteProfile` per actor under `<data_dir>/cinema`."""

    def __init__(self, cinema_dir: Path | str) -> None:
        self.cinema_dir = Path(cinema_dir).resolve()
        self.cinema_dir.mkdir(parents=True, exist_ok=True)
        self.file_path = self.cinema_dir / "taste_profiles.json"

    def _read_all(self) -> dict[str, Any]:
        if not self.file_path.is_file():
            return {}
        try:
            raw = json.loads(self.file_path.read_text(encoding="utf-8"))
            return raw if isinstance(raw, dict) else {}
        except Exception:
            return {}

    def _write_all(self, payload: dict[str, Any]) -> None:
        self.cinema_dir.mkdir(parents=True, exist_ok=True)
        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=self.cinema_dir,
                prefix=".taste_",
                suffix=".tmp",
                delete=False,
            ) as tmp:
                json.dump(payload, tmp, indent=2, sort_keys=True)
                tmp.flush()
                os.fsync(tmp.fileno())
                tmp_path = Path(tmp.name)
            os.replace(tmp_path, self.file_path)
            tmp_path = None
        finally:
            if tmp_path is not None and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    def get_profile(self, actor_id: str = "default") -> DirectorTasteProfile:
        key = (actor_id or "default").strip() or "default"
        all_profiles = self._read_all()
        entry = all_profiles.get(key)
        if isinstance(entry, dict):
            try:
                return DirectorTasteProfile.model_validate(entry)
            except Exception:
                pass
        return DirectorTasteProfile(actor_id=key)

    def record_treatment_choice(
        self,
        *,
        actor_id: str = "default",
        mood_preset: str,
        archetype: str,
        dials: VibeDials,
    ) -> DirectorTasteProfile:
        prof = self.get_profile(actor_id)
        if mood_preset:
            prof.preferred_mood_presets[mood_preset] = (
                prof.preferred_mood_presets.get(mood_preset, 0) + 1
            )
        if archetype and archetype != "auto":
            prof.preferred_archetypes[archetype] = prof.preferred_archetypes.get(archetype, 0) + 1
        alpha = 0.30
        prof.dial_ema = VibeDials(
            scale=_clamp01(prof.dial_ema.scale * (1 - alpha) + dials.scale * alpha),
            pacing=_clamp01(prof.dial_ema.pacing * (1 - alpha) + dials.pacing * alpha),
            mood_contrast=_clamp01(
                prof.dial_ema.mood_contrast * (1 - alpha) + dials.mood_contrast * alpha
            ),
            dialogue_density=_clamp01(
                prof.dial_ema.dialogue_density * (1 - alpha) + dials.dialogue_density * alpha
            ),
        )
        prof.reactions_count += 1
        prof.updated_at = datetime.now(UTC).isoformat()
        self._save_profile(prof)
        return prof

    def record_reaction(
        self,
        *,
        actor_id: str = "default",
        event: TasteReactionEvent,
        active_signature: TasteSignature,
    ) -> DirectorTasteProfile:
        prof = self.get_profile(actor_id)
        chip = find_reaction_chip(event.chip_or_note)
        dials = active_signature.dials

        if chip is not None:
            deltas = chip.get("dial_delta", {})
            nudged = VibeDials(
                scale=_clamp01(dials.scale + float(deltas.get("scale", 0.0))),
                pacing=_clamp01(dials.pacing + float(deltas.get("pacing", 0.0))),
                mood_contrast=_clamp01(
                    dials.mood_contrast + float(deltas.get("mood_contrast", 0.0))
                ),
                dialogue_density=_clamp01(
                    dials.dialogue_density + float(deltas.get("dialogue_density", 0.0))
                ),
            )
        else:
            nudged = infer_dials_from_vibe_note(dials, event.chip_or_note)

        alpha = 0.25
        prof.dial_ema = VibeDials(
            scale=_clamp01(prof.dial_ema.scale * (1 - alpha) + nudged.scale * alpha),
            pacing=_clamp01(prof.dial_ema.pacing * (1 - alpha) + nudged.pacing * alpha),
            mood_contrast=_clamp01(
                prof.dial_ema.mood_contrast * (1 - alpha) + nudged.mood_contrast * alpha
            ),
            dialogue_density=_clamp01(
                prof.dial_ema.dialogue_density * (1 - alpha) + nudged.dialogue_density * alpha
            ),
        )

        for mp in active_signature.mood_presets[:2]:
            if event.polarity == "love":
                prof.preferred_mood_presets[mp] = prof.preferred_mood_presets.get(mp, 0) + 2

        note_txt = event.chip_or_note.strip()
        if note_txt:
            if event.polarity == "love" and note_txt not in prof.love_notes:
                prof.love_notes = (prof.love_notes + [note_txt])[-12:]
            elif event.polarity == "reject" and note_txt not in prof.avoid_notes:
                prof.avoid_notes = (prof.avoid_notes + [note_txt])[-12:]

        prof.reactions_count += 1
        prof.updated_at = datetime.now(UTC).isoformat()
        self._save_profile(prof)
        return prof

    def _save_profile(self, profile: DirectorTasteProfile) -> None:
        all_profiles = self._read_all()
        all_profiles[profile.actor_id] = profile.model_dump()
        self._write_all(all_profiles)


__all__ = [
    "CINEMA_MOOD_CATALOG",
    "STORY_ARCHETYPE_CATALOG",
    "TASTE_REACTION_CHIPS",
    "TasteProfileStore",
    "compile_taste_signature",
    "find_reaction_chip",
    "get_mood_preset",
    "infer_dials_from_vibe_note",
]
