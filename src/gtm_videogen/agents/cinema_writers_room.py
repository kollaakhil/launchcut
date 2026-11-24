"""Cinema Studio Writers' Room: Treatments + Hierarchical Parallel JSON Compiler + Table Read.

Implements:
  1. Stage 1 Pitching: 3 distinct Director Treatments + real Look-Test Keyframes ($0.005 or $0 mock).
  2. Hierarchical Compilation (solving Trap #1 LLM token truncation):
     - Call 1 (Showrunner): `WorldVault` (Characters, Dual-Wall Locations, Stateful Props) + `Therefore/But` Scene Spine.
     - Parallel Scene Cinematographers: 4-6 richly blocked `CinemaShot` JSONs per scene.
  3. Pixar "Table Read First" Pass (solving Trap #2 clipped dialogue):
     - Synthesizes character TTS lines *before* locking shot durations or keyframes.
  4. Plain-English & Taste-Chip Ripple Director (`apply_cinema_taste_or_note`).
"""

from __future__ import annotations

import asyncio
import math
import re
import struct
import wave
from pathlib import Path
from typing import Any
from uuid import uuid4

from gtm_videogen.agents.director_studio import (
    _generate_flux_image,
    _render_mock_keyframe_jpeg,
)
from gtm_videogen.agents.stitch_engine import (
    _call_openrouter_json,
    scrape_optional_url,
)
from gtm_videogen.core.config import Settings, get_settings
from gtm_videogen.core.job_registry import CancellationToken
from gtm_videogen.core.media_store import MediaStore
from gtm_videogen.media._subprocess import probe_media
from gtm_videogen.models.cinema_compiler import (
    autofix_cinema_project_gate,
    choose_smart_anchor_strategy,
    evaluate_story_rubric,
)
from gtm_videogen.models.cinema_project import (
    CardinalZones,
    CausalLink,
    CharacterAsset,
    CharacterVoiceCasting,
    CinemaAspect,
    CinemaProject,
    CinemaScene,
    CinemaShot,
    ColorScriptBeat,
    DialogueCueSpec,
    DirectorTasteProfile,
    FrameStateSpec,
    LocationAsset,
    MotionDeltaBeat,
    PropAsset,
    SceneWorldState,
    ShotCameraSpec,
    ShotLightingSpec,
    ShotPropInstance,
    ShotScale,
    ShotSoundStems,
    SpatialZonesSpec,
    StoryArchetype,
    TasteReactionEvent,
    TasteSignature,
    TreatmentPitch,
    VibeDials,
    WorkflowMode,
    WorldVault,
)
from gtm_videogen.models.schemas import VoiceCue
from gtm_videogen.providers.audio import AudioProvider
from gtm_videogen.providers.gemini_cli import (
    gemini_generate_image,
    gemini_generate_tts,
    is_gemini_cli_ready,
)
from gtm_videogen.rules.cinema_taste import (
    TasteProfileStore,
    compile_taste_signature,
    find_reaction_chip,
    get_mood_preset,
)
from gtm_videogen.rules.voice_casting import VOICE_CATALOG


def _extract_subject_noun(idea: str, brand_name: str | None = None) -> str:
    if brand_name and brand_name.strip():
        return brand_name.strip()
    words = [
        w.strip(".,!?:;\"'()") for w in idea.strip().split() if len(w.strip(".,!?:;\"'()")) >= 3
    ]
    stop = {
        "the",
        "and",
        "for",
        "with",
        "that",
        "from",
        "into",
        "about",
        "where",
        "when",
        "who",
        "finds",
        "discovers",
        "makes",
        "film",
        "movie",
        "story",
        "short",
    }
    picked = [w for w in words if w.lower() not in stop][:4]
    return " ".join(picked).title() if picked else "The Signal"


def _cast_distinct_voices(count: int = 3) -> list[CharacterVoiceCasting]:
    """Pick distinct voices from `VOICE_CATALOG` so characters never share the same timbre."""
    preferred_order = (
        ("v_helena_f_mature_neutral", "Helena", "measured, grounded cinema realism"),
        ("v_arthur_m_mature_uk", "Arthur", "low gravitas, weathered restraint"),
        ("v_maya_f_adult_us_warm", "Maya", "intimate, warm, quiet honesty"),
        ("v_liam_m_adult_us", "Liam", "direct, natural, understated"),
    )
    out: list[CharacterVoiceCasting] = []
    catalog_ids = {v.voice_id: v for v in VOICE_CATALOG}
    for vid, vname, baseline in preferred_order:
        if len(out) >= count:
            break
        prof = catalog_ids.get(vid)
        out.append(
            CharacterVoiceCasting(
                provider="fal-ai/elevenlabs/tts/eleven-v3",
                voice_id=prof.voice_id if prof else vid,
                voice_name=prof.name if prof else vname,
                delivery_baseline=baseline,
            )
        )
    return out


def _synthesize_mock_dialogue_wav(
    text: str,
    out_path: Path,
    *,
    voice_index: int = 0,
) -> tuple[Path, float]:
    """Synthesize a syllable-cadenced warm vocal-formant WAV offline ($0) for Table Read & Leica Reel."""
    words = [w for w in text.strip().split() if w]
    dur_sec = round(max(1.4, min(11.0, len(words) / 2.65)), 2)
    sr = 24000
    total_frames = int(sr * dur_sec)
    base_freq = (165.0, 122.0, 192.0, 138.0)[voice_index % 4]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    buf = bytearray()
    syllable_rate = max(2.2, len(words) * 1.45 / dur_sec)

    for i in range(total_frames):
        t = i / float(sr)
        # Natural breath envelope + syllable cadence modulation
        fade = min(1.0, t / 0.08, (dur_sec - t) / 0.12)
        syl_env = max(0.0, math.sin(math.pi * ((t * syllable_rate) % 1.0))) ** 0.65
        f0 = base_freq * (1.0 + 0.04 * math.sin(2.0 * math.pi * 1.3 * t))
        sample = (
            0.55 * math.sin(2.0 * math.pi * f0 * t)
            + 0.28 * math.sin(2.0 * math.pi * f0 * 2.0 * t)
            + 0.12 * math.sin(2.0 * math.pi * f0 * 3.0 * t)
        )
        amp = int(max(-32000, min(32000, sample * syl_env * fade * 5200.0)))
        buf.extend(struct.pack("<h", amp))

    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(bytes(buf))
    return out_path, dur_sec


# ---------------------------------------------------------------------------
# Stage 1: Pitch 3 Director Treatments + Visual Look-Test Stills
# ---------------------------------------------------------------------------


def _deterministic_treatments(
    idea: str,
    signature: TasteSignature,
    brand_name: str | None = None,
) -> list[TreatmentPitch]:
    clean_idea = idea.strip() or "A late-night discovery changes everything inside a quiet workshop"
    primary_mood = signature.mood_presets[0] if signature.mood_presets else "cinema_a24"
    m1 = get_mood_preset(primary_mood)
    m2 = get_mood_preset("cinema_noir" if primary_mood != "cinema_noir" else "cinema_imax")
    m3 = get_mood_preset("cinema_natgeo" if primary_mood != "cinema_natgeo" else "cinema_pastel")

    subj = _extract_subject_noun(clean_idea, brand_name)
    is_brand = bool(brand_name and brand_name.strip())

    return [
        TreatmentPitch(
            treatment_id="treat_intimate",
            title=f"{subj} · Quiet Turning Point",
            archetype="founder_origin_story" if is_brand else "intimate_character_arc",
            logline_hook=f"At 2:14 AM, a single physical detail forces an irreversible choice: {clean_idea.rstrip('.')}.",
            story_arc_summary=(
                "Opens inside an exhausted, tactile moment of friction -> an unexpected physical clue interrupts routine (BUT) "
                "-> quiet obsession unlocks the breakthrough (THEREFORE) -> lands on an earned, resonant final image."
            ),
            visual_and_sonic_vibe=f"{m1['label']}: {m1['lens_hint']}; {m1['score_hint']}.",
            mood_preset=m1["id"],
            dials=VibeDials(**m1["dials"]),
            palette_hex=list(m1["palette_hex"]),
            sample_look_prompt=(
                f"35mm cinema still frame ({m1['texture_hint']}): {clean_idea.rstrip('.')}. "
                f"Lighting: {m1['lighting_hint']}. Optics: {m1['lens_hint']}. "
                "Authentic lived-in atmosphere, zero readable text, zero watermarks."
            ),
            rubric_score=9.4,
        ),
        TreatmentPitch(
            treatment_id="treat_tension",
            title=f"{subj} · Ticking-Clock Nocturne",
            archetype="high_stakes_thriller",
            logline_hook=f"Every second narrows the margin for error as {clean_idea[0].lower() + clean_idea[1:] if len(clean_idea) > 1 else clean_idea}.",
            story_arc_summary=(
                "Drops straight into high-stakes motion -> primary plan fractures under pressure (BUT) "
                "-> decisive physical improvisation flips the odds (THEREFORE) -> stark, unforgettable payoff."
            ),
            visual_and_sonic_vibe=f"{m2['label']}: {m2['lens_hint']}; {m2['score_hint']}.",
            mood_preset=m2["id"],
            dials=VibeDials(**m2["dials"]),
            palette_hex=list(m2["palette_hex"]),
            sample_look_prompt=(
                f"Anamorphic noir cinema frame ({m2['texture_hint']}): {clean_idea.rstrip('.')}. "
                f"Lighting: {m2['lighting_hint']}. High contrast, wet reflections, zero readable text."
            ),
            rubric_score=9.1,
        ),
        TreatmentPitch(
            treatment_id="treat_sensory",
            title=f"{subj} · Tactile Visual Poem",
            archetype="visual_poem",
            logline_hook=f"Told almost entirely through macro touch, sound, and light: {clean_idea.rstrip('.')}.",
            story_arc_summary=(
                "Establishes sensory contrast in extreme macro detail -> friction disrupts harmony (BUT) "
                "-> physical transformation restores clarity (THEREFORE) -> poised, luminous hero stillness."
            ),
            visual_and_sonic_vibe=f"{m3['label']}: {m3['lens_hint']}; {m3['score_hint']}.",
            mood_preset=m3["id"],
            dials=VibeDials(**m3["dials"]),
            palette_hex=list(m3["palette_hex"]),
            sample_look_prompt=(
                f"90mm macro cinema still ({m3['texture_hint']}): {clean_idea.rstrip('.')}. "
                f"Lighting: {m3['lighting_hint']}. Razor-thin bokeh, golden dust motes, zero text."
            ),
            rubric_score=8.9,
        ),
    ]


async def pitch_cinema_treatments(
    *,
    idea: str,
    aspect_ratio: CinemaAspect = "16:9",
    target_duration_sec: int = 90,
    workflow_mode: WorkflowMode = "guided",
    mood_presets: list[str] | None = None,
    story_archetype: StoryArchetype = "auto",
    user_vibe_note: str = "",
    dials: VibeDials | None = None,
    source_url: str | None = None,
    brand_name: str | None = None,
    brand_tagline: str | None = None,
    uploaded_reference_paths: list[str] | None = None,
    video_model_id: str = "auto",
    llm_model_id: str = "",
    budget_cap_usd: float = 15.0,
    actor_id: str = "default",
    project_id: str | None = None,
    media_dir: Path | None = None,
    taste_store: TasteProfileStore | None = None,
    settings: Settings | None = None,
    mock: bool = False,
    cancel: CancellationToken | None = None,
) -> CinemaProject:
    """Create a `CinemaProject` with 3 visual Director Treatments and Look-Test stills."""
    resolved = settings or get_settings()
    pid = project_id or f"cin_{uuid4().hex[:10]}"
    t_store = taste_store or TasteProfileStore(resolved.data_dir / "cinema")
    taste_prof: DirectorTasteProfile = t_store.get_profile(actor_id)

    scraped_summary = ""
    detected_brand = brand_name
    if source_url and source_url.strip() and not mock:
        scraped = await scrape_optional_url(source_url.strip(), settings=resolved, cancel=cancel)
        scraped_summary = str(scraped.get("summary") or "")[:900]
        if not detected_brand and scraped.get("brand_name"):
            detected_brand = str(scraped["brand_name"])

    signature = compile_taste_signature(
        mood_presets=mood_presets,
        story_archetype=story_archetype,
        user_vibe_note=user_vibe_note,
        dials_override=dials,
        taste_profile=taste_prof,
    )

    treatments = _deterministic_treatments(idea, signature, brand_name=detected_brand)

    if not mock:
        sys_prompt = (
            "You are an Academy-caliber Film Showrunner and Creative Director. "
            "Given a 1-line idea and user taste profile, pitch 3 genuinely distinct 1-3 minute film treatments. "
            "Avoid all commercial clichés ('in a world', 'game-changer'). Every story arc MUST use 'BUT' / 'THEREFORE' "
            "causality and concrete physical storytelling. Return JSON: "
            '{"treatments": [{"title": str, "archetype": "mystery_reveal"|"high_stakes_thriller"|"intimate_character_arc"|"founder_origin_story"|"visual_poem", '
            '"logline_hook": str, "story_arc_summary": str, "visual_and_sonic_vibe": str, '
            '"mood_preset": "cinema_a24"|"cinema_imax"|"cinema_noir"|"cinema_pastel"|"cinema_grit"|"cinema_natgeo"|"cinema_neon"|"cinema_anime", '
            '"palette_hex": [str, str, str], "sample_look_prompt": str, "rubric_score": float}]}'
        )
        usr_prompt = (
            f"1-Line Idea: {idea}\n"
            f"Brand / Context: {detected_brand or 'Pure Narrative Cinema'} {scraped_summary}\n"
            f"Preferred Moods: {signature.mood_presets}\n"
            f"Archetype Preference: {story_archetype}\n"
            f"User Vibe Note: {user_vibe_note or 'None'}\n"
            f"Past Director Avoid Notes: {taste_prof.avoid_notes[:4]}"
        )
        llm_doc = await _call_openrouter_json(
            system_prompt=sys_prompt,
            user_prompt=usr_prompt,
            settings=resolved,
            llm_model_id=llm_model_id,
            max_tokens=2600,
            temperature=0.72,
            cancel=cancel,
        )
        raw_list = llm_doc.get("treatments") if isinstance(llm_doc, dict) else None
        if isinstance(raw_list, list) and len(raw_list) >= 3:
            parsed_t: list[TreatmentPitch] = []
            for idx, item in enumerate(raw_list[:3]):
                if not isinstance(item, dict):
                    continue
                mp = str(item.get("mood_preset") or signature.mood_presets[0])
                mp_meta = get_mood_preset(mp)
                arch_raw = str(item.get("archetype") or "intimate_character_arc")
                if arch_raw not in (
                    "mystery_reveal",
                    "high_stakes_thriller",
                    "intimate_character_arc",
                    "founder_origin_story",
                    "visual_poem",
                ):
                    arch_raw = "intimate_character_arc"
                pal = item.get("palette_hex")
                clean_pal = (
                    [str(c) for c in pal[:3] if str(c).startswith("#")]
                    if isinstance(pal, list)
                    else []
                ) or list(mp_meta["palette_hex"])
                parsed_t.append(
                    TreatmentPitch(
                        treatment_id=f"treat_{idx + 1}",
                        title=str(item.get("title") or f"Treatment {idx + 1}")[:72],
                        archetype=arch_raw,  # type: ignore[arg-type]
                        logline_hook=str(item.get("logline_hook") or idea)[:260],
                        story_arc_summary=str(item.get("story_arc_summary") or "")[:420],
                        visual_and_sonic_vibe=str(
                            item.get("visual_and_sonic_vibe") or mp_meta["tagline"]
                        )[:240],
                        mood_preset=mp_meta["id"],
                        dials=VibeDials(**mp_meta["dials"]),
                        palette_hex=clean_pal,
                        sample_look_prompt=str(
                            item.get("sample_look_prompt") or treatments[idx].sample_look_prompt
                        )[:520],
                        rubric_score=round(
                            max(7.5, min(9.9, float(item.get("rubric_score") or 9.1))), 2
                        ),
                    )
                )
            if len(parsed_t) == 3:
                treatments = parsed_t

    # Render Look-Test Keyframe for each Treatment ("Show, Don't Ask")
    actual_usd = 0.0
    if media_dir is not None:
        look_dir = media_dir / "look_tests"
        look_dir.mkdir(parents=True, exist_ok=True)
        has_fal = bool(resolved.fal_key.strip()) and not mock and video_model_id != "mock"
        has_gem = is_gemini_cli_ready(resolved) and not mock and video_model_id != "mock"

        async def _render_one_look(tr: TreatmentPitch) -> float:
            out_jpg = look_dir / f"{tr.treatment_id}_look.jpg"
            if has_fal:
                try:
                    _, url, cost = await _generate_flux_image(
                        prompt=tr.sample_look_prompt,
                        aspect=aspect_ratio,
                        out_path=out_jpg,
                        settings=resolved,
                        anchor_url=None,
                        cancel=cancel,
                    )
                    tr.look_test_image_path = str(out_jpg)
                    tr.look_test_image_url = url
                    return float(cost)
                except Exception:
                    pass
            elif has_gem:
                gem_p = await gemini_generate_image(
                    prompt=tr.sample_look_prompt,
                    out_path=out_jpg,
                    aspect_ratio=aspect_ratio,
                    settings=resolved,
                    cancel=cancel,
                )
                if isinstance(gem_p, Path) and gem_p.is_file():
                    tr.look_test_image_path = str(out_jpg)
                    return 0.0

            _render_mock_keyframe_jpeg(
                title=tr.title,
                subtitle=tr.logline_hook[:105],
                footer=f"Look Test · {get_mood_preset(tr.mood_preset)['label']}",
                aspect=aspect_ratio,
                palette=tr.palette_hex,
                out_path=out_jpg,
                fonts_dir=resolved.fonts_dir,
            )
            tr.look_test_image_path = str(out_jpg)
            return 0.0

        costs = await asyncio.gather(*(_render_one_look(t) for t in treatments))
        actual_usd = round(sum(costs), 4)

    vault = WorldVault(taste_signature=signature)
    title = _extract_subject_noun(idea, detected_brand)
    project = CinemaProject(
        project_id=pid,
        title=title,
        idea=idea.strip(),
        source_url=source_url,
        scraped_context=scraped_summary,
        brand_name=detected_brand,
        brand_tagline=brand_tagline,
        uploaded_reference_paths=list(uploaded_reference_paths or []),
        aspect_ratio=aspect_ratio,
        target_duration_sec=target_duration_sec,
        workflow_mode=workflow_mode,
        video_model_id=video_model_id,
        llm_model_id=llm_model_id,
        budget_cap_usd=budget_cap_usd,
        treatments=treatments,
        selected_treatment_id=treatments[0].treatment_id if treatments else None,
        vault=vault,
        status="treatments_ready",
        status_message="3 visual Director Treatments ready — react to your favorite look to build the script.",
        progress_pct=15,
        progress_stage="1/4 · Treatments & Look-Tests Ready",
        actual_usd=actual_usd,
    )
    return project


# ---------------------------------------------------------------------------
# Stage 2: Hierarchical Master Script Compiler (Showrunner + Parallel Scenes)
# ---------------------------------------------------------------------------


def _plan_scene_count(target_duration_sec: int) -> int:
    if target_duration_sec <= 45:
        return 3
    if target_duration_sec <= 95:
        return 4
    if target_duration_sec <= 140:
        return 5
    return 6


def _build_deterministic_vault_and_spine(
    project: CinemaProject,
    treatment: TreatmentPitch,
) -> tuple[WorldVault, list[CinemaScene]]:
    """Construct a complete, coherent `WorldVault` and `Therefore/But` Scene Spine."""
    voices = _cast_distinct_voices(3)
    sig = compile_taste_signature(
        mood_presets=[treatment.mood_preset, *project.vault.taste_signature.mood_presets[:1]],
        story_archetype=treatment.archetype,
        user_vibe_note=project.vault.taste_signature.user_vibe_note,
        dials_override=treatment.dials,
    )
    mood_meta = get_mood_preset(treatment.mood_preset)
    pal = treatment.palette_hex or list(mood_meta["palette_hex"])

    hero_prop_name = (
        f"{project.brand_name} Prototype"
        if project.brand_name
        else "Brass & Quartz Signal Cylinder"
    )
    uploaded_refs = [p for p in project.uploaded_reference_paths if p and Path(p).is_file()]
    char_upload = uploaded_refs[0] if uploaded_refs and not project.brand_name else None
    prop_upload = (
        uploaded_refs[0]
        if (uploaded_refs and project.brand_name)
        else (uploaded_refs[1] if len(uploaded_refs) > 1 else None)
    )

    characters = [
        CharacterAsset(
            id="CHAR_PROTAG",
            element_tag="@Element1",
            name="Elena Rostova",
            role="protagonist",
            identity_lock="36-year-old woman with observant dark hazel eyes, subtle tiredness under eyes, shoulder-length dark waves tucked behind ears, natural skin pores",
            wardrobe_lock="charcoal wool work overshirt with rolled cuffs over muted slate henley, brushed steel analog watch on left wrist",
            posture_and_mannerisms="precise, economical hand movements; quiet intensity before speaking",
            voice_casting=voices[0],
            uploaded_ref_path=char_upload,
        ),
        CharacterAsset(
            id="CHAR_ALLY",
            element_tag="@Element2",
            name="Julian Mercer",
            role="deuteragonist",
            identity_lock="44-year-old man, salt-and-pepper close beard, wire-rim spectacles catching warm reflections, weathered expressive brow",
            wardrobe_lock="burnt-umber corduroy jacket over dark knit crewneck",
            posture_and_mannerisms="leans against doorframe with arms crossed, softens when listening",
            voice_casting=voices[1],
        ),
    ]

    locations = [
        LocationAsset(
            id="LOC_ATELIER",
            name="Nocturnal Signal Atelier",
            time_of_day_and_weather="2:00 AM, gentle rain streaking exterior glass",
            cardinal_zones=CardinalZones(
                A_north_primary="tall industrial steel-mullioned rain window overlooking distant city mist, scarred walnut workbench in foreground",
                B_south_reverse="warm cedar shelving lined with analog oscilloscopes, amber indicator bulbs, and heavy frosted-glass studio door",
                west_side="chalkboard schematic wall with pinned brass calipers",
                east_side="dark espresso bar corner with steaming ceramic cup",
            ),
            axis_180_rule="Camera remains on East/South-East side; Elena faces North/West at workbench, Julian enters from South door facing North",
            surface_textures=[
                "hand-oiled walnut workbench with fine caliper scratches",
                "rain condensation droplets sliding down dark window panes",
                "brushed anodized aluminum and warm glowing vacuum filaments",
            ],
            ambient_particles="subtle warm tungsten dust motes suspended in studio air",
            room_tone_sfx_prompt="quiet 2AM rain pattering on thick studio glass with faint warm analog transformer hum",
        ),
        LocationAsset(
            id="LOC_ROOFTOP",
            name="Misty Dawn Terrace",
            time_of_day_and_weather="first blue-gold dawn light after rain, wet stone",
            cardinal_zones=CardinalZones(
                A_north_primary="open panoramic horizon where warm amber dawn breaks through slate storm clouds above wet parapet",
                B_south_reverse="weathered brick studio penthouse doorway with warm interior tungsten spilling onto wet flagstones",
                west_side="copper weathervane glistening with rainwater",
                east_side="low glass skylight reflecting pale gold sky",
            ),
            axis_180_rule="Camera stays on East parapet side looking West/North across horizon",
            surface_textures=[
                "rain-darkened slate flagstones with mirror puddles",
                "oxidized copper railing with water beads",
                "damp wool fabric catching first golden rim light",
            ],
            ambient_particles="cool morning mist drifting gently across wet stone",
            room_tone_sfx_prompt="open dawn breeze, distant city awakening hush, single water drop from eaves",
        ),
    ]

    props = [
        PropAsset(
            id="PROP_HERO",
            element_tag="@Element3",
            name=hero_prop_name,
            is_brand_hero=bool(project.brand_name),
            geometry_and_scale="compact palm-sized precision instrument crafted from beveled optical glass and dark knurled metal",
            surface_textures="micro-machined dark metal edges, clear high-index glass core, subtle fingerprint oil sheen",
            states={
                "dormant": "unlit, cold, resting silently beside brass tools on dark felt",
                "flickering": "unstable faint amber pulse stuttering inside the optical core",
                "awakened": "steady luminous golden-amber harmonic core casting crisp caustic light ripples",
            },
            default_state="dormant",
            setup_scene_id="SC_01",
            payoff_scene_id=f"SC_0{_plan_scene_count(project.target_duration_sec)}",
            uploaded_ref_path=prop_upload,
        )
    ]

    n_scenes = _plan_scene_count(project.target_duration_sec)
    per_scene_sec = max(14, int(round(project.target_duration_sec / float(n_scenes))))

    arc_templates: list[tuple[str, CausalLink, str, str, int, str, str, list[str]]] = [
        (
            "The 2 AM Anomaly",
            "opening_hook",
            f"Elena is seconds from packing away the failed prototype ({project.idea[:80]}), when a silent tremor stirs on the workbench.",
            "weary resignation -> sharp curiosity",
            5,
            "LOC_ATELIER",
            "dormant",
            [pal[0], "#3A506B", pal[2]],
        ),
        (
            "Friction Under Pressure",
            "but",
            "Elena isolates the signal with Julian watching in skepticism, BUT the unstable pulse threatens to overload and vanish.",
            "curiosity -> urgent dread",
            7,
            "LOC_ATELIER",
            "flickering",
            [pal[0], pal[1], "#A8B8C8"],
        ),
        (
            "The Quiet赌 / Decisive Adjustment",
            "therefore",
            "THEREFORE Elena abandons the automated rig and calibrates the core by bare hand, locking the pure harmonic state.",
            "high tension -> hushed awe",
            9,
            "LOC_ATELIER",
            "awakened",
            ["#101318", pal[1], pal[2]],
        ),
        (
            "First Light Resonance",
            "therefore",
            "THEREFORE they carry the awakened instrument into first dawn light, seeing its real-world impact clearly for the first time.",
            "hushed awe -> grounded clarity",
            6,
            "LOC_ROOFTOP",
            "awakened",
            ["#1B2636", "#E6A145", "#F8EFE2"],
        ),
        (
            "The Unexpected Echo",
            "but",
            "BUT across the horizon an answering pulse confirms they were never alone in building this.",
            "clarity -> electric wonder",
            8,
            "LOC_ROOFTOP",
            "awakened",
            ["#151E2B", pal[1], "#FFF4E4"],
        ),
        (
            "Final Stillness",
            "therefore",
            "THEREFORE Elena sets the resonant core down in steady morning light—the beginning of a new chapter.",
            "wonder -> resonant peace",
            4,
            "LOC_ROOFTOP",
            "awakened",
            ["#1E2733", "#E8AA4C", "#FAF2E6"],
        ),
    ]

    scenes: list[CinemaScene] = []
    color_beats: list[ColorScriptBeat] = []
    prev_state = "dormant"

    for idx in range(n_scenes):
        tpl = arc_templates[idx] if idx < len(arc_templates) else arc_templates[-1]
        # Ensure final scene always lands on rooftop dawn payoff when n_scenes >= 3
        if idx == n_scenes - 1 and n_scenes >= 3:
            tpl = arc_templates[3]
        s_title, c_link, turn, shift, tension, loc_id, exit_prop_st, sc_pal = tpl
        sc_id = f"SC_{idx + 1:02d}"
        loc_obj = next((loc_item for loc_item in locations if loc_item.id == loc_id), locations[0])
        slug = (
            f"{'EXT' if 'ROOFTOP' in loc_id else 'INT'}. {loc_obj.name.upper()} - "
            f"{'DAWN' if 'ROOFTOP' in loc_id else 'NIGHT'}"
        )

        chars_here = ["CHAR_PROTAG"] if idx == 0 else ["CHAR_PROTAG", "CHAR_ALLY"]
        entry_st = SceneWorldState(
            characters_present=chars_here,
            character_positions={"CHAR_PROTAG": "at primary focal area"},
            prop_states={"PROP_HERO": prev_state},
        )
        exit_st = SceneWorldState(
            characters_present=chars_here,
            character_positions={"CHAR_PROTAG": "settled at final hero mark"},
            prop_states={"PROP_HERO": exit_prop_st},
        )
        prev_state = exit_prop_st

        scenes.append(
            CinemaScene(
                scene_id=sc_id,
                scene_number=idx + 1,
                title=f"Scene {idx + 1} · {s_title}",
                slugline=slug,
                location_id=loc_id,
                causal_link_from_prev="opening_hook" if idx == 0 else c_link,
                dramatic_turn=turn,
                emotional_shift=shift,
                tension_level=tension,
                target_duration_sec=per_scene_sec,
                color_palette_hex=sc_pal,
                lighting_arc=mood_meta["lighting_hint"],
                scene_entry_state=entry_st,
                scene_exit_state=exit_st,
            )
        )
        color_beats.append(
            ColorScriptBeat(
                scene_id=sc_id,
                palette_hex=sc_pal,
                lighting_arc=mood_meta["lighting_hint"],
                emotional_note=shift,
            )
        )

    vault = WorldVault(
        taste_signature=sig,
        characters=characters,
        locations=locations,
        props=props,
        color_script=color_beats,
        score_direction=mood_meta["score_hint"],
    )
    return vault, scenes


def _deterministic_scene_shots(
    scene: CinemaScene,
    vault: WorldVault,
    project: CinemaProject,
    *,
    is_final_scene: bool = False,
) -> list[CinemaShot]:
    """Build 3-5 richly choreographed CinemaShots for one Scene with zero lazy repetition."""
    loc = vault.get_location(scene.location_id)
    protag = vault.characters[0] if vault.characters else None
    ally = vault.characters[1] if len(vault.characters) > 1 else protag
    hero_prop = vault.props[0] if vault.props else None

    protag_id = protag.id if protag else "CHAR_PROTAG"
    ally_id = ally.id if ally else protag_id
    prop_id = hero_prop.id if hero_prop else "PROP_HERO"

    entry_p_state = scene.scene_entry_state.prop_states.get(prop_id, "dormant")
    exit_p_state = scene.scene_exit_state.prop_states.get(prop_id, entry_p_state)

    avg_s = vault.taste_signature.compiled_craft_rules.target_avg_shot_sec
    n_shots = max(3, min(5, int(round(scene.target_duration_sec / max(4.5, avg_s)))))
    base_dur = max(5, min(10, int(round(scene.target_duration_sec / float(n_shots)))))
    snapped_dur = 5 if base_dur <= 5 else (6 if base_dur <= 6 else (8 if base_dur <= 8 else 10))

    # Curated cinema coverage blueprint per shot slot inside the scene
    coverage_plan: list[tuple[ShotScale, str, str, int, str, str]] = [
        (
            "wide",
            "establishing_atmosphere",
            "A_north_primary",
            35,
            "slow 20cm lateral dolly glide",
            "looking_screen_left",
        ),
        (
            "extreme_close_up_insert",
            "tactile_prop_insert",
            "A_north_primary",
            90,
            " subtle 8cm macro push-in",
            "looking_screen_right",
        ),
        (
            "medium_close_up",
            "protagonist_speaking_turn",
            "A_north_primary",
            65,
            "gentle 8cm breathing push-in",
            "looking_screen_left",
        ),
        (
            "over_the_shoulder",
            "reverse_angle_reaction",
            "B_south_reverse",
            50,
            "locked cinema tripod with subtle rack focus",
            "looking_screen_right",
        ),
        (
            "medium",
            "scene_payoff_hold",
            "A_north_primary",
            50,
            "poised pull-back settling into hero tableau",
            "looking_screen_left",
        ),
    ]

    dialogue_bank: dict[int, list[tuple[str, str, str]]] = {
        1: [
            (
                protag_id,
                "Six months of dead frequencies. Pack the bench—we're done.",
                "off_screen_reaction",
            ),
            (protag_id, "Wait. Listen to the glass.", "on_screen_speaking"),
        ],
        2: [
            (
                ally_id,
                "If you push the gain past threshold, the core shatters.",
                "over_the_shoulder",
            ),
            (
                protag_id,
                "It isn't interference. Something is pacing the storm.",
                "on_screen_speaking",
            ),
        ],
        3: [
            (protag_id, "Kill the automated loop. I'm tuning it by hand.", "on_screen_speaking"),
            (ally_id, "The harmonic just locked. Zero drift.", "off_screen_reaction"),
        ],
        4: [
            (
                ally_id,
                "We built it in the dark... I didn't think it would feel this quiet.",
                "off_screen_reaction",
            ),
            (
                protag_id,
                "Every clear signal starts as noise someone refused to ignore.",
                "on_screen_speaking",
            ),
        ],
    }
    sc_lines = dialogue_bank.get(((scene.scene_number - 1) % 4) + 1, dialogue_bank[1])

    shots: list[CinemaShot] = []
    for idx in range(n_shots):
        scale, role, wall_zone, lens_mm, move, eyeline = coverage_plan[idx % len(coverage_plan)]
        sh_id = f"{scene.scene_id}_SH{idx + 1:02d}"
        cur_p_state = entry_p_state if idx < n_shots // 2 else exit_p_state
        end_p_state = exit_p_state if idx >= n_shots // 2 - 1 else cur_p_state

        chars_in_shot = [protag_id]
        if wall_zone == "B_south_reverse" and len(scene.scene_entry_state.characters_present) > 1:
            chars_in_shot = [ally_id, protag_id]

        dlg: DialogueCueSpec | None = None
        if vault.taste_signature.dials.dialogue_density >= 0.20:
            if idx == 2 and sc_lines:
                spk, ln, mode = sc_lines[-1]
                dlg = DialogueCueSpec(
                    speaker_id=spk,
                    line=ln,
                    delivery_mode=mode,  # type: ignore[arg-type]
                    emotion="hushed, grounded cinema realism",
                    pre_pause_sec=0.6,
                )
            elif idx == 0 and len(sc_lines) > 1:
                spk, ln, mode = sc_lines[0]
                dlg = DialogueCueSpec(
                    speaker_id=spk,
                    line=ln,
                    delivery_mode="off_screen_reaction" if scale == "wide" else mode,  # type: ignore[arg-type]
                    emotion="quiet, understated",
                    pre_pause_sec=0.7,
                )

        if scale == "extreme_close_up_insert":
            chars_in_shot = []
            start_comp = f"Extreme macro close-up on {prop_id} resting on {loc.surface_textures[0] if loc and loc.surface_textures else 'scarred walnut surface'}, {protag_id}'s fingertips entering lower edge"
            mid_action = f"{protag_id}'s fingertips gently rotate the knurled ring of {prop_id}; tactile light shifts across its beveled glass core"
            end_comp = f"Locked macro hero frame on {prop_id} in '{end_p_state}' state casting crisp caustic reflections across wood grain"
            fg_txt = f"{prop_id} in razor-sharp macro focus"
            mg_txt = f"{protag_id}'s hand in tactile contact"
        elif wall_zone == "B_south_reverse":
            start_comp = f"Over-the-shoulder reverse angle past {protag_id} in soft foreground-left toward {ally_id} standing near the opposite wall"
            mid_action = f"{ally_id} steps half a pace forward into warm practical light, expression shifting from doubt to quiet focus"
            end_comp = f"Balanced medium reverse shot holding on {ally_id}'s silent reaction as warm rim light catches damp wool"
            fg_txt = f"blurred shoulder silhouette of {protag_id} on left edge"
            mg_txt = f"{ally_id} framed cleanly against reverse interior architecture"
        else:
            start_comp = f"{scale.replace('_', ' ').title()} of {protag_id} positioned beside {prop_id} inside {scene.location_id}"
            mid_action = f"{protag_id} leans closer to {prop_id} ({scene.dramatic_turn[:95].rstrip('.')}); focus holds crisply on eyes"
            end_comp = f"Poised {scale.replace('_', ' ')} of {protag_id} illuminated by {scene.color_palette_hex[1]} key light beside {prop_id}"
            fg_txt = f"{prop_id} resting on primary surface in lower third"
            mg_txt = f"{protag_id} in three-quarter cinema framing"

        bg_txt = (
            loc.cardinal_zones.B_south_reverse
            if wall_zone == "B_south_reverse" and loc
            else (loc.cardinal_zones.A_north_primary if loc else "atmospheric cinema depth")
        )

        # Optional crisp A24/Brand closing title card on final shot when brand_name is set
        render_eng: Any = "ai_cinema"
        t_head: str | None = None
        t_sub: str | None = None
        if is_final_scene and idx == n_shots - 1 and project.brand_name:
            t_head = project.brand_name
            t_sub = project.brand_tagline or project.title

        shot_obj = CinemaShot(
            shot_id=sh_id,
            scene_id=scene.scene_id,
            index_in_scene=idx,
            duration_sec=snapped_dur,
            render_engine=render_eng,
            shot_scale=scale,
            coverage_role=role,
            anchor_strategy="start_and_end_locked",
            characters_in_shot=chars_in_shot,
            camera=ShotCameraSpec(
                facing_wall_zone=wall_zone,  # type: ignore[arg-type]
                lens_mm=lens_mm,
                t_stop="T1.8 shallow anamorphic bokeh"
                if lens_mm >= 65
                else "T2.2 cinema prime depth",
                angle="eye-level cinema framing",
                movement=move.strip(),
                eyeline_vector=eyeline,
            ),
            spatial_zones=SpatialZonesSpec(
                foreground=fg_txt,
                midground=mg_txt,
                background=bg_txt,
            ),
            textures_and_atmosphere=list(loc.surface_textures[:3])
            if loc
            else ["organic 35mm grain"],
            props_in_shot=[
                ShotPropInstance(
                    prop_id=prop_id,
                    state=cur_p_state,
                    screen_zone="foreground",
                    interaction="physical tabletop anchor",
                )
            ],
            lighting=ShotLightingSpec(
                key=f"motivated directional key light ({scene.lighting_arc})",
                fill_and_rim="cool atmospheric rim separation with rich shadow roll-off",
                palette_hex=list(scene.color_palette_hex),
            ),
            start_frame=FrameStateSpec(
                composition=start_comp,
                character_pose="grounded natural stillness at beat start",
                focal_plane="tack-sharp focus on primary subject",
                prop_states={prop_id: cur_p_state},
            ),
            motion_delta=[
                MotionDeltaBeat(
                    window=f"0.0-{snapped_dur / 2.0:.1f}s",
                    action=mid_action,
                    camera_delta=move.strip(),
                ),
                MotionDeltaBeat(
                    window=f"{snapped_dur / 2.0:.1f}-{float(snapped_dur):.1f}s",
                    action=f"Motion settles smoothly into final pose ({scene.emotional_shift})",
                    camera_delta="gentle settle into locked hero framing",
                ),
            ],
            end_frame=FrameStateSpec(
                composition=end_comp,
                character_pose="focused resonant hold",
                focal_plane="sharp cinema focus on eyes / hero detail",
                prop_states={prop_id: end_p_state},
            ),
            dialogue=dlg,
            sound_stems=ShotSoundStems(
                foley_prompt=f"subtle cloth rustle, tactile contact on {loc.surface_textures[0] if loc and loc.surface_textures else 'wood'}, {loc.room_tone_sfx_prompt[:60] if loc else 'room air'}",
                j_cut_lead_sec=0.35 if (idx == 0 and scene.scene_number > 1) else 0.0,
                l_cut_trail_sec=0.40 if dlg is not None else 0.0,
            ),
            title_card_headline=t_head,
            title_card_subline=t_sub,
        )
        shot_obj.anchor_strategy = choose_smart_anchor_strategy(shot_obj)  # type: ignore[assignment]
        shots.append(shot_obj)

    return shots


async def _compile_single_scene_llm(
    scene: CinemaScene,
    vault: WorldVault,
    project: CinemaProject,
    *,
    is_final_scene: bool,
    settings: Settings,
    cancel: CancellationToken | None = None,
) -> list[CinemaShot]:
    """Compile 3-6 rich `CinemaShot` JSONs for ONE scene with full LLM attention budget."""
    fallback_shots = _deterministic_scene_shots(
        scene, vault, project, is_final_scene=is_final_scene
    )
    loc = vault.get_location(scene.location_id)
    char_summary = [
        {"id": c.id, "name": c.name, "role": c.role, "wardrobe": c.wardrobe_lock}
        for c in vault.characters
    ]
    prop_summary = [{"id": p.id, "name": p.name, "states": p.states} for p in vault.props]

    sys_prompt = (
        "You are a Master Cinematographer and Script Supervisor. "
        "Compile 3 to 5 cinema shots in strict JSON for ONE scene of a film. "
        "Rules:\n"
        "1. Reference characters, locations, and props by their exact Vault IDs (e.g. CHAR_PROTAG, PROP_HERO) inside compositions and lists.\n"
        "2. Alternate shot_scale across shots ('wide', 'medium', 'medium_close_up', 'close_up', 'extreme_close_up_insert', 'over_the_shoulder').\n"
        "3. Use facing_wall_zone='A_north_primary' for primary angles and 'B_south_reverse' for 180-deg reverse reaction angles.\n"
        "4. Every shot MUST specify spatial_zones (foreground, midground, background), textures_and_atmosphere, start_frame, motion_delta, and end_frame.\n"
        "5. Strictly ZERO readable signs/words inside ai_cinema frames. Keep dialogue sparse, natural, and subtextual.\n"
        'Return JSON: {"shots": [CinemaShot-compatible dicts]}'
    )
    usr_prompt = (
        f"Film Idea: {project.idea}\n"
        f"Craft Rules: {vault.taste_signature.compiled_craft_rules.model_dump_json()}\n"
        f"Characters: {char_summary}\n"
        f"Location: {loc.model_dump_json() if loc else scene.location_id}\n"
        f"Props: {prop_summary}\n"
        f"Scene {scene.scene_number} ({scene.slugline}):\n"
        f"  Causal Link: {scene.causal_link_from_prev.upper()} -> {scene.dramatic_turn}\n"
        f"  Emotional Shift: {scene.emotional_shift} (Tension {scene.tension_level}/10)\n"
        f"  Target Scene Duration: {scene.target_duration_sec}s\n"
        f"  Entry State: {scene.scene_entry_state.model_dump_json()}\n"
        f"  Exit State: {scene.scene_exit_state.model_dump_json()}"
    )
    doc = await _call_openrouter_json(
        system_prompt=sys_prompt,
        user_prompt=usr_prompt,
        settings=settings,
        llm_model_id=project.llm_model_id,
        max_tokens=3600,
        temperature=0.62,
        cancel=cancel,
    )
    raw_shots = doc.get("shots") if isinstance(doc, dict) else None
    if not isinstance(raw_shots, list) or len(raw_shots) < 2:
        return fallback_shots

    parsed: list[CinemaShot] = []
    for idx, raw in enumerate(raw_shots[:6]):
        if not isinstance(raw, dict):
            continue
        fb = fallback_shots[min(idx, len(fallback_shots) - 1)]
        try:
            merged = {
                **fb.model_dump(),
                **raw,
                "shot_id": f"{scene.scene_id}_SH{idx + 1:02d}",
                "scene_id": scene.scene_id,
                "index_in_scene": idx,
            }
            sh = CinemaShot.model_validate(merged)
            sh.anchor_strategy = choose_smart_anchor_strategy(
                sh, is_chained_from_prev=(sh.anchor_strategy == "chained_from_prev_to_end")
            )  # type: ignore[assignment]
            parsed.append(sh)
        except Exception:
            parsed.append(
                fb.model_copy(
                    update={"shot_id": f"{scene.scene_id}_SH{idx + 1:02d}", "index_in_scene": idx}
                )
            )
    return parsed if len(parsed) >= 2 else fallback_shots


async def run_table_read_timing_pass(
    project: CinemaProject,
    audio_dir: Path,
    *,
    settings: Settings | None = None,
    mock: bool = False,
    cancel: CancellationToken | None = None,
) -> CinemaProject:
    """Pixar 'Table Read First' pass: synthesize dialogue TTS & lock exact shot durations before keyframing."""
    resolved = settings or get_settings()
    audio_dir.mkdir(parents=True, exist_ok=True)
    has_fal = bool(resolved.fal_key.strip()) and not mock and project.video_model_id != "mock"
    has_gem = is_gemini_cli_ready(resolved) and not mock and project.video_model_id != "mock"

    audio_provider = (
        AudioProvider(settings=resolved, media_store=MediaStore(audio_dir / "store"))
        if has_fal
        else None
    )

    char_index_map = {c.id: idx for idx, c in enumerate(project.vault.characters)}
    sem = asyncio.Semaphore(4)

    async def _synth_shot_line(shot: CinemaShot) -> float:
        if shot.dialogue is None or not shot.dialogue.line.strip():
            return 0.0
        async with sem:
            if cancel is not None:
                cancel.raise_if_cancelled()
            dlg = shot.dialogue
            ch = project.vault.get_character(dlg.speaker_id)
            voice_name = ch.voice_casting.voice_name if ch else "Helena"
            voice_idx = char_index_map.get(dlg.speaker_id, 0)
            out_wav = audio_dir / f"{shot.shot_id}_dlg.wav"

            if audio_provider is not None:
                try:
                    rendered = await audio_provider.generate_voice(
                        VoiceCue(
                            shot_id=shot.shot_id,
                            text=dlg.line.strip(),
                            voice=voice_name,
                        ),
                        cancel=cancel,
                    )
                    src_p = Path(rendered.path)
                    if src_p.is_file():
                        out_wav.write_bytes(src_p.read_bytes())
                        pr = await probe_media(
                            out_wav, ffprobe_bin=resolved.ffprobe_bin, cancel=cancel
                        )
                        dlg.synthesized_audio_path = str(out_wav)
                        dlg.actual_speech_sec = round(
                            float(pr.duration_sec or rendered.duration_sec or 2.5), 2
                        )
                        return float(rendered.usd)
                except Exception:
                    pass
            elif has_gem:
                gem_wav = await gemini_generate_tts(
                    text=dlg.line.strip(),
                    out_path=out_wav,
                    voice="Kore" if voice_idx % 2 == 0 else "Charon",
                    style=dlg.emotion,
                    settings=resolved,
                    cancel=cancel,
                )
                if isinstance(gem_wav, Path) and gem_wav.is_file():
                    pr = await probe_media(gem_wav, ffprobe_bin=resolved.ffprobe_bin, cancel=cancel)
                    dlg.synthesized_audio_path = str(gem_wav)
                    dlg.actual_speech_sec = round(float(pr.duration_sec or 2.5), 2)
                    return 0.0

            _, mock_dur = await asyncio.to_thread(
                _synthesize_mock_dialogue_wav,
                dlg.line,
                out_wav,
                voice_index=voice_idx,
            )
            dlg.synthesized_audio_path = str(out_wav)
            dlg.actual_speech_sec = mock_dur
            return 0.0

    usd_list = await asyncio.gather(*(_synth_shot_line(sh) for sh in project.all_shots()))
    project.actual_usd = round(project.actual_usd + sum(usd_list), 4)

    # Elastic shot duration fit around real measured speech duration
    autofix_cinema_project_gate(project)
    return project


async def compile_cinema_master_script(
    project: CinemaProject,
    project_dir: Path,
    *,
    treatment_id: str | None = None,
    actor_id: str = "default",
    taste_store: TasteProfileStore | None = None,
    settings: Settings | None = None,
    mock: bool = False,
    on_progress: Any | None = None,
    cancel: CancellationToken | None = None,
) -> CinemaProject:
    """Compile `WorldVault` + parallel `CinemaScene`/`CinemaShot` JSON + Table-Read TTS timings."""
    resolved = settings or get_settings()

    async def _report(msg: str, pct: int, stage: str) -> None:
        project.status_message = msg
        project.progress_pct = pct
        project.progress_stage = stage
        if on_progress is not None:
            res = on_progress(project)
            if asyncio.iscoroutine(res):
                await res

    chosen_id = treatment_id or project.selected_treatment_id
    treatment = next(
        (t for t in project.treatments if t.treatment_id == chosen_id),
        project.treatments[0]
        if project.treatments
        else _deterministic_treatments(project.idea, project.vault.taste_signature)[0],
    )
    project.selected_treatment_id = treatment.treatment_id

    t_store = taste_store or TasteProfileStore(resolved.data_dir / "cinema")
    t_store.record_treatment_choice(
        actor_id=actor_id,
        mood_preset=treatment.mood_preset,
        archetype=treatment.archetype,
        dials=treatment.dials,
    )

    await _report(
        f"Designing WorldVault (Characters, Dual-Wall Sets & Stateful Props) for '{treatment.title}'...",
        22,
        "2/4 · Building WorldVault & Scene Spine",
    )

    vault, scenes = _build_deterministic_vault_and_spine(project, treatment)

    # Optional Live Showrunner enrichment of Vault & Scene Spine
    if not mock:
        voices = _cast_distinct_voices(3)
        sys_p = (
            "You are a Film Showrunner & Production Designer. Build a cohesive WorldVault and causal Scene Spine "
            f"({len(scenes)} scenes) for a {project.target_duration_sec}s movie. "
            "Every scene after Scene 1 MUST connect via 'but' or 'therefore'. "
            "Every location MUST define distinct A_north_primary and B_south_reverse walls for 180-deg coverage. "
            'Return JSON: {"title": str, "characters": [{"id": "CHAR_PROTAG", "name": str, "role": "protagonist", "identity_lock": str, "wardrobe_lock": str}], '
            '"locations": [{"id": "LOC_01", "name": str, "time_of_day_and_weather": str, "A_north_primary": str, "B_south_reverse": str, "surface_textures": [str], "room_tone_sfx_prompt": str}], '
            '"props": [{"id": "PROP_HERO", "name": str, "geometry_and_scale": str, "surface_textures": str, "states": {"dormant": str, "awakened": str}}], '
            '"scenes": [{"title": str, "slugline": str, "location_id": str, "causal_link_from_prev": "opening_hook"|"but"|"therefore", "dramatic_turn": str, "emotional_shift": str, "tension_level": int}]}'
        )
        usr_p = (
            f"Idea: {project.idea}\n"
            f"Selected Treatment: {treatment.title} — {treatment.story_arc_summary}\n"
            f"Brand Context: {project.brand_name or 'None'} {project.scraped_context[:400]}\n"
            f"Vibe & Craft: {vault.taste_signature.compiled_craft_rules.model_dump_json()}"
        )
        doc = await _call_openrouter_json(
            system_prompt=sys_p,
            user_prompt=usr_p,
            settings=resolved,
            llm_model_id=project.llm_model_id,
            max_tokens=3200,
            temperature=0.65,
            cancel=cancel,
        )
        if isinstance(doc, dict):
            if doc.get("title"):
                project.title = str(doc["title"])[:80]
            raw_chars = doc.get("characters")
            if isinstance(raw_chars, list) and raw_chars:
                built_chars: list[CharacterAsset] = []
                for idx, rc in enumerate(raw_chars[:3]):
                    if isinstance(rc, dict) and rc.get("identity_lock") and rc.get("wardrobe_lock"):
                        cid = str(rc.get("id") or f"CHAR_{idx + 1:02d}").upper()
                        cid = re.sub(r"[^A-Z0-9_]", "_", cid)
                        built_chars.append(
                            CharacterAsset(
                                id=cid,
                                element_tag=f"@Element{min(3, idx + 1)}",
                                name=str(rc.get("name") or f"Character {idx + 1}"),
                                role="protagonist" if idx == 0 else "deuteragonist",
                                identity_lock=str(rc["identity_lock"]),
                                wardrobe_lock=str(rc["wardrobe_lock"]),
                                voice_casting=voices[idx % len(voices)],
                                uploaded_ref_path=vault.characters[0].uploaded_ref_path
                                if idx == 0 and vault.characters
                                else None,
                            )
                        )
                if built_chars:
                    vault.characters = built_chars

            raw_locs = doc.get("locations")
            if isinstance(raw_locs, list) and raw_locs:
                built_locs: list[LocationAsset] = []
                for idx, rl in enumerate(raw_locs[:3]):
                    if (
                        isinstance(rl, dict)
                        and rl.get("A_north_primary")
                        and rl.get("B_south_reverse")
                    ):
                        lid = re.sub(
                            r"[^A-Z0-9_]", "_", str(rl.get("id") or f"LOC_{idx + 1:02d}").upper()
                        )
                        built_locs.append(
                            LocationAsset(
                                id=lid,
                                name=str(rl.get("name") or f"Location {idx + 1}"),
                                time_of_day_and_weather=str(
                                    rl.get("time_of_day_and_weather") or "night"
                                ),
                                cardinal_zones=CardinalZones(
                                    A_north_primary=str(rl["A_north_primary"]),
                                    B_south_reverse=str(rl["B_south_reverse"]),
                                ),
                                surface_textures=[
                                    str(t)
                                    for t in (
                                        rl.get("surface_textures") or ["natural cinema surfaces"]
                                    )[:4]
                                ],
                                room_tone_sfx_prompt=str(
                                    rl.get("room_tone_sfx_prompt") or "subtle natural room air"
                                ),
                            )
                        )
                if built_locs:
                    vault.locations = built_locs

            raw_props = doc.get("props")
            if isinstance(raw_props, list) and raw_props:
                rp = raw_props[0]
                if isinstance(rp, dict) and isinstance(rp.get("states"), dict) and rp["states"]:
                    pid = re.sub(r"[^A-Z0-9_]", "_", str(rp.get("id") or "PROP_HERO").upper())
                    st_dict = {str(k): str(v) for k, v in rp["states"].items()}
                    def_st = next(iter(st_dict.keys()))
                    vault.props = [
                        PropAsset(
                            id=pid,
                            element_tag="@Element3",
                            name=str(rp.get("name") or "Hero Prop"),
                            is_brand_hero=bool(project.brand_name),
                            geometry_and_scale=str(
                                rp.get("geometry_and_scale") or vault.props[0].geometry_and_scale
                            ),
                            surface_textures=str(
                                rp.get("surface_textures") or vault.props[0].surface_textures
                            ),
                            states=st_dict,
                            default_state=def_st,
                            uploaded_ref_path=vault.props[0].uploaded_ref_path
                            if vault.props
                            else None,
                        )
                    ]

            raw_scenes = doc.get("scenes")
            if isinstance(raw_scenes, list) and len(raw_scenes) >= 2:
                valid_loc_ids = {loc_item.id for loc_item in vault.locations}
                p_keys = list(vault.props[0].states.keys()) if vault.props else ["default"]
                rebuilt_scenes: list[CinemaScene] = []
                for idx, rs in enumerate(raw_scenes[:6]):
                    if not isinstance(rs, dict):
                        continue
                    fb_sc = scenes[min(idx, len(scenes) - 1)]
                    lid = str(rs.get("location_id") or fb_sc.location_id)
                    if lid not in valid_loc_ids:
                        lid = vault.locations[0].id
                    clink = str(
                        rs.get("causal_link_from_prev") or ("opening_hook" if idx == 0 else "but")
                    )
                    if clink not in ("opening_hook", "but", "therefore", "meanwhile"):
                        clink = "but" if idx > 0 else "opening_hook"
                    p_entry = p_keys[0] if idx == 0 else p_keys[min(idx - 1, len(p_keys) - 1)]
                    p_exit = p_keys[min(idx, len(p_keys) - 1)]
                    pid_0 = vault.props[0].id if vault.props else "PROP_HERO"
                    rebuilt_scenes.append(
                        fb_sc.model_copy(
                            update={
                                "scene_id": f"SC_{idx + 1:02d}",
                                "scene_number": idx + 1,
                                "title": str(rs.get("title") or fb_sc.title)[:72],
                                "slugline": str(rs.get("slugline") or fb_sc.slugline)[:72],
                                "location_id": lid,
                                "causal_link_from_prev": "opening_hook" if idx == 0 else clink,
                                "dramatic_turn": str(
                                    rs.get("dramatic_turn") or fb_sc.dramatic_turn
                                )[:280],
                                "emotional_shift": str(
                                    rs.get("emotional_shift") or fb_sc.emotional_shift
                                )[:120],
                                "tension_level": max(
                                    1, min(10, int(rs.get("tension_level") or fb_sc.tension_level))
                                ),
                                "scene_entry_state": SceneWorldState(
                                    characters_present=[c.id for c in vault.characters[:2]],
                                    prop_states={pid_0: p_entry},
                                ),
                                "scene_exit_state": SceneWorldState(
                                    characters_present=[c.id for c in vault.characters[:2]],
                                    prop_states={pid_0: p_exit},
                                ),
                            }
                        )
                    )
                if len(rebuilt_scenes) >= 2:
                    scenes = rebuilt_scenes

    await _report(
        f"Compiling shot-by-shot spatial JSON in parallel across {len(scenes)} Scenes...",
        30,
        "2/4 · Parallel Scene Cinematography",
    )

    sem = asyncio.Semaphore(4)

    async def _build_scene_idx(i: int, sc: CinemaScene) -> list[CinemaShot]:
        async with sem:
            if cancel is not None:
                cancel.raise_if_cancelled()
            is_last = i == len(scenes) - 1
            if mock:
                return _deterministic_scene_shots(sc, vault, project, is_final_scene=is_last)
            return await _compile_single_scene_llm(
                sc,
                vault,
                project,
                is_final_scene=is_last,
                settings=resolved,
                cancel=cancel,
            )

    compiled_shot_lists = await asyncio.gather(
        *(_build_scene_idx(i, s) for i, s in enumerate(scenes))
    )
    for sc, shot_list in zip(scenes, compiled_shot_lists, strict=True):
        sc.shots = shot_list

    project.vault = vault
    project.scenes = scenes

    await _report(
        "Running Pixar Table-Read TTS pass & locking elastic shot durations...",
        38,
        "2/4 · Table-Read Dialogue Timing",
    )
    await run_table_read_timing_pass(
        project,
        project_dir / "audio",
        settings=resolved,
        mock=mock,
        cancel=cancel,
    )

    project.story_rubric = evaluate_story_rubric(project)
    project.status = "script_ready"
    project.progress_pct = 42
    project.progress_stage = "2/4 · Master JSON Script & Table Read Ready"
    project.status_message = (
        f"Compiled {len(project.scenes)} Scenes ({len(project.all_shots())} Shots, "
        f"{project.total_duration_sec()}s) · Story Rubric {project.story_rubric.overall_score}/10."
    )
    return project


# ---------------------------------------------------------------------------
# Plain-English & Taste-Chip Ripple Director
# ---------------------------------------------------------------------------


async def apply_cinema_taste_or_note(
    project: CinemaProject,
    project_dir: Path,
    *,
    note_or_chip: str,
    target_type: str = "project",
    target_id: str = "",
    polarity: str = "adjust",
    actor_id: str = "default",
    taste_store: TasteProfileStore | None = None,
    settings: Settings | None = None,
    mock: bool = False,
    cancel: CancellationToken | None = None,
) -> CinemaProject:
    """Translate a human Taste Chip or Plain-English note into Vault & Shot JSON ripple updates."""
    resolved = settings or get_settings()
    clean_note = note_or_chip.strip()
    if not clean_note:
        return project

    chip = find_reaction_chip(clean_note)
    eff_polarity: Any = chip["polarity"] if chip else polarity
    if eff_polarity not in ("love", "adjust", "reject"):
        eff_polarity = "adjust"

    ev = TasteReactionEvent(
        target_type=target_type
        if target_type in ("treatment", "vault", "scene", "shot", "cut")
        else "shot",  # type: ignore[arg-type]
        target_id=target_id,
        chip_or_note=clean_note,
        polarity=eff_polarity,
    )
    project.reaction_history.append(ev)

    t_store = taste_store or TasteProfileStore(resolved.data_dir / "cinema")
    updated_prof = t_store.record_reaction(
        actor_id=actor_id,
        event=ev,
        active_signature=project.vault.taste_signature,
    )

    # Recompile TasteSignature with dial delta if it's an adjustment
    cur_dials = project.vault.taste_signature.dials
    if chip and chip.get("dial_delta"):
        dd = chip["dial_delta"]
        new_dials = VibeDials(
            scale=max(0.0, min(1.0, round(cur_dials.scale + float(dd.get("scale", 0.0)), 3))),
            pacing=max(0.0, min(1.0, round(cur_dials.pacing + float(dd.get("pacing", 0.0)), 3))),
            mood_contrast=max(
                0.0,
                min(1.0, round(cur_dials.mood_contrast + float(dd.get("mood_contrast", 0.0)), 3)),
            ),
            dialogue_density=max(
                0.0,
                min(
                    1.0,
                    round(cur_dials.dialogue_density + float(dd.get("dialogue_density", 0.0)), 3),
                ),
            ),
        )
    else:
        new_dials = cur_dials

    combined_vibe = project.vault.taste_signature.user_vibe_note
    if eff_polarity != "love" and not chip:
        combined_vibe = (f"{combined_vibe}; {clean_note}").strip("; ")[:300]

    project.vault.taste_signature = compile_taste_signature(
        mood_presets=project.vault.taste_signature.mood_presets,
        story_archetype=project.vault.taste_signature.story_archetype,
        user_vibe_note=combined_vibe,
        dials_override=new_dials,
        taste_profile=updated_prof,
    )

    if eff_polarity == "love":
        if target_type == "shot" and target_id:
            found = project.find_shot(target_id)
            if found:
                found[1].hearted = True
        project.status_message = "Saved ♥ preference to your Director Taste Profile."
        return project

    low = (chip["instruction"] if chip else clean_note).lower()
    dirty_shots: list[CinemaShot] = []

    # Target scope resolution
    scoped_shots: list[CinemaShot] = []
    if target_type == "shot" and target_id:
        found = project.find_shot(target_id)
        if found:
            scoped_shots = [found[1]]
    elif target_type == "scene" and target_id:
        for sc in project.scenes:
            if sc.scene_id == target_id:
                scoped_shots = list(sc.shots)
                break
    if not scoped_shots:
        scoped_shots = project.all_shots()

    # Deterministic surgical ripple rules (fast, reliable, works offline & online)
    if (
        any(w in low for w in ("coat", "jacket", "shirt", "suit", "dress", "wearing", "wardrobe"))
        and project.vault.characters
        and target_type in ("project", "vault")
    ):
        project.vault.characters[0].wardrobe_lock = (
            f"{clean_note}, {project.vault.characters[0].wardrobe_lock}"
        )
        project.vault.characters[0].turnaround_sheet_path = None
        dirty_shots.extend(scoped_shots)

    for sh in scoped_shots:
        changed = False
        if any(w in low for w in ("intimate", "closer", "close-up", "warmer")):
            sh.camera.lens_mm = max(65, sh.camera.lens_mm)
            if sh.shot_scale in ("wide", "medium_wide"):
                sh.shot_scale = "medium_close_up"
            if "warm" in low:
                sh.lighting.key = f"warm 2900K golden tungsten key light ({sh.lighting.key})"
            changed = True
        if any(w in low for w in ("grand", "bigger", "wider", "epic", "architectural")):
            sh.camera.lens_mm = min(35, sh.camera.lens_mm)
            if sh.shot_scale == "medium":
                sh.shot_scale = "wide"
            changed = True
        if any(w in low for w in ("tension", "darker", "noir", "moody", "claustrophobic")):
            sh.lighting.key = f"high-contrast chiaroscuro shadow key ({sh.lighting.key})"
            changed = True
        if any(w in low for w in ("slower", "breathe", "linger")):
            sh.duration_sec = min(12, sh.duration_sec + 2)
            changed = True
        if any(w in low for w in ("faster", "kinetic", "tighter")):
            sh.duration_sec = max(4, sh.duration_sec - 1)
            changed = True
        if (
            any(
                w in low
                for w in ("less commercial", "cheesy", "silent", "no dialogue", "trim dialogue")
            )
            and sh.dialogue is not None
            and sh.shot_scale in ("wide", "extreme_close_up_insert")
        ):
            sh.dialogue = None
            changed = True
        if not changed and target_type == "shot":
            # Apply custom plain-English director note directly to the targeted shot's composition & action
            sh.start_frame.composition = f"{sh.start_frame.composition.rstrip('.')} ({clean_note})"
            if sh.motion_delta:
                sh.motion_delta[
                    0
                ].action = f"{sh.motion_delta[0].action.rstrip('.')} ({clean_note})"
            changed = True

        if changed:
            sh.start_keyframe_path = None
            sh.start_keyframe_url = None
            sh.end_keyframe_path = None
            sh.end_keyframe_url = None
            dirty_shots.append(sh)

    await run_table_read_timing_pass(
        project,
        project_dir / "audio",
        settings=resolved,
        mock=mock,
        cancel=cancel,
    )
    project.status_message = f"Applied taste direction ('{clean_note[:48]}') across {len(dirty_shots)} shot(s) & updated Taste Memory."
    return project


__all__ = [
    "apply_cinema_taste_or_note",
    "compile_cinema_master_script",
    "pitch_cinema_treatments",
    "run_table_read_timing_pass",
]
