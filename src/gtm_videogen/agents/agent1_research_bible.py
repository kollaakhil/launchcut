"""Agent 1 (F14): Research & CreativeBible Rubric-Gated DeepAgent.

Owns:
- Pillar 1 (research fast / conceptDivergence + creativeBible flagship)
- Pillar 2 (Runtime brief-driven CreativeBible — zero hard templates)
- Pillar 3 (4 structurally orthogonal ConceptPitches + diversity scoring + anti-cliché)
- Pillar 6 (CreativeHistory read & non-repetition enforcement)
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, BaseMessage, ToolMessage
from langchain_core.tools import BaseTool, tool
from pydantic import ValidationError

from gtm_videogen.agents.rubric_utils import (
    EventCallback,
    is_default_unscripted_mock_model,
    is_offline_provider_bundle,
    make_tool_driven_grader_model,
    read_workspace_json,
    resolve_live_chat_model,
    run_rubric_deep_agent,
    write_workspace_json,
)
from gtm_videogen.core.job_registry import CancellationToken
from gtm_videogen.media.captions import BUNDLED_FONTS
from gtm_videogen.models.schemas import (
    BeatSpec,
    CameraGrammar,
    ConceptPitch,
    CreativeBible,
    CreativeHistory,
    MusicBrief,
    NarrativeStructure,
    ResearchBundle,
    RunInput,
    SonicIdentity,
    TypographyAndLayout,
    VisualLanguage,
    VoiceCasting,
)
from gtm_videogen.providers.mocks import ScriptedChatModel
from gtm_videogen.rules.creative_history import CreativeHistoryStore, format_previously_explored
from gtm_videogen.rules.diversity import diversity_scores, select_diverse

AGENT1_STAGE = "agent1_research_bible"

AGENT1_SYSTEM_PROMPT = """You are the Research & Creative Direction DeepAgent.
Your job is to research the brand, review any prior creative history so nothing is repeated,
ideate structurally orthogonal concept pitches, score their diversity, and synthesize a
bespoke, brief-driven CreativeBible with zero clichés or rigid templates.

Always use your tools in sequence:
1. scrape_brand_tool to gather verified brand facts, colors, and product photos.
2. load_creative_history_tool to inspect previously explored creative directions.
3. score_concept_diversity_tool with orthogonal ConceptPitch candidates.
4. save_creative_bible_tool with the validated bespoke CreativeBible.
"""

AGENT1_GRADER_SYSTEM_PROMPT = """You are the Research & CreativeBible Rubric Grader.
Always invoke `verify_research_and_bible_tool` to inspect the persisted workspace JSON
artifacts directly before issuing your structured GraderResponse verdict.
"""

AGENT1_RUBRIC = """Evaluate the Research & CreativeBible workspace artifacts:
1. grounded_research: Verified facts/claims and brand metadata are present and grounded.
2. complete_creative_bible: Every CreativeBible section is populated and aligned with the brief.
3. zero_cliches: No phrase from anti_cliche_list appears in concept_thesis or beat purposes.
4. concept_diversity: Orthogonal concept pitches were generated and scored for diversity.
5. history_non_repetition: The CreativeBible does not repeat a prior arc and visual world.
"""

# Distinct creative exploration dimensions used to synthesize brief-grounded orthogonal pitches
# during offline mock runs (rotated dynamically against CreativeHistory).
_EXPLORATION_AXES: tuple[dict[str, Any], ...] = (
    {
        "arc_name": "craft_provenance_reveal",
        "title_suffix": "Origin & Tactile Proof",
        "visual_world": (
            "sunlit workshop table with natural grain textures and crisp detail framing"
        ),
        "lighting_style": "directional morning daylight with gentle shadow falloff",
        "texture_and_grade": "clean editorial film grain with true-to-life brand color fidelity",
        "camera_style": "deliberate locked-off macro frames paired with smooth lateral glides",
        "allowed_moves": ["lateral_glide", "macro_focus_pull", "static_editorial"],
        "lens_range_mm": "35-85mm",
        "pacing_curve": "measured hook opening building to crisp proof cuts",
        "font_family": "Playfair Display",
        "caption_treatment": "editorial_lower_third",
        "end_card_layout": "split_brand_mark",
        "motion_feel": "composed_editorial",
        "audio_strategy": "intimate voiceover with subtle environmental room tone",
        "music_genre": "acoustic chamber minimalism",
        "music_mood": "grounded and confident",
        "instrumentation": ["nylon guitar", "soft brushed percussion", "warm upright bass"],
        "tempo_feel": "relaxed mid-tempo",
        "voice_casting": {
            "gender": "female",
            "age": "adult",
            "accent": "neutral",
            "energy": "warm",
            "pace": "measured",
            "delivery_style": "documentary narrator with understated authority",
        },
        "transition_style": "clean editorial cuts with occasional j_cut audio lead-ins",
        "ending_strategy": "cta",
        "emotional_register": "quiet mastery and earned trust",
    },
    {
        "arc_name": "kinetic_contrast_flip",
        "title_suffix": "Before-After Friction Flip",
        "visual_world": "high-contrast architectural studio with bold geometric light blocks",
        "lighting_style": "sculpted high-key studio lighting with crisp specular edges",
        "texture_and_grade": "modern commercial digital clarity with punchy brand accents",
        "camera_style": "snappy rhythmic whip-pans and tight eye-level framing",
        "allowed_moves": ["whip_pan", "snap_zoom", "handheld_micro_push"],
        "lens_range_mm": "24-50mm",
        "pacing_curve": "rapid scroll-stop opening settling into clear benefit payoff",
        "font_family": "Oswald",
        "caption_treatment": "bold_center_punch",
        "end_card_layout": "centered_minimal",
        "motion_feel": "snappy_kinetic",
        "audio_strategy": "brisk voiceover punctuated by crisp tactile foley hits",
        "music_genre": "minimalist percussive pulse",
        "music_mood": "energetic and focused",
        "instrumentation": ["crisp rimshot", "plucked synth bass", "muted hand percussion"],
        "tempo_feel": "brisk and driving",
        "voice_casting": {
            "gender": "male",
            "age": "young",
            "accent": "neutral",
            "energy": "energetic",
            "pace": "brisk",
            "delivery_style": "sharp direct-response strategist",
        },
        "transition_style": "rhythmic match cuts and crisp foley transitions",
        "ending_strategy": "punchline",
        "emotional_register": "decisive momentum and relief",
    },
    {
        "arc_name": "customer_dialogue_proof",
        "title_suffix": "Real-World Ritual Witness",
        "visual_world": (
            "authentic everyday kitchen and living space in soft golden afternoon light"
        ),
        "lighting_style": "warm practical window light with organic bounce fill",
        "texture_and_grade": "naturalistic 35mm documentary warmth with soft highlights",
        "camera_style": "intimate shoulder-height observational handheld movement",
        "allowed_moves": ["observational_handheld", "gentle_orbit", "over_shoulder_frame"],
        "lens_range_mm": "28-65mm",
        "pacing_curve": "conversational cadence with breathing room on proof lines",
        "font_family": "Roboto",
        "caption_treatment": "clean_bottom",
        "end_card_layout": "stacked_offer_card",
        "motion_feel": "organic_ease",
        "audio_strategy": "conversational voiceover layered over ambient domestic foley",
        "music_genre": "warm indie folk underscore",
        "music_mood": "welcoming and candid",
        "instrumentation": ["felt piano", "warm acoustic strums", "light shaker"],
        "tempo_feel": "conversational mid-tempo",
        "voice_casting": {
            "gender": "female",
            "age": "young",
            "accent": "neutral",
            "energy": "conversational",
            "pace": "natural",
            "delivery_style": "candid friend sharing a genuine recommendation",
        },
        "transition_style": "natural L-cuts and soft crossfades between moments",
        "ending_strategy": "cta",
        "emotional_register": "relatable warmth and social proof",
    },
    {
        "arc_name": "analytical_spec_spotlight",
        "title_suffix": "Precision Metric Breakdown",
        "visual_world": (
            "clean gallery plinth with neutral stone backdrop and razor-sharp macro focus"
        ),
        "lighting_style": "soft overhead diffused softbox with controlled rim separation",
        "texture_and_grade": (
            "crisp high-acutance studio finish with restrained monochrome neutrals"
        ),
        "camera_style": "precision overhead flat-lay and controlled arc tracking",
        "allowed_moves": ["overhead_flatlay", "controlled_arc", "micro_dolly"],
        "lens_range_mm": "50-100mm",
        "pacing_curve": "structured problem-evidence-verdict cadence",
        "font_family": "JetBrains Mono",
        "caption_treatment": "structured_spec_callout",
        "end_card_layout": "grid_metric_summary",
        "motion_feel": "precision_linear",
        "audio_strategy": "authoritative voiceover with minimal sparse tonal bed",
        "music_genre": "ambient modular minimalism",
        "music_mood": "precise and forward-looking",
        "instrumentation": ["soft analog pad", "clean marimba pulse", "subtle sub tone"],
        "tempo_feel": "steady and measured",
        "voice_casting": {
            "gender": "male",
            "age": "mature",
            "accent": "neutral",
            "energy": "authoritative",
            "pace": "measured",
            "delivery_style": "calm technical expert presenting verifiable facts",
        },
        "transition_style": "clean hard cuts aligned to metric reveals",
        "ending_strategy": "logo_linger",
        "emotional_register": "clarity and rational conviction",
    },
)


@dataclass
class Agent1Result:
    """Output of Agent 1: ResearchBundle, orthogonal ConceptPitches, and CreativeBible."""

    research: ResearchBundle
    concepts: list[ConceptPitch]
    creative_bible: CreativeBible
    rubric_evaluations: list[dict[str, Any]] = field(default_factory=list)
    status: str = "satisfied"
    workspace_dir: Path | None = None


def _synthesize_concepts_from_brief(
    run_input: RunInput,
    research: ResearchBundle,
    history: CreativeHistory,
) -> list[ConceptPitch]:
    """Generate structurally orthogonal brief-grounded ConceptPitches avoiding prior history."""
    lead_fact = (
        research.verified_claims[0]
        if research.verified_claims
        else (
            research.product_facts[0]
            if research.product_facts
            else f"{run_input.brand_name} delivers on {run_input.goal}"
        )
    )
    used_arcs = {e.arc_name.strip().lower() for e in history.entries}
    used_worlds = {e.visual_world.strip().lower() for e in history.entries}
    n_target = max(1, min(run_input.num_concepts, 8))

    # Order axes so unexplored ones come first
    ordered_axes = sorted(
        enumerate(_EXPLORATION_AXES),
        key=lambda item: (
            1 if item[1]["arc_name"].lower() in used_arcs else 0,
            1 if item[1]["visual_world"].lower() in used_worlds else 0,
            item[0],
        ),
    )

    pitches: list[ConceptPitch] = []
    seen_arcs = set(used_arcs)
    seen_worlds = set(used_worlds)
    variant_round = len(history.entries)
    for idx in range(n_target):
        cycle, axis_pos = divmod(idx, len(ordered_axes))
        _, axis = ordered_axes[axis_pos]
        variant_num = variant_round + cycle + 1
        arc = axis["arc_name"]
        world = axis["visual_world"]
        if arc.lower() in seen_arcs:
            candidate_num = variant_num
            while f"{arc}_v{candidate_num}".lower() in seen_arcs:
                candidate_num += 1
            arc = f"{arc}_v{candidate_num}"
        if world.lower() in seen_worlds:
            candidate_num = variant_num
            while f"{world} (fresh angle {candidate_num})".lower() in seen_worlds:
                candidate_num += 1
            world = f"{world} (fresh angle {candidate_num})"

        seen_arcs.add(arc.lower())
        seen_worlds.add(world.lower())

        title = (
            f"{run_input.brand_name}: {axis['title_suffix']} (Angle {cycle + 1})"
            if cycle > 0
            else f"{run_input.brand_name}: {axis['title_suffix']}"
        )
        thesis = (
            f"Prove {run_input.goal} from angle {cycle + 1} by highlighting that {lead_fact}."
            if cycle > 0
            else f"Prove {run_input.goal} by highlighting that {lead_fact}."
        )
        sonic_strategy = (
            f"{axis['audio_strategy']} with contrast pass {cycle + 1}"
            if cycle > 0
            else axis["audio_strategy"]
        )
        emotional_register = (
            f"{axis['emotional_register']} — bold variant {cycle + 1}"
            if cycle > 0
            else axis["emotional_register"]
        )

        pitches.append(
            ConceptPitch(
                id=f"concept_{idx + 1}",
                title=title,
                thesis=thesis,
                narrative_arc=arc,
                visual_world=world,
                sonic_strategy=sonic_strategy,
                emotional_register=emotional_register,
            )
        )
    return pitches


def _build_bible_from_concept(
    run_input: RunInput,
    research: ResearchBundle,
    concept: ConceptPitch,
    history: CreativeHistory,
) -> CreativeBible:
    """Synthesize a bespoke CreativeBible grounded in the selected concept and brand palette."""
    base_arc = concept.narrative_arc.split("_v")[0]
    axis = next(
        (a for a in _EXPLORATION_AXES if a["arc_name"] == base_arc),
        _EXPLORATION_AXES[0],
    )

    # Ensure unique thesis if history already used the exact same thesis
    used_hooks = {e.hook_opening.strip().lower() for e in history.entries}
    thesis = concept.thesis
    if thesis.strip().lower() in used_hooks:
        thesis = f"{concept.title} — {thesis}"

    palette = (
        list(research.brand_colors) if research.brand_colors else ["#1D3557", "#F1FAEE", "#E63946"]
    )
    target_dur = float(run_input.target_duration_sec)
    beat_count = 3 if target_dur <= 12.0 else 4
    each = round(target_dur / beat_count, 2)

    beats = [
        BeatSpec(
            name="opening_hook",
            purpose=f"Establish {run_input.brand_name} visual world and arrest attention",
            duration_hint_sec=each,
            emotional_shift="curiosity",
        ),
        BeatSpec(
            name="grounded_proof",
            purpose="Demonstrate verified product craft and tangible customer benefit",
            duration_hint_sec=each,
            emotional_shift="conviction",
        ),
    ]
    if beat_count >= 4:
        beats.append(
            BeatSpec(
                name="tactile_detail",
                purpose="Reinforce quality with concrete proof point or customer voice",
                duration_hint_sec=each,
                emotional_shift="desire",
            )
        )
    beats.append(
        BeatSpec(
            name="closing_action",
            purpose=f"Deliver clear brand sign-off aligned to {run_input.goal}",
            duration_hint_sec=max(1.5, round(target_dur - each * (len(beats)), 2)),
            emotional_shift="action",
        )
    )

    music_brief: MusicBrief | None = None
    if run_input.audio_mode in ("vo", "both"):
        music_brief = MusicBrief(
            genre=axis["music_genre"],
            mood=axis["music_mood"],
            instrumentation=list(axis["instrumentation"]),
            tempo_feel=axis["tempo_feel"],
            energy_curve="subtle opening swell, ducked under speech, clean resolve",
            prompt=(
                f"{axis['music_genre']} underscore, {axis['music_mood']}, "
                f"{', '.join(axis['instrumentation'])}, {axis['tempo_feel']}"
            ),
            tags=[axis["music_genre"], axis["music_mood"]],
        )

    font_family = axis["font_family"]
    if font_family not in BUNDLED_FONTS:
        font_family = "Roboto"

    return CreativeBible(
        concept_thesis=thesis,
        narrative_structure=NarrativeStructure(
            arc_name=concept.narrative_arc,
            beats=beats,
            ending_strategy=axis["ending_strategy"],
        ),
        visual_language=VisualLanguage(
            world=concept.visual_world,
            lighting_style=axis["lighting_style"],
            color_palette=palette,
            texture_and_grade=axis["texture_and_grade"],
        ),
        camera_grammar=CameraGrammar(
            dominant_style=axis["camera_style"],
            allowed_moves=list(axis["allowed_moves"]),
            lens_range_mm=axis["lens_range_mm"],
            pacing_curve=axis["pacing_curve"],
        ),
        typography_and_layout=TypographyAndLayout(
            font_family=font_family,
            caption_treatment=axis["caption_treatment"],
            end_card_layout=axis["end_card_layout"],
            motion_feel=axis["motion_feel"],
        ),
        sonic_identity=SonicIdentity(
            audio_strategy=axis["audio_strategy"],
            music_brief=music_brief,
            voice_casting=VoiceCasting.model_validate(axis["voice_casting"]),
            transition_style=axis["transition_style"],
        ),
        anti_cliche_list=[
            "Are you tired of",
            "POV:",
            "game-changer",
            "unsolicited dark neon void",
            "generic stock handshake",
        ],
    )


def _build_agent1_tools(
    *,
    run_input: RunInput,
    providers: Any,
    history_store: CreativeHistoryStore,
    workspace: Path,
    cancel: CancellationToken | None,
) -> tuple[list[BaseTool], list[BaseTool]]:
    """Create Agent 1 worker tools and its deterministic grader verification tool."""

    @tool("scrape_brand_tool")
    async def scrape_brand_tool(
        brand_url: str = "",
        inspiration_url: str = "",
    ) -> str:
        """Scrape brand facts, brand colors, and product photos via BrandScraper."""
        target_url = brand_url.strip() or run_input.brand_url
        insp_url = inspiration_url.strip() or run_input.inspiration_url
        scraper = providers.scraper
        bundle: ResearchBundle = await scraper.scrape_brand(
            target_url,
            inspiration_url=insp_url,
            cancel=cancel,
            brand_name=run_input.brand_name,
        )
        # Merge any explicit user-supplied facts so they are always grounded
        for fact in run_input.facts:
            if fact.strip() and fact not in bundle.verified_claims:
                bundle.verified_claims.append(fact.strip())
        write_workspace_json(workspace, "brand_research.json", bundle)
        return json.dumps(
            {
                "status": "ok",
                "brand_name": bundle.brand_name,
                "verified_claims_count": len(bundle.verified_claims),
                "product_facts_count": len(bundle.product_facts),
                "product_photos_count": len(bundle.product_photos),
                "brand_colors": bundle.brand_colors,
            }
        )

    @tool("load_creative_history_tool")
    def load_creative_history_tool(brand_url: str = "") -> str:
        """Load the last 10 CreativeFingerprints for brand_url to prevent repetition."""
        target_url = brand_url.strip() or run_input.brand_url
        history = history_store.get_history(target_url)
        write_workspace_json(workspace, "creative_history.json", history)
        return json.dumps(
            {
                "brand_url": history.brand_url,
                "count": len(history.entries),
                "formatted": format_previously_explored(history),
                "entries": [e.model_dump(mode="json") for e in history.entries],
            }
        )

    @tool("score_concept_diversity_tool")
    def score_concept_diversity_tool(concepts_json: str) -> str:
        """Score orthogonal ConceptPitch candidates for diversity and persist concepts.json."""
        raw_list = json.loads(concepts_json)
        pitches = [ConceptPitch.model_validate(item) for item in raw_list]
        scores = diversity_scores(pitches)
        for pitch, score in zip(pitches, scores, strict=True):
            pitch.diversity_score = score
        selected = select_diverse(pitches, k=len(pitches))
        write_workspace_json(
            workspace,
            "concepts.json",
            [p.model_dump(mode="json") for p in selected],
        )
        return json.dumps(
            {
                "status": "ok",
                "count": len(selected),
                "top_concept_id": selected[0].id if selected else None,
                "diversity_scores": {p.id: p.diversity_score for p in selected},
            }
        )

    @tool("save_creative_bible_tool")
    def save_creative_bible_tool(creative_bible_json: str) -> str:
        """Validate a Pydantic CreativeBible and persist creative_bible.json to workspace."""
        raw = json.loads(creative_bible_json)
        bible = CreativeBible.model_validate(raw)
        write_workspace_json(workspace, "creative_bible.json", bible)
        return json.dumps(
            {
                "status": "ok",
                "arc_name": bible.narrative_structure.arc_name,
                "beats": len(bible.narrative_structure.beats),
                "font_family": bible.typography_and_layout.font_family,
            }
        )

    @tool("verify_research_and_bible_tool")
    def verify_research_and_bible_tool() -> str:
        """Deterministically verify brand_research.json, concepts.json, and creative_bible.json."""
        raw_research = read_workspace_json(workspace, "brand_research.json")
        raw_concepts = read_workspace_json(workspace, "concepts.json")
        raw_bible = read_workspace_json(workspace, "creative_bible.json")
        history = history_store.get_history(run_input.brand_url)

        criteria: list[dict[str, Any]] = []

        # 1. Grounded research
        if raw_research is None:
            criteria.append(
                {
                    "name": "grounded_research",
                    "passed": False,
                    "gap": "brand_research.json is missing; call scrape_brand_tool first.",
                }
            )
            research_obj: ResearchBundle | None = None
        else:
            research_obj = ResearchBundle.model_validate(raw_research)
            fact_total = len(research_obj.verified_claims) + len(research_obj.product_facts)
            if fact_total < 1:
                criteria.append(
                    {
                        "name": "grounded_research",
                        "passed": False,
                        "gap": "ResearchBundle contains zero verified claims or product facts.",
                    }
                )
            else:
                criteria.append({"name": "grounded_research", "passed": True})

        # 2. Complete CreativeBible
        if raw_bible is None:
            criteria.append(
                {
                    "name": "complete_creative_bible",
                    "passed": False,
                    "gap": "creative_bible.json is missing; call save_creative_bible_tool.",
                }
            )
            bible_obj: CreativeBible | None = None
        else:
            bible_obj = CreativeBible.model_validate(raw_bible)
            missing_parts: list[str] = []
            if not bible_obj.concept_thesis.strip():
                missing_parts.append("concept_thesis")
            if not bible_obj.narrative_structure.arc_name.strip():
                missing_parts.append("narrative_structure.arc_name")
            if not bible_obj.narrative_structure.beats:
                missing_parts.append("narrative_structure.beats")
            if not bible_obj.visual_language.world.strip():
                missing_parts.append("visual_language.world")
            if not bible_obj.camera_grammar.dominant_style.strip():
                missing_parts.append("camera_grammar.dominant_style")
            if not bible_obj.sonic_identity.audio_strategy.strip():
                missing_parts.append("sonic_identity.audio_strategy")
            if missing_parts:
                criteria.append(
                    {
                        "name": "complete_creative_bible",
                        "passed": False,
                        "gap": (
                            f"CreativeBible missing required sections: {', '.join(missing_parts)}."
                        ),
                    }
                )
            else:
                criteria.append({"name": "complete_creative_bible", "passed": True})

        # 3. Zero clichés & zero hard templates
        if bible_obj is None:
            criteria.append(
                {
                    "name": "zero_cliches",
                    "passed": False,
                    "gap": "Cannot check clichés without creative_bible.json.",
                }
            )
        else:
            text_blob = " ".join(
                [
                    bible_obj.concept_thesis,
                    bible_obj.narrative_structure.arc_name,
                    bible_obj.visual_language.world,
                    *(b.purpose for b in bible_obj.narrative_structure.beats),
                ]
            ).lower()
            cliche_hits = [
                c
                for c in bible_obj.anti_cliche_list
                if c.strip() and c.strip().lower() in text_blob
            ]
            if cliche_hits:
                criteria.append(
                    {
                        "name": "zero_cliches",
                        "passed": False,
                        "gap": f"CreativeBible violates anti_cliche_list: {cliche_hits}.",
                    }
                )
            else:
                criteria.append({"name": "zero_cliches", "passed": True})

        # 4. Concept diversity
        if not raw_concepts or not isinstance(raw_concepts, list):
            criteria.append(
                {
                    "name": "concept_diversity",
                    "passed": False,
                    "gap": "concepts.json missing; call score_concept_diversity_tool.",
                }
            )
        else:
            pitches = [ConceptPitch.model_validate(item) for item in raw_concepts]
            if len(pitches) < run_input.num_concepts:
                criteria.append(
                    {
                        "name": "concept_diversity",
                        "passed": False,
                        "gap": (
                            f"Expected at least {run_input.num_concepts} concept pitches, "
                            f"got {len(pitches)}."
                        ),
                    }
                )
            elif len(pitches) > 1 and max(p.diversity_score for p in pitches) <= 0.0:
                criteria.append(
                    {
                        "name": "concept_diversity",
                        "passed": False,
                        "gap": (
                            "Concept pitches have zero diversity score; "
                            "generate orthogonal concepts."
                        ),
                    }
                )
            else:
                criteria.append({"name": "concept_diversity", "passed": True})

        # 5. History non-repetition
        if bible_obj is None:
            criteria.append(
                {
                    "name": "history_non_repetition",
                    "passed": False,
                    "gap": "Cannot check history non-repetition without creative_bible.json.",
                }
            )
        elif history.entries:
            repeated = any(
                e.arc_name.strip().lower() == bible_obj.narrative_structure.arc_name.strip().lower()
                and e.visual_world.strip().lower()
                == bible_obj.visual_language.world.strip().lower()
                for e in history.entries
            )
            if repeated:
                criteria.append(
                    {
                        "name": "history_non_repetition",
                        "passed": False,
                        "gap": (
                            "CreativeBible repeats a previously explored "
                            f"(arc_name, visual_world): {bible_obj.narrative_structure.arc_name!r}."
                        ),
                    }
                )
            else:
                criteria.append({"name": "history_non_repetition", "passed": True})
        else:
            criteria.append({"name": "history_non_repetition", "passed": True})

        all_passed = all(c["passed"] for c in criteria)
        failed_names = [c["name"] for c in criteria if not c["passed"]]
        return json.dumps(
            {
                "passed": all_passed,
                "result": "satisfied" if all_passed else "needs_revision",
                "explanation": (
                    "ResearchBundle, orthogonal concepts, and CreativeBible passed all checks."
                    if all_passed
                    else f"Failed criteria: {', '.join(failed_names)}."
                ),
                "criteria": criteria,
            }
        )

    worker_tools: list[BaseTool] = [
        scrape_brand_tool,
        load_creative_history_tool,
        score_concept_diversity_tool,
        save_creative_bible_tool,
    ]
    grader_tools: list[BaseTool] = [verify_research_and_bible_tool]
    return worker_tools, grader_tools


def _make_offline_agent1_model(
    *,
    run_input: RunInput,
    history_store: CreativeHistoryStore,
    workspace: Path,
) -> ScriptedChatModel:
    """Create a genuine tool-calling offline ScriptedChatModel for Agent 1."""
    step = {"idx": 0}

    def _responder(messages: list[BaseMessage]) -> dict[str, Any] | str:
        # Inspect which tools have already been executed in this thread
        seen_tools = {getattr(m, "name", "") for m in messages if isinstance(m, ToolMessage)}
        step["idx"] += 1

        if "scrape_brand_tool" not in seen_tools:
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": f"tc_a1_scrape_{step['idx']}",
                        "name": "scrape_brand_tool",
                        "args": {
                            "brand_url": run_input.brand_url,
                            "inspiration_url": run_input.inspiration_url or "",
                        },
                        "type": "tool_call",
                    }
                ],
            }

        if "load_creative_history_tool" not in seen_tools:
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": f"tc_a1_hist_{step['idx']}",
                        "name": "load_creative_history_tool",
                        "args": {"brand_url": run_input.brand_url},
                        "type": "tool_call",
                    }
                ],
            }

        if "score_concept_diversity_tool" not in seen_tools:
            raw_research = read_workspace_json(workspace, "brand_research.json") or {}
            research = ResearchBundle.model_validate(raw_research)
            history = history_store.get_history(run_input.brand_url)
            pitches = _synthesize_concepts_from_brief(run_input, research, history)
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": f"tc_a1_div_{step['idx']}",
                        "name": "score_concept_diversity_tool",
                        "args": {
                            "concepts_json": json.dumps(
                                [p.model_dump(mode="json") for p in pitches]
                            )
                        },
                        "type": "tool_call",
                    }
                ],
            }

        if "save_creative_bible_tool" not in seen_tools:
            raw_research = read_workspace_json(workspace, "brand_research.json") or {}
            raw_concepts = read_workspace_json(workspace, "concepts.json") or []
            research = ResearchBundle.model_validate(raw_research)
            pitches = [ConceptPitch.model_validate(item) for item in raw_concepts]
            history = history_store.get_history(run_input.brand_url)
            chosen = (
                pitches[0]
                if pitches
                else _synthesize_concepts_from_brief(run_input, research, history)[0]
            )
            bible = _build_bible_from_concept(run_input, research, chosen, history)
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": f"tc_a1_bible_{step['idx']}",
                        "name": "save_creative_bible_tool",
                        "args": {
                            "creative_bible_json": bible.model_dump_json(),
                        },
                        "type": "tool_call",
                    }
                ],
            }

        return "Completed brand research, concept divergence, and bespoke CreativeBible synthesis."

    return ScriptedChatModel(responder=_responder, model_name="agent1-offline-research-bible")


def _make_agent1_prepare_messages(workspace: Path) -> Any:
    """Sync raw JSON responses into workspace if a custom scripted model skips tool calls."""

    def _prepare(messages: list[AnyMessage]) -> list[AnyMessage]:
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and isinstance(msg.content, str):
                text = msg.content.strip()
                if text.startswith("{") and text.endswith("}"):
                    try:
                        payload = json.loads(text)
                    except json.JSONDecodeError:
                        break
                    if isinstance(payload, dict):
                        if (
                            "research" in payload
                            and read_workspace_json(workspace, "brand_research.json") is None
                        ):
                            try:
                                bundle = ResearchBundle.model_validate(payload["research"])
                                write_workspace_json(workspace, "brand_research.json", bundle)
                            except (ValidationError, ValueError, TypeError):
                                pass
                        if "concepts" in payload and isinstance(payload["concepts"], list):
                            try:
                                pitches = [
                                    ConceptPitch.model_validate(c) for c in payload["concepts"]
                                ]
                                write_workspace_json(
                                    workspace,
                                    "concepts.json",
                                    [p.model_dump(mode="json") for p in pitches],
                                )
                            except (ValidationError, ValueError, TypeError):
                                pass
                        if "creative_bible" in payload:
                            try:
                                bible = CreativeBible.model_validate(payload["creative_bible"])
                                write_workspace_json(workspace, "creative_bible.json", bible)
                            except (ValidationError, ValueError, TypeError):
                                pass
                break
        return messages

    return _prepare


async def run_agent1_research_bible(
    *,
    run_input: RunInput,
    providers: Any,
    history_store: CreativeHistoryStore | None = None,
    work_dir: Path | None = None,
    run_id: str = "run",
    event_cb: EventCallback | None = None,
    cancel: CancellationToken | None = None,
    main_model: BaseChatModel | str | None = None,
    grader_model: BaseChatModel | str | None = None,
    max_iterations: int = 3,
    raise_on_failure: bool = True,
) -> Agent1Result:
    """Run Agent 1 (Research & CreativeBible DeepAgent) with RubricMiddleware gating."""
    if cancel is not None:
        cancel.raise_if_cancelled()

    base_dir = (
        Path(work_dir).resolve()
        if work_dir is not None
        else Path(tempfile.mkdtemp(prefix="gtm_a1_"))
    )
    workspace = base_dir / "agent1_workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    resolved_history = history_store or CreativeHistoryStore(base_dir / "creative_history.json")
    worker_tools, grader_tools = _build_agent1_tools(
        run_input=run_input,
        providers=providers,
        history_store=resolved_history,
        workspace=workspace,
        cancel=cancel,
    )

    is_offline_mode = run_input.mock or is_offline_provider_bundle(providers)

    resolved_main = main_model
    if resolved_main is None:
        provider_chat = getattr(providers, "chat_model", None)
        if provider_chat is not None and not is_default_unscripted_mock_model(provider_chat):
            resolved_main = provider_chat
        elif is_offline_mode:
            resolved_main = _make_offline_agent1_model(
                run_input=run_input,
                history_store=resolved_history,
                workspace=workspace,
            )
        else:
            resolved_main = resolve_live_chat_model("creative_bible", providers=providers)

    if grader_model is not None:
        resolved_grader = grader_model
    elif is_offline_mode or isinstance(resolved_main, ScriptedChatModel):
        resolved_grader = make_tool_driven_grader_model(
            "verify_research_and_bible_tool",
            model_name="agent1-rubric-grader",
        )
    else:
        resolved_grader = resolve_live_chat_model("claims_check", providers=providers)

    prompt = (
        f"Research brand '{run_input.brand_name}' ({run_input.brand_url}) "
        f"for goal: {run_input.goal}. "
        f"Target duration: {run_input.target_duration_sec}s, aspect: {run_input.aspect}, "
        f"audio_mode: {run_input.audio_mode}, concepts: {run_input.num_concepts}."
    )

    outcome = await run_rubric_deep_agent(
        stage=AGENT1_STAGE,
        run_id=run_id,
        prompt=prompt,
        rubric=AGENT1_RUBRIC,
        main_model=resolved_main,
        grader_model=resolved_grader,
        system_prompt=AGENT1_SYSTEM_PROMPT,
        grader_system_prompt=AGENT1_GRADER_SYSTEM_PROMPT,
        tools=worker_tools,
        grader_tools=grader_tools,
        max_iterations=max_iterations,
        event_cb=event_cb,
        cancel=cancel,
        prepare_messages_for_grader=_make_agent1_prepare_messages(workspace),
        raise_on_failure=raise_on_failure,
    )

    raw_research = read_workspace_json(workspace, "brand_research.json")
    raw_concepts = read_workspace_json(workspace, "concepts.json") or []
    raw_bible = read_workspace_json(workspace, "creative_bible.json")

    research = (
        ResearchBundle.model_validate(raw_research)
        if raw_research is not None
        else ResearchBundle(brand_url=run_input.brand_url, brand_name=run_input.brand_name)
    )
    concepts = [ConceptPitch.model_validate(item) for item in raw_concepts]
    if raw_bible is not None:
        creative_bible = CreativeBible.model_validate(raw_bible)
    else:
        fallback_concept = (
            concepts[0]
            if concepts
            else _synthesize_concepts_from_brief(
                run_input, research, resolved_history.get_history(run_input.brand_url)
            )[0]
        )
        creative_bible = _build_bible_from_concept(
            run_input, research, fallback_concept, resolved_history.get_history(run_input.brand_url)
        )

    return Agent1Result(
        research=research,
        concepts=concepts,
        creative_bible=creative_bible,
        rubric_evaluations=outcome.evaluations,
        status=outcome.status,
        workspace_dir=workspace,
    )


__all__ = [
    "AGENT1_RUBRIC",
    "AGENT1_STAGE",
    "Agent1Result",
    "run_agent1_research_bible",
]
