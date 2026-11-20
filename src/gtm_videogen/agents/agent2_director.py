"""Agent 2 (F15): Script & ShotSheet Director Rubric-Gated DeepAgent.

Owns:
- Pillar 1 (scriptWrite flagship / sheetCompile fast / soundDesign text_model)
- Pillar 4 (open shot.beat, expanded TRANSITIONS, GraphicsTheme, bespoke SoundPlan)
- Pillar 5 (Pure Script Rules ONLY + originality & bible_coherence craft scoring)
- BUG-17 (attribute-matched select_voice with CreativeHistory repeat avoidance)
- BUG-18 (has_product_photos always passed into script gate)
- BUG-21 (fit_timings bounded water-filling enforcement)
- BUG-23 (non-overridable claim/safety errors)
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
from gtm_videogen.models.schemas import (
    ClaimCitation,
    CreativeBible,
    GraphicsTheme,
    MusicCue,
    QAReport,
    ResearchBundle,
    RunInput,
    Script,
    ScriptLine,
    SfxCue,
    Shot,
    ShotSheet,
    SoundPlan,
    VoiceCue,
)
from gtm_videogen.providers.mocks import ScriptedChatModel
from gtm_videogen.rules.anti_garbled_text import DEFAULT_NEGATIVE_PROMPT
from gtm_videogen.rules.creative_history import CreativeHistoryStore
from gtm_videogen.rules.script_gate import evaluate_script_gate
from gtm_videogen.rules.speakability import beat_fit_problem, estimate_spoken_duration
from gtm_videogen.rules.timing import MAX_SHOT_SEC, MIN_SHOT_SEC, fit_timings
from gtm_videogen.rules.voice_casting import select_voice

AGENT2_STAGE = "agent2_director"

AGENT2_SYSTEM_PROMPT = """You are the Script, ShotSheet & SoundPlan Director DeepAgent.
Translate the CreativeBible and verified ResearchBundle provided in the user prompt into a
production-ready Script, dynamic ShotSheet, and bespoke SoundPlan using ONLY pure script craft
rules and zero hardcoded visual/story/audio templates.

Always call your tools in order:
1. `fit_shot_timings_tool` with the beat durations to get exact durations summing to target_duration_sec.
2. `select_voice_tool` to cast an attribute-matched voice avoiding CreativeHistory repeats.
3. `save_director_artifacts_tool` with valid JSON for `shot_sheet_json`, `sound_plan_json`, and `script_json`.

Required ShotSheet fields (every shot MUST include):
- `id` (e.g. "s1"), `index` (0-based int), `beat` (from CreativeBible), `duration_sec` (exact fitted float)
- `engine`: "ai_video" | "image_to_video" | "motion_card" | "manim" (use "image_to_video" + `product_hero: true` + `reference_image` ONLY when scraped product photos exist)
- `visual_prompt`: rich, specific visual direction matching CreativeBible world & lighting with continuous visible motion (e.g., gentle camera glide, rising steam, pouring motion, focus rack — avoid purely static frames on `ai_video` shots so Media QA `freezedetect` passes; never ask AI video to render dense text)
- `camera_move`: non-empty dynamic camera move from CreativeBible `camera_grammar.allowed_moves` (e.g. "smooth lateral glide", "slow push-in")
- `camera_angle`: e.g. "close_up", "eye_level", "macro"
- `vo_line`: concise spoken line grounded ONLY in verified facts/claims (required when audio_mode is "vo" or "both", fitting comfortably inside `duration_sec` at ~2.5 words/sec)
- `theme` / `shot_sheet.theme`: match CreativeBible `typography_and_layout.font_family` and `visual_language.color_palette`.
- `shot_sheet.end_card_title`: ALWAYS set to the brand name (and optional `end_card_subtitle` for the offer/CTA) so `motion_card` end frames display clean brand typography instead of beat names.
If `save_director_artifacts_tool` reports any `blocking_errors`, fix them and call `save_director_artifacts_tool` again before finishing.
"""

AGENT2_GRADER_SYSTEM_PROMPT = """You are the Pure Script Rules & Craft Rubric Grader.
Always call `evaluate_pure_script_rules_tool` first. That tool deterministically validates the
persisted workspace ShotSheet/Script directly against the authoritative `CreativeBible` and
`ResearchBundle` in memory (including claim grounding, speakability, timing math, schema,
anti-garbled text, originality, and bible coherence). Base your structured `GraderResponse`
verdict directly on `evaluate_pure_script_rules_tool`'s returned `passed`, `result`, and `criteria`.
"""

AGENT2_RUBRIC = """Evaluate the Director artifacts strictly against pure script rules and craft:
1. claim_grounding: Zero ungrounded claims, banned phrases, superlatives, or placeholders.
2. speakability_budget: Every shot's VO fits comfortably within its shot duration.
3. timing_math: Shot durations sum to target_duration_sec and respect [MIN_SHOT, MAX_SHOT].
4. schema_and_product_photos: Complete visual/camera/audio spec; product_hero requires photos.
5. anti_garbled_text_and_b2: AI video prompts avoid dense rendered text; one clear idea per shot.
6. craft_originality_and_coherence: Strong originality and CreativeBible coherence (>= 6.0/10).
"""


@dataclass
class Agent2Result:
    """Output of Agent 2: validated Script, ShotSheet, SoundPlan, and script-gate QAReport."""

    script: Script
    shot_sheet: ShotSheet
    sound_plan: SoundPlan
    qa_report: QAReport
    rubric_evaluations: list[dict[str, Any]] = field(default_factory=list)
    status: str = "satisfied"
    workspace_dir: Path | None = None


def _score_craft_dimensions(
    sheet: ShotSheet,
    bible: CreativeBible,
) -> tuple[float, float]:
    """Compute deterministic (originality_score, bible_coherence_score) in [1.0, 10.0]."""
    prompt_blob = " ".join(
        f"{s.beat} {s.visual_prompt} {s.vo_line or ''} {s.on_screen_text or ''}"
        for s in sheet.shots
    ).lower()

    # Originality starts at 8.5 and deducts for anti_cliche violations or repetitive prompts
    originality = 8.5
    for cliche in bible.anti_cliche_list:
        if cliche.strip() and cliche.strip().lower() in prompt_blob:
            originality -= 1.5
    unique_prompts = len({s.visual_prompt.strip().lower() for s in sheet.shots})
    if len(sheet.shots) > 1 and unique_prompts < len(sheet.shots):
        originality -= 1.0
    if len({s.beat.strip().lower() for s in sheet.shots}) >= min(3, len(sheet.shots)):
        originality += 0.5

    # Bible coherence checks typography, color palette, camera style, and beat variety
    coherence = 7.5
    if (
        sheet.theme.font_family.strip().lower()
        == bible.typography_and_layout.font_family.strip().lower()
    ):
        coherence += 0.8
    if bible.visual_language.color_palette:
        palette_lower = {c.strip().lower() for c in bible.visual_language.color_palette}
        if (
            sheet.theme.background_color.strip().lower() in palette_lower
            or sheet.theme.accent_color.strip().lower() in palette_lower
        ):
            coherence += 0.7
    if any(s.camera_move.strip() for s in sheet.shots):
        coherence += 0.5
    if len(sheet.shots) >= 2:
        coherence += 0.3

    return (
        round(max(1.0, min(10.0, originality)), 2),
        round(max(1.0, min(10.0, coherence)), 2),
    )


def _trim_vo_to_fit(text: str, duration_sec: float) -> str:
    """Shorten a grounded VO line cleanly if needed so it always fits beat_fit_problem."""
    candidate = " ".join(text.strip().split())
    if not candidate:
        return ""
    if beat_fit_problem(candidate, duration_sec) is None:
        return candidate
    words = candidate.split()
    while len(words) > 2:
        words.pop()
        shortened = " ".join(words).rstrip(",;:") + "."
        if beat_fit_problem(shortened, duration_sec) is None:
            return shortened
    return " ".join(words[:2]) + "."


def _build_script_from_sheet(sheet: ShotSheet, research: ResearchBundle) -> Script:
    """Derive a structured Script model from a ShotSheet."""
    lines: list[ScriptLine] = []
    citations: list[ClaimCitation] = []
    for shot in sheet.shots:
        if shot.vo_line and shot.vo_line.strip():
            vo = shot.vo_line.strip()
            lines.append(
                ScriptLine(
                    shot_id=shot.id,
                    text=vo,
                    word_count=len(vo.split()),
                    estimated_sec=round(estimate_spoken_duration(vo), 2),
                    delivery_note=shot.beat,
                )
            )
    for idx, claim in enumerate(research.verified_claims):
        if any(claim.lower() in line.text.lower() for line in lines):
            citations.append(
                ClaimCitation(
                    claim_text=claim,
                    source_index=idx,
                    source_fact=claim,
                )
            )
    return Script(
        full_vo_text=" ".join(line.text for line in lines),
        lines=lines,
        claims_used=citations,
    )


def _synthesize_director_artifacts(
    *,
    run_input: RunInput,
    research: ResearchBundle,
    creative_bible: CreativeBible,
    fitted_durations: list[float],
    selected_voice_id: str,
) -> tuple[Script, ShotSheet, SoundPlan]:
    """Synthesize a grounded, pure-script-rule-compliant Script, ShotSheet, and SoundPlan."""
    palette = list(creative_bible.visual_language.color_palette)
    bg_color = palette[0] if len(palette) >= 1 else "#111827"
    accent_color = palette[1] if len(palette) >= 2 else "#3B82F6"
    theme = GraphicsTheme(
        font_family=creative_bible.typography_and_layout.font_family or "Roboto",
        primary_color="#FFFFFF",
        accent_color=accent_color,
        background_color=bg_color,
        text_color="#FFFFFF",
        caption_treatment=creative_bible.typography_and_layout.caption_treatment or "clean_bottom",
    )

    beats = list(creative_bible.narrative_structure.beats)
    n_shots = len(fitted_durations)
    allowed_moves = list(creative_bible.camera_grammar.allowed_moves) or [
        "smooth lateral glide",
        "gentle push-in",
        "locked editorial frame",
    ]
    has_photos = bool(research.product_photos)
    first_photo = (
        research.product_photos[0].path or research.product_photos[0].url if has_photos else None
    )

    # Build grounded VO candidates strictly from verified research + input
    claim_1 = (
        research.verified_claims[0]
        if research.verified_claims
        else f"{run_input.brand_name} delivers everyday quality."
    )
    fact_1 = (
        research.product_facts[0]
        if research.product_facts
        else f"Designed by {run_input.brand_name}."
    )
    offer_line = run_input.offer.strip() if run_input.offer and run_input.offer.strip() else ""
    cta_line = offer_line if offer_line else f"Discover {run_input.brand_name} today."

    vo_pool: list[str] = [
        f"Meet {run_input.brand_name}.",
        claim_1,
        fact_1,
        cta_line,
    ]
    must_say_lines = [p.strip() for p in run_input.must_say if p and p.strip()]
    if must_say_lines:
        if len(must_say_lines) <= n_shots:
            start_slot = 0 if len(must_say_lines) >= n_shots else 1
            while len(vo_pool) < start_slot + len(must_say_lines):
                vo_pool.append(cta_line)
            for offset, phrase in enumerate(must_say_lines):
                vo_pool[start_slot + offset] = phrase
        else:
            vo_pool = [""] * n_shots
            for idx_m, phrase in enumerate(must_say_lines):
                slot = idx_m % n_shots
                vo_pool[slot] = f"{vo_pool[slot]} {phrase}".strip() if vo_pool[slot] else phrase

    shots: list[Shot] = []
    vo_cues: list[VoiceCue] = []
    sfx_cues: list[SfxCue] = []
    cursor_sec = 0.0

    for idx in range(n_shots):
        dur = float(fitted_durations[idx])
        beat_spec = beats[idx] if idx < len(beats) else beats[-1]
        is_last = idx == n_shots - 1
        shot_id = f"s{idx + 1}"
        move = allowed_moves[idx % len(allowed_moves)]

        vo_line: str | None = None
        if run_input.audio_mode in ("vo", "both"):
            raw_vo = vo_pool[idx] if idx < len(vo_pool) else cta_line
            vo_line = _trim_vo_to_fit(raw_vo, dur)
            if vo_line:
                vo_cues.append(
                    VoiceCue(
                        shot_id=shot_id,
                        at_sec=round(cursor_sec, 3),
                        duration_sec=dur,
                        text=vo_line,
                        delivery_note=creative_bible.sonic_identity.voice_casting.delivery_style,
                    )
                )

        sfx_note: str | None = None
        if run_input.audio_mode in ("foley", "both") and idx in (0, n_shots - 1):
            sfx_prompt = (
                f"subtle natural tactile sound matching {creative_bible.visual_language.world}"
                if idx == 0
                else "soft organic acoustic room tone resolve"
            )
            sfx_note = sfx_prompt
            sfx_cues.append(
                SfxCue(
                    shot_id=shot_id,
                    at_sec=round(cursor_sec, 3),
                    duration_sec=min(1.5, dur),
                    prompt=sfx_prompt,
                    gain_db=-8.0,
                )
            )

        if is_last:
            shots.append(
                Shot(
                    id=shot_id,
                    index=idx,
                    beat=beat_spec.name,
                    duration_sec=dur,
                    engine="ai_video",
                    visual_prompt=(
                        f"Clean closing hero frame for {run_input.brand_name} "
                        f"in {creative_bible.visual_language.world}"
                    ),
                    negative_prompt=DEFAULT_NEGATIVE_PROMPT,
                    camera_move=move,
                    camera_angle="eye_level",
                    transition="fade_black",
                    environment=creative_bible.visual_language.world,
                    lighting_rig=creative_bible.visual_language.lighting_style,
                    theme=theme,
                    product_hero=False,
                    on_screen_text=None,
                    vo_line=vo_line,
                    sfx_note=sfx_note,
                )
            )
        else:
            is_hero = idx == 1 and has_photos
            shots.append(
                Shot(
                    id=shot_id,
                    index=idx,
                    beat=beat_spec.name,
                    duration_sec=dur,
                    engine="image_to_video" if is_hero and first_photo else "ai_video",
                    visual_prompt=(
                        f"{creative_bible.visual_language.world}; "
                        f"{creative_bible.visual_language.lighting_style}; "
                        f"{beat_spec.purpose} for {run_input.brand_name}"
                    ),
                    negative_prompt=DEFAULT_NEGATIVE_PROMPT,
                    camera_move=move,
                    camera_angle="eye_level" if idx == 0 else "close_up",
                    transition="cut" if idx == 0 else "j_cut",
                    environment=creative_bible.visual_language.world,
                    lighting_rig=creative_bible.visual_language.lighting_style,
                    product_hero=is_hero,
                    on_screen_text=None,
                    vo_line=vo_line,
                    sfx_note=sfx_note,
                    reference_image=first_photo if is_hero else None,
                )
            )
        cursor_sec += dur

    sheet = ShotSheet(
        fps=run_input.fps,
        aspect=run_input.aspect,
        target_duration_sec=run_input.target_duration_sec,
        audio_mode=run_input.audio_mode,
        theme=theme,
        shots=shots,
        end_card_title=run_input.brand_name,
        end_card_subtitle=offer_line or None,
    )
    script = _build_script_from_sheet(sheet, research)

    music_cue: MusicCue | None = None
    if creative_bible.sonic_identity.music_brief is not None:
        mb = creative_bible.sonic_identity.music_brief
        music_cue = MusicCue(
            prompt=mb.prompt or f"{mb.genre} underscore, {mb.mood}",
            tags=list(mb.tags) if mb.tags else [mb.genre, mb.mood],
            duration_sec=float(run_input.target_duration_sec),
            gain_db=-16.0,
            duck_db=-10.0,
        )

    sound_plan = SoundPlan(
        voice_id=selected_voice_id,
        voice_provider="fal",
        voice_casting=creative_bible.sonic_identity.voice_casting,
        vo_cues=vo_cues,
        sfx_cues=sfx_cues,
        music_cue=music_cue,
    )
    return script, sheet, sound_plan


def _build_agent2_tools(
    *,
    run_input: RunInput,
    research: ResearchBundle,
    creative_bible: CreativeBible,
    history_store: CreativeHistoryStore,
    workspace: Path,
) -> tuple[list[BaseTool], list[BaseTool]]:
    """Create Agent 2 Director worker tools and its pure-script-rules grader tool."""

    @tool("fit_shot_timings_tool")
    def fit_shot_timings_tool(
        durations_json: str,
        target_sec: float = 0.0,
        min_shot: float = MIN_SHOT_SEC,
        max_shot: float = MAX_SHOT_SEC,
    ) -> str:
        """Fit shot durations within [min_shot, max_shot] summing to target_sec (BUG-21)."""
        raw_durations = [float(d) for d in json.loads(durations_json)]
        target = float(target_sec) if target_sec > 0 else float(run_input.target_duration_sec)
        fitted = fit_timings(raw_durations, target, min_shot=min_shot, max_shot=max_shot)
        write_workspace_json(workspace, "fitted_timings.json", fitted)
        return json.dumps(
            {
                "fitted_durations": fitted,
                "sum_sec": round(sum(fitted), 3),
                "target_sec": target,
            }
        )

    @tool("select_voice_tool")
    def select_voice_tool() -> str:
        """Select an attribute-matched voice avoiding CreativeHistory repeats (BUG-17)."""
        history = history_store.get_history(run_input.brand_url)
        profile = select_voice(
            creative_bible.sonic_identity.voice_casting,
            brand_history=history,
        )
        payload = {
            "voice_id": profile.voice_id,
            "name": profile.name,
            "gender": profile.gender,
            "age": profile.age,
            "accent": profile.accent,
            "energy": profile.energy,
            "pace": profile.pace,
        }
        write_workspace_json(workspace, "selected_voice.json", payload)
        return json.dumps(payload)

    @tool("save_director_artifacts_tool")
    def save_director_artifacts_tool(
        shot_sheet_json: str,
        sound_plan_json: str,
        script_json: str = "",
    ) -> str:
        """Validate and persist ShotSheet, SoundPlan, and Script JSON artifacts to workspace."""
        sheet = ShotSheet.model_validate(json.loads(shot_sheet_json))
        sound_plan = SoundPlan.model_validate(json.loads(sound_plan_json))
        if script_json.strip():
            script = Script.model_validate(json.loads(script_json))
            # If LLM wrote VO lines in script.lines but left shot.vo_line empty, sync them onto shots
            line_by_id = {ln.shot_id: ln.text.strip() for ln in script.lines if ln.text.strip()}
            for shot in sheet.shots:
                if not (shot.vo_line and shot.vo_line.strip()) and shot.id in line_by_id:
                    shot.vo_line = line_by_id[shot.id]
        else:
            script = _build_script_from_sheet(sheet, research)

        # If audio_mode expects VO and vo_cues was left empty, synthesize vo_cues from shot.vo_line
        if sheet.audio_mode in ("vo", "both") and not sound_plan.vo_cues:
            cursor = 0.0
            for shot in sheet.shots:
                if shot.vo_line and shot.vo_line.strip():
                    sound_plan.vo_cues.append(
                        VoiceCue(
                            shot_id=shot.id,
                            at_sec=round(cursor, 3),
                            duration_sec=shot.duration_sec,
                            text=shot.vo_line.strip(),
                            delivery_note=creative_bible.sonic_identity.voice_casting.delivery_style,
                        )
                    )
                cursor += shot.duration_sec

        write_workspace_json(workspace, "shot_sheet.json", sheet)
        write_workspace_json(workspace, "sound_plan.json", sound_plan)
        write_workspace_json(workspace, "script.json", script)

        gate = evaluate_script_gate(
            sheet,
            research,
            script=script,
            has_product_photos=bool(research.product_photos),
            extra_facts=run_input.facts,
            offer=run_input.offer,
            must_say=run_input.must_say,
            gate_override=False,
        )
        orig, coh = _score_craft_dimensions(sheet, creative_bible)
        return json.dumps(
            {
                "status": "ok" if gate.passed else "needs_fixes",
                "shot_count": len(sheet.shots),
                "sum_sec": round(sum(s.duration_sec for s in sheet.shots), 3),
                "voice_id": sound_plan.voice_id,
                "gate_passed": gate.passed,
                "originality_score": orig,
                "bible_coherence_score": coh,
                "blocking_errors": [
                    {"code": i.code, "shot_id": i.shot_id, "message": i.message}
                    for i in gate.blocking_errors
                ],
            }
        )

    @tool("evaluate_pure_script_rules_tool")
    def evaluate_pure_script_rules_tool() -> str:
        """Evaluate persisted ShotSheet/Script strictly against pure script rules + craft scores."""
        raw_sheet = read_workspace_json(workspace, "shot_sheet.json")
        raw_script = read_workspace_json(workspace, "script.json")

        if raw_sheet is None:
            return json.dumps(
                {
                    "passed": False,
                    "result": "needs_revision",
                    "explanation": "shot_sheet.json is missing; save director artifacts first.",
                    "criteria": [
                        {
                            "name": "claim_grounding",
                            "passed": False,
                            "gap": "No shot_sheet.json found in workspace.",
                        },
                        {
                            "name": "speakability_budget",
                            "passed": False,
                            "gap": "No shot_sheet.json found in workspace.",
                        },
                        {
                            "name": "timing_math",
                            "passed": False,
                            "gap": "No shot_sheet.json found in workspace.",
                        },
                        {
                            "name": "schema_and_product_photos",
                            "passed": False,
                            "gap": "No shot_sheet.json found in workspace.",
                        },
                        {
                            "name": "anti_garbled_text_and_b2",
                            "passed": False,
                            "gap": "No shot_sheet.json found in workspace.",
                        },
                        {
                            "name": "craft_originality_and_coherence",
                            "passed": False,
                            "gap": "No shot_sheet.json found in workspace.",
                        },
                    ],
                }
            )

        sheet = ShotSheet.model_validate(raw_sheet)
        script = Script.model_validate(raw_script) if raw_script is not None else None

        # BUG-18: ALWAYS pass has_product_photos=bool(research.product_photos)
        gate_report = evaluate_script_gate(
            sheet,
            research,
            script=script,
            has_product_photos=bool(research.product_photos),
            extra_facts=run_input.facts,
            offer=run_input.offer,
            must_say=run_input.must_say,
            gate_override=False,
        )
        originality, coherence = _score_craft_dimensions(sheet, creative_bible)
        gate_report.originality_score = originality
        gate_report.bible_coherence_score = coherence
        write_workspace_json(workspace, "director_qa.json", gate_report)

        claim_codes = {
            "UNGROUNDED_CLAIM",
            "BANNED_PHRASE",
            "UNSUPPORTED_SUPERLATIVE",
            "UNRESOLVED_PLACEHOLDER",
            "MISSING_MUST_SAY",
        }
        timing_codes = {
            "INVALID_SHOT_DURATION",
            "TIMING_SUM_MISMATCH",
            "SHOT_TOO_SHORT",
            "SHOT_TOO_LONG",
        }
        schema_codes = {
            "MISSING_VISUAL_PROMPT",
            "MISSING_CAMERA_SPEC",
            "PRODUCT_HERO_WITHOUT_PHOTOS",
            "MISSING_VO_LINES",
        }
        known_codes = (
            claim_codes
            | timing_codes
            | schema_codes
            | {"SPEAKABILITY_OVERFLOW", "GARBLED_TEXT_RISK"}
        )

        claim_errs = [i for i in gate_report.blocking_errors if i.code in claim_codes]
        speak_errs = [i for i in gate_report.blocking_errors if i.code == "SPEAKABILITY_OVERFLOW"]
        timing_errs = [i for i in gate_report.blocking_errors if i.code in timing_codes]
        unmapped_errs = [i for i in gate_report.blocking_errors if i.code not in known_codes]
        schema_errs = [
            i for i in gate_report.blocking_errors if i.code in schema_codes
        ] + unmapped_errs
        text_errs = [i for i in gate_report.blocking_errors if i.code == "GARBLED_TEXT_RISK"]
        craft_ok = originality >= 6.0 and coherence >= 6.0

        def _crit(name: str, issues: list[Any]) -> dict[str, Any]:
            if not issues:
                return {"name": name, "passed": True}
            return {
                "name": name,
                "passed": False,
                "gap": "; ".join(f"[{i.code}] {i.message}" for i in issues),
            }

        criteria = [
            _crit("claim_grounding", claim_errs),
            _crit("speakability_budget", speak_errs),
            _crit("timing_math", timing_errs),
            _crit("schema_and_product_photos", schema_errs),
            _crit("anti_garbled_text_and_b2", text_errs),
            (
                {"name": "craft_originality_and_coherence", "passed": True}
                if craft_ok
                else {
                    "name": "craft_originality_and_coherence",
                    "passed": False,
                    "gap": (
                        f"Craft scores below threshold "
                        f"(originality={originality}, coherence={coherence})."
                    ),
                }
            ),
        ]

        all_passed = gate_report.passed and craft_ok
        failed_names = [c["name"] for c in criteria if not c["passed"]]
        return json.dumps(
            {
                "passed": all_passed,
                "result": "satisfied" if all_passed else "needs_revision",
                "explanation": (
                    f"Pure script rules passed (originality={originality}, coherence={coherence})."
                    if all_passed
                    else f"Pure script rule failures in: {', '.join(failed_names)}."
                ),
                "originality_score": originality,
                "bible_coherence_score": coherence,
                "criteria": criteria,
            }
        )

    worker_tools: list[BaseTool] = [
        fit_shot_timings_tool,
        select_voice_tool,
        save_director_artifacts_tool,
    ]
    grader_tools: list[BaseTool] = [evaluate_pure_script_rules_tool]
    return worker_tools, grader_tools


def _make_offline_agent2_model(
    *,
    run_input: RunInput,
    research: ResearchBundle,
    creative_bible: CreativeBible,
    workspace: Path,
) -> ScriptedChatModel:
    """Create a genuine tool-calling offline ScriptedChatModel for Agent 2."""
    step = {"idx": 0}

    def _responder(messages: list[BaseMessage]) -> dict[str, Any] | str:
        seen_tools = {getattr(m, "name", "") for m in messages if isinstance(m, ToolMessage)}
        step["idx"] += 1

        if "fit_shot_timings_tool" not in seen_tools:
            hints = [float(b.duration_hint_sec) for b in creative_bible.narrative_structure.beats]
            if not hints:
                hints = [run_input.target_duration_sec / 3.0] * 3
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": f"tc_a2_fit_{step['idx']}",
                        "name": "fit_shot_timings_tool",
                        "args": {
                            "durations_json": json.dumps(hints),
                            "target_sec": float(run_input.target_duration_sec),
                        },
                        "type": "tool_call",
                    }
                ],
            }

        if "select_voice_tool" not in seen_tools:
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": f"tc_a2_voice_{step['idx']}",
                        "name": "select_voice_tool",
                        "args": {},
                        "type": "tool_call",
                    }
                ],
            }

        if "save_director_artifacts_tool" not in seen_tools:
            fitted = read_workspace_json(workspace, "fitted_timings.json") or fit_timings(
                [b.duration_hint_sec for b in creative_bible.narrative_structure.beats],
                run_input.target_duration_sec,
            )
            voice_data = read_workspace_json(workspace, "selected_voice.json") or {}
            voice_id = str(voice_data.get("voice_id") or "v_maya_warm")

            script, sheet, sound_plan = _synthesize_director_artifacts(
                run_input=run_input,
                research=research,
                creative_bible=creative_bible,
                fitted_durations=[float(d) for d in fitted],
                selected_voice_id=voice_id,
            )
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": f"tc_a2_save_{step['idx']}",
                        "name": "save_director_artifacts_tool",
                        "args": {
                            "shot_sheet_json": sheet.model_dump_json(),
                            "sound_plan_json": sound_plan.model_dump_json(),
                            "script_json": script.model_dump_json(),
                        },
                        "type": "tool_call",
                    }
                ],
            }

        return "Completed Script, ShotSheet, and bespoke SoundPlan."

    return ScriptedChatModel(responder=_responder, model_name="agent2-offline-director")


def _make_agent2_prepare_messages(
    workspace: Path,
    research: ResearchBundle,
) -> Any:
    """Sync raw JSON director outputs from AIMessage into workspace before grading.

    Enables custom scripted models (e.g., E2E tests returning raw JSON `{"shot_sheet": ...,
    "sound_plan": ...}`) to be graded by the exact same workspace-backed pure-script-rules tool.
    """

    def _prepare(messages: list[AnyMessage]) -> list[AnyMessage]:
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and isinstance(msg.content, str):
                text = msg.content.strip()
                if text.startswith("{") and text.endswith("}"):
                    try:
                        payload = json.loads(text)
                    except json.JSONDecodeError:
                        break
                    if isinstance(payload, dict) and "shot_sheet" in payload:
                        try:
                            sheet = ShotSheet.model_validate(payload["shot_sheet"])
                            write_workspace_json(workspace, "shot_sheet.json", sheet)
                            if "sound_plan" in payload and payload["sound_plan"] is not None:
                                sp = SoundPlan.model_validate(payload["sound_plan"])
                                write_workspace_json(workspace, "sound_plan.json", sp)
                            if "script" in payload and payload["script"] is not None:
                                sc = Script.model_validate(payload["script"])
                                write_workspace_json(workspace, "script.json", sc)
                            else:
                                write_workspace_json(
                                    workspace,
                                    "script.json",
                                    _build_script_from_sheet(sheet, research),
                                )
                        except (ValidationError, ValueError, TypeError):
                            pass
                break
        return messages

    return _prepare


async def run_agent2_director(
    *,
    run_input: RunInput,
    research: ResearchBundle,
    creative_bible: CreativeBible,
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
) -> Agent2Result:
    """Run Agent 2 (Script & ShotSheet Director DeepAgent) with RubricMiddleware gating."""
    if cancel is not None:
        cancel.raise_if_cancelled()

    base_dir = (
        Path(work_dir).resolve()
        if work_dir is not None
        else Path(tempfile.mkdtemp(prefix="gtm_a2_"))
    )
    workspace = base_dir / "agent2_workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    resolved_history = history_store or CreativeHistoryStore(base_dir / "creative_history.json")
    worker_tools, grader_tools = _build_agent2_tools(
        run_input=run_input,
        research=research,
        creative_bible=creative_bible,
        history_store=resolved_history,
        workspace=workspace,
    )

    is_offline_mode = run_input.mock or is_offline_provider_bundle(providers)

    resolved_main = main_model
    if resolved_main is None:
        provider_chat = getattr(providers, "chat_model", None)
        if provider_chat is not None and not is_default_unscripted_mock_model(provider_chat):
            resolved_main = provider_chat
        elif is_offline_mode:
            resolved_main = _make_offline_agent2_model(
                run_input=run_input,
                research=research,
                creative_bible=creative_bible,
                workspace=workspace,
            )
        else:
            resolved_main = resolve_live_chat_model("script_write", providers=providers)

    if grader_model is not None:
        resolved_grader = grader_model
    elif is_offline_mode or isinstance(resolved_main, ScriptedChatModel):
        resolved_grader = make_tool_driven_grader_model(
            "evaluate_pure_script_rules_tool",
            model_name="agent2-rubric-grader",
        )
    else:
        resolved_grader = resolve_live_chat_model("claims_check", providers=providers)

    research_context = {
        "brand_name": research.brand_name,
        "brand_url": research.brand_url,
        "goal": run_input.goal,
        "offer": run_input.offer,
        "must_say": run_input.must_say,
        "verified_claims": research.verified_claims[:10],
        "product_facts": (list(run_input.facts) + list(research.product_facts))[:15],
        "proof_points": research.proof_points[:8],
        "brand_colors": research.brand_colors[:6],
        "product_photos": [
            {"id": p.id, "url": p.url, "path": p.path, "alt": p.alt}
            for p in research.product_photos[:4]
        ],
        "banned_phrases": research.banned_phrases[:10],
    }
    prompt = (
        f"Direct a {run_input.target_duration_sec}s {run_input.aspect} ({run_input.fps}fps) "
        f"commercial for {run_input.brand_name} using CreativeBible arc "
        f"'{creative_bible.narrative_structure.arc_name}' and audio_mode='{run_input.audio_mode}'.\n\n"
        f"CreativeBible JSON:\n{creative_bible.model_dump_json(indent=2)}\n\n"
        f"Verified ResearchBundle Context:\n{json.dumps(research_context, indent=2)}"
    )

    outcome = await run_rubric_deep_agent(
        stage=AGENT2_STAGE,
        run_id=run_id,
        prompt=prompt,
        rubric=AGENT2_RUBRIC,
        main_model=resolved_main,
        grader_model=resolved_grader,
        system_prompt=AGENT2_SYSTEM_PROMPT,
        grader_system_prompt=AGENT2_GRADER_SYSTEM_PROMPT,
        tools=worker_tools,
        grader_tools=grader_tools,
        max_iterations=max_iterations,
        event_cb=event_cb,
        cancel=cancel,
        prepare_messages_for_grader=_make_agent2_prepare_messages(workspace, research),
        raise_on_failure=raise_on_failure,
    )

    raw_sheet = read_workspace_json(workspace, "shot_sheet.json")
    raw_sound = read_workspace_json(workspace, "sound_plan.json")
    raw_script = read_workspace_json(workspace, "script.json")
    raw_qa = read_workspace_json(workspace, "director_qa.json")

    if raw_sheet is None:
        voice = select_voice(
            creative_bible.sonic_identity.voice_casting,
            brand_history=resolved_history.get_history(run_input.brand_url),
        )
        fitted = fit_timings(
            [b.duration_hint_sec for b in creative_bible.narrative_structure.beats],
            run_input.target_duration_sec,
        )
        script, sheet, sound_plan = _synthesize_director_artifacts(
            run_input=run_input,
            research=research,
            creative_bible=creative_bible,
            fitted_durations=fitted,
            selected_voice_id=voice.voice_id,
        )
    else:
        sheet = ShotSheet.model_validate(raw_sheet)
        script = (
            Script.model_validate(raw_script)
            if raw_script is not None
            else _build_script_from_sheet(sheet, research)
        )
        if raw_sound is not None:
            sound_plan = SoundPlan.model_validate(raw_sound)
        else:
            voice = select_voice(
                creative_bible.sonic_identity.voice_casting,
                brand_history=resolved_history.get_history(run_input.brand_url),
            )
            sound_plan = SoundPlan(
                voice_id=voice.voice_id,
                voice_casting=creative_bible.sonic_identity.voice_casting,
            )

    if raw_qa is not None:
        qa_report = QAReport.model_validate(raw_qa)
    else:
        qa_report = evaluate_script_gate(
            sheet,
            research,
            script=script,
            has_product_photos=bool(research.product_photos),
            extra_facts=run_input.facts,
            offer=run_input.offer,
            must_say=run_input.must_say,
        )
        orig, coh = _score_craft_dimensions(sheet, creative_bible)
        qa_report.originality_score = orig
        qa_report.bible_coherence_score = coh

    return Agent2Result(
        script=script,
        shot_sheet=sheet,
        sound_plan=sound_plan,
        qa_report=qa_report,
        rubric_evaluations=outcome.evaluations,
        status=outcome.status,
        workspace_dir=workspace,
    )


__all__ = [
    "AGENT2_RUBRIC",
    "AGENT2_STAGE",
    "Agent2Result",
    "run_agent2_director",
]
