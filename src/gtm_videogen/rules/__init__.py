"""Public exports for gtm_videogen.rules (Pure Script Rules ONLY — zero hard templates)."""

from gtm_videogen.rules.anti_garbled_text import DEFAULT_NEGATIVE_PROMPT, rendered_text_problem
from gtm_videogen.rules.claim_grounding import (
    build_verified_corpus,
    check_claim_grounding,
    extract_banned_phrases,
    extract_claim_tokens,
    find_placeholders,
    find_unsupported_superlatives,
)
from gtm_videogen.rules.creative_history import (
    MAX_HISTORY_PER_BRAND,
    CreativeHistoryStore,
    format_previously_explored,
    normalize_brand_url,
)
from gtm_videogen.rules.diversity import diversity_scores, pairwise_distance, select_diverse
from gtm_videogen.rules.script_gate import evaluate_script_gate
from gtm_videogen.rules.speakability import (
    BEAT_PADDING_SEC,
    DOMAIN_SEC,
    MAX_TEMPO,
    SENTENCE_PAUSE_SEC,
    SPEECH_LEAD_SEC,
    SYLLABLES_PER_SEC,
    beat_fit_problem,
    count_syllables,
    count_text_syllables,
    estimate_spoken_duration,
    spoken_form,
)
from gtm_videogen.rules.timing import MAX_SHOT_SEC, MIN_SHOT_SEC, fit_timings
from gtm_videogen.rules.voice_casting import VOICE_CATALOG, VoiceProfile, select_voice

__all__ = [
    "BEAT_PADDING_SEC",
    "DEFAULT_NEGATIVE_PROMPT",
    "DOMAIN_SEC",
    "MAX_HISTORY_PER_BRAND",
    "MAX_SHOT_SEC",
    "MAX_TEMPO",
    "MIN_SHOT_SEC",
    "SENTENCE_PAUSE_SEC",
    "SPEECH_LEAD_SEC",
    "SYLLABLES_PER_SEC",
    "VOICE_CATALOG",
    "CreativeHistoryStore",
    "VoiceProfile",
    "beat_fit_problem",
    "build_verified_corpus",
    "check_claim_grounding",
    "count_syllables",
    "count_text_syllables",
    "diversity_scores",
    "estimate_spoken_duration",
    "evaluate_script_gate",
    "extract_banned_phrases",
    "extract_claim_tokens",
    "find_placeholders",
    "find_unsupported_superlatives",
    "fit_timings",
    "format_previously_explored",
    "normalize_brand_url",
    "pairwise_distance",
    "rendered_text_problem",
    "select_diverse",
    "select_voice",
    "spoken_form",
]
