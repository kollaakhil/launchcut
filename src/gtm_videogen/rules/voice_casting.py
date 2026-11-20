"""Attribute-matched voice casting avoiding CreativeHistory repeats (fixes BUG-17)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from gtm_videogen.models.schemas import CreativeFingerprint, CreativeHistory, VoiceCasting


@dataclass(frozen=True)
class VoiceProfile:
    """Structured voice metadata entry for multi-attribute casting."""

    voice_id: str
    name: str
    provider: str
    gender: str
    age: str
    accent: str
    energy: str
    pace: str
    style_tags: tuple[str, ...] = ()


VOICE_CATALOG: tuple[VoiceProfile, ...] = (
    VoiceProfile(
        voice_id="v_aria_f_adult_us",
        name="Aria",
        provider="fal",
        gender="female",
        age="adult",
        accent="us",
        energy="conversational",
        pace="natural",
        style_tags=("clear", "modern", "editorial"),
    ),
    VoiceProfile(
        voice_id="v_clara_f_young_uk",
        name="Clara",
        provider="fal",
        gender="female",
        age="young",
        accent="uk",
        energy="energetic",
        pace="brisk",
        style_tags=("bright", "crisp", "playful"),
    ),
    VoiceProfile(
        voice_id="v_helena_f_mature_neutral",
        name="Helena",
        provider="fal",
        gender="female",
        age="mature",
        accent="neutral",
        energy="authoritative",
        pace="measured",
        style_tags=("documentary", "confident", "cinematic"),
    ),
    VoiceProfile(
        voice_id="v_maya_f_adult_us_warm",
        name="Maya",
        provider="fal",
        gender="female",
        age="adult",
        accent="us",
        energy="warm",
        pace="slow",
        style_tags=("intimate", "storytelling", "calm"),
    ),
    VoiceProfile(
        voice_id="v_liam_m_adult_us",
        name="Liam",
        provider="fal",
        gender="male",
        age="adult",
        accent="us",
        energy="conversational",
        pace="natural",
        style_tags=("founder", "grounded", "direct"),
    ),
    VoiceProfile(
        voice_id="v_devon_m_young_us",
        name="Devon",
        provider="fal",
        gender="male",
        age="young",
        accent="us",
        energy="energetic",
        pace="brisk",
        style_tags=("punchy", "fast", "dynamic"),
    ),
    VoiceProfile(
        voice_id="v_arthur_m_mature_uk",
        name="Arthur",
        provider="fal",
        gender="male",
        age="mature",
        accent="uk",
        energy="authoritative",
        pace="measured",
        style_tags=("prestige", "refined", "cinematic"),
    ),
    VoiceProfile(
        voice_id="v_mateo_m_adult_neutral",
        name="Mateo",
        provider="fal",
        gender="male",
        age="adult",
        accent="neutral",
        energy="calm",
        pace="slow",
        style_tags=("reflective", "minimal", "steady"),
    ),
    VoiceProfile(
        voice_id="v_sam_n_adult_neutral",
        name="Sam",
        provider="fal",
        gender="neutral",
        age="adult",
        accent="neutral",
        energy="conversational",
        pace="natural",
        style_tags=("clean", "neutral", "modern"),
    ),
    VoiceProfile(
        voice_id="v_rowan_n_young_uk",
        name="Rowan",
        provider="fal",
        gender="neutral",
        age="young",
        accent="uk",
        energy="warm",
        pace="brisk",
        style_tags=("fresh", "witty", "editorial"),
    ),
)


def _extract_used_voices(
    brand_history: CreativeHistory | Sequence[CreativeFingerprint] | None,
) -> list[str]:
    if brand_history is None:
        return []
    entries = (
        brand_history.entries if isinstance(brand_history, CreativeHistory) else list(brand_history)
    )
    return [e.voice_id for e in entries if e.voice_id]


def _score_profile(
    profile: VoiceProfile,
    casting: VoiceCasting,
    used_voices: list[str],
) -> float:
    score = 0.0

    # Gender match (weight 3.0)
    if casting.gender == "any":
        score += 1.0
    elif profile.gender == casting.gender:
        score += 3.0
    elif profile.gender == "neutral":
        score += 1.0

    # Age match (weight 2.5)
    if casting.age == "any":
        score += 1.0
    elif profile.age == casting.age:
        score += 2.5

    # Accent match (weight 2.0)
    req_accent = casting.accent.strip().lower()
    if req_accent in ("", "any", "neutral"):
        score += 1.5 if profile.accent == "neutral" else 1.0
    elif profile.accent.lower() == req_accent or req_accent in profile.accent.lower():
        score += 2.0
    elif profile.accent == "neutral":
        score += 0.5

    # Energy match (weight 2.5)
    if profile.energy == casting.energy:
        score += 2.5
    elif {profile.energy, casting.energy} in (
        {"calm", "warm"},
        {"warm", "conversational"},
        {"energetic", "conversational"},
        {"authoritative", "calm"},
    ):
        score += 1.0

    # Pace match (weight 2.0)
    if profile.pace == casting.pace:
        score += 2.0
    elif {profile.pace, casting.pace} in (
        {"slow", "measured"},
        {"measured", "natural"},
        {"natural", "brisk"},
    ):
        score += 0.75

    # Delivery style tag overlap (weight up to 1.5)
    if casting.delivery_style:
        style_lower = casting.delivery_style.lower()
        hits = sum(1 for tag in profile.style_tags if tag in style_lower)
        score += min(1.5, hits * 0.75)

    # Strong anti-repetition penalty for voices already used in brand_history (Pillar 6)
    if profile.voice_id in used_voices:
        occurrences = used_voices.count(profile.voice_id)
        # Most recent entry gets strongest penalty
        recency_penalty = 2.0 if used_voices[-1] == profile.voice_id else 0.0
        score -= 6.0 + recency_penalty + (occurrences - 1) * 1.5

    return score


def select_voice(
    casting: VoiceCasting,
    brand_history: CreativeHistory | Sequence[CreativeFingerprint] | None = None,
    *,
    catalog: Sequence[VoiceProfile] | None = None,
) -> VoiceProfile:
    """Select the highest-scoring VoiceProfile matching all VoiceCasting attributes
    while avoiding voices recently used in CreativeHistory (BUG-17).
    """
    profiles = list(catalog) if catalog is not None else list(VOICE_CATALOG)
    if not profiles:
        raise ValueError("Voice catalog cannot be empty")

    used_voices = _extract_used_voices(brand_history)
    used_set = set(used_voices)
    unused = [p for p in profiles if p.voice_id not in used_set]
    candidates = unused if unused else profiles
    return max(candidates, key=lambda p: _score_profile(p, casting, used_voices))
