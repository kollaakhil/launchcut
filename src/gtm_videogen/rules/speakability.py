"""Speakability and syllable-budget math for voiceover lines."""

from __future__ import annotations

import math
import re

SYLLABLES_PER_SEC = 4.6
SPEECH_LEAD_SEC = 0.25
BEAT_PADDING_SEC = 0.3
SENTENCE_PAUSE_SEC = 0.4
DOMAIN_SEC = 0.45
MAX_TEMPO = 1.05

_ONES = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
)
_TENS = (
    "",
    "",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
)

_DOMAIN_RE = re.compile(
    r"\b([a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)*)\.(com|io|ai|co|org|net|app|dev)\b",
    re.IGNORECASE,
)
_CURRENCY_RE = re.compile(r"\$(\d+(?:,\d{3})*(?:\.\d+)?)\s*([kmbKMB])?\b")
_PERCENT_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*%")
_MULTIPLIER_RE = re.compile(r"\b(\d+(?:,\d{3})*(?:\.\d+)?)x\b", re.IGNORECASE)
_PLUS_NUM_RE = re.compile(r"\b(\d+(?:,\d{3})*(?:\.\d+)?)\+")
_PLAIN_NUM_RE = re.compile(r"\b(\d+(?:,\d{3})*(?:\.\d+)?)\b")
_VOWEL_GROUP_RE = re.compile(r"[aeiouy]+", re.IGNORECASE)
_WORD_RE = re.compile(r"[a-zA-Z]+")
_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+(?:\s+|$)")


def _int_to_words(n: int) -> str:
    if n < 0:
        return f"minus {_int_to_words(-n)}"
    if n < 20:
        return _ONES[n]
    if n < 100:
        tens, rem = divmod(n, 10)
        return _TENS[tens] if rem == 0 else f"{_TENS[tens]} {_ONES[rem]}"
    if n < 1000:
        hundreds, rem = divmod(n, 100)
        head = f"{_ONES[hundreds]} hundred"
        return head if rem == 0 else f"{head} {_int_to_words(rem)}"
    for scale_val, scale_name in (
        (1_000_000_000, "billion"),
        (1_000_000, "million"),
        (1_000, "thousand"),
    ):
        if n >= scale_val:
            head_num, rem = divmod(n, scale_val)
            head = f"{_int_to_words(head_num)} {scale_name}"
            return head if rem == 0 else f"{head} {_int_to_words(rem)}"
    return str(n)


def _number_str_to_words(num_str: str) -> str:
    cleaned = num_str.replace(",", "")
    if "." in cleaned:
        whole_str, frac_str = cleaned.split(".", 1)
        whole_words = _int_to_words(int(whole_str)) if whole_str else "zero"
        frac_words = " ".join(_ONES[int(ch)] for ch in frac_str if ch.isdigit())
        return f"{whole_words} point {frac_words}".strip()
    return _int_to_words(int(cleaned))


def _expand_tld(tld: str) -> str:
    low = tld.lower()
    if low in ("io", "ai"):
        return " ".join(low)
    return low


def spoken_form(text: str) -> str:
    """Expand symbols, numbers, currency, percentages, and domains into spoken English."""
    if not text:
        return ""

    out = text

    # 1. Domains (e.g. app.example.io -> app dot example dot i o)
    def _domain_sub(match: re.Match[str]) -> str:
        host_labels = match.group(1).split(".")
        return " dot ".join(host_labels) + f" dot {_expand_tld(match.group(2))}"

    out = _DOMAIN_RE.sub(_domain_sub, out)

    # 2. Currency ($29, $1.5M)
    def _currency_sub(match: re.Match[str]) -> str:
        num_words = _number_str_to_words(match.group(1))
        suffix = (match.group(2) or "").lower()
        scale = {"k": "thousand", "m": "million", "b": "billion"}.get(suffix, "")
        parts = [num_words]
        if scale:
            parts.append(scale)
        parts.append("dollars")
        return " ".join(parts)

    out = _CURRENCY_RE.sub(_currency_sub, out)

    # 3. Percentages (50% -> fifty percent)
    out = _PERCENT_RE.sub(lambda m: f"{_number_str_to_words(m.group(1))} percent", out)

    # 4. Multipliers (10x -> ten times)
    out = _MULTIPLIER_RE.sub(lambda m: f"{_number_str_to_words(m.group(1))} times", out)

    # 5. Plus numbers (10+ -> ten plus)
    out = _PLUS_NUM_RE.sub(lambda m: f"{_number_str_to_words(m.group(1))} plus", out)

    # 6. Standalone '+'
    out = out.replace("+", " plus ")

    # 7. Plain numbers
    out = _PLAIN_NUM_RE.sub(lambda m: _number_str_to_words(m.group(1)), out)

    return re.sub(r"\s+", " ", out).strip()


def count_syllables(word: str) -> int:
    """Estimate English syllable count for a single alphabetic word."""
    cleaned = re.sub(r"[^a-zA-Z]", "", word).lower()
    if not cleaned:
        return 0
    if len(cleaned) <= 3:
        return 1

    # Strip silent trailing 'e' unless word ends in '-le' preceded by a consonant
    trimmed = cleaned
    if trimmed.endswith("e") and not (
        len(trimmed) >= 3 and trimmed.endswith("le") and trimmed[-3] not in "aeiouy"
    ):
        trimmed = trimmed[:-1]

    groups = _VOWEL_GROUP_RE.findall(trimmed)
    return max(1, len(groups))


def count_text_syllables(text: str) -> int:
    """Count total spoken syllables after expanding symbols/numbers/domains."""
    expanded = spoken_form(text)
    return sum(count_syllables(word) for word in _WORD_RE.findall(expanded))


def estimate_spoken_duration(text: str) -> float:
    """Estimate spoken duration in seconds for a voiceover line."""
    if not text or not text.strip():
        return 0.0

    domain_count = len(_DOMAIN_RE.findall(text))
    syllables = count_text_syllables(text)
    if syllables == 0:
        return 0.0

    # Count internal sentence boundaries (punctuation followed by more words)
    stripped = text.strip()
    sentences = [s for s in _SENTENCE_SPLIT_RE.split(stripped) if s.strip()]
    internal_pauses = max(0, len(sentences) - 1)

    sec = (
        SPEECH_LEAD_SEC
        + BEAT_PADDING_SEC
        + (syllables / SYLLABLES_PER_SEC)
        + (internal_pauses * SENTENCE_PAUSE_SEC)
        + (domain_count * DOMAIN_SEC)
    )
    return round(sec, 3)


def beat_fit_problem(text: str, duration_sec: float) -> str | None:
    """Return a problem description if the VO line cannot fit comfortably in duration_sec."""
    if not text or not text.strip():
        return None
    if not math.isfinite(duration_sec) or duration_sec <= 0:
        return f"Shot duration must be a finite positive number (got {duration_sec}s)"

    estimated = estimate_spoken_duration(text)
    budget = duration_sec * MAX_TEMPO
    if estimated > budget:
        return (
            f"VO line is too long for shot: estimated {estimated:.2f}s exceeds "
            f"available {budget:.2f}s ({duration_sec:.2f}s * {MAX_TEMPO:.2f} max tempo)"
        )
    return None
