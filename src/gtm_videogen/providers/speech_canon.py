"""Speech canonicalization, ASR vocabulary extraction, and fuzzy brand matching."""

from __future__ import annotations

import re

from gtm_videogen.rules.speakability import spoken_form

_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]+")
_WHITESPACE_RE = re.compile(r"\s+")
_DOMAIN_RE = re.compile(r"\b[a-zA-Z0-9-]+\.(?:com|io|ai|co|org|net|app|dev)\b", re.IGNORECASE)
_PROPER_OR_COINED_RE = re.compile(r"\b[A-Z][a-zA-Z0-9'-]{2,}\b")


def levenshtein_distance(a: str, b: str) -> int:
    """Compute exact Levenshtein edit distance between two strings."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    if len(a) < len(b):
        a, b = b, a

    previous = list(range(len(b) + 1))
    for i, ch_a in enumerate(a, start=1):
        current = [i]
        for j, ch_b in enumerate(b, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            sub_cost = previous[j - 1] + (0 if ch_a == ch_b else 1)
            current.append(min(insert_cost, delete_cost, sub_cost))
        previous = current
    return previous[-1]


def canonicalize_speech(text: str) -> str:
    """Normalize text via spoken_form(), lowercasing, and punctuation stripping."""
    if not text or not text.strip():
        return ""
    expanded = spoken_form(text)
    expanded = re.sub(r"\bdollars(?=[a-zA-Z])", "dollars ", expanded)
    lowered = expanded.lower()
    cleaned = _NON_ALNUM_RE.sub(" ", lowered)
    return _WHITESPACE_RE.sub(" ", cleaned).strip()


def asr_vocabulary(
    brand_name: str,
    facts: list[str] | None = None,
    extra_terms: list[str] | None = None,
) -> list[str]:
    """Build concise ASR vocabulary hints (brand terms, domains, proper nouns)."""
    candidates: list[str] = []

    cleaned_brand = _WHITESPACE_RE.sub(" ", brand_name.strip()) if brand_name else ""
    if cleaned_brand:
        candidates.append(cleaned_brand)
        for token in cleaned_brand.split():
            if len(token) >= 3:
                candidates.append(token)

    for fact in facts or []:
        if not fact:
            continue
        for domain in _DOMAIN_RE.findall(fact):
            candidates.append(domain)
        for proper in _PROPER_OR_COINED_RE.findall(fact):
            if len(proper) >= 3:
                candidates.append(proper)

    for term in extra_terms or []:
        cleaned = _WHITESPACE_RE.sub(" ", term.strip()) if term else ""
        if cleaned:
            candidates.append(cleaned)

    seen: set[str] = set()
    out: list[str] = []
    for item in candidates:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
        if len(out) >= 24:
            break
    return out


_SHORT_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "our",
        "are",
        "for",
        "and",
        "try",
        "own",
        "new",
        "all",
        "one",
        "out",
        "you",
        "your",
        "her",
        "his",
        "its",
        "can",
        "get",
        "has",
        "had",
        "was",
        "not",
        "but",
        "too",
        "two",
        "see",
        "now",
        "way",
        "may",
        "who",
        "did",
        "how",
        "use",
        "any",
        "day",
        "big",
        "top",
        "pro",
        "buy",
        "pay",
        "run",
        "set",
        "win",
        "at",
        "in",
        "on",
        "to",
        "of",
        "by",
        "or",
        "is",
        "it",
        "be",
        "we",
        "he",
        "me",
        "my",
        "no",
        "so",
        "up",
        "go",
        "do",
        "if",
        "as",
    }
)


def _align_brand_terms(canon_text: str, brand_terms: list[str] | None) -> str:
    """Snap 1-to-3 token windows close in edit distance to coined brand terms."""
    if not canon_text or not brand_terms:
        return canon_text

    canon_brands = [canonicalize_speech(term) for term in brand_terms if term and term.strip()]
    canon_brands = [b for b in canon_brands if len(b) >= 3]
    if not canon_brands:
        return canon_text

    tokens = canon_text.split()

    def _best_at(pos: int) -> tuple[int, int, str] | None:
        best: tuple[int, int, str] | None = None
        for span in (1, 2, 3):
            if pos + span > len(tokens):
                continue
            window_joined = " ".join(tokens[pos : pos + span])
            window_compact = "".join(tokens[pos : pos + span])
            for brand in canon_brands:
                brand_compact = brand.replace(" ", "")
                if len(brand_compact) < 3:
                    max_dist = 0
                elif len(brand_compact) <= 5:
                    max_dist = 1
                else:
                    max_dist = max(2, len(brand_compact) // 3)
                dist = min(
                    levenshtein_distance(window_joined, brand),
                    levenshtein_distance(window_compact, brand_compact),
                )
                if dist > max_dist or abs(len(window_compact) - len(brand_compact)) > max_dist:
                    continue
                if dist > 0:
                    if len(window_compact) < 2:
                        continue
                    if len(brand_compact) == 3 and len(window_compact) != 3:
                        continue
                    if all(tok in _SHORT_STOPWORDS for tok in tokens[pos : pos + span]):
                        continue
                candidate = (dist, span, brand)
                if best is None or (candidate[0], candidate[1]) < (best[0], best[1]):
                    best = candidate
        return best

    out_tokens: list[str] = []
    idx = 0
    while idx < len(tokens):
        hit = _best_at(idx)
        if hit is not None:
            hit_dist, hit_span, hit_brand = hit
            if hit_span > 1 and idx + 1 < len(tokens):
                next_hit = _best_at(idx + 1)
                if next_hit is not None and next_hit[0] < hit_dist:
                    out_tokens.append(tokens[idx])
                    idx += 1
                    continue
            out_tokens.extend(hit_brand.split())
            idx += hit_span
        else:
            out_tokens.append(tokens[idx])
            idx += 1

    return " ".join(out_tokens)


def same_canon(
    expected: str,
    actual: str,
    brand_terms: list[str] | None = None,
    max_edit_ratio: float = 0.0,
) -> bool:
    """Compare expected vs transcribed speech using spoken_form + Levenshtein brand matching."""
    exp_canon = canonicalize_speech(expected)
    act_canon = canonicalize_speech(actual)

    if exp_canon == act_canon:
        return True
    if not exp_canon or not act_canon:
        return False

    act_aligned = _align_brand_terms(act_canon, brand_terms)

    if exp_canon == act_aligned:
        return True

    threshold = max(0.0, float(max_edit_ratio))
    if threshold <= 0.0:
        return False

    max_len = max(len(exp_canon), len(act_aligned))
    if max_len == 0:
        return True
    ratio = levenshtein_distance(exp_canon, act_aligned) / max_len
    return ratio <= threshold
