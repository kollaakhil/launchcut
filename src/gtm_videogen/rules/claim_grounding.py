"""Factual claim grounding, banned phrase detection, and placeholder rejection.

Fixes BUG-23: All claim/safety/placeholder errors are non-overridable blocking QAIssues.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from gtm_videogen.models.schemas import QAIssue, ResearchBundle

_CLAIM_TOKEN_RE = re.compile(
    r"(?i)"
    r"(?:#\d+\b)"
    r"|(?:\$\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:k|m|b|million|billion|thousand)\b)?\+?)"
    r"|(?:\b\d+(?:\.\d+)?\s*%)"
    r"|(?:\b\d+(?:\.\d+)?\s*x\b)"
    r"|(?:\b\d+(?:,\d{3})*(?:\.\d+)?\s*(?:million|billion|thousand|k|m|b)\b\+?)"
    r"|(?:\btop\s+\d+\b)"
    r"|(?:\branked\s+#?\d+\b)"
)

_GUARANTEE_RE = re.compile(
    r"(?i)\b(?:100%\s+guaranteed|money[- ]back\s+guarantee|risk[- ]free|"
    r"guaranteed\s+results?|guaranteed)\b"
)

_SUPERLATIVE_RE = re.compile(
    r"(?i)\b(?:"
    r"fastest|best|smartest|most\s+powerful|world'?s\s+first|world'?s\s+best|"
    r"number\s+one|#1|unmatched|unbeatable|the\s+only\s+one|industry[- ]leading|"
    r"market[- ]leading|cheapest|safest|strongest"
    r")\b"
)

_PLACEHOLDER_RE = re.compile(
    r"(?:\[(?:BRAND|PRODUCT|OFFER|URL|CTA|COMPANY|NAME|PRICE|STAT|LINK|INSERT[^\]]*)\])"
    r"|(?:\{\{[^}]+\}\})"
    r"|(?:<(?:BRAND|PRODUCT|OFFER|URL|CTA)>)"
    r"|(?:\b(?:TBD|TODO|FIXME|LOREM\s+IPSUM|PLACEHOLDER)\b)",
    re.IGNORECASE,
)

_BANNED_LINE_RE = re.compile(r"(?i)(?:not\s+verified|do\s+not\s+use|never\s+say|banned|forbidden)")
_QUOTED_PHRASE_RE = re.compile(r'"([^"]{2,120})"')


def _normalize_token(token: str) -> str:
    return re.sub(r"\s+", " ", token.strip().lower()).replace(",", "")


def _token_in_corpus(token: str, verified_corpus: str) -> bool:
    """Check whether token appears in verified_corpus with strict token boundaries.

    Prevents substring false-grounding such as "$49" matching "$499", "#1" matching "#10",
    "50%" matching "150%", or "best" matching "asbestos-free".
    """
    if not verified_corpus:
        return False
    norm = _normalize_token(token)
    if not norm:
        return False

    # Normalize optional spacing around currency/hash/percent symbols before building regex
    canonical = re.sub(r"([$#])\s+", r"\1", norm)
    canonical = re.sub(r"\s+%", "%", canonical)

    parts = re.split(r"(\s+)", canonical)
    escaped = "".join(r"\s+" if part.isspace() else re.escape(part) for part in parts)
    escaped = escaped.replace(r"\$", r"\$\s*").replace(r"\#", r"\#\s*").replace("%", r"\s*%")
    pattern = re.compile(rf"(?<![a-z0-9$#.]){escaped}(?![a-z0-9%]|\.\d)", re.IGNORECASE)
    return pattern.search(verified_corpus) is not None


def extract_claim_tokens(text: str) -> list[str]:
    """Extract numeric/ranking/guarantee claim tokens that require factual grounding."""
    if not text:
        return []
    tokens: list[str] = []
    seen: set[str] = set()
    for pattern in (_CLAIM_TOKEN_RE, _GUARANTEE_RE):
        for match in pattern.finditer(text):
            raw = match.group(0).strip()
            norm = _normalize_token(raw)
            if norm and norm not in seen:
                seen.add(norm)
                tokens.append(raw)
    return tokens


def extract_banned_phrases(research: ResearchBundle) -> list[str]:
    """Collect explicit banned phrases plus quoted phrases on 'not verified / do not use' lines."""
    phrases: list[str] = []
    seen: set[str] = set()

    for phrase in research.banned_phrases:
        cleaned = phrase.strip()
        if cleaned and cleaned.lower() not in seen:
            seen.add(cleaned.lower())
            phrases.append(cleaned)

    for line in (*research.product_facts, *research.verified_claims, *research.proof_points):
        if _BANNED_LINE_RE.search(line):
            for match in _QUOTED_PHRASE_RE.finditer(line):
                cleaned = match.group(1).strip()
                if cleaned and cleaned.lower() not in seen:
                    seen.add(cleaned.lower())
                    phrases.append(cleaned)

    return phrases


def find_placeholders(text: str) -> list[str]:
    """Return any unresolved template placeholders (e.g. [BRAND], TBD, TODO, {{...}})."""
    if not text:
        return []
    found: list[str] = []
    seen: set[str] = set()
    for match in _PLACEHOLDER_RE.finditer(text):
        raw = match.group(0).strip()
        key = raw.lower()
        if key not in seen:
            seen.add(key)
            found.append(raw)
    return found


def build_verified_corpus(
    research: ResearchBundle,
    *,
    extra_facts: Sequence[str] = (),
    offer: str | None = None,
) -> str:
    """Build normalized text corpus of all verified facts and proof points."""
    parts: list[str] = []
    for line in (
        *research.verified_claims,
        *research.product_facts,
        *research.proof_points,
        *research.voc_quotes,
        *extra_facts,
    ):
        if line and not _BANNED_LINE_RE.search(line):
            parts.append(line)
    if offer:
        parts.append(offer)
    return _normalize_token(" \n ".join(parts))


def find_unsupported_superlatives(text: str, verified_corpus: str) -> list[str]:
    """Return superlatives appearing in text that are not present in the verified corpus."""
    if not text:
        return []
    unsupported: list[str] = []
    seen: set[str] = set()
    for match in _SUPERLATIVE_RE.finditer(text):
        raw = match.group(0).strip()
        norm = _normalize_token(raw)
        if norm in seen:
            continue
        seen.add(norm)
        if not _token_in_corpus(norm, verified_corpus):
            unsupported.append(raw)
    return unsupported


def check_claim_grounding(
    text: str,
    research: ResearchBundle,
    *,
    extra_facts: Sequence[str] = (),
    offer: str | None = None,
    shot_id: str | None = None,
) -> list[QAIssue]:
    """Verify claims, banned phrases, superlatives, and placeholders.

    Returns non-overridable blocking QAIssues (fixes BUG-23).
    """
    issues: list[QAIssue] = []
    if not text or not text.strip():
        return issues

    verified_corpus = build_verified_corpus(research, extra_facts=extra_facts, offer=offer)
    text_lower = text.lower()

    # 1. Unresolved placeholders
    for placeholder in find_placeholders(text):
        issues.append(
            QAIssue(
                code="UNRESOLVED_PLACEHOLDER",
                severity="blocking",
                message=f"Unresolved placeholder found in copy: {placeholder!r}",
                shot_id=shot_id,
                overridable=False,
                details={"placeholder": placeholder},
            )
        )

    # 2. Banned phrases
    for phrase in extract_banned_phrases(research):
        if phrase.lower() in text_lower:
            issues.append(
                QAIssue(
                    code="BANNED_PHRASE",
                    severity="blocking",
                    message=f"Banned or unverified phrase used in copy: {phrase!r}",
                    shot_id=shot_id,
                    overridable=False,
                    details={"phrase": phrase},
                )
            )

    # 3. Numeric / ranking / guarantee claim tokens
    for token in extract_claim_tokens(text):
        if not _token_in_corpus(token, verified_corpus):
            issues.append(
                QAIssue(
                    code="UNGROUNDED_CLAIM",
                    severity="blocking",
                    message=(
                        f"Ungrounded numeric or guarantee claim not in verified facts: {token!r}"
                    ),
                    shot_id=shot_id,
                    overridable=False,
                    details={"claim_token": token},
                )
            )

    # 4. Unsupported superlatives
    for sup in find_unsupported_superlatives(text, verified_corpus):
        issues.append(
            QAIssue(
                code="UNSUPPORTED_SUPERLATIVE",
                severity="blocking",
                message=f"Unsupported superlative not backed by verified facts: {sup!r}",
                shot_id=shot_id,
                overridable=False,
                details={"superlative": sup},
            )
        )

    return issues
