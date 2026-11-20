"""Anti-garbled-text guardrails for AI video generation prompts."""

from __future__ import annotations

import re

DEFAULT_NEGATIVE_PROMPT = (
    "garbled text, illegible letters, misspelled typography, distorted watermark, "
    "corrupted UI elements, warped fingers, flickering artifacts, low resolution"
)

_QUOTED_TEXT_RE = re.compile(r'["“][^"”]{3,80}["”]')
_TEXT_REQUEST_RE = re.compile(
    r"(?i)\b(?:"
    r"(?:text|words?|letters?|headline|caption|title|subtitle|slogan|quote|copy)\s+"
    r"(?:reading|saying|that\s+says|displayed|written|printed|floating|appearing|on\s+screen)"
    r"|(?:display|show|render|write|print|spell)\s+(?:the\s+)?(?:text|words?|headline|title|url|website|logo\s+text)"
    r"|(?:dashboard|app|ui|webpage|screen|mobile)\s+screenshot"
    r"|(?:readable|legible|detailed)\s+"
    r"(?:ui|interface|dashboard|screen|webpage|spreadsheet|terminal|code|sign|billboard|menu|"
    r"newspaper|metrics|stats|numbers|charts|text|copy|letters?|words?)"
    r"|(?:close[- ]up\s+of\s+(?:a\s+)?(?:phone|laptop|computer|monitor)\s+screen\s+showing\s+"
    r"(?:text|ui|app|dashboard|messages|emails))"
    r"|(?:neon\s+sign\s+(?:with\s+(?:glowing\s+|bright\s+)?(?:words?|text|letters?)|reading|saying)"
    r"|billboard\s+(?:reading|saying)|watermark)"
    r")\b"
)


def rendered_text_problem(prompt: str) -> str | None:
    """Reject AI video prompts that ask the generator to bake readable text/UI into video frames."""
    if not prompt or not prompt.strip():
        return None

    match = _TEXT_REQUEST_RE.search(prompt)
    if match:
        return (
            f"Visual prompt asks AI video generator to render readable text/UI "
            f"({match.group(0)!r}); move readable copy to on_screen_text so the "
            "caption/graphics overlay renders it cleanly."
        )

    quoted = _QUOTED_TEXT_RE.search(prompt)
    if quoted and re.search(
        r"(?i)\b(sign|screen|text|title|card|label|banner|overlay|board|says|reads|written|"
        r"t-shirt|shirt|tee|hoodie|jersey|jacket|cap|hat|chest|printed|mug|bottle|poster|"
        r"packaging|wall)\b",
        prompt,
    ):
        return (
            f"Visual prompt includes quoted baked-in text ({quoted.group(0)}); "
            "move readable text to on_screen_text instead of baking it into AI video frames."
        )

    return None
