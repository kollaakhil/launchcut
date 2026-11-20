"""Persistent CreativeHistory anti-repetition memory (Pillar 6)."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import urlparse

from pydantic import ValidationError

from gtm_videogen.models.schemas import CreativeFingerprint, CreativeHistory

MAX_HISTORY_PER_BRAND = 10


def normalize_brand_url(url: str) -> str:
    """Normalize a brand URL to a canonical host+path key for history grouping."""
    raw = url.strip().lower()
    if not raw.startswith(("http://", "https://")):
        raw = f"https://{raw}"
    parsed = urlparse(raw)
    host = (parsed.hostname or "").removeprefix("www.")
    path = parsed.path.rstrip("/")
    return f"{host}{path}" if path else host


class CreativeHistoryStore:
    """Atomic JSON store retaining the last 10 CreativeFingerprints per brand_url."""

    def __init__(self, store_path: Path, *, max_entries: int = MAX_HISTORY_PER_BRAND) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        self.store_path = Path(store_path).resolve()
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_entries = max_entries

    def _load_all(self) -> dict[str, object]:
        if not self.store_path.is_file():
            return {}
        try:
            data = json.loads(self.store_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def _save_all(self, data: dict[str, object]) -> None:
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.store_path.parent,
                prefix=".tmp_creative_history_",
                suffix=".json",
                delete=False,
            ) as tmp:
                tmp_path = Path(tmp.name)
                json.dump(data, tmp, indent=2, sort_keys=True)
                tmp.flush()
                os.fsync(tmp.fileno())
            os.replace(tmp_path, self.store_path)
            tmp_path = None
        finally:
            if tmp_path is not None and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    @staticmethod
    def _parse_entries(raw_entries: object) -> list[CreativeFingerprint]:
        if not isinstance(raw_entries, list):
            return []
        valid: list[CreativeFingerprint] = []
        for item in raw_entries:
            try:
                valid.append(CreativeFingerprint.model_validate(item))
            except (ValidationError, TypeError, ValueError):
                continue
        return valid

    def get_history(self, brand_url: str) -> CreativeHistory:
        """Return the CreativeHistory for a normalized brand_url."""
        key = normalize_brand_url(brand_url)
        raw_all = self._load_all()
        entries = self._parse_entries(raw_all.get(key, []))
        return CreativeHistory(brand_url=key, entries=entries[-self.max_entries :])

    def record(self, fingerprint: CreativeFingerprint) -> CreativeHistory:
        """Append a CreativeFingerprint and retain only the most recent max_entries (10)."""
        key = normalize_brand_url(fingerprint.brand_url)
        raw_all = self._load_all()
        valid_existing = self._parse_entries(raw_all.get(key, []))
        serialized = [json.loads(fp.model_dump_json()) for fp in valid_existing]
        serialized.append(json.loads(fingerprint.model_dump_json()))
        trimmed = serialized[-self.max_entries :]
        raw_all[key] = trimmed
        self._save_all(raw_all)
        return self.get_history(key)


def format_previously_explored(
    history: CreativeHistory | Sequence[CreativeFingerprint],
) -> str:
    """Format CreativeHistory as a 'Previously Explored (Do Not Repeat)' prompt section."""
    entries = history.entries if isinstance(history, CreativeHistory) else list(history)
    if not entries:
        return ""

    lines = [
        "Previously Explored (Do Not Repeat):",
        "Avoid reusing these narrative arcs, hook openings, visual worlds, lighting styles, "
        "camera styles, music genres, or voices:",
    ]
    for idx, fp in enumerate(entries, start=1):
        lines.append(
            f"{idx}. arc={fp.arc_name!r} | hook={fp.hook_opening!r} | "
            f"world={fp.visual_world!r} | lighting={fp.lighting_style!r} | "
            f"camera={fp.camera_style!r} | music={fp.music_genre or 'none'!r} | "
            f"voice={fp.voice_id or 'none'!r} | outcome={fp.outcome}"
        )
    return "\n".join(lines)
