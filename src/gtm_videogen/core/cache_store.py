"""Content-addressed SHA-256 persistent caches for clips, audio, and judge evaluations.

Implements P4:
- Persistent clipCache, audioCache, and judgeCache keyed by SHA-256 of canonical inputs.
- Always verifies artifact file existence AND non-empty size on disk before returning a hit.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_CACHE_KEY_RE = re.compile(r"^[a-zA-Z0-9_-]{1,128}$")


@dataclass(frozen=True)
class CacheEntry:
    """Metadata returned on a verified cache hit."""

    key: str
    namespace: str
    artifact_path: Path | None
    metadata: dict[str, Any]
    created_at: str


def compute_cache_key(namespace: str, payload: dict[str, Any]) -> str:
    """Compute a deterministic SHA-256 hex digest from namespace + canonical JSON payload."""
    canonical = json.dumps(
        {"namespace": namespace, "payload": payload},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ContentCache:
    """Single-namespace SHA-256 content-addressed cache."""

    def __init__(self, root_dir: Path, namespace: str, *, require_file: bool = True) -> None:
        self.namespace = namespace
        self.require_file = require_file
        self.root_dir = Path(root_dir).resolve() / namespace
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def _meta_path(self, key: str) -> Path:
        if not isinstance(key, str) or not _CACHE_KEY_RE.fullmatch(key):
            raise ValueError(f"Unsafe cache key: {key!r}")
        root = self.root_dir.resolve()
        candidate = (root / f"{key}.json").resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Cache key escapes cache root: {key!r}") from exc
        return candidate

    def make_key(self, payload: dict[str, Any]) -> str:
        return compute_cache_key(self.namespace, payload)

    def get(self, key: str) -> CacheEntry | None:
        """Return a CacheEntry only if metadata is valid and target file exists & is non-empty."""
        meta_path = self._meta_path(key)
        if not meta_path.is_file():
            return None

        try:
            raw = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
            meta_path.unlink(missing_ok=True)
            return None

        if not isinstance(raw, dict):
            meta_path.unlink(missing_ok=True)
            return None

        artifact_str = raw.get("artifact_path")
        if artifact_str is not None and not isinstance(artifact_str, str):
            meta_path.unlink(missing_ok=True)
            return None
        artifact_path = Path(artifact_str) if artifact_str else None

        if self.require_file:
            if artifact_path is None or not artifact_path.is_file():
                meta_path.unlink(missing_ok=True)
                return None
            try:
                if artifact_path.stat().st_size <= 0:
                    meta_path.unlink(missing_ok=True)
                    return None
            except OSError:
                meta_path.unlink(missing_ok=True)
                return None

        raw_metadata = raw.get("metadata", {})
        metadata_dict = dict(raw_metadata) if isinstance(raw_metadata, dict) else {}

        return CacheEntry(
            key=key,
            namespace=self.namespace,
            artifact_path=artifact_path,
            metadata=metadata_dict,
            created_at=str(raw.get("created_at", "")),
        )

    def put(
        self,
        key: str,
        *,
        artifact_path: Path | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CacheEntry:
        """Persist a cache entry atomically."""
        resolved_artifact: Path | None = None
        if artifact_path is not None:
            resolved_artifact = Path(artifact_path).resolve()
            if self.require_file and (
                not resolved_artifact.is_file() or resolved_artifact.stat().st_size <= 0
            ):
                raise ValueError(
                    f"Cannot cache missing or empty artifact file: {resolved_artifact}"
                )
        elif self.require_file:
            raise ValueError(f"Cache namespace {self.namespace!r} requires a non-empty artifact")

        created_at = datetime.now(UTC).isoformat()
        record = {
            "key": key,
            "namespace": self.namespace,
            "artifact_path": str(resolved_artifact) if resolved_artifact else None,
            "metadata": metadata or {},
            "created_at": created_at,
        }

        meta_path = self._meta_path(key)
        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.root_dir,
                prefix=".tmp_cache_",
                suffix=".json",
                delete=False,
            ) as tmp:
                tmp_path = Path(tmp.name)
                json.dump(record, tmp, sort_keys=True)
                tmp.flush()
                os.fsync(tmp.fileno())
            os.replace(tmp_path, meta_path)
            tmp_path = None
        finally:
            if tmp_path is not None and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

        return CacheEntry(
            key=key,
            namespace=self.namespace,
            artifact_path=resolved_artifact,
            metadata=dict(metadata or {}),
            created_at=created_at,
        )

    def invalidate(self, key: str) -> None:
        self._meta_path(key).unlink(missing_ok=True)


class CacheStore:
    """Container for persistent clipCache, audioCache, and judgeCache."""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = Path(cache_dir).resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.clip_cache = ContentCache(self.cache_dir, "clipCache", require_file=True)
        self.audio_cache = ContentCache(self.cache_dir, "audioCache", require_file=True)
        self.judge_cache = ContentCache(self.cache_dir, "judgeCache", require_file=False)
