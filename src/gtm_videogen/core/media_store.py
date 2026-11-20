"""Path-based MediaStore avoiding base64 RAM churn and unsafe URL parsing.

Fixes:
- BUG-11: Resolves local paths, media://, /api/media/..., data: URLs, and https:// URLs cleanly.
- BUG-26: Strict data: URL validation + atomic temp-file write + shutil.move (cross-FS safe).
- P7: Stores files on disk and returns Path objects instead of passing multi-MB base64 strings.
"""

from __future__ import annotations

import base64
import binascii
import mimetypes
import re
import shutil
import tempfile
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

from gtm_videogen.core.http_client import ResilientHttpClient
from gtm_videogen.core.job_registry import CancellationToken
from gtm_videogen.core.ssrf import assert_public_url

_DATA_URL_RE = re.compile(
    r"^data:(?P<mime>[a-zA-Z0-9!#$&^_.+-]+/[a-zA-Z0-9!#$&^_.+-]+)(?:;[a-zA-Z0-9_.-]+=[^;,]+)*;base64,(?P<data>.+)$",
    re.DOTALL,
)
_SAFE_ID_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$")


class MediaStoreError(ValueError):
    """Raised when a media reference is malformed, unsafe, or missing."""


def parse_data_url(data_url: str) -> tuple[str, bytes]:
    """Strictly parse and decode a base64 data: URL (fixes BUG-11, BUG-26)."""
    if not isinstance(data_url, str) or not data_url.startswith("data:"):
        raise MediaStoreError("Not a valid data: URL")

    match = _DATA_URL_RE.match(data_url.strip())
    if not match:
        raise MediaStoreError("Malformed data: URL header (expected data:<mime>;base64,<payload>)")

    mime = match.group("mime").lower()
    raw_b64 = re.sub(r"\s+", "", match.group("data"))
    if not raw_b64:
        raise MediaStoreError("Empty base64 payload in data: URL")

    try:
        decoded = base64.b64decode(raw_b64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise MediaStoreError(f"Invalid base64 payload in data: URL: {exc}") from exc

    if not decoded:
        raise MediaStoreError("Decoded data: URL payload is empty")

    return mime, decoded


def _ext_for_mime(mime: str) -> str:
    known = {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
        "image/gif": ".gif",
        "video/mp4": ".mp4",
        "video/webm": ".webm",
        "audio/mpeg": ".mp3",
        "audio/wav": ".wav",
        "audio/x-wav": ".wav",
        "audio/aac": ".aac",
    }
    if mime in known:
        return known[mime]
    guessed = mimetypes.guess_extension(mime)
    return guessed or ".bin"


class MediaStore:
    """Filesystem-backed media store with atomic cross-device writes."""

    def __init__(
        self,
        base_dir: Path,
        *,
        http_client: ResilientHttpClient | None = None,
        resolve_dns_for_ssrf: bool = True,
    ) -> None:
        self.base_dir = Path(base_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._http_client = http_client
        self._resolve_dns_for_ssrf = resolve_dns_for_ssrf

    def save_bytes(
        self,
        data: bytes,
        *,
        ext: str = ".bin",
        subdir: str = "assets",
        media_id: str | None = None,
    ) -> tuple[str, Path]:
        """Atomically write bytes to a temp file and move into base_dir via shutil.move."""
        if not data:
            raise MediaStoreError("Cannot save empty media bytes")

        norm_ext = ext if ext.startswith(".") else f".{ext}"
        target_dir = (self.base_dir / subdir).resolve()
        self._ensure_within_base(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        mid = media_id or f"med_{uuid4().hex[:12]}{norm_ext}"
        if not _SAFE_ID_RE.match(mid):
            raise MediaStoreError(f"Unsafe media_id: {mid!r}")

        dest_path = (target_dir / mid).resolve()
        self._ensure_within_base(dest_path)

        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=target_dir,
                prefix=".tmp_media_",
                suffix=norm_ext,
                delete=False,
            ) as tmp:
                tmp_path = Path(tmp.name)
                tmp.write(data)
                tmp.flush()
            shutil.move(str(tmp_path), str(dest_path))
            tmp_path = None
        finally:
            if tmp_path is not None and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

        rel_id = dest_path.relative_to(self.base_dir).as_posix()
        return rel_id, dest_path

    def save_from_path(
        self,
        src: Path,
        *,
        subdir: str = "assets",
        media_id: str | None = None,
    ) -> tuple[str, Path]:
        """Copy an existing local file into the MediaStore atomically."""
        src_path = Path(src)
        if not src_path.is_file():
            raise MediaStoreError(f"Source media file does not exist: {src_path}")
        ext = src_path.suffix or ".bin"
        return self.save_bytes(src_path.read_bytes(), ext=ext, subdir=subdir, media_id=media_id)

    def lookup(self, media_id: str) -> Path:
        """Resolve a relative media_id within base_dir, preventing path traversal."""
        cleaned = media_id.lstrip("/")
        if not cleaned or ".." in Path(cleaned).parts:
            raise MediaStoreError(f"Unsafe media reference: {media_id!r}")
        candidate = (self.base_dir / cleaned).resolve()
        self._ensure_within_base(candidate)
        if not candidate.is_file():
            raise MediaStoreError(f"Media asset not found: {media_id!r}")
        return candidate

    async def resolve_to_path(
        self,
        ref: str | Path,
        *,
        max_download_bytes: int = 50 * 1024 * 1024,
        cancel: CancellationToken | None = None,
    ) -> Path:
        """Resolve local path, media://, /api/media/..., data:, or http(s):// to a local Path."""
        if isinstance(ref, Path):
            if not ref.is_file():
                raise MediaStoreError(f"Local media path does not exist: {ref}")
            return ref.resolve()

        if not isinstance(ref, str) or not ref.strip():
            raise MediaStoreError("Media reference must be a non-empty string or Path")

        value = ref.strip()

        if value.startswith("media://"):
            return self.lookup(value[len("media://") :])

        if value.startswith("/api/media/"):
            return self.lookup(value[len("/api/media/") :])

        if value.startswith("data:"):
            mime, payload = parse_data_url(value)
            ext = _ext_for_mime(mime)
            _, saved_path = self.save_bytes(payload, ext=ext, subdir="data_urls")
            return saved_path

        if value.startswith(("http://", "https://")):
            assert_public_url(value, resolve_dns=self._resolve_dns_for_ssrf)
            client = self._http_client
            owns_client = client is None
            if client is None:
                client = ResilientHttpClient(resolve_dns_for_ssrf=self._resolve_dns_for_ssrf)
            try:
                data = await client.get_bytes(
                    value,
                    public_only=True,
                    max_bytes=max_download_bytes,
                    cancel=cancel,
                )
            finally:
                if owns_client:
                    await client.aclose()

            parsed = urlparse(value)
            suffix = Path(parsed.path).suffix
            ext = suffix if suffix and len(suffix) <= 8 else ".bin"
            _, saved_path = self.save_bytes(data, ext=ext, subdir="downloads")
            return saved_path

        # Otherwise treat as local filesystem path
        local = Path(value)
        if not local.is_file():
            raise MediaStoreError(f"Local media file does not exist: {value}")
        return local.resolve()

    def _ensure_within_base(self, path: Path) -> None:
        try:
            path.relative_to(self.base_dir)
        except ValueError as exc:
            raise MediaStoreError(f"Path escapes MediaStore root: {path}") from exc
