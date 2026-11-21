"""SSRF-Safe Live Website Viewport Capture & Public Image Asset Downloader.

Captures a clean rendered webpage screenshot via Microlink's lightweight JSON screenshot API
(zero headless Chromium inside the 512MB Render container) and safely downloads public brand
logos / product hero images with strict SSRF validation and Pillow image verification.
"""

from __future__ import annotations

import io
from pathlib import Path
from urllib.parse import quote

from PIL import Image

from gtm_videogen.core.http_client import ResilientHttpClient
from gtm_videogen.core.job_registry import CancellationToken
from gtm_videogen.core.ssrf import assert_public_url

_MAX_IMAGE_BYTES = 6 * 1024 * 1024  # 6 MB safety cap


async def download_public_image(
    url: str,
    dest_path: Path,
    *,
    timeout_sec: float = 8.0,
    cancel: CancellationToken | None = None,
) -> Path | None:
    """Download and verify a public image URL into `dest_path`, returning None on failure."""
    clean = (url or "").strip()
    if not clean.startswith(("http://", "https://")):
        return None
    try:
        assert_public_url(clean)
        async with ResilientHttpClient(
            connect_timeout=5.0,
            read_timeout=timeout_sec,
            write_timeout=5.0,
            pool_timeout=5.0,
            max_retries=1,
            public_only=True,
        ) as client:
            raw = await client.get_bytes(clean, cancel=cancel)
        if not raw or len(raw) > _MAX_IMAGE_BYTES:
            return None
        with Image.open(io.BytesIO(raw)) as img:
            img.verify()
        with Image.open(io.BytesIO(raw)) as img2:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            if dest_path.suffix.lower() in (".jpg", ".jpeg"):
                img2.convert("RGB").save(dest_path, format="JPEG", quality=90)
            else:
                img2.convert("RGBA").save(dest_path, format="PNG")
        return dest_path
    except Exception:
        return None


async def capture_url_viewport_png(
    url: str,
    dest_path: Path,
    *,
    timeout_sec: float = 9.0,
    cancel: CancellationToken | None = None,
) -> Path | None:
    """Capture a live 1280x800 rendered screenshot of a public URL via Microlink JSON API."""
    clean = (url or "").strip()
    if not clean:
        return None
    if not clean.startswith(("http://", "https://")):
        clean = "https://" + clean
    try:
        assert_public_url(clean)
        api_url = (
            f"https://api.microlink.io/?url={quote(clean, safe='')}"
            "&screenshot=true&meta=false&viewport.width=1280&viewport.height=800"
        )
        async with ResilientHttpClient(
            connect_timeout=5.0,
            read_timeout=timeout_sec,
            write_timeout=5.0,
            pool_timeout=5.0,
            max_retries=0,
            public_only=True,
        ) as client:
            data = await client.get_json(api_url, cancel=cancel)
        if not isinstance(data, dict):
            return None
        shot_url = (
            data.get("data", {}).get("screenshot", {}).get("url")
            if isinstance(data.get("data"), dict)
            else None
        )
        if not isinstance(shot_url, str) or not shot_url.startswith(("http://", "https://")):
            return None
        return await download_public_image(shot_url, dest_path, timeout_sec=timeout_sec, cancel=cancel)
    except Exception:
        return None
