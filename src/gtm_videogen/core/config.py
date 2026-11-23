"""Application configuration loaded from environment variables and .env."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_fonts_dir() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    candidate = repo_root / "assets" / "fonts"
    return candidate if candidate.exists() else Path("assets/fonts")


class Settings(BaseSettings):
    """Typed runtime settings for gtm-videogen-deepagents."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # API Keys (all optional so offline mock mode runs without secrets)
    openrouter_api_key: str = Field(default="", alias="OPENROUTER_API_KEY")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    google_api_key: str = Field(default="", alias="GOOGLE_API_KEY")
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    hf_token: str = Field(default="", alias="HF_TOKEN")
    fal_key: str = Field(default="", alias="FAL_KEY")
    hf_api_key: str = Field(default="", alias="HF_API_KEY")
    hf_api_key_id: str = Field(default="", alias="HF_API_KEY_ID")
    hf_api_key_secret: str = Field(default="", alias="HF_API_KEY_SECRET")

    # Stage model defaults (live-verified active endpoints)
    fast_model: str = Field(default="anthropic/claude-haiku-5.5", alias="GTM_FAST_MODEL")
    flagship_model: str = Field(default="anthropic/claude-haiku-5.5", alias="GTM_FLAGSHIP_MODEL")
    text_model: str = Field(default="anthropic/claude-haiku-5.5", alias="GTM_TEXT_MODEL")
    vision_model: str = Field(default="google/gemini-3.8-flash", alias="GTM_VISION_MODEL")
    image_model: str = Field(default="fal-ai/flux/schnell", alias="GTM_IMAGE_MODEL")
    video_model: str = Field(
        default="kling/v4/flash/text-to-video",
        alias="GTM_VIDEO_MODEL",
    )
    voice_model: str = Field(default="fal-ai/elevenlabs/tts/eleven-v3", alias="GTM_VOICE_MODEL")
    sfx_model: str = Field(default="mirelo-ai/sfx2.0/text-to-audio", alias="GTM_SFX_MODEL")
    music_model: str = Field(
        default="fal-ai/stable-audio-25/text-to-audio",
        alias="GTM_MUSIC_MODEL",
    )

    # Storage paths
    data_dir: Path = Field(default=Path(".data"), alias="GTM_DATA_DIR")
    runs_dir: Path = Field(default=Path(".data/runs"), alias="GTM_RUNS_DIR")
    cache_dir: Path = Field(default=Path(".data/cache"), alias="GTM_CACHE_DIR")
    fonts_dir: Path = Field(default_factory=_default_fonts_dir, alias="GTM_FONTS_DIR")

    # Runtime limits & HTTP timeouts
    default_cap_usd: float = Field(default=5.0, ge=0.0, alias="GTM_DEFAULT_CAP_USD")
    max_concurrent_jobs: int = Field(default=4, ge=1, le=32, alias="GTM_MAX_CONCURRENT_JOBS")
    http_connect_timeout_sec: float = Field(
        default=15.0, gt=0.0, alias="GTM_HTTP_CONNECT_TIMEOUT_SEC"
    )
    http_read_timeout_sec: float = Field(default=90.0, gt=0.0, alias="GTM_HTTP_READ_TIMEOUT_SEC")
    http_write_timeout_sec: float = Field(default=30.0, gt=0.0, alias="GTM_HTTP_WRITE_TIMEOUT_SEC")
    http_pool_timeout_sec: float = Field(default=15.0, gt=0.0, alias="GTM_HTTP_POOL_TIMEOUT_SEC")
    http_max_retries: int = Field(default=3, ge=0, le=10, alias="GTM_HTTP_MAX_RETRIES")

    # Local media & Gemini CLI binaries
    ffmpeg_bin: str = Field(default="ffmpeg", alias="FFMPEG_BIN")
    ffprobe_bin: str = Field(default="ffprobe", alias="FFPROBE_BIN")
    tesseract_bin: str = Field(default="tesseract", alias="TESSERACT_BIN")
    gemini_cli_bin: str = Field(default="gemini-api", alias="GEMINI_CLI_BIN")
    ocr_lang: str = Field(default="eng", alias="OCR_LANG")
    whisper_bin: str = Field(default="", alias="WHISPER_BIN")
    whisper_model_path: str = Field(default="", alias="WHISPER_MODEL_PATH")
    whisper_lang: str = Field(default="en", alias="WHISPER_LANG")

    # API Server
    api_host: str = Field(default="0.0.0.0", alias="GTM_API_HOST")
    api_port: int = Field(default=8000, ge=1, le=65535, alias="GTM_API_PORT")
    allowed_origins: str = Field(
        default="http://localhost:3000,http://127.0.0.1:3000",
        alias="GTM_ALLOWED_ORIGINS",
    )
    log_level: str = Field(default="INFO", alias="GTM_LOG_LEVEL")

    def effective_gemini_key(self) -> str:
        """Return GEMINI_API_KEY, GOOGLE_API_KEY, GENIE_API_KEY, or `~/.config/gemini-api/config.yaml`."""
        import os
        import re

        direct = (
            self.gemini_api_key.strip()
            or self.google_api_key.strip()
            or os.getenv("GEMINI_API_KEY", "").strip()
            or os.getenv("GOOGLE_API_KEY", "").strip()
            or os.getenv("GENIE_API_KEY", "").strip()
            or os.getenv("GOOGLE_GEMINI_API_KEY", "").strip()
        )
        if direct:
            return direct
        cfg_file = Path.home() / ".config" / "gemini-api" / "config.yaml"
        if cfg_file.is_file():
            try:
                raw = cfg_file.read_text(encoding="utf-8", errors="ignore")
                m = re.search(r"""^\s*api[_-]?key\s*:\s*['"]?([A-Za-z0-9_.\-]{16,})['"]?\s*$""", raw, re.M)
                if m:
                    return m.group(1).strip()
            except Exception:
                pass
        return ""

    def allowed_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.allowed_origins.split(",") if origin.strip()]

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached singleton Settings instance."""
    return Settings()
