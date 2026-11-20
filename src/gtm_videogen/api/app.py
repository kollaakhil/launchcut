"""FastAPI application factory (`create_app`) and default module-level `app`."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from fastapi import FastAPI

from gtm_videogen.api.middleware import (
    OriginSecurityMiddleware,
    RateLimitMiddleware,
    TTLRateLimiter,
)
from gtm_videogen.api.routes import (
    AppServices,
    PipelineRunnerProtocol,
    RunEventBroker,
    router,
)
from gtm_videogen.core.config import Settings, get_settings
from gtm_videogen.core.job_registry import JobRegistry
from gtm_videogen.core.media_store import MediaStore
from gtm_videogen.core.run_store import RunStore


def create_app(
    *,
    data_dir: Path | str | None = None,
    settings: Settings | None = None,
    run_store: RunStore | None = None,
    media_store: MediaStore | None = None,
    job_registry: JobRegistry | None = None,
    runner: PipelineRunnerProtocol | None = None,
    event_broker: RunEventBroker | None = None,
    allowed_origins: Sequence[str] | None = None,
    rate_limit_requests: int = 300,
    rate_limit_window_sec: float = 60.0,
    rate_limit_max_keys: int = 10_000,
) -> FastAPI:
    """Create and wire a configured FastAPI application instance."""
    resolved_settings = settings or get_settings()
    resolved_data_dir = (
        Path(data_dir).resolve() if data_dir is not None else resolved_settings.data_dir.resolve()
    )
    resolved_data_dir.mkdir(parents=True, exist_ok=True)

    resolved_jobs = job_registry or JobRegistry(
        max_concurrent=resolved_settings.max_concurrent_jobs
    )
    resolved_runs = run_store or RunStore(
        resolved_data_dir / "runs",
        job_registry=resolved_jobs,
    )
    resolved_media = media_store or MediaStore(resolved_data_dir / "media")
    resolved_broker = event_broker or RunEventBroker()

    application = FastAPI(
        title="gtm-videogen-deepagents API",
        version="0.1.0",
        description=(
            "Rubric-gated LangChain DeepAgents GTM video generation API with SSE streaming."
        ),
    )

    limiter = (
        TTLRateLimiter(
            max_requests=rate_limit_requests,
            window_sec=rate_limit_window_sec,
            max_keys=rate_limit_max_keys,
        )
        if rate_limit_requests > 0
        else None
    )

    # Inner middleware: TTL-pruned rate limiter
    if limiter is not None:
        application.add_middleware(RateLimitMiddleware, limiter=limiter)

    # Outer middleware: Safe Origin/CORS validation (BUG-20)
    application.add_middleware(OriginSecurityMiddleware, allowed_origins=allowed_origins)

    application.state.services = AppServices(
        settings=resolved_settings,
        data_dir=resolved_data_dir,
        run_store=resolved_runs,
        media_store=resolved_media,
        job_registry=resolved_jobs,
        event_broker=resolved_broker,
        runner=runner,
    )
    application.state.rate_limiter = limiter

    application.include_router(router)
    return application


app = create_app()
