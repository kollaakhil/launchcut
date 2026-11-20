"""Typer CLI entrypoint (`gtm-videogen`) for running pipelines, serving the API, and diagnostics."""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from typing import Annotated, Any
from uuid import uuid4

import typer
import uvicorn
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from gtm_videogen.api.routes import PipelineRunnerProtocol
from gtm_videogen.core.config import Settings, get_settings
from gtm_videogen.media.manim_renderer import manim_diagnostics
from gtm_videogen.models.schemas import CostLedger, RunInput, RunState

app = typer.Typer(
    name="gtm-videogen",
    help="Rubric-gated LangChain DeepAgents pipeline for GTM video generation.",
    no_args_is_help=True,
)
console = Console()


def _build_pipeline_runner(
    *,
    run_state: RunState,
    work_dir: Path,
) -> PipelineRunnerProtocol:
    """Lazy-import `PipelineRunner` so CLI imports cleanly regardless of M4 build state."""
    from gtm_videogen.agents.pipeline import PipelineRunner
    from gtm_videogen.rules.creative_history import CreativeHistoryStore

    history_store = CreativeHistoryStore(work_dir / "creative_history.json")
    if run_state.input.mock:
        from gtm_videogen.providers.mocks import MockProviderBundle

        bundle = MockProviderBundle.create(work_dir=work_dir)
        return PipelineRunner(
            providers=bundle,
            work_dir=work_dir,
            history_store=history_store,
        )
    return PipelineRunner(
        work_dir=work_dir,
        history_store=history_store,
    )


def _format_api_key_checks(settings: Settings) -> list[tuple[str, bool, str]]:
    return [
        (
            "OPENROUTER_API_KEY",
            bool(settings.openrouter_api_key),
            "LLM / Vision / OpenRouter Video",
        ),
        ("OPENAI_API_KEY", bool(settings.openai_api_key), "OpenAI LLM direct"),
        ("ANTHROPIC_API_KEY", bool(settings.anthropic_api_key), "Anthropic Claude direct"),
        ("GOOGLE_API_KEY", bool(settings.google_api_key), "Google Gemini direct"),
        ("FAL_KEY", bool(settings.fal_key), "fal.ai Video / TTS / SFX / Music / Whisper"),
        (
            "HF_API_KEY",
            bool(settings.hf_api_key or (settings.hf_api_key_id and settings.hf_api_key_secret)),
            "Higgsfield Video",
        ),
    ]


@app.command("run")
def run_pipeline(
    brand_name: Annotated[
        str,
        typer.Option("--brand-name", "--brand", help="Brand name for the GTM video."),
    ],
    brand_url: Annotated[
        str,
        typer.Option("--brand-url", "--url", help="Public brand URL (http:// or https://)."),
    ],
    goal: Annotated[
        str,
        typer.Option("--goal", help="Primary campaign conversion or awareness goal."),
    ],
    offer: Annotated[
        str | None,
        typer.Option("--offer", help="Optional promotional offer."),
    ] = None,
    facts: Annotated[
        list[str] | None,
        typer.Option("--fact", help="Verified brand fact (repeatable)."),
    ] = None,
    duration: Annotated[
        float,
        typer.Option(
            "--duration",
            "--target-duration-sec",
            help="Target video duration in seconds (6-60).",
        ),
    ] = 15.0,
    aspect: Annotated[
        str,
        typer.Option("--aspect", help="Aspect ratio: 16:9, 9:16, 1:1, or 4:5."),
    ] = "16:9",
    fps: Annotated[
        int,
        typer.Option("--fps", help="Output frame rate: 24 or 30."),
    ] = 24,
    audio_mode: Annotated[
        str,
        typer.Option("--audio-mode", help="Audio mode: vo, foley, or both."),
    ] = "vo",
    cap_usd: Annotated[
        float,
        typer.Option("--cap-usd", help="Hard USD spend budget cap."),
    ] = 5.0,
    render_mode: Annotated[
        str,
        typer.Option("--render-mode", help="Render quality mode: preview or final."),
    ] = "preview",
    mock: Annotated[
        bool,
        typer.Option(
            "--mock/--no-mock",
            help="Run offline with deterministic mock providers + real FFmpeg.",
        ),
    ] = False,
    output_dir: Annotated[
        Path | None,
        typer.Option("--output-dir", help="Directory for rendered MP4 and run JSON artifacts."),
    ] = None,
) -> None:
    """Execute the 3-agent rubric-gated video generation pipeline end-to-end."""
    try:
        run_input = RunInput(
            brand_name=brand_name,
            brand_url=brand_url,
            goal=goal,
            offer=offer,
            facts=list(facts or []),
            target_duration_sec=duration,
            aspect=aspect,  # type: ignore[arg-type]
            fps=fps,  # type: ignore[arg-type]
            audio_mode=audio_mode,  # type: ignore[arg-type]
            cap_usd=cap_usd,
            render_mode=render_mode,  # type: ignore[arg-type]
            mock=mock,
        )
    except ValidationError as exc:
        console.print(f"[red]Invalid run input:[/red] {exc}")
        raise typer.Exit(code=2) from exc

    run_id = f"run_{uuid4().hex[:12]}"
    resolved_out_dir = (
        output_dir.resolve()
        if output_dir is not None
        else (Path.cwd() / "output" / run_id).resolve()
    )
    resolved_out_dir.mkdir(parents=True, exist_ok=True)

    run_state = RunState(
        run_id=run_id,
        created_by="cli",
        status="working",
        input=run_input,
        cost_ledger=CostLedger(run_id=run_id, cap_usd=run_input.cap_usd),
    )

    events_seen: list[dict[str, Any]] = []

    def _on_event(ev: dict[str, Any]) -> None:
        events_seen.append(ev)
        ev_type = ev.get("type", "event")
        stage = ev.get("stage") or "pipeline"
        if ev_type in {
            "stage_started",
            "stage_completed",
            "warning",
            "run_completed",
            "run_failed",
        }:
            console.print(f"[cyan][{stage}][/cyan] {ev_type}")

    runner = _build_pipeline_runner(run_state=run_state, work_dir=resolved_out_dir)

    try:
        final_state = asyncio.run(runner.run(run_state, event_cb=_on_event))
    except Exception as exc:
        run_state.status = "failed"
        run_state.error = str(exc)
        state_file = resolved_out_dir / f"run_{run_id}.json"
        state_file.write_text(run_state.model_dump_json(indent=2), encoding="utf-8")
        console.print(f"[red]Pipeline failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    # Persist final RunState JSON inside output_dir
    state_file = resolved_out_dir / f"run_{final_state.run_id}.json"
    state_file.write_text(final_state.model_dump_json(indent=2), encoding="utf-8")

    # Summary output
    stitched_path = (
        final_state.render_manifest.stitched_path
        if final_state.render_manifest is not None
        else None
    )
    console.print("\n[bold]Run Summary[/bold]")
    console.print(f"  Run ID:     {final_state.run_id}")
    console.print(f"  Status:     {final_state.status}")
    console.print(f"  Output MP4: {stitched_path or '(none)'}")
    console.print(f"  State JSON: {state_file}")

    ledger = final_state.cost_ledger
    console.print(
        f"  CostLedger: ${ledger.actual_usd:.4f} / ${ledger.cap_usd:.2f} "
        f"(estimated: ${ledger.estimated_usd:.4f}, entries: {len(ledger.entries)})"
    )

    qa = final_state.qa_report
    if qa is not None:
        qa_status = "[green]PASSED[/green]" if qa.passed else "[red]FAILED[/red]"
        lufs_str = f"{qa.loudness_lufs:.1f} LUFS" if qa.loudness_lufs is not None else "n/a"
        console.print(
            f"  QAReport:   {qa_status} | originality={qa.originality_score:.1f}/10 "
            f"| coherence={qa.bible_coherence_score:.1f}/10 | loudness={lufs_str} "
            f"| blocking={len(qa.blocking_errors)} | warnings={len(qa.warnings)}"
        )
    else:
        console.print("  QAReport:   (not generated)")

    if final_state.warnings:
        console.print(f"  Warnings:   {len(final_state.warnings)}")
        for warn in final_state.warnings:
            console.print(f"    - {warn}")

    passed = final_state.status == "done" and (qa is None or qa.passed)
    if not passed:
        if final_state.error:
            console.print(f"[red]Error:[/red] {final_state.error}")
        raise typer.Exit(code=1)


@app.command("serve")
def serve_api(
    host: Annotated[str, typer.Option("--host", help="Bind host address.")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", help="Bind TCP port.")] = 8000,
    reload: Annotated[
        bool,
        typer.Option("--reload/--no-reload", help="Enable auto-reload for development."),
    ] = False,
) -> None:
    """Launch the FastAPI SSE server via uvicorn."""
    console.print(f"Starting gtm-videogen API server on http://{host}:{port}")
    uvicorn.run("gtm_videogen.api.app:app", host=host, port=port, reload=reload)


@app.command("doctor")
def doctor() -> None:
    """Check local media tools, Manim diagnostics, bundled fonts, and provider API keys."""
    settings = get_settings()

    ffmpeg_path = shutil.which(settings.ffmpeg_bin)
    ffprobe_path = shutil.which(settings.ffprobe_bin)
    tesseract_path = shutil.which(settings.tesseract_bin)
    manim_info = manim_diagnostics()

    fonts_dir = settings.fonts_dir
    ttf_files = sorted(fonts_dir.glob("*.ttf")) if fonts_dir.is_dir() else []

    tools_table = Table(title="Local Toolchain & Assets")
    tools_table.add_column("Component", style="bold")
    tools_table.add_column("Status")
    tools_table.add_column("Details")

    tools_table.add_row(
        "ffmpeg",
        "[green]OK[/green]" if ffmpeg_path else "[red]MISSING[/red]",
        ffmpeg_path or f"{settings.ffmpeg_bin} not found on PATH",
    )
    tools_table.add_row(
        "ffprobe",
        "[green]OK[/green]" if ffprobe_path else "[red]MISSING[/red]",
        ffprobe_path or f"{settings.ffprobe_bin} not found on PATH",
    )
    tools_table.add_row(
        "tesseract",
        "[green]OK[/green]" if tesseract_path else "[yellow]OPTIONAL (MISSING)[/yellow]",
        tesseract_path or "Install tesseract-ocr for on-screen text OCR QA",
    )

    manim_ok = bool(manim_info.get("available"))
    latex_note = "yes" if manim_info.get("latex_available") else "no (non-LaTeX Pango/Cairo only)"
    manim_detail = (
        f"manim {manim_info.get('manim_version')} "
        f"(pycairo {manim_info.get('pycairo_version')}, "
        f"manimpango {manim_info.get('manimpango_version')}, "
        f"latex={latex_note})"
        if manim_ok
        else (
            "Optional extra not installed (`uv sync --extra manim`; "
            "requires libcairo2-dev libpango1.0-dev pkg-config gcc)"
        )
    )
    tools_table.add_row(
        "manim",
        "[green]OK[/green]" if manim_ok else "[yellow]OPTIONAL (NOT INSTALLED)[/yellow]",
        manim_detail,
    )

    fonts_ok = len(ttf_files) > 0
    tools_table.add_row(
        "Bundled Fonts",
        "[green]OK[/green]" if fonts_ok else "[yellow]MISSING[/yellow]",
        f"{len(ttf_files)} TTF files in {fonts_dir}"
        if fonts_ok
        else f"No .ttf files found in {fonts_dir}",
    )

    console.print(tools_table)

    key_checks = _format_api_key_checks(settings)
    keys_table = Table(title="Provider API Keys (Required for Live Mode Only)")
    keys_table.add_column("Env Var", style="bold")
    keys_table.add_column("Status")
    keys_table.add_column("Used For")

    any_llm_key = False
    for env_name, is_set, purpose in key_checks:
        if is_set and env_name in (
            "OPENROUTER_API_KEY",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "GOOGLE_API_KEY",
        ):
            any_llm_key = True
        keys_table.add_row(
            env_name,
            "[green]SET[/green]" if is_set else "[dim]NOT SET[/dim]",
            purpose,
        )
    console.print(keys_table)

    offline_ready = bool(ffmpeg_path and ffprobe_path)
    console.print("\n[bold]Honest Readiness Summary[/bold]")
    console.print(
        f"  Offline mode (`--mock` + real FFmpeg/ffprobe"
        f"{'/Tesseract' if tesseract_path else ''}"
        f"{'/Manim' if manim_ok else ''}): "
        + ("[green]READY[/green]" if offline_ready else "[red]MISSING FFMPEG/FFPROBE[/red]")
    )
    console.print(
        "  Live provider mode (real LLM/video/audio APIs): "
        + (
            "[green]API KEYS DETECTED[/green]"
            if (any_llm_key and bool(settings.fal_key))
            else "[yellow]REQUIRES API KEYS (e.g. OPENROUTER_API_KEY + FAL_KEY)[/yellow]"
        )
    )

    if not offline_ready:
        raise typer.Exit(code=1)
