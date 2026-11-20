"""Shared DeepAgents + RubricMiddleware helpers for gtm-videogen-deepagents.

Enforces Survey 1 verified contracts for `deepagents==0.7.41`:
- Always passes `model` explicitly to both `create_deep_agent` and `RubricMiddleware`.
- Captures terminal verdicts via BOTH `on_evaluation` callback AND `MemorySaver`
  checkpointer (`agent.get_state(config).values.get("_rubric_status")`), never
  assuming `AIMessage` presence implies `satisfied`.
- Persists structured JSON artifacts to a run workspace so grader `@tool` functions
  inspect full payloads directly (bypassing RubricMiddleware's 4,000-char transcript
  truncation).
- Forwards every `RubricEvaluation` to the pipeline `event_cb` (sync or async) using
  the canonical SSE event envelope.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import tempfile
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from deepagents import RubricMiddleware, create_deep_agent
from deepagents.middleware.rubric import RubricEvaluation
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import MemorySaver

from gtm_videogen.core.config import Settings, get_settings
from gtm_videogen.core.job_registry import CancellationToken
from gtm_videogen.providers.llm_router import LLMRouter
from gtm_videogen.providers.mocks import MockProviderBundle, ScriptedChatModel

EventCallback = Callable[[dict[str, Any]], Awaitable[None] | None]

LIVE_MODE_ERROR_MSG = (
    "Live mode (mock=False) requires an injected chat_model or a configured LLM API key "
    "(e.g., OPENROUTER_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY / GOOGLE_API_KEY). "
    "Pass mock=True for offline mode."
)


class RubricGateError(RuntimeError):
    """Raised when a rubric-gated DeepAgent terminates without `satisfied`."""

    def __init__(
        self,
        *,
        stage: str,
        status: str,
        evaluations: list[dict[str, Any]],
        message: str | None = None,
    ) -> None:
        self.stage = stage
        self.status = status
        self.evaluations = evaluations
        last_explanation = evaluations[-1].get("explanation", "") if evaluations else ""
        detail = message or (
            f"Rubric gate for stage {stage!r} terminated with status={status!r}"
            + (f": {last_explanation}" if last_explanation else "")
        )
        super().__init__(detail)


@dataclass
class RubricRunOutcome:
    """Structured terminal result of invoking a rubric-gated DeepAgent."""

    stage: str
    status: str
    passed: bool
    evaluations: list[dict[str, Any]] = field(default_factory=list)
    final_message: str = ""
    thread_id: str = ""


def make_sse_event(
    *,
    event_type: str,
    run_id: str,
    stage: str | None,
    data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a canonical SSE event payload matching PROJECT.md's contract."""
    return {
        "type": event_type,
        "run_id": run_id,
        "timestamp": datetime.now(UTC).isoformat(),
        "stage": stage,
        "data": dict(data or {}),
    }


async def emit_event(
    event_cb: EventCallback | None,
    *,
    event_type: str,
    run_id: str,
    stage: str | None,
    data: dict[str, Any] | None = None,
) -> None:
    """Emit an SSE event to a sync or async callback."""
    if event_cb is None:
        return
    payload = make_sse_event(
        event_type=event_type,
        run_id=run_id,
        stage=stage,
        data=data,
    )
    res = event_cb(payload)
    if inspect.isawaitable(res):
        await res


def write_workspace_json(workspace: Path, filename: str, payload: Any) -> Path:
    """Atomically write a JSON artifact inside the agent workspace directory."""
    workspace.mkdir(parents=True, exist_ok=True)
    dest = workspace / filename
    if hasattr(payload, "model_dump"):
        serialized = json.dumps(payload.model_dump(mode="json"), indent=2)
    else:
        serialized = json.dumps(payload, indent=2, default=str)

    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=dest.parent,
            prefix=f".tmp_{dest.stem}_",
            suffix=".json",
            delete=False,
        ) as tmp:
            tmp_path = Path(tmp.name)
            tmp.write(serialized)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp_path, dest)
        tmp_path = None
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
    return dest


def read_workspace_json(workspace: Path, filename: str) -> Any | None:
    """Read and parse a JSON artifact from the workspace, returning None if absent."""
    path = workspace / filename
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def is_default_unscripted_mock_model(model: Any) -> bool:
    """Return True if `model` is a ScriptedChatModel with no scripted responses or responder."""
    return (
        isinstance(model, ScriptedChatModel)
        and len(model.responses) == 0
        and model.responder is None
    )


def is_offline_provider_bundle(providers: Any) -> bool:
    """Return True if `providers` is an explicit offline MockProviderBundle."""
    return isinstance(providers, MockProviderBundle)


def ensure_live_llm_configured(settings: Settings | None = None) -> Settings:
    """Validate that at least one live LLM API key is configured; raise RuntimeError otherwise."""
    cfg = settings or get_settings()
    if not any(
        (
            cfg.openrouter_api_key.strip(),
            cfg.openai_api_key.strip(),
            cfg.anthropic_api_key.strip(),
            cfg.google_api_key.strip(),
        )
    ):
        raise RuntimeError(LIVE_MODE_ERROR_MSG)
    return cfg


def resolve_live_chat_model(
    stage: str,
    *,
    providers: Any = None,
    settings: Settings | None = None,
) -> BaseChatModel:
    """Construct a real LangChain BaseChatModel using LLMRouter + Settings for live mode.

    Raises RuntimeError with an explicit, honest error when no LLM API key is configured
    or when the required LangChain provider integration package is not installed.
    """
    router: LLMRouter = getattr(providers, "llm_router", None) or LLMRouter(
        settings=settings or get_settings()
    )
    cfg = ensure_live_llm_configured(settings or router.settings)
    model_id = router.resolve_model(stage)
    opts = router.build_chat_options(model_id, stage)
    norm_id = model_id.strip().lower()

    # 1. Anthropic (directly via langchain_anthropic when ANTHROPIC_API_KEY is configured)
    if cfg.anthropic_api_key.strip() and (
        "claude" in norm_id
        or norm_id.startswith("anthropic/")
        or not any(
            (cfg.openrouter_api_key.strip(), cfg.openai_api_key.strip(), cfg.google_api_key.strip())
        )
    ):
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is configured, but 'langchain-anthropic' is not installed."
            ) from exc
        bare_model = (
            model_id.split("/", 1)[1] if model_id.lower().startswith("anthropic/") else model_id
        )
        kwargs: dict[str, Any] = {
            "model": bare_model,
            "api_key": cfg.anthropic_api_key.strip(),
        }
        if "temperature" in opts:
            kwargs["temperature"] = opts["temperature"]
        if "max_tokens" in opts:
            kwargs["max_tokens"] = opts["max_tokens"]
        return ChatAnthropic(**kwargs)

    # 2. Google Gemini (directly via langchain_google_genai when GOOGLE_API_KEY is configured)
    if cfg.google_api_key.strip() and (
        "gemini" in norm_id
        or norm_id.startswith("google/")
        or not any((cfg.openrouter_api_key.strip(), cfg.openai_api_key.strip()))
    ):
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:
            raise RuntimeError(
                "GOOGLE_API_KEY is configured, but 'langchain-google-genai' is not installed."
            ) from exc
        bare_model = (
            model_id.split("/", 1)[1] if model_id.lower().startswith("google/") else model_id
        )
        kwargs = {
            "model": bare_model,
            "google_api_key": cfg.google_api_key.strip(),
        }
        if "temperature" in opts:
            kwargs["temperature"] = opts["temperature"]
        if "max_tokens" in opts:
            kwargs["max_output_tokens"] = opts["max_tokens"]
        return ChatGoogleGenerativeAI(**kwargs)

    # 3. OpenRouter / OpenAI (via langchain_openai if installed)
    if cfg.openrouter_api_key.strip() or cfg.openai_api_key.strip():
        try:
            from langchain_openai import ChatOpenAI  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "OPENROUTER_API_KEY / OPENAI_API_KEY is configured, but 'langchain-openai' is not "
                "installed. Install langchain-openai or set ANTHROPIC_API_KEY / GOOGLE_API_KEY."
            ) from exc
        if cfg.openrouter_api_key.strip():
            kwargs = {
                "model": model_id,
                "api_key": cfg.openrouter_api_key.strip(),
                "base_url": "https://openrouter.ai/api/v1",
            }
        else:
            bare_model = (
                model_id.split("/", 1)[1] if model_id.lower().startswith("openai/") else model_id
            )
            kwargs = {
                "model": bare_model,
                "api_key": cfg.openai_api_key.strip(),
            }
        if "temperature" in opts:
            kwargs["temperature"] = opts["temperature"]
        if "max_tokens" in opts:
            kwargs["max_tokens"] = opts["max_tokens"]
        return ChatOpenAI(**kwargs)

    raise RuntimeError(LIVE_MODE_ERROR_MSG)


def make_tool_driven_grader_model(
    tool_name: str,
    *,
    tool_args_factory: Callable[[], dict[str, Any]] | None = None,
    model_name: str = "scripted-rubric-grader",
) -> ScriptedChatModel:
    """Build an offline grader `BaseChatModel` that genuinely calls its verification `@tool`
    and translates the deterministic tool JSON output into a `GraderResponse` tool call.
    """
    call_counter = {"n": 0}

    def _responder(messages: list[BaseMessage]) -> dict[str, Any]:
        call_counter["n"] += 1
        last_tool_msg: ToolMessage | None = None
        for msg in reversed(messages):
            if isinstance(msg, ToolMessage) and getattr(msg, "name", None) == tool_name:
                last_tool_msg = msg
                break
            if isinstance(msg, HumanMessage):
                break

        if last_tool_msg is None:
            args = tool_args_factory() if tool_args_factory is not None else {}
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": f"tc_grade_tool_{call_counter['n']}",
                        "name": tool_name,
                        "args": args,
                        "type": "tool_call",
                    }
                ],
            }

        raw_content = (
            last_tool_msg.content
            if isinstance(last_tool_msg.content, str)
            else json.dumps(last_tool_msg.content)
        )
        try:
            verdict_data = json.loads(raw_content)
        except json.JSONDecodeError:
            verdict_data = {
                "result": "needs_revision",
                "explanation": f"Grader tool returned non-JSON output: {raw_content[:200]}",
                "criteria": [
                    {
                        "name": "tool_output_valid",
                        "passed": False,
                        "gap": "Verification tool output was not valid JSON.",
                    }
                ],
            }

        criteria = verdict_data.get("criteria") or [
            {"name": "verification_passed", "passed": bool(verdict_data.get("passed", False))}
        ]
        # Ensure every criterion conforms strictly to CriterionPass | CriterionFail
        normalized_criteria: list[dict[str, Any]] = []
        for c in criteria:
            passed_flag = bool(c.get("passed", False))
            if passed_flag:
                normalized_criteria.append(
                    {"name": str(c.get("name", "criterion")), "passed": True}
                )
            else:
                normalized_criteria.append(
                    {
                        "name": str(c.get("name", "criterion")),
                        "passed": False,
                        "gap": str(c.get("gap") or "Criterion failed verification."),
                    }
                )

        all_passed = all(c["passed"] for c in normalized_criteria)
        explicit_passed = verdict_data.get("passed")
        requested_result = str(verdict_data.get("result") or "")
        if requested_result == "failed":
            final_result = "failed"
        elif explicit_passed is False or requested_result == "needs_revision":
            final_result = "needs_revision"
            if all_passed:
                blocking_detail = str(
                    verdict_data.get("explanation")
                    or "Verification tool reported blocking issues (passed=False)."
                )
                normalized_criteria.append(
                    {
                        "name": "tool_verdict_passed",
                        "passed": False,
                        "gap": blocking_detail,
                    }
                )
                all_passed = False
        elif all_passed:
            final_result = "satisfied"
        else:
            final_result = "needs_revision"

        explanation = str(
            verdict_data.get("explanation")
            or ("All rubric criteria satisfied." if all_passed else "One or more criteria failed.")
        )

        return {
            "content": "",
            "tool_calls": [
                {
                    "id": f"tc_grader_resp_{call_counter['n']}",
                    "name": "GraderResponse",
                    "args": {
                        "result": final_result,
                        "explanation": explanation,
                        "criteria": normalized_criteria,
                    },
                    "type": "tool_call",
                }
            ],
        }

    return ScriptedChatModel(responder=_responder, model_name=model_name)


def build_rubric_deep_agent(
    *,
    main_model: BaseChatModel | str,
    grader_model: BaseChatModel | str,
    system_prompt: str,
    grader_system_prompt: str,
    tools: Sequence[BaseTool | Callable[..., Any]],
    grader_tools: Sequence[BaseTool],
    max_iterations: int = 3,
    on_evaluation: Callable[[RubricEvaluation], None] | None = None,
    prepare_messages_for_grader: Callable[[list[AnyMessage]], list[AnyMessage]] | None = None,
    name: str | None = None,
    checkpointer: MemorySaver | None = None,
) -> tuple[Any, MemorySaver]:
    """Construct a real `create_deep_agent` equipped with `RubricMiddleware` & `MemorySaver`."""
    if not main_model:
        raise ValueError("main_model must be explicitly provided (never None)")
    if not grader_model:
        raise ValueError("grader_model must be explicitly provided (never None)")

    saver = checkpointer or MemorySaver()
    middleware = RubricMiddleware(
        model=grader_model,
        system_prompt=grader_system_prompt,
        tools=list(grader_tools),
        max_iterations=max_iterations,
        on_evaluation=on_evaluation,
        prepare_messages_for_grader=prepare_messages_for_grader,
    )
    agent = create_deep_agent(
        model=main_model,
        tools=list(tools),
        system_prompt=system_prompt,
        middleware=[middleware],
        checkpointer=saver,
        name=name,
    )
    return agent, saver


async def run_rubric_deep_agent(
    *,
    stage: str,
    run_id: str,
    prompt: str,
    rubric: str,
    main_model: BaseChatModel | str,
    grader_model: BaseChatModel | str,
    system_prompt: str,
    grader_system_prompt: str,
    tools: Sequence[BaseTool | Callable[..., Any]],
    grader_tools: Sequence[BaseTool],
    max_iterations: int = 3,
    event_cb: EventCallback | None = None,
    cancel: CancellationToken | None = None,
    prepare_messages_for_grader: Callable[[list[AnyMessage]], list[AnyMessage]] | None = None,
    raise_on_failure: bool = True,
) -> RubricRunOutcome:
    """Execute a rubric-gated DeepAgent asynchronously and capture authoritative terminal status.

    Captures verdicts via BOTH `on_evaluation` AND checkpointer private state (`_rubric_status`)
    (Survey 1 Edge Cases #1-#2), forwarding every `RubricEvaluation` to `event_cb`.
    """
    if cancel is not None:
        cancel.raise_if_cancelled()

    evaluations: list[dict[str, Any]] = []
    pending_event_tasks: list[asyncio.Task[Any]] = []
    loop = asyncio.get_running_loop()

    def _on_evaluation(ev: RubricEvaluation) -> None:
        ev_dict = dict(ev)
        evaluations.append(ev_dict)
        if event_cb is not None:
            payload = make_sse_event(
                event_type="rubric_evaluation",
                run_id=run_id,
                stage=stage,
                data=ev_dict,
            )
            res = event_cb(payload)
            if inspect.isawaitable(res):
                pending_event_tasks.append(loop.create_task(res))

    agent, _ = build_rubric_deep_agent(
        main_model=main_model,
        grader_model=grader_model,
        system_prompt=system_prompt,
        grader_system_prompt=grader_system_prompt,
        tools=tools,
        grader_tools=grader_tools,
        max_iterations=max_iterations,
        on_evaluation=_on_evaluation,
        prepare_messages_for_grader=prepare_messages_for_grader,
        name=stage,
    )

    thread_id = f"{run_id}_{stage}_{uuid4().hex[:8]}"
    config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}

    invoke_task = asyncio.create_task(
        agent.ainvoke(
            {
                "messages": [HumanMessage(content=prompt)],
                "rubric": rubric,
            },
            config=config,
        )
    )
    if cancel is not None:
        cancel.track_task(invoke_task)

    try:
        result = await invoke_task
    finally:
        if not invoke_task.done():
            invoke_task.cancel()
            await asyncio.gather(invoke_task, return_exceptions=True)
        if pending_event_tasks:
            await asyncio.gather(*pending_event_tasks, return_exceptions=True)

    if cancel is not None:
        cancel.raise_if_cancelled()

    # Authoritative terminal status from checkpointer private state + on_evaluation fallback
    checkpointer_status: str | None = None
    try:
        snapshot = agent.get_state(config)
        values: Mapping[str, Any] = getattr(snapshot, "values", None) or {}
        raw_status = values.get("_rubric_status")
        if isinstance(raw_status, str) and raw_status:
            checkpointer_status = raw_status
    except Exception:
        checkpointer_status = None

    callback_status = str(evaluations[-1].get("result", "")) if evaluations else ""
    terminal_status = checkpointer_status or callback_status or "unknown"

    final_message = ""
    msgs = result.get("messages") if isinstance(result, Mapping) else None
    if isinstance(msgs, list):
        for msg in reversed(msgs):
            if isinstance(msg, AIMessage):
                final_message = (
                    msg.content if isinstance(msg.content, str) else json.dumps(msg.content)
                )
                break

    outcome = RubricRunOutcome(
        stage=stage,
        status=terminal_status,
        passed=(terminal_status == "satisfied"),
        evaluations=evaluations,
        final_message=final_message,
        thread_id=thread_id,
    )

    if raise_on_failure and not outcome.passed:
        raise RubricGateError(
            stage=stage,
            status=terminal_status,
            evaluations=evaluations,
        )

    return outcome


__all__ = [
    "LIVE_MODE_ERROR_MSG",
    "EventCallback",
    "RubricGateError",
    "RubricRunOutcome",
    "build_rubric_deep_agent",
    "emit_event",
    "ensure_live_llm_configured",
    "is_default_unscripted_mock_model",
    "is_offline_provider_bundle",
    "make_sse_event",
    "make_tool_driven_grader_model",
    "read_workspace_json",
    "resolve_live_chat_model",
    "run_rubric_deep_agent",
    "write_workspace_json",
]
