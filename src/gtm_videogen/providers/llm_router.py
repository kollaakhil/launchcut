"""Stage-aware LLM router, provider capability matrix, and tiered cost metering.

Implements Pillar 1 & fixes BUG-27:
- Explicit stage profiles across all 11 pipeline stages.
- `sound_design` explicitly uses `text_model`, NEVER `vision_model` (BUG-27).
- Provider capability-filtering matrix prevents HTTP 400s on reasoning models (o1/o3/o4),
  Anthropic models (no penalties, single temperature-or-top_p), and json_schema fallback.
- Tiered LLM cost estimation (`fast` vs `flagship`) integrated with CostMeter.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

from gtm_videogen.core.config import Settings, get_settings
from gtm_videogen.core.cost_meter import CostMeter, current_cost_meter
from gtm_videogen.models.schemas import ChatOptions, CostEntry

LLMTier = Literal["fast", "flagship"]
ModelSlot = Literal["text_model", "vision_model"]


@dataclass(frozen=True)
class StageProfile:
    """Sampling and model-routing defaults for a single pipeline stage."""

    stage: str
    tier: LLMTier
    model_slot: ModelSlot
    temperature: float
    top_p: float | None = 0.95
    max_tokens: int = 4096
    presence_penalty: float | None = 0.0
    frequency_penalty: float | None = 0.0


STAGE_PROFILES: dict[str, StageProfile] = {
    "research": StageProfile(
        stage="research",
        tier="fast",
        model_slot="text_model",
        temperature=0.15,
        top_p=0.9,
        max_tokens=2048,
        presence_penalty=0.0,
        frequency_penalty=0.0,
    ),
    "claims_check": StageProfile(
        stage="claims_check",
        tier="fast",
        model_slot="text_model",
        temperature=0.1,
        top_p=0.9,
        max_tokens=2048,
        presence_penalty=0.0,
        frequency_penalty=0.0,
    ),
    "qa": StageProfile(
        stage="qa",
        tier="fast",
        model_slot="vision_model",
        temperature=0.15,
        top_p=0.9,
        max_tokens=2048,
        presence_penalty=0.0,
        frequency_penalty=0.0,
    ),
    "shot_check": StageProfile(
        stage="shot_check",
        tier="fast",
        model_slot="vision_model",
        temperature=0.15,
        top_p=0.9,
        max_tokens=2048,
        presence_penalty=0.0,
        frequency_penalty=0.0,
    ),
    "creative_bible": StageProfile(
        stage="creative_bible",
        tier="flagship",
        model_slot="text_model",
        temperature=0.85,
        top_p=0.95,
        max_tokens=8192,
        presence_penalty=0.3,
        frequency_penalty=0.0,
    ),
    "concept_divergence": StageProfile(
        stage="concept_divergence",
        tier="flagship",
        model_slot="text_model",
        temperature=1.0,
        top_p=0.95,
        max_tokens=4096,
        presence_penalty=0.5,
        frequency_penalty=0.3,
    ),
    "draft_hooks": StageProfile(
        stage="draft_hooks",
        tier="flagship",
        model_slot="text_model",
        temperature=0.95,
        top_p=0.95,
        max_tokens=2048,
        presence_penalty=0.5,
        frequency_penalty=0.3,
    ),
    "script_write": StageProfile(
        stage="script_write",
        tier="flagship",
        model_slot="text_model",
        temperature=0.75,
        top_p=0.95,
        max_tokens=12000,
        presence_penalty=0.2,
        frequency_penalty=0.0,
    ),
    "sheet_compile": StageProfile(
        stage="sheet_compile",
        tier="fast",
        model_slot="text_model",
        temperature=0.25,
        top_p=0.9,
        max_tokens=12000,
        presence_penalty=0.0,
        frequency_penalty=0.0,
    ),
    "repair": StageProfile(
        stage="repair",
        tier="fast",
        model_slot="text_model",
        temperature=0.2,
        top_p=0.9,
        max_tokens=4096,
        presence_penalty=0.0,
        frequency_penalty=0.0,
    ),
    # BUG-27: sound_design MUST use text_model, never vision_model.
    "sound_design": StageProfile(
        stage="sound_design",
        tier="flagship",
        model_slot="text_model",
        temperature=0.7,
        top_p=0.95,
        max_tokens=3072,
        presence_penalty=0.2,
        frequency_penalty=0.0,
    ),
}


@dataclass(frozen=True)
class TierPricing:
    """Per-tier USD pricing for preflight estimation and token-based settlement."""

    default_call_usd: float
    input_usd_per_1m: float
    output_usd_per_1m: float


LLM_TIER_PRICING: dict[LLMTier, TierPricing] = {
    "fast": TierPricing(
        default_call_usd=0.02,
        input_usd_per_1m=0.30,
        output_usd_per_1m=1.20,
    ),
    "flagship": TierPricing(
        default_call_usd=0.10,
        input_usd_per_1m=3.00,
        output_usd_per_1m=15.00,
    ),
}

_REASONING_RE = re.compile(
    r"^(?:openai/)?(?:o1|o3|o4)(?:[-._:/].*)?$|"
    r"^(?:deepseek/)?deepseek-(?:r1|reasoner)(?:[-._:/].*)?$|"
    r"^(?:qwen/)?qwq(?:[-._:/].*)?$",
    re.IGNORECASE,
)

_ANTHROPIC_RE = re.compile(r"^(?:anthropic/)?claude[-._]", re.IGNORECASE)

_STRUCTURED_OUTPUT_RE = re.compile(
    r"^(?:openai/)?(?:gpt-4o|gpt-4\.1|gpt-5|o1|o3|o4)(?:[-._:/].*)?$|"
    r"^(?:google/)?gemini-(?:1\.5|2|3)(?:[-._:/].*)?$|"
    r"^(?:anthropic/)?claude-(?:3[.-]7|4|sonnet-4|opus-4|haiku-4)(?:[-._:/].*)?$",
    re.IGNORECASE,
)


def _normalize_model_id(model: str) -> str:
    return model.strip().lower()


def is_reasoning_model(model: str) -> bool:
    """Return True if model is a reasoning model that rejects sampling penalties/temperature."""
    norm = _normalize_model_id(model)
    if _REASONING_RE.match(norm):
        return True
    suffix = norm.rsplit("/", 1)[-1]
    return bool(_REASONING_RE.match(suffix))


def is_anthropic_model(model: str) -> bool:
    """Return True if model belongs to the Anthropic Claude family."""
    norm = _normalize_model_id(model)
    if _ANTHROPIC_RE.match(norm):
        return True
    suffix = norm.rsplit("/", 1)[-1]
    return bool(_ANTHROPIC_RE.match(suffix))


def supports_json_schema(model: str) -> bool:
    """Return True if model natively supports response_format type='json_schema'."""
    norm = _normalize_model_id(model)
    if _STRUCTURED_OUTPUT_RE.match(norm):
        return True
    suffix = norm.rsplit("/", 1)[-1]
    return bool(_STRUCTURED_OUTPUT_RE.match(suffix))


_CAMEL_BOUNDARY_RE = re.compile(r"([a-z0-9])([A-Z])")


def _normalize_stage_name(stage: str) -> str:
    """Normalize snake_case or camelCase stage names (e.g. soundDesign -> sound_design)."""
    cleaned = stage.strip().replace("-", "_")
    snake = _CAMEL_BOUNDARY_RE.sub(r"\1_\2", cleaned).lower()
    if snake == "write_sheet":
        return "script_write"
    return snake


def resolve_tier(stage_or_tier: str) -> LLMTier:
    """Resolve a stage name or explicit tier string ('fast'/'flagship') to an LLMTier."""
    if stage_or_tier in ("fast", "flagship"):
        return stage_or_tier  # type: ignore[return-value]
    norm = _normalize_stage_name(stage_or_tier)
    if norm in STAGE_PROFILES:
        return STAGE_PROFILES[norm].tier
    raise ValueError(f"Unknown LLM stage or tier: {stage_or_tier!r}")


def estimate_llm_cost(
    stage_or_tier: str,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
) -> float:
    """Estimate USD cost for an LLM call by stage or tier ('fast' vs 'flagship')."""
    tier = resolve_tier(stage_or_tier)
    pricing = LLM_TIER_PRICING[tier]
    if prompt_tokens is None and completion_tokens is None:
        return pricing.default_call_usd
    in_tok = max(0, int(prompt_tokens or 0))
    out_tok = max(0, int(completion_tokens or 0))
    cost = (in_tok / 1_000_000.0) * pricing.input_usd_per_1m + (
        out_tok / 1_000_000.0
    ) * pricing.output_usd_per_1m
    return round(cost, 6)


def record_llm_cost(
    cost_meter: CostMeter | None,
    *,
    stage: str,
    model: str,
    provider: str = "openrouter",
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    label: str | None = None,
) -> CostEntry | None:
    """Reserve and settle an LLM call against the provided or ContextVar-active CostMeter."""
    meter = cost_meter or current_cost_meter()
    if meter is None:
        return None
    est_usd = estimate_llm_cost(stage)
    actual_usd = estimate_llm_cost(
        stage,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
    res_id = meter.reserve(
        label or f"llm:{stage}",
        est_usd,
        stage=stage,
        provider=provider,
        model=model,
    )
    return meter.settle(res_id, actual_usd)


class LLMRouter:
    """Stage-aware model resolver and capability-filtered chat options builder."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def get_profile(self, stage: str) -> StageProfile:
        """Return the StageProfile for a known pipeline stage (snake_case or camelCase)."""
        norm = _normalize_stage_name(stage)
        if norm not in STAGE_PROFILES:
            raise ValueError(
                f"Unknown LLM stage {stage!r}; expected one of {sorted(STAGE_PROFILES)}"
            )
        return STAGE_PROFILES[norm]

    def resolve_model(self, stage: str, model_override: str | None = None) -> str:
        """Resolve the configured model ID for a stage (BUG-27: sound_design uses text_model)."""
        if model_override and model_override.strip():
            return model_override.strip()

        profile = self.get_profile(stage)
        if profile.model_slot == "vision_model":
            return self.settings.vision_model
        if profile.stage == "sound_design":
            return self.settings.text_model
        if profile.tier == "fast":
            return self.settings.fast_model
        return self.settings.flagship_model or self.settings.text_model

    def build_chat_options(
        self,
        model: str,
        stage: str,
        overrides: ChatOptions | None = None,
    ) -> dict[str, Any]:
        """Build provider-safe chat completion options using the capability-filtering matrix."""
        profile = self.get_profile(stage)

        temperature: float | None = profile.temperature
        top_p: float | None = profile.top_p
        max_tokens: int | None = profile.max_tokens
        presence_penalty: float | None = profile.presence_penalty
        frequency_penalty: float | None = profile.frequency_penalty
        reasoning_effort: str | None = None
        json_schema: dict[str, Any] | None = None

        if overrides is not None:
            if overrides.temperature is not None:
                temperature = overrides.temperature
            if overrides.top_p is not None:
                top_p = overrides.top_p
            if overrides.max_tokens is not None:
                max_tokens = overrides.max_tokens
            if overrides.presence_penalty is not None:
                presence_penalty = overrides.presence_penalty
            if overrides.frequency_penalty is not None:
                frequency_penalty = overrides.frequency_penalty
            if overrides.reasoning_effort is not None:
                reasoning_effort = overrides.reasoning_effort
            if overrides.json_schema is not None:
                json_schema = overrides.json_schema

        payload: dict[str, Any] = {}
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        reasoning_call = is_reasoning_model(model) or reasoning_effort is not None
        if reasoning_call:
            # Reasoning models reject temperature, top_p, presence_penalty, frequency_penalty.
            if reasoning_effort is not None:
                payload["reasoning_effort"] = reasoning_effort
        elif is_anthropic_model(model):
            # Anthropic rejects presence/frequency penalties and rejects sending BOTH
            # temperature and top_p simultaneously (prefer temperature; only send top_p if None).
            if temperature is not None:
                payload["temperature"] = temperature
            elif top_p is not None:
                payload["top_p"] = top_p
        else:
            if temperature is not None:
                payload["temperature"] = temperature
            if top_p is not None:
                payload["top_p"] = top_p
            if presence_penalty is not None:
                payload["presence_penalty"] = presence_penalty
            if frequency_penalty is not None:
                payload["frequency_penalty"] = frequency_penalty

        if json_schema is not None:
            if supports_json_schema(model):
                if "schema" in json_schema and "name" in json_schema:
                    schema_spec = json_schema
                else:
                    schema_spec = {
                        "name": f"{stage}_response",
                        "strict": True,
                        "schema": json_schema,
                    }
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": schema_spec,
                }
            else:
                payload["response_format"] = {"type": "json_object"}

        return {k: v for k, v in payload.items() if v is not None}

    def estimate_cost(
        self,
        stage: str,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
    ) -> float:
        """Estimate USD cost for a pipeline stage."""
        return estimate_llm_cost(
            stage,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    def record_cost(
        self,
        cost_meter: CostMeter | None,
        *,
        stage: str,
        model: str | None = None,
        provider: str = "openrouter",
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        label: str | None = None,
    ) -> CostEntry | None:
        """Record an LLM call against a CostMeter."""
        resolved_model = model or self.resolve_model(stage)
        return record_llm_cost(
            cost_meter,
            stage=stage,
            model=resolved_model,
            provider=provider,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            label=label,
        )
