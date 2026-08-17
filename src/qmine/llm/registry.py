"""Model routing, response caching, and structured output with repair.

**Two tiers.**  Borrowed from TradingAgents' deep/quick split: the architect,
the referee, and the tree auditor get the strong model because their mistakes
propagate into every downstream artifact; the six hundred annotation calls and
the sixty naming calls get the fast one.  A single ``role -> tier`` table keeps
the routing legible instead of scattering model names through the codebase.

**Caching is a correctness feature, not just a cost one.**  Keying responses by
``sha256(model, prompt, schema, temperature)`` means a re-run after a crash
replays the identical judgments rather than sampling fresh ones — which is the
only way a "re-runnable" pipeline with LLM steps can also be a *reproducible*
one.

**Three providers, one interface.**  ``anthropic`` for real runs, ``offline``
for CI and air-gapped machines, and ``auto``, which picks anthropic when a key
is present and offline otherwise, announcing which it chose.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Literal, TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from ..config import LLMConfig
from ..determinism import hash_params
from .budget import UsageLedger
from .offline import OfflineHeuristicModel, synthesize

log = logging.getLogger("qmine.llm")

T = TypeVar("T", bound=BaseModel)

Tier = Literal["deep", "fast"]

#: Which role gets which tier.  Roles absent from the table default to "fast".
ROLE_TIER: dict[str, Tier] = {
    # deep — decisions that everything downstream inherits
    "taxonomy_architect": "deep",
    "taxonomy_critic": "deep",
    "referee": "deep",
    "tree_auditor": "deep",
    "risk_sentinel": "deep",
    "reporter": "deep",
    "maintainer": "deep",
    # fast — high volume, narrow judgment
    "researcher": "fast",
    "annotator_a": "fast",
    "annotator_b": "fast",
    "adversary": "fast",
    "namer": "fast",
    "l2_interpreter": "fast",
}


class LLMUnavailable(RuntimeError):
    """Raised when a real provider was demanded but cannot be reached."""


class ModelRegistry:
    """Hands out models by role, accounts for usage, and caches responses."""

    def __init__(
        self,
        cfg: LLMConfig,
        *,
        cache_dir: str | os.PathLike[str] | None = None,
        ledger: UsageLedger | None = None,
    ) -> None:
        self.cfg = cfg
        self.ledger = ledger or UsageLedger(
            max_calls=cfg.max_total_calls, max_output_tokens=cfg.max_total_output_tokens
        )
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.provider = self._resolve_provider(cfg.provider)
        self._models: dict[str, BaseChatModel] = {}
        self.raw_log: list[dict[str, Any]] = []

    # -- provider selection -------------------------------------------------
    @staticmethod
    def _has_anthropic_credentials() -> bool:
        return bool(
            os.environ.get("ANTHROPIC_API_KEY")
            or os.environ.get("ANTHROPIC_AUTH_TOKEN")
            or os.environ.get("LANGSMITH_API_KEY")
        )

    def _resolve_provider(self, requested: str) -> str:
        if requested == "auto":
            chosen = "anthropic" if self._has_anthropic_credentials() else "offline"
            log.info(
                "llm provider auto-resolved to %r (credentials %s)",
                chosen,
                "found" if chosen == "anthropic" else "absent",
            )
            return chosen
        if requested == "mock":
            return "offline"
        return requested

    @property
    def is_offline(self) -> bool:
        return self.provider == "offline"

    def model_name(self, tier: Tier) -> str:
        if self.is_offline:
            return "offline-heuristic"
        return self.cfg.deep_model if tier == "deep" else self.cfg.fast_model

    def get(self, role: str) -> BaseChatModel:
        tier: Tier = ROLE_TIER.get(role, "fast")
        key = f"{self.provider}:{tier}"
        if key in self._models:
            return self._models[key]
        if self.is_offline:
            model: BaseChatModel = OfflineHeuristicModel()
        elif self.provider == "anthropic":
            from langchain_anthropic import ChatAnthropic

            name = self.model_name(tier)
            kwargs: dict[str, Any] = {
                "model": name,
                "max_tokens": self.cfg.max_tokens,
                "max_retries": self.cfg.max_retries,
            }
            if self.cfg.temperature is not None and _accepts_temperature(name):
                kwargs["temperature"] = self.cfg.temperature
            model = ChatAnthropic(**kwargs)
        else:  # pragma: no cover - other providers are opt-in
            from langchain.chat_models import init_chat_model

            kwargs = {"max_tokens": self.cfg.max_tokens}
            if self.cfg.temperature is not None:
                kwargs["temperature"] = self.cfg.temperature
            model = init_chat_model(f"{self.provider}:{self.model_name(tier)}", **kwargs)
        self._models[key] = model
        return model

    # -- caching ------------------------------------------------------------
    def _cache_path(self, key: str) -> Path | None:
        return None if not self.cache_dir else self.cache_dir / f"{key}.json"

    def _cache_get(self, key: str) -> Any | None:
        p = self._cache_path(key)
        if p and p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))["value"]
            except Exception:
                p.unlink(missing_ok=True)
        return None

    def _cache_put(self, key: str, value: Any, meta: dict[str, Any]) -> None:
        p = self._cache_path(key)
        if p:
            p.write_text(
                json.dumps({"value": value, "meta": meta}, ensure_ascii=False, default=str),
                encoding="utf-8",
            )

    # -- the call ------------------------------------------------------------
    def complete(
        self,
        role: str,
        system: str,
        user: str,
        *,
        schema: type[T] | None = None,
        max_repair: int = 2,
        temperature: float | None = None,
    ) -> Any:
        """One agent turn.  Returns a validated ``schema`` instance, or raw text.

        Structured output is attempted through the provider's native mechanism
        first.  If the result fails validation, we do not silently accept a
        half-parsed dict: we re-prompt with the validation error appended, which
        empirically fixes the overwhelming majority of schema misses, and only
        then give up loudly.
        """
        self.ledger.check(role)
        tier: Tier = ROLE_TIER.get(role, "fast")
        cache_key = hash_params(
            {
                # Role is part of the key on purpose. The cache exists so that the
                # SAME agent replays its own judgment on a re-run — not so that two
                # agents who happen to share a prompt collide. Two annotators
                # sharing a cache entry would agree perfectly by construction and
                # the kappa downstream would measure nothing.
                "role": role,
                "provider": self.provider,
                "model": self.model_name(tier),
                "system": system,
                "user": user,
                "schema": schema.__name__ if schema else None,
                "schema_fields": sorted(schema.model_fields) if schema else None,
                "temperature": temperature if temperature is not None else self.cfg.temperature,
            },
            length=32,
        )
        if self.cfg.cache_llm_calls:
            hit = self._cache_get(cache_key)
            if hit is not None:
                self.ledger.record(role, cached=True)
                return schema.model_validate(hit) if schema else hit

        t0 = time.time()
        if self.is_offline:
            # Seed by role as well as prompt: two annotators are two agents, and
            # if the stand-in returns identical output for both, the agreement
            # statistic downstream measures nothing at all.
            payload = synthesize(user, schema, system=f"{system}\n<role:{role}>")
            value = schema.model_validate(payload) if schema else str(payload)
            self.ledger.record(role, output_tokens=0)
            self._store(role, cache_key, value, system, user, tier, t0)
            return value

        model = self.get(role)
        messages = [SystemMessage(content=system), HumanMessage(content=user)]
        last_err = ""
        for attempt in range(max_repair + 1):
            try:
                if schema is not None:
                    runnable = model.with_structured_output(schema, include_raw=True)
                    out = runnable.invoke(messages)
                    raw: AIMessage | None = out.get("raw") if isinstance(out, dict) else None
                    _check_refusal(raw)
                    parsed = out.get("parsed") if isinstance(out, dict) else out
                    if parsed is None:
                        err = out.get("parsing_error") if isinstance(out, dict) else None
                        raise ValueError(f"structured output returned no parsed value ({err})")
                    self._account(role, raw)
                    self._store(role, cache_key, parsed, system, user, tier, t0)
                    return parsed
                resp = model.invoke(messages)
                _check_refusal(resp)
                self._account(role, resp)
                text = resp.content if isinstance(resp.content, str) else str(resp.content)
                self._store(role, cache_key, text, system, user, tier, t0)
                return text
            except Exception as exc:  # noqa: BLE001 — we deliberately repair on anything
                last_err = f"{type(exc).__name__}: {exc}"
                self.ledger.record(role, error=True)
                if attempt >= max_repair:
                    break
                log.warning("role=%s attempt=%d failed (%s); repairing", role, attempt, last_err[:160])
                messages = messages[:2] + [
                    HumanMessage(
                        content=(
                            "Your previous response could not be parsed into the required schema.\n"
                            f"Error: {last_err}\n"
                            "Return ONLY a valid instance of the schema. No prose, no markdown fence."
                        )
                    )
                ]
        raise LLMUnavailable(f"role={role} failed after {max_repair + 1} attempts: {last_err}")

    def _account(self, role: str, msg: Any) -> None:
        usage = getattr(msg, "usage_metadata", None) or {}
        self.ledger.record(
            role,
            input_tokens=int(usage.get("input_tokens", 0) or 0),
            output_tokens=int(usage.get("output_tokens", 0) or 0),
        )

    def _store(
        self, role: str, key: str, value: Any, system: str, user: str, tier: Tier, t0: float
    ) -> None:
        payload = value.model_dump() if isinstance(value, BaseModel) else value
        if self.cfg.cache_llm_calls:
            self._cache_put(
                key,
                payload,
                {"role": role, "tier": tier, "model": self.model_name(tier), "ts": time.time()},
            )
        if self.cfg.record_raw_outputs:
            self.raw_log.append(
                {
                    "role": role,
                    "tier": tier,
                    "model": self.model_name(tier),
                    "cache_key": key,
                    "latency_s": round(time.time() - t0, 2),
                    "system_head": system[:200],
                    "user_head": user[:400],
                    "output": payload,
                }
            )

    # -- reporting ----------------------------------------------------------
    def usage(self) -> dict[str, Any]:
        u = self.ledger.snapshot()
        u["provider"] = self.provider
        u["deep_model"] = self.model_name("deep")
        u["fast_model"] = self.model_name("fast")
        u["estimated_cost_usd"] = round(self.ledger.estimated_cost_usd(), 2)
        return u

    def provenance_note(self) -> str:
        """The sentence every report must carry about who actually judged things."""
        if self.is_offline:
            return (
                "LLM-judgment steps in this run were produced by the deterministic "
                "offline heuristic stand-in, NOT by a language model. Names, definitions "
                "and labels from those steps are computed from n-gram and regex evidence "
                "and are marked `offline-heuristic`. Quantitative results (embeddings, "
                "clustering, metrics) are unaffected and fully real."
            )
        return (
            f"LLM-judgment steps used {self.model_name('deep')} (deep tier) and "
            f"{self.model_name('fast')} (fast tier) at temperature {self.cfg.temperature}, "
            f"with responses cached by content hash for reproducibility."
        )


#: Models that removed the sampling parameters entirely and return 400 on any value.
_NO_TEMPERATURE = ("claude-opus-5", "claude-opus-4-8", "claude-opus-4-7", "claude-fable-5", "claude-mythos-5")


def _accepts_temperature(model: str) -> bool:
    return not any(model.startswith(m) for m in _NO_TEMPERATURE)


class ModelRefused(RuntimeError):
    """The model declined the request.  Distinct from a failure to parse."""


def _check_refusal(msg: Any) -> None:
    """Detect a refusal, which arrives as HTTP 200 with ``stop_reason='refusal'``.

    Without this check a refusal looks like an empty or malformed answer and
    burns the repair budget re-asking a question the model already declined.
    Branch on ``stop_reason`` — ``stop_details`` is null for every other reason.
    """
    if msg is None:
        return
    meta = getattr(msg, "response_metadata", None) or {}
    if meta.get("stop_reason") == "refusal":
        raise ModelRefused(
            f"model refused this request (categories: {meta.get('stop_details')}). "
            "Refusals are not parse failures — retrying the same prompt will refuse again."
        )


def strip_fence(text: str) -> str:
    """Remove a ```json fence if a model wrapped its JSON in one."""
    m = re.match(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", text, re.S)
    return m.group(1) if m else text
