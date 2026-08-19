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
    "domain_scout": "deep",
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
        self.plan: Any = None          # RoutingPlan, when routing is enabled
        #: Models whose native structured-output mode has already failed once in
        #: this run. Retrying it per call is expensive: at 240 annotation calls,
        #: a wasted first attempt of 30-180s each roughly doubles the run.
        self._no_native_schema: set[str] = set()
        self._routed: dict[str, tuple[str, str]] = {}   # role -> (provider, model)
        self._model_output_cap: dict[tuple[str, str], int] = {}
        if cfg.provider == "router":
            self._build_routing_plan(cache_dir)

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

    def _build_routing_plan(self, cache_dir: Any) -> None:
        """Resolve a model per role from the live catalogue and available keys.

        Degrades to the static two-tier behaviour whenever anything is missing —
        no keys, no catalogue, no network. A router that fails closed would make
        the pipeline unrunnable offline, which is a worse outcome than routing
        naively.
        """
        from .catalog import fetch
        from .providers import detect
        from .router import route

        av = detect()
        if not av.usable:
            log.warning("provider=router but no API keys are configured; using offline stand-in")
            self.provider = "offline"
            return
        try:
            cat = fetch(
                cache_dir=cache_dir, ttl=self.cfg.catalog_ttl_hours * 3600,
                allow_network=not self.cfg.catalog_offline, pinned=self.cfg.catalog_pinned,
            )
            self.plan = route(
                cat, av.usable, prefer=self.cfg.model_overrides or None,
                budget_usd=self.cfg.budget_usd,
                prefer_chinese_native=self.cfg.prefer_chinese_native,
            )
            self._routed = {
                r: (a.provider, a.api_model or a.model)
                for r, a in self.plan.assignments.items() if a.model
            }
            self._model_output_cap = {
                (a.provider, a.api_model or a.model): a.max_output_tokens
                for a in self.plan.assignments.values()
                if a.model and a.max_output_tokens
            }
            self.provider = "routed"
            log.info("routing plan: %d roles, estimated $%.2f, catalogue %s",
                     len(self._routed), self.plan.total_cost_usd, cat.sources)
        except Exception as exc:  # noqa: BLE001
            log.warning("routing unavailable (%s); falling back to the static tiers", exc)
            self.provider = self._resolve_provider("auto")

    def route_for(self, role: str) -> tuple[str, str] | None:
        """The (provider, model) chosen for a role, if routing is active.

        Roles arrive suffixed — ``researcher_log_reading``, ``namer_3``,
        ``annotator_a`` — because the same role runs as several distinct agents.
        The routing plan is keyed on the base role, so an exact-match-only lookup
        silently misses every suffixed agent and drops them onto the static
        fallback path. That is how a live run ended up asking for a model called
        ``routed:claude-sonnet-5``.
        """
        if role in self._routed:
            return self._routed[role]
        # longest registered prefix wins, so `annotator_a` does not match `annotator`
        # when `annotator_a` is itself registered.
        for base in sorted(self._routed, key=len, reverse=True):
            if role.startswith(base):
                return self._routed[base]
        return None

    def model_name(self, tier: Tier, role: str = "") -> str:
        if self.is_offline:
            return "offline-heuristic"
        routed = self.route_for(role) if role else None
        if routed:
            return routed[1]
        return self.cfg.deep_model if tier == "deep" else self.cfg.fast_model

    def get(self, role: str) -> BaseChatModel:
        routed = self.route_for(role)
        if routed:
            return self._build_routed(role, *routed)
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
                "timeout": self.cfg.request_timeout,
            }
            if self.cfg.temperature is not None and _accepts_temperature(name):
                kwargs["temperature"] = self.cfg.temperature
            model = ChatAnthropic(**kwargs)
        elif self.provider == "routed":
            # Routing is on but this role found no assignment. Fail loudly here
            # rather than constructing a nonsense model name that surfaces as an
            # opaque provider-inference error four frames deeper.
            raise LLMUnavailable(
                f"routing is enabled but role {role!r} has no assigned model. "
                f"Assigned roles: {sorted(self._routed)}. This is a routing bug, not a "
                "configuration problem — every role must resolve."
            )
        else:  # pragma: no cover - other providers are opt-in
            from langchain.chat_models import init_chat_model

            kwargs = {"max_tokens": self.cfg.max_tokens}
            if self.cfg.temperature is not None:
                kwargs["temperature"] = self.cfg.temperature
            model = init_chat_model(f"{self.provider}:{self.model_name(tier)}", **kwargs)
        self._models[key] = model
        return model

    def _build_routed(self, role: str, provider: str, model_id: str) -> BaseChatModel:
        """Instantiate a routed model, reusing one instance per (provider, model, timeout)."""
        from langchain.chat_models import init_chat_model

        from .providers import BY_KEY

        from .requirements import requirement_for

        req = requirement_for(role)
        timeout = max(self.cfg.request_timeout, req.timeout_seconds)
        # The role's declared budget is authoritative — `cfg.max_tokens` is the
        # default for roles that declare nothing, not a ceiling over those that
        # do. Clamping to it truncated the taxonomy architect at 16k while its
        # own requirement asked for 24k.
        cap = req.max_output_tokens
        published = self._model_output_cap.get((provider, model_id))
        if published:
            cap = min(cap, int(published))
        key = f"routed:{provider}:{model_id}:{int(timeout)}:{cap}"
        if key in self._models:
            return self._models[key]

        spec = BY_KEY.get(provider)
        kwargs: dict[str, Any] = {
            "max_tokens": cap,
            "timeout": timeout,
            "max_retries": self.cfg.max_retries,
        }
        if self.cfg.temperature is not None and _accepts_temperature(model_id):
            kwargs["temperature"] = self.cfg.temperature

        # Prefer the OpenAI-compatible path over a provider-specific LangChain
        # integration. Both work, but the dedicated integrations each need their
        # own package installed, and "pip install langchain-<vendor>" per provider
        # does not scale to eighteen of them — a missing one fails at call time,
        # deep into a run, rather than at planning time.
        if spec and spec.openai_base_url:
            # Everything else speaks the OpenAI wire format; that is the whole
            # reason a single adapter reaches a dozen labs.
            import os

            from langchain_openai import ChatOpenAI

            from .providers import resolve_base_url

            token = next((os.environ[v] for v in spec.env_vars if os.environ.get(v)), "")
            model = ChatOpenAI(model=model_id, base_url=resolve_base_url(provider),
                               api_key=token, **kwargs)
        elif spec and spec.langchain_prefix:
            model = init_chat_model(f"{spec.langchain_prefix}:{model_id}", **kwargs)
        else:
            model = init_chat_model(model_id, **kwargs)
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
                "model": self.model_name(tier, role),
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
                if schema is None:
                    resp = model.invoke(messages)
                    _check_refusal(resp)
                    self._account(role, resp)
                    text = resp.content if isinstance(resp.content, str) else str(resp.content)
                    self._store(role, cache_key, text, system, user, tier, t0)
                    return text

                model_key = f"{self.model_name(tier, role)}"
                if attempt == 0 and model_key not in self._no_native_schema:
                    # First try the provider's native structured output.
                    parsed, raw = self._structured_call(model, messages, schema)
                else:
                    # Repair path: plain text with the schema stated, parsed here.
                    # Provider-agnostic on purpose — the native mode on several
                    # OpenAI-compatible endpoints returns valid JSON wrapped in a
                    # markdown fence and then RAISES on it, so re-asking the same
                    # way reproduces the same failure. Parsing it ourselves does not.
                    parsed, raw = self._plain_json_call(model, messages, schema, last_err)

                if parsed is None:
                    raise ValueError("no parseable structured output")
                self._account(role, raw)
                self._store(role, cache_key, parsed, system, user, tier, t0)
                return parsed

            except ModelRefused:
                raise
            except Exception as exc:  # noqa: BLE001 — we deliberately repair on anything
                last_err = f"{type(exc).__name__}: {exc}"
                self.ledger.record(role, error=True)
                if attempt >= max_repair:
                    break
                if attempt == 0 and _native_schema_is_broken(last_err):
                    # Native structured output is unreliable on this model. Note it
                    # once and go straight to plain-JSON mode for the rest of the
                    # run rather than paying for the discovery on every call.
                    key = self.model_name(tier, role)
                    if key not in self._no_native_schema:
                        self._no_native_schema.add(key)
                        log.warning(
                            "%s returns unparseable structured output; switching to "
                            "plain-JSON mode for the remainder of this run", key
                        )
                log.warning("role=%s attempt=%d failed (%s); repairing via plain-JSON mode",
                            role, attempt, last_err[:160])

        raise LLMUnavailable(f"role={role} failed after {max_repair + 1} attempts: {last_err}")

    def _structured_call(self, model: Any, messages: list[Any], schema: type[T]) -> tuple[Any, Any]:
        """The provider's native structured output, with salvage on a soft miss."""
        try:
            runnable = model.with_structured_output(schema, method="json_schema", include_raw=True)
        except Exception:
            runnable = model.with_structured_output(schema, include_raw=True)
        out = runnable.invoke(messages)
        raw = out.get("raw") if isinstance(out, dict) else None
        _check_refusal(raw)
        parsed = out.get("parsed") if isinstance(out, dict) else out
        if parsed is None:
            parsed = _salvage(out, schema)
        return parsed, raw

    def _plain_json_call(
        self, model: Any, messages: list[Any], schema: type[T], last_err: str
    ) -> tuple[Any, Any]:
        """Ask for raw JSON with the schema shown, then parse it here.

        Works on any chat model, including ones whose structured-output mode is
        unreliable, because nothing depends on the provider honouring a schema
        parameter — only on it emitting JSON somewhere in its reply.
        """
        try:
            schema_text = json.dumps(schema.model_json_schema(), ensure_ascii=False, indent=1)[:4000]
        except Exception:
            schema_text = ", ".join(sorted(schema.model_fields))

        ask = HumanMessage(content=(
            (f"Your previous response could not be parsed.\nError: {last_err}\n\n" if last_err else "")
            + "Return ONLY a JSON object matching this schema. No prose, no markdown "
            "fence, nothing before or after the JSON:\n\n" + schema_text
        ))
        resp = model.invoke(messages[:2] + [ask])
        _check_refusal(resp)
        text = resp.content if isinstance(resp.content, str) else str(resp.content)
        for candidate in (strip_fence(text), _outermost_json(text), text):
            if not candidate:
                continue
            try:
                return schema.model_validate(json.loads(candidate)), resp
            except Exception:
                continue
        return None, resp

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
        if self.plan is not None:
            u["routing"] = self.plan.as_dict()
        u["deep_model"] = self.model_name("deep")
        u["fast_model"] = self.model_name("fast")
        u["estimated_cost_usd"] = round(self.ledger.estimated_cost_usd(), 2)
        return u

    def provenance_note(self, language: str = "en") -> str:
        """The sentence every report must carry about who actually judged things.

        Rendered in the deliverable's language: this line is the reader's only
        warning that names and definitions may have come from a stand-in rather
        than a model, and a warning nobody can read is not a warning.
        """
        zh = language == "zh"
        if self.is_offline:
            return (
                "本次运行中所有需要 LLM 判断的环节, 均由**确定性离线启发式替身**产出, "
                "**并非语言模型**。这些环节给出的名称、定义与标签由 n-gram 与正则证据算得, "
                "并以 `offline-heuristic` 标注。定量结果 (嵌入、聚类、各项指标) 不受影响, 完全真实。"
                if zh else
                "LLM-judgment steps in this run were produced by the deterministic "
                "offline heuristic stand-in, NOT by a language model. Names, definitions "
                "and labels from those steps are computed from n-gram and regex evidence "
                "and are marked `offline-heuristic`. Quantitative results (embeddings, "
                "clustering, metrics) are unaffected and fully real."
            )
        if zh:
            return (
                f"需要 LLM 判断的环节使用了 {self.model_name('deep')} (深层) 与 "
                f"{self.model_name('fast')} (快层), temperature={self.cfg.temperature}, "
                "响应按内容哈希缓存以保证可复现。"
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


def _salvage(out: Any, schema: type[BaseModel]) -> Any:
    """Rescue a structurally valid answer the strict parser refused.

    Tries, in order: the fenced block stripped; the outermost {...} span. Both are
    cheap and both recover a *correct* answer that would otherwise cost another
    full round trip — which on a 600-row annotation job is the difference between
    minutes and tens of minutes.
    """
    raw = out.get("raw") if isinstance(out, dict) else None
    text = getattr(raw, "content", None) if raw is not None else None
    if not isinstance(text, str) or not text.strip():
        return None

    for candidate in (strip_fence(text), _outermost_json(text)):
        if not candidate:
            continue
        try:
            return schema.model_validate(json.loads(candidate))
        except Exception:
            continue
    return None


def _outermost_json(text: str) -> str:
    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if 0 <= start < end else ""


#: Substrings that identify a failure of the *mechanism* rather than of one call.
#: Retrying such a call the same way reproduces it exactly, so the model is
#: switched to plain-JSON mode for the remainder of the run.
_NATIVE_SCHEMA_BROKEN = (
    "response_format",        # endpoint rejects the parameter: "unavailable now"
    "json_schema",
    "structured output",
    "does not support tools",
    "tool call validation",
    "function calling is not",
    # DashScope rejects json_object mode unless the prompt itself contains the
    # literal word "json". The role prompts are written in Chinese and mostly do
    # not, so native mode can never succeed on that endpoint for those roles —
    # a permanent condition, not a transient one.
    "must contain the word 'json'",
    'must contain the word "json"',
)


def _native_schema_is_broken(err: str) -> bool:
    """Whether `err` means native structured output cannot work on this model.

    A ValidationError means the model answered but not in the schema — on several
    OpenAI-compatible endpoints that is a markdown-fenced JSON body the SDK
    refuses. A BadRequest naming `response_format` means the endpoint never
    supported the parameter at all. Both are permanent for the run; a timeout or
    a rate limit is not, and must stay retryable.
    """
    if "ValidationError" in err:
        return True
    low = err.lower()
    return any(s in low for s in _NATIVE_SCHEMA_BROKEN)
