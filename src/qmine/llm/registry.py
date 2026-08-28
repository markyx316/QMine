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
import threading
import time
from pathlib import Path
from typing import Any, ClassVar, Literal, TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
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


#: Fields worth showing a human, in the order they read best. Deliberately
#: generic: the panel must be legible on a corpus and a schema this project has
#: never seen, so nothing here names a QMine class.
_SALIENT = ("query", "text", "name", "name_zh", "code", "label", "final_label",
            "title", "verdict", "category", "rationale", "why")


def _one_line(item: Any) -> str:
    """Render one element of a returned list the way a person would read it."""
    if not isinstance(item, dict):
        return str(item)[:60]
    keys = [k for k in _SALIENT if k in item and item[k] not in (None, "", [])]
    if not keys:
        keys = [k for k, v in list(item.items())[:2] if isinstance(v, (str, int, float))]
    if len(keys) >= 2:
        return f"{str(item[keys[0]])[:26]} → {str(item[keys[1]])[:26]}"
    return str(item[keys[0]])[:52] if keys else ""


def summarize_return(value: Any) -> str:
    """What an agent returned, as a count plus a sample of the actual content.

    "the agent finished" is not information. "labels=25" is better. What an
    operator actually wants is to see the work: `labels=25 · 什么是光合作用 →
    EXPLAIN_CONCEPT`. Reading the largest list and rendering its first element
    from a fixed set of human-meaningful keys gets that without knowing which
    schema this is — which matters, because the point of this pipeline is to run
    on datasets and taxonomies nobody has written a formatter for.
    """
    if value is None:
        return ""
    if isinstance(value, BaseModel):
        value = value.model_dump()
    if isinstance(value, (list, tuple)):
        head = _one_line(value[0]) if value else ""
        return f"{len(value)} items" + (f" · {head}" if head else "")
    if not isinstance(value, dict):
        return str(value)[:70].replace("\n", " ")

    lists = {k: v for k, v in value.items() if isinstance(v, (list, tuple))}
    counts = ", ".join(f"{k}={len(v)}" for k, v in lists.items())
    # Sample the largest list that actually has readable content. Picking purely
    # by length lands on a list of bare ids and renders "· 1", which tells the
    # reader nothing and costs the line that could have shown the work.
    readable = {k: v for k, v in lists.items() if v and isinstance(v[0], dict)}
    biggest = max((readable or lists).items(), key=lambda kv: len(kv[1]),
                  default=(None, []))
    sample = _one_line(biggest[1][0]) if biggest[1] else ""
    if not counts:
        # No lists at all — a verdict, a decision, a short answer.
        parts = [f"{k}={str(v)[:34]}" for k, v in value.items()
                 if isinstance(v, (str, int, float, bool)) and str(v).strip()]
        return " · ".join(parts[:2])[:120]
    return (counts + (f" · {sample}" if sample else ""))[:140]


def _plan_lines(plan: Any) -> list[str]:
    """One line per role, plus every warning the router attached.

    Kept in the LLM layer rather than the CLI so it reaches the run log, the
    dashboard and a headless run alike — the places an operator actually looks
    when a run is already going.
    """
    from .router import lab_of, route_label

    rows = sorted(plan.assignments.items(), key=lambda kv: -kv[1].estimated_cost_usd)
    width = max((len(r) for r, _ in rows), default=8)
    out = ["  role assignments for this run (model · lab · est. calls · est. $):"]
    for role, a in rows:
        if not a.model:
            out.append(f"    {role:<{width}}  [UNSERVED]")
            continue
        out.append(f"    {role:<{width}}  {route_label(a)}  "
                   f"lab={lab_of(a.model)}  {a.estimated_calls} calls  "
                   f"${a.estimated_cost_usd:.3f}")
    # The independence property double-blind annotation depends on, stated in the
    # terms the rule is written in — by LAB, not by gateway, because two labs can
    # reach you through one gateway and look identical in the column above.
    trio = {r: lab_of(plan.assignments[r].model)
            for r in ("annotator_a", "annotator_b", "referee")
            if r in plan.assignments and plan.assignments[r].model}
    if len(trio) == 3:
        ok = len(set(trio.values())) == 3
        out.append("    annotator/referee labs: "
                   + ", ".join(f"{r.split('_')[-1]}={lab}" for r, lab in trio.items())
                   + (" — independent" if ok else " — NOT INDEPENDENT"))
    for role, a in rows:
        for w in a.warnings:
            out.append(f"    ! {role}: {w}")
    return out


class ModelRegistry:
    """Hands out models by role, accounts for usage, and caches responses."""

    def __init__(
        self,
        cfg: LLMConfig,
        *,
        cache_dir: str | os.PathLike[str] | None = None,
        #: The whole run config, when available. Only used to scale the pre-run
        #: cost estimate by the knobs that actually drive call volume.
        run_cfg: Any = None,
        ledger: UsageLedger | None = None,
    ) -> None:
        self.cfg = cfg
        self.run_cfg = run_cfg
        self.ledger = ledger or UsageLedger(
            max_calls=cfg.max_total_calls, max_output_tokens=cfg.max_total_output_tokens
        )
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.provider = self._resolve_provider(cfg.provider)
        self._models: dict[str, BaseChatModel] = {}
        self.raw_log: list[dict[str, Any]] = []
        #: Called once per agent turn with what it cost and what it returned, so a
        #: watcher can see the team working rather than inferring it from phase
        #: lines. Set by the runner; ``None`` in tests and library use.
        self.on_call: Any = None
        #: The raw response of the attempt running on THIS thread. Eight
        #: annotators run concurrently, so a plain attribute would race and
        #: attribute one thread's tokens to another's failure.
        self._raw = threading.local()
        # The generation cap actually in force for the call in flight. The failure
        # path needs it to recognise a truncation from the token count, which is
        # ground truth no matter what prose the provider returns.
        self._cap = threading.local()
        #: Role -> cumulative output tokens at its last reported call, so a call
        #: can be announced with ITS OWN cost rather than the role's running total.
        self._reported_out: dict[str, int] = {}
        self._reported_lock = threading.Lock()
        #: Providers that have proved unusable this run, and why. A dead provider
        #: must be remembered: with eight concurrent batches, rediscovering it per
        #: call costs three failed attempts each time.
        self._dead_providers: dict[str, str] = {}
        #: MODEL -> multiplier on the output cap, raised when that model truncates.
        #: Keyed on the model, not the role: verbosity is a property of the model,
        #: so one discovery should serve every role using it. Keyed per role, five
        #: researchers each paid ~180s and a discarded call to learn the same fact
        #: about `glm-5.2` — observed identically across two runs. This mirrors
        #: `_no_native_schema`, which is per-model and sticks for the same reason.
        self._length_bump: dict[str, int] = {}
        #: role -> [(provider, model), ...] alternates the router already chose.
        self._fallbacks: dict[str, list[tuple[str, str]]] = {}
        #: What we actually failed over to, for the run summary.
        self.failovers: list[dict[str, Any]] = []
        self.plan: Any = None          # RoutingPlan, when routing is enabled
        #: Models whose native structured-output mode has already failed once in
        #: this run. Retrying it per call is expensive: at 240 annotation calls,
        #: a wasted first attempt of 30-180s each roughly doubles the run.
        self._no_native_schema: set[str] = set()
        self._quirks_lock = threading.Lock()
        self._load_quirks()
        self._routed: dict[str, tuple[str, str]] = {}   # role -> (provider, model)
        self._model_output_cap: dict[tuple[str, str], int] = {}
        if cfg.provider == "router":
            self._build_routing_plan(cache_dir)


    #: SHAPED LIKE THE SCHEMAS THAT ACTUALLY FAIL, not like the cheapest possible
    #: one. A first attempt used `{ok: bool, note: str}` and `deepseek-v4-flash`
    #: and `glm-5.2` both passed it — while the same run learned from real
    #: failures that both need plain-JSON mode. Native structured output can work
    #: for a flat pair of scalars and break on a keyed map of nested objects,
    #: which is exactly what `RefereeBatch` is. Probe the hard shape or the probe
    #: reports success on models that will fail in production.
    class _ProbeItem(BaseModel):
        label: str = ""
        confident: bool = False
        why: str = ""

    class _Probe(BaseModel):
        items: dict[str, "ModelRegistry._ProbeItem"] = {}
        summary: str = ''

    def _probe_structured_output(self) -> None:
        """Find out which routed models cannot do native structured output —
        BEFORE a real call pays for the discovery.

        The repair machinery already handles a model that returns markdown or
        prose instead of JSON: it switches that model to plain-JSON mode for the
        rest of the run and remembers it in `model_quirks.json`. What it cannot
        do is make the discovery cheap, because the discovery happens on whatever
        call runs first — and roles fan out CONCURRENTLY.

        Measured on live39: the referee launched 8 calls at once, every one
        checked `_no_native_schema` before any had failed, and all 8 paid the
        discovery. Those calls take 69-555 seconds each and returned nothing.
        The model in question (`glm-4.5-airx`) was new to this run — live38's
        referee was `glm-5.2` — so a persisted quirk from the previous run could
        not have helped either.

        One tiny call per distinct model, once, costs a few tokens and seconds and
        removes the whole class of waste. Models already known from the quirks
        file are skipped, so a stable fleet probes nothing.
        """
        models: dict[str, tuple[str, str]] = {}
        for role, (prov, mid) in self._routed.items():
            if mid not in self._no_native_schema:
                models.setdefault(mid, (prov, role))
        if not models:
            return
        log.info("probing native structured output on %d model(s) so a real call "
                 "does not pay for the discovery", len(models))

        def _probe_one(mid: str, role: str) -> tuple[str, str | None]:
            try:
                model = self.get(role)
                # The literal word "json" is REQUIRED by some OpenAI-compatible
                # endpoints when a JSON response format is requested — DashScope
                # 400s with "'messages' must contain the word 'json'" otherwise,
                # which the first version of this probe mistook for the model
                # being incapable.
                msgs = [
                    SystemMessage(content="Reply with a json object matching the schema."),
                    HumanMessage(content=
                        'Return this json exactly: {"items": {"1": {"label": "A", '
                        '"confident": true, "why": "probe"}}, "summary": "ok"}'),
                ]
                parsed, _ = self._structured_call(model, msgs, ModelRegistry._Probe)
                if parsed is None:
                    raise ValueError("no parseable structured output")
                return mid, None
            except Exception as exc:  # noqa: BLE001
                # Anything at all — an unparseable answer, a refusal, a transport
                # blip — is treated as "do not rely on native mode here". The
                # plain-JSON path is universal and only slightly more expensive,
                # so a false positive costs far less than a false negative.
                return mid, str(exc)[:110]

        # Probed CONCURRENTLY and under a hard deadline. A probe is a convenience;
        # it must never be able to delay, let alone block, a run. `self.get(role)`
        # carries the ROLE's timeout, which for the referee is minutes — long
        # enough for a hung endpoint to stall startup outright.
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FTimeout

        # NOT a `with` block. `ThreadPoolExecutor.__exit__` joins every worker, so
        # a hung endpoint blocks at TEARDOWN even after its deadline has passed —
        # the deadline governs how long we READ for, not how long the pool lives.
        # Shut down without waiting and let the HTTP client's own timeout collect
        # the straggler; the run proceeds either way.
        pool = ThreadPoolExecutor(max_workers=min(8, len(models)))
        try:
            futs = {pool.submit(_probe_one, mid, role): mid
                    for mid, (_prov, role) in models.items()}
            deadline = time.time() + self.PROBE_TIMEOUT_S
            for fut, mid in futs.items():
                try:
                    _mid, err = fut.result(timeout=max(0.0, deadline - time.time()))
                except _FTimeout:
                    _mid, err = mid, f"no answer in {self.PROBE_TIMEOUT_S:.0f}s"
                except Exception as exc:  # noqa: BLE001
                    _mid, err = mid, str(exc)[:110]
                if err:
                    self._no_native_schema.add(_mid)
                    log.warning("%s failed the structured-output probe (%s); it will "
                                "start in plain-JSON mode", _mid, err)
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
        self._save_quirks()

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

    #: Where the per-model lessons live. NOT under `runs/<id>/`: the point is to
    #: carry them ACROSS runs, and a per-run path relearns them every time.
    QUIRKS_PATH: ClassVar[Path] = Path(".cache") / "model_quirks.json"
    #: Providers do fix their endpoints. Re-learn rather than believe forever.
    #: A probe that has not answered by now is not worth waiting for; the model
    #: simply starts in plain-JSON mode, which works everywhere.
    PROBE_TIMEOUT_S: float = 45.0
    QUIRKS_TTL_DAYS: ClassVar[float] = 14.0

    def _load_quirks(self) -> None:
        """Start the run already knowing what earlier runs discovered.

        `_no_native_schema` and `_length_bump` are learned the expensive way: a
        call fails, the registry works out why, and every later call on that
        model avoids it. That knowledge was then thrown away at process exit, so
        every run paid for it again. Measured on `live38`: FIVE rediscoveries in
        one run — `deepseek-v4-pro` and `deepseek-v4-flash` each rejecting
        `response_format`, `qwen3-next-80b` demanding the word "json", and
        `glm-5.2` truncating a researcher for 226s and then 528s before
        completing in 11,634 tokens on the plain-JSON path it could have started
        on.

        Learned, not hardcoded: nothing here names a model in the source, so a
        provider we have never seen is treated on its merits, and a provider that
        FIXES its endpoint is re-learned once the entry ages out.
        """
        try:
            raw = json.loads(self.QUIRKS_PATH.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 — absent or unreadable is simply "learn it again"
            return
        cutoff = time.time() - self.QUIRKS_TTL_DAYS * 86400
        n_schema = n_bump = 0
        for model, rec in (raw.get("models") or {}).items():
            if float(rec.get("learned_at", 0)) < cutoff:
                continue
            if rec.get("no_native_schema"):
                self._no_native_schema.add(model)
                n_schema += 1
            bump = int(rec.get("length_bump", 1) or 1)
            if bump > 1:
                self._length_bump[model] = bump
                n_bump += 1
        if n_schema or n_bump:
            log.info("provider quirks recalled from %s: %d model(s) start in "
                     "plain-JSON mode, %d with a raised output cap — not "
                     "rediscovered this run", self.QUIRKS_PATH, n_schema, n_bump)

    def _save_quirks(self) -> None:
        """Record a lesson so the next run does not buy it twice."""
        try:
            with self._quirks_lock:
                try:
                    raw = json.loads(self.QUIRKS_PATH.read_text(encoding="utf-8"))
                except Exception:  # noqa: BLE001
                    raw = {}
                models = raw.setdefault("models", {})
                now = time.time()
                for m in self._no_native_schema:
                    models.setdefault(m, {})["no_native_schema"] = True
                    models[m]["learned_at"] = now
                for m, bump in self._length_bump.items():
                    if bump > 1:
                        # Keep the HIGHEST bump ever needed. A run that only got
                        # as far as 2x must not overwrite a 4x another run proved
                        # necessary — the point of remembering is to start from
                        # the worst case already known, not the most recent.
                        prev = int(models.get(m, {}).get("length_bump", 1) or 1)
                        models.setdefault(m, {})["length_bump"] = max(prev, int(bump))
                        models[m]["learned_at"] = now
                self.QUIRKS_PATH.parent.mkdir(parents=True, exist_ok=True)
                tmp = self.QUIRKS_PATH.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(raw, indent=1, sort_keys=True), encoding="utf-8")
                tmp.replace(self.QUIRKS_PATH)      # atomic: concurrent workers race here
        except Exception:  # noqa: BLE001 — never let bookkeeping break a run
            pass

    def _build_routing_plan(self, cache_dir: Any) -> None:
        """Resolve a model per role from the live catalogue and available keys.

        Degrades to the static two-tier behaviour whenever anything is missing —
        no keys, no catalogue, no network. A router that fails closed would make
        the pipeline unrunnable offline, which is a worse outcome than routing
        naively.
        """
        from .catalog import fetch
        from .providers import detect
        from .router import UnroutablePin, route

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
            scaled = None
            if self.run_cfg is not None:
                from .requirements import scaled_requirements

                scaled = scaled_requirements(self.run_cfg)
            self.plan = route(
                cat, av.usable, requirements=scaled,
                prefer=self.cfg.model_overrides or None,
            capable_models=self.cfg.capable_models or (),
                budget_usd=self.cfg.budget_usd,
                prefer_chinese_native=self.cfg.prefer_chinese_native,
                excluded_labs=self.cfg.excluded_labs,
            )
            self._routed = {
                r: (a.provider, a.api_model or a.model)
                for r, a in self.plan.assignments.items() if a.model
            }
            # Register the alternates the router already chose. They were being
            # computed, written into `run_manifest.json`, printed by `qmine models`
            # — and never consulted at call time, so a provider outage the router
            # had explicitly planned around still took the run down.
            self._fallbacks = {
                r: [tuple(f.split(":", 1)) for f in (a.fallbacks or []) if ":" in f]
                for r, a in self.plan.assignments.items() if a.model
            }
            # Teach the ledger what the chosen models actually charge, so spend is
            # measured at the prices the router used to choose them.
            self.ledger.rates = {
                r: (a.input_per_mtok, a.output_per_mtok)
                for r, a in self.plan.assignments.items()
                if a.model and a.input_per_mtok is not None
                and a.output_per_mtok is not None
            }
            self._model_output_cap = {
                (a.provider, a.api_model or a.model): a.max_output_tokens
                for a in self.plan.assignments.values()
                if a.model and a.max_output_tokens
            }
            self.provider = "routed"
            log.info("routing plan: %d roles, estimated $%.2f, catalogue %s",
                     len(self._routed), self.plan.total_cost_usd, cat.sources)
            # SHOW WHICH MODEL IS DOING WHICH JOB, BEFORE ANY OF IT HAPPENS.
            #
            # A run used to print one summary line, so the only way to learn that
            # the referee was on a lightweight model — or that the observer was
            # the same model as the architect it reviews — was to read
            # `run_manifest.json` afterwards, or to run `qmine models` separately
            # and hope the config matched. Both of those were discovered here the
            # hard way. The plan is known before the first call; printing it costs
            # nothing and is the last cheap moment to stop a misrouted run.
            for line in _plan_lines(self.plan):
                log.info("%s", line)
            self._probe_structured_output()
        except UnroutablePin:
            # A BAD PIN IS A CONFIG ERROR AND MUST NOT DEGRADE.
            #
            # The handler below exists so a missing catalogue or a dead network
            # cannot make the pipeline unrunnable. It also used to swallow an
            # unroutable pin, which is the opposite situation: nothing is
            # missing, the user asked for a specific model, and degrading runs
            # the whole pipeline on the static tiers instead — every pin
            # ignored, announced by a single `warning` line. Fail closed.
            raise
        except Exception as exc:  # noqa: BLE001
            log.warning("routing unavailable (%s); falling back to the static tiers", exc)
            self.provider = self._resolve_provider("auto")

    def _live_route(self, role: str) -> tuple[str, str] | None:
        """The role's route, skipping providers already proved dead this run.

        Consulted before the plan so that once a provider is known unusable, every
        later call goes straight to the alternate instead of rediscovering the
        outage — at eight concurrent batches that rediscovery costs three failed
        attempts per batch.
        """
        if not self._dead_providers:
            return None
        primary = self._routed.get(role) or self._prefix_route(role)
        if primary is None or primary[0] not in self._dead_providers:
            return None
        base = role if role in self._fallbacks else next(
            (b for b in sorted(self._fallbacks, key=len, reverse=True)
             if role.startswith(b)), None)
        for prov, model in self._fallbacks.get(base or role, []):
            if prov not in self._dead_providers:
                return (prov, model)
        return None

    def _prefix_route(self, role: str) -> tuple[str, str] | None:
        for base in sorted(self._routed, key=len, reverse=True):
            if role.startswith(base):
                return self._routed[base]
        return None

    def mark_provider_dead(self, provider: str, reason: str, role: str = "") -> None:
        """Record that a provider cannot serve this run, and what replaced it."""
        if provider in self._dead_providers:
            return
        self._dead_providers[provider] = reason
        alt = self._live_route(role) if role else None
        self.failovers.append({"provider": provider, "reason": reason[:160],
                               "role": role, "replaced_by": alt})
        log.warning("provider %s is unusable this run (%s); %s", provider, reason[:120],
                    f"failing over to {alt[0]}:{alt[1]}" if alt else
                    "NO USABLE FALLBACK — calls on this provider will fail")
        self._models.clear()          # cached clients point at the dead endpoint

    def route_for(self, role: str) -> tuple[str, str] | None:
        """The (provider, model) chosen for a role, if routing is active.

        Roles arrive suffixed — ``researcher_log_reading``, ``namer_3``,
        ``annotator_a`` — because the same role runs as several distinct agents.
        The routing plan is keyed on the base role, so an exact-match-only lookup
        silently misses every suffixed agent and drops them onto the static
        fallback path. That is how a live run ended up asking for a model called
        ``routed:claude-sonnet-5``.
        """
        live = self._live_route(role)
        if live is not None:
            return live
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
        # The role's declared budget is authoritative — `cfg.max_tokens` is the
        # default for roles that declare nothing, not a ceiling over those that
        # do. Clamping to it truncated the taxonomy architect at 16k while its
        # own requirement asked for 24k.
        cap = req.max_output_tokens * self._length_bump.get(model_id, 1)
        published = self._model_output_cap.get((provider, model_id))
        if published:
            cap = min(cap, int(published))
        # Derive the deadline from the cap actually in force. `req.timeout_seconds`
        # is computed from the DECLARED cap, so a bumped role kept the old
        # deadline: the architect once doubled to 84,000 tokens with 1,365s to
        # emit them, and a researcher at 24,000 needs ~630s against a 390s
        # timeout. Two constants that only make sense together, moved apart.
        self._cap.last = cap
        needed = (cap / req.THROUGHPUT_TOK_PER_SEC) * 1.3
        timeout = max(self.cfg.request_timeout, req.timeout_seconds,
                      min(1800.0, max(180.0, needed)))
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

        # TURN OFF REASONING WHERE IT IS PURE COST.
        #
        # `deepseek-v4-flash` and `glm-5.2` both began emitting `reasoning_content`
        # under the SAME model names between live40 and live41, and the tokens are
        # billed. Measured directly on both endpoints with one trivial probe:
        #
        #   deepseek-v4-flash  505 completion tokens, 497 of them reasoning ->   9
        #   glm-5.2            237 completion tokens, 227 of them reasoning ->  10
        #
        # — identical answers. On live41 that changed annotator_a from 202 to
        # 1,030 output tokens per label (5x) and took the run to $28 by phase 5
        # against a $9.11 estimate. It also caused the referee's 88% failure rate:
        # reasoning consumed the output budget before the JSON was written, so the
        # response truncated and surfaced as "no parseable structured output" —
        # which plain-JSON mode cannot fix, because the schema wrapper was never
        # the problem. Both models were already in plain-JSON mode and failing.
        #
        # Only bulk-classification roles are silenced. A taxonomy architect or an
        # observer is exactly where deliberation earns its tokens; an annotator
        # emitting 25 labels is not. `enable_thinking: false` does NOT work on
        # DeepSeek (measured: still 1,037 reasoning tokens) — this parameter does.
        # `extra_body`, NOT `model_kwargs`. LangChain forwards `model_kwargs` as
        # TOP-LEVEL arguments to the OpenAI SDK, so a non-standard body field
        # raises `TypeError: Completions.create() got an unexpected keyword
        # argument 'thinking'` on the first call and every retry. That is what it
        # did: live41 gen02 lost 24 annotation batches in one second, each failing
        # in 0.0s with no tokens billed, and the pilot gate then reported
        # "kappa nan on 0 queries". Vendor-specific body fields go in `extra_body`,
        # which the SDK passes through untouched.
        extra = reasoning_kwargs(role, provider)
        if extra:
            kwargs["extra_body"] = {**kwargs.get("extra_body", {}), **extra}

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
            self._raw.last = None
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
                # A rejected REQUEST (a 400) generated nothing and is genuinely
                # free. A rejected RESPONSE was generated and billed in full.
                # Recording both as zero made the ledger — and the output-token
                # ceiling that reads it — optimistic exactly on the paths that
                # misbehave, which is where a runaway would show up first.
                spent = getattr(self._raw, "last", None)
                usage = getattr(spent, "usage_metadata", None) or {}
                self.ledger.record(
                    role, error=True,
                    input_tokens=int(usage.get("input_tokens", 0) or 0),
                    output_tokens=int(usage.get("output_tokens", 0) or 0),
                )
                self.report_call(role, self.model_name(tier, role), tier,
                                 time.time() - t0, None, ok=False,
                                 note=type(exc).__name__)
                # A dead provider is not a retryable condition. Mark it, swing to
                # the alternate the router already picked, and re-issue THIS call
                # rather than burning the remaining attempts on the same endpoint
                # and dropping the batch.
                # Ran out of room, not out of capability. Give it more and ask
                # again — swapping providers here would abandon a working model
                # for a problem the model did not have.
                # A truncation is a TOKEN COUNT, not a phrase. `glm-5.2` returns
                # "no parseable structured output" for a response that stopped at
                # its ceiling, and the prose matcher below never fired: on
                # `live36` ten referee calls died at exactly 24,001 tokens — the
                # cap — and none was recognised as needing more room.
                cap_in_force = int(getattr(self._cap, "last", 0) or 0)
                out_spent = int(usage.get("output_tokens", 0) or 0)
                ran_out_of_room = bool(cap_in_force and out_spent >= cap_in_force)
                if (_hit_length_limit(last_err) or ran_out_of_room) and attempt < max_repair:
                    truncated = self.model_name(tier, role)
                    # RAISE TO a floor, do not multiply. Concurrent callers each
                    # truncate before any of them has recorded the switch, so
                    # multiplying compounds one problem into 4x or 8x — and the
                    # timeout now derives from the cap, so it inflates with it.
                    # The plain-JSON switch below is what actually fixes this;
                    # the multiplier only has to cover the difference between a
                    # schema-wrapped answer and a plain one.
                    # RAISE TO A FLOOR, never multiply: concurrent callers each
                    # truncate before any records the switch, and multiplying
                    # compounds one problem into 4x then 8x. Raising to a floor is
                    # idempotent — every racing caller computes the same value.
                    # But a 2x floor that ALSO truncates proves 2x was not enough,
                    # and `max(current, 2)` could never say so: the referee sat at
                    # 24,001 for ten consecutive calls. Step to a second floor on
                    # that evidence, and stop there.
                    self._length_bump[truncated] = _next_length_floor(
                        self._length_bump.get(truncated, 1))
                    # More room is only half of it. Measured four times on
                    # `glm-5.2`: the call truncates past 12,000 tokens in NATIVE
                    # structured-output mode and then completes in ~5,300 on the
                    # plain-JSON path — the same answer for less than half the
                    # tokens. Native mode's schema scaffolding is what overran the
                    # cap, so a truncation is also evidence that mode is unsuited
                    # to this model, exactly as an unparseable response is.
                    self._no_native_schema.add(truncated)
                    self._save_quirks()
                    log.warning(
                        "role=%s hit its output cap on %s; retrying with %dx room "
                        "in plain-JSON mode (both apply to every role on this model)",
                        role, truncated, self._length_bump[truncated])
                    model = self.get(role)
                    continue
                if _provider_is_unusable(last_err, exc):
                    failed_provider = (self.route_for(role) or (self.provider, ""))[0]
                    self.mark_provider_dead(failed_provider, last_err, role)
                    routed = self.route_for(role)
                    if routed and routed[0] != failed_provider:
                        model = self.get(role)
                        tier = ROLE_TIER.get(role, "fast")
                        # The alternate is a different model, so anything learned
                        # about the dead one's quirks does not carry over.
                        continue
                    raise LLMUnavailable(
                        f"role={role} provider {failed_provider} is unusable "
                        f"({last_err[:120]}) and no fallback is available"
                    ) from exc
                if attempt >= max_repair:
                    break
                if attempt == 0 and _native_schema_is_broken(last_err):
                    # Native structured output is unreliable on this model. Note it
                    # once and go straight to plain-JSON mode for the rest of the
                    # run rather than paying for the discovery on every call.
                    key = self.model_name(tier, role)
                    if key not in self._no_native_schema:
                        self._no_native_schema.add(key)
                        self._save_quirks()
                        log.warning(
                            "%s returns unparseable structured output; switching to "
                            "plain-JSON mode for the remainder of this run", key
                        )
                # SAY WHICH REMEDY IS ACTUALLY BEING APPLIED.
                #
                # This printed "repairing via plain-JSON mode" for every failure,
                # including transport errors where the schema is irrelevant and
                # nothing is being repaired — the call is simply retried. On live41
                # that produced 23 `APIConnectionError` lines all claiming a
                # JSON-mode repair, which reads as a schema problem with the model
                # and sent a reader looking in the wrong place.
                transport = not _native_schema_is_broken(last_err)
                log.warning(
                    "role=%s attempt=%d failed (%s); %s",
                    role, attempt, last_err[:160],
                    "transient transport error — retrying the same call" if transport
                    else "repairing via plain-JSON mode")

        raise LLMUnavailable(f"role={role} failed after {max_repair + 1} attempts: {last_err}")

    def _structured_call(self, model: Any, messages: list[Any], schema: type[T]) -> tuple[Any, Any]:
        """The provider's native structured output, with salvage on a soft miss."""
        try:
            runnable = model.with_structured_output(schema, method="json_schema", include_raw=True)
        except Exception:
            runnable = model.with_structured_output(schema, include_raw=True)
        out = runnable.invoke(messages)
        raw = out.get("raw") if isinstance(out, dict) else None
        # Stash it before validation can reject it. The provider has already
        # generated — and billed — these tokens; whether we can parse the result
        # is our problem, not a reason to record the attempt as free.
        self._raw.last = raw
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
        self._raw.last = resp
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

    def _external_key(self, role: str, tier: str, system: str, user: str) -> str:
        """The cache key for a turn that bypassed `complete`. One definition, so
        the reader and the writer cannot drift apart."""
        return hash_params({"role": role, "provider": self.provider,
                            "model": self.model_name(tier, role),
                            "system": system, "user": user, "schema": "tool-loop"},
                           length=32)

    def replay_external_turn(self, role: str, tier: str, system: str, user: str,
                             schema: type[T] | None = None) -> Any | None:
        """A previously cached tool-loop turn, if this exact prompt was asked before.

        The tool path wrote cache entries and never read them, which quietly made
        "resume after a failure" impossible: the two web-researching agents fetch
        live pages, so re-running them returns different candidates, which changes
        the architect's prompt, which misses ITS cache, and so on through every
        annotation call downstream. Twice today a resume that should have replayed
        ~55 minutes of work replayed almost none of it for exactly this reason.

        Replaying a web turn means reusing the pages that run saw rather than
        today's — which is the correct behaviour for reproducing a run, and the
        reason the entry is keyed on the prompt rather than on the fetch.
        """
        if not self.cfg.cache_llm_calls:
            return None
        hit = self._cache_get(self._external_key(role, tier, system, user))
        if hit is None:
            return None
        self.ledger.record(role, cached=True)
        try:
            return schema.model_validate(hit) if schema else hit
        except Exception:  # noqa: BLE001 — a stale shape must not break the run
            return None

    def record_external_turn(self, role: str, tier: str, system: str, user: str,
                             value: Any, latency: float) -> None:
        """File the two records `_store` files, for a turn that bypassed `complete`.

        `ToolAgent` drives `create_agent` directly, so the web-researching agents
        never reached `_store` — the only writer of `llm_cache/` and of the
        `raw_log` that becomes `agent_transcript.json`. The result: the two agents
        whose claims are *least* verifiable, because they cite pages nobody else
        saw, were the only two with no record of what they returned and no cached
        response to replay. Auditability should not depend on which code path an
        agent happened to take.
        """
        self._store(role, self._external_key(role, tier, system, user), value,
                    system, user, tier, time.time() - latency)

    def report_call(self, role: str, model: str, tier: str, latency: float,
                    value: Any, *, ok: bool = True, note: str = "",
                    call_key: str = "") -> None:
        """Announce one agent turn: what it was, what it cost, what it returned.

        `call_key` IS THE JOIN KEY, and it exists because two streams describe the
        same call and had nothing in common. `on_call` carries the one-line
        summary a dashboard shows; `raw_log` carries the full return. With no
        shared id the HTML dashboard paired them by POSITION — a global index over
        a reversed list, against a per-role chronological list — so an expanded
        row showed a different call's output. Observed on live42: the row headed
        `reporter ... 04:42:11` (the first attempt at `audit_and_limits`) opened
        onto the top-down taxonomy section, an earlier call entirely. A reader has
        no way to tell that apart from a correct answer, which makes it worse than
        showing nothing.
        """
        if not self.on_call:
            return
        u = self.ledger.by_role.get(role, {})
        total_out = int(u.get("output_tokens", 0) or 0)
        # THIS CALL's cost, not the role's running total. Announcing the total on
        # a per-call line reads as per-call and is wrong by a factor of the call
        # count: a referee line saying "out 298,406" was in fact 24,001 for that
        # call on a role that had spent 298,406 overall. Exact for a sequential
        # role; for a concurrent one the calls interleave, so the split between
        # simultaneous callers is approximate while their sum stays right.
        with self._reported_lock:
            call_out = max(0, total_out - self._reported_out.get(role, 0))
            self._reported_out[role] = total_out
        try:
            self.on_call({
                "role": role, "model": model, "tier": tier, "ok": ok,
                "latency_s": round(latency, 1), "note": note,
                "calls": u.get("calls", 0), "errors": u.get("errors", 0),
                "output_tokens": total_out, "call_output_tokens": call_out,
                "returned": summarize_return(value) if ok else note,
                "key": call_key,
            })
        except Exception:  # noqa: BLE001 — a watcher must never break a run
            pass

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
        # `model_name(tier)` without the role skips routing and falls back to the
        # tier default, so every routed run recorded `claude-opus-5` /
        # `claude-sonnet-5` no matter which provider actually answered. The cache
        # KEY passes the role and was always right, so replay was unaffected — but
        # `agent_transcript.json` is the audit trail, and it named the wrong model
        # for all 244 calls of the last live run.
        model = self.model_name(tier, role)
        if self.cfg.cache_llm_calls:
            self._cache_put(
                key,
                payload,
                {"role": role, "tier": tier, "model": model, "ts": time.time()},
            )
        self.report_call(role, model, tier, time.time() - t0, payload,
                         ok=True, call_key=key)
        if self.cfg.record_raw_outputs:
            _spent = getattr(self._raw, "last", None)
            _usage = getattr(_spent, "usage_metadata", None) or {}
            self.raw_log.append(
                {
                    "role": role,
                    "tier": tier,
                    "model": model,
                    "cache_key": key,
                    "latency_s": round(time.time() - t0, 2),
                    # PER CALL. Setting a role's budget used to require
                    # differencing cumulative totals out of `run.log`, which only
                    # works for sequential roles and silently misleads for
                    # concurrent ones. Record it where it is already known.
                    "input_tokens": int(_usage.get("input_tokens", 0) or 0),
                    "output_tokens": int(_usage.get("output_tokens", 0) or 0),
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
        # Same trap: with no role these resolve to the CONFIG defaults, not to
        # anything that ran. Keep them for the unrouted case, but say plainly
        # what they are and publish the models actually used alongside.
        if self._routed:
            u["models_used"] = sorted({m for _, m in self._routed.values()})
            u["deep_model"] = u["fast_model"] = None
            u["tier_defaults_unused"] = [self.cfg.deep_model, self.cfg.fast_model]
        else:
            u["deep_model"] = self.cfg.deep_model
            u["fast_model"] = self.cfg.fast_model
        u["estimated_cost_usd"] = round(self.ledger.estimated_cost_usd(), 2)
        # Name any role priced by the fallback rather than by its model's own
        # rate, so a guess is never read as a measurement.
        if self.ledger.unpriced_roles:
            u["unpriced_roles"] = self.ledger.unpriced_roles
        # A run that silently finished on its second-choice models is not the run
        # the routing plan describes, and every downstream number was produced by
        # whatever actually answered.
        if self.failovers:
            u["failovers"] = self.failovers
            u["dead_providers"] = dict(self._dead_providers)
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
        # NAME THE MODELS THAT ACTUALLY RAN. `model_name(tier)` takes no role, so
        # its routing lookup is skipped and it returns `cfg.deep_model` /
        # `cfg.fast_model` — the DEFAULTS. On `live38` that put
        # "claude-opus-5 / claude-sonnet-5" into a client-facing Chinese report
        # for a run served entirely by deepseek, qwen and zhipu. A deliverable
        # that misstates which models produced it is worse than one that says
        # nothing, and no test would catch it because the sentence is well-formed.
        by_model: dict[str, list[str]] = {}
        for role, (_prov, model) in sorted(self._routed.items()):
            by_model.setdefault(model, []).append(role)
        if by_model:
            parts = [f"{m} ({', '.join(rs)})" for m, rs in sorted(by_model.items())]
            if zh:
                return (
                    f"需要 LLM 判断的环节由路由按角色分派给 {len(by_model)} 个模型: "
                    + "; ".join(parts)
                    + f"。temperature={self.cfg.temperature}, 响应按内容哈希缓存以保证可复现。"
                )
            return (
                f"LLM-judgment steps were routed per role across {len(by_model)} model(s): "
                + "; ".join(parts)
                + f". temperature={self.cfg.temperature}, responses cached by content "
                  "hash for reproducibility."
            )

        # No routing plan: the tier defaults ARE what ran.
        if zh:
            return (
                f"需要 LLM 判断的环节使用了 {self.cfg.deep_model} (深层) 与 "
                f"{self.cfg.fast_model} (快层), temperature={self.cfg.temperature}, "
                "响应按内容哈希缓存以保证可复现。"
            )
        return (
            f"LLM-judgment steps used {self.cfg.deep_model} (deep tier) and "
            f"{self.cfg.fast_model} (fast tier) at temperature {self.cfg.temperature}, "
            f"with responses cached by content hash for reproducibility."
        )


#: Models that removed the sampling parameters entirely and return 400 on any value.
_NO_TEMPERATURE = ("claude-opus-5", "claude-opus-4-8", "claude-opus-4-7", "claude-fable-5", "claude-mythos-5")


#: Roles whose work is bulk classification against a fixed guide, where a
#: reasoning trace is billed output that no downstream step reads. Deliberation
#: roles are deliberately absent — see the block in `_build` that uses this.
#:
#: `referee` and `namer` were here and were REMOVED. Neither is bulk
#: classification against a fixed guide, which is the only thing this set is
#: meant to cover:
#:
#: * the referee adjudicates rows the two annotators DISAGREED on — the residue
#:   left after the easy cases are gone — and its output is not just a label but
#:   the adjudication rules that reach the annotator (see
#:   `test_every_rule_the_referee_drafts_reaches_the_annotator`). It is also the
#:   role whose stated discriminator was once vacuous against its own evidence.
#: * the namer writes the class names that appear in the deliverable, and a split
#:   re-names BOTH halves. A name is authored prose about a cluster's contents,
#:   not a lookup.
#:
#: Both are low-volume relative to the annotators, so the billed-trace argument
#: that justifies this set does not apply to them.
NO_REASONING_ROLES: frozenset[str] = frozenset({
    "annotator", "annotator_a", "annotator_b",
})

#: Providers measured to accept `thinking: {"type": "disabled"}`. Sending an
#: unknown parameter is not free — an endpoint that rejects it fails the call —
#: so this is an allowlist rather than a hopeful broadcast.
REASONING_TOGGLE_PROVIDERS: frozenset[str] = frozenset({"deepseek", "zhipu", "qwen"})


def reasoning_kwargs(role: str, provider: str) -> dict[str, Any]:
    """`{"thinking": {"type": "disabled"}}` for bulk roles on providers that take it.

    Split out so the RULE can be tested without an API key — building a routed
    model needs a live client, and a constant nothing sends is a constant that
    does nothing.
    """
    if role in NO_REASONING_ROLES and provider in REASONING_TOGGLE_PROVIDERS:
        return {"thinking": {"type": "disabled"}}
    return {}


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


#: Phrases that mean the PROVIDER cannot serve this call, however many times we
#: ask. Distinct from a schema miss or a truncation, which mean the model answered
#: and we could not use the answer — those have opposite remedies.
_PROVIDER_DEAD_PHRASES = (
    "insufficient balance", "invalid api key", "unauthorized", "payment required",
    "quota exceeded", "exceeded your quota", "billing", "account is suspended",
    "authenticationerror", "permissiondeniederror",
)

#: A status code only counts when it appears where a status code appears. Matching
#: bare digits killed a provider on `CompletionUsage(completion_tokens=4013,
#: prompt_tokens=8402)` — "4013" contains "401" and "8402" contains "402" — so a
#: model hitting its length limit marked the whole of Qwen dead and sent taxonomy
#: research to a coding model.
_STATUS_RE = re.compile(
    r"(?:error\s+code|status(?:\s+code)?|http)\s*[:=]?\s*(401|402|403)\b", re.I
)


#: HTTP statuses that mean the CREDENTIAL or the account is done, not this call.
#: 429 and 5xx are deliberately absent: they are transient, and abandoning a
#: working provider over one is worse than the outage it was meant to survive.
_DEAD_STATUS = frozenset({401, 402, 403})


def _provider_is_unusable(err: str, exc: BaseException | None = None) -> bool:
    """Whether this provider is done for the run, not just for this call.

    A 402 on a live run took twelve gold batches — 300 rows — while the retry
    loop asked the same dead endpoint three times per batch and the two declared
    fallbacks sat unused in the routing plan.

    PREFER THE STRUCTURED STATUS. The SDK raises typed errors carrying
    `status_code`, and reading it settles the question exactly. Scanning the
    error TEXT for a number cannot: `completion_tokens=4013` contains "401" and
    `prompt_tokens=8402` contains "402", which once killed every Qwen model on a
    live run and sent taxonomy research to a coding model. The regex below is now
    anchored, but the string path stays a fallback for providers that raise plain
    exceptions — the structured answer is always better when there is one.

    Rate limits, timeouts and truncations are deliberately NOT here.
    """
    code = getattr(exc, "status_code", None)
    if isinstance(code, bool):        # bool is an int subclass; never a status
        code = None
    if isinstance(code, int):
        return code in _DEAD_STATUS   # authoritative: no prose involved

    low = err.lower()
    if any(phrase in low for phrase in _PROVIDER_DEAD_PHRASES):
        return True
    return bool(_STATUS_RE.search(err))


def _next_length_floor(current: int) -> int:
    """Raise a model's generation cap to a FLOOR, never by a multiplier.

    Concurrent callers each truncate before any of them has recorded the switch,
    so multiplying compounds one problem into 4x and then 8x — and the timeout
    derives from the cap, so it inflates with it. A floor is idempotent: every
    racing caller computes the same value, however many of them run it.

    TWO floors, not one. `max(current, 2)` could not express "2x was still not
    enough", and on `live36` that cost ten consecutive referee calls: each died
    at exactly 24,001 tokens — the 12,000 cap already bumped to 2x — and each
    silently discarded 25 adjudications. Truncating while ALREADY at 2x is
    evidence for 4x. It stops there; beyond that the budget itself is wrong and
    should be fixed in `requirements.py`, not papered over here.
    """
    return 4 if current >= 2 else 2


def _hit_length_limit(err: str) -> bool:
    """Whether the model stopped because it ran out of room, not because it failed.

    A truncation is fixable and the fix is not a different provider: it is more
    room. `glm-5.2` is markedly more verbose than the `glm-4.5-airx` these budgets
    were measured on and overran the researcher's 12,000-token cap on its first
    call. The error text carries a `CompletionUsage(...)` with token counts in it,
    which is also what made this look like a `402` to a substring matcher.
    """
    low = err.lower()
    return ("lengthfinishreason" in low
            or "length limit was reached" in low
            or "finish_reason" in low and "length" in low)


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
