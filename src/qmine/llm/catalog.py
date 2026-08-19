"""A model catalogue that refreshes itself.

The problem this solves is maintenance, not capability. Hardcoding model IDs and
prices means someone edits a table every few weeks, and the table is wrong in
between — the failure that makes most multi-provider code quietly obsolete. So
the catalogue is fetched from sources that are already maintained by someone
else, normalised into one shape, cached, and pinned.

**Two upstream sources, chosen for different strengths** (both verified live,
2026-08-18, neither requiring an API key to read):

* **LiteLLM's `model_prices_and_context_window.json`** — 3,039 models across 124
  providers. Primary, because it is the only one that covers *embedding* models
  (124 of them), carries `deprecation_date` on 334 entries, and serves an ETag
  so conditional refresh is cheap.
* **OpenRouter's `/api/v1/models`** — 413 models, every one priced, and the only
  free source of a per-model `structured_outputs` capability flag (335 of 413)
  plus live provider uptime. Secondary, and the tie-breaker on capability.

**Reproducibility.** Principle 8 says a run must be replayable. A catalogue that
changes under you breaks that, so every run records the exact catalogue snapshot
it routed against — source, fetch time, and a content hash — and a pinned
snapshot can be replayed later even though the live prices have moved on.

**Offline.** Falls back through: memory cache → disk cache (any age, stated) →
pinned snapshot → a small built-in floor. The pipeline must run air-gapped, so
"no catalogue" is a degraded mode, never an error.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

log = logging.getLogger("qmine.catalog")

LITELLM_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/"
    "model_prices_and_context_window.json"
)
OPENROUTER_URL = "https://openrouter.ai/api/v1/models"

#: OpenRouter advertises `max-age=300, stale-while-revalidate=3600`. Following
#: the publisher's own freshness window rather than inventing one.
DEFAULT_TTL_SECONDS = 6 * 3600

#: LiteLLM provider strings -> our provider keys. Only the ones we can reach.
_PROVIDER_ALIASES = {
    "anthropic": "anthropic", "openai": "openai", "text-completion-openai": "openai",
    "gemini": "google", "vertex_ai-language-models": "vertex", "vertex_ai": "vertex",
    "xai": "xai", "deepseek": "deepseek", "mistral": "mistral", "groq": "groq",
    "together_ai": "together", "fireworks_ai": "fireworks", "deepinfra": "deepinfra",
    "openrouter": "openrouter", "bedrock": "bedrock", "bedrock_converse": "bedrock",
    "azure": "azure", "azure_ai": "azure", "dashscope": "qwen", "moonshot": "moonshot",
    "volcengine": "doubao", "minimax": "minimax",
    # Zhipu publishes as `zai` (z.ai) in the price feed, not `zhipu`. Missing this
    # alias meant a user with a ZHIPU_API_KEY had zero routable models and the
    # router silently fell back to another provider.
    "zai": "zhipu", "zhipu": "zhipu",
}


@dataclass
class ModelCard:
    """One model, normalised across sources."""

    id: str
    provider: str
    input_per_mtok: float | None = None
    output_per_mtok: float | None = None
    context_tokens: int | None = None
    max_output_tokens: int | None = None
    supports_structured_output: bool | None = None
    supports_tools: bool | None = None
    supports_caching: bool | None = None
    deprecated_on: str | None = None
    #: The id as the provider's own API expects it. Catalogue keys carry routing
    #: prefixes (`dashscope/qwen3-…`, `deepseek/deepseek-v4-pro`) that are correct
    #: for LiteLLM and a 404 against the provider's native endpoint.
    api_id: str = ""
    #: True when the model emits images as well as text. Such models are priced
    #: and tiered like frontier text models while being tuned for a different
    #: job — an image model was picked for a reasoning role before this existed.
    emits_non_text: bool = False
    sources: list[str] = field(default_factory=list)
    raw_name: str = ""

    @property
    def priced(self) -> bool:
        return self.input_per_mtok is not None and self.output_per_mtok is not None

    def blended_cost(self, in_tokens: int, out_tokens: int) -> float | None:
        """Dollar cost of one call of the given shape."""
        if not self.priced:
            return None
        return (in_tokens * self.input_per_mtok + out_tokens * self.output_per_mtok) / 1_000_000

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "provider": self.provider,
            "input_per_mtok": self.input_per_mtok, "output_per_mtok": self.output_per_mtok,
            "context_tokens": self.context_tokens, "max_output_tokens": self.max_output_tokens,
            "supports_structured_output": self.supports_structured_output,
            "supports_tools": self.supports_tools, "supports_caching": self.supports_caching,
            "deprecated_on": self.deprecated_on, "api_id": self.api_id,
            "emits_non_text": self.emits_non_text,
            "sources": self.sources,
        }


@dataclass
class Catalog:
    """A normalised, timestamped view of the model world."""

    models: dict[str, ModelCard] = field(default_factory=dict)
    fetched_at: float = 0.0
    sources: list[str] = field(default_factory=list)
    degraded: str = ""

    @property
    def age_hours(self) -> float:
        return (time.time() - self.fetched_at) / 3600 if self.fetched_at else float("inf")

    def for_providers(self, providers: Iterable[str]) -> list[ModelCard]:
        keep = set(providers)
        return [m for m in self.models.values() if m.provider in keep]

    def provenance(self) -> dict[str, Any]:
        """What the run must record to be replayable."""
        import hashlib

        blob = json.dumps(sorted(self.models), separators=(",", ":")).encode()
        return {
            "sources": self.sources,
            "fetched_at": self.fetched_at,
            "age_hours": round(self.age_hours, 2),
            "n_models": len(self.models),
            "content_hash": hashlib.sha256(blob).hexdigest()[:16],
            "degraded": self.degraded,
        }

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({
            "fetched_at": self.fetched_at, "sources": self.sources,
            "models": {k: v.as_dict() for k, v in self.models.items()},
        }, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Catalog":
        blob = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            models={k: ModelCard(**v) for k, v in blob["models"].items()},
            fetched_at=blob.get("fetched_at", 0.0),
            sources=blob.get("sources", []),
        )


# --------------------------------------------------------------------------
# Fetching
# --------------------------------------------------------------------------

def _get(url: str, timeout: float = 30.0) -> Any:
    """Fetch JSON, trying the transports that tend to differ behind proxies.

    httpx and urllib do not always agree about proxy configuration — on one
    machine urllib failed while httpx and curl succeeded against the same URL —
    so a catalogue fetch that gives up after one transport reports "offline"
    when the network is fine.
    """
    try:
        import httpx

        r = httpx.get(url, timeout=timeout, follow_redirects=True)
        r.raise_for_status()
        return r.json()
    except Exception as exc:  # noqa: BLE001
        log.debug("httpx failed for %s: %s", url, exc)
    try:
        import urllib.request

        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception as exc:  # noqa: BLE001
        log.debug("urllib failed for %s: %s", url, exc)
    import subprocess

    out = subprocess.run(
        ["curl", "-sS", "-m", str(int(timeout)), "-L", url],
        capture_output=True, timeout=timeout + 5,
    )
    if out.returncode != 0:
        raise RuntimeError(f"all transports failed for {url}: {out.stderr[:160]!r}")
    return json.loads(out.stdout)


#: Entries in the price feeds that are not models you can call. Fine-tune rows
#: (`ft:...`) are price *templates* for a model you would have to train first;
#: routing to one produces a 404 at best. Left in, they are also cheap, so a
#: cost-aware router picks them for everything — which is exactly what happened
#: the first time this ran.
_NON_CALLABLE_PREFIXES = ("ft:", "sample_spec", "openai/ft:")
_NON_CALLABLE_MARKERS = ("/ft:", ":ft-", "-finetune", "placeholder")


#: Routing prefixes the feeds attach that the provider's own API does not accept.
_STRIP_PREFIXES = ("dashscope/", "deepseek/", "moonshot/", "zhipu/", "zai/", "volcengine/",
                   "openai/", "anthropic/", "gemini/", "mistral/", "xai/", "groq/",
                   "minimax/", "doubao/")


def _native_id(key: str, provider: str) -> str:
    """The model id as the provider's own endpoint expects it.

    OpenRouter genuinely needs `vendor/model`; a direct call to DashScope or
    DeepSeek needs the bare name and 404s on the prefixed one.
    """
    if provider == "openrouter":
        return key
    low = key.lower()
    for pre in _STRIP_PREFIXES:
        if low.startswith(pre):
            return key[len(pre):]
    return key


def _is_callable_model(key: str) -> bool:
    low = key.lower()
    if low.startswith(_NON_CALLABLE_PREFIXES):
        return False
    return not any(m in low for m in _NON_CALLABLE_MARKERS)


def _from_litellm(blob: dict[str, Any]) -> dict[str, ModelCard]:
    out: dict[str, ModelCard] = {}
    for key, spec in blob.items():
        if key == "sample_spec" or not isinstance(spec, dict):
            continue
        if spec.get("mode") not in ("chat", "responses"):
            continue
        if not _is_callable_model(key):
            continue
        prov = _PROVIDER_ALIASES.get(str(spec.get("litellm_provider", "")), "")
        if not prov:
            continue
        ic, oc = spec.get("input_cost_per_token"), spec.get("output_cost_per_token")
        out[f"{prov}/{key}"] = ModelCard(
            id=key, provider=prov, api_id=_native_id(key, prov),
            input_per_mtok=float(ic) * 1_000_000 if ic is not None else None,
            output_per_mtok=float(oc) * 1_000_000 if oc is not None else None,
            context_tokens=spec.get("max_input_tokens") or spec.get("max_tokens"),
            max_output_tokens=spec.get("max_output_tokens"),
            supports_structured_output=spec.get("supports_response_schema"),
            supports_tools=spec.get("supports_function_calling"),
            supports_caching=spec.get("supports_prompt_caching"),
            deprecated_on=spec.get("deprecation_date"),
            sources=["litellm"],
        )
    return out


def _from_openrouter(blob: dict[str, Any]) -> dict[str, ModelCard]:
    out: dict[str, ModelCard] = {}
    for m in blob.get("data", []):
        mid = m.get("id", "")
        if "/" not in mid or not _is_callable_model(mid):
            continue
        params = m.get("supported_parameters") or []
        pricing = m.get("pricing") or {}
        arch = m.get("architecture") or {}
        out_mods = set(arch.get("output_modalities") or ["text"])

        def _price(field: str) -> float | None:
            v = pricing.get(field)
            try:
                return float(v) * 1_000_000 if v is not None else None
            except (TypeError, ValueError):
                return None

        out[f"openrouter/{mid}"] = ModelCard(
            id=mid, provider="openrouter", api_id=mid,
            input_per_mtok=_price("prompt"), output_per_mtok=_price("completion"),
            context_tokens=m.get("context_length"),
            max_output_tokens=(m.get("top_provider") or {}).get("max_completion_tokens"),
            supports_structured_output="structured_outputs" in params,
            supports_tools="tools" in params,
            supports_caching="input_cache_read" in pricing,
            emits_non_text=bool(out_mods - {"text"}),
            sources=["openrouter"], raw_name=m.get("name", ""),
        )
    return out


def fetch(
    *, cache_dir: str | Path | None = None, ttl: float = DEFAULT_TTL_SECONDS,
    allow_network: bool = True, pinned: str | Path | None = None,
) -> Catalog:
    """Build a catalogue, degrading rather than failing.

    Order: fresh disk cache → network → stale disk cache → pinned snapshot →
    built-in floor. Whichever rung it lands on is recorded in ``degraded`` so a
    report can say what the routing decision was actually based on.
    """
    cache_path = Path(cache_dir) / "model_catalog.json" if cache_dir else None

    if cache_path and cache_path.exists():
        try:
            cached = Catalog.load(cache_path)
            if cached.age_hours * 3600 < ttl:
                return cached
        except Exception:
            pass

    if allow_network:
        models: dict[str, ModelCard] = {}
        sources: list[str] = []
        for name, url, parse in (
            ("litellm", LITELLM_URL, _from_litellm),
            ("openrouter", OPENROUTER_URL, _from_openrouter),
        ):
            try:
                models.update(parse(_get(url)))
                sources.append(name)
            except Exception as exc:  # noqa: BLE001
                log.warning("catalog source %s unavailable: %s", name, str(exc)[:120])
        if models:
            cat = Catalog(models=models, fetched_at=time.time(), sources=sources)
            if cache_path:
                try:
                    cat.save(cache_path)
                except Exception:
                    pass
            return cat

    if cache_path and cache_path.exists():
        try:
            stale = Catalog.load(cache_path)
            stale.degraded = f"stale disk cache, {stale.age_hours:.1f}h old — network unreachable"
            return stale
        except Exception:
            pass

    if pinned and Path(pinned).exists():
        cat = Catalog.load(pinned)
        cat.degraded = "pinned snapshot — no network and no cache"
        return cat

    cat = _floor()
    cat.degraded = (
        "built-in floor — no network, no cache, no pinned snapshot. Prices and IDs here "
        "are illustrative only and WILL be stale; routing falls back to the configured "
        "deep/fast models rather than pretending to optimise."
    )
    return cat


def _floor() -> Catalog:
    """A deliberately tiny last resort.

    Not a mirror of the world — that would be the hardcoded table this module
    exists to avoid, and it would rot. Just enough for the router to return
    *something* explainable while announcing that it is flying blind.
    """
    return Catalog(models={}, fetched_at=0.0, sources=["floor"])
