"""Which model providers this machine can actually reach.

Routing starts here. There is no point ranking models the user cannot call, so
the first question is always "which API keys exist", and the answer has to come
from the environment rather than from a config file the user forgot to update.

Two design choices worth stating.

**Detection is by environment variable, and reported honestly.** A key being
present means the provider is *configured*, not that it *works* — the key may be
revoked, out of credit, or scoped to a different project. ``probe()`` exists to
make the stronger claim, and the router records which claim it relied on.

**Aggregators are first-class.** A single OpenRouter key reaches hundreds of
models across every major lab, which for a user who does not want a dozen
accounts is the whole answer. Treating it as just another provider would
understate what it unlocks.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Literal

log = logging.getLogger("qmine.providers")

ProviderKind = Literal["direct", "aggregator", "cloud"]


@dataclass(frozen=True)
class ProviderSpec:
    """A provider we know how to reach, and how to tell whether we can."""

    key: str
    display: str
    kind: ProviderKind
    env_vars: tuple[str, ...]
    #: LangChain provider prefix for ``init_chat_model``, when one exists.
    langchain_prefix: str = ""
    #: OpenAI-compatible base URL, for providers reachable that way.
    openai_base_url: str = ""
    #: Alternate base URLs to try when the primary rejects the key. Alibaba and a
    #: few others split their API by region, and a key issued for one region
    #: returns a plain 401 on the other — indistinguishable from a bad key unless
    #: you know to try the other host.
    alt_base_urls: tuple[str, ...] = ()
    #: Endpoint that lists models, if the provider publishes one.
    models_endpoint: str = ""
    notes: str = ""


PROVIDERS: tuple[ProviderSpec, ...] = (
    ProviderSpec("anthropic", "Anthropic", "direct", ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"),
                 langchain_prefix="anthropic",
                 models_endpoint="https://api.anthropic.com/v1/models"),
    ProviderSpec("openai", "OpenAI", "direct", ("OPENAI_API_KEY",),
                 langchain_prefix="openai",
                 openai_base_url="https://api.openai.com/v1",
                 models_endpoint="https://api.openai.com/v1/models"),
    ProviderSpec("google", "Google Gemini", "direct", ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
                 langchain_prefix="google_genai"),
    ProviderSpec("xai", "xAI Grok", "direct", ("XAI_API_KEY",),
                 langchain_prefix="xai",
                 openai_base_url="https://api.x.ai/v1"),
    ProviderSpec("deepseek", "DeepSeek", "direct", ("DEEPSEEK_API_KEY",),
                 langchain_prefix="deepseek",
                 openai_base_url="https://api.deepseek.com",
                 notes="Chinese-native lab; strong price/performance on Chinese text."),
    ProviderSpec("qwen", "Alibaba Qwen (DashScope)", "direct",
                 ("DASHSCOPE_API_KEY", "QWEN_API_KEY"),
                 openai_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                 alt_base_urls=("https://dashscope-intl.aliyuncs.com/compatible-mode/v1",),
                 notes="Chinese-native. Region-split API: a Singapore-issued key 401s on "
                       "the mainland host and vice versa, so both are tried."),
    ProviderSpec("moonshot", "Moonshot / Kimi", "direct", ("MOONSHOT_API_KEY",),
                 openai_base_url="https://api.moonshot.cn/v1",
                 notes="Chinese-native; long-context specialism."),
    ProviderSpec("zhipu", "Zhipu / GLM", "direct",
                 ("ZHIPU_API_KEY", "ZHIPUAI_API_KEY", "GLM_API_KEY"),
                 openai_base_url="https://open.bigmodel.cn/api/paas/v4",
                 notes="Chinese-native."),
    ProviderSpec("minimax", "MiniMax", "direct", ("MINIMAX_API_KEY",),
                 notes="Chinese-native."),
    ProviderSpec("mistral", "Mistral", "direct", ("MISTRAL_API_KEY",),
                 langchain_prefix="mistralai"),
    ProviderSpec("groq", "Groq", "aggregator", ("GROQ_API_KEY",),
                 langchain_prefix="groq",
                 openai_base_url="https://api.groq.com/openai/v1",
                 notes="Very fast inference on open-weight models."),
    ProviderSpec("together", "Together AI", "aggregator", ("TOGETHER_API_KEY",),
                 langchain_prefix="together",
                 openai_base_url="https://api.together.xyz/v1"),
    ProviderSpec("fireworks", "Fireworks", "aggregator", ("FIREWORKS_API_KEY",),
                 langchain_prefix="fireworks",
                 openai_base_url="https://api.fireworks.ai/inference/v1"),
    ProviderSpec("deepinfra", "DeepInfra", "aggregator", ("DEEPINFRA_API_KEY",),
                 openai_base_url="https://api.deepinfra.com/v1/openai"),
    ProviderSpec("openrouter", "OpenRouter", "aggregator", ("OPENROUTER_API_KEY",),
                 openai_base_url="https://openrouter.ai/api/v1",
                 models_endpoint="https://openrouter.ai/api/v1/models",
                 notes="One key reaches most major labs. Publishes a public model+price "
                       "catalogue that needs no key to read."),
    ProviderSpec("bedrock", "AWS Bedrock", "cloud",
                 ("AWS_BEARER_TOKEN_BEDROCK", "AWS_SECRET_ACCESS_KEY"),
                 langchain_prefix="bedrock_converse"),
    ProviderSpec("azure", "Azure OpenAI", "cloud",
                 ("AZURE_OPENAI_API_KEY",), langchain_prefix="azure_openai"),
    ProviderSpec("vertex", "Google Vertex AI", "cloud",
                 ("GOOGLE_APPLICATION_CREDENTIALS",), langchain_prefix="google_vertexai"),
)

BY_KEY: dict[str, ProviderSpec] = {p.key: p for p in PROVIDERS}

#: Labs whose models are trained primarily on Chinese and typically stronger per
#: dollar on Chinese text. Used as a routing hint, never as an override — the
#: catalogue's measured capabilities still decide.
CHINESE_NATIVE = frozenset({"deepseek", "qwen", "moonshot", "zhipu", "minimax"})


@dataclass
class Availability:
    """What this machine can reach, and on what evidence."""

    configured: list[str] = field(default_factory=list)
    verified: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    env_seen: dict[str, str] = field(default_factory=dict)

    @property
    def usable(self) -> list[str]:
        """Verified if anything was verified, else configured.

        Falling back to `configured` matters: probing costs a network round-trip
        per provider, and a run that only needs to *plan* should not have to pay
        it.
        """
        return self.verified or self.configured

    def summary(self) -> dict[str, Any]:
        return {
            "configured": self.configured,
            "verified": self.verified,
            "failed": self.failed,
            "chinese_native_available": sorted(set(self.usable) & CHINESE_NATIVE),
            "evidence": "probed" if self.verified else "environment variables only",
        }


def detect(env: dict[str, str] | None = None) -> Availability:
    """Which providers are configured, from environment variables alone.

    Cheap, offline, and honest about being a weaker claim than :func:`probe`.
    """
    e = env if env is not None else os.environ
    av = Availability()
    for spec in PROVIDERS:
        for var in spec.env_vars:
            if e.get(var):
                av.configured.append(spec.key)
                av.env_seen[spec.key] = var
                break
    return av


def probe(providers: list[str] | None = None, *, timeout: float = 8.0) -> Availability:
    """Actually call each configured provider's model-list endpoint.

    Upgrades "a key is present" to "a key works". Only providers publishing a
    list endpoint can be probed this way; the rest stay at configured, which is
    recorded rather than glossed.
    """
    av = detect()
    targets = providers or list(av.configured)
    for key in targets:
        spec = BY_KEY.get(key)
        if spec is None:
            continue
        url = spec.models_endpoint or (
            f"{spec.openai_base_url}/models" if spec.openai_base_url else ""
        )
        if not url:
            continue
        try:
            import urllib.request

            token = next((os.environ[v] for v in spec.env_vars if os.environ.get(v)), "")
            req = urllib.request.Request(url)
            if spec.key == "anthropic":
                req.add_header("x-api-key", token)
                req.add_header("anthropic-version", "2023-06-01")
            else:
                req.add_header("Authorization", f"Bearer {token}")
            with urllib.request.urlopen(req, timeout=timeout) as r:
                if 200 <= r.status < 300:
                    av.verified.append(key)
                else:
                    av.failed[key] = f"HTTP {r.status}"
        except Exception as exc:  # noqa: BLE001
            av.failed[key] = f"{type(exc).__name__}: {str(exc)[:80]}"
    return av


#: Resolved base URLs, cached per provider so the probe runs once per process.
_RESOLVED_BASE: dict[str, str] = {}
#: Serialises the probe. Annotator batches resolve from eight threads at once, so
#: without this every one of them probes before any of them caches.
_RESOLVE_LOCK = threading.Lock()

#: A cheap, always-present model to probe each region with.
_PROBE_MODEL: dict[str, str] = {"qwen": "qwen-turbo"}


def resolve_base_url(provider: str, *, timeout: float = 25.0) -> str:
    """The base URL whose region actually accepts this key.

    Probes with a one-token chat completion rather than ``GET /models``, because
    the model-list endpoint does not reliably discriminate: on Alibaba it can
    answer identically from either region while completions reject the key.

    Without this, a Singapore-issued key produces a bare 401 on the mainland host
    that reads exactly like a wrong credential — which is how an hour goes into
    re-copying a key that was correct all along.
    """
    if provider in _RESOLVED_BASE:
        return _RESOLVED_BASE[provider]
    with _RESOLVE_LOCK:
        if provider in _RESOLVED_BASE:      # another thread resolved while we waited
            return _RESOLVED_BASE[provider]
        return _probe_base_url(provider, timeout)


def _probe_base_url(provider: str, timeout: float) -> str:
    spec = BY_KEY.get(provider)
    if spec is None or not spec.openai_base_url:
        return ""
    token = next((os.environ[v] for v in spec.env_vars if os.environ.get(v)), "")
    if not token or not spec.alt_base_urls:
        return spec.openai_base_url

    probe_model = _PROBE_MODEL.get(provider, "")
    if not probe_model:
        return spec.openai_base_url

    try:
        import httpx
    except ImportError:
        return spec.openai_base_url

    unreachable = False
    for base in (spec.openai_base_url, *spec.alt_base_urls):
        try:
            r = httpx.post(
                f"{base}/chat/completions", timeout=timeout,
                headers={"Authorization": f"Bearer {token}"},
                json={"model": probe_model, "max_tokens": 1,
                      "messages": [{"role": "user", "content": "hi"}]},
            )
            if r.status_code != 401:
                _RESOLVED_BASE[provider] = base
                if base != spec.openai_base_url:
                    log.info("%s: key belongs to the %s region", provider, base.split("//")[1].split(".")[0])
                return base
        except Exception as exc:  # noqa: BLE001 - a network failure proves nothing
            log.debug("%s: probe of %s failed (%s)", provider, base, type(exc).__name__)
            unreachable = True
            continue
    # Nothing answered. Return the default so the run can proceed, but do NOT
    # cache it: a 401 means this host is definitively wrong, and a timeout means
    # we simply do not know. Caching either one pins every later call in the
    # process to a host we have positive evidence against — which is how one
    # transient probe failure turned into 48 AuthenticationErrors and cost an
    # annotator two thirds of its batches mid-run.
    if not unreachable:
        log.warning("%s: no configured region accepted this key; leaving it unresolved "
                    "so a later call can retry", provider)
    return spec.openai_base_url
