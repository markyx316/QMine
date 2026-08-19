"""Web research tools for the taxonomy researchers.

Phase 2a asks agents to ground a taxonomy in published work — e-commerce query
taxonomies, intent-discovery literature, competitor category trees. Without
these tools that instruction is theatre: the agent answers from parametric
memory, cites papers it half-remembers, and the "evidence" in the design record
is unverifiable.

Two tools, both returning *compact* results, because a research agent that
receives 20,000 characters of page text spends its context on boilerplate
instead of judgment.

**Search backends, best-available.** Tavily, Brave and Exa are better when a key
exists, but the default has to work with no key at all or the feature is only
available to users who already solved it. DuckDuckGo needs none.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

from langchain_core.tools import tool

log = logging.getLogger("qmine.tools.web")

_MAX_SNIPPET = 400
_MAX_PAGE_CHARS = 6000


def _search_tavily(query: str, k: int) -> list[dict[str, str]] | None:
    key = os.environ.get("TAVILY_API_KEY")
    if not key:
        return None
    try:
        import httpx

        r = httpx.post("https://api.tavily.com/search", timeout=25, json={
            "api_key": key, "query": query, "max_results": k,
            "search_depth": "advanced", "include_answer": False,
        })
        r.raise_for_status()
        return [{"title": x.get("title", ""), "url": x.get("url", ""),
                 "snippet": (x.get("content") or "")[:_MAX_SNIPPET]}
                for x in r.json().get("results", [])]
    except Exception as exc:  # noqa: BLE001
        log.debug("tavily failed: %s", exc)
        return None


def _search_brave(query: str, k: int) -> list[dict[str, str]] | None:
    key = os.environ.get("BRAVE_API_KEY") or os.environ.get("BRAVE_SEARCH_API_KEY")
    if not key:
        return None
    try:
        import httpx

        r = httpx.get("https://api.search.brave.com/res/v1/web/search",
                      params={"q": query, "count": k}, timeout=25,
                      headers={"X-Subscription-Token": key, "Accept": "application/json"})
        r.raise_for_status()
        return [{"title": x.get("title", ""), "url": x.get("url", ""),
                 "snippet": re.sub(r"<[^>]+>", "", x.get("description", ""))[:_MAX_SNIPPET]}
                for x in r.json().get("web", {}).get("results", [])]
    except Exception as exc:  # noqa: BLE001
        log.debug("brave failed: %s", exc)
        return None


def _search_ddg(query: str, k: int) -> list[dict[str, str]] | None:
    """The keyless default. Slower and noisier than the paid APIs, and present."""
    try:
        from ddgs import DDGS

        return [{"title": x.get("title", ""), "url": x.get("href", ""),
                 "snippet": (x.get("body") or "")[:_MAX_SNIPPET]}
                for x in DDGS().text(query, max_results=k)]
    except Exception as exc:  # noqa: BLE001
        log.debug("ddg failed: %s", exc)
        return None


def search_web(query: str, k: int = 6) -> dict[str, Any]:
    """Search, trying the best backend available and degrading in order."""
    for name, fn in (("tavily", _search_tavily), ("brave", _search_brave), ("duckduckgo", _search_ddg)):
        res = fn(query, k)
        if res:
            return {"backend": name, "query": query, "results": res}
    return {
        "backend": "none", "query": query, "results": [],
        "note": ("No search backend reachable — offline, or every provider failed. "
                 "Say so in your findings rather than answering from memory."),
    }


def fetch_page(url: str, max_chars: int = _MAX_PAGE_CHARS) -> dict[str, Any]:
    """Fetch a URL and return readable text, truncated to a context budget."""
    try:
        import httpx

        r = httpx.get(url, timeout=25, follow_redirects=True,
                      headers={"User-Agent": "Mozilla/5.0 (compatible; QMine research agent)"})
        r.raise_for_status()
        html = r.text
    except Exception as exc:  # noqa: BLE001
        return {"url": url, "ok": False, "error": f"{type(exc).__name__}: {str(exc)[:140]}"}

    text = re.sub(r"(?is)<(script|style|nav|footer|header)[^>]*>.*?</\1>", " ", html)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;?", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"\s+", " ", text).strip()
    truncated = len(text) > max_chars
    return {
        "url": url, "ok": True, "chars": len(text), "truncated": truncated,
        "text": text[:max_chars] + ("… [truncated]" if truncated else ""),
    }


# --------------------------------------------------------------------------
# LangChain tool wrappers
# --------------------------------------------------------------------------

@tool
def web_search(query: str) -> str:
    """Search the web for published work, industry practice, or competitor taxonomies.

    Use precise queries. "query intent taxonomy e-commerce SIGIR" finds a paper;
    "search queries" finds nothing useful. Returns titles, URLs and snippets —
    call fetch_url on anything worth reading properly.
    """
    res = search_web(query)
    if not res["results"]:
        return res.get("note", "no results")
    lines = [f"[{res['backend']}] {len(res['results'])} results for {query!r}:"]
    for i, r in enumerate(res["results"], 1):
        lines.append(f"{i}. {r['title']}\n   {r['url']}\n   {r['snippet']}")
    return "\n".join(lines)


@tool
def fetch_url(url: str) -> str:
    """Fetch a page and return its readable text, truncated to a context budget.

    Use after web_search to actually read a promising source. Quote what you find
    with its URL — an unattributed claim is indistinguishable from a guess.
    """
    res = fetch_page(url)
    if not res.get("ok"):
        return f"could not fetch {url}: {res.get('error')}"
    return f"{url} ({res['chars']} chars{', truncated' if res['truncated'] else ''}):\n\n{res['text']}"


RESEARCH_TOOLS = [web_search, fetch_url]


def tools_available() -> dict[str, Any]:
    """What research backends this machine can reach.  Reported in the manifest."""
    backends = []
    if os.environ.get("TAVILY_API_KEY"):
        backends.append("tavily")
    if os.environ.get("BRAVE_API_KEY") or os.environ.get("BRAVE_SEARCH_API_KEY"):
        backends.append("brave")
    try:
        import ddgs  # noqa: F401

        backends.append("duckduckgo")
    except ImportError:
        pass
    return {
        "search_backends": backends,
        "web_research_enabled": bool(backends),
        "note": ("Researchers can cite live sources."
                 if backends else
                 "No search backend — researchers will work from the log slice only, and "
                 "the literature angle degrades to unverifiable recall. Install `ddgs` or "
                 "set TAVILY_API_KEY."),
    }
