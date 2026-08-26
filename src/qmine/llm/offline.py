"""The offline model: how this pipeline stays runnable, testable, and honest.

Three things need to be true at once.  The pipeline must run end-to-end in CI
with no API key.  Tests must be deterministic.  And an offline run must not
quietly pretend to be an LLM run.

:class:`OfflineHeuristicModel` satisfies all three.  It answers structured-output
requests by *actually computing something* — cluster names come from the card's
own top n-grams, labels come from regex evidence in the prompt — and every
record it produces is stamped ``offline-heuristic`` so no reader can mistake its
output for model judgment.  Where no heuristic exists it falls back to
synthesising a schema-valid instance seeded by the prompt hash, which keeps the
graph flowing without inventing plausible-looking prose.
"""

from __future__ import annotations

import hashlib
import json
import re
from types import UnionType
from typing import Any, Sequence, Union, get_args, get_origin

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import BaseModel

_CJK = re.compile(r"[一-鿿]")

#: Closed-class English words: present in nearly every query, distinctive of none.
_STOPWORDS = frozenset(
    "a an the is are was were be been do does did to of for from in on at by with "
    "and or but if then than that this these those it its my your how what where "
    "when which who whom why can could should would will i you me we they".split()
)


def _seed_of(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def _pick(seq: Sequence[Any], seed: int) -> Any:
    return seq[seed % len(seq)] if seq else None


class OfflineHeuristicModel(BaseChatModel):
    """A deterministic stand-in for a chat model.

    It is *not* a language model and does not pretend to be one.  It exists so
    that the twelve-phase graph — every node, every gate, every artifact write —
    can be exercised without network access, and so that a test asserting
    "Phase 7 produced a naming for every leaf" is a test of our wiring rather
    than of a remote service's mood.
    """

    model_name: str = "offline-heuristic"
    tag: str = "offline-heuristic"

    @property
    def _llm_type(self) -> str:
        return "qmine-offline-heuristic"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        system = "\n".join(
            str(getattr(m, "content", "")) for m in messages if m.type == "system"
        )
        prompt = "\n".join(
            str(getattr(m, "content", "")) for m in messages if m.type != "system"
        )
        schema = kwargs.get("qmine_schema")
        payload = synthesize(prompt, schema, tag=self.tag, system=system)
        text = json.dumps(payload, ensure_ascii=False) if not isinstance(payload, str) else payload
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])


# --------------------------------------------------------------------------
# Heuristic synthesisers
# --------------------------------------------------------------------------

def top_terms(samples: Sequence[str], k: int = 4, min_len: int = 2) -> list[str]:
    """Terms shared by the samples — a crude but real signal, in any script.

    Script matters here. Chinese has no spaces, so shared *character n-grams* are
    the unit that carries a phrasing family ("的拼音"). English does have spaces,
    and its character n-grams are mostly noise ("ing", "the"), while shared
    *words* are exactly the signal. An earlier version only counted CJK n-grams,
    so every English cluster came back with a placeholder name and the offline
    mode was quietly Chinese-only.
    """
    if not samples:
        return []
    joined = " ".join(samples)
    cjk_ratio = len(_CJK.findall(joined)) / max(len(joined), 1)
    counts: dict[str, int] = {}

    if cjk_ratio >= 0.2:
        for s in samples:
            seen = set()
            for n in (4, 3, 2):
                for i in range(len(s) - n + 1):
                    g = s[i : i + n]
                    if len(g) >= min_len and _CJK.search(g) and g not in seen:
                        seen.add(g)
                        counts[g] = counts.get(g, 0) + 1
    else:
        # Word unigrams and bigrams, minus the closed-class words that appear in
        # everything and identify nothing.
        for s in samples:
            toks = [t for t in re.findall(r"[a-z0-9]+", s.lower()) if t not in _STOPWORDS]
            seen = set()
            for gram in toks + [" ".join(p) for p in zip(toks, toks[1:])]:
                if len(gram) >= 3 and gram not in seen:
                    seen.add(gram)
                    counts[gram] = counts.get(gram, 0) + 1

    ranked = sorted(counts.items(), key=lambda kv: (-kv[1] * len(kv[0]), kv[0]))
    out: list[str] = []
    for term, c in ranked:
        if c < max(2, len(samples) // 6):
            continue
        if any(term in o or o in term for o in out):
            continue
        out.append(term)
        if len(out) >= k:
            break
    return out


def _extract_samples(prompt: str) -> list[str]:
    """Pull data rows out of a rendered card or query list.

    Two shapes appear in this pipeline: bulleted member lines from a naming card,
    and numbered lines from an annotation batch. Anything long enough to be prose
    is dropped — a real query in these corpora is short, and a 200-character
    bullet is an instruction that wandered in.
    """
    rows = [
        m.group(1).strip()
        for m in re.finditer(r"^\s*(?:[-*]|\d+\.)\s+(.{1,120})$", prompt, re.M)
    ]
    return [r for r in rows if r and len(r) <= 60 and not r.endswith((":", "："))]


def synthesize(
    prompt: str, schema: Any = None, *, tag: str = "offline-heuristic", system: str = ""
) -> Any:
    """Produce a schema-valid payload for ``prompt``.

    Deterministic in ``(system, prompt)``: the same request always yields the
    same answer, which is what makes offline runs reproducible and tests stable.

    ``system`` is hashed but never mined for content. Role prompts are written in
    markdown and full of bullet lists, so treating them as data would drown the
    handful of real sample queries in instructional prose — and the heuristics
    below are only worth anything if they see the data.
    """
    seed = _seed_of(system + "\x00" + prompt)
    if schema is None:
        return f"[{tag}] no schema supplied; prompt hash {seed:08x}"

    fields = _schema_fields(schema)
    samples = _extract_samples(prompt)
    terms = top_terms(samples, k=8) if samples else []
    return {
        name: _fill_field(name, spec, seed, terms, samples, tag)
        for name, spec in fields.items()
    }


def _schema_fields(schema: Any) -> dict[str, Any]:
    if hasattr(schema, "model_fields"):
        return {
            n: {
                "annotation": f.annotation,
                "default": f.default,
                "required": f.is_required(),
                "description": f.description or "",
            }
            for n, f in schema.model_fields.items()
        }
    if isinstance(schema, dict) and "properties" in schema:
        req = set(schema.get("required", []))
        return {
            n: {"annotation": p.get("type", "string"), "default": None,
                "required": n in req, "description": p.get("description", ""), "json": p}
            for n, p in schema["properties"].items()
        }
    return {}


def _unwrap_optional(ann: Any) -> Any:
    """Strip ``| None`` so ``list[X] | None`` is treated as ``list[X]``."""
    origin = get_origin(ann)
    if origin in (Union, UnionType):
        inner = [a for a in get_args(ann) if a is not type(None)]
        if len(inner) == 1:
            return inner[0]
    return ann


def _is_model(ann: Any) -> bool:
    return isinstance(ann, type) and issubclass(ann, BaseModel)


def _fill_model(model: type[BaseModel], seed: int, terms: list[str], samples: list[str], tag: str, index: int = 0) -> dict[str, Any]:
    """Synthesise one valid instance of a nested model.

    ``index`` shifts which mined term this instance is built around, so a list of
    three candidates describes three different things rather than the same thing
    three times.
    """
    rotated = terms[index:] + terms[:index] if terms else []
    return {
        name: _fill_field(name, spec, seed + index * 7919, rotated, samples, tag, depth=1)
        for name, spec in _schema_fields(model).items()
    }


def _fill_field(
    name: str, spec: dict[str, Any], seed: int, terms: list[str], samples: list[str],
    tag: str, depth: int = 0,
) -> Any:
    ann = _unwrap_optional(spec.get("annotation"))
    ann_s = str(ann)
    low = name.lower()
    origin = get_origin(ann)

    # -- nested models and lists of them ------------------------------------
    if depth < 3:
        if _is_model(ann):
            return _fill_model(ann, seed, terms, samples, tag)
        if origin in (list, tuple) or "list[" in ann_s.lower():
            args = get_args(ann)
            inner = _unwrap_optional(args[0]) if args else str
            if _is_model(inner):
                # A model carrying a ``query`` field is a per-row verdict: one
                # entry per input row, or the batch silently drops rows and the
                # agreement statistics downstream are computed over a hole.
                if "query" in getattr(inner, "model_fields", {}) and samples:
                    return [
                        _fill_model(inner, seed, terms, samples, tag, index=i) | {"query": q}
                        for i, q in enumerate(samples)
                    ]
                n = min(3, max(1, len(terms) or 1))
                return [_fill_model(inner, seed, terms, samples, tag, index=i) for i in range(n)]
            if get_origin(inner) is dict or (isinstance(inner, type) and issubclass(inner, dict)):
                return []
            if inner in (int, float):
                return [seed % 5, (seed // 3) % 5]
            return (terms[:3] or [f"[{tag}] {name}"])
        if origin is dict or "dict[" in ann_s.lower():
            return {}

    # -- semantically meaningful fields get a real, if simple, computation ---
    if low in ("name_zh", "name", "display_name", "title"):
        if terms:
            return f"{terms[0]}相关查询" if _CJK.search(terms[0]) else f"{terms[0]} lookup"
        # With no terms this used to fall through to the generic branch and echo
        # the SCHEMA FIELD NAME — `[offline-heuristic] name_zh` — which then
        # shipped as a family heading and failed the report-language check on
        # every offline run. A harness that fails for a reason nobody will fix
        # gets ignored, and this one guards a real defect.
        return f"[{tag}] 未命名分组"
    if low == "code" and terms:
        return "grp_" + hashlib.sha1(terms[0].encode()).hexdigest()[:6]
    if low in ("user_need", "definition", "rationale", "summary", "lesson", "defect", "fix",
               "attack", "design_notes", "labeling_guide", "audit_notes", "when", "then"):
        # Written in the language of the data. A definition sentence in a
        # different script from the corpus cannot be checked against the corpus
        # by the people who own it.
        head = terms[0] if terms else ""
        if head and _CJK.search(head):
            return f"[{tag}] 与「{head}」相关的查询; 用户得到对应的直接答案即满足。"
        if head:
            return (f"[{tag}] the user asks something about \u201c{head}\u201d and is satisfied "
                    "on receiving the corresponding direct answer.")
        return f"[{tag}] no distinctive term found in this group's members."
    if low in ("coherence", "score", "rating", "severity_score"):
        return 3 + (seed % 3)
    if low in ("label", "final_label", "better_label", "assigned_label") and not _enum_options(spec):
        return _label_from_terms(terms, seed)
    if low == "query" and samples:
        return samples[seed % len(samples)]
    if low.endswith("_flag") or low.startswith("is_") or low.startswith("has_") or low in ("risk", "clean", "pragmatic"):
        return bool(seed % 7 == 0)
    if low in ("mix_notes", "risk_reason", "decline_reason", "proposed_rule", "evidence_query", "angle"):
        return ""

    # -- otherwise: a valid instance of the declared type -------------------
    opts = _enum_options(spec)
    if opts:
        return _pick(opts, seed)
    if ann is bool or "bool" in ann_s:
        return bool(seed % 2)
    if ann is int or "int" in ann_s:
        return seed % 5 + 1
    if ann is float or "float" in ann_s:
        return round((seed % 100) / 100, 3)
    if not spec.get("required", True) and spec.get("default") not in (None, ...):
        return spec["default"]
    return f"[{tag}] {name}"


def _label_from_terms(terms: list[str], seed: int) -> str:
    if not terms:
        return "OTHER"
    t = terms[seed % len(terms)]
    return "T_" + hashlib.sha1(t.encode()).hexdigest()[:6].upper()


def _enum_options(spec: dict[str, Any]) -> list[str]:
    ann_s = str(spec.get("annotation", ""))
    m = re.findall(r"Literal\[(.*?)\]", ann_s)
    if m:
        return [x.strip().strip("'\"") for x in m[0].split(",")]
    js = spec.get("json") or {}
    if isinstance(js, dict) and js.get("enum"):
        return [str(x) for x in js["enum"]]
    return []
