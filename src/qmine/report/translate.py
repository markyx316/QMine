"""Machine translation for authored prose, guarded so it cannot corrupt a number.

`i18n.PROSE_ZH` is 34 hand-written prefix mappings, and it has two structural
holes that no amount of diligence closes:

**A new authored string is English until a human notices.** Three separate
English leaks reached Chinese deliverables in one day — the p2e knn rationale,
the HDBSCAN density note, and the two annotator-balance strings — because the
mechanism has no way to know a string exists. The static AST guard added for
`deps.decision()` rationales covers one call shape out of many.

**An f-string can never be mapped.** All 22 `deps.gate()` messages interpolate
measured values, so no fixed prefix matches them, and they print verbatim into
Chinese reports. That class is unreachable by construction, which is why it had
to be frozen as debt rather than fixed.

A model closes both. What a model must never do here is change a number: this
pipeline asks readers to check its arithmetic, so a translator that renders
"kappa 0.802 on 200 queries" as "0.80" or "20" would be worse than English.

So every translation is verified before it is used:

* **numbers** — the multiset of numerals must be identical, in both directions;
* **identifiers** — every `` `backticked` `` span must survive verbatim;
* **non-empty, and actually Chinese** — a passthrough is not a translation.

Any failure returns the ENGLISH, which is exactly today's behaviour, so this
cannot make a report worse than the mechanism it extends.

Results are cached by content hash in a repository-level file. Translations are
corpus-independent, so the cache accumulates across runs and each string is paid
for once — and, more importantly, a given string renders identically on every
future run. A report that changes wording between runs for no measured reason is
its own kind of defect.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger("qmine.i18n")

CACHE_PATH = Path(".cache") / "translations.json"

#: Terms whose Chinese rendering is fixed across every report. Without this the
#: same metric acquires two names in one document, which is worse for a reader
#: than leaving it in English.
GLOSSARY: dict[str, str] = {
    "template fragmentation": "模板碎裂度",
    "replay stability": "重播稳定性",
    "silhouette": "轮廓系数",
    "gold set": "金标集",
    "leaf": "叶",
    "family": "家族",
    "annotator": "标注员",
    "referee": "裁判",
    "adjudication rule": "裁决规则",
    "phrasing group": "措辞群",
    "quality gate": "质量门",
    "held-out": "留出",
    "coverage": "覆盖率",
    "prescription": "处方",
}

_NUM = re.compile(r"-?\d+(?:\.\d+)?")
_CODE = re.compile(r"`([^`]+)`")
_CJK = re.compile(r"[一-鿿]")

_lock = threading.RLock()
_cache: dict[str, str] | None = None


def _key(text: str, language: str) -> str:
    return hashlib.sha256(f"{language}\x00{text}".encode()).hexdigest()[:16]


def _load() -> dict[str, str]:
    global _cache
    if _cache is None:
        try:
            _cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:                                    # noqa: BLE001
            _cache = {}
    return _cache


def _save() -> None:
    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = CACHE_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(_load(), ensure_ascii=False, indent=1,
                                  sort_keys=True), encoding="utf-8")
        tmp.replace(CACHE_PATH)
    except Exception as exc:                                 # noqa: BLE001
        log.debug("translation cache not saved: %s", exc)


def verify(source: str, translated: str) -> str:
    """`""` if the translation is usable, else why it is not.

    Separated from the call so the CONTRACT is testable without a model — the
    part that matters is not that a translation happened but that it preserved
    everything a reader might check.
    """
    if not translated or not translated.strip():
        return "empty translation"
    if not _CJK.search(translated):
        return "no CJK in the result — the model echoed the source"
    # Identifiers first, and STRIP them before counting numbers: a digit inside
    # `p2b_annotator_symmetry` is part of a name, not a measured value, and
    # counting it made a translated identifier report as a changed number —
    # the right complaint under the wrong heading.
    src_c, out_c = sorted(_CODE.findall(source)), sorted(_CODE.findall(translated))
    if src_c != out_c:
        return f"identifiers changed — {src_c} became {out_c}"
    bare_src, bare_out = _CODE.sub(" ", source), _CODE.sub(" ", translated)
    src_n, out_n = sorted(_NUM.findall(bare_src)), sorted(_NUM.findall(bare_out))
    if src_n != out_n:
        missing = [n for n in src_n if n not in out_n]
        added = [n for n in out_n if n not in src_n]
        return (f"numbers changed — missing {missing}, invented {added}. "
                "A translation may not alter a measured value.")
    return ""


def translate(text: str, language: str, call: Callable[[str], str] | None) -> str:
    """Translated `text`, or the ORIGINAL if anything is off. Never raises."""
    if language != "zh" or not text.strip():
        return text
    cache = _load()
    k = _key(text, language)
    with _lock:
        hit = cache.get(k)
    if hit is not None:
        return hit
    if call is None:
        return text
    try:
        out = (call(text) or "").strip()
    except Exception as exc:                                 # noqa: BLE001
        log.warning("translation call failed (%s); keeping the English", exc)
        return text
    why = verify(text, out)
    if why:
        # Loud, because a rejected translation means a reader gets English in a
        # Chinese report — the exact defect this module exists to remove.
        log.warning("translation rejected (%s); keeping the English: %.60s", why, text)
        return text
    with _lock:
        cache[k] = out
        _save()
    return out


def prompt_for(text: str) -> str:
    """The instruction sent to the model. Explicit about what may not change."""
    glossary = "\n".join(f"- {en} → {zh}" for en, zh in GLOSSARY.items())
    return (
        "把下面这段技术说明翻译成简体中文。这是一份数据科学交付物里的文字, "
        "读者会逐个核对里面的数字。\n\n"
        "硬性要求:\n"
        "1. **每一个数字必须原样保留** —— 不要改写、不要四舍五入、不要换算单位。\n"
        "2. 反引号包起来的内容 (`like_this`) 是标识符, 原样保留, 不要翻译。\n"
        "3. 不要增加原文没有的信息, 也不要省略原文有的信息。\n"
        "4. 只返回译文本身, 不要加任何解释或前言。\n\n"
        f"术语表 (必须照此翻译):\n{glossary}\n\n"
        f"原文:\n{text}"
    )


def registry_translator(registry: Any, role: str = "interpreter") -> Callable[[str], str]:
    """A `call` for `translate` backed by the run's model registry."""

    def _call(text: str) -> str:
        model = registry.get(role)
        return str(getattr(model.invoke(prompt_for(text)), "content", "")).strip()

    return _call
