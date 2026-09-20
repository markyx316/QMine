"""Guardrail copies of the health run's own reference documents (post-run operation on results).

The program's top-down and bottom-up definition documents print raw sample strings (rule examples,
template exemplars, centre/edge samples). A few of those fall under the 健康 quote block
(`EXTRA_QUOTE_BLOCK["健康"]`: minors + sexual content, explicit sexual text, named doctors) or are
named-doctor card strings; the report's hard-rule and co-occurrence layers apply too, and the run-named strings about minors or sexual content. This writes `<name>_引文护栏版.md` next to the report in `postprocessed/`
with ONLY query-like spans replaced (backtick spans, 「」 spans, table cells of <= 40 characters);
long descriptive prose (e.g. a governance rationale describing a risk) is kept. The originals in
`gen01/` are not touched. It asserts that nothing query-like still matches after redaction.
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

QM = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(QM / "analysis/pooled5"))
from p5_snapshot_classes import EXTRA_QUOTE_BLOCK, QUOTE_BLOCK, QUOTE_COOC  # noqa: E402

GEN = QM / "runs/health-pool2/gen01"
MARK = "（不引原文）"
pat = EXTRA_QUOTE_BLOCK["健康"]
allr = pd.read_parquet(QM / "analysis/pooled5/work/健康_all_rows.parquet")
doctor = set(allr.loc[allr["tier"].astype(str).isin(["S6_doctor_card_uniform", "H5_doctor_card"]), "query"].astype(str))
_rx = [x if hasattr(x, "search") else re.compile(x) for x in (QUOTE_BLOCK, *QUOTE_COOC)]
hard, cooc_a, cooc_b = _rx
# Run-named strings, PRIVACY-RELEVANT ONLY. The report's third quote layer takes every string the risk
# machinery ever named; in a reference document that removes ordinary health queries too (measured on the
# top-down definition: most of 56 redactions were folk-remedy, lead-gen, drug-interaction and
# self-diagnosis samples). Here only the sentinel findings about minors or sexual/adult content and the
# minors_and_sexual_content screen samples count.
_PRIVACY_FINDING = re.compile(r"未成年|性化|色情|成人内容|露骨|私密影像")
named: set[str] = set()
_tn = json.loads((GEN / "tree_naming.json").read_text(encoding="utf-8"))
for _f in (_tn.get("risk_report") or {}).get("findings", []) or []:
    if _PRIVACY_FINDING.search(str(_f.get("category") or "")):
        named.update(str(x).strip() for x in (_f.get("evidence") or []))
for _c in json.loads((GEN / "risk_screen.json").read_text(encoding="utf-8")).get("categories", []):
    if _c.get("name") == "minors_and_sexual_content":
        named.update(str(x).strip() for x in ([_c["exemplar"]] if _c.get("exemplar") else []) + list(_c.get("samples") or []))
named = {x for x in named if x}


def guarded(t: str) -> bool:
    """Hard rules, the co-occurrence rule, privacy-relevant run-named strings, the 健康 block."""
    return bool(pat.search(t) or hard.search(t) or (cooc_a.search(t) and cooc_b.search(t)) or t in named)


blocked = {q for q in allr["query"].astype(str) if guarded(q)} | doctor | named
long_blocked = [q for q in blocked if len(q) >= 4]


def bad(s: str) -> bool:
    t = s.strip().strip("`").strip()
    return bool(t) and (guarded(t) or t in blocked or any(q in t for q in long_blocked))


_AGE_PAREN = re.compile(r"[（(][^）)\n]*\d{1,2}\s*岁[^）)\n]*[）)]")


def redact(line: str) -> tuple[str, int]:
    n = 0

    def span(m, left, right):
        nonlocal n
        if bad(m.group(1)):
            n += 1
            return f"{left}{MARK}{right}"
        return m.group(0)

    line = re.sub(r"`([^`\n]+)`", lambda m: span(m, "`", "`"), line)
    line = re.sub(r"「([^」\n]+)」", lambda m: span(m, "「", "」"), line)
    # A descriptive line about a guarded topic (e.g. a governance rationale) quotes no query, but a
    # parenthetical listing an age and other specifics is still a description of the content. Keep the
    # sentence, drop the specifics.
    if pat.search(line):
        line, k = _AGE_PAREN.subn("（具体描述略）", line)
        n += k
    if line.lstrip().startswith("|"):
        cells = line.split("|")
        for i, c in enumerate(cells):
            if MARK not in c and 0 < len(c.strip()) <= 40 and bad(c):
                cells[i] = f" {MARK} "
                n += 1
        line = "|".join(cells)
    return line, n


def residual_query_like(line: str) -> bool:
    spans = re.findall(r"`([^`\n]+)`", line) + re.findall(r"「([^」\n]+)」", line)
    cells = [c for c in line.split("|")] if line.lstrip().startswith("|") else []
    return any(bad(s) for s in spans) or any(0 < len(c.strip()) <= 40 and bad(c) for c in cells)


out_dir = GEN / "postprocessed"
for name in ["health_zh_自上而下_意图体系完整定义.md", "health_zh_自下而上_聚类树完整定义.md", "分层对比_头尾结构差异.md"]:
    src = GEN / name
    lines = src.read_text(encoding="utf-8").split("\n")
    total, new = 0, []
    for ln in lines:
        r, n = redact(ln)
        total += n
        new.append(r)
    assert not any(residual_query_like(ln) for ln in new), f"{name}: a query-like span still matches"
    assert not any(pat.search(ln) and _AGE_PAREN.search(ln) for ln in new), f"{name}: an age-bearing description remains"
    # KNOWN PROGRAM DEFECT in the run that produced these documents: its p5_k_references_agree gate
    # hardcoded deciding_reference=phrasing_groups, but it located K by the declared reference column
    # (granularity.json triangulation.locator). Fixed in src on 2026-09-15 for NEW runs
    # (`ops/cluster.py: reference_sensitivity(..., locator_column=)` and the gate in
    # `graph/nodes/bottomup.py`); runs/health-pool2/gen01 predates the fix, so the annotation still
    # applies there. On a document from a fixed run the bad string is absent and this is a no-op.
    _loc = json.loads((GEN / "granularity.json").read_text(encoding="utf-8"))["triangulation"]
    _dec = str(_loc.get("deciding_reference") or "")
    _bad = "`deciding_reference`=phrasing_groups"
    if _dec and _dec != "phrasing_groups":
        _fix = (_bad + f"（⚠ 程序缺陷：这一栏写死为 phrasing_groups；本次实际定位 K 的参照系是 `{_dec}`，"
                f"见 granularity.json 的 triangulation.locator = `{_loc.get('locator')}`）")
        k_defect = sum(ln.count(_bad) for ln in new)
        new = [ln.replace(_bad, _fix) for ln in new]
        assert all(_bad + "（⚠" in ln for ln in new if _bad in ln)
        total += k_defect
    residual_prose = sum(1 for ln in new if pat.search(ln))
    note = (f"> **引文护栏版。** 本文件由 `{name}` 生成：命中健康域不可引护栏（未成年人与性相关、露骨性描述、具名医生卡）的 "
            f"{total} 处样例原文替换为「{MARK}」，只替换查询样例，不改任何数字与定义；原件保留在运行目录 `runs/health-pool2/gen01/`。")
    new.insert(1, "\n" + note + "\n")
    dst = out_dir / (src.stem + "_引文护栏版.md")
    dst.write_text("\n".join(new), encoding="utf-8")
    print(f"{name}: replaced {total} span(s); descriptive lines still mentioning a guarded topic: {residual_prose} -> {dst.name}")
