"""Mechanical check on authored paragraphs: every number, every quotation.

THIS NEVER EMITS A PASS. A number that matches a table is not thereby a number
used correctly — attributing class A's figure to class B matches, and so does
reading a correlation as a cause. Those need a person recomputing. What this
catches is the two failures that have been most expensive here: a figure written
from memory, and an example that was never in the corpus.

Only text inside a `<!--NARR:key-->…<!--/NARR:key-->` block is examined. Numbers
outside those blocks were rendered from a table by the renderer and are trusted
by construction — and that is exactly why the renderer is not allowed to compute
anything of its own.

THE POOL MUST CONTAIN EVERY TRUE NUMBER OR THE CHECK INVERTS. It once omitted
`summary.json`, so a correct `quotable_rows=21763` was reported as unmatched
while a wrong 21,800 happened to match a different table. Two revisers were
pushed off the right number by the check that existed to protect it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

NARR_BLOCK = re.compile(r"<!--NARR:([^>]+)-->(.*?)<!--/NARR:\1-->", re.S)
NUM = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")
#: BOTH quotation forms, and NO length cap. `「([^」]{1,80})」` silently ignored
#: every quotation over eighty characters and every one written `『…』` — which
#: is exactly the form the renderer uses for a query that itself contains 「」,
#: so the fabricated examples most likely to be long or odd were the ones the
#: check skipped.
QUOTE = re.compile(r"『([^』\n]+)』|「([^「」\n]+)」")


def forms(v: float) -> set[str]:
    """Every spelling a reader might reasonably write one table value as."""
    out: set[str] = set()
    for nd in range(5):
        r = round(v, nd)
        out.add(f"{r:.{nd}f}")
        if nd:
            out.add(f"{r:.{nd}f}".rstrip("0").rstrip("."))
        else:
            out.add(f"{v:.0f}")
        out.add(f"{round(100 * v, nd):.{nd}f}")
        out.add(f"{round(v / 100, nd):.{nd}f}")
    if abs(v) >= 1000:
        out.add(f"{v:,.0f}")
    return {x.lstrip("+") for x in out}


def _walk(obj: Any, sink: set[str]) -> None:
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        sink.update(forms(float(obj)))
    elif isinstance(obj, dict):
        for v in obj.values():
            _walk(v, sink)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _walk(v, sink)


def number_pool(table_dir: Path, extra: Iterable[float] = ()) -> set[str]:
    """Every number that appears anywhere in the tables, in every spelling.

    `extra` exists because the renderer computes a few figures the tables do not
    hold — a snapshot-size ratio, a detectability bound at n_min and n_max. A
    pool missing them reports a TRUE figure as unmatched, and a check that
    rejects the truth is worse than no check: it once pushed two revisers off
    the correct `quotable_rows=21763` onto a wrong 21,800 that happened to match
    another table.
    """
    pool: set[str] = set()
    files = sorted(table_dir.glob("*.csv"))
    assert files, (f"{table_dir} holds no CSV — the pool would be empty and every "
                   "number in the prose would read as unmatched. Build the tables first.")
    for f in files:
        t = pd.read_csv(f)
        for c in t.columns:
            for v in pd.to_numeric(t[c], errors="coerce").dropna():
                pool.update(forms(float(v)))
    sp = table_dir / "summary.json"
    if sp.exists():
        _walk(json.loads(sp.read_text(encoding="utf-8")), pool)
    for v in extra:
        pool.update(forms(float(v)))
    return pool


def check(report_text: str, table_dir: Path, *, corpus: set[str],
          quotable: set[str], year_like: Iterable[str] = (),
          extra_numbers: Iterable[float] = ()) -> dict[str, Any]:
    """Return the unmatched numbers and the bad quotations, per narrative block."""
    pool = number_pool(table_dir, extra=extra_numbers)
    years = set(year_like)
    blocks = NARR_BLOCK.findall(report_text)
    out: dict[str, Any] = {"blocks": [], "未匹配数字": 0, "问题引文": 0,
                           "叙述字数": 0, "块数": len(blocks)}
    for key, body in blocks:
        body = body.strip()
        nums = NUM.findall(body)
        miss = [n for n in nums
                if n not in years
                and n.replace(",", "").lstrip("-") not in pool
                and n.replace(",", "") not in pool
                and n not in pool]
        # `findall` on an alternation yields a tuple per match; the non-empty
        # group is the quotation. 『…』 must be allowed to contain 「」, because
        # that is precisely why the renderer switches to it.
        qs = [a or b for a, b in QUOTE.findall(body)]
        notreal = [q for q in qs if q not in corpus]
        notok = [q for q in qs if q in corpus and q not in quotable]
        out["blocks"].append({
            "key": key, "字数": len(body), "数字个数": len(nums),
            "未匹配数字": miss, "引文个数": len(qs),
            "语料里不存在": notreal, "命中引用护栏": notok})
        out["未匹配数字"] += len(miss)
        out["问题引文"] += len(notreal) + len(notok)
        out["叙述字数"] += len(body)
    return out


def render_verification(result: dict[str, Any], report_path: Path) -> str:
    """The verification note that ships beside the report."""
    lines = [
        "# 叙述段落的机械复核",
        "",
        f"报告：`{report_path.name}`　·　叙述块 {result['块数']} 个　·　"
        f"合计 {result['叙述字数']:,} 字",
        "",
        f"**未匹配数字合计 {result['未匹配数字']} 个；有问题的引文合计 {result['问题引文']} 个。**",
        "",
        "> 这不是「正确性」检查。一个数字对得上，不代表它被用对了地方——把 A 类的数安到 B 类头上、"
        "把相关读成因果，机械检查都拦不住，只有复核员逐条重算能拦。所以这份文件**从不给「通过」的结论**，"
        "只给未匹配清单。",
        "> 「」只用于真实 query，这样每一条引文都能回查到语料。",
        "",
    ]
    for b in result["blocks"]:
        lines.append(f"## {b['key']}")
        miss = b["未匹配数字"]
        lines.append(f"- 数字 {b['数字个数']} 个，未在表中找到 **{len(miss)}** 个"
                     + (f"：{', '.join('`' + m + '`' for m in miss[:40])}" if miss else ""))
        nr, no = b["语料里不存在"], b["命中引用护栏"]
        lines.append(
            f"- 引文 {b['引文个数']} 个，语料里不存在 **{len(nr)}** 个"
            + (f"（{', '.join(nr[:10])}）" if nr else "")
            + f"；存在但命中引用护栏 **{len(no)}** 个"
            + (f"（{', '.join(no[:10])}）" if no else ""))
        lines.append("")
    return "\n".join(lines)
