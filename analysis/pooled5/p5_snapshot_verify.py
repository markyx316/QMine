# -*- coding: utf-8 -*-
"""叙述段落的机械复核：叙述里出现的每个数字都要能在表里找到，每个「」引文都要是真实且可引的行。

这不是「正确性」检查——一个数字对得上，不代表它被用对了地方。它拦的是**凭空写出来的数字**和
**编出来的例子**，那两样是历次运行里最贵的缺陷。剩下的（把 A 类的数安到 B 类头上、把相关读成
因果）只有复核员逐条重算能拦，机械检查拦不住，所以这个脚本从不给「通过」的结论，只给未匹配清单。

    python analysis/pooled5/p5_snapshot_verify.py [domain ...]
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pooled5_common import DOMAINS, WORK, available, load, run_dir
from p5_snapshot_classes import QUOTE_BLOCK, QUOTE_COOC, _risk_indices, extra_quote_mask, risk_strings

ROOT = Path(__file__).resolve().parents[2]
YEARS = {"2025", "2026"}          # 快照的名字，不是测量值
NUM = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")
QUOTE = re.compile(r"「([^」]{1,80})」")
NARRBLOCK = re.compile(r"<!--NARR:([^>]+)-->(.*?)<!--/NARR:\1-->", re.S)


def _pool(domain: str) -> set[str]:
    """表里出现过的每一个数，按几种常见写法各存一遍。"""
    out: set[str] = set()
    p = WORK / domain / "snapshot_classes"
    files = list(p.glob("*.csv"))
    d = load(domain)
    for f in files:
        t = pd.read_csv(f)
        for c in t.columns:
            s = pd.to_numeric(t[c], errors="coerce").dropna()
            for v in s:
                out.update(_forms(float(v)))
    # summary.json 里的数（可引行数、命名叶数、风控命中数……）。**这一层原本漏了**，结果两位
    # 修订员都被迫绕开正确值去引别的数字：`quotable_rows=21763` 不在池子里会被判未匹配，而写错的
    # 21,800 恰好能在别的表里匹配上。机械检查把对的拦下、把错的放过，比没有检查更坏。
    sp = WORK / domain / "snapshot_classes" / "summary.json"
    if sp.exists():
        def walk(o):
            if isinstance(o, dict):
                for v in o.values():
                    walk(v)
            elif isinstance(o, list):
                for v in o:
                    walk(v)
            elif isinstance(o, (int, float)) and not isinstance(o, bool):
                out.update(_forms(float(o)))
        walk(json.loads(sp.read_text(encoding="utf-8")))
    # 行数、类目数一类的整数
    for v in [len(d), d["td_l1"].nunique(), d["bu_leaf"].nunique(), d["td_l2"].nunique(),
              d["bu_family_final"].nunique()] + list(d.source.value_counts()):
        out.update(_forms(float(v)))
    # 护栏本身的三层条数与差额：报告正文引用它们，但它们不在任何 CSV 里
    risk = _risk_indices(domain, len(d))
    q = d["query"].astype(str)
    m2 = q.str.contains(QUOTE_BLOCK)
    m3 = q.str.contains(QUOTE_COOC[0]) & q.str.contains(QUOTE_COOC[1])
    named = risk_strings(domain)
    m4 = q.isin(named)
    m5 = extra_quote_mask(domain, q)
    blocked = d.index.isin(risk) | m2 | m3 | m4 | m5
    for v in [len(risk), int(m2.sum()), int(m3.sum()), int(m4.sum()), int(blocked.sum()),
              len(d) - int(blocked.sum()), len(named)]:
        out.update(_forms(float(v)))
    for s_, g in d.groupby("source"):
        out.update(_forms(float(int(blocked[g.index].sum()))))
    for s, g in d.groupby("source"):
        out.update(_forms(float(len(g))))
    return out


def _forms(v: float) -> set[str]:
    out = set()
    for nd in (0, 1, 2, 3, 4):
        out.add(f"{round(v, nd):.{nd}f}".rstrip("0").rstrip(".") if nd else f"{v:.0f}")
        out.add(f"{round(v, nd):.{nd}f}")
        out.add(f"{round(100 * v, nd):.{nd}f}")          # 0.1234 写成 12.34
        out.add(f"{round(v / 100, nd):.{nd}f}")
    if abs(v) >= 1000:
        out.add(f"{v:,.0f}")
    return {x.lstrip("+") for x in out}


def check(domain: str, path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    blocks = NARRBLOCK.findall(text)
    pool = _pool(domain)
    d = load(domain).reset_index(drop=True)
    risk = _risk_indices(domain, len(d))
    named = risk_strings(domain)
    q = d["query"].astype(str)
    cooc = q.str.contains(QUOTE_COOC[0]) & q.str.contains(QUOTE_COOC[1])
    quotable = set(d.loc[~d.index.isin(risk) & ~q.str.contains(QUOTE_BLOCK) & ~cooc & ~q.isin(named)
                         & ~extra_quote_mask(domain, q),
                         "query"].astype(str))
    every = set(d["query"].astype(str))
    lines = [f"# {domain} 叙述段落机械复核", "",
             f"报告：`{path.resolve().relative_to(ROOT)}`　叙述块 {len(blocks)} 个，"
             f"合计 {sum(len(b[1]) for b in blocks):,} 字符。", "",
             "本检查只拦两件事：叙述里凭空出现的数字、不存在或不可引的例子。它**不**判断数字有没有被用对地方。",
             "",
             "**「」只用于引用语料里的真实 query。** 叙述里要强调一个词组时用“”，用「」会被这里当成引文回查。", ""]
    bad_n = bad_q = 0
    for name, body in blocks:
        nums = [n for n in NUM.findall(body)]
        miss = [n for n in nums if n not in YEARS
                and n.replace(",", "").lstrip("-") not in pool
                and n.replace(",", "") not in pool and n not in pool]
        qs = QUOTE.findall(body)
        notreal = [q for q in qs if q not in every]
        notok = [q for q in qs if q in every and q not in quotable]
        bad_n += len(miss)
        bad_q += len(notreal) + len(notok)
        lines.append(f"## {name}")
        lines.append(f"- 数字 {len(nums)} 个，未在表中找到 **{len(miss)}** 个"
                     + (f"：`{'`, `'.join(miss[:40])}`" if miss else ""))
        lines.append(f"- 引文 {len(qs)} 个，语料里不存在 **{len(notreal)}** 个"
                     + (f"：{notreal[:10]}" if notreal else "")
                     + f"；存在但命中引用护栏 **{len(notok)}** 个"
                     + (f"：{notok[:10]}" if notok else ""))
        lines.append("")
    lines.insert(4, f"**未匹配数字合计 {bad_n} 个；有问题的引文合计 {bad_q} 个。**"
                    + ("" if bad_n + bad_q == 0 else
                       " 未匹配的数字要么是笔误，要么是从表里算出来的派生量——派生量必须在正文里写明怎么算的，"
                       "否则删掉。") + "\n")
    out = WORK / domain / "snapshot_classes" / "verify.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"{domain}: 未匹配数字 {bad_n} · 问题引文 {bad_q} → {out.relative_to(ROOT)}")
    return "\n".join(lines)


if __name__ == "__main__":
    for dom in (sys.argv[1:] or available()):
        gen = run_dir(dom)
        books = sorted(gen.glob("*_query_挖掘结果.xlsx"))
        stem = books[0].stem.replace("_query_挖掘结果", "") if books else DOMAINS[dom]
        check(dom, gen / "postprocessed" / f"{stem}_意图与聚类叶_跨快照对比.zh.md")
