#!/usr/bin/env python
"""科室标注的信度：主标注（deepseek-v4-flash 全量）对三个独立读数各自一致多少。

三个读数，各测不同的东西——这一点很重要，不能混为一谈：

- **kimi-k3（换供应商，抽样 1,500）** —— 这是**信度**数字。同一个模型换几个视角只能测自洽
  （人物/影视那次实测 κ 0.99 是自洽，跨模型才掉到 0.96–0.99），所以只有换供应商的这一个算数。
- **deepseek-v4-pro（同族更大模型，前 1,000 串）** —— 测的是「换成更贵的模型会不会改判」，
  即容量差带来的上限，不是独立性。
- **规则词表（可审计，覆盖 23.5%）** —— 测的是「一个完全不看语义、只认锚点的仪器」会不会同意。
  它弃权得多，但开口的地方是任何人都能逐条复核的。

每个科室单独给一致率：整体一个数会把「肿瘤科几乎不会认错」和「全科/无法判断边界模糊」平均掉。

    HF_HOME=$(pwd)/.hf .venv/bin/python analysis/pooled5/p5_department_reliability.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

QM = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(QM / "analysis/pooled5"))
from p5_department_compare import NAME, NONCLIN  # noqa: E402

BASE = QM / "analysis/pooled5/work/科室"
OUT = BASE / "compare"


def cohen_kappa(a: pd.Series, b: pd.Series) -> float:
    cats = sorted(set(a) | set(b))
    idx = {c: i for i, c in enumerate(cats)}
    n = len(a)
    M = np.zeros((len(cats), len(cats)))
    for x, y in zip(a, b):
        M[idx[x], idx[y]] += 1
    po = float(np.trace(M) / n)
    pe = float((M.sum(0) / n * (M.sum(1) / n)).sum())
    return round((po - pe) / (1 - pe), 3) if pe < 1 else float("nan")


def pair(primary: pd.Series, other: pd.Series, tag: str) -> tuple[pd.DataFrame, dict]:
    j = pd.DataFrame({"a": primary, "b": other}).dropna()
    if not len(j):
        return pd.DataFrame(), {"n": 0, "note": f"{tag}: no overlap"}
    j["agree"] = j["a"] == j["b"]
    per = (j.groupby("a").agg(n=("agree", "size"), 一致=("agree", "sum")).reset_index())
    per["一致率%"] = (100 * per["一致"] / per["n"]).round(1)
    per["科室"] = per["a"].map(lambda k: NAME.get(k, (k,))[0])
    per["对照"] = tag
    # 最常见的分歧对：主标注说 X，对照说 Y
    dis = (j[~j["agree"]].groupby(["a", "b"]).size().sort_values(ascending=False).head(12))
    facts = {
        "n": int(len(j)), "一致率%": round(100 * float(j["agree"].mean()), 1),
        "cohen_kappa": cohen_kappa(j["a"], j["b"]),
        "临床/非临床粗分一致率%": round(100 * float(
            (j["a"].isin(NONCLIN) == j["b"].isin(NONCLIN)).mean()), 1),
        "最常见分歧": {f"{NAME.get(x, (x,))[0]}→{NAME.get(y, (y,))[0]}": int(v) for (x, y), v in dis.items()},
    }
    return per.sort_values("n", ascending=False), facts


def read(sub: str) -> pd.Series | None:
    p = BASE / sub / "labels.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p, encoding="utf-8-sig", keep_default_na=False)
    d = d[d["code"].astype(str).ne("")]
    return d.set_index("query")["code"]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    main_lab = read("deepseek")
    if main_lab is None:
        raise SystemExit("primary labels missing — run p5_department_label.py first")
    facts: dict = {"主标注": {"model": "deepseek-v4-flash", "n": int(len(main_lab))}}
    pers = []
    for sub, tag, note in (("kimi_s1500", "kimi-k3（换供应商）", "信度"),
                           ("deepseek_pro_probe", "deepseek-v4-pro（同族大模型）", "容量差"),
                           ("deepseek_s80", "deepseek-v4-pro（冒烟 80 串）", "容量差")):
        other = read(sub)
        if other is None:
            continue
        shared = other.index.intersection(main_lab.index)
        per, f = pair(main_lab.loc[shared], other.loc[shared], tag)
        f["测的是"] = note
        facts[tag] = f
        if len(per):
            pers.append(per)
        print(f"{tag}: n={f['n']:,} 一致 {f['一致率%']}% κ={f['cohen_kappa']} "
              f"（临床/非临床粗分 {f['临床/非临床粗分一致率%']}%）")
    # 规则词表
    r = pd.read_csv(BASE / "rule_labels.csv", encoding="utf-8-sig", keep_default_na=False)
    r["model"] = r["query"].astype(str).map(main_lab)
    spoke = r[~r["rule_code"].str.startswith("ABSTAIN") & r["model"].notna()]
    per, f = pair(spoke.set_index("query")["model"], spoke.set_index("query")["rule_code"], "规则词表（可审计）")
    f["测的是"] = "可复现的第二仪器"
    f["词表开口占比%"] = round(100 * len(spoke) / len(r), 1)
    facts["规则词表（可审计）"] = f
    pers.append(per)
    print(f"规则词表: n={f['n']:,} 一致 {f['一致率%']}% κ={f['cohen_kappa']} "
          f"（只覆盖 {f['词表开口占比%']}% 的串）")
    if pers:
        pd.concat(pers, ignore_index=True)[["对照", "科室", "a", "n", "一致", "一致率%"]].to_csv(
            OUT / "reliability_by_department.csv", index=False, encoding="utf-8-sig")
    (OUT / "reliability.json").write_text(json.dumps(facts, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"→ {OUT / 'reliability.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
