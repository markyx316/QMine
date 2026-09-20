# -*- coding: utf-8 -*-
"""补充风控筛查：领域档案**在运行之后**补进去的那些类，在同一份语料上是多少。

为什么需要这个文件。`screen_risk` 跑在聚类之前，正是为了让 p7 能问一句更锋利的话：**一个从没
被告知这些模式的 agent，自己找到了吗？** 2026-09-13 这两跑的答案都是找到了，而且找到的是档案
的盲区——软件的架构师给出 4 个档案里没有的风险类、哨兵再给 12 条发现；书籍文档的哨兵 6 条里
有 5 条落在盲区上，其中「自残轻生」18 行比档案原有的两类都大。

那些类**随后**被补进了 `configs/domains/*.yaml`，所以现在档案比运行当时更全。于是同一份语料
有两个风控数字，报告必须把它们分开写：

  * 运行当时的：`runs/<id>/gen01/risk_screen.json`，是那一跑真正用过的；
  * 补全后的：拿现在的档案重跑一遍 `screen_risk`。

这个脚本算的就是两者之差，并列出差在哪几类。它不重复任何正则——模式只有档案一处。

    python analysis/pooled5/p5_new2_risk_supplement.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from qmine.config import DomainProfile          # noqa: E402
from qmine.ops.audit import screen_risk         # noqa: E402
from pooled5_common import SOURCES, SRC_ZH, available, cross_dir, load, run_dir  # noqa: E402

#: 领域 -> 该领域档案的文件名（档案 key 与 pooled5 的中文域名不是一回事）。
PROFILE = {"书籍文档": "books_docs_zh", "软件": "software_apps_zh"}


def compare(domain: str) -> pd.DataFrame:
    prof = PROFILE.get(domain)
    if not prof:
        return pd.DataFrame()
    P = DomainProfile.load(Path(__file__).resolve().parents[2] / f"configs/domains/{prof}.yaml")
    d = load(domain).reset_index(drop=True)
    srcs = [s for s in SOURCES if (d.source == s).any()]
    now = screen_risk(d, P.risk_categories, text_col="query")
    then = json.loads((run_dir(domain) / "risk_screen.json").read_text(encoding="utf-8"))
    then_names = {c["name"] for c in then["categories"]}
    then_idx = {i for i in then.get("flag_mask_indices", []) if 0 <= i < len(d)}
    rows = []
    for cat, c in zip(P.risk_categories, now["categories"]):
        import numpy as np
        from qmine.ops.audit import _noncapturing
        q = d["query"].astype(str)
        m = np.zeros(len(d), bool)
        for pat in cat.patterns:
            m |= q.str.contains(_noncapturing(pat), regex=True, na=False).to_numpy()
        hit = set(np.where(m)[0].tolist())
        r = {"领域": domain, "类目": c["name"],
             "运行当时有没有": "有" if c["name"] in then_names else "**补在运行之后**",
             "行数": c["n_hits"], "占比%": round(100 * c["share"], 3),
             "policy": c["policy"],
             "其中运行自带筛查已覆盖": len(hit & then_idx),
             "运行自带筛查漏掉的": len(hit - then_idx)}
        for s in srcs:
            sel = (d.source == s).to_numpy()
            r[f"{SRC_ZH[s]}_条数"] = int(m[sel].sum())
        rows.append(r)
    t = pd.DataFrame(rows)
    print(f"{domain}: 运行当时 {then['total_flagged']} 行 ({100 * then['total_share']:.2f}%) → "
          f"补全后 {now['total_flagged']} 行 ({100 * now['total_share']:.2f}%) "
          f"，多出 {now['total_flagged'] - then['total_flagged']} 行")
    return t


if __name__ == "__main__":
    doms = [d for d in (sys.argv[1:] or available()) if d in PROFILE]
    if not doms:
        raise SystemExit(f"没有可比的领域（本脚本只覆盖 {sorted(PROFILE)}）")
    out = pd.concat([compare(d) for d in doms], ignore_index=True)
    f = cross_dir() / "risk_screen_then_vs_now.csv"
    out.to_csv(f, index=False, encoding="utf-8-sig")
    cols = ["领域", "类目", "运行当时有没有", "行数", "占比%", "运行自带筛查漏掉的"]
    print()
    print(out[cols].to_string(index=False))
    print("\nwrote", f)
