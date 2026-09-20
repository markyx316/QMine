#!/usr/bin/env python
"""叶名在某个意图里还适用吗：意图 × 叶 的「名实」测量（结果后处理，不改程序源码）。

叶是全语料上的话题簇，叶名描述的是整叶的多数；一个意图落在这个叶里的那部分行，可能只是叶的少数，
说的也可能是另一回事（例：「消息公告」意图落在「查询保险公司信息」叶里的行，是上市公司资料）。
所以用叶来图解意图时，要同时给出：
- 该意图占叶%：这个叶的行里有多少属于该意图（低 = 叶名主要在描述别的意图的行）；
- 叶主导意图与其占叶%；
- cos(意图内中心, 叶其余中心)：该意图在叶里的行，与叶里其余行的归一化均值向量的余弦（低 = 说的不是一回事）；
- 同叶基线：把叶的行随机切成同样大小的两份，同一余弦的中位（抽样噪声下「同一回事」应有的值）。
- 意图内中心例子 / 叶其余中心例子：两部分各自的可引 medoid（平均余弦最大的那条，引用护栏与主报告相同）；
  候选限定在该部分长度的 p10–p90 之间——一条堆满常见词的超长 prompt 与均值向量的余弦会很高，却不典型（实测：
  「股票行情与选股建议查询」叶的其余部分，未限长时中心是一条 150 字的荐股指令）。
读法：意图本身就是按语义分的，所以 cos差 几乎总是负的，读相对大小；落在全体低端的格子，叶名描述的是另一回事
（另一个话题，或同一话题下的另一种功能），这时看意图内中心例子，不看叶名。
输出：work/<域>/intent_structure/l1_leaf_namefit.csv；另写 leaf_family_granularity.json——叶与交付家族（bu_family_final）
两层在同一组指标上的对比（组数、每组行数、组内紧致度、意图×快照格子里的有效组数/覆盖 80% 所需组数），回答「改用家族会不会更好」。
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pooled5_common import WORK  # noqa: E402
from p5_intent_structure import _names, load, quotable_mask, rng_for, run_dir  # noqa: E402

MIN_ROWS = 10
BASE_B = 30


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


def _medoid_text(d: pd.DataFrame, E: np.ndarray, idx: np.ndarray) -> str:
    ln = d.loc[idx, "query"].astype(str).str.len().to_numpy()
    lo, hi = np.quantile(ln, [0.10, 0.90])
    q = idx[d.loc[idx, "_quotable"].to_numpy() & (ln >= lo) & (ln <= hi)]
    if not len(q):
        return ""
    c = _unit(E[idx].mean(0))
    return str(d.loc[q[int(np.argmax(E[q] @ c))], "query"])


def granularity(domain: str, d: pd.DataFrame, E: np.ndarray, name_leaf: dict) -> dict:
    d = d.assign(_fam=d["bu_family_final"].astype(int))
    lpf = d.groupby("_fam")["bu_leaf"].nunique()
    res = {"rows": len(d), "leaves": int(d["bu_leaf"].nunique()), "families": int(d["_fam"].nunique()),
           "families_with_more_than_one_leaf": int((lpf > 1).sum()),
           "rows_in_multi_leaf_families_%": round(100 * float(d["_fam"].map(lpf).gt(1).mean()), 1),
           "multi_leaf_families": [sorted(name_leaf.get(x, str(x)) for x in d.loc[d["_fam"] == f, "bu_leaf"].unique())
                                   for f in lpf[lpf > 1].sort_values(ascending=False).index]}
    for lvl, col in (("叶", "bu_leaf"), ("家族", "_fam")):
        sz = d.groupby(col).size()
        w, v = [], []
        for _, idx in d.groupby(col).groups.items():
            w.append(len(idx))
            v.append(float((E[idx] @ _unit(E[idx].mean(0))).mean()))
        effs, k80s, tops = [], [], []
        for _, cell in d.groupby(["td_l1", "source"]):
            if len(cell) < 30:
                continue
            p = cell[col].value_counts(normalize=True).to_numpy()
            effs.append(math.exp(-(p * np.log(p)).sum()))
            k80s.append(int(np.searchsorted(np.cumsum(p), 0.8) + 1))
            tops.append(float(p[0]))
        res[lvl] = {"groups": int(len(sz)), "rows_per_group_median": int(sz.median()), "rows_per_group_p10": int(sz.quantile(.1)),
                    "within_group_mean_cosine_to_centre_row_weighted": round(float(np.average(v, weights=w)), 3),
                    "cells_n>=30": len(effs), "effective_groups_median": round(float(np.median(effs)), 1),
                    "groups_for_80%_median": float(np.median(k80s)), "top_group_share_median_%": round(100 * float(np.median(tops)), 1)}
    (WORK / domain / "intent_structure" / "leaf_family_granularity.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "multi_leaf_families"}, ensure_ascii=False))
    return res


def run(domain: str) -> pd.DataFrame:
    d = load(domain).reset_index(drop=True)
    E = np.load(run_dir(domain) / "emb_hybrid.npy").astype(np.float32)
    assert len(E) == len(d), f"embeddings {len(E)} vs rows {len(d)}"
    E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-12
    nm = _names(domain, d)
    name_l1 = {k: v[0] for k, v in nm["l1"].items()}
    name_leaf = {k: v[0] for k, v in nm["leaf"].items()}
    d["bu_leaf"] = d["bu_leaf"].astype(int)
    d["_quotable"] = quotable_mask(domain, d).to_numpy()
    leaf_rows = d.groupby("bu_leaf").groups
    rows = []
    for (code, lf), g in d.groupby(["td_l1", "bu_leaf"], sort=False):
        if len(g) < MIN_ROWS:
            continue
        idx_all = np.asarray(leaf_rows[lf])
        rest = np.setdiff1d(idx_all, g.index.to_numpy())
        vc = d.loc[idx_all, "td_l1"].value_counts()
        cos = float(_unit(E[g.index].mean(0)) @ _unit(E[rest].mean(0))) if len(rest) >= MIN_ROWS else float("nan")
        base = float("nan")
        if len(rest) >= MIN_ROWS:
            rng = rng_for("leaf_namefit", str(code), str(lf))
            k = min(len(g), len(idx_all) // 2)
            vals = []
            for _ in range(BASE_B):
                perm = rng.permutation(idx_all)
                vals.append(float(_unit(E[perm[:k]].mean(0)) @ _unit(E[perm[k:]].mean(0))))
            base = float(np.median(vals))
        rows.append({"意图": code, "意图名": name_l1.get(code, code), "叶": int(lf), "叶名": name_leaf.get(lf, lf),
                     "意图内行数": len(g), "占该意图%": round(100 * len(g) / int((d.td_l1 == code).sum()), 1),
                     "叶总行数": len(idx_all), "该意图占叶%": round(100 * len(g) / len(idx_all), 1),
                     "叶主导意图": name_l1.get(vc.index[0], vc.index[0]), "主导占叶%": round(100 * vc.iloc[0] / len(idx_all), 1),
                     "cos_意图内vs叶其余": round(cos, 3), "cos_同叶基线": round(base, 3),
                     "cos差": round(cos - base, 3) if cos == cos and base == base else float("nan"),
                     "意图内中心例子": _medoid_text(d, E, g.index.to_numpy()),
                     "叶其余中心例子": _medoid_text(d, E, rest) if len(rest) >= MIN_ROWS else ""})
    t = pd.DataFrame(rows).sort_values(["意图", "意图内行数"], ascending=[True, False])
    granularity(domain, d, E, name_leaf)
    out = WORK / domain / "intent_structure" / "l1_leaf_namefit.csv"
    t.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"{domain}: {len(t)} intent x leaf cells (>= {MIN_ROWS} rows) -> {out}")
    ok = t.dropna(subset=["cos差"])
    print("cos差 quartiles:", ok["cos差"].quantile([.1, .25, .5, .75, .9]).round(3).to_dict())
    print("quotable medoids missing:", int((t["意图内中心例子"] == "").sum()), "intent-part,", int((t["叶其余中心例子"] == "").sum()), "rest-of-leaf")
    return t


if __name__ == "__main__":
    for dom in sys.argv[1:]:
        run(dom)
