#!/usr/bin/env python
"""意图内部结构与代表性样例：一个 L1 意图在不同快照里「装的是什么」，以及怎样选出配得上「代表」二字的例子。

只读运行产物、只写 `work/<域>/intent_structure/`，不改任何已交付文件，也不改其它 p5 脚本。
按 P5_COHORT 取域（金融8 起步：P5_COHORT=fin8 python p5_intent_structure.py 金融8）。

一、现有例子为什么不具代表性（实测，fin-pool8，每个意图 × 快照格子 n>=10 的 120 格）
    跨快照对比报告里每格印的是「可引行里 pv_norm 最高的 3 条」。三个问题：
    1. 流量在长尾里不是排序：随机层与语音1k 的 PV 几乎全是 1 或没有，27 格（随机/语音层 40%）第 3 条例子与
       3 行以上并列，「流量最高」实为文件顺序；
    2. 只覆盖格子内部的一小块：例子所在的叶合计只占该格行数的中位 33%（p25 22%）；
    3. 系统性偏短：长度分位均值 0.35——头部流量最高的串就是最短的导航式说法；另有 3 格同一串印了两遍。

二、代表性样例的做法（本脚本 A 部分）
    代表性拆成两件可测的事：**构成**（例子覆盖了格子内部多大份额、分配是否与真实构成一致）与
    **典型性**（每条例子是否是它所在子群的中心，而不是边缘或异类）。
    - 子群：意图格子按自下而上的叶分层；叶格子按 L1 意图分层；L2 格子按叶分层（两条路线互为对方的分层变量）。
    - 子群内取中心：embedding 行归一化后，平均余弦相似度 = x_i · 均值向量，所以按它排序就是按平均余弦的 medoid
      （精确，不是近似）；去掉完全重复的串和与已选例子余弦 > 0.97 的近重复。
    - 三种方法同一组指标并列（被代表质量、分配 TVD、典型性分位、长度分位、自助法 Jaccard）：
      旧（可引行里流量最高 3 条）；Hamilton k=3（同样 3 条，按各子群全部行的份额做最大余数分配，再取子群中心）；
      **构成显式**（子群按份额从大到小，每个子群一条中心例子并标出它代表的份额，覆盖到 80% 或 8 条为止；
      没有可引行的子群份额单独记为「不可代表质量」）。展示与交付用构成显式。
    - 为什么 3–5 条不够：fin-pool8 上 n>=30 的意图格子，覆盖 80% 行需要的叶数中位 9、p75 12——
      任何固定 3 条的列表都只能代表一小块；构成要靠「每条例子带份额」显式写出来。
    思路对应原型选择（Kim, Khanna & Koyejo 2016, MMD-critic）与按层比例分配的分层抽样；medoid 见 PAM
    （Kaufman & Rousseeuw）。这里不追求「最有信息量」，只追求「构成对、每条是中心、可复算」。

三、同一意图在不同快照之间到底哪里不同（B 部分），一次只变一个因素的对齐快照对：
    界面（头部：2026搜索↔助手头部；随机：2026搜索随机↔助手随机）、深度（搜索内、助手内）、时间（头部、随机）、
    输入方式（助手头部↔语音头部）。
    - 构成：意图内的叶构成与 L2 构成（完整分布，Wilson 区间），叶构成 TVD 配同源噪声上界；
    - 说法：金融语境的几个可数特征（咨询/求建议、事件与进度、问原因、实体绑定、时效、是非核实、问句），
      每格占比配 Wilson 区间，对齐快照对的差配 Newcombe 区间——这些词表是代理，结论要连同读样例一起用；
    - 用词：带信息先验的加权对数几率（Monroe, Colaresi & Quinn 2008, "Fightin' Words"），只输出词，不输出整句；
    - 「差别有多真」：分类器双样本检验（Lopez-Paz & Oquab 2017）——在该意图内部、只用两侧的 embedding，
      平衡抽样后 5 折交叉验证 AUC，打乱标签 B 次得到零分布；置换 p < 0.05 才读作「两侧可分」。

四、意图内部结构（C 部分）：每个意图 × 快照的有效叶数（exp 熵）、覆盖 80% 所需叶数、首位叶占比。

五、供判读的盲样本：对三个宽意图（裁决、个股消息、投资建议）按快照分层抽样导出，给盲标用（样本里不带快照）。
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pooled5_common import FORM, SOURCES, SRC_ZH, WORK, load, run_dir  # noqa: E402
from p5_snapshot_classes import (  # noqa: E402
    NEVER_QUOTE_CLASSES, QUOTE_BLOCK, QUOTE_COOC, _names, _risk_indices, extra_quote_mask, rng_for, risk_strings,
    screened_quote_block)

K_EXAMPLES = 5
NEAR_DUP = 0.97
BOOT_B = 20
NULL_B = 200
AUC_NULL_B = 100
MIN_PAIR_N = 40
FACET_TARGETS = ["VERDICT_QUESTION", "NEWS_EVENT", "INVEST_ADVICE"]
FACET_SAMPLE_PER_SNAPSHOT = 150
PAIRS = [("2026search", "assistant_top", "界面·头部层"), ("2026search_rand", "assistant_random", "界面·随机层"),
         ("2026search", "2026search_rand", "深度·搜索内"), ("assistant_top", "assistant_random", "深度·助手内"),
         ("2025search", "2026search", "时间·头部层"), ("2025search_rand", "2026search_rand", "时间·随机层"),
         ("assistant_top", "assistant_voice_top", "输入方式·头部层")]
FACETS = {
    "咨询/求建议": r"(值得|建议|推荐|怎么看|分析|能不能买|能买吗|可以买|该不该|要不要|适合|前景|预测|会涨|会跌|涨吗|跌吗|目标价|估值|可能性|机会|靠谱|可靠)",
    "事件与进度": r"(公告|消息|进展|进度|上市|重组|分红|业绩|财报|年报|中报|预告|停牌|复牌|解禁|增持|减持|回购|并购|IPO|发行|申购|中签)",
    "问原因": r"(为什么|为何|什么原因|怎么回事|原因)",
    "实体绑定": r"(股份|集团|科技|股票|股吧|证券|银行|基金|[0-9]{6}|控股|实业|电子|医药|能源|新材|智能|保险)",
    "时效": FORM["时效词"],
    "是非核实": FORM["是非核实"],
    "问句": FORM["疑问句"],
    "第一人称": FORM["第一人称"],
}


# ---------------------------------------------------------------- 小工具

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def newcombe(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float, float]:
    """p2 - p1 与 Newcombe 混合 Wilson 区间（比例，不是百分点）。"""
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    d = p2 - p1
    lo = d - math.sqrt((p2 - l2) ** 2 + (u1 - p1) ** 2)
    hi = d + math.sqrt((u2 - p2) ** 2 + (p1 - l1) ** 2)
    return d, lo, hi


def pct_rank(values: np.ndarray, v: float) -> float:
    return float((values < v).mean() + 0.5 * (values == v).mean())


def quotable_mask(domain: str, d: pd.DataFrame) -> pd.Series:
    """与 p5_snapshot_classes.build 的可引口径逐项相同（风控图层、硬规则、共现、被点名串、不可引类、按域第五层、逐串筛查名单）。"""
    q = d["query"].astype(str)
    risk_idx = _risk_indices(domain, len(d))
    named = risk_strings(domain)
    hard = q.map(lambda s: bool(QUOTE_BLOCK.search(s)))
    cooc = q.map(lambda s: bool(QUOTE_COOC[0].search(s) and QUOTE_COOC[1].search(s)))
    never = d["td_l1"].astype(str).isin(NEVER_QUOTE_CLASSES.get(domain, set()))
    extra = extra_quote_mask(domain, q)
    screened = q.isin(screened_quote_block(domain))
    return pd.Series(~d.index.isin(list(risk_idx)), index=d.index) & ~hard & ~cooc & ~q.isin(named) & ~never & ~extra & ~screened


def hamilton(shares: dict, k: int, caps: dict) -> dict:
    """最大余数法（Hamilton）：配额只算一次，整数部分先给，剩下的席位按小数部分从大到小给。

    每个子群不超过 caps（可引的不同串数）。只有当上限卡住、席位没分完时，才把剩余席位按份额在还有余量的子群里
    再分一轮。**第一版在每一轮都对「剩余席位」重新取整，结果最大的子群一轮拿一个、拿走了全部席位**——
    fin-pool8 上一个格子的 5 条例子全部出自同一个叶，就是这个错。"""
    alloc = {g: 0 for g in shares}
    total = min(k, sum(caps.get(g, 0) for g, v in shares.items() if v > 0))
    while sum(alloc.values()) < total:
        elig = {g: v for g, v in shares.items() if v > 0 and alloc[g] < caps.get(g, 0)}
        if not elig:
            break
        left = total - sum(alloc.values())
        tot = sum(elig.values())
        quota = {g: left * v / tot for g, v in elig.items()}
        add = {g: min(int(math.floor(quota[g])), caps[g] - alloc[g]) for g in elig}
        rem = left - sum(add.values())
        for g in sorted(elig, key=lambda x: (-(quota[x] - math.floor(quota[x])), -elig[x], str(x))):
            if rem <= 0:
                break
            if alloc[g] + add[g] < caps[g]:
                add[g] += 1
                rem -= 1
        if not any(add.values()):
            break
        for g, a in add.items():
            alloc[g] += a
    return alloc


# ---------------------------------------------------------------- A 代表性样例

def select_cell(cell: pd.DataFrame, E: np.ndarray, strata_col: str, k: int) -> tuple[list[dict], dict]:
    n = len(cell)
    q = cell["query"].astype(str)
    shares = cell[strata_col].value_counts(normalize=True).to_dict()
    qcell = cell[cell["_quotable"]]
    caps = {g: int(qcell.loc[qcell[strata_col] == g, "query"].nunique()) for g in shares}
    k_eff = min(k, int(qcell["query"].nunique()))
    info = {"n": n, "k": k_eff, "quotable_rows": int(len(qcell)),
            "unrepresentable_mass": round(float(sum(s for g, s in shares.items() if caps.get(g, 0) == 0)), 4)}
    if k_eff == 0:
        return [], {**info, "represented_mass": 0.0, "alloc_tvd": float("nan")}
    alloc = hamilton(shares, k_eff, caps)
    Ecell = E[cell["_row"].to_numpy()]
    cmean = Ecell.mean(0)
    typ_all = Ecell @ cmean
    L_all = q.str.len().to_numpy()
    chosen: list[dict] = []
    chosen_vecs: list[np.ndarray] = []
    for g, a in sorted(alloc.items(), key=lambda kv: -shares[kv[0]]):
        if a <= 0:
            continue
        sub = cell[cell[strata_col] == g]
        smean = E[sub["_row"].to_numpy()].mean(0)
        cand = sub[sub["_quotable"]].drop_duplicates("query")
        score = E[cand["_row"].to_numpy()] @ smean
        order = np.argsort(-score, kind="stable")
        got = 0
        for oi in order:
            r = cand.iloc[oi]
            v = E[int(r["_row"])]
            if any(float(v @ w) > NEAR_DUP for w in chosen_vecs):
                continue
            pos = cell.index.get_loc(r.name)
            chosen.append({"stratum": g, "stratum_share": round(shares[g], 4), "query": r["query"],
                           "typicality_pct": round(pct_rank(typ_all, float(typ_all[pos])), 3),
                           "length_pct": round(pct_rank(L_all, float(L_all[pos])), 3),
                           "pv_raw": r.get("pv_raw")})
            chosen_vecs.append(v)
            got += 1
            if got >= a:
                break
    k_got = max(1, len(chosen))
    got_by = pd.Series([c["stratum"] for c in chosen]).value_counts().to_dict() if chosen else {}
    tvd = 0.5 * sum(abs(got_by.get(g, 0) / k_got - s) for g, s in shares.items())
    return chosen, {**info, "k_got": len(chosen), "represented_mass": round(float(sum(shares[g] for g in got_by)), 4),
                    "alloc_tvd": round(float(tvd), 4)}


def boot_stability(cell: pd.DataFrame, E: np.ndarray, strata_col: str, k: int, rng: np.random.Generator) -> float:
    base, _ = select_cell(cell, E, strata_col, k)
    base_set = {c["stratum"] for c in base}
    if not base_set:
        return float("nan")
    js = []
    for _ in range(BOOT_B):
        bs = cell.iloc[rng.integers(0, len(cell), len(cell))]
        bs = bs[~bs.index.duplicated()]
        got, _ = select_cell(bs, E, strata_col, k)
        s = {c["stratum"] for c in got}
        js.append(len(base_set & s) / max(1, len(base_set | s)))
    return round(float(np.mean(js)), 3)


def select_cover(cell: pd.DataFrame, E: np.ndarray, strata_col: str, target: float = 0.8, cap: int = 8) -> tuple[list[dict], dict]:
    """构成显式的代表性样例：子群按份额从大到小，每个子群取一条中心例子，并标出它代表的份额；
    累计覆盖到 target 或例子数到 cap 为止。没有可引行的子群记为「不可代表质量」，跳过、不占名额。
    这样读者看到的每条例子都带着「它代表该格多少行」，没被代表的份额也明写出来。"""
    q = cell["query"].astype(str)
    shares = cell[strata_col].value_counts(normalize=True)
    Ecell = E[cell["_row"].to_numpy()]
    typ_all = Ecell @ Ecell.mean(0)
    L_all = q.str.len().to_numpy()
    chosen, vecs = [], []
    covered, unrep = 0.0, 0.0
    for g, sh in shares.items():
        if covered >= target or len(chosen) >= cap:
            break
        sub = cell[cell[strata_col] == g]
        cand = sub[sub["_quotable"]].drop_duplicates("query")
        if not len(cand):
            unrep += float(sh)
            continue
        smean = E[sub["_row"].to_numpy()].mean(0)
        score = E[cand["_row"].to_numpy()] @ smean
        for oi in np.argsort(-score, kind="stable"):
            r = cand.iloc[oi]
            v = E[int(r["_row"])]
            if any(float(v @ w) > NEAR_DUP for w in vecs):
                continue
            pos = cell.index.get_loc(r.name)
            chosen.append({"stratum": g, "stratum_share": round(float(sh), 4), "query": r["query"],
                           "typicality_pct": round(pct_rank(typ_all, float(typ_all[pos])), 3),
                           "length_pct": round(pct_rank(L_all, float(L_all[pos])), 3), "pv_raw": r.get("pv_raw")})
            vecs.append(v)
            covered += float(sh)
            break
    return chosen, {"n": len(cell), "k_got": len(chosen), "represented_mass": round(covered, 4),
                    "unrepresentable_mass": round(unrep, 4), "n_strata": int(len(shares))}


def boot_cover_stability(cell: pd.DataFrame, E: np.ndarray, strata_col: str, rng: np.random.Generator) -> float:
    base, _ = select_cover(cell, E, strata_col)
    base_set = {c["stratum"] for c in base}
    if not base_set:
        return float("nan")
    js = []
    for _ in range(BOOT_B):
        bs = cell.iloc[rng.integers(0, len(cell), len(cell))]
        bs = bs[~bs.index.duplicated()]
        got, _ = select_cover(bs, E, strata_col)
        st = {c["stratum"] for c in got}
        js.append(len(base_set & st) / max(1, len(base_set | st)))
    return round(float(np.mean(js)), 3)


def old_method_metrics(cell: pd.DataFrame, E: np.ndarray, strata_col: str, shown: list[str], rng: np.random.Generator) -> dict:
    shares = cell[strata_col].value_counts(normalize=True).to_dict()
    rows = cell[cell["query"].isin(shown)].drop_duplicates("query")
    got_by = rows[strata_col].value_counts().to_dict()
    k_got = max(1, len(shown))
    tvd = 0.5 * sum(abs(got_by.get(g, 0) / k_got - s) for g, s in shares.items())
    Ecell = E[cell["_row"].to_numpy()]
    typ_all = Ecell @ Ecell.mean(0)
    L_all = cell["query"].astype(str).str.len().to_numpy()
    pos = [cell.index.get_loc(i) for i in rows.index]
    qc = cell[cell["_quotable"]]
    third = qc.sort_values("pv_norm", ascending=False)["pv_norm"].iloc[min(2, len(qc) - 1)] if len(qc) else float("nan")
    ties = int((qc["pv_norm"] == third).sum()) if len(qc) else 0
    # 自助法：重抽行以后「流量最高 3 条」换了多少
    base = set(shown)
    js = []
    for _ in range(BOOT_B):
        bs = qc.iloc[rng.integers(0, len(qc), len(qc))] if len(qc) else qc
        top = set(bs.drop_duplicates("query").sort_values("pv_norm", ascending=False)["query"].head(3))
        js.append(len(base & top) / max(1, len(base | top)))
    return {"k_got": len(shown), "duplicates_shown": int(len(shown) - len(set(shown))),
            "represented_mass": round(float(sum(shares.get(g, 0) for g in got_by)), 4), "alloc_tvd": round(float(tvd), 4),
            "typicality_pct": round(float(np.mean([pct_rank(typ_all, float(typ_all[p])) for p in pos])), 3) if pos else float("nan"),
            "length_pct": round(float(np.mean([pct_rank(L_all, float(L_all[p])) for p in pos])), 3) if pos else float("nan"),
            "pv_ties_at_3rd": ties, "order_arbitrary": bool(ties > 3),
            "boot_jaccard_examples": round(float(np.mean(js)), 3) if js else float("nan")}


# ---------------------------------------------------------------- B 构成、说法、用词、可分性

def composition(d: pd.DataFrame, outer: str, inner: str, srcs: list[str], name_outer: dict, name_inner: dict) -> pd.DataFrame:
    rows = []
    for (o, s), g in d.groupby([outer, "source"], sort=False):
        if s not in srcs:
            continue
        n = len(g)
        for i, kcnt in g[inner].value_counts().items():
            lo, hi = wilson(int(kcnt), n)
            rows.append({"外层": o, "外层名": name_outer.get(o, str(o)), "快照": SRC_ZH[s], "source": s, "n": n,
                         "内层": i, "内层名": name_inner.get(i, str(i)), "条数": int(kcnt),
                         "占比%": round(100 * kcnt / n, 2), "95CI低%": round(100 * lo, 2), "95CI高%": round(100 * hi, 2)})
    return pd.DataFrame(rows)


def structure_summary(comp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (o, s), g in comp.groupby(["外层", "source"], sort=False):
        p = g["条数"].to_numpy() / g["条数"].sum()
        ps = np.sort(p)[::-1]
        rows.append({"外层": o, "外层名": g["外层名"].iloc[0], "快照": g["快照"].iloc[0], "n": int(g["n"].iloc[0]),
                     "内层个数": int(len(p)), "有效内层数(exp熵)": round(float(np.exp(-(p * np.log(p)).sum())), 2),
                     "覆盖80%所需内层数": int(np.searchsorted(np.cumsum(ps), 0.8) + 1),
                     "首位内层": g.sort_values("条数", ascending=False)["内层名"].iloc[0],
                     "首位占比%": round(100 * float(ps[0]), 2)})
    return pd.DataFrame(rows)


def mix_contrasts(d: pd.DataFrame, outer: str, inner: str, name_outer: dict, name_inner: dict, srcs: list[str]) -> pd.DataFrame:
    rows = []
    for o, g in d.groupby(outer, sort=False):
        for a, b, label in PAIRS:
            if a not in srcs or b not in srcs:
                continue
            ga, gb = g[g.source == a], g[g.source == b]
            if len(ga) < MIN_PAIR_N or len(gb) < MIN_PAIR_N:
                continue
            cats = sorted(set(ga[inner]) | set(gb[inner]), key=str)
            pa = ga[inner].value_counts(normalize=True).reindex(cats).fillna(0)
            pb = gb[inner].value_counts(normalize=True).reindex(cats).fillna(0)
            obs = float(0.5 * (pa - pb).abs().sum())
            codes = pd.Index(cats).get_indexer(pd.concat([ga[inner], gb[inner]]).to_numpy())
            rng = rng_for("intent_structure_mix", str(o), a, b)
            null = np.empty(NULL_B)
            for t in range(NULL_B):
                idx = rng.permutation(len(codes))
                x, y = codes[idx[:len(ga)]], codes[idx[len(ga):]]
                null[t] = 0.5 * np.abs(np.bincount(x, minlength=len(cats)) / len(x) - np.bincount(y, minlength=len(cats)) / len(y)).sum()
            # 差最大的三个内层，配 Newcombe
            diffs = []
            for c in cats:
                ka, kb = int((ga[inner] == c).sum()), int((gb[inner] == c).sum())
                dd, lo, hi = newcombe(ka, len(ga), kb, len(gb))
                diffs.append((abs(dd), c, dd, lo, hi))
            diffs.sort(key=lambda x: -x[0])
            top = "；".join(f"{name_inner.get(c, c)} {100*dd:+.1f}pp [{100*lo:+.1f}, {100*hi:+.1f}]{'*' if lo > 0 or hi < 0 else ''}"
                           for _, c, dd, lo, hi in diffs[:3])
            rows.append({"外层": o, "外层名": name_outer.get(o, str(o)), "对比": label, "a": SRC_ZH[a], "b": SRC_ZH[b],
                         "n_a": len(ga), "n_b": len(gb), "内层TVD": round(obs, 4),
                         "同源噪声上界": round(float(np.percentile(null, 95)), 4), "超出噪声": bool(obs > np.percentile(null, 95)),
                         "差最大的三个内层(b-a, *=区间不含0)": top})
    return pd.DataFrame(rows)


def facet_tables(d: pd.DataFrame, name_l1: dict, srcs: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    q = d["query"].astype(str)
    F = {f: q.map(lambda s, p=re.compile(rx): bool(p.search(s))) for f, rx in FACETS.items()}
    cell_rows, pair_rows = [], []
    for o, g in d.groupby("td_l1", sort=False):
        for s in srcs:
            gs = g[g.source == s]
            if not len(gs):
                continue
            rec = {"L1": o, "L1名": name_l1.get(o, o), "快照": SRC_ZH[s], "n": len(gs)}
            for f in FACETS:
                k = int(F[f].loc[gs.index].sum())
                lo, hi = wilson(k, len(gs))
                rec[f"{f}%"] = round(100 * k / len(gs), 2)
                rec[f"{f}_CI%"] = f"[{100*lo:.1f}, {100*hi:.1f}]"
            cell_rows.append(rec)
        for a, b, label in PAIRS:
            if a not in srcs or b not in srcs:
                continue
            ga, gb = g[g.source == a], g[g.source == b]
            if len(ga) < MIN_PAIR_N or len(gb) < MIN_PAIR_N:
                continue
            rec = {"L1": o, "L1名": name_l1.get(o, o), "对比": label, "a": SRC_ZH[a], "b": SRC_ZH[b], "n_a": len(ga), "n_b": len(gb)}
            for f in FACETS:
                ka, kb = int(F[f].loc[ga.index].sum()), int(F[f].loc[gb.index].sum())
                dd, lo, hi = newcombe(ka, len(ga), kb, len(gb))
                rec[f"{f}_a%"] = round(100 * ka / len(ga), 1)
                rec[f"{f}_b%"] = round(100 * kb / len(gb), 1)
                rec[f"{f}_差pp"] = f"{100*dd:+.1f} [{100*lo:+.1f}, {100*hi:+.1f}]{'*' if lo > 0 or hi < 0 else ''}"
            pair_rows.append(rec)
    return pd.DataFrame(cell_rows), pd.DataFrame(pair_rows)


def fightin_words(d: pd.DataFrame, name_l1: dict, srcs: list[str], top: int = 15, alpha0: float = 500.0) -> pd.DataFrame:
    import jieba
    jieba.setLogLevel(60)
    toks = d["query"].astype(str).map(lambda s: [t for t in jieba.lcut(s) if t.strip() and not re.fullmatch(r"[\W_]+", t)])
    bg: dict[str, int] = {}
    for ts in toks:
        for t in ts:
            bg[t] = bg.get(t, 0) + 1
    bg_tot = sum(bg.values())
    rows = []
    for o, g in d.groupby("td_l1", sort=False):
        for a, b, label in PAIRS:
            if a not in srcs or b not in srcs:
                continue
            ia, ib = g.index[g.source == a], g.index[g.source == b]
            if len(ia) < 50 or len(ib) < 50:
                continue
            ca: dict[str, int] = {}
            cb: dict[str, int] = {}
            for ts in toks.loc[ia]:
                for t in ts:
                    ca[t] = ca.get(t, 0) + 1
            for ts in toks.loc[ib]:
                for t in ts:
                    cb[t] = cb.get(t, 0) + 1
            na, nb = sum(ca.values()), sum(cb.values())
            res = []
            for t in set(ca) | set(cb):
                aw = alpha0 * bg[t] / bg_tot
                ya, yb = ca.get(t, 0), cb.get(t, 0)
                delta = (math.log((yb + aw) / (nb + alpha0 - yb - aw)) - math.log((ya + aw) / (na + alpha0 - ya - aw)))
                z = delta / math.sqrt(1 / (ya + aw) + 1 / (yb + aw))
                res.append((z, t, ya, yb))
            res.sort()
            for side, lst in (("偏a", res[:top]), ("偏b", res[::-1][:top])):
                for z, t, ya, yb in lst:
                    if abs(z) < 1.96:
                        continue
                    rows.append({"L1": o, "L1名": name_l1.get(o, o), "对比": label, "a": SRC_ZH[a], "b": SRC_ZH[b],
                                 "方向": side, "词": t, "a中次数": ya, "b中次数": yb, "z": round(z, 2)})
    return pd.DataFrame(rows)


def c2st(d: pd.DataFrame, E: np.ndarray, name_l1: dict, srcs: list[str]) -> pd.DataFrame:
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    rows = []
    for o, g in d.groupby("td_l1", sort=False):
        for a, b, label in PAIRS:
            if a not in srcs or b not in srcs:
                continue
            ia, ib = g["_row"][g.source == a].to_numpy(), g["_row"][g.source == b].to_numpy()
            if len(ia) < MIN_PAIR_N or len(ib) < MIN_PAIR_N:
                continue
            rng = rng_for("intent_structure_c2st", str(o), a, b)
            m = min(len(ia), len(ib), 400)
            xa, xb = rng.choice(ia, m, replace=False), rng.choice(ib, m, replace=False)
            X = E[np.concatenate([xa, xb])]
            y = np.r_[np.zeros(m), np.ones(m)]
            X = PCA(n_components=min(32, X.shape[0] - 1), random_state=0).fit_transform(X)

            def oof_auc(yy: np.ndarray, seed: int) -> float:
                p = np.zeros(len(yy))
                for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, yy):
                    clf = LogisticRegression(C=1.0, max_iter=2000).fit(X[tr], yy[tr])
                    p[te] = clf.predict_proba(X[te])[:, 1]
                return float(roc_auc_score(yy, p))

            obs = oof_auc(y, 0)
            null = np.array([oof_auc(rng.permutation(y), t + 1) for t in range(AUC_NULL_B)])
            rows.append({"L1": o, "L1名": name_l1.get(o, o), "对比": label, "a": SRC_ZH[a], "b": SRC_ZH[b],
                         "每侧抽样n": m, "AUC": round(obs, 3), "零分布95分位": round(float(np.percentile(null, 95)), 3),
                         "p": round((1 + int((null >= obs).sum())) / (AUC_NULL_B + 1), 3),
                         "两侧可分(p<0.05)": bool((1 + int((null >= obs).sum())) / (AUC_NULL_B + 1) < 0.05)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 驱动

def run(domain: str) -> None:
    out = WORK / domain / "intent_structure"
    out.mkdir(parents=True, exist_ok=True)
    d = load(domain).reset_index(drop=True)
    gen = run_dir(domain)
    E = np.load(gen / "emb_hybrid.npy").astype(np.float32)
    assert len(E) == len(d), f"embeddings {len(E)} vs rows {len(d)}"
    E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-12
    d["_row"] = np.arange(len(d))
    d["_quotable"] = quotable_mask(domain, d).to_numpy()
    summ_path = WORK / domain / "snapshot_classes" / "summary.json"
    if summ_path.exists():
        want = json.loads(summ_path.read_text(encoding="utf-8")).get("quotable_rows")
        assert want is None or int(d["_quotable"].sum()) == int(want), f"quotable {int(d['_quotable'].sum())} != report's {want}"
    srcs = [s for s in SOURCES if (d.source == s).any()]
    nm = _names(domain, d)
    name_l1 = {k: v[0] for k, v in nm["l1"].items()}
    name_leaf = {k: v[0] for k, v in nm["leaf"].items()}
    d["bu_leaf"] = d["bu_leaf"].astype(int)

    # A. 代表性样例。三种方法同一组指标：旧（流量最高 3 条）、Hamilton k=3（同样 3 条，按叶构成分配 + 子群中心）、
    #    构成显式（每个子群一条中心例子、标份额，覆盖到 80% 或 8 条）。展示用的是构成显式那一种。
    ex_rows, cell_rows = [], []
    old = WORK / domain / "snapshot_classes" / "examples_td_l1.csv"
    old_ex = pd.read_csv(old) if old.exists() else pd.DataFrame(columns=["key", "快照", "取法", "query"])
    for (o, s), cell in d.groupby(["td_l1", "source"], sort=False):
        if s not in srcs or len(cell) < 10:
            continue
        rng = rng_for("intent_structure_examples", str(o), s)
        cover, cinfo = select_cover(cell, E, "bu_leaf")
        cstab = boot_cover_stability(cell, E, "bu_leaf", rng)
        h3, hinfo = select_cell(cell, E, "bu_leaf", 3)
        hstab = boot_stability(cell, E, "bu_leaf", 3, rng)
        for c in cover:
            ex_rows.append({"层级": "L1意图", "类": o, "类名": name_l1.get(o, o), "快照": SRC_ZH[s], "n": len(cell),
                            "子群": name_leaf.get(c["stratum"], c["stratum"]), **{k: v for k, v in c.items() if k != "stratum"}})
        shown = old_ex[(old_ex["key"] == o) & (old_ex["快照"] == SRC_ZH[s]) & (old_ex["取法"] == "流量最高")]["query"].astype(str).tolist()
        om = old_method_metrics(cell, E, "bu_leaf", shown, rng) if shown else {}
        mean = lambda xs, k: round(float(np.mean([x[k] for x in xs])), 3) if xs else float("nan")  # noqa: E731
        cell_rows.append({"类": o, "类名": name_l1.get(o, o), "快照": SRC_ZH[s], "n": len(cell), "子群数": cinfo["n_strata"],
                          "旧_例子数": om.get("k_got"), "旧_重复": om.get("duplicates_shown"),
                          "旧_被代表质量%": round(100 * om["represented_mass"], 1) if om else float("nan"),
                          "旧_典型性分位": om.get("typicality_pct"), "旧_长度分位": om.get("length_pct"),
                          "旧_顺序实为任意": om.get("order_arbitrary"), "旧_自助法Jaccard": om.get("boot_jaccard_examples"),
                          "H3_例子数": hinfo.get("k_got", 0), "H3_被代表质量%": round(100 * hinfo["represented_mass"], 1),
                          "H3_分配TVD": hinfo["alloc_tvd"], "H3_典型性分位": mean(h3, "typicality_pct"), "H3_长度分位": mean(h3, "length_pct"),
                          "H3_自助法Jaccard": hstab,
                          "显式_例子数": cinfo["k_got"], "显式_被代表质量%": round(100 * cinfo["represented_mass"], 1),
                          "显式_不可代表质量%": round(100 * cinfo["unrepresentable_mass"], 1),
                          "显式_典型性分位": mean(cover, "typicality_pct"), "显式_长度分位": mean(cover, "length_pct"),
                          "显式_自助法Jaccard": cstab})
    for (l2, s), cell in d.groupby(["td_l2", "source"], sort=False):
        if s not in srcs or len(cell) < 10:
            continue
        cover, _ = select_cover(cell, E, "bu_leaf")
        for c in cover:
            ex_rows.append({"层级": "L2子意图", "类": l2, "类名": l2, "快照": SRC_ZH[s], "n": len(cell),
                            "子群": name_leaf.get(c["stratum"], c["stratum"]), **{k: v for k, v in c.items() if k != "stratum"}})
    for (lf, s), cell in d.groupby(["bu_leaf", "source"], sort=False):
        if s not in srcs or len(cell) < 10:
            continue
        cover, _ = select_cover(cell, E, "td_l1")
        for c in cover:
            ex_rows.append({"层级": "叶", "类": lf, "类名": name_leaf.get(lf, lf), "快照": SRC_ZH[s], "n": len(cell),
                            "子群": name_l1.get(c["stratum"], c["stratum"]), **{k: v for k, v in c.items() if k != "stratum"}})
    pd.DataFrame(ex_rows).to_csv(out / "representative_examples.csv", index=False, encoding="utf-8-sig")
    cells = pd.DataFrame(cell_rows)
    cells.to_csv(out / "example_method_comparison.csv", index=False, encoding="utf-8-sig")

    # B/C. 构成、结构、构成对比
    comp_leaf = composition(d, "td_l1", "bu_leaf", srcs, name_l1, name_leaf)
    comp_l2 = composition(d, "td_l1", "td_l2", srcs, name_l1, {})
    comp_leaf.to_csv(out / "l1_leaf_composition.csv", index=False, encoding="utf-8-sig")
    comp_l2.to_csv(out / "l1_l2_composition.csv", index=False, encoding="utf-8-sig")
    structure_summary(comp_leaf).to_csv(out / "l1_structure_summary.csv", index=False, encoding="utf-8-sig")
    mix_contrasts(d, "td_l1", "bu_leaf", name_l1, name_leaf, srcs).to_csv(out / "l1_leafmix_contrasts.csv", index=False, encoding="utf-8-sig")
    mix_contrasts(d, "td_l1", "td_l2", name_l1, {}, srcs).to_csv(out / "l1_l2mix_contrasts.csv", index=False, encoding="utf-8-sig")
    fc, fp = facet_tables(d, name_l1, srcs)
    fc.to_csv(out / "facets_by_cell.csv", index=False, encoding="utf-8-sig")
    fp.to_csv(out / "facets_pair_contrasts.csv", index=False, encoding="utf-8-sig")
    fightin_words(d, name_l1, srcs).to_csv(out / "distinctive_words.csv", index=False, encoding="utf-8-sig")
    c2st(d, E, name_l1, srcs).to_csv(out / "c2st_auc.csv", index=False, encoding="utf-8-sig")

    # 五. 盲样本（按快照分层；带快照的底表 + 不带快照、打乱顺序的盲表）
    for code in FACET_TARGETS:
        g = d[d.td_l1 == code]
        if not len(g):
            continue
        parts = []
        for s in srcs:
            gs = g[g.source == s]
            if not len(gs):
                continue
            take = gs if len(gs) <= FACET_SAMPLE_PER_SNAPSHOT else gs.sample(FACET_SAMPLE_PER_SNAPSHOT, random_state=20260915)
            parts.append(take.assign(抽样权重=len(gs) / len(take)))
        smp = pd.concat(parts)
        smp = smp.assign(sid=[f"{code[:3]}{i:04d}" for i in range(len(smp))])
        smp[["sid", "_row", "source", "query", "bu_leaf", "td_l2", "抽样权重"]].assign(叶名=smp["bu_leaf"].map(name_leaf)).to_csv(
            out / f"facet_sample_{code}.csv", index=False, encoding="utf-8-sig")
        blind = smp[["sid", "query"]].sample(frac=1.0, random_state=7)
        blind.to_csv(out / f"facet_blind_{code}.csv", index=False, encoding="utf-8-sig")

    summary = {"domain": domain, "rows": len(d), "quotable_rows": int(d["_quotable"].sum()),
               "cells_l1_x_snapshot_n>=10": int(len(cells)),
               "old_order_arbitrary_cells": int(cells["旧_顺序实为任意"].fillna(False).astype(bool).sum()),
               "median_represented_mass%": {m: float(cells[f"{m}_被代表质量%"].median()) for m in ("旧", "H3", "显式")},
               "mean_typicality_pct": {m: float(cells[f"{m}_典型性分位"].mean()) for m in ("旧", "H3", "显式")},
               "mean_length_pct": {m: float(cells[f"{m}_长度分位"].mean()) for m in ("旧", "H3", "显式")},
               "mean_boot_jaccard": {m: float(cells[f"{m}_自助法Jaccard"].mean()) for m in ("旧", "H3", "显式")},
               "explicit_mean_examples": float(cells["显式_例子数"].mean()),
               "facet_samples": {c: int(len(pd.read_csv(out / f"facet_sample_{c}.csv"))) for c in FACET_TARGETS if (out / f"facet_sample_{c}.csv").exists()}}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    for dom in sys.argv[1:]:
        run(dom)
