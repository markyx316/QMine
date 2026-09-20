#!/usr/bin/env python
"""宽意图的语用子功能：三名盲标的聚合、一致度、按快照加权的构成、与叶/L2 的交叉。

输入：工作流 `fin8-facet-coding` 的结果 JSON（每个意图一份码本 + 三个视角 × 分块的逐行标签）、
     `work/<域>/intent_structure/facet_sample_<CODE>.csv`（带快照与抽样权重的底表）。
输出：`work/<域>/intent_structure/facets/` 下的逐行多数标签、一致度、构成与对比表。

口径：
- 覆盖：每个 sid 必须被每个视角恰好标一次，否则拒绝（缺标或重复标直接报错，不静默丢）。唯一的例外是「位置可证的串号」：
  同一分块里某个 sid 出现两次、其中一份恰好落在一个缺标 sid 在盲表里的位置、且该块其余标签都按盲表顺序排列——
  这时丢掉错位的那一份，把缺标 sid 在该视角记为**声明缺票**（不替它补码：错位那份的码不一定属于它）。缺票行的多数用其余视角的票
  （仍须 >=2 票相同），κ 与三票全同只在票数完整的行上算；缺票总数超过样本 1% 仍然拒绝。每一处都写进 repairs.json。
  （实测：INVEST_ADVICE 的 answer 视角把 INV0660 写了两遍、漏了 INV0659，错位那份的码与 INV0659 的内容不符。）
- 多数：3 票里 >=2 票相同即为多数；三票各不相同记为「无多数」，单列，不并进任何码。
- 一致度：Fleiss κ（按码本全部码，含「不确定」码）；另报三票全同占比。
- 构成：抽样是按快照分层的（大快照抽 150，小快照全取），所以快照内占比直接用样本（每个快照内部等权），
  跨快照合并时才用抽样权重；区间用 Wilson（快照内 n 为样本行数，不虚增）。
- 对比：一次只变一个因素的对齐快照对，差配 Newcombe；两侧样本都 >=30 才算。
- 交叉：多数码 × 叶（该意图内部），看「同一个话题里是核实还是决策」。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pooled5_common import SRC_ZH, WORK  # noqa: E402
from p5_intent_structure import PAIRS, newcombe, wilson  # noqa: E402

MIN_N = 30
MAX_GAP_SHARE = 0.01


def fleiss_kappa(M: np.ndarray) -> float:
    """M: items x categories counts, each row sums to the same number of raters."""
    n = M.sum(1)[0]
    N = M.shape[0]
    p = M.sum(0) / (N * n)
    P = ((M * M).sum(1) - n) / (n * (n - 1))
    Pbar, Pe = P.mean(), (p * p).sum()
    return float((Pbar - Pe) / (1 - Pe)) if Pe < 1 else float("nan")


def aggregate(domain: str, wf_result: list[dict]) -> None:
    base = WORK / domain / "intent_structure"
    out = base / "facets"
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for item in wf_result:
        if not item:
            continue
        code, cb, anns = item["code"], item["codebook"], item["annotations"]
        smp = pd.read_csv(base / f"facet_sample_{code}.csv")
        sids = set(smp["sid"])
        codes = [c["code"] for c in cb["codes"]] + ([cb["unsure_code"]] if cb.get("unsure_code") and cb["unsure_code"] not in [c["code"] for c in cb["codes"]] else [])
        lab = {}
        problems = []
        gaps, repairs = set(), []
        blind_order = pd.read_csv(base / f"facet_blind_{code}.csv")["sid"].tolist()
        for a in anns:
            if not a or a.get("labels") is None:
                problems.append(f"missing annotation chunk {a.get('lens') if a else '?'} {a.get('range') if a else ''}")
                continue
            labels = list(a["labels"])
            sids_here = [r["sid"] for r in labels]
            lo = int(a["range"])
            expect = blind_order[lo:lo + len(labels)]
            dups = {x for x in sids_here if sids_here.count(x) > 1}
            missing_here = [x for x in expect if x not in set(sids_here)]
            in_order = sum(1 for k, x in enumerate(sids_here) if k < len(expect) and x == expect[k])
            drop = set()
            for x in dups:
                pos = [k for k, y in enumerate(sids_here) if y == x]
                misplaced = [k for k in pos if k < len(expect) and expect[k] != x and expect[k] in missing_here]
                kept = [k for k in pos if k not in misplaced]
                if len(kept) == 1 and len(misplaced) == len(pos) - 1 and in_order == len(labels) - len(misplaced):
                    for k in misplaced:
                        drop.add(k)
                        gaps.add((a["lens"], expect[k]))
                        repairs.append({"lens": a["lens"], "range": a["range"], "written_sid": x, "position": k, "dropped_code": labels[k]["code"],
                                        "declared_missing_vote_for": expect[k], "kept_position": kept[0]})
            for k, r in enumerate(labels):
                if k in drop:
                    continue
                key = (a["lens"], r["sid"])
                if key in lab:
                    problems.append(f"duplicate label {key}")
                lab[key] = r["code"]
        lenses = sorted({k[0] for k in lab})
        for lens in lenses:
            got = {s for (l, s) in lab if l == lens} | {s for (l, s) in gaps if l == lens}
            if got != sids:
                problems.append(f"{lens}: {len(sids - got)} sids unlabelled, {len(got - sids)} unknown sids")
        unknown = sorted({v for v in lab.values() if v not in codes})
        if unknown:
            problems.append(f"labels outside the codebook: {unknown}")
        if len(gaps) > MAX_GAP_SHARE * len(sids):
            problems.append(f"{len(gaps)} declared missing votes exceed {MAX_GAP_SHARE:.0%} of {len(sids)} rows")
        assert not problems, f"{code}: " + "; ".join(problems[:8])
        wide = pd.DataFrame({lens: {s: lab.get((lens, s), "缺票") for s in sids} for lens in lenses})
        wide.index.name = "sid"
        M = np.stack([(wide.values == c).sum(1) for c in codes], axis=1)
        top = M.max(1)
        maj = np.where(top >= 2, np.array(codes)[M.argmax(1)], "无多数")
        wide["多数"] = maj
        full = M.sum(1) == len(lenses)
        wide["三票全同"] = top == len(lenses)
        wide["票数完整"] = full
        d = smp.set_index("sid").join(wide)
        d["快照"] = d["source"].map(SRC_ZH)
        d.to_csv(out / f"{code}_labels.csv", encoding="utf-8-sig")
        kappa = fleiss_kappa(M[full])
        name = {c["code"]: c["name_zh"] for c in cb["codes"]}
        name.setdefault(cb.get("unsure_code", ""), "无法判断")
        name["无多数"] = "无多数"
        # 快照内构成（样本内等权），配 Wilson
        rows = []
        for s, g in d.groupby("source", sort=False):
            n = len(g)
            for c, k in g["多数"].value_counts().items():
                lo, hi = wilson(int(k), n)
                rows.append({"意图": code, "快照": SRC_ZH[s], "source": s, "样本n": n, "该快照该意图总行数": int(round(g["抽样权重"].sum())),
                             "子功能": c, "子功能名": name.get(c, c), "条数": int(k), "占比%": round(100 * k / n, 1),
                             "95CI低%": round(100 * lo, 1), "95CI高%": round(100 * hi, 1)})
        comp = pd.DataFrame(rows)
        comp.to_csv(out / f"{code}_composition_by_snapshot.csv", index=False, encoding="utf-8-sig")
        # 全意图合并（按抽样权重还原到总体）
        w = d.groupby("多数")["抽样权重"].sum()
        pooled = (w / w.sum() * 100).round(1).rename("加权占比%").reset_index().assign(子功能名=lambda x: x["多数"].map(name))
        pooled.to_csv(out / f"{code}_composition_pooled_weighted.csv", index=False, encoding="utf-8-sig")
        # 对齐快照对
        prow = []
        for a, b, label in PAIRS:
            ga, gb = d[d.source == a], d[d.source == b]
            if len(ga) < MIN_N or len(gb) < MIN_N:
                continue
            for c in codes + ["无多数"]:
                ka, kb = int((ga["多数"] == c).sum()), int((gb["多数"] == c).sum())
                if ka == 0 and kb == 0:
                    continue
                dd, lo, hi = newcombe(ka, len(ga), kb, len(gb))
                prow.append({"意图": code, "对比": label, "a": SRC_ZH[a], "b": SRC_ZH[b], "n_a": len(ga), "n_b": len(gb),
                             "子功能": c, "子功能名": name.get(c, c), "a%": round(100 * ka / len(ga), 1), "b%": round(100 * kb / len(gb), 1),
                             "差pp(b-a)": round(100 * dd, 1), "95CI低pp": round(100 * lo, 1), "95CI高pp": round(100 * hi, 1),
                             "显著": bool(lo > 0 or hi < 0)})
        pd.DataFrame(prow).to_csv(out / f"{code}_pair_contrasts.csv", index=False, encoding="utf-8-sig")
        # 与叶、L2 的交叉（样本内计数，按抽样权重加权以还原总体）
        for col, tag in (("叶名", "leaf"), ("td_l2", "l2")):
            ct = d.pivot_table(index=col, columns="多数", values="抽样权重", aggfunc="sum", fill_value=0)
            ct = ct.loc[ct.sum(1).sort_values(ascending=False).index]
            share = (ct.div(ct.sum(1), axis=0) * 100).round(1)
            share.insert(0, "加权行数", ct.sum(1).round(0).astype(int))
            share.insert(1, "样本行数", d.groupby(col).size().reindex(share.index).fillna(0).astype(int))
            share.rename(columns=name).to_csv(out / f"{code}_by_{tag}.csv", encoding="utf-8-sig")
        wf = wide[wide["票数完整"]]
        agree_pairs = {f"{x}~{y}": round(float((wf[x] == wf[y]).mean()), 3) for i, x in enumerate(lenses) for y in lenses[i + 1:]}
        summary[code] = {"n_sample": len(d), "codes": codes, "fleiss_kappa": round(kappa, 3), "n_kappa": int(full.sum()),
                         "declared_missing_votes": len(gaps), "repairs": repairs,
                         "unanimous_%": round(100 * float(wf["三票全同"].mean()), 1),
                         "no_majority_rows": int((wide["多数"] == "无多数").sum()), "pairwise_agreement": agree_pairs}
        (out / f"{code}_codebook.json").write_text(json.dumps(cb, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    domain, path = sys.argv[1], sys.argv[2]
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    aggregate(domain, raw if isinstance(raw, list) else raw.get("result", raw))
