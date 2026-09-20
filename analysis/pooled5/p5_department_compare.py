#!/usr/bin/env python
"""科室这一层的逐快照对照：把科室码接回两份医疗语料的每一行，然后用**和意图/叶完全相同的那套统计**
（`matrix` / `absence` / `newcombe_all` / `pairwise_tvd` / `interface_split`）算一遍。

复用那几个函数不是图省事：它们已经被测试和多次交付验证过，而且这样一来科室层的数字与同一份报告里
意图层、叶层的数字是同一口径下算出来的，可以横着读。它们只认 `_key` / `_disp` 两列，所以把科室码
塞进 `_key` 就够了。

纯后处理：不碰运行产物，不改已交付文件，只往 `work/科室/` 写。

    HF_HOME=$(pwd)/.hf .venv/bin/python analysis/pooled5/p5_department_compare.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

QM = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(QM / "analysis/pooled5"))
from pooled5_common import SRC_ZH, SOURCES, load  # noqa: E402
from p5_snapshot_classes import (absence, interface_split, matrix,  # noqa: E402
                                 newcombe_all, pairwise_tvd, surface_groups)
from p5_intent_structure import quotable_mask  # noqa: E402

BASE = QM / "analysis/pooled5/work/科室"
OUT = BASE / "compare"
CORPORA = {"医疗8": "data/raw/pooled5/医疗8_pooled5.parquet",
           "医疗随机": "data/raw/pooled5/医疗随机_pooled5.parquet"}

#: 码 -> (中文名, 一级归并). 一级归并用于「内科整体 vs 外科整体」这种粗口径，明细仍按二级读。
NAME = {
    "NK_HX": ("呼吸内科", "内科"), "NK_XH": ("消化内科", "内科"), "NK_SJ": ("神经内科", "内科"),
    "NK_XX": ("心血管内科", "内科"), "NK_XY": ("血液内科", "内科"), "NK_SN": ("肾内科", "内科"),
    "NK_NF": ("内分泌科", "内科"), "NK_FS": ("风湿免疫科", "内科"),
    "WK_PW": ("普通外科", "外科"), "WK_SJ": ("神经外科", "外科"), "WK_GK": ("骨科", "外科"),
    "WK_MN": ("泌尿外科", "外科"), "WK_XX": ("心胸外科", "外科"), "WK_SZ": ("烧伤整形科", "外科"),
    "FCK_FK": ("妇科", "妇产科"), "FCK_CK": ("产科", "妇产科"), "FCK_SZ": ("生殖医学科", "妇产科"),
    "EK": ("儿科", "儿科"), "YK": ("眼科", "五官"), "EB": ("耳鼻咽喉科", "五官"),
    "KQ": ("口腔科", "五官"), "PF": ("皮肤科", "皮肤"), "PF_XB": ("性病科", "皮肤"),
    "MR": ("医学美容科", "其它临床"), "JS": ("精神心理科", "其它临床"), "GR": ("感染科", "其它临床"),
    "ZL": ("肿瘤科", "其它临床"), "JZ": ("急诊科", "其它临床"), "KF": ("康复医学科", "其它临床"),
    "TT": ("疼痛科", "其它临床"), "ZY": ("中医科", "其它临床"), "NANKE": ("男科", "其它临床"),
    "JC": ("检验影像病理", "其它临床"), "QK": ("全科/综合", "其它临床"),
    "YP_GENERIC": ("药品通用问题", "非临床"), "YS": ("养生保健营养", "非临床"),
    "JY": ("就医事务", "非临床"), "FYL": ("非医疗内容", "非临床"), "WFPD": ("无法判断", "非临床"),
}
NONCLIN = {k for k, v in NAME.items() if v[1] == "非临床"}

#: 每个码在《医疗机构诊疗科目名录》里的对应编号。`matrix` 要一列 `_def`（类目定义），
#: 对科室来说最有用的定义就是「它在国家标准里是哪一条」——读者可以直接去查。
SPEC = {
    "NK_HX": "03.01 呼吸内科", "NK_XH": "03.02 消化内科", "NK_SJ": "03.03 神经内科",
    "NK_XX": "03.04 心血管内科", "NK_XY": "03.05 血液内科", "NK_SN": "03.06 肾病学",
    "NK_NF": "03.07 内分泌", "NK_FS": "03.08/03.09 免疫学与变态反应",
    "WK_PW": "04.01 普通外科", "WK_SJ": "04.02 神经外科", "WK_GK": "04.03 骨科",
    "WK_MN": "04.04 泌尿外科", "WK_XX": "04.05/04.06 胸外科与心脏大血管外科",
    "WK_SZ": "04.07/04.08 烧伤与整形外科",
    "FCK_FK": "05.01 妇科", "FCK_CK": "05.02 产科", "FCK_SZ": "05.05 生殖健康与不孕症",
    "EK": "07/08/09 儿科、小儿外科、儿童保健", "YK": "10 眼科", "EB": "11 耳鼻咽喉科",
    "KQ": "12 口腔科", "PF": "13.01 皮肤病", "PF_XB": "13.02 性传播疾病",
    "MR": "14 医疗美容科", "JS": "15 精神科", "GR": "16/17 传染科与结核病科",
    "ZL": "19 肿瘤科", "JZ": "20 急诊医学科", "KF": "21/22 康复医学科与运动医学科",
    "TT": "27 疼痛科（2007 年增设）", "ZY": "50/51/52 中医科、民族医学科、中西医结合科",
    "NANKE": "名录外——中国医院普遍设置的 de-facto 门诊，见码本偏离说明 2",
    "JC": "30/31/32 医学检验科、病理科、医学影像科", "QK": "02 全科医疗科",
    "YP_GENERIC": "非科室：定位不到疾病的用药问题", "YS": "非科室：养生保健与营养",
    "JY": "非科室：挂号、费用、医保、医院与医生查询", "FYL": "非科室：与医疗无关",
    "WFPD": "非科室：语义不足以判断",
}
assert set(SPEC) == set(NAME), f"SPEC/NAME mismatch: {set(SPEC) ^ set(NAME)}"


def load_labels() -> pd.DataFrame:
    p = BASE / "deepseek" / "labels.csv"
    if not p.exists():
        raise SystemExit(f"{p} missing — run p5_department_label.py first")
    lab = pd.read_csv(p, encoding="utf-8-sig", keep_default_na=False)
    lab = lab[lab["code"].astype(str).ne("")]
    bad = sorted(set(lab["code"]) - set(NAME))
    assert not bad, f"labels carry codes outside the codebook: {bad}"
    return lab.set_index("query")["code"]


def frame(domain: str, code_of: pd.Series) -> pd.DataFrame:
    """语料 + 该域运行交付的逐行标注，即其它几份分析用的同一个 frame。

    不直接读原始 parquet：引用护栏有一层是**类级**的（`NEVER_QUOTE_CLASSES` 按 `td_l1` 拦），
    原始 parquet 没有 `td_l1`，`quotable_mask` 会直接 KeyError。用 `load()` 还有一个好处——
    科室层与意图层落在同一份行上，可以交叉看。"""
    d = load(domain).reset_index(drop=True)
    d["_key"] = d["query"].astype(str).map(code_of)
    miss = int(d["_key"].isna().sum())
    assert miss == 0, f"{domain}: {miss} rows whose query never got a department label"
    d["_disp"] = d["_key"].map(lambda k: NAME[k][0])
    d["_l1"] = d["_key"].map(lambda k: NAME[k][1])
    d["_def"] = d["_key"].map(SPEC)
    return d


#: 逐串筛查名单是**按域**建的，而这份报告把两个医疗域的例子放进同一个文件。实测到的后果：
#: `女性到达顶峰什么症状` 在 医疗8 的名单上（med-pool8 逐串筛查时判为性内容），却作为 医疗随机
#: 的妇科例子印了出来——因为 医疗随机 自己的名单里没有它。两份语料是同一个垂类，写法本来就重叠。
#: 这里取两域名单的并集，只影响本报告；不改任何一个域自己的护栏配置，那属于另一件事（见 HANDOFF §2）。
MED_DOMAINS = ("医疗8", "医疗随机")


def examples(domain: str, d: pd.DataFrame, srcs: list[str], per: int = 3) -> pd.DataFrame:
    """每个科室 × 每个快照最多 `per` 条真实 query，**走与报告完全相同的七层引用护栏**，
    外加两个医疗域筛查名单的并集。护栏拦下的不换成占位符，直接少给——这一张表是给人读的证据，不是配额。"""
    from p5_snapshot_classes import screened_quote_block
    union = set().union(*(screened_quote_block(x) for x in MED_DOMAINS))
    ok = quotable_mask(domain, d).to_numpy() & ~d["query"].astype(str).isin(union).to_numpy()
    rows = []
    for key, g in d.groupby("_key", sort=False):
        for s in srcs:
            gs = g[(g.source == s) & ok[g.index]]
            if not len(gs):
                continue
            gs = gs.sort_values("pv_norm", ascending=False).head(per)
            for _, r in gs.iterrows():
                rows.append({"科室": NAME[key][0], "码": key, "快照": SRC_ZH[s],
                             "query": r["query"], "PV归一": round(float(r["pv_norm"]), 2)})
    return pd.DataFrame(rows)


#: 问句标记 / 第一人称 / 裸词——三个不看语义就能数的形态特征。
_ASK = re.compile(r"[?？]|吗|怎么|如何|为什么|是什么|能不能|可以吗|多久|哪些|怎样|要不要|会不会")
_SELF = re.compile(r"我|我的|自己|宝宝|孩子|老公|老婆|我家|家里")


def form_by_surface(d: pd.DataFrame) -> pd.DataFrame:
    """每个科室 × 每个界面的**问法形态**：串长、问句率、第一人称率。

    这是「为什么某个科室偏向某个界面」的证据，而不是解释。界面差异本身只说明分布不同；
    要说明白**为什么**，得看两边的人问法哪里不一样——这三个量都是数出来的，不需要判断。
    """
    rows = []
    q = d["query"].astype(str)
    d = d.assign(_len=q.str.len(), _ask=q.map(lambda s: bool(_ASK.search(s))),
                 _self=q.map(lambda s: bool(_SELF.search(s))))
    for key, g in d.groupby("_key", sort=False):
        rec = {"科室": NAME[key][0], "码": key, "总行数": len(g)}
        for surf in ("搜索", "AI助手"):
            gs = g[g.surface == surf]
            rec[f"{surf}_行数"] = len(gs)
            rec[f"{surf}_中位串长"] = round(float(gs["_len"].median()), 1) if len(gs) else None
            rec[f"{surf}_问句%"] = round(100 * float(gs["_ask"].mean()), 1) if len(gs) else None
            rec[f"{surf}_第一人称%"] = round(100 * float(gs["_self"].mean()), 1) if len(gs) else None
        rows.append(rec)
    t = pd.DataFrame(rows).sort_values("总行数", ascending=False)
    for c in ("中位串长", "问句%", "第一人称%"):
        t[f"差_{c}"] = (t[f"AI助手_{c}"] - t[f"搜索_{c}"]).round(1)
    return t


#: 医疗8 的八个快照不是一个维度上的八个点，而是三条轴各自的一对。把它们配成对来读，
#: 才能说「这是时间变化」还是「这是流量分层」还是「这是说话还是打字」——
#: 泛泛地两两比 TVD 会把三种完全不同的东西混成一张表。
AXES = {
    "医疗8": [("时间（随机层，口径可比）", "2025search_rand", "2026search_rand"),
             ("时间（头部层，PV 排序）", "2025search", "2026search"),
             ("流量分层（2026 搜索）", "2026search", "2026search_rand"),
             ("助手内分层", "assistant_top", "assistant_random"),
             ("语音 vs 文字（助手头部）", "assistant_voice_top", "assistant_top"),
             ("语音内分层", "assistant_voice_top", "assistant_voice")],
    "医疗随机": [("界面（同一天，都是随机 1w）", "msearch_2609r", "mai_2609r")],
}


def axis_table(domain: str, d: pd.DataFrame) -> pd.DataFrame:
    """按轴配对：每个科室在这一对快照上的占比差（百分点）。正数 = 在后一个快照里更多。"""
    rows = []
    for label, a, b in AXES.get(domain, []):
        if not ((d.source == a).any() and (d.source == b).any()):
            continue
        na, nb = int((d.source == a).sum()), int((d.source == b).sum())
        sa = d.loc[d.source == a, "_key"].value_counts(normalize=True)
        sb = d.loc[d.source == b, "_key"].value_counts(normalize=True)
        # 列名写成通用的「前/后」，快照名另开两列。一轴一套专名列会让表变成一片 NaN：
        # 六条轴 × 两个快照 = 十二列，每一行只有两格有值。
        for key in sorted(set(sa.index) | set(sb.index)):
            pa, pb = 100 * float(sa.get(key, 0.0)), 100 * float(sb.get(key, 0.0))
            rows.append({"轴": label, "科室": NAME[key][0], "码": key,
                         "前快照": SRC_ZH[a], "后快照": SRC_ZH[b],
                         "前%": round(pa, 2), "后%": round(pb, 2),
                         "差_pp": round(pb - pa, 2), "n_前": na, "n_后": nb})
    t = pd.DataFrame(rows)
    return t.reindex(t["差_pp"].abs().sort_values(ascending=False).index) if len(t) else t


def legacy_vs_model(d: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """医疗随机 独有：平台的 `legacy_dept` 与本层的对照。这张表测的是**平台标签**，不是本层。"""
    t = d.copy()
    t["legacy"] = t["legacy_dept"].astype(str)
    ct = pd.crosstab(t["legacy"], t["_disp"])
    empty = t["legacy"].isin(["(无科室)", "无明确科室"])
    onc = t["_key"].eq("ZL")
    facts = {
        "legacy_空档行": int(empty.sum()), "legacy_空档占比%": round(100 * float(empty.mean()), 2),
        "legacy_取值数": int(t["legacy"].nunique()),
        "本层判为肿瘤科的行": int(onc.sum()),
        "其中legacy给的标签": t.loc[onc, "legacy"].value_counts().to_dict(),
        "legacy内科被本层拆成": t.loc[t["legacy"] == "内科", "_disp"].value_counts().head(10).to_dict(),
        "legacy空档被本层救回临床科室的行": int((empty & ~t["_key"].isin(NONCLIN)).sum()),
        "legacy空档确实非临床的行": int((empty & t["_key"].isin(NONCLIN)).sum()),
    }
    return ct, facts


def rule_vs_model(code_of: pd.Series) -> tuple[pd.DataFrame, dict]:
    r = pd.read_csv(BASE / "rule_labels.csv", encoding="utf-8-sig", keep_default_na=False)
    r["model"] = r["query"].astype(str).map(code_of)
    spoke = r[~r["rule_code"].str.startswith("ABSTAIN") & r["model"].notna()].copy()
    spoke["agree"] = spoke["rule_code"] == spoke["model"]
    by = (spoke.groupby("rule_code")
          .agg(词表判此科室的串数=("agree", "size"), 与模型一致=("agree", "sum")).reset_index())
    by["一致率%"] = (100 * by["与模型一致"] / by["词表判此科室的串数"]).round(1)
    by["科室"] = by["rule_code"].map(lambda k: NAME.get(k, (k,))[0])
    facts = {"词表开口的串数": int(len(spoke)), "词表开口占比%": round(100 * len(spoke) / len(r), 1),
             "整体一致率%": round(100 * float(spoke["agree"].mean()), 1),
             "词表弃权_无证据": int((r["rule_code"] == "ABSTAIN_NO_EVIDENCE").sum()),
             "词表弃权_歧义": int((r["rule_code"] == "ABSTAIN_AMBIGUOUS").sum())}
    return by.sort_values("词表判此科室的串数", ascending=False), facts


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    code_of = load_labels()
    # 标注模型不写死：它换过一次（pro 太慢，实测 227 串/分钟），facts 必须说的是**实际跑的那个**。
    from p5_department_label import PROVIDERS
    facts: dict = {"码本": "analysis/pooled5/work/科室/codebook.md",
                   "标注模型": PROVIDERS["deepseek"]["model"],
                   "标注口径": "每个不同 query 一个码；码本预注册，锚在《医疗机构诊疗科目名录》"}
    for domain in CORPORA:
        d = frame(domain, code_of)
        srcs = [s for s in SOURCES if (d.source == s).any()]
        se, asr = surface_groups(d, srcs)
        m = matrix(d, srcs)
        m.insert(1, "一级归并", m["key"].map(lambda k: NAME[k][1]))
        m.to_csv(OUT / f"dept_matrix_{domain}.csv", index=False, encoding="utf-8-sig")
        absence(d, srcs).to_csv(OUT / f"dept_absence_{domain}.csv", index=False, encoding="utf-8-sig")
        newcombe_all(d, srcs).to_csv(OUT / f"dept_newcombe_{domain}.csv", index=False, encoding="utf-8-sig")
        pairwise_tvd(d, srcs, tag=f"{domain}|dept").to_csv(OUT / f"dept_tvd_{domain}.csv", index=False, encoding="utf-8-sig")
        interface_split(d, srcs).to_csv(OUT / f"dept_interface_{domain}.csv", index=False, encoding="utf-8-sig")
        examples(domain, d, srcs).to_csv(OUT / f"dept_examples_{domain}.csv", index=False, encoding="utf-8-sig")
        form_by_surface(d).to_csv(OUT / f"dept_form_{domain}.csv", index=False, encoding="utf-8-sig")
        ax = axis_table(domain, d)
        if len(ax):
            ax.to_csv(OUT / f"dept_axis_{domain}.csv", index=False, encoding="utf-8-sig")
        clin = ~d["_key"].isin(NONCLIN)
        facts[domain] = {
            "行数": int(len(d)), "快照": {SRC_ZH[s]: int((d.source == s).sum()) for s in srcs},
            "科室数_出现": int(d.loc[clin, "_key"].nunique()), "临床行占比%": round(100 * float(clin.mean()), 2),
            "搜索侧快照": [SRC_ZH[s] for s in se], "助手侧快照": [SRC_ZH[s] for s in asr],
            "全语料前十科室": (100 * d.loc[clin, "_disp"].value_counts(normalize=True).head(10)).round(2).to_dict(),
            "非临床构成%": (100 * d.loc[~clin, "_disp"].value_counts(normalize=True)).round(2).to_dict(),
            "一级归并%": (100 * d["_l1"].value_counts(normalize=True)).round(2).to_dict(),
        }
        if domain == "医疗随机":
            ct, lf = legacy_vs_model(d)
            ct.to_csv(OUT / "legacy_vs_model.csv", encoding="utf-8-sig")
            facts["legacy对照"] = lf
        print(f"{domain}: {len(d):,} 行 · {facts[domain]['科室数_出现']} 个临床科室 · "
              f"临床 {facts[domain]['临床行占比%']}%")
    by, rf = rule_vs_model(code_of)
    by.to_csv(OUT / "rule_vs_model.csv", index=False, encoding="utf-8-sig")
    facts["规则仪器对照"] = rf
    (OUT / "facts.json").write_text(json.dumps(facts, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"规则仪器：开口 {rf['词表开口占比%']}%，与模型一致 {rf['整体一致率%']}%")
    print(f"→ {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
