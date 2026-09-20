#!/usr/bin/env python
"""医疗3：把 health-pool3 的两个快照与新的「健康管家 Top1w」拼成一份三快照语料。

三个快照都是 **2026-09-14 同一天**，这是这份语料最值钱的地方：

| 快照 | 界面 | 抽样 | 来源 |
|---|---|---|---|
| 传统搜索随机1w | 搜索 | 随机 | health-pool3 原样 |
| 健康管家随机1w | AI助手 | 随机 | health-pool3 原样 |
| 健康管家Top1w | AI助手 | **头部（PV 排序）** | 本次新增 |

于是「界面」（搜索 vs 助手，随机层对随机层）与「流量分层」（同一个助手，头部对随机）这两条轴
第一次可以互相独立地读——health-pool3 只有前者，医疗8 的头部/随机对只在搜索侧。

**清洗口径与 health-pool3 逐条相同**，规则直接 import 自 `build_medrand_corpus`（不是抄一份）：
包装模板只剥不删、宽口径规则只用来挑要盲标的行、只有三个窄规则能删没盲标过的行、三票一致才删。
新快照的 989 个歧义串重跑了同一套三视角盲标（码本重建并做了 300 串校准，见 `work/医疗3/audit_summary.json`）。

**老快照的行一个字都不动**，标签也直接取运行交付的 `labels_full.csv`；只有新快照走
`med3_score.py` 的打分路径。

    HF_HOME=$(pwd)/.hf .venv/bin/python analysis/pooled5/build_med3_corpus.py          # 第一步：出 new_rows.csv
    HF_HOME=$(pwd)/.hf .venv/bin/python analysis/pooled5/med3_score.py                 # 第二步：打标签
    HF_HOME=$(pwd)/.hf .venv/bin/python analysis/pooled5/build_med3_corpus.py --assemble   # 第三步：拼语料
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_medrand_corpus as B  # noqa: E402
import clean_assistant_functional as C  # noqa: E402
import build_health_corpus as H  # noqa: E402

BASE = ROOT / "analysis/pooled5/work/医疗3"
DOMAIN = "医疗3"
NEW = {"file": "健康管家-医疗-Top1w.xlsx", "q": "query", "pv": "total_pv",
       "l1": "query_level1_type", "l2": "query_level2_type", "l3": "query_level3_type",
       "surface": "AI助手", "zh": "健康管家Top1w", "product": "健康管家"}
SRC = "mai_2609t"
FACTS: dict = {}


def tier_ai_new(t: pd.DataFrame) -> pd.DataFrame:
    """与 `build_medrand_corpus.tier_ai` 同一套规则与同一条判定链，只是读本批自己的盲标结果。

    没有直接复用那个函数，因为它把盲标文件的路径写死在模块里；规则本身（H/C 里的词表与正则）
    是 import 来的同一份对象，不是副本——这一点比复用函数更重要。
    """
    raw = t["query_raw"]
    t["wrapper_stripped"] = raw.str.startswith(B.WRAPPER).values
    t["query"] = raw.str.replace(rf"^{B.WRAPPER}", "", regex=True).str.strip()
    empty = t["query"].str.len() == 0
    t.loc[empty, "query"] = raw[empty]
    q = t["query"]
    t["flag_C1"] = (C.content_free_flags(q, surface="assistant") | q.str.match(H.ACK_HEALTH)).values
    t["flag_H5"] = q.str.contains(H.DOCTOR_CARD_AI).values
    t["flag_H2"] = (q.isin(H.FEATURE_HEALTH | H.FACET_HEALTH) | q.str.contains(H.TOOL) | q.str.match(H.FACET)).values
    t["flag_H1"] = q.map(H.intake_answer).values
    t["rule_tier"] = "user"
    for flag, tier in [("flag_H1", "H1_intake_answer"), ("flag_H2", "H2_feature_or_facet"),
                       ("flag_H5", "H5_doctor_card"), ("flag_C1", "C1_content_free")]:
        t.loc[t[flag], "rule_tier"] = tier

    NARROW = {"H2_feature_or_facet", "H5_doctor_card", "C1_content_free"}
    ap = BASE / "audit_labels.csv"
    assert ap.exists(), f"missing {ap} — run analysis/pooled5/med3_audit.py first"
    a = pd.read_csv(ap, encoding="utf-8-sig")
    a = a[a["kind"] == "new"].drop_duplicates("query").set_index("query")
    t["audit_votes"] = t["query"].map(a["votes"]).fillna("").values
    t["audit_majority"] = t["query"].map(a["majority"]).fillna("").values
    unanimous_product = t["audit_votes"].map(lambda v: len(v) == 3 and len(set(v)) == 1 and v[0] != "U")
    audited = t["audit_votes"].str.len() == 3
    t["tier"] = np.where(audited | t["rule_tier"].isin(NARROW), t["rule_tier"], "user")
    t.loc[unanimous_product, "tier"] = t.loc[unanimous_product, "audit_majority"].map(B.AUDIT_TIER)
    audit_user = t["audit_votes"].map(lambda v: len(v) == 3 and len(set(v)) == 1 and v[0] == "U")
    t.loc[audit_user, "tier"] = "user"
    split = t["audit_votes"].map(lambda v: len(v) == 3 and len(set(v)) > 1)
    t.loc[split, "tier"] = "user"
    t["tier_source"] = np.where(unanimous_product | audit_user, "audit_unanimous",
                                np.where(split, "audit_split_kept", "rule"))
    FACTS["ai_audit_new"] = {
        "audited": int(audited.sum()), "unanimous_product": int(unanimous_product.sum()),
        "unanimous_user": int(audit_user.sum()), "split_kept_as_user": int(split.sum()),
        "rule_only_rows": int((~audited).sum()),
        "rule_removed_outside_audit": int(((~audited) & (t["tier"] != "user")).sum()),
        "wrapper_rows": int(t["wrapper_stripped"].sum()),
        "wrapper_share%": round(100 * float(t["wrapper_stripped"].mean()), 1),
    }
    return t


def read_new() -> pd.DataFrame:
    B.SPEC[SRC] = NEW
    return tier_ai_new(B._read(SRC))


def step_prepare() -> int:
    BASE.mkdir(parents=True, exist_ok=True)
    t = read_new()
    t.to_parquet(BASE / "new_all_rows.parquet", index=False)
    user = t[t["tier"] == "user"].reset_index(drop=True)
    user[["query"]].to_csv(BASE / "new_rows.csv", index=False, encoding="utf-8-sig")
    FACTS["new_snapshot"] = {"rows_raw": int(len(t)), "rows_mined": int(len(user)),
                             "removed": int(len(t) - len(user)),
                             "tiers": t["tier"].value_counts().to_dict()}
    (BASE / "build_facts.json").write_text(json.dumps(FACTS, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(FACTS, ensure_ascii=False, indent=1))
    print(f"→ {BASE / 'new_rows.csv'}  ({len(user):,} rows to score)")
    return 0


def step_assemble() -> int:
    """三快照语料 + 逐行标签。老两个快照原样搬运，新快照接上打分结果。"""
    old = pd.read_parquet(ROOT / "data/raw/pooled5/医疗随机_pooled5.parquet")
    oldlab = pd.read_csv(ROOT / "runs/health-pool3/gen01/labels_full.csv", encoding="utf-8-sig")
    assert len(old) == len(oldlab), f"corpus {len(old)} vs labels {len(oldlab)}"
    assert (old["query"].astype(str).values == oldlab["query"].astype(str).values).all(), \
        "the delivered labels are not row-aligned with the corpus — refuse to join"

    new_all = pd.read_parquet(BASE / "new_all_rows.parquet")
    lab = pd.read_csv(BASE / "new_labels.csv", encoding="utf-8-sig", keep_default_na=False)
    user = new_all[new_all["tier"] == "user"].reset_index(drop=True)
    assert len(user) == len(lab) and (user["query"].astype(str).values == lab["query"].astype(str).values).all(), \
        "scored rows are not aligned with the mined rows"

    keep = [c for c in old.columns if c in user.columns]
    newc = user[keep].copy()
    newc["source"] = SRC
    newc["source_zh"] = NEW["zh"]
    newc["surface"] = NEW["surface"]
    newc["product"] = NEW["product"]
    newc["source_file"] = NEW["file"]
    newc["domain"] = "医疗"
    # PV 归一到 10,000，与其它快照同口径（快照大小差一个量级，原始 PV 不可比）
    newc["pv_norm"] = 10000 * newc["pv_raw"] / newc["pv_raw"].sum()
    # `row_id` / `row_id_all` 是构建时才有的行号，新快照的 all_rows 里没有：row_id 在本快照内重排，
    # row_id_all 在拼完之后统一重排，两者都不能从 old 的列表里硬取。
    newc["row_id"] = np.arange(len(newc))
    newc["row_id_all"] = -1
    missing = [c for c in old.columns if c not in newc.columns]
    assert not missing, f"the new snapshot is missing corpus columns: {missing}"
    corpus = pd.concat([old, newc[old.columns]], ignore_index=True)
    corpus["row_id_all"] = np.arange(len(corpus))
    out = ROOT / "data/raw/pooled5/医疗3_pooled5.parquet"
    corpus.to_parquet(out, index=False)

    cols = ["bu_leaf", "bu_leaf_name", "bu_family_final", "bu_margin", "bu_ambiguous",
            "td_l1", "td_confidence", "td_margin", "td_ambiguous", "td_l2"]
    newlab = lab[["query"] + [c for c in cols if c in lab.columns]].copy()
    # 名称列不能留空：`td_l1_name` 空着会以 NaN 读回来，下游 sorted() 直接在 float 与 str 之间比大小挂掉。
    # 这些名字本来就是**体系的属性**，不是行的属性——同一个码在哪个快照都是同一个中文名，
    # 所以直接从交付标签里按码查回来，而不是造一个新名字。
    for code_col, name_col in (("td_l1", "td_l1_name"), ("td_l1", "td_user_need"),
                               ("bu_leaf", "bu_user_need")):
        m = oldlab.drop_duplicates(code_col).set_index(code_col)[name_col]
        newlab[name_col] = newlab[code_col].map(m).fillna("")
    lm = oldlab.drop_duplicates("bu_leaf").set_index("bu_leaf")["bu_leaf_name"]
    newlab["bu_leaf_name"] = newlab["bu_leaf"].map(lm).fillna(newlab.get("bu_leaf_name", "")).fillna("")
    # 打分器给不出「治理前」的叶/家族——那是 p8 改写之前的状态，只有真跑过的行才有。写成最终值会撒谎，
    # 所以留空并在说明里写清楚：新快照没有治理前后的对照。
    newlab["bu_leaf_pre_governance"] = ""
    newlab["bu_family_pre_governance"] = ""
    newlab["td_decided_by"] = "scored_by_health-pool3_model"
    for c, src_col in (("ref_legacy_l2", "legacy_l2"), ("ref_legacy_dept", "legacy_dept"),
                       ("ref_legacy_type", "legacy_type")):
        newlab[c] = user[src_col].astype(str).values if src_col in user.columns else ""
    for c in oldlab.columns:
        if c not in newlab.columns:
            newlab[c] = ""
    newlab["snapshot"] = NEW["zh"]
    newlab["run_id"] = "health-pool3(scored)"
    newlab["generation"] = oldlab["generation"].iloc[0]
    labels = pd.concat([oldlab, newlab[oldlab.columns]], ignore_index=True)
    lout = BASE / "labels_full_3.csv"
    labels.to_csv(lout, index=False, encoding="utf-8-sig")

    facts = json.loads((BASE / "build_facts.json").read_text(encoding="utf-8"))
    facts["assembled"] = {"rows": int(len(corpus)),
                          "by_source": corpus["source"].value_counts().to_dict(),
                          "old_rows_untouched": int(len(old)),
                          "label_columns": list(labels.columns)}
    (BASE / "build_facts.json").write_text(json.dumps(facts, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"→ {out}  ({len(corpus):,} rows)")
    print(corpus.groupby(["source_zh", "surface"]).size().to_string())
    print(f"→ {lout}")
    return 0


if __name__ == "__main__":
    raise SystemExit(step_assemble() if "--assemble" in sys.argv else step_prepare())
