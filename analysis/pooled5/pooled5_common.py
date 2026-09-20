# -*- coding: utf-8 -*-
"""Shared loader and statistics for the POOLED-5 analysis.

THE JOIN IS POSITIONAL AND IT IS CHECKED. `labels_full.csv` preserves input row order exactly
(verified 20,000/20,000 on fin-pool, and asserted again here per domain against the query text).
The five-level `source` column never entered the run — declaring it would have asked the K
locator to find a partition that separates the surfaces, which is the comparison, not the frame.

Every share in this analysis is WITHIN SOURCE. Raw counts across sources are meaningless here:
search contributes 10,000 rows per year and the assistant ~1,000 per stratum, so a class's raw
count is mostly a statement about which export it came from.
"""
from __future__ import annotations
import math, os
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
WORK = Path(os.environ.get("P5_WORKDIR", Path(__file__).resolve().parent / "work"))
DOMAINS = {"金融": "fin", "医疗": "med", "教育": "edu", "影视": "film", "人物": "ppl",
           "书籍文档": "book", "软件": "soft", "金融8": "fin8", "健康": "health", "医疗8": "med8",
           "人物8": "ppl8", "影视8": "film8", "医疗随机": "medr", "医疗3": "med3"}
#: `金融8` 是同一个金融垂类、8 个快照的那一份语料（2026-09-14 新到三份导出：搜索两年的随机 1w、
#: 语音头部 1k）。它**不是**替换 `金融`——`data/raw/pooled5/金融_pooled5.parquet` 与已交付的
#: fin-pool5 的 `labels_full.csv` 是逐行断言绑定的，覆盖它会让那一份运行的每一张表都读不出来。
#: 两份语料并存，`金融8` 写在 `金融8_pooled5.parquet`。
#:
#: 新增的三个 source 追加在各自年份/界面的后面，**没有动已有五个的相对次序**，所以缺这三个的
#: 六个域列序一个字节都不变（已逐字节验证）。
SOURCES = ["2025search", "2025search_rand", "2026search", "2026search_rand",
           "assistant_top", "assistant_random", "assistant_voice_top", "assistant_voice",
           # 健康（2026-09-14）：同一周的两份周榜，按 query 合并过。追加在末尾，其它域列序不变。
           "hsearch_2609w", "hai_2609w",
           # 人物8 / 影视8（2026-09-16）：这两个域的第二份语音导出**自带 PV、是随机 1k**（医疗8 那份没有 PV、
           # 抽样方式未知，所以只能叫 assistant_voice）。同样追加在末尾：没有这个 source 的域列序一个字节不变。
           "assistant_voice_random",
           # 医疗随机（2026-09-16）：同一天的传统搜索随机 1 万与健康管家随机 1 万，都是**去重后的随机抽样**，
           # 与 健康 那一对（周榜 top1w）形状不同，所以是自己的两个 source。
           "msearch_2609r", "mai_2609r",
           # 医疗3（2026-09-17）：健康管家同一天的**头部** 1w，health-pool3 当时缺的那一层。
           "mai_2609t"]

#: **交付批次。** 跨领域产物必须说出自己是哪一批的。
#:
#: 2026-09-13 晚加入 书籍文档 / 软件 两个垂类以后，`DOMAINS` 有 7 个，而已交付的
#: 《POOLED5_2026_五域同体系对比》和它引用的 `work/cross/*.csv` 全部是**五域**的。
#: 如果 `available()` 直接返回「跑完的所有领域」，那么任何一次重跑 `p5_cross.py` /
#: `p5_tvd_ci.py` / `p5_report_tables.py` 都会把那份已交付报告底下的表换成七域的，
#: 数字全变而正文不变——上一轮已经踩过一次同类的坑（`p5_vs_unified.py 人物` 用单域结果
#: 覆盖了跨域表，T10 带着 1 个领域发了出去）。
#:
#: 所以：跨领域脚本一律**按批次**取域，并且各批次的 cross 目录分开。默认批次是 `pool5`，
#: 也就是已交付的那五个——不设环境变量时行为与加这两个域之前完全一致。
COHORTS = {
    "pool5": ["金融", "医疗", "教育", "影视", "人物"],
    "new2": ["书籍文档", "软件"],
    "all7": ["金融", "医疗", "教育", "影视", "人物", "书籍文档", "软件"],
    "fin8": ["金融8"],
    "health2": ["健康"],
    # 医疗8（2026-09-15）：同一个医疗垂类的 8 快照语料，与 `金融8` 同构；`医疗_pooled5.parquet` 不动。
    "med8": ["医疗8"],
    # 人物8 / 影视8（2026-09-16）：各自的 8 快照语料，与 `金融8` / `医疗8` 同构；pool5 的两份 parquet 不动。
    "ppl8": ["人物8"],
    "film8": ["影视8"],
    # 医疗随机（2026-09-16）：health-pool2 的两个产品、同一类目，但换成去重后的随机 1 万——那一次助手侧清洗后只剩 225 行。
    "medr": ["医疗随机"], "med3": ["医疗3"],
}
COHORT = os.environ.get("P5_COHORT", "pool5")
#: **短名只是列头，口径以报告开头的「快照总表」为准。** 尤其两条容易读错的：
#: `2025搜索` / `2026搜索` 一直是**按 PV 排序的头部 1 万**（新加的随机 1 万另有其名）；
#: `助手语音1k` 是**没有 PV 字段**的那份导出，抽样方式未知，所以不叫「随机」——
#: 只有 `助手语音头部1k` 是可验证的头部（它自带 wise_pv）。
SRC_ZH = {"2025search": "2025搜索", "2025search_rand": "2025搜索随机1w",
          "2026search": "2026搜索", "2026search_rand": "2026搜索随机1w",
          "assistant_top": "助手头部1k", "assistant_random": "助手随机1k",
          "assistant_voice_top": "助手语音头部1k", "assistant_voice": "助手语音1k",
          "hsearch_2609w": "健康搜索周榜", "hai_2609w": "健康AI管家周榜",
          "assistant_voice_random": "助手语音随机1k",
          "msearch_2609r": "传统搜索随机1w", "mai_2609r": "健康管家随机1w", "mai_2609t": "健康管家Top1w"}
# The contrasts the study is about, each with the ONE thing that differs.
CONTRASTS = [("2025search", "2026search", "时间（搜索内部，2025→2026）"),
             ("2026search", "assistant_top", "界面（同为头部流量）"),
             ("assistant_top", "assistant_random", "深度（助手头部→长尾）"),
             ("2026search", "assistant_random", "界面+深度（不可单独归因）"),
             ("assistant_top", "assistant_voice", "输入方式（打字头部→语音）"),
             ("assistant_random", "assistant_voice", "输入方式（打字长尾→语音）"),
             # 健康：只有一对，而且两边是同一周——这是本批次唯一的对比，它变的是界面（与产品）
             ("hsearch_2609w", "hai_2609w", "界面（同一周：健康搜索 → 专科健康 AI 管家）")]


#: Which run supplies each domain's labels. 人物 points at the RELAUNCH: `ppl-pool5` halted at
#: p7_audit (risk sentinel refused by the provider content filter, and the fallback object had
#: no `model_dump`), and its gen02 resume halted again at the branch-join guard. `ppl-pool5b` is
#: the same corpus with every Zhipu-assigned role routed to another lab — it recovered the fifth
#: research angle (12 candidates vs 0) and a real independent risk sweep (8 findings). The rule
#: for choosing it was written down before the results were visible: work/people_run_choice.md.
RUN_ID = {"金融": "fin-pool5", "医疗": "med-pool5", "教育": "edu-pool5",
          "影视": "film-pool5", "人物": "ppl-pool5b",
          "书籍文档": "book-pool5", "软件": "soft-pool5", "金融8": "fin-pool8", "健康": "health-pool2", "医疗8": "med-pool8",
          "人物8": "ppl-pool8b", "影视8": "film-pool8", "医疗随机": "health-pool3",
          # 医疗3 不是一次运行：它是 health-pool3 的两个快照 + 用该运行自己的
          # 分类器/叶心给新快照打分后拼起来的三快照语料。目录名带 scored 就是为了不被误读。
          "医疗3": "health-pool3_scored3snap"}



def work_file(name: str) -> Path:
    """A per-batch path under work/. `pool5` keeps the original filename so the delivered
    five-domain references stay valid; every other batch gets a suffix. Added after the `new2`
    dry-run wrote five-domain semantic-NN rows into the new batch's summary."""
    if COHORT == "pool5":
        return WORK / name
    stem, _, ext = name.rpartition(".")
    return WORK / f"{stem}_{COHORT}.{ext}"

def run_dir(domain: str, gen: str = "gen01") -> Path:
    """`P5_GEN_<批次>` 覆盖代次，和 `P5_RUN_<批次>` 覆盖 run id 是一对。
    需要它是因为一次运行的**交付代次不一定是 gen01**：ppl-pool8 的 gen01 把标注员漏标的 22 行
    当成了第 19 个 L1 类训练，修好之后重跑在 gen03，后处理必须读 gen03 而不是 gen01。默认不变。"""
    run = os.environ.get("P5_RUN_" + DOMAINS[domain], RUN_ID[domain])
    return ROOT / f"runs/{run}/{os.environ.get('P5_GEN_' + DOMAINS[domain], gen)}"


def cohort(name: str | None = None) -> list[str]:
    """The domains belonging to a delivery batch. See COHORTS for why this exists."""
    key = name or COHORT
    if key not in COHORTS:
        raise SystemExit(f"unknown cohort {key!r}; known: {sorted(COHORTS)}")
    return list(COHORTS[key])


def cross_dir(name: str | None = None) -> Path:
    """Where a batch's cross-domain tables live. `pool5` keeps the original `work/cross/` path so
    the delivered report's references stay valid; every other batch gets its own directory."""
    key = name or COHORT
    d = WORK / ("cross" if key == "pool5" else f"cross_{key}")
    d.mkdir(parents=True, exist_ok=True)
    return d


def available(gen: str = "gen01", name: str | None = None) -> list[str]:
    """Domains **in this batch** whose run has finished and written labels. Scripts that sweep a
    batch must skip the unfinished ones rather than dying on a missing file halfway through."""
    return [d for d in cohort(name) if (run_dir(d, gen) / "labels_full.csv").exists()]


def load(domain: str, gen: str = "gen01") -> pd.DataFrame:
    """Mined rows with their labels AND their source. One row per input row, same order."""
    src = pd.read_parquet(ROOT / f"data/raw/pooled5/{domain}_pooled5.parquet")
    lab = pd.read_csv(run_dir(domain, gen) / "labels_full.csv", encoding="utf-8-sig")
    if len(lab) != len(src):
        raise SystemExit(f"{domain}: {len(lab)} labelled rows vs {len(src)} input rows — the "
                         f"positional join is not safe; do not fall back to a text join, a "
                         f"string that appears in two sources would bind to the wrong one")
    mism = (lab["query"].astype(str).values != src["query"].astype(str).values).sum()
    if mism:
        raise SystemExit(f"{domain}: {mism} rows where the labelled text differs from the input text")
    out = pd.concat([src.reset_index(drop=True), lab.drop(columns=["query"]).reset_index(drop=True)], axis=1)
    out["source_zh"] = out["source"].map(SRC_ZH)
    return out


def all_rows(domain: str) -> pd.DataFrame:
    """Every row INCLUDING the ones the cleaning tiers removed before mining."""
    return pd.read_parquet(WORK / f"{domain}_all_rows.parquet")


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    z, p = 1.959964, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def newcombe(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float, float]:
    """Difference in proportions (2 minus 1) with a Newcombe hybrid-score 95% interval."""
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    d = k2 / n2 - k1 / n1 if n1 and n2 else 0.0
    return (d, d - math.sqrt((k2 / n2 - l2) ** 2 + (u1 - k1 / n1) ** 2),
            d + math.sqrt((u2 - k2 / n2) ** 2 + (k1 / n1 - l1) ** 2))


def cramers_v(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=float)
    n = counts.sum()
    if n == 0 or min(counts.shape) < 2:
        return 0.0
    exp = counts.sum(1, keepdims=True) @ counts.sum(0, keepdims=True) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.nansum(np.where(exp > 0, (counts - exp) ** 2 / exp, 0.0))
    return float(math.sqrt((chi2 / n) / (min(counts.shape) - 1)))


def shares(df: pd.DataFrame, label_col: str, weight: str = "pv_norm") -> pd.DataFrame:
    """Row share and traffic share of every class within every source, with Wilson bounds."""
    rows = []
    for src, g in df.groupby("source", sort=False):
        n = len(g)
        w = g[weight].sum()
        for lab, gg in g.groupby(label_col, sort=False):
            lo, hi = wilson(len(gg), n)
            rows.append({"source": src, "label": lab, "k": len(gg), "n": n,
                         "row_share": len(gg) / n, "lo": lo, "hi": hi,
                         "pv_share": (gg[weight].sum() / w) if w else float("nan")})
    return pd.DataFrame(rows)


def diff_table(df: pd.DataFrame, label_col: str, a: str, b: str) -> pd.DataFrame:
    """Every class's change from source a to source b, with a Newcombe interval."""
    ga, gb = df[df.source == a], df[df.source == b]
    na, nb = len(ga), len(gb)
    if not na or not nb:
        return pd.DataFrame()
    ca, cb = ga[label_col].value_counts(), gb[label_col].value_counts()
    rows = []
    for lab in sorted(set(ca.index) | set(cb.index)):
        ka, kb = int(ca.get(lab, 0)), int(cb.get(lab, 0))
        d, lo, hi = newcombe(ka, na, kb, nb)
        rows.append({"label": lab, "a": a, "b": b, "k_a": ka, "k_b": kb, "n_a": na, "n_b": nb,
                     "share_a": ka / na, "share_b": kb / nb, "diff_pp": 100 * d,
                     "lo_pp": 100 * lo, "hi_pp": 100 * hi,
                     "kind": ("emergent" if ka == 0 and kb else "receded" if kb == 0 and ka else "stable"),
                     "sig": (lo > 0 or hi < 0)})
    return pd.DataFrame(rows).sort_values("diff_pp")


FORM = {
    "疑问句": r"(怎么|如何|为什么|什么|哪个|哪家|哪些|多少|吗|呢|是不是|能不能|可不可以|\?|？)",
    "祈使/委托": r"(帮我|帮忙|请你|请帮|给我(写|画|生成|做|翻译|总结|出|来)|(写|画|生成|做|翻译|总结|改写|续写|润色)(一|个|篇|份|张|下|成|出)|^(写|画|生成|翻译|总结|改写|续写|润色).{2,})",
    "第一人称": r"(我|我的|我家|我们|本人|俺)",
    "称呼对方": r"(你|您)",
    "家庭成员": r"(孩子|儿子|女儿|宝宝|宝贝|老公|老婆|男朋友|女朋友|男友|女友|我妈|我爸|妈妈|爸爸|父母|老人|家人|婆婆|公公|闺女|对象)",
    "身份/人生阶段": r"(学生|同学|老师|家长|宝妈|孕妇|怀孕|备孕|退休|高考|中考|考研|考公|上班|打工|公司|老板|员工|单位|领导|同事|客户)",
    "情绪词": r"(焦虑|害怕|担心|难受|崩溃|烦死|伤心|委屈|生气|郁闷|痛苦|着急|心累|绝望)",
    "是非核实": r"(是不是|是否|真的|吗[？?。，,]?$|对吗|对不对)",
    "方法/怎么办": r"(怎么(做|办|弄|用|治|写|查|算|去|买|选|设置|处理|才能|吃|喝|调理|消除|恢复)|如何|怎样|咋|怎么办)",
    "选择清单": r"(哪个|哪些|哪家|哪种|哪里|哪儿)",
    "定义": r"(什么意思|是什么|啥意思|指什么)",
    "时效词": r"(今天|今日|明天|现在|最新|刚刚|实时|最近|昨天)",
    "输出约束": r"(\d+字|字数|字左右|格式|表格|简短|一句话|分点|分条|比例为|不超过|以内)",
    "导航词": r"(官网|下载|入口|app|APP|登录|直播|客户端|网址)",
    "资源词": r"(在线观看|免费观看|全集|原文|图片|网盘|百度云|完整版|高清)",
    "回指上文": r"(上面|上述|以上|刚才|刚刚|你说的|你刚|前面说|之前说|你给的|你写的|你画的|你生成的|这篇|这首诗|这段话|这张图|这道题|这个题|这句话|这份报告)",
    "多问句(一行≥2问)": r"[？?].+[？?]",
}


def form_table(df: pd.DataFrame) -> pd.DataFrame:
    """Surface-form markers by source. Regex rules: relative differences only, never precision."""
    out = []
    for src, g in df.groupby("source", sort=False):
        q = g["query"].astype(str)
        row = {"source": src, "n": len(g), "len_median": float(q.str.len().median()),
               "len_p90": float(q.str.len().quantile(.9)), "len_mean": float(q.str.len().mean())}
        for name, pat in FORM.items():
            row[name] = float(q.str.contains(pat, regex=True).mean())
        out.append(row)
    return pd.DataFrame(out)


def overlap_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Share of source A's distinct strings that also occur verbatim in source B."""
    sets = {s: set(g["query"].astype(str)) for s, g in df.groupby("source", sort=False)}
    out = pd.DataFrame(index=list(sets), columns=list(sets), dtype=float)
    for a, sa in sets.items():
        for b, sb in sets.items():
            out.loc[a, b] = len(sa & sb) / max(len(sa), 1)
    return out
