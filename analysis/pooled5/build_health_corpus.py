#!/usr/bin/env python
"""Build the `健康` corpus: one week (2026-09-04 .. 09-10) of health SEARCH and of a specialised
HEALTH AI assistant ("健康AI管家"), pooled into ONE file so one taxonomy labels both.

WHAT THE TWO EXPORTS ARE. Both are "top 10,000 (query x day) rows" over seven days, with the
platform's own three-level category on every row. So neither is a list of distinct queries:
a query that stays in the daily top on all seven days appears seven times. Measured:
  search    10,000 rows -> 2,260 distinct strings; 893 strings on all 7 days, 501 on one day
  AI管家     10,000 rows -> 1,887 distinct strings; 1,075 on all 7 days
Both files are text cells throughout (openpyxl: 10,000 's' cells each), so the leading-zero
defect found in the finance exports cannot occur here; `_read_checked` asserts that.

AGGREGATION, AND WHAT IT CANNOT RECOVER.
- One row per (source, query). `pv_week` = PV summed over every raw row of that query.
- Nine search strings have TWO rows on the SAME day with different category labels and
  different PV (e.g. a symptom query labelled both 症状 and 科普). They are not copies — the export
  split one query's traffic across two labels — so their PV is SUMMED, never de-duplicated.
- A query absent on a day was below that day's cutoff, not at zero. So `pv_week` is EXACT for a
  string present all 7 days and a LOWER BOUND otherwise. `pv_week_upper` adds, for each missing
  day, that day's own cutoff minus one: the most it could have had and still not been exported.
- Category labels: the PV-dominant value is kept as `legacy_l2` / `legacy_l3`; when a string
  carried more than one, `legacy_label_conflict` is set and the full split is kept.
- `legacy_l3` is a crossed label, 科室-类型 (内科-症状). It is split into two orthogonal facets,
  `legacy_dept` and `legacy_type`, which are what the run declares as reference columns.

CLEANING FLAGS, IT DOES NOT DELETE — the rule every pooled corpus here follows. Every string is
written to `work/健康_all_rows.parquet` with its tier and every individual flag; only
`tier == "user"` goes into mining.

HOW AN AI STRING'S TIER IS DECIDED, AND WHY NOT BY RULES ALONE. All 1,887 AI strings were labelled
blind by three independent readings under a criteria-first rubric that never saw these rules
(`work/health_ai_audit_labels.csv`; Fleiss kappa 0.945 over 8 classes, 95.6% unanimous, every
string has a majority). Measured against that majority, the rules below are almost perfectly
PRECISE (0.991) but cannot reach RECALL: answer chips are an open vocabulary (laterality,
temperature and blood-pressure ranges, pregnancy stage, treatment status …) and the best rule
set still left ~590 audit-confirmed chips/pushes (8.3% of AI PV) as "user" — about 40% of the
kept AI rows. So:
  - where all three readings AGREE, their label decides the tier   (`tier_source = audit_unanimous`)
  - where they SPLIT 2-1, the rules decide, so near-identical forms are treated alike (`rule`)
Every AI row keeps `rule_tier`, `audit_majority` and `audit_votes`, so either instrument can be
re-applied. A bare symptom / disease / drug / test name kept as user carries
`flag_bare_term_origin_unknown`: 49 of the 160 such strings the audit kept appear verbatim in a
human-typed log (this week's search, 2025/2026 medical search, the general assistant) and all the
rest appear inside longer typed searches — but a picker option and a typed bare term leave
identical rows, so the export cannot settle their origin and every figure can be recomputed
without them.

  search    C1 content_free · S5 headline (PV-gated, as elsewhere)
            S6 doctor_card_uniform — "科室 + name + 医生" AND on all 7 days AND day-to-day CV <= 0.03.
               Measured: 16 different named doctors received 164k–179k PV each, cross-doctor CV
               0.026, every single day; organic strings in the same PV band vary 0.422 day to day.
               Sixteen individuals do not attract near-identical interest. Both the format and the
               traffic signature are required, so a doctor searched organically is kept.
  AI管家    C1 content_free · H5 doctor/institution card · H4 template wrapper (我想咨询…)
            H2 feature or card-facet button · H1 intake answer chip · H3 pushed suggestion
            Intake answers alone are over 40% of the AI export's PV: 都没有 1.32M, 1周内, 3个月以上,
            男/女, 无发热 … — answers to the assistant's own symptom-intake questions, meaningless as
            queries without the question.

LABELS SAY WHERE EVERY ROW CAME FROM: `source` (key), `source_zh`, `product`, `source_file`, the
raw row count, the days present, and the platform's own category — so any figure can be traced
to the file, the day and the label it came from.

    python analysis/pooled5/build_health_corpus.py
"""
from __future__ import annotations
import re
import sys
from pathlib import Path
import numpy as np
import openpyxl
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import clean_assistant_functional as C  # noqa: E402

DOMAIN = "健康"
OUT_MINE = ROOT / "data/raw/pooled5"
OUT_WORK = Path(__file__).resolve().parent / "work"

SPEC = {
    "hsearch_2609w": dict(file="健康搜索_top1w.xlsx", query="original_query", pv="wise_pv",
                          l1="query_1st_category", l2="query_2nd_category", l3="query_3rd_category",
                          surface="搜索", zh="健康搜索周榜", product="健康搜索"),
    "hai_2609w": dict(file="健康ai管家_top1w.xlsx", query="query", pv="PV",
                      l1="query_level1_type", l2="query_level2_type", l3="query_level3_type",
                      surface="AI助手", zh="健康AI管家周榜", product="健康AI管家"),
}
ORDER = ["hsearch_2609w", "hai_2609w"]


# ------------------------------------------------------------------------------- reading

def _read_checked(src: str) -> pd.DataFrame:
    sp = SPEC[src]
    path = ROOT / "data/raw" / sp["file"]
    wb = openpyxl.load_workbook(path, read_only=True)
    ws = wb[wb.sheetnames[0]]
    it = ws.iter_rows(values_only=False)
    hdr = [c.value for c in next(it)]
    qi = hdr.index(sp["query"])
    numeric = sum(1 for r in it if r[qi].data_type == "n" and r[qi].value is not None)
    wb.close()
    assert numeric == 0, (f"{sp['file']}: {numeric} query cells are stored as numbers — leading zeros "
                          f"would already be gone (the finance export defect). Repair before building.")
    d = pd.read_excel(path)
    q = d[sp["query"]]
    empty = q.isna() | (q.astype("string").fillna("").str.strip() == "")
    assert not int(empty.sum()), f"{sp['file']}: {int(empty.sum())} empty query cells"
    qs = d[sp["query"]].astype(str)
    assert int((qs != qs.str.strip()).sum()) == 0, f"{sp['file']}: leading/trailing whitespace would split strings"
    assert d[sp["l1"]].nunique() == 1, f"{sp['file']}: level-1 category is not constant"
    return d


def _split_l3(v: str) -> tuple[str, str]:
    """科室-类型 -> (科室, 类型). `-` alone means the platform gave no department."""
    v = str(v)
    if v == "-" or not v.strip():
        return "(无科室)", "(无类型)"
    if "-" in v:
        dept, typ = v.split("-", 1)
        return dept, typ
    return v, "(未分类型)"


def _dominant(g: pd.DataFrame, col: str, pv: str) -> tuple[str, str, bool]:
    s = g.groupby(col)[pv].sum().sort_values(ascending=False)
    split = "|".join(f"{k}:{int(v)}" for k, v in s.items())
    return str(s.index[0]), split, len(s) > 1


def aggregate(src: str, d: pd.DataFrame) -> pd.DataFrame:
    sp = SPEC[src]
    q, pv = sp["query"], sp["pv"]
    d = d.assign(_q=d[q].astype(str))
    days = sorted(d.event_day.unique())
    assert len(days) == 7, f"{src}: expected a 7-day export, got {days}"
    floor = d.groupby("event_day")[pv].min()                  # each day's own export cutoff
    daily = d.groupby(["_q", "event_day"])[pv].sum().unstack()  # same-day split rows are SUMMED
    rows = []
    for s, g in d.groupby("_q", sort=False):
        dd = daily.loc[s].dropna()
        missing = [x for x in days if x not in dd.index]
        l2, l2_all, c2 = _dominant(g, sp["l2"], pv)
        l3, l3_all, c3 = _dominant(g, sp["l3"], pv)
        dept, typ = _split_l3(l3)
        rows.append({
            "query": s, "source": src, "source_zh": sp["zh"], "product": sp["product"],
            "source_file": sp["file"], "surface": sp["surface"], "domain": DOMAIN,
            "legacy_l1": str(g[sp["l1"]].iloc[0]), "legacy_l2": l2, "legacy_l3": l3,
            "legacy_dept": dept, "legacy_type": typ,
            "legacy_label_conflict": bool(c2 or c3),
            "legacy_l2_split": l2_all if c2 else "", "legacy_l3_split": l3_all if c3 else "",
            "pv_raw": float(g[pv].sum()),
            "pv_week_upper": float(g[pv].sum() + sum(int(floor[x]) - 1 for x in missing)),
            "n_days": int(len(dd)), "n_rows_raw": int(len(g)),
            "same_day_split_rows": int(len(g) - len(dd)),
            "first_day": int(min(dd.index)), "last_day": int(max(dd.index)),
            "days_present": ",".join(str(x)[4:] for x in dd.index),
            "daily_cv": float(np.std(dd.values) / np.mean(dd.values)) if len(dd) > 1 else float("nan"),
        })
    out = pd.DataFrame(rows)
    # ---- the aggregation must conserve every raw row and every unit of PV
    assert int(out.n_rows_raw.sum()) == len(d), f"{src}: rows not conserved"
    assert abs(out.pv_raw.sum() - d[pv].sum()) < 1e-6, f"{src}: PV not conserved"
    assert out["query"].is_unique, f"{src}: duplicate strings after aggregation"
    return out.sort_values("pv_raw", ascending=False, kind="stable").reset_index(drop=True)


# ------------------------------------------------------------------------------- tiers

#: 16 in this export, all uniform. `[一-鿿]{2,3}` is a given name + surname; the department
#: word before it is what makes it a doctor card rather than a disease ("…科医生" alone never matches).
DOCTOR_CARD_SEARCH = re.compile(r"^[一-鿿]{1,8}(?:科)[一-鿿]{2,3}医生$")
DOCTOR_CARD_AI = re.compile(r"(?:医院|中心|诊所|保健院|卫生院|门诊部).{0,10}(?:主任医师|副主任医师|主治医师|住院医师|医师|教授|专家).{0,8}医生$")
ACK_HEALTH = re.compile(r"^(?:好的|好|嗯|谢谢|谢谢你|好的谢谢|好的，谢谢|好的,谢谢|好的谢谢你)[！!。~～]*$")

_NUM = r"\d+(?:\.\d+)?"
_RANGE = rf"{_NUM}\s*(?:[-~～至到]\s*{_NUM})?"
INTAKE_PARTS = [
    # durations
    rf"(?:超过|大于|小于|少于|不到|约)?{_RANGE}\s*个?(?:小时|天|日|周|星期|个月|月|年)(?:以内|以上|内|左右|前)?",
    r"(?:半年|一年|两年)(?:以内|以上|内)?|6个月-2年",
    # age bands
    rf"(?:小于|大于)?{_RANGE}\s*岁(?:以上|以下|以内)?|{_NUM}个月\s*[-~～至到]\s*{_NUM}岁",
    # sex / who
    r"男|女|男性|女性|成人|儿童|老人|老年人|婴儿|婴幼儿|孕妇|宝宝|本人|自己|家人|孩子|父母",
    # yes / no / unsure / status
    r"是|否|有|无|没有|都没有|都不是|以上都没有|以上都不是|以上均无|不确定|不清楚|不知道|暂不清楚|暂时没有|"
    r"两者都有|都有|其它|其他|其它情况|其他情况|正常|基本正常|不正常|异常|偶尔|偶尔有|经常|一直|会|不会|"
    r"需要|不需要|做过|没做过|打算吃|正在服用|已停药|没吃过|吃过|差不多|不影响|有影响|有点影响|"
    r"严重|轻度|中度|重度|轻微|轻微可忍|能忍受|无法忍受|阴性|阳性|变大|变小|没变化|增多|减少",
    # "无/没有 + symptom word" — the symptom list is closed, so 无痛人流 cannot match
    r"(?:无|没有|没什么)(?:不适|疼痛|发热|发烧|咳嗽|异常|症状|其他症状|明显不适|明显症状|过敏|出血|分泌物|感觉|变化|影响|用药|其它不适|其他不适)",
    # frequency, colour, size, pain character
    rf"{_RANGE}\s*次(?:/?(?:天|周|月|晚))?|每天|每周|每月|每晚",
    r"(?:鲜红|暗红|淡红|红|白|黄|绿|黑|褐|咖啡|灰|透明|乳白|淡黄)色",
    rf"(?:小于|大于|超过|不到)?{_NUM}\s*(?:厘米|cm|毫米|mm|公分)(?:以上|以下|以内)?",
    r"刺痛|胀痛|隐痛|钝痛|绞痛|酸痛|灼痛|跳痛|抽痛|放射痛",
]
_CJK = r"[\u4e00-\u9fff]"
INTAKE_PARTS += [
    # a symptom wrapped in the product's option grammar is a chip; the bare symptom is not
    rf"(?:有|无|没有|没|轻微|明显|轻度|中度|重度|偶尔|持续|经常|严重){_CJK}{{1,6}}",
    # measurement + status. NOT 异常: 白带异常 is a symptom name, and the audit kept it as typed
    rf"{_CJK}{{1,4}}(?:偏高|偏低|正常|稳定|不稳定|升高|降低)|偏高|偏低|正常范围|控制良好|控制不佳|基本不变",
    r"(?:未|已|已经|没有|没|尚未|正在|已在|准备|还没)(?:服用|服药|用药|确诊|开始治疗|开始|治疗|就医|检查|手术|怀孕|使用|在服用|在吃|在用|用过|做过|吃|用|复查|接种|发烧|发热)",
    r"无上述(?:情况|症状)|无明显(?:诱因|规律|变化)|无异味|都没吃|都没做过|随机都可以|暂无不适|首次发现|首次|活动受限|不疼|不痛|不痒|自己用|给孩子用|给家人用",
    r"每(?:天|周|月|年)\s*\d*\s*(?:[-~～至到]\s*\d+)?\s*次?(?:都有)?|每天都有|每天多次|每周几次|每月几次|\d+次以上|很少(?:吃|喝|用)?|反复发作|一阵一阵|时好时坏|间断|偶发|频繁",
    r"少量|中等量|大量|微量|只有\s*\d+\s*个|\d+\s*(?:mg|毫克|ml|毫升|片|粒)",
    r"几(?:分钟|小时|天|周|个月)(?:内)?|\d+\s*分钟(?:以上|以内|左右)?|持续一整天|刚开始(?:用|吃)?|刚刚开始",
    r"(?:偶尔|从不|经常|每天)?(?:饮酒|喝酒|吸烟|抽烟|熬夜)|不吸烟|不饮酒",
    r"(?:双侧|单侧|左侧|右侧|两侧|一侧)(?:都有)?|左边|右边|两边",
    rf"{_NUM}\s*(?:[-~～至到]\s*{_NUM})?\s*(?:℃|度)(?:以上|以下|左右)?|{_NUM}\s*[-~～至到]\s*{_NUM}(?:mmHg|mmol/L)?",
    r"孕(?:早|中|晚)期|备孕(?:中|期)?|哺乳期|经期|绝经(?:前|后)",
    r"\d+岁及以上|\d+岁以下|中年人|成年人|老人小孩|中老年人|中老年|青少年|老年人|孕产妇",
    r"日常保健|日常养生|改善睡眠|缓解疼痛|补充营养|控制体重|提高免疫力|调理身体|手术治疗|中成药|西药|中药|药物治疗|保守治疗",
    r"以上均不属于|以上都不符合|以上均不符合|均不是|都不符合|以上皆无",
    r"(?:上|下|左|右|其他|全身|单只|双)?(?:腹部|肢|部位|脚|手|胸部|背部|头部|腰部|颈部)|身体部位不适|其他部位|腋下|面部|四肢",
    r"清水样|糊状|黏液|带血|成形|不成形|泡沫状|空腹|餐后|饭后|睡前|超重|肥胖|偏瘦|规律|不规律|常规体检|三甲医院|二甲医院|社区医院",
]
INTAKE_ONE = re.compile(r"^(?:" + "|".join(INTAKE_PARTS) + r")$")
INTAKE_SEP = re.compile(r"\s*[，,、;；]\s*")

#: Product feature entries and entity-card facet labels, read off the head of this export.
FEATURE_HEALTH = {"报告单解读", "报告解读", "皮肤病检测", "BMI计算器", "直接出结论", "跳过，直接出结论", "今日健康运势",
                  "我想免费问医生", "饮食营养解读", "免费定制饮食方案", "药盒识别", "经期计算", "排卵期计算"}
TOOL = re.compile(r"(?:计算|识别|测算)$|定制.{0,6}方案$|^跳过|风险自测$|自测$|^中医舌诊$|"
                  r"^我(?:要|想)(?:在线|免费)?(?:问|找)(?:真人|专家)?(?:医生|专家)$|^我想(?:找(?:医生|专家))?挂号$|^我要问真人医生$")
_FACETW = (r"(?:功效|作用|用途|主治|功效主治|功效用途|用途功效|作用功效|作用用途|适应症|适用症状|适用情况|不宜人群|禁忌人群|"
           r"适用人群|适合人群|服用方法|食用方法|用法用量|用法剂量|副作用|不良反应|起效时间|营养成分|主要成分|价格|注意事项|"
           r"治疗方案|治疗方式|治疗用药|用药管理|吃什么药|早期症状|症状表现|症状咨询|基本概念|血糖影响|明确病因|饮食营养解读)")
FACET = re.compile(rf"^(?:常见|主要|内服|外用|食用|药用|运动|对男性的)?{_FACETW}(?:[，,]\s*{_FACETW})*$")
FACET_HEALTH = {"用法用量", "副作用", "适用人群", "适合人群", "注意事项", "食用方法", "禁忌人群", "用法剂量",
                "功效与作用", "功效作用", "药物治疗"}
WRAPPER = re.compile(r"^我想(?:咨询|了解)|快看看你的")
PUSHED_FORM = re.compile(r"^\d+分钟.{0,12}自测|^请帮我出几道|^如何帮父母|自测,看清")
#: A polished question that no one typed into search that week, framed by the campaign themes
#: (caregiving, season, that week's events) or by template politeness, and not colloquial.
#: Measured against the audit: precision 0.991, recall 0.753 of the pushed class on its own.
PUSHED_THEME = re.compile(r"(?:父母|老人|老年人|爸爸|妈妈|家里|家人|家庭|孩子|儿童|小儿|学生|开学|秋季|入秋|夏季|夏天|换季|"
                          r"暴雨|洪涝|台风|高温|新冠|流感|甲醛|县级医院|周末|假期|入职|办公|久坐|屏幕|电脑)")
PUSHED_POLITE = re.compile(r"(?:如何|怎么|怎样|需要|是否|该|哪些|什么方向|先查什么|要注意什么)")
COLLOQUIAL = re.compile(r"(?:咋|啥|俺|了怎么办|怎么回事啊|吗吗|！！|\?\?|。。|…)")
BARE_ASK = re.compile(r"[吗呢？?]|什么|怎么|如何|为什么|哪|多少|是否|能不能|可以|会不会|要不要|严重|原因|办")
AUDIT_TIER = {"A": "H1_intake_answer", "F": "H2_feature_or_facet", "P": "H3_pushed_suggestion",
              "W": "H4_template_wrapper", "D": "H5_doctor_card", "C": "C1_content_free", "U": "user"}


def intake_answer(q: str) -> bool:
    parts = [x for x in INTAKE_SEP.split(q.strip()) if x]
    return bool(parts) and all(INTAKE_ONE.match(x) for x in parts)


def tier_search(t: pd.DataFrame) -> pd.DataFrame:
    q = t["query"]
    t["flag_C1"] = C.content_free_flags(q, surface="search").values
    t["flag_S5"] = C.headline_flags(q, t["pv_raw"]).values
    t["flag_S6"] = (q.str.match(DOCTOR_CARD_SEARCH) & (t.n_days == 7) & (t.daily_cv <= 0.03)).values
    t["tier"] = "user"
    t.loc[t.flag_S5, "tier"] = "S5_headline"
    t.loc[t.flag_S6, "tier"] = "S6_doctor_card_uniform"
    t.loc[t.flag_C1, "tier"] = "C1_content_free"
    return t


def tier_ai(t: pd.DataFrame, search_strings: set[str], audit: pd.DataFrame) -> pd.DataFrame:
    q = t["query"]
    in_search = q.isin(search_strings)
    L = q.str.len()
    t["flag_C1"] = (C.content_free_flags(q, surface="assistant") | q.str.match(ACK_HEALTH)).values
    t["flag_H5"] = q.str.contains(DOCTOR_CARD_AI).values
    t["flag_H4"] = q.str.contains(WRAPPER).values
    t["flag_H2"] = (q.isin(FEATURE_HEALTH | FACET_HEALTH) | q.str.contains(TOOL) | q.str.match(FACET)).values
    # a string a human also typed into search is not an answer chip, whatever it looks like
    t["flag_H1"] = (q.map(intake_answer) & ~in_search).values
    t["flag_H3"] = (q.str.contains(PUSHED_FORM)
                    | (~in_search & (L >= 8)
                       & (q.str.contains(PUSHED_THEME) | q.str.contains(r"[？?]$") | q.str.contains(PUSHED_POLITE))
                       & ~q.str.contains(COLLOQUIAL))).values
    t["query_unwrapped"] = q.str.replace(r"^我想(?:咨询|了解)", "", regex=True)
    t["rule_tier"] = "user"
    for flag, tier in [("flag_H3", "H3_pushed_suggestion"), ("flag_H1", "H1_intake_answer"),
                       ("flag_H2", "H2_feature_or_facet"), ("flag_H4", "H4_template_wrapper"),
                       ("flag_H5", "H5_doctor_card"), ("flag_C1", "C1_content_free")]:
        t.loc[t[flag], "rule_tier"] = tier                    # later assignment = higher precedence
    a = audit.set_index("query")
    missing = set(q) - set(a.index)
    assert not missing, f"{len(missing)} AI strings have no audit label — the audit file is stale"
    t["audit_majority"] = q.map(a["majority"]).values
    t["audit_votes"] = q.map(a["votes"]).values
    unanimous = t["audit_votes"].map(lambda v: len(set(str(v))) == 1)
    t["tier_source"] = np.where(unanimous, "audit_unanimous", "rule")
    t["tier"] = np.where(unanimous, t["audit_majority"].map(AUDIT_TIER), t["rule_tier"])
    t["flag_bare_term_origin_unknown"] = ((t["tier"] == "user") & (L <= 6) & ~q.str.contains(BARE_ASK)).values
    return t


# ------------------------------------------------------------------------------- main

def main() -> int:
    C._check_engine_semantics()
    OUT_MINE.mkdir(parents=True, exist_ok=True)
    agg = {s: aggregate(s, _read_checked(s)) for s in ORDER}
    s_set, a_set = set(agg["hsearch_2609w"]["query"]), set(agg["hai_2609w"]["query"])
    audit = pd.read_csv(OUT_WORK / "health_ai_audit_labels.csv")
    tiered = [tier_search(agg["hsearch_2609w"].copy()), tier_ai(agg["hai_2609w"].copy(), s_set, audit)]
    allrows = pd.concat(tiered, ignore_index=True)
    allrows["in_other_source"] = np.where(allrows.source == "hsearch_2609w",
                                          allrows["query"].isin(a_set), allrows["query"].isin(s_set))
    allrows["kept"] = allrows["tier"] == "user"
    allrows["row_id_all"] = range(len(allrows))
    mine = allrows[allrows.kept].reset_index(drop=True).copy()
    tot = mine.groupby("source")["pv_raw"].transform("sum")
    mine["pv_norm"] = (mine["pv_raw"] / tot * 10_000).astype(float)
    mine["row_id"] = range(len(mine))
    cols = ["query", "source", "source_zh", "product", "source_file", "surface", "domain",
            "legacy_l1", "legacy_l2", "legacy_l3", "legacy_dept", "legacy_type", "legacy_label_conflict",
            "pv_raw", "pv_norm", "pv_week_upper", "n_days", "n_rows_raw", "first_day", "last_day",
            "days_present", "daily_cv", "in_other_source", "tier", "row_id", "row_id_all"]
    for extra in ["tier_source", "audit_majority", "flag_bare_term_origin_unknown"]:
        mine[extra] = mine[extra] if extra in mine.columns else pd.NA
    cols += ["tier_source", "audit_majority", "flag_bare_term_origin_unknown"]
    mine[cols].to_parquet(OUT_MINE / f"{DOMAIN}_pooled5.parquet", index=False)
    allrows.to_parquet(OUT_WORK / f"{DOMAIN}_all_rows.parquet", index=False)
    rep = []
    for src in ORDER:
        g = allrows[allrows.source == src]
        rep.append({"domain": DOMAIN, "source": src, "raw_export_rows": int(g.n_rows_raw.sum()),
                    "rows": len(g), "kept": int(g.kept.sum()), "dropped": int((~g.kept).sum()),
                    "drop_%": round(100 * (~g.kept).mean(), 2),
                    "pv_dropped_%": round(100 * g.loc[~g.kept, "pv_raw"].sum() / g["pv_raw"].sum(), 2)})
    pd.DataFrame(rep).to_csv(OUT_WORK / "build_audit_health.csv", index=False, encoding="utf-8-sig")
    tiers = (allrows.groupby(["source", "tier"]).agg(串=("query", "size"), 周PV=("pv_raw", "sum"))
             .reset_index())
    tiers["周PV占该源%"] = (100 * tiers["周PV"] / tiers.groupby("source")["周PV"].transform("sum")).round(2)
    tiers.to_csv(OUT_WORK / "build_tiers_health.csv", index=False, encoding="utf-8-sig")
    print(pd.DataFrame(rep).to_string(index=False))
    print("\n" + tiers.to_string(index=False))
    print(f"\n{DOMAIN}: {len(allrows):,} 串 → 入挖掘 {len(mine):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
