#!/usr/bin/env python
"""Separate system-generated and content-free strings from user queries, for analysis.

WHY THIS EXISTS. The AI-assistant query log records more than what users typed.
Measured on `ai04`: three classes are 18.7% of rows and 76.8% of traffic, and the
largest contributors are strings the PRODUCT authored — `👌 好的，继续吧` in 32 of
33 categories, `我发射了很多表情。` with 45,534 PV, a 300-character portrait
template with 5,269 identical sends. Any intent share computed with those rows in
it describes the product's UI as much as its users.

NOTHING IS DELETED FROM A RUN. This writes an analysis copy with a tier per row,
so every figure can be recomputed with or without any tier.

THE TIERS, each a stated rule rather than a model's opinion:

  S1 system_repeated   the identical string recurs across topical categories
                       (>=10 categories, or >=3 categories at >=6 chars): chips,
                       suggested follow-ups, tapped trending headlines
  S2 system_template   an identical string whose traffic is implausible for
                       independent typing (>=15 chars at >=1,000 PV; or a
                       punctuation-perfect sentence at >=3,000 PV), a canonical
                       tool-button label, a named product feature entry or a
                       call-to-action label, or a fixed prompt template
  S3 card_passage      an encyclopedia-style passage (a person's or a work's
                       card text), usually with a templated question appended
  S4 suggested_chip    a suggested follow-up chip (`能否再…`, `有没有更…的…`,
                       `该公司…`) — the tap is the user's, the words are not
  S5 headline          news-headline syntax on a trafficked head row: a tapped
                       trending headline. Applied to BOTH surfaces
  C1 content_free      bare acknowledgements, yes/no replies, interjections,
                       wake words and greetings, punctuation/emoji
  C2 feed_control      feedback to a content feed (`多推`, `以后别给我推这种视频`)

BARE NUMBERS AND LETTERS ARE NOT REMOVED, on either surface, and that was a
correction. An earlier version dropped them as "option replies", which is right
for a conversation and wrong for search: it stripped `985` (the university tier)
from education search and 162 bare numbers from finance search. Applying it to
the assistant alone would bias the bare-entity share between the two surfaces,
so neither gets it; `flag_bare_number` records them for measurement instead.

S3 WAS FOUND BY READING, NOT BY A THRESHOLD. 60 of 964 kept 人物 head rows
(6.2%, 15,837 PV) were passages like `张又侠，男，汉族，1950年7月生…`, and 39 of the
60 end with a uniformly templated question (`…目前担任什么职务`, `…的主要成就有哪些`).
The same passage recurs with DIFFERENT appended questions at different traffic
(730 / 513 / 428 PV), which is what tapping suggested questions under an entity
card looks like. The user's choice is real; the text is the product's. Left in,
it reads as users writing long questions, which is precisely the comparison this
cleaning exists to protect. Identified by FORMAT in both strata, not by traffic.

THE SECOND AUDIT (a blind, criteria-first labelling of every tier) found removal
sound and recall not. Precision A/(A+B): S1 0.964 over all 1,103 rows, S2 0.851
[0.785, 0.900] on 150 sampled rows (PV-weighted 0.998), C1 0.966, C2 0.998. But
11.3% [8.2, 15.4] of KEPT top1k rows were still system or content-free text, and
they concentrated in a few intent classes: fact-verification was roughly doubled
by tapped headlines and chips. What changed, each with its measured basis:
  - S4 and S5 were added for those misses (chips 40/40 precise in top1k and 30/33
    in random1k; headline syntax 28/30 at PV >= 100 in top1k, but 3/29 in random1k,
    so S5 never touches the random stratum). A bio-card format joined S3, and
    yes/no replies, farewells and insults joined C1 (41 rows, all content-free).
  - Over-catches were undone. `这是谁` / `这是哪里` / `这是什么？` left the feature
    list: typed variants exist at similar volume (`这个人是谁` 2,883 PV), unlike
    `这是什么`, whose PV sits 93% in one category. Keyword-shaped long queries
    (`2026年养老金调整方案何时公布`) are exempt from the long-high-PV rule unless
    they carry a template mark. Other assistants' names said ONCE (`豆包` 5,900 PV)
    are a product lookup, not a wake word. Non-CJK scripts are content, not empty.
  - C1 now differs by surface, deliberately. In a chat an isolated `嗯` or `行` is a
    turn-taking reply; in a search box a single character is a plausible dictionary
    or title lookup (the audit found the stroke glyph `㇏`, a Russian word and the
    show title `哈哈哈哈哈` removed from search). Search keeps only the empty and
    multi-character-acknowledgement parts of C1.

"HAS CONTENT" IS DECIDED IN PYTHON, NOT BY A REGEX. The audit's suggested
`[^\\W_]` is Unicode-aware in Python's `re`, but pandas routes `.str.contains` on
its default string dtype through pyarrow's RE2, where `\\w` is ASCII-only. The
first run with it read every Chinese query as empty and tiered 81.7% of head rows
content-free. `str.isalnum` is Unicode-aware whichever engine pandas picks, and
`_check_engine_semantics` fails loudly if that ever stops being true.

WHAT THE RULES CANNOT DO, stated because it bounds every downstream figure: a
button tap and a user typing the same short command leave identical rows. The
tool-button rule therefore only fires at >=10,000 PV on <=6 characters, where
the canonical label dwarfs typed variants; below that, short edit commands are
kept as user intent. Sensitivity to that threshold is reported, not hidden.
"""
from __future__ import annotations

import argparse
import importlib.util
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

#: Product features that surfaced in the head as fixed entry strings. Each was
#: read in context before being listed: a daily-briefing card, a document
#: summariser, video generation, similar-image search, red-packet activities.
FEATURE_ENTRIES = {
    "总结全文概要", "生成视频", "变视频", "根据对话内容生成视频", "今日早报", "今日晚报",
    "相似图", "找相似图", "答题红包", "领红包", "百度推荐官", "翻译文字",
    # Found by the recall audit — kept head rows that were plainly product copy.
    "帮我全网文本搜题", "AI写真", "AI扩图", "证件照", "启动AI打分", "测测我的今日运势",
    "夏夜麦霸，测测歌声得几分", "点我帮你画出来看看", "我想唱信仰",
    "能否再提供一个更简洁的版本", "能否再提供一个更简短的版本", "能否再提供一个更押韵的版本",
    "发送我对以上文章进行编辑后的全部内容", "再生成几张不同场景的", "这个梦是否预示着什么",
    # Call-to-action button labels, found by the second audit.
    "领取AI志愿报告", "直接生成研究报告", "领现金",
    # Recognition-entry default prompt: `这是什么` carries 196k of its 197k PV in
    # 音频与演出 (song ID) and recurs in 8 categories. Its siblings were removed
    # from this set by the second audit — see the module docstring.
    "这是什么",
}

#: The assistant categories that share a topic with a search vertical.
DOMAINS = {"金融": ("fin-pool", "金融"), "医疗": ("med-pool", "医疗"),
           "教育": ("edu-pool", "教育培训"), "影视": ("film-pool", "影视动漫"),
           "人物": ("ppl-pool", "人物")}

INTERJ = re.compile(r"^[啊哦嗯唉哎呀哇噢喔哈嘿呃额嘻呵嗷咦]+[\s，,。！!~～.…]*$")
#: This product's own names. Another assistant's name said once is a lookup.
WAKE = re.compile(r"^\s*(百度AI助手|百度助手|文心助手|小度AI|AI助手|小度|文心一言|文心)"
                  r"([\s，,、。！!]*(小度|文心))?[\s，,。！!]*(你好)?[\s，,。！!~～]*$")
#: ...but said twice it is a mis-aimed wake call. Spelled out: RE2 has no backreferences.
_SEP = r"[\s，,、。！!]+"
WAKE_OTHER = re.compile(r"^\s*(豆包" + _SEP + r"豆包|元宝" + _SEP + r"元宝|千问" + _SEP + r"千问|[kK]imi" + _SEP
                        + r"[kK]imi|[dD]eep[sS]eek" + _SEP + r"[dD]eep[sS]eek)[\s，,。！!~～]*$")
GREET = re.compile(r"^\s*(喂+|你好+|您好|哈喽|hello|hi|Hello|Hi|HELLO)[\s，,。！!~～]*$")
REPLY = re.compile(r"^\s*(不是|不对|不不+|是吗|嗯[，,]\s*是|谢谢你|谢谢啦|多谢|快点儿?|去吧|生成吧|好滴|嗯呢|对对+|好嘞|懂了|"
                   r"晚安|早安|拜拜|再见|滚(蛋)?|sb|SB|傻逼|操你妈的?)[\s，,。！!~～？?]*$")
OPTION = re.compile(r"^\s*(\d{1,3}|[A-Za-z])\s*[。.]?\s*$")
FEED_SHORT = re.compile(r"^[\s，,。！!]*(以后)?(多|少|别|不要|不)(再)?(给我)?推(荐)?[\s，,。！!~～]*$")
FEED_CONTENT = re.compile(r"(多|少|别|不要|不|千万不要|别再|不要再)(再)?(给我|跟我)?推(荐)?"
                          r"(一点|些)?.{0,8}(视频|作者|内容|东西|主播|博主)")
ENCYCLOPEDIC = re.compile(r"(，(男|女)，|出生于|\d{4}年\d{1,2}月(\d{1,2}日)?(出生|生于)|（\d{4}年\d{1,2}月|"
                          r"是由.{2,30}(制作|出品|执导)|中国内地(男|女)?(演员|歌手)|现任.{2,20}(书记|主任|局长|市长|县长))")
BIO_CARD = re.compile(r"^[一-鿿]{2,4}[（(]?.{0,30}[）)]?[，,](男|女)[，,]")
EMOJI_CONTINUE = re.compile(r"^\s*[👌🆗]\s*(好的|行)[，,]?\s*继续吧\s*$")
PUNCT_END = re.compile(r"[。？！]$")
TEMPLATE_START = re.compile(r"^(生成一首适合|生成一期讨论)")
DREAM_TEMPLATE = re.compile(r"梦见.{1,20}(代表|预示)什么[？?].{0,4}好兆头")
CHIP = re.compile(r"^(能否|可否)|^有没有更.{1,8}的|^如果以上|该(公司|岗位|产品|企业|平台|机构|车型|品牌)"
                  r"|^用更.{1,6}的(语言|方式|语气)|^再生成几张|^能帮我找(到)?这张.{1,3}的原图吗$")
#: `来了$` needs a long string: short ones are titles (`爸爸回来了`, a variety show, 1,216 PV).
HEADLINE = re.compile(r"(已致\d+人|遇难|失联|被查|官宣|回应|通报|表态|曝光|^.{6,}来了$|去世$|离世$|身亡|被罚|获刑|落马|“[^”]{1,10}”)")
ASKS = re.compile(r"(吗|什么|怎么|如何|哪|多少|为何|为什么|是否|？|\?|能否|可以|帮我|请)")
NEWS_ASKS = re.compile(r"(吗|什么|怎么|如何|哪|多少|为何|为什么|是否|？|\?|能否|可以|帮我|请|攻略|图片|视频|在线|免费|下载|官网|入口|查询)")
#: Keyword-shaped lookups that the long-high-PV rule caught (autocomplete-shaped
#: policy, score-line and play-page queries) — exempt unless a template mark is present.
KEYWORD_ASK = re.compile(r"(是多少|是什么|何时(公布|发布)|怎么|怎样|可以.{0,8}吗$|吗$|在线观看|免费观看|全集|百度云|分数线|查询)")
TEMPLATE_MARK = re.compile(r"^(帮我画|生成一|以图|请|帮我|我想对作文|参考上传|如果以上)|[：:「」；]|如图|" + "‌"
                           + r"|[，,](男|女)[，,]|梦见|[？?].+[？?]")
#: Typed follow-up questions that the cross-category rules caught (39 wrong S1 rows in the census).
S1_KEEP = re.compile(r"^(这是(什么意思|怎么回事|什么情况)[？?]?|电话|deepseek|DeepSeek)$|百度视频$|中文百度$")
RECOGNITION_TYPED = re.compile(r"^这是(什么|谁|哪里)[？?]$|^这是(谁|哪里)$")
WRAPPER = re.compile(r"^请帮我(画一张图片|生成一个视频)[：:].*比例为")


def _ack_pattern() -> re.Pattern:
    spec = importlib.util.spec_from_file_location(
        "prepare_assistant_corpus", ROOT / "tools" / "prepare_assistant_corpus.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return re.compile(mod.ACK_PATTERN)


def has_content(q: pd.Series) -> pd.Series:
    """Any Unicode letter or digit, in any script. Deliberately not a regex — see the docstring."""
    return q.astype(str).map(lambda x: any(ch.isalnum() for ch in x)).astype(bool)


def _check_engine_semantics() -> None:
    probe = pd.Series(["血压正常值", "អ្នកមក", "。。", "👌", "985"])
    got = has_content(probe).tolist()
    if got != [True, True, False, False, True]:
        raise SystemExit(f"has_content is not Unicode-aware on this pandas: {got}")
    if not pd.Series(["豆包，豆包"]).str.match(WAKE_OTHER).iloc[0]:
        raise SystemExit("WAKE_OTHER does not match on this pandas string engine")


def content_free_flags(q: pd.Series, surface: str = "assistant") -> pd.Series:
    """C1: rows that carry no standalone content. On search, only empty strings and
    multi-character acknowledgements — a short word in a search box is a lookup."""
    ack = _ack_pattern()
    s = q.astype(str)
    empty = ~has_content(s)
    if surface == "search":
        return empty | (s.str.match(ack) & (s.str.len() >= 2))
    return (s.str.match(ack) | empty | s.str.match(INTERJ) | s.str.match(WAKE) | s.str.match(WAKE_OTHER)
            | s.str.match(GREET) | s.str.match(REPLY))


def headline_flags(q: pd.Series, pv: pd.Series) -> pd.Series:
    """S5 rule B: headline syntax, no question or request word, at PV >= 100."""
    s = q.astype(str)
    return s.str.contains(HEADLINE) & ~s.str.contains(ASKS) & (pv >= 100)


def feed_flags(q: pd.Series, td: pd.Series | None = None) -> pd.Series:
    s = q.astype(str)
    m = s.str.match(FEED_SHORT) | s.str.contains(FEED_CONTENT)
    if td is not None:
        m |= (td.astype(str) == "FEED_TUNING_COMMAND") & (s.str.len() <= 40)
    return m


def tier_assistant(d: pd.DataFrame, tool_pv: int = 10_000) -> pd.DataFrame:
    d = d.copy()
    d["qs"] = d["query"].astype(str)
    d["qlen"] = d["qs"].str.len()
    d["n_l1_same_string"] = d.groupby(["snapshot", "qs"])["l1"].transform("nunique")
    head = d["snapshot"] == "top1k"
    typed = d.qs.str.match(S1_KEEP) | d.qs.str.match(RECOGNITION_TYPED)
    s1 = (((head & (d.n_l1_same_string >= 10))
           | (head & (d.n_l1_same_string >= 3) & (d.qlen >= 6))
           | d.qs.str.match(EMOJI_CONTINUE)) & ~typed)
    long_high = head & (d.qlen >= 15) & (d.search_num >= 1_000)
    keyword_like = d.qs.str.contains(KEYWORD_ASK) & ~d.qs.str.contains(TEMPLATE_MARK)
    s2 = (((long_high & ~keyword_like)
           | (head & d.qs.str.contains(PUNCT_END) & (d.qlen >= 5) & (d.search_num >= 3_000))
           | (head & (d.td_l1.astype(str) == "TOOL_PANEL_SINGLE_STEP_COMMAND")
              & (d.qlen <= 6) & (d.search_num >= tool_pv))
           | d.qs.isin(FEATURE_ENTRIES)
           | d.qs.str.match(TEMPLATE_START)
           | d.qs.str.contains(DREAM_TEMPLATE)) & ~typed)
    s3 = (d.qs.str.contains(ENCYCLOPEDIC) & (d.qlen > 40)) | d.qs.str.match(BIO_CARD)
    s4 = d.qs.str.contains(CHIP)
    news_a = head & (d.l1.astype(str) == "新闻") & (d.qlen >= 8) & ~d.qs.str.contains(NEWS_ASKS)
    s5 = news_a | (head & headline_flags(d.qs, d.search_num))
    c1 = content_free_flags(d.qs, "assistant")
    c2 = feed_flags(d.qs, d.get("td_l1"))
    d["flag_S1"], d["flag_S2"], d["flag_S3"], d["flag_S4"], d["flag_S5"] = s1, s2, s3, s4, s5
    d["flag_C1"], d["flag_C2"] = c1, c2
    # Informational, never tiering: lets the report bound what the ~90%-precise
    # long-and-high-PV rule could have misplaced, and measure bare numbers and
    # typed content wrapped in a product template (kept as user intent).
    d["flag_S2_long_high_pv"] = long_high
    d["flag_bare_number"] = d.qs.str.match(OPTION)
    d["flag_wrapper_template"] = d.qs.str.match(WRAPPER)
    d["tier"] = "user"
    for name, m in (("C2_feed_control", c2), ("C1_content_free", c1), ("S5_headline", s5),
                    ("S4_suggested_chip", s4), ("S3_card_passage", s3),
                    ("S2_system_template", s2), ("S1_system_repeated", s1)):
        d.loc[m, "tier"] = name           # later assignment wins: S1 > S2 > S3 > S4 > S5 > C1 > C2
    return d.drop(columns=["qs"])


def search_2026_labeled() -> pd.DataFrame:
    """2026 search rows with their own run's labels, joined by VERIFIED position."""
    out = []
    for dom, (run, _cat) in DOMAINS.items():
        lab = pd.read_csv(ROOT / "runs" / run / "gen01" / "labels_full.csv")
        raw = pd.read_csv(ROOT / "data" / "raw" / f"{dom}query-pooled.csv")
        raw = raw[raw.original_query.notna()].reset_index(drop=True)
        if len(lab) != len(raw) or (lab["query"].astype(str).values
                                    != raw.original_query.astype(str).values).any():
            raise SystemExit(f"{run}: labels and pooled source are not row-aligned")
        x = lab.assign(wise_pv=raw.wise_pv.values, snap=raw._snapshot.astype(str).values)
        x = x[x.snap == "20260701"].copy()
        x = x.sort_values("wise_pv", ascending=False).reset_index(drop=True)
        x["rank"] = x.index + 1
        x["domain"] = dom
        out.append(x[["domain", "query", "wise_pv", "rank", "td_l1", "td_l1_name",
                      "bu_leaf_name", "bu_family_final"]])
    s = pd.concat(out, ignore_index=True)
    s["flag_C1"] = content_free_flags(s["query"], "search")
    s["flag_bare_number"] = s["query"].astype(str).str.match(OPTION)
    s["flag_C2"] = feed_flags(s["query"])
    s["flag_S5"] = headline_flags(s["query"], s["wise_pv"])
    s["tier"] = "user"
    s.loc[s.flag_C2, "tier"] = "C2_feed_control"
    s.loc[s.flag_C1, "tier"] = "C1_content_free"
    s.loc[s.flag_S5, "tier"] = "S5_headline"
    return s


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--labels", default=str(ROOT / "runs/ai04/gen01/postprocessed/labels_full_按垂类排序.csv"))
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--tool-pv", type=int, default=10_000)
    a = ap.parse_args()
    _check_engine_semantics()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    d = tier_assistant(pd.read_csv(a.labels), a.tool_pv)
    d.to_parquet(out / "assistant_tiered.parquet", index=False)
    s = search_2026_labeled()
    s.to_parquet(out / "search2026_tiered.parquet", index=False)

    for nm, df, pv in (("ASSISTANT", d, "search_num"), ("SEARCH 2026", s, "wise_pv")):
        grp = ["snapshot", "tier"] if "snapshot" in df.columns else ["tier"]
        t = df.groupby(grp).agg(rows=("query", "size"), pv=(pv, "sum"))
        print(f"\n== {nm} ==")
        print(t.to_string())
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
