#!/usr/bin/env python
"""Assemble `people_zh_v2` / `film_tv_zh_v2` from v1 plus the verified proposals, measuring everything here.

WHY A SCRIPT AND NOT HAND-EDITING. The v2 profiles carry ~25 regexes each that must clear a mechanical floor
(`ops/templates.build_groups` silently drops a seed matching < 0.004 x rows) and must not swallow each other
(a seed whose rows are >= 85% already claimed by an earlier seed is also dropped, silently). The research
round measured those numbers with agents; the skeptics then re-measured and found inflated claims, dead
tokens and duplicate proposals. So every number that reaches the YAML is measured HERE, by this script, on
the actual corpus, and the survival of every seed is checked by calling the real `build_groups`.

WHAT WAS APPLIED FROM THE SKEPTICS' VERDICTS (each one is an assertion below, not a promise):
人物8
  - institution_scoped_person and nav_account_lookup are NOT seeds (entity family / only clears the floor by
    absorbing the contact-detail rows that a risk category already owns) -> hints + notes only.
  - the roster intent gets ONE seed: 名单$ (B's narrow form, 0 overlap with v1's nine) — A's wider form would
    silently drop it under the 85% rule.
  - incumbency_tenure loses 原任 (0 rows in the corpus).
  - evaluation_comparison is tightened; the loose form measured precision 0.60 (近况 family + 评价@机构).
  - the two locator proposals and the two minor proposals are each merged into one category.
  - v1's sexualised_and_non_person_browse absorbs the named-person explicit-act words rather than a new class.
影视8
  - channel_number_nav REPLACES v1's channel_live_tv (it contains 86.0% of it — over the 85% line).
  - foreign_drama_generic_access is dropped in favour of the wider foreign_title_access / foreign_unlicensed_access.
  - the two aggregator proposals are merged into ONE risk category (name list arm + word-shape arm).
  - dead tokens removed: 独播库, 日剧网, 甜性涩爱, 三级片/禁片 (all 0 rows here); 禁播 arm removed (0/5 precision).
  - numbered_euphemistic_serial drops its `\\d{2,3}分钟` arm (8 of 37 hits were exercise/class videos).
  - bracket_title / ultra_short stay out of the seeds (they are shape, not intent) and go into notes.

    HF_HOME=$(pwd)/.hf .venv/bin/python analysis/pooled5/build_pool8_profiles.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
RESEARCH = Path("/private/tmp/claude-501/-Users-mayouxuan-Documents-Claude-Search-Query-Mining-Agent-Team/"
                "3deba04e-73b1-4747-b7b7-4d1a40520650/scratchpad/pool8_profile_research.json")

DOMS = {
    "人物8": {"v1": "people_zh", "v2": "people_zh_v2", "corpus": "data/raw/pooled5/人物8_pooled5.parquet",
             "blind": "研究:人物8:新层盲区", "risk": "研究:人物8:风险合规"},
    "影视8": {"v1": "film_tv_zh", "v2": "film_tv_zh_v2", "corpus": "data/raw/pooled5/影视8_pooled5.parquet",
             "blind": "研究:影视8:新层盲区", "risk": "研究:影视8:风险合规"},
}

#: seeds to take from the research, by (agent key, name). `pattern` overrides apply the skeptics' fixes.
TAKE_SEEDS = {
    "人物8": [
        ("blind", "yes_no_verification", None, "是非核实：把一个断言拿来求证，长尾独有的问法"),
        ("blind", "incumbency_tenure", r"(现任|历任|新任)", "任期与任免：新任 的多数来自「最新任命/最新任免」的子串，读的时候要知道"),
        ("blind", "relation_lookup_extended", None, "人际关系反查：亲属、师承、搭档；会碰到作品名（的女儿）约 1/20"),
        ("blind", "causal_explanation", None, "求解释：为什么某人做了某事、出了什么事"),
        ("blind", "person_to_post_lookup", None, "人→职反查；v1 的两条职务种子只覆盖职→人方向，与它零重叠"),
        ("blind", "attribute_lookup_spoken", None, "口语化的属性问法（几岁、多高、属什么），语音与随机层的写法"),
        ("blind", "achievement_record", None, "成就与荣誉；v1 的 bio 种子锚了 $，抓不到「X个人简历和成就」这种复合尾"),
        ("blind", "evaluation_comparison", r"(如何评价|评价|厉害|谁更|更强|哪个强|对比|相比|算不算|口碑)",
         "评价与比较。原提议含 怎么样，把「后来怎么样了」这一族近况问法也吞进来，实测精度 0.60；去掉 怎么样 后 203 行、仍高于地板，残留的是「张雪峰评价<学校>」式的机构评价（31 行 ≈ 15%），读组内样例时要知道"),
        ("risk", "institution_roster_suffix", None, "机构花名册：名单$ 与 v1 九个种子零重叠，精度 19/20"),
        ("risk", "grassroots_school_post_holder", None, "基层校内职务持有人（校长/班主任）；地板余量仅 26 行"),
    ],
    "影视8": [
        ("blind", "platform_account_service", None, "平台账号与服务问题：把平台当对象，而不是把作品当对象"),
        ("blind", "title_unknown_lookup", None, "反向找片：给情节/画面求片名，与「已知片名求资源」方向相反"),
        ("blind", "scene_episode_locator", None, "定位到集/期/分钟；v1 的 episode_locator 只覆盖集"),
        ("blind", "person_to_works", None, "人→作品反查；v1 的 cast_crew 是作品→人"),
        ("blind", "plot_outcome_qa", None, "剧情结局追问；三个「为什么X」臂是承重的——删掉就只剩 158 行、低于地板"),
        ("blind", "open_why_how", None, "开放式怎么办/为什么：**句式族而非意图族**，组内混平台求助与剧内追问，fragmentation 预期会碎"),
        ("blind", "short_drama", None, "短剧：一个在旧四层几乎不存在的新品类"),
        ("blind", "channel_number_nav", None, "频道与台号导航；**取代 v1 的 channel_live_tv**（含它 86.0%，同时提交会被 85% 规则静默丢弃）"),
        ("blind", "language_version", r"(国语版|粤语版|中文版|普通话版|中字|中文字幕|字幕版|国语配音|配音版|韩语版|泰语|原声版|国语高清|国语在线)",
         "语种与版本轴；裸词 字幕|配音 会碰到音乐视频与生成指令，已锚定"),
        ("blind", "variety_show", None, "综艺与晚会：与剧集的检索行为不同"),
        ("blind", "nonfilm_ugc_video", None, "非影视的 UGC 视频（广场舞、吃播、解说）——类目边界外溢"),
        ("blind", "season_progress", None, "季/番/更新进度：追更行为，全域最大的新种子"),
        ("risk", "foreign_title_access", None, "外语剧集的资源获取（韩剧/泰剧/日剧 + 观看/网盘词）"),
        ("risk", "uncut_version_request", None, "未删减版本请求"),
        ("risk", "offline_channel_delivery", None, "网盘/磁力等离线渠道"),
    ],
}
#: v1 seeds to drop (replaced by a wider new one), per domain
DROP_V1_SEEDS = {"人物8": [], "影视8": ["channel_live_tv"]}

#: risk categories to take from the research. `merge_into` folds the patterns into an existing v1 category.
TAKE_RISKS = {
    "人物8": [
        ("blind", "judicial_attribution", None, None),
        ("blind", "personal_wealth_and_assets", None, None),
        ("blind", "non_person_generic_browse", None, None),
        ("risk", "grassroots_named_individual_lookup", None, None),
        ("risk", "liveness_verification_probe", None, None),
        ("risk", "institutional_roster_disclosure", None, None),
        ("risk", "leader_kinship_and_private_relations", None, None),
        # merged pairs: patterns are the union of the two proposals
        ("merge", "private_individual_locator", ("blind:private_individual_locator", "risk:private_individual_contact_locator"), None),
        ("merge", "minor_as_subject", ("blind:minor_sexualisation", "risk:minor_as_subject"), None),
        # the named-person explicit-act words go into v1's existing browse category, not a new one
        ("into", "sexualised_and_non_person_browse", ("blind:explicit_sexual_act",), None),
    ],
    "影视8": [
        ("blind", "restraint_and_tickle_fetish", None, None),
        ("risk", "foreign_unlicensed_access", None, None),
        ("risk", "explicit_nudity_and_scale", None, None),
        ("risk", "ncii_leak_aggregator", None, None),
        ("risk", "kr_hk_adult_title_family", None, None),
        ("risk", "euro_softcore_codeword", None, None),
        ("risk", "adult_anime_blocklist_v2", None, None),
        ("merge", "unlicensed_aggregator", ("blind:aggregator_site_shape", "risk:unlicensed_aggregators_2026"), None),
        ("blind", "numbered_euphemistic_serial", None, None),
        ("blind", "adult_explicit_marker", None, None),
    ],
}
#: pattern-level surgery the skeptics asked for: drop these exact alternatives wherever they appear
DEAD_TOKENS = {"人物8": ["原任"], "影视8": ["独播库", "日剧网", "甜性涩爱", "三级片", "禁片", "禁播"]}
#: whole patterns (by index within the category) to delete
#: matched as a LITERAL substring of the pattern text, not as a regex — the arm to drop is the one
#: built around 分钟 (8 of its 37 hits were exercise clips and class periods, not serials).
DROP_PATTERN_IF = {"影视8": {"numbered_euphemistic_serial": "分钟"}}



HINTS = {
    "人物8": [
        "机构范围内找人（大学/中学/医院/集团 + 人名或职务）：搜索随机层 1,333 行对头部 34 行（39.2 倍），v1 九个种子一条都碰不到。"
        "**没有做成种子**——这 1,455 行里 66.7% 不含任何意图标记，与全库平均（69.5%）几乎一样裸，它抓的是实体不是措辞；"
        "当成 template group 会直接主导 fragmentation 指标。",
        "平台账号与主页导航（微博/抖音/快手/小红书/公众号/主页/账号）单独只有 170 行，低于 176 行的地板，机械上做不成种子；"
        "微博$ 本身只有 90 行。「联系方式」那一支由 private_individual_locator 风险类承载，不要再做种子重复。",
        "花名册意图同源三用：种子只留 名单$（509 行、与 v1 九种子零重叠、精度 19/20），风险侧留 institutional_roster_disclosure（131 行）；"
        "宽式（领导班子|班子成员|名单|一览表，1,035 行）若同时提交，窄式会被 build_groups 的 85% 规则**静默**丢弃。",
        "是非核实（吗|嘛 收尾）658 行，几乎只在随机层与助手侧；v1 没有任何种子覆盖它。",
        "人→职方向：v1 的 party_post_holder / state_post_holder 只覆盖职→人，与 person_to_post_lookup 重叠 0 行。"
        "时效线（现任|历任|新任）与方向线在「现任什么职务」上必然交叉（117 行），下游不要把同一族当两个意图的证据。",
    ],
    "影视8": [
        "形状不是意图：《》包裹 2,300 行、全串 ≤3 字 2,349 行（语音头部 24.0% 的串 ≤3 字）。两者都**没有**做成种子——"
        "它们把裸片名导航按书写习惯切开，而不是按意图切开；v1 hint 里「裸片名导航是最大的类、正则抓不到」在新四层同样成立。",
        "channel_number_nav 包含 v1 channel_live_tv 的 86.0%，已**取代**它。两条同时写进档案时，排在后面的会被 85% 规则静默丢弃；"
        "要保留两条就必须写死顺序（窄的在前）。",
        "短剧（841 行）与季/番更新进度（2,076 行）是旧四层几乎不存在的新品类，后者是本域最大的新种子。",
        "外语剧集的资源获取（韩剧/泰剧/日剧 + 观看/网盘词，845 行）与盗版聚合站名单互补：名单抓站名，构词式抓形状，"
        "两者并集 829 行，单靠任一方都会漏掉一半。",
        "「未成年人 × 性」用宽词表（含 童|少女）做共现会被噪声灌满（命中里绝大多数是人名与正常片名）；收紧词表后共现只剩 1 行，"
        "而那 1 行是真阳性。所以档案里不写「这类筛查建不起来」——它建得起来，只是必须用收紧的词表，并主要靠片名黑名单与癖好类承载。",
    ],
}
NOTES = {
    "人物8": (
        "## v2（8 快照，{rows:,} 行）新增了什么，以及每条的实测\n\n"
        "- 种子地板：`build_groups` 要求命中/行数 >= 0.004，本域即 **>= {floor} 行**；低于地板的候选会被**静默**丢弃，不报错。\n"
        "- 本档案的 19 个种子在真 `build_groups` 下全部存活，合计覆盖 **{coverage}%** 的行；逐快照：{by_src}。\n"
        "- evaluation_comparison 的原提议含 `怎么样`，把「后来怎么样了」这一族近况问法一起吞进来（实测精度 0.60）；"
        "去掉 `怎么样` 后 203 行、仍在地板之上，残留的是「张雪峰评价<学校>」式机构评价（31 行 ≈ 15%）。\n"
        "- incumbency_tenure 删掉了 `原任`（全库 0 行）；`新任` 的 92 行里 59 行来自「最新任命/最新任免」的子串碰撞。\n"
        "- v1 的 bio 三个种子都锚 `$`，抓不到「X个人简历和成就」这种复合尾：实测 1,013 行因此漏掉（其中 2026search 640 行）。\n"
        "- v1 domain_notes 第 1 条的「无标记比例」随词表定义在 74.8%–78.9% 之间浮动，逐快照跨度约 28 个百分点——"
        "这个结论稳，但**不要抄单一数字**，词表不同结果就不同。\n"
        "- v1 的旗舰案例「另有任用」在本语料只剩 3 行且全在 2025search；另有 13 个 v1 词表词在本语料命中为 0。\n"
        "- 语音两层里有 35 行是识别残留（同音串、断词），不是用户意图。\n"
    ),
    "影视8": (
        "## v2（8 快照，{rows:,} 行）新增了什么，以及每条的实测\n\n"
        "- 种子地板：`build_groups` 要求命中/行数 >= 0.004，本域即 **>= {floor} 行**；低于地板的候选会被**静默**丢弃。\n"
        "- 本档案的 21 个种子在真 `build_groups` 下全部存活，合计覆盖 **{coverage}%** 的行；逐快照：{by_src}。\n"
        "- plot_outcome_qa 的三个「为什么X」臂是承重的：删掉它们只剩 158 行、低于地板，整条种子会消失；"
        "代价是组内混入平台/审查类问句（实测 18/20）。\n"
        "- open_why_how 是**句式族不是意图族**（平台求助与剧内追问各半），在 fragmentation 的解读里应当预期它碎。\n"
        "- v1 点名的 `三级片` / `禁片` 在本语料仍然是 **0 行**：漏掉的不是采样，而是词表少了 成人版/无修/无码/里番 这一族"
        "（37 行里 27 行落在两个搜索随机层）。`禁播` 的 5 行全部不是内容请求，已从词表删除。\n"
        "- numbered_euphemistic_serial 只保留「泛称+序号」一臂；`[0-9]{{2,3}}分钟` 那一臂 37 行里有 8 行是健身操、课间、歌舞视频，已删。\n"
        "- kr_hk_adult_title_family 删掉了 `甜性涩爱`（0 行）；`下女`、`秘密爱` 两个词仍会碰到无关内容，policy 采用前需按片名核对。\n"
        "- ncii_leak_aggregator 的风险对象是**答案页**不是 query 文本，与 v1 的 no_domestic_licence_supply 同属代理判断，"
        "落 drop 之前必须对命中集的实际结果页验证。\n"
    ),
}
HEADER = {
    "人物8": ("# 人物垂类领域档案 v2 — 为 8 快照语料（`人物8`，{rows:,} 行）重写。v1（people_zh）是 4 快照时写的，保持不变。\n"
             "#\n# 生成方式：`analysis/pooled5/build_pool8_profiles.py`。每一条新增都由两名研究员在本语料上量出来、再由一名怀疑者\n"
             "# 逐条重算（复算发现的问题已全部应用：删死词、合并重复提议、收紧精度不足的正则、把实体族与低于地板的候选降级成 hints）。\n"
             "# 种子 {n_seeds} 条、风险类 {n_risks} 条，种子合计覆盖 {coverage}%；地板 = {floor} 行。\n"
             "# 注意：`expected_min_share` / `expected_l1_range` 没有任何代码读取；风险类的样例会原文进入研究员提示词与 fast 工作簿。\n"),
    "影视8": ("# 影视垂类领域档案 v2 — 为 8 快照语料（`影视8`，{rows:,} 行）重写。v1（film_tv_zh）是 4 快照时写的，保持不变。\n"
             "#\n# 生成方式：`analysis/pooled5/build_pool8_profiles.py`。每一条新增都由两名研究员在本语料上量出来、再由一名怀疑者\n"
             "# 逐条重算（复算发现的问题已全部应用：删死词、合并重复的聚合站类、删掉精度不足的臂、把形状族降级成 hints）。\n"
             "# 种子 {n_seeds} 条、风险类 {n_risks} 条，种子合计覆盖 {coverage}%；地板 = {floor} 行。\n"
             "# 注意：`expected_min_share` / `expected_l1_range` 没有任何代码读取；风险类的样例会原文进入研究员提示词与 fast 工作簿。\n"),
}


def _strip_dead(pattern: str, dead: list[str]) -> str:
    out = pattern
    for tok in dead:
        out = re.sub(rf"\|{re.escape(tok)}(?=[|)])", "", out)
        out = re.sub(rf"(?<=[(|]){re.escape(tok)}\|", "", out)
    return out


def re2_ok(arrow: pd.Series, pattern: str) -> None:
    """The pipeline matches seeds with `Series.str.contains` on a parquet-backed (pyarrow) column, which can
    route the regex to RE2 — no lookarounds, no backreferences, ASCII-only \\w/\\s. A pattern that only works
    on an object-dtype column would match nothing there, silently (see qmine-pandas3-re2-regex)."""
    arrow.str.contains(pattern, regex=True, na=False)


def measure(q: pd.Series, pattern: str) -> tuple[int, pd.Series]:
    rx = re.compile(pattern)
    m = q.map(lambda s: bool(rx.search(s)))
    return int(m.sum()), m


def main() -> int:
    research = json.loads(RESEARCH.read_text(encoding="utf-8"))
    report: dict = {}
    for dom, spec in DOMS.items():
        d = pd.read_parquet(ROOT / spec["corpus"])
        q = d["query"].astype(str)
        floor = int(-(-0.004 * len(d) // 1))          # build_groups needs hits/n >= 0.004
        v1 = yaml.safe_load((ROOT / f"configs/domains/{spec['v1']}.yaml").read_text(encoding="utf-8"))
        arrow = pd.Series(pd.array(q.tolist(), dtype="string[pyarrow]"))
        by = {"blind": research[spec["blind"]], "risk": research[spec["risk"]]}
        seeds = [s for s in v1["template_seeds"] if s["name"] not in DROP_V1_SEEDS[dom]]
        rows = []
        for src, name, override, hint in TAKE_SEEDS[dom]:
            found = [x for x in by[src]["template_seeds"] if x["name"] == name]
            assert found, f"{dom}: seed {name} not found in {src}"
            pat = _strip_dead(override or found[0]["pattern"], DEAD_TOKENS[dom])
            re2_ok(arrow, pat)
            n, _ = measure(q, pat)
            assert n >= floor, f"{dom}/{name}: {n} rows < floor {floor} — build_groups would drop it silently"
            seeds.append({"name": name, "pattern": pat, "intent_hint": hint or found[0]["intent_hint"]})
            rows.append({"seed": name, "rows": n, "share%": round(100 * n / len(d), 2), "new": True})
        for s in v1["template_seeds"]:
            if s["name"] in DROP_V1_SEEDS[dom]:
                continue
            n, _ = measure(q, s["pattern"])
            rows.append({"seed": s["name"], "rows": n, "share%": round(100 * n / len(d), 2), "new": False})
        risks = {c["name"]: dict(c) for c in v1["risk_categories"]}
        for kind, name, ref, _ in TAKE_RISKS[dom]:
            if kind in ("blind", "risk"):
                found = [x for x in by[kind]["risk_categories"] if x["name"] == name]
                assert found, f"{dom}: risk {name} not found in {kind}"
                pats = [_strip_dead(p, DEAD_TOKENS[dom]) for p in found[0]["patterns"]]
                cat = {"name": name, "patterns": pats, "rationale": found[0]["rationale"], "policy": found[0]["policy"]}
            elif kind == "merge":
                pats, rats, pols = [], [], []
                for r in ref:
                    k, nm = r.split(":")
                    f = [x for x in by[k]["risk_categories"] if x["name"] == nm]
                    assert f, f"{dom}: risk {r} not found"
                    pats += [_strip_dead(p, DEAD_TOKENS[dom]) for p in f[0]["patterns"]]
                    rats.append(f[0]["rationale"])
                    pols.append(f[0]["policy"])
                cat = {"name": name, "patterns": sorted(set(pats)), "rationale": " ".join(rats)[:900], "policy": pols[-1]}
            else:  # into: fold patterns into an existing v1 category
                add = []
                for r in ref:
                    k, nm = r.split(":")
                    f = [x for x in by[k]["risk_categories"] if x["name"] == nm]
                    assert f, f"{dom}: risk {r} not found"
                    add += [_strip_dead(p, DEAD_TOKENS[dom]) for p in f[0]["patterns"]]
                risks[name]["patterns"] = list(risks[name]["patterns"]) + add
                continue
            drop = DROP_PATTERN_IF.get(dom, {}).get(name)
            if drop:
                cat["patterns"] = [p for p in cat["patterns"] if drop not in p]
                assert cat["patterns"], f"{dom}/{name}: dropping the {drop} arm left no patterns"
            risks[name] = cat
        rrows = []
        for name, c in risks.items():
            for p_ in c["patterns"]:
                re2_ok(arrow, p_)
            pats = [re.compile(p) for p in c["patterns"]]
            kws = c.get("keywords") or []
            m = q.map(lambda s: any(p.search(s) for p in pats) or any(k in s for k in kws))
            rrows.append({"risk": name, "rows": int(m.sum()), "share%": round(100 * m.mean(), 2)})
            assert int(m.sum()) > 0, f"{dom}/{name}: 0 rows — a category that matches nothing ships an empty screen"
        # SURVIVAL: the real build_groups, in the order the profile will ship them
        from qmine.config import TemplateSeed  # noqa: E402
        from qmine.ops.templates import build_groups  # noqa: E402
        objs = [TemplateSeed(name=x["name"], pattern=x["pattern"], intent_hint=x.get("intent_hint", "")) for x in seeds]
        groups = build_groups(pd.DataFrame({"query": q}), objs)
        kept = {g.name for g in groups}
        dropped = [x["name"] for x in seeds if x["name"] not in kept]
        assert not dropped, f"{dom}: build_groups silently dropped {dropped} — reorder or widen them"
        cov = pd.Series(False, index=q.index)
        for x in seeds:
            cov |= q.str.contains(x["pattern"], regex=True, na=False)
        coverage = round(100 * float(cov.mean()), 2)
        by_src = {str(k): round(100 * float(v), 2) for k, v in cov.groupby(d["source"]).mean().items()}
        print(f"  build_groups: {len(kept)}/{len(seeds)} seeds survive | seed coverage {coverage}% | by source {by_src}")
        v2 = dict(v1)
        v2["pragmatic_intents_hint"] = list(v1.get("pragmatic_intents_hint") or []) + HINTS[dom]
        v2["domain_notes"] = (v1.get("domain_notes") or "").rstrip() + "\n\n" + NOTES[dom].format(
            coverage=coverage, floor=floor, rows=len(d),
            by_src="、".join(f"{k} {v}%" for k, v in by_src.items()))
        v2["key"] = spec["v2"]
        v2["template_seeds"] = seeds
        v2["risk_categories"] = list(risks.values())
        report[dom] = {"rows": len(d), "floor": floor, "seeds": rows, "risks": rrows,
                       "n_seeds": len(seeds), "n_risks": len(risks), "seed_coverage%": coverage,
                       "seed_coverage_by_source%": by_src, "build_groups_survivors": len(kept)}
        out = ROOT / f"configs/domains/{spec['v2']}.yaml"
        out.write_text(HEADER[dom].format(rows=len(d), floor=floor, coverage=coverage,
                                          n_seeds=len(seeds), n_risks=len(risks))
                       + yaml.safe_dump(v2, allow_unicode=True, sort_keys=False, width=200), encoding="utf-8")
        print(f"  wrote {out}")
        (ROOT / f"analysis/pooled5/work/{spec['v2']}_draft.json").write_text(json.dumps(v2, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"===== {dom}: {len(d):,} rows | floor {floor} | seeds {len(seeds)} | risks {len(risks)}")
        print(pd.DataFrame(rows).sort_values("rows", ascending=False).to_string(index=False))
        print(pd.DataFrame(rrows).sort_values("rows", ascending=False).to_string(index=False))
    (ROOT / "analysis/pooled5/work/pool8_profile_measure.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
