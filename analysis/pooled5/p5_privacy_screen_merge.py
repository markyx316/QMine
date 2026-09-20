#!/usr/bin/env python
"""医疗8 引文隐私筛查：合并两名筛查员与复核员的标记，写出不可引串名单，并做发布前的全量核对。

    HF_HOME=$(pwd)/.hf .venv/bin/python analysis/pooled5/p5_privacy_screen_merge.py merge
    HF_HOME=$(pwd)/.hf .venv/bin/python analysis/pooled5/p5_privacy_screen_merge.py check     # 重跑报告与护栏版之后

merge：
- DeepSeek：`deepseek/done_batches.jsonl`，必须覆盖全部批次（36,673 个不同串）；
- Claude：`claude/results.json`（工作流结果：每个分块的 flags、n_read、n_rows），必须每块 n_read == 行数，sid 必须属于该块；
- 复核员：`verifier_flagged_strings.json`（叙述工作流隐私复核员逐串读报告时标出的 38 串）。
三者取并集写 `quote_block.json`（隐私优先：任一方标出即不可引），并报告两名筛查员各自对复核员 38 串的召回、彼此重合与分码计数。

check：读重跑后的报告、叙述、三份护栏版文档、跨快照工作簿，断言
① 名单里没有任何一串还被「」『』引用或出现在工作簿单元格里；② 报告与叙述引用的每一个语料串都被 DeepSeek 读过；
③ 其中来自助手侧或落在敏感词表的串也被 Claude 读过；④ 叙述里的每一条引文仍然可引、仍在 examples 表里。
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

QM = Path(__file__).resolve().parents[2]
BASE = QM / "analysis/pooled5/work/医疗8/privacy_screen"
GEN = QM / "runs/med-pool8/gen01"
SC = QM / "analysis/pooled5/work/医疗8/snapshot_classes"


def _strings() -> pd.DataFrame:
    return pd.read_csv(BASE / "all_strings.csv", encoding="utf-8-sig", keep_default_na=False)


def merge() -> int:
    g = _strings()
    sid2q = dict(zip(g["sid"], g["query"].astype(str)))
    n_batches = -(-len(g) // 150)
    ds_lines = [json.loads(x) for x in (BASE / "deepseek/done_batches.jsonl").read_text(encoding="utf-8").splitlines()]
    covered = {j["batch"] for j in ds_lines}
    assert covered == set(range(n_batches)), f"DeepSeek batches missing: {sorted(set(range(n_batches)) - covered)[:10]}"
    ds = pd.DataFrame([{**f, "screener": "deepseek"} for j in ds_lines for f in j["flags"]])
    cl_raw = json.loads((BASE / "claude/results.json").read_text(encoding="utf-8"))
    cl_rows = []
    for item in cl_raw:
        chunk = pd.read_csv(BASE / f"claude_chunks/chunk_{item['chunk']}.csv", encoding="utf-8-sig", keep_default_na=False)
        assert item["ok"], f"chunk {item['chunk']}: screener returned nothing"
        assert int(item["n_read"]) == len(chunk) == int(item["n_rows"]), f"chunk {item['chunk']}: read {item['n_read']} of {len(chunk)} rows"
        allowed = set(chunk["sid"])
        stray = [f["sid"] for f in item["flags"] if f["sid"] not in allowed]
        assert not stray, f"chunk {item['chunk']}: {len(stray)} flagged sids not in the chunk, e.g. {stray[:3]}"
        cl_rows += [{**f, "screener": "claude"} for f in item["flags"]]
    cl = pd.DataFrame(cl_rows)
    ver = set(json.loads((BASE / "verifier_flagged_strings.json").read_text(encoding="utf-8")))
    ds_q = {sid2q[s] for s in ds["sid"]} if len(ds) else set()
    cl_q = {sid2q[s] for s in cl["sid"]} if len(cl) else set()
    sub = set(pd.read_csv(BASE / "claude_subset.csv", encoding="utf-8-sig", keep_default_na=False)["query"].astype(str))
    block = sorted(ds_q | cl_q | ver)
    (BASE / "quote_block.json").write_text(json.dumps({"strings": block}, ensure_ascii=False, indent=0), encoding="utf-8")
    flags = pd.concat([ds, cl], ignore_index=True)
    flags["query"] = flags["sid"].map(sid2q)
    flags.to_csv(BASE / "all_flags.csv", index=False, encoding="utf-8-sig")
    summ = {
        "distinct_strings": len(g), "claude_subset": len(sub),
        "flagged_deepseek": len(ds_q), "flagged_claude": len(cl_q), "flagged_verifier": len(ver),
        "both_screeners": len(ds_q & cl_q), "deepseek_only": len(ds_q - cl_q), "claude_only": len(cl_q - ds_q),
        "claude_only_outside_subset": len(cl_q - sub), "block_list": len(block),
        "recall_vs_verifier": {"deepseek": round(len(ver & ds_q) / len(ver), 3), "claude": round(len(ver & cl_q) / len(ver), 3),
                               "union_of_screeners": round(len(ver & (ds_q | cl_q)) / len(ver), 3)},
        "deepseek_agreement_within_claude_subset": {"deepseek_flags_in_subset": len(ds_q & sub), "claude_flags": len(cl_q),
                                                    "both": len(ds_q & cl_q)},
        "by_code_deepseek": ds.groupby("code")["sid"].nunique().to_dict() if len(ds) else {},
        "by_code_claude": cl.groupby("code")["sid"].nunique().to_dict() if len(cl) else {},
        "blocked_strings_in_current_report": int(g["query"].isin(block)[g["in_report"].astype(str).eq("True")].sum()),
    }
    (BASE / "merge_summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summ, ensure_ascii=False, indent=1))
    return 0


def check() -> int:
    sys.path.insert(0, str(QM / "analysis/pooled5"))
    import p5_intent_structure as S  # noqa: E402
    g = _strings()
    all_q = set(g["query"].astype(str))
    sub = set(pd.read_csv(BASE / "claude_subset.csv", encoding="utf-8-sig", keep_default_na=False)["query"].astype(str))
    block = set(json.loads((BASE / "quote_block.json").read_text(encoding="utf-8"))["strings"])
    report = (GEN / "postprocessed/medical_zh_v2_意图与聚类叶_跨快照对比.zh.md").read_text(encoding="utf-8")
    narr = (SC / "narrative.md").read_text(encoding="utf-8")
    guard = {p.name: p.read_text(encoding="utf-8") for p in (GEN / "postprocessed").glob("*_引文护栏版.md")}
    assert len(guard) == 3, f"expected 3 guardrail copies, found {sorted(guard)}"
    problems = []
    span = re.compile(r"[「『“]([^」』”\n]+)[」』”]")
    for name, text in [("report", report), ("narrative", narr), *guard.items()]:
        hits = [s for s in set(span.findall(text)) if s in block]
        sub_hits = [s for s in block if len(s) >= 4 and s in text]
        if hits or sub_hits:
            problems.append(f"{name}: {len(hits)} quoted spans and {len(sub_hits)} substrings are on the block list")
    x = pd.read_excel(GEN / "postprocessed/medical_zh_v2_意图与聚类叶_跨快照对比.xlsx", sheet_name=None, header=None)
    for sh, df in x.items():
        cells = set(df.astype(str).values.ravel())
        if cells & block:
            problems.append(f"workbook sheet {sh}: {len(cells & block)} cells on the block list")
    quoted = set(span.findall(report)) | set(span.findall(narr))
    corpus_quotes = {s for s in quoted if s in all_q}
    not_claude = sorted(s for s in corpus_quotes if s not in sub)
    lex = re.compile(r"性|精|孕|胎|私处|阴|乳|胸|臀|裤|女孩|女生|男孩|小孩|儿童|宝宝|岁|学生|医生|大夫|院长|主任|教授|医院|诊所|门诊|自杀|自伤|误食|过量|试管")
    risky_unread = [s for s in not_claude if lex.search(s)]
    if risky_unread:
        problems.append(f"{len(risky_unread)} quoted strings match the sensitive lexicon but were not in the Claude subset")
    d = S.load("医疗8").reset_index(drop=True)
    qm = S.quotable_mask("医疗8", d)
    quotable = set(d.loc[qm.to_numpy(), "query"].astype(str))
    ex = pd.concat([pd.read_csv(SC / f"examples_{lv}.csv", encoding="utf-8-sig", keep_default_na=False) for lv in ("td_l1", "bu_leaf")])
    ex_ok = set(ex.loc[~ex["取法"].astype(str).str.startswith("不引原文"), "query"].astype(str))  # 占位行写作「不引原文（…）」
    narr_q = [s for s in span.findall(narr) if s in all_q]
    bad_narr = [s for s in narr_q if s not in quotable or s not in ex_ok]
    if bad_narr:
        problems.append(f"narrative: {len(bad_narr)} quotes are no longer quotable or no longer in examples")
    summ = {"report_quoted_corpus_strings": len(corpus_quotes), "read_by_deepseek": len(corpus_quotes & all_q),
            "also_read_by_claude": len(corpus_quotes & sub), "deepseek_only_and_lexicon_clean": len(not_claude) - len(risky_unread),
            "narrative_corpus_quotes": len(narr_q), "block_list": len(block), "problems": problems}
    (BASE / "check_summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summ, ensure_ascii=False, indent=1))
    return 1 if problems else 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "merge":
        raise SystemExit(merge())
    if cmd == "check":
        raise SystemExit(check())
    raise SystemExit("usage: p5_privacy_screen_merge.py merge|check")
