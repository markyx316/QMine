#!/usr/bin/env python
"""医疗8 引文隐私筛查的轮次驱动：只筛「真正印出来的串」，印出新例子就再筛，直到一轮没有新串为止。

为什么是这个形状（2026-09-15 实测）：
- 前五层护栏（风控图层、硬规则、共现、被点名串、不可引类、按域正则）在报告与工作簿的例子里漏了 38 个违规串；
- DeepSeek 单独筛查对这 38 串只召回 15 串，不能单独当读者；Claude 逐串读（叙述复核员的做法）才是必需的读者，
  DeepSeek 作为第二视角取并集；
- 拦下一串，例子会换成下一行，所以要轮次推进：拦 → 重跑 → 找出新印出且没被 Claude 读过的串 → 再读。

    python p5_privacy_screen_round.py seed               # 第 1 轮前：复核员已逐串读过的报告引文记为「Claude 已读」
    python p5_privacy_screen_round.py apply r2           # 读 claude_r2/results.json（+ deepseek_r2 或 deepseek_round2），并入名单
    python p5_privacy_screen_round.py rerender r3        # 重跑四步，写出 claude_r3/ 待读分块（没有新串就不写）
    python p5_privacy_screen_round.py final              # 重做护栏版并做全量断言

状态文件（work/医疗8/privacy_screen/）：quote_block.json（不可引串名单）、claude_read.json（Claude 逐串读过的串）。
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

QM = Path(__file__).resolve().parents[2]
B = QM / "analysis/pooled5/work/医疗8/privacy_screen"
GEN = QM / "runs/med-pool8/gen01"
SC = QM / "analysis/pooled5/work/医疗8/snapshot_classes"
REPORT = GEN / "postprocessed/medical_zh_v2_意图与聚类叶_跨快照对比.zh.md"
WORKBOOK = GEN / "postprocessed/medical_zh_v2_意图与聚类叶_跨快照对比.xlsx"
SPAN = re.compile(r"[「『“]([^」』”\n]+)[」』”]")
CHUNK = 450


def _strings() -> pd.DataFrame:
    return pd.read_csv(B / "all_strings.csv", encoding="utf-8-sig", keep_default_na=False)


def _load_set(name: str) -> set[str]:
    p = B / name
    if not p.exists():
        return set()
    data = json.loads(p.read_text(encoding="utf-8"))
    return set(data["strings"] if isinstance(data, dict) else data)


def _save_set(name: str, s: set[str]) -> None:
    (B / name).write_text(json.dumps({"strings": sorted(s)}, ensure_ascii=False, indent=0), encoding="utf-8")


def printed_corpus_strings() -> set[str]:
    allq = set(_strings()["query"].astype(str))
    out = {s for s in SPAN.findall(REPORT.read_text(encoding="utf-8")) if s in allq}
    out |= {s for s in SPAN.findall((SC / "narrative.md").read_text(encoding="utf-8")) if s in allq}
    for df in pd.read_excel(WORKBOOK, sheet_name=None, header=None).values():
        out |= {c for c in df.astype(str).values.ravel() if c in allq}
    return out


def seed() -> int:
    g = _strings()
    read = set(g.loc[g["in_report"].astype(str).eq("True"), "query"].astype(str))
    _save_set("claude_read.json", read)
    print(f"claude_read seeded with {len(read)} report strings the narrative privacy reviewer read one by one")
    return 0


def apply(rnd: str) -> int:
    g = _strings()
    sid2q = dict(zip(g["sid"], g["query"].astype(str)))
    block, read = _load_set("quote_block.json"), _load_set("claude_read.json")
    res = json.loads((B / f"claude_{rnd}/results.json").read_text(encoding="utf-8"))
    added_c = set()
    for item in res:
        chunk = pd.read_csv(B / f"claude_{rnd}/chunk_{item['chunk']}.csv", encoding="utf-8-sig", keep_default_na=False)
        assert item["ok"], f"{rnd} chunk {item['chunk']}: screener returned nothing"
        assert int(item["n_read"]) == len(chunk), f"{rnd} chunk {item['chunk']}: read {item['n_read']} of {len(chunk)}"
        allowed = set(chunk["sid"])
        stray = [f["sid"] for f in item["flags"] if f["sid"] not in allowed]
        assert not stray, f"{rnd} chunk {item['chunk']}: flagged sids outside the chunk: {stray[:3]}"
        added_c |= {sid2q[f["sid"]] for f in item["flags"]}
        read |= set(chunk["query"].astype(str))
    added_d = set()
    for cand in (B / f"deepseek_{rnd}", B / f"deepseek_round{rnd.lstrip('r')}"):
        p = cand / "done_batches.jsonl"
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                added_d |= {sid2q[f["sid"]] for f in json.loads(line)["flags"]}
    new_block = block | added_c | added_d
    _save_set("quote_block.json", new_block)
    _save_set("claude_read.json", read)
    print(f"{rnd}: claude flagged {len(added_c)}, deepseek flagged {len(added_d)} (both {len(added_c & added_d)}); "
          f"block list {len(block)} -> {len(new_block)}; claude_read {len(read)}")
    return 0


def rerender(next_rnd: str) -> int:
    env = {**os.environ, "HF_HOME": str(QM / ".hf"), "P5_COHORT": "med8"}
    for step in ("p5_snapshot_classes", "p5_snapshot_figs", "p5_snapshot_report", "p5_snapshot_verify"):
        r = subprocess.run([str(QM / ".venv/bin/python"), f"analysis/pooled5/{step}.py", "医疗8"], cwd=QM, env=env,
                           capture_output=True, text=True)
        tail = [x for x in (r.stdout + r.stderr).splitlines() if "findfont" not in x and "Warning" not in x][-1:]
        print(f"{step}: exit {r.returncode} {tail}")
        if r.returncode:
            return r.returncode
    block, read = _load_set("quote_block.json"), _load_set("claude_read.json")
    printed = printed_corpus_strings()
    leaked = printed & block
    assert not leaked, f"{len(leaked)} block-listed strings are still printed"
    todo = sorted(printed - read)
    print(f"printed corpus strings {len(printed)} | read by Claude {len(printed & read)} | unread {len(todo)}")
    if todo:
        g = _strings()
        q2s = dict(zip(g["query"].astype(str), g["sid"]))
        d = B / f"claude_{next_rnd}"
        d.mkdir(exist_ok=True)
        for f in d.glob("chunk_*.csv"):
            f.unlink()
        df = pd.DataFrame({"sid": [q2s[s] for s in todo], "query": todo})
        for i in range(0, len(df), CHUNK):
            df.iloc[i:i + CHUNK].to_csv(d / f"chunk_{i // CHUNK:02d}.csv", index=False, encoding="utf-8-sig")
        df.to_csv(B / f"{next_rnd}_input.csv", index=False, encoding="utf-8-sig")
        print(f"wrote {-(-len(df) // CHUNK)} chunk(s) to {d.name}/")
    return 0


def final() -> int:
    env = {**os.environ, "HF_HOME": str(QM / ".hf")}
    r = subprocess.run([str(QM / ".venv/bin/python"), "analysis/pooled5/redact_med8_run_docs.py"], cwd=QM, env=env, capture_output=True, text=True)
    print("redact:", r.returncode, (r.stdout + r.stderr).strip().splitlines()[-4:])
    if r.returncode:
        return r.returncode
    sys.path.insert(0, str(QM / "analysis/pooled5"))
    os.environ["P5_COHORT"] = "med8"
    import p5_intent_structure as S  # noqa: E402
    block, read = _load_set("quote_block.json"), _load_set("claude_read.json")
    problems = []
    texts = {"report": REPORT.read_text(encoding="utf-8"), "narrative": (SC / "narrative.md").read_text(encoding="utf-8")}
    texts.update({p.name: p.read_text(encoding="utf-8") for p in (GEN / "postprocessed").glob("*_引文护栏版.md")})
    if len(texts) != 5:
        problems.append(f"expected report + narrative + 3 guardrail copies, found {sorted(texts)}")
    for name, t in texts.items():
        quoted = [s for s in set(SPAN.findall(t)) if s in block]
        inside = [s for s in block if len(s) >= 4 and s in t]
        if quoted or inside:
            problems.append(f"{name}: {len(quoted)} quoted and {len(inside)} embedded block-listed strings")
    for sh, df in pd.read_excel(WORKBOOK, sheet_name=None, header=None).items():
        hit = set(df.astype(str).values.ravel()) & block
        if hit:
            problems.append(f"workbook {sh}: {len(hit)} block-listed cells")
    printed = printed_corpus_strings()
    unread = printed - read
    if unread:
        problems.append(f"{len(unread)} printed corpus strings have not been read by a Claude screener")
    d = S.load("医疗8").reset_index(drop=True)
    quotable = set(d.loc[S.quotable_mask("医疗8", d).to_numpy(), "query"].astype(str))
    ex = pd.concat([pd.read_csv(SC / f"examples_{lv}.csv", encoding="utf-8-sig", keep_default_na=False) for lv in ("td_l1", "bu_leaf")])
    ex_ok = set(ex.loc[~ex["取法"].astype(str).str.startswith("不引原文"), "query"].astype(str))  # 占位行写作「不引原文（…）」
    allq = set(_strings()["query"].astype(str))
    narr_q = [s for s in SPAN.findall(texts["narrative"]) if s in allq]
    bad = [s for s in narr_q if s not in quotable or s not in ex_ok]
    if bad:
        problems.append(f"narrative: {len(bad)} quotes no longer quotable or no longer in examples")
    summ = {"block_list": len(block), "claude_read": len(read), "printed_corpus_strings": len(printed),
            "narrative_corpus_quotes": len(narr_q), "problems": problems}
    (B / "final_check.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summ, ensure_ascii=False, indent=1))
    return 1 if problems else 0


if __name__ == "__main__":
    cmd, arg = (sys.argv[1] if len(sys.argv) > 1 else ""), (sys.argv[2] if len(sys.argv) > 2 else "")
    fn = {"seed": lambda: seed(), "apply": lambda: apply(arg), "rerender": lambda: rerender(arg), "final": lambda: final()}.get(cmd)
    if fn is None:
        raise SystemExit("usage: p5_privacy_screen_round.py seed | apply <round> | rerender <next_round> | final")
    raise SystemExit(fn())
