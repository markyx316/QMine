#!/usr/bin/env python
"""Label search and assistant queries with ONE instrument, blind to source.

WHY. Each surface's own taxonomy was built separately and at a different
granularity (see `unified_intent_frame.py`), so comparing intent shares through
a crosswalk measures the mapping as much as the users. Here every sampled query —
search or assistant — goes through the same prompt and the same model, shuffled
together so the model cannot know which surface a row came from.

RELIABILITY IS MEASURED, NOT ASSUMED. A second model from a different lab labels
a stratified subset; agreement between them is the ceiling on how far any share
difference can be trusted. This mirrors the pipeline's own two-annotator design:
`deepseek-v4-flash` and `qwen3.8-flash` are the pair it validated for gold.

EVERY UNIQUE STRING IN EVERY CELL IS LABELLED — not a sample. Labels are a
property of the text alone, so a later change to the cleaning rules is a filter
over these labels rather than a relabelling, and share estimates carry labeller
error but no sampling error.

THE TOKEN BUDGET IS SIZED FOR A REASONING MODEL. `deepseek-v4-flash` spends its
completion budget thinking before it writes: in the pilot every failed call used
exactly 4,000 completion tokens and returned EMPTY content, while the one success
used 2,142. That is this project's documented per-role-budget failure, and a
preflight batch now aborts the run if content comes back empty.

Long pasted queries are excerpted to their first 300 characters for the prompt;
the task label is almost always stated at the start, and a 13,874-character row
would otherwise dominate its batch.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

MODELS = {
    "ds": ("DEEPSEEK_API_KEY", "https://api.deepseek.com", "deepseek-v4-flash"),
    "qw": ("QWEN_API_KEY", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1", "qwen3.8-flash"),
}
DOMAIN_CAT = {"金融": "金融", "医疗": "医疗", "教育": "教育培训", "影视": "影视动漫", "人物": "人物"}
EXCERPT = 300


def _env() -> None:
    for line in (ROOT / ".env").read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _frame():
    import importlib.util
    spec = importlib.util.spec_from_file_location("uif", ROOT / "tools" / "unified_intent_frame.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.FRAME


def system_prompt() -> str:
    cats = "\n".join(f"{k} {name}：{desc}" for k, (name, desc) in _frame().items())
    return (
        "你是一名严谨的中文查询意图标注员。你会收到若干条用户输入，它们可能来自搜索框，也可能来自 AI 助手，"
        "来源不会告诉你。请逐条判断这条输入最主要的「用户任务」，从下列 13 类中选且只选一类；"
        "同时判断它是否在陈述用户本人或其家人的具体处境。\n\n【类别】\n" + cats + "\n\n"
        "【判定规则】\n"
        "1. 只看这一行文字本身，不猜测没写出来的上下文；看不出要做什么就选 U13。\n"
        "2. 一行里有多个任务时，选用户最终想得到的那个结果。例：「五味子的功效是什么，我能吃吗」最终要个案判断 → U07。\n"
        "3. 只有名称、没有任何动作或问题 → U02；名称+官网/下载/直播/入口 → U01；名称+简介/是什么 → U04。\n"
        "4. 只写「翻译」「在线翻译」等工具名 → U01；给出了要翻译的具体文字 → U10。\n"
        "5. 编辑已上传的图片视频（去背景、换衣服、变清晰）→ U10；修改上一轮回答（再短一点、换一个）→ U11。\n"
        "6. 涉及成人色情、黑料爆料、普通人隐私 → U12，即使它同时是生成请求。\n\n"
        "【p 字段】该行陈述了用户本人或家人的具体情况（如「我血压89/65」「我孩子三岁发烧」「我男朋友说…」）"
        "→ p=1；仅泛泛提问（如「血压正常值多少」）→ p=0。\n\n"
        "【输出】只输出一个 JSON 数组，每条输入一个对象 {\"id\": 编号, \"u\": \"U01\"到\"U13\", \"p\": 0或1}，"
        "不要输出任何其他文字。"
    )


def build_cells(tiered: Path, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """All rows of the 15 comparison cells, every tier; and the unique strings to label."""
    a = pd.read_parquet(tiered / "assistant_tiered.parquet")
    s = pd.read_parquet(tiered / "search2026_tiered.parquet")
    parts = []
    for dom, cat in DOMAIN_CAT.items():
        sd = s[s.domain == dom].sort_values("wise_pv", ascending=False).head(1100)
        parts.append(pd.DataFrame({"domain": dom, "surface": "search_top1000", "query": sd["query"].astype(str),
                                   "tier": sd["tier"], "pv": sd["wise_pv"], "rank": sd["rank"]}))
        for snap, surf in (("top1k", "assistant_top1k"), ("random1k", "assistant_random1k")):
            ad = a[(a.l1 == cat) & (a.snapshot == snap)]
            parts.append(pd.DataFrame({"domain": dom, "surface": surf, "query": ad["query"].astype(str),
                                       "tier": ad["tier"], "pv": ad["search_num"], "rank": None}))
    cells = pd.concat(parts, ignore_index=True)
    uniq = pd.DataFrame({"query": cells["query"].drop_duplicates()})
    uniq = uniq.sample(frac=1.0, random_state=seed).reset_index(drop=True)   # blind: surfaces interleaved
    uniq["id"] = uniq.index + 1
    return cells, uniq


def build_sample(tiered: Path, per_cell: int, seed: int) -> pd.DataFrame:
    a = pd.read_parquet(tiered / "assistant_tiered.parquet")
    s = pd.read_parquet(tiered / "search2026_tiered.parquet")
    rows = []
    for dom, cat in DOMAIN_CAT.items():
        su = s[(s.domain == dom) & (s.tier == "user")].sort_values("wise_pv", ascending=False)
        cells = {
            "search_top1000": su.head(1000).drop_duplicates("query"),
            "assistant_top1k": a[(a.l1 == cat) & (a.snapshot == "top1k") & (a.tier == "user")].drop_duplicates("query"),
            "assistant_random1k": a[(a.l1 == cat) & (a.snapshot == "random1k") & (a.tier == "user")].drop_duplicates("query"),
        }
        for surface, cell in cells.items():
            take = cell.sample(min(per_cell, len(cell)), random_state=seed)
            for q in take["query"].astype(str):
                rows.append({"domain": dom, "surface": surface, "query": q})
    df = pd.DataFrame(rows)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)   # blind: surfaces interleaved
    df["id"] = df.index + 1
    return df


def _excerpt(q: str) -> str:
    q = re.sub(r"\s+", " ", q).strip()
    return q if len(q) <= EXCERPT else q[:EXCERPT] + "…"


def _call(model_key: str, batch: pd.DataFrame, sysmsg: str, raw_log: Path) -> dict[int, tuple[str, int]]:
    from openai import OpenAI
    key, base, model = MODELS[model_key]
    client = OpenAI(api_key=os.environ[key], base_url=base, timeout=180)
    user = "待标注：\n" + "\n".join(f"[{i}] {_excerpt(q)}" for i, q in zip(batch["id"], batch["query"]))
    want = set(batch["id"].tolist())
    got: dict[int, tuple[str, int]] = {}
    for attempt in range(3):
        try:
            t0 = time.time()
            r = client.chat.completions.create(
                model=model, temperature=0, max_tokens=16000,
                messages=[{"role": "system", "content": sysmsg}, {"role": "user", "content": user}])
            text = r.choices[0].message.content or ""
            with raw_log.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"model": model, "ids": sorted(want), "attempt": attempt,
                                     "usage": [r.usage.prompt_tokens, r.usage.completion_tokens],
                                     "finish": r.choices[0].finish_reason, "secs": round(time.time() - t0, 1),
                                     "text": text}, ensure_ascii=False) + "\n")
            arr = json.loads(text[text.index("["): text.rindex("]") + 1])
            for o in arr:
                i, u, p = int(o.get("id")), str(o.get("u", "")).upper(), int(o.get("p", 0))
                if i in want and re.fullmatch(r"U(0[1-9]|1[0-3])", u):
                    got[i] = (u, 1 if p else 0)
            if want <= set(got):
                return got
        except Exception as exc:  # noqa: BLE001
            with raw_log.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"model": model, "ids": sorted(want), "attempt": attempt,
                                     "error": f"{type(exc).__name__}: {str(exc)[:200]}"}, ensure_ascii=False) + "\n")
            time.sleep(2 + attempt * 3)
    return got


def run(model_key: str, df: pd.DataFrame, out: Path, batch: int, workers: int) -> pd.DataFrame:
    sysmsg = system_prompt()
    raw_log = out / f"raw_{model_key}.jsonl"
    batches = [df.iloc[i:i + batch] for i in range(0, len(df), batch)]
    res: dict[int, tuple[str, int]] = {}
    # PREFLIGHT: one batch, synchronously. An empty return here means the budget
    # or the endpoint is wrong, and spending on hundreds more calls would not fix it.
    first = _call(model_key, batches[0], sysmsg, raw_log)
    if len(first) < len(batches[0]) // 2:
        raise SystemExit(f"{model_key}: preflight labelled {len(first)}/{len(batches[0])} — "
                         f"aborting; read {raw_log}")
    res.update(first)
    done = 1
    with cf.ThreadPoolExecutor(workers) as ex:
        for got in ex.map(lambda b: _call(model_key, b, sysmsg, raw_log), batches[1:]):
            res.update(got)
            done += 1
            if done % 20 == 0 or done == len(batches):
                print(f"  {model_key}: {done}/{len(batches)} batches, {len(res):,} labelled", flush=True)
    lab = pd.DataFrame([{"id": i, f"u_{model_key}": u, f"p_{model_key}": p} for i, (u, p) in res.items()])
    return df.merge(lab, on="id", how="left")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tiered", required=True)
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--per-cell", type=int, default=300)
    ap.add_argument("--second-share", type=float, default=0.2)
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--pilot-batches", type=int, default=0)
    ap.add_argument("--full-cells", action="store_true",
                    help="label every unique string in all 15 cells instead of sampling")
    ap.add_argument("--seed", type=int, default=20260910)
    a = ap.parse_args()
    _env()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.full_cells:
        cells, uniq = build_cells(Path(a.tiered), a.seed)
        cells.to_parquet(out / "cells.parquet", index=False)
        print(f"cells: {len(cells):,} rows; unique strings to label: {len(uniq):,}", flush=True)
        t = time.time()
        lab = run("ds", uniq, out, a.batch, 16)
        print(f"deepseek labelled {lab.u_ds.notna().sum():,}/{len(lab):,} in {time.time()-t:.0f}s", flush=True)
        sub_ = uniq.sample(frac=a.second_share, random_state=a.seed)
        t = time.time()
        lab2 = run("qw", sub_, out, a.batch, 8)
        print(f"qwen labelled {lab2.u_qw.notna().sum():,}/{len(sub_):,} in {time.time()-t:.0f}s", flush=True)
        lab = lab.merge(lab2[["id", "u_qw", "p_qw"]], on="id", how="left")
        lab.to_parquet(out / "unique_labels.parquet", index=False)
        for mk in ("ds", "qw"):
            f = out / f"raw_{mk}.jsonl"
            u = [json.loads(x).get("usage") for x in f.read_text().splitlines()]
            u = [x for x in u if x]
            print(f"{mk}: {len(u)} calls, tokens in/out {sum(x[0] for x in u):,}/{sum(x[1] for x in u):,}", flush=True)
        return 0

    df = build_sample(Path(a.tiered), a.per_cell, a.seed)
    if a.pilot_batches:
        df = df.head(a.pilot_batches * a.batch)
    df.to_parquet(out / "sample.parquet", index=False)
    print(f"sample: {len(df):,} rows")
    print(df.groupby(["domain", "surface"]).size().unstack().to_string())

    t = time.time()
    df = run("ds", df, out, a.batch, 8)
    print(f"deepseek labelled {df.u_ds.notna().sum():,}/{len(df):,} in {time.time()-t:.0f}s")
    # `groupby().sample`, NOT `groupby().apply(lambda g: g.sample(...))`: under
    # pandas 3 `apply` drops the grouping columns, and the second model's subset
    # arrived with no `domain` or `surface` to stratify or report by.
    sub = df.groupby(["domain", "surface"]).sample(
        frac=a.second_share if not a.pilot_batches else 1.0, random_state=a.seed)
    t = time.time()
    lab2 = run("qw", sub[["id", "domain", "surface", "query"]], out, a.batch, 6)
    df = df.merge(lab2[["id", "u_qw", "p_qw"]], on="id", how="left")
    print(f"qwen labelled {df.u_qw.notna().sum():,}/{len(sub):,} in {time.time()-t:.0f}s")
    df.to_parquet(out / "labels.parquet", index=False)

    for mk in ("ds", "qw"):
        f = out / f"raw_{mk}.jsonl"
        if f.exists():
            u = [json.loads(x).get("usage") for x in f.read_text().splitlines()]
            u = [x for x in u if x]
            print(f"{mk}: {len(u)} calls, tokens in/out {sum(x[0] for x in u):,}/{sum(x[1] for x in u):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
