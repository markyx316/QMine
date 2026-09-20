#!/usr/bin/env python
"""医疗8 引文隐私筛查（DeepSeek 视角）：把语料里**每一个不同的查询串**按共用标准筛一遍。

为什么需要：med-pool8 的叙述复核员在报告表格引文里查出约 45 行违规（未成年人与性、露骨描述、具名医生、民营医院名、
试管选性别），全部穿过了风控图层、硬规则、共现、被点名串、不可引类与按域第五层——词表只能拦写得出来的写法，
语音头部这类题材串写法太杂。所以改为「印出来的每一个串都被读过」：本脚本读全部不同串，另有 Claude 视角读高风险子集，
两边标的取并集，写进按域的不可引串名单（`p5_snapshot_classes.SCREENED_QUOTE_BLOCK`）。

    HF_HOME=$(pwd)/.hf .venv/bin/python analysis/pooled5/p5_privacy_screen_deepseek.py

输入：work/医疗8/privacy_screen/{criteria.md, all_strings.csv}
输出：work/医疗8/privacy_screen/deepseek/{raw.jsonl, flags.csv, summary.json}；已完成的批次可断点续跑。
失败处理：一个批次被服务商的内容过滤拒收时，对半拆开重试，拆到单条仍被拒收的串**记为标出**（码 PROVIDER_REFUSED）——
服务商拒收本身就是这串敏感的强信号，宁可多拦。其它错误（网络、解析、覆盖不全）重试 3 次后报错退出，不静默丢。
"""
from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
import pandas as pd

QM = Path(__file__).resolve().parents[2]
BASE = QM / "analysis/pooled5/work/医疗8/privacy_screen"
# SCREEN_SUBSET=printed：只读报告、叙述与护栏版里真正印出来的串（快速路径）；SCREEN_INPUT 可指定另一份 sid,query 表
SUBSET = os.environ.get("SCREEN_SUBSET", "all")
OUT = BASE / ("deepseek" if SUBSET == "all" else f"deepseek_{SUBSET}")
MODEL = "deepseek-v4-pro"
URL = "https://api.deepseek.com/chat/completions"
BATCH = 150
WORKERS = int(os.environ.get("SCREEN_WORKERS", "20"))
ATTEMPTS = 6
CODES = {"MINOR_SEX", "EXPLICIT_SEX", "NAMED_PERSON", "NAMED_PRIVATE_FACILITY", "SELF_HARM_CRISIS", "ILLEGAL_REPRO", "PII"}


def _key() -> str:
    for line in (QM / ".env").read_text(encoding="utf-8").splitlines():
        if line.startswith("DEEPSEEK_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("DEEPSEEK_API_KEY missing from QMine/.env")


class Refused(Exception):
    pass


def _call(criteria: str, rows: list[tuple[str, str]], key: str) -> str:
    listing = "\n".join(f"{sid}\t{q}" for sid, q in rows)
    msgs = [
        {"role": "system", "content": "你是一名严格的中文内容合规审核员。只按给定标准判断，输出 JSON。"},
        {"role": "user", "content": f"{criteria}\n\n下面是 {len(rows)} 个待审串（sid<TAB>串）：\n{listing}\n\n"
                                    '只输出 JSON：{"flags": [{"sid": "...", "code": "...", "confidence": "高|中", "reason": "..."}], '
                                    '"n_read": 读过的串数}。没有要标的就输出 {"flags": [], "n_read": N}。'},
    ]
    body = {"model": MODEL, "messages": msgs, "temperature": 0, "response_format": {"type": "json_object"}, "max_tokens": 8000}
    with httpx.Client(timeout=600) as c:
        r = c.post(URL, json=body, headers={"Authorization": f"Bearer {key}"})
    if r.status_code == 400 and re.search(r"risk|content|sensitive|inappropriate", r.text, re.I):
        raise Refused(r.text[:200])
    r.raise_for_status()
    j = r.json()
    ch = j["choices"][0]
    if ch.get("finish_reason") == "content_filter":
        raise Refused("finish_reason=content_filter")
    return ch["message"]["content"]


def screen(criteria: str, rows: list[tuple[str, str]], key: str, log: list) -> list[dict]:
    want = {sid for sid, _ in rows}
    err = ""
    for attempt in range(ATTEMPTS):
        try:
            text = _call(criteria, rows, key)
        except Refused as e:
            if len(rows) == 1:
                log.append({"sids": [rows[0][0]], "refused": str(e)})
                return [{"sid": rows[0][0], "code": "PROVIDER_REFUSED", "confidence": "中", "reason": "服务商内容过滤拒收"}]
            mid = len(rows) // 2
            log.append({"sids": sorted(want), "refused_split": str(e)})
            return screen(criteria, rows[:mid], key, log) + screen(criteria, rows[mid:], key, log)
        except Exception as e:  # network / HTTP (incl. 429 rate limits and 5xx): back off and retry
            err = f"{type(e).__name__}: {str(e)[:200]}"
            time.sleep(10 * (attempt + 1))
            continue
        log.append({"sids": sorted(want), "response": text})
        try:
            t = text.strip()
            m = re.search(r"```(?:json)?\s*(.*?)```", t, re.S)
            j = json.loads(m.group(1) if m else t)
            flags = j.get("flags") or []
            bad = [f for f in flags if f.get("sid") not in want or f.get("code") not in CODES]
            if bad:
                raise ValueError(f"{len(bad)} flags with unknown sid or code, e.g. {bad[0]}")
            if int(j.get("n_read", len(rows))) < len(rows):
                raise ValueError(f"n_read {j.get('n_read')} < {len(rows)}")
            return flags
        except Exception as e:
            err = f"parse: {str(e)[:200]}"
    raise RuntimeError(f"batch of {len(rows)} failed after {ATTEMPTS} attempts: {err}")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    criteria = (BASE / "criteria.md").read_text(encoding="utf-8")
    g = pd.read_csv(BASE / "all_strings.csv", encoding="utf-8-sig", keep_default_na=False)
    if os.environ.get("SCREEN_INPUT"):
        want = set(pd.read_csv(os.environ["SCREEN_INPUT"], encoding="utf-8-sig", keep_default_na=False)["sid"])
        g = g[g["sid"].isin(want)].reset_index(drop=True)
    elif SUBSET == "printed":
        g = g[g["in_report"].astype(str).eq("True") | g["in_docs"].astype(str).eq("True")].reset_index(drop=True)
    rows = list(zip(g["sid"], g["query"].astype(str)))
    batches = [rows[i:i + BATCH] for i in range(0, len(rows), BATCH)]
    done_path = OUT / "done_batches.jsonl"
    done = set()
    if done_path.exists():
        # a killed run can leave a half-written last line: keep only complete records, rewrite the file, then append
        valid = []
        for line in done_path.read_text(encoding="utf-8").splitlines():
            try:
                j = json.loads(line)
            except json.JSONDecodeError:
                continue
            if j.get("batch") not in done:
                done.add(j["batch"])
                valid.append(line)
        done_path.write_text("".join(x + "\n" for x in valid), encoding="utf-8")
    key = _key()
    todo = [i for i in range(len(batches)) if i not in done]
    print(f"distinct strings {len(rows):,} | batches {len(batches)} | already done {len(done)} | to run {len(todo)}", flush=True)

    def run(i: int) -> tuple[int, list, list]:
        log: list = []
        return i, screen(criteria, batches[i], key, log), log

    with ThreadPoolExecutor(max_workers=WORKERS) as ex, done_path.open("a", encoding="utf-8") as dp, \
            (OUT / "raw.jsonl").open("a", encoding="utf-8") as rp:
        futs = [ex.submit(run, i) for i in todo]
        for k, fu in enumerate(as_completed(futs), 1):
            i, flags, log = fu.result()
            for x in log:
                rp.write(json.dumps({"batch": i, **x}, ensure_ascii=False) + "\n")
            dp.write(json.dumps({"batch": i, "flags": flags}, ensure_ascii=False) + "\n")
            dp.flush()
            rp.flush()
            if k % 20 == 0 or k == len(futs):
                print(f"{time.strftime('%H:%M:%S')} {k}/{len(futs)} batches", flush=True)
    flags = []
    for line in done_path.read_text(encoding="utf-8").splitlines():
        j = json.loads(line)
        flags += [{"batch": j["batch"], **f} for f in j["flags"]]
    covered = {json.loads(line)["batch"] for line in done_path.read_text(encoding="utf-8").splitlines()}
    assert covered == set(range(len(batches))), f"batches not covered: {sorted(set(range(len(batches))) - covered)[:10]}"
    fl = pd.DataFrame(flags, columns=["batch", "sid", "code", "confidence", "reason"]).drop_duplicates(["sid", "code"])
    fl = fl.merge(g[["sid", "query", "sources", "quotable_any", "in_report"]], on="sid", how="left")
    fl.to_csv(OUT / "flags.csv", index=False, encoding="utf-8-sig")
    summ = {"model": MODEL, "distinct_strings": len(rows), "batches": len(batches), "flagged_strings": int(fl["sid"].nunique()),
            "by_code": fl.groupby("code")["sid"].nunique().to_dict(), "flagged_in_report": int(fl.drop_duplicates("sid")["in_report"].sum())}
    (OUT / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summ, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
