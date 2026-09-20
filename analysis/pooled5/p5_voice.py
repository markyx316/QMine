# -*- coding: utf-8 -*-
"""What is different about the VOICE export (finance, medical) — before any intent label.

The voice file carries no PV, no stratum and no punctuation, and its longest strings stop at
around 40 characters. Those are properties of the INSTRUMENT (ASR + whatever truncation the
export applies), and they have to be measured before any difference in intent is read as a
difference in what users want. Everything here is computed against the typed assistant rows and
the 2026 search rows of the same domain, so each marker has two reference points.

    python analysis/pooled5/p5_voice.py > work/voice_profile.txt
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pooled5_common import FORM, SRC_ZH, WORK, work_file

ROOT = Path(__file__).resolve().parents[2]
VOICE_DOMS = ["金融", "医疗"]
# 句末语气词 ONLY. The bare characters are a trap in this corpus: `吧` matches 股吧 (a huge
# finance search term) and `哈` matches 哈尔滨/哈啰, so the unanchored version read 14.3% on
# finance SEARCH — an artefact, not speech. Anchored at the end it reads 0.1%.
SPOKEN = {"句末语气词": r"(啊|呢|吧|嘛|哦|噢|嗯|哈)[\s？?。！!~～]*$", "填充词": r"(那个|就是说|然后就|我想问|请问一下|我想知道|帮我看看|你帮我)",
          "称呼助手": r"(小度|豆包|助手|你好)", "数字串": r"\d{3,}", "标点": r"[，。？！、,.?!]",
          "空格": r"\s", "英文字母": r"[A-Za-z]{2,}"}


def load_src(domain: str) -> pd.DataFrame:
    return pd.read_parquet(ROOT / f"data/raw/pooled5/{domain}_pooled5.parquet")


def main() -> None:
    rows, dupes = [], []
    for dom in VOICE_DOMS:
        d = load_src(dom)
        print(f"\n================ {dom} · 语音 query 的仪器特征 ================")
        for src in ("assistant_voice", "assistant_top", "assistant_random", "2026search"):
            g = d[d.source == src]
            if not len(g):
                continue
            q = g["query"].astype(str)
            r = {"domain": dom, "source": src, "n": len(q), "distinct": int(q.nunique()),
                 "len_median": float(q.str.len().median()), "len_p95": float(q.str.len().quantile(.95)),
                 "len_max": int(q.str.len().max()), "len>=38字%": round(100 * float((q.str.len() >= 38).mean()), 2)}
            # 股吧/贴吧 END in 吧 and are forum names, not speech: finance search read 14.1%
            # on the anchored particle rule until they were stripped, and 0.2% after.
            q_p = q.str.replace(r"(股吧|贴吧)$", "", regex=True)
            for k, pat in SPOKEN.items():
                col = q_p if k == "句末语气词" else q
                r[k] = round(100 * float(col.str.contains(pat, regex=True).mean()), 1)
            for k, pat in FORM.items():
                r[f"形态·{k}"] = round(100 * float(q.str.contains(pat, regex=True).mean()), 1)
            rows.append(r)
        v = d[d.source == "assistant_voice"]["query"].astype(str)
        vc = v.value_counts()
        rep = vc[vc > 1]
        print(f"  重复串 {int(rep.sum() - len(rep))} 行来自 {len(rep)} 个字符串；最常见：")
        for s, c in rep.head(8).items():
            print(f"    ×{c}  {s[:40]}")
        dupes += [{"domain": dom, "query": s, "n": int(c)} for s, c in rep.items()]
        typed = set(d[d.source.isin(["assistant_top", "assistant_random"])]["query"].astype(str))
        srch = set(d[d.source.isin(["2025search", "2026search"])]["query"].astype(str))
        print(f"  与打字助手行逐字重合 {100*len(set(v) & typed)/len(set(v)):.1f}% ；与搜索行逐字重合 {100*len(set(v) & srch)/len(set(v)):.1f}%")
        cut = v[v.str.len() >= 38]
        print(f"  疑似被截断（≥38字）{len(cut)} 行，示例：")
        for s in cut.head(4):
            print(f"    {s}")
    t = pd.DataFrame(rows)
    WORK.mkdir(parents=True, exist_ok=True)
    t.to_csv(work_file("voice_profile.csv"), index=False)
    pd.DataFrame(dupes).to_csv(WORK / "voice_repeats.csv", index=False)
    cols = ["domain", "source", "n", "distinct", "len_median", "len_p95", "len_max", "len>=38字%"] + list(SPOKEN)
    print("\n================ 汇总 ================")
    x = t[cols].copy()
    x["source"] = x["source"].map(SRC_ZH)
    print(x.to_string(index=False))
    f = [c for c in t.columns if c.startswith("形态·")]
    y = t[["domain", "source"] + f].copy()
    y["source"] = y["source"].map(SRC_ZH)
    y.columns = [c.replace("形态·", "") for c in y.columns]
    print("\n形态标记（行占比 %）")
    print(y.to_string(index=False))


if __name__ == "__main__":
    main()
