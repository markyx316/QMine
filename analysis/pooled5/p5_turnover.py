# -*- coding: utf-8 -*-
"""Where the 2025→2026 difference actually comes from, and what the two surfaces share.

A string gets exactly one label per run (verified: 0 strings with two labels in any domain), so
the year-on-year intent difference cannot contain any "the same query now means something else".
It is entirely TURNOVER — which strings entered the top 10,000 and which fell out. This script
splits it that way instead of leaving it as one number, and does the same for the strings the
two surfaces share.

    python analysis/pooled5/p5_turnover.py [domain ...]
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pooled5_common import WORK, available, cross_dir, load

LBL = "td_l1_name"


def turnover(domain: str) -> dict:
    d = load(domain)
    a = d[d.source == "2025search"]
    b = d[d.source == "2026search"]
    sa, sb = set(a["query"]), set(b["query"])
    stay, gone, new = sa & sb, sa - sb, sb - sa
    out = WORK / domain
    rows = []
    for name, keys, src in (("留存(两年都在)", stay, b), ("掉榜(只在2025)", gone, a), ("新进(只在2026)", new, b)):
        g = src[src["query"].isin(keys)]
        vc = g[LBL].value_counts(normalize=True)
        for lab, v in vc.items():
            rows.append({"bucket": name, "n": len(g), "label": lab, "share": round(float(v), 4)})
    pd.DataFrame(rows).to_csv(out / "turnover_classes.csv", index=False)
    # what the two SURFACES share, and what that shared core is about
    # TYPED ONLY for the headline number. Including the voice rows put 1,000 rows into the
    # denominator for 金融 and 医疗 and none for the other three, and voice overlaps search far
    # less than typed rows do (row level 7.4% / 1.7%; over distinct strings 4.07% / 0.79%), so it
    # silently diluted exactly those two columns and flipped the ranking. Voice is reported
    # beside it, not inside it. (Caught by the cross-domain synthesis verifier, 2026-09-13; the
    # first version of this comment quoted 0.6–0.8%, which was medical's per-year distinct-string
    # figure copied onto both domains.)
    typed = ["assistant_top", "assistant_random"]
    asst = d[d.source.isin(typed)]
    voice = d[d.source == "assistant_voice"]
    search_all = sa | sb
    shared = set(asst["query"]) & search_all
    sh = d[d["query"].isin(shared) & d.source.isin(typed)]
    only = d[~d["query"].isin(shared) & d.source.isin(typed)]
    cmp_rows = []
    for name, g in (("助手行·在搜索里也出现", sh), ("助手行·搜索里没有", only)):
        vc = g[LBL].value_counts(normalize=True)
        for lab, v in vc.items():
            cmp_rows.append({"bucket": name, "n": len(g), "label": lab, "share": round(float(v), 4)})
    pd.DataFrame(cmp_rows).to_csv(out / "shared_core_classes.csv", index=False)
    res = {"domain": domain, "n_2025": len(a), "n_2026": len(b),
           "stay": len(stay), "gone": len(gone), "new": len(new),
           "stay_share_of_2026": round(len(stay) / max(len(sb), 1), 4),
           "assistant_typed_rows": len(asst), "assistant_typed_also_in_search": len(sh),
           "assistant_typed_shared_share": round(len(sh) / max(len(asst), 1), 4),
           "voice_rows": len(voice),
           "voice_shared_share": (round(float(voice["query"].isin(search_all).mean()), 4)
                                  if len(voice) else float("nan"))}
    print(f"{domain}: 两年共有 {len(stay):,} 串（占 2026 的 {res['stay_share_of_2026']*100:.1f}%）· "
          f"掉榜 {len(gone):,} · 新进 {len(new):,} || 打字助手行里有 {res['assistant_typed_shared_share']*100:.1f}% "
          f"的字符串在搜索里也出现"
          + (f"（语音 {res['voice_shared_share']*100:.1f}%）" if len(voice) else ""))
    return res


if __name__ == "__main__":
    rs = [turnover(x) for x in (sys.argv[1:] or available())]
    pd.DataFrame(rs).to_csv(cross_dir() / "turnover_summary.csv", index=False)
