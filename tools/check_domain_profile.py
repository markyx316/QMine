#!/usr/bin/env python
"""Measure a domain profile's template seeds and risk categories against a corpus.

A `TemplateSeed`'s contract is that *everything matching this regex is almost
certainly the same intent* — the groups judge the alpha sweep and are the
denominator of template fragmentation. That is a claim about data, so it can be
checked against data instead of eyeballed, which is what this does.

Reports, per seed: rows matched, share of rows, share of PV, and how many of
those rows ALSO match another seed. Overlap is the number that matters: a query
in two seeds is a query the miner cannot assign to one intent, and a seed with
high overlap is too broad whatever its coverage looks like.

    python tools/check_domain_profile.py configs/domains/finance_zh.yaml \
        data/raw/金融query-250701.xlsx --text-column original_query --weight wise_pv
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
import yaml


def _load(path: str, text: str, weight: str | None):
    p = Path(path)
    df = pd.read_excel(p) if p.suffix in (".xlsx", ".xls") else pd.read_csv(p)
    if text not in df.columns:
        sys.exit(f"--text-column {text!r} not in {list(df.columns)}")
    w = df[weight] if weight and weight in df.columns else pd.Series(1, index=df.index)
    return df[text].astype(str), w


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("profile")
    ap.add_argument("corpus")
    ap.add_argument("--text-column", default="original_query")
    ap.add_argument("--weight", default="wise_pv")
    ap.add_argument("--show", action="store_true",
                    help="print sample matches per seed — the only way to check "
                         "the one-intent claim, which no count can verify")
    a = ap.parse_args()
    args_show = a.show

    prof = yaml.safe_load(Path(a.profile).read_text(encoding="utf-8"))
    q, w = _load(a.corpus, a.text_column, a.weight)
    total_pv = w.sum() or 1

    seeds = prof.get("template_seeds") or []
    masks = {}
    print(f"\n{'=' * 78}\n{a.profile}  vs  {a.corpus}   (n={len(q):,})\n{'=' * 78}")
    print(f"\nTEMPLATE SEEDS ({len(seeds)})")
    print(f"  {'seed':28s} {'rows':>7s} {'%rows':>7s} {'%pv':>7s} {'overlap':>8s}")
    for s in seeds:
        try:
            m = q.str.contains(s["pattern"], regex=True, na=False)
        except re.error as e:
            print(f"  {s['name']:28s}  BAD REGEX: {e}")
            continue
        masks[s["name"]] = m
    for name, m in masks.items():
        others = [o for n2, o in masks.items() if n2 != name]
        ov = (m & pd.concat(others, axis=1).any(axis=1)).sum() if others else 0
        flag = "  <-- AMBIGUOUS" if m.sum() and ov / m.sum() > 0.25 else ""
        print(f"  {name:28s} {m.sum():7,} {m.mean() * 100:6.1f}% "
              f"{w[m].sum() / total_pv * 100:6.1f}% {ov:8,}{flag}")
        # THE SEED'S CONTRACT IS ABOUT INTENT, WHICH NO COUNT CAN CHECK.
        # Coverage and overlap are necessary and not sufficient: a regex can
        # match 500 rows, overlap nothing, and still sweep two intents into one
        # group — which is exactly what corrupts the alpha sweep it judges. So
        # print a spread of what it actually caught (head, middle, tail) and
        # read them. If they are not obviously the same ask, the seed is wrong
        # however good its numbers look.
        if args_show and m.sum():
            hits = q[m].tolist()
            picks = [hits[0]] if len(hits) < 3 else [hits[0], hits[len(hits) // 2], hits[-1]]
            print(f"      {' | '.join(x[:26] for x in picks)}")
    if masks:
        any_seed = pd.concat(list(masks.values()), axis=1).any(axis=1)
        print(f"  {'TOTAL COVERAGE':28s} {any_seed.sum():7,} {any_seed.mean() * 100:6.1f}% "
              f"{w[any_seed].sum() / total_pv * 100:6.1f}%")
        # p1 reports coverage but does not gate on it; below ~20% the template
        # fragmentation metric rests on a small and noisy base.
        if any_seed.mean() < 0.20:
            print("  ⚠ coverage below 20% — template fragmentation will rest on a thin base")

    risks = prof.get("risk_categories") or []
    print(f"\nRISK CATEGORIES ({len(risks)})")
    for r in risks:
        pats = list(r.get("patterns") or []) + [re.escape(k) for k in (r.get("keywords") or [])]
        if not pats:
            print(f"  {r['name']:28s}  NO PATTERNS")
            continue
        try:
            m = q.str.contains("|".join(pats), regex=True, na=False)
        except re.error as e:
            print(f"  {r['name']:28s}  BAD REGEX: {e}")
            continue
        ex = " / ".join(q[m].head(2).tolist())[:52]
        print(f"  {r['name']:28s} {m.sum():7,} {m.mean() * 100:6.2f}% "
              f"{w[m].sum() / total_pv * 100:6.2f}%  {ex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
