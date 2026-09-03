#!/usr/bin/env python
"""Cross-run figures for the README, rendered from `tools/run_evidence.py`.

WHY THESE EXIST AS A SCRIPT. Every figure here compares thirteen runs. Hand-drawn
or hand-copied, each would go stale the next time a run lands, and a stale figure
is read as current — the failure mode the fast-mode banner's hardcoded "13
documents" already demonstrated once.

WHY LIGHT AND DARK. GitHub serves READMEs in the reader's theme, and a white PNG
on a dark page is a glare block. Each figure is written twice and referenced from
a `<picture>` element with `prefers-color-scheme`, which is GitHub's supported
mechanism for exactly this.

WHY NEAR-SQUARE. The README column is ~890 px. The repo's existing panorama
figures are 2350x669 (aspect 3.5), so they render ~250 px tall with sub-pixel
Chinese labels — unreadable without clicking through. Everything here targets
1.2-1.6:1 and is legible at column width.

TWO FIGURES THIS DELIBERATELY DOES NOT DRAW.

* **Cost against corpus size.** It would be a fabrication. 60.3% of live44's and
  59.7% of med04's tokens belong to roles the catalogue had no price for, and
  those fall back to a frontier rate — so their dollar figures cover ~40% of the
  work. Rows do not drive cost either: med04 is 10k rows and the most expensive
  run in the set, live38 is 49,999 rows at $1.10 (519 of 577 calls were cache
  hits — the price of a replay). `fig_cost_coverage` draws the COVERAGE instead,
  so the dollar figure cannot be read apart from how much of the run it covers.
* **A kappa bar chart over all runs.** Nine of thirteen runs have no kappa at all
  (fast mode is single-annotator). Plotting them as zero, or omitting them
  silently, both lie.

Usage:  HF_HOME=$(pwd)/.hf .venv/bin/python tools/readme_figures.py [--out docs/img]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

#: Column width of a rendered README on github.com, in CSS pixels. Figures are
#: sized so that text is legible AT this width, not at native resolution.
GH_COLUMN_PX = 890

THEMES = {
    "light": dict(fg="#1f2328", muted="#656d76", grid="#d0d7de", bg="#ffffff",
                  a="#0969da", b="#bf3989", warn="#9a6700"),
    "dark":  dict(fg="#e6edf3", muted="#8b949e", grid="#30363d", bg="#0d1117",
                  a="#58a6ff", b="#f778ba", warn="#d29922"),
}

#: Which corpus each run id is. Run ids are not self-describing and the whole
#: point of these figures is the comparison ACROSS corpora.
DOMAIN = {
    "live38": "K12", "live39": "K12", "live40": "K12", "live42": "K12",
    "live44": "K12", "fin01": "finance", "fin02": "finance", "fin03": "finance",
    "fin-pool": "finance", "med04": "medical", "med-pool": "medical",
    "film-pool": "film/TV", "filmdrift": "film/TV",
    "edu-pool": "education", "ppl-pool": "people",
}


def _style(ax, t):
    ax.set_facecolor(t["bg"])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=t["muted"], labelsize=9, length=0)
    ax.xaxis.label.set_color(t["muted"])
    ax.title.set_color(t["fg"])


def fig_route_agreement(rows, t, path):
    """Do the two routes agree? Measured, across every run that recorded it.

    The README's central design claim is that a clustering tree and an intent
    taxonomy answer different questions and must both be delivered. This is the
    evidence: across every run that measured it they land between 0.31 and 0.68
    and never near 1.0, so neither route is recoverable from the other.

    The title deliberately does NOT say the spread is a property of the corpus.
    It looks that way — 0.309 on film/TV against 0.679 on education — but the
    four finance runs alone span 0.510 to 0.641, so run-to-run variation is a
    live competing explanation and this figure cannot separate them.

    A dot plot, not bars: AMI is a position on a bounded 0-1 scale with no
    meaningful zero-anchored area, and two paired series per row read as dots
    but collide as bars.
    """
    d = sorted([r for r in rows if r.get("ami_leaf") is not None],
               key=lambda r: r["ami_leaf"])
    if not d:
        return None
    y = range(len(d))
    fig, ax = plt.subplots(figsize=(8.2, 5.6), dpi=110)
    fig.patch.set_facecolor(t["bg"])
    for i, r in enumerate(d):
        ax.plot([r["ami_family"], r["ami_leaf"]], [i, i], color=t["grid"], lw=2, zorder=1)
    ax.scatter([r["ami_family"] for r in d], y, s=58, color=t["b"], zorder=3, label="family level")
    ax.scatter([r["ami_leaf"] for r in d], y, s=58, color=t["a"], zorder=3, label="leaf level")
    ax.set_yticks(list(y))
    ax.set_yticklabels([f"{r['run']}  ·  {DOMAIN.get(r['run'], '?')}" for r in d])
    ax.set_xlim(0, 1)
    ax.set_xlabel("AMI between the top-down taxonomy and the bottom-up tree  (1.0 = identical)")
    ax.set_title("How much do the two routes agree? 0.31 to 0.68 — never near 1",
                 fontsize=11.5, pad=14, loc="left")
    ax.grid(axis="x", color=t["grid"], lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    _style(ax, t)
    leg = ax.legend(loc="lower right", frameon=False, fontsize=9)
    for txt in leg.get_texts():
        txt.set_color(t["muted"])
    fig.text(0.01, 0.015,
             f"{len(d)} of {len(rows)} complete live runs; live38/live39/live40 predate this "
             "measurement and are absent, not zero.\n"
             "The four finance runs alone span 0.510-0.641, so the spread is not corpus alone.",
             color=t["muted"], fontsize=8)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(path, facecolor=t["bg"])
    plt.close(fig)
    return path


def fig_cost_coverage(rows, t, path):
    """What a run costs, inseparable from how much of it the price covers.

    Reporting `$61.09` alone would be the money version of quoting a kappa
    without its n. The bar is the priced share; the dollar figure sits at the end
    of it, so the eye cannot take one without the other.
    """
    d = sorted([r for r in rows if r.get("cost_usd")], key=lambda r: -r["cost_usd"])
    if not d:
        return None
    fig, ax = plt.subplots(figsize=(8.2, 5.4), dpi=110)
    fig.patch.set_facecolor(t["bg"])
    for i, r in enumerate(d):
        cov = r["priced_share"]
        # 0-13% unpriced is routine across eleven runs; 40% is not. A cutoff at
        # 0.9 painted 88% the same colour as 40% and made the routine look alarming.
        ax.barh(i, cov * 100, color=t["a"] if cov >= 0.75 else t["warn"], height=0.62)
        ax.text(cov * 100 + 1.5, i, f"${r['cost_usd']:.2f}", va="center",
                fontsize=9, color=t["fg"])
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels([f"{r['run']}  ·  {r['mode']}  ·  {r['rows']:,} rows" for r in d])
    ax.invert_yaxis()
    ax.set_xlim(0, 118)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("share of the run's tokens that had a published price  (%)")
    ax.set_title("Read the price with its coverage", fontsize=11.5, pad=14, loc="left")
    ax.grid(axis="x", color=t["grid"], lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    _style(ax, t)
    fig.text(0.01, 0.015,
             "Amber: a large share of this run's tokens belong to roles whose model publishes no "
             "price, so the figure is a lower bound.\n"
             "live38's $1.10 is a replay, not a run — 519 of its 577 calls were cache hits.",
             color=t["muted"], fontsize=8)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(path, facecolor=t["bg"])
    plt.close(fig)
    return path


def fig_delivered_shape(rows, t, path):
    """The delivered shape is NOT stable across runs of the same corpus.

    This started as a portability figure — same pipeline, six corpora, six
    shapes. The data says something more useful and less flattering: the four
    K12 runs are the SAME 49,999 rows and deliver 25 to 53 leaves and 7 to 23
    families. Run-to-run variation on one corpus is as wide as the variation
    across six different ones.

    That is the empirical foundation for two rules this project already follows:
    never diff two runs' labels (pool the snapshots into one run instead), and
    never quote a shape from one run as a property of the corpus.

    Grouped by corpus so the within-corpus spread is the thing the eye lands on.
    """
    order = ["K12", "finance", "medical", "education", "people", "film/TV"]
    d = sorted(rows, key=lambda r: (order.index(DOMAIN.get(r["run"], "film/TV")),
                                    -(r["leaves"] or 0)))
    fig, ax = plt.subplots(figsize=(8.2, 5.8), dpi=110)
    fig.patch.set_facecolor(t["bg"])
    y = list(range(len(d)))
    for i, r in enumerate(d):
        ax.plot([r["families"], r["leaves"]], [i, i], color=t["grid"], lw=2, zorder=1)
    ax.scatter([r["families"] for r in d], y, s=54, color=t["b"], zorder=3, label="families")
    ax.scatter([r["leaves"] for r in d], y, s=54, color=t["a"], zorder=3, label="delivered leaves")

    # Bracket the corpus that is measured four times, because that is the finding.
    k12 = [i for i, r in enumerate(d) if DOMAIN.get(r["run"]) == "K12"]
    if len(k12) > 1:
        lo, hi = min(k12), max(k12)
        span = [r for r in d if DOMAIN.get(r["run"]) == "K12"]
        # Draw the bracket in AXES coordinates: the data x-range starts near 7,
        # so a data-space x of 4.2 fell outside the axes and rendered nothing.
        ax.annotate("", xy=(0.035, 1 - (lo + 0.15) / len(d)),
                    xytext=(0.035, 1 - (hi + 0.85) / len(d)),
                    xycoords="axes fraction", textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="-", color=t["warn"], lw=1.8))
        ax.text(0.055, 1 - (lo + hi + 1) / (2 * len(d)),
                f"one corpus, four runs: {min(r['leaves'] for r in span)}-"
                f"{max(r['leaves'] for r in span)} leaves",
                transform=ax.transAxes, color=t["warn"], fontsize=8.5, va="center")

    ax.set_yticks(y)
    ax.set_yticklabels([f"{DOMAIN.get(r['run'], '?'):9s} {r['run']}  ·  {r['rows']:,}" for r in d],
                       fontfamily="monospace")
    ax.set_xlabel("count in the DELIVERED partition (after phase-8 governance)")
    ax.set_title("The same corpus does not give the same tree twice",
                 fontsize=11.5, pad=14, loc="left")
    ax.grid(axis="x", color=t["grid"], lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    ax.invert_yaxis()
    _style(ax, t)
    leg = ax.legend(loc="lower right", frameon=False, fontsize=9)
    for txt in leg.get_texts():
        txt.set_color(t["muted"])
    fig.text(0.01, 0.015,
             "Delivered counts, not phase-7's — governance rewrites the tree before delivery.\n"
             "This is why two runs cannot be diffed, and why snapshots are pooled into one run.",
             color=t["muted"], fontsize=8)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(path, facecolor=t["bg"])
    plt.close(fig)
    return path


def fig_churn_vs_drift(t, path, data):
    """The surface churns; the intents persist. Measured on five verticals.

    This is the argument for an intent layer, and it is not obvious. Between the
    same two dates one year apart, film/TV recycles only 16.6% of its distinct
    queries and only 36.7% of its traffic sits on queries that existed a year
    earlier — the titles turn over almost completely — yet its intent
    composition moved LESS (total variation 0.203) than the people corpus, whose
    queries are twice as stable (jaccard 0.290) but whose intent mix moved most
    (0.288). A pipeline that tracked query strings would call film/TV the most
    changed corpus and people one of the calmest. Both readings are wrong.

    Two independent axes, so a scatter — the point is precisely that position on
    one does not predict position on the other.
    """
    fig, ax = plt.subplots(figsize=(7.6, 5.6), dpi=110)
    fig.patch.set_facecolor(t["bg"])
    for name, jac, tv, note, dx, dy in data:
        ax.scatter(jac, tv, s=170, color=t["a"], zorder=3, alpha=0.85)
        ax.annotate(f"{name}\n{note}", (jac, tv), textcoords="offset points",
                    xytext=(dx, dy), fontsize=9, color=t["fg"],
                    ha="right" if dx < 0 else "left")
    ax.set_xlabel("query overlap between the two years  (Jaccard over distinct queries)")
    ax.set_ylabel("intent-mix movement  (total variation, traffic-weighted)")
    ax.set_title("The queries churn and the intents persist — independently",
                 fontsize=11.5, pad=14, loc="left")
    ax.set_xlim(0.05, 0.62)
    ax.set_ylim(0.05, 0.34)
    ax.grid(color=t["grid"], lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    _style(ax, t)
    ax.yaxis.label.set_color(t["muted"])
    fig.text(0.01, 0.015,
             "Five verticals, 2025-07-01 vs 2026-07-01, ~20,000 pooled rows each, one taxonomy "
             "per corpus labelling both years.\n"
             "Percentages beside each point: the share of THIS year's traffic sitting on queries "
             "that also existed last year.",
             color=t["muted"], fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, facecolor=t["bg"])
    plt.close(fig)
    return path


def fig_governance_shift(rows, t, path):
    """The K locator does not set the family count — phase-8 governance does.

    A reader who sees "K location" in the phase diagram reasonably concludes the
    located K is the delivered shape. It is not: governance then isolates leaves
    that do not belong with their siblings, each becoming its own family. The
    delivered count exceeds the located K in **13 of 14 runs**, by a median of
    2.3x and up to 4.0x (med-pool, 5 to 20).

    This is the mechanism behind the run-to-run variance in
    `fig_delivered_shape`, and it is why a family count quoted from one run says
    little about the corpus.
    """
    d = sorted([r for r in rows if r.get("located_k") and r.get("families")],
               key=lambda r: r["families"] / r["located_k"])
    if not d:
        return None
    fig, ax = plt.subplots(figsize=(8.2, 5.6), dpi=110)
    fig.patch.set_facecolor(t["bg"])
    y = list(range(len(d)))
    for i, r in enumerate(d):
        # live40 delivered exactly the located K; an arrow there rendered as a
        # backwards stub between two coincident dots.
        if r["families"] != r["located_k"]:
            ax.annotate("", xy=(r["families"], i), xytext=(r["located_k"], i),
                        arrowprops=dict(arrowstyle="->", color=t["grid"], lw=1.9))
    ax.scatter([r["located_k"] for r in d], y, s=52, color=t["muted"], zorder=3,
               label="families the K locator chose")
    ax.scatter([r["families"] for r in d], y, s=52, color=t["a"], zorder=3,
               label="families actually delivered")
    for i, r in enumerate(d):
        note = (f"+{r['isolated']} isolated" if r["families"] != r["located_k"]
                else f"unchanged ({r['isolated']} isolated, absorbed)")
        ax.text(r["families"] + 0.9, i, note,
                va="center", fontsize=7.8, color=t["muted"])
    ax.set_yticks(y)
    ax.set_yticklabels([f"{DOMAIN.get(r['run'], '?'):9s} {r['run']}" for r in d],
                       fontfamily="monospace")
    ax.set_xlabel("number of families")
    ax.set_title("Governance sets the family count, not the K locator",
                 fontsize=11.5, pad=14, loc="left")
    ax.set_xlim(0, max(r["families"] for r in d) + 12)
    ax.grid(axis="x", color=t["grid"], lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    _style(ax, t)
    leg = ax.legend(loc="lower right", frameon=False, fontsize=9)
    for txt in leg.get_texts():
        txt.set_color(t["muted"])
    fig.text(0.01, 0.015,
             "Delivered count exceeds the located K in 13 of 14 runs; median 2.3x, up to 4.0x.\n"
             "Governance isolates a leaf that does not belong with its siblings, and each "
             "isolate becomes its own family.",
             color=t["muted"], fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, facecolor=t["bg"])
    plt.close(fig)
    return path


def _priced_share(run: str, gen: str) -> float:
    p = f"runs/{run}/{gen}/run_summary.json"
    try:
        with open(p, encoding="utf-8") as fh:
            u = (json.load(fh).get("llm_usage") or {})
    except Exception:  # noqa: BLE001
        return 1.0
    by = u.get("by_role") or {}
    unp = sum(by.get(r, {}).get("input_tokens", 0) + by.get(r, {}).get("output_tokens", 0)
              for r in (u.get("unpriced_roles") or []))
    tot = (u.get("input_tokens") or 0) + (u.get("output_tokens") or 0)
    return 1.0 if not tot else max(0.0, 1.0 - unp / tot)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="docs/img")
    ap.add_argument("--evidence", default="/tmp/qmine_evidence.json")
    a = ap.parse_args()

    if not os.path.exists(a.evidence):
        subprocess.run([sys.executable, "tools/run_evidence.py", "--json", a.evidence], check=True)
    with open(a.evidence, encoding="utf-8") as fh:
        rows = json.load(fh)
    for r in rows:
        r["priced_share"] = _priced_share(r["run"], r["generation"])

    os.makedirs(a.out, exist_ok=True)
    # Measured with ops/drift.py over the five pooled corpora (labels joined to
    # the pooled source POSITIONALLY; education uses labels_full_reconciled.csv
    # because one empty-text row was dropped at p1).
    # dx/dy are hand-placed label offsets: medical and education sit close enough
    # that the default placement overlapped their two labels into one blur.
    churn = [("medical", 0.4630, 0.111, "78% of traffic recurs", -14, 6),
             ("education", 0.5053, 0.103, "78%", 14, -10),
             ("finance", 0.3743, 0.216, "81%", 14, -2),
             ("people", 0.2898, 0.288, "56%", 14, -2),
             ("film/TV", 0.1656, 0.203, "37%", 14, -2)]
    made = []
    for theme, t in THEMES.items():
        p = fig_churn_vs_drift(t, os.path.join(
            a.out, f"fig_churn_vs_drift{'' if theme == 'light' else '_dark'}.png"), churn)
        made.append(p)
    # located K and isolation count, for fig_governance_shift
    for r in rows:
        g = f"runs/{r['run']}/{r['generation']}"
        try:
            with open(f"{g}/granularity.json", encoding="utf-8") as fh:
                r["located_k"] = (json.load(fh).get("triangulation") or {}).get("chosen_family_k")
        except Exception:  # noqa: BLE001
            r["located_k"] = None
        try:
            with open(f"{g}/governance.json", encoding="utf-8") as fh:
                ex = (json.load(fh).get("execution") or {})
            r["isolated"] = len((ex.get("isolations") or {}).get("isolated") or [])
        except Exception:  # noqa: BLE001
            r["isolated"] = 0

    for name, fn in (("fig_route_agreement", fig_route_agreement),
                     ("fig_governance_shift", fig_governance_shift),
                     ("fig_cost_coverage", fig_cost_coverage),
                     ("fig_delivered_shape", fig_delivered_shape)):
        for theme, t in THEMES.items():
            suffix = "" if theme == "light" else "_dark"
            p = fn(rows, t, os.path.join(a.out, f"{name}{suffix}.png"))
            if p:
                made.append(p)
    print(f"{len(made)} files written to {a.out}/")
    for m in made:
        print("  ", m)
    print(f"\nSized for a ~{GH_COLUMN_PX}px README column; reference with a <picture> element "
          "so the dark variant is served in dark mode.")


if __name__ == "__main__":
    main()
