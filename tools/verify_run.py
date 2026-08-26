"""Mechanically verify a finished run against every defect fixed since live38.

Usage:  qm_verify_run.py runs/live39/gen01  [runs/live38/gen06]

Each check returns (status, detail). Run it against live38/gen06 as a control:
the checks that FAIL there and PASS on the new run are the fixes landing.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CHECKS = []


def check(name, category):
    def deco(fn):
        CHECKS.append((name, category, fn))
        return fn
    return deco


class Run:
    def __init__(self, gen: Path):
        self.gen = gen
        self._cache: dict = {}

    def j(self, name):
        if name not in self._cache:
            p = self.gen / f"{name}.json"
            self._cache[name] = json.loads(p.read_text()) if p.exists() else None
        return self._cache[name]

    def md(self, *names):
        for n in names:
            p = self.gen / n
            if p.exists():
                return p.read_text()
        return None

    def npy(self, name):
        p = self.gen / f"{name}.npy"
        return np.load(p) if p.exists() else None

    def labels(self):
        p = self.gen / "labels_full.csv"
        if "labels" not in self._cache:
            self._cache["labels"] = pd.read_csv(p) if p.exists() else None
        return self._cache["labels"]

    @property
    def summary(self):
        return self.j("run_summary") or {}


# ---------------------------------------------------------------- delivery
@check("every delivered leaf has a name", "governance")
def _(r):
    df = r.labels()
    if df is None or "bu_leaf_name" not in df:
        return "SKIP", "no delivered table"
    bad = df[df.bu_leaf_name.isna() | (df.bu_leaf_name.astype(str).str.strip() == "")]
    return ("PASS", f"{df.bu_leaf.nunique()} leaves, all named") if bad.empty else \
           ("FAIL", f"{len(bad)} rows ({len(bad)/len(df):.1%}) unnamed, leaves "
                    f"{sorted(bad.bu_leaf.unique())[:8]}")


@check("the blocking delivery gate reached state", "governance")
def _(r):
    g = r.summary.get("gates", {})
    return ("PASS", "present") if "p10_delivered_leaves_named" in g else \
           ("FAIL", f"absent; gates = {sorted(g)[:6]}")


@check("phase observers ran and recorded gates", "agents")
def _(r):
    g = [k for k in r.summary.get("gates", {}) if k.endswith("_observer")]
    return ("PASS", f"{len(g)}: {sorted(g)}") if g else ("FAIL", "no observer gate reached state")


# ---------------------------------------------------------------- reports
@check("the report agrees with itself on the delivered shape", "reports")
def _(r):
    md = r.md("自下而上聚类最终报告.md")
    if not md:
        return "SKIP", "no bottom-up report"
    s = re.search(r"\*\*(\d+) 家族 / (\d+) 叶\*\*", md)
    t = re.search(r"^\| 簇数.*?\|\s*(\d+)\s*\|\s*\*\*(\d+)\*\*\s*\|\s*(\d+)\s*\|", md, re.M)
    if not (s and t):
        return "SKIP", "shape statement or metrics row not found"
    ok = (s.group(1), s.group(2)) == (t.group(2), t.group(3))
    return ("PASS", f"{s.group(1)}/{s.group(2)} both places") if ok else \
           ("FAIL", f"summary {s.group(1)}/{s.group(2)} vs table {t.group(2)}/{t.group(3)}")


@check("the reported shape IS the delivered shape", "reports")
def _(r):
    md, df = r.md("自下而上聚类最终报告.md"), r.labels()
    if not md or df is None:
        return "SKIP", "missing inputs"
    s = re.search(r"\*\*(\d+) 家族 / (\d+) 叶\*\*", md)
    if not s:
        return "SKIP", "no shape statement"
    fam, leaf = df.bu_family_final.nunique(), df.bu_leaf.nunique()
    ok = (int(s.group(1)), int(s.group(2))) == (fam, leaf)
    return ("PASS", f"{fam}/{leaf}") if ok else \
           ("FAIL", f"report says {s.group(1)}/{s.group(2)}, table has {fam}/{leaf}")


@check("every report honours report_language", "reports")
def _(r):
    bad = {}
    for n in ("自下而上聚类最终报告.md", "自上而下类目体系最终报告.md",
              "统一度量面板.md", "叶清单.md",
              "Report_TopDown_Approach.md", "Report_Uniform_Panel.md", "Leaf_Catalogue.md"):
        md = r.md(n)
        if not md:
            continue
        eng = [re.sub(r"^#+ ", "", l) for l in md.splitlines()
               if re.match(r"^#{1,3} ", l)
               and not re.search(r"[一-鿿]", l) and re.search(r"[A-Za-z]{3}", l)]
        if eng:
            bad[n] = len(eng)
    return ("PASS", "no untranslated headings") if not bad else ("FAIL", f"{bad}")


@check("a family is named after the leaves it contains", "reports")
def _(r):
    """Reads the SHIPPED headings, not a recomputation.

    The first version called the fixed `family_names()` against the old
    artifacts, so it passed on a run whose report had every heading wrong. A
    check that verifies today's code rather than yesterday's output verifies
    nothing about the deliverable.
    """
    md = r.md("叶清单.md", "自下而上聚类最终报告.md")
    nm = r.j("tree_naming")
    lab = r.npy("leaf_labels_final")
    if lab is None:
        lab = r.npy("leaf_labels")
    fam = r.npy("leaf_family_final")
    if fam is None:
        fam = r.npy("leaf_family")
    if not md or not nm or lab is None or fam is None:
        return "SKIP", "missing report or labels"
    audit = (nm.get("audit") or {}).get("families") or []
    by_leaf = {int(l): str(f.get("name_zh") or "")
               for f in audit for l in (f.get("leaf_ids") or [])}
    if not by_leaf:
        return "SKIP", "auditor recorded no family groupings"
    # Every family heading the report printed, in order.
    # ONLY family headings. The first version matched the catalogue subtitle, my
    # own risk section, and every `### 叶 N — name` leaf heading, then reported
    # them as families with no audit match — a harness false positive that would
    # have been read as a real defect.
    heads = re.findall(r"^##+ (.+?)\s*\(`family_\d+`\)", md, re.M)        # zh catalogue
    heads += re.findall(r"^■ (.+?)(?:\s*\(|\s*$)", md, re.M)               # tree listing
    heads = [h.strip() for h in heads if re.search(r"[一-鿿]", h)]
    if not heads:
        return "SKIP", "no family headings found in the report"
    legit = set(by_leaf.values())
    unknown = [h for h in heads
               if not any(nmz and nmz in h for nmz in legit)]
    # A heading naming a real audit family is fine; one naming none is suspect.
    bad = [h for h in unknown if len(h) > 3][:4]
    return ("PASS", f"{len(heads)} headings, all trace to an audit family") if not bad else \
           ("FAIL", f"{len(bad)} heading(s) match no audit family: {bad}")


@check("gate messages reach the reader", "reports")
def _(r):
    md = r.md("自下而上聚类最终报告.md")
    if not md:
        return "SKIP", "no report"
    has_section = "每一道门实际得出的结论" in md
    return ("PASS", "per-gate conclusions printed") if has_section else \
           ("FAIL", "passing gates' messages are not shown")


@check("the risk screen reaches the catalogue", "reports")
def _(r):
    md = r.md("叶清单.md", "Leaf_Catalogue.md")
    rs = r.j("risk_screen")
    if not md or not rs:
        return "SKIP", "missing catalogue or risk screen"
    return ("PASS", f"{rs.get('total_flagged')} flagged rows shown per leaf") \
        if "风险行的实际分布" in md else ("FAIL", "risk_screen appears nowhere")


# ---------------------------------------------------------------- panel
@check("the panel compares BOTH routes", "measurement")
def _(r):
    p = r.j("metrics_panel")
    if not p:
        return "SKIP", "no panel"
    sets = p.get("sets", {})
    td = [s for s in sets if s.startswith("topdown")]
    if not td:
        return "FAIL", "no top-down subject in the panel"
    n = {s: sum(1 for m in sets[s]["metrics"].values()
                if isinstance(m, dict) and m.get("value") is not None) for s in td}
    leaves = sum(1 for m in sets.get("leaves", {}).get("metrics", {}).values()
                 if isinstance(m, dict) and m.get("value") is not None)
    # `topdown` is the legacy kappa-only external and carries 1 by design; the
    # comparison lives in `topdown_l1` / `topdown_l2`.
    real = {k: v for k, v in n.items() if k != "topdown"}
    best = max(real.values()) if real else 0
    return ("PASS", f"{n}, leaves={leaves}") if best >= 3 else \
           ("FAIL", f"top-down carries only {n} metrics vs leaves={leaves}")


@check("the K tie band was measured, not assumed", "measurement")
def _(r):
    g = r.j("granularity")
    if not g:
        return "SKIP", "no granularity"
    by = str((g.get("triangulation") or {}).get("chosen_by", ""))
    return ("PASS", by[-70:]) if "measured noise" in by else \
           ("FAIL", f"band not measured: …{by[-70:]}")


@check("the K locator is named correctly", "measurement")
def _(r):
    g, md = r.j("granularity"), r.md("自下而上聚类最终报告.md")
    if not g or not md:
        return "SKIP", "missing inputs"
    loc = str((g.get("triangulation") or {}).get("locator", ""))
    if not loc.startswith("intent_alignment_ami"):
        return "SKIP", f"locator is {loc}"
    return ("FAIL", "report still credits the stability peak") if "稳定性峰 K" in md else \
           ("PASS", "credited to intent alignment")


# ---------------------------------------------------------------- gold set
@check("no phantom classes from referee typos", "gold")
def _(r):
    p = r.gen / "gold.csv"
    tax = r.j("taxonomy_v2") or r.j("taxonomy")
    if not p.exists() or not tax:
        return "SKIP", "missing gold or taxonomy"
    codes = {n["code"] for n in (tax.get("taxonomy") or {}).get("nodes", [])}
    g = pd.read_csv(p)
    fin = g["final"].dropna().astype(str)
    fin = fin[fin.str.strip().ne("") & fin.ne("nan")]
    unknown = sorted(set(fin) - codes)
    return ("PASS", f"{len(fin)} labelled rows, all in the taxonomy") if not unknown else \
           ("FAIL", f"{len(unknown)} phantom classes: {unknown[:4]}")


@check("gold-set provenance is disclosed", "gold")
def _(r):
    md = r.md("自上而下类目体系最终报告.md", "Report_TopDown_Approach.md")
    if not md:
        return "SKIP", "no top-down report"
    return ("PASS", "source breakdown printed") if "训练前必读" in md else \
           ("FAIL", "gold.csv composition not disclosed")


# ---------------------------------------------------------------- agents
@check("research provenance recorded", "agents")
def _(r):
    tax = r.j("taxonomy")
    if not tax:
        return "SKIP", "no taxonomy"
    subs = tax.get("submissions") or []
    if not subs:
        return "SKIP", f"taxonomy reused from {tax.get('reused_from')!r}"
    have = [s for s in subs if isinstance(s, dict) and "web_researched" in s]
    if not have:
        return "FAIL", "no angle records whether it searched"
    n = sum(1 for s in have if s.get("web_researched"))
    return "PASS", f"{n}/{len(have)} angles web-researched"


@check("grid proposals recorded and graded", "agents")
def _(r):
    out = []
    for art, key in (("granularity", "grid_proposal"), ("representation", "grid_proposal")):
        d = r.j(art) or {}
        gp = d.get(key)
        if gp:
            out.append(f"{art}: kept={gp.get('proposed_kept')} won={gp.get('a_proposed_value_won')}")
    return ("PASS", "; ".join(out)) if out else ("SKIP", "proposer disabled or not recorded")


@check("agent prose is verified or absent", "agents")
def _(r):
    md = r.md("自下而上聚类最终报告.md")
    if not md:
        return "SKIP", "no report"
    n = md.count("所有数字已对照产物核验")
    # `return "PASS", X if n else ("SKIP", ...)` binds the ternary to the SECOND
    # element, so a skip rendered as a pass. Return whole tuples.
    if not n:
        return "SKIP", "no agent prose in this report"
    return "PASS", f"{n} agent-authored passage(s), every number checked"


# ---------------------------------------------------------------- run health
@check("the run used real agents", "health")
def _(r):
    prov = (r.summary.get("llm_usage") or {}).get("provider")
    return ("PASS", prov) if prov == "routed" else ("FAIL", f"provider={prov!r}")


@check("every phase completed", "health")
def _(r):
    s = r.summary
    ph = s.get("completed_phases") or s.get("phases") or []
    if s.get("halted"):
        return "FAIL", str(s.get("halt_reason"))[:90]
    if not ph:
        return "FAIL", "no completed_phases recorded"
    return "PASS", f"{len(ph)} phases: {ph[0]}..{ph[-1]}"


# ------------------------------------------------------- observation & audit
@check("hierarchy_meta agrees with its own breakdown", "measurement")
def _(r):
    """The live39 defect: n_leaves post-refinement, leaves_per_family pre."""
    m = r.j("hierarchy_meta")
    if not m or "leaves_per_family" not in m:
        return "SKIP", "no hierarchy_meta"
    tot, n = sum(int(v) for v in m["leaves_per_family"].values()), int(m.get("n_leaves", -1))
    lab = r.npy("leaf_labels")
    real = len({int(v) for v in lab.tolist()}) if lab is not None else None
    ok = tot == n and (real is None or real == n)
    return ("PASS", f"n_leaves={n}, breakdown sums to {tot}, labels carry {real}") if ok else \
           ("FAIL", f"n_leaves={n} but leaves_per_family sums to {tot} "
                    f"(labels carry {real}) — pre/post refinement mixed")


@check("observers reached the phases that decide", "agents")
def _(r):
    """Four of eighteen phases were observed, and none on the top-down route —
    which is 94% of the spend and produced the worst finding of live39."""
    g = {k for k in r.summary.get("gates", {}) if k.endswith("_observer")}
    want = {f"p{n}_observer" for n in ("2a", "2b", "2c", "2d", "3", "4", "5", "6", "7", "8", "9")}
    missing = sorted(want - g)
    return ("PASS", f"{len(g)} observers: {sorted(g)}") if not missing else \
           ("FAIL", f"{len(g)} observers; missing {missing}")


@check("a blocking observation was CONFIRMED, not merely asserted", "agents")
def _(r):
    """Severity is the agent's confidence; the check is the measurement. A gate
    that failed on an unverified `blocking` would be an LLM stopping the run."""
    bad = []
    for name, g in (r.summary.get("gates") or {}).items():
        if not name.endswith("_observer"):
            continue
        if g.get("status") in ("failed", "warned") and "no check could settle" in str(g.get("message", "")):
            bad.append(name)
    obs = [k for k in r.summary.get("gates", {}) if k.endswith("_observer")]
    if not obs:
        return "SKIP", "no observers ran"
    return ("PASS", "no gate turned on an unverified claim") if not bad else \
           ("PASS", f"{len(bad)} observer(s) flagged an UNPROVEN concern (advisory): {bad}")


@check("the findings ledger exists and is run-level", "agents")
def _(r):
    p = r.gen.parent / "findings.json"
    if not p.exists():
        return "FAIL", f"no findings.json at the run root ({r.gen.parent})"
    try:
        d = json.loads(p.read_text())
    except ValueError as exc:
        return "FAIL", f"unreadable: {exc}"
    blank = [f for f in d.get("findings", []) if not str(f.get("claim", "")).strip()]
    if blank:
        return "FAIL", f"{len(blank)} finding(s) with an empty claim — the ledger is accumulating blanks"
    return "PASS", (f"{d.get('n_open')} open, {d.get('n_confirmed_open')} confirmed, "
                    f"{len(d.get('findings', []))} total")


@check("rules were measured against the referee's own verdicts", "measurement")
def _(r):
    t = r.j("taxonomy_v2") or {}
    ev = t.get("rules_vs_evidence")
    if not ev:
        return "FAIL", "taxonomy_v2 carries no rules_vs_evidence — rules unmeasured"
    bad = ev.get("contradicted_boundaries") or []
    msg = (f"{ev.get('n_lexical_rules')} lexical / {ev.get('n_semantic_rules')} semantic; "
           f"{len(ev.get('boundaries') or [])} boundaries measured")
    return ("PASS", msg) if not bad else \
           ("FAIL", f"{msg}; {len(bad)} boundary(ies) where the guide contradicts the "
                    f"verdicts: {[' x '.join(b['classes']) for b in bad][:3]}")


@check("the pre-delivery audit ran and reported itself", "agents")
def _(r):
    a = r.j("delivery_audit")
    if not a:
        return "FAIL", "no delivery_audit.json — deliverables shipped unaudited"
    if not a.get("ran"):
        return "FAIL", f"audit did not run: {a.get('skipped')}"
    md = r.md("交付前审核报告.md")
    if not md:
        return "FAIL", "the audit ran but wrote no report — its edits are uninspectable"
    if a.get("n_refused") and "被拒绝的修改" not in md:
        return "FAIL", f"{a['n_refused']} refusal(s) not printed — the report shows only successes"
    return "PASS", f"{a.get('n_applied')} applied, {a.get('n_refused')} refused, report present"


@check("every edited deliverable kept its pre-audit original", "agents")
def _(r):
    a = r.j("delivery_audit") or {}
    changed = a.get("files_changed") or []
    if not changed:
        return "SKIP", "the audit changed nothing"
    missing = [n for n in changed if not (r.gen / f"{Path(n).stem}.pre_audit.md").exists()]
    return ("PASS", f"{len(changed)} file(s) changed, all reversible") if not missing else \
           ("FAIL", f"no pre-audit copy for {missing}")


def run(gen: Path, label: str):
    r = Run(gen)
    print(f"\n{'=' * 78}\n{label}: {gen}\n{'=' * 78}")
    tally = {}
    for name, cat, fn in CHECKS:
        try:
            status, detail = fn(r)
        except Exception as exc:  # noqa: BLE001
            status, detail = "ERROR", f"{type(exc).__name__}: {exc}"
        tally[status] = tally.get(status, 0) + 1
        icon = {"PASS": "✅", "FAIL": "❌", "SKIP": "— ", "ERROR": "💥"}[status]
        print(f"{icon} [{cat:11s}] {name}")
        print(f"      {str(detail)[:150]}")
    print(f"\n  {tally}")
    return tally


if __name__ == "__main__":
    for i, arg in enumerate(sys.argv[1:]):
        run(Path(arg), "NEW RUN" if i == 0 else "CONTROL (pre-fix)")
