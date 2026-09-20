---
name: qmine-explain-the-method
description: Explain how QMine actually works — the two label routes, the twelve phases, what the gates do, what generations are, and which claims the method is and is not entitled to make.
whenToUse: Someone asks how the system works, what a phase or a gate is, why there are two sets of labels, how trustworthy the output is, or how to explain the method to a colleague.
---

# How QMine works

## Two routes over the same rows

Every run produces **two independent label systems** for the same corpus.

**Bottom-up (`bu_*`)** — represent the queries, cluster them, name the clusters
from their members. It finds what is actually there, including things nobody
would have thought to ask about. It is also happy to split one intent across
three clusters because of phrasing.

**Top-down (`td_l1` → `td_l2` → `td_l3`)** — researchers draft a taxonomy from
the domain, a critic attacks it, annotators apply it row by row, a referee
adjudicates the contested rows, and a classifier extends it to the whole corpus.
It produces labels a person can act on, and it can only see categories somebody
thought of.

Neither is the answer. **Where they agree, the structure is real; where they
disagree is the interesting question.** Reporting only one of them throws away
the check that makes the other believable.

## The twelve phases, in one line each

| phase | what happens |
|---|---|
| p0 | foundation: config, routing, provider check |
| p1 | corpus audit: language profile, phrasing families, risk screen, reference columns |
| p2a | taxonomy drafted by researchers, attacked by a critic, piloted |
| p2b | gold set: annotators label, a referee adjudicates, agreement is measured |
| p2c–p2e | classifier trained, validated, sub-intents derived |
| p3 | representations for the bottom-up route |
| p4–p6 | K located by intent alignment, tree built, held-out reproduction checked |
| p7 | leaves named blind, then audited |
| p8 | governance: the tree is **rewritten** — merges, splits, renames |
| p9 | the metrics panel |
| p10 | the delivered partition |
| p10b | drift / cross-snapshot distance |
| p10c | the per-class cross-snapshot comparison, tables, figures, workbooks |
| p11 | the report |
| p12 | maintenance prescriptions |

**p8 matters more than its line suggests.** It rewrites the tree, so anything
produced before it describes a tree that no longer exists. Only the **delivered**
partition (p10) is final. If a number you are quoting came from a pre-p8
artifact, say which it is.

## Gates

A gate is a mechanical check with a status, a message and a remediation. Read
them in `qmine_overview`.

- `passed` — the check ran and held.
- `warned` — the check ran and found something worth knowing; the run continues.
- `skipped` — **the check did not run.** This is not a pass. In `fast` mode most
  of the agreement gates are skipped, and `fast_skipped` names them.
- `failed` + `blocking` — the run halts.

A **confirmed** check proves the assertion failed. It does not prove a defect
exists — 2 of 13 machine-confirmed checks on one run were real defects. Say what
the check asserted, not what it implies.

## Agents describe; measurements decide

Every model in the pipeline enters through a door with a mechanical guardrail.
An agent can propose a grid, write prose, observe, name a cluster, draft a
prescription — **none of them can change a parameter or a number.** Severity in
an agent's output is a confidence, never an authority. This is why the numbers
survive the models being wrong.

## Generations

`runs/<id>/genNN/`. Re-running a run id opens a **new generation** rather than
overwriting the old one, and the paid model calls replay from a run-level cache,
so a report can be rebuilt without paying again. The old generation stays as
evidence. When you cite a number, cite its generation — two generations of one
run can legitimately differ.

## What the method is entitled to claim

- That a partition reproduces on held-out rows (p6 measures it).
- That two independent label systems agree or disagree, and where.
- That a difference between snapshots is or is not larger than same-source noise.
- In full mode, how much two annotators agreed, and on what they did not.

## What it cannot claim

- **Why** anything is the way it is. It measures structure, not cause.
- That a taxonomy is *the* taxonomy. It is one defensible cut, and a second run
  would draw a different one.
- Anything about a minority language below ~5% of the corpus — those collapse
  into one junk cluster, and a multilingual encoder does not fix it.
- Anything at all, when the provider reads `offline`: that output was written by
  a deterministic stand-in.
