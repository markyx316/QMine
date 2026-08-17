---
name: qmine-diagnose
description: Diagnose a failed QMine run, a warned or failed quality gate, or a metric that looks wrong — template coverage out of window, low kappa, unstable clusters, silhouette disagreeing with the chosen option, unexecuted governance. Use when a run halted, a gate failed, or the numbers look suspicious.
---

# Diagnosing a QMine run

```bash
qmine inspect <run-id> --what gates       # start here
qmine inspect <run-id> --what summary
qmine inspect <run-id> --what panel
```

Every gate carries its own `remediation`. Read it before theorising.

## Gate failures, in order of how often they happen

### `p1_template_coverage` out of the 20–40% window

**Below 20%** — the miner did not find enough. The fragmentation metric will
rest on too few rows and every α and K decision downstream inherits that
weakness. Add seed patterns to the domain profile, or raise the number of mined
affixes.

**Above 40%** — the groups are too loose and no longer imply shared intent. A
pattern matching half the corpus is a language detector. Tighten the seeds; the
selector will re-choose.

### `p2b_kappa` below 0.90

The guide is ambiguous — this is almost never carelessness. Look at
`gold_agreement.json` → `new_rules`: the referee drafts a rule for every gap a
disagreement exposed. Fold those into the taxonomy and re-annotate. Do not train
on a gold set that failed this gate; every accuracy number downstream would be
measuring agreement with noise.

A kappa of exactly 1.000 is also a failure signal — check the provider. Offline
stand-ins are deterministic functions and agree with themselves trivially.

### `p6_heldout_reproduction` below 0.98

The partition does not survive resampling: it describes this sample, not the
phenomenon. Usually K is too high. Check `granularity.json` → the stability
curve; if stability was already sagging at the chosen K, the triangulation
probably did not converge and the run said so.

### `p8_governance_executed` failed

A prescription never reached the data. This is the gate that catches a report
claiming a fix the delivered CSV does not contain. Look at `governance.json` →
`ledger` for rows still `proposed`. Every one must end `executed` (with an
evidence pointer) or `declined` (with a stated reason).

## Metrics that look wrong but are not

**"Silhouette is tiny — 0.05."** Expected, and not a problem. Short queries in
high-dimensional embedding space produce low absolute silhouette. It is advisory
here precisely because its absolute value is uninformative and its *ordering*
is actively misleading: identically-phrased queries form the tightest possible
cluster, so optimising silhouette selects for one intent split into several
phrasing-shaped families.

**"Silhouette would have chosen a different α."** That disagreement is recorded
on purpose, in `representation.json` → `silhouette_disagrees`. It is evidence the
decisive metrics did their job.

**"Fragmentation went up after governance."** Check `n_families` in the same row.
Merging families reduces the count, and fewer families fragment *less*, so
fragmentation should fall — if it rose, the merge probably joined two genuinely
different intents. Read the audit's rationale for that prescription.

**"Distillation accuracy is 0.95 — are the clusters right?"** It does not mean
that. It means the clusters are a learnable function of the representation.
Human agreement needs the gold set and the adversarial pass.

**"The ambiguous rate is 16%."** Semantic boundaries are genuinely softer than
phrasing boundaries. Report it; route those rows to a fallback. A pipeline
claiming 2% ambiguity on this kind of data is hiding something.

## Run failures

The state is checkpointed after every node, so nothing before the failure is
lost:

```bash
qmine resume <run-id>
```

`run_summary.json` → `halt_reason` names the node and the exception. A
`KeyError` on an artifact name means a phase ran without its dependency —
check whether the phase order was edited.
