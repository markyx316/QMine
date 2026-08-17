---
name: qmine-run
description: Run the twelve-phase query-intent mining pipeline end to end on a query log — data audit, template mining, taxonomy design, gold annotation, representation bake-off, clustering, blind naming, governance, metrics, deployment, and reports. Use when the user has a file of search queries and wants an intent taxonomy, a cluster tree, labelled data, or a deployable classifier.
---

# Running the QMine pipeline

## Before you start

Ask for, or infer, three things. Getting these wrong wastes an entire run:

1. **The file and the text column.** CSV, Parquet, or XLSX.
2. **The domain.** One of `k12_zh`, `finance_zh`, `sports_zh`, `politics_zh`,
   `ecommerce_en`, or a new profile (see the `qmine-new-domain` skill). The
   profile supplies phrasing seeds, risk categories, tokenizer, and n-gram
   ranges — a mismatched profile produces a technically valid run whose template
   groups are meaningless.
3. **Whether legacy labels exist.** Pass them as `--reference-columns`. They are
   used for measurement only and never as supervision — but they matter, because
   Phase 1 audits them and Phase 9 reports alignment against them.

## Command

```bash
qmine run \
  --input path/to/queries.csv \
  --domain k12_zh \
  --text-column query \
  --reference-columns legacy_l1,legacy_l2
```

Useful flags:

| flag | when |
|---|---|
| `--sample N` | first run on an unfamiliar corpus; keeps the loop tight |
| `--fast` | wiring check — shrinks every grid, skips HDBSCAN and UMAP |
| `--offline` | no network or no API key; see the honesty note below |
| `--human-review` | pause for sign-off after taxonomy, tree, and panel |
| `--provider anthropic` | force real agents (fails loudly without a key) |

Run `qmine doctor` first if anything looks unhealthy.

## What to expect

Roughly 20–60 minutes for 50k rows with real agents; a few minutes with
`--fast --sample 5000`. Phase 3's encoder bake-off and Phase 4's battery are the
slow parts. Everything checkpoints, so a crash resumes with
`qmine resume <run-id>`.

## Reading the result

Deliverables land in `runs/<run-id>/gen01/`:

- `Report_BottomUp_Approach.md` — representation, tree, governance, deployment
- `Report_TopDown_Approach.md` — taxonomy, gold set, classifier, validation
- `Report_Uniform_Panel.md` — the cross-candidate comparison
- `Leaf_Catalogue.md` — every leaf with its `user_need` sentence
- `labels_full.csv` — the delivered table, both label systems side by side
- `Walkthrough.ipynb` — executed, with every number computed in-cell

Start with the panel report. If a gate warned or failed, read that gate's
`remediation` before reading anything else — a downstream number computed on a
foundation that failed its own test is not worth interpreting.

## The honesty requirement

**If the run was offline, say so in your summary.** Offline mode replaces every
agent with a deterministic heuristic stand-in: cluster names come from n-gram
frequency, labels from regex evidence. The clustering, metrics, and embeddings
are entirely real; the *judgments* are not model judgments and every record they
produce is stamped `offline-heuristic`. Reporting an offline run as though
agents named the clusters is the one failure mode that would make this whole
system untrustworthy.

Check `run_summary.json` → `llm_usage.provider`. If it says `offline`, lead with
that fact.

## When a blocking gate fails

The run halts on purpose. Do not work around it — read the gate's remediation,
fix the cause, and re-run. The blocking gates exist because each one guards a
downstream conclusion that would otherwise be quietly wrong:

- `p2b_kappa` — annotators disagree, so the gold set does not define anything
- `p6_heldout_reproduction` — the partition does not survive resampling
- `p8_governance_executed` — a prescription never reached the delivered data
