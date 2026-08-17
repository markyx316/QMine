---
name: qmine-interpret-tree
description: Read and explain a finished QMine cluster tree, leaf catalogue, or delivered label table — what the families mean, where the two label systems agree, which clusters are risky, what to build from the results. Use when a run is complete and the user wants to understand or act on the output.
---

# Interpreting QMine output

## The two label systems are not competing

`labels_full.csv` carries both, side by side, and they answer different
questions:

- `td_l1` — a human-designed intent. Includes categories that are invisible in
  the wording: verification versus definition, solve-this versus explain-this.
- `bu_family_final` / `bu_leaf` — how the data actually organises itself.
  Content-shaped, discovered without supervision, drifts with the corpus.

Where they disagree, neither is wrong. A disagreement usually marks a *pragmatic*
intent — one clustering is structurally blind to — and those rows are worth
reading, because they are where a purely unsupervised system would silently
mis-serve users.

Neither column overwrites the other. Downstream consumers pick per use case:
supply-side planning wants the families, response routing wants the intents.

## Reading the leaf catalogue

Each leaf has a `user_need` sentence: *"having received X, the user is
satisfied."* That sentence is doing three jobs at once — annotation guideline,
acceptance criterion, and downstream product requirement — which is why a name
alone is not enough. Names are ambiguous; the sentence is checkable.

Two leaves whose `user_need` sentences are interchangeable should be one leaf.
If you find such a pair after governance, that is a finding worth reporting.

## Using the results

**Ranking work by traffic.** Group `labels_full.csv` by `bu_family_final` and
weight by frequency. The biggest family is usually the most templated and the
easiest to serve well.

**Finding the gaps.** Filter `bu_ambiguous == True`. These sit between clusters
and are where a naive classifier would be confidently wrong. They are also the
best candidates for the next annotation round.

**Shipping the classifier.** The deployed model is a centroid matrix — a few
hundred kilobytes. Inference is `encode(query) → hybrid transform → argmax(x @
centroids.T)`. There is no train/serve skew possible, because that expression
*is* the assignment rule the tree was built with rather than a model fitted to
its output.

**Watching for drift.** `maintenance.json` holds the baseline. Re-run quarterly
(monthly for fast-moving verticals), and compare `config_hash` **first** — if it
changed, the two trees are not comparable and any apparent drift may be method
change.

**Replacing labels with a newer run's.** Do not just overwrite them:

```bash
qmine promote --old runs/q1/gen01/labels_full.csv --new runs/q2/gen01/labels_full.csv
```

This judges only the disagreements, blind and with randomised side order, and
promotes only on a statistically significant win. A newer model is not evidence
of a better model, and this is the cheap way to find out which you have.

## What to say about the numbers

Be precise about what each one licenses:

| number | what it means | what it does not mean |
|---|---|---|
| stability ARI | the partition survives a seed change | the partition is semantically right |
| held-out reproduction | structure generalises to unseen rows | the clusters match human categories |
| template fragmentation | one intent stays in one family | anything about intents with no phrasing family |
| distillation accuracy | clusters are learnable | agreement with human judgment |
| coherence | blind reviewers found members consistent | reviewers were right |
| κ | annotators applied the guide consistently | the guide is correct |

And check `run_summary.json` → `llm_usage.provider` before describing any
agent-produced artifact. If it is `offline`, the names and definitions came from
a deterministic heuristic, not a model, and every summary you write must say so.
