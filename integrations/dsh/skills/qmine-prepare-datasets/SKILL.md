---
name: qmine-prepare-datasets
description: Turn a pile of raw exports into ONE corpus a run can mine — what to measure first, the ten named cleaning operations, and the decisions the data cannot make for anyone.
whenToUse: Someone has arrived with spreadsheets or CSVs and wants to mine or compare them, or asks what shape their data needs to be in.
---

# From raw exports to one corpus

## 0. One corpus or several?

Ask this before anything else, because it cannot be undone later.

- Files that are **snapshots of the same vertical** and are meant to be compared
  — two years, two interfaces, head versus tail — go into **ONE** run.
- Files from **different verticals** are separate runs, and their labels will not
  be comparable. That is not a limitation to route around; it is what the labels
  mean.

One run derives one taxonomy. Two runs over the same 10,000 rows produced 20 and
19 classes sharing **zero** codes.

## 1. Look before deciding anything

`qmine_inspect_inputs` measures the files and writes nothing: columns and their
types, row counts, duplicate rate, length distribution, which column looks like
the query, which looks like traffic, and any shared prefixes that smell like a
product's own wrapper.

Read it out to the person. It is also where you find the things nobody told you:
a date column, a legacy label column worth declaring as a reference, an export
that is a head sample rather than a full extract.

## 2. Let the planner propose, then read the concerns

`qmine_plan_corpus` returns a plan of **named operations** — never code, never a
free-text instruction. Only these ten exist:

| op | what it does |
|---|---|
| `trim` | strip surrounding whitespace |
| `drop_empty` | no letter, digit or CJK character — never a query |
| `drop_duplicates` | keep the first occurrence within one input |
| `length_filter` | outside `[min_len, max_len]` |
| `strip_prefix` | remove a product's own wrapper — **never drops the row** |
| `flag_regex` | tag matching rows; they still enter the mining |
| `drop_regex` | tier matching rows out of the mining |
| `head_cut` | keep the top n by weight (a declared head sample) |
| `aggregate_by_query` | one row per string, weights summed (per-day exports) |
| `merge_collisions` | after stripping, sum weights of now-identical rows |

Two rules the plan enforces so nothing disappears quietly:

- **Nothing is deleted.** Every removal assigns a *tier* — a reason, verbatim,
  like `product_answer_option` — and the removed rows ship beside the corpus.
- **A rule that would remove more than `max_drop_share` of a source is
  downgraded to a FLAG.** An over-broad pattern cannot silently eat a corpus.

The plan's `⚠` concerns are the questions the data cannot answer. Put them to the
person rather than choosing for them. The two that matter most:

1. **The comparison axis** (`time` or `stratum`) — see the
   `qmine-compare-snapshots` skill. Undeclared defaults to `stratum`.
2. **What each snapshot is called.** These names reach every table and figure.

## 3. Execute, and say where it lands

`qmine_prepare_corpus` writes: one pooled corpus, the removed rows with their
tiers, and the plan that produced them. Say the output directory before you call
it.

The corpus carries a `snapshot` column (the canonical name post-preparation) and
`pv_norm`, traffic normalised to 10,000 **within each snapshot** — which is why a
share is always a share of its own snapshot's traffic.

## 4. Then, and only then, cost it

`qmine_estimate_cost` gives the model calls and the money a full run on this
corpus would take. Hand over the number before anyone talks about starting one.
Preparation is free and re-runnable; mining is neither.

## 5. What good preparation is not

It is not cleaning until the data looks tidy. A wrapper the product added
(`帮我查一下…`) is noise and comes off with `strip_prefix`; a phrasing the user
chose is the signal and stays, however odd it looks. When you cannot tell which
one something is, `flag_regex` it and let the mining decide — a flag is
reversible, a removal is an argument you have to win first.
