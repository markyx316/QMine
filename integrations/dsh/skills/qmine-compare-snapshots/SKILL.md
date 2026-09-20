---
name: qmine-compare-snapshots
description: Read and write a cross-snapshot comparison correctly — the axis, the same-source noise ceiling, absence verdicts, and the sentences each statistic does and does not license.
whenToUse: The study pooled two or more snapshots and the person wants to know what changed, what differs between interfaces, or whether a difference is real.
---

# Comparing snapshots

## 0. The comparison must exist

`qmine_list_runs` reports `has_cross_snapshot_comparison` per run. If it is
false, the run has its labels but not the cross-snapshot layer;
`qmine_build_comparison` builds it from the run's own artifacts — **no model
call, no money, re-runnable**. Say that before you run it: it writes files.

## 1. Settle the axis before you choose a verb

`time` — the snapshots are different periods of the same thing.
`stratum` — they are different samples of ONE period: two interfaces, two
sampling depths, head versus tail.

The arithmetic is identical. **The prose inverts.** On a `stratum` axis there is
no "grew", no "fell", no "since last year" — there is only "the assistant
surface carries more of X than search does".

The axis is an **assumption**, not a measurement. Cross-file overlap cannot
settle it: measured on this project, two random samples of the *same* surface a
year apart shared 0.02% of their strings, two head exports of that surface shared
63.3%, and a *different* surface shared 0.00%. Overlap separates head from tail,
not time from interface. An undeclared axis defaults to `stratum` — the safer
error — and is reported as an assumption. If the person knows better, they can
say so and `qmine_build_comparison` takes `axis`.

## 2. The noise ceiling is the whole point

Every distance ships beside a **same-source noise ceiling** (同源噪声上界): pool
both snapshots, re-split at the observed sizes, and see how far apart two halves
of ONE source land. That is what "no difference" looks like at these n.

- distance **below** the ceiling → the data cannot tell the snapshots apart.
  That is not "they are the same"; it is "this sample cannot answer it".
- distance **above** the ceiling → a difference the sampling noise does not
  explain. Now it is worth a sentence.

Never quote a TVD, a Cramér's V or a share gap without its ceiling beside it.
A ceiling drawn at half the sample size runs 1.4×–1.8× too high and hides real
differences — this one is built at the observed sizes, which is why it can be
trusted.

## 3. A zero is a measurement with a bound

`qmine_class` returns an absence verdict for a class missing from a snapshot:

| verdict | means |
|---|---|
| `真缺席` | the snapshot was big enough that a real share would almost certainly have shown up |
| `偏少但证据弱` | consistent with being rarer, but also with chance |
| `不可判定` | the snapshot is too small to distinguish absence from bad luck |

Report the verdict and its one-sided bound ("zero of 950 rows is consistent with
a share up to 0.39%"). A bare 0 is the most misread number this system produces.

## 4. Which number answers which question

| question | field |
|---|---|
| how different are the snapshots overall? | per-level TVD, with its ceiling |
| how strong is the association at all? | Cramér's V |
| is this class's share really different? | Newcombe interval on the difference |
| how precise is one snapshot's share? | Wilson interval |
| is a class concentrated in one snapshot? | 均衡指数 / 均衡归属% |
| how big is the sample that actually carries traffic? | 流量有效n |

`qmine_glossary` defines each one. Shares are computed on `pv_norm` — traffic
normalised to 10,000 **within each snapshot** — so a share is a share of that
snapshot's traffic, never of raw rows and never across snapshots.

## 5. Writing the finding

Order the report by what the ceiling licenses, not by effect size:

1. differences clearly above the ceiling, with their intervals;
2. absences with a `真缺席` verdict;
3. everything the sample cannot resolve, named as such rather than omitted.

Then ground each one in rows: `qmine_examples` for the class and snapshot you
just described. Every printed row goes through the seven-layer quote guard; never
paste text from anywhere else.

## 6. What this comparison cannot do

- It cannot compare **two runs**. One run derives one taxonomy; two runs over the
  same rows produced 20 and 19 classes sharing zero codes. Pool into one run.
- It cannot tell you **why**. It measures that a class moved, not what caused it.
- It cannot upgrade an assumption into a measurement. An axis the person asserted
  stays an assumption, and this comparison never scores its confidence as high
  because of it.
