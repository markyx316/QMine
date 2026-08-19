# Language and unknown domains

Two questions this pipeline has to answer before it can be pointed at an
arbitrary query log: *what happens to queries that are not in the main
language*, and *what happens when nobody has written a profile for the vertical*.

Both answers below are measured on real corpora rather than assumed.

---

## 1. Minority-language queries collapse into a junk bucket

The setting: a log that is mostly Chinese with a small English minority — the
normal case for a Chinese-market product with some international traffic.

**Measured, on a 10,000-row Chinese corpus with English mixed in, K=20:**

| English share | biggest English cluster | that cluster's purity | clusters holding ≥5% of English |
|---|---|---|---|
| 2% | **97% of all English rows** | 100% English | 1 |
| 8% | 52% of all English rows | 100% English | 2 |

At a realistic 2% minority, essentially every English query — whatever it wanted
— lands in one 100%-English cluster. The tree acquires a family that means
"these are the English ones" and resolves none of their intents.

This is not really an encoder defect. When a minority language is also a
minority of the *content*, language is the largest systematic difference in the
data, and a partitional algorithm finds the largest difference first.

### Swapping in a multilingual encoder does not fix it

The obvious move is to reach for a multilingual model. Measured on a
parallel-intent probe — the same ten intents written in Chinese and English:

| encoder | same-intent cosine | different-intent cosine | **separation** | correct translation ranked #1 |
|---|---|---|---|---|
| `bge-base-zh-v1.5` (monolingual) | 0.496 | 0.232 | **+0.264** | **100%** |
| `multilingual-e5-small` | 0.867 | 0.758 | **+0.109** | 90% |

The multilingual model has a far higher *absolute* same-intent similarity and a
distinctly worse *contrast*, because it is anisotropic — everything sits near
everything, baseline 0.758. On the clustering task it still put 97.5% of English
into one cluster.

This is the playbook's Principle 3 in a new costume: the number that looks like
the objective (raw similarity) is not the objective (telling intents apart). So
`ops/language.alignment_probe` decides on **separation and rank-1 accuracy**,
reports raw cosine, and gives it no vote — and it is meant to be run on *your*
corpus, not trusted from this table.

### What the pipeline does about it

1. **Phase 1 measures the composition** and classifies the corpus as
   `monolingual`, `minority_at_risk` (0.5–5%), or `genuinely_multilingual`. The
   middle band is the dangerous one: too small to earn clusters, too big to lose.
   A warning gate fires.
2. **`tokenizer: auto`** resolves from the measured script mix rather than from
   an assumption. A Chinese tokeniser on a Latin corpus is not a subtle loss.
3. **Phase 6 resolves intents inside minority families** using a
   script-appropriate character n-gram space, where language is constant and
   therefore uninformative. On the mixed corpus this split a 398-row English
   bucket into comparison queries, spec lookups and how-tos.
4. **That result ships as a column, not as tree leaves.** A leaf here is
   *defined* as a centroid in the hybrid space, and Phase 10 deploys exactly
   that rule. Sub-clusters found in another space are not centroid regions of
   the deployed one — measured at 91.8% consistency against 98% for ordinary
   leaves, and injecting them cost held-out reproduction (0.9743 → 0.9706). So
   they become `minority_sub_intent` in the delivered table and the tree keeps
   its invariant.

### The honest limit, and the recommendation

**The hybrid space cannot express intent within a minority language.** The facet
recovers the intents; it does not make them first-class.

Also measured: on the mixed corpus, held-out structure reproduction was **0.974**
— below the 0.98 deliverable threshold that the pure-Chinese corpus cleared at
0.991. Adding 2% foreign content genuinely destabilises the tree.

So if a minority language matters commercially, **give it its own tree and its
own centroid model** and dispatch on language at serving time. One space serving
two languages is a compromise the numbers do not support. The pipeline makes
this visible rather than deciding it for you.

---

## 2. An unknown or mixed vertical

`configs/domains/generic.yaml` is the profile for a log with no known vertical.
Its design rule: **supply what is universal, invent nothing that is
domain-specific.**

- **Zero template seeds.** A phrasing family is exactly the thing that differs
  between verticals, so seeding them would be guessing.
- **Seven universal risk categories** — gambling, fraud, self-harm, minors and
  adult content, regulated professional advice, weapons, personal-data exposure
  — with patterns in both Chinese and English, since the dominant language is
  unknown at profile time. A floor to extend, never a ceiling.
- **Seven universal pragmatic-intent hints.** These are facts about how language
  encodes intent, not facts about a vertical: verification, procedural how-to,
  navigational, transactional, troubleshooting, opinion-seeking, noise.

### Bootstrapping template families with no seeds

Phase 1 mines candidate affixes from the corpus. Phase 3 then makes each one
**earn** the claim a seed gets for free — *everything matching this is one
intent* — by being measurably tighter than chance: mean pairwise cosine among
members, divided by the same statistic over random rows, must clear 1.35.

**Measured on the K12 corpus with the profile's seeds removed entirely:**

| discovered group | lift over random | verdict |
|---|---|---|
| `笔顺` (stroke order) | 2.30 | trusted |
| `近义词` (synonyms) | 1.91 | trusted |
| `怎么读` (pronunciation) | 1.87 | trusted |
| `拼音` (pinyin) | 1.86 | trusted |
| `组词` (word formation) | 1.79 | trusted |
| … | | |
| `有哪些` ("which ones are there") | 1.33 | **rejected** |
| `什么意思` ("what does it mean") | 1.23 | **rejected** |
| `是什么` ("what is") | **0.976** | **rejected** |

21 of 26 trusted. With no seeds at all, the procedure rediscovered five of the
six families the K12 profile hand-writes — and rejected the interrogatives.
`是什么` scores at chance because "what is X" attaches to *every* topic: it is a
question form, not an intent, and it fails the one standard that matters.

Note that it also rejects `什么意思`, which the K12 profile *does* seed. That is
not a bug in either direction — the playbook itself observed that 释义 queries
fragment by topic under pure semantics. The validator rediscovered that
independently. Seeds keep their human vouch and are retained; the disagreement
is reported.

### Closing the loop

A `domain_scout` agent reads a stratified sample when the profile is generic and
returns a vertical, candidate seeds, vertical-specific risk categories, and
pragmatic-intent hints — explicitly as *hypotheses*. Its seeds go through the
same cohesion test as mined ones. Its findings are written back so the second
run on that vertical starts warm.

This is the same posture used everywhere else in the pipeline: the profile is a
prior, the corpus is an observation, and the observation wins.
