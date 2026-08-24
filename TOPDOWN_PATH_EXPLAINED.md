# The top-down path, step by step

*What Phase 2 actually does, why each step exists, and how the rules and guide
get built, audited, and repaired. Written against the code as of 2026-08-24;
every number quoted is measured from a real run, not illustrative.*

English rather than Chinese on purpose: this is developer documentation, like
`HANDOFF.md`. The *deliverables* (reports, notebooks, figure labels) are Chinese.

**Every block marked "real" below is copied verbatim from `runs/live38/`** — its
`llm_cache/` (each agent's actual returned object) and its `run.log`. Nothing is
paraphrased or invented. To pull your own:

```bash
# what any agent actually returned, by role
python - <<'EOF'
import json, pathlib
for f in pathlib.Path("runs/live38/llm_cache").glob("*.json"):
    if f.name == "model_catalog.json": continue
    b = json.loads(f.read_text()); m = b.get("meta") or {}
    if m.get("role") == "taxonomy_critic":      # or referee / annotator_a / researcher_*
        print(json.dumps(b["value"], ensure_ascii=False, indent=1)[:2000])
        break
EOF

# every gate decision, with the numbers it was decided on
grep -E "gate .*(PASSED|FAILED)" runs/live38/run.log

# what each phase wrote
ls runs/live38/gen*/ ; cat runs/live38/index.jsonl | jq -r '.name, .summary'
```

`qmine inspect` reads `run_summary.json`, which is written per **generation** —
and it defaults to generation 1. Point it at the generation that actually got
somewhere, or it returns an empty table:

```bash
qmine inspect live38 --generation 2 --what gates
qmine inspect live38 --generation 2 --what summary
```

---

## 1. The one-paragraph version

The pipeline has two independent paths that meet at the end. **Bottom-up**
(phases 3–7) clusters the corpus and asks "what shapes are in this data?".
**Top-down** (phase 2) asks a different question: *"what are users actually
trying to do, including the things clustering can never see?"* Phase 2 builds a
**taxonomy** (a list of user intents), proves a human-equivalent annotator can
apply it **consistently**, produces a **gold set** of labelled queries, and
trains a **classifier** that can label the remaining 47,000 rows.

The reason it is this elaborate: **a taxonomy nobody can apply consistently is
worthless, and you cannot tell a good one from a bad one by reading it.** Every
step in phase 2 exists to make that failure visible early, while it is cheap.

---

## 2. The steps, in order

| step | what it produces | what it costs |
|---|---|---|
| **2a** research → architect → critic → rules → **pilot** | a taxonomy + adjudication rules, tested on 200 queries | ~50 min |
| **2b** gold annotation → referee → guide repair | 3,000 double-labelled queries with adjudicated verdicts | ~2 h |
| **2c** classifier | a model that labels the other 47,000 rows | minutes |
| **2d** adversarial validation | an honest accuracy estimate | minutes |
| **2e** sub-intents (L2) | a second level under each L1 class | minutes |

Everything before 2b is deliberately cheap, because 2b is where the money goes.

---

## 3. Phase 2a — building a taxonomy that can survive contact with annotators

### 3.1 Five researchers, deliberately given different evidence

Five agents propose candidate intent categories. They are **not** given the same
brief, and this is the whole point:

> *"Context isolation is not politeness here — a researcher handed the full
> evidence bundle writes the same generic taxonomy every other researcher would,
> and the fan-out stops buying anything."* — `_evidence_for_angle`

| angle | what it is asked to find | evidence it receives | web? |
|---|---|---|---|
| **log_reading** | categories grounded *only* in rows it actually read | 400 raw queries, stratified | no |
| **literature** | which published taxonomies transfer here — **and which do not** | corpus profile + phrasing families | yes |
| **legacy_audit** | is each legacy class real structure, or "unsorted traffic wearing a label"? | the existing taxonomy + samples | no |
| **pragmatic_intents** | intents **invisible in the wording** | 300 longer queries + known phrasing families | no |
| **risk_compliance** | safety/legal/compliance categories, *regardless of volume* | risk pre-screen + 200 random rows | yes |

Two of these deserve emphasis.

**`pragmatic_intents` is the highest-leverage angle**, and its brief says so:
> *"two queries can be phrased near-identically and want opposite things
> (verification vs definition; solve-this vs explain-this). Unsupervised
> clustering will never recover these, so if you do not find them, nobody will."*

This is the entire justification for having a top-down path at all. Clustering
groups by *surface form*. `字的拼音` and `字怎么读` cluster together and should.
But *"is 的 or 得 correct here"* and *"what does 得 mean"* look almost identical
and want opposite answers. Only a reader looking for that finds it.

**`log_reading` gets no web access on purpose:**
> *"its whole value is that one agent forms its view from the raw rows and
> nothing else, and giving it a search box invites it to replace observation
> with recall."*

#### What `pragmatic_intents` actually returned on live38

```
angle: pragmatic_intents_invisible_in_wording

candidate  code: verify_correct_form   name: 验证正确写法
  definition: 用户给出两个形近写法（如红装/红妆、源远/渊远），表面上问"区别"，
              实际上想知道哪个是正确写法、应该用哪个。

candidate  code: compute_or_convert    name: 计算或换算数值
  definition: 用户输入一个数学式或换算需求，要系统直接给出计算结果，
              而非解释概念或推导过程。
```

and, more valuable than the candidates, its **observations**:

```
"区别"这一表层形式至少映射两种截然不同的意图：写法验证（红装/红妆哪个对）
与概念辨析（断章取义/以偏概全怎么不同）。自动聚类会将二者合并，丢失关键区分。

裸诗句在长查询中占比可观，且没有任何问句标记 —— 同一个"碧玉妆成一树高…"
可能要续句、要全文、要释义、要出处，表面完全无法区分。
```

Read that second one carefully, because it is the justification for this entire
path in one sentence: **a bare line of poetry carries no marker at all**, and the
same string may want the next line, the full poem, a translation, or the source.
Clustering sees one shape. Four intents hide inside it.

The first observation is the same point in miniature: `区别` ("difference")
surfaces two unrelated intents, and — in the researcher's own words —
*"自动聚类会将二者合并，丢失关键区分"* (automatic clustering will merge them and
lose the distinction). That is a claim about what the bottom-up path **cannot**
recover, made by the agent whose job is to find exactly that.

### 3.2 The architect — synthesis, not a sixth opinion

One agent reads all five submissions and produces **one** class list: codes,
names, definitions, `user_need`, positive and negative examples. It is also told
which intents the researchers flagged as invisible to clustering, because
**nothing downstream will find them if the taxonomy omits them**.

It also records `dropped_candidates` — the researcher proposals it rejected.
Rejections are evidence, not waste.

#### A real class it produced

```
code:       LOOKUP_CHAR_PRONUNCIATION
name:       查字词读音
definition: 用户询问某字或词的规范读音、拼音、声调或粤语/姓氏等特殊读音。
user_need:  获得目标字/词的规范拼音（含声调）后，用户即可确认读法，不再猜测。
yes:        ['瑭怎么读', '憎恶的读音', '狴犴读音']
no:         ['温庭筠读音yun还是jun → JUDGE_LANGUAGE_USAGE',
             '旻字的意思是什么 → LOOKUP_WORD_MEANING']
```

Note the shape of `no`: every negative example **names the class it actually
belongs to**. That is what makes the boundary checkable rather than decorative —
`温庭筠读音yun还是jun` contains the word 读音 and is still *not* a pronunciation
lookup, because the user is asking which of two readings is correct.

Note also that `user_need` is written so it can be **falsified**: "obtains the
standard pinyin, and stops guessing" is checkable against a real answer. The
critic rejects `user_need`s that cannot be ("the user understands the topic").

### 3.3 The critic — adversarial review before anyone is paid to annotate

A separate agent is told to **break** the taxonomy:

> *"a review that returns no findings is a review that did not happen — every
> taxonomy at this stage has real defects."*

It checks six specific failure modes, in order: **overlap** (two classes a real
query could honestly belong to), **gaps** (a sample query with no home),
**catch-all pressure** (which class will silently absorb everything hard — flagged
if >5%), **form-defined classes** (defined by query *shape* rather than intent —
a defect "regardless of how well it predicts"), **untestable definitions** (a
`user_need` you cannot check against a real answer), and **missing risk
categories**.

#### What it actually found on live38 — 14 findings, verdict `revise`

```
kind: overlap   classes: [SCHOOL_EXAM_POLICY_LOOKUP, UNDERSPECIFIED_OR_NOISE]
evidence_query: 宏志中学
defect: SCHOOL_EXAM_POLICY_LOOKUP 的正例包含裸校名，而 UNDERSPECIFIED_OR_NOISE
        的定义是裸短词/无动作标记，同一裸校名可同时落入两处；
        RULE_ACTION_CUE_BEATS_BARE 没有覆盖学校名。

kind: overlap   classes: [LOOKUP_WORD_MEANING, JUDGE_LANGUAGE_USAGE]
evidence_query: 小公举是男孩还是女孩的意思
defect: 查询表面问词义，但含"是A还是B"的二元判断，JUDGE_LANGUAGE_USAGE 又收
        易混词判断；没有规则说明"词义+二选一"应归哪边。
```

This is the critic doing precisely its job. Each finding names **two classes**, a
**real query** that could honestly go to either, and **which rule fails to cover
it**. `宏志中学` is a bare school name: the policy class lists bare school names
as positives, and the catch-all is defined as bare phrases with no action marker.
Both definitions are individually reasonable and jointly ambiguous — and the
existing rule about action cues does not mention school names.

Every one of these is a disagreement the annotators would otherwise have
discovered later, at 3,000-row prices.

### 3.4 The rule writer — a separate call, for two measured reasons

Adjudication rules are tie-breaks: *"when the query looks like both A and B, pick
A."* They are written by a **separate call** from the architect, and the reason is
recorded in the code:

> *"Asking one call for the classes **and** the rules exceeded a 42,000-token
> output ceiling — most of it reasoning — and when the rule requirement was
> hardened to stop it writing only one rule, it satisfied that by returning two
> classes instead of nineteen."*

That is a general lesson worth stating: **prompt emphasis is zero-sum.**
Hardening one requirement in a prompt degrades the competing one. Splitting the
call removed both failures *structurally* rather than by instruction.

The rule writer is also *shown the final class list*, so it cannot invent a
tie-break between classes that do not exist — which had made two dozen otherwise
good rules unusable.

#### Real rules it wrote

```
[r001_pronunciation_marker]  classes=[LOOKUP_CHAR_PRONUNCIATION, LOOKUP_WORD_MEANING]
  when: 查询给出一个汉字或词，并出现读音类标记（怎么读、读音、拼音、声调、念什么、
        粤语、姓氏读法）；该字词同时也可能被理解为查释义。
  then: LOOKUP_CHAR_PRONUNCIATION

[r002_meaning_marker]  classes=[LOOKUP_WORD_MEANING, LOOKUP_CHAR_PRONUNCIATION, LOOKUP_CHAR_FORM]
  when: 查询只给普通字词、成语、俗语或典故，并带"什么意思、释义、含义"等标记，
        且没有古诗文原句或"怎么读、怎么写、组词"等其它动作。
  then: LOOKUP_WORD_MEANING
```

The structure is the point. A rule names the **pair it disambiguates**
(`classes`), the **observable trigger** (`when` — note it is written in terms of
markers *present in the query text*, not in terms of what the user "wants"), and
the **winner** (`then`). `r001` explicitly acknowledges the ambiguity it is
resolving: *"该字词同时也可能被理解为查释义"* — this word could also be read as a
meaning lookup — and then rules for pronunciation anyway, because the marker is
observable and the intent is not.

> ⚠️ **Known gap, visible in this very data.** The rule writer is told *"Every
> rule's `then` MUST be one of the codes above, exactly as written."* Measured on
> live38: **18 of 50 rules put a whole sentence there instead** —
> `then='归 JUDGE_LANGUAGE_USAGE，不归 LOOKUP_CHAR_PRONUNCIATION。'` It cost
> something real: `_dedupe_rules` compares `then` as a string, so on gen05 it
> withheld **two valid rules** as "contradictory" because one said
> `LOOKUP_WORD_MEANING` and the other said `LOOKUP_WORD词语释义` — a hallucinated
> variant that is not a class code at all. **An instruction is not a validation.**
> See `HANDOFF.md` §2 item 5f.

### 3.5 The pilot — the step that makes the rest trustworthy

This is the most important idea in phase 2, and the least obvious.

Before paying for 3,000 gold labels, two annotators label **200** queries and we
measure **Cohen's κ** — agreement corrected for chance. Low κ means the *guide*
is ambiguous, not that the annotators are careless.

But a raw κ cannot tell you *which* problem you have, because **an ambiguous
guide and an unreliable annotator produce the same number**, and they have
opposite remedies. So the pilot also measures a **ceiling**:

> the same annotator, the same queries, **re-asked in a different batch order**

That is the annotator's **self-consistency** — and it is the ceiling any two
independent annotators could possibly reach, because two annotators cannot agree
with each other more reliably than one agrees with itself.

Comparing the two numbers separates the cases:

- **κ well below the ceiling** → the guide has recoverable slack. Redraft.
- **κ at the ceiling** → this is as good as this annotator gets on this corpus.
  No amount of guide work will help; you need a stronger model or humans.

#### The real gate line from live38

```
gate p2a_pilot_agreement: PASSED — pilot: kappa 0.824 (95% upper 0.871) on 200
queries; raw agreement 84.0%, but kappa 0.9 needs about 90.9%; annotator
self-consistency kappa 0.9228 (0.8925 of ceiling reached)
```

Everything needed to judge the taxonomy is in that one line:

- **κ 0.824** — agreement between two independent annotators
- **raw agreement 84.0%** — before correcting for chance
- **"kappa 0.9 needs about 90.9%"** — what raw agreement the target would demand
- **ceiling 0.9228** — one annotator against *itself*
- **0.8925 of ceiling reached** — the annotators are at 89% of the best they
  could possibly do

Without the ceiling, κ 0.824 is uninterpretable: you cannot tell a fixable guide
from an annotator at its limit. With it, you can say precisely — there is about
0.10 of recoverable slack, and the 0.90 target sits at **97.5% of the ceiling**,
which is a very demanding bar to set for two independent readers.

**Why this matters historically:** the playbook's 0.90 target was set on a
project whose annotators self-agreed at 0.966. Ours self-agree at ~0.92.
Requiring 0.90 *between* two of them demanded more reliability than one of them
has — *"a perfect guide would still have halted, every time, forever."*

### 3.6 The redraw loop — and an honest verdict on it

The pilot also splits disagreements into two kinds:

- **structural** — the *same* annotator resolved the query differently on the
  second pass. The boundary is **not in the data**; no tie-break rule can rescue
  it. Merge or re-cut.
- **guide** — the two annotators differ but each was self-consistent. The
  boundary exists and is merely *unstated*. That is what a rule is for.

Structural pairs go to a **redraw** agent that rewrites only those boundaries. If
κ drops, the redraw is reverted.

#### The real top confusions from live38

```
top confusions ['CLASSICAL_CHINESE_LOOKUP × LOOKUP_WORD_MEANING (5)',
                'LOOKUP_WORD_MEANING × UNDERSPECIFIED_OR_NOISE (3)',
                'ADULT_OR_ABUSE_RISK × LOOKUP_WORD_MEANING (2)']
```

Look at the counts: **5, 3, 2**. That is the whole basis on which six boundaries
would be sent for redrawing — and it is why the redraw does not work. At n=200
with 36 disagreements spread over 25 pairs, the expected count per pair is ~1.4,
so a pair seen 2 or 3 times is barely distinguishable from chance.

> **Honest verdict: the redraw has never been shown to work.** Four controlled
> trials: −0.017, +0.025, −0.011, −0.072. se(κ) at n=200 is ~0.03, so the first
> three are inside one standard error with disagreeing signs. Worse, the
> disagreement is *flat* — 36 disagreements spread over 25 distinct pairs, 19 seen
> exactly once — so "the worst 6 boundaries" does not survive re-measurement
> (3 of 6 overlap between consecutive pilots). Treat the redraw as a **diagnostic
> that reports contested regions**, not a remedy. See
> `memory/qmine-redraw-targets-are-noise.md`.

---

## 4. Phase 2b — the gold set, and the three things that repair it

### 4.1 Double-blind annotation

3,000 queries (6% of the corpus), each labelled independently by **two**
annotators routed to **different labs** — deepseek and qwen, with the referee on
zhipu. Independence is by *lab*, not by gateway: two models from the same lab
share an architecture, and their agreement would measure that rather than the
guide.

live38: raw agreement 0.839, **κ 0.822**, 483 disagreements.

#### What an annotator actually returns

```
query: 神经酰胺怎么读
  label=LOOKUP_CHAR_PRONUNCIATION  confidence=high  rule_cited=r001_pronunciation_marker
  rationale: 含明确"怎么读"标记，目标词读音需求。

query: 刚愎自用的三大生肖
  label=ZODIAC_GAMBLING_RISK  confidence=high  rule_cited=r019_zodiac_gambling_vs_puzzle
  rationale: 含"生肖"且要求推测答案，属博彩风险类。
```

Every label carries the **rule it cited**. That is what makes the guide auditable:
when two annotators disagree you can see *which rule each of them applied*, and
whether the rule or the reader was at fault. The second example is also the
`risk_compliance` researcher's work paying off — `刚愎自用的三大生肖` looks like an
idiom question and is actually a lottery probe. No clustering finds that.

### 4.2 The referee — settling boundaries, not rows

The referee sees only the disagreements. Crucially it is settling **boundaries**,
not individual rows: if the same class pair appears in two separate batches and
each decides independently, the rule set acquires **two rules that fire on the
same trigger and give opposite answers**.

Batches are therefore grouped so that **no class pair spans two batches** — which
makes the batches independent, so they run concurrently. A pair too large for one
call is split into *sequential* chunks that stay in one group and thread their
earlier ruling forward.

live38: 483 disagreements over 83 class pairs → 28 groups, 39 calls, **~37 min**
(the earlier row-position batching projected 5–6 h). Coverage **465/483 = 96%**.

#### Real referee verdicts

Settling a contested row against an existing rule:

```
query: 芬顿反应
  final_label: UNDERSPECIFIED_OR_NOISE   rule_gap: False   both_defensible: False
  rationale: 查询仅给裸学科术语"芬顿反应"，无"是什么/定义/公式/原理"等动作标记，
             无法稳定判定要定义还是机理。按无动作标记裸词归入意图不明兜底。
```

And **finding a gap and drafting a rule to close it** — the mechanism by which
the guide improves:

```
query: 乙烯与氯化氢反应
  final_label: SUBJECT_CONCEPT_LOOKUP   rule_gap: True   both_defensible: True
  rationale: 查询为裸学科主题短语（化学反应描述），无动作标记。现有规则无法区分
             此类裸学科短语应归概念查询还是解题答案请求。
  proposed_rule: 当查询为裸学科主题短语（如"X与Y反应"），无动作标记（是什么、称为、
             公式、求等），归 SUBJECT_CONCEPT_LOOKUP，因为裸学科短语无考试外壳
             标记时默认为知识查询而非解题。
```

Note `both_defensible: True` — the referee is saying *both annotators were
reasonable*, which is precisely the signal that the **guide** was at fault rather
than either reader. It then writes the rule that would have prevented the
disagreement. That is the loop by which 82 new rules were drafted on live38.

The referee also **drafts new rules** as it goes — each written in response to a
disagreement actually observed.

### 4.3 Guide repair — deciding the boundaries the referee could not

Some boundaries the referee itself resolved *both ways*. No rule can be written
for a boundary nobody has decided. So the repair:

1. finds those still-open pairs (`contested_boundaries`)
2. looks for **lexical markers** that separate them
3. for rows with *no marker at all* — 64% of disagreements on the motivating
   corpus — takes a default from **rows both annotators already agreed on**:
   evidence with no arbitration in it
4. requires that majority to be **decisive** (support ≥20, precision ≥0.75), and
   returns `None` otherwise — *"a 64% lean is noise dressed as a rule and would
   manufacture as much disagreement as it settles"*
5. appends the rulings to the guide as a binding `## 边界裁定` section
6. re-annotates a **fresh** sample under the repaired guide

Step 6 uses a fresh sample deliberately: re-scoring the rows the rules were
derived from measures how well those rules fit *those rows*, which is not the
question.

#### The real repair log from gen05

```
referee adjudicated 465/483, drafted 82 rules, 18 left unresolved (0 batch failures)
38 duplicate rule(s) dropped before reaching the guide
⚠ rule R112 and R053 fire on the same trigger and disagree
   (LOOKUP_WORD词语释义 vs LOOKUP_WORD_MEANING) — both withheld, boundary left open
⚠ 1 contradictory rule pair(s) withheld
82/122 new rules reach the guide
guide repair 1/1: 26 open boundaries, 26 decided from agreed rows, 112 rules added
112/112 new rules reach the guide
re-annotating 3000 fresh queries under the repaired guide
```

Read top to bottom, that is the whole repair mechanism working:

- the referee settled **465 of 483** and could not settle 18
- **38 duplicates** were dropped — the referee writes one rule per disagreement,
  and many disagreements share a boundary
- **one contradictory pair was withheld rather than resolved** — when two rules
  fire on the same trigger and disagree, picking one arbitrarily would encode a
  coin flip, so both are withheld and the boundary is left *honestly open*
- the repair then found **26 boundaries still open** and decided every one of
  them from rows both annotators had already agreed on
- **112/112 rules reach the guide** — which only became true today; before the
  truncation fix that number was effectively 0

> ⚠️ **Two honest caveats.**
> **(a) There is no revert guard.** Unlike the redraw, nothing reverts a repair
> that makes things worse.
> **(b) The comparison cannot detect degradation anyway** — round 1 is scored on
> sample A and round 2 on sample B, so the delta mixes "guide improved" with
> "sample differed". The code says so in its own log line. The missing piece is a
> **control arm**: annotate sample B with the *old* guide too.
>
> And a third, discovered 2026-08-24: **the repair's own rules were being
> truncated out of the prompts meant to apply them.** The rule block reached
> 46,814 chars against a 9,000 head-only budget, so every referee rule was cut.
> Both previously measured "flat" results (−0.002 and +0.010) were obtained that
> way. Fixed; the next measurement is the first honest one.

### 4.4 Active learning — a second batch aimed at the boundary

A cheap character TF-IDF model is fitted on round 1 and used to rank rows by how
close they sit to a decision boundary (lowest top1−top2 margin). Those rows get
annotated as a second batch — spending annotation where the model is least
certain rather than uniformly.

---

## 5. Phases 2c–2e — turning the gold set into something usable

> **These three have never run with live agents.** Every prior run halted at or
> before the p2b gate, so unlike everything above there are **no real outputs to
> show** — the descriptions below are from the code, not from observation. That
> gap is the single largest untested surface in the pipeline, and it is why the
> audit of these phases (`HANDOFF.md` §8) hunted them specifically.

**2c — the classifier.** Rules first, then a linear head over dense ⊕ sparse ⊕
rule-flag features. This is what labels the ~47,000 rows nobody annotated.
Calibration (ECE) is reported *out-of-fold*, because phase 10 **routes on
confidence** — a model that says 0.9 and is right 0.6 of the time makes the
routing threshold meaningless.

**2d — adversarial validation.** An agent is asked to **prove each label wrong**.
The rate at which it *fails* is an honest accuracy estimate, rather than the
rubber stamp you get from asking "is this right?". The coverage is reported
beside it: an accuracy computed over the verdicts that came *back* is not an
accuracy over the rows attacked.

**2e — sub-intents (L2).** For each L1 class, ask whether the embedding can even
*see* it, by comparing kNN agreement against **chance** (a class's own share)
rather than a flat threshold. A class the geometry cannot separate is not
deleted — it is marked **rule-dependent**, meaning it gets its accuracy from the
rule layer instead. *"The taxonomy answers to users rather than to geometry."*

---

## 6. Why the whole thing is shaped like this

Three ideas explain almost every design decision above.

**1. Make failure visible while it is cheap.** The pilot costs 200 queries and
four LLM calls, and it can halt the run before 3,000 gold labels are bought. The
critic runs before the pilot. The gates are ordered by cost.

**2. Separate "the instructions are bad" from "the annotator is bad".** They
produce identical κ and have opposite remedies. The self-consistency ceiling is
the only thing that distinguishes them, and it costs one extra annotation pass.

**3. Never let a number hide what it left out.** This is the project's
characteristic bug, found repeatedly: a referee dropping its hardest rows, an
accuracy shrinking its denominator, a calibration figure changing basis without
changing label, a prompt block discarding the guidance it exists to carry. The
rule *"read `n` before believing any metric"* generalises to **make every
mechanism say what it dropped.**

---

## 7. Where to look in the code

| thing | file |
|---|---|
| research angles and briefs | `agents/roles.py` → `RESEARCH_ANGLES` |
| what evidence each angle gets | `graph/nodes/topdown.py` → `_evidence_for_angle` |
| architect / critic / rule writer prompts | `agents/prompts/{architect,critic,rule_writer}.md` |
| the pilot and its ceiling | `graph/nodes/topdown.py` → `_pilot_agreement` |
| redraw loop | `_redraw_until_stable` |
| gold set + referee | `p2b_gold`, `_batch_by_class_pair` |
| guide repair | `_repair_guide_and_reannotate`, `ops/classify.py` → `boundary_default` |
| classifier / adversary / L2 | `p2c_classifier`, `p2d_validate`, `p2e_subintents` |

Open questions and known gaps live in `HANDOFF.md` §2. Durable lessons live in
Claude's memory directory, indexed by `MEMORY.md`.

---

## 8. A reading of the example trail, end to end

Follow one idea through every step above and the design explains itself.

The **`pragmatic_intents` researcher** observes that `区别` maps to two unrelated
intents and that *"自动聚类会将二者合并"*. The **architect** turns that into
`JUDGE_LANGUAGE_USAGE`, distinct from `LOOKUP_WORD_MEANING`, and writes a negative
example that names the boundary: `温庭筠读音yun还是jun → JUDGE_LANGUAGE_USAGE`.
The **critic** immediately attacks it — `小公举是男孩还是女孩的意思` is *"词义+二选一"*
and no rule says which side wins. The **rule writer** produces `r002_meaning_marker`
to separate meaning-lookups from their neighbours. The **pilot** measures whether
any of it can actually be applied: κ 0.824 against a 0.9228 ceiling. The
**annotators** cite those rules by id on every row. The **referee** finds the rows
where the rules ran out — `乙烯与氯化氢反应`, `both_defensible: True` — and writes the
rule that was missing. The **repair** takes the 26 boundaries still open and settles
each from rows both annotators already agreed on.

At no point does anyone assert that the taxonomy is good. Every step produces a
number or a counter-example that could show it is not.
