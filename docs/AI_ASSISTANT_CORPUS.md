# Mining a multi-vertical AI-assistant log

> 中文版：[`AI_ASSISTANT_CORPUS.zh.md`](AI_ASSISTANT_CORPUS.zh.md)

How the two `ai助手` exports were turned into something the pipeline can mine, and
the general procedure to repeat when the next export arrives. Every number here
was measured on the pooled 65,986-row corpus; nothing is inherited from the five
single-vertical corpora this project was built on.

The short version: **three properties of this corpus decide everything else, and
none of them is visible from the column names.**

---

## 0. What the two files are

| | `ai助手_Top1000query.xlsx` | `ai助手_随机1000query.xlsx` |
|---|---|---|
| rows | 33,000 | 33,000 |
| design | top 1,000 by `search_num` per category | random 1,000 per category |
| categories | 33 first-level, 213 second-level | same |
| traffic | 30,428,147 | 42,414 |
| median query length | 3–12 chars by category | 10–23 chars |
| shared query strings | **159**, a Jaccard of 0.2% | |

They are **two sampling strata of one period**, not two periods. That single fact
drives §2.

---

## 1. Profile before you configure anything

Four measurements, in this order. Each one changed a configuration decision.

**1.1 Is the head made of queries?** Rank by traffic and read the top 25. Here:
`总结全文概要` (5.80M), `变清晰` (5.23M), `去水印` (1.78M), `👌 好的，继续吧` (2.13M
across 32 categories), `嗯`, `需要`, `好`. These are tool-panel buttons, suggested-
reply chips and bare conversational turns — not typed intents.

**1.2 How wide is each string?** Count the distinct first-level categories each
string appears under. A string in 32 of 33 topical categories is not topical.

> **Seven strings carry 15.97% of pooled head traffic and appear in 29–32 of the
> 33 categories.** The full acknowledgement family is 1,065 rows, and under a
> weight that treats categories equally it is **28.9% of head traffic** — median
> **30.1%** per category, **71.9%** in 生活和情感.

**1.3 Is traffic comparable across categories?** Take each category's minimum
`search_num` in the head file — that is its top-N floor. Here they run **3
(招商加盟) to 419 (书籍文档)**, so the file is a union of 33 censuses cut at 33
different depths and raw traffic cannot be compared across them. Under raw PV two
categories hold 55% of pooled weight and one string holds 17%.

**1.4 Can you reweight to the population?** Test it, do not assume it. The
capture–recapture bridge — what fraction of the random sample sits at or above the
head floor — **fails here**: 17 of 33 categories have *zero* random rows above
their floor and 28 have fewer than five, so the estimator returns 1,000,000
distinct queries off a single row.

> **Cross-category traffic comparison is not recoverable from these two files.**
> Say so; do not estimate it. This is the same discipline as the abandoned
> power-law extrapolation in `DRIFT_ANALYSIS.md`.

---

## 2. Decide the comparison axis before you pool

Both wrong answers are tempting.

**Running the two files separately** loses the comparison entirely. Measured on
this project's own corpora: `fin01` and `fin02`, the same vertical run twice,
produced 20 and 19 classes sharing **zero** codes — `LOOKUP_FX_RATE` versus
`FX_RATE_LOOKUP`, the same thing, not joinable. Two runs give two vocabularies.

**Passing them as `--input head,tail`** pools them correctly and then *describes*
them wrongly. That path tags rows `_snapshot` and ships 「快照对比 · 漂移分析」,
whose interpretive sentences are 「不是趋势」, 「同月同日不等于季节可比」,
「时段性事件」 — every one false about a head/tail split. One of them **inverts**:

> 「两期的抽样方式必须一致」 warns that differing sampling would masquerade as a
> real change. Between a top-N head and a random tail the differing sampling **is
> the independent variable**. A reader applying that caveat concludes the document
> is confounded when it is measuring exactly what it set out to.

**The third way, which is what we do:** pool into one file with a `stratum`
column and set `data.comparison_axis: stratum`. `ops/drift.py` never knew about
time — it compares two groups' composition — so only the vocabulary changes. The
run ships `分层对比_头尾结构差异.md`. The `time` branch is unchanged, pinned
byte-for-byte over seven payload shapes.

---

## 3. Build the corpus

```bash
python tools/prepare_assistant_corpus.py \
    --head ~/Downloads/ai助手_Top1000query.xlsx \
    --tail ~/Downloads/ai助手_随机1000query.xlsx \
    -o data/raw/ai_assistant_pooled.parquet --show-ack
```

Four decisions, each recorded in that tool's docstring:

| decision | why |
|---|---|
| pool, don't run twice | §2 — one taxonomy must label both |
| collapse duplicates **only within (stratum, l1, l2)** | the category is data; collapsing across it would move `👌 好的，继续吧`'s 2.13M PV into whichever category was modal. 13 exact duplicate rows collapsed, cross-category occurrences kept with an `n_l1_categories` column |
| weight on `pv_norm`, not `search_num` | §1.3. Normalised within (stratum, category) to 1,000, so a traffic share reads as "share of a typical category's stratum traffic". `search_num` is still in the file |
| tag the strata `top1k` / `random1k`, not `head` / `tail` | `tail` is a **conclusion** about where the rows sit in the traffic distribution; the file is a **random sample**, which is a statement about how it was drawn. Only 0.35% of the random rows actually fall inside the top-1000 band — so they are nearly disjoint, but that is measured, not guaranteed, and a reader told `tail` cannot tell which claim they were given |
| **flag** acknowledgements, never drop them | that they are ~29% of head traffic is the largest finding in the corpus, and a dropped row cannot be reported. `is_ack` uses one stated, auditable pattern, anchored at both ends so `好的` matches and `好的，帮我写一份年终总结` does not |

---

## 4. Seed the domain profile from measurement

`configs/domains/ai_assistant_zh.yaml`. The `TemplateSeed` contract is *everything
matching this regex is almost certainly the same intent* — a claim about data, so
check it against data:

```bash
python tools/check_domain_profile.py configs/domains/ai_assistant_zh.yaml \
    data/raw/ai_assistant_pooled.parquet --text-column query --weight pv_norm
```

**Overlap is the number that matters**, not coverage. All 12 seeds overlap ≤3.8%;
eight overlap at 0. Two candidates were cut for failing this — an early
`roleplay` seed overlapped 18.9% with `write_compose` and was narrowed to
`roleplay_persona`.

Three seeds match 10–23 rows and are kept deliberately: `image_edit_op`'s 19 rows
carry **1.69%** of normalised traffic and `doc_summarize`'s 10 carry **1.36%**.
They are canonical single-string commands — tight, real, and tiny in rows. Do not
loosen them to make the coverage number look better.

Total seed coverage is 5.1% of rows. That is fine here **because this corpus has a
real reference taxonomy**: `l1`/`l2` vary row-to-row and reach 100%, so
`k_locator: auto` locates K against them rather than against the seeds. Every
previous corpus carried a `query_1st_category` holding one constant value — a
filename, which p1 drops.

---

## 5. Screen for risk, then audit the false positives

An assistant **generates**, so a mis-served row produces content rather than a bad
result page. Eight categories are declared. Two things the audit found:

**Long fiction prompts over-trigger every keyword pattern.** Fiction markers are
2.0% of corpus rows and **15% of risk hits** — a 7.5× enrichment, because a
500-character roleplay prompt contains many words and only has to contain one.
`self_harm` is the extreme: 38% of its hits are fiction-marked and its hits have a
**median length of 565 characters** against the corpus's 10. Query length is the
cheapest triage filter for any of these lists.

Three words were removed from `financial_advice` for measured reasons: `抄底` fired
on a Rothschild character description; `带单` matched exactly one row in 65,986 and
it was 宽**带单**独计费 — "broadband billed separately", the same substring trap as
`finance_zh`'s 澳门 note. What survives requires an advice-seeking *frame*.

**`self_harm` is kept broad anyway, and this is the one place the trade goes the
other way.** All 16 hits are third-person — fiction, history (杜聿明), news, a
cartoon, an abstract law question. Zero are first-person. On this corpus it is a
pure false-positive generator, and it stays, because a false negative here is a
person in crisis getting a plot summary.

**One row worth a human decision.** `minor_sexualisation` fires once: a
self-identified 15-year-old asking about her own body measurement — and the
export's own taxonomy files it under 生活和情感/两性知识, a general sex-education
bucket with no minor-specific handling. A category that fires once is still
correct to declare; n=1 is not evidence the next export has none.

---

## 6. Run it

```bash
python tools/prepare_assistant_corpus.py --head … --tail … -o data/raw/ai_assistant_pooled.parquet
.venv/bin/qmine models --config configs/live_ai_assistant.yaml --domain ai_assistant_zh
.venv/bin/qmine run --config configs/live_ai_assistant.yaml --domain ai_assistant_zh \
    --input data/raw/ai_assistant_pooled.parquet --run-id ai01 --smoke --offline   # wiring
.venv/bin/qmine run --config configs/live_ai_assistant.yaml --domain ai_assistant_zh \
    --input data/raw/ai_assistant_pooled.parquet --run-id ai01                     # real
```

`--input` is required on the command line even though the config names it — a
fresh run refuses to start without it.

Afterwards, extract what the pooling actually bought:

```bash
python tools/vertical_crosstab.py runs/ai01/gen01 --weight-column weight
```

---

## 7. What this corpus cannot support, whatever the run says

- **Row counts across categories carry no information.** The export allocates
  exactly 1,000 rows per category per stratum. A class that is 80% one category
  partly reflects that allocation.
- **No population estimate** (§1.4). Shares here are within-stratum,
  within-category; they do not convert to "how much of the whole log".
- **A large share of rows are conversation turns, not queries.** 「他与皮肤不粘连，
  是独立个体」(医疗), 「目前没发现低于6.8的」(商品), 「前面讲的还记得吗」 are
  uninterpretable without a context the export does not carry. They are real
  traffic and they are analysed; no single-row method recovers their intent, and
  the deliverables should name them for what they are rather than guess.
- **The reference labels are noisiest exactly where the traffic is.**
  `👌 好的，继续吧` is filed under 32 of the 33 categories, and 1.54% of
  (category, query) pairs carry more than one second-level label.
