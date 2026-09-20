# 健康 corpus — how the two exports were prepared, and how the AI export was cleaned

Written 2026-09-14 for `analysis/pooled5/build_health_corpus.py`. Every number below was measured
on the files; nothing here is an estimate. Privacy: no sexual, minor-related or personal-name string
is quoted in this note — those families are described by count only.

## 1. The two exports

| | 健康搜索_top1w.xlsx | 健康ai管家_top1w.xlsx |
|---|---|---|
| what it is | health search, one week | a specialised health AI assistant, the same week |
| window | 2026-09-04 .. 09-10 (7 days) | 2026-09-04 .. 09-10 (7 days) |
| raw rows | 10,000 = top (query x day) | 10,000 = top (query x day) |
| distinct strings | **2,260** | **1,887** |
| strings on all 7 days | **893** | 1,075 |
| strings on one day | 501 | 272 |
| daily export cutoff (lowest PV) | 2,296 | 195 |
| week PV | 73,297,116 | 14,681,275 |
| cell types | 10,000 text cells | 10,000 text cells |

Correction (2026-09-15): an earlier version of this table gave 890 search strings on all 7 days. That
number was the count of strings with exactly 7 raw rows. Recounted on the raw export (`original_query` x
`event_day`), 893 strings are present on all 7 days: 5 of them have 8 raw rows because the export split them
across two labels on one day, and 2 strings with 7 raw rows are present on fewer days. The builder's
aggregation and `product_layer_aggregation.csv` already used 893.

The finance exports had stored all-digit query cells as numbers and lost leading zeros; both of
these files are text throughout, which the builder asserts.

**The AI export had the same day-duplication as search**, although only the search file was
mentioned: its strings recur across days just the same (1,075 of them on all 7 days), so both are
aggregated.

## 2. Aggregation

- One row per (source, query). PV is summed over every raw row of the string.
- **Nine search strings have two rows on the same day** with different category labels and
  different PV. They are not copies (0 exact duplicate rows) — the export split one query's traffic
  across two labels — so their PV is summed, not de-duplicated.
- A string absent on a day was below that day's cutoff, not at zero. `pv_raw` (the week total) is
  **exact** when `n_days == 7` and a **lower bound** otherwise; `pv_week_upper` adds, for each missing
  day, that day's own cutoff minus one.
- Category labels: 19 search strings carry conflicting labels (3 at level 2, 16 at level 3). The
  PV-dominant label is kept and the full split is stored (`legacy_l2_split`, `legacy_l3_split`).
  The AI export's labels are consistent per string.
- `legacy_l3` is 科室-类型 crossed (内科-症状). It is split into `legacy_dept` and `legacy_type`.
- The builder asserts that rows and PV are conserved exactly.

## 3. Search cleaning (rules only)

| tier | strings | week PV | rule |
|---|---|---|---|
| S6 doctor_card_uniform | 16 | 3.71% | "科室 + name + 医生" **and** on all 7 days **and** day-to-day CV <= 0.03 |
| S5 headline | 8 | 0.14% | news-headline syntax on a trafficked row (the rule used for every pooled corpus) |
| user | 2,236 | 96.16% | |

Why S6: sixteen different named doctors received 164,350–179,330 PV each over the week — a
cross-doctor coefficient of variation of 0.026, and 0.023–0.032 on every single day. Organic search
strings in the same weekly PV band (150–190k) vary 0.422 day to day, and 0.077 across strings.
Sixteen individuals do not attract near-identical interest. Low day-to-day CV alone was **not**
used: 69 full-week strings sit below 0.03, many of them plainly organic evergreen lookups, because
this week's traffic is flat overall (organic day index 0.945–1.026).

Not removed from search, deliberately: the private-hospital lead-generation strings (60 strings,
6.57% of search PV). They are user searches, often answering an advertisement; they are screened as
a risk category in `configs/domains/health_zh.yaml` instead.

## 4. AI cleaning

### 4.1 The blind audit

- Every one of the 1,887 distinct AI strings was labelled by **three independent readings** with
  different lenses (text first; as the product's designer; evidence first), under a criteria-first
  rubric that never saw the builder's rules. Eight classes: U typed user query, A intake answer,
  F feature or card-facet button, P pushed suggestion, W template wrapper, D doctor/institution card,
  C content-free, X unsure.
- Coverage: 12 passes (3 per chunk x 4 chunks), 5,661 labels, every string exactly 3 times, no
  unknown or duplicate ids.
- Agreement: **Fleiss kappa 0.945** over 8 classes, 0.849 over U / non-U / X; **95.6% unanimous**;
  every string has a 2-of-3 majority. One pass (chunk 0) labelled 3.2% of rows U against 9.5–13.0%
  for the other eleven; the majority absorbs it.
- Stored as `analysis/pooled5/work/health_ai_audit_labels.csv` (`majority`, `votes`).

| majority | strings | week PV of the AI export |
|---|---|---|
| A intake answer | 1,243 | 60.85% |
| F feature / facet | 69 | 23.55% |
| P pushed suggestion | 245 | 8.39% |
| C content-free | 11 | 2.37% |
| W template wrapper | 118 | 0.74% |
| D doctor card | 2 | 0.01% |
| **U typed user query** | **199** | **4.09%** |

### 4.2 Rules, measured against the audit

- The first draft of the rules was **precise but blind to most of the product layer**: precision
  1.000 (PV 1.000), recall 0.323 (PV 0.764). The existing general-assistant cleaner caught 13 strings
  (5.90% of PV) and missed 37 of the top 40.
- Extended to every family the audit exposed (status and degree answers, laterality, temperature and
  blood-pressure ranges, pregnancy stage, frequency, body location, quantity, lifestyle, goals;
  tool entries and card facets; the `我想了解` wrapper; health acknowledgements; doctor cards at
  保健院; pushed questions framed by caregiving, season or that week's events), the final rules alone
  reach **precision 0.993, recall 0.707 (PV precision 0.999, PV recall 0.939)**.
- One rule was withdrawn on the audit's evidence: a `X异常` status suffix caught 白带异常, a symptom
  name the audit kept as typed.
- The recall ceiling is structural: answer chips are an open vocabulary.

### 4.3 How the tier is decided

- **Where the three readings agree, their label decides** (`tier_source = audit_unanimous`, 1,804 strings).
- **Where they split 2-1, the rules decide** (`tier_source = rule`, 83 strings), so near-identical
  forms are treated alike. The rules disagree with the 2-1 majority on 34 of those 83 — almost all on
  the boundary described in 4.4.
- Every AI row keeps `rule_tier`, `audit_majority` and `audit_votes`.

| final tier | strings | week PV of the AI export |
|---|---|---|
| H1 intake answer | 1,219 | 60.35% |
| H2 feature / facet | 69 | 23.55% |
| H3 pushed suggestion | 243 | 8.19% |
| C1 content-free | 11 | 2.37% |
| H4 template wrapper | 118 | 0.74% |
| H5 doctor card | 2 | 0.01% |
| **user (mined)** | **225** | **4.78%** |

### 4.4 The one boundary the export cannot settle: bare terms

The readings split almost only on bare symptom names. The rubric said "a bare disease / drug / food
name is U" and was silent on symptoms, so this is a rubric gap, and the data was asked instead:

| group | strings | verbatim in this week's search | verbatim in 2025/26 medical search | verbatim in the general assistant | median days | median day CV |
|---|---|---|---|---|---|---|
| majority U, bare (<= 6 chars, no question word) | 160 | 14 | 49 | 16 | 7 | 0.06 |
| majority U, questions / longer | 39 | 8 | 1 | 1 | 2 | 0.24 |
| majority A, bare | 1,180 | 1 | 8 | 11 | 7 | 0.07 |

- 49 of the 160 bare terms the audit kept appear **verbatim** in a human-typed log, and all of the
  other 111 appear **inside longer typed searches** — people do search these symptoms.
- The chips the audit rejected almost never appear verbatim anywhere.
- Traffic shape does not separate the two (both median 7 days, day CV 0.06–0.07).

So a bare symptom / disease / drug / test name is **kept as user** and carries
`flag_bare_term_origin_unknown` (184 of the 225 kept AI rows): a picker option and a typed bare term
leave identical rows, and every figure can be recomputed without them. A symptom wrapped in the
product's option grammar (有头晕 / 轻微头晕 / 血糖偏高) is an answer chip.

## 5. Corpus and reference columns

- Mined: **2,461 rows** = search 2,236 + AI 225. `pv_norm` normalises week PV within source to 10,000.
- Reference columns, decided by a condition written before the final corpus existed (each column:
  Cramér's V with surface <= 0.55 and no class above 1% of rows on one surface only):

| column | V | one-sided classes | decision |
|---|---|---|---|
| legacy_l2 | 0.251 | 0 | declared (first) |
| legacy_type | 0.362 | 1 (< 1% of rows) | declared |
| legacy_dept | 0.348 | 5 (2 above 1% of rows) | **withdrawn** |

## 6. What this means for every downstream figure

- The AI side of the pooled comparison is **225 rows**. Its shares carry wide intervals, and most
  of it is bare terms of uncertain origin.
- The product layer that was removed is not lost: `work/健康_all_rows.parquet` keeps every string
  with its tier, so the assistant's feature usage, intake answers and pushed topics can be described
  directly — as product usage, not as user intent.
