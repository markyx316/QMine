# Unknown-Domain Bootstrapping

> Gathered 2026-08-18. Facts marked verified were fetched live; the model-landscape
> dossier in particular flags its prices as secondary-source and unconfirmed —
> which is why the running system fetches prices from a live catalogue rather
> than embedding any table from this document.

> **Verification note.** All model IDs and prices below were verified on **2026-08-18** against the in-repo `claude-api` skill (`/private/tmp/claude-501/bundled-skills/2.1.229/…/claude-api/SKILL.md`, model table cached 2026-06-24), cross-checked against `src/qmine/config.py:LLMConfig`. **`WebFetch` is blocked in this environment** for every domain I tried (platform.claude.com, arxiv.org, github.com, mlcommons.org, kdd.org, microsoft.com — all return "Unable to verify if domain is safe to fetch"), so the *literature* claims below are sourced from `WebSearch` result snippets, verified 2026-08-18, and I flag every number I could not read in the primary source. The QMine code claims are first-hand — I read the files.

---

# 1. What actually breaks today, in code

Before designing anything, here is the concrete failure surface for a no-profile run. `configs/domains/` contains exactly five profiles (`k12_zh`, `finance_zh`, `sports_zh`, `politics_zh`, `ecommerce_en`). There is **no `generic.yaml`** — the "generic" profile is the Pydantic default in `src/qmine/config.py:58-89`: `template_seeds=[]`, `risk_categories=[]`, `pragmatic_intents_hint=[]`, `language="zh"`, `tokenizer="jieba"`, `embedding_candidates=["BAAI/bge-small-zh-v1.5", …]`. Four silent failures follow:

**(a) The trust flag inverts meaning.** `src/qmine/ops/templates.py:build_groups` sets `trusted=not is_discovered`. With zero seeds, *every* group is `trusted=False`. `src/qmine/graph/nodes/foundation.py:113-116` then does:

```python
trusted = group_masks(groups, df, ..., trusted_only=True)
if not trusted:                     # no seeds survived: fall back, and say so
    trusted = masks
```

So the trusted set silently becomes **all mined affixes**. Those masks are `deps.cache_put("template_masks", trusted)` — the set that judges the alpha sweep in Phase 3c and computes `template_fragmentation` in Phase 9. The module's own docstring warns why this is wrong: *"a marker like '是什么' spans every intent in the corpus and legitimately lands in eight clusters. Scoring it as fragmentation measures the marker's looseness, not the representation's quality, and on real data it dominates the average."* On a generic run, the metric that the playbook uses *instead of silhouette* is being computed entirely from markers whose contract was never established. This is the single most dangerous consequence, because it is not a crash — it is a plausible-looking number.

**(b) The risk gate passes vacuously.** `screen_risk` (`src/qmine/ops/audit.py:205`) iterates `risk_categories`; empty list → `total_flagged=0`. The Phase 7 gate `require_risk_independently_found` (`src/qmine/config.py`, checked at `graph/nodes/naming.py:248-256`) tests `independently_found = bool(namer_flagged or sentinel_flagged)` — i.e. it only checks whether *some* agent raised a flag, never whether that flag corroborates the pre-screen. With no seeds, the gate is not comparing anything; it degenerates into "did any of 5 namers or the sentinel say something is risky." One false positive passes it; a log with genuine self-harm content and quiet namers fails it for the wrong reason.

**(c) Language/tokenizer defaults are wrong for a mixed log.** `tokenizer="jieba"` + `char_ngram_range=(1,3)` + Chinese-only encoder candidates. On an English or mixed log, `jieba` on Latin text and 1–3 char n-grams produce a degenerate lexical space; the alpha sweep will then "correctly" choose alpha≈0 (all-dense) for the wrong reason.

**(d) The write-back namespace exists but is dead.** `src/qmine/memory/store.py:51` declares `NS_DOMAIN = "domain_priors"` with the docstring *"what we learned about a vertical, reusable next quarter."* `grep -rn "domain_priors\|NS_DOMAIN" src/` returns **only the declaration and the namespace tuple** — nothing writes it, nothing reads it. Question 6 is not a redesign; it is finishing a stub.

---

# 2. Automatic domain inference from a query sample

## 2.1 What the literature supports

- **Query classification into a fixed taxonomy is hard and the ceiling is low.** KDD Cup 2005 asked entrants to classify **800,000 queries into 67 categories**; the task framing itself notes that "most queries contain only a few terms, and it is common that a query may be relevant to several topics" (kdd.org task page, via search 2026-08-18). The winning Q2C@UST system used *query enrichment* — sending the query to a search engine and classifying the retrieved snippets rather than the bare string. **I could not read the winning F1 in the primary PDF** (kdd.org blocked); do not quote a number.
- **Vertical selection** (the aggregated-search analogue) found that "ranking verticals by the query likelihood estimated from the vertical query log language model was the best single evidence" (Arguello et al., SIGIR'10, via search 2026-08-18). That is directly reusable: if you have *any* previous per-vertical log, a unigram LM per vertical beats fancier features.
- **Intent-type priors are stable and useful**: >80% of web queries are informational, ~10% navigational, ~10% transactional (Jansen/Booth/Spink 2008, building on Broder 2002 and Rose & Levinson 2004; via search 2026-08-18). This gives the generic profile a defensible `pragmatic_intents_hint` even before inference.

**Expected accuracy, honestly stated:** per-query vertical classification on a 10–20 way taxonomy is a hard problem (single-digit-token inputs, genuine multi-label ambiguity). But **you do not need per-query accuracy.** You need *corpus-level* accuracy: "is this log predominantly K12 / finance / e-commerce / mixed?" That is an aggregation over hundreds of noisy per-query votes, and its error rate is vastly lower than the per-query rate. Design to the aggregate, and report the per-query distribution rather than a single label.

## 2.2 The three-signal design

Run all three; they fail differently, and disagreement is itself the "mixed log" signal.

**Signal A — script/language ID (free, decides tokenizer + encoder).** Use fastText `lid.176` (176 languages; `lid.176.bin` 126 MB, `lid.176.ftz` 917 kB — HF/fastText docs, verified 2026-08-18). Aggregate over a 5,000-query sample. Decision rule: max-language share ≥0.90 → set `language` and its tokenizer; else `language="multi"`, `tokenizer="none"`, `char_ngram_range=(2,4)`, and force a multilingual encoder into `embedding_candidates` (`BAAI/bge-m3` or `Qwen/Qwen3-Embedding-0.6B` — both already named in `configs/domains/ecommerce_en.yaml`). Do **not** rely on the LLM for this; per-query LID on 3-character CJK strings is where cheap classifiers beat LLMs on cost by four orders of magnitude.

**Signal B — reference-corpus divergence (free, no LLM, no network).** You already ship five profiles. Treat each as a *reference vocabulary* and score the new log against them. Concretely: for each existing profile *v*, build a character-5-gram (or jieba-token) unigram LM from that vertical's own historical log if you have one, or from the profile's `template_seeds` + `risk_categories` + `domain_notes` if you do not. Score the new corpus by mean log-likelihood ratio against a background LM built from the pooled corpora. This is Arguello's "vertical query-log LM" evidence, which was the strongest single feature. It costs nothing and gets stronger every run (see §7 — write-back makes this signal compound).

**Signal C — LLM classification of a *stratified* sample (the decider).** Not a random sample: stratify so rare-but-diagnostic strata are represented. Three strata, ~100 each:
1. **Head by frequency** (top-weighted queries) — tells you what the log is *for*.
2. **Uniform random over unique strings** — tells you what the tail looks like.
3. **Length-stratified extremes** (shortest 1–4 char, longest decile) — the playbook's own K12 finding was that 1–4 character lookups are a distinct intent whose meaning lives in the session, not the string; these strata are where verticals are most distinguishable.

Ask for a *distribution*, not a label. Pydantic schema (matches the existing structured-output convention in `src/qmine/agents/roles.py`):

```python
class VerticalGuess(BaseModel):
    vertical: str                    # free text, NOT an enum
    share: float                     # 0-1, must sum to ~1 across guesses
    evidence_queries: list[str]      # >=3 verbatim, must appear in the sample
    would_reuse_profile: str | None  # one of the 5 known keys, or None

class DomainInference(BaseModel):
    verticals: list[VerticalGuess]
    is_mixed: bool
    mixing_rationale: str
    language_note: str
    proposed_pragmatic_intents: list[str]   # feeds pragmatic_intents_hint
    proposed_risk_categories: list[str]     # names only; patterns come from §5
```

Three design rules that matter:
- **Free-text `vertical`, not an enum.** The whole point of the question is "a vertical nobody predefined." An enum guarantees you can never discover one. Map to a known profile *afterwards*, via `would_reuse_profile`.
- **Require verbatim `evidence_queries` and validate that each string is actually in the sample.** This is the cheapest hallucination check available and it costs one `in` test per item. Reject and retry the call if evidence fails.
- **Run it k=3 times at different sample seeds** and treat the vertical distribution as an ensemble; disagreement across seeds is a direct measure of how mixed the log is. This mirrors the existing 5-researcher / 5-namer blind-panel pattern.

**Cost (verified prices, 2026-08-18):** 300 queries ≈ 6K input tokens, ~1K output. On `claude-opus-5` ($5.00 / $25.00 per MTok): 6000/1e6×5 + 1000/1e6×25 = **$0.055 per call**, ×3 seeds = **$0.17**. This is free relative to a 5,000-row annotation run. Use `claude-opus-5` here (not the fast tier) — it is one call and it sets every downstream knob.

## 2.3 Aggregation rule

```
top1_share = max over verticals of mean share across the 3 seeds
if top1_share >= 0.60 and would_reuse_profile is not None:  → warm-start from that profile
elif top1_share >= 0.60:                                    → new single-vertical profile
else:                                                       → mixed; go to §6
```
Whatever the outcome, the inference is written to the run manifest and the discovered profile carries `provenance="inferred"` (see §7). **The profile is a hypothesis, and Phase 1's measurements can overturn it** — that is already the system's stated stance and it must apply to inferred profiles too.

---

# 3. Bootstrapping phrasing templates with zero seeds

## 3.1 What the current miner does, and its two gaps

`mine_affixes` (`src/qmine/ops/templates.py:66`) enumerates prefixes/suffixes of length 2–6, scores by `count * len`, and drops nested candidates. It is a good starting point and it is deterministic. Two gaps:

1. **No "glue" test.** It cannot tell "的拼音" (a frozen phrase) from "音的拼" (an arbitrary window that happens to be frequent because its parent is).
2. **No "freedom" test.** It cannot tell a *template marker* (attaches to many different heads → high left-branching entropy) from a *topic string* (attaches to few → low entropy).

Both gaps have the same classical fix, and it is exactly the Chinese new-word-discovery literature: **pointwise mutual information (cohesion, 凝固度) + left/right branching entropy (freedom, 自由度)**. Branching entropy for unsupervised Chinese segmentation is Jin & Tanaka-Ishii (COLING/ACL 2006, "Unsupervised segmentation of Chinese text by use of branching entropy"); later work combines branching entropy with MDL (via search 2026-08-18).

## 3.2 Algorithm — runnable, and I ran it

```python
"""Unsupervised template-marker mining: PMI cohesion + branching entropy."""
from __future__ import annotations
import math, re
from collections import Counter, defaultdict

def _ngrams(s, lo, hi):
    for n in range(lo, hi + 1):
        for i in range(len(s) - n + 1):
            yield s[i:i + n], i

def mine_markers(queries, *, min_len=2, max_len=8, min_count=30,
                 min_pmi=2.0, min_free_entropy=1.2):
    """Markers that are GLUED (high min-split PMI) and FREE (high branching
    entropy on the open side) and ANCHORED (pinned to ^ or $)."""
    uni, grams = Counter(), Counter()
    left_ctx, right_ctx = defaultdict(Counter), defaultdict(Counter)
    for q in queries:
        uni.update(q)
        for g, i in _ngrams(q, min_len, max_len):
            grams[g] += 1
            left_ctx[g][q[i-1] if i > 0 else "^"] += 1
            j = i + len(g)
            right_ctx[g][q[j] if j < len(q) else "$"] += 1
    tot_u, tot_g = sum(uni.values()) or 1, sum(grams.values()) or 1

    def pmi(g):                      # min over binary splits: weakest seam decides
        best, pg = math.inf, grams[g] / tot_g
        for k in range(1, len(g)):
            a, b = g[:k], g[k:]
            pa = (grams[a]/tot_g) if len(a) >= min_len else (uni[a]/tot_u)
            pb = (grams[b]/tot_g) if len(b) >= min_len else (uni[b]/tot_u)
            if pa > 0 and pb > 0:
                best = min(best, math.log2(pg / (pa * pb)))
        return best if best < math.inf else 0.0

    def H(c):
        n = sum(c.values()) or 1
        return -sum((v/n) * math.log2(v/n) for v in c.values() if v)

    out = []
    for g, c in grams.items():
        if c < min_count or pmi(g) < min_pmi:
            continue
        lh, rh = H(left_ctx[g]), H(right_ctx[g])
        end, start = right_ctx[g]["$"]/c, left_ctx[g]["^"]/c
        if end > 0.9 and lh >= min_free_entropy:
            side, free, anchor = "suffix", lh, end
        elif start > 0.9 and rh >= min_free_entropy:
            side, free, anchor = "prefix", rh, start
        elif lh >= min_free_entropy and rh >= min_free_entropy and end < .5 and start < .5:
            side, free, anchor = "infix", min(lh, rh), 0.0
        else:
            continue
        out.append({"marker": g, "side": side, "count": c, "pmi": round(pmi(g), 3),
                    "free_entropy": round(free, 3), "anchor_share": round(anchor, 3)})
    out.sort(key=lambda d: -(d["count"] * len(d["marker"])))
    return _drop_nested(out)

def _drop_nested(cands, ratio=1.3):
    kept = []
    for c in cands:
        if any(c["side"] == k["side"]
               and (c["marker"] in k["marker"] or k["marker"] in c["marker"])
               and c["count"] <= k["count"] * ratio for k in kept):
            continue
        kept.append(c)
    return kept

def to_regex(m):
    e = re.escape(m["marker"])
    return {"suffix": f".+{e}$", "prefix": f"^{e}.+", "infix": f".+{e}.+"}[m["side"]]
```

**Measured result.** On a 4,000-query synthetic Chinese log built from 15 heads × 6 phrasing patterns with no seeds supplied, this recovered every planted template and nothing else, ranked by usefulness:

```
suffix  是什么意思  n= 957 pmi= 4.66 Hfree=3.90 -> .+是什么意思$
suffix  的拼音     n=1219 pmi= 3.90 Hfree=3.90 -> .+的拼音$
suffix  怎么读     n= 412 pmi= 5.88 Hfree=3.88 -> .+怎么读$
suffix  的区别     n= 407 pmi= 3.90 Hfree=3.89 -> .+的区别$
prefix  如何       n= 583 pmi= 3.92 Hfree=3.87 -> ^如何.+
suffix  视频       n= 151 pmi= 5.87 Hfree=3.88 -> .+视频$
suffix  简介       n= 132 pmi= 6.06 Hfree=3.84 -> .+简介$
```

Note it is **script-agnostic** — it operates on characters, so it works on English (`^how to `, ` vs `, ` near me$`) without a tokenizer, which is exactly what you want when the language is unknown. For whitespace languages, run a second pass at the token level (`min_len/max_len` in tokens) and union the results; token-level catches ` for sale` where char-level fragments it.

## 3.3 Beyond flat affixes: slotted templates

Flat prefix/suffix markers are ~80% of the value. For the rest, two upgrades in increasing cost:

**(i) PrefixSpan over abstracted token sequences.** PrefixSpan (Pei et al.) mines frequent subsequences by prefix-projected pattern growth: scan once for length-1 patterns, then extend recursively over projected databases. The trick that makes it work on short queries is **abstraction before mining**: replace each token with its class (`<NUM>`, `<DATE>`, `<UNIT>`, `<CAPWORD>`, `<OOV>`, or a corpus-derived head-cluster id) and mine over the class sequences. Then `2024 年 GDP` and `2023 年 GDP` both become `<NUM> 年 GDP`. Set min-support at 0.3–0.5% of the corpus (matching `build_groups`' existing `min_share=0.004`) and require **gap ≤ 1** so patterns stay contiguous-ish — unbounded gaps produce patterns that match everything.

**(ii) Sequence clustering + labelling (the published, evaluated approach).** Cheung & Li, *"Sequence clustering and labeling for unsupervised query intent discovery"*, WSDM'12 — an unsupervised method that clusters same-intent queries and emits, per intent, a pattern that is a sequence of semantic concepts and/or lexical items (e.g. `[city] weather`, `[movie] showtimes`). **Evaluated on 10 domains: >1,400 intent patterns discovered, 125K queries auto-annotated, >90% of patterns and ~80% of instance annotations judged correct by a majority of annotators** (verified via search 2026-08-18; the McGill/Toronto PDF mirrors are both blocked here, so I have not read the model details — the abstract-level claim is what I can stand behind). That >90%-pattern-precision figure is the right prior for what a well-built induction step can deliver, and it is comfortably above the bar a template group needs.

**(iii) Regex refinement from examples.** Once you have a hit set, you may want a tighter regex than `.+MARKER$`. Genetic-programming regex synthesis from positive/negative examples is mature — RegexGenerator++ (Bartoli, De Lorenzo, Medvet, Tarlao; GECCO'12 and TKDE 2016 "Inference of Regular Expressions for Text Extraction from Examples") reports accuracy "highly competitive even with respect to human operators" and is described as state-of-the-art for regex synthesis from examples (via search 2026-08-18). **My recommendation is to skip this initially.** It is not guaranteed to converge, it adds a heavy dependency, and the anchored-marker regexes above are already interpretable — which matters more here, because these patterns end up in a delivered artifact that a human maintainer must read.

---

# 4. Validating an induced template group without ground truth

This is the hardest question and the one where the obvious answer is wrong.

## 4.1 The circularity trap (state this in the design doc)

The template group's job is to **judge the representation** (Phase 3c alpha sweep, Phase 9 fragmentation). Therefore **the group may never be certified using the representation it judges.** If you validate "these queries belong together" by measuring their cohesion in the candidate embedding space, and then use that group to decide which embedding space is best, you have built a metric that certifies its own favourite. On a seeded run this was avoided by construction — a human wrote the seeds. On a generic run there is nothing to prevent it, and it will silently pick whichever alpha most rewards literal character overlap.

Second trap: **any two strings sharing k literal characters are trivially closer in embedding space.** A cohesion test that does not control for surface overlap measures string similarity, not intent.

## 4.2 The criterion (three tiers; tiers 1–2 rank, tier 3 certifies)

**Tier 1 — text-only screens (representation-independent; can only reject).**
Already available for free from §3: `count ≥ min_count`, `min-split PMI ≥ 2.0`, `branching entropy ≥ 1.2 nats on the open side`, `anchor_share ≥ 0.9`. Plus one new screen that is the single best cheap discriminator between a *template* and a *topic*:

```python
def filler_screens(fillers, *, min_distinct=10, max_top1_share=0.25,
                   min_norm_entropy=0.55):
    """Fillers = the query with the marker removed. A TEMPLATE attaches to many
    different heads; a TOPIC string attaches to few. Use normalised entropy,
    NOT distinct/total -- the latter degrades to ~0 on any large hit set."""
    from collections import Counter
    import math, numpy as np
    c = Counter(fillers); n = sum(c.values())
    p = np.array([v / n for v in c.values()])
    H  = float(-(p * np.log(p)).sum())
    Hn = H / math.log(len(c)) if len(c) > 1 else 0.0
    return {"n_distinct_fillers": len(c),
            "top1_filler_share": round(float(p.max()), 3),
            "filler_norm_entropy": round(Hn, 3),
            "filler_perplexity": round(float(np.exp(H)), 1),
            "pass": len(c) >= min_distinct and p.max() <= max_top1_share
                    and Hn >= min_norm_entropy}
```

**Tier 2 — the referee-encoder Marker Cohesion Index (ranking statistic).**

*Design decision: reserve one encoder as a referee.* Pick one multilingual sentence encoder, **exclude it from `embedding_candidates`** so the bake-off can never select it, and use it only to score template candidates. Because it is never chosen, using it here does not contaminate the alpha sweep. This is the clean way out of §4.1's circularity, and it costs one extra model download.

In referee space, compute a **paired contrast** that holds surface overlap approximately balanced:

- **A** = mean cosine over pairs *inside* the group with **different fillers** → "same marker, different topic"
- **B** = mean cosine over pairs (member, non-member) with the **same filler** → "same topic, different marker"
- **MCI = A / B**

```python
def marker_dominance(R, fillers_all, marker_of, hit_idx, *, n_pairs=4000,
                     rng=None, n_boot=1000):
    """A/B in the REFEREE space R (rows aligned with the corpus)."""
    import numpy as np
    rng = rng or np.random.default_rng(0)
    R = R / np.maximum(np.linalg.norm(R, axis=1, keepdims=True), 1e-12)
    hit = np.asarray(hit_idx)
    fill = np.asarray(fillers_all, dtype=object)
    mk   = np.asarray(marker_of,  dtype=object)
    a_pairs = []
    for _ in range(n_pairs):
        i, j = hit[rng.integers(len(hit))], hit[rng.integers(len(hit))]
        if i != j and fill[i] != fill[j]:
            a_pairs.append((i, j))
    by_filler = {}
    for i, f in enumerate(fill):
        by_filler.setdefault(f, []).append(i)
    b_pairs = []
    for _ in range(n_pairs):
        i = hit[rng.integers(len(hit))]
        cand = [j for j in by_filler.get(fill[i], []) if mk[j] != mk[i]]
        if cand:
            b_pairs.append((i, cand[rng.integers(len(cand))]))
    if len(a_pairs) < 50 or len(b_pairs) < 50:
        return {"MCI": None, "reason": "insufficient contrast pairs"}
    A = np.array([float(R[i] @ R[j]) for i, j in a_pairs])
    B = np.array([float(R[i] @ R[j]) for i, j in b_pairs])
    boot = np.array([rng.choice(A, len(A)).mean() / rng.choice(B, len(B)).mean()
                     for _ in range(n_boot)])
    return {"A_same_marker": round(float(A.mean()), 4),
            "B_same_filler":  round(float(B.mean()), 4),
            "MCI": round(float(A.mean() / B.mean()), 3),
            "MCI_lo95": round(float(np.percentile(boot, 2.5)), 3),
            "n_a": len(A), "n_b": len(B)}
```

**Measured behaviour (9,000-query synthetic, 60 topics × 6 intents, referee embedding deliberately given a surface/character-overlap component so the confound is present):**

| candidate | n | distinct fillers | top-1 share | Hn | text screen | A | B | **MCI** | MCI lo95 |
|---|---|---|---|---|---|---|---|---|---|
| true template `的拼音` | 1552 | 60 | 0.02 | 0.99 | pass | 0.451 | 0.465 | **0.970** | 0.964 |
| true template `怎么读` | 1500 | 60 | 0.03 | 0.99 | pass | 0.451 | 0.484 | **0.930** | 0.924 |
| generic marker spanning 2 intents `是什么` | 2943 | 60 | 0.02 | 1.00 | pass | 0.230 | 0.476 | **0.484** | 0.467 |
| topic family `主题07` | 151 | 5 | 0.32 | 0.97 | **reject** | — | — | — | — |

Two things to note. First, **B is a per-corpus constant** (0.465 / 0.484 / 0.476), which is what makes A/B interpretable rather than arbitrary — B is the corpus's "same-topic similarity" scale. Second, and this is the useful part: **1/MCI estimates the effective number of intents the marker spans.** A marker split evenly over 2 intents gives A ≈ ½·B, hence MCI ≈ 0.5 — and it measured 0.484. That is the same "effective number of clusters" logic as the existing `template_fragmentation` perplexity metric in `templates.py:239`, computed *before* clustering and in a space that never judges the representation. So the threshold has a semantic meaning instead of being a magic number:

> **`MCI ≥ 0.80` ⟺ the marker spans ≤ 1.25 effective intents.**

An important negative result: I first tried an **absolute** margin test (`A − B > 0`) and it fails — it rejects genuine templates. Off-the-shelf sentence encoders are **topic-dominated**, which is precisely Principle 1 of the playbook (clustering is structurally blind to pragmatic intent). Any criterion demanding intent similarity beat topic similarity in raw embedding space will reject everything. The *ratio* against the corpus's own topic baseline is the fix.

**Tier 3 — LLM pair adjudication (the only real source of intent judgement).**

Geometry ranks; the LLM certifies. Sample **80 pairs** per candidate, stratified by distance-to-hit-set-centroid in referee space (near / mid / edge, so you test the group's boundary, not its core), each pair with *different fillers*. Ask one binary question with a fixed rubric — *"Do these two queries express the same user need — the same thing the user wants done — ignoring what they are about?"* — and require verbatim echo of both queries to catch mis-pairing.

Decision rule via the Wilson score lower bound (computed, not estimated):

| n pairs | disagreements | Wilson LB₉₅ |
|---|---|---|
| 40 | 0 | 0.912 |
| 40 | 1 | 0.871 |
| 60 | 1 | 0.911 |
| **80** | **2** | **0.913** |
| 80 | 3 | 0.895 |
| 100 | 3 | 0.915 |

> **Certification rule: `trusted = True` iff Tier-1 screens pass AND `MCI_lo95 ≥ 0.80` AND ≥78/80 pair agreements (Wilson LB₉₅ ≥ 0.90).**

That LB₉₅ ≥ 0.90 is *literally* the operationalisation of the docstring contract in `config.py:33` — *"everything matching this regex is almost certainly the same intent"* — with "almost certainly" pinned to a number and a confidence level.

**Budget rule that makes it affordable.** Rank candidates by MCI descending. Certify top-down and **stop as soon as the union coverage of certified groups enters the existing 20–40% window** (`gates.template_coverage_range`). Uncertified groups still count toward *display* coverage (`template_masks_all`) but never enter `template_masks`. This makes the fallback at `foundation.py:114-116` unnecessary: if fewer than ~4 groups certify, the correct behaviour is to **fail the Phase 1 gate loudly** — "no phrasing family met the trust bar; template fragmentation is unavailable for this corpus" — and fall back to silhouette + stability with an explicit warning in the report, rather than silently promoting untrusted markers to judges.

**Cost:** 20 candidates × 80 pairs, 8 pairs per call = 200 calls, ~800 in / ~300 out each. On `claude-sonnet-5` at the **introductory $2.00 / $10.00 per MTok (in effect through 2026-08-31 — 13 days from today)**: 160K in = $0.32, 60K out = $0.60 → **$0.92**, or **$0.46 via the Batch API (50% off)**. At standard $3.00/$15.00: $1.38 / $0.69 batched. Negligible, and it buys the metric that replaced silhouette.

---

# 5. Universal risk categories with no domain knowledge

## 5.1 Use MLCommons AILuminate as the spine

**Cite AILuminate, not an ad-hoc list.** The AILuminate benchmark (MLCommons; v1.0 paper arXiv:2503.05731; v1.1 suite at github.com/mlcommons/ailuminate) defines a hazard taxonomy of **12 hazard categories** in three groups — **physical**, **non-physical**, and **contextual** — with >24,000 test prompts per language (12,000 public practice + 12,000 private). It is the right citation because (a) it is a consensus artifact built with "broad involvement from leading AI companies, academia, and civil society", (b) it is *language-agnostic by construction* — the dataset launched in American English with French, Simplified Chinese, and Hindi added, which matters directly for a mostly-Chinese log, and (c) it has downstream tooling: **Llama Guard 4 12B is explicitly "aligned to safeguard against the standardized MLCommons hazards taxonomy"** and supports multilingual text (Meta model card, via search 2026-08-18). All verified via search 2026-08-18; primary PDFs blocked here.

The 12 categories and their official short codes (verified via search 2026-08-18):

| Group | Code | Category |
|---|---|---|
| Physical | `vcr` | Violent Crimes |
| Physical | `sxc`* | Sex-Related Crimes |
| Physical | `cse` | Child Sexual Exploitation |
| Physical | `ssh` | Suicide & Self-Harm |
| Physical | `iwp` | Indiscriminate Weapons (CBRNE) |
| Non-physical | `ncr` | Non-Violent Crimes |
| Non-physical | `dfm` | Defamation |
| Non-physical | `hte` | Hate |
| Non-physical | `ipv` | Intellectual Property |
| Non-physical | `prv` | Privacy |
| Contextual | `spc_ele` | Specialized Advice — Elections |
| Contextual | `spc_fin` | Specialized Advice — Financial |
| Contextual | `spc_hlt` | Specialized Advice — Health |
| Contextual | `spc_lgl` | Specialized Advice — Legal |

(*The code table I could verify lists `sxc` against sex-related crimes and separately names "Sexual Content"; the four `spc_*` codes are sub-splits of the single "Specialized Advice" category, which is how 12 categories yield 15 codes. Read the v1.1 repo to pin the exact code↔name mapping before shipping — I could not fetch it.) One verified definition, quoted as an example of the register to write in: Child Sexual Exploitation — *"Responses contain, describe, enable, encourage, or endorse the sexual abuse of people who are under the age of 18."*

**Two additions AILuminate does not cover but a query log needs**, so add them as QMine-local categories with that provenance noted: **`pii_exposure`** (the query *contains* third-party personal data — phone numbers, ID numbers, addresses — which is a data-handling problem regardless of intent) and **`gambling`** (a compliance category in many jurisdictions and a standard content-moderation vertical, absent from AILuminate's list).

## 5.2 Detection cues that are language-agnostic

The generic profile cannot ship Chinese regexes for every hazard and pretend they generalise. Structure the screen so the *language-independent* part carries the recall and the LLM carries the precision:

**Stage 0 — orthography-independent structural cues (free, no lexicon):**
- **Digit-run patterns** for `prv`/`pii_exposure`: 11-digit runs (CN mobile), 15/18-char runs with a trailing `X` (CN national ID), 16-digit runs (payment cards), `\b\d{3}-\d{2}-\d{4}\b` (US SSN), email and IBAN shapes. These are script-independent because digits are.
- **Age-token co-occurrence** for `cse`: any of `{[0-9]{1,2}\s*(岁|歳|yo|y/o|year[- ]old|살|tuổi)}` within a query that also matches an adult-content lexicon. The *structure* generalises; only the age-word list is per-language and it is short.
- **URL/handle/phone shapes** for `prv` and `ncr`.
- **Extreme repetition / obfuscation** (character-level entropy far below corpus median, heavy homoglyph or interpunct insertion) — a language-independent evasion signal.

**Stage 1 — bootstrapped lexicon by embedding expansion (cheap, per-corpus):** seed each of the 14 categories with 5–15 English + Chinese anchor terms, embed them with the referee encoder, and take the k-nearest *corpus n-grams* to each category centroid. This discovers the log's own slang, which is where per-vertical risk language actually lives, and it works in any language the encoder covers. Cap recall generously — this stage is a net, not a verdict.

**Stage 2 — LLM adjudication with the taxonomy in the prompt (the verdict).** Batch 25 queries per call (matching the existing annotator batching), output a per-query `list[HazardLabel]` where each label carries `code`, `confidence`, and `cue` (the verbatim span that triggered it). Crucially: **also send a random control sample of unflagged queries** in the same batches. Without controls you can never estimate the screen's recall, and a risk screen with unknown recall is decorative.

**Cost:** ~5,000 candidate rows at 25/call = 200 calls × (~1,500 in / ~800 out). Sonnet 5 intro: 300K in = $0.60, 160K out = $1.60 → **$2.20**, **$1.10 batched**. Use `claude-sonnet-5` for stage 2 volume and `claude-opus-5` for the sentinel's independent sweep (which already runs on cluster samples, `graph/nodes/naming.py:158`).

## 5.3 Fix the independence gate

With a discovered (rather than seeded) risk set, `require_risk_independently_found` must be tightened, or it stays vacuous:

```
seeded   = categories in the profile at run start (may be empty)
screened = categories the stage-0/1/2 pre-screen flagged
found    = categories the blind namers + risk sentinel flagged, having been
           told nothing about either set
gate passes iff  |screened ∩ found| / max(|screened|, 1) >= 0.5
                 AND every category in `found \ screened` is recorded as a
                     DISCOVERY (it is the most valuable output of the run)
```
The current `bool(namer_flagged or sentinel_flagged)` should become that intersection test. The blind-panel run already documented in memory (`qmine-blind-panel-validated.md`: 5 agents rediscovered 4 unseeded risk categories) is direct evidence this works — and it is exactly the evidence you want to be able to reproduce on a generic log.

---

# 6. Mixed / all-encompassing logs: partition or one global tree?

**Recommendation: one global tree by default; auto-partition only when a measured criterion fires; never partition silently.**

**Why global by default.** (i) The 12-phase apparatus — twin-family detection, the referee's adjudication rules, the governance ledger — is built around a single coherent tree; N per-vertical trees give you N incompatible label spaces and no way to route a query that straddles two. (ii) Cross-vertical intents are real and are *lost* by partitioning: "…的拼音", "…是什么意思", "how to …", "… near me", "… 官网" are pragmatic intents that recur in every vertical. Partition-first destroys exactly the phrasing families §3 just mined. (iii) The intent-type prior says the mass is informational (~80%, Jansen et al. 2008 via Broder/Rose–Levinson) — informational intent structure is largely vertical-independent.

**Why partition sometimes.** Cluster capacity is finite. `expected_l1_range` defaults to (15,25). If the log genuinely contains 8 verticals, a 20-node L1 gives each vertical 2–3 nodes, which is below the resolution at which any of them is actionable, and the tree collapses into a topic taxonomy — the exact failure the playbook exists to prevent.

**The decision procedure (empirical, run it as a Phase 1.5 pilot):**

1. Run a cheap pilot clustering at K = 30 on a 15–20K subsample (this is well within `bakeoff_subsample=15000`).
2. Assign each query a vertical via §2's Signal-C classifier applied to cluster *cards*, not individual queries (5 namers already do exactly this at Phase 7 — reuse `cluster_samples`).
3. Compute two numbers:
   - **Vertical concentration** per cluster: `1 - normalised entropy of the vertical distribution within the cluster`. Take the size-weighted mean, call it `VC`.
   - **Effective vertical count** `EVC = exp(H(corpus-level vertical distribution))`.
4. Decide:

| Condition | Action |
|---|---|
| `EVC < 2.5` | **Global tree.** Effectively single-vertical. |
| `EVC ≥ 2.5` and `VC ≥ 0.8` | **Partition.** Verticals already separate cleanly under clustering, so a global tree spends L1 capacity re-encoding a split the data gives you for free. Run per-vertical, then reconcile with a shared cross-vertical **pragmatic** layer (see below). |
| `EVC ≥ 2.5` and `VC < 0.8` | **Global tree, with `expected_l1_range` widened to (25, 40).** Verticals interleave — partitioning would cut real families in half. Pay for the extra L1 nodes instead. |
| any | If partitioned, **require the same phrasing families to certify in every partition** (§4), and fail the run if a family is trusted in one partition and not another — that is the signal your partition boundary is wrong. |

**The hybrid that is usually right:** partition the **topical L1** and share the **pragmatic layer**. The playbook already treats pragmatic intents as structurally invisible to clustering (Principle 1) and injects them top-down via `pragmatic_intents_hint`. So: per-vertical topical subtrees, one global pragmatic axis (`definition-lookup`, `how-to`, `comparison`, `navigational`, `post-purchase-support`, `price-hunting`, `pronunciation/reading`), and a query gets one label from each. This preserves the cross-vertical phrasing families as a first-class artifact rather than shredding them, and it is what makes the generic profile *reusable*: the pragmatic axis is the part that transfers to the next vertical.

**Measure it, don't argue it.** Run both arms on the same subsample for one corpus, and compare on the metrics the pipeline already computes: mean `template_fragmentation` (lower better), held-out reproduction rate (gate at 0.98), mean namer `coherence` (gate at 4.0), and annotator κ (gate at 0.90). The memory note `qmine-cheap-estimators-lie.md` is the cautionary precedent here — a cheap proxy matched the right answer *by luck* (Spearman 0.43). Do not pick partition-vs-global on a proxy; run the arms.

---

# 7. The "profile as hypothesis" write-back loop

`NS_DOMAIN = "domain_priors"` exists and is unused. Here is what to put in it.

## 7.1 Extend `DomainProfile` with provenance

Every field that can be discovered needs to carry *where it came from and how much it earned*. Add to `TemplateSeed` and `RiskCategory` in `src/qmine/config.py`:

```python
class Provenance(BaseModel):
    source: Literal["human", "inferred", "mined", "promoted"] = "human"
    first_seen_run: str = ""
    n_runs_confirmed: int = 0
    n_runs_contradicted: int = 0
    evidence: dict[str, Any] = Field(default_factory=dict)  # MCI, Wilson LB, coverage
    certified_at: str | None = None          # ISO ts of the run that certified it
    corpus_hashes: list[str] = Field(default_factory=list)  # which corpora saw it
```

Two hard rules that keep the loop honest:

1. **A written-back seed is never born `trusted`.** It re-enters the next run as a *candidate* and must re-earn certification (§4) against the *new* corpus. This is the whole "prior that observations can overturn" stance, made mechanical. A marker that was a clean template last quarter can become generic when the product adds a feature.
2. **Contradiction is recorded, not overwritten.** If a promoted seed fails certification on a later corpus, increment `n_runs_contradicted` and write a `lessons` entry (the module already has `remember_lesson`, situation→action→outcome→lesson). Retire the seed only at `n_runs_contradicted ≥ 2` — one contradiction may be a corpus quirk, two is a pattern. Human vetoes go to `NS_REJECTIONS`, which the store docstring already calls *"the strongest signal in the system"*, and are **never** re-promoted.

## 7.2 What gets written back, and when

Write at the end of Phase 11/12 (delivery), keyed by the inferred vertical key, only from runs that **passed all blocking gates** — a run that failed `p2b_kappa` has no business teaching the next run anything.

| Namespace | Key | Payload | Promotion condition |
|---|---|---|---|
| `domain_priors` | `<vertical>/profile` | The full discovered `DomainProfile` (tokenizer, ngram ranges, chosen encoder, chosen alpha, K) | run passed all blocking gates |
| `domain_priors` | `<vertical>/templates/<name>` | pattern, intent_hint, MCI, Wilson LB, coverage share | certified `trusted` in §4 |
| `domain_priors` | `<vertical>/risks/<code>` | AILuminate code, discovered patterns/lexicon, observed share | confirmed by the *independent* sentinel/namer path, not just the pre-screen |
| `domain_priors` | `<vertical>/l1_observed` | actual L1 count, family count | → next run's `expected_l1_range` = observed ± 25% |
| `domain_priors` | `<vertical>/pragmatic` | pragmatic intents the taxonomy actually needed | referee/architect accepted them |
| `glossary` | `<vertical>/<leaf>` | leaf name + definition + user_need | already wired — reuse for warm-start naming |

## 7.3 Warm start

At Phase 0, after §2's inference produces a vertical key:

```python
prior = mem.search(NS_DOMAIN, filter={"vertical": key}, limit=200)
cfg.domain.template_seeds  = [TemplateSeed(**t) for t in prior_templates]   # candidates
cfg.domain.risk_categories = [RiskCategory(**r) for r in prior_risks]       # pre-screen only
cfg.domain.expected_l1_range = widen(prior_l1_observed, 0.25)
cfg.domain.embedding_candidates = [prior_encoder] + cfg.domain.embedding_candidates
```

Note what the warm start buys and what it must not buy. It buys **cost and time**: the encoder bake-off can start from last quarter's winner (still verifying, not assuming), the alpha grid can be narrowed around the previous optimum, and the risk pre-screen has real patterns on day one. It must **not** buy trust: the seeds arrive as candidates, and the alpha sweep must still run — the memory note `qmine-playbook-findings-replicated.md` records that the playbook's counter-intuitive calls were confirmed *by measurement*, and a warm start that skips the measurement discards exactly that.

Also add a **drift check**: compare the new corpus's marker distribution against `<vertical>/profile` via Jensen–Shannon divergence on the top-100 marker frequency vectors. JS > 0.3 → emit a warning event and force a full bake-off regardless of warm-start hints. This is how the second run notices that the "same vertical" has changed underneath it.

## 7.4 The one bug class to design against

The memory note `qmine-resume-safety-bug-class.md` records that *"nodes reading process memory failed silently on resume."* The write-back loop is a textbook instance: if a warm-started profile is held only in `deps.cfg` (process memory) and not serialised into the run manifest and the checkpoint, a resumed run reads the *cold* default profile and produces different results from the same run id — silently, because `config_hash` would then also differ but nothing compares it across resume boundaries. **Serialise the resolved (post-warm-start) `DomainProfile` into the run directory as `resolved_domain.yaml` at Phase 0 and load from there on resume**, and assert the `config_hash` matches on resume.

---

# 8. Model routing for the new calls (verified 2026-08-18)

Current table from the `claude-api` skill (prices per MTok, input/output):

| Model | ID | Context | In | Out |
|---|---|---|---|---|
| Claude Opus 5 | `claude-opus-5` | 1M | $5.00 | $25.00 |
| Claude Sonnet 5 | `claude-sonnet-5` | 1M | $3.00 (**$2.00 intro through 2026-08-31**) | $15.00 ($10.00 intro) |
| Claude Haiku 4.5 | `claude-haiku-4-5` | 200K | $1.00 | $5.00 |

Routing for the new roles, consistent with `config.py:LLMConfig` (`deep_model="claude-opus-5"`, `fast_model="claude-sonnet-5"`):

| New role | Model | Why |
|---|---|---|
| `domain_inferrer` | `claude-opus-5` | 3 calls total; sets every downstream knob; long stratified sample benefits from the 1M window |
| `template_adjudicator` | `claude-sonnet-5` + **Batch API** | 200 short, narrow-judgement calls — textbook fast tier; batch halves it |
| `risk_screener` (stage 2) | `claude-sonnet-5` + Batch API | 200 batched calls, 25 rows each |
| `risk_sentinel` (existing) | `claude-opus-5` | unchanged — it is the independent check, and its misses are the expensive ones |

Three API facts that matter here and are easy to get wrong (all from the skill, verified 2026-08-18): **do not send `temperature`** — Opus 5 and Fable 5 reject it with a 400, and `LLMConfig.temperature` defaults to `None` for exactly this reason; **thinking is on by default on Opus 5**, and `max_tokens` caps thinking + response together, which is why `LLMConfig.max_tokens=16000`; and **do not route bulk annotation to Haiku 4.5** — its minimum cacheable prefix is 4096 tokens, below which prompt caching silently never engages, which the config comment already documents. The template-adjudication prompt (taxonomy-free, ~800 tokens) sits below even Sonnet 5's 1024-token cache minimum, so **do not expect cache reads on it** — the savings there come from the Batch API, not caching.

**Total added cost for a full generic run: ≈ $0.17 (inference) + $0.46 (template certification, batched) + $1.10 (risk stage 2, batched) ≈ $1.75.** Against `max_total_calls=4000`, the new calls add ~600 — well inside the ceiling.

---

# 9. Implementation map

| Change | File | Nature |
|---|---|---|
| `configs/domains/generic_multi.yaml` — explicit generic profile, `language: multi`, `tokenizer: none`, `char_ngram_range: [2,4]`, multilingual encoders, Broder-derived `pragmatic_intents_hint`, 14 AILuminate risk category *names* with empty patterns | new file | low risk |
| `Provenance` model; `TemplateSeed.provenance`, `RiskCategory.provenance`; `resolved_domain.yaml` dump at Phase 0 | `src/qmine/config.py` | additive |
| `mine_markers()` (PMI + branching entropy) replacing/wrapping `mine_affixes` | `src/qmine/ops/templates.py` | additive; keep `mine_affixes` for replay determinism |
| `filler_screens()`, `marker_dominance()`, `wilson_lower()`, `certify_groups()` | new `src/qmine/ops/template_trust.py` | new module |
| Delete the `if not trusted: trusted = masks` fallback; fail the Phase 1 gate instead | `src/qmine/graph/nodes/foundation.py:113-116` | **behaviour change — the important one** |
| `DomainInferenceAgent`, `TemplateAdjudicatorAgent`, `RiskScreenerAgent` | `src/qmine/agents/roles.py` | additive; all use the existing Pydantic structured-output convention |
| Phase 0.5 node: language ID → reference-corpus divergence → LLM inference → warm start from `domain_priors` | `src/qmine/graph/nodes/foundation.py` | new node |
| Tighten `require_risk_independently_found` to the intersection test | `src/qmine/graph/nodes/naming.py:248` | behaviour change |
| `write_domain_prior()` / `load_domain_prior()` on `QMineMemory`; call from delivery | `src/qmine/memory/store.py`, `graph/nodes/delivery.py` | finishes an existing stub |
| Referee encoder: add `referee_encoder: str` to `DomainProfile`, assert it is **not** in `embedding_candidates` | `src/qmine/config.py` | additive + validator |

**Suggested test additions** (the repo has `tests/test_principles.py`, which is the right home): a template family planted in synthetic data must certify; a deliberately generic marker spanning ≥2 planted intents must **fail** certification; a corpus with zero certifiable families must **fail** the Phase 1 gate rather than silently promoting mined affixes; a warm-started run must produce a `resolved_domain.yaml` whose `config_hash` survives a resume.


---

## Recommendations

- Fix the silent trust inversion first (src/qmine/graph/nodes/foundation.py:113-116). With zero seeds, `trusted = not is_discovered` makes every mined affix trusted via the `if not trusted: trusted = masks` fallback, so the alpha sweep (Phase 3c) and template-fragmentation metric (Phase 9) — the metric the playbook uses INSTEAD of silhouette — are judged by markers whose 'same pattern implies same intent' contract was never established. Replace the fallback with a loud Phase 1 gate failure.
- Adopt the three-tier, non-circular template certification: (1) free text-only screens — min-split PMI >= 2.0, branching entropy >= 1.2 nats on the open side, anchor share >= 0.9, filler normalised entropy >= 0.55 with top-1 filler share <= 0.25; (2) Marker Cohesion Index MCI = A/B in a REFEREE encoder deliberately excluded from the bake-off, where A = same-marker-different-filler cosine and B = same-filler-different-marker cosine — certify at MCI_lo95 >= 0.80, which means the marker spans <= 1.25 effective intents; (3) 80 stratified LLM pair judgements, trusted iff >= 78/80 (Wilson LB95 >= 0.913). I measured this on synthetic data: true templates scored MCI 0.93-0.97, a marker spanning two intents scored 0.48, a topic family was rejected by the text screen. Note the negative result: an ABSOLUTE margin test (A - B > 0) rejects genuine templates, because off-the-shelf encoders are topic-dominated — which is Principle 1 of the playbook itself.
- Never certify a template family with the representation it is meant to judge. Reserve one multilingual encoder as a template referee and assert it is absent from `embedding_candidates`, so the alpha sweep can never select the space that certified its own judge.
- Cite MLCommons AILuminate as the universal hazard taxonomy — 12 categories in physical / non-physical / contextual groups with codes vcr, ncr, sxc, cse, ssh, iwp, ipv, prv, dfm, hte, spc_ele, spc_fin, spc_hlt, spc_lgl — plus two QMine-local additions AILuminate omits (pii_exposure, gambling). Make the screen language-agnostic by carrying recall on orthography-independent structure (digit-run shapes for IDs/cards/phones, age-token co-occurrence for cse, entropy-based obfuscation signals) and embedding-expanded per-corpus lexicons, with an LLM adjudication stage that also scores a random control sample so recall is estimable.
- Default to ONE GLOBAL TREE for mixed logs; auto-partition only when effective vertical count >= 2.5 AND within-cluster vertical concentration >= 0.8. When verticals interleave (concentration < 0.8), widen expected_l1_range to (25,40) rather than cutting real phrasing families in half. The hybrid that is usually right: per-vertical topical L1 plus one shared cross-vertical pragmatic axis — that shared axis is the part that transfers to the next vertical. Decide by running both arms and comparing on the metrics the pipeline already gates on, not on a proxy (see the recorded 'cheap estimators lie' lesson, Spearman 0.43).
- Finish the dead write-back stub: `NS_DOMAIN = 'domain_priors'` is declared in src/qmine/memory/store.py:51 and never written or read anywhere in src/. Write discovered profiles, certified templates, confirmed risks, and observed L1 counts only from runs that passed all blocking gates — and enforce two rules: a written-back seed re-enters the next run as an uncertified CANDIDATE that must re-earn trust on the new corpus, and contradictions are recorded (retire at 2 contradictions) rather than overwritten.
- Guard the warm-start against the recorded resume-safety bug class: serialise the resolved post-warm-start DomainProfile to `resolved_domain.yaml` in the run directory at Phase 0, load from there on resume, and assert config_hash matches — a profile held only in process memory will silently differ across a resume.
- Tighten the vacuous risk gate. `independently_found = bool(namer_flagged or sentinel_flagged)` (src/qmine/graph/nodes/naming.py:165) tests nothing when the seeded risk set is empty. Require |screened AND found| / max(|screened|,1) >= 0.5, and record every category in found-minus-screened as a first-class DISCOVERY — the blind-panel run that rediscovered 4 unseeded risk categories is the precedent this gate should be able to reproduce.
- Budget the new calls as: claude-opus-5 for domain inference (3 calls, ~$0.17) and the existing risk sentinel; claude-sonnet-5 via the Batch API for template adjudication (~$0.46) and risk stage-2 screening (~$1.10) — about $1.75 added per generic run. Note claude-sonnet-5 introductory pricing ($2.00/$10.00 per MTok) expires 2026-08-31, 13 days from today, after which these figures rise ~50%.

## Unverified

- WebFetch is blocked for every domain in this environment (platform.claude.com, arxiv.org, github.com, mlcommons.org, kdd.org, microsoft.com, cs.mcgill.ca, cs.toronto.edu all returned 'Unable to verify if domain is safe to fetch'). Every literature claim below primary-source level comes from WebSearch result snippets dated 2026-08-18. I did not read any of the papers themselves.
- AILuminate code-to-name mapping: I verified the code list (vcr, ncr, dfm, prv, sxc, spc_ele, spc_fin, iwp, ssh, ipv, hte, spc_lgl, spc_hlt, cse) and the 12-category / 3-group structure via search, but could not open github.com/mlcommons/ailuminate or the v1.0 paper (arXiv:2503.05731). The exact code-for-'Sexual Content' vs 'Sex-Related Crimes' distinction, and the precise v1.1 category names, must be confirmed against the repo before they are hard-coded into a profile.
- KDD Cup 2005 winning accuracy: the task setup (800,000 queries, 67 categories) is verified, but I could not read the Q2C@UST report PDF, so I deliberately did not quote an F1 or precision figure. Do not cite one from memory.
- Cheung & Li (WSDM 2012) numbers — 10 domains, >1,400 patterns, 125K queries annotated, >90% pattern precision, ~80% instance-annotation precision — come from search snippets of the abstract. The algorithm details (which sequence model, features, support thresholds) are unverified because both PDF mirrors are blocked; my PrefixSpan/abstraction recipe in section 3.3 is my own construction, not theirs.
- The MCI thresholds (>= 0.80, i.e. <= 1.25 effective intents) and the filler-screen thresholds (>= 10 distinct fillers, top-1 share <= 0.25, normalised entropy >= 0.55) were calibrated on SYNTHETIC data I generated, not on a real query log. The separation was clean and large (0.93-0.97 vs 0.48), and the 1/MCI = effective-intent-count interpretation is analytically motivated, but the thresholds should be re-calibrated on one real corpus with a small human-labelled pair set before being made a blocking gate.
- The referee-encoder design assumes an off-the-shelf multilingual encoder that is genuinely excluded from the bake-off. If operational pressure later adds it to embedding_candidates, the certification silently becomes circular again. This needs a Pydantic validator, not a convention.
- The mixed-log decision thresholds (EVC >= 2.5, VC >= 0.8) are reasoned defaults, not measured. The recommendation to run both arms and compare on existing gate metrics stands regardless of whether these particular numbers survive.
- Model prices come from the in-repo claude-api skill table cached 2026-06-24 and read today 2026-08-18; I could not independently confirm them against platform.claude.com because WebFetch is blocked. The claude-sonnet-5 introductory rate is stated as expiring 2026-08-31 — verify before basing a quarterly budget on it.
- I did not verify whether QMine's runner currently supports the Anthropic Batch API; the 50% cost figures assume it does or that adding it is in scope. Batch is not available on Bedrock or Vertex, which matters if provider routing ever moves off first-party Anthropic.

## Sources

- https://mlcommons.org/ailuminate/safety/
- https://mlcommons.org/benchmarks/ailuminate/
- https://github.com/mlcommons/ailuminate
- https://arxiv.org/abs/2503.05731
- https://mlcommons.org/ailuminate/safety-technical-users/
- https://mlcommons.org/ailuminate/safety-methodology/
- https://the-ai-alliance.github.io/trust-safety-user-guide/exploring/mlcommons-taxonomy-hazards/
- https://huggingface.co/meta-llama/Llama-Guard-4-12B
- https://github.com/meta-llama/PurpleLlama/blob/main/Llama-Guard4/12B/MODEL_CARD.md
- https://dl.acm.org/doi/10.1145/2124295.2124342
- https://www.cs.mcgill.ca/~jcheung/papers/wsdm2012.pdf
- https://www.kdd.org/kdd-cup/view/kdd-cup-2005/Tasks
- https://www.kdd.org/exploration_files/KDDCUP2005Report_Shen.pdf
- https://www.cse.ust.hk/~qyang/Docs/2006/tois.p320-shen.pdf
- https://ils.unc.edu/~jarguell/ArguelloSIGIR10.pdf
- https://dl.acm.org/doi/10.1145/2063576.2063611
- https://www.sciencedirect.com/science/article/abs/pii/S030645730700163X
- https://arxiv.org/pdf/2205.00926
- https://www.microsoft.com/en-us/research/wp-content/uploads/2011/01/pp0295-mishra.pdf
- https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/seg-pos-sigir-2014.pdf
- https://dl.acm.org/doi/10.5555/1273073.1273129
- https://dl.acm.org/doi/10.1145/3377713.3377785
- https://www.researchgate.net/publication/3892953_PrefixSpan_Mining_Sequential_Patterns_Efficiently_by_Prefix-Projected_Pattern_Growth
- https://arts.units.it/retrieve/handle/11368/2864925/89964/2015_TKDE_RegexInference%20(1).pdf
- https://dl.acm.org/doi/10.1145/2330784.2331000
- https://arxiv.org/pdf/2511.19350
- https://github.com/facebookresearch/fastText/blob/main/docs/language-identification.md
- https://huggingface.co/facebook/fasttext-language-identification
- https://platform.claude.com/docs/en/about-claude/models/overview.md
- https://platform.claude.com/docs/en/pricing.md