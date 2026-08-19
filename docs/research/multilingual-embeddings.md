# Multilingual Embeddings for Mixed CJK/Latin Logs

> Gathered 2026-08-18. Facts marked verified were fetched live; the model-landscape
> dossier in particular flags its prices as secondary-source and unconfirmed —
> which is why the running system fetches prices from a live catalogue rather
> than embedding any table from this document.

> **Verification stance.** Every model spec below was pulled live from the Hugging Face API/`config.json` on **2026-08-18** from inside this machine's sandbox. Every claim in §1 and §3 is a **measurement I ran on this machine on 2026-08-18** against the models already cached in `QMine/.hf` — not recalled from memory. Commercial API prices are the weakest part of this report (see Uncertainties): `WebFetch` to `huggingface.co`, `github.com`, `openai.com` etc. is blocked by network policy in this environment, so prices come from search-result aggregators, not vendor pages.

> **Important pre-existing context the assignment did not mention:** QMine **already has** `src/qmine/ops/language.py` (340 lines), wired into `graph/nodes/foundation.py` and `graph/nodes/bottomup.py`. It already implements script profiling, a `nonlinguistic` class, an `alignment_probe`, a `minority_dilution` metric, and `subdivide_minority_families`. Its module docstring records a prior measurement: *at a 2% English minority, a Chinese-monolingual encoder put 97% of English queries into a single 100%-English cluster*. My work below **reproduces that finding independently** and then **overturns one of that module's stated conclusions**. Do not treat this as a greenfield design.

---

## 1. The failure mode: what a Chinese-monolingual encoder does to out-of-language input

### 1a. Tokenizer coverage — and a live configuration bug

`bge-*-zh-v1.5` uses a **BERT-Chinese WordPiece vocabulary of 21,128 tokens** (verified from `config.json`: `vocab_size=21128` for `bge-large-zh-v1.5`). By contrast every multilingual encoder here uses ~250,000 (XLM-R sentencepiece). Composition of the bge-zh vocab, counted directly:

| class | count | share |
|---|---|---|
| Han | 14,642 | 69.3% |
| Latin (whole-word 2,084 + `##` continuation 1,281) | 3,365 | 15.9% |
| digits | 1,015 | 4.8% |
| other Unicode | 1,631 | 7.7% |
| other ASCII | 475 | 2.2% |

So Latin coverage is not zero — but it is a ~2k-word English vocabulary, and the whole-word entries are visibly scraped web junk (`facebooktwitterpinterestgoogle`, `ubuntuforumwikilinuxpastechat`, `pixnetfacebookyahoo`). English therefore over-fragments:

```
'how to reset router password' -> ['how','to','re','##set','ro','##ute','##r','pass','##word']   (9 tokens for 5 words)
'cheap flights to tokyo'       -> ['ch','##ea','##p','f','##light','##s','to','tokyo']
```

Measured mean tokens/query on matched 10-query sets (2026-08-18):

| tokenizer | vocab | mean tok EN | mean tok ZH | EN/ZH |
|---|---|---|---|---|
| `bge-base-zh-v1.5` | 21,128 | **6.7** | 6.1 | 1.10 |
| `multilingual-e5-small` | 250,002 | **4.6** | 5.4 | 0.85 |

bge-zh spends **46% more tokens** on the same English than e5 does.

**The bug — and it is live in your bake-off.** WordPiece marks the *entire word* `[UNK]` if any sub-piece is missing. The bge-zh vocab contains only **7 tokens with any uppercase character at all**. Whether that matters depends on `do_lower_case`, and **the three bge-zh sizes disagree** (read from the cached `tokenizer_config.json`, which matches upstream `lastModified`):

| model | `do_lower_case` | capitalised-token test |
|---|---|---|
| `BAAI/bge-small-zh-v1.5` | **`false`** | **8/8 destroyed** |
| `BAAI/bge-base-zh-v1.5` | `true` | 0/8 |
| `BAAI/bge-large-zh-v1.5` | `true` | 0/8 |

Under `bge-small-zh-v1.5`:

```
'iPhone 15 Pro'      -> ['[UNK]', '15', '[UNK]']
'Nike 运动鞋'         -> ['[UNK]', '运', '动', '鞋']
'MacBook Air 保护套'  -> ['[UNK]', '[UNK]', '保', '护', '套']
'WiFi 密码'           -> ['[UNK]', '密', '码']
'B站'                 -> ['[UNK]', '站']
'GPT'                 -> ['[UNK]']
```

Every brand name, every product model, every acronym — the exact tokens that carry commercial intent — is **erased before the encoder sees it**. `'iPhone 保护壳'` and `'MacBook 保护壳'` become *literally the same input*. Note this is not a foreign-language issue: `B站`, `WiFi 密码`, `PS5 手柄` are **Chinese queries**. If your P3 bake-off ever selects `small` (it is the cheapest and fastest, so it plausibly wins on a speed-weighted score), you silently lose all mixed-script content. **Action: pass `do_lower_case=True` explicitly wherever a bge-zh tokenizer is constructed, and assert it.** This also defeats `classify_row`, which will correctly label `iPhone 保护壳` as `mixed:han+latin` while the encoder has already thrown the Latin half away.

### 1b. Embedding geometry — English does collapse

48 queries, 4 intents × {zh, en} × 6, balanced. Measured 2026-08-18:

| | `bge-small-zh` | `bge-base-zh` | `multilingual-e5-small` |
|---|---|---|---|
| mean cos zh–zh | 0.279 | 0.268 | 0.820 |
| mean cos **en–en** | **0.470** | **0.352** | 0.784 |
| en–en *minus* zh–zh | **+0.191** | **+0.085** | −0.036 |
| intent separation, zh | +0.221 | +0.200 | +0.057 |
| intent separation, **en** | **+0.074** | **+0.101** | +0.066 |
| translation top-1 (zh→en) | 46% | 58% | **92%** |

Read the third and fifth rows together: **English packs into a tighter ball, and inside that ball intent separation drops to about a third (small) or half (base) of the Chinese value.** That is precisely the hypothesised degenerate region. Confirmed at the clustering level — k=4 on `bge-small-zh` produces:

```
c0  n=18  {price:2, howto:6, fix:4, meaning:6}   langs={en:18}   <-- all four intents, one junk cluster
c3  n= 6  {meaning:6}                            langs={zh:6}
```

and k=2 gives **ARI vs language = 0.837** — the single dominant split in the data is "is this English".

### 1c. Anisotropy is a real confound, and it is the trap in your existing module

`multilingual-e5-small` has mean |cos| over all pairs of **0.776** vs **0.265** for `bge-base-zh`. Its *absolute* same-intent cosine (0.836) looks wonderful and its *separation* (+0.066) looks terrible. This is exactly the pattern `language.py`'s docstring already warns about (it recorded 0.87/+0.11 for multilingual vs 0.50/+0.26 for monolingual) and on that basis concluded the multilingual model was *"the worse representation."*

**That conclusion is wrong, and §3 shows why.**

---

## 2. Multilingual encoder options, as of 2026-08-18

Params/dims/vocab/context read directly from each repo's `config.json`; licence and `lastModified` from the HF API, today.

| Model (exact HF id) | Params | Dim | Max ctx | Vocab | Licence | Last mod | Notes |
|---|---|---|---|---|---|---|---|
| `Qwen/Qwen3-Embedding-0.6B` | 595,776,512 | 1024 | 32,768 | 151,669 | **apache-2.0** | 2026-04-20 | MRL; instruction-aware |
| `Qwen/Qwen3-Embedding-4B` | 4,021,774,336 | 2560 | 40,960 | 151,665 | apache-2.0 | 2025-06-20 | |
| `Qwen/Qwen3-Embedding-8B` | 7,567,295,488 | 4096 | 40,960 | 151,665 | apache-2.0 | 2025-07-07 | MTEB-multilingual #1, 70.58 (2025-06-05) |
| `BAAI/bge-m3` | (n/a) | 1024 | 8,194 | 250,002 | **mit** | 2024-07-03 | dense+sparse+ColBERT; 100+ langs; MIRACL dense 67.8 / all 70.0; MKQA dense 75.1 |
| `intfloat/multilingual-e5-small` | 117,654,272 | **384** | 512 | 250,037 | mit | 2026-04-02 | 94 langs; needs `query:`/`passage:` prefix |
| `intfloat/multilingual-e5-base` | 278,044,162 | 768 | 512 | 250,002 | mit | 2026-04-02 | 94 langs |
| `intfloat/multilingual-e5-large` | 559,890,946 | 1024 | 512 | 250,002 | mit | 2026-04-02 | 94 langs |
| `intfloat/multilingual-e5-large-instruct` | 559,890,432 | 1024 | 514 | 250,002 | mit | 2025-07-10 | instruction-tuned |
| `Alibaba-NLP/gte-multilingual-base` | 305,369,089 | 768 | **8,192** | 250,048 | apache-2.0 | 2025-07-05 | 75 langs; strong size/quality point |
| `jinaai/jina-embeddings-v3` | 572,310,396 | 1024 | 8,194 | 250,002 | **cc-by-nc-4.0** | 2026-04-08 | 5 task-LoRAs. **Non-commercial** |
| `jinaai/jina-embeddings-v4` | 3,754,885,248 | — | — | — | **Qwen Research Licence** | 2026-04-08 | multimodal; check licence |
| `sentence-transformers/LaBSE` | 470,927,360 | 768 | 512 | 501,153 | apache-2.0 | 2025-03-06 | **translation-mining objective — avoid, see §3** |
| `BAAI/bge-large-zh-v1.5` (incumbent) | — | 1024 | 512 | **21,128** | mit | 2024-04-02 | Chinese-monolingual |

Commercial APIs (prices from aggregators, **not** vendor pages — verify before quoting):

| Model | Dim | Price /1M tok | Notes |
|---|---|---|---|
| `text-embedding-3-large` (OpenAI) | 3072 (MRL) | ~$0.13 | |
| `gemini-embedding-001` (Google) | 3072 default, MRL 768/1536/3072 | ~$0.15 (batch ~$0.075) | reportedly +2pp on multilingual MTEB vs OpenAI |
| `embed-v4.0` (Cohere) | 256/512/1024/1536 | ~$0.12 | 128k ctx, 100+ langs, Matryoshka |
| `voyage-3-large` | — | ~$0.18 | |

**Strong on both Chinese and English:** `Qwen3-Embedding-*` (Apache-2.0, CMTEB + MTEB-multi), `BGE-M3` (MIT, built by BAAI for exactly this), `gte-multilingual-base` (best quality-per-parameter), `multilingual-e5-*`. **Exclude `jina-v3` (CC-BY-NC)** for any commercial deployment. **Exclude LaBSE** on task grounds.

---

## 3. Cross-lingual alignment: unified vs stratified — and the finding that changes the answer

### The naive framing is a trap

The question "do we want 苹果价格 and apple price in the same cluster?" assumes swapping to a multilingual encoder is what merges them. **It is not.** Measured at realistic imbalance — 3,968 synthetic queries, 8 intents × 8 topics, **92% zh / 8% en**, k=16, 2026-08-18:

| representation | ARI-noun | ARI-lang | **EN concentration** | mixed-lang clusters |
|---|---|---|---|---|
| `bge-base-zh`, raw | 0.627 | 0.025 | **0.91** | 6/16 |
| `bge-base-zh`, global-centred | 0.609 | 0.024 | 0.95 | 4/16 |
| `bge-base-zh`, **per-language-centred** | 0.671 | 0.007 | **0.25** | 12/16 |
| `multilingual-e5-small`, raw | 0.645 | 0.028 | **1.00** | **0/16** |
| `multilingual-e5-small`, global-centred | 0.657 | 0.028 | 1.00 | 0/16 |
| `multilingual-e5-small`, **per-language-centred** | 0.722 | −0.001 | **0.12** | **16/16** |

*EN concentration* is `minority_dilution`'s metric: share of English rows in the single cluster holding most of them. Two results matter enormously:

1. **Swapping to a multilingual encoder made the collapse WORSE, not better.** `e5` raw put **100% of English into one cluster (0/16 mixed)** — strictly worse than the monolingual model's 91%. This independently confirms `language.py`'s warning *"Do NOT simply hope a multilingual encoder fixes it"* and refutes the assignment's implicit premise that a multilingual encoder is the fix.

2. **Per-language mean-centring is the fix**, and only it: 1.00 → **0.12**, 0/16 → **16/16** mixed clusters. Global centring does nothing (1.00 → 1.00). The two interventions are complementary — the multilingual encoder is only *better* than monolingual once centred (0.12 vs 0.25).

### Why: language is one dominant near-linear direction

PCA on the balanced probe, correlating each PC with the language label:

| | `bge-base-zh` | `multilingual-e5-small` |
|---|---|---|
| PC1 | 10.8% var, r=−0.128 | **14.3% var, r=+0.922** |
| PC2 | **9.4% var, r=+0.883** | 13.0% var, r=−0.324 |

Language is a single axis worth 9–14% of variance, and k-means finds it first. Removing it is three lines. On the balanced probe, intent recovery:

| | raw | global-centred | per-lang-centred |
|---|---|---|---|
| `bge-base-zh` | 0.391 | 0.450 | **0.783** |
| `multilingual-e5-small` | 0.364 | 0.943 | **1.000** |

**This is the load-bearing correction to `language.py`.** Its `alignment_probe` measures separation on *raw* embeddings, where the anisotropic multilingual model scores badly (+0.066 vs +0.200) — so the probe as written will keep **rejecting the better encoder**. The probe must centre before measuring separation, or it inverts its own decision.

### So: unified or stratified?

**Unified space, plus per-language centring, plus a language-aware audit.** Rationale:

- **(a) pure unified** is what you get today: language becomes the top-level split and the tree spends a family on "foreign", resolving no intent.
- **(b) fully stratified trees** are defensible at ≥5% minority but cost you the cross-language intent comparison the business actually wants ("is `退款` demand the same shape as `refund` demand?"), and at 8% English you'd be inducing a taxonomy from ~320 rows — below the point where your own annotation and stability machinery is trustworthy.
- **(c) unified + centring** merges by intent *and* keeps the language label as an attribute you can slice by, which is strictly more informative than either.

Do we *want* 苹果价格 ≈ apple price merged? **Yes for the taxonomy, no for the reporting.** Merge in the clustering space so that low-volume English intents inherit the taxonomy induced from 3,600 Chinese rows instead of forming a junk bucket; then report volume split by `row_language`. This is why **LaBSE is wrong for this task**: it was trained for bitext mining, so it maximises translation-pair similarity at the cost of *within-language semantic* discrimination — the opposite of the trade you want.

---

## 4. Language detection for short text

Benchmarked on this machine, 2026-08-18, on 41–43 realistic short queries (Chinese ~4–7 chars, English ~2–4 words), including mixed-script and non-linguistic rows. Two independent runs:

| library | version | overall | **zh** | en | speed | install |
|---|---|---|---|---|---|---|
| **lingua (restricted to 9 langs)** | 2.2.0 | **95–98%** | **17/17** | 11–12/14 | 0.016–0.108 ms/q | pure Py, ~90MB |
| lingua (all 75 langs) | 2.2.0 | 86% | 17/17 | 8/14 | 0.378 ms/q | same |
| py3langid | 0.3.0 | 84–88% | 17/17 | 10/12 | 0.021–0.026 ms/q | tiny, BSD |
| fastText `lid.176` | via `fasttext-langdetect` 1.1.1 | 85–86% | **13/17** | 10–12/14 | **0.003 ms/q** | ~130MB bin, MIT |

**`lingua` restricted to a candidate set wins decisively, and the restriction is the whole trick** — narrowing from 75 languages to 9 gained **+9pp accuracy and a 24× speedup** simultaneously (0.378 → 0.016 ms/q). Unrestricted lingua guesses Esperanto/Xhosa/Tswana on short English.

**Disqualifying finding for fastText:** `lid.176` misclassified **4/17 Chinese queries**, mostly as Japanese — `'空调不制冷'→ja`, `'PS5 手柄漂移'→ja`, `'2026 高考'→ja`, `'氢怎么读'→sr`. In a 90%-Chinese corpus, an error mode concentrated on the *dominant* language is unacceptable regardless of speed.

**Every detector failed on non-linguistic rows** (`ipad`, `gpt4`, `nihao`) — which validates `classify_row`'s existing `nonlinguistic` class. **Keep script detection as the primary router and use statistical LID only to disambiguate within Latin script.** GlotLID (2,100+ labels, ~3,127 samples/s) is aimed at low-resource coverage you do not need; cld3/pycld3 is effectively unmaintained for new Python.

```python
# pip install lingua-language-detector
from functools import lru_cache
from lingua import Language, LanguageDetectorBuilder

_CANDIDATES = [Language.CHINESE, Language.ENGLISH, Language.JAPANESE,
               Language.KOREAN, Language.RUSSIAN, Language.SPANISH,
               Language.FRENCH, Language.VIETNAMESE, Language.THAI]

_DET = (LanguageDetectorBuilder.from_languages(*_CANDIDATES)
        .with_preloaded_language_models()   # pay load cost once, at import
        .build())

@lru_cache(maxsize=200_000)                 # query logs are heavily duplicated
def detect_language(text: str) -> str:
    """Script first (cheap, near-certain); statistical LID only inside Latin."""
    from qmine.ops.language import classify_row
    script = classify_row(text)
    if script in ("nonlinguistic",) or script.startswith("mixed"):
        return script                        # do NOT force a language on these
    if script != "latin":
        return script                        # han/kana/hangul/cyrillic/thai: script IS the answer
    r = _DET.detect_language_of(text)
    return r.iso_code_639_1.name.lower() if r else "unknown"
```

---

## 5. Script and segment handling, per row

`language.py` already has `char_ngram_for(script)` returning `(1,3)` for Han and `(3,5)` for Latin, but `subdivide_minority_families` applies **one range to a whole family** based on its majority script. For `iPhone 保护壳` that is wrong under either choice: `(1,3)` over-fragments the Latin run, `(3,5)` destroys the Han morphemes.

**Segment the row, vectorise each run under its own range, then concatenate.** This makes the range a property of the *span*, not the corpus or the family:

```python
import re
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer

_HAN = re.compile(r'[一-鿿㐀-䶿]+')
_LAT = re.compile(r'[A-Za-z][A-Za-z0-9\-\']*')

def split_runs(text: str) -> tuple[str, str]:
    """Return (han_text, latin_text) so each is vectorised under its own n-gram range."""
    return " ".join(_HAN.findall(text)), " ".join(w.lower() for w in _LAT.findall(text))

def mixed_script_features(texts, seed=0):
    han = [split_runs(t)[0] for t in texts]
    lat = [split_runs(t)[1] for t in texts]
    v_han = TfidfVectorizer(analyzer="char", ngram_range=(1, 3), min_df=2, sublinear_tf=True)
    v_lat = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)
    blocks, used = [], []
    if any(han): blocks.append(v_han.fit_transform(han)); used.append("han(1,3)")
    if any(lat): blocks.append(v_lat.fit_transform(lat)); used.append("latin(3,5)")
    return hstack(blocks).tocsr(), used
```

Three details that matter:
- Use **`char_wb`** for the Latin block so n-grams do not straddle word boundaries; plain `char` is right for Han, where there are no boundaries to respect.
- **Lower-case only the Latin run.** `Nike`/`nike` must unify; Han has no case.
- **Pinyin is not English.** `钦州的拼音`, `qinzhou pinyin`, `nihao` are romanised Chinese; lingua called `nihao` Swahili/Vietnamese. Because `(3,5)` char n-grams on the Latin run capture pinyin syllable structure directly, romanised Chinese clusters with itself without needing a correct language label — a good reason to keep the char-n-gram block rather than relying on the dense encoder alone. Do **not** route pinyin to an English sub-tree.
- **Code-switching**: keep `classify_row`'s `mixed:*` labels as first-class values in `row_language`. Do **not** collapse them to the majority script — a `mixed:han+latin` row centred by the `zh` mean is being corrected for the wrong offset. Give `mixed:*` its own centring group when it has ≥30 rows.

---

## 6. Practical recipe for 90% zh / 8% en / 2% other

**Pipeline**

1. **P1 — profile.** `profile_corpus()` already returns `posture`. At 8% English this is `genuinely_multilingual` (≥5%), so the data decides; at 2% it would be `minority_at_risk`.
2. **P1 — label rows.** Script first, `lingua`-9 inside Latin, `nonlinguistic` preserved. Persist `row_language`.
3. **P3 — fix the tokenizer bug first.** Force `do_lower_case=True` for any bge-zh tokenizer and assert `tok("iPhone")` contains no `[UNK]`. Do this *before* the bake-off, or the bake-off compares a corrupted candidate.
4. **P3 — bake off 3 encoders** on your corpus: `bge-large-zh-v1.5` (incumbent), `Alibaba-NLP/gte-multilingual-base` (768d, 8k ctx, Apache-2.0 — best size/quality), `Qwen/Qwen3-Embedding-0.6B` (1024d, Apache-2.0, MRL). Add `BAAI/bge-m3` if you want sparse+dense in one model.
5. **P3 — centre per language, then normalise.** This is the change that matters most:

```python
import numpy as np

def language_centered(X, row_language, min_rows=30):
    """Remove the per-language mean, then re-normalise to the unit sphere.

    Language is a single dominant axis (measured: 9-14% of variance, |r| up to
    0.92 with the language label). k-means finds it before it finds intent, so a
    minority language becomes one junk family. Groups too small to estimate their
    own mean fall back to the global mean rather than a noisy one.
    """
    X = np.asarray(X, dtype=np.float64)
    lang = np.asarray(list(row_language))
    out = X.copy()
    global_mu = X.mean(0)
    for lg in np.unique(lang):
        m = lang == lg
        out[m] = X[m] - (X[m].mean(0) if m.sum() >= min_rows else global_mu)
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)
```

   `min_rows=30` is the guard that matters: 2% of a 600-row log is 12 rows, and a mean estimated from 12 points in 1024-D adds more noise than it removes.

6. **P4 — cluster on the centred matrix.** Keep `row_language` as an attribute for reporting, never as a feature.
7. **P5+ — keep `subdivide_minority_families` as the safety net**, upgraded with §5's per-run vectoriser.

**Decision rule (stratify vs unify)**

Run both arms; decide on measurements you already compute, in this order:

- **Unify (default)** when, after centring, `minority_dilution(...)["concentration"] < 0.5` for every minority language **and** replay-stability ARI is within 0.03 of the dominant-only baseline. At 90/8/2 with centring I measured concentration **0.12**, so this branch should win.
- **Stratify** when concentration stays **> 0.8** after centring (the space genuinely cannot resolve that language), **or** the minority exceeds ~15% (it can support its own taxonomy), **or** template fragmentation for the unified arm is worse than stratified by >0.15.
- **Neither — subdivide locally** in the `minority_at_risk` band (0.5–5%): too few rows for its own tree, too many to lose. That is what `subdivide_minority_families` is for.

**How to measure which is better.** Use the project's decisive metrics, not silhouette:

| metric | where | direction | what it catches |
|---|---|---|---|
| `replay_stability(X, k, seeds=(0,1))` | `ops/cluster.py:89` | higher | is the partition real at all |
| `template_fragmentation(labels, masks)` | `ops/templates.py:239` | **lower** (1.0 perfect) | did one intent stay together |
| `minority_dilution(labels, row_language, "en")` | `ops/language.py:201` | concentration **lower** | the junk-foreign-cluster failure |
| `alignment_probe(...)` **run on centred vectors** | `ops/language.py:162` | separation + rank1 higher | encoder choice |

Compute all four per arm. Two cautions from my own runs: `template_fragmentation` is negatively correlated with cluster count (`ops/panel.py:15`), so **compare arms only at equal k**; and centring changes the geometry, so **re-select k after centring** rather than reusing the incumbent's.

**Honest caveat on my numbers.** The 3,968-query corpus is *synthetic and templated*. Its clusters track topic (ARI-noun 0.63–0.72), not intent (ARI-intent ≈0.05) — my generator crossed 8 intents with 8 topics and k-means found the topic axis, which is your own "template twin" phenomenon rather than a fact about encoders. I therefore rest the §3 conclusion on **EN-concentration and mixed-cluster count** (unambiguous, large effect, stable across 3 seeds), not on ARI-intent. The perfect ARI-intent=1.000 figure comes from the clean 48-query balanced probe and is small-n. **Re-run both on your real log before committing** — which is exactly what `language.py` already insists on, and it is right to.


---

## Recommendations

- FIX FIRST — a live tokenizer bug: `BAAI/bge-small-zh-v1.5` ships `do_lower_case: false` while `base` and `large` ship `true`. Its vocab has only 7 tokens containing any uppercase letter, so WordPiece maps any capitalised word entirely to `[UNK]`. Verified 8/8: 'iPhone 15 Pro'→['[UNK]','15','[UNK]'], 'Nike 运动鞋'→['[UNK]','运','动','鞋'], 'B站'→['[UNK]','站'], 'WiFi 密码'→['[UNK]','密','码']. These are Chinese queries, not foreign ones. Force `do_lower_case=True` explicitly and assert on it BEFORE the P3 bake-off, or a corrupted candidate is being scored.
- Do NOT expect a multilingual encoder to fix the junk-foreign-cluster problem — measured at 92% zh / 8% en (n=3,968), `multilingual-e5-small` raw put 100% of English into ONE cluster (0/16 mixed-language clusters), strictly WORSE than `bge-base-zh` raw at 91%. This independently confirms the warning already written in `src/qmine/ops/language.py`.
- The actual fix is per-language mean-centring, and it is ~10 lines. Language is one dominant near-linear axis (9-14% of variance, |r| up to 0.92 with the language label), so k-means finds it before intent. Per-language centring drove EN-concentration from 1.00 to 0.12 (0/16 to 16/16 mixed clusters) on e5, and 0.91 to 0.25 on bge-base. Global centring does nothing (1.00 to 1.00) — it must be per-language. Use `min_rows=30` fallback to the global mean, since 2% of a 600-row log is only 12 rows.
- CORRECT `alignment_probe` in `src/qmine/ops/language.py:162` — it measures separation on RAW embeddings, where anisotropy penalises the multilingual model (+0.066 vs +0.200 for monolingual; mean |cos| 0.776 vs 0.265). The module's docstring concludes from this that multilingual is 'the worse representation'. That inverts once you centre: intent ARI goes bge 0.391→0.783 but e5 0.364→1.000. As written the probe will keep rejecting the better encoder. Centre before measuring separation.
- Adopt unified-space + per-language centring, not stratified trees, at 90/8/2 — merge in the clustering space so low-volume English intents inherit the taxonomy induced from the Chinese majority, then report volume sliced by `row_language`. Stratify only if concentration stays >0.8 after centring, or the minority exceeds ~15%.
- Use `lingua` restricted to an explicit ~9-language candidate set for language ID: 95-98% on short queries, 17/17 on Chinese, 0.016 ms/query. Restricting the candidate set gained +9pp accuracy AND a 24x speedup versus all-75-languages. Reject fastText lid.176 despite being 5x faster — it misclassified 4/17 Chinese queries, mostly as Japanese ('空调不制冷'→ja, '2026 高考'→ja); an error mode concentrated on the dominant language is disqualifying here.
- Keep script detection as the primary router and use statistical LID only inside Latin script. Every detector tested failed on non-linguistic rows ('ipad', 'gpt4', 'nihao'), which validates the existing `nonlinguistic` class in `classify_row`. Treat pinyin as romanised Chinese, not English — (3,5) char n-grams cluster it correctly without a language label.
- Encoder shortlist for the bake-off, all verified 2026-08-18: `Alibaba-NLP/gte-multilingual-base` (305M, 768d, 8192 ctx, apache-2.0 — best quality per parameter), `Qwen/Qwen3-Embedding-0.6B` (596M, 1024d, 32768 ctx, apache-2.0, MRL), `BAAI/bge-m3` (1024d, 8194 ctx, MIT, dense+sparse+ColBERT). EXCLUDE `jinaai/jina-embeddings-v3` — cc-by-nc-4.0, non-commercial. EXCLUDE LaBSE — its bitext-mining objective maximises translation similarity at the cost of within-language discrimination, the opposite of what query mining needs.
- Upgrade `subdivide_minority_families` to segment mixed-script rows per run rather than per family: vectorise Han spans at char (1,3) and Latin spans at char_wb (3,5), then concatenate. A single range applied to 'iPhone 保护壳' is wrong either way. Lower-case only the Latin run, and give `mixed:*` rows their own centring group when they reach 30+ rows.
- Compare arms at EQUAL k using replay-stability ARI, template fragmentation, and `minority_dilution` concentration — never silhouette. Template fragmentation is negatively correlated with cluster count (`ops/panel.py:15`), so unequal-k comparisons are invalid, and centring changes the geometry enough that k must be re-selected rather than inherited from the incumbent.

## Unverified

- Commercial API prices (OpenAI ~$0.13/M, Cohere embed-v4.0 ~$0.12/M, gemini-embedding-001 ~$0.15/M, voyage-3-large ~$0.18/M) come from third-party aggregator sites via web search, NOT from vendor pricing pages — WebFetch to huggingface.co, github.com and vendor domains is blocked by network policy in this environment. Treat these as indicative and re-verify against vendor docs before any budget decision.
- The 3,968-query corpus in §3 is SYNTHETIC and templated. Its clusters track topic (ARI-noun 0.63-0.72) rather than intent (ARI-intent ~0.05), because the generator crossed 8 intents with 8 topics and k-means found the topic axis. I therefore rest the conclusion on EN-concentration and mixed-cluster counts (large, unambiguous, stable across 3 seeds) and NOT on ARI-intent.
- The headline 'intent ARI 0.364 -> 1.000 after centring' comes from a 48-query balanced probe. That is small-n and the perfect 1.000 should be read as 'the language axis was the whole problem on this probe', not as an expected production number.
- My probes are balanced or synthetically imbalanced; they are not your real log. The magnitude of the centring benefit will differ on real data, particularly where 'English' rows are actually brand names, model numbers, or pinyin rather than English sentences. Re-run both arms on the real corpus, as `language.py` already insists.
- I did not benchmark the recommended encoders (gte-multilingual-base, Qwen3-Embedding-0.6B, BGE-M3) on this machine — only bge-{small,base}-zh-v1.5 and multilingual-e5-small were cached or downloadable within the session. The multilingual findings are demonstrated on e5-small and are expected but NOT verified to transfer to the larger multilingual models.
- CMTEB and MTEB-multilingual leaderboard scores in §2 are quoted from search results and blog aggregators with varying dates (the Qwen3-8B 70.58 figure is dated 2025-06-05). MTEB rankings shift weekly; I could not reach the live leaderboard to confirm current standings.
- `BAAI/bge-m3` and several other repos return no `safetensors` param count via the HF API (weights stored as pytorch_model.bin), so parameter counts are missing rather than verified for those rows.
- The `do_lower_case` values were read from the locally cached `tokenizer_config.json` snapshots. They match upstream `lastModified` timestamps from the HF API today, but I could not diff against the live file directly; `bge-small-zh-v1.5` also has a `refs/pr/1` in the cache, so confirm you are pinning the intended revision.
- `jinaai/jina-embeddings-v4` returned no licence field via the HF API; the 'Qwen Research Licence' attribution comes from search results and needs direct confirmation before any use.
- The lingua timing varied between two runs (0.016 vs 0.108 ms/query) depending on whether language models were preloaded; the accuracy figures (95% vs 98%) differ because the two runs used slightly different test-set sizes (43 vs 41 items) after I trimmed non-linguistic probes.

## Sources

- Local empirical measurement, 2026-08-18: tokenizer vocabulary composition, do_lower_case comparison across bge-{small,base,large}-zh-v1.5, and UNK behaviour — run against models cached in /Users/mayouxuan/Documents/Claude/Search Query Mining Agent Team/QMine/.hf
- Local empirical measurement, 2026-08-18: embedding geometry (anisotropy, intra/inter-language cosine, intent separation, translation top-1 retrieval), PCA language-axis correlation, and per-language centring ablation across bge-small-zh-v1.5, bge-base-zh-v1.5, multilingual-e5-small
- Local empirical measurement, 2026-08-18: language-ID benchmark (lingua 2.2.0 all-75 and 9-language subset, py3langid 0.3.0, fastText lid.176 via fasttext-langdetect 1.1.1) on 41-43 short queries
- Hugging Face API model metadata and config.json, fetched live 2026-08-18 via huggingface_hub HfApi for Qwen3-Embedding-{0.6B,4B,8B}, BAAI/bge-m3, intfloat/multilingual-e5-{small,base,large,large-instruct}, Alibaba-NLP/gte-multilingual-base, sentence-transformers/LaBSE, jinaai/jina-embeddings-{v3,v4}, BAAI/bge-{small,base,large}-zh-v1.5
- Existing QMine source: /Users/mayouxuan/Documents/Claude/Search Query Mining Agent Team/QMine/src/qmine/ops/language.py (script_profile, classify_row, profile_corpus, alignment_probe, minority_dilution, char_ngram_for, subdivide_minority_families)
- Existing QMine source: src/qmine/ops/cluster.py:89 (replay_stability), src/qmine/ops/templates.py:239 (template_fragmentation), src/qmine/ops/panel.py:15, src/qmine/graph/nodes/foundation.py:139-217, src/qmine/graph/nodes/bottomup.py:286-309
- https://huggingface.co/BAAI/bge-large-zh-v1.5
- https://huggingface.co/BAAI/bge-m3
- https://huggingface.co/Qwen/Qwen3-Embedding-8B
- https://qwenlm.github.io/blog/qwen3-embedding/
- https://github.com/QwenLM/Qwen3-Embedding
- https://arxiv.org/pdf/2506.05176
- https://arxiv.org/html/2402.03216v3
- https://arxiv.org/abs/2409.10173
- https://jina.ai/news/jina-embeddings-v3-a-frontier-multilingual-embedding-model/
- https://jina.ai/models/jina-embeddings-v4/
- https://arxiv.org/pdf/2310.16248
- https://arxiv.org/pdf/2602.13139
- https://arxiv.org/pdf/2601.18026
- https://arxiv.org/pdf/2509.17768
- https://www.edenai.co/post/top-free-language-detection-tools-apis-and-open-source-models
- https://docs.cohere.com/changelog/embed-multimodal-v4
- https://ai.google.dev/gemini-api/docs/embeddings
- https://developers.googleblog.com/gemini-embedding-available-gemini-api/
- https://pecollective.com/tools/text-embedding-models-compared/
- https://embeddingcost.com/