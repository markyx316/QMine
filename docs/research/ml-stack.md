# Clustering & Embedding Stack (2026)

> Dossier gathered 2026-08-17 for the QMine build. Signatures marked *verified* were
> introspected from installed packages; everything else is from live documentation.

# Python ML Stack for Query Intent Mining & Clustering — Verified as of 2026-08-17

All version numbers below were pulled live from the PyPI JSON API on 2026-08-17. All API signatures were extracted from the current published docs (scikit-learn 1.9.0 stable) or directly from library source on GitHub master. Nothing here is from memory.

---

## 0. Pinned stack (verified latest on PyPI, 2026-08-17)

| Package | Latest | `requires_python` | Notes |
|---|---|---|---|
| `scikit-learn` | **1.9.0** (released June 2026) | `>=3.11` | deps: `numpy>=1.24.1, scipy>=1.10.0, joblib>=1.4.0, narwhals>=2.0.1, threadpoolctl>=3.5.0` |
| `numpy` | **2.5.2** | `>=3.12` | |
| `scipy` | **1.18.0** | `>=3.12` | |
| `pandas` | **3.0.5** | `>=3.11` | pandas 3 uses `StringDtype` not `object` — see gotcha below |
| `sentence-transformers` | **5.7.0** (2026-08-06) | `>=3.10` | deps `transformers<6.0.0,>=4.41.0`, `torch>=1.11.0` |
| `transformers` | **5.15.0** | `>=3.10` | v5 is the current major |
| `torch` | **2.13.0** | `>=3.10` | |
| `umap-learn` | **0.5.12** (2026-04-08) | `>=3.9` | deps `scikit-learn>=1.6`, `numba>=0.51.2`, `pynndescent>=0.5` |
| `numba` | **0.67.0** | `>=3.10` | **`numpy<2.6,>=1.22`** ← the binding constraint on the whole stack |
| `hdbscan` (contrib pkg) | **0.8.44** | `>=3.10` | deps `numpy<3,>=1.20`, `scikit-learn>=1.6` |
| `faiss-cpu` | **1.15.0** | `>=3.10` | `cp310-abi3-macosx_14_0_arm64.whl` exists → Apple Silicon wheel, macOS 14+ |
| `hnswlib` | **0.8.0** | — | **sdist only, no wheels** → needs a C++ toolchain. Prefer faiss. |
| `jieba` | **0.42.1** | — | last upload **2020-01-20**. Effectively unmaintained. |
| `pkuseg` | 0.0.25 | — | last upload **2020-06-27**. Dead. Use `spacy-pkuseg` (1.0.1, 2025-07-14) instead. |
| `hanlp` | 2.1.3 (2025-10-19) | `>=3.6` | actively maintained |
| `ltp` | 4.2.14 (2024-06-08) | `>=3.6,<4` | |
| `threadpoolctl` | 3.6.0 | | |
| `joblib` | 1.5.3 | | |
| `matplotlib` | 3.11.1 | `>=3.11` | |
| `pyarrow` | 25.0.1 | `>=3.10` | |

**Recommended Python: 3.13.** `scikit-learn>=1.9` needs ≥3.11, `numpy>=2.5`/`scipy>=1.18` need ≥3.12, `numba 0.67` ships `cp312/cp313/cp314-macosx_12_0_arm64` wheels, `hdbscan 0.8.44` ships `cp313-macosx_10_13_universal2`. 3.13 is the sweet spot; 3.14 also works for numba/hdbscan/faiss but check torch.

**Hard pin `numpy<2.6`** in `constraints.txt`. `numba 0.67.0` declares `numpy<2.6`; the day numpy 2.6 lands, `pip install umap-learn` will either downgrade numpy under you or fail resolution. This is the #1 reproducibility landmine in this stack.

```
# constraints.txt (P0)
numpy>=2.4,<2.6
scikit-learn==1.9.0
scipy==1.18.0
sentence-transformers==5.7.0
transformers==5.15.0
torch==2.13.0
umap-learn==0.5.12
numba==0.67.0
hdbscan==0.8.44
faiss-cpu==1.15.0
threadpoolctl==3.6.0
joblib==1.5.3
pandas==3.0.5
```

---

## 1. Embeddings (P3) — sentence-transformers 5.7.0

### 1.1 `encode()` — exact current signature (from `sentence_transformers/sentence_transformer/model.py` master)

```python
def encode(
    self,
    inputs: Sequence[SingleInput] | SingleInput,   # ← RENAMED from `sentences` in v5.4
    prompt_name: str | None = None,
    prompt: str | None = None,
    batch_size: int = 32,
    show_progress_bar: bool | None = None,
    output_value: Literal["sentence_embedding", "token_embeddings"] | None = "sentence_embedding",
    precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] = "float32",
    convert_to_numpy: bool = True,
    convert_to_tensor: bool = False,
    device: str | list[str | torch.device] | None = None,   # a LIST triggers multi-process
    normalize_embeddings: bool = False,
    truncate_dim: int | None = None,                        # Matryoshka truncation
    pool: dict | None = None,
    chunk_size: int | None = None,
    **kwargs,   # notably processing_kwargs={"text": {"max_length": 256, "truncation": True}}
) -> ...
```

### 1.2 v5.4+ renames you MUST adopt (old names still work but emit `DeprecationWarning`)

From the official migration guide (`docs/migration_guide.md`, section "Migrating from v5.x to v5.4+"):

| Old (v5.0–5.3) | New (v5.4+) |
|---|---|
| `model.encode(sentences=...)` | `model.encode(inputs=...)` |
| `model.get_sentence_embedding_dimension()` | `model.get_embedding_dimension()` |
| `model.truncate_sentence_embeddings(d)` | `model.truncate_embeddings(d)` |
| `SentenceTransformer(..., tokenizer_kwargs=...)` | `SentenceTransformer(..., processor_kwargs=...)` |
| `from sentence_transformers.losses import X` | `from sentence_transformers.sentence_transformer.losses import X` |
| `from sentence_transformers.models import X` | `from sentence_transformers.sentence_transformer.modules import X` |
| `from sentence_transformers.evaluation import X` | `from sentence_transformers.sentence_transformer.evaluation import X` |
| `from sentence_transformers.quantization import quantize_embeddings` | `from sentence_transformers.util.quantization import quantize_embeddings` |
| `from sentence_transformers.similarity_functions import SimilarityFunction` | `from sentence_transformers.util.similarity import SimilarityFunction` |
| `Trainer(tokenizer=...)` | `Trainer(processing_class=...)` |
| `CrossEncoder.max_length` | `CrossEncoder.max_seq_length` |
| `Pooling(word_embedding_dimension=)` | `Pooling(embedding_dimension=)` |

Top-level `from sentence_transformers import SentenceTransformer, CrossEncoder, SparseEncoder` is unchanged.

**Forward deprecation:** from v6.0, loading a model whose `modules.json` references a class outside `sentence_transformers` requires `trust_remote_code=True` — *including local directories*, which were previously implicitly trusted. v5.6.0 already emits a `FutureWarning`. Set this explicitly now for `bge-m3`/Qwen3 style custom-code models so the v6 bump is a no-op. The master docs already contain a "Migrating from v5.x to v6.x" section (v6 introduces `MultiVectorEncoder`, requires `transformers` v5, `torch>=2.2`, `huggingface-hub` v1), so plan for it but pin `==5.7.0` today.

### 1.3 Model bake-off shortlist with verified C-MTEB Clustering numbers

From the `Qwen/Qwen3-Embedding-0.6B` model card (paper: arXiv **2506.05176**):

**C-MTEB (Chinese) — Clustering column:**

| Model | Params | C-MTEB Clust. | C-MTEB Mean(Task) |
|---|---|---|---|
| multilingual-e5-large-instruct | 0.6B | 48.23 | 58.08 |
| gte-Qwen2-1.5B-instruct | 1.5B | 54.61 | 67.12 |
| gte-Qwen2-7B-instruct | 7.6B | 66.06 | 71.62 |
| ritrieve_zh_v1 | 0.3B | 66.50 | 72.71 |
| **Qwen3-Embedding-0.6B** | 0.6B | **68.74** | 66.33 |
| **Qwen3-Embedding-4B** | 4B | **77.89** | 72.27 |
| **Qwen3-Embedding-8B** | 8B | **80.08** | 73.84 |

**MTEB Multilingual — Clustering column:** BGE-M3 40.88, mE5-large-instruct 50.75, Gemini Embedding 54.59, Qwen3-0.6B 52.33, Qwen3-4B 57.15, **Qwen3-8B 57.65** (Mean(Task) 70.58, #1 as of 2025-06-05).

**Verdict for this playbook:** `Qwen3-Embedding-0.6B` is the default. It beats gte-Qwen2-7B (12x larger) on C-MTEB Clustering (68.74 vs 66.06) and is the only sub-1B model in that class. Spec: 28 layers, 32K context, **1024 dims with MRL support down to 32**, instruction-aware, last-token pooling, `padding_side="left"` recommended. `bge-base-zh-v1.5` (BertModel, 768 dims, `max_position_embeddings=512`, CLS pooling) stays in the bake-off as the fast/cheap baseline — it is 3-4 years old and no longer competitive on clustering, but it is 10x faster on CPU and its `config_sentence_transformers.json` has no `prompts` block, so `prompt_name="query"` will KeyError; bge-zh's asymmetric instruction (`为这个句子生成表示以用于检索相关文章：`) is for retrieval **queries only** and should NOT be applied for clustering/STS.

`bge-m3` (XLM-R backbone, 567M, 8192 ctx) is worth including for long/multilingual logs but scores only 40.88 on MTEB-multilingual clustering — it is a retrieval model, not a clustering model.

### 1.4 Instructions/prompts for clustering

Qwen3-Embedding is instruction-aware; the card states instructions typically give **+1% to +5%** and should be **written in English** even for Chinese input. For clustering, use a symmetric task instruction on *all* texts (not just "queries"):

```python
from sentence_transformers import SentenceTransformer

CLUSTER_PROMPT = "Instruct: Identify the search intent category of the given user query\nQuery:"

model = SentenceTransformer(
    "Qwen/Qwen3-Embedding-0.6B",
    processor_kwargs={"padding_side": "left"},     # v5.4+ name; was tokenizer_kwargs
    model_kwargs={"torch_dtype": "float32"},       # see MPS/CPU dtype warning below
    device="mps",
)
emb = model.encode(
    queries,
    prompt=CLUSTER_PROMPT,
    batch_size=64,
    normalize_embeddings=True,   # ALWAYS for clustering — see §5
    convert_to_numpy=True,
).astype("float32")
```

Critical anti-anchoring note for P7: the *same* instruction string must be used at index time and at deployment time, and it must be recorded in the run manifest. Changing the prompt silently invalidates every centroid.

### 1.5 Backends (`onnx` / `openvino`) and the CPU dtype trap

Install extras: `pip install sentence-transformers[onnx]` (CPU) / `[onnx-gpu]` / `[openvino]`. Note 5.7.0's onnx extra now resolves to **`optimum-onnx[onnxruntime]`** (optimum-onnx was split out of `optimum`).

```python
from sentence_transformers import (
    SentenceTransformer,
    export_optimized_onnx_model,
    export_dynamic_quantized_onnx_model,
)

model = SentenceTransformer("BAAI/bge-base-zh-v1.5", backend="onnx")
export_optimized_onnx_model(model, optimization_config="O3", model_name_or_path="artifacts/bge-zh")
export_dynamic_quantized_onnx_model(model, quantization_config="arm64",   # ← arm64 on Apple Silicon
                                    model_name_or_path="artifacts/bge-zh")
# then reload the artifact:
model = SentenceTransformer("artifacts/bge-zh", backend="onnx",
                            model_kwargs={"file_name": "onnx/model_qint8_arm64.onnx"})
```

`quantization_config` ∈ `{"arm64","avx2","avx512","avx512_vnni"}`. The docs state all four give "roughly equivalent speedups"; use `arm64` on Apple Silicon.

**Verified benchmark facts from the official efficiency doc (RTX 3090 / i7-13700K, median over batch sizes):**
- CPU: `onnx` 1.35x, `openvino` 1.24x on short text; `onnx-qint8` **3.23x**, `openvino-qint8` **5.29x**, at "<0.5%" quality cost.
- **`torch-fp16` and `torch-bf16` on CPU collapse to ~0.01x** — they run emulated. *Never* set `torch_dtype="float16"` for a CPU/MPS-fallback path. This is the single biggest CPU footgun.
- GPU: fp16+FlashAttention-2 with input unpadding is 3.87x and now beats every ONNX level; ONNX on GPU dropped to ~1.2x.
- OpenVINO is Intel-oriented; the official decision flowchart routes non-Intel CPUs to `onnx`. **On Apple Silicon: use `backend="onnx"` with `arm64` int8**, or plain torch fp32 on `mps`.

`model.compile()` finally speeds up `encode()` as of **v5.7.0** (#3848); before that it was a silent no-op for inference. `mode="reduce-overhead"` is CUDA-graph based (irrelevant on MPS); `dynamic=True` is the safe general option. Compilation is lazy — warm up before benchmarking.

### 1.6 Apple Silicon / MPS specifics

- `SentenceTransformer` device auto-selection order is `cuda` → `mps` → `cpu`, so `device=` can be omitted on a Mac.
- **MPS does not support `float64`** (`TypeError: Cannot convert a MPS Tensor to float64`). Any downstream code that upcasts (scipy, some sklearn paths) must run on CPU numpy. Cast embeddings to `float32` immediately after `encode()` and keep them there.
- MPS RNG has known non-determinism (dropout is non-deterministic even with a seed; pytorch#84516). For *inference-only* embedding this does not matter — eval mode has no dropout — but it means you cannot rely on MPS for any stochastic step. Do KMeans/UMAP/HDBSCAN on CPU.
- `torch.use_deterministic_algorithms(True)` on MPS causes severe perf degradation (pytorch#122394). Don't enable it globally; enable it only in the training/fine-tuning nodes if you have any.
- ST v5.6.0 added explicit MPS fixes for cached losses and `SparseEncoder` sparsity stats (#3812, #3818).
- ST v5.6.1 fixed a **silent quality regression** for RoBERTa/XLM-R family models (`bge-m3`, `multilingual-e5-*`) when flash attention was requested on transformers v5: packed `position_ids` were offset wrong, dropping stsb Spearman from 0.8485 → 0.7239. Affects 5.5.0/5.5.1/5.6.0. **If you embedded a corpus with `bge-m3` + `attn_implementation="flash_attention_2"` on those versions, re-encode.** Pinning 5.7.0 avoids it.

---

## 2. scikit-learn 1.9.0 — clustering & classification APIs (P4, P6, P10)

### 2.1 Deprecations that will bite

**`AgglomerativeClustering`: `affinity=` is GONE.** The current signature is:
```python
AgglomerativeClustering(n_clusters=2, *, metric='euclidean', memory=None, connectivity=None,
                        compute_full_tree='auto', linkage='ward', distance_threshold=None,
                        compute_distances=False)
```
`metric` accepts `"euclidean" | "l1" | "l2" | "manhattan" | "cosine" | "precomputed"` (added 1.2). With `linkage="ward"` only `"euclidean"` and — **new in 1.9** — `"l2"` are accepted (#24681). For cosine-geometry hierarchies use `linkage="average"` + `metric="cosine"`, or L2-normalize and use `ward`+`euclidean` (equivalent ordering, different absolute distances).

**`LogisticRegression`: `penalty` is deprecated.** Current signature:
```python
LogisticRegression(penalty='deprecated', *, C=1.0, l1_ratio=0.0, dual=False, tol=1e-4,
                   fit_intercept=True, intercept_scaling=1, class_weight=None,
                   random_state=None, solver='lbfgs', max_iter=100, verbose=0,
                   warm_start=False, n_jobs=None)
```
- `penalty` deprecated in **1.8, removed in 1.10**. Migration: `l1_ratio=0` ⇒ old `penalty='l2'`; `l1_ratio=1` ⇒ `'l1'`; `0<l1_ratio<1` ⇒ `'elasticnet'`; `C=np.inf` ⇒ `penalty=None`.
- `l1_ratio` default changed **None → 0.0** in 1.8; `None` removed in 1.10.
- `n_jobs` "does not have any effect", deprecated in 1.8, removed in 1.10.
- `multi_class` is gone entirely — all solvers except `liblinear` minimize the full multinomial loss for `n_classes>=3`; `liblinear` now **raises** for multiclass.
- **1.9 efficiency change:** `solver="lbfgs"` now computes the gradient at **float32** when fit on float32 `X`. If you need the old numerics, cast `X` to float64 explicitly. For a reproducible classifier gate this matters — pin the input dtype.

For P2's high-dim sparse char-TFIDF classifier:
```python
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(solver="saga", l1_ratio=0.0, C=4.0,
                         max_iter=3000, tol=1e-4, class_weight="balanced",
                         random_state=SEED)      # saga: sparse-friendly, multinomial, supports 0<=l1<=1
```
Solver/penalty table (from the docs): `lbfgs`/`newton-cg`/`newton-cholesky`/`sag` → `l1_ratio=0` only, multinomial yes. `liblinear` → `l1_ratio∈{0,1}`, multinomial **no**. `saga` → `0<=l1_ratio<=1`, multinomial yes. `newton-cholesky` is good for `n_samples >> n_features*n_classes` but its memory is quadratic in `n_features*n_classes` — do **not** use it on 200k-dim char TF-IDF.

**Other 1.9 API changes to note:** `svm.SVC(probability=)` deprecated (not thread-safe), removed in 1.11 → use `CalibratedClassifierCV(..., ensemble=False)`. `metrics.log_loss(y_pred=)` → `y_proba=`. `LogisticRegressionCV(scoring=None)` default will change to `"neg_log_loss"` in 1.11 — pass `scoring` explicitly to silence.

### 2.2 `n_init` semantics — three different defaults, verified

```python
KMeans(n_clusters=8, *, init='k-means++', n_init='auto', max_iter=300, tol=1e-4,
       verbose=0, random_state=None, copy_x=True, algorithm='lloyd')
```
- `KMeans.n_init='auto'` (default since **1.4**) ⇒ **10** runs if `init='random'` or callable; **1** run if `init='k-means++'` or array-like.
- `MiniBatchKMeans(n_clusters=8, *, init='k-means++', max_iter=100, batch_size=1024, ..., n_init='auto', reassignment_ratio=0.01)` — `'auto'` ⇒ **3** if `init='random'`/callable, **1** if `k-means++`. Note MBK runs the algorithm **once**, using the best of `n_init` *initializations* by inertia (different semantics from KMeans).
- `BisectingKMeans(n_clusters=8, *, init='random', n_init=1, random_state=None, max_iter=300, tol=1e-4, copy_x=True, algorithm='lloyd', bisecting_strategy='biggest_inertia')` — note **`init='random'` and `n_init=1` by default**, and `n_init` here is per-bisection.

**For the P4 battery, set `n_init` explicitly on every algorithm.** `n_init='auto'` with `k-means++` silently means a single restart, which makes your KMeans arm systematically weaker than a reader expects and makes the stability analysis in P5 measure init-noise rather than data structure. Use `n_init=10` for KMeans/BisectingKMeans, `n_init=3` for MiniBatchKMeans, and record it.

`MiniBatchKMeans` `batch_size` default is 1024 (changed from 100 in 1.0). Docs: "set `batch_size > 256 * number_of_cores` to enable parallelism on all cores" → on a 10-core M-series, `batch_size=4096` or `8192`.

**1.9 bugfix:** `MiniBatchKMeans` now correctly handles `sample_weight` (mini-batch indices are subsampled *with replacement* using normalized weights as probabilities, #30751). If you weight by query frequency, MBK results will differ from 1.8. **1.9 bugfix:** `BisectingKMeans` with a custom callable `init` and `n_clusters>2` was broken (#33148).

`GaussianMixture(n_components=1, *, covariance_type='full', tol=1e-3, reg_covar=1e-6, max_iter=100, n_init=1, init_params='kmeans', ..., random_state=None, warm_start=False)`. No changes in 1.9. On 1024-dim embeddings `covariance_type='full'` is 1024² per component — use `'diag'` or `'spherical'`, or run GMM on the SVD/UMAP-reduced representation only.

### 2.3 Metrics

```python
silhouette_score(X, labels, *, metric='euclidean', sample_size=None, random_state=None, **kwds)
adjusted_rand_score(labels_true, labels_pred)
normalized_mutual_info_score(labels_true, labels_pred, *, average_method='arithmetic')
adjusted_mutual_info_score(labels_true, labels_pred, *, average_method='arithmetic')
cohen_kappa_score(y1, y2, *, labels=None, weights=None, sample_weight=None,
                  replace_undefined_by=np.nan)   # replace_undefined_by NEW IN 1.9
```

`cohen_kappa_score(replace_undefined_by=...)` is **new in 1.9** and directly relevant to your P2 κ≥0.9 gate: kappa is undefined when one annotator uses only one label, which previously raised `UndefinedMetricWarning` and returned NaN. Pass `replace_undefined_by=0.0` so a degenerate annotator scores 0 (fails the gate) rather than NaN (silently propagates through `np.mean`).

New in 1.9 and useful: `metrics.metric_at_thresholds` (compute any metric across all thresholds — good for P10 margin-routing threshold selection), `PrecisionRecallDisplay.from_cv_results`, `metrics.d2_brier_score`.

### 2.4 Calibration + ECE (P10)

```python
CalibratedClassifierCV(estimator=None, *, method='sigmoid', cv=None, n_jobs=None, ensemble='auto')
```
`method='temperature'` was **added in 1.8** and is the right choice here: sigmoid/isotonic only support binary natively and extend to multiclass via OvR + renormalization, whereas **temperature scaling natively supports multiclass** via `softmax(logits/T)` optimizing log loss. For a 20-80-class intent classifier that is the correct calibrator. Isotonic is documented as overfitting below ~1000 calibration samples.

```python
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
cal = CalibratedClassifierCV(FrozenEstimator(fitted_clf), method="temperature", cv=None)
cal.fit(X_cal, y_cal)   # X_cal must be disjoint from training data — not enforced!
```

**sklearn has no ECE function.** Implement it once, deterministically:

```python
import numpy as np

def expected_calibration_error(y_true, proba, n_bins=15, strategy="uniform"):
    """Top-label ECE. proba: (n, n_classes). Returns (ece, mce, per-bin table)."""
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    acc = (pred == y_true).astype(np.float64)
    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    else:  # 'quantile' — equal-mass bins, more stable when confidences pile up near 1
        edges = np.quantile(conf, np.linspace(0, 1, n_bins + 1))
        edges[0], edges[-1] = 0.0, 1.0
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    n = len(conf); ece = 0.0; mce = 0.0; rows = []
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            rows.append((b, 0, np.nan, np.nan, 0.0)); continue
        a, c, w = acc[m].mean(), conf[m].mean(), m.sum() / n
        gap = abs(a - c)
        ece += w * gap; mce = max(mce, gap)
        rows.append((b, int(m.sum()), float(a), float(c), float(w)))
    return float(ece), float(mce), rows
```
Pair with `sklearn.calibration.calibration_curve` for the reliability diagram and `metrics.brier_score_loss` / `d2_brier_score` for a proper scoring rule. **Report ECE with `n_bins`, `strategy`, and the exact eval sample id-list in the metrics panel** — ECE is bin-count sensitive and is not comparable across configurations otherwise.

### 2.5 Char TF-IDF for Chinese (P1, P3)

```python
TfidfVectorizer(*, input='content', encoding='utf-8', decode_error='strict', strip_accents=None,
                lowercase=True, preprocessor=None, tokenizer=None, analyzer='word',
                stop_words=None, token_pattern='(?u)\\b\\w\\w+\\b', ngram_range=(1,1),
                max_df=1.0, min_df=1, max_features=None, vocabulary=None, binary=False,
                dtype=np.float64, norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False)
```

```python
import unicodedata, re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline

def norm_zh(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)      # full-width → half-width, ＡＢ→AB, ，→, — ESSENTIAL for CN logs
    s = re.sub(r"\s+", "", s)                 # Chinese queries: intra-string whitespace is noise
    return s.lower()

tfidf = TfidfVectorizer(
    analyzer="char",              # NOT 'char_wb': char_wb pads at *word* boundaries, and unsegmented
                                  # Chinese is one "word", so char_wb adds spurious edge n-grams.
    ngram_range=(2, 4),
    min_df=3, max_df=0.5,
    sublinear_tf=True,
    dtype=np.float32,             # halves memory; TruncatedSVD accepts float32
    preprocessor=norm_zh,
    lowercase=False,              # already done in preprocessor
)
lsa = make_pipeline(
    TruncatedSVD(n_components=256, algorithm="randomized", n_iter=7, random_state=SEED),
    Normalizer(copy=False),       # LSA output is NOT normalized — always follow with Normalizer
)
```

`TruncatedSVD(n_components=2, *, algorithm='randomized', n_iter=5, n_oversamples=10, power_iteration_normalizer='auto', random_state=None, tol=0.0)`. Docs recommend `n_components=100` for LSA; for hybrid concat 128–256 is the practical range. `random_state` matters only for `algorithm='randomized'` — but **sign flips of components are not stabilized**, so `svd.components_` differs run-to-run in sign; this does not affect distances but does affect any per-dimension logging. A 1.9 fix corrected a typo in the allowed `power_iteration_normalizer` values (`"OR"` → `"QR"`).

**Hybrid concat with α-sweep (P3):**
```python
from sklearn.preprocessing import normalize
def hybrid(E_dense, X_lsa, alpha):
    """alpha in [0,1]: weight on the neural embedding."""
    A = normalize(E_dense.astype(np.float32))      # L2
    B = normalize(X_lsa.astype(np.float32))        # L2
    H = np.hstack([alpha * A, (1.0 - alpha) * B])
    return normalize(H)   # re-normalize so cosine on H is well-defined and comparable across alpha
```
Re-normalizing after concat is what makes the α-sweep comparable — without it, changing α changes the vector norms and every distance-based metric moves for a trivial reason.

### 2.6 New in 1.9: `sparse_interface` config

```python
import sklearn
sklearn.set_config(sparse_interface="sparray")   # return scipy sparse *arrays* not spmatrix
```
Default is still `"spmatrix"`; the plan is to flip in a few releases. Set it explicitly in P0 so a future sklearn bump does not change the type flowing out of `TfidfVectorizer` into your custom code.

---

## 3. HDBSCAN: `sklearn.cluster.HDBSCAN` vs the `hdbscan` package (P4, P12)

### sklearn 1.9 signature
```python
sklearn.cluster.HDBSCAN(min_cluster_size=5, min_samples=None, cluster_selection_epsilon=0.0,
                        max_cluster_size=None, metric='euclidean', metric_params=None, alpha=1.0,
                        algorithm='auto', leaf_size=40, n_jobs=None, cluster_selection_method='eom',
                        allow_single_cluster=False, store_centers=None, copy='warn')
```

### `hdbscan` 0.8.44 signature
```python
hdbscan.HDBSCAN(min_cluster_size=5, min_samples=None, cluster_selection_epsilon=0.0,
                cluster_selection_persistence=0.0, max_cluster_size=0, metric='euclidean',
                alpha=1.0, p=None, algorithm='best', leaf_size=40, memory=Memory(location=None),
                approx_min_span_tree=True, gen_min_span_tree=False, core_dist_n_jobs=4,
                cluster_selection_method='eom', allow_single_cluster=False,
                prediction_data=False, branch_detection_data=False,
                match_reference_implementation=False, cluster_selection_epsilon_max=inf, **kwargs)
```

### The differences that matter

1. **`min_samples` is off by one between them.** sklearn's Notes section, verbatim: "The `min_samples` parameter includes the point itself, whereas the implementation in scikit-learn-contrib/hdbscan does not. To get the same results in both versions, the value of `min_samples` here must be **1 greater** than the value used in scikit-learn-contrib/hdbscan." If P4 benchmarks one and P12 deploys the other with the same number, the results silently diverge.

2. **`copy='warn'` in sklearn 1.9, changing to `True` in 1.10.** Pass `copy=True` explicitly today. It only applies with `metric="precomputed"` + dense/CSR + `algorithm="brute"`, but that's exactly the precomputed-cosine path.

3. **sklearn labels noise more finely:** `-1` = noise, `-2` = sample contained ±inf, `-3` = sample had missing data. Any `labels >= 0` mask is correct; any `labels != -1` mask is a bug.

4. **`store_centers`** (sklearn only) ∈ `{None, "centroid", "medoid", "both"}`. **`"medoid"` is the right choice on normalized embeddings** — the docstring warns the centroid "uses the euclidean metric and does not guarantee the output will be an observed data point", and a euclidean centroid of unit vectors is not itself a unit vector. `medoids_` is well-defined for arbitrary metrics and is an actual query, which is exactly what P7's blind-naming agents need to see.

5. **Only the `hdbscan` package has the novelty-sentinel machinery.** Confirmed present in 0.8.44's API reference and absent from sklearn: `PredictionData`, `approximate_predict()`, `approximate_predict_scores()`, `membership_vector()`, `all_points_membership_vectors()`, `relative_validity_` (DBCV), `validity_index()`, plus branch detection (`BranchDetector`, `detect_branches_in_clusters`, `approximate_predict_branch`) and `weighted_cluster_centroid()` / `weighted_cluster_medoid()`.

**Recommendation:** use `sklearn.cluster.HDBSCAN` inside the P4 battery (one dependency, one `random_state`-free deterministic path, `store_centers="medoid"`), and use the `hdbscan` package **only** for P12's drift/novelty sentinel where you need `prediction_data=True` + `approximate_predict`:

```python
import hdbscan
sentinel = hdbscan.HDBSCAN(min_cluster_size=30, min_samples=10,
                           metric="euclidean",          # on L2-normalized vectors
                           cluster_selection_method="eom",
                           prediction_data=True, gen_min_span_tree=True,
                           core_dist_n_jobs=8).fit(E_ref)
dbcv = sentinel.relative_validity_       # requires gen_min_span_tree=True
labels_new, strengths = hdbscan.approximate_predict(sentinel, E_new)
novelty_rate = (labels_new == -1).mean() # ← the drift sentinel signal
soft = hdbscan.membership_vector(sentinel, E_new)   # per-cluster soft membership for borderline routing
```

### Behaviour on normalized embeddings
HDBSCAN has **no `metric="cosine"`** path that is efficient — `"cosine"` forces `algorithm="brute"` (it's not a valid KDTree/BallTree metric), which is O(n²) and infeasible above ~50k rows. **L2-normalize and use `metric="euclidean"`**: for unit vectors `d_euc² = 2(1 − cos)`, so euclidean and cosine induce identical neighbor rankings, and KDTree/BallTree work. This is the single most important HDBSCAN setup detail for this playbook.

### Parameter meanings (for P5/P6 tuning)
- `min_cluster_size`: smallest admissible cluster. This is your **governance floor** — set it to the minimum cluster size an analyst would accept as a real intent (e.g. `max(30, 0.0005*n)`).
- `min_samples`: conservativeness / how much is declared noise. Higher ⇒ more noise, tighter cores. Default `None` ⇒ equals `min_cluster_size`, which is usually too aggressive; set it independently (10–25 is typical).
- `cluster_selection_epsilon`: distance floor below which clusters are merged (Malzer & Baum hybrid). Useful in P8 to enforce "don't split below this granularity" as a *parameter* rather than a post-hoc merge.
- `cluster_selection_method`: `"eom"` (excess of mass, fewer/larger, the standard) vs `"leaf"` (finest, most homogeneous). **P5 uses this as a granularity dial: `"leaf"` for the overclustering-survival arm, `"eom"` for the final tree.**
- `max_cluster_size`: caps EOM cluster size; no effect under `"leaf"`.

### The "high-dimensional density" problem
The official umap docs demonstrate this concretely: HDBSCAN on 50-dim PCA of MNIST clustered only **17%** of the data (ARI 0.054 overall) because density is meaningless in 50-D. Expect the same on raw 1024-dim embeddings. This is *the* reason people reach for UMAP first — see next section.

---

## 4. UMAP 0.5.12 (P3 viz, P5 diagnostics)

### Reproducibility: `random_state` silently kills `n_jobs` — confirmed in source

From `umap/umap_.py` master (line ~2003):
```python
if self.n_jobs < -1 or self.n_jobs == 0:
    raise ValueError("n_jobs must be a postive integer, or -1 (for all cores)")
if self.n_jobs != 1 and self.random_state is not None:
    self.n_jobs = 1
    warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
```
The default is `n_jobs=-1`. So **setting `random_state` forces single-threaded and warns**, and the docs measure this as roughly a 2–3x wall-clock penalty (their example: 1m56s seeded vs much faster unseeded, with CPU≈wall time once seeded). Budget for it — on 500k×1024 this is the slowest step in the pipeline. There is no "reproducible and parallel" option.

Also: `numba.set_num_threads(self.n_jobs)` is called globally inside `fit` (line ~2467) — UMAP mutates numba's global thread count as a side effect, which affects *everything else numba-backed in the process afterwards*. Reset it explicitly if you run UMAP inside a long-lived agent process.

### Params: visualization vs clustering (from the official "Using UMAP for Clustering" doc)
```python
# VISUALIZATION (P11 figures)
viz = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2,
                metric="cosine", random_state=SEED).fit_transform(E)

# "CLUSTERABLE" (the doc's exact recipe)
clusterable = umap.UMAP(n_neighbors=30, min_dist=0.0, n_components=2,
                        metric="cosine", random_state=SEED).fit_transform(E)
```
`min_dist=0.0` is the key difference — it packs points inside clusters so gaps open up. On MNIST this took HDBSCAN from 17% of points clustered (ARI 0.054) to **99.2% clustered (ARI 0.924)**.

### The "don't cluster on UMAP output" caveat — what the primary source actually says
The umap doc itself opens the clustering page with: "This is somewhat controversial, and should be attempted with care… UMAP, like t-SNE, does not completely preserve density. UMAP, like t-SNE, can also **create false tears in clusters, resulting in a finer clustering than is necessarily present in the data**." Secondary literature is harsher: UMAP+HDBSCAN "generally increased both the number of clusters and unclustered samples"; results are "heavily dependent on the hyperparameters of UMAP (`n_neighbors` and `min_dist`)"; relative distances are not preserved so **borderline-case detection and anomaly typing are invalid** on UMAP coordinates.

**Concrete policy for this playbook:**
1. **Never let UMAP coordinates decide K or the final partition.** Cluster in the original (or SVD-reduced) space; the P9 metrics panel must be computed on the *same* representation the labels came from.
2. If you use UMAP as a clustering preprocessor at all, treat it as *one more arm of the P4 battery* with its params recorded, and require it to survive P5 stability the same as every other arm — the false-tear failure mode shows up precisely as low bootstrap ARI.
3. Use UMAP freely for P11 figures and for the human-in-the-loop review UI. Fit the viz embedding **once**, with a fixed seed, and reuse the same 2-D coordinates for every figure across all phases so reviewers see a stable map.
4. Cluster-count inflation is measurable: report `n_clusters` and noise fraction for the UMAP arm alongside the non-UMAP arms and let the discrepancy be visible.

### Alternative to UMAP for the clustering path
`TruncatedSVD(n_components=100–256)` or `PCA` is linear, deterministic, parallel, has an exact `transform` for new data, and preserves the global geometry that HDBSCAN's density estimates need in a defensible way. Prefer it as the P4 dimensionality-reduction arm; keep UMAP for eyes only.

---

## 5. K selection / stability (P5)

### 5.1 DeepAligned overclustering-survival — exact algorithm, read from source

From `thuiar/DeepAligned-Clustering` (`DeepAligned.py::predict_k` and `dataloader.py`), verbatim logic:

```python
# dataloader.py:21
self.num_labels = int(len(self.all_label_list) * args.cluster_num_factor)   # K' = factor * K_ref

# DeepAligned.py:64-84
def predict_k(feats, K_prime):
    km = KMeans(n_clusters=K_prime).fit(feats)
    y_pred = km.labels_
    drop_out = len(feats) / K_prime          # = the MEAN cluster size under K'
    cnt = sum(1 for c in np.unique(y_pred) if (y_pred == c).sum() < drop_out)
    return len(np.unique(y_pred)) - cnt      # K_hat = number of SURVIVING clusters
```

So: **overcluster at `K' = factor · K_ref`, drop every cluster whose size is below the mean cluster size `N/K'`, and the count of survivors is the estimate of K.** `cluster_num_factor` is a required CLI arg (`default=1.0` in the parser but the paper's unknown-K setting uses 2.0 and 4.0). Note the reference implementation calls `KMeans(n_clusters=K')` **without a `random_state`** — you must add one.

Reusable, seed-controlled version for your P5 node:

```python
def deepaligned_k(X, k_ref, factor=2.0, seed=0, n_init=10):
    k_prime = int(k_ref * factor)
    km = KMeans(n_clusters=k_prime, n_init=n_init, random_state=seed).fit(X)
    sizes = np.bincount(km.labels_, minlength=k_prime)
    threshold = len(X) / k_prime
    return int((sizes >= threshold).sum()), sizes, threshold
```
Sweep `factor ∈ {1.5, 2, 3, 4}` and `seed ∈ {0..4}` and report the K̂ distribution — the estimator is noticeably factor-sensitive, and reporting a single number would be overclaiming. This is your "overclustering survival" leg of the triangulation.

### 5.2 Bootstrap/resampling ARI stability (von Luxburg)

Reference: U. von Luxburg, *Clustering Stability: An Overview*, Foundations and Trends in ML 2(3), arXiv:1007.1075. Key caution from that survey for your write-up: many stability protocols exist, **no comprehensive comparison of protocols exists**, and normalization is unresolved — so report the protocol explicitly rather than "stability = 0.87".

Protocol that works at your scale (the two-subsample / overlap variant, which avoids clustering the full 750k B times):

```python
def stability_ari(X, k, cluster_fn, n_reps=25, frac=0.8, seed=0):
    """Two-subsample overlap ARI. Returns (mean, std, all)."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]; m = int(frac * n); out = []
    for r in range(n_reps):
        i1 = rng.choice(n, m, replace=False)
        i2 = rng.choice(n, m, replace=False)
        both = np.intersect1d(i1, i2, assume_unique=True)
        if len(both) < 100:
            continue
        l1 = cluster_fn(X[i1], k, seed=1000 + r)
        l2 = cluster_fn(X[i2], k, seed=2000 + r)
        p1 = {ix: lb for ix, lb in zip(i1, l1)}
        p2 = {ix: lb for ix, lb in zip(i2, l2)}
        out.append(adjusted_rand_score([p1[i] for i in both], [p2[i] for i in both]))
    a = np.asarray(out)
    return a.mean(), a.std(ddof=1), a
```
Use **subsampling without replacement (`frac=0.8`), not bootstrap-with-replacement**, for centroid methods: duplicated points from a bootstrap distort KMeans centroids and inflate apparent stability. Then take the **stability peak over K** — but note von Luxburg's central result: stability curves reliably detect *too large* K, and are much weaker at detecting *too small* K, and monotone-decreasing stability with K is the common degenerate case. Always report the full curve, not the argmax.

Complementary K-selection legs worth having in the panel:
- **Consensus clustering**: build the co-association matrix `C[i,j] = P(i,j same cluster)` over B runs (on a fixed 20–50k subsample so it fits: 30k² float32 = 3.6 GB — use `uint8` counts, 0.9 GB), then read the consensus CDF / PAC (proportion of ambiguous clustering) as a function of K. Choose K minimizing PAC.
- **Gap statistic** (Tibshirani): compare `log W_k` to uniform reference data drawn in the PCA-aligned bounding box. On L2-normalized embeddings the correct null is the **uniform distribution on the sphere**, not a box — `Z = rng.normal(size=(n,d)); Z /= norm(Z)`. Using the box null on sphere data is a common and invalidating error.
- **Silhouette peak** (cosine, fixed subsample) — see §7.
- **BIC/AIC** from `GaussianMixture` on the reduced representation.
- **Expert intuition prior**, recorded as an interval `[K_lo, K_hi]` *before* any of the above runs, so the triangulation is auditable and not post-hoc.

Triangulation rule for the P5 gate: require ≥2 of {stability peak, DeepAligned survival, silhouette peak} to fall inside the expert interval; otherwise escalate to the human gate with the three curves attached.

---

## 6. Chinese NLP (P1, P3, P11)

### Segmentation: current reality
- **`jieba` 0.42.1 — last release 2020-01-20.** Six years unmaintained. Still works (pure Python, no compiled deps, no numpy coupling) and is still the pragmatic default for *feature extraction* because it is fast and its behaviour is frozen (which is a **reproducibility asset**: a frozen tokenizer can't drift on you between quarterly reruns). Add a domain user dict (`jieba.load_userdict`) with your K12 subject terms.
- **`pkuseg` 0.0.25 — last release 2020-06-27, dead.** Use **`spacy-pkuseg` 1.0.1 (2025-07-14)**, the maintained fork, which is what spaCy's Chinese pipeline depends on. Offers domain models (news/web/medicine/tourism).
- **`hanlp` 2.1.3 (2025-10-19)** — actively maintained, transformer-based, highest accuracy, but pulls a heavy dependency tree and is slow on 750k rows.
- **`ltp` 4.2.14 (2024-06-08)** — pins `python<4,>=3.6`; check torch compatibility before adopting.

**Recommendation:** for template-family mining (P1) and char-TFIDF (P3), **you mostly do not need segmentation at all.** Character n-grams (2–4) on NFKC-normalized text outperform word features for short noisy queries and remove the tokenizer as a source of drift entirely. Use `jieba` only for (a) human-readable top-term reports and (b) regex template family induction where word boundaries help. Record `jieba.__version__` and the userdict hash in the manifest.

### Normalization pipeline (do this before everything)
```python
import unicodedata, re
FULLWIDTH_DIGITS = str.maketrans("０１２３４５６７８９", "0123456789")

def canonicalize(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)     # ＡＢＣ→ABC, ，→, ？→?, ﾊ→ハ
    s = s.translate(FULLWIDTH_DIGITS)
    s = re.sub(r"[​-‏﻿]", "", s)   # zero-width chars — very common in copy-pasted logs
    s = re.sub(r"\s+", " ", s).strip()
    return s
```
NFKC is non-negotiable for CN query logs: full-width and half-width variants of the same query will otherwise land in different template families and different clusters. Also normalize Traditional→Simplified if your log mixes them (`opencc-python-reimplemented`).

### Template-family regex mining (P1)
Digit/date/name slotting before hashing is what makes template families collapse:
```python
SLOTS = [
    (re.compile(r"\d{4}[-/年]\d{1,2}[-/月]\d{1,2}日?"), "<DATE>"),
    (re.compile(r"[一二三四五六七八九十百千万亿\d]+(?=年级|年|月|日|分|元|块)"), "<NUM>"),
    (re.compile(r"\d+(\.\d+)?"), "<NUM>"),
    (re.compile(r"[A-Za-z]{2,}"), "<EN>"),
]
def templatize(s):
    for pat, tok in SLOTS:
        s = pat.sub(tok, s)
    return s
```
Then group by `templatize(canonicalize(q))` and report family size, distinct-fill count, and the "expected cluster count" entropy (§7.3) per family to quantify **template fragmentation**.

### CJK matplotlib fonts on macOS — verified on this machine
Actual fonts present in `/System/Library/Fonts/` and `.../Supplemental/`:
```
Hiragino Sans GB.ttc     (/System/Library/Fonts/)
STHeiti Light.ttc        (/System/Library/Fonts/)
STHeiti Medium.ttc       (/System/Library/Fonts/)
Songti.ttc               (/System/Library/Fonts/Supplemental/)
Arial Unicode.ttf        (/System/Library/Fonts/Supplemental/)
```
```python
import matplotlib
from matplotlib import font_manager

CJK_CANDIDATES = ["PingFang SC", "Hiragino Sans GB", "Heiti SC", "STHeiti",
                  "Songti SC", "Arial Unicode MS", "Noto Sans CJK SC", "SimHei"]
available = {f.name for f in font_manager.fontManager.ttflist}
chosen = next((f for f in CJK_CANDIDATES if f in available), None)
if chosen is None:                      # e.g. inside a Linux CI container
    raise RuntimeError(f"No CJK font found; install fonts-noto-cjk. Available: {sorted(available)[:20]}")
matplotlib.rcParams["font.sans-serif"] = [chosen, "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False    # else the minus sign renders as a tofu box
```
`.ttc` collections are sometimes not picked up by `font_manager` automatically — if `PingFang SC` is missing from `ttflist`, register explicitly with `font_manager.fontManager.addfont("/System/Library/Fonts/Supplemental/Songti.ttc")` and clear `~/.matplotlib/fontlist-*.json`. **Fail loudly** (as above) rather than silently rendering tofu boxes into a report — a report full of □□□ is a worse failure than a crashed job. Add a P0 smoke test that renders `plt.text(0,0,"小学数学")` and asserts the rendered bitmap is not blank.

---

## 7. Metrics panel (P9) — deterministic implementations

### 7.1 Cosine silhouette on a fixed subsample

`silhouette_score(X, labels, sample_size=N, random_state=S)` internally does `indices = check_random_state(S).permutation(n)[:N]` and then `silhouette_samples(X[indices], labels[indices])`. That is deterministic **only if `n` is identical across runs** — if a filtering step changes `n` by even one row, the permutation changes and you get a different sample. Since P9 demands "same code/sample/seed", pin the index set as an artifact:

```python
import numpy as np, json, hashlib
from sklearn.metrics import silhouette_score, silhouette_samples

def make_eval_index(row_ids, n_sample=10_000, seed=0, path="artifacts/p9_eval_idx.json"):
    """Compute ONCE, commit the file, reuse for every arm and every rerun."""
    rng = np.random.default_rng(seed)
    ids = np.sort(rng.choice(np.asarray(row_ids), size=n_sample, replace=False))
    payload = {"seed": seed, "n_sample": n_sample, "row_ids": ids.tolist()}
    payload["sha256"] = hashlib.sha256(json.dumps(payload["row_ids"]).encode()).hexdigest()
    json.dump(payload, open(path, "w"))
    return ids

def cosine_silhouette(X, labels, idx_positions):
    Xs, ls = X[idx_positions], labels[idx_positions]
    keep = ls >= 0                       # drop HDBSCAN noise (-1/-2/-3) BEFORE scoring
    Xs, ls = Xs[keep], ls[keep]
    if len(np.unique(ls)) < 2:
        return float("nan")
    return float(silhouette_score(Xs, ls, metric="cosine"))  # NO sample_size — we already sampled
```
Notes:
- Stratify the eval index by cluster only if you disclose it; a uniform sample is the honest default and comparable across arms with different K.
- `metric="cosine"` is `1 − cos_sim` via `pairwise_distances`. Even on L2-normalized data, cosine silhouette ≠ euclidean silhouette numerically (silhouette is a ratio of raw distances, and `d_euc = sqrt(2·d_cos)` is a non-affine map), so **the metric string must be fixed in the panel config**, not left to a default.
- Complexity is O(N²·d). At N=10k, d=1024 that's ~1e11 flops — ~10–60 s. At N=50k it's 25x that. **N=5k–20k is the practical band.** Reduce cost by running silhouette on the SVD-256 representation and disclosing it, or keep d=1024 and N=10k.
- `sklearn.set_config(working_memory=512)` (MB) tunes `pairwise_distances_chunked` inside `silhouette_samples` — affects peak memory only, not the result.
- Always report `silhouette_samples` percentiles too, not just the mean; a mean of 0.11 hides whether three clusters are excellent and twenty are garbage.

### 7.2 ARI / NMI
```python
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score
ari = adjusted_rand_score(gold, pred)
nmi = normalized_mutual_info_score(gold, pred, average_method="arithmetic")   # sklearn default
ami = adjusted_mutual_info_score(gold, pred, average_method="arithmetic")
```
Report **AMI alongside NMI**: NMI is biased upward with increasing K, so comparing K=40 against K=120 on NMI alone systematically favours the larger K. AMI is chance-corrected and is the honest comparator across granularities. Both are permutation-invariant, so P8's lookup-table remap does not change them — which is exactly the invariant your governance-merge node should assert in a unit test.

### 7.3 Expected cluster count (perplexity of the size distribution)
```python
def effective_n_clusters(labels, exclude_noise=True, base=np.e):
    l = np.asarray(labels)
    if exclude_noise:
        l = l[l >= 0]
    _, counts = np.unique(l, return_counts=True)
    p = counts / counts.sum()
    H = -(p * np.log(p)).sum()          # Shannon entropy, nats
    return float(np.exp(H))             # perplexity == "expected cluster count"
```
`exp(H) ≤ K` always, with equality iff clusters are equal-sized. **`K / exp(H)` is the fragmentation/imbalance ratio** and is the right single number for "template fragmentation" in P1 and for cluster-size health in P6: a value near 1.0 means a balanced taxonomy; 4.0 means most of your K is spent on tiny clusters and P8 has merging to do. Report `(K, exp(H), K/exp(H), noise_frac)` as a quadruple for every arm. Use `np.log2` and `2**H` if you prefer bits — just be consistent, it changes the number.

### 7.4 Uniformity discipline
The panel node should take a single frozen config object and refuse to score an arm that doesn't match it:
```python
@dataclass(frozen=True)
class MetricConfig:
    eval_idx_sha256: str
    silhouette_metric: str = "cosine"
    representation: str = "hybrid_alpha0.7"   # which matrix metrics are computed on
    seed: int = 0
    n_bins_ece: int = 15
    exclude_noise: bool = True
```
Hash it into every metrics row. This is what makes P9 "same code/sample/seed" actually enforceable rather than aspirational.

---

## 8. Scale: 50k–750k rows

### Memory arithmetic (float32)
| n × d | bytes |
|---|---|
| 750k × 1024 | **3.07 GB** |
| 750k × 768 | 2.30 GB |
| 500k × 1024 | 2.05 GB |
| 750k × 256 (SVD) | 0.77 GB |
| 750k × 1024 float16 | 1.54 GB |

Store embeddings as `float32` `.npy` (memory-mappable via `np.load(path, mmap_mode="r")`) plus a parquet sidecar of row ids. Do **not** put 1024-dim vectors in a pandas column — pandas 3 will box them.

### Algorithm choice by n
- **n ≤ 100k**: full `KMeans(algorithm="lloyd", n_init=10)`, `AgglomerativeClustering`, `GaussianMixture(covariance_type="diag")`, `HDBSCAN` all feasible.
- **n > 200k**: full `KMeans` costs ~`n·K·d` per iteration — at n=750k, K=200, d=1024 that is **1.5e11 flops/iteration**, minutes per iteration on a laptop. Switch to `MiniBatchKMeans(batch_size=8192, n_init=3, max_iter=300, reassignment_ratio=0.01, random_state=SEED)`.
- **Agglomerative above ~50k is O(n²) memory** — `connectivity=kneighbors_graph(X, 30, include_self=False)` makes it sparse and tractable, but note that structured agglomerative gives different (and generally better-behaved) results than unstructured, so it is a different arm, not a faster version of the same arm.
- **Two-stage strategy for the full corpus:** MiniBatchKMeans to K'≈2000 micro-clusters → Agglomerative/Ward on the 2000 centroids (weighted by micro-cluster size) → assign every row to its micro-cluster's macro label. This gives you the hierarchical structure P6 needs at 750k scale, deterministically, in minutes.

### kNN / ANN
`faiss-cpu==1.15.0` ships `cp310-abi3-macosx_14_0_arm64.whl` — a real Apple Silicon wheel (macOS 14+ required; the x86_64 wheel is `macosx_15_0`). `hnswlib 0.8.0` is **sdist-only** and needs a compiler, so prefer faiss.
```python
import faiss, numpy as np
faiss.omp_set_num_threads(8)
E = np.ascontiguousarray(E, dtype="float32")
faiss.normalize_L2(E)                    # in-place; then inner product == cosine
index = faiss.IndexFlatIP(E.shape[1])    # exact; fine up to ~1M x 1024 on a laptop
index.add(E)
D, I = index.search(E, 31)               # 31 = self + 30 neighbours
```
`IndexFlatIP` is **exact and fully deterministic** — use it, not `IndexIVFFlat`, for anything that feeds a reported metric. Reserve IVF/HNSW for the P10 serving path where approximate is acceptable, and record `nprobe`/`efSearch` because they change results.

### Threads & joblib
```python
import os
# Set BEFORE importing numpy/sklearn.
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"    # pip numpy/scipy link OpenBLAS
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["VECLIB_MAXIMUM_THREADS"] = "8"  # Apple Accelerate
os.environ["TOKENIZERS_PARALLELISM"] = "false"   # silences the HF fork warning
```
Verify with `OMP_NUM_THREADS=2 python -m threadpoolctl -i numpy scipy sklearn` (the exact command the sklearn docs give). Inside the process, prefer `threadpoolctl.threadpool_limits(n)` as a context manager over env vars for per-node control.

**Oversubscription:** joblib's loky backend already caps child threads at `n_cpus // n_jobs`, but that mitigation does **not** apply to the `threading` backend or to nested numba (UMAP). On an M-series chip, `os.cpu_count()` counts efficiency cores too — using all of them is usually *slower* for BLAS work. Set threads to the **performance-core count** (8 on M3/M4 Pro, 12 on Max), not `cpu_count()`.

`SKLEARN_WORKING_MEMORY` / `sklearn.set_config(working_memory=)` controls chunk size in `pairwise_distances_chunked`, which is what silhouette and `pairwise_distances_argmin` use. A 1.9 fix removed a quadratic-time path in `pairwise_distances_argmin[_min]` when many distances tie (#33252) — relevant when you have exact-duplicate queries, which query logs are full of. **Deduplicate before clustering and carry a frequency weight** — it cuts n substantially and removes the tie pathology.

---

## 9. Reproducibility (P0) — what actually breaks it

### Seeding
```python
import os, random, numpy as np, torch

def seed_everything(seed: int = 20260817):
    os.environ["PYTHONHASHSEED"] = str(seed)   # only effective if set before interpreter start
    random.seed(seed)
    np.random.seed(seed)                       # legacy global — sklearn's random_state=int does NOT use it
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```
`np.random.seed` is *not* what makes sklearn reproducible. **Pass `random_state=SEED` (an int) to every estimator.** Passing a `RandomState`/`Generator` *instance* makes the estimator consume and advance shared state, so results depend on call order — an explicit int is the only reproducible choice, and `Generator` instances are still not accepted by most sklearn estimators.

### What breaks determinism, ranked by how often it bites
1. **`numpy` version drift via numba's `numpy<2.6` ceiling.** Pin numpy.
2. **`n_init='auto'`.** Its meaning depends on `init` and differs between KMeans (10/1) and MiniBatchKMeans (3/1). Always pass an integer.
3. **Thread count.** OpenMP/BLAS reductions are floating-point non-associative, so `KMeans` inertia and `pairwise_distances` can differ in the last bits across different `OMP_NUM_THREADS`. Pin the thread count in the manifest and set it before import. Differences are ~1e-12 and usually irrelevant, but they can flip an argmax at a tie and change a label assignment.
4. **dtype.** float32 vs float64 changes KMeans results outright (and, new in 1.9, changes `LogisticRegression(solver="lbfgs")` gradient precision). Fix dtype at the artifact boundary: `E = E.astype(np.float32)`, and assert it.
5. **UMAP with `random_state=None`.** Non-deterministic and parallel; with a seed, deterministic and single-threaded. There is no middle ground.
6. **MPS.** Non-deterministic RNG (dropout, pytorch#84516), no float64. Inference-only embedding is fine; do not run stochastic steps there.
7. **HDBSCAN `min_samples` semantics** differing between sklearn and the contrib package (off by one).
8. **`sklearn.set_config(sparse_interface=)`** default is scheduled to flip. Set it.
9. **pandas 3 `StringDtype`.** A 1.9 fix (#33491) makes `check_array(dtype="numeric")` correctly *reject* pandas 3 string columns that it previously silently accepted as object dtype. Code that "worked" on pandas 2 may now raise `ValueError`. This is a correctness fix — welcome it, but expect the break.
10. **`fit_transform` vs `fit().transform()`** are not always bit-identical (TruncatedSVD randomized path). Pick one and stay.

### Manifest to hash into every artifact
```python
import sklearn, numpy, scipy, torch, sentence_transformers, umap, hdbscan, platform, json, hashlib
manifest = {
    "python": platform.python_version(),
    "platform": platform.platform(),
    "seed": SEED,
    "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
    "versions": {
        "numpy": numpy.__version__, "scipy": scipy.__version__,
        "sklearn": sklearn.__version__, "torch": torch.__version__,
        "sentence_transformers": sentence_transformers.__version__,
        "umap": umap.__version__, "hdbscan": hdbscan.__version__,
    },
    "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
    "embedding_revision": "<pin the HF commit sha — NOT 'main'>",
    "prompt": CLUSTER_PROMPT,
    "normalize_embeddings": True,
    "dtype": "float32",
    "eval_idx_sha256": eval_idx_sha,
}
run_id = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:16]
```
**Pin the HF `revision` to a commit sha.** Model repos get silently updated; `main` is not a version. `SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", revision="<sha>")`.

---

## 10. Deployment (P10)

Centroid classifier on L2-normalized embeddings — cosine reduces to a single matmul, is exactly reproducible, and needs no model file beyond the centroid matrix:

```python
class CentroidRouter:
    def __init__(self, C, labels, tau_accept=0.10, tau_reject=0.35):
        self.C = normalize(C.astype("float32"))   # (K, d) — use MEDOIDS for HDBSCAN clusters
        self.labels, self.tau_accept, self.tau_reject = labels, tau_accept, tau_reject

    def route(self, E):
        S = normalize(E.astype("float32")) @ self.C.T        # (n, K) cosine
        order = np.argsort(-S, axis=1)
        top1, top2 = order[:, 0], order[:, 1]
        s1 = S[np.arange(len(S)), top1]
        margin = s1 - S[np.arange(len(S)), top2]
        decision = np.where(s1 < self.tau_reject, "reject",
                    np.where(margin < self.tau_accept, "review", "accept"))
        return top1, s1, margin, decision
```
Pick `tau_accept` / `tau_reject` with `metrics.metric_at_thresholds` (new in 1.9) on the held-out gold set, targeting a fixed review budget (e.g. accept ≥ 90% of traffic at ≥ 95% precision). Calibrate the *scores* with `CalibratedClassifierCV(method="temperature")` on the LR distillation head so the reported confidences are meaningful, then report ECE (§2.4) at the same thresholds. Distill to `LogisticRegression(solver="saga", l1_ratio=0.0)` on char-TFIDF for a millisecond-latency fallback that needs no embedding model at inference time.


---

## Recommendations carried into the design

- Pin `numpy>=2.4,<2.6` in constraints.txt: numba 0.67.0 (which umap-learn 0.5.12 requires) declares `numpy<2.6`, making this the single binding constraint on the entire stack, and an unpinned `pip install` will silently downgrade numpy the day 2.6 ships.
- Default to `Qwen/Qwen3-Embedding-0.6B` (1024-d, MRL, instruction-aware) with a fixed English clustering instruction and `normalize_embeddings=True` — it scores 68.74 on C-MTEB Clustering, beating gte-Qwen2-7B (66.06) at 1/12 the size, with `bge-base-zh-v1.5` kept only as the fast baseline arm.
- Set `n_init` explicitly on every clustering estimator: `n_init='auto'` means 1 restart with `init='k-means++'` for KMeans and MiniBatchKMeans, and `BisectingKMeans` defaults to `init='random', n_init=1` — leaving defaults makes P5 stability measure init noise rather than data structure.
- Migrate all sklearn calls off the 1.8/1.9 deprecations now: `AgglomerativeClustering(metric=)` not `affinity=` (removed), and `LogisticRegression(l1_ratio=0.0, C=...)` not `penalty=` (removed in 1.10, along with `n_jobs`; `multi_class` is already gone).
- L2-normalize embeddings and run HDBSCAN with `metric='euclidean'`, never `metric='cosine'` — cosine forces `algorithm='brute'` (O(n^2), infeasible above ~50k rows) while on unit vectors euclidean induces identical neighbour rankings.
- Use `sklearn.cluster.HDBSCAN(copy=True, store_centers='medoid')` in the P4 battery but the `hdbscan` 0.8.44 package for the P12 novelty sentinel — only the contrib package has `prediction_data=True`, `approximate_predict`, `membership_vector`, and `relative_validity_` (DBCV).
- Remember that `min_samples` differs by one between the two HDBSCAN implementations (sklearn includes the point itself) — encode the +1 conversion in a shared config adapter so P4 benchmarks and P12 deployment cannot silently diverge.
- Implement DeepAligned K-estimation exactly as the reference source does — overcluster at K'=factor*K_ref, drop clusters whose size is below the mean N/K', count survivors — but add the `random_state` the reference omits and sweep factor in {1.5,2,3,4} x 5 seeds, reporting the K-hat distribution rather than a point estimate.
- Freeze the P9 evaluation subsample as a committed, SHA-256-hashed list of row ids rather than relying on `silhouette_score(sample_size=, random_state=)`, whose internal permutation changes whenever n changes by even one row.
- Report `(K, exp(Shannon entropy), K/exp(H), noise_fraction)` for every clustering arm — `K/exp(H)` is the right single number for template fragmentation and cluster-size health, and pair NMI with AMI since NMI is biased upward with K and would systematically favour finer granularities.
- Never set `torch_dtype='float16'` or `'bfloat16'` on a CPU or CPU-fallback path: the official sentence-transformers benchmark measures them at roughly 0.01x speed because they run emulated; on Apple Silicon use torch fp32 on `mps`, or `backend='onnx'` with `quantization_config='arm64'` for ~3x on CPU.
- Accept that `umap.UMAP(random_state=SEED)` force-overrides `n_jobs` to 1 (confirmed in umap_.py master) and also mutates numba's global thread count as a side effect — budget the 2-3x wall-clock cost, and use `TruncatedSVD` rather than UMAP as the P4 dimensionality-reduction arm since UMAP demonstrably creates false tears and inflates cluster counts.
- Use `CalibratedClassifierCV(method='temperature')` (added in sklearn 1.8) for the P10 intent classifier — it is the only method with native multiclass support via `softmax(logits/T)`, whereas sigmoid and isotonic extend to multiclass only through OvR plus renormalization.
- Pass `cohen_kappa_score(..., replace_undefined_by=0.0)` (new in 1.9) for the P2 kappa>=0.9 gate so a degenerate annotator who used only one label scores 0 and fails the gate, rather than returning NaN that silently propagates through downstream aggregation.
- For 750k rows, run MiniBatchKMeans to ~2000 micro-clusters then Ward-agglomerate the size-weighted centroids to build the P6 two-level hierarchy — full KMeans at K=200, d=1024 costs ~1.5e11 flops per iteration and full Agglomerative is O(n^2) memory.
- Use `faiss-cpu` 1.15.0's `IndexFlatIP` (exact, deterministic, real macosx_14_0_arm64 wheel) for all kNN feeding reported metrics, and avoid `hnswlib`, which ships sdist-only and requires a C++ toolchain.
- Apply `unicodedata.normalize('NFKC', ...)` plus zero-width-character stripping to every Chinese query before any hashing, template matching, or vectorization — full-width and half-width variants otherwise land in different template families and different clusters.
- Prefer `TfidfVectorizer(analyzer='char', ngram_range=(2,4), dtype=np.float32, sublinear_tf=True)` over word segmentation for Chinese features: `char_wb` pads at word boundaries and unsegmented Chinese is one word, while dropping jieba entirely removes an unmaintained (last release 2020) dependency from the reproducibility surface.
- Add a P0 smoke test that renders CJK text through matplotlib and asserts the bitmap is non-blank, selecting from the fonts verified present on macOS (`PingFang SC`, `Hiragino Sans GB`, `STHeiti`, `Songti SC`, `Arial Unicode MS`) and raising loudly rather than emitting a report full of tofu boxes.
- Pin the Hugging Face model `revision` to a commit SHA rather than `main`, and hash the full manifest (versions, seed, OMP_NUM_THREADS, prompt string, dtype, eval-index SHA) into every artifact so the quarterly P12 rerun can prove bit-level provenance.
- Pin `sentence-transformers==5.7.0` and adopt the v5.4+ renames (`encode(inputs=)`, `processor_kwargs=`, `get_embedding_dimension()`, `sentence_transformers.sentence_transformer.losses`) now, and note that any bge-m3 or multilingual-e5 corpus embedded on 5.5.0-5.6.0 with flash attention must be re-encoded due to the silent position_ids bug fixed in 5.6.1.

## Unverified or version-dependent

- The MTEB/C-MTEB numbers I cite come from the Qwen3-Embedding model card, which snapshots the leaderboard as of 2025-05-24 / 2025-06-05 — over a year stale. The live https://huggingface.co/spaces/mteb/leaderboard is a Gradio Space that I could not fetch programmatically, so current #1 rankings (Gemini Embedding, jina-embeddings-v4, newer Chinese models like Conan-embedding-v2 or KaLM-Embedding-V2) are unverified. Run your own bake-off on your own data rather than trusting any leaderboard ordering.
- I could not verify jina-embeddings-v3/v4 or gte/e5 current versions and API details directly — searches surfaced only secondary blog sources. Treat any claim about those families as unconfirmed.
- sentence-transformers v6.0 is documented in the master migration guide (MultiVectorEncoder, requires transformers v5 / torch 2.2+ / huggingface-hub v1, custom module classes require trust_remote_code=True even locally) but 5.7.0 is the latest on PyPI. Whether v6.0 has shipped since 2026-08-06 and what its final breaking-change list is are unverified.
- The claim that OMP_NUM_THREADS can change KMeans results in the last bits via non-associative floating-point reduction is well-established behaviour but I could not find it stated explicitly in the current sklearn 1.9 docs — the parallelism page covers oversubscription and thread control but not determinism. Verify empirically on your data before asserting it in a report.
- Apple Silicon throughput numbers are not benchmarked here. The sentence-transformers benchmark hardware is an RTX 3090 / i7-13700K, so its ONNX/OpenVINO/int8 speedup ratios (3.23x onnx-qint8, 5.29x openvino-qint8) are x86-derived and may not transfer to arm64; the arm64 quantization config is documented as roughly equivalent to avx512 but this is the author's single-machine observation.
- faiss-cpu 1.15.0 ships only a macosx_14_0_arm64 wheel for Apple Silicon, so macOS 13 or earlier will fall back to building from source. Not tested here.
- hdbscan 0.8.44 declares scikit-learn>=1.6 with no upper bound; actual compatibility with sklearn 1.9.0 and numpy 2.5.x is inferred from the declared ranges, not tested. The contrib package has historically broken on sklearn minor bumps.
- umap-learn's readthedocs still renders as '0.5.8 documentation' even at /en/latest/, so the prose I quoted (clustering caveats, reproducibility timings) may lag the 0.5.12 code. The n_jobs/random_state override was verified directly in master source, but other doc claims were not.
- I did not verify current best practice for Chinese full-width/traditional normalization libraries (opencc-python-reimplemented version and maintenance status is unchecked), nor benchmark char n-gram vs. word features on actual Chinese query logs — the recommendation to skip segmentation is a reasoned inference, not a measured result.
- The two-subsample overlap ARI protocol, the sphere-null gap statistic correction, the consensus/PAC approach, and the ECE implementation are my own compositions from standard method descriptions, not copied from a verified reference implementation. They should be unit-tested against synthetic data with known structure before being trusted as a gate.
- Whether pandas 3.0.5 introduces further breaks in the sklearn interop path beyond the StringDtype check_array fix (#33491) is unverified; sklearn 1.9 also adopted narwhals as a new dependency for dataframe handling, and the interaction of narwhals with pandas 3 at scale is untested here.
- CalibratedClassifierCV's `ensemble='auto'` default semantics (when 'auto' resolves to True vs False) were not fully extracted from the docs — verify before relying on the ensembling behaviour for the P10 calibration node.

## Sources

- https://pypi.org/pypi/scikit-learn/json
- https://pypi.org/pypi/scikit-learn/1.9.0/json
- https://pypi.org/pypi/sentence-transformers/json
- https://pypi.org/pypi/sentence-transformers/5.7.0/json
- https://pypi.org/pypi/umap-learn/0.5.12/json
- https://pypi.org/pypi/hdbscan/0.8.44/json
- https://pypi.org/pypi/numba/0.67.0/json
- https://pypi.org/pypi/faiss-cpu/1.15.0/json
- https://pypi.org/pypi/hnswlib/json
- https://pypi.org/pypi/jieba/json
- https://pypi.org/pypi/pkuseg/json
- https://pypi.org/pypi/spacy-pkuseg/json
- https://pypi.org/pypi/hanlp/json
- https://pypi.org/pypi/ltp/json
- https://pypi.org/pypi/torch/json
- https://pypi.org/pypi/transformers/json
- https://pypi.org/pypi/numpy/json
- https://pypi.org/pypi/pandas/json
- https://scikit-learn.org/stable/whats_new/v1.9.html
- https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
- https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html
- https://scikit-learn.org/stable/modules/generated/sklearn.cluster.BisectingKMeans.html
- https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html
- https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html
- https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html
- https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html
- https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html
- https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html
- https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
- https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
- https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html
- https://scikit-learn.org/stable/computing/parallelism.html
- https://scikit-learn.org/stable/faq.html
- https://raw.githubusercontent.com/UKPLab/sentence-transformers/master/docs/sentence_transformer/usage/efficiency.rst
- https://raw.githubusercontent.com/UKPLab/sentence-transformers/master/docs/migration_guide.md
- https://raw.githubusercontent.com/UKPLab/sentence-transformers/master/sentence_transformers/sentence_transformer/model.py
- https://api.github.com/repos/UKPLab/sentence-transformers/releases
- https://github.com/huggingface/sentence-transformers/releases/tag/v5.1.0
- https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/raw/main/README.md
- https://huggingface.co/BAAI/bge-m3/raw/main/README.md
- https://huggingface.co/BAAI/bge-base-zh-v1.5/raw/main/config.json
- https://arxiv.org/abs/2506.05176
- https://raw.githubusercontent.com/lmcinnes/umap/master/umap/umap_.py
- https://umap-learn.readthedocs.io/en/latest/clustering.html
- https://umap-learn.readthedocs.io/en/latest/reproducibility.html
- https://umap-learn.readthedocs.io/en/latest/release_notes.html
- https://hdbscan.readthedocs.io/en/latest/api.html
- https://raw.githubusercontent.com/thuiar/DeepAligned-Clustering/main/DeepAligned.py
- https://raw.githubusercontent.com/thuiar/DeepAligned-Clustering/main/dataloader.py
- https://raw.githubusercontent.com/thuiar/DeepAligned-Clustering/main/init_parameter.py
- https://arxiv.org/abs/2012.08987
- https://arxiv.org/abs/1007.1075
- https://people.eecs.berkeley.edu/~jordan/sail/readings/luxburg_ftml.pdf
- https://github.com/lmcinnes/umap/issues/1080
- https://github.com/pytorch/pytorch/issues/84516
- https://github.com/pytorch/pytorch/issues/122394
- https://qwenlm.github.io/blog/qwen3-embedding/