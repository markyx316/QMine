"""Phase 3 — representations: dense semantics, sparse phrasing, and the dial between them.

The load-bearing idea of this phase, and arguably of the whole playbook, is the
hybrid concatenation::

    H = L2norm([ e  ⊕  α·s ])

where ``e`` is a unit-norm sentence embedding and ``s`` is a unit-norm SVD
compression of character TF-IDF.  Because both blocks are unit vectors, the
cosine between two hybrid vectors expands to

    cos(H, H′) = (cos_semantic + α² · cos_surface) / (1 + α²)

so the *phrasing block votes with weight α², not α*.  That squared term is the
whole reason a value like 0.1 is useful and a value like 0.5 is destructive:

* α = 0.5 → phrasing holds 20% of the vote, which is more than enough to
  outvote semantics on near-ties and split one intent into several
  phrasing-shaped families ("template twins");
* α = 0.1 → phrasing holds about 1%, which does nothing at all except break
  semantic ties — pulling identically-phrased queries back together exactly
  where the embedding could not decide.

That is the behaviour we want: a tie-breaker, not a co-equal signal.  The sweep
in :func:`alpha_sweep` re-derives the value per domain, because the K12 answer
is an answer about K12 phrasing ecology, not a constant of nature.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any, Sequence

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from ..determinism import SEED_METRIC

log = logging.getLogger("qmine.represent")


# ==========================================================================
# Dense encoders
# ==========================================================================

class HashingEncoder:
    """A deterministic, dependency-free stand-in for a sentence encoder.

    It is a *character n-gram hashing* projection, not a semantic model: it
    captures surface similarity and nothing else.  Its job is to keep the graph
    runnable on a machine with no torch and no network, so that wiring bugs
    surface in CI rather than in a two-hour GPU run.  Any artifact it produces
    is stamped ``hashing`` so no reader mistakes it for semantics.
    """

    def __init__(self, dim: int = 256, ngram: tuple[int, int] = (1, 3), seed: int = SEED_METRIC):
        self.dim = dim
        self.ngram = ngram
        self.seed = seed
        self.name = f"hashing-{dim}d"

    def encode(self, texts: Sequence[str], **_: Any) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        lo, hi = self.ngram
        for i, t in enumerate(texts):
            for n in range(lo, hi + 1):
                for j in range(max(len(t) - n + 1, 0)):
                    g = t[j : j + n]
                    h = int(hashlib.md5(f"{self.seed}:{g}".encode("utf-8")).hexdigest()[:8], 16)
                    out[i, h % self.dim] += 1.0 if h % 2 else -1.0
        return normalize(out)


def load_encoder(name: str, *, offline: bool = False, cache_folder: str | None = None) -> Any:
    """Return an object with ``.encode(list[str]) -> np.ndarray``."""
    if offline or name.startswith("hashing"):
        dim = int(re.sub(r"\D", "", name) or 256)
        return HashingEncoder(dim=dim)
    from sentence_transformers import SentenceTransformer

    kw: dict[str, Any] = {}
    if cache_folder:
        kw["cache_folder"] = cache_folder
    model = SentenceTransformer(name, **kw)
    model.name = name  # type: ignore[attr-defined]
    return model


def encode_corpus(
    encoder: Any,
    texts: Sequence[str],
    *,
    batch_size: int = 256,
    instruction: str | None = None,
    show_progress: bool = False,
) -> np.ndarray:
    """Encode and L2-normalise.

    Normalisation is not optional here: every downstream distance is cosine, the
    hybrid algebra above assumes unit blocks, and KMeans on a unit sphere is the
    approximation that makes the whole Phase 4 result hold.
    """
    inputs = [f"{instruction}{t}" for t in texts] if instruction else list(texts)
    if hasattr(encoder, "encode") and encoder.__class__.__name__ == "SentenceTransformer":
        emb = encoder.encode(
            inputs,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        )
    else:
        emb = encoder.encode(inputs)
    return normalize(np.asarray(emb, dtype=np.float32))


# ==========================================================================
# Sparse block
# ==========================================================================

def tokenize(texts: Sequence[str], tokenizer: str = "jieba") -> list[str]:
    """Whitespace-joined tokens, used for the word-level TF-IDF block."""
    if tokenizer == "jieba":
        try:
            import jieba

            jieba.setLogLevel(logging.ERROR)
            return [" ".join(jieba.lcut(t)) for t in texts]
        except Exception:
            log.warning("jieba unavailable; falling back to character tokenisation")
            return [" ".join(t) for t in texts]
    if tokenizer == "whitespace":
        return [" ".join(re.findall(r"\w+", t.lower())) for t in texts]
    return [" ".join(t) for t in texts]


def build_sparse(
    texts: Sequence[str],
    *,
    analyzer: str = "char",
    ngram_range: tuple[int, int] = (1, 3),
    min_df: int = 2,
    sublinear_tf: bool = True,
    svd_dims: int = 256,
    seed: int = SEED_METRIC,
    tokenizer: str = "jieba",
) -> dict[str, Any]:
    """Character (or word) TF-IDF, then SVD down to a concatenable dense block.

    The SVD is compression, not modelling — it exists so a 60,000-dimension
    sparse matrix can sit next to a 768-dimension dense one in a single cosine
    space.  At 256 components it typically retains enough of the phrasing signal
    that fragmentation scores move the way the raw sparse matrix would.
    """
    if analyzer == "word":
        docs = tokenize(texts, tokenizer)
        vec = TfidfVectorizer(
            ngram_range=ngram_range, min_df=min_df, sublinear_tf=sublinear_tf, token_pattern=r"\S+"
        )
    else:
        docs = list(texts)
        vec = TfidfVectorizer(
            analyzer="char", ngram_range=ngram_range, min_df=min_df, sublinear_tf=sublinear_tf
        )
    X = vec.fit_transform(docs)
    k = min(svd_dims, max(X.shape[1] - 1, 2), max(X.shape[0] - 1, 2))
    svd = TruncatedSVD(n_components=k, random_state=seed)
    Z = normalize(svd.fit_transform(X).astype(np.float32))
    return {
        "matrix": X,
        "svd_block": Z,
        "vectorizer": vec,
        "svd": svd,
        "vocab_size": int(X.shape[1]),
        "explained_variance": float(svd.explained_variance_ratio_.sum()),
        "n_components": int(k),
    }


# ==========================================================================
# Hybrid
# ==========================================================================

def hybrid(dense: np.ndarray, sparse_block: np.ndarray, alpha: float) -> np.ndarray:
    """``L2norm([e ⊕ α·s])`` — see the module docstring for why α is squared."""
    if alpha == 0:
        return normalize(dense.astype(np.float32))
    return normalize(
        np.hstack([dense, (alpha * sparse_block).astype(np.float32)]).astype(np.float32)
    )


def surface_vote_share(alpha: float) -> float:
    """The fraction of the cosine that the phrasing block actually controls.

    Printed in every alpha report, because ``α=0.5`` reads like "half" to a
    reader and means 20%, and that gap is exactly where the K12 project lost a
    week to template twins.
    """
    return alpha**2 / (1 + alpha**2)


def alpha_sweep(
    dense: np.ndarray,
    sparse_block: np.ndarray,
    *,
    alphas: Sequence[float],
    k: int,
    template_masks: dict[str, np.ndarray],
    seeds: tuple[int, int] = (0, 1),
    silhouette_sample: int = 8000,
    reference_labels: np.ndarray | None = None,
    tie_band: float = 0.05,
) -> dict[str, Any]:
    """Sweep α and pick a winner using only decisive metrics.

    The selection rule is the point.  Silhouette is computed, reported, and
    given **no vote**, because it is maximised by exactly the failure mode we are
    trying to avoid: identically-phrased queries are the tightest possible
    cluster, so optimising tightness selects for phrasing-shaped families and
    calls it success.  The vote goes to template fragmentation (does one intent
    stay in one family?) and replay stability (would we get this partition again
    with a different seed?).
    """
    from .cluster import kmeans_labels, replay_stability
    from .templates import template_fragmentation

    rows: list[dict[str, Any]] = []
    for a in alphas:
        H = hybrid(dense, sparse_block, a)
        labels = kmeans_labels(H, k, seed=seeds[0])
        frag = template_fragmentation(labels, template_masks)
        stab = replay_stability(H, k, seeds=seeds, sample=silhouette_sample)
        sil = _silhouette(H, labels, sample=silhouette_sample)
        row: dict[str, Any] = {
            "alpha": a,
            "surface_vote_share": round(surface_vote_share(a), 4),
            "template_fragmentation": frag["mean_fragmentation"],
            "stability_ari": stab,
            "silhouette": sil,
            "per_group_fragmentation": frag["per_group"],
        }
        if reference_labels is not None:
            from sklearn.metrics import normalized_mutual_info_score

            row["nmi_reference"] = round(
                float(normalized_mutual_info_score(reference_labels, labels)), 4
            )
        rows.append(row)

    # --- the decision -----------------------------------------------------
    # The playbook asks for BOTH lowest fragmentation and highest stability, and
    # a strict lexicographic sort does not deliver that: it lets a
    # noise-sized fragmentation difference (2.76 vs 2.89) outrank a large
    # stability difference (0.67 vs 0.90). So fragmentation differences within
    # `tie_band` are treated as ties and broken on stability, which is the
    # sturdier of the two measurements.
    # `tie_band` STAYS A CONSTANT HERE, DELIBERATELY. It is the same class of
    # single-corpus constant that `ami_tie_band` was, and replacing it with the
    # measured `noise_floor` was tried and REVERTED: that estimator reads
    # point-to-point roughness as noise, which holds for the K sweep (a smooth
    # curve) and fails here. Fragmentation across alpha on live39 runs
    # 1.98, 2.02, 2.42, 1.98, 2.41, 2.29, 2.72 — genuinely non-monotone, so the
    # roughness IS signal. The estimate came out at se=0.2205, widening the band
    # from 2.08 to 2.42 and making alpha=0.5 (fragmentation 2.41) "tied" with
    # 1.98; the winner flipped from 0.1 to 0.5 on nothing.
    #
    # Measuring this band properly needs repeated fits at the SAME alpha under
    # different seeds — separating noise from signal requires replication, which
    # a single sweep cannot supply. Until then the constant is honest and its
    # provenance is stated in `chosen_by`.
    best_frag = min(r["template_fragmentation"] for r in rows)
    band = best_frag * (1 + tie_band)
    # NO INTERPOLATION. `prose()` matches a literal prefix, and an f-string whose
    # placeholder lands early has no literal prefix to match — so a translatable
    # sentence must keep its numbers out. The percentage goes out as data below.
    band_src = ("configured relative band, NOT measured — this sweep is non-monotone "
                "in alpha, so its roughness cannot estimate noise")
    contenders = [r for r in rows if r["template_fragmentation"] <= band]
    winner = max(contenders, key=lambda r: (r["stability_ari"], -r["template_fragmentation"]))
    sil_winner = max(rows, key=lambda r: r["silhouette"])
    return {
        "rows": rows,
        "chosen_alpha": winner["alpha"],
        # Numbers as data, not baked into an English sentence. `prose()` matches a
        # prefix and returns a FIXED translation, so a sentence carrying
        # interpolated values can only be shipped untranslated — which is how the
        # English selection rule reached the Chinese report in the first place.
        "tie_band_value": round(float(band), 4),
        "tie_band_source": band_src,
        "tie_band_relative_pct": round(tie_band * 100, 1),
        # SAY WHAT THE RULE DOES, NOT WHAT IT SOUNDS LIKE IT DOES.
        #
        # This read "lowest template_fragmentation within a tie-band, broken on
        # highest stability_ari", which a reader parses as "the winner has the
        # lowest fragmentation". It usually does not: the band admits every alpha
        # within 5% of the minimum and stability then picks among them, so the
        # winner is normally NOT the minimum. live41's observer confirmed it
        # mechanically on the run's own artifacts — chosen alpha=0.1 while
        # alpha=0.0 fragments less — and the same mismatch was in live40.
        "chosen_by": (
            "every alpha within the tie-band of the LOWEST template_fragmentation "
            "is treated as tied; among those the highest stability_ari wins. The "
            "winner is therefore usually not the lowest-fragmentation alpha — that "
            "is the rule working, not a contradiction of it"
        ),
        "tie_band": tie_band,
        "contenders": [r["alpha"] for r in contenders],
        "silhouette_would_have_chosen": sil_winner["alpha"],
        "silhouette_disagrees": sil_winner["alpha"] != winner["alpha"],
        "note": (
            "silhouette has no vote (Principle 3): it is maximised by "
            "phrasing-tight clusters, which is the failure mode this sweep exists to avoid"
        ),
    }


def _silhouette(X: np.ndarray, labels: np.ndarray, *, sample: int = 8000) -> float:
    from sklearn.metrics import silhouette_score

    if len(set(labels.tolist())) < 2:
        return float("nan")
    return round(
        float(
            silhouette_score(
                X, labels, metric="cosine", sample_size=min(sample, len(X)), random_state=SEED_METRIC
            )
        ),
        4,
    )


def encoder_bakeoff(
    texts: Sequence[str],
    candidates: Sequence[str],
    *,
    k: int,
    template_masks: dict[str, np.ndarray],
    subsample: int = 15000,
    seeds: tuple[int, int] = (0, 1),
    reference_labels: np.ndarray | None = None,
    offline: bool = False,
    cache_folder: str | None = None,
    instruction: str | None = None,
) -> dict[str, Any]:
    """Phase 3a — choose the base encoder on *your* clustering task.

    Two results from the source project make this bake-off non-optional.  A
    larger encoder produced a *better classifier* and a *worse clustering* — the
    two abilities are not the same ability, and leaderboard retrieval scores
    measure neither.  And an instruction prefix raised stability while lowering
    silhouette, which is only a contradiction if you thought silhouette was the
    objective.
    """
    from .cluster import kmeans_labels, replay_stability
    from .determinism_helpers import subsample_indices
    from .templates import template_fragmentation

    idx = subsample_indices(len(texts), subsample, SEED_METRIC)
    sub_texts = [texts[i] for i in idx]
    sub_masks = {name: m[idx] for name, m in template_masks.items()}
    sub_ref = reference_labels[idx] if reference_labels is not None else None

    rows: list[dict[str, Any]] = []
    for name in candidates:
        try:
            enc = load_encoder(name, offline=offline, cache_folder=cache_folder)
            E = encode_corpus(enc, sub_texts, instruction=instruction)
        except Exception as exc:  # noqa: BLE001
            log.warning("encoder %s unavailable: %s", name, exc)
            rows.append({"encoder": name, "status": "unavailable", "error": str(exc)[:200]})
            continue
        labels = kmeans_labels(E, k, seed=seeds[0])
        row = {
            "encoder": name,
            "status": "ok",
            "dim": int(E.shape[1]),
            "stability_ari": replay_stability(E, k, seeds=seeds),
            "template_fragmentation": template_fragmentation(labels, sub_masks)["mean_fragmentation"],
            "silhouette": _silhouette(E, labels),
        }
        if sub_ref is not None:
            from sklearn.metrics import normalized_mutual_info_score

            row["nmi_reference"] = round(float(normalized_mutual_info_score(sub_ref, labels)), 4)
        rows.append(row)
        del enc

    ok = [r for r in rows if r.get("status") == "ok"]
    if not ok:
        raise RuntimeError("no encoder candidate could be loaded")
    ranked = sorted(ok, key=lambda r: (-r["stability_ari"], r["template_fragmentation"]))
    best = ranked[0]
    sil_best = max(ok, key=lambda r: r["silhouette"])
    return {
        "rows": rows,
        "chosen_encoder": best["encoder"],
        "chosen_by": "stability_ari desc, then template_fragmentation asc",
        "silhouette_would_have_chosen": sil_best["encoder"],
        "silhouette_disagrees": sil_best["encoder"] != best["encoder"],
        "subsample_size": int(len(idx)),
    }
