"""Phases 2c and 10 — the rules+ML hybrid classifier, and centroid deployment.

Two models with different jobs.

The **top-down classifier** predicts human-defined intents from a gold set.  Its
feature recipe is deliberately heterogeneous — character n-grams, word n-grams,
regex hit flags, surface numerics, and the dense embedding, concatenated.  The
sparse blocks carry phrasing, the dense block carries meaning, and the gain from
having both is real rather than decorative.  The head is **linear**, and that is
not a default: gradient-boosted trees fed raw embedding coordinates collapse on
this task, because a tree must reconstruct directional similarity out of
axis-aligned splits and a large label set spreads the per-class boosting budget
far too thin.

The **bottom-up classifier** is not trained at all.  A cluster *is* its centroid,
so classification is one matrix product: ``argmax(x @ C.T)``.  The deployed model
is a few hundred kilobytes and its predictions are, by construction, exactly the
assignment rule the tree was built with — no train/serve skew is possible.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, normalize

from ..determinism import SEED_METRIC


# ==========================================================================
# Agreement
# ==========================================================================

UNLABELED = "UNLABELED"
"""Sentinel for "this annotator returned nothing for this row"."""


def agreement(labels_a: Sequence[str], labels_b: Sequence[str]) -> dict[str, Any]:
    """Raw agreement and Cohen's kappa between two independent annotators.

    ``replace_undefined_by=0.0`` is deliberate.  Kappa is undefined when an
    annotator uses a single label for everything, and the default NaN would
    propagate silently into a mean and past the gate.  Scoring that degenerate
    case as 0 makes it fail loudly, which is what it deserves.

    Both numbers are reported because kappa alone misleads under skew: on a
    corpus where one class holds most of the traffic, high raw agreement can
    coexist with mediocre kappa, and the two together tell you which situation
    you are in.
    """
    a_all, b_all = list(labels_a), list(labels_b)
    if not a_all or len(a_all) != len(b_all):
        return {"n": 0, "raw_agreement": float("nan"), "kappa": float("nan")}

    # A missing answer is not a disagreement. When an annotator's batch fails or
    # its response omits a query, the caller fills UNLABELED — scoring that as
    # "the two annotators disagree" charges an infrastructure failure to the
    # methodology and depresses a *blocking* metric. Those rows are excluded and
    # counted separately, so a coverage problem reads as a coverage problem.
    keep = [i for i, (x, y) in enumerate(zip(a_all, b_all))
            if x != UNLABELED and y != UNLABELED]
    n_unlabelled = len(a_all) - len(keep)
    a = [a_all[i] for i in keep]
    b = [b_all[i] for i in keep]
    if not a:
        return {"n": 0, "raw_agreement": float("nan"), "kappa": float("nan"),
                "n_unscored_unlabelled": n_unlabelled,
                "note": "every row was unlabelled — the annotators did not run"}
    raw = float(np.mean([x == y for x, y in zip(a, b)]))
    try:
        k = float(cohen_kappa_score(a, b, replace_undefined_by=0.0))
    except TypeError:  # older sklearn without the parameter
        k = float(cohen_kappa_score(a, b))
        if np.isnan(k):
            k = 0.0
    disagreements = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    return {
        "n": len(a),
        "n_submitted": len(a_all),
        "n_unscored_unlabelled": n_unlabelled,
        "raw_agreement": round(raw, 4),
        "kappa": round(k, 4),
        "n_disagreements": len(disagreements),
        "disagreement_indices": disagreements,
        "n_classes_a": len(set(a)),
        "n_classes_b": len(set(b)),
    }


# ==========================================================================
# Rule engine
# ==========================================================================

class RuleEngine:
    """High-precision regex rules that fire before the model does.

    Rules earn their place by precision on the gold set, not by how clever they
    look.  Anything below the floor is *kept as a feature* but not allowed to
    decide a label: a pattern that is 80% accurate is genuinely informative and
    genuinely unsafe to obey.
    """

    def __init__(self, rules: Sequence[dict[str, str]], *, precision_floor: float = 0.98):
        self.rules = list(rules)
        self.precision_floor = precision_floor
        self.validated: list[dict[str, Any]] = []

    def validate(self, queries: Sequence[str], gold: Sequence[str]) -> list[dict[str, Any]]:
        s = pd.Series(list(queries), dtype="string")
        g = np.array(list(gold))
        out = []
        for r in self.rules:
            m = s.str.contains(r["pattern"], regex=True, na=False).to_numpy()
            n = int(m.sum())
            if n == 0:
                out.append({**r, "n_hits": 0, "precision": float("nan"), "accepted": False,
                            "reason": "no hits in gold"})
                continue
            prec = float((g[m] == r["label"]).mean())
            accepted = prec >= self.precision_floor
            out.append({
                **r, "n_hits": n, "precision": round(prec, 4), "accepted": accepted,
                "reason": "" if accepted else f"precision {prec:.3f} < floor {self.precision_floor}",
            })
        self.validated = out
        return out

    def accepted_rules(self) -> list[dict[str, Any]]:
        return [r for r in self.validated if r.get("accepted")]

    def apply(self, queries: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(labels, fired)`` where unfired rows hold ``''``."""
        s = pd.Series(list(queries), dtype="string")
        labels = np.array([""] * len(s), dtype=object)
        fired = np.zeros(len(s), dtype=bool)
        for r in self.accepted_rules():
            m = s.str.contains(r["pattern"], regex=True, na=False).to_numpy() & ~fired
            labels[m] = r["label"]
            fired |= m
        return labels, fired

    def feature_matrix(self, queries: Sequence[str]) -> np.ndarray:
        """One binary column per rule — informative even when precision is low."""
        s = pd.Series(list(queries), dtype="string")
        if not self.rules:
            return np.zeros((len(s), 0), dtype=np.float32)
        return np.column_stack([
            s.str.contains(r["pattern"], regex=True, na=False).to_numpy().astype(np.float32)
            for r in self.rules
        ])


# ==========================================================================
# Feature assembly
# ==========================================================================

SURFACE_COLUMNS = [
    "len", "n_han", "n_lat", "n_dig", "n_punct", "n_space",
    "r_han", "has_lat", "has_dig", "has_punct", "has_qmark", "is_mixed_script",
]


def build_features(
    df: pd.DataFrame,
    dense: np.ndarray,
    *,
    sparse_blocks: Sequence[np.ndarray] = (),
    rule_features: np.ndarray | None = None,
    scaler: StandardScaler | None = None,
) -> tuple[np.ndarray, StandardScaler]:
    """Concatenate dense semantics, compressed phrasing, rule flags, and surface stats."""
    cols = [c for c in SURFACE_COLUMNS if c in df.columns]
    surf = df[cols].to_numpy(dtype=np.float32)
    if scaler is None:
        scaler = StandardScaler().fit(surf)
    blocks: list[np.ndarray] = [dense.astype(np.float32), normalize(scaler.transform(surf)).astype(np.float32)]
    blocks += [np.asarray(b, dtype=np.float32) for b in sparse_blocks]
    if rule_features is not None and rule_features.size:
        blocks.append(rule_features.astype(np.float32))
    return np.hstack(blocks), scaler


def train_classifier(
    X: np.ndarray,
    y: Sequence[str],
    *,
    C_grid: Sequence[float] = (0.5, 1.0, 2.0, 4.0, 8.0),
    folds: int = 5,
    seed: int = SEED_METRIC,
    weights: np.ndarray | None = None,
) -> dict[str, Any]:
    """Fit a linear head with a small C sweep, reporting cross-validated quality.

    ``weights`` carries full-log frequencies when available, so the headline
    accuracy is population-weighted: getting the head of the distribution right
    matters more than getting an equal number of rare shapes right, and an
    unweighted number quietly says otherwise.
    """
    y = np.asarray(list(y))
    counts = pd.Series(y).value_counts()
    keep = np.isin(y, counts[counts >= folds].index)
    Xk, yk = X[keep], y[keep]
    if len(np.unique(yk)) < 2:
        raise ValueError("need at least two classes with enough support to train")

    cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
    best: dict[str, Any] = {"C": None, "accuracy": -1.0}
    for C in C_grid:
        pred = cross_val_predict(LogisticRegression(max_iter=2000, C=C), Xk, yk, cv=cv)
        acc = float(accuracy_score(yk, pred))
        if acc > best["accuracy"]:
            best = {"C": C, "accuracy": round(acc, 4), "pred": pred}

    model = LogisticRegression(max_iter=3000, C=best["C"]).fit(Xk, yk)
    pred = best.pop("pred")
    w = weights[keep] if weights is not None else None
    result: dict[str, Any] = {
        "model": model,
        "C": best["C"],
        "cv_accuracy": best["accuracy"],
        "n_train": int(keep.sum()),
        "n_dropped_rare": int((~keep).sum()),
        "n_classes": int(len(np.unique(yk))),
        "classes": model.classes_.tolist(),
        "report": classification_report(yk, pred, zero_division=0, output_dict=True),
        "macro_f1": round(float(classification_report(yk, pred, zero_division=0, output_dict=True)["macro avg"]["f1-score"]), 4),
    }
    if w is not None:
        result["population_weighted_accuracy"] = round(
            float(np.average((pred == yk).astype(float), weights=w)), 4
        )
    result["ece"] = expected_calibration_error(model, Xk, yk)
    return result


def expected_calibration_error(model: Any, X: np.ndarray, y: Sequence[str], *, bins: int = 10) -> float:
    """How far the model's stated confidence is from its actual accuracy.

    Needed because Phase 10 routes on confidence.  A model that says 0.9 and is
    right 0.6 of the time makes the routing threshold meaningless.
    """
    proba = model.predict_proba(X)
    conf = proba.max(1)
    pred = model.classes_[proba.argmax(1)]
    correct = (pred == np.asarray(list(y))).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for i in range(bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.sum() == 0:
            continue
        ece += (m.mean()) * abs(correct[m].mean() - conf[m].mean())
    return round(float(ece), 4)


# ==========================================================================
# Phase 10 — centroid deployment
# ==========================================================================

class CentroidClassifier:
    """The whole deployed bottom-up model: a centroid matrix and an argmax.

    Serialises to a few hundred kilobytes and cannot drift from the tree it came
    from, because it *is* the tree's assignment rule rather than an approximation
    fitted to the tree's output.
    """

    def __init__(
        self,
        centroids: np.ndarray,
        leaf_family: np.ndarray,
        *,
        names: dict[int, str] | None = None,
        margin_threshold: float = 0.02,
        alpha: float = 0.0,
    ) -> None:
        self.centroids = normalize(np.asarray(centroids, dtype=np.float32))
        self.leaf_family = np.asarray(leaf_family, dtype=np.int64)
        self.names = names or {}
        self.margin_threshold = margin_threshold
        self.alpha = alpha

    def predict(self, X: np.ndarray) -> dict[str, np.ndarray]:
        sims = np.asarray(X, dtype=np.float32) @ self.centroids.T
        order = np.argsort(-sims, axis=1)
        leaf = order[:, 0]
        top1 = sims[np.arange(len(sims)), order[:, 0]]
        top2 = sims[np.arange(len(sims)), order[:, 1]] if sims.shape[1] > 1 else np.zeros(len(sims))
        margin = top1 - top2
        return {
            "leaf": leaf,
            "family": self.leaf_family[leaf],
            "score": top1,
            "margin": margin,
            "ambiguous": margin < self.margin_threshold,
            "runner_up_leaf": order[:, 1] if sims.shape[1] > 1 else leaf,
        }

    def route(self, X: np.ndarray) -> dict[str, Any]:
        """Split predictions into direct-serve and fallback, and say how many.

        The ambiguous share is reported, not hidden.  Semantic boundaries are
        genuinely softer than phrasing boundaries, so a double-digit ambiguous
        rate is a property of the problem — concealing it would just move the
        surprise to production.
        """
        p = self.predict(X)
        amb = p["ambiguous"]
        return {
            "predictions": p,
            "n_direct": int((~amb).sum()),
            "n_fallback": int(amb.sum()),
            "ambiguous_rate": round(float(amb.mean()), 4),
            "threshold": self.margin_threshold,
            "policy": (
                "rows below the margin threshold route to a fallback path "
                "(human review or a larger model); the rest serve directly"
            ),
        }

    def size_bytes(self) -> int:
        return int(self.centroids.nbytes + self.leaf_family.nbytes)


# ==========================================================================
# Phase 2d step 1 — kNN label-error scan
# ==========================================================================

def knn_label_scan(
    X: np.ndarray,
    indices: Sequence[int],
    labels: Sequence[str],
    *,
    k: int = 10,
    disagreement_threshold: float = 0.8,
) -> dict[str, Any]:
    """Flag gold rows whose neighbourhood disagrees with their label.

    **These flags are candidates for human review, never automatic relabels.**
    That restriction is the whole lesson of this step: in the source project 134
    rows were flagged and only 6–7 were real annotation errors. The other ~89
    were an artifact — queries sharing a phrasing template cluster so tightly in
    the embedding that they out-vote their own correct labels. Auto-applying the
    flags would have corrupted the gold set to make it agree with a known
    representation bug.

    The artifact is not noise, either. Discovering that a template pins a
    neighbourhood harder than meaning does is precisely the evidence that
    motivates the hybrid representation and its alpha sweep, so the flags are
    kept and reported rather than discarded.
    """
    idx = np.asarray(list(indices), dtype=np.int64)
    y = np.asarray(list(labels))
    if idx.size < k + 1:
        return {"flags": [], "n_flagged": 0, "note": "too few labelled rows to scan"}

    Xg = X[idx]
    sims = Xg @ Xg.T
    np.fill_diagonal(sims, -np.inf)
    kk = min(k, Xg.shape[0] - 1)
    top = np.argpartition(-sims, kth=kk - 1, axis=1)[:, :kk]

    flags: list[dict[str, Any]] = []
    for i in range(len(idx)):
        neigh = y[top[i]]
        disagree = float((neigh != y[i]).mean())
        if disagree < disagreement_threshold:
            continue
        vals, counts = np.unique(neigh, return_counts=True)
        majority = str(vals[int(counts.argmax())])
        flags.append({
            "row": int(idx[i]),
            "label": str(y[i]),
            "neighbour_majority": majority,
            "disagreement": round(disagree, 3),
            "neighbour_labels": [str(v) for v in neigh[:6]],
        })

    flags.sort(key=lambda f: -f["disagreement"])
    return {
        "flags": flags,
        "n_flagged": len(flags),
        "n_scanned": int(idx.size),
        "flag_rate": round(len(flags) / max(idx.size, 1), 4),
        "k": kk,
        "action": "manual_review_only",
        "warning": (
            "Do NOT auto-apply these. Most flags in a templated corpus are "
            "representation artifacts — phrasing-tight neighbourhoods out-voting "
            "correct labels — not annotation errors. Route them to a reviewer, and "
            "read a high flag rate as evidence about the representation."
        ),
    }


# ==========================================================================
# Phase 2b step 4 — active learning round 2
# ==========================================================================

def select_active_learning_batch(
    model: Any,
    X: np.ndarray,
    *,
    already_labelled: Sequence[int],
    batch: int = 200,
    seed: int = SEED_METRIC,
    diversity_fraction: float = 0.3,
) -> dict[str, Any]:
    """Pick the next rows to annotate: the ones the model is least sure about.

    Two signals, deliberately combined. **Margin uncertainty** (top-1 minus
    top-2 probability) finds rows sitting on a decision boundary, which is where
    a new label teaches the most. But pure uncertainty sampling collapses onto
    one confusing region and re-annotates fifty variants of the same edge case,
    so a **diversity** quota spends part of the budget on uncertain rows that are
    far from the ones already chosen.
    """
    labelled = set(int(i) for i in already_labelled)
    # `.shape[0]`, not `len(X)`: the caller passes a char-TFIDF matrix, and
    # scipy refuses len() on sparse ('length is ambiguous'). That TypeError was
    # swallowed upstream, so this whole round silently never ran.
    pool = np.array([i for i in range(X.shape[0]) if i not in labelled], dtype=np.int64)
    if pool.size == 0:
        return {"selected": [], "n": 0, "note": "no unlabelled rows remain"}

    proba = model.predict_proba(X[pool])
    part = np.partition(proba, -2, axis=1)
    margin = part[:, -1] - part[:, -2]

    n_unc = int(batch * (1 - diversity_fraction)) if diversity_fraction else batch
    order = np.argsort(margin)
    chosen = list(pool[order[:n_unc]])

    # diversity half: uncertain, but far from what we already picked
    remaining = [int(i) for i in pool[order[n_unc : n_unc * 4]]]
    if remaining and len(chosen) < batch:
        picked_vecs = normalize(X[np.array(chosen)]) if chosen else None
        scored = []
        for i in remaining:
            if picked_vecs is None:
                scored.append((0.0, i))
            else:
                scored.append((float((X[i] @ picked_vecs.T).max()), i))
        scored.sort()  # lowest max-similarity = most novel
        chosen += [i for _, i in scored[: batch - len(chosen)]]

    sel = sorted(int(i) for i in chosen[:batch])
    return {
        "selected": sel,
        "n": len(sel),
        "median_margin_selected": round(float(np.median(margin[order[: len(sel)]])), 4),
        "strategy": (
            f"{int((1 - diversity_fraction) * 100)}% lowest-margin (boundary) + "
            f"{int(diversity_fraction * 100)}% uncertain-and-distant (diversity), so the "
            "batch does not collapse onto a single confusing region"
        ),
    }


# ==========================================================================
# Phase 2b step 5 — deciding the boundaries the referee could not settle
# ==========================================================================

def contested_boundaries(rows: Sequence[Any]) -> list[dict[str, Any]]:
    """Class pairs the referee resolved *inconsistently* across the gold set.

    A pair the referee always sent the same way is settled — a rule can be
    written from it. A pair it sent both ways is an open boundary, and no rule
    can be written for a boundary nobody has decided. On the run that motivated
    this, 30 of 42 disagreeing pairs were settled and 12 were not; resolving only
    the settled ones lifts kappa to 0.8978, short of the 0.90 gate, while the
    unsettled dozen holds the remaining 55 disagreements.
    """
    seen: dict[frozenset[str], list[str]] = {}
    for r in rows:
        if getattr(r, "agreed", False) or not getattr(r, "final", ""):
            continue
        key = frozenset((r.label_a, r.label_b))
        if len(key) == 2:
            seen.setdefault(key, []).append(r.final)
    out = []
    for key, finals in seen.items():
        counts = Counter(finals)
        if len(counts) > 1:
            out.append({"pair": sorted(key), "n": len(finals),
                        "resolved_as": dict(counts)})
    return sorted(out, key=lambda d: -d["n"])


def discriminating_markers(
    rows: Sequence[Any], pair: Sequence[str], *,
    min_support: int = 4, min_precision: float = 0.90, max_markers: int = 4,
) -> list[dict[str, Any]]:
    """Substrings that separate two classes, learned from rows both annotators agreed on.

    The agreed rows are the only labels in the set with no arbitration in them,
    which makes them the right evidence for settling a boundary the referee kept
    flip-flopping on. Deciding it from this evidence beats asking another model
    for an opinion: the resulting rule is checkable, reproducible, and stated in
    the corpus's own vocabulary. On the motivating corpus the marker 意思 held
    59/59 for WORD_MEANING over IDIOM_PHRASE among agreed rows — a discriminator
    the referee overrode in three separate verdicts.
    """
    a, b = pair[0], pair[1]
    pool = [r for r in rows if getattr(r, "agreed", False) and r.final in (a, b)]
    if len(pool) < min_support * 2:
        return []

    counts: dict[str, Counter] = {}
    for r in pool:
        q = str(r.query)
        # Character n-grams: the corpus may have no whitespace to tokenise on.
        grams = {q[i : i + n] for n in (2, 3, 4) for i in range(len(q) - n + 1)}
        for g in grams:
            counts.setdefault(g, Counter())[r.final] += 1

    out = []
    for gram, c in counts.items():
        total = c[a] + c[b]
        if total < min_support:
            continue
        winner, n_win = (a, c[a]) if c[a] >= c[b] else (b, c[b])
        precision = n_win / total
        if precision >= min_precision:
            out.append({"marker": gram, "then": winner, "support": total,
                        "precision": round(precision, 3)})

    # Greedy set cover, not a substring filter. Character n-grams necessarily
    # produce overlapping fragments of one pattern (意思, 什么意, 么意思, 是什么意…),
    # and no containment rule separates the useful short form from the noise —
    # 意思 and 的意 overlap in neither direction yet say the same thing. Selecting
    # by how many *not-yet-covered* rows a marker explains sidesteps that: a
    # fragment of an already-chosen marker covers nothing new and drops out on
    # its own, and what survives is the smallest set that reaches the most rows.
    by_marker = {c["marker"]: c for c in out}
    matches = {m: {i for i, r in enumerate(pool) if m in str(r.query)} for m in by_marker}
    covered: set[int] = set()
    chosen: list[dict[str, Any]] = []
    while len(chosen) < max_markers:
        taken = {k["marker"] for k in chosen}
        # Rank every remaining candidate outright rather than tracking a running
        # best: ties on coverage are common (的意 and 意思 both explain the same
        # twelve rows), and resolving them by dict order makes the rule set
        # depend on insertion order. Shortest wins, then lexicographic — so the
        # same corpus always yields the same guide.
        ranked = sorted(
            ((len(matches[m] - covered), -len(m), [-ord(ch) for ch in m], c)
             for m, c in by_marker.items() if m not in taken),
            reverse=True,
        )
        if not ranked:
            break
        best_gain, _, _, best = ranked[0]
        if best_gain < min_support:
            break
        chosen.append(best)
        covered |= matches[best["marker"]]
    return chosen


def boundary_default(
    rows: Sequence[Any], pair: Sequence[str], markers: Sequence[str] = (),
    *, min_support: int = 20, min_precision: float = 0.75,
) -> dict[str, Any] | None:
    """A tie-breaker for rows on this boundary that carry no marker at all.

    Markers cannot reach the hardest rows. On the motivating corpus 64% of all
    disagreements were queries with no intent marker whatsoever — a bare
    four-character idiom like 飞檐走壁 states its object and not what the user
    wants to know about it, so an annotator with no default must guess, and two
    annotators guess differently.

    The default is taken from the marker-less rows both annotators agreed on,
    and only when that majority is decisive: a 64% lean is noise dressed as a
    rule and would manufacture as much disagreement as it settles. Returns
    ``None`` when the evidence does not clear the floor, which leaves the
    boundary honestly open rather than closing it on a coin flip.
    """
    a, b = pair[0], pair[1]
    bare = [r for r in rows
            if getattr(r, "agreed", False) and r.final in (a, b)
            and not any(m in str(r.query) for m in markers)]
    if len(bare) < min_support:
        return None
    counts = Counter(r.final for r in bare)
    winner, n = counts.most_common(1)[0]
    precision = n / len(bare)
    if precision < min_precision:
        return None
    return {"then": winner, "support": len(bare), "precision": round(precision, 3)}
