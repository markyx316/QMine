"""Language handling for mixed-language query logs.

The situation this module exists for: a log that is mostly one language with a
minority of others — the common case for a Chinese-market product that also sees
English queries.

**The failure it prevents, measured rather than assumed.** On a real corpus at a
2% English minority, clustering with a Chinese-monolingual encoder puts 97% of
all English queries into a single 100%-English cluster. Every English intent,
whatever it was, collapses into one "foreign language" family and the tree
resolves none of them. This is not an encoder defect so much as a fact about
variance: when a minority language is also a minority of the *content*, language
is the largest systematic difference in the data, and KMeans finds it first.

**Why swapping in a multilingual encoder is not automatically the fix.** Two
things are easy to conflate:

* *cross-lingual alignment* — does "苹果的拼音" sit near "how to pronounce apple"?
* *anisotropy* — does everything sit near everything?

A model can improve the first number while destroying the contrast that makes it
useful. On a parallel-intent probe, a Chinese-monolingual encoder scored a
same-intent cosine of 0.50 against a 0.23 baseline (separation +0.26, correct
translation ranked first every time), while a multilingual model scored 0.87
against a 0.76 baseline (separation +0.11, rank-1 90%). The higher absolute
number was the worse representation. This is Principle 3 in a new costume, so
the decision here is made by a *probe run on your own corpus*, never by a
leaderboard or by this docstring.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------

_SCRIPT_RANGES: list[tuple[str, tuple[int, int]]] = [
    ("han", (0x4E00, 0x9FFF)), ("han", (0x3400, 0x4DBF)), ("han", (0xF900, 0xFAFF)),
    ("kana", (0x3040, 0x30FF)),
    ("hangul", (0xAC00, 0xD7AF)), ("hangul", (0x1100, 0x11FF)),
    ("latin", (0x0041, 0x024F)),
    ("cyrillic", (0x0400, 0x04FF)),
    ("arabic", (0x0600, 0x06FF)),
    ("devanagari", (0x0900, 0x097F)),
    ("thai", (0x0E00, 0x0E7F)),
    ("hebrew", (0x0590, 0x05FF)),
]


def script_profile(text: str) -> dict[str, float]:
    """Share of each script among the text's letters.

    Script is used rather than a statistical language ID because queries are
    brutally short — a Chinese query averages about seven characters and an
    English one about four words — and short-text language ID is unreliable
    exactly where it matters. Script is a weaker claim that we can actually
    make: it does not distinguish Chinese from Japanese kanji, and it is right
    essentially always about Han-versus-Latin, which is the split that decides
    tokenisation and n-gram range.
    """
    counts: dict[str, int] = {}
    total = 0
    for ch in text:
        if not ch.isalpha():
            continue
        cp = ord(ch)
        total += 1
        for name, (lo, hi) in _SCRIPT_RANGES:
            if lo <= cp <= hi:
                counts[name] = counts.get(name, 0) + 1
                break
        else:
            counts["other"] = counts.get("other", 0) + 1
    if not total:
        return {}
    return {k: v / total for k, v in counts.items()}


def classify_row(text: str, *, mixed_threshold: float = 0.2) -> str:
    """One label per query: a script name, ``mixed``, or ``nonlinguistic``.

    ``nonlinguistic`` matters more than it sounds: bare model numbers, URLs and
    punctuation strings are a real slice of any query log, they belong to no
    language, and letting them be counted as "Latin" inflates the apparent
    foreign-language share and can trigger a stratification that is not needed.
    """
    prof = script_profile(text)
    if not prof:
        return "nonlinguistic"
    ranked = sorted(prof.items(), key=lambda kv: -kv[1])
    top, share = ranked[0]
    if len(ranked) > 1 and ranked[1][1] >= mixed_threshold:
        return f"mixed:{top}+{ranked[1][0]}"
    return top if share >= 0.5 else "mixed"


def profile_corpus(queries: Sequence[str]) -> dict[str, Any]:
    """Language/script composition of the corpus, and what it implies.

    The recommendation returned here is a *hypothesis to test*, not a decision.
    Phase 3 runs the alignment probe and the decisive metrics before committing.
    """
    labels = [classify_row(q) for q in queries]
    n = max(len(labels), 1)
    counts: dict[str, int] = {}
    for lab in labels:
        counts[lab] = counts.get(lab, 0) + 1
    shares = {k: v / n for k, v in sorted(counts.items(), key=lambda kv: -kv[1])}

    linguistic = {k: v for k, v in shares.items() if k != "nonlinguistic"}
    dominant, dom_share = (max(linguistic.items(), key=lambda kv: kv[1])
                           if linguistic else ("unknown", 0.0))
    minority = {k: v for k, v in linguistic.items()
                if k != dominant and not k.startswith("mixed") and v >= 0.005}
    minority_share = sum(minority.values())

    if minority_share < 0.005:
        posture = "monolingual"
        rationale = (
            f"{dominant} accounts for {dom_share:.1%} and no other script reaches 0.5%. "
            "Use a monolingual encoder for the dominant language; there is nothing to stratify."
        )
    elif minority_share < 0.05:
        posture = "minority_at_risk"
        rationale = (
            f"minority scripts total {minority_share:.1%} — small enough that they will be "
            "swallowed into a single junk cluster at a normal K, and large enough that losing "
            "them matters. Either route them to their own sub-tree or accept that their intents "
            "will not be resolved. Do NOT simply hope a multilingual encoder fixes it."
        )
    else:
        posture = "genuinely_multilingual"
        rationale = (
            f"minority scripts total {minority_share:.1%}. Test unified-versus-stratified "
            "explicitly with the decisive metrics; at this share both are defensible and the "
            "data should choose."
        )

    return {
        "shares": shares,
        "dominant": dominant,
        "dominant_share": round(dom_share, 4),
        "minority": {k: round(v, 4) for k, v in minority.items()},
        "minority_share": round(minority_share, 4),
        "nonlinguistic_share": round(shares.get("nonlinguistic", 0.0), 4),
        "posture": posture,
        "rationale": rationale,
        "row_labels": labels,
    }


# --------------------------------------------------------------------------
# The probe that actually decides
# --------------------------------------------------------------------------

def alignment_probe(
    encoder: Any, pairs: Sequence[tuple[str, str]], *, encode: Any = None
) -> dict[str, Any]:
    """Measure cross-lingual alignment as CONTRAST, not as raw similarity.

    Given pairs that mean the same thing in two languages, a useful encoder puts
    each pair closer to each other than to the other pairs. The headline number
    is therefore ``separation`` — same-intent cosine minus different-intent
    cosine — and ``rank1``, the share of queries whose true translation is the
    nearest of all candidates.

    Raw same-intent cosine is reported and deliberately not used to decide,
    because a model with high anisotropy scores well on it while being *less*
    able to tell one intent from another.
    """
    from .represent import encode_corpus

    enc = encode or encode_corpus
    a = enc(encoder, [p[0] for p in pairs])
    b = enc(encoder, [p[1] for p in pairs])
    S = np.asarray(a) @ np.asarray(b).T
    k = len(pairs)
    same = float(np.mean(np.diag(S)))
    diff = float((S.sum() - np.trace(S)) / max(S.size - k, 1))
    return {
        "n_pairs": k,
        "same_intent_cosine": round(same, 4),
        "different_intent_cosine": round(diff, 4),
        "separation": round(same - diff, 4),
        "rank1_accuracy": round(float(np.mean(np.argmax(S, axis=1) == np.arange(k))), 4),
        "decided_on": "separation and rank1_accuracy",
        "note": (
            "Raw same-intent cosine is reported but does not decide. An anisotropic "
            "model scores highly on it while separating intents worse — the same "
            "proxy-betrayal pattern that makes silhouette advisory elsewhere in this pipeline."
        ),
    }


def minority_dilution(
    labels: np.ndarray, row_language: Sequence[str], minority: str
) -> dict[str, Any]:
    """How badly a minority language collapsed under a given partition.

    ``concentration`` is the share of all minority-language rows sitting in the
    single cluster that holds most of them. At 1.0 the tree has one bucket
    labelled "the other language" and has resolved none of its intents.
    """
    lang = np.asarray(list(row_language))
    m = lang == minority
    n_min = int(m.sum())
    if n_min == 0:
        return {"language": minority, "n": 0, "note": "not present"}
    lab = labels[m]
    vals, counts = np.unique(lab, return_counts=True)
    top = int(counts.max())
    top_cluster = int(vals[int(counts.argmax())])
    in_cluster = int((labels == top_cluster).sum())
    return {
        "language": minority,
        "n": n_min,
        "concentration": round(top / n_min, 4),
        "largest_cluster": top_cluster,
        "cluster_purity": round(top / max(in_cluster, 1), 4),
        "clusters_touched": int(len(vals)),
        "clusters_with_5pct": int((counts >= 0.05 * n_min).sum()),
        "verdict": (
            "collapsed — this language has become one undifferentiated bucket"
            if top / n_min > 0.8 else
            "partially collapsed" if top / n_min > 0.5 else
            "distributed by intent"
        ),
    }


def tokenizer_for(script: str) -> str:
    """Tokeniser implied by a script, for per-corpus configuration."""
    return {"han": "jieba", "kana": "none", "hangul": "none"}.get(script, "whitespace")


def char_ngram_for(script: str) -> tuple[int, int]:
    """Character n-gram range implied by a script.

    Han characters are morphemes, so 1-3 spans a word or two. Latin letters are
    phonemes, so the same range captures noise; 3-5 is where English morphology
    starts to show.
    """
    return (1, 3) if script in ("han", "kana", "hangul") else (3, 5)


def minority_sub_intents(
    queries: Sequence[str],
    row_language: Sequence[str],
    leaf_labels: np.ndarray,
    leaf_family: np.ndarray,
    *,
    dominant: str,
    min_purity: float = 0.6,
    min_size: int = 40,
    max_sub: int = 6,
    min_sub_size: int = 15,
    seed: int = 0,
) -> dict[str, Any]:
    """Resolve intents inside minority-language families, as a FACET not as leaves.

    The global representation put these rows together *because of their
    language*. Inside such a family language is constant and carries no further
    information, so a script-appropriate character n-gram space can separate the
    intents that remain — and it does: on a mixed corpus it split a 398-row
    English bucket into comparison queries, spec lookups, and how-tos.

    **Why this returns a facet column rather than new tree leaves.** A leaf in
    this pipeline is defined as a centroid in the hybrid space, and Phase 10
    deploys exactly that rule: ``argmax(x @ centroids.T)``. Sub-clusters found in
    a *different* space are not centroid regions of the deployed one — measured,
    they were 91.8% consistent against 98% for ordinary leaves, and injecting
    them cost held-out reproduction. So they ship as a parallel column that the
    serving layer can use, while the tree keeps its invariant.

    The honest reading, which belongs in the report: the hybrid space cannot
    express intent *within* a minority language. If that language matters
    commercially, give it its own tree and its own centroid model rather than
    asking one space to serve both.
    """
    from sklearn.cluster import KMeans
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    q = np.asarray(list(queries))
    lang = np.asarray(list(row_language))
    labels = np.asarray(leaf_labels).copy()
    fam_of_row = np.asarray(leaf_family)[labels]

    treated: list[dict[str, Any]] = []
    facet = np.array([""] * len(q), dtype=object)

    for fam in sorted(set(fam_of_row.tolist())):
        m = fam_of_row == fam
        n = int(m.sum())
        if n < min_size:
            continue
        # Purity is measured as "share of rows that are NOT the dominant
        # language", not as the share of the single most common label. Mixed-script
        # rows ("iPhone 保护壳") carry their own labels, so counting only the top
        # label reads a family that is 69% non-Chinese as 69% "latin" and lets it
        # slip under the threshold — which is exactly the family that needed help.
        fam_langs = lang[m]
        non_dominant = np.array([
            not (str(l) == dominant or str(l) == "nonlinguistic") for l in fam_langs
        ])
        purity = float(non_dominant.mean())
        if purity < min_purity:
            continue
        minority_labels = [str(l) for l in fam_langs[non_dominant]]
        if not minority_labels:
            continue
        # The script to tune the n-gram range for: the most common non-dominant
        # one, with a mixed label resolved to its leading script.
        resolved = [l.split(":", 1)[1].split("+")[0] if l.startswith("mixed:") else l
                    for l in minority_labels]
        vals, cnts = np.unique(resolved, return_counts=True)
        top_lang = str(vals[int(cnts.argmax())])

        texts = [str(t) for t in q[m]]
        lo, hi = char_ngram_for(top_lang)
        try:
            vec = TfidfVectorizer(analyzer="char", ngram_range=(lo, hi), min_df=2,
                                  sublinear_tf=True)
            Xs = vec.fit_transform(texts)
            k_comp = min(64, max(2, min(Xs.shape) - 1))
            Z = normalize(TruncatedSVD(k_comp, random_state=seed).fit_transform(Xs))
        except Exception:
            continue

        best_k = max(2, min(max_sub, n // max(min_sub_size, 1)))
        if best_k < 2:
            continue
        sub = KMeans(best_k, random_state=seed, n_init=10).fit_predict(Z)

        idx = np.where(m)[0]
        for j in range(best_k):
            facet[idx[sub == j]] = f"{top_lang}_{fam}_{j + 1}"

        treated.append({
            "family": int(fam), "language": top_lang, "purity": round(purity, 3),
            "n_rows": n, "split_into": best_k,
            "ngram_range": [lo, hi],
            "why": (f"family was {purity:.0%} non-{dominant} (mostly {top_lang}) in a "
                    f"{dominant}-dominant corpus, so the global space had separated it by "
                    "language and had nothing left to split on"),
        })

    return {
        "sub_intent_facet": facet,
        "families_treated": treated,
        "n_sub_intents": int(len({f for f in facet if f})),
        "contract": (
            "a parallel column, not tree leaves — these clusters live in a "
            "script-appropriate character space, not in the deployed hybrid space, so "
            "making them leaves would break the centroid rule the classifier depends on"
        ),
    }
