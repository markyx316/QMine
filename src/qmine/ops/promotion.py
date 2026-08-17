"""The referee upgrade protocol — how a new model's labels earn the right to ship.

Every model change is a temptation to assume improvement.  The new encoder is
newer, the new taxonomy is more considered, the new run had better metrics — so
the new labels replace the old ones and nobody checks.  Then a regression on the
head of the distribution surfaces three weeks later in a support ticket.

The protocol is narrow and cheap.  Sample the rows where the two label sets
*disagree* — agreements carry no information about which system is better —
and put each disagreement to a referee **blind**: it sees the query and two
candidate labels, not which system produced which. The new labels are promoted
only if the new system wins by a margin that is not explicable by chance.

Two details do the work:

* **Blind, with randomised presentation order.** LLM judges have a documented
  position bias. Presenting the new label first every time would manufacture
  exactly the win we are testing for.
* **Old labels are never destroyed.** They move to a ``_v1`` column and every
  overturned row is stamped ``label_source='referee'``, so a promotion is
  reversible and every changed row is traceable to the judgment that changed it.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from ..determinism import rng


def find_disagreements(
    old: Sequence[str], new: Sequence[str], *, limit: int = 400, seed: int = 0
) -> dict[str, Any]:
    """Sample the rows where the two systems differ.

    Agreements are dropped deliberately: on a corpus where two systems agree 90%
    of the time, judging a random sample spends 90% of the budget confirming
    what both already know, and the confidence interval on the interesting 10%
    stays wide.
    """
    old_a, new_a = np.asarray(list(old), dtype=object), np.asarray(list(new), dtype=object)
    if len(old_a) != len(new_a):
        raise ValueError(f"label sets differ in length: {len(old_a)} vs {len(new_a)}")
    idx = np.where(old_a != new_a)[0]
    r = rng(seed)
    picked = np.sort(r.choice(idx, size=min(limit, idx.size), replace=False)) if idx.size else idx
    return {
        "n_rows": int(len(old_a)),
        "n_disagreements": int(idx.size),
        "disagreement_rate": round(float(idx.size / max(len(old_a), 1)), 4),
        "sampled_indices": picked.tolist(),
    }


def build_blind_matchups(
    queries: Sequence[str],
    old: Sequence[str],
    new: Sequence[str],
    indices: Sequence[int],
    *,
    seed: int = 0,
) -> tuple[list[dict[str, Any]], dict[int, str]]:
    """Present each disagreement as A-vs-B with the sides randomly swapped.

    Returns the matchups and the key mapping each row back to which side was the
    new system — kept out of the prompt entirely.
    """
    r = rng(seed)
    matchups, key = [], {}
    for i in indices:
        i = int(i)
        new_is_a = bool(r.randint(0, 2))
        matchups.append({
            "row": i,
            "query": str(queries[i]),
            "label_a": str(new[i] if new_is_a else old[i]),
            "label_b": str(old[i] if new_is_a else new[i]),
        })
        key[i] = "a" if new_is_a else "b"
    return matchups, key


def score_verdicts(
    verdicts: Sequence[dict[str, Any]], key: dict[int, str], *, alpha: float = 0.05
) -> dict[str, Any]:
    """Tally the head-to-head and decide whether the new system earned promotion.

    The test is a two-sided binomial against a fair coin, using the normal
    approximation. Rows the referee called equally defensible are excluded from
    the denominator rather than split — a tie is evidence of ambiguity in the
    query, not half a win for either side.
    """
    new_wins = old_wins = ties = 0
    per_row: list[dict[str, Any]] = []
    for v in verdicts:
        row = int(v.get("row", -1))
        winner = str(v.get("winner", "")).lower().strip()
        if winner in ("tie", "both", "equal", ""):
            ties += 1
            outcome = "tie"
        elif winner == key.get(row):
            new_wins += 1
            outcome = "new"
        else:
            old_wins += 1
            outcome = "old"
        per_row.append({"row": row, "outcome": outcome, "rationale": v.get("rationale", "")})

    decided = new_wins + old_wins
    share = new_wins / decided if decided else 0.0
    # two-sided binomial test against p=0.5, normal approximation
    if decided >= 10:
        z = (new_wins - decided / 2) / math.sqrt(decided * 0.25)
        p_value = math.erfc(abs(z) / math.sqrt(2))
    else:
        z, p_value = float("nan"), 1.0

    promote = bool(decided >= 10 and share > 0.5 and p_value < alpha)
    return {
        "n_judged": len(verdicts),
        "new_wins": new_wins,
        "old_wins": old_wins,
        "ties": ties,
        "n_decided": decided,
        "new_win_share": round(share, 4),
        "z": round(z, 3) if not math.isnan(z) else None,
        "p_value": round(p_value, 5),
        "alpha": alpha,
        "promote": promote,
        "verdict": (
            f"PROMOTE — the new labels won {new_wins}/{decided} decided matchups "
            f"(p={p_value:.4f})"
            if promote else
            f"HOLD — {new_wins}/{decided} decided matchups is not a significant win "
            f"(p={p_value:.4f}); the old labels stay authoritative"
        ),
        "per_row": per_row,
        "note": (
            "ties are excluded from the denominator rather than split: a referee "
            "calling both labels defensible is telling us the query is ambiguous, "
            "not awarding half a point"
        ),
    }


def apply_promotion(
    df: Any,
    old_col: str,
    new_labels: Sequence[str],
    scoring: dict[str, Any],
    *,
    final_col: str = "label_final",
) -> Any:
    """Write the promoted labels, preserving the old column and the audit trail.

    The old labels move to ``<old_col>_v1`` and never leave the table. Rows the
    referee actually overturned are marked ``label_source='referee'``, so the
    delivered data answers "who changed this row, and on what evidence" without
    reference to a log.
    """
    out = df.copy()
    out[f"{old_col}_v1"] = out[old_col]
    overturned = {r["row"] for r in scoring["per_row"] if r["outcome"] == "new"}

    if scoring["promote"]:
        out[final_col] = list(new_labels)
        out["label_source"] = [
            "referee" if i in overturned else "model_v2" for i in range(len(out))
        ]
    else:
        out[final_col] = out[old_col]
        out["label_source"] = "model_v1"
    out.attrs["promotion"] = {k: v for k, v in scoring.items() if k != "per_row"}
    return out
