"""Are the two annotators actually comparable? Measured, not assumed.

Kappa is supposed to say whether the GUIDE is clear. It cannot, if the two
annotators differ in capability: a gap between *them* shows up as disagreement no
guide fix can close, and p2a's pilot then reads it as a "structural confusion"
and redraws boundaries that were never the problem.

This was discovered by hand, two runs late, by grepping gold sets:

    live38  referee glm-5.2      annotator_a's label won 78.3% (z=+12.1, n=460)
    live39  referee glm-4.5-airx annotator_a's label won 55.1% (z=+2.2,  n=459)

Same annotator models in both. The first is a real capability gap; the second is
an adjudicator that cannot discriminate, which looks like parity and is not.
Both readings matter, and neither was visible in any artifact — so the pipeline
now measures it every run.

**Read the two numbers together.** A large asymmetry means the annotators are
mismatched; a win-rate pinned near 50% with a large n can mean they are matched
*or* that the referee is deciding at chance. The second is only distinguishable
by looking at the referee, which is why the record carries its model id.

**DEMOTED, deliberately: this is a diagnostic, not a way to choose models.**

It is the weakest of the three tests available, and the other two are better:

    test                          referee-free?  before spending?
    win-rate on refereed rows          NO             NO      <- this module
    per-annotator self-consistency     yes            yes (200-row pilot)
    an independent evaluation          yes            yes

Its verdict conflates two different things — a genuine capability gap between the
annotators, and a referee that cannot discriminate — which is why the record has
to carry the referee's model id to be readable at all. An instrument needing that
caveat is not one to select models with. Read it as "did the pairing we chose
turn out lopsided", never as "which model should annotator_b be".

Nothing here changes a label. It reports a property of the measuring instrument.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence


@dataclass
class AnnotatorBalance:
    n_contested: int = 0
    a_won: int = 0
    b_won: int = 0
    neither: int = 0
    referee_model: str = ""

    @property
    def n_decided(self) -> int:
        return self.a_won + self.b_won

    @property
    def a_share(self) -> float:
        return self.a_won / self.n_decided if self.n_decided else float("nan")

    @property
    def z(self) -> float:
        """Standard normal deviate of a's win-rate against a 50/50 null."""
        n = self.n_decided
        if n < 2:
            return float("nan")
        return (self.a_share - 0.5) / math.sqrt(0.25 / n)

    #: Below this, the win-rate is not evidence about anything. At n=10 a 7/3
    #: split is z=+1.3 and a 9/1 split is z=+2.5 — neither reaches the bar, so a
    #: "not lopsided" verdict here means "we could not tell", which is a
    #: different claim and must not be printed as the first one.
    MIN_DECIDED: int = 20

    @property
    def undecidable(self) -> bool:
        """Too few adjudicated rows to say anything either way.

        Reported separately from `lopsided` because collapsing them says "the
        annotators are comparable" on zero evidence — the failure this project
        keeps re-meeting: an empty adversary response scoring 1.000, a kappa
        computed on 199 of 600 rows. Coverage first, verdict second.
        """
        return self.n_decided < self.MIN_DECIDED

    @property
    def lopsided(self) -> bool:
        """One annotator systematically beaten by the other.

        |z| > 3 rather than a share threshold: the same 60/40 split is noise on
        40 rows and decisive on 400, and a fixed share cannot tell them apart.
        """
        if self.undecidable:
            return False              # not a verdict; see `undecidable`
        z = self.z
        return z == z and abs(z) > 3.0

    def as_record(self) -> dict[str, Any]:
        return {
            "n_contested": self.n_contested, "n_decided": self.n_decided,
            "annotator_a_won": self.a_won, "annotator_b_won": self.b_won,
            "referee_chose_neither": self.neither,
            "annotator_a_share": (round(self.a_share, 4) if self.n_decided else None),
            "z_vs_even": (round(self.z, 2) if self.n_decided >= 2 else None),
            "lopsided": self.lopsided,
            "undecidable": self.undecidable,
            "min_decided_for_a_verdict": self.MIN_DECIDED,
            "referee_model": self.referee_model,
            "how_to_read": (
                "A large |z| means the annotators are NOT comparable, so kappa is "
                "measuring the gap between them rather than the clarity of the guide "
                "— and the pilot will read that gap as a structural confusion and "
                "redraw boundaries that were never the problem. A share near 50% on a "
                "large n means either that they are matched or that the REFEREE is "
                "deciding at chance; the two are only separable by looking at the "
                "referee, whose model id is recorded here for that reason."),
        }


def annotator_balance(gold_rows: Sequence[Any], referee_model: str = "") -> AnnotatorBalance:
    """Who won the rows the referee actually adjudicated."""
    bal = AnnotatorBalance(referee_model=referee_model)
    for row in gold_rows:
        if not getattr(row, "adjudicated", False):
            continue
        final = str(getattr(row, "final", "") or "")
        if not final:
            continue
        a = str(getattr(row, "label_a", "") or "")
        b = str(getattr(row, "label_b", "") or "")
        if a == b:
            continue                       # not contested; nothing to adjudicate
        bal.n_contested += 1
        if final == a:
            bal.a_won += 1
        elif final == b:
            bal.b_won += 1
        else:
            # The referee picked a third class. Counted separately: it is neither
            # annotator's win, and folding it into either would bias the share.
            bal.neither += 1
    return bal
