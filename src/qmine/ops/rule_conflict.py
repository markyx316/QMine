"""Detecting rule contradictions by MEASURING them on the corpus.

`_dedupe_rules` compares rules structurally: identical trigger on an identical
class pair, or — for prose rules — 85% text similarity. That catches restatements
and exact collisions. It cannot see the case that actually hurts: two rules whose
conditions are written differently, OVERLAP in the queries they fire on, and point
at different classes. An annotator meeting a query in that overlap is given two
instructions and can only follow one.

Text similarity is the wrong instrument, and this codebase has the scar: comparing
rendered `when` sentences once shredded 32 of 41 rules on a live run, because two
markers for ONE boundary render as the same template differing by a couple of
characters — and a marker pair pointing at opposite classes is precisely what
settling a boundary looks like.

**So measure the overlap instead.** A rule with a `trigger` is an executable
predicate over queries. Run both against the corpus and count the rows where they
BOTH fire and disagree. Measured on live39:

    R006 / R007   4 rows  (jaccard 0.001)  -> a legitimate discriminating pair
    R018 / R019 301 rows  (jaccard 0.089)  -> a real ambiguity zone

Four rows is noise; 301 is 301 queries whose instructions contradict.

**Nothing is withheld.** The remedy for a 301-row overlap is not to delete both
rules — that removes the guidance and leaves the boundary unaddressed. It is to
report the zone, with its size and its actual queries, so the boundary can be
given a tie-break. Withholding is what caused the 32-of-41 disaster.

**Prose rules have no predicate**, so they cannot be measured this way. For them
the available signal is CONCENTRATION: many rules piling onto one class pair and
pointing both ways means the boundary is contested, whoever is right. live39 put
**14 rules on `TEXT_INTERPRETATION x WORD_MEANING_LOOKUP`**, 9 one way and 5 the
other, 13 of them prose. That is a taxonomy problem surfacing as a rule problem,
and it is worth saying so.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass
class Overlap:
    """One measured contradiction: two rules that co-fire and disagree."""

    rule_a: str
    rule_b: str
    classes: tuple[str, ...]
    then_a: str
    then_b: str
    n_both: int
    n_a: int
    n_b: int
    jaccard: float
    examples: list[str] = field(default_factory=list)

    def as_record(self) -> dict[str, Any]:
        return {
            "rules": [self.rule_a, self.rule_b], "classes": list(self.classes),
            "then": [self.then_a, self.then_b], "n_rows_both_fire": self.n_both,
            "n_rows_a": self.n_a, "n_rows_b": self.n_b,
            "jaccard": round(self.jaccard, 4), "examples": self.examples[:5],
        }


@dataclass
class ConflictReport:
    overlaps: list[Overlap] = field(default_factory=list)
    crowded_pairs: list[dict[str, Any]] = field(default_factory=list)
    n_measurable: int = 0
    n_rules: int = 0
    bad_regex: list[str] = field(default_factory=list)

    def as_record(self) -> dict[str, Any]:
        return {
            "n_rules": self.n_rules,
            "n_with_executable_trigger": self.n_measurable,
            "measured_overlaps": [o.as_record() for o in self.overlaps],
            "crowded_class_pairs": self.crowded_pairs,
            "unparseable_triggers": self.bad_regex,
            "note": ("Overlaps are MEASURED on the corpus, not inferred from wording. "
                     "Nothing is withheld: a large overlap means the boundary needs a "
                     "tie-break for that region, not that both rules are wrong."),
        }


def find_conflicts(
    rules: Sequence[Any],
    queries: Sequence[str],
    *,
    min_rows: int = 25,
    crowded_at: int = 5,
) -> ConflictReport:
    """Measure which rules contradict each other on THIS corpus.

    `min_rows` is a floor on what counts as a zone worth reporting — a handful of
    incidental co-hits is not a contradiction anyone will meet. `crowded_at` is
    when a class pair has enough two-way rules that the boundary itself, rather
    than any individual rule, is the thing to look at.
    """
    import numpy as np

    rep = ConflictReport(n_rules=len(rules))
    qs = [str(q) for q in queries]
    masks: dict[str, Any] = {}
    for r in rules:
        trig = str(getattr(r, "trigger", "") or "")
        rid = str(getattr(r, "id", "") or "")
        if not trig or not rid:
            continue
        try:
            pat = re.compile(trig)
        except re.error:
            rep.bad_regex.append(rid)
            continue
        masks[rid] = np.fromiter((bool(pat.search(q)) for q in qs), bool, len(qs))
    rep.n_measurable = len(masks)

    by_pair: dict[tuple[str, ...], list[Any]] = {}
    for r in rules:
        cs = tuple(sorted(str(c) for c in (getattr(r, "classes", None) or [])))
        if len(cs) >= 2:
            by_pair.setdefault(cs, []).append(r)

    for cs, rs in by_pair.items():
        thens = {str(getattr(r, "then", "")).strip() for r in rs}
        if len(rs) >= crowded_at and len(thens) > 1:
            rep.crowded_pairs.append({
                "classes": list(cs), "n_rules": len(rs),
                "distinct_targets": sorted(thens),
                "n_with_trigger": sum(1 for r in rs if getattr(r, "trigger", "")),
                "why_it_matters": ("Many rules pointing both ways at one boundary means "
                                   "the boundary is contested. Most carry no executable "
                                   "trigger, so their overlap cannot be measured — the "
                                   "count is the only signal available."),
            })
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                a, b = rs[i], rs[j]
                ia, ib = str(getattr(a, "id", "")), str(getattr(b, "id", ""))
                if str(getattr(a, "then", "")) == str(getattr(b, "then", "")):
                    continue                      # same answer: not a conflict
                if ia not in masks or ib not in masks:
                    continue                      # prose: no predicate to measure
                both = masks[ia] & masks[ib]
                n_both = int(both.sum())
                if n_both < min_rows:
                    continue
                na, nb = int(masks[ia].sum()), int(masks[ib].sum())
                rep.overlaps.append(Overlap(
                    rule_a=ia, rule_b=ib, classes=cs,
                    then_a=str(getattr(a, "then", "")), then_b=str(getattr(b, "then", "")),
                    n_both=n_both, n_a=na, n_b=nb,
                    jaccard=n_both / max(1, na + nb - n_both),
                    examples=[qs[k] for k in np.flatnonzero(both)[:5]],
                ))
    rep.overlaps.sort(key=lambda o: -o.n_both)
    return rep
