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

## Why most rules have no trigger, and why demanding one would be wrong

Measured on live39: 24 of 104 rules carry an executable trigger, and **all 24
came from the architect**. The 79 the referee drafted carry none. The obvious
reading — the referee forgot, so ask it harder — is wrong, and acting on it would
have made this worse.

Read what the referee actually writes:

    "when the query is a proverb, idiom or aphorism and the user wants its moral
     rather than a dictionary definition, choose TEXT_INTERPRETATION"

That is a **semantic** condition. There is no regex for it. Of the 80 trigger-less
rules on live39, exactly **one** contains an extractable marker list. Demanding a
trigger from the other 79 does not produce 79 predicates; it produces 79 *wrong*
predicates — and a wrong predicate is strictly worse than a missing one here,
because `find_conflicts` would then report overlaps that are artifacts of a bad
regex and, worse, report *no* overlap where the semantic rules really do collide.
This codebase has been bitten by exactly that shape before: comparing rendered
`when` sentences shredded 32 of 41 rules on a live run.

So a trigger is never assumed and never demanded. `validate_trigger` requires one
to earn its status: it must compile, fire on the rule's own evidence, and select a
plausible slice of the corpus. What passes is `lexical` and individually testable.
What does not is `semantic`, is *labelled* as such, and is measured the only
honest way left — see below.

## What replaces the missing predicate: the referee's own verdicts

A rule that cannot be run against the corpus can still be run against **the
evidence that produced it**. Every rule names a class pair, and the gold set
records how the referee actually ruled on that pair. That is a measurement, it
needs no regex and no extra call, and it found something real on live39:

    OTHER x TEXT_INTERPRETATION
      referee ruled TEXT_INTERPRETATION on 15 of 21 rows (71%)
      6 rules in the shipped guide - 5 of them point at OTHER

The guide contradicts the evidence it was built from, and nothing was measuring
it.

**The confound, and why this is reported per BOUNDARY.** A rule is conditional:
"when <condition>, choose T". A rule carving out a genuine minority exception
*should* point away from the pair's majority, so a per-rule "disagrees with the
majority" score would call every legitimate exception a defect — a false finding
of exactly the kind this project keeps having to unlearn. Aggregated over a
boundary the confound washes out: five of six rules cannot all be narrow
exceptions. A `lexical` rule escapes the confound entirely, because its trigger
says which rows it claims and the verdicts on *those* rows can be read directly.

**Nothing is deleted, and no rule is called wrong.** The output is the boundary,
its verdict distribution, and which way its rules point. A boundary where the
evidence is decisive and the rules lean the other way is a boundary someone must
look at — not a set of rules to remove.
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
    #: Rules whose `then` names zero or several classes, so it cannot be compared.
    unkeyed_then: list[str] = field(default_factory=list)

    def as_record(self) -> dict[str, Any]:
        return {
            "n_rules": self.n_rules,
            "n_with_executable_trigger": self.n_measurable,
            "measured_overlaps": [o.as_record() for o in self.overlaps],
            "crowded_class_pairs": self.crowded_pairs,
            "unparseable_triggers": self.bad_regex,
            # Scoped in the name: only the rules encountered while comparing
            # candidate pairs, which is a strict subset of the rules on a
            # measured boundary. See `rules_against_evidence.as_record`.
            "rules_in_compared_pairs_whose_then_is_not_a_class":
                sorted(set(self.unkeyed_then)),
            "note": ("Overlaps are MEASURED on the corpus, not inferred from wording. "
                     "Nothing is withheld: a large overlap means the boundary needs a "
                     "tie-break for that region, not that both rules are wrong."),
        }


#: What `_repair_guide` writes when a boundary's tie-break is "the query carries
#: NO marker of this boundary". It is a sentinel, not a pattern.
NO_MARKER = "<no-marker>"


@dataclass
class TriggerCheck:
    ok: bool
    reason: str = ""
    n_hits: int = 0
    share: float = 0.0


def validate_trigger(
    trigger: str,
    queries: Sequence[str],
    *,
    evidence: Sequence[str] = (),
    max_share: float = 0.5,
) -> TriggerCheck:
    """Decide whether a trigger is a predicate or just a string that compiles.

    Three demands, each after a way a trigger can be useless while looking fine:

    - **It compiles.** A trigger that does not is not a predicate.
    - **It fires on the rule's own evidence.** Every referee rule records the
      query whose disagreement created it. A trigger that does not match that
      query is not describing the rule it is attached to — the single cheapest
      way to catch a plausible-looking regex that means something else.
    - **It selects a slice, not the corpus.** A trigger matching nothing can
      never conflict with anything, and one matching most of the corpus makes
      every pair of rules look like it overlaps. Both produce measurements that
      are about the trigger rather than about the rules.

    A trigger that fails any of these is dropped rather than repaired. It costs
    one rule's measurability; keeping it costs the meaning of every overlap the
    report prints.
    """
    trig = (trigger or "").strip()
    if not trig:
        return TriggerCheck(False, "no trigger")
    if trig == NO_MARKER:
        return TriggerCheck(True, "boundary default — evaluated against the pair's markers")
    try:
        pat = re.compile(trig)
    except re.error as exc:
        return TriggerCheck(False, f"not a valid pattern: {exc}")
    ev = [str(e) for e in evidence if str(e).strip()]
    if ev and not any(pat.search(e) for e in ev):
        return TriggerCheck(False, "does not fire on the rule's own example or "
                                   "originating query — it describes something else")
    n = sum(1 for q in queries if pat.search(q))
    share = n / max(1, len(queries))
    if n == 0:
        return TriggerCheck(False, "fires on no row of this corpus", 0, 0.0)
    if share > max_share:
        return TriggerCheck(False, f"fires on {share:.0%} of the corpus — that is not a "
                                   "condition", n, share)
    return TriggerCheck(True, "", n, share)


def _pair_masks(rules: Sequence[Any], qs: Sequence[str]) -> tuple[dict[str, Any], list[str]]:
    """Build one boolean mask per measurable rule.

    `<no-marker>` is handled here rather than being compiled, because compiling
    it is silently wrong: `re.compile("<no-marker>")` is a valid pattern that
    matches the literal text, so it produced an all-False mask, contributed to no
    overlap, and was still counted in `n_with_executable_trigger`. The count of
    what we could measure was inflated by rules we were measuring as nothing.

    Its real meaning is the negation of its own boundary: the query carries none
    of the markers the other rules on this class pair fire on.
    """
    import numpy as np

    bad: list[str] = []
    compiled: dict[str, Any] = {}
    for r in rules:
        trig, rid = str(getattr(r, "trigger", "") or ""), str(getattr(r, "id", "") or "")
        if not trig or not rid or trig == NO_MARKER:
            continue
        try:
            compiled[rid] = re.compile(trig)
        except re.error:
            bad.append(rid)

    by_pair_markers: dict[tuple[str, ...], list[Any]] = {}
    for r in rules:
        rid = str(getattr(r, "id", "") or "")
        if rid in compiled:
            cs = tuple(sorted(str(c) for c in (getattr(r, "classes", None) or [])))
            by_pair_markers.setdefault(cs, []).append(compiled[rid])

    masks: dict[str, Any] = {}
    for rid, pat in compiled.items():
        masks[rid] = np.fromiter((bool(pat.search(q)) for q in qs), bool, len(qs))
    for r in rules:
        if str(getattr(r, "trigger", "") or "") != NO_MARKER:
            continue
        rid = str(getattr(r, "id", "") or "")
        cs = tuple(sorted(str(c) for c in (getattr(r, "classes", None) or [])))
        pats = by_pair_markers.get(cs) or []
        if not pats:
            # No marker rule on this boundary means "carries no marker" has no
            # referent. Left unmeasurable rather than treated as "fires on
            # everything", which would swamp every overlap on the pair.
            continue
        hit = np.zeros(len(qs), bool)
        for pat in pats:
            hit |= np.fromiter((bool(pat.search(q)) for q in qs), bool, len(qs))
        masks[rid] = ~hit
    return masks, bad


def find_conflicts(
    rules: Sequence[Any],
    queries: Sequence[str],
    *,
    min_rows: int = 25,
    crowded_at: int = 5,
    codes: Sequence[str] = (),
) -> ConflictReport:
    """Measure which rules contradict each other on THIS corpus.

    `min_rows` is a floor on what counts as a zone worth reporting — a handful of
    incidental co-hits is not a contradiction anyone will meet. `crowded_at` is
    when a class pair has enough two-way rules that the boundary itself, rather
    than any individual rule, is the thing to look at.

    `codes` is the declared class list. Given it, a rule whose `then` is not one
    of them is skipped rather than string-compared: on live38, 17 rules held a
    sentence naming BOTH sides of a boundary, and comparing those as strings
    makes two phrasings of one answer look like a disagreement. Passing nothing
    keeps the old behaviour, which is right only when no class list is available.
    """
    import numpy as np

    rep = ConflictReport(n_rules=len(rules))
    qs = [str(q) for q in queries]
    masks, rep.bad_regex = _pair_masks(rules, qs)
    rep.n_measurable = len(masks)

    by_pair: dict[tuple[str, ...], list[Any]] = {}
    for r in rules:
        cs = tuple(sorted(str(c) for c in (getattr(r, "classes", None) or [])))
        if len(cs) >= 2:
            by_pair.setdefault(cs, []).append(r)

    valid = set(codes)

    def _key(r: Any) -> str | None:
        t = str(getattr(r, "then", "")).strip()
        if not valid:
            return t or None
        return t if t in valid else None

    for cs, rs in by_pair.items():
        thens = {t for t in (_key(r) for r in rs) if t}
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
                ka, kb = _key(a), _key(b)
                if ka is None or kb is None:
                    # One of them does not resolve to a single class, so "do they
                    # disagree?" has no answer. Reporting an overlap here would be
                    # a fact about the sentence, not about the rules.
                    # BOTH, when both are unkeyed. Recording one made the report
                    # understate how much of the rule set it could not compare.
                    rep.unkeyed_then.extend(
                        [x for x, k in ((ia, ka), (ib, kb)) if k is None])
                    continue
                if ka == kb:
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


# ====================================================================
# Rules against the evidence that produced them
# ====================================================================
@dataclass
class BoundaryEvidence:
    """How one class pair was actually adjudicated, and where its rules point."""

    classes: tuple[str, ...]
    n_adjudicated: int
    verdicts: dict[str, int]
    majority: str
    majority_share: float
    ci95: tuple[float, float]
    decisive: bool
    n_rules: int
    rules_toward_majority: list[str]
    rules_against_majority: list[str]
    #: Rules on this boundary whose `then` is not a single class code, so they
    #: are excluded from the tally rather than counted as pointing anywhere.
    rules_not_a_key: list[str] = field(default_factory=list)

    #: Rules drafted from a row ruled the MAJORITY way, and the minority way.
    #: Recorded because it is the confound that retired the old verdict — see
    #: `direction_is_confounded`.
    n_rules_from_majority_rows: int = 0
    n_rules_from_minority_rows: int = 0
    n_majority_rows: int = 0
    n_minority_rows: int = 0

    @property
    def direction_is_confounded(self) -> bool:
        """Were rules drafted at a very different rate on the two sides?

        A referee drafts a rule only where it judges the guide to have FAILED,
        which concentrates on the side that goes against the prevailing pattern.
        Measured on live39's `OTHER x TEXT_INTERPRETATION`: 5 of 6 minority rows
        produced a rule (83%) against 1 of 15 majority rows (7%). When that gap
        is large, "most of this boundary's rules point away from the majority" is
        the EXPECTED shape of a healthy exception set and carries no information
        about whether anything is wrong.
        """
        maj = self.n_rules_from_majority_rows / max(1, self.n_majority_rows)
        mino = self.n_rules_from_minority_rows / max(1, self.n_minority_rows)
        return mino > 2 * maj

    @property
    def contradicted(self) -> bool:
        """RETIRED as a verdict. Always False; the tally is kept as context.

        This counted rule directions against the adjudicated majority, and three
        measured defects retired it. It is confounded by the drafting rate above
        — the check would fire identically on a guide with no defect, which is
        this codebase's own named trap: never test a mechanism with a
        distribution it has already filtered. It had no resolution: 15/21 gives a
        Wilson lower bound of 0.5004 against a bar of 0.5, so a single row
        decided whether the boundary was reported. And its magnitude counted one
        drafting template emitted five times, which measures the referee's habit
        rather than the number of conflicts.

        `stated_grounds` replaces it and shares none of those problems.
        """
        return False

    def as_record(self) -> dict[str, Any]:
        return {
            "classes": list(self.classes), "n_adjudicated": self.n_adjudicated,
            "referee_verdicts": self.verdicts, "majority": self.majority,
            "majority_share": round(self.majority_share, 4),
            "ci95": [round(self.ci95[0], 4), round(self.ci95[1], 4)],
            "evidence_is_decisive": self.decisive,
            "n_rules": self.n_rules,
            "rules_toward_majority": self.rules_toward_majority,
            "rules_against_majority": self.rules_against_majority,
            "rules_excluded_then_not_a_class": self.rules_not_a_key,
            "rules_drafted_from_majority_rows": self.n_rules_from_majority_rows,
            "rules_drafted_from_minority_rows": self.n_rules_from_minority_rows,
            "direction_is_confounded": self.direction_is_confounded,
            "direction_caveat": (
                "The rule directions above are CONTEXT, not a verdict. A referee drafts "
                "a rule only where it judges the guide to have failed, which concentrates "
                "on the minority side — so 'most rules point away from the majority' is "
                "the expected shape of a healthy exception set. Read "
                "`stated_grounds` for the measurement that is not confounded."),
        }


@dataclass
class RuleEvidenceReport:
    boundaries: list[BoundaryEvidence] = field(default_factory=list)
    #: Per-rule, and ONLY for rules whose trigger was validated — the conditional
    #: test that the boundary aggregate cannot do.
    lexical_rules: list[dict[str, Any]] = field(default_factory=list)
    n_lexical: int = 0
    n_semantic: int = 0
    n_rejected_triggers: int = 0
    rejected: list[dict[str, str]] = field(default_factory=list)
    unkeyed_then: list[str] = field(default_factory=list)
    #: The measurement that replaced the direction verdict.
    stated_grounds: list[Any] = field(default_factory=list)

    @property
    def vacuous_grounds(self) -> list[Any]:
        """Boundaries whose stated discriminator divides none of their rows."""
        return [g for g in self.stated_grounds if not g.separates]

    @property
    def contradicted(self) -> list[BoundaryEvidence]:
        """Retired — always empty. Kept so callers do not break; read
        `vacuous_grounds` instead."""
        return [b for b in self.boundaries if b.contradicted]

    def as_record(self) -> dict[str, Any]:
        return {
            "n_lexical_rules": self.n_lexical,
            "n_semantic_rules": self.n_semantic,
            "n_triggers_rejected": self.n_rejected_triggers,
            # NAMED for the population it actually covers. `find_conflicts`
            # records only the rules it met while comparing PAIRS; this one
            # records every rule on a measured boundary. Two different
            # populations under one field name read as a contradiction — the
            # p2b observer confirmed exactly that on live40.
            "rules_on_measured_boundaries_whose_then_is_not_a_class":
                sorted(set(self.unkeyed_then)),
            # A count beside a truncated list is a self-contradicting artifact:
            # live40 shipped `n_triggers_rejected: 13` next to 12 entries, and
            # the observer proved it. Say what was cut.
            "rejected_triggers": self.rejected[:12],
            "rejected_triggers_truncated": max(0, len(self.rejected) - 12),
            "boundaries": [b.as_record() for b in self.boundaries],
            "stated_grounds": [g.as_record() for g in self.stated_grounds],
            "boundaries_whose_stated_ground_separates_nothing":
                [g.as_record() for g in self.vacuous_grounds],
            "lexical_rule_checks": self.lexical_rules[:20],
            "note": ("The load-bearing measurement is `stated_grounds`: the words the "
                     "rules THEMSELVES name as the discriminator, checked against the rows "
                     "the referee actually adjudicated. A ground that every row falls on "
                     "one side of did not decide the boundary, whatever the rules say. "
                     "Counting which way rules point was tried and retired — it is "
                     "confounded by the rate at which rules get drafted on each side. "
                     "Nothing here deletes a rule."),
        }


def _rule_evidence(rule: Any) -> list[str]:
    """Everything that can serve as the rule's own evidence for its trigger."""
    ev = [str(x) for x in (getattr(rule, "examples", None) or [])]
    because = str(getattr(rule, "added_because", "") or "")
    m = re.search(r"['\"](.+?)['\"]", because)
    if m:
        ev.append(m.group(1))
    return ev


def rules_against_evidence(
    rules: Sequence[Any],
    gold_rows: Sequence[Any],
    queries: Sequence[str] = (),
    *,
    min_rows: int = 8,
    codes: Sequence[str] = (),
) -> RuleEvidenceReport:
    """Measure every rule against the referee's own verdicts on its class pair.

    This is what a rule with no executable trigger can still be held to. It needs
    no regex, no extra agent call, and no data the run does not already have:
    the gold set records, for each disagreement, which pair it was on and how the
    referee settled it.

    `min_rows` is a floor on saying anything at all about a boundary — a
    direction read off three rows is noise. Decisiveness is judged by a Wilson
    interval rather than a fixed share, so the bar tightens on small samples on
    its own instead of importing a number that fitted one corpus.

    `codes` matters more here than anywhere else. The whole measurement is
    `rule.then == the referee's majority verdict`, and a `then` holding a
    sentence can never equal a class code — so without the class list, every
    prose rule counts as pointing AGAINST the evidence and a perfectly sound
    boundary is reported contradicted. live38 had 17 such rules; live39 had none,
    which is a prompt getting it right rather than a guarantee.
    """
    from .stats import wilson_interval

    rep = RuleEvidenceReport()
    qs = [str(q) for q in queries]

    # -- which rules earned a trigger -------------------------------------
    masks, _ = _pair_masks(rules, qs) if qs else ({}, [])
    lexical: dict[str, Any] = {}
    for r in rules:
        rid = str(getattr(r, "id", "") or "")
        trig = str(getattr(r, "trigger", "") or "")
        if not trig:
            rep.n_semantic += 1
            continue
        chk = validate_trigger(trig, qs, evidence=_rule_evidence(r)) if qs else TriggerCheck(True)
        if chk.ok and rid in masks:
            rep.n_lexical += 1
            lexical[rid] = masks[rid]
        else:
            # A trigger that cannot be trusted makes the rule semantic, not
            # broken. It keeps every other guarantee; it just stops claiming a
            # predicate it does not have.
            rep.n_semantic += 1
            rep.n_rejected_triggers += 1
            rep.rejected.append({"rule": rid, "trigger": trig[:60], "why": chk.reason})

    # -- what the referee actually did ------------------------------------
    by_pair: dict[tuple[str, ...], list[Any]] = {}
    for row in gold_rows:
        if not getattr(row, "adjudicated", False):
            continue
        final = str(getattr(row, "final", "") or "")
        if not final:
            continue
        key = tuple(sorted((str(getattr(row, "label_a", "")), str(getattr(row, "label_b", "")))))
        if len(key) == 2 and all(key):
            by_pair.setdefault(key, []).append(row)

    rules_by_pair: dict[tuple[str, ...], list[Any]] = {}
    for r in rules:
        cs = tuple(sorted(str(c) for c in (getattr(r, "classes", None) or [])))
        if len(cs) == 2:
            rules_by_pair.setdefault(cs, []).append(r)

    for pair, rows in by_pair.items():
        if len(rows) < min_rows:
            continue
        counts: dict[str, int] = {}
        for row in rows:
            f = str(row.final)
            counts[f] = counts.get(f, 0) + 1
        maj = max(counts, key=lambda k: counts[k])
        n, k = len(rows), counts[maj]
        lo, hi = wilson_interval(k, n)
        rs = rules_by_pair.get(pair, [])
        valid = set(codes)

        def _keyed(r: Any) -> bool:
            return (not valid) or str(getattr(r, "then", "")).strip() in valid

        unkeyed = [str(getattr(r, "id", "")) for r in rs if not _keyed(r)]
        rs = [r for r in rs if _keyed(r)]
        toward = [str(getattr(r, "id", "")) for r in rs if str(getattr(r, "then", "")) == maj]
        against = [str(getattr(r, "id", "")) for r in rs if str(getattr(r, "then", "")) != maj]
        rep.unkeyed_then.extend(unkeyed)
        rep.boundaries.append(BoundaryEvidence(
            classes=pair, n_adjudicated=n, verdicts=counts, majority=maj,
            majority_share=k / n, ci95=(lo, hi),
            # The whole interval above chance. A point estimate of 0.71 on 21 rows
            # is not a direction; its lower bound sitting above 0.5 is.
            decisive=lo > 0.5,
            n_rules=len(rs), rules_toward_majority=toward, rules_against_majority=against,
            rules_not_a_key=unkeyed,
            n_majority_rows=k, n_minority_rows=n - k,
            **_drafting_rates(rs, rows, maj),
        ))

        # -- the conditional test, for rules that earned a trigger ---------
        for r in rs:
            rid = str(getattr(r, "id", ""))
            mask = lexical.get(rid)
            if mask is None or not qs:
                continue
            idx = {int(getattr(row, "idx", -1)) for row in rows}
            fired = [row for row in rows
                     if 0 <= int(getattr(row, "idx", -1)) < len(mask)
                     and mask[int(getattr(row, "idx", -1))]]
            if not fired:
                rep.lexical_rules.append({
                    "rule": rid, "then": str(getattr(r, "then", "")),
                    "classes": list(pair), "n_rows_trigger_fired": 0,
                    "reading": ("the trigger fires on no adjudicated row of this "
                                "boundary — the rule is untested, not wrong"),
                })
                continue
            agree = sum(1 for row in fired if str(row.final) == str(getattr(r, "then", "")))
            rep.lexical_rules.append({
                "rule": rid, "then": str(getattr(r, "then", "")), "classes": list(pair),
                "n_rows_trigger_fired": len(fired), "n_referee_agreed": agree,
                "agreement": round(agree / len(fired), 4),
                "n_rows_on_boundary": len(idx),
                "reading": ("measured ON THE ROWS THE RULE CLAIMS, so a low value here "
                            "is not the exception-carving confound — it is the referee "
                            "disagreeing with the rule where the rule applies"),
            })

    rep.stated_grounds = stated_grounds(rules, gold_rows, min_rows=min_rows)
    rep.boundaries.sort(key=lambda b: -b.n_rules)
    return rep


def _drafting_rates(rules_on_pair: Sequence[Any], rows: Sequence[Any],
                    majority: str) -> dict[str, int]:
    """How many rules were drafted from a majority row vs a minority row.

    Read out of `added_because`, which records the query whose disagreement
    created the rule. This is the confound that retired the direction verdict, so
    it is measured on every boundary rather than assumed.
    """
    verdict_of = {str(getattr(r, "query", "")): str(getattr(r, "final", "")) for r in rows}
    maj = mino = 0
    for rule in rules_on_pair:
        m = re.search(r"['\"](.+?)['\"]", str(getattr(rule, "added_because", "") or ""))
        if not m:
            continue
        v = verdict_of.get(m.group(1))
        if v is None:
            continue
        if v == majority:
            maj += 1
        else:
            mino += 1
    return {"n_rules_from_majority_rows": maj, "n_rules_from_minority_rows": mino}


# ====================================================================
# `then` has to BE a class code, not merely mention one
# ====================================================================
@dataclass
class ThenResult:
    """What a rule's `then` field actually resolves to."""

    code: str | None                  # the single class, or None if it is not a key
    found: list[str] = field(default_factory=list)
    original: str = ""
    note: str = ""

    @property
    def is_key(self) -> bool:
        return bool(self.code)


def _code_pattern(code: str) -> Any:
    # Bounded on the token, so `OTHER` does not match inside `ANOTHER` and
    # `WORD_MEANING` does not match inside `WORD_MEANING_LOOKUP`.
    return re.compile(rf"(?<![A-Za-z0-9_]){re.escape(code)}(?![A-Za-z0-9_])")


def normalise_then(then: Any, codes: Sequence[str]) -> ThenResult:
    """Resolve `then` to ONE class code, or say honestly that it is not a key.

    `AdjudicationRule.then` is documented as "the class that wins", and on live38
    **18 of 132 rules held a whole sentence instead**:

        then = '归 JUDGE_LANGUAGE_USAGE，不归 LOOKUP_CHAR_PRONUNCIATION。'

    The existing validator asked `any(c in then for c in codes)` — whether the
    field *mentions* a real class — which that string passes while remaining
    unusable as a key. Everything downstream treats `then` as a key:
    `_dedupe_rules` compares it with `==` (so two phrasings of one answer read as
    a CONTRADICTION and both valid rules are withheld — observed live on R112 vs
    R053, where a hallucinated code variant cost two good rules), `find_conflicts`
    calls two rules compatible when their `then` strings match, and
    `rules_against_evidence` asks whether `then` equals the referee's majority
    verdict — which a sentence can never equal, so every prose rule would be
    counted as pointing AGAINST the evidence and the boundary would look
    contradicted when it is not.

    live39 shows 0 of 104, because `RuleWriterAgent` is shown the final class list
    and told to emit a code. That is a prompt getting it right, not a guarantee —
    hence this.

    Three outcomes, and the third is the point:

    - exactly one class named  -> that is the key; the original text is preserved
      by the caller into `rationale`, where the annotator still reads it
    - **two or more named**    -> genuinely ambiguous. `归 A，不归 B` means A, but
      `有裁决框架的归 A；单纯问 X 的归 B` is two rules in one sentence, and no
      mechanical reading separates them. Refuse to guess: mark it not-a-key so
      the comparisons above SKIP it instead of comparing garbage.
    - none named               -> left to the caller's typo repair, which is a
      different question (a near-miss on one code) and already handled.
    """
    raw = str(then or "").strip()
    if not raw:
        return ThenResult(None, [], raw, "empty")
    if raw in set(codes):
        return ThenResult(raw, [raw], raw)
    hits = [c for c in codes if c and _code_pattern(c).search(raw)]
    # Keep only maximal matches, so a code that is a strict substring of another
    # match is not counted twice.
    maximal = [c for c in hits if not any(c != o and c in o for o in hits)]
    if len(maximal) == 1:
        return ThenResult(maximal[0], maximal, raw,
                          "the field held a sentence naming exactly one class; "
                          "the sentence is kept in the rationale")
    if len(maximal) > 1:
        return ThenResult(None, maximal, raw,
                          f"names {len(maximal)} classes ({', '.join(maximal[:4])}) — "
                          "cannot be used as a key, and guessing which one wins would "
                          "silently pick a side")
    return ThenResult(None, [], raw, "names no declared class")


# ====================================================================
# Does the ground a rule STATES actually separate the boundary?
# ====================================================================
#: The referee enumerates its discriminators verbatim, e.g.
#: `无明确意图标记（如'什么意思'、'寓意'、'翻译'等）`. 31 of live39's 80
#: trigger-less rules do this — nearly four times the number that carry a
#: trigger, so this reaches rules no regex ever could.
_QUOTED = re.compile(r"[‘'“\"「]([^’'”\"」]{1,12})[’'”\"」]")
#: Negation immediately governing the marker list: the rule fires on ABSENCE.
_NEGATED = re.compile(r"(无|没有|不含|不包含|未|缺(?:少|乏)?)[^，。；]{0,8}$")


#: Separators the referee uses INSIDE one quoted span: `怎么读/意思/翻译/部首`
#: is four markers, not one, and testing it whole matches nothing.
_MARKER_SPLIT = re.compile(r"[/、,，|｜]+")
#: A quoted span that is a TEMPLATE rather than a literal: `X的意思`,
#: `X读yun还是jun`, `X与Y反应`. Built from a placeholder letter, it can never
#: match a real query — so counting it as "absent from every row" would report a
#: vacuous ground that is really just an unusable pattern. Excluded and named.
_TEMPLATE = re.compile(r"(?:^|[^A-Za-z])[XYZNxyz](?![A-Za-z])")


#: Longest a discriminator word plausibly is. Beyond this the quoted span is an
#: EXAMPLE QUERY, not a marker — `两三个人聚集在一起诗歌` is a query someone typed.
_MAX_MARKER_CHARS = 8


def usable_markers(quoted: Sequence[str]) -> tuple[list[str], list[str]]:
    """Split enumerations, drop what is not a literal marker.

    Rules quote two different things in `when`, and only one of them is a test:
    an enumerated discriminator (`如'什么意思'、'寓意'、'翻译'等`) and an example
    query (`'二年级青蛙卖泥塘'`). Counting an example as a marker makes a
    boundary look like its stated ground separates when the ground was really an
    arbitrary string matching one row.

    Three filters, each for a shape seen in live output:

    - **split enumerations** — `怎么读/意思/翻译/部首` is four markers, and testing
      it whole matches nothing, which would report a vacuous ground that is
      really an unusable pattern;
    - **drop templates** — `X的意思`, `X读yun还是jun` can never match a query;
    - **drop example-shaped spans** — containing a space or a digit, or longer
      than `_MAX_MARKER_CHARS`.

    **Documented limit.** A short, ordinary word quoted as an example — `孩子`,
    `课文` — is indistinguishable from a marker by shape, and will be treated as
    one. That direction is the safe one: it can only make a ground look like it
    separates, never manufacture a vacuous verdict. The extracted markers are
    printed in the record so a reader can see what was tested.

    Scoping extraction to spans introduced by 如/包含/含有 was tried and rejected:
    measured on live39 it kept 16 of 37 rules and threw away real marker lists
    (R010's `怎么读/意思/翻译/部首`, R016's `等于多少/计算`), losing far more signal
    than it removed noise.
    """
    usable, rejected = [], []
    for span in quoted:
        for part in _MARKER_SPLIT.split(str(span)):
            m = part.strip().strip("「」『』()（）").strip()
            if not m:
                continue
            if (_TEMPLATE.search(m) or len(m) > _MAX_MARKER_CHARS
                    or any(ch.isdigit() for ch in m) or " " in m or "\u3000" in m):
                rejected.append(m)
            else:
                usable.append(m)
    return sorted(set(usable)), sorted(set(rejected))


@dataclass
class StatedGround:
    """One boundary's stated discriminator, measured against its own verdicts."""

    classes: tuple[str, ...]
    markers: list[str]
    rejected_markers: list[str]
    rules_citing: list[str]
    n_rows: int
    n_matching: int
    verdicts_when_present: dict[str, int]
    verdicts_when_absent: dict[str, int]

    @property
    def coverage(self) -> float:
        return self.n_matching / max(1, self.n_rows)

    @property
    def separates(self) -> bool:
        """Does the stated marker actually split the boundary at all?

        Vacuous in both directions: a marker present in every row and a marker
        present in none divide nothing. live39's OTHER x TEXT_INTERPRETATION
        cites 什么意思 / 寓意 / 翻译 and **not one of its 21 adjudicated queries
        contains any of them** — so "no marker" is true of every row, including
        all 15 the referee ruled the other way.
        """
        return 0 < self.n_matching < self.n_rows

    def as_record(self) -> dict[str, Any]:
        return {
            "classes": list(self.classes), "markers": self.markers[:10],
            "quoted_spans_rejected_as_templates": self.rejected_markers[:8],
            "rules_citing_it": self.rules_citing, "n_adjudicated": self.n_rows,
            "n_rows_containing_a_stated_marker": self.n_matching,
            "coverage": round(self.coverage, 4),
            "separates_the_boundary": self.separates,
            "verdicts_when_marker_present": self.verdicts_when_present,
            "verdicts_when_marker_absent": self.verdicts_when_absent,
            "why_it_matters": (
                "These are the words the rules themselves name as the discriminator. "
                "If every row on the boundary falls on one side of that test, the test "
                "is not what decided the boundary — whatever the rules say — and an "
                "annotator applying it literally gets no guidance at all."),
        }


def stated_grounds(
    rules: Sequence[Any],
    gold_rows: Sequence[Any],
    *,
    min_rows: int = 8,
) -> list[StatedGround]:
    """Measure each boundary's stated marker test against its own adjudications.

    **Why this replaced counting rule directions.** The first version of this
    module flagged a boundary when its adjudicated majority was decisive and most
    of its rules pointed the other way. That signal is confounded, and the
    confound is this codebase's own named trap — testing a mechanism with a
    distribution it has already filtered. A referee drafts a rule only where it
    judges the guide to have FAILED, which is concentrated on the side it rules
    against the prevailing pattern: measured on live39's
    `OTHER x TEXT_INTERPRETATION`, **5 of 6 minority rows produced a rule (83%)
    against 1 of 15 majority rows (7%)**. "Most rules point away from the
    majority" is therefore the *expected* shape of a healthy exception set, and
    the old check would have fired identically on a guide with no defect.

    Two further defects, both measured: the Wilson bar had no resolution — 15/21
    gives a lower bound of **0.5004** against a threshold of 0.5, so one row in
    either direction decided whether the boundary was reported at all — and the
    "five rules" magnitude was one drafting template emitted five times, which
    measures the referee's habit rather than the number of conflicts.

    This test has none of those problems. It is per-boundary, it never counts
    which way a rule points, and it asks only whether the discriminator the rules
    THEMSELVES enumerate divides the rows the referee actually adjudicated.
    """
    by_pair_rows: dict[tuple[str, ...], list[Any]] = {}
    for row in gold_rows:
        if not getattr(row, "adjudicated", False) or not str(getattr(row, "final", "") or ""):
            continue
        key = tuple(sorted((str(getattr(row, "label_a", "")), str(getattr(row, "label_b", "")))))
        if len(key) == 2 and all(key):
            by_pair_rows.setdefault(key, []).append(row)

    by_pair_markers: dict[tuple[str, ...], tuple[set[str], set[str], list[str]]] = {}
    for r in rules:
        cs = tuple(sorted(str(c) for c in (getattr(r, "classes", None) or [])))
        if len(cs) != 2:
            continue
        found = _QUOTED.findall(str(getattr(r, "when", "") or ""))
        if not found:
            continue
        ok, bad = usable_markers(found)
        ms, rej, ids = by_pair_markers.setdefault(cs, (set(), set(), []))
        ms.update(ok)
        rej.update(bad)
        ids.append(str(getattr(r, "id", "")))

    out: list[StatedGround] = []
    for cs, (ms, rej, ids) in by_pair_markers.items():
        rows = by_pair_rows.get(cs) or []
        # No usable literal marker means there is no stated test to evaluate.
        # Reporting "the ground separates nothing" here would be a fact about our
        # extraction, not about the guide — the same fabricated-predicate trap
        # that `validate_trigger` exists to avoid.
        if len(rows) < min_rows or not ms:
            continue
        pat = re.compile("|".join(re.escape(m) for m in sorted(ms)))
        present: dict[str, int] = {}
        absent: dict[str, int] = {}
        for row in rows:
            bucket = present if pat.search(str(getattr(row, "query", ""))) else absent
            f = str(row.final)
            bucket[f] = bucket.get(f, 0) + 1
        out.append(StatedGround(
            classes=cs, markers=sorted(ms), rejected_markers=sorted(rej),
            rules_citing=sorted(set(ids)),
            n_rows=len(rows), n_matching=sum(present.values()),
            verdicts_when_present=present, verdicts_when_absent=absent,
        ))
    out.sort(key=lambda s: (s.separates, -s.n_rows))
    return out
