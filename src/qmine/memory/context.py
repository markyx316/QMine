"""The blindness firewall — Principle 5, turned from a promise into an assertion.

Anchoring is not a hypothetical risk in taxonomy work; it is the default
outcome.  Show a naming agent the existing category list and it will file
clusters under those categories, because that is what a helpful assistant does.
The tree you get back is then a picture of the old taxonomy wearing new
coordinates, and the data's actual shape stays invisible.

So the naming agents are told nothing.  Not the top-down intent names, not the
legacy labels, not each other's answers, not even the fact that a taxonomy
exists.  They see thirty member queries and some n-grams.

Enforcing that by careful prompt-writing would be enforcing it by hope.  This
module instead builds the forbidden vocabulary from the actual label sources in
the run and *scans every payload* before it can reach a prompt.  A leak raises;
it does not warn.  The one thing that makes the risk-cluster finding credible —
that an agent which was never told about gambling flagged the gambling cluster
anyway — is only true if this check has teeth.
"""

from __future__ import annotations

import logging

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence


class BlindnessViolation(AssertionError):
    """Raised when label vocabulary reaches a payload that must be blind."""


log = logging.getLogger("qmine.prompt")

def _normalise(s: str) -> str:
    return re.sub(r"\s+", "", str(s)).lower()


@dataclass
class BlindnessFirewall:
    """Holds the forbidden vocabulary for one run and checks payloads against it.

    ``min_token_len`` exists because short strings produce false positives — a
    two-character class name will appear inside ordinary queries by chance, and a
    firewall that cries wolf gets disabled.  Tokens shorter than the threshold
    are checked only as whole-field equality rather than substring containment.
    """

    forbidden: set[str] = field(default_factory=set)
    min_token_len: int = 3
    allow: set[str] = field(default_factory=set)

    # -- construction -------------------------------------------------------
    def add_terms(self, terms: Iterable[str]) -> "BlindnessFirewall":
        for t in terms:
            t = str(t).strip()
            if t and _normalise(t) not in {_normalise(a) for a in self.allow}:
                self.forbidden.add(t)
        return self

    def add_taxonomy(self, taxonomy: Any) -> "BlindnessFirewall":
        """Every name and code from the top-down route becomes forbidden."""
        if taxonomy is None:
            return self
        if hasattr(taxonomy, "label_vocabulary"):
            return self.add_terms(taxonomy.label_vocabulary())
        if isinstance(taxonomy, dict):
            for n in taxonomy.get("nodes", []):
                self.add_terms([n.get("name", ""), n.get("code", "")])
        return self

    def add_reference_labels(self, values: Iterable[str]) -> "BlindnessFirewall":
        """Legacy taxonomy labels — the most seductive anchor of all."""
        return self.add_terms({str(v) for v in values})

    def add_peer_outputs(self, namings: Sequence[Any]) -> "BlindnessFirewall":
        """Other naming agents' answers.  Shards must not see one another."""
        for n in namings or []:
            self.add_terms([getattr(n, "name_zh", ""), getattr(n, "code", "")])
        return self

    # -- enforcement --------------------------------------------------------
    def scan(self, payload: Any, *, path: str = "$") -> list[dict[str, str]]:
        """Walk a nested payload and return every leak found."""
        leaks: list[dict[str, str]] = []
        if isinstance(payload, dict):
            for k, v in payload.items():
                leaks += self.scan(k, path=f"{path}.<key>")
                leaks += self.scan(v, path=f"{path}.{k}")
        elif isinstance(payload, (list, tuple, set)):
            for i, v in enumerate(payload):
                leaks += self.scan(v, path=f"{path}[{i}]")
        elif hasattr(payload, "model_dump"):
            leaks += self.scan(payload.model_dump(), path=path)
        elif isinstance(payload, str):
            norm = _normalise(payload)
            for term in self.forbidden:
                nt = _normalise(term)
                if not nt:
                    continue
                hit = (nt == norm) if len(nt) < self.min_token_len else (nt in norm)
                if hit:
                    leaks.append({"path": path, "term": term, "value": payload[:120]})
        return leaks

    def assert_blind(self, payload: Any, *, what: str = "payload") -> None:
        """Raise unless ``payload`` is free of every forbidden term."""
        leaks = self.scan(payload)
        if leaks:
            head = "; ".join(f"{l['term']!r} at {l['path']}" for l in leaks[:5])
            raise BlindnessViolation(
                f"{what} leaks {len(leaks)} label term(s) into a blind-review context: {head}. "
                "A naming agent that sees existing labels will file clusters under them "
                "(Principle 5), so this is a hard failure rather than a warning."
            )

    # -- the card check -----------------------------------------------------
    #
    # Fields whose content comes verbatim (or by direct derivation) from the
    # corpus. These are exempt from the LEXICAL scan, and the reason is worth
    # stating because getting it wrong makes the firewall unusable.
    #
    # A good category name is drawn from its domain's own vocabulary, so legacy
    # labels and ordinary query words overlap heavily. On the K12 corpus the
    # legacy label "作文" appears inside the genuine query "我的自画像作文350字".
    # Scanning member queries lexically flagged that row as a leak and silently
    # dropped ten clusters from the naming pass — a false positive severe enough
    # to hole the deliverable.
    #
    # A member query cannot *anchor* a namer, because it is the thing being
    # judged. What anchors a namer is label vocabulary arriving as ANNOTATION —
    # a hint field, a peer's answer, a category the card was filed under. So the
    # real check is structural: the card may carry these fields and no others.
    CORPUS_DERIVED = frozenset({"center_samples", "random_samples", "edge_samples", "top_ngrams"})
    CARD_FIELDS = frozenset({
        "leaf_id", "size", "share", "center_samples", "random_samples",
        "edge_samples", "top_ngrams", "length_stats",
    })

    def assert_card_blind(self, card: Any, *, what: str = "naming card") -> None:
        """The Phase 7 check: structure first, then lexical scan of the rest.

        1. **Field whitelist.** Any key outside :attr:`CARD_FIELDS` raises. This
           is what actually catches anchoring — a smuggled ``legacy_label`` or
           ``taxonomy_hint`` field cannot pass, whatever it contains.
        2. **Lexical scan** of everything that is not corpus-derived, so a label
           smuggled into a prose field is still caught.
        """
        data = card.model_dump() if hasattr(card, "model_dump") else dict(card)
        extra = set(data) - self.CARD_FIELDS
        if extra:
            raise BlindnessViolation(
                f"{what} carries field(s) {sorted(extra)} that are not part of the blind "
                "card contract. A namer must see member queries and n-grams — anything "
                "else is annotation, and annotation is what anchors (Principle 5)."
            )
        self.assert_blind(
            {k: v for k, v in data.items() if k not in self.CORPUS_DERIVED and k not in
             ("leaf_id", "size", "share", "length_stats")},
            what=what,
        )

    def summary(self) -> dict[str, Any]:
        return {
            "n_forbidden_terms": len(self.forbidden),
            "min_token_len": self.min_token_len,
            "sample": sorted(self.forbidden)[:15],
        }


# --------------------------------------------------------------------------
# Context assembly
# --------------------------------------------------------------------------

def render_card(card: Any, *, firewall: BlindnessFirewall | None = None) -> str:
    """Render a naming card to the exact text an agent will see.

    Rendering and checking happen in the same function on purpose: if the check
    lived at the call site, someone would eventually build a prompt without it.
    """
    data = card.model_dump() if hasattr(card, "model_dump") else dict(card)
    if firewall is not None:
        firewall.assert_card_blind(card, what=f"naming card for leaf {data.get('leaf_id')}")
    lines = [
        f"## Cluster {data['leaf_id']}",
        f"size: {data['size']} rows ({data['share'] * 100:.2f}% of corpus)",
        "",
        "### Members closest to the centre",
    ]
    lines += [f"- {s}" for s in data.get("center_samples", [])]
    lines += ["", "### Random members"]
    lines += [f"- {s}" for s in data.get("random_samples", [])]
    lines += ["", "### Members at the edge (these reveal impurity — judge them too)"]
    lines += [f"- {s}" for s in data.get("edge_samples", [])]
    if data.get("top_ngrams"):
        lines += ["", "### Distinctive n-grams", ", ".join(data["top_ngrams"])]
    if data.get("length_stats"):
        st = data["length_stats"]
        lines += [
            "",
            f"### Length: mean {st.get('mean', 0):.1f}, median {st.get('median', 0):.0f}, "
            f"max {st.get('max', 0):.0f} characters",
        ]
    return "\n".join(lines)


#: Characters reserved so the withheld-count note itself always fits.
_WITHHELD_NOTE_ROOM = 220


def budget_text(text: str, max_chars: int, *, tail: int = 0, label: str = "") -> str:
    """Trim a long block to a character budget, keeping head and optionally tail.

    Used wherever a prompt embeds evidence whose size we do not control (a data
    audit, a metrics panel).  Truncation is announced in-band so the agent knows
    it is reasoning over an excerpt.

    AND ANNOUNCED IN THE LOG. The in-band marker tells the *model* it is reading
    an excerpt; it tells the operator nothing, because nobody reads the prompt.
    On `live38` the referee drafted 83 adjudication rules, the rendered rule
    block reached 18,496 characters against a 9,000 budget, and the entire
    referee contribution was cut — silently — from every annotation prompt of
    the guide-repair round whose whole purpose was to apply it. The measured
    "guide repair does nothing" result had been obtained that way three times.
    Pass `label` so the log says WHICH block lost content.
    """
    if len(text) <= max_chars:
        return text

    lost = len(text) - max_chars
    log.warning("prompt block%s truncated: %d of %d chars dropped (%.0f%% kept)%s",
                f" {label!r}" if label else "", lost, len(text),
                100.0 * max_chars / max(len(text), 1),
                "" if tail else " — HEAD ONLY, so anything appended is lost first")

    if tail <= 0:
        return text[:max_chars] + f"\n… [truncated {lost} chars]"
    head = max_chars - tail
    return (
        text[:head]
        + f"\n… [truncated {lost} chars] …\n"
        + text[-tail:]
    )


#: The character budget the researcher evidence block gets. A PROMPT-CONTEXT
#: number, not a corpus number — it says how much a model is given to read, and
#: it is the same on every corpus. Defined here so `agents/roles.py` (which
#: enforces it) and `graph/nodes/topdown.py` (which builds within it) cannot
#: drift apart; they did, and the builder had no idea what it had to fit inside.
RESEARCHER_EVIDENCE_CHARS = 24_000


def fair_caps(lengths: Sequence[int], budget: int) -> list[int]:
    """Max-min fair allocation of `budget` across items of these lengths.

    Progressive filling: everything at or under the current equal share is
    satisfied in full and releases what it did not use; the remainder is shared
    among the rest; repeat. Items still over the share at the end split what is
    left equally.

    Carries no absolute length, which is the point — the share is derived from
    the budget and the item count, so it fits a corpus of ten-character queries
    and one of thousand-character documents without being retuned for either.
    """
    lengths = list(lengths)
    if not lengths:
        return []
    caps = [0] * len(lengths)
    unsettled = set(range(len(lengths)))
    remaining = budget
    while unsettled:
        share = remaining // len(unsettled)
        fits = [i for i in unsettled if lengths[i] <= share]
        if not fits:
            for i in unsettled:
                caps[i] = share
            break
        for i in fits:
            caps[i] = lengths[i]
            remaining -= lengths[i]
            unsettled.discard(i)
    return caps


def fair_excerpts(
    items: Sequence[str],
    max_chars: int,
    *,
    unit: str = "item",
    label: str = "",
) -> list[str]:
    """Cap each item at its FAIR SHARE, so no one item crowds the others out.

    THE THIRD STRATEGY, and the only one that keeps every item.  `budget_text`
    cuts the block mid-string and severs whatever is at the cut.  `budget_units`
    keeps whole units from the head and DROPS THE TAIL.  Both are wrong when
    every item must be present — a list of risk categories, say — and one item
    happens to be enormous.

    Measured on `ai02`: the risk_compliance evidence came to 37,205 characters
    against a 24,000 budget, and `self_harm`'s six samples were **22,244 of
    them** (two at 8,194) because on a conversational corpus its hits are long
    fiction prompts.  `budget_text` cut inside `self_harm`, so
    `medical_self_diagnosis`, `financial_advice`, `legal_outcome_prediction` and
    `circumvention` — 440 characters between them — never reached the one
    researcher whose whole assignment is safety.

    **NO ABSOLUTE LENGTH APPEARS HERE, deliberately.**  An earlier fix capped
    every row at a flat 300 characters, which is a K12-style imported constant:
    tuned on a corpus whose median row is 10 characters, and on a corpus of
    contracts or code (median row 800) it would gut every row while fixing
    nothing.  `test_gates_do_not_import_thresholds_that_only_fit_one_corpus`
    exists for exactly that mistake.  The share is derived from the budget and
    the item count instead, so the same code adapts to any corpus.

    Max-min fair allocation (progressive filling): items at or under the current
    equal share keep their FULL text and release what they did not use; the
    released budget is redistributed among the rest; repeat until only items
    above the share remain, and those split what is left equally.

    **If everything already fits, this returns the items unchanged and logs
    nothing.**  That is the property that makes it safe to adopt everywhere:
    on every corpus this project has run before — maximum row 28 to 64
    characters — it is a no-op.
    """
    items = [str(i) for i in items]
    if not items:
        return []
    lengths = [len(i) for i in items]
    if sum(lengths) <= max_chars:
        return items

    caps = fair_caps(lengths, max_chars)

    out: list[str] = []
    n_cut = 0
    for text, cap in zip(items, caps):
        if len(text) <= cap:
            out.append(text)
            continue
        n_cut += 1
        # The omission is stated in-band. A silently shortened example reads as
        # a complete one, which is how a model concludes things about content it
        # was never shown.
        out.append(text[:cap] + f"…[+{len(text) - cap} chars]")
    log.warning("prompt block%s: %d of %d %s(s) excerpted to fit %d chars "
                "(fair share %d chars; longest was %d)",
                f" {label!r}" if label else "", n_cut, len(items), unit,
                max_chars, max_chars // max(len(items), 1), max(lengths))
    return out


def budget_units(
    units: Sequence[str],
    max_chars: int,
    *,
    joiner: str = "\n",
    unit: str = "item",
    label: str = "",
) -> str:
    """Trim to a budget by whole UNITS, and say how many were withheld.

    `budget_text` cuts mid-string, which fails in two ways this one does not.

    It severs a unit. A rule, a document or a table row is cut in half, so the
    reader — human or model — gets a fragment that looks whole.

    And it is silent about COUNT. On `live44` the delivery auditor was handed
    `budget_text(deliverables, 90000, tail=12000)`, was shown **39%** of the
    documents (142,957 of 232,957 characters dropped out of the MIDDLE), and
    returned an audit of "the deliverables" — it had no way to know it had seen
    a third of them. The in-band marker `budget_text` writes gives a character
    count, which no reader can convert into "which documents am I missing".

    So this reports the count in the unit the caller actually thinks in, in the
    log AND in the prompt, and tells the model in words not to generalise from
    what it was shown. Ordering is the caller's job: units are kept from the
    HEAD, so pass them most-important-first.
    """
    units = list(units)
    whole = joiner.join(units)
    if len(whole) <= max_chars:
        return whole

    room = max_chars - _WITHHELD_NOTE_ROOM
    kept: list[str] = []
    used = 0
    for u in units:
        cost = len(u) + len(joiner)
        if used + cost > room:
            break
        kept.append(u)
        used += cost

    # Even one unit does not fit. A character cut is worse than a fragment of
    # nothing, so fall back rather than return an empty block — but say so.
    if not kept:
        log.warning("prompt block%s: not even one %s fits in %d chars — "
                    "falling back to a character cut",
                    f" {label!r}" if label else "", unit, max_chars)
        return budget_text(whole, max_chars, label=label)

    dropped = len(units) - len(kept)
    log.warning("prompt block%s truncated: %d of %d %s(s) withheld "
                "(%d of %d chars kept)",
                f" {label!r}" if label else "", dropped, len(units), unit,
                used, len(whole))
    return joiner.join(kept) + (
        f"{joiner}… [{dropped} of {len(units)} {unit}s withheld for length. "
        f"You are seeing {len(kept)}. Do NOT describe this as the complete set, "
        f"and do not conclude anything about the {unit}s you were not shown.]"
    )
