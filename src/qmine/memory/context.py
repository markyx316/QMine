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

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence


class BlindnessViolation(AssertionError):
    """Raised when label vocabulary reaches a payload that must be blind."""


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


def budget_text(text: str, max_chars: int, *, tail: int = 0) -> str:
    """Trim a long block to a character budget, keeping head and optionally tail.

    Used wherever a prompt embeds evidence whose size we do not control (a data
    audit, a metrics panel).  Truncation is announced in-band so the agent knows
    it is reasoning over an excerpt.
    """
    if len(text) <= max_chars:
        return text
    if tail <= 0:
        return text[:max_chars] + f"\n… [truncated {len(text) - max_chars} chars]"
    head = max_chars - tail
    return (
        text[:head]
        + f"\n… [truncated {len(text) - max_chars} chars] …\n"
        + text[-tail:]
    )
