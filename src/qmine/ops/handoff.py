"""Running the blind-naming protocol outside this process.

The naming step is the one place where a stronger model, or an actual panel of
humans, is worth the cost — a mis-named family propagates into the catalogue,
the report, and every downstream consumer.  So the protocol is exportable: the
same cards the built-in namers see can be written to disk, handed to whatever
panel you like, and the verdicts read back in.

The blindness guarantee travels with the cards.  :func:`export_shards` renders
through the firewall exactly as the in-process namer does, so an exported shard
is checked for label vocabulary before it is written — a panel working from
these files is under the same constraint as the agents are, and neither can be
handed an answer by accident.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from ..memory.context import BlindnessFirewall, render_card
from ..records import LeafNaming, NamingCard

INSTRUCTIONS = """\
# Blind cluster naming — shard {shard} of {total}

You are naming groups of search queries that an algorithm placed together.

**You have been told nothing else, and that is deliberate.** There is no
existing category list and no other reviewer's answers. If you had them you
would file these clusters under them — everyone does — and the data's actual
shape would vanish behind labels it was never measured against.

For each cluster below, return one JSON object with these fields:

| field | meaning |
|---|---|
| `leaf_id` | copy it from the heading |
| `name_zh` | action-object phrase for what the user wants the system to **do** ("汉字组词查询", not "汉字") |
| `code` | English snake_case |
| `user_need` | one sentence: "having received X, the user is satisfied" — concrete enough to check |
| `coherence` | 1-5. 5 = every member wants the same thing; 3 = two intents mixed; 1 = noise |
| `mix_notes` | if coherence <= 3, name the distinct intents you can see |
| `risk_flag` | true if answering these naively is a safety, legal, or policy problem |
| `risk_reason` | why, if flagged |

Write `name_zh` and `user_need` **in the language the member queries are written
in** — a definition sentence in a different language from the data cannot be
checked against the data by the people who own it.

Return a JSON array of exactly {n} objects, one per cluster, and nothing else.

Judge only what is in front of you. The **edge members** are included on purpose:
they are the ones that barely belong, and they are where impurity shows.

---

"""


def export_shards(
    cards: Sequence[NamingCard],
    out_dir: str | Path,
    *,
    n_shards: int = 5,
    firewall: BlindnessFirewall | None = None,
) -> dict[str, Any]:
    """Write one self-contained markdown brief per shard.

    Sharding matters as much as blinding: reviewers who see the whole tree start
    naming clusters *relative to each other* and produce a taxonomy rather than
    independent judgments, which is precisely what the audit step needs to be
    independent of.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fw = firewall or BlindnessFirewall()

    ordered = sorted(cards, key=lambda c: c.leaf_id)
    shards: list[list[NamingCard]] = [[] for _ in range(max(n_shards, 1))]
    for i, c in enumerate(ordered):
        shards[i % len(shards)].append(c)

    written: list[dict[str, Any]] = []
    for i, shard in enumerate(shards):
        if not shard:
            continue
        body = INSTRUCTIONS.format(shard=i + 1, total=len([s for s in shards if s]), n=len(shard))
        body += "\n\n---\n\n".join(render_card(c, firewall=fw) for c in shard)
        path = out / f"shard_{i + 1:02d}.md"
        path.write_text(body, encoding="utf-8")
        written.append({"shard": i + 1, "path": str(path), "leaf_ids": [c.leaf_id for c in shard],
                        "n_clusters": len(shard), "chars": len(body)})

    manifest = {
        "n_cards": len(ordered),
        "n_shards": len(written),
        "shards": written,
        "firewall": fw.summary(),
        "contract": (
            "each shard was rendered through the blindness firewall: it contains member "
            "queries and n-grams only, with no taxonomy names, legacy labels, or other "
            "reviewers' answers. Reviewers must not be shown one another's output."
        ),
        "return_format": "a JSON array per shard, or one merged array, matching LeafNaming",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def import_namings(payload: Any, *, named_by: str = "external-panel") -> list[LeafNaming]:
    """Read verdicts back, tolerating the shapes a panel actually returns.

    Accepts a bare array, a ``{"namings": [...]}`` wrapper, or a list of
    per-shard arrays — because insisting on one exact envelope turns a five-minute
    handoff into a formatting argument.
    """
    if isinstance(payload, (str, Path)) and Path(payload).exists():
        payload = json.loads(Path(payload).read_text(encoding="utf-8"))
    elif isinstance(payload, str):
        payload = json.loads(payload)

    rows: list[dict[str, Any]] = []

    def _collect(obj: Any) -> None:
        if isinstance(obj, dict):
            if "leaf_id" in obj:
                rows.append(obj)
            else:
                for v in obj.values():
                    _collect(v)
        elif isinstance(obj, list):
            for v in obj:
                _collect(v)

    _collect(payload)

    out: list[LeafNaming] = []
    seen: set[int] = set()
    for r in rows:
        try:
            lid = int(r["leaf_id"])
        except (KeyError, TypeError, ValueError):
            continue
        if lid in seen:
            continue
        seen.add(lid)
        out.append(LeafNaming(
            leaf_id=lid,
            name_zh=str(r.get("name_zh") or r.get("name") or ""),
            code=str(r.get("code") or ""),
            user_need=str(r.get("user_need") or ""),
            coherence=int(r.get("coherence") or 3),
            mix_notes=str(r.get("mix_notes") or ""),
            risk_flag=bool(r.get("risk_flag")),
            risk_reason=str(r.get("risk_reason") or ""),
            named_by=str(r.get("named_by") or named_by),
        ))
    return sorted(out, key=lambda n: n.leaf_id)


def coverage_report(namings: Sequence[LeafNaming], expected_ids: Sequence[int]) -> dict[str, Any]:
    """Which clusters came back, and which were dropped.

    A panel that silently skips ten clusters leaves ten unnamed leaves in the
    catalogue, so the gap is surfaced rather than discovered later.
    """
    got = {n.leaf_id for n in namings}
    want = set(int(i) for i in expected_ids)
    return {
        "n_expected": len(want),
        "n_received": len(got & want),
        "missing": sorted(want - got),
        "unexpected": sorted(got - want),
        "complete": not (want - got),
    }


# ==========================================================================
# Phase 12 — comparing two runs
# ==========================================================================

def diff_runs(previous: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    """Compare two maintenance baselines and separate drift from method change.

    The first thing this checks is the config hash, and it refuses to interpret
    anything else until that question is settled. Two trees built with different
    encoders, alphas or K values are not comparable, and a "new family" between
    them may be nothing but a different random initialisation. Reading method
    change as content drift is the characteristic way a maintenance loop
    generates false alarms and then gets ignored.
    """
    pb = previous.get("baseline", previous)
    cb = current.get("baseline", current)
    same_config = pb.get("config_hash") == cb.get("config_hash")

    prev_names = {int(k): v for k, v in (pb.get("leaf_names") or {}).items()}
    cur_names = {int(k): v for k, v in (cb.get("leaf_names") or {}).items()}
    prev_sizes = pb.get("leaf_sizes") or []
    cur_sizes = cb.get("leaf_sizes") or []

    by_name_prev = {v: k for k, v in prev_names.items() if v}
    by_name_cur = {v: k for k, v in cur_names.items() if v}
    appeared = sorted(set(by_name_cur) - set(by_name_prev))
    vanished = sorted(set(by_name_prev) - set(by_name_cur))

    grown, shrunk = [], []
    tot_p, tot_c = max(sum(prev_sizes), 1), max(sum(cur_sizes), 1)
    for name in sorted(set(by_name_prev) & set(by_name_cur)):
        pi, ci = by_name_prev[name], by_name_cur[name]
        if pi >= len(prev_sizes) or ci >= len(cur_sizes):
            continue
        ps, cs = prev_sizes[pi] / tot_p, cur_sizes[ci] / tot_c
        delta = cs - ps
        if abs(delta) < 0.005:
            continue
        row = {"name": name, "prev_share": round(ps, 4), "cur_share": round(cs, 4),
               "delta": round(delta, 4)}
        (grown if delta > 0 else shrunk).append(row)
    grown.sort(key=lambda r: -r["delta"])
    shrunk.sort(key=lambda r: r["delta"])

    return {
        "config_comparable": same_config,
        "config_hashes": {"previous": pb.get("config_hash"), "current": cb.get("config_hash")},
        "shape": {
            "previous": {"families": pb.get("n_families"), "leaves": pb.get("n_leaves"),
                         "alpha": pb.get("alpha"), "family_k": pb.get("family_k")},
            "current": {"families": cb.get("n_families"), "leaves": cb.get("n_leaves"),
                        "alpha": cb.get("alpha"), "family_k": cb.get("family_k")},
        },
        "appeared": appeared,
        "vanished": vanished,
        "grown": grown[:15],
        "shrunk": shrunk[:15],
        "verdict": (
            "comparable — differences below are candidate content drift"
            if same_config else
            "NOT COMPARABLE — the config hash changed, so encoder/alpha/K may differ. "
            "Any 'new family' here may be method change rather than drift. Re-run the "
            "previous config on the new data before drawing conclusions."
        ),
    }
