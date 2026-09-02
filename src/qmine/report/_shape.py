"""The shape of the tree that ships — not the shape p6 built.

`hierarchy_meta` is written in p6, BEFORE p8 governance changes the partition.
Reading it for a headline count made both reports claim live38 delivered "10
families / 29 leaves" when the delivered table held 12 and 36 — and the Chinese
report contradicted its own metrics table three lines below, which reads the
panel. The panel measures the partition it labels, so it is the source of truth
for anything described as final. `meta` remains correct where the prose says
"after refinement", which is what p6 actually produced.
"""
from __future__ import annotations

from typing import Any


def delivered_shape(panel: dict[str, Any], meta: dict[str, Any]) -> tuple[Any, Any]:
    """Return (n_families_final, n_leaves) as delivered, falling back to `meta`."""
    def _n(subject: str, meta_key: str) -> Any:
        try:
            v = panel["sets"][subject]["metrics"]["n_clusters"]["value"]
        except (KeyError, TypeError):
            v = None
        return int(v) if v is not None else meta.get(meta_key, "?")

    return _n("families_final", "n_families"), _n("leaves", "n_leaves")


def family_names(naming: dict[str, Any], leaf_family: Any, sizes: Any) -> dict[int, str]:
    """Map each PARTITION family id to a name, joining on `leaf_ids`.

    `tree_naming["audit"]["families"]` carries its own `family_id` numbering, and
    it is NOT the partition's. On live38 the auditor described **19** families
    while the partition had 10 (pre-governance) and 12 (final) — so looking a
    partition family up by integer id matched a different family every time.
    Measured: **19 of 19 audit families disagreed with the partition family of the
    same id**, and the shipped catalogue titled a family of four classical-poetry
    leaves "中考录取分数与学校排名查询" (high-school admission scores).

    The only sound join is through the leaves themselves. A partition family may
    span several audit families — governance merges make that normal — so it is
    named after whichever audit family covers the most of its rows, and the caller
    is told when the name is a partial description via the returned suffix.
    """
    # DELIVERED names win. `families_final` is written in p8 against the
    # partition that actually ships, so it needs no leaf-join and no composition
    # label. The audit's families describe the Phase 7 tree — a different id
    # space, and routinely several of them per delivered family, which is why
    # the fallback below can only say `混合·主要成分「X」N%`. That string is a
    # diagnostic, and it was being used AS the family's name in headings, table
    # cells, a Mermaid node and a CSV column.
    final = (naming or {}).get("families_final") or []
    if final:
        out = {int(f["family_id"]): str(f.get("name_zh") or "").strip()
               for f in final if f.get("name_zh")}
        if out:
            return out

    fams = ((naming or {}).get("audit", {}) or {}).get("families", []) or []
    by_leaf: dict[int, str] = {}
    for f in fams:
        for lid in (f.get("leaf_ids") or []):
            by_leaf[int(lid)] = str(f.get("name_zh") or "")
    if not by_leaf or leaf_family is None:
        return {}

    # ONE COMPUTATION, ONE TRUTH. The label is a projection of the composition,
    # not a second parallel tally — the two disagreed: a family whose single
    # audit contributor covered only 44% of its rows was labelled with that name
    # plainly, because the old branch counted only NAMED contributors and live42's
    # family 1 has 56% of its rows in a governance-created leaf with no audit
    # name at all. `exact` is the honest test: one contributor AND nothing
    # unnamed.
    comp = family_composition(naming, leaf_family, sizes)
    out: dict[int, str] = {}
    for fam, rec in comp.items():
        contributors = rec["contributors"]
        if not contributors:
            # EVERY DELIVERED FAMILY GETS A LABEL, including one the audit never
            # saw. Returning nothing left the caller to fall back to a bare id,
            # which reads as "unnamed" — but "governance built this entirely out
            # of leaves the audit did not cover" is a different and more useful
            # fact than "nobody named it".
            out[fam] = "树审计未覆盖 (治理新建)"
            continue
        if rec["exact"]:
            out[fam] = contributors[0][0]
        else:
            name, _rows, share = contributors[0]
            out[fam] = f"混合·主要成分「{name}」{share:.0%}"

    # One audit family can be spread over several partition families — on live38
    # "汉语字词释义查询" is the dominant name for three of them at once. Three
    # identically-titled sections a reader cannot tell apart is not better than a
    # wrong title, so a collided name is qualified by its own largest leaf.
    seen: dict[str, list[int]] = {}
    for fam, name in out.items():
        seen.setdefault(name, []).append(fam)
    leaf_name = {int(n["leaf_id"]): str(n.get("name_zh") or "")
                 for n in ((naming or {}).get("namings") or [])}
    for name, fams_sharing in seen.items():
        if len(fams_sharing) < 2:
            continue
        for fam in fams_sharing:
            mine = [lid for lid in leaf_name
                    if lid < len(leaf_family) and int(leaf_family[lid]) == fam
                    and leaf_name[lid]]
            if not mine:
                out[fam] = f"{name} (family {fam})"
                continue
            biggest = max(mine, key=lambda lid: int(sizes[lid]) if sizes is not None else 0)
            out[fam] = f"{name} · 主要叶「{leaf_name[biggest]}」"
    return out


def family_composition(naming: dict[str, Any], leaf_family: Any,
                       sizes: Any) -> dict[int, dict[str, Any]]:
    """What each DELIVERED family is actually made of, with honest denominators.

    `family_names` has to fit a table cell, so it can only say which component is
    largest. This is the rest of the story, and it exists because compressing it
    into a label produced a percentage whose denominator was neither the family
    nor the corpus.

    Per delivered family:

    * ``exact`` — one audit family covers every named leaf, so its name really is
      this family's name.
    * ``contributors`` — ``[(audit_name, rows, share_of_family)]``, largest
      first. The share is **of the family**, so the numbers on one family sum to
      the named fraction rather than to 1.
    * ``unnamed_leaves`` / ``unnamed_rows`` — leaves p8 governance created after
      p7 named the tree. They belong to the family and to no audit family, and
      the old label left them out of its own denominator.
    """
    fams = ((naming or {}).get("audit", {}) or {}).get("families", []) or []
    by_leaf: dict[int, str] = {}
    for f in fams:
        for lid in (f.get("leaf_ids") or []):
            by_leaf[int(lid)] = str(f.get("name_zh") or "")
    if leaf_family is None:
        return {}

    out: dict[int, dict[str, Any]] = {}
    for lid in range(len(leaf_family)):
        fam = int(leaf_family[lid])
        n = int(sizes[lid]) if sizes is not None and lid < len(sizes) else 0
        if not n:                     # a leaf with no rows is not in the delivery
            continue
        rec = out.setdefault(fam, {"rows": 0, "weight": {},
                                   "unnamed_leaves": 0, "unnamed_rows": 0})
        rec["rows"] += n
        nm = by_leaf.get(lid, "")
        if nm:
            rec["weight"][nm] = rec["weight"].get(nm, 0) + n
        else:
            rec["unnamed_leaves"] += 1
            rec["unnamed_rows"] += n

    for fam, rec in out.items():
        total = max(1, rec["rows"])
        rec["contributors"] = [
            (nm, n, n / total)
            for nm, n in sorted(rec["weight"].items(), key=lambda kv: -kv[1])]
        rec["exact"] = len(rec["contributors"]) == 1 and not rec["unnamed_leaves"]
        rec.pop("weight")
    return out
