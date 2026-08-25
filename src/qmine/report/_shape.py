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
    fams = ((naming or {}).get("audit", {}) or {}).get("families", []) or []
    by_leaf: dict[int, str] = {}
    for f in fams:
        for lid in (f.get("leaf_ids") or []):
            by_leaf[int(lid)] = str(f.get("name_zh") or "")
    if not by_leaf or leaf_family is None:
        return {}

    weight: dict[int, dict[str, int]] = {}
    for lid, name in by_leaf.items():
        if lid >= len(leaf_family) or not name:
            continue
        fam = int(leaf_family[lid])
        n = int(sizes[lid]) if sizes is not None and lid < len(sizes) else 1
        weight.setdefault(fam, {})
        weight[fam][name] = weight[fam].get(name, 0) + n

    out: dict[int, str] = {}
    for fam, names in weight.items():
        ranked = sorted(names.items(), key=lambda kv: -kv[1])
        best, share = ranked[0][0], ranked[0][1] / max(1, sum(names.values()))
        out[fam] = best if len(ranked) == 1 else f"{best} 等 {len(ranked)} 类 ({share:.0%})"

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
