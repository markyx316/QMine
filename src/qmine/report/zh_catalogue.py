"""叶清单 (leaf catalogue) 的中文版本。

`Leaf_Catalogue.md` was 62% Chinese — the leaf names and `user_need` sentences
come from the namer and were already Chinese, wrapped in English scaffolding on a
run configured `zh`.

Beyond language, this version reports the thing the English one structurally could
not: **which delivered leaves have no name.** The old catalogue iterated
`tree_naming.namings`, so a leaf p8 governance created after p7 named the tree was
simply absent — 4,931 rows, 9.9% of live38's delivered table, silently missing
from a document titled "every leaf". Here the iteration is over the DELIVERED
partition, and anything unnamed is listed as a defect rather than omitted.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ._shape import family_names
from .i18n import pct


def build(state: Any, deps: Any) -> str:
    naming = deps.load("tree_naming") if deps.has("tree_naming") else {}
    labels = deps.leaf_labels_final() if deps.has("leaf_labels") else None
    fam = deps.leaf_family_final() if deps.has("leaf_family") else None
    gen_dir = Path(deps.store.gen_dir).name

    L = [
        "# 叶清单 (Leaf Catalogue)",
        "## 每一个叶子, 它的定义, 以及「用户拿到什么才算被满足」",
        "",
        f"**运行**: `{state.get('run_id')}` / {gen_dir} · **领域**: `{deps.cfg.domain.key}`",
        "",
        "每一条的 `user_need` 是**一句话**, 说明用户必须拿到什么才算被满足。"
        "它同时是**标注规范**、**验收标准**和**下游产品需求** —— 这就是为什么"
        "只给一个名字是不够的: **名字是可以各自理解的, 一句定义是可以被检验的。**", "",
    ]
    namings = naming.get("namings", []) or []
    if not namings or labels is None:
        return "\n".join(L + ["_本次运行没有可用的命名结果。_"])

    by_id = {int(n["leaf_id"]): n for n in namings}
    sizes = np.bincount(labels)
    total = int(len(labels))
    delivered = sorted({int(v) for v in np.unique(labels)})
    unnamed = [i for i in delivered if not str(by_id.get(i, {}).get("name_zh", "")).strip()]

    L += [f"- 交付划分中的叶子数: **{len(delivered)}**",
          f"- 已命名: **{len(delivered) - len(unnamed)}**",
          f"- 覆盖行数: **{total:,}**", ""]

    # The completeness statement the old catalogue could not make.
    if unnamed:
        miss = int(sum(int(sizes[i]) for i in unnamed))
        L += [f"> ❌ **有 {len(unnamed)} 个已交付的叶子没有名字** "
              f"(叶 {', '.join(str(i) for i in unnamed[:10])}"
              f"{' 等' if len(unnamed) > 10 else ''}), 共 **{miss:,}** 行 "
              f"({pct(miss / total)})。", "",
              "这通常意味着某个阶段在 p7 命名之后又改动了划分 —— 治理阶段的 "
              "`split_leaf` / `isolate_leaf` 会产生新的叶子。**这些行在交付表中"
              "`bu_leaf_name` 一列是空的**, 下游若按名字聚合会直接丢掉它们。", ""]
    else:
        L += [f"> ✅ **{len(delivered)} 个已交付的叶子全部有名字。** "
              "这一条是对**交付划分**做的检查, 而不是对命名阶段当时的划分做的检查 —— "
              "后者会在治理阶段新增叶子之后失效。", ""]

    # ---------------------------------------------------------------- 家族分组
    by_family: dict[int, list[int]] = {}
    for lid in delivered:
        f = int(fam[lid]) if fam is not None and lid < len(fam) else 0
        by_family.setdefault(f, []).append(lid)

    # See `family_names`: the auditor's family_id is a different namespace from
    # the partition's, and matching them by integer mislabelled every family.
    fam_names = family_names(naming, fam, sizes)

    L += ["---", ""]
    for f in sorted(by_family, key=lambda k: -sum(int(sizes[i]) for i in by_family[k])):
        members = sorted(by_family[f], key=lambda i: -int(sizes[i]))
        fn = int(sum(int(sizes[i]) for i in members))
        title = fam_names.get(f) or f"家族 {f}"
        L += [f"## {title}  (`family_{f}`)",
              f"*{fn:,} 行 · {pct(fn / total)} · {len(members)} 个叶子*", ""]
        for lid in members:
            n = by_id.get(lid, {})
            sz = int(sizes[lid])
            name = str(n.get("name_zh", "")).strip()
            if not name:
                L += [f"### 叶 {lid} — ⚠️ **未命名**",
                      f"- 规模: **{sz:,}** 行 ({pct(sz / total)})",
                      "- **本叶子没有名称与 user_need**, 交付表中该列为空。", ""]
                continue
            L += [f"### 叶 {lid} — {name}"
                  + (f"  (`{n.get('code')}`)" if n.get("code") else ""),
                  f"- 规模: **{sz:,}** 行 ({pct(sz / total)})",
                  f"- **user_need**: {n.get('user_need', '') or '—'}"]
            coh = n.get("coherence")
            if coh is not None:
                L.append(f"- 盲评连贯性: {coh}/5"
                         + (f" — {n.get('mix_notes')}" if n.get("mix_notes") else ""))
            if n.get("named_by"):
                L.append(f"- 命名者: `{n['named_by']}`")
            if n.get("risk_flag"):
                L.append(f"- ⚠️ **盲评命名者标记的风险**: {n.get('risk_reason', '')}")
            L.append("")
    return "\n".join(L)
