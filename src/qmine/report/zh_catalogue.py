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

    # ---------------------------------------------------------------- 风险分布
    # The blind namer flags a leaf when its SAMPLE looks risky. The risk screen
    # scans every row. On live38 those disagreed badly: 1,499 rows were screened
    # as risky and 813 of them (54%) sat in leaves the namer had not flagged, so
    # a reader trusting the catalogue's risk marks would have missed most of the
    # risk. `risk_screen.json` appeared nowhere in any deliverable.
    risk_by_leaf: dict[int, int] = {}
    risk_total = 0
    try:
        rs = deps.load("risk_screen") if deps.has("risk_screen") else {}
        idx = rs.get("flag_mask_indices") or []
        risk_total = int(rs.get("total_flagged") or len(idx))
        for i in idx:
            if 0 <= int(i) < len(labels):
                lid = int(labels[int(i)])
                risk_by_leaf[lid] = risk_by_leaf.get(lid, 0) + 1
    except Exception:  # noqa: BLE001
        pass
    if risk_by_leaf:
        namer_flagged = {i for i in delivered if by_id.get(i, {}).get("risk_flag")}
        inside = sum(n for l, n in risk_by_leaf.items() if l in namer_flagged)
        outside = sum(risk_by_leaf.values()) - inside
        L += ["## 风险行的实际分布", "",
              f"- 风险筛查命中: **{risk_total:,}** 行, 分布在 **{len(risk_by_leaf)}** 个叶中",
              f"- 位于**盲评命名者标记为风险**的叶内: **{inside:,}** 行",
              f"- 位于**未被标记**的叶内: **{outside:,}** 行"
              + (f" ({pct(outside / max(1, inside + outside))})" if inside + outside else ""),
              ""]
        if outside:
            L += ["> ⚠️ **命名者的风险标记不能当作风险清单来用。** 它看的是每个叶的"
                  "**抽样卡片**, 而风险筛查扫的是**每一行** —— 两者本就不该一致。"
                  "下面每个叶都单独给出自己的风险命中数, 请按这个数做处置, "
                  "不要只看「风控标记」那一行。", ""]
        top = sorted(risk_by_leaf.items(), key=lambda kv: -kv[1])[:10]
        L += ["| 叶 | 名称 | 风险命中 | 占该叶 | 命名者标记 |", "|---|---|---|---|---|"]
        for lid, n in top:
            nm = str(by_id.get(lid, {}).get("name_zh", "")) or f"(未命名 {lid})"
            share = pct(n / max(1, int(sizes[lid])))
            mark = "✓" if lid in namer_flagged else "—"
            L.append(f"| {lid} | {nm} | {n:,} | {share} | {mark} |")
        L.append("")

    # ---------------------------------------------------------------- 家族分组
    by_family: dict[int, list[int]] = {}
    for lid in delivered:
        f = int(fam[lid]) if fam is not None and lid < len(fam) else 0
        by_family.setdefault(f, []).append(lid)

    # See `family_names`: the auditor's family_id is a different namespace from
    # the partition's, and matching them by integer mislabelled every family.
    fam_names = family_names(naming, fam, sizes)

    # A NAME OR CODE THAT IS NOT UNIQUE CANNOT BE A KEY. live38 shipped two
    # leaves called 生僻字词释义查询 and two carrying `chinese_pinyin_lookup`;
    # anything downstream that groups by either silently merges them.
    from collections import Counter

    dup_names = [n for n, c in Counter(
        str(by_id.get(i, {}).get("name_zh", "")).strip() for i in delivered).items()
        if c > 1 and n]
    dup_codes = [c for c, n in Counter(
        str(by_id.get(i, {}).get("code", "")).strip() for i in delivered).items()
        if n > 1 and c]
    if dup_names or dup_codes:
        L += ["> ⚠️ **有叶子共用同一个名称或代码, 因此二者都不能当作主键。**", ""]
        for nm in dup_names:
            ids = [i for i in delivered if str(by_id.get(i, {}).get("name_zh", "")).strip() == nm]
            L.append(f"> - 名称 「{nm}」 → 叶 {', '.join(str(i) for i in ids)}")
        for cd in dup_codes:
            ids = [i for i in delivered if str(by_id.get(i, {}).get("code", "")).strip() == cd]
            L.append(f"> - 代码 `{cd}` → 叶 {', '.join(str(i) for i in ids)}")
        L += ["", "> 下游请按 **`bu_leaf` 的整数 id** 聚合 —— 它唯一。", ""]

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
            hits = risk_by_leaf.get(lid, 0)
            if hits:
                L.append(f"- 🔍 **风险筛查命中 {hits:,} 行** ({pct(hits / max(1, sz))} 的该叶)"
                         + ("" if n.get("risk_flag") else " —— 命名者未标记此叶"))
            L.append("")
    return "\n".join(L)
