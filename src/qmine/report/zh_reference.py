"""参考手册: 类目清单 / 标注规范与裁定规则 / 家族与叶层级 / 双路对照。

**The run was producing these and delivering none of them.**

The six existing documents argue: they explain what was decided, on what evidence,
and what is still uncertain. None of them is the thing a person reaches for when
they want to *use* the taxonomy — the list of classes with their definitions, the
rules an annotator followed, the shape of the tree. That content existed in the
artifacts the whole time:

* `taxonomy_v2.json → taxonomy.labeling_guide` — the guide the annotators were
  actually given. Grepped against all six deliverables and the notebook: **zero
  occurrences.** It was never written anywhere a reader could see it.
* `taxonomy_v2.json → taxonomy.rules` — 139 adjudication rules, each with its
  trigger, its target class, its rationale and worked examples. The reports print
  the *count*. 33 of the 39 first-round rule ids appear in no deliverable at all.
* `taxonomy.nodes[].positive_examples` / `negative_examples` — 212 and 84 worked
  examples across the 21 classes, none of them delivered.
* `tree_naming.audit.families[].definition` — family-level definitions, delivered
  nowhere.
* `route_crosswalk.csv` — the only artifact that says how the two routes line up,
  named in no document.

The full class table did reach the top-down report, but inside a `<details>`
block in a 72KB file — collapsed, and invisible to in-page search on most
renderers.

These are REFERENCE documents, and that is a different genre from the reports
beside them. A report is read once, top to bottom, and argues. A reference is
opened at a row, cited, and diffed between runs. That is why they are Markdown
with one row per entity and stable ordering rather than a rendered page: they
have to survive being copied, archived, grepped and reviewed line-by-line in a
diff. Each ships a CSV twin for the same reason a controlled vocabulary always
ships one — a person reads the prose, a program reads the table, and neither has
to parse the other.

**The delivered partition decides.** p8 governance rewrites the tree after p7
names it, so `tree_naming.audit.families` describes a 20-family tree while 24
were delivered. Every count here is taken from `leaf_labels_final` /
`leaf_family_final` and `topdown_labels.parquet`; audit material is joined
through `_shape.family_names`, which maps by leaf membership rather than by an id
that means something different on each side.
"""
from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any

import numpy as np

from ._shape import family_names
from .i18n import pct, prose


def _head(state: Any, deps: Any, title: str, subtitle: str) -> list[str]:
    gen_dir = Path(deps.store.gen_dir).name
    return [f"# {title}", f"## {subtitle}", "",
            f"**运行**: `{state.get('run_id')}` / {gen_dir} · "
            f"**领域**: `{deps.cfg.domain.key}`", ""]


def _tax(deps: Any) -> dict[str, Any]:
    """The DELIVERED taxonomy — v2 if the referee revised it, else the draft."""
    for name in ("taxonomy_v2", "taxonomy"):
        if deps.has(name):
            raw = deps.load(name)
            t = raw.get("taxonomy", raw) if isinstance(raw, dict) else {}
            if t.get("nodes"):
                return t
    return {}


def _csv(rows: list[dict[str, Any]], cols: list[str]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


# ==========================================================================
# 1. 类目清单 — the top-down route's delivered labels
# ==========================================================================


def class_rows(state: Any, deps: Any) -> list[dict[str, Any]]:
    """One row per L1 class, with its DELIVERED size. Shared with the CSV twin."""
    t = _tax(deps)
    nodes = [n for n in (t.get("nodes") or []) if int(n.get("level", 1)) == 1]
    rules = t.get("rules") or []

    counts: dict[str, int] = {}
    total = 0
    if deps.has("topdown_labels"):
        try:
            df = deps.load("topdown_labels")
            col = "l1_pred" if "l1_pred" in df.columns else df.columns[1]
            counts = {str(k): int(v) for k, v in df[col].value_counts().items()}
            total = int(len(df))
        except Exception:  # noqa: BLE001 — a missing column must not lose the doc
            counts, total = {}, 0

    rules_to: dict[str, list[str]] = {}
    for r in rules:
        rules_to.setdefault(str(r.get("then") or ""), []).append(str(r.get("id") or ""))

    out = []
    for n in nodes:
        code = str(n.get("code") or "")
        n_rows = counts.get(code, 0)
        out.append({
            "code": code,
            "name": str(n.get("name") or ""),
            "definition": str(n.get("definition") or ""),
            "user_need": str(n.get("user_need") or ""),
            "n_delivered": n_rows,
            "share_delivered": (n_rows / total) if total else None,
            "expected_share": n.get("expected_share"),
            "cluster_invisible": bool(n.get("pragmatic_only")),
            "risk": bool(n.get("risk")),
            "n_rules": len(rules_to.get(code, [])),
            "rule_ids": " ".join(rules_to.get(code, [])),
            "positive_examples": n.get("positive_examples") or [],
            "negative_examples": n.get("negative_examples") or [],
        })
    out.sort(key=lambda r: -r["n_delivered"])
    return out


def build_classes(state: Any, deps: Any) -> str:
    rows = class_rows(state, deps)
    L = _head(state, deps, "类目清单 (L1 意图类目)",
              "自上而下路线交付了哪些类目, 每一类怎么定义, 以及实际落了多少行")
    if not rows:
        return "\n".join(L + ["_本次运行没有可用的类目体系。_"])

    total = sum(r["n_delivered"] for r in rows)
    invisible = [r for r in rows if r["cluster_invisible"]]
    L += [
        "这份文件是**自上而下路线的标签清单**, 与 `叶清单.md` (自下而上的叶标签) 对称。"
        "每一行的 `definition` 说明这一类**是什么**, `user_need` 说明"
        "**用户拿到什么才算被满足** —— 后者才是可检验的验收标准。", "",
        f"- L1 类目数: **{len(rows)}**",
        f"- 已打标行数: **{total:,}**",
        f"- 标注为「聚类不可见」的类目: **{len(invisible)}**", "",
        "> 「聚类不可见」指这一类的成员在向量空间里不聚成团, 因此**自下而上路线在原理上"
        "抓不到它** —— 这正是两条路线必须并行的理由, 不是一个修饰性标注。", "",
        "## 1. 交付规模 (按实际行数排序)", "",
        "| 类目 | 代码 | 交付行数 | 占比 | 架构师预估 | 规则数 | 聚类不可见 | 风险类 |",
        "|---|---|---:|---:|---:|---:|:---:|:---:|",
    ]
    for r in rows:
        exp = pct(r["expected_share"]) if r["expected_share"] is not None else "—"
        L.append(
            f"| {r['name']} | `{r['code']}` | {r['n_delivered']:,} | "
            f"{pct(r['share_delivered']) if r['share_delivered'] is not None else '—'} | "
            f"{exp} | {r['n_rules']} | {'✓' if r['cluster_invisible'] else ''} | "
            f"{'⚠' if r['risk'] else ''} |")

    L += ["", "> **预估与实际的差距本身是证据。** 架构师的 `expected_share` 是在看到语料"
          "分布之前写下的; 差得远的类目要么是类目定义偏了, 要么是分类器学偏了, 两种都值得"
          "单独复核。", "",
          "## 2. 每一类的完整定义与判例", "",
          "每一类下面的**正例**来自架构师的类目设计, **反例**同时写出它实际该归到哪一类 —— "
          "反例才是边界, 正例只是中心。", ""]

    for r in rows:
        L += [f"### `{r['code']}` — {r['name']}", ""]
        badges = []
        if r["cluster_invisible"]:
            badges.append("**聚类不可见**")
        if r["risk"]:
            badges.append("**风险类目**")
        if badges:
            L += ["> " + " · ".join(badges), ""]
        L += [f"- **定义**: {r['definition']}",
              f"- **满足条件 (user_need)**: {r['user_need']}",
              f"- **交付**: {r['n_delivered']:,} 行"
              + (f" ({pct(r['share_delivered'])})" if r["share_delivered"] is not None else ""),
              f"- **裁定规则**: {r['n_rules']} 条"
              + (f" (`{'`, `'.join(r['rule_ids'].split()[:8])}`"
                 + (" 等" if r["n_rules"] > 8 else "") + ")" if r["n_rules"] else ""), ""]
        if r["positive_examples"]:
            L += ["**正例**", ""]
            L += [f"- `{q}`" for q in r["positive_examples"][:12]]
            L += [""]
        if r["negative_examples"]:
            L += ["**反例 (括号内是它实际所属的类)**", ""]
            L += [f"- `{q}`" for q in r["negative_examples"][:12]]
            L += [""]
    return "\n".join(L)


def classes_csv(state: Any, deps: Any) -> str:
    rows = []
    for r in class_rows(state, deps):
        d = dict(r)
        d["positive_examples"] = " | ".join(r["positive_examples"])
        d["negative_examples"] = " | ".join(r["negative_examples"])
        rows.append(d)
    return _csv(rows, ["code", "name", "definition", "user_need", "n_delivered",
                       "share_delivered", "expected_share", "cluster_invisible",
                       "risk", "n_rules", "rule_ids",
                       "positive_examples", "negative_examples"])


# ==========================================================================
# 2. 标注规范与裁定规则
# ==========================================================================


def build_rules(state: Any, deps: Any) -> str:
    """The guide and every rule, grouped by the class each one routes to.

    Grouped by TARGET rather than listed by id because that is how the document
    is used: an annotator arrives holding a query and a candidate class, not a
    rule id. The provenance stays on every row — a rule the referee added after
    seeing real disagreements is a different kind of evidence from one the
    architect predicted in advance, and only the first was earned.
    """
    t = _tax(deps)
    guide = str(t.get("labeling_guide") or "").strip()
    rules = t.get("rules") or []
    names = {str(n.get("code")): str(n.get("name") or "")
             for n in (t.get("nodes") or [])}

    lang = getattr(getattr(deps, "cfg", None), "report_language", "zh") or "zh"
    L = _head(state, deps, "标注规范与裁定规则",
              "两名标注员和裁判实际遵循的那一份指南, 以及全部裁定规则")
    if not guide and not rules:
        return "\n".join(L + ["_本次运行没有记录标注规范或裁定规则。_"])

    by_round: dict[Any, int] = {}
    for r in rules:
        by_round[r.get("added_in_round")] = by_round.get(r.get("added_in_round"), 0) + 1
    rounds = ", ".join(f"第 {k} 轮 {v} 条" for k, v in sorted(
        by_round.items(), key=lambda kv: (kv[0] is None, kv[0])))

    L += [
        "这份文件是**复现本次标注所需的全部规范**。上一次交付里它只以「139 条规则」"
        "这个数字出现过 —— 数字不能用来标注。", "",
        f"- 裁定规则总数: **{len(rules)}** ({rounds})",
        f"- 覆盖类目: **{len({r.get('then') for r in rules if r.get('then')})}**", "",
    ]

    if guide:
        L += ["## 1. 标注指南 (原文)", "",
              "> 以下是架构师下发给标注员的原文, 未经改写。", "",
              "```text", guide, "```", ""]

    L += ["## 2. 裁定规则", "",
          "每一条规则的形式是 **当 (when) → 归入 (then)**, 附上理由与判例。"
          "`第 0 轮` 是架构师在看到分歧之前**预判**的; "
          "`第 1 轮及以后` 是裁判在真实分歧上**实测**之后补的 —— "
          "后者是被数据挣来的, 前者还没有。", ""]

    grouped: dict[str, list[dict[str, Any]]] = {}
    for r in rules:
        grouped.setdefault(str(r.get("then") or "(未指定类目)"), []).append(r)

    for code in sorted(grouped, key=lambda c: -len(grouped[c])):
        rs = grouped[code]
        title = names.get(code, "")
        L += [f"### 归入 `{code}`" + (f" — {title}" if title else "")
              + f" · {len(rs)} 条", ""]
        for r in rs:
            rid = str(r.get("id") or "")
            rnd = r.get("added_in_round")
            origin = "架构师预判" if rnd in (0, "0") else f"裁判补充 (第 {rnd} 轮)"
            # THE RULES THE REFEREE EARNED ARE THE ONES THAT SHIPPED IN ENGLISH.
            #
            # `rationale` and `added_because` are OUR strings, hardcoded English
            # in `graph/nodes/topdown.py`, and they carry 100 of these 139 rules
            # — the ones drafted against real disagreements rather than
            # predicted in advance. An annotation manual in Chinese whose most
            # evidence-backed half is in English is not usable by the annotator
            # it is written for. `prose()` covers the fixed sentence; the
            # templated one keeps its query, which must not be translated.
            when = str(r.get("when") or "")
            if when[:5].lower() == "when ":
                when = when[5:]              # the referee echoes the field name
            L += [f"**`{rid}`** · {origin}", "",
                  f"- **当**: {when}",
                  f"- **理由**: {prose(str(r.get('rationale') or ''), lang)}"]
            because = str(r.get("added_because") or "")
            if because.startswith("disagreement on "):
                because = f"标注员在 {because[len('disagreement on '):]} 上产生分歧"
            if because:
                L.append(f"- **因何而加**: {prose(because, lang)}")
            trig = r.get("trigger")
            if trig:
                L.append(f"- **触发词**: `{trig}`")
            ex = r.get("examples") or []
            if ex:
                L.append("- **判例**: " + ", ".join(f"`{e}`" for e in ex[:8]))
            L.append("")
    return "\n".join(L)


def rules_csv(state: Any, deps: Any) -> str:
    t = _tax(deps)
    rows = []
    for r in (t.get("rules") or []):
        rows.append({
            "id": r.get("id", ""), "then": r.get("then", ""),
            "when": r.get("when", ""), "rationale": r.get("rationale", ""),
            "trigger": r.get("trigger", ""),
            "added_in_round": r.get("added_in_round", ""),
            "added_because": r.get("added_because", ""),
            "examples": " | ".join(r.get("examples") or []),
        })
    return _csv(rows, ["id", "then", "when", "rationale", "trigger",
                       "added_in_round", "added_because", "examples"])


# ==========================================================================
# 3. 家族与叶层级 — the delivered two-level tree
# ==========================================================================


def build_tree(state: Any, deps: Any) -> str:
    """The delivered tree, family by family.

    `叶清单.md` is a flat per-leaf reference and does not say which leaves sit
    together; this one is the shape. It reads the DELIVERED arrays, not the
    naming-time ones — p8 governance rewrote the tree after p7 named it, so the
    audit describes 20 families where 24 were delivered, and joining the two by
    integer id names a family after a different family's leaves.
    """
    naming = deps.load("tree_naming") if deps.has("tree_naming") else {}
    labels = deps.leaf_labels_final() if deps.has("leaf_labels") else None
    fam = deps.leaf_family_final() if deps.has("leaf_family") else None

    L = _head(state, deps, "家族与叶层级 (两层树)",
              "自下而上路线交付的树: 哪些叶子归在同一个家族下, 以及每个家族有多大")
    if labels is None or fam is None:
        return "\n".join(L + ["_本次运行没有可用的层级划分。_"])

    sizes = np.bincount(labels)
    delivered_leaves = sorted({int(v) for v in np.unique(labels)})
    fam_of = {lid: int(fam[lid]) for lid in delivered_leaves if lid < len(fam)}
    delivered_fams = sorted(set(fam_of.values()))
    fnames = family_names(naming, fam, sizes)
    leaf = {int(n["leaf_id"]): n for n in (naming.get("namings") or [])}

    audit_fams = ((naming or {}).get("audit", {}) or {}).get("families", []) or []
    n_audit = len(audit_fams)
    # AN AUDIT DEFINITION IS ONLY THIS FAMILY'S IF IT DESCRIBES ONLY THIS FAMILY.
    #
    # The first version of this document took the first definition it found among
    # a family's leaves and printed it as 「审计给出的定义」. Measured on live42:
    # 14 of the 24 delivered families then carried a definition shared with one or
    # two others — family 8 (17 leaves, 14,171 rows) and family 10 (1 leaf) got
    # the identical sentence. That is this project's own delivered-partition trap:
    # the audit describes the 20-family PRE-GOVERNANCE tree, so its definitions
    # are not addressed to the families that shipped.
    #
    # The join is computed both ways. A definition is presented as this family's
    # only when its leaves come from ONE audit family AND that audit family backs
    # no other delivered family. Otherwise it is labelled for what it is — a
    # description of a neighbouring shape — or dropped.
    audit_of_leaf: dict[int, int] = {}
    for i, f in enumerate(audit_fams):
        for lid in (f.get("leaf_ids") or []):
            audit_of_leaf[int(lid)] = i
    srcs_of_fam: dict[int, set[int]] = {}
    for lid, ff in fam_of.items():
        a = audit_of_leaf.get(lid)
        if a is not None:
            srcs_of_fam.setdefault(ff, set()).add(a)
    fams_of_src: dict[int, set[int]] = {}
    for ff, ss in srcs_of_fam.items():
        for a in ss:
            fams_of_src.setdefault(a, set()).add(ff)

    total = int(len(labels))
    L += [
        "这份文件回答的是 `叶清单.md` 回答不了的那个问题: **哪些叶子是一伙的。**"
        "叶清单是逐叶的定义表, 这里是树的形状。", "",
        f"- 交付家族数: **{len(delivered_fams)}**",
        f"- 交付叶数: **{len(delivered_leaves)}**",
        f"- 覆盖行数: **{total:,}**", "",
    ]
    if n_audit and n_audit != len(delivered_fams):
        L += [f"> ⚠️ **树审计描述的是 {n_audit} 个家族, 交付的是 "
              f"{len(delivered_fams)} 个。** p8 治理在 p7 命名之后重写了这棵树, "
              "因此审计给出的家族定义**不能按编号对上交付的家族**。下面的家族名与定义"
              "是**按叶成员**关联回来的 (`_shape.family_names`), 一个交付家族可能"
              "覆盖审计里的好几个家族 —— 名称后面的「等 N 类」就是这种情况。", ""]

    L += ["## 1. 树的形状", "", "| 家族 | 名称 | 叶数 | 行数 | 占比 |",
          "|---:|---|---:|---:|---:|"]
    fam_rows = []
    for f in delivered_fams:
        leaves = [lid for lid, ff in fam_of.items() if ff == f]
        n = int(sum(int(sizes[lid]) for lid in leaves))
        fam_rows.append((f, leaves, n))
    fam_rows.sort(key=lambda r: -r[2])
    for f, leaves, n in fam_rows:
        L.append(f"| {f} | {fnames.get(f, '(未命名)')} | {len(leaves)} | "
                 f"{n:,} | {pct(n / total if total else 0)} |")

    # A TREE IS A SHAPE, AND A TABLE IS NOT A SHAPE.
    #
    # GitHub and GitLab both render Mermaid from a fenced block, so a diagram
    # costs nothing extra to ship and needs no server — the same property the
    # rest of this shelf is chosen for. Only the TOP level is drawn: Mermaid caps
    # diagram source at 50,000 characters, and 58 leaves under 24 families is a
    # hairball nobody reads. The leaves are below, where they can be searched.
    drawn = [(f, leaves, n) for f, leaves, n in fam_rows if n]
    if drawn:
        L += ["", "## 1b. 树的顶层 (按行数)", "", "```mermaid", "flowchart LR",
              f'  ROOT["全部查询<br/>{total:,} 行"]']
        for f, leaves, n in drawn[:24]:
            # Quote the label: Chinese family names carry (), 「」 and / , all of
            # which end a Mermaid node id early and produce a parse error that
            # renders as a red box where the diagram should be.
            # A node label is a glance, not a definition. `family_names` appends
            # a disambiguating suffix ("… 等 6 类 (42%)", "… · 主要叶「…」") that is
            # exactly right in a table and unreadable in a box — and truncating
            # it mid-suffix leaves an unbalanced "(" , which ends a Mermaid label
            # early and renders the whole diagram as a parse error. Take the
            # primary name and drop the suffix; the full one is in the table
            # directly above and in the section below.
            nm = str(fnames.get(f, "(未命名)"))
            for sep in (" 等 ", " · "):
                nm = nm.split(sep)[0]
            nm = nm.replace('"', "'").replace("(", "（").replace(")", "）")
            if len(nm) > 20:
                nm = nm[:19] + "…"
            # The id goes IN the label. Dropping the disambiguating suffix is
            # what makes the labels readable, and it is also what lets two
            # families render as identical boxes — `family_names` adds that
            # suffix precisely because one audit name can dominate several
            # delivered families. The id is the primary key everywhere else in
            # this document, so it is the honest way to tell them apart.
            L.append(f'  ROOT --> F{f}["家族 {f} · {nm}<br/>'
                     f'{len(leaves)} 叶 · {n:,} 行"]')
        L += ["```", ""]
        if len(drawn) > 24:
            L += [f"> 图中只画了最大的 24 个家族, 其余 {len(drawn) - 24} 个见上表。", ""]

    # THE ONE ARTIFACT THAT SAYS HOW THE TWO ROUTES LINE UP, AND IT WAS NAMED IN
    # NO DOCUMENT.
    #
    # `route_crosswalk.csv` is keyed by DELIVERED family, which is why it belongs
    # here rather than in the panel: the reader is already looking at the family,
    # and a table in another file that has to be joined by hand is a table nobody
    # joins. `td_effective_classes` is the perplexity of the top-down
    # distribution inside this family — 1.0 means the family sits inside a single
    # intent class, and a large number means the clusterer grouped rows that the
    # intent taxonomy separates. Neither is a defect on its own; the two routes
    # answer different questions, and that is the thesis this delivery exists to
    # test.
    cross: dict[int, dict[str, Any]] = {}
    if deps.has("route_crosswalk"):
        try:
            cw = deps.load("route_crosswalk")
            for rec in cw.to_dict("records"):
                cross[int(rec["bu_family_final"])] = rec
        except Exception:  # noqa: BLE001 — a shape change must not lose the tree
            cross = {}
    if cross:
        L += ["", "## 1c. 每个家族落在自上而下的哪些类目上", "",
              "同一批查询, 两条路线各自给了标签。下表是**逐家族**的对照: "
              "`主导类目占比` 高说明这个家族基本就是那一个意图; "
              "`等效类目数` 是该家族内自上而下标签分布的困惑度 —— "
              "接近 1 表示落在单一意图内, 数值大表示聚类把意图体系区分开的行归到了一起。", "",
              "| 家族 | 名称 | 行数 | 主导类目 | 主导占比 | 触及类目 | 等效类目数 | 判定 |",
              "|---:|---|---:|---|---:|---:|---:|---|"]
        for f, leaves, n in fam_rows:
            rec = cross.get(f)
            if not rec:
                continue
            share = rec.get("td_dominant_share")
            L.append(
                f"| {f} | {fnames.get(f, '(未命名)')} | {n:,} | "
                f"`{rec.get('td_dominant', '')}` | "
                f"{pct(float(share)) if share is not None else '—'} | "
                f"{rec.get('td_classes_touched', '')} | "
                f"{rec.get('td_effective_classes', '')} | {rec.get('verdict', '')} |")
        L.append("")

    L += ["", "## 2. 每个家族下面是什么", ""]
    for f, leaves, n in fam_rows:
        L += [f"### 家族 {f} — {fnames.get(f, '(未命名)')}", "",
              f"- **规模**: {len(leaves)} 个叶 · {n:,} 行 "
              f"({pct(n / total if total else 0)})", ""]
        srcs = sorted(srcs_of_fam.get(f, set()))
        if len(srcs) == 1 and len(fams_of_src.get(srcs[0], set())) == 1:
            d = str(audit_fams[srcs[0]].get("definition") or "")
            if d:
                L += [f"- **审计给出的定义**: {d}", ""]
        elif srcs:
            names = "、".join(str(audit_fams[a].get("name_zh") or f"#{a}") for a in srcs)
            others = sorted({x for a in srcs for x in fams_of_src.get(a, set())} - {f})
            L += [f"- **没有专属于本家族的定义。** 它的叶子来自树审计里的 {names} "
                  f"({len(srcs)} 个审计家族)"
                  + (f", 而这些审计家族同时也覆盖了交付家族 "
                     f"{'、'.join(str(x) for x in others)}" if others else "")
                  + " —— 治理重写这棵树之后, 审计的定义已经不是对本家族说的, "
                    "因此这里不把它当成本家族的定义来展示。", ""]
        L += ["| 叶 | 名称 | 行数 | 占本家族 | user_need |", "|---:|---|---:|---:|---|"]
        for lid in sorted(leaves, key=lambda x: -int(sizes[x])):
            nm = leaf.get(lid, {})
            sz = int(sizes[lid])
            L.append(f"| {lid} | {nm.get('name_zh', '(未命名)')} | {sz:,} | "
                     f"{pct(sz / n if n else 0)} | {nm.get('user_need', '')} |")
        L.append("")
    return "\n".join(L)


def tree_csv(state: Any, deps: Any) -> str:
    naming = deps.load("tree_naming") if deps.has("tree_naming") else {}
    labels = deps.leaf_labels_final() if deps.has("leaf_labels") else None
    fam = deps.leaf_family_final() if deps.has("leaf_family") else None
    if labels is None or fam is None:
        return _csv([], ["family_id", "family_name", "leaf_id", "leaf_name",
                         "code", "n_rows", "user_need"])
    sizes = np.bincount(labels)
    fnames = family_names(naming, fam, sizes)
    leaf = {int(n["leaf_id"]): n for n in (naming.get("namings") or [])}
    rows = []
    for lid in sorted({int(v) for v in np.unique(labels)}):
        f = int(fam[lid]) if lid < len(fam) else -1
        nm = leaf.get(lid, {})
        rows.append({"family_id": f, "family_name": fnames.get(f, ""),
                     "leaf_id": lid, "leaf_name": nm.get("name_zh", ""),
                     "code": nm.get("code", ""), "n_rows": int(sizes[lid]),
                     "user_need": nm.get("user_need", "")})
    return _csv(rows, ["family_id", "family_name", "leaf_id", "leaf_name",
                       "code", "n_rows", "user_need"])


# ==========================================================================
# 4. 00_索引 — the front door
# ==========================================================================

#: What each delivered document is FOR, and who should open it. Keyed by the
#: `refs` key the builder uses, so a document that stops being produced stops
#: being listed rather than becoming a dead link.
_WHAT_FOR: dict[str, tuple[str, str, str]] = {
    "report_final":    ("00_最终报告.md", "先读这一份",
                        "由 agent 撰写的贯通全文: 两条路线为什么都要跑, 各自得到什么, "
                        "在同一把尺子下怎么对拍。每个数字都逐值核对过。"),
    "report_panel":    ("统一度量面板.md", "要比较两条路线时读",
                        "同一套度量下的横向对照, 以及本次运行的全部质量门与未关闭问题。"),
    "class_catalogue": ("类目清单.md", "要用自上而下的标签时读",
                        "21 个 L1 意图类目: 定义、user_need、正反例、实际交付行数、"
                        "以及有多少条裁定规则指向它。"),
    "leaf_catalogue":  ("叶清单.md", "要用自下而上的标签时读",
                        "每一个已交付的叶子: 名称、user_need、规模、风险命中。"),
    "tree":            ("家族与叶层级.md", "要看树的形状时读",
                        "交付的两层树: 哪些叶子归在同一家族下, 每个家族多大。"),
    "rules":           ("标注规范与裁定规则.md", "要复现标注或训练新标注员时读",
                        "标注指南原文, 以及全部裁定规则 (当→归入→理由→判例), "
                        "按目标类目分组。"),
    "report_topdown":  ("自上而下类目体系最终报告.md", "要审查自上而下这条路线时读",
                        "类目体系怎么长出来、金标准怎么建、κ 怎么读、分类器与对抗验证。"),
    "report_bottomup": ("自下而上聚类最终报告.md", "要审查自下而上这条路线时读",
                        "表征选型、K 的定位、层级构建、治理执行, 以及每一个被否决的方案。"),
    "delivery_audit":  ("交付前审核报告.md", "要知道交付前改过什么时读",
                        "交付前审核对正文做的逐条修订, 以及拒绝执行的修订。"),
    "notebook":        ("自下而上聚类全流程.ipynb", "要复算时打开",
                        "可执行的全流程, 图都是在这里跑出来的。"),
}


def build_index(state: Any, deps: Any, refs: dict[str, Any] | None = None) -> str:
    """The reading order. There was no front door at all.

    Ten documents landed in one directory with no index, no stated reading
    order, and no note saying what `*.pre_audit.md` is — so the first decision a
    reader has to make, which file to open, was the one thing the delivery did
    not help with. The material was already written: every `put_markdown` call
    passes a `summary`, and every one of them was thrown away.

    Ordered by what a reader needs first, not by how the pipeline produced it.
    """
    gen_dir = Path(deps.store.gen_dir)
    L = _head(state, deps, "交付物索引", "这次运行交付了什么, 以及先读哪一份")
    L += [
        "> 这一份是**目录**。下面按「你想做什么」排序, 不是按流水线的产出顺序。", "",
        "## 按用途", "", "| 文件 | 什么时候读 | 内容 |", "|---|---|---|",
    ]
    present = []
    for key, (fname, when, what) in _WHAT_FOR.items():
        if refs is not None and key not in refs and not (gen_dir / fname).exists():
            continue
        if refs is None and not (gen_dir / fname).exists():
            continue
        present.append(fname)
        L.append(f"| [`{fname}`]({fname}) | {when} | {what} |")
    if not present:
        L.append("| _(本次运行没有产出可索引的交付物)_ | | |")

    L += [
        "", "## 机器可读的副本", "",
        "下面几份是上面同一批内容的 CSV, 供程序直接读取 —— "
        "**人读散文, 程序读表格, 两边不必互相解析**。数字与人读版本同源。", "",
        "| 文件 | 一行是什么 |", "|---|---|",
        "| `类目.csv` | 一个 L1 意图类目 |",
        "| `规则.csv` | 一条裁定规则 |",
        "| `家族与叶.csv` | 一个已交付的叶子 (含它所属的家族) |",
        "| `route_crosswalk.csv` | 一个已交付的家族, 以及它在自上而下类目上的分布 |",
        "| `labels_full.csv` | 一行查询, 两条路线各自给它的标签 |",
        "",
        "## 两类文件的区别", "",
        "- **报告** (`…最终报告.md`, `统一度量面板.md`) 是**论证**: 为什么这么选, "
        "证据是什么, 哪里还不确定。从头读到尾。",
        "- **清单/手册** (`类目清单.md`, `叶清单.md`, `家族与叶层级.md`, "
        "`标注规范与裁定规则.md`) 是**参考**: 按行查, 按行引用, 跨运行做 diff。"
        "不需要从头读。", "",
        "> `*.pre_audit.md` 是**交付前审核之前**的同名文档快照, 保留下来是为了让"
        "「审核改了什么」可被核对 —— 正式交付请读没有这个后缀的那一份。", "",
    ]
    return "\n".join(L)
