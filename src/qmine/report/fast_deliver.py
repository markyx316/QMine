"""The three documents a `mode="fast"` run ships, and the banner none of them can omit.

A full run delivers thirteen documents that ARGUE: each one takes a decision the
pipeline made, shows the measurement behind it, and says why the alternative lost.
A fast run has no argument to make — it removed the layer that produces one — so
it ships three documents that REFER: what the classes are, what the tree is, and
what every row was labelled. That is the shape of the reference deliverables in
`BottomUp_Approach_Final_Report.md`'s companion set, and it is what someone who
wants to USE a taxonomy opens.

**Fewer documents, not less evidence.** Every artifact a full run writes is still
written — the store is untouched by mode — and each of these three ends in
`§ 原始档案位置`, which maps every table back to the artifact it was read from.
The reduction is in what gets argued, never in what can be audited.

**The banner is generated, not written.** `_banner()` renders `cfg.fast_skipped`,
which `QMineConfig._fast_mode_drops_the_second_opinion` populates as it turns each
component off. There is exactly one list, so a skip cannot exist without appearing
in every deliverable, and a banner cannot claim a skip that did not happen. The
failure this guards is the one that makes fast mode dangerous at all: a document
that reads exactly like a full run's, describing numbers nobody checked.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

#: What each `fast_skipped` entry cost the reader, in the reader's language.
#: Keyed by the identifier the config validator appends. An entry with no row
#: here still appears in the banner — `_banner` falls back to the raw key rather
#: than dropping it, because a silently omitted skip is the failure mode.
_SKIP_MEANING: dict[str, tuple[str, str]] = {
    "dual_annotation": (
        "双标注",
        "金标准由**一名**标注员完成, 不是两名独立标注员"),
    "kappa_agreement": (
        "标注一致性 (kappa)",
        "**未测量**。没有第二份独立判读, kappa 无从计算 —— 不是 0, 也不是 1"),
    "pilot_ceiling": (
        "试点与标注员自洽上限",
        "未运行。无法判断混淆来自类目边界还是标注指南"),
    "kappa_repair": (
        "指南修复轮次",
        "未运行。一致性低时本应重写指南并重标, 本次没有触发条件"),
    "taxonomy_redraw": (
        "类目边界重画",
        "未运行。试点没有运行, 就没有据以重画的证据"),
    "phase_observers": (
        "各阶段观察员",
        "未运行。每个阶段的产物没有第二个 agent 独立复核并提出可执行的质疑"),
    "adversarial_validation": (
        "对抗验证",
        "未运行。没有独立估计标签在被主动挑错时的稳健性"),
    "narrative_report": (
        "agent 撰写的总报告",
        "未生成。本次交付的三份文档全部由程序从产物直接生成"),
    "delivery_audit": (
        "交付前审核",
        "未运行。没有 agent 在交付前通读全部文档并修正其中的错误"),
    "result_interpretation": (
        "结果解读",
        "未生成。文档只陈述测得的数字, 不含针对本语料的解释性论断"),
}


def _skipped(deps: Any) -> list[str]:
    return list(getattr(deps.cfg, "fast_skipped", []) or [])


def is_fast(deps: Any) -> bool:
    return getattr(deps.cfg, "mode", "full") == "fast"


def _banner(state: Any, deps: Any) -> list[str]:
    """The block every fast deliverable opens with. Rendered from `fast_skipped`.

    Never hand-written and never conditional on the document: the reader of any
    one of the three learns the same thing about what was not done, because all
    three call this.
    """
    skips = _skipped(deps)
    out = [
        "> ## ⚠ 本次运行为 **fast 模式** —— 分析完整, 复核层已移除",
        ">",
        "> **分析本身与 full 模式相同**: 相同的语料全量、相同的 α 与 K 网格、"
        "相同的金标准规模、相同的研究员数量、相同的 12 个阶段。fast 模式"
        "**不缩小任何一项分析**。",
        ">",
        "> **被移除的是「第二意见」层** —— 下列每一项都只负责复核别人的产物, "
        "不决定任何参数、K 值、α 值或标签。因此移除它们不会改变结果, "
        "但会让结果**缺少被独立检验过的证据**:",
        ">",
    ]
    for key in skips:
        name, why = _SKIP_MEANING.get(key, (f"`{key}`", "本次运行未执行"))
        out.append(f"> - **{name}** —— {why}")
    if not skips:
        out.append("> - (配置未记录任何跳过项 —— 这本身异常, 请核对 `config.resolved.yaml`)")
    out += [
        ">",
        "> **怎么读这份文档**: 里面的每一个数字都是真实测出来的, 与 full 模式"
        "会得到的数字一致; 缺的是「这个数字被独立复核过」这一层证据。需要该证据时, "
        "用 `mode=full` 重跑同一份数据。",
        ">",
        "> **证据没有减少**: 交付文档从 13 份减为 3 份, 但中间产物一份未少 —— "
        "见文末 §原始档案位置。",
        "",
    ]
    return out


def _head(state: Any, deps: Any, title: str, subtitle: str) -> list[str]:
    gen_dir = Path(deps.store.gen_dir).name
    return ([f"# {title}", f"## {subtitle}", "",
             f"**运行**: `{state.get('run_id')}` / {gen_dir} · "
             f"**领域**: `{deps.cfg.domain.key}` · **模式**: `fast`", ""]
            + _banner(state, deps))


def demote(md: str, by: int = 1, *, drop_title: bool = True) -> str:
    """Push every heading down `by` levels so a document can be nested in another.

    Two things this must not do, both of which a naive `re.sub(r"^#", ...)` does:

    * **Demote inside a fenced code block.** `# 注释` in a ```python fence is a
      comment, not a heading, and shifting it corrupts the code a reader is meant
      to copy. Fences are tracked.
    * **Push a heading past `######`.** Markdown has six levels; a seventh renders
      as literal `#######` text. Levels clamp.

    `drop_title` removes the leading `# ...`/`## ...` pair that every source
    builder emits via its own `_head`, because the composite supplies its own.
    """
    lines = md.splitlines()
    out: list[str] = []
    fence: str | None = None
    dropped = 0
    for ln in lines:
        stripped = ln.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            marker = stripped[:3]
            fence = None if fence == marker else (fence or marker)
            out.append(ln)
            continue
        if fence is None and ln.startswith("#"):
            level = len(ln) - len(ln.lstrip("#"))
            if drop_title and dropped < 2 and level <= 2 and not out:
                dropped += 1
                continue
            out.append("#" * min(6, level + by) + ln[level:])
            continue
        out.append(ln)
    # A `_head` block is `# title`, `## subtitle`, "", "**运行**: ...", "" — the
    # run line repeats what the composite already states once at the top.
    while drop_title and out and (not out[0].strip() or out[0].startswith("**运行**")):
        out.pop(0)
    return "\n".join(out)


#: Full-mode deliverable -> where its content lives in the fast set. The source
#: builders cross-reference each other by filename ("与 `叶清单.md` 对称"), and
#: fast mode ships neither the file nor a stub for it — so a delivered document
#: told the reader to open something that is not in the directory. Observed on
#: `fin01`: four such references across the two markdown files.
_RELINK = {
    "叶清单.md": "本文档 §二「每一个叶的完整定义」",
    "类目清单.md": "《自上而下 · 意图体系完整定义》§一「类目体系」",
    "标注规范与裁定规则.md": "《自上而下 · 意图体系完整定义》§二「标注规范与裁定规则」",
    "家族与叶层级.md": "本文档 §一「家族与叶层级」",
    "统一度量面板.md": "`metrics_panel.json` (fast 模式不单独出面板文档)",
    "自下而上聚类最终报告.md": "《自下而上 · 聚类树完整定义》",
    "自上而下类目体系最终报告.md": "《自上而下 · 意图体系完整定义》",
    "00_最终报告.md": "本文档 (fast 模式不生成 agent 撰写的总报告)",
}


def relink(md: str) -> str:
    """Point cross-references at what this run actually shipped.

    Only the exact filenames above are touched, backticks and all, so a sentence
    that merely mentions a word is never rewritten. A reference this map does not
    know is left alone rather than mangled — a wrong pointer is worse than a
    stale one.
    """
    for name, where in _RELINK.items():
        md = md.replace(f"`{name}`", where).replace(name, where)
    return md


def _section(state: Any, deps: Any, fn: Any, *args: Any) -> str:
    """Run one source builder, or say in the document that it could not run.

    A builder that raises must not take the whole deliverable with it: on a fast
    run these three files are the ONLY deliverables, so an exception here is the
    difference between a partial document and no document at all. The failure is
    written into the page rather than logged and forgotten — an empty section a
    reader cannot explain is worse than a named one.
    """
    try:
        return relink(demote(fn(state, deps, *args)))
    except Exception as exc:  # noqa: BLE001
        deps.emit(f"  ⚠ fast deliverable section {getattr(fn, '__name__', fn)} "
                  f"failed: {type(exc).__name__}: {exc}")
        return (f"> ⚠ 本节生成失败: `{type(exc).__name__}: {exc}`。"
                f"对应的原始产物仍在运行目录中, 见 §原始档案位置。")


# ==========================================================================
# 原始档案位置 — the section that keeps "fewer documents" from meaning
# "less evidence". Every table above is traced to the artifact it was read from.
# ==========================================================================

def _artifact_path(deps: Any, name: str) -> Path | None:
    """Where an artifact actually is, across generations, or `None`.

    `deps.store.get(name).path` is the only correct answer on a re-render: the
    store keeps refs from every generation at or below the current one, so an
    artifact written in gen01 is still resolvable while writing gen02. Looking in
    `store.gen_dir` finds nothing and reports the run as having produced nothing.
    """
    try:
        if deps.has(name):
            p = Path(deps.store.get(name).path)
            return p if p.exists() else None
    except Exception:  # noqa: BLE001
        pass
    # Fall back to the current generation for files written outside the index.
    for suffix in (".json", ".csv", ".parquet", ".npy", ".md", ".joblib"):
        cand = Path(deps.store.gen_dir) / f"{name}{suffix}"
        if cand.exists():
            return cand
    return None


def _archive(state: Any, deps: Any, used: list[tuple[str, str]]) -> list[str]:
    """List the artifacts this document was generated from, with their real paths.

    `used` is `(artifact_name, what_it_backs)`. Names are checked against the
    store: an artifact this run did not write is listed as MISSING rather than
    printed as a working path, because a reference the reader cannot follow is
    the exact failure this section exists to prevent.
    """
    gen = Path(deps.store.gen_dir)
    out = ["", "---", "", "## 原始档案位置", "",
           "本文档的每一张表都由程序从下列产物直接生成, 没有任何一处是转述或估计。"
           "路径相对于运行目录; 需要复核任意一个数字时, 打开对应文件即可。", "",
           f"**运行目录**: `{gen}`", "",
           "| 产物 | 支撑本文档的哪一部分 | 文件 |", "|---|---|---|"]
    for name, backs in used:
        # RESOLVE THROUGH THE STORE, NOT THE TARGET DIRECTORY.
        #
        # `qmine render` writes into a NEW generation, and the artifacts it
        # re-renders from live in the old one — the store resolves them
        # (`ref.generation <= self.generation`), a raw `gen / name` does not. A
        # rendered fast deliverable therefore reported "未生成" for 13 of its own
        # sources, telling the reader every piece of evidence was missing when
        # all of it was one directory up. Caught by re-rendering `fin01`.
        hit = _artifact_path(deps, name)
        out.append(f"| `{name}` | {backs} | "
                   + (f"`{hit.name}`" if hit else
                      "**未生成 —— 本次运行没有这一步的产物**") + " |")
    out += [
        "",
        "> **fast 模式没有减少任何中间产物。** 上表之外, 运行目录里还有嵌入矩阵、"
        "聚类标签、网格扫描的每一格分数、门禁台账 (`run_summary.json` 的 `gates`) "
        "与逐次模型调用的缓存 (`llm_cache/`)。要重算本文档的任何一个数字, "
        "材料都在。", ""]
    return out


def build_topdown(state: Any, deps: Any) -> str:
    """Deliverable 1 — the top-down intent system, as a reference not an argument.

    Composed from the builders a full run already uses, rather than re-derived:
    `zh_reference.build_classes` (definitions, examples, delivered size) and
    `zh_reference.build_rules` (the guide verbatim plus every adjudication rule).
    Re-deriving them would create a second implementation of the same tables that
    could disagree with the full-mode ones, and a fast run's numbers must be a
    full run's numbers.
    """
    from . import zh_reference as _ref
    from .zh_topdown import build as _td

    parts = _head(state, deps, "自上而下 · 意图体系完整定义",
                  "类目定义、判例、裁定规则、金标准与分类器 —— 一份可直接使用的参考")
    parts += ["## 一、类目体系", ""]
    parts.append(_section(state, deps, _ref.build_classes))
    parts += ["", "## 二、标注规范与裁定规则", ""]
    parts.append(_section(state, deps, _ref.build_rules))
    parts += ["", "## 三、金标准、分类器与本次运行的完整记录", "",
              "以下内容来自 full 模式同名报告的生成器, 未作删改 —— 其中涉及"
              "对抗验证、一致性与观察员的小节, 在本次运行中没有数据, 会显示为未测量。", ""]
    parts.append(_section(state, deps, _td))
    parts += _archive(state, deps, [
        ("taxonomy_v2", "类目定义、裁定规则、标注指南原文"),
        ("taxonomy", "架构师最初起草的类目体系 (未折入裁定规则)"),
        ("gold", "金标准逐行标注结果"),
        ("gold_agreement", "标注一致性记录 (fast 模式下为「未测量」)"),
        ("topdown_metrics", "分类器交叉验证准确率、macro-F1、校准 (ECE)"),
        ("topdown_labels", "全量逐行 L1 预测、置信度与判定来源"),
        ("topdown_l2_labels", "全量逐行 L2 子意图"),
        ("subintents", "子意图切分与几何可学习性审计"),
        ("adversarial_validation", "对抗验证 (fast 模式下未运行, 文件记录了原因)"),
        ("risk_screen", "风险内容筛查"),
        ("centroid_classifier", "训练好的分类器本体, 可直接加载推理"),
    ])
    return "\n".join(parts)


def build_bottomup(state: Any, deps: Any) -> str:
    """Deliverable 2 — the delivered cluster tree, family by family and leaf by leaf.

    `build_tree` and `zh_catalogue` both read the DELIVERED partition
    (`leaf_labels_final` / `leaf_family_final`), which is the reason they are
    reused here rather than rebuilt: p8 governance rewrites the tree, and any
    fresh implementation reading `hierarchy_meta` or p7's namings would describe
    a tree that no longer exists. See `.claude/rules/report-generators.md`.
    """
    from . import zh_reference as _ref
    from .zh_bottomup import build as _bu
    from .zh_catalogue import build as _cat

    parts = _head(state, deps, "自下而上 · 聚类树完整定义",
                  "已交付的家族与叶: 命名、定义、规模、代表性查询与判别边界")
    parts += ["## 一、家族与叶层级 (已交付)", ""]
    parts.append(_section(state, deps, _ref.build_tree))
    parts += ["", "## 二、每一个叶的完整定义", ""]
    parts.append(_section(state, deps, _cat))
    parts += ["", "## 三、表征、K 值与树是怎么定下来的", "",
              "以下内容来自 full 模式同名报告的生成器, 未作删改 —— 其中依赖"
              "观察员的小节在本次运行中没有数据。", ""]
    parts.append(_section(state, deps, _bu, {}))
    parts += _archive(state, deps, [
        ("tree_naming", "每个叶的命名、user_need、盲评一致性与风险标注"),
        ("naming_cards", "命名 agent 看到的原始卡片 (中心/随机/边缘样本 + n-gram)"),
        ("hierarchy_meta", "p6 产出的树 —— 注意这是治理前的形状"),
        ("governance", "治理台账: 每一次合并/隔离/保留及其执行结果"),
        ("leaf_labels_final", "**已交付**的逐行叶标签"),
        ("leaf_family_final", "**已交付**的叶 → 家族映射"),
        ("representation", "α 扫描、表征对照与最终选用的向量空间"),
        ("battery", "算法与 K 值的全网格扫描分数"),
        ("granularity", "K 值定位: 意图对齐、稳定性否决与最终裁定"),
        ("metrics_panel", "统一度量面板 —— 两条路线在同一把尺子下的对照"),
        ("template_groups", "模板/措辞群及其覆盖率"),
    ])
    return "\n".join(parts)


# ==========================================================================
# Deliverable 3 — the workbook
# ==========================================================================

def _rows_or_note(fn: Any, note: str) -> tuple[list[dict[str, Any]], str]:
    """Build a sheet's rows, converting any failure into a row the reader can see.

    A sheet that silently comes back empty is indistinguishable from a category
    that genuinely has no members — "0 risky queries" and "the risk screen
    crashed" look identical in a spreadsheet. So a failure becomes a single row
    naming the exception.
    """
    try:
        rows = fn() or []
        return list(rows), ""
    except Exception as exc:  # noqa: BLE001
        return [{"⚠": f"{note} 生成失败: {type(exc).__name__}: {exc}"}], f"{type(exc).__name__}"


def build_workbook(state: Any, deps: Any) -> Path:
    """Deliverable 3 — every row with both routes' labels, plus the definition sheets.

    The first sheet is `说明`, not data. A spreadsheet gets opened at whatever tab
    the reader last used and forwarded without its documents, so the workbook
    carries the same skipped-components disclosure the two markdown files do —
    stripped of markdown quoting, otherwise identical, and from the same
    `fast_skipped` list. A workbook that could travel alone without saying what
    was not checked is the whole risk of shipping a smaller deliverable set.
    """
    import json

    import pandas as pd

    gen = Path(deps.store.gen_dir)
    path = gen / f"{deps.cfg.domain.key}_query_挖掘结果.xlsx"

    def _load(name: str) -> Any:
        # Through the store, for the same reason `_archive` does: on a re-render
        # these live in the previous generation. Reading `gen / name.json`
        # returned `{}` for every sheet, and the rendered workbook shipped with
        # 8 of its 10 sheets empty.
        try:
            if deps.has(name):
                return deps.load(name)
        except Exception:  # noqa: BLE001
            pass
        p = _artifact_path(deps, name)
        if p is None or p.suffix != ".json":
            return {}
        return json.loads(p.read_text(encoding="utf-8"))

    # -- 说明 -------------------------------------------------------------
    # `**bold**` and `` `code` `` are markdown; in a cell they are literal
    # asterisks and backticks the reader has to mentally delete. Same words, same
    # order, same source list — only the markup goes.
    def _plain(ln: str) -> str:
        return (ln.lstrip("> ").rstrip()
                .replace("**", "").replace("`", "").replace("## ", "").lstrip("- "))

    banner = [_plain(ln) for ln in _banner(state, deps)]
    notes = [{"说明": f"运行 {state.get('run_id')} / {gen.name} · 领域 {deps.cfg.domain.key} · 模式 fast"},
             {"说明": ""}]
    notes += [{"说明": ln} for ln in banner if ln.strip()]
    notes += [{"说明": ""},
              {"说明": "本工作簿的每一个 sheet 都由程序从运行目录中的产物直接生成。"},
              {"说明": "全量标注 sheet 对应 labels_full.csv, 两条路线的标签并列在同一行上。"},
              {"说明": "需要复核任何一个数字, 见两份 .md 文档末尾的「原始档案位置」。"}]

    sheets: dict[str, list[dict[str, Any]]] = {"说明": notes}

    # -- 全量标注 ---------------------------------------------------------
    labels_csv = _artifact_path(deps, "labels_full")
    full = pd.read_csv(labels_csv) if labels_csv is not None else pd.DataFrame()

    # -- distributions, computed from the DELIVERED labels ----------------
    def _dist(col: str, label: str) -> list[dict[str, Any]]:
        if full.empty or col not in full.columns:
            return [{"⚠": f"{label}: labels_full.csv 缺少 `{col}` 列"}]
        vc = full[col].value_counts(dropna=False)
        n = int(vc.sum()) or 1
        return [{label: ("(空)" if pd.isna(k) else k), "条数": int(v),
                 "占比": round(float(v) / n, 4)} for k, v in vc.items()]

    sheets["意图分布"], _ = _rows_or_note(lambda: _dist("td_l1", "L1 意图"), "意图分布")
    sheets["聚类家族分布"], _ = _rows_or_note(lambda: _dist("bu_family_final", "家族"), "聚类家族分布")

    # -- 意图体系定义 -----------------------------------------------------
    def _classes() -> list[dict[str, Any]]:
        from . import zh_reference as _ref

        return _ref.class_rows(state, deps)

    sheets["意图体系定义"], _ = _rows_or_note(_classes, "意图体系定义")

    # -- 裁决规则 ---------------------------------------------------------
    def _rules() -> list[dict[str, Any]]:
        tax = (_load("taxonomy_v2") or {}).get("taxonomy") or _load("taxonomy")
        out = []
        for r in tax.get("rules", []) or []:
            out.append({"规则 id": r.get("id", ""),
                        "适用类目": ", ".join(r.get("classes", []) or []),
                        "触发条件 (if)": r.get("if", "") or r.get("when", ""),
                        "判定 (then)": r.get("then", ""),
                        "理由": r.get("because", "") or r.get("added_because", ""),
                        "加入轮次": r.get("added_in_round", 1)})
        return out

    sheets["裁决规则"], _ = _rows_or_note(_rules, "裁决规则")

    # -- 聚类叶定义 -------------------------------------------------------
    def _leaves() -> list[dict[str, Any]]:
        namings = (_load("tree_naming") or {}).get("namings") or []
        out = []
        for nm in namings:
            out.append({"叶 id": nm.get("cluster_id", nm.get("leaf", "")),
                        "名称": nm.get("name", ""),
                        "user_need": nm.get("user_need", ""),
                        "规模": nm.get("size", ""),
                        "盲评一致性": nm.get("coherence", ""),
                        "风险标注": nm.get("risk", "")})
        return out

    sheets["聚类叶定义"], _ = _rows_or_note(_leaves, "聚类叶定义")

    # -- 模板群定义 -------------------------------------------------------
    def _templates() -> list[dict[str, Any]]:
        tg = _load("template_groups") or {}
        return [{"模板群": g.get("name", g.get("pattern", "")),
                 "覆盖条数": g.get("n", g.get("size", "")),
                 "示例": " / ".join((g.get("examples") or [])[:3]),
                 "是否可信": g.get("trusted", "")}
                for g in (tg.get("groups") or [])]

    sheets["模板群定义"], _ = _rows_or_note(_templates, "模板群定义")

    # -- 风险 -------------------------------------------------------------
    def _risk_dist() -> list[dict[str, Any]]:
        rs = _load("risk_screen") or {}
        cats = rs.get("categories") or {}
        items = cats.items() if isinstance(cats, dict) else [(c.get("name"), c) for c in cats]
        out = []
        for name, c in items:
            c = c if isinstance(c, dict) else {"n": c}
            out.append({"风险类别": name, "命中条数": c.get("n", c.get("count", "")),
                        "占比": c.get("share", ""), "正则/判据": c.get("pattern", "")})
        return out

    sheets["风控图层分布"], _ = _rows_or_note(_risk_dist, "风控图层分布")

    def _risk_rows() -> list[dict[str, Any]]:
        rs = _load("risk_screen") or {}
        idx = rs.get("flag_mask_indices") or []
        if full.empty or not idx:
            return []
        keep = [i for i in idx if 0 <= int(i) < len(full)]
        cols = [c for c in ("query", "td_l1", "bu_leaf_name", "bu_family_final")
                if c in full.columns]
        return full.iloc[keep][cols].to_dict("records")

    sheets["风险内容清单"], _ = _rows_or_note(_risk_rows, "风险内容清单")

    # -- write ------------------------------------------------------------
    with pd.ExcelWriter(path, engine="openpyxl") as xl:
        # 说明 FIRST so it is the tab a reader lands on.
        pd.DataFrame(sheets.pop("说明")).to_excel(xl, sheet_name="说明", index=False)
        if not full.empty:
            full.to_excel(xl, sheet_name="全量标注", index=False)
        else:
            pd.DataFrame([{"⚠": "labels_full.csv 未生成 —— 本次运行没有全量标注结果"}]
                         ).to_excel(xl, sheet_name="全量标注", index=False)
        for name, rows in sheets.items():
            df = pd.DataFrame(rows if rows else [{"(本次运行无此项)": ""}])
            # Excel caps sheet names at 31 chars and forbids []:*?/\ — every name
            # here is short and Chinese, but truncating defensively is cheaper
            # than an exception that loses the whole workbook.
            df.to_excel(xl, sheet_name=name[:31], index=False)
    deps.emit(f"  工作簿: {path.name} ({len(full):,} 行 × {len(full.columns)} 列, "
              f"{len(sheets) + 2} 个 sheet)")
    return path
