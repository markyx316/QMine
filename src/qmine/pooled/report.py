"""Render the cross-snapshot report. EVERY NUMBER COMES OUT OF A TABLE.

There is exactly one table renderer and exactly one ordering authority, and
neither is duplicated anywhere in this module. Both rules were bought:

* The top-ten section used to re-sort the matrix itself instead of reading the
  members table. On a tie at rank ten the ten classes shown were not the ten
  summed in the traffic row — measured 5.25pp apart on one corpus.
* Shares in the matrix are already rounded; formatting a rounded value again
  printed one quantity as 98.24 in one table and 98.23 in another.

Authored prose enters only at `<!--NARR:key-->` placeholders and may contain no
number that is not in these tables — `pooled.verify` checks that afterwards.
`「」` is reserved for real queries so every quotation can be looked back up;
run-authored text (definitions, user needs, audit notes) is converted to `“”`
before it is printed, and a query that itself contains `「」` is wrapped in `『』`.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from .manifest import SnapshotManifest
from .stats import one_sided_upper

NARR_KEYS = ["导读", "意图层", "叶层", "独有性", "路线交叉", "边界"]
_TRUE = {"True", "true", "是", "1"}


def _md(t: pd.DataFrame | None, cols: list[str] | None = None) -> str:
    """The only table renderer."""
    if t is None or not len(t):
        return "_（本表在这次运行里为空。）_\n"
    t = t[[c for c in (cols or list(t.columns)) if c in t.columns]]
    head = "| " + " | ".join(str(c) for c in t.columns) + " |"
    sep = "|" + "|".join("---" for _ in t.columns) + "|"
    body = []
    for _, r in t.iterrows():
        cells = []
        for c in t.columns:
            v = r[c]
            # BOOLS FIRST. `isinstance(True, int)` is True in Python, so the
            # integer branch caught every boolean column and printed `超出噪声`
            # as `1` — a column whose whole job is to say 是 or 否.
            if isinstance(v, (bool, np.bool_)):
                s_ = "是" if v else "否"
                cells.append(s_)
                continue
            if isinstance(v, (int, np.integer)):
                s = f"{v:,}" if abs(int(v)) >= 10000 else str(int(v))
            elif isinstance(v, (float, np.floating)):
                s = "" if pd.isna(v) else ("∞" if np.isinf(v) else f"{v:g}")
            elif isinstance(v, (bool, np.bool_)):
                s = "是" if v else "否"
            else:
                s = str(v)
                s = "是" if s == "True" else "否" if s == "False" else s
            cells.append(s.replace("|", r"\|").replace("\n", " "))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, sep, *body]) + "\n"


def _pct(x: Any) -> str:
    return "—" if x is None or (isinstance(x, float) and pd.isna(x)) else f"{float(x):.2f}"


def _f2(x: Any) -> str:
    """Two decimals, from the UNROUNDED value, everywhere.

    A table that did `.round(2)` and a card that did `f"{v:.2f}"` disagreed on
    every value ending in 5: the same range printed as 1.74 in the table and
    1.75 in the card three screens later. One formatter, one answer.
    """
    return "" if x is None or pd.isna(x) else f"{float(x):.2f}"


def _blank(x: Any) -> str:
    """A CSV round-trip turns an empty cell into the float nan, and `str(nan)`
    is the four characters `nan` — which shipped as `缺席于 nan`."""
    s = str(x if x is not None else "").strip()
    return "" if s.lower() in ("nan", "none", "<na>") else s


def _prose(t: Any) -> str:
    """Run-authored text: release the quotation marks reserved for real queries."""
    return str(t or "").replace("「", "“").replace("」", "”")


def _q(t: Any) -> str:
    s = str(t)
    return f"『{s}』" if ("「" in s or "」" in s) else f"「{s}」"


def _toc(text: str) -> str:
    out = ["## 目录", ""]
    for line in text.splitlines():
        m = re.match(r"^(#{2,3}) (.+)$", line)
        if not m or m.group(2).strip() == "目录":
            continue
        depth = len(m.group(1)) - 2
        title = m.group(2).strip()
        anchor = re.sub(r"[^\w一-鿿-]+", "-", title.replace(" ", "-")).strip("-").lower()
        out.append("  " * depth + f"- [{title}](#{anchor})")
    out.append("")
    block = "\n".join(out)
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("## "):
            return "\n".join(lines[:i] + [block] + lines[i:])
    return text + "\n" + block


def _splice(text: str, narrative: dict[str, str] | None) -> str:
    """Fill the authored placeholders. An absent key leaves the placeholder empty."""
    if not narrative:
        return text
    unknown = sorted(set(narrative) - set(NARR_KEYS))
    if unknown:
        raise ValueError(f"narrative keys {unknown} are not placeholders in this report; "
                         f"allowed: {NARR_KEYS}")
    for k, body in narrative.items():
        if not str(body).strip():
            continue
        text = text.replace(f"<!--NARR:{k}-->\n<!--/NARR:{k}-->",
                            f"<!--NARR:{k}-->\n{str(body).strip()}\n<!--/NARR:{k}-->")
    return text


def _narr(key: str) -> str:
    return f"<!--NARR:{key}-->\n<!--/NARR:{key}-->\n"


# ----------------------------------------------------------------- the report

def build_report(frames: dict[str, pd.DataFrame], summary: dict[str, Any],
                 man: SnapshotManifest, *, title: str, run_id: str, generation: str,
                 img_dir: str = "img", narrative: dict[str, str] | None = None,
                 workbook_name: str = "") -> str:
    P: list[str] = []
    a = P.append
    snaps = man.snapshots
    sizes = man.sizes
    n_rows = summary["n_rows"]
    axis_word = "分层" if man.axis == "stratum" else "时间"

    a(f"# {title} —— 每个意图、每个聚类叶在 {len(snaps)} 个快照上的逐类对照\n")
    a(f"> 运行 `{run_id}` / `{generation}`，语料 {n_rows:,} 行，"
      f"{len(snaps)} 个快照（{'、'.join(f'{man.display(s)} {sizes[s]:,} 行' for s in snaps)}）。\n"
      f"> 对比轴是**{axis_word}**："
      + ("这些快照是同一时期的不同抽样方式，之间没有先后，所以不要读成「变化」。\n"
         if man.axis == "stratum" else "这些快照按时间先后排列。\n")
      + (f"> 逐行数据在 `{workbook_name}`。\n" if workbook_name else ""))

    # ---------------------------------------------------- the preamble, computed
    n_min = min(sizes.values())
    n_max_name = max(sizes, key=lambda s: sizes[s])
    n_max = sizes[n_max_name]
    a("\n## 读这份表之前必须知道的六件事\n")
    a(f"1. **一切占比都是快照内占比。** 各快照行数相差 {n_max / max(n_min, 1):.1f} 倍"
      f"（最大 {man.display(n_max_name)} {n_max:,} 行，最小 {n_min:,} 行），"
      "跨快照比原始条数只是在比导出文件的大小。\n")
    a(f"2. **0 条不等于不存在。** 在 {n_min:,} 行里 0 条，**单侧 97.5% 上界**"
      f"（即 95% 双侧区间的上端）仍有 {100 * one_sided_upper(n_min):.3f}%；"
      f"同样的 0 条落在 {n_max:,} 行里，上界只有 "
      f"{100 * one_sided_upper(n_max):.3f}%。所以每个缺席都配了期望条数与 P(0)，"
      "正文只引用「判定」列。\n")
    a("3. **原始条数的「独占」会被样本量带偏。** 除了 `独占快照`，每张矩阵还给 `均衡归属%`"
      "（该快照内占比 ÷ 各快照内占比之和）——把每个快照当成同等大小来看归属。\n")
    a("4. **TVD 必须配同源噪声上界。** 把两个快照的行合在一起、再按它们**原来的行数**随机切回两份，"
      "这两份之间本来就有距离；观测值不超过那个上界，这一格就读作「测不出差别」，而不是「没有差别」。"
      "（切回原来的行数是关键：改成各自对半切，上界会高 1.4–1.8 倍，差得越悬殊越严重，"
      "真实的差别就被判成「测不出」。）\n")
    a("5. **两条路线是独立产生的。** 意图体系由架构师写、聚类叶由嵌入得到，"
      "所以「这个叶主要落在哪个意图」是一次测量，不是定义。\n")
    qg = summary.get("quote_guard", {})
    a(f"6. **例子是真实的行，但不是所有行都能引。** 七层引用护栏拦下 "
      f"{qg.get('任一层拦下', 0):,} 行，可引 {qg.get('可引行数', 0):,} 行；"
      "护栏只影响**引用**，不影响任何一个统计。一个类如果所有行都被拦下，"
      "卡片会写明条数与原因，而不是留白。\n")
    a("\n" + _narr("导读"))

    # ------------------------------------------------------------- 1. profiles
    a("\n## 1　每个快照各自长什么样\n")
    cov = frames.get("coverage")
    for level, zh in _levels(summary):
        a(f"\n### 表 1-{zh}\n")
        a(_md(cov[cov["层级"] == zh] if cov is not None else None,
              ["快照", "n", "类目数", "有效类目数", "熵", "HHI", "首位类目",
               "首位占比%", "前3占比%", "前5占比%", "前10占比%", "前10流量占比%", "流量有效n"]))
    a("\n> `有效类目数` = exp(熵)，回答「这个快照实际上被多少个类撑起来」。"
      "`流量有效n` = 1/Σw²：流量口径的有效样本量远小于行数，一行的权重可以占整个快照的几个百分点，"
      "不给这个数，读者会拿行占比的精度去读流量占比。\n")
    a("\n### 表 1-总览\n")
    a(_md(pd.DataFrame([{"层级": zh, **{k: v for k, v in d.items() if k != "level"}}
                        for zh, d in summary["levels"].items()])))

    # ------------------------------------------------------------ 2. distances
    a(f"\n## 2　快照两两之间隔多远（{len(summary['levels'])} 个层级）\n")
    tvd = frames.get("pairwise_tvd")
    for level, zh in _levels(summary):
        a(f"\n### 表 2-{zh}\n")
        a(_md(tvd[tvd["层级"] == zh] if tvd is not None else None,
              ["a", "b", "n_a", "n_b", "TVD", "TVD低", "TVD高", "同源噪声上界",
               "超出噪声(区间下端)", "CramersV", "秩相关rho", "前5重合个数", "首位是否相同"]))
    a(f"\n![快照两两距离]({img_dir}/快照两两距离.png)\n")
    a("\n> 图中蓝条是 TVD 与它的自助法区间，红线是同源噪声上界。"
      "两处的「超出噪声」用的**不是同一条线**，所以列名写明了是哪条："
      "这里是**区间下端**高过上界（更严），表 交叉-A/B 是**点估计**高过上界（更松）。`秩相关rho` 看的是次序、"
      "TVD 看的是量差：两个快照可以量差很大而次序几乎不变（份额整体缩放），"
      "也可以量差不大而次序全乱。\n")

    # --------------------------------------------------------------- 2.2 top-n
    a("\n## 2.2　每个快照自己的前十\n")
    a("\n> **这张表只能竖着读。** 每个快照各排各的，第 10 名在两个快照里几乎从来不是同一个类；"
      "它回答「这个快照的头部有多集中」，不回答「同样这十个类在别的快照里占多少」。\n")
    mem, tp = frames.get("topn_members"), frames.get("topn_coverage")
    for level, zh in _levels(summary):
        if level not in ("td_l1", "bu_leaf"):
            continue
        a(f"\n### 表 2.2-{zh}\n")
        a(_topn_table(mem, tp, zh, man))

    # ------------------------------------------------------ 3..N per level
    for i, (level, zh) in enumerate(_levels(summary), start=3):
        a(f"\n## {i}　{zh} × 快照\n")
        m = frames.get(f"matrix_{level}")
        if level == "td_l1":
            a(f"\n![L1意图占比热图]({img_dir}/L1意图_占比热图.png)\n")
            a(f"\n![L1意图均衡指数热图]({img_dir}/L1意图_均衡指数热图.png)\n")
            a("\n" + _narr("意图层"))
        if level == "bu_leaf":
            a(f"\n![叶占比热图]({img_dir}/叶_占比热图.png)\n")
            a(f"\n![叶均衡指数热图]({img_dir}/叶_均衡指数热图.png)\n")
            a("\n" + _narr("叶层"))
        a(f"\n### 表 {zh}-A　每个类在每个快照里的条数与快照内占比\n")
        a(_md(_table_a(m, man)))
        a(f"\n### 表 {zh}-B　Wilson 95% 区间\n")
        a(_md(_table_b(m, man)))
        a(f"\n### 表 {zh}-C　归属与均衡指数\n")
        a(_md(_table_c(m, man)))
        if m is not None and len(m) and all(f"{man.display(s)}_流量占比%" in m.columns
                                            for s in snaps):
            a(f"\n### 表 {zh}-D　流量加权占比\n")
            a(_md(m[["类目"] + [f"{man.display(s)}_流量占比%" for s in snaps]]))
            if man.uniform_weight:
                a(f"\n> {'、'.join(man.display(s) for s in man.uniform_weight)} 的导出没有可用权重，"
                  "按均匀权重处理，所以这几个快照的流量占比恒等于行占比。\n")
        surf = frames.get(f"surface_{level}")
        if surf is not None and len(surf):
            a(f"\n### 表 {zh}-E　分组逐类差异\n")
            a(_md(surf, [c for c in surf.columns if c != "key"]))
        ab = frames.get(f"absence_{level}")
        a(f"\n### 表 {zh}-F　缺席的可检出性\n")
        if ab is None or not len(ab):
            a(f"_本层没有任何类在任何快照里是 0 条——{summary['levels'][zh]['n_classes']} 个类"
              f"全部出现在全部 {len(snaps)} 个快照里。_\n")
        else:
            a(_md(ab, ["类目", "缺席快照", "该快照n", "参照占比%", "期望条数", "P(0)",
                       "该快照上界%", "最高占比快照", "最高占比%", "判定"]))
        conf = frames.get(f"confidence_{level}")
        if conf is not None and len(conf):
            a(f"\n### 表 {zh}-H　标注可靠度\n")
            a(_md(_table_h(conf, man)))
            a("\n> 占比差本身不说明边界靠不靠得住：一个类可以份额稳定，而那一层的行恰好全是"
              "分类器勉强判下来的。每格都带 n，`置信度最低快照` 只在该快照 ≥20 行时参与。\n")
        nc = frames.get(f"newcombe_{level}")
        a(f"\n### 表 {zh}-G　变动最大的 30 项\n")
        if nc is not None and len(nc):
            g = nc[nc["显著"].astype(str).isin(_TRUE)]
            g = g.reindex(g["差_pp"].abs().sort_values(ascending=False).index).head(30)
            a(f"\n_{len(nc)} 对比较中 {int(nc['显著'].astype(str).isin(_TRUE).sum())} 对显著。_\n")
            a(_md(g, ["类目", "a", "b", "占比_a%", "占比_b%", "差_pp", "低_pp", "高_pp", "类型"]))
        else:
            a(_md(None))

    # -------------------------------------------------------------- uniqueness
    n = len(summary["levels"]) + 3
    a(f"\n## {n}　独有性：哪些类只出现在一边\n")
    a("\n" + _narr("独有性"))
    for level, zh in _levels(summary):
        m, surf = frames.get(f"matrix_{level}"), frames.get(f"surface_{level}")
        excl = (m[m["独占快照"].map(_blank).str.len() > 0] if m is not None else None)
        a(f"\n### {zh}\n")
        if excl is not None and len(excl):
            a(f"\n{len(excl)} 个类的全部行来自同一个快照：\n")
            a(_md(excl, ["类目", "全语料条数", "独占快照"]))
        else:
            a(f"\n_没有任何{zh}只出现在一个快照里。_\n")
        if surf is not None and len(surf) and "归属" in surf.columns:
            only = surf[surf["归属"].astype(str).str.startswith("仅")]
            if len(only):
                a(_md(only, ["类目", "归属", "另一边期望条数", "另一边P(0)", "缺席判定"]))
    pw = frames.get("exclusive_power")
    if pw is not None and len(pw):
        a("\n### 检出力：如果真有只在一边出现的类，这份数据看得见吗\n")
        a(_md(pw))
        a("\n> 「一边独有 = 0」只有配上检出力才是结论。这里取较小一侧最稀有的类，"
          "问同样的发生率放到较大一侧会期望几条、一条都不出现的概率多大。"
          "两个方向的检出力不对称是**结构性的**（n 不同），不是一个结果。\n")

    # ---------------------------------------------------------- characteristic
    a(f"\n## {n + 1}　每个快照的特征类\n")
    a("\n> 判据：对**其余每一个**快照做 Newcombe 检验，方向全部一致且区间都不含 0。"
      "快照越多这个条件越严。原始「独占」几乎永远是 0（一个类只要在别处出现一条就不算），"
      "所以那个口径对「这个快照有什么特别的」几乎没有信息量。\n")
    for level, zh in _levels(summary):
        sig = frames.get(f"signature_{level}")
        a(f"\n### {zh}\n")
        if sig is None or not len(sig):
            a("\n_本层没有任何类通过这个判据。_\n")
            continue
        for direction, label in (("显著高于其余全部", "特征类"), ("显著低于其余全部", "反特征类")):
            s = sig[sig["方向"] == direction].sort_values(
                ["快照", "该快照占比%"], ascending=[True, False])
            a(f"\n**{label}**（{len(s)} 项）\n")
            a(_md(s, ["快照", "类目", "该快照条数", "该快照n", "该快照占比%",
                      "其余快照合并占比%", "最弱一对的区间端点pp", "对比了几个快照"]))

    # ------------------------------------------------------------- cross-route
    a(f"\n## {n + 2}　同一个类，在不同快照里内部构成一样吗\n")
    a("\n> 这是「占比一样但实现方式不同」的检验。内层 TVD 必须配同源噪声上界：在几十到几百行的"
      "格子里，两份来自同一分布的样本本来就能给出 0.2–0.3 的 TVD。\n")
    a("\n" + _narr("路线交叉"))
    for key, label in (("intent_leafmix", "表 交叉-A　同一个意图，内部的叶构成"),
                       ("leaf_intentmix", "表 交叉-B　同一个叶，内部的意图构成")):
        t = frames.get(key)
        a(f"\n### {label}\n")
        if t is not None and len(t):
            a(f"\n_{len(t)} 格，其中 "
              f"{int(t['超出噪声(点估计)'].astype(str).isin(_TRUE).sum())} 格超出噪声。_\n")
            a(_md(t.sort_values("内层TVD", ascending=False).head(40)))
        else:
            a("\n_没有任何一格两侧都达到最小行数要求。_\n")

    # ------------------------------------------------------------------- cards
    a(f"\n## {n + 3}　逐类卡片\n")
    for level, zh in _levels(summary):
        if level not in ("td_l1", "bu_leaf"):
            continue
        a(f"\n### {zh}\n")
        a(_cards(frames, level, man))

    # ------------------------------------------------------------------- facts
    a(f"\n## {n + 4}　机械事实与引用护栏\n")
    a("\n" + _narr("边界"))
    a(_md(pd.DataFrame([{"项": k, "值": v} for k, v in qg.items()])))
    a("\n> 七层护栏，从深到浅：L1 运行自己的风控图层命中的行；L2 与垂类无关的硬规则；"
      "L3 两类词共现；L4 运行的风控机器点名过的串；L5 整类不引（引用即复述）；"
      "L6 声明的追加正则；L7 逐串读过之后标出的名单。**护栏只影响引用，不影响任何统计。**\n")
    a(f"\n> 复现：`qmine compare {run_id} --generation {generation}`。"
      "所有表在同目录的 `tables/` 下，报告里的每一个数字都由脚本从那些表里取出来渲染。\n")

    text = "".join(P)
    text = _splice(text, narrative)
    return _toc(text)


# -------------------------------------------------------------------- helpers

def _levels(summary: dict[str, Any]) -> list[tuple[str, str]]:
    return [(d["level"], zh) for zh, d in summary["levels"].items()]


def _table_a(m: pd.DataFrame | None, man: SnapshotManifest) -> pd.DataFrame | None:
    if m is None or not len(m):
        return None
    out = pd.DataFrame({"类目": m["类目"], "全语料条数": m["全语料条数"]})
    for s in man.snapshots:
        z = man.display(s)
        out[z] = [f"{_pct(p)}% ({int(k):,})"
                  for p, k in zip(m[f"{z}_占比%"], m[f"{z}_条数"])]
    out["极差pp"] = [_f2(v) for v in m["极差pp"]]
    out["最高快照"] = [_blank(v) for v in m["最高快照"]]
    out["出现快照数"] = m["出现快照数"]
    return out


def _table_b(m: pd.DataFrame | None, man: SnapshotManifest) -> pd.DataFrame | None:
    if m is None or not len(m):
        return None
    out = pd.DataFrame({"类目": m["类目"]})
    for s in man.snapshots:
        z = man.display(s)
        out[z] = [f"{_pct(p)} [{_pct(lo)}–{_pct(hi)}]" for p, lo, hi in
                  zip(m[f"{z}_占比%"], m[f"{z}_95CI低%"], m[f"{z}_95CI高%"])]
    return out


def _table_c(m: pd.DataFrame | None, man: SnapshotManifest) -> pd.DataFrame | None:
    if m is None or not len(m):
        return None
    out = pd.DataFrame({"类目": m["类目"]})
    for s in man.snapshots:
        z = man.display(s)
        out[f"{z}_归属%"] = ["" if pd.isna(v) else f"{float(v):.1f}" for v in m[f"{z}_均衡归属%"]]
        out[f"{z}_均衡指数"] = [_f2(v) for v in m[f"{z}_均衡指数"]]
    out["独占快照"] = [_blank(v) for v in m["独占快照"]]
    return out


def _table_h(conf: pd.DataFrame, man: SnapshotManifest) -> pd.DataFrame:
    out = pd.DataFrame({"类目": conf["类目"]})
    for s in man.snapshots:
        z = man.display(s)
        if f"{z}_置信度均值" not in conf.columns:
            continue
        cells = []
        for c, nn, am, bm in zip(conf[f"{z}_置信度均值"], conf[f"{z}_n"],
                                 conf.get(f"{z}_意图歧义%", pd.Series(np.nan, index=conf.index)),
                                 conf.get(f"{z}_聚类歧义%", pd.Series(np.nan, index=conf.index))):
            cells.append("—" if pd.isna(c) else
                         f"{float(c):.3f}（n={int(nn)}；歧义 {float(am):.1f}%/{float(bm):.1f}%）")
        out[z] = cells
    for c in ("置信度最低快照", "置信度极差", "全语料置信度均值"):
        if c in conf.columns:
            out[c] = conf[c].replace("", "（各快照均 <20 行，不比）")
    return out


def _topn_table(mem: pd.DataFrame | None, tp: pd.DataFrame | None,
                zh: str, man: SnapshotManifest) -> str:
    """Ranks 1..10 read from the MEMBERS table — the single ordering authority."""
    if mem is None or tp is None or not len(mem):
        return _md(None)
    mm = mem[mem["层级"] == zh]
    tt = tp[tp["层级"] == zh].set_index("快照")
    rows: list[dict[str, Any]] = []
    for rank in range(1, 11):
        r: dict[str, Any] = {"名次": f"第{rank}"}
        for s in man.snapshots:
            z = man.display(s)
            g = mm[(mm["快照"] == z) & (mm["名次"] == rank)]
            if len(g) and z in tt.index:
                r[z] = f"{g['类目'].iloc[0]} {100 * int(g['条数'].iloc[0]) / int(tt.at[z, 'n']):.2f}%"
            else:
                r[z] = ""
        rows.append(r)
    for cut in (3, 5, 10):
        r = {"名次": f"前{cut}合计"}
        for s in man.snapshots:
            z = man.display(s)
            r[z] = f"{float(tt.at[z, f'前{cut}行占比%']):.2f}%" if z in tt.index else ""
        rows.append(r)
    r = {"名次": "前10合计（流量）"}
    notes: list[str] = []
    for s in man.snapshots:
        z = man.display(s)
        if z not in tt.index:
            r[z] = ""
            continue
        v = float(tt.at[z, "前10流量占比%"])
        swing = float(tt.at[z, "前10流量摆动pp"])
        # A swing WIDTH printed beside a point value reads as a symmetric
        # interval. The range goes inside the cell.
        if swing > 0.005:
            r[z] = (f"{v:.2f}%（并列可在 {float(tt.at[z, '前10流量占比%低']):.2f}–"
                    f"{float(tt.at[z, '前10流量占比%高']):.2f} 之间摆动）")
        else:
            r[z] = f"{v:.2f}%"
        if str(tt.at[z, "边界并列"]) in _TRUE:
            notes.append(f"{z}：第 10 名有 {int(tt.at[z, '并列类数'])} 个类并列在 "
                         f"{int(tt.at[z, '第10名条数'])} 条，需选 {int(tt.at[z, '并列需选'])} 个"
                         + ("；行占比合计对挑法免疫，流量合计不免疫。"
                            if swing > 0.005 else
                            "；该快照权重均匀，行与流量合计都与挑法无关。"))
    rows.append(r)
    out = _md(pd.DataFrame(rows))
    if notes:
        out += "\n" + "\n".join(f"> {x}" for x in notes) + "\n"
    return out


def _cards(frames: dict[str, pd.DataFrame], level: str, man: SnapshotManifest) -> str:
    m = frames.get(f"matrix_{level}")
    if m is None or not len(m):
        return _md(None)
    ex = frames.get(f"examples_{level}")
    nc = frames.get(f"newcombe_{level}")
    out: list[str] = []
    for _, r in m.iterrows():
        out.append(f"\n#### {r['类目']}　（全语料 {int(r['全语料条数']):,} 行，"
                   f"占 {_pct(r['全语料占比%'])}%）\n")
        if _blank(r.get("定义")):
            out.append(f"\n> {_prose(r['定义'])}\n")
        shares = " · ".join(
            f"{man.display(s)} **{_pct(r[f'{man.display(s)}_占比%'])}%**"
            f"({int(r[f'{man.display(s)}_条数']):,})" for s in man.snapshots)
        out.append(f"\n- 逐快照占比：{shares}\n")
        line = (f"- 极差 {_f2(r['极差pp'])}pp，最高在 {_blank(r['最高快照'])}、"
                f"最低在 {_blank(r['最低快照'])}")
        if _blank(r.get("缺席快照")):
            line += f"；缺席于 {_blank(r['缺席快照'])}"
        if _blank(r.get("独占快照")):
            line += f"；**独占 {_blank(r['独占快照'])}**"
        out.append(line + "\n")
        if nc is not None and len(nc):
            g = nc[(nc["key"].astype(str) == str(r["key"]))
                   & nc["显著"].astype(str).isin(_TRUE)]
            if len(g):
                g = g.reindex(g["差_pp"].abs().sort_values(ascending=False).index).head(2)
                out.append("- 最大的显著变动：" + "；".join(
                    f"{x['a']}→{x['b']} {float(x['差_pp']):+.2f}pp "
                    f"[{float(x['低_pp']):+.2f}, {float(x['高_pp']):+.2f}]"
                    for _, x in g.iterrows()) + "\n")
            else:
                out.append("- 没有任何一对快照的差异显著\n")
        for extra, fmt in (("主导意图", "主导意图占比%"), ("主导叶", "主导叶占比%")):
            if extra in m.columns and _blank(r.get(extra)):
                out.append(f"- {extra}：{r[extra]}（{_pct(r[fmt])}%）\n")
        if "风险标注" in m.columns and _blank(r.get("风险标注")):
            out.append(f"- 命名时被标为风险叶：{_prose(r['风险标注'])}\n")
        if "体系内风险标注" in m.columns and str(r.get("体系内风险标注", "")) == "是":
            out.append("- 体系内标为风险类目\n")
        if ex is not None and len(ex):
            for s in man.snapshots:
                z = man.display(s)
                g = ex[(ex["key"].astype(str) == str(r["key"])) & (ex["快照"] == z)]
                top = g[g["取法"] == "流量最高"]["query"].head(3).tolist()
                if top:
                    # THE WHOLE STRING, NOT 26 CHARACTERS OF IT. A truncated
                    # quote is not a row anyone can look up: the hard-rule scan
                    # searches the document for corpus strings, and a prefix
                    # matches none of them — so a blocked string printed in
                    # truncated form was reported as "0 printed". Queries are
                    # capped at `data.max_query_len` anyway.
                    out.append(f"- {z}：" + " / ".join(_q(str(x)) for x in top) + "\n")
                elif len(g):
                    out.append(f"- {z}：不引原文（该快照下这个类的 "
                               f"{int(g['该类该快照条数'].iloc[0]):,} 行全部命中风控图层或引用护栏）\n")
    return "".join(out)
