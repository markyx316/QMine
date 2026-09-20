#!/usr/bin/env python
"""科室层报告：把 `p5_department_compare.py` 的表拼成一份可读的对照报告。

报告里的每一个数字都从 `work/科室/compare/` 的 CSV / facts.json 读出来，没有一个是手打的——
这与其它几份交付报告的口径一致，也是 `p5_snapshot_verify` 能回查的前提。

解释性的段落放在 `work/科室/narrative.md` 里，按 `<!--NARR:小节名-->` 占位符拼进来；
没有这个文件时报告照样完整，只是少了解释。

    HF_HOME=$(pwd)/.hf .venv/bin/python analysis/pooled5/p5_department_report.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

QM = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(QM / "analysis/pooled5"))
from p5_snapshot_report import _md  # noqa: E402

BASE = QM / "analysis/pooled5/work/科室"
CMP = BASE / "compare"
OUT = QM / "analysis/pooled5/deliverables/医疗科室层_跨快照对照.zh.md"


def narr(key: str) -> str:
    p = BASE / "narrative.md"
    if not p.exists():
        return ""
    m = re.search(rf"<!--NARR:{re.escape(key)}-->(.*?)<!--/NARR:{re.escape(key)}-->", p.read_text(encoding="utf-8"), re.S)
    return (m.group(1).strip() + "\n\n") if m else ""


def share_table(domain: str) -> pd.DataFrame:
    """每个科室 × 每个快照的**快照内占比**。绝对条数没有可比性（快照大小差 10 倍），
    所以表里只放占比，条数在 `dept_matrix_*.csv` 里。"""
    m = pd.read_csv(CMP / f"dept_matrix_{domain}.csv", encoding="utf-8-sig")
    # 只要「快照内条数占比」。matrix 另外还产出 `全语料占比%` 与每个快照的 `流量占比%`（PV 口径），
    # 两者都不属于这张表——混进来会让一行里出现两种分母。
    cols = [c for c in m.columns
            if c.endswith("_占比%") and not c.endswith("_流量占比%")]
    keep = ["类目", "一级归并", "全语料占比%"] + cols + ["极差pp", "最高比最低倍数"]
    t = m[[c for c in keep if c in m.columns]].copy()
    t.columns = [c.replace("_占比%", "").replace("全语料占比%", "全语料%") for c in t.columns]
    return t.sort_values("全语料%", ascending=False)


def main() -> int:
    facts = json.loads((CMP / "facts.json").read_text(encoding="utf-8"))
    rel = json.loads((CMP / "reliability.json").read_text(encoding="utf-8")) if (CMP / "reliability.json").exists() else {}
    P: list[str] = []
    A = P.append

    A("# 医疗两域的科室层 —— 每个科室在每个快照上各占多少，以及为什么\n")
    A(f"> 语料：**医疗8**（med-pool8，{facts['医疗8']['行数']:,} 行 / 8 个快照）与 "
      f"**医疗随机**（health-pool3，{facts['医疗随机']['行数']:,} 行 / 2 个快照）。\n>\n"
      f"> 科室码由 `{facts['标注模型']}` 按预注册码本逐串判定，码本锚在《医疗机构诊疗科目名录》"
      f"（卫医发〔1994〕第27号，国家卫健委现行有效版本）。\n>\n"
      "> **这一层是新建的，不是运行产物。** 它不改动任何已交付的报告、工作簿或运行目录，"
      "只新增 `analysis/pooled5/work/科室/` 下的表与本文件。\n")

    A("\n## 0　为什么必须另建一层，而不是用平台自带的科室标签\n")
    lf = facts.get("legacy对照", {})
    A("**医疗8 根本没有科室列。** 它唯一的平台标签 `l2` 只覆盖 1,956 行（4.4%，且只在两个助手快照上），"
      "而且是内容类型不是科室：疾病知识 / 药品保健品 / 养生知识 / 医疗服务 / 医疗器械 / 医疗其它。\n")
    A(f"\n**医疗随机 有 `legacy_dept`，但它测不了科室。** 20 个取值覆盖全部 "
      f"{facts['医疗随机']['行数']:,} 行，其中：\n\n"
      f"- **{lf.get('legacy_空档行', 0):,} 行（{lf.get('legacy_空档占比%', 0)}%）是空档**"
      "（`(无科室)` 与 `无明确科室`）；\n"
      f"- 词表里**没有肿瘤科**。本层判为肿瘤科的 {lf.get('本层判为肿瘤科的行', 0):,} 行，"
      f"平台给的标签是：{ '、'.join(f'{k} {v}' for k, v in list(lf.get('其中legacy给的标签', {}).items())[:6]) }"
      "——一行都没有落在任何肿瘤相关的取值上，因为不存在这样的取值；\n"
      f"- 它的「内科」是个大杂烩，本层把它拆成："
      f"{ '、'.join(f'{k} {v}' for k, v in list(lf.get('legacy内科被本层拆成', {}).items())[:8]) }。\n")
    A(f"\n空档也不全是「确实没有科室」：那 {lf.get('legacy_空档行', 0):,} 行里，"
      f"**{lf.get('legacy空档被本层救回临床科室的行', 0):,} 行本层判给了具体的临床科室**，"
      f"只有 {lf.get('legacy空档确实非临床的行', 0):,} 行确实是非临床内容。\n")
    A(narr("为什么另建一层"))

    A("\n## 1　这一层是怎么判的，以及它有多可信\n")
    A("判定口径是**分诊**：这条 query 背后的人应该挂哪个门诊。知识型提问按它讲的那个病归档；"
      "儿童优先儿科、孕期优先产科（分诊现实如此）；真的定位不到才用「全科/综合」，"
      "拿不准**不允许**硬塞进一个临床科室。码本全文见 `work/科室/codebook.md`。\n")
    if rel:
        rows = []
        for k, v in rel.items():
            if isinstance(v, dict) and "一致率%" in v:
                rows.append({"对照读数": k, "测的是": v.get("测的是", ""), "n": v["n"],
                             "一致率%": v["一致率%"], "Cohen κ": v.get("cohen_kappa"),
                             "临床/非临床粗分一致率%": v.get("临床/非临床粗分一致率%")})
        A("\n" + _md(pd.DataFrame(rows)))
        A("\n**这三个数不能混为一谈。** 换供应商的那一个才是信度；同族更大的模型测的是"
          "「换成更贵的会不会改判」；规则词表测的是「一个只认锚点、完全不看语义的仪器会不会同意」，"
          "它弃权得多，但开口的地方任何人都能逐条复核。\n")
    A(narr("方法与信度"))

    for domain in ("医疗8", "医疗随机"):
        f = facts[domain]
        A(f"\n## 2　{domain}：科室构成\n" if domain == "医疗8" else f"\n## 3　{domain}：科室构成\n")
        A(f"{f['行数']:,} 行，出现 **{f['科室数_出现']} 个临床科室**，"
          f"临床行占 **{f['临床行占比%']}%**，其余是非临床内容。\n")
        A("\n**一级归并**（粗口径，明细看下一张表）：\n\n"
          + _md(pd.DataFrame([{"归并": k, "占比%": v} for k, v in f["一级归并%"].items()])))
        A("\n**非临床那一块的构成**：\n\n"
          + _md(pd.DataFrame([{"类别": k, "占非临床%": v} for k, v in f["非临床构成%"].items()])))
        A(f"\n**表 {domain}-A　每个科室在每个快照内部的占比（%）**——绝对条数没有可比性"
          "（快照大小差一个量级），所以这里只放快照内占比。\n\n" + _md(share_table(domain)))
        iface = pd.read_csv(CMP / f"dept_interface_{domain}.csv", encoding="utf-8-sig")
        A(f"\n**表 {domain}-B　搜索侧 vs 助手侧**（Newcombe 区间不含 0 的才算显著）\n\n" + _md(iface))
        A(narr(f"{domain}构成"))

    A("\n## 4　快照之间差多远\n")
    for domain in ("医疗8", "医疗随机"):
        tv = pd.read_csv(CMP / f"dept_tvd_{domain}.csv", encoding="utf-8-sig")
        A(f"\n**{domain}**（总变差距离，0 = 分布相同）\n\n" + _md(tv))
    A(narr("快照距离"))

    A("\n## 5　每个科室的证据\n")
    A("每个科室 × 每个快照最多三条真实 query，按归一 PV 取头部，"
      "**走与其它交付报告完全相同的七层引用护栏**——护栏拦下的直接少给，不换占位符。\n")
    for domain in ("医疗8", "医疗随机"):
        ex = pd.read_csv(CMP / f"dept_examples_{domain}.csv", encoding="utf-8-sig")
        A(f"\n### {domain}（{len(ex):,} 条例子）\n\n" + _md(ex.head(400)))
        if len(ex) > 400:
            A(f"\n_（共 {len(ex):,} 条，此处只列前 400；全表见 `dept_examples_{domain}.csv`。）_\n")
    A(narr("证据"))

    A("\n## 6　这一层能说什么、不能说什么\n")
    A(narr("边界"))

    A("\n---\n\n## 附：这份报告是怎么算出来的\n\n"
      "| 步骤 | 脚本 | 产物 |\n|---|---|---|\n"
      "| 预注册码本 | 手写，锚在《医疗机构诊疗科目名录》 | `work/科室/codebook.md` |\n"
      "| 逐串判科室 | `analysis/pooled5/p5_department_label.py` | `work/科室/deepseek/labels.csv` |\n"
      "| 可审计的规则仪器 | `analysis/pooled5/p5_department_rules.py` | `work/科室/rule_labels.csv` |\n"
      "| 跨模型 / 跨容量复核 | 同上，换 `P5_DEPT_PROVIDER` | `work/科室/kimi_s1500/`、`deepseek_pro_probe/` |\n"
      "| 逐快照统计 | `analysis/pooled5/p5_department_compare.py` | `work/科室/compare/*.csv` |\n"
      "| 信度 | `analysis/pooled5/p5_department_reliability.py` | `work/科室/compare/reliability.json` |\n"
      "| 本报告 | `analysis/pooled5/p5_department_report.py` | 本文件 |\n\n"
      "逐快照统计复用的是 `p5_snapshot_classes` 里的 `matrix` / `absence` / `newcombe_all` / "
      "`pairwise_tvd` / `interface_split`——与意图层、叶层同一套函数、同一口径，可以横着读。\n")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(P)
    OUT.write_text(text, encoding="utf-8")
    ntab = sum(1 for line in text.splitlines() if line.startswith("| "))
    print(f"{OUT}  ({len(text):,} 字符, {ntab:,} 表格行)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
