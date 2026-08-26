"""交付前审核报告 —— what the auditor changed, what it refused, and why.

This report is not a courtesy. It is the condition under which the audit is
allowed to exist at all.

An agent with permission to edit finished deliverables is only acceptable if
every exercise of that permission is visible. A silent audit would leave a reader
holding documents that had been altered by a process they cannot inspect, which is
strictly worse than documents that were never audited — at least those are what
the pipeline produced.

So three things are printed, and the second matters as much as the first:

1. **Every applied edit**, with the before, the after, the reason and the artifact
   the correction came from.
2. **Every REFUSED edit**, with the rule that refused it. A refusal is a finding
   about the auditor — it wanted to make a change the guardrails would not allow —
   and a report showing only the successes would be a sales document.
3. **Every warning it read and dismissed**, with why. Silence there is
   indistinguishable from not having looked.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def build(audit: dict[str, Any], state: Any, deps: Any) -> str:
    gen = Path(deps.store.gen_dir).name
    L = [
        "# 交付前审核报告 (Pre-delivery Audit)",
        "## 交付之前, 有一个 agent 把「本次运行提出过的每一条告警」和"
        "「即将交付的每一份文档」放在一起读了一遍",
        "",
        f"**运行**: `{state.get('run_id')}` / {gen}",
        "",
    ]

    if not audit.get("ran"):
        why = audit.get("skipped", "未启用")
        L += ["> ⚠️ **本次交付没有经过交付前审核** —— 原因: "
              f"`{why}`。", "",
              "这一行是刻意保留的。审核跑不起来时**不能沉默** —— "
              "否则读者拿到的文档和「审核通过的文档」在外观上完全一样。", ""]
        return "\n".join(L)

    n_ap, n_rf = audit.get("n_applied", 0), audit.get("n_refused", 0)
    unfix = audit.get("unfixable") or []
    dism = audit.get("dismissed") or []

    L += ["## 0. 这个 agent 被允许做什么, 不被允许做什么", "",
          "它是本流程中**唯一**可以改动交付物的 agent。这个权限之所以是安全的, "
          "不是因为信任它, 而是因为它只能做**一种**操作:", "",
          "| 约束 | 含义 | 不满足时 |", "|---|---|---|",
          "| 锚点唯一 | 必须逐字给出被替换的原文, 且在文件中**恰好出现一次** | 拒绝 |",
          "| 数字有出处 | 新文本中的每个数字都必须出现在它所引用的**那个** artifact 子树里 | 拒绝 |",
          "| 不得删证据 | 不能把一个数字换成「若干」这类没有数字的表述 | 拒绝 |",
          "| 语言正确 | 替换文本必须使用配置的交付语言 | 拒绝 |",
          "| 必须给理由 | 没有理由的改动无法复核 | 拒绝 |", "",
          "**它不能改动任何 artifact、代码、参数或可执行 notebook。** "
          "报告是对测量结果的**描述**, 改一处错误的表述不会改变任何测量; "
          "改 artifact 就是改测量本身, 没有任何 agent 拥有这个权限。", "",
          "**改动前的原文件保留为 `*.pre_audit.md`**, 每一处改动都可回退、可比对。", "",
          f"## 1. 结果: 应用 {n_ap} 处, 拒绝 {n_rf} 处", ""]
    if audit.get("summary"):
        L += ["> " + str(audit["summary"]).replace("\n", "\n> "), ""]

    # ------------------------------------------------------------ applied
    applied = audit.get("applied") or []
    if applied:
        L += ["## 2. 已应用的修改", "",
              "每一处都给出**改前 / 改后 / 理由 / 出处**。"
              "出处那一列不是装饰 —— 新文本中的每个数字都是在**那个** artifact "
              "子树里核对过的。", ""]
        for i, a in enumerate(applied, 1):
            L += [f"### 2.{i} `{a.get('file')}`", "",
                  f"**理由**: {a.get('reason', '')}", "",
                  f"**出处**: `{a.get('artifact_key', '')}`",
                  (f" · **改后应成立的断言**: `{a.get('check')}`" if a.get("check") else ""),
                  "",
                  "改前:", "", "```", str(a.get("anchor", "")), "```", "",
                  "改后:", "", "```", str(a.get("replacement", "")), "```", ""]
    else:
        L += ["## 2. 已应用的修改", "",
              "本次审核**没有改动任何文档**。", "",
              "这是一个有效结果, 不是审核失败 —— 为了显得尽职而制造修改, "
              "比什么都没找到更糟。", ""]

    # ------------------------------------------------------------ refused
    refused = audit.get("refused") or []
    if refused:
        L += ["## 3. 被拒绝的修改", "",
              "**这一节和第 2 节同等重要。** 被拒绝意味着这个 agent 想做一处"
              "护栏不允许的改动 —— 这是关于**审核者本身**的发现。"
              "只列成功项的报告是宣传材料, 不是记录。", "",
              "| 文件 | 它想改什么 | 被哪条规则拦下 |", "|---|---|---|"]
        for r in refused:
            L.append(f"| `{r.get('file')}` | {str(r.get('reason', ''))[:70]} | "
                     f"{str(r.get('why', ''))[:110]} |")
        L.append("")

    # ------------------------------------------------------------ unfixable
    if unfix:
        L += ["## 4. 发现了但无法用「定点替换」修复的问题", "",
              "结构性问题往往没有一行就能改好的写法。硬塞成一处替换只会被拒绝, "
              "所以它们被登记进**发现台账** —— 会被下一个 generation 继承, "
              "并且只有在它自己的断言重新成立时才会自动关闭。", ""]
        for o in unfix:
            L.append(f"- **[{o.get('severity', '')}]** {o.get('claim', '')} "
                     f"← `{o.get('artifact_key', '')}`")
        L.append("")

    # ------------------------------------------------------------ dismissed
    if dism:
        L += ["## 5. 读过并判定无需处理的告警", "",
              "列出来是因为**沉默和没看过无法区分**。", ""]
        for d in dism:
            L.append(f"- {d}")
        L.append("")
    return "\n".join(x for x in L if x is not None)
