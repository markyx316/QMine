# -*- coding: utf-8 -*-
"""Combine the five domain deep-dives into docs/SEARCH_VS_ASSISTANT_2026_领域深挖.zh.md, demoting headings one level
under a per-domain H1 and prefixing an editor's note. Fails if a domain file is missing (unless --partial)."""
import os, re, sys
SP = __import__("os").environ.get("SVA_WORKDIR", __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "work"))
QM = __import__("os").path.abspath(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", ".."))
ORDER = [("金融", "fin"), ("医疗", "med"), ("教育", "edu"), ("影视", "film"), ("人物", "ppl")]
head = """# 2026 同题对比：搜索 vs AI 助手 —— 分领域深挖

> 本文是主报告《[SEARCH_VS_ASSISTANT_2026](SEARCH_VS_ASSISTANT_2026.zh.md)》§7 的全文附件，每个领域由独立的分析完成，口径与主报告一致（清洗 v3、user 行、同一模型盲标的统一意图框架）。
>
> **编者注**：各领域深挖写于统一意图标注全部完成之前：金融、教育两节没有“搜索前1万”的统一意图占比，五节都没有第二模型的一致性（整体 κ 0.855、方向检验全部一致，见主报告表 T14 与 §5.1）。文中“行占比差 <4 个百分点不作解读”的门槛写于第二轮清洗审计之前；主报告 §1.4 的最小可读差异为 5 域合计头部 5 个百分点、长尾 7 个百分点，分领域只解读大于该领域漏检率上界（头部 6%–32%）的差异，人物头部不解读小类差异。人物一节助手头部“核实与动态” 11.0% 未扣除第二轮审计普查发现的系统文本（占 73.5%），修正后约为 3.7%，以主报告 §2.3、§5.2 为准。各领域文中的其余数字均在 v3 清洗口径上由代码重新计算，并附有中间文件名（位于分析工作目录，复现方式见主报告附录 E）。
"""
parts = [head]
for name, key in ORDER:
    p = f"{SP}/sva_dom_{key}.md"
    if not os.path.exists(p):
        if "--partial" in sys.argv:
            parts.append(f"\n# {name}\n\n（待补）\n"); continue
        raise SystemExit(f"missing {p}")
    t = open(p, encoding="utf-8").read().strip()
    lines = t.splitlines()
    if lines and re.match(r"^#{1,2} ", lines[0]):
        lines = lines[1:]
    body = "\n".join(("#" + l) if re.match(r"^#{1,5} ", l) else l for l in lines)
    parts.append(f"\n---\n\n# {name}\n\n{body.strip()}\n")
out = f"{QM}/docs/SEARCH_VS_ASSISTANT_2026_领域深挖.zh.md"
open(out, "w", encoding="utf-8").write("\n".join(parts))
print("wrote", out, sum(len(x) for x in parts), "chars")
