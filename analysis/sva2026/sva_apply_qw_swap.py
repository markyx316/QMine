# -*- coding: utf-8 -*-
"""Promote the search-10k Qwen labels: swap the _qw outputs in as canonical, and update every prose statement that
quotes T14 or the direction check. Each replacement must match exactly once (assert), so a clash with a later edit
fails loudly instead of silently skipping. Run once, after the verifier's fixes are in."""
import os, shutil
SP = __import__("os").environ.get("SVA_WORKDIR", __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "work"))
QM = __import__("os").path.abspath(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", ".."))

# 1. files
for src, dst in (("sva_final_rows2_qw.parquet", "sva_final_rows2.parquet"),
                 ("report_tables_intent_qw.md", "report_tables_intent_v3.md"),
                 ("sva_intent2_qw.txt", "sva_intent2.txt"),
                 ("sva_intent2_qw_shares.csv", "sva_intent2_shares.csv"),
                 ("sva_intent2_qw_diffs.csv", "sva_intent2_diffs.csv"),
                 ("sva_intent2_qw_assistant_nn.parquet", "sva_intent2_assistant_nn.parquet"),
                 ("sva_intent2_qw_wrap_pairs.csv", "sva_intent2_wrap_pairs.csv"),
                 ("sva_intent2_qw_examples.txt", "sva_intent2_examples.txt"),
                 ("sva_intent2_qw_ladder_examples.txt", "sva_intent2_ladder_examples.txt")):
    shutil.copy2(f"{SP}/{src}", f"{SP}/{dst}.tmp"); os.replace(f"{SP}/{dst}.tmp", f"{SP}/{dst}")

def edit(path, subs):
    s = open(path, encoding="utf-8").read()
    for old, new in subs:
        n = s.count(old)
        assert n == 1, f"{os.path.basename(path)}: expected 1 match, found {n}: {old[:50]}"
        s = s.replace(old, new)
    open(path, "w", encoding="utf-8").write(s)

# 2. §5.1 prose
edit(f"{SP}/report_draft_B.md", [
    ("与主模型一致率 83.7%，Cohen κ 0.812（2,815 条唯一查询）", "与主模型一致率 88.0%，Cohen κ 0.855（7,032 条唯一查询）"),
    ("搜索切片 κ 0.85–0.86", "搜索切片 κ 0.85–0.88"),
    ("最稳的类：导航（同判率 95.3%）、生成与编辑（93.1%）、获取现成内容（90.3%）、裸实体（89.7%）",
     "最稳的类：导航（同判率 96.9%）、获取现成内容（94.1%）、裸实体（91.0%）、生成与编辑（90.5%）"),
    ("最不稳的类：违规或灰色（54.2%）、会话与系统指令（65.5%）、无法判定（69.1%）、核实与动态（74.6%）",
     "最不稳的类：违规或灰色（54.5%）、会话与系统指令（65.5%）、无法判定（66.7%）、核实与动态（71.8%）"),
    ("一致率 97.7%，但 κ 只有 0.614", "一致率 98.7%，但 κ 只有 0.548"),
    ("换成搜索前1万视角，也是 32/32", "换成搜索前1万视角，30 个组合也全部一致（30/30）"),
])
# 3. §0 summary
edit(f"{SP}/report_head.md", [
    ("两家公司的模型对这些差异的方向判断 32/32 一致", "两家公司的模型对这些差异的方向判断全部一致（对比搜索前1000 为 32/32，对比搜索前1万为 30/30）"),
])
# 4. companion-document editor's note
edit(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "sva_assemble_domains.py"), [("整体 κ 0.812", "整体 κ 0.855")])
# 5. HANDOFF session log
edit(f"{QM}/HANDOFF.md", [
    ("DeepSeek vs Qwen: κ 0.812 (n=2,815); assistant tail 0.761.", "DeepSeek vs Qwen: κ 0.855 (n=7,032); assistant tail 0.761."),
    ("All 32 of 32 intent differences ≥3pp have the same direction under both models.",
     "Every intent difference ≥3pp has the same direction under both models: 32/32 against search top1000, 30/30 against top10k."),
])
print("swap + prose updates applied")
