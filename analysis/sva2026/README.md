# analysis/sva2026 — 2026 搜索 vs AI 助手 同题对比（复现包）

本目录复现两份报告：

- 主报告：[`docs/SEARCH_VS_ASSISTANT_2026.zh.md`](../../docs/SEARCH_VS_ASSISTANT_2026.zh.md)
- 领域深挖：[`docs/SEARCH_VS_ASSISTANT_2026_领域深挖.zh.md`](../../docs/SEARCH_VS_ASSISTANT_2026_领域深挖.zh.md)

这里全是**跑完之后的分析脚本**，不是 QMine 流水线的一部分：

- 不修改 `src/`，测试也不收集这个目录；
- 脚本读取的是已完成 run 的产物（`runs/ai04`、`runs/*-pool`）和 `data/raw/`。

## 目录与路径

- **脚本**：放在本目录。数据与中间产物放在 `./work/`，可以用环境变量 `SVA_WORKDIR` 指到别处。
- **运行方式**：在本目录下执行，例如：
  ```bash
  HF_HOME=../../.hf ../../.venv/bin/python -W ignore sva_report_tables.py
  ```
- **已验证可移植**：`sva_report_tables.py` 和 `sva_report_tables_intent.py` 在本目录下重跑，生成的表格文件与报告里插入的完全一致（逐字节相同）。
- **例外**：`audit2_*` 系列脚本由审计员编写，里面还有少量绝对路径，只用于留档。

## 运行顺序

**会调用付费 API 的步骤标了【付费】。**

1. **清洗分层**（v3）
   ```bash
   ../../.venv/bin/python ../../tools/clean_assistant_functional.py -o work/clean_v3
   ```
   规则写在该工具的 docstring 里；v2 → v3 的差异由 `sva_tier_diff.py` 输出。
2. **统一意图标注**【付费】
   ```bash
   ../../.venv/bin/python ../../tools/label_unified_intent.py --tiered work/clean_v3 -o work/label_full --full-cells
   ```
   - 主标注 deepseek-v4-flash，第二模型 qwen3.8-flash 抽 20%。
   - `sva_reconstruct_ds.py` 可以从原始日志 `raw_ds.jsonl` 重建标签，重建结果与正式输出逐条一致。
3. **补标**【付费】
   - `sva_label_extra.py --run`：为语义近邻补标。
   - `sva_label_search10k.py`：给搜索前1万的其余字符串补标，含 1,000 条锚点复测，qwen 抽 10%。
4. **语义近邻**（本地向量模型 bge-base-zh-v1.5）
   - `sva_semantic_nn_v3.py`：计算近邻，结果含两个对照组。
   - `sva_nn_lenctrl.py`：按查询长度分档检验，并用 bge-large 做稳健性检验。
5. **分析总表**
   ```bash
   LABELS=work/label_full/unique_labels.parquet \
   EXTRA_LABEL_FILES=work/label_extra/labels.parquet,work/label_search10k/labels.parquet \
     ../../.venv/bin/python sva_build_final2.py
   ```
   产物是 `work/sva_final_rows2.parquet`。`label_search10k/labels.parquet` 带 qwen 列（`u_qw`），表 T14 的一致性和方向检验才包括搜索前1万扩展部分（搜索前1万共 5,439 行有第二模型标签）。
6. **意图分析**：`sva_intent2.py`。输出占比与差异（带置信区间）、双模型一致性、方向检验、个人处境、crosswalk、语义距离、包装与近邻转移、六级阶梯。
7. **形态与行为指标**：
   - `sva_metrics.py`、`sva_question_specificity.py`、`sva_conv_markers.py`
   - `sva_logodds.py`、`sva_overlap.py`
   - `sva_behaviour_v3.py`、`sva_persona_cues.py`
   - `sva_own_tax.py`、`sva_pvw_cleanimpact_v3.py`
8. **报告表格与图**：
   - `sva_report_tables.py`：生成 T1–T11 和 T21。
   - `sva_report_tables_intent.py`：生成 T12–T20。
   - `sva_figs.py 1 2 3 4 5 6`：输出到 `docs/img/sva2026/`。
9. **拼装**：
   - `sva_assemble_domains.py`：拼装领域深挖文档。
   - `sva_assemble_report.py`：拼装主报告。它把 `{{T..}}` 替换成生成的表格，把 `{{NAME}}` 替换成 `report_vars.json` 或 `report_var_<NAME>.md` 的内容；只要还有占位符没被替换就报错。
   - `sva_check_quotes.py 文档路径…`：检查报告里引用的例子是否都是真实数据行（需要传入文档路径）。
   - `sva_apply_qw_swap.py`、`sva_apply_verifier_fixes.py`：留档用的一次性脚本，前者把带搜索前1万 qwen 标签的结果换成正式版本，后者按独立复核的 54 条意见修改报告草稿。每处替换都先断言锚点恰好出现一次，否则整批不写入。

## 其他文件

- **领域深挖**：`sva_dom_{fin,med,edu,film,ppl}_*.py` 为五位分析员的脚本，中间表在 `work/sva_dom_*`。
- **审计记录**：`work/audit_clean_*`（第一轮）、`work/audit2_*`（第二轮）。
- **外部数据核查记录**：`work/sva_web_research.md`，每条都标注了核实状态。
