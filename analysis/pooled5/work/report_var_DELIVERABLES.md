| 交付物 | 位置 | 内容 |
|---|---|---|
| 逐行标注表（五份） | `analysis/pooled5/deliverables/<领域>_POOLED5_逐行标注.xlsx` | 每行带 `来源/source/surface/domain/l2/pv_raw/pv_norm/tier` 与两套标签（td_* 自上而下意图、bu_* 聚类），外加「被清洗剔除」「条数/行占比/流量占比 × 来源」「意图体系定义」「裁决规则」「聚类叶定义」「风控图层分布」等页 |
| 同一份数据的 parquet | `analysis/pooled5/deliverables/<领域>_pooled5_labeled.parquet` | 供程序读取，列与上表一致 |
| 分领域深挖 | `docs/POOLED5_2026_领域深挖.zh.md` | 五份领域报告全文，每份文末带「修订记录」 |
| 图 | `docs/img/pooled5/`（9 张 PNG） | 四张主图 fig1 语料构成、fig3 对比距离、fig4 语义近邻、fig5 写法标记，加 fig2 的五张分领域意图占比热力图（每个领域一张）；fig1 / fig3 / fig4 已分别内嵌在 §2 / §4 / §6 |
| 挖掘运行产物 | `runs/{fin,med,edu,film}-pool5/gen01`、`runs/ppl-pool5b/gen01` | 意图体系定义、聚类树定义、逐行标签、分层对比文档、风控筛查、各阶段图与门禁 |
| 分析脚本与中间表 | `analysis/pooled5/`（`work/` 下为中间产物） | 见附录 A |
