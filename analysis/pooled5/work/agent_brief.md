# POOLED-5 领域分析员 · 任务书

你分析 **一个领域**。本领域的五个来源已经合并成一份语料，由 QMine 挖掘程序**一次运行**完成标注，
所以五个来源共享同一套意图体系（td_*）和同一棵聚类树（bu_*）。分开跑的两次运行不可比（实测同样
10,000 行的两次金融运行共享 0/35 个类目编码），这就是为什么要合并跑。

## 来源（source 列）

| source | 含义 | 行数级别 | 权重 |
|---|---|---|---|
| 2025search | 2025-07-01 搜索前 1 万 | ~10,000 | wise_pv |
| 2026search | 2026-07-01 搜索前 1 万 | ~10,000 | wise_pv |
| assistant_top | AI 助手该类目 Top1000（头部） | ~950 | search_num |
| assistant_random | AI 助手该类目随机 1000（长尾） | ~980 | search_num |
| assistant_voice | AI 助手语音 query 1000（仅金融、医疗） | 1,000 | 无 PV，均匀 |

四条对比线，每条只差一件事：
1. **时间**：2025search → 2026search（同一界面、同一抽样、相隔一年）
2. **界面**：2026search → assistant_top（同为各自的头部流量）
3. **深度**：assistant_top → assistant_random（同一界面、头部→长尾）
4. **输入方式**：assistant_top/random → assistant_voice（同一界面、打字→语音）
另有 2026search → assistant_random，界面与深度同时变化，**不能单独归因**。

## 你要读的文件（都在 `analysis/pooled5/`）

- `work/<领域>/digest.txt` —— 数字底稿，先读这个
- `work/<领域>/*.csv` —— shares_* / diffs_* / contrast_summary_* / form.csv / overlap.csv /
  fit.csv / concentration_* / examples_* / class_definitions.csv / risk_by_source.csv /
  taxonomy_delta_*.csv / cleaning_removed.csv
- `deliverables/<领域>_pooled5_labeled.parquet` —— 逐行数据，自己复算用
- `work/semantic_nn.parquet`、`work/wrap_pairs.csv`、`work/voice_profile.txt`（金融/医疗）
- 运行产物：`runs/<key>-pool5/gen01/`（taxonomy_v2.json、tree_naming.json、
  分层对比_头尾结构差异.md 或 快照对比_漂移分析.md、metrics_panel.json、granularity.json）

## 硬性规则

1. **一切占比都是「来源内占比」**。各来源行数差 10 倍，跨来源的原始条数没有意义。
2. **区间不含 0 才算差异**；diffs_*.csv 里的 `sig` 列已经算好 Newcombe 95% 区间。
   低于 4 个百分点的差异即使显著也只作方向性描述。
3. **助手头部的 PV 由清洗决定**（影视头部 78% 的 PV 是功能入口），所以 PV 加权结论只作补充，
   主结论一律用行占比。语音没有 PV，它的流量占比等于行占比，不要拿它跟别的来源比流量。
4. **两份数据相隔的时间**：搜索快照 7 月初，助手约 8 月下旬，语音时间未知。热点、热播、行情相关的
   差异不能归因于界面。语音的抽样方式（头部还是随机）导出方没有说明，必须写进局限。
5. **语音是 ASR 文本**：几乎没有标点、英文小写、约 40 字处疑似截断（见 voice_profile.txt）。
   凡是跟长度、标点、完整度有关的差异，先归因到仪器，再谈用户。
6. **体系是在 87% 搜索行上拟合的**（fit.csv）。某个来源大量落在低置信度/模糊行，本身就是结论。
7. **例子必须是数据里的真实行**，可以截断但不能改写。涉及普通个人信息、色情内容，或涉及未成年人的
   隐私与性内容，只作概括描述，不引用原文。
8. **不要把头尾差异写成界面差异**，也不要把查询重合写成用户重合（没有 user id）。
9. 每一个数字都要能追到一张表或一行代码。你写的每条结论后面附 `(表名: 字段=值)`。

## 你的产出

写一份 Markdown 到 `analysis/pooled5/work/dom_<key>.md`（key: fin/med/edu/film/ppl），结构固定：

1. **这个领域挖出了什么体系**（L1 类目数、每类定义一句话、哪些类是助手行带来的
   —— 用 taxonomy_delta_newclass_mix.csv 的 assistant_share）
2. **加入助手行以后，同样 2 万条搜索行的划分变了多少**（AMI/ARI，以及哪些老类被拆开或合并）
3. **四条对比线**，每条一小节：整体距离（TVD）、显著变化的类目（含区间）、3–6 个真实例子
4. **形态与可表达性**：form.csv 的关键标记、semantic_nn 的近邻分档（注意「完全相同」是余弦 ≥0.999，
   不等于字符串相同，字符串重合看 overlap.csv）、包装对
5. **谁在问、为什么这样问**（画像推断，每条都要写明依据和什么证据能推翻它）
6. **风险与合规**（risk_by_source.csv + 你在长尾里实际看到的行）
7. **对产品的启示**（3–5 条，每条对应上面的一个测量）
8. **本领域的局限**（数据能说什么、不能说什么）

同时把你引用的关键数字写成 `work/dom_<key>_claims.csv`，列：claim, value, source_table, how_computed。
这张表会被独立复核，凡是复核不通过的结论都会被删掉，所以宁可少写、写准。

## 追加：两条已经查实的口径约束（复核员发现，所有领域适用）

1. **同一个字符串在一次运行里必然拿到同一个标签。** 实测：医疗 13,184 行重复串、教育 13,621、金融 11,755、影视 6,030，
   同串拿到两个 td 标签或两个聚类叶的**一个也没有**。所以：
   - 「两年共有的串标签完全一致」**不是测量结果，是标注方式的构造必然**，不能当作体系稳定的证据；
   - 2025→2026 的意图差异**只可能来自榜单换血**（哪些串进/出前 1 万），本方法测不到「同一句话含义变了」；
   - 跨来源逐字重合的那部分行对任何意图差异的贡献为 0，差异全部来自不重合的串。
   这一条必须写进局限，也必须约束正文措辞。
2. **文本 join 会把来源串错。** 同一字符串常常同时出现在 2025/2026/助手多个来源，按 query 文本 join 会一对多放大。
   一律用逐行 parquet 的 (query, source) 或 row_id 定位，不要用 merge on query。
