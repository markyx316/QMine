# analysis/pooled5 —— 2026 五域同体系对比（复现包）

本目录复现两份报告：

- 主报告：[`docs/POOLED5_2026_五域同体系对比.zh.md`](../../docs/POOLED5_2026_五域同体系对比.zh.md)
- 分领域深挖：[`docs/POOLED5_2026_领域深挖.zh.md`](../../docs/POOLED5_2026_领域深挖.zh.md)

这里全是**跑完之后的分析脚本**，不是 QMine 流水线的一部分：不修改 `src/`，测试也不收集这个目录；
脚本读的是已完成运行的产物（`runs/*-pool5`、`runs/ppl-pool5b`）与 `data/raw/`。

## 这次研究在做什么

把每个领域的**五个来源**合并成一份语料，由挖掘程序**一次跑完**，于是五个来源共享同一套意图体系与同一棵聚类树：

| source | 含义 | 行数级别 |
|---|---|---|
| `2025search` | 2025-07-01 搜索该垂类前 1 万 | ~10,000 |
| `2026search` | 2026-07-01 搜索该垂类前 1 万 | ~10,000 |
| `assistant_top` | AI 助手该类目 Top1000 | ~950 |
| `assistant_random` | AI 助手该类目随机 1000 | ~980 |
| `assistant_voice` | AI 助手语音 query 1000（仅金融、医疗） | 1,000 |

分开跑两次不可比：同一份 10,000 行金融语料跑两次（`fin02`/`fin03`）**共享 0 个类目编码**。

## 批次（cohort）——跨领域产物必须说出自己是哪一批的

2026-09-13 晚接入 **书籍文档** 与 **软件** 两个垂类以后，`DOMAINS` 有 7 个，而已交付的
《POOLED5_2026_五域同体系对比》和它引用的 `work/cross/*.csv` 全部是**五域**的。如果跨领域脚本
直接扫「所有跑完的领域」，任何一次重跑都会把那份已交付报告底下的表换成七域的——数字全变而正文不变。
上一轮已经踩过同类的坑（`p5_vs_unified.py 人物` 用单域结果覆盖了跨域表，T10 带着 1 个领域发了出去）。

所以脚本一律按批次取域：

| 批次 | 领域 | cross 目录 |
|---|---|---|
| `pool5`（默认） | 金融 / 医疗 / 教育 / 影视 / 人物 | `work/cross/` |
| `new2` | 书籍文档 / 软件 | `work/cross_new2/` |
| `all7` | 上面全部 | `work/cross_all7/` |

```bash
HF_HOME=$(pwd)/.hf .venv/bin/python analysis/pooled5/p5_tables.py              # 默认 pool5，行为与加新域之前完全一致
P5_COHORT=new2 HF_HOME=$(pwd)/.hf .venv/bin/python analysis/pooled5/p5_tables.py   # 只做新接入的两个域
```

不设 `P5_COHORT` 时，所有脚本的行为与加这两个域之前**逐字节相同**（已验证：
`route_agreement_by_source.csv` / `concentration_by_source.csv` / `residue_breakdown.csv`
重跑后与改动前 `cmp` 一致）。

## 目录

| 文件 | 作用 |
|---|---|
| `build_pooled5_corpus.py` | 合并五个来源、套用 v3 清洗分层、按来源归一权重，输出 `data/raw/pooled5/*.parquet` |
| `pooled5_common.py` | 载入（带断言的位置对齐）、Wilson / Newcombe / Cramér's V / TVD、形态正则、`RUN_ID` 映射 |
| `p5_tables.py` | 逐域标准表：占比、差异、对比距离、形态、重合、适配度、集中度、风控、例子 |
| `p5_digest.py` | 逐域「数字底稿」，分析员先读它 |
| `p5_deliverables.py` | 逐行交付表（xlsx + parquet），含来源与被清洗剔除的行 |
| `p5_postprocess_run_xlsx.py` | 给**运行自己交付的** xlsx 加来源列，写进 `runs/<id>/gen01/postprocessed/`，原件不动 |
| `p5_semantic_nn.py` | 每一行在 2026 搜索里的最近邻（bge-base-zh），2025 搜索为时间对照 |
| `p5_wrap.py` | 包装对：搜索词被原样写进更长的助手查询 |
| `p5_turnover.py` | 年度差异的分解：留存 / 掉榜 / 新进；以及两界面共有的字符串 |
| `p5_depth_control.py` | 把搜索截到自己的前 1,000 行，检验界面差异是不是深度造成的 |
| `p5_residue.py` | 残余类拆解：看不懂 vs 体系没有位置 |
| `p5_route_agreement.py` | 聚类树与意图体系在各来源上的一致度 |
| `p5_concentration.py` | 各来源内部的流量集中度 |
| `p5_voice.py` | 语音导出的仪器特征（无标点、截断、重复串） |
| `p5_taxonomy_delta.py` | 同样 2 万条搜索行在「只有搜索」与「搜索+助手」两次运行里的划分差异，**带同数据两次运行的噪声基线** |
| `p5_vs_unified.py` | 领域体系 vs 上一版 13 类通用框架（同样的行） |
| `p5_tvd_ci.py` | 每个 TVD 的自助法区间与同源噪声上界 |
| `p5_cross.py` | 跨领域汇总到 `work/cross/` |
| `p5_report_tables.py` | 报告里的全部表格（T1–T15） |
| `p5_figs.py` | 图 1–5 → `docs/img/pooled5/` |
| `p5_assemble_report.py` / `p5_assemble_domains.py` | 拼装主报告与领域深挖 |
| `p5_check_quotes.py` | 检查**主报告**里引用的每个 query 是不是真实行（主报告约定「」只用于 query；领域深挖由各自的复核员逐条按 (query, source) 查证，不适用这个机械检查） |
| `run_all.sh` / `watch_runs.sh` / `watch_ppl.sh` | 五次挖掘运行的调度与监控 |
| `p5_snapshot_classes.py` | **逐类 × 逐快照**的全部表：占比与 Wilson 区间、均衡归属、缺席的可检出性判定、全部快照对的 Newcombe 差异与 TVD/秩相关、意图↔叶的条件分布、逐类例子、**每个快照前 1/3/5/10 名的行与流量覆盖**（`topn_coverage.csv`）|
| `p5_snapshot_figs.py` | 逐领域六张图（意图/叶的占比热图与均衡指数热图、快照两两距离、界面散点）→ `runs/<id>/gen01/postprocessed/img/` |
| `p5_snapshot_report.py` | 把上面的表拼成逐领域报告 → `runs/<id>/gen01/postprocessed/*_意图与聚类叶_跨快照对比.zh.md`；叙述从 `work/<领域>/snapshot_classes/narrative.md` 填入占位 |
| `p5_snapshot_verify.py` | 机械复核叙述：每个数字回查到表、每条引文回查到真实且可引的行 |
| `run_new2.sh` / `run_new2_analysis.sh` | `new2` 批次的挖掘调度与全套后处理 |
| `p5_new2_risk_supplement.py` | 风控筛查的 then-vs-now：运行当时用的档案 vs 补进 agent 发现之后的档案 |

运行顺序见主报告附录 A。中间产物在 `work/`，跨领域表在 `work/cross/`。

## 第二份交付：逐领域「意图 / 聚类叶 × 快照」对照

上面两份报告回答「两个界面整体差多远」。另有一份**逐领域**的对照报告，回答**每一个类目在每一个
快照里各占多少、哪些类只在某个快照出现**，交付在各运行自己的 `postprocessed/` 目录里（原始产物不动）：

```bash
HF_HOME=$(pwd)/.hf .venv/bin/python analysis/pooled5/p5_snapshot_classes.py   # 表 + 工作簿
HF_HOME=$(pwd)/.hf .venv/bin/python analysis/pooled5/p5_snapshot_figs.py      # 图
HF_HOME=$(pwd)/.hf .venv/bin/python analysis/pooled5/p5_snapshot_report.py    # 报告（会自动填叙述）
HF_HOME=$(pwd)/.hf .venv/bin/python analysis/pooled5/p5_snapshot_verify.py    # 叙述的机械复核
```

产物（每个领域一套）：

| 文件 | 内容 |
|---|---|
| `runs/<id>/gen01/postprocessed/<stem>_意图与聚类叶_跨快照对比.zh.md` | 报告正文：四个层级的逐类 × 逐快照表 + 逐意图/逐叶卡片 + 六张图 |
| `runs/<id>/gen01/postprocessed/<stem>_意图与聚类叶_跨快照对比.xlsx` | 同样的表的全量版（含 L2/家族层、逐对差异全量、例子） |
| `runs/<id>/gen01/postprocessed/img/*.png` | 六张图 |
| `analysis/pooled5/work/<领域>/snapshot_classes/*.csv` | 全部中间表 + `narrative.md`（分析员写的叙述）+ `verify.md` |

三条这份报告特有的口径：

1. **「只在某快照出现」必须配可检出性。** 助手快照 n≈1,000，0 条的单侧 97.5% 上界仍有约 0.4%；
   同样 0 条落在 10,000 行的搜索里上界只有 0.04%。所以「只出现在搜索里」远比「只出现在助手里」
   容易达成。`absence_*.csv` / `interface_*.csv` 的 `缺席判定` 列已经把这件事算好，正文只引判定。
2. **均衡归属 / 均衡指数** 把「搜索行数是助手十倍」除掉：归属% = 该快照内占比 ÷ 各快照内占比之和；
   均衡指数 = 该快照内占比 ÷ 各快照占比的未加权均值。`指数` 一列则与 `class_profile.csv` 同义
   （分母是全语料占比，里面 87% 是搜索行），两者都给，不要混用。
3. **叙述里的每个数字都要能回查到表。** `p5_snapshot_verify.py` 逐个回查，未匹配数字与问题引文
   都必须是 0；它只拦「凭空写出来的数字」和「编出来的例子」，拦不住「把 A 类的数安到 B 类头上」，
   后者靠独立复核员逐条重算。
4. **内层 TVD 也要配同源噪声上界。** `intent_leafmix.csv` / `leaf_intentmix.csv` 的格子 n 只有几十到
   几百，两份来自同一分布的样本本来就能给出 0.2–0.3 的 TVD。上界 = 两侧行合并后按原比例随机对半切
   300 次取 95 分位；`超出噪声=False` 的格子既不能读成「内部换了」，**也不能读成「内部没换」**。
   实测各域只有 37%–94% 的格子超出噪声。
5. **每一处重抽都按 (用途, 领域, 层级, 快照对) 各自播种。** 原来是一条模块级随机流，于是任何一处
   新增的抽样都会把后面每个领域的自助法区间挪动一点——点估计不变、区间变，写进正文的区间端点下一次
   重跑就对不上。现在单独重跑一个领域与在五域批量里跑，结果逐位相同（已验证）。
6. **`p5_snapshot_verify.py` 的数字池包含 `summary.json` 与三层护栏的逐层条数。** 漏掉这一层的后果
   实测过：正确的 `quotable_rows=21763` 会被判未匹配，而写错的 `21,800` 恰好能在别的表里匹配上——
   机械检查把对的拦下、把错的放过，比没有检查更坏。
7. **一个量只要显示在两个地方，就必须从同一个未舍入的值格式化。** `topn()` 一度先 round 到 3 位、
   表 1 与表 2.2 再各自 round 到 2 位，同一个覆盖率于是印成 98.24 与 98.23（前5 印成 69.43 与 69.44），
   两边各自看都「对」。现在 `topn()` 存未舍入值，表 1、表 2.2、工作簿两页全部从它格式化；
   未舍入的值在 `topn_coverage.csv` 里。改完以后 `前10 + 前10之外` 才逐格等于 100.00。
8. **前 N 名覆盖是「每个快照各排各的」，而且成员只能有一条选取路径。** 同一名次在两列里通常不是
   同一个类，只能竖着读。`前10流量占比%` 是**上面那十个类**（按行占比选出）按 `pv_norm` 加权，
   **不是**按流量重排以后的前十；语音没有 PV、按均匀权重，所以语音那一格的「流量−行」恒为 0.0，
   可当自检用。**十个类由 `topn_members.csv` 定下来，报告只负责显示**——原来 `topn()` 用
   `value_counts()`、报告另用 `sort_values()`，两条次序在并列格上选到不同的类，教育 助手随机1k
   叶层的流量合计因此印成 73.98%，而表里显示的十个类对应 79.23%，差 5.25pp。
9. **并列时，行合计免疫、流量合计不免疫——而且要印区间，不能印 ±。** 并列的类条数相同
   （所以 `前3/前5/前10合计`、`前10之外（行）` 是定值），`pv_norm` 不同（所以流量合计取决于挑哪一个）。
   **点值几乎总落在区间的一端**（实测 10 格里 9 格），所以 `44.42% ±0.64pp` 会被读成 43.78–45.06，
   而真区间是 43.78–44.42——上半截是凭空的。一律印 `[低–高]`，并加一条恒等式：**点值必须落在
   它自己印的区间内**。实测 120 格里 10 格有摆动，全部在助手侧；搜索侧一万行从不并列。
   语音快照 PV 均匀，并列的类 `pv_norm` 必然相等 ⇒ 摆动恒为 0，所以提示语要按**实际摆动**
   分支，不能按「有没有并列」分支（否则会印出「可在 66.70%–66.70% 之间摆动」）。
9b. **流量列比它看起来松得多，三条必须随表走。**（a）`pv_norm` 按来源各自归一到 10,000，只能竖着读；
   （b）有效样本量是 `1/Σw²` 而不是行数——金融 2026搜索 9,999 行、有效 111.5，单行最高占 7.39%，
   所以流量是裸点估计，报告不给它区间就必须明说；（c）`助手头部1k` 是按 PV 取前 1,000 条**再过清洗层**，
   被清洗掉的行带走了该导出 42%–98% 的原始 PV（`work/build_audit.csv: pv_dropped_%`，软件 98.47%），
   所以那一格的流量主要是清洗规则的函数。影视的叙述早就据此「全程只用行占比」。
10. **表 2.2 的第N格由 `条数 ÷ n` 直接算，不读 `matrix_*.csv` 的 3 位值。** 走那条路会把
   12.4549% 印成 12.46，而表 1 的同一个量印 12.45。`matrix_*.csv` 仍存 3 位（叙述引的是 3 位值），
   只是表 2.2 不再经过它。
11. **改了护栏就必须重跑 `p5_snapshot_classes.py`，只重渲染报告不够。** 人物报告曾把
   `可引行 20,542` 一直带着走，实际是 20,541——上一次只重渲染、没重跑 classes，报告读的是旧的
   `summary.json`。`p5_snapshot_verify.py` **抓不到这一类**：报告与数池读的是同一个陈旧文件。

## 第三份交付：2026-09-13 新接入的两个垂类（`new2` 批次）

书籍文档 / 软件，各 4 个快照（**无语音导出**）。与五域同一条流水线、同一套后处理，
但跨领域产物全部写进 `work/cross_new2/`，五域的 `work/cross/` 一个字节都没动。

```bash
HF_HOME=$(pwd)/.hf .venv/bin/python analysis/pooled5/build_pooled5_corpus.py   # 七个域一起建
./analysis/pooled5/run_new2.sh            # 两次挖掘（fast），约 65 分钟
./analysis/pooled5/run_new2_analysis.sh   # 全套后处理，P5_COHORT=new2
```

| 领域 | 运行 | L1 | L2 | 交付叶 | 交付家族 | 挖掘行 |
|---|---|---|---|---|---|---|
| 书籍文档 | `book-pool5/gen01` | 20 | 59 | 30 | 21 | 21,786 |
| 软件 | `soft-pool5/gen01` | 25 | 58 | 39 | 26 | 21,920 |

**这一批特有的四条口径：**

1. **类目名与内容对不上。** 书籍文档不是书：图像类约 32% 的行、漫画 12.85%、小说进不了前十。
   软件主要是导航：仅产品名消歧 26.73% + 下载安装 17.53%。
2. **助手头部的 PV 几乎全是产品功能位。** v3 清洗剔除后带走该导出
   **88.45%**（书籍文档，首行 `总结全文概要`）与 **98.47%**（软件，整排图像编辑功能位）的 PV。
   这两个域的助手侧**一律用行占比**，PV 只作补充。
3. **引用护栏是四层**（比五域多一层）：第四层**按运行自己的分类**整类不引，只用于性与自伤类目。
   起因是软件 `生成露骨性图像编辑` 51 行里正则只拦得住 12 行——绕过靠的是换说法不是换字符。
4. **三项分析略去，且是带着覆盖率数字略去的**：`p5_taxonomy_delta`（没有纯搜索对照运行）、
   `p5_residue` 与 `p5_vs_unified`（13 类通用框架在本批覆盖 0.5% / 0.1%，五域皆 100%）。
   `p5_residue.py` 会带着实测覆盖率拒绝运行，不会吐出一张全是噪声的表。

**风控数字有两个，别混用**：`runs/<id>/gen01/risk_screen.json` 是那一跑真正用过的档案算的；
档案在跑完之后补进了 agent 找到的类，所以现在更全。两者之差用
`p5_new2_risk_supplement.py` 算（书籍文档 1,096 → 1,240，软件 1,150 → 1,311）。

## 几条不能忘的口径

1. **一切占比都是来源内占比。** 各来源行数差 10 倍，跨来源原始条数无意义。
2. **同一字符串在一次运行内必然同标签**（实测五域零例外），所以年度差异只能来自榜单换血，
   本方法测不到「同一句话含义变了」。
3. **AMI 的参照系是 0.774 / 0.687**（同一份数据跑两次），不是 1.0。
4. **语义近邻的「完全相同」是余弦 ≥0.999**，不等于字符串相同；逐字重合看 overlap。
5. **风控命中率各领域不可比**（正则由各自的 domain profile 定义）。
6. **语音是 ASR 文本**：无标点、约 40 字截断、抽样方式未知；长度与完整度相关的差异先归因到仪器。
7. **人物领域用 `ppl-pool5b`**，理由与「先定后看」的选用规则见 `work/people_run_choice.md`。

## 第四份交付：金融8（8 快照金融，2026-09-14）

同一个金融垂类，补来三份导出（两年的搜索随机 1 万、语音头部 1 千）以后重跑成 8 个快照。**不覆盖**
`金融_pooled5.parquet`（它与 fin-pool5 的 labels_full.csv 逐行绑定），自有语料、运行、cohort。

```bash
HF_HOME=$(pwd)/.hf .venv/bin/python analysis/pooled5/build_fin8_corpus.py     # 断言五个共有快照与交付语料逐格相同
./analysis/pooled5/run_fin8.sh                                                  # fin-pool8，--domain finance_zh_v2
export P5_COHORT=fin8
for s in p5_postprocess_run_xlsx p5_deliverables p5_snapshot_classes p5_snapshot_figs p5_snapshot_report p5_snapshot_verify; do
  HF_HOME=$(pwd)/.hf .venv/bin/python analysis/pooled5/$s.py; done
```

- 源 xlsx 把纯数字 query 存成数字，证券代码前导零在进任何代码前就没了；构建时补回 839 行（证据链在构建脚本说明里）。
- 8 个快照是 28 对，`p5_snapshot_figs.py` 在 ≥15 对时把 TVD 图改画下三角矩阵；≤10 对仍走柱状图（逐字节不变）。

## 第五份交付：健康（同一周的健康搜索周榜 + 专科健康 AI 管家周榜，2026-09-14）

```bash
HF_HOME=$(pwd)/.hf .venv/bin/python analysis/pooled5/build_health_corpus.py   # 按 query 合并、清洗、盲标决定层级
./analysis/pooled5/run_health.sh                                                # health-pool2，--domain health_zh，--reference-columns legacy_l2,legacy_type
export P5_COHORT=health2
# 同上一组后处理脚本
```

两份导出都是「query × 天」前 1 万行，按 query 合并成周粒度；AI 管家的作答选项、功能按钮、推送问题、包装模板、
医生卡由三视角盲标（κ 0.945）一致时决定、分票时由规则决定，只留 225 串入挖掘。口径全文：`work/health_clean_audit.md`。

## 这一轮新增的规则（每一条都是实测踩过的）

12. **界面按每行自己的 `surface` 分，不按 source 名单分。** 名单写死成旧五个 source，金融8 的两份随机搜索与语音头部被
    表 *-E、「仅搜索/仅助手」与界面散点图**静默排除**（搜索n 19,997、助手n 2,937，应为 39,997 / 3,937）。
    `surface_groups()` 在任何快照没有、或有两个 surface 值时直接报错。
13. **只对某个批次成立的措辞必须逐域给。** 报告里「本域是 2026-09-13 新接入的」「两个搜索快照都在 7 月初」「搜索侧那两万行」、
    交付物里「source 的五个取值」，印在金融8/健康上都是错的。现在 `DOMAIN_INTRO` / `DOMAIN_TIME` / `DOMAIN_BUILD`、
    `DOMAIN_SRC_COLS` / `DOMAIN_NOTE` 只对之后接入的语料生效，**已交付七域的文字逐字节不变**（每次改动后都做 SHA-256 比对）。
14. **条件表达式挂在隐式拼接后面，会吞掉整段。** `("…①" "…②" f"③{x}" if x else "。")` 在 x 为空时把①②③一起换成「。」——
    金融8 报告因此一个字的流量警示都没有。
15. **带 `一-鿿` 这类转义的正则不要交给 `Series.str.contains`。** parquet 读进来是 pyarrow 字符串，pandas 走 RE2，
    RE2 不认这种转义，直接抛 `ArrowInvalid`。用 Python `re` 逐行求值。
16. **参考列先写条件、再看结果。** 健康的平台分类在最终语料出来之前写下条件（与界面 V ≤ 0.55、无占比 >1% 的单边类），
    结果 `legacy_dept` 不满足、被撤下；`reference_columns_check()` 每次重测并断言与运行声明一致。
