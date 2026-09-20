# QMine — 交接说明 / Handoff

> **How to maintain this file** (the contract `CLAUDE.md` refers to):
> - **§1 Status** — overwrite every session. It must always describe *now*.
> - **§2 Open questions** — **edit, never append.** Resolve an item → delete it here and
>   record the resolution in that session's section below.
> - **§3 Durable notes** — things worth not re-learning. Rarely changes.
> - **§4+ Session log** — append a new dated section per session. Never edit old ones.
>
> This file is a **log, not a specification.** When it disagrees with the code or the
> tests, the code and tests are right. Verify before relying on anything here.

---

## 1. Status — last updated 2026-09-20

> **2026-09-20：后处理与语料准备成为程序的功能，另加一个对话入口。全量 823 通过（747 + 76）。**
> 三个新包，全部是新增 + 三处必要接线，已有测试一个没改：
> `src/qmine/pooled/`（跨快照对比，阶段 `p10c` + `qmine compare`）、
> `src/qmine/prepare/`（`qmine prepare`、`run --prepare`）、`src/qmine/chat/`（`qmine chat`）。
> **`analysis/pooled5/` 与所有已交付的运行、产物一个字节没动**，那批脚本仍是各自那次交付的证据。
> 对着 health-pool3 逐项复算：137 个类的条数与占比逐个相同，TVD / 秩相关 / Cramér's V 四位小数一致；
> 自助法区间端点差 ≤0.0016（种子按「测什么」派生，标签串不同，结论不变）。
> 途中在 `src/` 里修了七个真缺陷（property 当方法调、护栏拼 alternation 失效为放行、表与卡片各自舍入、
> CSV 往返的 `nan` 印进报告、整类不引用单个汉字命中而误拦 28.6% 的语料、无效的 ArtifactKind 被吞掉、
> 重渲染的代次认领它没有的文件），加上对抗式复核提出、逐条反驳后仍成立的 15 条，全部修掉；
> 每一条都带回归测试，55 个变异体逐一验过全被杀。
> 另按用户「绝不引具名医生」的长期规矩，把具名医生放进第二层通用硬规则（`configs/` 最终没动）。
> 详见文末当日 session。

> **2026-09-16：人物8 / 影视8 / 医疗随机 三个域跑完并交付，途中在 `src/` 里修了三个真缺陷。**
> 三份语料：`人物8_pooled5.parquet` 43,802 行、`影视8_pooled5.parquet` 43,933 行（各 8 个快照，新增语音随机 1k），
> `医疗随机_pooled5.parquet` 20,316 行（传统搜索随机 1w + 健康管家随机 1w，都是 2026-09-14 单日导出）。
> 新档案 `configs/domains/{people_zh_v2,film_tv_zh_v2}.yaml`（19 / 21 个种子，覆盖 27.95% / 32.68%）。
> 三次运行都是 fast、routed、未 halt；人物域把 Zhipu 的角色全部改路由到 Kimi，全程 0 次内容过滤。
> **交付运行：人物8 用 `runs/ppl-pool8b/gen01`（全新单次跑完，无 resume），影视8 / 医疗随机 用各自的 gen01。**
> `ppl-pool8` 整个 run id 作废：gen01 有幻影类，gen02 体系漂了，gen03 是两次 resume 的产物——
> 只有 14 个闸门（少了 `p2a_pilot_agreement` / `p2a_taxonomy_shape`，因为 `--reuse-taxonomy` 跳过了 p2a）、
> 6 个 decision、`elapsed_s` 只记了 resume 之后那一段。ppl-pool8b 这三项分别是 16 / 7 / 5,318 秒。
> 交付形状：人物 21 L1 / 53 L2 / 37 族 / 37 叶；影视 17 / 54 / 52 / 58；医疗随机 20 / 51 / 33 / 33。
> 三份报告 `未匹配数字 0 · 问题引文 0`。按用户要求，人物 / 影视**不跑叙述工作流**（报告的散文段留空，表与图完整），
> 且例子表**印真实 query**。
>
> **三个源码缺陷（都带回归测试，全量 747 通过、`ruff --select F` clean）：**
> 1. `graph/nodes/topdown.py::_active_learning_round` 没有 round 1 的两道保护。标注员漏标的 22 行以
>    `final="UNLABELED"` 进了金标，22 行过了 5 折支撑下限，于是 18 类的体系训出 19 类分类器，幻影类落到 11 行语料、
>    进了 5 份交付文档。修好后 macro-F1 0.526 → 0.557。只有 ppl-pool8 中招（其余 7 次运行实测 0 行）。
> 2. 同一文件 `_require_both_branches` 给 `Deps.gate()` 传了不存在的 `blocking=True`。这是**只有出错时才走的分支**，
>    它的测试用 `**kw` 的假 deps，所以一直是绿的；gen03 真的在汇合点撞上缺分支时，运行死在 TypeError 而不是那道闸门。
>    测试的假对象现在绑定真签名。
> 3. `config.py` 的 fast 校验器只在开关「本来是开的」时才往 `fast_skipped` 里追加，于是每一次 `--resume` 重建出的清单
>    只有 4 项。横幅是由这份清单生成的，gen03 的三份参考文档因此声称双标注、观察员、对抗验证、交付前审核都**跑过**。
>    改为按 mode 推导，并断言重复校验幂等。
>
> **后处理侧（不动 `src/`）：** `pooled5_common.run_dir` 新增 `P5_GEN_<批次>` 代次覆盖；
> `p5_postprocess_run_xlsx` 增 `DOMAIN_SRC_COLS["医疗随机"]`；`p5_snapshot_classes` 的产品层按语料时间口径改列名
> （单日语料用 `当日PV`），并把 `EXTRA_QUOTE_BLOCK["医疗随机"]` 指到 医疗8 那条实测正则、`SCREENED_QUOTE_BLOCK`
> 扩到四个域；`p5_snapshot_report` 增 `PREP_TEXT` 按域分流数据准备一节。
> 引文护栏用 `p5_quote_hardrule_scan.py`（三条硬规则 + 医疗域加具名医生一条）复核：人物 9 命中 / 影视 9 / 医疗随机 76，
> **三个域都是「已印进交付文档 0」**；命中串仍写进各域的 `privacy_screen/quote_block.json` 作为换例子时的保险。
> 887 份既有交付文件哈希不变（唯一变的是按批次重写的 `work/snapshot_classes_all.json`，只写不读）。


> **2026-09-15（晚）：医疗8（med-pool8）跑完并交付，纯后处理，`src/` 未动。**
> fast、routed、未 halt、229 次调用、1 小时 53 分；`verify_run` 本运行 PASS 21 / FAIL 0（对照 ppl-pool5 PASS 9 / FAIL 3）。
> 交付 17 L1 / 55 L2 / 41 叶 / 37 族，留出复现 0.9704。报告 `runs/med-pool8/gen01/postprocessed/medical_zh_v2_意图与聚类叶_跨快照对比.zh.md`
> （叙述经两名独立复核员三轮对抗复核，「未匹配数字 0 · 问题引文 0」）。**本次最重要的发现是引文护栏的漏洞**：前五层正则 / 名单护栏
> 在报告与工作簿的例子里漏了 38 个违规串；改为逐串阅读印出来的串（Claude 必读、DeepSeek 并集，轮次推进到收敛，再做独立第二遍），
> 不可引名单 634 串，印出的 2,080 个语料串全部被 Claude 读过。测试 744 通过；已交付的 782 个文件除一个只写不读的汇总外哈希不变。

> **2026-09-15（傍晚）：fin8 深挖报告已交付，纯后处理，`src/` 未动。**
> `runs/fin-pool8/gen01/postprocessed/finance_zh_v2_意图内部结构与代表性样例.zh.md` + `.xlsx` + `img/意图结构_*.png`。
> 回答三件事：主报告卡片例子（流量前 3）覆盖中位只有 32.8%，构成显式取例到 80.5%；同一意图跨快照的差别拆成
> 「关于什么」（叶构成）与「要什么」（三个宽意图的盲标子功能，DeepSeek 跨模型 κ 0.963–0.986）；叶 × 子功能交叉回答
> 「核实还是决策」（裁决意图 68.7% 是核实规则套到自己身上，直接要建议 3.8%）。叶优于家族（实测）。med-pool8 仍在跑。

> **2026-09-15：修了一个披露层的源码缺陷（不改 K，不改任何数字）。**
> `ops/cluster.py: reference_sensitivity(sweep, k, *, locator_column)` 现在只把**真正定位 K 的那一列**标为
> `decides`（稳定性兜底时谁都不标）；`p5_k_references_agree` 闸门的 `observed.deciding_reference` 不再写死
> `phrasing_groups`。声明的参考列定位 K 的运行（ai04、aiwire01、health-pool2、三个 k12_zh、live41 gen01/gen03、
> live42、live44 —— 69 份 granularity.json 里 10 份）同一产物曾自相矛盾。实测：24 份措辞群定位的已存 sweep 用新旧代码
> 重算**逐字节相同**，10 份受影响的只差两个 decides 与注记末句。新增 3 个测试（在原代码镜像里全部失败，5 个变异体各至少
> 被一个抓到）；全量 **744 通过、exit 0**，`ruff --select F src/qmine/ tools/` clean。已存产物不重写（`qmine render`
> 修不了，只有新 generation 重跑 p5 才会带上修复）。同类的姊妹缺陷记在 §2 第 20、21 条。详见文末当日 session。

> **2026-09-13（深夜）：新增两个垂类 书籍文档 / 软件，全套跑完并交付。**
> 语料 `data/raw/pooled5/{书籍文档,软件}_pooled5.parquet`（21,786 / 21,920 行，各 4 个快照，无语音）；
> 领域档案 `configs/domains/{books_docs_zh,software_apps_zh}.yaml`（每个占比都是实测）；
> 运行 `book-pool5` / `soft-pool5`，fast、routed、未 halt，`verify_run.py` 各 21 PASS / 0 FAIL；
> 交付在各自 `postprocessed/`，跨批次表在 `work/cross_new2/`（**没有碰五域的 `work/cross/`**）。
> **最硬的一条**：软件 `生成露骨性图像编辑` 是七个领域里**唯一**的「仅助手」类（51 行，搜索期望
> 530.04，P(0)=0.0000）。**方法层面的一条**：领域档案是 hypothesis-first 写的，两个域的 agent
> 都找到了它的盲区，最大的一类都比档案里已写的类大。详见本文件末尾当日 session。
> 测试 741 通过、exit 0；ruff clean；**`src/` 未改动**（发现一个 openpyxl 公式缺陷，按约束只做后置修复）。

> **2026-09-13（晚）：逐类 × 逐快照对照报告已交付**，每个领域一份，写在各运行自己的
> `runs/<id>/gen01/postprocessed/` 里（`*_意图与聚类叶_跨快照对比.zh.md` + `.xlsx` + `img/`）。
> 它回答的是主报告没回答的那一层：**每一个意图、每一个聚类叶在五个快照上各占多少、哪些类只出现在
> 某个快照**。脚本在 `analysis/pooled5/p5_snapshot_{classes,figs,report,verify}.py`。
> **纯后置分析，`src/` 与 `tests/` 本次一行未动**（两者的 mtime 仍是 09-12，上一次会话的风控哨兵修复）；
> 全套测试 **741 通过、exit 0**，`ruff --select F src/qmine/ tools/` clean。
> 关键口径：0 条必须配可检出性判定（助手 n≈1,000 时 0 条的上界仍有 0.39%）；「独占某快照」几乎
> 恒为 0，改用「与其余每一个快照逐一比较都显著更高」的**特征类**。详见本文件末尾的当日 session。

> **2026-09-13:** five POOLED-5 runs delivered (`fin/med/edu/film-pool5`, `ppl-pool5b`) — each
> domain's 2025 search + 2026 search + assistant head + assistant tail + (finance/medical) voice
> mined as ONE corpus so one taxonomy labels every source. Reports:
> `docs/POOLED5_2026_五域同体系对比.zh.md` + `docs/POOLED5_2026_领域深挖.zh.md`, figures in
> `docs/img/pooled5/`, reproduction package in `analysis/pooled5/`. Source-tagged copies of each
> run's own workbook are in `runs/<id>/gen01/postprocessed/`; the originals are untouched.
> **One pipeline fix shipped** (`naming.py`: the risk sentinel's fallback is a real `RiskReport`,
> so a provider content filter degrades instead of halting p7 — `tests/test_risk_sentinel_degradation.py`).
> **One fix deliberately NOT shipped**: the p2c branch-join guard, see open question 0v — its
> diagnosis is false on a resume, so making it authoritative would have been worse than the crash.
> Tests **741** pass, exit 0; `ruff --select F src/qmine/ tools/` clean.

> **2026-09-10:** post-run analyses only, no pipeline source changed. Delivered `docs/SEARCH_VS_ASSISTANT_2026.zh.md`, its companion `docs/SEARCH_VS_ASSISTANT_2026_领域深挖.zh.md`, and the reproduction package `analysis/sva2026/`. New analysis tools: `tools/clean_assistant_functional.py` (v3), `tools/unified_intent_frame.py`, `tools/label_unified_intent.py`. Tests: **739 pass** (full suite, `-x`, no failures); `ruff --select F src/qmine/ tools/` clean. Details are in the 2026-09-10 session log below. The `ai04` status that follows is unchanged.


# `ai04` DELIVERED: the multi-vertical AI-assistant corpus is mined end-to-end in fast mode, with the stratum comparison shipping under its own name. Two defects were found and fixed during the run.

| | |
|---|---|
| Tests | **739** passing, exit 0; `ruff --select F src/qmine/ tools/` clean (was 716) |
| Run | `ai04`, `mode=fast`, `provider=routed`, not halted, **448 calls / $7.91 / 3h31m** |
| Corpus | `data/raw/ai_assistant_pooled.parquet` — 65,986 rows, 33 L1 / 213 L2 reference |
| Delivered | **24 top-down intents**, **154 leaves**, 121 families (156 leaves pre-governance; 56 governance ops, 6 declined) |
| Held-out reproduction | **96.9%** (95% CI 0.966-0.972, n=13,198) |
| Gold | 3,000 rows, **coverage 92.50%** — 225 rows came back UNLABELED (the gate said 100%; see below) |
| Deliverables | 3 fast-mode reference documents + `分层对比_头尾结构差异.md` + `垂类交叉表.md/.csv` |
| verify_run vs live42 control | `ai04` **PASS 19 / N-A 6 / SKIP 2 / FAIL 1**; control PASS 20 / FAIL 6 / SKIP 2 |

### Post-run corrections to `ai04`, and the one thing that was general

The maintainer read the delivered tables and found three problems. Deciding which
belonged in SOURCE and which in a post-run step was settled by measurement, not
preference — is it wrong on every corpus, or only on this export?

**GENERAL, so fixed in source.** Every drift/stratum report ends 「原始数据:
`labels_full.csv`（逐行标签，含分层列）」 and **that column was never written** —
false on all six pooled runs on disk (fin/med/edu/film/ppl-pool, filmdrift) as
well as `ai04`. `p10` now carries `snapshot` into the delivered labels when the
run has one; additive, so single-snapshot runs are unchanged.

**CORPUS-SPECIFIC, so post-run** (`tools/postprocess_assistant_run.py`, writes to
`<gen>/postprocessed/`, non-destructive):

* **Sort order.** The delivered table came out in corpus order — stratum then
  query — which reads as alphabetical and scatters each category. Re-sorted to
  (category, sub-category, stratum, descending traffic), with the stratum order
  STATED (`top1k` before `random1k`); sorting those two by name puts the random
  sample first, which is backwards for every reader.
* **Stratum names.** `head`/`tail` -> `top1k`/`random1k`. `tail` is a CONCLUSION
  about where rows sit; the file is a RANDOM sample, which is how it was drawn.
  `tools/prepare_assistant_corpus.py` now emits the new tags (`HEAD_TAG` /
  `RANDOM_TAG`) so future runs never carry the old ones.
* **What the comparison MEANS**, appended to the document as a computed addendum.

### What the stratum comparison actually compares — and one caveat that was wrong

`top1k` is a **census** of each category's 1,000 highest-traffic queries, complete
above that category's own floor (**3 to 419 raw PV**, measured). `random1k` is a
**uniform sample of distinct queries** (raw PV median **1**, mean **1.29** — a
PV-weighted draw would lift the mean far above that). So the comparison is
*"the composition of the highest-traffic band vs the composition of the query
vocabulary"*. Only **115 of 33,000 random rows (0.35%)** fall inside the head
band and 17 of 33 categories have none, so it FUNCTIONS as head-vs-rest — a
measured property of this export, not a guarantee.

**This inverts the document's own reading advice.** 「请按 Δ流量的大小读」 is right
when both sides are censuses. Here the random side's traffic share is a
high-variance estimate: one query holds a median **3.4%** of a category's sample
traffic and up to **36.4%** (交通出行), with 4 of 33 categories above 10%. **Read
the ROW share; treat Δ流量 as indicative.**

And 「不能推回总体」 was too strong. Correctly: within a category the random sample
IS an unbiased estimate of row composition (±1.4pp at 5%, ±3.1pp at 50%, n=1,000);
it cannot be combined ACROSS categories (sizes unknown, capture-recapture fails),
and its traffic share cannot be read as population traffic.

**A bug I shipped into that addendum and caught by reading the output:** the first
version computed the overlap from `weight`, which is normalised WITHIN (stratum,
category) — two different scales — and reported **100%** overlap where the truth
is 0.35%. It is the exact incomparability this corpus preparation exists to
handle. It now computes from raw counts via `--source-corpus`, or states that it
did not compute the figure. Pinned by
`test_the_stratum_addendum_will_not_state_an_overlap_it_cannot_compute`.

### `_AXIS["stratum"]` was over-fitted to this corpus, and is now frame-agnostic

It said 「头部按流量取 TopN，尾部是随机抽样」 — this export's design, written into
the shipped vocabulary, where it would mis-describe any other stratum pair (two
devices, two collection methods). The shipped text now says only what is true of
every stratum comparison; what each frame IS lives in the post-run addendum,
computed from the run's own data. The `time` branch remains byte-identical to the
pre-change module.

### The stratum axis works end to end

`分层对比_头尾结构差异.md` shipped under the stratum name, and contains **zero**
occurrences of 不是趋势 / 同月同日 / 时段性事件 / 两期的抽样方式必须一致 / 漂移.
The inverted caveat is replaced by its opposite (「抽样口径不同正是本报告的自变量」)
and the real estimand limit is stated (「不能推回总体」). Measured: head and tail
share **159 queries, a Jaccard of 0.2%**; Cramér's V **0.3232**.

### The `UNLABELED` sentinel is counted as a CLASS in three places

`ops.classify.UNLABELED` fills a row no annotator or classifier could label. It is
not a class, and three separate readers treat it as one:

1. **`p2b_kappa`'s coverage** — hidden entirely; the gate reported 100% while 225
   of 3,000 gold rows held the sentinel (fixed, below).
2. **`verify_run.py`'s phantom-class check** — surfaces it, but under the label
   "referee typos", which is the wrong diagnosis for the right observation.
3. **The class COUNT.** `taxonomy.json` has **24** nodes; `labels_full.csv` has
   **25** distinct `td_l1` because ONE row carries `UNLABELED`, and
   `分层对比_头尾结构差异.md` therefore says 「共 25 类」. The delivered taxonomy is
   24. `tools/run_evidence.py` reports 24 and is right.

Only (1) is fixed. (2) and (3) are cosmetic on this run — one row — but the same
sentinel would inflate a class count by however many rows a real outage lost.

### The one FAIL, and why it is the same defect as the coverage bug

`❌ [gold] no phantom classes from referee typos — 1 phantom classes: ['UNLABELED']`.
Not a referee typo: `UNLABELED` is the sentinel filling the **250 gold+pilot rows
the annotator lost**. Two independent instruments, one cause — the p2b gate hid it
behind a tautological 100%, the mechanical verifier surfaced it as a phantom
class. Fixed in the gate (see below); the verifier's message could be clearer
about the distinction but was not touched.

### What a reader must not take at face value

* **Coherence 3.37/5, 26 of 156 leaves below 3.0** — worse than the 3.90-4.06 of
  previous corpora, and it is the CORPUS, not the clustering. The four worst
  leaves are 过滤无意义表情符号与乱码输入, 短词查询 (工作/房子/塑料/海豚),
  识别日常口语片段并应答, 杂项查询意图识别 — the acknowledgement, emoji and
  short-fragment population. 86 of 156 leaves sit at 4-5; the mean is dragged by a
  noise floor a search log does not have.
* **59 of 156 leaves (38%) are risk-flagged.** Plausible given an entire 成人色情
  vertical plus the fiction/roleplay population, but unread by a human and fast
  mode dropped the adversarial validation that would probe it.
* **ECE 0.0807, 11.4 sd above this run's calibrated null (0.0190 ± 0.0054).** The
  classifier's confidence is miscalibrated, which matters because rows under a
  0.02 margin route to a fallback on that confidence.
* **`ai04`'s deliverables carry the false 「覆盖率 100%」**, written to state before
  the fix. `qmine render ai04` into a new generation is the cheap correction;
  whether the gate line regenerates on render is UNVERIFIED.

### Cost of getting here

`ai01` $2.31 (halted for fast mode; found the alpha knife-edge), `ai02` $0.37
(halted: truncation defect), `ai03` $0.55 (halted: lost the log_reading angle),
`ai04` $7.91 delivered. **$11.14 total.**

## 2. Open questions — EDIT THIS SECTION, DO NOT APPEND

**逐串筛查名单是按域建的，而两个医疗域的内容是重叠的。**（2026-09-17）
`SCREENED_QUOTE_BLOCK` 每个域一份名单。做科室层报告时实测到：`女性到达顶峰什么症状` 在 医疗8 的名单上
（med-pool8 逐串筛查判为性内容），却作为 医疗随机 的妇科例子印了出来——医疗随机 自己的名单里没有它。
两份语料是同一个垂类，写法本来就重叠，一串在这边被判为不可引、在那边却可引，这个不一致是结构性的。
科室层报告里已经按两域名单的并集处理（只影响那一份新报告）；**没有动任何一个域自己的护栏配置**，
因为那会改变 医疗随机 已交付报告的重渲染结果。
要不要改：把 `screened_quote_block(domain)` 改成按「同垂类域组」取并集（医疗8 + 医疗随机 + 健康），
然后重渲染这三个域的报告并逐条核对例子变化。收益是口径一致，代价是三份已交付报告的例子会变。

**`p2c_both_branches_arrived` 会停机，但闸门台账把它记成 `warned`。**（2026-09-16）
它不在 `cfg.gates.blocking` 里，所以 `Deps.gate` 算出来是 `status='warned', blocking=False,
halts_run=False`；真正的停机走的是节点返回的 `{"halted": True}`，`_gate_router` / `_wrap` 认这个键。
后果：一次因为分支缺失而停掉的运行，`run_summary` 里仍然是 `gates_failed=0`，读台账的人看不出它是被拦下的。
没有动它，因为改法是改闸门策略（把这个名字加进 `gates.blocking`），而运行本来就停了——属于披露层而不是行为层。
要改的话：加进 blocking 列表后，确认 `test_the_delivered_leaves_gate_reaches_the_operator` 与
`test_the_join_refuses_to_run_when_a_branch_never_arrived` 两侧都还成立。

**并发分支的排程：金标从缓存重放时会被排到 `p456_tree` 之后。**（2026-09-16，ppl-pool8 gen03）
gen01 里 p2b_gold 用了 100 秒真调用、远早于 p456_tree 的 638 秒完成，汇合点没事；gen03 全部缓存重放，
p2b_gold 6.5 秒，却是在 p456_tree 完成的**同一秒**才开始，于是 p2c 在顶向下分支没到齐时就触发了。
闸门本身已经修好（缺陷 ②），现在会干净地 halt 并说清缺哪个分支，但**为什么这么排没有查**：
是 LangGraph 的 superstep 边界，还是两个 CPU 密集的自下而上节点把线程占满、饿死了另一条分支？
影响：任何「上游全缓存命中」的 resume 都可能撞上，代价是一次干净的 halt + 一次 `--resume`。
查法：在 `graph/build.py` 里给两条分支的节点加进入 / 退出时间戳跑一次全缓存 resume，看重叠区间。

Resolved items are **deleted** here and their resolution recorded in that
session's log below. A struck-through entry is a maintenance failure, not a
record.

### P1 — worth doing next

0v. **The concurrent-branch join guard misdiagnoses a resume, and its own gate call is a
   latent TypeError. Both are still there — deliberately — and they have to be fixed
   together.** Measured 2026-09-12 on `ppl-pool5` gen02 (a `--resume` after
   `new-generation`), the only time this guard has fired in 84 runs on disk:

   ```
   20:58:23  !! 分支缺失: p2b_gold 从未运行, 但流程已到达汇合点
   20:58:23  node p2c_classifier failed
             TypeError: Deps.gate() got an unexpected keyword argument 'blocking'
   20:59:59  ✔ p2b_gold completed in 95.3s          <- 96 seconds LATER
   ```

   - **The diagnosis is false.** `gen02/run_summary.json` lists `p2b` in
     `completed_phases`; `gold.csv` and `gold_agreement.json` are on disk. The branch had
     not "never run" — it had not FINISHED when p2c reached the join.
     `_require_both_branches` (`graph/nodes/topdown.py:2739`) reads `phase_status`, which
     records completion, so at a concurrent fan-in it cannot separate *never ran* from
     *still running*.
   - **The gate call cannot work either.** `deps.gate(..., blocking=True)` —
     `Deps.gate` (`graph/deps.py:282`) has no `blocking` parameter; it derives blocking
     from `name in cfg.gates.blocking`. So the guard raises from inside itself and the
     remediation it carries ("open a new generation and run it in one go; the cache
     replays paid calls") never reaches the operator. `_wrap` turns the crash into
     `halt_kind="crash"` plus a lesson pointing at the wrong thing
     ("p2c_classifier is not robust to this input").
   - **Why nothing was fixed.** Repairing only the gate call was tried and then reverted
     on 2026-09-12: it upgrades a FALSE diagnosis into a clean, authoritative halt whose
     remediation tells the operator to spend a fresh generation on a branch that was 96
     seconds from finishing. A crash at least reads as "this is not understood". Putting
     the gate in `cfg.gates.blocking` was tried too and reverted: an independent audit
     found it would list `p2c_both_branches_arrived` under
     `declared_gates_never_evaluated` in **every healthy run's summary**.
   - **The order to fix it in:** first establish how `phase_status` is written at the
     fan-in on a resumed run (does the join node ever run before a sibling branch
     completes in a NON-resumed run?), then make the premise able to say "not finished
     yet"; only then make the gate speak authoritatively.
   - **Test note:** `tests/test_concurrent_branches.py`'s fake `_gate` takes `**kw` and
     derives `halts_run` from a `blocking=` argument the real API rejects — which is why
     the TypeError was invisible to the suite. A fake that accepts more than the real
     thing cannot catch a call the real thing rejects. The clean way to close that hole is
     the real `Deps` from `conftest.py:54`, not a hand-synced fake.

0t. **`--fast` is SILENTLY IGNORED on `--resume`, and a config file's `mode:` is
   silently overruled.** Both measured 2026-09-07, both the same failure class as
   `test_an_unset_cli_flag_does_not_overrule_the_config` (which pins
   `text_column` and `reference_columns` and does not cover `mode`).

   - **On resume:** the `if resume and run_id:` branch never calls `_load_config`.
     It loads `config.resolved.yaml` and applies exactly two things —
     `cfg.run_root` and `cfg.taxonomy.reuse_taxonomy_from`. So
     `qmine run --resume --run-id X --fast` resumes in **full** mode and pays for
     the second-opinion layer the flag asked to skip. The `reuse_taxonomy` line
     right there carries a comment explaining that this exact branch swallowed
     that flag once already; `mode` is the same bug, un-fixed.
   - **From a config file:** `cli.run` always passes `mode="fast" if fast else
     "full"`, never `None`, so the override loop always fires. Verified: a config
     saying `mode: fast` loads as `mode='full'` when `--fast` is absent.

   **The two halves need OPPOSITE fixes, and an earlier draft of this entry got
   the resume half wrong.** `mode="fast"` is not a display flag: a pydantic model
   validator (`config.py:641`) applies it at CONSTRUCTION, setting
   `taxonomy.annotators = 1`, zeroing `kappa_repair_rounds` and
   `max_taxonomy_redraws`, and switching off `observe_phases`,
   `validate_adversarial`, `final_report`, `delivery_audit` and
   `interpret_results`. Honouring `--fast` on a resume would therefore apply
   those to a run that has already executed phases under the other setting — a
   run whose first half annotated with two annotators and second half with one,
   whose summary then claims `mode: fast`. That is worse than ignoring it.

   So: **the resume branch should REFUSE `--fast` (or warn loudly and continue
   full), not honour it** — the silence is the defect, not the ignoring. The
   config half is a plain bug with a clean fix: give `--fast` a tri-state default
   so an unset flag stops overruling a config file, exactly as `text_column` and
   `reference_columns` already do.

   Note the RENDER path gets this right already (`runner.py:613`): it restores
   `mode` from the previous run's summary and re-validates, so a fast run's
   re-render stays fast.

   **Consequence today:** the ONLY way to get a fast run is `--fast` on the
   command line of a FRESH run id.

0i. **`model_overrides` silently ignores a suffixed role.** Routing resolves
   `researcher_log_reading` to its BASE role `researcher` before looking up a
   model, so only `researcher` is ever consulted. An entry keyed on the suffixed
   role sits in the config doing nothing — no warning, no log line, and
   `qmine models` even ECHOES it back ("researcher_log_reading=deepseek-v4-pro"),
   which makes it look applied.

   Found by trying to route around 0h and watching med03 fail at 903.7s on the
   very model the override named away from. Per-angle pinning is not supported
   and currently cannot be discovered except by a run.

   Fix: either consult the suffixed role before falling back to the base (the
   `_prefix_route` longest-match rule already does this for `_routed`, so the two
   tables disagree), or REFUSE an override whose key resolves to no routable
   role, so dead config fails loudly at startup instead of being echoed as if
   live.

0h. **`researcher_log_reading` fails deterministically at ~903s — now THREE
   runs.** med01 903.5s, med02 902.9s, med03 903.7s. All `InternalServerError`
   with an HTML body and `out 0`, on glm-5.3-flash, reproduced to within 0.8s
   across three independent runs. Refuted explanation: not a duration ceiling —
   `risk_compliance` completed a single uninterrupted 1,024.8s call on the same
   model. Request-specific; `log_reading` reads raw corpus rows, the largest and
   least structured payload any researcher gets.

   The obvious workaround (pin the angle elsewhere) does NOT work — see 0i — and
   the obvious target is forbidden: the comment above the researcher pin records
   that deepseek-v4-pro 400s on `tool_choice` for researcher roles, returning
   parametric-knowledge candidates with zero tool calls. I proposed that target
   without reading the warning directly above the line I edited.

   Cost: ~15 minutes plus a retry, every run. Not data loss — the retry succeeds.
   Next step: fix 0i first, then pin this ONE angle to a provider that is neither
   zhipu (fails) nor deepseek-v4-pro (tool_choice 400) — e.g. the moonshot or
   qwen tier already in the plan — and confirm from the run log, not from
   `qmine models`, which echoes overrides it has not applied.

0g. **The alpha optimum can sit at the GRID'S EDGE with the metric still
   improving, and nothing says so.** Machine-confirmed by the p3 observer on
   med03: `alpha_sweep.chosen_alpha < max(grid_proposal.widened)` FAILED.

       alpha  0.0    0.1    0.2    0.3    0.5    0.7    1.0
       frag   2.487  2.433  2.719  2.623  2.204  1.771  1.481  <- best, at the edge
       stab   0.841  0.650  0.765  0.642  0.752  0.808  0.642  <- worst, at the edge

   Template fragmentation is still FALLING at alpha=1.0, the largest value
   searched, so the grid does not bracket the optimum — alpha > 1.0 may be
   better and nothing looked. At 1.0 the phrasing block controls **50%** of the
   cosine (`surface_vote_share = a^2/(1+a^2)`), which is a large representational
   commitment to make at an unexplored boundary.

   The winner is also the LEAST stable point in the sweep (0.642 against 0.841 at
   alpha=0). That is legitimate under the documented rule — fragmentation
   locates, stability only vetoes, and 0.642 clears the floor — but "best on the
   deciding metric, worst on the veto metric, and at the edge of the searched
   range" is three facts a reader should get together, and currently gets none of.

   NOT a reason to extend the grid automatically: `ops/propose.py` is blind to
   scores ON PURPOSE so its additions are pre-registered, and widening because
   the winner sits at the edge would use the scores it must not see.

   Fix (AFTER med03): a boundary DISCLOSURE, not an automatic extension. When the
   chosen value is the min or max of the swept grid AND the deciding metric is
   still improving in that direction, say so in the artifact and the report —
   "the optimum was not bracketed; the true optimum may lie outside the searched
   range". Same treatment as the singleton-agreement and cross-k fixes: the
   measurement stands, the overclaim goes.

0. **The duplicate audit runs BEFORE governance creates duplicates — PARTLY
   ADDRESSED.** p7 audits the tree, p8 then splits it, so a duplicate governance
   itself introduces was never audited (live44 split leaf 30 into {30, 50} and
   both halves were named `汉字笔画数查询`).

   `p8_leaves_are_distinguishable` now measures the DELIVERED partition after
   naming, and `DisambiguatorAgent` either names the difference or prescribes
   `merge_leaves`. The K12 demo proved it works: it caught `的拼音相关查询` on
   leaves [15, 16] on its first run.

   STILL OPEN: the gate is deterministic on EXACT name equality only. Semantic
   near-duplicates (live44's 27/29, 部首 vs 偏旁部首 — different strings, same
   concept) are caught only by the auditor's cosine list, which returned `null`
   for 4 of the 14 pairs it reported on that run. A geometric duplicate check on
   the delivered partition would close it; it needs a threshold, which is exactly
   what the exact-match gate was designed to avoid. p7 audits
   the tree, p8 then splits leaves — so a duplicate governance itself introduces
   is never audited. live44 split leaf 30 into {30, 50} and both halves were
   named `汉字笔画数查询`, byte-identical, in the same family. The split had real
   geometric support (lift 0.1565 over null, ARI 0.9887); it was **semantically**
   empty, and nothing measures that.

   `merge_leaves` (added 2026-09-01) lets the auditor act on duplicates it CAN
   see. It cannot see these. Options, none yet taken: re-run the duplicate check
   after governance; or refuse a split whose two halves would receive the same
   name, which is cheap because p8 already names both halves. Same shape as every
   other "a gate before the operation that breaks its invariant guarantees
   nothing".

1. **Does the narrative writer still return an empty JSON?** live42 lost three
   sections to `{"markdown": "", "covered": []}` on all three attempts. Today's
   `render --agents` against the same artifacts returned **9/9 sections**, so it
   did not recur — but the fact sheets changed too (the sign fix, the widened
   citable pool), so this is evidence, not a controlled test. Both real repros
   before it also failed to reproduce a blank. Do NOT add mitigation machinery
   until one reproduces: the three prior suspicions here (truncation, sheet size,
   prompt-block truncation) were each refuted by measurement.

2. **Two English strings reach three Chinese reports each (6 lines).** Confirmed
   still present today by `verify_run.py` on live42. "DIAGNOSTIC ONLY — do not
   choose a model from this number…" and "Match the two annotators…", both from
   the annotator-balance work, never routed through `prose()`. The static AST
   guard covers `deps.decision()` rationales only — it does not see gate messages
   or remediations, which is where these live.

   NOT closed by the 2026-09-01 work. That fixed a THIRD string of the same class
   (`{n} L1 intents across {m} axes`, `topdown.py`) by making the decision choice
   symbolic. These two are gate messages and still English. Note `prose()` cannot
   rescue an f-string: `PROSE_ZH` returns a fixed string, so it cannot carry the
   numbers. Either author them symbolically as well, or install a translator.

3. **The resume rewind silently drops a concurrent branch.** Made LOUD (the join
   halts with `p2c_both_branches_arrived`), not fixed. Open: whether
   `update_state(as_node=...)` can restore a multi-superstep fan-out at all, or
   whether a gap in `phase_status` should force a clean re-run of the generation.
   Until answered: open a new generation and run it ONCE, never restart mid-flight.

### P2 — measured, disclosed, not acted on

0y. **其它域的已交付报告与工作簿例子没有逐串读过，可能同样漏引违规串（2026-09-15 在医疗8 实测）。** 医疗8 的前五层护栏在印出的例子里
   漏了 38 串（未成年人与性、露骨 / 恋物题材、具名医生、民营医院、试管选性别），大多来自语音快照；DeepSeek 单独筛查只召回 15/38，
   单名 Claude 读者 22/30。敏感域（医疗 med-pool5、健康 health-pool2）风险最高。做法现成：`analysis/pooled5/p5_privacy_screen_round.py`
   （seed → 逐轮 apply / rerender → final）加 `p5_snapshot_classes.SCREENED_QUOTE_BLOCK` 按域登记名单文件；需要把路径按域参数化。

0z. **旧报告里写死的「语料 87% 是搜索行」只对五域的金融、医疗成立**（87.2% / 87.1%）；教育 91.6%、影视 91.2%、书籍文档 91.7%、
   软件 91.2%、金融8 91.0%、健康 90.9%。已改为按域计算（`p5_snapshot_report._search_share`），**已交付的旧报告没有重渲**。

0x. **fin8 主报告的意图卡片例子是「流量前 3」，不代表该格（2026-09-15 测）。** 覆盖中位 32.8%，27 格的排序其实是文件顺序，
   自助法 Jaccard 0.44。构成显式取例（`analysis/pooled5/p5_intent_structure.select_cover`：每叶中心、标份额、≤8 条到 80%）
   已在深挖报告里用上，**没有**接进 `p5_snapshot_classes.examples`——那份报告七个域共用，改了要回归所有域的交付哈希。
   同一次还测到：INVEST_ADVICE 约三分之一的行不是求建议（无评价标准的标的清单 20.4% + 事实/制度信息 14.0%），
   下次 fin 运行值得写进领域档案的 `domain_notes` 作为边界提醒。

0l. **Six measured limitations rescued from the README (2026-09-03).** The
   README's "Known limitations" section was deleted at the owner's request during
   the README overhaul. Four of these six existed **nowhere else** — not in this
   file, not in the code — so they are recorded here verbatim rather than lost to
   git history. None is resolved; each is disclosed and unacted-on.

   **The request timeout assumes one throughput for every role, and it is wrong by
   5x in both directions.** `timeout_seconds` derives from a single constant of 40
   output tokens/sec. Measured across roles on `live44`, real throughput runs from
   **7.4 tok/s** (a tool-free researcher) to **181.7** (an annotator). The slow end
   is reasoning: with no tool round-trips, wall time is dominated by thinking, and
   thinking tokens are not counted in the output total — so they land in the
   denominator and not the numerator.

   **The α decision sits inside its own noise.** Across five seed replicates the
   winner was 0.1, 0.5, 0.1, 0.0, 0.1. The tie band is roughly 4.5× narrower than the
   metric's own spread. Widening the band makes it worse — at a measured 2-sd band
   the run elects an α its own panel shows fragments intents more. The fix is
   replication, which has not been done.

   **Refinement converges on some corpora and not others.** `fin01`/`fin02`/`fin03`
   (finance) and `ecom01`/`ecom02` (e-commerce) all report `converged: true`;
   `live39`-`live44` (K12) and `med04` (medical) all hit the iteration limit, so
   the delivered leaf count depends partly on which round it stopped at. Disclosed in
   the reports; not fixed.

   **Restarting into the concurrent region is unreliable.** The two routes fork, and
   a resume that lands inside the fork has silently dropped a branch. The join now
   halts loudly instead, but the underlying question — whether a multi-superstep
   fan-out can be restored at all — is open. Open a new generation and run it once.

   **Model behaviour is a live variable.** One provider returned the JSON *schema*
   instead of data on 69 of 197 `annotator_b` calls on `live44` (35%), against 1 of
   137 for the other annotator; because every field on the response
   model had a default, that validated into a valid-but-empty result and silently
   lost half a gold set before it was caught. It is now rejected before validation
   and retried — but the class of failure is general, and a permissive default
   anywhere is a place it can recur.

   **The annotated distribution is not the corpus distribution.** Gold rows are drawn
   by cluster-stratified sampling precisely so rare intents survive selection
   ([Rao et al.][rao] avoid random sampling for the same reason), which means class
   shares on the gold set are *not* an estimate of class shares in the corpus. Those
   come from the delivered labels over all rows. Anywhere the two are printed near
   each other is a place to check the denominator.


0k. **Delta concentration is computed on raw query strings, so ONE ENTITY can
   read as dispersed.** `WATCH_LIVE_TV` scored top1=23% across 311 distinct
   queries, which the report calls "concentrated" — but its top five deltas are
   all cctv5 phrasings, i.e. essentially ONE entity. The number is honest about
   what it measures and the shipped query list makes the truth visible, so this
   is a sharpness limit, not a defect. The literature review's suggestion is to
   recompute the delta with named-entity spans replaced by a placeholder: if the
   rise survives, it is a pattern; if it vanishes, it was an entity. Not done —
   it needs an entity recogniser this pipeline does not have.

0j. **The five pooled runs on disk have no `drift_analysis.json`.** fin-pool,
   film-pool, med-pool, edu-pool and ppl-pool predate `p10b`, and `qmine render`
   cannot add it — render replays report generators, and p10b is a phase. Their
   drift figures exist only via `tools/drift_report.py` (which is why that script
   was kept). Either re-run one pooled corpus to get a first-class artifact, or
   accept the external script as the record for those five.


5. **The declared per-role token budgets are miscalibrated in both directions,
   and now quantified.** Measured against live42's own `usage.json`: the
   annotators are 500 of 702 calls, declared 12,000 output tokens, actual
   1,612-1,751 — a **7x over-estimate on the roles that dominate the run**.
   Observers, researchers and the delivery auditor run **2-6x UNDER** their
   declared budget. The declared number drives the cost estimate, the cap
   (`3x`) and the timeout, so recalibrating moves truncation risk — it deserves
   its own pass, not a drive-by edit.

6. **The spend ledger is wrong in both directions.** Provider prompt-cache
   discounts are not recorded (input is 7.3x output on live42, annotators 94% of
   it, and the prompt is already ordered for prefix caching), so cost
   OVER-reports; a pinned model with no published price counts as $0.00, so it
   UNDER-reports. `_pin_warnings` says so out loud; nothing corrects it.

7. **`challenger_beats_incumbent` has no production call site.** Four docstrings
   and a model-budget decision rested on it; they now say so, and the CLAUDE.md
   invariant row no longer states it as a live guarantee. Needs a signature
   change (`propose_grid` returns a flat list, so selection cannot tell a proposed
   value from a configured one). Applying it as written flips live40 to a worse K.
   Still open — only the overclaim was fixed, not the wiring.

0s. **FIXED 2026-09-02 — `merge_leaves` no longer voids a risk isolation.**
   med04 shipped leaves 14 and 24 merged away AND "isolated", so two risk
   clusters ([21,10,14] and [43,24,35]) were unisolated and ghost families 42/33
   held no rows — the artifact said 36 families where 34 had content.

   Two changes, both verified against med04's real conflict:
   - `execute_prescriptions` reconciles AFTER the prescription loop: a leaf that
     is both merged and isolated has its MERGE declined with a reason, because
     isolation is a SAFETY action and merging a QUALITY one. Redirecting the
     isolation to the survivor was rejected — it would isolate the survivor's
     other rows on a guess.
   - `isolate_leaves` now takes `leaf_labels` and REFUSES an empty leaf,
     recording why. Defence in depth for any other path that empties one.

   The cascade is right: both leaves stay live, the isolation moves real rows, no
   ghost family, and the surviving duplicate routes to
   `p8_leaves_are_distinguishable` — the safety action wins and the quality
   problem goes to the quality mechanism.
   Tests: `test_a_leaf_merge_never_voids_a_pending_risk_isolation`,
   `test_declining_a_merge_still_counts_as_settled`.

0n. **The observer's `decisions` channel is per-phase and narrow, so an agent
   that saw a decision id in an ARTIFACT cannot cite it.** med04 dropped four
   citations: `granularity.triangulation.k_tie_set`, `D004.decisive_metrics`,
   `D006.evidence`, `decisions[0].evidence.critic_verdict`.

   Callers pass `decisions=[decision]` (p4, p5) or `decisions=[]` (p6), so the
   id-indexing added for 10a only covers that phase's own decision. An agent
   reads D004 inside an artifact, cites it the way the record prints itself, and
   is refused. Residual of 10a, not a regression of it.

   Fix: index decision ids found in the ARTIFACTS too, or pass the full decision
   ledger to every observer. Prefer the latter — the ledger is small and an
   observer reasoning about a decision it can see is the point.

0f. **8 figures on med04 against live44's 11, unexplained.** Not investigated;
   `test_one_figure_per_quantity` passes, so it is not duplication. Check whether
   three figures are conditional on artifacts this corpus lacks (the untrusted
   template groups are the obvious candidate) — a figure that silently does not
   render is the same class as a section that silently does not ship.

0c2. **The 0c disclosure reaches NO artifact, and the fix is still unverified.**
   Two separate corrections to what I claimed on med04.

   FIRST: the `1.0344 of ceiling reached` in med04's `p2b_kappa` gate is NOT the
   0c fix. There are two ratios and two gates:
   - the PILOT gate prints `share_of_ceiling_reached`, which comes from
     `headroom` — the variable 0c changed
   - `p2b_kappa` prints `share_of_ceiling` (topdown.py ~974), a DIFFERENT
     variable that was never clamped
   med04's pilot ratio was 0.9603, below 1.0, so the clamp never engaged and the
   fix was never exercised. Verifying it needs a run whose PILOT kappa exceeds
   the PILOT ceiling — med01 (0.930 vs 0.9188) and med02 (0.922 vs 0.9033) did;
   med04 did not.

   SECOND, and larger: **no artifact on disk contains `self_consistency` at
   all.** The pilot dict lives only in LangGraph state, so
   `self_consistency_is_lower_bound`, `kappa_exceeds_self_consistency` and the
   explanatory note are computed and then reach no artifact, no report and no
   reader. That is the same write-only pattern as `decisions` and
   `risk_isolated` — a value nothing consumes.

   Fix: persist the pilot block into an artifact (it is the evidence behind the
   ceiling argument, which the reports already discuss), then verify 0c on a run
   that reproduces the pilot-level inversion. Until both are done, treat 0c as
   WRITTEN BUT UNPROVEN — it was nearly recorded as verified on the strength of a
   number produced by different code.

0d2. **A SECOND batch-loss signature the three-tier key matching does not
   rescue: ALL keys unmatched, not one.** med04:

       ⚠ annotator[b] batch lost 25 rows after 3 attempts:
         ValueError: returned 25/25 labels (25 queries unlabelled)

   med02's case was `1 queries unlabelled` — one key with a trimmed space or a
   full-width comma, which exact -> NFKC -> whitespace-stripped matching now
   resolves. Here NONE of the 25 matched, so the model applied a SYSTEMATIC
   transformation to every echoed key (a paraphrase, a truncation, a translation
   — unknown, because the retry discards the response).

   Not caused by the fix and not made worse by it: this batch would have been
   lost before it too. The fix rescues per-character noise; it cannot rescue a
   wholesale rewrite, and it should not try — matching 25 rewritten strings back
   to 25 queries by similarity would risk assigning labels to the wrong rows,
   which is worse than losing the batch.

   Impact is bounded and DISCLOSED: 1 batch of 120, `annotator[b] labelled
   2975/3000` against annotator[a]'s 3000/3000, so 0.83% coverage on one side and
   the number says so. Contrast live43, where the same underlying failure
   produced `1500/3000` with zero warnings.

   Worth doing before the next fix attempt: LOG THE FIRST FEW RETURNED KEYS on a
   total mismatch. The diagnosis needs the actual transformation, and every
   attempt so far has had to infer it from a count. One `deps.emit` of
   `sorted(got)[:3]` would settle what the model is doing.

8b. **The TAXONOMY is not stable across runs, and neither is its
   annotatability.** Same corpus, same method, same five angles:

   | run | classes | rules | pilot kappa | ceiling | ordering |
   |---|---|---|---|---|---|
   | med01 | 22 | 51 | 0.930 | 0.9188 | inverted |
   | med02 | 20 | 28 | 0.922 | 0.9033 | inverted |
   | med04 | **18** | 47 | **0.835** | 0.869 | correct |

   Kappa spans **0.835-0.930** on identical data. It is NOT explained by:
   - the 0h fix — `log_reading` contributed 12 candidates in ALL THREE runs (it
     succeeded on retry in med01/med02), so making it succeed faster changed the
     wall clock, not the input. Hypothesis checked and refuted.
   - class count — med04 has the FEWEST classes (18, coarser distinctions) and
     the LOWEST agreement, which is backwards if granularity drove it.

   What differs is the partition itself: three genuinely different carvings of
   the same space, each internally coherent, each differently annotatable. This
   is the top-down analogue of item 8 — there the GRID decided alpha; here the
   architect's draw decides the taxonomy, and a headline kappa inherits that
   variance.

   Consequence for reading any single run: kappa is a property of THIS
   taxonomy-and-annotator pair, not of the corpus or the method. Comparing kappa
   across runs — or citing one run's kappa as "the" agreement for this corpus —
   is comparing different objects. n=3, so this is an observation with a
   direction, not a measured distribution.

   Worth doing before trusting any kappa as a methodology result: replicate the
   taxonomy draw on one corpus and report the spread, the same way the alpha
   noise floor was established.

8. **The alpha decision is decided by the GRID, not by the corpus — three
   answers from identical data.** Stronger evidence than the seed-replicate note
   this replaces. The medical sweep rows are byte-identical across med01/02/03
   (alpha=0 is always frag 2.4868, stab 0.8405), yet:

   | run | grid top | tie band | contenders | chosen |
   |---|---|---|---|---|
   | med01 | 0.85, 1.0 | 1.5548 | 0.85 (stab .775), 1.0 (stab .642) | **0.85** |
   | med02 | 0.57 | 2.0239 | 0.57 only | **0.57** |
   | med03 | 1.0 | 1.5548 | **1.0 only** — 0.85 was not proposed | **1.0** |

   med03 chose 1.0 BY DEFAULT: with 0.85 absent from the blind proposer's grid,
   only one value cleared the band. Same corpus, same metrics, three answers.

   **The selection rule is NOT the problem and has not been changed**:
   `contenders = frag <= band` then `max(stability)`. Fragmentation defines the
   band; stability picks inside it. The problem is that the band is computed from
   `min(frag)` over WHATEVER the grid contains, so a proposer that omits one
   value moves the band and changes the winner.

   **Open question worth deciding deliberately** (raised by the user 2026-09-01,
   and the evidence supports it): alpha=0 has the BEST stability in the whole
   medical sweep (0.8405) while the chosen alpha=1.0 has the WORST (0.6419), so
   the delivered tree is the least reproducible option available. At 1.0 the
   phrasing block controls 50% of the cosine, which on a medical log risks
   clustering by question form — `黄精的功效` beside `布洛芬的功效` because both
   are `X的功效与作用`. Alpha=0 never enters contention because fragmentation
   alone defines the band and alpha=0 has the worst fragmentation.

   Counter-evidence, and it matters: `template_fragmentation` is NOT merely
   circular with alpha. On K12 it moves the OPPOSITE way — 1.9799 at alpha=0
   rising to 2.5713 at alpha=1.0 — so higher phrasing weight makes it worse
   there. The metric responds to corpus structure, not mechanically to alpha.

   Candidate resolutions, none applied: (a) require the winner to clear a
   stability floor relative to the sweep's best, not just the absolute veto;
   (b) make the tie band fixed rather than derived from the grid's own minimum;
   (c) report the alpha decision as a range when the grid is sparse near the
   optimum. Do NOT let the proposer see scores — that is the pre-registration.

9. **Annotator asymmetry, unattributed.** live42: annotator_a won 34.1% of 270
   contested rows, z=-5.23. NO baseline exists (the gate postdates live40). The
   parsimonious reading is the pairing — a flash tier against a plus tier. Note
   the pins have since changed (`annotator_b` is now `qwen:qwen3.8-flash`), so a
   re-test would not reproduce live42's pairing. Testing whether disabling
   reasoning contributed needs a paired re-run of the same rows.

11. **Observers cost ~2.5 hours of a 4-hour run.** 11 observers, median 809s on
    live41 against 285s on live40 for the SAME output volume — latency, not token
    inflation. They earn it (three real defects), but the trade should be a
    decision, not an accident.

12. **Refinement converges on some corpora, not others** (was: "has not converged
    on any run" — falsified 2026-09-02 by `hierarchy_meta.converged`:
    `fin01`/`fin02`/`fin03` and `ecom01`/`ecom02` are all `true`; every K12 and
    medical run is `false`). live40, live41, live42 all hit
    the 5-round limit. Honestly disclosed in the report, but it means the
    delivered leaf count depends on which round it stopped at.

13. **Two vacuous rule triggers found live and not repaired.**
    `academic_knowledge_qa x problem_solving` names 求/解/计算 with 0 of 19
    contested rows carrying any; `navigational x school_info` names 主页/入口/好不好
    with 0 of 8. The detector works; nothing acts on what it finds.

14. **Two of nine governance splits are geometrically unsupported** (leaf 19 -> 54
    at ARI 0.0595 with a near-duplicate name; leaf 0 -> 49 at 0.3876). Disclosed
    by design (measure-don't-veto). Nobody has looked at them.

20. **Four delivered strings still credit the phrasing groups with locating K** (found
   2026-09-15 by the review of the `reference_sensitivity` fix, and left out of that fix on
   purpose so it stayed a disclosure-field change). They print on every run a declared column
   located — 10 of the 34 stored generations that carry `reference_sensitivity`. Locations
   checked against the working tree on 2026-09-15:
   - (a) `graph/nodes/bottomup.py` ~569, p5 decision record: rejected K get
     `why_rejected = "lower alignment with the phrasing groups"` and `metrics.intent_alignment_ami`
     whatever the locator was (live44's own auditor raised it as D005).
   - (b) `report/zh_bottomup.py:339/342`: the tie-set column 「意图对齐 AMI (定位指标)」 prints
     `intent_alignment_ami`, not the locator's column (which `tie_set` already carries).
   - (c) `report/zh_bottomup.py:367`: 「K = 与措辞群的对齐度 (AMI) 定位」.
   - (d) `report/i18n.py` ~300: the p5 rationale is translated as 「K 由与模板群的对齐度 (AMI) 定位」,
     although the English source now says "the reference named in `deciding_reference`".
   - (e) `report/narrative_brief.py:355`: narrator remit 「K 由与模板群的对齐度定位」.
   Also stale prose that describes the phrasing groups as the only locator: `ops/cluster.py`
   ~696, ~722, ~761; `graph/nodes/bottomup.py` ~366; `docs/PLAYBOOK_MAPPING.md:14`.
   **Fix pattern:** read `tri["locator"]` / `tri["deciding_reference"]` as the 2026-09-15 fix does;
   make (d)/(e) name the reference rather than the method. Pin with a test that renders a
   declared-column `granularity` through `zh_bottomup` and asserts none of these strings appear.
   **Follow-up check for `tools/verify_run.py`:** "the K locator is named correctly" SKIPs every
   locator that is not `intent_alignment_ami`, so it was blind to all 10 affected runs. Proposed:
   "the reference credited with deciding K is the one that located it" — SKIP without
   `granularity` or without `reference_sensitivity.by_reference`; for `intent_alignment_ami` /
   `ami_vs_<col>` expect the only `decides=true` entry to be that reference, `deciding_reference`
   to equal it, and any 「决定权在」 note to name it; in the stability fallback expect no entry.
   Run it with `live42/gen01` as the known-broken control.

### P3 — small, known, deliberately not growing

15. **22 `deps.gate()` messages are English f-strings** printed verbatim into
    Chinese deliverables. Frozen by gate NAME in `GATE_MESSAGE_DEBT`
    (`tests/test_pipeline.py`); the set may shrink, anything new fails the test.

16. **Two domain profiles are untested on real data:** `sports_zh`,
    `politics_zh`. (`finance_zh` was exercised on 2026-09-02 by `fin01`/`fin02`/
    `fin03` and works — its risk categories, template seeds and
    `expected_l1_range` [15,22] all held, and all three runs landed 16-20 L1
    classes.) `generic.yaml` ships 0 template seeds, so a corpus without a profile
    locates K against unvalidated mined groups — now gated by
    `p3_locator_reference_validated`, but never exercised.

18. **`zh_panel`'s `fixed` and `waived` lists are unsorted** (`entries.values()`),
    unlike `open_findings`, which sorts blocking-first. Their caps are disclosed
    now, so nothing hides silently, but the order is arbitrary.

19. **The reference shelf is Chinese-only.** `builder.build_all_reports` emits
    `类目清单` / `标注规范与裁定规则` / `家族与叶层级` / `00_索引` under `zh` only, so
    an English run delivers the six reports and none of the reference documents.

21. **The stability fallback still names `choose_locator`'s pick as the deciding reference.**
   When `triangulate_k` cannot use the chosen locator (no finite values on stable rows) it ranks K
   by stability and sets `locator` to a free-text string, but p5 still records the pick:
   `graph/nodes/bottomup.py:438` sets `tri["deciding_reference"] = _deciding` unconditionally;
   `:462` `is_the_deciding_reference = (locator_key == "intent_alignment_ami")` tests the REQUESTED
   column; `:494` the reach gate observes `_deciding`; `report/narrative_brief.py:692-706` then
   requires the final report to say 「由参照系 `X` 定位」 verbatim. `choose_locator`
   (`ops/cluster.py:955/957/962`) silently returns `phrasing_groups` for an empty reach, for
   `k_locator: phrasing` with no masks, and for an unknown `k_locator` name — the last hides a
   misconfiguration. The fallback string blames "no phrasing groups available" even when a declared
   column was chosen and had no values (`ops/cluster.py:761-766`). Related and pre-existing:
   `graph/nodes/delivery.py` ~329-381 (p10 `locator_reference_validation`) describes the phrasing
   groups as the K locator's reference. **0 of 69 stored generations reached the fallback.** The
   2026-09-15 fix already makes `reference_sensitivity` and the references-disagree gate name nobody
   there; `test_the_disagreement_gate_names_no_reference_when_stability_decided` deliberately does
   not assert `tri["deciding_reference"]`, so fixing this item will not fight that test.
   Also: live44's open finding `c751301fcba30ec4` checks `["decides"] == true`; the evaluator reads
   the lowercase `true` as an artifact name, so the check is unverifiable on old and fixed
   artifacts alike and will never auto-close. Waive it by hand once a new generation confirms the
   fix. Observer-written checks containing JSON literals (`true`/`false`/`null`) are silently
   unverifiable in general.

## 3. Durable notes — worth not re-learning

- **`researcher_log_reading` on `moonshotai/kimi-k3` fails an attempt ROUTINELY and
  recovers; ~5% of runs it does not.** Re-measured 2026-09-09 over the **39** runs
  on disk that actually ran the angle: **37 produced it, 2 exhausted all three
  attempts** (`ai03`, `med03`), and **10 failed an attempt then recovered**.
  Failing attempt 0 and succeeding on attempt 1 is normal — live34/35/36/42/43/44,
  med01/02 and `ai01` all did it. The failure mode is `ValueError: no parseable
  structured output` while already in plain-JSON mode, i.e. the model emits
  malformed JSON, not a timeout and not a schema-support problem.

  **So a single failure is not evidence of a wrong pairing.** Before re-pinning
  this role, re-measure the base rate; `glm-5.3-flash` is documented in
  `live.yaml` as failing DETERMINISTICALLY on this angle (903s, reproduced three
  times), so the obvious alternative is known-bad and a swap is a real risk.

  `max_repair = 2` (3 attempts) is **hardcoded** at `llm/registry.py:779`, not a
  config knob. Raising it for researchers would convert most of the remaining
  failures — the config's warning against retry amplification is about TIMEOUTS
  (re-issuing an identical request with an identical deadline), which does not
  apply to a parse failure that demonstrably succeeds on retry. Not done.

  Losing this angle is not cosmetic: it is the only researcher whose sole job is
  reading raw rows with no other framing, and on `ai01` it produced
  `续写虚构剧情并接续角色扮演 → continue_fiction_roleplay`, an intent no other angle
  found. Halt and relaunch rather than let a taxonomy be built without it.

- `runs/*/llm_cache` is keyed on `(role, provider, model, system, user, schema)` and
  **not** on `max_tokens` — so token-budget changes do not invalidate it. Copying a
  cache directory into a new run is a legitimate way to skip replayable work.
  Web-research calls will still miss, because the fetched content differs each time.
- The offline stand-in produces a **degenerate taxonomy** (one node,
  `[offline-heuristic] code`), so `td_l1_name` is legitimately empty in offline runs.
  The code reports that rather than shipping a column of blanks.
- `completed_phases` uses an `operator.add` reducer — it cannot be pruned via
  `update_state`. Rewind by graph **position** (`as_node=<predecessor>`) instead.
- Verify a live run really used live agents: `run_summary.json` →
  `llm_usage.provider` must read `routed`, not `offline`.

---

## 4. Session (2026-09-07) — a multi-vertical assistant corpus, and the axis the drift report assumed

Two new exports arrived: `ai助手_Top1000query.xlsx` and `ai助手_随机1000query.xlsx`,
33,000 rows each, **33 first-level and 213 second-level categories in one file**.
Everything this project had been run on before was a single vertical.

### What the corpus actually is (measured before anything was configured)

| | measured |
|---|---|
| shared query strings between the two files | **159**, a Jaccard of **0.2%** |
| seven strings' share of pooled head traffic | **15.97%**, each appearing in **29–32 of the 33 categories** |
| acknowledgement family | 1,065 rows; **17.0%** of raw head PV, **28.9%** under equal-category weighting; median **30.1%** per category, **71.9%** in 生活和情感 |
| head PV in queries ≤6 characters | **72.4%** (the tail's mass is at 8–20 chars) |
| per-category top-N traffic floor | **3 (招商加盟) to 419 (书籍文档)** |
| raw PV held by 2 of the 33 categories | **55%**; one string (`变清晰`) holds **17%** |
| fiction-marker enrichment among risk hits | **2.0% of rows → 15% of hits**, a 7.5× lift |

Three conclusions, each of which changed a configuration decision:

1. **The two files are sampling STRATA of one period, not two periods.**
2. **The head is largely not queries.** `总结全文概要`, `变清晰`, `去水印`,
   `👌 好的，继续吧`, `嗯` — tool-panel buttons, suggested-reply chips and bare
   conversational turns. A string in 32 of 33 topical categories is not topical.
3. **Raw traffic is not comparable across categories** — the file is a union of 33
   censuses cut at 33 different depths.

### A failed method, reported rather than shipped

Reweighting to the population needs each category's total traffic. The
capture–recapture bridge — what share of the random sample sits at or above the
head floor — **fails**: 17 of 33 categories have ZERO random rows above their
floor, 28 have fewer than five, so the estimator returns 1,000,000 distinct
queries off a single row. **Cross-category traffic comparison is not recoverable
from these two files.** Same discipline as the abandoned power-law fit.

### The comparison axis (`data.comparison_axis`)

`ops/drift.py` never knew about time — it compares two groups' composition.
`report/zh_drift.py` did, in prose: 「不是趋势」, 「同月同日不等于季节可比」,
「时段性事件」. All false about a head/tail split, and **one inverts**:
「两期的抽样方式必须一致」 warns that differing sampling would masquerade as a real
change, but here the differing sampling IS the independent variable — a reader
applying it concludes the document is confounded when it is measuring exactly
what it set out to.

`time` is the default and is **byte-identical to the pre-change module**, verified
against the original file over seven payload shapes (full / no purity / clean
purity / no churn / no labels / empty class lists / no snapshot tags). One
16-character clause was dropped during the parameterisation and restored; the
diff caught it. `stratum` ships `分层对比_头尾结构差异.md` instead.

### What shipped

| | |
|---|---|
| Tests | **734** passing, exit 0; `ruff --select F src/qmine/ tools/` clean (was 716) |
| New config field | `data.comparison_axis: "time" \| "stratum"`, default `time` |
| New tool | `tools/prepare_assistant_corpus.py` — pools, tags the stratum, normalises weight within (stratum, category), flags acknowledgements |
| New tool | `tools/vertical_crosstab.py` — delivered classes × source categories; the payoff for pooling 33 verticals, which nothing was delivering |
| New profile | `configs/domains/ai_assistant_zh.yaml` — 12 seeds (overlap ≤3.8%, eight at 0), 8 risk categories |
| New config | `configs/live_ai_assistant.yaml` |
| New doc | `docs/AI_ASSISTANT_CORPUS.md` |
| Changed | `report/zh_drift.py` (`_AXIS`, `_caveats`), `graph/nodes/delivery.py` (deliverable name), `tools/check_domain_profile.py` (reads parquet) |

**The first corpus here with a real reference taxonomy.** Every previous export
carried `query_1st_category` holding one constant value — a filename, which p1
drops. `l1`/`l2` vary row to row and reach 100%, so `k_locator: auto` locates K
against them rather than against the 5.1%-reach phrasing seeds.

### Two rows a human should look at

- `minor_sexualisation` fires **once**: a self-identified 15-year-old asking about
  her own body measurement, which the export's taxonomy files under
  生活和情感/两性知识 — a general sex-education bucket with no minor-specific
  handling.
- `self_harm` fires 16 times and **zero are first-person** — fiction (老九门,
  喜羊羊), history (杜聿明), news, an abstract law question. On this corpus it is a
  pure false-positive generator. It is kept broad anyway: unlike `finance_zh`'s
  澳门 note, a false negative here is a person in crisis getting a plot summary.

### The live run, and why it was halted

`ai01` was launched full-mode and **halted by the maintainer at 12:29** — an
errand, plus the decision to run this corpus in `--fast` instead. Not a failure.
~28 minutes, **$2.31**, 10 priced calls, `provider=routed`. `p0`/`p1`/`p3`
complete; `p2a`'s calls all returned and are cached but the phase artifact was
never written. See §1 for the exact relaunch command and why `ai01`'s spend is
not transferable.

Four things it established that the offline smoke run could not:

1. **The reference taxonomy works as a reference.** `p1_reference_columns_declared`
   PASSED on `l1, l2`. The legacy-audit researcher, which does nothing without
   them, returned intents the source taxonomy lacks — `按字数要求生成文本 →
   word_count_writing`.
2. **The web-using researchers named this corpus's own problems** rather than
   generic ones: `续写虚构剧情并接续角色扮演 → continue_fiction_roleplay`,
   `对图像去衣或生成性化图像 → sexualised_image_request` — the latter matching a
   risk category seeded independently from measurement.
3. **The architect found the acknowledgement family without being told**, and
   described it as the measurement does: 「这些短句的功能是推进或结束当前对话状态，
   属于明确的会话管理轮次」. 24 nodes, 57 rules after merge. That is independent
   confirmation of this session's largest finding.
4. **The observer found the alpha decision is knife-edge here.** Two CONFIRMED
   checks: `contenders` held one member, so the stability tiebreak its own
   rationale claims never occurred; and alpha=0.0 sat outside the tie band by
   **0.0067 (~0.3% relative)** while scoring HIGHER on stability (0.6151 vs the
   winner's 0.5919). This is open question **8** ("the alpha decision is decided
   by the GRID, not by the corpus") appearing again on a sixth corpus.

### Two flags that are accepted and ignored (open question 0t)

Found while working out how to relaunch in fast mode, measured both ways:
`--fast` on `--resume` never reaches the config (the resume branch does not call
`_load_config` and applies only `run_root` and `reuse_taxonomy`), and a config
file's `mode: fast` is overruled by the CLI's non-None default. **Deliberately
not fixed** — the maintainer was away, and the correct command avoids both — but
it is the same bug `test_an_unset_cli_flag_does_not_overrule_the_config` was
written for, on a third flag.

### Not done / next

- **Relaunch as `ai02 --fast`.** Command in §1.
- CLAUDE.md is **225 lines** against its own 200-line target. It was 220 before
  this session; three table rows were added. Moving area-specific entries into
  `.claude/rules/` is overdue and was not attempted here.
- `tools/vertical_crosstab.py` has only ever run against an offline smoke run,
  so its family names have only been seen in `[offline-heuristic]` form. The
  naming join itself is exercised; the output has not been read on real names.

---

## 4. Session (2026-09-03) — pooled snapshots become a phase

Five pooled runs had already been produced by hand (`tools/pool_snapshots.py` +
`tools/drift_report.py`). This session turned that into pipeline functionality.

**Built.** `data.input_paths` / `data.snapshot_column` (additive);
`foundation._load_input` stacks and tags, refusing duplicate tags and warning on
schema mismatch; `build_frame(snapshots=)` carries the tag beside `weight` and
**never** as a reference column (declaring it as one would ask the K locator to
find a K separating 2025 from 2026 — the opposite of the shared frame the
comparison needs); `ops/drift.py`; `p10b_drift`; `report/zh_drift.py`.

**Wired into both modes deliberately.** A multi-snapshot run exists *for* the
comparison, so losing it to the cheap mode would defeat the point. The document
is a pure lookup over `drift_analysis.json`, which is what lets it ship in fast
mode without a model call. `_p11_fast`'s deliverable count is now derived from an
`_expected` list rather than hardcoded, so it cannot go stale again.

**Validated before shipping.** `ops/drift.py` was checked against the fin-pool
data by reproducing the earlier manual measurements exactly (`+12.802pp`, purity
0/54, median 0.507, jaccard 0.3743). Then rendered against real film-pool data
and READ — which is what caught both defects below.

**Corrected a claim I had written into the code.** The `drift.py` docstring and
the CLI comment both said two runs on *the same 10,000 rows* shared "0 of 35
class codes". Measured: `fin01` and `fin02` are **different** files (2025-07 vs
2026-07) with **20 and 19** classes and zero overlap. The real finding is
stronger and on-point; both sites now state it correctly.

**Adopted from the literature review** (`drift-phase-design` workflow, 4 agents):
total-variation distance as an interpretable magnitude, delta concentration (HHI)
to separate a broad shift from a single entity, plus two caveats — this measures
*prior-probability shift over a fixed taxonomy* and is structurally blind to real
concept drift, and same-date-one-year-apart is defeated by moving calendars
(lunar new year, exam/results dates, sports fixtures). **Its headline critique did
not apply**: it warned that page-view weights would destroy the significance
tests, but the module already refuses p-values on traffic share, and
`_z_two_proportion` takes row counts at its single call site.

**README overhauled (2026-09-03).** 905 → **661** lines. The cross-run `Results`
section was written, then moved to `docs/RESULTS.md` at the owner's request — it
was more for a first-time reader to digest than the README should ask. The README
keeps two figures (decision chain, churn-vs-drift) beside the sections they
illustrate; the four cross-run figures went with the tables. Nothing was deleted,
and `tools/run_evidence.py` / `tools/readme_figures.py` regenerate both. Evidence is now
cross-run (14 complete live runs, six corpora) rather than the single `live42`,
which was the one run whose report demonstrably failed. Two long literature
sections moved to `docs/WHY_NOT_A_PROMPT.md` and `docs/WHAT_ITS_FOR.md`; "Known
limitations" and "Status and contributing" deleted at the owner's request, with
six measured findings from the first rescued into §2 above.

Two new tools: `tools/run_evidence.py` (one comparable table over every complete
live run) and `tools/readme_figures.py` (the five cross-run figures, light and
dark, regenerated from that table). Four README claims were false and are fixed:
"625 tests" (711), the `qmine doctor` warning (that defect is fixed), 21 classes /
139 rules quoted under a `live44` pointer (20 / 162), and every bare `qmine`
command (the entry point is `.venv/bin/qmine`; `make install` does not touch PATH).
`Makefile:41` said `~$30` for a run that costs $5-$7.

**Cost figures were materially misleading and are now qualified.** 60.3% of
`live44`'s and 59.7% of `med04`'s tokens belong to roles whose model publishes no
price and fall back to a frontier rate, so `$61.09` and `$65.05` cover ~40% of
those runs. `live39` ($5.52) and `live40` ($7.01) are 0% unpriced and are now the
quoted anchors; `live38`'s $1.10 is a replay (519 of 577 calls cached).

**Licence contradiction resolved (owner decision).** `pyproject.toml` declared
MIT with no `LICENSE` file while the README said all-rights-reserved. The MIT
claim was **removed** from the packaging metadata and the README's conservative
wording kept; a comment at that spot warns against reinstating a licence field
without the matching file, since metadata is what tools read.

**A Chinese edition of the cross-domain analysis ships alongside the English one**
(`docs/DRIFT_ANALYSIS.zh.md`, 546 lines), with **Chinese-labelled figures** —
`tools/drift_figures.py --lang zh` writes a `*_zh.png` set using the same CJK font
stack as `report/zh_notebook.py`, and every figure was read back to confirm the
glyphs render rather than boxing. Verbatim primary-source quotes (the Baidu 6-K
sentences) stay in English on purpose.

Two guards keep the pair honest, because a translation is a second copy of ~310
numbers and copies drift:
`test_the_translated_analysis_carries_the_same_numbers_as_the_original` compares the
numeric content of both documents — allowing only an explicit, arithmetic-checked
万/亿 conversion table — and also pins heading and table-row counts;
`test_the_chinese_analysis_points_at_chinese_figures` refuses an English-labelled
figure inside the Chinese report. Both were mutation-tested.

**The sampling question is largely CLOSED (2026-09-04), by reframing rather than by
new data.** Two corrections to the earlier write-up, both material:

1. **A top-N-by-PV export is a CENSUS, not a sample.** It contains every query at or
   above its floor. So for any threshold T at or above both years' floors, the
   population {PV >= T} is fully observed in both years and comparing it needs no
   model and no assumption about how many queries exist. Exact results:
   finance **+35.5%** traffic (and its qualifying-query count ROSE), film/TV −28.1%,
   people −24.6%, medical **−59.8%**, education **−61.5%** — and the *count* of
   queries clearing the bar fell 55.9% (medical) and 60.2% (education). Those are
   measurements, not estimates. The earlier "algebraically undecidable" framing was
   too pessimistic and has been removed.
2. **The snapshots are single DAYS, tested not assumed.** `event_day` is one value per
   file and both fall on the 1st, which is also how monthly partitions are labelled.
   Two content tests settle it: the education files contain the EARLY admissions stage
   (分数线 366 queries) and **zero** 投档 and **zero** 开学/报到, which a whole-July
   aggregate must contain; and the 2026 film file has **zero** 决赛/半决赛/冠军/八强/16强
   despite the World Cup Final falling on 19 July.

**What remains open, quantified.** Only the sub-threshold region. For any decline to be
pure redistribution, the tail must have absorbed the lost traffic, and since each
hidden query carries less than T that sets a hard minimum: education **19,276** new
near-threshold queries, medical **17,166**, film/TV 13,056, people 11,863 — and that is
the generous case, with every new query sitting exactly at the bar. Each vertical's
entire observed above-bar population is only 4,000-10,000 queries. Inside the census
band all four declining verticals flattened mildly (Gini −0.021 to −0.042), the
direction dispersion predicts, while finance CONCENTRATED (+0.051) while growing.

**A method that failed, kept as a warning.** Fitting the rank-PV curve and
extrapolating past rank 10,000 does not survive its diagnostic: the exponent drifts
systematically with rank (medical 0.339 -> 0.765, education 0.300 -> 0.764), so it is
not a single power law, and two defensible fitting choices gave contradictory signs for
finance. `tools/` deliberately does NOT ship this.

**All five pooled runs now have drift reports, WITHOUT re-running anything
(2026-09-04).** `p10b` makes no model calls, so it is a deterministic function of
artifacts already on disk. `tools/backfill_drift.py` recomputes it from the delivered
labels plus the pooled source and writes into a NEW generation
(`runs/*-pool/gen02/`). Re-running would have cost money AND produced different
labels — the same 20,000 film rows run twice delivered 12 leaves and then 34.

**Verified three ways.** (1) Backfilling `filmdrift`, which executed p10b live,
reproduces its shipped artifact with **zero differences** —
`test_a_backfilled_drift_analysis_equals_what_the_live_phase_wrote` pins this.
(2) All five reproduce independent earlier measurements exactly (shared queries,
Jaccard, total variation, top mover). (3) Every number in every document traces to
its artifact; the only untraceable strings are the z threshold 1.96, the 30-row
floor, dates and section numbers.

**The cross-domain analysis is `docs/DRIFT_ANALYSIS.md`** (+ three figures from
`tools/drift_figures.py`). Its findings, in the order they matter:

- **The corpus is a top-10,000-by-PV cut and the cut MOVED.** PV floor and top-10k
  total move in lockstep (education −47.6% vs −48.1%), so "medical search fell 47%"
  is not established — only that the top 10k carried 47% less. Finance is the
  informative exception: floor +3.3% against total +34.7%, a divergence that means
  new traffic arrived above the floor.
- **A share can rise while the audience falls.** 药品功效与副作用查询 gained 5.44pp of
  share and lost **14% of its traffic**. In medical and education essentially NO
  class grew absolutely; every riser fell more slowly than its vertical.
- **Depth control flips a headline.** Truncating both years at the common floor moves
  education's 高等院校信息查询 from −2.82pp to **+0.85pp**, and the real riser is
  university rankings at +8.71pp. finance/film/people are robust (TV moves ≤0.004);
  medical and education are understated by 40-50% in the raw cut.
- **The AI-complexity hypothesis is not supported in the head.** Length did not rise;
  question share fell in all five but the shift-share decomposition shows it is MIX,
  not behaviour (finance −5.78pp total = −6.01 mix + −0.24 within), and medical
  became MORE question-like within its classes (+1.46pp). The sharpest test — does a
  class's question-likeness predict its share loss — fails in all five (best: finance
  ρ=−0.271, p=0.054). Scope: the tail is absent by construction and untested.
- **One vertical's biggest finding is a football match.** 2026-07-01 was in the World
  Cup Round of 32 (28 Jun-3 Jul), live on CCTV-5 (92 of 104 matches) — both verified
  against primary sources. cctv5 PV 8.3×; 8 of the top 12 film/TV queries. It is ~35%
  of that vertical's drift (TV 0.203 → 0.131), and removing it UNMASKS Korean content
  (+4.06 → +4.82pp).
- **The people phrasing shift is real** and survives every control: 资料 6.35% → 18.03%
  of rows, bare short names 70.1% → 50.8%, while the interrogative 是谁 does not move.
  Two 2025 news figures (陈小江 11.19% of that snapshot, 马兴瑞 5.74%) explain only 13%
  of the drift, and removing them makes the shift stronger.

**The research pass (72 agents) then corrected four things in the first draft**, all
now folded in:

- **Finance's growth is intensity, not breadth.** I inferred from the low delta-HHI
  that many new tickers arrived. Measured at a common floor: distinct queries +3.2%,
  PV per query **+31.3%**. Same queries, busier. Several top risers (英伟达, 美光科技)
  are US-listed, which a domestic account cannot buy.
- **The gaokao calendar is EXCLUDED, not merely unverified.** Beijing (6/27-7/1),
  Jiangsu (6/28-7/2) and Shanghai (7/1-7/2) application windows were identical in both
  years. So education's **-48% has no established cause** — the cohort decline is 14x
  too small and the platform-wide contraction 4x. That is now stated as the largest
  unexplained fact in the dataset.
- **The two AI hypotheses were conflated.** H1 (AI makes queries more complex) and H2
  (AI absorbs question-shaped demand) predict opposite things. H2 is refuted
  everywhere; H1 is refuted in four verticals and **weakly SUPPORTED in medical**,
  whose traffic-weighted length rose in both cuts and whose within-class question
  share rose +1.46pp. Medical is the open lead, not a refutation.
- **The sampling ambiguity is algebraic, not merely practical.** A uniform demand fall
  and a constant-demand dispersion apply the identical transformation to everything in
  a fixed-N window, so floor and total move together under both and every
  scale-invariant statistic is unchanged. Only a quantity from outside the window
  breaks the tie — the cheapest being total PV per vertical per day.

**Verified external anchors (primary sources):** Baidu App MAU 735m -> 644m (-12.4%,
SEC 6-K, exactly this window); CNNIC search users 877.82m -> 782.06m (-10.9%); MoF H1
2026 securities stamp duty +97.3% YoY; CSDC 20.16m new A-share accounts (+60%); CMG
holding World Cup rights with CCTV-5 carrying 92 of 104 matches, Round of 32 running
28 Jun-3 Jul so 2026-07-01 was a match day. A background contraction of ~11-12% is
therefore real and documented — and medical/education fell **four times** that, so
those are vertical-specific, not industry weather.

**Also ruled out** (each with a failed prediction): query-suggestion reshuffling
(collapsing token-reordered synonyms leaves TV unchanged to 4dp); 限韩令 easing (the
Korean class's largest query is a *Thai* drama); one blockbuster driving the streaming
collapse (哪吒2 bounded at <=0.83pp of -13.62pp); short-drama displacement (短剧 rows
fall 158 -> 34 at a matched floor); a generic live-TV shift (CCTV-5 +719% while
CCTV-6 -31% and CCTV-8 -72%).

**The search cap applies to WebSearch only; WebFetch still works**, and the agents
left URLs. Verified directly afterwards: Baidu's two 6-Ks (*"In June 2025, Baidu
App's MAUs reached 735 million"* / *"644 million in June 2026"* = −12.4% over exactly
this window, plus online marketing revenue already −15% YoY in Q2 2025); and the
MoF's own page (「证券交易印花税1549亿元，同比增长97.3%」). Both are now `[verified]`.

**Education's calendar question was closed with internal arithmetic instead of more
research.** Admission-related queries are only **6.4%** of 2025 education PV and
**8.6%** of the decline — had they gone to zero the vertical would still have fallen
**46.9%**. So no exam-calendar shift of any size explains it, and the 2026 provincial
dates (which I could not reach) do not need resolving.

**A per-vertical accounting of what the identified causes actually cover:** film/TV
**103%** (the five named 2025 dramas lost more than the vertical did net), people
**79%** (two political-news figures), medical **3%**, education **7%**. Medical and
education are ~95% unexplained and no external series closes the gap — the documented
platform contraction is 11-12% against their 47-48%. Given the algebraic
indistinguishability above, the likeliest resolution is that their gap is not a demand
fall but a change in the window. One number decides it: total PV per vertical per day.

**The drift phase has now run live** (`filmdrift`, 影视 pooled 20,000 rows, routed,
fast): **18 phases** (17 + p10b), 217 calls, **$3.91**, 1.97 h, `verify_run`
**21 PASS / 6 N/A / 0 FAIL / 1 SKIP**. It shipped `快照对比_漂移分析.md` from a real
weighted corpus and independently reproduced the figures measured by hand from
film-pool — 2,841 shared queries, 16.6% Jaccard, 36.7% of 2026 traffic on queries
that existed in 2025. `p10b_snapshots_share_one_frame` WARNED on 1 group, which is
the designed behaviour: a prompt to look, not a block.

**Reading that document caught a fourth defect.** Its preamble carried the same
retracted "same 10,000 rows / 0 of 35 codes" claim that had already been corrected
in `ops/drift.py` and `cli.py` — a THIRD copy, and the only one that reaches a
reader who cannot check it against the repo.
`test_the_pooling_rationale_states_the_measurement_that_was_actually_taken` now
pins all three sites.

**And it produced the controlled experiment the pooling argument was missing.**
`film-pool` and `filmdrift` ran the SAME 20,000 rows — corpora verified
byte-identical and in the same order — through the same config chain in the same
mode. They delivered **12 leaves / 12 families** and **34 leaves / 22 families**:
2.8x on identical input. The earlier fin01/fin02 argument compared two *different*
files, so it confounded corpus change with run variance; this does not. It is the
decisive evidence for never diffing two runs, and it is now the pooling argument's
primary citation.

**A third defect, found only by RUNNING it.** The snapshot tag is
low-cardinality text, so `_label_like_columns` reported `_snapshot` as an
undeclared legacy label on every pooled run — and that warning tells the operator
to pass it via `--reference-columns`, which is exactly what
`test_the_snapshot_tag_never_becomes_a_reference_column` forbids (it would ask the
K locator to find a K separating 2025 from 2026). No unit test could have caught
it: the column does not exist until the pipeline adds it. `skip` now excludes it;
the gate reads PASSED — "no reference label columns, and the corpus offers none"
— on both an offline and a live two-snapshot run.

**Controls, because a count means nothing on its own.** An offline+smoke+fast
two-snapshot run scored 17 PASS / 6 N/A / 4 FAIL / 1 SKIP. A **single-input**
run of the same corpus and flags — which never enters p10b — scored the
**identical** 17/6/4/1, including the same family-naming FAIL. All four are
offline-stand-in artifacts (`provider='offline'`, `[offline-heuristic]` prose, an
audit that covers few families), none drift-related. Verified separately that
`_family_display` does not leak: `labels_full.csv` still carries integer
`bu_family_final`.

**Two defects the first real render caught.** The concentrated bucket was
labelled 「疑似单一事件」 and the first case refuted it (cctv5 = one entity across
many phrasings, not one event) — it now names the measurement and ships the
evidence queries. And the family axis rendered bare ids; `_shape.family_names`
now supplies names by leaf-membership join.

**Tests.** +12 in `tests/test_fast_mode.py`. One of them failed on first run and
the fixture was wrong, not the code: it asserted "a pure base-rate change is not
drift" on a frame whose two snapshots had genuinely different compositions.

**`tools/pool_snapshots.py` and `tools/drift_report.py` are superseded but kept**
— the first built the five pooled runs on disk, the second is an out-of-pipeline
check on p10b and the only way to get drift figures for runs that predate it.

**Method note worth keeping.** `inspect.getsource` re-reads the file from disk at
the imported function's line numbers, so **editing source during a suite run
produces false failures**. Three `test_render_command.py` failures were attributed
to this twice before it was confirmed; the second time I contaminated the
verifying run myself by editing `cli.py` while it was in flight.

---

## 4. Session (2026-09-01) — live44, and the queue cleared

`live44` completed: 17/17 phases, `provider=routed`, 841 calls, $61.09, 9.81h,
κ 0.880 on n=3000, 53 leaves / 23 families delivered, `verify_run` 26/0/2 against
live42's 20/6/2. Then the whole defect queue was fixed. **625 → 642 tests.**

**Every defect was found by reading, not by the harness.** `verify_run.py` scored
live44 clean on all 26 applicable checks while seven real defects sat in the
deliverables. Two of them the pipeline had already found and thrown away.

### The compounding failure

The delivery auditor was shown **39%** of the deliverables (`budget_text` cut
142,957 of 232,957 chars out of the MIDDLE), found 3 defects in that third, and
had **2 of the 3 deleted** because the citation resolver only knew `artifacts`
while the auditor had also been handed `gates` and `findings`. The two deleted
were both real: the index claiming 21 L1 classes against a taxonomy of 20, and
one quantity published with two values. A last line of defence reading a third
of the evidence and losing two thirds of its conclusions.

### Five of my own diagnoses were wrong before they were fixed

Recorded because the pattern repeated: **measure one thing, assert a cause about
another.** Each was checkable in under a minute.

1. "Researchers cost most and retrieve least" — they are *deliberately tool-free*
   (`RESEARCH_ANGLES`, `web: False`), and returned the MOST candidates (12 each).
2. "P7 drops risk findings" — all ten `risk_report` clusters were isolated by p8.
   They travel via the risk sentinel, not `audit.risk_findings`. The log said so.
3. "A shared client-side deadline" — no shared deadline. `max_repair+1 = 3` outer
   attempts x SDK `max_retries=2` = **9 HTTP requests**, and 9 x 292s = 2,628s
   against the 2,638s observed.
4. "Reasoning made rules verbose" — density is identical (204.7 vs 207.5
   chars/rule). live44 simply produced more rules, and live42 sat at 86% of the
   budget unnoticed.
5. "The referee block would drop 85%" — wrong density applied. It is at 72%.

The ECE null landed the same lesson: a perfectly calibrated model at n≈5,800
still shows ECE ≈ 0.0074 ± 0.0028, so live42's 0.023 was ~5 sd out and was never
the clean baseline I had been comparing live44's 0.065 against.

### Fixed

| area | change |
|---|---|
| truncation | `budget_units` — whole rules/documents, count in log AND in-band, plus an explicit "do not describe this as the complete set" |
| attribution | the `label=` test found **9** unlabelled budget calls, incl. the narrator's fact sheet and the observer's own decisions/gates |
| observation door | `citable_namespace` feeds resolver AND check evaluator — widening one alone would demote a measurable claim to advisory |
| retries | SDK `max_retries` 2 → 0; per-tier throughput (researcher 585s → 1560s, maintainer 292s → 780s) |
| calibration | `ece_noise_floor` + `p2c_calibration` gate, against the run's own null, never a constant |
| kNN flags | nearest-first, full neighbourhood published — the 6-of-k sample manufactured false confirmations |
| safety booleans | `coherent` / `risk` / `risk_isolated` tri-state; they defaulted to the reassuring answer |
| family names | `FamilyNamerAgent` names the DELIVERED partition; `混合·主要成分「X」N%` is now only a fallback |
| duplicates | `merge_leaves` prescription + executor; the auditor must give every listed pair a disposition |
| `doctor` | `detect()` across all providers — it checked `ANTHROPIC_API_KEY` while the project routes to DeepSeek/Zhipu/Qwen |

### Deliberately not done

- **Did not loosen the English-prose detector** for the third English string. Its
  table-row and code-span exclusions are deliberate; widening it would flag every
  identifier — the documented grounding-false-positive trap. Fixed the source.
- **`p2c_calibration` is warn-only.** A new blocking gate on a metric nothing has
  ever gated would halt the next run on a pre-existing condition.

## 4. Session 1 (2026-08-18) — deliverables, providers, guide repair

### Deliverables now match the reference documents

- **`report/zh_figures.py` (new)** — the six-figure suite modelled on
  `K12_Embedding_Attempts_Comparison.ipynb`: K-sweep three-panel, α-decision with a
  phrasing-vote secondary axis, algorithm-battery scatter, embedding-space
  projections, per-intent split with `exp(H)` effective-family counts, and the
  uniform panel as bars with advisory metrics hatched. The notebook had **one**
  figure before; the reference carries six.
- **Figure production unified.** The report and the notebook each drew their own
  K-sweep, α, UMAP and panel charts — two different pictures of the same number in
  one deliverable. The notebook now executes first and its figures fill those slots,
  with `ops/viz` as fallback. Guarded by `test_one_figure_per_quantity`.
- **Both routes are equal citizens in `labels_full.csv`.** Bottom-up shipped 8
  columns including a margin and an ambiguity flag; top-down shipped 2 bare codes.
  Top-down now also has `td_l1_name`, `td_user_need`, `td_confidence`, `td_margin`,
  `td_ambiguous`, `td_decided_by` (`rule` vs `model`). `td_ambiguous` uses the same
  threshold as `bu_ambiguous` so the word means one thing.
- **`route_crosswalk.csv` (new)** — per bottom-up family: dominant top-down L1, its
  share, classes touched, `exp(H)` effective classes, and a verdict of
  `routes agree` / `partial overlap` / `routes disagree`. Neither route is an answer
  key, so the useful artifact is a map of where they concur.

### Gold-set sizing — a real deviation from the playbook, now fixed

`gold_sample_size` was hardcoded at **600**. The playbook says
`分层抽 3,000-5,000 条` (line 191) and separately `<1 万条 → 金标比例提高`
(line 119) — a small corpus needs a higher *proportion*, not the same absolute
count. A constant satisfied neither.

`config.gold_size_for()` now derives it: 3,000 on a 12k+ corpus, a rising share
below 10k, capped so gold can never become most of the corpus. Pin
`taxonomy.gold_sample_size` to an integer to force a cheap run.

**`p2a_pilot_agreement` now exists.** It was in the blocking-gates list and emitted
by no node. 50 queries, four LLM calls, hard stop below 85% agreement, reporting the
top confused pairs. This is what makes a 3,000-row gold set affordable — it catches
an ambiguous guide before the expensive annotation, per
`一致率 <85% 则回炉改指南/裁决规则, 而非直接开标`.

### The guide-repair loop (playbook `达不到先修指南再重标`)

Was absent entirely: κ was measured once, *before* the referee's rules existed, and
the gate's own `remediation` string said "fold the referee's rules into the guide and
re-annotate" — printed, never executed.

Now, when κ misses: find boundaries the referee resolved **inconsistently**, settle
each from rows **both annotators agreed on** (`discriminating_markers`, greedy set
cover, deterministic tie-break), add a `boundary_default` for marker-less queries
only where the agreed rows lean ≥75%, rewrite the guide, and re-annotate a **fresh
disjoint sample**.

**Known open question — see §4.** Whether this actually raises κ is still unproven.

### Provider and infrastructure fixes

| Defect | Effect | Fix |
|---|---|---|
| `resolve_base_url` cached a host after *every* probe errored | One transient failure pinned the process to a host that had just 401'd — 48 auth errors, an annotator lost 24 batches mid-run | Never cache a host we have evidence against; serialise the probe behind a lock |
| Global `max_tokens=16000` overrode per-role budgets | Taxonomy architect truncated at exactly 16001 tokens, twice | Role budget is authoritative, bounded only by the model's published ceiling |
| `_native_schema_is_broken` only matched `ValidationError` | DashScope's `response_format unavailable` and `must contain the word 'json'` were never learned; every call wasted an attempt | Both patterns added |
| Annotator batch failure → `return {}` | 400 rows became `UNLABELED` silently | Retry ×3 with backoff, then report the loss loudly |
| `agreement()` counted `UNLABELED` as disagreement | Charged infrastructure failures to the methodology; cost 0.008 κ | Excluded and counted as `n_unscored_unlabelled` |
| `p2b_kappa` reported a verdict at 33% coverage | "κ 0.813" read as a judgement on the guide; it described the survivors | `min_annotation_coverage` (0.90) → `MEASUREMENT UNSOUND`, neither pass nor fail |

### Correctness fixes found by running, not reading

- **`stratified_sample` returned index *labels* in its stratified branch and
  *positions* in the other two.** They coincide on a `RangeIndex` — which every
  caller had always passed — so passing `df.iloc[unseen]` blew up with
  `IndexError: index 11410 out of bounds for size 11400`. Fixed at the contract:
  positional in all three branches.
- **`select_active_learning_batch` called `len()` on a scipy sparse matrix.** Round-2
  active learning had *never once run* in this codebase; the exception was swallowed
  into a one-line log message. Fixed, plus the caller's `diversity_fraction=0.0`
  workaround removed — the diversity pass works fine on sparse, it was routing around
  a bug one line earlier.
- **The notebook executed against the wrong Python.** `kernel_name="python3"`
  resolved through the user's Jupyter kernelspecs to an unrelated project's venv
  lacking `pyarrow`; cell 1 raised and every figure below silently never drew. Now
  pinned to `sys.executable`.
- **`_dedupe_rules` was a rule-set shredder.** It compared rendered `when` sentences,
  where two markers for one boundary differ by ~2 characters in 45 (0.957 similar).
  It flagged every legitimate discriminating pair as a contradiction and withheld
  *both* halves — **9 of 41 rules survived**. Now compares structured
  `(class pair, trigger)` keys. Measured live tonight: **28/28**.
- **Rules could name a class that does not exist.** `R12 → 选 EXOD_INFO` against a
  taxonomy declaring `EXAM_INFO`, rendered verbatim into both annotators' prompts.
  Now validated at 2a and repaired by a dominance test.
- **A failed referee batch stamped `adjudicated=True`** on rows with no referee
  behind them, silently favouring annotator A. Now left visibly unresolved.
- **Resume could not tell a crash from a refusal.** Three separate gaps: it replayed
  the halt instead of retrying; clearing the flag still skipped the phase because the
  checkpoint stores *position*; and a cleared-but-unrewound checkpoint reported
  **success for a pipeline that stopped at phase three**. Now `halt_kind`
  (`crash`/`gate`/`review`) plus position derived from `completed_phases`.

---

### Live attempts — what each cost and taught

| run | outcome | cost | lesson |
|---|---|---|---|
| live01–02 | died at P2a | ~$7 | fenced JSON; 16k token truncation |
| live05 | crashed in the repair round | $6.72 | `stratified_sample` label/position contract |
| live10 | halted p2b, κ 0.813 on **199/600** | $7.39 | poisoned region cache; κ was an artifact, not a result |
| live20 | halted p2b, κ 0.831 on **596/596** — a sound measurement | $16.67 | first clean repair readout: **Δκ = −0.002**, i.e. no effect |

Roughly **$38** spent on live runs. Every failure was a distinct real defect, all
fixed with regression tests.

---

## 5. Session 2 (2026-08-19) — decision architecture, audit trail, portability

**Verdict on the playbook: mostly right, targeted repair — not redesign.** Five
independent audits agreed. The phase ORDER is correct and a joint grid search is
measurably *worse* (held-out ARI 0.653 vs greedy 0.739 at 3.4-4.7x the cost;
measured directly: 819 s for a 50-cell (alpha x K) surface on only 4k rows, and
greedy's alpha was already optimal at the final K). The defect was never the
sequence — it was **reading single noisy draws as truth**.

### K selection was noise
Replay stability's seed-to-seed sd is ~0.10; the gaps between adjacent K are ~0.05.
Four draws at one K gave 0.63 / 0.60 / 0.38 / 0.69. A tie-aware selector returns
**the whole grid**. It is also degenerate: K=2 scores ARI **1.0000** on both the 8k
and the real 50k corpus, so only the `k_sweep` list's hardcoded lower bound stood
between the pipeline and a two-way split. `expected_family_range` never constrained
anything.

**Now:** stability only *rejects* (a reproducibility floor — the role the literature
gives it); K is *located* by **AMI against the phrasing groups**, the one metric here
with a two-sided penalty and therefore an interior optimum (0.427 at K=2 → 0.724 at
K=25 → 0.664 at K=100), and **~10x more precise** (sd 0.005-0.023). It yields a
unique winner where stability ties everything. Runs now ship a **tie set** — the
honest answer, and the "several equally-good results" deliverable.

Dead ends not worth repeating: raw fragmentation is *structurally* monotone in K
(rho +0.97…+1.00 in all 13 runs) so it prefers K=1; the chance-adjusted version
(`adjusted_template_fragmentation`, implemented) prefers K=∞. Neither locates K.
Both are valid at *fixed* K, which is why alpha selection was always sound.

### Other confirmed defects, fixed
- **The panel measured the wrong object.** `stability_ari` was `replay_stability(X, k)`
  — corpus and cluster count only — so the "decisive" number on the delivered leaves
  described a fresh KMeans run and was pessimistic by ~0.25 ARI. Now
  `partition_stability`: half-sample centroid replay on the actual partition
  (leaves 0.893 ± 0.007 vs the 0.640 previously reported).
- **The algorithm battery was decorative** — `build_hierarchy` hardcodes KMeans, and
  the report announced `gmm_diag_k15` as the winner while arguing why KMeans is
  right. Now a falsification probe, and the report says so.
- **The pilot gate was dropped on the floor** — `deps.gate(...)` called without
  capturing the return, so it could never halt anything. My own
  `declared_gates_never_evaluated` diagnostic named it in five runs.

### The audit trail is now IN the deliverable
Previously the run recorded 7 decisions, 10 gates, 5 prescriptions and 332 agent
calls, and the report rendered roughly half the decisions and none of the rest.
Added as sections 9-11, in execution order:
- **§9 全流程决策链** — every decision with candidates, winner, who decided, decisive
  metric, full rationale, and the rejected options with their numbers. Plus
  `fig_decision_chain` (candidates → survivors per step).
- **§10 质量门总账** — every gate with observed vs threshold and its remediation, plus
  gates declared blocking that never fired. Plus `fig_gates` (headroom, normalised).
- **§11 治理台账** — every prescription's final disposition.

All authored rationale/remediation prose is translated via `report/i18n.prose()`,
guarded by `test_every_authored_rationale_reaches_the_reader_in_the_report_language`
which fails on any untranslated line reaching a Chinese report — it caught a string
I added minutes later.

**State:** 134 tests pass; report 15 sections / 664 lines / **0 lines of English**;
11 figures; notebook executes with 0 errors.

---

## 6. Session 3 (2026-08-20) — CLAUDE.md, portability, and the first live 50k run

### Pre-flight paid for itself
A **full-50k offline run** (`runs/preflight50k`) completed all 17 phases in 21.7
minutes, clean — the first time the whole pipeline has run at real scale with the
current code. Before that, an audit of `live20`'s per-role usage caught two defects
that would have failed the paid run:
- **`taxonomy_architect` at 99% of its cap** (23,759 of 24,000) — the role that had
  already truncated twice.
- **`annotator_b` exceeding its cap outright** (11,910 vs 4,500), because reasoning
  tokens are billed and capped as output while its partner emitted 1,439 for the
  same task.

### The live run: $3.02, halted at p2a, and worth it
`live30`, full 49,999 rows, real agents. The **annotator-ceiling pilot** — built
that morning on reasoning alone — returned its first real measurement:

```
pilot: kappa 0.761 (95% upper 0.814) on 200 queries
       annotator self-consistency kappa 0.8997 (0.8457 of ceiling reached)
top confusions: QUERY_POETRY_TEXT × QUERY_WORD_USAGE, EXPLAIN_WORD_MEANING × QUERY_POETRY_TEXT
```

Two annotators agree at 0.761; **one agrees with itself at 0.900**. So the 14-point
gap is guide ambiguity, not model noise — the *fixable* branch, established rather
than assumed. Yesterday the same situation cost $16.67 to reach a worse-informed
conclusion.

**Root cause:** the architect shipped **19 classes and 1 adjudication rule**. No
truncation (20,441 tokens against a 42,000 cap) — it simply followed
`"Aim for at least {{MIN_RULES}} rules"`, which is exactly the soft phrasing
Anthropic's own guidance says gets ignored.

### Three fixes, each of which exposed the next
Found by ~$0.05 probes against `live30`'s stored submissions, not by paid runs:
1. `"Aim for"` → **`YOU MUST`**. Result: 1 rule → 24. But classes went 19 → **2**,
   and the rules named a dozen classes that did not exist. Emphasis raises
   adherence to the emphasised instruction *and lowers it for what competes*.
2. Made the two requirements **joint**. Result: 20 classes, 22 rules — but only by
   emitting **42,001 tokens**, hitting the ceiling, and recovering via the
   plain-JSON repair path.
3. **Split the call**: architect writes classes, a new `RuleWriterAgent` writes
   rules *shown the finalised class list*. Probed: 19 classes, 22 rules,
   **22/22 naming a real class**, 731s. The invalid-rule failure is now structurally
   impossible rather than discouraged.

### Also this session
- **`timeout_seconds` is derived from the generation cap.** Raising the architect's
  cap to 42,000 left it 420s to emit them — at ~49 tok/s measured, about half what
  it needs. Two independently-tuned constants that only make sense together.
- **Three gates de-imported from K12.** Coverage gates on *rows* not a share (the
  e-commerce corpus was flagged for being more templated); held-out is bounded by
  the partition's own reproducibility; coherence reads the **weak tail**, not the
  mean (a tree with 16 good leaves and 4 incoherent ones passed on average).
- **`p2a_taxonomy_shape` now has three outcomes**: missing rules halt, a wildly-off
  class count halts, a near-miss warns.
- **`CLAUDE.md` written** (106 lines) plus three path-scoped `.claude/rules/` files.
  Verifying it caught two globs that matched **nothing**.

---

## 7. Session 4 (2026-08-21) — the operator's view, and a gate that could be won

Three live runs today (`live31`, `live32`, `live33`), 176 tests (was 141).

### The dashboard existed and had never been rendered
241 lines, on by default, zero tests, never once looked at. Rendering a real
`run.log` through it found six defects in twenty minutes: the two panes never
split (`Columns` sizes by content, so it stacked); Rich ate `researcher[log_reading]`
as markup so three agents showed as three identical lines; `P3a/b/c` were emitted
while `p3` was declared, so that row could never light up; gate notes were cut
mid-word at 70 chars; metric labels were sliced mid-token; and a *halted* run left
its last phase spinning at ◐, because a blocking gate returns rather than raises.

Then rendering the **live** log found a seventh the recording could not: eight
concurrent batches fail identically in one second, each wrapping to two display
lines, so one benign already-handled error filled all six activity slots and
pushed out the progress. **Build against a recording, re-render against live
traffic** — a clean run has no failure *concurrency*.

Also added: `qmine watch RUN_ID` (the panel reads `run.log`, so a run can be
launched detached and still watched), per-phase explanations, and an agent panel
showing role · model · elapsed · out-tokens · *what it returned*.

### `run.log` did not exist at all
The CLI quieted the **logger** to give the panel the screen, and there was no file
handler anywhere — so choosing the pretty view meant choosing to have no record.
Levels belong on handlers. Fixing it exposed that `open_run()` had **zero callers**:
34 lines duplicating the resource setup of the three functions that are real. My
first fix went into it and did nothing.

### The cost ledger was optimistic exactly where things go wrong
Three instances of one pattern: `ToolAgent.run` recorded a hardcoded
`output_tokens=0` for every tool loop — so the budget ceiling was blind to the one
path that *iterates*; `complete()`'s failure branch recorded zero for responses the
provider had already generated and billed; and `qmine models` still assumes one
output-tokens-per-call figure across roles (annotator_a 1,439, annotator_b 12,435).
First two fixed. Tool-loop turns also now write a cache entry and a transcript
entry — the web-researching agents were the only ones leaving no record of what
they said, which is a poor property for the agents citing pages nobody else saw.

### `TaxonomyNode.adjudication_rules` was write-only
Declared to hold rule *ids*, filled by the models with rule *text*, and read by
**nothing**. `_render_rules` rendered only the top-level list. Measured recovery:
`live30` would have shown the annotator **42** rules instead of 1; `live31` 70
instead of 46. This reframes `live30` retroactively — its κ 0.761 was achieved with
one visible rule, and the shape gate's "1 adjudication rules" was accurate about
what reached the annotator while 55 sat unread.

### The three taxonomy defects, measured
Replaying all 600 pilot labels out of `live31`'s cache separates intrinsic
ambiguity from fixable guide gaps — a query where annotator A disagrees with
*itself* is a boundary not in the data. That split 57 disagreements into **36
structural / 27 guide**, and attributed ~half to three corpus-independent defects:
overlapping siblings, siblings cut on different bases, and a catch-all defined by
content. The architect prompt was **requiring** the third ("a catch-all must be
defined by what it *is*") and simultaneously telling the architect both to write
and not to write adjudication rules.

A first hypothesis — that the LOOKUP/EXPLAIN *axis* was the problem — did not
survive its own significance test (z≈1.0). Recorded because it looked convincing.

### The κ gate was measuring its own confidence interval
The playbook's ≥0.9 came with "K12 达 0.966" — a floor beneath what *that* project's
annotators reached. Ours self-agree at 0.883. Worse, the gate tested the *upper*
bound, so the bar moved with the pilot size:

| pilot n | κ demanded |
|---|---|
| 50 | 0.801 |
| 200 | 0.857 |
| 3000 | **0.890 — above the ceiling, unwinnable** |

Now two independent conditions: **annotator fitness** (`ceiling ≥ 0.80`, the
conventional reliability threshold applied to the quantity it describes) and **no
significant recoverable slack**. 0.90 is reported as the playbook's aspiration.

### P2a can now redraw and re-pilot
`TaxonomyRedrawAgent` is shown the current taxonomy and the pairs one annotator
could not reproduce, and told to merge or re-cut *those* and leave the rest
byte-identical — not the architect, which rebuilds from evidence and re-rolls the
classes that were fine. Bounded at 2 rounds, reverts any redraw that lowers κ.

Extracting it into `_redraw_until_stable` to make it testable immediately found
two bugs that lint and the offline run both passed: the revert filtered the
*redrawn* nodes by the old codes (keeping new definitions under old names), and
the `return` sat inside the `for`, so the loop exited after one iteration on
success and returned `None` on any break.

### Results
| | live30 | live31 | live32 |
|---|---|---|---|
| pilot κ | 0.761 | 0.688 | **0.777** |
| ceiling | 0.8997 | 0.8023 | **0.883** |
| citable rules | 1 | 46 | **106** |
| prescriptions | 0 | 0 | **12** |

Predicted κ≈0.84 / ceiling≈0.90 from the defect attribution; got 0.777 / 0.883.
Direction right, magnitude about half. All three halted at `p2a_pilot_agreement`.

### Evening: OpenRouter, and five more live runs

An OpenRouter key was added mid-session. It changed more than it looked like it would.

**Every cost figure this project ever reported was fiction.**
`UsageLedger.estimated_cost_usd` hardcoded `in_rate=3.0, out_rate=15.0` per million —
frontier rates — while runs were on `deepseek-v4-flash` ($0.44/$1.32) and
`qwen3-next-80b` ($0.15/$1.20). `live32`: reported **$4.68**, actual **$0.67**. The
ledger now prices from the routing plan, and names any role priced by the fallback
rather than letting a guess read as a measurement. Fixing it exposed a second bug in
the fix: roles arrive suffixed (`researcher_log_reading`) while the plan is keyed on
the base role, so exact-match lookup dropped four roles back onto frontier rates.

**Independence was checked on the gateway, not the lab.** With an aggregator in the
pool `zhipu/zai/glm-5.1` and `openrouter/z-ai/glm-5.3` read as independent and are
one lab. `lab_of()` now resolves the originating lab, applied to the primary choice
AND the fallback chain — a fallback within one lab is one outage and one architecture.
The referee must now differ from BOTH annotators, not just annotator_b from annotator_a.

**Three things the price-as-capability proxy did, only one of which was intended.**
Removing price from `_assign_tiers` was tried and reverted three times: each attempt
let something worse win every role — a date stamp parsed as version 28
(`qwen-flash-2025-07-28`), then `:free` variants, then `openrouter/auto`, a
meta-endpoint. Price was also silently excluding those. Those exclusions are now
explicit in `_eligible` (`:batch`, `:free`, preview, unpriced) and the ordering stays
priced. The narrow fix for the real complaint — a newer model rejected for being
cheaper — is a same-LAB generation upgrade after scoring.

**Failover exists now**, and cost three attempts to get right. A `402 Insufficient
Balance` mid gold-annotation took twelve batches while two declared fallbacks sat
unused. Then the classifier killed a whole provider on
`CompletionUsage(completion_tokens=4013, prompt_tokens=8402)` — "4013" contains "401".
Then a truncation was misread as a dead provider when the remedy was more room.

**`glm-5.2` truncates in native structured-output mode and does not need to.**
Measured four times: truncates past 12,000 tokens natively, completes in ~5,300 on the
plain-JSON path — the same answer for less than half the tokens. A truncation now
raises the cap AND abandons native mode, both keyed by MODEL so one discovery serves
every role. Previously five researchers each paid ~180s to learn it separately.

**Latency: the gateway, not the model.** Same `deepseek-v4-flash`, same phase —
OpenRouter median 67.8s / max 429.9s, direct 83-98s with no tail. The router now
prefers the DIRECT route when the same bare model name is reachable both ways. Costs
~70% more on the estimate; buys back roughly two hours of wall-clock on a gold phase.

**The gate proceeds once its remedy is exhausted.** `live35` sat at kappa 0.844 with
0.080 of significant slack and a redraw that had run and failed. Halting there asks
the operator to do by hand what the pipeline just could not, while 0.844 is above the
reliability floor. It now passes — narrowly: the redraw must have RUN and FAILED, and
kappa must clear the floor. The message and `run_summary.json` both record that it
proceeded with residual slack.

**Re-running a run id: three separate traps, all new because we had never done it.**
The CHECKPOINT carries `halted=True` and exits in 3.1s without re-reaching the gate —
delete `checkpoints.sqlite`, keep `llm_cache/`. The TOOL path wrote cache entries and
never read them, so the two web researchers re-fetched live pages and cascaded a miss
through everything downstream (this is what defeated the `live33` resume too).
And `qmine watch` treated ANY `run_summary.json` as "finished", so it exited within
seconds of attaching to a re-run. All three fixed; the researcher fan-out now replays
in **one second** against ten minutes.

**Also:** `--config` silently discarded `--domain`, swapping k12_zh for `generic` and
halving template coverage in a deterministic phase with no error anywhere — caught
only because 18,298 had been read three times that day.

### Results across five live runs

| | live30 | live31 | live32 | live33 | live35 |
|---|---|---|---|---|---|
| pilot kappa | 0.761 | 0.688 | 0.777 | 0.839 | **0.844** |
| ceiling | 0.900 | 0.802 | 0.883 | 0.861 | **0.924** |
| citable rules | 1 | 46 | 106 | 92 | 58 |
| prescriptions | 0 | 0 | 12 | 12 | 12 |
| redraw fired | — | — | — | no | **yes, reverted** |

226 tests, up from 141 at the start of the day.


---

## 8. Session 5 (2026-08-24) — past p2b for the first time, and why the referee was losing half its work

**`live36` gen02 passed the p2a gate and completed the 3,000-row gold set** — the
first time this pipeline has ever got past p2b. κ 0.814, raw agreement 0.829, 511
disagreements, annotator b short 11 rows (reported, not absorbed). It then died in
the referee and was halted deliberately; `live38` is the clean re-run.

### RESOLVED — the redraw loop has no demonstrated effect (was §2 item 5)

Three controlled before/after trials now exist, not one:

| run | κ before | κ after | Δ |
|---|---|---|---|
| live35 | 0.844 | 0.827 | −0.017 |
| live36 r1 | 0.781 | 0.806 | **+0.025** |
| live36 r2 | 0.806 | 0.795 | −0.011 |

se(κ) at n=200 is 0.027–0.031. **All three sit inside one standard error and the
signs disagree.** Resolving an effect this size needs a pilot of ~1,150 queries,
~6× the current 200 — more than the gold set it protects. The earlier "points the
wrong way" reading was over-confident: the honest finding is *unresolvable at this
n*, not *harmful*.

**Why, measured.** Replaying all three live36 pilots (class codes byte-identical
throughout, so pairs are comparable): 36 disagreements over **25 distinct pairs**,
19 seen exactly once, expected count per pair ≈1.4 — and the **top-6 overlap
between consecutive pilots was 3/6, twice**. `structural.most_common(6)` was
handing the architect a target list that was half resampling noise, at ~350s of
frontier model plus a full three-pass re-pilot per round. Now requires ≥2
observations, and logs what it dropped.

**The redraw does real work — it just never merges.** All 24 codes stayed
byte-identical while 24/24 definitions changed substantively, converting
inferential boundaries to surface-signal ones ("必须有公式/推导信号"). That is the
prompt's option 2 done properly. It never takes option 1, *merge*, which the
prompt ranks first and which is the only move that removes an unresolvable
boundary — the prompt states preservation twice and structurally against one line
of merge preference.

### The referee was truncating at exactly 24,001 tokens, ten times

10 of 15 referee calls died at *precisely* 24,001 output tokens; the successful
ones emitted 19,279–19,597. Three coupled defects:

1. the declared budget (4,000 → cap 12,000) was measured at 8,179 **on a model
   long since replaced**;
2. `_hit_length_limit` matched the error's **prose**, and glm-5.2 returns
   `no parseable structured output` for a truncation — so the bump never fired;
3. `_length_bump` was clamped at `max(current, 2)`, so a 2× floor that still
   truncated could never grow.

Fixed as: budget 12,000 (cap 36,000, from measurement); truncation detected by
`output_tokens >= cap_in_force`, which is ground truth on any provider; and a
second floor 2→4 on evidence of truncating while already bumped. **Confirmed live
on `live38` within minutes** — a researcher escalated 2× → 4× and completed.

A failed batch now **bisects and retries each half** instead of discarding 25
adjudications. This matters beyond waste: the referee adjudicates exactly the
contested rows, so dropping them strips the *hardest* cases and every downstream
number reads optimistically.

### The p2a gate punished a redraw that worked

`remedy_exhausted = redraw_attempted and not any(r["kept"] ...)`. On live36 gen01
redraw 1 improved κ and redraw 2 was reverted — the loop was out of moves, but
`any(kept)` was True, so the gate **failed a run whose agreement had gone up**. An
improving redraw was strictly worse for the gate than a failing one. Zero test
coverage before; five tests now, three of which fail against the old code.

Honest note: gen02's redraw did *not* help, so `redraw_helped` was False there and
the old code would have passed that gate too. The fix was necessary for gen01's
numbers, not gen02's.

### Four defects on the recovery path the halt message itself recommends

`new-generation` → `qmine run --resume` had never been run end to end:
both resume paths hardcoded **generation 1** (and the thread id is
`{run_id}-gen{generation}`, so it reopened the old halted thread); `new_generation`
writes no `config.resolved.yaml`, so config resolution found nothing and fell into
the refuse-existing-id guard; `--input` was required even when resuming; and a new
generation is a **new thread with no checkpoint**, so invoking with `None` raised
`Received no input for __start__`, which `open_memory` masked as "generator didn't
stop after throw()". Also `resume_run` wired no `registry.on_call`, so a resumed
run logged **zero** agent lines — the dashboard would have been empty for its whole
duration. Event wiring is now shared by both entry points.

**CLAUDE.md's re-run instruction was wrong** and has been corrected: the guard
fires on `llm_cache/` as well as `checkpoints.sqlite`, so "delete the checkpoint,
keep the cache" lands on the guard. Use `new-generation` + `--resume`.

### Budgets go stale because the MODEL changes, not the task

Re-measured with no prompt changed: architect **23,759 → 38,073**, referee
**8,179 → 19,597**. Both roughly tripled. The architect was at 91% of its cap and
failed this project's own `cap >= observed × 1.2` rule; raised to 48,000.
`test_output_budgets_cover_what_the_roles_actually_emit` now carries the live36
figures and the model they were taken on. The agent transcript now records
**per-call** input/output tokens, so the next budget question is answered by
reading a file instead of differencing cumulative totals out of `run.log` — which
only works for sequential roles and silently misleads for concurrent ones.

Related: the per-call agent line in `run.log` was printing the role's *cumulative*
output, which reads as per-call and is wrong by a factor of the call count. It sent
this session's own diagnosis down the wrong path for several minutes.

### Building the real routing plan (since `qmine models` cannot)

```python
from qmine.cli import _load_env; _load_env()
from qmine.config import QMineConfig
from qmine.llm.registry import ModelRegistry
from pathlib import Path
cfg = QMineConfig.load(Path("configs/live.yaml")); cfg.llm.provider = "router"
reg = ModelRegistry(cfg.llm, cache_dir=Path(".cache/preflight_llm"), run_cfg=cfg)
print((reg.usage().get("routing") or {}).get("assignments"))
```

Verified before launching `live38`: `provider: routed`, and annotator_a /
annotator_b / referee on **deepseek / qwen / zhipu** — three distinct labs.

### The confusion archetypes recur in a fifth taxonomy

live36 gen02's top confusions — `AMBIGUOUS_BARE_QUERY × LOOKUP_CLASSICAL_TEXT_TRANSLATION`,
`LIST_WORDS_BY_CONDITION × SOLVE_CHARACTER_PUZZLE` — are archetypes B and C again,
under yet another set of class names. Five independent taxonomies, same contested
regions. This belongs in the deliverable: it is a finding about the **corpus**, not
about any taxonomy we drew.

**RESOLVED same session — the referee bisect now has a test.** Extracted as
`_run_batch_with_bisect(run_batch, fold, chunk) -> (failed, covered)`, a pure
higher-order function, and covered by five tests; four fail against the old
drop-the-batch behaviour. Two properties they pin that are easy to lose: the
split is **not recursive** (bounded at two extra calls, so a systematically
failing referee cannot cause a retry storm), and `fold` runs **per call, not once
at the end** — a recovered half must bind the halves after it, or the rule set
acquires two rules that fire on the same trigger with opposite answers.

**A merge orphaned its own rules, and that confounds every merge trial.** The
redraw swaps `nodes` and leaves `rules` untouched, so a class merged away keeps
the rules that route to it. Measured on live36 gen02: dropping
`INTERPRET_LITERARY_MEANING` left **4 of 45 rules dangling, 2 routing directly to
the deleted code**, and both governed `INTERPRET_LITERARY_MEANING ×
LOOKUP_WORD_MEANING` — a pair that redraw had targeted. The annotator was
instructed to assign a label that was not in its class list, on precisely the rows
the before/after comparison is decided by.

Consequence for the redraw question: sort the trials by whether they merged.
live35 and both live36 gen01 rounds did **not** merge (24→24) and are clean.
live36 gen02 **did** (20→19) and is confounded. The only trials that exercised
merging — the remedy the prompt ranks first — are the unusable ones. `live38` is
merging too (22→21 with 50 rules), so its comparison is confounded as well; read
its `redraw 1:` line accordingly.

Fixed: rules whose `then` names a removed class are pruned before the re-pilot and
the count is announced. Two tests; one fails against the old behaviour.

**RESOLVED same session — `qmine models` now pre-flights the real configuration.**
It took `--config`, mirrors `ModelRegistry`'s own `route()` call argument for
argument (`prefer`, `budget_usd`, `prefer_chinese_native`, `excluded_labs`), and
prints the policy in force. Verified: it reports **$6.34**, the same figure
`live38` printed at launch, where before it silently planned against labs the
live config forbids.

It also now prints the annotator/referee **labs**, not just the gateways:

```
annotator/referee labs: a=deepseek, b=qwen, referee=zhipu — independent
```

The model column shows `qwen:dashscope/...` for BOTH annotator_b and the referee
because that is the gateway; their labs are qwen and zhipu. An operator reading
the table alone would reasonably conclude the referee shares a lab with an
annotator — the exact gateway-vs-lab confusion this repo already has a test for.
The independence rule is by lab, so it is now stated by lab.

### A multi-agent audit of the never-run phases, and what it found

Five read-only review lenses over everything after the referee — code the live
run reaches last and that has never executed with live agents — each finding then
handed to a separate agent instructed to REFUTE it. 15 candidates, 8 verified,
**7 confirmed, 1 refuted**. All seven are fixed with tests. Two would have ended
a paid run.

1. **`max_total_output_tokens = 6_000_000` was HALF what an honest run needs.**
   The rule (~2x the whole-run estimate) was right; its input was stale.
   Declared estimate 3,074,000 vs honest 12,096,234, so the ceiling was 2.0x the
   declared figure and **0.50x** the real one — a runaway guard that fires on
   correct behaviour. Root cause: `annotator_a` declared 5,000 output tokens per
   call and emits **21,975**. Annotator budgets → 22,000, ceiling re-derived to
   24,000,000. The pre-run estimate moved $6.34 → $17.15, against ~$14
   extrapolated from live38's actual spend — the old figure understated by half.

3. **Guide repair discarded the referee's entire output.** With
   `repair_on_fresh_sample` (the default) round 2 annotates DISJOINT queries, and
   `rows = repair_meta["rows"]` swapped the list — throwing away ~3,000 round-1
   rows including every adjudication, and substituting a set whose only labelled
   rows are ones both annotators already agreed on, because the referee runs
   before repair and never sees round 2. **The gold set became agreement-only**,
   i.e. systematically the easy rows, so every classifier number computed from it
   read high for that reason alone. Now merges.

4. **`UNLABELED == UNLABELED` counted as agreement.** `agreement()` already
   excluded it from kappa, so the METRIC was safe while the GOLD SET was not: a
   row both annotators omitted was recorded agreed with `final="UNLABELED"`,
   skipped the referee, and passed p2c's non-empty filter into the classifier as
   a real class.

5. **An empty adversary response scored 1.000.** `n = max(len(verdicts), 1)` gave
   `1 - 0/1` — perfect accuracy manufactured by a provider failure. Denominator
   is now verdicts returned, coverage travels into the artifact and the Chinese
   report, and no verdicts means undefined. The chunk call was also unguarded, so
   a blip crashed the phase.

6. **L2 "visible to the embedding" used a flat 0.5** with no config path. kNN
   agreement means different things at different class counts — chance is ~4.5%
   at 22 classes, ~20% at 5 — so a flat bar calls large classes visible on their
   PRIOR and small ones invisible regardless of geometry. Now
   `max(floor, 2 x chance)` with the lift recorded. The test shows it: a dominant
   class embedded as pure noise clears 0.5 and fails the new bar. `n`/`share`
   renamed to `n_in_subsample`/`share_in_subsample` — they were subsample counts
   printed beside population-scale numbers.

1. **Fail-fast.** An encoder download, an OOM, or a clustering bug currently
   surfaces at hour three. Concurrently it surfaces at minute twenty-five, before
   the expensive half is paid for.
3. **It scales the right way.** Embedding and clustering cost grow with corpus
   size; the gold set is capped by config. On a 500k corpus the bottom-up branch
   dominates, and the saving grows with it.

**Not implemented, deliberately, and the ordering matters:** a fork-join changes
halt propagation (a p2a gate failure must stop a branch already in flight),
checkpoint/resume semantics, and concurrent artifact writes — in a pipeline that
has **never once completed end to end live**. Get one complete run with real
deliverables first; parallelise against a known-good baseline, where any
regression is attributable. Doing it in the other order means debugging
concurrency and a never-finished pipeline at the same time.

### Three more open questions closed this session

**`--reuse-taxonomy` is wired** (`RUN_ID`, `RUN_ID/genNN`, or a path). It skips
p2a entirely and reuses a finished `taxonomy.json`, which is the root fix for the
resume cascade: the web-using researchers are non-deterministic, so re-deriving a
taxonomy changes every annotator prompt and misses the cache on all 3,000 gold
rows. It RAISES on a bad spec rather than falling back to re-deriving — a run
that silently ignores the flag would pay for a full architect pass and miss the
cache it was pointed at, which is the very failure it exists to prevent.

**Error classification now keys off the SDK's `status_code`,** with the prose
scan kept only as a fallback for providers that raise plain exceptions. This was
the "bitten three times in one day" item. Both halves are now structured: a
truncation is `output_tokens >= cap_in_force`, and a dead provider is a status in
{401, 402, 403}. `completion_tokens=4013` can no longer read as a 401. (`bool` is
excluded explicitly — `isinstance(True, int)` is True in Python, so a flag would
otherwise have read as a status.)

**The planner's accuracy has been re-derived** now that the ledger prices at the
routed model's own rates. `qmine models --config configs/live.yaml` reproduced
`live38`'s launch figure exactly ($6.34 at the time), and after the budget
corrections reports $7.01 against ~$14 extrapolated from live38's actual spend —
under, but within the same order, where the pre-correction number was less than
half. Remaining gap is that `output_tokens_per_call` is now the pair AVERAGE for
the two annotators, since which one draws the reasoning model is not known until
routing has already happened.

### Silent prompt truncation, and the class of bug it represents

Four fixes, after finding that the referee's entire contribution was being cut
from the prompts meant to apply it:

1. **`budget_text` announces what it drops** — with the block's name, the share
   kept, and an explicit warning when the trim is HEAD-ONLY so anything appended
   is lost first. The in-band `… [truncated N chars]` marker told the MODEL it
   was reading an excerpt and told the operator nothing, because nobody reads the
   prompt. Every instance of this class hid behind that.
3. **Rules budget 9,000 → 20,000**, sized against the 18,496 the pipeline
   actually produces. Verified 83/83 referee rules now survive, against 0 under
   the old budget and 29 under an intermediate 12,000.
4. **`taxonomy_v2` is persisted.** `taxonomy.rules.extend(new_rules)` and the
   repaired `labeling_guide` were applied to the in-memory object only and p2b
   returned no taxonomy artifact — so `gen02/taxonomy.json` showed 50 rules and a
   guide with no 边界裁定 while the run held 133 rules and a rewritten guide. A
   resumed run recovered the PRE-referee taxonomy. `deps.taxonomy()` already
   preferred `taxonomy_v2`; nothing had ever written it.

**The pattern, stated once:** this project's characteristic bug is *a number or a
payload that stays silent about what it left out* — the referee dropping its
hardest rows, adversarial accuracy shrinking its denominator, ECE changing basis
without changing label, a runaway guard derived from a stale projection, and now
a prompt block discarding the guidance it exists to carry. "Read `n` before
believing any metric" generalises: **make every mechanism say what it dropped.**

### Session 5 close — what to know before touching anything

**The pipeline reached p4 for the first time.** Every earlier run died at p2a or
p2b. The gold set that carried it through is **6,000 rows with 465 referee
adjudications**, and it exists only because of three fixes made today: the merge
that stopped guide repair destroying the refereed rows, the `UNLABELED`
exclusion, and the revert guard that undid a repair measuring 0.028 worse.

**First live results from phases that had never run:**

| phase | result |
|---|---|
| p2c classifier | CV accuracy **0.852**, macro-F1 **0.771**, ECE **0.023** out-of-fold, on 5,534 rows |
| p2d adversary | **0.953** survived attack, 7 wrong / 24 defensible, **coverage 100%** |
| p2e L2 | **5 of 21** classes rule-dependent |

Read all of them against the p2b gate's own caveat: κ 0.822 with ~0.10 of
residual slack, so these are accuracies against a gold set two annotators agreed
on 82% of the time — not against ground truth.

**Two claims this run falsified**, both now corrected in code:

1. The report asserted adversarial accuracy is *"lower and more trustworthy than
   cross-validated accuracy"*. Measured: **0.953 vs 0.852 — higher.** They are
   computed on different populations (the gold set is enriched with contested
   rows; the attack samples the corpus at random and draws mostly easy ones), so
   they are not comparable as levels at all.
2. My own L2 "chance-relative" bar never bound: `max(0.5, 2 x share)` with 22
   classes means the 0.5 floor always wins, so the flat constant still decided
   every verdict — and called a class separated **74x above chance**
   rule-dependent while passing one at 42x. Now `median - 1.0 x MAD`, which
   reproduces K12's five flagged classes exactly and flags nothing when every
   class clusters equally well.

**The habit to watch in this codebase, stated once more:** four separate
source-text assertions I wrote today misfired — matching inside a comment, inside
a longer identifier, on a legitimate read, and on a budget literal I later
changed. Test the behaviour, not the implementation text; where source inspection
is genuinely needed, match whole statements and assert the PROPERTY (a budget
exceeds what is measured) rather than the number.


## Session (2026-08-25, evening) — live39, and what only reading found

**The run.** `live39` gen01: 17/17 phases, 201 min, $5.52, provider=routed,
κ 0.8341, CV 0.8586, 11 families / 39 leaves, all named. First run carrying the
corrected reports, the phase observers, the interpreter and the grid proposer.

**Verification.** Built `qm_verify_run.py` (scratch) — 19 mechanical checks, one
per defect fixed since live38. Validated against live38 FIRST, where it scored
2/19: a harness that passes on the broken run proves nothing. live39: **18/19**.
The single failure is expected — p2a ran before `web_researched` existed.

Building it caught three bugs in the harness itself, one serious: a check that
called the FIXED `family_names()` against live38's artifacts and passed, on a run
whose every family heading was wrong. **A check that verifies today's code against
yesterday's output verifies nothing about the deliverable.**

**Five defects only reading found — three of them mine.**

| defect | note |
|---|---|
| `fig3_battery.png` MISSING | my colour fix used `matplotlib.cm.get_cmap`, removed in mpl 3.9 (env 3.11). I checked the cell PARSED and never rendered it. |
| the lost figure reported "0 cell errors" | the cell caught the exception and printed it, so nothing counted it |
| fig3 plots 15 of 18 configs | 3 HDBSCAN carry `stability_ari=NaN`; matplotlib drops NaN silently. The reason — **86-91% of rows came back as noise** — was invisible. Now stated on the chart. |
| `"chosen K (stability peak)"` in `viz.py` | my earlier fix searched for the KEY NAME, not the PHRASE. `grep "stability peak"` finds all four sites at once. |
| English `chosen_by` in the Chinese report | pre-existing; exposed when my longer text tripped the language test |

**Two things I reverted or re-did.**

* Replacing the α `tie_band` constant with the measured `noise_floor` — **tried and
  reverted.** The alpha sweep is non-monotone (1.98, 2.02, 2.42, 1.98, 2.41…), so
  its roughness is SIGNAL; the estimate widened the band from 2.08 to 2.42 and
  flipped the winner from α=0.1 to α=0.5 on nothing. Recorded in
  `.claude/rules/measurement.md`: `noise_floor` needs a smooth sweep, and
  separating noise from signal on a jumpy one needs REPLICATION.
* While fixing English-in-Chinese in the battery figure I **introduced the same
  defect** in the α rationale. `prose()` matches a literal prefix, so a
  translatable sentence must carry NO interpolated numbers — the shape that works
  is a stable sentence plus the numbers as separate DATA fields.

**A real defect the observer found, verified by hand.** The α decision rationale
said "Lowest template fragmentation with highest stability" — and α=0.1 had
NEITHER (2.0193 vs 1.9799 at α=0.0; α=0.5 more stable). The artifact's `chosen_by`
was right all along; only the sentence a reader sees was false. Now states the real
rule with this run's numbers.

**Not started:** the p6 `leaves_per_family` inconsistency (see §1), extending
triggers to all referee rules, and the three untested domain profiles.

---

## Session — 2026-08-26: agent authority, and what an agent may be trusted to do

The day's question was the observer's powerlessness on live39, and it turned into
a general one: **what is an agent allowed to do, and what makes each permission
safe?** The answer that held up across three separate mechanisms is the same one:

> An agent may supply the measurement that would settle its own claim. Only the
> measurement carries authority.

### The observer did not need permission — it needed a way to be proven right

Giving it write access would have put an unaudited LLM judgement in charge of the
run. But its live39 finding was *arithmetic over an artifact*, and nothing could
evaluate it. So an observation may now carry a `check`, and a confirmed one is an
assertion that failed rather than an opinion — which is what makes it safe to
block on. `severity` no longer decides anything; it is the model's own confidence.

The evaluator (`ops/checks.py`) is a security boundary: the string comes from a
model. `a.b` is a **dict lookup**, never `getattr`, so `__class__` resolves to a
missing key rather than to a Python object and the standard escape chain has no
first step. A mutation swapping one `return _MISSING` for a `getattr` survived the
whole suite at first — the hostile-expression tests only reached the dict branch.
Now each container branch is probed separately, and a parsed (not grepped) test
asserts `_lookup` contains no `getattr` call.

### Findings that cannot be forgotten

The other half of "nothing consumed the finding". `ops/findings.py` is run-level,
so a new generation inherits it like the LLM cache. The only automatic exit is a
measurement — the entry's own check passing again. A mutation relaxing that to
"the check is not failing" survived, and it mattered: a phase that stopped writing
an artifact would have silently closed every finding about it.

### The referee diagnosis was wrong, and measuring first is what caught it

Yesterday's plan was "make the referee emit a trigger". Reading its actual rules
first showed they are semantic conditions with no regex, 79 of 80. Holding them to
the referee's own verdicts found a defect nothing could see:

    OTHER × TEXT_INTERPRETATION — referee ruled TEXT_INTERPRETATION 15/21,
    and FIVE of the six rules say "no intent marker → OTHER"

Five rules restating one principle, each the opposite of what the referee had just
done on the rows in front of it. Reported per boundary, because a rule is
conditional and one at the minority is a legitimate exception — a per-rule score
would have flagged every honest exception as a defect.

### The one agent with write authority

`agents/audit_delivery.py` reads every gate, the ledger, the artifacts and the
finished documents together and may edit the reports. What makes that safe is not
trust, it is the shape of the operation: an anchored replacement, anchor proven
unique, every number sourced from **the artifact the edit cites** (which keeps the
pool small — `agents/verify.py` documents its own blind spot on large pools),
language checked, reason required, originals kept. `.md` only. Refusals are
printed beside the edits, because a report showing only successes is a sales
document.

### Method note

15 mutations across the new guardrails; 3 survived the first pass and each one was
a real hole in the tests rather than dead code. Two end-to-end offline runs, and
the second existed only because the first put three empty-claim rows in the ledger
— a defect that no test would have found and that reading the output did.

---

## Session — 2026-08-26 (late): the graph forks

The question was whether the bottom-up route really has to wait hours for p2b.
It does not: `p3_represent` reads only `template_groups`, from p1.

**What made this a design problem rather than an edge rewrite** was that
langgraph's scheduling is not what it looks like. Three behaviours, each measured
with a throwaway graph before any production code changed:

1. parallel branches lock-step per superstep — so the node GROUPING sets the
   cost, and 2-against-2 beats 1-against-3 by 22 simulated minutes;
2. a fan-in node fires once per incoming edge unless the edges arrive together;
3. a state field written by two branches in one superstep is a runtime error.

Guessing any of these wrong produces a pipeline that is slower, or that trains
its classifier twice, or that dies 107 minutes in. Measuring all three first cost
about ten minutes.

**The most valuable finding was not about scheduling.** Concurrency does not
create races; it reveals the ones a serial graph was hiding. Four shared
read-modify-writes were live, and the worst was `ops/findings.py` — added earlier
the same day, and measured here losing **6 of 8** concurrent filings. That is the
"nothing consumed the finding" failure the module exists to prevent,
reintroduced by the scheduler instead of by a missing consumer.

**Race tests lie.** Three of the four passed against a deliberately unlocked
implementation. A `RecordingLock` asserting the critical section is held is
deterministic and is what the invariant actually means; the stress test stays
beside it, because the structural check cannot see a section that is held but too
narrow — which is exactly what mutation testing then found in `deps.decision`.

---

## Session — 2026-08-26 (night): live40, and what a confirmed check is worth

**The mechanism found real defects and produced more false ones than true.**
13 machine-confirmed findings; 2 real. The dominant error was not bad arithmetic
but a bad inference from good arithmetic: two fields differ, and they were never
supposed to agree because they count different populations.

That is worth stating as a design property. `ops/checks.py` converts a claim into
a measurement, and the measurement it makes is narrow: *this assertion is false*.
Everything the observer wraps around that — which fields to compare, whether
they are comparable, what it means if they differ — is still an LLM judgement
with no guardrail on it. The confirmation makes the arithmetic trustworthy and
the conclusion no more trustworthy than before.

**The fleet found more outside the confirmed list than inside it.** Four false
statements reaching readers of shipped documents, and one executor no-op
touching the delivered partition — none of which any check had flagged, because
no observer thought to assert on them.

**And the auditor was right four times and refused four times.** Both causes were
mine: inverted check semantics for edits, and a citation namespace that excluded
the very evidence the auditor is handed. One of my own tests was pinning the
first bug — the suite was protecting the defect. That is the second time this
week a test encoded a wrong invariant rather than a right one.

---

## Session — 2026-08-26 (late): the reports were unreadable, and why

The user read live40's two markdown reports and could not follow them: "very hard
to follow", "can't grasp their overall logic and reasoning steps", "unnatural and
not that coherent and with patches everywhere". They asked whether script
generation caused it — explicitly saying scripts are *not* wrong and are important
for correctness — and whether an agent could write a genuinely synthesised final
report instead.

**Diagnosis: two independent causes, and the reader was right about both.**

*Mechanical defects in the templates.* §9 of the bottom-up report interpolated a
raw value into prose, so `p2e`'s per-class audit — a list of 13 dicts — reached a
Chinese deliverable as ~1,900 characters of `[{'class': 'OFFTOPIC_RISK_NOISE',
...}]` mid-sentence. Three rationales shipped in **English** inside Chinese
reports. 45 lines exceeded 400 characters. All fixed: `_short_value()` summarises
containers by size and leaves them in the artifact they came from.

*No narrative spine.* Fourteen sections, each generated by an independent
function, none referring to the previous. Reading order is generation order.
Every defect ever fixed added its caveat paragraph where it happened rather than
where a reader needs it. That is structural and no amount of template polish
fixes it.

**The guard that should have caught the English had TWO independent failures.**
`test_every_authored_rationale_reaches_the_reader_in_the_report_language` runs on
the offline fixture, which never takes the `p2e` audit branch or the HDBSCAN
screen — the known coverage gap. But its detector was also too weak to fire at
all: it asked for three consecutive ≥6-letter lowercase words, which real English
almost never contains, because function words break every run. **It would have
passed on the live report too.** Replaced with a CJK-absence detector
(`i18n.looks_like_english_prose`, shared by test and runtime so they cannot
drift), and backed by a *static* AST test over every `deps.decision()` rationale
literal in the source — coverage that no fixture's reach can hide.

Turning the detector up immediately exposed a larger class: **all 22
`deps.gate()` messages are English f-strings** and the ledger prints them
verbatim. That is a restructure, not a mapping, so it is frozen by gate NAME in
`GATE_MESSAGE_DEBT` — the set may shrink, and anything not in it fails. Visible
debt beats an invisible allowlist. **This is open work.**

### What was built: `agents/narrate.py` + `report/narrative_brief.py`

Research first, since the design question was real. Classic NLG separates content
determination → document planning → realization; the scripts do realization only,
which is *why* there is no spine. Two findings bound the design:

- **Ungrounded content drives fabrication.** RotoWire grounds only ~60% of its
  summary content in the box score, and that deficiency is what teaches a model
  to emit unconditioned facts. So a fact sheet must be scoped **and sufficient** —
  withholding a number a section must state induces invention rather than
  caution. This bit immediately: reading the taxonomy one level too shallow gave
  that bundle 2 facts and no class names.
- **Precision-only grounding permits selective reporting.** Coverage-aware
  evaluation exists because a system can report only favourable, easy-to-express
  facts and score perfectly on precision. `check_numbers` is precision-only.

**Design.** Two passes: a planner sees a *map* of the run (bundle titles + the
must-cover list, no numbers) and writes its own outline; a writer then does one
section at a time against evidence scoped to that section, carrying its own
outline and the previous section's tail. Structure is global and agent-authored,
grounding is local and mechanical. Four checks per section (numbers, figures,
language, length) plus a whole-document coverage check.

**The vestigial `ReporterAgent` was the natural home** — registered in
`ALL_ROLES`, never called, exactly the anti-pattern CLAUDE.md names. Its shape
(one call, whole report, 60k evidence blob) was precisely the configuration the
literature says fails. Repurposed into `StoryPlannerAgent` + `StoryWriterAgent`.

### Three defects found by running it, not by reasoning about it

1. **Coverage was satisfiable by boilerplate.** Run over the *assembled*
   document, the provenance banner — which names `自下而上聚类最终报告.md` —
   marked "both routes must be explained" covered on a run where no section
   explained either. Now scoped to authored, accepted sections only.
3. **The panel bundle was 783 facts** — the artifact under a different name.
   Flattened to subject × metric.

### Deliberate: the fallback spine

If the planner fails structural validation three times, a code-chosen section
*order* is used and the document says so. This is the one thing a template may
decide about this report, and it decides only running order — every sentence is
still the agent's. Two reasons: delivering nothing on planner failure is worse,
and the offline stand-in cannot produce a valid outline, which would otherwise
leave the entire section-writing path unexercised by any test that runs the graph.
The stand-in was also taught to emit a number-free Chinese paragraph for
`markdown`, so offline runs traverse write → verify → assemble → coverage.

### Not done

- **Never run with a real model.** No agent-planned outline has ever executed.
- The 22 English gate messages (`GATE_MESSAGE_DEBT`).
- `verify_run.py` has no check for the final report.

---

## Session — 2026-08-27: how the bottom-up path actually decides, and what was wrong

The user asked how alpha, the algorithm, K and the leaf counts are really chosen,
why Phase 4 stopped sweeping algorithms, whether "7 families vs best kmeans k=15"
is a discrepancy, and whether silhouette is under-weighted. A 27-agent audit
raised 44 claims; 15 survived adversarial refutation. Several of the most
important findings came from measurements run here, not from reading.

### The three headline measurements

**1. Silhouette is precise and measures the wrong thing.** Subsample sd is 0.0007
(families) / 0.0040 (leaves) over 15 replicates — it is *not* noisy, and saying so
would be contradicted by anyone who measures. What it cannot do is rank across k:
**Spearman(k, silhouette) = -0.888**, peaking at the sweep's lower bound, so it has
no interior optimum. It also rises monotonically with alpha at every k (would elect
maximum surface weight, the template-twin failure) and, at fixed k, favours exactly
the geometry KMeans optimises (k=15: kmeans 0.070, agglo 0.009 — while agglo has
the second-best stability). **Prediction strength was tested as an alternative and
degenerates the same way** (Spearman -0.895, preferring k=2 at 0.98). All three
intrinsic criteria collapse toward small k; the external reference is a necessity,
not a shortcut. The fix is calibration (`lift_over_null`), not weighting.

**2. The K locator's reference is a decision nobody registered.** Holding
everything fixed and swapping only the reference partition: 6 trusted phrasing
groups -> peak k=12, all 12 groups -> k=12, the 25-class top-down L1 -> **k=25**.
AMI is also scored on only 33.4% of rows. live40 declared its 15-25 domain prior
wrong (`该修的是先验`) on a number that would have **agreed** with that prior under
a reference it already had.

**3. The two routes agree — at the layer nobody compared.** `route_crosswalk`
compared 7 families against 25 top-down classes and printed "routes disagree" on
every row, which 7-vs-25 forces arithmetically. At the leaf layer (25 vs 25):
AMI 0.5395 -> **0.6175**, median single-intent share 39.5% -> **80.3%**, 19/25
leaves majority-one-intent against 1/7 families. leaf 19 生僻字查询 is **100%**
BARE_TERM_LOOKUP. The prior was right about how many intent classes exist and wrong
only about which layer carries them.

### The largest defect: 36% of the delivered leaf layer was unmeasured

`choose_local_k` applies a null test and a stability floor to every leaf the
measured rule creates. `ops/governance.py:split_leaves` applied a `min_size` guard
and nothing else — and it made **9 of the 25 delivered leaves**. The two facts sat
in different artifacts and had never been read together.

Cause, found by measurement: ranking local k on **raw silhouette** hit the small-k
attractor — 5 of 7 families took k=2, the minimum, and none took 4-8. The Phase 7
audit noticed and prescribed 9 splits. **Replayed through `choose_local_k`'s own
tests, all 9 pass** (parent 6: lift 0.876 — a textbook-clean split the rule missed
entirely, and the very leaf that is 100% one top-down intent). The agent was
correctly compensating for a measurable defect in the geometry rule.

So splits are now **measured and not vetoed**, deliberately: a veto built on the
same biased geometry would reject the corrections to its own bias, and a split can
be semantically right and geometrically unsupported. The number ships beside the
split; a failing one is disclosed, not blocked.

### The "15 vs 7" verdict

Not a discrepancy — three different things wear the number 15. `battery_k` is a
diagnostic grid for a falsification probe that never selects K. But it *is* a real
gap that the probe runs at k in {15,20,30} and **never at the delivered k=7**,
below its own grid floor, so "this structure is a property of the corpus" is an
extrapolation across a >2x granularity gap stated as a measurement.

### Fixed this session

- `_rank_local_candidates` extracted and corrected: ranks on `lift_over_null`;
  both deltas measured against the same reference (they were not — `d_sil` vs
  `top`, `d_stab` vs the running `pick`, so the loop was order-dependent); a
  Pareto-dominated candidate can no longer ship (live40 family 3 shipped k=2 while
  k=3 dominated it on both axes). Replay: 16 -> 18 leaves, and **both halves of the
  fix do independent work** (ablated: metric alone fixes families 2 and 3, loop
  alone fixes only 3).
- Battery: `KMEANS_FAMILY` replaces a name-prefix filter that let `minibatch_*`
  and `bisecting_*` count as "structurally different" from KMeans; the margin is
  now **paired within k**, which flips live40's own sign at k=20 (`gmm_diag` ahead
  by 0.060) — invisible under the old unpaired comparison.
- `reference_profile()` records the locator's cardinality and coverage; p5 emits it.
- `route_concordance` artifact compares the routes at **every** bottom-up level.
- `stability_floor` is configurable and threaded (was a bare default in two places
  with no config entry and no test).
- HDBSCAN's claimed Phase-12 role removed from four places — the sentinel is a
  max-centroid cosine percentile and never touches HDBSCAN.
- Report fixes: the "K = 稳定性峰" mislabel (contradicted twice by the same
  report), the retracted "淘汰赛" sentence 23 lines above its own correction, a
  hardcoded noise claim that consulted no number, an English rule string in a
  Chinese figure title, and the English report path's "won a six-algorithm battery".
- The four docstrings claiming `challenger_beats_incumbent` protects the widened
  grid now say it is **not wired**. Deliberately documentation-first: applying the
  toll as written flips live40 to K=10, worse on both reported metrics.

### AMI's own noise, measured for the first time

K is located by argmax of `intent_alignment_ami`, and nobody had ever measured that
metric's noise — `noise_floor()` estimates it from the ROUGHNESS of a single-seed
curve, which is not a standard error over anything. Refitting KMeans with 5
independent seeds at k=7,8,10,12,15 on live40's full corpus:

| k | mean AMI | sd |
|---|---|---|
| 7 | 0.7508 | **0.0009** |
| 8 | 0.7114 | 0.0182 |
| 10 | 0.7495 | **0.0441** |
| 12 | 0.7562 | 0.0225 |
| 15 | 0.7248 | 0.0218 |

Pooled sd **0.0255**, against `noise_floor`'s 0.0129 — understated ~2x. The whole
live40 podium (k7 0.7495 / k10 0.7534 / k12 0.7507) spans **0.0039**, about 15% of
one sd, so **the argmax across k is noise**. Per-seed argmax was [7,10,10,10,10] and
the 5-seed mean argmax is k=12, not the k=10 that shipped as "best".

Two things follow. The tie band is `2 * se` = 0.0258, which lands within 2% of the
real 1-sd figure — so the tie set {7,10,12} is **correct, by an accident of doubling
an estimate that was half the true value**. Do not "fix" the factor without
re-measuring the band. And the noise is strongly heteroscedastic: k=7 is stable to
the fourth decimal while k=10 ranges 0.6728-0.7757 across seeds. **That is a far
better argument for the delivered K=7 than the "simplest tree" tie-break actually
used** — and nothing currently measures it. `noise_floor`'s docstring now carries
these numbers; its previous "validation" was one live38 figure agreeing with a
rounded note.

### Are the "trusted phrasing groups" a real reference, or our own priors?

The K locator scores AMI against them, so this is the question the whole selection
rests on. Findings, all measured on live40:

**They are hand-written.** `trusted = not is_discovered` (`templates.py:169`). All
6 trusted groups are seed regexes from `configs/domains/k12_zh.yaml`; **zero** mined
groups earned trust.

**But they are accurate.** Judged against the top-down taxonomy — an independent
methodology on the same corpus — median single-intent purity is **87.8%** against a
**14.2%** chance baseline, a ~6x lift. verse_continuation 95.8%, pronunciation
90.6%, meaning 88.6%, lexical_relation 86.9%, stroke_order 82.5%, word_formation
81.2%. So AMI is **not** measuring our priors back at us; the groups really are
same-intent. Mined groups are worse but not worthless (median 71.7%) with real
outliers both ways: `suffix:00字` 99.0%, `suffix:是什么` **42.0% across 7 intents**.

**The gate that vets them is anti-correlated with what matters.**
`validate_group_cohesion` scores mean pairwise cosine vs random — TOPICAL
tightness. An intent group spans topics by construction ("X的意思" for thousands of
X), so the gate penalises exactly the broad intent groups that make the best
references. Spearman(lift, purity) = **-0.60** (n=6, p=0.21): it PASSED
`word_formation` (lift 1.670, purity 81.2%, worst of the six) and REJECTED
`meaning` (lift 1.269, purity 88.6%, third best). `kept_because_seeded` — which
read like a courtesy to the human — was on this corpus **the more accurate call**.

**The portability path was silent.** `deps.template_masks(trusted=True)` falls back
to unvalidated mined groups under a comment reading "fall back loudly", with no
log, no gate, no artifact. `generic.yaml` ships **0 seeds**, so that is the default
for any corpus without a hand-written profile. Now emits and raises
`p3_locator_reference_validated`.

**Added:** `locator_reference_validation.json` at p10 measures each group's purity
against the top-down labels. **Limitation to keep in mind: it is retrospective.**
p5 runs concurrently with p2 under the fork, so no top-down labels exist when K is
chosen. It tells you whether the K you got rested on a good reference; it does not
improve the choice.

### The user asked to move p5 after p2 so K could use the top-down labels. Do not.

The instinct was right — the deciding reference is our own hand-written templates —
but the proposed fix is the one thing that breaks the project's headline result.
Measured on live40's full corpus before deciding:

1. **It makes the route-concordance result circular.** "Two independent routes found
   the same structure" (leaf-layer AMI 0.6175, 19/25 leaves majority-one-intent) is
   evidence ONLY because the tree was built without seeing the taxonomy. Locate K
   against `td_l1` and family-layer AMI moves 0.5748 -> 0.6308; that +0.056 is the
   fit, not agreement.
3. **`BlindnessFirewall.add_taxonomy` already forbids it.**
4. **It is unnecessary.** `ref_legacy_l1` — the corpus's own labelling, 9 classes,
   complete, external to BOTH routes, available at **p1** — locates **K=18 too**.
   The signal is obtainable free, with no serialisation and no circularity.

The premise also does not survive checking: `td_l1` on all 50k rows is a
**classifier's prediction** (cv_accuracy 0.8625, macro_f1 0.797) from a taxonomy
with κ=0.8427 and adversarial accuracy 0.82. A phrasing group is a deterministic
regex match. And the 87.8% purity figure for the phrasing groups was measured
AGAINST `td_l1`, so part of that 12.2% "impurity" is td_l1's own error — the groups
are plausibly better than the number says.

**Built instead:** `k_sweep` scores AMI against every declared reference column;
`reference_sensitivity()` reports where each one would locate K; `p5_k_references_agree`
fires when they disagree. Decision authority is unchanged — this is disclosure.

**The finding that matters for the next run:** on live40, phrasing groups locate
K=7 while BOTH non-template references locate K=18. The reference that decides is
the outlier. That is now visible in `granularity.json` and gated, but **the delivered
K did not change** — changing it is a methodology call for the user, not a silent fix.

### (3) Why the templates say K=7 — diagnosed, and it is COVERAGE not cardinality

**This corrects an earlier claim in this file and in two rule files.** I had said
the located K tracks the reference's *cardinality* (measured on a 15k subsample).
A full-corpus experiment with a coverage control refutes it.

Fixing the row set at the 33.4% the templates match and varying only class count:
6, 9 and 25 classes **all locate K=7**. Fixing the reference (`td_l1`) and varying
only the rows, all three sets identically sized:

| rows scored | locates K |
|---|---|
| rows our templates match | **7** |
| a random sample of the same size | **18** |
| rows our templates miss | **18** |

Not a subsample-size effect, not cardinality. **The six seed regexes select a
structurally atypical third of the corpus** — narrow lexical-lookup queries — and
K is located for that third, then applied to all of it.

**No label-free repair exists.** Background-as-one-class and downsampled-background
both still return K=7: the templates carry no information about rows they never
match, and reweighting cannot invent it.

### (2) Implemented — and the rule is portable, which was the user's real concern

Their objection to a hardcoded "prefer legacy" was correct: not every corpus has
legacy labels, and not every corpus admits accurate templates. So the rule names no
column. `locator_reach()` measures, **with no external labels**, what fraction of
clusters a reference holds a real share of — usable on any corpus:

- live40 @ k=18: phrasing groups **38.9%**, `ref_legacy_l1` **100%**
- reach is NOT row coverage: 33% of rows spread evenly reaches everything; 100% of
  rows concentrated in two clusters reaches almost nothing

`clustering.k_locator: auto` (default) gives the highest-reach reference the
locator role; `p5_locator_reaches_the_corpus` fires below 0.80 reach, saying the K
is scoped to the part of the corpus its reference could see.

**Consequence to expect on the next live run: K will change.** On live40 this hands
the locator to `ref_legacy_l1`, which locates K≈18 rather than 7 — a materially
different tree, cascading into leaves, naming and every downstream count. Nothing
was re-run, so this is untested against a real model. `k_locator: phrasing`
restores the old behaviour exactly.

### Not done — and one is now the top open question

- **`challenger_beats_incumbent` still has no call site.** Needs a signature change
  (`propose_grid` returns a flat list, so selection cannot tell a proposed value
  from a configured one) and a redesign, not a call.
- **The alpha sweep decides inside its own noise.** Winners across 5 seed
  replicates: 0.1, 0.5, 0.1, 0.0, 0.1. `tie_band=0.05` is an unreachable default
  ~4.5x narrower than the metric's own spread. **Do not simply widen it** — at a
  measured 2-sd band live40 elects alpha=0.5, which its own k=7 panel shows
  fragments worse. The fix is replication, not a wider band.
- The probe still does not run at the delivered K.
- Phase-8 metric deltas are one aggregate stamped on every prescription and
  computed on pre-split labels (+0.061 recorded vs +0.250 delivered).

---

## Session — 2026-08-27 (evening): a browsable dashboard, and three bugs it exposed

The terminal panel keeps `agents[-8:]`, `activity[-6:]`, `metrics[-8:]` because
that is what fits on a screen. live40 made **696 agent calls** over four hours, so
the panel showed roughly **1%** of the run and dropped the rest as it went. The
questions an operator actually has — what did that agent return, what exists so
far, what is queued — need scrolling, folding and search.

**`ui/web.py` renders `LiveDashboard`'s state as `runs/<id>/dashboard.html`.** One
model, two views, so they cannot disagree. A file rather than a server: no port,
no dependency, no lifecycle, and it keeps working after the run because the page
IS the record. Atomic `.tmp`-rename write, throttled to 3s, meta-refresh until the
run finishes, plus a daemon heartbeat every 10s.

### Three bugs found by replaying live40's own log

1. **The panel misreported the fork.** `current` was a single phase, so when P2a
   and P3a both start at 14:56:08 one branch was marked done while still running —
   p3 visibly flipped done → running. Completion is now **emitted** by `_wrap`
   (`✔ <node> completed in Xs`) instead of inferred from "a different phase
   started", which cannot be right for a forked graph. A branch phase never closes
   the other branch; a spine phase closes both, because reaching the spine means
   the join happened.
3. **Inferred durations were inflated** — live40's p3 rendered as 72 min when its
   work took ~12; the rest was the branch waiting at the superstep boundary.
   live41 measured it directly: **p3 = 1548.6s**.

Both fixes are mutation-tested.

### Two things found by looking rather than assuming

- Embedding all 696 full returns produced a **5 MB page that hung the browser's
  renderer**. Now 457 KB: newest `DETAIL_LIMIT=40` in full, older keep their
  result line and point at `agent_transcript.json`.
- The page **froze during quiet stretches** — it only rewrote on events, and a
  40-minute researcher fan-out emits almost none, so the meta-refresh reloaded the
  same stale elapsed time. Fixed with the heartbeat thread. **live41 started
  minutes before that landed, so its page ticks on events only.**

### live41 (running)

`--config configs/live.yaml --reference-columns legacy_l1,legacy_l2 --provider
router`. Pre-flight clean: annotator labs independent, est. $9.11, `reporter`
routed. Watch for **K landing near 18 rather than 7** if `legacy_l1` wins the
locator role on reach — that is the change from this morning and it has never run
live.

### Found and fixed DURING live41: a rule id that resolves to nothing

`p2a_observer` WARNED with a machine-confirmed check: 3 node-cited
`adjudication_rules` ids exist in no rule. Verified rather than assumed — 6
citing slots across 4 classes, referencing `RULE_POEM_TEXT_CHILD_OF_FULL_TEXT`,
`RULE_MATH_SCIENCE_OVER_FULL_TEXT`, `RULE_FULL_TEXT_RESOURCE_OVER_MATH_SCIENCE`.
The architect had invented a second id convention (SCREAMING_SNAKE with a `RULE_`
prefix) against the registry's lowercase snake_case.

**The consequence was concrete.** `_render_rules` treated a bare id as a
cross-reference only when it MATCHED a real rule; a dangling one fell through to
the free-text branch and reached the annotator's guide as

    - [POEM_TEXT_LOOKUP] RULE_POEM_TEXT_CHILD_OF_FULL_TEXT

— a line that looks like a rule, carries no adjudication content, and consumes a
budgeted section. So those boundaries had no tie-break **and** the guide gained
three lines of noise. Dropping is strictly better than passing through.

Fixed: `_is_bare_identifier` (ASCII-only, so an English rule's spaces and a
Chinese rule's CJK both survive) drops it, and `dangling_rule_references()`
reports every citing slot. live41's taxonomy renders 42 lines instead of 45 under
the fix. Mutation-tested.

**live41 is unaffected by the fix** — it passed p2a before the change, so its gold
set was annotated with the three noise lines present and those two boundaries
unruled. Worth checking whether POEM_TEXT/FULL_TEXT and MATH_SCIENCE/FULL_TEXT
show up as confused pairs in its referee log.

### live41 exercised the reach rule — and showed reach alone is NOT enough

The machinery worked exactly as designed and the design was incomplete. Reach
reproduced the offline measurement on fresh data (phrasing groups **40%** @ k=20
against 38.9% @ k=18 offline), `legacy_l1` won on 100% reach, and K moved 7 → 12.

**But 12 came out of an eight-way tie** — `[12,18,20,25,30,40,50,65]` — with the
raw argmax at 30. `min(tie_set)` returned the smallest of eight indistinguishable
values. Measured on live41's own sweep:

| reference | reach | range | range/se | peak K | tie set |
|---|---|---|---|---|---|
| phrasing groups | 40% | 0.2190 | 17.5 | 6 | 4 |
| **`legacy_l1`** ← chosen | 100% | 0.0373 | **5.9** | 30 | **8** |
| `legacy_l2` | 100% | 0.1072 | 17.5 | 18 | **3** |

Reach picked the LEAST discriminating of the three. `legacy_l1` is nine coarse
classes with two holding ~79% of the corpus, so its entire curve spans 0.037
against a 0.0126 tie band. The honest reading of its output is "this reference
cannot tell 12 from 65 apart", and it was reported as a located K.

**A locator must do two things: see the whole partition, and tell different K
apart.** `discrimination()` = range / `noise_floor`, and `choose_locator` now
requires reach ≥ 0.80 first, then maximises discrimination. On live41's data that
picks `legacy_l2` — same reach, 3x the signal-to-noise, a 3-way tie, locating
**K=18**, which is what every full-coverage reference gave on live40.
Mutation-tested against reach-only.

**live41 keeps K=12** — it passed p5 before the change.

### Also found by live41's observer: the decision record named the wrong locator

`evidence.locator` said `ami_vs_legacy_l1` while the same record's `rationale` and
`decisive_metrics` were hardcoded to the phrasing groups, left over from when they
were the only locator. Both now read from `tri`, and `deciding_reference` and
`locator_reach` are recorded beside them.

### Closed during the run

The observer machine-confirmed the alpha `chosen_by` mismatch flagged in the
audit: the string said "lowest template_fragmentation, broken on stability", and
the winner (alpha=0.1) is not the minimum (alpha=0.0 is). That is the tie band
working as designed and the sentence describing something else. `chosen_by` now
states the actual rule. **The selection itself is unchanged** — this was a
description defect, not a measurement one.

---

## Session — 2026-08-27 (night): three models silently became reasoning models

live41 gen01 was halted at **$28.14** against a $9.11 estimate, with the referee at
an **88% failure rate** and one batch of 11 rows abandoned outright. One root cause
explained all of it.

### The cause, probed directly rather than inferred

`deepseek-v4-flash`, `glm-5.2` and `qwen3.7-plus` all began returning
`reasoning_content` under the SAME model names since live40, and the tokens are
billed. One trivial prompt against each endpoint:

| model | completion tokens | of which reasoning | with `thinking: disabled` |
|---|---|---|---|
| deepseek-v4-flash | 505 | 497 | **9** |
| glm-5.2 | 237 | 227 | **10** |
| qwen3.7-plus | 586 | 573 | **10** |

Identical answers in every case. `enable_thinking: false` does NOT work on
DeepSeek (still 1,037 reasoning tokens); `thinking: {"type": "disabled"}` works on
all three.

### It caused three symptoms that looked unrelated

1. **Cost.** annotator_a went from 202 to 1,030 output tokens per label. Not the
   prompt — input HALVED (14,714 → 7,836/call) while output rose 5x, so the
   out/in ratio moved 0.35 → 3.25.
3. **The APIConnectionErrors.** Not the operator's network — proven by temporal
   segregation: qwen failed alone 20:15-21:15 while zhipu was clean, zhipu failed
   alone from 21:30 (the moment the referee fired 30 calls), and only **1** of 48
   failures had a cross-provider co-occurrence within 60s. glm-5.2's failure rate
   went 14% → 86% with load. Calls held connections open for 200-1000s because of
   reasoning; short calls should stop that.

### Fixed

`reasoning_kwargs(role, provider)` sends `thinking: {"type": "disabled"}` for
`NO_REASONING_ROLES` (annotators, referee, namer) on `REASONING_TOGGLE_PROVIDERS`
(deepseek, zhipu, qwen — an allowlist, because an endpoint that rejects an unknown
field fails the call). Deliberation roles keep their reasoning: an architect or an
observer is where those tokens earn their cost. Mutation-tested both ways.

**Output caps were deliberately NOT lowered** and **concurrency was deliberately
left at 8** — one variable at a time, and reasoning alone may fix the connection
errors by shortening calls ~20x. If they persist, lower `max_concurrency` next.

### Two defects of MINE, found by running it

**1. The reasoning field went in the wrong place.** I put `thinking` in
`model_kwargs`. LangChain forwards those as TOP-LEVEL arguments to the OpenAI SDK,
so every annotator call raised `TypeError: Completions.create() got an unexpected
keyword argument 'thinking'`. gen02 lost **24 annotation batches in one second**
and the pilot gate reported "kappa nan on 0 queries". Cost $0.62 — the calls failed
in 0.0s with nothing billed, and the gate caught it immediately.

The failure behind the failure: I verified the parameter was SET in the kwargs dict
and that `_build_routed` called the rule, and both were true while every call
failed. I had probed the raw HTTP API (which works) but never a call through the
actual client. Vendor body fields belong in **`extra_body`**. Now verified by real
calls: `annotator_a` → 9 completion tokens, reasoning None; `taxonomy_architect` →
124 tokens, 114 reasoning, both succeeding.

The test was strengthened to assert the field lands in `extra_body` specifically —
the original asserted the rule returned the right dict and that the builder called
it, which passed green while nothing worked.

**2. `new-generation --from-generation` defaulted to 1.** So running it on a run
already at gen02 created "the next after gen01" — gen02 AGAIN — and overwrote it
rather than advancing. That silently put a resumed run back into the generation
whose pilot had just failed, with nothing in the typed command to say so.
`artifacts.latest_generation` exists precisely because both RESUME paths had this
same hardcoded 1 (its docstring records that); this call site was missed. Now
defaults to the newest generation, verified by the command itself printing
"generation 3 inherits generation 2's config".

### Three more dashboard defects, all from one operator observation

The operator reported "my dashboard is still displaying old messages". Three
independent bugs, all mine from this session, all with that one symptom.

**1. A `--resume` run never had a dashboard at all.** The resume branch calls
`resume_run(...)` and RETURNS before the fresh-run setup — its own comment warns
"this branch returns before the fresh-run setup below ever sees it", and I walked
into exactly that. `resume_run` already accepted `on_event`; the CLI never passed
it. Both paths now go through `_attach_dashboard`.

**2. The page was gated on the terminal panel.** I built the HTML writer inside
`if use_dash:`, and `use_dash = dashboard and verbose` — both off without a TTY.
So a detached run, the case that most needs a browsable page because there is no
terminal to watch, wrote none. `enabled` now only switches the Rich view off.

**3. `qmine watch` fought the run for the same file.** Both wrote
`runs/<id>/dashboard.html`, and the follower always won on content because it
replays `run.log` from offset 0 — and `run.log` is append-only ACROSS generations,
so a run at gen03 had gen01 and gen02 events rendered into the page being read as
current. The follower now writes `dashboard.watch.html`.

**Plus a display bug the above exposed:** `_clock` ran `time.localtime` on a
REPLAY timestamp (seconds since midnight), so 22:03:48 rendered as "06:03:48" once
the timezone offset was applied. Durations were unaffected — differences cancel —
which is why it went unnoticed. Now discriminated on magnitude.

**The lesson, twice in one session:** I verified the writer was CONSTRUCTED rather
than that a page APPEARED, exactly as I earlier verified a kwarg was set rather
than that a call succeeded. Both times the check passed while the thing did not
work. Verified this time by watching the file appear, refresh, and contain only
the current generation.

### gen03 CONFIRMED every fix from today, on live data

**The reasoning fix.** annotator_a went 1,030 -> **67** output tokens per label
(15x), annotator_b 254 -> 70. The pilot completed **200/200 with zero rows lost**,
where gen02 lost all 200. Errors 53 -> 4, APIConnectionError 33 -> 2, batches lost
24 -> 0, spend $28.14 -> $1.38 at the same point.

**The battery fixes.** `best_alternative = agglo_average_k15` — correctly NOT a
KMeans-family variant, which the old name-prefix filter would have allowed. Paired
within-k margins expose the sign flip the unpaired comparison hid entirely:
k=15 **-0.0776**, k=20 **+0.0603** (gmm_diag ahead), k=30 **-0.1574**.

**The discrimination locator — and it vindicated the correction.** Measured live:

| reference | reach | discrimination | outcome |
|---|---|---|---|
| phrasing_groups | 0.40 | 17.48 | rejected on reach |
| `legacy_l1` | 1.00 | **5.88** | rejected on discrimination |
| **`legacy_l2`** | **1.00** | **17.19** | **locates K** |

**K = 18, tie set [18, 25, 40].** Against gen01's reach-only rule, which gave
K=12 out of an **eight**-way tie. 18 is the value every full-coverage reference
gave on live40. The two-stage rule (reach admits, discrimination decides) is doing
exactly what it was corrected to do.

### Small honesty fixes made during gen03

- `battery.note` still described the retracted "election" ("the winner is then
  fitted on the full corpus"), contradicting `verdict.role` in the same artifact.
  live41's p4 observer caught it.
- The probe's materiality bar (0.10 ARI) was applied and never recorded, so the
  conclusion "no alternative is MATERIALLY more reproducible" could not be checked.
  Now `materiality_threshold_ari` in the verdict.
- The p5 log said "K located by X (highest reach)" even when reach was TIED and
  discrimination decided — on live41 both legacy columns reach 1.0. Now names the
  criterion that actually broke the tie.

### A confirmed observation that was NOT a defect

`p2a_observer` WARNED: one query appears in two classes. Verified — it is a
`positive_example` of RIDDLE_BRAIN_TEASER and `source_evidence` for
RISK_COMPLIANCE_INTERCEPT. Different roles (one teaches a label, one records what
motivated the class), and **both classes are risk=True**, so handling is identical.
Checked the sharper property directly: 136 positive examples, 136 distinct
queries, **zero** duplicates and zero positive/negative contradictions. Another
instance of confirmed != defective.

### gen03

Clean branch from gen02, resumed off the 330-entry run-level cache — the five
researchers replay in under a second against gen01's 5+ minutes each. gen01 and
gen02 both kept as evidence.

**Note gen01's K=12 is superseded twice over** — it used the reach-only locator
AND ran before the discrimination fix.

---

## Session (2026-08-27 night → 08-28) — live41 abandoned, live42 run end to end

> These six subsections were appended as top-level headings, which broke the
> one-dated-section-per-session contract at the top of this file. Grouped here
> without editing their content. The exact night/morning boundary is not
> recoverable, hence the range.

### The resume rewind silently drops a concurrent branch

**The most serious defect of the day, and the hardest to see.** live41 gen03 ran
for 40 minutes through P4, P5 and P6 emitting healthy gates while the entire
top-down branch was missing.

| run | how it started | P4 | P2b |
|---|---|---|---|
| gen01 | fresh, no resume | 20:24:58 | **20:24:58** — same second |
| gen03 | resumed, then **rewound** | 23:45:23 | **never ran** |

`--resume` found a partial checkpoint and took the rewind path,
`graph.update_state(config, {...}, as_node="p1_audit")`. The fan-out survives for
the FIRST superstep — p2a and p3 both started — and not the second: `p456_tree`
was queued from p3's completion, `p2b_gold` never was. `phase_status` confirms it:
`{p0, p1, p2a, p3}` only, `halted: False`, no halting gate, `gold.csv` absent.

**This is NOT caused by the `--from-generation` fix** — that only chose a
directory. It is the pre-existing rewind path, which a fresh run never touches.
Restarting gen03 three times is what put the run on it.

**Why it is the worst shape of failure:** everything downstream still produced
artifacts and passed its gates, so nothing looked wrong. The run would have
reached the join and trained the classifier on a gold set that does not exist.

**Guard added at the join.** `_require_both_branches` reads `phase_status` and
halts with `p2c_both_branches_arrived` naming the phases that never ran. The gate
is RETURNED rather than registered — `deps.gate` only builds the record, and a
discarded gate cannot halt anything, which is exactly what
`test_the_delivered_leaves_gate_reaches_the_operator` was written against.
Mutation-tested: discarding the gate fails the test.

**The rewind itself is NOT fixed** — only made loud. Open question: whether
`update_state(as_node=...)` can restore a multi-superstep fan-out at all, or
whether a gap in `phase_status` should force a clean re-run of the generation
instead of a surgical rewind. Until then: **open a new generation and run it once,
without restarting mid-flight.**

### live42

live41 abandoned. Its three generations stay as evidence: gen01 (reasoning
tokens, 88% referee failure), gen02 (my `model_kwargs` TypeError), gen03 (the
dropped branch). live42 starts cold — no cache — which is the price of a clean
test of every fix at once.

---

### live42 results as they land (fresh run, all of today's fixes active)

**Gold set — better than live40 on two of three measures, with reasoning OFF:**
kappa **0.8928** on n=2999/3000 (live40: 0.8427), adversarial estimated accuracy
**0.9333** (live40: 0.82), classifier CV 0.857 (live40: 0.8625). The reasoning
disable did not cost label quality. NOT a controlled comparison — different
taxonomy, 21 L1 classes against 25 — so this is consistent evidence, not proof.

**`p2b_annotator_symmetry` fired for the first time ever:** annotator_a won only
34.1% of 270 contested rows, z=-5.2. There is NO baseline (the gate postdates
live40 and live41 gen01 never finished its referee), so this cannot be attributed
to the reasoning change. The parsimonious reading is the pairing itself —
`deepseek-v4-flash` against `qwen3.7-plus` is flash tier against plus tier.
Testing the reasoning hypothesis needs a paired re-run of the same rows.

**`p2b_rules_match_their_evidence` found 2 vacuous discriminators:**
`academic_knowledge_qa x problem_solving` names 求/解/计算 with **0 of 19**
contested rows carrying any; `navigational x school_info` names 主页/入口/好不好
with **0 of 8**. Rules naming a marker that appears in none of the rows they
adjudicate.

**The split measurement DISCRIMINATES — 7/9, not 9/9.** live40's retroactive
replay passed all nine, which made it look like a rubber stamp. live42:

| parent -> new | lift | stability | supported |
|---|---|---|---|
| 19 -> 54 | 0.0610 | **0.0595** | **no** |
| 0 -> 49 | 0.0778 | **0.3876** | **no** |
| 7 others | ok | 0.68-1.00 | yes |

Both failures are on STABILITY, not the null test — real structure that does not
reproduce. And the naming corroborates independently: leaf 19
「作文范文与写作指导查询」 split into leaf 54 「作文范文查询」, a near-duplicate of
its parent, at an ARI of 0.0595 (chance). The namer never saw the stability
number and the measurement never saw the name.

**This vindicates measure-don't-veto.** A veto would have blocked a split the
audit had semantic reason to want; disclosure lets a human weigh both.

**Tree:** 18 families / 49 leaves pre-governance, per-family k spanning 1-8
(`{1:1, 2:9, 3:4, 4:3, 8:1}`) against live40's 5-of-7-at-the-minimum. Half still
sit at k=2, so the small-k attractor is weakened, not gone.

**`converged: False` for the third run running.** The refinement loop hits its
5-round limit on live40, live41 and live42. Honestly disclosed in the report, but
a standing weakness: the delivered leaf count depends on which round it stopped at.

### The narrative report's FIRST live run: 3/9 sections, and why

The agent-written report ran against a real model for the first time. The
structural machinery worked: **the agent-authored outline passed on attempt 1**
(9 sections, the path the offline stand-in could never exercise), and **6 of 7
must-cover items were covered**, with the seventh disclosed in the document.

**But only 3 of 9 sections passed**, and the guardrails failed closed on the rest
— safe, and not useful. Two distinct causes, one of them mine.

### Cause B (fixed): the pool did not match the sheet

`sheet()` renders dotted paths, so a dict keyed by id displays numbers to the
author that `verify._flatten` — which pools only VALUES — refuses:

    execution.splits.32.new_leaf = 49      <- the agent reads "32"

`governance_and_risk` was rejected three times for citing `32, 40, 42, 43, 44,
45` — the leaf ids it had just been shown — and shipped as a hole. It was doing
exactly what it was told. `citable_numbers()` now pools every number the rendered
sheet SHOWS, and no more: an id absent from the text is still refused, verified
both ways and mutation-tested through `_reject` (a first version tested
`check_numbers` directly and the mutation passed — the test has to walk the path
the code walks).

### Cause A (OPEN): three sections returned EMPTY prose, three times each

`vector_choice_first`, `two_level_tree` and `samples_and_deployment` each failed
with `空白正文` on all three attempts — the model returned an empty `markdown`
field, not a wrong one. That is not a grounding failure and the retry feedback
cannot help it, because there is nothing to give feedback on.

Suspected but NOT established: 10 `prompt block truncated` events fired during
the report (fact sheets of 50-53k against a 40,000-char budget). Note the total of
all 16 bundles measured only 33k with EMPTY gates/decisions, so the live gates and
decisions bundles are what push a section over. Whether truncation causes the
empty returns is unverified — do not fix it as though it were established.

**Next step for this: reproduce one empty section offline against the recorded
sheet.** The section ids and their bundles are in `final_report_meta.json`.

### live42 finished — 17/17 phases, and what the narrative report is really like

All 17 phases completed and every deliverable shipped. **The run then crashed at
teardown**: `DecisionRecord` is no longer msgpack-serializable, so the SQLite
checkpointer failed, fell back to in-memory, and the store's context manager
raised `RuntimeError: generator didn't stop after throw()` on exit. Consequence:
**`run_summary.json` was never written** — the artifact `verify_run.py` and
several tests read. The deliverables themselves are intact.

### The pre-delivery auditor worked for the first time: 3 applied, 0 refused

On live40 it made 4 proposals and MY bugs refused all four. Here all three landed,
and they are good:

- `统一度量面板.md` — the prose claimed fragmentation "rises monotonically with
  cluster count" while the panel's own table falls from 1.8661 (k=18) to 1.827
  (k=24). It caught the report contradicting its own table.
- `自上而下类目体系最终报告.md` — a count mismatch around
  `n_triggers_rejected=39` with only 12 shown and 27 truncated.
- `00_最终报告.md` — the narrative report said the references "disagree"
  QUALITATIVELY without the numbers, so a reader could not follow the gate's own
  instruction to read the K together with its reference. It added the values.

One finding was dropped rather than applied: it cited `annotator_balance.n_contested`,
which does not resolve. The content was real (294 disagreements vs ~270 contested
rows adjudicated — two populations again), and the citation guard correctly
refused an edit it could not source.

### The report reads well where it passes — and has an error the checks cannot see

§4 explains why the locator is `ami_vs_legacy_l2`, names the competing indicators
(silhouette peak K=5, expert range 15-25, deep_aligned 28), reports the FULL tie
set 18/25/30/40 with every metric, shows the metrics disagreeing, embeds the
figure where it argues, and hands off to the next section.

**But it writes 「交付的 K=18 是参照 phrasing_groups 的粒度锚点」 — naming the
WRONG reference.** K=18 was located by `legacy_l2`; `phrasing_groups` located 10,
which the same paragraph states correctly two lines earlier.

`check_numbers` is precision-only on NUMBERS. A wrong noun is invisible to it, and
the must-cover anchor matched because every reference name appears somewhere in
the text. **The narrative door has a numeric guarantee and no ATTRIBUTION
guarantee** — same class as the `locator_reference` contradiction the p5 observer
caught. This is the top open item for the report, ahead of the empty-section
problem.

### Translation: from 34 hardcoded prefixes to a guarded model call

`PROSE_ZH` maps an English PREFIX to fixed Chinese at 20 call sites. Two holes no
diligence closes: a newly authored string is English until a human notices (three
separate leaks in one day), and an f-string can never be matched by a fixed prefix
— which stranded all 22 `deps.gate()` messages permanently.

`report/translate.py` adds a third tier to `prose()`, below the curated mapping
and above the English fallthrough. What makes it safe is that **nothing is
trusted** — every result is verified before use:

* **numbers** — the numeral multiset must match both ways. A rounded value is a
  changed value.
* **identifiers** — every backticked span survives verbatim. Numerals INSIDE
  backticks are excluded from the number count, because the "2" in
  `p2b_annotator_symmetry` is part of a name; counting it made a translated
  identifier report as a changed NUMBER, which a test caught.
* **actually translated** — a result with no CJK is the model echoing the source.

Any failure keeps the ENGLISH, exactly the old behaviour, so this cannot make a
report worse than the mapping it extends. Results cache by content hash in
`.cache/translations.json`: a string is paid for once and renders identically on
every future run — wording that drifts between runs for no measured reason is its
own defect.

**Verified against a real model** on the two strings that leaked into live42:
both translated cleanly, and `kappa 0.8928 on 2999 rows` returned as
`2999 行上的 kappa 0.8928` with both numbers intact. A gate message carrying five
interpolated values also translates — a class previously unreachable.

`GATE_MESSAGE_DEBT` is now OFFLINE-ONLY (the fixture installs no translator). Off
in `offline`; `cfg.translate_prose` disables it.

---

## Session (2026-08-28, late) — model pins, reasoning, and why the report was two-thirds empty

### The empty sections: measured, not inferred

live42's `00_最终报告.md` delivered **3 of 9 sections**. The document's own
placeholders under-report this — three of them say only 「空白正文」 three times,
and all of them say 「未通过校验」. `final_report_meta.json` and `run.log` carry the
real picture:

| section | what actually happened |
|---|---|
| `question_and_two_routes` | rejected: `90, 99, 10` |
| `vector_choice_first` | **blank body** ×3 |
| `topdown_taxonomy_and_labels` | rejected once (`5.23, 5.2`), then passed |
| `bottomup_k_not_single` | rejected once (`0.014048, 0.007024`), then passed |
| `two_level_tree` | **blank body** ×3 |
| `governance_and_risk` | rejected: `0.0169`, then `32, 40, 42, 43, 44, 45` |
| `unified_panel` | rejected once (`0.0162`), then passed |
| `samples_and_deployment` | **blank body** ×3 |
| `audit_and_limits` | rejected: `21, 5.23`, then `5.23, 5.2` |

So the number check touched **6 of 9** sections and killed 3 outright. Reading the
cached drafts (`runs/live42/llm_cache/`, `meta.role == "reporter"`) shows what the
author had actually written:

- `z_vs_even=-5.23` → extracted as `+5.23`. **`_NUMBER` had no sign.** A negative
  fact was uncitable: the author copied the sheet exactly and was told the number
  was not in the sheet. No retry can satisfy that.
- `裁判模型是 qwen:glm-5.2` → a phantom claim of `5.2`. A hyphen before a digit
  read as a discarded minus, so naming the model that did the work was a
  fabrication.
- `第 90 百分位为 13`, `家族 32、40、42` → the sheet SHOWS these (`length.p90`,
  dict keys) but the value-only pool did not carry them.

This selects against the sections that matter: negative numbers are where the
**warnings** live, so it deletes governance, audit-and-limits and the panel.

Verified with `check_numbers(text, {"z_vs_even": -5.23})` → unsupported, control
passing. Fixed in `verify._NUMBER` with two lookbehinds separating a minus from an
identifier hyphen by what precedes it. **Every one of live42's number-rejections
now passes**, replayed against the run's own artifacts; the three cases that must
still fail (fabrication, sign flip, miscount) still do. Mutation-tested.

Two further fixes from the same reading:

- **`annotator_balance` was in no bundle.** It is measured and lives in
  `taxonomy_v2.json`, which `build_catalogue` never read. Shown `n_contested=274`
  and `annotator_a_won=92` through other bundles, the narrator DERIVED the rest —
  `178 = 270-92`, `0.3407 = 92/270` — and every one was correctly refused. A
  starved sheet induces derivation, not caution. Added to `topdown_gold`, where
  `lopsided` belongs on the merits anyway.
- **Everything the writer was shown is now citable** (`_reject(shown=...)`):
  must-cover items, figure captions, previous section. A must-cover arrives under
  「必须原样包含这句话」, so a number inside one ORDERS the author to write what the
  check forbids. The outline is excluded — `_plan` verifies nothing numeric, so
  pooling the thesis would launder a number into every section — and so is the
  rejection notice, which prints the offending values.

The blank-body residue is unresolved and is now §2 item 2. It did not reproduce in
two real calls; both spent 93-94% of their output tokens on reasoning.

### Model pins

`glm-5.3-flash` and `qwen3.8-flash` are real and answer on their providers' direct
endpoints (verified by real calls). `qwen3.8-flash-next` **does not exist** (404).
Neither flash model is in the 1,930-model catalogue, because the catalogue is
fetched and they are newer than the price feed.

A bare pin with no card became `provider="explicit"`, a sentinel nothing handles —
so the run died on that role's **first real call**, after `qmine models` printed a
clean plan. Now: `resolve_pin` accepts `provider:model`, and an unresolvable pin
raises `UnroutablePin`, which `_build_routing_plan` re-raises rather than degrading
to the static tiers (its blanket `except` is for a missing catalogue, not a config
error). `configs/live.yaml` pins `zhipu:glm-5.3-flash` (referee, researcher) and
`qwen:qwen3.8-flash` (annotator_b); all three verified with real calls.

`glm-5.3-max` does not exist either — error 1214 on both endpoints with **both**
the user's abroad-registered and China-registered keys, so the China key changes
nothing and can be ignored. Note qwen IS region-split: an abroad key 401s on
`dashscope.aliyuncs.com` and works on `dashscope-intl`.

### Reasoning re-enabled for `referee` and `namer`

Both left `NO_REASONING_ROLES`. Neither is bulk classification: the referee
adjudicates the residue the annotators disagreed on and drafts rules that reach
them; the namer authors names that appear in the deliverable.

**This required raising `namer`'s budget.** The trace shares the role's output cap,
and measured here it runs 8-10x the content. `namer` had
`output_tokens_per_call=1200` → a 3,600 cap, which is inside a single trace — the
same shape as the referee's 88% failure rate on live41. Raised to 3,000 (cap
9,000); the declared budget moves too, or the ledger under-reports. `referee` had
36,000 already. Now pinned by test.

### State

- **571 tests passing**, `ruff --select F src/qmine/` clean.
- Verified live: the three pins route direct and answer; `referee` emits reasoning
  tokens (160) while `annotator_b` stays at `None`.
- Not yet run end to end. The next live run is the first to exercise the pins, the
  widened citable pool and the two reasoning roles together.

---

## Session (2026-08-28, later) — the reference shelf, and a dashboard that was showing the wrong call

### Budgets for the two roles that now reason — measured, and my first sizing was wrong

I sized `namer` by extrapolating the REPORTER's reasoning ratio (93-94% of
completion tokens). That does not transfer. Measured directly, on the real
prompts:

| role | completion | reasoning | content |
|---|---|---|---|
| namer | 696 / 1,149 | 531 / 994 | 165 / 155 |
| referee | 6,145 / 2,497 | 3,873 / 598 | 2,272 / 1,899 |

Reasoning is a roughly FIXED cost here, not a ratio — a few hundred to a few
thousand tokens, not 8-10x the content.

**The namer raise was still right, for a different reason than I gave.** live42's
namer spent 2,542-3,039 output tokens per call with reasoning OFF, against a
3,600 cap — 20% headroom. Adding a measured 531-994 exceeds it. Now 3,000/9,000.
**The referee needed nothing**: 1,486/call on live42 against a 36,000 cap.

**Bonus finding, from live42's own `usage.json`:** the declared budgets are badly
calibrated in both directions and this is HANDOFF item #7, now quantified. The
annotators are 500 of 702 calls, declared 12,000, actual 1,612-1,751 — a **7x
over-estimate on the dominant roles**. Observers, researchers and the delivery
auditor run 2-6x UNDER their declared budget. Not changed: recalibrating moves
caps, which moves truncation risk, and that deserves its own pass.

### The dashboard was showing a different call's output

Four defects, all verified in code AND against the user's screenshots:

1. **The agent detail was mispaired.** `raw_log` and `on_call` are two streams
   from the same `_store` with no join key, so `_agents()` paired them by index —
   a global reversed index against a per-role chronological list. The user's
   screenshot proves it: the row headed `reporter … 04:42:11` (attempt 1 at
   `audit_and_limits`) opened onto the top-down taxonomy section. Both streams
   now carry `cache_key`.
3. **The detail was `str(dict)`** — a Python repr of Chinese prose.
4. **`agent_transcript.json` never existed for live42** (killed by the teardown
   bug, since fixed), so the fallback pointed at a missing file.

Plus, found by audit and verified here: the artifact column read `key` where
`index.jsonl` writes `name` (blank for every run); replay elapsed printed
**496,632h**; and §8 of the top-down report shipped six rows of `| ? |  | — |`
because `_failure_history` reads `option`/`why_rejected` while the architect's
dropped candidates carry `name`/`why_dropped`.

**The event log** is now faceted: severity (from `logging`'s own level when
replaying, glyph otherwise) and phase, both captured at emit time, both with
counts, plus a 「怎么读这一栏」 line. Replaying live42 gives 99 warnings / 3 edits /
227 info across 12 phases — the log level alone caught 45 warnings the glyph
convention missed, including the two teardown failures.

### The reference shelf — the run was producing this and delivering none of it

`zh_reference.py`, wired into `builder.py`, four documents plus three CSV twins:

- **类目清单.md** — the 21 L1 classes with definition, `user_need`, positive and
  negative examples, **measured delivered size** beside the architect's
  prediction, and how many rules route to each. Symmetric to 叶清单.md.
- **标注规范与裁定规则.md** — the labeling guide VERBATIM (it appeared zero times in
  the whole delivery) and all 139 rules, one section each, grouped by target
  class, marked 架构师预判 vs 裁判补充.
- **家族与叶层级.md** — the delivered two-level tree, with a Mermaid top level.
  Reads `leaf_*_final`, and says out loud that the audit describes 20 families
  where 24 were delivered.
- **00_索引.md** — the reading order. Ten files landed in one directory with no
  index; every `put_markdown` call already passed a `summary` and all of them
  were thrown away.

Format follows the evidence (W3C DWBP BP 12; GitHub renders MD to 400KB and CSV
as a searchable table to 512KB; these are 7-38KB): Markdown is the reading
surface, CSV is the machine twin, and rules get one section each rather than a
table because a row cannot hold `when → then` plus rationale and examples.

Also fixed: the referee's rules shipped in English (`drafted by the referee to
close a gap…`) on 100 of the 139 — our own hardcoded template, now through
`prose()`, with the disagreeing query left verbatim as evidence.

Two further gaps from the same audit, closed in the same pass:

- **叶清单.md now shows the evidence each name was made FROM.** `naming_cards.json`
  holds the exact sample the blind namer saw — 15 centroid, 10 random, 5 edge per
  leaf — and none of the 1,470 sampled queries reached any deliverable. The EDGE
  samples carry the weight: leaf 1 is named 「2026年中小学暑假放假时间查询」 and its
  edges are `目瑙纵歌2026年时间表`, `退潮赶海时间表` — queries about times in general.
  A reader sees the boundary immediately. The sampling is mechanical, which is
  what makes it admissible rather than a flattering selection.
- **家族与叶层级.md now carries the cross-route mapping per family.**
  `route_crosswalk.csv` is the only artifact that says how the two routes line
  up and was named in no document. It is keyed by delivered family, so it belongs
  beside the family rather than in a table a reader has to join by hand. On
  live42 it reads well: families 5/7/11 agree with the intent taxonomy at 91-96%,
  while family 8 (14,171 rows) has 8.33 effective classes and 「routes disagree」.

### Five more, found by the workflow's synthesis re-checking current source

It correctly identified everything already fixed, and surfaced five live defects.
All five verified here before acting; all five fixed.

1. **The family definitions in my OWN new document were borrowed.** 14 of the 24
   delivered families carried a definition shared with one or two others —
   family 8 (17 leaves, 14,171 rows) and family 10 (1 leaf) got the identical
   sentence. This is this project's own delivered-partition trap, documented in
   `report-generators.md` and then walked into. A definition is now shown only
   when the family's leaves come from ONE audit family AND that audit family
   backs no other delivered family; the other 16 are told plainly whose
   definition it is and which delivered families share it.
3. **§2.1 L2 子意图 had never rendered.** It read a LIST from `subintents` or
   `groups`; the artifact carries a DICT under `subdivision`, keyed by L1 code.
   Neither key has ever existed. 54 sub-intents across 19 of 21 classes, and the
   panel's strongest comparative claim rests on them. Now rendered from the real
   shape, stating that they are UNNAMED and disclosing where silhouette disagreed.
4. **The escape hatch pointed at a file that need not exist** — unconditional
   "full return in agent_transcript.json".
5. **`qmine watch` hardcoded `provider=""`**, so every replayed page read
   "provider ?" while `usage.json`, already loaded for the KPIs, says `routed`.
   That is the exact field this project uses to decide whether a run was real.

### Still open from the synthesis, NOT done

- `fig_gates.png` plots 7 of 26 gates and treats `True` as `1` (`viz.py:310-314`).
- `zh_panel.py:187-200` slices findings `[:12]`/`[:8]` with no "showing N of M",
  hiding two BLOCKING findings, and cuts claims mid-token at `[:90]`/`[:150]`.
- Broken link to `Report_Uniform_Panel.md` (`zh_topdown.py:89`) and eight wrong
  `§9` cross-references (`zh_bottomup.py:827`).
- The `<details>` rule table in the top-down report should become a link to
  `标注规范与裁定规则.md` rather than have its truncation fixed twice.
- **No CLI path regenerates reports from finished artifacts.** Everything above
  was verified through a scratch harness; a `qmine render-reports RUN_ID`
  writing into a new generation would make it a one-command, zero-LLM operation.
  This is the highest-value remaining item: the reference shelf exists and has
  never been delivered by the pipeline itself.

### State

- **594 tests passing**, `ruff --select F src/qmine/` clean.
- Verified by replaying live42: the reference documents build from its real
  artifacts (14,395 / 33,252 / 10,992 chars), and the dashboard renders with
  correct elapsed, artifact names, facet counts and per-call detail.
- **Not yet exercised by a live run.** `live42/gen01` was deliberately left
  byte-identical to what it delivered; the new documents were built into
  `/tmp/qmine_refs` instead.

---

## Session (2026-08-31) — `qmine render`, and the five deferred items

### 1. Are the reference documents produced by a run? YES — verified end to end

`make demo` now emits all of them: 类目清单.md, 标注规范与裁定规则.md,
家族与叶层级.md, 00_索引.md and the three CSV twins, alongside the six original
documents. They were absent from `runs/live42/gen01` only because that generation
was deliberately left byte-identical to what it delivered.

**`qmine demo` was itself broken** and had to be fixed to check this. It calls
`run()` as a plain function while naming only 13 of its 16 parameters, so
`reuse_taxonomy`, `resume` and `dashboard` arrived as `typer.models.OptionInfo`
sentinels and one reached a Pydantic model:
`PydanticSerializationError: Unable to serialize unknown type`. `make demo` is
what CLAUDE.md points at to check wiring and it could not run. It now reads the
declared defaults out of `run`'s signature, so a new option arrives with its own
default instead of reintroducing the bug silently.

### 2. The five deferred items — all done

- **`fig_gates.png`** paired the FIRST numeric in `observed` against the FIRST in
  `threshold` with nothing tying them together, so a gate observing
  `{"n": 600, "kappa": 0.8928}` against `{"min_kappa": 0.70}` was drawn as 600
  versus 0.70. And `isinstance(True, int)` is True, so a boolean assertion
  contributed a value of 1. Both now go through `records.paired_gate_metric`,
  one definition shared with `_passed_below_threshold` so a figure and a table
  cannot disagree; gates with no numeric bar are counted under the axis instead
  of vanishing (live42: 7 of 26 plotted, all green, none of the four warned).
- **Panel findings** were capped at `[:12]`/`[:8]` with no "showing N of M" and
  claims cut mid-token. Both fixed. **Correction to the audit that raised it:**
  it claimed the cap hides two blocking findings — it does not.
  `FindingsLedger.open_findings` sorts blocking-first, so they survive the cut.
- **Two broken cross-references**: the panel link named `Report_Uniform_Panel.md`,
  the English build's filename, in every Chinese delivery; and `§9 失败史` pointed
  at the decision chain (the failure history is §12). The section is now cited by
  NAME, which is correct in both documents that share the helper.
- **The two `<details>` blocks** in the top-down report are gone. The class table
  and the referee's rules now link to 类目清单.md and 标注规范与裁定规则.md, which
  carry them un-truncated.

### 3. `qmine render RUN_ID [--agents]`

Rebuilds a finished run's deliverables into a NEW generation. Verified on live42:
gen01 → gen02, nine documents, gen01 byte-identical afterwards. Every fix above
is visible in the re-rendered output — 0 `?` rows (was 6), 0 `<details>` (was 2),
§2.1 L2 子意图 rendering for the first time, the findings cap disclosed.

Four defects found and fixed while building it, each recorded in
`report-generators.md`: the auditor had no config gate so `--no-agents` still ran
it against the offline stand-in and wrote `[offline-heuristic] file` into a
deliverable; `render` did not call `_load_env()` so `--agents --provider router`
silently ran offline and reported "10/10 sections verified" for a report no model
wrote; a rendered generation had no `run_summary.json` so re-rendering FROM it
lost the gate ledger; and `GateResult.model_validate(value)` raises because
`name` is the dict key, which would lose the ledger to an `except` two frames up.

### A regression the suite caught, and it was mine

Unifying the gate pairing put a `break` in `gate_metric_pairs`, so one threshold
naming several observed values (`min_kappa` matches `kappa` AND
`self_consistency_kappa`) kept only the first. `_passed_below_threshold` then
stopped flagging a gate whose LATER number is under its bar — and the Chinese
「带保留通过」 prefix that flag adds was the only CJK on a line whose message is
authored in English, so an untranslated gate conclusion reached a Chinese report.
Caught by `test_every_authored_rationale_reaches_the_reader_in_the_report_language`,
localised by stashing files one at a time. Now pinned by its own test.

### Still open

- **The checkpointer degrades mid-run.** The clean demo wrote 5 checkpoints for
  17 phases and ended with `TypeError: Type is not msgpack serializable:
  DecisionRecord` — with no "SQLite checkpointer unavailable" line, so the sqlite
  saver WAS in use. Both `_serializer()` and a default `JsonPlusSerializer`
  round-trip a top-level `DecisionRecord`, all 16 record models are allowlisted,
  and every model reachable from `PipelineState`'s annotations is declared. NOT
  isolated. It matters: `render`'s best state source is the checkpoint (live40
  has 17 rows and recovers everything; live42 has none and loses `observations`
  and `metrics`).
- `zh_panel`'s `fixed` / `waived` lists are still unsorted (`entries.values()`).
- The reference shelf has no English build — `builder` emits it only under `zh`.

### State

- **605 tests passing**, `ruff --select F src/qmine/` clean.
- `runs/live42/gen01` remains byte-identical to what it delivered.

---

## Session (2026-08-31, later) — CLAUDE.md and HANDOFF audited against the code

Both files were audited claim by claim rather than edited from memory. Every
command in CLAUDE.md was run, every test name checked against the suite, every
file path resolved.

### CLAUDE.md — what was wrong

- **`448 tests, ~4 min`** — measured 605 in 3 min. Replaced with `~600`: an exact
  count rots on every session that adds a test, and a stale number in the file
  that loads into *every* session is the exact failure it warns about.
- **`26 mechanical checks` / `live39 18/19, live38 2/19`** — the harness has **28**
  checks and the denominators were wrong. Re-measured today: live40 25 pass / 1
  fail / 2 skip, live39 19/7/2, live38 2/9/17. The scores were replaced with the
  PRINCIPLE (always pass a known-broken control; read PASS/FAIL/SKIP, never PASS
  alone) because scores rot too.
- **`live42 lost 6 of 9 report sections to one [grounding false positive]`** —
  overstated. 3 were lost to a false rejection, **3 to empty model returns**.
- `make demo ~3 min` — measured 3.7. Now `~4 min`.
- One over-long line left by an earlier mid-paragraph insertion, and four
  area-specific invariant rows moved to `measurement.md` to hold 200 lines.

Verified sound and left alone: all 37 test names exist; every file path resolves;
`run --resume --run-id` is correct; `16 leaves vs 25 on live40` is correct
(`hierarchy_meta.n_leaves=16`, delivered 25).

### HANDOFF.md — what was wrong

- **§1 was four days stale and 88 lines of session narrative**, contradicting its
  own contract ("overwritten each session; must describe *now*"). It claimed 552
  tests, `25/26 PASS 0 FAIL`, and that no narrative section had ever been written
  by a real model — all false. Rewritten to 38 lines that describe now, plus a
  reading order for a new session. Its fork and confirmed-≠-defective narrative
  was already duplicated in the session log, so it was dropped rather than moved.
- **§2 kept resolved items struck through** where the contract says delete them,
  left orphaned prose under a `### P0 — none open` heading, and listed the same
  English-strings item **twice** — once struck, once open. Rebuilt: 0 struck-through
  entries, renumbered, and today's measurements folded in (the budget
  miscalibration is now quantified rather than asserted).
- **Six top-level headings were undated fragments**, breaking the one-dated-
  section-per-session contract. Grouped under
  `## Session (2026-08-27 night → 08-28)` and demoted, content untouched. The
  exact night/morning boundary is not recoverable, hence the range.
- `(2026-08-28, late)` sorted *before* `(2026-08-28, evening)`, contradicting file
  order. The second is now `later`.

### Two defects the audit surfaced in `render`

1. **A render overwrote the run's own spend record.** `_wire_events` wrote
   `root/usage.json` unconditionally, so re-rendering live42 replaced 702 calls /
   $29.69 with 11 calls / $0.78 — unrecoverably, because live42's teardown bug
   meant no `run_summary.json` held a second copy. `_wire_events` now takes a
   `usage_path` and a render writes into its own generation.
1. **`configs/live.yaml` is the default config** (`cli._load_config`) and it
   declares `provider: router`. Forgetting the routing policy was the one launch
   mistake nothing caught: the router then picks on price and discards lab
   independence, the capability list and every pin.
3. **`auto` asked only `_has_anthropic_credentials()`.** With deepseek, qwen,
   zhipu and openrouter all configured it still resolved to the OFFLINE stand-in,
   because none of them was Anthropic — on a project whose own live config
   EXCLUDES Anthropic. So `auto` could never route at all. It now asks
   `detect().usable`, and warns loudly when the answer is empty.
4. **`p0_provider` gate.** A warning in `run.log` is not enough: with the stand-in
   the pipeline emits a full set of deliverables, every gate passes, and no
   document says the prose was not written by a model. The question "was this run
   real?" now has an answer in the artifacts, recorded either way.

Verified end to end: with keys, a bare `qmine run` resolves to `routed`; with
every key removed, to `offline` with the warning.

**`demo` and `full` are now explicitly offline.** They are the documented ~4-minute
and ~25-minute checks, and the routing default would have quietly turned them into
paid runs on 8,000 and 50,000 rows.

### Correction to the previous entry

The `make live` target no longer hardcodes the corpus: `LIVE_INPUT`,
`LIVE_DOMAIN`, `LIVE_TEXT` and `LIVE_REFS` are overridable, and `LIVE_REFS=`
emits no flag at all — correct for a corpus with no legacy labels. Calling it
"the only way to launch" was wrong, and contradicted the project's own first
line: this runs on **any** query dataset.

### live43 halted at 80 minutes — half the gold set was silently unlabelled

`annotator[b] labelled 1500/3000` with **zero** lost-batch warnings. The cause was
not a lost batch and not batch size:

    annotator_a  136 cached batches, 3400 labels -> 25.0 per batch  {25: 136}
    annotator_b  128 cached batches, 1650 labels -> 12.9 per batch  {0: 62, 25: 66}

Bimodal — 0 or 25, never partial. Probing the real model with live43's own prompt
showed every empty return was byte-identical: `finish_reason=stop`, 309 output
tokens, 833 characters beginning `{"$defs": {"QueryLabel": {"properties": ...`.

**`qwen3.8-flash` was returning the JSON SCHEMA instead of the data**, and
`AnnotationBatch.model_validate()` accepted it: pydantic ignores unknown keys and
`labels` defaults to `[]`, so a schema echo becomes a valid empty batch. No
exception, so `_one`'s three-attempt retry never ran, and 25 rows disappeared per
occurrence with nothing logged.

This is the same shape as `SectionDraft.markdown` defaulting to `""` — a
permissive default turning a failed generation into a successful empty one. It
cost the narrative report six sections in August; here it cost half the gold set,
which is the artifact everything in the top-down route rests on.

Fixed in three places: `_is_schema_echo` rejects the echo before validation in
both `_plain_json_call` and `_salvage`; `_one` raises when a batch returns fewer
labels than queries so the existing retry fires; and the plain-JSON ask now says
not to return the schema. **Verified on the real model** — 4/4 complete batches at
sizes 25, 12 and 6, against 3/4, 3/4 and 1/4 before.

Also fixed while reading the log: the repair message announced "repairing via
plain-JSON mode" for a model ALREADY in that mode, so six lines claimed a mode
switch that had happened three days earlier.

**Why a fresh run and not a resume.** `llm_cache/` is run-level and stores parsed
values, so the 62 empty batches replay on any resume AND on a new generation of
live43 — the fix would never fire. Independently, p2b sits inside the forked
region, which is the case §2 warns never to restart mid-flight. live43 stopped at
$11.75 / 295 calls and stays as evidence.

---

## Session 2026-09-02 — fast mode

**What was asked:** a mode that returns results faster by skipping the
double-checking, delivering three files (two per-route reference documents and
one fully-labelled dataset) without reducing the evidence a user can audit.

**The collision found first.** `fast_mode` already existed and meant "shrink the
grids for a wiring smoke test" — α grid to 3 values, k sweep to 5, gold to 120
rows, researchers to 3. That degrades the ANALYSIS. The requested mode is the
opposite: full analysis, no checking. Shipping both under `--fast` would mean a
user asking for quick results silently getting a degraded smoke test wearing a
production label. Renamed the old one to `smoke_mode` / `--smoke` (its own
docstring already called it that) across 32 sites; `--fast` is now the new mode.
The rename was done with a **word-boundary** regex — `fast_mode` is a substring of
`fast_model`, and a naive replace renames the model tier.

**What fast mode removes** (all second opinions; none decides a parameter):
dual annotation, kappa, the pilot and its self-consistency ceiling, guide repair,
boundary redraw, phase observers, adversarial validation, the narrative report,
the delivery audit, result interpretation. **What it does not touch:** every grid,
the corpus, the gold size, the researcher panel, `propose_grids` (a widener, not
a check), and every `store.put_*` call.

**Three design decisions worth keeping:**

1. **`_annotate_both` returns `(labels, None)`, never `(labels, labels)`.** The
   copy would let all four call sites run unedited and write `kappa: 1.000` — a
   perfect score for a measurement nobody took. The `None` forces each caller to
   state what it does with one reading. `GoldRow.n_annotators` was added so
   `agreed` cannot be misread as agreement.
2. **`deps.gate(skipped=True)`.** `GateStatus` already had `"skipped"` and
   `deps.gate` could not produce it. `p2a_pilot_agreement`, `p2b_kappa` and
   `p2b_annotator_symmetry` now record `skipped`; `passed=True` would have left a
   ledger entry identical to a full run's.
3. **One list drives every banner.** `_fast_mode_drops_the_second_opinion`
   populates `cfg.fast_skipped` as it turns each component off, and
   `fast_deliver._banner()` renders that list. A skip cannot exist without
   appearing in all three deliverables. An unknown key degrades to the raw key
   rather than being dropped — `test_an_unknown_skip_key_is_still_disclosed`.

**Verified by mutation, not by passing.** Three guardrails were deliberately
broken (banner drops unknown keys; `_annotate_both` returns a copy; fast mode
shrinks the α grid) and each was caught by exactly the test that describes it.

**Also changed:** `verify_run.py` takes `@check(..., needs=[...])` and reports
`N/A` — never PASS — for a check whose component was skipped; it also knows which
fast deliverable absorbed each full-mode document, which turned 8 SKIPs into real
checks (17 PASS vs 12). `run_summary.json` records `mode` and `fast_skipped`.
`scaled_requirements` zeroes the silent roles so `qmine models --fast` prices what
will actually run. `make fast RUN=x` added.

**One defect of mine, caught by the suite:** `getattr(ctx.cfg.taxonomy, ...)`
assumed `cfg.taxonomy` exists; the seven annotator-concurrency tests build a
`SimpleNamespace` with only `cfg.llm`. Fixed by fetching `cfg.taxonomy`
defensively too — the same shape as the `getattr(ctx.cfg.llm, ...)` beside it.

**Two defects found by running the render, not by reading it:**

1. **`qmine render` upgraded a fast run to a full one.** `render` builds its
   config from the CLI, where `mode` defaults to "full", so re-rendering
   `/tmp/fastrun/f1` produced the thirteen full-mode documents — 叶清单.md,
   类目清单.md, 统一度量面板.md — with **no banner anywhere in them**. The one
   command whose purpose is re-deriving deliverables was the one that could strip
   the disclosure off them. Fixed by `runner.inherit_mode`, which reads the source
   generation's own `config.resolved.yaml`; the RECORDED `fast_skipped` wins over
   the validator's rebuilt list so a banner describes the run, not today's code.
2. **Every render lost the domain — pre-existing, and not mine.** A full-mode
   render of a `k12_zh` run also wrote "**领域**: `generic`". Cosmetic until fast
   mode, whose deliverable filenames carry the domain key: the render deposited
   `generic_自上而下_….md` beside `k12_zh_自上而下_….md`, one document under two
   names. Fixed in the same function.

**Two more defects, found by running `make demo` after adding a parameter to
`run()` — both mine, both in the p8 delivered-leaf collision check:**

3. **The collision gate passed on a check that had crashed.** `cents` was bound
   only inside `if new_labels is not None and not array_equal(...)`, the branch
   that runs when governance actually rewrote the partition. On every run where
   governance changed nothing, `_resolve_indistinguishable_leaves` raised
   `UnboundLocalError`, its own `except` swallowed it, `still_colliding` stayed
   `[]`, and the gate reported "every delivered leaf is distinguishable from its
   siblings by name" having compared nothing. Present in every offline run this
   session; absent from `med04`, where governance did rewrite the tree — which is
   why no live run had shown it. Fixed by binding `cents` before the branch (when
   governance is a no-op the delivered partition IS the pre-governance one) and by
   giving the gate `skipped=not collision_check_ran` so a crashed check can never
   read as a pass.
4. **That gate never reached state.** `deps.gate(...)` at what is now
   naming.py:711 was called without assignment, so the gate was logged and
   dropped: `run_summary.json` did not contain `p8_leaves_are_distinguishable` on
   any run. Invisible to the router, unable to halt anything, unreadable
   afterwards — the same mistake `topdown.py` documents having made once before
   and found five runs later. Now captured and returned.

   The error handler that hid #3 printed only `(UnboundLocalError)` — no file, no
   line, no message. It now names the frame, which is how #3 was found at all.

**The finance corpora, and two more defects the dry run caught before spending.**
`data/raw/金融query-{250701,260701}.xlsx` — 10,000 rows each, one year apart, same
schema: `original_query`, `wise_pv` (full-log frequency), and a
`query_1st_category` that is CONSTANT ("金融"), i.e. the slice that produced the
file rather than a label. A `finance_zh` domain profile already existed.

5. **A CLI default silently overruled the config file.** `cfg.data.text_column =
   text_column` assigned unconditionally, so Typer's default `"query"` overwrote
   a config saying `original_query`, and p1 halted with `KeyError: 'query'`
   before reading a row. `reference_label_columns` had the same bug, failing more
   quietly — declared columns silently become none. The identical defect had
   already been found and fixed for `provider` eleven lines below, with a comment
   saying "a config option that the command line always wins is not an option";
   these two were left standing. Both now guard on the flag being given, and
   `--text-column` takes a `None` default so "unset" is representable.
6. **`--config` REPLACES the default config; there was no way to extend it.**
   `_load_config` loads exactly one file, so a corpus config stating only a text
   column would have discarded the whole of `live.yaml` — the role pins, the
   excluded labs, and the lab-independence requirement double-blind annotation
   rests on. `_load_config`'s own docstring calls that "the one launch mistake
   nothing catches", and I nearly made it. `QMineConfig.load` now honours
   `extends:` (resolved relative to the file, recursive, extending file wins per
   key), and `configs/live_finance.yaml` uses it.

**`fin01` — the first live fast run** (`金融query-250701.xlsx`, finance_zh,
`configs/live_finance.yaml`). Estimated $4.53 fast vs $8.80 full; both
under-report, because three pinned models publish no price and `qmine models`
says so. Early confirmation from the log: `p0_provider PASSED — provider=routed`,
`p1_reference_columns_declared PASSED — no reference label columns, and the
corpus offers none`, and the alpha sweep ran the FULL seven-value grid
`[0.0, 0.1, 0.2, 0.3, 0.5, 0.65, 0.8]` — fast mode shrank nothing.

Corpus notes for whoever reads `fin01`: template coverage is **18.6%**, below the
20% floor, so template fragmentation rests on a small base — a property of this
corpus, reported not gated. `researcher[legacy_audit]` returned no candidates,
correctly: there are no legacy labels to audit.

**Class CODES do not reproduce across runs; the structure largely does.**
Measured on the two finance runs (same source, one year apart, independently
designed taxonomies):

| | |
|---|---|
| classes | `fin01` 20, `fin02` 19 |
| **exact code overlap** | **0** — 0% of the union |

Zero. Yet the Chinese names line up pair for pair: `LOOKUP_FX_RATE` /
`FX_RATE_LOOKUP` (汇率查询), `CONVERT_CURRENCY` / `CURRENCY_AMOUNT_CONVERSION`,
`STOCK_FORUM` / `STOCK_FORUM_NAVIGATION`, `DAILY_ANSWER_RETRIEVAL` /
`DAILY_QUIZ_ANSWER`, `FIND_INSTITUTION_CONTACT` /
`CUSTOMER_SERVICE_PHONE_LOOKUP`, `VERIFY_PLATFORM_LEGITIMACY` /
`TRUST_VERIFICATION`, `OTHER` / `UNKNOWN_OR_OTHER`, and so on — plus two genuine
SPLITS (`LOOKUP_SECURITY_QUOTE` → `STOCK_QUOTE_LOOKUP` + `FUND_NAV_LOOKUP`;
`LOOKUP_COMMODITY_QUOTE` → `FUTURES_COMMODITIES_QUOTE_LOOKUP` +
`PRECIOUS_METAL_PRICE_LOOKUP`).

**STRONGER EVIDENCE, no corpus confound:** `fin02` and `fin03` ran the SAME file
(`金融query-260701.xlsx`) under the same config, and still share **0 of 35 codes**
(19 vs 16 classes). The fin01/fin02 comparison below was confounded by being
different corpora; this one is not. The architect re-invents its naming
convention every run.

Two `fin03` differences are structural, not cosmetic:
* it MERGED pairs `fin02` split (futures + precious metals -> one
  `COMMODITY_CRYPTO_QUOTE`; opinion + commentary -> `SECURITY_INFO_OPINION`);
* it produced **no catch-all class at all** — `fin01` had `OTHER`, `fin02`
  `UNKNOWN_OR_OTHER`, `fin03` none. Gold is unaffected (unfittable rows go to
  `UNLABELED` and are dropped), but at INFERENCE every row is forced into a real
  class with no escape hatch. Worth watching in that run's delivered
  distribution.

**The 0% is measured; the correspondence is EYEBALLED from the Chinese names and
is not a measurement.** Making it one would mean labelling one corpus under both
taxonomies and computing agreement — worth doing, not done here.

Two consequences that matter now:

1. **Never diff two runs by class code.** It reports that nothing reproduced when
   most of it did. This is the concrete evidence behind CLAUDE.md's existing
   advice to reuse the TAXONOMY rather than the run when cross-run comparability
   matters — `--reuse-taxonomy` is the only thing that holds codes fixed.
2. A few differences look like real corpus change rather than naming drift:
   `fin01` carries `GRAY_APP_DOWNLOAD`, `CREDIT_REPORT_CHANNEL` and
   `CALLER_ID_FRAUD_CHECK`, which `fin02` does not; `fin02` adds `CHART_LOOKUP`
   and `SECURITY_CODE_IDENTIFICATION`. Whether those are drift or a genuine
   year-over-year shift is undetermined and would need the same joint labelling.

**Rule EXECUTABILITY swung 72% -> 0% between the two finance runs, and it is
the architect varying, not a bug.** Same config, same source, one year apart:

| | rules | executable | rejected |
|---|---|---|---|
| `fin01` | 46 | **33** | 13 |
| `fin02` | 52 | **0** | 52 |

The rejection reason is identical in both (`does not fire on the rule's own
example or originating query`). The difference is what the architect wrote:
`fin01` named literal phrases (`还会涨/跌吗、未来走势、亏不亏`), `fin02` named
CATEGORIES (`裸数字代码`, `主观/推荐词`, `具体金额换算`). A category is not a
test, and `rules_against_evidence` correctly refuses to pretend otherwise. **No
code was changed for this** — the check is right and the variance is upstream of
it. Open question: an architect prompt that demands literal markers would make
the rules mechanically checkable, but see
[[qmine-prompt-emphasis-is-zero-sum]] — hardening one requirement here has broken
a competing one before.

I chased two wrong hypotheses first (a quoting-style mismatch, then an extractor
bug) and the measurement killed both before either became an edit. It did surface
one real gap on the way: `_QUOTED`'s character class had 「」 but not 『』, while
`usable_markers` already stripped both. Four `fin02` rules gain genuinely usable
markers from the fix (『净值』, 『主连/合约/期货/连续』, 『k线图/图表/走势图』,
『净值/基金』). That is a real defect and a small one; it explains 4 rules, not 52,
and the test says so explicitly so nobody later reads it as the cause.

**Template COVERAGE and locator REACH move independently — do not read one as a
proxy for the other.** Measured across the two finance runs:

| | template coverage (share of ROWS) | locator reach (share of CLUSTERS) |
|---|---|---|
| `fin01` (2025) | 18.6% | 24% |
| `fin02` (2026) | **36.3%** | **12%** |

Coverage doubled and reach halved. The grid proposer's own note on `fin02` says
why: its 12 phrasing groups are built around stock-code prefixes (600/300/60)
plus 走势图/行情/股吧, so the covered rows pile into a few clusters. A phrasing
group that matches many rows in ONE cluster raises coverage and does nothing for
reach — and reach is what decides whether the K located on the reference frame
generalises to the corpus it is then applied to.

Both runs therefore WARN on `p5_locator_reaches_the_corpus`, and `fin02` warns
harder despite looking better on the headline number. Anyone using coverage to
predict reach will get the sign wrong; I did.

**`fin02` was damaged by a mid-run macOS file-access revocation, and the damage is
instructive.** The grant was pulled while the run was in flight. My session
recovered on restart; the run's own process (pid 91953) never did, so from
17:16 it could not open any file it had not already read.

| phase | state |
|---|---|
| p1-p7 | CLEAN — finished before the revocation |
| p8 | DEGRADED — `families_final: 0` (family naming could not read its prompt), leaf disambiguation skipped |
| p11 | FAILED — `ModuleNotFoundError: qmine.report.builder` (lazy import, first touched after the revocation) |

**Three fixes from this session were validated by the accident, not by design:**

1. `p8_leaves_are_distinguishable` reported **SKIPPED** — naming the failing frame
   (`PermissionError ... disambiguator.md — at pathlib.py:1013 in open`). Before
   this session it would have reported "PASSED — every delivered leaf is
   distinguishable from its siblings by name" on a check that had crashed, with
   no file, no line and no message. This is the exact failure the fix was written
   against, reproduced by an accident nobody could have staged.
2. `qmine render fin02` recovered all three deliverables from artifacts —
   **10,000 rows x 19 cols, 10 sheets**, and the `原始档案位置` table resolved down
   to 2 unresolved entries. Before the store-resolution fix the same render
   produced 8 empty sheets, 0 rows, and 13 x "未生成". Verified on a real run.
3. The render carried `领域: finance_zh` and `模式: fast` — mode and domain
   inheritance both holding on a live run rather than a fixture.

**`fin02` IS a valid verification run and is NOT a reference delivery.** Its
p1-p7 results stand (19 intents, 33 leaves, coherence 3.79, held-out
reproduction 99.3%, no phantom classes in 3,200 gold rows). Its family names do
not: `families_final` is empty, so the documents fall back to
`树审计未覆盖 (治理新建) · 主要叶「…」`. No resume path repairs this — p8
"completed", so `--resume` restarts at p11 and re-runs nothing that was degraded.

`verify_run`: `fin02/gen02` **17 PASS / 6 N/A / 3 FAIL / 2 SKIP** against
`fin01/gen01` 19/6/2/1. All three failures are the damage, correctly named:
the unnamed families, the p11 halt, and the known render limitation on
gold-set provenance.

**The phantom-class fix is CONSISTENT but not independently proven.** `fin02`
carries zero off-schema labels across 3,200 gold rows — but the guard never
fired, and `fin01` hit the condition once in 3,200 (0.03%), so zero is equally
consistent with luck. `test_a_solo_annotator_cannot_invent_a_class` remains the
actual verification.

**Two §2 open questions closed today** (removed there, recorded here per the
file's contract):

* **#17 `verify_run.py` on a rendered generation** — taught it the store's
  cross-generation resolution. It now searches DOWN through generations for
  artifacts but deliberately NOT for documents: an older generation's report is
  the thing a re-render replaces, and reading it would silently verify the
  previous version. Measured on `fin01/gen03`: 8 PASS / 12 SKIP before,
  **19 PASS / 2 SKIP** after.
* **#16 `finance_zh` untested** — narrowed, not deleted. Three real runs
  exercised it; its risk categories, template seeds and `expected_l1_range`
  [15,22] all held (16-20 L1 classes across the three). `sports_zh` and
  `politics_zh` remain untested.

**A wrong number was shipping inside every fast deliverable.** The banner said
"交付文档从 13 份减为 3 份". A full run ships **10** markdown documents (`med04`,
`live44`; `live42` shipped 9). The count appeared in 8 places including the
banner and the `--fast` help text. The banner now states no full-mode count at
all — a reader of a fast deliverable needs to know what they hold and that
nothing was withheld, and a hard number there goes stale silently inside every
shipped document. The docs carry the real figure with its composition.

**TEMPORAL DRIFT: pool the snapshots, never diff two runs.** Two runs do not
produce comparable labels — `fin02` and `fin03` ran the SAME 10,000 rows under the
same config and shared **0 of 35 class codes**. So a year-over-year comparison has
to happen INSIDE one run.

* `tools/pool_snapshots.py` stacks snapshots with a `_snapshot` tag.
* `tools/drift_report.py` joins `labels_full.csv` back **positionally** (verified
  20,000/20,000) — NOT on query text, which fans out on repeated queries.
* `_snapshot` is deliberately NOT passed as a reference label column: reference
  columns are the frame the K locator scores against, so declaring it would ask
  the clustering to find a K separating 2025 from 2026.
* Shares are WITHIN-SNAPSHOT, never raw. 医疗 PV falls 9.74M -> 5.21M (-47%), so
  raw PV reports every class as declining. Row share answers "did the variety of
  asks change", PV share "did traffic change". Row shares get a z-test; PV gets
  none by design — traffic is the population, not a sample.

**The degeneracy check is the one that matters**, and `fin-pool` passes it: of 54
leaves and 17 intents, **0 are >95% one snapshot** (share-of-2025 median 0.51).
The clusters span both years, so the frame really is shared. Run this check on
every pooled run before believing its drift table.

**`fin-pool` measured:** 21 PASS / 6 N/A / 0 FAIL (control `fin03` identical),
17/17 phases, 2.11 h, 252 calls, $4.49, 54 delivered leaves, 34/34 families named.
Pooling was CHEAPER than feared and better in two ways: gold is capped at 3,000
regardless of corpus size, ECE 0.0159 **PASSED** where every single-snapshot
finance run warned (0.0247-0.0289), and 0/59 leaves fell below the coherence
floor. Finding: the 2026 finance mix consolidated onto `LOOKUP_MARKET_QUOTE`
(66.2% -> 79.0% of PV) and stock-forum browsing, while every service and
informational intent receded — corroborated by regexes that never touch the
pipeline's labels (stock-forum +0.7pp in both, exactly).

**Domain profiles: `medical_zh` had ZERO template seeds**, which is how the first
pooled medical run reported "NO template group passed the cohesion check (0/12)"
and rested its alpha decision on untrusted masks. Now 8 seeds, 54.5%/60.1%
coverage across the two snapshots, worst overlap 11-12%. Three decisions, each
measured: `有哪些` REJECTED as a seed (a phrasing, not an intent — it enumerates
symptoms, foods, types and hospitals alike); `禁忌/副作用` REJECTED (67% of its
matches also matched efficacy — a facet, not a family); efficacy SPLIT by dosage
form into drug/substance, disjoint at 835 + 1,917 = 2,752 exactly.
`people_zh`, `film_tv_zh`, `education_zh` added the same way.

**An empty text cell halted a launched run.** Under pandas 3.0 `.astype(str)` no
longer turns NA into "nan", so one blank in 20,000 rows reached `char_profile` and
died with `object of type 'float' has no len()`. `edu-pool` halted on row 17,717
AFTER launch. p1 now drops empty-text rows with a loud count and REFUSES a corpus
that is >5% empty, because that is a broken export and analysing what survives
hides it.

**Open, and deliberately not resolved here:** no paid fast run has been made, so
the single-annotator gold set is untested against a real annotator. `med04`'s
40.3% vs `live38`'s 78.3% annotator-a win rate says which annotator is better
flips by corpus and model, so `primary_annotator` is a recorded default, not a
finding.


---

## Session 2026-09-10 — 2026 search vs AI assistant: cleaning, one-instrument intent, report

**Delivered (post-run analysis only; no pipeline source changed).**

- **Reports:**
  - `docs/SEARCH_VS_ASSISTANT_2026.zh.md` (main report, §0–§9 plus appendices A–E)
  - `docs/SEARCH_VS_ASSISTANT_2026_领域深挖.zh.md` (five domain deep-dives)
  - figures `docs/img/sva2026/fig1–fig6`
- **Correction box** added to `docs/DRIFT_2025_2026_WITH_ASSISTANT.zh.md` §3.3. Its "15 comparisons all point the same way" mixed search-head rows with assistant-tail rows. At matched depth both surfaces have a median query length of 6.
- **Reproduction package:** `analysis/sva2026/` holds 89 scripts and `work/` (48 MB derived data). The exported `sva_report_tables.py` and `sva_report_tables_intent.py` regenerate the inserted tables byte-for-byte.

**New tools** (not imported by tests; `ruff --select F src/qmine/ tools/` is clean):

- **`tools/clean_assistant_functional.py` v3.** Tiers are S1 repeated, S2 template/feature, S3 card, S4 suggested chip, S5 headline, C1 content-free, C2 feed-control.
  - First audit (on v2): removal precision was sound (S1 0.964, S2 0.851, C1 0.966, C2 0.998), but 11.3% of kept head rows were still system text.
  - Second audit (fresh samples): kept-head miss rate 7.0% [4.6, 10.5], with 人物 head at 20.0%. Precision is 96.7% for new removals, 96% for S4 and 90% for S5.
  - A census found 51.6% of kept head U05 rows are untagged headlines or topic strings. Corrected, pooled head U05 drops from 3.95% to 2.05% (人物 head 11.0% → 3.7%; the correction also removes the sample miss rate from the denominator).
- **`tools/unified_intent_frame.py`:** 13-class frame plus a crosswalk covering all 112 classes of the six runs.
- **`tools/label_unified_intent.py`:** blind, shuffled labelling with one prompt. Primary model is deepseek-v4-flash at `max_tokens=16000` with a preflight batch; qwen3.8-flash labels a 20% subset.

**Measured, and worth not re-learning.**

- **Crosswalk shares are not valid across surfaces.** Crosswalk and unified labels agree on only 35.9% of assistant head rows, against 76.6% for search head. For example, ai04 "投资理财建议" is 69.5% bare tickers and quotes in 金融.
- **Reliability.**
  - DeepSeek vs Qwen: κ 0.855 (n=7,032); assistant tail 0.761.
  - Test-retest on 1,000 anchors with different batch neighbours: κ 0.850.
  - Every intent difference ≥3pp has the same direction under both models: 32/32 against search top1000, 30/30 against top10k.
- **Search PV floors dwarf the assistant's.** The search 10,000th query (158–477 PV) outranks the assistant top1k floor (44–280). So search top10k is labelled too and reported as a second lens. Search intent mix shifts a lot between top1000 and top10k in 教育 (U02 77% → 49%) and 医疗.
- **The assistant random1k has no search counterpart.** 81.2% of its 5-domain rows (65.7–90.5% per domain) have no search top10k neighbour with cosine ≥0.70, against 5.5–27.9% for search's own deepest 1,000 rows; the gap holds after length control. Do not quote the ladder's D level (56.0%) as the no-neighbour share: D is what remains after the conversation (F) and personal-case (E) rows are taken out first.

**Defects found and fixed in the process.**

1. **pandas 3 string regex runs through RE2** (`\w` is ASCII-only). The audit-suggested `[^\W_]` read every Chinese query as empty, and 81.7% of head rows went C1. Fixed with `str.isalnum` plus `_check_engine_semantics()`. Memory `qmine-pandas3-re2-regex`.
2. **S5 `来了$` caught a variety-show title** (`爸爸回来了`). The rule now requires ≥7 characters.
3. **Label resilience.** Labels were reconstructable from `raw_ds.jsonl`: the reconstruction matches the delivered labels 14,452/14,452, so a failing second-model pass cannot lose paid primary labels.

**Open, deliberately not resolved.**

- **Audit-2 rules not applied.** They are validated on held-out rows but would desynchronise every table from the domain deep-dives, so they belong in a v4 for the next data: `有没有更多` chips, `#…#` feed copy, the `我想对作文《` template, the S5 news-keyword exemption, and three search-S5 exemptions.
- **Residues known in v3.** 影视 has 31 `《X》的结局是什么 / 给我《X》的完整演员表` rows. 金融 has 10 identical `XX未来有上涨空间吗` rows (PV 44–76). 人物 head U05 topic strings are handled by the census correction, not a rule.
- **Same-period data** (search and assistant ~2 months apart) and a typed-vs-tapped source field in the assistant log are the two things that would remove the report's largest confounds.

**Adversarial verification of both documents.** An independent reviewer re-derived every number, example and citation read-only: 54 issues, 0 critical, 11 major, 43 minor, all applied. The majors were overclaims rather than wrong tables:

- a removal-precision summary that ignored search S5's 68.2%;
- a medical question-rate "reversal" whose CI contains 0;
- the ladder D level quoted as the raw no-neighbour share;
- wrap-pair and nearest-neighbour destinations conflated;
- per-domain gaps below the audit's error bounds presented as findings;
- head-vs-tail differences written as interface differences;
- query overlap read as user overlap;
- the 人物 U05 11.0% left uncorrected in the domain summary.

The generated tables matched their generators with 0 mismatches. Every edit went through an assert-guarded script (`sva_apply_verifier_fixes.py`, in the package) that writes nothing unless every anchor matches exactly once. After the Qwen labels on the search top10k extension were promoted, T14 is n=7,032 and κ 0.855.

**Tests:** 739 pass on the full suite (`-x`, no failures); `ruff --select F src/qmine/ tools/` clean.


## Session 2026-09-12/13 — POOLED-5: five sources per domain, one taxonomy each

**What was mined.** Per domain, one corpus: 2025 search top-10k + 2026 search top-10k + assistant
top1k + assistant random1k + (finance, medical) assistant voice 1k. Built by
`analysis/pooled5/build_pooled5_corpus.py`; assistant rows carry the audited v3 cleaning tiers,
search rows get the same C1/S5 rules, voice gets content-free rules only (no PV, so no PV-gated
rule can fire). Only `tier == user` rows are mined; the rest ship in the deliverable's own sheet.

| run | rows | note |
|---|---:|---|
| `fin-pool5` | 22,934 | 18 L1 classes |
| `med-pool5` | 22,952 | 20 |
| `edu-pool5` | 21,799 | 20; `researcher_pragmatic_intents` refused by Zhipu, 4 of 5 angles |
| `film-pool5` | 21,934 | 17, no catch-all class |
| `ppl-pool5b` | 21,804 | 19; **relaunch** with every Zhipu role rerouted |

**Why `ppl-pool5b` exists.** `ppl-pool5` gen01 halted at p7_audit: Zhipu's content filter refused
the risk sentinel 3/3 on a corpus of Chinese public figures, and the fallback object had no
`model_dump`. gen02 (a resume) halted again at the branch-join guard (open question 0v). The
relaunch routes researcher/domain_scout/maintainer/reporter/referee/risk_sentinel to
Moonshot/DeepSeek — `researcher_pragmatic_intents` then returned **12 candidates** (was 0) and the
sentinel returned **8 findings**. The selection rule was written down *before* results were visible:
`analysis/pooled5/work/people_run_choice.md`.

**Measured, and worth not re-learning.**

- **A string gets ONE label per run.** Zero strings carry two `td_l1` labels in any domain
  (duplicated rows: 5,638–13,621 per domain). So "the queries both years share kept their label"
  is a property of the instrument, not a finding, and the year-on-year difference can only come
  from turnover. Real concept drift is invisible to this method.
- **Depth beats interface in 4 of 5 domains.** TVD: depth .260–.525, interface .239–.643, time
  .074–.256, typed→voice .373/.410. Education is the exception (interface .643). Every contrast
  clears its same-source null band (.03–.13, `cross/tvd_ci.csv`).
- **The equal-depth control goes the other way from the obvious objection.** Cutting search to its
  own top-1,000 makes the interface TVD *larger* in all five domains (+.026 to +.185).
- **Cross-domain regularities, measured without any class mapping** (one regex, five domains):
  on the depth axis six markers × five domains are all same-signed and significant; on the
  interface axis 疑问句 splits direction (人物 +11.8, 金融 +13.4, 影视 +9.7 vs 医疗 −6.7, 教育 −4.4).
  Only 第一人称 (5/5) and 是非核实 (4/5) replicate across the interface change.
- **Adding assistant rows did not measurably change how search rows are carved up.** AMI 0.59–0.74
  against a **same-data-twice baseline of 0.774/0.687** (`fin02`/`fin03`). What did happen: classes
  whose rows are majority assistant — finance 4, film 2, people 2, education 1, medical 0.
- **Domain tuning cuts both ways.** It splits (education's generic 裸实体 → 按裸校名检索 44% +
  启动翻译工具 30%) and merges (finance's 查询金融资产实时行情 is 52% 裸实体 + 46% 事实与数值 under
  the generic frame). It also has no class for assistant-native behaviour: of education's 61.9%
  assistant-head residue, 13.5% is 会话与系统指令 and 16.9% 裸实体 under the generic frame.
- **Two independent routes agree less on assistant rows.** AMI(cluster leaf, top-down intent):
  education .75 on search vs .39 head / .32 tail; finance .43 vs .36 / .20. Independent of the
  classifier's own confidence.

**Verification.** Every domain deep-dive was written by one agent, re-derived claim by claim by a
second, then revised against that report; the cross-domain synthesis went through two rounds and
lost three headline conclusions. Counts: 医疗 112/122, 教育 160/176, 金融 108/128, 影视 98/120,
人物 121/130 confirmed; synthesis round 2: 90/96, 0 critical. Two defects the verifiers found in
MY code: `p5_turnover.py` counted voice rows in the assistant denominator (only finance and medical
have voice, so it diluted exactly those two and flipped the ranking), and a voice-overlap figure
copied medical's number onto finance. Both fixed; the round-2 verifier also confirmed
`pooled5_common.newcombe` is a correct Newcombe hybrid-score after the first verifier's own
implementation was found to have its bounds contributions swapped.

**What the report's verification actually caught** (three lenses on the assembled document —
numbers / overclaim / consistency — 458 items, 53 findings, then a repair round and a re-check that
found 3 more the repair had introduced). The five that would have misled a reader:

1. **"Shared strings contribute 0 to intent differences" was backwards.** Removing the strings both
   surfaces share RAISES the interface TVD (finance .281→.595, people .309→.542, film .239→.302,
   education .643→.651, medical .308→.308), so every interface distance in the report is a LOWER
   BOUND. The depth line is unaffected — assistant head and assistant tail share **zero** strings in
   all five domains, which is why the repair's first attempt at this fix (extending "lower bound" to
   the depth line) was itself wrong and had to be undone.
2. **The shared-bucket composition was a different quantity than its sentence claimed.** "字面共用的
   那一半 … 人物 100%/医疗 96%/影视 84%" was the U-purity of each domain's largest shared class over
   ALL its rows. Recomputed over the shared rows themselves (172–513 per domain, U coverage 100%):
   人物 95%, 影视 57%, 医疗 50%, 金融 `事实与数值` 45% + `裸实体` 42%, 教育 `无法判定` 51%.
   `cross/shared_bucket_u_mix.csv`.
3. **A 20-row cell caveat was attached to the wrong class**, making people's `核实与动态` read as
   1 row when it is 98 (98→155). `个案判断与建议` is the class with the tiny cells (3/0/1 on the
   assistant-head side for education/film/people).
4. **A partial script run silently destroyed a cross-domain table.** `p5_vs_unified.py 人物`
   overwrote `cross/vs_unified_*.csv`, so T10 shipped with one domain while the prose discussed
   three others. Both `p5_vs_unified.py` and the report now guard against it.
5. **`p5_turnover.py` counted voice rows in the assistant denominator** — only two domains have
   voice, so it diluted exactly those two and flipped the ranking (finance 20.0%→26.5%,
   medical 9.8%→13.9%).

**Pipeline change.** `graph/nodes/naming.py` only: the risk sentinel's except branch now binds a
real `RiskReport` instead of `SimpleNamespace(findings=[])`, which had no `model_dump` and killed
p7 fifty lines later — the exact outcome the branch's own comment says must not happen. Regression
test reproduces the production traceback against the old code.

---

## Session 2026-09-13 (evening) — 逐类 × 逐快照：每个意图、每个叶在五个快照上的对照

主报告问「两个界面整体差多远」，用的是汇总距离。团队要的另一件事是**逐类**的：每一个意图、每一个
聚类叶，在 2025搜索 / 2026搜索 / 助手头部1k / 助手随机1k / 助手语音1k 上各占多少、哪些类只出现在
某个快照。本节只做**后置分析**——`src/` 一行未改，五次挖掘运行的产物一个字节未动。

### 交付

每个领域一套，写进各运行自己的 `postprocessed/`（原始产物保持不动）：

| 文件 | 内容 |
|---|---|
| `runs/<id>/gen01/postprocessed/<stem>_意图与聚类叶_跨快照对比.zh.md` | 报告正文（11.9 万–21.9 万字符，936–1,551 行表格）：四个层级（L1 意图 / L2 子意图 / 家族 / 叶）各七张表 + 逐意图与逐叶卡片 + 六张图 + 目录 |
| `runs/<id>/gen01/postprocessed/<stem>_意图与聚类叶_跨快照对比.xlsx` | 26–28 页全量表 |
| `runs/<id>/gen01/postprocessed/img/*.png` | 六张图 |
| `analysis/pooled5/work/<领域>/snapshot_classes/` | 全部中间表 + `narrative.md`（分析员写的叙述）+ `verify.md` |

新脚本四个，都在 `analysis/pooled5/`：`p5_snapshot_classes.py`（表 + 工作簿）、
`p5_snapshot_figs.py`（图）、`p5_snapshot_report.py`（渲染 + 填叙述 + 目录）、
`p5_snapshot_verify.py`（叙述的机械复核）。README 与主报告附录 B 已指向它们。

### 三个测量上的决定

1. **「只在某快照出现」必须配可检出性。** 助手快照 n≈900–1,000，0 条的单侧 97.5% 上界仍有约
   0.39%；同样 0 条落在 10,000 行的搜索快照上，上界只有 0.037%——**差一个数量级**。所以每个 0 条
   都配了「该类在其余快照的合并占比 × 这个快照的 n」得到的期望条数与 P(0)，据此判
   `真缺席 / 偏少但证据弱 / 不可判定`。报告只引判定，不自己解释 0。

2. **「独占」这个口径几乎没有信息量，换成「特征类」。** 一个类只要在别的快照里出现过一条就不算
   独占，实测五域四个层级里独占单快照的类总共只有 1 个（影视的一个 L2）。改用：该类在这个快照里
   的占比，**与其余每一个快照逐一做 Newcombe 检验都显著更高**。这个口径能承重，快照越多越严。

3. **两个指数都给，写清分母。** `指数` = 快照内占比 ÷ 全语料占比（与已交付的 `class_profile.csv`
   同义，分母里 87% 是搜索行）；`均衡指数` = 快照内占比 ÷ 各快照占比的**未加权均值**。后者把
   「搜索行数是助手十倍」除掉，热图用的是它。

### 测出来的结果（五域一致）

- **L1 意图层：没有任何一个类只属于某一个快照**（五域全部）。四个层级上**「仅助手」一律是 0**——
  每一个在助手里出现的类，在 20,000 行搜索里也都出现过。这一侧 n 够大（助手里占 0.05% 的类在搜索
  里期望 10 行），所以这不是检出力问题。**但它是关于交付分区的陈述，不是关于需求的**：体系拟合
  在一份 87–92% 是搜索行的语料上，助手独有的表达如果没有类目，会被并进残余类。
- 通过可检出性判定的「仅搜索」全域只有两条，都在教育：`查询考试成绩与出分动态`（L1）与
  `查询高校一本二本层次`（叶），两条 P(0) 都是 0。
- 界面差异显著的类：金融 13/18、医疗 14/20、教育 15/20、影视 13/17、人物 15/19（L1 层）。
- 新发现的一类结构：**份额测不出差别、内部的叶完全换掉**。逐域都有，例如金融
  `查找官方渠道与客服入口`（2025搜索 vs 助手语音1k 份额差区间含 0，意图内叶分布 TVD 0.5707），
  医疗 `解读用户本人的检查数值或报告分级`（界面差不显著，内层 TVD 0.6784）。表 7-A / 7-B。

### 隐私护栏：机械取例第一版选错了行

逐类卡片的例子是机械选的（该类该快照流量最高的可引行）。第一版的护栏是「风控图层命中 + 一组
露骨词/号码串正则」，它**漏掉了「十岁苗条小女孩」这种「低龄指向 + 外貌描述」的写法**——那两行
出现在人物领域一个被命名员和风控哨兵同时标记的叶里，并被选成了该叶的代表。

考虑过按叶屏蔽（不引任何被标风险的叶），实测代价太大：影视会掉 90% 的可引行、金融 60%——因为
那些叶的风险是「不该那样作答」，不是「这句话不能被引用」。最后落在**三层窄护栏**：

1. 风控图层命中的行（`risk_screen.json` 的 `flag_mask_indices`）；
2. 与领域无关的硬规则：露骨内容、**低龄指向 + 外貌/性相关词（相邻）**、
   **年轻女性称谓 + 身体/情色描述（共现，不要求相邻）**、软色情图片检索、可识别到个人的联系方式
   与证件号、名誉与非自愿私密影像、成人向题材标记；
3. **运行自己的风控机器点名过的具体字符串**（`tree_naming.json` 的 findings.evidence /
   risk_reason / rationale 里加引号的串，`risk_screen.json` 各类的 exemplar / samples）。

代价：可引行 94.2%–99.3%。共现那一条在全五域只多拦 3 行，但拦的正是长 prompt 那一类（相邻规则
看不见隔着几十个字的两类词）。护栏只影响**引用**，每一张表的分母仍是全部行。

### 复核

- 表：用**另一套实现**（不 import `pooled5_common` 的任何 helper）重算了 5,347 个矩阵/界面/缺席值、
  2,209 个 TVD / Cramér's V / Newcombe 值、735 个特征类/归属/排名值——**0 处不符**。
  逐类占比还与已交付的 `shares_td_l1_name.csv` / `class_profile.csv` 逐格比对通过。
- 叙述：五个领域各一位分析员写、一位独立复核员逐条重算，再修订。
  `p5_snapshot_verify.py` 把叙述里的每个数字回查到表、每条引文回查到真实且可引的行——
  五域全部 **未匹配数字 0 · 问题引文 0**。
- 全文引文核查（不只叙述，含卡片与代表串）：**4,178 条引文，0 条不存在、0 条命中护栏**。
- 图逐张读回。**热图第一版的字色规则是照发散色标写的**（低值=深蓝→白字），套到顺序色标上成了
  浅黄底白字，整片读不出来；改成按格子实际亮度选字色。
- 测试 **741 通过，exit 0**；`ruff --select F src/qmine/ tools/` clean。`src/` 未改动。

### 复核员抓到的四类错，以及它们暴露的三个工具缺陷

五位独立复核员逐条重算后提出的必须改项，修订员各自复算后**全部成立**（只有一条复核员给的替代说法
本身是错的，被驳回）。四个代表：

- **人物**：叙述的开头断言「这个领域没有一个类是某一侧独有的」是**错的**——L2 的
  `PERSON_ATTRIBUTE_LOOKUP__2`（搜索 82 行 / 0.411%，助手 0 行，期望 7.55、P(0)=0.0005）与
  `GROUP_ROSTER_ENUMERATION__3`（期望 5.99、P(0)=0.0025）都判真缺席，而前者全文一次没出现。
- **医疗**：叙述写的可引行 21,800 与同一份报告附录的 21,763 直接打架。
- **金融**：叙述把引用护栏说成只拦了 6 行，实际 150 行（5 + 59 + 91，并集），也与附录打架。
- **影视**：一个「特征类」的 40 行里 22 行是同一部 8 月下旬在播的剧，剔掉后三对区间全部含 0——
  姊妹报告《领域深挖》早已写明这个单剧混淆，叙述又踩了一次。

这四类各自暴露了一个工具缺陷，都已修：

1. **机械检查的数字池漏了 `summary.json`。** 于是正确的 `quotable_rows=21763` 会被判未匹配，
   而写错的 `21,800` 恰好能在别的表里匹配上——**把对的拦下、把错的放过，比没有检查更坏**。
   现在池子包含 summary.json 的全部数值与三层护栏的逐层条数。
2. **内层 TVD 没有同源噪声上界。** 两位复核员各自补算了一遍才敢下判断。现在
   `intent_leafmix.csv` / `leaf_intentmix.csv` 带 `同源噪声上界` / `超出噪声` 两列（口径与
   `pairwise_tvd.csv` 一致，300 次对半切取 95 分位）。实测各域只有 37%–94% 的格子超出噪声。
3. **自助法区间不可复现。** 原来是一条模块级随机流，任何一处新增抽样都会把后面每个领域的区间挪动
   一点；复核员引进正文的区间端点，下一次重跑就对不上（实测 14 处）。现在按
   (用途, 领域, 层级, 快照对) 各自播种，**单独重跑一个领域与在五域批量里跑逐位相同**（已验证）。

### 补了噪声上界之后，五个领域各有结论被推翻

官方的 `同源噪声上界` 列加进去以后，又跑了一轮专门的复核（每域一位），结果五个领域**都有**要改的：

- **金融**：`查询A股股票及板块信息` 内层 TVD 0.0513 < 上界 0.1065。原文拿它证「叶内部意图分布也稳」
  「换掉的是标的，不是需求」——**两处都是反向误用**：低于上界既证不了换了，也证不了没换。
  另一处 0.0928 < 0.1246，据此下的「所以这个位移不是时间造的」整句删除。
- **医疗**：官方列**推翻**了「构成没动」——原文只看了 助手头部×语音 一格（0.3179 < 0.3398），
  而十对里有三对超出噪声。「四种组合都能找到实例」这句随之删除。原文「离上界还有很远」也不成立
  （0.1722 对 0.2053，只差 0.033）。
- **教育**：0.0329 < 0.0869，「两个搜索年内部没换」改为测不出；另修正两处过期的
  `pairwise_tvd.csv` 噪声上界，其中一处是**定位到了错误的行**。
- **影视**：0.0385 < 0.1076，改为两个方向都无证据；三处「自算 2000 次重抽」换成官方列。
- **人物**：导读一处把方向**写反了**（0.0294 > 上界 0.0254，是可测的小变化，原文写成「低于噪声底」）。

终检：五域 `未匹配数字 0 · 问题引文 0`；全文 4,181 条引文 0 条不存在、0 条命中护栏；
结构审计 0 问题；叙述引用的 26 个内层 TVD 全部能在表里定位，**超出噪声的都写出了上界，未超出的
都写成了双向测不出**。表的可复现性也验证过：单独重跑一个领域与在五域批量里跑，10 张抽样表逐字节相同。

### 两条排版约定（新加，会被机械检查）

- **「」只包语料里的真实 query。** 叙述里强调词组一律用 “”。运行自己写的类定义里也有 「」，
  渲染时统一换成 “”（query 文本一个字不动）。
- query 自己带 「」 的（例如 `「AI视频」一家三口温馨比心`），外层改用 『』，原文不改。

---

## Session 2026-09-13 (night) — 两个新垂类：书籍文档 / 软件

用户给了两个从未分析过的垂类（各 2 份搜索导出 + 1 份助手导出），要求：先研究并写领域档案，
再跑挖掘（fast），再做与五域同样的全套后处理与分析。本节记录**运行之前**已经定下来的部分；
运行结果与报告见本节末尾。

### 数据：新导出与已审计的助手语料是同一份

两份助手 xlsx 与 `data/raw/ai_assistant_pooled.parquet` 逐条比对：**query 集合完全相同**，
`search_num` 与 `l2` 一致率 0.987–1.000（差异只来自重复串取首行）。所以助手侧走**同一条已审计的
v3 清洗路径**（`assistant_tiered.parquet`），结果与已交付的五域可比，不另开分支。
搜索侧文件名不是 `<domain>query-<yy>.xlsx` 那一套，builder 加了 `SEARCH_FILE` 映射。

语料：**书籍文档 21,786 行**（9,988 + 9,995 + 850 + 953）、**软件 21,920 行**（9,997 + 9,999 + 940 + 984）。
**两个域都没有语音导出**，所以是 4 个快照，不是 5 个。已验证五域语料重建后行数逐个不变。

### 领域档案：从测量出发，不从标签出发

`configs/domains/books_docs_zh.yaml`、`configs/domains/software_apps_zh.yaml`。
每一个占比都是 `str.contains` 在建好的语料上实测的，PV 一律来源内归一。

**最重要的一条：类目名与内容对不上。**
- **书籍文档不是书**：图片/头像/壁纸/表情包/素材 **28.74%** 行、漫画 **11.80%**、
  可直接发的文案 **6.91%**、文档模板 **3.82%**、小说 **0.21%**。从标签出发会把前两大类的量级
  搞错一个数量级。
- **软件主要是导航**：下载/安装 **20.36%**、官网/入口 **10.57%**、社交平台机制 **22.33%**
  （后者是实体名不是措辞模板，故意没做成种子——它与 howto_setting 重叠 927 行）。

模板种子并集：书籍文档 32.9%、软件 36.4%，都在档案约定的 20–40% 内。运行实测**种子全部存活**
（书籍 7/7、软件 7/8，`file_convert` 太小被吸收），另各长出 5 个挖掘族。

风险类目全部实测（行数 / 来源内搜索 PV 25→26）：

| 类目 | 行数 | 搜索 PV | 方向 |
|---|---|---|---|
| 未授权漫画聚合 | 992 (4.55%) | 4.94% → 6.18% | 涨 |
| 成人漫画平台 | 110 (0.51%) | 1.52% → 1.41% | 平 |
| 翻墙工具 | 779 (3.55%) | 6.29% → 7.33% | 涨 |
| 破解/内购解锁 | 131 (0.60%) | 0.41% → 0.18% | **缩**（91 → 36 行） |
| 口令/侵入/监看 | 122 (0.56%) | 1.28% → 0.92% | 缩 |
| 未备案分发渠道 | 80 (0.37%) | 0.53% → 1.18% | **翻倍** |
| 成人 App 分发 | 72 (0.33%) | 0.21% → 0.12% | 缩 |

法条锚点来自当次检索：两高《侵犯知识产权刑事案件解释》2025-04-26 施行（刑法 217/218）、
《计算机信息网络国际联网管理暂行规定》6/14 条（个人翻墙是**行政**责任，≤¥15,000；提供工具才
可能到刑法 285(3) 或非法经营）、刑法 253-1（行踪轨迹/通信内容 50 条为情节严重）、288/225
（干扰与器材）、363/364（淫秽物品，快播 2016）、工信部 App 备案。

### 三件只有测量才能发现的事

1. **`色` 作为成人内容信号完全没用**——它匹配 颜色/红色/色卡。第一版正则据此报了 520 行，
   真正的信号是平台名（e站/jmcomic/哔咔/歪歪/土豪），110 行。
2. **`胸`/`大腿` 单独做触发词会误伤**——匹配到一件摇粒绒卫衣和一只青蛙的大腿。收紧后 15 行，
   零误报。
3. **语料里有刻意绕过过滤的写法**：`ⅴpn加速器免费`（U+2174 罗马数字五）、
   `wⅰf|万能钥匙下载`（U+2170 + 竖线）。

### 一个真缺陷：`screen_risk` 不传 `case=False`

`ops/audit.py:239` 的 `q.str.contains(_noncapturing(pat), regex=True, na=False)` **没有** `case=False`，
所以只写小写的模式会漏掉 `VPN` / `Telegram` / `EhViewer` / `E站`。在 软件 这种拉丁字符密集的语料上
实测漏 **126 行**（circumvention 653 → 779）。新档案的模式全部加了 `(?i)`（`_noncapturing` 的
`\((?!\?)` 前瞻会跳过 `(?i)`，已验证）。

**已交付的五域档案有同样的缺口，但没有改**：实测影响是 影视 **1 行**、医疗 **0 行**
（`ai_assistant_zh` 未在本批语料上测）。为 1 行改动会让已交付报告里的 risk_screen 数字失效，
不值得。要改就要连带重跑并重新核对那几份报告。

### 批次（cohort）：跨领域产物必须说出自己是哪一批的

加这两个域以后 `DOMAINS` 有 7 个，而已交付的五域报告和它引用的 `work/cross/*.csv` 全是五域的。
任何一次重跑跨领域脚本都会把那份报告底下的表换成七域——数字全变而正文不变。
所以引入 `COHORTS`（`pool5` / `new2` / `all7`）+ `cross_dir()` + `work_file()`：

- 不设 `P5_COHORT` 时行为与加新域之前**逐字节相同**（`cmp` 验证过
  `route_agreement_by_source.csv` / `concentration_by_source.csv` / `residue_breakdown.csv`
  / `semantic_nn_summary.csv`）。
- **试运行时真的抓到了一次泄漏**：`p5_cross.py` 把五域的 `semantic_nn.parquet` 汇总成了
  new2 批次的 `semantic_nn_summary.csv`（17 行，第一行是 人物）。原因是 work/ 下有一批
  **单文件、不按域分**的产物。现已全部按批次命名。
- 空批次的扫描脚本从 `KeyError('source')` 改成一句明确的 SystemExit。

### 这个批次做不了、五域做得了的两项分析

- `p5_taxonomy_delta.py` 需要同样行的**纯搜索**对照运行（`fin-pool` vs `fin-pool5`）。
  没有 `book-pool` / `soft-pool`，所以没有第二个框架可比。要补就得每域多跑一次挖掘。
- `p5_vs_unified.py` 需要 13 类通用框架标在同样的串上。实测覆盖率：
  书籍文档 **0.5%**（63/11,798）、软件 **0.1%**（14/11,923），五域全部 100%。
  **在 0.1% 覆盖率上跑这个比较没有意义，所以整项略去**，报告里要写明是略去不是没发现。

### 跑完了：两次都没有 halt，交付结构如下

| 领域 | 运行 | 用时 | 调用 | L1 | L2 | 交付叶 | 交付家族 | verify_run |
|---|---|---|---|---|---|---|---|---|
| 书籍文档 | `book-pool5/gen01` | 63 分 | 210 | 20 | 59 | 30 | 21 | 21 PASS / 0 FAIL |
| 软件 | `soft-pool5/gen01` | 65 分 | 214 | 25 | 58 | 39 | 26 | 21 PASS / 0 FAIL |

两次都是 `mode=fast`、`provider=routed`、`halted=False`。控制组 `runs/ppl-pool5/gen01`（上一轮
halt 掉的那次）在同一套检查下 3 FAIL，含 `p7_audit: SimpleNamespace has no model_dump`——
所以这套检查是能区分的，不是见谁都 PASS。

**预防性改路由的结果**：两跑都干净地过了 `p7_audit`（人物 halt 过两次的那一相），哨兵分别给出
6 条与 12 条发现，全程没有 contentFilter。**但这只证明预测的故障没发生，不证明改路由是必要的**，
config 里就是按「这是一个预测」写的。

### 交付物（每域一套，写在各自运行的 postprocessed/ 里）

`<stem>_query_挖掘结果_含来源.xlsx/.csv`、`<stem>_意图与聚类叶_跨快照对比.zh.md`（书籍 25.5 万字符 /
软件 28.7 万字符）、同名 `.xlsx`、`img/` 六张图。跨批次表在 `work/cross_new2/`，
T1–T16 在 `work/report_tables_new2.md`——**都没有碰 `work/cross/` 与五域的那份**。

### 本批最硬的一条：七个领域里第一个「仅助手」类

**软件 `生成露骨性图像编辑`（NON_CONSENSUAL_EXPLICIT_IMAGE_EDIT）：51 行全在助手侧
（头部 33 / 随机 18），两个搜索快照合计 0 行；同发生率下搜索侧期望 530.04 行，P(0)=0.0000，
判定真缺席。** 已交付五域加上书籍文档，四个层级的「仅助手」全是 0，这是唯一例外。

机制上说得通：「P掉外衣」「把衣服变成透视装」是对着**手里那张图**下的编辑指令，搜索没有对应
写法。生成面造出了一个搜索面无法表达的需求。33/51 在助手**头部**，不是长尾。

### 领域档案是 hypothesis-first 写的，所以它漏的东西有规律

两个域的 agent 都找到了档案的盲区，而且最大的一类都比档案里已经写了的类大：

- 软件：档案 5 类 → 架构师另给 4 类（`获取灰色辅助或作弊工具` 101 行 > 档案的 80 / 72 两类）、
  哨兵另给 12 条（HTML 载荷 20 行、AI 换脸 14 行、伪造证据 24 行、门禁卡克隆 2 行）；
- 书籍文档：档案 4 类 → 哨兵另给 5 类（彩票谜语伪装、数字暗语 78 行、暴力图像生成、侮辱话术 29 行）、
  架构师另给自残轻生（**18 行，比档案原有两类都大**）与情绪倾诉。

**两条具体教训：**
1. **彩票谜语这个模式仓库里早就有** —— `education_zh.yaml` 的 `zodiac_lottery_riddle`
   在教育语料上 463 行，而书籍文档 PV 前 25 就有「脑筋急转弯大全6-12岁」。模式在隔壁档案里、
   语料也对得上，还是没被搬过来，因为档案是按「我猜这个域有什么」写的，不是按「仓库已经知道什么」。
2. **绕过有两种，档案只处理了一种。** 换字符（`ⅴpn` U+2174、`wⅰf|` U+2170）处理了；
   **换说法**（`P掉外衣` / `三角裤消除` / `删除裙子` / `泳装透明度替换为100%`）完全没想到。
   正则只拦得住那 51 行里的 12 行——这是后来加第四层护栏的直接原因。

补进档案以后（`p5_new2_risk_supplement.py` 算 then-vs-now）：书籍文档 1,096 → 1,240 行（+144）、
软件 1,150 → 1,311 行（+161）。新补的每一类，运行自带筛查几乎都是全漏——不是重叠，是盲区。

### 引用护栏加到第四层：按运行自己的分类，不按正则

正则在 `生成露骨性图像编辑` 上只拦住 51 行里的 12 行（换个衣物名词、换个动词就漏）。运行已经把
这 51 行判进了一个带风险标注的类，那个判断比正则可靠，所以直接用类目拦。
**只列性与自伤两类**：全量拦所有风险类要掉 11.4%（软件）/ 23.0%（书籍文档），而引用
`漫蛙`/`vpn`/`请假条模板` 不构成伤害。结果：露骨类 0 行可引，总可引率 93.1% / 91.9%。

### 两个数字层面的自我更正

1. **「先验错了」说过头了。** 交付家族数 21 / 26、交付叶数 30 / 39，**全部落在
   `expected_family_range: [12, 30]` 里**。先验对不上的只是 `chosen_family_k`（6 / 7）那个
   治理前的数，档案这个字段确实是拿去和它比的、系统也确实按设计忽略了它——但「这个垂类只有
   6–7 个家族」是错的。CLAUDE.md 那条「最终结果必须来自交付分区」，我数结构时守住了，
   写结论时没守住。
2. **「助手侧 12.76% 是情绪倾诉」是错读。** 书籍文档 `倾诉并寻求情绪支持` 274 行里主导叶是
   `生成角色扮演对话` 233/274 = 85%，置信度只有 0.551（全域 0.784），样例大量是安全培训作业、
   的/地用法、素材检索。它是低置信度兜底类。**不受影响的是**：自残轻生按词面 18 行确实存在，
   而档案没有为它建任何类目。

**顺带一个规律，值得写进报告**：书籍文档置信度最低的 5 个 L1 类里有 3 个带风险标注
（0.517 / 0.551 / 0.577，全域均值 0.784）。**最需要准确的一批，正是分类器最没把握的一批。**

### 一个 `src/` 里的真缺陷，没有改（按用户「不动源码」的约束）

**openpyxl 把以 `=` 开头的 query 写成公式**，读回来是 NaN。`fast_deliver.py:395` 写的交付工作簿
因此丢了 软件 第 4725 行的原文（`===发货啦~发货啦===…`，正是哨兵标的钓鱼/垃圾载荷那一行）。
七个域逐格比对：**只有这 1 格**，五域与书籍文档都是 0。

`p5_postprocess_run_xlsx.py` 的位置对齐断言**拦住了它**（拒绝生成，而不是把来源绑到坏行上）。
现在按后置操作修：只容忍「工作簿是 NaN 且语料原文以 `=` 开头」这一种情况，从语料 parquet 修回来，
并在「来源说明」页记一行；**其它任何一格文本对不上仍然拒绝生成**。
要根治得改 `fast_deliver.py`（写 xlsx 前给 `=+-@` 开头的串加前导单引号），本次未改。

### 这一批做不了的三项分析，报告里写成「略去」不是「没发现」

- `p5_taxonomy_delta`：需要同样行的纯搜索对照运行，没有 `book-pool` / `soft-pool`；
- `p5_residue`、`p5_vs_unified`：需要 13 类通用框架标在同样的串上，实测覆盖
  **书籍文档 0.5%（63/11,798）、软件 0.1%（14/11,923）**，五域皆 100%。
  `p5_residue.py` 已改成带着覆盖率数字拒绝运行，而不是抛 `KeyError('source')`。

### 交付前最后一处修正：四快照域不该印 ASR 口径

`p5_snapshot_report.py` §8「机械事实」原本无条件印一句「语音快照是 ASR 文本（无标点、约 40 字
截断、抽样方式未知）」。**书籍文档 / 软件 / 教育 / 影视 / 人物 五个域没有语音导出**，这句话在那里
是凭空多出一条对比线。改成按 `"assistant_voice" in srcs` 分支：没有语音的域改印

> 本域**没有语音导出**，所以"输入方式"这条对比线整条不存在——凡本文没有语音的地方，
> 都是没有这条线，不是测了没差别；

金融 / 医疗 两个域文字不变。七份报告全部重出，`p5_snapshot_verify.py` 两批 **未匹配数字 0 ·
问题引文 0**；逐域核对「语音」字样只在 fin/med 出现（软件那 1 处是叶名 `微信语音与声音设置查询`，
是内容不是口径）。741 测试通过，`ruff --select F` clean。

**顺带记一条口径**：`governance.json` 的 `delivered_leaves` / `delivered_families` 在这两个运行里
读出来是 31/6、37/7，**不是交付形状**——前者是治理前叶数、后者是 `chosen_family_k`。交付形状要从
`leaf_labels_final.npy` / `leaf_family_final.npy` 数：书籍文档 **30 叶 / 21 族**，软件 **39 叶 / 26 族**。

## Session 2026-09-14 — 前 N 名覆盖：把「前十合起来占多少」加进两张表

用户看到报告 §2.2 的「各快照前十」只列了十个名字，要求补上**合计行**：这十个类合起来占了
这个快照多少 PV / 多少行。

### 加在哪里

- **表 2.2-L1 意图 / 表 2.2-聚类叶**（每个领域各两张）在 `第10` 下面多了五行：
  `前3合计` / `前5合计` / `前10合计`（另附 `条数/该快照总行数`）/ `前10合计（流量）` / `前10之外`（附「余 K 类」）。
- **表 1-<层级>**（四个层级都有）多了两列：`前10占比%`、`前10流量占比%`——原来的阶梯只到前5。
  L2 子意图与家族两层只在表 1 里有，§2.2 仍只渲染 L1 与叶（用户问的是 intents/leaves）。
- 工作簿多一页 `前N名覆盖`（4 层级 × 每个快照，含 前1/前3/前5/前10 两种单位 + 并列标记）。
- 新文件 `work/<域>/snapshot_classes/topn_coverage.csv`（**未舍入**，复核脚本的数池自动收录）。

### 口径（三条，都写进了表注）

1. **每个快照各排各的前十。** 同一名次在两列里通常不是同一个类，所以这张表只能竖着读。
2. **`前10合计（流量）` 是同样这十个类**（仍按行占比选出）按 `pv_norm` 加权后的占比，
   **不是**「按流量重排以后的前十」。`pv_norm` 按来源各自归一到 10,000，只能在同一快照内读；
   助手头部1k 本身就是按 PV 取的前 1,000 条，它的「流量」是这一千条内部的流量。
   语音没有 PV、按均匀权重，所以语音那一格的「流量−行」恒等于 0.0——实测金融/医疗都是 0.0，
   这是一条自带的自检。
3. **并列：行合计免疫，流量合计不免疫。** 第 10 名的位置上常有多个条数相同的类（实测 120 格里
   有 10 格，**全部在助手侧**——搜索侧一万行从不并列）。并列的类条数相同，所以 `前3/前5/前10合计`
   与 `前10之外` 与挑中哪一个无关；但它们 `pv_norm` 不同，所以 `前10合计（流量）` **会变**。
   现在给出 `前10流量占比%低/高` 与 `前10流量摆动pp`，表 2.2 在那一格后面印 `±x.xxpp`，
   表 1 多一列 `前10流量摆动pp`（0 = 无并列）。摆动最大的是教育 助手随机1k 叶层 **5.25pp**。

### 两个自己造出来又修掉的缺陷（都是**双重舍入**）

1. `topn()` 原本先 round 到 3 位，表 1 与表 2.2 再各自 round 到 2 位，于是**同一个量在同一份
   文档里印成 98.24 和 98.23**（前5 也有 69.43 / 69.44）。改成 topn() 存未舍入值、两张表都从它
   格式化。副作用：`前10 + 前10之外` 现在逐格等于 100.00（之前有 99.99 的格子）。
2. 工作簿 `前N名覆盖` 页 round 到 3 位、`快照画像` 页 round 到 2 位，于是同一个工作簿里
   98.235 挨着 98.24。改成两页都 2 位；未舍入的值留在 `topn_coverage.csv`。

**教训**：一个量只要在两个地方显示，就必须从**同一个未舍入的值**格式化。中间舍入一次，
两处就会在半分点上分家，而且两边各自看都「对」。

### 顺带修掉一处**陈旧数字**

人物报告写「三层过后可引行 20,542 / 21,804」，重跑 `p5_snapshot_classes.py` 后是 **20,541**。
实测三次都是 20,541，护栏本身没有随机性——上一次会话收紧护栏之后**只重渲染了报告、没有重跑
classes**，于是报告读的是旧的 `summary.json`。
**`p5_snapshot_verify.py` 抓不到这一类**：报告与数池读的是同一个陈旧文件，自己跟自己对得上。
改护栏以后必须重跑 classes，不能只重渲染。

### 核对

- 独立复算（不 import `p5_snapshot_classes`，直接从 parquet + `labels_full.csv` 重算）
  **1,200 个数字 · 120 组单调性 · 0 处不一致**，含「表 1 的 `前10占比%` 与表 2.2 的 `前10合计`
  逐位相同」这条跨表恒等式。
- 七份报告 `p5_snapshot_verify.py` **未匹配数字 0 · 问题引文 0**；741 测试通过；ruff clean。
- 逐行 diff：书籍文档重跑**逐字节相同**（可复现）；其余六份的改动**全部落在 §1 与 §2.2 之间**，
  唯一的例外就是上面那处陈旧数字。
- `src/` 与 `tests/` 本次一行未动。

### 这份新数据自己说了什么

- **意图层几乎没有信息量**：前10 在七域 × 全快照都覆盖 81.7%–98.2%，因为 L1 本来就只有 17–25 个类。
- **叶层才是分水岭**：软件 2025搜索 前10 只覆盖 51.4%、金融 54.4%，而书籍文档的助手两层是 91.6% / 92.9%。
- **搜索侧的头部叶普遍「行多流量少」**：同样这十个叶，流量占比比行占比低 —— 金融 −8.8/−9.5/−8.2pp、
  软件 2026搜索 −10.8pp、书籍文档 −6.0/−7.1pp。金融的机制是干净的：`今日金价查询` 只有 93 行
  （0.93%）却占 7.21% 流量，而按行数排第 2 的 `股票股吧查询` 969 行只占 3.98% 流量。
  **按行数排出来的「头部类」会系统性高估它实际拿到的流量。**

### 复核工作流找出的两个真缺陷（我自己的检查看不见它们）

我自己的独立复算 1,200 个数字全对——**因为我复算时是照着代码选前十的**，所以「选错了十个类」
这一类错误对它天然隐形。七个并行复核 agent（每域一个，只给定义、不许 import 被测代码）找出了：

1. **`前10合计（流量）` 汇总的十个类 ≠ 表里显示的十个类。** `topn()` 用 `value_counts()` 的次序，
   报告另有一条 `sort_values(占比%)` 的次序——两条路径在**并列格**上会选到不同的类。
   教育 助手随机1k 叶层实测：表里第 10 显示 `查询高校与职业院校信息`，流量行印的 73.98% 却是
   挑 `录取分数线查询` 才得到的值，**差 5.25pp**，而表注还写着「同样这十个类」。
   修法：`topn()` 按 `(条数降序, key 升序)` 定下唯一次序并输出 `topn_members.csv`，
   **报告的第1..第10 只从这张成员表渲染**，显示与汇总不可能再分家。
2. **表注「合计行不受影响」是错的**（见上面口径 3）。只有行占比的合计免疫，流量合计不免疫。

### 顺带修掉一个**既有**缺陷：第N格的双重舍入

表 2.2 的 `第1..第10` 原本取 `matrix_*.csv` 里已经舍入到 3 位的 `_占比%` 再格式化 2 位，于是
书籍文档 叶 2025搜索 印 `12.46%`，而表 1 的 `首位占比%` 印 `12.45`（真值 1244/9988 = 12.4549%）
——**同一份文档里同一个量两个数**。现在第N格一律由 `条数 ÷ 该快照行数` 直接算。
实测 600 个第N格全部改为精确值；跨表恒等式 180 处全部成立。
**注意 `matrix_*.csv` 本身仍存 3 位**，没有动——叙述里引的是 3 位值（如 `23.198%`），
改存储精度会波及复核数池与全部叙述。§3/§4 的表 *-A 仍按 3 位显示，那是它自己的口径，不与 2 位打架。

### 核对（重做）

- 独立复算 **907 个数字**（含 600 个第N格、7 个带并列摆动的格）· 0 不一致；
  跨表恒等式 **180 处**（表1 的 前10占比%/前10流量占比%/首位占比% ↔ 表2.2 的 前10合计/流量/第1）· 0 不一致。
- 七份报告 `p5_snapshot_verify.py` 未匹配数字 0 · 问题引文 0；741 测试通过；ruff clean；`src/` 未动。

### 第二轮：复核工作流的「批评」通道又找出五处，其中两处是我印出来的假数

前一轮的七个复核 agent 只查算术。三个**批评** agent（标注/统计/一致性三条视角）查的是
「这个数被标成了什么、周围的话还成不成立」，又找出：

1. **`±0.64pp` 这个记号本身是假的。** 点值几乎总落在区间的**一端**（实测 10 格里 9 格），
   所以 `44.42%　±0.64pp` 会被读成 43.78–45.06，而真区间是 43.78–44.42——**上半截是凭空的**，
   而且同一张表三行以上的并列提示写的正是 43.78%–44.42%，**一格与它自己的注解打架**。
   方向还会翻：书籍文档印 `92.79%　±0.50pp`，真区间 92.79–93.29，点在**低**端。
   改成直接印区间：表 2.2 印 `79.23%　[73.98–79.23]`，表 1 印 `79.23（并列 73.98–79.23）`。
   复核加了一条恒等式：**点值必须落在它自己印的区间内**（17 格全过）。
2. **语音快照上的并列提示是胡话。** 医疗 助手语音1k 有并列，但语音 PV 均匀 ⇒ 并列的类
   `pv_norm` 必然相等 ⇒ 摆动恒为 0，于是报告印出「流量合计因此可在 **66.70%–66.70%** 之间摆动」，
   后面还跟一句「但它们的 `pv_norm` 不同」。两句都错。现在按**实际摆动**而不是「有没有并列」
   分支，均匀权重的情形单独措辞。
3. **`traffic_note` 只在表 1 用了，表 2.2 一个字都没有**——而表 2.2 才是那一行被横着读的地方
   （代码注释还写着「两张表共用一句话」）。现在两张表都带。
4. **「助手头部1k 的流量是这一千条内部的流量」是错的，而且错得很多。** 头部 1k 先按 PV 取，
   **再过清洗层**：被清洗掉的行带走了该导出 **软件 98.47% / 书籍文档 88.45% / 影视 78.33% /
   教育 57.81% / 金融 42.32%** 的原始 PV（`work/build_audit.csv: pv_dropped_%`）。
   所以那一格的「流量」是清洗后剩下那一小部分头部流量内部的份额，**主要是清洗规则的函数**。
   影视的叙述早就写过这件事（「所以本叙述全程只用行占比，一次也没有引用流量列」），
   我加流量列时没有把这条带过来。现在按域插入实测数字。
5. **流量口径根本没有精度声明。** 全文每个类占比都配 Wilson 区间、每个 TVD 都配噪声上界，
   只有流量是裸点估计。流量的有效样本量是 `1/Σw²`，**不是行数**：金融 2026搜索 9,999 行、
   有效 **111.5**，单行最高占该快照流量的 **7.39%**。表 1 加一列 `流量有效n`，
   表注明说「本表没有给它区间」。

另外改掉三处措辞：表 2.2 的「占比**一律**由条数 ÷ 行数算出」对流量行不成立；
「后**五**行是合计」——`前10之外` 是余额不是合计；`前10之外` 改名 `前10之外（行）`，
因为它紧贴在流量行下面，原来那一列读起来像是要和 44.42 相加。

**核对（第二轮）**：独立复算 **1,827 个数字**（含 10 个并列区间、点值落区间恒等式）· 0 不一致；
七份报告未匹配数字 0 · 问题引文 0；741 测试通过；ruff clean；`src/` 未动。

**教训**：算术复核与标注复核是两件事。第一轮七个 agent 把 2,256 个数字算了一遍，没有一个
说「这个 ± 记号是假的」——因为 `±0.64` 里的 0.64 **确实**等于 hi−lo，算术是对的，错的是那个
记号承诺的形状。**能被算对的数字，仍然可以是被标错的数字。**

## Session 2026-09-14（下午）— 金融 8 快照：三份新导出，以及一处**上游数据缺陷**

用户新给了三份金融导出：`金融25_random10k.xlsx` / `金融26_random10k.xlsx`（搜索两年的**随机** 1 万）
与 `金融ai_voice_top1k.xlsx`（**按 PV 排序**的语音头部 1 千，自带 wise_pv）。连同原有五份共 8 个快照。

### 为什么这三份重要（不是「多一点数据」）

原来的金融设计把**界面**和**深度**混在一起：搜索只有头部 1 万，助手却有头部和随机两层，所以
「搜索 vs 助手」的每一条发现都可能其实是「头部 vs 长尾」。两份搜索随机 1 万把这个洞补上了，
设计变成 **界面 × 层**，年份再交叉进搜索侧。原始文件实测：头部与随机是真正不同的层
（25头 ∩ 25随机 = 30 串 / 各 1 万，PV 中位 285 vs 1），而且**长尾逐年换血、头部不换**
（头25 ∩ 头26 = 5,447 串；随25 ∩ 随26 = 30）。

### 最重要的发现：**Excel 把纯数字 query 存成了数字，前导零在进入任何代码之前就没了**

`data/raw/金融query-250701.xlsx` 的 `original_query` 列里有 **1,005** 个数字型单元格
（openpyxl `data_type=="n"`），260701 有 **1,057** 个。于是 `000519` 变成 `519`。
三条独立证据，全部在本语料上复算过：

1. **前导零检验**：1,318 条裸 6 位 query 里以 `0` 开头的 = **0 条**；而长 query 内部出现的
   1,467 个 6 位码里 **513 个（35.0%）**以 `0` 开头。若用户真的原样键入，期望约 461 条。
2. **孪生检验**：617 个不同的裸短串里 **223 个（36.1%）**补零后的 6 位形式出现在**同一份语料**的
   长 query 里（`519`↔`000519`、`2015`↔`002015协鑫能科股吧`、`1`↔`000001`）。
   同码段随机码的零假设是 **7.6% ± 1.0%**（200 次重抽）。
3. **竞争假设排除**：港股假设不成立（裸 4 位在港股最密的 0001–0999 段里 0 条）；
   年份假设只影响 51 行，其中 13 个值有语料内孪生，真正像年份又无佐证的只有 10 行。

**这解释了用户问的那个 8.21% 残余类**：它的 40.3% 是被截断的证券代码。而且
「残余置信度 0.671 vs 域均值 0.872」是两个群体的混合——**数字那一半是 0.817，是高置信度地判错了**，
非数字那一半才是 0.572 的「判不出」。

**其它域也中招，但轻得多**：教育 22/35、书籍文档 22/29、影视 8/5、人物 5/4、软件 1/1、**医疗 0**。
金融最重，因为证券代码本来就是纯数字。已交付的七个域**未重建**——修它们要重跑挖掘。

### 用户那个例子的实测结论（与直觉相反，如实记下）

`十大股东占比16.7%为高还是低` 这个形态在 22,934 行里只有 **6 行**——是轶事，不是类目。
残余类里 **88.5% 的行连一个疑问标记都没有**（没有 吗/呢/？/为什么/怎么/哪/什么/如何/多少）。
真正成立的是同族的另一半：「描述一段已发生的行情再问为什么」（阳光电源今天为什么大跌），
助手随机层 2.23% vs 搜索侧 0.02–0.03%，约 75 倍——**但它是边界问题不是残余问题**：
其中 18 行现在落在**风险标注类**「寻求个股买卖与持有决策建议」里，把一次解释请求路由到了投顾合规路径上。

### 这一轮建了什么

- `analysis/pooled5/build_fin8_corpus.py` → `data/raw/pooled5/金融8_pooled5.parquet`（44,000 行 → 入挖掘 43,934）。
  **不覆盖 `金融_pooled5.parquet`**（它与 fin-pool5 的 labels_full.csv 逐行绑定）。
  脚本对五个共有快照逐格断言与已交付语料相同（用 `query_raw` 比，因为 `query` 已修复）。
  补零 839 行，`code_repaired` / `repair_conf` 标出，10 行标为 `low_yearlike`。
- `configs/domains/finance_zh_v2.yaml`：8 条 pragmatic hint、8 个风控类（v1 的 3 类在 43,934 行上
  合计只命中 21 行；新加的 `debt_collection_distress` 单独 57 行，其中 40 行在随机层）、L1 先验放宽到 [18,30]。
- `configs/pool8_fin.yaml`、`analysis/pooled5/run_fin8.sh`。
- `pooled5_common.py`：新增伪领域 `金融8`→`fin-pool8`、cohort `fin8`、三个新 source 与中文名。
  **新 source 追加在各自年份/界面之后，没有动已有五个的相对次序**——七个已交付域逐字节验证不变。
- `p5_snapshot_figs.py`：快照对数 ≥15 时 TVD 图改画下三角矩阵（8 快照是 28 对，柱状图必糊）。
  ≤10 对仍走原路径，金融 10 对的图逐字节相同。

### 运行结果：fin-pool8

`fin-pool8` fast/routed/未 halt，270 次调用，102 分钟；`verify_run.py` **21 PASS / 0 FAIL**
（对照 `ppl-pool5` 照常 FAIL，证明这套检查会失败）。
交付 **L1 20 类 / L2 58 / 叶 62 / 家族 46 / 43,934 行**（fin-pool5 是 18 / 47 / 68 / ? / 22,934）。

**残余类 8.21% → 3.86%**，而且成色完全变了：纯数字从 759 行降到 **7 行**，中位长度从 4 字升到
10 字，剩下的是真正的残余（蚂蚁庄园答题、营销短信、含混的消费金融问题）。
`查询金融资产实时行情` 这个 55% 的巨类也拆开了，新体系里 `查询可报价品种实时行情与价格` 是 35.06%。

**架构师自己把边界切得比我的 hint 更准**：类名叫 `解析裸非可报价名称`——「非可报价」正是
机构名 vs 证券名那条线；我 hint 里的例子把 `铜牛信息`（带 信息 后缀，实际是证券，对照组实测
623 行里 618 行落在行情类、置信度 0.956）和 `人民银行`（机构）混为一谈，是写错了。
运行没有被我带偏。

### 用户那个例子：先判「是轶事」，最后证明是**头部语料的假象**

`十大股东占比16.7%为高还是低` 在旧的 22,934 行里只有 6 行，我据此写了「是轶事不是类目」。
**新语料把这条结论推翻了**：新体系里 `对"是不是/是A还是B"给出裁决结论` 逐快照占比是

| 2025搜索 | 2025随机1w | 2026搜索 | 2026随机1w | 助手头部 | 助手随机 | 语音头部 | 语音1k |
|---|---|---|---|---|---|---|---|
| 0.80% | **10.88%** | 0.27% | **9.00%** | 0.84% | **7.70%** | 0.60% | **8.50%** |

它是长尾里的头部意图，在头部语料里几乎不存在（0.27–0.84%）。旧语料只有头部，所以只看得见 6 行。
**用户的直觉是对的，看不见它的原因恰好就是他补数据的原因。**

同族的 `解释已发生市场现象的原因` 也成立：助手随机 2.74% vs 2026搜索 0.05%，55 倍。

### 这批新数据买到的最硬的一条：**深度 > 界面 > 时间**

一次只变一个因素（L1 层，TVD，全部超出同源噪声上界）：

| 变什么 | 对 | TVD |
|---|---|---|
| 只变年份（都在头部） | 2025搜索 ↔ 2026搜索 | 0.15 |
| 只变年份（都在随机层） | 2025随机 ↔ 2026随机 | 0.12 |
| **只变层（同一天、同一个搜索导出内部）** | 2025搜索 ↔ 2025随机 | **0.57** |
| **只变层（同上，2026）** | 2026搜索 ↔ 2026随机 | **0.60** |
| **只变层（助手内部）** | 助手头部 ↔ 助手随机 | **0.57** |
| 只变界面（都在头部） | 2026搜索 ↔ 助手头部 | 0.29 |
| 只变界面（都在随机层） | 2026随机 ↔ 助手随机 | 0.28 |
| 只变输入方式（都在头部） | 助手头部 ↔ 助手语音头部 | 0.26 |
| 界面+层一起变（**旧设计只能测到这个**） | 2026搜索 ↔ 助手随机 | **0.77** |

**同一天、同一个导出、同一个界面，只换抽样层，距离就有 0.57–0.60；而把层对齐以后，
搜索与助手只差 0.28–0.29。** 旧的五快照设计里搜索只有头部，所以「界面差异」永远和深度绑在一起，
测到的 0.77 是两个因素之和。**已交付的 fin-pool5 报告里每一条「搜索 vs 助手」的结论都带着这个混淆**，
这不是那份报告写错了，是它的语料没法把两者分开。

**顺带测出语音那份「抽样方式未知」的导出是什么**：它到 2026搜索随机1w 的 TVD 只有 **0.11**，
到 助手语音头部1k 是 **0.62**。八个快照里它离一份随机搜索样本最近。
证据支持「它是长尾/随机样本」，但仍然不改名——导出本身没有任何抽样字段。

### 后处理

`P5_COHORT=fin8` 跑完整套：含来源 xlsx/csv、跨快照对比工作簿与报告（**351,334 字符 / 2,022 表格行**）、
六张图；`p5_snapshot_verify.py` **未匹配数字 0 · 问题引文 0**。
28 对快照的 TVD 图走新的下三角矩阵路径（已看图确认可读）。

## Session 2026-09-14（夜）— fin8 的三处交付缺陷；健康两份新导出的准备与清洗

### 先修已交付的 fin8（三处缺陷，都已重建并复核）

1. **界面层（表 *-E、「仅搜索/仅助手」、界面散点图）静默漏掉三个快照。** `p5_snapshot_classes.py`
   把 `SEARCH` / `ASSIST` 写死成旧五个 source，于是两年的搜索随机 1 万和语音头部 1 千没进界面对比：
   搜索n 读成 19,997、助手n 读成 2,937。现在从每行自己的 `surface` 列分界面（`surface_groups`，
   一个快照没有或有两个 surface 值就直接报错）。修后 39,997 / 3,937。
2. **fin8 报告里一句流量警示都没有。** 构建审计只查 `build_audit.csv`（fin8 写的是 `build_audit_fin8.csv`），
   而且那段代码是跨三行的隐式拼接后挂 `if _drop else "。"`——条件表达式优先级低于拼接，找不到行时
   **①②③整段**被换成「。」。现在①②无条件印，③按任意 `build_audit*.csv` 里被清洗掉 ≥5% PV 的快照印。
3. **只对旧批次成立的措辞印在了 fin8 上**：「本域是 2026-09-13 新接入的」「搜索每年 ~10,000 行」
   「两个搜索快照都在 7 月初」「搜索侧那两万行」、附录里的构建脚本与配置名。改为逐域文字
   （`DOMAIN_INTRO` / `DOMAIN_TIME` / `DOMAIN_BUILD`）并按实际快照数与行数渲染。

**回归测试抓到了我自己的一处越界**：第一版把通用的③措辞也用在了七个已交付域上，七份报告各变 6 处。
恢复成原句后七份逐字节相同；医疗的逐类 CSV 逐文件相同。**fin8 仍缺叙述段（7 个 NARR 块全空）**，
新两域是有的——待补。

### 健康：两份导出的形态

- `健康搜索_top1w.xlsx` 与 `健康ai管家_top1w.xlsx`：同一周（2026-09-04 至 09-10），都是
  「query × 天」取前 1 万行，都带平台自己的三级分类。**AI 文件也有跨天重复**（1,075 串在 7 天都出现），
  不只是用户点名的搜索文件，所以两份都按 query 合并。两份都是纯文本单元格，没有金融那种前导零缺陷。
- 搜索 10,000 → 2,260 串；9 串在**同一天**有两行、分类不同、PV 不同——是一次导出把同一 query 的流量
  拆到了两个标签上，PV 相加。19 串分类冲突，保留 PV 占优的标签并存完整拆分。
- 周 PV 只在 7 天都在榜时是精确值，其余是下界；`pv_week_upper` 按每个缺席日自己的截断线补上界。

### 健康：清洗（详见 `analysis/pooled5/work/health_clean_audit.md`）

- **搜索**只去掉 16 个「科室+姓名+医生」串（16 位不同医生周 PV 164k–179k，跨医生 CV 0.026，每天都如此；
  同 PV 段的有机串日 CV 0.422）和 8 个健康新闻标题。**日 CV 低本身不是信号**：69 个 7 天齐全的串 CV<0.03，
  大多是常青的有机查询。私立医院导流（60 串，搜索 PV 6.57%）是用户搜索，进风控类，不清洗。
- **AI 管家**：三个视角盲标全部 1,887 串（Fleiss κ 0.945，95.6% 三票全同）。**多数判为用户键入的只有
  199 串（PV 4.09%）**：作答 chip 1,243（60.85%）、功能/卡片按钮 69（23.55%）、推送问题 245（8.39%）、
  包装模板 118。原有的通用助手清洗器只抓到 13 串（PV 5.90%），头部 40 串漏 37。
- 规则扩展到审计暴露的每个族以后，**单用规则**对多数：精确率 0.993、召回 0.707（PV 0.999 / 0.939）——
  chip 是开放词表，召回有结构性上限。所以 **三票全同时用盲标决定，分票（83 串）时用规则决定**，每行记
  `tier_source` / `rule_tier` / `audit_majority` / `audit_votes`。
- **唯一定不下来的边界是裸词**（头晕 / 糖尿病 / 阿司匹林 / 血常规）：审计保留的 160 个裸词里 49 个原样出现在
  人工键入日志中，其余 111 个都嵌在更长的键入搜索里；被判为 chip 的 1,180 个几乎从不原样出现；两者流量
  形态分不开。**保留为 user 并加 `flag_bare_term_origin_unknown`**（入挖掘的 225 个 AI 串里 184 个）。
- 最终入挖掘 **2,461 行 = 搜索 2,236 + AI 225**。

### 健康：参考列（用户说明：只改运行命令，不改源码）

预注册条件（在最终语料出来之前写进配置）：每列与界面的 Cramér's V ≤ 0.55，且没有占比 >1% 的单边类。
最终：`legacy_l2` V=0.251 → 声明（第一列，负责金标/试点/读日志分层）；`legacy_type` V=0.362 → 声明；
`legacy_dept` V=0.348 但有两个 >1% 的单边类 → **撤下**。命令里显式传 `--reference-columns legacy_l2,legacy_type`
（`analysis/pooled5/run_health.sh`），与配置一致。

### 健康：新建的文件

`analysis/pooled5/build_health_corpus.py` · `configs/domains/health_zh.yaml`（医疗种子原样沿用，实测覆盖 30.8%；
医疗的 9 个风控类在这周几乎不点火，新增 私立医院导流 / 偏方与速效根治 / 未成年+性，年龄写成
`(?:^|[^0-9])`，因为 `47岁` 实测会被当成 `7岁`）· `configs/pool2_health.yaml` · `analysis/pooled5/run_health.sh` ·
`analysis/pooled5/work/health_ai_audit_labels.csv` / `health_clean_audit.md` / `build_audit_health.csv` / `build_tiers_health.csv`。
`pooled5_common.py` 登记伪领域 `健康`→`health-pool2`、cohort `health2`、两个 source 与一个对比对（追加在末尾，
其它域列序不变）。`p5_snapshot_classes.py` 加了**按领域**的第五层不可引护栏（只对 健康 生效：未成年+私密、
露骨性行为描述、具名医生），verify 同步套用；医疗与书籍文档重跑后逐文件、逐字节不变。

### 同一夜的补充：交付脚本逐域化、健康专用分析、以及「有没有回退」的实证

- **交付脚本也写死了五域口径，已逐域化**：`p5_postprocess_run_xlsx.py`（`来源说明`页写「五个取值」、硬取 `l2` 列——健康没有这一列会直接报错）与
  `p5_deliverables.py`（五域说明、硬取被剔除行的 `l2`）。现在 `DOMAIN_SRC_COLS` / `DOMAIN_NOTE` / `DOMAIN_ORDER_EXTRA` /
  `DOMAIN_REMOVED_COLS` 只对 金融8、健康 生效；医疗重跑后 `含来源.csv` 逐字节相同、`labeled.parquet` 内容相同。
  金融8 的两份交付物已按 8 个来源与修复列重建（`金融8_all_rows.parquet` 写于前导零修复之前，所以被剔除行不带 `query_raw`）。
- **健康专用分析，全部按文件/列存在与否开关，其它域不产出**：`product_layer()`（被清洗掉的产品层按来源×层级的串数与周 PV、
  每层头部串——具名医生卡只给条数、命中护栏的不引原文、盲标一致度与规则单独的精确率/召回、合并前后统计）；
  `reference_columns_check()`（重测预注册条件，并**断言与运行实际声明的参考列一致**，不一致直接拒绝）；
  `bare_term_sensitivity()`（去掉 AI 侧来源不明裸词以后，每个类的界面差异是否翻转）。报告新增「## 0 数据准备：合并、清洗与盲标」，
  同样只在这些 CSV 存在时渲染。
- **自己踩的坑**：第五层不可引护栏第一版用 `Series.str.contains`，pyarrow 字符串会把正则交给 RE2，而 RE2 不认 `一-鿿`
  转义，健康一跑就抛 `ArrowInvalid`；只对健康生效所以其它域的回归抓不到，是验证时撞上的。改为 Python `re` 求值。
- **用户问「fin8 是不是有缺陷、会不会回退」，实证如下**：`src/` 与 `tests/` 最后修改 2026-09-12 22:51；fin-pool8 挖掘产物全部是
  运行当时（09-14 17:26–18:37）的，gen01 里唯一更新的文件是 Finder 写的 `.DS_Store`；此前交付的七份逐类对照报告与交付版
  **SHA-256 相同**（mtime 变了是因为回归测试重渲染过）。fin8 报告相对交付版有 25 处变动，全部来自那三处修复。
  **fin8 的缺陷全在后处理交付物里，不在运行里**；我向用户报告过的 fin8 结论（深度>界面>时间、裁决类长尾占比、残余 3.86%、
  20/58/62/46）都来自逐快照的表，不受界面层那处缺陷影响。
- 叙述任务书：`work/snapshot_narrative_brief_fin8.md`、`work/snapshot_narrative_brief_health.md`（后者跑完再补具体数字）。
  fin8 叙述工作流（写作→独立复核→修订）在跑。
- **fin8 叙述已完成**（工作流 wf_4a29d549-b3d，3 个代理）：复核员不采信任何表，把 parquet 与 labels_full 按行位置对齐后重算
  270 个数字、12 条引文，提出 13 条（3 必须改 / 6 应该改 / 4 可选）。三条必须改都是**结论被区间推翻**：两层对齐后方向一致的类是
  8 个不是 7 个（漏了寻找绕征信借款）；「差在层，不在界面」被只换界面也显著推翻；「集中在一个子意图」被 signature_td_l2 推翻
  （INVEST_ADVICE__1 同样是特征类）。修订员先自己复算再改，第 9 条连复核员的机制解释也说过头了，没照写。
  **我自己又独立数了一遍**：newcombe_td_l1.csv 两层都显著且同向 = 8 个，与修订一致；22 个定义 L1、20 个交付、2 个 0 行一致。
  重渲染 + verify：「未匹配数字 0 · 问题引文 0」，叙述 10,159 字符，报告 364,177 字符。任务书自己也写错一处
  （流量有效n 最低的是 2026搜索随机1w 37.2，不是助手头部1k）——叙述照 CSV 写，是对的。fin8 四件交付物已重发用户。
- **回归基线**：健康后处理开跑前，对 8 个 run 的跨快照报告 / 工作簿 / 含来源 csv 与 `deliverables/` 下 csv/parquet 共 32 个文件
  记了 SHA-256（scratchpad `legacy_sha_before_health.txt`），健康后处理结束后对比。
- **health-pool2 跑完**（2026-09-15 00:54 退出 0，约 60 分钟）：`run_summary.json` mode=fast、provider=routed、halted=False；
  架构师第 0 次返回不可解析 JSON，程序自己重问一次成功（无人工干预）。`tools/verify_run.py runs/health-pool2/gen01 runs/ppl-pool5/gen01`：
  新运行 **20 PASS / 0 FAIL / 6 N/A / 2 SKIP**，对照组 3 FAIL。交付形状：定义 21 个 L1、交付 20 个（RED_FLAG_TRIAGE 0 行）、
  L2 23、叶 16、**家族 16 且每族恰好 1 叶**——治理前 13 叶 / 4 族，p8 执行 3 次拆叶 + 8 次风险隔离、拒 3 条，隔离把几乎每个叶
  挪进各自的风险家族。K 由 `legacy_l2` 定位到 4（治理前）。risk_screen 167 行；医疗档案原九类里四类 0 命中。
- **健康后处理**（`analysis/pooled5/post_health.sh`，六步，exit 0）：报告 93,653 字符 / 985 表格行，机械复核 0 / 0（叙述前）。
  最硬的一条：L1 界面差异显著 12 类，**去掉 AI 侧 184 行来源不明的裸词后只剩 6 类、翻转 12 类**；AI 侧首位意图
  「由自身症状反查原因」67.11% 里 94% 是裸词。仍成立且方向不变的只有功效类、定位机构（两者 AI 0 行）与反查原因。
- **后处理代码里又抓到两处写死的旧措辞**（都在 `p5_snapshot_figs.py`，都是看图看出来的）：界面散点轴名写死「搜索（2025+2026 合并）/
  助手（各层合并）」→ `IFACE_AXES` 按域覆盖；快照距离图标题写死「25搜/26搜 = 两个搜索快照…」→ 只在刻度真的用了缩写时才印。
  按每个域的 `pairwise_tvd.csv` 实测：七个旧域的刻度全是缩写（标题逐字不变），金融8 走矩阵图（不经过这段），只有健康变。
- **运行自己的两份定义文档会印样例原文**（架构师判例、模板示例、质心/边缘样本），其中有露骨性描述与成人内容标题。
  `analysis/pooled5/redact_health_run_docs.py` 在 `postprocessed/` 生成 `*_引文护栏版.md`，只替换查询样例（反引号 / 「」/ ≤40 字的表格单元），
  护栏 = 硬规则 + 共现规则 + 健康第五层 + 具名医生 + **只取与未成年人/性内容有关的**运行点名串。第一版用了报告的全部点名串，
  自上而下文档被换掉 56 处，按层拆开看大半是偏方、导流、药物相互作用、自我诊断的样本——正常健康查询，所以收窄。原件留在 gen01。收窄后：自上而下替换 14 处、自下而上 2 处、分层对比 0 处；护栏版里剩下的
  模式命中只在护栏版自己的说明行与治理理由的描述性文字里（不是查询样例），脚本对「查询样例仍命中」做了断言。另：描述性文字里若在护栏话题的同一行带「（…14岁…）」这类
  含年龄的具体描述括注，替换成「（具体描述略）」（自下而上治理理由 1 处），也有断言。
- **回归**：健康后处理前记下的 32 个文件（8 个 run 的跨快照报告 / 工作簿 / 含来源 csv + deliverables 下 csv/parquet）SHA-256 全部相同；
  `ruff --select F src/qmine/ tools/ analysis/pooled5/` 通过。`src/`、`tests/` 本会话未改。
  全量 `pytest tests/ -q`：29 个文件、741 个测试，exit 0（健康后处理与两处作图修正之后跑的）。
- **健康叙述已完成**（工作流 wf_12ecd73b-df3）：复核员逐行对齐重算 195 个数字、40 条引文，隐私与引文全部通过（P007 隔离的叶 9 没有一条引文），
  提出 16 条（3 必须改 / 8 应该改 / 5 可选）。三条必须改：导读对 4 个翻转类只给了全部行一套数；路线交叉对翻转类「解读医学名词」只给一套数；
  边界把被清洗掉的产品推送层当成「周内热点推高两边」的直接证据（入挖掘两侧行里开学/新冠/暴雨等题材词 0 行）。修订员 11 条全改。
  我自己重渲染 + verify：「未匹配数字 0 · 问题引文 0」，报告 105,901 字符；隐私扫描：报告里唯一的模式命中是「未成年人与性相关」这个类名 + 条数。
  健康报告已发用户。回归：32 个基线文件 SHA-256 再次全部相同；`src/`、`tests/` 未改。
- **程序缺陷（未改源码，已记录、已开独立任务）**：`src/qmine/ops/cluster.py: reference_sensitivity()` 写死 `"decides": key == "intent_alignment_ami"`，
  注记写「决定权在 `phrasing_groups`」；`graph/nodes/bottomup.py` 的 `p5_k_references_agree` 闸门 observed 写死 `deciding_reference="phrasing_groups"`。
  当声明的参考列定位 K 时（health-pool2：`triangulation.locator=ami_vs_legacy_l2`）产物自相矛盾，自下而上定义文档的闸门行因此写错。
  同一文件里决策记录早先修过同一类错（live41 注释），这两处漏了。只影响披露文字，不改任何 K 与数字。live41/42/44、ai04、aiwire01、三个 k12_zh 的
  granularity.json 也带这条注记（其中 locator 本来就是措辞组的那些是对的，要逐个看）。**本次处理**：交付给用户的自下而上护栏版在那一行加注
  「程序缺陷：本次实际定位 K 的参照系是 legacy_l2」（`redact_health_run_docs.py`，带断言）；健康叙述里也点明了。源码修复需要改两处 + 回归测试，
  按用户要求单独审慎处理，未在本次动。发现者是健康叙述的写作代理——我起初以为它读错了，逐字查文件后确认它是对的。
- **我自己审计笔记里的一个错数**（复核员发现）：`health_clean_audit.md` 与 `build_health_corpus.py` 说明里写搜索侧 7 天都在榜 890 个串，
  实为 893。原始导出逐行重算：890 是「恰好 7 条原始行」的串数；同一天被拆成两行的 5 个全周串有 8 行，另有 2 个 7 行串不满 7 天。
  第一次我猜「890 = 满 7 天且恰 7 行」，断言没过就没改（那个数是 888）；按原始导出确认原因后才改。报告表 0-A 一直是 893。
  另：交付的自下而上护栏版多了一处缺陷注记，已与改正后的审计笔记一起重发用户。

## 4. Session (2026-09-15) — the reference credited with deciding K

Resolves the bullet 「程序缺陷（未改源码，已记录、已开独立任务）」 in the 2026-09-14 session above,
which recorded this defect during the health-pool2 work and deferred the source fix.

### The defect (disclosure only)

- `ops/cluster.py: reference_sensitivity()` set `"decides": key == "intent_alignment_ami"` for every
  reference and its note ended 「决定权在 `phrasing_groups`」 — true while the phrasing groups were the
  only locator, silently wrong once `choose_locator` could hand K to a declared column.
- `graph/nodes/bottomup.py`, gate `p5_k_references_agree`: `observed.deciding_reference` was the literal
  `"phrasing_groups"`, and the bottom-up reports print that gate row.
- A second wrong output of the same hardcode: with declared columns and **no** phrasing groups, every
  `decides` was false, so the only reference "decided nothing".
- Stored damage, measured read-only over all 69 `runs/*/gen*/granularity.json`: 10 generations — ai04,
  aiwire01 (`ami_vs_l2` / `ami_vs_l1`), health-pool2, the three k12_zh-20260901-*, live41 gen01+gen03,
  live42, live44 (`ami_vs_legacy_*`). The other 24 of the 34 that carry `reference_sensitivity` were
  phrasing-located and correct. None reached the stability fallback.
- The same bug class was fixed twice before (decision record `decisive_metrics`, locator profile);
  these two sites were missed both times.

### The change

- `reference_sensitivity(sweep, chosen_k, *, locator_column: str | None)` — **required keyword-only, no
  default**, because a default is exactly how this defect recurred. `decides = key == locator_column`;
  `None` marks nothing. The value is the sweep key (`intent_alignment_ami` / `ami_vs_<col>`), the same
  namespace as `triangulate_k`'s `locator_key`. **No keys added**, so `granularity.json` stays
  byte-identical for phrasing-located runs and their p5 observer prompts still replay from `llm_cache`.
- `triangulate_k` passes `locator_column=locator_key if located else None` (in the fallback `locator` is
  a free-text stability string and nothing located K).
- The gate records `tri.get("deciding_reference") if tri.get("locator") == locator_key else None`. The
  guard matters: in the fallback `tri["deciding_reference"]` still names `choose_locator`'s pick (§2 #21).
- `tests/test_measurement_soundness.py`: `test_k_is_reported_under_every_available_reference` now passes
  `locator_column="intent_alignment_ami"` explicitly (assertions unchanged). New:
  `test_the_reference_marked_as_deciding_is_the_one_that_located_k` (pure; phrasing, declared column,
  declared-only single and pair, fallback with an unscored column and with a column scored only at a K
  stability rejected), `test_the_disagreement_gate_names_the_reference_that_actually_located_k` and
  `test_the_disagreement_gate_names_no_reference_when_stability_decided` (real `p5_granularity` on a
  4x3-blob synthetic corpus via the `deps` fixture, ~0.1 s each, every precondition asserted).
- Invariant rows went to `.claude/rules/measurement.md`, not `CLAUDE.md` (already 227 lines against its
  200-line target). `analysis/pooled5/redact_health_run_docs.py`'s comment updated: its annotation still
  applies to health-pool2 gen01 and is a no-op on a fixed run.

### How it was checked

- **Before:** full suite passed (exit 0), ruff clean, both files backed up; HEAD equals the backups.
- **Discovery workflow** (three read-only lenses): every consumer (renderers, audit, findings recheck,
  render/resume/verify_run replay), a behavioural harness, and a branch-by-branch table.
- **A trap worth not re-learning:** `tests/conftest.py:17` inserts the working-tree `src` at
  `sys.path[0]`, so a mutation check run as `PYTHONPATH=<copy>/src pytest tests/...` silently tests the
  FIXED code — the first mutation check "passed" on the original for exactly that reason. It was redone
  as full mirror projects (src + tests + pyproject, configs/data symlinked) with a probe test asserting
  `qmine.ops.cluster.__file__` lies inside the mirror.
- **Mutation matrix on the final tests:** fixed passes all four; the original fails all four;
  M1 gate hardcoded → both gate tests; M2 `decides` keyed on phrasing → pure + located-gate;
  M3 call site passes `locator_key` unconditionally → pure; M4 gate unguarded → fallback-gate test;
  M5 name compared with column → existing + pure + located-gate.
- **Review workflow** (four lenses, three skeptics per finding; 12 upheld, 9 dropped). Applied: docstring
  column names and counts; the M3 and M4 pins (both mutants had survived the first version of the
  tests). Recorded: §2 #20, #21. Measured by the review: `reference_sensitivity` recomputed with old and
  new code on all 34 stored sweeps — 24 identical, 10 differ in exactly three leaves (two `decides`
  flags and the note's last clause); the whole `triangulate_k` output identical for 59 of 69 and
  `chosen_family_k` identical for all; `None` renders as 无 / null everywhere; the behavioural test is
  deterministic across seeds and thread counts.
- **After:** full suite **744 passed, exit 0** on the exact final tree (741 + 3 new);
  `ruff --select F src/qmine/ tools/` clean. The 3 F findings in `tests/test_measurement_soundness.py`
  are identical at HEAD. Pre-existing working-tree diffs in `graph/nodes/naming.py` and
  `tools/postprocess_assistant_run.py` were not touched.

### Deliberately not done

- No stored artifact rewritten. `qmine render` re-projects artifacts and cannot repair the 10
  generations; a new generation that re-runs p5 gets the fix (and, for those 10 only, a p5 observer
  cache miss). They remain usable as the known-broken control for a future `verify_run` check (§2 #20).
- Sibling strings that still credit the phrasing groups in delivered documents (§2 #20) and the
  fallback-only attribution (§2 #21) — each needs report-generator or fallback-semantics changes that
  deserve their own review.

## Session 2026-09-15（下午）— 医疗 8 快照：三份新导出、语料构建、v2 档案（进行中）

用户补来三份医疗导出（`25医疗搜索随机1w.xlsx`、`26医疗搜索随机1w.xlsx`、`医疗ai_voice_top1k.xlsx`），要求与 fin-pool8
同样的一整套：补进已交付的 `med-pool5` 这一侧（**不是** health-pool2），fast 模式重跑 8 快照，并做同样的后处理、分析与报告。

### 数据实测（原始文件，不含 query 原文）

- 六份医疗导出的 `original_query` **全部是文本单元格**（数字型 0 个），所以金融8 的前导零修复不需要、也没有跑。
- 头部与随机是真正不同的层：25头 ∩ 25随机 = 26 串，26头 ∩ 26随机 = 4；PV 中位数 604 / 303 vs 1。
  头部逐年延续（25头 ∩ 26头 = 6,329），长尾几乎完全换血（25随机 ∩ 26随机 = 2）。
- **`医疗ai_voice_top1k.xlsx` 实为 10,000 行**（金融的同名导出是 1,000 行），按 `wise_pv` 降序。第 1,000 行 PV=11，
  PV=11 的并列行有 152 行（第 916–1,067 行）。**决定：取 PV ≥ 11 的全部 1,067 行**——在并列处按导出顺序截断会凭顺序定成员；
  其余 8,933 行不入挖掘。构建脚本断言排序与截断。列头仍叫「助手语音头部1k」（`SRC_ZH` 是跨域共用的，改名会动金融8 的输出），
  报告与交付说明里写明本域是 1,067 行。
- 模板种子（v1 在头部挖出来的）ANY 覆盖：头部 54.5% / 60.1%，随机层 21.7% / 19.4%，语音头部 20.1%，语音1k 23.9%。
  疑问标记占比：头部 43.6%，随机层 64.6–65.4%；语音头部中位长度 4 字、疑问标记 9.3%。
- 风控词形按层：未成年人与性相关 头部 0 / 随机 7；私立医院导流 头部 8 / 随机 35；自行用药、药物相互作用、特殊人群用药、
  个人信息都是随机层更多；「偏方与根治」头部 476 / 随机 63——头部偏重，精度待审（v2 研究工作流在查）。
- med-pool5 的残余类 `OUT_OF_DOMAIN_OR_UNCLASSIFIABLE` 1,405 行（6.12%，搜索 4.4–4.6%，助手 17.8–19.5%，置信度 0.589 vs 0.811），
  自下而上叶却是明明白白的医疗叶（药品功效 166、药品产品 151、正常值 124、治疗手术 124、孕产流产 111）——
  和金融 v1 一样，是边界问题不是「非医疗」。私下读过的样本里有药企名、缺指代的助手追问、营养/卡路里计算、ASR 碎片。
  **其中一条露骨性描述没有被报告的任何一层引用护栏拦住**——医疗8 需要自己的、实测过的第五层（后处理时加）。

### 建了什么

- `analysis/pooled5/build_med8_corpus.py` → `data/raw/pooled5/医疗8_pooled5.parquet`（44,067 行 → 入挖掘 44,019），
  `work/医疗8_all_rows.parquet`、`work/build_audit_med8.csv`、`work/build_facts_med8.json`、`work/med8_build_audit.md`。
  **不覆盖 `医疗_pooled5.parquet`**。五个共有快照在 query/source/surface/l2/tier/pv_raw/pv_norm 七列逐格断言相同。
- **构建时自己的断言先报错，查明是表示差异**：旧 parquet 把缺失 `l2` 存成字符串类型的 NaN，新构建是 pd.NA，
  `astype(str)` 比较把 9,998 行全判成不同。逐列实测「两边都缺失 / 一边缺失 / 取值不同」= 9,998 / 0 / 0（五个快照七列全部如此）
  以后，才把断言改成「两边都缺失即相等，一边缺失或取值不同仍失败」。金融8 的断言根本没比 `l2`，所以没撞上。
- `configs/pool8_med.yaml`（照 `pool8_fin.yaml`；无参考列：`l2` 只覆盖助手 1,956 行 = 4.4%）、`analysis/pooled5/run_med8.sh`
  （`--domain medical_zh_v2 --run-id med-pool8 --fast`）。
- 注册：`pooled5_common.py` 新增 `医疗8`→`med8`、cohort `med8`、`RUN_ID` `med-pool8`；`p5_snapshot_report.py` 的
  `DOMAIN_INTRO/TIME/BUILD`、`p5_postprocess_run_xlsx.py` 的 `DOMAIN_NOTE`、`p5_deliverables.py` 的 `domain_note` 各加一条 `医疗8`。
  全是按域追加，其它域不经过这些分支。后处理前记下 782 个交付文件的 SHA-256（scratchpad `med8_baseline_sha.txt`）。

### 进行中 / 待办

- `medical_zh_v2.yaml`：五个视角的研究工作流（残余与边界、长尾风险精度审计、种子纯度、先验与提示、法规原文核验），
  每条提议两名怀疑者独立复算，再由整合评审查冲突。**v1 不动**（med-pool5 的 resolved config 要能复现）。
- 之后：启动 `med-pool8`；运行期间准备第五层引用护栏与叙述任务书；跑完 `verify_run`（对照组）、后处理六步、
  隐私扫描（含运行自己的两份定义文档）、叙述工作流、回归哈希、交付。

### 档案 `configs/domains/medical_zh_v2.yaml`（2026-09-15）

- **研究**（工作流 wf_6a245f6b-818，84 个代理，0 错误）：五个视角提出 39 条改动，每条两名怀疑者独立复算，34 条两人都未推翻，
  再由整合评审合并冲突（五对风险类合并、六条边界提醒改放 domain_notes、四处种子互相打架的修正）。整合评审从代码里查实了几件决定
  取舍的事：`pragmatic_intents_hint` 只进架构师、标题是「本体系必须承载的意图」，所以「不要立类」的边界提醒放不进去；类别的
  `rationale` / `policy` 没有任何决策代码读；`expected_min_share`、`expected_l1_range` 没人读（架构师的界是运行配置里的
  `taxonomy.l1_target_range` 15–25）；种子不过凝聚度闸门，但 `build_groups` 对种子也施加 0.004×行数（176 行）的静默下限；
  **`screen_risk` 的样本会原文进 risk_compliance 研究员的提示词和 fast 工作簿的风险清单——不可引类必须在后处理里处理。**
- **组装工作流中途叫停**（用户指出档案已耗时过久、ROI 不划算）：没有产出任何草稿，改为我直接从整合评审的**测量脚本里的最终
  模式对象**生成 YAML（不手抄 20 个片段），`domain_notes` 自己写（4,105 字，含逐层实测形态、v1 的中西医与风险两段原文、
  边界决定 (a)–(f)、法规锚点 (A)–(F)；麻精目录引用 NMPA 公告2025年第55号，删掉无人核实的右美沙芬条款）。
- **校验（程序自己的函数）**：`_load_domain("medical_zh_v2")` 通过；`screen_risk` 13 类逐类命中与预期完全一致，并集 2,870 行
  （v1 1,977）；12 个种子逐源命中与计划一致，并集 41.59%（v1 37.5%），随机层 29.6 / 28.4%（v1 21.7 / 19.4%）；
  v1 的六条提示逐字保留；`qmine doctor` 列出 `medical_zh_v2` ok。`medical_zh.yaml`（v1）未改。
- 与整合计划的唯一差别：`accidental_ingestion_and_overdose` 用计划写的 36 行版本（加了「吃/喝了十几片」那一式），
  整合脚本 `FINAL` 里是 34 行版本。
- **K 网格没有改**：三次运行（med-pool5 / fin-pool8 / health-pool2）的网格提议者都自己加了 8 以下的 K（[4,7] / [6] / [4,5,6,7]），
  为了与 fin-pool8 可比，本次沿用默认配置。

### 运行 `med-pool8` 已启动；第五层引用护栏

- `analysis/pooled5/run_med8.sh`：`--config configs/pool8_med.yaml --domain medical_zh_v2 --run-id med-pool8 --fast`。
- `p5_snapshot_classes.py` 新增 `EXTRA_QUOTE_BLOCK["医疗8"]`：健康那一条原样 + 露骨性行为词（44,019 行里 57 行原本三层护栏
  都没拦住）+ 具名医生两种写法（排除泛称，21 行原本可引，几乎都在随机层）+ 伴侣与性行为描述共现（构建前抽查里穿过所有护栏的写法）。
  只有键为 `医疗8` 时生效，其它域不经过。

## Session 2026-09-15（傍晚）— fin8 深挖：例子的代表性、同一意图的跨快照差异、叶 × 语用子功能

用户问：主报告每个意图、每个快照的真实例子怎么选的、能不能代表；同一意图在不同快照之间差在哪、差多真
（例：助手侧「消息公告」例子更像咨询）；能不能用自下而上的叶把宽 L1（例：「是不是 / 是A还是B」裁决）拆开看是核实还是决策。
中途追问：用叶会不会太碎、改用家族是否更好。

### 交付（纯后处理，`src/` 与 `tests/` 未动）

- `runs/fin-pool8/gen01/postprocessed/finance_zh_v2_意图内部结构与代表性样例.zh.md`（约 6.5 万字）+ 同名 `.xlsx` + `img/意图结构_*.png`（8 张）。
  不改主报告任何数字。
- 脚本（`analysis/pooled5/`）：`p5_intent_structure.py`（三种取例法同一组指标、叶/L2 构成、说法词表、Fightin' Words、C2ST、盲样本）、
  `p5_leaf_namefit.py`（意图 × 叶的名实 + 叶与家族的粒度对比）、`p5_facet_aggregate.py`（三视角聚合）、
  `p5_facet_crossmodel.py`（DeepSeek 跨模型盲标）、`p5_intent_structure_report.py`（渲染；叙述在 `work/金融8/intent_structure/narrative.md`）。
- 盲标工作流 `wf_7aebad7e-0ba`（33 个代理，0 错误）的 journal 拷在 `work/金融8/intent_structure/facets/`，结果由 journal 重建。

### 测到的

- **例子**：主报告卡片 = 每格可引行 `pv_norm` 前 3。n≥10 的 120 格：覆盖中位 32.8%；27 格的第 3 条与 3 行以上并列（随机层与语音 PV 几乎全是 1），
  排序实为文件顺序；自助法 Jaccard 0.44（助手语音1k 0.06）。Hamilton k=3（同样 3 条）到 56.4%、0.82；构成显式中位 80.5%、平均 6.3 条。
- **叶名是整叶的多数**：意图 × 叶 361 格，cos差（意图内中心 vs 叶其余中心，减同大小随机切分基线）中位 −0.093、p10 −0.188。
  例：「计算存贷款利息」叶里属于消息公告意图的行是个股利好消息（−0.327）。报告里每个案例都给叶名 + 意图内中心例子 + 叶其余中心例子。
- **叶 vs 家族（用户追问）**：62 叶 / 46 族，7 个多叶族装 42.1% 的行；意图 × 快照格（n≥30）有效组数 12.4 → 10.8，覆盖 80% 9 → 8，
  组内到中心余弦 0.660 → 0.634（更松）；有族把「查询电话号码归属」与代码类叶、「蚂蚁庄园今日答案查询」与「查询市场行情」并在一起。
  分散来自意图本身横跨话题，不是叶切得太细，所以以叶为单位。
- **盲标子功能**（每意图 9 码；样本 542 / 582 / 682 行，按快照分层）：三 Claude 视角 κ 0.994 / 0.980 / 0.988 是**同模型自洽**
  （查过 transcript：各自手工逐行给码，没有互读文件）；**DeepSeek 用同一码本独立盲标子样本 κ 0.986 / 0.963 / 0.980（n 175 / 164 / 182）**。
- **NEWS_EVENT**：界面随机层（2026搜索随机1w ↔ 助手随机1k）分析型（预测 + 评估 + 问原因）16.7% → 45.6%（+28.9pp [+16.4, +41.0]），
  预测 9.3% → 27.8%；叶构成 TVD 0.265 < 噪声 0.297。助手侧仍有 50.6% 是查找型。主报告助手随机1k 的 3 条卡片里 2 条预测型：方向对，程度放大。
  搜索内头部 → 长尾：浏览最新消息 −25.6、进度 −18.7、事件数据 +36.1；两年随机层之间没有显著子功能差。
- **VERDICT_QUESTION**：按权重 68.7% 核实规则/资格/后果（业务规则 41.7、资格承保 12.9、补救 8.3、后果 5.8），16.2% 类别辨析与事实状态，
  13.8% 辅助决策（安全质量评估 10.0、直接建议 3.8）。搜索头部辅助决策 27.5–29.6%，随机层规则类 68–75%。界面随机层规则类 −20.6pp，
  事实状态/时事走势 +15.1pp [+6.3, +25.7]（唯一显著单项）。叶：「贷款提前还款查询」规则类 98.6%，「车险购买与理赔咨询」直接建议 19.8%。
- **INVEST_ADVICE**：名单 32.7（无评价标准清单 20.4 + 荐股择优 12.2）、判断/预测 30.8、信息/方法 22.7、要一个决定 12.0。
  头部界面：清单 51.9 → 6.5（−45.4pp），预测 16.7 → 53.2（+36.6pp）。「通用规律/投资方法」2025 → 2026 头部 19.3 → 3.7、随机 17.3 → 8.0，两层同向显著。
  预测集中在价格类叶（「查询价格走势图」92.1%，「查询金价」90.2%）；L2 `__3` 88.2% 清单、`__4` 79.2% 预测。

### 自己犯的错与被断言拦下的

- `hamilton` 初版逐轮取整，把名额全给首位叶；改最大余数法后单测 `{'a':2,'b':1,'c':1,'d':1,'e':0,'f':0}`。
- C2ST 30 次置换时出现「标显著但 p=0.065」，改 100 次、以 p<0.05 为准。
- 叶名适用度表的中心例子一度选中 150 字的荐股指令（堆常见词的长串与均值向量余弦高）；改为只在该部分长度 p10–p90 内取 medoid。
- 叙述里两条示例的子功能是我推断的、不在盲标样本里，逐条对照后换成三票一致、可引、离该码中心最近的行。
- 盲标工作流里一个代理把 `INV0660` 写了两遍、漏了 `INV0659`：覆盖断言拒收。其余 199 条按盲表顺序排列、错位那份恰在 `INV0659` 的位置，
  但它的码（DECIDE）与该行内容不符，所以**不补码**，记为声明缺票（上限 1%），多数用其余两票（都是 SCREEN），κ 只在完整行上算；写进 `summary.json` 的 `repairs`。
- 图：Set2 只有 8 色，第 9 个码与第 1 个撞色；改 tab10，无法判定一律灰色放最后。码本代理用英文写了两份定义，进了中文交付；
  译文放 `facets/codebook_zh.json`，拉丁字母占比高的定义没有译文就拒绝渲染。
- 家族段落初稿说「39 个家族只含一个叶，家族与叶几乎是同一个划分」：按个数对，按行数误导（多叶族装 42.1% 的行），已改为实测对比。

### 核验

- 叙述里 124 个带小数的数全部能在报告表格或 `work/金融8/intent_structure/` 的 CSV/JSON 里逐字找到；119 个「」片段中没有任何一条是不可引的语料行；
  全报告 419 个引号片段无催收 / 征信修复 / 套现类词。`ruff --select F` 覆盖新脚本。

### 环境陷阱

- 本机 `HTTP(S)_PROXY` 下，`urllib` 请求 DeepSeek 0.0 秒被断开（关沙箱仍一样），`httpx` 正常 200。跨模型脚本用 `httpx`。

### 没做

- 主报告卡片仍是流量前 3（见 §2 新条目）。盲标只做了三个宽意图；其余意图的「要什么」只有说法词表。

## Session 2026-09-15（晚）— 医疗8：运行、后处理、叙述，以及引文护栏的第六层（逐串阅读）

### 运行 `med-pool8`
- `analysis/pooled5/run_med8.sh`（`--domain medical_zh_v2 --fast`），15:00 启动、16:53 结束，exit 0；`run_summary.json`：mode=fast、provider=routed、
  halted=False、229 次调用、18 个阶段、失败闸门 0。最长的是 p2a 分类体系 2,490 秒与 p8 治理 1,694 秒。
- `verify_run`（对照 `ppl-pool5/gen01`）：本运行 PASS 21 / N/A 6 / SKIP 1 / FAIL 0；对照 PASS 9 / FAIL 3 / SKIP 10 / N/A 6。
- 形状：L1 定义 18、交付 17（`SELF_HARM_CRISIS` 0 行）；L2 55；治理前 49 叶 / 8 族，交付 41 叶 / 37 族；留出复现 0.9704（n=8,804）；
  K 由意图对齐定位，决定性参考 `phrasing_groups`（没有声明参考列）。治理 32 条（执行 23、拒绝 9）：P013 / P014 是**整叶移出家族**（叶里各混入
  一条自伤求法 / 伪装成医学评测的成人内容），不是拆出敏感行；风控哨兵 10 条发现。风控屏 13 类并集 2,870 行（6.52%）。

### 后处理（按域 `医疗8`，其它域不经过；每一处都复算过老域可引行数逐字相同）
- `post_med8.sh` 六步（脚本没有执行权限，exit 126，改用 `bash` 调用）。
- `NEVER_QUOTE_CLASSES["医疗8"]`：自伤危机类（0 行）与误食过量类（6 行）。
- `EXTRA_QUOTE_BLOCK["医疗8"]` 加三式：色情片俗称、治理理由点名的成人标题、「让自己患病 / 弄伤自己」式自伤求法（44,019 行上新增 3 行）。
- `redact_med8_run_docs.py`（护栏版定义文档）：加自伤正则；治理理由里的“”引号片段与长单元格（原件的治理表把自伤求法与成人标题印在里面）；
  名单串出现在行内任何位置都替换；自伤方法的散文转述只替换那几个字，类定义原文保留。
- `p5_snapshot_report._search_share`：「语料 X% 是搜索行」按域计算（原来写死 87%，见 §2）。

### 叙述（工作流 wf_75d0cc9a-b43：写作 → 数字与断言 / 隐私与引文两名复核员 × 3 轮 → 修订；10 个代理）
- 结论要点（数字都在 `work/医疗8/snapshot_classes/` 的表里）：医疗也是**深度 > 界面 > 时间**——L1 只变层 0.38 / 0.40（搜索）、0.33（助手），
  只变界面头部 0.207、随机 0.102（低于噪声上界）；只变年份 0.042 / 0.032。许可式问法（能否吃用做）是长尾现象：2026 头部 1.48% → 随机 13.81%。
  语音1k 的分布像随机层（证据强）；语音头部在叶层自成一格。自伤危机类 0 行配上界 0.0084%，风控词表命中 13 行被判进 7 个别的意图。
- 数字复核员三轮都没有算错的数，问题都在措辞与证据强度；隐私复核员三轮都放行叙述，但**拦下报告表格**（下一节）。第 3 轮修订后没有再复核，
  由 `p5_snapshot_verify`（0/0）与本节的最终闸兜底。叙述里一条引文在例子重排后不再是示例行，换成同一格的「还有导尿管」。

### 引文护栏的漏洞，与第六层
- 隐私复核员逐串读了报告里 1,037 个引文串，标出约 45 行 / 38 串：未成年人与性、露骨与恋物题材、具名医生、民营医院全名、试管选性别，
  **全部穿过了前五层**，大多来自两个语音快照。另外，跨快照工作簿印的例子比 Markdown 报告多约 1,000 串，没有人读过。
- 召回实测（以这 38 串为参照）：DeepSeek 15/38；单名 Claude 读者 22/30。所以 Claude 是必需的读者，DeepSeek 只作并集。
- 第六层 `SCREENED_QUOTE_BLOCK`（`p5_snapshot_classes`，名单是数据文件 `work/医疗8/privacy_screen/quote_block.json`；`p5_intent_structure.quotable_mask`
  与护栏版脚本同样读它；文件不存在时是空集合）。标准写在 `privacy_screen/criteria.md`（7 个码）。
- 轮次（`p5_privacy_screen_round.py`）：r1（被中止的全量子集筛查已完成的 8 块，6,080 串，标 571）→ r2（1,322 串，Claude 49 / DeepSeek 23）
  → r3（182 串，7 / 3）→ r4（15 串，0）→ 独立第二遍（当时印出的全部 2,081 串，5 名读者，12 标，其中 5 串是新的）→ r5（11 串，2）→ r6（4 串，0）。
  最终名单 634 串；Claude 读过 7,314 串；`final` 闸：报告、叙述、三份护栏版、工作簿里名单串 0 处，印出的 2,080 个语料串全部读过，叙述 12 条引文全部有效。
- 代价：7 个 L1 格与 3 个叶格只剩「不引原文」占位（误食类四个搜索格与语音头部的几个 1–4 行小格）。
- 用户中途要求降低这一项的投入：DeepSeek 全量筛查（36,673 串）只跑了 10/245 批就停了，全量子集的 Claude 筛查停在 8/11 块；最后两个小轮次由主会话读。
  结果文件都在 `privacy_screen/` 下，可断点续跑。

### 自己犯的错与被拦下的
- 第一版护栏版脚本只替换反引号、「」与短单元格，漏了治理理由里的“”引号；自伤求法的散文转述也漏了。
- 名单布线脚本的「json 已导入」断言写死了写法，整段中止（没有写入任何文件），改成正则判断后重做。
- DeepSeek 脚本有一个未用的 import，`ruff && run` 链没启动；它的并发 8 太慢，改成可配置、6 次退避重试、可续跑。
- 复核员把被标串存成「行号 → 哈希」，按引文串求哈希对不上（口径未知），改从它的逐串清单按序号还原 38 串。
- 例子表的占位行写作「不引原文（…）」，最终闸初版按「== 不引原文」比较，已改为前缀匹配。

### 核验
- `p5_snapshot_verify`：未匹配数字 0 · 问题引文 0。全量测试 744 通过（pytest 的 `-q -q` 不打印汇总行，按点数与 exit 0 计）；`ruff --select F src/qmine/ tools/` clean；
  `src/`、`tests/` 自测试以来没有文件改动。
- 已交付文件的 SHA-256 基线（782 个）：781 个不变；变的是 `work/snapshot_classes_all.json`——每次按当前 cohort 覆盖写、代码里没有任何读者。
- 老域可引行数在每次护栏改动后逐域复算：金融 22,784、医疗 21,763、书籍文档 20,016、软件 20,400、金融8 43,497、健康 2,235，全部与已交付的 summary.json 相同。

### 没做
- 其它域的例子没有逐串读（§2）；医疗8 没有做金融8 那份「意图内部结构」深挖。
- 全量数据工作簿（`*_query_挖掘结果_含来源.xlsx/.csv`、`医疗8_POOLED5_逐行标注.xlsx`）按设计含全部 44,019 行原文——那是数据，不是引文，没有做护栏版。

---

## 4. Session (2026-09-16) — 人物 / 影视 / 医疗随机，以及一个哨兵被当成类训练

三个域并行：人物8（ppl-pool8）、影视8（film-pool8）、医疗随机（health-pool3）。全部 fast、routed、未 halt。
用户的三条要求：人物域把 Zhipu 的角色改路由到 Kimi（此前实测过中文公众人物触发 contentFilter）；
人物 / 影视**跳过叙述工作流**；例子表**印真实 query**。三条都照做了，且都验证过：人物运行 log 里
`glm` 出现 0 次、contentFilter 0 次；两份报告的散文段留空而表与图完整；例子表没有占位符。

### 语料与档案

| 语料 | 行数 | 快照 | 档案 | 构建脚本 |
|---|---|---|---|---|
| 人物8 | 43,802（44,000 进） | 8 | `people_zh_v2`（19 种子，覆盖 27.95%） | `build_pool8_corpus.py` |
| 影视8 | 43,933（44,000 进） | 8 | `film_tv_zh_v2`（21 种子，覆盖 32.68%） | `build_pool8_corpus.py` |
| 医疗随机 | 20,316（20,622 进，删 305） | 2 | `medical_zh_v2` | `build_medrand_corpus.py` |

两份 pool8 语料新增第 8 个快照 `assistant_voice_random`。语音头部那一份是**已经截断过的** 1,000 行导出，
所以 `tie_boundary_checkable: False` 写进了构建事实——不像 医疗8 的 1 万行导出可以自己查并列边界。

医疗随机的两份导出 `event_day` 都是 **20260914**（单日，09-15 导出），来源侧已去重。
口径按用户要求比 health-pool2 放宽：助手输入的包装模板只从串里**剥掉**、行保留（`wrapper_stripped` 记住），
宽口径规则只用来挑要盲标的行，**只有三个窄规则**（功能/卡片位、医生卡、无内容）可以删没盲标过的行，
盲标三票一致判为非用户内容才删。结果：助手侧留存 97.1%（health-pool2 是 12%），删的正好是 305 行三票一致的产品层。
三个参考列（`legacy_l2` / `legacy_dept` / `legacy_type`）都满足预注册条件（Cramér's V ≤ 0.55、无 >1% 的单边类），
全部声明；health-pool2 当时撤回过一列。Fleiss κ 0.617 只在歧义那一片上算，报告里照此披露。

### 三个源码缺陷

**① 哨兵被当成第 19 个 L1 类训练。** `_active_learning_round` 建金标行时没有 round 1 的两道保护。
fast 模式只有一个标注员，漏标的行由调用方填 `UNLABELED`；`a2 == b2 == UNLABELED` 满足相等判断，于是
`agreed=True, final="UNLABELED"` 进了金标，过了 p2c 的非空过滤，22 行又过了 5 折支撑下限——
18 个码的体系训出 19 类分类器，幻影类预测到 11 行语料，进了 5 份交付文档，
`topdown_metrics.json` 的 macro-F1 / ECE / CV accuracy 全部是在一个不存在的 19 类问题上算的。
`verify_run` 对照 ppl-pool5b 抓到（新运行 FAIL 1，对照 PASS）。实测爆炸半径：8 次运行里只有 ppl-pool8 中招
（fin-pool8 / med-pool8 / film-pool8 / health-pool3 / ppl-pool5b / film-pool5 / health-pool2 / book-pool5 都是 0 行）——
人物域中招是因为主动学习取的是边界行，而边界行正好是标注员会拒答的敏感尾巴。
修好后 3,178 行训练、18 类、macro-F1 0.526 → 0.557。

**② 只有出错时才走的分支，自己就是坏的。** `_require_both_branches` 传了 `blocking=True`，
而 `Deps.gate()` 从来没有这个参数。它的测试用 `SimpleNamespace(**kw)` 当假 deps，所以一直绿着。
gen03 因为金标从缓存 6.5 秒重放、被排到 `p456_tree` 之后，汇合点真的遇到缺分支——
那道本该报出「缺哪个分支 + 怎么补」的闸门抛了 TypeError，运行死在 traceback 上。
测试的假对象现在 `inspect.signature(Deps.gate).bind(...)`，多一个关键字就报错。

**③ `fast_skipped` 在 resume 时缩水成 4 项。** fast 校验器只在开关「本来是开的」时才追加，
而 `--resume` 读的是源代已经规范化过的配置，于是重建出的清单只剩 4 个无条件项。
横幅是由这份清单生成的，gen03 的三份参考文档因此写着双标注、各阶段观察员、对抗验证、agent 报告、
交付前审核、结果解读**都跑过**——交付文档里的一句假话。改成按 mode 推导，测试断言重复校验幂等。

### 交付代次

人物8 的交付代次是 **gen03**，不是 gen01：
- gen01 = 有幻影类的那一次，留作证据；
- gen02 = 试图用 `--resume` 重跑，结果 `taxonomy_architect` **没命中缓存**、返回 21 个节点（gen01 是 18），
  整棵体系漂了，当场杀掉。这正是 `CLAUDE.md` 警告的「用 web 的研究员不确定，改了 architect 的提示词就一路 miss」；
- gen03 = `--resume --reuse-taxonomy ppl-pool8/gen01`，p2a 0.0 秒跳过、沿用 gen01 的 18 个意图 45 条规则，
  只让修好的金标构造重算 p2c 及其下游。55 次调用（gen01 是 196），其余全部缓存重放；
- gen04 = `qmine render`，只为把缺陷 ③ 修好后的横幅重新渲染出来（三份参考文档 + 运行工作簿）。
  gen03 的 `config.resolved.yaml` 里那份缩水的 `fast_skipped` 已按 mode 推导更正，更正理由写进了 gen03 的 `_why.txt`。

后处理因此需要指定代次：`pooled5_common.run_dir` 新增 `P5_GEN_<批次>` 环境变量（与既有的 `P5_RUN_<批次>` 成对），
`post_ppl8.sh` 里写死 `P5_GEN_ppl8=gen03`。默认仍是 gen01，其它域不受影响。

### 引文护栏

人物 / 影视按用户要求印真实 query，所以**没有**走 医疗8 那套广谱逐串筛查（那次拦了 634 串、掏空了 10 个格子）。
改用 `p5_quote_hardrule_scan.py`：只按三条硬规则扫（未成年人与性、露骨性内容、普通个人可识别信息），
命中不自动拦，打印出来交给人逐条看。医疗随机多一条 `NAMED_DOCTOR`，作为第五层正则的**复核**。

| 域 | 命中行 / 去重串 | 扫描时仍可引 | 已印进交付文档 | 最终 |
|---|---|---|---|---|
| 人物8 | 9 / 9 | 0 | 0 | 0 可引 |
| 影视8 | 9 / 8 | 4 | 0 | 0 可引 |
| 医疗随机 | 76 / 75 | 8（全部是未成年人与性） | 0 | 0 可引 |

三个域都是**没有一个命中串出现在交付文档里**。仍然把命中串写进了各域的
`work/<域>/privacy_screen/quote_block.json`，作为以后换例子时的保险；影视那 8 串里包含一部真实作品名，
按「未成年人与性」从严一并拦下。`EXTRA_QUOTE_BLOCK["医疗随机"]` 指向 医疗8 那条按语料实测出来的正则——
新域少这一层，等于把 med-pool8 验证过的三类写法重新放行。

### 回归

全量 **747 通过**（原 744 + 3 个新测试），`ruff --select F src/qmine/ tools/ analysis/pooled5/*.py` clean。
三个新测试在各自的缺陷代码上都验证过会失败。887 份既有交付文件哈希不变，唯一变的是
`work/snapshot_classes_all.json`——按批次重写的中间汇总，只写不读，不是交付物。

### 没做
- 人物 / 影视没有叙述工作流（用户明确跳过），两份报告的散文段是空的。
- 医疗随机也没做叙述工作流——用户没提，而它是和人物 / 影视同一批交付的。需要的话单独说一声。
- 意图内部结构深挖仍然只有 fin-pool8 一份。
- 缺陷 ② 只修了「闸门建不出来」，**没有修排程本身**：gen03 第一次跑时 p2b_gold 确实被排到了 p456_tree 之后。
  现在再撞上会干净地 halt 并说清楚缺哪个分支，但为什么会这样排还没查。见 §2。

### 补记（2026-09-16 傍晚）—— 人物域重跑成 ppl-pool8b，并复查了这次会话的全部源码改动

用户提了两件事：一是源码改动必须经得起推敲、不能把已经稳定的程序改坏；二是 **resume + fast 本来就是
§2 里没关掉的口子**（0t / 0v），那 gen03 的交付到底可不可信。两件都成立，处理如下。

**① 人物域整个 run id 作废，重跑 `ppl-pool8b`。** 不是因为 gen03 的数字查出了错，而是因为它的**记录**
确实比别人薄，而且这一薄是可测量的：

| | ppl-pool8 gen03 | ppl-pool8b gen01 |
|---|---|---|
| 闸门 | 14（缺 `p2a_pilot_agreement`、`p2a_taxonomy_shape`） | **16** |
| decisions | 6 | **7** |
| `elapsed_s` | 141.5（只有 resume 之后那一段） | **5,318** |
| `fast_skipped` | 4（事后更正） | **10**（本来就对） |
| resumed | True | **False** |
| provenance | 两次 resume + 一次崩溃 | 一次跑完 |

缺的那两个闸门是真的质量检查，因为 `--reuse-taxonomy` 跳过了 p2a 才没跑。其余七次运行全是单次全新运行，
人物域不该是唯一的例外。ppl-pool8b：fast、routed、未 halt、238 次调用、88.6 分钟、18 个阶段，
`verify_run` 对照 ppl-pool8/gen01 是 **21 PASS / 0 FAIL**（对照仍然 FAIL 幻影类那一项，说明工具在咬）。
分支汇合点这次是正常的：p2b_gold 16:25:04 ✔、p456_tree 16:32:58 ✔、p2c 16:33:01 ✔，两条分支都到齐了，
护栏没触发——**gen03 那次错序是缓存重放的 resume 造成的，不是全新运行里潜伏的缺陷**。

标注员这次一条没漏（3000/3000 + 200/200），所以修好的保护根本没用上：gold 3,200 行、0 个哨兵。
gen01 那 22 行漏标是随机的（一个 batch 三次重试后丢了行），不是语料的性质。

体系是独立重推的：**22 个 L1，与 gen01 的 18 个只共用 2 个码**——两次运行不共享标签空间，这是
CLAUDE.md 写明的，所以不要拿两次的类目互相比。交付形状 21 L1 / 53 L2 / 37 族 / 37 叶
（gen03 是 18 / 50 / 22 / 23），叶和族都厚了不少。分类器：21 类、CV 0.719、macro-F1 0.399、
**按流量加权的准确率 0.703（三者里最高）**、ECE 0.022（三者里最好）。macro-F1 掉是因为类数从 18 变 21、
分母里多了几个稀有类，不是退化——读这个数必须同时读基数。
`n_dropped_rare=3` 是**行数不是类数**：真正训不了的类只有 1 个（`JUDICIAL_RECORD_CHECK` 涉案判决与失信记录核查，
金标只有 3 行），而且交付语料里没有任何一行落在它下面。

**② 三处源码改动经过了一轮对抗复核**（20 个 agent、6 个维度、每条发现都由独立的怀疑者试着推翻；
14 条候选里推翻 12 条、确认 2 条）。三处改动本身**全部通过**，其中最值得记的一条排除：
空的 `final` 写进 csv 再读回来，会不会变成字符串 `"nan"`（长度 3）从而穿过 p2c 的 `str.len() > 0`
过滤、把本该剔除的行又放回训练集？三种方法实测：不会。这是改动 ① 唯一可能「白改」的方式。

确认的两条都是我自己的，都不在挖掘程序里：
- **我新写的那个测试断言不可能失败。** `gate.status != "PASSED"` —— `GateStatus` 是小写字面量；
  `getattr(gate, "warn_only", False)` —— `GateResult` 根本没有这个字段。变异测试里 `passed=True`、
  `skipped=True` 都活了下来。更糟的是它的失败信息说的是假话：这个闸门实测是
  `status='warned', blocking=False, halts_run=False`，因为 `p2c_both_branches_arrived` 不在
  `cfg.gates.blocking` 里。已改成断言真正承载停机的东西（返回字典的 `halted` / `halt_kind` /
  `halt_reason` + 闸门的真实 status + remediation 非空），现在三个变异体全部被咬掉。
- **`P5_GEN` 改了「读哪个代次」，但生成文案里四处写死的 `gen01` 没跟着改**
  （p5_snapshot_report.py:479/857/860、p5_snapshot_classes.py:1094）。于是人物域的交付文档正文写着
  `ppl-pool8/gen01`、复现表指向一个不存在的图目录。四处都改成跟 `run_dir` 走；gen01 的域重渲染后
  `.md` 逐字节相同、工作簿每一格相同（只有 zip 时间戳变）。

**③ 复核者判错了一条，我自己复查后确认它是真的。**
`p5_quote_hardrule_scan.py` 写死 gen01 而 `load()` 跟着 `P5_GEN`，所以**人物域那次扫描读到的文档数是 0**，
「已印进交付文档 0」是一句空话而不是一个测量结果——而我把这个数字当成结论报给了用户。
按真实文档重测，答案仍然是 0（结论没错，但当时没有证据）。脚本已改成跟 `run_dir` 走，并且
**读不到文档就直接 assert 失败**，让沉默不可能再被当成通过。影视 / 医疗随机当时各读到 1 份文档，是真测量。

回归：全量 **747 通过**，`ruff --select F` clean，887 份既有交付文件里 886 份哈希不变
（唯一变的仍是按批次重写、只写不读的 `work/snapshot_classes_all.json`）。

### 补记（2026-09-16 深夜）—— CLAUDE.md 例行维护

228 行 → **199 行**（目标线以下），内容一条没丢：40 条不变量里 34 条**搬进** `.claude/rules/` 的对应文件
（按 `paths:` 只在打开相关文件时才加载），CLAUDE.md 只留 6 条真正跨领域的——它们正好各自对应「Rules」一节里
的一条规则，表变成了那一节的执行索引。搬家后逐条核对：40 个测试名仍然全部出现在文档里，
8 个规则文件的 `paths:` 通配符全部命中真实文件，文档里点名的每个测试在 `tests/` 里都恰好定义一次。

**改对的事实（都是实测，不是读 HANDOFF 得来的）：**
- 测试数 ~730 → ~750（实测 747），`make demo` ~4 分钟 → ~3 分钟（与 Makefile 自己的说明一致）。
- **文档里的 pytest 命令自己把答案藏起来了。** `pyproject.toml` 的 `addopts` 已经带了 `-q`，
  命令行再加一个就是 `-qq`：`-k` 匹配不到任何测试时输出**一个字节**，而不是 `N deselected`。
  本次会话里我自己中过两次（`pytest -q | grep passed` 什么都没有）。命令里的 `-q` 已删，并写明不要加回去。
- **`--fast` 只有在全新 run id 的命令行上才生效。** 实测：`_load_config` 总是传显式 mode，所以
  `--config` 文件里写 `mode: fast` 会被读成 `full`；`--resume` 根本不看这个标志，只沿用源代次的配置，
  且只应用 `run_root` 与 `--reuse-taxonomy`。这条原来完全没写在 CLAUDE.md 里，而它决定一次运行花多少钱、
  有没有 kappa。（§2 的 0t 记着这件事，但措辞是「resume 会变成 full 模式」——不准确，实际是「沿用源代次的模式」。）
- **`extends:` 是一条链，不是一行。** 原文写「必须以 `extends: live.yaml` 开头」，而
  `configs/live_finance.yaml` 其实 extends `corpus_wise_export.yaml`。改成「链必须到达 live.yaml」，
  并给了直接（`pool5_fin.yaml`）与两级（`live_finance.yaml`）两个例子。
- `verify_run` 的状态是五个不是四个，`ERROR` 原来没写。
- README 的「711 tests」同步刷新。

**新增的三条通用规则**（都来自本次会话踩到的真事，且都不是某个子系统专属）：
- **`--fast` 只在全新 run id 生效**（上面那条）。
- **断言不只用于脚本改写，也用于扫描的输入集**：glob 之后 `assert files`——空的输入集是 SKIP，不是 PASS。
- **不可能失败的测试会永远是绿的**：新测试先证明它在缺陷上会挂，再信它；比真接口松的假对象
  （`SimpleNamespace(**kw)`）和断言一个类型根本没有的字段，都会在坏代码上保持绿色。

**没改的：** `make full ~25 分钟`（复核 agent 实测 12 分钟，但 Makefile 自己的说明也写 25，两处一致，
且这个数字不指导任何决定，留着）；「fast 模式交付 3 份参考文档」（实测两次 pooled fast 运行都正好 3 份，
复核 agent 提的「pooled 时 4 份」与实测不符）。

### 补记（2026-09-17）—— 给两个医疗域新建了一层「科室」

用户问：这两份医疗语料里到底涉及哪些医院科室、跨快照怎么比、为什么。现有标签答不了这个问题，
所以**新建了一层**，纯后处理，不改任何运行产物或已交付文件。

**为什么不能用现成标签（实测）：**
- **医疗8 没有科室列。** 唯一的平台标签 `l2` 只覆盖 1,956/44,019 行（4.4%，只在两个助手快照），
  而且是内容类型不是科室（疾病知识/药品保健品/养生知识/医疗服务/医疗器械/医疗其它）。
- **医疗随机 的 `legacy_dept` 测的不是科室。** 20 个取值覆盖全部行，但 **44.62%（9,066 行）是空档**；
  词表里**没有肿瘤科**——本层判为肿瘤科的 476 行，平台给的是 内科 257、两个空档 100、妇产科 64……
  一行都没落到任何肿瘤相关取值上；「内科」4,104 行被本层拆成 消化 781、心血管 541、内分泌 397、
  呼吸 322、检验影像 261、肿瘤 257、神经 228……；取值里还混着 综合医院、医疗服务其它、五官科
  （现行名录里已被眼科/耳鼻咽喉科/口腔科取代）这类根本不是科室的东西。
- 空档也不是随机缺的：9,066 行空档里 **5,163 行本层判给了具体临床科室**，只有 3,903 行真是非临床。

**做法：** 预注册码本（`work/科室/codebook.md`）锚在**《医疗机构诊疗科目名录》**（卫医发〔1994〕第27号，
国家卫健委现行有效版本，34 个一级 / 148 个二级）。三处偏离都写明理由：内科与外科**拆到二级**
（不拆就是复制平台标签的缺陷）、增设**男科**（名录外但中国医院普遍设置）、性病在皮肤科下单列。
判定口径是**分诊**（该挂哪个门诊），并设 5 个非临床码，不把养生/用药/挂号硬塞进科室。
56,937 个不同 query 全量逐串判定（`deepseek-v4-flash`，0 条未标）。

**信度（三个读数测的是三件事，不能混）：** 跨供应商 kimi-k3 n=1,500 **一致 82.4%、κ 0.815**（这才是信度）；
同族 deepseek-v4-pro n=1,000 一致 88.3%、κ 0.866（测容量差）；可审计的规则词表覆盖 23.5%、一致 76.9%、
κ 0.755（测「只认锚点的仪器」）。**越独立的仪器越不一致**。临床/非临床粗分跨供应商一致 **92.7%**。
不一致集中在几对固有模糊边界：痛风（内分泌↔风湿免疫）、淋巴瘤（肿瘤↔血液）、早孕（产科↔妇科）。

**主要发现：**
- 两份语料各出现 **34 个临床科室**；临床行占比 医疗8 67.24%、医疗随机 73.65%。
- **时间几乎什么都没发生**：2025→2026 随机层最大科室变动 0.86pp、头部层 0.95pp。
- **分层比时间大近八倍**：同一天头部 vs 随机最大 6.61pp。头部堆中医（12.72% vs 6.37%）、用药、养生；
  随机尾部才是就医事务、儿科、产科。**任何只看头部 1w 的医疗分析都会系统性高估中医养生、低估儿科产科。**
  两年独立复现同一形状。
- **两份语料的「搜索 vs 助手」方向相反。** 医疗8：39 类中 17 类显著，**14 类搜索侧更高**，助手侧唯一
  大幅上升的是「无法判断」+12.09pp；形态表给出原因——该助手上的串**更短**（中位 8.5 vs 10.4 字）、
  **更不像问句**（30.9% vs 37.8%）。医疗随机：18 类显著，**12 类助手侧更高**，串**更长**（12.2 vs 10.7）、
  **更像问句**（55.4% vs 46.4%）、**第一人称多近两倍**（5.9% vs 2.2%）。原因是产品不同：前者是通用助手的
  医疗切片，后者是健康管家。
- **肿瘤科在两份语料里符号相反**：通用助手 −0.46pp（搜索更高），健康管家 +1.29pp（助手更高），两处都显著。
- **语音**：语音头部「无法判断」24.65% vs 文字头部 16.18%；「就医事务」在语音里几乎消失（0.37% vs 7.83%）。

**护栏：** 例子走与其它交付报告相同的七层护栏，**外加两个医疗域筛查名单的并集**——这个并集是必要的，
实测到一个 医疗8 名单上的串作为 医疗随机 的例子印了出来（见 §2 新增条目）。最终 1,100 条例子
**四条硬规则命中 0、筛查名单命中 0**。

**交付：** `deliverables/医疗科室层_跨快照对照.zh.md`（55,048 字符 / 870 表格行）+ 同名 `.xlsx`（18 页）
+ `work/科室/` 下的码本、逐串标注、规则词表、19 张对照表。
新脚本：`p5_department_label.py`、`p5_department_rules.py`、`p5_department_compare.py`、
`p5_department_reliability.py`、`p5_department_report.py`。

**没做：** 没有图（这一层的结论靠表就能读，图会重复）；没有把科室层交叉到意图层（`load()` 已经让两者
落在同一份行上，随时可做）；没有动任何域的护栏配置。

### 补记（2026-09-17 晚）—— 医疗3：用 health-pool3 自己的产物给新快照打标签，凑齐三快照

新数据：`健康管家-医疗-Top1w.xlsx`（11,464 行），与 health-pool3 的两个快照**同一天**（`event_day` 都是
20260914）。它补上的正是那次运行缺的那一层——健康管家的**头部**。于是三快照是：

| 快照 | 界面 | 抽样 | 行数（清洗后） |
|---|---|---|---|
| 传统搜索随机1w | 搜索 | 随机 | 10,000 |
| 健康管家随机1w | AI助手 | 随机 | 10,316 |
| **健康管家Top1w** | AI助手 | **头部** | **11,252** |

**没有重新推导任何体系**，交付形状与 health-pool3 逐项相同：**20 L1 / 51 L2 / 33 家族 / 33 叶**。

**打标签怎么做的（关键）：** 用该运行自己的产物打分，不重训不重聚类。
- 稀疏空间在**原语料的 20,316 条文本**上重拟合，实测**逐位复现**该运行的 `emb_svd_char.npy`
  （max abs diff **0.000e+00**，vocab 50,732、evr 0.218149 全对得上）；稠密编码器
  `shibing624/text2vec-base-chinese` 复现到 3.3e-07、平均余弦 1.00000000。所以新串落进的是**同一个**空间。
- 叶与家族用 `centroid_classifier.joblib`（37 叶心 × 1024 维），L1 用 `topdown_model.joblib`
  （RuleEngine + StandardScaler + LogReg），L2 用从原语料 (`emb_base`, `td_l1`, `td_l2`) 现算的 L1 内子中心。
- **打分器准确性是测出来的**（同一条路径给原语料 20,316 行打一遍，比交付标签）：
  **td_l1 100.00%、td_l2 99.57%、叶与家族 95.51%**。叶那 4.5% 的差是最近中心法在逼近
  「聚类 + p8 治理」的结果，而这正是该运行 `deployment.json` 自己写明的服务路径。
- **老两个快照的标签一个都没重打**，直接取 `labels_full.csv`。

**清洗：口径相同，但仪器是重建的，而且重建过一次才达标。**
规则直接 import 自 `build_medrand_corpus`（同一份对象，不是副本）。盲标码本当年写在工作流提示里没落盘，
这次重建并**落到文件**（`work/医疗3/audit_codebook.md`）。为了证明重建复刻得了原仪器，除 989 个新歧义串外
还重标了原来 1,192 串里的 300 串做校准：
- **第一版：删/留结论只一致 78.7%，且偏差完全单向**——多删 0 串、少删 64 串，其中 71 串原仪器判 A、
  重建版判 U。读那批串后定位到定义错误：**把「描述自己的症状」当成了用户键入的证据，而分诊表单的选项
  本来就在描述症状**。
- 补上「A 的识别 signature」（选项兜底词「都没有/不清楚/都不是」、无连接词的平列症状、纯检查数值罗列）后
  **第二版：多数码 90.7%、删/留 91.3%、逐票 82.3%，Fleiss κ 0.748 → 0.838**。
- **残余偏差 8.0pp，仍是单向偏保守**（原仪器删 73、重建版删 49，只多删 1）。必须修的理由是可比性：
  仪器比另外两个快照宽松，就会把仪器差异算到快照头上。第一版产物留在 `audit_v1/`。
- 新快照：11,464 → **11,252**（留存 98.2%，随机快照是 97.1%）；包装模板占比 **85.8%**（随机快照 75.4%）。

**主要发现（三快照同一天，所以时间不是任何一对的混杂项）：**
- **分层差异 > 界面差异，而且是两倍多。** L1 层 TVD：搜索随机 ↔ 助手随机（界面，两个产品）**0.147**；
  助手随机 ↔ 助手头部（分层，**同一个产品、同一天**）**0.324**。
- 助手的**头部是通用知识查询**，尾部才是个人化的临床问题：
  「药品/保健品/食物信息与功效」头部 **30.88%** vs 随机 9.68%（+21.2pp）、
  「医学名词/检查/疾病定义」21.86% vs 11.86%（+10.0pp）；
  反过来「症状或体征的原因」8.14% vs 17.89%（−9.7pp）、
  「判读个人测量值/检查结果」3.71% vs 11.36%（−7.7pp）。
- 这与 医疗8 在**搜索侧**看到的头尾结构是同一个形状，现在在**同一个助手产品内部**独立复现了一次。

**护栏：** 新快照没有运行自己的风控图层（第一层按行号对齐，只覆盖前 20,316 行）。补法：把
`EXTRA_QUOTE_BLOCK["医疗3"]` 指到 医疗8 那条实测正则、`NAMED_DOCTOR` 复核规则扩到 医疗3、
筛查名单取 **医疗8 + 医疗随机 + 本次硬规则命中**的并集（769 串）。最终 `未匹配数字 0 · 问题引文 0`，
硬规则命中 137 行中**仍可引 0、已印进交付文档 0**。

**派生目录不是运行：** `runs/health-pool3_scored3snap/gen01`，带 `_why.txt` 写明它没有 run_summary /
run.log / llm_cache，**不要对它跑 `verify_run.py`**，以及那份复制来的运行工作簿只为定义页
（逐行页只覆盖 20,316 行，与这里 31,568 行不对齐，所以 `p5_postprocess_run_xlsx` 不在链条里）。

**顺手修的一个通用缺陷：** `p5_deliverables` 用 `src_x.exists()` 判断运行工作簿是否存在，而
glob 落空时 `next(..., "")` 会让 `src_x` 变成运行**目录**，目录也 exists——于是 pd.ExcelFile 收到一个文件夹。
改成 `.is_file()`，有工作簿的域行为不变。

**回归：** 全量 **747 通过**、`ruff --select F` clean、887 份既有交付文件里 886 份哈希不变
（唯一变的仍是按批次重写、只写不读的 `work/snapshot_classes_all.json`）。health-pool3 的运行目录与
它的交付物一个字节都没动。

**没做：** 按用户要求**不跑叙述工作流**（报告的散文段留空，表与图完整）；没做科室层的三快照版
（那一层的码本与脚本都在，随时可以对 医疗3 跑一遍）。

---

## 2026-09-20 —— 把后处理与语料准备做成程序的功能，并加一个对话入口

**三件事一起落地，全部是新增模块 + 三处必要的接线；已有 747 个测试一个没动。**

### 1. `src/qmine/pooled/` —— 跨快照对比，成为阶段 `p10c`

`analysis/pooled5/` 那 56 个脚本的**计算内核**搬进程序，十张按语料写死的登记表
（`DOMAINS` / `SOURCES` / `SRC_ZH` / `RUN_ID` / `COHORTS` / `CONTRASTS` …）全部换成**从运行自己的产物推导**：
快照是 `p0` 打的标签，类目是交付的那棵树，风控层是运行自己的 `risk_screen.json`。
用户只能提供编辑性的东西——显示名、分组、要对比哪几对——而且都有能直接用的默认值。

模块：`manifest`（快照、分组、对比对、`pv_norm` 快照内归一到 10,000）、`stats`（Wilson /
Newcombe / TVD + 自助法区间 + **同源噪声上界**，种子按「测什么」派生而不是按调用顺序）、
`classes`（逐类 × 逐快照矩阵、缺席可检出性、特征类、前十、条件构成、例子）、`guards`（七层引用护栏）、
`figures`、`report`、`verify`、`workbook`、`pipeline`。

**对着已交付的 health-pool3 逐项复算过**：四个层级 20 / 51 / 33 / 33 共 137 个类，
条数与占比**逐个相同**；TVD 0.1473 / 0.1540 / 0.1154 / 0.1154、秩相关、Cramér's V 四位小数**完全一致**。
自助法区间端点差 ≤0.0016——那是**设计上的**：种子由 `(用途, 层级, 快照对)` 派生，新旧标签串不同；
四个层级「超出噪声」的结论两边都一样。

接线三处：`graph/build.py` 的 `PHASE_NODES` 里 p10b 与 p11 之间插 `p10c_pooled`（`SEQUENTIAL_TAIL` 自动派生，
`p10c` 是 `p10` 的后缀所以仪表盘自动折叠）；`state._PHASE_ORDER` 加 `p10b` / `p10c`；
`p11` 两条路径把 p10c 的文件计入交付物，`zh_reference._WHAT_FOR` 加三行索引。
新的 `pooled:` 配置段；**fast 模式不丢它**（不调用任何模型，fast 去掉的是复核不是分析），
`smoke` 把重抽次数降到 60/40。
另加 `qmine compare RUN_ID`，对已完成的运行随时重跑。

### 2. `src/qmine/prepare/` —— 把几份导出合并成一份语料

`inspect` 先**纯机械地量**每个文件（列与 dtype、重复率、长度分布、权重是否降序与集中度、
高覆盖前缀、互相重合度）。`planner` 从这些数推出一份方案；`CorpusPrepAgent` 可选地读这些**测量值**
（不是原始行）提出自己的方案，`validate_plan` 逐条校验后才采用，否则用机械方案并记下为什么被拒。

**方案是数据，不是模型写的代码**：十个有界的具名操作。执行器强制两件方案管不了自己的事——
**任何一行都不删**（每次移除只是打 `tier`，被移除的行连同原因写进 `prepared_all_rows.parquet`）；
**超过 `max_drop_share`（默认 0.25）的规则被降级为标记并记为「拒绝执行」**，返回全 False 的掩码
才算拒绝，只警告的护栏是在语料已经变了之后才被读到的。

实测：在 `健康管家-医疗-Top1w.xlsx` 上，检查器自己找出 **85.81% 的行以「我想咨询」开头**——
与手工构建器写死的 `WRAPPER` 完全一致，而且是零模型调用推出来的。

`qmine prepare A,B,C`（`--dry-run` / `--agent`）与 `qmine run --prepare`（需要 `--run-id`，
准备好的语料落在运行目录里当证据）。

### 3. `src/qmine/chat/` —— `qmine chat`

模型只能从**固定的动作目录**里挑一个并填写有类型的参数；程序决定跑什么。
`ChatStep.action` 是 `Literal`，目录外的动作过不了 schema；执行器再查一遍。
免费且可逆的动作直接做，**花钱或写文件的停下来**并打印等价命令行——
每一步都打印命令，是为了让人最终不需要这个界面。关键词路由不是没人跑的兜底：没有模型时就用它，
测试用的是它，而且它的答案会作为基线交给模型，模型要改就得给出理由。

### 途中自己发现并修掉的七个真缺陷（都带回归测试）

1. **`deps.df` 是 property，我按方法调用了**，而节点里那个宽 `except` 把自己的 TypeError
   变成了「语料不可用」——一次 0.0 秒、什么都没产出、日志里看不出来的「优雅跳过」。
   现在每条跳过路径都 `emit`。
2. **引用护栏把声明的正则拼成一个 alternation**，而领域档案里真的有 `(?i)` 开头的模式——
   `global flags not at the start of the expression`，整层编译失败，护栏**失效为放行**。改成逐条编译。
3. **表与卡片各自四舍五入**：`.round(2)` 与 `f"{v:.2f}"` 在所有以 5 结尾的值上不一致，
   同一个量印成 `1.8` 与 `1.80`。统一成一个格式化函数。
4. **CSV 往返把空格变成 `nan`**，`缺席于 nan` 印进了报告。加 `_blank()`。
5. **「整类不引」那一层用单个汉字命中**：一个裸 `裸` 命中了 `查询物品或活动的功效与作用`
   定义里的「**裸**名称」，把 med-pool8 的 **28.6%（12,602 行）**整类排除在引用之外——
   那一类的风险是「不该那样作答」，引「阿司匹林的功效」不伤害任何人。词表改成**只收复合词**，
   七个已交付语料重测：med8 / health3 只剩 `SELF_HARM_CRISIS`（0 行），
   软件 1.2% / 书籍 1.6% / 人物 1.0%（正是该拦的那几类），金融与影视 0。
   med-pool8 实测：拦下 15,335 → 2,976，可引 28,684 → 41,043，
   82 个「叶 × 快照」格子里有真实例子的从 79 个变成 **82 个**。
   这一条是**按规矩去看图与表才发现的**——数字本身不会说自己不对。
6. **`register_file(..., "report", ...)` 的 kind 不在 `ArtifactKind` 里**，于是它在节点自己的
   `except` 里被吞掉，报成「文件未登记」——文档写在盘上，却没有任何产物指向它们，p11 也就无从交付。
   改成 `markdown`；测试从「有 `pooled_comparison` 这个 json」收紧到「三个文件都登记了且路径存在」。
7. **重渲染的代次会认领它没有的文件**：代次继承上一代的产物索引，所以 `deps.has("report_pooled")`
   为真而文件在上一层目录，而索引里的链接是相对的——读者会点到一个不存在的路径。
   新增 `_pooled_refs()`：只认这一代目录里真的有的那几个。p10c 不在重渲染路径上，
   所以重渲染出来的代次就是没有对比，如实说出来才对；要的话 `qmine compare` 重建。

另外两处按实测改的判断：高覆盖前缀现在**延伸到完整长度**（原来停在三个字，得到 `我想咨` 而不是
`我想咨询`，剥完留下一个「询」，后面的合并就无事可做）；「像头部导出」的判断改成**以降序为准**、
集中度只作修饰（原来两个条件都要，漏掉了尾部平坦的真头部导出）。

### 按用户长期规矩补上的一条通用护栏（`configs/` 最终没动）

用户的长期规矩是「绝不引具名医生」。实测发现它**当时靠的是运气**：med-pool8 里 26 条
「科室/医院 + 姓名 + 医生」的串有 **25 条仍然可引**，没被印出来只是因为按流量取例没轮到它们。

先试着补进医疗档案的 `personal_data_exposure`（那一类的政策原文就是「never surface identifiable
patient or record data」）。**这条路不够**：一次运行的 resolved config 会把领域档案**冻结**下来，
所以档案里的规则保得住以后的运行，保不住对已有运行重跑 `qmine compare`。于是改成放进**第二层通用硬规则**，
并把档案改回原样——一条规则只放一个地方。

实测代价：八个已交付语料里最多 0.11% 的行，其中五个是 0。
也试过用**姓氏表**收窄以减少误伤，**实测更差**：`新晃县中医院妇科主任唐医生`、`胸外科刁明强医生`
这类姓名跟在科室后面的真具名医生会被漏掉。宽的那版误伤的是 `善良的医生`、`吸血鬼医生` 这种剧名，
**错的方向是便宜的那一边**，所以保留宽版，只把实测出来的泛称（`急诊科医生` / `产科医生` /
`年轻医生` / `什么是医生` …，以及 `科?` 这个让「儿科医生」与「急诊科医生」同等对待的写法）排除掉。

### 对抗式复核：四个视角提了 30 条，逐条反驳后 15 条成立，全部修掉

写完以后让四个独立视角（数值正确性 / 护栏与复核 / 语料准备 / 接线）把新代码读一遍，
每条发现再派一个「专门去驳倒它」的复核员。**30 条里 15 条经得住反驳**，其中最重的几条：

1. **同源噪声上界是按「各自对半切」抽的**，而观测值比的是 n_a 与 n_b。TVD 噪声 ~1/√n，
   对半切把上界抬高 **1,000/1,000 时 1.41 倍、10,000/10,316 时 1.44 倍、400/4,000 时 1.80 倍**——
   快照越悬殊越严重，真实的 5pp 差别会被判成「测不出」。方向是保守的（只会藏住差别、不会造出差别），
   所以它活了下来；而**同一个包里的 `conditional_mix` 本来就是对的**（合并后按原行数切回），
   两处在同一个列名下算的是两个问题。已按 `conditional_mix` 的做法统一。
   已交付语料重测：四个层级的结论都不变，上界从 0.040–0.062 降到 0.028–0.043。
2. **p10c 把 `cfg.data.weight_column` 传给了 p1 之后的帧**。那个名字是**原始文件**的列
   （`wise_pv` / `total_pv`），p1 已经把它改名成 `weight`——于是每个快照都退化成均匀权重，
   **每一个「流量占比」悄悄变成了行占比**。
3. **卡片把 query 截到 26 字再加「」**，而硬规则扫描是在文档里找语料串——前缀一个都匹配不上，
   于是「已印进交付文档 0」这个数（那一层存在的全部意义）在截断的情况下是假的。
   改成印整串，并让扫描同时读 `tables/examples_*.csv` 且支持前缀匹配。
4. **`head_cut` 在权重列只有 6.7% 有值时静静剔掉 93% 的行**——它按设计豁免于「过大移除」护栏，
   所以正是它需要自己的下限。现在低于 90% 覆盖率直接拒绝。
5. **`aggregate_by_query` 把「没有权重」聚合成 0.0**（pandas 对全 NaN 组求和给 0），
   于是 `head_cut` 的「没有可用权重」拒绝被解除，在一列 0 上排序剔行。
6. **`merge_collisions` 让 all_rows 少于原始行数**，而报表只按合并后的行数算剔除率，
   于是 3 行变 2 行被报成「剔除 0.0%」。现在加 `n_rows_raw`（带断言）与「合并掉的行 / 占原始行数%」。
7. 还有：日期列被当成权重列、空单元格因为 `str(nan)=='nan'` 而算「有内容」、
   缺失权重被补成**中位数**（改成最小值）、两个快照显示名相同会静默合并成一列、
   `enrich` 无条件读另一条路线的列（单路线运行会丢掉整个对比）、
   `verify` 漏掉超过 80 字与写成 `『』` 的引文、数字池缺少报告自己算出来的数、
   `--previous` 读不出来时复现检查无声消失、`run --prepare` 永远把轴当成已声明的、
   `--text-column` 会覆盖准备好的语料的列名、对话里把 warn 当成 failed、
   `isinstance(True, int)` 让布尔列印成 `1` 而不是「是」。

被反驳掉的 15 条里，典型的是「p10c 崩了会不会把整次运行带停」——不会，节点自己接住并降级成告警门，
而且有测试钉着。

### 一个测出来、因此没有做的设计

想用跨文件重合度自动判对比轴。**实测否掉了**：同一界面的两份随机 1w 抽样、相隔一年，重合 **0.02%**；
同一界面的两份头部导出重合 **63.3%**；不同界面重合 **0.00%**。重合度分的是头部与长尾，不是时间与界面。
所以推出来的轴一律写进 `concerns` 当**假设**说出来，且永远不给 `confidence: high`。

### 验证

76 个新测试；**55 个变异体逐一验过全部被杀**（每个测试都先证明它在它所针对的缺陷上会失败）。
全量套件 823 通过（747 原有 + 76 新增）；`ruff --select F src/qmine/ tools/` clean。
`analysis/pooled5/` 一个字节没动，已交付的运行与产物一个字节没动。

**没做**：`pooled` 只写进 `runs/<id>/<gen>/pooled/`，不碰任何已交付目录；
产品层的盲标（三视角 + 码本）没有进程序——它需要人写码本并做校准，机械方案只做得到词表能做到的部分，
在 医疗3 上是 98.5% 保留率 vs 手工的 98.2%，差的 38 行正是盲标那一层。
