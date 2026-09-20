# Audit 2 — v3 cleaning rules on fresh samples

Independent audit of `tools/clean_assistant_functional.py` (v3 output in `clean_v3/`). Every number below is computed by `audit2_report_build.py` from `audit2_samples.parquet`, `audit2_round2_samples.parquet` and the label dicts in `audit2_labels.py` / `audit2_labels2.py`; no project file or pre-existing scratchpad file was modified.

## 1. Method

- **Criteria** were the fixed A / B / C definitions given with the task (A = system-authored or content-free; B = user; C = ambiguous; single characters in education search → C unless the lookup reading is unambiguous).
- **One label per distinct string per surface**, assigned from a shuffled list showing only id, surface, category and the string — never tier, PV, snapshot or sample. Labels are stored as dicts keyed by exact string; `check()` asserts every key matches a sampled row and every sampled row has a label.
- **Fresh samples.** All 1,133 strings in `audit_clean_samples_labeled.parquet` (`qs`) and `audit_clean_search_samples.parquet` were excluded. Rows removed by that exclusion: 3a changed 11, 3b S4 11, 3c S5 14, 4b search non-user 18, search v2→v3 restored 16. The 16 restored search rows (C1 → user) were ALL first-audit strings, so no fresh restored-search rows exist and none are scored.
- **Round 1** (seed 20260910): samples 1–4 below, 1,006 distinct strings. **Round 2** (seed 20260911, frozen before labelling): a census of every kept 5-domain head row labelled U05, holdout rows for each rule proposed from round-1 misses, a replay of the search S5 rule on the 2025 search snapshot, and 55 decoy rows — 410 distinct strings, shuffled together so rule membership was not shown.
- **Limitations.** One labeller. Before labelling I had read `sva_tier_diff.txt`, which quotes ~100 of the changed rows scored in sample 3a. The headline/keyword boundary is a judgement: a noun phrase ending in `最新消息/最新进展/后续` was labelled keyword-shaped (B), a complete reported-event clause A; a bare name + `去世` on search B. Rule holdouts mix matches from several rules plus decoys, which reduces but does not remove anchoring.

## 2. Key numbers

| sample | rows (distinct) | A | B | C | metric | rate [95% Wilson] | excl. C | PV-weighted |
|---|---|---|---|---|---|---|---|---|
| 1. Kept head (5 domains top1k) | 300 (299) | 21 | 245 | 34 | miss A/n | 7.0% [4.6, 10.5] | 7.9% [5.2, 11.8] | 6.1% |
| 2. Kept tail (5 domains random1k) | 200 (200) | 10 | 165 | 25 | miss A/n | 5.0% [2.7, 9.0] | 5.7% [3.1, 10.2] | 4.1% |
| 3a. Newly removed (v2 user → v3 non-user) | 150 (150) | 145 | 4 | 1 | precision A/n | 96.7% [92.4, 98.6] | 97.3% [93.3, 99.0] | 99.4% |
| 3a. Newly restored (v2 non-user → v3 user) | 3 (3) | 0 | 3 | 0 | precision B/n | 100.0% [43.8, 100.0] | 100.0% [43.8, 100.0] | 100.0% |
| 3b. S4_suggested_chip, 100 random (33 cats) | 100 (99) | 96 | 3 | 1 | precision A/n | 96.0% [90.2, 98.4] | 97.0% [91.5, 99.0] | 99.9% |
| 3c. S5_headline, 100 random (33 cats) | 100 (100) | 90 | 10 | 0 | precision A/n | 90.0% [82.6, 94.5] | 90.0% [82.6, 94.5] | 89.0% |
| 4a. Search kept, top 1,000 by wise_pv (30/domain) | 150 (150) | 0 | 150 | 0 | miss A/n | 0.0% [0.0, 2.5] | 0.0% [0.0, 2.5] | 0.0% |
| 4b. Search non-user rows (all fresh) | 22 (22) | 15 | 6 | 1 | precision A/n | 68.2% [47.3, 83.6] | 71.4% [50.0, 86.2] | 76.5% |

PV-weighted miss share within sample 1: **6.1%** (domain-stratified bootstrap 95% [3.2, 9.7], 5,000 resamples). Domain-weighted estimates over the kept-row pools: head A 6.8%, C 10.9%; tail A 5.0%, C 12.3%. Decoys (40 random kept top1k rows from all 33 categories, round 2): A 7.5% [2.6, 19.9], C 3/40. For scale, the first audit (v2, 33 categories) reported 11.3% [8.2, 15.4] of kept top1k rows as A; the scopes differ, so this is context, not a like-for-like comparison.

### Per domain (miss rate A/n, n = 60 head / 40 tail)

| domain | head A/B/C | head miss | tail A/B/C | tail miss |
|---|---|---|---|---|
| 金融 | 1/59/0 | 1.7% [0.3, 8.9] | 1/37/2 | 2.5% [0.4, 12.9] |
| 医疗 | 0/57/3 | 0.0% [0.0, 6.0] | 1/38/1 | 2.5% [0.4, 12.9] |
| 教育 | 4/32/24 | 6.7% [2.6, 15.9] | 1/36/3 | 2.5% [0.4, 12.9] |
| 影视 | 4/51/5 | 6.7% [2.6, 15.9] | 4/31/5 | 10.0% [4.0, 23.1] |
| 人物 | 12/46/2 | 20.0% [11.8, 31.8] | 3/23/14 | 7.5% [2.6, 19.9] |

### Removal / restoration breakdown (sample 3a) and S4/S5 by stratum

| sample | n | A | B | C | rate [CI] |
|---|---|---|---|---|---|
| 3a_removed:C1_content_free:top1k | 2 | 2 | 0 | 0 | 100.0% [34.2, 100.0] |
| 3a_removed:S3_card_passage:random1k | 1 | 1 | 0 | 0 | 100.0% [20.7, 100.0] |
| 3a_removed:S4_suggested_chip:random1k | 85 | 81 | 3 | 1 | 95.3% [88.5, 98.2] |
| 3a_removed:S4_suggested_chip:top1k | 42 | 42 | 0 | 0 | 100.0% [91.6, 100.0] |
| 3a_removed:S5_headline:top1k | 20 | 19 | 1 | 0 | 95.0% [76.4, 99.1] |
| 3b_S4:random1k | 49 | 46 | 3 | 0 | 93.9% [83.5, 97.9] |
| 3b_S4:top1k | 51 | 50 | 0 | 1 | 98.0% [89.7, 99.7] |
| 3c_S5:top1k | 100 | 90 | 10 | 0 | 90.0% [82.6, 94.5] |
| 4b_search_nonuser:S5_headline | 22 | 15 | 6 | 1 | 68.2% [47.3, 83.6] |

## 3. Every miss and every wrong move

### 3.1 Misses — kept `user` rows labelled A (samples 1, 2, 4a; n = 31)

`why` is the criterion the label used. Sample 4a (search) had no misses.

| string | domain | category | snapshot | pv | tier | why | u_ds |
|---|---|---|---|---|---|---|---|
| 你妈 | 教育 | 教育培训 | top1k | 727 | user | content_free | U13 |
| 什么 | 教育 | 教育培训 | top1k | 537 | user | content_free | U13 |
| 随便 | 教育 | 教育培训 | top1k | 326 | user | content_free | U13 |
| AI做爆款视频 | 影视 | 影视动漫 | top1k | 186 | user | feature | U13 |
| #爆款短剧火爆短剧来袭好看又上头# | 影视 | 影视动漫 | top1k | 262 | user | feed_copy | U13 |
| 孙宇晨 我的女友景甜 | 人物 | 人物 | top1k | 1150 | user | headline | U05 |
| 包贝尔疑似出轨 | 人物 | 人物 | top1k | 355 | user | headline | U05 |
| 景甜一夜涨粉3.1万 | 人物 | 人物 | top1k | 344 | user | headline | U05 |
| 覃伟中辞去深圳市市长职务 | 人物 | 人物 | top1k | 324 | user | headline | U05 |
| 宋丹丹菜市场被偶遇 | 人物 | 人物 | top1k | 320 | user | headline | U05 |
| 景甜早期颜值好美 | 人物 | 人物 | top1k | 274 | user | headline | U13 |
| 黄晓明陈冠希同框状态差距大 | 人物 | 人物 | top1k | 246 | user | headline | U05 |
| 孙俪杨幂提名宴友好互动 | 人物 | 人物 | top1k | 172 | user | headline | U05 |
| AI短剧《非妖哉》导演：1集成本1万 | 影视 | 影视动漫 | top1k | 166 | user | headline | U04 |
| 景甜财务曾为其扛事进去 | 人物 | 人物 | top1k | 134 | user | headline | U12 |
| 导演称吴镇宇转型老好人律师 | 人物 | 人物 | top1k | 124 | user | headline | U05 |
| 梅母去世梅艳芳遗产全捐 | 人物 | 人物 | top1k | 122 | user | headline | U05 |
| 歼轰7总师陈一坚逝世 | 人物 | 人物 | top1k | 117 | user | headline | U05 |
| 花开不设限拟邀杨幂闫妮 | 影视 | 影视动漫 | top1k | 108 | user | headline | U05 |
| 英伟达即将发布业绩 | 金融 | 金融 | top1k | 73 | user | headline | U05 |
| 我想对作文《《西游记》读后感》进行个性化编辑 | 教育 | 教育培训 | top1k | 338 | user | template | U10 |
| 符合哺乳纲核心特征包括：胎生（胚胎在母体子宫内通过胎盘发育）；乳腺哺育（雌性具功能性乳腺，产后分…(+5) | 教育 | 教育培训 | random1k | 1 | user | card | U04 |
| [bio-card passage on a named local civil servant + templated question] | 人物 | 人物 | random1k | 1 | user | card | U03 |
| 有没有更多日本后宫动漫推荐 | 影视 | 影视动漫 | random1k | 1 | user | chip | U08 |
| 马九斤和格格在剧中有哪些感人瞬间 | 影视 | 影视动漫 | random1k | 1 | user | chip | U08 |
| 第50集里小蓝苏醒时有哪些感人瞬间 | 影视 | 影视动漫 | random1k | 1 | user | chip | U08 |
| [named individual]是否公开过个人履历 | 人物 | 人物 | random1k | 1 | user | chip | U05 |
| [named individual]的出生年月有官方报道吗 | 人物 | 人物 | random1k | 1 | user | chip | U05 |
| 神经病，脑残一言不合的智障 | 医疗 | 医疗 | random1k | 1 | user | content_free | U11 |
| 日元三年升值100%？广场协议后汇率变化全解析 | 金融 | 金融 | random1k | 1 | user | headline | U04 |
| 请帮我生成一个视频：1558558，比例为1:1。 | 影视 | 影视动漫 | random1k | 1 | user | template | U10 |

### 3.2 Wrong removals — v2 user → v3 non-user rows NOT labelled A (n = 5)

| string | domain | category | snapshot | pv | tier | lab | why | u_ds |
|---|---|---|---|---|---|---|---|---|
| 叶建春落马后江西省长由谁接任 | 人物 | 人物 | top1k | 188 | S5_headline | B | user | U05 |
| 能否增加一些怪盗基德半夜偷跑出偷宝石，还受伤的过程吗？ | 影视 | 影视动漫 | random1k | 1 | S4_suggested_chip | B | user | U10 |
| 能否把第三道题，第四道题，第五道题短到20个字 | 教育 | 教育培训 | random1k | 1 | S4_suggested_chip | B | user | U11 |
| 及管理部门在对某企业进行安全检查中发现，该企业重大危险源未登记建档，责令限期整改后逾期仍未改正。…(+20) | 教育 | 教育培训 | random1k | 1 | S4_suggested_chip | B | user | U09 |
| 能否用这三个词造个句子 | 教育 | 教育培训 | random1k | 1 | S4_suggested_chip | C | origin_equal | U10 |

### 3.3 Wrong restorations — v2 non-user → v3 user rows NOT labelled B (n = 0)

None. Only 3 fresh restored rows exist in the 5 domains (5 of the 8 restorations were first-audit strings); all 3 are B.

### 3.4 S4_suggested_chip sample rows not labelled A (n = 4)

| string | domain | category | snapshot | pv | tier | lab | why | u_ds |
|---|---|---|---|---|---|---|---|---|
| 能否对比下南通中公和华图 | — | 软件 | random1k | 1 | S4_suggested_chip | B | user | — |
| 可否配十安充电器 | — | 交通出行 | random1k | 1 | S4_suggested_chip | B | user | — |
| 能否把题目文字发给我 | — | 书籍文档 | random1k | 1 | S4_suggested_chip | B | user | — |
| 能否用大白话解释天翼账号隐私政策 | — | 通信 | top1k | 8 | S4_suggested_chip | C | origin_equal | — |

### 3.5 S5_headline sample rows not labelled A (n = 10)

| string | domain | category | snapshot | pv | tier | lab | why | u_ds |
|---|---|---|---|---|---|---|---|---|
| 台风“沙德尔”的最新路径图 | — | 政务 | top1k | 878 | S5_headline | B | user | — |
| 四川8633坠机事故 | — | 新闻 | top1k | 864 | S5_headline | B | user | — |
| 西藏吉隆县泥石流救援进展 | — | 新闻 | top1k | 375 | S5_headline | B | user | — |
| 油价调整最新消息 | — | 新闻 | top1k | 323 | S5_headline | B | user | — |
| 吉隆口岸受灾前后对比 | — | 新闻 | top1k | 247 | S5_headline | B | user | — |
| 普京登岛事件后续最新消息 | — | 新闻 | top1k | 170 | S5_headline | B | user | — |
| 工龄工资调整最新通知 | — | 新闻 | top1k | 103 | S5_headline | B | user | — |
| A 股印花税最新调整政策 | — | 新闻 | top1k | 103 | S5_headline | B | user | — |
| 大连英博北京国安冲突后续 | — | 新闻 | top1k | 102 | S5_headline | B | user | — |
| 大连车主被锁车最新进展 | — | 新闻 | top1k | 98 | S5_headline | B | user | — |

### 3.6 Search non-user rows not labelled A (n = 7)

| string | domain | category | snapshot | pv | tier | lab | why | u_ds |
|---|---|---|---|---|---|---|---|---|
| 召唤魅魔结果是义魔母来了 | 影视 | 影视 | search2026 | 1169 | S5_headline | B | user | — |
| 冯小刚对韩红言论有何回应 | 人物 | 人物 | search2026 | 608 | S5_headline | B | user | — |
| 妹爷去世 | 人物 | 人物 | search2026 | 322 | S5_headline | B | user | — |
| “七一勋章”获得者 | 人物 | 人物 | search2026 | 247 | S5_headline | B | user | — |
| “今日金价” | 金融 | 金融 | search2026 | 206 | S5_headline | B | user | — |
| 江泽民去世 | 人物 | 人物 | search2026 | 187 | S5_headline | B | user | — |
| 去世 | 人物 | 人物 | search2026 | 242 | S5_headline | C | fragment | — |

### 3.7 Ambiguous (C) kept rows (samples 1, 2; n = 59)

Not errors, listed because they bound what any rule can decide. Names of non-public individuals are replaced.

| string | domain | category | snapshot | pv | tier | why | u_ds |
|---|---|---|---|---|---|---|---|
| 少 | 教育 | 教育培训 | top1k | 2028 | user | fragment | U13 |
| 的 | 教育 | 教育培训 | top1k | 1874 | user | fragment | U13 |
| 3 | 教育 | 教育培训 | top1k | 1499 | user | fragment | U13 |
| 走 | 教育 | 教育培训 | top1k | 1296 | user | fragment | U13 |
| 小 | 教育 | 教育培训 | top1k | 1265 | user | fragment | U13 |
| 错 | 教育 | 教育培训 | top1k | 1249 | user | fragment | U13 |
| 第二题 | 教育 | 教育培训 | top1k | 1029 | user | fragment | U13 |
| 上 | 教育 | 教育培训 | top1k | 859 | user | fragment | U13 |
| 年代 | 教育 | 教育培训 | top1k | 827 | user | fragment | U02 |
| 开 | 教育 | 教育培训 | top1k | 783 | user | fragment | U13 |
| 五年级 | 教育 | 教育培训 | top1k | 636 | user | fragment | U02 |
| 5 | 教育 | 教育培训 | top1k | 567 | user | fragment | U13 |
| 女 | 教育 | 教育培训 | top1k | 524 | user | fragment | U13 |
| 七 | 教育 | 教育培训 | top1k | 509 | user | fragment | U13 |
| 把 | 教育 | 教育培训 | top1k | 406 | user | fragment | U13 |
| 六年级 | 教育 | 教育培训 | top1k | 388 | user | fragment | U13 |
| 坐下 | 教育 | 教育培训 | top1k | 388 | user | fragment | U13 |
| 需 | 教育 | 教育培训 | top1k | 343 | user | fragment | U13 |
| v | 教育 | 教育培训 | top1k | 314 | user | fragment | U13 |
| 站着 | 教育 | 教育培训 | top1k | 312 | user | fragment | U13 |
| 后背 | 医疗 | 医疗 | top1k | 249 | user | fragment | U02 |
| 初中生 | 人物 | 人物 | top1k | 245 | user | fragment | U02 |
| 道歉 | 影视 | 影视动漫 | top1k | 143 | user | fragment | U13 |
| 领导班子 | 人物 | 人物 | top1k | 120 | user | fragment | U02 |
| 好看的 | 影视 | 影视动漫 | top1k | 104 | user | fragment | U13 |
| 片子 | 影视 | 影视动漫 | top1k | 103 | user | fragment | U02 |
| 使用更高级的表达 | 教育 | 教育培训 | top1k | 1123 | user | origin_equal | U11 |
| 退出 | 教育 | 教育培训 | top1k | 661 | user | origin_equal | U11 |
| 帮我画 教师节手抄报 | 教育 | 教育培训 | top1k | 441 | user | origin_equal | U10 |
| 这个谜语还有其他可能的答案吗 | 教育 | 教育培训 | top1k | 338 | user | origin_equal | U11 |
| 《蝉》的剧情简介是什么 | 影视 | 影视动漫 | top1k | 199 | user | origin_equal | U04 |
| 《我们的少年时代2》有哪些亮点 | 影视 | 影视动漫 | top1k | 199 | user | origin_equal | U04 |
| 这个处方有副作用吗 | 医疗 | 医疗 | top1k | 129 | user | origin_equal | U07 |
| 这个处方适合哪些人群 | 医疗 | 医疗 | top1k | 122 | user | origin_equal | U13 |
| 衣全白 | 教育 | 教育培训 | random1k | 3 | user | fragment | U02 |
| 晚上8点左右 | 教育 | 教育培训 | random1k | 2 | user | fragment | U13 |
| 这个小脚丫 | 人物 | 人物 | random1k | 1 | user | fragment | U13 |
| 美女男 | 人物 | 人物 | random1k | 1 | user | fragment | U13 |
| arc | 人物 | 人物 | random1k | 1 | user | fragment | U02 |
| [named official]离开安庆市的原因是什么 | 人物 | 人物 | random1k | 2 | user | origin_equal | U05 |
| 金控财智中心是否有可能改造为感知中心 | 金融 | 金融 | random1k | 1 | user | origin_equal | U07 |
| 授权代办方式是否更便捷 | 金融 | 金融 | random1k | 1 | user | origin_equal | U13 |
| [named doctor]在山东省心内科的地位如何 | 医疗 | 医疗 | random1k | 1 | user | origin_equal | U04 |
| 山西华严寺还有哪些著名诗人题咏 | 教育 | 教育培训 | random1k | 1 | user | origin_equal | U08 |
| 武动乾坤第七季有哪些看点 | 影视 | 影视动漫 | random1k | 1 | user | origin_equal | U08 |
| 《赤霞雕流年》的豆瓣评分是多少 | 影视 | 影视动漫 | random1k | 1 | user | origin_equal | U03 |
| 苗侨伟是否记得最难忘的穆念慈戏份 | 影视 | 影视动漫 | random1k | 1 | user | origin_equal | U03 |
| 第24集里许小成受伤严重吗 | 影视 | 影视动漫 | random1k | 1 | user | origin_equal | U03 |
| 如果队员强行检查手镯会怎样 | 影视 | 影视动漫 | random1k | 1 | user | origin_equal | U13 |
| [two named officials]分管哪些具体工作 | 人物 | 人物 | random1k | 1 | user | origin_equal | U03 |
| [named individual]的教育背景是什么 | 人物 | 人物 | random1k | 1 | user | origin_equal | U03 |
| [named individual]在山西交通企业协会任职多久了 | 人物 | 人物 | random1k | 1 | user | origin_equal | U03 |
| [named individual]现在在哪里任职 | 人物 | 人物 | random1k | 1 | user | origin_equal | U05 |
| [named officer]离休后有没有被追授更高军衔 | 人物 | 人物 | random1k | 1 | user | origin_equal | U05 |
| [named businessman]在郑州还有哪些关联企业 | 人物 | 人物 | random1k | 1 | user | origin_equal | U08 |
| 曾国藩的三甲进士身份有何影响 | 人物 | 人物 | random1k | 1 | user | origin_equal | U04 |
| 能提供更多2026年中国最长寿的10位名人信息吗 | 人物 | 人物 | random1k | 1 | user | origin_equal | U08 |
| [named individual]毕业后最初在哪个机构任职 | 人物 | 人物 | random1k | 1 | user | origin_equal | U03 |
| 陕西铁塔领导班子是否有新成员加入 | 人物 | 人物 | random1k | 1 | user | origin_equal | U05 |

## 4. Bias check against the single-instrument intent labels (`u_ds`)

### 4.1 Which classes the misses carry

| u_ds | name | 1_head_kept | 2_tail_kept |
|---|---|---|---|
| U03 | 事实与数值 | 0 | 1 |
| U04 | 解释与介绍 | 1 | 2 |
| U05 | 核实与动态 | 12 | 2 |
| U08 | 清单与推荐 | 0 | 3 |
| U10 | 生成与编辑 | 1 | 1 |
| U11 | 会话与系统指令 | 0 | 1 |
| U12 | 违规或灰色内容 | 1 | 0 |
| U13 | 无法判定 | 6 | 0 |

In the head, 12 of 21 misses are **U05 核实与动态** — tapped headlines, 12 of them in 人物. Tail misses spread over U03/U04/U05/U08/U10/U11 (cards, LLM-voice follow-ups, a template).

### 4.2 Maximum distortion of each class's row share — head (5 domains top1k, kept rows)

Method: `q_k` = share of kept rows that are A and in class k (from sample 1), `r` = Σ q_k. Share after removing A rows = (p_k − q_k)/(1 − r). **Max inflation** uses the Wilson upper bound of q_k (capped at p_k, no other removals); **max deflation** uses the Wilson lower bound of q_k with the Wilson-upper total r. Row U05 is replaced by the census (§4.4), whose bound is exact up to the total-miss uncertainty.

| u_ds | name | kept share | sample rows in class | A in class | within-class A [CI] | share after removal | shift pp | max inflation pp | max deflation pp |
|---|---|---|---|---|---|---|---|---|---|
| U01 | 导航直达 | 2.91 | 8 | 0 | 0% [0, 32] | 3.12 | -0.22 | 1.24 | 0.34 |
| U02 | 裸实体 | 30.34 | 85 | 0 | 0% [0, 4] | 32.62 | -2.28 | 0.89 | 3.55 |
| U03 | 事实与数值 | 9.52 | 29 | 0 | 0% [0, 12] | 10.24 | -0.72 | 1.16 | 1.11 |
| U04 | 解释与介绍 | 11.97 | 32 | 1 | 3% [1, 16] | 12.51 | -0.54 | 1.67 | 1.39 |
| U05 | 核实与动态 | 3.95 | 19 | 12 | census 52% [44, 59] | 2.05 | 1.89 | 1.95 | 0.36 |
| U06 | 办事与操作 | 2.82 | 6 | 0 | 0% [0, 39] | 3.03 | -0.21 | 1.24 | 0.33 |
| U07 | 个案判断与建议 | 3.45 | 6 | 0 | 0% [0, 39] | 3.71 | -0.26 | 1.24 | 0.4 |
| U08 | 清单与推荐 | 5.4 | 16 | 0 | 0% [0, 19] | 5.81 | -0.41 | 1.21 | 0.63 |
| U09 | 获取现成内容 | 8.24 | 23 | 0 | 0% [0, 14] | 8.86 | -0.62 | 1.18 | 0.96 |
| U10 | 生成与编辑 | 2.17 | 6 | 1 | 17% [3, 56] | 1.97 | 0.2 | 1.86 | 0.25 |
| U11 | 会话与系统指令 | 2.82 | 8 | 0 | 0% [0, 32] | 3.03 | -0.21 | 1.24 | 0.33 |
| U12 | 违规或灰色内容 | 0.59 | 2 | 1 | 50% [9, 91] | 0.27 | 0.31 | 0.59 | 0.07 |
| U13 | 无法判定 | 15.83 | 60 | 6 | 10% [5, 20] | 14.87 | 0.96 | 3.78 | 1.69 |

Total A in kept head rows: 7.0% (Wilson upper 10.5%).

### 4.3 Maximum distortion — tail (5 domains random1k, kept rows)

| u_ds | name | kept share | sample rows in class | A in class | within-class A [CI] | share after removal | shift pp | max inflation pp | max deflation pp |
|---|---|---|---|---|---|---|---|---|---|
| U01 | 导航直达 | 0.41 | 1 | 0 | 0% [0, 79] | 0.43 | -0.02 | 0.41 | 0.04 |
| U02 | 裸实体 | 5.2 | 14 | 0 | 0% [0, 22] | 5.47 | -0.27 | 1.82 | 0.51 |
| U03 | 事实与数值 | 14.9 | 33 | 1 | 3% [1, 15] | 15.15 | -0.26 | 2.43 | 1.45 |
| U04 | 解释与介绍 | 14.88 | 26 | 2 | 8% [2, 24] | 14.61 | 0.27 | 3.15 | 1.42 |
| U05 | 核实与动态 | 8.64 | 22 | 2 | 9% [3, 28] | 8.04 | 0.6 | 3.38 | 0.82 |
| U06 | 办事与操作 | 4.69 | 14 | 0 | 0% [0, 22] | 4.93 | -0.25 | 1.83 | 0.46 |
| U07 | 个案判断与建议 | 14.1 | 19 | 0 | 0% [0, 17] | 14.84 | -0.74 | 1.65 | 1.39 |
| U08 | 清单与推荐 | 9.52 | 23 | 3 | 13% [5, 32] | 8.44 | 1.08 | 4.08 | 0.88 |
| U09 | 获取现成内容 | 6.36 | 8 | 0 | 0% [0, 32] | 6.7 | -0.33 | 1.8 | 0.63 |
| U10 | 生成与编辑 | 5.87 | 14 | 1 | 7% [1, 31] | 5.66 | 0.22 | 2.69 | 0.57 |
| U11 | 会话与系统指令 | 2.41 | 3 | 1 | 33% [6, 79] | 2.02 | 0.4 | 2.41 | 0.24 |
| U12 | 违规或灰色内容 | 1.08 | 3 | 0 | 0% [0, 56] | 1.14 | -0.06 | 1.08 | 0.11 |
| U13 | 无法判定 | 11.95 | 20 | 0 | 0% [0, 16] | 12.58 | -0.63 | 1.69 | 1.18 |

Total A in kept tail rows: 5.0% (Wilson upper 9.0%). n = 200, so every class bound is at least ~1.7 pp whatever the data show.

### 4.4 U05 census — every kept 5-domain head row labelled U05

| domain | kept head rows | U05 rows | A | B | C | A share of U05 | U05 share (v3) | U05 share after A removed |
|---|---|---|---|---|---|---|---|---|
| 金融 | 950 | 50 | 9 | 36 | 5 | 18.0% | 5.26% | 4.39% |
| 医疗 | 958 | 15 | 0 | 15 | 0 | 0.0% | 1.57% | 1.57% |
| 教育 | 859 | 2 | 1 | 1 | 0 | 50.0% | 0.23% | 0.12% |
| 影视 | 955 | 17 | 12 | 5 | 0 | 70.6% | 1.78% | 0.56% |
| 人物 | 889 | 98 | 72 | 21 | 5 | 73.5% | 11.02% | 3.66% |
| 5 domains | 4611 | 182 | 94 | 78 | 10 | 51.6% [44.4, 58.8] | 3.95% | 2.05% [2.00, 2.13] |

94 of 182 U05 head rows (19,291 of 41,621 PV) are system text — almost all tapped headlines. Removing them takes the pooled U05 head share from 3.95% to 2.05% (1.82% if the 10 C rows are also dropped): **v3 still roughly doubles U05 in the head**, and triples it in 人物. The first audit's v2 finding (fact-verification roughly doubled) therefore still holds in the 5-domain head after S4/S5. Highest-PV examples: `孙宇晨 我的女友景甜` (1150) · `杨幂孙俪获年度女演员` (704) · `[headline about a school's hair rule for students]` (662) · `国防部评《龙餐馆》` (637) · `鲁迅长孙周令飞出席文学周` (554) · `赵丽颖突发身体不适` (406) · `景甜富豪男友疑似孙宇晨` (371) · `马伊琍杨紫首度同台` (370) · `包贝尔疑似出轨` (355) · `朱忠明任上海市政府党组书记` (351).

### 4.5 Ambiguous (C) rows — share and classes

C share: head 11.3% (Wilson upper 15.4%), tail 12.5% (upper 17.8%).

| u_ds | name | 1_head_kept | 2_tail_kept |
|---|---|---|---|
| U02 | 裸实体 | 6 | 2 |
| U03 | 事实与数值 | 0 | 7 |
| U04 | 解释与介绍 | 2 | 2 |
| U05 | 核实与动态 | 0 | 4 |
| U07 | 个案判断与建议 | 1 | 1 |
| U08 | 清单与推荐 | 0 | 4 |
| U10 | 生成与编辑 | 1 | 0 |
| U11 | 会话与系统指令 | 3 | 0 |
| U13 | 无法判定 | 21 | 5 |

Head C rows are mostly U13 (21 of 34), as expected: bare education fragments (`少`, `的`, `3`, `第二题`) with 300–2,000 PV. If all head C rows were dropped, U13 would fall by 5.9 pp and U02 rise by 1.6 pp. Tail C rows are NOT mostly U13 (5 of 25); they are formal follow-up questions about named local officials and works (U03 7, U05 4, U08 4) whose origin — typed or a suggested question — the text cannot decide; dropping them would move U03 by 1.9 pp.

Single-character kept head rows (a measured count; kept by design, and a lookup vs a voice/OCR fragment cannot be told apart):

| domain | kept head rows | single-character rows | of which U13 | U13 share | U13 share without them |
|---|---|---|---|---|---|
| 金融 | 950 | 3 | 1 | 4.6% | 4.5% |
| 医疗 | 958 | 4 | 1 | 12.0% | 11.9% |
| 教育 | 859 | 115 | 102 | 41.6% | 34.3% |
| 影视 | 955 | 4 | 3 | 13.3% | 13.0% |
| 人物 | 889 | 3 | 1 | 9.8% | 9.7% |
| 5 domains | 4611 | 129 | 108 | 15.8% | 13.9% |

## 5. Systematic patterns and proposed rules (NOT applied)

Each rule was written from round-1 strings only, checked for false positives on round-1 labels, frozen in `audit2_build_round2.py`, and then scored on round-2 rows that no round-1 or first-audit label had touched.

| # | pattern | measured count | proposed rule | holdout precision | verdict |
|---|---|---|---|---|---|
| 1 | Tapped headlines outside the 新闻 category (人物/影视/金融 head); S5 rule B's verb list and PV>=100 miss them | 16 of 31 sample misses; census: 94 of 182 U05 head rows | R1: head, not 新闻, no ask word, PV>=100, len>=6, contains 逝世/去世/离世/疑似/偶遇/同框/拟邀/辞去/涨粉/自曝/发声/现身/互动/热议/官宣/离婚/出轨/夺冠/当选/即将/首次/回应/表态/被查/落马/获刑/身亡/否认/曝光/爆料. R1b: identical string is also a 新闻 top1k row, no NEWS_ASKS word | R1 15/22 = 68.2% [47.3, 83.6]; R1b 28/40 = 70.0% [54.6, 81.9]; R1∪R1b inside the U05 census holdout 93.8% [79.9, 98.3], recall 38.0% [28.1, 49.0] | Do not apply. False positives: explicit-content edit commands matching 互动/涨粉/离婚, weather and pension keyword queries cross-listed in 新闻, bare celebrity names. Precise only where the intent label already says U05, which would make cleaning depend on the quantity being compared. |
| 2 | `有没有更多…` suggested chips (CHIP only covers `有没有更…的`) | 14 kept rows, 33 cats (1 in 5-domain tail) | R2a: `^有没有更多` | 12/13 = 92.3% [66.7, 98.6] | Apply (tiny effect on the 5 domains). One typed false positive with a `+` keyword join. |
| 3 | LLM-voice follow-ups without a chip prefix (`有哪些感人瞬间`, `是否公开过`, `有官方报道吗`) | 4 tail misses; 5 kept rows | R2b: `有哪些感人瞬间$\|是否公开过\|有官方报道吗[？?]?$` | 1/1 = 100.0% [20.7, 100.0] | Unproven: 1 holdout row. The family is open-ended; 20 kept tail sample rows in the same formal register were labelled C, not A. |
| 4 | Bare replies and insults not in REPLY (`什么`, `随便`, `你妈`) | 3 head misses; 9 kept rows | R3: `^\s*(什么\|啥\|随便\|都行\|无所谓\|你妈的?)[\s。！!~～？?]*$` | 6/6 = 100.0% [61.0, 100.0] | Apply. Assistant surface only (on search a bare word is a lookup). |
| 5 | Feed hashtag copy (`#爆款短剧…#`, `#百家流量扶持计划#`) | 1 head miss; 42 kept rows (6 in 5-domain head) | R4: `^\s*#[^#]{2,}#\s*$` | 30/30 = 100.0% [88.6, 100.0] | Apply. |
| 6 | Essay-edit slot template (`我想对作文《…》进行个性化编辑`) — TEMPLATE_MARK knows it but only uses it to veto an exemption | 1 head miss; 15 kept rows, all 教育 (10 head, 5 tail) | R6: `^我想对作文《` | 14/14 = 100.0% [78.5, 100.0] | Apply. |
| 7 | Pasted answer text / bio cards without `男/女` (U+200C marks text copied from an answer; `名，现任…`) | 2 tail misses; 82 kept rows | R5: `^[一-鿿]{2,4}[，,]现任` or contains U+200C | 7/40 = 17.5% [8.7, 32.0]; U+200C part 6/39 | Reject U+200C: it marks copying, and users append their own request or paste their own prompts. The `现任` part has 3 kept rows in total — not worth a rule. |
| 8 | S5 rule A (新闻 category, no NEWS_ASKS) removes keyword news queries | 10 of 100 S5 sample rows; all end in a keyword suffix | R7 exemption: `(最新消息\|最新进展\|最新通知\|最新政策\|调整政策\|后续\|进展\|路径图\|事故\|前后对比)$` | exempted rows labelled B: 30/30 = 100.0% [88.6, 100.0] | Apply. Affects 40 S5 rows, none in the 5 domains. Caveat: rests on the keyword-vs-headline judgement in §1. |
| 9 | Search S5 removes verification lookups (`江泽民去世`), quoted keywords (`“今日金价”`) and questions (`…有何回应`) | 6 B + 1 C of 22 fresh search S5 rows | Exempt `^“…”$`, `^[一-鿿]{2,4}(去世\|逝世)$`, and `有何\|何时\|为何\|是谁\|咋` | 2025 snapshot replay (n = 21): S5 as-is 71.4% [50.0, 86.2]; exempted 4 rows, B 75.0% [30.1, 95.4]; S5 after exemption 88.2% [65.7, 96.7] | Apply. The remaining false positives are titles that contain 曝光/来了. Volume is 22 rows of 50k. |
| 10 | S4 removes typed `能否…吗？`, comma-listed requests and a pasted exam question | 3 of 5 wrong removals in 3a | Exempt `吗[？?]?$`, two commas, or length > 40 | 3b rows not in 3a: exempted 1, labelled A (0/1); the 3 B rows in the holdout were not exempted | Reject: does not generalise. S4 is already 96% precise. |
| 11 | CHIP's unanchored `该(公司\|企业\|平台…)` fires inside long pasted company/app descriptions | 12 S4 rows are caught only by this branch (8 longer than 30 chars) | Anchor it (`^该…`) and let S3 take the passages | not measured (no fresh labelled rows) | Tier hygiene only: the passages are A either way; the few typed `如何查询该公司…` rows are unlabelled. |
| 12 | Education head single-character fragments (C, not A) | 115 kept 教育 head rows (102 carry U13) | None for removal; report U13 with and without them | n/a | U13 in 教育 head 41.6% → 34.3%; pooled 15.8% → 13.9%. |

False positives of R1 / R1b / R2a on the holdout (R5's 33 are summarised above): `[explicit-content rewrite command]` (R1) · `[explicit-content continuation command]` (R1) · `[explicit image-generation prompt]` (R1) · `[explicit-content rewrite command]` (R1) · `[request for sentences about a child starting school]` (R1) · `离婚律师咨询电话` (R1) · `女人涨粉最快的网名` (R1) · `退休金调整政策最新消息` (R1b) · `台风最新消息台风路径` (R1b) · `西藏泥石流现场` (R1b) · `[a bare personal name]` (R1b) · `王菲` (R1b) · `最新台风实时路径消息` (R1b) · `台风最新路径` (R1b) · `宁波交通事故` (R1b) · `养老金调整最新消息` (R1b) · `中雨大雨特大暴雨` (R1b) · `女排 朱婷` (R1b) · `租客看房东病情后续` (R1b) · `有没有更多幸运昵称+女三个字` (R2a).

## 6. Is v3 fit for comparing unified-intent shares between surfaces?

Pooled over the 5 domains; kept rows only; shares from `sva_final_rows.parquet`. **Threshold** = worst-case cleaning bias on the assistant side (head: §4.2 bound, U05 from the census; tail: §4.3 bound plus 1.96·SE because random1k is a sample) + worst-case bias on the search side (sample 4a found 0 misses in 150: the bound is up to ~2.5 pp, capped at the class's own share because a class cannot lose more rows than it has). **readable** = yes when the difference exceeds the threshold by more than 0.5 pp, marginal when it exceeds it by less. The bounds are conservative, and they ignore labelling error in `u_ds` itself, which this audit did not measure. For U05 read the `head, A removed` column, not the raw head share.

| u_ds | name | search % | asst head % | head, A removed % | head − search pp | head threshold pp | head readable | asst tail % | tail − search pp | tail threshold pp | tail readable |
|---|---|---|---|---|---|---|---|---|---|---|---|
| U01 | 导航直达 | 6.4 | 2.9 | 3.1 | -3.5 | 3.6 | no | 0.4 | -6.0 | 3.0 | yes |
| U02 | 裸实体 | 44.5 | 30.3 | 32.6 | -14.1 | 5.0 | yes | 5.2 | -39.3 | 3.9 | yes |
| U03 | 事实与数值 | 11.0 | 9.5 | 10.2 | -1.5 | 3.4 | no | 14.9 | +3.9 | 5.7 | no |
| U04 | 解释与介绍 | 23.2 | 12.0 | 12.5 | -11.3 | 3.6 | yes | 14.9 | -8.4 | 6.1 | yes |
| U05 | 核实与动态 | 0.2 | 3.9 | 2.1 | +3.7 | 2.2 | yes | 8.6 | +8.4 | 4.4 | yes |
| U06 | 办事与操作 | 0.9 | 2.8 | 3.0 | +1.9 | 2.2 | no | 4.7 | +3.8 | 3.3 | marginal |
| U07 | 个案判断与建议 | 1.2 | 3.4 | 3.7 | +2.3 | 2.4 | no | 14.1 | +12.9 | 3.8 | yes |
| U08 | 清单与推荐 | 2.0 | 5.4 | 5.8 | +3.4 | 3.3 | marginal | 9.5 | +7.5 | 6.9 | yes |
| U09 | 获取现成内容 | 8.8 | 8.2 | 8.9 | -0.6 | 3.5 | no | 6.4 | -2.5 | 4.8 | no |
| U10 | 生成与编辑 | 0.1 | 2.2 | 2.0 | +2.1 | 1.9 | marginal | 5.9 | +5.8 | 3.4 | yes |
| U11 | 会话与系统指令 | 0.0 | 2.8 | 3.0 | +2.8 | 3.7 | no | 2.4 | +2.4 | 5.3 | no |
| U12 | 违规或灰色内容 | 0.8 | 0.6 | 0.3 | -0.2 | 1.3 | no | 1.1 | +0.3 | 2.1 | no |
| U13 | 无法判定 | 0.9 | 15.8 | 14.9 | +15.0 | 4.6 | yes | 12.0 | +11.1 | 3.5 | yes |

Across classes holding at least 2% on some surface (U13 excluded), the largest head threshold is **5.0 pp** (U02) and the largest tail threshold **6.9 pp** (U08). U13's own threshold is 4.6 pp (head).

## 7. Bottom line

- **Removal is sound.** New removals are 96.7% precise, S4 96%, S5 90% (its errors are keyword news queries outside the 5 domains). Search S5 is the weak rule (68% on 22 rows; 71% replayed on the 2025 snapshot) but touches 22 of 50k rows.
- **Recall is adequate in aggregate; 人物 head is the exception, and its misses are mostly one class** (10 of 12 are U05). Kept head rows are 7.0% [4.6, 10.5] system text (5.0% tail; 0 of 150 on search). 人物 head is 20% [12, 32]. The census shows 94 of 182 U05 head rows are tapped headlines, so the head U05 share is inflated from 2.1% to 3.9% (人物: 3.7% → 11.0%).
- **Fit for the pooled 5-domain comparison, with three conditions.** (1) As a blanket rule, do not read a head difference smaller than **5 pp** or a tail difference smaller than **7 pp**; §6 gives smaller class-specific thresholds for small classes, which cannot be inflated by more than their own share. (2) Read head U05 only after subtracting the census rows: search 0.2% vs head 2.1%, not 3.9%. The corrected gap is 1.8 pp with about 0.4 pp of remaining cleaning uncertainty, so its direction reads, but it is half the raw gap. (3) Exclude U13 (head 15.8%). It is inflated by 教育, where U13 is 41.6% of kept head rows and 102 of its 357 U13 rows are single characters. On that basis the large gaps survive: bare entity U02 search 44% vs head 30%, explanation U04 23% vs 12%, and case advice U07 1% vs tail 14%.
- **Not fit per domain at n = 60/40.** The per-domain Wilson upper bound on the miss rate ranges 6–32%. 人物 head is not fit per class at all: 20% of its kept rows are system text, and the census corrects only U05. **The smallest share difference that should be read is 5 percentage points (pooled head) / 7 pp (pooled tail).**
- Rules worth applying next (measured on holdout rows, none applied here): `^有没有更多` chips, bare replies, feed hashtags, the essay-edit template, the 新闻 keyword exemption, and the three search-S5 exemptions. None of them fixes the U05 headline problem. Two headline rules were tried and failed their holdout at ~70%.

## 8. Files

- `/private/tmp/claude-501/-Users-mayouxuan-Documents-Claude-Search-Query-Mining-Agent-Team/3deba04e-73b1-4747-b7b7-4d1a40520650/scratchpad/audit2_report.md` — this report
- `/private/tmp/claude-501/-Users-mayouxuan-Documents-Claude-Search-Query-Mining-Agent-Team/3deba04e-73b1-4747-b7b7-4d1a40520650/scratchpad/audit2_labels.py` — round-1 labels (LAB_ASSISTANT, LAB_SEARCH, REASON_*) + check()
- `/private/tmp/claude-501/-Users-mayouxuan-Documents-Claude-Search-Query-Mining-Agent-Team/3deba04e-73b1-4747-b7b7-4d1a40520650/scratchpad/audit2_labels2.py` — round-2 labels + check()
- `/private/tmp/claude-501/-Users-mayouxuan-Documents-Claude-Search-Query-Mining-Agent-Team/3deba04e-73b1-4747-b7b7-4d1a40520650/scratchpad/audit2_samples.parquet` — round-1 sampled rows (sample, rid, tier, v2_tier, pv…)
- `/private/tmp/claude-501/-Users-mayouxuan-Documents-Claude-Search-Query-Mining-Agent-Team/3deba04e-73b1-4747-b7b7-4d1a40520650/scratchpad/audit2_round2_samples.parquet` — round-2 rows (set = census / rule holdout / replay / decoy)
- `/private/tmp/claude-501/-Users-mayouxuan-Documents-Claude-Search-Query-Mining-Agent-Team/3deba04e-73b1-4747-b7b7-4d1a40520650/scratchpad/audit2_label_list.txt / audit2_round2_list.txt` — the blind lists that were labelled
- `/private/tmp/claude-501/-Users-mayouxuan-Documents-Claude-Search-Query-Mining-Agent-Team/3deba04e-73b1-4747-b7b7-4d1a40520650/scratchpad/audit2_make_labels.py / audit2_make_labels2.py` — id-level labelling record → label dicts
- `/private/tmp/claude-501/-Users-mayouxuan-Documents-Claude-Search-Query-Mining-Agent-Team/3deba04e-73b1-4747-b7b7-4d1a40520650/scratchpad/audit2_build_samples.py / audit2_build_round2.py` — sampling, exclusions, frozen rules
- `/private/tmp/claude-501/-Users-mayouxuan-Documents-Claude-Search-Query-Mining-Agent-Team/3deba04e-73b1-4747-b7b7-4d1a40520650/scratchpad/audit2_analysis.py / audit2_round2_analysis.py / audit2_report_build.py` — metrics and this report
