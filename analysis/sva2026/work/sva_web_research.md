# 外部证据核查：中国用户使用 AI 助手 vs 传统搜索（2025–2026），以及用户是谁

核查日期：2026-09-10

**方法。** 每个数字都要打开原页面或 PDF 核对。关键数字一律做了原文文本检索：用 curl 下载，再用 pypdf 或 HTML 抽取文本后 grep。这一步不依赖 WebFetch 的摘要模型。原因是本次核查中，WebFetch 摘要模型至少两次给出了原文里没有的"引语"或数字，见文末不可引用清单。少数条目只通过 WebFetch 读到、没能拿到原文 grep，均单独注明。Pew 与 Reuters Institute 站点直连会 TLS 失败，所以改从 archive.org 快照取原文。

**状态定义**
- **VERIFIED-PRIMARY**：在发布方原始页面或原始 PDF 上逐字核到。第三方站点镜像的原始 PDF、archive.org 快照也算此类。
- **VERIFIED-SECONDARY**：在媒体、转载页或第三方转录稿上核到，但不是发布方原件。
- **NOT PRESENT ON PAGE**：搜索摘要或 WebFetch 摘要声称有，打开原文后没有。
- **NOT FOUND**：没有找到可信来源。

"派生计算"指我根据同一来源的原文数字自己算出的结果，原文里没有这个数，引用时须注明。

---

## 0. 与"已核实清单"的对照（请先读）

| 已核实清单中的条目 | 本次结果 |
|---|---|
| "QuestMobile: 64.6% of AI users male" 被判 NOT PRESENT | **有矛盾**。QuestMobile 官网《2026年一季度AI应用洞察》(2026-04-21) 页面上确有一句："截至今年3月，男性用户占比达64.6%，同比提升4.5%，用户规模增加1.2亿"。不过这句话放在"用户版图朝'银发+下沉'双向延伸"的小标题下，没有交代分母（推测是 AI 原生 App 用户），语义可疑，而且与 CNNIC 调查口径的 50.3% 相差很大。见 Q1-12。**结论：页面上有，但不建议作为事实引用。** 此前判 NOT PRESENT，很可能是因为当时打开的是另一个页面。 |
| CNNIC 生成式AI用户的年龄/性别/学历分布被判 NOT PRESENT | **有矛盾**。CNNIC 第57次报告官方 PDF（cnnic.cn，第42–45页）完整给出了性别、年龄、学历、职业结构和使用目的。光明网等新闻稿里确实没有这些内容，此前核查的应是新闻稿。CNNIC《生成式人工智能应用发展报告（2025）》PDF 也给出了 2025 年 6 月的结构数据。见 Q1-02 至 Q1-09。 |
| QuestMobile "June 2026: AI native apps averaged 92.7 uses" | **月份标注在官网页面上自相矛盾**。《2026年AI应用市场发展半年报》(2026-07-14) 的导语写"截止到2026年5月……月活分别为4.99亿……月人均次数分别为92.7次"，正文却写"2026年6月，AI原生App整体规模4.99亿"。引用时建议写"2026年5–6月（QuestMobile 页面标注不一致）"。 |
| 豆包 3.82亿 (+172.1%)、DeepSeek 1.3亿 (−20.3%) | 3.82亿、1.67亿和 1.29亿（约 1.3亿）在官网页面上核到，时间是 2026 年 6 月。**+172.1% 和 −20.3% 两个增速在该官网页面上未找到**，可能出自完整报告或微信版，请另行核对来源。 |
| 豆包 72.2%、DeepSeek 62.0% | PDF 确认（CNNIC 2025 报告，2025 年 6 月调查）。个别二手摘要写 DeepSeek "60.0%"，PDF 中没有这个数。 |
| 6.02亿 / 42.8% / +141.7%；5.15亿 / 36.5% | 与官方 PDF 一致。**注意分母**：42.8% 和 36.5% 是"整体人口"中过去半年使用过的比例（PDF 脚注 64、17），不是占网民的比例。 |
| 搜索 −19.1% / −13.5%（2026 年 5 月） | 一致。原文措辞是"传统搜索行业人均使用次数、时长同比分别下降"，属于人均指标，不是总量。 |

---

## Q1. 谁在使用生成式 AI（人口结构、使用目的）

### Q1-01 CNNIC 第57次：生成式 AI 用户规模与普及率（复核）
- **论断**：截至 2025 年 12 月，生成式 AI 用户 6.02 亿，普及率 42.8%。
- **数字**：6.02亿人；较 2024 年底 +141.7%；普及率 42.8%（+25.2 个百分点）。派生计算：6.02 亿 ÷ 网民 11.25 亿 ≈ 53.5%，即略超一半的网民用过。
- **原文引语**：「生成式人工智能用户规模达6.02亿人」
- **发布方**：中国互联网络信息中心（CNNIC）
- **发布日期**：2026-02-05（官网 PDF 上传于 2026-03-04）
- **数据期**：2025 年 12 月
- **口径**：抽样调查（中国互联网络发展状况统计调查）。普及率指过去半年**整体人口**中使用过的比例。
- **URL**：https://cnnic.cn/NMediaFile/2026/0304/MAIN1772588317069TUXN3827X8.pdf （报告页 https://cnnic.cn/n4/2026/0304/c88-11549.html）
- **状态**：VERIFIED-PRIMARY

### Q1-02 性别结构：生成式 AI 用户与网民整体接近
- **论断**：生成式 AI 用户男女比例基本均衡，与网民整体接近。
- **数字**：生成式 AI 用户 男 50.3% : 女 49.7%；网民整体 男 51.2% : 女 48.8%（同一报告）。
- **原文引语**：「男女比例较为均衡，为50.3:49.7」；「网民男女比例为51.2:48.8」
- **发布方 / 日期 / 数据期 / URL**：同 Q1-01
- **口径**：调查；占用户的比例
- **状态**：VERIFIED-PRIMARY

### Q1-03 年龄结构：生成式 AI 用户明显偏年轻（与网民整体对比）
- **论断**：19 岁及以下是生成式 AI 用户中占比最大的年龄段。40 岁以上用户占比在半年内上升。与网民整体相比，AI 用户明显偏年轻。
- **数字（生成式 AI 用户，2025.12）**：19岁及以下 26.4%；20–29岁 21.3%；30–39岁 22.0%；40–49岁 15.1%；50–59岁 10.1%；60岁及以上 5.1%。40 岁以上合计 30.3%，半年提高 4.9 个百分点。
- **数字（网民整体，2025.12，同一报告图17）**：10岁以下 4.2%（图注文字层为"6-10岁"）；10–19岁 13.9%；20–29岁 12.8%；30–39岁 18.5%；40–49岁 16.1%；50–59岁 19.0%；60岁及以上 15.4%。原文："10-49岁网民占比合计为61.4%；50岁及以上……34.4%"。
- **派生计算**：19岁及以下，AI 用户 26.4% vs 网民 18.1%；20–29岁 21.3% vs 12.8%；50岁及以上 15.2% vs 34.4%。
- **原文引语**：「19岁及以下用户占比最高，达26.4%」
- **发布方 / 日期 / 数据期 / URL**：同 Q1-01（PDF 第18、42–43页）
- **口径**：调查；占用户的比例。分项数值来自 PDF 图表文字层，与正文合计数一致。
- **状态**：VERIFIED-PRIMARY

### Q1-04 学历结构
- **论断**：生成式 AI 用户中，大专及以上占比最高，初中其次。
- **数字**：小学及以下 12.7%；初中 29.9%；高中/中专/技校 18.8%；大专及以上 38.6%。
- **原文引语**：「大专及以上群体，占比达38.6%」
- **对比缺口**：第57次报告全文检索没有找到"网民学历结构"，无法在同一报告内与网民整体对比，标 NOT PRESENT。
- **发布方 / 日期 / 数据期 / URL**：同 Q1-01
- **状态**：VERIFIED-PRIMARY

### Q1-05 职业结构：学生占三成
- **论断**：学生是生成式 AI 用户中占比最大的职业群体。
- **数字**：学生 30.1%；企业/公司管理人员/一般职员 16.2%；个体户/自由职业者 14.2%（以上为正文）。图表另列：退休/无业 14.0%、专业技术人员 7.6%、党政机关事业单位 5.2%、商业服务业职工 3.3%、农林牧渔劳动者 2.3%、农村外出务工人员 1.9%、制造生产型企业 1.3%、其他 3.9%。图表部分按 PDF 文字层顺序对应，引用前建议对照原图。
- **原文引语**：「学生是使用生成式人工智能用户的主要群体」
- **发布方 / 日期 / 数据期 / URL**：同 Q1-01（PDF 第43–44页）
- **状态**：VERIFIED-PRIMARY
- **与本报告的关联**：学生占 30.1%，与教育域 AI 助手查询占比高的现象方向一致。这只能算旁证，不是因果。

### Q1-06 使用目的（2025.12）："回答问题"遥遥领先
- **论断**：回答问题是最广泛的使用目的，其次是图片/视频生成、文本生成、办公材料。
- **数字（多选，占生成式 AI 用户）**：回答问题 76.0%；生成处理图片/视频 47.8%；生成处理文本 37.6%；工作总结/会议纪要/PPT 32.5%（以上为正文）。图表另列：生活助手 30.6%，娱乐 30.1%，生成处理代码 10.8%。
- **原文引语**：「回答问题仍是最广泛的应用场景」
- **口径缺口**：选项里**没有**"智能搜索""学习"这两个独立类目，不能引用这两类的百分比（NOT PRESENT）。
- **发布方 / 日期 / 数据期 / URL**：同 Q1-01（PDF 第45页）
- **状态**：VERIFIED-PRIMARY

### Q1-07 产品形态使用率与月人均使用时长（2025.12）
- **论断**：独立 App 是主要载体，手机厂商内置助手次之。原生 App 的月人均时长远高于插件和厂商助手。
- **数字**：各类产品在用户中的使用率：APP 86.0%；移动设备内置智能助手 41.4%（以上为正文）；网页端 33.7%、客户端 24.7%、移动设备应用内插件 22.8%（图表）。月人均使用时长：原生 APP 143.2 分钟；AI 应用插件 34.7 分钟（同比 +69.6%）；手机厂商 AI 助手 5.7 分钟。
- **原文引语**：「APP在用户中的使用率最高，达86.0%」
- **发布方 / 日期 / 数据期 / URL**：同 Q1-01（PDF 第44–46页）
- **状态**：VERIFIED-PRIMARY

### Q1-08 CNNIC 2025 报告（2025.6）：性别、年龄、学历
- **论断**：2025 年 6 月，生成式 AI 用户女性略多，19 岁及以下占三分之一。
- **数字**：男 47.6 : 女 52.4（"与整体网民的性别结构差异不大"）。年龄：19岁及以下 33.8%；20–29岁 21.0%；30–39岁 19.8%；40岁及以上合计 25.4%（40–49 12.7%、50–59 8.6%、60+ 4.2%）。学历：小学及以下 14.3%；初中 29.8%；高中/中专/技校 18.4%；大专 8.6%；本科及以上 28.9%。
- **派生计算**：到 2025 年 12 月，19 岁及以下占比从 33.8% 降到 26.4%，40 岁及以上从 25.4% 升到 30.3%。后者与第57次报告"半年提高4.9个百分点"的说法一致。
- **原文引语**：「19 岁及以下用户占比最高，达到 33.8%」
- **发布方**：CNNIC《生成式人工智能应用发展报告（2025）》
- **发布日期**：2025 年 10 月（PDF 封面）
- **数据期**：2025 年 6 月
- **口径**：调查；占用户的比例。两份报告的学历分档不同：本报告拆分"大专/本科及以上"，第57次合并为"大专及以上"。
- **URL**：https://pdf.dfcfw.com/pdf/H301_AP202510241768289458_1.pdf （CNNIC 原文 PDF，东方财富站点镜像）
- **状态**：VERIFIED-PRIMARY

### Q1-09 CNNIC 2025 报告（2025.6）：使用目的
- **数字（多选）**：回答问题 80.9%；生成、处理文本 36.0%；生成图片、视频 33.0%（以上为正文）。图表另列：作为生活助手 30.0%、生成会议纪要/PPT 29.7%、休闲娱乐 23.6%、帮助写代码 10.3%。
- **原文引语**：「回答问题的用户最为广泛，达80.9%」
- **发布方 / 日期 / 数据期 / URL**：同 Q1-08
- **状态**：VERIFIED-PRIMARY

### Q1-10 CNNIC 2025 报告（2025.6）：各产品使用率与首选率（含文心一言）
- **数字**：使用率：豆包 72.2%、DeepSeek 62.0%、腾讯元宝 16.5%、Kimi 16.4%、**文心一言 12.2%**（图表）。首选：豆包 47.1%、DeepSeek 34.0%、其他 18.8%。
- **原文引语**：「34.0%的用户表示自己会首先选择使用 DeepSeek」
- **口径**：使用率指过去半年使用过该产品的用户占比（脚注 18）。
- **发布方 / 日期 / 数据期 / URL**：同 Q1-08
- **状态**：VERIFIED-PRIMARY

### Q1-11 CNNIC 2024 报告（2024.6）：组内使用率（旧口径，仅作背景）
- **数字**：用户 2.3 亿。**20–29 岁网民的组内使用率** 40.5%；**大专及以上网民的组内使用率** 44.0%；62.2% 的用户会用来提问。
- **原文引语**：「20-29岁网民使用生成式人工智能产品的比例最高」
- **口径警示**：这里是"该组网民中使用过的比例"，与 Q1-03 的"占 AI 用户的比例"不是同一口径，**不可混比**。
- **发布方**：CNNIC《生成式人工智能应用发展报告（2024）》。本条经多知网转载、腾讯新闻页面读取。
- **发布日期**：2024-12-05
- **URL**：https://news.qq.com/rain/a/20241205A03SB200
- **状态**：VERIFIED-SECONDARY（仅 WebFetch 读取）

### Q1-12 QuestMobile 2026Q1：AI 原生 App 规模、黏性、"男性 64.6%"（语义存疑）
- **论断**：2026 年 3 月 AI 原生 App 月活约 4.4–4.46 亿，月人均 87.1 次、173.3 分钟。页面称"男性用户占比达64.6%"。
- **数字**：导语：月活 4.4 亿；豆包 3.45 亿、千问 1.66 亿、DeepSeek 1.27 亿；一季度平均活跃率 33.5% / 17.1% / 21%。正文：规模 4.46 亿（较 2025 年 11 月 +43.4%）；月人均 87.1 次、173.3 分钟；"男性用户占比达64.6%，同比提升4.5%，用户规模增加1.2亿；60后及三线以下用户占比分别提升1.7%和2.4%"。
- **原文引语**：「男性用户占比达64.6%，同比提升4.5%」
- **发布方**：QuestMobile 研究院（官网）
- **发布日期**：2026-04-21
- **数据期**：2026 年 3 月 / 2026 年一季度
- **口径**：终端监测（telemetry），范围仅限 AI 原生 App，性别由 QuestMobile 模型推断。CNNIC 是调查口径，覆盖全部生成式 AI 用户（含内置助手和网页端），两者**不可直接比较**。这句话没有写明分母，而且放在"银发+下沉"的段落中，可能是编辑错误。
- **URL**：https://www.questmobile.com.cn/research/report/2046482337382842370/
- **状态**：VERIFIED-PRIMARY（文字确实存在），但**建议不作为事实引用**，只能作为"QuestMobile 口径显示……（分母未说明）"。

### Q1-13 QuestMobile 2026 半年报：AI 原生 App 4.99 亿、92.7 次；传统搜索人均次数与时长下降
- **数字**：导语："截止到2026年5月"，AI 原生 APP / 插件 / 终端厂商 AI / PC 网页端 / PC 客户端的月活分别为 4.99亿 / 6.44亿 / 7.55亿 / 1.72亿 / 0.18亿，同比 +85.4% / −7.9% / +14.0% / −22.8% / +20.1%；月人均次数 92.7 / 60.9 / 51.4 / 25.3 / 26.9。正文："2026年6月，AI原生App整体规模4.99亿"，与导语月份矛盾。2026 年 6 月豆包 3.82亿、千问 1.67亿、DeepSeek 1.29亿。"2026年5月，传统搜索行业人均使用次数、时长同比分别下降19.1%和13.5%"。
- **原文引语**：「传统搜索行业人均使用次数、时长同比分别下降19.1%和13.5%」
- **发布方**：QuestMobile 研究院（官网）
- **发布日期**：2026-07-14
- **口径**：终端监测。搜索是"人均"次数和时长，不是总量。
- **URL**：https://www.questmobile.cn/research/report/2076954943839809537/
- **状态**：VERIFIED-PRIMARY。月份不一致；+172.1% 和 −20.3% 在该页上 NOT PRESENT。

### Q1-14 生成式 AI 用户的城乡/城市线级结构
- CNNIC 第57次报告中，生成式 AI 用户结构只有性别、年龄、学历、职业四项，**没有城乡或线级**（NOT PRESENT）。网民整体的城乡结构是城镇 72.1%、农村 27.9%，不能替代。QuestMobile 只给出"三线以下用户占比提升2.4%"这一增量，没有给结构（见 Q1-12）。
- **状态**：NOT FOUND（可信的城市线级结构）

---

## Q2. 百度（2025–2026）

### Q2-01 百度移动搜索结果页中含 AI 生成内容的页面占比（时间序列）
- **论断**：2025 年内，百度移动搜索结果页中含 AI 生成内容的比例从约两成升到约七成。
- **数字**：22%（2025 年 1 月）→ 35%（4 月）[CNNIC 2025 报告]；超过 50%（6 月底）→ 64%（7 月）[百度 Q2 2025 财报新闻稿]；约 70%（10 月）[百度 Q3 2025 财报新闻稿]。
- **原文引语**：「By July, 64% of mobile search result pages contained AI-generated content」；「roughly 70% of mobile search result pages contained AI-generated content」；「百度移动搜索结果页面中人工智能生成内容已从1月的22%提升至4月的35%」
- **发布方**：百度（PR Newswire 官方发布渠道）；CNNIC
- **发布日期**：Q2 2025 新闻稿 2025-08-20；Q3 2025 新闻稿 2025 年 11 月（具体日期未从页面抓取）；CNNIC 2025 年 10 月
- **口径**：公司口径，指**"含 AI 生成内容的结果页"占比**。它不是"AI 回答的查询占比"，也不是流量占比，且百度未公开定义。2025 年 10 月之后的季度新闻稿中未再找到更新数字。
- **URL**：https://www.prnewswire.com/news-releases/baidu-announces-second-quarter-2025-results-302534389.html ；https://www.prnewswire.com/news-releases/baidu-announces-third-quarter-2025-results-302618226.html ；CNNIC PDF 同 Q1-08（第25页）
- **状态**：VERIFIED-PRIMARY

### Q2-02 百度 App MAU：2025 年 9 月以来持续下滑
- **数字**：7.08 亿（2025 年 9 月，同比 +1%）→ 6.79 亿（2025 年 12 月，同比持平）→ 6.55 亿（2026 年 3 月）→ 6.44 亿（2026 年 6 月）。派生计算：2025 年 9 月至 2026 年 6 月减少 6,400 万，降幅约 9.0%。
- **原文引语**：「Baidu App's MAUs reached 644 million in June 2026.」
- **发布方**：百度财报新闻稿（PR Newswire）
- **发布日期**：Q3 2025（2025 年 11 月）；Q4 2025（2026-02-26，据发布渠道 URL）；Q1 2026（2026-05-18，据 ir.baidu.com 标题）；Q2 2026（2026 年 8 月）
- **口径**：公司口径 MAU。李彦宏在 Q4 2025 电话会上说"around 700 million"，属于四舍五入，以新闻稿的 6.79 亿为准。
- **URL**：…302618226.html（Q3 2025）；https://www.prnewswire.com/news-releases/baidu-announces-fourth-quarter-and-fiscal-year-2025-results-302698026.html ；https://www.prnewswire.com/news-releases/baidu-announces-first-quarter-2026-results-302774476.html ；https://www.prnewswire.com/news-releases/baidu-announces-second-quarter-2026-results-302853860.html
- **状态**：VERIFIED-PRIMARY

### Q2-03 文心助手（ERNIE Assistant）MAU 2.02 亿（2025 年 12 月）
- **原文引语**：「ERNIE Assistant's MAU reached 202 million in December 2025.」
- **发布方 / 日期**：百度 Q4 2025 财报新闻稿，2026-02-26
- **口径**：公司口径 MAU。英文"ERNIE Assistant"对应中文"文心助手"。统计范围是否含百度 App 内入口，新闻稿未说明。
- **URL**：https://www.prnewswire.com/news-releases/baidu-announces-fourth-quarter-and-fiscal-year-2025-results-302698026.html
- **状态**：VERIFIED-PRIMARY

### Q2-04 文心助手 DAU：2026Q1 同比近翻倍，日均对话轮次超过 3 倍
- **原文引语**：「daily active users of ERNIE Assistant nearly doubled year-over-year」（李彦宏）
- **其他**：日均对话轮次"more than tripled"；次日留存"improved meaningfully"。
- **发布方**：百度 Q1 2026 财报电话会。转录稿托管于 Investing.com。
- **日期**：财报发布于 2026-05-18
- **口径**：只有同比倍数，没有绝对值。Q1 2026 新闻稿中**没有**这一 DAU 表述。
- **URL**：https://www.investing.com/news/transcripts/earnings-call-transcript-baidu-q1-2026-highlights-ai-growth-revenue-challenges-93CH-4705487
- **状态**：VERIFIED-SECONDARY（仅 WebFetch 读取；Investing.com、Benzinga、GuruFocus 均无法原文 grep）

### Q2-05 文心助手 DAU：2026 年 6 月同比 +83%，日均对话轮次超过 3 倍
- **原文引语**：「ERNIE Assistant's daily active users grew 83% year-over-year」
- **发布方**：百度 Q2 2026 财报电话会（Motley Fool 转录稿，已原文 grep）
- **日期**：2026-08-18（转录稿 URL）；数据期 2026 年 6 月
- **口径**：只有同比增速，没有绝对值。**Q2 2026 新闻稿全文没有"ERNIE Assistant"字样**，某些搜索摘要称新闻稿里有，不准确。
- **URL**：https://www.fool.com/earnings/call-transcripts/2026/08/18/baidu-bidu-q2-2026-earnings-call-transcript/
- **状态**：VERIFIED-SECONDARY（管理层原话，第三方转录）

### Q2-06 百度把 AI 搜索与文心助手整合成多轮对话，并刻意延后 AI 搜索变现
- **原文引语**：「more coherent interactive multi-round conversations」；「deliberately holding back on monetizing the AI search」
- **发布方 / 日期 / URL**：同 Q2-05
- **与本报告的关联**：百度自己把搜索答案往"多轮对话"方向引导。因此 AI 助手查询与搜索查询的分布差异，部分可能是产品设计造成的，不完全是用户自发的偏好。
- **状态**：VERIFIED-SECONDARY

### Q2-07 CNNIC 对百度"智能搜索"改造的描述（定性）
- **原文引语**：「百度将传统搜索结果页升级为“百看”结果页」（第57次）；「百度在2025 年 1 月正式上线“AI 搜”功能」（2025 报告）
- **补充**：2025 报告称百度 2025 年 5 月推出"深度搜索"；第57次报告称"百看"结果页"为用户提供AI总结回答"。
- **URL**：Q1-01 PDF（第25–26页）；Q1-08 PDF（第25页）
- **状态**：VERIFIED-PRIMARY（定性描述）

### Q2-08 搜索引擎用户：规模与使用率在 2025 年明显下降（CNNIC）
- **数字**：7.82 亿人，占网民 69.5%（2025 年 12 月）；同一报告表格中 2024 年 12 月为 8.78 亿（87,782 万）、79.2%。派生计算：一年减少约 9,576 万人，下降 9.7 个百分点。
- **原文引语**：「搜索引擎用户规模达7.82亿人」
- **发布方 / 日期 / URL**：同 Q1-01（PDF 第25–26、30页）
- **口径**：调查，指过去半年使用过搜索引擎的网民比例。AI 搜索与传统搜索的边界正在变模糊，问卷口径是否受影响无法核实，引用时应说明这是调查数据。
- **状态**：VERIFIED-PRIMARY

### Q2-09 百度关于"用户 query 变长、更口语化"的官方量化表述
- Q4 2025、Q1 2026、Q2 2026 的财报新闻稿和电话会转录稿中，**都没有**关于 query 长度或复杂度的量化表述。Q2 2026 转录稿原文 grep 了 "longer" 和 "complex"，只出现在无关语境中。
- 网传"AI 场景下 query 长度是传统搜索的两倍以上"出自极客公园/智源社区转载的**匿名嘉宾圆桌**（2024-12-10），不是百度官方，**不可引用**。https://hub.baai.ac.cn/view/41728
- **状态**：NOT FOUND（百度官方）

### Q2-10 百度 Q4 2025 电话会：搜索结果中加入 AI 生成的信息图，MCP 接入电商、医疗、本地生活
- **原文引语**：「introduced AI-generated infographics into our search results」（李彦宏）
- **发布方 / 日期**：百度 Q4 2025 电话会（Investing.com 转录稿），2026-02-26
- **URL**：https://www.investing.com/news/transcripts/earnings-call-transcript-baidu-q4-2025-reveals-ai-growth-amid-revenue-decline-93CH-4527912
- **状态**：VERIFIED-SECONDARY（仅 WebFetch 读取）

---

## Q3. 搜索 vs 聊天机器人：国际可信研究

### Q3-01 OpenAI / NBER《How People Use ChatGPT》：非工作消息占比升至 73%
- **数字**：非工作相关消息占比 53%（2024 年 6 月）→ 73%（2025 年 6 月）（论文表1）；摘要写 "more than 70%"。
- **原文引语**：「non-work-related messages, which have grown from 53% to more than 70%」
- **发布方**：Chatterji, Cunningham, Deming, Hitzig, Ong, Shan, Wadman（OpenAI 与哈佛等机构合作）；NBER Working Paper 34255
- **发布日期**：2025 年 9 月
- **数据期**：消费者版（Free/Plus/Pro）消息随机样本，2024 年 5 月至 2025 年 6 月
- **口径**：遥测数据，由自动分类器判定，指**消息占比**，只含消费者版。
- **URL**：https://www.nber.org/papers/w34255 ；全文 https://cdn.openai.com/pdf/a253471f-8260-40c6-a2cc-aa93fe9f142e/economic-research-chatgpt-usage-paper.pdf
- **状态**：VERIFIED-PRIMARY

### Q3-02 同上：话题构成（实用指导、信息搜寻、写作）
- **数字**：三类合计约 77%（摘要写 "nearly 80%"）。Practical Guidance 约 29%，保持稳定；Seeking Information 从 14% 升到 24%；Writing 从 36% 降到 24%（2024 年 7 月 → 2025 年 7 月）。Tutoring or Teaching 占全部消息的 10.2%。
- **原文引语**：「Seeking Information has grown from 14% to 24% of all usage」
- **URL / 状态**：同 Q3-01；VERIFIED-PRIMARY
- **注**：论文称 Seeking Information「appears to be a very close substitute for web search」。

### Q3-03 同上：Asking / Doing / Expressing
- **数字**：Asking 49%，Doing 40%，Expressing 11%。工作相关消息中约 56% 是 Doing；Writing 约占工作消息的 40%。
- **原文引语**：「about 49% of messages are Asking, 40% are Doing, and 11% are Expressing」
- **URL / 状态**：同 Q3-01；VERIFIED-PRIMARY
- **本次未取到**：Health 类话题的具体占比，文本抽取未得到数值，不报告。

### Q3-04 微软研究院 Suri 等：Bing Copilot 的任务更偏知识工作、认知复杂度更高
- **数字**：72.9% 的 Bing Copilot 对话属于知识工作领域，Bing 搜索会话为 37%。高复杂度类别（Apply/Analyze/Evaluate/Create）中，Copilot 占 37.0%，搜索占 13.4%。
- **原文引语**：「72.9% of Bing Copilot conversations are in knowledge work domains」；「Only 13.4% of Bing Search sessions fall into the higher complexity categories」
- **发布方**：Microsoft（Suri, Counts, … Ryen W. White, … Longqi Yang）；arXiv 2404.04268
- **发布日期**：2024-03-19
- **数据期**：2023-05-28 至 2023-07-22 共八周；约 8 万条 Copilot 对话和约 8 万个搜索会话（登录用户）；由 GPT-4 标注。
- **口径**：遥测数据，样本是 Bing Chat 早期、英文为主的用户。
- **URL**：https://arxiv.org/abs/2404.04268 （全文 https://arxiv.org/pdf/2404.04268）
- **状态**：VERIFIED-PRIMARY
- **更正**：搜索摘要称"分析了 2024 年 20 万条对话"，原文中没有（NOT PRESENT）。

### Q3-05 Microsoft《Copilot Usage Report 2025》：手机端以健康话题为主，"搜索"是最常见意图
- **数字**：3,750 万条去标识化对话（2025 年 1–9 月）。手机端健康话题在每个小时、每个月都居首。"Searching" 是最常见的意图。
- **原文引语**：「On mobile, health is the dominant topic」
- **发布方**：Microsoft AI（Costa-Gomes, Spielman 等）；arXiv 2512.11879；博客发布于 2025-12-10
- **口径**：遥测数据，消费者版 Copilot，全球样本。博客没有给出话题的百分比。
- **URL**：https://arxiv.org/pdf/2512.11879v1 ；https://microsoft.ai/news/its-about-time-the-copilot-usage-report-2025/
- **状态**：VERIFIED-PRIMARY

### Q3-06 Google（Alphabet Q4 2025 电话会 CEO 讲话稿）：AI Mode 查询长度是传统搜索的 3 倍
- **原文引语**：「Queries in AI Mode are three-times longer than traditional searches.」
- **其他**：AI Mode 中 "a significant portion of queries" 会引出追问；近六分之一的 AI Mode 查询是语音或图片等非文本形式；美国每位用户的日均 AI Mode 查询量自上线以来翻倍。
- **发布方**：Alphabet / Sundar Pichai（blog.google）
- **日期**：Q4 2025 财报（2026 年初；页面日期未抓取）
- **口径**：公司遥测。"长度"按词还是按字符、基线如何选取，均未披露。
- **URL**：https://blog.google/company-news/inside-google/message-ceo/alphabet-earnings-q4-2025/
- **状态**：VERIFIED-PRIMARY

### Q3-07 Google 官方博客（2026-05-19）：美国 AI Mode 平均查询长度是传统查询的三倍
- **原文引语**：「the average AI Mode search is triple the length of a traditional Search query」
- **其他**：过去 6 个月，规划类查询的增速比 AI Mode 整体快 80%；"where to""ideas for"等开头的查询在增长。
- **发布方**：Google，Shivani Mohan（VP, Data Science and UXR）
- **口径**：美国；公司遥测
- **URL**：https://blog.google/products-and-platforms/products/search/ai-mode-us-insights/
- **状态**：VERIFIED-PRIMARY

### Q3-08 Google（Q2 2025 与 Q2 2026 讲话稿）：AI 功能带来查询增量
- **原文引语**：「particularly for longer and more complex questions」（Q2 2025，谈 AI Mode）；「driving over 10% more queries globally for the types of queries that show them」（Q2 2025，谈 AI Overviews）；「AI Mode is driving an incremental increase in Search queries overall」（Q2 2026）
- **其他**：Q2 2026 讲话稿称 AI Mode 全球月活超过 10 亿。
- **URL**：https://blog.google/company-news/inside-google/message-ceo/alphabet-earnings-q2-2025/ ；https://blog.google/company-news/inside-google/message-ceo/alphabet-earnings-q2-2026/
- **状态**：VERIFIED-PRIMARY

### Q3-09 Anthropic Economic Index（2026-01-15）：增强型使用重新超过自动化
- **数字（Claude.ai）**：directive 对话占比从 39%（2025 年 8 月）回落到 32%（2025 年 11 月）；增强型 52% vs 自动化 45%。用途：工作 46%、课业 19%、个人 35%。"Educational Instruction and Library" 类任务从 9%（2025 年 1 月）升到 15%（2025 年 11 月）。API 端：directive 64%，工作相关 74%。
- **原文引语**：「Overall, Claude.ai use is 46% work, 19% coursework, and 35% personal.」
- **数据期 / 样本**：2025-11-13 至 11-20；100 万条 Claude.ai 对话，另加 100 万条 API 记录
- **口径**：遥测数据加分类器，指对话占比。Claude 的用户群偏专业，不能代表大众。
- **URL**：https://www.anthropic.com/research/anthropic-economic-index-january-2026-report
- **状态**：VERIFIED-PRIMARY

### Q3-10 Pew（2025-07-22）：越长、越像问句的 Google 搜索越容易触发 AI 摘要
- **数字**：1–2 个词的搜索只有 8% 触发 AI 摘要，10 个词及以上为 53%；以 who/what/when/why 开头的查询为 60%；含名词加动词的完整句为 36%。
- **原文引语**：「Longer searches are more likely to produce an AI summary.」
- **发布方**：Pew Research Center（Athena Chapekis, Anna Lieb）
- **数据期 / 样本**：2025 年 3 月；900 名美国成人（KnowledgePanel Digital）的浏览行为数据
- **URL**：https://www.pewresearch.org/short-reads/2025/07/22/google-users-are-less-likely-to-click-on-links-when-an-ai-summary-appears-in-the-results/ （经 archive.org 快照核对）
- **状态**：VERIFIED-PRIMARY

### Q3-11 "ChatGPT 提示词平均 23 个词 vs Google 查询 4.2 个词"（Semrush）
- 这是 SEO 厂商的研究，可信度低。**当前 Semrush 页面（2026-04-07 更新）上没有这组数字**。
- URL：https://www.semrush.com/blog/chatgpt-search-insights/
- **状态**：NOT PRESENT ON PAGE / 仅 SEO 厂商 → 不可引用

---

## Q4. 健康

### Q4-01 KFF（2026-03-25）：32% 的美国成人过去一年用 AI 聊天机器人获取健康信息
- **数字**：总计 32%；身体健康 29%；心理健康 16%。用 AI 获取心理健康信息的比例，30 岁以下为 28%，50 岁及以上为 8%。
- **原文引语**：「About a third (32%) of adults nationally say they have turned to」
- **发布方**：KFF Tracking Poll on Health Information and Trust
- **数据期 / 样本**：2026-02-24 至 03-02；1,343 名美国成人；网络与电话；误差 ±3 个百分点
- **口径**：概率样本调查
- **URL**：https://www.kff.org/health-information-trust/poll-1-in-3-adults-are-turning-to-ai-chatbots-for-health-information-equaling-the-share-who-use-social-media-for-health/
- **状态**：VERIFIED-PRIMARY

### Q4-02 KFF：使用原因、是否就医跟进、上传病历
- **数字**：使用者中，65% 把"快速或即时获得信息"列为主要原因；41% 是想先查一查再决定是否就医；36% 觉得私下查询更自在。不就医跟进的比例：问心理健康的 58%，问身体健康的 42%。41% 的使用者上传过个人医疗信息。
- **原文引语**：「did not follow up with a doctor or other health professional」
- **URL / 状态**：同 Q4-01；VERIFIED-PRIMARY

### Q4-03 Microsoft AI "Health Check"（2026-03-10）：手机端症状类提问是桌面端的 2 倍
- **数字**：超过 50 万条健康对话（2026 年 1 月；全球样本，约 22% 来自美国，约 45% 为英文；不含企业和教育账户）。约 40% 的提问是了解症状、疾病和治疗；近五分之一涉及个人症状评估或病情讨论；症状类对话中七分之一是替别人问。手机端询问症状和病情管理的比例是桌面端的 2 倍，情绪健康类多 75%；夜间个人症状类查询上升。
- **原文引语**：「at twice the rate they do on desktop」；「one in seven are on behalf of someone else」
- **发布方**：Microsoft AI（Tolmachev, Costa-Gomes, Sounderajah；论文 2026 年 3 月）
- **口径**：遥测数据，指对话占比
- **URL**：https://microsoft.ai/news/health-check-how-people-use-copilot-for-health/ ；https://www.microsoft.com/en-us/research/wp-content/uploads/2026/03/copilot-health-usage-report.pdf
- **状态**：VERIFIED-PRIMARY

### Q4-04 CNNIC 第57次：互联网医疗用户 4.11 亿；报告引用蚂蚁阿福月活超过 3000 万
- **数字**：互联网医疗用户 4.11 亿，占网民 36.5%（2025 年 12 月；表中 2024 年 12 月为 4.18 亿、37.7%）。报告另引用：蚂蚁阿福月活"超3000万"（来源为蚂蚁集团官网，2025-12-15）；京东健康"AI京医"系列智能体累计服务用户超过 5000 万（来源为京东健康中期业绩公告）。
- **原文引语**：「“蚂蚁阿福”的月活用户也超3000万」
- **发布方**：CNNIC（阿福和 AI 京医数据由 CNNIC 转引公司口径）
- **URL**：同 Q1-01（PDF 第39–40页）
- **状态**：互联网医疗用户为 VERIFIED-PRIMARY。阿福和 AI 京医为 VERIFIED-SECONDARY（公司自报；蚂蚁原页面需 JS 渲染，未能直接打开）。

### Q4-05 丁香园《2026中国医生AI行为与态度调研报告》：近 60% 的医生经常遇到带着 AI 诊疗建议来就诊的患者
- **数字**：近 60% 的医生经常遇到这类患者；超过 70% 支持 AI 参与医患沟通；近 80% 的医生已在工作中使用 AI，约 75% 每天使用。
- **原文引语**：「携带 AI 生成的诊疗建议前来咨询」
- **发布方**：丁香园（问卷由麦肯锡设计）。本条经中国日报网转载的鲁网报道读取。
- **发布日期**：发布会 2026-07-28；报道 2026-07-31
- **样本**：丁香园平台 4,150 份医生问卷，非概率样本
- **URL**：https://cn.chinadaily.com.cn/a/202607/31/WS6a6c0e96a310d709c2fc0b79.html
- **状态**：VERIFIED-SECONDARY（商业平台调查，媒体转述）
- **与本报告的关联**：从医生侧间接说明中国患者会用 AI 做健康咨询。

### Q4-06 国家卫健委等五部门《关于促进和规范“人工智能+医疗卫生”应用发展的实施意见》
- **要点**：政策解读强调"坚持人工智能赋能而不替代的定位"；要求二级及以上医院提供"智能预问诊"、分诊导诊、云陪诊和智能随访；鼓励发展智能健康咨询、健康管理等业态。
- **原文引语**：「坚持人工智能赋能而不替代的定位」
- **发布方**：国家卫生健康委办公厅等五部门
- **发布日期**：文件落款 2025-10-20；北京市卫生经济学会 2025-11-05 转载
- **URL**：https://www.bjhea.org.cn/weishengjingji/35/202511/1308.html
- **状态**：VERIFIED-SECONDARY（学会网站全文转载，国家卫健委官网原页未打开）

### Q4-07 中国公众"用 AI 获取健康信息或问诊"比例的代表性调查
- **状态**：NOT FOUND

---

## Q5. 教育 / K12

### Q5-01 CNNIC 2025 报告转引第6次全国未成年人互联网使用情况调查（2024.9）：各学段使用率
- **数字**：未成年网民使用过生成式 AI 的比例：小学 13.9%，初中 23.2%，高中 26.1%，中等职业教育 19.6%。
- **原文引语**：「高中阶段网民使用过生成式人工智能产品的比例最高」
- **发布方**：CNNIC（数据源为共青团中央与 CNNIC 的第6次全国未成年人互联网使用情况调查）
- **数据期**：2024 年 9 月，数据偏旧
- **URL**：同 Q1-08（PDF 第12页）
- **状态**：VERIFIED-PRIMARY

### Q5-02 同上：未成年人使用目的
- **数字（分母是全部未成年网民，不是 AI 用户）**：学习课外知识 15.2%；写作业 10.8%；获取消息、查信息 10.6%；出于好奇试试 10.4%；写小说、写故事 7.2%；制作视频 6.8%；编写或修改代码 4.4%。
- **原文引语**：「为了学习课外知识」（15.2%）
- **URL / 状态**：同 Q5-01；VERIFIED-PRIMARY
- **警示**：分母是全部未成年网民，所以换算到"AI 用户中"的比例会更高。不要写成"AI 用户中 15.2%"。

### Q5-03 中国青少年研究中心（2025.6–8，8,563 名中小学生）：超过六成用过 AI
- **数字**：超过六成用过 AI，18.3% 经常用。分学段：小学 60.0%，初中 56.1%，高中 69.5%。城市 63.7%，农村 60.2%。样本构成：小学 25.8%、初中 40.0%、高中 34.2%；城市 44.7%、农村 55.3%。
- **原文引语**：「超六成受访中小学生用过AI」
- **发布方**：中国青少年研究中心，孙宏艳署名文章，刊于《中国青年报》2026-03-26 第08版
- **数据期**：2025 年 6–8 月；北京、广东、江苏、河南、四川、陕西、辽宁 7 省市；问卷调查，非全国概率样本
- **URL**：https://zqb.cyol.com/pc/content/202603/26/content_423920.html
- **状态**：VERIFIED-PRIMARY（研究机构研究员署名发表）
- **更正**：虎嗅等转载称"61.7%"和"2025年5–7月"，原文中没有（NOT PRESENT）。原文写的是"超六成"和"2025年6月至8月"。

### Q5-04 同上：首要用途是辅助写作业
- **数字**：辅助完成作业 71.0%；学习文化知识 43.5%；创作 43.0%；聊天或说心里话 35.2%；娱乐 32.5%。农村学生用 AI 辅助写作业的比例为 73.2%，城市为 68.4%。
- **原文引语**：「用AI辅助完成作业」（71.0%）
- **口径**：原文没有明确说明分母是全部受访者还是用过 AI 的学生
- **URL / 状态**：同 Q5-03；VERIFIED-PRIMARY

### Q5-05 同上：风险与依赖
- **数字**：34.6% 遇到过 AI 提供错误信息；47.4% 担心隐私泄露；21.0% 因过多使用 AI 感到焦虑；20.5% "想依赖AI思考不想自己思考"（农村 22.8%，城市 17.7%）。
- **原文引语**：「遇到过AI提供错误的信息」（34.6%）
- **URL / 状态**：同 Q5-03；VERIFIED-PRIMARY

### Q5-06 Pew（2026-02-24）：美国青少年使用聊天机器人
- **数字**：64% 的青少年说自己用聊天机器人，约三成每天用。用途：搜索信息 57%，帮助写作业 54%，娱乐 47%，约五分之一用来获取新闻。10% 的人全部或大部分作业借助聊天机器人完成，21% 部分借助，23% 少量借助。
- **原文引语**：「One-in-ten teens say they do all or most of their schoolwork」
- **发布方**：Pew Research Center
- **数据期 / 样本**：2025-09-25 至 10-09；1,458 名 13–17 岁美国青少年及其家长（Ipsos KnowledgePanel）
- **URL**：https://www.pewresearch.org/internet/2026/02/24/how-teens-use-and-view-ai/ （经 archive.org 快照核对）
- **状态**：VERIFIED-PRIMARY

### Q5-07 Pew（2025-01-15）：用 ChatGPT 做作业的美国青少年比例翻倍
- **数字**：26%（2024 年），2023 年为 13%。
- **原文引语**：「use ChatGPT for their schoolwork has risen to 26%」
- **样本**：2024-09-18 至 10-10；1,391 名美国青少年
- **URL**：https://www.pewresearch.org/short-reads/2025/01/15/about-a-quarter-of-us-teens-have-used-chatgpt-for-schoolwork-double-the-share-in-2023/ （经 archive.org 快照核对）
- **状态**：VERIFIED-PRIMARY

### Q5-08 CNNIC 第57次：在线教育用户 3.27 亿
- **数字**：3.27 亿，占网民 29.1%（2025 年 12 月）
- **原文引语**：「在线教育用户规模达3.27亿人」
- **URL / 状态**：同 Q1-01；VERIFIED-PRIMARY
- **交叉引用**：学生占生成式 AI 用户 30.1%（Q1-05）；ChatGPT 消息中 Tutoring/Teaching 占 10.2%（Q3-02）；Claude.ai 课业用途 19%、教育类任务 15%（Q3-09）。

---

## Q6. 金融

### Q6-01 清华五道口与蚂蚁集团研究院：四成多个人投资者用过 AI 工具
- **数字（个人投资者，n=1,514）**：深度使用 14.80%，偶尔使用 28.14%，尝试过但已不用 4.95%，从未使用 52.11%。派生计算：目前在用（深度加偶尔）为 42.94%。
- **原文引语**：「超过四成个人投资者已开始使用 AI 工具」
- **发布方**：清华大学五道口金融学院财富管理研究中心、蚂蚁集团研究院《AI财富管理服务现状与趋势研究（2025年）》（研究报告 2025 年第4期）
- **发布日期**：2025-09-25
- **样本**：线上问卷，有效问卷 1,627 份，其中个人投资者 1,514 份、机构 113 份
- **口径**：非概率样本。合作方蚂蚁集团有商业利益。"AI 工具"泛指财富管理过程中的任何 AI 工具，不等于"AI 荐股"。
- **URL**：https://thuifr.pbcsf.tsinghua.edu.cn/AIcaifuguanlifuwuxianzhuangyuqushi.pdf
- **状态**：VERIFIED-PRIMARY

### Q6-02 同上：用户认为 AI 的主要价值
- **数字**："随时随地提供服务" 24.14%；"降低了专业理财服务的门槛" 23.31%；"提供个性化的投资建议" 20.69%。报告另转引北京大学黄益平团队的研究："当前 AI 理财用户以年轻、男性及风险偏好较高群体为主"，这一句属于报告内的二手引用。
- **原文引语**：「随时随地提供服务」（24.14%）
- **URL / 状态**：同 Q6-01；VERIFIED-PRIMARY（黄益平团队的结论为报告内二手转引）

### Q6-03 证监会（2025-12-31）：警示投资者不轻信"金融网红""智能投顾"
- **要点**：警示投资者不轻信"金融网红""智能投顾"，应独立审慎决策，考虑工具与自身情况是否适配，并警惕利益冲突和工具局限。同时开展"人工智能大讲堂"，帮助投资者了解智能工具的功能与局限。
- **原文引语**：「警示投资者不轻信“金融网红”“智能投顾”」
- **发布方**：中国证券监督管理委员会（2025 年"世界投资者周"活动总结）
- **URL**：https://www.csrc.gov.cn/csrc/c100210/c7605970/content.shtml
- **状态**：VERIFIED-PRIMARY

### Q6-04 中国证券业协会"投资者之家"（2025-03-18）：假"AI荐股"、真投资骗局
- **要点**：不法分子兜售"AI 智能炒股机器人"等非法软件牟利；券商在开展防非宣传。
- **原文引语**：「假“AI荐股”真投资骗局」
- **URL**：https://www.sac.net.cn/tzzzj/zxsd/xscyw/202503/t20250319_67487.html
- **状态**：VERIFIED-SECONDARY（行业协会网站转载的媒体报道，文中有"记者从券商处获悉"）

### Q6-05 缺口
- 中国散户用 AI 获取股票信息或荐股的**代表性**调查：NOT FOUND
- 金融监管总局专门针对"AI 投资建议"的表态：NOT FOUND（未打开任何原页面）

---

## Q7. 信任与核验行为

### Q7-01 Reuters Institute DNR 2026：从 AI 点回原始新闻来源的人极少
- **数字**：在 27 个问了该题的市场中，"总是或经常"点回原始新闻来源的比例：AI 4%，搜索 19%，社交媒体 17%。用 AI 聊天机器人获取新闻的比例为 9%（27 个市场）。全球每周用 AI 聊天机器人看新闻的比例从 7% 升到 10%。
- **原文引语**：「just 4% of respondents overall say they always or often click through」
- **发布方**：Reuters Institute for the Study of Journalism，Dr Amy Ross Arguedas（Digital News Report 2026 专章）
- **发布日期**：2026-06-16
- **口径**：YouGov 在线调查，全样本（含非用户）
- **URL**：https://reutersinstitute.politics.ox.ac.uk/digital-news-report/2026/emerging-uses-ai-chatbots-news-and-what-it-means-journalism （经 archive.org 快照核对）
- **状态**：VERIFIED-PRIMARY

### Q7-02 同上：AI 用户点回来源多是为了核实；使用者对 AI 新闻的信任明显高于非使用者
- **要点**：AI 聊天机器人用户比搜索和社交媒体用户更可能"为了核实新闻或了解来源"而点回原文。44% 的 AI 聊天机器人用户信任来自 AI 的新闻，非用户为 17%，一般人群为 20%。
- **原文引语**：「click through because they want to verify the news」
- **URL / 状态**：同 Q7-01；VERIFIED-PRIMARY
- **更正**：搜索摘要称"44% 为核实事实、43% 为查看来源"。原文中 44% 是**信任度**，核实动机的具体百分比在正文中没有找到（NOT PRESENT）。

### Q7-03 Pew（2025-07-22）：出现 AI 摘要时，用户更少点链接、更常直接结束浏览
- **数字**：2025 年 3 月，18% 的 Google 搜索出现了 AI 摘要，58% 的受访者至少遇到过一次。点击传统搜索结果的比例：有摘要 8%，无摘要 15%。点击摘要内链接仅 1%。浏览会话结束的比例：有摘要 26%，无摘要 16%。88% 的摘要引用了 3 个及以上来源；摘要长度中位数 67 词。
- **原文引语**：「clicked on a traditional search result link in 8% of all visits」
- **URL / 状态**：同 Q3-10；VERIFIED-PRIMARY
- **口径**：美国成人浏览行为遥测（n=900），不是自报

### Q7-04 KFF：AI 与就医之间既有互补也有替代
- 41% 的使用者先用 AI 查询，再决定是否就医（互补）。但问心理健康的使用者中 58%、问身体健康的 42% 事后没有找医生跟进。见 Q4-02。**状态**：VERIFIED-PRIMARY

### Q7-05 Google：AI 功能带来搜索查询的增量（互补证据）
- AI Overviews 使相关类型查询增加超过 10%（Q2 2025）；AI Mode 带来整体查询增量（Q2 2026）。见 Q3-08。**状态**：VERIFIED-PRIMARY
- **口径提醒**：这是 Google 自报的公司口径，有利益相关。

### Q7-06 中国侧的间接证据
- 中国青少年研究中心：34.6% 的中小学生遇到过 AI 给出错误信息（Q5-05）。
- 丁香园：近 60% 的医生经常遇到带着 AI 诊疗建议来的患者（Q4-05）。
- CNNIC：搜索引擎用户使用率从 79.2% 降到 69.5%（Q2-08）；QuestMobile：传统搜索的人均次数和时长同比下降（Q1-13）。

### Q7-07 中国用户"用搜索核验 AI 答案"或"AI 与搜索互补使用"的实证研究
- **状态**：NOT FOUND

---

## 可放心引用（VERIFIED）

**VERIFIED-PRIMARY**（共 44 条）

1. Q1-01 CNNIC：生成式 AI 用户 6.02 亿，普及率 42.8%（分母为整体人口），+141.7%
2. Q1-02 性别：AI 用户 50.3:49.7，网民 51.2:48.8
3. Q1-03 年龄：AI 用户 19 岁及以下 26.4%、20–29 岁 21.3% … 60 岁及以上 5.1%；网民年龄结构（派生对比需注明）
4. Q1-04 学历：大专及以上 38.6%，初中 29.9%
5. Q1-05 职业：学生 30.1%
6. Q1-06 使用目的：回答问题 76.0%，图片/视频 47.8%，文本 37.6%，办公材料 32.5%
7. Q1-07 产品形态：APP 86.0%，内置助手 41.4%；原生 App 月人均 143.2 分钟
8. Q1-08 CNNIC 2025.6：性别 47.6:52.4；19 岁及以下 33.8%
9. Q1-09 CNNIC 2025.6：回答问题 80.9%
10. Q1-10 CNNIC 2025.6：豆包 72.2%、DeepSeek 62.0%、文心一言 12.2%；首选豆包 47.1%
11. Q1-13 QuestMobile：传统搜索人均次数 −19.1%、时长 −13.5%（2026.5）；AI 原生 App 4.99 亿、92.7 次（月份标注不一）；豆包 3.82 亿、千问 1.67 亿、DeepSeek 1.29 亿（2026.6）
12. Q2-01 百度移动搜索结果页 AI 生成内容占比：22%（2025.1）→ 35%（4 月）→ 64%（7 月）→ 约 70%（10 月）
13. Q2-02 百度 App MAU：7.08 亿 → 6.79 亿 → 6.55 亿 → 6.44 亿（2025.9–2026.6）
14. Q2-03 文心助手 MAU 2.02 亿（2025.12）
15. Q2-07 CNNIC 对"百看""AI 搜""深度搜索"的定性描述
16. Q2-08 搜索引擎用户 7.82 亿 / 69.5%（2025.12），2024.12 为 8.78 亿 / 79.2%
17. Q3-01 ChatGPT 非工作消息 53% → 73%
18. Q3-02 Practical Guidance 约 29%，Seeking Information 14% → 24%，Writing 36% → 24%，Tutoring 10.2%
19. Q3-03 Asking 49% / Doing 40% / Expressing 11%
20. Q3-04 Bing Copilot 知识工作 72.9% vs 搜索 37%；高复杂度 37.0% vs 13.4%
21. Q3-05 Copilot 2025：手机端健康话题居首；Searching 是最常见意图
22. Q3-06 Google：AI Mode 查询是传统搜索的 3 倍长（Q4 2025 讲话稿）
23. Q3-07 Google 博客：美国 AI Mode 平均查询长度为三倍（2026-05-19）
24. Q3-08 Google：AI Overviews 相关查询 +10% 以上；AI Mode 带来查询增量
25. Q3-09 Anthropic：directive 32%；增强 52% / 自动化 45%；工作 46% / 课业 19% / 个人 35%；教育类 15%
26. Q3-10 Pew：1–2 词搜索触发 AI 摘要 8% vs 10 词及以上 53%
27. Q4-01 KFF：32% 用 AI 获取健康信息（身体 29%，心理 16%）
28. Q4-02 KFF：65% 为求快；41% 先查再决定是否就医；58% / 42% 未就医跟进；41% 上传过病历
29. Q4-03 Microsoft Health：手机端症状类提问为桌面端 2 倍；七分之一替他人问
30. Q4-04 CNNIC：互联网医疗用户 4.11 亿 / 36.5%
31. Q5-01 未成年网民生成式 AI 使用率：小学 13.9%、初中 23.2%、高中 26.1%（2024.9）
32. Q5-02 未成年网民使用目的：学课外知识 15.2%、写作业 10.8%、查信息 10.6%（分母为全部未成年网民）
33. Q5-03 中国青少年研究中心：超六成用过 AI，18.3% 经常用
34. Q5-04 71.0% 用 AI 辅助写作业
35. Q5-05 34.6% 遇到过 AI 错误信息；20.5% 想依赖 AI 思考
36. Q5-06 Pew 青少年：57% 用来搜索信息，54% 用来做作业，10% 全部或大部分作业借助 AI
37. Q5-07 Pew：用 ChatGPT 做作业 26%（2024），2023 年为 13%
38. Q5-08 在线教育用户 3.27 亿 / 29.1%
39. Q6-01 清华五道口与蚂蚁研究院：个人投资者深度使用 14.80%，偶尔使用 28.14%，从未使用 52.11%
40. Q6-02 AI 价值认知：24.14% / 23.31% / 20.69%
41. Q6-03 证监会：不轻信"金融网红""智能投顾"
42. Q7-01 DNR 2026：点回来源的比例，AI 4% vs 搜索 19% vs 社交 17%
43. Q7-02 DNR 2026：AI 用户点回多为核实；AI 用户对 AI 新闻信任 44% vs 非用户 17%
44. Q7-03 Pew：点击传统结果 8% vs 15%；点摘要内链接 1%；结束会话 26% vs 16%

交叉引用条目（Q7-04、Q7-05、Q7-06）复用上列来源，不重复计数。

**VERIFIED-SECONDARY**（共 9 条；可引用，但须注明是转述或转录）

- Q1-11 CNNIC 2024 组内使用率：20–29 岁 40.5%，大专及以上 44.0%（多知网转载）
- Q2-04 文心助手 DAU 2026Q1 同比近翻倍（Investing.com 转录稿，仅 WebFetch 读取）
- Q2-05 文心助手 DAU 2026.6 同比 +83%，对话轮次超过 3 倍（Motley Fool 转录稿，已原文 grep）
- Q2-06 AI 搜索整合为多轮对话；刻意延后 AI 搜索变现（同上）
- Q2-10 Q4 2025：搜索结果加入 AI 信息图（仅 WebFetch 读取）
- Q4-04 蚂蚁阿福月活超过 3000 万、AI 京医累计超过 5000 万（CNNIC 转引公司口径）
- Q4-05 丁香园：近 60% 医生常遇到带 AI 诊疗建议的患者（商业平台调查，媒体转述）
- Q4-06 五部门实施意见："赋能而不替代"、智能预问诊（学会转载）
- Q6-04 中证协：假"AI 荐股"骗局（协会网站转载媒体报道）

---

## 不可引用（NOT PRESENT / NOT FOUND / 仅 SEO 厂商 / 语义存疑）

**NOT PRESENT ON PAGE**（摘要声称有，原文没有）

1. QuestMobile "+172.1%"（豆包）和 "−20.3%"（DeepSeek）：不在 QuestMobile 官网半年报页面上，需另找来源。
2. CNNIC 2025 报告中"DeepSeek 使用率 60.0%"：PDF 中是 62.0%。
3. 中国青少年研究中心"61.7%""2025年5–7月"：原文是"超六成""2025年6月至8月"。
4. Reuters DNR 2026"44% 为核实事实 / 43% 为查看来源"：原文中 44% 是信任度，核实动机的百分比正文里没有。
5. Suri 等"2024 年 20 万条 Copilot 对话"：原文为约 8 万条，时间是 2023 年。
6. 证监会页面"理性认识AI工具，增强对各类AI骗局的辨别能力"：WebFetch 摘要给出的引语，原文 grep "AI工具""AI骗局"均为 0 条。
7. Semrush"ChatGPT 提示词 23 词 vs 搜索 4.2 词"：当前页面上没有，而且是 SEO 厂商研究。
8. 百度 Q2 2026 财报新闻稿中的文心助手 DAU：新闻稿里没有，只在电话会中出现。
9. KFF 页面上"77% 的公众担心隐私"：只出现在 WebFetch 摘要里，原文 grep 未覆盖，暂不引用。
10. CNNIC 第57次报告中"智能搜索""学习"作为使用目的的百分比：选项里没有这两个类目。
11. CNNIC 第57次报告中生成式 AI 用户的城乡或线级结构、网民学历结构：没有。

**语义存疑，不建议作为事实引用**

12. QuestMobile"男性用户占比达64.6%，同比提升4.5%"：页面上有，但没有交代分母，所处段落主题矛盾，与 CNNIC 调查口径（50.3%）冲突。

**NOT FOUND**

13. 百度官方关于"query 变长 / 更口语化"的量化表述。
14. 中国公众使用 AI 获取健康信息或问诊比例的代表性调查。
15. 中国散户使用 AI 获取股票信息的代表性调查；金融监管总局专门针对 AI 投资建议的表态。
16. 中国用户"用搜索核验 AI 答案"的实证研究。
17. 可信的生成式 AI 用户城市线级结构。

**仅 SEO 厂商或匿名观点（低可信度）**

18. Semrush 等 SEO 厂商的提示词长度研究。
19. "AI 场景下 query 长度是传统搜索两倍以上"：极客公园/智源社区转载的匿名嘉宾圆桌，2024-12。
20. 知乎文章中的"71.5% 受访者会用 AI 完成部分搜索需求""Z 世代 40% 倾向用 AI 提问"：来源不明，未打开核实。
21. 第6次全国未成年人互联网使用情况调查"19% 的未成年网民用过生成式 AI"：只见于搜索摘要，原报告未打开。
