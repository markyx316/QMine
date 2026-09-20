# 人物领域用哪一次运行（**在看到结果之前定下**）

两次运行都在跑：

- `ppl-pool5` gen02 —— 在 gen01 于 p7_audit 崩溃后开的新世代。gen01 崩溃原因：风控哨兵被供应商内容过滤
  拒绝（400 contentFilter），而降级用的占位对象没有 `model_dump`。已修复（`RiskReport` 代替
  `SimpleNamespace`，并补了回归测试）。gen02 里 `researcher_pragmatic_intents` 仍被 Zhipu 拒绝 3/3，
  所以这次的体系是**用 5 个研究角度里的 4 个**设计出来的。
- `ppl-pool5b` —— 全新 run id，把**所有** Zhipu 角色改路由（researcher / domain_scout / maintainer /
  reporter / referee → Moonshot 或 DeepSeek，risk_sentinel → Moonshot）。标注员的实验室独立性不变
  （a=deepseek, b=qwen, referee=moonshot）。

**选用规则（先定后看）**：若 `ppl-pool5b` 跑完且未 halt，则人物领域用 `ppl-pool5b`，理由是它拿到了
完整的 5 个研究角度和一次真正独立的风控扫描；否则用 `ppl-pool5` gen02。另一次运行的产物全部保留作证据，
并在报告里说明两次运行的差别与选用理由。

**不做的事**：不比较两次运行的类目占比后再挑"更好看"的那次。两次运行的标签体系本来就不可比
（同一份数据跑两次共享 0 个类目编码），事后挑选等于用噪声选结论。
