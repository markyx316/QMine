全部脚本与中间产物在 `analysis/pooled5/`，按顺序跑：

```bash
# 1. 合并五份语料（搜索 2025/2026 + 助手头部/随机 + 语音），套用 v3 清洗分层
HF_HOME=$(pwd)/.hf .venv/bin/python analysis/pooled5/build_pooled5_corpus.py

# 2. 五次挖掘运行（fast 模式；人物用改路由的 pool5_ppl_b.yaml）
.venv/bin/qmine run --config configs/pool5_fin.yaml  --input data/raw/pooled5/金融_pooled5.parquet --domain finance_zh   --run-id fin-pool5  --fast
.venv/bin/qmine run --config configs/pool5_med.yaml  --input data/raw/pooled5/医疗_pooled5.parquet --domain medical_zh   --run-id med-pool5  --fast
.venv/bin/qmine run --config configs/pool5_edu.yaml  --input data/raw/pooled5/教育_pooled5.parquet --domain education_zh --run-id edu-pool5  --fast
.venv/bin/qmine run --config configs/pool5_film.yaml --input data/raw/pooled5/影视_pooled5.parquet --domain film_tv_zh   --run-id film-pool5 --fast
.venv/bin/qmine run --config configs/pool5_ppl_b.yaml --input data/raw/pooled5/人物_pooled5.parquet --domain people_zh   --run-id ppl-pool5b --fast

# 3. 逐域表格、逐行交付表、数字底稿
HF_HOME=$(pwd)/.hf .venv/bin/python analysis/pooled5/p5_tables.py
HF_HOME=$(pwd)/.hf .venv/bin/python analysis/pooled5/p5_deliverables.py
HF_HOME=$(pwd)/.hf .venv/bin/python analysis/pooled5/p5_digest.py 金融 > analysis/pooled5/work/金融/digest.txt   # 五个领域各一次

# 4. 对照与跨领域（顺序无关）
for s in p5_semantic_nn p5_wrap p5_voice p5_turnover p5_depth_control p5_residue \
         p5_concentration p5_route_agreement p5_form_contrasts p5_taxonomy_delta \
         p5_vs_unified p5_tvd_ci p5_cross; do
  HF_HOME=$(pwd)/.hf .venv/bin/python analysis/pooled5/$s.py
done

# 5. 表格、图、拼装
HF_HOME=$(pwd)/.hf .venv/bin/python analysis/pooled5/p5_report_tables.py
HF_HOME=$(pwd)/.hf .venv/bin/python analysis/pooled5/p5_figs.py
HF_HOME=$(pwd)/.hf .venv/bin/python analysis/pooled5/p5_assemble_report.py
HF_HOME=$(pwd)/.hf .venv/bin/python analysis/pooled5/p5_assemble_domains.py
HF_HOME=$(pwd)/.hf .venv/bin/python analysis/pooled5/p5_check_quotes.py docs/POOLED5_2026_*.zh.md
```

`p5_form_contrasts.py` 是 `cross/form_contrasts.csv` 的唯一生产者，而 T16 与 §4.4 全靠它，漏跑就重建不出来。
**注意**：`p5_vs_unified.py` 与 `p5_wrap.py` 接受领域参数，但落盘时会覆盖整张跨领域表——只传一个领域会把其余四个域的结果冲掉（本轮发生过一次）。跨领域表一律不带参数重跑。

`pooled5_common.py` 里的 `RUN_ID` 决定每个领域读哪一次运行；`P5_RUN_<key>` 环境变量可以临时覆盖。
所有统计函数（Wilson、Newcombe、Cramér's V、TVD）都在 `pooled5_common.py` 里，复核员用的是同一份实现。
