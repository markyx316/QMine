#!/bin/bash
# Launch the 8-snapshot 人物 run, SECOND ATTEMPT — a single fresh pass, no --resume anywhere. One run, one taxonomy, eight sources — because two runs over the
# same rows share 0 class codes, so a source-by-source comparison must come from one run.
#
# 43,802 rows (ppl-pool5b mined 21,804). Profile `people_zh_v2`; v1 stays as the delivered
# ppl-pool5b run's profile. Routing: every Zhipu-assigned role goes to Moonshot/DeepSeek — see the
# header of configs/pool8_ppl.yaml for the measured refusals that rule came from.
#
# 为什么重跑而不是沿用 ppl-pool8：那个 run id 的交付代次 gen03 是**两次 resume** 的产物
# （gen03 首次启动是 --resume 进冷代次，p2c 撞到汇合点缺分支后崩掉，又 --resume 了一次），
# 而 HANDOFF §2 的 0t / 0v 明确写着 resume + fast 这条路上还有没关掉的缺陷。本次会话里也确实
# 逮到一个：`fast_skipped` 在 resume 时缩水，于是 gen03 的三份参考文档声称六个组件「跑过」。
# 其余七次运行（fin-pool8 / med-pool8 / film-pool8 / health-pool3 / ppl-pool5b / film-pool5 /
# health-pool2）全部是单次全新运行，人物域不该是唯一的例外。这一次：新 run id、空 llm_cache、
# 全程真调用、一次跑完、不 resume、不 reuse-taxonomy。
QM="/Users/mayouxuan/Documents/Claude/Search Query Mining Agent Team/QMine"
cd "$QM" || exit 1
export HF_HOME="$QM/.hf"
LOG="$QM/analysis/pooled5/work/driver_ppl8.log"
echo "$(date +%H:%M:%S) launching ppl-pool8b (domain people_zh_v2, corpus 人物8)" >> "$LOG"
.venv/bin/qmine run --config "configs/pool8_ppl.yaml" \
    --input "data/raw/pooled5/人物8_pooled5.parquet" \
    --domain people_zh_v2 --run-id ppl-pool8b --fast \
    > "$QM/analysis/pooled5/work/ppl-pool8b.launch.log" 2>&1
echo "$(date +%H:%M:%S) ppl-pool8b exited $?" >> "$LOG"
