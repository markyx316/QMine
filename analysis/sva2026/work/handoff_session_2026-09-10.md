

---

## Session 2026-09-10 — 2026 search vs AI assistant: cleaning, one-instrument intent, report

**Delivered (post-run analysis only; no pipeline source changed).**

- **Reports:**
  - `docs/SEARCH_VS_ASSISTANT_2026.zh.md` (main report, §0–§9 plus appendices A–E)
  - `docs/SEARCH_VS_ASSISTANT_2026_领域深挖.zh.md` (five domain deep-dives)
  - figures `docs/img/sva2026/fig1–fig6`
- **Correction box** added to `docs/DRIFT_2025_2026_WITH_ASSISTANT.zh.md` §3.3. Its "15 comparisons all point the same way" mixed search-head rows with assistant-tail rows. At matched depth both surfaces have a median query length of 6.
- **Reproduction package:** `analysis/sva2026/` holds 86 scripts and `work/` (48 MB derived data). The exported `sva_report_tables.py` and `sva_report_tables_intent.py` regenerate the inserted tables byte-for-byte.

**New tools** (not imported by tests; `ruff --select F src/qmine/ tools/` is clean):

- **`tools/clean_assistant_functional.py` v3.** Tiers are S1 repeated, S2 template/feature, S3 card, S4 suggested chip, S5 headline, C1 content-free, C2 feed-control.
  - First audit (on v2): removal precision was sound (S1 0.964, S2 0.851, C1 0.966, C2 0.998), but 11.3% of kept head rows were still system text.
  - Second audit (fresh samples): kept-head miss rate 7.0% [4.6, 10.5], with 人物 head at 20.0%. Precision is 96.7% for new removals, 96% for S4 and 90% for S5.
  - A census found 51.6% of kept head U05 rows are untagged headlines or topic strings. Corrected, pooled head U05 drops from 3.95% to 2.05%.
- **`tools/unified_intent_frame.py`:** 13-class frame plus a crosswalk covering all 112 classes of the six runs.
- **`tools/label_unified_intent.py`:** blind, shuffled labelling with one prompt. Primary model is deepseek-v4-flash at `max_tokens=16000` with a preflight batch; qwen3.8-flash labels a 20% subset.

**Measured, and worth not re-learning.**

- **Crosswalk shares are not valid across surfaces.** Crosswalk and unified labels agree on only 35.9% of assistant head rows, against 76.6% for search head. For example, ai04 "投资理财建议" is 69.5% bare tickers and quotes in 金融.
- **Reliability.**
  - DeepSeek vs Qwen: κ 0.812 (n=2,815); assistant tail 0.761.
  - Test-retest on 1,000 anchors with different batch neighbours: κ 0.850.
  - All 32 of 32 intent differences ≥3pp have the same direction under both models.
- **Search PV floors dwarf the assistant's.** The search 10,000th query (158–477 PV) outranks the assistant top1k floor (44–280). So search top10k is labelled too and reported as a second lens. Search intent mix shifts a lot between top1000 and top10k in 教育 (U02 77% → 49%) and 医疗.
- **The assistant random1k has no search counterpart.** 56% of its rows have no search top10k neighbour with cosine ≥0.70, against 5–28% for search's own deepest 1,000 rows; the gap holds after length control.

**Defects found and fixed in the process.**

1. **pandas 3 string regex runs through RE2** (`\w` is ASCII-only). The audit-suggested `[^\W_]` read every Chinese query as empty, and 81.7% of head rows went C1. Fixed with `str.isalnum` plus `_check_engine_semantics()`. Memory `qmine-pandas3-re2-regex`.
2. **S5 `来了$` caught a variety-show title** (`爸爸回来了`). The rule now requires ≥7 characters.
3. **Label resilience.** Labels were reconstructable from `raw_ds.jsonl`: the reconstruction matches the delivered labels 14,452/14,452, so a failing second-model pass cannot lose paid primary labels.

**Open, deliberately not resolved.**

- **Audit-2 rules not applied.** They are validated on held-out rows but would desynchronise every table from the domain deep-dives, so they belong in a v4 for the next data: `有没有更多` chips, `#…#` feed copy, the `我想对作文《` template, the S5 news-keyword exemption, and three search-S5 exemptions.
- **Residues known in v3.** 影视 has 31 `《X》的结局是什么 / 给我《X》的完整演员表` rows. 金融 has 10 identical `XX未来有上涨空间吗` rows (PV 44–76). 人物 head U05 topic strings are handled by the census correction, not a rule.
- **Same-period data** (search and assistant ~2 months apart) and a typed-vs-tapped source field in the assistant log are the two things that would remove the report's largest confounds.

**Tests:** {{TESTS}}
