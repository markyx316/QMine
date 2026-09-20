# Brief for domain deep-dive agents — 2026 search vs AI assistant

## Environment
- Project: `/Users/mayouxuan/Documents/Claude/Search Query Mining Agent Team/QMine`
- Run Python ONLY as: `cd "/Users/mayouxuan/Documents/Claude/Search Query Mining Agent Team/QMine" && HF_HOME=$(pwd)/.hf .venv/bin/python -W ignore -` (heredoc). pandas, pyarrow, jieba are installed.
- Scratchpad: `/private/tmp/claude-501/-Users-mayouxuan-Documents-Claude-Search-Query-Mining-Agent-Team/3deba04e-73b1-4747-b7b7-4d1a40520650/scratchpad` (below: `<SP>`). Write working files ONLY here, ONLY with your assigned filename prefix. Do NOT modify any project file.
- Shared helpers: `<SP>/sva_common.py` (import after `sys.path.insert(0, "<SP>")`).

## The data
Both surfaces are from 2026. Search: five verticals (金融/医疗/教育/影视/人物), each the top 10,000 queries by page views (`wise_pv`) on 2026-07-01, labelled by that vertical's own run. Assistant: a Baidu-family AI assistant log (users mention 百度/文心/小度; part is voice), 33 topical categories × two strata: `top1k` = each category's top 1,000 by page views (`search_num`); `random1k` = a uniform random 1,000 distinct queries per category (median PV 1). The assistant categories 金融/医疗/教育培训/影视动漫/人物 share a topic with the five search verticals.

**Analysis table** (built after labelling): `<SP>/sva_final_rows.parquet` — one row per (domain, surface, query) with: `domain`, `surface` ∈ {`search_top1000`, `assistant_top1k`, `assistant_random1k`}, `query`, `pv`, `rank` (search only), `tier` (final cleaning tier; analyse `tier == "user"` unless told otherwise), `u_ds`/`p_ds` (single-instrument intent code U01–U13 and personal-situation flag from deepseek-v4-flash, blind to surface), `u_qw`/`p_qw` (same from qwen3.8-flash on a 20% subset — for agreement only), `own_intent` (the label from that surface's own run taxonomy, Chinese name), `own_leaf` (bottom-up leaf name).
Also: `<SP>/clean_v2/assistant_tiered.parquet`, `<SP>/clean_v2/search2026_tiered.parquet` (all rows, all tiers), and search `top10k` via `sva_common.cells()`.

**Unified intent frame (U01–U13)** — definitions in `tools/unified_intent_frame.py` (`FRAME`): U01 导航直达, U02 裸实体, U03 事实与数值, U04 解释与介绍, U05 核实与动态, U06 办事与操作, U07 个案判断与建议, U08 清单与推荐, U09 获取现成内容, U10 生成与编辑, U11 会话与系统指令, U12 违规或灰色内容, U13 无法判定.

**Cleaning tiers** (see `tools/clean_assistant_functional.py` docstring): `S1_system_repeated` (chips, suggested follow-ups, tapped headlines), `S2_system_template` (templates, tool buttons, named feature entries), `S3_card_passage` (encyclopedia card text + templated question), `C1_content_free`, `C2_feed_control`, `user`.

## Comparison design (non-negotiable)
- **Matched depth**: compare `search_top1000` with `assistant_top1k` — both are "the most-trafficked ~1,000 queries of this topic on this surface". `assistant_random1k` describes the assistant's long tail, which search data CANNOT show (the search export's floor is 158–477 PV; >99.98% of assistant tail rows sit below it). Never present a head-vs-tail difference as a surface difference.
- Intent shares come from `u_ds` (one instrument). Each run's own labels (`own_intent`) describe that surface in its own terms and supply examples, but their SHARES are not comparable across surfaces (different taxonomies, different granularity).
- Report `n` for every share and a Wilson 95% CI for every share you state (compute in code).

## Already-computed cross-domain evidence (read these first; recompute anything you quote for your domain)
All files in `<SP>`; all on final tiers, user rows only, matched surfaces as above.
- `sva_metrics_v2.txt` — length, question/imperative/first-person/persona/navigation/resource-word rates by domain × surface. Regexes in `sva_metrics.py`.
- `sva_question_specificity.txt` — question subtypes (因果/方法/是非/可行/选择/数量/定义), specificity (numbers+units, dates, places, named A-vs-B), and a rank-band control inside search.
- `sva_conv_markers.txt` — conversational markers (follow-up openers, references to earlier turns, talking to the assistant, delegation openers, output constraints) and task verbs. Script: `sva_conv_markers.py`.
- `sva_behaviour_markers.txt` — image-referencing, vetting non-celebrities, events/gossip, homework, personal health/finance situations, hypotheticals, pasted material.
- `sva_overlap_v2.txt` — exact overlap and "wrapping" (a search query embedded inside a longer assistant query).
- `sva_logodds.txt` — distinctive vocabulary (weighted log-odds, jieba doc-frequency) and vocabulary coverage.
- `sva_semantic_nn.parquet` / `.txt` — for each assistant user query, its nearest search top-10k query in the same domain (bge-base-zh-v1.5 cosine `sim`, `nn_query`, `nn_search_rank`), with within-search controls.
- `sva_arrival_voice_brand.txt` — tier composition of assistant head traffic, voice signals, brand mentions.

## Your deliverable
Write `<SP>/sva_dom_<英文前缀>.md` in Chinese with exactly these sections (keep each tight, evidence-first):
1. **数据与清洗影响** — n per surface per tier; the share of head PV each tier removed; 3–5 concrete removed examples.
2. **同深度查询形态** — 6–10 measurements that matter most for THIS domain (search_top1000 vs assistant_top1k, with assistant_random1k as the tail), each with n and a real example pair.
3. **意图对比** — `u_ds` shares with Wilson CIs for all three surfaces; the 3–5 largest shifts; for each shift, 2–3 real queries per side. Then each surface's OWN taxonomy (`own_intent`): the top classes and their shares within that surface, and how the two taxonomies' slicing differs (what one names that the other has no word for).
4. **同一需求，两种问法** — 8–12 matched pairs (same entity or need; search query ↔ assistant query), grouped by what changed (added constraint / added person / asked for judgement / asked for generation …). Use `sva_semantic_nn.parquet` and entity string matching to find them.
5. **助手才有的需求** — what assistant queries have no close search counterpart (low `sim`), sized with n and shares, and grouped into 3–6 themes with examples.
6. **用户画像与使用方式推断** — who seems to be asking on each surface and why they chose it. Every inference names its evidence (a number) and what would falsify it.
7. **对产品/运营的启示** — 3–5 bullets, each tied to a number above.
Also save any per-domain table you compute as `<SP>/sva_dom_<英文前缀>_*.csv`. End your final reply with the 5 most important findings (one line each, with numbers).

## Rules
1. Every number you report must be computed by you in code in this task. Put the exact filter you used next to each table. No numbers from memory.
2. Examples must be real rows. Do NOT quote rows that contain a private individual's personal details (names of non-public people with personal circumstances, phone numbers, IDs), explicit sexual text, or anything involving minors and sexuality — describe such rows generically instead. Public figures' public roles are fine.
3. Distinguish a measurement from an interpretation. When you infer WHY users behave differently, say what evidence supports it and what would falsify it.
4. Write in Chinese (the final report is Chinese). Keep tables compact.
