# 影视8 语料构建实测（`build_pool8_corpus.py`，2026-09-16 10:54）

全部读入 44,000 行 → 入挖掘 43,933 行。四个与 `影视_pooled5.parquet` 共有的快照逐格相同：

  2025search: 9,998 行逐格相同
  2026search: 9,999 行逐格相同
  assistant_top: 955 行逐格相同
  assistant_random: 982 行逐格相同

## 每个快照

| domain | source | rows | kept | dropped | drop_% | pv_dropped_% |
|---|---|---|---|---|---|---|
| 影视8 | 2025search | 10000 | 9998 | 2 | 0.02 | 0.0 |
| 影视8 | 2025search_rand | 10000 | 10000 | 0 | 0.0 | 0.0 |
| 影视8 | 2026search | 10000 | 9999 | 1 | 0.01 | 0.01 |
| 影视8 | 2026search_rand | 10000 | 10000 | 0 | 0.0 | 0.0 |
| 影视8 | assistant_top | 1000 | 955 | 45 | 4.5 | 78.33 |
| 影视8 | assistant_random | 1000 | 982 | 18 | 1.8 | 1.66 |
| 影视8 | assistant_voice_top | 1000 | 999 | 1 | 0.1 | 1.42 |
| 影视8 | assistant_voice_random | 1000 | 1000 | 0 | 0.0 | 0.0 |

## 语音头部的边界

- 导出就是头部：1,000 行，按 PV 降序；最后一行 PV = 21，与它同 PV 的有 40 行。
- **这个边界是导出方切的，不是本仓库测出来的**：文件在 1,000 行处截断，行外是否还有同 PV 的行无从判断（医疗8 的语音导出有 1 万行，所以那次可以切在无并列的边界上）。

## 快照之间的重合（去重后的入挖掘串）

| 快照对 | 重合串数 |
|---|---:|
| 2025search∩2026search | 2,841 |
| 2026search∩assistant_voice_top | 699 |
| 2025search∩assistant_voice_top | 345 |
| 2026search∩assistant_top | 205 |
| 2025search∩assistant_top | 141 |
| assistant_top∩assistant_voice_top | 132 |
| 2025search∩2025search_rand | 19 |
| 2026search∩assistant_voice_random | 12 |
| 2025search_rand∩2026search | 9 |
| 2025search∩assistant_voice_random | 7 |
| assistant_voice_top∩assistant_voice_random | 4 |
| 2025search∩2026search_rand | 3 |
| 2026search∩2026search_rand | 2 |
| assistant_top∩assistant_voice_random | 2 |
| 2025search∩assistant_random | 1 |
| 2025search_rand∩assistant_top | 1 |
| 2025search_rand∩assistant_voice_top | 1 |
| 2026search∩assistant_random | 1 |

## 空文本丢弃

无
