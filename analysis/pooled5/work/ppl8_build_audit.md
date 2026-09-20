# 人物8 语料构建实测（`build_pool8_corpus.py`，2026-09-16 10:54）

全部读入 44,000 行 → 入挖掘 43,802 行。四个与 `人物_pooled5.parquet` 共有的快照逐格相同：

  2025search: 9,983 行逐格相同
  2026search: 9,982 行逐格相同
  assistant_top: 889 行逐格相同
  assistant_random: 950 行逐格相同

## 每个快照

| domain | source | rows | kept | dropped | drop_% | pv_dropped_% |
|---|---|---|---|---|---|---|
| 人物8 | 2025search | 10000 | 9983 | 17 | 0.17 | 0.12 |
| 人物8 | 2025search_rand | 10000 | 9999 | 1 | 0.01 | 0.47 |
| 人物8 | 2026search | 10000 | 9982 | 18 | 0.18 | 0.14 |
| 人物8 | 2026search_rand | 10000 | 10000 | 0 | 0.0 | 0.0 |
| 人物8 | assistant_top | 1000 | 889 | 111 | 11.1 | 33.43 |
| 人物8 | assistant_random | 1000 | 950 | 50 | 5.0 | 7.4 |
| 人物8 | assistant_voice_top | 1000 | 999 | 1 | 0.1 | 0.33 |
| 人物8 | assistant_voice_random | 1000 | 1000 | 0 | 0.0 | 0.0 |

## 语音头部的边界

- 导出就是头部：1,000 行，按 PV 降序；最后一行 PV = 10，与它同 PV 的有 142 行。
- **这个边界是导出方切的，不是本仓库测出来的**：文件在 1,000 行处截断，行外是否还有同 PV 的行无从判断（医疗8 的语音导出有 1 万行，所以那次可以切在无并列的边界上）。

## 快照之间的重合（去重后的入挖掘串）

| 快照对 | 重合串数 |
|---|---:|
| 2025search∩2026search | 4,493 |
| 2026search∩assistant_voice_top | 691 |
| 2025search∩assistant_voice_top | 642 |
| 2025search∩assistant_top | 399 |
| 2026search∩assistant_top | 397 |
| assistant_top∩assistant_voice_top | 206 |
| 2025search∩assistant_voice_random | 15 |
| 2026search∩assistant_voice_random | 15 |
| 2025search∩2025search_rand | 14 |
| 2026search∩2026search_rand | 10 |
| 2025search∩2026search_rand | 8 |
| 2025search_rand∩2026search | 7 |
| 2026search∩assistant_random | 5 |
| 2025search∩assistant_random | 4 |
| assistant_voice_top∩assistant_voice_random | 4 |
| 2026search_rand∩assistant_top | 1 |

## 空文本丢弃

无
