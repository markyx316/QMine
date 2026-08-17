---
name: qmine-new-domain
description: Build a QMine domain profile for a new vertical (finance, healthcare, travel, gaming, a new language) — phrasing seeds, risk categories, tokenizer and n-gram settings, expected taxonomy size. Use when the user wants to run query mining on a corpus that none of the bundled profiles fits.
---

# Writing a domain profile

A domain profile is the pile of settings the playbook says **must be re-derived
per domain**. Everything else — the twelve phases, the metric authorities, the
blind-review protocol, the hybrid formula — transfers unchanged.

## What must be re-derived, and what must not

| re-derive for every domain | copy verbatim |
|---|---|
| phrasing seed patterns | the twelve phases |
| risk categories | metric authority table |
| tokenizer and n-gram ranges | the α² algebra |
| encoder candidates | blind-naming protocol |
| expected taxonomy size | governance execution rule |
| **α** (re-run the sweep — never inherit) | uniform panel contract |

The most common mistake is copying `alpha: 0.1` from the K12 profile. That value
is a fact about K12's phrasing ecology, not a constant. The sweep re-derives it
in about a minute; inheriting it can cost a whole tree.

## Procedure

**1. Look at the data before writing anything.**

```bash
python -c "
import pandas as pd, collections, re
df = pd.read_csv('QUERIES.csv')
q = df['query'].astype(str)
print(q.str.len().describe())
c = collections.Counter()
for s in q.head(30000):
    for n in (2,3,4):
        for i in range(len(s)-n+1): c[s[i:i+n]] += 1
print([w for w,_ in c.most_common(60)])
"
```

**2. Write 5–8 phrasing seeds.** The contract is strict: *everything matching
this pattern is almost certainly the same intent*. If you can imagine two
different intents matching, the pattern is too loose. Target 20–40% union
coverage — Phase 1 gates on it, and both ends of that window are failure modes.

**3. Write the risk categories.** Ask what a naive answer would cost, not how
much traffic it is. Every vertical has its version of the same problem:

| domain | the category that must never blend in |
|---|---|
| finance | stock tipping, requests for individualised advice |
| health | symptom-to-diagnosis, dosage requests |
| sports | betting-market probes |
| education | gambling probes wearing riddle clothing |
| news | rumour amplification, locally sensitive topics |

For jurisdiction-specific categories, leave the patterns **empty** with a comment
saying they need local counsel. An empty list with a reason is honest; a copied
list from another market is worse than nothing.

**4. Set language mechanics.** CJK: `jieba`, char 1–3 grams. Latin: whitespace,
char 3–5 grams plus word 1–2 grams. Mixed-language: BGE-M3 or Qwen3-Embedding,
and audit by language before deciding whether to build one tree or several.

**5. List the pragmatic intents.** The categories clustering will be blind to —
where two queries are phrased alike and want different things. These get
assigned to the top-down route *before* work starts, because nothing downstream
will discover them.

## Then validate

```bash
qmine run --input QUERIES.csv --domain configs/domains/yours.yaml --sample 8000 --fast
```

Check three things: template coverage in the 20–40% window, the risk pre-screen
finding roughly what you expected, and the α sweep converging on something other
than the grid edge. An α at the boundary means the grid is wrong, not that the
answer is 0.5.

Use `configs/domains/finance_zh.yaml` as the reference — it is the most heavily
commented, and the comments explain *why* each choice was made.
