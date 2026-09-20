# -*- coding: utf-8 -*-
"""Assemble docs/SEARCH_VS_ASSISTANT_2026.zh.md from draft parts, replacing {{T..}} with the code-generated tables
and {{NAME}} with scalars from report_vars.json. Fails loudly if any placeholder is left unreplaced."""
import json, os, re, sys
SP = __import__("os").environ.get("SVA_WORKDIR", __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "work"))
QM = __import__("os").path.abspath(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", ".."))
PARTS = ["report_head.md", "report_draft_A.md", "report_draft_B.md", "report_draft_C.md", "report_draft_D.md"]
TABLES = ["report_tables_v3.md", "report_tables_intent_v3.md"]
blocks = {}
for t in TABLES:
    p = f"{SP}/{t}"
    if not os.path.exists(p):
        continue
    cur = None
    for line in open(p, encoding="utf-8").read().splitlines():
        m = re.match(r"^### (T\d+[a-z]?)\s", line)
        if m:
            cur = m.group(1); blocks[cur] = []
            line = "#### " + line[4:]          # demote to fit under report sections
        if cur:
            blocks[cur].append(line)
blocks = {k: "\n".join(v).strip() for k, v in blocks.items()}
vars_ = json.load(open(f"{SP}/report_vars.json", encoding="utf-8")) if os.path.exists(f"{SP}/report_vars.json") else {}
for fn in sorted(os.listdir(SP)):                      # multi-paragraph values live in report_var_<NAME>.md
    m = re.match(r"^report_var_([A-Za-z0-9_]+)\.md$", fn)
    if m:
        vars_[m.group(1)] = open(f"{SP}/{fn}", encoding="utf-8").read().strip()
text = "\n\n".join(open(f"{SP}/{p}", encoding="utf-8").read().strip() for p in PARTS if os.path.exists(f"{SP}/{p}"))
def rep(m):
    k = m.group(1)
    if k in blocks: return blocks[k]
    if k in vars_: return str(vars_[k])
    return m.group(0)
text = re.sub(r"\{\{([A-Za-z0-9_]+)\}\}", rep, text)
left = sorted(set(re.findall(r"\{\{([A-Za-z0-9_]+)\}\}", text)))
out = f"{QM}/docs/SEARCH_VS_ASSISTANT_2026.zh.md"
if left and "--draft" not in sys.argv:
    raise SystemExit(f"unreplaced placeholders: {left}")
open(out if "--draft" not in sys.argv else f"{SP}/report_preview.md", "w", encoding="utf-8").write(text + "\n")
print("tables:", sorted(blocks), "| unreplaced:", left, "| chars:", len(text))
