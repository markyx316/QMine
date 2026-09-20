# -*- coding: utf-8 -*-
"""Assemble the POOLED-5 report from parts. `{{Tn}}` is replaced by the matching block of
`work/report_tables.md`, `{{NAME}}` by `work/report_var_<NAME>.md` or a key of
`work/report_vars.json`. Any placeholder left unreplaced is an error, not a blank.

    python analysis/pooled5/p5_assemble_report.py
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pooled5_common import WORK

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/POOLED5_2026_五域同体系对比.zh.md"
PARTS = ["report_head.md", "report_A.md", "report_B.md", "report_C.md", "report_D.md"]


def tables() -> dict[str, str]:
    p = WORK / "report_tables.md"
    if not p.exists():
        return {}
    blocks, cur, name = {}, [], None
    for line in p.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^#{3,4} (T\d+[a-z]?)\s", line)
        if m:
            if name:
                blocks[name] = "\n".join(cur).strip()
            name, cur = m.group(1), [line]
        elif name:
            cur.append(line)
    if name:
        blocks[name] = "\n".join(cur).strip()
    return blocks


def main() -> None:
    text = "\n\n".join((WORK / p).read_text(encoding="utf-8").strip() for p in PARTS if (WORK / p).exists())
    tb = tables()
    varsj = json.loads((WORK / "report_vars.json").read_text(encoding="utf-8")) if (WORK / "report_vars.json").exists() else {}

    def repl(m: re.Match) -> str:
        k = m.group(1)
        if k in tb:
            return tb[k]
        f = WORK / f"report_var_{k}.md"
        if f.exists():
            return f.read_text(encoding="utf-8").strip()
        if k in varsj:
            return str(varsj[k])
        return m.group(0)

    text = re.sub(r"\{\{([A-Za-z0-9_]+)\}\}", repl, text)
    left = re.findall(r"\{\{([A-Za-z0-9_]+)\}\}", text)
    if left:
        raise SystemExit(f"unreplaced placeholders: {sorted(set(left))}")
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT} — {len(text):,} chars, tables used: {sorted(tb)}")


if __name__ == "__main__":
    main()
