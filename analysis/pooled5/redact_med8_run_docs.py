"""Guardrail copies of the med-pool8 run's own reference documents (post-run operation on results).

Same method as `redact_health_run_docs.py`, keyed to 医疗8. The program's top-down and bottom-up
definition documents print raw sample strings (rule examples, template exemplars, centre/edge samples),
and the pipeline does not enforce never-quote: `screen_risk` keeps 8 samples + an exemplar per category
(measured while writing `configs/domains/medical_zh_v2.yaml`). This writes `<name>_引文护栏版.md` into
`postprocessed/` with ONLY query-like spans replaced (backtick spans, 「」 spans, table cells of <= 40
characters); descriptive prose is kept, the originals in `gen01/` are not touched, and it asserts that
nothing query-like still matches afterwards.

What counts as guarded here:
- the report's hard-rule and co-occurrence layers, and `EXTRA_QUOTE_BLOCK["医疗8"]` (the health block plus
  explicit sexual-act wording, named-clinician shapes and partner + act co-occurrence);
- rows matched by the profile's never-quote patterns, read from `medical_zh_v2.yaml` itself:
  `minors_and_sexual_content`, `body_reshape_instruction`, `self_harm_and_crisis` (its two patterns reproduce the 13 rows
  the program's own screen flagged), and reproductive_and_sensitive's explicit-act line;
- strings the run's own risk sentinel named under a minors / sexual-content / self-harm finding (med-pool8's sentinel
  named a self-harm method request inside an ordinary first-aid cluster), and the screen samples of the never-quote categories.

Governance rationales are a second leak path: med-pool8's tree auditor quoted a self-harm method query and an adult title
inside curly quotes in the P013/P014 rationale cells (long table cells the span rules never reached). So “…” spans are
redacted too, the curly-quoted spans of governance rationales that mention self-harm / adult content are added to the named
set, and any blocked corpus string of 4+ characters is replaced wherever it appears inside a line.

It also checks, on a run produced after the 2026-09-15 source fix, that a printed `p5_k_references_agree`
gate row names the same deciding reference as `granularity.json` — the defect the health copy had to annotate.
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd
import yaml

QM = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(QM / "analysis/pooled5"))
from p5_snapshot_classes import EXTRA_QUOTE_BLOCK, QUOTE_BLOCK, QUOTE_COOC, screened_quote_block  # noqa: E402

DOMAIN = "医疗8"
GEN = QM / "runs/med-pool8/gen01"
MARK = "（不引原文）"
pat = EXTRA_QUOTE_BLOCK[DOMAIN]
_rx = [x if hasattr(x, "search") else re.compile(x) for x in (QUOTE_BLOCK, *QUOTE_COOC)]
hard, cooc_a, cooc_b = _rx

_prof = yaml.safe_load((QM / "configs/domains/medical_zh_v2.yaml").read_text(encoding="utf-8"))
_cats = {c["name"]: c for c in _prof["risk_categories"]}
_rep4 = _cats["reproductive_and_sensitive"]["patterns"][-1]
assert "夫妻生活" in _rep4 and "性交(?:" in _rep4, "reproductive_and_sensitive's last line is no longer the explicit-act line"
NEVER_PATTERNS = ([re.compile(p) for p in _cats["minors_and_sexual_content"]["patterns"]]
                  + [re.compile(p) for p in _cats["body_reshape_instruction"]["patterns"]]
                  + [re.compile(p) for p in _cats["self_harm_and_crisis"]["patterns"]]
                  + [re.compile(_rep4)])
NEVER_CATEGORIES = {"minors_and_sexual_content", "body_reshape_instruction", "self_harm_and_crisis", "reproductive_and_sensitive"}

_PRIVACY_FINDING = re.compile(r"未成年|性化|色情|成人内容|露骨|私密影像|性行为|身体形态|自伤|自残|自杀|轻生")
named: set[str] = set()
_tn_path = GEN / "tree_naming.json"
if _tn_path.exists():
    _tn = json.loads(_tn_path.read_text(encoding="utf-8"))
    for _f in (_tn.get("risk_report") or {}).get("findings", []) or []:
        if _PRIVACY_FINDING.search(str(_f.get("category") or "") + str(_f.get("rationale") or "")):
            named.update(str(x).strip() for x in (_f.get("evidence") or []))
for _c in json.loads((GEN / "risk_screen.json").read_text(encoding="utf-8")).get("categories", []):
    if _c.get("name") in NEVER_CATEGORIES:
        for x in ([_c["exemplar"]] if _c.get("exemplar") else []) + list(_c.get("samples") or []):
            x = str(x).strip()
            # reproductive samples are mostly ordinary health questions: only its explicit-act line is never-quote
            if _c["name"] != "reproductive_and_sensitive" or re.search(_rep4, x):
                named.add(x)
_GOV_PRIVACY = re.compile(r"未成年|色情|成人内容|成人向|露骨|性行为|自伤|自残|自杀|轻生|致病|患上")
for _e in json.loads((GEN / "governance.json").read_text(encoding="utf-8")).get("ledger", []) or []:
    _r = str(_e.get("rationale") or "")
    if _GOV_PRIVACY.search(_r):
        named.update(x.strip() for x in re.findall(r"“([^”\n]{2,80})”", _r))
named = {x for x in named if x}


def guarded(t: str) -> bool:
    return bool(pat.search(t) or hard.search(t) or (cooc_a.search(t) and cooc_b.search(t))
                or any(p.search(t) for p in NEVER_PATTERNS) or t in named)


corpus = pd.read_parquet(QM / f"data/raw/pooled5/{DOMAIN}_pooled5.parquet")["query"].astype(str)
blocked = {q for q in set(corpus) if guarded(q)} | named | screened_quote_block(DOMAIN)
long_blocked = sorted((q for q in blocked if len(q) >= 4), key=len, reverse=True)


def bad(s: str) -> bool:
    t = s.strip().strip("`").strip()
    return bool(t) and (guarded(t) or t in blocked or any(q in t for q in long_blocked))


_AGE_PAREN = re.compile(r"[（(][^）)\n]*\d{1,2}\s*岁[^）)\n]*[）)]")
# A governance or sentinel rationale can paraphrase a self-harm method in plain prose (med-pool8 P016). This replaces only
# the method phrase; the class definition wording ("主动索取让自己患病或受伤的方法") has no 弄/搞 and is kept.
_SELF_HARM_METHOD = re.compile(r"(?:如何|怎么|怎样)?(?:将|把)?自己的?.{0,4}(?:弄|搞)(?:伤|断|骨折|残)")


def redact(line: str) -> tuple[str, int]:
    n = 0

    def span(m, left, right):
        nonlocal n
        if bad(m.group(1)):
            n += 1
            return f"{left}{MARK}{right}"
        return m.group(0)

    line = re.sub(r"`([^`\n]+)`", lambda m: span(m, "`", "`"), line)
    line = re.sub(r"「([^」\n]+)」", lambda m: span(m, "「", "」"), line)
    line = re.sub(r"“([^”\n]+)”", lambda m: span(m, "“", "”"), line)
    for qq in long_blocked:
        if qq in line:
            n += line.count(qq)
            line = line.replace(qq, MARK)
    if pat.search(line) or any(p.search(line) for p in NEVER_PATTERNS):
        line, k = _AGE_PAREN.subn("（具体描述略）", line)
        n += k
    line, k = _SELF_HARM_METHOD.subn(MARK, line)
    n += k
    if line.lstrip().startswith("|"):
        cells = line.split("|")
        for i, c in enumerate(cells):
            if MARK not in c and 0 < len(c.strip()) <= 40 and bad(c):
                cells[i] = f" {MARK} "
                n += 1
        line = "|".join(cells)
    return line, n


def residual_query_like(line: str) -> bool:
    spans = re.findall(r"`([^`\n]+)`", line) + re.findall(r"「([^」\n]+)」", line) + re.findall(r"“([^”\n]+)”", line)
    if any(qq in line for qq in long_blocked) or _SELF_HARM_METHOD.search(line):
        return True
    cells = line.split("|") if line.lstrip().startswith("|") else []
    return any(bad(s) for s in spans) or any(0 < len(c.strip()) <= 40 and bad(c) for c in cells)


def main() -> int:
    out_dir = GEN / "postprocessed"
    out_dir.mkdir(exist_ok=True)
    tri = json.loads((GEN / "granularity.json").read_text(encoding="utf-8"))["triangulation"]
    names = sorted(p.name for p in GEN.glob("*_自上而下_意图体系完整定义.md")) + \
        sorted(p.name for p in GEN.glob("*_自下而上_聚类树完整定义.md"))
    if (GEN / "分层对比_头尾结构差异.md").exists():
        names.append("分层对比_头尾结构差异.md")
    assert names, "no definition documents found in the run directory"
    print(f"blocked corpus strings: {len(blocked)} | run-named privacy strings: {len(named)}")
    for name in names:
        src = GEN / name
        lines = src.read_text(encoding="utf-8").split("\n")
        # the post-fix gate row must agree with granularity.json (no annotation should be needed any more)
        for ln in lines:
            m = re.search(r"`deciding_reference`=([^\s|;]+)", ln)
            if m and "p5_k_references_agree" in ln:
                shown = m.group(1)
                want = tri.get("deciding_reference") if tri.get("locator", "").startswith(("intent_alignment_ami", "ami_vs_")) else "无"
                assert shown == str(want), f"{name}: gate row shows {shown!r}, granularity says {want!r}"
                print(f"{name}: gate row deciding_reference={shown} agrees with granularity.json")
        total, new = 0, []
        for ln in lines:
            r, n = redact(ln)
            total += n
            new.append(r)
        assert not any(residual_query_like(ln) for ln in new), f"{name}: a query-like span still matches"
        note = (f"> **引文护栏版。** 本文件由 `{name}` 生成：命中医疗8 不可引护栏（未成年人与性相关、露骨性描述、改动人体形态的指令、"
                f"自伤与危机、具名医生）的 {total} 处样例原文替换为「{MARK}」，只替换查询样例，不改任何数字与定义；原件保留在运行目录 "
                f"`runs/med-pool8/gen01/`。")
        new.insert(1, "\n" + note + "\n")
        dst = out_dir / (src.stem + "_引文护栏版.md")
        dst.write_text("\n".join(new), encoding="utf-8")
        print(f"{name}: replaced {total} span(s) -> {dst.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
