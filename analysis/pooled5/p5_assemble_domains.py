# -*- coding: utf-8 -*-
"""Assemble the five verified domain deep-dives into one companion document."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pooled5_common import WORK

ROOT = Path(__file__).resolve().parents[2]
ORDER = [("金融", "fin"), ("医疗", "med"), ("教育", "edu"), ("影视", "film"), ("人物", "ppl")]
HEAD = """# 2026 五域同体系对比 —— 分领域深挖

> 本文是主报告《[POOLED5_2026_五域同体系对比](POOLED5_2026_五域同体系对比.zh.md)》的全文附件。每个领域由一位独立分析员完成，
> 再由另一位复核员**逐条重算**全部数字（不采信正文，也不采信分析员自己的 how_computed），最后按复核意见修订。
> 每节末尾的「修订记录」列出了被改掉或删掉的结论。
>
> 口径与主报告一致：占比一律为**来源内占比**；显著性以 Newcombe 95% 区间不含 0 为准；<4 个百分点的差异只作方向性描述；
> AMI 的参照系是同一份数据跑两次的 0.774/0.687，不是 1.0；语义近邻的「完全相同」是余弦 ≥0.999，不等于字符串相同。
"""


def main() -> None:
    parts = [HEAD]
    for zh, key in ORDER:
        p = WORK / f"dom_{key}.md"
        if not p.exists():
            parts.append(f"\n---\n\n# {zh}\n\n（本领域的分析尚未完成）\n")
            continue
        body = p.read_text(encoding="utf-8").strip().splitlines()
        if body and body[0].startswith("#"):
            body = body[1:]
        body = "\n".join(("#" + ln) if ln.startswith("#") else ln for ln in body)
        parts.append(f"\n---\n\n# {zh}\n\n{body.strip()}\n")
    out = ROOT / "docs/POOLED5_2026_领域深挖.zh.md"
    out.write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {out} — {sum(len(x) for x in parts):,} chars")


if __name__ == "__main__":
    main()
