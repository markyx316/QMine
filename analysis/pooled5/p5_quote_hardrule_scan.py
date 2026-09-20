#!/usr/bin/env python
"""报告里印出来的真实 query，只按**硬规则**筛一遍：未成年人与性、露骨性内容、普通个人的可识别信息。

与 医疗8 那一套的区别（重要）：med-pool8 的做法是「逐串读 + 两名筛查员 + 拦下就换例子」，最终拦了 634 串，
代价是 10 个小格子只剩占位。人物 / 影视两域**要印真实 query**（用户 2026-09-16 明确要求），而且这两域的
主题本身就是公众人物与作品名——把人名、机构名、作品名一律拦掉会把报告掏空。所以这里不做广谱拦截，只做三条硬规则：

    MINOR_SEX   未成年人线索 + 性 / 性器官 / 性行为 / 性化描述
    EXPLICIT    露骨性行为或色情作品式写法（影视域的三级片、未删减、色情站引流等）
    PRIVATE_PII 普通个人（非公众人物）的可识别信息：手机号、身份证、住址、学籍、工号、车牌

命中不会自动拦下：脚本把命中行打印出来**交给人看**（数量少到可以逐条读），确认属实的串才写进
`p5_snapshot_classes.SCREENED_QUOTE_BLOCK` 指向的名单文件。这样既守住硬规则，又不牺牲例子的真实性。

    P5_COHORT=ppl8 HF_HOME=$(pwd)/.hf .venv/bin/python analysis/pooled5/p5_quote_hardrule_scan.py 人物8

输出：`work/<域>/quote_hardrule_hits.csv`（命中串、规则、来源分布、是否已被现有护栏拦下）。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

QM = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(QM / "analysis/pooled5"))
from pooled5_common import WORK, load, run_dir  # noqa: E402
from p5_intent_structure import quotable_mask  # noqa: E402

SPAN = re.compile(r"[「『“]([^」』”\n]+)[」』”]")

#: 未成年人线索（阿拉伯数字与中文数字年龄、学段、儿童称谓）与性相关词的**共现**才算。
MINOR = (r"(?:(?<![0-9])(?:[1-9]|1[0-7])岁|[一二三四五六七八九十]{1,3}岁|未成年|初中|小学|中学|高中|学生妹|"
         r"幼儿|儿童|小孩|女孩|男孩|少女|少年|萝莉|正太)")
SEXUAL = (r"(?:性行为|性交|做爱|口交|肛交|自慰|手淫|裸体|裸照|脱衣|内裤|内衣|胸部|乳房|阴道|阴茎|私处|下体|"
          r"发生关系|开房|援交|卖淫|嫖|色情|情色|黄片|三级片|成人片|涩涩|福利姬|擦边)")
RULES = {
    "MINOR_SEX": re.compile(rf"(?=.*{MINOR})(?=.*{SEXUAL})"),
    "EXPLICIT": re.compile(r"三级片|色情片|黄片|成人电影|av号|无码|里番|工口|情色电影|福利姬|擦边视频|"
                           r"做爱|口交|肛交|自慰|手淫|裸聊|约炮|一夜情细节|性爱视频|偷拍(?:裙底|厕所|浴室)"),
    # 普通个人的可识别信息。公众人物的公开身份（职务、作品、籍贯）不算——那是这两个垂类的正题。
    "PRIVATE_PII": re.compile(r"(?<![0-9])1[3-9][0-9]{9}(?![0-9])|"            # 手机号
                              r"(?<![0-9A-Za-z])[1-9][0-9]{5}(?:19|20)[0-9]{2}(?:0[1-9]|1[0-2])(?:[0-2][0-9]|3[01])[0-9]{3}[0-9Xx](?![0-9A-Za-z])|"  # 身份证
                              r"[一-鿿]{2,}(?:小区|花园|公寓|村)[0-9]{1,3}(?:栋|号楼|单元|室)|"  # 住址
                              r"(?:身份证号|手机号码是|电话号码是|家庭住址|工号是|学号是|车牌号)"),
}


#: 医疗随机只多一条规则：**具名的个人医生**。医疗垂类里「某科 + 姓名 + 医生」是可识别到个人的信息，
#: 用户定的规则里明确不引。第五层（EXTRA_QUOTE_BLOCK["医疗随机"]，沿用 医疗8 的正则）已经会自动拦，
#: 这里再扫一遍是**复核**：如果这条在扫描里命中、而「已被护栏拦下」是 False，说明第五层漏了。
_GENERIC_DOC = (r"(?:中医|西医|牙科|儿科|眼科|皮肤科|妇科|男科|外科|内科|骨科|口腔|家庭|社区|私人|心理|值班|主治|实习|"
                r"乡村|全科|专科|兽医|宠物|中西医|男医|女医|好的|优秀|执业|助理|住院|门诊|急诊|在线|网上|医院|首席|主任)")
DOMAIN_RULES = {d: {"NAMED_DOCTOR": re.compile(
    rf"(?:科|院)[一-鿿]{{2,3}}(?:医生|大夫)$|^(?!{_GENERIC_DOC}(?:医生|大夫)$)[一-鿿]{{2,3}}(?:医生|大夫)$|"
    r"(?:医院|保健院|卫生院|中心|诊所).{0,10}(?:主任医师|副主任医师|主治医师|住院医师|医师|教授|专家).{0,8}医生$")}
                # 医疗随机 与 医疗3 是同一批数据（后者多一个同产品的头部快照），同一条复核规则。
                for d in ("医疗随机", "医疗3")}


def scan(domain: str) -> int:
    # 代次跟着 `run_dir` 走，不写死 gen01。写死的后果实测过：人物域的交付代次是 gen03，
    # `runs/ppl-pool8/gen01/postprocessed/` 根本不存在，于是 `printed` 是空集合，
    # 「已印进交付文档 0」不是一个测量结果，而是一句空话——沉默被当成了通过。
    gen = run_dir(domain)
    d = load(domain).reset_index(drop=True)
    q = d["query"].astype(str)
    quotable = quotable_mask(domain, d).to_numpy()
    texts = {p.name: p.read_text(encoding="utf-8") for p in (gen / "postprocessed").glob("*.md")}
    # 空集合必须报错，不能静悄悄地得出「一个都没印」。这一层的全部价值就在于它读的是**真正印出来的串**。
    assert texts, (f"{domain}: {gen / 'postprocessed'} 里没有任何 .md —— 先跑完后处理再扫，"
                   f"否则「已印进交付文档」这一列是空话")
    printed = set()
    for t in texts.values():
        printed |= set(SPAN.findall(t))
    wb = gen / "postprocessed" / f"{sorted(texts)[0].split('_意图')[0]}_意图与聚类叶_跨快照对比.xlsx" if texts else None
    if wb and wb.exists():
        for df in pd.read_excel(wb, sheet_name=None, header=None).values():
            printed |= {str(c) for c in df.astype(str).values.ravel()}
    rows = []
    for name, rx in {**RULES, **DOMAIN_RULES.get(domain, {})}.items():
        hit = q.map(lambda s: bool(rx.search(s))).to_numpy()
        for i in d.index[hit]:
            s = q.iloc[i]
            rows.append({"规则": name, "query": s, "source": d.at[i, "source"],
                         "已被护栏拦下": not bool(quotable[i]), "出现在交付文档里": s in printed})
    out = pd.DataFrame(rows).drop_duplicates(["规则", "query", "source"])
    path = WORK / domain / "quote_hardrule_hits.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"{domain}: {len(d):,} 行 | 硬规则命中 {len(out)} 行（去重串 {out['query'].nunique() if len(out) else 0}）"
          f" | 其中仍可引 {int((~out['已被护栏拦下']).sum()) if len(out) else 0}"
          f" | 已印进交付文档 {int(out['出现在交付文档里'].sum()) if len(out) else 0}")
    if len(out):
        print(out.groupby(["规则", "已被护栏拦下"]).size().to_string())
        shown = out[out["出现在交付文档里"]]
        if len(shown):
            print("\n印进文档、需要人工确认的：")
            print(shown[["规则", "source", "query"]].to_string(index=False))
    print(f"→ {path}")
    return 0


if __name__ == "__main__":
    for dom in sys.argv[1:] or ["人物8", "影视8"]:
        scan(dom)
