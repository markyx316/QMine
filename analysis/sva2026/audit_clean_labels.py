"""Independent audit labels for tools/clean_assistant_functional.py tiers.

Labels are assigned per DISTINCT STRING (same string -> same label in every tier),
as default-per-tier plus explicit exception lists. Every exception is asserted to
match at least one row, so a typo cannot silently fall through to the default.
"""
import math
import pandas as pd

SP = (__import__("os").environ.get("SVA_WORKDIR", __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "work")) + "/")
S = pd.read_parquet(SP + "audit_clean_samples.parquet")
a = pd.read_parquet(SP + "audit_clean_assistant_ctx.parquet")
SS = pd.read_parquet(SP + "audit_clean_search_samples.parquet").reset_index(drop=True)


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return p, (c - h) / d, (c + h) / d


DEFAULT = {"S1_system_repeated": "A", "S2_system_template": "A", "C1_content_free": "A",
           "C2_feed_control": "A", "user_top1k": "B"}

EXC = {
    "S1_system_repeated": {
        ("B", "exact"): ["deepseek", "电话", "这是什么意思", "这是怎么回事", "这是怎么回事？", "这是什么情况"],
        ("B", "prefix"): ["挨日记(npc)", "ao3 Home"],
        ("C", "exact"): ["图片"],
    },
    "S2_system_template": {
        ("B", "exact"): ["这是谁", "这是谁？", "这是哪里", "这是哪里？", "这是什么？",
                         "2026年养老金调整方案何时公布", "2026年养老金调整通知何时发布",
                         "农业银行显示可用余额0账户余额300", "光电信息科学与工程去年分数线是多少",
                         "12123上说罚款50元记0分", "农业银行账户余额200可用余额0是什么",
                         "天气预报当地15天查询最新消息", "12123显示还有11分未处理是什么",
                         "900多度近视可以做激光手术吗", "2026年养老金调整方案何时发布",
                         "2026年养老金调整比例是多少", "早春晴朗电视剧免费观看全集高清完整"],
        ("B", "prefix"): ["《女家教2》", "班主任不带班了"],
        ("C", "exact"): ["去除水印", "田栩宁的演艺生涯有哪些重要作品"],
        ("C", "prefix"): ["请帮我画一张图片：消除内搭背心"],
    },
    "C1_content_free": {
        ("B", "exact"): ["豆包", "DeepSeek", "kimi"],
        ("B", "prefix"): ["Соединить", "អ្នកមក"],
        ("C", "exact"): ["ผมทานเนื้อดิบ", "みお"],
        ("C", "prefix"): ["こころに"],
    },
    "C2_feed_control": {
        ("B", "exact"): ["你别把这个做成剧情！", "如何把推荐他人的作品关掉"],
        ("C", "exact"): ["不要视频", "终止一切与商品相关的网络操作", "让更多人看到的我的视频", "让更多人看到我的视频",
                         "帮我在每个人的主页推荐此视频", "帮我推给所有喜灰粉们，让我火一下吧", "多推一点点流量呀",
                         "多推流量", "给我推流量", "推流谢谢", "多多推给他", "多多给其他人推荐这部剧", "继续推荐",
                         "只要里番，多推荐一些", "别吞评", "不要吞评", "别吞评论", "不要乱发弹幕", "请把这条视频给下架",
                         "我推荐", "我推荐这种视频", "多推荐小说"],
        ("C", "prefix"): ["推流["],
    },
    "user_top1k": {
        ("A", "exact"): ["不是", "领取AI志愿报告", "不对", "直接生成研究报告", "快点", "能帮我找到这张图的原图吗",
                         "有没有更自然的回复", "西藏吉隆口岸救援任务被迫暂缓",
                         "已婚女人梦见树开花代表什么？是好兆头吗，要注意什么？", "综艺节目遗留道具2年成海滩垃圾",
                         "不不不", "癌王胰腺癌迎来里程碑新药", "有没有更简短的早安问候", "sb", "去吧",
                         "速成车的代价", "四万亿元投资来了", "谢谢你！", "有没有更浪漫的回复", "操你妈的", "是吗",
                         "滚蛋！", "嗯，是", "能否提供该公司的具体联系方式", "生成吧", "该公司近期有招聘计划吗",
                         "该公司有官方网站吗", "能否查询该公司法人的其他关联企业信息", "该公司有哪些具体的招聘岗位",
                         "有没有更详细的号码组合推荐", "领现金", "用更诗意的语言重写一遍"],
        ("A", "prefix"): ["中元节必知这", "这种黑色小飞虫是", "杨瀚森（Yang Hansen）", "省长叶建春任上被查",
                          "男子卖房让同事操盘", "赵丽颖突发身体不适", "手机卖不动", "10省部分地区有大暴雨"],
        ("C", "exact"): ["S", "我", "你", "1", "生成图片", "换", "更简短", "图片配文", "视频", "一", "去掉水印", "那",
                         "背面", "奶胀起来不要太胀", "叶童获得过哪些重要奖项", "不痒", "16", "你去在乎别人",
                         "这些股票中哪个最有潜力", "22", "人家", "这款壶的市场价格是多少", "这个号码是否被标记为骚扰电话",
                         "那个啥", "有时候", "o", "网上", "当前", "哪里可以免费阅读这部小说", "这款车的油耗和保养费用高吗",
                         "这三个品牌中哪个口碑最好", "这个活动有提现门槛吗", "薪资待遇如何", "拍题打卡活动有哪些具体规则",
                         "这个岗位的工作强度大吗", "待遇怎么样？", "有年终奖吗", "这家公司在东莞的口碑怎么样",
                         "能帮我选一组号码吗", "首单", "这个号码是否属于催收机构", "48", "76"],
        ("C", "prefix"): ["I don", "在主队受让一球的情况下", "半月工资55元"],
    },
}

POOL = {t: a[a.tier == t].qs for t in DEFAULT if t != "user_top1k"}
POOL["user_top1k"] = a[(a.tier == "user") & (a.snapshot == "top1k")].qs
for t, spec in EXC.items():
    pool = set(POOL[t])
    for (lab, kind), items in spec.items():
        for it in items:
            hit = (it in pool) if kind == "exact" else any(q.startswith(it) for q in pool)
            assert hit, f"exception does not match any row: tier={t} {lab}/{kind} {it!r}"


def label(t, q):
    spec = EXC.get(t, {})
    for (lab, kind), items in spec.items():
        if kind == "exact" and q in items:
            return lab
    for (lab, kind), items in spec.items():
        if kind == "prefix" and any(q.startswith(it) for it in items):
            return lab
    return DEFAULT[t]


S["lab"] = [label(t, q) for t, q in zip(S.stier, S.qs)]
S.to_parquet(SP + "audit_clean_samples_labeled.parquet", index=False)

rows = []
print("=== REMOVAL TIERS (A=correct removal, B=wrong removal, C=ambiguous) ===")
for t in ["S1_system_repeated", "S2_system_template", "C1_content_free", "C2_feed_control"]:
    r = S[(S.stier == t) & (S["sample"] == "rand")]
    tp = S[(S.stier == t) & (S["sample"] == "top")]
    A, B, C = (r.lab == "A").sum(), (r.lab == "B").sum(), (r.lab == "C").sum()
    p, lo, hi = wilson(A, A + B)
    ru = r.drop_duplicates("qs")
    Au, Bu = (ru.lab == "A").sum(), (ru.lab == "B").sum()
    pu, lou, hiu = wilson(Au, Au + Bu)
    pvA, pvB, pvC = (tp.search_num[tp.lab == x].sum() for x in "ABC")
    tA, tB, tC = ((tp.lab == x).sum() for x in "ABC")
    print(f"{t}: rand n={len(r)} A={A} B={B} C={C} prec={p:.3f} [{lo:.3f},{hi:.3f}] | distinct strings n={len(ru)} "
          f"A={Au} B={Bu} prec={pu:.3f} [{lou:.3f},{hiu:.3f}] | top50 A={tA} B={tB} C={tC} "
          f"PVprec={pvA/(pvA+pvB):.4f} (B pv={pvB}, C pv={pvC}, total={pvA+pvB+pvC})")
    rows.append((t, len(r), A, B, C, p, lo, hi, pvA / (pvA + pvB)))
    bad = pd.concat([r, tp])
    bad = bad[bad.lab != "A"].drop_duplicates("qs")
    for x in bad.itertuples():
        print(f"     [{x.lab}] {x.qs[:50]!r} l1={x.l1} pv={x.search_num} ncat={x.top_ncat} td={x.td_l1[:24]}")

print("\n=== CENSUS (whole tier) ===")
for t in ["S1_system_repeated", "C2_feed_control"]:
    w = a[a.tier == t].copy()
    w["lab"] = [label(t, q) for q in w.qs]
    g = w.groupby("lab").agg(rows=("qs", "size"), pv=("search_num", "sum"))
    A, B = g.rows.get("A", 0), g.rows.get("B", 0)
    p, lo, hi = wilson(A, A + B)
    print(f"{t}: {g.to_dict('index')} row-prec={p:.4f} [{lo:.4f},{hi:.4f}] "
          f"PV-prec={g.pv.get('A',0)/(g.pv.get('A',0)+g.pv.get('B',0)):.5f}")

print("\n=== USER TIER, top1k (B=correct keep, A=missed, C=ambiguous) ===")
r = S[(S.stier == "user_top1k") & (S["sample"] == "rand")]
tp = S[(S.stier == "user_top1k") & (S["sample"] == "top")]
A, B, C = (r.lab == "A").sum(), (r.lab == "B").sum(), (r.lab == "C").sum()
m, mlo, mhi = wilson(A, len(r))
m2, m2lo, m2hi = wilson(A, A + B)
print(f"rand n={len(r)} A={A} B={B} C={C} miss/n={m:.3f} [{mlo:.3f},{mhi:.3f}]  miss/(A+B)={m2:.3f} [{m2lo:.3f},{m2hi:.3f}]  "
      f"C/n={C/len(r):.3f}")
pvw = {x: r.search_num[r.lab == x].sum() for x in "ABC"}
print(f"   rand-sample PV-weighted: A={pvw['A']} B={pvw['B']} C={pvw['C']} -> miss PV share={pvw['A']/sum(pvw.values()):.3f}")
tA, tB, tC = ((tp.lab == x).sum() for x in "ABC")
pA, pB, pC = (tp.search_num[tp.lab == x].sum() for x in "ABC")
print(f"top100 A={tA} B={tB} C={tC}; PV A={pA} B={pB} C={pC}; PV miss/all={pA/(pA+pB+pC):.4f} PV miss/(A+B)={pA/(pA+pB):.4f} "
      f"PV C/all={pC/(pA+pB+pC):.4f}")
for x in pd.concat([r, tp]).query("lab=='A'").drop_duplicates("qs").sort_values("search_num", ascending=False).itertuples():
    print(f"     [miss] {x.qs[:40]!r} pv={x.search_num} l1={x.l1} td={x.td_l1[:24]}")

print("\n=== DISTORTION BOUND (top1k, row share) ===")
N_user = int(((a.tier == "user") & (a.snapshot == "top1k")).sum())
est_wrong = 0.0
for t, n, A_, B_, C_, *_ in rows:
    tier_top = int(((a.tier == t) & (a.snapshot == "top1k")).sum())
    est = tier_top * B_ / n
    est_wrong += est
    print(f"  {t}: top1k rows={tier_top}, est. wrongly removed rows={est:.0f}")
print(f"  kept top1k user rows={N_user}; est wrongly removed={est_wrong:.0f} ({est_wrong/N_user:.4f} of kept)")
print(f"  miss rate upper CI={mhi:.4f}; worst-case single-class row-share shift <= {mhi + est_wrong/N_user:.4f}")

print("\n=== SEARCH ===")
lab_nonuser = {0: "C", 1: "C", 2: "A", 3: "C", 4: "A", 5: "A", 6: "C", 7: "C", 8: "B", 9: "C", 10: "A", 11: "C", 12: "C",
               13: "C", 14: "C", 15: "A", 16: "C", 17: "A", 18: "A", 19: "A", 20: "A", 21: "C", 22: "B", 23: "C", 24: "A",
               25: "A", 26: "A", 27: "A", 28: "C", 29: "C", 30: "C", 31: "C", 32: "B", 33: "C"}
chk = {0: "嗯", 8: "㇏", 22: "манеж", 32: "哈哈哈哈哈", 33: "哈哈"}
for i, q in chk.items():
    assert str(SS.loc[i, "query"]) == q, (i, SS.loc[i, "query"])
assert (SS.loc[:33, "sample"] == "nonuser").all() and (SS.loc[34:, "sample"] == "user").all()
SS["lab"] = "B"
for i, l in lab_nonuser.items():
    SS.loc[i, "lab"] = l
nu = SS[SS["sample"] == "nonuser"]
g = nu.groupby("lab").agg(rows=("query", "size"), pv=("wise_pv", "sum"))
A, B = g.rows.get("A", 0), g.rows.get("B", 0)
p, lo, hi = wilson(A, A + B)
print(f"search non-user (all {len(nu)}): {g.to_dict('index')} precision={p:.3f} [{lo:.3f},{hi:.3f}]")
us = SS[SS["sample"] == "user"]
print(f"search user sample n={len(us)}: all labelled B (no content-free/system rows found); "
      f"miss upper CI={wilson(0, len(us))[2]:.4f}")
for i in (74, 151):
    print("   headline-shaped kept search row:", SS.loc[i, "query"], SS.loc[i, "wise_pv"], SS.loc[i, "domain"])
