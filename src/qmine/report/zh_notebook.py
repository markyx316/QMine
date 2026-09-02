"""The walkthrough notebook, in Chinese, in the reference deliverable's style.

Modelled on `K12_Embedding_Attempts_Comparison.ipynb`: a reporting notebook whose
distinguishing habit is that **every metric is derived on screen** rather than
quoted. The reference does not say "fragmentation 1.73"; it prints the phrasing
family, its distribution across families, the Shannon entropy, and then the
exponential — four numbered steps a reader can check.

That habit is the reason the notebook is persuasive, so it is the thing this
generator reproduces. Cells are assembled with `nbformat` and executed headless
with `nbclient`, so the delivered file is a code artifact that can be diffed and
rebuilt — never a state someone clicked into being.
"""

from __future__ import annotations

from typing import Any

from . import zh_figures as figs


def cells(state: Any, deps: Any) -> list[tuple[str, str]]:
    """Return ``(kind, source)`` pairs; kind is 'md' or 'code'."""
    gen = str(deps.store.gen_dir)
    cfg = deps.cfg
    alpha = state.get("chosen_alpha", 0.0)
    encoder = state.get("chosen_encoder", "?")
    out: list[tuple[str, str]] = []

    out.append(("md", f"""# Query 意图挖掘 — 自下而上聚类全记录
## 运行 `{state.get('run_id')}` · 领域 `{cfg.domain.key}` · 配置指纹 `{cfg.config_hash}`

本 notebook 面向汇报场景, 完整重现端到端方法、决策依据、**统一口径评分**、
嵌入空间可视化, 以及完整的家族→叶子清单。**所有数字与图表均由 cell 现场计算**, 无一粘贴。

> {deps.registry.provenance_note('zh')}

## 端到端方法论

```
① 表征构建 → ② 调优(算法证伪检验 + K 扫描) → ③ 两层层级(AMI 对齐度定家族 K, 家族内「相对随机基线的提升」选 k 定叶)
→ ④ 迭代精化(merge/split/reassign 至收敛) → ⑤ 盲评命名 + 树审计 → ⑥ 治理合并(执行) → ⑦ 部署验证
```

| 环节 | 本次取值 |
|---|---|
| 底座 encoder | `{encoder}` |
| hybrid α | **{alpha}** (措辞话语权 = α²/(1+α²) = {alpha**2/(1+alpha**2)*100:.1f}%) |
| 聚类算法 | `{state.get('chosen_algorithm','?')}` |
| 家族 K | {state.get('family_k','?')} |
"""))

    out.append(("code", f"""# %% [1] 环境与全部产物
import json, os, warnings, logging
warnings.filterwarnings('ignore')
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
import numpy as np, pandas as pd
from pathlib import Path
import matplotlib, matplotlib.pyplot as plt
matplotlib.rcParams.update({{'font.family': ['Arial Unicode MS','Heiti TC','PingFang SC','sans-serif'],
                            'axes.unicode_minus': False, 'figure.dpi': 115}})
BLUE, ORANGE, GREEN, MUTED = '#2a78d6', '#eb6834', '#1baf7a', '#898781'

GEN = Path({gen!r})
def J(name):  return json.loads((GEN/f'{{name}}.json').read_text())
def NPY(name): return np.load(GEN/f'{{name}}.npy', allow_pickle=True)
def CSV(name): return pd.read_csv(GEN/f'{{name}}.csv')

audit   = J('data_audit');       tmpl = J('template_groups')
rep     = J('representation');   gran = J('granularity')
meta    = J('hierarchy_meta');   naming = J('tree_naming')
gov     = J('governance');       dep  = J('deployment')
panel   = J('metrics_panel')['table']
# THE DELIVERED partition on both halves. `leaf_labels` is p6's, written before
# governance splits leaves, so pairing it with `leaf_family_final` described two
# different trees at once — 29 leaves' sizes grouped into 12 final families.
labels  = NPY('leaf_labels_final') if (GEN/'leaf_labels_final.npy').exists() else NPY('leaf_labels')
fam     = NPY('leaf_family_final') if (GEN/'leaf_family_final.npy').exists() else NPY('leaf_family')
famrow  = np.asarray(fam)[labels]   # fam 是 叶→家族 的映射, famrow 是每行的家族
df      = pd.read_parquet(GEN/'corpus.parquet')
print(f'语料 {{len(df):,}} 条 | 家族 {{len(set(fam.tolist())):,}} | 叶 {{int(labels.max())+1}}')

{figs.save_helper(gen)}"""))

    out.append(("md", """## 一、数据画像与模板群挖掘

模板群是本方法论的特色产物 — **一次挖掘, 三处复用**: Phase 3 是 α-sweep 的裁判,
Phase 9 是碎裂度指标的地基, Phase 11 是展示选样的模式来源。

每群的标准很严格: **凡命中者几乎必是同一意图**。"是什么"这类疑问句式附着于所有话题,
不满足这个前提, 会在内聚检验中被剔除。"""))

    out.append(("code", """# %% [2] 语料画像 + 模板群
print(f"条数 {audit['n_rows']:,} | 去重后 {audit['n_unique']:,} | 重复率 {audit['duplicate_rate']:.3%}")
print(f"长度 中位 {audit['length']['p50']:.0f} / p90 {audit['length']['p90']:.0f} / 最长 {audit['length']['max']}")
print(f"字符构成 {audit['script_mix']}")
print()
tg = pd.DataFrame([{'模板群': g['name'], '命中': g['n_hits'], '占比': f"{g['share']:.2%}",
                    '来源': '种子' if not g['discovered'] else '挖掘',
                    '示例': (g['examples'] or [''])[0][:20]} for g in tmpl['groups']])
display(tg)
print(f"\\n合计覆盖率 {tmpl['coverage']['union_coverage']:.1%}  (质量门窗口 20–40%)")"""))

    out.append(("md", """## 二、评分计算方法 (精确定义 + 现场演算)

### silhouette (轮廓系数)
对每个样本 i: `a(i)` = 到**本簇**其他成员的平均距离; `b(i)` = 到**最近其他簇**成员的平均距离;
`s(i) = (b−a)/max(a,b) ∈ [−1,1]`。整体分数 = 全体 s(i) 均值。
**本实现**: cosine 距离, 固定子样 + 固定 seed, 各方案共用同一子样保证可比。
锚点: ≈0 表示连续体无天然边界; >0.2 才是"出现真实岛屿"。

### 重播稳定性 ARI
同数据同算法, **只换随机种子**跑两次, 得两套划分; ARI 衡量两者对"任意样本对是否同簇"
判断的一致度, 并扣除随机碰巧一致的期望: `ARI = (RI − E[RI]) / (max(RI) − E[RI])`。
**为什么是一票否决项**: 聚类没有真值, 但有可复现性 — 不可复现的划分不是数据的真实结构,
无论 silhouette 多好看。

### 家族碎裂度 (本方法论自研)
取一组**已知同意图的措辞群**, 看它被劈进几个"有效家族":
家族分布 p → 香农熵 `H = −Σ p·ln p` → **有效家族数 = exp(H)**。全在一族 = 1.0。"""))

    out.append(("code", """# %% [3] 现场演算: 碎裂度逐步计算
import re
# TRUSTED groups only. Fragmentation asks "a set of KNOWN same-intent phrasings —
# how many families did it get split into?"; a group whose cohesion check failed
# is not known-same-intent, so its split is not evidence about the clustering and
# counting it inflates the number. This is the same rule `foundation.py` applies
# with `trusted_only=True` for the masks that judge representations.
groups = {g['name']: g['pattern'] for g in tmpl['groups'] if g.get('trusted', True)}
_untrusted = [g['name'] for g in tmpl['groups'] if not g.get('trusted', True)]
if _untrusted:
    print(f"排除 {len(_untrusted)} 个未通过内聚检验的模板群: {_untrusted}")
fam_of_row = fam[labels]
rows = []
for name, pat in groups.items():
    hit = df['query'].astype(str).str.contains(pat, regex=True, na=False).values
    if hit.sum() < 30: continue
    p = pd.Series(fam_of_row[hit]).value_counts(normalize=True)
    H = float(-(p * np.log(p)).sum())
    rows.append({'模板群': name, '命中': int(hit.sum()), '香农熵 H': round(H,4),
                 '有效家族数 exp(H)': round(float(np.exp(H)),2), '最大家族占比': f"{p.iloc[0]:.1%}"})
# NO ROWS = NO TABLE. `pd.DataFrame([])` has no columns, so sorting by name
# raises KeyError and the notebook dies at this cell. med04: ALL 12 template
# groups came back trusted=False, `groups` was empty, and the deliverable ran
# 3 of 18 cells. A generated cell must survive its own corpus.
if not rows:
    _why = (f'{len(_untrusted)} 个模板群都未通过内聚检验' if _untrusted
            else '所有模板群命中数均 < 30')
    print(f'⚠ 跳过碎裂度演算 —— {_why}。')
    print('  碎裂度依赖「已知同意图」的模板群; 一个都没有时这一节无可计算,')
    print('  报告中的 template_fragmentation 也应连同这一点一起读。')
    frag = pd.DataFrame(columns=['模板群', '命中', '香农熵 H',
                                 '有效家族数 exp(H)', '最大家族占比'])
else:
    frag = pd.DataFrame(rows).sort_values('有效家族数 exp(H)', ascending=False)
display(frag)

if len(frag):
    worst = frag.iloc[0]
    pat = groups[worst['模板群']]
    hit = df['query'].astype(str).str.contains(pat, regex=True, na=False).values
    p = pd.Series(fam_of_row[hit]).value_counts(normalize=True)
    print(f"\\n—— 以碎裂最严重的「{worst['模板群']}」为例, 一步步算 ——")
    print(f"① 模板群 n = {hit.sum():,} 条")
    print("② 它们的家族分布 p:")
    for fid, share in p.head(6).items():
        print(f"     {share:6.1%}  家族 {int(fid)}")
    H = float(-(p*np.log(p)).sum())
    print(f"③ 香农熵 H = -Σ p·ln p = {H:.4f}")
    print(f"④ 有效家族数 = exp(H) = {np.exp(H):.2f}   ← 这就是该群的碎裂度")
    print(f"⑤ 全部 {len(frag)} 群平均 = {frag['有效家族数 exp(H)'].mean():.2f}   ← 报告中的碎裂度")"""))

    out.append(("md", """## 三、表征选型: bake-off 与 α-sweep

**两条反直觉结论**, 是本方法论最贵的两课:

1. **更大 ≠ 结构发现更好** — 判别力与几何结构是两回事。选型必须用**你自己的聚类任务**
   做 bake-off, 不能只看榜单检索分。
2. **silhouette 是错误的代理指标** — 它衡量"簇内紧、簇间远", 但**措辞相同的 query 天然最紧**,
   用它选型会系统性偏向"模板孪生簇", 恰好与"家族可解释"的目标相反。
   因此它全程**只报告、不投票**。"""))

    out.append(("code", """# %% [4] bake-off + α-sweep 决策证据
bake = rep.get('bakeoff', {})
if bake.get('rows'):
    display(pd.DataFrame(bake['rows'])[['encoder','dim','stability_ari','template_fragmentation','silhouette']])
    print(f"当选: {bake['chosen_encoder']}   规则: {bake['chosen_by']}")
    if bake.get('silhouette_disagrees'):
        print(f"⚠️  silhouette 会选 {bake['silhouette_would_have_chosen']} — 已记录并否决")

sw = rep.get('alpha_sweep', {})
if sw.get('rows'):
    t = pd.DataFrame(sw['rows'])[['alpha','surface_vote_share','template_fragmentation','stability_ari','silhouette']]
    t.columns = ['α','措辞话语权','碎裂度↓(主裁判)','稳定性↑(主裁判)','silhouette(仅参考)']
    display(t)
    print(f"\\n当选 α = {sw['chosen_alpha']}   规则: {sw['chosen_by']}")
    if sw.get('silhouette_disagrees'):
        print(f"⚠️  silhouette 会选 α={sw['silhouette_would_have_chosen']} — 已记录并否决")
    print("\\nα 的精确含义: cos(H,H′) = (cos_semantic + α²·cos_surface)/(1+α²)")
    print("  → 措辞块的话语权是 α², 不是 α。这是「1% 的轻推」与「20% 反客为主」的差别。")"""))

    out.append(("code", figs.fig_alpha()))

    out.append(("md", """## 四、算法选型 battery 与 K 值三角验证

算法阶段跑的是**证伪检验**而不是淘汰赛 —— 交付的树始终是 KMeans。K 值则由三条**独立**路线交叉验证 —
一条路线得出的 K 可疑, 三条收敛到同一尺度才是强证据。"""))

    out.append(("code", """# %% [5] battery + K 三角验证
try:
    battery = J('battery')
    b = pd.DataFrame(battery['rows'])[['algorithm','n_clusters','stability_ari','silhouette','noise_rate']]
    b.columns = ['算法','簇数','稳定性 ARI','silhouette','噪声率']
    display(b)
    print('当选:', battery['verdict']['chosen'], '|', battery['verdict']['chosen_by'])
except Exception as e:
    print('battery 未产出:', e)

tri = gran['triangulation']
print('\\n三条独立路线的 K 估计:')
for k, v in tri['estimates'].items(): print(f'   {k:<34} {v}')
print(f"\\n定案 K = {tri['chosen_family_k']}  ({tri['chosen_by']})")
if tri.get('divergence_note'): print('⚠️ ', tri['divergence_note'])"""))

    out.append(("code", figs.fig_battery()))
    out.append(("code", figs.fig_ksweep()))

    out.append(("md", """## 四之二、嵌入空间可视化 — 看形状, 不下结论

投影图只回答一个问题: **同一个意图有没有被劈开?**
所有裁决指标都在原始高维空间计算 — 2-D 投影必然丢信息, 用它下结论是明确禁止的一步。"""))

    out.append(("code", figs.fig_umap_families()))
    out.append(("code", figs.fig_umap_intent()))

    out.append(("md", """## 五、两层层级构建与收敛

**定义先行**: 一个家族/叶子在数学上就是一个**质心**; "属于 g" = "离质心 g 最近"。
名字与定义是事后由盲评补写的 — **先有结构, 后有语言**。

**为什么不用一步到位的大 K?** K 扫描显示细粒度全局划分可复现性太差;
"稳定粗分 + 家族内局部细分"让每层都工作在各自更好复现的尺度上。"""))

    out.append(("code", """# %% [6] 精化收敛轨迹 + held-out 结构复现
h = pd.DataFrame(meta['refinement_history'])
if len(h):
    h.columns = [{'round':'轮','merges':'合并','splits':'拆分','moved_fraction':'移动行占比',
                  'n_leaves':'叶数','silhouette':'silhouette'}.get(c,c) for c in h.columns]
    display(h)
hr = meta['heldout_reproduction']
print(f"held-out 结构复现 = {hr['agreement']:.3%}  (n={hr['n_test']:,})")
sv = hr.get('statistical_verdict', {})
if sv:
    print(f"   95% CI [{sv['ci95'][0]:.3f}, {sv['ci95'][1]:.3f}] → 判定: {sv['verdict']}")
    print(f"   {sv['note']}")
print("\\n> 这是「结构是真的」的最终背书: 只在看得见全部数据时才存在的划分,")
print("> 是对这份样本的描述, 而不是对现象的描述。")"""))

    out.append(("md", """## 六、盲评命名与树审计

命名 agent **看不到任何既有标签** — 不给旧分类、不给自上而下的意图名、不给彼此的答案,
只看成员样本卡片。锚定效应是真实的: 见过既有体系的命名者会把簇"认领"到旧类目下,
掩盖数据的真实形状。

卡片 = 质心最近若干 + 随机若干 + **边缘若干**。边缘样本是故意放进去的 —
它们是"勉强属于"的成员, 是杂质显形的地方, 也是 coherence 评分有意义的前提。"""))

    out.append(("code", """# %% [7] 完整家族 → 叶子清单
sizes = np.bincount(labels, minlength=int(labels.max())+1)
total = len(labels)
# Join on leaf_ids, NOT family_id: the auditor numbers its own families (19 on
# live38) and the partition numbers its own (12). Matching by integer id was
# wrong for 19 of 19 and titled a family of classical-poetry leaves
# "中考录取分数与学校排名查询".
# ONE IMPLEMENTATION, IMPORTED — not a second copy of the join.
# This cell used to re-implement `_shape.family_names` inline, and the two
# drifted: the notebook printed `X 等 6 类` while the reports printed
# `混合·主要成分「X」38%`, so one run produced two documents that disagreed about
# what a family is called. The notebook already reads this run's artifacts by
# path, so depending on the package that wrote them costs nothing it did not
# already owe.
from qmine.report._shape import family_names as _family_names
fam_names = _family_names(naming, fam, sizes)
by_fam = {}
for n in naming['namings']:
    by_fam.setdefault(int(fam[n['leaf_id']]), []).append(n)

print(f"════ {len(by_fam)} 家族 / {len(naming['namings'])} 叶  ════\\n")
for f in sorted(by_fam, key=lambda k: -sum(int(sizes[x['leaf_id']]) for x in by_fam[k])):
    members = sorted(by_fam[f], key=lambda x: -int(sizes[x['leaf_id']]))
    fn = int(sum(sizes[x['leaf_id']] for x in members))
    risk = ' (风控标记)' if any(x.get('risk_flag') for x in members) else ''
    print(f"■ {fam_names.get(f, f'家族 {f}')}{risk}  (n={fn:,}, {len(members)}叶, {fn/total:.1%})")
    for x in members:
        print(f"   ├─ {x.get('name_zh','')}  n={int(sizes[x['leaf_id']]):,}")
    print()
print(f"平均 coherence = {naming.get('mean_coherence')}/5")"""))

    out.append(("code", """# %% [8] 完整命名档案: 每叶的 user_need 定义句
print('名字会歧义, 定义句不会 — user_need 同时是标注指南 / 验收标准 / 下游产品需求说明\\n')
for f in sorted(by_fam):
    print(f"■ {fam_names.get(f, f'家族 {f}')}")
    for n in sorted(by_fam[f], key=lambda x: -int(sizes[x['leaf_id']])):
        st = '★' * int(n.get('coherence') or 0)
        print(f"  - {n.get('name_zh','')} ({st}, n={int(sizes[n['leaf_id']]):,})")
        print(f"      {n.get('user_need','')}")
        if n.get('mix_notes'): print(f"      ⟨杂质⟩ {n['mix_notes']}")
        if n.get('risk_flag'): print(f"      ⚠️ 风控: {n.get('risk_reason','')}")
    print()"""))

    out.append(("md", """## 七、治理合并 — 执行, 不是记录

审计发现的问题必须落到**交付数据的标签列**上, 而不是只写进报告的"建议"一节。
家族合并 = 改写"叶→家族"查找表 (叶分配与质心完全不动), 原家族列保留可追溯。

**交付前自检**: 报告里每一句"建议 X", 数据里有没有对应的列?"""))

    out.append(("code", """# %% [9] 治理台账 + 指标变化
led = pd.DataFrame(gov['ledger'])
if len(led):
    led = led[['id','kind','targets','status','executed_column','rationale']]
    led.columns = ['处方','类型','目标','状态','落到哪一列','理由']
    display(led)
print('执行机制:', gov['mechanism'])
print('指标变化:', gov['execution'].get('metric_deltas'))
declined = [r for r in gov['ledger'] if r['status']=='declined']
if declined:
    print('\\n有意保留的划分 (是决策, 不是遗漏):')
    for r in declined: print(f"  - {r['targets']}: {r['decline_reason']}")"""))

    out.append(("md", """## 八、统一度量面板

对比任何两个以上方案时, 所有指标**用同一套代码、同一子样、同一种子重算**。
严禁各自引用各自实验时期的数字 — 那是在比较两个恰好同名的不同测量。"""))

    out.append(("code", """# %% [10] 统一面板 + 裁决权
rows = pd.DataFrame(panel['rows'])
display(rows)
print(f"panel id = {panel['panel_id']}")
print(f"配置: {panel['panel_config']}\\n")
print('指标裁决权:')
zh = {'decisive':'主裁判','advisory':'仅参考 (无投票权)','diagnostic':'描述性'}
for m in panel['metrics']:
    print(f"   {m['name']:<28} {zh.get(m['authority'], m['authority'])}")
print()
for i, f in enumerate(panel['footnotes'], 1): print(f'{i}. {f}\\n')"""))

    out.append(("code", figs.fig_panel_bars()))

    out.append(("md", """## 九、部署验证与确定性样本展示

**确定性选样** (原则 7): 每个模板群取**命中集合的中位数下标**实例 —
从机制上排除 cherry-picking。展示的说服力来自"你无法挑样本", 而不是样本本身多好看。"""))

    out.append(("code", """# %% [11] 部署指标 + 新 query 现场路由 + 确定性样本
r = dep['routing']
print(f"实时分类 = {dep['inference']}")
print(f"模型体积 = {dep['model_bytes']/1024:.0f} KB (仅质心矩阵)")
print(f"margin 模糊率 = {r['ambiguous_rate']:.1%}  (阈值 {r['threshold']})")
print(f"  {r['policy']}\\n")
if dep.get('live_demo'):
    print('新 query 现场路由演示:')
    display(pd.DataFrame(dep['live_demo']))
if dep.get('deterministic_exemplars'):
    print('\\n确定性样本 (中位数下标, 无法挑选):')
    display(pd.DataFrame(dep['deterministic_exemplars'])[['pattern','n_hits','exemplar']])"""))

    out.append(("code", """# %% [12] 全量交付表: 双路线并排
lab = CSV('labels_full')
print('交付列:', [c for c in lab.columns])
display(lab.head(10))
print(f"\\n模糊行占比 {lab['bu_ambiguous'].mean():.1%}")
if 'td_l1' in lab.columns:
    print('双路线并排交付 ✓  (td_* = 自上而下意图轴, bu_* = 自下而上内容轴, 互不覆盖)')"""))

    out.append(("md", """## 十、这些数字不代表什么

- **蒸馏分类器精度度量的是可蒸馏性**, 即这套聚类标签能否被表征线性学出 —
  **不是**与人类判断的一致性。人类一致性需由金标 (Cohen's κ) 与对抗验证单独测量。
- **silhouette 全程仅参考**。它被最大化时会长成"模板孪生簇", 正是本流程要防的失效。
- **碎裂度须与家族数同看**: 家族越少越难碎, 跨方案对比必须双条件表述("细而不碎")。
- **held-out 复现检验的是结构稳定性, 不是语义正确性** — 一个可复现的错误划分仍然是错的。
- **聚类对功能型意图结构性不可见**: 措辞与内容正常、意图藏在语用里的类别
  (用法判断、解题、导航、闲聊) 只能由自上而下体系承担, 本报告不硬凑。"""))

    return out


def build(state: Any, deps: Any) -> Any:
    """Assemble and execute the Chinese walkthrough."""
    import nbformat

    nb = nbformat.v4.new_notebook()
    nb.cells = [
        nbformat.v4.new_markdown_cell(src) if kind == "md" else nbformat.v4.new_code_cell(src)
        for kind, src in cells(state, deps)
    ]
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    path = deps.store.gen_dir / "自下而上聚类全流程.ipynb"

    executed, error = False, ""
    try:
        _client(nb, deps.store.gen_dir).execute()
        executed = True
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"

    nbformat.write(nb, path)
    n_err = sum(1 for c in nb.cells for o in (c.get("outputs") or [])
                if o.get("output_type") == "error")
    deps.emit(f"  notebook(中文): {'已执行' if executed else '未执行'}, {n_err} 个 cell 报错"
              + (f" — {error[:120]}" if error else ""))
    return deps.store.register_file(
        "notebook_zh", path, "notebook", producer="p11",
        summary=f"中文全流程 notebook, {'已执行' if executed else '未执行'}, {n_err} errors",
    )


def _client(nb: Any, workdir: Any) -> Any:
    """An nbclient bound to *this* interpreter.

    `kernel_name="python3"` resolves through the user's Jupyter kernelspecs,
    which routinely point at an unrelated environment — on the machine this was
    written, at a different project's virtualenv that lacked pyarrow. The
    notebook then fails in its first cell and every figure below it is silently
    missing. Launch ipykernel from `sys.executable` so the notebook sees exactly
    the packages the pipeline just used to produce the artifacts it reads.
    """
    import sys

    import nbclient

    kwargs: dict[str, Any] = dict(
        timeout=1800, allow_errors=False,
        resources={"metadata": {"path": str(workdir)}},
    )
    try:
        from jupyter_client.manager import KernelManager

        km = KernelManager(kernel_name="python3")
        km.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher",
                               "-f", "{connection_file}"]
        return nbclient.NotebookClient(nb, km=km, **kwargs)
    except Exception:  # noqa: BLE001 - fall back to whatever kernel is configured
        return nbclient.NotebookClient(nb, kernel_name="python3", **kwargs)
