"""The figure suite for the Chinese walkthrough notebook.

Modelled cell-for-cell on the reference deliverable
(`K12_Embedding_Attempts_Comparison.ipynb`), which carries six figures. Five of
them exist to make a *decision* visible rather than to decorate: the reader is
meant to look at the picture and see why the run chose what it chose.

Each function returns notebook source, not a rendered image. The figures are
therefore recomputed from the run's own artifacts every time the notebook is
executed — a reader who distrusts a number can edit the cell above it and watch
the picture move. Figures are also written to `figures/` as PNG so the Markdown
report can embed the identical image rather than a look-alike.

The projection figures (4 and 5) are the expensive pair; they subsample and fall
back from UMAP to PCA so the notebook still executes on a machine without
`umap-learn`.
"""

from __future__ import annotations

# Shared by both projection figures: build a 2-D view of a high-dimensional
# space, preferring UMAP's local-structure preservation but never requiring it.
_PROJECT = '''
def project(X, seed=20240601, n=6000):
    """2-D view of X. UMAP when available (keeps local neighbourhoods), else PCA."""
    idx = np.arange(len(X))
    if len(X) > n:
        idx = np.random.RandomState(seed).choice(len(X), n, replace=False)
    Xs = X[idx]
    try:
        import umap
        P = umap.UMAP(n_neighbors=15, min_dist=0.10, metric='cosine',
                      random_state=seed).fit_transform(Xs)
        how = 'UMAP(cosine, n_neighbors=15)'
    except Exception as e:
        from sklearn.decomposition import PCA
        P = PCA(n_components=2, random_state=seed).fit_transform(Xs)
        how = f'PCA (UMAP 不可用: {type(e).__name__})'
    return P, idx, how
'''


def fig_ksweep(chosen_k_expr: str = "tri['chosen_family_k']") -> str:
    """K sweep, three panels. Replaces a twin-axis plot that hid the third metric."""
    return f"""# %% 图 1 — K 扫描三联: 三个指标, 三条独立的曲线
ks = pd.DataFrame(gran['k_sweep']).sort_values('k')
K = {chosen_k_expr}
fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))
spec = [('stability_ari',        '稳定性 ARI ↑ (主裁判)',       BLUE,  'o-'),
        ('template_fragmentation','模板碎裂度 ↓ (主裁判)',       GREEN, 'D-'),
        ('silhouette',           'silhouette (仅参考, 无投票权)', MUTED, 's--')]
for ax, (col, title, colour, style) in zip(axes, spec):
    if col not in ks: ax.axis('off'); continue
    ax.plot(ks['k'], ks[col], style, color=colour, lw=1.9, ms=5)
    ax.axvline(K, color=ORANGE, ls=':', lw=2)
    ax.annotate(f'定案 K={{K}}', xy=(K, ax.get_ylim()[1]), xytext=(4, -12),
                textcoords='offset points', color=ORANGE, fontsize=9, va='top')
    ax.margins(y=.18)                       # headroom so the peak label stays inside
    best = ks.loc[ks[col].idxmin() if 'frag' in col else ks[col].idxmax()]
    if int(best['k']) != int(K):
        ax.plot(best['k'], best[col], '*', color=colour, ms=15, zorder=5)
        # Park the label in the corner diagonally opposite the star and run a
        # leader line to it — an offset from the point lands on the curve.
        lo, hi = ax.get_ylim(); x0, x1 = ax.get_xlim()
        fx = (best['k'] - x0) / max(1e-9, x1 - x0)
        fy = (best[col] - lo) / max(1e-9, hi - lo)
        tx, ha = (0.97, 'right') if fx < 0.5 else (0.03, 'left')
        ty, va = (0.97, 'top') if fy < 0.5 else (0.03, 'bottom')
        ax.annotate(f"该指标自己的峰 K={{int(best['k'])}}", xy=(best['k'], best[col]),
                    xytext=(tx, ty), textcoords='axes fraction', fontsize=8,
                    color=colour, ha=ha, va=va,
                    arrowprops=dict(arrowstyle='-', color=colour, lw=.7, alpha=.55,
                                    shrinkA=2, shrinkB=7))
    ax.set_xlabel('K'); ax.set_title(title, fontsize=10)
    ax.grid(alpha=.25, ls=':')
fig.suptitle('K 扫描: 三个指标很少同峰 — 这正是「必须先指定谁是裁判」的现场证据', fontsize=11)
plt.tight_layout(); SAVE(fig, 'fig1_ksweep'); plt.show()

_peaks = {{c: int(ks.loc[ks[c].idxmin() if 'frag' in c else ks[c].idxmax(), 'k'])
           for c, *_ in spec if c in ks}}
print('各指标各自的峰值 K:', _peaks)
print(f'定案 K = {{K}} — 依据「{{tri["chosen_by"]}}」, 而不是三者的平均。')
print('若取平均, 会得到一个三条证据都不支持的 K; 分歧本身已记录在案。')"""


def fig_alpha() -> str:
    """The alpha decision. The reference's most-cited panel."""
    return """# %% 图 2 — α 决策三联: 措辞话语权换来了什么
sw = rep.get('alpha_sweep', {})
if sw.get('rows'):
    a = pd.DataFrame(sw['rows']).sort_values('alpha')
    A = sw['chosen_alpha']
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))
    spec = [('template_fragmentation', '模板碎裂度 ↓ (主裁判)',        GREEN, True),
            ('stability_ari',          '稳定性 ARI ↑ (次裁判)',        BLUE,  False),
            ('silhouette',             'silhouette (仅参考, 无投票权)', MUTED, False)]
    for ax, (col, title, colour, lower_better) in zip(axes, spec):
        if col not in a: ax.axis('off'); continue
        ax.plot(a['alpha'], a[col], 'o-', color=colour, lw=1.9, ms=6)
        ax.axvline(A, color=ORANGE, ls=':', lw=2)
        ax.set_xlabel('α'); ax.grid(alpha=.25, ls=':'); ax.margins(y=.20)
        ax.set_title(title, fontsize=10, pad=26 if col == spec[0][0] else 6)
        for x, y in zip(a['alpha'], a[col]):
            ax.annotate(f'{y:.3f}', (x, y), textcoords='offset points',
                        xytext=(0, 7), fontsize=7.5, ha='center', color=colour)
    # Second x-axis in the units that actually matter: the phrasing block's vote.
    # Both directions receive arrays, so guard elementwise — `max()` on an array raises.
    def _share(v):  v = np.asarray(v, float); return v**2/(1+v**2)*100
    def _alpha(p):  r = np.asarray(p, float)/100; return np.sqrt(r/np.maximum(1e-9, 1-r))
    tw = axes[0].secondary_xaxis('top', functions=(_share, _alpha))
    tw.set_xlabel('措辞话语权 α²/(1+α²)  (%)', fontsize=8.5)
    fig.suptitle(f'α 决策: 当选 α={A} — 依据「{sw["chosen_by"]}」', fontsize=11)
    plt.tight_layout(); SAVE(fig, 'fig2_alpha'); plt.show()

    share = A**2/(1+A**2)
    print(f'α = {A}  →  措辞话语权 = α²/(1+α²) = {A}²/(1+{A}²) = {share:.1%}')
    print('注意这是 α 的 **平方**。α=0.5 看着像「轻推一下」, 实际是把 20% 的投票权交给了措辞。')
    if sw.get('silhouette_disagrees'):
        print(f'⚠️  silhouette 会选 α={sw["silhouette_would_have_chosen"]} — 已记录并否决 (原则三)')
    else:
        print('本次 silhouette 与主裁判恰好同选 — 属于巧合, 不改变它没有投票权这一点。')
else:
    print('α-sweep 未产出 (fast_mode?)')"""


def fig_battery() -> str:
    """Algorithm bake-off scatter: 'upper right is better'."""
    return """# %% 图 3 — 算法 battery: 淘汰赛全景 (右上更好)
try:
    battery = J('battery')
    b = pd.DataFrame(battery['rows'])
    chosen = battery['verdict']['chosen']
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    fam_of = b['algorithm'].str.replace(r'_k\\d+$', '', regex=True)
    palette = {f: c for f, c in zip(sorted(fam_of.unique()),
               ['#2a78d6','#eb6834','#1baf7a','#9257c9','#c9a227','#5c6f7a','#d24d78'] * 4)}
    for f in sorted(fam_of.unique()):
        m = (fam_of == f).to_numpy()
        ax.scatter(b.loc[m, 'silhouette'], b.loc[m, 'stability_ari'], s=88, alpha=.82,
                   color=palette[f], label=f, edgecolor='white', linewidth=.9, zorder=3)
    for _, r in b.iterrows():
        ax.annotate(str(r['algorithm']).replace('_', ' '), (r['silhouette'], r['stability_ari']),
                    xytext=(0, -13), textcoords='offset points', fontsize=6.8,
                    ha='center', color='#4a4a4a')
    hit = b[b['algorithm'] == chosen]
    if len(hit):
        r = hit.iloc[0]
        ax.scatter([r['silhouette']], [r['stability_ari']], s=340, facecolor='none',
                   edgecolor=ORANGE, linewidth=2.6, zorder=4)
        ax.annotate('当选', (r['silhouette'], r['stability_ari']), xytext=(13, 11),
                    textcoords='offset points', color=ORANGE, fontweight='bold', fontsize=10)
    ax.set_xlabel('silhouette  (仅参考 — 横轴没有投票权)')
    ax.set_ylabel('稳定性 ARI  (主裁判 — 纵轴决定名次)')
    ax.set_title(f'算法选优 battery ({len(b)} 个配置): 只有纵轴在裁决\\n当选 {chosen} — {battery["verdict"]["chosen_by"]}',
                 fontsize=10.5)
    ax.grid(alpha=.25, ls=':'); ax.legend(fontsize=8, title='算法族', title_fontsize=8.5)
    plt.tight_layout(); SAVE(fig, 'fig3_battery'); plt.show()

    rank = pd.DataFrame(battery['verdict']['ranking'])
    print('按主裁判排名 (前 5):'); display(rank.head(5))
    if battery['verdict'].get('density_note'): print('密度类算法:', battery['verdict']['density_note'])
except Exception as e:
    print('battery 未产出:', type(e).__name__, e)"""


def fig_umap_families() -> str:
    """Side-by-side projections: what the alpha choice did to the geometry."""
    return f"""# %% 图 4 — 嵌入空间全景: α 到底改变了什么形状
{_PROJECT}
spaces = [('emb_base', f'语义底座 (α=0)'), ('emb_hybrid', f'hybrid (α={{rep["alpha_sweep"].get("chosen_alpha","?")}})')]
avail = [(k, t) for k, t in spaces if (GEN/f'{{k}}.npy').exists()]
fig, axes = plt.subplots(1, len(avail), figsize=(6.4*len(avail), 5.6), squeeze=False)
for ax, (key, title) in zip(axes[0], avail):
    X = NPY(key); P, idx, how = project(X)
    f = famrow[idx]
    nF = len(np.unique(f))
    ax.scatter(P[:,0], P[:,1], c=f, cmap='tab20', s=3.2, alpha=.62, linewidths=0)
    ax.set_title(f'{{title}} — {{nF}} 个家族\\n{{how}}', fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle('两个嵌入空间的 2-D 投影 (着色 = 最终家族)', fontsize=11.5)
plt.tight_layout(); SAVE(fig, 'fig4_spaces'); plt.show()
print('读图提示: 投影只用于「看形状」, 不用于判定。所有裁决指标都在原始高维空间上计算 —')
print('2-D 投影必然丢信息, 用它下结论是本方法论明确禁止的一步。')"""


def fig_umap_intent() -> str:
    """The single most persuasive figure: one intent, scattered across families."""
    return f"""# %% 图 5 — 同一意图被劈进几个家族? (模板群 = 已知同意图的探针)
{_PROJECT}
masks = J('template_masks') if (GEN/'template_masks.json').exists() else {{}}
groups = sorted(tmpl['groups'], key=lambda g: -g['n_hits'])[:3]
if groups and (GEN/'emb_hybrid.npy').exists():
    X = NPY('emb_hybrid'); P, idx, how = project(X)
    famv = famrow[idx]
    q = df['query'].astype(str).to_numpy()[idx]
    fig, axes = plt.subplots(1, len(groups), figsize=(5.9*len(groups), 5.4), squeeze=False)
    for ax, g in zip(axes[0], groups):
        import re
        hit = np.array([bool(re.search(g['pattern'], s)) for s in q])
        ax.scatter(P[~hit,0], P[~hit,1], c='#d9d9d9', s=2.4, alpha=.45, linewidths=0)
        # Remap the families actually present to 0..n-1 so no two share a colour.
        present = np.unique(famv[hit]) if hit.sum() else np.array([])
        rank = {{int(v): i for i, v in enumerate(present)}}
        cmap = 'tab10' if len(present) <= 10 else 'tab20'
        ax.scatter(P[hit,0], P[hit,1],
                   c=[rank[int(v)] for v in famv[hit]] if hit.sum() else [],
                   cmap=cmap, vmin=0, vmax=max(1, len(present)-1), s=15, alpha=.92,
                   linewidths=.25, edgecolor='white')
        # Effective family count = exp(Shannon entropy) — the fragmentation metric,
        # computed here for this one intent so the picture and the number agree.
        if hit.sum():
            _, cnt = np.unique(famv[hit], return_counts=True)
            p = cnt / cnt.sum(); ef = float(np.exp(-(p*np.log(p)).sum()))
        else:
            ef = float('nan')
        ax.set_title(f"{{g['name']}}  (n={{int(hit.sum())}})\\n有效散布于 {{ef:.2f}} 个家族", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle('同一意图 (彩色, 按家族着色) 被劈开的程度 — 灰点为其余 query', fontsize=11.5)
    plt.tight_layout(); SAVE(fig, 'fig5_intent_split'); plt.show()
    print('「有效家族数」= exp(香农熵), 与图 1/图 2 里的碎裂度是同一个公式 —')
    print('1.00 表示该意图完整落在一个家族里; 3.00 表示它被切成了三份等大的碎片。')
else:
    print('模板群或 hybrid 空间缺失, 跳过。')"""


def fig_panel_bars() -> str:
    """The uniform panel, as bars, with the advisory metric visually demoted."""
    return """# %% 图 6 — 统一度量面板: 同一把尺子量所有候选
rows = pd.DataFrame(panel['rows'])
name_col = 'candidate' if 'candidate' in rows else rows.columns[0]
spec = [('template_fragmentation', '模板碎裂度 ↓', GREEN, True),
        ('stability_ari',          '重播稳定性 ARI ↑', BLUE,  False),
        ('nmi_reference',          'NMI vs 参照标签 ↑', '#9257c9', False),
        ('ambiguous_rate',         '模糊行占比 ↓', '#c9a227', True),
        ('silhouette',             'silhouette (无投票权)', MUTED, False)]
missing = [c for c, *_ in spec if c not in rows.columns or not rows[c].notna().any()]
spec = [s for s in spec if s[0] not in missing]
fig, axes = plt.subplots(1, len(spec), figsize=(3.15*len(spec), 4.5), squeeze=False)
for ax, (col, title, colour, lower_better) in zip(axes[0], spec):
    d = rows[[name_col, col]].dropna().sort_values(col, ascending=lower_better)
    bars = ax.barh([str(x) for x in d[name_col]], d[col], color=colour,
                   alpha=.55 if col == 'silhouette' else .9,
                   hatch='//' if col == 'silhouette' else None)
    for b, v in zip(bars, d[col]):
        ax.text(b.get_width(), b.get_y()+b.get_height()/2, f' {v:.3f}',
                va='center', fontsize=8, color='#333')
    ax.set_title(title, fontsize=10)
    ax.tick_params(axis='y', labelsize=8)
    ax.grid(alpha=.22, ls=':', axis='x')
    ax.margins(x=.18)
fig.suptitle('统一面板 — 同一 panel_id, 同一子样本, 同一随机种子。'
             '斜纹条 = 只报告不投票的指标', fontsize=10.5)
plt.tight_layout(); SAVE(fig, 'fig6_panel'); plt.show()
if missing:
    print('本次面板未产出的指标 (已从图中略去, 而不是画成 0):', ', '.join(missing))
print('这张图的意义不在某根条更长, 而在于**所有条都是同一把尺子量出来的**。')
print('跨 run、跨 notebook 抄来的数字不能进这张图 — 那是本方法论最常见的失效方式。')"""


def save_helper(fig_dir: str) -> str:
    """Cell fragment defining SAVE(); every figure is also written as PNG."""
    return f"""FIGDIR = Path({fig_dir!r}); FIGDIR.mkdir(parents=True, exist_ok=True)
def SAVE(fig, name):
    p = FIGDIR / f'{{name}}.png'
    fig.savefig(p, dpi=150, bbox_inches='tight')
    return p"""


ALL = ("fig1_ksweep", "fig2_alpha", "fig3_battery",
       "fig4_spaces", "fig5_intent_split", "fig6_panel")
