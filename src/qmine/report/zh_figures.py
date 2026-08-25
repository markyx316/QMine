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

# Every projection figure compares the SAME set of spaces, so a reader can carry an
# impression from one picture into the next. WHICH spaces is a decision, not a
# constant: the reference deliverable hard-coded base / a=0.5 / a=0.1 because those
# were that project's three attempts. Generalised here to base, the alpha the run
# chose, and — when they differ — the alpha *silhouette* would have chosen. That
# third panel is Principle 3 rendered as a picture: it shows what the rejected
# criterion would have built, next to what was built instead.
_SPACES = "\ndef spaces_to_compare():\n    # [(alpha, label), ...] - base, the chosen alpha, and a contrast.\n    sw = rep.get('alpha_sweep', {}) or {}\n    rows = sw.get('rows', []) or []\n    chosen = float(sw.get('chosen_alpha') or 0.0)\n    out = [(0.0, 'base (α=0, 纯语义)')]\n    if chosen > 0:\n        out.append((chosen, f'hybrid α={chosen} (最终)'))\n    sil = sw.get('silhouette_would_have_chosen')\n    if sil is None and rows:\n        sil = max(rows, key=lambda r: r.get('silhouette', -9)).get('alpha')\n    if sil is not None and float(sil) > 0 and abs(float(sil) - chosen) > 1e-9:\n        out.append((float(sil), f'hybrid α={sil} (silhouette 会选)'))\n    else:\n        alt = [r['alpha'] for r in rows if abs(r['alpha'] - chosen) > 1e-9 and r['alpha'] > 0]\n        if alt:\n            out.append((float(max(alt)), f'hybrid α={max(alt)}'))\n    return out[:3]\n\n\ndef space_matrix(alpha):\n    # Rebuild a hybrid space at any alpha from the two blocks already on disk.\n    if not alpha or not (GEN/'emb_svd_char.npy').exists():\n        return NPY('emb_base')\n    from qmine.ops.represent import hybrid\n    return hybrid(NPY('emb_base'), NPY('emb_svd_char'), float(alpha))\n\n\ndef families_in(X, k, seed=0):\n    # Each panel is coloured by ITS OWN families, not the chosen space's - the\n    # spaces disagreeing about what the families ARE is the point of the figure.\n    from sklearn.cluster import KMeans\n    return KMeans(n_clusters=int(k), n_init=4, random_state=seed).fit_predict(X)\n"

# Shared by every projection figure: a 2-D view of a high-dimensional space,
# preferring UMAP's local-structure preservation but never requiring it.
_PROJECT = "\ndef project(X, seed=20240601, n=6000):\n    # 2-D view of X. UMAP when available (keeps local neighbourhoods), else PCA.\n    idx = np.arange(len(X))\n    if len(X) > n:\n        idx = np.random.RandomState(seed).choice(len(X), n, replace=False)\n    Xs = X[idx]\n    try:\n        import umap\n        P = umap.UMAP(n_neighbors=15, min_dist=0.10, metric='cosine',\n                      random_state=seed).fit_transform(Xs)\n        how = 'UMAP(cosine, n_neighbors=15)'\n    except Exception as e:\n        from sklearn.decomposition import PCA\n        P = PCA(n_components=2, random_state=seed).fit_transform(Xs)\n        how = f'PCA (UMAP 不可用: {type(e).__name__})'\n    return P, idx, how\n"


def fig_ksweep(chosen_k_expr: str = "tri['chosen_family_k']") -> str:
    """K sweep — three metrics, and one line per candidate space.

    Two things at once, both from the reference deliverable. Across panels: the
    three metrics rarely peak together, which is why the run must name its judge
    before it looks. Across lines within a panel: the spaces do not even agree on
    the shape of the curve, so a K chosen in one space is not transferable to
    another. Recomputed on a sub-sample because this is a picture, not a
    decision — the delivered K comes from the full-effort sweep in Phase 5.
    """
    return f"""# %% 图 1 — K 扫描: 三个指标 × 各候选空间
{_SPACES}
ks = pd.DataFrame(gran['k_sweep']).sort_values('k')
K = {chosen_k_expr}
_grid = [int(k) for k in ks['k']][:10]
_sub = np.random.RandomState(0).choice(len(df), min(6000, len(df)), replace=False)

def sweep(X, grid):
    from sklearn.metrics import adjusted_rand_score
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    out = []
    Xs = X[_sub]
    for k in grid:
        a1 = KMeans(k, n_init=2, random_state=0).fit_predict(Xs)
        a2 = KMeans(k, n_init=2, random_state=1).fit_predict(Xs)
        out.append({{'k': k,
                    'silhouette': float(silhouette_score(Xs, a1, metric='cosine')),
                    'stability_ari': float(adjusted_rand_score(a1, a2))}})
    return pd.DataFrame(out)

_curves = {{}}
for al, label in spaces_to_compare():
    try:
        _curves[label] = sweep(space_matrix(al), _grid)
    except Exception as exc:
        print(f'  {{label}}: 扫描失败 ({{type(exc).__name__}})')

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
_cols = [BLUE, ORANGE, GREEN, MUTED]
for ax, (col, title) in zip(axes, [('stability_ari', '重播稳定性 ARI vs K  (主裁判)'),
                                   ('silhouette',    'silhouette vs K  (同一空间内有票, 跨空间不可比)')]):
    for (label, cur), colour in zip(_curves.items(), _cols):
        ax.plot(cur['k'], cur[col], 'o-', ms=4.5, lw=1.8, color=colour, label=label)
    ax.axvline(K, color='#c0392b', ls=':', lw=1.6)
    ax.set_xlabel('K'); ax.set_title(title, fontsize=10)
    ax.grid(alpha=.25, ls=':'); ax.legend(fontsize=8)
_loc = str(tri.get('locator', '')).split(' ')[0]
_locz = {{'intent_alignment_ami': '意图对齐 AMI', 'stability_ari': '重播稳定性 ARI'}}.get(_loc, _loc or '未记录')
fig.suptitle(f'K 扫描 (6k 子样重算): 定案 K={{K}} 由「{{_locz}}」定位, 稳定性只用于否决不可复现的 K — 各空间曲线形状并不一致', fontsize=10)
plt.tight_layout(); SAVE(fig, 'fig1_ksweep'); plt.show()

# The full-effort sweep for the chosen space, including the metric the two
# panels above cannot show, because fragmentation needs the template masks.
fig2, axes2 = plt.subplots(1, 3, figsize=(13.5, 3.7))
spec = [('stability_ari',        '稳定性 ARI ↑ (主裁判)',        BLUE,  'o-'),
        ('template_fragmentation','模板碎裂度 ↓ (主裁判)',        GREEN, 'D-'),
        ('silhouette',           'silhouette (同一空间内可比)',   MUTED, 's--')]
for ax, (col, title, colour, style) in zip(axes2, spec):
    if col not in ks: ax.axis('off'); continue
    ax.plot(ks['k'], ks[col], style, color=colour, lw=1.9, ms=5)
    ax.axvline(K, color=ORANGE, ls=':', lw=2)
    ax.margins(y=.18)
    best = ks.loc[ks[col].idxmin() if 'frag' in col else ks[col].idxmax()]
    if int(best['k']) != int(K):
        ax.plot(best['k'], best[col], '*', color=colour, ms=15, zorder=5)
        lo, hi = ax.get_ylim(); x0, x1 = ax.get_xlim()
        fx = (best['k'] - x0) / max(1e-9, x1 - x0); fy = (best[col] - lo) / max(1e-9, hi - lo)
        tx, ha = (0.97, 'right') if fx < 0.5 else (0.03, 'left')
        ty, va = (0.97, 'top') if fy < 0.5 else (0.03, 'bottom')
        ax.annotate(f"该指标自己的峰 K={{int(best['k'])}}", xy=(best['k'], best[col]),
                    xytext=(tx, ty), textcoords='axes fraction', fontsize=8, color=colour,
                    ha=ha, va=va, arrowprops=dict(arrowstyle='-', color=colour, lw=.7,
                                                  alpha=.55, shrinkA=2, shrinkB=7))
    ax.set_xlabel('K'); ax.set_title(title, fontsize=10); ax.grid(alpha=.25, ls=':')
fig2.suptitle('当选空间的全量 K 扫描: 三个指标很少同峰 — 所以必须先指定谁是裁判', fontsize=11)
plt.tight_layout(); SAVE(fig2, 'fig1b_ksweep_metrics'); plt.show()

_peaks = {{c: int(ks.loc[ks[c].idxmin() if 'frag' in c else ks[c].idxmax(), 'k'])
           for c, *_ in spec if c in ks}}
print('各指标各自的峰值 K:', _peaks)
print(f'定案 K = {{K}} — 依据「{{tri["chosen_by"]}}」, 而不是三者的平均。')"""


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
    """The algorithm bake-off, laid out as the reference deliverable lays it out.

    Stability on x, silhouette on y, and *upper right is better* — a Pareto view
    rather than a ranked list. That framing is deliberate: within a single fixed
    representation every configuration encodes phrasing identically, so
    silhouette's phrasing bias is a constant offset across the panel and its
    variation carries real information about geometric fit. It is between
    representations, where the bias tracks the very thing being varied, that
    silhouette stops being comparable. Density algorithms are drawn as crosses
    with their noise rate, because a silhouette computed over a partition that
    discards 43% of the corpus as noise is not on the same axis as one that
    keeps everything.
    """
    return """# %% 图 3 — 算法 battery: 淘汰赛全景 (右上更好)
try:
    battery = J('battery')
    rows = pd.DataFrame(battery['rows'])
    chosen = battery['verdict']['chosen']
    algo_fam = rows['algorithm'].str.replace(r'_k\\d+$', '', regex=True)
    is_density = rows.get('noise_rate', pd.Series(0, index=rows.index)).fillna(0) > 0.01

    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    palette = {f: c for f, c in zip(sorted(algo_fam.unique()),
               ['#4c3bcf','#eb6834','#1baf7a','#2a78d6','#c9a227','#9257c9','#d24d78'] * 4)}
    for f in sorted(algo_fam.unique()):
        m = ((algo_fam == f) & ~is_density).to_numpy()
        if m.any():
            ax.scatter(rows.loc[m, 'stability_ari'], rows.loc[m, 'silhouette'], s=190,
                       alpha=.88, color=palette[f], label=f, edgecolor='white',
                       linewidth=1.1, zorder=3)
    for i, r in rows[is_density].iterrows():
        ax.scatter([r['stability_ari']], [r['silhouette']], marker='x', s=150,
                   color=MUTED, linewidth=2.2, zorder=3)
        ax.annotate(f"HDBSCAN {int(r['n_clusters'])}簇/噪声{r['noise_rate']:.1%}",
                    (r['stability_ari'], r['silhouette']), xytext=(9, 4),
                    textcoords='offset points', fontsize=7.5, color=MUTED)
    for _, r in rows[~is_density].iterrows():
        k = str(r['algorithm']).split('_k')[-1]
        ax.annotate(f'k={k}', (r['stability_ari'], r['silhouette']), xytext=(0, -14),
                    textcoords='offset points', fontsize=7, ha='center', color='#4a4a4a')
    hit = rows[rows['algorithm'] == chosen]
    if len(hit):
        r = hit.iloc[0]
        ax.scatter([r['stability_ari']], [r['silhouette']], s=430, facecolor='none',
                   edgecolor=ORANGE, linewidth=2.6, zorder=4)
        ax.annotate('当选', (r['stability_ari'], r['silhouette']), xytext=(15, 12),
                    textcoords='offset points', color=ORANGE, fontweight='bold', fontsize=10)
    ax.set_xlabel('重播稳定性 ARI  (→ 可复现)')
    ax.set_ylabel('silhouette  (→ 结构紧致)')
    ax.set_title(f'算法选优 battery ({len(rows)} 个配置): Upper right is better'
                 f'\\n当选 {chosen} — {battery["verdict"]["chosen_by"]}', fontsize=10.5)
    ax.grid(alpha=.25, ls=':')
    ax.legend(fontsize=8, title='算法族', title_fontsize=8.5, loc='lower right')
    plt.tight_layout(); SAVE(fig, 'fig3_battery'); plt.show()

    print('固定表征下, 两个轴都在说话: 稳定性回答「换个种子还在不在」,')
    print('silhouette 回答「簇是否真的紧而分得开」。同一空间内 silhouette 的措辞偏置是常数偏移,')
    print('所以它的**变化**是可比的 — 这与跨 α 比较时的情形完全不同 (见图 2)。')
    print()
    display(pd.DataFrame(battery['verdict']['ranking']).head(5))
    if battery['verdict'].get('density_note'):
        print('密度类算法:', battery['verdict']['density_note'])
except Exception as e:
    print('battery 未产出:', type(e).__name__, e)"""


def fig_umap_families() -> str:
    """One panel per candidate space, coloured by that space's own families.

    The reference deliverable puts three spaces side by side, and that is the
    whole argument: the same 12k points, projected the same way, partitioned into
    visibly different family structures depending only on how much phrasing the
    representation encodes. A single-space projection is decoration; three of them
    is evidence.
    """
    return f"""# %% 图 4 — 嵌入空间全景: α 到底改变了什么形状
{_SPACES}
{_PROJECT}
_spaces = spaces_to_compare()
K = int(gran['triangulation']['chosen_family_k'])
fig, axes = plt.subplots(1, len(_spaces), figsize=(6.2*len(_spaces), 5.6), squeeze=False)
_seen = {{}}
for ax, (a, title) in zip(axes[0], _spaces):
    X = space_matrix(a)
    P, idx, how = project(X)
    f = families_in(X, K)[idx]           # each space keeps its OWN families
    nF = len(np.unique(f))
    _seen[title] = nF
    ax.scatter(P[:,0], P[:,1], c=f, cmap='tab20', s=3.2, alpha=.62, linewidths=0)
    ax.set_title(f'{{title}} — {{nF}} 个家族', fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle(f'各嵌入空间的 2-D 投影 (同一批点, 同一投影参数; 着色 = 该空间自己的家族)  ·  {{how}}',
             fontsize=11.5)
plt.tight_layout(); SAVE(fig, 'fig4_spaces'); plt.show()
print('读图提示: 投影只用于「看形状」, 不用于判定 —')
print('所有裁决指标都在原始高维空间上计算; 2-D 投影必然丢信息, 用它下结论是本方法论明确禁止的一步。')
print('各空间在家族尺度上的簇数:', _seen)"""


def fig_umap_intent() -> str:
    """The same spaces, with one known-single-intent group lit up in each.

    This is the most persuasive figure in the reference deliverable and the reason
    fragmentation exists as a metric: a phrasing family is a set of queries we
    already know share an intent, so the number of clusters it lands in is a direct
    reading of how badly that space splits intents. Shown across spaces, it is the
    alpha decision argued visually.
    """
    return f"""# %% 图 5 — 同一意图在各空间被劈进几个家族? (模板群 = 已知同意图的探针)
{_SPACES}
{_PROJECT}
import re
_spaces = spaces_to_compare()
K = int(gran['triangulation']['chosen_family_k'])
groups = sorted(tmpl['groups'], key=lambda g: -g['n_hits'])
probe = groups[0] if groups else None
if probe is None:
    print('无模板群, 跳过。')
else:
    q_all = df['query'].astype(str).to_numpy()
    fig, axes = plt.subplots(1, len(_spaces), figsize=(5.9*len(_spaces), 5.6), squeeze=False)
    for ax, (a, title) in zip(axes[0], _spaces):
        X = space_matrix(a)
        P, idx, how = project(X)
        famv = families_in(X, K)[idx]
        q = q_all[idx]
        hit = np.array([bool(re.search(probe['pattern'], t)) for t in q])
        ax.scatter(P[~hit,0], P[~hit,1], c='#d9d9d9', s=2.4, alpha=.45, linewidths=0)
        if hit.sum():
            present = np.unique(famv[hit])
            rank = {{int(v): i for i, v in enumerate(present)}}
            ax.scatter(P[hit,0], P[hit,1], c=[rank[int(v)] for v in famv[hit]],
                       cmap='tab10' if len(present) <= 10 else 'tab20',
                       vmin=0, vmax=max(1, len(present)-1), s=15, alpha=.92,
                       linewidths=.25, edgecolor='white')
            _, cnt = np.unique(famv[hit], return_counts=True)
            pr = cnt / cnt.sum(); ef = float(np.exp(-(pr*np.log(pr)).sum()))
        else:
            ef = float('nan')
        ax.set_title(f'{{title}}\\n「{{probe["name"]}}」有效散布于 {{ef:.2f}} 个家族', fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f'同一意图 (彩色, 按家族着色) 在各空间被劈开的程度 — 灰=其余 query  ·  n={{int(hit.sum())}}',
                 fontsize=11.5)
    plt.tight_layout(); SAVE(fig, 'fig5_intent_split'); plt.show()
    print('「有效家族数」= exp(香农熵), 与碎裂度是同一个公式 —')
    print('1.00 = 该意图完整落在一个家族里; 3.00 = 被切成三份等大的碎片。')
    print('跨面板比较才是重点: 措辞话语权越高的空间, 同一意图散得越开。')"""


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
