"""Phase 11 figures.

Every plot here answers a question a reader will actually ask, and several are
designed to make a *mistake* visible rather than to make the result look good.
The alpha decision plot draws the silhouette curve next to the fragmentation
curve precisely so the reader can see them disagree; the template-spread plot
exists to show phrasing families being torn across clusters.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, Sequence

import numpy as np

log = logging.getLogger("qmine.viz")


def setup_matplotlib(language: str = "zh") -> Any:
    """Configure matplotlib once, including CJK fonts.

    Without this every Chinese label renders as a row of empty boxes and the
    font manager floods the log with warnings — a small thing that makes an
    otherwise finished deliverable look broken.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if language in ("zh", "multi"):
        plt.rcParams["font.family"] = [
            "Arial Unicode MS", "Heiti TC", "PingFang SC", "Songti SC",
            "Noto Sans CJK SC", "SimHei", "DejaVu Sans",
        ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 130
    plt.rcParams["savefig.bbox"] = "tight"
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", message=".*missing from font.*")
    return plt


def plot_k_sweep(sweep: list[dict[str, Any]], path: Path, chosen_k: int | None = None, language: str = "zh") -> Path:
    """Stability and silhouette against K, with the chosen point marked."""
    plt = setup_matplotlib(language)
    ks = [r["k"] for r in sweep]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(ks, [r["stability_ari"] for r in sweep], "o-", color="#2563eb",
            label="replay stability (ARI) — decisive")
    ax.set_xlabel("K (number of clusters)")
    ax.set_ylabel("stability ARI", color="#2563eb")
    ax.tick_params(axis="y", labelcolor="#2563eb")
    ax2 = ax.twinx()
    ax2.plot(ks, [r["silhouette"] for r in sweep], "s--", color="#9ca3af",
             label="silhouette — advisory, no vote")
    ax2.set_ylabel("silhouette", color="#9ca3af")
    ax2.tick_params(axis="y", labelcolor="#9ca3af")
    if chosen_k:
        ax.axvline(chosen_k, color="#dc2626", ls=":", lw=1.6)
        ax.annotate(f"chosen K={chosen_k}\n(stability peak)", xy=(chosen_k, max(r["stability_ari"] for r in sweep)),
                    xytext=(6, -28), textcoords="offset points", color="#dc2626", fontsize=9)
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], loc="best", fontsize=8)
    ax.set_title("Granularity: the two curves do not peak together")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_alpha_decision(rows: list[dict[str, Any]], path: Path, chosen: float | None = None, language: str = "zh") -> Path:
    """The alpha decision, drawn so the proxy's disagreement is visible."""
    plt = setup_matplotlib(language)
    a = [r["alpha"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax1.plot(a, [r["template_fragmentation"] for r in rows], "o-", color="#dc2626",
             label="template fragmentation (lower better) — decisive")
    ax1.set_xlabel("alpha"); ax1.set_ylabel("effective clusters per phrasing family")
    ax1b = ax1.twinx()
    ax1b.plot(a, [r["stability_ari"] for r in rows], "^-", color="#2563eb",
              label="replay stability — decisive")
    ax1b.set_ylabel("stability ARI")
    if chosen is not None:
        ax1.axvline(chosen, color="#16a34a", ls=":", lw=1.8)
    lines = ax1.get_lines() + ax1b.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], fontsize=8, loc="best")
    ax1.set_title("What decided alpha")

    ax2.plot(a, [r["silhouette"] for r in rows], "s--", color="#9ca3af", label="silhouette")
    ax2.plot(a, [r["surface_vote_share"] for r in rows], "d-", color="#7c3aed",
             label="phrasing's share of the cosine (alpha²/(1+alpha²))")
    if chosen is not None:
        ax2.axvline(chosen, color="#16a34a", ls=":", lw=1.8)
    ax2.set_xlabel("alpha")
    ax2.legend(fontsize=8)
    ax2.set_title("Silhouette gets no vote here\n(it rewards phrasing-tight clusters)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_template_spread(spread: dict[str, Any], path: Path, language: str = "zh") -> Path:
    """How far each phrasing family was scattered — the twin-family photograph."""
    plt = setup_matplotlib(language)
    names = list(spread)[:12]
    if not names:
        return path
    fig, ax = plt.subplots(figsize=(8, max(3.2, 0.42 * len(names) + 1)))
    for i, n in enumerate(names):
        shares = [c["share"] for c in spread[n]["clusters"]]
        left = 0.0
        for j, s in enumerate(shares):
            ax.barh(i, s, left=left, color=plt.cm.tab20(j % 20), edgecolor="white", height=0.62)
            if s > 0.08:
                ax.text(left + s / 2, i, f"{s * 100:.0f}%", va="center", ha="center", fontsize=7.5)
            left += s
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([f"{n}\n(n={spread[n]['n_members']}, {spread[n]['n_clusters_touched']} clusters)"
                        for n in names], fontsize=8)
    ax.set_xlabel("share of the phrasing family's members")
    ax.set_title("Where each phrasing family landed\nOne solid bar = one intent, one cluster. Many bands = a twin split.")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_panel(table: dict[str, Any], path: Path, metrics: Sequence[str] = (), language: str = "zh") -> Path:
    """The comparison panel as grouped bars, annotated with metric authority."""
    plt = setup_matplotlib(language)
    metrics = list(metrics) or ["stability_ari", "template_fragmentation", "silhouette"]
    rows = table["rows"]
    auth = {m["name"]: m["authority"] for m in table["metrics"]}
    subjects = [r["subject"] for r in rows]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 3.8))
    if len(metrics) == 1:
        axes = [axes]
    colors = {"decisive": "#2563eb", "advisory": "#9ca3af", "diagnostic": "#a3a3a3"}
    for ax, m in zip(axes, metrics):
        vals = [r.get(m) if r.get(m) is not None else np.nan for r in rows]
        ax.bar(range(len(subjects)), vals, color=colors.get(auth.get(m, "diagnostic"), "#a3a3a3"))
        ax.set_xticks(range(len(subjects)))
        ax.set_xticklabels(subjects, rotation=38, ha="right", fontsize=7.5)
        ax.set_title(f"{m}\n({auth.get(m, 'diagnostic')})", fontsize=9.5)
    fig.suptitle(f"Uniform panel {table['panel_id']} — same code, same sub-sample, same seed", fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_umap(
    X: np.ndarray, labels: np.ndarray, path: Path, *, sample: int = 12000, seed: int = 42,
    title: str = "", language: str = "zh",
) -> Path | None:
    """A 2-D projection, on a fixed sub-sample so two spaces can be compared.

    The projection is for *looking*, never for clustering — cluster structure in
    a UMAP layout is partly an artefact of the layout.
    """
    try:
        import umap
    except Exception as exc:  # noqa: BLE001
        log.warning("umap unavailable: %s", exc)
        return None
    from ..determinism import deterministic_subsample

    plt = setup_matplotlib(language)
    idx = deterministic_subsample(len(X), min(sample, len(X)), seed)
    emb = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=seed).fit_transform(X[idx])
    fig, ax = plt.subplots(figsize=(6.6, 6))
    ax.scatter(emb[:, 0], emb[:, 1], c=labels[idx] % 20, cmap="tab20", s=2.2, alpha=0.62, linewidths=0)
    ax.set_title(title or "Corpus layout (UMAP, cosine)")
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.01, 0.01, "layout only — distances here are not the distances the tree was built on",
            transform=ax.transAxes, fontsize=7, color="#6b7280")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_refinement(history: list[dict[str, Any]], path: Path, language: str = "zh") -> Path:
    """Leaf count and row movement across refinement rounds."""
    plt = setup_matplotlib(language)
    if not history:
        return path
    r = [h["round"] for h in history]
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    ax.plot(r, [h["n_leaves"] for h in history], "o-", color="#2563eb", label="leaves")
    ax.set_xlabel("refinement round"); ax.set_ylabel("leaf count", color="#2563eb")
    ax2 = ax.twinx()
    ax2.plot(r, [h["moved_fraction"] * 100 for h in history], "s--", color="#dc2626", label="rows moved %")
    ax2.set_ylabel("rows moved (%)", color="#dc2626")
    ax.set_title("Refinement converges when movement stops, not at a fixed round count")
    ax.set_xticks(r)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_decision_chain(decisions: list[dict[str, Any]], path: Path, language: str = "zh",
                        localise: Any = None) -> Path:
    """The parameter search as one picture: candidates considered, and survivors.

    A reader can follow the decision table, but they cannot *see* it — and the
    shape of this search is the most-asked question about the pipeline ("how many
    things did you actually try?"). Drawn as a funnel per decision, with the
    deciding metric named on each bar, the whole chain fits in one glance and the
    steps that considered exactly one candidate become visibly conspicuous.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    zh = language == "zh"
    rows = [d for d in decisions if d.get("question")]
    if not rows:
        return path
    labels, tried, kept, metrics = [], [], [], []
    for d in rows:
        n_rej = len(d.get("rejected") or [])
        q = str(d.get("question", ""))
        if localise is not None:
            q = localise(q)
        labels.append(f"[{d.get('phase','?')}] {q[:24]}")
        tried.append(n_rej + 1)
        kept.append(1)
        metrics.append(", ".join(d.get("decisive_metrics") or []) or ("—"))

    h = max(3.2, 0.62 * len(rows))
    fig, ax = plt.subplots(figsize=(11.5, h))
    y = list(range(len(rows)))[::-1]
    ax.barh(y, tried, color="#d6d9de", height=0.62,
            label="考虑过的候选" if zh else "candidates considered")
    ax.barh(y, kept, color="#2a78d6", height=0.62,
            label="最终采纳" if zh else "adopted")
    for yi, t, m, lab in zip(y, tried, metrics, labels):
        ax.text(t + 0.12, yi, f"{t} 选 1" if zh else f"{t}→1", va="center", fontsize=8.5,
                color="#333")
        ax.text(t + 1.15, yi, f"裁定: {m}" if zh else f"by: {m}", va="center",
                fontsize=7.8, color="#6a6a6a")
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("候选数" if zh else "candidates")
    ax.set_xlim(0, max(tried) + 4.5)
    ax.set_title("决策链: 每个环节试了几个方案, 由哪个指标裁定"
                 if zh else "Decision chain: candidates per step and the deciding metric",
                 fontsize=11)
    # Outside the axes: with a short chain the last bar reaches the lower right.
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.005, 1.0), frameon=False)
    ax.grid(axis="x", alpha=.25, ls=":")
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    return path


def plot_gates(gates: dict[str, Any], path: Path, language: str = "zh") -> Path:
    """Every quality gate as distance from its own bar, on one normalised axis.

    Gates carry incomparable units — a kappa, a coverage share, an agreement rate
    — so the only way to show them together is as *headroom*: how far the observed
    value sits above or below the threshold it was judged against, in units of that
    threshold. Zero is the bar. Anything left of zero failed.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    zh = language == "zh"
    colour = {"passed": "#1baf7a", "warned": "#c9a227", "failed": "#d24d78",
              "rejected": "#8e44ad", "skipped": "#b0b0b0"}
    rows = []
    for name, g in gates.items():
        obs = g.get("observed") if isinstance(g, dict) else getattr(g, "observed", None)
        thr = g.get("threshold") if isinstance(g, dict) else getattr(g, "threshold", None)
        st = g.get("status") if isinstance(g, dict) else getattr(g, "status", "")
        o = next((v for v in (obs or {}).values() if isinstance(v, (int, float))), None)
        t = next((v for v in (thr or {}).values() if isinstance(v, (int, float))), None)
        if o is None or t in (None, 0):
            continue
        rows.append((name, (o - t) / abs(t), st, o, t))
    if not rows:
        return path
    rows.sort(key=lambda r: r[1])

    fig, ax = plt.subplots(figsize=(10.5, max(2.8, 0.5 * len(rows))))
    y = list(range(len(rows)))
    ax.barh(y, [r[1] for r in rows], color=[colour.get(r[2], "#888") for r in rows], height=0.6)
    ax.axvline(0, color="#333", lw=1.4)
    for yi, r in zip(y, rows):
        ax.text(r[1] + (0.01 if r[1] >= 0 else -0.01), yi,
                f"{r[3]:.3g} vs {r[4]:.3g}", va="center",
                ha="left" if r[1] >= 0 else "right", fontsize=8, color="#333")
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
    ax.set_xlabel("相对门槛的余量 (0 = 恰好达标)" if zh else "headroom relative to threshold")
    ax.set_title("质量门: 每一道门离自己的门槛有多远" if zh else "Quality gates: headroom", fontsize=11)
    ax.margins(x=.22); ax.grid(axis="x", alpha=.25, ls=":")
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    return path
