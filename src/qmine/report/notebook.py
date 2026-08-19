"""Building the walkthrough notebook programmatically, then executing it.

A notebook assembled by hand is a state you cannot diff and cannot rebuild.  A
notebook assembled by `nbformat` from a script *is* a code artifact, and running
it through `nbclient` before delivery means the numbers in it were computed by
the reader's own kernel rather than pasted in.

Delivering an unexecuted notebook is not delivering a notebook — nobody can tell
whether it runs, and the figures are absent when it matters most.  So a build
that cannot execute cleanly says so instead of shipping empty cells.
"""

from __future__ import annotations

from typing import Any

from ..artifacts import ArtifactRef


def build_walkthrough(state: Any, deps: Any) -> ArtifactRef:
    """Assemble and execute the end-to-end walkthrough for this run."""
    import nbformat

    gen = str(deps.store.gen_dir)
    cells: list[Any] = []

    def md(text: str) -> None:
        cells.append(nbformat.v4.new_markdown_cell(text))

    def code(src: str) -> None:
        cells.append(nbformat.v4.new_code_cell(src))

    md(
        f"# QMine walkthrough — run `{state.get('run_id')}`\n\n"
        f"Domain `{deps.cfg.domain.key}` · generation {state.get('generation')} · "
        f"config hash `{deps.cfg.config_hash}`\n\n"
        f"> {deps.registry.provenance_note()}\n\n"
        "Every number below is computed by this notebook from the artifacts on disk. "
        "Nothing is pasted."
    )
    code(
        "import json, numpy as np, pandas as pd\n"
        "from pathlib import Path\n"
        f"GEN = Path({gen!r})\n"
        "def load(name, ext='json'):\n"
        "    p = GEN / f'{name}.{ext}'\n"
        "    if ext == 'json':  return json.loads(p.read_text())\n"
        "    if ext == 'npy':   return np.load(p)\n"
        "    if ext == 'csv':   return pd.read_csv(p)\n"
        "    return pd.read_parquet(p)\n"
        "sorted(p.name for p in GEN.iterdir())[:30]"
    )

    md("## Phase 1 — what the corpus is")
    code(
        "audit = load('data_audit')\n"
        "print('rows      ', audit['n_rows'])\n"
        "print('unique    ', audit['n_unique'])\n"
        "print('length p50', audit['length']['p50'], ' p90', audit['length']['p90'])\n"
        "print('script mix', audit['script_mix'])"
    )
    md(
        "### Phrasing families\n\n"
        "These are the sets of queries we *know* share an intent without anyone labelling "
        "anything. They judge the alpha sweep, define the fragmentation metric, and supply "
        "display exemplars — mined once, spent three times."
    )
    code(
        "tg = load('template_groups')\n"
        "pd.DataFrame([{'name': g['name'], 'n': g['n_hits'], 'share': g['share'],\n"
        "               'source': 'mined' if g['discovered'] else 'seed',\n"
        "               'example': g['examples'][0] if g['examples'] else ''}\n"
        "              for g in tg['groups']])"
    )
    code("print('union coverage: %.1f%%  (gate window 20-40%%)' % (tg['coverage']['union_coverage']*100))")

    md(
        "## Phase 3 — the alpha sweep\n\n"
        "`H = L2norm([e ⊕ α·s])`. Because both blocks are unit vectors,\n\n"
        "`cos(H,H′) = (cos_semantic + α²·cos_surface) / (1 + α²)`\n\n"
        "so the phrasing block votes with weight **α², not α**. Watch the fourth column: at "
        "α=0.5 phrasing controls 20% of the cosine, which is enough to outvote semantics on "
        "near-ties and split one intent into several phrasing-shaped families."
    )
    code(
        "rep = load('representation')\n"
        "df = pd.DataFrame(rep['alpha_sweep']['rows'])[\n"
        "    ['alpha','surface_vote_share','template_fragmentation','stability_ari','silhouette']]\n"
        "print('chosen alpha:', rep['alpha_sweep']['chosen_alpha'])\n"
        "print('silhouette would have chosen:', rep['alpha_sweep']['silhouette_would_have_chosen'])\n"
        "df"
    )

    md("## Phase 5 — granularity, triangulated")
    code(
        "g = load('granularity')\n"
        "print(json.dumps(g['triangulation']['estimates'], indent=1))\n"
        "print('chosen K =', g['triangulation']['chosen_family_k'], '| converged:', g['triangulation']['converged'])\n"
        "pd.DataFrame(g['k_sweep'])"
    )

    md("## Phase 6 — the tree, and proof it reproduces")
    code(
        "meta = load('hierarchy_meta')\n"
        "print('families', meta['n_families'], '→ leaves', meta['n_leaves'])\n"
        "print('held-out structure reproduction:', meta['heldout_reproduction']['agreement'])\n"
        "pd.DataFrame(meta['refinement_history'])"
    )

    md(
        "## Phase 7 — blind naming\n\n"
        "Namers saw member queries and n-grams. No taxonomy, no legacy labels, no other "
        "agent's answer — enforced by `Send` payload isolation plus a firewall that scans "
        "every card before it becomes a prompt."
    )
    code(
        "nm = load('tree_naming')\n"
        "print('mean coherence:', nm.get('mean_coherence'))\n"
        "print('risk found without being told:', nm['independent_risk_discovery']['found_without_being_told'])\n"
        "pd.DataFrame([{'leaf': n['leaf_id'], 'name': n['name_zh'], 'coherence': n['coherence'],\n"
        "               'risk': n['risk_flag'], 'user_need': n['user_need'][:70]}\n"
        "              for n in nm['namings']]).sort_values('leaf').head(30)"
    )

    md(
        "## Phase 8 — governance actually executed\n\n"
        "Every prescription is `executed` with an evidence pointer, or `declined` with a "
        "stated reason. The run fails if any is left merely `proposed`."
    )
    code(
        "gov = load('governance')\n"
        "print(gov['mechanism'])\n"
        "print()\n"
        "print('metric deltas:', gov['execution'].get('metric_deltas'))\n"
        "pd.DataFrame(gov['ledger'])"
    )

    md(
        "## Phase 9 — the uniform panel\n\n"
        "Every candidate re-measured by one code path, on one sub-sample, under one seed. "
        "Note the `authority` column: silhouette is present and barred from deciding anything."
    )
    code(
        "panel = load('metrics_panel')['table']\n"
        "print('panel', panel['panel_id'], panel['panel_config'])\n"
        "auth = {m['name']: m['authority'] for m in panel['metrics']}\n"
        "print(json.dumps(auth, indent=1))\n"
        "pd.DataFrame(panel['rows'])"
    )
    code("for i, f in enumerate(panel['footnotes'], 1): print(f'{i}. {f}\\n')")

    md("## Phase 10 — deployment, and a live routing demo")
    code(
        "dep = load('deployment')\n"
        "print('inference :', dep['inference'])\n"
        "print('model size: %.0f KB' % (dep['model_bytes']/1024))\n"
        "print('routing   :', {k: v for k, v in dep['routing'].items() if k != 'policy'})\n"
        "pd.DataFrame(dep['live_demo'])"
    )
    md(
        "### Deterministic exemplars\n\n"
        "One example per phrasing family, taken at the **median index of the hit set**. The "
        "persuasive force comes from the fact that nobody could have chosen them."
    )
    code("pd.DataFrame(dep['deterministic_exemplars'])")

    md("## The delivered table — both label systems, side by side")
    code(
        "labels = load('labels_full', 'csv')\n"
        "print(labels.shape)\n"
        "print(list(labels.columns))\n"
        "labels.head(12)"
    )
    code(
        "print('ambiguous rows: %.1f%%' % (labels['bu_ambiguous'].mean()*100))\n"
        "labels.groupby('bu_family_final').size().sort_values(ascending=False).head(20)"
    )

    md(
        "## What these numbers do not mean\n\n"
        "- Distillation accuracy measures **learnability** of the clusters, not agreement with "
        "human judgment.\n"
        "- Silhouette is advisory everywhere it appears.\n"
        "- Template fragmentation is negatively correlated with cluster count — read it beside "
        "`n_clusters`.\n"
        "- Clustering is structurally blind to pragmatic intents; those belong to the top-down "
        "route and are delivered alongside, never merged in."
    )

    nb = nbformat.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    path = deps.store.gen_dir / "Walkthrough.ipynb"

    executed = False
    error = ""
    try:
        from .zh_notebook import _client

        _client(nb, deps.store.gen_dir).execute()
        executed = True
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"

    nbformat.write(nb, path)
    n_err = sum(
        1 for c in nb.cells
        for o in (c.get("outputs") or []) if o.get("output_type") == "error"
    )
    deps.emit(
        f"  notebook: {'executed' if executed else 'NOT executed'}, {n_err} cell errors"
        + (f" — {error[:140]}" if error else "")
    )
    return deps.store.register_file(
        "notebook", path, "notebook", producer="p11",
        summary=f"walkthrough, {'executed' if executed else 'unexecuted'}, {n_err} errors",
    )
