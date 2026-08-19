"""Command line: ``qmine run``, ``qmine resume``, ``qmine inspect``, ``qmine doctor``."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .config import DomainProfile, QMineConfig

app = typer.Typer(add_completion=False, help="QMine — a twelve-phase query-intent mining agent team.")
console = Console()

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def _load_env() -> None:
    """Pick up a .env before any provider detection runs."""
    from .llm.env import load_dotenv

    loaded = load_dotenv()
    if loaded:
        console.print(f"[dim]loaded {len(loaded)} key(s) from {next(iter(loaded.values()))}[/dim]")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _load_config(config: Optional[str], domain: Optional[str], **over) -> QMineConfig:
    if config:
        cfg = QMineConfig.load(config)
    else:
        cfg = QMineConfig()
        if domain:
            path = Path(domain)
            if not path.exists():
                path = CONFIG_DIR / "domains" / f"{domain}.yaml"
            cfg.domain = DomainProfile.load(path)
    for k, v in over.items():
        if v is None:
            continue
        if "." in k:
            section, field = k.split(".", 1)
            setattr(getattr(cfg, section), field, v)
        else:
            setattr(cfg, k, v)
    # Re-validate: `fast_mode` is applied by a model validator, and assigning it
    # after construction would leave the full-size grids in place while the flag
    # claimed otherwise — a silent, expensive discrepancy.
    return QMineConfig.model_validate(cfg.model_dump())


@app.command()
def run(
    input: str = typer.Option(..., "--input", "-i", help="CSV/Parquet/XLSX of queries."),
    domain: str = typer.Option("k12_zh", "--domain", "-d", help="Domain profile name or path."),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Full config YAML."),
    text_column: str = typer.Option("query", "--text-column"),
    reference_columns: str = typer.Option("", "--reference-columns", help="Comma-separated legacy label columns."),
    sample: Optional[int] = typer.Option(None, "--sample", help="Subsample N rows."),
    run_root: str = typer.Option("runs", "--run-root"),
    run_id: Optional[str] = typer.Option(None, "--run-id"),
    fast: bool = typer.Option(False, "--fast", help="Shrink grids for a wiring smoke test."),
    offline: bool = typer.Option(False, "--offline", help="No network: hashing encoder + heuristic agents."),
    provider: str = typer.Option("auto", "--provider", help="auto | anthropic | mock."),
    human_review: bool = typer.Option(False, "--human-review", help="Pause at reviewer sign-off points."),
    resume: bool = typer.Option(False, "--resume", help="Continue an existing run-id instead of restarting it."),
    dashboard: bool = typer.Option(True, "--dashboard/--plain",
                                   help="Live phase/agent/metric panel (needs a TTY)."),
    verbose: bool = typer.Option(True, "--verbose/--quiet"),
) -> None:
    """Run the full twelve-phase pipeline."""
    _load_env()
    _setup_logging(verbose)
    from .runner import resume_run, run_pipeline

    # A long run on a laptop gets interrupted — the lid closes, the network
    # drops, the process dies. Rather than make the user remember a second
    # command for that case, `--resume` continues from the last checkpoint and
    # falls through to a fresh run when there is nothing to continue.
    if resume and run_id:
        ckpt = Path(run_root) / run_id / "checkpoints.sqlite"
        resolved = Path(run_root) / run_id / f"gen{1:02d}" / "config.resolved.yaml"
        if ckpt.exists() and resolved.exists():
            cfg = QMineConfig.load(resolved)
            cfg.run_root = run_root
            console.print(f"[dim]resuming {run_id} from {ckpt} (config {cfg.config_hash})[/dim]")
            out = resume_run(cfg, run_id)
            _print_summary(out["summary"])
            return
        console.print(f"[yellow]nothing to resume for {run_id}; starting a fresh run[/yellow]")

    cfg = _load_config(config, domain, run_root=run_root, fast_mode=fast, offline=offline)
    cfg.data.input_path = input
    cfg.data.text_column = text_column
    cfg.data.reference_label_columns = [c.strip() for c in reference_columns.split(",") if c.strip()]
    if sample:
        cfg.data.sample_size = sample
    cfg.llm.provider = provider  # type: ignore[assignment]
    if offline:
        cfg.llm.provider = "mock"  # type: ignore[assignment]

    console.rule(f"[bold]QMine[/bold] · domain [cyan]{cfg.domain.key}[/cyan] · config [dim]{cfg.config_hash}[/dim]")

    # The dashboard needs a TTY and suppresses log output while it owns the
    # screen; --plain (or --quiet, or a pipe) falls back to plain lines so CI and
    # `tee` keep working.
    use_dash = dashboard and verbose
    if use_dash:
        import logging as _logging

        from .ui.live import LiveDashboard

        _logging.getLogger("qmine").setLevel(_logging.WARNING)
        dash = LiveDashboard(run_id=run_id or "", domain=cfg.domain.key,
                             provider=cfg.llm.provider, language=cfg.report_language)
        with dash:
            if not dash.enabled:
                _logging.getLogger("qmine").setLevel(_logging.INFO)
            result = run_pipeline(cfg, run_id=run_id, human_review=human_review,
                                  on_event=dash.handle)
    else:
        result = run_pipeline(cfg, run_id=run_id, human_review=human_review,
                              on_event=lambda m: console.print(f"  [dim]{m}[/dim]"))
    _print_summary(result["summary"])


@app.command()
def resume(
    run_id: str = typer.Argument(..., help="Run to resume."),
    domain: str = typer.Option("k12_zh", "--domain", "-d"),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    run_root: str = typer.Option("runs", "--run-root"),
    decision: Optional[str] = typer.Option(None, "--decision", help="approve | reject"),
    reason: str = typer.Option("", "--reason"),
    generation: int = typer.Option(1, "--generation"),
) -> None:
    """Resume a checkpointed run, optionally answering a pending review."""
    _load_env()
    _setup_logging(True)
    from .runner import resume_run

    # Restore the run's OWN config. Rebuilding a default here was a real bug: the
    # resumed run lost `reference_label_columns`, so the blindness firewall armed
    # with zero forbidden terms and the anti-anchoring guarantee quietly lapsed.
    resolved = Path(run_root) / run_id / f"gen{generation:02d}" / "config.resolved.yaml"
    if resolved.exists() and not config:
        cfg = QMineConfig.load(resolved)
        cfg.run_root = run_root
        console.print(f"[dim]restored config from {resolved} (hash {cfg.config_hash})[/dim]")
    else:
        cfg = _load_config(config, domain, run_root=run_root)
    value = {"decision": decision, "reason": reason} if decision else None
    out = resume_run(cfg, run_id, generation=generation, resume_value=value)
    _print_summary(out["summary"])


@app.command()
def inspect(
    run_id: str = typer.Argument(...),
    run_root: str = typer.Option("runs", "--run-root"),
    generation: int = typer.Option(1, "--generation"),
    what: str = typer.Option("summary", "--what",
                             help="summary | artifacts | gates | decisions | panel | governance | leaves"),
) -> None:
    """Inspect a finished run without re-running anything."""
    gen = Path(run_root) / run_id / f"gen{generation:02d}"
    if not gen.exists():
        console.print(f"[red]no such generation: {gen}[/red]")
        raise typer.Exit(1)

    def _load(name: str):
        p = gen / f"{name}.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    if what == "artifacts":
        t = Table("artifact", "size", title=f"{run_id} gen{generation}")
        for f in sorted(gen.iterdir()):
            t.add_row(f.name, f"{f.stat().st_size / 1024:.0f} KB")
        console.print(t)
    elif what == "summary":
        console.print_json(data=_load("run_summary") or {})
    elif what == "gates":
        s = _load("run_summary") or {}
        t = Table("gate", "status", "blocking", "message")
        for k, g in (s.get("gates") or {}).items():
            t.add_row(k, g["status"], str(g["blocking"]), g["message"][:80])
        console.print(t)
    elif what == "panel":
        p = _load("metrics_panel") or {}
        table = p.get("table", {})
        if table:
            cols = [m["name"] for m in table["metrics"]]
            t = Table("candidate", *cols)
            for r in table["rows"]:
                t.add_row(r["subject"], *[f"{r.get(c):.4g}" if isinstance(r.get(c), float) else str(r.get(c)) for c in cols])
            console.print(t)
            for f in table.get("footnotes", []):
                console.print(f"[dim]• {f}[/dim]")
    elif what == "governance":
        console.print_json(data=_load("governance") or {})
    elif what == "leaves":
        nm = _load("tree_naming") or {}
        t = Table("leaf", "name", "coherence", "risk", "user_need")
        for n in nm.get("namings", []):
            t.add_row(str(n["leaf_id"]), n.get("name_zh", ""), str(n.get("coherence")),
                      "yes" if n.get("risk_flag") else "", (n.get("user_need") or "")[:60])
        console.print(t)
    elif what == "decisions":
        console.print_json(data=(_load("run_summary") or {}).get("n_decisions"))
    else:
        console.print(f"[red]unknown --what {what}[/red]")


@app.command()
def promote(
    old: str = typer.Option(..., "--old", help="CSV with the current labels."),
    new: str = typer.Option(..., "--new", help="CSV with the challenger labels."),
    label_column: str = typer.Option("bu_leaf_name", "--label-column"),
    text_column: str = typer.Option("query", "--text-column"),
    out: str = typer.Option("promoted_labels.csv", "--out"),
    sample: int = typer.Option(300, "--sample", help="Disagreements to judge."),
    domain: str = typer.Option("k12_zh", "--domain", "-d"),
    alpha: float = typer.Option(0.05, "--alpha", help="Significance level for promotion."),
) -> None:
    """Referee upgrade protocol: let a challenger label set earn promotion, or not.

    Judges only the rows where the two systems DISAGREE, blind and with
    randomised presentation order, and promotes only on a statistically
    significant win. The old labels are preserved in a `_v1` column either way.
    """
    _setup_logging(True)
    import pandas as pd

    from .agents.base import AgentContext
    from .agents.roles import RefereeAgent
    from .artifacts import ArtifactStore
    from .llm.registry import ModelRegistry
    from .memory.context import BlindnessFirewall
    from .memory.store import open_memory
    from .ops.promotion import (
        apply_promotion, build_blind_matchups, find_disagreements, score_verdicts,
    )

    cfg = _load_config(None, domain)
    df_old, df_new = pd.read_csv(old), pd.read_csv(new)
    queries = df_old[text_column].astype(str).tolist()
    old_labels = df_old[label_column].astype(str).tolist()
    new_labels = df_new[label_column].astype(str).tolist()

    d = find_disagreements(old_labels, new_labels, limit=sample)
    console.print(f"{d['n_disagreements']} disagreements ({d['disagreement_rate'] * 100:.1f}%); "
                  f"judging {len(d['sampled_indices'])}")
    if not d["sampled_indices"]:
        console.print("[green]the two label sets agree everywhere — nothing to judge[/green]")
        raise typer.Exit(0)

    matchups, key = build_blind_matchups(queries, old_labels, new_labels, d["sampled_indices"])
    with open_memory(None, domain=cfg.domain.key) as mem:
        ctx = AgentContext(cfg=cfg, registry=ModelRegistry(cfg.llm),
                           store=ArtifactStore(Path(cfg.run_root) / "promote"),
                           memory=mem, firewall=BlindnessFirewall())
        agent = RefereeAgent(ctx)
        verdicts = []
        for i in range(0, len(matchups), 25):
            chunk = matchups[i : i + 25]
            rows = [{"query": m["query"], "label_a": m["label_a"], "label_b": m["label_b"],
                     "rationale_a": "", "rationale_b": ""} for m in chunk]
            batch = agent.run(disagreements=rows, classes="", rules="")
            for m, v in zip(chunk, batch.verdicts):
                winner = "a" if str(v.final_label) == m["label_a"] else (
                    "b" if str(v.final_label) == m["label_b"] else "tie")
                verdicts.append({"row": m["row"], "winner": "tie" if v.both_defensible else winner,
                                 "rationale": v.rationale})

    scoring = score_verdicts(verdicts, key, alpha=alpha)
    console.print(f"[bold]{scoring['verdict']}[/bold]")
    result = apply_promotion(df_old, label_column, new_labels, scoring)
    result.to_csv(out, index=False, encoding="utf-8-sig")
    console.print(f"wrote {out} — old labels preserved in `{label_column}_v1`, "
                  f"overturned rows marked label_source='referee'")


@app.command("export-cards")
def export_cards(
    run_id: str = typer.Argument(..., help="Run whose clusters should be named."),
    run_root: str = typer.Option("runs", "--run-root"),
    generation: int = typer.Option(1, "--generation"),
    shards: int = typer.Option(5, "--shards", help="Independent reviewers to split across."),
    out: Optional[str] = typer.Option(None, "--out", help="Directory for the briefs."),
) -> None:
    """Export blind-naming briefs so an external panel can run Phase 7.

    Use this when the naming step deserves a stronger model, or real humans,
    than the run itself used. The briefs are rendered through the same blindness
    firewall the built-in namers work under.
    """
    _setup_logging(False)
    import json as _json

    from .memory.context import BlindnessFirewall
    from .ops.handoff import export_shards
    from .records import NamingCard

    gen = Path(run_root) / run_id / f"gen{generation:02d}"
    cards_file = gen / "naming_cards.json"
    if not cards_file.exists():
        console.print(f"[red]{cards_file} not found — has this run reached Phase 7?[/red]")
        raise typer.Exit(1)
    payload = _json.loads(cards_file.read_text(encoding="utf-8"))
    cards = [NamingCard.model_validate(c) for c in payload["cards"]]

    fw = BlindnessFirewall()
    corpus = gen / "corpus.parquet"
    if corpus.exists():
        import pandas as pd

        df = pd.read_parquet(corpus)
        for col in df.columns:
            if col.startswith("legacy_") or col.startswith("ref_"):
                fw.add_reference_labels(df[col].astype(str).unique().tolist())
    tax = gen / "taxonomy.json"
    if tax.exists():
        fw.add_taxonomy(_json.loads(tax.read_text(encoding="utf-8")).get("taxonomy"))

    target = Path(out) if out else gen / "naming_handoff"
    m = export_shards(cards, target, n_shards=shards, firewall=fw)
    t = Table("shard", "clusters", "leaf ids", "file")
    for sh in m["shards"]:
        t.add_row(str(sh["shard"]), str(sh["n_clusters"]),
                  str(sh["leaf_ids"])[:44], Path(sh["path"]).name)
    console.print(t)
    console.print(f"[dim]{m['firewall']['n_forbidden_terms']} label terms were checked against "
                  f"and excluded from every brief[/dim]")
    console.print(f"[green]briefs written to {target}[/green]")


@app.command("import-namings")
def import_namings_cmd(
    run_id: str = typer.Argument(...),
    payload: str = typer.Argument(..., help="JSON file of verdicts from the panel."),
    run_root: str = typer.Option("runs", "--run-root"),
    generation: int = typer.Option(1, "--generation"),
    named_by: str = typer.Option("external-panel", "--named-by"),
) -> None:
    """Read an external panel's verdicts back into a run."""
    _setup_logging(False)
    import json as _json

    from .ops.handoff import coverage_report, import_namings

    gen = Path(run_root) / run_id / f"gen{generation:02d}"
    cards = _json.loads((gen / "naming_cards.json").read_text(encoding="utf-8"))["cards"]
    expected = [c["leaf_id"] for c in cards]
    namings = import_namings(payload, named_by=named_by)
    cov = coverage_report(namings, expected)

    out = gen / "tree_naming_external.json"
    out.write_text(_json.dumps({
        "namings": [n.model_dump() for n in namings],
        "coverage": cov,
        "source": named_by,
        "mean_coherence": round(sum(n.coherence for n in namings) / max(len(namings), 1), 3),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    console.print(f"imported [bold]{cov['n_received']}[/bold]/{cov['n_expected']} clusters "
                  f"from {named_by}")
    if cov["missing"]:
        console.print(f"[yellow]missing verdicts for clusters {cov['missing']}[/yellow]")
    console.print(f"[green]written to {out}[/green]")


@app.command("diff")
def diff_cmd(
    previous: str = typer.Argument(..., help="Earlier run id."),
    current: str = typer.Argument(..., help="Later run id."),
    run_root: str = typer.Option("runs", "--run-root"),
    generation: int = typer.Option(1, "--generation"),
) -> None:
    """Compare two runs: what drifted, and what merely changed method."""
    _setup_logging(False)
    import json as _json

    from .ops.handoff import diff_runs

    def _load(rid: str):
        p = Path(run_root) / rid / f"gen{generation:02d}" / "maintenance.json"
        if not p.exists():
            console.print(f"[red]{p} not found — has {rid} reached Phase 12?[/red]")
            raise typer.Exit(1)
        return _json.loads(p.read_text(encoding="utf-8"))

    d = diff_runs(_load(previous), _load(current))
    colour = "green" if d["config_comparable"] else "red"
    console.print(f"[{colour}]{d['verdict']}[/{colour}]\n")
    t = Table("", "previous", "current")
    for k in ("families", "leaves", "alpha", "family_k"):
        t.add_row(k, str(d["shape"]["previous"].get(k)), str(d["shape"]["current"].get(k)))
    console.print(t)
    if d["appeared"]:
        console.print(f"\n[bold]appeared[/bold]: {d['appeared']}")
    if d["vanished"]:
        console.print(f"[bold]vanished[/bold]: {d['vanished']}")
    if d["grown"] or d["shrunk"]:
        g = Table("family", "prev share", "cur share", "delta")
        for r in d["grown"] + d["shrunk"]:
            g.add_row(r["name"], f"{r['prev_share']:.3f}", f"{r['cur_share']:.3f}", f"{r['delta']:+.3f}")
        console.print(g)


@app.command("models")
def models_cmd(
    refresh: bool = typer.Option(False, "--refresh", help="Ignore the cache and refetch."),
    offline: bool = typer.Option(False, "--offline", help="Never touch the network."),
    budget: Optional[float] = typer.Option(None, "--budget", help="Flag if the plan exceeds this."),
    chinese: bool = typer.Option(False, "--prefer-chinese-native",
                                 help="Nudge multilingual roles toward Chinese-native labs."),
    cache_dir: str = typer.Option(".cache", "--cache-dir"),
) -> None:
    """Show which providers are reachable and which model each agent role would use."""
    _load_env()
    _setup_logging(False)
    from .llm.catalog import fetch
    from .llm.providers import detect
    from .llm.router import route

    av = detect()
    t = Table("provider", "kind", "via")
    from .llm.providers import BY_KEY

    for k in av.configured:
        spec = BY_KEY[k]
        t.add_row(spec.display, spec.kind, av.env_seen.get(k, ""))
    if not av.configured:
        t.add_row("[yellow]none configured[/yellow]", "", "set e.g. ANTHROPIC_API_KEY")
    console.print(t)

    cat = fetch(cache_dir=cache_dir, ttl=0 if refresh else 6 * 3600, allow_network=not offline)
    console.print(f"\n[dim]catalogue: {len(cat.models)} models from {cat.sources or 'nothing'}, "
                  f"{cat.age_hours:.1f}h old[/dim]")
    if cat.degraded:
        console.print(f"[yellow]{cat.degraded}[/yellow]")
    if not av.configured:
        console.print("\n[yellow]No API keys — nothing to route. Showing catalogue only.[/yellow]")
        return

    plan = route(cat, av.usable, budget_usd=budget, prefer_chinese_native=chinese)
    r = Table("role", "model", "tier", "calls", "est. $", "fallback")
    for role, a in sorted(plan.assignments.items(), key=lambda kv: -kv[1].estimated_cost_usd):
        r.add_row(role, f"{a.provider}:{a.model}" if a.model else "[red]none[/red]",
                  a.tier, str(a.estimated_calls), f"{a.estimated_cost_usd:.3f}",
                  (a.fallbacks[:1] or [""])[0][:34])
    console.print(r)
    console.print(f"\n[bold]estimated total: ${plan.total_cost_usd:.2f}[/bold] per full run")
    for n in plan.notes:
        console.print(f"[yellow]{n}[/yellow]")
    for role, a in plan.assignments.items():
        for w in a.warnings:
            console.print(f"[red]{role}: {w}[/red]")


@app.command()
def doctor() -> None:
    """Check the environment: packages, credentials, models, fonts."""
    _load_env()
    _setup_logging(False)
    t = Table("check", "status", "detail")

    for mod in ("numpy", "pandas", "sklearn", "scipy", "langgraph", "langchain_core",
                "langchain_anthropic", "sentence_transformers", "torch", "umap",
                "matplotlib", "nbformat", "nbclient", "jieba"):
        try:
            m = __import__(mod)
            t.add_row(mod, "[green]ok[/green]", getattr(m, "__version__", ""))
        except Exception as exc:  # noqa: BLE001
            t.add_row(mod, "[yellow]missing[/yellow]", str(exc)[:60])

    import os

    has_key = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
    t.add_row("ANTHROPIC_API_KEY", "[green]set[/green]" if has_key else "[yellow]absent[/yellow]",
              "real agents" if has_key else "will fall back to the deterministic offline stand-in")

    try:
        from matplotlib import font_manager

        fonts = {f.name for f in font_manager.fontManager.ttflist}
        cjk = [f for f in ("Arial Unicode MS", "Heiti TC", "PingFang SC", "Songti SC") if f in fonts]
        t.add_row("CJK fonts", "[green]ok[/green]" if cjk else "[yellow]none[/yellow]", ", ".join(cjk) or "charts will show boxes")
    except Exception as exc:  # noqa: BLE001
        t.add_row("CJK fonts", "[yellow]?[/yellow]", str(exc)[:60])

    t.add_row("domain profiles", "[green]ok[/green]",
              ", ".join(sorted(p.stem for p in (CONFIG_DIR / "domains").glob("*.yaml"))))
    console.print(t)


@app.command()
def demo(
    run_root: str = typer.Option("runs", "--run-root"),
    fast: bool = typer.Option(True, "--fast/--full"),
    sample: int = typer.Option(8000, "--sample"),
) -> None:
    """Run the bundled K12 dataset end to end — the fastest way to see the whole thing."""
    data = Path(__file__).resolve().parents[2] / "data" / "raw" / "k12_queries_50k.csv"
    if not data.exists():
        console.print(f"[red]bundled dataset not found at {data}[/red]")
        raise typer.Exit(1)
    run(input=str(data), domain="k12_zh", config=None, text_column="query",
        reference_columns="legacy_l1,legacy_l2", sample=sample, run_root=run_root,
        run_id=None, fast=fast, offline=False, provider="auto", human_review=False, verbose=True)


def _print_summary(s: dict) -> None:
    console.rule("[bold]run summary")
    t = Table("field", "value")
    for k in ("run_id", "generation", "elapsed_s", "halted", "halt_reason", "n_decisions", "n_prescriptions"):
        if s.get(k) not in (None, ""):
            t.add_row(k, str(s[k]))
    t.add_row("phases", ", ".join(s.get("completed_phases", [])))
    t.add_row("artifacts", str(len(s.get("artifacts", []))))
    u = s.get("llm_usage", {})
    t.add_row("llm", f"{u.get('provider')} · {u.get('calls')} calls · {u.get('cache_hits')} cached "
                     f"· ${u.get('estimated_cost_usd', 0)}")
    t.add_row("output", s.get("artifact_root", ""))
    console.print(t)
    gates = s.get("gates", {})
    if gates:
        g = Table("gate", "status", "message")
        for k, v in gates.items():
            colour = {"passed": "green", "warned": "yellow", "failed": "red", "rejected": "red"}.get(v["status"], "white")
            g.add_row(k, f"[{colour}]{v['status']}[/{colour}]", v["message"][:70])
        console.print(g)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
