"""Command line: ``qmine run``, ``qmine resume``, ``qmine inspect``, ``qmine doctor``."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import typer
import yaml
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


def _console_level(level: int) -> None:
    """Set the verbosity of the stderr handler, leaving loggers permissive.

    Handlers decide what reaches a destination; loggers decide what exists at
    all. Only the former should move when the UI wants a quiet screen.
    """
    for h in logging.getLogger().handlers:
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
            h.setLevel(level)


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )



def _load_domain(domain: str) -> "DomainProfile":
    """Resolve a domain name or path, and say what exists when it does not.

    An unknown name used to raise a bare `FileNotFoundError` naming a path inside
    the package, which tells a user nothing about what they could have typed.
    """
    path = Path(domain)
    if path.exists():
        return DomainProfile.load(path)
    path = CONFIG_DIR / "domains" / f"{domain}.yaml"
    if path.exists():
        return DomainProfile.load(path)
    known = sorted(p.stem for p in (CONFIG_DIR / "domains").glob("*.yaml"))
    raise SystemExit(
        f"Unknown domain {domain!r}.\n"
        f"  Built-in profiles: {', '.join(known)}\n"
        f"  Or pass a path to your own profile YAML: --domain ./my_domain.yaml\n"
        f"  Or use --domain generic, which assumes no vertical: it mines phrasing\n"
        f"  families from the corpus and carries only universal risk categories."
    )


def _load_config(config: Optional[str], domain: Optional[str], **over) -> QMineConfig:
    """Build the run config from a YAML file and/or a domain profile.

    `--config` used to win outright and silently drop `--domain`, so a config file
    holding nothing but a provider policy also replaced the domain profile with
    the generic default. The visible symptom was subtle: template coverage halved
    from 36.6% to 19.1% in a *deterministic* phase, because the domain's phrasing
    seeds had quietly gone missing. Everything downstream of template mining —
    fragmentation, intent alignment, the K locator — rests on those rows.

    A config file that declares its own `domain:` still wins; one that does not
    inherits the profile the user asked for on the command line.
    """
    if config:
        cfg = QMineConfig.load(config)
        declares_domain = "domain" in (yaml.safe_load(Path(config).read_text()) or {})
        if domain and not declares_domain:
            cfg.domain = _load_domain(domain)
    else:
        cfg = QMineConfig()
        # NO --domain still means "unknown vertical", not "Chinese". Load the
        # generic profile explicitly rather than relying on class defaults: it
        # carries the seven universal risk categories and the pragmatic-intent
        # hints, which a bare `DomainProfile()` does not.
        cfg.domain = _load_domain(domain or "generic")
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
    input: Optional[str] = typer.Option(
        None, "--input", "-i",
        help="CSV/Parquet/XLSX of queries. Not needed with --resume: a resumed "
             "run restores its own input path from its saved config."),
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
    reuse_taxonomy: Optional[str] = typer.Option(
        None, "--reuse-taxonomy",
        help="Reuse a finished taxonomy (RUN_ID, RUN_ID/genNN, or a taxonomy.json path) "
             "instead of re-deriving one. Skips p2a and lets the gold set replay from cache."),
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
        from .artifacts import latest_generation, resolved_config_path

        ckpt = Path(run_root) / run_id / "checkpoints.sqlite"
        # Resume the run's CURRENT generation, not generation 1. The thread id is
        # per-generation, so after `new-generation` a hardcoded 1 reopens the old
        # halted thread and exits having done nothing.
        gen = latest_generation(Path(run_root) / run_id)
        resolved = resolved_config_path(Path(run_root) / run_id, gen)
        if ckpt.exists() and resolved is not None:
            cfg = QMineConfig.load(resolved)
            cfg.run_root = run_root
            # The reuse flag has to be applied HERE. This branch returns before
            # the fresh-run setup below ever sees it, so `--reuse-taxonomy` on a
            # `--resume` was accepted, ignored, and the run silently re-derived a
            # taxonomy — cascading a cache miss across every gold row, which is
            # exactly the failure the flag exists to prevent.
            if reuse_taxonomy:
                cfg.taxonomy.reuse_taxonomy_from = reuse_taxonomy
                console.print(f"[dim]reusing taxonomy from {reuse_taxonomy}[/dim]")
            console.print(f"[dim]resuming {run_id} generation {gen} from {ckpt} "
                          f"(config {cfg.config_hash})[/dim]")
            _dash = _attach_dashboard(cfg, run_id, run_root,
                                      enabled=dashboard and verbose)
            with _dash:
                if _dash.enabled:
                    _console_level(logging.WARNING)
                out = resume_run(cfg, run_id, generation=gen,
                                 on_event=_dash.handle)
                _dash.finish(ok=not out["summary"].get("halted", False))
            _console_level(logging.INFO)
            _print_summary(out["summary"])
            return
        console.print(f"[yellow]nothing to resume for {run_id}; starting a fresh run[/yellow]")

    # Refuse to re-run an id that already exists. `run` is not the resume path:
    # it opens the SAME LangGraph thread and the SAME llm_cache, and the two
    # disagree. Observed four times in one day — the checkpoint replayed
    # `halted=True` and exited in 3.1s without re-reaching the gate; and the cache
    # matched an architect entry written by a DIFFERENT aborted attempt, so the
    # rule writer silently built on a 21-class taxonomy where the run it was
    # meant to continue had 24. Neither failed loudly; both wasted an hour.
    if run_id and not resume:
        existing = Path(run_root) / run_id
        if (existing / "checkpoints.sqlite").exists() or (existing / "llm_cache").exists():
            console.print(
                f"[red]run id {run_id!r} already exists[/red] at {existing}.\n"
                "`run` is not the resume path — it would reopen that run's graph "
                "thread and its LLM cache, and those two disagree. Pick one:\n"
                f"  • [bold]qmine run --resume --run-id {run_id}[/bold]  "
                "— continue it properly (retries a crash; refuses to overturn a gate)\n"
                f"  • [bold]qmine new-generation {run_id} --reason '...'[/bold]  "
                "— keep the evidence, start a fresh generation after fixing a gate\n"
                "  • [bold]--run-id <something-new>[/bold]  — a clean run with clean provenance\n"
                f"  • delete {existing} if you truly meant to discard it"
            )
            raise typer.Exit(2)

    # Only a FRESH run needs an input path; a resumed one returned above with the
    # path restored from its own config. Requiring it unconditionally made the
    # command the gate-halt message tells you to run — `qmine run --resume
    # --run-id X` — fail with "Missing option '--input'".
    if input is None:
        console.print("[red]--input is required to start a new run[/red] "
                      "(it is restored automatically when you --resume one).")
        raise typer.Exit(2)

    cfg = _load_config(config, domain, run_root=run_root, fast_mode=fast, offline=offline)
    cfg.data.input_path = input
    cfg.data.text_column = text_column
    cfg.data.reference_label_columns = [c.strip() for c in reference_columns.split(",") if c.strip()]
    if sample:
        cfg.data.sample_size = sample
    if reuse_taxonomy:
        cfg.taxonomy.reuse_taxonomy_from = reuse_taxonomy
    cfg.llm.provider = provider  # type: ignore[assignment]
    if offline:
        cfg.llm.provider = "mock"  # type: ignore[assignment]

    console.rule(f"[bold]QMine[/bold] · domain [cyan]{cfg.domain.key}[/cyan] · config [dim]{cfg.config_hash}[/dim]")

    # The dashboard needs a TTY and suppresses log output while it owns the
    # screen; --plain (or --quiet, or a pipe) falls back to plain lines so CI and
    # `tee` keep working.
    use_dash = dashboard and verbose

    # THE BROWSABLE PAGE IS NOT PART OF THE TERMINAL PANEL, AND MUST NOT BE
    # GATED ON IT.
    #
    # It was built inside `if use_dash:`, so a run launched detached — no TTY, so
    # `verbose` is off — wrote no page at all. That is exactly backwards: the
    # headless run is the one that most needs a browsable view, because there is
    # no terminal to watch. Worse, the only page then on disk was the one a
    # `qmine watch` follower wrote, which replays `run.log` from offset 0 and so
    # re-rendered superseded generations into the file the operator was reading.
    dash = _attach_dashboard(cfg, run_id or "", run_root, enabled=use_dash)

    with dash:
        # Quiet the *console handler*, not the logger. Lowering the logger here
        # used to take `<run>/run.log` down with it, so choosing the pretty view
        # meant choosing to have no record of the run at all.
        if dash.enabled:
            _console_level(logging.WARNING)
        result = run_pipeline(cfg, run_id=run_id, human_review=human_review,
                              on_event=dash.handle)
        # The dashboard cannot tell a halt from ordinary progress — a blocking
        # gate returns normally — so the outcome is handed to it.
        dash.finish(ok=not result["summary"].get("halted", False))
    _console_level(logging.INFO)
    _print_summary(result["summary"])


def _attach_dashboard(cfg: "QMineConfig", run_id: str, run_root: str, *, enabled: bool):
    """Build the dashboard and its browsable page for either run path.

    Factored because the RESUME branch returns before the fresh-run setup, which
    the branch's own comment warns about — so `--resume` had no terminal panel
    and no page at all, and the only `dashboard.html` on disk was whatever a
    `qmine watch` follower had written by replaying every generation.
    """
    from .ui.live import PHASES, LiveDashboard
    from .ui.web import HtmlWriter, artifacts_from_index

    dash = LiveDashboard(run_id=run_id or "", domain=cfg.domain.key,
                         provider=cfg.llm.provider, language=cfg.report_language,
                         enabled=enabled)
    if run_id:
        root = Path(run_root) / run_id
        root.mkdir(parents=True, exist_ok=True)
        dash.html_writer = HtmlWriter(root / "dashboard.html", PHASES)
        dash.artifacts_fn = lambda: artifacts_from_index(root / "index.jsonl")
        console.print(f"  [dim]dashboard → {root / 'dashboard.html'}[/dim]")
    return dash


@app.command()
def watch(
    run_id: str = typer.Argument(..., help="Run to follow."),
    run_root: str = typer.Option("runs", "--run-root"),
    language: str = typer.Option("zh", "--language", help="zh | en."),
    poll: float = typer.Option(0.5, "--poll", help="Seconds between reads."),
) -> None:
    """Attach the live dashboard to a run, following ``<run>/run.log``.

    The panel used to be welded to the process that owned the terminal, which
    forced a choice: run it yourself and watch, or hand it off and see nothing.
    Reading the log instead separates the two, so a run can be launched detached
    — or by someone else entirely — and still be watched, re-watched after it
    finishes, or watched from two terminals at once.
    """
    from .ui.live import PHASES, LiveDashboard, parse_log_clock, parse_log_line
    from .ui.web import HtmlWriter, artifacts_from_index

    root = Path(run_root) / run_id
    log_path, usage_path = root / "run.log", root / "usage.json"
    if not root.exists():
        console.print(f"[red]no run at {root}[/red]")
        raise typer.Exit(1)

    def usage() -> dict:
        try:
            return json.loads(usage_path.read_text())
        except Exception:  # noqa: BLE001
            return {}

    def finished() -> bool:
        """A summary exists AND nothing has been logged since it was written.

        Checking only for the file's existence made `watch` unusable on exactly
        the case it is most wanted for: a re-run of a run id that halted earlier
        leaves the previous attempt's summary on disk, so the follower replayed
        the log and exited within seconds while the new run was still going.
        """
        summaries = list(root.glob("gen*/run_summary.json"))
        if not summaries:
            return False
        newest = max(f.stat().st_mtime for f in summaries)
        if log_path.exists() and log_path.stat().st_mtime > newest + 1.0:
            return False          # the log has moved on past that summary
        return True

    def halted() -> bool:
        for f in sorted(root.glob("gen*/run_summary.json")):
            try:
                return bool(json.loads(f.read_text()).get("halted"))
            except Exception:  # noqa: BLE001
                pass
        return False

    dash = LiveDashboard(run_id=run_id, domain="", provider="", language=language)
    dash.usage_fn = usage
    dash.artifacts_fn = lambda: artifacts_from_index(root / "index.jsonl")
    # A follower rebuilds the browsable page too, so a run launched detached — or
    # by someone else — still gets one.
    # A SEPARATE FILE FROM THE RUN'S OWN PAGE.
    #
    # Both were written to `dashboard.html`, so a follower attached to a LIVE run
    # fought the run for the same path — and the follower always won on content,
    # because it replays `run.log` from offset 0. `run.log` is append-only across
    # generations, so watching a run at gen03 re-rendered gen01 and gen02 into the
    # page the run was trying to keep current. The operator saw stale events from
    # a generation that had been superseded twice.
    dash.html_writer = HtmlWriter(root / "dashboard.watch.html", PHASES)
    offset, idle = 0, 0.0
    with dash:
        try:
            while True:
                if log_path.exists():
                    size = log_path.stat().st_size
                    # A re-run of the same id truncates the log; without this the
                    # follower would read from a stale offset into the middle of
                    # a line and silently show nothing for the rest of the run.
                    if size < offset:
                        offset = 0
                    if size > offset:
                        with log_path.open(encoding="utf-8", errors="replace") as fh:
                            fh.seek(offset)
                            for line in fh:
                                msg = parse_log_line(line)
                                if msg:
                                    # The LOG's clock, not the follower's. A
                                    # finished run replays a thousand lines in
                                    # under a second, so wall-clock timing showed
                                    # every phase as "0s" — the follower's whole
                                    # point is re-watching, and it timed nothing.
                                    dash.handle(msg, at=parse_log_clock(line))
                            offset = fh.tell()
                        idle = 0.0
                    else:
                        idle += poll
                else:
                    idle += poll
                # Only stop once the run has written a summary AND gone quiet —
                # a summary alone can appear before the last lines are flushed.
                if finished() and idle >= 3.0:
                    break
                time.sleep(poll)
        except KeyboardInterrupt:
            pass
        dash.finish(ok=not halted())
    console.print(f"[dim]{log_path} — {'halted' if halted() else 'finished' if finished() else 'detached'}[/dim]")


@app.command()
def resume(
    run_id: str = typer.Argument(..., help="Run to resume."),
    domain: str = typer.Option("k12_zh", "--domain", "-d"),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    run_root: str = typer.Option("runs", "--run-root"),
    decision: Optional[str] = typer.Option(None, "--decision", help="approve | reject"),
    reason: str = typer.Option("", "--reason"),
    generation: Optional[int] = typer.Option(
        None, "--generation", help="Defaults to the run's newest generation."),
) -> None:
    """Resume a checkpointed run, optionally answering a pending review."""
    _load_env()
    _setup_logging(True)
    from .artifacts import latest_generation, resolved_config_path
    from .runner import resume_run

    if generation is None:
        generation = latest_generation(Path(run_root) / run_id)

    # Restore the run's OWN config. Rebuilding a default here was a real bug: the
    # resumed run lost `reference_label_columns`, so the blindness firewall armed
    # with zero forbidden terms and the anti-anchoring guarantee quietly lapsed.
    resolved = resolved_config_path(Path(run_root) / run_id, generation)
    if resolved is not None and not config:
        cfg = QMineConfig.load(resolved)
        cfg.run_root = run_root
        console.print(f"[dim]restored config from {resolved} (hash {cfg.config_hash})[/dim]")
    else:
        cfg = _load_config(config, domain, run_root=run_root)
    value = {"decision": decision, "reason": reason} if decision else None
    out = resume_run(cfg, run_id, generation=generation, resume_value=value)
    _print_summary(out["summary"])


@app.command("new-generation")
def new_generation_cmd(
    run_id: str = typer.Argument(..., help="Run to open a new generation of."),
    reason: str = typer.Option(..., "--reason", help="Why the previous generation was set aside."),
    run_root: str = typer.Option("runs", "--run-root"),
    from_generation: Optional[int] = typer.Option(
        None, "--from-generation",
        help="Branch from this generation instead of the newest one."),
    config: Optional[str] = typer.Option(
        None, "--config", "-c",
        help="Config the NEW generation will run under. Without it the new "
             "generation inherits the old one's, including any limit since fixed."),
    domain: Optional[str] = typer.Option(None, "--domain", "-d"),
) -> None:
    """Open the next generation of a run, keeping the old one intact.

    This is the move `resume` tells you to make after a blocking gate — resume
    deliberately will not overturn a gate's refusal — and until now there was no
    way to make it. A rejected generation is evidence, not waste: the source
    project's discarded 107-leaf tree became its phrasing-pattern library.
    """
    from .artifacts import latest_generation
    from .runner import new_generation

    cfg = _load_config(config, domain, run_root=run_root)
    # DEFAULT TO THE NEWEST GENERATION, NOT GENERATION 1.
    #
    # This defaulted to 1, so "open the next generation" on a run already at gen02
    # re-created **gen02** and overwrote it, rather than advancing to gen03. On
    # live41 that silently put a resumed run back into the generation whose pilot
    # had just failed, and the operator had no way to tell from the command they
    # typed. `artifacts.latest_generation` exists precisely because both RESUME
    # paths had this same hardcoded 1 — its docstring says so — and this call site
    # was missed.
    if from_generation is None:
        from_generation = latest_generation(Path(run_root) / run_id)
    nxt = new_generation(cfg, run_id, from_generation=from_generation, reason=reason)

    # SNAPSHOT THE CONFIG THE NEW GENERATION WILL RUN UNDER. Without this,
    # `resolved_config_path` falls back to the OLD generation's config — so a
    # generation opened precisely because a limit was wrong would inherit that
    # same wrong limit and fail the same way. live38 was opened after
    # `max_total_output_tokens` was found to be half what an honest run needs,
    # and its gen01 config still pinned the old 6,000,000.
    if config or domain:
        gen_dir = Path(run_root) / run_id / f"gen{nxt:02d}"
        gen_dir.mkdir(parents=True, exist_ok=True)
        # INHERIT THE RUN'S DATA SETTINGS. `input_path`, `text_column` and
        # `reference_label_columns` come from CLI flags at launch, not from the
        # config file, so a snapshot built from the file alone drops them and the
        # resumed run dies in p1 with "input_path is not set". A new generation
        # of the SAME run is by definition the same corpus.
        from .artifacts import resolved_config_path as _rcp

        prev = _rcp(Path(run_root) / run_id, from_generation)
        if prev is not None:
            old_cfg = QMineConfig.load(prev)
            cfg.data.input_path = old_cfg.data.input_path
            cfg.data.text_column = old_cfg.data.text_column
            cfg.data.reference_label_columns = list(old_cfg.data.reference_label_columns)
            cfg.data.sample_size = old_cfg.data.sample_size
        cfg.dump(gen_dir / "config.resolved.yaml")
        console.print(f"[dim]generation {nxt} config written "
                      f"({gen_dir / 'config.resolved.yaml'}, hash {cfg.config_hash})[/dim]")
    else:
        console.print("[yellow]no --config given: generation "
                      f"{nxt} inherits generation {from_generation}'s config, "
                      "including any limit you may have just fixed[/yellow]")

    console.print(f"[green]{run_id}[/green]: generation {from_generation} kept; "
                  f"now on generation [bold]{nxt}[/bold] — {reason}")
    console.print(f"Continue with: [bold]qmine run --resume --run-id {run_id}[/bold]")


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
        apply_promotion,
        build_blind_matchups,
        find_disagreements,
        score_verdicts,
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
        ctx = AgentContext(cfg=cfg, registry=ModelRegistry(cfg.llm, run_cfg=cfg),
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
    config: Optional[str] = typer.Option(
        None, "--config", "-c",
        help="Route against THIS config's provider policy — the one a run will use."),
    domain: Optional[str] = typer.Option(None, "--domain",
                                         help="Price against this domain profile's settings."),
    gold: Optional[int] = typer.Option(None, "--gold-sample-size",
                                       help="Price against this gold-set size."),
) -> None:
    """Show which providers are reachable and which model each agent role would use."""
    _load_env()
    _setup_logging(False)
    from .config import QMineConfig
    from .llm.catalog import fetch
    from .llm.providers import detect
    from .llm.router import route

    # Route against the config a RUN would use. Without this the command built a
    # bare `QMineConfig()`, so `excluded_labs` and `model_overrides` were invisible
    # and the pre-flight reported a different plan from the one that would execute
    # — it listed labs the live config excludes and ignored every pinned model.
    # A pre-flight that pre-flights a different configuration is worse than none.
    cfg = _load_config(config, domain, run_root="runs") if (config or domain) else QMineConfig()
    if gold:
        cfg.taxonomy.gold_sample_size = gold

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

    # Scale the volumes by the config that will actually run, so the number an
    # operator checks before spending moves when the spending does. Without it the
    # estimate read the same for a 600-row gold set and a 3,000-row one.
    from .llm.requirements import scaled_requirements

    # Mirror `ModelRegistry`'s own call exactly — every argument it passes, this
    # passes. Any divergence here silently re-introduces the bug above.
    plan = route(
        cat, av.usable, requirements=scaled_requirements(cfg),
        prefer=cfg.llm.model_overrides or None,
            capable_models=cfg.llm.capable_models or (),
        budget_usd=budget if budget is not None else cfg.llm.budget_usd,
        prefer_chinese_native=chinese or cfg.llm.prefer_chinese_native,
        excluded_labs=cfg.llm.excluded_labs,
    )
    if cfg.llm.excluded_labs:
        console.print(f"[dim]excluded labs: {', '.join(cfg.llm.excluded_labs)}[/dim]")
    if cfg.llm.model_overrides:
        pins = ", ".join(f"{r}={m}" for r, m in cfg.llm.model_overrides.items())
        console.print(f"[dim]pinned: {pins}[/dim]")
    console.print(f"[dim]volumes scaled to this config: gold set "
                  f"{cfg.taxonomy.gold_sample_size or 'derived from corpus size'}, "
                  f"{cfg.taxonomy.kappa_repair_rounds} repair round(s)[/dim]")
    r = Table("role", "model", "tier", "calls", "est. $", "fallback")
    for role, a in sorted(plan.assignments.items(), key=lambda kv: -kv[1].estimated_cost_usd):
        r.add_row(role, f"{a.provider}:{a.model}" if a.model else "[red]none[/red]",
                  a.tier, str(a.estimated_calls), f"{a.estimated_cost_usd:.3f}",
                  (a.fallbacks[:1] or [""])[0][:34])
    console.print(r)
    # The table's "model" column shows the GATEWAY (`qwen:dashscope/...`), and two
    # different labs reach you through the same gateway — so annotator_b and the
    # referee can look identical there while being properly independent. The
    # independence rule is by LAB, so state it in those terms or an operator
    # cannot check the one property that makes double-blind annotation mean
    # anything.
    from .llm.router import lab_of

    trio = {r: lab_of(plan.assignments[r].model)
            for r in ("annotator_a", "annotator_b", "referee")
            if r in plan.assignments and plan.assignments[r].model}
    if len(trio) == 3:
        ok = len(set(trio.values())) == 3
        mark = "[green]independent[/green]" if ok else "[red]NOT INDEPENDENT[/red]"
        console.print("\nannotator/referee labs: "
                      + ", ".join(f"{r.split('_')[-1]}={lab}" for r, lab in trio.items())
                      + f" — {mark}")

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
