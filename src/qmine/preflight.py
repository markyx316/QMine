"""Is this run going to work? — everything checkable before any money is spent.

A run costs $5-$7 and 3-4 hours. The failures that hurt most are not the subtle
ones; they are the ones that were knowable at second zero and were not looked
for: a run id that already exists, an input path that does not, a pinned model
nothing can route, a `--domain` naming a profile that is not there. Each of
those either dies minutes in or, worse, quietly produces a stand-in result that
looks complete.

TWO SEVERITIES, AND THE DIFFERENCE MATTERS.

* **blocking** — the run cannot succeed, or would produce output that means
  nothing about the corpus. `qmine_start_run` refuses on any of these no matter
  how spending was authorised, because a human clicking "approve" should not be
  able to start a doomed run.
* **warning** — it will run, and the operator should know something anyway: no
  risk screening, an assumed comparison axis, a first-run encoder download.

A PREFLIGHT THAT CRIES WOLF IS WORSE THAN NONE. Every check here has to be
quiet on a healthy run, so each one is written against a real failure and says
what to do about it. `qmine models` already carries the same lesson in its own
comment — a preflight that preflights a *different* configuration is worse than
none — which is why the routing check below builds the config exactly the way
`run` will.
"""

from __future__ import annotations

import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

Severity = Literal["blocking", "warning", "ok"]

#: Below this a taxonomy is being derived from too little evidence to mean much.
#: Not blocking — a small pilot corpus is a legitimate thing to run.
THIN_CORPUS_ROWS = 2_000

#: A full run writes embeddings for every row, several representations deep.
#: Measured on the delivered 20k-row runs: ~250 MB of `emb_*.npy` per run.
MIN_FREE_GB = 2.0


@dataclass
class Check:
    name: str
    severity: Severity
    detail: str
    fix: str = ""

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if not d["fix"]:
            d.pop("fix")
        return d


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)

    def add(self, name: str, severity: Severity, detail: str, fix: str = "") -> None:
        self.checks.append(Check(name, severity, detail, fix))

    @property
    def blocking(self) -> list[Check]:
        return [c for c in self.checks if c.severity == "blocking"]

    @property
    def warnings(self) -> list[Check]:
        return [c for c in self.checks if c.severity == "warning"]

    def as_dict(self, *, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        go = not self.blocking
        out: dict[str, Any] = {
            "verdict": "go" if go else "no_go",
            "blocking": [c.as_dict() for c in self.blocking],
            "warnings": [c.as_dict() for c in self.warnings],
            "passed": [c.name for c in self.checks if c.severity == "ok"],
            "summary": self.summary(),
        }
        out.update(extra or {})
        return out

    def summary(self) -> str:
        if self.blocking:
            return (f"NO-GO — {len(self.blocking)} blocking problem(s): "
                    + "; ".join(c.name for c in self.blocking))
        if self.warnings:
            return (f"GO, with {len(self.warnings)} thing(s) to know: "
                    + "; ".join(c.name for c in self.warnings))
        return "GO — every check passed."


# --------------------------------------------------------------------- checks

def _check_run_id(rep: Report, run_id: str | None, run_root: str) -> None:
    if not run_id:
        rep.add("run_id", "blocking", "No run id was given.",
                "Pick one: it names the directory the whole study lands in.")
        return
    existing = Path(run_root) / run_id
    # MIRRORS cli.run's own guard exactly (cli.py: `checkpoints.sqlite` OR
    # `llm_cache`). Deleting only the checkpoint still trips it, so a preflight
    # that checked one of the two would clear a run that `run` then refuses.
    if (existing / "checkpoints.sqlite").exists() or (existing / "llm_cache").exists():
        rep.add("run_id", "blocking",
                f"run id {run_id!r} already exists at {existing}: `run` refuses it, "
                "because reopening that graph thread and its LLM cache means two "
                "sources that disagree.",
                f"`qmine run --resume --run-id {run_id}` to continue it, "
                f"`qmine new-generation {run_id} --reason '...'` to keep the evidence "
                "and start fresh, or choose a new id.")
    else:
        rep.add("run_id", "ok", f"{run_id!r} is free.")


def _check_inputs(rep: Report, inputs: list[str]) -> list[Path]:
    if not inputs:
        rep.add("inputs", "blocking", "No input files were given.", "Name at least one export.")
        return []
    missing = [p for p in inputs if not Path(p).expanduser().exists()]
    if missing:
        rep.add("inputs", "blocking",
                f"{len(missing)} input path(s) do not exist: {', '.join(missing[:4])}",
                "Check the paths. A bare filename is read relative to the QMine checkout, "
                "not to your chat workspace.")
        return []
    empty = [p for p in inputs if Path(p).expanduser().stat().st_size == 0]
    if empty:
        rep.add("inputs", "blocking", f"empty file(s): {', '.join(empty)}", "Re-export them.")
        return []
    rep.add("inputs", "ok", f"{len(inputs)} file(s) present.")
    return [Path(p).expanduser() for p in inputs]


def _check_corpus(rep: Report, paths: list[Path], text_column: str | None) -> None:
    """Read each input once: row count, and whether a query column is findable."""
    if not paths:
        return
    try:
        from .prepare.inspect import is_textual, read_any
    except Exception as exc:  # noqa: BLE001 - environment, not logic
        rep.add("corpus", "warning", f"could not import the reader: {exc}")
        return
    total = 0
    for p in paths:
        try:
            df = read_any(p)
        except Exception as exc:  # noqa: BLE001
            rep.add("corpus", "blocking", f"{p.name} cannot be read: "
                    f"{type(exc).__name__}: {exc}",
                    "Re-export it as .xlsx, .csv, .parquet or .jsonl.")
            return
        total += len(df)
        if text_column and text_column not in df.columns:
            rep.add("corpus", "blocking",
                    f"{p.name} has no column {text_column!r}. It has: "
                    f"{', '.join(map(str, df.columns[:8]))}",
                    "Name the real query column, or drop --text-column and let it be "
                    "detected.")
            return
        if not text_column and not any(is_textual(str(df[c].dtype)) for c in df.columns):
            rep.add("corpus", "blocking", f"{p.name} has no textual column to mine.",
                    "The query text has to be in there somewhere.")
            return
    rep.add("corpus", "ok", f"{total:,} rows across {len(paths)} file(s), query column found.")
    if total < THIN_CORPUS_ROWS:
        rep.add("corpus_size", "warning",
                f"{total:,} rows is thin. A taxonomy derived from it describes this "
                "sample more than the phenomenon, and the K locator has little to work with.",
                "Fine for a pilot; say so when reporting the result.")


def _check_pooled(rep: Report, paths: list[Path], axis: str | None) -> None:
    """Only meaningful when several files are pooled into one run."""
    if len(paths) < 2:
        return
    stems = [p.stem for p in paths]
    if len(set(stems)) != len(stems):
        rep.add("snapshot_tags", "blocking",
                f"two inputs would take the same snapshot tag: {stems}",
                "Duplicate tags are refused rather than silently merged. Rename a file, "
                "or set the tags explicitly.")
    else:
        rep.add("snapshot_tags", "ok", f"{len(stems)} distinct snapshots: {', '.join(stems)}")
    if not axis:
        rep.add("comparison_axis", "warning",
                "No comparison axis was declared, so it defaults to `stratum` — the "
                "snapshots will be read as different SAMPLES of one period, not as a "
                "timeline. The arithmetic is identical either way; the prose inverts.",
                "If these are two periods of the same thing, say `axis=time`. Overlap "
                "between the files cannot settle it: measured 0.02% for the same surface "
                "a year apart and 63.3% for two head exports of it.")


def _check_domain(rep: Report, domain: str | None, config_dir: Path) -> None:
    profiles = sorted(p.stem for p in (config_dir / "domains").glob("*.yaml"))
    if domain and domain not in profiles:
        rep.add("domain_profile", "blocking",
                f"no domain profile named {domain!r}. Available: {', '.join(profiles)}",
                "Pick one of those, or drop --domain and accept generic defaults.")
    elif domain:
        rep.add("domain_profile", "ok", f"{domain} profile found.")
    else:
        rep.add("domain_profile", "warning",
                "No domain profile: generic Chinese defaults and NO vertical risk "
                "screening. Nothing will flag medical, financial or adult content as "
                "sensitive.",
                f"Consider --domain from: {', '.join(profiles)}")


def _check_routing(rep: Report, cfg: Any, fast: bool) -> dict[str, Any]:
    """Every role must resolve to a reachable model, against the RUN's config."""
    out: dict[str, Any] = {}
    try:
        from .llm.catalog import fetch
        from .llm.providers import detect
        from .llm.requirements import scaled_requirements
        from .llm.router import route
    except Exception as exc:  # noqa: BLE001
        rep.add("routing", "warning", f"could not load the router: {exc}")
        return out

    av = detect()
    if not av.configured:
        # A "paid" run with no keys silently becomes a deterministic stand-in
        # whose output looks complete and means nothing about the corpus. Asking
        # to spend and getting that back is a blocking mismatch, not a warning.
        rep.add("providers", "blocking",
                "No provider credentials found, so this would run the OFFLINE stand-in: "
                "complete-looking output that says nothing about your corpus.",
                "Put DEEPSEEK / ZHIPU / QWEN / OPENROUTER keys in QMine/.env, or run it "
                "deliberately with --offline if a wiring check is what you wanted.")
        return out
    rep.add("providers", "ok", f"reachable: {', '.join(sorted(av.configured))}")

    try:
        cat = fetch(cache_dir=".cache", ttl=6 * 3600, allow_network=True)
        plan = route(cat, av.usable, requirements=scaled_requirements(cfg),
                     prefer=cfg.llm.model_overrides or None,
                     capable_models=cfg.llm.capable_models or (),
                     budget_usd=cfg.llm.budget_usd,
                     prefer_chinese_native=cfg.llm.prefer_chinese_native,
                     excluded_labs=cfg.llm.excluded_labs)
    except Exception as exc:  # noqa: BLE001
        rep.add("routing", "blocking", f"the routing plan could not be built: "
                f"{type(exc).__name__}: {exc}",
                "Run `qmine models` to see it fail in full.")
        return out

    assignments = getattr(plan, "assignments", {}) or {}
    # AN EMPTY PLAN IS NOT A CLEAN ONE. When the catalogue cannot be built,
    # `fetch` returns the built-in floor with `models={}` and `route()` takes its
    # `if not cards:` early return — ZERO assignments, no exception. The
    # `unrouted` comprehension below is then empty too, so this check used to
    # report "routing ok" and the run died at its first agent call. Measured: the
    # p0_provider gate also passes, because `provider` reads `routed`.
    if not getattr(cat, "models", None) or not assignments:
        rep.add("routing", "blocking",
                f"the routing plan is EMPTY ({len(assignments)} roles, "
                f"{len(getattr(cat, 'models', {}) or {})} models in the catalogue). Nothing "
                "would be reachable; the run dies at its first agent call, after the free "
                "phases have run.",
                "Restore network access so the price feeds can be fetched, or point "
                "`llm.catalog_pinned` at a saved snapshot. Never launch on the floor "
                "catalogue.")
        return out
    # A KEY FOR THE PROVIDER A PIN NAMES IS NEVER CHECKED. A qualified pin
    # (`zhipu:glm-5.3-flash`) synthesises a card from the provider NAME alone, so
    # the plan looks clean and the client is built with an empty api_key. The 401
    # arrives deep in the run — and `annotator_b`, a 256-call role, ships with no
    # fallbacks, so there it is fatal after both annotators have been paid for.
    have = set(av.configured)
    orphan = {r: getattr(a, "provider", "?") for r, a in assignments.items()
              if getattr(a, "model", None) and getattr(a, "provider", None)
              and a.provider not in have}
    if orphan:
        from .llm.providers import BY_KEY

        needs = sorted({v for p in orphan.values()
                        for v in getattr(BY_KEY.get(p), "env_vars", ()) or ()})
        rep.add("provider_keys", "blocking",
                "role(s) are routed to a provider with no key: "
                + ", ".join(f"{r}→{p}" for r, p in sorted(orphan.items())),
                f"Set one of {', '.join(needs) or 'the provider key'} in QMine/.env, or "
                "repin those roles to a provider you hold. A pin is never checked against "
                "your keys, so this only surfaces as a 401 once the run is under way.")
    unrouted = [r for r, a in assignments.items() if not getattr(a, "model", None)]
    if unrouted:
        rep.add("routing", "blocking",
                f"{len(unrouted)} role(s) resolve to no model: {', '.join(sorted(unrouted))}. "
                "An unroutable pin is treated as a config error that stops the run.",
                "Check `llm.model_overrides` in the config, and whether the lab it names "
                "is in `excluded_labs`.")
    else:
        rep.add("routing", "ok", f"{len(assignments)} role(s) routed.")

    no_fallback = [r for r, a in assignments.items()
                   if not getattr(a, "fallbacks", ()) and getattr(a, "model", None)]
    if no_fallback:
        rep.add("routing_fallbacks", "warning",
                f"{len(no_fallback)} role(s) are pinned with NO fallback "
                f"({', '.join(sorted(no_fallback)[:5])}): one provider outage removes that "
                "role from the run.")

    cost = getattr(plan, "total_cost_usd", None)
    if cost is not None:
        out["estimated_cost_usd"] = cost
        out["estimated_calls"] = sum(int(getattr(a, "estimated_calls", 0) or 0)
                                     for a in assignments.values())
    unpriced = [r for r, a in assignments.items()
                if getattr(a, "input_per_mtok", None) is None and getattr(a, "model", None)]
    if unpriced:
        rep.add("cost_is_partly_unknown", "warning",
                f"{len(unpriced)} role(s) use a model with no published price "
                f"({', '.join(sorted(unpriced)[:5])}), so the estimate UNDER-reports: the "
                "cost is unknown, not zero.")
    if fast:
        rep.add("mode", "warning",
                "FAST mode: the analysis is full size but the second-opinion layer is "
                "dropped, so kappa will be ABSENT — not 1.0, not 0.0.",
                "Use a full run when the result has to defend its own reliability.")
    return out


def _check_environment(rep: Report, run_root: str) -> None:
    # THE RUN DOES NOT READ $HF_HOME. p3 passes `cache_folder=<run_root>/.hf`, so
    # checking the environment variable answered a question nobody asked: a
    # machine with a full $HF_HOME still died at p3 with "no encoder candidate
    # could be loaded". Measured here on 2026-09-21, after this preflight had
    # already said GO.
    cache = Path(run_root).resolve() / ".hf"
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        rep.add("encoders", "blocking",
                "sentence-transformers is not installed, so phase 3 cannot embed anything.",
                "`.venv/bin/pip install -e '.[all]'`")
        return
    cached = [d for d in cache.glob("models--*") if any(d.rglob("config.json"))]
    if not cached:
        rep.add("encoders", "warning",
                f"no encoder is cached in {cache} — the run downloads a few hundred MB "
                "before phase 3, which looks like a hang. Note this is the RUN's cache, "
                "not $HF_HOME.",
                "Expected once per run root; later runs there reuse it.")
    else:
        rep.add("encoders", "ok", f"{len(cached)} encoder(s) cached in {cache.name}.")
    try:
        free_gb = shutil.disk_usage(Path(run_root).resolve().parent).free / 1e9
        if free_gb < MIN_FREE_GB:
            rep.add("disk", "blocking",
                    f"only {free_gb:.1f} GB free where runs are written; embeddings alone "
                    f"need about {MIN_FREE_GB:.0f} GB.",
                    "Free some space or point --run-root elsewhere.")
        else:
            rep.add("disk", "ok", f"{free_gb:.0f} GB free.")
    except Exception:  # noqa: BLE001 - not worth failing a preflight over
        pass


# ----------------------------------------------------------------- entry point

def preflight(*, inputs: list[str], run_id: str | None, domain: str | None = None,
              config: str | None = None, fast: bool = False, axis: str | None = None,
              text_column: str | None = None, run_root: str = "runs") -> dict[str, Any]:
    """Everything knowable before the first paid call. Makes no model call."""
    rep = Report()
    _check_run_id(rep, run_id, run_root)
    paths = _check_inputs(rep, inputs)
    _check_corpus(rep, paths, text_column)
    _check_pooled(rep, paths, axis)

    from .cli import CONFIG_DIR, _load_config
    from .llm.env import load_dotenv

    # LOAD `.env` FIRST, exactly as a run does — and FROM THE CHECKOUT, not from
    # the working directory. `load_dotenv()` searches upward from `cwd`, which is
    # fine for a person in a shell and wrong for the PreToolUse hook: dsh runs it
    # somewhere else, so the hook found no credentials, reported "this would run
    # the OFFLINE stand-in", and DENIED a perfectly healthy run. Measured on
    # 2026-09-21 by running the hook from /tmp. The cwd search is kept after it
    # for layouts where the corpus config lives beside the caller.
    load_dotenv(Path(__file__).resolve().parents[2])
    load_dotenv()
    _check_domain(rep, domain, CONFIG_DIR)
    extra: dict[str, Any] = {}
    try:
        cfg = _load_config(config, domain, run_root=run_root,
                           mode="fast" if fast else "full")
    except Exception as exc:  # noqa: BLE001
        rep.add("config", "blocking",
                f"the config does not load: {type(exc).__name__}: {exc}",
                "Fix the YAML, or drop --config to use configs/live.yaml.")
    else:
        rep.add("config", "ok", f"mode={cfg.mode}, "
                f"gold={cfg.taxonomy.gold_sample_size or 'derived'}")
        extra = _check_routing(rep, cfg, fast)
    _check_environment(rep, run_root)
    return rep.as_dict(extra=extra)
