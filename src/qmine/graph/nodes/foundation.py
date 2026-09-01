"""Phases 0-1 — engineering foundation, data audit, template mining, risk screen."""

from __future__ import annotations

import platform
import sys
import time
from typing import Any

import numpy as np
import pandas as pd

from ...determinism import SeedPolicy, hash_texts, seed_everything
from ...ops.audit import audit_corpus, build_frame, screen_risk
from ...ops.templates import (
    build_groups,
    coverage,
    group_masks,
    mine_affixes,
    select_groups_for_coverage,
)
from ...state import PipelineState
from ..deps import Deps


def p0_foundation(state: PipelineState, deps: Deps) -> dict[str, Any]:
    """Pin the environment, seed every RNG, write the run manifest.

    The manifest is the answer to "what produced this number".  It records the
    interpreter, the library versions, the config hash, the seed policy, the
    prompt hashes, and the hash of the input corpus — so a result can be
    attributed to a specific state of the world rather than to "the pipeline".
    """
    from ...agents.base import prompt_manifest

    seed_everything(deps.cfg.seed_metric)
    deps.emit(f"P0 foundation — run {deps.run_id}, generation {state.get('generation', 1)}")

    versions: dict[str, str] = {}
    for mod in ("numpy", "pandas", "scipy", "sklearn", "langgraph", "langchain_core",
                "sentence_transformers", "torch", "umap"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:
            versions[mod] = "not installed"

    manifest = {
        "run_id": deps.run_id,
        "generation": state.get("generation", 1),
        "created_at": time.time(),
        "config_hash": deps.cfg.config_hash,
        "domain": deps.cfg.domain.key,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "interpreter": sys.executable,
        "versions": versions,
        "seed_policy": SeedPolicy(
            metric=deps.cfg.seed_metric, viz=deps.cfg.seed_viz, replay=tuple(deps.cfg.seed_replay)
        ).as_dict(),
        "llm": deps.registry.usage(),
        "prompt_hashes": prompt_manifest(),
        "provenance_note": deps.registry.provenance_note(),
    }
    ref = deps.store.put_json("run_manifest", manifest, producer="p0", summary="run provenance")
    deps.cfg.dump(deps.store.gen_dir / "config.resolved.yaml")

    # RUNNING ON THE STAND-IN IS A FACT ABOUT THE RUN, NOT A LOG LINE.
    #
    # This pipeline IS a team of agents; with the stand-in it is a deterministic
    # function wearing their output shape. It produces a full set of
    # deliverables, every gate passes, and nothing in the documents says the
    # prose was not written by a model — which is why `verify_run.py` checks
    # `llm_usage.provider` at all. A warning in `run.log` is not enough: the
    # question "was this run real?" has to be answerable from the artifacts, so
    # it is a gate, and it is recorded whichever way it goes.
    provider = deps.registry.provider
    live = provider not in ("offline", "mock", "")
    try:
        from ...llm.providers import detect

        reachable = list(detect().usable)
    except Exception:  # noqa: BLE001 — the gate must not be what breaks a run
        reachable = []
    deps.gate(
        "p0_provider", "p0",
        passed=live,
        observed={"provider": provider, "configured_providers": reachable},
        threshold={"rule": "a run must use real models unless offline was asked for"},
        message=(f"real agents: provider={provider}" if live else
                 "THE OFFLINE STAND-IN WROTE THIS RUN. Every deliverable below is "
                 "the output of a deterministic function, not of a model — it "
                 "looks complete and means nothing about the corpus."),
        remediation=("Put provider keys in `QMine/.env` and relaunch. `qmine models` "
                     "shows what is reachable and spends nothing."),
        warn_only=True,
    )
    if not live:
        deps.emit("  ⚠️  OFFLINE STAND-IN — no model wrote anything in this run")

    return {
        "phase": "p1",
        "artifacts": {"run_manifest": ref},
        "completed_phases": ["p0"],
        "events": [f"P0 complete — {deps.registry.provider} provider, seed {deps.cfg.seed_metric}"],
    }


def p1_audit(state: PipelineState, deps: Deps) -> dict[str, Any]:
    """Profile the corpus, mine template groups, screen for risk.

    Deliberately measurement-only.  The temptation at this point is to start
    sketching categories from the first thousand rows you read, which anchors
    the taxonomy on whatever happened to be at the top of the file.  Design
    happens in Phase 2, by agents who read a stratified slice on purpose.
    """
    cfg = deps.cfg
    deps.emit("P1 audit — profiling corpus")

    raw = _load_input(cfg)
    declared = [c for c in cfg.data.reference_label_columns if c in raw.columns]
    unused = _label_like_columns(raw, cfg) if not declared else []
    ref_labels = {c: raw[c].astype(str).tolist() for c in declared}
    weights = raw[cfg.data.weight_column].tolist() if cfg.data.weight_column in raw.columns else None
    df = build_frame(raw[cfg.data.text_column].astype(str).tolist(), reference_labels=ref_labels, weights=weights)

    if cfg.data.sample_size and cfg.data.sample_size < len(df):
        from ...determinism import deterministic_subsample

        df = df.iloc[deterministic_subsample(len(df), cfg.data.sample_size, cfg.seed_metric)].reset_index(drop=True)
        df["row_id"] = np.arange(len(df))

    corpus_ref = deps.store.put_table("corpus", df, producer="p1", summary=f"{len(df)} queries with surface features")
    deps.cache_put("corpus", df)

    # A CORPUS THAT CARRIES LEGACY LABELS AND A RUN THAT DECLARES NONE LOOK THE
    # SAME IN EVERY LOG LINE, AND THEY ARE NOT THE SAME RUN.
    #
    # `--reference-columns` is a launch flag with an empty default. Omitting it on
    # a corpus that has them changes EIGHT things quietly: the gold set and the
    # pilot stop being stratified by legacy label (so kappa and everything
    # downstream shift), the blindness firewall is armed with fewer forbidden
    # terms, the legacy-audit researcher returns nothing, the corpus audit loses
    # its legacy distribution, and the delivered table loses its crosswalk. Only
    # the researcher says anything, and it says "this angle contributed nothing",
    # which reads like a finding about the corpus rather than about the command.
    if unused:
        deps.emit(f"  ⚠ this corpus carries {len(unused)} label-like column(s) that no "
                  f"`--reference-columns` flag declared: {', '.join(unused[:4])}")
    ref_gate = deps.gate(
        "p1_reference_columns_declared", "p1",
        passed=not unused,
        observed={"declared": declared, "undeclared_label_like": unused,
                  "all_columns": [str(c) for c in raw.columns][:20]},
        threshold={"rule": "a label-like column present in the input must be declared or "
                           "knowingly ignored"},
        message=(f"reference labels: {', '.join(declared)}" if declared else
                 ("no reference label columns, and the corpus offers none"
                  if not unused else
                  f"the corpus carries {', '.join(unused[:4])} and the run declared NONE — "
                  "the gold set and pilot are UNSTRATIFIED by legacy label and the "
                  "blindness firewall is armed without those values")),
        remediation=("Relaunch with `--reference-columns " + ",".join(unused[:3]) + "`, or "
                     "record that these columns are deliberately ignored. Runs that differ "
                     "in this are not comparable: the gold set is sampled differently."),
        warn_only=True,
    )

    report = audit_corpus(df, text_col=cfg.data.text_column, reference_cols=cfg.data.reference_label_columns)
    report["input_hash"] = hash_texts(df[cfg.data.text_column].astype(str).tolist())

    # --- template mining: seeds from the profile, plus what the corpus offers
    affixes = mine_affixes(df[cfg.data.text_column].astype(str).tolist())
    discovered = (
        [{"affix": a["affix"], "side": "suffix"} for a in affixes["suffixes"][:25]]
        + [{"affix": a["affix"], "side": "prefix"} for a in affixes["prefixes"][:12]]
    )
    candidates = build_groups(df, seeds=cfg.domain.template_seeds, discovered=discovered, text_col=cfg.data.text_column)
    groups, selection = select_groups_for_coverage(candidates, df, text_col=cfg.data.text_column)
    cov = coverage(groups, df, text_col=cfg.data.text_column)

    masks = group_masks(groups, df, text_col=cfg.data.text_column)
    trusted = group_masks(groups, df, text_col=cfg.data.text_column, trusted_only=True)
    if not trusted:                     # no seeds survived: fall back, and say so
        trusted = masks
    deps.cache_put("template_masks", trusted)      # judges representations
    deps.cache_put("template_masks_all", masks)    # coverage and display

    tg_ref = deps.store.put_json(
        "template_groups",
        {
            "groups": [g.model_dump() for g in groups],
            "trusted_groups": [g.name for g in groups if g.trusted],
            "metric_contract": (
                "only trusted families judge a representation. A mined marker like "
                "'是什么' attaches to many intents, so its spread across clusters "
                "measures the marker rather than the partition."
            ),
            "coverage": cov,
            "selection": selection,
            "all_candidates": [g.model_dump() for g in candidates],
            "affixes": affixes,
        },
        producer="p1",
        summary=f"{len(groups)} phrasing families covering {cov['union_coverage'] * 100:.1f}%",
    )

    # --- language composition -------------------------------------------
    # Run before anything chooses a tokeniser or an encoder, because both
    # decisions depend on what scripts are actually present rather than on what
    # the profile assumed.
    from ...ops.language import char_ngram_for, profile_corpus, tokenizer_for

    langprof = profile_corpus(df[cfg.data.text_column].astype(str).tolist())
    deps.cache_put("language_profile", langprof)
    lang_ref = deps.store.put_json(
        "language_profile",
        {k: v for k, v in langprof.items() if k != "row_labels"},
        producer="p1",
        summary=f"{langprof['dominant']} {langprof['dominant_share']:.1%}, "
                f"posture {langprof['posture']}",
    )
    deps.store.put_matrix(
        "row_language",
        np.array(langprof["row_labels"], dtype=object).astype("U24"),
        producer="p1", summary="per-row script label",
    )

    # If the profile's assumptions contradict the data, say so loudly. A Chinese
    # tokeniser on a Latin corpus is not a subtle degradation.
    implied_tok = tokenizer_for(langprof["dominant"])
    implied_ngram = char_ngram_for(langprof["dominant"])

    # Resolve `auto` now that we know what the corpus actually is, and record the
    # resolution so the run manifest shows what was used rather than what was asked for.
    if cfg.domain.tokenizer == "auto":
        cfg.domain.tokenizer = implied_tok
        cfg.domain.char_ngram_range = implied_ngram
        deps.emit(f"  tokenizer auto-resolved to {implied_tok!r}, char n-grams {implied_ngram} "
                  f"({langprof['dominant']}-dominant corpus)")
        langprof["resolved_tokenizer"] = implied_tok
        langprof["resolved_char_ngram_range"] = list(implied_ngram)
    elif implied_tok != cfg.domain.tokenizer:
        deps.emit(
            f"  NOTE: corpus is {langprof['dominant']}-dominant, which implies tokenizer "
            f"{implied_tok!r} and char n-grams {implied_ngram}, but the profile says "
            f"{cfg.domain.tokenizer!r} / {tuple(cfg.domain.char_ngram_range)}"
        )

    lang_gate = deps.gate(
        "p1_minority_language_risk", "p1",
        passed=langprof["posture"] != "minority_at_risk",
        observed={"dominant": langprof["dominant"], "dominant_share": langprof["dominant_share"],
                  "minority": langprof["minority"], "posture": langprof["posture"]},
        threshold={"minority_share_window": "under 0.5% (ignorable) or over 5% (handled explicitly)"},
        message=langprof["rationale"],
        remediation=(
            "A minority language between 0.5% and 5% is the dangerous band: too small to earn "
            "its own clusters at a normal K, too big to lose. Phase 6 will subdivide any "
            "minority-dominated family in a language-appropriate representation; check the "
            "minority_dilution metric in the Phase 9 panel to confirm it worked."
        ),
        warn_only=True,
    )

    risk = screen_risk(df, cfg.domain.risk_categories, text_col=cfg.data.text_column)
    risk_ref = deps.store.put_json("risk_screen", risk, producer="p1",
                                   summary=f"{risk['total_flagged']} rows pre-flagged")
    audit_ref = deps.store.put_json("data_audit", report, producer="p1", summary="corpus profile")

    # Gate on the thing that is actually required — enough labelled rows for the
    # downstream metrics to mean anything — not on a coverage SHARE.
    #
    # The share was a K12 observation (20-40%) applied to every corpus, and it does
    # not travel: the e-commerce corpus scores 0.534 and gets flagged for being more
    # templated than K12, which is a fact about e-commerce queries rather than a
    # defect. There is no reason high coverage is bad in itself. The risk the upper
    # bound was reaching for — groups so broad they stop implying a shared intent —
    # is already tested directly, per group, by the cohesion check that rejects a
    # candidate whose members do not cluster together. A share cannot detect it and
    # the cohesion check does not need it.
    #
    # The lower bound is real but belongs in rows: the fragmentation and alignment
    # metrics are computed over grouped rows, so what matters is their count.
    covered_rows = int(round(cov["union_coverage"] * report['n_rows']))
    min_rows = max(cfg.gates.min_template_rows,
                   int(cfg.gates.min_template_row_fraction * report['n_rows']))
    gate = deps.gate(
        "p1_template_coverage",
        "p1",
        passed=covered_rows >= min_rows and len(groups) >= 2,
        observed={"covered_rows": covered_rows, "union_coverage": cov["union_coverage"],
                  "n_groups": len(groups)},
        threshold={"min_covered_rows": min_rows, "min_groups": 2},
        message=(
            f"{len(groups)} phrasing families cover {covered_rows:,} rows "
            f"({cov['union_coverage'] * 100:.1f}% of the corpus; the share is reported, "
            "not gated — it is a property of the corpus)"
            + (f" — {selection['diagnosis']}" if selection.get("diagnosis") else "")
        ),
        remediation=(
            "Too few grouped rows: every downstream interpretability metric — "
            "fragmentation, and the intent-alignment score that locates K — is computed "
            "over these rows, so below this count they measure noise. Mine more affixes "
            "or loosen the seed patterns. Note that HIGH coverage is not a defect; "
            "whether a group is too broad is tested per group by the cohesion check."
        ),
        warn_only=True,
    )

    events = [
        f"P1: {report['n_rows']} rows, {report['n_unique']} unique, "
        f"median length {report['length']['p50']:.0f}",
        f"P1: {len(groups)} template groups ({sum(g.trusted for g in groups)} trusted "
        f"to judge representations) → {cov['union_coverage'] * 100:.1f}% coverage",
        f"P1: risk pre-screen flagged {risk['total_flagged']} rows ({risk['total_share'] * 100:.2f}%)",
        f"P1: language — {langprof['dominant']} {langprof['dominant_share']:.1%}, "
        f"minority {langprof['minority_share']:.1%} ({langprof['posture']})",
    ]
    if report.get("reference_taxonomy"):
        for col, info in report["reference_taxonomy"].items():
            n_sus = len(info["form_defined_suspects"])
            if n_sus:
                events.append(
                    f"P1: legacy column {col} has {n_sus} shape-defined or catch-all classes "
                    "— reference only, not an inheritable skeleton"
                )

    # WHEN THE VERTICAL IS UNKNOWN, GO AND LOOK. `DomainScoutAgent` has existed,
    # been registered, been given a prompt and a routing requirement — and was
    # never called, so a run on an unfamiliar corpus proceeded on the generic
    # profile's deliberate blanks and nothing tried to fill them.
    #
    # It emits HYPOTHESES, not configuration. Its seeds are candidate phrasing
    # families that Phase 1's miner and Phase 3's tightness test still have to
    # accept or reject; its risk categories are candidates for a human to
    # approve; its vertical is a steer for the P2a researchers, who are the ones
    # actually chartered to work out what the corpus is. Nothing it says changes
    # a parameter, and it runs ONLY when no vertical was declared — a supplied
    # profile is the operator's statement about their own data and outranks it.
    scout_ref = None
    if cfg.domain.key == "generic" and not deps.registry.is_offline:
        scout_ref = _scout_unknown_domain(deps, df, report)
        if scout_ref is not None:
            events.append("P1: no vertical was declared — the domain scout's "
                          "hypotheses are recorded for P2a, and nothing else")

    arts = {"corpus": corpus_ref, "data_audit": audit_ref,
            "template_groups": tg_ref, "risk_screen": risk_ref,
            "language_profile": lang_ref}
    if scout_ref is not None:
        arts["domain_scout"] = scout_ref
    return {
        "phase": "p2",
        "artifacts": arts,
        # `deps.gate()` returns a GateResult and registers nothing — a gate this
        # node creates and does not return here reaches the log and no operator.
        "gates": {gate.name: gate, lang_gate.name: lang_gate, ref_gate.name: ref_gate},
        "completed_phases": ["p1"],
        "events": events,
    }



def _scout_unknown_domain(deps: Deps, df: Any, audit: dict[str, Any]) -> Any:
    """Ask what this corpus is, when nobody told us. Hypotheses only.

    Fails soft on purpose: a scout that cannot run must not stop a run that is
    otherwise fine, and its absence simply leaves the generic profile's blanks
    blank — which is where they started.
    """
    from ...agents.roles import DomainScoutAgent
    from ...ops.audit import stratified_sample

    cfg = deps.cfg
    try:
        n = min(300, len(df))
        idx = stratified_sample(df, n, seed=cfg.seed_metric)
        sample = [str(x) for x in df[cfg.data.text_column].iloc[idx].tolist()]
        # The measured profile only — no scores, and no guesses about the
        # vertical, which is the thing being asked.
        profile = {k: audit.get(k) for k in
                   ("n_rows", "n_unique", "median_chars", "p90_chars") if k in audit}
        out = DomainScoutAgent(deps.agent_ctx()).run(sample=sample, profile=profile)
    except Exception as exc:  # noqa: BLE001
        deps.emit(f"  domain scout unavailable ({type(exc).__name__}) — "
                  "continuing on the generic profile")
        return None

    deps.emit(f"  domain scout: {out.vertical or '(undetermined)'} "
              f"(confidence {out.confidence}"
              + (", spans multiple verticals" if out.spans_multiple_verticals else "")
              + f"); {len(out.candidate_template_seeds)} candidate phrasing seed(s), "
              f"{len(out.candidate_risk_categories)} candidate risk categor(ies)")
    if out.confidence == "low":
        deps.emit("  ⚠ the scout is NOT confident — treat its vertical as a guess and "
                  "read P2a's researchers as the actual answer")
    return deps.store.put_json(
        "domain_scout",
        {**out.model_dump(),
         "status": "HYPOTHESES ONLY — seeds are validated by the Phase 1 miner and "
                   "the Phase 3 tightness test; risk categories need human approval; "
                   "nothing here changed a parameter in this run"},
        producer="p1",
        summary=f"vertical={out.vertical or '?'} ({out.confidence})")

def _load_input(cfg: Any) -> pd.DataFrame:
    path = cfg.data.input_path
    if not path:
        raise ValueError("config.data.input_path is not set")
    if str(path).endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    if str(path).endswith(".parquet"):
        return pd.read_parquet(path)
    if str(path).endswith((".jsonl", ".json")):
        return pd.read_json(path, lines=str(path).endswith(".jsonl"))
    return pd.read_csv(path)


def _label_like_columns(raw: Any, cfg: Any) -> list[str]:
    """Columns that look like pre-existing labels nobody declared.

    Deliberately conservative: a low-cardinality text column that is not the
    text, weight or id column. The point is to notice a corpus that HAS legacy
    labels while the run declares none — a free-text column has cardinality near
    the row count and never trips this.
    """
    skip = {cfg.data.text_column, cfg.data.weight_column, "row_id", "id", "query_id"}
    out = []
    for col in raw.columns:
        if str(col) in skip:
            continue
        try:
            series = raw[col]
            # NOT `dtype != object`: pandas 3 reads text columns as `str`, so an
            # object check silently matched nothing and the guard passed on the
            # very corpus it was written for. Ask what the column is NOT instead.
            if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
                continue
            n_unique = series.nunique(dropna=True)
        except Exception:  # noqa: BLE001
            continue
        if 2 <= n_unique <= max(2, int(0.05 * len(raw))):
            out.append(str(col))
    return out
