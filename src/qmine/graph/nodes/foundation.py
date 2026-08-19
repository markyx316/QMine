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
    ref_labels = {c: raw[c].astype(str).tolist() for c in cfg.data.reference_label_columns if c in raw.columns}
    weights = raw[cfg.data.weight_column].tolist() if cfg.data.weight_column in raw.columns else None
    df = build_frame(raw[cfg.data.text_column].astype(str).tolist(), reference_labels=ref_labels, weights=weights)

    if cfg.data.sample_size and cfg.data.sample_size < len(df):
        from ...determinism import deterministic_subsample

        df = df.iloc[deterministic_subsample(len(df), cfg.data.sample_size, cfg.seed_metric)].reset_index(drop=True)
        df["row_id"] = np.arange(len(df))

    corpus_ref = deps.store.put_table("corpus", df, producer="p1", summary=f"{len(df)} queries with surface features")
    deps.cache_put("corpus", df)

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

    lo, hi = cfg.gates.template_coverage_range
    gate = deps.gate(
        "p1_template_coverage",
        "p1",
        passed=lo <= cov["union_coverage"] <= hi,
        observed={"union_coverage": cov["union_coverage"], "n_groups": len(groups)},
        threshold={"range": [lo, hi]},
        message=(
            f"{len(groups)} phrasing families cover {cov['union_coverage'] * 100:.1f}% of the corpus"
            + (f" — {selection['diagnosis']}" if selection.get("diagnosis") else "")
        ),
        remediation=(
            "Coverage below the window means the fragmentation metric will rest on too "
            "few rows — mine more affixes or loosen the seed patterns. Above it means the "
            "groups have stopped implying shared intent and are matching the language itself."
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

    return {
        "phase": "p2",
        "artifacts": {"corpus": corpus_ref, "data_audit": audit_ref,
                      "template_groups": tg_ref, "risk_screen": risk_ref,
                      "language_profile": lang_ref},
        "gates": {gate.name: gate, lang_gate.name: lang_gate},
        "completed_phases": ["p1"],
        "events": events,
    }


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
