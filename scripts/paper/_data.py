"""Shared data loaders and scoring helpers for paper figures.

All paper figure scripts pull through this module so we have one source of
truth for: which run files we use, how CRPS / Brier are computed, how
continuous CRPS is normalized, and which models exist in which run.

Normalization (continuous CRPS):
    For each continuous template we divide raw CRPS (and MAE) by a fixed
    per-template value range, matching the human-pilot bin edges:
        cities      -> 40
        techs       -> 60
        treasury    -> 2000
    A model's reported "normalized CRPS" is the mean of per-question
    normalized CRPS across all of that model's continuous predictions
    (or a sub-slice when filtered by template/horizon). This is the same
    convention used in `scripts/compare_human_to_existing_models.py` and
    the human-baseline analysis.

Brier score (binary) is already in [0, 1] and is not normalized.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from ._style import REPO_ROOT


# --------------------------------------------------------------------------
# Run file locations
# --------------------------------------------------------------------------

EVAL_RUNS_DIR = REPO_ROOT / "data" / "evaluations" / "runs"
RUN_NONH0 = EVAL_RUNS_DIR / "final_2026-03-07_newq_uncond_nonh0_bin_plus_cont_all30.json"
RUN_H0_BINARY = EVAL_RUNS_DIR / "final_2026-03-07_newq_uncond_h0_all30.json"

# Archived 10-model runs. The "anthropic_retry" variants supersede the
# originals (same questions, retried Anthropic predictions).
ARCHIVE_RUNS_DIR = REPO_ROOT / "archive" / "20260302_121948" / "evaluations" / "runs"
ARCHIVE_CONT_UNCOND = ARCHIVE_RUNS_DIR / "cont_uncond_all.anthropic_retry.json"
ARCHIVE_REPUBLIC_BASELINE_BIN = ARCHIVE_RUNS_DIR / "republic_baseline_binary_all.anthropic_retry.json"
ARCHIVE_REPUBLIC_CONDITIONAL_BIN = ARCHIVE_RUNS_DIR / "republic_conditional_binary_all.anthropic_retry.json"
ARCHIVE_REPUBLIC_BASELINE_CONT = ARCHIVE_RUNS_DIR / "republic_baseline_continuous_all.json"
ARCHIVE_REPUBLIC_CONDITIONAL_CONT = ARCHIVE_RUNS_DIR / "republic_conditional_continuous_all.anthropic_retry.json"
ARCHIVE_GOLD500_BASELINE_BIN = ARCHIVE_RUNS_DIR / "gold500_baseline_binary_all.anthropic_retry.json"
ARCHIVE_GOLD500_CONDITIONAL_BIN = ARCHIVE_RUNS_DIR / "gold500_conditional_binary_all.anthropic_retry.json"
ARCHIVE_GOLD500_BASELINE_CONT = ARCHIVE_RUNS_DIR / "gold500_baseline_continuous_all.anthropic_retry.json"
ARCHIVE_GOLD500_CONDITIONAL_CONT = ARCHIVE_RUNS_DIR / "gold500_conditional_continuous_all.anthropic_retry.json"

# H0 continuous outputs (commit 3.5 — produced by run_h0_continuous_model_eval.py).
H0_CONT_DIR = REPO_ROOT / "data" / "human_baseline" / "aggregate_model_comparison" / "h0_continuous_eval"
H0_CONT_RESULTS = H0_CONT_DIR / "h0_results.json"
H0_CONT_QUESTIONS = H0_CONT_DIR / "h0_questions.csv"

# Human-baseline aggregate comparison artifacts (commit 3).
HB_AGG_DIR = REPO_ROOT / "data" / "human_baseline" / "aggregate_model_comparison"
HB_AGG_COMPARISON = HB_AGG_DIR / "aggregate_comparison.csv"
HB_HUMAN_REF = HB_AGG_DIR / "human_reference_summary.csv"
HB_MODEL_SETTING = HB_AGG_DIR / "model_setting_summary.csv"
HB_MODEL_TEMPLATE_HORIZON = HB_AGG_DIR / "model_template_horizon_summary.csv"


# --------------------------------------------------------------------------
# Continuous template metadata
# --------------------------------------------------------------------------

# Template-id (model side) -> human-baseline template family + value range.
# Value ranges match data/human_baseline/bin_schema.json bin edges.
CONTINUOUS_TEMPLATES: dict[str, dict[str, Any]] = {
    "cities_count_continuous": {"family": "cities",   "value_range": 40.0},
    "techs_continuous":        {"family": "techs",    "value_range": 60.0},
    "treasury_continuous":     {"family": "treasury", "value_range": 2000.0},
}

QUANTILE_LEVELS = {"p10": 0.10, "p25": 0.25, "p50": 0.50, "p75": 0.75, "p90": 0.90}

TURN_TO_HORIZON = {
    60: "H0",
    90: "H1",
    120: "H2",
    150: "H3",
    180: "H4",
    210: "H5",
    240: "H6",
    270: "H7",
}
HORIZON_TURN_OFFSET = {"H0": 0, "H1": 30, "H2": 60, "H3": 90, "H4": 120, "H5": 150, "H6": 180, "H7": 210}


# --------------------------------------------------------------------------
# Run loaders
# --------------------------------------------------------------------------

def load_run(path: Path) -> dict:
    """Load an evaluation run JSON. Raises if missing."""
    if not path.exists():
        raise FileNotFoundError(f"Run not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def models_in_run(run: dict) -> list[str]:
    return list(run["model_results"].keys())


# --------------------------------------------------------------------------
# Question parsing helpers
# --------------------------------------------------------------------------

def infer_horizon(question_text: str) -> tuple[int | None, str | None]:
    match = re.search(r"turn\s+(\d+)", question_text)
    if not match:
        return None, None
    turn = int(match.group(1))
    return turn, TURN_TO_HORIZON.get(turn)


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def valid_percentiles(prediction: dict) -> dict[str, float] | None:
    if not prediction:
        return None
    if prediction.get("error"):
        return None
    # Run files store quantiles either flat on the prediction or nested under
    # a `percentiles` key. Accept both.
    source = prediction.get("percentiles") or prediction
    out: dict[str, float] = {}
    for key in QUANTILE_LEVELS:
        value = source.get(key)
        if value is None:
            return None
        try:
            out[key] = float(value)
        except (TypeError, ValueError):
            return None
    return out


def quantile_crps(percentiles: dict[str, float], true_value: float) -> float:
    """Pinball-loss CRPS approximation across the five quantile levels."""
    total = 0.0
    for key, tau in QUANTILE_LEVELS.items():
        residual = true_value - percentiles[key]
        total += tau * residual if residual >= 0 else (tau - 1) * residual
    return (2 / len(QUANTILE_LEVELS)) * total


def brier(prob: float, outcome: int) -> float:
    return (prob - outcome) ** 2


# --------------------------------------------------------------------------
# Score frames
# --------------------------------------------------------------------------

@dataclass
class ContinuousScore:
    model: str
    question_id: str
    template_id: str
    family: str          # cities | techs | treasury
    horizon: str | None  # H0..H6
    turn: int | None
    crps: float
    mae: float
    normalized_crps: float
    normalized_mae: float


@dataclass
class BinaryScore:
    model: str
    question_id: str
    template_id: str
    horizon: str | None
    turn: int | None
    brier: float


def score_continuous(run: dict) -> list[ContinuousScore]:
    """Score every continuous prediction in a run with normalized CRPS/MAE."""
    rows: list[ContinuousScore] = []
    for question in run["questions"]:
        if question.get("question_type") != "continuous":
            continue
        template_id = question.get("template_id")
        meta = CONTINUOUS_TEMPLATES.get(template_id)
        if meta is None:
            continue
        value_range = meta["value_range"]
        family = meta["family"]
        turn, horizon = infer_horizon(question["question_text"])
        gt = float(question["ground_truth"])
        for model_id, prediction in question["predictions"].items():
            pcts = valid_percentiles(prediction)
            if pcts is None:
                continue
            crps = quantile_crps(pcts, gt)
            mae = abs(pcts["p50"] - gt)
            rows.append(ContinuousScore(
                model=model_id,
                question_id=question["question_id"],
                template_id=template_id,
                family=family,
                horizon=horizon,
                turn=turn,
                crps=crps,
                mae=mae,
                normalized_crps=crps / value_range,
                normalized_mae=mae / value_range,
            ))
    return rows


def score_binary(run: dict) -> list[BinaryScore]:
    rows: list[BinaryScore] = []
    for question in run["questions"]:
        if question.get("question_type") != "binary":
            continue
        gt = question.get("ground_truth")
        if gt is None:
            continue
        outcome = int(bool(gt))
        turn, horizon = infer_horizon(question["question_text"])
        for model_id, prediction in question["predictions"].items():
            if not prediction or prediction.get("error"):
                continue
            prob = prediction.get("probability")
            if prob is None:
                continue
            try:
                p = float(prob)
            except (TypeError, ValueError):
                continue
            rows.append(BinaryScore(
                model=model_id,
                question_id=question["question_id"],
                template_id=question.get("template_id", ""),
                horizon=horizon,
                turn=turn,
                brier=brier(p, outcome),
            ))
    return rows


# --------------------------------------------------------------------------
# Continuous unconditional with Opus spliced in
# --------------------------------------------------------------------------

def score_continuous_with_opus() -> list[ContinuousScore]:
    """Continuous scores from the main 30-model run, with Opus 4.5 spliced
    in from the archived 10-model run.

    The archived `cont_uncond_all.anthropic_retry.json` contains the exact
    same 2310 continuous questions as the main run (verified at smoke-test
    time), just with a different model set. Opus is in the archive but not
    in the main run, so we union the two: every model gets one score row
    per (question, model) prediction.
    """
    main_run = load_run(RUN_NONH0)
    main_scores = score_continuous(main_run)
    main_seen = {(s.model, s.question_id) for s in main_scores}

    if not ARCHIVE_CONT_UNCOND.exists():
        return main_scores

    archive_run = load_run(ARCHIVE_CONT_UNCOND)
    archive_scores = score_continuous(archive_run)
    extras = [s for s in archive_scores if (s.model, s.question_id) not in main_seen]
    return main_scores + extras

def load_h0_continuous_scores() -> list[ContinuousScore]:
    """Load H0 continuous scores produced by run_h0_continuous_model_eval.py.

    The H0 runner writes h0_forecast_scores.csv and h0_results.json. We rely
    on h0_results.json so we can recompute scores under the same code path.
    Returns an empty list if commit 3.5 has not landed yet.
    """
    if not H0_CONT_RESULTS.exists():
        return []
    with H0_CONT_RESULTS.open("r", encoding="utf-8") as f:
        results = json.load(f)
    rows: list[ContinuousScore] = []
    for question in results.get("questions", []):
        # H0 results use template_id="h0_<source>" and carry the original
        # source_template_id alongside for metric lookup.
        source_template = question.get("source_template_id") or question.get("template_id", "").removeprefix("h0_")
        meta = CONTINUOUS_TEMPLATES.get(source_template)
        if meta is None:
            continue
        gt = float(question["ground_truth"])
        for model_id, prediction in question["predictions"].items():
            pcts = valid_percentiles(prediction)
            if pcts is None:
                continue
            crps = quantile_crps(pcts, gt)
            mae = abs(pcts["p50"] - gt)
            rows.append(ContinuousScore(
                model=model_id,
                question_id=question["question_id"],
                template_id=source_template,
                family=meta["family"],
                horizon="H0",
                turn=question.get("resolution_turn", 60),
                crps=crps,
                mae=mae,
                normalized_crps=crps / meta["value_range"],
                normalized_mae=mae / meta["value_range"],
            ))
    return rows


# --------------------------------------------------------------------------
# Aggregations
# --------------------------------------------------------------------------

def mean_normalized_crps_by_model(scores: Iterable[ContinuousScore]) -> dict[str, float]:
    by_model: dict[str, list[float]] = defaultdict(list)
    for s in scores:
        by_model[s.model].append(s.normalized_crps)
    return {m: mean(v) for m, v in by_model.items() if v}


def mean_brier_by_model(scores: Iterable[BinaryScore]) -> dict[str, float]:
    by_model: dict[str, list[float]] = defaultdict(list)
    for s in scores:
        by_model[s.model].append(s.brier)
    return {m: mean(v) for m, v in by_model.items() if v}


def mean_normalized_crps_by_model_horizon(scores: Iterable[ContinuousScore]) -> dict[tuple[str, str], float]:
    by_key: dict[tuple[str, str], list[float]] = defaultdict(list)
    for s in scores:
        if s.horizon is None:
            continue
        by_key[(s.model, s.horizon)].append(s.normalized_crps)
    return {k: mean(v) for k, v in by_key.items() if v}


def mean_brier_by_model_horizon(scores: Iterable[BinaryScore]) -> dict[tuple[str, str], float]:
    by_key: dict[tuple[str, str], list[float]] = defaultdict(list)
    for s in scores:
        if s.horizon is None:
            continue
        by_key[(s.model, s.horizon)].append(s.brier)
    return {k: mean(v) for k, v in by_key.items() if v}


def rank_models(score_by_model: dict[str, float], lower_is_better: bool = True) -> list[tuple[int, str, float]]:
    """Return [(rank_1_based, model_id, score), ...] sorted best to worst."""
    items = sorted(score_by_model.items(), key=lambda kv: kv[1], reverse=not lower_is_better)
    return [(i + 1, model, score) for i, (model, score) in enumerate(items)]


# --------------------------------------------------------------------------
# Sanity check helpers
# --------------------------------------------------------------------------

def assert_finite(value: float, label: str) -> None:
    if not math.isfinite(value):
        raise ValueError(f"Non-finite value for {label}: {value}")
