#!/usr/bin/env python3
"""Score the anonymized CivBench human baseline pilot.

The human pilot collected five-bin probability distributions for continuous
questions. This script scores those responses in two complementary ways:

1. Convert each histogram to p10/p25/p50/p75/p90 quantiles and use the same
   quantile CRPS approximation used for model continuous forecasts.
2. Score the native five-bin forecast with ranked probability score (RPS).

Outputs are written under data/human_baseline/analysis by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Callable, Iterable


PROB_COLUMNS = [
    "prob_bin_1_pct",
    "prob_bin_2_pct",
    "prob_bin_3_pct",
    "prob_bin_4_pct",
    "prob_bin_5_pct",
]

QUANTILE_LEVELS = {
    "p10": 0.10,
    "p25": 0.25,
    "p50": 0.50,
    "p75": 0.75,
    "p90": 0.90,
}


@dataclass(frozen=True)
class ForecastRow:
    participant_id: str
    question_order: int
    question_id: str
    world_id: str
    civilization: str
    template: str
    horizon: str
    resolution_turn: int
    ground_truth: float
    bin_min: float
    bin_max: float
    bin_unit: str
    probabilities_pct: tuple[float, float, float, float, float]
    prob_sum_pct: float
    response_time_ms: int

    @property
    def value_range(self) -> float:
        return self.bin_max - self.bin_min


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/human_baseline"),
        help="Directory containing pilot_responses_clean.csv and questions.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/human_baseline/analysis"),
        help="Directory for generated analysis tables.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=10000,
        help="Number of bootstrap resamples for confidence intervals.",
    )
    parser.add_argument("--seed", type=int, default=20260429)
    return parser.parse_args()


def read_forecasts(path: Path) -> list[ForecastRow]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = []
        for row in csv.DictReader(f):
            probs = tuple(float(row[col]) for col in PROB_COLUMNS)
            rows.append(
                ForecastRow(
                    participant_id=row["participant_id"],
                    question_order=int(row["question_order"]),
                    question_id=row["question_id"],
                    world_id=row["world_id"],
                    civilization=row["civilization"],
                    template=row["template"],
                    horizon=row["horizon"],
                    resolution_turn=int(row["resolution_turn"]),
                    ground_truth=float(row["ground_truth"]),
                    bin_min=float(row["bin_min"]),
                    bin_max=float(row["bin_max"]),
                    bin_unit=row["bin_unit"],
                    probabilities_pct=probs,
                    prob_sum_pct=float(row["prob_sum_pct"]),
                    response_time_ms=int(float(row["response_time_ms"])),
                )
            )
    return rows


def read_questions(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def normalize_probs(probabilities_pct: Iterable[float]) -> list[float]:
    total = sum(probabilities_pct)
    if total <= 0:
        return [1.0 / len(PROB_COLUMNS)] * len(PROB_COLUMNS)
    return [p / total for p in probabilities_pct]


def observed_bin(row: ForecastRow) -> int:
    width = row.value_range / len(PROB_COLUMNS)
    if width <= 0:
        raise ValueError(f"Non-positive bin range for {row.question_id}")
    idx = int((row.ground_truth - row.bin_min) / width)
    return min(max(idx, 0), len(PROB_COLUMNS) - 1)


def histogram_quantiles(row: ForecastRow, probs: list[float]) -> dict[str, float]:
    """Convert a five-bin histogram to quantiles assuming uniform mass per bin."""
    width = row.value_range / len(probs)
    quantiles = {}
    for key, tau in QUANTILE_LEVELS.items():
        cumulative = 0.0
        value = row.bin_max
        for idx, prob in enumerate(probs):
            next_cumulative = cumulative + prob
            if tau <= next_cumulative or idx == len(probs) - 1:
                if prob <= 0:
                    frac = 0.5
                else:
                    frac = (tau - cumulative) / prob
                frac = min(max(frac, 0.0), 1.0)
                value = row.bin_min + (idx + frac) * width
                break
            cumulative = next_cumulative
        quantiles[key] = value
    return quantiles


def quantile_crps(percentiles: dict[str, float], true_value: float) -> float:
    total_loss = 0.0
    for key, tau in QUANTILE_LEVELS.items():
        residual = true_value - percentiles[key]
        if residual >= 0:
            total_loss += tau * residual
        else:
            total_loss += (tau - 1) * residual
    return (2 / len(QUANTILE_LEVELS)) * total_loss


def ranked_probability_score(probs: list[float], obs_bin: int) -> float:
    total = 0.0
    cumulative = 0.0
    for idx in range(len(probs) - 1):
        cumulative += probs[idx]
        observed_cdf = 1.0 if obs_bin <= idx else 0.0
        total += (cumulative - observed_cdf) ** 2
    return total / (len(probs) - 1)


def histogram_mean(row: ForecastRow, probs: list[float]) -> float:
    width = row.value_range / len(probs)
    midpoints = [row.bin_min + (idx + 0.5) * width for idx in range(len(probs))]
    return sum(p * x for p, x in zip(probs, midpoints))


def entropy(probs: list[float]) -> float:
    if not probs:
        return float("nan")
    raw = -sum(p * math.log(p) for p in probs if p > 0)
    return raw / math.log(len(probs))


def score_row(row: ForecastRow, source: str = "human_individual") -> dict:
    probs = normalize_probs(row.probabilities_pct)
    obs = observed_bin(row)
    quantiles = histogram_quantiles(row, probs)
    crps = quantile_crps(quantiles, row.ground_truth)
    mae = abs(quantiles["p50"] - row.ground_truth)
    rps = ranked_probability_score(probs, obs)
    value_range = row.value_range
    out = {
        "source": source,
        "n_component_participants": 1 if source == "human_individual" else 0,
        "participant_id": row.participant_id,
        "question_order": row.question_order,
        "question_id": row.question_id,
        "world_id": row.world_id,
        "civilization": row.civilization,
        "template": row.template,
        "horizon": row.horizon,
        "resolution_turn": row.resolution_turn,
        "ground_truth": row.ground_truth,
        "bin_min": row.bin_min,
        "bin_max": row.bin_max,
        "bin_unit": row.bin_unit,
        "observed_bin": obs + 1,
        "prob_sum_pct": row.prob_sum_pct,
        "prob_sum_abs_error_pct": abs(row.prob_sum_pct - 100.0),
        "response_time_ms": row.response_time_ms,
        "mean_prediction": histogram_mean(row, probs),
        "p10": quantiles["p10"],
        "p25": quantiles["p25"],
        "p50": quantiles["p50"],
        "p75": quantiles["p75"],
        "p90": quantiles["p90"],
        "crps": crps,
        "mae": mae,
        "normalized_crps": crps / value_range,
        "normalized_mae": mae / value_range,
        "rps": rps,
        "prob_true_bin": probs[obs],
        "entropy": entropy(probs),
    }
    for idx, prob in enumerate(probs, start=1):
        out[f"prob_bin_{idx}"] = prob
    return out


def score_uniform(row: ForecastRow) -> dict:
    uniform = ForecastRow(
        participant_id="uniform",
        question_order=row.question_order,
        question_id=row.question_id,
        world_id=row.world_id,
        civilization=row.civilization,
        template=row.template,
        horizon=row.horizon,
        resolution_turn=row.resolution_turn,
        ground_truth=row.ground_truth,
        bin_min=row.bin_min,
        bin_max=row.bin_max,
        bin_unit=row.bin_unit,
        probabilities_pct=(20.0, 20.0, 20.0, 20.0, 20.0),
        prob_sum_pct=100.0,
        response_time_ms=0,
    )
    out = score_row(uniform, source="uniform_bins")
    out["n_component_participants"] = 0
    return out


def score_crowd(question_rows: list[ForecastRow]) -> dict:
    first = question_rows[0]
    avg_probs = tuple(mean(row.probabilities_pct[idx] for row in question_rows) for idx in range(5))
    crowd = ForecastRow(
        participant_id="crowd_mean",
        question_order=0,
        question_id=first.question_id,
        world_id=first.world_id,
        civilization=first.civilization,
        template=first.template,
        horizon=first.horizon,
        resolution_turn=first.resolution_turn,
        ground_truth=first.ground_truth,
        bin_min=first.bin_min,
        bin_max=first.bin_max,
        bin_unit=first.bin_unit,
        probabilities_pct=avg_probs,
        prob_sum_pct=sum(avg_probs),
        response_time_ms=0,
    )
    out = score_row(crowd, source="human_crowd_mean")
    out["n_component_participants"] = len({row.participant_id for row in question_rows})
    return out


def rows_by(rows: Iterable[dict], key: str) -> dict[str, list[dict]]:
    groups = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    return dict(groups)


def summarize_scores(rows: list[dict], group_name: str, group_value: str, source: str) -> dict:
    if not rows:
        raise ValueError("Cannot summarize empty rows")
    if source == "human_crowd_mean":
        n_participants = max(row.get("n_component_participants", 0) for row in rows)
    elif source == "uniform_bins":
        n_participants = 0
    else:
        n_participants = len({row["participant_id"] for row in rows})
    return {
        "source": source,
        "group": group_name,
        "value": group_value,
        "n_forecasts": len(rows),
        "n_participants": n_participants,
        "n_questions": len({row["question_id"] for row in rows}),
        "crps": mean(row["crps"] for row in rows),
        "mae": mean(row["mae"] for row in rows),
        "normalized_crps": mean(row["normalized_crps"] for row in rows),
        "normalized_mae": mean(row["normalized_mae"] for row in rows),
        "rps": mean(row["rps"] for row in rows),
        "prob_true_bin": mean(row["prob_true_bin"] for row in rows),
        "entropy": mean(row["entropy"] for row in rows),
        "median_response_time_ms": median(row["response_time_ms"] for row in rows if row["response_time_ms"] > 0)
        if any(row["response_time_ms"] > 0 for row in rows)
        else 0,
    }


def aggregate_summaries(
    individual_scores: list[dict],
    crowd_scores: list[dict],
    uniform_scores: list[dict],
) -> list[dict]:
    outputs = []
    for source, rows in [
        ("human_individual", individual_scores),
        ("human_crowd_mean", crowd_scores),
        ("uniform_bins", uniform_scores),
    ]:
        outputs.append(summarize_scores(rows, "overall", "all", source))
        for key in ["template", "horizon", "world_id"]:
            for value, group_rows in sorted(rows_by(rows, key).items()):
                outputs.append(summarize_scores(group_rows, key, value, source))
    return outputs


def participant_summaries(individual_scores: list[dict]) -> list[dict]:
    outputs = []
    for participant_id, rows in sorted(rows_by(individual_scores, "participant_id").items()):
        summary = summarize_scores(rows, "participant_id", participant_id, "human_individual")
        summary["total_response_time_min"] = sum(row["response_time_ms"] for row in rows) / 60000
        summary["mean_prob_sum_abs_error_pct"] = mean(row["prob_sum_abs_error_pct"] for row in rows)
        outputs.append(summary)
    return outputs


def question_summaries(
    individual_scores: list[dict],
    crowd_scores: list[dict],
    uniform_scores: list[dict],
) -> list[dict]:
    by_question = rows_by(individual_scores, "question_id")
    crowd_by_question = {row["question_id"]: row for row in crowd_scores}
    uniform_by_question = {row["question_id"]: row for row in uniform_scores}
    outputs = []
    for question_id, rows in sorted(by_question.items()):
        first = rows[0]
        crowd = crowd_by_question[question_id]
        uniform = uniform_by_question[question_id]
        out = {
            "question_id": question_id,
            "world_id": first["world_id"],
            "civilization": first["civilization"],
            "template": first["template"],
            "horizon": first["horizon"],
            "resolution_turn": first["resolution_turn"],
            "ground_truth": first["ground_truth"],
            "observed_bin": first["observed_bin"],
            "n_participants": len(rows),
            "mean_individual_crps": mean(row["crps"] for row in rows),
            "mean_individual_mae": mean(row["mae"] for row in rows),
            "mean_individual_normalized_crps": mean(row["normalized_crps"] for row in rows),
            "mean_individual_normalized_mae": mean(row["normalized_mae"] for row in rows),
            "mean_individual_rps": mean(row["rps"] for row in rows),
            "crowd_crps": crowd["crps"],
            "crowd_mae": crowd["mae"],
            "crowd_normalized_crps": crowd["normalized_crps"],
            "crowd_normalized_mae": crowd["normalized_mae"],
            "crowd_rps": crowd["rps"],
            "uniform_crps": uniform["crps"],
            "uniform_mae": uniform["mae"],
            "uniform_normalized_crps": uniform["normalized_crps"],
            "uniform_normalized_mae": uniform["normalized_mae"],
            "uniform_rps": uniform["rps"],
            "crowd_minus_uniform_normalized_crps": crowd["normalized_crps"] - uniform["normalized_crps"],
            "mean_individual_minus_uniform_normalized_crps": mean(row["normalized_crps"] for row in rows)
            - uniform["normalized_crps"],
        }
        for idx in range(1, 6):
            out[f"crowd_prob_bin_{idx}"] = crowd[f"prob_bin_{idx}"]
        for key in ["p10", "p25", "p50", "p75", "p90"]:
            out[f"crowd_{key}"] = crowd[key]
        outputs.append(out)
    return outputs


def quality_checks(forecasts: list[ForecastRow], questions: list[dict]) -> dict:
    participants = sorted({row.participant_id for row in forecasts})
    question_ids = sorted({row.question_id for row in forecasts})
    prob_errors = [abs(row.prob_sum_pct - 100.0) for row in forecasts]
    response_times = [row.response_time_ms for row in forecasts]
    per_participant_counts = {
        participant: sum(1 for row in forecasts if row.participant_id == participant)
        for participant in participants
    }
    return {
        "n_participants": len(participants),
        "n_questions": len(question_ids),
        "n_question_manifest_rows": len(questions),
        "n_forecasts": len(forecasts),
        "per_participant_forecast_counts": per_participant_counts,
        "prob_sum_pct": {
            "min": min(row.prob_sum_pct for row in forecasts),
            "median": median(row.prob_sum_pct for row in forecasts),
            "max": max(row.prob_sum_pct for row in forecasts),
            "mean_abs_error_from_100": mean(prob_errors),
            "max_abs_error_from_100": max(prob_errors),
        },
        "response_time_ms": {
            "min": min(response_times),
            "median": median(response_times),
            "max": max(response_times),
            "mean": mean(response_times),
        },
        "worlds": sorted({row.world_id for row in forecasts}),
        "templates": sorted({row.template for row in forecasts}),
        "horizons": sorted({row.horizon for row in forecasts}),
    }


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = (len(ordered) - 1) * pct
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return ordered[int(idx)]
    frac = idx - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def bootstrap_ci(
    items: list,
    metric: Callable[[list], float],
    samples: int,
    rng: random.Random,
) -> tuple[float, float]:
    if not items:
        return (float("nan"), float("nan"))
    values = []
    for _ in range(samples):
        draw = [items[rng.randrange(len(items))] for _ in items]
        values.append(metric(draw))
    return percentile(values, 0.025), percentile(values, 0.975)


def bootstrap_summaries(
    individual_scores: list[dict],
    question_summary_rows: list[dict],
    samples: int,
    seed: int,
) -> list[dict]:
    rng = random.Random(seed)
    by_participant = list(rows_by(individual_scores, "participant_id").values())

    def flatten(groups: list[list[dict]]) -> list[dict]:
        return [row for group in groups for row in group]

    def mean_metric(rows: list[dict], key: str) -> float:
        return mean(row[key] for row in rows)

    outputs = []
    participant_metrics = [
        ("human_individual", "normalized_crps", lambda groups: mean_metric(flatten(groups), "normalized_crps")),
        ("human_individual", "normalized_mae", lambda groups: mean_metric(flatten(groups), "normalized_mae")),
        ("human_individual", "rps", lambda groups: mean_metric(flatten(groups), "rps")),
    ]
    for source, metric_name, metric_fn in participant_metrics:
        point = metric_fn(by_participant)
        lo, hi = bootstrap_ci(by_participant, metric_fn, samples, rng)
        outputs.append(
            {
                "source": source,
                "resample_unit": "participant",
                "group": "overall",
                "value": "all",
                "metric": metric_name,
                "point_estimate": point,
                "ci_low": lo,
                "ci_high": hi,
                "bootstrap_samples": samples,
            }
        )

    question_metrics = [
        ("human_crowd_mean", "normalized_crps", "crowd_normalized_crps"),
        ("human_crowd_mean", "normalized_mae", "crowd_normalized_mae"),
        ("human_crowd_mean", "rps", "crowd_rps"),
        ("uniform_bins", "normalized_crps", "uniform_normalized_crps"),
        ("uniform_bins", "normalized_mae", "uniform_normalized_mae"),
        ("uniform_bins", "rps", "uniform_rps"),
        ("human_crowd_mean_minus_uniform", "normalized_crps", "crowd_minus_uniform_normalized_crps"),
    ]
    for source, metric_name, key in question_metrics:
        metric_fn = lambda rows, k=key: mean(row[k] for row in rows)
        point = metric_fn(question_summary_rows)
        lo, hi = bootstrap_ci(question_summary_rows, metric_fn, samples, rng)
        outputs.append(
            {
                "source": source,
                "resample_unit": "question",
                "group": "overall",
                "value": "all",
                "metric": metric_name,
                "point_estimate": point,
                "ci_low": lo,
                "ci_high": hi,
                "bootstrap_samples": samples,
            }
        )
    return outputs


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_readme(path: Path) -> None:
    path.write_text(
        """# Human Baseline Analysis

Generated by `scripts/analyze_human_baseline.py` from the anonymized files in `data/human_baseline`.

## Files

- `forecast_scores.csv`: one row per individual human forecast, including five-bin probabilities normalized to fractions, derived quantiles, CRPS, MAE, normalized scores, RPS, and response-time fields.
- `participant_summary.csv`: one row per anonymized participant, useful for checking whether aggregate performance is driven by one respondent.
- `question_summary.csv`: one row per question, including mean individual performance, crowd-mean performance, uniform-bin baseline performance, and the crowd-average forecast.
- `aggregate_summary.csv`: overall, template, horizon, and world breakdowns for individual humans, the crowd mean, and the uniform-bin baseline.
- `bootstrap_summary.csv`: simple 95% bootstrap intervals for the main human-baseline quantities. Individual-human intervals resample participants; crowd and uniform intervals resample questions.
- `quality_checks.json`: counts, probability-sum diagnostics, response-time diagnostics, and task coverage.

For `human_crowd_mean`, `n_participants` is the number of humans aggregated into each crowd forecast. For `uniform_bins`, `n_participants` is zero because it is a synthetic baseline.

## Scoring

The survey collected probability mass over five equal-width bins. For comparison with CivBench model outputs, the analysis converts each histogram into p10/p25/p50/p75/p90 quantiles assuming uniform mass within each bin, then computes the same quantile-pinball CRPS approximation used by the model evaluator. MAE is the absolute error of the derived p50.

Because gold, city count, and technology count have different numeric ranges, normalized CRPS and normalized MAE divide by each question's forecast range. These normalized metrics are the recommended headline scores for comparing across templates.

RPS is included as a native ordinal-bin scoring rule for the human survey format. Lower is better for all reported error metrics.
""",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    forecasts = read_forecasts(args.input_dir / "pilot_responses_clean.csv")
    questions = read_questions(args.input_dir / "questions.csv")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    individual_scores = [score_row(row) for row in forecasts]

    dataclass_groups = defaultdict(list)
    for row in forecasts:
        dataclass_groups[row.question_id].append(row)

    crowd_scores = [score_crowd(rows) for _, rows in sorted(dataclass_groups.items())]
    representative_rows = [rows[0] for _, rows in sorted(dataclass_groups.items())]
    uniform_scores = [score_uniform(row) for row in representative_rows]

    aggregate_rows = aggregate_summaries(individual_scores, crowd_scores, uniform_scores)
    participant_rows = participant_summaries(individual_scores)
    question_rows = question_summaries(individual_scores, crowd_scores, uniform_scores)
    bootstrap_rows = bootstrap_summaries(
        individual_scores,
        question_rows,
        samples=args.bootstrap_samples,
        seed=args.seed,
    )
    checks = quality_checks(forecasts, questions)

    write_csv(args.output_dir / "forecast_scores.csv", individual_scores)
    write_csv(args.output_dir / "participant_summary.csv", participant_rows)
    write_csv(args.output_dir / "question_summary.csv", question_rows)
    write_csv(args.output_dir / "aggregate_summary.csv", aggregate_rows)
    write_csv(args.output_dir / "bootstrap_summary.csv", bootstrap_rows)
    (args.output_dir / "quality_checks.json").write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")
    write_readme(args.output_dir / "README.md")

    print(json.dumps(checks, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
