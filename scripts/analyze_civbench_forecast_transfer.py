#!/usr/bin/env python3
"""
Analyze CivBench -> ForecastBench transfer with world/model holdouts.

Computes:
1) H1 model ranking stability across worlds and grouped worlds.
2) H1 template-horizon difficulty stability across worlds and grouped worlds.
3) H2 transfer methods evaluated on identical world/model holdouts:
   - Path 3 scalar baseline: FB = a + b * CivBench_Brier
   - Path 1 pair weighting: linear ridge over (horizon, template) features
   - Path 2 grouped pair weighting: Path 1 with grouped-world feature averaging
   - Fixed world split (default: train seed1..seed8, test seed9..seed10)
   - Exhaustive 8/2 world splits
   - Leave-one-world-out
   - 1-world train -> 1-world test matrix
   - Optional model holdout (leave-one-model-out) within each world split
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


CIVBENCH_TO_FORECASTBENCH = {
    "openai/gpt-3.5-turbo-0125": "GPT-3.5-Turbo-0125",
    "openai/gpt-4o": "GPT-4o-2024-05-13",
    "openai/gpt-4.1-2025-04-14": "GPT-4.1-2025-04-14",
    "openai/gpt-5-2025-08-07": "GPT-5-2025-08-07",
    "openai/gpt-5-mini-2025-08-07": "GPT-5-Mini-2025-08-07",
    "openai/gpt-5-nano-2025-08-07": "GPT-5-Nano-2025-08-07",
    "openai/gpt-5.1-2025-11-13": "GPT-5.1-2025-11-13",
    "openai/o3-2025-04-16": "O3-2025-04-16",
    "openai/o3-mini-2025-01-31": "O3-Mini-2025-01-31",
    "openai/o4-mini-2025-04-16": "O4-Mini-2025-04-16",
    "anthropic/claude-3-haiku-20240307": "Claude-3-Haiku-20240307",
    "anthropic/claude-haiku-4-5-20251001": "Claude-Haiku-4-5-20251001",
    "google/gemini-2.0-flash-lite-001": "Gemini-2.0-Flash-Lite-001",
    "google/gemini-2.5-flash": "Gemini-2.5-Flash",
    "xai/grok-4-fast-reasoning": "Grok-4-Fast-Reasoning",
    "xai/grok-4-1-fast-reasoning": "Grok-4-1-Fast-Reasoning",
    "mistral/mistral-large-2411": "Mistral-Large-2411",
    "mistral/mistral-large-latest": "Mistral-Large-Latest",
    "anthropic/claude-opus-4-6": "Claude-Opus-4-6",
}


@dataclass
class EvalAgg:
    seeds: list[str]
    models: list[str]
    pairs: list[tuple[str, str]]
    questions_per_seed: dict[str, int]
    sum_seed_model: dict[tuple[str, str], float]
    cnt_seed_model: dict[tuple[str, str], int]
    sum_seed_model_pair: dict[tuple[str, str, tuple[str, str]], float]
    cnt_seed_model_pair: dict[tuple[str, str, tuple[str, str]], int]
    sum_seed_pair: dict[tuple[str, tuple[str, str]], float]
    cnt_seed_pair: dict[tuple[str, tuple[str, str]], int]
    n_questions_non_h0: int
    n_questions_total: int


def seed_sort_key(seed: str) -> tuple[int, int | str]:
    s = str(seed)
    m = re.fullmatch(r"seed(\d+)", s, flags=re.IGNORECASE)
    if m:
        return 0, int(m.group(1))
    m = re.search(r"(\d+)$", s)
    if m:
        return 1, int(m.group(1))
    return 2, s


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    rho = stats.spearmanr(x, y).statistic
    return float(rho) if (rho is not None and np.isfinite(rho)) else math.nan


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return math.nan
    rho = stats.pearsonr(x, y).statistic
    return float(rho) if np.isfinite(rho) else math.nan


def parse_metric_with_n(value: str) -> float | None:
    m = re.match(r"([\d.]+)\s*\(([\d,]+)\)", str(value).strip())
    if not m:
        return None
    return float(m.group(1))


def summarize(values: list[float]) -> dict[str, float]:
    arr = np.array([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return {
            "n": 0,
            "min": math.nan,
            "p10": math.nan,
            "median": math.nan,
            "mean": math.nan,
            "max": math.nan,
        }
    return {
        "n": int(arr.size),
        "min": float(np.min(arr)),
        "p10": float(np.percentile(arr, 10)),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "max": float(np.max(arr)),
    }


def fit_linear_mapping(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    X = np.column_stack([np.ones_like(x), x])
    beta = np.linalg.pinv(X) @ y
    return beta


def predict_linear_mapping(beta: np.ndarray, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return beta[0] + beta[1] * x


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "spearman": safe_spearman(y_true, y_pred),
        "pearson": safe_pearson(y_true, y_pred),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
    }


def parse_seed_csv(csv_val: str | None) -> list[str]:
    if not csv_val:
        return []
    return [s.strip() for s in csv_val.split(",") if s.strip()]


def load_forecastbench_dataset_scores(path: Path) -> dict[str, float]:
    df = pd.read_csv(path)
    df = df[df["Model"].astype(str).str.contains(r"\(zero shot\)", case=False, regex=True)]
    out: dict[str, float] = {}
    for _, row in df.iterrows():
        raw = str(row["Model"]).strip()
        base = re.sub(r"\s*\(zero shot\)\s*$", "", raw, flags=re.IGNORECASE).strip().lower()
        score = parse_metric_with_n(row.get("Dataset (N)", ""))
        if score is not None:
            out[base] = score
    return out


def load_eval_aggregates(eval_json_path: Path, include_h0: bool) -> EvalAgg:
    data = json.loads(eval_json_path.read_text())
    questions = data.get("questions", [])
    metadata_models = data.get("metadata", {}).get("models", [])
    if metadata_models:
        models = sorted(str(m) for m in metadata_models)
    else:
        models = sorted(
            {
                str(model_id)
                for q in questions
                for model_id in q.get("predictions", {}).keys()
            }
        )

    sum_seed_model: dict[tuple[str, str], float] = defaultdict(float)
    cnt_seed_model: dict[tuple[str, str], int] = defaultdict(int)
    sum_seed_model_pair: dict[tuple[str, str, tuple[str, str]], float] = defaultdict(float)
    cnt_seed_model_pair: dict[tuple[str, str, tuple[str, str]], int] = defaultdict(int)
    sum_seed_pair: dict[tuple[str, tuple[str, str]], float] = defaultdict(float)
    cnt_seed_pair: dict[tuple[str, tuple[str, str]], int] = defaultdict(int)
    questions_per_seed: dict[str, int] = defaultdict(int)
    seeds_set: set[str] = set()
    pairs_set: set[tuple[str, str]] = set()

    n_total = len(questions)
    n_non_h0 = 0

    for q in questions:
        template_id = str(q.get("template_id", ""))
        if not include_h0 and template_id.startswith("h0_"):
            continue
        n_non_h0 += 1

        seed_raw = q.get("seed")
        if seed_raw in (None, ""):
            seed_raw = q.get("game_id")
        seed = str(seed_raw) if seed_raw not in (None, "") else ""
        if not seed:
            continue

        gt = q.get("ground_truth")
        if gt is None:
            continue
        gt_val = 1.0 if bool(gt) else 0.0

        horizon = str(q.get("horizon", ""))
        pair = (horizon, template_id)

        seeds_set.add(seed)
        pairs_set.add(pair)
        questions_per_seed[seed] += 1

        preds = q.get("predictions", {})
        for model, pred in preds.items():
            if pred is None:
                continue
            brier_raw = pred.get("brier_score")
            if brier_raw is None:
                prob = pred.get("probability")
                if prob is None:
                    continue
                brier = (float(prob) - gt_val) ** 2
            else:
                brier = float(brier_raw)
            if not np.isfinite(brier):
                continue

            sum_seed_model[(seed, str(model))] += brier
            cnt_seed_model[(seed, str(model))] += 1
            sum_seed_model_pair[(seed, str(model), pair)] += brier
            cnt_seed_model_pair[(seed, str(model), pair)] += 1

            sum_seed_pair[(seed, pair)] += brier
            cnt_seed_pair[(seed, pair)] += 1

    seeds = sorted(seeds_set, key=seed_sort_key)
    pairs = sorted(pairs_set, key=lambda x: (x[0], x[1]))

    return EvalAgg(
        seeds=seeds,
        models=models,
        pairs=pairs,
        questions_per_seed=dict(questions_per_seed),
        sum_seed_model=sum_seed_model,
        cnt_seed_model=cnt_seed_model,
        sum_seed_model_pair=sum_seed_model_pair,
        cnt_seed_model_pair=cnt_seed_model_pair,
        sum_seed_pair=sum_seed_pair,
        cnt_seed_pair=cnt_seed_pair,
        n_questions_non_h0=n_non_h0,
        n_questions_total=n_total,
    )


def compute_model_matrix(agg: EvalAgg) -> np.ndarray:
    S = len(agg.seeds)
    M = len(agg.models)
    out = np.full((S, M), np.nan, dtype=float)
    for i, seed in enumerate(agg.seeds):
        for j, model in enumerate(agg.models):
            c = agg.cnt_seed_model.get((seed, model), 0)
            if c > 0:
                out[i, j] = agg.sum_seed_model[(seed, model)] / c
    return out


def compute_pair_difficulty_matrix(agg: EvalAgg) -> tuple[np.ndarray, list[tuple[str, str]]]:
    complete_pairs = [
        pair
        for pair in agg.pairs
        if all(agg.cnt_seed_pair.get((seed, pair), 0) > 0 for seed in agg.seeds)
    ]
    S = len(agg.seeds)
    P = len(complete_pairs)
    out = np.full((S, P), np.nan, dtype=float)
    for i, seed in enumerate(agg.seeds):
        for k, pair in enumerate(complete_pairs):
            c = agg.cnt_seed_pair[(seed, pair)]
            out[i, k] = agg.sum_seed_pair[(seed, pair)] / c
    return out, complete_pairs


def model_vector_for_group(agg: EvalAgg, group: tuple[str, ...]) -> np.ndarray:
    v = np.full(len(agg.models), np.nan, dtype=float)
    for j, model in enumerate(agg.models):
        total = 0.0
        count = 0
        for seed in group:
            c = agg.cnt_seed_model.get((seed, model), 0)
            if c > 0:
                total += agg.sum_seed_model[(seed, model)]
                count += c
        if count > 0:
            v[j] = total / count
    return v


def pair_difficulty_vector_for_group(
    agg: EvalAgg, complete_pairs: list[tuple[str, str]], group: tuple[str, ...]
) -> np.ndarray:
    v = np.full(len(complete_pairs), np.nan, dtype=float)
    for k, pair in enumerate(complete_pairs):
        total = 0.0
        count = 0
        for seed in group:
            c = agg.cnt_seed_pair.get((seed, pair), 0)
            if c > 0:
                total += agg.sum_seed_pair[(seed, pair)]
                count += c
        if count > 0:
            v[k] = total / count
    return v


def summarize_group_stability(
    vectors_by_group: dict[tuple[str, ...], np.ndarray]
) -> list[tuple[str, list[float]]]:
    groups = list(vectors_by_group.keys())
    all_pairs: list[float] = []
    disjoint_pairs: list[float] = []

    for i, g1 in enumerate(groups):
        for j, g2 in enumerate(groups):
            if j <= i:
                continue
            rho = safe_spearman(vectors_by_group[g1], vectors_by_group[g2])
            if np.isfinite(rho):
                all_pairs.append(rho)
                if set(g1).isdisjoint(g2):
                    disjoint_pairs.append(rho)

    return [
        ("all_pairs", all_pairs),
        ("disjoint_pairs", disjoint_pairs),
    ]


def model_features_for_group(
    agg: EvalAgg, matched_models: list[str], group: tuple[str, ...]
) -> np.ndarray:
    x = np.full(len(matched_models), np.nan, dtype=float)
    for i, model in enumerate(matched_models):
        total = 0.0
        count = 0
        for seed in group:
            c = agg.cnt_seed_model.get((seed, model), 0)
            if c > 0:
                total += agg.sum_seed_model[(seed, model)]
                count += c
        if count > 0:
            x[i] = total / count
    return x


def complete_pairs_for_models(agg: EvalAgg, models: list[str]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for pair in agg.pairs:
        complete = True
        for seed in agg.seeds:
            for model in models:
                if agg.cnt_seed_model_pair.get((seed, model, pair), 0) <= 0:
                    complete = False
                    break
            if not complete:
                break
        if complete:
            out.append(pair)
    return out


def pair_feature_matrix_for_group(
    agg: EvalAgg,
    models: list[str],
    pairs: list[tuple[str, str]],
    group: tuple[str, ...],
) -> np.ndarray:
    X = np.full((len(models), len(pairs)), np.nan, dtype=float)
    for i, model in enumerate(models):
        for j, pair in enumerate(pairs):
            total = 0.0
            count = 0
            for seed in group:
                c = agg.cnt_seed_model_pair.get((seed, model, pair), 0)
                if c > 0:
                    total += agg.sum_seed_model_pair[(seed, model, pair)]
                    count += c
            if count > 0:
                X[i, j] = total / count
    return X


def world_groups(worlds: tuple[str, ...], group_size: int) -> list[tuple[str, ...]]:
    if group_size <= 0:
        raise ValueError("group_size must be positive.")
    if len(worlds) <= group_size:
        return [tuple(worlds)]
    return [tuple(g) for g in itertools.combinations(worlds, group_size)]


def fit_ridge_feature_model(X_train_raw: np.ndarray, y_train: np.ndarray, alpha: float) -> dict[str, np.ndarray]:
    X_train_raw = np.asarray(X_train_raw, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    if X_train_raw.ndim != 2:
        raise ValueError("X_train_raw must be 2D.")
    if X_train_raw.shape[0] != y_train.shape[0]:
        raise ValueError("X/y row mismatch.")
    if np.isnan(X_train_raw).any():
        raise ValueError("X_train_raw contains NaN.")
    if np.isnan(y_train).any():
        raise ValueError("y_train contains NaN.")

    mu = X_train_raw.mean(axis=0)
    sigma = X_train_raw.std(axis=0)
    sigma = np.where(sigma <= 1e-12, 1.0, sigma)
    X = (X_train_raw - mu) / sigma

    n_train, n_features = X.shape
    X1 = np.hstack([np.ones((n_train, 1)), X])
    reg = np.eye(n_features + 1)
    reg[0, 0] = 0.0
    beta = np.linalg.solve(X1.T @ X1 + float(alpha) * reg, X1.T @ y_train)
    return {"mu": mu, "sigma": sigma, "beta": beta}


def predict_ridge_feature_model(model: dict[str, np.ndarray], X_raw: np.ndarray) -> np.ndarray:
    X_raw = np.asarray(X_raw, dtype=float)
    X = (X_raw - model["mu"]) / model["sigma"]
    X1 = np.hstack([np.ones((X.shape[0], 1)), X])
    return X1 @ model["beta"]


def ridge_coefficients_original_scale(model: dict[str, np.ndarray]) -> tuple[float, np.ndarray]:
    beta = model["beta"]
    mu = model["mu"]
    sigma = model["sigma"]
    coef_std = beta[1:]
    coef = coef_std / sigma
    intercept = float(beta[0] - np.sum((mu / sigma) * coef_std))
    return intercept, coef


def run_path1_pair_ridge_eval(
    agg: EvalAgg,
    models: list[str],
    y_true: np.ndarray,
    pairs: list[tuple[str, str]],
    train_worlds: tuple[str, ...],
    test_worlds: tuple[str, ...],
    eval_mode: str,
    ridge_alpha: float,
) -> tuple[dict[str, float], pd.DataFrame]:
    X_train_all = pair_feature_matrix_for_group(agg, models, pairs, train_worlds)
    X_test_all = pair_feature_matrix_for_group(agg, models, pairs, test_worlds)

    if eval_mode == "same_models":
        model = fit_ridge_feature_model(X_train_all, y_true, alpha=ridge_alpha)
        y_pred = predict_ridge_feature_model(model, X_test_all)
        intercept, coef = ridge_coefficients_original_scale(model)
        metrics = regression_metrics(y_true=y_true, y_pred=y_pred)
        pred_df = pd.DataFrame(
            {
                "model": models,
                "target_fb_brier": y_true,
                "predicted_fb_brier": y_pred,
                "eval_mode": eval_mode,
                "fit_intercept": intercept,
                "fit_feature_l1_norm": float(np.sum(np.abs(coef))),
                "fit_feature_l2_norm": float(np.linalg.norm(coef)),
            }
        )
        return metrics, pred_df

    if eval_mode == "holdout_models":
        n_models = len(models)
        y_pred = np.full(n_models, np.nan, dtype=float)
        fit_intercepts = np.full(n_models, np.nan, dtype=float)
        fit_l1 = np.full(n_models, np.nan, dtype=float)
        fit_l2 = np.full(n_models, np.nan, dtype=float)
        for i in range(n_models):
            keep = np.ones(n_models, dtype=bool)
            keep[i] = False
            model = fit_ridge_feature_model(X_train_all[keep, :], y_true[keep], alpha=ridge_alpha)
            y_pred[i] = float(predict_ridge_feature_model(model, X_test_all[i : i + 1, :])[0])
            intercept, coef = ridge_coefficients_original_scale(model)
            fit_intercepts[i] = intercept
            fit_l1[i] = float(np.sum(np.abs(coef)))
            fit_l2[i] = float(np.linalg.norm(coef))
        metrics = regression_metrics(y_true=y_true, y_pred=y_pred)
        pred_df = pd.DataFrame(
            {
                "model": models,
                "target_fb_brier": y_true,
                "predicted_fb_brier": y_pred,
                "eval_mode": eval_mode,
                "fit_intercept": fit_intercepts,
                "fit_feature_l1_norm": fit_l1,
                "fit_feature_l2_norm": fit_l2,
            }
        )
        return metrics, pred_df

    raise ValueError(f"Unknown eval_mode: {eval_mode}")


def run_path2_grouped_pair_ridge_eval(
    agg: EvalAgg,
    models: list[str],
    y_true: np.ndarray,
    pairs: list[tuple[str, str]],
    train_worlds: tuple[str, ...],
    test_worlds: tuple[str, ...],
    eval_mode: str,
    ridge_alpha: float,
    group_size: int,
) -> tuple[dict[str, float], pd.DataFrame]:
    train_groups = world_groups(train_worlds, group_size=group_size)
    test_groups = world_groups(test_worlds, group_size=group_size)

    X_train_blocks = [pair_feature_matrix_for_group(agg, models, pairs, g) for g in train_groups]
    X_test_blocks = [pair_feature_matrix_for_group(agg, models, pairs, g) for g in test_groups]

    n_models = len(models)
    if eval_mode == "same_models":
        X_train = np.vstack(X_train_blocks)
        y_train = np.concatenate([y_true for _ in train_groups])
        model = fit_ridge_feature_model(X_train, y_train, alpha=ridge_alpha)

        pred_blocks = [predict_ridge_feature_model(model, Xb) for Xb in X_test_blocks]
        y_pred = np.mean(np.vstack(pred_blocks), axis=0)

        intercept, coef = ridge_coefficients_original_scale(model)
        metrics = regression_metrics(y_true=y_true, y_pred=y_pred)
        pred_df = pd.DataFrame(
            {
                "model": models,
                "target_fb_brier": y_true,
                "predicted_fb_brier": y_pred,
                "eval_mode": eval_mode,
                "n_train_groups": len(train_groups),
                "n_test_groups": len(test_groups),
                "fit_intercept": intercept,
                "fit_feature_l1_norm": float(np.sum(np.abs(coef))),
                "fit_feature_l2_norm": float(np.linalg.norm(coef)),
            }
        )
        return metrics, pred_df

    if eval_mode == "holdout_models":
        y_pred = np.full(n_models, np.nan, dtype=float)
        fit_intercepts = np.full(n_models, np.nan, dtype=float)
        fit_l1 = np.full(n_models, np.nan, dtype=float)
        fit_l2 = np.full(n_models, np.nan, dtype=float)

        for holdout_i in range(n_models):
            train_rows = []
            train_targets = []
            for Xb in X_train_blocks:
                keep = np.ones(n_models, dtype=bool)
                keep[holdout_i] = False
                train_rows.append(Xb[keep, :])
                train_targets.append(y_true[keep])
            X_train = np.vstack(train_rows)
            y_train = np.concatenate(train_targets)
            model = fit_ridge_feature_model(X_train, y_train, alpha=ridge_alpha)

            preds_i = []
            for Xtb in X_test_blocks:
                preds_i.append(float(predict_ridge_feature_model(model, Xtb[holdout_i : holdout_i + 1, :])[0]))
            y_pred[holdout_i] = float(np.mean(preds_i))

            intercept, coef = ridge_coefficients_original_scale(model)
            fit_intercepts[holdout_i] = intercept
            fit_l1[holdout_i] = float(np.sum(np.abs(coef)))
            fit_l2[holdout_i] = float(np.linalg.norm(coef))

        metrics = regression_metrics(y_true=y_true, y_pred=y_pred)
        pred_df = pd.DataFrame(
            {
                "model": models,
                "target_fb_brier": y_true,
                "predicted_fb_brier": y_pred,
                "eval_mode": eval_mode,
                "n_train_groups": len(train_groups),
                "n_test_groups": len(test_groups),
                "fit_intercept": fit_intercepts,
                "fit_feature_l1_norm": fit_l1,
                "fit_feature_l2_norm": fit_l2,
            }
        )
        return metrics, pred_df

    raise ValueError(f"Unknown eval_mode: {eval_mode}")


def run_all_h2_methods_for_split(
    agg: EvalAgg,
    models: list[str],
    y_true: np.ndarray,
    scalar_train_feature: np.ndarray,
    scalar_test_feature: np.ndarray,
    pair_feature_list: list[tuple[str, str]],
    train_worlds: tuple[str, ...],
    test_worlds: tuple[str, ...],
    scalar_eval_mode: str,
    path1_ridge_alpha: float,
    path2_group_size: int,
    path2_ridge_alpha: float,
) -> tuple[list[dict[str, object]], list[pd.DataFrame]]:
    metric_rows: list[dict[str, object]] = []
    pred_dfs: list[pd.DataFrame] = []

    # Baseline scalar path.
    m_scalar, pred_scalar = run_world_split_eval(
        x_train=scalar_train_feature,
        x_test=scalar_test_feature,
        y_true=y_true,
        model_names=models,
        eval_mode=scalar_eval_mode,
    )
    metric_rows.append(
        {
            "method": "path3_scalar_linear",
            "eval_mode": scalar_eval_mode,
            **m_scalar,
        }
    )
    pred_scalar = pred_scalar.copy()
    pred_scalar["method"] = "path3_scalar_linear"
    pred_dfs.append(pred_scalar)

    # Path 1: pair-feature ridge.
    m_path1, pred_path1 = run_path1_pair_ridge_eval(
        agg=agg,
        models=models,
        y_true=y_true,
        pairs=pair_feature_list,
        train_worlds=train_worlds,
        test_worlds=test_worlds,
        eval_mode=scalar_eval_mode,
        ridge_alpha=path1_ridge_alpha,
    )
    metric_rows.append(
        {
            "method": "path1_pair_ridge",
            "eval_mode": scalar_eval_mode,
            **m_path1,
        }
    )
    pred_path1 = pred_path1.copy()
    pred_path1["method"] = "path1_pair_ridge"
    pred_path1["ridge_alpha"] = float(path1_ridge_alpha)
    pred_path1["n_pairs"] = int(len(pair_feature_list))
    pred_dfs.append(pred_path1)

    # Path 2: grouped pair-feature ridge.
    m_path2, pred_path2 = run_path2_grouped_pair_ridge_eval(
        agg=agg,
        models=models,
        y_true=y_true,
        pairs=pair_feature_list,
        train_worlds=train_worlds,
        test_worlds=test_worlds,
        eval_mode=scalar_eval_mode,
        ridge_alpha=path2_ridge_alpha,
        group_size=path2_group_size,
    )
    method_name = f"path2_grouped_pair_ridge_n{path2_group_size}"
    metric_rows.append(
        {
            "method": method_name,
            "eval_mode": scalar_eval_mode,
            **m_path2,
        }
    )
    pred_path2 = pred_path2.copy()
    pred_path2["method"] = method_name
    pred_path2["ridge_alpha"] = float(path2_ridge_alpha)
    pred_path2["n_pairs"] = int(len(pair_feature_list))
    pred_path2["group_size_requested"] = int(path2_group_size)
    pred_dfs.append(pred_path2)

    return metric_rows, pred_dfs


def run_world_split_eval(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_true: np.ndarray,
    model_names: list[str],
    eval_mode: str,
) -> tuple[dict[str, float], pd.DataFrame]:
    if eval_mode == "same_models":
        beta = fit_linear_mapping(x_train, y_true)
        y_pred = predict_linear_mapping(beta, x_test)
        metrics = regression_metrics(y_true=y_true, y_pred=y_pred)
        pred_df = pd.DataFrame(
            {
                "model": model_names,
                "target_fb_brier": y_true,
                "feature_train_civbench_brier": x_train,
                "feature_test_civbench_brier": x_test,
                "predicted_fb_brier": y_pred,
                "eval_mode": eval_mode,
            }
        )
        pred_df["fit_intercept"] = float(beta[0])
        pred_df["fit_slope"] = float(beta[1])
        return metrics, pred_df

    if eval_mode == "holdout_models":
        n = len(y_true)
        y_pred = np.full(n, np.nan, dtype=float)
        intercepts = np.full(n, np.nan, dtype=float)
        slopes = np.full(n, np.nan, dtype=float)
        for i in range(n):
            keep = np.ones(n, dtype=bool)
            keep[i] = False
            beta = fit_linear_mapping(x_train[keep], y_true[keep])
            y_pred[i] = float(predict_linear_mapping(beta, np.array([x_test[i]]))[0])
            intercepts[i] = float(beta[0])
            slopes[i] = float(beta[1])
        metrics = regression_metrics(y_true=y_true, y_pred=y_pred)
        pred_df = pd.DataFrame(
            {
                "model": model_names,
                "target_fb_brier": y_true,
                "feature_train_civbench_brier": x_train,
                "feature_test_civbench_brier": x_test,
                "predicted_fb_brier": y_pred,
                "eval_mode": eval_mode,
                "fit_intercept": intercepts,
                "fit_slope": slopes,
            }
        )
        return metrics, pred_df

    raise ValueError(f"Unknown eval_mode: {eval_mode}")


def infer_default_split(seeds: list[str]) -> tuple[list[str], list[str]]:
    parsed = [(s, seed_sort_key(s)) for s in seeds]
    numeric = [s for s, key in parsed if key[0] in (0, 1)]
    if len(numeric) >= 10:
        train = [s for s in numeric if seed_sort_key(s)[1] <= 8]
        test = [s for s in numeric if seed_sort_key(s)[1] >= 9]
        if train and test:
            return train, test
    # fallback: first 80% train, rest test
    n = len(seeds)
    n_train = max(1, int(round(0.8 * n)))
    return seeds[:n_train], seeds[n_train:]


def scan_max_turn_in_game_json(path: Path) -> int | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    max_turn = -1

    def walk(node: object) -> None:
        nonlocal max_turn
        if isinstance(node, dict):
            for k, v in node.items():
                if k == "turn" and isinstance(v, int):
                    max_turn = max(max_turn, v)
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(data)
    return max_turn if max_turn >= 0 else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze CivBench -> ForecastBench transfer.")
    parser.add_argument(
        "--eval-json",
        default="data/evaluations/parallel_eval_all19_allq_seed42.json",
        help="Evaluation JSON with per-question predictions.",
    )
    parser.add_argument(
        "--forecastbench",
        default="data/ForecastBench.csv",
        help="ForecastBench CSV.",
    )
    parser.add_argument(
        "--games-dir",
        default="data/games",
        help="Directory with seed*_data.json files for max-turn checks.",
    )
    parser.add_argument(
        "--output-prefix",
        default="data/ridge/civbench_forecast_transfer_analysis",
        help="Output prefix for CSV/JSON artifacts.",
    )
    parser.add_argument("--include-h0", action="store_true")
    parser.add_argument(
        "--max-group-size",
        type=int,
        default=5,
        help="Max N for grouped-world stability (N worlds per group).",
    )
    parser.add_argument(
        "--fixed-train-worlds",
        default="",
        help="Comma-separated fixed train worlds. Default auto: seed1..seed8 when available.",
    )
    parser.add_argument(
        "--fixed-test-worlds",
        default="",
        help="Comma-separated fixed test worlds. Default auto: seed9..seed10 when available.",
    )
    parser.add_argument(
        "--stability-thresholds",
        default="0.8,0.85,0.9",
        help="Comma-separated thresholds used to estimate minimum N.",
    )
    parser.add_argument(
        "--path1-ridge-alpha",
        type=float,
        default=1.0,
        help="Ridge alpha for Path 1 pair-feature weighting.",
    )
    parser.add_argument(
        "--path2-group-size",
        type=int,
        default=3,
        help="Path 2 group size N for grouped-world feature averaging.",
    )
    parser.add_argument(
        "--path2-ridge-alpha",
        type=float,
        default=1.0,
        help="Ridge alpha for Path 2 grouped pair-feature weighting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    def resolve(path_like: str) -> Path:
        p = Path(path_like)
        return p if p.is_absolute() else (project_root / p)

    eval_json_path = resolve(args.eval_json)
    forecastbench_path = resolve(args.forecastbench)
    games_dir = resolve(args.games_dir)
    output_prefix = resolve(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    agg = load_eval_aggregates(eval_json_path=eval_json_path, include_h0=args.include_h0)
    model_matrix = compute_model_matrix(agg)
    difficulty_matrix, complete_pairs = compute_pair_difficulty_matrix(agg)

    fb_scores = load_forecastbench_dataset_scores(forecastbench_path)

    matched_models: list[str] = []
    target_fb: list[float] = []
    for model in agg.models:
        fb_name = CIVBENCH_TO_FORECASTBENCH.get(model)
        if not fb_name:
            continue
        score = fb_scores.get(fb_name.lower())
        if score is None:
            continue
        matched_models.append(model)
        target_fb.append(score)
    if len(matched_models) < 3:
        raise ValueError("Need at least 3 matched models with ForecastBench dataset scores.")
    y_true = np.array(target_fb, dtype=float)

    pair_features = complete_pairs_for_models(agg, matched_models)
    if not pair_features:
        raise ValueError("No complete (horizon, template) pairs for matched models.")

    # H1: model world-vs-world pairwise
    model_pair_rows: list[dict[str, object]] = []
    model_pair_rhos: list[float] = []
    for i, world_a in enumerate(agg.seeds):
        for j, world_b in enumerate(agg.seeds):
            if j <= i:
                continue
            rho = safe_spearman(model_matrix[i, :], model_matrix[j, :])
            model_pair_rows.append(
                {
                    "world_a": world_a,
                    "world_b": world_b,
                    "spearman": rho,
                }
            )
            if np.isfinite(rho):
                model_pair_rhos.append(rho)
    model_pair_df = pd.DataFrame(model_pair_rows).sort_values(["world_a", "world_b"])
    model_pair_path = output_prefix.with_name(f"{output_prefix.name}_h1_model_world_pairwise.csv")
    model_pair_df.to_csv(model_pair_path, index=False)

    # H1: difficulty world-vs-world pairwise
    diff_pair_rows: list[dict[str, object]] = []
    diff_pair_rhos: list[float] = []
    for i, world_a in enumerate(agg.seeds):
        for j, world_b in enumerate(agg.seeds):
            if j <= i:
                continue
            rho = safe_spearman(difficulty_matrix[i, :], difficulty_matrix[j, :])
            diff_pair_rows.append(
                {
                    "world_a": world_a,
                    "world_b": world_b,
                    "spearman": rho,
                }
            )
            if np.isfinite(rho):
                diff_pair_rhos.append(rho)
    diff_pair_df = pd.DataFrame(diff_pair_rows).sort_values(["world_a", "world_b"])
    diff_pair_path = output_prefix.with_name(f"{output_prefix.name}_h1_difficulty_world_pairwise.csv")
    diff_pair_df.to_csv(diff_pair_path, index=False)

    # H1 grouped stability summaries
    grouping_rows: list[dict[str, object]] = []
    max_group_size = min(int(args.max_group_size), len(agg.seeds))
    for n_worlds in range(1, max_group_size + 1):
        groups = list(itertools.combinations(agg.seeds, n_worlds))

        model_vectors = {g: model_vector_for_group(agg, g) for g in groups}
        for scope, vals in summarize_group_stability(model_vectors):
            stats_dict = summarize(vals)
            grouping_rows.append(
                {
                    "domain": "model_ranking",
                    "group_size": n_worlds,
                    "comparison_scope": scope,
                    "n_pairs": stats_dict["n"],
                    "spearman_min": stats_dict["min"],
                    "spearman_p10": stats_dict["p10"],
                    "spearman_median": stats_dict["median"],
                    "spearman_mean": stats_dict["mean"],
                    "spearman_max": stats_dict["max"],
                }
            )

        diff_vectors = {g: pair_difficulty_vector_for_group(agg, complete_pairs, g) for g in groups}
        for scope, vals in summarize_group_stability(diff_vectors):
            stats_dict = summarize(vals)
            grouping_rows.append(
                {
                    "domain": "pair_difficulty",
                    "group_size": n_worlds,
                    "comparison_scope": scope,
                    "n_pairs": stats_dict["n"],
                    "spearman_min": stats_dict["min"],
                    "spearman_p10": stats_dict["p10"],
                    "spearman_median": stats_dict["median"],
                    "spearman_mean": stats_dict["mean"],
                    "spearman_max": stats_dict["max"],
                }
            )

    grouping_df = pd.DataFrame(grouping_rows).sort_values(
        ["domain", "comparison_scope", "group_size"]
    )
    grouping_path = output_prefix.with_name(f"{output_prefix.name}_h1_grouping_summary.csv")
    grouping_df.to_csv(grouping_path, index=False)

    # H1 threshold-based minimum N (based on median disjoint-pairs stability).
    thresholds = [float(x.strip()) for x in args.stability_thresholds.split(",") if x.strip()]
    min_n_by_threshold: dict[str, dict[str, int | None]] = {}
    for domain in ["model_ranking", "pair_difficulty"]:
        domain_df = grouping_df[
            (grouping_df["domain"] == domain)
            & (grouping_df["comparison_scope"] == "disjoint_pairs")
        ]
        domain_map: dict[str, int | None] = {}
        for thr in thresholds:
            keep = domain_df[domain_df["spearman_median"] >= thr]
            best_n = int(keep["group_size"].min()) if len(keep) > 0 else None
            domain_map[str(thr)] = best_n
        min_n_by_threshold[domain] = domain_map

    # H2 fixed split
    fixed_train = parse_seed_csv(args.fixed_train_worlds)
    fixed_test = parse_seed_csv(args.fixed_test_worlds)
    if not fixed_train or not fixed_test:
        auto_train, auto_test = infer_default_split(agg.seeds)
        if not fixed_train:
            fixed_train = auto_train
        if not fixed_test:
            fixed_test = auto_test
    unknown_fixed = [s for s in (fixed_train + fixed_test) if s not in agg.seeds]
    if unknown_fixed:
        raise ValueError(f"Unknown fixed split worlds: {unknown_fixed}")
    if set(fixed_train) & set(fixed_test):
        raise ValueError("Fixed train/test worlds must be disjoint.")

    x_fixed_train = model_features_for_group(agg, matched_models, tuple(fixed_train))
    x_fixed_test = model_features_for_group(agg, matched_models, tuple(fixed_test))

    fixed_metrics_same, fixed_pred_same = run_world_split_eval(
        x_train=x_fixed_train,
        x_test=x_fixed_test,
        y_true=y_true,
        model_names=matched_models,
        eval_mode="same_models",
    )
    fixed_metrics_holdout, fixed_pred_holdout = run_world_split_eval(
        x_train=x_fixed_train,
        x_test=x_fixed_test,
        y_true=y_true,
        model_names=matched_models,
        eval_mode="holdout_models",
    )
    fixed_pred_df = pd.concat([fixed_pred_same, fixed_pred_holdout], ignore_index=True)
    fixed_pred_df["train_worlds"] = ",".join(fixed_train)
    fixed_pred_df["test_worlds"] = ",".join(fixed_test)
    fixed_pred_path = output_prefix.with_name(f"{output_prefix.name}_h2_fixed_split_predictions.csv")
    fixed_pred_df.to_csv(fixed_pred_path, index=False)

    # H2 exhaustive 8/2 world splits
    worldsplit_rows: list[dict[str, object]] = []
    worldsplit_pred_rows: list[pd.DataFrame] = []
    for test_worlds in itertools.combinations(agg.seeds, 2):
        train_worlds = tuple(w for w in agg.seeds if w not in test_worlds)
        x_train = model_features_for_group(agg, matched_models, train_worlds)
        x_test = model_features_for_group(agg, matched_models, test_worlds)

        for mode in ["same_models", "holdout_models"]:
            m, pred_df = run_world_split_eval(
                x_train=x_train,
                x_test=x_test,
                y_true=y_true,
                model_names=matched_models,
                eval_mode=mode,
            )
            worldsplit_rows.append(
                {
                    "train_worlds": ",".join(train_worlds),
                    "test_worlds": ",".join(test_worlds),
                    "eval_mode": mode,
                    **m,
                }
            )
            pred_df = pred_df.copy()
            pred_df["train_worlds"] = ",".join(train_worlds)
            pred_df["test_worlds"] = ",".join(test_worlds)
            worldsplit_pred_rows.append(pred_df)

    worldsplit_df = pd.DataFrame(worldsplit_rows).sort_values(["eval_mode", "test_worlds"])
    worldsplit_path = output_prefix.with_name(f"{output_prefix.name}_h2_worldsplit_8_2_metrics.csv")
    worldsplit_df.to_csv(worldsplit_path, index=False)

    worldsplit_pred_df = pd.concat(worldsplit_pred_rows, ignore_index=True).sort_values(
        ["eval_mode", "test_worlds", "model"]
    )
    worldsplit_pred_path = output_prefix.with_name(
        f"{output_prefix.name}_h2_worldsplit_8_2_predictions.csv"
    )
    worldsplit_pred_df.to_csv(worldsplit_pred_path, index=False)

    # H2 leave-one-world-out
    lowo_rows: list[dict[str, object]] = []
    for test_world in agg.seeds:
        train_worlds = tuple(w for w in agg.seeds if w != test_world)
        x_train = model_features_for_group(agg, matched_models, train_worlds)
        x_test = model_features_for_group(agg, matched_models, (test_world,))
        for mode in ["same_models", "holdout_models"]:
            m, _ = run_world_split_eval(
                x_train=x_train,
                x_test=x_test,
                y_true=y_true,
                model_names=matched_models,
                eval_mode=mode,
            )
            lowo_rows.append(
                {
                    "train_worlds": ",".join(train_worlds),
                    "test_world": test_world,
                    "eval_mode": mode,
                    **m,
                }
            )
    lowo_df = pd.DataFrame(lowo_rows).sort_values(["eval_mode", "test_world"])
    lowo_path = output_prefix.with_name(f"{output_prefix.name}_h2_lowo_metrics.csv")
    lowo_df.to_csv(lowo_path, index=False)

    # H2 single-world -> single-world matrix
    s2s_rows: list[dict[str, object]] = []
    for train_world in agg.seeds:
        for test_world in agg.seeds:
            if train_world == test_world:
                continue
            x_train = model_features_for_group(agg, matched_models, (train_world,))
            x_test = model_features_for_group(agg, matched_models, (test_world,))
            for mode in ["same_models", "holdout_models"]:
                m, _ = run_world_split_eval(
                    x_train=x_train,
                    x_test=x_test,
                    y_true=y_true,
                    model_names=matched_models,
                    eval_mode=mode,
                )
                s2s_rows.append(
                    {
                        "train_world": train_world,
                        "test_world": test_world,
                        "eval_mode": mode,
                        **m,
                    }
                )
    s2s_df = pd.DataFrame(s2s_rows).sort_values(["eval_mode", "train_world", "test_world"])
    s2s_path = output_prefix.with_name(f"{output_prefix.name}_h2_single_to_single_metrics.csv")
    s2s_df.to_csv(s2s_path, index=False)

    # H2 method comparison (Path 3 scalar vs Path 1/2 pair weighting) on same holdouts.
    def add_split_context(df: pd.DataFrame, **kwargs: object) -> pd.DataFrame:
        out = df.copy()
        for k, v in kwargs.items():
            out[k] = v
        return out

    # Fixed split metrics/predictions by method.
    fixed_method_metric_rows: list[dict[str, object]] = []
    fixed_method_pred_rows: list[pd.DataFrame] = []
    for mode in ["same_models", "holdout_models"]:
        metrics_rows, pred_dfs = run_all_h2_methods_for_split(
            agg=agg,
            models=matched_models,
            y_true=y_true,
            scalar_train_feature=x_fixed_train,
            scalar_test_feature=x_fixed_test,
            pair_feature_list=pair_features,
            train_worlds=tuple(fixed_train),
            test_worlds=tuple(fixed_test),
            scalar_eval_mode=mode,
            path1_ridge_alpha=float(args.path1_ridge_alpha),
            path2_group_size=int(args.path2_group_size),
            path2_ridge_alpha=float(args.path2_ridge_alpha),
        )
        fixed_method_metric_rows.extend(
            [
                {
                    **row,
                    "train_worlds": ",".join(fixed_train),
                    "test_worlds": ",".join(fixed_test),
                }
                for row in metrics_rows
            ]
        )
        for pdf in pred_dfs:
            fixed_method_pred_rows.append(
                add_split_context(
                    pdf,
                    train_worlds=",".join(fixed_train),
                    test_worlds=",".join(fixed_test),
                )
            )

    fixed_methods_metrics_df = pd.DataFrame(fixed_method_metric_rows).sort_values(
        ["eval_mode", "method"]
    )
    fixed_methods_metrics_path = output_prefix.with_name(
        f"{output_prefix.name}_h2_methods_fixed_split_metrics.csv"
    )
    fixed_methods_metrics_df.to_csv(fixed_methods_metrics_path, index=False)

    fixed_methods_pred_df = pd.concat(fixed_method_pred_rows, ignore_index=True).sort_values(
        ["eval_mode", "method", "model"]
    )
    fixed_methods_pred_path = output_prefix.with_name(
        f"{output_prefix.name}_h2_methods_fixed_split_predictions.csv"
    )
    fixed_methods_pred_df.to_csv(fixed_methods_pred_path, index=False)

    # 8/2 world split metrics/predictions by method.
    worldsplit_method_rows: list[dict[str, object]] = []
    worldsplit_method_pred_rows: list[pd.DataFrame] = []
    for test_worlds in itertools.combinations(agg.seeds, 2):
        train_worlds = tuple(w for w in agg.seeds if w not in test_worlds)
        x_train = model_features_for_group(agg, matched_models, train_worlds)
        x_test = model_features_for_group(agg, matched_models, test_worlds)

        for mode in ["same_models", "holdout_models"]:
            metrics_rows, pred_dfs = run_all_h2_methods_for_split(
                agg=agg,
                models=matched_models,
                y_true=y_true,
                scalar_train_feature=x_train,
                scalar_test_feature=x_test,
                pair_feature_list=pair_features,
                train_worlds=train_worlds,
                test_worlds=test_worlds,
                scalar_eval_mode=mode,
                path1_ridge_alpha=float(args.path1_ridge_alpha),
                path2_group_size=int(args.path2_group_size),
                path2_ridge_alpha=float(args.path2_ridge_alpha),
            )
            worldsplit_method_rows.extend(
                [
                    {
                        **row,
                        "train_worlds": ",".join(train_worlds),
                        "test_worlds": ",".join(test_worlds),
                    }
                    for row in metrics_rows
                ]
            )
            for pdf in pred_dfs:
                worldsplit_method_pred_rows.append(
                    add_split_context(
                        pdf,
                        train_worlds=",".join(train_worlds),
                        test_worlds=",".join(test_worlds),
                    )
                )

    worldsplit_methods_df = pd.DataFrame(worldsplit_method_rows).sort_values(
        ["eval_mode", "method", "test_worlds"]
    )
    worldsplit_methods_path = output_prefix.with_name(
        f"{output_prefix.name}_h2_methods_worldsplit_8_2_metrics.csv"
    )
    worldsplit_methods_df.to_csv(worldsplit_methods_path, index=False)

    worldsplit_methods_pred_df = pd.concat(worldsplit_method_pred_rows, ignore_index=True).sort_values(
        ["eval_mode", "method", "test_worlds", "model"]
    )
    worldsplit_methods_pred_path = output_prefix.with_name(
        f"{output_prefix.name}_h2_methods_worldsplit_8_2_predictions.csv"
    )
    worldsplit_methods_pred_df.to_csv(worldsplit_methods_pred_path, index=False)

    # Leave-one-world-out metrics by method.
    lowo_method_rows: list[dict[str, object]] = []
    for test_world in agg.seeds:
        train_worlds = tuple(w for w in agg.seeds if w != test_world)
        x_train = model_features_for_group(agg, matched_models, train_worlds)
        x_test = model_features_for_group(agg, matched_models, (test_world,))
        for mode in ["same_models", "holdout_models"]:
            metrics_rows, _ = run_all_h2_methods_for_split(
                agg=agg,
                models=matched_models,
                y_true=y_true,
                scalar_train_feature=x_train,
                scalar_test_feature=x_test,
                pair_feature_list=pair_features,
                train_worlds=train_worlds,
                test_worlds=(test_world,),
                scalar_eval_mode=mode,
                path1_ridge_alpha=float(args.path1_ridge_alpha),
                path2_group_size=int(args.path2_group_size),
                path2_ridge_alpha=float(args.path2_ridge_alpha),
            )
            lowo_method_rows.extend(
                [
                    {
                        **row,
                        "train_worlds": ",".join(train_worlds),
                        "test_world": test_world,
                    }
                    for row in metrics_rows
                ]
            )
    lowo_methods_df = pd.DataFrame(lowo_method_rows).sort_values(
        ["eval_mode", "method", "test_world"]
    )
    lowo_methods_path = output_prefix.with_name(f"{output_prefix.name}_h2_methods_lowo_metrics.csv")
    lowo_methods_df.to_csv(lowo_methods_path, index=False)

    # Single-world -> single-world metrics by method.
    s2s_method_rows: list[dict[str, object]] = []
    for train_world in agg.seeds:
        for test_world in agg.seeds:
            if train_world == test_world:
                continue
            x_train = model_features_for_group(agg, matched_models, (train_world,))
            x_test = model_features_for_group(agg, matched_models, (test_world,))
            for mode in ["same_models", "holdout_models"]:
                metrics_rows, _ = run_all_h2_methods_for_split(
                    agg=agg,
                    models=matched_models,
                    y_true=y_true,
                    scalar_train_feature=x_train,
                    scalar_test_feature=x_test,
                    pair_feature_list=pair_features,
                    train_worlds=(train_world,),
                    test_worlds=(test_world,),
                    scalar_eval_mode=mode,
                    path1_ridge_alpha=float(args.path1_ridge_alpha),
                    path2_group_size=int(args.path2_group_size),
                    path2_ridge_alpha=float(args.path2_ridge_alpha),
                )
                s2s_method_rows.extend(
                    [
                        {
                            **row,
                            "train_world": train_world,
                            "test_world": test_world,
                        }
                        for row in metrics_rows
                    ]
                )
    s2s_methods_df = pd.DataFrame(s2s_method_rows).sort_values(
        ["eval_mode", "method", "train_world", "test_world"]
    )
    s2s_methods_path = output_prefix.with_name(
        f"{output_prefix.name}_h2_methods_single_to_single_metrics.csv"
    )
    s2s_methods_df.to_csv(s2s_methods_path, index=False)

    # Optional world max-turn verification
    max_turn_by_world: dict[str, int | None] = {}
    for seed in agg.seeds:
        path = games_dir / f"{seed}_data.json"
        max_turn_by_world[seed] = scan_max_turn_in_game_json(path)
    all_turns_available = all(v is not None for v in max_turn_by_world.values())
    all_turns_ge_270 = all((v is not None and v >= 270) for v in max_turn_by_world.values())

    def mode_summary(df: pd.DataFrame) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for mode in ["same_models", "holdout_models"]:
            sub = df[df["eval_mode"] == mode]
            out[mode] = {
                "n_splits": int(len(sub)),
                "spearman_mean": float(sub["spearman"].mean()),
                "spearman_median": float(sub["spearman"].median()),
                "spearman_min": float(sub["spearman"].min()),
                "spearman_max": float(sub["spearman"].max()),
                "pearson_mean": float(sub["pearson"].mean()),
                "mae_mean": float(sub["mae"].mean()),
                "rmse_mean": float(sub["rmse"].mean()),
            }
        return out

    def method_mode_summary(df: pd.DataFrame) -> dict[str, dict[str, dict[str, float]]]:
        out: dict[str, dict[str, dict[str, float]]] = {}
        methods = sorted(df["method"].unique().tolist())
        for method in methods:
            out[method] = {}
            for mode in ["same_models", "holdout_models"]:
                sub = df[(df["method"] == method) & (df["eval_mode"] == mode)]
                if len(sub) == 0:
                    continue
                out[method][mode] = {
                    "n_splits": int(len(sub)),
                    "spearman_mean": float(sub["spearman"].mean()),
                    "spearman_median": float(sub["spearman"].median()),
                    "spearman_min": float(sub["spearman"].min()),
                    "spearman_max": float(sub["spearman"].max()),
                    "pearson_mean": float(sub["pearson"].mean()),
                    "mae_mean": float(sub["mae"].mean()),
                    "rmse_mean": float(sub["rmse"].mean()),
                }
        return out

    s2s_summary = mode_summary(s2s_df)
    worldsplit_summary = mode_summary(worldsplit_df)
    lowo_summary = mode_summary(lowo_df)
    methods_worldsplit_summary = method_mode_summary(worldsplit_methods_df)
    methods_lowo_summary = method_mode_summary(lowo_methods_df)
    methods_s2s_summary = method_mode_summary(s2s_methods_df)

    summary = {
        "method": "civbench_to_forecastbench_transfer",
        "mapping_form": "Includes scalar path and pair-feature weighting paths",
        "inputs": {
            "eval_json": str(eval_json_path),
            "forecastbench_csv": str(forecastbench_path),
            "include_h0": bool(args.include_h0),
        },
        "h2_methods_config": {
            "path3_scalar_linear": {},
            "path1_pair_ridge": {
                "ridge_alpha": float(args.path1_ridge_alpha),
            },
            "path2_grouped_pair_ridge": {
                "group_size": int(args.path2_group_size),
                "ridge_alpha": float(args.path2_ridge_alpha),
            },
        },
        "dataset": {
            "n_questions_total": int(agg.n_questions_total),
            "n_questions_non_h0": int(agg.n_questions_non_h0),
            "n_worlds": int(len(agg.seeds)),
            "worlds": agg.seeds,
            "n_models_eval": int(len(agg.models)),
            "n_models_matched_to_forecastbench": int(len(matched_models)),
            "n_complete_pairs_for_pair_weighting_methods": int(len(pair_features)),
            "questions_per_world": agg.questions_per_seed,
            "n_horizon_template_pairs_complete_for_difficulty": int(len(complete_pairs)),
        },
        "world_turn_check": {
            "games_dir": str(games_dir),
            "max_turn_by_world": max_turn_by_world,
            "all_turn_files_present": all_turns_available,
            "all_worlds_reach_turn_270_or_more": all_turns_ge_270,
        },
        "h1_model_world_pairwise": summarize(model_pair_rhos),
        "h1_difficulty_world_pairwise": summarize(diff_pair_rhos),
        "h1_grouping_min_n_by_threshold_median_disjoint": min_n_by_threshold,
        "h2_fixed_split": {
            "train_worlds": fixed_train,
            "test_worlds": fixed_test,
            "same_models": fixed_metrics_same,
            "holdout_models": fixed_metrics_holdout,
        },
        "h2_worldsplit_8_2_summary": worldsplit_summary,
        "h2_lowo_summary": lowo_summary,
        "h2_single_world_to_single_world_summary": s2s_summary,
        "h2_method_comparison_fixed_split_metrics": fixed_methods_metrics_df.to_dict(orient="records"),
        "h2_method_comparison_worldsplit_8_2_summary": methods_worldsplit_summary,
        "h2_method_comparison_lowo_summary": methods_lowo_summary,
        "h2_method_comparison_single_world_to_single_world_summary": methods_s2s_summary,
        "artifacts": {
            "h1_model_world_pairwise_csv": str(model_pair_path),
            "h1_difficulty_world_pairwise_csv": str(diff_pair_path),
            "h1_grouping_summary_csv": str(grouping_path),
            "h2_fixed_split_predictions_csv": str(fixed_pred_path),
            "h2_worldsplit_8_2_metrics_csv": str(worldsplit_path),
            "h2_worldsplit_8_2_predictions_csv": str(worldsplit_pred_path),
            "h2_lowo_metrics_csv": str(lowo_path),
            "h2_single_to_single_metrics_csv": str(s2s_path),
            "h2_methods_fixed_split_metrics_csv": str(fixed_methods_metrics_path),
            "h2_methods_fixed_split_predictions_csv": str(fixed_methods_pred_path),
            "h2_methods_worldsplit_8_2_metrics_csv": str(worldsplit_methods_path),
            "h2_methods_worldsplit_8_2_predictions_csv": str(worldsplit_methods_pred_path),
            "h2_methods_lowo_metrics_csv": str(lowo_methods_path),
            "h2_methods_single_to_single_metrics_csv": str(s2s_methods_path),
        },
    }

    summary_path = output_prefix.with_name(f"{output_prefix.name}_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))

    print("=" * 80)
    print("CIVBENCH -> FORECASTBENCH TRANSFER ANALYSIS")
    print("=" * 80)
    print(f"Worlds: {len(agg.seeds)} -> {agg.seeds}")
    print(f"Eval models: {len(agg.models)} | Matched to ForecastBench: {len(matched_models)}")
    print(f"Non-H0 questions: {agg.n_questions_non_h0}")
    print(f"Complete (horizon, template) pairs for difficulty: {len(complete_pairs)}")
    print(f"Complete (horizon, template) pairs for Path 1/2 weighting: {len(pair_features)}")
    print()
    print(
        "H1 model world-pair Spearman: "
        f"median={summary['h1_model_world_pairwise']['median']:.3f}, "
        f"min={summary['h1_model_world_pairwise']['min']:.3f}"
    )
    print(
        "H1 difficulty world-pair Spearman: "
        f"median={summary['h1_difficulty_world_pairwise']['median']:.3f}, "
        f"min={summary['h1_difficulty_world_pairwise']['min']:.3f}"
    )
    print()
    print(
        "H2 fixed split (same models): "
        f"rho={summary['h2_fixed_split']['same_models']['spearman']:.3f}, "
        f"mae={summary['h2_fixed_split']['same_models']['mae']:.3f}"
    )
    print(
        "H2 fixed split (holdout models): "
        f"rho={summary['h2_fixed_split']['holdout_models']['spearman']:.3f}, "
        f"mae={summary['h2_fixed_split']['holdout_models']['mae']:.3f}"
    )
    print()
    print("H2 fixed split method comparison:")
    for _, row in fixed_methods_metrics_df.iterrows():
        print(
            f"  {row['method']} [{row['eval_mode']}]: "
            f"rho={float(row['spearman']):.3f}, mae={float(row['mae']):.3f}"
        )
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
