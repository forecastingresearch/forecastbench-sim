#!/usr/bin/env python3
"""
Path 3 only: model-level regressors with leave-one-model-out (LOO) evaluation.

Uses pooled all-world CivBench features (horizon-template Brier vector per model)
to predict ForecastBench Dataset Brier, comparing multiple regressors.

Default methods:
- ridge
- lasso
- elasticnet
- xgboost
- random_forest
- gradient_boosting
- ols
"""

from __future__ import annotations

import argparse
import json
import math
import re
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, Lasso

try:
    from xgboost import XGBRegressor  # type: ignore

    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False


# CivBench model id -> ForecastBench model name
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
class EvalTensor:
    seeds: list[str]
    models: list[str]
    pairs: list[tuple[str, str]]
    brier_by_seed_model_pair: np.ndarray  # [S, M, P]
    brier_by_model_pair: np.ndarray  # [M, P], weighted across seeds
    question_counts_by_pair: np.ndarray  # [P]
    question_counts_by_seed_pair: np.ndarray  # [S, P]
    dropped_incomplete_pairs: int


@dataclass
class ForecastTarget:
    model_indices: list[int]
    model_names: list[str]
    forecast_scores: np.ndarray


def seed_sort_key(seed: str) -> tuple[int, int | str]:
    s = str(seed)
    m = re.fullmatch(r"seed(\d+)", s, flags=re.IGNORECASE)
    if m:
        return 0, int(m.group(1))
    m = re.search(r"(\d+)$", s)
    if m:
        return 1, int(m.group(1))
    return 2, s


def parse_metric_with_n(value: str) -> float | None:
    m = re.match(r"([\d.]+)\s*\(([\d,]+)\)", str(value).strip())
    if not m:
        return None
    return float(m.group(1))


def load_forecastbench_dataset_scores(path: Path) -> dict[str, float]:
    df = pd.read_csv(path)
    df = df[df["Model"].astype(str).str.contains(r"\(zero shot\)", case=False, regex=True)]
    scores: dict[str, float] = {}
    for _, row in df.iterrows():
        raw_name = str(row["Model"]).strip()
        base_name = re.sub(r"\s*\(zero shot\)\s*$", "", raw_name, flags=re.IGNORECASE).strip()
        score = parse_metric_with_n(row.get("Dataset (N)", ""))
        if score is not None:
            scores[base_name.lower()] = score
    return scores


def load_eval_tensor(eval_json_path: Path, include_h0: bool = False) -> EvalTensor:
    data = json.loads(eval_json_path.read_text())
    questions = data.get("questions", [])

    models_meta = data.get("metadata", {}).get("models", [])
    if models_meta:
        models = sorted(str(m) for m in models_meta)
    else:
        models = sorted(
            {
                str(model_id)
                for q in questions
                for model_id in q.get("predictions", {}).keys()
            }
        )

    sums_seed: dict[tuple[str, str, tuple[str, str]], float] = defaultdict(float)
    cnts_seed: dict[tuple[str, str, tuple[str, str]], int] = defaultdict(int)
    counts_seed_pair: dict[tuple[str, tuple[str, str]], int] = defaultdict(int)

    seed_set: set[str] = set()
    pair_set: set[tuple[str, str]] = set()

    for q in questions:
        template_id = str(q.get("template_id", ""))
        if not include_h0 and template_id.startswith("h0_"):
            continue

        horizon = str(q.get("horizon", ""))
        seed_val = q.get("seed", None)
        if seed_val in (None, ""):
            seed_val = q.get("game_id", None)
        seed = str(seed_val) if seed_val is not None else ""
        if not seed:
            continue

        gt = q.get("ground_truth", None)
        if isinstance(gt, bool):
            gt_value = 1.0 if gt else 0.0
        elif gt is None:
            continue
        else:
            gt_value = float(gt)

        pair = (horizon, template_id)
        preds = q.get("predictions", {})
        if not preds:
            continue

        seed_set.add(seed)
        pair_set.add(pair)

        for model_id, pred in preds.items():
            if pred is None:
                continue
            brier_raw = pred.get("brier_score", None)
            if brier_raw is None:
                prob_raw = pred.get("probability", None)
                if prob_raw is None:
                    continue
                prob = float(prob_raw)
                if not np.isfinite(prob):
                    continue
                prob = min(max(prob, 0.0), 1.0)
                brier = (prob - gt_value) ** 2
            else:
                brier = float(brier_raw)
            if not np.isfinite(brier):
                continue

            key = (seed, str(model_id), pair)
            sums_seed[key] += brier
            cnts_seed[key] += 1

        counts_seed_pair[(seed, pair)] += 1

    seeds = sorted(seed_set, key=seed_sort_key)
    pairs = sorted(pair_set, key=lambda x: (x[0], x[1]))
    if not seeds or not pairs:
        raise ValueError("No usable seeds/pairs found in eval JSON.")

    seed_idx = {s: i for i, s in enumerate(seeds)}
    model_idx = {m: i for i, m in enumerate(models)}
    pair_idx = {p: i for i, p in enumerate(pairs)}

    S, M, P = len(seeds), len(models), len(pairs)
    brier_smp = np.full((S, M, P), np.nan, dtype=float)
    counts_sp = np.zeros((S, P), dtype=int)

    for (seed, model, pair), total in sums_seed.items():
        if model not in model_idx:
            continue
        i = seed_idx[seed]
        j = model_idx[model]
        k = pair_idx[pair]
        brier_smp[i, j, k] = total / cnts_seed[(seed, model, pair)]

    for (seed, pair), n in counts_seed_pair.items():
        counts_sp[seed_idx[seed], pair_idx[pair]] = n

    complete_mask = np.isfinite(brier_smp).all(axis=(0, 1)) & (counts_sp > 0).all(axis=0)
    dropped = int(np.sum(~complete_mask))
    if dropped > 0:
        keep = np.where(complete_mask)[0]
        pairs = [pairs[k] for k in keep]
        brier_smp = brier_smp[:, :, keep]
        counts_sp = counts_sp[:, keep]

    if brier_smp.shape[2] == 0:
        raise ValueError("No complete (horizon, template) pairs remain after filtering.")

    denom = counts_sp.sum(axis=0)
    weighted_sum_mp = np.einsum("smp,sp->mp", brier_smp, counts_sp)
    brier_mp = weighted_sum_mp / denom[None, :]

    return EvalTensor(
        seeds=seeds,
        models=models,
        pairs=pairs,
        brier_by_seed_model_pair=brier_smp,
        brier_by_model_pair=brier_mp,
        question_counts_by_pair=denom.astype(int),
        question_counts_by_seed_pair=counts_sp,
        dropped_incomplete_pairs=dropped,
    )


def build_forecast_target(models: list[str], forecast_scores: dict[str, float]) -> ForecastTarget:
    model_to_index = {m: i for i, m in enumerate(models)}
    matched: list[tuple[int, str, float]] = []
    for model in models:
        fb_name = CIVBENCH_TO_FORECASTBENCH.get(model)
        if not fb_name:
            continue
        score = forecast_scores.get(fb_name.lower())
        if score is None:
            continue
        matched.append((model_to_index[model], model, score))

    if len(matched) < 3:
        raise ValueError("Need at least 3 matched models between CivBench and ForecastBench Dataset.")

    return ForecastTarget(
        model_indices=[m[0] for m in matched],
        model_names=[m[1] for m in matched],
        forecast_scores=np.array([m[2] for m in matched], dtype=float),
    )


def spearman_safe(x: np.ndarray, y: np.ndarray) -> float:
    rho = stats.spearmanr(x, y).statistic
    if rho is None or np.isnan(rho):
        return math.nan
    return float(rho)


def standardize_train_test(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    sigma = np.where(sigma <= 1e-12, 1.0, sigma)
    return (X_train - mu) / sigma, (X_test - mu) / sigma


def method_grid(method: str) -> list[dict[str, float]]:
    if method == "ridge":
        return [{"alpha": a} for a in [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]]
    if method == "lasso":
        return [{"alpha": a} for a in [1e-4, 1e-3, 1e-2, 1e-1, 1.0]]
    if method == "elasticnet":
        return [
            {"alpha": a, "l1_ratio": r}
            for a in [1e-3, 1e-2, 1e-1]
            for r in [0.3, 0.7]
        ]
    if method == "xgboost":
        return [{"n_estimators": 300.0, "learning_rate": 0.05, "max_depth": 3.0}]
    if method == "random_forest":
        return [{"n_estimators": 400.0, "max_depth": 6.0, "min_samples_leaf": 2.0}]
    if method == "gradient_boosting":
        return [{"n_estimators": 400.0, "learning_rate": 0.03, "max_depth": 2.0}]
    if method == "ols":
        return [{}]
    raise ValueError(f"Unknown method: {method}")


def fit_predict_regressor(
    method: str,
    config: dict[str, float],
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    X_pred_raw: np.ndarray,
    random_seed: int,
) -> np.ndarray:
    if method in {"ridge", "lasso", "elasticnet", "ols"}:
        X_train, X_pred = standardize_train_test(X_train_raw, X_pred_raw)
    else:
        X_train, X_pred = X_train_raw, X_pred_raw

    if method == "ridge":
        alpha = float(config.get("alpha", 1.0))
        # Closed-form ridge with intercept, unregularized intercept.
        n_train, n_features = X_train.shape
        X1 = np.hstack([np.ones((n_train, 1)), X_train])
        P = n_features + 1
        reg = np.eye(P)
        reg[0, 0] = 0.0
        beta = np.linalg.solve(X1.T @ X1 + alpha * reg, X1.T @ y_train)
        Xp1 = np.hstack([np.ones((X_pred.shape[0], 1)), X_pred])
        return Xp1 @ beta

    if method == "lasso":
        alpha = float(config.get("alpha", 1e-2))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            model = Lasso(alpha=alpha, fit_intercept=True, max_iter=20000, random_state=random_seed)
            model.fit(X_train, y_train)
        return model.predict(X_pred)

    if method == "elasticnet":
        alpha = float(config.get("alpha", 1e-2))
        l1_ratio = float(config.get("l1_ratio", 0.5))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            model = ElasticNet(
                alpha=alpha,
                l1_ratio=l1_ratio,
                fit_intercept=True,
                max_iter=20000,
                random_state=random_seed,
            )
            model.fit(X_train, y_train)
        return model.predict(X_pred)

    if method == "ols":
        # OLS via pseudo-inverse with intercept.
        n_train = X_train.shape[0]
        X1 = np.hstack([np.ones((n_train, 1)), X_train])
        beta = np.linalg.pinv(X1) @ y_train
        Xp1 = np.hstack([np.ones((X_pred.shape[0], 1)), X_pred])
        return Xp1 @ beta

    if method == "xgboost":
        if not HAS_XGBOOST:
            raise RuntimeError("xgboost is not available in this environment.")
        model = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=int(config.get("n_estimators", 300)),
            learning_rate=float(config.get("learning_rate", 0.05)),
            max_depth=int(config.get("max_depth", 3)),
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.0,
            reg_lambda=1.0,
            random_state=random_seed,
            verbosity=0,
        )
        model.fit(X_train, y_train)
        return model.predict(X_pred)

    if method == "random_forest":
        model = RandomForestRegressor(
            n_estimators=int(config.get("n_estimators", 400)),
            max_depth=None if config.get("max_depth", 6.0) <= 0 else int(config.get("max_depth", 6)),
            min_samples_leaf=int(config.get("min_samples_leaf", 2)),
            random_state=random_seed,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        return model.predict(X_pred)

    if method == "gradient_boosting":
        model = GradientBoostingRegressor(
            n_estimators=int(config.get("n_estimators", 400)),
            learning_rate=float(config.get("learning_rate", 0.03)),
            max_depth=int(config.get("max_depth", 2)),
            random_state=random_seed,
        )
        model.fit(X_train, y_train)
        return model.predict(X_pred)

    raise ValueError(f"Unknown method: {method}")


def select_config_by_inner_loo(
    method: str,
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    random_seed: int,
) -> tuple[dict[str, float], pd.DataFrame]:
    grid = method_grid(method)
    if len(grid) == 1:
        df = pd.DataFrame(
            [{"config_json": json.dumps(grid[0], sort_keys=True), "inner_loo_spearman": math.nan, "inner_loo_mae": math.nan}]
        )
        return grid[0], df

    n = X_train_raw.shape[0]
    rows = []
    for cfg_idx, cfg in enumerate(grid):
        preds = np.full(n, np.nan, dtype=float)
        for i in range(n):
            keep = np.ones(n, dtype=bool)
            keep[i] = False
            try:
                pred_i = fit_predict_regressor(
                    method=method,
                    config=cfg,
                    X_train_raw=X_train_raw[keep, :],
                    y_train=y_train[keep],
                    X_pred_raw=X_train_raw[~keep, :],
                    random_seed=random_seed + 1000 * cfg_idx + i,
                )
                preds[i] = float(pred_i[0])
            except Exception:
                preds[i] = math.nan

        mask = np.isfinite(preds)
        if np.sum(mask) < 3:
            rho = -math.inf
            mae = math.inf
        else:
            rho_raw = spearman_safe(preds[mask], y_train[mask])
            rho = -math.inf if not np.isfinite(rho_raw) else float(rho_raw)
            mae = float(np.mean(np.abs(preds[mask] - y_train[mask])))

        rows.append(
            {
                "config_json": json.dumps(cfg, sort_keys=True),
                "inner_loo_spearman": rho,
                "inner_loo_mae": mae,
            }
        )

    df = pd.DataFrame(rows)
    best = df.sort_values(
        ["inner_loo_spearman", "inner_loo_mae"],
        ascending=[False, True],
    ).iloc[0]
    best_cfg = json.loads(best["config_json"])
    return best_cfg, df


def plot_actual_vs_predicted_multimethod(
    out_path: Path,
    actual: np.ndarray,
    pred_by_method: dict[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    methods = list(pred_by_method.keys())
    n = len(methods)
    ncols = min(3, n)
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes_arr = np.array(axes).reshape(-1)
    metrics: dict[str, dict[str, float]] = {}

    for i, method in enumerate(methods):
        ax = axes_arr[i]
        pred = pred_by_method[method]
        mask = np.isfinite(actual) & np.isfinite(pred)
        x = actual[mask]
        y = pred[mask]
        rho = spearman_safe(x, y)
        pear = stats.pearsonr(x, y).statistic if len(x) >= 2 else math.nan
        mae = float(np.mean(np.abs(y - x)))

        ax.scatter(x, y)
        lo = min(float(np.min(x)), float(np.min(y)))
        hi = max(float(np.max(x)), float(np.max(y)))
        pad = 0.02 * (hi - lo + 1e-12)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "r--", alpha=0.5)
        ax.set_xlabel("Actual FB Brier")
        ax.set_ylabel("Predicted FB Brier")
        ax.set_title(f"{method} (rho={rho:.3f}, mae={mae:.3f})")
        metrics[method] = {"spearman": float(rho), "pearson": float(pear), "mae": float(mae)}

    for j in range(i + 1, len(axes_arr)):
        axes_arr[j].axis("off")

    fig.suptitle("Path 3 LOO: Actual vs Predicted ForecastBench Brier", fontsize=12, y=1.01)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return metrics


def plot_method_comparison(out_path: Path, summary_df: pd.DataFrame) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = summary_df.sort_values("oof_spearman", ascending=False).reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].bar(df["method"], df["oof_spearman"])
    axes[0].set_title("OOF Spearman (higher is better)")
    axes[0].set_ylim(0, 1)
    axes[0].tick_params(axis="x", rotation=30)

    axes[1].bar(df["method"], df["oof_mae"])
    axes[1].set_title("OOF MAE (lower is better)")
    axes[1].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def run_multimethod_loo(
    tensor: EvalTensor,
    target: ForecastTarget,
    output_prefix: Path,
    methods: list[str],
    random_seed: int,
) -> dict[str, object]:
    X_all_raw = tensor.brier_by_model_pair[target.model_indices, :]
    y = target.forecast_scores
    model_names = target.model_names
    n_models = len(y)

    folds_rows: list[dict[str, object]] = []
    tuning_rows: list[dict[str, object]] = []
    pred_by_method: dict[str, np.ndarray] = {}
    metrics_rows: list[dict[str, object]] = []

    for m_idx, method in enumerate(methods):
        print(f"Running method: {method}")
        oof_pred = np.full(n_models, np.nan, dtype=float)

        for holdout_i in range(n_models):
            keep = np.ones(n_models, dtype=bool)
            keep[holdout_i] = False

            Xtr = X_all_raw[keep, :]
            ytr = y[keep]
            Xho = X_all_raw[~keep, :]

            best_cfg, tune_df = select_config_by_inner_loo(
                method=method,
                X_train_raw=Xtr,
                y_train=ytr,
                random_seed=random_seed + 100_000 * m_idx + holdout_i,
            )
            tune_df = tune_df.copy()
            tune_df["method"] = method
            tune_df["outer_fold"] = holdout_i + 1
            tune_df["holdout_model"] = model_names[holdout_i]
            tuning_rows.extend(tune_df.to_dict(orient="records"))

            pred_i = fit_predict_regressor(
                method=method,
                config=best_cfg,
                X_train_raw=Xtr,
                y_train=ytr,
                X_pred_raw=Xho,
                random_seed=random_seed + 10_000 * m_idx + holdout_i,
            )
            pred_val = float(pred_i[0])
            true_val = float(y[holdout_i])
            oof_pred[holdout_i] = pred_val

            folds_rows.append(
                {
                    "method": method,
                    "fold": holdout_i + 1,
                    "holdout_sampling_scheme": "leave_one_model_out",
                    "holdout_model": model_names[holdout_i],
                    "best_config_json": json.dumps(best_cfg, sort_keys=True),
                    "forecastbench_dataset_brier": true_val,
                    "predicted_fb_from_all_world_features": pred_val,
                    "abs_error": float(abs(pred_val - true_val)),
                    "sq_error": float((pred_val - true_val) ** 2),
                }
            )

        pred_by_method[method] = oof_pred
        oof_spear = spearman_safe(oof_pred, y)
        oof_pear = stats.pearsonr(oof_pred, y).statistic if n_models >= 2 else math.nan
        oof_mae = float(np.mean(np.abs(oof_pred - y)))
        oof_rmse = float(np.sqrt(np.mean((oof_pred - y) ** 2)))

        metrics_rows.append(
            {
                "method": method,
                "oof_spearman": float(oof_spear),
                "oof_pearson": float(oof_pear),
                "oof_mae": float(oof_mae),
                "oof_rmse": float(oof_rmse),
                "n_models": int(n_models),
                "n_pairs": int(X_all_raw.shape[1]),
            }
        )

    folds_df = pd.DataFrame(folds_rows).sort_values(["method", "holdout_model"])
    folds_path = output_prefix.with_name(f"{output_prefix.name}_path3_model_holdout_folds.csv")
    folds_df.to_csv(folds_path, index=False)

    oof_df = pd.DataFrame({"model": model_names, "forecastbench_dataset_brier": y})
    for method, pred in pred_by_method.items():
        oof_df[f"holdout_pred_fb_{method}"] = pred
    oof_path = output_prefix.with_name(f"{output_prefix.name}_path3_model_holdout_oof_predictions.csv")
    oof_df.to_csv(oof_path, index=False)

    tuning_df = pd.DataFrame(tuning_rows)
    tuning_path = output_prefix.with_name(f"{output_prefix.name}_path3_hyperparam_selection.csv")
    tuning_df.to_csv(tuning_path, index=False)

    metrics_df = pd.DataFrame(metrics_rows).sort_values("oof_spearman", ascending=False)
    metrics_path = output_prefix.with_name(f"{output_prefix.name}_path3_method_comparison.csv")
    metrics_df.to_csv(metrics_path, index=False)

    scatter_path = output_prefix.with_name(f"{output_prefix.name}_path3_model_holdout_actual_vs_predicted.png")
    scatter_metrics = plot_actual_vs_predicted_multimethod(scatter_path, y, pred_by_method)

    compare_plot_path = output_prefix.with_name(f"{output_prefix.name}_path3_method_comparison.png")
    plot_method_comparison(compare_plot_path, metrics_df)

    summary = {
        "method_family": "path3_model_level_regression",
        "feature_mode": "all_worlds",
        "holdout_sampling_scheme": "leave_one_model_out",
        "methods_requested": methods,
        "methods_run": list(pred_by_method.keys()),
        "n_models": int(n_models),
        "n_pairs": int(X_all_raw.shape[1]),
        "method_metrics": metrics_df.to_dict(orient="records"),
        "folds_csv": str(folds_path),
        "oof_predictions_csv": str(oof_path),
        "hyperparam_selection_csv": str(tuning_path),
        "method_comparison_csv": str(metrics_path),
        "actual_vs_predicted_plot": str(scatter_path),
        "actual_vs_predicted_metrics": scatter_metrics,
        "method_comparison_plot": str(compare_plot_path),
    }

    summary_path = output_prefix.with_name(f"{output_prefix.name}_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    summary["summary_json"] = str(summary_path)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Path 3 multimethod LOO comparison (all worlds).")
    parser.add_argument("--eval-json", default="data/evaluations/parallel_eval_all19_allq_seed42.json")
    parser.add_argument("--forecastbench", default="data/ForecastBench.csv")
    parser.add_argument("--include-h0", action="store_true")
    parser.add_argument(
        "--methods",
        default="ridge,lasso,elasticnet,xgboost,random_forest,gradient_boosting,ols",
        help="Comma-separated methods.",
    )
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--output-prefix",
        default="data/forecastbench_aligned_optimizer_holdout_models_allworlds",
        help="Output prefix.",
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
    output_prefix = resolve(args.output_prefix)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    allowed = {"ridge", "lasso", "elasticnet", "xgboost", "random_forest", "gradient_boosting", "ols"}
    invalid = [m for m in methods if m not in allowed]
    if invalid:
        raise ValueError(f"Invalid methods: {invalid}. Allowed: {sorted(allowed)}")
    if "xgboost" in methods and not HAS_XGBOOST:
        print("Warning: xgboost requested but unavailable; skipping.")
        methods = [m for m in methods if m != "xgboost"]

    print("=" * 80)
    print("PATH 3 MULTIMETHOD LOO (ALL WORLDS)")
    print("=" * 80)
    print(f"Eval JSON: {eval_json_path}")
    print(f"ForecastBench: {forecastbench_path}")
    print(f"Methods: {methods}")

    tensor = load_eval_tensor(eval_json_path, include_h0=args.include_h0)
    target = build_forecast_target(tensor.models, load_forecastbench_dataset_scores(forecastbench_path))

    print(f"Seeds: {len(tensor.seeds)} -> {tensor.seeds}")
    print(f"Models: {len(tensor.models)}")
    print(f"(Horizon, template) pairs: {len(tensor.pairs)}")
    if tensor.dropped_incomplete_pairs > 0:
        print(f"Dropped incomplete pairs: {tensor.dropped_incomplete_pairs}")
    print(f"Matched models to ForecastBench Dataset: {len(target.model_indices)}")

    summary = run_multimethod_loo(
        tensor=tensor,
        target=target,
        output_prefix=output_prefix,
        methods=methods,
        random_seed=args.random_seed,
    )

    top = summary["method_metrics"][0] if summary["method_metrics"] else None
    if top is not None:
        print(
            "Best by OOF Spearman: "
            f"{top['method']} (rho={top['oof_spearman']:.4f}, mae={top['oof_mae']:.4f})"
        )
    print(f"Saved summary: {summary['summary_json']}")


if __name__ == "__main__":
    main()
