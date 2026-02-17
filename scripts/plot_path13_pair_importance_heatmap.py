#!/usr/bin/env python3
"""
Plot template-horizon importance heatmaps for Path 1, Path 2, and Path 3.

Path 1:
- Explicit ridge over (horizon, template) pair features.

Path 2:
- Grouped-world pair ridge over the same (horizon, template) feature space.

Path 3 (scalar baseline in analyze_civbench_forecast_transfer.py):
- FB_hat = a + b * CivBenchScalar
- CivBenchScalar is a weighted mean of pair-level Brier features.
- Implicit pair coefficient for pair p is approximated as: b * w_p,
  where w_p is the mean training-set pair frequency across matched models.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_transfer_module(path: Path):
    spec = importlib.util.spec_from_file_location("transfer_mod", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module at {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def parse_csv_list(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def horizon_sort_key(h: str) -> tuple[int, str]:
    txt = str(h)
    if txt.upper().startswith("H"):
        try:
            return int(txt[1:]), txt
        except ValueError:
            pass
    return math.inf, txt


def build_grid(
    pairs: list[tuple[str, str]],
    values: np.ndarray,
) -> tuple[list[str], list[str], np.ndarray]:
    horizons = sorted({h for h, _ in pairs}, key=horizon_sort_key)
    templates = sorted({t for _, t in pairs})
    h_idx = {h: i for i, h in enumerate(horizons)}
    t_idx = {t: i for i, t in enumerate(templates)}
    grid = np.full((len(horizons), len(templates)), np.nan, dtype=float)
    for pair, val in zip(pairs, values, strict=False):
        h, t = pair
        grid[h_idx[h], t_idx[t]] = float(val)
    return horizons, templates, grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Path 1/2/3 pair-importance heatmap.")
    parser.add_argument(
        "--eval-json",
        default="data/evaluations/parallel_eval_all19_allq_seed42.json",
        help="Evaluation JSON.",
    )
    parser.add_argument(
        "--forecastbench",
        default="data/ForecastBench.csv",
        help="ForecastBench CSV.",
    )
    parser.add_argument(
        "--transfer-script",
        default="scripts/analyze_civbench_forecast_transfer.py",
        help="Path to transfer analysis module.",
    )
    parser.add_argument(
        "--output-prefix",
        default="data/ridge/civbench_forecast_transfer_all19_seed42_path123_pair_importance",
        help="Output prefix for CSV/PNG/JSON.",
    )
    parser.add_argument(
        "--train-worlds",
        default="",
        help="Optional comma-separated train worlds. Default: inferred split train worlds.",
    )
    parser.add_argument(
        "--path1-ridge-alpha",
        type=float,
        default=100.0,
        help="Ridge alpha for Path 1 fit used for importance.",
    )
    parser.add_argument(
        "--path2-group-size",
        type=int,
        default=2,
        help="World-group size for Path 2 grouped ridge.",
    )
    parser.add_argument(
        "--path2-ridge-alpha",
        type=float,
        default=1000.0,
        help="Ridge alpha for Path 2 grouped fit used for importance.",
    )
    parser.add_argument(
        "--include-h0",
        action="store_true",
        help="Include H0 templates.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=15,
        help="Top-k pairs to report in summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    def resolve(path_like: str) -> Path:
        p = Path(path_like)
        return p if p.is_absolute() else (project_root / p)

    eval_json = resolve(args.eval_json)
    forecastbench = resolve(args.forecastbench)
    transfer_script = resolve(args.transfer_script)
    output_prefix = resolve(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    out_csv = output_prefix.with_name(f"{output_prefix.name}_pair_coefficients.csv")
    out_png = output_prefix.with_name(f"{output_prefix.name}_heatmap.png")
    out_json = output_prefix.with_name(f"{output_prefix.name}_summary.json")

    mod = load_transfer_module(transfer_script)
    agg = mod.load_eval_aggregates(eval_json, include_h0=bool(args.include_h0))
    fb_scores = mod.load_forecastbench_dataset_scores(forecastbench)

    matched_models: list[str] = []
    y_list: list[float] = []
    for model in agg.models:
        fb_name = mod.CIVBENCH_TO_FORECASTBENCH.get(model)
        if not fb_name:
            continue
        score = fb_scores.get(fb_name.lower())
        if score is None:
            continue
        matched_models.append(model)
        y_list.append(float(score))

    if len(matched_models) < 3:
        raise ValueError("Need at least 3 matched models.")
    y_true = np.array(y_list, dtype=float)

    train_worlds = parse_csv_list(args.train_worlds)
    if not train_worlds:
        inferred_train, _ = mod.infer_default_split(agg.seeds)
        train_worlds = inferred_train

    unknown = [w for w in train_worlds if w not in agg.seeds]
    if unknown:
        raise ValueError(f"Unknown train worlds: {unknown}")
    train_worlds_tuple = tuple(train_worlds)

    pair_features = mod.complete_pairs_for_models(agg, matched_models)
    if not pair_features:
        raise ValueError("No complete (horizon, template) pairs for matched models.")

    # Path 1 explicit pair-feature ridge coefficients.
    X_path1 = mod.pair_feature_matrix_for_group(agg, matched_models, pair_features, train_worlds_tuple)
    path1_model = mod.fit_ridge_feature_model(X_path1, y_true, alpha=float(args.path1_ridge_alpha))
    path1_intercept, path1_coef = mod.ridge_coefficients_original_scale(path1_model)

    # Path 2 grouped-world pair-feature ridge coefficients.
    train_groups = mod.world_groups(train_worlds_tuple, group_size=int(args.path2_group_size))
    X_train_blocks = [mod.pair_feature_matrix_for_group(agg, matched_models, pair_features, g) for g in train_groups]
    X_path2 = np.vstack(X_train_blocks)
    y_path2 = np.concatenate([y_true for _ in train_groups])
    path2_model = mod.fit_ridge_feature_model(X_path2, y_path2, alpha=float(args.path2_ridge_alpha))
    path2_intercept, path2_coef = mod.ridge_coefficients_original_scale(path2_model)

    # Path 3 scalar baseline.
    x_scalar = mod.model_features_for_group(agg, matched_models, train_worlds_tuple)
    beta_scalar = mod.fit_linear_mapping(x_scalar, y_true)
    scalar_intercept = float(beta_scalar[0])
    scalar_slope = float(beta_scalar[1])

    # Decompose scalar feature into implicit pair coefficients:
    # scalar ~= sum_p w_p * pair_feature_p -> implicit coef_p = scalar_slope * w_p
    counts_mp = np.zeros((len(matched_models), len(pair_features)), dtype=float)
    for i, model in enumerate(matched_models):
        for j, pair in enumerate(pair_features):
            counts_mp[i, j] = float(
                sum(agg.cnt_seed_model_pair.get((seed, model, pair), 0) for seed in train_worlds_tuple)
            )
    totals_m = counts_mp.sum(axis=1, keepdims=True)
    if np.any(totals_m <= 0):
        raise ValueError("Encountered model with zero training questions while computing Path 3 weights.")
    weights_mp = counts_mp / totals_m
    mean_pair_weight = weights_mp.mean(axis=0)
    path3_implicit_coef = scalar_slope * mean_pair_weight

    rows: list[dict[str, object]] = []
    for pair, c1, c2, w, c3 in zip(
        pair_features, path1_coef, path2_coef, mean_pair_weight, path3_implicit_coef, strict=False
    ):
        horizon, template_id = pair
        rows.append(
            {
                "horizon": horizon,
                "template_id": template_id,
                "path1_coef": float(c1),
                "path1_abs_coef": float(abs(c1)),
                "path2_coef": float(c2),
                "path2_abs_coef": float(abs(c2)),
                "path3_mean_pair_weight": float(w),
                "path3_implicit_coef": float(c3),
                "path3_implicit_abs_coef": float(abs(c3)),
            }
        )

    coef_df = pd.DataFrame(rows).sort_values(["horizon", "template_id"], key=lambda s: s.map(str))
    coef_df.to_csv(out_csv, index=False)

    # Heatmaps for |coefficients| (importance magnitude), vertically stacked.
    horizons, templates, path1_grid = build_grid(pair_features, np.abs(path1_coef))
    _, _, path2_grid = build_grid(pair_features, np.abs(path2_coef))
    _, _, path3_grid = build_grid(pair_features, np.abs(path3_implicit_coef))

    finite_vals = np.concatenate(
        [
            path1_grid[np.isfinite(path1_grid)],
            path2_grid[np.isfinite(path2_grid)],
            path3_grid[np.isfinite(path3_grid)],
        ]
    )
    vmax = float(np.max(finite_vals)) if finite_vals.size else 1.0
    vmax = max(vmax, 1e-12)

    width = max(14.0, 0.8 * len(templates))
    height = max(10.0, 2.0 + 1.6 * len(horizons))
    fig, axes = plt.subplots(3, 1, figsize=(width, height), sharex=True, constrained_layout=True)

    im0 = axes[0].imshow(path1_grid, cmap="viridis", aspect="auto", vmin=0.0, vmax=vmax)
    axes[0].set_title(f"Path 1 |coef| (ridge alpha={float(args.path1_ridge_alpha):g})")
    axes[0].set_ylabel("Horizon")

    im1 = axes[1].imshow(path2_grid, cmap="viridis", aspect="auto", vmin=0.0, vmax=vmax)
    axes[1].set_title(
        "Path 2 |coef| "
        f"(group_size={int(args.path2_group_size)}, ridge alpha={float(args.path2_ridge_alpha):g})"
    )
    axes[1].set_ylabel("Horizon")

    im2 = axes[2].imshow(path3_grid, cmap="viridis", aspect="auto", vmin=0.0, vmax=vmax)
    axes[2].set_title("Path 3 implicit |coef| (scalar slope x pair frequency)")
    axes[2].set_ylabel("Horizon")
    axes[2].set_xlabel("Template")

    for ax in axes:
        ax.set_xticks(np.arange(len(templates)))
        ax.set_xticklabels(templates, rotation=90, fontsize=8)
        ax.set_yticks(np.arange(len(horizons)))
        ax.set_yticklabels(horizons, fontsize=9)

    cbar = fig.colorbar(im2, ax=axes.ravel().tolist(), shrink=0.9)
    cbar.set_label("|Coefficient| toward ForecastBench Brier")

    fig.suptitle(
        "Template-Horizon Importance toward ForecastBench Prediction\n"
        f"Train worlds: {','.join(train_worlds)} | matched models: {len(matched_models)}"
    )
    plt.savefig(out_png, dpi=200)
    plt.close(fig)

    top_k = max(1, int(args.top_k))
    top_path1 = (
        coef_df.sort_values("path1_abs_coef", ascending=False)
        .head(top_k)[["horizon", "template_id", "path1_coef", "path1_abs_coef"]]
        .to_dict(orient="records")
    )
    top_path2 = (
        coef_df.sort_values("path2_abs_coef", ascending=False)
        .head(top_k)[["horizon", "template_id", "path2_coef", "path2_abs_coef"]]
        .to_dict(orient="records")
    )
    top_path3 = (
        coef_df.sort_values("path3_implicit_abs_coef", ascending=False)
        .head(top_k)[["horizon", "template_id", "path3_implicit_coef", "path3_implicit_abs_coef"]]
        .to_dict(orient="records")
    )

    summary = {
        "eval_json": str(eval_json),
        "forecastbench_csv": str(forecastbench),
        "include_h0": bool(args.include_h0),
        "train_worlds": train_worlds,
        "n_matched_models": int(len(matched_models)),
        "n_pairs": int(len(pair_features)),
        "path1": {
            "ridge_alpha": float(args.path1_ridge_alpha),
            "intercept": float(path1_intercept),
            "feature_l1_norm": float(np.sum(np.abs(path1_coef))),
            "feature_l2_norm": float(np.linalg.norm(path1_coef)),
            "top_pairs_by_abs_coef": top_path1,
        },
        "path2_grouped": {
            "group_size": int(args.path2_group_size),
            "ridge_alpha": float(args.path2_ridge_alpha),
            "n_train_groups": int(len(train_groups)),
            "intercept": float(path2_intercept),
            "feature_l1_norm": float(np.sum(np.abs(path2_coef))),
            "feature_l2_norm": float(np.linalg.norm(path2_coef)),
            "top_pairs_by_abs_coef": top_path2,
        },
        "path3_scalar": {
            "intercept": scalar_intercept,
            "slope": scalar_slope,
            "top_pairs_by_implicit_abs_coef": top_path3,
        },
        "artifacts": {
            "pair_coefficients_csv": str(out_csv),
            "heatmap_png": str(out_png),
        },
    }
    out_json.write_text(json.dumps(summary, indent=2))

    print(f"Saved pair coefficients: {out_csv}")
    print(f"Saved heatmap: {out_png}")
    print(f"Saved summary: {out_json}")
    print(f"Path 2 grouped config: group_size={int(args.path2_group_size)}, alpha={float(args.path2_ridge_alpha):g}")
    print(f"Path 3 scalar slope: {scalar_slope:.6f}")


if __name__ == "__main__":
    main()
