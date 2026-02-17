#!/usr/bin/env python3
"""
Sweep Path 2 (grouped pair-feature ridge) hyperparameters and plot results.

Evaluates:
- fixed split (seed1..8 -> seed9..10, fallback to inferred split)
- all 8/2 world splits
- leave-one-world-out (LOWO)

Selection metric:
- Mean Spearman on 8/2 splits with holdout-model evaluation.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
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


def parse_csv_ints(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_csv_floats(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep Path 2 hyperparameters.")
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
        help="Path to transfer analysis script module.",
    )
    parser.add_argument(
        "--group-sizes",
        default="2,3,4,5",
        help="Comma-separated Path 2 group sizes.",
    )
    parser.add_argument(
        "--alphas",
        default="0.01,0.1,1,10,100",
        help="Comma-separated ridge alphas.",
    )
    parser.add_argument(
        "--output-prefix",
        default="data/ridge/civbench_forecast_transfer_all19_seed42_path2_sweep",
        help="Output prefix for CSV/JSON/PNG.",
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

    output_csv = output_prefix.with_name(f"{output_prefix.name}_summary.csv")
    output_png = output_prefix.with_name(f"{output_prefix.name}_heatmap.png")
    output_json = output_prefix.with_name(f"{output_prefix.name}_best.json")

    group_sizes = parse_csv_ints(args.group_sizes)
    alphas = parse_csv_floats(args.alphas)
    if not group_sizes or not alphas:
        raise ValueError("Need non-empty group sizes and alphas.")

    mod = load_transfer_module(transfer_script)
    agg = mod.load_eval_aggregates(eval_json, include_h0=False)
    fb_scores = mod.load_forecastbench_dataset_scores(forecastbench)

    matched_models: list[str] = []
    target_fb: list[float] = []
    for model in agg.models:
        fb_name = mod.CIVBENCH_TO_FORECASTBENCH.get(model)
        if not fb_name:
            continue
        score = fb_scores.get(fb_name.lower())
        if score is None:
            continue
        matched_models.append(model)
        target_fb.append(score)

    if len(matched_models) < 3:
        raise ValueError("Need at least 3 matched models.")
    y_true = np.array(target_fb, dtype=float)

    pairs = mod.complete_pairs_for_models(agg, matched_models)
    if not pairs:
        raise ValueError("No complete (horizon, template) pairs for matched models.")

    # fixed split: seed1..8 train, seed9..10 test when available
    all_seeds = agg.seeds
    train_fixed = [
        s for s in all_seeds
        if mod.seed_sort_key(s)[0] in (0, 1) and mod.seed_sort_key(s)[1] <= 8
    ]
    test_fixed = [
        s for s in all_seeds
        if mod.seed_sort_key(s)[0] in (0, 1) and mod.seed_sort_key(s)[1] >= 9
    ]
    if not train_fixed or not test_fixed:
        train_fixed, test_fixed = mod.infer_default_split(all_seeds)
    train_fixed = tuple(train_fixed)
    test_fixed = tuple(test_fixed)

    rows: list[dict[str, float | int]] = []
    total = len(group_sizes) * len(alphas)
    done = 0
    for group_size in group_sizes:
        for alpha in alphas:
            done += 1
            print(f"[{done}/{total}] group_size={group_size}, alpha={alpha}")

            fixed_same, _ = mod.run_path2_grouped_pair_ridge_eval(
                agg=agg,
                models=matched_models,
                y_true=y_true,
                pairs=pairs,
                train_worlds=train_fixed,
                test_worlds=test_fixed,
                eval_mode="same_models",
                ridge_alpha=float(alpha),
                group_size=int(group_size),
            )
            fixed_hold, _ = mod.run_path2_grouped_pair_ridge_eval(
                agg=agg,
                models=matched_models,
                y_true=y_true,
                pairs=pairs,
                train_worlds=train_fixed,
                test_worlds=test_fixed,
                eval_mode="holdout_models",
                ridge_alpha=float(alpha),
                group_size=int(group_size),
            )

            ws_same = []
            ws_hold = []
            for test_worlds in itertools.combinations(all_seeds, 2):
                train_worlds = tuple(w for w in all_seeds if w not in test_worlds)
                m_same, _ = mod.run_path2_grouped_pair_ridge_eval(
                    agg=agg,
                    models=matched_models,
                    y_true=y_true,
                    pairs=pairs,
                    train_worlds=train_worlds,
                    test_worlds=test_worlds,
                    eval_mode="same_models",
                    ridge_alpha=float(alpha),
                    group_size=int(group_size),
                )
                m_hold, _ = mod.run_path2_grouped_pair_ridge_eval(
                    agg=agg,
                    models=matched_models,
                    y_true=y_true,
                    pairs=pairs,
                    train_worlds=train_worlds,
                    test_worlds=test_worlds,
                    eval_mode="holdout_models",
                    ridge_alpha=float(alpha),
                    group_size=int(group_size),
                )
                ws_same.append(m_same)
                ws_hold.append(m_hold)

            lowo_same = []
            lowo_hold = []
            for test_world in all_seeds:
                train_worlds = tuple(w for w in all_seeds if w != test_world)
                m_same, _ = mod.run_path2_grouped_pair_ridge_eval(
                    agg=agg,
                    models=matched_models,
                    y_true=y_true,
                    pairs=pairs,
                    train_worlds=train_worlds,
                    test_worlds=(test_world,),
                    eval_mode="same_models",
                    ridge_alpha=float(alpha),
                    group_size=int(group_size),
                )
                m_hold, _ = mod.run_path2_grouped_pair_ridge_eval(
                    agg=agg,
                    models=matched_models,
                    y_true=y_true,
                    pairs=pairs,
                    train_worlds=train_worlds,
                    test_worlds=(test_world,),
                    eval_mode="holdout_models",
                    ridge_alpha=float(alpha),
                    group_size=int(group_size),
                )
                lowo_same.append(m_same)
                lowo_hold.append(m_hold)

            rows.append(
                {
                    "group_size": int(group_size),
                    "ridge_alpha": float(alpha),
                    "n_models_matched": int(len(matched_models)),
                    "n_pairs": int(len(pairs)),
                    "fixed_same_spearman": float(fixed_same["spearman"]),
                    "fixed_same_mae": float(fixed_same["mae"]),
                    "fixed_holdout_spearman": float(fixed_hold["spearman"]),
                    "fixed_holdout_mae": float(fixed_hold["mae"]),
                    "worldsplit_8_2_same_spearman_mean": float(np.mean([m["spearman"] for m in ws_same])),
                    "worldsplit_8_2_same_mae_mean": float(np.mean([m["mae"] for m in ws_same])),
                    "worldsplit_8_2_holdout_spearman_mean": float(np.mean([m["spearman"] for m in ws_hold])),
                    "worldsplit_8_2_holdout_mae_mean": float(np.mean([m["mae"] for m in ws_hold])),
                    "lowo_same_spearman_mean": float(np.mean([m["spearman"] for m in lowo_same])),
                    "lowo_same_mae_mean": float(np.mean([m["mae"] for m in lowo_same])),
                    "lowo_holdout_spearman_mean": float(np.mean([m["spearman"] for m in lowo_hold])),
                    "lowo_holdout_mae_mean": float(np.mean([m["mae"] for m in lowo_hold])),
                }
            )

    sweep_df = pd.DataFrame(rows).sort_values(["group_size", "ridge_alpha"])
    sweep_df.to_csv(output_csv, index=False)

    # Primary ranking: highest 8/2 holdout Spearman mean, then lower MAE.
    best = sweep_df.sort_values(
        ["worldsplit_8_2_holdout_spearman_mean", "worldsplit_8_2_holdout_mae_mean"],
        ascending=[False, True],
    ).iloc[0]
    output_json.write_text(json.dumps(best.to_dict(), indent=2))

    # Heatmap visual
    pivot = sweep_df.pivot(
        index="group_size",
        columns="ridge_alpha",
        values="worldsplit_8_2_holdout_spearman_mean",
    )
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    im = ax.imshow(pivot.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{c:g}" for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(v) for v in pivot.index])
    ax.set_xlabel("Ridge alpha")
    ax.set_ylabel("Path 2 group size (N worlds)")
    ax.set_title("Path 2 Sweep: Mean Spearman on 8/2 Holdout (LOO models)")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = float(pivot.values[i, j])
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8, color="white" if v < 0.45 else "black")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean Spearman (higher is better)")
    plt.tight_layout()
    plt.savefig(output_png, dpi=180)
    plt.close(fig)

    print(f"Saved sweep CSV: {output_csv}")
    print(f"Saved sweep heatmap: {output_png}")
    print(f"Saved best config JSON: {output_json}")
    print(
        "Best combo: "
        f"group_size={int(best['group_size'])}, "
        f"alpha={float(best['ridge_alpha'])}, "
        f"8/2 holdout mean rho={float(best['worldsplit_8_2_holdout_spearman_mean']):.4f}, "
        f"8/2 holdout mean MAE={float(best['worldsplit_8_2_holdout_mae_mean']):.4f}"
    )


if __name__ == "__main__":
    main()
