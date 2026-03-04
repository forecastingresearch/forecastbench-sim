#!/usr/bin/env python3
"""
Correlate CivBench binary Brier score with ARC-AGI scores across models.

This script analyzes baseline/conditional binary evaluation runs and computes
Spearman correlations between model binary Brier and ARC-AGI scores
(both ARC-AGI-1 and ARC-AGI-2). It also generates scatter plots per run.

Usage:
    python3 scripts/analysis/correlate_binary_brier_arc_agi.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RUNS_DIR = DATA_DIR / "evaluations" / "runs"
ARC_CSV = DATA_DIR / "ARC-AGI.csv"
DEFAULT_OUT_DIR = DATA_DIR / "analysis" / "binary_arc_agi"


RUN_FILES = {
    "gold500_baseline_binary_all": RUNS_DIR / "gold500_baseline_binary_all.json",
    "gold500_conditional_binary_all": RUNS_DIR / "gold500_conditional_binary_all.json",
    "republic_baseline_binary_all": RUNS_DIR / "republic_baseline_binary_all.json",
    "republic_conditional_binary_all": RUNS_DIR / "republic_conditional_binary_all.json",
}

# Explicit mapping from evaluated model IDs to ARC-AGI table "AI System" rows.
MODEL_TO_ARC_SYSTEM = {
    "anthropic/claude-opus-4-5-20251101": "Opus 4.5 (Thinking, None)",
    "anthropic/claude-sonnet-4-5-20250929": "Claude Sonnet 4.5",
    "google/gemini-2.5-flash": "Gemini 2.5 Flash (Preview)",
    "google/gemini-2.5-pro": "Gemini 2.5 Pro (Preview)",
    "google/gemini-3-pro-preview": "Gemini 3 Pro",
    "openai/gpt-4.1-2025-04-14": "GPT-4.1",
    "openai/gpt-5-2025-08-07": "GPT-5 (Medium)",
    "openai/gpt-5-mini-2025-08-07": "GPT-5 Mini (Medium)",
    "openai/gpt-5.1-2025-11-13": "GPT-5.1 (Thinking, Medium)",
    "openai/o3-2025-04-16": "o3 (Medium)",
}

MODEL_SHORT_LABELS = {
    "anthropic/claude-opus-4-5-20251101": "opus4.5",
    "anthropic/claude-sonnet-4-5-20250929": "sonnet4.5",
    "google/gemini-2.5-flash": "gemini2.5flash",
    "google/gemini-2.5-pro": "gemini2.5pro",
    "google/gemini-3-pro-preview": "gemini3pro",
    "openai/gpt-4.1-2025-04-14": "gpt4.1",
    "openai/gpt-5-2025-08-07": "gpt5",
    "openai/gpt-5-mini-2025-08-07": "gpt5mini",
    "openai/gpt-5.1-2025-11-13": "gpt5.1",
    "openai/o3-2025-04-16": "o3",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Correlate binary Brier scores with ARC-AGI scores."
    )
    parser.add_argument(
        "--exclude-model",
        action="append",
        default=[],
        help="Model ID to exclude. Can be passed multiple times.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUT_DIR}).",
    )
    return parser.parse_args()


def parse_percent(raw: str) -> float:
    raw = (raw or "").strip()
    if not raw or raw.upper() == "N/A":
        return float("nan")
    if raw.endswith("%"):
        raw = raw[:-1]
    return float(raw)


def load_arc_scores(path: Path) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["AI System"]] = {
                "arc_agi_1": parse_percent(row["ARC-AGI-1"]),
                "arc_agi_2": parse_percent(row["ARC-AGI-2"]),
            }
    return out


def load_run_brier(path: Path) -> dict[str, float]:
    with path.open(encoding="utf-8") as f:
        run = json.load(f)
    model_results = run["model_results"]
    brier_by_model: dict[str, float] = {}
    for model_id, metrics in model_results.items():
        brier = metrics.get("binary", {}).get("brier_score", float("nan"))
        brier_by_model[model_id] = float(brier) if brier is not None else float("nan")
    return brier_by_model


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def compute_linear_r2(x: list[float], y: list[float]) -> float:
    """Compute ordinary least-squares R^2 for y ~ x."""
    if len(x) < 2:
        return float("nan")
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    slope, intercept = np.polyfit(x_arr, y_arr, 1)
    y_hat = slope * x_arr + intercept
    ss_res = float(np.sum((y_arr - y_hat) ** 2))
    ss_tot = float(np.sum((y_arr - np.mean(y_arr)) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - (ss_res / ss_tot)


def make_scatter(
    run_name: str,
    merged_rows: list[dict],
    summary_rows: list[dict],
    out_path: Path,
) -> None:
    fig = plt.figure(figsize=(14, 7), dpi=150)
    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[12, 2],
        left=0.06,
        right=0.98,
        top=0.86,
        bottom=0.08,
        wspace=0.22,
        hspace=0.30,
    )
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    note_axes = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    metric_specs = [("arc_agi_1", "ARC-AGI-1"), ("arc_agi_2", "ARC-AGI-2")]

    for ax, note_ax, (metric_key, metric_label) in zip(axes, note_axes, metric_specs):
        points = [
            r for r in merged_rows
            if not math.isnan(r[metric_key]) and not math.isnan(r["brier"])
        ]
        x = np.array([r[metric_key] for r in points], dtype=float)
        y = np.array([r["brier"] for r in points], dtype=float)

        ax.scatter(x, y, s=55, alpha=0.9, color="#1f77b4", edgecolor="black", linewidth=0.5)

        for r in points:
            ax.annotate(
                MODEL_SHORT_LABELS.get(r["model_id"], r["model_id"].split("/")[-1]),
                (r[metric_key], r["brier"]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=7,
            )

        if len(points) >= 2:
            slope, intercept = np.polyfit(x, y, 1)
            xline = np.array([float(np.min(x)), float(np.max(x))], dtype=float)
            yline = slope * xline + intercept
            ax.plot(xline, yline, linestyle="--", linewidth=1.4, color="#d62728")

        stats_row = next(
            s for s in summary_rows
            if s["run_name"] == run_name and s["arc_metric"] == metric_key
        )
        rho = stats_row["spearman_rho"]
        pval = stats_row["spearman_p_value"]
        n = stats_row["n_models"]
        r2 = stats_row["linear_r2"]
        r2_text = "nan" if math.isnan(r2) else f"{r2:.3f}"
        note_ax.axis("off")
        note_ax.text(
            0.5,
            0.5,
            f"Stats ({metric_label}): Spearman rho = {rho:.3f} | p = {pval:.4g} | R^2 = {r2_text} | n = {n}",
            va="center",
            ha="center",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#999999"},
        )

        ax.set_xlabel(f"{metric_label} score (%)")
        ax.set_ylabel("CivBench binary Brier (lower is better)")
        ax.set_title(f"{metric_label} vs Brier")
        ax.grid(True, alpha=0.25)

    fig.suptitle(f"{run_name}: ARC-AGI vs CivBench Binary Brier", fontsize=13, y=0.95)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    exclude_models = set(args.exclude_model)
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    arc_scores = load_arc_scores(ARC_CSV)
    merged_rows_all: list[dict] = []
    summary_rows: list[dict] = []
    model_to_arc_system_used = {
        model_id: arc_system
        for model_id, arc_system in MODEL_TO_ARC_SYSTEM.items()
        if model_id not in exclude_models
    }
    if not model_to_arc_system_used:
        raise RuntimeError("No models left after applying exclusions.")

    missing_map = []
    for model_id, arc_system in model_to_arc_system_used.items():
        if arc_system not in arc_scores:
            missing_map.append((model_id, arc_system))
    if missing_map:
        raise RuntimeError(f"Missing ARC mapping targets: {missing_map}")

    for run_name, run_path in RUN_FILES.items():
        brier_by_model = load_run_brier(run_path)
        merged_rows: list[dict] = []
        run_group = "conditional" if "conditional" in run_name else "unconditional"

        for model_id, arc_system in model_to_arc_system_used.items():
            if model_id not in brier_by_model:
                continue
            row = {
                "run_name": run_name,
                "run_group": run_group,
                "model_id": model_id,
                "arc_system": arc_system,
                "brier": brier_by_model[model_id],
                "arc_agi_1": arc_scores[arc_system]["arc_agi_1"],
                "arc_agi_2": arc_scores[arc_system]["arc_agi_2"],
            }
            merged_rows.append(row)
            merged_rows_all.append(row)

        for metric_key in ("arc_agi_1", "arc_agi_2"):
            points = [
                r for r in merged_rows
                if not math.isnan(r[metric_key]) and not math.isnan(r["brier"])
            ]
            x = [r[metric_key] for r in points]
            y = [r["brier"] for r in points]
            rho, pval = spearmanr(x, y) if len(points) >= 3 else (float("nan"), float("nan"))
            linear_r2 = compute_linear_r2(x, y)
            summary_rows.append(
                {
                    "run_name": run_name,
                    "run_group": run_group,
                    "arc_metric": metric_key,
                    "n_models": len(points),
                    "spearman_rho": float(rho),
                    "spearman_p_value": float(pval),
                    "linear_r2": float(linear_r2),
                }
            )

        run_points_csv = out_dir / f"{run_name}_model_points.csv"
        write_csv(
            run_points_csv,
            ["run_name", "run_group", "model_id", "arc_system", "brier", "arc_agi_1", "arc_agi_2"],
            merged_rows,
        )

        plot_path = out_dir / f"{run_name}_arc_vs_brier.png"
        make_scatter(run_name, merged_rows, summary_rows, plot_path)

    write_csv(
        out_dir / "arc_brier_spearman_summary.csv",
        ["run_name", "run_group", "arc_metric", "n_models", "spearman_rho", "spearman_p_value", "linear_r2"],
        summary_rows,
    )
    write_csv(
        out_dir / "arc_brier_all_points.csv",
        ["run_name", "run_group", "model_id", "arc_system", "brier", "arc_agi_1", "arc_agi_2"],
        merged_rows_all,
    )

    with (out_dir / "model_to_arc_mapping_used.json").open("w", encoding="utf-8") as f:
        json.dump(model_to_arc_system_used, f, indent=2)

    print(f"Wrote analysis outputs to: {out_dir}")
    print("Key file: arc_brier_spearman_summary.csv")


if __name__ == "__main__":
    main()
