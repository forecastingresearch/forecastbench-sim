#!/usr/bin/env python3
"""Analyze held-out 8B base-vs-LoRA evaluation outputs and generate plots.

This script is designed for results produced by:
  scripts/evaluate_llm_forecasts_parallel.py

It can read from either:
  1) A results tarball (e.g., heldout_8b_results_2026-03-03.tgz), or
  2) A directory containing the six expected result JSON files.
"""

from __future__ import annotations

import argparse
import csv
import json
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


EXPECTED_FILES = {
    ("republic", "baseline"): "republic_baseline_base_vs_lora.json",
    ("republic", "conditional"): "republic_conditional_base_vs_lora.json",
    ("gold500", "baseline"): "gold500_baseline_base_vs_lora.json",
    ("gold500", "conditional"): "gold500_conditional_base_vs_lora.json",
    ("mapmaking", "baseline"): "mapmaking_baseline_base_vs_lora.json",
    ("mapmaking", "conditional"): "mapmaking_conditional_base_vs_lora.json",
}


def _load_from_tar(results_tgz: Path) -> dict[str, dict[str, Any]]:
    data: dict[str, dict[str, Any]] = {}
    with tarfile.open(results_tgz, "r:gz") as tf:
        members = {m.name: m for m in tf.getmembers()}
        for filename in EXPECTED_FILES.values():
            member_path = f"data/evaluations/heldout_8b/{filename}"
            member = members.get(member_path)
            if member is None:
                continue
            with tf.extractfile(member) as f:
                assert f is not None
                data[filename] = json.load(f)
    return data


def _load_from_dir(results_dir: Path) -> dict[str, dict[str, Any]]:
    data: dict[str, dict[str, Any]] = {}
    for filename in EXPECTED_FILES.values():
        path = results_dir / filename
        if path.exists():
            data[filename] = json.loads(path.read_text())
    return data


def _continuous_valid(pred: dict[str, Any] | None) -> bool:
    if not pred or pred.get("error") is not None:
        return False
    p = pred.get("percentiles")
    if not isinstance(p, dict):
        return False
    return all(p.get(k) is not None for k in ("p10", "p25", "p50", "p75", "p90"))


def _binary_valid(pred: dict[str, Any] | None) -> bool:
    if not pred or pred.get("error") is not None:
        return False
    return pred.get("probability") is not None


def _collect_rows(eval_data: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], Counter]:
    rows: list[dict[str, Any]] = []
    failures = Counter()

    for (intervention, condition), filename in EXPECTED_FILES.items():
        result = eval_data.get(filename)
        if result is None:
            continue

        questions = result.get("questions", [])
        model_results = result.get("model_results", {})
        models = sorted(model_results.keys())

        for model_id in models:
            total_binary = 0
            total_cont = 0
            binary_ok = 0
            cont_ok = 0

            for q in questions:
                qtype = q.get("question_type", "binary")
                pred = q.get("predictions", {}).get(model_id, {})
                if qtype == "continuous":
                    total_cont += 1
                    if _continuous_valid(pred):
                        cont_ok += 1
                    else:
                        failures[(model_id, pred.get("error") or "missing_percentiles")] += 1
                else:
                    total_binary += 1
                    if _binary_valid(pred):
                        binary_ok += 1
                    else:
                        failures[(model_id, pred.get("error") or "missing_probability")] += 1

            binary_metrics = model_results.get(model_id, {}).get("binary", {})
            cont_metrics = model_results.get(model_id, {}).get("continuous", {})
            rows.append(
                {
                    "file": filename,
                    "intervention": intervention,
                    "condition": condition,
                    "model_id": model_id,
                    "binary_total": total_binary,
                    "binary_valid": binary_ok,
                    "binary_coverage": (binary_ok / total_binary) if total_binary else float("nan"),
                    "binary_brier": binary_metrics.get("brier_score"),
                    "continuous_total": total_cont,
                    "continuous_valid": cont_ok,
                    "continuous_coverage": (cont_ok / total_cont) if total_cont else float("nan"),
                    "continuous_crps": cont_metrics.get("crps"),
                    "continuous_mae": cont_metrics.get("mae"),
                }
            )

    return rows, failures


def _short_model(model_id: str) -> str:
    if model_id.endswith("civbench-8b-base"):
        return "base-8b"
    if model_id.endswith("civbench-8b-lora"):
        return "lora-8b"
    return model_id.split("/")[-1]


def _plot_grouped_bars(
    rows: list[dict[str, Any]],
    value_key: str,
    ylabel: str,
    title: str,
    outpath: Path,
) -> None:
    ordered_rows = sorted(rows, key=lambda r: (r["intervention"], r["condition"], r["model_id"]))
    labels = [f'{r["intervention"]}-{r["condition"]}\n{_short_model(r["model_id"])}' for r in ordered_rows]
    values = [r.get(value_key, float("nan")) for r in ordered_rows]
    colors = ["#3b82f6" if "base" in _short_model(r["model_id"]) else "#f97316" for r in ordered_rows]

    plt.figure(figsize=(14, 5))
    x = np.arange(len(labels))
    bars = plt.bar(x, values, color=colors, edgecolor="black", linewidth=0.8)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", alpha=0.25)

    for bar, val in zip(bars, values):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def _plot_failure_breakdown(failures: Counter, outpath: Path) -> None:
    by_model: dict[str, Counter] = {}
    for (model_id, reason), count in failures.items():
        by_model.setdefault(model_id, Counter())[reason] += count

    models = sorted(by_model.keys())
    reasons = sorted({reason for c in by_model.values() for reason in c.keys()})
    x = np.arange(len(models))
    bottom = np.zeros(len(models))

    plt.figure(figsize=(10, 5))
    color_map = {
        "missing_probability": "#ef4444",
        "missing_percentiles": "#f59e0b",
        "all_parse_failed": "#8b5cf6",
    }

    for reason in reasons:
        vals = np.array([by_model[m].get(reason, 0) for m in models], dtype=float)
        plt.bar(
            x,
            vals,
            bottom=bottom,
            label=reason,
            color=color_map.get(reason, None),
            edgecolor="black",
            linewidth=0.7,
        )
        bottom += vals

    plt.xticks(x, [_short_model(m) for m in models])
    plt.ylabel("Failed predictions")
    plt.title("Failure Breakdown by Model")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def _plot_conditional_gap(rows: list[dict[str, Any]], outpath: Path) -> None:
    # Gap = conditional brier - baseline brier, per intervention and model.
    key = {}
    for r in rows:
        key[(r["intervention"], r["condition"], r["model_id"])] = r

    interventions = ["republic", "gold500", "mapmaking"]
    model_ids = sorted({r["model_id"] for r in rows})
    x = np.arange(len(interventions))
    width = 0.35

    plt.figure(figsize=(10, 5))
    for i, model_id in enumerate(model_ids):
        gaps = []
        for iv in interventions:
            b = key.get((iv, "baseline", model_id), {}).get("binary_brier")
            c = key.get((iv, "conditional", model_id), {}).get("binary_brier")
            if b is None or c is None:
                gaps.append(np.nan)
            else:
                gaps.append(c - b)
        offset = (-width / 2) if i == 0 else (width / 2)
        plt.bar(
            x + offset,
            gaps,
            width=width,
            label=_short_model(model_id),
            edgecolor="black",
            linewidth=0.8,
        )

    plt.axhline(0.0, color="black", linewidth=1)
    plt.xticks(x, interventions)
    plt.ylabel("Conditional gap in binary Brier (cond - base)")
    plt.title("Conditional Forecasting Gap by Intervention")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze held-out 8B base-vs-LoRA results")
    parser.add_argument(
        "--results-tgz",
        type=Path,
        default=Path("/Users/jaeholee0404/Downloads/heldout_8b_results_2026-03-03.tgz"),
        help="Path to heldout results tarball",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Optional directory containing *_base_vs_lora.json files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/evaluations/heldout_8b_analysis"),
        help="Directory for summary outputs and plots",
    )
    args = parser.parse_args()

    if args.results_dir is not None:
        eval_data = _load_from_dir(args.results_dir)
    else:
        eval_data = _load_from_tar(args.results_tgz)

    if not eval_data:
        raise RuntimeError("No evaluation files loaded. Check --results-tgz or --results-dir.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows, failures = _collect_rows(eval_data)

    summary_json = args.output_dir / "summary.json"
    summary_csv = args.output_dir / "summary.csv"
    failures_json = args.output_dir / "failure_counts.json"

    with open(summary_json, "w") as f:
        json.dump(rows, f, indent=2)

    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with open(failures_json, "w") as f:
        payload = [
            {"model_id": m, "reason": reason, "count": count}
            for (m, reason), count in sorted(failures.items())
        ]
        json.dump(payload, f, indent=2)

    _plot_grouped_bars(
        rows,
        value_key="binary_brier",
        ylabel="Binary Brier (lower is better)",
        title="8B Held-out: Binary Brier by Intervention/Condition/Model",
        outpath=args.output_dir / "binary_brier.png",
    )
    _plot_grouped_bars(
        rows,
        value_key="binary_coverage",
        ylabel="Binary parse coverage",
        title="8B Held-out: Binary Parse Coverage",
        outpath=args.output_dir / "binary_coverage.png",
    )
    _plot_grouped_bars(
        rows,
        value_key="continuous_crps",
        ylabel="Continuous CRPS (lower is better)",
        title="8B Held-out: Continuous CRPS by Intervention/Condition/Model",
        outpath=args.output_dir / "continuous_crps.png",
    )
    _plot_grouped_bars(
        rows,
        value_key="continuous_mae",
        ylabel="Continuous MAE (lower is better)",
        title="8B Held-out: Continuous MAE by Intervention/Condition/Model",
        outpath=args.output_dir / "continuous_mae.png",
    )
    _plot_grouped_bars(
        rows,
        value_key="continuous_coverage",
        ylabel="Continuous parse coverage",
        title="8B Held-out: Continuous Parse Coverage",
        outpath=args.output_dir / "continuous_coverage.png",
    )
    _plot_conditional_gap(rows, args.output_dir / "conditional_gap_binary_brier.png")
    _plot_failure_breakdown(failures, args.output_dir / "failure_breakdown.png")

    print("Wrote analysis outputs:")
    print(f"  {summary_json}")
    print(f"  {summary_csv}")
    print(f"  {failures_json}")
    print(f"  {args.output_dir / 'binary_brier.png'}")
    print(f"  {args.output_dir / 'binary_coverage.png'}")
    print(f"  {args.output_dir / 'continuous_crps.png'}")
    print(f"  {args.output_dir / 'continuous_mae.png'}")
    print(f"  {args.output_dir / 'continuous_coverage.png'}")
    print(f"  {args.output_dir / 'conditional_gap_binary_brier.png'}")
    print(f"  {args.output_dir / 'failure_breakdown.png'}")


if __name__ == "__main__":
    main()
