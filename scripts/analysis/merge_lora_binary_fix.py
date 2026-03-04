#!/usr/bin/env python3
"""Merge LoRA binary-only rerun outputs into existing base-vs-LoRA eval files.

Expected inputs:
- Base files: data/evaluations/heldout_8b/*_base_vs_lora.json
- Fix files:  data/evaluations/heldout_8b_lora_binary_fix/*_lora_binary_fix.json

For each intervention/condition pair, this script:
1) Replaces LoRA binary predictions in base file with predictions from fix file
2) Recomputes model_results metrics (binary + continuous)
3) Writes results in place (with optional backup)
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from civrealm.metrics import (  # noqa: E402
    compute_aggregate_crps,
    compute_aggregate_mae,
    compute_brier_score,
    compute_calibration_error,
)


PAIRS = [
    ("republic", "baseline"),
    ("republic", "conditional"),
    ("gold500", "baseline"),
    ("gold500", "conditional"),
    ("mapmaking", "baseline"),
    ("mapmaking", "conditional"),
]


def compute_metrics(results: list[dict[str, Any]], model_ids: list[str]) -> dict[str, Any]:
    """Recompute model metrics using current predictions in `results`."""
    model_metrics: dict[str, Any] = {}

    for model_id in model_ids:
        binary_results = [r for r in results if r.get("question_type", "binary") == "binary"]
        continuous_results = [r for r in results if r.get("question_type") == "continuous"]

        # Binary
        preds: list[float] = []
        outs: list[bool] = []
        by_template: dict[str, tuple[list[float], list[bool]]] = defaultdict(lambda: ([], []))
        for row in binary_results:
            pred = row.get("predictions", {}).get(model_id, {})
            p = pred.get("probability")
            if p is None:
                continue
            gt = bool(row.get("ground_truth"))
            preds.append(float(p))
            outs.append(gt)
            tid = row.get("template_id", "unknown")
            by_template[tid][0].append(float(p))
            by_template[tid][1].append(gt)

        brier = compute_brier_score(preds, outs) if preds else float("nan")
        ece = compute_calibration_error(preds, outs) if preds else float("nan")
        brier_by_template = {
            t: compute_brier_score(tp, to)
            for t, (tp, to) in sorted(by_template.items())
            if tp
        }
        num_failures_binary = sum(
            1
            for row in binary_results
            if row.get("predictions", {}).get(model_id, {}).get("probability") is None
        )

        # Continuous
        all_percentiles: list[dict[str, float]] = []
        all_true: list[float] = []
        by_template_cont: dict[str, tuple[list[dict[str, float]], list[float]]] = defaultdict(lambda: ([], []))
        for row in continuous_results:
            pred = row.get("predictions", {}).get(model_id, {})
            pcts = pred.get("percentiles")
            if not isinstance(pcts, dict):
                continue
            if not all(k in pcts and pcts[k] is not None for k in ("p10", "p25", "p50", "p75", "p90")):
                continue
            all_percentiles.append(pcts)
            true_val = float(row.get("ground_truth"))
            all_true.append(true_val)
            tid = row.get("template_id", "unknown")
            by_template_cont[tid][0].append(pcts)
            by_template_cont[tid][1].append(true_val)

        if all_percentiles:
            crps = compute_aggregate_crps(all_percentiles, all_true)
            mae = compute_aggregate_mae([p["p50"] for p in all_percentiles], all_true)
        else:
            crps = float("nan")
            mae = float("nan")

        crps_by_template = {}
        mae_by_template = {}
        for t, (pcts_list, true_list) in sorted(by_template_cont.items()):
            crps_by_template[t] = compute_aggregate_crps(pcts_list, true_list)
            mae_by_template[t] = compute_aggregate_mae([p["p50"] for p in pcts_list], true_list)

        model_metrics[model_id] = {
            "binary": {
                "brier_score": brier,
                "brier_by_template": brier_by_template,
                "ece": ece,
                "num_predictions": len(preds),
                "num_failures": num_failures_binary,
            },
            "continuous": {
                "crps": crps,
                "mae": mae,
                "crps_by_template": crps_by_template,
                "mae_by_template": mae_by_template,
                "num_predictions": len(all_percentiles),
                "num_failures": len(continuous_results) - len(all_percentiles),
            },
        }

    return model_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge LoRA binary fix files into base-vs-LoRA outputs")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("data/evaluations/heldout_8b"),
        help="Directory with *_base_vs_lora.json files",
    )
    parser.add_argument(
        "--fix-dir",
        type=Path,
        default=Path("data/evaluations/heldout_8b_lora_binary_fix"),
        help="Directory with *_lora_binary_fix.json files",
    )
    parser.add_argument(
        "--lora-model-id",
        type=str,
        default="openai/civbench-8b-lora",
        help="Model ID to replace from fix files",
    )
    parser.add_argument(
        "--backup-dir",
        type=Path,
        default=Path("data/evaluations/heldout_8b_backup_pre_lora_binary_fix"),
        help="Backup directory for original base files",
    )
    args = parser.parse_args()

    args.backup_dir.mkdir(parents=True, exist_ok=True)

    total_replaced = 0
    total_binary = 0
    merged_files = 0

    for intervention, condition in PAIRS:
        base_path = args.base_dir / f"{intervention}_{condition}_base_vs_lora.json"
        fix_path = args.fix_dir / f"{intervention}_{condition}_lora_binary_fix.json"

        if not base_path.exists():
            print(f"Skipping (missing base): {base_path}")
            continue
        if not fix_path.exists():
            print(f"Skipping (missing fix): {fix_path}")
            continue

        # Backup original once
        backup_path = args.backup_dir / base_path.name
        if not backup_path.exists():
            shutil.copy2(base_path, backup_path)

        base_data = json.loads(base_path.read_text())
        fix_data = json.loads(fix_path.read_text())

        fix_by_qid = {
            q.get("question_id"): q
            for q in fix_data.get("questions", [])
            if q.get("question_type", "binary") == "binary"
        }

        replaced = 0
        binary_count = 0
        for q in base_data.get("questions", []):
            if q.get("question_type", "binary") != "binary":
                continue
            binary_count += 1
            qid = q.get("question_id")
            fix_row = fix_by_qid.get(qid)
            if not fix_row:
                continue
            fix_pred = fix_row.get("predictions", {}).get(args.lora_model_id)
            if fix_pred is None:
                continue
            q.setdefault("predictions", {})[args.lora_model_id] = fix_pred
            replaced += 1

        # Recompute model metrics
        model_ids = sorted(
            {
                mid
                for q in base_data.get("questions", [])
                for mid in q.get("predictions", {}).keys()
            }
        )
        base_data["model_results"] = compute_metrics(base_data.get("questions", []), model_ids)

        # Add merge note
        md = base_data.setdefault("metadata", {})
        md["lora_binary_fix_merged"] = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "source_file": str(fix_path),
            "replaced_binary_predictions": replaced,
            "binary_questions": binary_count,
            "lora_model_id": args.lora_model_id,
        }

        base_path.write_text(json.dumps(base_data, indent=2))

        print(
            f"{base_path.name}: replaced {replaced}/{binary_count} LoRA binary predictions"
        )

        total_replaced += replaced
        total_binary += binary_count
        merged_files += 1

    print("\nDone.")
    print(f"Files merged: {merged_files}")
    print(f"Total replacements: {total_replaced}/{total_binary}")
    print(f"Backups in: {args.backup_dir}")


if __name__ == "__main__":
    main()
