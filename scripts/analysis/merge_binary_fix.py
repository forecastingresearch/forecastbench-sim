#!/usr/bin/env python3
"""Merge binary-only rerun outputs (base and/or LoRA) into base-vs-LoRA files.

This generalizes merge_lora_binary_fix.py so either model can be replaced.
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


def merge_one_model(
    *,
    base_path: Path,
    fix_path: Path,
    model_id: str,
    note_key: str,
) -> tuple[int, int]:
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
        fix_pred = fix_row.get("predictions", {}).get(model_id)
        if fix_pred is None:
            continue
        q.setdefault("predictions", {})[model_id] = fix_pred
        replaced += 1

    model_ids = sorted(
        {
            mid
            for q in base_data.get("questions", [])
            for mid in q.get("predictions", {}).keys()
        }
    )
    base_data["model_results"] = compute_metrics(base_data.get("questions", []), model_ids)

    md = base_data.setdefault("metadata", {})
    md[note_key] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_file": str(fix_path),
        "replaced_binary_predictions": replaced,
        "binary_questions": binary_count,
        "model_id": model_id,
    }

    base_path.write_text(json.dumps(base_data, indent=2))
    return replaced, binary_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge binary-only fix files into base-vs-LoRA outputs")
    parser.add_argument("--base-dir", type=Path, default=Path("data/evaluations/heldout_8b"))
    parser.add_argument("--lora-fix-dir", type=Path, default=Path("data/evaluations/heldout_8b_lora_binary_fix"))
    parser.add_argument("--base-fix-dir", type=Path, default=Path("data/evaluations/heldout_8b_base_binary_fix"))
    parser.add_argument(
        "--backup-dir",
        type=Path,
        default=Path("data/evaluations/heldout_8b_backup_pre_binary_fix"),
    )
    parser.add_argument("--skip-lora", action="store_true")
    parser.add_argument("--skip-base", action="store_true")
    args = parser.parse_args()

    args.backup_dir.mkdir(parents=True, exist_ok=True)

    total_replaced = 0
    total_binary = 0
    files_done = 0

    for iv, cond in PAIRS:
        base_path = args.base_dir / f"{iv}_{cond}_base_vs_lora.json"
        if not base_path.exists():
            print(f"Skipping missing base file: {base_path}")
            continue

        backup_path = args.backup_dir / base_path.name
        if not backup_path.exists():
            shutil.copy2(base_path, backup_path)

        file_replaced = 0
        file_binary = None

        if not args.skip_lora:
            fix_path = args.lora_fix_dir / f"{iv}_{cond}_lora_binary_fix.json"
            if fix_path.exists():
                rep, bcount = merge_one_model(
                    base_path=base_path,
                    fix_path=fix_path,
                    model_id="openai/civbench-8b-lora",
                    note_key="lora_binary_fix_merged",
                )
                file_replaced += rep
                file_binary = bcount if file_binary is None else file_binary
            else:
                print(f"  Missing LoRA fix file: {fix_path}")

        if not args.skip_base:
            fix_path = args.base_fix_dir / f"{iv}_{cond}_base_binary_fix.json"
            if fix_path.exists():
                rep, bcount = merge_one_model(
                    base_path=base_path,
                    fix_path=fix_path,
                    model_id="openai/civbench-8b-base",
                    note_key="base_binary_fix_merged",
                )
                file_replaced += rep
                file_binary = bcount if file_binary is None else file_binary
            else:
                print(f"  Missing base fix file: {fix_path}")

        files_done += 1
        total_replaced += file_replaced
        total_binary += (file_binary or 0)
        print(f"{base_path.name}: replaced {file_replaced} binary predictions")

    print("\nDone.")
    print(f"Files processed: {files_done}")
    print(f"Total replacements (across selected models): {total_replaced}")
    print(f"Backup dir: {args.backup_dir}")


if __name__ == "__main__":
    main()
