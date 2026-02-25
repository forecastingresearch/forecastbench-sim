#!/usr/bin/env python3
"""
Per-template and per-horizon directionality analysis.

For each matched question pair where fork_truth != baseline_truth:
  - Does the model shift its prediction in the right direction?
  - How much of the needed shift does it capture?

Breaks down by template_id and by resolution horizon.
"""

import json
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "results"

DATASETS = {
    "republic": {
        "baseline": DATA_DIR / "republic_baseline_binary_all.json",
        "conditional": DATA_DIR / "republic_conditional_binary_all.json",
    },
    "gold500": {
        "baseline": DATA_DIR / "gold500_baseline_binary_all.json",
        "conditional": DATA_DIR / "gold500_conditional_binary_all.json",
    },
}


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def short_name(model_id: str) -> str:
    parts = model_id.split("/")
    name = parts[-1] if len(parts) > 1 else model_id
    for suffix in ["-20251101", "-20250929", "-2025-04-16", "-2025-04-14",
                   "-2025-08-07", "-2025-11-13"]:
        name = name.replace(suffix, "")
    return name


def extract_template(template_id: str) -> str:
    """Strip 'conditional_' prefix to get base template."""
    return template_id.replace("conditional_", "")


def extract_horizon(question_text: str) -> int | None:
    """Extract resolution turn from question text (e.g., 'at turn 90')."""
    import re
    m = re.search(r"at turn (\d+)", question_text)
    if m:
        return int(m.group(1))
    return None


def analyze_dataset(name: str, baseline_path: Path, conditional_path: Path):
    baseline_data = load_json(baseline_path)
    conditional_data = load_json(conditional_path)

    baseline_by_id = {q["question_id"]: q for q in baseline_data["questions"]}

    # Collect per matched pair: (model, template, horizon, direction_correct, magnitude_fraction)
    records = []

    for cond_q in conditional_data["questions"]:
        if cond_q["question_type"] != "binary":
            continue

        base_id = cond_q["question_id"].replace("_intervention", "")
        base_q = baseline_by_id.get(base_id)
        if base_q is None:
            continue

        baseline_truth = base_q["ground_truth"]
        fork_truth = cond_q["ground_truth"]
        template = extract_template(base_q.get("template_id", "unknown"))
        horizon = extract_horizon(cond_q.get("question_text", ""))

        for model_id, cond_pred in cond_q["predictions"].items():
            base_pred = base_q["predictions"].get(model_id)
            if base_pred is None:
                continue

            p_base = base_pred["probability"]
            p_cond = cond_pred["probability"]

            if p_base is None or p_cond is None:
                continue

            truth_changed = baseline_truth != fork_truth
            pred_delta = p_cond - p_base

            # Direction: did truth go up or down?
            if truth_changed:
                truth_direction = 1 if fork_truth else -1  # True=1, False=0
                # If baseline was True and fork is False, truth went down (-1)
                # If baseline was False and fork is True, truth went up (+1)

                if abs(pred_delta) < 1e-6:
                    direction_status = "no_pred_change"
                elif (pred_delta > 0 and truth_direction > 0) or (pred_delta < 0 and truth_direction < 0):
                    direction_status = "correct"
                else:
                    direction_status = "wrong"

                # Magnitude: what fraction of the needed shift did the model capture?
                needed_shift = float(fork_truth) - float(baseline_truth)  # +1 or -1
                magnitude_fraction = pred_delta / needed_shift if needed_shift != 0 else 0
            else:
                direction_status = "no_truth_change"
                magnitude_fraction = None

            records.append({
                "model": model_id,
                "template": template,
                "horizon": horizon,
                "truth_changed": truth_changed,
                "direction_status": direction_status,
                "pred_delta": pred_delta,
                "magnitude_fraction": magnitude_fraction,
                "baseline_truth": baseline_truth,
                "fork_truth": fork_truth,
            })

    return records


def print_template_table(records, dataset_name):
    """Print direction accuracy by template, aggregated across all models."""
    print(f"\n{'='*80}")
    print(f"PER-TEMPLATE DIRECTIONALITY: {dataset_name.upper()}")
    print(f"{'='*80}")

    # Group by template
    by_template = defaultdict(lambda: {"correct": 0, "wrong": 0, "no_pred_change": 0, "no_truth_change": 0, "mag_fracs": []})

    for r in records:
        t = by_template[r["template"]]
        if r["direction_status"] == "correct":
            t["correct"] += 1
            t["mag_fracs"].append(r["magnitude_fraction"])
        elif r["direction_status"] == "wrong":
            t["wrong"] += 1
            t["mag_fracs"].append(r["magnitude_fraction"])
        elif r["direction_status"] == "no_pred_change":
            t["no_pred_change"] += 1
        else:
            t["no_truth_change"] += 1

    # Compute base rate info
    br_by_template = defaultdict(lambda: {"bl_true": 0, "fk_true": 0, "total": 0})
    seen = set()
    for r in records:
        key = (r["template"], id(r))  # unique per record
        br = br_by_template[r["template"]]
        br["total"] += 1
        if r["baseline_truth"]:
            br["bl_true"] += 1
        if r["fork_truth"]:
            br["fk_true"] += 1

    templates = sorted(by_template.keys())

    header = f"{'Template':<25} {'Correct':>7} {'Wrong':>7} {'NoPred':>7} {'NoΔTruth':>8} {'DirAcc':>7} {'MedMag':>7} {'BLBR':>6} {'FKBR':>6} {'ΔBR':>6}"
    print(header)
    print("-" * len(header))

    total_correct = 0
    total_wrong = 0
    total_no_pred = 0
    total_no_truth = 0
    all_mag_fracs = []

    for template in templates:
        t = by_template[template]
        total_correct += t["correct"]
        total_wrong += t["wrong"]
        total_no_pred += t["no_pred_change"]
        total_no_truth += t["no_truth_change"]
        all_mag_fracs.extend(t["mag_fracs"])

        dir_denom = t["correct"] + t["wrong"]
        dir_acc = t["correct"] / dir_denom if dir_denom > 0 else float("nan")
        med_mag = sorted(t["mag_fracs"])[len(t["mag_fracs"]) // 2] if t["mag_fracs"] else float("nan")

        br = br_by_template[template]
        bl_br = br["bl_true"] / br["total"] if br["total"] > 0 else 0
        fk_br = br["fk_true"] / br["total"] if br["total"] > 0 else 0
        delta_br = fk_br - bl_br

        print(f"{template:<25} {t['correct']:>7} {t['wrong']:>7} {t['no_pred_change']:>7} {t['no_truth_change']:>8} {dir_acc:>6.1%} {med_mag:>7.3f} {bl_br:>6.3f} {fk_br:>6.3f} {delta_br:>+6.3f}")

    # Totals
    total_denom = total_correct + total_wrong
    total_dir_acc = total_correct / total_denom if total_denom > 0 else float("nan")
    total_med_mag = sorted(all_mag_fracs)[len(all_mag_fracs) // 2] if all_mag_fracs else float("nan")
    print("-" * len(header))
    print(f"{'TOTAL':<25} {total_correct:>7} {total_wrong:>7} {total_no_pred:>7} {total_no_truth:>8} {total_dir_acc:>6.1%} {total_med_mag:>7.3f}")


def print_horizon_table(records, dataset_name):
    """Print direction accuracy by horizon, aggregated across all models."""
    print(f"\n{'='*80}")
    print(f"PER-HORIZON DIRECTIONALITY: {dataset_name.upper()}")
    print(f"{'='*80}")

    by_horizon = defaultdict(lambda: {"correct": 0, "wrong": 0, "no_pred_change": 0, "no_truth_change": 0, "mag_fracs": []})

    for r in records:
        h = r["horizon"]
        if h is None:
            continue
        t = by_horizon[h]
        if r["direction_status"] == "correct":
            t["correct"] += 1
            t["mag_fracs"].append(r["magnitude_fraction"])
        elif r["direction_status"] == "wrong":
            t["wrong"] += 1
            t["mag_fracs"].append(r["magnitude_fraction"])
        elif r["direction_status"] == "no_pred_change":
            t["no_pred_change"] += 1
        else:
            t["no_truth_change"] += 1

    horizons = sorted(by_horizon.keys())

    header = f"{'Horizon':>8} {'Correct':>7} {'Wrong':>7} {'NoPred':>7} {'NoΔTruth':>8} {'DirAcc':>7} {'MedMag':>7}"
    print(header)
    print("-" * len(header))

    for h in horizons:
        t = by_horizon[h]
        dir_denom = t["correct"] + t["wrong"]
        dir_acc = t["correct"] / dir_denom if dir_denom > 0 else float("nan")
        med_mag = sorted(t["mag_fracs"])[len(t["mag_fracs"]) // 2] if t["mag_fracs"] else float("nan")
        print(f"{h:>8} {t['correct']:>7} {t['wrong']:>7} {t['no_pred_change']:>7} {t['no_truth_change']:>8} {dir_acc:>6.1%} {med_mag:>7.3f}")


def print_per_model_template_table(records, dataset_name):
    """Print direction accuracy by template for each model."""
    print(f"\n{'='*80}")
    print(f"PER-MODEL × TEMPLATE DIRECTIONALITY: {dataset_name.upper()}")
    print(f"{'='*80}")

    # Get all templates and models
    templates = sorted(set(r["template"] for r in records))
    models = sorted(set(r["model"] for r in records))

    # Group by (model, template)
    by_mt = defaultdict(lambda: {"correct": 0, "wrong": 0})

    for r in records:
        if r["direction_status"] in ("correct", "wrong"):
            by_mt[(r["model"], r["template"])][r["direction_status"]] += 1

    # Print header
    name_width = max(len(short_name(m)) for m in models)
    # Abbreviate templates
    tmpl_abbrevs = {t: t.replace("_comparative", "").replace("_completed", "").replace("_discovered", "_disc").replace("_at", "") for t in templates}
    tmpl_width = max(len(tmpl_abbrevs[t]) for t in templates)

    header_parts = [f"{'Model':<{name_width}}"]
    for t in templates:
        header_parts.append(f"{tmpl_abbrevs[t]:>{max(tmpl_width, 7)}}")
    header_parts.append(f"{'Overall':>8}")
    header = "  ".join(header_parts)
    print(header)
    print("-" * len(header))

    for model_id in models:
        parts = [f"{short_name(model_id):<{name_width}}"]
        total_c, total_w = 0, 0
        for t in templates:
            d = by_mt[(model_id, t)]
            c, w = d["correct"], d["wrong"]
            total_c += c
            total_w += w
            denom = c + w
            if denom > 0:
                acc = c / denom
                parts.append(f"{acc:>{max(tmpl_width, 7)}.1%}")
            else:
                parts.append(f"{'—':>{max(tmpl_width, 7)}}")
        overall = total_c / (total_c + total_w) if (total_c + total_w) > 0 else float("nan")
        parts.append(f"{overall:>8.1%}")
        print("  ".join(parts))


def main():
    for dataset_name, paths in DATASETS.items():
        print(f"\n\n{'#'*80}")
        print(f"# DATASET: {dataset_name.upper()}")
        print(f"{'#'*80}")

        records = analyze_dataset(dataset_name, paths["baseline"], paths["conditional"])
        print(f"\nTotal records: {len(records)}")

        print_template_table(records, dataset_name)
        print_horizon_table(records, dataset_name)
        print_per_model_template_table(records, dataset_name)


if __name__ == "__main__":
    main()
