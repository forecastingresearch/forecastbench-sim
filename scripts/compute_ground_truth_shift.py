"""
Compute how much the Republic intervention changes ground truth values.

For continuous questions: compute absolute and relative differences by template.
For binary questions: compute fraction of answers that flip by template.

This tells us whether the small continuous CRPS gap is because the intervention
doesn't actually change non-treasury values (nothing different to predict) vs
models being good at forward simulation.
"""

import json
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "results"


def load_json(filename: str) -> dict:
    with open(DATA_DIR / filename) as f:
        return json.load(f)


def match_questions(baseline_qs: list, conditional_qs: list) -> list:
    """Match baseline and conditional questions by (question_id, game_id).

    Conditional question_ids have '_intervention' suffix stripped for matching.
    Returns list of (baseline_q, conditional_q) tuples.
    """
    base_map = {}
    for q in baseline_qs:
        key = (q["question_id"], q["game_id"])
        base_map[key] = q

    pairs = []
    for q in conditional_qs:
        stripped_id = q["question_id"].replace("_intervention", "")
        key = (stripped_id, q["game_id"])
        if key in base_map:
            pairs.append((base_map[key], q))
    return pairs


def base_template(template_id: str) -> str:
    """Strip 'conditional_' prefix to get base template name."""
    return template_id.replace("conditional_", "")


def analyze_continuous():
    """Analyze how much ground truth changes for continuous questions."""
    baseline = load_json("republic_baseline_continuous_all.json")
    conditional = load_json("republic_conditional_continuous_all.json")

    pairs = match_questions(baseline["questions"], conditional["questions"])
    print(f"Continuous: {len(pairs)} matched question pairs")
    print()

    # Group by template
    by_template = defaultdict(list)
    for bq, cq in pairs:
        template = base_template(bq["template_id"])
        b_val = float(bq["ground_truth"])
        c_val = float(cq["ground_truth"])
        abs_diff = abs(b_val - c_val)
        # Relative diff: avoid division by zero
        if b_val != 0:
            rel_diff = abs_diff / abs(b_val) * 100
        else:
            rel_diff = float("inf") if abs_diff > 0 else 0.0
        by_template[template].append({
            "abs_diff": abs_diff,
            "rel_diff": rel_diff,
            "b_val": b_val,
            "c_val": c_val,
            "differs": b_val != c_val,
            "differs_10pct": rel_diff > 10,
        })

    # Print table
    header = f"{'Template':<28} {'N':>4} {'Mean |Δ|':>10} {'Mean Δ%':>10} {'Frac ≠0':>10} {'Frac >10%':>10} {'Mean Base':>12} {'Mean Fork':>12}"
    print(header)
    print("-" * len(header))

    all_entries = []
    for template in sorted(by_template.keys()):
        entries = by_template[template]
        n = len(entries)
        mean_abs = sum(e["abs_diff"] for e in entries) / n
        # Filter out inf for relative diff mean
        finite_rels = [e["rel_diff"] for e in entries if e["rel_diff"] != float("inf")]
        mean_rel = sum(finite_rels) / len(finite_rels) if finite_rels else float("nan")
        frac_diff = sum(1 for e in entries if e["differs"]) / n
        frac_10pct = sum(1 for e in entries if e["differs_10pct"]) / n
        mean_base = sum(e["b_val"] for e in entries) / n
        mean_fork = sum(e["c_val"] for e in entries) / n

        print(f"{template:<28} {n:>4} {mean_abs:>10.1f} {mean_rel:>9.1f}% {frac_diff:>10.1%} {frac_10pct:>10.1%} {mean_base:>12.1f} {mean_fork:>12.1f}")
        all_entries.extend(entries)

    # Overall
    n = len(all_entries)
    mean_abs = sum(e["abs_diff"] for e in all_entries) / n
    finite_rels = [e["rel_diff"] for e in all_entries if e["rel_diff"] != float("inf")]
    mean_rel = sum(finite_rels) / len(finite_rels) if finite_rels else float("nan")
    frac_diff = sum(1 for e in all_entries if e["differs"]) / n
    frac_10pct = sum(1 for e in all_entries if e["differs_10pct"]) / n
    mean_base = sum(e["b_val"] for e in all_entries) / n
    mean_fork = sum(e["c_val"] for e in all_entries) / n
    print("-" * len(header))
    print(f"{'OVERALL':<28} {n:>4} {mean_abs:>10.1f} {mean_rel:>9.1f}% {frac_diff:>10.1%} {frac_10pct:>10.1%} {mean_base:>12.1f} {mean_fork:>12.1f}")

    # Also print non-treasury overall
    non_treasury = [e for t, entries in by_template.items() for e in entries if "treasury" not in t]
    n_nt = len(non_treasury)
    if n_nt > 0:
        mean_abs_nt = sum(e["abs_diff"] for e in non_treasury) / n_nt
        finite_rels_nt = [e["rel_diff"] for e in non_treasury if e["rel_diff"] != float("inf")]
        mean_rel_nt = sum(finite_rels_nt) / len(finite_rels_nt) if finite_rels_nt else float("nan")
        frac_diff_nt = sum(1 for e in non_treasury if e["differs"]) / n_nt
        frac_10pct_nt = sum(1 for e in non_treasury if e["differs_10pct"]) / n_nt
        mean_base_nt = sum(e["b_val"] for e in non_treasury) / n_nt
        mean_fork_nt = sum(e["c_val"] for e in non_treasury) / n_nt
        print(f"{'NON-TREASURY':<28} {n_nt:>4} {mean_abs_nt:>10.1f} {mean_rel_nt:>9.1f}% {frac_diff_nt:>10.1%} {frac_10pct_nt:>10.1%} {mean_base_nt:>12.1f} {mean_fork_nt:>12.1f}")


def analyze_binary():
    """Analyze how much ground truth changes for binary questions."""
    baseline = load_json("republic_baseline_binary_all.json")
    conditional = load_json("republic_conditional_binary_all.json")

    pairs = match_questions(baseline["questions"], conditional["questions"])
    print(f"\nBinary: {len(pairs)} matched question pairs")
    print()

    # Group by template
    by_template = defaultdict(list)
    for bq, cq in pairs:
        template = base_template(bq["template_id"])
        b_val = bq["ground_truth"]
        c_val = cq["ground_truth"]
        by_template[template].append({
            "flipped": b_val != c_val,
            "b_val": b_val,
            "c_val": c_val,
        })

    header = f"{'Template':<28} {'N':>5} {'Flipped':>8} {'Flip %':>8} {'Base T%':>8} {'Fork T%':>8}"
    print(header)
    print("-" * len(header))

    all_entries = []
    for template in sorted(by_template.keys()):
        entries = by_template[template]
        n = len(entries)
        n_flip = sum(1 for e in entries if e["flipped"])
        flip_pct = n_flip / n * 100
        base_true_pct = sum(1 for e in entries if e["b_val"]) / n * 100
        fork_true_pct = sum(1 for e in entries if e["c_val"]) / n * 100
        print(f"{template:<28} {n:>5} {n_flip:>8} {flip_pct:>7.1f}% {base_true_pct:>7.1f}% {fork_true_pct:>7.1f}%")
        all_entries.extend(entries)

    # Overall
    n = len(all_entries)
    n_flip = sum(1 for e in all_entries if e["flipped"])
    flip_pct = n_flip / n * 100
    base_true_pct = sum(1 for e in all_entries if e["b_val"]) / n * 100
    fork_true_pct = sum(1 for e in all_entries if e["c_val"]) / n * 100
    print("-" * len(header))
    print(f"{'OVERALL':<28} {n:>5} {n_flip:>8} {flip_pct:>7.1f}% {base_true_pct:>7.1f}% {fork_true_pct:>7.1f}%")

    # Non-treasury overall
    non_treasury = [e for t, entries in by_template.items() for e in entries if "treasury" not in t]
    n_nt = len(non_treasury)
    if n_nt > 0:
        n_flip_nt = sum(1 for e in non_treasury if e["flipped"])
        flip_pct_nt = n_flip_nt / n_nt * 100
        base_true_nt = sum(1 for e in non_treasury if e["b_val"]) / n_nt * 100
        fork_true_nt = sum(1 for e in non_treasury if e["c_val"]) / n_nt * 100
        print(f"{'NON-TREASURY':<28} {n_nt:>5} {n_flip_nt:>8} {flip_pct_nt:>7.1f}% {base_true_nt:>7.1f}% {fork_true_nt:>7.1f}%")

    # Treasury-related only
    treasury = [e for t, entries in by_template.items() for e in entries if "treasury" in t]
    n_t = len(treasury)
    if n_t > 0:
        n_flip_t = sum(1 for e in treasury if e["flipped"])
        flip_pct_t = n_flip_t / n_t * 100
        base_true_t = sum(1 for e in treasury if e["b_val"]) / n_t * 100
        fork_true_t = sum(1 for e in treasury if e["c_val"]) / n_t * 100
        print(f"{'TREASURY ONLY':<28} {n_t:>5} {n_flip_t:>8} {flip_pct_t:>7.1f}% {base_true_t:>7.1f}% {fork_true_t:>7.1f}%")


if __name__ == "__main__":
    print("=" * 80)
    print("GROUND TRUTH SHIFT ANALYSIS: Republic Intervention")
    print("Does the intervention actually change absolute values by template?")
    print("=" * 80)
    print()

    print("─" * 80)
    print("CONTINUOUS QUESTIONS: Absolute value differences")
    print("─" * 80)
    analyze_continuous()

    print()
    print("─" * 80)
    print("BINARY QUESTIONS: Answer flip rates")
    print("─" * 80)
    analyze_binary()

    print()
    print("=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print("""
If non-treasury continuous templates show low Frac≠0 and low Mean Δ%, then the
Republic intervention barely changes those values. The small CRPS gap would then
be because there's nothing different to predict — not because models are good at
forward simulation under intervention.

Compare with binary flip rates: if binary templates show high flip rates for the
same underlying quantities (e.g., score_comparative flips a lot but
scores_continuous doesn't change much), that tells us rankings shuffle even when
absolute values stay similar.
""")
