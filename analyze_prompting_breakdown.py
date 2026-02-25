"""
Breakdown analysis of prompting experiment results.

Computes Brier scores and direction accuracy across:
1. Template (10 question types)
2. Horizon (resolution turn: 90, 120, 150, 180, 210, 240)
3. Template x Horizon pairs
4. Identifies which slices drive the most deviation from overall averages
"""

import json
import re
from collections import defaultdict

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
with open("data/results/prompting_experiment_20260223_142528.json") as f:
    data = json.load(f)

questions = data["questions"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def brier(p, truth_bool):
    """Brier score for a single forecast."""
    t = 1.0 if truth_bool else 0.0
    return (p - t) ** 2


def extract_horizon(q):
    """Extract the resolution turn number from the question text."""
    text = q.get("conditional_text") or q.get("bare_text") or ""
    m = re.search(r"turn (\d+)", text)
    if m:
        return int(m.group(1))
    return None


# Map of condition name -> how to get the probability from a question dict
CONDITIONS = {
    "control":      lambda q: q["control_conditional_p"],
    "independence": lambda q: q["baseline_p"],
    "two_step":     lambda q: q["two_step_p"],
    "effect_size":  lambda q: q["effect_size_p"],
}

TRUTH_KEY = "fork_truth"

# Short labels for templates
SHORT = {
    "conditional_city_count_comparative": "city_count",
    "conditional_government_at": "government",
    "conditional_population_comparative": "population",
    "conditional_score_comparative": "score_comp",
    "conditional_score_rank_1": "score_rank1",
    "conditional_tech_comparative": "tech_comp",
    "conditional_tech_discovered": "tech_disc",
    "conditional_territory_comparative": "territory",
    "conditional_treasury_comparative": "treasury",
    "conditional_wonder_completed": "wonder",
}


def direction_correct(q):
    """
    For two-step prompting: did two_step_p move closer to fork_truth
    than baseline_p (the independence/bare forecast)?

    This measures whether the two-step decomposition actually helped
    the model adjust in the right direction from the unconditional baseline.
    """
    ft = 1.0 if q["fork_truth"] else 0.0
    baseline_err = abs(q["baseline_p"] - ft)
    two_step_err = abs(q["two_step_p"] - ft)
    return two_step_err < baseline_err


def direction_correct_control(q):
    """
    Did control_conditional_p move closer to fork_truth than baseline_p?
    For comparison with two-step direction accuracy.
    """
    ft = 1.0 if q["fork_truth"] else 0.0
    baseline_err = abs(q["baseline_p"] - ft)
    control_err = abs(q["control_conditional_p"] - ft)
    return control_err < baseline_err


def direction_correct_effect(q):
    """
    Did effect_size_p move closer to fork_truth than baseline_p?
    """
    ft = 1.0 if q["fork_truth"] else 0.0
    baseline_err = abs(q["baseline_p"] - ft)
    effect_err = abs(q["effect_size_p"] - ft)
    return effect_err < baseline_err


# ---------------------------------------------------------------------------
# Compute per-question metrics
# ---------------------------------------------------------------------------

# Attach horizon to each question
for q in questions:
    q["_horizon"] = extract_horizon(q)
    q["_short_template"] = SHORT.get(q["template_id"], q["template_id"])

# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def aggregate(group_questions):
    """Return dict of condition -> mean Brier, plus direction accuracy metrics."""
    result = {}
    for cond_name, get_p in CONDITIONS.items():
        scores = [brier(get_p(q), q[TRUTH_KEY]) for q in group_questions]
        result[cond_name] = sum(scores) / len(scores) if scores else float("nan")

    # Direction accuracy: did each condition move p closer to fork_truth than baseline?
    for label, fn in [("two_step_dir_acc", direction_correct),
                      ("control_dir_acc", direction_correct_control),
                      ("effect_dir_acc", direction_correct_effect)]:
        hits = [fn(q) for q in group_questions]
        result[label] = sum(hits) / len(hits) if hits else float("nan")

    result["n"] = len(group_questions)
    return result


def group_by(questions, key_fn):
    """Group questions by a key function, return {key: [questions]}."""
    groups = defaultdict(list)
    for q in questions:
        k = key_fn(q)
        if k is not None:
            groups[k].append(q)
    return dict(sorted(groups.items()))


# ---------------------------------------------------------------------------
# Compute aggregates
# ---------------------------------------------------------------------------

overall = aggregate(questions)

by_template = {k: aggregate(qs) for k, qs in group_by(questions, lambda q: q["_short_template"]).items()}
by_horizon = {k: aggregate(qs) for k, qs in group_by(questions, lambda q: q["_horizon"]).items()}
by_pair = {k: aggregate(qs) for k, qs in group_by(questions, lambda q: (q["_short_template"], q["_horizon"])).items()}

# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

COND_ORDER = ["control", "independence", "two_step", "effect_size"]

def fmt(v, width=8):
    if isinstance(v, float):
        return f"{v:.3f}".rjust(width)
    return str(v).rjust(width)


DIR_ACC_COLS = [
    ("ctrl_dir%", "control_dir_acc"),
    ("2s_dir%", "two_step_dir_acc"),
    ("es_dir%", "effect_dir_acc"),
]

def print_table(title, data_dict, show_dir_acc=True):
    """Print a table of Brier scores per condition for each key."""
    print()
    print("=" * 110)
    print(f"  {title}")
    print("  (dir% = fraction of questions where condition moved p closer to fork_truth than baseline)")
    print("=" * 110)

    # Header
    hdr = "Slice".ljust(28) + "  n"
    for c in COND_ORDER:
        hdr += fmt(c, 12)
    if show_dir_acc:
        for label, _ in DIR_ACC_COLS:
            hdr += fmt(label, 10)
    print(hdr)
    print("-" * len(hdr))

    for key, agg in sorted(data_dict.items(), key=lambda x: str(x[0])):
        label = str(key).ljust(28)
        row = label + fmt(agg["n"], 3)
        for c in COND_ORDER:
            row += fmt(agg[c], 12)
        if show_dir_acc:
            for _, acc_key in DIR_ACC_COLS:
                row += fmt(agg[acc_key] * 100, 10)
        print(row)

    # Overall row
    print("-" * len(hdr))
    row = "OVERALL".ljust(28) + fmt(overall["n"], 3)
    for c in COND_ORDER:
        row += fmt(overall[c], 12)
    if show_dir_acc:
        for _, acc_key in DIR_ACC_COLS:
            row += fmt(overall[acc_key] * 100, 10)
    print(row)


print_table("BRIER SCORES BY TEMPLATE", by_template)
print_table("BRIER SCORES BY HORIZON", by_horizon)
print_table("BRIER SCORES BY TEMPLATE x HORIZON", by_pair)

# ---------------------------------------------------------------------------
# Deviation analysis — which slices deviate most from overall?
# ---------------------------------------------------------------------------

print()
print("=" * 90)
print("  DEVIATION FROM OVERALL AVERAGE (sorted by absolute deviation)")
print("=" * 90)

for cond in COND_ORDER:
    print(f"\n--- {cond.upper()} (overall = {overall[cond]:.3f}) ---")
    deviations = []
    for key, agg in by_pair.items():
        dev = agg[cond] - overall[cond]
        deviations.append((key, agg[cond], dev, agg["n"]))
    deviations.sort(key=lambda x: abs(x[2]), reverse=True)
    print(f"  {'Slice':<35} {'Brier':>7} {'Dev':>8} {'n':>3}")
    for key, val, dev, n in deviations[:15]:
        sign = "+" if dev > 0 else ""
        print(f"  {str(key):<35} {val:>7.3f} {sign}{dev:>7.3f} {n:>3}")

# Direction accuracy deviations for two-step
print(f"\n--- TWO-STEP DIRECTION ACCURACY (overall = {overall['two_step_dir_acc']*100:.1f}%) ---")
deviations = []
for key, agg in by_pair.items():
    dev = (agg["two_step_dir_acc"] - overall["two_step_dir_acc"]) * 100
    deviations.append((key, agg["two_step_dir_acc"] * 100, dev, agg["n"]))
deviations.sort(key=lambda x: abs(x[2]), reverse=True)
print(f"  {'Slice':<35} {'Acc%':>7} {'Dev%':>8} {'n':>3}")
for key, val, dev, n in deviations[:15]:
    sign = "+" if dev > 0 else ""
    print(f"  {str(key):<35} {val:>7.1f} {sign}{dev:>7.1f} {n:>3}")

# ---------------------------------------------------------------------------
# Template-level deviations (more useful since n=30 per template)
# ---------------------------------------------------------------------------

print()
print("=" * 90)
print("  TEMPLATE-LEVEL DEVIATIONS (n=30 each, more statistically stable)")
print("=" * 90)

for cond in COND_ORDER:
    print(f"\n--- {cond.upper()} (overall = {overall[cond]:.3f}) ---")
    deviations = []
    for key, agg in by_template.items():
        dev = agg[cond] - overall[cond]
        deviations.append((key, agg[cond], dev, agg["n"]))
    deviations.sort(key=lambda x: abs(x[2]), reverse=True)
    print(f"  {'Template':<28} {'Brier':>7} {'Dev':>8} {'n':>3}")
    for key, val, dev, n in deviations:
        sign = "+" if dev > 0 else ""
        print(f"  {key:<28} {val:>7.3f} {sign}{dev:>7.3f} {n:>3}")

print(f"\n--- TWO-STEP DIRECTION ACCURACY (overall = {overall['two_step_dir_acc']*100:.1f}%) ---")
deviations = []
for key, agg in by_template.items():
    dev = (agg["two_step_dir_acc"] - overall["two_step_dir_acc"]) * 100
    deviations.append((key, agg["two_step_dir_acc"] * 100, dev, agg["n"]))
deviations.sort(key=lambda x: abs(x[2]), reverse=True)
print(f"  {'Template':<28} {'Acc%':>7} {'Dev%':>8} {'n':>3}")
for key, val, dev, n in deviations:
    sign = "+" if dev > 0 else ""
    print(f"  {key:<28} {val:>7.1f} {sign}{dev:>7.1f} {n:>3}")

# ---------------------------------------------------------------------------
# Horizon-level deviations
# ---------------------------------------------------------------------------

print()
print("=" * 90)
print("  HORIZON-LEVEL DEVIATIONS")
print("=" * 90)

for cond in COND_ORDER:
    print(f"\n--- {cond.upper()} (overall = {overall[cond]:.3f}) ---")
    deviations = []
    for key, agg in by_horizon.items():
        dev = agg[cond] - overall[cond]
        deviations.append((key, agg[cond], dev, agg["n"]))
    deviations.sort(key=lambda x: abs(x[2]), reverse=True)
    print(f"  {'Horizon':<12} {'Brier':>7} {'Dev':>8} {'n':>3}")
    for key, val, dev, n in deviations:
        sign = "+" if dev > 0 else ""
        print(f"  turn {key:<6} {val:>7.3f} {sign}{dev:>7.3f} {n:>3}")

print(f"\n--- TWO-STEP DIRECTION ACCURACY (overall = {overall['two_step_dir_acc']*100:.1f}%) ---")
deviations = []
for key, agg in by_horizon.items():
    dev = (agg["two_step_dir_acc"] - overall["two_step_dir_acc"]) * 100
    deviations.append((key, agg["two_step_dir_acc"] * 100, dev, agg["n"]))
deviations.sort(key=lambda x: abs(x[2]), reverse=True)
print(f"  {'Horizon':<12} {'Acc%':>7} {'Dev%':>8} {'n':>3}")
for key, val, dev, n in deviations:
    sign = "+" if dev > 0 else ""
    print(f"  turn {key:<6} {val:>7.1f} {sign}{dev:>7.1f} {n:>3}")

# ---------------------------------------------------------------------------
# Summary: best and worst conditions per slice type
# ---------------------------------------------------------------------------

print()
print("=" * 90)
print("  SUMMARY: BEST CONDITION PER TEMPLATE (lowest Brier)")
print("=" * 90)
for key, agg in sorted(by_template.items()):
    best_cond = min(COND_ORDER, key=lambda c: agg[c])
    worst_cond = max(COND_ORDER, key=lambda c: agg[c])
    spread = agg[worst_cond] - agg[best_cond]
    print(f"  {key:<28} best={best_cond:<14} ({agg[best_cond]:.3f})  "
          f"worst={worst_cond:<14} ({agg[worst_cond]:.3f})  spread={spread:.3f}")

print()
print("=" * 90)
print("  SUMMARY: BEST CONDITION PER HORIZON (lowest Brier)")
print("=" * 90)
for key, agg in sorted(by_horizon.items()):
    best_cond = min(COND_ORDER, key=lambda c: agg[c])
    worst_cond = max(COND_ORDER, key=lambda c: agg[c])
    spread = agg[worst_cond] - agg[best_cond]
    print(f"  turn {key:<6} best={best_cond:<14} ({agg[best_cond]:.3f})  "
          f"worst={worst_cond:<14} ({agg[worst_cond]:.3f})  spread={spread:.3f}")

# ---------------------------------------------------------------------------
# Base rate analysis: how often is fork_truth True vs False?
# ---------------------------------------------------------------------------
print()
print("=" * 90)
print("  BASE RATES: fork_truth distribution per slice")
print("=" * 90)

print(f"\n  OVERALL: {sum(1 for q in questions if q['fork_truth'])}/{len(questions)} "
      f"= {sum(1 for q in questions if q['fork_truth'])/len(questions)*100:.1f}% True")

print(f"\n  By template:")
for key, qs in sorted(group_by(questions, lambda q: q["_short_template"]).items()):
    n_true = sum(1 for q in qs if q["fork_truth"])
    print(f"    {key:<28} {n_true}/{len(qs)} = {n_true/len(qs)*100:.1f}% True")

print(f"\n  By horizon:")
for key, qs in sorted(group_by(questions, lambda q: q["_horizon"]).items()):
    n_true = sum(1 for q in qs if q["fork_truth"])
    print(f"    turn {key:<6} {n_true}/{len(qs)} = {n_true/len(qs)*100:.1f}% True")
