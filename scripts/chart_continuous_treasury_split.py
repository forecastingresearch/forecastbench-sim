"""Chart: Continuous CRPS gap split into treasury vs non-treasury templates."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json
from collections import defaultdict

# --- Config ---
BASELINE_PATH = "data/results/republic_baseline_continuous_all.json"
CONDITIONAL_PATH = "data/results/republic_conditional_continuous_all.json"
OUTPUT_PATH = "/Users/elsehow/Projects/fri-vault/_artifacts/static/study3_treasury_split.pdf"

MODEL_SHORT_NAMES = {
    "openai/o3-2025-04-16": "o3",
    "openai/gpt-4.1-2025-04-14": "GPT-4.1",
    "openai/gpt-5-2025-08-07": "GPT-5",
    "openai/gpt-5-mini-2025-08-07": "GPT-5 mini",
    "google/gemini-2.5-pro": "Gemini 2.5 Pro",
    "google/gemini-2.5-flash": "Gemini 2.5 Flash",
    "anthropic/claude-opus-4-5-20251101": "Opus 4.5",
    "anthropic/claude-sonnet-4-5-20250929": "Sonnet 4.5",
    "google/gemini-3-pro-preview": "Gemini 3 Pro",
    "openai/gpt-5.1-2025-11-13": "GPT-5.1",
}

TAU_LEVELS = [0.10, 0.25, 0.50, 0.75, 0.90]


def crps_from_quantiles(quantiles, tau_levels, observation):
    """Compute CRPS approximation from quantile predictions."""
    total = 0.0
    for q, tau in zip(quantiles, tau_levels):
        error = observation - q
        if error >= 0:
            total += tau * error
        else:
            total += (tau - 1) * error
    return total / len(tau_levels)


def is_treasury(template_id):
    """Check if a template is the treasury template."""
    return "treasury" in template_id


def compute_crps_by_model_and_category(questions):
    """Compute mean CRPS per model for treasury and non-treasury questions.

    Returns dict: model_id -> {"treasury": [crps_values], "non_treasury": [crps_values]}
    """
    scores = defaultdict(lambda: {"treasury": [], "non_treasury": []})

    for q in questions:
        if q["question_type"] != "continuous":
            continue
        observation = q["ground_truth"]
        template_id = q["template_id"]
        category = "treasury" if is_treasury(template_id) else "non_treasury"

        for model_id, pred in q["predictions"].items():
            if pred.get("error") or pred.get("percentiles") is None:
                continue
            percentiles = pred["percentiles"]
            quantiles = [
                percentiles["p10"],
                percentiles["p25"],
                percentiles["p50"],
                percentiles["p75"],
                percentiles["p90"],
            ]
            crps = crps_from_quantiles(quantiles, TAU_LEVELS, observation)
            scores[model_id][category].append(crps)

    return scores


def mean_crps(values):
    """Compute mean of a list, or NaN if empty."""
    if not values:
        return float("nan")
    return sum(values) / len(values)


# --- Load data ---
with open(BASELINE_PATH) as f:
    baseline_data = json.load(f)
with open(CONDITIONAL_PATH) as f:
    conditional_data = json.load(f)

baseline_scores = compute_crps_by_model_and_category(baseline_data["questions"])
conditional_scores = compute_crps_by_model_and_category(conditional_data["questions"])

# --- Compute gaps ---
models = list(MODEL_SHORT_NAMES.keys())

rows = []
for model_id in models:
    short = MODEL_SHORT_NAMES[model_id]

    b_treasury = mean_crps(baseline_scores[model_id]["treasury"])
    c_treasury = mean_crps(conditional_scores[model_id]["treasury"])
    treasury_gap_pct = (c_treasury - b_treasury) / b_treasury * 100

    b_non = mean_crps(baseline_scores[model_id]["non_treasury"])
    c_non = mean_crps(conditional_scores[model_id]["non_treasury"])
    non_treasury_gap_pct = (c_non - b_non) / b_non * 100

    # Total gap for sorting
    b_all = mean_crps(
        baseline_scores[model_id]["treasury"] + baseline_scores[model_id]["non_treasury"]
    )
    c_all = mean_crps(
        conditional_scores[model_id]["treasury"] + conditional_scores[model_id]["non_treasury"]
    )
    total_gap_pct = (c_all - b_all) / b_all * 100

    rows.append({
        "model_id": model_id,
        "short_name": short,
        "b_treasury": b_treasury,
        "c_treasury": c_treasury,
        "treasury_gap_pct": treasury_gap_pct,
        "b_non_treasury": b_non,
        "c_non_treasury": c_non,
        "non_treasury_gap_pct": non_treasury_gap_pct,
        "total_gap_pct": total_gap_pct,
    })

# Sort by total gap ascending
rows.sort(key=lambda r: r["total_gap_pct"])

# --- Print data table ---
print(f"{'Model':<18} {'B-Treas':>9} {'C-Treas':>9} {'Treas Gap%':>11} "
      f"{'B-NonTr':>9} {'C-NonTr':>9} {'NonTr Gap%':>11} {'Total Gap%':>11}")
print("-" * 100)
for r in rows:
    print(f"{r['short_name']:<18} {r['b_treasury']:>9.1f} {r['c_treasury']:>9.1f} "
          f"{r['treasury_gap_pct']:>+10.1f}% {r['b_non_treasury']:>9.1f} "
          f"{r['c_non_treasury']:>9.1f} {r['non_treasury_gap_pct']:>+10.1f}% "
          f"{r['total_gap_pct']:>+10.1f}%")

# --- Chart ---
labels = [r["short_name"] for r in rows]
treasury_gaps = np.array([r["treasury_gap_pct"] for r in rows])
non_treasury_gaps = np.array([r["non_treasury_gap_pct"] for r in rows])

fig, ax = plt.subplots(figsize=(10, 5))

x = np.arange(len(labels))
width = 0.35

bars_treasury = ax.bar(
    x - width / 2, treasury_gaps, width,
    label="Treasury", color="#CC3333", edgecolor="none"
)
bars_non = ax.bar(
    x + width / 2, non_treasury_gaps, width,
    label="Non-treasury", color="#555555", edgecolor="none"
)

# Reference line at 0%
ax.axhline(y=0, color="black", linewidth=0.8, linestyle="-")

ax.set_ylabel("CRPS gap (%)", fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
ax.legend(frameon=False, fontsize=9, loc="upper left")

# Clean academic style
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(False)
ax.tick_params(axis="both", which="both", length=4)

plt.tight_layout()
plt.savefig(OUTPUT_PATH, bbox_inches="tight")
print(f"\nSaved chart to {OUTPUT_PATH}")
