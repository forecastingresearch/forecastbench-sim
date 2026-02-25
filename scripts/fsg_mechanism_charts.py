"""Generate two-panel chart for FSG Study 4: Structure of the Gap.

Left panel: Brier gap vs |ΔBR| per template (scatter + correlation)
Right panel: Brier gap vs temporal horizon (inverted-U)

Usage:
    uv run python scripts/fsg_mechanism_charts.py
"""

import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy import stats


MODEL_NAMES = {
    "openai/o3-2025-04-16": "o3",
    "openai/gpt-5.1-2025-11-13": "GPT-5.1",
    "openai/gpt-5-2025-08-07": "GPT-5",
    "openai/gpt-5-mini-2025-08-07": "GPT-5 mini",
    "openai/gpt-4.1-2025-04-14": "GPT-4.1",
    "google/gemini-3-pro-preview": "Gemini 3 Pro",
    "google/gemini-2.5-pro": "Gemini 2.5 Pro",
    "google/gemini-2.5-flash": "Gemini 2.5 Flash",
    "anthropic/claude-opus-4-5-20251101": "Opus 4.5",
    "anthropic/claude-sonnet-4-5-20250929": "Sonnet 4.5",
}

MODEL_ORDER = [
    "openai/o3-2025-04-16",
    "openai/gpt-5-mini-2025-08-07",
    "anthropic/claude-opus-4-5-20251101",
    "openai/gpt-5-2025-08-07",
    "anthropic/claude-sonnet-4-5-20250929",
    "openai/gpt-5.1-2025-11-13",
    "google/gemini-2.5-pro",
    "google/gemini-2.5-flash",
    "google/gemini-3-pro-preview",
    "openai/gpt-4.1-2025-04-14",
]

SHORT_TEMPLATES = {
    "treasury_comparative": "treasury",
    "territory_comparative": "territory",
    "score_comparative": "score",
    "score_rank_1": "score rank",
    "population_comparative": "population",
    "city_count_comparative": "city count",
    "tech_comparative": "tech",
    "tech_discovered": "tech disc",
    "government_at": "govt",
    "wonder_completed": "wonder",
}

INTERVENTIONS = {
    "republic": {
        "eval_baseline": "data/results/republic_baseline_binary_all.json",
        "eval_conditional": "data/results/republic_conditional_binary_all.json",
        "source_baseline": "data/conditional/republic/baseline/seed*/questions.json",
        "source_conditional": "data/conditional/republic/conditional/seed*/conditional_questions.json",
        "label": "Republic",
        "color": "#2166ac",
        "marker": "o",
    },
    "gold500": {
        "eval_baseline": "data/results/gold500_baseline_binary_all.json",
        "eval_conditional": "data/results/gold500_conditional_binary_all.json",
        "source_baseline": "data/conditional/gold500/baseline/seed*/questions.json",
        "source_conditional": "data/conditional/gold500/conditional/seed*/conditional_questions.json",
        "label": "Gold +500",
        "color": "#b2182b",
        "marker": "s",
    },
}


def build_source_lookup(source_glob):
    lookup = {}
    for qfile in glob.glob(source_glob):
        data = json.load(open(qfile))
        game_id = data["game_id"]
        for q in data["questions"]:
            if q["question_type"] == "binary":
                lookup[(game_id, q["question_id"])] = {
                    "horizon": q["horizon"],
                    "template_id": q["template_id"],
                }
    return lookup


def join_eval_with_source(eval_path, source_lookup):
    eval_data = json.load(open(eval_path))
    questions = []
    for eq in eval_data["questions"]:
        if eq["question_type"] != "binary":
            continue
        key = (eq["game_id"], eq["question_id"])
        if key in source_lookup:
            src = source_lookup[key]
            template = (
                src["template_id"]
                .replace("conditional_", "")
                .replace("null_conditional_", "")
            )
            questions.append({
                "question_id": eq["question_id"],
                "game_id": eq["game_id"],
                "template_id": template,
                "horizon": src["horizon"],
                "ground_truth": 1 if eq["ground_truth"] else 0,
                "predictions": eq["predictions"],
            })
    return questions


def compute_grouped_brier(questions, model_id, group_key):
    groups = defaultdict(list)
    for q in questions:
        pred = q["predictions"].get(model_id)
        if pred and pred.get("error") is None:
            groups[q[group_key]].append({
                "prob": pred["probability"],
                "ground_truth": q["ground_truth"],
            })
    results = {}
    for group_val, qs in sorted(groups.items()):
        n = len(qs)
        brier = sum((q["prob"] - q["ground_truth"]) ** 2 for q in qs) / n
        base_rate = sum(q["ground_truth"] for q in qs) / n
        results[group_val] = {"brier": brier, "n": n, "base_rate": base_rate}
    return results


def get_template_data(config, models):
    """Get per-template avg gap and ΔBR for one intervention."""
    bl_lookup = build_source_lookup(config["source_baseline"])
    cd_lookup = build_source_lookup(config["source_conditional"])
    bl_questions = join_eval_with_source(config["eval_baseline"], bl_lookup)
    cd_questions = join_eval_with_source(config["eval_conditional"], cd_lookup)

    cd_seeds = set(q["game_id"] for q in cd_questions)
    cd_horizons = set(q["horizon"] for q in cd_questions)
    bl_matched = [q for q in bl_questions
                  if q["game_id"] in cd_seeds and q["horizon"] in cd_horizons]

    templates = sorted(set(q["template_id"] for q in cd_questions))

    template_data = []
    for t in templates:
        gaps = []
        bl_brs = []
        cd_brs = []
        for model_id in models:
            bl_by_t = compute_grouped_brier(bl_matched, model_id, "template_id")
            cd_by_t = compute_grouped_brier(cd_questions, model_id, "template_id")
            if t in bl_by_t and t in cd_by_t:
                gaps.append(cd_by_t[t]["brier"] - bl_by_t[t]["brier"])
                bl_brs.append(bl_by_t[t]["base_rate"])
                cd_brs.append(cd_by_t[t]["base_rate"])

        if gaps:
            avg_gap = sum(gaps) / len(gaps)
            avg_bl_br = sum(bl_brs) / len(bl_brs)
            avg_cd_br = sum(cd_brs) / len(cd_brs)
            delta_br = avg_cd_br - avg_bl_br
            template_data.append({
                "template": t,
                "short": SHORT_TEMPLATES.get(t, t[:10]),
                "gap": avg_gap,
                "delta_br": delta_br,
                "abs_delta_br": abs(delta_br),
            })

    return template_data


def get_horizon_data(config, models):
    """Get per-horizon avg gap for one intervention."""
    bl_lookup = build_source_lookup(config["source_baseline"])
    cd_lookup = build_source_lookup(config["source_conditional"])
    bl_questions = join_eval_with_source(config["eval_baseline"], bl_lookup)
    cd_questions = join_eval_with_source(config["eval_conditional"], cd_lookup)

    cd_seeds = set(q["game_id"] for q in cd_questions)
    cd_horizons = set(q["horizon"] for q in cd_questions)
    bl_matched = [q for q in bl_questions
                  if q["game_id"] in cd_seeds and q["horizon"] in cd_horizons]

    horizons = sorted(set(q["horizon"] for q in cd_questions))

    horizon_data = []
    for h in horizons:
        gaps = []
        for model_id in models:
            bl_by_h = compute_grouped_brier(bl_matched, model_id, "horizon")
            cd_by_h = compute_grouped_brier(cd_questions, model_id, "horizon")
            if h in bl_by_h and h in cd_by_h:
                gaps.append(cd_by_h[h]["brier"] - bl_by_h[h]["brier"])
        if gaps:
            horizon_data.append({
                "horizon": h,
                "gap": sum(gaps) / len(gaps),
            })

    return horizon_data


def main():
    models = MODEL_ORDER

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # ── Left panel: Gap vs |ΔBR| scatter ──

    all_abs_dbr = []
    all_gaps = []

    for name, config in INTERVENTIONS.items():
        tdata = get_template_data(config, models)

        x = [d["abs_delta_br"] for d in tdata]
        y = [d["gap"] for d in tdata]

        ax1.scatter(x, y, c=config["color"], marker=config["marker"],
                    s=50, label=config["label"], zorder=3, edgecolors="white",
                    linewidth=0.5)

        # Label points
        for d in tdata:
            # Offset labels to avoid overlap
            offset_x = 0.012
            offset_y = 0.012
            # Special cases for crowded regions
            if d["short"] in ("govt",):
                offset_y = -0.025
            if d["short"] in ("wonder",) and name == "gold500":
                offset_x = -0.06
                offset_y = -0.025

            ax1.annotate(d["short"], (d["abs_delta_br"], d["gap"]),
                         xytext=(d["abs_delta_br"] + offset_x,
                                 d["gap"] + offset_y),
                         fontsize=6.5, color=config["color"], alpha=0.8)

        all_abs_dbr.extend(x)
        all_gaps.extend(y)

    # Regression line across both interventions
    x_arr = np.array(all_abs_dbr)
    y_arr = np.array(all_gaps)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_arr, y_arr)
    x_line = np.linspace(0, max(x_arr) * 1.05, 100)
    y_line = slope * x_line + intercept
    ax1.plot(x_line, y_line, color="gray", linewidth=1, linestyle="--", alpha=0.7,
             zorder=1)
    ax1.text(0.55, 0.08, f"r = {r_value:.2f}",
             transform=ax1.transAxes, fontsize=10, color="gray")

    ax1.set_xlabel("|ΔBR|  (base-rate divergence)", fontsize=10)
    ax1.set_ylabel("Brier gap  (conditional − baseline)", fontsize=10)
    ax1.set_title("Gap tracks ground-truth divergence", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=9, loc="upper left")
    ax1.axhline(y=0, color="black", linewidth=0.5, alpha=0.3)
    ax1.set_xlim(-0.02, max(all_abs_dbr) * 1.12)
    ax1.set_ylim(min(all_gaps) - 0.05, max(all_gaps) + 0.08)

    # ── Right panel: Gap vs horizon ──

    for name, config in INTERVENTIONS.items():
        hdata = get_horizon_data(config, models)
        x = list(range(len(hdata)))
        y = [d["gap"] for d in hdata]
        labels = [d["horizon"] for d in hdata]

        ax2.plot(x, y, color=config["color"], marker=config["marker"],
                 markersize=7, linewidth=2, label=config["label"],
                 markeredgecolor="white", markeredgewidth=0.5)

    ax2.set_xlabel("Temporal horizon", fontsize=10)
    ax2.set_ylabel("Brier gap  (conditional − baseline)", fontsize=10)
    ax2.set_title("Gap peaks where intervention\neffect is strongest", fontsize=11,
                  fontweight="bold")

    # Use Republic horizons for x-ticks (superset)
    rep_hdata = get_horizon_data(INTERVENTIONS["republic"], models)
    ax2.set_xticks(range(len(rep_hdata)))
    ax2.set_xticklabels([d["horizon"] for d in rep_hdata], fontsize=9)
    ax2.legend(fontsize=9, loc="upper right")
    ax2.axhline(y=0, color="black", linewidth=0.5, alpha=0.3)

    plt.tight_layout()
    out_path = "../fri-vault/_artifacts/static/study4_gap_structure.pdf"
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    print(f"Saved to {out_path}")

    # Also save PNG for quick preview
    out_png = "../fri-vault/_artifacts/static/study4_gap_structure.png"
    plt.savefig(out_png, bbox_inches="tight", dpi=150)
    print(f"Saved to {out_png}")


if __name__ == "__main__":
    main()
