#!/usr/bin/env python3
"""Plot conditional vs unconditional Brier scores from evaluation results."""

import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt

def compute_brier(questions: list[dict], model: str) -> float | None:
    """Compute Brier score for a set of questions."""
    preds, outcomes = [], []
    for q in questions:
        pred = q.get("predictions", {}).get(model, {}).get("probability")
        if pred is not None:
            preds.append(pred)
            outcomes.append(float(q["ground_truth"]))
    if not preds:
        return None
    return sum((p - o) ** 2 for p, o in zip(preds, outcomes)) / len(preds)

def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_conditional_comparison.py <results.json> [output.png]")
        sys.exit(1)

    results_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else results_path.with_suffix(".png")

    with open(results_path) as f:
        results = json.load(f)

    questions = results.get("questions", [])
    models = list(results.get("model_results", {}).keys())

    # Split by conditional status
    conditional = [q for q in questions if q["template_id"].startswith("conditional_")]
    unconditional = [q for q in questions if not q["template_id"].startswith("conditional_")]

    print(f"Conditional questions: {len(conditional)}")
    print(f"Unconditional questions: {len(unconditional)}")

    # Compute Brier for each model and type
    data = []
    for model in models:
        cond_brier = compute_brier(conditional, model)
        uncond_brier = compute_brier(unconditional, model)
        data.append({
            "model": model.split("/")[-1],  # Short name
            "conditional": cond_brier,
            "unconditional": uncond_brier,
        })
        print(f"{model}: conditional={cond_brier:.4f}, unconditional={uncond_brier:.4f}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(data))
    width = 0.35

    cond_scores = [d["conditional"] or 0 for d in data]
    uncond_scores = [d["unconditional"] or 0 for d in data]

    bars1 = ax.bar([i - width/2 for i in x], uncond_scores, width, label="Unconditional", color="#4C78A8")
    bars2 = ax.bar([i + width/2 for i in x], cond_scores, width, label="Conditional", color="#F58518")

    ax.set_ylabel("Brier Score (lower is better)")
    ax.set_xlabel("Model")
    ax.set_title("CivBench: Conditional vs Unconditional Forecasting")
    ax.set_xticks(x)
    ax.set_xticklabels([d["model"] for d in data], rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, max(max(cond_scores), max(uncond_scores)) * 1.2)

    # Add value labels on bars
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")

if __name__ == "__main__":
    main()
