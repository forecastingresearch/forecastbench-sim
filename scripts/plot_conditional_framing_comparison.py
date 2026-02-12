#!/usr/bin/env python3
"""
Generate bar chart comparing Brier scores across conditional framing experiment conditions.

Conditions:
- Baseline (unconditional): "Will X have more Y than Z at turn H?"
- Conditional (intervention): "If X received +500 gold next turn, would X..."
- Conditional-no: "If X does NOT receive +500 gold next turn, would X..."
"""

import matplotlib.pyplot as plt
import numpy as np

# Results from Opus 4.5 evaluations on seed0 (same world report for all conditions)
# All evaluations use identical game state context
# Baseline: data/evaluations/results/seed0_baseline_opus45.json
# Conditional: data/evaluations/results/seed0_conditional_opus45.json
# Conditional-no: data/evaluations/results/seed0_conditional_no_opus45.json

results = {
    "Baseline\n(unconditional)": {
        "brier": 0.1647,
        "n": 100,
        "framing": '"Will X..."',
    },
    "Conditional\n(intervention)": {
        "brier": 0.1884,
        "n": 96,
        "framing": '"If X received +500 gold..."',
    },
    "Conditional-no\n(no intervention)": {
        "brier": 0.1732,
        "n": 80,
        "framing": '"If X does NOT receive..."',
    },
}

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Data for plotting
conditions = list(results.keys())
brier_scores = [results[c]["brier"] for c in conditions]
sample_sizes = [results[c]["n"] for c in conditions]

# Create bars
x = np.arange(len(conditions))
width = 0.6

# Color scheme: baseline blue, conditional orange, conditional-no green
colors = ["#4477AA", "#EE7733", "#228833"]

bars = ax.bar(x, brier_scores, width, color=colors, edgecolor="black", linewidth=1)

# Add value labels on bars
for bar, score, n in zip(bars, brier_scores, sample_sizes):
    height = bar.get_height()
    ax.annotate(
        f"{score:.4f}\n(n={n})",
        xy=(bar.get_x() + bar.get_width() / 2, height),
        xytext=(0, 3),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )

# Customize plot
ax.set_ylabel("Brier Score (lower is better)", fontsize=12)
ax.set_title(
    "Opus 4.5 Forecasting Performance: Conditional Framing Experiment",
    fontsize=14,
    fontweight="bold",
)
ax.set_xticks(x)
ax.set_xticklabels(conditions, fontsize=10)
ax.set_ylim(0, max(brier_scores) * 1.3)

# Add horizontal line for random baseline
ax.axhline(y=0.25, color="gray", linestyle="--", linewidth=1, label="Random (0.25)")
ax.legend(loc="upper right")

# Add framing descriptions at bottom
framings = [results[c]["framing"] for c in conditions]
for i, framing in enumerate(framings):
    ax.text(
        i,
        -0.02,
        framing,
        ha="center",
        va="top",
        fontsize=8,
        style="italic",
        color="gray",
        transform=ax.get_xaxis_transform(),
    )

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)

# Save figure
output_path = "data/evaluations/plots/conditional_framing_comparison.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"Saved chart to: {output_path}")

# Also save as PDF for publication
pdf_path = "data/evaluations/plots/conditional_framing_comparison.pdf"
plt.savefig(pdf_path, bbox_inches="tight")
print(f"Saved PDF to: {pdf_path}")

plt.show()
