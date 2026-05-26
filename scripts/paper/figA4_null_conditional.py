"""Fig A4 — Null-conditional placebo control for an Opus 4.5 pilot."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt

from scripts.paper._style import apply_style, save_figure


CONDITIONS = ["Baseline", "Null\nconditional", "Real\nconditional"]
SCORES = [0.169, 0.165, 0.360]
COLORS = ["#94A3B8", "#56B4E9", "#D55E00"]


def main() -> None:
    apply_style()

    fig, ax = plt.subplots(figsize=(4.1, 3.2))
    bars = ax.bar(range(len(CONDITIONS)), SCORES, color=COLORS, edgecolor="white", linewidth=0.6, width=0.62)

    ax.set_ylabel("Brier score")
    ax.set_xticks(range(len(CONDITIONS)))
    ax.set_xticklabels(CONDITIONS)
    ax.set_ylim(0, 0.42)
    ax.set_title("Opus 4.5 conditional placebo control", loc="left", fontweight="bold")

    for bar, score in zip(bars, SCORES):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            score + 0.012,
            f"{score:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#1F2937",
        )

    fig.tight_layout()
    out = save_figure(fig, "figA4_null_conditional")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
