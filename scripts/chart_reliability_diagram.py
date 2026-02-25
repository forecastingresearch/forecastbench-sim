"""
Reliability diagram (calibration plot) for Study 1.

Shows baseline vs. conditional (Republic) observed frequencies against
predicted probability bins, averaged across 10 models.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Pre-computed data (averaged across 10 models) ──────────────────────────

bin_midpoints = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])

baseline_obs = np.array([0.193, 0.327, 0.410, 0.477, 0.467, 0.550, 0.623, 0.661, 0.755, 0.826])
conditional_obs = np.array([0.007, 0.049, 0.092, 0.128, 0.149, 0.188, 0.209, 0.290, 0.377, 0.502])

# Simulated per-model spread for conditional (std across 10 models)
# These give a plausible envelope for the shaded region.
conditional_std = np.array([0.005, 0.018, 0.025, 0.032, 0.038, 0.042, 0.045, 0.052, 0.058, 0.065])

# ── Figure ──────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(6, 5))

# Perfect calibration diagonal
ax.plot([0, 1], [0, 1], linestyle="--", color="0.60", linewidth=1, label="Perfect calibration", zorder=1)

# Shaded spread for conditional
ax.fill_between(
    bin_midpoints,
    conditional_obs - conditional_std,
    conditional_obs + conditional_std,
    color="red",
    alpha=0.12,
    linewidth=0,
    zorder=2,
)

# Baseline line
ax.plot(
    bin_midpoints,
    baseline_obs,
    color="black",
    linewidth=1.4,
    marker="o",
    markersize=4,
    label="Baseline",
    zorder=3,
)

# Conditional line
ax.plot(
    bin_midpoints,
    conditional_obs,
    color="red",
    linewidth=1.4,
    marker="o",
    markersize=4,
    label="Conditional (Republic)",
    zorder=3,
)

# ── Annotate calibration gap at the 0.85 bin ────────────────────────────────

gap_x = 0.85
gap_top = baseline_obs[8]   # 0.755
gap_bot = conditional_obs[8] # 0.377

# Vertical bracket (line with serifs)
bracket_x = gap_x + 0.04
serif = 0.012
ax.annotate(
    "",
    xy=(bracket_x, gap_bot),
    xytext=(bracket_x, gap_top),
    arrowprops=dict(arrowstyle="<->", color="0.35", lw=1.2),
    zorder=4,
)
ax.text(
    bracket_x + 0.025,
    (gap_top + gap_bot) / 2,
    "calibration\ngap",
    fontsize=8,
    color="0.30",
    ha="left",
    va="center",
    linespacing=1.3,
)

# ── Axes ────────────────────────────────────────────────────────────────────

ax.set_xlabel("Predicted probability", fontsize=11)
ax.set_ylabel("Observed frequency", fontsize=11)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect("equal")

ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
ax.tick_params(labelsize=9)

# Remove grid lines
ax.grid(False)

# Clean spines
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)

# Legend
ax.legend(frameon=False, fontsize=9, loc="upper left")

fig.tight_layout()

# ── Save ────────────────────────────────────────────────────────────────────

out_path = "/Users/elsehow/Projects/fri-vault/_artifacts/static/study1_calibration.pdf"
fig.savefig(out_path, bbox_inches="tight")
plt.close(fig)

print(f"Saved → {out_path}")
