"""Chart 7: Null conditional control (Opus 4.5)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Colorblind-safe palette
COLOR_BASELINE = '#999999'      # gray
COLOR_NULL = '#56B4E9'          # sky blue
COLOR_CONDITIONAL = '#D55E00'   # vermillion

OUTPUT = '/Users/elsehow/Projects/fri-vault/_artifacts/static/study2_null_conditional.pdf'

# Data
conditions = ['Baseline', 'Null conditional', 'Real conditional']
scores = [0.169, 0.165, 0.360]
colors = [COLOR_BASELINE, COLOR_NULL, COLOR_CONDITIONAL]

fig, ax = plt.subplots(figsize=(4.5, 3.5))

x = np.arange(len(conditions))
bars = ax.bar(x, scores, width=0.55, color=colors, edgecolor='white', linewidth=0.5)

ax.set_ylabel('Brier score')
ax.set_xticks(x)
ax.set_xticklabels(conditions, fontsize=9)
ax.set_title('Null conditional control (Opus 4.5)', fontsize=11, pad=10)

# Remove gridlines and top/right spines
ax.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Set y limit
ax.set_ylim(0, 0.45)

# Annotate with Brier values
for bar, val in zip(bars, scores):
    ax.annotate(f'{val:.3f}',
                xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                xytext=(0, 4),
                textcoords='offset points',
                ha='center', va='bottom',
                fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT, bbox_inches='tight')
print(f'Saved to {OUTPUT}')
