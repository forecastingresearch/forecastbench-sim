"""Chart 4: 2x2 heatmap of framing x factuality interaction."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# --- Data ---
# Rows: factuality (True event / Counterfactual)
# Cols: framing ("If" / "Given that")
data = np.array([
    [-0.001, -0.012],
    [+0.149, +0.146],
])

row_labels = ['True event\n(lookback)', 'Counterfactual\n(fork)']
col_labels = ['"If"', '"Given that"']

# --- Plot ---
fig, ax = plt.subplots(figsize=(5, 3.5))

# Diverging colormap: blue (negative/good) to red (positive/bad)
# Use RdBu_r: red for positive, blue for negative
vmax = max(abs(data.min()), abs(data.max()))
im = ax.imshow(data, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')

# Annotate cells
for i in range(2):
    for j in range(2):
        val = data[i, j]
        # Choose text color for readability
        text_color = 'white' if abs(val) > vmax * 0.5 else 'black'
        ax.text(j, i, f'{val:+.3f}', ha='center', va='center',
                fontsize=13, fontweight='bold', color=text_color)

ax.set_xticks([0, 1])
ax.set_xticklabels(col_labels, fontsize=11)
ax.set_yticks([0, 1])
ax.set_yticklabels(row_labels, fontsize=10)
ax.set_title('Framing \u00d7 Factuality interaction (Brier gap)', fontsize=12, pad=10)

# Colorbar
cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.08)
cbar.set_label('Brier gap', fontsize=10)
cbar.outline.set_visible(False)

# Clean style
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.tick_params(axis='both', which='both', length=0)

plt.tight_layout()
out = '/Users/elsehow/Projects/fri-vault/_artifacts/static/study2_framing_2x2.pdf'
plt.savefig(out, bbox_inches='tight', dpi=300)
print(f'Saved: {out}')
