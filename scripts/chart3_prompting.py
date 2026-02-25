"""Chart 3: Grouped bar chart of Brier scores for 3 models x 7 conditions."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# --- Data ---
models = ['o3', 'Opus 4.5', 'GPT-4.1']

conditions = [
    'Unconditional\nbaseline',
    'Conditional\nbaseline',
    'CoT',
    'Explicit\nupdating',
    'Structured\nreasoning',
    'Base rate\nreanchoring',
    'Calibration\nnudge',
]

scores = {
    'o3':       [0.17, 0.339, 0.294, 0.338, 0.329, 0.339, 0.309],
    'Opus 4.5': [0.17, 0.310, 0.320, 0.314, 0.310, 0.297, 0.309],
    'GPT-4.1':  [0.22, 0.466, 0.476, 0.471, 0.510, 0.504, 0.471],
}

baselines = {m: scores[m][0] for m in models}

# --- Colors: Tol bright, colorblind-safe ---
model_colors = {
    'o3':       '#332288',   # indigo
    'Opus 4.5': '#CC6677',   # rose
    'GPT-4.1':  '#DDCC77',   # sand
}

fig, ax = plt.subplots(figsize=(10, 5))

n_conditions = len(conditions)
n_models = len(models)
width = 0.22
x = np.arange(n_conditions)

for i, model in enumerate(models):
    offset = (i - (n_models - 1) / 2) * width
    vals = scores[model]
    ax.bar(x + offset, vals, width, label=model, color=model_colors[model], edgecolor='none')

# Dashed horizontal lines for unconditional baselines
line_styles = {
    'o3':       (0, (5, 3)),
    'Opus 4.5': (0, (3, 2)),
    'GPT-4.1':  (0, (1, 1)),
}
for model in models:
    ax.axhline(baselines[model], color=model_colors[model], linestyle=line_styles[model],
               linewidth=1.2, alpha=0.7)

ax.set_ylabel('Brier score', fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(conditions, fontsize=9)
ax.legend(frameon=False, fontsize=10, loc='upper left')

# Clean style
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(False)
ax.tick_params(axis='both', which='both', length=4)
ax.set_ylim(0, ax.get_ylim()[1] * 1.05)

plt.tight_layout()
out = '/Users/elsehow/Projects/fri-vault/_artifacts/static/study2_prompting.pdf'
plt.savefig(out, bbox_inches='tight', dpi=300)
print(f'Saved: {out}')
