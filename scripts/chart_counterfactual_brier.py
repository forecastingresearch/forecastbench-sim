"""Chart: Counterfactual Brier — baseline, counterfactual, and conditional Brier for 10 models."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# --- Data ---
data = [
    ('o3',               0.147, 0.165, 0.323),
    ('GPT-5 mini',       0.138, 0.185, 0.324),
    ('Opus 4.5',         0.130, 0.152, 0.339),
    ('Sonnet 4.5',       0.124, 0.154, 0.351),
    ('GPT-5.1',          0.123, 0.152, 0.353),
    ('GPT-5',            0.139, 0.159, 0.367),
    ('Gemini 3 Pro',     0.125, 0.154, 0.401),
    ('Gemini 2.5 Pro',   0.148, 0.169, 0.403),
    ('Gemini 2.5 Flash', 0.151, 0.179, 0.415),
    ('GPT-4.1',          0.147, 0.155, 0.446),
]

# Sort by conditional Brier (ascending)
data.sort(key=lambda r: r[3])

labels      = [r[0] for r in data]
baseline    = np.array([r[1] for r in data])
counterfact = np.array([r[2] for r in data])
conditional = np.array([r[3] for r in data])

# --- Plot ---
fig, ax = plt.subplots(figsize=(10, 5))

x = np.arange(len(labels))
width = 0.25

bars_base = ax.bar(x - width, baseline,    width, label='Baseline predictions vs baseline truth',
                   color='black', edgecolor='none')
bars_cf   = ax.bar(x,          counterfact, width, label='Conditional predictions vs baseline truth',
                   color='#888888', edgecolor='none')
bars_cond = ax.bar(x + width,  conditional, width, label='Conditional predictions vs fork truth',
                   color='#CC3333', edgecolor='none')

ax.set_ylabel('Brier score', fontsize=11)
ax.set_ylim(0, 0.5)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
ax.legend(frameon=False, fontsize=9, loc='upper left')

# Clean academic style
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(False)
ax.tick_params(axis='both', which='both', length=4)

# --- Bracket annotation: black and gray bars are nearly identical ---
# Draw a horizontal bracket spanning the full x range, sitting just above the tallest
# black/gray bar cluster.
bracket_y = max(max(baseline), max(counterfact)) + 0.018
bracket_left  = x[0] - width - width * 0.3
bracket_right = x[-1] + width * 0.3  # span black and gray groups only
tick_h = 0.008

# Horizontal line
ax.plot([bracket_left, bracket_right], [bracket_y, bracket_y],
        color='#555555', linewidth=1.0, clip_on=False)
# Left tick
ax.plot([bracket_left, bracket_left], [bracket_y - tick_h, bracket_y],
        color='#555555', linewidth=1.0, clip_on=False)
# Right tick
ax.plot([bracket_right, bracket_right], [bracket_y - tick_h, bracket_y],
        color='#555555', linewidth=1.0, clip_on=False)
# Label
ax.text((bracket_left + bracket_right) / 2, bracket_y + 0.006,
        'Baseline and counterfactual Brier nearly identical',
        ha='center', va='bottom', fontsize=8, color='#555555')

plt.tight_layout()
out = '/Users/elsehow/Projects/fri-vault/_artifacts/static/study1_counterfactual_brier.pdf'
plt.savefig(out, bbox_inches='tight', dpi=300)
print(f'Saved: {out}')
