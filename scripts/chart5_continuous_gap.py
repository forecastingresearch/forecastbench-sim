"""Chart 5: CRPS gap (%) for continuous questions across models, Republic vs Gold."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Colorblind-safe palette (Wong 2011)
COLOR_REPUBLIC = '#D55E00'  # vermillion
COLOR_GOLD = '#0072B2'      # blue

OUTPUT = '/Users/elsehow/Projects/fri-vault/_artifacts/static/study1_continuous_gap.pdf'

# Load data
def load_crps(path):
    with open(path) as f:
        data = json.load(f)
    out = {}
    for mid, r in data['model_results'].items():
        if 'continuous' in r and 'crps' in r['continuous']:
            out[mid] = r['continuous']['crps']
    return out

rb = load_crps('data/results/republic_baseline_continuous_all.json')
rc = load_crps('data/results/republic_conditional_continuous_all.json')
gb = load_crps('data/results/gold500_baseline_continuous_all.json')
gc = load_crps('data/results/gold500_conditional_continuous_all.json')

# Find models that have all 4 conditions
common = set(rb) & set(rc) & set(gb) & set(gc)

# Compute percentage gaps
models = []
rep_gaps = []
gold_gaps = []
for mid in common:
    rep_gap = (rc[mid] - rb[mid]) / rb[mid] * 100
    gold_gap = (gc[mid] - gb[mid]) / gb[mid] * 100
    models.append(mid)
    rep_gaps.append(rep_gap)
    gold_gaps.append(gold_gap)

# Sort by Republic gap (descending)
order = np.argsort(rep_gaps)[::-1]
models = [models[i] for i in order]
rep_gaps = [rep_gaps[i] for i in order]
gold_gaps = [gold_gaps[i] for i in order]

# Clean model names
def short_name(mid):
    name_map = {
        'anthropic/claude-opus-4-5-20251101': 'Opus 4.5',
        'anthropic/claude-sonnet-4-5-20250929': 'Sonnet 4.5',
        'openai/o3-2025-04-16': 'o3',
        'openai/gpt-4.1-2025-04-14': 'GPT-4.1',
        'openai/gpt-5-2025-08-07': 'GPT-5',
        'openai/gpt-5-mini-2025-08-07': 'GPT-5 mini',
        'openai/gpt-5.1-2025-11-13': 'GPT-5.1',
        'google/gemini-2.5-pro': 'Gemini 2.5 Pro',
        'google/gemini-2.5-flash': 'Gemini 2.5 Flash',
        'google/gemini-3-pro-preview': 'Gemini 3 Pro',
    }
    return name_map.get(mid, mid.split('/')[-1])

labels = [short_name(m) for m in models]

# Plot
fig, ax = plt.subplots(figsize=(5, 3.5))

x = np.arange(len(models))
width = 0.35

bars_r = ax.bar(x - width/2, rep_gaps, width, label='Republic', color=COLOR_REPUBLIC, edgecolor='white', linewidth=0.5)
bars_g = ax.bar(x + width/2, gold_gaps, width, label='Gold $500', color=COLOR_GOLD, edgecolor='white', linewidth=0.5)

ax.axhline(0, color='black', linewidth=0.8)
ax.set_ylabel('CRPS gap (%)')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=0)
ax.legend(frameon=False, fontsize=9)

# Remove gridlines and top/right spines
ax.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add value labels on bars
for bar, val in zip(list(bars_r) + list(bars_g), rep_gaps + gold_gaps):
    ypos = bar.get_height()
    offset = 2 if ypos >= 0 else -5
    ax.annotate(f'{val:.0f}%',
                xy=(bar.get_x() + bar.get_width()/2, ypos),
                xytext=(0, offset),
                textcoords='offset points',
                ha='center', va='bottom' if ypos >= 0 else 'top',
                fontsize=8)

plt.tight_layout()
plt.savefig(OUTPUT, bbox_inches='tight')
print(f'Saved to {OUTPUT}')
print(f'Models: {labels}')
print(f'Republic gaps: {rep_gaps}')
print(f'Gold gaps: {gold_gaps}')
