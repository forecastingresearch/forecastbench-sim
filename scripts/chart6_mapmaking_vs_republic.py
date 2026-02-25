"""Chart 6: Plausible (Map Making) vs implausible (Republic) interventions produce same gap."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Colorblind-safe palette
COLOR_BASELINE = '#999999'   # gray
COLOR_REPUBLIC = '#D55E00'   # vermillion
COLOR_MAPMAKING = '#0072B2'  # blue

OUTPUT = '/Users/elsehow/Projects/fri-vault/_artifacts/static/study2_mapmaking_vs_republic.pdf'

# Map Making data (from experiment)
mapmaking = {
    'o3':      {'baseline': 0.194, 'conditional': 0.329},
    'Opus 4.5': {'baseline': 0.197, 'conditional': 0.327},
    'GPT-4.1': {'baseline': 0.221, 'conditional': 0.404},
}

# Republic data (from JSON files)
model_ids = {
    'o3': 'openai/o3-2025-04-16',
    'Opus 4.5': 'anthropic/claude-opus-4-5-20251101',
    'GPT-4.1': 'openai/gpt-4.1-2025-04-14',
}

with open('data/results/republic_baseline_binary_all.json') as f:
    rb = json.load(f)
with open('data/results/republic_conditional_binary_all.json') as f:
    rc = json.load(f)

republic = {}
for short, full in model_ids.items():
    republic[short] = {
        'baseline': rb['model_results'][full]['binary']['brier_score'],
        'conditional': rc['model_results'][full]['binary']['brier_score'],
    }

models = ['o3', 'Opus 4.5', 'GPT-4.1']

# Extract values
base_rep = [republic[m]['baseline'] for m in models]
cond_rep = [republic[m]['conditional'] for m in models]
base_mm = [mapmaking[m]['baseline'] for m in models]
cond_mm = [mapmaking[m]['conditional'] for m in models]

# Plot: for each model, show baseline (shared concept), Republic conditional, Map Making conditional
# Actually show all three bars: Republic baseline, Republic conditional, Map Making conditional
# But baselines are similar so we can show: Baseline (avg or Republic), Republic cond, MapMaking cond

fig, ax = plt.subplots(figsize=(6, 4))

x = np.arange(len(models))
width = 0.22

bars1 = ax.bar(x - width, base_rep, width, label='Baseline (Republic)', color=COLOR_BASELINE, edgecolor='white', linewidth=0.5)
bars2 = ax.bar(x, cond_rep, width, label='Republic conditional', color=COLOR_REPUBLIC, edgecolor='white', linewidth=0.5)
bars3 = ax.bar(x + width, cond_mm, width, label='Map Making conditional', color=COLOR_MAPMAKING, edgecolor='white', linewidth=0.5)

ax.set_ylabel('Brier score')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend(frameon=False, fontsize=8, loc='upper left')
ax.set_title('Plausible vs implausible interventions\nproduce same gap', fontsize=11, pad=10)

# Remove gridlines and top/right spines
ax.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Set y limit
ax.set_ylim(0, 0.55)

# Add value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{h:.3f}',
                    xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3),
                    textcoords='offset points',
                    ha='center', va='bottom',
                    fontsize=7)

plt.tight_layout()
plt.savefig(OUTPUT, bbox_inches='tight')
print(f'Saved to {OUTPUT}')

# Print summary
for m in models:
    rep_gap = (republic[m]['conditional'] - republic[m]['baseline']) / republic[m]['baseline'] * 100
    mm_gap = (mapmaking[m]['conditional'] - mapmaking[m]['baseline']) / mapmaking[m]['baseline'] * 100
    print(f'{m}: Republic gap={rep_gap:.1f}%, Map Making gap={mm_gap:.1f}%')
