"""Chart 1: Grouped bar chart of Brier score gap (conditional - baseline) for 10 models."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# --- Load data ---
def load_brier(path):
    with open(path) as f:
        d = json.load(f)
    return {m: v['binary']['brier_score'] for m, v in d['model_results'].items()}

rep_base = load_brier('data/results/republic_baseline_binary_all.json')
rep_cond = load_brier('data/results/republic_conditional_binary_all.json')
gold_base = load_brier('data/results/gold500_baseline_binary_all.json')
gold_cond = load_brier('data/results/gold500_conditional_binary_all.json')

# Compute gaps
models = sorted(rep_base.keys())
rep_gap = {m: rep_cond[m] - rep_base[m] for m in models}
gold_gap = {m: gold_cond[m] - gold_base[m] for m in models}

# Short names
def short_name(m):
    name = m.split('/')[-1]
    # Clean up version suffixes
    for suffix in ['-20251101', '-20250929', '-2025-04-14', '-2025-08-07',
                   '-2025-11-13', '-2025-04-16', '-preview']:
        name = name.replace(suffix, '')
    # Capitalize nicely
    renames = {
        'claude-opus-4-5': 'Opus 4.5',
        'claude-sonnet-4-5': 'Sonnet 4.5',
        'gemini-2.5-flash': 'Gemini 2.5 Flash',
        'gemini-2.5-pro': 'Gemini 2.5 Pro',
        'gemini-3-pro': 'Gemini 3 Pro',
        'gpt-4.1': 'GPT-4.1',
        'gpt-5': 'GPT-5',
        'gpt-5-mini': 'GPT-5 mini',
        'gpt-5.1': 'GPT-5.1',
        'o3': 'o3',
    }
    return renames.get(name, name)

# Sort by Republic gap (descending)
sorted_models = sorted(models, key=lambda m: rep_gap[m], reverse=True)
labels = [short_name(m) for m in sorted_models]
rep_vals = [rep_gap[m] for m in sorted_models]
gold_vals = [gold_gap[m] for m in sorted_models]

# --- Plot ---
# Colorblind-safe: Tol bright palette
c_republic = '#332288'  # indigo
c_gold = '#CC6677'      # rose

fig, ax = plt.subplots(figsize=(8, 4.5))

x = np.arange(len(labels))
width = 0.35

bars1 = ax.bar(x - width/2, rep_vals, width, label='Republic', color=c_republic, edgecolor='none')
bars2 = ax.bar(x + width/2, gold_vals, width, label='Gold +$500', color=c_gold, edgecolor='none')

ax.axhline(0, color='black', linewidth=0.8, zorder=0)

ax.set_ylabel('Brier score gap\n(conditional \u2212 baseline)', fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=40, ha='right', fontsize=9)
ax.legend(frameon=False, fontsize=10)

# Clean academic style
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(False)
ax.tick_params(axis='both', which='both', length=4)

plt.tight_layout()
out = '/Users/elsehow/Projects/fri-vault/_artifacts/static/study1_brier_gap.pdf'
plt.savefig(out, bbox_inches='tight', dpi=300)
print(f'Saved: {out}')
