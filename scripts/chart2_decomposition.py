"""Chart 2: Stacked bar chart of Murphy decomposition of the gap."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# --- Data ---
data = {
    'o3':               {'dREL': 0.106, 'dRES': 0.000, 'dUNC': 0.017},
    'GPT-5.1':          {'dREL': 0.138, 'dRES': 0.013, 'dUNC': 0.017},
    'GPT-5':            {'dREL': 0.136, 'dRES': 0.007, 'dUNC': 0.017},
    'GPT-5 mini':       {'dREL': 0.099, 'dRES': 0.002, 'dUNC': 0.017},
    'GPT-4.1':          {'dREL': 0.216, 'dRES': 0.010, 'dUNC': 0.017},
    'Gemini 3 Pro':     {'dREL': 0.179, 'dRES': 0.011, 'dUNC': 0.017},
    'Gemini 2.5 Pro':   {'dREL': 0.173, 'dRES': 0.001, 'dUNC': 0.017},
    'Gemini 2.5 Flash': {'dREL': 0.176, 'dRES': -0.002, 'dUNC': 0.017},
    'Opus 4.5':         {'dREL': 0.134, 'dRES': 0.007, 'dUNC': 0.017},
    'Sonnet 4.5':       {'dREL': 0.144, 'dRES': 0.002, 'dUNC': 0.017},
}

# Sort by total gap descending
models = sorted(data.keys(), key=lambda m: sum(data[m].values()), reverse=True)

dREL = [data[m]['dREL'] for m in models]
dRES = [data[m]['dRES'] for m in models]
dUNC = [data[m]['dUNC'] for m in models]

# --- Plot ---
# Colorblind-safe: Tol bright
c_rel = '#332288'   # indigo
c_res = '#88CCEE'   # cyan
c_unc = '#DDCC77'   # sand

fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(len(models))

ax.bar(x, dREL, label=r'$\Delta$REL (reliability)', color=c_rel, edgecolor='none')
ax.bar(x, dRES, bottom=dREL, label=r'$-\Delta$RES (resolution)', color=c_res, edgecolor='none')
# Stack dUNC on top of dREL + dRES
bottoms = [r + s for r, s in zip(dREL, dRES)]
ax.bar(x, dUNC, bottom=bottoms, label=r'$\Delta$UNC (uncertainty)', color=c_unc, edgecolor='none')

ax.set_ylabel('Brier score gap', fontsize=11)
ax.set_title('Murphy decomposition of conditional gap (Republic)', fontsize=12, pad=10)
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=40, ha='right', fontsize=9)
ax.legend(frameon=False, fontsize=9, loc='upper right')

# Clean style
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(False)
ax.tick_params(axis='both', which='both', length=4)

plt.tight_layout()
out = '/Users/elsehow/Projects/fri-vault/_artifacts/static/study2_brier_decomposition.pdf'
plt.savefig(out, bbox_inches='tight', dpi=300)
print(f'Saved: {out}')
