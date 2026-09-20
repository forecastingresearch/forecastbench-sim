#!/usr/bin/env python3
"""saturation.py RESULTS_DIR — which sets, families and items carry signal across models, from a sample run.

Per set: item-level mean excess across models, the share of items where every model is within `eps` of the truth (too easy),
the share where every model errs the same way (no spread between models), and the between-model spread of item means.
Per family (bank/tails/continuous): mean excess, between-model SD of the family mean, and rank correlation of the family mean
with the model's overall mean (does the family sort models the same way as the whole set?).
"""
import sys, glob, json, collections, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_v1 as sv
d = sys.argv[1]; only = sys.argv[2].split(',') if len(sys.argv) > 2 else None   # optional: comma list of model ids to restrict to
S, C, N = sv.load_sets(); R = sv.load_results([d])
if only: R = {k: v for k, v in R.items() if k[0] in only}
rows = sv.score_items(R, S, C, N, impute=None)
by_set = collections.defaultdict(list)
for r in rows: by_set[r['set']].append(r)
MET = {'bank': 'excess_brier', 'tails': 'excess_bits', 'mirrors': 'excess_brier', 'extra': 'excess_brier', 'continuous': 'excess_crps', 'natcond': 'excess_t2'}
for s, rs in by_set.items():
    met = MET[s]; rs = [r for r in rs if r.get(met) is not None]
    if not rs: continue
    items = collections.defaultdict(dict)
    for r in rs: items[r['item']][r['model']] = r[met]
    models = sorted({r['model'] for r in rs}); full = {i: v for i, v in items.items() if len(v) >= max(3, int(len(models) * 0.8))}
    if not full: print(f'\n== {s}: too few items with enough models'); continue
    M = np.array([[full[i].get(m, np.nan) for m in models] for i in full])   # items x models
    item_mean = np.nanmean(M, 1); model_mean = np.nanmean(M, 0)
    eps = {'bank': 0.01, 'tails': 0.02, 'mirrors': 0.01, 'extra': 0.01, 'natcond': 0.01}.get(s)
    easy = float(np.mean(np.nanmax(M, 1) < eps)) if eps else float('nan')
    spread = np.nanstd(M, 1)                              # between-model spread per item
    print(f'\n== {s}: {len(full)} items x {len(models)} models  metric {met}')
    print(f'   item mean {met}: median {np.median(item_mean):.4f}, IQR {np.percentile(item_mean, 25):.4f}-{np.percentile(item_mean, 75):.4f}; '
          f'items every model gets within {eps}: {easy:.0%}; items with between-model SD < 0.02: {float(np.mean(spread < 0.02)):.0%}')
    print(f'   between-model spread of the SET mean: SD {np.std(model_mean):.4f} (range {model_mean.min():.4f}-{model_mean.max():.4f})')
    # family view
    fam_of = {r['item']: r['family'] for r in rs}
    fams = sorted({fam_of[i] for i in full})
    if len(fams) > 1 and s != 'natcond':
        from scipy.stats import spearmanr
        print(f'   {"family":24s} {"n":>3s} {"mean":>8s} {"modelSD":>8s} {"rho_vs_set":>10s}  {"easy%":>5s}')
        for f in fams:
            idx = [k for k, i in enumerate(full) if fam_of[i] == f]
            if len(idx) < 2: continue
            fm = np.nanmean(M[idx], 0); rho = spearmanr(fm, model_mean).correlation if len(models) > 3 else float('nan')
            e = float(np.mean(np.nanmax(M[idx], 1) < eps)) if eps else float('nan')
            print(f'   {f:24s} {len(idx):3d} {np.nanmean(M[idx]):8.4f} {np.std(fm):8.4f} {rho:10.2f}  {e:5.0%}')
    if s == 'natcond':
        blocks = collections.defaultdict(list)
        for r in rs: blocks[r['block']].append(r)
        for b, br in sorted(blocks.items()):
            print(f"   block {b}: n={len({r['item'] for r in br})} cells; mean excess_t2 {np.mean([r['excess_t2'] for r in br]):.4f}, stay {np.mean([r['stay'] for r in br]):.4f}, gain {np.mean([r['gain'] for r in br]):+.4f}, mean |move| {np.mean([abs(r['move']) for r in br]):.3f}, mean |target| {np.mean([abs(r['target']) for r in br]):.3f}")
    # worst and easiest items
    order = np.argsort(item_mean); keys = list(full)
    print('   easiest:', [(keys[k], round(float(item_mean[k]), 4)) for k in order[:3]])
    print('   hardest:', [(keys[k], round(float(item_mean[k]), 4)) for k in order[-3:]])
