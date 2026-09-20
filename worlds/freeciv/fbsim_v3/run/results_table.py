#!/usr/bin/env python3
"""results_table.py RESULTS_DIR OUT_DIR — the FreeCiv results in the shapes Nick (StarSim) and Fabio (Micropolis) shared.

  OUT_DIR/freeciv_results_table.md      headline table per model: ECI, FB score, binary recov by horizon, tails bits, continuous
                                        nCRPS_global by horizon, natcond turn-2 excess Brier, calls, cost; Spearman vs ECI per column
  OUT_DIR/freeciv_results_wide.csv      one row per model, every set x horizon x metric, n_items, calls, cost
  OUT_DIR/freeciv_results_long.csv      one row per model x set x family/block x horizon: n_items, n_calls, cost, reasoning tokens, metrics
  OUT_DIR/freeciv_results_binary.csv    Fabio's binary schema: model, question_type (regular|tail|mirror), horizon, nforecasts, nvalid,
                                        brier (expected), expected_brier, calibration (= excess Brier)
  OUT_DIR/freeciv_results_continuous.csv Fabio's continuous schema: model, metric (= family), horizon, nforecasts, nvalid, CRPS,
                                        nCRPS_global (+ excess variants)
Truth for every item is the 1,000-replay frequency/distribution, so "brier" here is the expected Brier (f-p)^2 + p(1-p), as in Fabio's
expected_brier column; recov = 1 - excess/(0.5-p)^2-style is NOT used — recov follows Nick: share of the recoverable Brier score,
1 - excess_brier / excess_brier(always 0.5).
"""
import sys, os, csv, json, glob, collections, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_v1 as sv
from scipy.stats import spearmanr
d, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
S, C, N = sv.load_sets(); R = sv.load_results([d]); rows = sv.score_items(R, S, C, N, impute=0.5)
models_meta = {m['openrouter_id']: m for m in csv.DictReader(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models_v1.csv')))}
try:
    fb = {r['OpenRouterName']: r for r in csv.DictReader(open('/Users/jaeholee0404/Downloads/model_scores.csv'))}
    fb_by_id = {models_meta[k]['openrouter_id']: fb.get(models_meta[k]['name']) for k in models_meta}
except FileNotFoundError: fb_by_id = {}
calls = collections.defaultdict(lambda: collections.Counter()); cost = collections.defaultdict(lambda: collections.Counter()); rtok = collections.defaultdict(list)
for (m, item, arm), r in R.items():
    calls[m][arm] += 1; cost[m][arm] += r.get('cost') or 0
    if r.get('tokens_reasoning') is not None: rtok[m].append(r['tokens_reasoning'])
models = sorted({r['model'] for r in rows}, key=lambda m: -float(models_meta.get(m, {}).get('eci') or 0))
HZ = {'bank': [90, 120, 150, 180, 210], 'tails': [90, 120, 150, 180, 210], 'mirrors': [90, 120, 150, 180, 210], 'continuous': [90, 120, 150, 180, 210], 'natcond': [120, 150, 180, 210]}
def mean(v): v = [x for x in v if x is not None and not (isinstance(x, float) and math.isnan(x))]; return float(np.mean(v)) if v else float('nan')
def recov(rs):   # Nick's "share of recoverable Brier": 1 - excess(model) / excess(always 0.5)
    ex = mean([r['excess_brier'] for r in rs]); base = mean([(0.5 - r['q']) ** 2 for r in rs]); return 1 - ex / base if base else float('nan')
wide, long_rows = [], []
for m in models:
    W = {'model': m, 'eci': models_meta.get(m, {}).get('eci', ''), 'fb_overall': (fb_by_id.get(m) or {}).get('FBOverall', '')}
    byset = collections.defaultdict(list)
    for r in rows:
        if r['model'] == m: byset[r['set']].append(r)
    for s, hz in HZ.items():
        rs = byset.get(s, [])
        for T in hz + ['all']:
            g = [r for r in rs if T == 'all' or r['T'] == T]; key = f'{s}_{T}'
            parsed = sum(r['parsed'] for r in g); W[f'{key}_n_items'] = len(g); W[f'{key}_n_valid'] = parsed
            if s in ('bank', 'tails', 'mirrors'):
                W[f'{key}_excess_brier'] = mean([r['excess_brier'] for r in g]); W[f'{key}_brier'] = mean([r['brier'] for r in g]); W[f'{key}_recov'] = recov(g)
                if s == 'tails': W[f'{key}_excess_bits'] = mean([r['excess_bits'] for r in g]); W[f'{key}_logloss_bits'] = mean([r['logloss_bits'] for r in g])
            elif s == 'continuous':
                for met in ('ncrps_global', 'excess_ncrps_global', 'excess_crps_norm', 'crps5', 'cov50', 'cov90'): W[f'{key}_{met}'] = mean([r.get(met) for r in g])
            else:
                for met in ('excess_t2', 'stay', 'gain', 'excess_nonews', 'move', 'drift'): W[f'{key}_{met}'] = mean([r.get(met) for r in g])
        # long rows: by family (or block) x horizon
        subkey = 'block' if s == 'natcond' else 'family'
        for sub in sorted({r[subkey] for r in rs}):
            for T in hz + ['all']:
                g = [r for r in rs if r[subkey] == sub and (T == 'all' or r['T'] == T)]
                if not g: continue
                L = dict(model=m, eci=W['eci'], fb_overall=W['fb_overall'], question_type=s, group=sub, horizon=T, n_items=len(g), n_valid=sum(r['parsed'] for r in g))
                if s in ('bank', 'tails', 'mirrors'): L.update(excess_brier=mean([r['excess_brier'] for r in g]), brier=mean([r['brier'] for r in g]), recov=recov(g), excess_bits=mean([r['excess_bits'] for r in g]))
                elif s == 'continuous': L.update(ncrps_global=mean([r.get('ncrps_global') for r in g]), excess_ncrps_global=mean([r.get('excess_ncrps_global') for r in g]), crps=mean([r.get('crps5') for r in g]))
                else: L.update(excess_t2=mean([r.get('excess_t2') for r in g]), stay=mean([r.get('stay') for r in g]), gain=mean([r.get('gain') for r in g]), excess_nonews=mean([r.get('excess_nonews') for r in g]))
                long_rows.append(L)
    W['total_calls'] = sum(calls[m].values()); W['total_cost_usd'] = round(sum(cost[m].values()), 4); W['reasoning_tokens_mean'] = mean(rtok[m])
    for arm in ('t1', 't2', 'nonews'): W[f'calls_{arm}'] = calls[m][arm]; W[f'cost_{arm}_usd'] = round(cost[m][arm], 4)
    wide.append(W)
keys = list(dict.fromkeys(k for w in wide for k in w))
with open(f'{out}/freeciv_results_wide.csv', 'w', newline='') as f: wr = csv.DictWriter(f, fieldnames=keys); wr.writeheader(); wr.writerows(wide)
lkeys = list(dict.fromkeys(k for l in long_rows for k in l))
with open(f'{out}/freeciv_results_long.csv', 'w', newline='') as f: wr = csv.DictWriter(f, fieldnames=lkeys); wr.writeheader(); wr.writerows(long_rows)
# Fabio-format files
with open(f'{out}/freeciv_results_binary.csv', 'w', newline='') as f:
    wr = csv.writer(f); wr.writerow(['model', 'question_type', 'horizon', 'nforecasts', 'nvalid', 'brier', 'expected_brier', 'calibration', 'excess_bits'])
    for m in models:
        for s, qt in [('bank', 'regular'), ('tails', 'tail'), ('mirrors', 'mirror')]:
            for T in HZ[s] + ['all']:
                g = [r for r in rows if r['model'] == m and r['set'] == s and (T == 'all' or r['T'] == T)]
                if g: wr.writerow([m, qt, f'T{T}' if T != 'all' else 'all', len(g), sum(r['parsed'] for r in g), mean([r['brier'] for r in g]), mean([r['brier'] for r in g]), mean([r['excess_brier'] for r in g]), mean([r['excess_bits'] for r in g])])
with open(f'{out}/freeciv_results_continuous.csv', 'w', newline='') as f:
    wr = csv.writer(f); wr.writerow(['model', 'metric', 'horizon', 'nforecasts', 'nvalid', 'CRPS', 'nCRPS_global', 'excess_nCRPS_global', 'excess_CRPS_over_IQR'])
    for m in models:
        fams = sorted({r['family'] for r in rows if r['set'] == 'continuous'}) + ['all']
        for fam in fams:
            for T in HZ['continuous'] + ['all']:
                g = [r for r in rows if r['model'] == m and r['set'] == 'continuous' and (fam == 'all' or r['family'] == fam) and (T == 'all' or r['T'] == T)]
                if g: wr.writerow([m, fam, f'T{T}' if T != 'all' else 'all', len(g), sum(r['parsed'] for r in g), mean([r.get('crps5') for r in g]) if fam != 'all' else '', mean([r.get('ncrps_global') for r in g]), mean([r.get('excess_ncrps_global') for r in g]), mean([r.get('excess_crps_norm') for r in g])])
# headline markdown
L = ['# FreeCiv results table — FBSim paper (built ' + __import__('datetime').date.today().isoformat() + ')', '',
     'Per model, one forecast per item, truth = frequency/distribution over 1,000 replays. Binary: `recov` = share of the recoverable Brier score '
     '(1 = matches the truth probabilities, 0 = always 0.5) and excess Brier (p−q)². Tails (q ≤ 0.05): excess bits = KL(q‖p). Continuous: nCRPS_global = CRPS / fixed '
     'per-family constant (Fabio\'s rule; lower is better) and excess over the simulator\'s own floor. Natcond: turn-2 excess Brier vs p(Y|X) after one revealed fact, '
     '`gain` = improvement over not updating. Cost = OpenRouter USD for that model\'s calls. All at the lowest reasoning effort per model (1,024-token budget for the four budget models).', '',
     '| ECI | FB | model | bin T90 | bin T150 | bin T210 | **binary** (recov) | excess Brier | tails bits | mirrors recov | cont T90 | cont T210 | **continuous** (nCRPS) | excess nCRPS | **natcond** excess_t2 | gain | valid % | calls | cost $ |',
     '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
f3 = lambda x: '' if x is None or (isinstance(x, float) and math.isnan(x)) else f'{x:.3f}'
for W in wide:
    valid = (W['bank_all_n_valid'] + W['tails_all_n_valid'] + W['mirrors_all_n_valid'] + W['continuous_all_n_valid']) / max(1, W['bank_all_n_items'] + W['tails_all_n_items'] + W['mirrors_all_n_items'] + W['continuous_all_n_items'])
    L.append(f"| {W['eci']} | {W['fb_overall']} | {W['model']} | {f3(W['bank_90_recov'])} | {f3(W['bank_150_recov'])} | {f3(W['bank_210_recov'])} | **{f3(W['bank_all_recov'])}** | {f3(W['bank_all_excess_brier'])} | {f3(W['tails_all_excess_bits'])} | {f3(W['mirrors_all_recov'])} | "
             f"{f3(W['continuous_90_ncrps_global'])} | {f3(W['continuous_210_ncrps_global'])} | **{f3(W['continuous_all_ncrps_global'])}** | {f3(W['continuous_all_excess_ncrps_global'])} | **{f3(W['natcond_all_excess_t2'])}** | {f3(W['natcond_all_gain'])} | {valid:.1%} | {W['total_calls']} | {W['total_cost_usd']:.2f} |")
L += ['', '## Capability gradient (Spearman ρ vs ECI)', '', '| column | metric | ρ | p | n |', '|---|---|---|---|---|']
for col, met, sign in [('binary', 'bank_all_recov', 1), ('binary excess Brier', 'bank_all_excess_brier', -1), ('tails', 'tails_all_excess_bits', -1), ('mirrors', 'mirrors_all_recov', 1),
                       ('continuous', 'continuous_all_ncrps_global', -1), ('continuous excess', 'continuous_all_excess_ncrps_global', -1), ('natcond excess_t2', 'natcond_all_excess_t2', -1), ('natcond gain', 'natcond_all_gain', 1)]:
    pts = [(float(W['eci']), W[met]) for W in wide if W.get('eci') and W.get(met) == W.get(met)]
    if len(pts) > 3:
        rho, p = spearmanr([a for a, _ in pts], [b * sign for _, b in pts]); L.append(f'| {col} | {met} (sign-adjusted so higher = better) | {rho:.2f} | {p:.3f} | {len(pts)} |')
tot = sum(W['total_cost_usd'] for W in wide); L += ['', f'Total OpenRouter cost for the rows scored here: ${tot:.2f} over {len(wide)} models.']
open(f'{out}/freeciv_results_table.md', 'w').write('\n'.join(L)); print('written', out)
