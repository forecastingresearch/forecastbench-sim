#!/usr/bin/env python3
"""effort_compare.py LOW_DIR MEDIUM_DIR — same items, two effort levels: per model excess Brier, bias, slope, reasoning tokens,
cost; and the Spearman of each with ECI across the compared models."""
import sys, os, csv, json, collections, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_v1 as sv
from scipy.stats import spearmanr
S, C, N = sv.load_sets(); low = sv.load_results([sys.argv[1]]); med = sv.load_results([sys.argv[2]])
meta = {m['openrouter_id']: m for m in csv.DictReader(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models_v1.csv')))}
models = sorted({m for (m, _, _) in med}, key=lambda m: -float(meta[m]['eci']))
items = {i for (m, i, a) in med if a == 't1'}
def stats(R, m):
    g = [(S[i]['qAll'], (R.get((m, i, 't1')) or {}).get('value')) for i in items if i in S]
    g = [(q, p) for q, p in g if p is not None]
    if len(g) < 20: return None
    q = np.array([x[0] for x in g]); p = np.array([x[1] for x in g])
    rt = [(R.get((m, i, 't1')) or {}).get('tokens_reasoning') or 0 for i in items]; cost = sum((R.get((m, i, 't1')) or {}).get('cost') or 0 for i in items)
    return dict(n=len(g), excess=float(np.mean((p - q) ** 2)), bias=float(np.mean(p - q)), slope=float(np.polyfit(q, p, 1)[0]), rho=spearmanr(p, q).correlation, reas=float(np.mean(rt)), cost=cost,
                unparsed=sum(1 for i in items if (R.get((m, i, 't1')) or {}).get('value') is None))
print(f"same {len(items)} bank items, low vs medium effort\n{'model':28s} {'ECI':>5s} | {'excess L':>8s} {'excess M':>8s} | {'bias L':>7s} {'bias M':>7s} | {'slope L':>7s} {'slope M':>7s} | {'reas L':>6s} {'reas M':>6s} | {'$ M':>5s} {'unp':>3s}")
rows = []
for m in models:
    a, b = stats(low, m), stats(med, m)
    if not a or not b: print(m, 'incomplete'); continue
    rows.append((m, float(meta[m]['eci']), a, b))
    print(f"{m.split('/')[-1]:28s} {float(meta[m]['eci']):5.0f} | {a['excess']:8.3f} {b['excess']:8.3f} | {a['bias']:+7.2f} {b['bias']:+7.2f} | {a['slope']:7.2f} {b['slope']:7.2f} | {a['reas']:6.0f} {b['reas']:6.0f} | {b['cost']:5.2f} {b['unparsed']:3d}")
if len(rows) >= 4:
    e = [r[1] for r in rows]
    for key, sign in [('excess', -1), ('bias', 1), ('slope', 1), ('rho', 1)]:
        rl = spearmanr(e, [sign * r[2][key] for r in rows]).correlation; rm = spearmanr(e, [sign * r[3][key] for r in rows]).correlation
        print(f"  Spearman vs ECI across these {len(rows)} models, {key:6s}: low {rl:+.2f}   medium {rm:+.2f}")
print('flat-0.5 excess on these items:', round(float(np.mean([(0.5 - S[i]['qAll']) ** 2 for i in items if i in S])), 3))
