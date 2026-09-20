#!/usr/bin/env python3
"""Candidate NEW reveal types and outcome templates (all exact-channel), with their conditional effect sizes.
Effect = p(Y|X) - p(Y) over all replays, cells with n_x >= 100, 0 < p < 1, horizons 120..210. No selection on effect."""
import sys, os, pickle, numpy as np, pandas as pd, collections
CONSOL = sys.argv[1]; WORLDS = sys.argv[2:]
WAR, CEASE, ARMI, PEACE, ALLY, NOMET = 1, 2, 3, 4, 5, 0
rows = []
for world in WORLDS:
    d = pickle.load(open(f'{CONSOL}/{world}.pkl', 'rb'))
    turns = d['turns']; tidx = {t: i for i, t in enumerate(turns)}; civs = d['civs']; pairs = d['pairs']; ev = d['events']
    dst = d['dstate']; m = d['metrics']; gov = d['government']; n = len(d['tags']); cidx = {c: i for i, c in enumerate(civs)}
    inv_gov = {v: k for k, v in d['gov_vocab'].items()}; ANARCHY = d['gov_vocab']['Anarchy']
    w61, w90 = tidx[61], tidx[90]
    conq = [[(t, dd[0], dd[1]) for (t, ty, p, dd) in e if ty == 'city_conquered'] for e in ev]
    civil = [[(t, dd[0]) for (t, ty, p, dd) in e if ty == 'civil_war_transfer'] for e in ev]
    wond = [[(t, p, dd[0]) for (t, ty, p, dd) in e if ty == 'wonder_completed'] for e in ev]
    govch = [[(t, p, dd[1]) for (t, ty, p, dd) in e if ty == 'government_change'] for e in ev]
    # ---------------- reveals (turns 61-90) ----------------
    X = {}
    for pi, (a, b) in enumerate(pairs):
        seg = dst[:, w61:w90 + 1, pi]
        X[('pair: war began', (a, b))] = ((seg == WAR).any(axis=1)) & (dst[:, tidx[60], pi] != WAR)
        X[('pair: ceasefire/armistice reached', (a, b))] = np.isin(seg, (CEASE, ARMI)).any(axis=1)
        X[('pair: peace signed', (a, b))] = ((seg == PEACE).any(axis=1)) & (dst[:, tidx[60], pi] != PEACE)
        X[('pair: first contact', (a, b))] = ((seg != NOMET).any(axis=1)) & (dst[:, tidx[60], pi] == NOMET)
    for c in civs:
        ci = cidx[c]
        X[('civ: captured a city', c)] = np.array([any(61 <= t <= 90 and new == c for (t, prev, new) in cv) for cv in conq])
        X[('civ: lost a city', c)] = np.array([any(61 <= t <= 90 and prev == c for (t, prev, new) in cv) for cv in conq])
        X[('civ: lost >=3 cities', c)] = np.array([sum(1 for (t, prev, new) in cv if 61 <= t <= 90 and prev == c) >= 3 for cv in conq])
        X[('civ: changed government', c)] = np.array([any(61 <= t <= 90 and p == c and to != 'Anarchy' for (t, p, to) in g) for g in govch])
        X[('civ: entered Anarchy', c)] = (gov[:, w61:w90 + 1, ci] == ANARCHY).any(axis=1)
        X[('civ: civil war split', c)] = np.array([any(61 <= t <= 90 and prev == c for (t, prev) in cw) for cw in civil])
        X[('civ: completed a wonder', c)] = np.array([any(61 <= t <= 90 and p == c for (t, p, w) in wv) for wv in wond])
        cc = m['cities_count']; X[('civ: cities fell by >=2 (t60->t90)', c)] = (cc[:, w90, ci] <= cc[:, tidx[60], ci] - 2)
        X[('civ: cities grew by >=4 (t60->t90)', c)] = (cc[:, w90, ci] >= cc[:, tidx[60], ci] + 4)
        tk = m['techs_known']; med = np.nanmedian(tk[:, w90, ci]); X[('civ: techs at t90 above median', c)] = tk[:, w90, ci] > med
        sc = m['scores']; X[('civ: score leader at t90', c)] = np.where(np.isnan(sc[:, w90, ci]), False, sc[:, w90, ci] > np.nanmax(sc[:, w90, [cidx[o] for o in civs if o != c]], axis=1))
    # ---------------- outcomes ----------------
    Y = {}
    for T in (120, 150, 180, 210):
        tj = tidx[T]; wn = slice(w61, tj + 1)
        for pi, (a, b) in enumerate(pairs):
            Y[('NW1_war_at', (a, b), T)] = dst[:, tj, pi] == WAR
            Y[('W6_peace_at', (a, b), T)] = dst[:, tj, pi] == PEACE
            Y[('W2_directed_conq', (a, b), T)] = np.array([any(61 <= t <= T and new == a and prev == b for (t, prev, new) in cv) for cv in conq])
            Y[('NEW pair: alliance at T', (a, b), T)] = dst[:, tj, pi] == ALLY
            for metric in ('population', 'cities_count', 'territory_size', 'scores'):
                va, vb = m[metric][:, tj, cidx[a]], m[metric][:, tj, cidx[b]]
                Y[(f'NEW EX_comparative {metric}', (a, b), T)] = np.where(np.isnan(va) | np.isnan(vb), False, va > vb)
        for c in civs:
            ci = cidx[c]; others = [cidx[o] for o in civs if o != c]
            Y[('W5_tech_lead', (c,), T)] = np.where(np.isnan(m['techs_known'][:, tj, ci]), False, m['techs_known'][:, tj, ci] > np.nanmax(m['techs_known'][:, tj, others], axis=1))
            Y[('NEW NW4 eliminated by T', (c,), T)] = (m['is_alive'][:, w61:tj + 1, ci] == 0).any(axis=1)
            Y[('NEW EX_rank1 score at T', (c,), T)] = np.where(np.isnan(m['scores'][:, tj, ci]), False, m['scores'][:, tj, ci] > np.nanmax(m['scores'][:, tj, others], axis=1))
            for g in set(gov[:, tj, ci]) - {-1}: Y[('EX_government_at', (c,), T, inv_gov[g])] = gov[:, tj, ci] == g
            Y[('NEW anarchy turns >=3 in window', (c,), T)] = (gov[:, wn, ci] == ANARCHY).sum(axis=1) >= 3
            for metric in ('population', 'cities_count', 'territory_size', 'scores', 'military_units_count', 'treasury'):
                v = m[metric][:, tj, ci]
                for q in (0.25, 0.5, 0.75):
                    K = np.nanquantile(v, q); Y[(f'NEW NB1_threshold {metric}', (c,), T, round(float(K), 1))] = np.where(np.isnan(v), False, v >= K)
                path = m[metric][:, wn, ci]; base = m[metric][:, tidx[60], ci]
                if metric in ('population', 'cities_count', 'territory_size', 'treasury'):
                    f = {'population': 0.8, 'cities_count': 0.8, 'territory_size': 0.8, 'treasury': 0.5}[metric]
                    Y[(f'NEW NB4_drawdown {metric}', (c,), T, f)] = np.where(np.isnan(base) | (base <= 0), False, np.nanmin(path, axis=1) < f * base)
            Y[('NEW capital-less: lost >=5 cities by T', (c,), T)] = np.array([sum(1 for (t, prev, new) in cv if 61 <= t <= T and prev == c) >= 5 for cv in conq])
            Y[('NEW civil war by T', (c,), T)] = np.array([any(61 <= t <= T and prev == c for (t, prev) in cw) for cw in civil])
            Y[('NEW at war with >=2 civs at T', (c,), T)] = sum((dst[:, tj, pi] == WAR) for pi, (a, b) in enumerate(pairs) if c in (a, b)) >= 2
    for (xt, xc), x in X.items():
        nx = int(x.sum())
        if nx < 100: continue
        for key, y in Y.items():
            tmpl, ycs, T = key[0], key[1], key[2]; p = y.mean()
            if p <= 0.01 or p >= 0.99: continue
            related = (xc in ycs) if not isinstance(xc, tuple) else (xc == ycs or (len(ycs) == 2 and set(xc) == set(ycs)) or (len(ycs) == 1 and ycs[0] in xc))
            rows.append(dict(world=world, reveal=xt, template=tmpl, T=T, related=related, delta=y[x].mean() - p, p=p, nx=nx))
    print(world, len(rows), flush=True)
df = pd.DataFrame(rows); df['abs'] = df.delta.abs()
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 200)
def summ(g): return pd.Series({'cells': len(g), 'share>=.08': (g['abs'] >= 0.08).mean(), 'share>=.15': (g['abs'] >= 0.15).mean(), 'p90|d|': g['abs'].quantile(.9), 'max|d|': g['abs'].max()})
print("\n== BY REVEAL TYPE (related cells only) — how much does each kind of news move related questions?")
print(df[df.related].groupby('reveal').apply(summ).sort_values('share>=.08', ascending=False).round(3).to_string())
print("\n== BY OUTCOME TEMPLATE (related cells only)")
print(df[df.related].groupby('template').apply(summ).sort_values('share>=.08', ascending=False).round(3).to_string())
print("\n== TOP reveal x template pairs by share of |d|>=0.08 (related, >=40 cells)")
pt = df[df.related].groupby(['reveal', 'template']).apply(summ); pt = pt[pt.cells >= 40].sort_values('share>=.08', ascending=False).head(30).round(3); print(pt.to_string())
print("\n== by horizon (related cells): share>=.08"); print(df[df.related].groupby('T').apply(summ).round(3).to_string())
df.to_json('/Users/jaeholee0404/civbench/tmp/fbsim_v3_run/state/natcond_new_cells.json', orient='records')
