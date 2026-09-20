#!/usr/bin/env python3
"""natcond_draw_sim.py CONSOL_DIR OUT_JSON — natural-conditional cell census over all worlds with the bank
templates as outcomes and the spec's reveal types (turns 61-90). No selection on effect size: every cell with
n_x >= 100 (all replays) is kept, with its effect Delta = p(Y|X) - p(Y), template, horizon, reveal type,
related (same civ) and nested flags. Prints the composition an equal draw per stratum would give."""
import sys, os, pickle, json, collections, itertools, numpy as np, pandas as pd
CONSOL, OUT = sys.argv[1], sys.argv[2]
WAR, CEASE, ARMI, PEACE, ALLY = 1, 2, 3, 4, 5
rows = []
for fn in sorted(f for f in os.listdir(CONSOL) if f.endswith('.pkl')):
    d = pickle.load(open(os.path.join(CONSOL, fn), 'rb')); world = d['world']
    turns = d['turns']; tidx = {t: i for i, t in enumerate(turns)}; civs = d['civs']; pairs = d['pairs']; ev = d['events']
    dst = d['dstate']; m = d['metrics']; gov = d['government']; n = len(d['tags']); cidx = {c: i for i, c in enumerate(civs)}
    inv_gov = {v: k for k, v in d['gov_vocab'].items()}
    # per-replay event tables
    conq = [[(t, dd[0], dd[1]) for (t, ty, p, dd) in e if ty == 'city_conquered'] for e in ev]
    found = [[(t, p) for (t, ty, p, dd) in e if ty == 'city_founded'] for e in ev]
    govch = [[(t, p, dd[1]) for (t, ty, p, dd) in e if ty == 'government_change'] for e in ev]
    tech = [{(p, dd[0]): t for (t, ty, p, dd) in e if ty == 'tech_discovered'} for e in ev]
    wond = [[(t, p, dd[0]) for (t, ty, p, dd) in e if ty == 'wonder_completed'] for e in ev]
    # reveals over turns 61-90
    X = {}
    for c in civs:
        X[('captured', c)] = np.array([any(61 <= t <= 90 and new == c for (t, prev, new) in cv) for cv in conq])
        X[('lost', c)] = np.array([any(61 <= t <= 90 and prev == c for (t, prev, new) in cv) for cv in conq])
        X[('govchange', c)] = np.array([any(61 <= t <= 90 and p == c and to != 'Anarchy' for (t, p, to) in g) for g in govch])
    # outcomes = bank templates
    Y = {}
    for T in (120, 150, 180, 210):
        tj = tidx[T]; wn = slice(tidx[61], tj + 1)
        for pi, (a, b) in enumerate(pairs):
            Y[('NW1_war_at', (a, b), T)] = dst[:, tj, pi] == WAR
            Y[('W6_peace_at', (a, b), T)] = dst[:, tj, pi] == PEACE
            Y[('NW2_ceasefire_by', (a, b), T)] = np.isin(dst[:, wn, pi], (CEASE, ARMI)).any(axis=1)
            Y[('NW2_alliance_by', (a, b), T)] = (dst[:, wn, pi] == ALLY).any(axis=1)
            Y[('NW2_peace_by', (a, b), T)] = (dst[:, wn, pi] == PEACE).any(axis=1)
        tk = m['techs_known'][:, tj, :]
        for c in civs:
            ci = cidx[c]; others = [cidx[o] for o in civs if o != c]
            for g in set(gov[:, tj, ci]) - {-1}:
                Y[('EX_government_at', (c,), T, inv_gov[g])] = gov[:, tj, ci] == g
            for k in (1, 2):
                Y[('S4_govchange_ge_k', (c,), T, k)] = np.array([sum(1 for (t, p, to) in g_ if 61 <= t <= T and p == c and to != 'Anarchy') >= k for g_ in govch])
            techs = {tn for tm in tech for (p, tn) in tm if p == c}
            for tn in techs:
                Y[('EX_tech_discovered', (c,), T, tn)] = np.array([tm.get((c, tn), 10**9) <= T for tm in tech])
            Y[('W5_tech_lead', (c,), T)] = np.where(np.isnan(tk[:, ci]), False, tk[:, ci] > np.nanmax(tk[:, others], axis=1))
            for q in (0.25, 0.5, 0.75):
                K = int(round(np.nanquantile(tk[:, ci], q))); Y[('NB1_tech_threshold', (c,), T, K)] = np.where(np.isnan(tk[:, ci]), False, tk[:, ci] >= K)
            for k in (1, 2):
                Y[('S5_civ_wonders_ge_k', (c,), T, k)] = np.array([sum(1 for (t, p, w) in wv if 61 <= t <= T and p == c) >= k for wv in wond])
            for k in (1, 3, 5):
                Y[('S6_founds_ge_k', (c,), T, k)] = np.array([sum(1 for (t, p) in fv if 61 <= t <= T and p == c) >= k for fv in found])
            for k in (1, 2, 3):
                Y[('W3_capture_k', (c,), T, k)] = np.array([sum(1 for (t, prev, new) in cv if 61 <= t <= T and new == c) >= k for cv in conq])
                Y[('W4_lose_k', (c,), T, k)] = np.array([sum(1 for (t, prev, new) in cv if 61 <= t <= T and prev == c) >= k for cv in conq])
            for c2 in civs:
                if c2 != c: Y[('W2_directed_conq', (c, c2), T)] = np.array([any(61 <= t <= T and new == c and prev == c2 for (t, prev, new) in cv) for cv in conq])
        wnames = {w for wv in wond for (t, p, w) in wv}
        for w in wnames: Y[('NW5_wonder_any', (), T, w)] = np.array([any(61 <= t <= T and ww == w for (t, p, ww) in wv) for wv in wond])
        for k in (3, 10, 20): Y[('NB6_conquests_ge_k', (), T, k)] = np.array([sum(1 for (t, *_ ) in cv if 61 <= t <= T) >= k for cv in conq])
        Y[('NB6_any_conquered', (), T)] = np.array([any(61 <= t <= T for (t, *_ ) in cv) for cv in conq])
    for (xt, xc), x in X.items():
        nx = int(x.sum())
        if nx < 100: continue
        px = x.mean()
        for key, y in Y.items():
            tmpl, ycs, T = key[0], key[1], key[2]
            p = y.mean()
            if p <= 0 or p >= 1: continue           # degenerate questions carry no information
            pyx = y[x].mean(); delta = pyx - p
            related = xc in ycs
            nested = related and ((xt == 'captured' and tmpl in ('W3_capture_k', 'W2_directed_conq')) or (xt == 'lost' and tmpl == 'W4_lose_k')
                                  or (xt == 'govchange' and tmpl in ('S4_govchange_ge_k', 'EX_government_at'))) or (xt == 'captured' and tmpl in ('NB6_conquests_ge_k', 'NB6_any_conquered'))
            rows.append(dict(world=world, reveal=xt, template=tmpl, T=T, related=related, nested=nested, p=p, pyx=pyx, delta=delta, nx=nx))
    print(world, len(rows), flush=True)
df = pd.DataFrame(rows); df.to_json(OUT, orient='records')
bins = [0, 0.03, 0.08, 0.15, 1.01]; names = ['null', 'small', 'medium', 'large']
df['stratum'] = pd.cut(df.delta.abs(), bins, labels=names, right=False)
pd.set_option('display.width', 200)
print(f"\nCELLS (n_x>=100, 0<p<1): {len(df)}  worlds {df.world.nunique()}")
print("\n== supply: cells per template x horizon"); print(pd.crosstab(df.template, df['T']).to_string())
print("\n== effect-size share per template (all cells)"); print(pd.crosstab(df.template, df.stratum, normalize='index').round(3).to_string())
print("\n== effect-size share per horizon"); print(pd.crosstab(df['T'], df.stratum, normalize='index').round(3).to_string())
print("\n== effect-size share per reveal type"); print(pd.crosstab(df.reveal, df.stratum, normalize='index').round(3).to_string())
print("\n== related / nested"); print(pd.crosstab([df.related, df.nested], df.stratum, normalize='index').round(3).to_string()); print(pd.crosstab([df.related, df.nested], df.stratum).to_string())
# equal draw: per (T, template) stratum, uniform sample -> composition = mean of stratum distributions
eq = pd.crosstab([df['T'], df.template], df.stratum, normalize='index')
print("\n== EQUAL DRAW composition (equal weight per horizon x template stratum): expected share and count per 150/horizon")
comp = eq.groupby(level=0).mean(); print((comp.round(3)).to_string()); print((comp * 150).round(1).to_string())
print("\n== EQUAL DRAW by horizon x template x reveal (mean over strata):", (pd.crosstab([df['T'], df.template, df.reveal], df.stratum, normalize='index').groupby(level=0).mean() * 150).round(1).to_string())
