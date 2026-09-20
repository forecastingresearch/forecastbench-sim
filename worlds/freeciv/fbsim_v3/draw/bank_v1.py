#!/usr/bin/env python3
"""bank_v1.py CONSOL_DIR OUT_DIR — generate EVERY roster instance per world with its full yes-vector (1000 replays),
plus continuous instances (value vectors) and reveal vectors (turns 61-90). One source for bank, tails, mirrors,
continuous and natural-conditional draws. Truth = all replays (qAll); qA/qB kept for reference.
Output: OUT_DIR/<world>.pkl = {'bin': [inst...], 'Y': bool matrix [n_inst, n_rep], 'cont': [inst...], 'V': float matrix,
'reveals': [rev...], 'X': bool matrix, 'civ_names', 'n'}"""
import os, sys, pickle, itertools, collections, numpy as np
CONSOL, OUT = sys.argv[1], sys.argv[2]; os.makedirs(OUT, exist_ok=True)
WAR, CEASE, ARMI, PEACE, ALLY, NOMET = 1, 2, 3, 4, 5, 0
HZ = (90, 120, 150, 180, 210)
def nice(m): return {'scores': 'score', 'treasury': 'treasury', 'population': 'population', 'cities_count': 'number of cities',
                     'territory_size': 'territory size', 'military_units_count': 'number of military units', 'techs_known': 'number of technologies'}[m]
for fn in sorted(f for f in os.listdir(CONSOL) if f.endswith('.pkl')):
    d = pickle.load(open(os.path.join(CONSOL, fn), 'rb')); world = d['world']
    turns = d['turns']; tidx = {t: i for i, t in enumerate(turns)}; civs = d['civs']; pairs = d['pairs']; ev = d['events']
    dst = d['dstate']; m = d['metrics']; gov = d['government']; n = len(d['tags']); A, B = d['halfA'], d['halfB']
    inv_gov = {v: k for k, v in d['gov_vocab'].items()}; ANARCHY = d['gov_vocab']['Anarchy']; cidx = {c: i for i, c in enumerate(civs)}
    name = lambda c: d['civ_names'].get(str(c), str(c))
    conq = [[(t, dd[0], dd[1]) for (t, ty, p, dd) in e if ty == 'city_conquered'] for e in ev]
    found = [[(t, p) for (t, ty, p, dd) in e if ty == 'city_founded'] for e in ev]
    destr = [[t for (t, ty, p, dd) in e if ty == 'city_destroyed'] for e in ev]
    wond = [[(t, p, dd[0]) for (t, ty, p, dd) in e if ty == 'wonder_completed'] for e in ev]
    govch = [[(t, p, dd[0], dd[1]) for (t, ty, p, dd) in e if ty == 'government_change'] for e in ev]
    tech = [{(p, dd[0]): t for (t, ty, p, dd) in e if ty == 'tech_discovered'} for e in ev]
    civil = [[(t, dd[0]) for (t, ty, p, dd) in e if ty == 'civil_war_transfer'] for e in ev]
    w61 = tidx[61]; w90 = tidx[90]
    m = dict(m); m['techs_known'] = m['techs_known'] - 1.0          # savegame tech sets include the A_NONE placeholder; questions count real technologies
    # admissibility: a civ (or pair, or the whole world for world-level state questions) is askable at T only if
    # alive at T in EVERY replay; questions about eliminated civilizations are never asked
    safe = {T: {c: bool((m['is_alive'][:, tidx[T], cidx[c]] == 1).all()) for c in civs} for T in HZ}
    bins, Ys = [], []
    WORLD_STATE = ('W5_tech_lead', 'S3_wars_at_T')
    def addb(family, T, text, yes, subj, **kw):
        yes = np.asarray(yes, bool); q = float(yes.mean())
        if q <= 0 or q >= 1: return
        if family not in ('NW4_survival', 'NEW_civil_war'):
            need = civs if family in WORLD_STATE else list(subj)
            if not all(safe[T][c] for c in need): return
        bins.append(dict(world=world, family=family, T=T, text=text, subj=list(subj), qAll=q, qA=float(yes[A].mean()), qB=float(yes[B].mean()), **kw)); Ys.append(yes)
    for T in HZ:
        tj = tidx[T]; wn = slice(w61, tj + 1); dT = dst[:, tj, :]; dU = dst[:, wn, :]
        for pi, (a, b) in enumerate(pairs):
            na, nb = name(a), name(b)
            addb('NW1_war_at', T, f"Will {na} and {nb} be at war at turn {T}?", dT[:, pi] == WAR, (a, b))
            addb('W6_peace_at', T, f"Will {na} and {nb} have a formal peace in effect at turn {T}?", dT[:, pi] == PEACE, (a, b))
            s60 = int(dst[0, tidx[60], pi])   # the turn-60 state is shared by all replays; 'reach' means the pair is not already there
            if s60 != PEACE: addb('NW2_diplo', T, f"Will {na} and {nb} be at formal Peace at any point between turns 61 and {T}?", (dU[:, :, pi] == PEACE).any(axis=1), (a, b), kind='peace_by')
            if s60 != ALLY: addb('NW2_diplo', T, f"Will {na} and {nb} form an Alliance at any point between turns 61 and {T}?", (dU[:, :, pi] == ALLY).any(axis=1), (a, b), kind='alliance_by')
            if s60 not in (CEASE, ARMI): addb('NW2_diplo', T, f"Will {na} and {nb} reach a cease-fire or armistice at any point between turns 61 and {T}?", np.isin(dU[:, :, pi], (CEASE, ARMI)).any(axis=1), (a, b), kind='ceasefire_by')
            addb('W2_directed_conquest', T, f"Will {na} capture at least one city from {nb} by turn {T}?", np.array([any(61 <= t <= T and new == a and prev == b for (t, prev, new) in cv) for cv in conq]), (a, b))
            addb('W2_directed_conquest', T, f"Will {nb} capture at least one city from {na} by turn {T}?", np.array([any(61 <= t <= T and new == b and prev == a for (t, prev, new) in cv) for cv in conq]), (b, a))
            for metric in ('population', 'cities_count', 'territory_size', 'scores'):
                va, vb = m[metric][:, tj, cidx[a]], m[metric][:, tj, cidx[b]]
                addb('EX_comparative', T, f"Will {na} have a higher {nice(metric)} than {nb} at turn {T}?", np.where(np.isnan(va) | np.isnan(vb), False, va > vb), (a, b), metric=metric)
                addb('EX_comparative', T, f"Will {nb} have a higher {nice(metric)} than {na} at turn {T}?", np.where(np.isnan(va) | np.isnan(vb), False, vb > va), (b, a), metric=metric)
        wars = (dT[:, :] == WAR).sum(axis=1)
        for k in (1, 2, 3, 4): addb('S3_wars_at_T', T, f"Will at least {k} pairs of civilizations be at war at exactly turn {T}?", wars >= k, (), k=k)
        addb('S7_any_destroyed', T, f"Will any city be destroyed by turn {T}?", np.array([any(61 <= t <= T for t in dv) for dv in destr]), ())
        addb('NB6_event', T, f"Will any city be conquered between turn 61 and turn {T}?", np.array([any(61 <= t <= T for (t, *_ ) in cv) for cv in conq]), (), kind='any')
        for k in (3, 10, 20): addb('NB6_event', T, f"Will at least {k} cities be conquered (in total) by turn {T}?", np.array([sum(1 for (t, *_ ) in cv if 61 <= t <= T) >= k for cv in conq]), (), kind=f'ge{k}')
        wnames = sorted({w for wv in wond for (t, p, w) in wv if 61 <= t <= T})
        firsts = [{} for _ in range(n)]
        for r, wv in enumerate(wond):
            for (t, p, w) in sorted(wv):
                if 61 <= t <= T and w not in firsts[r]: firsts[r][w] = p
        def wname(w):  # article for wonder names that take one
            return w if ("'" in w or w in ("Women's Suffrage",)) else f"the {w}"
        for w in wnames:
            addb('NW5_wonder', T, f"Will {wname(w)} be completed by any civilization by turn {T}?", np.array([w in f for f in firsts]), (), wonder=w)
            cnt = collections.Counter(f[w] for f in firsts if w in f)
            for p, c in cnt.most_common(2):
                if p in civs: addb('W1_wonder_race', T, f"Will {name(p)} complete {wname(w)} by turn {T}?", np.array([f.get(w) == p for f in firsts]), (p,), wonder=w)
        tk = m['techs_known'][:, tj, :]
        for c in civs:
            ci = cidx[c]; nc = name(c); others = [cidx[o] for o in civs if o != c]
            for g in sorted(set(gov[:, tj, ci]) - {-1}): addb('EX_government_at', T, f"Will {nc}'s government be {inv_gov[g]} at turn {T}?", gov[:, tj, ci] == g, (c,), gov=inv_gov[g])
            # every recorded transition counts, including into and out of anarchy (anarchy is a government
            # state in the report's timeline); the criteria say so explicitly
            gc = np.array([sum(1 for (t, p, f_, to) in g_ if 61 <= t <= T and p == c) for g_ in govch])
            for k in (1, 2, 3): addb('S4_gov_change_count', T, f"Will {nc}'s government change at least {k} time{'s' if k > 1 else ''} by turn {T}? (Anarchy counts as a form of government, for this question.)", gc >= k, (c,), k=k)
            techs = sorted({tn for tm in tech for (p, tn) in tm if p == c})
            for tn in techs: addb('EX_tech_discovered', T, f"Will {nc} have discovered {tn} by turn {T}?", np.array([tm.get((c, tn), 10**9) <= T for tm in tech]), (c,), tech=tn)
            addb('W5_tech_lead', T, f"Will {nc} know strictly more technologies than every other civilization at turn {T}?", np.where(np.isnan(tk[:, ci]), False, tk[:, ci] > np.nanmax(tk[:, others], axis=1)), (c,))
            for metric in ('techs_known', 'scores', 'treasury', 'population', 'cities_count', 'territory_size', 'military_units_count'):
                v = m[metric][:, tj, ci]; fam = 'NB1_threshold' if metric == 'techs_known' else 'NB1_value_threshold'
                for ql in (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95):
                    x = int(round(float(np.nanquantile(v, ql))))
                    if x <= 0: continue
                    addb(fam, T, f"Will {nc}'s {nice(metric)} be at least {x} at turn {T}?", np.where(np.isnan(v), False, v >= x), (c,), metric=metric, x=x, sel_q=ql)
            for metric, fs in {'treasury': (0.5, 0.25), 'territory_size': (0.9, 0.7), 'cities_count': (0.99, 0.7), 'population': (0.9, 0.75)}.items():
                base = m[metric][:, tidx[60], ci]; lo = np.nanmin(m[metric][:, wn, ci], axis=1)
                for f in fs: addb('NB4_drawdown', T, f"Will {nc}'s {nice(metric)} drop below {int(f*100)}% of its turn-60 level at any point up to turn {T}?", np.where(np.isnan(lo) | np.isnan(base) | (base <= 0), False, lo < f * base), (c,), metric=metric, f=f)
            for k in (1, 2, 3): addb('S5_civ_wonders', T, f"Will {nc} complete at least {k} wonder{'s' if k > 1 else ''} by turn {T}?", np.array([sum(1 for (t, p, w) in wv if 61 <= t <= T and p == c) >= k for wv in wond]), (c,), k=k)
            for k in (1, 3, 5, 8): addb('S6_city_founding', T, f"Will {nc} found at least {k} new cit{'ies' if k > 1 else 'y'} by turn {T}?", np.array([sum(1 for (t, p) in fv if 61 <= t <= T and p == c) >= k for fv in found]), (c,), k=k)
            for k in (1, 2, 3, 5): addb('W3_capture_k', T, f"Will {nc} capture at least {k} cit{'ies' if k > 1 else 'y'} by turn {T}?", np.array([sum(1 for (t, prev, new) in cv if 61 <= t <= T and new == c) >= k for cv in conq]), (c,), k=k)
            for k in (1, 2, 3, 5): addb('W4_lose_k', T, f"Will {nc} lose at least {k} cit{'ies' if k > 1 else 'y'} by turn {T}?", np.array([sum(1 for (t, prev, new) in cv if 61 <= t <= T and prev == c) >= k for cv in conq]), (c,), k=k)
            addb('NW4_survival', T, f"Will {nc} be eliminated (cease to exist as a civilization) by turn {T}?", (m['is_alive'][:, wn, ci] == 0).any(axis=1), (c,))
            addb('NEW_civil_war', T, f"Will {nc} suffer a civil war (a breakaway civilization splitting off) by turn {T}?", np.array([any(61 <= t <= T and prev == c for (t, prev) in cw) for cw in civil]), (c,))
    seen = set(); keep = [i for i, b in enumerate(bins) if not (b['text'] in seen or seen.add(b['text']))]
    dropped = len(bins) - len(keep); bins = [bins[i] for i in keep]; Ys = [Ys[i] for i in keep]
    Y = np.array(Ys, dtype=bool) if Ys else np.zeros((0, n), bool)
    # ---------------- continuous ----------------
    conts, Vs = [], []
    def addc(family, T, text, vals, subj, **kw):
        vals = np.asarray(vals, float); f = vals[~np.isnan(vals)]
        if len(f) < 100: return
        need = civs if family == 'P6_world_techs' else list(subj)
        if not all(safe[T][c] for c in need): return
        iqr = float(np.subtract(*np.percentile(f, [75, 25])))
        conts.append(dict(world=world, family=family, T=T, text=text, subj=list(subj), medAll=float(np.median(f)), iqrAll=iqr, p05=float(np.percentile(f, 5)), p95=float(np.percentile(f, 95)), censAll=float(np.mean(np.isnan(vals))), **kw)); Vs.append(vals)
    for T in HZ:
        tj = tidx[T]
        for c in civs:
            ci = cidx[c]; nc = name(c)
            addc('P1_civ_conquests', T, f"How many cities will {nc} capture by turn {T}?", [float(sum(1 for (t, prev, new) in cv if 61 <= t <= T and new == c)) for cv in conq], (c,))
            addc('P2_civ_losses', T, f"How many cities will {nc} lose by turn {T}?", [float(sum(1 for (t, prev, new) in cv if 61 <= t <= T and prev == c)) for cv in conq], (c,))
            addc('P3_civ_founds', T, f"How many new cities will {nc} found by turn {T}?", [float(sum(1 for (t, p) in fv if 61 <= t <= T and p == c)) for fv in found], (c,))
            addc('P5_techs_at_T', T, f"How many technologies will {nc} know at turn {T}?", m['techs_known'][:, tj, ci], (c,))
        for c in civs:
            ci = cidx[c]; nc = name(c)
            for metric, noun in (('scores', 'score'), ('population', 'population'), ('cities_count', 'number of cities'), ('territory_size', 'territory size')):
                addc('NC1_value_at_T', T, f"What will {nc}'s {noun} be at turn {T}?", m[metric][:, tj, ci], (c,), metric=metric)
        addc('P6_world_techs', T, f"What will the combined number of technologies known by all five civilizations be at turn {T}?", np.nansum(m['techs_known'][:, tj, :], axis=1), ())
        addc('NC5_world_captures', T, f"How many cities will be captured (in total, across all civilizations) by turn {T}?", [float(sum(1 for (t, *_ ) in cv if 61 <= t <= T)) for cv in conq], ())
        addc('NC14_world_wonders', T, f"How many wonders will be completed (across all civilizations) between turn 61 and turn {T}?", [float(sum(1 for (t, p, w) in wv if 61 <= t <= T)) for wv in wond], ())
        addc('S7_world_razings', T, f"How many cities will be destroyed in total by turn {T}?", [float(sum(1 for t in dv if 61 <= t <= T)) for dv in destr], ())
    V = np.array(Vs, dtype=float) if Vs else np.zeros((0, n))
    # ---------------- reveals (turns 61-90) ----------------
    revs, Xs = [], []
    def addx(kind, text, x, subj, strength):
        x = np.asarray(x, bool)
        if x.sum() < 100: return
        revs.append(dict(world=world, kind=kind, text=text, subj=list(subj), strength=strength, nx=int(x.sum()))); Xs.append(x)
    for pi, (a, b) in enumerate(pairs):
        seg = dst[:, w61:w90 + 1, pi]; na, nb = name(a), name(b)
        addx('pair_war_began', f"{na} and {nb} went to war.", (seg == WAR).any(axis=1) & (dst[:, tidx[60], pi] != WAR), (a, b), 'diplo')
        addx('pair_ceasefire', f"{na} and {nb} reached a cease-fire or armistice.", np.isin(seg, (CEASE, ARMI)).any(axis=1), (a, b), 'weak')
        addx('pair_peace', f"{na} and {nb} signed a peace treaty.", (seg == PEACE).any(axis=1) & (dst[:, tidx[60], pi] != PEACE), (a, b), 'weak')
        addx('pair_contact', f"{na} and {nb} made first contact.", (seg != NOMET).any(axis=1) & (dst[:, tidx[60], pi] == NOMET), (a, b), 'weak')
    for c in civs:
        ci = cidx[c]; nc = name(c); cc = m['cities_count']
        addx('civ_captured', f"{nc} captured a city.", [any(61 <= t <= 90 and new == c for (t, prev, new) in cv) for cv in conq], (c,), 'weak')
        addx('civ_lost', f"{nc} lost a city.", [any(61 <= t <= 90 and prev == c for (t, prev, new) in cv) for cv in conq], (c,), 'loss')
        addx('civ_lost3', f"{nc} lost at least three cities.", [sum(1 for (t, prev, new) in cv if 61 <= t <= 90 and prev == c) >= 3 for cv in conq], (c,), 'magnitude')
        addx('civ_cities_fell2', f"{nc} had at least two fewer cities at turn 90 than at turn 60.", cc[:, w90, ci] <= cc[:, tidx[60], ci] - 2, (c,), 'magnitude')
        addx('civ_govchange', f"{nc} changed its form of government (not counting a spell of anarchy).", [any(61 <= t <= 90 and p == c and to != 'Anarchy' for (t, p, f_, to) in g_) for g_ in govch], (c,), 'weak')
        addx('civ_anarchy', f"{nc} fell into anarchy at some point.", (gov[:, w61:w90 + 1, ci] == ANARCHY).any(axis=1), (c,), 'weak')
        addx('civ_wonder', f"{nc} completed a wonder.", [any(61 <= t <= 90 and p == c for (t, p, w) in wv) for wv in wond], (c,), 'weak')
        addx('civ_cities_grew4', f"{nc} had at least four more cities at turn 90 than at turn 60.", cc[:, w90, ci] >= cc[:, tidx[60], ci] + 4, (c,), 'weak')
        sc = m['scores']; addx('civ_score_leader', f"{nc} had the highest score of any civilization at turn 90.", np.where(np.isnan(sc[:, w90, ci]), False, sc[:, w90, ci] > np.nanmax(sc[:, w90, [cidx[o] for o in civs if o != c]], axis=1)), (c,), 'loss')
        K = int(round(float(np.nanmedian(m['techs_known'][:, w90, ci])))); addx('civ_techs_K', f"{nc} knew at least {K} technologies at turn 90.", m['techs_known'][:, w90, ci] >= K, (c,), 'weak')
    X = np.array(Xs, dtype=bool) if Xs else np.zeros((0, n), bool)
    pickle.dump({'world': world, 'n': n, 'civ_names': d['civ_names'], 'civs': civs, 'bin': bins, 'Y': Y, 'cont': conts, 'V': V, 'reveals': revs, 'X': X, 'halfA': A, 'halfB': B}, open(os.path.join(OUT, f'{world}.pkl'), 'wb'), protocol=4)
    print(f"{world}: {len(bins)} binary ({dropped} duplicate texts dropped), {len(conts)} continuous, {len(revs)} reveals", flush=True)
