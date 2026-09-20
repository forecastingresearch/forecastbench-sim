#!/usr/bin/env python3
"""sweep_v4.py <consol_dir> <out_json>  — the S / W / P family sweeps (spec rev 26.1) on v4 pickles.

Rewritten 2026-09-02 from the family_specs semantics (the original sweep scripts were not retained; only
sweep2.json / sweep3.json / cont_expansion.json outputs were). Output format matches those files:
binary instances {world, T, qB, qA, n, text, ...params}; continuous {world, T, text, medA, medB, iqrA, iqrB,
p05, p95, (censB)}. Selection parameters (k grids, K quartiles, top-2 builders) from half A; truth from half B.
Semantics: windows are turns 61..T inclusive; 'at exactly T' = state at T; ties -> NO.
Conquest semantics (v4 exact events): a capture = a city changing owner (kind capture|barbarian);
civil-war transfers are NOT captures. Civ-level captures require new_owner == civ; civ-level losses require
prev_owner == civ (losses to barbarians count). World counts include captures by barbarians/splinters.
"""
import collections, itertools, json, os, pickle, sys
import numpy as np

CONSOL, OUTF = sys.argv[1], sys.argv[2]
WAR, CEASE, ARMI, PEACE, ALLY, NOMET = 1, 2, 3, 4, 5, 0
R = {}
def add(t, i): R.setdefault(t, []).append(i)
def addbin(w, typ, T, text, yes, **kw):
    add(typ, dict(world=w['world'], T=T, qB=float(np.mean(yes[w['halfB']])), qA=float(np.mean(yes[w['halfA']])),
                  qAll=float(np.mean(yes)), n=int(w['halfB'].sum()), nAll=int(len(yes)), text=text, **kw))
def addcont(w, typ, T, text, vals, cens=False, **kw):
    vA, vB = vals[w['halfA']], vals[w['halfB']]
    fA, fB = vA[~np.isnan(vA)], vB[~np.isnan(vB)]
    if len(fA) < 5 or len(fB) < 5: return
    fAll = vals[~np.isnan(vals)]
    d = dict(world=w['world'], T=T, text=text, medA=float(np.median(fA)), medB=float(np.median(fB)), medAll=float(np.median(fAll)),
             p05All=float(np.percentile(fAll, 5)), p95All=float(np.percentile(fAll, 95)), nAllv=int(len(fAll)),
             iqrA=float(np.subtract(*np.percentile(fA, [75, 25]))), iqrB=float(np.subtract(*np.percentile(fB, [75, 25]))),
             p05=float(np.percentile(fB, 5)), p95=float(np.percentile(fB, 95)), nB=int(len(fB)), **kw)
    if cens: d['censB'] = float(np.mean(np.isnan(vB)))
    add(typ, d)

for fn in sorted(f for f in os.listdir(CONSOL) if f.endswith('.pkl')):
    w = pickle.load(open(f'{CONSOL}/{fn}', 'rb'))
    world, turns, civs, pairs = w['world'], w['turns'], w['civs'], w['pairs']
    tidx = {t: i for i, t in enumerate(turns)}; n_r = len(w['tags']); A = w['halfA']; idxA = np.where(A)[0]
    m, dst, gov, evs = w['metrics'], w['dstate'], w['government'], w['events']
    name = lambda cid: w['civ_names'].get(str(cid), str(cid))
    alive60 = [c for ci, c in enumerate(civs) if np.nanmedian(m['techs_known'][:, 0, ci]) > 0]
    cidx = {c: i for i, c in enumerate(civs)}
    HZ = [t for t in (90, 120, 150, 180, 210) if t in tidx]; TMAX = HZ[-1]
    pair_alive = [pi for pi, (a, b) in enumerate(pairs) if a in alive60 and b in alive60]
    # per-replay event tables
    conq = [[(t, d[0], d[1], d[2], d[3]) for (t, ty, p, d) in ev if ty == 'city_conquered' and t > 60] for ev in evs]
    found = [[(t, p) for (t, ty, p, d) in ev if ty == 'city_founded' and t > 60] for ev in evs]
    destr = [[t for (t, ty, p, d) in ev if ty == 'city_destroyed' and t > 60] for ev in evs]
    wond = [[(t, p, d[0]) for (t, ty, p, d) in ev if ty == 'wonder_completed' and t > 60] for ev in evs]
    govch = [[(t, p, d[0], d[1]) for (t, ty, p, d) in ev if ty == 'government_change' and t > 60] for ev in evs]
    techs = [{(p, d[0]): t for (t, ty, p, d) in ev if ty == 'tech_discovered' and t > 60} for ev in evs]
    def win(T): return slice(tidx[61], tidx[T] + 1)
    for T in HZ:
        dT = dst[:, tidx[T], :]; dU = dst[:, win(T), :]
        # ---- S families -------------------------------------------------------------
        for pi in pair_alive:
            a, b = pairs[pi]
            addbin(w, 'S1_met_by', T, f"Will {name(a)} and {name(b)} have made contact by turn {T}?",
                   (dU[:, :, pi] > NOMET).any(axis=1), pair=[a, b])
            if (dst[:, 0, pi] == WAR).all():
                addbin(w, 'S2_war_end', T, f"Will the war between {name(a)} and {name(b)} (ongoing at turn 60) be interrupted by any other state by turn {T}?",
                       (dU[:, :, pi] != WAR).any(axis=1), pair=[a, b])
        wars = (dT[:, pair_alive] == WAR).sum(axis=1).astype(float)
        for k in (1, 2, 3, 4, 5):
            addbin(w, 'S3b_wars_ge_k', T, f"Will at least {k} pairs of civilizations be at war at exactly turn {T}?", wars >= k, k=k)
        addcont(w, 'S3c_wars_at_T', T, f"How many pairs of civilizations will be at war at exactly turn {T}?", wars)
        for c in alive60:
            gc = np.array([sum(1 for (t, p, f_, to) in g if p == c and t <= T and to != 'Anarchy') for g in govch], float)
            for k in (1, 2, 3):
                addbin(w, 'S4b_govchanges_ge_k', T, f"Will {name(c)} change its government at least {k} time{'s' if k>1 else ''} by turn {T}?", gc >= k, civ=c, k=k)
            addcont(w, 'S4c_govchanges', T, f"How many times will {name(c)} change its government by turn {T}?", gc, civ=c)
            wc = np.array([sum(1 for (t, p, d) in wv if p == c and t <= T) for wv in wond], float)
            for k in (1, 2, 3):
                addbin(w, 'S5_civ_wonders_ge_k', T, f"Will {name(c)} complete at least {k} wonder{'s' if k>1 else ''} by turn {T}?", wc >= k, civ=c, k=k)
            fc = np.array([sum(1 for (t, p) in fv if p == c and t <= T) for fv in found], float)
            for k in (1, 3, 5, 8):
                addbin(w, 'S6_founds_ge_k', T, f"Will {name(c)} found at least {k} new cit{'ies' if k>1 else 'y'} by turn {T}?", fc >= k, civ=c, k=k)
            addcont(w, 'P3_civ_founds', T, f"How many new cities will {name(c)} found by turn {T}?", fc, civ=c)
            cap = np.array([sum(1 for (t, prev, new, city, kind) in cv if new == c and t <= T) for cv in conq], float)
            los = np.array([sum(1 for (t, prev, new, city, kind) in cv if prev == c and t <= T) for cv in conq], float)
            for k in (1, 2, 3, 5):
                addbin(w, 'W3_capture_k', T, f"Will {name(c)} capture at least {k} cit{'ies' if k>1 else 'y'} by turn {T}?", cap >= k, civ=c, k=k)
            for k in (1, 2, 3, 5):
                addbin(w, 'W4_lose_k', T, f"Will {name(c)} lose at least {k} cit{'ies' if k>1 else 'y'} by turn {T}?", los >= k, civ=c, k=k)
            addcont(w, 'P1_civ_conquests', T, f"How many cities will {name(c)} capture by turn {T}?", cap, civ=c)
            addcont(w, 'P2_civ_losses', T, f"How many cities will {name(c)} lose by turn {T}?", los, civ=c)
            for c2 in alive60:
                if c2 == c: continue
                yes = np.array([any(new == c and prev == c2 and t <= T for (t, prev, new, city, kind) in cv) for cv in conq])
                addbin(w, 'W2_directed_conq', T, f"Will {name(c)} capture at least one city from {name(c2)} by turn {T}?", yes, civA=c, civB=c2)
            tk = m['techs_known'][:, tidx[T], :]
            others = [cidx[o] for o in alive60 if o != c]
            lead = np.where(np.isnan(tk[:, cidx[c]]), False, tk[:, cidx[c]] > np.nanmax(tk[:, others], axis=1))
            addbin(w, 'W5_tech_lead', T, f"Will {name(c)} know strictly more technologies than every other civilization at turn {T}?", lead, civ=c)
            addcont(w, 'P5_techs_at_T', T, f"How many technologies will {name(c)} know at turn {T}?", tk[:, cidx[c]], civ=c)
        for pi in pair_alive:
            a, b = pairs[pi]
            addbin(w, 'W6_peace_at', T, f"Will {name(a)} and {name(b)} have a formal peace in effect at turn {T}?", dT[:, pi] == PEACE, pair=[a, b])
        addbin(w, 'S7_any_destroyed', T, f"Will any city be destroyed (razed) by turn {T}?", np.array([any(t <= T for t in dv) for dv in destr]))
        addcont(w, 'S7c_destroyed_count', T, f"How many cities will be destroyed (razed) in total by turn {T}?", np.array([float(sum(t <= T for t in dv)) for dv in destr]))
        addcont(w, 'P6_world_techs', T, f"What will the combined number of technologies known by all five civilizations be at turn {T}?",
                np.nansum(m['techs_known'][:, tidx[T], [cidx[c] for c in alive60]], axis=1))
        addcont(w, 'NC5_count_v4', T, f"How many cities will be conquered (in total, all civs) by turn {T}?", np.array([float(sum(t <= T for (t, *_) in cv)) for cv in conq]))
        addcont(w, 'NC14_wonders_built_v4', T, f"How many wonders will be completed (all civs) between turn 61 and {T}?", np.array([float(sum(t <= T for (t, p, d) in wv)) for wv in wond]))
        # W1 wonder race: first completion after 60 of wonder w, top-2 builders by half-A frequency
        firsts = [{} for _ in range(n_r)]
        for r, wv in enumerate(wond):
            for (t, p, d) in sorted(wv):
                if t <= T and d not in firsts[r]: firsts[r][d] = p
        cnt = collections.Counter((d, p) for r in idxA for d, p in firsts[r].items() if p in alive60)
        bywonder = collections.defaultdict(list)
        for (d, p), c in cnt.items(): bywonder[d].append((c, p))
        for d, lst in bywonder.items():
            for c, p in sorted(lst, reverse=True)[:2]:
                yes = np.array([firsts[r].get(d) == p for r in range(n_r)])
                addbin(w, 'W1_wonder_race', T, f"Will {name(p)}, rather than any other civilization, be the one to complete {d} by turn {T}?", yes, civ=p, wonder=d)
    # ---- timing families (T210-anchored) ---------------------------------------------
    T = TMAX
    for c in alive60:
        fc = np.array([min((t for (t, prev, new, city, kind) in cv if new == c and t <= T), default=np.nan) for cv in conq], float)
        addcont(w, 'P7_first_capture', T, f"On what turn will {name(c)} first capture a city? ('never by {T}' allowed)", fc, cens=True, civ=c)
        fw = np.array([min((t for (t, p, d) in wv if p == c and t <= T), default=np.nan) for wv in wond], float)
        addcont(w, 'P8_first_wonder', T, f"On what turn will {name(c)} complete its first wonder? ('never by {T}' allowed)", fw, cens=True, civ=c)
        path = m['techs_known'][:, win(T), cidx[c]]
        vA = m['techs_known'][A, tidx[T], cidx[c]]; vA = vA[~np.isnan(vA)]
        if len(vA) < 5: continue
        for q in (0.25, 0.5, 0.75):
            K = int(round(float(np.quantile(vA, q))))
            hit = np.where(np.isnan(path), False, path >= K)
            ft = np.where(hit.any(axis=1), hit.argmax(axis=1) + 61.0, np.nan)
            addcont(w, 'P9_time_to_K_multi', T, f"On what turn will {name(c)} first know at least {K} technologies? ('never by {T}' allowed)", ft, cens=True, civ=c, K=K, q=q)
    print(f"{world}: alive {len(alive60)}/5, HZ {HZ}, replays {n_r}", flush=True)
json.dump(R, open(OUTF, 'w'))
print({k: len(v) for k, v in R.items()}); print('wrote', OUTF)
