#!/usr/bin/env python3
"""draw_v1.py BANK_DIR OUT_DIR [--seed 2026]
Stratified random draws (structural strata only; never on the noisy estimate beyond the band a question sits in):
  bank      750 = 5 horizons x 5 bands over (.05,.95) x 30; caps: <=6/world and <=6/family per band-cell, NB1_threshold <=150 total
  tails     300 = 60/horizon with qAll <= .05 (0 < q); caps NB1 <=45 total, others <=40 total and <=10/family/horizon, <=12/world/horizon
  mirrors    50 = 10/horizon with qAll >= .95 (q < 1), same families as tails
  continuous 300 = 240 spread (~6/family/horizon over 8 families, T90 4) + 60 timing at T210 (P9 40, P7 15, P8 5); IQR>0
  natcond   600 = 4 horizons (120..210) x 150 from BANK questions x reveals, blocks A30 B20 C1 20 C2 50 D30,
            template-balanced inside blocks, n_x>=100, <=2 reveals (distinct kinds) per question, <=8 cells per revealed event
Truth for everything = all replays. Effect sizes for natcond are reported after the draw, never used for it."""
import os, sys, json, pickle, random, collections, numpy as np
BANK, OUT = sys.argv[1], sys.argv[2]; SEED = int(sys.argv[sys.argv.index('--seed') + 1]) if '--seed' in sys.argv else 2026
os.makedirs(OUT, exist_ok=True); rng = random.Random(SEED)
def opt(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default
NC_QUOTA = {k: int(v) for k, v in (kv.split('=') for kv in opt('--nc-quota', 'A=30,B=20,C1=20,C2=50,D=30').split(','))}
STRONG_ONLY = '--strong-only' in sys.argv       # C2/D restricted to magnitude reveals (lost>=3, cities fell>=2, war began)
STRONG_FIRST = '--strong-first' in sys.argv     # C2/D: strong reveals drawn first, weaker ones only fill what remains
STRONG = ('civ_lost3', 'civ_cities_fell2', 'pair_war_began')
W = {fn[:-4]: pickle.load(open(os.path.join(BANK, fn), 'rb')) for fn in sorted(os.listdir(BANK)) if fn.endswith('.pkl')}
HZ = (90, 120, 150, 180, 210); BANDS = [(0.05, 0.23), (0.23, 0.41), (0.41, 0.59), (0.59, 0.77), (0.77, 0.95)]
def band_of(q): return next((i for i, (lo, hi) in enumerate(BANDS) if lo < q <= hi), None)
# ---------------- pools ----------------
binpool = []
for w, D in W.items():
    for i, inst in enumerate(D['bin']): binpool.append(dict(inst, id=f"{w}:b{i}", idx=i))
def subject_key(i): return (i['world'], tuple(sorted(i['subj'])), i['family'], i['T'])
def draw_cells(cands, per_cell, cell_key, caps, order=None, balance=None):
    """random draw per cell with caps; with balance='family' the cell is filled round-robin over families
    (each family's candidates shuffled), so every family present gets an equal share before any family repeats."""
    chosen = []; counts = collections.Counter(); cells = collections.defaultdict(list)
    for c in cands: cells[cell_key(c)].append(c)
    keys = order or sorted(cells)
    for key in keys:
        lst = cells[key][:]; rng.shuffle(lst); got = 0
        ok = lambda c: not any(counts[(k.__name__, k(c), scope(key))] >= mx for k, mx, scope in caps)
        def take(c):
            nonlocal got; chosen.append(c); got += 1
            for k, mx, scope in caps: counts[(k.__name__, k(c), scope(key))] += 1
        if balance == 'family':
            fams = sorted({c['family'] for c in lst}); rng.shuffle(fams); by = {f: [c for c in lst if c['family'] == f] for f in fams}
            i = 0; stall = 0
            while got < per_cell and stall < len(fams) + 1:
                f = fams[i % len(fams)]; i += 1; stall += 1
                while by[f]:
                    c = by[f].pop()
                    if ok(c): take(c); stall = 0; break
        else:
            for c in lst:
                if got >= per_cell: break
                if ok(c): take(c)
    return chosen
def world(c): return c['world']
def family(c): return c['family']
def subj(c): return subject_key(c)
CELL = lambda key: key; HORIZON = lambda key: key[1] if isinstance(key, tuple) else key; GLOBAL = lambda key: 'g'
# bank
rng = random.Random(SEED + 1)
cands = [c for c in binpool if band_of(c['qAll']) is not None]
bank = draw_cells(cands, 30, lambda c: (band_of(c['qAll']), c['T']),
                  caps=[(world, 6, CELL), (family, 6, CELL), (subj, 1, GLOBAL), (family, 150, GLOBAL)], balance='family')
# NB1 techs filler cap handled by (family,150,GLOBAL) for every family; fine since others never reach 150
used = {c['id'] for c in bank}
# tails: long horizons first
tcands = [c for c in binpool if 0 < c['qAll'] <= 0.05 and c['id'] not in used]
tails = draw_cells(tcands, 60, lambda c: c['T'], caps=[(family, 10, CELL), (world, 12, CELL), (subj, 1, GLOBAL), (family, 45, GLOBAL)], order=[210, 180, 150, 120, 90], balance='family')
used |= {c['id'] for c in tails}
mcands = [c for c in binpool if 0.95 <= c['qAll'] < 1 and c['id'] not in used]
mirrors = draw_cells(mcands, 10, lambda c: c['T'], caps=[(family, 3, CELL), (world, 3, CELL), (subj, 1, GLOBAL)], balance='family')
# continuous
rng = random.Random(SEED + 2)
contpool = []
for w, D in W.items():
    for i, inst in enumerate(D['cont']): contpool.append(dict(inst, id=f"{w}:c{i}", idx=i))
COUNT_FAMS = ('P1_civ_conquests', 'P2_civ_losses', 'P3_civ_founds', 'P5_techs_at_T', 'P6_world_techs', 'NC5_world_captures', 'NC14_world_wonders', 'S7_world_razings')
spread = [c for c in contpool if c['family'] in COUNT_FAMS and c['iqrAll'] > 0 and c['medAll'] > 0]
values = [c for c in contpool if c['family'] == 'NC1_value_at_T' and c['iqrAll'] > 0 and c['medAll'] > 0]
cont = []
for T in HZ:
    got = draw_cells([c for c in spread if c['T'] == T], 6, lambda c: (c['family'], c['T']), caps=[(world, 2, CELL), (subj, 1, GLOBAL)])
    got += draw_cells([c for c in values if c['T'] == T], 3, lambda c: (c['metric'], c['T']), caps=[(world, 1, CELL), (subj, 1, GLOBAL)])
    if len(got) < 60:   # a family short of supply (e.g. world techs where a civ may be eliminated): top up from the other count families
        used = {c['id'] for c in got}; pool = [c for c in spread if c['T'] == T and c['id'] not in used]
        got += draw_cells(pool, 60 - len(got), lambda c: c['T'], caps=[(world, 3, CELL), (subj, 1, GLOBAL), (family, 9, CELL)], balance='family')
    cont += got
# ---------------- natcond from bank questions ----------------
value_fams = {'NB1_value_threshold', 'EX_comparative', 'NB4_drawdown'}
state_fams = {'NW1_war_at', 'W6_peace_at', 'EX_government_at', 'W5_tech_lead', 'NW2_diplo', 'NW5_wonder', 'W1_wonder_race', 'S3_wars_at_T'}
infer_fams = {'NW1_war_at', 'NW2_diplo', 'NEW_civil_war', 'W2_directed_conquest', 'W4_lose_k', 'NW4_survival', 'W6_peace_at'}
def block(rev, q):
    related = bool(set(rev['subj']) & set(q['subj']))
    if not related: return 'A'
    st = rev['strength']
    if st in ('diplo', 'loss', 'magnitude') and q['family'] in infer_fams: return 'D'
    if STRONG_ONLY and st in ('diplo', 'loss', 'magnitude') and rev['kind'] not in ('civ_lost3', 'civ_cities_fell2', 'pair_war_began'): return None
    if st in ('magnitude', 'loss') and q['family'] in value_fams:
        same = (rev['kind'] in ('civ_lost3', 'civ_cities_fell2')) and q.get('metric') == 'cities_count'
        return 'C1' if same else 'C2'
    if st == 'weak' and (q['family'] in state_fams or q['family'] in value_fams): return 'B'
    return None
QUOTA = NC_QUOTA; rng = random.Random(SEED + 3)
bank_by_world = collections.defaultdict(list)
for q in bank:
    if q['T'] >= 120: bank_by_world[q['world']].append(dict(q, from_bank=True))
# top-up pool for blocks C1/C2: value-series questions NOT in the bank (same bands, same subject rule); they get
# their own turn-1 elicitation (natcond_extra_turn1.json) so cells still condition on a turn-1 answer.
bank_ids = {q['id'] for q in bank}; extra_by_world = collections.defaultdict(list)
for c in binpool:
    if c['T'] >= 120 and c['family'] in value_fams and c['id'] not in bank_ids and band_of(c['qAll']) is not None:
        extra_by_world[c['world']].append(dict(c, from_bank=False))
cells = []
for w, D in W.items():
    X, Y = D['X'], D['Y']
    for ri, rev in enumerate(D['reveals']):
        x = X[ri]; nx = int(x.sum())
        if nx < 100: continue
        for q in bank_by_world[w] + extra_by_world[w]:
            b = block(rev, q)
            if b is None: continue
            if not q['from_bank'] and b not in ('C1', 'C2'): continue
            y = Y[q['idx']]; p = float(y.mean()); pyx = float(y[x].mean())
            if pyx <= 0.02 or pyx >= 0.98: continue          # admissibility: the reveal must not fix the answer
            cells.append(dict(world=w, T=q['T'], block=b, qid=q['id'], question=q['text'], family=q['family'], from_bank=q['from_bank'], rev_id=f"{w}:r{ri}", reveal=rev['text'], rev_kind=rev['kind'], p=p, p_given=pyx, delta=pyx - p, nx=nx))
natcond = []; q_used = collections.Counter(); q_kinds = collections.defaultdict(set); ev_count = collections.Counter()
TOTAL_PER_T = int(opt('--total-per-horizon', '0'))      # if set, block shortfalls are refilled from C2, then D, C1, B, A
def fill_block(T, b, qn):
    pool = [c for c in cells if c['T'] == T and c['block'] == b]
    fams = sorted({c['family'] for c in pool}); rng.shuffle(fams)
    by_fam = {f: [c for c in pool if c['family'] == f] for f in fams}
    for lst in by_fam.values():
        rng.shuffle(lst); lst.sort(key=lambda c: c['from_bank'])
        if STRONG_FIRST and b in ('C2', 'D'): lst.sort(key=lambda c: c['rev_kind'] in STRONG)
    got = 0; i = 0; stall = 0
    while got < qn and fams and stall < 5 * len(fams) + 10:
        f = fams[i % len(fams)]; i += 1; stall += 1; lst = by_fam[f]
        while lst:
            c = lst.pop()
            if q_used[c['qid']] >= 2 or c['rev_kind'] in q_kinds[c['qid']] or ev_count[c['rev_id']] >= 8: continue
            natcond.append(c); q_used[c['qid']] += 1; q_kinds[c['qid']].add(c['rev_kind']); ev_count[c['rev_id']] += 1; got += 1; stall = 0; break
    return got
for T in (120, 150, 180, 210):
    got_T = 0
    for b, qn in QUOTA.items():
        got = fill_block(T, b, qn); got_T += got
        if got < qn: print(f"  natcond shortfall T{T} block {b}: {got}/{qn}")
    if TOTAL_PER_T and got_T < TOTAL_PER_T:
        for b in ('C2', 'D', 'C1', 'B', 'A'):
            if got_T >= TOTAL_PER_T: break
            extra = fill_block(T, b, TOTAL_PER_T - got_T); got_T += extra
            if extra: print(f"  T{T}: refilled {extra} from block {b}")
    if False:
        pool = [c for c in cells if c['T'] == T and c['block'] == b]
        pass
# ---------------- controls ----------------
rng_c = random.Random(SEED + 4); n_nc = len(natcond)
nn = set(rng_c.sample(range(n_nc), min(300, n_nc))); sp = set(rng_c.sample(range(n_nc), min(300, n_nc)))   # independent arms
for i, c in enumerate(natcond): c['control_no_news'] = i in nn; c['control_single_prompt'] = i in sp
# ---------------- write ----------------
strip = lambda c: {k: v for k, v in c.items() if k != 'idx'}
json.dump([strip(c) for c in bank], open(f'{OUT}/bank_750.json', 'w'), indent=0)
json.dump([strip(c) for c in tails], open(f'{OUT}/tails_300.json', 'w'), indent=0)
json.dump([strip(c) for c in mirrors], open(f'{OUT}/mirrors_50.json', 'w'), indent=0)
json.dump([strip(c) for c in cont], open(f'{OUT}/continuous_300.json', 'w'), indent=0)
json.dump(natcond, open(f'{OUT}/natcond_600.json', 'w'), indent=0)
extra_q = {c['qid']: c for c in natcond if not c['from_bank']}
by_id = {c['id']: c for c in binpool}
json.dump([strip(by_id[q]) for q in sorted(extra_q)], open(f'{OUT}/natcond_extra_turn1.json', 'w'), indent=0)
print(f"natcond cells from bank {sum(c['from_bank'] for c in natcond)}, from extra value pool {sum(not c['from_bank'] for c in natcond)} ({len(extra_q)} extra turn-1 questions)")
import pandas as _pd; _n=_pd.DataFrame(natcond); _n['abs']=_n.delta.abs(); _n['st']=_pd.cut(_n['abs'],[0,.03,.08,.15,1.01],labels=['null','small','medium','large'],right=False)
print('natcond composition:', _pd.crosstab(_n.block,_n.st,margins=True).to_string())
print(f"bank {len(bank)}  tails {len(tails)}  mirrors {len(mirrors)}  continuous {len(cont)}  natcond {len(natcond)}  (natcond pool {len(cells)} cells)")
