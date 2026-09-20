#!/usr/bin/env python3
"""consolidate_v4.py <corpus_root> <out_dir> [--workers N] [--anchors 7001 ...]

fbsim v3 corpus (2026-09-02 rerun) -> per-world pickles in the mine_v3.py schema, with every
channel taken from the server savegames (observer-independent):

  metrics[m][replay, turn, civ]   15 legacy series (data.json, savegame-overridden) + is_alive
  government[replay, turn, civ]   int8 code (gov_vocab), from the savegame government series
  dstate[replay, turn, pair]      diplomacy state code (STATE_VOCAB), cross-checked vs savegame tables
  events[replay]                  list of (turn, type, player_id:str, metadata tuple), EXACT, from the
                                  per-turn savegame tables (<user>_sgtables.json.gz):
     tech_discovered (tech_name,)                 government_change (from, to)
     city_conquered (prev_owner, new_owner, city_name, kind)   kind: capture | barbarian | civil_war
     city_founded (city_name,)   city_destroyed (city_name,)   wonder_completed (wonder_name, city_name)
     player_died ()              diplomacy_change (other, from, to)   [also emitted; legacy miners ignore]
     NOTE: kind=civil_war transfers are emitted as type 'civil_war_transfer', NOT city_conquered.
  end_turn[replay], truncated[replay]   last recorded turn; True when the fork carried a TRUNCATED marker
  player_kind, tech_vector, wonder_names, checks (per-world cross-check tallies)

Layout accepted: <root>/**/a<anchor>/forks/rng<N>/{seed<anchor>forkrng<N>_data.json.gz, *_sgtables.json.gz, TRUNCATED}
Split-half discipline identical to consolidate_v3.py: tags sorted numeric-aware, min tag reserved,
even index -> half A (selection), odd -> half B (scoring).
"""
import argparse, collections, glob, gzip, json, os, pickle, re, sys
import numpy as np
from multiprocessing import Pool

METRICS = ['treasury','population','science','scores','territory_size','arable_land',
           'food_production','shield_production','trade_production','culture',
           'techs_known','cities_count','units_count','military_units_count','wonders_count','is_alive']
STATE_VOCAB = {'Never met':0,'War':1,'Cease-fire':2,'Armistice':3,'Peace':4,'Alliance':5,'No Contact':0,'Team':5}
GOV_VOCAB = {'':-1,'Anarchy':0,'Despotism':1,'Tribal':2,'Monarchy':3,'Communism':4,'Fundamentalism':5,
             'Federation':6,'Republic':7,'Democracy':8}
N_MAIN = 5

def _tag_key(tag):
    m = re.fullmatch(r"([A-Za-z_]*)(\d+)", tag)
    return (m.group(1), int(m.group(2))) if m else (tag, -1)

def discover(root, anchors=None):
    by_world = collections.defaultdict(dict)
    for fp in sorted(glob.glob(f'{root}/**/*_data.json.gz', recursive=True)):
        m = re.match(r'(seed\d+)forkrng(\d+)_data', os.path.basename(fp))
        if not m: continue
        if anchors and int(m.group(1)[4:]) not in anchors: continue
        tag = 'rng' + m.group(2)
        by_world[m.group(1)].setdefault(tag, fp)      # first by sorted path wins (dedupe)
    return by_world

def load(fp):
    with gzip.open(fp, 'rt') as f: return json.load(f)

def wonder_set(ruleset_path):
    r = json.load(open(ruleset_path)); return {v['name'] for v in r['improvements'].values() if v.get('genus') == 0}

def exact_events(sg, wonders, tech_vec):
    """Rebuild the event list from the savegame tables, classifying owner changes."""
    tabs = sg['tables']; turns = sorted(int(t) for t in tabs)
    first_seen = {}
    for t in turns:
        for p, alive in tabs[str(t)]['alive'].items():
            first_seen.setdefault(p, t)
    barb = (sg.get('static') or {}).get('barbarian_type') or {}
    def kind_of(p): 
        if barb.get(p, 'None') not in ('None', 'NOT_A_BARBARIAN', None): return 'barbarian'
        return 'main' if int(p) < N_MAIN else 'splinter'
    city_name = {}
    for t in turns:
        for p, cl in tabs[str(t)]['cities'].items():
            for c in cl: city_name[c['id']] = c['name']
    out = []
    for e in sg['events']:
        t, ty = e.get('turn'), e.get('type')
        if ty == 'tech_discovered':
            tid = str(e.get('tech_id')); name = tech_vec[int(tid)] if tid.isdigit() and int(tid) < len(tech_vec) else tid
            out.append((t, 'tech_discovered', str(e['player_id']), (name,)))
        elif ty == 'government_change':
            out.append((t, 'government_change', str(e['player_id']), (e.get('from', ''), e.get('to', ''))))
        elif ty == 'city_conquered':
            prev, new = str(e['prev_owner']), str(e['new_owner'])
            if kind_of(new) == 'splinter' and first_seen.get(new) == t:
                out.append((t, 'civil_war_transfer', new, (prev, new, e.get('city_name', ''), 'civil_war')))
            else:
                k = 'barbarian' if kind_of(new) == 'barbarian' else 'capture'
                out.append((t, 'city_conquered', new, (prev, new, e.get('city_name', ''), k)))
        elif ty == 'city_founded':
            out.append((t, 'city_founded', str(e['player_id']), (e.get('city_name', ''),)))
        elif ty == 'city_destroyed':
            out.append((t, 'city_destroyed', str(e['player_id']), (e.get('city_name', ''),)))
        elif ty == 'improvement_built':
            if e.get('name') in wonders:
                out.append((t, 'wonder_completed', str(e['player_id']), (e['name'], city_name.get(e.get('city_id'), ''))))
        elif ty == 'player_died':
            out.append((t, 'player_died', str(e['player_id']), ()))
        elif ty == 'diplomacy_change':
            out.append((t, 'diplomacy_change', str(e['player_id']), (str(e.get('other')), e.get('from', ''), e.get('to', ''))))
    kinds = {p: kind_of(p) for p in first_seen}
    return sorted(out, key=lambda x: (x[0], x[1])), kinds

def do_world(args):
    world, tagmap, out_dir, wonders = args
    tags = sorted(tagmap, key=_tag_key)
    reserved, rest = tags[0], tags[1:]
    halfA_tags = {t for i, t in enumerate(rest) if i % 2 == 0}
    first = load(tagmap[tags[0]])
    turns = sorted(int(t) for t in first['time_series']['scores'])
    civs = sorted(first['time_series']['scores'][str(turns[0])].keys(), key=int)[:N_MAIN]
    civ_names = {cid: (ci.get('name') if isinstance(ci, dict) else str(ci)) for cid, ci in (first.get('civilizations') or {}).items()}
    pairs = [(a, b) for i, a in enumerate(civs) for b in civs[i+1:]]
    n_r, n_t, n_c = len(tags), len(turns), len(civs)
    tidx = {t: i for i, t in enumerate(turns)}
    arrs = {m: np.full((n_r, n_t, n_c), np.nan, dtype=np.float32) for m in METRICS}
    gov = np.full((n_r, n_t, n_c), -1, dtype=np.int8)
    dstate = np.full((n_r, n_t, len(pairs)), -1, dtype=np.int8)
    events, end_turn, truncated, kinds_all = [], np.zeros(n_r, int), np.zeros(n_r, bool), collections.Counter()
    checks = collections.Counter(); tech_vec = None; bad = []
    for ri, tag in enumerate(tags):
        fp = tagmap[tag]; d = load(fp); ts = d['time_series']
        fdir = os.path.dirname(fp)
        sgp = glob.glob(os.path.join(fdir, '*_sgtables.json.gz'))
        truncated[ri] = os.path.exists(os.path.join(fdir, 'TRUNCATED'))
        rturns = sorted(int(t) for t in ts['scores']); end_turn[ri] = rturns[-1]
        for m in METRICS:
            tm = ts.get(m) or {}
            for t, tj in tidx.items():
                row = tm.get(str(t))
                if row is None: continue
                for ci, c in enumerate(civs):
                    v = row.get(c)
                    if v is not None: arrs[m][ri, tj, ci] = float(v)
        gm = ts.get('government') or {}
        for t, tj in tidx.items():
            row = gm.get(str(t)) or {}
            for ci, c in enumerate(civs):
                g = row.get(c)
                if g is not None: gov[ri, tj, ci] = GOV_VOCAB.get(g, -1)
                if g is not None and g not in GOV_VOCAB: checks['unknown_gov:' + g] += 1
        rel = (d.get('diplomacy') or {}).get('relations') or {}
        for pi, (a, b) in enumerate(pairs):
            tt = rel.get(f'{a}_{b}') or rel.get(f'{b}_{a}') or {}
            for t, v in tt.items():
                tj = tidx.get(int(t))
                if tj is not None: dstate[ri, tj, pi] = STATE_VOCAB.get(str(v.get('state')), -1)
        if (arrs['scores'][ri] == -1).any(): checks['negative_scores'] += 1
        if not sgp:
            checks['missing_sgtables'] += 1; bad.append(tag); events.append([]); continue
        sg = load(sgp[0]); st = sg.get('static') or {}
        if tech_vec is None and st.get('technology_vector'): tech_vec = st['technology_vector']
        evs, kinds = exact_events(sg, wonders, tech_vec or [])
        events.append(evs); kinds_all.update(kinds.values())
        # wonders_count: GREAT wonders only, from the per-city improvement tables (the serializer's
        # series counts the Palace, a small wonder, for every civ)
        for t, tb in sg['tables'].items():
            tj = tidx.get(int(t))
            if tj is None: continue
            imps = tb.get('imps') or {}
            for ci, c in enumerate(civs):
                arrs['wonders_count'][ri, tj, ci] = float(sum(1 for cid, ws in (imps.get(c) or {}).items() for w_ in ws if w_ in wonders))
        # truncated forks: the serializer leaves the death turn's row empty; fill the exact per-player
        # values from the savegame tables so the last recorded turn is resolvable.
        tabs = sg['tables']
        for t in rturns:
            tj = tidx[t]; tb = tabs.get(str(t))
            if tb is None or not np.isnan(arrs['scores'][ri, tj, :]).all(): continue
            for ci, c in enumerate(civs):
                if c in tb['cities']: arrs['cities_count'][ri, tj, ci] = len(tb['cities'][c])
                if c in tb['techs']: arrs['techs_known'][ri, tj, ci] = len(tb['techs'][c])
                if c in tb['alive']: arrs['is_alive'][ri, tj, ci] = float(bool(tb['alive'][c]))
                if c in tb['gold']: arrs['treasury'][ri, tj, ci] = float(tb['gold'][c])
                if tb['scores'].get(c) is not None: arrs['scores'][ri, tj, ci] = float(tb['scores'][c])
                if c in tb['gov']: gov[ri, tj, ci] = GOV_VOCAB.get(tb['gov'][c], -1)
                if c in tb['cities']: arrs['population'][ri, tj, ci] = float(sum(int(x.get('size', 0)) for x in tb['cities'][c]))
            checks['rows_filled_from_tables'] += 1
        # cross-checks: savegame tables vs data.json series / dstate
        for t in rturns:
            tb = tabs.get(str(t))
            if tb is None: checks['sg_turn_missing'] += 1; continue
            tj = tidx[t]
            for ci, c in enumerate(civs):
                cc = tb['cities'].get(c)
                if cc is not None and not np.isnan(arrs['cities_count'][ri, tj, ci]) and len(cc) != int(arrs['cities_count'][ri, tj, ci]): checks['cities_mismatch'] += 1
                tk = tb['techs'].get(c)
                if tk is not None and not np.isnan(arrs['techs_known'][ri, tj, ci]) and len(tk) != int(arrs['techs_known'][ri, tj, ci]): checks['techs_mismatch'] += 1
                gv = tb['gov'].get(c)
                if gv is not None and GOV_VOCAB.get(gv, -1) != gov[ri, tj, ci]: checks['gov_mismatch'] += 1
                al = tb['alive'].get(c)
                if al is not None and not np.isnan(arrs['is_alive'][ri, tj, ci]) and bool(al) != bool(arrs['is_alive'][ri, tj, ci]): checks['alive_mismatch'] += 1
            for pi, (a, b) in enumerate(pairs):
                s = (tb['dipl'].get(a) or {}).get(b)
                if s is not None and STATE_VOCAB.get(s, -1) != dstate[ri, tj, pi]: checks['dstate_mismatch'] += 1
            checks['cells_checked'] += n_c
    halfA = np.array([t in halfA_tags for t in tags])
    halfB = np.array([(t not in halfA_tags) and t != reserved for t in tags])
    wonder_names = sorted({e[3][0] for evs in events for e in evs if e[1] == 'wonder_completed'})
    out = {'world': world, 'turns': turns, 'civs': civs, 'civ_names': civ_names, 'pairs': pairs, 'tags': tags,
           'reserved': reserved, 'halfA': halfA, 'halfB': halfB, 'metrics': arrs, 'government': gov,
           'gov_vocab': GOV_VOCAB, 'dstate': dstate, 'state_vocab': STATE_VOCAB, 'events': events,
           'end_turn': end_turn, 'truncated': truncated, 'player_kind_counts': dict(kinds_all),
           'tech_vector': tech_vec, 'wonder_names': wonder_names, 'events_source': 'savegame_tables_v4', 'wonders_count_source': 'savegame great wonders (genus 0) only',
           'checks': dict(checks), 'unreadable': bad}
    with open(f'{out_dir}/{world}.pkl', 'wb') as fh: pickle.dump(out, fh, protocol=4)
    return world, n_r, int(halfA.sum()), int(halfB.sum()), n_t, int(truncated.sum()), dict(checks), bad

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('root'); ap.add_argument('out')
    ap.add_argument('--workers', type=int, default=4); ap.add_argument('--anchors', nargs='*', type=int)
    ap.add_argument('--ruleset', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'state', 'ruleset.json'))
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    wonders = wonder_set(a.ruleset)
    bw = discover(a.root, a.anchors)
    print({w: len(t) for w, t in sorted(bw.items())}, flush=True)
    summary = {}
    with Pool(a.workers) as p:
        for w, nr, na, nb, nt, ntr, checks, bad in p.imap_unordered(do_world, [(w, t, a.out, wonders) for w, t in sorted(bw.items())]):
            print(f'{w}: {nr} replays (A {na} / B {nb} / 1 reserved), {nt} turns, truncated {ntr}, checks {checks}, unreadable {bad[:5]}', flush=True)
            summary[w] = dict(replays=nr, halfA=na, halfB=nb, turns=nt, truncated=ntr, checks=checks, unreadable=bad)
    json.dump(summary, open(a.out.rstrip('/') + '_SUMMARY.json', 'w'), indent=1)
