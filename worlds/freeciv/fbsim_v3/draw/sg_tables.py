#!/usr/bin/env python3
"""sg_tables.py FORK_DIR [FORK_DIR ...] [--out-suffix _sgtables.json.gz] [--workers N]

Exact, observer-independent per-turn tables from the retained server savegames of a fork
(or anchor): for every turn and every player -> alive, government, gold, score, techs (ids),
cities [{id,name,size,x,y}], wonders (improvement genus GreatWonder per city), and pairwise
diplomacy states; plus events DERIVED from consecutive savegames:
  city_founded (new city id under owner), city_conquered (id changes owner), city_destroyed
  (id disappears), government_change, tech_discovered (set diff), wonder_completed (new wonder
  in any city), diplomacy_change (pair state change), player_died (alive True->False).
Writes <fork_dir>/<user>_sgtables.json.gz. Uses the bundle's savegame_parser.
"""
import argparse, glob, gzip, json, os, re, sys
from multiprocessing import Pool
BUNDLE = '/Users/jaeholee0404/civbench/tmp/pilot_v2/fleet_prep/pod_bundle'
if os.path.isdir('/workspace/civbench'): BUNDLE = '/workspace/civbench'
sys.path.insert(0, os.path.join(BUNDLE, 'worlds', 'freeciv'))
from freeciv_world.world_reports.utils import savegame_parser as sp  # noqa

def load_save(path):
    raw = open(path, 'rb').read()
    return sp.decompress_savegame_content(raw, path)

def wonders_in(txt, cities):
    """{pid: {city_id: [wonder_name,...]}} using improvement_vector + ruleset genus when available."""
    try:
        vec = sp.parse_improvement_vector(txt)
    except Exception:
        vec = []
    out = {}
    for pid, cl in cities.items():
        for c in cl:
            imp = c.get('improvements') or ''
            names = [vec[i] for i, ch in enumerate(imp) if ch == '1' and i < len(vec)]
            if names:
                out.setdefault(pid, {})[c['id']] = names
    return out

def vectors_for(txt):
    """Static per-game vectors from the [savefile] section: tech id -> name, improvement bit -> name, ruleset."""
    def vec(key):
        m = re.search(r'^' + key + r'=(.*?)(?=^\w+=|^\[)', txt, re.S | re.M)
        return re.findall(r'"([^"]*)"', m.group(1)) if m else []
    m = re.search(r'^rulesetdir="?([^"\n]*)"?', txt, re.M)
    try:
        barb = sp.parse_player_barbarian_types(txt)
    except Exception:
        barb = {}
    return {'technology_vector': vec('technology_vector'), 'improvement_vector': vec('improvement_vector'),
            'rulesetdir': m.group(1) if m else '', 'barbarian_type': barb}

def table_for(path):
    txt = load_save(path)
    cities = sp.parse_player_cities(txt)
    return {
        'alive': sp.parse_player_alive(txt),
        'gov': sp.parse_player_governments(txt),
        'gold': sp.parse_player_gold(txt),
        'scores': {p: r.get('total') for p, r in sp.parse_player_scores(txt).items()},
        'techs': {p: sorted(v, key=lambda x: int(x) if str(x).isdigit() else 10**6) for p, v in sp.parse_player_technologies(txt).items()},
        'cities': cities,
        'imps': wonders_in(txt, cities),
        'dipl': {p: {q: r.get('state') for q, r in d.items()} for p, d in sp.parse_player_diplomacy(txt).items()},
        'nations': sp.parse_player_nations(txt),
    }

def turn_of(name):
    m = re.search(r'_T(\d+)_', name); return int(m.group(1)) if m else None

def process(fork_dir):
    saves = sorted(glob.glob(os.path.join(fork_dir, 'savegames', '*_T*.sav*')), key=lambda p: turn_of(os.path.basename(p)))
    if not saves: return fork_dir, 'no savegames'
    user = re.sub(r'_T\d+_.*$', '', os.path.basename(saves[0]))
    tables, events, prev, prev_t = {}, [], None, None
    for path in saves:
        t = turn_of(os.path.basename(path))
        try:
            tb = table_for(path)
        except Exception as e:
            events.append({'turn': t, 'type': 'parse_error', 'error': repr(e)}); continue
        tables[t] = tb
        if prev is not None:
            # cities by id -> owner
            pc = {c['id']: (p, c) for p, cl in prev['cities'].items() for c in cl}
            cc = {c['id']: (p, c) for p, cl in tb['cities'].items() for c in cl}
            for cid, (p, c) in cc.items():
                if cid not in pc: events.append({'turn': t, 'type': 'city_founded', 'player_id': p, 'city_id': cid, 'city_name': c['name']})
                elif pc[cid][0] != p: events.append({'turn': t, 'type': 'city_conquered', 'player_id': p, 'prev_owner': pc[cid][0], 'new_owner': p, 'city_id': cid, 'city_name': c['name']})
            for cid, (p, c) in pc.items():
                if cid not in cc: events.append({'turn': t, 'type': 'city_destroyed', 'player_id': p, 'city_id': cid, 'city_name': c['name']})
            for p, g in tb['gov'].items():
                if prev['gov'].get(p) is not None and prev['gov'].get(p) != g:
                    events.append({'turn': t, 'type': 'government_change', 'player_id': p, 'from': prev['gov'].get(p), 'to': g})
            for p, techs in tb['techs'].items():
                for x in set(techs) - set(prev['techs'].get(p, [])):
                    events.append({'turn': t, 'type': 'tech_discovered', 'player_id': p, 'tech_id': x})
            pw = {(p, cid, w) for p, d in prev['imps'].items() for cid, ws in d.items() for w in ws}
            for p, d in tb['imps'].items():
                for cid, ws in d.items():
                    for w in ws:
                        if (p, cid, w) not in pw: events.append({'turn': t, 'type': 'improvement_built', 'player_id': p, 'city_id': cid, 'name': w})
            for p, d in tb['dipl'].items():
                for q, s in d.items():
                    ps = prev['dipl'].get(p, {}).get(q)
                    if ps is not None and ps != s: events.append({'turn': t, 'type': 'diplomacy_change', 'player_id': p, 'other': q, 'from': ps, 'to': s})
            for p, a in tb['alive'].items():
                if prev['alive'].get(p) and not a: events.append({'turn': t, 'type': 'player_died', 'player_id': p})
        prev, prev_t = tb, t
    try:
        static = vectors_for(load_save(saves[-1]))
    except Exception as e:
        static = {'error': repr(e)}
    out = {'user': user, 'static': static, 'turns': sorted(tables), 'tables': {str(t): {k: ({str(p): v for p, v in d.items()} if isinstance(d, dict) else d) for k, d in tb.items()} for t, tb in tables.items()}, 'events': events}
    op = os.path.join(fork_dir, f'{user}_sgtables.json.gz')
    with gzip.open(op, 'wt') as f: json.dump(out, f)
    return fork_dir, f'{len(saves)} saves, {len(events)} events -> {os.path.basename(op)}'

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('dirs', nargs='+'); ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    with Pool(a.workers) as pool:
        for d, msg in pool.imap_unordered(process, a.dirs): print(d, msg, flush=True)
