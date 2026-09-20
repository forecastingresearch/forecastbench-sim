#!/usr/bin/env python3
"""anchor_qc.py seedN_data.json.gz [...] — per-civ turn-60 table + health flags for candidate anchors."""
import gzip, json, sys, os, collections
def row(ts, m, t, pid, d=None):
    return (ts.get(m) or {}).get(str(t), {}).get(pid, d)
for path in sys.argv[1:]:
    d = json.load(gzip.open(path, 'rt')); ts = d['time_series']; md = d.get('metadata', {})
    turns = sorted(int(t) for t in ts['scores']); T = 60 if 60 in turns else max(turns)
    civs = sorted((d.get('civilizations') or {}).items(), key=lambda kv: int(kv[0]))
    ev = d.get('events') or []
    techs_by = collections.Counter(str(e['player_id']) for e in ev if e.get('type') == 'tech_discovered')
    found_by = collections.Counter(str(e['player_id']) for e in ev if e.get('type') == 'city_founded')
    conq = sum(1 for e in ev if e.get('type') == 'city_conquered'); wond = sum(1 for e in ev if e.get('type') == 'wonder_completed')
    gov_ch = sum(1 for e in ev if e.get('type') == 'government_change')
    truth = md.get('truth_source') or {}
    print(f"\n=== {os.path.basename(path)}  turns {turns[0]}-{turns[-1]} (n={len(turns)}) T={T}  coverage={md.get('savegame_coverage')}  truth={truth.get('source')} {truth.get('turns_overridden')}/{truth.get('turns_total')}  events: conq={conq} wonders={wond} govch={gov_ch}")
    print(f"{'pid':>3} {'name':<14} {'alive':>5} {'gov':<10} {'cit':>3} {'pop':>4} {'tech':>4} {'tev':>3} {'fnd':>3} {'score':>5} {'gold':>5} {'unit':>4} {'mil':>3} {'terr':>4} {'won':>3}")
    flags = []
    main = [c for c in civs if int(c[0]) < 5]
    for pid, info in main:
        name = info.get('name') if isinstance(info, dict) else str(info)
        alive = row(ts, 'is_alive', T, pid); gov = row(ts, 'government', T, pid, '')
        cit = row(ts, 'cities_count', T, pid); pop = row(ts, 'population', T, pid); tech = row(ts, 'techs_known', T, pid)
        sc = row(ts, 'scores', T, pid); gold = row(ts, 'treasury', T, pid); un = row(ts, 'units_count', T, pid)
        mil = row(ts, 'military_units_count', T, pid); terr = row(ts, 'territory_size', T, pid); won = row(ts, 'wonders_count', T, pid)
        print(f"{pid:>3} {name:<14} {str(alive):>5} {str(gov):<10} {cit!s:>3} {pop!s:>4} {tech!s:>4} {techs_by[pid]:>3} {found_by[pid]:>3} {sc!s:>5} {gold!s:>5} {un!s:>4} {mil!s:>3} {terr!s:>4} {won!s:>3}")
        if alive is not True: flags.append(f"{name} not alive")
        if not cit: flags.append(f"{name} 0 cities")
        if techs_by[pid] == 0: flags.append(f"{name} FROZEN (no tech events)")
        if sc is None or sc < 0: flags.append(f"{name} score {sc}")
        if not gov: flags.append(f"{name} empty government")
    rel = (d.get('diplomacy') or {}).get('relations') or {}
    states = collections.Counter()
    for k, tt in rel.items():
        a, b = k.split('_'); 
        if int(a) >= 5 or int(b) >= 5: continue
        st = (tt.get(str(T)) or {}).get('state'); states[st] += 1
    print(f"diplomacy@T{T}: {dict(states)}   contacts={sum(v for k,v in states.items() if k not in ('Never met','No Contact'))}/10")
    print("FLAGS: " + ("; ".join(flags) if flags else "none"))
