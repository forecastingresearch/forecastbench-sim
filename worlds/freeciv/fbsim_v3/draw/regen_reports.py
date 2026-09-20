#!/usr/bin/env python3
"""regen_reports.py SEED [SEED ...] — regenerate turn-60 world reports from the v3 anchors (savegame truth).
Reads state/anchors_raw/seed<N>/seed<N>_data.json.gz, stamps metadata.map_size from the T60 savegame settings,
renders with the patched freeciv_world renderer (no territory maps), writes
/Users/jaeholee0404/civbench/tmp/fbsim_v3_corpus/reports/seed<N>/turn_060_report.txt (+ the data json used)."""
import gzip, json, lzma, re, shutil, sys, glob
from pathlib import Path
from freeciv_world.world_reports.txt_report import generate_txt_report
BASE = Path('/Users/jaeholee0404/civbench/tmp/fbsim_v3_run/state')
OUT = Path('/Users/jaeholee0404/civbench/tmp/fbsim_v3_corpus/reports')
for seed in sys.argv[1:]:
    src = BASE / f'anchors_raw/seed{seed}'
    d = json.load(gzip.open(src / f'seed{seed}_data.json.gz', 'rt'))
    sav = sorted(glob.glob(str(src / f'savegames/seed{seed}_T60_*.sav.xz')))[0]
    txt = lzma.open(sav, 'rt', errors='replace').read()
    xs = re.search(r'^"xsize",(\d+)', txt, re.M); ys = re.search(r'^"ysize",(\d+)', txt, re.M)
    if xs and ys: d.setdefault('metadata', {})['map_size'] = [int(xs.group(1)), int(ys.group(1))]
    # ---- borders from the tile-ownership grid (exact): do two civs' territories touch, and if not, how far apart
    rows = [m.group(1).split(',') for m in re.finditer(r'^owner\d+="([^"]*)"', txt, re.M)]
    wrapx = bool(re.search(r'^"wrap","WRAPX"', txt, re.M)); H = len(rows); Wd = len(rows[0]) if rows else 0
    owned = {}
    for y, row in enumerate(rows):
        for x, cell in enumerate(row):
            if cell.strip() not in ('', '-'):
                try: owned.setdefault(int(cell), []).append((x, y))
                except ValueError: pass
    import numpy as _np
    def gap(a, b):
        A = _np.array(owned.get(a, [])); B = _np.array(owned.get(b, []))
        if len(A) == 0 or len(B) == 0: return None
        dx = _np.abs(A[:, None, 0] - B[None, :, 0]); dy = _np.abs(A[:, None, 1] - B[None, :, 1])
        if wrapx: dx = _np.minimum(dx, Wd - dx)
        return int(_np.maximum(dx, dy).min())      # Chebyshev distance: 1 = adjacent tiles (touching territories)
    _names = {int(c): (v.get('name') if isinstance(v, dict) else str(v)) for c, v in (d.get('civilizations') or {}).items()}
    nm_ = lambda p_: _names.get(int(p_), f'Player {p_}')
    main = [c for c in sorted(_names) if c < 5]
    border_rows = []
    for i_, a in enumerate(main):
        for b in main[i_ + 1:]:
            g = gap(a, b)
            if g is None: continue
            border_rows.append(f"{nm_(a):<14}| {nm_(b):<14}| {'yes' if g <= 1 else 'no':<4}| {('0' if g <= 1 else str(g - 1)):>3}")
    barb = re.search(r'^"barbarians","([^"]*)"', txt, re.M)
    d.setdefault('metadata', {})['borders_section'] = ("BORDERS (TURN 60)\nWhether each pair's territories are adjacent (share a border) at turn 60, and otherwise the gap in tiles between their nearest owned tiles.\n"
        + f"{'Civ A':<14}| {'Civ B':<14}| {'Touch':<4}| Gap\n" + "-" * 14 + "+" + "-" * 15 + "+" + "-" * 5 + "+----\n" + "\n".join(border_rows))
    d['metadata']['barbarians_setting'] = barb.group(1) if barb else None
    # exact events for turns <= 60 from the savegame tables (the serializer ledger is the observer's view)
    sgp = src / f'seed{seed}_sgtables.json.gz'
    if sgp.exists():
        sg = json.load(gzip.open(sgp, 'rt')); tv = (sg.get('static') or {}).get('technology_vector') or []
        rs = json.load(open(BASE / 'ruleset.json')); wonders = {v['name'] for v in rs['improvements'].values() if v.get('genus') == 0}
        names = {int(c): (v.get('name') if isinstance(v, dict) else str(v)) for c, v in (d.get('civilizations') or {}).items()}
        nm = lambda p: names.get(int(p), f'Player {p}')
        cname = {}
        for t, tb in sg['tables'].items():
            for pl, cl in tb['cities'].items():
                for c in cl: cname[c['id']] = c['name']
        ex = []
        for e in sg['events']:
            t, ty, pl = e.get('turn'), e.get('type'), e.get('player_id')
            if t is None or t > 60: continue
            if ty == 'tech_discovered':
                tn = tv[int(e['tech_id'])] if str(e['tech_id']).isdigit() and int(e['tech_id']) < len(tv) else str(e['tech_id'])
                ex.append(dict(turn=t, type=ty, player_id=int(pl), description=f'{nm(pl)} discovered {tn}', metadata={'tech_name': tn}))
            elif ty == 'city_founded':
                ex.append(dict(turn=t, type=ty, player_id=int(pl), description=f'{nm(pl)} founded {e.get("city_name","")}', metadata={'city_name': e.get('city_name','')}))
            elif ty == 'city_conquered':
                ex.append(dict(turn=t, type=ty, player_id=int(e['new_owner']), description=f'{nm(e["new_owner"])} captured {e.get("city_name","")} from {nm(e["prev_owner"])}', metadata={'city_name': e.get('city_name',''), 'prev_owner': int(e['prev_owner']), 'new_owner': int(e['new_owner'])}))
            elif ty == 'city_destroyed':
                ex.append(dict(turn=t, type=ty, player_id=int(pl), description=f'{e.get("city_name","")} ({nm(pl)}) was destroyed', metadata={'city_name': e.get('city_name','')}))
            elif ty == 'government_change':
                ex.append(dict(turn=t, type=ty, player_id=int(pl), description=f'{nm(pl)} changed government from {e.get("from")} to {e.get("to")}', metadata={'from': e.get('from'), 'to': e.get('to')}))
            elif ty == 'improvement_built' and e.get('name') in wonders:
                ex.append(dict(turn=t, type='wonder_completed', player_id=int(pl), description=f'{nm(pl)} completed {e["name"]} in {cname.get(e.get("city_id"), "?")}', metadata={'wonder_name': e['name'], 'city_name': cname.get(e.get('city_id'), '')}))
        # wonders_count series: great wonders only (the serializer counted the Palace for every civ)
        wc = {}
        for t, tb in sg['tables'].items():
            imps = tb.get('imps') or {}
            wc[str(t)] = {pl: sum(1 for cid, ws in (imps.get(pl) or {}).items() for w_ in ws if w_ in wonders) for pl in (tb.get('alive') or {})}
        d.setdefault('time_series', {})['wonders_count'] = {t: wc.get(t, v) for t, v in d['time_series'].get('wonders_count', {}).items()}
        # diplomacy changes (exact, one per unordered pair per turn) so the EVENT TYPES list is true
        seen = set()
        for e in sg['events']:
            if e.get('type') != 'diplomacy_change' or e.get('turn', 99) > 60: continue
            a, b = int(e['player_id']), int(e['other'])
            if a >= 5 or b >= 5: continue
            key = (e['turn'], min(a, b), max(a, b))
            if key in seen: continue
            seen.add(key)
            ex.append(dict(turn=e['turn'], type='diplomatic_change', player_id=min(a, b), description=f'{nm(min(a,b))} and {nm(max(a,b))}: {e.get("from")} -> {e.get("to")}', metadata={'from': e.get('from'), 'to': e.get('to')}))
        # real technologies only: the savegame set carries the A_NONE placeholder for every civ
        for t, row in d['time_series'].get('techs_known', {}).items():
            for pl, v in row.items():
                if v is not None: row[pl] = max(0, int(v) - 1)
        led = [e for e in d.get('events', []) if e.get('turn', 0) <= 60]
        print(f"seed{seed}: events ledger={len(led)} exact={len(ex)}  (tech {sum(1 for e in led if e['type']=='tech_discovered')}->{sum(1 for e in ex if e['type']=='tech_discovered')}, founded {sum(1 for e in led if e['type']=='city_founded')}->{sum(1 for e in ex if e['type']=='city_founded')}, gov {sum(1 for e in led if e['type']=='government_change')}->{sum(1 for e in ex if e['type']=='government_change')}, wonders {sum(1 for e in led if e['type']=='wonder_completed')}->{sum(1 for e in ex if e['type']=='wonder_completed')})")
        d['events'] = sorted(ex, key=lambda e: (e['turn'], e['type']))
        d.setdefault('metadata', {})['events_source'] = 'savegame_tables_v4'
    out = OUT / f'seed{seed}'; out.mkdir(parents=True, exist_ok=True)
    js = out / f'seed{seed}_data.json'; json.dump(d, open(js, 'w'))
    rs = BASE / 'ruleset.json'
    if rs.exists() and not (src / 'ruleset.json').exists(): shutil.copy(rs, src / 'ruleset.json')
    p, _ = generate_txt_report(js, src, out, 60, 5, 0)
    body = open(p).read()
    # cuts: RANKINGS section; Adjective + Nation ID columns; the Metadata column of EVENTS
    body = re.sub(r'\nRANKINGS \(derived from scores\)\n(?:.*\n)*?\n', '\n', body)
    # insert BORDERS after the DIPLOMACY table; add the barbarian setting to the header
    bs = d['metadata'].get('borders_section')
    if bs: body = re.sub(r'(\nDIPLOMACY \(TURN 60\)\n(?:.*\n)*?)\n', lambda m: m.group(1) + '\n' + bs + '\n\n', body, count=1)
    # barbarians run at FreeCiv's default settings (the savegame lists only changed settings, and 'barbarians' is not
    # among them): raids by barbarians and pirates begin around turn 60; none exist at the snapshot.
    body = body.replace('\nMap size:', "\nBarbarians: default settings (barbarian and pirate raids begin around turn 60; none exist at turn 60)\nMap size:", 1)
    lines = body.split('\n'); out_lines = []; mode = None
    for ln in lines:
        if ln.startswith('CIVILIZATIONS'): mode = 'civ'
        elif ln.startswith('EVENTS (chronological)'): mode = 'ev'
        elif ln.strip() == '' : mode = None
        if mode == 'civ' and ln and set(ln) <= set('-+'): ln = '---+---------------'
        elif mode == 'civ' and '|' in ln:
            cells = [c.strip() for c in ln.split('|')]; ln = f"{cells[0]:<3}| {cells[1]}"
        if mode == 'ev' and ln.count('|') == 4:
            ln = ln.rsplit('|', 1)[0].rstrip()
        out_lines.append(ln)
    body = '\n'.join(out_lines); open(p, 'w').write(body)
    print(f"seed{seed}: {p.name} {len(body.splitlines())} lines, '?' cells: {body.count(' ? ')}, map {d['metadata'].get('map_size')}")
