#!/usr/bin/env python3
"""corpus_qc.py <raw_root> — per-anchor inventory and integrity checks on the pulled corpus (data files only)."""
import collections, glob, gzip, json, os, re, sys
root = sys.argv[1]
by = collections.defaultdict(dict); dups = collections.Counter(); trunc = collections.Counter(); sgt = collections.Counter()
bad = []; stamps = collections.Counter(); neg = collections.Counter(); turns = collections.Counter(); hz = collections.Counter()
for fp in sorted(glob.glob(f'{root}/**/*_data.json.gz', recursive=True)):
    m = re.match(r'seed(\d+)forkrng(\d+)_data', os.path.basename(fp))
    if not m: continue
    a, r = m.group(1), m.group(2)
    if r in by[a]: dups[a] += 1; continue
    by[a][r] = fp
    d = os.path.dirname(fp)
    if os.path.exists(os.path.join(d, 'TRUNCATED')): trunc[a] += 1
    if glob.glob(os.path.join(d, '*_sgtables.json.gz')): sgt[a] += 1
    try:
        j = json.load(gzip.open(fp, 'rt'))
    except Exception as e:
        bad.append((fp, repr(e))); continue
    md = j.get('metadata', {}); ts = j.get('time_series', {})
    ok = (md.get('truth_source') or {}).get('source') == 'savegame' and md.get('savegame_coverage') == 1.0 and 'is_alive' in ts and 'government' in ts
    stamps[(a, ok)] += 1
    if any(v == -1 for row in ts.get('scores', {}).values() for v in row.values()): neg[a] += 1
    tt = sorted(int(t) for t in ts.get('scores', {})); turns[(a, tt[0], tt[-1])] += 1
    saves = glob.glob(os.path.join(d, 'savegames', '*'))
    have = {int(mm.group(1)) for s in saves for mm in [re.search(r'_T(\d+)_', s)] if mm}
    hz[(a, tuple(sorted(t for t in (60, 61, 90, 120, 150, 180, 210, 211) if t in have)))] += 1
print(f"{'anchor':>6} {'forks':>5} {'dups':>4} {'trunc':>5} {'sgtables':>8} {'stamp_ok':>8} {'neg_scores':>10}  turn ranges")
for a in sorted(by):
    rng = {k[1:]: v for k, v in turns.items() if k[0] == a}
    print(f"{a:>6} {len(by[a]):>5} {dups[a]:>4} {trunc[a]:>5} {sgt[a]:>8} {stamps[(a, True)]:>8} {neg[a]:>10}  {rng}")
print('horizon-save sets:', {k: v for k, v in hz.items() if k[1] != (60, 61, 90, 120, 150, 180, 210, 211)} or 'all forks have T60/61/90/120/150/180/210/211 saves')
print('unreadable:', bad[:5], len(bad))
