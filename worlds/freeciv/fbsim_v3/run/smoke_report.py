#!/usr/bin/env python3
"""smoke_report.py RESULTS_DIR — per-model health table over every results.jsonl under RESULTS_DIR (last row per key wins):
rows, errors, parse strict / lenient / none, providers seen, mean output and reasoning tokens by arm, cost, and the
first error text per model. Also projects the full-run cost from the observed per-call cost."""
import sys, glob, json, collections, os, statistics as st
d = sys.argv[1]; R = {}
for f in sorted(glob.glob(f'{d}/*/results.jsonl')):
    for line in open(f):
        try: r = json.loads(line)
        except Exception: continue
        R[(r['model'], r['item'], r['arm'])] = r
by = collections.defaultdict(list)
for (m, _, _), r in R.items(): by[m].append(r)
FULL = {'t1': 1524, 't2': 400, 'nonews': 99, 'single': 0}   # v1.8 arms
print(f"{'model':38s} {'rows':>4s} {'err':>3s} {'strict':>6s} {'len':>4s} {'none':>4s} {'out/t1':>7s} {'reas/t1':>7s} {'out/t2':>7s} {'$smoke':>7s} {'$full':>7s}  providers")
tot_full = 0; problems = []
for m in sorted(by, key=lambda m: sum(r.get('cost') or 0 for r in by[m])):
    rs = by[m]; err = [r for r in rs if r.get('error')]
    strict = sum(1 for r in rs if r.get('parse_mode') == 'strict' or (r.get('value') is not None and 'parse_mode' not in r)); lenient = sum(1 for r in rs if r.get('parse_mode') == 'lenient')   # rows from before parse_mode existed were strict parses
    none = sum(1 for r in rs if r.get('value') is None and not r.get('error'))
    prov = collections.Counter(r.get('provider') for r in rs if r.get('provider'))
    def mean(arm, key):
        v = [r[key] for r in rs if r['arm'] == arm and r.get(key) is not None]
        return f"{st.mean(v):7.0f}" if v else f"{'-':>7s}"
    cost = sum(r.get('cost') or 0 for r in rs)
    per_arm = {a: st.mean([r['cost'] for r in rs if r['arm'] == a and r.get('cost')]) for a in FULL if any(r['arm'] == a and r.get('cost') for r in rs)}
    full = sum(FULL[a] * per_arm.get(a, per_arm.get('t1', 0)) for a in FULL); tot_full += full
    print(f"{m:38s} {len(rs):4d} {len(err):3d} {strict:6d} {lenient:4d} {none:4d} {mean('t1','tokens_out')} {mean('t1','tokens_reasoning')} {mean('t2','tokens_out')} {cost:7.2f} {full:7.0f}  {dict(prov)}")
    if err: problems.append((m, len(err), err[0]['error'][:160]))
    for r in rs:
        if r.get('value') is None and not r.get('error'): problems.append((m, 'unparsed', r['item'], r['arm'], repr((r.get('text') or '')[-160:])))
print(f"\nprojected full run (2,023 calls/model = t1 1524 + t2 400 + nonews 99, observed per-call cost incl. caching): ${tot_full:,.0f} over {len(by)} models")
print('\nPROBLEMS'); [print(' ', p) for p in problems]
