#!/usr/bin/env python3
"""reparse.py RESULTS_DIR — re-run the current parsers over rows whose value is null but whose text is non-empty; append a
corrected copy of each row that now parses (last row per key wins in the scorer and the harness). No API calls."""
import sys, glob, json, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from elicit_v1 import parse_prob, parse_pct
fixed = total = 0
for f in glob.glob(f'{sys.argv[1]}/*/results.jsonl'):
    rows = [json.loads(l) for l in open(f)]; last = {}
    for r in rows: last[(r['item'], r['arm'])] = r
    add = []
    for r in last.values():
        if r.get('value') is None and (r.get('text') or '').strip():
            total += 1
            v, mode = parse_pct(r['text'], True) if r['kind'] == 'cont' else parse_prob(r['text'], True)
            if v is not None: r = dict(r, value=v, parse_mode=mode, reparsed=True); add.append(r); fixed += 1
    if add:
        with open(f, 'a') as fh:
            for r in add: fh.write(json.dumps(r) + '\n')
print(f'rows with text but no value: {total}; now parse: {fixed}')
