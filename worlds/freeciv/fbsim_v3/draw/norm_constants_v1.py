#!/usr/bin/env python3
"""norm_constants_v1.py DRAW_DIR [--run NAME] — write DRAW_DIR/continuous_norm_constants.json for score_v2.py: one constant
per continuous family (per metric inside NC1_value_at_T), the median over the family's items of the truth p05-p95 range,
rounded to one significant figure (the paper's rule, draw v1.8). Timing families (P7-P9) get none, as in the paper.
--run NAME or FBSIM_RUN: DRAW_DIR defaults to the run's sets/draw_v1/."""
import json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); import runcfg
RUN = runcfg.pop_run()
D, = runcfg.positional(RUN, ['sets/draw_v1'], usage='norm_constants_v1.py DRAW_DIR')
TIMING = ('P7_', 'P8_', 'P9_')
RULE = ("Fabio-style nCRPS_global: CRPS / one fixed constant per metric (Micropolis: population 20000, traffic 20, "
        "crime/pollution/land value 60, inferred from his CSV). FreeCiv analog: one constant per family (per metric inside "
        "NC1_value_at_T) = the median over the family's items of the truth p05-p95 range, rounded to one significant figure. "
        "Fixed at draw v1.8, 2026-09-08.")
ranges = {}
for i in json.load(open(os.path.join(D, 'continuous_300.json'))):
    if i['family'].startswith(TIMING): continue
    key = i['family'] + ('/' + i['metric'] if i['family'] == 'NC1_value_at_T' else '')
    ranges.setdefault(key, []).append(i['p95'] - i['p05'])
keys = sorted(k for k in ranges if '/' not in k) + sorted(k for k in ranges if '/' in k)
C = {}
for k in keys:
    med = float(np.median(ranges[k]))
    C[k] = {'n': len(ranges[k]), 'median_p05_p95_range': med, 'constant': float(f'{med:.1g}') if med > 0 else None}
json.dump({'rule': RULE, 'constants': C}, open(os.path.join(D, 'continuous_norm_constants.json'), 'w'), indent=1)
for k, v in C.items(): print(f"{k:<32} n={v['n']:>3} median range {v['median_p05_p95_range']:<8g} constant {v['constant']}")
