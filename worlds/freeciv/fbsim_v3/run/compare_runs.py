#!/usr/bin/env python3
"""compare_runs.py RUN1_WIDE RUN2_WIDE [RUN2_VALIDATION_JSON] — run 1 (one question per prompt) against run 2 (batched), per set.

Prints, for the four headline sets and the mirrors: the mean score over the 24 models, its range, the Spearman rho of the
score with ECI (sign-adjusted so positive = more capable models score better) with a percentile bootstrap over models, the
rank correlation between the two runs' per-model scores, and the parsed share.  Then the natural-conditional summary.
"""
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

r1 = pd.read_csv(sys.argv[1]).set_index("model")
r2 = pd.read_csv(sys.argv[2]).set_index("model")
models = sorted(set(r1.index) & set(r2.index))
eci = r2.loc[models, "eci"].astype(float)
rng = np.random.default_rng(2026)


def rho_ci(x, y, n_boot=5000):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rho = spearmanr(x, y)[0]
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    b = np.array([spearmanr(x[i], y[i])[0] for i in idx])
    b = b[~np.isnan(b)]
    return rho, np.percentile(b, 2.5), np.percentile(b, 97.5)


SETS = [("Bank excess Brier", "bank_all_excess_brier", "bank_all"), ("Tail excess bits", "tails_all_excess_bits", "tails_all"),
        ("Mirror excess Brier", "mirrors_all_excess_brier", "mirrors_all"), ("Continuous nCRPS", "continuous_all_ncrps_global", "continuous_all"),
        ("Continuous excess nCRPS", "continuous_all_excess_ncrps_global", "continuous_all"), ("Natcond turn-2 excess Brier", "natcond_all_excess_t2", "natcond_all"),
        ("Natcond gain", "natcond_all_gain", "natcond_all")]
print(f"{'score':30s} | {'run':5s} {'mean':>7s} {'min':>7s} {'max':>7s} | {'rho vs ECI':>10s} {'95% CI (models)':>17s} | {'parsed':>7s}")
print("-" * 105)
for label, col, pre in SETS:
    for name, r in (("run 1", r1), ("run 2", r2)):
        v = r.loc[models, col].astype(float)
        sign = 1 if "gain" in col else -1
        rho, lo, hi = rho_ci(eci, sign * v)
        parsed = r.loc[models, f"{pre}_n_valid"].sum() / r.loc[models, f"{pre}_n_items"].sum()
        print(f"{label if name == 'run 1' else '':30s} | {name:5s} {v.mean():7.3f} {v.min():7.3f} {v.max():7.3f} | {rho:10.2f} {f'[{lo:.2f}, {hi:.2f}]':>17s} | {100 * parsed:6.1f}%")
    rr = spearmanr(r1.loc[models, col].astype(float), r2.loc[models, col].astype(float))[0]
    print(f"{'':30s} | rank correlation of per-model scores between runs: {rr:.2f}")
print()
print("Best model, run 1 -> run 2:")
for label, col, _ in SETS[:6]:
    f = min if "gain" not in col else max
    b1 = f(models, key=lambda m: r1.loc[m, col]); b2 = f(models, key=lambda m: r2.loc[m, col])
    print(f"  {label:30s} {b1.split('/')[1]:28s} -> {b2.split('/')[1]}")
c1, c2 = r1.loc[models, "total_cost_usd"].sum(), r2.loc[models, "total_cost_usd"].sum()
print(f"\ncost of the scored rows: run 1 ${c1:.2f} ({int(r1.loc[models, 'total_calls'].sum()):,} calls), run 2 ${c2:.2f} ({int(r2.loc[models, 'total_calls'].sum()):,} calls incl. reused natcond calls)")
