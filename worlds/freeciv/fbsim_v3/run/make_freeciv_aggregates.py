#!/usr/bin/env python3
"""make_freeciv_aggregates.py SCORE_ITEMS OUT_DIR — the two aggregate files the paper's figures and difficulty table read.

  OUT_DIR/family_horizon_scores.csv  model x set (bank, tails, mirrors, continuous) x family x horizon: n, bias (mean p-q),
                                     bits (mean excess bits), excess (mean excess Brier, or mean excess nCRPS for continuous),
                                     ncrps (mean nCRPS), cov90, mederr, p, q; four decimals
  OUT_DIR/reliability_bands.csv      per model, the bank forecasts in ten equal-width bands of q on [0.05, 0.95] (right-closed):
                                     band, n, mean q, mean p

Reproduces the run-1 files of 2026-09-09 exactly (imputed binary rows included, as the scorer left them) from scores_v1/score_items.csv(.gz); run 2 uses the same code.
"""
import sys, os
import numpy as np, pandas as pd

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
si = pd.read_csv(src, low_memory=False)
# imputed binary rows (p = 0.5 for an unparsed answer) stay in, as in run 1; unparsed continuous rows carry NaN metrics and drop out of the means
rows = []
for (m, s, fam, T), g in si[si["set"].isin(["bank", "tails", "mirrors", "continuous"])].groupby(["model", "set", "family", "T"]):
    r = dict(model=m, set=s, family=fam, horizon=int(T), n=len(g))
    if s == "continuous":
        r.update(excess=g["excess_ncrps_global"].mean(), ncrps=g["ncrps_global"].mean(), cov90=g["cov90"].mean(), mederr=g["median_err"].mean())
    else:
        r.update(bias=(g["p"] - g["q"]).mean(), bits=g["excess_bits"].mean(), excess=g["excess_brier"].mean(), p=g["p"].mean(), q=g["q"].mean())
    rows.append(r)
fh = pd.DataFrame(rows)[["model", "set", "family", "horizon", "bias", "bits", "cov90", "excess", "mederr", "n", "ncrps", "p", "q"]]
fh = fh.sort_values(["model", "set", "horizon", "family"]).round(4)
fh.to_csv(os.path.join(out, "family_horizon_scores.csv"), index=False)
bank = si[si["set"] == "bank"].copy()
edges = np.round(np.linspace(0.05, 0.95, 11), 6)                                     # rounded, so q values on an edge bin as the figure script bins them
bank["band"] = pd.cut(bank["q"], edges, labels=False, right=True, include_lowest=True).astype(int)   # right-closed bins of width 0.09
rb = bank.groupby(["model", "band"]).agg(n=("q", "size"), q=("q", "mean"), p=("p", "mean")).reset_index().round(4)
rb.to_csv(os.path.join(out, "reliability_bands.csv"), index=False)
print(f"wrote {out}/family_horizon_scores.csv ({len(fh)} rows) and reliability_bands.csv ({len(rb)} rows)")
