#!/usr/bin/env python3
"""describe_draw.py DRAW_DIR — composition tables for every set + QC_SAMPLES.md (2-3 random items per cell)."""
import json, sys, random, collections, os
import pandas as pd, numpy as np
D = sys.argv[1]; rng = random.Random(11); pd.set_option('display.width', 250); pd.set_option('display.max_rows', 300)
bank = pd.DataFrame(json.load(open(f'{D}/bank_750.json'))); tails = pd.DataFrame(json.load(open(f'{D}/tails_300.json')))
mirrors = pd.DataFrame(json.load(open(f'{D}/mirrors_50.json'))); cont = pd.DataFrame(json.load(open(f'{D}/continuous_300.json')))
nc = pd.DataFrame(json.load(open(f'{D}/natcond_600.json')))
BANDS = [(0.05, 0.23), (0.23, 0.41), (0.41, 0.59), (0.59, 0.77), (0.77, 0.95)]
bank['band'] = bank.qAll.apply(lambda q: next(f"{lo:.2f}-{hi:.2f}" for lo, hi in BANDS if lo < q <= hi))
out = []
def P(*a): s = ' '.join(str(x) for x in a); print(s); out.append(s)
P("# Draw v1 — composition\n"); P(f"bank {len(bank)}  tails {len(tails)}  mirrors {len(mirrors)}  continuous {len(cont)}  natcond {len(nc)}\n")
P("## Binary bank 750: family x horizon"); P(pd.crosstab(bank.family, bank['T'], margins=True).to_string())
P("\n## Binary bank: band x horizon"); P(pd.crosstab(bank.band, bank['T'], margins=True).to_string())
P("\n## Binary bank: world x horizon"); P(pd.crosstab(bank.world, bank['T'], margins=True).to_string())
P("\n## Tails 300 (q<=.05): family x horizon"); P(pd.crosstab(tails.family, tails['T'], margins=True).to_string())
P("tail q quantiles:", tails.qAll.quantile([.1, .5, .9]).round(4).to_dict(), " q==0 excluded by construction; per world:", tails.world.value_counts().to_dict())
P("\n## Mirrors 50 (q>=.95): family x horizon"); P(pd.crosstab(mirrors.family, mirrors['T'], margins=True).to_string())
P("\n## Continuous 300: family x horizon"); P(pd.crosstab(cont.family, cont['T'], margins=True).to_string())
P("median IQR by family:", cont.groupby('family').iqrAll.median().round(1).to_dict())
nc['abs'] = nc.delta.abs(); nc['stratum'] = pd.cut(nc['abs'], [0, .03, .08, .15, 1.01], labels=['null<.03', 'small', 'medium', 'large>=.15'], right=False)
P("\n## Natural conditionals 600: block x horizon"); P(pd.crosstab(nc.block, nc['T'], margins=True).to_string())
P("\n## Natcond: effect size (post hoc, all 1000 replays) by block"); P(pd.crosstab(nc.block, nc.stratum, margins=True).to_string())
P("\n## Natcond: effect size by horizon"); P(pd.crosstab(nc['T'], nc.stratum, margins=True).to_string())
P("\n## Natcond: question family x block"); P(pd.crosstab(nc.family, nc.block, margins=True).to_string())
P("\n## Natcond: reveal type x block"); P(pd.crosstab(nc.rev_kind, nc.block, margins=True).to_string())
P("sign among |d|>=.08: +", int(((nc['abs'] >= .08) & (nc.delta > 0)).sum()), " -", int(((nc['abs'] >= .08) & (nc.delta < 0)).sum()), "; controls: no-news", int(nc.control_no_news.sum()), "single-prompt", int(nc.control_single_prompt.sum()))
open(f'{D}/COMPOSITION.md', 'w').write('\n'.join(out))
# ---------------- QC samples ----------------
L = ["# QC samples — 2-3 random items per cell (draw v1)\n"]
def samp(df, k=3): return df.sample(min(k, len(df)), random_state=rng.randint(0, 10**6))
L.append("## Binary bank: per band x horizon\n")
for band in sorted(bank.band.unique()):
    for T in (90, 120, 150, 180, 210):
        cell = bank[(bank.band == band) & (bank['T'] == T)]; L.append(f"### band {band}, T{T} ({len(cell)})")
        for _, r in samp(cell).iterrows(): L.append(f"- [{r.world} {r.family}] {r.text}  → q={r.qAll:.3f}")
L.append("\n## Tails: per horizon\n")
for T in (90, 120, 150, 180, 210):
    cell = tails[tails['T'] == T]; L.append(f"### T{T} ({len(cell)})")
    for _, r in samp(cell, 3).iterrows(): L.append(f"- [{r.world} {r.family}] {r.text}  → q={r.qAll:.3f}")
L.append("\n## Mirrors: per horizon\n")
for T in (90, 120, 150, 180, 210):
    cell = mirrors[mirrors['T'] == T]; L.append(f"### T{T} ({len(cell)})")
    for _, r in samp(cell, 2).iterrows(): L.append(f"- [{r.world} {r.family}] {r.text}  → q={r.qAll:.3f}")
L.append("\n## Continuous: per family x horizon\n")
for fam in sorted(cont.family.unique()):
    for T in sorted(cont[cont.family == fam]['T'].unique()):
        cell = cont[(cont.family == fam) & (cont['T'] == T)]; L.append(f"### {fam}, T{T} ({len(cell)})")
        for _, r in samp(cell, 2).iterrows(): L.append(f"- [{r.world}] {r.text}  → median {r.medAll:g}, IQR {r.iqrAll:g}, p05–p95 {round(r.p05):g}–{round(r.p95):g}" + (f", never {r.censAll:.0%}" if r.censAll > 0 else ''))
L.append("\n## Natural conditionals: per block x horizon\n")
for b in ('A', 'B', 'C1', 'C2', 'D'):
    for T in (120, 150, 180, 210):
        cell = nc[(nc.block == b) & (nc['T'] == T)]; L.append(f"### block {b}, T{T} ({len(cell)})")
        for _, r in samp(cell, 3).iterrows(): L.append(f"- [{r.world} {r.family}] Q: {r.question}  | reveal: {r.reveal}  → p {r.p:.2f} → p|X {r.p_given:.2f} (Δ {r.delta:+.2f}, n_x={r.nx})")
open(f'{D}/QC_SAMPLES.md', 'w').write('\n'.join(L)); print(f"\nwrote {D}/COMPOSITION.md and QC_SAMPLES.md ({len(L)} lines)")
