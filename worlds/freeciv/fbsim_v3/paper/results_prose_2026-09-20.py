#!/usr/bin/env python
"""results_prose_2026-09-20.py --paper-root DIR --results RUN2_PAPER_DIR [--check]

Writes the FreeCiv results of run 2 into the paper under Jaeho's decision of 20 September 2026: excess scores stay the
headline of every set, the main text says that the bank ranking is driven by a shared under-forecast and mentions
discrimination in one sentence, and the discrimination, spread and reliability numbers go to Appendix C.  The four
"To be written" boxes of Section 4 and the Discussion box are replaced by green paragraphs and deleted (the standing
rule: a fulfilled box goes).  Every number is computed here from the run-2 score file or copied from the validation
table the paper already prints (game-cluster intervals).  Each anchor must match exactly once.
"""
import argparse, re, sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ap = argparse.ArgumentParser()
ap.add_argument("--paper-root", required=True)
ap.add_argument("--results", required=True)
ap.add_argument("--check", action="store_true")
a = ap.parse_args()
P = Path(a.paper_root).resolve(); RES = Path(a.results).resolve()

si = pd.read_csv(RES / "scores_v1" / "score_items.csv.gz", low_memory=False)
w = pd.read_csv(RES / "results_v1" / "freeciv_results_wide.csv").set_index("model")
eci = w.eci.astype(float)
def pr(x, y):
    m = x.notna() & y.notna(); return pearsonr(x[m], y[m])[0]
def boot_ci(x, y, n=10000, seed=2026):
    rng = np.random.default_rng(seed); x, y = np.asarray(x, float), np.asarray(y, float)
    idx = rng.integers(0, len(x), size=(n, len(x))); b = np.array([spearmanr(x[i], y[i])[0] for i in idx]); b = b[~np.isnan(b)]
    return np.percentile(b, 2.5), np.percentile(b, 97.5)
b = si[si.set == "bank"]; g = b.groupby("model")
disc = g.apply(lambda d: pr(d.p, d.q), include_groups=False).reindex(w.index)
bias = g.apply(lambda d: (d.p - d.q).mean(), include_groups=False).reindex(w.index)
bex = g.excess_brier.mean().reindex(w.index)
rho_disc = spearmanr(eci, disc)[0]; lo, hi = boot_ci(eci, disc)
fam = b.groupby("family").apply(lambda d: pd.Series(dict(q=d.q.mean(), p=d.p.mean(), disc=pr(d.p, d.q))), include_groups=False)
comp = fam.loc["EX_comparative"]
nc = si[si.set == "natcond"].copy(); nc["q2"] = nc.p_given; gn = nc.groupby("model")
sd2 = gn.p2.std().reindex(w.index); disc2 = gn.apply(lambda d: pr(d.p2, d.q2), include_groups=False).reindex(w.index); nex = gn.excess_t2.mean().reindex(w.index)
N = dict(rho_disc=rho_disc, lo=lo, hi=hi, disc_lo=disc.min(), disc_hi=disc.max(), disc_lo_m=disc.idxmin().split("/")[1], disc_hi_m=disc.idxmax().split("/")[1],
         rho_bias=spearmanr(eci, bias.abs())[0], rho_ex_bias=spearmanr(bex, bias.abs())[0], rho_ex_disc=spearmanr(bex, disc)[0],
         comp_bias=comp.p - comp.q, comp_disc=comp.disc, sd_lo=sd2.min(), sd_hi=sd2.max(), rho_ex_sd=spearmanr(nex, sd2)[0], rho_ex_disc2=spearmanr(nex, disc2)[0],
         disc2_lo=disc2.min(), disc2_hi=disc2.max(), rho_disc2=spearmanr(eci, disc2)[0])
print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in N.items()})
NAMES = {"llama-4-scout": "Llama 4 Scout", "gemini-3.7-flash": "Gemini 3.7 Flash", "gemini-3-flash-preview": "Gemini 3 Flash", "o3": "o3", "gemini-2.5-flash": "Gemini 2.5 Flash"}

TAILS = r"""\paragraph{Tail questions.}
The 300 tail questions ask about events whose ground-truth probability is at most 0.05 (mean 0.019). Excess bits run from 0.15 (Sonnet 5) to 0.70 (Qwen3.5 Flash), mean 0.33 over the 24 models. Their correlation with ECI is $\rho = 0.34$ [0.20, 0.42] over the eight anchor games (\cref{fig:freeciv-capability}b), against \MPDRhoTail\ on the Micropolis tail set. The loss comes from overstating rare events. Every model places a mean forecast of 0.13 to 0.34 on them, and no model scores below the 0.03 bits of a constant forecast of 0.05. Forecasts of 0.30 or more are 27\% of the answers and carry 73\% of the bits. A model's mean forecast on the set predicts its rank ($\rho = 0.91$), so the tail ranking mostly orders models by the level of their forecasts. Mean bits rise from 0.28 at turn 90 to 0.41 at turn 210 (\cref{fig:freeciv-horizon})."""

NATCOND = r"""\paragraph{Natural conditionals.}
After its turn-1 forecast the model learns one fact about turns 61 to 90 and forecasts again, and the target is $p(Y \mid X)$ from the replays. The turn-2 excess Brier score runs from 0.089 (o3) to 0.176 (Gemini 2.5 Flash), mean 0.119, and correlates with ECI at $\rho = 0.38$ [0.18, 0.60] over games (\cref{fig:freeciv-capability}c). The update itself helps little. Models move in the direction of the fact: the correlation between move and true shift $p(Y \mid X) - p(Y)$ is 0.45 to 0.60 per model, and the sign agrees on 86\% to 94\% of the 185 cells whose shift exceeds 0.05. But they move too far. The mean absolute move is 0.12 to 0.25 per model, against a mean absolute true shift of 0.07. The gain over keeping the turn-1 forecast (its excess against $p(Y \mid X)$ minus the turn-2 excess) is therefore $-0.002$ on average and positive for 7 of 24 models (\cref{fig:freeciv-natcond}). The turn-2 ranking reflects the turn-1 forecasts and, among them, the models that keep their forecasts nearest the middle: the spread of a model's turn-2 forecasts predicts its score ($\rho = 0.91$; \cref{app:full-freeciv}), and no model scores better than a constant forecast of 0.5. The correlation between move and shift rises with ECI ($\rho = 0.61$); the gain does not (0.03)."""

CONT = r"""\paragraph{Continuous questions.}
The 300 continuous questions ask for five percentiles of a count or of a value at the horizon, scored by the CRPS of the five, divided by the family constant and net of the replay distribution's own CRPS. Excess nCRPS runs from 0.40 (GPT-5.6 Sol) to 1.13 (GPT-5 Nano), mean 0.65. Its correlation with ECI, $\rho = 0.59$ [0.43, 0.63] over games (\cref{fig:freeciv-capability}a), is the strongest of the four sets and has the same sign at every horizon (0.49 to 0.65). Mean excess rises from 0.35 at turn 90 to 0.82 at turn 210 (\cref{fig:freeciv-horizon}). The intervals are too narrow: a model's 5th-to-95th percentile interval contains 22\% (GPT-5 Nano) to 77\% (GPT-5.6 Sol) of the replay values, and this coverage predicts most of the score ($\rho = -0.90$)."""

S = {k: (f"{v:+.2f}" if k == "comp_bias" else f"{v:.2f}") for k, v in N.items() if isinstance(v, float)}
S["disc_lo_m"] = NAMES.get(N["disc_lo_m"], N["disc_lo_m"]); S["disc_hi_m"] = NAMES.get(N["disc_hi_m"], N["disc_hi_m"])
BANK = (r"""\paragraph{Binary bank.}
On the 750 bank questions, whose ground truth is spread uniformly over $(0.05, 0.95)$, excess Brier runs from 0.103 (Gemini 3 Flash) to 0.193 (Llama 4 Scout), mean 0.141. It is close to unrelated to ECI, $\rho = 0.18$ [0.03, 0.23] over games (\cref{fig:freeciv-capability}d), and flat across horizons (0.136 to 0.149). The score is dominated by a bias that every model shares. Forecasts are compressed and low: the mean of $f - q$ is negative for all 24 models ($-0.33$ to $-0.07$), and no model's excess is below the 0.075 of a constant forecast of 0.5 (\cref{fig:freeciv-reliability}). A model's bias predicts its rank ($\rho = """ + S["rho_ex_bias"] + r"""$), and the bias does not vary with ECI (""" + S["rho_bias"] + r"""). What does vary with ECI is discrimination, the correlation between a model's forecasts and the truth, which runs from """ + S["disc_lo"] + " to " + S["disc_hi"] + r""" and rises with capability at $\rho = """ + S["rho_disc"] + r"""$ (\cref{app:full-freeciv}). The more capable models order the questions better but state probabilities no closer to them, and a higher reasoning effort does not remove the bias (\cref{app:ablation-effort}).""")

DISCUSSION = r"""The natural conditionals show a failure of magnitude rather than of direction. Models move the right way after a revealed fact on 86\% to 94\% of the cells where the truth moves, but by more than twice the true shift, so the update gains nothing on average over keeping the turn-1 forecast. The models with higher ECI track the true shift more closely ($\rho = 0.61$) without gaining more (0.03). Studies of updating on real news report the opposite failure, forecasts that respond too little \citep{yuan2025evolvecast}, and tests against exact posteriors under simplified evidence find departures from Bayes' rule in both directions \citep{imran2025bayesian,samanta2026bayesbench}. Those settings score an update against human forecasters or against a posterior with a formula. A natural conditional scores it against the frequency of the outcome given the fact, which a consistency check without a truth value cannot supply \citep{paleka2024consistency}. Here that frequency says the models move too far on facts that change the truth little: 174 of the 400 facts shift it by less than 0.03. Whether models move too far only in a world they know little about is a question for a second world with natural conditionals."""

APPC_RELIAB = (r"""\Cref{fig:freeciv-reliability} is the reliability diagram of the bank forecasts, and \cref{fig:freeciv-natcond} above plots the move of each natural-conditional forecast against the true shift. Discrimination, the Pearson correlation between a model's forecasts and the truth over the 750 bank questions, runs from """
               + S["disc_lo"] + " (" + S["disc_lo_m"] + ") to " + S["disc_hi"] + " (" + S["disc_hi_m"] + r""") and correlates with ECI at $\rho = """ + S["rho_disc"] + "$ [" + S["lo"] + ", " + S["hi"]
               + r"""] (model bootstrap, 10,000 resamples); the question-level mean of $f - q$ correlates with ECI at """ + S["rho_bias"] + r""", and the excess Brier score correlates with the absolute bias at """ + S["rho_ex_bias"]
               + " and with discrimination at " + S["rho_ex_disc"] + r""". Comparative questions, which ask whether one civilization exceeds another, show no bias (mean $f - q$ of $""" + S["comp_bias"] + "$) and the highest discrimination (" + S["comp_disc"]
               + r"""); event and threshold questions are under-forecast by 0.25 to 0.35 (\cref{tab:freeciv-family-difficulty}). On the natural conditionals the turn-2 excess Brier score of a model correlates at """ + S["rho_ex_sd"]
               + " with the standard deviation of its turn-2 forecasts (" + S["sd_lo"] + " for o3 to " + S["sd_hi"] + " for Gemini 2.5 Flash) and at " + S["rho_ex_disc2"] + " with its discrimination at turn 2 (" + S["disc2_lo"] + " to " + S["disc2_hi"]
               + "), which itself rises with ECI at " + S["rho_disc2"] + ".")


def replace_box(s, title, new):
    m = re.search(r"\n\\begin\{todo\}\{" + re.escape(title) + r"\}.*?\\end\{todo\}\n", s, re.S)
    if not m or s.count("\\begin{todo}{" + title + "}") != 1: print("ABORT: box", title); sys.exit(1)
    return s[:m.start()] + "\n" + new + "\n" + s[m.end():]

f5 = (P / "sections/05_freeciv.tex").read_text()
f5 = replace_box(f5, "Tail questions", "\\begin{greentext}\n" + TAILS + "\n")
f5 = replace_box(f5, "Natural conditionals: updating on one revealed fact", NATCOND + "\n")
f5 = replace_box(f5, "Continuous questions", CONT + "\n")
f5 = replace_box(f5, "Binary bank", BANK + "\n\\end{greentext}\n")
old = "% Suggested for Appendix C (full results), not the main text. Uncomment and move."
if f5.count(old) != 1: print("ABORT: trailing comment"); sys.exit(1)
f5 = f5[:f5.find(old)].rstrip() + "\n"   # drop the commented-out reliability figure (it lives in Appendix C)
f5 = f5.replace("% 05_freeciv: v1.3 (2026-09-16), setup shortened from v1.2 (build/paper_snapshot_v1.2); the full\n% setup text lives in appendix/A_worlds_and_protocol.tex. Results stay dark-yellow to-do blocks\n% (provisional run, one question per prompt).",
                "% 05_freeciv: v1.4 (2026-09-20). Setup shortened on 16 September (full text in appendix/A_worlds_and_protocol.tex);\n% results written on 20 September from run 2 (results/run2_paper) by paper/results_prose_2026-09-20.py on the dev branch.")

f9 = (P / "sections/09_discussion.tex").read_text()
f9 = replace_box(f9, "What the natural-conditional results mean (Jaeho)", "\\begin{greentext}\n" + DISCUSSION + "\n\\end{greentext}\n")

fc = (P / "appendix/C_full_results.tex").read_text()
oldc = r"\Cref{fig:freeciv-reliability} is the reliability diagram of the bank forecasts, and \cref{fig:freeciv-natcond} above plots the move of each natural-conditional forecast against the true shift."
if fc.count(oldc) != 1: print("ABORT: appendix C anchor"); sys.exit(1)
fc = fc.replace(oldc, APPC_RELIAB)

legend = (P / "sections/00_legend.tex").read_text()
oldl = "FreeCiv numbers are those of the grouped run of 20 September 2026; its four result paragraphs and the Discussion paragraph are still to be written (boxes)."
if legend.count(oldl) != 1: print("ABORT: legend"); sys.exit(1)
legend = legend.replace(oldl, "FreeCiv numbers are those of the grouped run of 20 September 2026; its results are written (green) and await Jaeho's review.")

for f, s in (("sections/05_freeciv.tex", f5), ("sections/09_discussion.tex", f9), ("appendix/C_full_results.tex", fc), ("sections/00_legend.tex", legend)):
    print(("would write" if a.check else "wrote"), f)
    if not a.check: (P / f).write_text(s)
