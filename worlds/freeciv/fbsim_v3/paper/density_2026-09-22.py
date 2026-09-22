#!/usr/bin/env python
"""density_2026-09-22.py --paper-root DIR [--check]

Jaeho's decisions of 22 September 2026: the four FreeCiv result paragraphs state each finding with one supporting number
and leave the brackets, the per-model extremes and the secondary statistics to Figure 3, Table 3 and Appendix C (the
80/20 of the density pass); Related work loses its fourth paragraph into the third and its restated contrasts (options
A and B).  Each anchor must match exactly once.
"""
import argparse, re, sys
from pathlib import Path

ap = argparse.ArgumentParser(); ap.add_argument("--paper-root", required=True); ap.add_argument("--check", action="store_true")
a = ap.parse_args(); P = Path(a.paper_root).resolve()

TAILS = r"""\paragraph{Tail questions.}
The 300 tail questions ask about events whose ground-truth probability is at most 0.05 (mean 0.019). Excess bits average 0.33 over the 24 models and correlate with ECI at $\rho = 0.34$ (\cref{fig:freeciv-capability}b), against \MPDRhoTail\ on the Micropolis tail set. Models lose these bits by overstating rare events. Every model places a mean forecast of 0.13 to 0.34 on events that occur with probability 0.02, and none scores below a constant forecast of 0.05. A model's mean forecast on the set predicts its rank, so the tail ranking mostly orders models by the level of their forecasts. Mean bits rise with the horizon (\cref{app:full-freeciv})."""
NATCOND = r"""\paragraph{Natural conditionals.}
After its turn-1 forecast the model learns one fact about turns 61 to 90 and forecasts again, and the target is $p(Y \mid X)$ from the replays. The turn-2 excess Brier score averages 0.119 and correlates with ECI at $\rho = 0.38$ (\cref{fig:freeciv-capability}c). The update itself helps little. Models move in the direction of the fact on 86\% to 94\% of the cells where the truth moves, but they move too far, by 0.12 to 0.25 on average against a true shift of 0.07. The gain over keeping the turn-1 forecast is therefore about zero for every model (\cref{fig:freeciv-natcond}). The turn-2 ranking rewards the models that keep their forecasts nearest the middle, and no model scores better than a constant forecast of 0.5. Models with higher ECI track the true shift more closely without gaining more (\cref{app:full-freeciv})."""
CONT = r"""\paragraph{Continuous questions.}
The 300 continuous questions ask for five percentiles of a count or of a value at the horizon, scored by the CRPS of the five, divided by the family constant and net of the replay distribution's own CRPS. Excess nCRPS averages 0.65 and correlates with ECI at $\rho = 0.59$ (\cref{fig:freeciv-capability}a), the strongest gradient of the four sets and of the same sign at every horizon. Mean excess more than doubles from turn 90 to turn 210 (\cref{app:full-freeciv}). Models draw their intervals too narrow. A model's 5th-to-95th percentile interval contains 22\% to 77\% of the replay values, and this coverage predicts most of the score."""
BANK = r"""\paragraph{Mid-range questions.}
On the 750 mid-range questions, whose ground truth is spread uniformly over $(0.05, 0.95)$, excess Brier averages 0.141, barely tracks ECI ($\rho = 0.18$, \cref{fig:freeciv-capability}d), and is flat across horizons. A bias that every model shares dominates the score. Every model compresses its forecasts and sets them low. The mean of $f - q$ is negative for all 24 models, and none scores below a constant forecast of 0.5 (\cref{fig:freeciv-reliability}). What does vary with ECI is discrimination, the correlation between a model's forecasts and the truth, which rises with capability at $\rho = 0.72$ (\cref{app:full-freeciv}). The more capable models order the questions better but state probabilities no closer to them, and a higher reasoning effort does not remove the bias (\cref{app:ablation-effort})."""

f5 = (P / "sections/05_freeciv.tex").read_text()
for head, new in (("Tail questions", TAILS), ("Natural conditionals", NATCOND), ("Continuous questions", CONT), ("Mid-range questions", BANK)):
    m = re.search(r"\\paragraph\{" + re.escape(head) + r"\.\}\n.*?(?=\n\n|\n\\paragraph|\n\\end\{greentext\})", f5, re.S)
    if not m: print("ABORT: paragraph", head); sys.exit(1)
    f5 = f5[:m.start()] + new + f5[m.end():]

f8 = (P / "sections/08_related_work.tex").read_text()
R8 = [
    (r"The excess score drops that term. \citet{lerch2017forecasterdilemma} showed that evaluating only the cases in which an extreme event occurred rewards forecasters who overstate its probability. We select tail questions on $q$ before any outcome is drawn. Because calibration overall does not imply calibration on rare events \citep{allen2024tailcalibration,merkle2013choosing}, we also score tail questions in excess bits.",
     r"\citet{lerch2017forecasterdilemma} showed that scoring only the cases in which an extreme event occurred rewards forecasters who overstate its probability, which selecting tail questions on $q$ before any outcome is drawn avoids. Because calibration overall does not imply calibration on rare events \citep{allen2024tailcalibration,merkle2013choosing}, we also score tail questions in excess bits."),
    (r"\paragraph{Causal and conditional evaluation.}", r"\paragraph{Causal, conditional and updating evaluation.}"),
    (r"IARPA's FOCUS program proposed in 2017 to measure the probability of a counterfactual by running a simulated world repeatedly, with forecasters reading reports of AI-played Civilization games \citep{lehner2017focus,wiblin2019tetlock80k}, and described its metric as not a proper scoring rule. An interventional conditional follows the same idea with a proper rule, paired replays of a disease model and a language model as the forecaster.",
     r"IARPA's FOCUS program proposed in 2017 to score counterfactual forecasts by rerunning a simulated world, with forecasters reading reports of AI-played Civilization games \citep{lehner2017focus,wiblin2019tetlock80k}. An interventional conditional follows the same idea with a proper rule, paired replays of a disease model and a language model as the forecaster."),
    ("\n\\paragraph{Belief updating and simulated environments.}\nEvolveCast updates", " EvolveCast updates"),
    (r" Games and simulated worlds are established environments for LLMs that act \citep{qi2024civrealm,paglieri2025balrog} or that populate the world as agents \citep{park2023generativeagents,vezhnevets2023concordia}. Here the model acts on nothing, and we score it against the replay distribution.", ""),
]
for old, new in R8:
    if f8.count(old) != 1: print(f"ABORT: related work: {f8.count(old)} matches: {old[:80]!r}"); sys.exit(1)
    f8 = f8.replace(old, new)
for f, s in (("sections/05_freeciv.tex", f5), ("sections/08_related_work.tex", f8)):
    print("would write" if a.check else "wrote", f)
    if not a.check: (P / f).write_text(s)
