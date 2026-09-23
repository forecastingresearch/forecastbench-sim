#!/usr/bin/env python
"""repeats_2026-09-22.py --paper-root DIR [--check]

Jaeho's request of 22 September 2026 (page budget, 80/20): cut facts the main text states more than once.
  - Starsim facts stated three to five times (Section 2, Section 5, its figure caption, the Discussion); Nick allowed any cut.
  - FreeCiv setup restated in the results paragraphs, in Section 2 (batching limits) and in Appendix A (criteria, groups).
  - The Discussion's natural-conditional paragraph, which restated Section 4's results and the Related-work contrast.
  - Section 6 text that restated its table caption; the Figure 1 caption that restated Section 2's definitions.
  - Wording: "Simulation" for the column that names Micropolis, FreeCiv and Starsim; "Worlds and question sets" for the
    FreeCiv and Starsim subsections (eight and four worlds); Appendix A named the mid-range set after itself (a leftover of
    the rename of 21 September), where it means the files' name, bank.
The Discussion no longer says models move "more than twice the true shift": per model the ratio of mean move to mean true
shift runs 1.66 (GPT-5) to 3.46 (Gemini 2.5 Flash), pooled 2.47 (data/freeciv/freeciv_natcond_forecasts.csv).
Each anchor must match exactly once.
"""
import argparse, sys
from pathlib import Path

ap = argparse.ArgumentParser(); ap.add_argument("--paper-root", required=True); ap.add_argument("--check", action="store_true")
a = ap.parse_args(); P = Path(a.paper_root).resolve()
R = [
    # ---- Figure 1 caption: the definitions are in Section 2 on the same page; the Starsim exception stays
    ("sections/03_benchmark_design.tex",
     r"""\caption{\green{ForecastBench-Sim. \blue{A simulation} (FreeCiv, Micropolis or Starsim) is run to a snapshot and described in a world report, and a question generator produces binary, continuous and conditional questions from templates. Each question is resolved by $N$ replays of the snapshot under new random seeds; Starsim instead filters runs from the initial state by their day-20 count. The replays give the ground-truth probability \blue{$q$} or an outcome distribution, a tail label when that probability is at most $0.05$, the probability under an intervention applied to a paired world, and the natural-conditional probability $p(Y\mid X)$ over the replays in which the revealed fact $X$ holds. A model reads the report and the questions and returns forecasts, scored by their excess over the ground truth (excess Brier score, excess CRPS, and excess bits on tail questions); lower is better.}}""",
     r"""\caption{\green{ForecastBench-Sim. \blue{A simulation} is run to a snapshot and described in a world report, and questions are generated from templates. Replays of the snapshot under new random seeds give each question its ground-truth probability \blue{$q$} or outcome distribution. Starsim instead keeps the runs from its initial state that match the shown run at day 20. A model reads the report and the questions, and its forecasts are scored by their excess over the ground truth.}}"""),
    # ---- Section 2: three Starsim sentences that Section 5 and Table 1 state (Nick's, cut with his permission)
    ("sections/03_benchmark_design.tex",
     r""" \red{Starsim's ground truth is the distribution over 131 to 290 matched replays per world, and its continuous items are answered as five percentiles (\cref{sec:starsim}).}""", ""),
    ("sections/03_benchmark_design.tex",
     r""" \red{The Starsim tables report the excess CRPS divided by one constant per horizon, computed from the replays alone.}""", ""),
    ("sections/03_benchmark_design.tex", r""" \red{Starsim's sets of 8 items stay at one question per prompt.}""", ""),
    # ---- Section 4: setup stated once
    ("sections/05_freeciv.tex", r"\subsection{World and question sets}", r"\subsection{Worlds and question sets}"),
    ("sections/05_freeciv.tex",
     r"Questions come from 23 binary and nine continuous templates, each a family with parameters whose resolution criteria state only how we read the quantity from the saved game, at horizons",
     r"Questions come from 23 binary and nine continuous templates at horizons"),
    ("sections/05_freeciv.tex",
     r"We drew the cells in five structural groups defined by the relation between the fact and the question, from news about another civilization to a change on the question's own series, and not by any measured effect.",
     r"We drew the cells by the relation between the fact and the question, not by any measured effect."),
    ("sections/05_freeciv.tex",
     r" We asked the unconditional sets in prompts of up to 50 binary or 20 continuous questions, the limits of the Micropolis run. Each natural-conditional question had its own prompt, since turn 2 of a cell continues that exchange by revealing one sentence and asking the question again.", ""),
    ("sections/05_freeciv.tex",
     r"""\green{\Cref{fig:freeciv-capability} plots each set's headline score against ECI. \Cref{app:full-results} shows the same scores by horizon (\cref{fig:freeciv-horizon}) and the turn-2 moves on the natural-conditional cells (\cref{fig:freeciv-natcond}).}
""", ""),
    ("sections/05_freeciv.tex",
     r"The 300 tail questions ask about events whose ground-truth probability is at most 0.05 (mean 0.019). Excess bits average 0.33 over the 24 models",
     r"On the 300 tail questions, excess bits average 0.33 over the 24 models"),
    ("sections/05_freeciv.tex", r"on events that occur with probability 0.02,", r"on events that occur with probability 0.02 on average,"),
    ("sections/05_freeciv.tex",
     r"After its turn-1 forecast the model learns one fact about turns 61 to 90 and forecasts again, and the target is $p(Y \mid X)$ from the replays. The turn-2 excess Brier score averages",
     r"After the fact is revealed, the turn-2 excess Brier score averages"),
    ("sections/05_freeciv.tex",
     r"The 300 continuous questions ask for five percentiles of a count or of a value at the horizon, scored by the CRPS of the five, divided by the family constant and net of the replay distribution's own CRPS. Excess nCRPS averages",
     r"On the 300 continuous questions, excess nCRPS averages"),
    # ---- Section 5 (Nick's): the case for causal forecasting is the introduction's; n = 23 is in the text below, the figure and Table 2; the CRPS is Section 2's
    ("sections/06_starsim.tex", r"\subsection{World and question sets}", r"\subsection{Worlds and question sets}"),
    ("sections/06_starsim.tex",
     r"Causal forecasting asks how an intervention changes an outcome, but real-world interventions rarely supply clean counterfactuals. Simulation lets us pair runs with and without an intervention. We use Starsim",
     r"We use Starsim"),
    ("sections/06_starsim.tex",
     r" DeepSeek V4 Flash completed only 21 of 24 interventional items, so interventional correlations use $n=23$ models rather than 24.", ""),
    ("sections/06_starsim.tex",
     r"The score is a five-quantile approximation to CRPS, net of the ground-truth distribution's own score at those quantiles (\cref{sec:design-scoring}). We divide this excess",
     r"We divide the excess CRPS of \cref{sec:design-scoring}"),
    ("sections/06_starsim.tex",
     r"""\caption{\red{Interventional excess CRPS in Starsim against the Epoch Capabilities Index, one point per model, pooled over both horizons; lower is better, 0 attains the floor at the scored quantiles. The excess is divided by one constant per horizon (200 people at day 40, 300 at day 60). The main panel pools the three coverage levels; the side panels show 25, 50 and 90 percent coverage. Spearman $\rho$ with a 95 percent percentile bootstrap interval over models (10,000 draws, seed 0); a negative $\rho$ means that the more capable models have the lower excess. DeepSeek V4 Flash (open marker) completed 21 of 24 items, so we exclude it from every $\rho$ and $n = 23$.}}""",
     r"""\caption{\red{Interventional excess CRPS in Starsim against the Epoch Capabilities Index, pooled over both horizons. Lower is better. The main panel pools the three coverage levels, and the side panels show each. Spearman $\rho$ with a 95 percent percentile bootstrap interval over models, where a negative $\rho$ means that the more capable models have the lower excess. DeepSeek V4 Flash (open marker) completed 21 of 24 items and is left out of every $\rho$.}}"""),
    # ---- Section 6: the text no longer restates the caption; the caption points to Appendix C for the seeds
    ("sections/07_validation.tex",
     r"""\Cref{tab:validation} gives, for each score, the Spearman rank correlation $\rho$ with ECI over the $n$ models, pooled over horizons and signed so that a positive $\rho$ means that more capable models score better, with a percentile bootstrap over models. For FreeCiv, \cref{app:validation-full} adds a cluster bootstrap over its eight worlds.""",
     r"""\Cref{tab:validation} gives the Spearman correlation of each score with ECI, pooled over horizons."""),
    ("sections/07_validation.tex",
     r"""Two rows carry a caveat. The Starsim binary question is close to saturation, with 14 of 24 models within 0.05 bit of the ground truth, and the Starsim interventional row has $n=23$.""",
     r"""The Starsim binary question is close to saturation, with 14 of 24 models within 0.05 bit of the ground truth."""),
    ("sections/07_validation.tex",
     r"""95\% CI: percentile bootstrap over models (\blue{\MPDResamples\ resamples, at seed \MPDSeed\ for Micropolis}; seed 0 for Starsim; seed 2026 for FreeCiv), with FreeCiv's interval over its eight worlds in \cref{app:validation-full}; $p$: two-sided.""",
     r"""95\% CI: percentile bootstrap over models, with the seeds and FreeCiv's interval over its eight worlds in \cref{app:validation-full}; $p$: two-sided."""),
    # ---- Discussion: the Starsim correlation in Table 2's sign convention; the natural-conditional paragraph without the repeats
    ("sections/09_discussion.tex",
     r"""\red{In Starsim their excess correlates with ECI at $\rho = -0.66$ [$-0.84$, $-0.34$] with $n = 23$, where a negative value means that the more capable models have the lower excess, about as strongly as the unconditional questions of the same world ($-0.64$ and $-0.69$).}""",
     r"""\red{In Starsim their score tracks ECI about as closely as the unconditional scores of the same world, at $\rho = 0.66$ against $0.64$ and $0.69$ in \cref{tab:validation}.}"""),
    ("sections/09_discussion.tex",
     r"""Models move the right way after a revealed fact on 86\% to 94\% of the cells where the truth moves, but by more than twice the true shift, so the update gains nothing on average over keeping the turn-1 forecast. The models with higher ECI track the true shift more closely ($\rho = 0.61$) without gaining more (0.03). Studies""",
     r"""Models move the right way after a revealed fact, but by 1.7 to 3.5 times the true shift, so the update gains nothing on average (\cref{sec:freeciv-results}). Studies"""),
    ("sections/09_discussion.tex",
     r""" Those settings score an update against human forecasters or against a posterior with a formula. A natural conditional scores it against the frequency of the outcome given the fact, which a consistency check without a truth value cannot supply \citep{paleka2024consistency}. Here that frequency says the models move too far on facts that change the truth little, since 174 of the 400 facts shift it by less than 0.03.""", ""),
    # ---- Wording: the column names a simulation
    ("data/validation_table_main.tex", r"World & Score & $n$ & $\rho$ & 95\% CI & $p$ \\", r"Simulation & Score & $n$ & $\rho$ & 95\% CI & $p$ \\"),
    ("data/validation_table.tex", r"World & Score & $n$ & $\rho$ & CI (models) & $p$ & CI (worlds) \\", r"Simulation & Score & $n$ & $\rho$ & CI (models) & $p$ & CI (worlds) \\"),
    ("data/validation_table_forecastbench.tex", r"World & Score & $n$ & $\rho$ & CI (models) & $p$ & CI (worlds) \\", r"Simulation & Score & $n$ & $\rho$ & CI (models) & $p$ & CI (worlds) \\"),
    ("data/appendix_tables/cost_per_item.tex", r"World & Question kind & Items & Calls/model & Cost/model (\$) & Cents/item \\", r"Simulation & Question kind & Items & Calls/model & Cost/model (\$) & Cents/item \\"),
    ("appendix/A_worlds_and_protocol.tex", r"The mid-range set, which the released files call the mid-range set,", r"The mid-range set, which the released files call the bank,"),
]
texts = {}
for f, old, new in R:
    s = texts.get(f) or (P / f).read_text()
    if s.count(old) != 1: print(f"ABORT: {f}: {s.count(old)} matches: {old[:90]!r}"); sys.exit(1)
    texts[f] = s.replace(old, new)
for f, s in texts.items():
    print("would write" if a.check else "wrote", f)
    if not a.check: (P / f).write_text(s)
