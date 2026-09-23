#!/usr/bin/env python
"""page_budget_2026-09-22.py --paper-root DIR [--check]

Jaeho's decision of 22 September 2026, after repeats_2026-09-22.py: Fabio's three Micropolis cuts and our remaining six.
  Fabio's (Section 3), each following what the FreeCiv and Starsim sections already do:
    - the engine-fix footnote goes (Appendix A's Micropolis paragraph already states the two fixes), and the disaster
      examples become one sentence of the three kinds;
    - the results no longer restate the tail metric's rationale (Section 2), the bootstrap (Table 2's caption) or the best
      and worst models with their scores (orange in the figure);
    - the figure caption points to Table 2 for rho and its interval.
    Our rewritten words inside his text are green, so he can see them.
  Ours:
    - the combined-score paragraph keeps two sentences; the rest moves to Appendix C, which already held the fit and the
      leave-one-out ranges and now also holds the ten-cell interval, the 0.98 agreement and the caveat;
    - the introduction's "we instantiate" paragraph merges into contribution 2, which said the same thing;
    - the ForecastBench leaderboard sentence goes (the parity claim and its citations stay);
    - the FreeCiv section title matches the other two ("Micropolis: rare events in a city simulator");
    - Related work no longer re-cites the scoring rules that Section 2 cites;
    - the Discussion no longer repeats Section 6's saturation sentence.
Each anchor must match exactly once.
"""
import argparse, sys
from pathlib import Path

ap = argparse.ArgumentParser(); ap.add_argument("--paper-root", required=True); ap.add_argument("--check", action="store_true")
a = ap.parse_args(); P = Path(a.paper_root).resolve()
R = [
    # ---- Fabio's Section 3
    ("sections/04_micropolis.tex",
     r"""\footnote{\blue{While implementing our benchmark, we found and fixed an out-of-bounds crash in the engine and a reseeding of its random number generator from the clock on every load. With the fixes in place, evolution is fully deterministic once a random seed is set.}}""", ""),
    ("sections/04_micropolis.tex",
     r"""Some of these happen at fixed background rates (e.g., earthquakes and tornadoes), \blue{others need a feature of the city to occur at all (a flood needs a shoreline, an airplane crash an airport, a meltdown a nuclear power plant, whose number sets the rate)} and yet others have threshold conditions (monster attacks happen only if average pollution exceeds a value). They have varying degrees of impact in different aspects of future evolution of the city.""",
     r"""\green{Some happen at fixed background rates, others need a feature of the city, and others occur only above a threshold.}"""),
    ("sections/04_micropolis.tex",
     r"""Tail questions are scored in excess bits $\mathrm{KL}(q\,\|\,f)$ with the clipping of \cref{sec:design-scoring}, the tail metric shared with FreeCiv, since at $q$ below 5\% an always-No forecast has a near-zero excess Brier and separates nothing; that correlation is""",
     r"""\green{For tail questions in excess bits it is}"""),
    ("sections/04_micropolis.tex",
     r""" Each interval is a percentile bootstrap over the \MPDNModelsBinary\ models, which asks how far the correlation could move under a different panel of models.""", ""),
    ("sections/04_micropolis.tex",
     r""" Excess Brier runs from \MPDBestBinary\ (\MPDBestBinaryModel) to \MPDWorstBinary\ (\MPDWorstBinaryModel) on mid-range questions, and excess bits from \MPDBestTail\ (\MPDBestTailModel) to \MPDWorstTail\ (\MPDWorstTailModel) on tail questions.}""", "}"),
    ("sections/04_micropolis.tex",
     r""", overlapping the intervals of both binary scores, and the score runs from \MPDBestContinuous\ (\MPDBestContinuousModel) to \MPDWorstContinuous\ (\MPDWorstContinuousModel).}""",
     r""", overlapping the intervals of both binary scores.}"""),
    ("sections/04_micropolis.tex",
     r""" $\rho$ is Spearman rank correlation of ECI with $-$score, sign-adjusted so that positive means more capable models forecast better; brackets are 95\% percentile intervals from \MPDResamples\ bootstrap resamples over the models. Orange marks the best and worst model.}}""",
     r""" \green{$\rho$ and its interval are those of \cref{tab:validation}, and orange marks the best and worst model.}}}"""),
    # ---- Section 6: combined score in two sentences; the rest in Appendix C
    ("sections/07_validation.tex",
     r"""Every model ran in every row of \cref{tab:validation}, so the ten rows can be read as the items of one test. We fit $\log \ell_{mc} = \delta_c - \theta_m + \varepsilon_{mc}$ to each model's reported loss $\ell_{mc}$ in cell $c$ by alternating least squares, where $\delta_c$ is the difficulty of the cell and $\theta_m$ the skill of the model, centered at zero. The logarithm lets cells with different loss scales share one axis. Over the six Micropolis and Starsim cells, $\theta_m$ correlates with ECI at $\rho = \CombinedSixRho$ [\CombinedSixLo, \CombinedSixHi], against \blue{$\MPDRhoContinuous$} for the best single cell, and leaving out any one cell moves $\rho$ between \CombinedSixLooLo{} and \CombinedSixLooHi{}. With the four FreeCiv cells added, $\rho = \CombinedTenRho$ [\CombinedTenLo, \CombinedTenHi], and the two orderings agree at $\rho = \CombinedAgreement$.""",
     r"""Every model ran in every row of \cref{tab:validation}, so the ten rows can be read as the items of one test. A latent skill fitted to the log losses of the six Micropolis and Starsim cells correlates with ECI at $\rho = \CombinedSixRho$ [\CombinedSixLo, \CombinedSixHi], against \blue{$\MPDRhoContinuous$} for the best single cell, and at $\rho = \CombinedTenRho$ with the four FreeCiv cells added (\cref{app:combined})."""),
    ("sections/07_validation.tex",
     r""" ECI is itself a latent-ability fit across benchmarks \citep{ho2025rosetta}. The fit weights every cell equally and is one summary, not a headline. \Cref{app:combined} lists $\theta_m$ per model.""", ""),
    ("appendix/C_full_results.tex",
     r"""and \cref{fig:combined-score} plots both against ECI. The fit is""",
     r"""and \cref{fig:combined-score} plots both against ECI. The logarithm lets cells with different loss scales share one axis. With the four FreeCiv cells added, the correlation with ECI is \CombinedTenRho{} [\CombinedTenLo, \CombinedTenHi], and the six-cell and ten-cell orderings agree at $\rho = \CombinedAgreement$. ECI is itself a latent-ability fit across benchmarks \citep{ho2025rosetta}, and this fit weights every cell equally, so it is one summary of the ten rows and not a headline. The fit is"""),
    # ---- Introduction
    ("sections/02_introduction.tex",
     "\\begin{greentext}\nWe instantiate the design on \\blue{three simulations} and score a shared panel of 24 models on each. Micropolis, a city simulation built on the open-source SimCity engine \\citep{micropolis2008}, provides tail questions. FreeCiv, an open-source strategy game \\citep{freeciv2026}, provides natural conditionals. Starsim, an agent-based disease model \\citep{kerr2024starsim}, provides interventional conditionals.\n\\end{greentext}\n",
     ""),
    ("sections/02_introduction.tex",
     r"""\item \green{Results for a shared panel of 24 models on \blue{three simulations} (\Cref{sec:micropolis,sec:freeciv,sec:starsim}).""",
     r"""\item \green{Results for a shared panel of 24 models on \blue{three simulations}: Micropolis \citep{micropolis2008} for tail questions, FreeCiv \citep{freeciv2026} for natural conditionals and Starsim \citep{kerr2024starsim} for interventional conditionals (\Cref{sec:micropolis,sec:freeciv,sec:starsim}).""" ),
    ("sections/02_introduction.tex",
     r""" On the ForecastBench tournament leaderboard, the two leading submitted systems rank above the superforecaster median with overlapping confidence intervals, while models run without tools or added context remain below it \citep{forecastbench2026leaderboard}.}""",
     r"""}"""),
    # ---- Section 4 title, Related work, Discussion
    ("sections/05_freeciv.tex",
     r"""\section{FreeCiv: tail, natural-conditional, continuous and mid-range questions in a strategy game}""",
     r"""\section{FreeCiv: natural conditionals in a strategy game}"""),
    ("sections/08_related_work.tex",
     r"""Our scoring rests on the Brier score \citep{brier1950verification}, the CRPS \citep{matheson1976scoring} and the theory of strictly proper scoring rules \citep{gneiting2007proper}. \citet{murphy1973vector} partitioned the Brier score into reliability, resolution and uncertainty. For a rare event the uncertainty term is most of the raw score,""",
     r"""For a rare event the uncertainty term of the Brier score \citep{murphy1973vector} is most of the raw score,"""),
    ("sections/09_discussion.tex", " The Starsim binary question is close to saturation.\n", "\n"),
]
texts = {}
for f, old, new in R:
    s = texts.get(f) or (P / f).read_text()
    if s.count(old) != 1: print(f"ABORT: {f}: {s.count(old)} matches: {old[:90]!r}"); sys.exit(1)
    texts[f] = s.replace(old, new)
for f, s in texts.items():
    print("would write" if a.check else "wrote", f)
    if not a.check: (P / f).write_text(s)
