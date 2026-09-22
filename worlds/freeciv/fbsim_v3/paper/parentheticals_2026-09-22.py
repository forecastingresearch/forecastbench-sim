#!/usr/bin/env python
"""parentheticals_2026-09-22.py --paper-root DIR [--check]

Jaeho's request of 22 September 2026: fewer parentheticals (ours ran 14.8 per 1,000 words against a corpus median of 8),
starting with the least informative: repeated cross-references (one pointer per paragraph), restated ranges, asides
that read as clauses; ECI spelled out only where it is defined; and the one redundant passage the scan found, Nick's
coverage-gradient sentence in the Discussion that repeats Section 5 (cut with his permission).  Each anchor must match
exactly once.
"""
import argparse, sys
from pathlib import Path

ap = argparse.ArgumentParser(); ap.add_argument("--paper-root", required=True); ap.add_argument("--check", action="store_true")
a = ap.parse_args(); P = Path(a.paper_root).resolve()
R = [
    # Section 1 (Jaeho's own sentence): the example becomes a clause
    ("sections/02_introduction.tex", r"On causal conditionals (for example, forecasting the effect of a public policy decision), real-world forecasts do not give us a clean counterfactual.",
     r"On causal conditionals, such as the effect of a public policy decision, real-world forecasts do not give us a clean counterfactual."),
    # Section 2: one pointer to Section 5 is enough; the validation pointer after "rank models" is obvious; Nick's asides become clauses
    ("sections/03_benchmark_design.tex", r"\red{The Starsim tables report the excess CRPS divided by one constant per horizon, computed from the replays alone (\cref{sec:starsim}).}",
     r"\red{The Starsim tables report the excess CRPS divided by one constant per horizon, computed from the replays alone.}"),
    ("sections/03_benchmark_design.tex", r"\red{Starsim scores its binary questions in excess bits as well (\cref{sec:starsim}).}", r"\red{Starsim scores its binary questions in excess bits as well.}"),
    ("sections/03_benchmark_design.tex", r"We use ECI only to rank models (\cref{sec:validation}).", r"We use ECI only to rank models."),
    ("sections/03_benchmark_design.tex", r"In Micropolis and FreeCiv each question was asked once, and in Starsim five times (binary) or three times (continuous) with the median taken.}",
     r"In Micropolis and FreeCiv each question was asked once, and in Starsim five times for binary and three times for continuous questions, with the median taken.}"),
    ("sections/03_benchmark_design.tex", r"\red{Starsim's sets of 8 items stay at one question per prompt (\cref{sec:starsim}).}", r"\red{Starsim's sets of 8 items stay at one question per prompt.}"),
    # Section 4 setup: one appendix pointer; the median q is in Appendix A
    ("sections/05_freeciv.tex", r"and the tail set 300 questions with $0<q\le 0.05$ (median $q$ 0.013).", r"and the tail set 300 questions with $0<q\le 0.05$."),
    ("sections/05_freeciv.tex", r"from news about another civilization to a change on the question's own series, and not by any measured effect (\cref{app:worlds}).",
     r"from news about another civilization to a change on the question's own series, and not by any measured effect."),
    ("sections/05_freeciv.tex", r"since turn 2 of a cell continues that exchange by revealing one sentence and asking the question again (\cref{app:worlds}).",
     r"since turn 2 of a cell continues that exchange by revealing one sentence and asking the question again."),
    # Section 4 results: ECI after its definition; one appendix pointer per paragraph; the range is stated in the setup
    ("sections/05_freeciv.tex", r"\green{\Cref{fig:freeciv-capability} plots each set's headline score against the Epoch Capabilities Index.", r"\green{\Cref{fig:freeciv-capability} plots each set's headline score against ECI."),
    ("sections/05_freeciv.tex", r"Mean bits rise with the horizon (\cref{app:full-freeciv}).", r"Mean bits rise with the horizon."),
    ("sections/05_freeciv.tex", r"Models with higher ECI track the true shift more closely without gaining more (\cref{app:full-freeciv}).", r"Models with higher ECI track the true shift more closely without gaining more."),
    ("sections/05_freeciv.tex", r"Mean excess more than doubles from turn 90 to turn 210 (\cref{app:full-freeciv}).", r"Mean excess more than doubles from turn 90 to turn 210."),
    ("sections/05_freeciv.tex", r"On the 750 mid-range questions, whose ground truth is spread uniformly over $(0.05, 0.95)$, excess Brier averages 0.141,", r"On the 750 mid-range questions, excess Brier averages 0.141,"),
    # Section 6 and 8: ECI after its definition
    ("sections/07_validation.tex", r"We check whether \blue{each simulation's} headline score orders models as the Epoch Capabilities Index does.", r"We check whether \blue{each simulation's} headline score orders models as ECI does."),
    ("sections/09_discussion.tex", r"\blue{In every simulation} the models that rank higher on the Epoch Capabilities Index forecast better on most question sets (\cref{tab:validation}),",
     r"\blue{In every simulation} the models that rank higher on ECI forecast better on most question sets (\cref{tab:validation}),"),
    # Discussion: Nick's coverage-gradient sentence repeats Section 5 (cut with his permission)
    ("sections/09_discussion.tex", r" \red{The gradient is strongest at 25 percent coverage ($\rho = -0.66$), weaker at 50 percent ($-0.46$) and weaker still at 90 percent ($-0.25$), where mean excess is lowest (0.33, against 0.98 at 25 percent and 0.65 at 50 percent).}", ""),
]
texts = {}
for f, old, new in R:
    s = texts.get(f) or (P / f).read_text()
    if s.count(old) != 1: print(f"ABORT: {f}: {s.count(old)} matches: {old[:90]!r}"); sys.exit(1)
    texts[f] = s.replace(old, new)
for f, s in texts.items():
    print("would write" if a.check else "wrote", f)
    if not a.check: (P / f).write_text(s)
