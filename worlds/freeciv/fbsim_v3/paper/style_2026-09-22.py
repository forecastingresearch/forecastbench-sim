#!/usr/bin/env python
"""style_2026-09-22.py --paper-root DIR [--check]

Jaeho's style pass of 22 September 2026 on the FreeCiv-owned text (and, with Nick's permission, two of his sentences in
Section 2): semicolons and clause-joining colons become separate sentences (colons that introduce a list or a definition
stay), sentences with an abstract subject get an agent where one exists, the longest sentences are split, two of the
least important per-simulation clauses of Section 2 go (Nick's percentile comparison and his reason for bits, kept as a
short sentence), Figure 1 writes the ground truth as q, p(Y|X) is lower-case everywhere, and the four British spellings
become American.  Each replacement must match exactly once.
"""
import argparse, re, sys
from pathlib import Path

ap = argparse.ArgumentParser(); ap.add_argument("--paper-root", required=True); ap.add_argument("--check", action="store_true")
a = ap.parse_args(); P = Path(a.paper_root).resolve()
R = [
    # ---------------- Section 1 (Jaeho's own paragraph, then the green parts)
    ("sections/02_introduction.tex",
     r"Second, real-world forecasting questions are poorly suited to understanding tail events: rare events are rare by definition, meaning they are sparse in any given evaluation period, making it difficult to estimate how good models are at estimating tail events. (Tail events are outsized in their importance to practical forecasting questions; we very frequently want to know how likely an unlikely event is, and how its likelihood is changing over time).",
     r"Second, real-world forecasting questions are poorly suited to understanding tail events. Rare events are rare by definition, so they are sparse in any evaluation period, and that makes it difficult to estimate how well models forecast them. Tail events matter out of proportion to their number, because we very often want to know how likely an unlikely event is and how that likelihood is changing over time."),
    ("sections/02_introduction.tex",
     r"On correlative conditionals, the real world provides us with only one timeline of events; our ability to understand how often X and Y co-occur is naturally limited in the real world.",
     r"On correlative conditionals, the real world provides us with only one timeline of events, so we can learn only so much about how often X and Y occur together."),
    ("sections/02_introduction.tex",
     r"\green{We run a world to a snapshot, describe its state in a world report, pose questions about its future, and resolve each question by replay: fresh continuations of the world from its snapshot under new random seeds, or, in Starsim, runs matched approximately to the displayed count (\Cref{fig:schematic}). The frequency of an event across replays is its ground-truth probability, so resolution is immediate, and a forecast is scored by its excess score, the loss beyond that of the ground-truth probability itself. Replays also yield three question types that real-world benchmarks cannot easily evaluate: tail questions, interventional conditionals and natural conditionals. \Cref{sec:design} defines these terms.}",
     r"\green{We run a world to a snapshot, describe its state in a world report, pose questions about its future, and resolve each question by replay, that is, by fresh continuations of the world from its snapshot under new random seeds (\Cref{fig:schematic}). The frequency of an event across replays is its ground-truth probability, so resolution is immediate. We score a forecast by its excess score, the loss beyond that of the ground-truth probability itself. Replays also yield three question types that real-world benchmarks cannot easily evaluate: tail questions, interventional conditionals and natural conditionals. \Cref{sec:design} defines these terms.}"),
    ("sections/02_introduction.tex",
     r"Micropolis, a city simulation built on the open-source SimCity engine \citep{micropolis2008}, provides tail questions; FreeCiv, an open-source strategy game \citep{freeciv2026}, provides natural conditionals; Starsim, an agent-based disease model \citep{kerr2024starsim}, provides interventional conditionals.",
     r"Micropolis, a city simulation built on the open-source SimCity engine \citep{micropolis2008}, provides tail questions. FreeCiv, an open-source strategy game \citep{freeciv2026}, provides natural conditionals. Starsim, an agent-based disease model \citep{kerr2024starsim}, provides interventional conditionals."),
    ("sections/02_introduction.tex",
     r"(\Cref{sec:micropolis,sec:freeciv,sec:starsim}); on Micropolis tail questions, on FreeCiv's tail and continuous sets, and on Starsim's interventional questions in unnormalized units, forecasts move further from the ground truth as the horizon grows.}",
     r"(\Cref{sec:micropolis,sec:freeciv,sec:starsim}). On Micropolis tail questions, on FreeCiv's tail and continuous sets, and on Starsim's interventional questions in unnormalized units, forecasts move further from the ground truth as the horizon grows.}"),
    ("sections/02_introduction.tex",
     r"is positive, and its 95\% bootstrap interval over models excludes zero for eight of the ten; a latent skill fitted \blue{across simulations} correlates with capability at \CombinedSixRho{}.}",
     r"is positive, and its 95\% bootstrap interval over models excludes zero for eight of the ten. A latent skill fitted \blue{across simulations} correlates with capability at \CombinedSixRho{}.}"),
    # ---------------- Section 2
    ("sections/03_benchmark_design.tex",
     r"ForecastBench-Sim layers a forecasting benchmark on a \blue{\emph{simulation}}: a program, such as a game or a disease model, whose state changes under fixed rules from a random seed.",
     r"ForecastBench-Sim layers a forecasting benchmark on a \blue{\emph{simulation}}, a program such as a game or a disease model whose state changes under fixed rules from a random seed."),
    ("sections/03_benchmark_design.tex",
     r"We run the world to a chosen turn, the \emph{snapshot}, and write a \emph{world report}: a text document that gives the state at the snapshot, the history of the main quantities and the events so far, and that is the only information a forecaster sees. A question generator produces questions from templates, each with a \emph{horizon}, the turn at which it resolves. Rather than wait for the world to reach the horizon once, we \emph{replay} it: $N$ new continuations from the snapshot under new random seeds, with the outcome read off in each (\cref{fig:schematic}).",
     r"We run the world to a chosen turn, the \emph{snapshot}, and write a \emph{world report}, a text document that gives the state at the snapshot, the history of the main quantities and the events so far. The report is the only information a forecaster sees. A question generator produces questions from templates, each with a \emph{horizon}, the turn at which it resolves. Rather than wait for the world to reach the horizon once, we \emph{replay} it, running $N$ new continuations from the snapshot under new random seeds and reading off the outcome in each (\cref{fig:schematic})."),
    ("sections/03_benchmark_design.tex",
     r"A \emph{natural conditional} reveals one true fact $X$ about the interval after the snapshot and asks the same question $Y$ again; its ground truth is $P(Y\mid X)$, the share of replays in which $Y$ occurs among those in which $X$ holds. An \emph{interventional conditional} applies an intervention, such as a vaccination campaign, to a paired copy of the world at the snapshot and asks about the outcome in that copy; its ground truth comes from the replays of the paired world, and the unconditional question provides the counterfactual.",
     r"A \emph{natural conditional} reveals one true fact $X$ about the interval after the snapshot and asks the same question $Y$ again. Its ground truth is $p(Y\mid X)$, the share of replays in which $Y$ occurs among those in which $X$ holds. An \emph{interventional conditional} applies an intervention, such as a vaccination campaign, to a paired copy of the world at the snapshot and asks about the outcome in that copy. Its ground truth comes from the replays of the paired world, and the unconditional question provides the counterfactual."),
    ("sections/03_benchmark_design.tex",
     r"We score forecasts against the replay ground truth using proper scoring rules or their quantile approximations and report the \emph{excess score}: the expected score of the forecast under the replay distribution minus the expected score of the ground truth itself.",
     r"We score forecasts against the replay ground truth using proper scoring rules or their quantile approximations, and we report the \emph{excess score}, the expected score of the forecast under the replay distribution minus the expected score of the ground truth itself."),
    ("sections/03_benchmark_design.tex",
     r"\red{The Starsim tables report the excess CRPS divided by one constant per horizon, computed from the replays alone (\cref{sec:starsim}); Starsim elicits the 10th to 90th percentiles where FreeCiv elicits the 5th to 95th.} Squared error does little to distinguish tail forecasts: for any forecast in $[0, 0.05]$, the excess Brier score on a tail question is at most $0.0025$. FreeCiv therefore also scores tail questions in \emph{excess bits}, $\mathrm{KL}(q\,\|\,f)$, the excess of the logarithmic score, which penalizes the ratio of $f$ to $q$ rather than their difference; forecasts are bounded to $[0.001, 0.999]$ first (\cref{app:protocol}). \red{Starsim scores its binary questions the same way, since every one of their ground truths is 0, 1 or within 0.03 of one of them (\cref{sec:starsim}).}",
     r"\red{The Starsim tables report the excess CRPS divided by one constant per horizon, computed from the replays alone (\cref{sec:starsim}).} Squared error does little to distinguish tail forecasts, since for any forecast in $[0, 0.05]$ the excess Brier score on a tail question is at most $0.0025$. FreeCiv therefore also scores tail questions in \emph{excess bits}, $\mathrm{KL}(q\,\|\,f)$, the excess of the logarithmic score, which penalizes the ratio of $f$ to $q$ rather than their difference. We bound forecasts to $[0.001, 0.999]$ first (\cref{app:protocol}). \red{Starsim scores its binary questions in excess bits as well (\cref{sec:starsim}).}"),
    ("sections/03_benchmark_design.tex",
     r"\red{In \blue{every simulation}, reasoning effort was set to the lowest level each model supports, or to a 1,024-token limit where no level exists; in Micropolis and FreeCiv each question was asked once, and in Starsim five times (binary) or three times (continuous) with the median taken.} The prompts differ \blue{across simulations} in the role assigned to the model, in whether the scoring rule is named, in questions per prompt and in repeated calls (\cref{tab:protocol} in \cref{app:worlds}; \cref{app:prompts} reproduces every prompt).",
     r"\red{In \blue{every simulation}, reasoning effort was set to the lowest level each model supports, or to a 1,024-token limit where no level exists. In Micropolis and FreeCiv each question was asked once, and in Starsim five times (binary) or three times (continuous) with the median taken.} The prompts differ \blue{across simulations} in the role assigned to the model, in whether the scoring rule is named, in questions per prompt and in repeated calls (\cref{tab:protocol} in \cref{app:worlds}, and \cref{app:prompts} reproduces every prompt)."),
    ("sections/03_benchmark_design.tex",
     r"\green{FreeCiv used the same limits for its unconditional sets and one question per prompt for its natural conditionals, whose second turn continues the conversation about that question.}",
     r"\green{FreeCiv used the same limits, and one question per prompt for its natural conditionals, whose second turn continues the conversation.}"),
    # ---------------- Section 4
    ("sections/05_freeciv.tex",
     r"Most facts move the truth little: 174 cells shift by less than 0.03 and 63 by at least 0.15.",
     r"Most facts move the truth little. Only 63 cells shift by at least 0.15, and 174 shift by less than 0.03."),
    ("sections/05_freeciv.tex",
     r"\green{\Cref{fig:freeciv-capability} plots each set's headline score against the Epoch Capabilities Index; \cref{app:full-results} shows the same scores by horizon (\cref{fig:freeciv-horizon}) and the turn-2 moves on the natural-conditional cells (\cref{fig:freeciv-natcond}).}",
     r"\green{\Cref{fig:freeciv-capability} plots each set's headline score against the Epoch Capabilities Index. \Cref{app:full-results} shows the same scores by horizon (\cref{fig:freeciv-horizon}) and the turn-2 moves on the natural-conditional cells (\cref{fig:freeciv-natcond}).}"),
    ("sections/05_freeciv.tex",
     r"The loss comes from overstating rare events. Every model places a mean forecast of 0.13 to 0.34 on them, and no model scores below the 0.03 bits of a constant forecast of 0.05.",
     r"Models lose these bits by overstating rare events. Every model places a mean forecast of 0.13 to 0.34 on them, and none scores below the 0.03 bits of a constant forecast of 0.05."),
    ("sections/05_freeciv.tex",
     r"The update itself helps little. Models move in the direction of the fact: the correlation between move and true shift $p(Y \mid X) - p(Y)$ is 0.45 to 0.60 per model, and the sign agrees on 86\% to 94\% of the 185 cells whose shift exceeds 0.05. But they move too far.",
     r"The update itself helps little. Models move in the direction of the fact. The correlation between a model's move and the true shift $p(Y \mid X) - p(Y)$ is 0.45 to 0.60, and the sign agrees on 86\% to 94\% of the 185 cells whose shift exceeds 0.05. But they move too far."),
    ("sections/05_freeciv.tex",
     r"The turn-2 ranking reflects the turn-1 forecasts and, among them, the models that keep their forecasts nearest the middle: the spread of a model's turn-2 forecasts predicts its score ($\rho = 0.91$; \cref{app:full-freeciv}), and no model scores better than a constant forecast of 0.5. The correlation between move and shift rises with ECI ($\rho = 0.61$); the gain does not (0.03).",
     r"The turn-2 ranking reflects the turn-1 forecasts and, among them, rewards the models that keep their forecasts nearest the middle. The spread of a model's turn-2 forecasts predicts its score ($\rho = 0.91$, \cref{app:full-freeciv}), and no model scores better than a constant forecast of 0.5. The correlation between move and shift rises with ECI ($\rho = 0.61$), and the gain does not (0.03)."),
    ("sections/05_freeciv.tex",
     r"The intervals are too narrow: a model's 5th-to-95th percentile interval contains 22\% (GPT-5 Nano) to 77\% (GPT-5.6 Sol) of the replay values, and this coverage predicts most of the score ($\rho = -0.90$).",
     r"Models draw their intervals too narrow. A model's 5th-to-95th percentile interval contains 22\% (GPT-5 Nano) to 77\% (GPT-5.6 Sol) of the replay values, and this coverage predicts most of the score ($\rho = -0.90$)."),
    ("sections/05_freeciv.tex",
     r"The score is dominated by a bias that every model shares. Forecasts are compressed and low: the mean of $f - q$ is negative for all 24 models ($-0.33$ to $-0.07$), and no model's excess is below the 0.075 of a constant forecast of 0.5 (\cref{fig:freeciv-reliability}).",
     r"A bias that every model shares dominates the score. Every model compresses its forecasts and sets them low. The mean of $f - q$ is negative for all 24 models ($-0.33$ to $-0.07$), and no model's excess is below the 0.075 of a constant forecast of 0.5 (\cref{fig:freeciv-reliability})."),
    # ---------------- Section 6
    ("sections/07_validation.tex",
     r"The interval is a percentile bootstrap over the $n$ models in every row, which measures how much $\rho$ would move under a different panel of models and conditions on the question sets; for FreeCiv, \cref{app:validation-full} adds a cluster bootstrap over its eight worlds with the panel fixed.",
     r"The interval is a percentile bootstrap over the $n$ models in every row. It measures how much $\rho$ would move under a different panel of models and conditions on the question sets. For FreeCiv, \cref{app:validation-full} adds a cluster bootstrap over its eight worlds with the panel fixed."),
    ("sections/07_validation.tex",
     r"and the interval lies above zero in eight of the ten rows; it includes zero for the FreeCiv tail and mid-range rows, whose gradients are the weakest. Two rows have a limitation: the Starsim binary question is close to saturation, with 14 of 24 models within 0.05 bit of the ground truth, and the Starsim interventional row has $n=23$.",
     r"and the interval lies above zero in eight of the ten rows. It includes zero for the FreeCiv tail and mid-range rows, whose gradients are the weakest. Two rows carry a caveat. The Starsim binary question is close to saturation, with 14 of 24 models within 0.05 bit of the ground truth, and the Starsim interventional row has $n=23$."),
    ("sections/07_validation.tex",
     r"where $\delta_c$ is the difficulty of the cell and $\theta_m$ the skill of the model, centered at zero; the logarithm lets cells with different loss scales share one axis.",
     r"where $\delta_c$ is the difficulty of the cell and $\theta_m$ the skill of the model, centered at zero. The logarithm lets cells with different loss scales share one axis."),
    ("sections/07_validation.tex",
     r"The fit weights every cell equally and is one summary, not a headline; \cref{app:combined} lists $\theta_m$ per model.",
     r"The fit weights every cell equally and is one summary, not a headline. \Cref{app:combined} lists $\theta_m$ per model."),
    # ---------------- Section 7 (related work): semicolons only; length options are separate
    ("sections/08_related_work.tex",
     r"Benchmarks of LLM forecasting draw their questions from tournaments, markets and the news: ForecastQA and Autocast assembled resolved questions \citep{jin2021forecastqa,zou2022autocast}, ForecastBench generates new questions over time and scores models beside human forecasters \citep{karger2024forecastbench}, and Prophet Arena scores live market questions \citep{yang2026prophetarena}.",
     r"Benchmarks of LLM forecasting draw their questions from tournaments, markets and the news. ForecastQA and Autocast assembled resolved questions \citep{jin2021forecastqa,zou2022autocast}, ForecastBench generates new questions over time and scores models beside human forecasters \citep{karger2024forecastbench}, and Prophet Arena scores live market questions \citep{yang2026prophetarena}."),
    ("sections/08_related_work.tex",
     r"A 2026 survey lists one simulated-world benchmark \citep{xu2026llmforecastingagents}: the FreeCiv-only design of \citet{lee2026forecastbenchsim}, which resolved each question against a single hidden continuation.",
     r"A 2026 survey lists one simulated-world benchmark \citep{xu2026llmforecastingagents}, the FreeCiv-only design of \citet{lee2026forecastbenchsim}, which resolved each question against a single hidden continuation."),
    ("sections/08_related_work.tex",
     r"\citet{murphy1973vector} partitioned the Brier score into reliability, resolution and uncertainty; for a rare event the uncertainty term is most of the raw score, and \citet{paleka2025pitfalls} show that averaging raw scores across base rates hides rare-event skill.",
     r"\citet{murphy1973vector} partitioned the Brier score into reliability, resolution and uncertainty. For a rare event the uncertainty term is most of the raw score, and \citet{paleka2025pitfalls} show that averaging raw scores across base rates hides rare-event skill."),
    ("sections/08_related_work.tex",
     r"\citet{lerch2017forecasterdilemma} showed that evaluating only the cases in which an extreme event occurred rewards forecasters who overstate its probability; we select tail questions on $q$ before any outcome is drawn.",
     r"\citet{lerch2017forecasterdilemma} showed that evaluating only the cases in which an extreme event occurred rewards forecasters who overstate its probability. We select tail questions on $q$ before any outcome is drawn."),
    ("sections/08_related_work.tex",
     r"or let an agent intervene in a small generative model \citep{gandhi2025boxinggym,yang2026causalab}; InterveneBench scores against the single realized outcome of a real study \citep{shi2026intervenebench}.",
     r"or let an agent intervene in a small generative model \citep{gandhi2025boxinggym,yang2026causalab}. InterveneBench scores against the single realized outcome of a real study \citep{shi2026intervenebench}."),
    ("sections/08_related_work.tex",
     r"EvolveCast updates LLM forecasts on real news and can compare each update only with those of human forecasters \citep{yuan2025evolvecast}; other work tests whether updates are consistent with Bayes' theorem without an event-level truth \citep{imran2025bayesian,he2025martingale}, and BayesBench scores beliefs against exact posteriors under a specified likelihood \citep{samanta2026bayesbench}. A natural conditional scores an update against the replay estimate of $P(Y\mid X)$ for a fact whose likelihood has no formula.",
     r"EvolveCast updates LLM forecasts on real news and can compare each update only with those of human forecasters \citep{yuan2025evolvecast}. Other work tests whether updates are consistent with Bayes' theorem without an event-level truth \citep{imran2025bayesian,he2025martingale}, and BayesBench scores beliefs against exact posteriors under a specified likelihood \citep{samanta2026bayesbench}. A natural conditional scores an update against the replay estimate of $p(Y\mid X)$ for a fact whose likelihood has no formula."),
    ("sections/08_related_work.tex",
     r"or that populate the world as agents \citep{park2023generativeagents,vezhnevets2023concordia}; here the model acts on nothing, and we score it against the replay distribution.",
     r"or that populate the world as agents \citep{park2023generativeagents,vezhnevets2023concordia}. Here the model acts on nothing, and we score it against the replay distribution."),
    # ---------------- Section 8 and 9
    ("sections/09_discussion.tex",
     r"Horizon effects depend on the question set: Starsim interventional excess rises in people while unconditional continuous excess falls.",
     r"Horizon effects depend on the question set. Starsim's interventional excess rises with the horizon in people while its unconditional continuous excess falls."),
    ("sections/09_discussion.tex",
     r"In Micropolis and FreeCiv the ground truth conditions on the full saved state while the model sees only the report; Starsim instead approximately matches the displayed count. These information differences limit comparisons with a zero-excess forecaster.",
     r"In Micropolis and FreeCiv the ground truth conditions on the full saved state while the model sees only the report, and Starsim instead approximately matches the displayed count. These information differences limit comparisons with a zero-excess forecaster."),
    ("sections/09_discussion.tex",
     r"And with 24 models the intervals in \cref{tab:validation} are at least 0.34 wide; which adjacent models differ needs item-level intervals, computed here for FreeCiv only.",
     r"And with 24 models the intervals in \cref{tab:validation} are at least 0.34 wide. Telling adjacent models apart needs item-level intervals, which we computed for FreeCiv only."),
    ("sections/09_discussion.tex",
     r"The result says that more capable models forecast outcomes under a stated intervention on an epidemic more accurately; it does not say how the same models would forecast a policy whose mechanism is not a simulator.",
     r"The result says that more capable models forecast outcomes under a stated intervention on an epidemic more accurately. It does not say how the same models would forecast a policy whose mechanism is not a simulator."),
    ("sections/09_discussion.tex",
     r"Here that frequency says the models move too far on facts that change the truth little: 174 of the 400 facts shift it by less than 0.03.",
     r"Here that frequency says the models move too far on facts that change the truth little, since 174 of the 400 facts shift it by less than 0.03."),
    ("sections/09_discussion.tex",
     r"Several directions use the design as it stands: \blue{more simulations}, including a social simulator built on generative agents \citep{vezhnevets2023concordia}; sequential forecasting over several snapshots of one replay set; human baselines at scale, beyond the pilot of \cref{app:human-pilot}; conditionals in \blue{every simulation}; and fresh draws on a schedule, so that a released set that enters a training corpus can be replaced.",
     r"Several directions use the design as it stands. \blue{More simulations} can be added, including a social simulator built on generative agents \citep{vezhnevets2023concordia}. Sequential forecasting over several snapshots of one replay set, human baselines at scale beyond the pilot of \cref{app:human-pilot}, and conditionals in \blue{every simulation} are direct extensions. Fresh draws on a schedule would let a released set that enters a training corpus be replaced."),
    # ---------------- Figure 1: the ground truth is q
    ("figures/fig_schematic.tex", r"{true $p$, or an\\outcome\\distribution\\{\color{figorange}\itshape tail}: $p \le 0.05$};", r"{true $q$, or an\\outcome\\distribution\\{\color{figorange}\itshape tail}: $q \le 0.05$};"),
    ("figures/fig_schematic.tex", r"{$p_{\mathrm{do}}$\\{\color{figorange}\itshape interventional}};", r"{$q_{\mathrm{do}}$\\{\color{figorange}\itshape interventional}};"),
]
texts = {}
for f, old, new in R:
    s = texts.get(f) or (P / f).read_text()
    if s.count(old) != 1: print(f"ABORT: {f}: {s.count(old)} matches: {old[:90]!r}"); sys.exit(1)
    texts[f] = s.replace(old, new)
# American spelling in every tex file of the paper (four British forms remain)
for f in list(P.glob("sections/*.tex")) + list(P.glob("appendix/*.tex")) + list(P.glob("data/**/*.tex")):
    rel = str(f.relative_to(P)); s = texts.get(rel) or f.read_text()
    s2 = re.sub(r"\b([Nn]ormalis)(ed|e|es|ing|ation)\b", lambda m: m.group(1)[:-1] + "z" + m.group(2), s); s2 = re.sub(r"\bcentred\b", "centered", s2)
    if s2 != s: texts[rel] = s2
for f, s in texts.items():
    print("would write" if a.check else "wrote", f)
    if not a.check: (P / f).write_text(s)
