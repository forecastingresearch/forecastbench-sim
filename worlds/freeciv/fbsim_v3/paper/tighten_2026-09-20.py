#!/usr/bin/env python
"""tighten_2026-09-20.py --paper-root DIR --results RUN2_PAPER_DIR [--check]

The FreeCiv-owned edits of 20 September 2026, after run 2 was delivered:

  1. Main text (Jaeho's request: a verbosity pass).  Sentences among the recently added green text that repeat the
     appendix or the neighbouring sentence are cut; setup detail moves to Appendix A; every statement that run 2
     made stale is corrected (provisional labels, "one-question run", the FreeCiv bank rho, the six "completed" cells).
  2. Appendix A: the FreeCiv elicitation paragraph describes the grouped prompts, the run of 9 September is a short
     record, the hosting paragraph reports the pins of the run the paper reports, the model-file names are current,
     and the two scoring details cut from Section 2 (the tail definitions, the bits floor) land in the protocol subsection.
  3. Appendix B: the FreeCiv prompts are the grouped prompts of 20 September, with the one-question natural-conditional
     prompt kept, and the parsing paragraph describes both.
  4. Appendix C: the FreeCiv prose paragraphs quote run-2 numbers (the tables and captions were regenerated already).
  5. Appendix D: a FreeCiv questions-per-prompt subsection (run 1 vs run 2, and the order/cap attempt), and the no-news
     table and paragraph recomputed from the run-2 rows.

Only green or dark-yellow text in FreeCiv-owned passages is touched; red (Nick), blue (Fabio) and black text is
never changed.  Each replacement must match exactly once, or the script stops before writing anything.
"""
import argparse, gzip, re, sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ap = argparse.ArgumentParser()
ap.add_argument("--paper-root", required=True)
ap.add_argument("--results", required=True, help="results/run2_paper")
ap.add_argument("--check", action="store_true")
a = ap.parse_args()
P = Path(a.paper_root).resolve()
RES = Path(a.results).resolve()

# ----------------------------------------------------------------------------------------------------------------------
# numbers for the no-news control (Appendix D), from the run-2 rows
si = pd.read_csv(RES / "scores_v1" / "score_items.csv.gz", low_memory=False)
w = pd.read_csv(RES / "results_v1" / "freeciv_results_wide.csv").set_index("model")
nc = si[(si.set == "natcond") & si.p_nonews.notna()].copy()
nc["drift"] = nc.p_nonews - nc.p1
nc["move"] = (nc.p2 - nc.p1).abs()
eps = 1e-9
g = nc.groupby("model")
T = pd.DataFrame(dict(eci=w.eci.astype(float), n=g.size(), abs_drift=g.drift.apply(lambda s: s.abs().mean()), drift=g.drift.mean(),
                      unchanged=g.drift.apply(lambda s: (s.abs() < eps).mean()), gt05=g.drift.apply(lambda s: (s.abs() > 0.05 + eps).mean()),
                      move=g.move.mean())).sort_values("eci", ascending=False)
NAMES = {"claude-fable-5": "Fable", "claude-opus-5": "Opus 5", "gpt-5.6-sol": "GPT-5.6 Sol", "gpt-5.5": "GPT-5.5", "gemini-3.7-flash": "Gemini 3.7 Flash",
         "gpt-5.6-luna": "GPT-5.6 Luna", "claude-sonnet-5": "Sonnet 5", "deepseek-v4-flash-0731": "DeepSeek V4 Flash", "gemini-3-flash-preview": "Gemini 3 Flash Preview",
         "gpt-5": "GPT-5", "o3": "o3", "gpt-5.4-nano": "GPT-5.4 Nano", "o4-mini": "o4-mini", "gpt-5-mini": "GPT-5 Mini", "gemini-3.1-flash-lite-preview": "Gemini 3.1 Flash Lite",
         "qwen3.5-flash-02-23": "Qwen3.5 Flash", "claude-haiku-4.5": "Haiku 4.5", "gpt-5-nano": "GPT-5 Nano", "kimi-k2": "Kimi K2", "gemini-2.5-flash": "Gemini 2.5 Flash",
         "qwen3-235b-a22b": "Qwen3 235B", "gpt-4.1": "GPT-4.1", "deepseek-chat": "DeepSeek V3", "llama-4-scout": "Llama 4 Scout"}
rows = []
for m, r in T.iterrows():
    ad = "$<0.001$" if r.abs_drift < 0.0005 else f"{r.abs_drift:.3f}"
    rows.append(f"\\green{{{NAMES[m.split('/')[1]]}}} & \\green{{{r.eci:.1f}}} & \\green{{{ad}}} & \\green{{${r.drift:+.3f}$}} & "
                f"\\green{{{round(100 * r.unchanged):.0f}\\%}} & \\green{{{round(100 * r.gt05):.0f}\\%}} & \\green{{{r.move:.3f}}} \\\\")
NONEWS_ROWS = "\n".join(rows)
rng = np.random.default_rng(2026)
x, y = T.eci.values, T.abs_drift.values
rho_d, p_d = spearmanr(x, y)
idx = rng.integers(0, len(x), size=(10000, len(x)))
b = np.array([spearmanr(x[i], y[i])[0] for i in idx]); b = b[~np.isnan(b)]
lo_d, hi_d = np.percentile(b, 2.5), np.percentile(b, 97.5)
acc = (g.excess_nonews.mean() - g.excess_t1_uncond.mean())
acc_mean, acc_max_m, acc_max = acc.mean(), acc.idxmax(), acc.max()
N = dict(abs_drift=T.abs_drift.mean(), lo_model=NAMES[T.abs_drift.idxmin().split("/")[1]], lo=T.abs_drift.min(), hi_model=NAMES[T.abs_drift.idxmax().split("/")[1]],
         hi=T.abs_drift.max(), n_small=int((T.abs_drift < 0.005).sum()), signed=T.drift.mean(), kimi=T.loc["moonshotai/kimi-k2", "drift"],
         haiku=T.loc["anthropic/claude-haiku-4.5", "drift"], move=T.move.mean(), ratio=T.move.mean() / T.abs_drift.mean(), rho=rho_d, lo_ci=lo_d, hi_ci=hi_d,
         p=p_d, acc_mean=acc_mean, acc_max=acc_max, acc_max_model=NAMES[acc_max_m.split("/")[1]],
         gpt55_unch=round(100 * T.loc["openai/gpt-5.5", "unchanged"]), sonnet_unch=round(100 * T.loc["anthropic/claude-sonnet-5", "unchanged"]),
         two_pct=sorted({NAMES[m.split("/")[1]] for m in T.index[T.unchanged <= 0.025]}))
print("no-news numbers:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in N.items()})

# ----------------------------------------------------------------------------------------------------------------------
R = []  # (file, old, new)

# ---- 1. main text ---------------------------------------------------------------------------------------------------
R += [
    ("sections/02_introduction.tex",
     r"in Micropolis and FreeCiv, and on StarSim's interventional questions in unnormalised units, forecasts move further from the ground truth as the horizon grows.}",
     r"in Micropolis, on FreeCiv's tail and continuous sets, and on StarSim's interventional questions in unnormalised units, forecasts move further from the ground truth as the horizon grows.}"),

    ("sections/03_benchmark_design.tex",
     r"A model reads the report and the question and returns a forecast $f$, scored by its excess over the ground truth: excess Brier score, excess CRPS, and, in FreeCiv and StarSim, excess bits on tail questions. Lower excess is better.}}",
     r"A model reads the report and the questions and returns forecasts, scored by their excess over the ground truth (excess Brier score, excess CRPS, and excess bits on tail questions); lower is better.}}"),
    ("sections/03_benchmark_design.tex",
     r"We do not wait for the world to reach the horizon once. We \emph{replay} it: we run $N$ new continuations from the snapshot under new random seeds and read off the outcome in each (\cref{fig:schematic}).",
     r"Rather than wait for the world to reach the horizon once, we \emph{replay} it: $N$ new continuations from the snapshot under new random seeds, with the outcome read off in each (\cref{fig:schematic})."),
    ("sections/03_benchmark_design.tex",
     r"so a world contributes as many rare events as its replays contain.\footnote{Micropolis assigns a question to the tail when $q<0.05$ and includes $q=0$; FreeCiv requires $0<q\le 0.05$.} A \emph{natural conditional}",
     r"so a world contributes as many rare events as its replays contain (\cref{app:protocol} gives each world's exact rule). A \emph{natural conditional}"),
    ("sections/03_benchmark_design.tex",
     r"For a strictly proper rule without clipping, this is at least zero and zero only at the ground truth \citep{gneiting2007proper}; quantile approximations identify only the scored quantiles. It measures skill net of the uncertainty under the replay distribution.",
     r"For a strictly proper rule this is at least zero and zero only at the ground truth \citep{gneiting2007proper}, so it measures skill net of the uncertainty that remains under the replay distribution."),
    ("sections/03_benchmark_design.tex",
     r"forecasts are bounded to $[0.001, 0.999]$ first. At $q=0$ or $1$, this clipping gives a minimum of $-\log_2(0.999) \approx 0.00144$ bits. \red{",
     r"forecasts are bounded to $[0.001, 0.999]$ first (\cref{app:protocol}). \red{"),
    ("sections/03_benchmark_design.tex",
     r"ECI runs from 130.5 to 162.5 over the set, and we use it only to rank models (\cref{sec:validation}). Every prompt has the same shape: the world report, then the question, then an instruction to end the answer with one probability or with a set of quantiles.",
     r"We use ECI only to rank models (\cref{sec:validation}). Every prompt has the same shape: the world report, then the questions, then an instruction to end the answer with a probability or a set of quantiles for each."),
    ("sections/03_benchmark_design.tex",
     r"\green{FreeCiv uses the same limits for its unconditional sets, 50 binary and 20 continuous questions per prompt. Its natural conditionals keep one question per prompt, since the second turn continues the conversation about that question.}",
     r"\green{FreeCiv used the same limits for its unconditional sets and one question per prompt for its natural conditionals, whose second turn continues the conversation about that question.}"),
    ("sections/03_benchmark_design.tex", r" & \green{Micropolis} & \green{FreeCiv (provisional)} & \green{StarSim} \\", r" & \green{Micropolis} & \green{FreeCiv} & \green{StarSim} \\"),

    ("sections/05_freeciv.tex",
     r"The world report, about 4,900 tokens, gives each civilization's state at turn 60, the diplomatic state of every pair, the same quantities every five turns from turn 1, and a log of every city, government, diplomatic, technology and wonder event so far; it contains no map and names no research target (\cref{app:prompts}).",
     r"The world report, about 4,900 tokens, gives each civilization's state at turn 60 and at every fifth turn before it, the diplomatic state of every pair and a log of the events so far, but no map (\cref{app:prompts})."),
    ("sections/05_freeciv.tex",
     r"Questions come from 23 binary and nine continuous templates, each a family with parameters whose resolution criteria state only how we read the quantity from the saved game, at horizons of turn 90, 120, 150, 180 and 210. The binary bank holds 750 questions, 30 from each of five equal-width intervals of $q$ on $(0.05, 0.95)$ at each horizon. The tail set holds 300 questions with $0<q\le 0.05$ (median $q$ 0.013). The continuous set holds 300 questions on counts and on values at the horizon, answered with five percentiles and scored by CRPS divided by one fixed constant per family. The natural-conditional set holds 400 cells at horizons 120 to 210; each pairs one of 355 binary questions with one fact about turns 61 to 90 that held in at least 100 replays. We drew cells in five structural groups, not by measured effect: A, news about a civilization or pair the question is not about; B, related but weak news; C1, a change of magnitude on the question's own series; C2, the same kind of change on a different series; and D, news from which the outcome must be inferred (\cref{app:worlds}). After the draw, 174 cells have a true shift $|p(Y \mid X) - p(Y)|$ below 0.03 and 63 a shift of at least 0.15. The questions of one game share its report. We asked the bank, tail, mirror and continuous questions in prompts of at most 50 binary or 20 continuous questions, the limits of the Micropolis run, in horizon order for binary questions and by quantity for continuous ones. Each natural-conditional question had its own prompt, because turn 2 of a cell continues that exchange: it reveals one sentence and asks the question again. A no-news control on 100 cells continues the exchange with no new information (\cref{app:worlds}).",
     r"Questions come from 23 binary and nine continuous templates, each a family with parameters whose resolution criteria state only how we read the quantity from the saved game, at horizons of turn 90, 120, 150, 180 and 210. The binary bank holds 750 questions, 30 from each of five equal-width intervals of $q$ on $(0.05, 0.95)$ at each horizon, and the tail set 300 questions with $0<q\le 0.05$ (median $q$ 0.013). The continuous set holds 300 questions on counts and on values at the horizon, answered with five percentiles and scored by CRPS divided by one fixed constant per family. The natural-conditional set holds 400 cells at horizons 120 to 210, each pairing one of 355 binary questions with one fact about turns 61 to 90 that held in at least 100 replays. We drew the cells in five structural groups defined by the relation between the fact and the question, from news about another civilization to a change on the question's own series, and not by any measured effect (\cref{app:worlds}). Most facts move the truth little: 174 cells shift by less than 0.03 and 63 by at least 0.15. We asked the unconditional sets in prompts of up to 50 binary or 20 continuous questions, the limits of the Micropolis run. Each natural-conditional question had its own prompt, since turn 2 of a cell continues that exchange by revealing one sentence and asking the question again (\cref{app:worlds})."),
    ("sections/05_freeciv.tex", r"\subsection{Results (provisional)}", r"\subsection{Results}"),
    ("sections/05_freeciv.tex",
     r"and the turn-2 moves on the natural-conditional cells (\cref{fig:freeciv-natcond}). Every panel comes from the one-question run.}",
     r"and the turn-2 moves on the natural-conditional cells (\cref{fig:freeciv-natcond}).}"),
    ("sections/05_freeciv.tex",
     r"$\rho$ is sign-adjusted Spearman with a 95\% bootstrap over models (5,000 resamples, within 0.02 of the intervals of \cref{sec:validation}); orange marks the best and worst model.}}",
     r"$\rho$ is sign-adjusted Spearman with a 95\% bootstrap over models (5,000 resamples; \cref{tab:validation} gives the interval over the eight anchor games); orange marks the best and worst model.}}"),

    ("sections/07_validation.tex",
     r"For Micropolis and StarSim the interval is a percentile bootstrap over the $n$ models, which measures how much $\rho$ would move under a different panel of models. This conditions on the question sets; StarSim has four underlying worlds and no world-resampled interval here. For FreeCiv it is a cluster bootstrap over the eight anchor games with the panel fixed (\cref{app:validation-full} gives both intervals for every row).",
     r"For Micropolis and StarSim the interval is a percentile bootstrap over the $n$ models, which measures how much $\rho$ would move under a different panel of models and conditions on the question sets; for FreeCiv it is a cluster bootstrap over the eight anchor games with the panel fixed (\cref{app:validation-full} gives both intervals for every row)."),
    ("sections/07_validation.tex", r"Every point estimate is positive, from 0.12 on the FreeCiv bank to", r"Every point estimate is positive, from 0.18 on the FreeCiv bank to"),
    ("sections/07_validation.tex", r"The StarSim interventional row leaves out DeepSeek V4 Flash (21 of 24 items completed). The FreeCiv rows are provisional.}}", r"The StarSim interventional row leaves out DeepSeek V4 Flash (21 of 24 items completed).}}"),
    ("sections/07_validation.tex",
     r"the logarithm lets cells with different loss scales share one axis (losses are floored at $10^{-4}$ before the logarithm; no cell reaches the floor).",
     r"the logarithm lets cells with different loss scales share one axis."),
    ("sections/07_validation.tex",
     r"ECI is itself a latent-ability fit across benchmarks \citep{ho2025rosetta}. The continuous Micropolis and FreeCiv cells use raw nCRPS; all other cells use excess losses. The fit weights every cell equally",
     r"ECI is itself a latent-ability fit across benchmarks \citep{ho2025rosetta}. The fit weights every cell equally"),

    ("sections/09_discussion.tex",
     r"In the two completed worlds the models that rank higher on the Epoch Capabilities Index forecast better (\cref{tab:validation}), which suggests that the measured skill is related to general capability and not specific to one world, and horizon effects depend on the question set: StarSim interventional excess rises in people while unconditional continuous excess falls. The StarSim binary question is close to saturation. \yellow{The FreeCiv results, including the natural conditionals, wait on the batched rerun.}",
     r"In every world the models that rank higher on the Epoch Capabilities Index forecast better on most question sets (\cref{tab:validation}), which suggests that the measured skill is related to general capability and not specific to one world, and horizon effects depend on the question set: StarSim interventional excess rises in people while unconditional continuous excess falls. The StarSim binary question is close to saturation."),
    ("sections/09_discussion.tex",
     r"and the latent skill fitted across the six completed cells correlates at 0.78, compared with \blue{$\MPDRhoContinuous$} for the best single cell. With provisional FreeCiv cells included, the point estimate rises to 0.88. The worlds",
     r"and a latent skill fitted across six Micropolis and StarSim cells correlates at 0.78, compared with \blue{$\MPDRhoContinuous$} for the best single cell. The worlds"),
    ("sections/09_discussion.tex",
     r"conditionals in every world; training forecasters on simulated worlds, which supply resolved questions on demand, with transfer to real questions as the first test; and fresh draws on a schedule,",
     r"conditionals in every world; and fresh draws on a schedule,"),

    ("sections/00_legend.tex",
     r"All FreeCiv numbers are provisional (one question per prompt) pending the batched rerun, so FreeCiv results text is dark yellow and its figures are labelled provisional.",
     r"FreeCiv numbers are those of the grouped run of 20 September 2026; its four result paragraphs and the Discussion paragraph are still to be written (boxes)."),
]

# ---- 2. Appendix A --------------------------------------------------------------------------------------------------
R += [
    ("appendix/A_worlds_and_protocol.tex",
     r"The shared shape, world report before question and an instruction to end with a probability or a set of quantiles, is described in \cref{sec:design-protocol}.",
     r"The shared shape, world report before the questions and an instruction to end with a probability or a set of quantiles, is described in \cref{sec:design-protocol}. Two scoring details of \cref{sec:design} belong here. Micropolis assigns a question to the tail when $q<0.05$ and includes $q=0$, while FreeCiv requires $0<q\le 0.05$. The excess bits bound forecasts to $[0.001, 0.999]$ before the logarithm, so at $q=0$ or $1$ the score has a floor of $-\log_2(0.999) \approx 0.00144$ bits."),
    ("appendix/A_worlds_and_protocol.tex",
     r"We send every prompt through OpenRouter with the report before the questions. On first-party hosts the prompt has two content parts with a cache breakpoint between them; elsewhere it is one string. A prompt with several questions has 8,759 to 17,575 tokens; a natural-conditional prompt has 7,565 to 8,544. There is no system message and no persona. The prompt says a proper scoring rule will evaluate the answers and names none. It asks for one line per question in a delimited block, a probability or the five percentiles; a natural-conditional prompt asks for the probability inside a tag. Every request sets an output limit of 32,768 tokens and provider-default sampling. We set reasoning effort to the lowest level each model supports, with a 1,024-token budget for four models and no setting for four others (\cref{tab:roster}). We pin every model to the host that Micropolis used (\cref{tab:hosting}), so the two worlds queried the same endpoints. Every stored row records the host, the prompt, completion and reasoning token counts, the cost, the finish reason, the reasoning parameter sent and, where returned, the reasoning text. We parse a reply from its tag block or, failing that, from a labelled line, which we flag; a reply that parses neither way is asked once more and then stored as missing. We score a missing binary answer as 0.5 and drop missing continuous and conditional answers. Per model the run makes 1,524 turn-1 calls, 400 turn-2 calls and 99 no-news calls, 2,023 in all. The no-news control continues the turn-1 exchange of 100 cells, stratified by group, with a sentence that says the game has continued and no new information is available.",
     r"We send every prompt through OpenRouter with the report before the questions. On first-party hosts the prompt has two content parts with a cache breakpoint between them; elsewhere it is one string. There is no system message and no persona. The prompt says a proper scoring rule will evaluate the answers and names none. A grouped prompt of the run the paper reports has 8,759 to 17,575 tokens and asks for one line per question in a delimited block, a probability or the five percentiles; a natural-conditional prompt has 7,565 to 8,544 tokens and asks for the probability inside a tag (\cref{app:prompts}). Every request uses provider-default sampling and an output limit of 32,768 tokens (16,384 in the run of 9 September, with 24,000 for DeepSeek V4 Flash and 6,000 for Kimi K2). We set reasoning effort to the lowest level each model supports, with a 1,024-token budget for four models and no setting for four others (\cref{tab:roster}), and pin every model to one host (\cref{tab:hosting}). Every stored row records the host, the prompt, completion and reasoning token counts, the cost, the finish reason, the reasoning parameter sent and, where returned, the reasoning text. We read a grouped reply from its last delimited block, matching each line to its question by the stated number, and fall back to numbered lines outside the block or, for percentiles, to five numbers on the line after a restated question. We read a natural-conditional reply from its tag block or, failing that, from a labelled line, which we flag, and ask once more when neither can be read. A missing binary answer scores as a forecast of 0.5; missing continuous and conditional answers are dropped. The no-news control continues the turn-1 exchange of 100 cells, stratified by group, with a sentence that says the game has continued and no new information is available."),
    ("appendix/A_worlds_and_protocol.tex",
     r"The run of 9 September 2026 produced 48,543 of 48,552 expected rows. Twenty-three models finished in 41 minutes; DeepSeek V3 took 85 minutes because its host limited it to about 8 calls per minute. Twenty-one models parsed on every call. Kimi K2 (26 rows) and DeepSeek V4 Flash (27 rows) ran to the output limit with no answer, and we score those rows as 0.5 in the binary sets. DeepSeek V3 lacks 9 turn-2 rows whose turn-1 answer was empty, and GPT-5 Nano at minimal effort needed the lenient parse on about 12\% of its calls. DeepSeek V4 Flash ignores the effort setting and returns 10,000 to 15,000 reasoning tokens per call. The scored rows cost \$393.83 across the 24 models. A check at medium effort on 7 models and 200 bank items cost a further \$34 (\cref{app:ablation-effort}). That run asked one question per prompt.",
     r"The run of 9 September 2026 asked one question per prompt, 2,023 calls per model (1,524 turn-1, 400 turn-2 and 99 no-news), and produced 48,543 of 48,552 expected rows. Twenty-three models finished in 41 minutes; DeepSeek V3 took 85 minutes because its host limited it to about 8 calls per minute. Twenty-one models parsed on every call. Kimi K2 (26 rows) and DeepSeek V4 Flash (27 rows) ran to the output limit with no answer, DeepSeek V3 lacks 9 turn-2 rows whose turn-1 answer was empty, and GPT-5 Nano at minimal effort needed the lenient parse on about 12\% of its calls. DeepSeek V4 Flash ignores the effort setting and returns 10,000 to 15,000 reasoning tokens per call. The scored rows cost \$393.83 across the 24 models, and a check at medium effort on 7 models and 200 bank items a further \$34 (\cref{app:ablation-effort}). The paper uses this run's natural-conditional forecasts for 22 models and compares its other sets with the grouped run in \cref{app:ablation-grouping}."),
    ("appendix/A_worlds_and_protocol.tex",
     r"We sent each prompt once, and an answer we could not read stays missing.",
     r"We sent each prompt once and did not ask again for an answer we could not read."),
    ("appendix/A_worlds_and_protocol.tex",
     r"\Cref{tab:hosting} gives the hosts and the cost. In FreeCiv we pin every proprietary model to its developer's own endpoint (Google models to Google AI Studio, with Google as fallback), Qwen models to Alibaba, Llama 4 Scout and DeepSeek V3 to DeepInfra, and DeepSeek V4 Flash to DeepSeek; every other request excludes Novita. Kimi K2 is the one exception: on OpenRouter only Novita serves it, so we pin it there. DeepInfra serves DeepSeek V3 in 4-bit precision; the only other host of the original model does not state its precision. The Micropolis registry (\cref{tab:openrouter-ids}) pins the same developers' endpoints for proprietary models and Alibaba for Qwen, but routes Llama 4 Scout to Google Vertex and both DeepSeek models to StreamLake.",
     r"\Cref{tab:hosting} gives the hosts and the cost. In the FreeCiv run of 20 September we pinned every model to the host that the Micropolis registry pins (\cref{tab:openrouter-ids}): the developers' own services for the OpenAI, Google and Anthropic models, Alibaba for the Qwen models, DeepInfra for Llama 4 Scout, StreamLake for both DeepSeek models (DeepSeek V4 Flash in 8-bit precision), and Novita for Kimi K2, the one host that serves it. The per-call records show each model served throughout by its pinned host. The run of 9 September had pinned DeepSeek V3 to DeepInfra, which serves it in 4-bit precision, and DeepSeek V4 Flash to DeepSeek, so those two models' natural-conditional forecasts were elicited again on the StreamLake endpoints."),
    ("appendix/A_worlds_and_protocol.tex",
     r"The FreeCiv run cost \$393.83 for the scored rows over 48,551 calls, and the StarSim run \$40.99 over 5,045 recorded scored calls. Claude Fable, GPT-5.5 and Claude Opus 5 account for 45\% of the FreeCiv cost.",
     r"The FreeCiv rows cost \$174.15 over the 24 models: \$26.10 for the 1,056 grouped prompts of 20 September and \$148.05 for the natural-conditional arm, of which \$2.93 is the two reruns and the rest the calls of 9 September for the other 22 models. Claude Fable, GPT-5.5 and Claude Opus 5 account for 48\% of it. The StarSim run cost \$40.99 over 5,045 recorded scored calls."),
    ("appendix/A_worlds_and_protocol.tex",
     r"the reasoning setting in FreeCiv (\texttt{models\_v1.csv}) and Micropolis",
     r"the reasoning setting in FreeCiv (\texttt{models\_v2.csv}) and Micropolis"),
    ("appendix/A_worlds_and_protocol.tex",
     r"OpenRouter identifiers of the 24 models, as used in every results file (\texttt{models\_v1.csv}; the Micropolis \texttt{models.csv} and the StarSim tables use the same identifiers), and the host pinned for each model in the Micropolis registry (\texttt{model\_specs.json5}, as committed; \texttt{openai/default} is OpenAI's standard tier).",
     r"OpenRouter identifiers of the 24 models, as used in every results file (\texttt{models\_v2.csv}; the Micropolis \texttt{models.csv} and the StarSim tables use the same identifiers), and the host pinned for each model in the Micropolis registry (\texttt{model\_specs.json5}, as committed; \texttt{openai/default} is OpenAI's standard tier), which the FreeCiv run of 20 September adopted."),
]

# ---- 3. Appendix B --------------------------------------------------------------------------------------------------
B_OLD_START = r"Our harness sends one user message per question and no system message (\texttt{PREFLIGHT\_prompts.md})."
B_OLD_END = r"When neither could be read we asked once more (\texttt{ELICITATION.md})."
B_NEW = r"""Our harness sends one user message per prompt and no system message. The message has two parts. The first holds the opening instruction, a one-line question background and the full world report; the second holds the questions and the answer instructions. First-party hosts receive the first part with a cache breakpoint, and every other host receives the same text as one string. In the run the paper reports, a prompt for the bank, tail, mirror and continuous sets carries up to 50 binary or 20 continuous questions of one game, numbered, each with its resolution criteria, and asks for one answer line per question inside a delimited block; the instructions follow the Micropolis prompts above, with the FreeCiv background line and no persona. A natural-conditional cell keeps the one-question prompt of the run of 9 September: the same first part, then one question with its criteria and an answer in tags. Turn 2 of a cell continues that conversation with three messages: the turn-1 message, the model's own turn-1 reply, and a third message that reveals one fact and asks the question again. The no-news control replaces the third message with a statement that nothing new is known. The wording of every part follows, in green because our harness wrote it; the item names are those of the released question sets.
\end{greentext}

\begin{greentext}
\paragraph{Grouped binary prompt, opening of the first part} The world report follows the third line.
\begin{small}
\begin{verbatim}
You will be asked several binary forecasting questions about a simulation in
    progress. For each question, your task is to estimate the probability that
    the event described in the question occurs. Please provide your reasoning
    before stating your final probabilities.

Question Background: This is a partial report on a FreeCiv game simulation in
    progress, observed at turn 60. Five AI civilizations are competing.
\end{verbatim}
\end{small}

\paragraph{Second part, grouped binary} Prompt \texttt{seed7008\_bin\_c1of4}, 43 questions; the first two are shown.
\begin{small}
\begin{verbatim}
Questions (43 in total):

1. Will Peruvian have a higher population than Latvian at turn 90?
   Resolution Criteria: Resolves YES if Peruvian's population as recorded at the
    end of turn 90 is strictly greater than Latvian's. Equal values resolve NO.
    The quantity is the same quantity the world report calls Population.

2. Will Latvian have discovered University by turn 90?
   Resolution Criteria: Resolves YES if Latvian discovers University on any turn
    from 61 through 90 inclusive (an event on turn 90 itself counts). Resolves
    NO if it is not discovered by the end of turn 90.

[questions 3 to 43]

Think step by step about the information provided, reason about uncertainty, and
    then state your final confidence that each question resolves YES. Each
    probability should be a number between 0 and 1.

Your answers will be evaluated with a proper scoring rule, so your best strategy
    is to report your honest probability estimate for every question. You MUST
    provide a probability for every question.

Your response SHOULD STRICTLY END with your probabilities in this exact format,
    one line per question, in order:
<<<PROBABILITIES>>>
Q1: 0.65
Q2: 0.03
<<<END>>>
Provide one such line for each of the 43 questions, replacing the example values
    with your probability that the question resolves YES.
\end{verbatim}
\end{small}

\paragraph{Grouped continuous prompt} The first part opens with the paragraph below instead; the background line and the report are the same as in the binary prompt.
\begin{small}
\begin{verbatim}
You will be asked several numeric forecasting questions about a simulation in
    progress. For each question, your task is to estimate the full range of
    plausible values for the quantity described, expressed as percentiles of
    your forecast distribution. Please provide your reasoning before stating
    your final percentiles.
\end{verbatim}
\end{small}

\paragraph{Second part, grouped continuous} Prompt \texttt{seed7003\_cont\_c1of3}, 16 questions; the first is shown.
\begin{small}
\begin{verbatim}
Questions (16 in total):

1. How many wonders will be completed (across all civilizations) between turn 61
    and turn 90?
   Resolution Criteria: Resolves to the number of great wonders completed by all
    civilizations on any turn from 61 through 90 inclusive (an event on turn 90
    itself counts). Small wonders such as a palace do not count.

[questions 2 to 16]

Think step by step about the information provided, reason about uncertainty, and
    give your final answer to each question as five percentiles of your forecast
    distribution: p5, p25, p50, p75, p95. "p5" means you estimate a 5% chance
    the true value falls below that number; "p50" is your median. Your
    percentiles must be non-decreasing (p5 <= p25 <= p50 <= p75 <= p95). Every
    answer is a whole number.

Your answers will be evaluated with a proper scoring rule, so your best strategy
    is to report your honest estimate of each distribution - percentiles that
    are too narrow and percentiles that are too wide will both cost you. You
    MUST provide percentiles for every question.

Your response SHOULD STRICTLY END with your percentiles in this exact format, one
    line per question, in order:
<<<PERCENTILES>>>
Q1: p5=5, p25=10, p50=15, p75=20, p95=25
Q2: p5=100, p25=200, p50=300, p75=400, p95=500
<<<END>>>
Provide one such line for each of the 16 questions, replacing the example values
    with your actual percentile estimates.
\end{verbatim}
\end{small}

\paragraph{Natural-conditional turn-1 prompt, opening of the first part} One question per prompt, as in the run of 9 September. The world report follows the third line.
\begin{small}
\begin{verbatim}
You will be asked a binary forecasting question about a simulation in progress.
    Your task is to estimate the probability that the event described in the
    question occurs. Please provide your reasoning before stating your final
    probability.

Question Background: This is a partial report on a FreeCiv game simulation in
    progress, observed at turn 60. Five AI civilizations are competing.
\end{verbatim}
\end{small}

\paragraph{Second part, natural-conditional turn 1} Item \texttt{seed7003:b687}.
\begin{small}
\begin{verbatim}
Question Title: Will Ghana and Sammarinese reach a cease-fire or armistice at
    any point between turns 61 and 120?

Resolution Criteria: Resolves YES if the diplomatic state between Ghana and
    Sammarinese is Cease-fire or Armistice at the end of any turn from 61
    through 120 inclusive. Resolves NO if the state never occurs in that window
    (war, peace and alliance do not count).

Think step by step about the information provided, reason about uncertainty, and
    put your final confidence that the question resolves YES in <probability>
    </probability> tags. The probability should be a number between 0 and 1.

Your answer will be evaluated with a proper scoring rule, so your best strategy
    is to report your honest probability estimate.

Your final answer should be the probability that the event resolves YES and your
    response SHOULD STRICTLY END with <probability> </probability> tags.
\end{verbatim}
\end{small}

\paragraph{Turn-2 message of a natural-conditional cell} Cell \texttt{seed7003:b687|seed7003:r30}. The first sentence is the same in every cell up to the revealed fact; the turn on which the fact occurred is never disclosed.
\begin{small}
\begin{verbatim}
One fact about turns 61 through 90 has been revealed to you: Rhenish had at
    least four more cities at turn 90 than at turn 60.

Given this, please answer the same question again. Reason about what, if
    anything, this changes, and end your response with your updated probability
    in <probability> </probability> tags.
\end{verbatim}
\end{small}

\paragraph{No-news control message}
\begin{small}
\begin{verbatim}
The game has continued. No new information about turns 61 through 90 is
    available. If you wish, revise your forecast.

Please answer the same question again and end your response with your
    probability in <probability> </probability> tags.
\end{verbatim}
\end{small}

\paragraph{Reading the answer} For a grouped prompt we read the last delimited block in the reply and match each line to its question by the stated number. When the block is missing we accept numbered lines outside it and, for percentiles, five numbers on one line or on the line after a restated question, and mark the row as leniently parsed; each grouped prompt was sent once, a missing binary answer scores as 0.5, and a missing continuous answer is dropped. For a natural-conditional prompt the answer is the number inside the last \texttt{<probability>} tags; when the tags were missing we accepted a trailing ``probability \ldots 0.xx'' or a bare \texttt{<0.xx>} at the end and marked the row as leniently parsed, and when neither could be read we asked once more."""

# ---- 4. Appendix C --------------------------------------------------------------------------------------------------
R += [
    ("appendix/C_full_results.tex",
     r"it is 98.5\% or more for every model. The cost column is each model's OpenRouter charge for all of its calls, \$393.83 over the 24 models. The gain is positive for 8 of 24 models, with mean $-0.002$ and range $-0.018$ to $0.016$.",
     r"it is 99.1\% or more for every model. The cost column is each model's OpenRouter charge for the calls behind its scored rows, \$174.15 over the 24 models (\cref{tab:hosting}). The gain is positive for 7 of 24 models, with mean $-0.002$ and range $-0.018$ to $0.016$."),
    ("appendix/C_full_results.tex",
     r"Excess Brier runs from 0.112 (GPT-4.1) to 0.452 (o4-mini). A constant forecast of 0.5 has an excess of 0.233 on this set; eight models score below it, one of them by 0.0004, and a ninth is within 0.0001 of it. The signed Spearman correlation with ECI is 0.08 [$-0.36$, 0.51], $p = 0.72$, with the model bootstrap of \cref{sec:validation}.",
     r"Excess Brier runs from 0.263 (Gemini 3 Flash Preview) to 0.665 (GPT-5 Nano). A constant forecast of 0.5 has an excess of 0.233 on this set, and no model scores below it: the grouped prompts place low forecasts on these near-certain events, as they do on the bank (\cref{app:ablation-grouping}). The signed Spearman correlation with ECI is 0.10 [$-0.37$, 0.50], $p = 0.65$, with the model bootstrap of \cref{sec:validation}."),
    ("appendix/C_full_results.tex",
     r"Across the 24 models the mean bank score stays between 0.124 and 0.135 over the five horizons, and the mean natural-conditional score between 0.112 and 0.125 over its four. The mean tail bits rise from 0.45 at turn 90 to 0.74 at turn 210, and the mean nCRPS from 0.35 to 0.87. \Cref{fig:freeciv-horizon} plots the excess nCRPS instead, which subtracts the floor and rises from 0.28 to 0.65 over the same turns.",
     r"Across the 24 models the mean bank score stays between 0.136 and 0.149 over the five horizons, and the mean natural-conditional score between 0.112 and 0.124 over its four. The mean tail bits rise from 0.28 at turn 90 to 0.41 at turn 210, and the mean nCRPS from 0.41 to 1.03. \Cref{fig:freeciv-horizon} plots the excess nCRPS instead, which subtracts the floor and rises from 0.35 to 0.82 over the same turns."),
    ("appendix/C_full_results.tex",
     r"The gain over not updating is negative for 23 of 24 models in group C1, where the mean gain is $-0.054$. In the other four groups the mean gain lies between 0.004 and 0.011, and it is negative for 3, 8, 9 and 2 models in groups A, B, C2 and D.",
     r"The gain over not updating is negative for 23 of 24 models in group C1, where the mean gain is $-0.053$. In the other four groups the mean gain lies between 0.003 and 0.010, and it is negative for 3, 9, 9 and 4 models in groups A, B, C2 and D."),
    ("appendix/C_full_results.tex",
     r"\Cref{fig:freeciv-reliability} is the reliability diagram of the bank forecasts, and \cref{fig:freeciv-natcond} in the main text plots the move of each natural-conditional forecast against the true shift.",
     r"\Cref{fig:freeciv-reliability} is the reliability diagram of the bank forecasts, and \cref{fig:freeciv-natcond} above plots the move of each natural-conditional forecast against the true shift."),
    ("appendix/C_full_results.tex",
     r"In the bank the three worst families (any city destroyed, a count threshold, and the number of government changes) hold 82 of 750 items and 16\% of the pooled excess Brier score. In the tail set the three worst (a wonder completed, a technology discovered, a value threshold) hold 63 of 300 items and 34\% of the pooled bits. In the continuous set the two technology-count families and world captures hold 83 of 300 items and 56\% of the pooled normalised CRPS, so that set's score rests on few families.",
     r"In the bank the three worst families (any city destroyed, the number of government changes, and cities lost) hold 69 of 750 items and 14\% of the pooled excess Brier score. In the tail set the three worst (a government in place at the horizon, a peace in effect, a wonder completed) hold 63 of 300 items and 36\% of the pooled bits. In the continuous set the two technology-count families and world wonders hold 82 of 300 items and 58\% of the pooled normalised CRPS, so that set's score rests on few families."),
    ("appendix/C_full_results.tex",
     r"left out of its sums and $\theta_m$ centred at zero. The residual standard deviation",
     r"left out of its sums and $\theta_m$ centred at zero. Losses are floored at $10^{-4}$ before the logarithm; no cell reaches the floor. The continuous Micropolis and FreeCiv cells use raw nCRPS; all other cells use excess losses. The residual standard deviation"),
]

# ---- 5. Appendix D --------------------------------------------------------------------------------------------------
GROUPING = r"""\subsection{Questions per prompt (FreeCiv)}
\label{app:ablation-grouping}

\begin{greentext}
FreeCiv asked its bank, tail, mirror and continuous questions twice: on 9 September with one question per prompt, and on 20 September in the grouped prompts of \cref{app:worlds-freeciv}, which the paper reports. \Cref{tab:freeciv-grouping} compares the two. Grouping lowered the tail score (mean excess bits 0.556 to 0.332) and raised the bank, mirror and continuous scores (0.129 to 0.141, 0.294 to 0.406 and 0.524 to 0.648), while the correlation of each score with ECI moved by at most 0.06 and the rank correlation of the per-model scores between the two runs was 0.68 to 0.85. The change is one of level rather than of ordering. On the 231 bank questions that also anchor natural-conditional cells, and were therefore asked both ways, the mean forecast was 0.42 when asked alone and 0.33 when asked in a group, against a mean truth of 0.49; grouped answers sit lower on the probability scale, which helps on tail questions and hurts on the rest. A first grouped run on the same day, with the sets of a game mixed in a seeded random order and 50 questions per prompt for both kinds, gave means within 0.013 of the reported run on every set and per-model rank correlations of 0.90 to 0.95; in it o4-mini declined four of its eight continuous prompts of 38 to 48 questions, which is why the reported run keeps the Micropolis limit of 20.
\end{greentext}

\begin{table}[htbp]
\caption{\green{FreeCiv scores under one question per prompt (9 September 2026) and under the grouped prompts of 20 September, mean over the 24 models, with the sign-adjusted Spearman correlation with ECI and the rank correlation of the per-model scores between the two runs. Lower scores are better. Source: \texttt{compare\_runs.py} on the two \texttt{freeciv\_results\_wide.csv} files.}}
\label{tab:freeciv-grouping}
\centering
\small
\setlength{\tabcolsep}{5pt}
\begin{tabular}{@{}lrrrrr@{}}
\toprule
 & \multicolumn{2}{c}{\green{One per prompt}} & \multicolumn{2}{c}{\green{Grouped}} & \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
\green{Set} & \green{Mean} & \green{$\rho$} & \green{Mean} & \green{$\rho$} & \green{Rank corr.} \\
\midrule
\green{Bank, excess Brier} & \green{0.129} & \green{0.12} & \green{0.141} & \green{0.18} & \green{0.72} \\
\green{Tail, excess bits} & \green{0.556} & \green{0.35} & \green{0.332} & \green{0.34} & \green{0.68} \\
\green{Mirror, excess Brier} & \green{0.294} & \green{0.08} & \green{0.406} & \green{0.10} & \green{0.85} \\
\green{Continuous, excess nCRPS} & \green{0.524} & \green{0.61} & \green{0.648} & \green{0.59} & \green{0.70} \\
\bottomrule
\end{tabular}
\end{table}

\subsection{World report and prompt variants (Micropolis)}"""
R += [
    ("appendix/D_ablations.tex", r"\subsection{World report and prompt variants (Micropolis)}", GROUPING),
    ("appendix/D_ablations.tex",
     r"The mean absolute drift over the 24 models is 0.026, from below 0.001 (GPT-5.5) to 0.081 (Kimi K2). Six models drift by less than 0.005 on average. GPT-5.5 and Sonnet 5 return exactly their turn-1 forecast on 96\% and 94\% of cells; Kimi K2, GPT-5 Nano and Gemini 3 Flash Preview return it on 2\%. The drift is mostly downward: its mean with sign is $-0.009$ over models, $-0.064$ for Kimi K2 and $-0.038$ for Haiku 4.5. On the same 100 cells the mean absolute move after a revealed fact is 0.189, about seven times the drift, so the moves in \cref{fig:freeciv-natcond} are not produced by asking again alone. Drift falls with capability: the Spearman correlation between ECI and mean absolute drift is $-0.74$ [$-0.88$, $-0.46$], $p<0.001$, $n=24$, with a percentile bootstrap over models (10,000 samples). Asking again does not change accuracy. The excess Brier score of the no-news answer against $p(Y)$ differs from that of the turn-1 answer by $-0.0001$ on average over models, and by at most $+0.014$ (Gemini 3 Flash Preview). \yellow{Provisional, one question per prompt; the control will be run again in the second run, where turn 2 of the natural conditionals has to be designed again.}",
     (r"The mean absolute drift over the 24 models is {abs_drift:.3f}, from below 0.001 ({lo_model}) to {hi:.3f} ({hi_model}). {n_small_word} models drift by less than 0.005 on average. GPT-5.5 and Sonnet 5 return exactly their turn-1 forecast on {gpt55_unch}\% and {sonnet_unch}\% of cells; {two_pct} return it on 2\%. The drift is mostly downward: its mean with sign is ${signed:+.3f}$ over models, ${kimi:+.3f}$ for Kimi K2 and ${haiku:+.3f}$ for Haiku 4.5. On the same 100 cells the mean absolute move after a revealed fact is {move:.3f}, about {ratio:.0f} times the drift, so the moves in \cref{{fig:freeciv-natcond}} are not produced by asking again alone. Drift falls with capability: the Spearman correlation between ECI and mean absolute drift is ${rho:.2f}$ [${lo_ci:.2f}$, ${hi_ci:.2f}$], $p<0.001$, $n=24$, with a percentile bootstrap over models (10,000 samples). Asking again does not change accuracy. The excess Brier score of the no-news answer against $p(Y)$ differs from that of the turn-1 answer by ${acc_mean:+.4f}$ on average over models, and by at most ${acc_max:+.3f}$ ({acc_max_model}). The control was asked one question per prompt, as the natural-conditional arm was; the rows of DeepSeek V3 and DeepSeek V4 Flash come from their rerun of 20 September (\cref{{app:worlds-freeciv}})."
      ).format(**{k: v for k, v in N.items() if k != "two_pct"}, n_small_word={5: "Five", 6: "Six", 7: "Seven"}.get(N["n_small"], str(N["n_small"])), two_pct=", ".join(N["two_pct"][:-1]) + " and " + N["two_pct"][-1])),
]


def apply_all(check):
    texts = {}
    for f, old, new in R:
        s = texts.get(f) or (P / f).read_text()
        n = s.count(old)
        if n != 1:
            print(f"ABORT: {f}: pattern found {n} times: {old[:90]!r}"); sys.exit(1)
        texts[f] = s.replace(old, new)
    # Appendix B block replacement
    f = "appendix/B_prompts_and_templates.tex"; s = texts.get(f) or (P / f).read_text()
    i, j = s.find(B_OLD_START), s.find(B_OLD_END)
    if i < 0 or j < 0 or s.count(B_OLD_START) != 1 or s.count(B_OLD_END) != 1:
        print("ABORT: appendix B anchors"); sys.exit(1)
    texts[f] = s[:i] + B_NEW + s[j + len(B_OLD_END):]
    # Appendix D no-news table rows
    f = "appendix/D_ablations.tex"; s = texts[f]
    m = re.search(r"(\\label\{tab:nonews\}.*?\\midrule\n)(.*?)(\\bottomrule)", s, re.S)
    if not m: print("ABORT: no-news table"); sys.exit(1)
    assert m.group(2).count("\\\\") == 24, m.group(2).count("\\\\")
    texts[f] = s[:m.start(2)] + NONEWS_ROWS + "\n" + s[m.end(2):]
    for f, s in texts.items():
        print(("would write" if check else "wrote"), f)
        if not check: (P / f).write_text(s)


apply_all(a.check)
