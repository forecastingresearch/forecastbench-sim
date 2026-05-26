# ForecastBench-Sim Workshop Outline

Working title: **ForecastBench-Sim: A Simulated-World Forecasting Benchmark**

Draft status: planning document for the ICML 2026 Forecasting Workshop
submission. The paper should read primarily as a benchmark-design paper,
not as the full results paper. Treat the existing model and human analyses
as validation and existence proofs.

## Core Positioning

The central claim is that simulated worlds complement real-world forecasting
benchmarks by giving forecasters realistic, information-rich world states
while preserving experimental control. ForecastBench-Sim can generate hidden
future outcomes, resolve forecasts immediately by continuing the simulation,
and create matched causal or conditional variants by mutating the savegame
before rollout.

Do not make "instant resolution" the headline contribution. Use it as an
operational advantage. The research contributions to emphasize are:

1. **Controlled forecasting benchmark design.** The benchmark has dynamic
   multi-agent worlds, hidden future states, fixed reports, generated
   questions, and automatic scoring.
2. **Distributional and binary evaluation under matched conditions.** The
   same world reports support binary Brier scoring and continuous CRPS
   scoring over quantile forecasts.
3. **Conditional and causal forecast evaluation.** Simulated worlds allow
   intervention worlds, e.g. change a government type or treasury value,
   continue both rollouts, and ask whether models forecast the induced
   effect.
4. **Tail-risk evaluation with immediate ground truth.** Rare or disruptive
   outcomes can be sampled by running many worlds, instead of waiting for rare
   real-world events.
5. **Human baseline feasibility.** The anonymized pilot shows that the same
   reports can be used with human participants, while H0 checks verify that
   models can read the reports.

## Figure Decisions

Main-paper figures should be few and explanatory. Four pages is too short
for a dense benchmark paper plus a full leaderboard plus multiple appendix-
style diagnostics.

| Item | Placement | Decision | Rationale | Next edit |
|---|---:|---|---|---|
| Fig. 1 benchmark schematic | Main | Ready, with optional caption polish | This is the benchmark-design anchor. It should teach the reader how the benchmark works. | Current output: `fig1_benchmark_schematic.png`; it now uses ForecastBench-Sim wording and shows controlled variants. |
| Fig. 2 compact validation | Main | Ready | The full leaderboard is too tall for 4 pages and makes the paper feel results-first. | Current output: `fig2_model_validation_compact.png`; full `fig2_leaderboard.png` stays appendix-only. |
| Fig. 3 horizon curve | Main | Keep, with careful framing | Shows the benchmark is not trivial and that horizon changes difficulty. This is a validation result, not the main contribution. | Caption should say "example validation slice" and avoid over-claiming inverse scaling. |
| Fig. 4 intervention gap | Appendix | Keep | This is important for Nick's causal/conditional framing, but the current data are an existence proof rather than the paper's core empirical claim. | Caption should explain savegame mutation -> rollout -> paired scoring. |
| Fig. 5 binary vs continuous | Appendix or cut | Usually appendix | Useful if there is room to show metric-conditional behavior, but it belongs in the later results paper more than this workshop paper. | Include only if referenced in one sentence; otherwise omit from submitted PDF. |
| Fig. A1 human vs models | Appendix | Keep | Human baseline is useful but currently pilot-sized and aggregate-matched, not question-matched to every model run. | Caption must state this is a pilot aggregate comparison, not a definitive human-vs-model ranking. |
| Fig. A2 template heatmap | Appendix | Keep if space | Good for showing benchmark breadth and heterogeneity. | Use as appendix support for "not one monolithic task." |
| Fig. A3 reliability | Appendix | Keep if appendix is not crowded | The revised ECE/bias bar chart is clearer than the old reliability curves. | Use as optional calibration diagnostic. |
| Table 1 benchmark composition | Main | Ready | This can replace text and reduce word count. | Current output: `table1_benchmark_composition.tex`. |
| Table A1 H0 comprehension | Appendix | Keep | H0 establishes that failures are not report-reading failures. | Already generated as TeX. |
| Tail-risk figure from NeurIPS draft | Appendix only, optional | Add only if it does not create salami-slicing confusion | Tail risk is a benchmark affordance. The detailed inverse-scaling result belongs to the NeurIPS paper under review. | If included, use as "example downstream analysis enabled by ForecastBench-Sim" and cite the under-review manuscript anonymously if needed. |

Recommended main figure set for a 4-page workshop paper:

- **Fig. 1:** benchmark schematic, one-column or full-width depending
  readability.
- **Table 1:** compact benchmark composition table.
- **Fig. 2:** compact model validation leaderboard.
- **Fig. 3:** horizon/difficulty validation curve.

Everything else should be appendix.

## Workshop-Length Comparison and Feasibility

Comparable 4-page ML workshop papers usually have one central schematic,
one or two compact empirical figures, and most ablations in the appendix.
They do not have enough space for a full benchmark introduction, complete
related work, and a full empirical study. The feasible target here is about
2,500--2,900 main-text words including captions. With three compact figures
and one small table, the text should aim for about 2,400--2,600 words.

Proposed word budget:

| Component | Target words | Notes |
|---|---:|---|
| Abstract | 150--175 | State problem, benchmark, affordances, validation. |
| Introduction | 500--600 | Motivation and contributions. |
| Benchmark design | 700--850 | Simulation, reports, questions, scoring, H0. |
| Controlled evaluation affordances | 350--450 | Conditional/causal and tail-risk mechanics. |
| Validation results | 450--600 | Compact model results, horizon behavior, human pilot. |
| Limitations and conclusion | 250--350 | External validity, human-pilot limits, release posture. |
| Main captions | 250--350 | Keep captions descriptive but not paper-within-paper. |

This sums to roughly 2,650--3,375 including captions. The draft should start
near the lower end. If it exceeds 4 pages, first shorten related work and
results, not the benchmark mechanics.

## Page and Figure Placement

Use the official ICML 2026 two-column template, but enforce the workshop's
4-page main-body limit manually.

Page 1:
- Title and abstract.
- Intro paragraphs 1--3.
- If space allows, start "Benchmark Design" with the design-goals paragraph.

Page 2:
- Place Fig. 1 near the top, preferably one-column if readable; use
  full-width only if the labels become too small.
- Benchmark construction paragraphs: world generation, world reports,
  question families.
- Place Table 1 after the question-family paragraph or in the margin of the
  methods section.

Page 3:
- Scoring paragraph and controlled-evaluation affordances.
- Explain H0 checks in one short paragraph and point to Table A1.
- Place compact Fig. 2 near the bottom or top of page 3.

Page 4:
- Place Fig. 3 near the top.
- Validation results paragraph.
- Human pilot paragraph.
- Limitations and conclusion.

Appendix:
- A. Full benchmark construction details.
- B. Full leaderboard and model list.
- C. H0 comprehension table.
- D. Human baseline details.
- E. Conditional/causal intervention mechanics and Fig. 4.
- F. Optional tail-risk/calibration examples.

## Paragraph-by-Paragraph Draft Plan

### Abstract

One paragraph, 150--175 words. Say that real-world forecasting benchmarks
are valuable but limited by delayed resolution, sparse tail events, and
limited control over counterfactuals. Introduce ForecastBench-Sim as a
simulated-world forecasting benchmark built from Freeciv rollouts. Mention
that models and humans receive fixed world reports and answer binary and
distributional questions whose outcomes are hidden until the simulation is
continued. State the three key affordances: immediate resolution, paired
conditional/causal interventions, and dense tail-risk sampling. End with
validation: H0 checks, multi-model benchmark runs, and a human pilot.

Possible abstract draft:

> Forecasting benchmarks for general-purpose AI systems usually inherit the
> constraints of the real world: outcomes resolve slowly, tail events are
> rare, and counterfactual questions are difficult to score. We introduce
> ForecastBench-Sim, a simulated-world forecasting benchmark built from
> Freeciv game rollouts. Forecasters receive a turn-level world report and
> answer binary and distributional questions about hidden future states; the
> benchmark then continues the simulation and scores forecasts with Brier
> score and CRPS. Because the world is simulated, the same setup can also
> generate H0 comprehension checks, paired intervention worlds for
> conditional or causal questions, and many resolved examples of rare or
> disruptive outcomes. We describe the benchmark pipeline, question families,
> scoring protocol, and release artifacts, then report validation slices from
> model evaluations and an anonymized human pilot. ForecastBench-Sim is meant
> to complement real-world forecasting benchmarks by providing controlled,
> immediately resolvable tasks for studying probabilistic reasoning under
> dynamic world states.

### Introduction

Paragraph 1: Motivation. Forecasting is central to claims about general
intelligence, but current AI forecasting evaluations are constrained by
real-world resolution. Binary benchmarks can score many questions only after
time passes, continuous distributional questions are harder to validate, and
rare tail events are under-sampled.

Paragraph 2: Why simulation helps. Simulated worlds preserve forecasting's
core structure: partial information, evolving state, interacting agents,
and stochastic outcomes. Unlike toy tasks, Freeciv-style worlds contain
economies, cities, diplomacy, technology, wars, and path-dependent dynamics.
Unlike real-world tasks, the benchmark can reveal the future by rolling
forward and can create controlled variants by changing the savegame.

Paragraph 3: Contributions. Use a compact list if space allows. Contributions:
benchmark pipeline; binary and continuous question generation; H0
comprehension checks; conditional/causal intervention mechanism; validation
against model runs and a human pilot. Keep this paragraph concrete.

### Benchmark Design

Paragraph 4: World generation. Explain that the benchmark starts from
Freeciv/civrealm simulated game states. A world is observed at a fixed
snapshot turn, while later turns are withheld from forecasters. Avoid
claiming that the simulated world is a perfect substitute for the real
world; say it is a controlled complement.

Paragraph 5: World report. Describe the report as the only information given
to forecasters: state tables, histories, maps/reports, and other structured
evidence. Introduce Fig. 1 here. Mention that humans should receive the same
tabular information as LLMs where possible.

Paragraph 6: Question families. Explain binary questions and continuous
questions. Use one example pair: binary "will treasury exceed X?" and
continuous "what will treasury be?" State horizons H1--H7 and H0 checks.
Include Table 1 around here.

Paragraph 7: Scoring. Define Brier for binary forecasts and normalized CRPS
for quantile forecasts in prose. Put formulas in appendix unless needed.
State that H0 is not a forecasting task; it checks report comprehension.

### Controlled Evaluation Affordances

Paragraph 8: Immediate resolution. State that the benchmark resolves by
continuing rollouts, which makes dense evaluation possible without waiting
for calendar time. Do not over-sell this as the main novelty.

Paragraph 9: Conditional and causal questions. Explain the mechanism clearly:
copy a savegame, mutate a state variable or condition, continue both the
baseline and intervention worlds, and compare outcomes. This enables scored
questions about conditional probabilities and causal effects because both
branches have ground truth. Point to appendix Fig. 4.

Paragraph 10: Tail risk. Explain that simulated worlds can produce many
disruptive outcomes, allowing evaluation of tail calibration and rare-event
performance. If borrowing a result from the NeurIPS submission, present it
only as an example downstream analysis and cite the anonymous under-review
paper or put the details in appendix.

### Validation Results

Paragraph 11: Model benchmark runs. Report model/sample coverage from the
current artifacts: 30 binary models over the main run, 31 continuous models
with one archived continuous splice, H0 binary on 562 questions, H0
continuous on 165 questions for the curated nine. Use compact Fig. 2. State
that the purpose is validation of the benchmark surface, not exhaustive model
ranking.

Paragraph 12: Horizon behavior. Use Fig. 3 to show that performance varies
with forecast horizon. The interpretation should be modest: horizon curves
show the benchmark captures increasing difficulty and reveals different
behavior across task types. Do not center the paper on inverse scaling.

Paragraph 13: Human pilot. Summarize the anonymized pilot as evidence that
humans can use the reports and produce forecasts on the same task family.
State limitations: small pilot, aggregate comparison, not a fully powered
human-vs-model study. Point to appendix details.

### Limitations and Conclusion

Paragraph 14: Limitations. Name the important ones directly: Freeciv worlds
are not the real world; report formatting can matter; the human pilot is
small; conditional/intervention coverage is currently a demonstration; some
model runs are not perfectly matched across every artifact.

Paragraph 15: Conclusion. One short paragraph. ForecastBench-Sim provides a
controlled, immediately resolvable benchmark for studying forecasting,
distributional calibration, conditional probabilities, causal effects, and
tail risk in dynamic worlds. It is designed as a reusable benchmark substrate
for later empirical papers.

## Claims to Avoid

- Do not claim the human pilot definitively ranks humans against models.
- Do not claim all model conditions are perfectly equivalent unless the
  matched question set and prompting setup are explicitly identical.
- Do not foreground "LLMs playing Civ"; the task is forecasting over reports
  from simulated worlds.
- Do not claim causal effects are solved by the current model results. The
  contribution is that the benchmark can create and score paired causal or
  conditional questions.
- Do not include author-revealing references to local repo paths or private
  Overleaf/GDrive links in the submission.

## Concrete To-Do List for Commit 4

1. Edit Fig. 1 text from "CivBench" to "ForecastBench-Sim" and add one visual
   cue or caption phrase for intervention branches and tail-risk sampling.
   Status: done in `fig1_benchmark_schematic.png`.
2. Create compact Fig. 2 for the main paper. Keep the current full
   `fig2_leaderboard.png` as an appendix artifact.
   Status: done as `fig2_model_validation_compact.png`.
3. Review Fig. 3 caption and labels so it reads as benchmark validation.
4. Convert Table 1 to LaTeX or make a compact main-text table.
   Status: done as `table1_benchmark_composition.tex`.
5. Decide whether to include an appendix tail-risk figure from the NeurIPS
   draft. If yes, cite it as an anonymous under-review companion result and
   frame it as an example analysis enabled by the benchmark.
6. Start the LaTeX draft from `paper/icml2026_template/example_paper.tex`,
   enforcing the workshop 4-page limit manually.
