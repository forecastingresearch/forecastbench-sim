# WhatIf-Bench exploration notes — 2026-07-12

Synthesis of a cross-branch survey (all 19 branches + 3 stashes) plus a literature/cost
review, answering three questions: (a) a training pipeline to make a small LLM better at
forecasting on forecastbench-sim, (b) the state of causal forecasting here and the
"golden question" experiment the simulator uniquely enables, (c) why correlation with
real ForecastBench is only moderate and what to fix.

---

## 0. Where things stand (one paragraph per thread)

- **Benchmark (main):** 9,426 questions, 26 templates, 14 seeds, horizons H0–H7. Binary →
  Brier/ECE, continuous → quantile CRPS/MAE. Turn-60 ASCII world report (~5–6k tokens,
  numeric time series sampled every 5 turns), batched 20-questions-per-prompt elicitation
  of probabilities between `<<<PROBABILITIES>>>` delimiters.
- **ForecastBench correlation (icml-workshop-sprint paper):** pooled H1–H7 Spearman
  ρ = **+0.43** (p=.018, N=30 models) vs FB Dataset Brier; **H1 alone ρ = +0.63**; H4/H5
  not significant; ρ = +0.48 vs Epoch Capabilities Index.
- **Monte-Carlo re-rollout (forecast-uplift-study):** working RNG-reseed forking (Method B,
  pure savegame edit — Python port of FreeCiv's Mitchell-Moore RNG incl. 10k-step warm-up).
  ~70 s per 30-turn rollout; one rollout resolves *all* (~55) questions from a snapshot.
  Dense study (220 q × 14 models): ranking robust to dense labels (ρ=0.974), but **18% of
  hard 0/1 labels sit on the minority side** of the MC distribution and **~39% of questions
  are genuinely stochastic** (p_mc ∈ (0.1, 0.9)).
- **RL/fine-tuning (rlvr-dense-signal):** Qwen3-8B, LoRA SFT + GRPO (−Brier reward), ~$24.
  SFT learned in-distribution (Brier 0.230→0.147) but **negative transfer** to real
  ForecastBench; RL reward flat because the −1.0 parse-failure penalty dwarfed the
  0.01–0.10 Brier signal; template-holdout generalization failed.
- **Conditional forecasting:** full fork pipeline (gold / gold_add / government / tech
  interventions; 5 framing variants incl. post-intervention world reports). Every measured
  result is negative: **null-conditional framing alone worsens Opus 4.5 Brier +36%**
  (0.178→0.242); conditional gaps of +0.07–0.09 Brier for base/LoRA Qwen-8B across
  republic/gold500/mapmaking. No evidence yet of *correct* causal updating.
- **Downstream value (forecast-uplift-study Track A):** giving a weak Gemini Flash agent
  game-true probabilities *reduced* end-gold (−7.4, 95% CI [−13.1,−1.9]) vs control;
  scrambled ≈ true. Track B (government lever) was a structural null (AI reverts forced
  government within ~2 turns; VOI ≈ 0).
- **Gaps in intervention coverage:** no way to force **wars** or proper revolutions;
  government edit is cosmetic; tech grant is a fragile bit-flip; unit/city edit packets
  exist (`event_injector.py`) but aren't wired into the fork pipeline.

---

## (a) Pipeline: make a small LLM better at forecastbench-sim

### Literature anchors
- **OpenForecaster** (arXiv:2512.25070, Qwen3-8B): GRPO with **accuracy + Brier** reward
  (Brier-alone collapses exploration → "Unknown" at near-zero confidence), SFT→RL ordering
  best, retrieval +9–18%. Final RL run ≈ 1,000 H100-hr (~$2.5k); SFT 40 H100-hr (~$100);
  data generation ~$3k (which the simulator gives us for free).
- **Turtel et al.** (arXiv:2505.17989): GRPO σ-normalization **destroys calibration
  signal** — use mean-only advantages or ReMax; ungated RL → degenerate confidence
  (39% of predictions in extreme buckets); guardrail penalties needed.
- **DAR** (arXiv:2605.20740): set-level **CRPS reward** over K rollouts with leave-one-out
  credit — the closest published mechanism to "reward = CRPS vs MC distribution."
- **Position paper** (arXiv:2507.19477) explicitly calls for simulated environments with
  MC ground truth for dense rewards — i.e., our setting is a recognized open niche:
  **nobody has published RL against simulator MC-distributional targets yet.**

### Why our own rlvr-dense-signal run failed, and what changes
1. **Hard 0/1 targets** → 18% label noise + coin-flip variance. → Train against **p_mc**
   (dense). Reward = −(p̂ − p_mc)² has *zero outcome noise*; per-sample signal is orders
   of magnitude cleaner than real-world RLVR forecasting.
2. **Parse penalty dominated reward.** → Guided decoding / structured output so format
   reward is a solved constraint, not a gradient direction (session 2 already showed
   guided decoding fixes parsing).
3. **Tiny data (641 q).** → Scale via sim: questions are free; MC labels cost only
   sim-CPU (below).
4. **No SFT-from-strong-traces stage.** → Distill filtered reasoning traces first
   (Halawi-style outcome filtering: keep traces whose forecast lands near p_mc).

### Proposed pipeline (v1)
1. **Data:** ~500 snapshots (≈40 seeds × ~12 checkpoint turns) × N=100 rollouts at H1
   (and a subset at H2/H3) → ~25–30k questions with p_mc at ±0.05 precision. Filter to
   0.05 < p_mc < 0.95 for training; stratify by template; hold out entire seeds AND
   entire templates AND one horizon.
2. **SFT stage:** generate ~10k reasoning traces from a frontier model on train questions;
   keep traces whose final probability is within ε of p_mc; LoRA-SFT Qwen3-4B or 8B.
3. **RL stage:** GRPO variant — mean-only advantage (no σ-division) or ReMax; reward =
   −(p̂ − p_mc)² for binary + pinball/CRPS vs the MC quantiles for continuous; K=8
   rollouts; KL to SFT; format enforced by guided decoding.
4. **Eval:** (i) held-out seeds/templates/horizons in-sim; (ii) real ForecastBench
   (the transfer question — our prior run says don't expect it; a *clean* null with 40×
   the data is itself the publishable result); (iii) conditional sets (does dense-signal
   training help or hurt conditional updating?).

### Cost estimate
| Item | Amount | Cost |
|---|---|---|
| MC labeling: 500 snapshots × 100 rollouts × ~70 s | ~970 container-hours (CPU, embarrassingly parallel across ports/boxes) | ~$50–200 cloud CPU, or free-but-slow locally |
| SFT trace generation (10k traces, frontier API) | — | $100–300 |
| LoRA SFT, 4–8B | ~40 H100-hr | ~$100 |
| GRPO, 4–8B, ~20k q × 3 epochs | ~60–150 H100-hr (scaled from OpenForecaster's 0.004 H100-hr/question-pass) | $150–400 @ $2.50/hr |
| Eval sweeps (sim + FB + conditional) | few H100-hr + API | $50–150 |
| **Total v1 experiment** | | **≈ $500–1,200** |
| Full study with ablations (reward shape × data scale × model size) | | $3–6k |

---

## (b) Causal forecasting: the golden question

### What's been done (short version)
Fork-and-compare infra works end to end (checkpoint → savegame edit → re-simulate →
paired resolution). Interventions: gold set/add (solid), tech grant (fragile bit-flip),
government (cosmetic — AI reverts in ~2 turns). Five framings per question set
(baseline / conditional / conditional-no / given-that / post-intervention-report).
All measured results so far: conditional framing *hurts* Brier; no demonstration of
correct causal updating. Wars — the motivating example from the 7/9 meeting — cannot
currently be imposed.

### The real-world golden question we can never answer
> **"Was the model's conditional forecast causally correct?"**
> i.e. does the model's Δ = P̂(Y|X) − P̂(Y) match the *true interventional* effect
> P(Y|do(X)) − P(Y)?

Two reasons it's unanswerable in the real world:
1. **No counterfactual arm.** Only one branch of history is realized. If the war doesn't
   happen, the conditional forecast resolves N/A; if it does, you observe one sample of
   one branch. You can never estimate the *distribution shift* an intervention causes —
   real conditional-market scoring only ever scores the realized branch.
2. **Conditioning ≠ intervening.** Even when X happens naturally, worlds-where-X-happens
   are a selected subpopulation (wars break out in worlds that were already going badly).
   Real resolved conditional questions measure P(Y | X observed), confounded, not
   P(Y | do(X)). The VP's "what if Gemini is #1 on only 20% of benchmarks" is a do()
   question; any real-world data answers only the observational one.

### The FreeCiv equivalent (the experiment to run)
For a snapshot at turn T and target Y at T+30, estimate **three** ground-truth quantities
via MC rollouts:

1. **P(Y)** — N rollouts, no intervention (exists today).
2. **P(Y | do(X))** — N rollouts from the X-edited savegame (exists for gold/tech;
   **needs a war intervention** — edit the diplomacy state block in the savegame, the
   main missing piece; worth ~a week of savegame-format spelunking given the existing
   `SavegameModifier` scaffolding).
3. **P(Y | X observed)** — filter the no-intervention rollouts to those where X happened
   naturally (wars do break out on their own; the event parser can detect them). Possible
   today with zero new infrastructure.

Then elicit from the model p̂(Y), p̂(Y|X-as-guarantee), p̂(Y|X-observed) and score:

- **Primary metric: Δ-error** = (p̂(Y|do X) − p̂(Y)) − (P(Y|do X) − P(Y)). Paired scoring
  cancels both the model's baseline miscalibration *and* the documented framing penalty —
  this is the correction to the current conditional evals, which confound framing overhead
  with causal skill.
- **Placebo row:** null interventions with Δ*≈0 (we already know models fail this —
  keep it as the floor).
- **The uniquely-simulatable test:** does the model give *different* answers for
  "given that a war breaks out" (evidential, ground truth #3) vs "given a war is imposed"
  (interventional, #2)? The gap between #2 and #3 is the confounding bias — a quantity
  that is literally unmeasurable in the real world and the strongest legibility story for
  the proposal: *this is the capability OpenDC-style causal-graph simulators are meant to
  test, demonstrated first here.*

Cost: each (snapshot, intervention) cell needs ~2N rollouts ≈ 2×100×70 s ≈ 4 container-hours;
a 50-snapshot × 3-intervention study ≈ 600 container-hours ≈ $50–150 of CPU.

---

## (c) Why ForecastBench correlation is only ρ≈0.43, and fixes

Diagnosis, ordered by estimated contribution:

1. **Label noise from single-rollout 0/1 resolution.** 18% of labels resolve on the MC
   minority side; ~39% of questions are genuinely stochastic. This puts an irreducible
   noise floor under between-model Brier differences and attenuates rank correlation.
   *Fix:* score against p_mc (dense resolution) — infrastructure exists; ranking already
   validated (ρ=0.974) at 4 seeds; extend MC resolution to the full benchmark.
2. **Near-coin-flip question construction.** Thresholds deliberately calibrated to ~40%
   base rate + comparative questions decided by 1-unit margins (techs 26 vs 27) 30+ turns
   out. *Fix:* filter/weight questions by MC discriminability (drop p_mc≈0.5 noise AND
   p_mc≈0/1 triviality; drop near-tie comparatives by MC margin).
3. **Skill mismatch, worst at mid horizons.** The sim measures numeric trend extrapolation
   from a 5-turn-sampled table; ForecastBench measures world-knowledge + base-rate
   judgment. Correlation is best exactly where judgment-like signal is highest (H1 ρ=0.63,
   H0 0.55) and dies where sim chaos dominates (H4/H5 n.s.). *Fix:* make H1 (+ H0
   comprehension gate) the headline metric; treat H4+ as a separate "long-horizon chaos"
   track, not part of the FB-comparable score.
4. **Elicitation artifacts.** Batched 20-per-prompt answers → ordering misalignment risk
   (one dropped line shifts every subsequent answer); parse failures silently *dropped*
   from Brier rather than scored 0.5 (models are ranked on different question subsets;
   FB penalizes non-response); clamp-to-[0,1] masks mis-parses; first-decimal fallback can
   grab a turn number. *Fix:* structured output / one-question calls (or echo question
   IDs and validate alignment), impute 0.5 for failures, reject rather than clamp.
5. **Brier conflates calibration with discrimination.** Documented case of a model
   matching the target mean yet scoring worse than a constant predictor. *Fix:* report a
   Murphy decomposition (reliability / resolution) and correlate the components with FB —
   discrimination is likely what tracks capability.
6. **Statistical width.** ρ=0.43 over N=30 models has a wide CI; H1-only already reaches
   0.63, essentially matching ForecastBench's own internal capability correlation (0.68).
   Cheap win: more models × more H1 questions rather than more horizons.

Expected outcome: dense resolution + discriminability filtering + H1 focus + elicitation
fixes should push the pooled correlation toward the H1 ceiling (~0.6+), and it's an
afternoon-scale rescoring exercise on existing eval outputs for items 1, 2, 5, 6 (no new
model calls needed — labels change, predictions don't).

---

## Sources
- In-repo: `docs/monte_carlo_{rollouts,dense_resolution}.md`, `docs/forecast_uplift_{study,results}.md`
  (branch `forecast-uplift-study`); `runpod/WRITEUP.md`, `SESSION_REPORT*.md` (branch
  `rlvr-dense-signal`); `paper/forecastbench_sim.tex` (branch `icml-workshop-sprint`);
  `docs/experiments/null_conditional_framing.md`, `docs/notes/conditional_forecasting.md`,
  `data/evaluations/heldout_8b_analysis/summary.json` (main).
- External: OpenForecaster arXiv:2512.25070; Halawi et al. arXiv:2402.18563; Turtel et al.
  arXiv:2505.17989; Future-as-Label arXiv:2601.06336; FutureWorld arXiv:2604.26733;
  DAR arXiv:2605.20740; position paper arXiv:2507.19477; Mantic×Thinking Machines blog.
