# Small-LLM forecasting uplift v2 — dense-vs-hard RLVR A/B (design)

Status: designed, awaiting approval to spend. 2026-07-12.

## Question

Take a small instruct LLM that forecasts poorly on FB-Sim binary questions and run a
scaled-down OpenForecaster gauntlet on it. Two claims to test, in order:

1. **Does it improve at all?** (RLVR on FB-Sim questions → better held-out FB-Sim Brier)
2. **Does the dense signal make it improve faster/better?** RL reward = −(p̂ − p_mc)²
   (Monte-Carlo dense target) vs reward = −(p̂ − y)² (hard 0/1 label), everything else
   byte-identical.

The forensic review of `rlvr-dense-signal` (see runpod/ docs on that branch) established
that **the dense-vs-hard A/B has never actually been run** — every prior arm used p_mc.
And `data/training/{train,val}.jsonl` already carries both `p_mc` and `baseline_01` per
record, so the A/B needs no new rollouts to start.

## Design (changes from v1, each grounded in a v1 failure)

| # | v2 choice | v1 failure it fixes |
|---|---|---|
| 1 | Primary A/B: **RL-from-base**, dense vs hard reward, only the target differs | Never tested; SFT-init was a harmful basin (E5 beat E3) |
| 2 | **Guided decoding during training AND eval** (regex-constrained `PROBABILITY:` line); no parse penalty in the reward at all | −1.0 (even −0.25) parse penalty dominated the 0.01–0.10 calibration signal; reward stayed flat 300 steps |
| 3 | **No 4·p·(1−p) reward multiplier**; emphasize uncertain questions via sampling if needed | The multiplier shrank the calibration signal on exactly the interior questions it meant to emphasize |
| 4 | Identical decoding regime (thinking off, same max tokens) in train and eval | Three separate v1 regressions traced to format seams |
| 5 | G=8 rollouts/group, ≥2 RNG seeds per arm, report variance | G=2 gave ~30% zero-advantage groups; all v1 conclusions single-run |
| 6 | **Gate on template-holdout** (reuse `data/training_E4` split) before any transfer claim | v1's "win" was memorization; E4 lost to constant on every held-out template |
| 7 | Fixed eval subsets + baselines computed once up front; parse rate reported separately | Per-condition parsed-subset baselines made v1 arms incomparable |
| 8 | Val set upgraded with higher-N p_mc (N=40 from the causal-pilot rollouts, seeds 0–3) | 57-example val with seed-level distribution shift (wonder_completed train 0.13 vs val 0.83) |
| 9 | Primary endpoint is **in-sim held-out** (seeds + templates), FB transfer secondary | v1 chased FB transfer with a target distribution (mean 0.50) mismatched to FB (0.19) |
| 10 | Optional later arm: SFT with *reasoning traces* filtered by closeness to p_mc | v1 SFT taught "emit a number, skip thinking" |

## Model choice

Need a small instruct model that (i) is weak on FB-Sim binary, (ii) has open weights for
TRL training, (iii) is servable cheaply. Candidates: **Qwen3-4B-Instruct** (preferred:
smaller = more headroom + cheaper), Qwen3-8B (comparable to v1), Llama-3.1-8B-Instruct.
Step 0 is a ~$5 baseline eval of these on the 220 MC-labeled H1 questions (dense + hard
scoring) to confirm "does poorly" and pick the weakest reasonable one.

## Execution plan and costs

| Step | What | Where | Cost |
|---|---|---|---|
| 0 | Baseline eval: 3 candidate models × 220 MC questions (+ existing 14-model rankings for context) | Fireworks serverless (needs FIREWORKS_API_KEY) | ~$5 |
| 1 | Data prep: rebuild train/val jsonl with both targets, no weight expansion; template-holdout split | local | $0 |
| 2 | RL A/B: 2 arms × 2 seeds × ~300 steps GRPO (TRL, guided decoding, G=8) | RunPod H200 ($4.39/hr, ~3–4h/run) | ~$105–140 |
| 3 | Eval sweeps: base + 4 checkpoints/run on val + template-holdout + FB-401 | same pod, amortized | ~$10–20 |
| 4 | (Conditional on 2–3 results) SFT-with-reasoning arms, higher-N p_mc retrain | TBD | ~$50–100 |
| | **Total for steps 0–3** | | **~$120–165** |

Alternative to RunPod for step 2: Fireworks RFT (user has credits) — supported but gives
less control over GRPO internals (advantage normalization is exactly what v1 got wrong),
so RunPod + TRL is recommended; Fireworks credits are better spent on step-0/serving.

## Endpoints

- Primary: held-out-seed Brier-vs-p_mc and template-holdout Brier-vs-p_mc, dense arm vs
  hard arm vs base, with bootstrap CIs over questions and across RNG seeds.
- Learning-curve comparison (checkpoints at 25/50/75/100% of steps) → "improves faster?"
- Secondary: ForecastBench-401 Brier + parse rate + ECE (transfer; expectation from v1
  is null — a clean null at 2 seeds is still informative).
- Guardrail metrics: prediction-distribution mean/entropy per arm (degenerate-confidence
  check per Turtel et al.).
