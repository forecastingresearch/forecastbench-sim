# Big experiments — dependency map & execution state (2026-07-19)

Source of intent: `big-experiments-plan.txt`. Budget: ~$490 Fireworks, ~$150 RunPod
(verify RunPod balance before any pod — account hit −$0.40 on 7/13).
Local: 10 cores / 16 GB, ~84 GB disk free.

## Dependency graph

```
[S0] Stage 0: ~60 new worlds, dense p_mc + hard labels + causal savegames
     (local CPU, ~40h, $0)  ──────────────┬────────────────────────────┐
                                          v                            v
[A1] Eval existing fbsim-rft-dense-cal   [A2] 3-way RFT experiment    [B2] single-turn causal
     vs base (Fireworks, $5-15)  ──gate──> (Fireworks free RFT)        LLM eval ($10-30)
     GATE A1: train-reward must improve       |                          ^
                                              v                          |
                                         [A3] RunPod GRPO replication  [B1b] seed2 scale-up
                                              (gated, ≤$120)            150-200/arm (CPU, $0)
                                              |                          |
                                              v                          v
                                         [B4] RL-transfer matrix <──  [B3] multi-turn causal
[C] evaluator fixes (local, $0) — prerequisite for interpreting FB transfer; filler work
```

## Kill/pause conditions

- **A1 fails on training reward** → RL pipeline broken → pause Goal A entirely, debug before A2.
- **S0 world gen fails integrity checks** → everything downstream pauses (both goals need it).
- **B1b shows gaps were noise** → causal benchmark demoted to Δ_do-only; B2 proceeds on
  Δ_do questions (still valid), confounding-gap claims dropped.
- **A2 shows no signal at 2 seeds** → skip A3 (save $120), report clean null.
- Fireworks RFT no longer free / starts charging → recheck budget before A2 (4 jobs).

## What already exists (do not redo)

- Step-0 baselines (qwen3-4b weakest reasonable; constant-0.5 ≈ 0.145 vs p_mc on 698q).
- Causal pilot N=40×4 seeds: Δ_do solid (19 vs 8.8 null, top z=3.8 Bonferroni-safe);
  confounding gap suggestive (13/220 sig vs 11 chance) — needs B1b.
- Fireworks RFT calibration job nnwolhuh COMPLETE → model `fbsim-rft-dense-cal` READY.
  1 epoch, dense arm, G=8, temp 1.0, maxOut 8192, ctx 24576, LoRA r8.
- Hardened RunPod runbook + TTL'd create_pod.sh (5-layer cost controls).
- Training data with both p_mc and baseline_01 targets: `data/training/{train,val}.jsonl`.

## Parallelization plan

- CPU (10 cores): S0 world gen workers + B1b seed2 rollouts share the box.
  Rollouts are ~single-core; run ~3 S0 workers + 2-3 B1b workers, watch RAM (16 GB).
- Network/Fireworks: A1 eval runs concurrently with CPU work.
- [C] evaluator fixes are pure local code — do while babysitting.

## Gate A1 result (2026-07-19, ~$8)

Paired eval (869 rows/model, temp 1.0 training-matched, 100% parse both models):
- Train: reward diff +0.0079 [-0.0072, +0.0227] (n.s.); Brier-vs-hard −0.0183
  [−0.0354, −0.0010] (significant). Val (seed3): −0.0137 reward (n.s., wrong direction).
- No confidence shrinkage — RFT model is MORE confident (mean|p−0.5| .335 vs .311).
- Verdict: **pipeline healthy, signal weak-positive on train, none on val — exactly
  what 1 epoch × 641 rows should look like. PROCEED to A2** with 6 epochs + Stage-0
  data (~2,200 train rows). Wall-clock calibration: 1 epoch × 641 rows × G=8 ≈ 3h20m
  free RFT. Both eval deployments deleted after use.

## Gate B1b result (2026-07-19, $0 local)

seed2 scaled to N=180/arm (baseline + do(The Republic→Benin)), 65 natural-X rollouts:
- **Confounding gap CERTIFIED**: 3/54 questions survive |z|>=3.29 (~Bonferroni;
  chance 0.05): q0045 gap −0.19 (z=−5.44), q0020 +0.27 (z=+4.55), q0000 +0.27
  (z=+3.97). Textbook selection-vs-intervention: worlds that discover The Republic
  naturally are science-strong (p_obs≈p0) while imposing it redirects research
  (p_do far off).
- Δ_do: 9/54 at |z|>=1.96 (chance 2.7), 5 at |z|>=3.29.
- Data: tmp/causal/seed2/{baseline,do-tech80p0} manifests, stats in
  tmp/causal/gap_stats_seed2.json (scripts/uplift_v2/causal_gap_stats.py).

## Goal B Stage 2 result — baseline models (2026-07-19, ~$20)

220 (seed,Y) pairs × 5 framings (base/do/obs/placebo_do/placebo_obs) × 2 samples,
temp 1.0, OF prompt. Parse: q4b 1/2200 failed, oss120 ~2%.

| metric | qwen3-4b | gpt-oss-120b | ideal |
|---|---|---|---|
| corr(Δ̂_do, Δ*_do) | −0.02 | +0.15 | high |
| sign acc, |Δ*|>0.15 (n=18) | 0.39 | **0.78** | 1.0 |
| MAE(Δ̂_do) vs no-update 0.049 | 0.144 | 0.118 | <0.049 |
| corr(do−obs diff, true gap) | −0.01 | −0.06 | >0 |
| placebo movement vs real | equal | equal | ≈0 |

**Headline: models move on framing, not causality.** Both over-update ~2.4× the true
effect scale and move as much for exact-null placebos as for certified interventions;
neither's do-vs-obs difference tracks the true confounding gap. gpt-oss-120b does
know effect DIRECTION (78% vs 39% for 4B) — capability gradient exists. Per plan,
this is the "clean negative": models forecast trajectories but cannot update under
interventions. Remaining B2 cell: the A2 dense-RL model when trained.
Results: tmp/causal/elicit_{q4b,oss120}.json, scripts/uplift_v2/causal_elicit*.py.

## A2 launch record (2026-07-19 ~18:50 PT)

4 jobs live: fbsim-v2-{dense,hard}-s{1,2}, qwen3-4b, 3 epochs, lr 1e-4, LoRA r8,
G=8, temp 1.0, maxOut 8192, ctx 24576, chunkSize 200, evaluator
test-forecast-test-forecast (reward = 1-(p-gt)^2 on <probability> tag).
Data: data/uplift_v2/rft_v2/train_{dense,hard}.jsonl — 1682 rows, 30 fresh worlds
(seed300-339 minus 303-307,323-327 casualties; retries running), identical prompts,
dense gt mean .487 (48% interior), hard gt 0/1, mean|dense-hard|=.198.
Tooling: scripts/uplift_v2/fw_rft.py (upload/launch/status/cancel).
Stage-0 val (340-344) & test (345-359) worlds labeling at N=40 in worker C.

IMPORTANT eval-set bookkeeping: the 4 jobs trained on exactly 30 worlds =
seed300-339 MINUS {303,304,305,306,307,323,324,325,326,327} (those failed during
training-set assembly and were retried later). The 10 retried worlds are labeled
now but NOT in the training data → they form an extra "held-out same-distribution
seeds" eval set alongside val (340-344, N=40) and test (345-359, N=40).
data/training_v2/train.jsonl was later regenerated with all 40 worlds (2243 rows)
— do NOT confuse it with the pinned 1682-row Fireworks copy
(data/uplift_v2/rft_v2/train_{dense,hard}.jsonl is the ground truth for what was
trained on).

## RESUME CHECKLIST (paused 2026-07-20 ~afternoon, laptop closing)

State at pause: all 4 A2 jobs finishing on Fireworks (dense-s2 COMPLETED, others
80-90%); training continues unattended. All local evals stopped cleanly and both
eval deployments DELETED (no idle spend). Stage 0 fully done (60/60 worlds).

To resume (evening, lid open ~3h):
1. `set -a; source .env; set +a; uv run python scripts/uplift_v2/fw_rft.py status`
   — expect all 4 COMPLETED; output models named rftj-fbsim-v2-<job>-<suffix>.
2. Recreate deployments (scale-to-zero, 1×H100) for: base qwen3-4b + each rftj
   model — pattern in scripts/uplift_v2/a2_autopilot.sh.
3. Run scripts/uplift_v2/eval_a2.py per model (val 228×2 + test 840×1 calls,
   temp 1.0 matched) -> tmp/uplift_v2/a2_<tag>.json. ~$9/model, ~1.5h each,
   run all 5 concurrently (separate deployments).
4. Compare: dense-s1/s2 vs hard-s1/s2 vs base — paired bootstrap
   (compare_a1.py logic), world-level aggregation, post-hoc-calibrated variant,
   shrinkage check. Success criteria in this doc + big-experiments-plan.txt.
5. Delete all deployments after; then B2 dense-model cell (causal_elicit.py
   --model <best dense>), then decide A3 RunPod gate.

## Stage-3 GRPO-on-cheap-GPUs plan (derisked 2026-07-21, pre-approval)

Live RunPod prices (community): RTX4090 24GB $0.34/h, **A40 48GB $0.35/h**,
L40S 48GB $0.79, A100-80 $1.19, H100SXM $2.69. Key facts:
- Community cloud has NO network volumes → bake a docker image with the full
  pinned verl env (torch 2.7.0+cu126, vllm 0.9.2, flash-attn wheel, transformers
  4.52.4 — all six July-13 fixes) and push to a registry; pods boot ready.
- Bring-up/debug on **1×A40 at $0.35/h** (vs $23.92/h last incident — 68× cheaper
  to be wrong). A40=sm_86 Ampere, safest with the pinned stack; 4090=sm_89 needs a
  $2 smoke first. 48GB VRAM removes verl-colocate OOM risk for 4B (+12k prompts).
- Throughput estimate (8×A40): ~47M gen tokens/epoch ÷ ~15-20k tok/s ≈ 45-60
  min/epoch + LoRA update overhead → ~1h/epoch vs Fireworks' ~8.5h.
- Cost: bring-up+bake ~$3; smoke ~$2; 12-epoch probe w/ early stop ~$30-35;
  then per final run (~8 epochs) ~$25. Two arms × 1 seed ≈ $50; × 2 seeds ≈ $100.
  Full program $85-140 within the $150 budget (choose seeds after probe).
- All five cost-control layers from RUNBOOK.md mandatory (TTL, dead-man, idle
  watchdog, external checks, low prepaid balance).

## ML best practices skipped in the Fireworks arms — recoverable in Stage 3

1. **Validation during training** — stop/select on val Brier, not train reward.
   (Fireworks has an unused `evaluationDataset` field — set it on ANY future RFT
   job to get val curves even in the black box!)
2. Early stopping with patience (user's 3-epoch rule) — native in our loop.
3. Per-epoch data reshuffling (Fireworks provably reuses fixed chunks in order —
   period-9 sawtooth in the reward curve).
4. Checkpoints every N steps + held-out learning curves (recovers the "learning
   speed" endpoint A2 lost).
5. Explicit RNG seeds per run (Fireworks exposes none; "s1 vs s2" differ only by
   sampling nondeterminism).
6. Mean-only GRPO advantages (the OpenForecaster fix; the entire Stage-3 point).
7. Explicit, logged KL coefficient — a hidden KL tether would explain the
   still-climbing-at-3-epochs curve.
8. LR warmup + cosine schedule (OF: 5e-6 AdamW, 1% warmup, min_lr_ratio 0.1).
9. Data curriculum: drop/downweight extreme-label rows (40% of ours have
   p_mc<0.05 or >0.95 where dense≡hard and gradients vanish).
10. Guardrail metrics logged live: per-group advantage std / zero-advantage rate,
    prediction entropy, parse rate (catch collapse during, not after).
OpenForecaster reference dose: 5 epochs × 54k samples (~1300 steps); our 3 epochs
× 1682 rows is tiny by comparison — consistent with undertraining.

## A2 FINAL RESULT (2026-07-21, ~$70 total Fireworks incl. eval mishaps)

Test = 15 unseen worlds, 839 questions, N=40 labels. Brier vs p_mc (raw / affine-
calibrated fit-on-val): const-0.5 .154/—; base .225/.157; hard-s1 .189/.141;
hard-s2 .226/.142; dense-s1 .172/.145; dense-s2 .155/.137.
- **RL effect: YES** — 3/4 trained models beat base SIG (paired bootstrap);
  dense seeds win 15/15 worlds each.
- **Dense-signal effect: YES** — dense-arm beats hard-arm −0.044 [−.054,−.035]
  paired; both dense seeds beat both hard seeds (no overlap).
- **Mechanism**: hard-label training pushes confidence to extremes (|p−.5|
  .33/.40 vs dense .25/.28) — its target IS 0/1; sometimes pays (s1), sometimes
  collapses to base level (s2). Dense arm is better AND more seed-stable.
- **All four trained models beat calibrated-base and const-0.5 after post-hoc
  calibration** → all learned real discrimination; plan's stronger-success
  criterion met for the dense arm.
- Seed spread: dense .017, hard .037 raw Brier → hard arm is high-variance;
  Stage-3 replication should run 2 seeds/arm (~$140) or at least 2 for hard.
Gate → **Stage-3 RunPod GRPO replication JUSTIFIED** (awaiting user go).
Models: rftj-fbsim-v2-{dense,hard}-s{1,2}-*; analysis scripts/uplift_v2/final_a2.py.

## Sanity-check reminders (from goal directive)

- Measure real tokenized prompt lengths before setting any max_prompt/ctx (prompts are
  9.7k-11.2k tok; ASCII tables ≈2.8 chars/token).
- No arbitrary timeouts; no max_tokens that truncate reasoning (8192 out was OK for RFT).
- After each result: does this change the plan? Gate before building on it.
