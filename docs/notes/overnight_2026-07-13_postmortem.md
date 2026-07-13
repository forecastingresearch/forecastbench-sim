# Overnight run 2026-07-13 — results and post-mortem

## What was delivered

### (b) Causal golden-question pilot — COMPLETE (simulation side), $0
320 rollouts (4 seeds × 40 baseline + 4 × 40 intervention), all local CPU.
Ground truth for P(Y), P(Y|do X), P(Y|X observed) on 220 questions.
- **Phase-4 gate passed:** 19 questions with |Δ_do| > 0.15 (needed ≥10).
- **Headline: the confounding gap is real and large.** 46/220 questions have
  |Δ_obs − Δ_do| > 0.15. Best example (seed2, do(The Republic) for player 0):
  interventional effect **−0.38**, observational effect **+0.05** — opposite signs.
  An LLM that treats "given that X" evidentially vs interventionally identically
  is measurably wrong here, and this quantity cannot be measured outside a simulator.
- Caveat: N=40/arm (SE ≈ 0.08–0.11; observational arm 10–22 rollouts, SE up to ~0.16).
  Some of the 46 gaps are noise; the aggregate and the top examples are far outside it.
- Data: `tmp/causal/` (manifests + per-rollout archives), analysis in
  `tmp/causal/analysis.json` (committed), runner `scripts/causal_pilot.py`.
- **Phase 4 (LLM elicitation, ~$5–15 API) is designed but awaiting explicit OK.**

### (a) Step-0 small-model baselines — COMPLETE, ~$10–15 of Fireworks credits
698 MC-labeled binary questions, single-question elicitation, format ladder.
| model | parse rate | Brier vs p_mc | Brier vs 0/1 | ECE (hard) |
|---|---|---|---|---|
| qwen3-8b (thinking) | 98.9% | 0.198 | 0.272 | 0.207 |
| qwen3-4b-instruct-2507 | 100% | 0.214 | 0.296 | 0.277 |
| llama-v3p1-8b-instruct | 100% | 0.235 | 0.308 | 0.252 |
All three are comfortably weak (a constant-0.5 predictor scores ≈0.145 vs p_mc on
this set); format failures are effectively solved. Results in `tmp/uplift_v2/`.
All Fireworks deployments verified deleted.

### (a) A/B training — NOT RUN. Environment debugged, everything reproducible.
Six environment failures were found and fixed on the pod (all fixes committed):
1. OF `setup.sh` assumes uv's old `~/.cargo/env` path → shim.
2. `torchcodec` 0.14 (CUDA-13 build) fatally imported by `datasets` → uninstall.
3. Vendored verl reads **parquet only** → convert jsonl → parquet.
4. `uv pip install -e .` had silently upgraded torch to 2.11+cu130 → flash-attn ABI
   break → repinned stack: torch 2.7.0+cu126, vllm 0.9.2, matching flash-attn wheel.
5. transformers ≥4.54 `aimv2` config clash with vllm 0.9.2 → pin 4.52.4.
6. Prompts tokenize to 9.7k–11.2k tokens (ASCII tables ≈2.8 chars/token, not 4)
   → `max_prompt_length` 8192 filtered ALL rows ("Train dataloader is empty!")
   → raised to 12288.
The 6th fix was applied and a final smoke launched at ~04:23; no verified-clean
smoke was observed before monitoring stopped.

## The money post-mortem

RunPod balance went **$150.05 → −$0.40**. RunPod force-exited the pod at 07:02 PT;
pod + volume now terminated, account has no running resources.

Where it went (8×H100 SXM @ $23.92/hr):
- ~$4 — pod #1 (recreated: no sshd without PUBLIC_KEY env).
- ~$18 — bootstrap (OF install, flash-attn, 23GB model downloads).
- ~$65 — **six debug cycles with the full 8-GPU pod metering the whole time.**
- ~$63 — **pod idling unattended 04:23→07:02** after the last smoke attempt,
  because the monitoring loop stopped firing (the session stopped receiving
  scheduled wakeups; there was no server-side kill switch).

Two process failures, both mine:
1. **Debugging on the expensive topology.** Environment bring-up should have been
   done on a 1×GPU (or CPU) pod against the same volume, scaling to 8× only for
   the verified run. Would have cut the debug burn ~8×.
2. **No dead-man switch on the pod itself.** Cost control depended entirely on my
   heartbeat loop, which died silently. A pod-side `sleep 4h && poweroff`, a spend
   cap, or an idle-watchdog would have capped the damage at ~$40 regardless.

## Cheapest paths to the A/B result from here

1. **$0: Fireworks RFT arm first.** RFT is free for <16B models; per-row
   ground_truth reaches the reward evaluator, so dense-vs-hard works. Sacrifices
   mean-only advantages (not exposed) — it becomes a "standard-GRPO" variant of
   the A/B, still informative as a first signal.
2. **~$60–80: rerun on RunPod with the hardened plan** — all six fixes baked into
   bootstrap; env brought up on 1×H100 (~$3/hr), snapshot the volume, then 8×H100
   only for smoke + two arms with a pod-side auto-shutdown timer.
3. **~$35–45: same but 4×H100** (needs a 1-line judge-device fix in verifier.py).
