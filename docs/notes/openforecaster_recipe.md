# OpenForecaster recipe — settings we replicate (and our deviations)

Extracted 2026-07-13 from arXiv:2512.25070 **v2** (Appendix C) and
github.com/OpenForecaster/scaling-forecasting-training (verl vendored at
`libraries/verl/`). Flagship OpenForecaster-8B is **RL-only** (GRPO on Qwen3-8B,
thinking); SFT→RL is an ablation with the same RL config, only the init differs.

## Their exact RL settings (binary questions)

| Item | Value |
|---|---|
| Framework | vendored verl, `main_ppo`, FSDP actor (bf16, param+optim offload, grad ckpt) + vLLM rollout (TP=1, mem_util 0.5) |
| Model | Qwen3-8B, thinking mode, **full fine-tune** |
| Reward (binary) | `-(1-p)^2 if y==1 else -(p^2)`, p = last `<probability>` tag after `</think>`, clipped to [0,1]; unparseable → −0.25 default Brier + −1 format = **−1.25**; no correctness bonus for binary |
| Group size K | 8, rollout temp 1.0 |
| Batch | 256 prompts / PPO mini-batch 64, 1 PPO epoch |
| LR | 5e-6 AdamW, cosine, 1% warmup, min_lr_ratio 0.1 |
| KL | in-loss vs frozen init policy, `kl_loss_coef=0.005`, `low_var_kl` |
| Advantage | **group-mean-centered only** (`norm_adv_by_std_in_grpo=False` — no σ division) |
| Clip | 0.2 / 0.28 (DAPO asymmetric), dual-clip c=10 |
| Entropy bonus | none |
| Lengths | max prompt 4096 (overlong **dropped**, never truncated), max response 8192 |
| Epochs | 5 over ~54k samples ≈ 1300 steps, 8×H100, ~1000 H100-hr |
| Validation | temp 0.6 / top-p 0.95, best-val checkpoint |
| Eval | n=3 samples, temp 0.6/top-p 0.95, Brier = −(p−y)² |

Binary prompt template: reproduced verbatim in `scripts/uplift_v2/build_of_dataset.py`
(no-retrieval variant; asks for reasoning then `<probability> </probability>` tags,
includes the Brier-scoring explanation text — the reward parser and prompt are coupled).

## Our replication: what changes and why (each deviation justified)

1. **Data source only (the point):** 641 FB-Sim binary questions (world report as
   Question Background), val 57 (held-out game seed3). Arms `hard` (ground_truth =
   baseline 0/1) vs `dense` (ground_truth = p_mc) — **byte-identical prompts, same
   shuffle order**; the A/B is only the number in `reward_model.ground_truth`.
2. **Reward generalization (2 lines):** binary branch → `-(p - ground_truth)^2`.
   Identical to theirs when ground_truth ∈ {0,1}; dense arm scores against p_mc.
   Unparseable remains −1.25, per theirs.
3. **max_prompt_length 8192** (theirs 4096): our prompts are ~5.5–6.5k tokens
   (world report); their 4096 with drop-overlong would discard the entire dataset.
4. **Batch 64 / mini 16** (theirs 256/64; ratio preserved): 641 prompts at batch 256
   is 2.5 steps/epoch — too coarse for learning curves. 64 → 10 steps/epoch.
5. **Epochs 8 (~80 steps)** vs their 5: small-data regime; KL-to-init and val-based
   checkpoint selection guard overfitting.
6. **4×H100 node** vs 8×: cost; FSDP offload accommodates 8B full FT.
7. No judge needed (binary-only → their code path skips the judge entirely).

Everything else (K, temps, LR, KL, advantage mode, clips, thinking mode, tag format,
prompt text, unparseable penalty, val protocol) is kept as theirs.
