# RLVR-on-Freeciv overnight session report

**Run window**: 2026-06-03 ~02:30–07:15 PT (RunPod H200 SXM secure cloud).
**Total spend (approx.)**: $20–24 of pre-loaded RunPod balance.
**Pods used**: 4 (two community A100 80GB attempts hit OOM-kill issues; one secure A100; final secure H200 carried SFT+RL+evals).
**Base model**: `Qwen/Qwen3-8B` (instruct, post-trained).
**Methodology recap**: SFT on `p_mc` Brier targets (4·p·(1−p) weighted) → eval → GRPO with `−Brier(pred,p_mc)` reward → eval. Hypothesis: dense p_mc reward gives more sample-efficient transfer to ForecastBench than 0/1 labels would.

---

## TL;DR

1. **Baselines** confirmed the base model is barely-calibrated on either suite (Brier worse than constant-baseline) and **catastrophically loop-degenerates on FB at temperature 0** (only 36% parseable). Setting `temperature=0.6` fixed the parse rate to 93% and is the right eval default going forward.

2. **R1-Distill-Qwen-7B comparison**: cleaner parse on FB but **worse Brier on both suites** than Qwen3-8B. R1's verbose CoT runs out of tokens on the 6800-token Freeciv prompts (only 68% parse on val). Qwen3-8B is the right base.

3. **SFT outcome — Freeciv learned, FB harmed.** On a per-question apples-to-apples basis (common parsed set), SFT epoch-4 is the **only condition that beats constant-baseline on Freeciv-val** (Brier 0.146 vs 0.151) — a real signal that the model learned the in-distribution probability mapping. But it cost FB performance: parse rate dropped 93% → 80%, Brier on FB worsened 0.173 → 0.224. **Classic transfer-failure pattern** from the README decision gate.

4. **RL outcome — undid the SFT lesson.** GRPO with the Brier reward + KL-to-SFT-ref + parse-fail penalty restored parse rate to 98% on both suites and recovered FB to roughly baseline (Brier 0.192 vs 0.173 baseline). But it lost the Freeciv-specific gain (Brier 0.223 — worse than constant). Net: model is back to ~baseline behavior with very high format compliance.

5. **Reading**: the RL reward landscape is dominated by the parse-failure penalty (`-1.0`) and the KL pull-back, not by the noisy 4·p·(1−p)-weighted Brier signal. With only 641 weighted training examples and 300 steps × G=2, the model never accumulated enough gradient on the calibration objective to override the format-compliance pressure. The SFT-good-on-Freeciv adapter is the more interesting checkpoint than the post-RL adapter.

6. **Bottom line on the hypothesis**: **partial confirmation, no transfer**. The dense p_mc signal *did* let SFT learn the held-out Freeciv game's probability mapping (0.230 → 0.146 Brier, ~37% relative improvement). But it did not transfer to real-world ForecastBench, and RL made things worse rather than better. The "publishable transfer failure" finding from the gate is intact.

---

## Headline numbers

All Brier values are computed at `temperature=0.6, max_new_tokens=1024`, with the strict-parse rule from `04_eval.py`. "Constant" = Brier of predicting the parsed-subset base rate every time (per-condition, since parsed subsets differ).

### Full-suite (reported numbers, different parsed subsets)

| condition | FB parse | FB Brier | FB constant | Freeciv parse | Freeciv Brier | Freeciv constant |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-8B greedy | 36% | 0.182 | 0.133 | 96% | 0.197 | 0.152 |
| **Qwen3-8B t=0.6 (baseline)** | **93%** | **0.173** | **0.158** | 84% | 0.230 | 0.150 |
| R1-Distill-Qwen-7B t=0.6 | 97% | 0.204 | 0.154 | 68% | 0.247 | 0.133 |
| SFT epoch-1 (step 69) | 62% | 0.206 | 0.174 | 88% | 0.199 | 0.152 |
| SFT epoch-4 (step 276) | 80% | 0.210 | 0.165 | **98%** | **0.147** | 0.152 |
| **Post-RL (step 300)** | **98%** | **0.176** | 0.158 | 98% | 0.215 | 0.149 |

### Apples-to-apples (only questions all three SFT/RL conditions parsed)

| | FB common (n=211) | Freeciv common (n=49) |
|---|---:|---:|
| SFT epoch-1 | 0.210 | 0.185 |
| **SFT epoch-4** | 0.224 | **0.146** ← beats constant (0.151) |
| **Post-RL** | **0.192** | 0.223 |
| constant baseline | 0.178 | 0.151 |

**Key reading**: SFT ep4 is the only condition that genuinely beats the constant-baseline on Freeciv-val. No condition beats constant on FB.

---

## What happened, step by step

### 0. Setup and infra mishaps (cost: ~$3 wasted)

- **Community-cloud A100 80GB**: pip/huggingface-cli processes kept getting SIGKILL'd (exit 137) by the container cgroup at ~30–50GB RSS, well below the reported 167GB `memory.max`. Confirmed via web research that RunPod community pods have an unadvertised lower limit + host overcommit pressure. Switching to **secure cloud** ($1.39/hr A100 PCIe → later $4.39/hr H200) eliminated the OOM.
- **Stack pinning**: TRL 0.17 → 0.21 needed for `vllm_mode=colocate` arg in GRPOConfig. transformers 4.50 → 4.51+ for Qwen3 `model_type`. vllm 0.6 → 0.11.2 for Qwen3 architecture support. All resolved via `uv pip install --system --no-cache-dir`.
- **Flash-attn**: tried installing, but no pre-built wheel for cu128+torch2.8 → would compile from source (20–40 min). Killed and fell back to SDPA, which is fast enough on H200.

### 1. Baseline characterization (cost: ~$0.50)

`Qwen/Qwen3-8B` at `temperature=0` on FB: **64% of outputs unparseable** because of degenerate "The rationale should not contain... The rationale should not contain..." loops. At `temperature=0.6` parse rate jumps to 93%. On Freeciv-val the opposite — greedy is fine (96% parsed) but `temperature=0.6` hurts (loses ~10pp parse). This dataset-dependent temperature preference is informative: FB needs sampling to escape loops, Freeciv's structured numeric world reports benefit from greedy.

R1-Distill-Qwen-7B was the user's suggestion as a smarter alternative. It cleanly fixes the loop problem on FB (97% parse) but: (a) **Brier is worse** on FB by 0.03 vs Qwen3 t=0.6, (b) **on Freeciv it runs out of 2048-token completion budget** before producing PROBABILITY: line in 32% of cases — its long-form CoT doesn't suit our long structured prompts. Stuck with Qwen3-8B.

### 2. SFT (cost: ~$4)

- **Final config**: bf16 base (no quant), LoRA r=16 (43.6M trainable params, 0.53% of model), 4 epochs, batch=2, grad-accum=8 (eff. batch 16), max-seq=8192, gradient checkpointing on, attn=sdpa. lr=2e-4 cosine, warmup_ratio=0.05, weight-expansion K=4 on 4·p·(1−p) → 641 train examples expanded to **1095** weighted examples.
- **Performance**: 23.9 sec/step on H200, total wall **6649 sec (1h 51min)** for 276 steps.
- **Custom logging callback** added (`JsonlLogCallback`) writing every step to `runs/sft/sft_metrics.jsonl` and printing `[sft] step=N loss=X ...` lines for monitor visibility.
- **Train loss collapsed fast**: 0.29 (step 1) → 0.13 (step 22) → 0.04 (step 47) → 0.005 (step 73) → 0.0005 (step 138, plateau).
- **Eval CE per epoch**: 0.348 / 0.497 / 0.508 / 0.513. CE-overfit visible from epoch 1.
- **Crucially**, Brier-by-epoch is *not* monotonic with CE. Even though epoch 1 had the lowest CE, **epoch 4 had the lowest Brier on Freeciv-val** (0.146 vs 0.199 for epoch 1). The model continued to refine its probability *calibration* in later epochs despite the per-token CE getting worse from over-confident-but-slightly-off-token predictions.
- All 4 epoch checkpoints saved (`runs/sft/checkpoint-{69,138,207,276}`) thanks to `save_total_limit=10`. Originally launched with `save_total_limit=1` — caught this mid-run, killed and restarted; cost was only ~15 min.

### 3. RL (cost: ~$13)

- **Final config**: GRPO via TRL 0.21 GRPOTrainer with `--dr-grpo` (Dr.GRPO: `scale_rewards=False`), G=2 generations per step (vs default 4 — cut rollout cost in half, accepting noisier advantage), 300 steps (vs default 600), max_completion=256 (vs 512), `--use-weights`, `--use-vllm` with `vllm_mode=colocate` and `vllm_gpu_memory_utilization=0.35`, KL coef 0.04 to SFT-ref. Started from SFT-epoch-4 adapter.
- **Performance**: 33.5 sec/step on H200, total wall **10155 sec (2h 49min)** for 300 steps. Without `use_vllm` this would have been ~8h.
- **Reward curve over 300 steps**: roughly flat in the −0.30 to −0.45 range. Some early improvement (steps 30 → 175 dipped from −0.42 to −0.27 average) but then drift back up to −0.35. No clear "learning happened" signal in the reward.
- **`completions/mean_length` pegged at 256** (max) throughout — the model was using the full token budget every time, suggesting the format-compliance penalty was dominant and the reward signal from `−Brier` was being averaged out.
- 3 checkpoints saved (`runs/rl/checkpoint-{100,200,300}`).

### 4. Pod and infra notes

- Final pod `kah13v5p4hvnbd` (H200, secure CA datacenter), total uptime ~5h 30min before termination, cost ~$24.
- SSH timeouts during RL training were frequent (the colocated vLLM + LoRA training is GPU-and-CPU heavy). All `Operation timed out` events resolved themselves on retry; no crashes.
- One brief moment when the pod's `runtime` field went `null` via the GraphQL API for ~1 minute — pod was actually fine, just the API briefly stopped reporting runtime info.

---

## What didn't work / what I'd do differently

1. **Train/eval format mismatch I never resolved**: TRL 0.17+ SFTTrainer automatically wraps `text`-field datasets in the model's chat template ("Converting train dataset to ChatML"). Our `04_eval.py` and `03_rl.py` pass raw text. This is a real mismatch — the SFT model is being asked at eval and RL time to produce its trained behavior from a prompt format it wasn't trained on. I noticed this mid-SFT and noted it, but didn't refactor in time. **This is the #1 thing to fix next.** It might explain why FB performance regressed during SFT (chat-template mismatch + format brittleness) and why RL pulled so hard back toward base behavior.

2. **Reward shaping** is too simple. `−(pred−target)²` with `unparseable=−1.0` and `4·p·(1−p)` per-example multiplier means the marginal reward for going from Brier 0.20 → 0.10 (i.e. from "kind of close" to "very close") is tiny compared to the marginal reward for going from "no PROBABILITY line" to "any PROBABILITY line". RL spent its capacity on the latter. **For next iteration**: shape the reward so calibration improvements are at least comparable magnitude to format compliance — e.g. unparseable penalty = current parse-rate-weighted mean Brier minus 0.1, not a fixed −1.0.

3. **Save-strategy mistake**: original `save_total_limit=1` would have lost all but the final epoch's adapter. Caught mid-SFT and fixed, but it's the kind of footgun worth fixing in the repo. Pushed.

4. **G=2 might be too few generations.** GRPO needs intra-group reward variance to compute advantages; with G=2 you get a single comparison per prompt and `frac_reward_zero_std` (logged by TRL) was ~30%, meaning 30% of groups had both generations land at the same reward and contributed zero gradient. **For next iteration**: G=4 with vllm should still fit time budget on H200; just halve `--steps` proportionally if needed.

5. **No Brier-on-val callback during training.** I have CE-on-val (TRL default `eval_strategy="epoch"`) but CE and Brier diverge in this setup (epoch 4 worst CE, best Brier on Freeciv). **For next iteration**: a custom `TrainerCallback.on_evaluate` that does a short greedy/sampled generation pass on val and reports Brier. Adds ~30s per epoch, much better picture.

6. **Per-template heterogeneity is large.** On Freeciv-val (post-SFT ep4) per-template Brier ranges from 0.029 (`city_count_comparative`) to 0.453 (`wonder_completed`). Some templates are easy, some are hard. The `wonder_completed` template with n=3 is too small a sample to draw conclusions from but worth a separate look.

---

## Artifacts

All preserved locally under `tmp/rl_session/` (gitignored). Key files:

| path | contents |
|---|---|
| `tmp/rl_session/sft/sft_metrics.jsonl` | 276 step + 4 eval lines, per-step loss / lr / grad_norm / mean_token_accuracy |
| `tmp/rl_session/sft/checkpoint-{69,138,207,276}/` | LoRA adapter weights + trainer_state.json |
| `tmp/rl_session/post_sft/{ep1,ep4}/eval.json` | full prompt-by-prompt predictions + Brier stats for the two SFT epochs evaluated |
| `tmp/rl_session/rl/checkpoint-{100,200,300}/` | RL adapter weights at 3 points + trainer_state.json (60 GRPO log entries) |
| `tmp/rl_session/rl/adapter/` | final RL adapter (= checkpoint-300, the live one) |
| `tmp/rl_session/post_rl/eval.json` | full post-RL eval at t=0.6 on FB + Freeciv-val |

Repo state: branch `rlvr-dense-signal` at f7e9362 + uncommitted edits to:
- `runpod/02_sft.py`: added `--no-quant`, `--grad-checkpoint`, `JsonlLogCallback`, flash-attn auto-detect, `save_total_limit=10`, `disable_tqdm=True`.
- `runpod/03_rl.py`: added `--use-vllm`, `--no-quant`, `--grad-checkpoint`, `vllm_mode=colocate`, `save_steps=100`.

All these script edits are worth committing, with a note that the `save_total_limit=1` default was a footgun.

---

## Concrete next experiments (in priority order)

1. **Refactor the train/eval pipeline to consistently use Qwen3's chat template** in SFT, RL, and eval. This is the most likely source of the FB regression during SFT. Should be a 1-day change.

2. **Re-shape the RL reward**: scale `−Brier` by 10× and reduce `unparseable_penalty` from −1.0 to roughly −(median parsed Brier + 0.05) so format penalty isn't dominant. Re-run RL on SFT-ep4 with new reward.

3. **Run with G=4 and 200 steps** instead of G=2 and 300 — same total rollouts but better advantage estimates.

4. **Train SFT for fewer epochs** (1 or 2) and compare. The clean Brier-by-epoch curve (0.199 → 0.147 → ? → ?) suggests epoch 4 was best on Freeciv-val but we never measured Brier at epochs 2 and 3. With save_total_limit=10 in place that's a 15-min eval to run on the saved checkpoints.

5. **Larger training set**: 641 weighted examples is small. Generate more Freeciv MC runs to push toward 2-3k training examples; with the dense-signal hypothesis the marginal value of more games should be high.

6. **A "no-FB" diagnostic**: train SFT on a *subset* of templates and eval on held-out templates within Freeciv. If even within-Freeciv-template generalization is poor, the model is just memorizing the train-game world reports, not learning anything resembling forecasting. That would be a stronger negative result than the FB-failure.

7. **Try with a different base model.** R1-Distill-Qwen-7B was worse out-of-the-box but might respond better to SFT. Llama-3.1-8B-Instruct is the other obvious candidate (gated, needs HF token).

---

## Addendum: chat-template mismatch — partial answer via local tokenizer inspection

Attempted to run a full **base Qwen3-8B eval with `--chat-template` flag** on a fresh pod to test the "train/eval format mismatch caused FB regression" hypothesis. **Couldn't run the inference experiment** — both A100 pods I provisioned (CA and US datacenters) had HF Hub downloads stalling out at ~1.2 GB and ~360 MB respectively, even with and without `hf_transfer`. Spent ~$1 across two pods before cutting losses. Likely HF Hub rate-limiting today; not a pod-side issue (the same model downloaded cleanly on the original H200 pod 4 hours earlier).

**However**, I did a local check on what TRL's chat-template application actually looks like — no GPU needed:

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-8B', trust_remote_code=True)

# Raw text (what our SFT/eval pipeline actually passes)
raw = rec['prompt'] + '\nPROBABILITY: 0.40' + tok.eos_token

# Chat-template version (what TRL is sometimes accused of applying)
messages = [{'role':'user','content':rec['prompt']},
            {'role':'assistant','content':'PROBABILITY: 0.40'}]
chat = tok.apply_chat_template(messages, tokenize=False)
```

**Token counts**: raw = **10,913**, chat-templated = **10,926**. Only **13 tokens of difference**. The chat-template version wraps the prompt with `<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nPROBABILITY: 0.40<|im_end|>\n` — a handful of role-marker tokens plus an empty `<think></think>` block that Qwen3's template inserts by default.

**Refined reading of the hypothesis**:

1. **Token-count mismatch is small** (≈0.1% extra tokens). It is unlikely to have been the dominant cause of the SFT FB regression. If the model can handle the raw 10,913-token format at SFT time, it can also handle it at eval time — the surface forms are essentially identical.

2. **What IS notably different**: the chat-template path explicitly injects `<think>\n\n</think>` between the user prompt and assistant answer. Qwen3 is trained to use this token sequence to gate its reasoning mode. Running raw-completion **bypasses the model's intended "thinking-then-answer" behavior**. The base model at temperature=0 going into degenerate loops on FB is consistent with the model never entering its thinking mode and getting stuck in its general-completion prior.

3. So the most likely revised story for the original session results:
   - **Baseline t=0.6 doing OK on FB**: temperature noise compensates for missing thinking-mode
   - **SFT-on-raw-format making FB worse**: SFT trained the model to skip thinking entirely and go straight to `PROBABILITY: 0.xx` immediately — that's good for Freeciv where the model just needs to read structured numbers, but bad for FB where the model loses what little reasoning it would naturally do
   - **RL restoring FB**: the KL-to-SFT-reference penalty + parse-failure pressure dragged behavior back toward general "produce some text then answer", undoing the no-thinking specialization

4. **Concrete prediction this hypothesis would make**: if we re-ran eval with `--chat-template` on the **base model**, FB Brier should drop noticeably (because thinking mode kicks in) and parse rate should stay high even at temperature 0. **This is the experiment that didn't run today** due to HF Hub being slow.

5. **One thing the local check confirms beyond doubt**: TRL 0.21's "Converting train dataset to ChatML" message during SFT is a **misnomer**. With `dataset_text_field="text"` set (which our 02_sft.py does), TRL does NOT actually apply the chat template — it tokenizes the raw text. So my original train/eval mismatch concern was based on a misread of the TRL log line. The two formats differ by 13 tokens, not by full chat-wrapping. The SFT and eval pipelines were actually consistent on raw-text format. The mismatch was between *our pipeline's raw format* and *the model's natively-trained chat format*, not between SFT-time and eval-time within our pipeline.

**Bottom line**: the chat-template hypothesis is *refined, not refuted*. The mismatch matters because of the missing `<think></think>` reasoning trigger, not because of a token-distribution shift. Re-running the experiment when HF Hub cooperates is still the right next step — but the framing should be "does adding the thinking-mode trigger fix things?" not "does chat-template format match?".

**Final RunPod balance**: $7.12 (started $35.96, spent $28.84 total across the full session).
