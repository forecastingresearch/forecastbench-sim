# RunPod RLVR-on-Freeciv experiment

End-to-end recipe to test whether Brier-vs-`p_mc` RL fine-tuning of a 14B model
transfers to real-world forecasting (ForecastBench).

## Files

| file | what it does |
|---|---|
| `00_setup.sh` | install Python deps; pre-download base model |
| `01_build_training_data.py` | join Freeciv `questions/` + `world_report/` + `p_mc` → JSONL |
| `02_sft.py` | SFT (LoRA) the base model to emit `PROBABILITY: 0.xx` given a world report + question |
| `03_rl.py` | GRPO/Dr.GRPO RL on top of SFT, reward = −Brier vs `p_mc` |
| `04_eval.py` | vLLM eval against ForecastBench and the Freeciv held-out val |

## Pod spec

- **GPU**: 1× H100 80GB SXM5. Community-cloud tier (~$1.99/hr) is plenty.
  H200 also fine, ~$3.50/hr.
- **Container disk**: 100 GB. **Volume**: 200 GB (model cache + checkpoints).
- **Image**: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` or any
  PyTorch ≥ 2.4 + CUDA 12.4 image.
- **Env vars**: `HF_TOKEN` if you need a gated model (Llama). Qwen3 doesn't need one.

## Pre-pod (run locally — produces the data the pod needs)

Both data dirs are git-ignored; build them deterministically from the scripts
in this repo, then rsync the results to the pod.

```bash
# 1. ForecastBench eval set: stratified-sample ~400 of 1246 post-cutoff binary
#    real-world questions (preserves source × resolved_to proportions, base rate 0.19).
uv run python scripts/build_forecastbench_eval.py \
    --cutoff 2025-01-01 \
    --sample 400 \
    --output data/forecastbench/eval_post2025.csv

# 2. Freeciv training JSONL (joins questions + world reports + p_mc)
uv run python runpod/01_build_training_data.py \
    --val-games seed3 \
    --output-dir data/training

# 3. ship to pod
rsync -avz data/forecastbench/ <pod>:/workspace/civbench/data/forecastbench/
rsync -avz data/training/      <pod>:/workspace/civbench/data/training/
```

## Workflow (on the pod)

```bash
# 1. one-time setup
bash runpod/00_setup.sh

# 2. baseline eval (no fine-tuning yet)
python runpod/04_eval.py \
    --base Qwen/Qwen3-14B \
    --forecastbench data/forecastbench/eval_post2025.csv \
    --freeciv-val data/training/val.jsonl \
    --output runs/baseline/eval.json

# 3. SFT
python runpod/02_sft.py \
    --train data/training/train.jsonl \
    --val   data/training/val.jsonl \
    --model Qwen/Qwen3-14B \
    --output runs/sft \
    --use-weights

# 4. eval the SFT checkpoint
python runpod/04_eval.py \
    --base Qwen/Qwen3-14B --adapter runs/sft/adapter \
    --forecastbench data/forecastbench/eval_post2025.csv \
    --freeciv-val data/training/val.jsonl \
    --output runs/sft/eval.json

# 5. RL (Dr.GRPO mode, recommended given noisy p_mc)
python runpod/03_rl.py \
    --train data/training/train.jsonl \
    --val   data/training/val.jsonl \
    --base  Qwen/Qwen3-14B \
    --adapter runs/sft/adapter \
    --dr-grpo --use-weights \
    --output runs/rl

# 6. eval the RL checkpoint
python runpod/04_eval.py \
    --base Qwen/Qwen3-14B --adapter runs/rl/adapter \
    --forecastbench data/forecastbench/eval_post2025.csv \
    --freeciv-val data/training/val.jsonl \
    --output runs/rl/eval.json
```

## Decision gates

After step 4 (post-SFT eval):

- **If SFT held-out Freeciv Brier doesn't drop ≥10% vs base**: the dense
  signal isn't useful for this base model. Stop.
- **If SFT Freeciv Brier drops but ForecastBench Brier doesn't move**: you
  have a transfer failure — the model learned Freeciv idiosyncrasies, not
  forecasting skill. Stop the RL run; the headline result is the
  transfer failure, which is publishable.
- **If both drop**: do the RL phase.

After step 6: compare base vs SFT vs RL on ForecastBench. If RL adds nothing
on top of SFT, that's a real finding (SFT-on-p_mc captured everything; CoT
exploration didn't help further).

## Cost rough math

Qwen3-8B (instruct, thinking-mode-capable) on RunPod community H100
($1.99/hr), 641-train / 57-val Freeciv + 401 stratified-sampled ForecastBench:

| step | wall time | $ |
|---|---|---|
| setup + Qwen3-8B download (~16 GB) | 8 min | $0.27 |
| baseline eval (401 + 57 prompts, vLLM) | 8 min | $0.27 |
| SFT (4 epochs, ~2400 weighted examples) | 1 h | $2 |
| eval after SFT | 8 min | $0.27 |
| RL (600 steps, G=4) | ~8 h | $16 |
| eval after RL | 8 min | $0.27 |
| **total** | **~10 h** | **~$19** |

Budget **$30** for one full loop incl. retries; **$80** for a 3-way sweep
(±dr-grpo, ±weights, ±KL). Stop after SFT if you only care about the
transfer-failure-vs-transfer decision — that's ~$3 of pod time.

For Qwen3-14B substitute, multiply RL/SFT time by ~1.7 (so ~$45 full loop).

## Notes / common gotchas

- `vllm.lora.request.LoRARequest` requires the adapter directory to contain
  `adapter_config.json` + `adapter_model.safetensors` (peft default).
- `bnb_4bit_quant_type="nf4"` is the recommended quant for inference + RL on
  H100. For H200, you can skip 4-bit and load bf16 directly.
- If the model's `max_model_len` is < your longest prompt, vLLM will silently
  truncate; check `eval.json["results"][...]["stats"]["n_parsed"]`. Defaults
  here are tuned for the actual Freeciv prompt distribution (p95 ≈ 6.8K
  tokens): SFT `--max-seq 8192`, RL `--max-prompt 7600`, eval `max_model_len
  10240`. Drop these only if you hit OOM, and only to fit the p50 prompt.
- The Qwen3 thinking-mode token (`<think>...</think>`) is fine inside our
  CoT region; the parser only cares about the final `PROBABILITY:` line.
- For Llama-3.1-8B / Qwen3-8B (cheaper), bump `--batch 4 --grad-accum 4`.
