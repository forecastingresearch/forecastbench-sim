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
# 1. ForecastBench eval set (~1200 post-cutoff binary real-world questions)
uv run python scripts/build_forecastbench_eval.py \
    --cutoff 2025-01-01 \
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

At RunPod community H100 ($1.99/hr) using the existing 220-example dataset:

| step | wall time | $ |
|---|---|---|
| setup + base download | 15 min | $0.50 |
| baseline eval (1246 + 57 prompts, vLLM) | 30 min | $1 |
| SFT (4 epochs, ~660 weighted examples) | 1.5 h | $3 |
| eval after SFT | 30 min | $1 |
| RL (600 steps, G=4) | ~18 h | $35 |
| eval after RL | 30 min | $1 |
| **total** | **~22 h** | **~$42** |

Budget **$80** for one full loop incl. retries; **$200** for a 3-way sweep
(±dr-grpo, ±weights, ±KL).

With more training data (after running step 7 below), RL time scales roughly
linearly with #examples × steps; budget proportionally.

## Notes / common gotchas

- `vllm.lora.request.LoRARequest` requires the adapter directory to contain
  `adapter_config.json` + `adapter_model.safetensors` (peft default).
- `bnb_4bit_quant_type="nf4"` is the recommended quant for inference + RL on
  H100. For H200, you can skip 4-bit and load bf16 directly.
- If the model's `max_model_len` is < your longest prompt, vLLM will silently
  truncate; check `eval.json["results"][...]["stats"]["n_parsed"]`.
- The Qwen3 thinking-mode token (`<think>...</think>`) is fine inside our
  CoT region; the parser only cares about the final `PROBABILITY:` line.
- For Llama-3.1-8B / Qwen3-8B (cheaper), bump `--batch 4 --grad-accum 4`.
