# CivBench 8B LoRA/QLoRA Pilot Pipeline

This runbook prepares everything needed to run the **8B pilot only** without launching the experiment automatically.

## What This Pipeline Includes

- `build_civbench_sft_dataset.py`
  - Builds SFT prompt-completion data from conditional question banks.
  - Uses seeds **1-10** by default.
  - Normalizes question IDs (e.g. `_intervention`, `_control`) for pairing.
  - Mixes conditional/baseline examples to a target ratio (default final mix: 70/30).
  - Uses the existing CivBench prompt builders from `parallel_evaluator.py`.

- `train_lora_sft.py`
  - QLoRA training for `meta-llama/Llama-3.1-8B-Instruct`.
  - Single-GPU focused (A100 80GB recommended).
  - Uses PEFT LoRA with configurable rank/targets.

- `merge_lora_adapter.py`
  - Optional adapter merge step after training.

- `check_finetune_env.py`
  - Basic environment sanity checker.

## Recommended Pilot Defaults (8B)

- Model: `meta-llama/Llama-3.1-8B-Instruct`
- Method: QLoRA (4-bit NF4)
- LoRA rank: `16`
- LoRA targets: `all-linear`
- LR: `2e-4`
- Scheduler: cosine
- Epochs: `4`
- Max length: `4096`
- Attention: `sdpa` (safe default)

## Step 1: Start a Cloud GPU

Pick one:

- **RunPod**:
  - Docs: <https://docs.runpod.io/get-started>
  - Deploy pod: <https://docs.runpod.io/user/console/pods/deploy>

- **Lambda Cloud**:
  - Docs hub: <https://docs.lambda.ai/public-cloud>
  - Quick start / first instance: <https://docs.lambda.ai/public-cloud/getting-started/quick-start/>

Use a template/image with CUDA + PyTorch preinstalled if possible.

## Step 2: Prepare Environment on the GPU VM/Pod

```bash
cd ~
git clone <your-civbench-repo-url> civbench
cd civbench

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Install PyTorch matching the VM CUDA if not already available.
# If the image already has working torch+cuda, you can skip this.

pip install -r scripts/finetune/requirements-lora-sft.txt
```

Sanity check:

```bash
python scripts/finetune/check_finetune_env.py
```

## Step 3: Build SFT Dataset (Seeds 1-10)

```bash
python scripts/finetune/build_civbench_sft_dataset.py \
  --train-seeds 1-10 \
  --interventions republic gold500 mapmaking \
  --baseline-mix 0.30 \
  --validation-ratio 0.05 \
  --response-style minimal \
  --output-dir data/finetune/llama31_8b_pilot
```

Outputs:

- `data/finetune/llama31_8b_pilot/train.jsonl`
- `data/finetune/llama31_8b_pilot/val.jsonl`
- `data/finetune/llama31_8b_pilot/all.jsonl`
- `data/finetune/llama31_8b_pilot/manifest.json`

## Step 4: Launch LoRA/QLoRA Training (8B Pilot)

```bash
python scripts/finetune/train_lora_sft.py \
  --model-id meta-llama/Llama-3.1-8B-Instruct \
  --train-file data/finetune/llama31_8b_pilot/train.jsonl \
  --val-file data/finetune/llama31_8b_pilot/val.jsonl \
  --output-dir outputs/lora_llama31_8b_pilot \
  --run-name civbench-lora-8b-pilot \
  --load-in-4bit \
  --bnb-double-quant \
  --bnb-quant-type nf4 \
  --bnb-compute-dtype bfloat16 \
  --gradient-checkpointing \
  --packing \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --target-modules all-linear \
  --use-rslora \
  --learning-rate 2e-4 \
  --num-train-epochs 4 \
  --warmup-ratio 0.03 \
  --max-length 4096 \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --attn-implementation sdpa \
  --logging-steps 10 \
  --save-steps 100 \
  --eval-steps 100
```

Artifacts will be in:

- `outputs/lora_llama31_8b_pilot/`

## Step 5: Optional Merge

```bash
python scripts/finetune/merge_lora_adapter.py \
  --adapter-dir outputs/lora_llama31_8b_pilot \
  --output-dir outputs/lora_llama31_8b_pilot_merged \
  --dtype bfloat16
```

## Notes

- `response-style minimal` in dataset building keeps completion targets focused on forecast payload.
- If you prefer strict delimiter training targets, rebuild with `--response-style full`.
- This pipeline intentionally does **not** auto-run evaluation or bootstrap stats.

