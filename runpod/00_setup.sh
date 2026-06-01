#!/usr/bin/env bash
# Run once on a fresh RunPod pod (image: runpod/pytorch:2.4-cu124-py311 or similar).
# Installs Python deps and pre-warms HF model cache. ~10 minutes.
set -euxo pipefail

# Upgrade pip + install the RL + serving stack.
python -m pip install --upgrade pip
python -m pip install \
    "torch>=2.4" \
    "transformers>=4.46" \
    "trl>=0.13" \
    "peft>=0.13" \
    "accelerate>=1.0" \
    "datasets>=3.0" \
    "bitsandbytes>=0.44" \
    "vllm>=0.6.3" \
    "huggingface_hub[cli]>=0.26"

# Optional: HF token for gated models (Qwen3 is open, Llama is gated).
if [ -n "${HF_TOKEN:-}" ]; then
    huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential
fi

# Pre-download base model into /workspace/.cache to avoid re-downloads
# across training/eval. Default Qwen3-14B (~28GB). Override with $BASE_MODEL.
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-14B}"
export HF_HOME=/workspace/.hf_cache
mkdir -p "$HF_HOME"
huggingface-cli download "$BASE_MODEL" --local-dir-use-symlinks False \
    --max-workers 8 || true

echo "OK — environment ready. Free disk:"
df -h /workspace || df -h .
