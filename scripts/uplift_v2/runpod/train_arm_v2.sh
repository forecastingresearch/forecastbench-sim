#!/bin/bash
# NOTE steps/epoch depends on dataset rows: pass STEPS_PER_EPOCH to monitor_val.
# v2: Train one A/B arm — Stage-3 replication of the A2 Fireworks result.
# Changes from v1: Qwen3-4B (match A2 base), v2 data (1682 rows / 30 worlds),
# 12-epoch ceiling + best-val checkpoint selection (early stop by patience via
# monitor_val.sh), explicit SEED, 8xA40 topology (48GB, Ampere).
# Usage: ARM=dense|hard [SMOKE=1] bash train_arm.sh
# Deviations from upstream (see docs/notes/openforecaster_recipe.md): data source,
# max_prompt 8192 (our prompts are ~6.5k tokens), batch 64/mini 16 (641-question
# dataset), 5 epochs (~50 steps), console logger, val_before_train=True.
set -euo pipefail

ARM="${ARM:?set ARM=dense|hard}"
SMOKE="${SMOKE:-0}"
SEED="${SEED:-1}"

cd /workspace/scaling-forecasting-training
source forecast/bin/activate

NUM_GPUS="${NUM_GPUS:-8}"
[ -z "${CUDA_VISIBLE_DEVICES:-}" ] && export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export RAY_DEDUP_LOGS=0
export RAY_IGNORE_UNHANDLED_ERRORS=1
export RAY_DISABLE_IMPORT_WARNING=1
export HYDRA_FULL_ERROR=1
export VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1

clip_ratio_low=0.2
clip_ratio_high=0.28
PROJECT_NAME="fbsim-ab"
LR=5e-6
KL_COEFF=0.005
MODEL_PATH="/workspace/models/Qwen3-4B"  # v2: match the A2 Fireworks arms
JUDGE_PATH="/workspace/models/Qwen3-4B-Instruct-2507"   # resident but unused (binary-only)
MAX_PROMPT_LENGTH=12288   # prompts tokenize to 9.7k-11.2k (ASCII tables ~2.8 chars/token)
MAX_RESPONSE_LENGTH=8192
EXTRA=()
if [ "$SMOKE" = "1" ]; then
  EXTRA+=("+trainer.total_training_steps=3" "trainer.save_freq=2" "trainer.test_freq=2")
  EXP_NAME="smoke-$ARM"
else
  EXTRA+=("trainer.save_freq=13" "trainer.test_freq=13" "trainer.total_epochs=10")
  EXP_NAME="qwen3-4b-$ARM-seed$SEED"
fi

mkdir -p /workspace/logs /workspace/ckpts
# run manifest: everything a reviewer needs to reconstruct the run
python3 - <<PYEOF
import json, hashlib, subprocess
h = {f: hashlib.md5(open(f'/workspace/data/{f}.parquet','rb').read()).hexdigest()
     for f in ('train_${ARM}','val_${ARM}')}
json.dump({'arm':'$ARM','seed':$SEED,'epochs':10,'lr':'$LR','kl':'$KL_COEFF',
           'model':'$MODEL_PATH','batch':64,'mini':16,'K':8,
           'max_prompt':$MAX_PROMPT_LENGTH,'max_response':$MAX_RESPONSE_LENGTH,
           'clips':[$clip_ratio_low,$clip_ratio_high],'adv':'mean-only',
           'data_md5':h}, open('/workspace/ckpts/${EXP_NAME}_manifest.json','w'), indent=1)
PYEOF
python3 -m verl.trainer.main_ppo \
algorithm.adv_estimator=grpo \
algorithm.norm_adv_by_std_in_grpo=False \
reward_model.enable=True \
reward_model.model.path=$JUDGE_PATH \
reward_model.strategy=matcher \
reward_model.reward_manager=naive \
+reward_model.add_correctness=True \
reward_model.micro_batch_size=0 \
data.train_files=/workspace/data/train_${ARM}.parquet \
data.val_files=/workspace/data/val_${ARM}.parquet \
data.train_batch_size=64 \
data.val_batch_size=64 \
data.shuffle=True \
++data.seed=$SEED \
data.filter_overlong_prompts=True \
data.max_prompt_length=$MAX_PROMPT_LENGTH \
data.max_response_length=$MAX_RESPONSE_LENGTH \
data.truncation='error' \
actor_rollout_ref.model.path=$MODEL_PATH \
actor_rollout_ref.actor.optim.lr=$LR \
actor_rollout_ref.actor.optim.warmup_style=cosine \
actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.01 \
actor_rollout_ref.actor.optim.min_lr_ratio=0.1 \
actor_rollout_ref.model.use_remove_padding=True \
actor_rollout_ref.actor.ppo_mini_batch_size=16 \
actor_rollout_ref.actor.use_dynamic_bsz=True \
actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
actor_rollout_ref.actor.clip_ratio_c=10.0 \
actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
actor_rollout_ref.actor.use_kl_loss=True \
actor_rollout_ref.actor.kl_loss_coef=$KL_COEFF \
actor_rollout_ref.actor.kl_loss_type=low_var_kl \
actor_rollout_ref.model.enable_gradient_checkpointing=True \
actor_rollout_ref.actor.fsdp_config.param_offload=True \
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
actor_rollout_ref.rollout.name=vllm \
actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
actor_rollout_ref.rollout.n=8 \
actor_rollout_ref.rollout.load_format=safetensors \
actor_rollout_ref.rollout.temperature=1.0 \
actor_rollout_ref.rollout.enable_chunked_prefill=True \
actor_rollout_ref.rollout.max_num_batched_tokens=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
actor_rollout_ref.ref.fsdp_config.param_offload=True \
actor_rollout_ref.rollout.val_kwargs.do_sample=True \
actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
actor_rollout_ref.rollout.val_kwargs.n=1 \
trainer.critic_warmup=0 \
"trainer.logger=['console']" \
++trainer.val_before_train=True \
trainer.n_gpus_per_node=$NUM_GPUS \
trainer.nnodes=1 \
trainer.project_name=$PROJECT_NAME \
trainer.experiment_name=$EXP_NAME \
trainer.default_local_dir="/workspace/ckpts/${EXP_NAME}" \
"${EXTRA[@]}" 2>&1 | tee -a /workspace/logs/${EXP_NAME}.log

echo "TRAIN_ARM_DONE $ARM" | tee -a /workspace/logs/${EXP_NAME}.log
