#!/usr/bin/env python3
"""Stage 2: RL fine-tune (GRPO) on top of the SFT adapter with -Brier reward.

Reward = -(pred - p_mc)^2 where pred is parsed from the final
"PROBABILITY: 0.xx" line of the generated response. Unparseable -> -1 (Turtel
et al.'s strict-parse trick, prevents format hacking).

Uses TRL's GRPOTrainer. The Dr.GRPO variant (robust to noisy rewards — exactly
our N=20 p_mc situation) is enabled via scale_rewards=False per the Dr.GRPO
paper. To approximate ReMax, set --num-generations 1 and use the SFT model's
greedy as the implicit baseline.

Importance-weight by 4*p*(1-p) via per-example reward scaling (advantages are
unchanged in expectation; trivial questions contribute ~0 gradient).

Run on RunPod:
    python runpod/03_rl.py \\
        --train data/training/train.jsonl \\
        --val   data/training/val.jsonl \\
        --base  Qwen/Qwen3-14B \\
        --adapter runs/sft/adapter \\
        --output runs/rl
"""
from __future__ import annotations

import re
import json
import argparse
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          BitsAndBytesConfig)
from peft import PeftModel, LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer


PROB_RE = re.compile(r"PROBABILITY:\s*([01](?:\.\d+)?|\.\d+)", re.IGNORECASE)


def parse_probability(text: str) -> float | None:
    """Strict parse of the last 'PROBABILITY: 0.xx' in the completion."""
    matches = PROB_RE.findall(text or "")
    if not matches:
        return None
    try:
        p = float(matches[-1])
    except ValueError:
        return None
    if not (0.0 <= p <= 1.0):
        return None
    return p


def load_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--base", default="Qwen/Qwen3-14B")
    ap.add_argument("--adapter", default=None,
                    help="Path to SFT LoRA adapter to start from. "
                         "Omit for RL from the base model.")
    ap.add_argument("--output", default="runs/rl")
    ap.add_argument("--num-generations", type=int, default=4,
                    help="G in GRPO. 1 = REINFORCE-with-baseline (~ReMax-style).")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--kl-coef", type=float, default=0.04,
                    help="KL penalty to the reference (SFT) model.")
    ap.add_argument("--max-prompt", type=int, default=7600)
    ap.add_argument("--max-completion", type=int, default=512)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--dr-grpo", action="store_true",
                    help="Dr.GRPO mode: scale_rewards=False (robust to noisy rewards).")
    ap.add_argument("--use-weights", action="store_true",
                    help="Multiply per-example reward by 4*p*(1-p).")
    ap.add_argument("--unparseable-penalty", type=float, default=-1.0)
    ap.add_argument("--use-vllm", action="store_true",
                    help="Use vLLM for rollout generation (5-10x faster than HF generate).")
    ap.add_argument("--no-quant", action="store_true",
                    help="Skip 4-bit quant; load base in bf16. Needed on H200 with use_vllm.")
    ap.add_argument("--grad-checkpoint", action="store_true",
                    help="Enable gradient checkpointing. Required at seq>4K on most GPUs.")
    args = ap.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    adapter_desc = (f"SFT adapter from {args.adapter}" if args.adapter
                    else "new trainable LoRA adapter")
    if args.no_quant:
        print(f"Loading base {args.base} in bf16 + {adapter_desc}...")
        bnb = None
    else:
        print(f"Loading base {args.base} in 4-bit + {adapter_desc}...")
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16,
                                 bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, quantization_config=bnb, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
        attn_implementation="sdpa")
    if args.no_quant and args.grad_checkpoint:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
    if args.adapter is None:
        print("[rl] starting from base model (no SFT adapter)")
        lora = LoraConfig(
            r=16, lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
        model = get_peft_model(model, lora)
    else:
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=True)
    # Merge nothing — keep adapter trainable. TRL will optimize the LoRA.

    # Build dataset: each example carries the prompt and a precomputed p_mc.
    # The reward function looks up p_mc per-example via a closure on this dict.
    train = load_jsonl(args.train)
    val = load_jsonl(args.val)

    def to_record(ex):
        msgs = [{"role": "user", "content": ex["prompt"]}]
        prompt_text = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        return {"prompt": prompt_text, "p_mc": float(ex["p_mc"]),
                "weight": float(ex.get("weight", 1.0)), "qid": ex["qid"],
                "game_id": ex["game_id"]}

    train_ds = Dataset.from_list([to_record(r) for r in train])
    val_ds = Dataset.from_list([to_record(r) for r in val])

    # Reward fn closed over per-example p_mc + weight (TRL passes the
    # per-example fields from the dataset as kwargs to the reward function).
    def brier_reward(completions, p_mc, weight, **kwargs):
        rewards = []
        for comp, p, w in zip(completions, p_mc, weight):
            text = comp if isinstance(comp, str) else (
                comp[0].get("content") if isinstance(comp, list) and comp else "")
            pred = parse_probability(text)
            if pred is None:
                rewards.append(args.unparseable_penalty)
                continue
            r = -(pred - p) ** 2
            if args.use_weights:
                r = r * w
            rewards.append(float(r))
        return rewards

    cfg = GRPOConfig(
        output_dir=args.output,
        num_train_epochs=1,
        max_steps=args.steps,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=5,
        save_strategy="steps", save_steps=100, save_total_limit=2,
        report_to=[],
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt,
        max_completion_length=args.max_completion,
        beta=args.kl_coef,
        temperature=0.7,
        scale_rewards=not args.dr_grpo,
        log_completions=True,
        num_completions_to_print=2,
        use_vllm=args.use_vllm,
        vllm_mode="colocate" if args.use_vllm else None,
        vllm_gpu_memory_utilization=0.25 if args.use_vllm else None,
        gradient_checkpointing=args.grad_checkpoint,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[brier_reward],
        args=cfg,
        train_dataset=train_ds,
        processing_class=tok,
    )
    trainer.train()
    trainer.model.save_pretrained(f"{args.output}/adapter")
    tok.save_pretrained(f"{args.output}/adapter")
    print(f"Saved RL adapter to {args.output}/adapter")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
