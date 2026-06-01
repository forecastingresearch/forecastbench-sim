#!/usr/bin/env python3
"""Stage 1: SFT a 14B-class model to predict p_mc from a world report.

Target output format is exactly one line: "PROBABILITY: 0.xx" — keeping the
SFT objective close to plain regression-to-p_mc. CoT exploration is reserved
for the RL stage. Sample-weighted by 4*p*(1-p) (information weight): the
trivial 0/1 questions contribute zero gradient, the interior questions
contribute the most.

LoRA on Qwen3-14B by default; one H100 80GB SXM5 is plenty in 4-bit.

Run on RunPod (after `bash runpod/00_setup.sh`):
    python runpod/02_sft.py \\
        --train data/training/train.jsonl \\
        --val   data/training/val.jsonl \\
        --model Qwen/Qwen3-14B \\
        --output runs/sft

Outputs:
    runs/sft/adapter/        (LoRA adapter)
    runs/sft/eval.json       (val Brier + overconfidence frac)
"""
from __future__ import annotations

import os
import json
import argparse
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          BitsAndBytesConfig, TrainingArguments)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig


def fmt_target(p_mc: float) -> str:
    return f"PROBABILITY: {p_mc:.2f}"


def load_jsonl(p: str) -> list[dict]:
    return [json.loads(l) for l in open(p) if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-14B")
    ap.add_argument("--output", default="runs/sft")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--max-seq", type=int, default=6144)
    ap.add_argument("--use-weights", action="store_true",
                    help="Sample-weight by 4*p*(1-p); else uniform.")
    args = ap.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model} in 4-bit...")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_r * 2,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"],
                      lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    # Build dataset: prompt + target text. Sample-weight via repeat-counts if
    # --use-weights (cheapest way to apply weights with SFTTrainer).
    train = load_jsonl(args.train)
    val = load_jsonl(args.val)

    def to_text(ex: dict) -> dict:
        return {"text": ex["prompt"] + "\n" + fmt_target(ex["p_mc"]) + tok.eos_token,
                "weight": ex.get("weight", 1.0)}

    train_recs = [to_text(r) for r in train]
    val_recs = [to_text(r) for r in val]

    if args.use_weights:
        # Replicate each example by ceil(weight * K) — equivalent to importance
        # weighting via repetition. K chosen so the trivial questions dominate
        # less while the dataset doesn't explode.
        K = 4
        expanded = []
        for r in train_recs:
            n = max(1, round(r["weight"] * K))
            expanded.extend([r] * n)
        print(f"Train: {len(train_recs)} -> {len(expanded)} after weight expansion")
        train_recs = expanded

    train_ds = Dataset.from_list([{"text": r["text"]} for r in train_recs])
    val_ds = Dataset.from_list([{"text": r["text"]} for r in val_recs])

    cfg = SFTConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        report_to=[],
        max_seq_length=args.max_seq,
        dataset_text_field="text",
        packing=False,
    )

    trainer = SFTTrainer(model=model, args=cfg, train_dataset=train_ds,
                         eval_dataset=val_ds, processing_class=tok)
    trainer.train()
    trainer.model.save_pretrained(f"{args.output}/adapter")
    tok.save_pretrained(f"{args.output}/adapter")
    print(f"Saved adapter to {args.output}/adapter")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
