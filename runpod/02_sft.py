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
import time
import argparse
import re
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          BitsAndBytesConfig, TrainerCallback)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig


class JsonlLogCallback(TrainerCallback):
    """Append every TRL log event (loss, lr, grad_norm, eval_loss) to a JSONL
    file. Also echoes to stdout in a parseable form: [step <N>] key=val key=val ..."""

    def __init__(self, path: str):
        self.path = path
        self.t_start = None
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # Truncate previous run.
        open(self.path, "w").close()

    def on_train_begin(self, args, state, control, **kw):
        self.t_start = time.time()
        print(f"[sft] train_begin t=0", flush=True)

    def on_log(self, args, state, control, logs=None, **kw):
        if not logs:
            return
        rec = {"step": state.global_step,
               "epoch": logs.get("epoch", state.epoch),
               "wall": time.time() - (self.t_start or time.time())}
        rec.update(logs)
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        # Print compact form for monitor visibility
        kvs = " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                       for k, v in logs.items())
        print(f"[sft] step={state.global_step} {kvs}", flush=True)

    def on_save(self, args, state, control, **kw):
        print(f"[sft] saved checkpoint at step {state.global_step}", flush=True)


class BrierOnValCallback(TrainerCallback):
    """Generate on a small val subset at epoch end and log task-level Brier.

    Token CE was a poor checkpoint selector in Session 1; this callback measures
    the final parsed probability directly so E3 can start from the best adapter.
    """

    PROB_RE = re.compile(
        r"PROBABILITY:\s*([01](?:\.\d+)?|\.\d+)", re.IGNORECASE)

    def __init__(self, val_records: list[dict], tokenizer, output_dir: str,
                 max_eval: int = 30, max_new_tokens: int = 1024):
        self.val_records = val_records[:max_eval]
        self.tok = tokenizer
        self.max_new_tokens = max_new_tokens
        self.out_path = Path(output_dir) / "brier_curve.jsonl"
        self.raw_path = Path(output_dir) / "brier_callback_raw.jsonl"
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        open(self.out_path, "w").close()
        open(self.raw_path, "w").close()

    def on_epoch_end(self, args, state, control, model=None, **kw):
        if model is None:
            return
        model.eval()
        device = next(model.parameters()).device
        squared_errors = []
        n_parsed = 0
        raw_rows = []
        with torch.no_grad():
            for ex in self.val_records:
                msgs = [{"role": "user", "content": ex["prompt"]}]
                ids = self.tok.apply_chat_template(
                    msgs, return_tensors="pt", add_generation_prompt=True,
                    enable_thinking=False,
                ).to(device)
                try:
                    out = model.generate(
                        ids,
                        attention_mask=torch.ones_like(ids, device=device),
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                        pad_token_id=self.tok.pad_token_id,
                    )
                    text = self.tok.decode(
                        out[0][ids.shape[1]:], skip_special_tokens=True)
                    matches = self.PROB_RE.findall(text)
                    raw_rows.append({
                        "epoch": state.epoch,
                        "step": state.global_step,
                        "qid": ex.get("qid"),
                        "target": float(ex["p_mc"]),
                        "parsed": bool(matches),
                        "raw_tail": text[-500:],
                    })
                    if not matches:
                        continue
                    pred = float(matches[-1])
                    if not (0.0 <= pred <= 1.0):
                        continue
                    squared_errors.append((pred - float(ex["p_mc"])) ** 2)
                    n_parsed += 1
                except Exception as e:
                    print(f"[brier-callback] error on val example: {e}",
                          flush=True)
        brier = (sum(squared_errors) / n_parsed) if n_parsed else None
        with open(self.out_path, "a") as f:
            f.write(json.dumps({
                "epoch": state.epoch,
                "step": state.global_step,
                "val_brier": brier,
                "n_parsed": n_parsed,
                "n_total": len(self.val_records),
            }) + "\n")
        with open(self.raw_path, "a") as f:
            for row in raw_rows:
                f.write(json.dumps(row) + "\n")
        print(f"[brier-callback] epoch {state.epoch:.1f}: "
              f"val_brier={brier} "
              f"(n_parsed={n_parsed}/{len(self.val_records)})",
              flush=True)
        model.train()


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
    ap.add_argument("--max-seq", type=int, default=8192)
    ap.add_argument("--use-weights", action="store_true",
                    help="Sample-weight by 4*p*(1-p); else uniform.")
    ap.add_argument("--no-quant", action="store_true",
                    help="Skip 4-bit quant; load model in bf16. ~2-3x faster on A100 80GB.")
    ap.add_argument("--grad-checkpoint", action="store_true",
                    help="Enable gradient checkpointing. Default off (saves recompute time).")
    args = ap.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Try flash_attention_2 if installed (lower mem, faster), else SDPA.
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "sdpa"
    print(f"[sft] attn_implementation={attn_impl}")

    if args.no_quant:
        print(f"Loading {args.model} in bf16 (no quant)...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16,
            device_map="auto", trust_remote_code=True,
            attn_implementation=attn_impl)
        if args.grad_checkpoint:
            # LoRA + grad_ckpt needs this; prepare_model_for_kbit_training does it
            # automatically in the 4-bit path.
            model.enable_input_require_grads()
            model.gradient_checkpointing_enable()
    else:
        print(f"Loading {args.model} in 4-bit...")
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16,
                                 bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, quantization_config=bnb, torch_dtype=torch.bfloat16,
            device_map="auto", trust_remote_code=True,
            attn_implementation=attn_impl)
        model = prepare_model_for_kbit_training(model,
                                                use_gradient_checkpointing=args.grad_checkpoint)
    lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_r * 2,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"],
                      lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    # Build dataset in chat messages format so TRL applies Qwen3's native chat
    # template, including the thinking-mode trigger. Sample-weight via
    # repeat-counts if --use-weights.
    train = load_jsonl(args.train)
    val = load_jsonl(args.val)

    def to_messages(ex: dict) -> dict:
        return {
            "messages": [
                {"role": "user", "content": ex["prompt"]},
                {"role": "assistant", "content": fmt_target(ex["p_mc"])},
            ],
            "weight": ex.get("weight", 1.0),
        }

    train_recs = [to_messages(r) for r in train]
    val_recs = [to_messages(r) for r in val]

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

    train_ds = Dataset.from_list([{"messages": r["messages"]} for r in train_recs])
    val_ds = Dataset.from_list([{"messages": r["messages"]} for r in val_recs])

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
        logging_steps=1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=10,  # keep every epoch; pick best via post-SFT Brier
        report_to=[],
        disable_tqdm=True,  # quiet the progress bar; rely on JsonlLogCallback
        max_length=args.max_seq,
        packing=False,
        gradient_checkpointing=args.grad_checkpoint,
    )

    metrics_path = f"{args.output}/sft_metrics.jsonl"
    log_cb = JsonlLogCallback(metrics_path)
    brier_cb = BrierOnValCallback(val, tok, args.output)
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=train_ds,
                         eval_dataset=val_ds, processing_class=tok,
                         callbacks=[log_cb, brier_cb])
    print(f"[sft] starting train; metrics -> {metrics_path}")
    trainer.train()
    print(f"[sft] train done; final loss={trainer.state.log_history[-1] if trainer.state.log_history else None}")
    trainer.model.save_pretrained(f"{args.output}/adapter")
    tok.save_pretrained(f"{args.output}/adapter")
    print(f"Saved adapter to {args.output}/adapter")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
