#!/usr/bin/env python3
"""Merge a LoRA adapter into a full model checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer


def parse_dtype(name: str) -> torch.dtype:
    value = name.strip().lower()
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp16", "float16"}:
        return torch.float16
    if value in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--device-map", type=str, default="auto")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    dtype = parse_dtype(args.dtype)

    print("Loading adapter model...")
    model = AutoPeftModelForCausalLM.from_pretrained(
        str(args.adapter_dir),
        torch_dtype=dtype,
        device_map=args.device_map,
    )

    print("Merging adapter...")
    merged_model = model.merge_and_unload()

    print("Saving merged model...")
    merged_model.save_pretrained(str(args.output_dir), safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(str(args.adapter_dir), use_fast=True)
    tokenizer.save_pretrained(str(args.output_dir))

    print(f"Merged model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
