#!/usr/bin/env python3
"""Build CivBench prompt-completion SFT data for LoRA fine-tuning.

This script:
1. Loads paired baseline/conditional question banks from
   data/conditional/<intervention>/{baseline,conditional}/seed<N>/
2. Normalizes question IDs so pairs align across suffix conventions
   (e.g., _intervention, _control)
3. Builds prompts using the existing evaluator prompt builders
4. Builds completion targets from ground truth
5. Mixes conditional and baseline examples (default 70/30 final mix)
6. Writes train/val JSONL files for TRL SFTTrainer prompt-completion training

Default split for the 8B pilot is seeds 1-10, intentionally excluding seed0.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from civrealm.evaluation.parallel_evaluator import (  # noqa: E402
    build_batch_prompt,
    build_continuous_batch_prompt,
)


QUESTION_ID_SUFFIXES = (
    "_intervention",
    "_control",
    "_given",
    "_null",
    "_postintervention",
)


@dataclass
class Example:
    """A single prompt-completion training example."""

    prompt: str
    completion: str
    intervention: str
    seed: int
    condition: str
    question_type: str
    question_id: str
    normalized_question_id: str
    paired_question_id: str
    resolution_turn: int

    def to_record(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "prompt": self.prompt,
            "completion": self.completion,
            # Kept for flexibility if you later switch to conversational SFT.
            "messages": [
                {"role": "user", "content": self.prompt},
                {"role": "assistant", "content": self.completion},
            ],
            "metadata": {
                "intervention": self.intervention,
                "seed": self.seed,
                "condition": self.condition,
                "question_type": self.question_type,
                "question_id": self.question_id,
                "normalized_question_id": self.normalized_question_id,
                "paired_question_id": self.paired_question_id,
                "resolution_turn": self.resolution_turn,
            },
        }


def parse_seed_spec(seed_spec: str) -> list[int]:
    """Parse a seed spec like '1-10,12,15' into sorted unique ints."""
    seeds: set[int] = set()
    for chunk in seed_spec.split(","):
        part = chunk.strip()
        if not part:
            continue
        if "-" in part:
            start_str, end_str = part.split("-", 1)
            start = int(start_str)
            end = int(end_str)
            if end < start:
                raise ValueError(f"Invalid seed range: {part}")
            seeds.update(range(start, end + 1))
        else:
            seeds.add(int(part))
    if not seeds:
        raise ValueError("No seeds parsed from --train-seeds")
    return sorted(seeds)


def normalize_question_id(question_id: str) -> str:
    """Strip known suffixes so baseline/conditional question IDs align."""
    normalized = question_id
    changed = True
    while changed:
        changed = False
        for suffix in QUESTION_ID_SUFFIXES:
            if normalized.endswith(suffix):
                normalized = normalized[: -len(suffix)]
                changed = True
                break
    return normalized


def load_json(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def load_world_report(path: Path) -> str:
    with open(path) as f:
        return f.read()


def extract_ground_truth(question: dict[str, Any]) -> float | None:
    """Extract binary or continuous ground truth as float."""
    qtype = question.get("question_type", "binary")
    resolution = question.get("resolution", {})

    if qtype == "continuous":
        value = resolution.get("value")
        if value is None:
            value = resolution.get("value_at_resolution")
        if value is None:
            return None
        return float(value)

    answer = resolution.get("answer")
    if answer is None:
        return None
    return 1.0 if bool(answer) else 0.0


def format_completion(question_type: str, ground_truth: float, response_style: str) -> str:
    """Build assistant completion from ground-truth target."""
    if question_type == "continuous":
        value = float(ground_truth)
        line = (
            f"p10={value:.6f}, p25={value:.6f}, p50={value:.6f}, "
            f"p75={value:.6f}, p90={value:.6f}"
        )
        if response_style == "full":
            return f"<<<PERCENTILES>>>\n{line}\n<<<END>>>"
        return line

    prob = max(0.0, min(1.0, float(ground_truth)))
    if response_style == "full":
        return f"<<<PROBABILITY>>>\n{prob:.6f}\n<<<END>>>"
    return f"{prob:.6f}"


def build_prompt(question: dict[str, Any], world_report: str) -> str:
    """Use existing evaluator prompt builders for strict train/eval consistency."""
    qtype = question.get("question_type", "binary")
    if qtype == "continuous":
        return build_continuous_batch_prompt([question], world_report)
    return build_batch_prompt([question], world_report)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")


def summarize_examples(examples: list[Example]) -> dict[str, Any]:
    summary: dict[str, Any] = {}

    condition_counts = Counter(x.condition for x in examples)
    type_counts = Counter(x.question_type for x in examples)
    intervention_counts = Counter(x.intervention for x in examples)

    by_seed = defaultdict(Counter)
    for ex in examples:
        by_seed[str(ex.seed)][ex.condition] += 1

    summary["count"] = len(examples)
    summary["condition_counts"] = dict(condition_counts)
    summary["question_type_counts"] = dict(type_counts)
    summary["intervention_counts"] = dict(intervention_counts)
    summary["counts_by_seed"] = {k: dict(v) for k, v in sorted(by_seed.items(), key=lambda kv: int(kv[0]))}
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CivBench SFT dataset for LoRA training")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=REPO_ROOT / "data" / "conditional",
        help="Root directory containing intervention subfolders (default: data/conditional)",
    )
    parser.add_argument(
        "--interventions",
        nargs="+",
        default=["republic", "gold500", "mapmaking"],
        help="Interventions to include",
    )
    parser.add_argument(
        "--train-seeds",
        type=str,
        default="1-10",
        help="Seed spec, e.g. 1-10 or 1-10,12",
    )
    parser.add_argument(
        "--baseline-mix",
        type=float,
        default=0.30,
        help="Target baseline share in final mixed dataset (default: 0.30)",
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.05,
        help="Validation split ratio after mixing (default: 0.05)",
    )
    parser.add_argument(
        "--response-style",
        choices=["minimal", "full"],
        default="minimal",
        help=(
            "Completion style. 'minimal' keeps targets to probability/percentile payload only; "
            "'full' includes delimiter blocks."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling/splitting",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data" / "finetune" / "llama31_8b_pilot",
        help="Directory for output JSONL files",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional cap for quick dry runs",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail immediately on missing files or unmatched IDs",
    )
    args = parser.parse_args()

    if not 0.0 <= args.baseline_mix < 1.0:
        raise ValueError("--baseline-mix must be in [0.0, 1.0)")
    if not 0.0 <= args.validation_ratio < 1.0:
        raise ValueError("--validation-ratio must be in [0.0, 1.0)")

    train_seeds = parse_seed_spec(args.train_seeds)
    rng = random.Random(args.seed)

    conditional_examples: list[Example] = []
    baseline_pool: list[Example] = []
    issues: list[str] = []

    for intervention in args.interventions:
        for seed in train_seeds:
            seed_name = f"seed{seed}"
            base_dir = args.data_root / intervention / "baseline" / seed_name
            cond_dir = args.data_root / intervention / "conditional" / seed_name

            baseline_questions_path = base_dir / "questions.json"
            conditional_questions_path = cond_dir / "conditional_questions.json"
            baseline_world_report_path = base_dir / "world_report" / "turn_060_report.txt"
            conditional_world_report_path = cond_dir / "world_report" / "turn_060_report.txt"

            missing = [
                str(p)
                for p in (
                    baseline_questions_path,
                    conditional_questions_path,
                    baseline_world_report_path,
                    conditional_world_report_path,
                )
                if not p.exists()
            ]
            if missing:
                msg = (
                    f"Skipping {intervention}/{seed_name}: missing required files: "
                    + ", ".join(missing)
                )
                if args.strict:
                    raise FileNotFoundError(msg)
                issues.append(msg)
                continue

            baseline_bank = load_json(baseline_questions_path)
            conditional_bank = load_json(conditional_questions_path)
            baseline_world_report = load_world_report(baseline_world_report_path)
            conditional_world_report = load_world_report(conditional_world_report_path)

            baseline_questions = baseline_bank.get("questions", [])
            conditional_questions = conditional_bank.get("questions", [])

            baseline_by_normalized_id: dict[str, dict[str, Any]] = {}
            for q in baseline_questions:
                qid = q.get("question_id")
                if not qid:
                    continue
                normalized_id = normalize_question_id(qid)
                baseline_by_normalized_id[normalized_id] = q

            for conditional_question in conditional_questions:
                cqid = conditional_question.get("question_id")
                if not cqid:
                    continue
                normalized_id = normalize_question_id(cqid)

                baseline_question = baseline_by_normalized_id.get(normalized_id)
                if baseline_question is None:
                    msg = (
                        f"Unmatched conditional question in {intervention}/{seed_name}: "
                        f"{cqid} (normalized={normalized_id})"
                    )
                    if args.strict:
                        raise ValueError(msg)
                    issues.append(msg)
                    continue

                cond_gt = extract_ground_truth(conditional_question)
                base_gt = extract_ground_truth(baseline_question)
                if cond_gt is None or base_gt is None:
                    msg = (
                        f"Skipping question with missing ground truth in {intervention}/{seed_name}: "
                        f"{cqid}"
                    )
                    if args.strict:
                        raise ValueError(msg)
                    issues.append(msg)
                    continue

                cond_prompt = build_prompt(conditional_question, conditional_world_report)
                cond_completion = format_completion(
                    conditional_question.get("question_type", "binary"),
                    cond_gt,
                    args.response_style,
                )
                conditional_examples.append(
                    Example(
                        prompt=cond_prompt,
                        completion=cond_completion,
                        intervention=intervention,
                        seed=seed,
                        condition="conditional",
                        question_type=conditional_question.get("question_type", "binary"),
                        question_id=cqid,
                        normalized_question_id=normalized_id,
                        paired_question_id=baseline_question.get("question_id", ""),
                        resolution_turn=int(conditional_question.get("resolution_turn", 0)),
                    )
                )

                base_prompt = build_prompt(baseline_question, baseline_world_report)
                base_completion = format_completion(
                    baseline_question.get("question_type", "binary"),
                    base_gt,
                    args.response_style,
                )
                baseline_pool.append(
                    Example(
                        prompt=base_prompt,
                        completion=base_completion,
                        intervention=intervention,
                        seed=seed,
                        condition="baseline",
                        question_type=baseline_question.get("question_type", "binary"),
                        question_id=baseline_question.get("question_id", ""),
                        normalized_question_id=normalized_id,
                        paired_question_id=cqid,
                        resolution_turn=int(baseline_question.get("resolution_turn", 0)),
                    )
                )

    if not conditional_examples:
        raise RuntimeError("No conditional examples were built. Check input paths and seed/intervention filters.")

    if args.baseline_mix > 0:
        baseline_target = round(
            len(conditional_examples) * args.baseline_mix / (1.0 - args.baseline_mix)
        )
        baseline_target = min(baseline_target, len(baseline_pool))
        if baseline_target < len(baseline_pool):
            baseline_selected = rng.sample(baseline_pool, baseline_target)
        else:
            baseline_selected = list(baseline_pool)
    else:
        baseline_target = 0
        baseline_selected = []

    mixed_examples = list(conditional_examples) + baseline_selected
    rng.shuffle(mixed_examples)
    if args.max_examples is not None:
        mixed_examples = mixed_examples[: args.max_examples]

    if args.validation_ratio > 0.0 and len(mixed_examples) > 1:
        grouped: dict[str, list[Example]] = {"conditional": [], "baseline": []}
        for ex in mixed_examples:
            grouped.setdefault(ex.condition, []).append(ex)

        train_examples: list[Example] = []
        val_examples: list[Example] = []

        for _, group in grouped.items():
            if not group:
                continue
            rng.shuffle(group)
            n_val = int(round(len(group) * args.validation_ratio))
            n_val = min(max(n_val, 1), len(group) - 1) if len(group) > 1 else 0
            val_examples.extend(group[:n_val])
            train_examples.extend(group[n_val:])

        rng.shuffle(train_examples)
        rng.shuffle(val_examples)
    else:
        train_examples = mixed_examples
        val_examples = []

    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_records = [x.to_record() for x in train_examples]
    val_records = [x.to_record() for x in val_examples]
    all_records = [x.to_record() for x in mixed_examples]

    train_path = args.output_dir / "train.jsonl"
    val_path = args.output_dir / "val.jsonl"
    all_path = args.output_dir / "all.jsonl"
    manifest_path = args.output_dir / "manifest.json"

    write_jsonl(train_path, train_records)
    write_jsonl(val_path, val_records)
    write_jsonl(all_path, all_records)

    manifest = {
        "config": {
            "data_root": str(args.data_root),
            "interventions": args.interventions,
            "train_seeds": train_seeds,
            "baseline_mix": args.baseline_mix,
            "validation_ratio": args.validation_ratio,
            "response_style": args.response_style,
            "seed": args.seed,
            "max_examples": args.max_examples,
        },
        "counts": {
            "conditional_pool": len(conditional_examples),
            "baseline_pool": len(baseline_pool),
            "baseline_selected": len(baseline_selected),
            "mixed_total": len(mixed_examples),
            "train": len(train_examples),
            "val": len(val_examples),
        },
        "summaries": {
            "mixed": summarize_examples(mixed_examples),
            "train": summarize_examples(train_examples),
            "val": summarize_examples(val_examples),
        },
        "issues": issues,
        "files": {
            "train": str(train_path),
            "val": str(val_path),
            "all": str(all_path),
        },
    }

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print("Built SFT dataset")
    print(f"  train: {train_path} ({len(train_examples)} examples)")
    print(f"  val:   {val_path} ({len(val_examples)} examples)")
    print(f"  all:   {all_path} ({len(mixed_examples)} examples)")
    print(f"  manifest: {manifest_path}")
    if issues:
        print(f"  issues logged: {len(issues)}")


if __name__ == "__main__":
    main()
