#!/usr/bin/env python3
"""
CivBench conditional prompting experiment.

Tests two prompting interventions against the standard conditional baseline:
1. Two-step: direction estimation (unanchored) → magnitude update (anchored)
2. Effect-size: estimate probability shift with calibration warning

Both conditions provide the model's own baseline P(Y) to isolate conditional reasoning.

Usage:
    cd /Users/elsehow/Projects/civbench
    python scripts/run_conditional_prompting_experiment.py [--n-per-template 30] [--dry-run]
"""

import argparse
import asyncio
import json
import logging
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from civrealm.evaluation.models import LiteLLMModel, load_api_keys_from_gcp

load_api_keys_from_gcp()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("prompting_experiment")

# ── Config ──────────────────────────────────────────────────────────────────

MODEL_ID = "claude-sonnet-4-5-20250929"
SONNET_KEY = "anthropic/claude-sonnet-4-5-20250929"  # key in results JSON
MAX_CONCURRENT = 5  # Anthropic concurrency
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# ── Prompt templates ────────────────────────────────────────────────────────

def strip_conditional_prefix(question_text: str) -> str:
    """Strip 'If X switches to Republic next turn, ' prefix and capitalize."""
    # Match patterns like "If Dacian switches to Republic next turn, would ..."
    m = re.match(r"If .+? next turn, (would |will )", question_text, re.IGNORECASE)
    if m:
        rest = question_text[m.start(1):]
        return rest[0].upper() + rest[1:]
    return question_text


def build_direction_prompt(world_report: str, bare_question: str) -> str:
    return f"""You are an expert superforecaster analyzing a FreeCiv game simulation.

## World Report
{world_report}

## Task
A player switches to Republic government next turn. Republic changes trade bonuses, corruption rates, and military unit maintenance costs.

Consider the following question about game outcomes:
"{bare_question}"

Will adopting Republic make this outcome MORE LIKELY or LESS LIKELY compared to not adopting Republic?

Think step by step about how Republic's effects on trade, corruption, and military costs would propagate to affect this outcome. Then give your answer.

You MUST end your response with your direction in this exact format:
<<<DIRECTION>>>
MORE or LESS
<<<END>>>

Replace MORE or LESS with your assessment."""


def build_magnitude_prompt(
    world_report: str, bare_question: str, direction: str, baseline_p: float
) -> str:
    dir_word = "MORE" if direction == "MORE" else "LESS"
    adj = "higher" if direction == "MORE" else "lower"
    return f"""You are an expert superforecaster analyzing a FreeCiv game simulation.

## World Report
{world_report}

## Task
A player switches to Republic government next turn.

You previously assessed that Republic will make the following outcome {dir_word} LIKELY:
"{bare_question}"

Your baseline estimate (without Republic) was: P = {baseline_p:.3f}

Now, given that Republic WILL be adopted, and given your directional assessment, update your probability estimate. Your answer should be {adj} than your baseline of {baseline_p:.3f}.

You MUST end your response with your probability in this exact format:
<<<PROBABILITY>>>
0.65
<<<END>>>

Replace 0.65 with your actual probability estimate between 0.0 and 1.0."""


def build_effect_size_prompt(
    world_report: str, bare_question: str, baseline_p: float
) -> str:
    return f"""You are an expert superforecaster analyzing a FreeCiv game simulation.

## World Report
{world_report}

## Task
A player switches to Republic government next turn. Republic changes trade bonuses, corruption rates, and military unit maintenance costs.

Question: "{bare_question}"

Your baseline estimate (without Republic) is: P = {baseline_p:.3f}

Estimate how much Republic changes the probability of this outcome.

IMPORTANT: Most intervention effects are SMALL. The typical effect of a government change on any single game outcome is a shift of 0.05 to 0.15 in probability. Large shifts (> 0.20) are rare and require a direct causal mechanism.

Step 1: What is the direction? (Republic makes this more or less likely)
Step 2: What is the magnitude of the shift? (a number from 0.00 to 0.50)
Step 3: Compute your final probability = baseline ± shift

You MUST end your response with your final probability in this exact format:
<<<PROBABILITY>>>
0.65
<<<END>>>

Replace 0.65 with your actual probability estimate between 0.0 and 1.0."""


# ── Parsing ─────────────────────────────────────────────────────────────────

def parse_direction(response: str) -> str | None:
    m = re.search(r"<<<DIRECTION>>>(.*?)<<<END>>>", response, re.DOTALL | re.IGNORECASE)
    if m:
        text = m.group(1).strip().upper()
        if "MORE" in text:
            return "MORE"
        if "LESS" in text:
            return "LESS"
    # Fallback: look for MORE/LESS anywhere in last 200 chars
    tail = response[-200:].upper()
    if "MORE LIKELY" in tail and "LESS LIKELY" not in tail:
        return "MORE"
    if "LESS LIKELY" in tail and "MORE LIKELY" not in tail:
        return "LESS"
    return None


def parse_probability(response: str) -> float | None:
    m = re.search(r"<<<PROBABILITY>>>(.*?)<<<END>>>", response, re.DOTALL | re.IGNORECASE)
    if m:
        num = re.search(r"(\d*\.?\d+)", m.group(1).strip())
        if num:
            return max(0.0, min(1.0, float(num.group(1))))
    # Fallback: last number in response
    nums = re.findall(r"(?<!\d)0?\.\d+|1\.0(?!\d)", response)
    if nums:
        return max(0.0, min(1.0, float(nums[-1])))
    return None


# ── Async API calls ─────────────────────────────────────────────────────────

async def call_model(
    model: LiteLLMModel,
    prompt: str,
    semaphore: asyncio.Semaphore,
    label: str = "",
    max_retries: int = 3,
) -> str | None:
    async with semaphore:
        for attempt in range(max_retries):
            try:
                response = await asyncio.wait_for(
                    model.get_response_async(prompt, temperature=0.0),
                    timeout=120,
                )
                return response
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = 2 ** (attempt + 1)
                    logger.warning(f"  [{label}] Attempt {attempt+1} failed: {e}. Retrying in {wait}s...")
                    await asyncio.sleep(wait)
                else:
                    logger.error(f"  [{label}] All retries failed: {e}")
                    return None


# ── Data loading ────────────────────────────────────────────────────────────

def load_data():
    """Load baseline and conditional data, match pairs, load world reports."""
    with open(DATA_DIR / "results/republic_baseline_binary_all.json") as f:
        baseline_data = json.load(f)
    with open(DATA_DIR / "results/republic_conditional_binary_all.json") as f:
        conditional_data = json.load(f)

    baseline_by_id = {q["question_id"]: q for q in baseline_data["questions"]}

    # Load world reports per seed
    world_reports = {}
    for seed_dir in sorted(DATA_DIR.glob("questions/seed*")):
        report_path = seed_dir / "world_report" / "turn_060_report.txt"
        if report_path.exists():
            world_reports[seed_dir.name] = report_path.read_text()

    # Build matched pairs
    pairs = []
    for cq in conditional_data["questions"]:
        if cq["question_type"] != "binary":
            continue
        base_id = cq["question_id"].replace("_intervention", "")
        bq = baseline_by_id.get(base_id)
        if bq is None:
            continue

        game_id = cq["game_id"]
        if game_id not in world_reports:
            continue

        # Get Sonnet 4.5 baseline prediction
        sonnet_pred = None
        for key in bq["predictions"]:
            if "sonnet-4-5" in key:
                sonnet_pred = bq["predictions"][key]
                break
        if sonnet_pred is None or sonnet_pred["probability"] is None:
            continue

        # Get Sonnet 4.5 conditional prediction (control)
        sonnet_cond = None
        for key in cq["predictions"]:
            if "sonnet-4-5" in key:
                sonnet_cond = cq["predictions"][key]
                break
        if sonnet_cond is None or sonnet_cond["probability"] is None:
            continue

        pairs.append({
            "question_id": cq["question_id"],
            "game_id": game_id,
            "template_id": cq["template_id"],
            "conditional_text": cq["question_text"],
            "bare_text": strip_conditional_prefix(cq["question_text"]),
            "baseline_truth": bq["ground_truth"],
            "fork_truth": cq["ground_truth"],
            "baseline_p": sonnet_pred["probability"],
            "control_conditional_p": sonnet_cond["probability"],
            "world_report": world_reports[game_id],
        })

    return pairs


def stratified_sample(pairs: list[dict], n_per_template: int, rng) -> list[dict]:
    """Sample n_per_template questions from each template."""
    by_template = defaultdict(list)
    for p in pairs:
        by_template[p["template_id"]].append(p)

    sampled = []
    for template, qs in sorted(by_template.items()):
        n = min(n_per_template, len(qs))
        chosen = rng.choice(len(qs), size=n, replace=False)
        for i in chosen:
            sampled.append(qs[i])
        logger.info(f"  Template {template}: {n}/{len(qs)} sampled")

    return sampled


# ── Scoring ─────────────────────────────────────────────────────────────────

def brier(p, outcome):
    return (p - float(outcome)) ** 2


def compute_results(sampled, two_step_probs, effect_size_probs):
    """Compute all metrics and comparisons."""
    results = {
        "control": [],
        "two_step": [],
        "effect_size": [],
        "independence": [],
    }
    direction_correct = 0
    direction_wrong = 0
    direction_none = 0

    for i, q in enumerate(sampled):
        fork_truth = q["fork_truth"]
        baseline_p = q["baseline_p"]
        control_p = q["control_conditional_p"]

        results["control"].append(brier(control_p, fork_truth))
        results["independence"].append(brier(baseline_p, fork_truth))

        ts_p = two_step_probs[i]
        if ts_p is not None:
            results["two_step"].append(brier(ts_p, fork_truth))
        else:
            results["two_step"].append(None)

        es_p = effect_size_probs[i]
        if es_p is not None:
            results["effect_size"].append(brier(es_p, fork_truth))
        else:
            results["effect_size"].append(None)

        # Directionality for two-step
        if ts_p is not None:
            truth_dir = float(fork_truth) - baseline_p
            pred_dir = ts_p - baseline_p
            if abs(truth_dir) > 0.001 and abs(pred_dir) > 0.001:
                if (truth_dir > 0) == (pred_dir > 0):
                    direction_correct += 1
                else:
                    direction_wrong += 1
            else:
                direction_none += 1

    return results, direction_correct, direction_wrong, direction_none


def print_summary(results, direction_correct, direction_wrong, direction_none, n_total):
    """Print comparison table."""
    def safe_mean(xs):
        valid = [x for x in xs if x is not None]
        return sum(valid) / len(valid) if valid else float("nan")

    def safe_n(xs):
        return sum(1 for x in xs if x is not None)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    ctrl_mean = safe_mean(results["control"])
    indep_mean = safe_mean(results["independence"])

    for condition in ["control", "independence", "two_step", "effect_size"]:
        m = safe_mean(results[condition])
        n = safe_n(results[condition])
        improvement = indep_mean - m
        signal_pct = improvement / (indep_mean - ctrl_mean) * 100 if (indep_mean - ctrl_mean) != 0 else 0
        # Significance via bootstrap
        valid_pairs = [(results[condition][i], results["independence"][i])
                       for i in range(len(results[condition]))
                       if results[condition][i] is not None]
        if len(valid_pairs) > 10:
            diffs = np.array([ind - cond for cond, ind in valid_pairs])
            rng = np.random.default_rng(42)
            boot = np.array([rng.choice(diffs, len(diffs), replace=True).mean() for _ in range(5000)])
            ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
            p_count = np.sum(np.array([
                (diffs * rng.choice([-1,1], len(diffs))).mean() for _ in range(5000)
            ]) >= diffs.mean())
            p_val = p_count / 5000
            sig = "*" if p_val < 0.001 else ""
            ci_str = f"[{ci_lo:+.4f}, {ci_hi:+.4f}]"
        else:
            ci_str = "—"
            sig = ""
            p_val = 1.0

        print(f"\n{condition:15s}  Brier={m:.4f}  n={n:>3d}  "
              f"vs indep: {improvement:+.4f} {ci_str} {sig}")

    # Directionality
    decided = direction_correct + direction_wrong
    if decided > 0:
        acc = direction_correct / decided
        print(f"\nTwo-step direction accuracy: {direction_correct}/{decided} = {acc:.1%}")
    else:
        print("\nNo directional data")

    print(f"\nControl direction acc (from existing data): ~62.8% (Sonnet 4.5, Republic)")
    print("=" * 70)


# ── Main ────────────────────────────────────────────────────────────────────

async def run_experiment(n_per_template: int = 30, dry_run: bool = False):
    logger.info("Loading data...")
    pairs = load_data()
    logger.info(f"Loaded {len(pairs)} matched pairs with Sonnet 4.5 predictions")

    rng = np.random.default_rng(42)
    sampled = stratified_sample(pairs, n_per_template, rng)
    logger.info(f"Sampled {len(sampled)} questions")

    if dry_run:
        print(f"\n[DRY RUN] Would run {len(sampled)} questions × 2 conditions")
        print(f"  Two-step: {len(sampled)} × 2 calls = {len(sampled)*2} API calls")
        print(f"  Effect-size: {len(sampled)} calls")
        print(f"  Total: {len(sampled)*3} API calls")
        q = sampled[0]
        print(f"\n--- Sample direction prompt ---")
        print(build_direction_prompt(q["world_report"][:500] + "...", q["bare_text"]))
        print(f"\n--- Sample effect-size prompt ---")
        print(build_effect_size_prompt(q["world_report"][:500] + "...", q["bare_text"], q["baseline_p"]))
        return

    model = LiteLLMModel(id=MODEL_ID)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    # ── Phase 1: Direction (two-step) + Effect-size — all run concurrently ──
    logger.info(f"Phase 1: Running direction prompts + effect-size prompts concurrently...")
    start_time = time.monotonic()

    direction_tasks = []
    effect_size_tasks = []

    for i, q in enumerate(sampled):
        # Direction prompt (two-step, call 1)
        prompt = build_direction_prompt(q["world_report"], q["bare_text"])
        direction_tasks.append(call_model(model, prompt, semaphore, label=f"dir-{i}"))

        # Effect-size prompt
        prompt = build_effect_size_prompt(q["world_report"], q["bare_text"], q["baseline_p"])
        effect_size_tasks.append(call_model(model, prompt, semaphore, label=f"eff-{i}"))

    # Run all concurrently
    all_responses = await asyncio.gather(*direction_tasks, *effect_size_tasks)

    direction_responses = all_responses[:len(sampled)]
    effect_size_responses = all_responses[len(sampled):]

    phase1_elapsed = time.monotonic() - start_time
    logger.info(f"Phase 1 complete in {phase1_elapsed:.0f}s")

    # Parse directions
    directions = [parse_direction(r) if r else None for r in direction_responses]
    n_parsed = sum(1 for d in directions if d is not None)
    logger.info(f"Parsed {n_parsed}/{len(sampled)} directions")

    # Parse effect-size probabilities
    effect_size_probs = [parse_probability(r) if r else None for r in effect_size_responses]
    n_parsed_es = sum(1 for p in effect_size_probs if p is not None)
    logger.info(f"Parsed {n_parsed_es}/{len(sampled)} effect-size probabilities")

    # ── Phase 2: Magnitude (two-step, call 2) ──────────────────────────────
    logger.info("Phase 2: Running magnitude prompts...")
    start_time2 = time.monotonic()

    magnitude_tasks = []
    magnitude_indices = []  # Track which questions have valid directions

    for i, q in enumerate(sampled):
        if directions[i] is not None:
            prompt = build_magnitude_prompt(
                q["world_report"], q["bare_text"], directions[i], q["baseline_p"]
            )
            magnitude_tasks.append(call_model(model, prompt, semaphore, label=f"mag-{i}"))
            magnitude_indices.append(i)
        # If direction parsing failed, we'll use None for this question

    magnitude_responses = await asyncio.gather(*magnitude_tasks)

    phase2_elapsed = time.monotonic() - start_time2
    logger.info(f"Phase 2 complete in {phase2_elapsed:.0f}s")

    # Build two-step probabilities
    two_step_probs = [None] * len(sampled)
    for j, resp in enumerate(magnitude_responses):
        idx = magnitude_indices[j]
        if resp:
            two_step_probs[idx] = parse_probability(resp)

    n_parsed_ts = sum(1 for p in two_step_probs if p is not None)
    logger.info(f"Parsed {n_parsed_ts}/{len(sampled)} two-step final probabilities")

    # ── Compute and print results ───────────────────────────────────────────
    results, dir_correct, dir_wrong, dir_none = compute_results(
        sampled, two_step_probs, effect_size_probs
    )
    print_summary(results, dir_correct, dir_wrong, dir_none, len(sampled))

    # ── Save results ────────────────────────────────────────────────────────
    output = {
        "metadata": {
            "model": MODEL_ID,
            "n_sampled": len(sampled),
            "n_per_template": n_per_template,
            "timestamp": datetime.now().isoformat(),
            "phase1_seconds": phase1_elapsed,
            "phase2_seconds": phase2_elapsed,
        },
        "questions": [],
    }
    for i, q in enumerate(sampled):
        output["questions"].append({
            "question_id": q["question_id"],
            "template_id": q["template_id"],
            "game_id": q["game_id"],
            "conditional_text": q["conditional_text"],
            "bare_text": q["bare_text"],
            "baseline_p": q["baseline_p"],
            "control_conditional_p": q["control_conditional_p"],
            "fork_truth": q["fork_truth"],
            "baseline_truth": q["baseline_truth"],
            "direction": directions[i],
            "two_step_p": two_step_probs[i],
            "effect_size_p": effect_size_probs[i],
        })

    out_path = DATA_DIR / "results" / f"prompting_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Results saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="CivBench conditional prompting experiment")
    parser.add_argument("--n-per-template", type=int, default=30,
                        help="Questions per template (default: 30, ~300 total)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show prompts without making API calls")
    args = parser.parse_args()

    asyncio.run(run_experiment(n_per_template=args.n_per_template, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
