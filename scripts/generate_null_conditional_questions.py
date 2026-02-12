#!/usr/bin/env python3
"""
Generate null conditional framed questions for the conditional framing experiment.

This script takes unconditional questions and reframes them in conditional format
but with a negation - testing whether the conditional *framing* alone affects
Brier scores, independent of any actual intervention.

Example transformations:
- Original: "Will Egypt have more technologies than Rome at turn 120?"
- Null conditional: "If Egypt does NOT receive +500 gold next turn, would Egypt
  have more technologies than Rome at turn 120?"

The ground truth remains the same since no actual intervention happens.

Usage:
    python scripts/generate_null_conditional_questions.py --output data/conditional/null_conditional

    # Then evaluate with:
    python scripts/evaluate_llm_forecasts_parallel.py \
        --data-dir data/conditional/null_conditional \
        --models anthropic/claude-opus-4-5-20251101 \
        -n 10
"""

import argparse
import json
import os
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# Null conditional framing templates
# These mirror the actual conditional intervention format exactly, but with negation
# Intervention: "If X received +500 gold next turn, would X have..."
# Null conditional: "If X does NOT receive +500 gold next turn, would X have..."
NULL_CONDITIONAL_TEMPLATES = {
    "treasury_comparative": (
        "If {civ_a} does NOT receive +500 gold next turn, "
        "would {civ_a} have a larger treasury than {civ_b} at turn {resolution_turn}?"
    ),
    "score_comparative": (
        "If {civ_a} does NOT receive +500 gold next turn, "
        "would {civ_a} have a higher score than {civ_b} at turn {resolution_turn}?"
    ),
    "tech_comparative": (
        "If {civ_a} does NOT receive +500 gold next turn, "
        "would {civ_a} have more technologies than {civ_b} at turn {resolution_turn}?"
    ),
    "population_comparative": (
        "If {civ_a} does NOT receive +500 gold next turn, "
        "would {civ_a} have a larger total population than {civ_b} at turn {resolution_turn}?"
    ),
    "city_count_comparative": (
        "If {civ_a} does NOT receive +500 gold next turn, "
        "would {civ_a} have more cities than {civ_b} at turn {resolution_turn}?"
    ),
    "territory_comparative": (
        "If {civ_a} does NOT receive +500 gold next turn, "
        "would {civ_a} control more tiles than {civ_b} at turn {resolution_turn}?"
    ),
    "score_rank_1": (
        "If {civ} does NOT receive +500 gold next turn, "
        "would {civ} be ranked #1 at turn {resolution_turn}?"
    ),
    "tech_discovered": (
        "If {civ} does NOT receive +500 gold next turn, "
        "would {civ} have discovered {tech_name} by turn {resolution_turn}?"
    ),
    "wonder_completed": (
        "If no civilization receives +500 gold next turn, "
        "would {wonder_name} be completed by any civilization by turn {resolution_turn}?"
    ),
    "government_at": (
        "If {civ} does NOT receive +500 gold next turn, "
        "would {civ} be in {government_type} at turn {resolution_turn}?"
    ),
}


def transform_question_to_null_conditional(question: dict) -> dict:
    """
    Transform an unconditional question to null conditional framing.

    The ground truth remains the same - we're only changing the framing.
    """
    q = deepcopy(question)
    template_id = q.get("template_id", "")
    params = q.get("parameters", {})

    # Get the null conditional template
    null_template = NULL_CONDITIONAL_TEMPLATES.get(template_id)

    if null_template is None:
        # For templates without a specific null conditional version,
        # use a generic prefix
        original_text = q.get("question_text", "")
        # Convert "Will X" to "Given no intervention, will X"
        if original_text.startswith("Will "):
            q["question_text"] = (
                "Given that no interventions will occur, w" + original_text[1:]
            )
        else:
            q["question_text"] = f"Given no intervention, {original_text}"
    else:
        # Format with the specific template
        try:
            q["question_text"] = null_template.format(**params)
        except KeyError as e:
            # Fallback if parameters don't match
            print(f"Warning: Missing parameter {e} for template {template_id}, using generic framing")
            original_text = q.get("question_text", "")
            q["question_text"] = f"Given no intervention, {original_text}"

    # Mark as null conditional for tracking
    q["template_id"] = f"null_conditional_{template_id}"
    q["question_id"] = f"null_{q.get('question_id', '')}"

    # Add metadata about the transformation
    q["null_conditional_metadata"] = {
        "original_template_id": template_id,
        "framing_type": "null_conditional",
        "transformed_at": datetime.now().isoformat() + "Z",
    }

    return q


def transform_question_bank(bank: dict) -> dict:
    """Transform an entire question bank to null conditional framing."""
    new_bank = deepcopy(bank)

    # Transform all questions
    new_bank["questions"] = [
        transform_question_to_null_conditional(q)
        for q in bank.get("questions", [])
    ]

    # Update metadata
    new_bank["generated_at"] = datetime.now().isoformat() + "Z"
    new_bank["null_conditional_metadata"] = {
        "source_game_id": bank.get("game_id"),
        "framing_experiment": "null_conditional",
        "description": "Unconditional questions reframed with null conditional framing",
    }

    return new_bank


def process_questions_directory(
    input_dir: Path,
    output_dir: Path,
    verbose: bool = False,
) -> tuple[int, int]:
    """
    Process all question files in a directory.

    Returns (files_processed, questions_transformed).
    """
    files_processed = 0
    questions_transformed = 0

    # Find all questions.json files
    for questions_file in input_dir.rglob("questions.json"):
        # Skip conditional questions
        if "conditional" in questions_file.name:
            continue

        # Compute relative path for output
        rel_path = questions_file.relative_to(input_dir)
        output_path = output_dir / rel_path

        # Load original bank
        with open(questions_file) as f:
            bank = json.load(f)

        # Transform
        new_bank = transform_question_bank(bank)

        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(new_bank, f, indent=2)

        files_processed += 1
        questions_transformed += len(new_bank.get("questions", []))

        if verbose:
            print(f"  {questions_file} -> {output_path} ({len(new_bank.get('questions', []))} questions)")

    # Also copy world_report directories (needed for evaluation context)
    # Copy the entire world_report folder for each game
    import shutil
    for world_report_dir in input_dir.rglob("world_report"):
        if world_report_dir.is_dir():
            rel_path = world_report_dir.relative_to(input_dir)
            output_wr = output_dir / rel_path
            if not output_wr.exists():
                shutil.copytree(world_report_dir, output_wr)
                if verbose:
                    print(f"  Copied world_report: {world_report_dir} -> {output_wr}")

    return files_processed, questions_transformed


def main():
    parser = argparse.ArgumentParser(
        description="Generate null conditional framed questions for framing experiment"
    )
    parser.add_argument(
        "--input-dir", type=str, default="data/questions",
        help="Input directory containing question banks"
    )
    parser.add_argument(
        "--output-dir", "-o", type=str, default="data/conditional/null_conditional",
        help="Output directory for null conditional questions"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print details about each file processed"
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    print(f"Generating null conditional questions...")
    print(f"  Input:  {input_dir}")
    print(f"  Output: {output_dir}")

    files, questions = process_questions_directory(
        input_dir, output_dir, verbose=args.verbose
    )

    print(f"\nDone! Processed {files} files, transformed {questions} questions.")
    print(f"\nTo evaluate, run:")
    print(f"  python scripts/evaluate_llm_forecasts_parallel.py \\")
    print(f"    --data-dir {output_dir} \\")
    print(f"    --models anthropic/claude-opus-4-5-20251101 \\")
    print(f"    -n 10")


if __name__ == "__main__":
    main()
