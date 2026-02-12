#!/usr/bin/env python3
"""
Compute empirical difficulty scores for CivBench questions.

This script aggregates evaluation results across all models and runs,
computes per-question difficulty scores (mean Brier score), and updates
the question JSON files with the computed difficulty.

Usage:
    python scripts/compute_difficulty_scores.py
    python scripts/compute_difficulty_scores.py --dry-run
    python scripts/compute_difficulty_scores.py --min-evaluations 3
    python scripts/compute_difficulty_scores.py --report
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from civrealm.evaluation.difficulty import (
    aggregate_evaluations,
    compute_question_difficulties,
    compute_percentile_ranks,
    update_question_files,
    get_difficulty_statistics,
)


def main():
    parser = argparse.ArgumentParser(
        description="Compute empirical difficulty scores for CivBench questions"
    )
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=Path("data/evaluations/results"),
        help="Directory containing evaluation JSON files (default: data/evaluations/results)",
    )
    parser.add_argument(
        "--questions-dir",
        type=Path,
        default=Path("data/questions"),
        help="Directory containing question JSON files (default: data/questions)",
    )
    parser.add_argument(
        "--min-evaluations",
        type=int,
        default=1,
        help="Minimum number of evaluations required per question (default: 1)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without modifying files",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Print detailed statistics report",
    )

    args = parser.parse_args()

    # Validate directories
    if not args.eval_dir.exists():
        print(f"Error: Evaluation directory not found: {args.eval_dir}")
        sys.exit(1)

    if not args.questions_dir.exists():
        print(f"Error: Questions directory not found: {args.questions_dir}")
        sys.exit(1)

    print("=" * 60)
    print("CivBench Difficulty Score Computation")
    print("=" * 60)

    # Step 1: Aggregate evaluations
    print("\n[1/4] Aggregating evaluation results...")
    aggregated = aggregate_evaluations(args.eval_dir)
    print(f"      Found {len(aggregated)} unique questions across evaluations")

    # Step 2: Compute difficulties
    print("\n[2/4] Computing difficulty scores...")
    difficulties = compute_question_difficulties(
        aggregated,
        min_evaluations=args.min_evaluations,
    )
    print(f"      Computed difficulty for {len(difficulties)} questions")
    print(f"      (min {args.min_evaluations} evaluation(s) required)")

    # Step 3: Compute percentiles
    print("\n[3/4] Computing percentile ranks...")
    difficulties = compute_percentile_ranks(difficulties)

    # Step 4: Update question files
    if args.dry_run:
        print("\n[4/4] Dry run - skipping file updates")
        questions_updated, files_updated = update_question_files(
            args.questions_dir, difficulties, dry_run=True
        )
        print(f"      Would update {questions_updated} questions in {files_updated} files")
    else:
        print("\n[4/4] Updating question files...")
        questions_updated, files_updated = update_question_files(
            args.questions_dir, difficulties, dry_run=False
        )
        print(f"      Updated {questions_updated} questions in {files_updated} files")

    # Print statistics
    stats = get_difficulty_statistics(difficulties)

    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)

    if stats["count"] > 0:
        print(f"\nQuestions with difficulty scores: {stats['count']}")
        print(f"\nDifficulty score distribution (Brier score, 0-1):")
        print(f"  Min:    {stats['min']:.4f}")
        print(f"  p10:    {stats['p10']:.4f}")
        print(f"  p25:    {stats['p25']:.4f}")
        print(f"  Median: {stats['median']:.4f}")
        print(f"  Mean:   {stats['mean']:.4f}")
        print(f"  p75:    {stats['p75']:.4f}")
        print(f"  p90:    {stats['p90']:.4f}")
        print(f"  Max:    {stats['max']:.4f}")

        if args.report:
            print(f"\nModels contributing to difficulty scores:")
            for model, count in sorted(
                stats["models_contributing"].items(),
                key=lambda x: -x[1]
            ):
                print(f"  {model}: {count} questions")

            # Print sample of hardest and easiest questions
            print("\n" + "-" * 60)
            print("Sample: 5 Easiest Questions (lowest Brier = easiest)")
            print("-" * 60)
            easiest = sorted(
                difficulties.items(),
                key=lambda x: x[1].score if x[1].score else 1.0
            )[:5]
            for key, diff in easiest:
                print(f"  {key}: score={diff.score:.4f}, percentile={diff.percentile:.1f}")

            print("\n" + "-" * 60)
            print("Sample: 5 Hardest Questions (highest Brier = hardest)")
            print("-" * 60)
            hardest = sorted(
                difficulties.items(),
                key=lambda x: -(x[1].score if x[1].score else 0.0)
            )[:5]
            for key, diff in hardest:
                print(f"  {key}: score={diff.score:.4f}, percentile={diff.percentile:.1f}")
    else:
        print("\nNo questions with sufficient evaluations found.")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
