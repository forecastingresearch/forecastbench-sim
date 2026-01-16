#!/usr/bin/env python3
"""
Generate zero-turn (H0) comprehension questions from game data.

These questions have answers directly readable from the snapshot turn data,
with no prediction required. Expected Brier score should be ~0 for perfect
comprehension.

Usage:
    # Generate H0 questions for a single game (loads from data/games/)
    python scripts/generate_zero_turn_questions.py --game-id s12201 \
        --output data/questions/s12201/h0_questions.json

    # Generate H0 questions for all games in data/games/
    python scripts/generate_zero_turn_questions.py --all --snapshot-turn 50
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from civrealm.world_reports.questions.generator import QuestionGenerator
from civrealm.world_reports.questions.resolver import QuestionResolver
from civrealm.world_reports.questions.io import save_question_bank


def generate_h0_questions_for_game(
    game_id: str,
    games_dir: Path,
    snapshot_turn: int | None = None,
    resolve: bool = True,
) -> tuple[str, int, int]:
    """
    Generate H0 questions for a single game.

    Args:
        game_id: Game identifier (e.g., "s12201")
        games_dir: Directory containing game data files (data/games/)
        snapshot_turn: Turn to use as snapshot (defaults to data's turn)
        resolve: Whether to resolve questions (compute answers)

    Returns:
        Tuple of (bank, num_questions, num_true)
    """
    # Load game data from data/games/{game_id}_data.json
    data_file = games_dir / f"{game_id}_data.json"

    if not data_file.exists():
        raise FileNotFoundError(f"No game data found: {data_file}")

    with open(data_file) as f:
        game_data = json.load(f)

    data_turn = game_data.get("metadata", {}).get("turn", 50)

    # Use specified snapshot_turn or data's turn
    if snapshot_turn is None:
        snapshot_turn = data_turn
    elif snapshot_turn > data_turn:
        print(f"  Warning: snapshot_turn ({snapshot_turn}) > data turn ({data_turn}), using {data_turn}")
        snapshot_turn = data_turn

    # Generate H0 questions
    generator = QuestionGenerator()
    bank = generator.generate_zero_turn_questions(
        game_id=game_id,
        game_data=game_data,
        snapshot_turn=snapshot_turn,
    )

    # Optionally resolve questions
    if resolve:
        resolver = QuestionResolver()
        bank = resolver.resolve_batch(bank, game_data)

    # Count answers
    true_count = sum(
        1 for q in bank.questions
        if q.resolution and q.resolution.answer
    )

    return bank, len(bank.questions), true_count


def main():
    parser = argparse.ArgumentParser(
        description='Generate zero-turn (H0) comprehension questions'
    )
    parser.add_argument(
        '--game-id',
        help='Game ID (e.g., s12201). Required unless --all is specified.'
    )
    parser.add_argument(
        '--games-dir', type=Path, default=Path('data/games'),
        help='Directory containing game data files (default: data/games)'
    )
    parser.add_argument(
        '--output-dir', type=Path, default=Path('data/questions'),
        help='Output directory for question files (default: data/questions)'
    )
    parser.add_argument(
        '--output', '-o',
        help='Output JSON file path (default: {output_dir}/{game_id}/h0_questions.json)'
    )
    parser.add_argument(
        '--snapshot-turn', type=int, default=None,
        help='Turn at which data is read (default: uses turn from data)'
    )
    parser.add_argument(
        '--all', action='store_true',
        help='Generate for all games in games-dir'
    )
    parser.add_argument(
        '--no-resolve', action='store_true',
        help='Skip resolving questions (do not compute answers)'
    )
    args = parser.parse_args()

    games_dir = args.games_dir
    output_dir = args.output_dir

    if args.all:
        # Process all game data files
        if not games_dir.is_dir():
            print(f"Error: {games_dir} is not a directory")
            return 1

        game_files = list(games_dir.glob("*_data.json"))

        if not game_files:
            print(f"Error: No game data files found in {games_dir}")
            return 1

        # Extract game IDs from filenames (e.g., s12201_data.json -> s12201)
        game_ids = [f.stem.replace("_data", "") for f in game_files]

        print(f"Found {len(game_ids)} game data files")

        total_questions = 0
        total_true = 0

        for game_id in sorted(game_ids):
            try:
                bank, num_q, num_true = generate_h0_questions_for_game(
                    game_id,
                    games_dir,
                    snapshot_turn=args.snapshot_turn,
                    resolve=not args.no_resolve,
                )

                game_output_dir = output_dir / game_id
                game_output_dir.mkdir(parents=True, exist_ok=True)
                output_path = game_output_dir / "h0_questions.json"
                save_question_bank(bank, output_path)

                print(f"  {game_id}: {num_q} questions ({num_true} True)")
                total_questions += num_q
                total_true += num_true

            except Exception as e:
                print(f"  {game_id}: Error - {e}")

        print(f"\nTotal: {total_questions} questions across {len(game_ids)} games")
        if total_questions > 0:
            print(f"Base rate: {total_true/total_questions:.1%} True")

    else:
        # Process single game
        if not args.game_id:
            print("Error: --game-id is required unless --all is specified")
            return 1

        game_id = args.game_id
        print(f"Processing game: {game_id}")

        bank, num_q, num_true = generate_h0_questions_for_game(
            game_id,
            games_dir,
            snapshot_turn=args.snapshot_turn,
            resolve=not args.no_resolve,
        )

        # Determine output path
        if args.output:
            output_path = Path(args.output)
        else:
            game_output_dir = output_dir / game_id
            game_output_dir.mkdir(parents=True, exist_ok=True)
            output_path = game_output_dir / "h0_questions.json"

        save_question_bank(bank, output_path)

        print(f"Generated {num_q} H0 questions")
        print(f"Answers: {num_true} True, {num_q - num_true} False")
        print(f"Saved to: {output_path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
