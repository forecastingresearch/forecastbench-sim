#!/usr/bin/env python3
"""
Generate world reports (JSON + HTML) for all recorded games.

Usage:
    python scripts/generate_world_reports.py \
        --input-dir logs/recordings \
        --output-base data/questions \
        --turn 50
"""

import argparse
import sys
from pathlib import Path

# Add src to path for world report imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from civrealm.world_reports import ReportGenerator, ReportConfig


def generate_for_game(recording_dir: Path, output_base: Path, turn: int) -> bool:
    """Generate report for a single game directory."""
    game_id = recording_dir.name
    output_dir = output_base / game_id / "world_report"
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = ReportConfig(
        recording_dir=str(recording_dir),
        output_dir=str(output_dir),
        report_turns=[turn],
        formats=["html"],  # JSON always emitted; HTML optional
        plot_style="seaborn",
        dpi=150,
    )

    generator = ReportGenerator(cfg)
    if not generator.validate_config():
        print(f"  ✗ {game_id}: invalid config or missing data")
        return False

    try:
        generator.generate_reports()
        return True
    except Exception as e:
        print(f"  ✗ {game_id}: error {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Generate world reports from recorded games"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="logs/recordings",
        help="Directory containing game recordings (default: logs/recordings)",
    )
    parser.add_argument(
        "--output-base",
        type=str,
        default="data/questions",
        help="Base directory to write reports under <output-base>/<game_id>/world_report/ (default: data/questions)",
    )
    parser.add_argument(
        "--turn",
        type=int,
        default=50,
        help="Turn number to generate report for (default: 50)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_base = Path(args.output_base)

    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}")
        return 1

    recording_dirs = sorted(d for d in input_dir.iterdir() if d.is_dir())
    if not recording_dirs:
        print(f"No recording directories found in {input_dir}")
        return 1

    print(f"Found {len(recording_dirs)} games in {input_dir}")
    print(f"Output base: {output_base}")
    print(f"Report turn: {args.turn}")
    print()

    succeeded = 0
    failed = 0

    for rec_dir in recording_dirs:
        print(f"Generating report for {rec_dir.name}...")
        ok = generate_for_game(rec_dir, output_base, args.turn)
        if ok:
            succeeded += 1
        else:
            failed += 1

    print()
    print("=" * 60)
    print(f"Done. Success: {succeeded}, Failed: {failed}")
    print(f"Reports live under {output_base}/<game_id>/world_report/")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
