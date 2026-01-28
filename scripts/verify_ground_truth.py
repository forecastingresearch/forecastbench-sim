#!/usr/bin/env python3
"""
Verify that question ground truths match the world report data.

This script parses world reports and compares extracted facts against
the resolved ground truth in questions to detect mismatches.

Usage:
    python scripts/verify_ground_truth.py --game-id s122
    python scripts/verify_ground_truth.py --all
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class WorldReportParser:
    """Parse TXT world reports to extract facts."""

    def __init__(self, report_text: str):
        self.text = report_text
        self.snapshot_turn = self._extract_snapshot_turn()

    def _extract_snapshot_turn(self) -> int:
        """Extract the snapshot turn from the report header."""
        match = re.search(r"Snapshot turn:\s*(\d+)", self.text)
        return int(match.group(1)) if match else 60

    def get_diplomatic_state(self, player_a: int, player_b: int, turn: int = None) -> str | None:
        """
        Get diplomatic state between two players at a given turn.

        Returns the state (War, Peace, Alliance, etc.) or None if not found.
        """
        if turn is None:
            turn = self.snapshot_turn

        # Find the diplomacy section for this pair
        # Pattern: PAIR X-Name vs Y-Name
        pattern = rf"PAIR {player_a}-\w+ vs {player_b}-\w+\s*\n\s*State:\s*([^\n]+)"
        match = re.search(pattern, self.text)

        if not match:
            # Try reverse order
            pattern = rf"PAIR {player_b}-\w+ vs {player_a}-\w+\s*\n\s*State:\s*([^\n]+)"
            match = re.search(pattern, self.text)

        if not match:
            return None

        # Parse state timeline: "1 Never met; 10 Cease-fire; 23 Armistice; 39 Peace; 42 Alliance"
        timeline_str = match.group(1)
        state_changes = []

        for entry in timeline_str.split(";"):
            entry = entry.strip()
            parts = entry.split(maxsplit=1)
            if len(parts) == 2:
                try:
                    change_turn = int(parts[0])
                    state = parts[1].strip()
                    state_changes.append((change_turn, state))
                except ValueError:
                    continue

        # Find state at the target turn
        current_state = None
        for change_turn, state in sorted(state_changes):
            if change_turn <= turn:
                current_state = state
            else:
                break

        return current_state

    def get_score(self, player_id: int, turn: int = None) -> int | None:
        """Get player score at snapshot turn."""
        if turn is None:
            turn = self.snapshot_turn

        # Parse CURRENT STATE table
        # ID | Name | Score | ...
        pattern = rf"^{player_id}\s+\|\s+\w+\s+\|\s+(-?\d+|NA)\s+\|"
        match = re.search(pattern, self.text, re.MULTILINE)

        if match:
            score_str = match.group(1)
            if score_str != "NA":
                return int(score_str)
        return None

    def get_tech_count(self, player_id: int) -> int | None:
        """Get technology count for a player at snapshot turn."""
        # Parse CURRENT STATE table
        # ID | Name | Score | Rank | Government | Treasury | Population | Techs | ...
        pattern = rf"^{player_id}\s+\|\s+[\w\s]+\|\s+[^\|]+\|\s+[^\|]+\|\s+[^\|]+\|\s+[^\|]+\|\s+[^\|]+\|\s+(\d+|None)\s+\|"
        match = re.search(pattern, self.text, re.MULTILINE)

        if match:
            tech_str = match.group(1)
            if tech_str != "None":
                return int(tech_str)
        return None

    def get_wonder_count(self, player_id: int) -> int | None:
        """Get wonder count for a player at snapshot turn."""
        # Parse CURRENT STATE table
        # ID | Name | Score | Rank | Government | Treasury | Population | Techs | Wonders | ...
        pattern = rf"^{player_id}\s+\|\s+[\w\s]+\|\s+[^\|]+\|\s+[^\|]+\|\s+[^\|]+\|\s+[^\|]+\|\s+[^\|]+\|\s+[^\|]+\|\s+(\d+|None)\s+\|"
        match = re.search(pattern, self.text, re.MULTILINE)

        if match:
            wonder_str = match.group(1)
            if wonder_str != "None":
                return int(wonder_str)
        return None

    def get_city_count(self, player_id: int) -> int | None:
        """Get city count for a player at snapshot turn."""
        # Parse CURRENT STATE table - Cities column (9th value after ID)
        # Try to find the row and parse all values
        lines = self.text.split("\n")
        for line in lines:
            if line.startswith(f"{player_id}  |") or line.startswith(f"{player_id} |"):
                # Parse the row
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 9:
                    city_str = parts[8]  # Cities is the 9th column (0-indexed: 8)
                    if city_str and city_str != "None":
                        try:
                            return int(city_str)
                        except ValueError:
                            pass
        return None

    def get_government(self, player_id: int, turn: int = None) -> str | None:
        """Get government type for a player at a given turn."""
        if turn is None:
            turn = self.snapshot_turn

        # Parse GOVERNMENT TIMELINE section
        pattern = rf"^{player_id}\s+[\w\s]+:\s+([^\n]+)"
        match = re.search(pattern, self.text, re.MULTILINE)

        if not match:
            return None

        timeline_str = match.group(1)
        if "NA" in timeline_str:
            return None

        # Parse: "1 Despotism; 20 Anarchy; 37 Despotism"
        gov_changes = []
        for entry in timeline_str.split(";"):
            entry = entry.strip()
            parts = entry.split(maxsplit=1)
            if len(parts) == 2:
                try:
                    change_turn = int(parts[0])
                    gov = parts[1].strip()
                    gov_changes.append((change_turn, gov))
                except ValueError:
                    continue

        # Find government at target turn
        current_gov = None
        for change_turn, gov in sorted(gov_changes):
            if change_turn <= turn:
                current_gov = gov
            else:
                break

        return current_gov


def verify_question(
    question: dict,
    parser: WorldReportParser,
    snapshot_turn: int,
    h0_only: bool = True,
) -> dict | None:
    """
    Verify a single question's ground truth against world report.

    Args:
        question: Question dict with parameters and resolution
        parser: WorldReportParser for the game's world report
        snapshot_turn: The snapshot turn (usually 60)
        h0_only: If True, only verify H0 questions (which can be validated
                 from snapshot data). Non-H0 questions ask about future turns
                 that aren't in the world report.

    Returns a dict with mismatch details if there's a discrepancy, None if matches.
    """
    template_id = question.get("template_id", "")
    params = question.get("parameters", {})
    resolution = question.get("resolution", {})
    ground_truth = resolution.get("answer")
    resolution_turn = question.get("resolution_turn", snapshot_turn)

    # Handle H0 templates
    is_h0 = template_id.startswith("h0_")
    base_template = template_id[3:] if is_h0 else template_id

    # Only verify H0 questions if h0_only is True
    # Non-H0 questions ask about future turns that aren't in the world report
    if h0_only and not is_h0:
        return None

    result = {
        "question_id": question.get("question_id"),
        "template_id": template_id,
        "ground_truth": ground_truth,
        "resolution_turn": resolution_turn,
    }

    # Verify based on template type
    if base_template in ["alliance_dyad", "at_war_dyad"]:
        player_a = params.get("player_id_a")
        player_b = params.get("player_id_b")

        if player_a is None or player_b is None:
            return None

        report_state = parser.get_diplomatic_state(player_a, player_b, resolution_turn)
        target_state = "Alliance" if base_template == "alliance_dyad" else "War"

        report_answer = report_state == target_state

        if report_answer != ground_truth:
            result["report_state"] = report_state
            result["expected_state"] = target_state
            result["report_answer"] = report_answer
            result["mismatch_type"] = "diplomatic_state"
            return result

    elif base_template == "at_war_any":
        player_id = params.get("player_id")
        if player_id is None:
            return None

        # Check all pairs involving this player
        report_at_war = False
        for other_id in range(8):  # Check all potential players
            if other_id == player_id:
                continue
            state = parser.get_diplomatic_state(player_id, other_id, resolution_turn)
            if state == "War":
                report_at_war = True
                break

        if report_at_war != ground_truth:
            result["report_answer"] = report_at_war
            result["mismatch_type"] = "at_war_any"
            return result

    elif base_template == "score_comparative":
        player_a = params.get("player_id_a")
        player_b = params.get("player_id_b")

        if player_a is None or player_b is None:
            return None

        score_a = parser.get_score(player_a, resolution_turn)
        score_b = parser.get_score(player_b, resolution_turn)

        if score_a is not None and score_b is not None:
            report_answer = score_a > score_b

            if report_answer != ground_truth:
                result["score_a"] = score_a
                result["score_b"] = score_b
                result["report_answer"] = report_answer
                result["mismatch_type"] = "score_comparative"
                return result

    elif base_template == "tech_comparative":
        player_a = params.get("player_id_a")
        player_b = params.get("player_id_b")

        if player_a is None or player_b is None:
            return None

        tech_a = parser.get_tech_count(player_a)
        tech_b = parser.get_tech_count(player_b)

        if tech_a is not None and tech_b is not None:
            report_answer = tech_a > tech_b

            if report_answer != ground_truth:
                result["tech_a"] = tech_a
                result["tech_b"] = tech_b
                result["report_answer"] = report_answer
                result["mismatch_type"] = "tech_comparative"
                return result

    # No mismatch detected
    return None


def load_world_report_parsers(questions_dir: Path) -> dict[str, WorldReportParser]:
    """Load and parse world reports for all games."""
    parsers = {}

    for game_dir in questions_dir.iterdir():
        if not game_dir.is_dir() or not game_dir.name.startswith("s"):
            continue

        report_file = game_dir / "world_report" / "turn_060_report.txt"
        if report_file.exists():
            with open(report_file) as f:
                parsers[game_dir.name] = WorldReportParser(f.read())

    return parsers


def verify_all_questions(
    questions_file: Path,
    questions_dir: Path,
    game_filter: str | None = None,
    h0_only: bool = True,
) -> list[dict]:
    """Verify all questions against their world reports.

    Args:
        questions_file: Path to combined questions JSON file
        questions_dir: Directory containing game world reports
        game_filter: Optional game ID to filter to
        h0_only: If True, only verify H0 questions (default). Non-H0 questions
                 ask about future turns that aren't in the world report.
    """
    # Load all world report parsers
    parsers = load_world_report_parsers(questions_dir)
    print(f"Loaded {len(parsers)} world reports")

    # Load questions
    with open(questions_file) as f:
        data = json.load(f)

    questions = data.get("questions", [])
    snapshot_turn = data.get("metadata", {}).get("snapshot_turn", 60)

    print(f"Loaded {len(questions)} questions")
    if h0_only:
        print("Verifying H0 questions only (snapshot turn data)")
    else:
        print("Warning: Verifying all questions (non-H0 will show false positives)")

    # Verify each question
    mismatches = []
    verified_count = 0
    skipped_count = 0

    for q in questions:
        game_id = q.get("parameters", {}).get("game_id")

        if not game_id:
            skipped_count += 1
            continue

        if game_filter and game_id != game_filter:
            continue

        if game_id not in parsers:
            skipped_count += 1
            continue

        parser = parsers[game_id]
        mismatch = verify_question(q, parser, snapshot_turn, h0_only=h0_only)

        if mismatch:
            mismatch["game_id"] = game_id
            mismatches.append(mismatch)
            verified_count += 1
        elif q.get("template_id", "").startswith("h0_") or not h0_only:
            verified_count += 1

    print(f"Verified {verified_count} questions, skipped {skipped_count}")

    return mismatches


def main():
    parser = argparse.ArgumentParser(
        description="Verify question ground truths against world reports"
    )
    parser.add_argument(
        "--game-id",
        help="Filter to specific game ID (e.g., s122)",
    )
    parser.add_argument(
        "--questions-file",
        type=Path,
        default=Path("data/questions/questions_all.json"),
        help="Combined questions file",
    )
    parser.add_argument(
        "--questions-dir",
        type=Path,
        default=Path("data/questions"),
        help="Directory containing game world reports",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file for mismatch report (JSON)",
    )
    parser.add_argument(
        "--all-questions",
        action="store_true",
        help="Verify all questions, not just H0 (will show false positives for non-H0)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Ground Truth Verification")
    print("=" * 70)

    print(f"\nQuestions file: {args.questions_file}")
    print(f"World reports dir: {args.questions_dir}")
    if args.game_id:
        print(f"Filtering to game: {args.game_id}")

    all_mismatches = verify_all_questions(
        questions_file=args.questions_file,
        questions_dir=args.questions_dir,
        game_filter=args.game_id,
        h0_only=not args.all_questions,
    )

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    if all_mismatches:
        print(f"\nTotal mismatches found: {len(all_mismatches)}")

        # Group by type
        by_type = defaultdict(list)
        for m in all_mismatches:
            by_type[m.get("mismatch_type", "unknown")].append(m)

        print("\nMismatches by type:")
        for mtype, items in sorted(by_type.items()):
            print(f"  {mtype}: {len(items)}")

        # Group by template
        by_template = defaultdict(list)
        for m in all_mismatches:
            by_template[m["template_id"]].append(m)

        print("\nMismatches by template:")
        for template, items in sorted(by_template.items()):
            print(f"  {template}: {len(items)}")

    else:
        print("\nNo mismatches found - all ground truths verified!")

    # Save output
    if args.output and all_mismatches:
        with open(args.output, "w") as f:
            json.dump({
                "total_mismatches": len(all_mismatches),
                "mismatches": all_mismatches,
            }, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    return 1 if all_mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
