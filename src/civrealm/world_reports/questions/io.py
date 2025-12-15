"""
I/O utilities for question bank serialization.
"""

import json
from pathlib import Path
from typing import Any

from .schema import (
    QuestionBank,
    QuestionInstance,
    Resolution,
    WorldReportConfig,
    CivilizationInfo,
)


def question_bank_to_dict(bank: QuestionBank) -> dict[str, Any]:
    """
    Convert a QuestionBank to a JSON-serializable dictionary.

    Args:
        bank: QuestionBank to convert

    Returns:
        Dictionary matching the output format specification
    """
    return {
        "game_id": bank.game_id,
        "snapshot_turn": bank.snapshot_turn,
        "game_max_turn": bank.game_max_turn,
        "generated_at": bank.generated_at,
        "world_report_config": {
            "start_turn": bank.world_report_config.start_turn,
            "sections": bank.world_report_config.sections,
            "include_all_civs": bank.world_report_config.include_all_civs,
            "territory_snapshot_turns": bank.world_report_config.territory_snapshot_turns,
        },
        "civilizations": {
            str(pid): {"name": civ.name, "nation_id": civ.nation_id}
            for pid, civ in bank.civilizations.items()
        },
        "questions": [
            _question_to_dict(q) for q in bank.questions
        ],
    }


def _question_to_dict(q: QuestionInstance) -> dict[str, Any]:
    """Convert a QuestionInstance to a dictionary."""
    result = {
        "question_id": q.question_id,
        "template_id": q.template_id,
        "resolution_turn": q.resolution_turn,
        "difficulty": {
            "horizon": q.horizon,
            "info_availability": q.info_availability,
            "composite": q.difficulty,
        },
        "parameters": q.parameters,
        "question_text": q.question_text,
    }

    if q.resolution is not None:
        result["resolution"] = _resolution_to_dict(q.resolution)

    return result


def _resolution_to_dict(r: Resolution) -> dict[str, Any]:
    """Convert a Resolution to a dictionary, omitting None values."""
    result: dict[str, Any] = {"answer": r.answer}

    # Comparative question fields
    if r.value_a is not None:
        result["value_a"] = r.value_a
    if r.value_b is not None:
        result["value_b"] = r.value_b

    # Milestone/rank fields
    if r.value_at_resolution is not None:
        result["value_at_resolution"] = r.value_at_resolution

    # Event fields
    if r.event_occurred is not None:
        result["event_occurred"] = r.event_occurred
    if r.event_details is not None:
        result["event_details"] = r.event_details

    # State check fields
    if r.state_at_resolution is not None:
        result["state_at_resolution"] = r.state_at_resolution

    return result


def question_bank_to_json(bank: QuestionBank, indent: int = 2) -> str:
    """
    Serialize a QuestionBank to JSON string.

    Args:
        bank: QuestionBank to serialize
        indent: JSON indentation level

    Returns:
        JSON string
    """
    return json.dumps(question_bank_to_dict(bank), indent=indent)


def save_question_bank(bank: QuestionBank, path: str | Path) -> None:
    """
    Save a QuestionBank to a JSON file.

    Args:
        bank: QuestionBank to save
        path: Output file path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        f.write(question_bank_to_json(bank))


def load_question_bank(path: str | Path) -> QuestionBank:
    """
    Load a QuestionBank from a JSON file.

    Args:
        path: Input file path

    Returns:
        QuestionBank instance
    """
    with open(path) as f:
        data = json.load(f)

    return dict_to_question_bank(data)


def dict_to_question_bank(data: dict[str, Any]) -> QuestionBank:
    """
    Convert a dictionary to a QuestionBank.

    Args:
        data: Dictionary loaded from JSON

    Returns:
        QuestionBank instance
    """
    # Parse world report config
    wrc_data = data.get("world_report_config", {})
    world_report_config = WorldReportConfig(
        start_turn=wrc_data.get("start_turn", 0),
        sections=wrc_data.get("sections", []),
        include_all_civs=wrc_data.get("include_all_civs", True),
        territory_snapshot_turns=wrc_data.get("territory_snapshot_turns"),
    )

    # Parse civilizations
    civilizations = {}
    for pid_str, civ_data in data.get("civilizations", {}).items():
        civilizations[int(pid_str)] = CivilizationInfo(
            name=civ_data.get("name", ""),
            nation_id=civ_data.get("nation_id", 0),
        )

    # Parse questions
    questions = [
        _dict_to_question(q_data)
        for q_data in data.get("questions", [])
    ]

    return QuestionBank(
        game_id=data.get("game_id", ""),
        snapshot_turn=data.get("snapshot_turn", 0),
        game_max_turn=data.get("game_max_turn", 0),
        world_report_config=world_report_config,
        civilizations=civilizations,
        questions=questions,
        generated_at=data.get("generated_at", ""),
    )


def _dict_to_question(data: dict[str, Any]) -> QuestionInstance:
    """Convert a dictionary to a QuestionInstance."""
    difficulty = data.get("difficulty", {})

    resolution = None
    if "resolution" in data:
        resolution = _dict_to_resolution(data["resolution"], data.get("resolution_turn", 0))

    # Handle both old "base_rate" and new "info_availability" for backwards compatibility
    info_availability = difficulty.get("info_availability") or difficulty.get("base_rate", "I1")

    return QuestionInstance(
        question_id=data.get("question_id", ""),
        template_id=data.get("template_id", ""),
        resolution_turn=data.get("resolution_turn", 0),
        horizon=difficulty.get("horizon", "H1"),
        info_availability=info_availability,
        difficulty=difficulty.get("composite", 2),
        parameters=data.get("parameters", {}),
        question_text=data.get("question_text", ""),
        resolution=resolution,
    )


def _dict_to_resolution(data: dict[str, Any], resolution_turn: int) -> Resolution:
    """Convert a dictionary to a Resolution."""
    return Resolution(
        answer=data.get("answer", False),
        resolution_turn=resolution_turn,
        value_a=data.get("value_a"),
        value_b=data.get("value_b"),
        value_at_resolution=data.get("value_at_resolution"),
        event_occurred=data.get("event_occurred"),
        event_details=data.get("event_details"),
        state_at_resolution=data.get("state_at_resolution"),
        computed_at=data.get("computed_at"),
    )
