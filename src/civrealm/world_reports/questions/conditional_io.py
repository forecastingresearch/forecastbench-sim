"""
I/O utilities for conditional question bank serialization.

Provides conversion to QuestionInstances for compatibility with the
existing evaluation framework.
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
    classify_horizon,
)
from .conditional_schema import (
    Condition,
    ConditionalQuestion,
    ConditionalResult,
    ForkOutcome,
    ConditionalQuestionBank,
)


def conditional_bank_to_dict(bank: ConditionalQuestionBank) -> dict[str, Any]:
    """
    Convert a ConditionalQuestionBank to a JSON-serializable dictionary.

    Args:
        bank: ConditionalQuestionBank to convert

    Returns:
        Dictionary for JSON serialization
    """
    return {
        "game_id": bank.game_id,
        "checkpoint_turn": bank.checkpoint_turn,
        "end_turn": bank.end_turn,
        "generated_at": bank.generated_at,
        "conditions": [_condition_to_dict(c) for c in bank.conditions],
        "questions": [_conditional_question_to_dict(q) for q in bank.questions],
        "results": {
            cid: _conditional_result_to_dict(r)
            for cid, r in bank.results.items()
        },
    }


def _condition_to_dict(c: Condition) -> dict[str, Any]:
    """Convert a Condition to a dictionary."""
    return {
        "condition_id": c.condition_id,
        "condition_type": c.condition_type,
        "player_id": c.player_id,
        "value": c.value,
        "description": c.description,
    }


def _conditional_question_to_dict(q: ConditionalQuestion) -> dict[str, Any]:
    """Convert a ConditionalQuestion to a dictionary."""
    return {
        "conditional_id": q.conditional_id,
        "condition": _condition_to_dict(q.condition),
        "target_template_id": q.target_template_id,
        "target_parameters": q.target_parameters,
        "checkpoint_turn": q.checkpoint_turn,
        "resolution_turn": q.resolution_turn,
    }


def _fork_outcome_to_dict(o: ForkOutcome) -> dict[str, Any]:
    """Convert a ForkOutcome to a dictionary."""
    result = {
        "fork_name": o.fork_name,
        "success": o.success,
        "final_turn": o.final_turn,
        "player_states": o.player_states,
    }
    if o.error is not None:
        result["error"] = o.error
    return result


def _conditional_result_to_dict(r: ConditionalResult) -> dict[str, Any]:
    """Convert a ConditionalResult to a dictionary."""
    return {
        "conditional_id": r.conditional_id,
        "control_outcome": _fork_outcome_to_dict(r.control_outcome),
        "intervention_outcome": _fork_outcome_to_dict(r.intervention_outcome),
        "answer_control": r.answer_control,
        "answer_intervention": r.answer_intervention,
        "conditional_effect": r.conditional_effect,
        "computed_at": r.computed_at,
    }


def conditional_bank_to_json(bank: ConditionalQuestionBank, indent: int = 2) -> str:
    """
    Serialize a ConditionalQuestionBank to JSON string.

    Args:
        bank: ConditionalQuestionBank to serialize
        indent: JSON indentation level

    Returns:
        JSON string
    """
    return json.dumps(conditional_bank_to_dict(bank), indent=indent)


def save_conditional_bank(bank: ConditionalQuestionBank, path: str | Path) -> None:
    """
    Save a ConditionalQuestionBank to a JSON file.

    Args:
        bank: ConditionalQuestionBank to save
        path: Output file path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        f.write(conditional_bank_to_json(bank))


def load_conditional_bank(path: str | Path) -> ConditionalQuestionBank:
    """
    Load a ConditionalQuestionBank from a JSON file.

    Args:
        path: Input file path

    Returns:
        ConditionalQuestionBank instance
    """
    with open(path) as f:
        data = json.load(f)

    return dict_to_conditional_bank(data)


def dict_to_conditional_bank(data: dict[str, Any]) -> ConditionalQuestionBank:
    """
    Convert a dictionary to a ConditionalQuestionBank.

    Args:
        data: Dictionary loaded from JSON

    Returns:
        ConditionalQuestionBank instance
    """
    conditions = [_dict_to_condition(c) for c in data.get("conditions", [])]
    conditions_by_id = {c.condition_id: c for c in conditions}

    questions = []
    for q_data in data.get("questions", []):
        # Condition may be inline or reference by ID
        cond_data = q_data.get("condition", {})
        if isinstance(cond_data, str):
            condition = conditions_by_id.get(cond_data)
        else:
            condition = _dict_to_condition(cond_data)

        questions.append(ConditionalQuestion(
            conditional_id=q_data.get("conditional_id", ""),
            condition=condition,
            target_template_id=q_data.get("target_template_id", ""),
            target_parameters=q_data.get("target_parameters", {}),
            checkpoint_turn=q_data.get("checkpoint_turn", 0),
            resolution_turn=q_data.get("resolution_turn", 0),
        ))

    results = {}
    for cid, r_data in data.get("results", {}).items():
        results[cid] = _dict_to_conditional_result(r_data)

    return ConditionalQuestionBank(
        game_id=data.get("game_id", ""),
        checkpoint_turn=data.get("checkpoint_turn", 0),
        end_turn=data.get("end_turn", 0),
        conditions=conditions,
        questions=questions,
        results=results,
        generated_at=data.get("generated_at", ""),
    )


def _dict_to_condition(data: dict[str, Any]) -> Condition:
    """Convert a dictionary to a Condition."""
    return Condition(
        condition_id=data.get("condition_id", ""),
        condition_type=data.get("condition_type", "gold"),
        player_id=data.get("player_id", 0),
        value=data.get("value", 0),
        description=data.get("description", ""),
    )


def _dict_to_fork_outcome(data: dict[str, Any]) -> ForkOutcome:
    """Convert a dictionary to a ForkOutcome."""
    return ForkOutcome(
        fork_name=data.get("fork_name", ""),
        success=data.get("success", False),
        final_turn=data.get("final_turn", 0),
        player_states={int(k): v for k, v in data.get("player_states", {}).items()},
        error=data.get("error"),
    )


def _dict_to_conditional_result(data: dict[str, Any]) -> ConditionalResult:
    """Convert a dictionary to a ConditionalResult."""
    return ConditionalResult(
        conditional_id=data.get("conditional_id", ""),
        control_outcome=_dict_to_fork_outcome(data.get("control_outcome", {})),
        intervention_outcome=_dict_to_fork_outcome(data.get("intervention_outcome", {})),
        answer_control=data.get("answer_control"),
        answer_intervention=data.get("answer_intervention"),
        conditional_effect=data.get("conditional_effect"),
        computed_at=data.get("computed_at", ""),
    )


# ============================================================================
# Conversion to QuestionInstances for evaluation compatibility
# ============================================================================

def to_question_instances(
    bank: ConditionalQuestionBank,
    civilizations: dict[int, CivilizationInfo] | None = None,
) -> list[QuestionInstance]:
    """
    Convert resolved conditional questions to QuestionInstances.

    Each conditional question produces TWO QuestionInstances:
    1. Intervention question (condition applied)
    2. Control question (no condition)

    These can then be evaluated using the standard evaluation pipeline.

    Args:
        bank: ConditionalQuestionBank with results populated
        civilizations: Optional civ info for parameters

    Returns:
        List of QuestionInstances ready for evaluation
    """
    instances = []

    for cq in bank.questions:
        result = bank.results.get(cq.conditional_id)
        if result is None:
            continue  # No results yet

        # Get condition details
        cond = cq.condition
        civ_name = _get_civ_name(cond.player_id, civilizations)

        # Build base parameters
        base_params = dict(cq.target_parameters)
        base_params["condition_type"] = cond.condition_type
        base_params["condition_player"] = cond.player_id
        base_params["condition_value"] = cond.value

        # Compute horizon
        horizon = classify_horizon(cq.checkpoint_turn, cq.resolution_turn)

        # Intervention question
        if result.intervention_outcome.success and result.answer_intervention is not None:
            intervention_text = _build_conditional_question_text(
                cond, cq.target_template_id, cq.target_parameters,
                cq.resolution_turn, is_intervention=True, civ_name=civ_name
            )

            instances.append(QuestionInstance(
                question_id=f"{cq.conditional_id}_intervention",
                template_id=f"conditional_{cq.target_template_id}",
                resolution_turn=cq.resolution_turn,
                horizon=horizon,
                parameters=base_params,
                question_text=intervention_text,
                resolution=Resolution(
                    answer=result.answer_intervention,
                    resolution_turn=cq.resolution_turn,
                    computed_at=result.computed_at,
                ),
            ))

        # Control question
        if result.control_outcome.success and result.answer_control is not None:
            control_text = _build_conditional_question_text(
                cond, cq.target_template_id, cq.target_parameters,
                cq.resolution_turn, is_intervention=False, civ_name=civ_name
            )

            instances.append(QuestionInstance(
                question_id=f"{cq.conditional_id}_control",
                template_id=f"conditional_{cq.target_template_id}",
                resolution_turn=cq.resolution_turn,
                horizon=horizon,
                parameters=base_params,
                question_text=control_text,
                resolution=Resolution(
                    answer=result.answer_control,
                    resolution_turn=cq.resolution_turn,
                    computed_at=result.computed_at,
                ),
            ))

    return instances


def _get_civ_name(player_id: int, civilizations: dict[int, CivilizationInfo] | None) -> str:
    """Get civilization name for a player."""
    if civilizations and player_id in civilizations:
        return civilizations[player_id].name
    return f"Player {player_id}"


def _build_conditional_question_text(
    condition: Condition,
    target_template_id: str,
    target_params: dict[str, Any],
    resolution_turn: int,
    is_intervention: bool,
    civ_name: str,
) -> str:
    """
    Build the question text for a conditional question.

    For intervention: "If X received Y next turn, would Z?"
    For control: "Without any X bonus, would Z?"
    """
    # Build condition description
    if condition.condition_type == "gold":
        if is_intervention:
            condition_phrase = f"If {civ_name} received {condition.value} gold next turn"
        else:
            condition_phrase = f"Without any gold bonus"
    elif condition.condition_type == "gold_add":
        if is_intervention:
            condition_phrase = f"If {civ_name} received +{condition.value} gold next turn"
        else:
            condition_phrase = f"Without any gold bonus"
    elif condition.condition_type == "government":
        if is_intervention:
            condition_phrase = f"If {civ_name} changed to {condition.value} government"
        else:
            condition_phrase = f"Without changing government"
    elif condition.condition_type == "tech":
        if is_intervention:
            condition_phrase = f"If {civ_name} were granted technology {condition.value}"
        else:
            condition_phrase = f"Without any technology grant"
    else:
        condition_phrase = condition.description

    # Build target question based on template
    target_question = _build_target_question(target_template_id, target_params, resolution_turn)

    return f"{condition_phrase}, {target_question}"


def _build_target_question(
    template_id: str,
    params: dict[str, Any],
    resolution_turn: int,
) -> str:
    """Build the target question portion of conditional text."""
    civ_a = params.get("civ_a", params.get("civ", "the civilization"))
    civ_b = params.get("civ_b", "the other civilization")

    if template_id == "treasury_comparative":
        return f"would {civ_a} have a larger treasury than {civ_b} at turn {resolution_turn}?"
    elif template_id == "score_comparative":
        return f"would {civ_a} have a higher score than {civ_b} at turn {resolution_turn}?"
    elif template_id == "tech_comparative":
        return f"would {civ_a} have more technologies than {civ_b} at turn {resolution_turn}?"
    elif template_id == "population_comparative":
        return f"would {civ_a} have a larger population than {civ_b} at turn {resolution_turn}?"
    elif template_id == "score_rank_1":
        return f"would {civ_a} be ranked #1 at turn {resolution_turn}?"
    elif template_id == "government_at":
        gov_type = params.get("government_type", "Republic")
        return f"would {civ_a} be in {gov_type} at turn {resolution_turn}?"
    elif template_id == "tech_discovered":
        tech_name = params.get("tech_name", "the technology")
        return f"would {civ_a} have discovered {tech_name} by turn {resolution_turn}?"
    else:
        return f"would the target outcome occur at turn {resolution_turn}?"


def to_question_bank(
    conditional_bank: ConditionalQuestionBank,
    civilizations: dict[int, CivilizationInfo] | None = None,
) -> QuestionBank:
    """
    Convert a ConditionalQuestionBank to a standard QuestionBank.

    This enables using the same save/load infrastructure and evaluation
    pipeline as unconditional questions.

    Args:
        conditional_bank: ConditionalQuestionBank with results
        civilizations: Optional civ info

    Returns:
        QuestionBank containing conditional QuestionInstances
    """
    from datetime import datetime

    instances = to_question_instances(conditional_bank, civilizations)

    return QuestionBank(
        game_id=conditional_bank.game_id,
        snapshot_turn=conditional_bank.checkpoint_turn,
        game_max_turn=conditional_bank.end_turn,
        world_report_config=WorldReportConfig(),
        civilizations=civilizations or {},
        questions=instances,
        generated_at=datetime.now().isoformat() + "Z",
    )


def save_as_question_bank(
    conditional_bank: ConditionalQuestionBank,
    path: str | Path,
    civilizations: dict[int, CivilizationInfo] | None = None,
) -> None:
    """
    Save conditional questions as a standard questions.json file.

    This enables the existing evaluation framework to load and process
    conditional questions without modification.

    Args:
        conditional_bank: ConditionalQuestionBank with results
        path: Output path (e.g., 'conditional_questions.json')
        civilizations: Optional civ info
    """
    from .io import save_question_bank

    qbank = to_question_bank(conditional_bank, civilizations)
    save_question_bank(qbank, path)
