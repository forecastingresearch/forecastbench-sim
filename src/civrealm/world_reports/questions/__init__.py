"""
Question generation module for creating forecasting questions from game data.

This module provides:
- Data model schema (QuestionBank, QuestionInstance, Resolution, etc.)
- Question templates (I1, I2, I3 information availability levels)
- QuestionGenerator for creating question banks
- QuestionResolver for computing answers

Example usage:
    from civrealm.world_reports.questions import (
        QuestionGenerator,
        QuestionResolver,
        QuestionBank,
    )

    # Generate questions
    generator = QuestionGenerator()
    question_bank = generator.generate_question_bank(
        game_id="seed_42",
        game_data=game_data,
        snapshot_turn=50,
    )

    # Resolve questions
    resolver = QuestionResolver()
    resolved_bank = resolver.resolve_batch(question_bank, game_data)

    # Export to JSON
    from civrealm.world_reports.questions.io import question_bank_to_json
    json_str = question_bank_to_json(resolved_bank)
"""

from .schema import (
    QuestionTemplate,
    QuestionInstance,
    Resolution,
    WorldReportConfig,
    CivilizationInfo,
    QuestionBank,
    classify_horizon,
    horizon_to_int,
    info_availability_to_int,
    calculate_difficulty,
)

from .templates import (
    ALL_TEMPLATES,
    I1_TEMPLATES,
    I2_TEMPLATES,
    I3_TEMPLATES,
    TEMPLATES_BY_ID,
    TEMPLATES_BY_INFO_AVAILABILITY,
    get_template,
    get_templates_by_info_availability,
)

from .generator import QuestionGenerator

from .resolver import QuestionResolver

from .io import (
    question_bank_to_dict,
    question_bank_to_json,
    save_question_bank,
    load_question_bank,
    dict_to_question_bank,
)


__all__ = [
    # Schema
    "QuestionTemplate",
    "QuestionInstance",
    "Resolution",
    "WorldReportConfig",
    "CivilizationInfo",
    "QuestionBank",
    "classify_horizon",
    "horizon_to_int",
    "info_availability_to_int",
    "calculate_difficulty",
    # Templates
    "ALL_TEMPLATES",
    "I1_TEMPLATES",
    "I2_TEMPLATES",
    "I3_TEMPLATES",
    "TEMPLATES_BY_ID",
    "TEMPLATES_BY_INFO_AVAILABILITY",
    "get_template",
    "get_templates_by_info_availability",
    # Generator & Resolver
    "QuestionGenerator",
    "QuestionResolver",
    # I/O
    "question_bank_to_dict",
    "question_bank_to_json",
    "save_question_bank",
    "load_question_bank",
    "dict_to_question_bank",
]
