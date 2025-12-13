"""
Question generation module for creating forecasting questions from game data.

This module provides:
- Data model schema (QuestionBank, QuestionInstance, Resolution, etc.)
- Question templates (B1, B2, B3 signal types)
- Configurable thresholds
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
    ThresholdConfig,
    CivilizationInfo,
    QuestionBank,
    classify_horizon,
    horizon_to_int,
    base_rate_to_int,
    calculate_difficulty,
)

from .templates import (
    ALL_TEMPLATES,
    B1_TEMPLATES,
    B2_TEMPLATES,
    B3_TEMPLATES,
    TEMPLATES_BY_ID,
    TEMPLATES_BY_SIGNAL_TYPE,
    get_template,
    get_templates_by_signal_type,
)

from .thresholds import (
    DEFAULT_THRESHOLDS,
    get_default_thresholds,
    get_thresholds_for_signal,
    select_threshold,
    create_threshold_config,
    select_threshold_for_rate,
    select_growth_threshold_for_rate,
)

from .signal_statistics import (
    get_threshold_for_rate,
    get_growth_threshold_for_rate,
    get_event_probability,
    SIGNAL_STATS,
    GROWTH_STATS,
    EVENT_PROBABILITIES,
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
    "ThresholdConfig",
    "CivilizationInfo",
    "QuestionBank",
    "classify_horizon",
    "horizon_to_int",
    "base_rate_to_int",
    "calculate_difficulty",
    # Templates
    "ALL_TEMPLATES",
    "B1_TEMPLATES",
    "B2_TEMPLATES",
    "B3_TEMPLATES",
    "TEMPLATES_BY_ID",
    "TEMPLATES_BY_SIGNAL_TYPE",
    "get_template",
    "get_templates_by_signal_type",
    # Thresholds
    "DEFAULT_THRESHOLDS",
    "get_default_thresholds",
    "get_thresholds_for_signal",
    "select_threshold",
    "create_threshold_config",
    "select_threshold_for_rate",
    "select_growth_threshold_for_rate",
    # Signal statistics
    "get_threshold_for_rate",
    "get_growth_threshold_for_rate",
    "get_event_probability",
    "SIGNAL_STATS",
    "GROWTH_STATS",
    "EVENT_PROBABILITIES",
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
