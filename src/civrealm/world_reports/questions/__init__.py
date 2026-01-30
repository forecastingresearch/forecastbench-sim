"""
Question generation module for creating forecasting questions from game data.

This module provides:
- Data model schema (QuestionBank, QuestionInstance, Resolution, etc.)
- Question templates
- QuestionGenerator for creating question banks
- QuestionResolver for computing answers
- Conditional questions for P(B|A) forecasting

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

    # Conditional questions (P(B|A) forecasting)
    from civrealm.world_reports.questions import (
        ConditionalQuestionGenerator,
        ConditionalQuestionRunner,
        ConditionalQuestionBank,
    )

    cond_generator = ConditionalQuestionGenerator()
    cond_bank = cond_generator.generate_conditional_bank(
        game_id="seed_42",
        game_data=game_data,
        checkpoint_turn=50,
        end_turn=100,
    )

    runner = ConditionalQuestionRunner(recording_dir, base_seed=42)
    cond_bank = runner.run_batch(cond_bank)
"""

from .schema import (
    QuestionTemplate,
    QuestionInstance,
    Resolution,
    WorldReportConfig,
    CivilizationInfo,
    QuestionBank,
    classify_horizon,
)

from .templates import (
    ALL_TEMPLATES,
    TEMPLATES_BY_ID,
    get_template,
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

# Conditional questions
from .conditional_schema import (
    Condition,
    ConditionalQuestion,
    ConditionalResult,
    ForkOutcome,
    ConditionalQuestionBank,
)

from .conditional_generator import (
    ConditionalQuestionGenerator,
    create_condition,
)

from .conditional_runner import ConditionalQuestionRunner

from .conditional_io import (
    conditional_bank_to_dict,
    conditional_bank_to_json,
    save_conditional_bank,
    load_conditional_bank,
    dict_to_conditional_bank,
    to_question_instances,
    to_question_bank,
    save_as_question_bank,
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
    # Templates
    "ALL_TEMPLATES",
    "TEMPLATES_BY_ID",
    "get_template",
    # Generator & Resolver
    "QuestionGenerator",
    "QuestionResolver",
    # I/O
    "question_bank_to_dict",
    "question_bank_to_json",
    "save_question_bank",
    "load_question_bank",
    "dict_to_question_bank",
    # Conditional Schema
    "Condition",
    "ConditionalQuestion",
    "ConditionalResult",
    "ForkOutcome",
    "ConditionalQuestionBank",
    # Conditional Generator & Runner
    "ConditionalQuestionGenerator",
    "create_condition",
    "ConditionalQuestionRunner",
    # Conditional I/O
    "conditional_bank_to_dict",
    "conditional_bank_to_json",
    "save_conditional_bank",
    "load_conditional_bank",
    "dict_to_conditional_bank",
    "to_question_instances",
    "to_question_bank",
    "save_as_question_bank",
]
