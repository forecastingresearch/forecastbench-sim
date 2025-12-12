"""
Data model schema for question generation.

This module defines the core dataclasses for:
- QuestionTemplate: Reusable question patterns with placeholders
- QuestionInstance: Specific questions generated from templates
- Resolution: Computed answers with supporting data
- WorldReportConfig: Configuration for world report generation
- ThresholdConfig: Configurable threshold values
- QuestionBank: Top-level container for a game's questions
"""

from dataclasses import dataclass, field
from typing import Literal, Any


@dataclass
class QuestionTemplate:
    """Defines a reusable question pattern with placeholders."""

    template_id: str
    """Unique identifier for this template, e.g., 'tech_count_gte'"""

    signal_type: Literal["B1", "B2", "B3"]
    """Base rate availability category"""

    signal_name: str
    """Name of the signal being measured, e.g., 'techs_known', 'treasury'"""

    question_template: str
    """Question text with placeholders, e.g., 'Will {civ} have ≥{threshold} technologies at turn {resolution_turn}?'"""

    resolution_type: Literal["threshold", "comparison", "event", "state_check"]
    """How this question type is resolved"""

    data_path: str
    """Path to data in game_data dict, e.g., 'time_series.techs_known.{player_id}.{resolution_turn}'"""

    comparison_op: str
    """Comparison operator: '>=', '>', '==', 'contains', 'exists', 'any', 'first'"""

    required_params: list[str]
    """Parameters required to instantiate this template"""

    optional_params: list[str] = field(default_factory=list)
    """Optional parameters for dyadic or complex questions"""


@dataclass
class Resolution:
    """Computed answer for a question."""

    answer: bool
    """The final yes/no answer"""

    resolution_turn: int
    """Turn at which the question was resolved"""

    # For threshold questions (B1, B2)
    value_at_resolution: float | int | None = None
    """Actual value at resolution turn"""

    threshold: float | int | None = None
    """Threshold being compared against"""

    comparison_op: str | None = None
    """Comparison operator used"""

    # For event questions (B3)
    event_occurred: bool | None = None
    """Whether the event occurred in the time window"""

    event_details: dict[str, Any] | None = None
    """Details about the event (turn, players, etc.)"""

    # For state check questions (B3)
    state_at_resolution: str | None = None
    """State value at resolution turn, e.g., 'War', 'Alliance'"""

    computed_at: str | None = None
    """ISO timestamp when resolution was computed"""


@dataclass
class QuestionInstance:
    """A specific question generated from a template."""

    question_id: str
    """Unique identifier for this question instance"""

    template_id: str
    """Reference to the QuestionTemplate used"""

    resolution_turn: int
    """Turn at which this question resolves"""

    horizon: Literal["H1", "H2", "H3"]
    """Time horizon category (derived from resolution_turn - snapshot_turn)"""

    base_rate: Literal["B1", "B2", "B3"]
    """Base rate availability category (from template)"""

    difficulty: int
    """Composite difficulty score: H + B (range 2-6)"""

    parameters: dict[str, Any]
    """Filled parameter values, e.g., {'civ': 'Greek', 'player_id': 1, 'threshold': 25}"""

    question_text: str
    """Rendered question text"""

    resolution: Resolution | None = None
    """Computed answer (None until resolved)"""


@dataclass
class WorldReportConfig:
    """Configuration for world report generation."""

    start_turn: int = 0
    """First turn to include in historical data"""

    sections: list[str] = field(default_factory=lambda: [
        "overview", "economics", "technology", "politics", "military"
    ])
    """Which report sections to include"""

    include_all_civs: bool = True
    """Whether to include all civilizations (needed for comparative questions)"""

    territory_snapshot_turns: list[int] | None = None
    """Turns for territory map snapshots (auto-selected if None)"""


@dataclass
class ThresholdConfig:
    """Configurable threshold values for question generation."""

    defaults: dict[str, list[int | float]] = field(default_factory=dict)
    """Per-signal default thresholds, e.g., {'techs_known': [10, 15, 20, 25, 30]}"""


@dataclass
class CivilizationInfo:
    """Basic information about a civilization."""

    name: str
    """Civilization name/adjective, e.g., 'Greek'"""

    nation_id: int
    """Nation ID from ruleset"""


@dataclass
class QuestionBank:
    """A collection of questions for a single game with a single snapshot."""

    game_id: str
    """Simulation run identifier (seed)"""

    snapshot_turn: int
    """The single turn at which forecasters see data"""

    game_max_turn: int
    """How far the simulation ran"""

    world_report_config: WorldReportConfig
    """Configuration for generating the world report"""

    civilizations: dict[int, CivilizationInfo]
    """Map of player_id to civilization info"""

    questions: list[QuestionInstance]
    """All questions for this game"""

    generated_at: str
    """ISO timestamp when this bank was generated"""


# Horizon classification helper
def classify_horizon(snapshot_turn: int, resolution_turn: int) -> Literal["H1", "H2", "H3"]:
    """
    Classify the time horizon based on turn delta.

    H1: Short (10-30 turns) - trends visible in recent history
    H2: Medium (50-100 turns) - requires reasoning about second-order effects
    H3: Long (150+ turns) - regime changes likely, compounding uncertainty
    """
    delta = resolution_turn - snapshot_turn
    if delta <= 30:
        return "H1"
    elif delta <= 100:
        return "H2"
    else:
        return "H3"


def horizon_to_int(horizon: Literal["H1", "H2", "H3"]) -> int:
    """Convert horizon category to numeric value for difficulty calculation."""
    return {"H1": 1, "H2": 2, "H3": 3}[horizon]


def base_rate_to_int(base_rate: Literal["B1", "B2", "B3"]) -> int:
    """Convert base rate category to numeric value for difficulty calculation."""
    return {"B1": 1, "B2": 2, "B3": 3}[base_rate]


def calculate_difficulty(horizon: Literal["H1", "H2", "H3"], base_rate: Literal["B1", "B2", "B3"]) -> int:
    """Calculate composite difficulty score (2-6)."""
    return horizon_to_int(horizon) + base_rate_to_int(base_rate)
