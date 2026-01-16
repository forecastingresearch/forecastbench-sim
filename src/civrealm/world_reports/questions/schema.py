"""
Data model schema for question generation.

This module defines the core dataclasses for:
- QuestionTemplate: Reusable question patterns with placeholders
- QuestionInstance: Specific questions generated from templates
- Resolution: Computed answers with supporting data
- WorldReportConfig: Configuration for world report generation
- QuestionBank: Top-level container for a game's questions
"""

from dataclasses import dataclass, field
from typing import Literal, Any


@dataclass
class QuestionTemplate:
    """Defines a reusable question pattern with placeholders."""

    template_id: str
    """Unique identifier for this template, e.g., 'tech_comparative'"""

    signal_name: str
    """Name of the signal being measured, e.g., 'techs_known', 'treasury'"""

    question_template: str
    """Question text with placeholders, e.g., 'Will {civ_a} have more technologies than {civ_b} at turn {resolution_turn}?'"""

    resolution_type: Literal["comparative", "milestone", "event", "state_check", "rank"]
    """How this question type is resolved:
    - comparative: Compare two civs on a metric
    - milestone: Check if specific achievement reached
    - event: Check if event occurred in time window
    - state_check: Check state at resolution turn
    - rank: Check ranking position
    """

    data_path: str
    """Path to data in game_data dict, e.g., 'time_series.techs_known.{player_id}.{resolution_turn}'"""

    comparison_op: str
    """Comparison operator: '>', '==', 'contains', 'exists', 'any', 'first', 'rank==1'"""

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

    # For comparative questions
    value_a: float | int | None = None
    """Value for civ A (in comparative questions)"""

    value_b: float | int | None = None
    """Value for civ B (in comparative questions)"""

    # For milestone/rank questions
    value_at_resolution: float | int | None = None
    """Actual value at resolution turn"""

    # For event questions
    event_occurred: bool | None = None
    """Whether the event occurred in the time window"""

    event_details: dict[str, Any] | None = None
    """Details about the event (turn, players, etc.)"""

    # For state check questions
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

    horizon: Literal["H0", "H1", "H2", "H3"]
    """Time horizon category (derived from resolution_turn - snapshot_turn)
    H0: Zero horizon (comprehension questions, resolution_turn == snapshot_turn)
    H1: Short horizon (≤20 turns)
    H2: Medium horizon (21-80 turns)
    H3: Long horizon (>80 turns)
    """

    parameters: dict[str, Any]
    """Filled parameter values, e.g., {'civ_a': 'Greek', 'civ_b': 'Roman', 'player_id_a': 1, 'player_id_b': 2}"""

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
def classify_horizon(snapshot_turn: int, resolution_turn: int) -> Literal["H0", "H1", "H2", "H3"]:
    """
    Classify the time horizon based on turn delta.

    H0: Zero (0 turns) - comprehension questions, answer observable at snapshot
    H1: Short (≤20 turns) - trends visible in recent history
    H2: Medium (21-80 turns) - requires reasoning about second-order effects
    H3: Long (>80 turns) - regime changes likely, compounding uncertainty
    """
    delta = resolution_turn - snapshot_turn
    if delta <= 0:
        return "H0"
    elif delta <= 20:
        return "H1"
    elif delta <= 80:
        return "H2"
    else:
        return "H3"
