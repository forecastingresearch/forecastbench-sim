"""
Data model schema for conditional forecasting questions.

Conditional questions test P(B | A=yes) vs P(B | A=no) where:
- Condition (A): An intervention applied via savegame modification
- Target Question (B): An existing unconditional question template

The workflow:
1. Define a Condition (gold boost, tech grant, government change)
2. Pair it with target questions from existing templates
3. Run paired forks (control vs intervention)
4. Compare outcomes to measure conditional effects
"""

from dataclasses import dataclass, field
from typing import Literal, Any


@dataclass
class Condition:
    """Defines an intervention to apply to a savegame fork."""

    condition_id: str
    """Unique identifier, e.g., 'gold_5000_p0'"""

    condition_type: Literal["gold", "government", "tech"]
    """Type of intervention:
    - gold: Add gold to a player's treasury
    - government: Change a player's government type
    - tech: Grant a technology to a player
    """

    player_id: int
    """Player to apply the intervention to"""

    value: int | str
    """Intervention value:
    - gold: Amount to add (e.g., 5000)
    - government: Government name (e.g., 'Republic')
    - tech: Tech ID (e.g., 23 for Iron Working)
    """

    description: str
    """Human-readable description, e.g., 'Rome receives 5000 gold'"""


@dataclass
class ConditionalQuestion:
    """A conditional question pairing a condition with a target question."""

    conditional_id: str
    """Unique identifier for this conditional question"""

    condition: Condition
    """The intervention to apply"""

    target_template_id: str
    """Template ID of the target question (e.g., 'treasury_comparative')"""

    target_parameters: dict[str, Any]
    """Parameters for instantiating the target question"""

    checkpoint_turn: int
    """Turn to fork from (when condition is applied)"""

    resolution_turn: int
    """Turn to resolve the target question at"""


@dataclass
class ForkOutcome:
    """Results from running a single fork (control or intervention)."""

    fork_name: str
    """Fork identifier: 'control' or 'intervention'"""

    success: bool
    """Whether the fork completed successfully"""

    final_turn: int
    """Turn the fork reached"""

    player_states: dict[int, dict]
    """Final player states: {player_id: {score, gold, is_alive, ...}}"""

    error: str | None = None
    """Error message if fork failed"""


@dataclass
class ConditionalResult:
    """Results from running a conditional question (both forks)."""

    conditional_id: str
    """Reference to the ConditionalQuestion"""

    control_outcome: ForkOutcome
    """Results from control fork (no intervention)"""

    intervention_outcome: ForkOutcome
    """Results from intervention fork (with condition applied)"""

    answer_control: bool | None
    """Target question answer in control fork"""

    answer_intervention: bool | None
    """Target question answer in intervention fork"""

    conditional_effect: float | None
    """1.0 if answers differ, 0.0 if same, None if either fork failed"""

    computed_at: str
    """ISO timestamp when result was computed"""


@dataclass
class ConditionalQuestionBank:
    """Collection of conditional questions for a game."""

    game_id: str
    """Simulation run identifier (seed)"""

    checkpoint_turn: int
    """Turn from which forks are created"""

    end_turn: int
    """Turn to run forks until"""

    conditions: list[Condition]
    """All conditions used in this bank"""

    questions: list[ConditionalQuestion]
    """All conditional questions"""

    results: dict[str, ConditionalResult] = field(default_factory=dict)
    """Results keyed by conditional_id"""

    generated_at: str = ""
    """ISO timestamp when bank was generated"""
