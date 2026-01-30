"""
Generator for conditional forecasting questions.

Creates ConditionalQuestions by pairing:
- Conditions (interventions): gold boost, tech grant, government change
- Target questions: existing templates like treasury_comparative, score_rank_1

The generator selects appropriate target questions based on condition type
to ensure meaningful causal relationships.
"""

from datetime import datetime
from typing import Any

from .conditional_schema import (
    Condition,
    ConditionalQuestion,
    ConditionalQuestionBank,
)
from .schema import CivilizationInfo


class ConditionalQuestionGenerator:
    """
    Generates conditional questions pairing interventions with target questions.

    Condition-Target Mappings:
    - gold: treasury_comparative, score_comparative, score_rank_1
    - government: government_at, score_comparative, population_comparative
    - tech: tech_discovered, tech_comparative, score_comparative
    """

    # Which target templates make sense for each condition type
    CONDITION_TARGET_MAP: dict[str, list[str]] = {
        "gold": ["treasury_comparative", "score_comparative", "score_rank_1"],
        "government": ["government_at", "score_comparative", "population_comparative"],
        "tech": ["tech_discovered", "tech_comparative", "score_comparative"],
    }

    # Default gold amounts for interventions
    DEFAULT_GOLD_AMOUNTS = [5000, 10000]

    # Common government types for interventions
    DEFAULT_GOVERNMENTS = ["Republic", "Monarchy", "Democracy"]

    def __init__(self):
        """Initialize the conditional question generator."""
        self._question_counter = 0

    def generate_conditional_bank(
        self,
        game_id: str,
        game_data: dict[str, Any],
        checkpoint_turn: int,
        end_turn: int,
        conditions: list[Condition] | None = None,
        target_templates: list[str] | None = None,
    ) -> ConditionalQuestionBank:
        """
        Generate a complete conditional question bank for a game.

        Args:
            game_id: Simulation run identifier (e.g., seed)
            game_data: Output from MetricsCollector.collect_all()
            checkpoint_turn: Turn to create forks from
            end_turn: Turn to run forks until (resolution turn)
            conditions: Specific conditions to use; auto-generated if None
            target_templates: Specific templates to use; uses CONDITION_TARGET_MAP if None

        Returns:
            ConditionalQuestionBank with questions ready to run
        """
        # Extract civilizations
        civilizations = self._extract_civilizations(game_data, checkpoint_turn)

        # Auto-generate conditions if not provided
        if conditions is None:
            conditions = self._auto_generate_conditions(game_data, checkpoint_turn, civilizations)

        # Generate questions pairing conditions with targets
        questions = []
        for condition in conditions:
            new_questions = self._pair_condition_with_questions(
                condition=condition,
                game_data=game_data,
                checkpoint_turn=checkpoint_turn,
                end_turn=end_turn,
                civilizations=civilizations,
                target_templates=target_templates,
            )
            questions.extend(new_questions)

        return ConditionalQuestionBank(
            game_id=game_id,
            checkpoint_turn=checkpoint_turn,
            end_turn=end_turn,
            conditions=conditions,
            questions=questions,
            results={},
            generated_at=datetime.now().isoformat() + "Z",
        )

    def _extract_civilizations(
        self,
        game_data: dict[str, Any],
        checkpoint_turn: int,
    ) -> dict[int, CivilizationInfo]:
        """Extract civilization info from game data at checkpoint turn."""
        civs = {}

        # Determine which civs exist at checkpoint_turn
        existing_at_turn = set()
        time_series = game_data.get("time_series", {})
        for metric_data in time_series.values():
            turn_data = metric_data.get(str(checkpoint_turn), {})
            for player_id_str in turn_data.keys():
                existing_at_turn.add(int(player_id_str))

        for player_id_str, civ_data in game_data.get("civilizations", {}).items():
            player_id = int(player_id_str)
            if player_id not in existing_at_turn:
                continue

            civs[player_id] = CivilizationInfo(
                name=civ_data.get("name", f"Player {player_id}"),
                nation_id=civ_data.get("nation_id", 0),
            )

        return civs

    def _auto_generate_conditions(
        self,
        game_data: dict[str, Any],
        checkpoint_turn: int,
        civilizations: dict[int, CivilizationInfo],
    ) -> list[Condition]:
        """
        Auto-generate meaningful conditions for all players.

        Creates:
        - Gold boost (+5000) for each player
        - Government change (to Republic if not already)
        """
        conditions = []

        for player_id, civ_info in civilizations.items():
            civ_name = civ_info.name

            # Gold boost condition
            gold_amount = self.DEFAULT_GOLD_AMOUNTS[0]
            conditions.append(Condition(
                condition_id=f"gold_{gold_amount}_p{player_id}",
                condition_type="gold",
                player_id=player_id,
                value=gold_amount,
                description=f"{civ_name} receives {gold_amount} gold",
            ))

            # Government change condition (to Republic)
            current_gov = self._get_current_government(game_data, player_id, checkpoint_turn)
            target_gov = "Republic"
            if current_gov == target_gov:
                target_gov = "Monarchy"  # Use alternative if already Republic

            conditions.append(Condition(
                condition_id=f"gov_{target_gov.lower()}_p{player_id}",
                condition_type="government",
                player_id=player_id,
                value=target_gov,
                description=f"{civ_name} changes to {target_gov}",
            ))

        return conditions

    def _get_current_government(
        self,
        game_data: dict[str, Any],
        player_id: int,
        turn: int,
    ) -> str | None:
        """Get the government type for a player at a given turn."""
        events = game_data.get("events", [])

        # Find most recent government change before or at turn
        player_id_str = str(player_id)
        gov_events = [
            e for e in events
            if e.get("type") == "government_change"
            and str(e.get("player_id")) == player_id_str
            and e.get("turn", float("inf")) <= turn
        ]

        if gov_events:
            latest = max(gov_events, key=lambda e: e.get("turn", 0))
            return latest.get("metadata", {}).get("to")

        return "Despotism"  # Default starting government

    def _pair_condition_with_questions(
        self,
        condition: Condition,
        game_data: dict[str, Any],
        checkpoint_turn: int,
        end_turn: int,
        civilizations: dict[int, CivilizationInfo],
        target_templates: list[str] | None = None,
    ) -> list[ConditionalQuestion]:
        """
        Create conditional questions pairing a condition with appropriate targets.

        Args:
            condition: The condition to pair
            game_data: Game data for parameter generation
            checkpoint_turn: Fork turn
            end_turn: Resolution turn
            civilizations: Available civilizations
            target_templates: Override templates; uses CONDITION_TARGET_MAP if None

        Returns:
            List of ConditionalQuestions
        """
        questions = []

        # Determine which templates to use
        if target_templates is None:
            templates = self.CONDITION_TARGET_MAP.get(condition.condition_type, [])
        else:
            # Filter to templates valid for this condition type
            valid = set(self.CONDITION_TARGET_MAP.get(condition.condition_type, []))
            templates = [t for t in target_templates if t in valid]

        # Generate questions for each template
        for template_id in templates:
            new_questions = self._generate_for_template(
                condition=condition,
                template_id=template_id,
                game_data=game_data,
                checkpoint_turn=checkpoint_turn,
                end_turn=end_turn,
                civilizations=civilizations,
            )
            questions.extend(new_questions)

        return questions

    def _generate_for_template(
        self,
        condition: Condition,
        template_id: str,
        game_data: dict[str, Any],
        checkpoint_turn: int,
        end_turn: int,
        civilizations: dict[int, CivilizationInfo],
    ) -> list[ConditionalQuestion]:
        """Generate conditional questions for a specific template."""
        questions = []
        cond_player = condition.player_id
        cond_civ = civilizations.get(cond_player)
        cond_civ_name = cond_civ.name if cond_civ else f"Player {cond_player}"

        if template_id in ["treasury_comparative", "score_comparative", "tech_comparative", "population_comparative"]:
            # Comparative: condition player vs each other player
            for other_id, other_civ in civilizations.items():
                if other_id == cond_player:
                    continue

                params = {
                    "civ_a": cond_civ_name,
                    "civ_b": other_civ.name,
                    "player_id_a": cond_player,
                    "player_id_b": other_id,
                    "resolution_turn": end_turn,
                    "checkpoint_turn": checkpoint_turn,
                }

                questions.append(ConditionalQuestion(
                    conditional_id=f"cond_q{self._question_counter:04d}",
                    condition=condition,
                    target_template_id=template_id,
                    target_parameters=params,
                    checkpoint_turn=checkpoint_turn,
                    resolution_turn=end_turn,
                ))
                self._question_counter += 1

        elif template_id == "score_rank_1":
            # Rank: just for the condition player
            params = {
                "civ": cond_civ_name,
                "player_id": cond_player,
                "resolution_turn": end_turn,
                "checkpoint_turn": checkpoint_turn,
            }

            questions.append(ConditionalQuestion(
                conditional_id=f"cond_q{self._question_counter:04d}",
                condition=condition,
                target_template_id=template_id,
                target_parameters=params,
                checkpoint_turn=checkpoint_turn,
                resolution_turn=end_turn,
            ))
            self._question_counter += 1

        elif template_id == "government_at":
            # Government: check if player is in a specific government
            # For government conditions, ask about the target government
            if condition.condition_type == "government":
                gov_type = condition.value
            else:
                gov_type = "Republic"

            params = {
                "civ": cond_civ_name,
                "player_id": cond_player,
                "government_type": gov_type,
                "resolution_turn": end_turn,
                "checkpoint_turn": checkpoint_turn,
            }

            questions.append(ConditionalQuestion(
                conditional_id=f"cond_q{self._question_counter:04d}",
                condition=condition,
                target_template_id=template_id,
                target_parameters=params,
                checkpoint_turn=checkpoint_turn,
                resolution_turn=end_turn,
            ))
            self._question_counter += 1

        elif template_id == "tech_discovered":
            # Tech: check if player discovers a specific tech
            # For tech conditions, ask about the granted tech
            if condition.condition_type == "tech":
                tech_id = condition.value
                tech_name = self._get_tech_name(game_data, tech_id)
            else:
                # Pick a tech the player doesn't have yet
                tech_info = self._get_undiscovered_tech(
                    game_data, cond_player, checkpoint_turn
                )
                if tech_info is None:
                    return questions
                tech_name, tech_id = tech_info

            params = {
                "civ": cond_civ_name,
                "player_id": cond_player,
                "tech_name": tech_name,
                "tech_id": tech_id,
                "resolution_turn": end_turn,
                "checkpoint_turn": checkpoint_turn,
            }

            questions.append(ConditionalQuestion(
                conditional_id=f"cond_q{self._question_counter:04d}",
                condition=condition,
                target_template_id=template_id,
                target_parameters=params,
                checkpoint_turn=checkpoint_turn,
                resolution_turn=end_turn,
            ))
            self._question_counter += 1

        return questions

    def _get_tech_name(self, game_data: dict[str, Any], tech_id: int) -> str:
        """Get technology name from tech ID."""
        events = game_data.get("events", [])
        for e in events:
            if e.get("type") == "tech_discovered":
                meta = e.get("metadata", {})
                if meta.get("tech_id") == tech_id:
                    return meta.get("tech_name", f"Tech {tech_id}")
        return f"Tech {tech_id}"

    def _get_undiscovered_tech(
        self,
        game_data: dict[str, Any],
        player_id: int,
        checkpoint_turn: int,
    ) -> tuple[str, int] | None:
        """Find a tech the player hasn't discovered by checkpoint turn."""
        events = game_data.get("events", [])
        player_id_str = str(player_id)

        # Find techs this player has
        player_techs = set()
        for e in events:
            if (e.get("type") == "tech_discovered"
                and str(e.get("player_id")) == player_id_str
                and e.get("turn", float("inf")) <= checkpoint_turn):
                tech_name = e.get("metadata", {}).get("tech_name")
                if tech_name:
                    player_techs.add(tech_name)

        # Find a tech any player discovers later
        for e in events:
            if e.get("type") == "tech_discovered":
                tech_name = e.get("metadata", {}).get("tech_name")
                tech_id = e.get("metadata", {}).get("tech_id")
                if tech_name and tech_name not in player_techs:
                    return (tech_name, tech_id)

        return None


def create_condition(
    condition_type: str,
    player_id: int,
    value: int | str,
    civ_name: str | None = None,
) -> Condition:
    """
    Factory function to create a Condition.

    Args:
        condition_type: 'gold', 'government', or 'tech'
        player_id: Target player ID
        value: Condition value (gold amount, gov name, or tech ID)
        civ_name: Optional civ name for description

    Returns:
        Condition instance
    """
    civ_name = civ_name or f"Player {player_id}"

    if condition_type == "gold":
        return Condition(
            condition_id=f"gold_{value}_p{player_id}",
            condition_type="gold",
            player_id=player_id,
            value=int(value),
            description=f"{civ_name} receives {value} gold",
        )
    elif condition_type == "government":
        return Condition(
            condition_id=f"gov_{str(value).lower()}_p{player_id}",
            condition_type="government",
            player_id=player_id,
            value=str(value),
            description=f"{civ_name} changes to {value}",
        )
    elif condition_type == "tech":
        return Condition(
            condition_id=f"tech_{value}_p{player_id}",
            condition_type="tech",
            player_id=player_id,
            value=int(value),
            description=f"{civ_name} discovers tech {value}",
        )
    else:
        raise ValueError(f"Unknown condition type: {condition_type}")
