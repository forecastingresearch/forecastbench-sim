"""
Generator for conditional forecasting questions.

Creates ConditionalQuestions by pairing:
- Conditions (interventions): gold boost, tech grant, government change
- Target questions: ALL templates (same as unconditional questions)

Uses the same templates as unconditional questions - let the data show
what's causally linked rather than assuming it.
"""

from datetime import datetime
from typing import Any

from .conditional_schema import (
    Condition,
    ConditionalQuestion,
    ConditionalQuestionBank,
)
from .schema import CivilizationInfo
from .templates import get_template


class ConditionalQuestionGenerator:
    """
    Generates conditional questions pairing interventions with target questions.

    Uses ALL templates (same as unconditional) for every condition type.
    """

    # All templates used for conditional questions (same as unconditional)
    ALL_TARGET_TEMPLATES: list[str] = [
        "treasury_comparative",
        "score_comparative",
        "tech_comparative",
        "population_comparative",
        "city_count_comparative",
        "territory_comparative",
        "score_rank_1",
        "tech_discovered",
        "wonder_completed",
        "government_at",
        "techs_continuous",
        "treasury_continuous",
        "population_continuous",
        "cities_count_continuous",
        "territory_continuous",
        "scores_continuous",
    ]

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
        resolution_turns: list[int] | None = None,
        conditions: list[Condition] | None = None,
        target_templates: list[str] | None = None,
    ) -> ConditionalQuestionBank:
        """
        Generate a complete conditional question bank for a game.

        Args:
            game_id: Simulation run identifier (e.g., seed)
            game_data: Output from MetricsCollector.collect_all()
            checkpoint_turn: Turn to create forks from
            end_turn: Maximum turn (used for auto-selecting resolution turns)
            resolution_turns: Specific turns to resolve questions at; auto-selected if None
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

        # Auto-select resolution turns matching unconditional horizons (H1, H2, H3)
        if resolution_turns is None:
            resolution_turns = self._auto_select_resolution_turns(checkpoint_turn, end_turn)

        # Generate questions pairing conditions with targets for each resolution turn
        questions = []
        for condition in conditions:
            for resolution_turn in resolution_turns:
                new_questions = self._pair_condition_with_questions(
                    condition=condition,
                    game_data=game_data,
                    checkpoint_turn=checkpoint_turn,
                    resolution_turn=resolution_turn,
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

    def _get_government_types(
        self,
        game_data: dict[str, Any],
        player_id: int,
        checkpoint_turn: int,
    ) -> list[str]:
        """
        Get government types to ask about for a player.

        Returns government types that make sense to ask about:
        - The player's current government
        - Common government types (Republic, Monarchy, Democracy)
        - Governments used by other players
        """
        gov_types = set()

        # Add player's current government
        current_gov = self._get_current_government(game_data, player_id, checkpoint_turn)
        if current_gov:
            gov_types.add(current_gov)

        # Add common government types
        for gov in self.DEFAULT_GOVERNMENTS:
            gov_types.add(gov)

        # Add governments from game events (other players' governments)
        events = game_data.get("events", [])
        for e in events:
            if e.get("type") == "government_change" and e.get("turn", 0) <= checkpoint_turn:
                gov_to = e.get("metadata", {}).get("to")
                if gov_to:
                    gov_types.add(gov_to)

        return sorted(gov_types)

    def _auto_select_resolution_turns(self, checkpoint_turn: int, max_turn: int) -> list[int]:
        """
        Select resolution turns for H1, H2, H3 horizons.

        Matches the unconditional question generator horizons:
        - H1: checkpoint_turn + 30
        - H2: checkpoint_turn + 60
        - H3: checkpoint_turn + 90
        """
        horizons = [30, 60, 90]  # H1, H2, H3
        turns = [checkpoint_turn + h for h in horizons]
        return [t for t in turns if t <= max_turn]

    def _pair_condition_with_questions(
        self,
        condition: Condition,
        game_data: dict[str, Any],
        checkpoint_turn: int,
        resolution_turn: int,
        civilizations: dict[int, CivilizationInfo],
        target_templates: list[str] | None = None,
    ) -> list[ConditionalQuestion]:
        """
        Create conditional questions pairing a condition with ALL templates.

        Args:
            condition: The condition to pair
            game_data: Game data for parameter generation
            checkpoint_turn: Fork turn
            resolution_turn: Turn to resolve question at
            civilizations: Available civilizations
            target_templates: Override templates; uses ALL_TARGET_TEMPLATES if None

        Returns:
            List of ConditionalQuestions
        """
        questions = []

        # Use all templates (same as unconditional)
        templates = target_templates if target_templates else self.ALL_TARGET_TEMPLATES

        # Generate questions for each template
        for template_id in templates:
            new_questions = self._generate_for_template(
                condition=condition,
                template_id=template_id,
                game_data=game_data,
                checkpoint_turn=checkpoint_turn,
                resolution_turn=resolution_turn,
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
        resolution_turn: int,
        civilizations: dict[int, CivilizationInfo],
    ) -> list[ConditionalQuestion]:
        """Generate conditional questions for a specific template."""
        questions = []
        cond_player = condition.player_id
        cond_civ = civilizations.get(cond_player)
        cond_civ_name = cond_civ.name if cond_civ else f"Player {cond_player}"

        if template_id in [
            "treasury_comparative",
            "score_comparative",
            "tech_comparative",
            "population_comparative",
            "city_count_comparative",
            "territory_comparative",
        ]:
            # Comparative: condition player vs each other player
            for other_id, other_civ in civilizations.items():
                if other_id == cond_player:
                    continue

                params = {
                    "civ_a": cond_civ_name,
                    "civ_b": other_civ.name,
                    "player_id_a": cond_player,
                    "player_id_b": other_id,
                    "resolution_turn": resolution_turn,
                    "checkpoint_turn": checkpoint_turn,
                }

                questions.append(ConditionalQuestion(
                    conditional_id=f"cond_q{self._question_counter:04d}",
                    condition=condition,
                    target_template_id=template_id,
                    target_parameters=params,
                    checkpoint_turn=checkpoint_turn,
                    resolution_turn=resolution_turn,
                ))
                self._question_counter += 1

        elif template_id == "score_rank_1":
            # Rank: just for the condition player
            params = {
                "civ": cond_civ_name,
                "player_id": cond_player,
                "resolution_turn": resolution_turn,
                "checkpoint_turn": checkpoint_turn,
            }

            questions.append(ConditionalQuestion(
                conditional_id=f"cond_q{self._question_counter:04d}",
                condition=condition,
                target_template_id=template_id,
                target_parameters=params,
                checkpoint_turn=checkpoint_turn,
                resolution_turn=resolution_turn,
            ))
            self._question_counter += 1

        elif template_id == "government_at":
            # Government: check if player is in specific governments
            # Generate questions for multiple government types (like unconditional)
            gov_types_to_ask = self._get_government_types(game_data, cond_player, checkpoint_turn)

            for gov_type in gov_types_to_ask:
                params = {
                    "civ": cond_civ_name,
                    "player_id": cond_player,
                    "government_type": gov_type,
                    "resolution_turn": resolution_turn,
                    "checkpoint_turn": checkpoint_turn,
                }

                questions.append(ConditionalQuestion(
                    conditional_id=f"cond_q{self._question_counter:04d}",
                    condition=condition,
                    target_template_id=template_id,
                    target_parameters=params,
                    checkpoint_turn=checkpoint_turn,
                    resolution_turn=resolution_turn,
                ))
                self._question_counter += 1

        elif template_id == "tech_discovered":
            # Tech: check if player discovers specific techs
            # Generate questions for multiple undiscovered techs (like unconditional)
            if condition.condition_type == "tech":
                # For tech conditions, ask about the granted tech
                tech_id = condition.value
                tech_name = self._get_tech_name(game_data, tech_id)
                techs_to_ask = [(tech_name, tech_id)]
            else:
                # Get multiple techs the player doesn't have yet
                techs_to_ask = self._get_undiscovered_techs(
                    game_data, cond_player, checkpoint_turn, limit=3
                )

            for tech_name, tech_id in techs_to_ask:
                params = {
                    "civ": cond_civ_name,
                    "player_id": cond_player,
                    "tech_name": tech_name,
                    "tech_id": tech_id,
                    "resolution_turn": resolution_turn,
                    "checkpoint_turn": checkpoint_turn,
                }

                questions.append(ConditionalQuestion(
                    conditional_id=f"cond_q{self._question_counter:04d}",
                    condition=condition,
                    target_template_id=template_id,
                    target_parameters=params,
                    checkpoint_turn=checkpoint_turn,
                    resolution_turn=resolution_turn,
                ))
                self._question_counter += 1

        elif template_id == "wonder_completed":
            # Wonder: check if any civ completes a specific wonder
            # Get wonders not yet completed by checkpoint_turn
            wonders_to_ask = self._get_uncompleted_wonders(
                game_data, checkpoint_turn, limit=3
            )

            for wonder_name, wonder_id in wonders_to_ask:
                params = {
                    "civ": cond_civ_name,
                    "player_id": cond_player,
                    "wonder_name": wonder_name,
                    "wonder_id": wonder_id,
                    "snapshot_turn": checkpoint_turn,
                    "resolution_turn": resolution_turn,
                }

                questions.append(ConditionalQuestion(
                    conditional_id=f"cond_q{self._question_counter:04d}",
                    condition=condition,
                    target_template_id=template_id,
                    target_parameters=params,
                    checkpoint_turn=checkpoint_turn,
                    resolution_turn=resolution_turn,
                ))
                self._question_counter += 1

        elif template_id in [
            "techs_continuous", "treasury_continuous", "population_continuous",
            "cities_count_continuous", "territory_continuous", "scores_continuous",
        ]:
            # Continuous: one question for the affected civ only
            template = get_template(template_id)
            params = {
                "civ": cond_civ_name,
                "player_id": cond_player,
                "metric": template.signal_name,
                "resolution_turn": resolution_turn,
                "checkpoint_turn": checkpoint_turn,
            }
            questions.append(ConditionalQuestion(
                conditional_id=f"cond_q{self._question_counter:04d}",
                condition=condition,
                target_template_id=template_id,
                target_parameters=params,
                checkpoint_turn=checkpoint_turn,
                resolution_turn=resolution_turn,
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

    def _get_undiscovered_techs(
        self,
        game_data: dict[str, Any],
        player_id: int,
        checkpoint_turn: int,
        limit: int = 3,
    ) -> list[tuple[str, int]]:
        """Find multiple techs the player hasn't discovered by checkpoint turn."""
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

        # Find techs any player discovers (that this player doesn't have)
        undiscovered = []
        seen_techs = set()
        for e in events:
            if e.get("type") == "tech_discovered":
                tech_name = e.get("metadata", {}).get("tech_name")
                tech_id = e.get("metadata", {}).get("tech_id")
                if tech_name and tech_name not in player_techs and tech_name not in seen_techs:
                    undiscovered.append((tech_name, tech_id))
                    seen_techs.add(tech_name)
                    if len(undiscovered) >= limit:
                        break

        return undiscovered

    def _get_uncompleted_wonders(
        self,
        game_data: dict[str, Any],
        checkpoint_turn: int,
        limit: int = 3,
    ) -> list[tuple[str, int]]:
        """Find wonders not yet completed by checkpoint turn."""
        events = game_data.get("events", [])

        # Find wonders completed after checkpoint_turn
        uncompleted = []
        seen_wonders = set()
        for e in events:
            if e.get("type") == "wonder_completed":
                turn = e.get("turn", 0)
                if turn > checkpoint_turn:
                    wonder_name = e.get("metadata", {}).get("wonder_name")
                    wonder_id = e.get("metadata", {}).get("wonder_id")
                    if wonder_name and wonder_name not in seen_wonders:
                        uncompleted.append((wonder_name, wonder_id))
                        seen_wonders.add(wonder_name)
                        if len(uncompleted) >= limit:
                            break

        return uncompleted


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
    elif condition_type == "gold_add":
        return Condition(
            condition_id=f"goldadd_{value}_p{player_id}",
            condition_type="gold_add",
            player_id=player_id,
            value=int(value),
            description=f"{civ_name} receives +{value} gold",
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
