"""
Question generator for creating question banks from game data.
"""

from datetime import datetime
from typing import Any
import itertools

from .schema import (
    QuestionBank,
    QuestionInstance,
    QuestionTemplate,
    WorldReportConfig,
    CivilizationInfo,
    ThresholdConfig,
    classify_horizon,
    calculate_difficulty,
)
from .templates import (
    ALL_TEMPLATES,
    TEMPLATES_BY_ID,
    TEMPLATES_BY_SIGNAL_TYPE,
    get_template,
)
from .thresholds import DEFAULT_THRESHOLDS, select_threshold


class QuestionGenerator:
    """
    Generates question banks from game data.

    The generator creates questions using predefined templates and configurable
    thresholds, supporting all B1/B2/B3 signal types and H1/H2/H3 time horizons.
    """

    def __init__(
        self,
        templates: list[QuestionTemplate] | None = None,
        thresholds: ThresholdConfig | None = None,
    ):
        """
        Initialize the question generator.

        Args:
            templates: List of question templates to use; defaults to ALL_TEMPLATES
            thresholds: Threshold configuration; defaults to DEFAULT_THRESHOLDS
        """
        self.templates = templates or ALL_TEMPLATES
        self.thresholds = thresholds or DEFAULT_THRESHOLDS
        self._templates_by_id = {t.template_id: t for t in self.templates}

    def generate_question_bank(
        self,
        game_id: str,
        game_data: dict[str, Any],
        snapshot_turn: int,
        resolution_turns: list[int] | None = None,
        signal_types: list[str] | None = None,
        world_report_config: WorldReportConfig | None = None,
    ) -> QuestionBank:
        """
        Generate a complete question bank for a single game.

        Args:
            game_id: Simulation run identifier (e.g., seed)
            game_data: Output from MetricsCollector.collect_all()
            snapshot_turn: Turn at which forecasters see data
            resolution_turns: Specific turns to resolve questions at; auto-selected if None
            signal_types: List of signal types to include (["B1", "B2", "B3"]); all if None
            world_report_config: Custom world report config; auto-generated if None

        Returns:
            QuestionBank with all generated questions
        """
        # Validate inputs
        max_turn = game_data["metadata"]["turn"]
        if snapshot_turn >= max_turn:
            raise ValueError(f"snapshot_turn ({snapshot_turn}) must be < max_turn ({max_turn})")

        # Auto-select resolution turns for different horizons if not specified
        if resolution_turns is None:
            resolution_turns = self._auto_select_resolution_turns(snapshot_turn, max_turn)

        # Filter to valid resolution turns
        resolution_turns = [t for t in resolution_turns if snapshot_turn < t <= max_turn]

        # Select templates by signal type
        if signal_types is None:
            signal_types = ["B1", "B2", "B3"]

        templates_to_use = []
        for st in signal_types:
            templates_to_use.extend(TEMPLATES_BY_SIGNAL_TYPE.get(st, []))

        # Extract civilizations
        civilizations = self._extract_civilizations(game_data)

        # Generate world report config if not provided
        if world_report_config is None:
            world_report_config = self._create_world_report_config(snapshot_turn)

        # Generate questions
        questions = []
        question_counter = 0

        for template in templates_to_use:
            for resolution_turn in resolution_turns:
                new_questions = self._generate_questions_for_template(
                    template=template,
                    game_data=game_data,
                    snapshot_turn=snapshot_turn,
                    resolution_turn=resolution_turn,
                    civilizations=civilizations,
                    question_id_start=question_counter,
                )
                questions.extend(new_questions)
                question_counter += len(new_questions)

        return QuestionBank(
            game_id=game_id,
            snapshot_turn=snapshot_turn,
            game_max_turn=max_turn,
            world_report_config=world_report_config,
            civilizations=civilizations,
            questions=questions,
            generated_at=datetime.utcnow().isoformat() + "Z",
        )

    def _auto_select_resolution_turns(self, snapshot_turn: int, max_turn: int) -> list[int]:
        """
        Auto-select resolution turns for H1, H2, H3 horizons.

        H1: 10-30 turns ahead
        H2: 50-100 turns ahead
        H3: 150+ turns ahead
        """
        turns = []

        # H1: Short horizon (20 turns ahead)
        h1_turn = snapshot_turn + 20
        if h1_turn <= max_turn:
            turns.append(h1_turn)

        # H2: Medium horizon (75 turns ahead)
        h2_turn = snapshot_turn + 75
        if h2_turn <= max_turn:
            turns.append(h2_turn)

        # H3: Long horizon (150 turns ahead)
        h3_turn = snapshot_turn + 150
        if h3_turn <= max_turn:
            turns.append(h3_turn)

        return turns

    def _extract_civilizations(self, game_data: dict[str, Any]) -> dict[int, CivilizationInfo]:
        """Extract civilization info from game data."""
        civs = {}
        for player_id_str, civ_data in game_data.get("civilizations", {}).items():
            player_id = int(player_id_str)
            civs[player_id] = CivilizationInfo(
                name=civ_data.get("name", f"Player {player_id}"),
                nation_id=civ_data.get("nation_id", 0),
            )
        return civs

    def _create_world_report_config(self, snapshot_turn: int) -> WorldReportConfig:
        """Create default world report configuration."""
        # Auto-select territory snapshot turns
        territory_turns = []
        if snapshot_turn >= 10:
            territory_turns.append(10)
        if snapshot_turn >= 30:
            territory_turns.append(30)
        if snapshot_turn >= 50:
            territory_turns.append(50)
        territory_turns.append(snapshot_turn)

        return WorldReportConfig(
            start_turn=0,
            sections=["overview", "economics", "technology", "politics", "military"],
            include_all_civs=True,
            territory_snapshot_turns=territory_turns,
        )

    def _generate_questions_for_template(
        self,
        template: QuestionTemplate,
        game_data: dict[str, Any],
        snapshot_turn: int,
        resolution_turn: int,
        civilizations: dict[int, CivilizationInfo],
        question_id_start: int,
    ) -> list[QuestionInstance]:
        """Generate all questions for a single template at a specific resolution turn."""
        questions = []
        question_counter = question_id_start

        # Dispatch based on template type
        if template.template_id in ["tech_count_gte", "population_gte", "score_gte",
                                    "territory_gte", "treasury_gte", "cities_gte"]:
            # Single-civ threshold questions
            for player_id, civ_info in civilizations.items():
                q = self._generate_threshold_question(
                    template=template,
                    game_data=game_data,
                    snapshot_turn=snapshot_turn,
                    resolution_turn=resolution_turn,
                    player_id=player_id,
                    civ_name=civ_info.name,
                    question_id=f"q{question_counter:04d}",
                )
                if q:
                    questions.append(q)
                    question_counter += 1

        elif template.template_id == "territory_gain":
            # Territory gain (needs both snapshot and resolution turn)
            for player_id, civ_info in civilizations.items():
                q = self._generate_territory_gain_question(
                    template=template,
                    game_data=game_data,
                    snapshot_turn=snapshot_turn,
                    resolution_turn=resolution_turn,
                    player_id=player_id,
                    civ_name=civ_info.name,
                    question_id=f"q{question_counter:04d}",
                )
                if q:
                    questions.append(q)
                    question_counter += 1

        elif template.template_id == "score_rank_1":
            # Rank question - one per civ
            for player_id, civ_info in civilizations.items():
                q = self._generate_rank_question(
                    template=template,
                    snapshot_turn=snapshot_turn,
                    resolution_turn=resolution_turn,
                    player_id=player_id,
                    civ_name=civ_info.name,
                    question_id=f"q{question_counter:04d}",
                )
                questions.append(q)
                question_counter += 1

        elif template.template_id in ["at_war_dyad", "alliance_dyad"]:
            # Dyadic questions - all pairs
            player_ids = list(civilizations.keys())
            for p1, p2 in itertools.combinations(player_ids, 2):
                q = self._generate_dyadic_question(
                    template=template,
                    snapshot_turn=snapshot_turn,
                    resolution_turn=resolution_turn,
                    player_id_a=p1,
                    player_id_b=p2,
                    civ_name_a=civilizations[p1].name,
                    civ_name_b=civilizations[p2].name,
                    question_id=f"q{question_counter:04d}",
                )
                questions.append(q)
                question_counter += 1

        elif template.template_id == "at_war_any":
            # Single-civ "at war with any" questions
            for player_id, civ_info in civilizations.items():
                q = self._generate_at_war_any_question(
                    template=template,
                    snapshot_turn=snapshot_turn,
                    resolution_turn=resolution_turn,
                    player_id=player_id,
                    civ_name=civ_info.name,
                    question_id=f"q{question_counter:04d}",
                )
                questions.append(q)
                question_counter += 1

        elif template.template_id in ["city_founded", "city_lost", "anarchy_event"]:
            # Single-civ event questions
            for player_id, civ_info in civilizations.items():
                q = self._generate_civ_event_question(
                    template=template,
                    snapshot_turn=snapshot_turn,
                    resolution_turn=resolution_turn,
                    player_id=player_id,
                    civ_name=civ_info.name,
                    question_id=f"q{question_counter:04d}",
                )
                questions.append(q)
                question_counter += 1

        elif template.template_id == "city_conquered_any":
            # Global event question - just one
            q = self._generate_global_event_question(
                template=template,
                snapshot_turn=snapshot_turn,
                resolution_turn=resolution_turn,
                question_id=f"q{question_counter:04d}",
            )
            questions.append(q)
            question_counter += 1

        elif template.template_id == "tech_discovered":
            # Generate tech discovery questions based on techs seen in events
            new_questions = self._generate_tech_discovered_questions(
                template=template,
                game_data=game_data,
                snapshot_turn=snapshot_turn,
                resolution_turn=resolution_turn,
                civilizations=civilizations,
                question_id_start=question_counter,
            )
            questions.extend(new_questions)
            question_counter += len(new_questions)

        elif template.template_id == "government_at":
            # Generate government questions for common government types
            new_questions = self._generate_government_questions(
                template=template,
                game_data=game_data,
                snapshot_turn=snapshot_turn,
                resolution_turn=resolution_turn,
                civilizations=civilizations,
                question_id_start=question_counter,
            )
            questions.extend(new_questions)
            question_counter += len(new_questions)

        elif template.template_id in ["wonder_completed", "wonder_first"]:
            # Generate wonder questions based on wonders seen in events
            new_questions = self._generate_wonder_questions(
                template=template,
                game_data=game_data,
                snapshot_turn=snapshot_turn,
                resolution_turn=resolution_turn,
                civilizations=civilizations,
                question_id_start=question_counter,
            )
            questions.extend(new_questions)
            question_counter += len(new_questions)

        return questions

    def _get_signal_value(
        self,
        time_series: dict[str, Any],
        player_id: int,
        turn: int,
    ) -> float | int | None:
        """
        Get signal value for a player at a specific turn.

        Handles both data structures:
        - turn -> player_id -> value
        - player_id -> turn -> value
        """
        if not time_series:
            return None

        # Check structure
        first_key = list(time_series.keys())[0]
        first_value = time_series.get(first_key)

        if isinstance(first_value, dict):
            # Could be turn -> player or player -> turn
            # Check if first_key looks like a turn number
            try:
                int(first_key)
                # Structure: turn -> player_id -> value
                turn_data = time_series.get(str(turn), {})
                return turn_data.get(str(player_id))
            except ValueError:
                # Structure: player_id -> turn -> value
                player_data = time_series.get(str(player_id), {})
                if isinstance(player_data, dict):
                    return player_data.get(str(turn))

        return None

    def _generate_threshold_question(
        self,
        template: QuestionTemplate,
        game_data: dict[str, Any],
        snapshot_turn: int,
        resolution_turn: int,
        player_id: int,
        civ_name: str,
        question_id: str,
    ) -> QuestionInstance | None:
        """Generate a threshold-based question."""
        # Get current value to select appropriate threshold
        signal_name = template.signal_name
        time_series = game_data.get("time_series", {}).get(signal_name, {})

        # Handle both structures: turn -> player_id or player_id -> turn
        current_value = self._get_signal_value(time_series, player_id, snapshot_turn)

        if current_value is None:
            return None  # No data available

        # Also get expected value at resolution turn to pick better threshold
        future_value = self._get_signal_value(time_series, player_id, resolution_turn)

        # Select threshold that will result in ~50% True answers
        # Use a value closer to the future value to make questions answerable
        try:
            if future_value is not None and future_value > current_value:
                # Pick a threshold likely to be reached
                # Use 70% of the way from current to future (biased toward reachable)
                target_value = current_value + 0.7 * (future_value - current_value)
                threshold = select_threshold(signal_name, target_value, self.thresholds, strategy="nearest_below")
                # Ensure threshold is above current value (otherwise question is trivial)
                if threshold <= current_value:
                    threshold = select_threshold(signal_name, current_value, self.thresholds, strategy="nearest_above")
            else:
                threshold = select_threshold(signal_name, current_value, self.thresholds, strategy="nearest_above")
        except ValueError:
            # No thresholds configured for this signal
            return None

        # Build parameters
        params = {
            "civ": civ_name,
            "player_id": player_id,
            "threshold": threshold,
            "resolution_turn": resolution_turn,
        }

        # Render question text
        question_text = template.question_template.format(**params)

        # Calculate difficulty
        horizon = classify_horizon(snapshot_turn, resolution_turn)
        difficulty = calculate_difficulty(horizon, template.signal_type)

        return QuestionInstance(
            question_id=question_id,
            template_id=template.template_id,
            resolution_turn=resolution_turn,
            horizon=horizon,
            base_rate=template.signal_type,
            difficulty=difficulty,
            parameters=params,
            question_text=question_text,
            resolution=None,
        )

    def _generate_territory_gain_question(
        self,
        template: QuestionTemplate,
        game_data: dict[str, Any],
        snapshot_turn: int,
        resolution_turn: int,
        player_id: int,
        civ_name: str,
        question_id: str,
    ) -> QuestionInstance | None:
        """Generate a territory gain question."""
        # Get threshold (using territory_gain signal)
        try:
            threshold = select_threshold("territory_gain", 0, self.thresholds, strategy="median")
        except ValueError:
            return None

        params = {
            "civ": civ_name,
            "player_id": player_id,
            "threshold": threshold,
            "snapshot_turn": snapshot_turn,
            "resolution_turn": resolution_turn,
        }

        question_text = template.question_template.format(**params)
        horizon = classify_horizon(snapshot_turn, resolution_turn)
        difficulty = calculate_difficulty(horizon, template.signal_type)

        return QuestionInstance(
            question_id=question_id,
            template_id=template.template_id,
            resolution_turn=resolution_turn,
            horizon=horizon,
            base_rate=template.signal_type,
            difficulty=difficulty,
            parameters=params,
            question_text=question_text,
            resolution=None,
        )

    def _generate_rank_question(
        self,
        template: QuestionTemplate,
        snapshot_turn: int,
        resolution_turn: int,
        player_id: int,
        civ_name: str,
        question_id: str,
    ) -> QuestionInstance:
        """Generate a rank-based question."""
        params = {
            "civ": civ_name,
            "player_id": player_id,
            "resolution_turn": resolution_turn,
        }

        question_text = template.question_template.format(**params)
        horizon = classify_horizon(snapshot_turn, resolution_turn)
        difficulty = calculate_difficulty(horizon, template.signal_type)

        return QuestionInstance(
            question_id=question_id,
            template_id=template.template_id,
            resolution_turn=resolution_turn,
            horizon=horizon,
            base_rate=template.signal_type,
            difficulty=difficulty,
            parameters=params,
            question_text=question_text,
            resolution=None,
        )

    def _generate_dyadic_question(
        self,
        template: QuestionTemplate,
        snapshot_turn: int,
        resolution_turn: int,
        player_id_a: int,
        player_id_b: int,
        civ_name_a: str,
        civ_name_b: str,
        question_id: str,
    ) -> QuestionInstance:
        """Generate a dyadic (two-civ) question."""
        params = {
            "civ_a": civ_name_a,
            "civ_b": civ_name_b,
            "player_id_a": player_id_a,
            "player_id_b": player_id_b,
            "resolution_turn": resolution_turn,
        }

        question_text = template.question_template.format(**params)
        horizon = classify_horizon(snapshot_turn, resolution_turn)
        difficulty = calculate_difficulty(horizon, template.signal_type)

        return QuestionInstance(
            question_id=question_id,
            template_id=template.template_id,
            resolution_turn=resolution_turn,
            horizon=horizon,
            base_rate=template.signal_type,
            difficulty=difficulty,
            parameters=params,
            question_text=question_text,
            resolution=None,
        )

    def _generate_at_war_any_question(
        self,
        template: QuestionTemplate,
        snapshot_turn: int,
        resolution_turn: int,
        player_id: int,
        civ_name: str,
        question_id: str,
    ) -> QuestionInstance:
        """Generate an 'at war with any' question."""
        params = {
            "civ": civ_name,
            "player_id": player_id,
            "resolution_turn": resolution_turn,
        }

        question_text = template.question_template.format(**params)
        horizon = classify_horizon(snapshot_turn, resolution_turn)
        difficulty = calculate_difficulty(horizon, template.signal_type)

        return QuestionInstance(
            question_id=question_id,
            template_id=template.template_id,
            resolution_turn=resolution_turn,
            horizon=horizon,
            base_rate=template.signal_type,
            difficulty=difficulty,
            parameters=params,
            question_text=question_text,
            resolution=None,
        )

    def _generate_civ_event_question(
        self,
        template: QuestionTemplate,
        snapshot_turn: int,
        resolution_turn: int,
        player_id: int,
        civ_name: str,
        question_id: str,
    ) -> QuestionInstance:
        """Generate a civ-specific event question."""
        params = {
            "civ": civ_name,
            "player_id": player_id,
            "snapshot_turn": snapshot_turn,
            "resolution_turn": resolution_turn,
        }

        question_text = template.question_template.format(**params)
        horizon = classify_horizon(snapshot_turn, resolution_turn)
        difficulty = calculate_difficulty(horizon, template.signal_type)

        return QuestionInstance(
            question_id=question_id,
            template_id=template.template_id,
            resolution_turn=resolution_turn,
            horizon=horizon,
            base_rate=template.signal_type,
            difficulty=difficulty,
            parameters=params,
            question_text=question_text,
            resolution=None,
        )

    def _generate_global_event_question(
        self,
        template: QuestionTemplate,
        snapshot_turn: int,
        resolution_turn: int,
        question_id: str,
    ) -> QuestionInstance:
        """Generate a global event question (not civ-specific)."""
        params = {
            "snapshot_turn": snapshot_turn,
            "resolution_turn": resolution_turn,
        }

        question_text = template.question_template.format(**params)
        horizon = classify_horizon(snapshot_turn, resolution_turn)
        difficulty = calculate_difficulty(horizon, template.signal_type)

        return QuestionInstance(
            question_id=question_id,
            template_id=template.template_id,
            resolution_turn=resolution_turn,
            horizon=horizon,
            base_rate=template.signal_type,
            difficulty=difficulty,
            parameters=params,
            question_text=question_text,
            resolution=None,
        )

    def _generate_tech_discovered_questions(
        self,
        template: QuestionTemplate,
        game_data: dict[str, Any],
        snapshot_turn: int,
        resolution_turn: int,
        civilizations: dict[int, CivilizationInfo],
        question_id_start: int,
    ) -> list[QuestionInstance]:
        """Generate tech discovery questions based on techs seen in events."""
        questions = []
        question_counter = question_id_start
        events = game_data.get("events", [])

        # Find all unique techs discovered in the game
        all_techs = set()
        for e in events:
            if e.get("type") == "tech_discovered":
                tech_name = e.get("metadata", {}).get("tech_name")
                tech_id = e.get("metadata", {}).get("tech_id")
                if tech_name:
                    all_techs.add((tech_name, tech_id))

        if not all_techs:
            return questions

        # For each player, ask about techs they haven't discovered yet (by snapshot_turn)
        for player_id, civ_info in civilizations.items():
            # Find techs this player has already discovered by snapshot_turn
            player_techs = set()
            for e in events:
                if (e.get("type") == "tech_discovered"
                    and e.get("player_id") == player_id
                    and e.get("turn", float("inf")) <= snapshot_turn):
                    tech_name = e.get("metadata", {}).get("tech_name")
                    if tech_name:
                        player_techs.add(tech_name)

            # Ask about techs they don't have yet (limit to 3 per player)
            techs_to_ask = []
            for tech_name, tech_id in all_techs:
                if tech_name not in player_techs:
                    techs_to_ask.append((tech_name, tech_id))
                if len(techs_to_ask) >= 3:
                    break

            for tech_name, tech_id in techs_to_ask:
                params = {
                    "civ": civ_info.name,
                    "player_id": player_id,
                    "tech_name": tech_name,
                    "tech_id": tech_id,
                    "resolution_turn": resolution_turn,
                }

                question_text = template.question_template.format(**params)
                horizon = classify_horizon(snapshot_turn, resolution_turn)
                difficulty = calculate_difficulty(horizon, template.signal_type)

                questions.append(QuestionInstance(
                    question_id=f"q{question_counter:04d}",
                    template_id=template.template_id,
                    resolution_turn=resolution_turn,
                    horizon=horizon,
                    base_rate=template.signal_type,
                    difficulty=difficulty,
                    parameters=params,
                    question_text=question_text,
                    resolution=None,
                ))
                question_counter += 1

        return questions

    def _generate_government_questions(
        self,
        template: QuestionTemplate,
        game_data: dict[str, Any],
        snapshot_turn: int,
        resolution_turn: int,
        civilizations: dict[int, CivilizationInfo],
        question_id_start: int,
    ) -> list[QuestionInstance]:
        """Generate government type questions."""
        questions = []
        question_counter = question_id_start
        events = game_data.get("events", [])

        # Find all government types seen in events
        gov_types = set()
        for e in events:
            if e.get("type") == "government_change":
                gov_from = e.get("metadata", {}).get("from")
                gov_to = e.get("metadata", {}).get("to")
                if gov_from:
                    gov_types.add(gov_from)
                if gov_to:
                    gov_types.add(gov_to)

        # Remove Anarchy as it's typically transitional
        gov_types.discard("Anarchy")

        # If no governments found in events, use common defaults
        if not gov_types:
            gov_types = {"Despotism", "Monarchy", "Republic", "Democracy"}

        # For each player, ask about their government at resolution turn
        for player_id, civ_info in civilizations.items():
            # Find current government at snapshot_turn
            current_gov = None
            for e in sorted(events, key=lambda x: x.get("turn", 0), reverse=True):
                if (e.get("type") == "government_change"
                    and e.get("player_id") == player_id
                    and e.get("turn", float("inf")) <= snapshot_turn):
                    current_gov = e.get("metadata", {}).get("to")
                    break

            # Ask about government types different from current (limit to 2 per player)
            gov_count = 0
            for gov_type in gov_types:
                if gov_type != current_gov and gov_count < 2:
                    params = {
                        "civ": civ_info.name,
                        "player_id": player_id,
                        "government_type": gov_type,
                        "resolution_turn": resolution_turn,
                    }

                    question_text = template.question_template.format(**params)
                    horizon = classify_horizon(snapshot_turn, resolution_turn)
                    difficulty = calculate_difficulty(horizon, template.signal_type)

                    questions.append(QuestionInstance(
                        question_id=f"q{question_counter:04d}",
                        template_id=template.template_id,
                        resolution_turn=resolution_turn,
                        horizon=horizon,
                        base_rate=template.signal_type,
                        difficulty=difficulty,
                        parameters=params,
                        question_text=question_text,
                        resolution=None,
                    ))
                    question_counter += 1
                    gov_count += 1

        return questions

    def _generate_wonder_questions(
        self,
        template: QuestionTemplate,
        game_data: dict[str, Any],
        snapshot_turn: int,
        resolution_turn: int,
        civilizations: dict[int, CivilizationInfo],
        question_id_start: int,
    ) -> list[QuestionInstance]:
        """Generate wonder completion questions based on wonders seen in events."""
        questions = []
        question_counter = question_id_start
        events = game_data.get("events", [])

        # Find all wonders and when they were completed
        wonders = {}  # wonder_name -> {completed_turn, player_id}
        for e in events:
            if e.get("type") == "wonder_completed":
                wonder_name = e.get("metadata", {}).get("wonder_name")
                wonder_id = e.get("metadata", {}).get("wonder_id")
                if wonder_name:
                    wonders[wonder_name] = {
                        "wonder_id": wonder_id,
                        "completed_turn": e.get("turn"),
                        "player_id": e.get("player_id"),
                    }

        if not wonders:
            return questions

        if template.template_id == "wonder_completed":
            # Ask about wonders not yet completed by snapshot_turn
            for wonder_name, wonder_info in wonders.items():
                if wonder_info["completed_turn"] > snapshot_turn:
                    params = {
                        "wonder_name": wonder_name,
                        "wonder_id": wonder_info["wonder_id"],
                        "snapshot_turn": snapshot_turn,
                        "resolution_turn": resolution_turn,
                    }

                    question_text = template.question_template.format(**params)
                    horizon = classify_horizon(snapshot_turn, resolution_turn)
                    difficulty = calculate_difficulty(horizon, template.signal_type)

                    questions.append(QuestionInstance(
                        question_id=f"q{question_counter:04d}",
                        template_id=template.template_id,
                        resolution_turn=resolution_turn,
                        horizon=horizon,
                        base_rate=template.signal_type,
                        difficulty=difficulty,
                        parameters=params,
                        question_text=question_text,
                        resolution=None,
                    ))
                    question_counter += 1

        elif template.template_id == "wonder_first":
            # Ask which civ will complete each uncompleted wonder first
            for wonder_name, wonder_info in wonders.items():
                if wonder_info["completed_turn"] > snapshot_turn:
                    # Ask for each civilization
                    for player_id, civ_info in civilizations.items():
                        params = {
                            "civ": civ_info.name,
                            "player_id": player_id,
                            "wonder_name": wonder_name,
                            "wonder_id": wonder_info["wonder_id"],
                            "snapshot_turn": snapshot_turn,
                            "resolution_turn": resolution_turn,
                        }

                        question_text = template.question_template.format(**params)
                        horizon = classify_horizon(snapshot_turn, resolution_turn)
                        difficulty = calculate_difficulty(horizon, template.signal_type)

                        questions.append(QuestionInstance(
                            question_id=f"q{question_counter:04d}",
                            template_id=template.template_id,
                            resolution_turn=resolution_turn,
                            horizon=horizon,
                            base_rate=template.signal_type,
                            difficulty=difficulty,
                            parameters=params,
                            question_text=question_text,
                            resolution=None,
                        ))
                        question_counter += 1

        return questions
