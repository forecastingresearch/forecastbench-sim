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
    classify_horizon,
)
from .templates import (
    ALL_TEMPLATES,
    TEMPLATES_BY_ID,
    get_template,
)


class QuestionGenerator:
    """
    Generates question banks from game data.

    The generator creates questions using predefined templates,
    supporting H0/H1/H2/H3 time horizons.
    """

    def __init__(
        self,
        templates: list[QuestionTemplate] | None = None,
    ):
        """
        Initialize the question generator.

        Args:
            templates: List of question templates to use; defaults to ALL_TEMPLATES
        """
        self.templates = templates or ALL_TEMPLATES
        self._templates_by_id = {t.template_id: t for t in self.templates}

    def generate_question_bank(
        self,
        game_id: str,
        game_data: dict[str, Any],
        snapshot_turn: int,
        resolution_turns: list[int] | None = None,
        world_report_config: WorldReportConfig | None = None,
    ) -> QuestionBank:
        """
        Generate a complete question bank for a single game.

        Args:
            game_id: Simulation run identifier (e.g., seed)
            game_data: Output from MetricsCollector.collect_all()
            snapshot_turn: Turn at which forecasters see data
            resolution_turns: Specific turns to resolve questions at; auto-selected if None
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

        # Extract civilizations (only those that exist at snapshot_turn)
        civilizations = self._extract_civilizations(game_data, snapshot_turn)

        # Generate world report config if not provided
        if world_report_config is None:
            world_report_config = self._create_world_report_config(snapshot_turn)

        # Generate questions using all templates
        questions = []
        question_counter = 0

        for template in self.templates:
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

    def generate_zero_turn_questions(
        self,
        game_id: str,
        game_data: dict[str, Any],
        snapshot_turn: int,
        world_report_config: WorldReportConfig | None = None,
    ) -> QuestionBank:
        """
        Generate zero-turn (H0) comprehension questions.

        These questions have resolution_turn == snapshot_turn, meaning the answer
        is directly observable from the data at the snapshot turn. Used to test
        model comprehension of the world state without any forecasting.

        Args:
            game_id: Simulation run identifier (e.g., seed)
            game_data: Output from data/games/{game_id}_data.json
            snapshot_turn: Turn at which to evaluate (resolution_turn = snapshot_turn)
            world_report_config: Custom world report config; auto-generated if None

        Returns:
            QuestionBank with H0 questions (resolution_turn == snapshot_turn)
        """
        max_turn = game_data["metadata"]["turn"]
        if snapshot_turn > max_turn:
            snapshot_turn = max_turn

        # Extract civilizations at snapshot_turn
        civilizations = self._extract_civilizations(game_data, snapshot_turn)

        # Generate world report config if not provided
        if world_report_config is None:
            world_report_config = self._create_world_report_config(snapshot_turn)

        # Generate H0 questions - resolution_turn == snapshot_turn
        questions = []
        question_counter = 0

        # Comparative questions (H0): compare values at snapshot_turn
        comparative_templates = [
            t for t in self.templates
            if t.resolution_type == "comparative"
        ]

        for template in comparative_templates:
            # Create H0 version: resolution_turn = snapshot_turn
            new_questions = self._generate_h0_comparative_questions(
                template=template,
                game_data=game_data,
                snapshot_turn=snapshot_turn,
                civilizations=civilizations,
                question_id_start=question_counter,
            )
            questions.extend(new_questions)
            question_counter += len(new_questions)

        # Milestone questions (H0): tech discovered by snapshot_turn
        tech_template = self._templates_by_id.get("tech_discovered")
        if tech_template:
            new_questions = self._generate_h0_tech_discovered_questions(
                template=tech_template,
                game_data=game_data,
                snapshot_turn=snapshot_turn,
                civilizations=civilizations,
                question_id_start=question_counter,
            )
            questions.extend(new_questions)
            question_counter += len(new_questions)

        # State check questions (H0): government at snapshot_turn
        government_template = self._templates_by_id.get("government_at")
        if government_template:
            new_questions = self._generate_h0_government_questions(
                template=government_template,
                game_data=game_data,
                snapshot_turn=snapshot_turn,
                civilizations=civilizations,
                question_id_start=question_counter,
            )
            questions.extend(new_questions)
            question_counter += len(new_questions)

        # Event questions (H0): wonder completed by snapshot_turn
        wonder_completed_template = self._templates_by_id.get("wonder_completed")
        if wonder_completed_template:
            new_questions = self._generate_h0_wonder_completed_questions(
                template=wonder_completed_template,
                game_data=game_data,
                snapshot_turn=snapshot_turn,
                question_id_start=question_counter,
            )
            questions.extend(new_questions)
            question_counter += len(new_questions)

        # Rank question (H0): is civ ranked #1 at snapshot_turn?
        rank_template = self._templates_by_id.get("score_rank_1")
        if rank_template:
            for player_id, civ_info in civilizations.items():
                params = {
                    "civ": civ_info.name,
                    "player_id": player_id,
                    "resolution_turn": snapshot_turn,
                }
                question_text = f"At turn {snapshot_turn}, is {civ_info.name} currently ranked #1 by score?"
                questions.append(QuestionInstance(
                    question_id=f"q{question_counter:04d}",
                    template_id="h0_score_rank_1",
                    resolution_turn=snapshot_turn,
                    horizon="H0",
                    parameters=params,
                    question_text=question_text,
                    resolution=None,
                ))
                question_counter += 1

        # Diplomatic state questions (H0): current war/alliance status
        for p1, p2 in itertools.combinations(civilizations.keys(), 2):
            civ_a = civilizations[p1].name
            civ_b = civilizations[p2].name

            # At war?
            questions.append(QuestionInstance(
                question_id=f"q{question_counter:04d}",
                template_id="h0_at_war_dyad",
                resolution_turn=snapshot_turn,
                horizon="H0",
                parameters={
                    "civ_a": civ_a,
                    "civ_b": civ_b,
                    "player_id_a": p1,
                    "player_id_b": p2,
                    "snapshot_turn": snapshot_turn,
                },
                question_text=f"At turn {snapshot_turn}, are {civ_a} and {civ_b} currently at war?",
                resolution=None,
            ))
            question_counter += 1

            # Alliance?
            questions.append(QuestionInstance(
                question_id=f"q{question_counter:04d}",
                template_id="h0_alliance_dyad",
                resolution_turn=snapshot_turn,
                horizon="H0",
                parameters={
                    "civ_a": civ_a,
                    "civ_b": civ_b,
                    "player_id_a": p1,
                    "player_id_b": p2,
                    "snapshot_turn": snapshot_turn,
                },
                question_text=f"At turn {snapshot_turn}, do {civ_a} and {civ_b} currently have an alliance?",
                resolution=None,
            ))
            question_counter += 1

        # At war with any?
        for player_id, civ_info in civilizations.items():
            questions.append(QuestionInstance(
                question_id=f"q{question_counter:04d}",
                template_id="h0_at_war_any",
                resolution_turn=snapshot_turn,
                horizon="H0",
                parameters={
                    "civ": civ_info.name,
                    "player_id": player_id,
                    "snapshot_turn": snapshot_turn,
                },
                question_text=f"At turn {snapshot_turn}, is {civ_info.name} currently at war with any civilization?",
                resolution=None,
            ))
            question_counter += 1

        return QuestionBank(
            game_id=game_id,
            snapshot_turn=snapshot_turn,
            game_max_turn=max_turn,
            world_report_config=world_report_config,
            civilizations=civilizations,
            questions=questions,
            generated_at=datetime.utcnow().isoformat() + "Z",
        )

    def _generate_h0_comparative_questions(
        self,
        template: QuestionTemplate,
        game_data: dict[str, Any],
        snapshot_turn: int,
        civilizations: dict[int, CivilizationInfo],
        question_id_start: int,
    ) -> list[QuestionInstance]:
        """
        Generate H0 comparative questions at snapshot_turn.

        Uses rank-adjacent pairings for ~50% base rate.
        """
        questions = []
        question_counter = question_id_start

        # Get rankings at snapshot turn
        rankings = self._get_rankings_for_signal(
            game_data, template.signal_name, snapshot_turn, civilizations
        )

        if len(rankings) < 2:
            return questions

        # Generate questions for rank-adjacent pairs
        for i in range(len(rankings) - 1):
            player_id_a, _ = rankings[i]      # Higher ranked
            player_id_b, _ = rankings[i + 1]  # Lower ranked

            civ_name_a = civilizations[player_id_a].name
            civ_name_b = civilizations[player_id_b].name

            # H0 template ID
            h0_template_id = f"h0_{template.template_id}"

            # H0 question text (asks about current state, not future)
            signal_display = template.signal_name.replace("_", " ")
            question_text = f"At turn {snapshot_turn}, does {civ_name_a} currently have more {signal_display} than {civ_name_b}?"

            params = {
                "civ_a": civ_name_a,
                "civ_b": civ_name_b,
                "player_id_a": player_id_a,
                "player_id_b": player_id_b,
                "snapshot_turn": snapshot_turn,
            }

            questions.append(QuestionInstance(
                question_id=f"q{question_counter:04d}",
                template_id=h0_template_id,
                resolution_turn=snapshot_turn,
                horizon="H0",
                parameters=params,
                question_text=question_text,
                resolution=None,
            ))
            question_counter += 1

        return questions

    def _auto_select_resolution_turns(self, snapshot_turn: int, max_turn: int) -> list[int]:
        """
        Auto-select resolution turns for H1, H2, H3 horizons.

        H1: 20 turns ahead (short extrapolation)
        H2: 80 turns ahead (medium, second-order effects)
        H3: 150 turns ahead (long, regime changes likely)
        """
        turns = []

        # H1: Short horizon (20 turns ahead)
        h1_turn = snapshot_turn + 20
        if h1_turn <= max_turn:
            turns.append(h1_turn)

        # H2: Medium horizon (80 turns ahead)
        h2_turn = snapshot_turn + 80
        if h2_turn <= max_turn:
            turns.append(h2_turn)

        # H3: Long horizon (150 turns ahead)
        h3_turn = snapshot_turn + 150
        if h3_turn <= max_turn:
            turns.append(h3_turn)

        return turns

    def _extract_civilizations(
        self,
        game_data: dict[str, Any],
        snapshot_turn: int | None = None,
    ) -> dict[int, CivilizationInfo]:
        """Extract civilization info from game data.

        Args:
            game_data: The full game data dict
            snapshot_turn: If provided, only include civs that exist at this turn

        Returns:
            Dict mapping player_id to CivilizationInfo
        """
        civs = {}

        # Determine which civs exist at snapshot_turn by checking time series data
        existing_at_snapshot = None
        if snapshot_turn is not None:
            existing_at_snapshot = set()
            time_series = game_data.get("time_series", {})
            # Check any time series metric for player presence at snapshot_turn
            for metric_name, metric_data in time_series.items():
                turn_data = metric_data.get(str(snapshot_turn), {})
                for player_id_str in turn_data.keys():
                    existing_at_snapshot.add(int(player_id_str))

        for player_id_str, civ_data in game_data.get("civilizations", {}).items():
            player_id = int(player_id_str)

            # Skip civs that don't exist at snapshot_turn
            if existing_at_snapshot is not None and player_id not in existing_at_snapshot:
                continue

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
        if template.resolution_type == "comparative":
            # Comparative questions use rank-adjacent pairings
            new_questions = self._generate_comparative_questions(
                template=template,
                game_data=game_data,
                snapshot_turn=snapshot_turn,
                resolution_turn=resolution_turn,
                civilizations=civilizations,
                question_id_start=question_counter,
            )
            questions.extend(new_questions)
            question_counter += len(new_questions)

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

        elif template.template_id in ["at_war_dyad", "alliance_dyad", "border_contact"]:
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

        elif template.template_id in ["city_founded", "city_lost", "anarchy_event", "treasury_zero"]:
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

    def _get_rankings_for_signal(
        self,
        game_data: dict[str, Any],
        signal_name: str,
        turn: int,
        civilizations: dict[int, CivilizationInfo],
    ) -> list[tuple[int, float | int]]:
        """
        Get ranked list of (player_id, value) for a signal at a specific turn.
        Returns list sorted by value descending.
        """
        if signal_name == "scores":
            # Scores are in snapshots
            snapshots = game_data.get("snapshots", {})
            turn_snapshot = snapshots.get(str(turn), {})
            scores = turn_snapshot.get("scores", {})
            if scores:
                values = [(int(pid), val) for pid, val in scores.items() if int(pid) in civilizations]
            else:
                # Fall back to time_series when snapshots are missing for this turn
                time_series = game_data.get("time_series", {}).get("scores", {})
                values = []
                for player_id in civilizations.keys():
                    val = self._get_signal_value(time_series, player_id, turn)
                    if val is not None:
                        values.append((player_id, val))
        else:
            # Other signals are in time_series
            time_series = game_data.get("time_series", {}).get(signal_name, {})
            values = []
            for player_id in civilizations.keys():
                val = self._get_signal_value(time_series, player_id, turn)
                if val is not None:
                    values.append((player_id, val))

        # Sort by value descending
        values.sort(key=lambda x: x[1], reverse=True)
        return values

    def _generate_h0_tech_discovered_questions(
        self,
        template: QuestionTemplate,
        game_data: dict[str, Any],
        snapshot_turn: int,
        civilizations: dict[int, CivilizationInfo],
        question_id_start: int,
    ) -> list[QuestionInstance]:
        """Generate H0 tech discovery questions at snapshot_turn."""
        questions = []
        question_counter = question_id_start
        events = game_data.get("events", [])

        all_techs: dict[str, int | None] = {}
        discovered_by_player: dict[int, set[str]] = {pid: set() for pid in civilizations.keys()}

        for e in events:
            if e.get("type") != "tech_discovered":
                continue
            tech_name = e.get("metadata", {}).get("tech_name")
            tech_id = e.get("metadata", {}).get("tech_id")
            if tech_name:
                all_techs.setdefault(tech_name, tech_id)
            if e.get("turn", float("inf")) <= snapshot_turn and tech_name:
                player_id = e.get("player_id")
                if player_id in discovered_by_player:
                    discovered_by_player[player_id].add(tech_name)

        if not all_techs:
            return questions

        sorted_techs = sorted(all_techs.items(), key=lambda x: x[0])

        for player_id, civ_info in civilizations.items():
            discovered = discovered_by_player.get(player_id, set())
            true_candidates = [(name, tid) for name, tid in sorted_techs if name in discovered]
            false_candidates = [(name, tid) for name, tid in sorted_techs if name not in discovered]

            for tech_name, tech_id in (true_candidates[:1] + false_candidates[:1]):
                params = {
                    "civ": civ_info.name,
                    "player_id": player_id,
                    "tech_name": tech_name,
                    "tech_id": tech_id,
                    "resolution_turn": snapshot_turn,
                    "snapshot_turn": snapshot_turn,
                }

                question_text = (
                    f"At turn {snapshot_turn}, has {civ_info.name} discovered {tech_name}?"
                )

                questions.append(QuestionInstance(
                    question_id=f"q{question_counter:04d}",
                    template_id="h0_tech_discovered",
                    resolution_turn=snapshot_turn,
                    horizon="H0",
                    parameters=params,
                    question_text=question_text,
                    resolution=None,
                ))
                question_counter += 1

        return questions

    def _generate_h0_government_questions(
        self,
        template: QuestionTemplate,
        game_data: dict[str, Any],
        snapshot_turn: int,
        civilizations: dict[int, CivilizationInfo],
        question_id_start: int,
    ) -> list[QuestionInstance]:
        """Generate H0 government questions at snapshot_turn."""
        questions = []
        question_counter = question_id_start
        events = game_data.get("events", [])

        gov_types = set()
        for e in events:
            if e.get("type") == "government_change":
                gov_from = e.get("metadata", {}).get("from")
                gov_to = e.get("metadata", {}).get("to")
                if gov_from:
                    gov_types.add(gov_from)
                if gov_to:
                    gov_types.add(gov_to)

        gov_types.discard("Anarchy")
        if not gov_types:
            gov_types = {"Despotism", "Monarchy", "Republic", "Democracy"}

        sorted_govs = sorted(gov_types)

        for player_id, civ_info in civilizations.items():
            current_gov = None
            for e in sorted(events, key=lambda x: x.get("turn", 0), reverse=True):
                if (e.get("type") == "government_change"
                    and str(e.get("player_id")) == str(player_id)
                    and e.get("turn", float("inf")) <= snapshot_turn):
                    current_gov = e.get("metadata", {}).get("to")
                    break

            if current_gov is None:
                continue

            candidates = [current_gov]
            for gov_type in sorted_govs:
                if gov_type != current_gov:
                    candidates.append(gov_type)
                    break

            for gov_type in candidates:
                params = {
                    "civ": civ_info.name,
                    "player_id": player_id,
                    "government_type": gov_type,
                    "resolution_turn": snapshot_turn,
                    "snapshot_turn": snapshot_turn,
                }

                question_text = (
                    f"At turn {snapshot_turn}, is {civ_info.name} in {gov_type}?"
                )

                questions.append(QuestionInstance(
                    question_id=f"q{question_counter:04d}",
                    template_id="h0_government_at",
                    resolution_turn=snapshot_turn,
                    horizon="H0",
                    parameters=params,
                    question_text=question_text,
                    resolution=None,
                ))
                question_counter += 1

        return questions

    def _generate_h0_wonder_completed_questions(
        self,
        template: QuestionTemplate,
        game_data: dict[str, Any],
        snapshot_turn: int,
        question_id_start: int,
    ) -> list[QuestionInstance]:
        """Generate H0 wonder completion questions at snapshot_turn."""
        questions = []
        question_counter = question_id_start
        events = game_data.get("events", [])

        wonders = {}
        for e in events:
            if e.get("type") == "wonder_completed":
                wonder_name = e.get("metadata", {}).get("wonder_name")
                wonder_id = e.get("metadata", {}).get("wonder_id")
                if wonder_name:
                    wonders[wonder_name] = wonder_id

        if not wonders:
            return questions

        for wonder_name, wonder_id in sorted(wonders.items(), key=lambda x: x[0]):
            params = {
                "wonder_name": wonder_name,
                "wonder_id": wonder_id,
                "snapshot_turn": 0,
                "resolution_turn": snapshot_turn,
            }

            question_text = (
                f"At turn {snapshot_turn}, has any civilization completed {wonder_name}?"
            )

            questions.append(QuestionInstance(
                question_id=f"q{question_counter:04d}",
                template_id="h0_wonder_completed",
                resolution_turn=snapshot_turn,
                horizon="H0",
                parameters=params,
                question_text=question_text,
                resolution=None,
            ))
            question_counter += 1

        return questions

    def _generate_comparative_questions(
        self,
        template: QuestionTemplate,
        game_data: dict[str, Any],
        snapshot_turn: int,
        resolution_turn: int,
        civilizations: dict[int, CivilizationInfo],
        question_id_start: int,
    ) -> list[QuestionInstance]:
        """
        Generate comparative questions using rank-adjacent pairings.

        Per the spec: "Use rank-adjacent pairings at snapshot turn" to produce
        ~50% base rates.
        """
        questions = []
        question_counter = question_id_start

        # Get rankings at snapshot turn
        rankings = self._get_rankings_for_signal(
            game_data, template.signal_name, snapshot_turn, civilizations
        )

        if len(rankings) < 2:
            return questions

        # Generate questions for rank-adjacent pairs
        for i in range(len(rankings) - 1):
            player_id_a, _ = rankings[i]      # Higher ranked
            player_id_b, _ = rankings[i + 1]  # Lower ranked

            civ_name_a = civilizations[player_id_a].name
            civ_name_b = civilizations[player_id_b].name

            params = {
                "civ_a": civ_name_a,
                "civ_b": civ_name_b,
                "player_id_a": player_id_a,
                "player_id_b": player_id_b,
                "resolution_turn": resolution_turn,
            }

            question_text = template.question_template.format(**params)
            horizon = classify_horizon(snapshot_turn, resolution_turn)

            questions.append(QuestionInstance(
                question_id=f"q{question_counter:04d}",
                template_id=template.template_id,
                resolution_turn=resolution_turn,
                horizon=horizon,
                parameters=params,
                question_text=question_text,
                resolution=None,
            ))
            question_counter += 1

        return questions

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

        return QuestionInstance(
            question_id=question_id,
            template_id=template.template_id,
            resolution_turn=resolution_turn,
            horizon=horizon,
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
            "snapshot_turn": snapshot_turn,
            "resolution_turn": resolution_turn,
        }

        question_text = template.question_template.format(**params)
        horizon = classify_horizon(snapshot_turn, resolution_turn)

        return QuestionInstance(
            question_id=question_id,
            template_id=template.template_id,
            resolution_turn=resolution_turn,
            horizon=horizon,
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

        return QuestionInstance(
            question_id=question_id,
            template_id=template.template_id,
            resolution_turn=resolution_turn,
            horizon=horizon,
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

        return QuestionInstance(
            question_id=question_id,
            template_id=template.template_id,
            resolution_turn=resolution_turn,
            horizon=horizon,
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

        return QuestionInstance(
            question_id=question_id,
            template_id=template.template_id,
            resolution_turn=resolution_turn,
            horizon=horizon,
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

                questions.append(QuestionInstance(
                    question_id=f"q{question_counter:04d}",
                    template_id=template.template_id,
                    resolution_turn=resolution_turn,
                    horizon=horizon,
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

                    questions.append(QuestionInstance(
                        question_id=f"q{question_counter:04d}",
                        template_id=template.template_id,
                        resolution_turn=resolution_turn,
                        horizon=horizon,
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

                    questions.append(QuestionInstance(
                        question_id=f"q{question_counter:04d}",
                        template_id=template.template_id,
                        resolution_turn=resolution_turn,
                        horizon=horizon,
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

                        questions.append(QuestionInstance(
                            question_id=f"q{question_counter:04d}",
                            template_id=template.template_id,
                            resolution_turn=resolution_turn,
                            horizon=horizon,
                            parameters=params,
                            question_text=question_text,
                            resolution=None,
                        ))
                        question_counter += 1

        return questions
