"""
Question resolver for computing answers from game data.
"""

from datetime import datetime
from typing import Any

from .schema import QuestionInstance, QuestionBank, Resolution
from .templates import get_template


class QuestionResolver:
    """
    Resolves questions by computing answers from game data.

    The resolver reads the complete game data and determines the ground truth
    answer for each question based on its template and parameters.
    """

    def resolve(
        self,
        question: QuestionInstance,
        game_data: dict[str, Any],
        snapshot_turn: int,
    ) -> Resolution:
        """
        Compute the answer for a single question.

        Args:
            question: The question to resolve
            game_data: Complete game data from MetricsCollector
            snapshot_turn: The snapshot turn (for event window calculations)

        Returns:
            Resolution with computed answer
        """
        template = get_template(question.template_id)

        # Dispatch based on resolution type and template
        if template.resolution_type == "comparative":
            return self._resolve_comparative(question, game_data, template)
        elif template.resolution_type == "rank":
            return self._resolve_rank(question, game_data, template)
        elif template.resolution_type == "milestone":
            return self._resolve_milestone(question, game_data, template)
        elif template.resolution_type == "event":
            return self._resolve_event(question, game_data, template, snapshot_turn)
        elif template.resolution_type == "state_check":
            return self._resolve_state_check(question, game_data, template)
        else:
            raise ValueError(f"Unknown resolution_type: {template.resolution_type}")

    def resolve_batch(
        self,
        question_bank: QuestionBank,
        game_data: dict[str, Any],
    ) -> QuestionBank:
        """
        Resolve all questions in a question bank.

        Args:
            question_bank: QuestionBank with questions to resolve
            game_data: Complete game data from MetricsCollector

        Returns:
            QuestionBank with all resolutions populated
        """
        for question in question_bank.questions:
            resolution = self.resolve(question, game_data, question_bank.snapshot_turn)
            question.resolution = resolution

        return question_bank

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

    def _resolve_comparative(
        self,
        question: QuestionInstance,
        game_data: dict[str, Any],
        template: Any,
    ) -> Resolution:
        """Resolve a comparative question (comparing two civs)."""
        params = question.parameters
        resolution_turn = question.resolution_turn
        player_id_a = params.get("player_id_a")
        player_id_b = params.get("player_id_b")

        signal_name = template.signal_name

        # Get values at resolution turn
        if signal_name == "scores":
            # Scores are in snapshots
            snapshots = game_data.get("snapshots", {})
            turn_snapshot = snapshots.get(str(resolution_turn), {})
            scores = turn_snapshot.get("scores", {})
            value_a = scores.get(str(player_id_a))
            value_b = scores.get(str(player_id_b))
        else:
            # Other signals are in time_series
            time_series = game_data.get("time_series", {}).get(signal_name, {})
            value_a = self._get_signal_value(time_series, player_id_a, resolution_turn)
            value_b = self._get_signal_value(time_series, player_id_b, resolution_turn)

        if value_a is None or value_b is None:
            # No data - default to False
            return Resolution(
                answer=False,
                resolution_turn=resolution_turn,
                value_a=value_a,
                value_b=value_b,
                computed_at=datetime.utcnow().isoformat() + "Z",
            )

        # Compare: template.comparison_op is ">" for "more than"
        answer = value_a > value_b

        return Resolution(
            answer=answer,
            resolution_turn=resolution_turn,
            value_a=value_a,
            value_b=value_b,
            computed_at=datetime.utcnow().isoformat() + "Z",
        )

    def _resolve_rank(
        self,
        question: QuestionInstance,
        game_data: dict[str, Any],
        template: Any,
    ) -> Resolution:
        """Resolve a rank-based question (e.g., rank #1)."""
        params = question.parameters
        resolution_turn = question.resolution_turn
        player_id = params.get("player_id")

        # Check if player is ranked #1
        snapshots = game_data.get("snapshots", {})
        turn_snapshot = snapshots.get(str(resolution_turn), {})
        rankings = turn_snapshot.get("rankings", [])

        rank = None
        for entry in rankings:
            if entry.get("player_id") == player_id:
                rank = entry.get("rank")
                break

        answer = rank == 1

        return Resolution(
            answer=answer,
            resolution_turn=resolution_turn,
            value_at_resolution=rank,
            state_at_resolution=f"rank {rank}" if rank else None,
            computed_at=datetime.utcnow().isoformat() + "Z",
        )

    def _resolve_milestone(
        self,
        question: QuestionInstance,
        game_data: dict[str, Any],
        template: Any,
    ) -> Resolution:
        """Resolve a milestone question (e.g., tech discovery)."""
        params = question.parameters
        resolution_turn = question.resolution_turn

        if template.template_id == "tech_discovered":
            player_id = str(params.get("player_id"))
            tech_name = params.get("tech_name")
            tech_id = params.get("tech_id")

            # Check events for tech_discovered by this player up to resolution turn
            events = game_data.get("events", [])
            tech_events = [
                e for e in events
                if e.get("type") == "tech_discovered"
                and str(e.get("player_id")) == player_id
                and e.get("turn", float("inf")) <= resolution_turn
                and (e.get("metadata", {}).get("tech_name") == tech_name
                     or e.get("metadata", {}).get("tech_id") == tech_id)
            ]

            discovered = len(tech_events) > 0
            discovery_turn = tech_events[0].get("turn") if tech_events else None

            return Resolution(
                answer=discovered,
                resolution_turn=resolution_turn,
                state_at_resolution=tech_name if discovered else None,
                event_occurred=discovered,
                event_details={"discovery_turn": discovery_turn} if discovered else None,
                computed_at=datetime.utcnow().isoformat() + "Z",
            )

        # Default fallback
        return Resolution(
            answer=False,
            resolution_turn=resolution_turn,
            computed_at=datetime.utcnow().isoformat() + "Z",
        )

    def _resolve_event(
        self,
        question: QuestionInstance,
        game_data: dict[str, Any],
        template: Any,
        snapshot_turn: int,
    ) -> Resolution:
        """Resolve an event-based question."""
        params = question.parameters
        resolution_turn = question.resolution_turn
        events = game_data.get("events", [])

        # Filter events in time window
        start_turn = params.get("snapshot_turn", snapshot_turn)
        events_in_window = [
            e for e in events
            if start_turn < e.get("turn", 0) <= resolution_turn
        ]

        event_occurred = False
        event_details = None

        if template.template_id == "city_founded":
            player_id = str(params.get("player_id"))
            matching = [
                e for e in events_in_window
                if e.get("type") == "city_founded" and str(e.get("player_id")) == player_id
            ]
            event_occurred = len(matching) > 0
            if matching:
                event_details = {"count": len(matching), "first_turn": matching[0].get("turn")}

        elif template.template_id == "treasury_zero":
            # Check if treasury hits 0 at any point in the window
            player_id = params.get("player_id")
            time_series = game_data.get("time_series", {}).get("treasury", {})

            # Scan turns in window for treasury == 0
            for turn in range(start_turn + 1, resolution_turn + 1):
                value = self._get_signal_value(time_series, player_id, turn)
                if value is not None and value <= 0:
                    event_occurred = True
                    event_details = {"turn": turn, "value": value}
                    break

        elif template.template_id == "border_contact":
            # Check if borders touch by resolution turn
            # This requires territory map data
            player_id_a = params.get("player_id_a")
            player_id_b = params.get("player_id_b")

            territory_snapshots = game_data.get("territory_snapshots", {})
            # Check the resolution turn snapshot (or closest available)
            for turn in range(resolution_turn, start_turn, -1):
                snapshot = territory_snapshots.get(str(turn))
                if snapshot and "tile_owner" in snapshot.get("map_state", {}):
                    # Check adjacency in tile_owner map
                    event_occurred = self._check_border_contact(
                        snapshot["map_state"]["tile_owner"],
                        player_id_a,
                        player_id_b
                    )
                    if event_occurred:
                        event_details = {"contact_turn": turn}
                    break

        elif template.template_id == "city_conquered_any":
            matching = [e for e in events_in_window if e.get("type") == "city_conquered"]
            event_occurred = len(matching) > 0
            if matching:
                event_details = {"count": len(matching), "first_turn": matching[0].get("turn")}

        elif template.template_id == "city_lost":
            player_id = str(params.get("player_id"))
            matching = [
                e for e in events_in_window
                if e.get("type") == "city_conquered"
                and str(e.get("metadata", {}).get("prev_owner")) == player_id
            ]
            event_occurred = len(matching) > 0
            if matching:
                event_details = {"count": len(matching), "first_turn": matching[0].get("turn")}

        elif template.template_id == "anarchy_event":
            player_id = str(params.get("player_id"))
            matching = [
                e for e in events_in_window
                if e.get("type") == "government_change"
                and str(e.get("player_id")) == player_id
                and e.get("metadata", {}).get("to") == "Anarchy"
            ]
            event_occurred = len(matching) > 0
            if matching:
                event_details = {"turn": matching[0].get("turn")}

        elif template.template_id == "wonder_completed":
            wonder_name = params.get("wonder_name")
            wonder_id = params.get("wonder_id")
            matching = [
                e for e in events_in_window
                if e.get("type") == "wonder_completed"
                and (e.get("metadata", {}).get("wonder_name") == wonder_name
                     or e.get("metadata", {}).get("wonder_id") == wonder_id)
            ]
            event_occurred = len(matching) > 0
            if matching:
                event_details = {
                    "turn": matching[0].get("turn"),
                    "player_id": matching[0].get("player_id"),
                }

        elif template.template_id == "wonder_first":
            player_id = str(params.get("player_id"))
            wonder_name = params.get("wonder_name")
            wonder_id = params.get("wonder_id")
            matching = [
                e for e in events_in_window
                if e.get("type") == "wonder_completed"
                and (e.get("metadata", {}).get("wonder_name") == wonder_name
                     or e.get("metadata", {}).get("wonder_id") == wonder_id)
            ]
            if matching:
                first_completion = min(matching, key=lambda e: e.get("turn", float("inf")))
                event_occurred = str(first_completion.get("player_id")) == player_id
                event_details = {
                    "turn": first_completion.get("turn"),
                    "completed_by": first_completion.get("player_id"),
                }

        return Resolution(
            answer=event_occurred,
            resolution_turn=resolution_turn,
            event_occurred=event_occurred,
            event_details=event_details,
            computed_at=datetime.utcnow().isoformat() + "Z",
        )

    def _check_border_contact(
        self,
        tile_owner: list,
        player_id_a: int,
        player_id_b: int,
    ) -> bool:
        """
        Check if two players' territories are adjacent.

        Args:
            tile_owner: 2D list of tile ownership (player_id or -1 for unowned)
            player_id_a: First player ID
            player_id_b: Second player ID

        Returns:
            True if any tile owned by A is adjacent to a tile owned by B
        """
        import numpy as np

        try:
            tile_owner_arr = np.array(tile_owner)
            # Handle both row-major and column-major layouts
            if tile_owner_arr.ndim == 1:
                # Assume it's flattened, try to reshape (80x50 is common)
                try:
                    tile_owner_arr = tile_owner_arr.reshape(50, 80)
                except ValueError:
                    return False

            ysize, xsize = tile_owner_arr.shape

            # Find all tiles owned by player A
            player_a_tiles = np.argwhere(tile_owner_arr == player_id_a)

            # Check 8-directional adjacency for each tile
            for y, x in player_a_tiles:
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < ysize and 0 <= nx < xsize:
                            if tile_owner_arr[ny, nx] == player_id_b:
                                return True

            return False
        except Exception:
            # If numpy operations fail, return False
            return False

    def _resolve_state_check(
        self,
        question: QuestionInstance,
        game_data: dict[str, Any],
        template: Any,
    ) -> Resolution:
        """Resolve a state check question."""
        params = question.parameters
        resolution_turn = question.resolution_turn

        if template.template_id in ["at_war_dyad", "alliance_dyad"]:
            player_id_a = params.get("player_id_a")
            player_id_b = params.get("player_id_b")
            target_state = "War" if template.template_id == "at_war_dyad" else "Alliance"

            # Get diplomatic state
            relations = game_data.get("diplomacy", {}).get("relations", {})

            # Try both orderings of the pair
            pair_key = f"{player_id_a}_{player_id_b}"
            pair_key_alt = f"{player_id_b}_{player_id_a}"

            pair_data = relations.get(pair_key) or relations.get(pair_key_alt) or {}

            # Find state at resolution turn
            state = self._get_diplomatic_state_at_turn(pair_data, resolution_turn)

            answer = state == target_state

            return Resolution(
                answer=answer,
                resolution_turn=resolution_turn,
                state_at_resolution=state,
                computed_at=datetime.utcnow().isoformat() + "Z",
            )

        elif template.template_id == "at_war_any":
            player_id = params.get("player_id")
            relations = game_data.get("diplomacy", {}).get("relations", {})

            # Check all relations involving this player
            at_war = False
            war_with = []

            for pair_key, pair_data in relations.items():
                parts = pair_key.split("_")
                if len(parts) != 2:
                    continue

                p1, p2 = int(parts[0]), int(parts[1])
                if player_id not in (p1, p2):
                    continue

                state = self._get_diplomatic_state_at_turn(pair_data, resolution_turn)
                if state == "War":
                    at_war = True
                    other = p2 if p1 == player_id else p1
                    war_with.append(other)

            return Resolution(
                answer=at_war,
                resolution_turn=resolution_turn,
                state_at_resolution="War" if at_war else "Peace",
                event_details={"at_war_with": war_with} if war_with else None,
                computed_at=datetime.utcnow().isoformat() + "Z",
            )

        elif template.template_id == "government_at":
            player_id = params.get("player_id")
            target_gov = params.get("government_type")

            # Check events for most recent government change
            events = game_data.get("events", [])
            # Convert player_id to string for comparison (events store it as string)
            player_id_str = str(player_id)
            gov_events = [
                e for e in events
                if e.get("type") == "government_change"
                and str(e.get("player_id")) == player_id_str
                and e.get("turn", float("inf")) <= resolution_turn
            ]

            if gov_events:
                latest = max(gov_events, key=lambda e: e.get("turn", 0))
                current_gov = latest.get("metadata", {}).get("to")
                last_change_turn = latest.get("turn")
            else:
                # No government change recorded - assume starting government (typically Despotism)
                current_gov = None
                last_change_turn = None

            answer = current_gov == target_gov

            return Resolution(
                answer=answer,
                resolution_turn=resolution_turn,
                state_at_resolution=current_gov,
                event_details={"last_change_turn": last_change_turn} if last_change_turn else None,
                computed_at=datetime.utcnow().isoformat() + "Z",
            )

        # Default fallback
        return Resolution(
            answer=False,
            resolution_turn=resolution_turn,
            computed_at=datetime.utcnow().isoformat() + "Z",
        )

    def _get_diplomatic_state_at_turn(
        self,
        pair_data: dict[str, Any],
        target_turn: int,
    ) -> str | None:
        """Get diplomatic state at or closest before target turn."""
        if not pair_data:
            return None

        # Convert keys to int and sort
        turns = sorted([int(t) for t in pair_data.keys()])

        # Find closest turn <= target
        for turn in reversed(turns):
            if turn <= target_turn:
                return pair_data[str(turn)].get("state")

        return None
