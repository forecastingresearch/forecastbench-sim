"""
Event injection for world forking experiments.

Injects events into loaded game state via FreeCiv edit packets.
This enables conditional forecasting: "P(B | A happened)" with ground-truth from simulation.
"""

from civrealm.freeciv.utils.version.v13 import (
    packet_edit_mode,
    packet_edit_player,
    packet_edit_unit_create,
    packet_edit_unit_remove,
    packet_edit_unit_remove_by_id,
    packet_edit_city,
    packet_edit_game,
)


class EventInjector:
    """Inject events into loaded game state via edit packets."""

    def __init__(self, civ_controller):
        """
        Initialize event injector with a civ_controller instance.

        Args:
            civ_controller: A CivController instance with active websocket connection
        """
        self.ctrl = civ_controller
        self.edit_mode_enabled = False

    def _send_packet(self, packet: dict) -> None:
        """Send a packet via the websocket client."""
        self.ctrl.ws_client.send_request(packet)

    def enable_edit_mode(self) -> None:
        """Enable FreeCiv edit mode. Required before any edit operations."""
        if not self.edit_mode_enabled:
            self._send_packet({"pid": packet_edit_mode, "state": True})
            self.edit_mode_enabled = True

    def disable_edit_mode(self) -> None:
        """Disable FreeCiv edit mode."""
        if self.edit_mode_enabled:
            self._send_packet({"pid": packet_edit_mode, "state": False})
            self.edit_mode_enabled = False

    def _ensure_edit_mode(self) -> None:
        """Ensure edit mode is enabled before edit operations."""
        if not self.edit_mode_enabled:
            self.enable_edit_mode()

    # -------------------------------------------------------------------------
    # Player modifications (gold, tech)
    # -------------------------------------------------------------------------

    def set_gold(self, player_id: int, gold: int) -> None:
        """
        Set a player's gold to a specific amount.

        Args:
            player_id: The player number (0-indexed)
            gold: The new gold amount
        """
        self._ensure_edit_mode()
        # PACKET_EDIT_PLAYER uses 'id' field for player identification
        self._send_packet({
            "pid": packet_edit_player,
            "id": player_id,
            "gold": gold,
        })

    def give_gold(self, player_id: int, amount: int) -> None:
        """
        Give gold to a player (adds to current amount).

        Args:
            player_id: The player number
            amount: Amount of gold to add
        """
        current_gold = self.ctrl.player_ctrl.players[player_id].get('gold', 0)
        self.set_gold(player_id, current_gold + amount)

    def grant_tech(self, player_id: int, tech_id: int) -> None:
        """
        Grant a technology to a player.

        Uses PACKET_EDIT_PLAYER with the inventions field (boolean array).

        Args:
            player_id: The player number
            tech_id: The technology ID to grant
        """
        self._ensure_edit_mode()
        # PACKET_EDIT_PLAYER has 'inventions' as BOOL[A_LAST + 1]
        # We need to send the full inventions array with the tech set to True
        # For now, try sending just the specific index - may need full array
        self._send_packet({
            "pid": packet_edit_player,
            "id": player_id,
            "inventions": {tech_id: True},  # Try dict format first
        })

    def grant_tech_by_name(self, player_id: int, tech_name: str) -> int:
        """
        Grant a technology by name.

        Args:
            player_id: The player number
            tech_name: The name of the technology (e.g., "Iron Working")

        Returns:
            The tech_id that was granted, or -1 if not found
        """
        # Find tech_id from name
        for tech_id, tech_info in self.ctrl.rule_ctrl.techs.items():
            if tech_info.get('name', '').lower() == tech_name.lower():
                self.grant_tech(player_id, tech_id)
                return tech_id
        return -1

    # -------------------------------------------------------------------------
    # Unit modifications
    # -------------------------------------------------------------------------

    def create_unit(self, player_id: int, unit_type: int, x: int, y: int) -> None:
        """
        Create a unit at the specified location.

        Args:
            player_id: The player who owns the unit
            unit_type: The unit type ID
            x: X coordinate on the map
            y: Y coordinate on the map
        """
        self._ensure_edit_mode()
        self._send_packet({
            "pid": packet_edit_unit_create,
            "owner": player_id,
            "type": unit_type,
            "x": x,
            "y": y,
        })

    def remove_unit(self, unit_id: int) -> None:
        """
        Remove a unit by its ID.

        Args:
            unit_id: The unit's unique ID
        """
        self._ensure_edit_mode()
        self._send_packet({
            "pid": packet_edit_unit_remove_by_id,
            "id": unit_id,
        })

    # -------------------------------------------------------------------------
    # Chat commands (alternative approach)
    # -------------------------------------------------------------------------

    def send_command(self, command: str) -> None:
        """
        Send a server command via chat.

        Some modifications may be easier via chat commands than edit packets.
        Example: /give <player> <tech>

        Args:
            command: The command to send (without leading /)
        """
        self.ctrl.ws_client.send_message(f"/{command}")

    def set_gold_via_command(self, player_name: str, gold: int) -> None:
        """
        Set gold using server debug command.

        Args:
            player_name: The player's name (not ID)
            gold: Amount of gold to set
        """
        # FreeCiv debug command format
        self.send_command(f"setplayer {player_name} gold {gold}")

    def give_tech_via_command(self, player_name: str, tech_name: str) -> None:
        """
        Grant tech using server command (may require server-side support).

        Args:
            player_name: The player's name
            tech_name: The technology name
        """
        self.send_command(f"give {player_name} {tech_name}")

    # -------------------------------------------------------------------------
    # State queries
    # -------------------------------------------------------------------------

    def get_player_gold(self, player_id: int) -> int:
        """Get current gold for a player."""
        return self.ctrl.player_ctrl.players[player_id].get('gold', 0)

    def get_player_techs(self, player_id: int) -> list:
        """
        Get list of known tech IDs for a player.

        Returns:
            List of tech_ids that the player has researched
        """
        from civrealm.freeciv.tech.tech_const import TECH_KNOWN

        research = self.ctrl.player_ctrl.research_data.get(player_id, {})
        inventions = research.get('inventions', [])

        known_techs = []
        for tech_id, state in enumerate(inventions):
            if state == TECH_KNOWN:
                known_techs.append(tech_id)
        return known_techs

    def player_has_tech(self, player_id: int, tech_id: int) -> bool:
        """Check if a player has a specific technology."""
        return tech_id in self.get_player_techs(player_id)
