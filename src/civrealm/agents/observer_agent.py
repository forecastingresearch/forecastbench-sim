"""Observer Agent for recording complete game state

This agent connects as an observer to record complete, unfiltered game state
for world report generation, while not participating in gameplay.
"""


class ObserverAgent:
    """Agent that observes game without taking actions

    Connects as an observer to record complete game state including
    all players' data (no fog of war), but does not take any actions.
    Used purely for data collection for world reports.
    """

    def __init__(self):
        self.name = "Observer"

    def act(self, observation, info):
        """Observer takes no actions, just records state

        Args:
            observation: Game observation (with complete state for observer)
            info: Additional game information

        Returns:
            None to indicate no action
        """
        # Observer never takes actions
        return None
