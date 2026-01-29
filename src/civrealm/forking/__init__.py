# World forking infrastructure for conditional forecasting
#
# Primary approach: SavegameModifier
#   - Modify savegame files directly (reliable, tested)
#   - Load → modify text → save → reload in game
#
# ForkManager: Orchestrate multiple parallel simulations
#   - Create forks from checkpoints with different modifications
#   - Run forks and compare outcomes
#   - Enables conditional forecasting: P(B|A) questions
#
# Experimental: EventInjector
#   - Edit packets require all fields (complex)
#   - Console commands don't seem to work for gold/tech
#   - May work for simpler operations

from civrealm.forking.savegame_modifier import SavegameModifier
from civrealm.forking.event_injector import EventInjector
from civrealm.forking.fork_manager import ForkManager, Fork, ForkResult

__all__ = ['SavegameModifier', 'EventInjector', 'ForkManager', 'Fork', 'ForkResult']
