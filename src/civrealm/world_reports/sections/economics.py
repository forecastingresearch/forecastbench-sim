"""Section 4: Economics

Tracks economic metrics including treasury, trade, agriculture, and industry.
"""

from typing import Dict
from .base_section import BaseSection, SectionData
from ..utils import metrics, graphs
from ..utils.savegame_parser import get_savegame_data_for_report


class EconomicsSection(BaseSection):
    """Generate economics section"""

    def __init__(self):
        super().__init__('economics')

    def generate(
        self,
        states: Dict[int, Dict],
        config,
        data_loader,
        visualizer
    ) -> SectionData:
        """Generate economics section

        Args:
            states: Dict mapping turn numbers to game states
            config: ReportConfig instance
            data_loader: DataLoader instance
            visualizer: MapVisualizer instance

        Returns:
            SectionData with economic graphs and tables
        """
        html_parts = []
        images = {}

        # Sort turns
        sorted_turns = sorted(states.keys())

        # Collect player names from all states (to include late-joining players)
        player_names = self._collect_all_player_names(states, data_loader)
        # Filter to only active players
        player_names = self._filter_active_players(player_names, states)

        html_parts.append('<h2>4. Economics</h2>')

        # Calculate economic metrics for all turns
        gold_data = {}      # {turn: {player_id: gold}}
        trade_data = {}     # {turn: {player_id: trade}}
        food_data = {}      # {turn: {player_id: food}}
        shields_data = {}   # {turn: {player_id: shields}}

        for turn in sorted_turns:
            state = states[turn]
            gold_data[turn] = {}
            trade_data[turn] = {}
            food_data[turn] = {}
            shields_data[turn] = {}

            if 'player' in state:
                for pid_str in state['player'].keys():
                    pid = int(pid_str)

                    # Treasury (gold) - from recordings
                    gold_data[turn][pid] = metrics.get_player_gold(state, pid)

                    # For production, use recordings (fog-of-war limited)
                    trade_data[turn][pid] = metrics.aggregate_city_metric(state, pid, 'prod_trade')
                    food_data[turn][pid] = metrics.aggregate_city_metric(state, pid, 'prod_food')
                    shields_data[turn][pid] = metrics.aggregate_city_metric(state, pid, 'prod_shield')

            # Try to override with complete savegame data for this turn
            savegame_data = get_savegame_data_for_report(config, turn)
            if savegame_data and 'production' in savegame_data:
                for player_id, prod in savegame_data['production'].items():
                    trade_data[turn][player_id] = prod['trade']
                    food_data[turn][player_id] = prod['food']
                    shields_data[turn][player_id] = prod['shields']

                    # Add player names for players that only appear in savegame data
                    # Only add if they are alive in the final state
                    if player_id not in player_names:
                        # Check if this player is alive in the final state before adding
                        final_turn = max(sorted_turns)
                        final_state = states[final_turn]
                        if 'player' in final_state:
                            player_key = str(player_id)
                            player_info = final_state['player'].get(player_key) or final_state['player'].get(player_id)
                            if isinstance(player_info, dict) and player_info.get('is_alive', False):
                                # Try to get nation name from savegame data
                                if 'nations' in savegame_data and player_id in savegame_data['nations']:
                                    nation_id = savegame_data['nations'][player_id]
                                    civ_name = self._get_nation_name_from_savegame_data(nation_id, data_loader)
                                    if civ_name:
                                        player_names[player_id] = civ_name
                                    else:
                                        player_names[player_id] = f'Player {player_id}'
                                else:
                                    player_names[player_id] = self._get_player_name(player_id, state, data_loader)

        # Section 4.1: Treasury (Gold)
        html_parts.append('<h3>4.1 Treasury</h3>')
        if gold_data:
            try:
                img_buf = graphs.create_time_series_graph(
                    data=gold_data,
                    title='Treasury Over Time',
                    ylabel='Gold',
                    player_names=player_names,
                    dpi=config.dpi if hasattr(config, 'dpi') else 150
                )
                img_name = 'treasury_over_time'
                images[img_name] = img_buf
                html_parts.append(f'<div class="graph">')
                html_parts.append(f'<img src="{img_name}.png" alt="Treasury over time"/>')
                html_parts.append('</div>')
            except Exception as e:
                print(f"Warning: Failed to generate treasury graph: {e}")
                html_parts.append('<p>Treasury data not available.</p>')
        else:
            html_parts.append('<p>No treasury data available.</p>')

        # Section 4.2: Trade
        html_parts.append('<h3>4.2 Trade</h3>')
        if trade_data:
            try:
                img_buf = graphs.create_time_series_graph(
                    data=trade_data,
                    title='Trade Production Over Time',
                    ylabel='Trade Output',
                    player_names=player_names,
                    dpi=config.dpi if hasattr(config, 'dpi') else 150
                )
                img_name = 'trade_over_time'
                images[img_name] = img_buf
                html_parts.append(f'<div class="graph">')
                html_parts.append(f'<img src="{img_name}.png" alt="Trade production over time"/>')
                html_parts.append('</div>')
            except Exception as e:
                print(f"Warning: Failed to generate trade graph: {e}")
                html_parts.append('<p>Trade data not available.</p>')
        else:
            html_parts.append('<p>No trade data available.</p>')

        # Section 4.3: Agriculture
        html_parts.append('<h3>4.3 Agriculture</h3>')
        if food_data:
            try:
                img_buf = graphs.create_time_series_graph(
                    data=food_data,
                    title='Food Production Over Time',
                    ylabel='Food Output',
                    player_names=player_names,
                    dpi=config.dpi if hasattr(config, 'dpi') else 150
                )
                img_name = 'food_over_time'
                images[img_name] = img_buf
                html_parts.append(f'<div class="graph">')
                html_parts.append(f'<img src="{img_name}.png" alt="Food production over time"/>')
                html_parts.append('</div>')
            except Exception as e:
                print(f"Warning: Failed to generate food graph: {e}")
                html_parts.append('<p>Food production data not available.</p>')
        else:
            html_parts.append('<p>No food production data available.</p>')

        # Section 4.4: Industry
        html_parts.append('<h3>4.4 Industry</h3>')
        if shields_data:
            try:
                img_buf = graphs.create_time_series_graph(
                    data=shields_data,
                    title='Industrial Production Over Time',
                    ylabel='Shield Output',
                    player_names=player_names,
                    dpi=config.dpi if hasattr(config, 'dpi') else 150
                )
                img_name = 'shields_over_time'
                images[img_name] = img_buf
                html_parts.append(f'<div class="graph">')
                html_parts.append(f'<img src="{img_name}.png" alt="Industrial production over time"/>')
                html_parts.append('</div>')
            except Exception as e:
                print(f"Warning: Failed to generate shields graph: {e}")
                html_parts.append('<p>Industrial production data not available.</p>')
        else:
            html_parts.append('<p>No industrial production data available.</p>')

        content = '\n'.join(html_parts)

        return SectionData(
            title='Economics',
            content=content,
            images=images,
            metadata={
                'turns_analyzed': len(sorted_turns)
            }
        )
