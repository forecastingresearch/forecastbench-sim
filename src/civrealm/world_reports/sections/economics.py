"""Section 4: Economics

Tracks economic metrics including treasury, trade, agriculture, and industry.
"""

from typing import Dict
from .base_section import BaseSection, SectionData
from ..utils import metrics, graphs


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

        # Get player names from first state
        first_state = states[sorted_turns[0]] if sorted_turns else {}
        player_names = {}
        if 'player' in first_state:
            for pid, player in first_state['player'].items():
                if isinstance(player, dict):
                    pid_int = int(pid)
                    player_names[pid_int] = self._get_player_name(pid_int, first_state, data_loader)

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

                    # Treasury (gold)
                    gold_data[turn][pid] = metrics.get_player_gold(state, pid)

                    # Trade production
                    trade_data[turn][pid] = metrics.aggregate_city_metric(state, pid, 'prod_trade')

                    # Food production
                    food_data[turn][pid] = metrics.aggregate_city_metric(state, pid, 'prod_food')

                    # Shield production (industry)
                    shields_data[turn][pid] = metrics.aggregate_city_metric(state, pid, 'prod_shield')

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
