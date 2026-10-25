"""Section 5.1: Demographics

Tracks population growth and distribution over time.
"""

from typing import Dict
from .base_section import BaseSection, SectionData
from ..utils import metrics, graphs


class DemographicsSection(BaseSection):
    """Generate demographics section"""

    def __init__(self):
        super().__init__('demographics')

    def generate(
        self,
        states: Dict[int, Dict],
        config,
        data_loader,
        visualizer
    ) -> SectionData:
        """Generate demographics section

        Args:
            states: Dict mapping turn numbers to game states
            config: ReportConfig instance
            data_loader: DataLoader instance
            visualizer: MapVisualizer instance

        Returns:
            SectionData with population graphs
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

        html_parts.append('<h2>5. Social Characteristics</h2>')
        html_parts.append('<h3>5.1 Demographics</h3>')

        # Calculate population data for all turns
        population_data = {}  # {turn: {player_id: population}}

        for turn in sorted_turns:
            state = states[turn]
            population_data[turn] = {}

            if 'player' in state:
                for pid_str in state['player'].keys():
                    pid = int(pid_str)
                    # Population is sum of all city sizes
                    population_data[turn][pid] = metrics.aggregate_city_metric(state, pid, 'size')

        # Generate population graph
        if population_data:
            try:
                img_buf = graphs.create_time_series_graph(
                    data=population_data,
                    title='Population Over Time',
                    ylabel='Total Population',
                    player_names=player_names,
                    dpi=config.dpi if hasattr(config, 'dpi') else 150
                )
                img_name = 'population_over_time'
                images[img_name] = img_buf
                html_parts.append(f'<div class="graph">')
                html_parts.append(f'<img src="{img_name}.png" alt="Population over time"/>')
                html_parts.append('</div>')

                # Add summary statistics table
                html_parts.append('<h4>Population Statistics</h4>')

                # Calculate final populations
                final_turn = max(sorted_turns)
                final_populations = population_data[final_turn]

                table_rows = []
                for pid in sorted(player_names.keys()):
                    name = player_names[pid]
                    pop = final_populations.get(pid, 0)
                    table_rows.append([name, str(int(pop))])

                # Sort by population descending
                table_rows.sort(key=lambda x: int(x[1]), reverse=True)

                pop_table = self._format_html_table(
                    headers=['Civilization', 'Final Population'],
                    rows=table_rows,
                    caption=f'Population at Turn {final_turn}'
                )
                html_parts.append(pop_table)

            except Exception as e:
                print(f"Warning: Failed to generate population graph: {e}")
                html_parts.append('<p>Population data not available.</p>')
        else:
            html_parts.append('<p>No population data available.</p>')

        content = '\n'.join(html_parts)

        return SectionData(
            title='Demographics',
            content=content,
            images=images,
            metadata={
                'turns_analyzed': len(sorted_turns)
            }
        )
