"""Section 6: Science and Technology

Tracks scientific progress and technological advancement.
"""

from typing import Dict
from .base_section import BaseSection, SectionData
from ..utils import metrics, graphs


class TechnologySection(BaseSection):
    """Generate technology section"""

    def __init__(self):
        super().__init__('technology')

    def generate(
        self,
        states: Dict[int, Dict],
        config,
        data_loader,
        visualizer
    ) -> SectionData:
        """Generate technology section

        Args:
            states: Dict mapping turn numbers to game states
            config: ReportConfig instance
            data_loader: DataLoader instance
            visualizer: MapVisualizer instance

        Returns:
            SectionData with science and technology graphs
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

        html_parts.append('<h2>6. Science and Technology</h2>')

        # Calculate science and tech data for all turns
        science_data = {}    # {turn: {player_id: science_per_turn}}
        tech_count_data = {} # {turn: {player_id: num_techs}}

        for turn in sorted_turns:
            state = states[turn]
            science_data[turn] = {}
            tech_count_data[turn] = {}

            if 'player' in state:
                for pid_str, player in state['player'].items():
                    if isinstance(player, dict):
                        pid = int(pid_str)

                        # Science production per turn
                        science_data[turn][pid] = metrics.get_player_science_production(state, pid)

                        # Count known technologies
                        tech_count_data[turn][pid] = metrics.count_known_techs(player)

        # Section 6.1: Science Production
        html_parts.append('<h3>6.1 Science Production</h3>')
        if science_data:
            try:
                img_buf = graphs.create_time_series_graph(
                    data=science_data,
                    title='Science Production Over Time',
                    ylabel='Science per Turn',
                    player_names=player_names,
                    dpi=config.dpi if hasattr(config, 'dpi') else 150
                )
                img_name = 'science_over_time'
                images[img_name] = img_buf
                html_parts.append(f'<div class="graph">')
                html_parts.append(f'<img src="{img_name}.png" alt="Science production over time"/>')
                html_parts.append('</div>')
            except Exception as e:
                print(f"Warning: Failed to generate science production graph: {e}")
                html_parts.append('<p>Science production data not available.</p>')
        else:
            html_parts.append('<p>No science production data available.</p>')

        # Section 6.2: Technology Count
        html_parts.append('<h3>6.2 Technological Progress</h3>')
        if tech_count_data:
            try:
                img_buf = graphs.create_time_series_graph(
                    data=tech_count_data,
                    title='Technologies Known Over Time',
                    ylabel='Number of Technologies',
                    player_names=player_names,
                    dpi=config.dpi if hasattr(config, 'dpi') else 150
                )
                img_name = 'tech_count_over_time'
                images[img_name] = img_buf
                html_parts.append(f'<div class="graph">')
                html_parts.append(f'<img src="{img_name}.png" alt="Technology count over time"/>')
                html_parts.append('</div>')

                # Add summary statistics table
                html_parts.append('<h4>Technology Summary</h4>')

                # Calculate final tech counts
                final_turn = max(sorted_turns)
                final_tech_counts = tech_count_data[final_turn]

                table_rows = []
                for pid in sorted(player_names.keys()):
                    name = player_names[pid]
                    tech_count = final_tech_counts.get(pid, 0)

                    # Calculate science production at final turn
                    final_science = science_data[final_turn].get(pid, 0)

                    table_rows.append([
                        name,
                        str(int(tech_count)),
                        str(int(final_science))
                    ])

                # Sort by tech count descending
                table_rows.sort(key=lambda x: int(x[1]), reverse=True)

                tech_table = self._format_html_table(
                    headers=['Civilization', 'Technologies Known', 'Science per Turn'],
                    rows=table_rows,
                    caption=f'Technological Status at Turn {final_turn}'
                )
                html_parts.append(tech_table)

            except Exception as e:
                print(f"Warning: Failed to generate technology count graph: {e}")
                html_parts.append('<p>Technology data not available.</p>')
        else:
            html_parts.append('<p>No technology data available.</p>')

        content = '\n'.join(html_parts)

        return SectionData(
            title='Science and Technology',
            content=content,
            images=images,
            metadata={
                'turns_analyzed': len(sorted_turns)
            }
        )
