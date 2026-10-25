"""Graph generation from JSON data

This module provides functions to create graphs from world report JSON data.
It wraps the low-level matplotlib functions in utils/graphs.py with a
JSON-friendly interface.
"""

from typing import Dict, Any
from io import BytesIO

from ..utils import graphs


class GraphGenerator:
    """Generates graphs from world report JSON data"""

    def __init__(self, dpi: int = 150, style: str = 'seaborn'):
        """Initialize graph generator

        Args:
            dpi: Image DPI (default: 150)
            style: Matplotlib style (default: 'seaborn')
        """
        self.dpi = dpi
        self.style = style

    def create_time_series_graph(
        self,
        json_data: Dict[str, Any],
        metric_name: str,
        title: str,
        ylabel: str,
        xlabel: str = 'Turn'
    ) -> BytesIO:
        """Create a time series graph from JSON data

        Args:
            json_data: Complete world report JSON data
            metric_name: Name of metric in time_series (e.g., 'treasury')
            title: Graph title
            ylabel: Y-axis label
            xlabel: X-axis label (default: 'Turn')

        Returns:
            BytesIO buffer containing PNG image

        Raises:
            KeyError: If metric_name not found in JSON data
        """
        # Extract time series data for the metric
        if 'time_series' not in json_data:
            raise KeyError("'time_series' not found in JSON data")

        if metric_name not in json_data['time_series']:
            raise KeyError(f"Metric '{metric_name}' not found in time_series")

        metric_data = json_data['time_series'][metric_name]

        # Convert string keys to integers for consistency
        data = {}
        for turn_str, player_values in metric_data.items():
            turn = int(turn_str)
            data[turn] = {}
            for player_id_str, value in player_values.items():
                player_id = int(player_id_str)
                data[turn][player_id] = float(value)

        # Get player names
        player_names = self._get_player_names(json_data)

        # Generate graph using low-level function
        return graphs.create_time_series_graph(
            data=data,
            title=title,
            ylabel=ylabel,
            player_names=player_names,
            xlabel=xlabel,
            dpi=self.dpi,
            style=self.style
        )

    def create_treasury_graph(self, json_data: Dict[str, Any]) -> BytesIO:
        """Create treasury over time graph

        Args:
            json_data: Complete world report JSON data

        Returns:
            BytesIO buffer containing PNG image
        """
        return self.create_time_series_graph(
            json_data=json_data,
            metric_name='treasury',
            title='Treasury Over Time',
            ylabel='Gold'
        )

    def create_population_graph(self, json_data: Dict[str, Any]) -> BytesIO:
        """Create population over time graph

        Args:
            json_data: Complete world report JSON data

        Returns:
            BytesIO buffer containing PNG image
        """
        return self.create_time_series_graph(
            json_data=json_data,
            metric_name='population',
            title='Population Over Time',
            ylabel='Total Population'
        )

    def create_science_graph(self, json_data: Dict[str, Any]) -> BytesIO:
        """Create science production over time graph

        Args:
            json_data: Complete world report JSON data

        Returns:
            BytesIO buffer containing PNG image
        """
        return self.create_time_series_graph(
            json_data=json_data,
            metric_name='science',
            title='Science Production Over Time',
            ylabel='Science per Turn'
        )

    def create_tech_count_graph(self, json_data: Dict[str, Any]) -> BytesIO:
        """Create technologies known over time graph

        Args:
            json_data: Complete world report JSON data

        Returns:
            BytesIO buffer containing PNG image
        """
        return self.create_time_series_graph(
            json_data=json_data,
            metric_name='techs_known',
            title='Technologies Known Over Time',
            ylabel='Number of Technologies'
        )

    def create_trade_graph(self, json_data: Dict[str, Any]) -> BytesIO:
        """Create trade production over time graph

        Args:
            json_data: Complete world report JSON data

        Returns:
            BytesIO buffer containing PNG image
        """
        return self.create_time_series_graph(
            json_data=json_data,
            metric_name='trade_production',
            title='Trade Production Over Time',
            ylabel='Trade Output'
        )

    def create_food_graph(self, json_data: Dict[str, Any]) -> BytesIO:
        """Create food production over time graph

        Args:
            json_data: Complete world report JSON data

        Returns:
            BytesIO buffer containing PNG image
        """
        return self.create_time_series_graph(
            json_data=json_data,
            metric_name='food_production',
            title='Food Production Over Time',
            ylabel='Food Output'
        )

    def create_shields_graph(self, json_data: Dict[str, Any]) -> BytesIO:
        """Create industrial production over time graph

        Args:
            json_data: Complete world report JSON data

        Returns:
            BytesIO buffer containing PNG image
        """
        return self.create_time_series_graph(
            json_data=json_data,
            metric_name='shield_production',
            title='Industrial Production Over Time',
            ylabel='Shield Output'
        )

    def create_territory_graph(self, json_data: Dict[str, Any]) -> BytesIO:
        """Create territory size over time graph

        Args:
            json_data: Complete world report JSON data

        Returns:
            BytesIO buffer containing PNG image
        """
        return self.create_time_series_graph(
            json_data=json_data,
            metric_name='territory_size',
            title='Territory Size Over Time',
            ylabel='Tiles Controlled'
        )

    def create_arable_land_graph(self, json_data: Dict[str, Any]) -> BytesIO:
        """Create arable land over time graph

        Args:
            json_data: Complete world report JSON data

        Returns:
            BytesIO buffer containing PNG image
        """
        return self.create_time_series_graph(
            json_data=json_data,
            metric_name='arable_land',
            title='Arable Land Over Time',
            ylabel='Arable Tiles'
        )

    def _get_player_names(self, json_data: Dict[str, Any]) -> Dict[int, str]:
        """Extract player names from JSON data

        Args:
            json_data: Complete world report JSON data

        Returns:
            Dict mapping player_id to civilization name
        """
        if 'civilizations' not in json_data:
            return {}

        player_names = {}
        for player_id_str, civ_info in json_data['civilizations'].items():
            player_id = int(player_id_str)
            player_names[player_id] = civ_info.get('name', f'Player {player_id}')

        return player_names
