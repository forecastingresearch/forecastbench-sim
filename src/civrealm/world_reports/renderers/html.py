"""HTML renderer for world reports"""

from typing import Dict, Any, List
from pathlib import Path
from io import BytesIO

from .graph_generator import GraphGenerator


class HTMLRenderer:
    """Render world report JSON data to HTML"""

    def __init__(self, output_dir: str, turn: int, recording_dir: str = None,
                 data_loader: Any = None, visualizer: Any = None):
        """Initialize HTML renderer

        Args:
            output_dir: Directory to save output files
            turn: Turn number for this report
            recording_dir: Path to recording directory (for territory maps)
            data_loader: DataLoader instance (for territory maps)
            visualizer: MapVisualizer instance (for territory maps)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.turn = turn
        self.images_dir = self.output_dir / f'turn_{turn:03d}_images'
        self.images_dir.mkdir(exist_ok=True)

        self.recording_dir = recording_dir
        self.data_loader = data_loader
        self.visualizer = visualizer

        # Images dictionary: {name: BytesIO}
        self.images = {}

    def render_from_json(
        self,
        json_data: Dict[str, Any],
        output_file: str = None
    ) -> str:
        """Render complete HTML report from JSON data

        Args:
            json_data: World report JSON data
            output_file: Optional specific output file path

        Returns:
            Path to generated HTML file
        """
        # Initialize graph generator
        graph_gen = GraphGenerator(dpi=150, style='seaborn')

        # Build HTML document
        html_parts = []
        html_parts.append(self._get_html_header())

        # Title page
        turn = json_data['metadata']['turn']
        html_parts.append('<div class="title-page">')
        html_parts.append(f'<h1>World Report</h1>')
        html_parts.append(f'<h2>Turn {turn}</h2>')
        html_parts.append('<p class="subtitle">Comprehensive Analysis of World State</p>')
        html_parts.append('</div>')

        # Table of contents
        html_parts.append(self._render_toc())

        # Section 1: Overview
        html_parts.append('<div id="section1" class="section">')
        html_parts.append(self._render_overview(json_data))
        html_parts.append('</div>')

        # Section 2: Historical Events
        html_parts.append('<div id="section2" class="section">')
        html_parts.append(self._render_historical_events(json_data, graph_gen))
        html_parts.append('</div>')

        # Section 3: Economics (split into subsections)
        html_parts.append('<div id="section3" class="section">')
        html_parts.append(self._render_economics(json_data, graph_gen))
        html_parts.append('</div>')

        # Section 4: Demographics
        html_parts.append('<div id="section4" class="section">')
        html_parts.append(self._render_demographics(json_data, graph_gen))
        html_parts.append('</div>')

        # Section 5: Science and Technology
        html_parts.append('<div id="section5" class="section">')
        html_parts.append(self._render_technology(json_data, graph_gen))
        html_parts.append('</div>')

        html_parts.append(self._get_html_footer())

        # Save all images
        for img_name, img_buf in self.images.items():
            img_path = self.images_dir / f'{img_name}.png'
            img_buf.seek(0)
            with open(img_path, 'wb') as f:
                f.write(img_buf.read())

        # Write HTML file
        if output_file:
            html_file = Path(output_file)
        else:
            html_file = self.output_dir / f'turn_{self.turn:03d}_report.html'

        with open(html_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(html_parts))

        return str(html_file)

    def _render_toc(self) -> str:
        """Render table of contents"""
        sections = [
            "Scenario Overview",
            "Major Historical Events",
            "Economics",
            "Social Characteristics",
            "Science and Technology"
        ]

        html = ['<div class="toc">']
        html.append('<h2>Table of Contents</h2>')
        html.append('<ol>')
        for idx, section_title in enumerate(sections, 1):
            html.append(f'<li><a href="#section{idx}">{section_title}</a></li>')
        html.append('</ol>')
        html.append('</div>')

        return '\n'.join(html)

    def _render_overview(self, json_data: Dict[str, Any]) -> str:
        """Render overview section from JSON"""
        html = []
        metadata = json_data['metadata']
        civilizations = json_data['civilizations']

        html.append('<h2>1. Scenario Overview</h2>')

        # Basic info
        html.append('<h3>1.1 World Information</h3>')
        html.append('<div class="overview-info">')

        turns = metadata.get('turns_analyzed', [])
        if turns:
            first_turn = min(turns)
            last_turn = max(turns)
            html.append(f'<p><strong>Turns Covered:</strong> {first_turn} to {last_turn} ({len(turns)} turns)</p>')

        html.append(f'<p><strong>Number of Civilizations:</strong> {metadata["num_civilizations"]}</p>')

        map_size = metadata.get('map_size', [0, 0])
        if map_size[0] > 0:
            html.append(f'<p><strong>Map Size:</strong> {map_size[0]} × {map_size[1]}</p>')

        html.append('</div>')

        # Civilizations table
        if civilizations:
            html.append('<h3>1.2 Civilizations</h3>')

            # Get final scores from snapshots
            snapshots = json_data.get('snapshots', {})
            turn_str = str(metadata['turn'])

            if turn_str in snapshots and 'rankings' in snapshots[turn_str]:
                rankings = snapshots[turn_str]['rankings']

                rows = []
                for ranking in rankings:
                    rank = ranking['rank']
                    player_id = ranking['player_id']
                    score = ranking['score']
                    civ_name = civilizations.get(str(player_id), {}).get('name', f'Player {player_id}')
                    rows.append(f'<tr><td>{rank}</td><td>{civ_name}</td><td>{score}</td></tr>')

                html.append('<table class="data-table">')
                html.append(f'<caption>Civilizations at Turn {metadata["turn"]}</caption>')
                html.append('<thead><tr><th>Rank</th><th>Civilization</th><th>Score</th></tr></thead>')
                html.append('<tbody>')
                html.append('\n'.join(rows))
                html.append('</tbody>')
                html.append('</table>')

        # World statistics
        html.append('<h3>1.3 World Statistics</h3>')

        if turn_str in snapshots and 'world_totals' in snapshots[turn_str]:
            totals = snapshots[turn_str]['world_totals']
            html.append('<div class="statistics">')
            html.append('<ul>')
            html.append(f'<li><strong>Total Cities:</strong> {totals.get("total_cities", 0)}</li>')
            html.append(f'<li><strong>Total Units:</strong> {totals.get("total_units", 0)}</li>')
            html.append(f'<li><strong>Total Population:</strong> {totals.get("total_population", 0)}</li>')
            html.append('</ul>')
            html.append('</div>')

        return '\n'.join(html)

    def _render_historical_events(
        self,
        json_data: Dict[str, Any],
        graph_gen: GraphGenerator
    ) -> str:
        """Render historical events section from JSON"""
        html = []
        events = json_data.get('events', [])

        html.append('<h2>2. Major Historical Events</h2>')

        # City events
        html.append('<h3>2.1 City Founding and Conquest</h3>')
        city_events = [e for e in events if e['type'] in
                      ['city_founded', 'city_conquered', 'city_destroyed']]

        if city_events:
            rows = []
            for event in city_events:
                turn_str = f"Turn {event['turn']}"
                event_type = event['type'].replace('_', ' ').title()
                description = event['description']

                if 'location' in event:
                    x, y = event['location']
                    description += f" at ({x}, {y})"

                rows.append(f'<tr><td>{turn_str}</td><td>{event_type}</td><td>{description}</td></tr>')

            html.append('<table class="data-table">')
            html.append('<caption>City Events Throughout History</caption>')
            html.append('<thead><tr><th>Turn</th><th>Event Type</th><th>Description</th></tr></thead>')
            html.append('<tbody>')
            html.append('\n'.join(rows))
            html.append('</tbody>')
            html.append('</table>')
        else:
            html.append('<p>No city events recorded.</p>')

        # Territory maps (if visualizer available)
        html.append('<h3>2.2 Territorial Control</h3>')

        if self.visualizer and self.data_loader:
            territory_turns = json_data.get('territory_snapshots', {}).get('turns', [])
            if territory_turns:
                html.append('<div class="mini-maps-grid">')

                for snapshot_turn in territory_turns:
                    try:
                        # Load state for this turn
                        state = self.data_loader.get_state(snapshot_turn)
                        if state and 'map' in state and 'player' in state:
                            # Generate territory map
                            img_buf = self.visualizer.render_territory_map(
                                map_state=state['map'],
                                player_state=state['player'],
                                title=f'Turn {snapshot_turn}',
                                highlight_cities=True,
                                show_legend=True
                            )

                            img_name = f'territory_turn_{snapshot_turn}'
                            self.images[img_name] = img_buf

                            html.append('<div class="mini-map">')
                            html.append(f'<img src="turn_{self.turn:03d}_images/{img_name}.png" alt="Territory at turn {snapshot_turn}"/>')
                            html.append(f'<div class="caption">Turn {snapshot_turn}</div>')
                            html.append('</div>')
                    except Exception as e:
                        print(f"Warning: Failed to generate territory map for turn {snapshot_turn}: {e}")

                html.append('</div>')
        else:
            html.append('<p><em>Territory maps require visualizer and data_loader.</em></p>')

        # Territory graphs
        html.append('<h3>2.3 Territory Expansion</h3>')
        try:
            img_buf = graph_gen.create_territory_graph(json_data)
            img_name = 'territory_over_time'
            self.images[img_name] = img_buf
            html.append(f'<div class="graph">')
            html.append(f'<img src="turn_{self.turn:03d}_images/{img_name}.png" alt="Territory over time"/>')
            html.append('</div>')
        except Exception as e:
            print(f"Warning: Failed to generate territory graph: {e}")
            html.append('<p>Territory data not available.</p>')

        # Arable land graph
        html.append('<h3>2.4 Arable Land</h3>')
        try:
            img_buf = graph_gen.create_arable_land_graph(json_data)
            img_name = 'arable_land_over_time'
            self.images[img_name] = img_buf
            html.append(f'<div class="graph">')
            html.append(f'<img src="turn_{self.turn:03d}_images/{img_name}.png" alt="Arable land over time"/>')
            html.append('</div>')
        except Exception as e:
            print(f"Warning: Failed to generate arable land graph: {e}")
            html.append('<p>Arable land data not available.</p>')

        # Technology discoveries
        html.append('<h3>2.5 Technology Discoveries</h3>')
        tech_events = [e for e in events if e['type'] == 'tech_discovered']

        if tech_events:
            rows = []
            for event in tech_events:  # Show all discoveries
                turn_str = f"Turn {event['turn']}"
                description = event['description']
                rows.append(f'<tr><td>{turn_str}</td><td>{description}</td></tr>')

            html.append('<table class="data-table">')
            html.append(f'<caption>Technology Discoveries ({len(tech_events)} total)</caption>')
            html.append('<thead><tr><th>Turn</th><th>Discovery</th></tr></thead>')
            html.append('<tbody>')
            html.append('\n'.join(rows))
            html.append('</tbody>')
            html.append('</table>')
        else:
            html.append('<p>No technology discoveries recorded.</p>')

        # Government changes
        html.append('<h3>2.6 Government Changes</h3>')
        gov_events = [e for e in events if e['type'] == 'government_change']

        if gov_events:
            rows = []
            for event in gov_events:
                turn_str = f"Turn {event['turn']}"
                description = event['description']
                rows.append(f'<tr><td>{turn_str}</td><td>{description}</td></tr>')

            html.append('<table class="data-table">')
            html.append('<caption>Government Changes</caption>')
            html.append('<thead><tr><th>Turn</th><th>Change</th></tr></thead>')
            html.append('<tbody>')
            html.append('\n'.join(rows))
            html.append('</tbody>')
            html.append('</table>')
        else:
            html.append('<p>No government changes recorded.</p>')

        return '\n'.join(html)

    def _render_economics(
        self,
        json_data: Dict[str, Any],
        graph_gen: GraphGenerator
    ) -> str:
        """Render economics section from JSON"""
        html = []

        html.append('<h2>3. Economics</h2>')

        # Treasury
        html.append('<h3>3.1 Treasury</h3>')
        try:
            img_buf = graph_gen.create_treasury_graph(json_data)
            img_name = 'treasury_over_time'
            self.images[img_name] = img_buf
            html.append(f'<div class="graph">')
            html.append(f'<img src="turn_{self.turn:03d}_images/{img_name}.png" alt="Treasury over time"/>')
            html.append('</div>')
        except Exception as e:
            print(f"Warning: Failed to generate treasury graph: {e}")
            html.append('<p>Treasury data not available.</p>')

        # Trade
        html.append('<h3>3.2 Trade</h3>')
        try:
            img_buf = graph_gen.create_trade_graph(json_data)
            img_name = 'trade_over_time'
            self.images[img_name] = img_buf
            html.append(f'<div class="graph">')
            html.append(f'<img src="turn_{self.turn:03d}_images/{img_name}.png" alt="Trade production over time"/>')
            html.append('</div>')
        except Exception as e:
            print(f"Warning: Failed to generate trade graph: {e}")
            html.append('<p>Trade data not available.</p>')

        # Agriculture
        html.append('<h3>3.3 Agriculture</h3>')
        try:
            img_buf = graph_gen.create_food_graph(json_data)
            img_name = 'food_over_time'
            self.images[img_name] = img_buf
            html.append(f'<div class="graph">')
            html.append(f'<img src="turn_{self.turn:03d}_images/{img_name}.png" alt="Food production over time"/>')
            html.append('</div>')
        except Exception as e:
            print(f"Warning: Failed to generate food graph: {e}")
            html.append('<p>Food production data not available.</p>')

        # Industry
        html.append('<h3>3.4 Industry</h3>')
        try:
            img_buf = graph_gen.create_shields_graph(json_data)
            img_name = 'shields_over_time'
            self.images[img_name] = img_buf
            html.append(f'<div class="graph">')
            html.append(f'<img src="turn_{self.turn:03d}_images/{img_name}.png" alt="Industrial production over time"/>')
            html.append('</div>')
        except Exception as e:
            print(f"Warning: Failed to generate shields graph: {e}")
            html.append('<p>Industrial production data not available.</p>')

        return '\n'.join(html)

    def _render_demographics(
        self,
        json_data: Dict[str, Any],
        graph_gen: GraphGenerator
    ) -> str:
        """Render demographics section from JSON"""
        html = []

        html.append('<h2>4. Social Characteristics</h2>')
        html.append('<h3>4.1 Demographics</h3>')

        # Population graph
        try:
            img_buf = graph_gen.create_population_graph(json_data)
            img_name = 'population_over_time'
            self.images[img_name] = img_buf
            html.append(f'<div class="graph">')
            html.append(f'<img src="turn_{self.turn:03d}_images/{img_name}.png" alt="Population over time"/>')
            html.append('</div>')

            # Population statistics table
            html.append('<h4>Population Statistics</h4>')

            civilizations = json_data['civilizations']
            time_series = json_data['time_series']
            population_data = time_series.get('population', {})

            if population_data:
                # Get final turn data
                turns = sorted([int(t) for t in population_data.keys()])
                final_turn = max(turns)
                final_pop = population_data[str(final_turn)]

                rows = []
                for player_id_str in sorted(civilizations.keys(), key=int):
                    player_id = int(player_id_str)
                    civ_name = civilizations[player_id_str]['name']
                    pop = final_pop.get(str(player_id), 0)
                    rows.append(f'<tr><td>{civ_name}</td><td>{int(pop)}</td></tr>')

                # Sort by population
                rows.sort(key=lambda x: int(x.split('<td>')[2].split('</td>')[0]), reverse=True)

                html.append('<table class="data-table">')
                html.append(f'<caption>Population at Turn {final_turn}</caption>')
                html.append('<thead><tr><th>Civilization</th><th>Final Population</th></tr></thead>')
                html.append('<tbody>')
                html.append('\n'.join(rows))
                html.append('</tbody>')
                html.append('</table>')

        except Exception as e:
            print(f"Warning: Failed to generate population content: {e}")
            html.append('<p>Population data not available.</p>')

        return '\n'.join(html)

    def _render_technology(
        self,
        json_data: Dict[str, Any],
        graph_gen: GraphGenerator
    ) -> str:
        """Render technology section from JSON"""
        html = []

        html.append('<h2>5. Science and Technology</h2>')

        # Science production
        html.append('<h3>5.1 Science Production</h3>')
        try:
            img_buf = graph_gen.create_science_graph(json_data)
            img_name = 'science_over_time'
            self.images[img_name] = img_buf
            html.append(f'<div class="graph">')
            html.append(f'<img src="turn_{self.turn:03d}_images/{img_name}.png" alt="Science production over time"/>')
            html.append('</div>')
        except Exception as e:
            print(f"Warning: Failed to generate science graph: {e}")
            html.append('<p>Science production data not available.</p>')

        # Technology count
        html.append('<h3>5.2 Technological Progress</h3>')
        try:
            img_buf = graph_gen.create_tech_count_graph(json_data)
            img_name = 'tech_count_over_time'
            self.images[img_name] = img_buf
            html.append(f'<div class="graph">')
            html.append(f'<img src="turn_{self.turn:03d}_images/{img_name}.png" alt="Technology count over time"/>')
            html.append('</div>')

            # Technology summary table
            html.append('<h4>Technology Summary</h4>')

            civilizations = json_data['civilizations']
            time_series = json_data['time_series']
            tech_data = time_series.get('techs_known', {})
            science_data = time_series.get('science', {})

            if tech_data and science_data:
                # Get final turn data
                turns = sorted([int(t) for t in tech_data.keys()])
                final_turn = max(turns)
                final_tech = tech_data[str(final_turn)]
                final_science = science_data[str(final_turn)]

                rows = []
                for player_id_str in sorted(civilizations.keys(), key=int):
                    player_id = int(player_id_str)
                    civ_name = civilizations[player_id_str]['name']
                    tech_count = final_tech.get(str(player_id), 0)
                    science = final_science.get(str(player_id), 0)
                    rows.append(f'<tr><td>{civ_name}</td><td>{int(tech_count)}</td><td>{int(science)}</td></tr>')

                # Sort by tech count
                rows.sort(key=lambda x: int(x.split('<td>')[2].split('</td>')[0]), reverse=True)

                html.append('<table class="data-table">')
                html.append(f'<caption>Technological Status at Turn {final_turn}</caption>')
                html.append('<thead><tr><th>Civilization</th><th>Technologies Known</th><th>Science per Turn</th></tr></thead>')
                html.append('<tbody>')
                html.append('\n'.join(rows))
                html.append('</tbody>')
                html.append('</table>')

        except Exception as e:
            print(f"Warning: Failed to generate technology content: {e}")
            html.append('<p>Technology data not available.</p>')

        return '\n'.join(html)

    def _get_html_header(self) -> str:
        """Get HTML document header with CSS"""
        return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>World Report - Turn ''' + str(self.turn) + '''</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f5f5;
            padding: 20px;
        }

        .title-page {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 80px 40px;
            text-align: center;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }

        .title-page h1 {
            font-size: 3em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }

        .title-page h2 {
            font-size: 2em;
            margin-bottom: 10px;
        }

        .subtitle {
            font-size: 1.2em;
            opacity: 0.9;
        }

        .toc {
            background: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .toc h2 {
            color: #667eea;
            margin-bottom: 20px;
        }

        .toc ol {
            margin-left: 30px;
        }

        .toc li {
            margin: 10px 0;
        }

        .toc a {
            color: #764ba2;
            text-decoration: none;
            font-size: 1.1em;
        }

        .toc a:hover {
            text-decoration: underline;
        }

        .section {
            background: white;
            padding: 40px;
            margin-bottom: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        h2 {
            color: #667eea;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
        }

        h3 {
            color: #764ba2;
            margin-top: 30px;
            margin-bottom: 15px;
        }

        h4 {
            color: #555;
            margin-top: 20px;
            margin-bottom: 10px;
        }

        p {
            margin: 10px 0;
        }

        .overview-info {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 5px;
            margin: 20px 0;
        }

        .statistics {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 5px;
            margin: 20px 0;
        }

        .statistics ul {
            list-style: none;
            margin-left: 0;
        }

        .statistics li {
            padding: 5px 0;
        }

        table.data-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }

        table.data-table caption {
            caption-side: top;
            padding: 10px;
            font-weight: bold;
            font-size: 1.1em;
            color: #667eea;
            text-align: left;
        }

        table.data-table th {
            background: #667eea;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }

        table.data-table td {
            padding: 10px 12px;
            border-bottom: 1px solid #e0e0e0;
        }

        table.data-table tr:hover {
            background: #f5f5f5;
        }

        table.data-table tr:last-child td {
            border-bottom: none;
        }

        .mini-maps-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(700px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }

        .mini-map {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            text-align: center;
        }

        .mini-map img {
            max-width: 100%;
            height: auto;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .mini-map .caption {
            margin-top: 10px;
            font-size: 0.95em;
            color: #555;
        }

        img {
            max-width: 100%;
            height: auto;
        }

        @media print {
            body {
                background: white;
                padding: 0;
            }

            .section {
                page-break-inside: avoid;
                box-shadow: none;
            }

            .title-page {
                page-break-after: always;
            }
        }
    </style>
</head>
<body>
'''

    def _get_html_footer(self) -> str:
        """Get HTML document footer"""
        return '''
</body>
</html>
'''
