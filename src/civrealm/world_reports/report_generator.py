"""Main report generator orchestrator"""

from pathlib import Path
from typing import Dict, List
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

from .config import ReportConfig
from .data_loader import DataLoader
from .utils.visualizations import MapVisualizer
from .extractors import MetricsCollector, write_world_data, read_world_data
from .renderers.html import HTMLRenderer


class ReportGenerator:
    """Main orchestrator for generating world reports"""

    def __init__(self, config: ReportConfig):
        """Initialize report generator

        Args:
            config: ReportConfig instance
        """
        self.config = config
        self.data_loader = DataLoader(config.recording_dir)
        self.visualizer = MapVisualizer(dpi=config.dpi, style=config.plot_style, data_loader=self.data_loader)

    def generate_reports(self):
        """Generate reports for all configured turns

        This is the main entry point. It generates a report for each
        turn specified in config.report_turns.
        """
        print(f"World Report Generator")
        print(f"=====================")
        print(f"Recording directory: {self.config.recording_dir}")
        print(f"Output directory: {self.config.output_dir}")
        print()

        # Check data availability
        summary = self.data_loader.get_turn_summary()
        print(f"Data Summary:")
        print(f"  Total turns available: {summary['total_turns']}")
        print(f"  Turn range: {summary['turn_range'][0]} to {summary['turn_range'][1]}")
        print(f"  Total state files: {summary['total_files']}")
        print()

        # Generate reports
        print(f"Generating reports for turns: {self.config.report_turns}")
        print()

        for turn in self.config.report_turns:
            print(f"Generating report for turn {turn}...")
            try:
                self.generate_report_for_turn(turn)
                print(f"  ✓ Report for turn {turn} completed")
            except Exception as e:
                print(f"  ✗ Error generating report for turn {turn}: {e}")
                import traceback
                traceback.print_exc()

        print()
        print(f"Report generation complete!")
        print(f"Reports saved to: {self.config.output_dir}")

    def generate_report_for_turn(self, turn: int):
        """Generate a single report covering turns 0 to turn

        Uses two-stage pipeline: data extraction → JSON → HTML rendering

        Args:
            turn: Target turn number (report covers 0 to turn inclusive)
        """
        # Stage 1: Extract data (returns dict with actual turn in metadata)
        print(f"  Stage 1: Extracting data...")
        data = self.generate_data(turn, output_file=None)

        # Get actual turn from the data (may be less than requested if data unavailable)
        actual_turn = data['metadata']['turn']

        # Warn if we got less data than requested
        if actual_turn < turn:
            print(f"  Warning: Requested turn {turn}, but only data up to turn {actual_turn} available")

        # Use actual turn for filenames
        json_file = f"{self.config.output_dir}/turn_{actual_turn:03d}_data.json"

        # Write JSON with correct filename
        from .extractors import write_world_data
        write_world_data(data, json_file)

        # Stage 2: Render HTML from JSON
        print(f"  Stage 2: Rendering HTML...")
        html_file = self.render_from_json(json_file)

        print(f"    JSON: {json_file}")
        print(f"    HTML: {html_file}")

    def generate_data(self, turn: int, output_file: str = None) -> Dict:
        """Stage 1: Extract data and save to JSON

        Args:
            turn: Target turn number (report covers 0 to turn inclusive)
            output_file: Optional path to save JSON file

        Returns:
            World report data dictionary
        """
        # Load all states from turn 0 to target turn
        states = self.data_loader.get_states_range(0, turn)

        if not states:
            raise ValueError(f"No data available for turns 0 to {turn}")

        # Collect metrics
        collector = MetricsCollector()
        data = collector.collect_all(
            states=states,
            config=self.config,
            data_loader=self.data_loader
        )

        # Save to JSON if output file specified
        if output_file:
            write_world_data(data, output_file)

        return data

    def render_from_json(self, json_file: str, output_file: str = None) -> str:
        """Stage 2: Render HTML from JSON

        Args:
            json_file: Path to world report JSON file
            output_file: Optional path for output HTML file

        Returns:
            Path to generated HTML file
        """
        # Read JSON data
        data = read_world_data(json_file)

        # Create renderer with visualizer for territory maps
        turn = data['metadata']['turn']
        renderer = HTMLRenderer(
            output_dir=self.config.output_dir,
            turn=turn,
            recording_dir=self.config.recording_dir,
            data_loader=self.data_loader,
            visualizer=self.visualizer
        )

        # Render HTML
        html_file = renderer.render_from_json(data, output_file)

        return html_file

    def get_section_list(self) -> List[str]:
        """Get list of available section names

        Returns:
            List of section names
        """
        return [
            'overview',
            'historical_events',
            'economics',
            'demographics',
            'technology'
        ]

    def validate_config(self) -> bool:
        """Validate configuration and check data availability

        Returns:
            True if configuration is valid and data is available
        """
        # Check recording directory exists
        if not Path(self.config.recording_dir).exists():
            print(f"Error: Recording directory not found: {self.config.recording_dir}")
            return False

        # Check if any data is available
        summary = self.data_loader.get_turn_summary()
        if summary['total_turns'] == 0:
            print(f"Error: No state files found in {self.config.recording_dir}")
            return False

        # Check if requested turns are available
        available_turns = summary['turns_available']
        max_available = max(available_turns)

        for turn in self.config.report_turns:
            if turn > max_available:
                print(f"Warning: Turn {turn} requested but only {max_available} turns available")

        return True
