"""Main report generator orchestrator"""

from pathlib import Path
from typing import Dict, List
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

from .config import ReportConfig
from .data_loader import DataLoader
from .utils.visualizations import MapVisualizer
from .sections.base_section import BaseSection, SectionData
from .sections.overview import OverviewSection
from .sections.historical_events import HistoricalEventsSection
from .sections.economics import EconomicsSection
from .sections.demographics import DemographicsSection
from .sections.technology import TechnologySection
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

        # Map section names to section classes
        self.available_sections = {
            'overview': OverviewSection,
            'historical_events': HistoricalEventsSection,
            'economics': EconomicsSection,
            'demographics': DemographicsSection,
            'technology': TechnologySection,
            # Add more sections as implemented
            # 'politics': PoliticsSection,
        }

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

        Args:
            turn: Target turn number (report covers 0 to turn inclusive)
        """
        # Load all states from turn 0 to target turn
        states = self.data_loader.get_states_range(0, turn)

        if not states:
            raise ValueError(f"No data available for turns 0 to {turn}")

        # Generate sections
        sections = []
        for section_name in self.config.enabled_sections:
            if section_name not in self.available_sections:
                print(f"  Warning: Section '{section_name}' not implemented yet, skipping")
                continue

            section_class = self.available_sections[section_name]
            section_instance = section_class()

            print(f"  Generating section: {section_name}...")
            section_data = section_instance.generate(
                states=states,
                config=self.config,
                data_loader=self.data_loader,
                visualizer=self.visualizer
            )
            sections.append(section_data)

        # Render outputs
        for fmt in self.config.formats:
            if fmt == 'html':
                print(f"  Rendering HTML...")
                renderer = HTMLRenderer(self.config.output_dir, turn)
                output_file = renderer.render(sections)
                print(f"    Saved: {output_file}")
            elif fmt == 'pdf':
                print(f"  PDF rendering not yet implemented, skipping")
                # TODO: Implement PDF renderer
            elif fmt == 'markdown':
                print(f"  Markdown rendering not yet implemented, skipping")
                # TODO: Implement Markdown renderer

    def get_section_list(self) -> List[str]:
        """Get list of available section names

        Returns:
            List of section names
        """
        return list(self.available_sections.keys())

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
