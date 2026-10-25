#!/usr/bin/env python3
"""Generate world reports from existing game recordings

This script generates world reports from recorded game data without
re-running the game. Useful for testing report generation and styling.

Usage:
    python test_world_report.py
"""

import sys
from pathlib import Path

# Add src to path for world report imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from civrealm.world_reports import ReportGenerator, ReportConfig


def main():
    """Generate world reports from existing recordings"""

    # Configuration
    report_config = ReportConfig(
        # Input: where game recordings are stored
        recording_dir='logs/recordings/myagent2/',

        # Output: where to save the report
        output_dir='reports/latest_game/',

        # Generate reports at specific turns
        # Set to the turns you want to analyze
        report_turns=[380],  # Test with the full game data

        # Enable all implemented sections
        enabled_sections=['overview', 'historical_events', 'economics', 'demographics', 'technology'],

        # Output formats
        formats=['html'],

        # Visualization settings
        plot_style='seaborn',
        dpi=150
    )

    print("="*60)
    print("WORLD REPORT GENERATOR (Test Mode)")
    print("="*60)
    print()
    print(f"Input: {report_config.recording_dir}")
    print(f"Output: {report_config.output_dir}")
    print(f"Report turns: {report_config.report_turns}")
    print()

    # Create generator and generate reports
    print("Initializing World Report Generator...")
    generator = ReportGenerator(report_config)

    print("Validating configuration...")
    if not generator.validate_config():
        print("Configuration validation failed!")
        return 1

    print("Configuration validated successfully!")
    print()
    print("Generating reports...")
    generator.generate_reports()

    print()
    print("="*60)
    print("WORLD REPORTS GENERATED!")
    print("="*60)
    print(f"\nReports saved to: {report_config.output_dir}")
    print("\nOpen the HTML files in your browser to view the reports.")
    print()

    return 0


if __name__ == '__main__':
    sys.exit(main())
