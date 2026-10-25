"""Configuration for World Report Generation"""

from dataclasses import dataclass, field
from typing import List
from pathlib import Path


@dataclass
class ReportConfig:
    """Configuration for world report generation

    Attributes:
        recording_dir: Path to logs/recordings/username/ directory
        output_dir: Directory where reports will be saved
        report_turns: Pre-specified turns to generate reports for.
                     Example: [10, 25, 50] generates:
                     - Report 1: covers turns 0-10
                     - Report 2: covers turns 0-25
                     - Report 3: covers turns 0-50
                     Reports are NOT generated for intermediate turns.
        enabled_sections: List of report sections to include
        formats: Output formats (markdown, html, pdf)
        plot_style: Matplotlib style for visualizations
        dpi: Resolution for generated images
    """
    recording_dir: str
    output_dir: str
    report_turns: List[int]

    enabled_sections: List[str] = field(default_factory=lambda: [
        'overview',
        'historical_events',
        'politics',
        'economics',
        'social',
        'technology'
    ])

    formats: List[str] = field(default_factory=lambda: ['html', 'pdf'])
    plot_style: str = 'seaborn'
    dpi: int = 150

    def __post_init__(self):
        """Validate configuration after initialization"""
        self.recording_dir = str(Path(self.recording_dir).resolve())
        self.output_dir = str(Path(self.output_dir).resolve())

        if not self.report_turns:
            raise ValueError("report_turns cannot be empty")

        if not all(isinstance(t, int) and t > 0 for t in self.report_turns):
            raise ValueError("All report_turns must be positive integers")

        valid_sections = {
            'overview', 'historical_events', 'politics',
            'economics', 'demographics', 'social', 'technology'
        }
        invalid = set(self.enabled_sections) - valid_sections
        if invalid:
            raise ValueError(f"Invalid sections: {invalid}")

        valid_formats = {'markdown', 'html', 'pdf'}
        invalid_fmt = set(self.formats) - valid_formats
        if invalid_fmt:
            raise ValueError(f"Invalid formats: {invalid_fmt}")
