"""World Reports Package

Generates comprehensive PDF/HTML reports from CivRealm game state recordings.
"""

from .report_generator import ReportGenerator
from .config import ReportConfig

__all__ = ['ReportGenerator', 'ReportConfig']
