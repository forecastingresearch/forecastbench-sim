"""
Configurable threshold values for question generation.

These are placeholder values that will be calibrated after collecting
a corpus with base rates from simulation runs.
"""

from .schema import ThresholdConfig


# Default threshold configuration
# Calibrated from game data using calibrate_thresholds.py
DEFAULT_THRESHOLDS = ThresholdConfig(
    defaults={
        # B1 signals - calibrated from percentiles at turns 50-200
        "techs_known": [6, 8, 13, 24, 28, 38, 40, 44, 50],
        "population": [5, 9, 19, 23, 27, 43],
        "score": [100, 250, 500, 1000, 2000, 5000],  # Not yet calibrated

        # B2 signals - calibrated from percentiles at turns 50-200
        "territory_size": [26, 38, 63, 67, 90, 142, 149],
        "territory_gain": [10, 25, 50, 75, 100],
        "treasury": [100, 222, 268, 741, 804, 956, 1253],
        "cities_count": [1, 3, 5, 8, 15],
    }
)


def get_default_thresholds() -> ThresholdConfig:
    """Get the default threshold configuration."""
    return DEFAULT_THRESHOLDS


def get_thresholds_for_signal(signal_name: str, config: ThresholdConfig | None = None) -> list[int | float]:
    """
    Get threshold options for a specific signal.

    Args:
        signal_name: Name of the signal (e.g., 'techs_known', 'treasury')
        config: Optional custom threshold config; uses defaults if None

    Returns:
        List of threshold values for the signal

    Raises:
        ValueError: If signal_name has no configured thresholds
    """
    if config is None:
        config = DEFAULT_THRESHOLDS

    if signal_name not in config.defaults:
        raise ValueError(f"No thresholds configured for signal: {signal_name}")

    return config.defaults[signal_name]


def select_threshold(
    signal_name: str,
    current_value: float | int,
    config: ThresholdConfig | None = None,
    strategy: str = "nearest_above"
) -> int | float:
    """
    Select an appropriate threshold based on current value and strategy.

    Args:
        signal_name: Name of the signal
        current_value: Current value of the signal at snapshot turn
        config: Optional custom threshold config
        strategy: Selection strategy:
            - "nearest_above": Select lowest threshold above current value
            - "nearest_below": Select highest threshold below current value
            - "median": Select median threshold
            - "random": Random selection from available thresholds

    Returns:
        Selected threshold value
    """
    thresholds = get_thresholds_for_signal(signal_name, config)

    if strategy == "nearest_above":
        # Find lowest threshold above current value
        above = [t for t in thresholds if t > current_value]
        if above:
            return min(above)
        return max(thresholds)  # Fallback to highest if none above

    elif strategy == "nearest_below":
        # Find highest threshold below current value
        below = [t for t in thresholds if t < current_value]
        if below:
            return max(below)
        return min(thresholds)  # Fallback to lowest if none below

    elif strategy == "median":
        sorted_thresholds = sorted(thresholds)
        mid = len(sorted_thresholds) // 2
        return sorted_thresholds[mid]

    elif strategy == "random":
        import random
        return random.choice(thresholds)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def create_threshold_config(overrides: dict[str, list[int | float]] | None = None) -> ThresholdConfig:
    """
    Create a threshold config with optional overrides.

    Args:
        overrides: Dictionary of signal_name -> threshold list to override defaults

    Returns:
        New ThresholdConfig with defaults and overrides merged
    """
    merged = dict(DEFAULT_THRESHOLDS.defaults)
    if overrides:
        merged.update(overrides)
    return ThresholdConfig(defaults=merged)
