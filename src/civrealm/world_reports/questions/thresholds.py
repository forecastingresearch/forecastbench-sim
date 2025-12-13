"""
Configurable threshold values for question generation.

This module provides two threshold selection strategies:

1. Static thresholds (DEFAULT_THRESHOLDS): Fixed lists of values per signal.
   Use with select_threshold() for backward compatibility.

2. Statistics-based thresholds: Dynamically computed from empirical data.
   Use select_threshold_for_rate() for calibrated ~40% True rates.

Statistics are derived from 103 game simulations.
See signal_statistics.py for the underlying data.
"""

from .schema import ThresholdConfig


# Default threshold configuration (static fallback)
# These are used when resolution_turn is not available
DEFAULT_THRESHOLDS = ThresholdConfig(
    defaults={
        # B1 signals - techs/population/score
        "techs_known": [8, 12, 20, 28, 35, 42, 50],
        "population": [5, 15, 30, 50, 70, 90],
        "score": [200, 500, 1000, 2000, 3500, 5000],

        # B2 signals - territory/treasury/cities
        "territory_size": [30, 70, 130, 200, 280, 350],
        "territory_gain": [10, 40, 100, 180, 250],
        "treasury": [100, 400, 800, 1200, 1800],
        "cities_count": [3, 8, 15, 25, 35],
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


def select_threshold_for_rate(
    signal_name: str,
    resolution_turn: int,
    target_rate: float = 0.4,
    current_value: float | None = None,
) -> int | float:
    """
    Select a threshold targeting a specific True rate using empirical statistics.

    This is the recommended threshold selection method. It uses percentile
    statistics from 103 game simulations to pick thresholds that achieve
    approximately the target True rate.

    For a question "Will signal >= threshold by resolution_turn?":
    - target_rate=0.4 means ~40% of outcomes will be True
    - We select the (1-target_rate) percentile, e.g., 60th percentile for 40% True

    Args:
        signal_name: Name of the signal (e.g., 'techs_known', 'population')
        resolution_turn: Turn at which the question is resolved
        target_rate: Target fraction of True answers (default 0.4 = 40%)
        current_value: Optional current value to ensure threshold is above it

    Returns:
        Threshold value calibrated for the target rate

    Example:
        >>> select_threshold_for_rate("techs_known", resolution_turn=125, target_rate=0.4)
        35  # 60th percentile of tech count at turn 125
    """
    from .signal_statistics import get_threshold_for_rate, get_canonical_signal_name

    canonical_name = get_canonical_signal_name(signal_name)

    try:
        return get_threshold_for_rate(
            canonical_name,
            resolution_turn=resolution_turn,
            target_rate=target_rate,
            current_value=current_value,
        )
    except ValueError:
        # Fall back to static thresholds if signal not in statistics
        thresholds = get_thresholds_for_signal(signal_name)
        # Return median of static thresholds
        sorted_t = sorted(thresholds)
        return sorted_t[len(sorted_t) // 2]


def select_growth_threshold_for_rate(
    signal_name: str,
    snapshot_turn: int,
    resolution_turn: int,
    target_rate: float = 0.4,
) -> int | float:
    """
    Select a growth threshold (change from snapshot to resolution) for target rate.

    For questions like "Will territory increase by X from snapshot to resolution?"

    Args:
        signal_name: Name of the signal
        snapshot_turn: Starting turn (when question is asked)
        resolution_turn: Ending turn (when question is resolved)
        target_rate: Target True rate

    Returns:
        Growth threshold value
    """
    from .signal_statistics import get_growth_threshold_for_rate

    try:
        return get_growth_threshold_for_rate(
            signal_name,
            snapshot_turn=snapshot_turn,
            resolution_turn=resolution_turn,
            target_rate=target_rate,
        )
    except ValueError:
        # Fall back to static thresholds
        thresholds = get_thresholds_for_signal("territory_gain")
        sorted_t = sorted(thresholds)
        return sorted_t[len(sorted_t) // 2]
