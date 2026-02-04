"""
Metrics for evaluating forecasts and predictions.
"""


def compute_brier_score(predictions: list[float], outcomes: list[bool]) -> float:
    """
    Compute Brier score: mean squared error between predictions and outcomes.

    The Brier score measures the accuracy of probabilistic predictions.
    Formula: BS = (1/N) × Σ(prediction_i - outcome_i)²

    Args:
        predictions: List of predicted probabilities (floats between 0 and 1)
        outcomes: List of actual outcomes (True/False or 1/0)

    Returns:
        Brier score (float). Lower is better:
        - 0.0 = perfect predictions
        - 0.25 = uninformed 50% baseline
        - 1.0 = maximally wrong predictions

    Example:
        >>> compute_brier_score([0.9, 0.1], [True, False])
        0.01
        >>> compute_brier_score([0.5, 0.5], [True, False])
        0.25
    """
    if not predictions:
        return float('nan')
    if len(predictions) != len(outcomes):
        raise ValueError("predictions and outcomes must have the same length")
    return sum((p - int(o))**2 for p, o in zip(predictions, outcomes)) / len(predictions)


def compute_calibration_error(
    predictions: list[float],
    outcomes: list[bool],
    n_bins: int = 10
) -> float:
    """
    Compute Expected Calibration Error (ECE).

    Groups predictions into bins and measures the difference between
    average predicted probability and actual frequency in each bin.

    Args:
        predictions: List of predicted probabilities
        outcomes: List of actual outcomes
        n_bins: Number of bins to use (default: 10)

    Returns:
        ECE (float). Lower is better, 0.0 = perfectly calibrated.
    """
    if not predictions:
        return float('nan')

    # Create bins
    bins = [[] for _ in range(n_bins)]
    for p, o in zip(predictions, outcomes):
        bin_idx = min(int(p * n_bins), n_bins - 1)
        bins[bin_idx].append((p, int(o)))

    # Compute weighted average of |avg_pred - avg_outcome| per bin
    total_error = 0.0
    total_count = len(predictions)

    for bin_data in bins:
        if not bin_data:
            continue
        avg_pred = sum(p for p, _ in bin_data) / len(bin_data)
        avg_outcome = sum(o for _, o in bin_data) / len(bin_data)
        total_error += len(bin_data) * abs(avg_pred - avg_outcome)

    return total_error / total_count
