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


def compute_crps(percentiles: dict, true_value: float) -> float:
    """
    Compute Continuous Ranked Probability Score using quantile-weighted pinball loss.

    CRPS measures the quality of probabilistic forecasts for continuous outcomes.
    This implementation uses a quantile-based approximation via pinball loss.

    Args:
        percentiles: Dictionary with keys "p10", "p25", "p50", "p75", "p90"
                    mapping to predicted quantile values
        true_value: Actual observed value

    Returns:
        CRPS (float). Lower is better:
        - 0.0 = perfect prediction (all quantiles equal true value)
        - Non-negative for all inputs

    Example:
        >>> p = {'p10': 100, 'p25': 100, 'p50': 100, 'p75': 100, 'p90': 100}
        >>> compute_crps(p, 100)
        0.0
        >>> p = {'p10': 80, 'p25': 90, 'p50': 100, 'p75': 110, 'p90': 120}
        >>> compute_crps(p, 100) > 0
        True
    """
    quantile_levels = {"p10": 0.10, "p25": 0.25, "p50": 0.50, "p75": 0.75, "p90": 0.90}
    total_loss = 0.0
    for key, tau in quantile_levels.items():
        q = percentiles[key]
        residual = true_value - q
        if residual >= 0:
            total_loss += tau * residual
        else:
            total_loss += (tau - 1) * residual
    return (2 / len(quantile_levels)) * total_loss


def compute_mae(predicted: float, true_value: float) -> float:
    """
    Compute Mean Absolute Error between predicted and true value.

    Args:
        predicted: Predicted value (typically median/p50)
        true_value: Actual observed value

    Returns:
        MAE (float). Lower is better:
        - 0.0 = perfect prediction
        - Non-negative for all inputs

    Example:
        >>> compute_mae(50, 100)
        50.0
        >>> compute_mae(100, 50)
        50.0
    """
    return abs(predicted - true_value)


def compute_aggregate_crps(all_percentiles: list[dict], all_true_values: list[float]) -> float:
    """
    Compute mean CRPS across multiple forecasts.

    Args:
        all_percentiles: List of percentile dictionaries (may contain None entries)
        all_true_values: List of true values corresponding to each forecast

    Returns:
        Mean CRPS (float), or float('nan') if no valid entries

    Example:
        >>> p1 = {'p10': 80, 'p25': 90, 'p50': 100, 'p75': 110, 'p90': 120}
        >>> p2 = {'p10': 200, 'p25': 300, 'p50': 400, 'p75': 500, 'p90': 600}
        >>> compute_aggregate_crps([p1, p2, None], [100, 100, 100]) > 0
        True
    """
    if len(all_percentiles) != len(all_true_values):
        raise ValueError("all_percentiles and all_true_values must have the same length")

    valid_scores = []
    for percentiles, true_value in zip(all_percentiles, all_true_values):
        if percentiles is not None:
            valid_scores.append(compute_crps(percentiles, true_value))

    if not valid_scores:
        return float('nan')

    return sum(valid_scores) / len(valid_scores)


def compute_aggregate_mae(all_p50: list[float], all_true_values: list[float]) -> float:
    """
    Compute mean MAE across multiple forecasts.

    Args:
        all_p50: List of predicted median values
        all_true_values: List of true values corresponding to each forecast

    Returns:
        Mean MAE (float), or float('nan') if empty

    Example:
        >>> compute_aggregate_mae([100, 400], [100, 100])
        150.0
    """
    if not all_p50:
        return float('nan')
    if len(all_p50) != len(all_true_values):
        raise ValueError("all_p50 and all_true_values must have the same length")

    return sum(compute_mae(pred, true) for pred, true in zip(all_p50, all_true_values)) / len(all_p50)
