"""
Tests for src/civrealm/metrics.py
"""

import math
import pytest
from civrealm.metrics import compute_brier_score, compute_calibration_error


class TestBrierScore:
    """Tests for compute_brier_score function."""

    def test_perfect_predictions(self):
        """Perfect predictions should score 0.0."""
        predictions = [1.0, 0.0, 1.0, 0.0]
        outcomes = [True, False, True, False]
        assert compute_brier_score(predictions, outcomes) == 0.0

    def test_worst_predictions(self):
        """Maximally wrong predictions should score 1.0."""
        predictions = [0.0, 1.0, 0.0, 1.0]
        outcomes = [True, False, True, False]
        assert compute_brier_score(predictions, outcomes) == 1.0

    def test_uninformed_baseline_50_percent(self):
        """Predicting 50% for everything should score 0.25."""
        predictions = [0.5, 0.5, 0.5, 0.5]
        outcomes = [True, True, False, False]
        assert compute_brier_score(predictions, outcomes) == 0.25

    def test_base_rate_baseline(self):
        """Predicting base rate should score p*(1-p)."""
        base_rate = 0.3
        outcomes = [True, True, True, False, False, False, False, False, False, False]
        predictions = [base_rate] * 10
        expected = base_rate * (1 - base_rate)  # Variance of Bernoulli
        assert abs(compute_brier_score(predictions, outcomes) - expected) < 1e-10

    def test_better_than_baseline(self):
        """Informed predictions should beat the base rate baseline."""
        base_rate = 0.3
        outcomes = [True, True, True, False, False, False, False, False, False, False]

        # Base rate baseline
        baseline_score = compute_brier_score([base_rate] * 10, outcomes)

        # Slightly informed: predict higher for true, lower for false
        informed_predictions = [0.5, 0.5, 0.5, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
        informed_score = compute_brier_score(informed_predictions, outcomes)

        assert informed_score < baseline_score

    def test_empty_predictions(self):
        """Empty predictions should return NaN."""
        assert math.isnan(compute_brier_score([], []))

    def test_single_prediction(self):
        """Single prediction should work."""
        assert compute_brier_score([0.7], [True]) == pytest.approx(0.09)
        assert compute_brier_score([0.7], [False]) == pytest.approx(0.49)

    def test_integer_outcomes(self):
        """Should handle integer outcomes (0/1) as well as booleans."""
        predictions = [0.8, 0.2]
        bool_outcomes = [True, False]
        int_outcomes = [1, 0]

        bool_score = compute_brier_score(predictions, bool_outcomes)
        int_score = compute_brier_score(predictions, int_outcomes)

        assert bool_score == int_score

    def test_length_mismatch_raises_error(self):
        """Mismatched lengths should raise ValueError."""
        with pytest.raises(ValueError):
            compute_brier_score([0.5, 0.5], [True])


class TestCalibrationError:
    """Tests for compute_calibration_error function."""

    def test_perfectly_calibrated(self):
        """Perfectly calibrated predictions should have ECE near 0."""
        # If we predict 0.8 and 80% are true, that's calibrated
        predictions = [0.8] * 10
        outcomes = [True] * 8 + [False] * 2
        ece = compute_calibration_error(predictions, outcomes)
        assert ece == pytest.approx(0.0, abs=0.01)

    def test_overconfident_predictions(self):
        """Overconfident predictions should have positive ECE."""
        # Predict 0.9 but only 50% are true
        predictions = [0.9] * 10
        outcomes = [True] * 5 + [False] * 5
        ece = compute_calibration_error(predictions, outcomes)
        assert ece > 0.3  # Should be about 0.4

    def test_empty_predictions(self):
        """Empty predictions should return NaN."""
        assert math.isnan(compute_calibration_error([], []))
