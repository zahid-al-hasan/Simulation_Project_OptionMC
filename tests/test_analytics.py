"""Tests for convergence analysis, sensitivity, and metrics."""

import numpy as np
import pytest

from optionmc.experiments import convergence_analysis, moneyness_analysis, parameter_sensitivity

from optionmc.analytics import (
    compute_relative_error,
    efficiency_ratio,
    variance_reduction_ratio,
)


class TestMetrics:
    def test_relative_error(self):
        assert compute_relative_error(10.0, 10.0) == 0.0
        assert compute_relative_error(9.0, 10.0) == pytest.approx(0.1)
        assert compute_relative_error(1.0, 0.0) == float("inf")

    def test_variance_reduction_ratio(self):
        assert variance_reduction_ratio(4.0, 1.0) == 4.0
        assert variance_reduction_ratio(1.0, 0.0) == float("inf")

    def test_efficiency_ratio(self):
        ratio = efficiency_ratio(0.2, 1.0, 0.1, 2.0)
        assert ratio == pytest.approx(2.0)


class TestAnalyses:
    def test_convergence_result_shape(self):
        paths = [1_000, 5_000]
        result = convergence_analysis(
            100, 100, 0.05, 0.2, 1.0, paths, method="control_variate", seed=42
        )
        np.testing.assert_array_equal(result["n_paths"], paths)
        for key in [
            "prices",
            "std_errors",
            "ci_lowers",
            "ci_uppers",
            "runtimes",
            "variances",
            "relative_errors",
        ]:
            assert result[key].shape == (2,)
        assert np.all(result["std_errors"] > 0)

    def test_moneyness_analysis(self):
        strikes = [90, 100, 110]
        result = moneyness_analysis(
            100, strikes, 0.05, 0.2, 1.0, 10_000, method="control_variate"
        )
        np.testing.assert_array_equal(result["strikes"], strikes)
        assert np.all(np.diff(result["analytical_prices"]) < 0)
        assert np.max(result["relative_errors"]) < 0.02

    def test_parameter_sensitivity(self):
        volatility = [0.1, 0.2, 0.3]
        result = parameter_sensitivity(
            100,
            100,
            0.05,
            0.2,
            1.0,
            "sigma",
            volatility,
            10_000,
            method="control_variate",
        )
        np.testing.assert_array_equal(result["param_range"], volatility)
        assert np.all(np.diff(result["analytical_prices"]) > 0)

    def test_unknown_method_is_rejected(self):
        with pytest.raises(ValueError):
            convergence_analysis(
                100, 100, 0.05, 0.2, 1.0, [100], method="unknown"
            )
