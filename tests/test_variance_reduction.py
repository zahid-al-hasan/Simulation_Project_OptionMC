"""Tests for variance-reduction building blocks."""

import numpy as np
import pytest

from optionmc.models import GeometricBrownianMotion
from optionmc.variance_reduction import (
    AntitheticVariates,
    ControlVariates,
    QuasiMonteCarlo,
    StratifiedSampling,
)


class TestAntitheticVariates:
    def test_negation_and_output_count(self):
        draws = np.array([-1.0, 0.0, 2.0])
        original, antithetic = AntitheticVariates.transform(draws)
        np.testing.assert_array_equal(original, draws)
        np.testing.assert_array_equal(antithetic, -draws)
        assert original.size + antithetic.size == 2 * draws.size

    def test_reduces_call_estimator_variance(self):
        rng = np.random.default_rng(42)
        z = rng.standard_normal(100_000)
        model = GeometricBrownianMotion(100, 0.05, 0.2, 1.0)
        discount = np.exp(-0.05)
        positive = discount * np.maximum(model.get_stock_price(1.0, z) - 100, 0)
        negative = discount * np.maximum(model.get_stock_price(1.0, -z) - 100, 0)
        paired_variance = np.var((positive + negative) / 2, ddof=1) / z.size
        standard_variance = np.var(np.concatenate([positive, negative]), ddof=1) / (2 * z.size)
        assert paired_variance < standard_variance


class TestControlVariates:
    @pytest.fixture
    def control(self):
        return ControlVariates(100, 100, 0.05, 0.2, 1.0)

    def test_optimal_beta(self, control):
        rng = np.random.default_rng(2)
        x = rng.normal(100, 10, 20_000)
        y = 3.0 * x + rng.normal(0, 1, x.size)
        assert control.optimal_coefficient(y, x) == pytest.approx(3.0, abs=0.01)

    def test_adjusted_variance_lower(self, control):
        rng = np.random.default_rng(4)
        x = rng.normal(100, 10, 20_000)
        y = 2.0 * x + rng.normal(0, 5, x.size)
        beta = control.optimal_coefficient(y, x)
        adjusted = control.adjusted_estimates(y, x, beta)
        assert np.var(adjusted, ddof=1) < np.var(y, ddof=1)


class TestStratifiedSampling:
    def test_coverage_and_output_count(self):
        n_strata, samples = 10, 5
        values = StratifiedSampling(n_strata, samples, seed=42).stratified_uniform()
        assert values.shape == (n_strata * samples,)
        by_stratum = values.reshape(n_strata, samples)
        for index, row in enumerate(by_stratum):
            assert np.all(row >= index / n_strata)
            assert np.all(row < (index + 1) / n_strata)


class TestQuasiMonteCarlo:
    @pytest.mark.parametrize("method", ["sobol", "halton"])
    def test_generates_positive_terminal_prices(self, method):
        model = GeometricBrownianMotion(100, 0.05, 0.2, 1.0)
        prices = QuasiMonteCarlo(256, method=method, seed=42).generate_paths(model)
        assert prices.shape == (256,)
        assert np.all(np.isfinite(prices))
        assert np.all(prices > 0)

    def test_rejects_unknown_method(self):
        with pytest.raises(ValueError):
            QuasiMonteCarlo(100, method="unknown")
