"""Tests for geometric Brownian motion and analytical pricing."""

import numpy as np
import pytest

from optionmc.models import BlackScholesAnalytical, GeometricBrownianMotion
from optionmc.samplers import StandardNormalSampler


class TestGeometricBrownianMotion:
    def test_positive_prices(self):
        model = GeometricBrownianMotion(100, 0.05, 0.2, 1.0)
        prices = model.simulate(StandardNormalSampler(10_000, seed=42))
        assert np.all(prices > 0)

    def test_expected_mean(self):
        model = GeometricBrownianMotion(100, 0.05, 0.2, 1.0)
        prices = model.simulate(StandardNormalSampler(100_000, seed=42))
        expected = 100 * np.exp(0.05)
        assert abs(np.mean(prices) - expected) < 0.25

    def test_log_price_moments(self):
        model = GeometricBrownianMotion(100, 0.05, 0.2, 1.0)
        prices = model.simulate(StandardNormalSampler(100_000, seed=3))
        log_returns = np.log(prices / 100)
        assert np.mean(log_returns) == pytest.approx(0.03, abs=0.002)
        assert np.std(log_returns, ddof=1) == pytest.approx(0.2, abs=0.002)

    def test_zero_time_returns_initial_spot(self):
        model = GeometricBrownianMotion(100, 0.05, 0.2, 1.0)
        prices = model.get_stock_price(0.0, np.array([-2.0, 0.0, 2.0]))
        np.testing.assert_array_equal(prices, np.array([100.0, 100.0, 100.0]))


class TestBlackScholesAnalytical:
    @pytest.fixture
    def model(self):
        return BlackScholesAnalytical(100, 100, 0.05, 0.2, 1.0)

    def test_known_values(self, model):
        assert model.call_price() == pytest.approx(10.4505835722, abs=1e-10)
        assert model.put_price() == pytest.approx(5.5735260223, abs=1e-10)

    def test_put_call_parity(self, model):
        expected = 100 - 100 * np.exp(-0.05)
        assert model.call_price() - model.put_price() == pytest.approx(expected)

    def test_intrinsic_value_bounds(self, model):
        lower = max(100 - 100 * np.exp(-0.05), 0)
        assert lower <= model.call_price() <= 100

    def test_expiry_payoffs(self, model):
        assert model.call_price(t=1.0, spot=120) == 20
        assert model.put_price(t=1.0, spot=80) == 20

    def test_zero_volatility(self):
        model = BlackScholesAnalytical(100, 100, 0.05, 0.0, 1.0)
        assert model.call_price() == pytest.approx(100 - 100 * np.exp(-0.05))
        assert model.put_price() == 0.0
