"""Tests for models.py — GBM simulation and Black-Scholes analytical pricing."""

import numpy as np
import pytest
from optionmc.models import GeometricBrownianMotion, BlackScholesAnalytical
from optionmc.samplers import StandardNormalSampler


class TestGeometricBrownianMotion:
    """Test GBM terminal price simulation."""

    def setup_method(self):
        self.S0 = 100.0
        self.r = 0.05
        self.sigma = 0.2
        self.T = 1.0
        self.n_paths = 200_000
        self.gbm = GeometricBrownianMotion(self.S0, self.r, self.sigma, self.T)

    def test_positive_prices(self):
        """All simulated prices should be positive."""
        S_T = self.gbm.simulate(StandardNormalSampler(self.n_paths, seed=42))
        assert S_T.shape == (self.n_paths,)
        assert (S_T > 0).all()

    def test_expected_mean(self):
        """Sample mean should be close to S0*exp(rT)."""
        S_T = self.gbm.simulate(StandardNormalSampler(self.n_paths, seed=42))
        expected = self.S0 * np.exp(self.r * self.T)
        assert S_T.mean() == pytest.approx(expected, rel=0.005)

    def test_log_normality(self):
        """Log(prices) should be approximately normally distributed."""
        S_T = self.gbm.simulate(StandardNormalSampler(self.n_paths, seed=42))
        log_S_T = np.log(S_T)
        mu = np.log(self.S0) + (self.r - self.sigma**2 / 2) * self.T
        sd = self.sigma * np.sqrt(self.T)
        assert log_S_T.mean() == pytest.approx(mu, rel=0.005)
        assert log_S_T.std(ddof=1) == pytest.approx(sd, rel=0.005)

    def test_get_stock_price_matches_analytic_drift(self):
        """get_stock_price should reproduce the closed-form GBM increments."""
        t = 0.5
        z = np.array([1.0])
        expected = self.S0 * np.exp((self.r - self.sigma**2 / 2) * t + self.sigma * np.sqrt(t) * z)
        actual = self.gbm.get_stock_price(t, z)
        assert actual == pytest.approx(expected)


class TestBlackScholesAnalytical:
    """Test Black-Scholes closed-form solutions."""

    def setup_method(self):
        self.S0 = 100.0
        self.K = 100.0
        self.r = 0.05
        self.sigma = 0.2
        self.T = 1.0
        self.z = 0.0
        self.bs = BlackScholesAnalytical(self.S0, self.K, self.r, self.sigma, self.T)

    def test_put_call_parity(self):
        """C - P = S0 - K*exp(-rT) should hold."""
        call = self.bs.call_price(0, self.z)
        put = self.bs.put_price(0, self.z)
        assert call - put == pytest.approx(self.S0 - self.K * np.exp(-self.r * self.T), abs=1e-6)

    def test_known_value(self):
        """Compare against a known BS price (d1=0.35, d2=0.15)."""
        expected_call = 10.4506
        expected_put = 5.5735
        assert self.bs.call_price(0, self.z) == pytest.approx(expected_call, abs=2e-3)
        assert self.bs.put_price(0, self.z) == pytest.approx(expected_put, abs=2e-3)

    def test_intrinsic_value_bounds(self):
        """Call price should be between max(S-Kexp(-rT),0) and S."""
        call = self.bs.call_price(0, self.z)
        assert call >= self.S0 - self.K * np.exp(-self.r * self.T)
        assert call <= self.S0