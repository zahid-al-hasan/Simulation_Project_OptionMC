"""Tests for models.py — GBM simulation and Black-Scholes analytical pricing."""

import numpy as np
import pytest
from optionmc.models import GeometricBrownianMotion, BlackScholesAnalytical


class TestGeometricBrownianMotion:
    """Test GBM terminal price simulation."""

    def test_positive_prices(self):
        """All simulated prices should be positive."""
        # TODO
        pass

    def test_expected_mean(self):
        """Sample mean should be close to S0*exp(rT)."""
        # TODO
        pass

    def test_log_normality(self):
        """Log(prices) should be approximately normally distributed."""
        # TODO
        pass


class TestBlackScholesAnalytical:
    """Test Black-Scholes closed-form solutions."""

    def test_put_call_parity(self):
        """C - P = S0 - K*exp(-rT) should hold."""
        # TODO
        pass

    def test_known_value(self):
        """Compare against a known BS price."""
        # TODO
        pass

    def test_intrinsic_value_bounds(self):
        """Call price should be between max(S-K,0) and S."""
        # TODO
        pass
