"""Tests for pricing.py — MC pricing engine with all methods."""

import numpy as np
import pytest
from optionmc.pricing import OptionPricing
from optionmc.models import BlackScholesAnalytical


@pytest.fixture
def default_params():
    return dict(S0=100, K=100, r=0.05, sigma=0.2, T=1.0, n_paths=100000, seed=42)


class TestStandardMC:

    def test_price_close_to_analytical(self, default_params):
        # TODO
        pass

    def test_returns_required_keys(self, default_params):
        # TODO
        pass


class TestAntitheticMC:

    def test_price_close_to_analytical(self, default_params):
        # TODO
        pass

    def test_lower_variance_than_standard(self, default_params):
        # TODO
        pass


class TestControlVariateMC:

    def test_price_close_to_analytical(self, default_params):
        # TODO
        pass


class TestStratifiedMC:

    def test_price_close_to_analytical(self, default_params):
        # TODO
        pass


class TestQuasiMC:

    def test_price_close_to_analytical_sobol(self, default_params):
        # TODO
        pass

    def test_price_close_to_analytical_halton(self, default_params):
        # TODO
        pass
