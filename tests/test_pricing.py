"""Tests for the Monte Carlo pricing engine."""

import pytest

from optionmc.models import BlackScholesAnalytical
from optionmc.pricing import OptionPricing


@pytest.fixture(scope="module")
def exact_call():
    return BlackScholesAnalytical(100, 100, 0.05, 0.2, 1.0).call_price()


@pytest.fixture
def pricer():
    return OptionPricing(100, 100, 0.05, 0.2, 1.0, n_paths=100_000, seed=42)


class TestStandardMC:
    def test_price_and_confidence_interval(self, pricer, exact_call):
        result = pricer.standard_mc()
        assert abs(result["price"] - exact_call) < 3 * result["std_error"]
        assert result["ci_lower"] < exact_call < result["ci_upper"]

    def test_returns_required_statistics(self, pricer):
        result = pricer.standard_mc()
        required = {
            "price",
            "std_error",
            "ci_lower",
            "ci_upper",
            "runtime",
            "variance",
            "sample_variance",
            "n_paths",
            "effective_samples",
            "method",
        }
        assert required <= result.keys()
        assert result["std_error"] > 0
        assert result["runtime"] >= 0

    def test_put_price(self):
        pricer = OptionPricing(100, 100, 0.05, 0.2, 1.0, 100_000, seed=42)
        exact = BlackScholesAnalytical(100, 100, 0.05, 0.2, 1.0).put_price()
        result = pricer.standard_mc("put")
        assert abs(result["price"] - exact) < 3 * result["std_error"]


class TestVarianceReductionPricing:
    def test_antithetic_is_accurate_and_reduces_error(self, pricer, exact_call):
        standard = pricer.standard_mc()
        antithetic = pricer.antithetic_mc()
        assert abs(antithetic["price"] - exact_call) < 3 * antithetic["std_error"]
        assert antithetic["std_error"] < standard["std_error"]

    def test_control_variate_is_accurate_and_reduces_error(self, pricer, exact_call):
        standard = pricer.standard_mc()
        controlled = pricer.control_variate_mc()
        assert abs(controlled["price"] - exact_call) < 3 * controlled["std_error"]
        assert controlled["std_error"] < standard["std_error"]
        assert controlled["beta"] > 0

    def test_stratified_is_accurate_and_reduces_error(self, pricer, exact_call):
        standard = pricer.standard_mc()
        stratified = pricer.stratified_mc(n_strata=10)
        assert abs(stratified["price"] - exact_call) < 3 * stratified["std_error"]
        assert stratified["std_error"] < standard["std_error"]

    @pytest.mark.parametrize("method", ["sobol", "halton"])
    def test_quasi_mc_is_accurate(self, pricer, exact_call, method):
        if method == "sobol":
            # Direct API calls retain support for non-power-of-two budgets,
            # but must expose SciPy's warning rather than hiding lost balance.
            with pytest.warns(UserWarning, match="balance properties"):
                result = pricer.quasi_mc(method=method)
        else:
            result = pricer.quasi_mc(method=method)
        assert abs(result["price"] - exact_call) < 4 * result["std_error"]
        assert result["replications"] >= 2

    def test_quasi_mc_is_reproducible_for_a_fixed_outer_seed(self):
        first = OptionPricing(100, 100, 0.05, 0.2, 1.0, 8_192, seed=99)
        second = OptionPricing(100, 100, 0.05, 0.2, 1.0, 8_192, seed=99)
        assert first.quasi_mc(method="sobol")["price"] == second.quasi_mc(
            method="sobol"
        )["price"]


class TestPricingValidation:
    def test_invalid_option_type(self, pricer):
        with pytest.raises(ValueError):
            pricer.standard_mc("american")

    def test_antithetic_requires_even_paths(self):
        odd = OptionPricing(100, 100, 0.05, 0.2, 1.0, 999, seed=42)
        with pytest.raises(ValueError):
            odd.antithetic_mc()

    def test_stratification_requires_divisibility(self):
        pricer = OptionPricing(100, 100, 0.05, 0.2, 1.0, 1_001, seed=42)
        with pytest.raises(ValueError):
            pricer.stratified_mc(n_strata=10)
