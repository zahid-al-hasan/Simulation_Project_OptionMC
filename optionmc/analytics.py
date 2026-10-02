"""Convergence analysis and performance metrics."""

from collections.abc import Iterable

import numpy as np

from optionmc.models import BlackScholesAnalytical
from optionmc.pricing import OptionPricing


METHODS = {"standard", "antithetic", "control_variate", "stratified", "quasi"}


def _run_pricing_method(
    pricer: OptionPricing,
    method: str,
    option_type: str,
    qmc_method: str = "sobol",
) -> dict:
    if method not in METHODS:
        raise ValueError(f"method must be one of {sorted(METHODS)}")
    if method == "standard":
        return pricer.standard_mc(option_type)
    if method == "antithetic":
        return pricer.antithetic_mc(option_type)
    if method == "control_variate":
        return pricer.control_variate_mc(option_type)
    if method == "stratified":
        n_strata = min(10, pricer.n_paths)
        while n_strata > 1 and pricer.n_paths % n_strata:
            n_strata -= 1
        return pricer.stratified_mc(option_type, n_strata=n_strata)
    return pricer.quasi_mc(option_type, method=qmc_method)


def _analytical_price(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    option_type: str,
) -> float:
    analytical = BlackScholesAnalytical(S0, K, r, sigma, T)
    if option_type == "call":
        return float(analytical.call_price())
    if option_type == "put":
        return float(analytical.put_price())
    raise ValueError("option_type must be 'call' or 'put'")


def convergence_analysis(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n_paths_list: Iterable[int],
    method: str = "standard",
    option_type: str = "call",
    seed: int | None = 42,
    qmc_method: str = "sobol",
) -> dict:
    """Run one pricing method over increasing path counts."""
    path_counts = np.asarray(list(n_paths_list), dtype=int)
    if path_counts.ndim != 1 or path_counts.size == 0 or np.any(path_counts < 2):
        raise ValueError("n_paths_list must contain integers of at least 2")
    if method == "antithetic" and np.any(path_counts % 2):
        raise ValueError("antithetic convergence requires even path counts")
    analytical_price = _analytical_price(S0, K, r, sigma, T, option_type)
    runs = []
    for n_paths in path_counts:
        pricer = OptionPricing(S0, K, r, sigma, T, int(n_paths), seed=seed)
        runs.append(_run_pricing_method(pricer, method, option_type, qmc_method))
    prices = np.asarray([run["price"] for run in runs])
    return {
        "n_paths": path_counts,
        "prices": prices,
        "std_errors": np.asarray([run["std_error"] for run in runs]),
        "ci_lowers": np.asarray([run["ci_lower"] for run in runs]),
        "ci_uppers": np.asarray([run["ci_upper"] for run in runs]),
        "runtimes": np.asarray([run["runtime"] for run in runs]),
        "variances": np.asarray([run["variance"] for run in runs]),
        "relative_errors": np.asarray(
            [compute_relative_error(price, analytical_price) for price in prices]
        ),
        "analytical_price": analytical_price,
        "method": method,
        "option_type": option_type,
    }


def compute_relative_error(mc_price: float, analytical_price: float) -> float:
    """Compute absolute relative error against an analytical reference."""
    numerator = abs(float(mc_price) - float(analytical_price))
    denominator = abs(float(analytical_price))
    if denominator == 0:
        return 0.0 if numerator == 0 else float("inf")
    return numerator / denominator


def variance_reduction_ratio(var_standard: float, var_reduced: float) -> float:
    """Return standard-estimator variance divided by reduced-estimator variance."""
    if var_standard < 0 or var_reduced < 0:
        raise ValueError("variances cannot be negative")
    if var_reduced == 0:
        return float("inf") if var_standard > 0 else 1.0
    return float(var_standard / var_reduced)


def efficiency_ratio(
    error_standard: float,
    time_standard: float,
    error_reduced: float,
    time_reduced: float,
) -> float:
    """Compare mean-squared-error runtime products; values above one favor reduction."""
    if error_standard < 0 or error_reduced < 0:
        raise ValueError("errors cannot be negative")
    if time_standard <= 0 or time_reduced <= 0:
        raise ValueError("runtimes must be positive")
    standard_cost = error_standard**2 * time_standard
    reduced_cost = error_reduced**2 * time_reduced
    if reduced_cost == 0:
        return float("inf") if standard_cost > 0 else 1.0
    return float(standard_cost / reduced_cost)


def moneyness_analysis(
    S0: float,
    K_range: Iterable[float],
    r: float,
    sigma: float,
    T: float,
    n_paths: int,
    method: str = "standard",
    option_type: str = "call",
    seed: int | None = 42,
    qmc_method: str = "sobol",
) -> dict:
    """Compare simulated and analytical prices across strikes."""
    strikes = np.asarray(list(K_range), dtype=float)
    if strikes.ndim != 1 or strikes.size == 0 or np.any(strikes <= 0):
        raise ValueError("K_range must contain positive strikes")
    mc_prices = []
    analytical_prices = []
    std_errors = []
    for strike in strikes:
        pricer = OptionPricing(S0, strike, r, sigma, T, n_paths, seed=seed)
        result = _run_pricing_method(pricer, method, option_type, qmc_method)
        mc_prices.append(result["price"])
        std_errors.append(result["std_error"])
        analytical_prices.append(
            _analytical_price(S0, strike, r, sigma, T, option_type)
        )
    mc_array = np.asarray(mc_prices)
    analytical_array = np.asarray(analytical_prices)
    return {
        "strikes": strikes,
        "moneyness": strikes / S0,
        "mc_prices": mc_array,
        "analytical_prices": analytical_array,
        "std_errors": np.asarray(std_errors),
        "relative_errors": np.asarray(
            [
                compute_relative_error(mc, exact)
                for mc, exact in zip(mc_array, analytical_array)
            ]
        ),
        "method": method,
        "option_type": option_type,
    }


def parameter_sensitivity(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    param_name: str,
    param_range: Iterable[float],
    n_paths: int,
    method: str = "standard",
    option_type: str = "call",
    seed: int | None = 42,
    qmc_method: str = "sobol",
) -> dict:
    """Vary one option parameter while holding the others constant."""
    if param_name not in {"S0", "K", "r", "sigma", "T"}:
        raise ValueError("param_name must be one of S0, K, r, sigma, or T")
    values = np.asarray(list(param_range), dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("param_range cannot be empty")
    base = {"S0": S0, "K": K, "r": r, "sigma": sigma, "T": T}
    mc_prices = []
    analytical_prices = []
    std_errors = []
    for value in values:
        parameters = base.copy()
        parameters[param_name] = float(value)
        pricer = OptionPricing(**parameters, n_paths=n_paths, seed=seed)
        result = _run_pricing_method(pricer, method, option_type, qmc_method)
        mc_prices.append(result["price"])
        std_errors.append(result["std_error"])
        analytical_prices.append(
            _analytical_price(**parameters, option_type=option_type)
        )
    mc_array = np.asarray(mc_prices)
    analytical_array = np.asarray(analytical_prices)
    return {
        "param_name": param_name,
        "param_range": values,
        "mc_prices": mc_array,
        "analytical_prices": analytical_array,
        "std_errors": np.asarray(std_errors),
        "relative_errors": np.asarray(
            [
                compute_relative_error(mc, exact)
                for mc, exact in zip(mc_array, analytical_array)
            ]
        ),
        "method": method,
        "option_type": option_type,
    }
