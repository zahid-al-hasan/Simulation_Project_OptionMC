"""Experiment execution: budgets, independent runs, scenarios, and saved records."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import math
from typing import Any

import numpy as np

from optionmc.models import BlackScholesAnalytical
from optionmc.pricing import OptionPricing, SUPPORTED_METHODS, SUPPORTED_OPTION_TYPES, price_once
from optionmc.analytics import (
    compute_relative_error as _relative_error,
)


def _analytical_price(parameters, option_type):
    if option_type not in SUPPORTED_OPTION_TYPES:
        raise ValueError("option_type must be call or put")
    model = BlackScholesAnalytical(**parameters)
    return float(model.call_price() if option_type == "call" else model.put_price())


def _validate_grid(
    path_counts: Sequence[int],
    repetitions: int,
    methods: Sequence[str],
    option_types: Sequence[str],
) -> None:
    if not isinstance(repetitions, (int, np.integer)) or repetitions < 2:
        raise ValueError("repetitions must be at least 2")
    if not path_counts or any(int(n) != n or n < 2 for n in path_counts):
        raise ValueError("path_counts must contain integers of at least 2")
    unknown_methods = set(methods) - set(SUPPORTED_METHODS)
    if unknown_methods:
        raise ValueError(f"unknown methods: {sorted(unknown_methods)}")
    unknown_options = set(option_types) - set(SUPPORTED_OPTION_TYPES)
    if unknown_options:
        raise ValueError(f"unknown option types: {sorted(unknown_options)}")
    if "antithetic" in methods and any(n % 2 for n in path_counts):
        raise ValueError("all antithetic path counts must be even")
    if "antithetic_control" in methods and any(n < 8 or n % 2 for n in path_counts):
        raise ValueError("combined control requires even budgets of at least eight")


def run_repeated_experiments(
    parameters: Mapping[str, float],
    path_counts: Iterable[int],
    repetitions: int = 30,
    methods: Sequence[str] = SUPPORTED_METHODS,
    option_types: Sequence[str] = SUPPORTED_OPTION_TYPES,
    seed_start: int = 1_000,
    warmup: bool = True,
    scenario_id: str = "baseline",
) -> list[dict[str, Any]]:
    """Run the full method, option-type, path-count, and seed experiment grid."""
    required = {"S0", "K", "r", "sigma", "T"}
    if set(parameters) != required:
        raise ValueError(f"parameters must contain exactly {sorted(required)}")
    counts = tuple(path_counts)
    method_names = tuple(methods)
    option_names = tuple(option_types)
    _validate_grid(counts, repetitions, method_names, option_names)

    if not method_names or not option_names:
        raise ValueError("methods and option_types cannot be empty")
    if warmup:
        for option_type in option_names:
            for method in method_names:
                price_once(OptionPricing(**parameters, n_paths=counts[0], seed=seed_start - 1),
                           method, option_type)
    jobs = [(option, method, n, repeat)
            for option in option_names for method in method_names
            for n in counts for repeat in range(repetitions)]
    # Interleave methods to reduce timing bias from machine drift. Pricing seeds
    # remain independent across repetitions and shared across methods for comparison.
    np.random.default_rng(seed_start).shuffle(jobs)
    rows = []
    for option_type, method, n_paths, repetition in jobs:
        analytical = _analytical_price(parameters, option_type)
        seed = seed_start + repetition
        result = price_once(OptionPricing(**parameters, n_paths=n_paths, seed=seed),
                            method, option_type)
        signed_error = result["price"] - analytical
        absolute_error = abs(signed_error)
        rows.append(
            {
                "scenario_id": scenario_id,
                **parameters,
                "option_type": option_type,
                "method": method,
                "n_paths": n_paths,
                "repetition": repetition,
                "seed": seed,
                "analytical_price": analytical,
                "mc_price": result["price"],
                "signed_error": signed_error,
                "absolute_error": absolute_error,
                "relative_error": (
                    absolute_error / abs(analytical)
                    if analytical != 0
                    else math.inf
                ),
                "squared_error": signed_error**2,
                "reported_std_error": result["std_error"],
                "ci_lower": result["ci_lower"],
                "ci_upper": result["ci_upper"],
                "ci_covers_exact": int(
                    result["ci_lower"]
                    <= analytical
                    <= result["ci_upper"]
                ),
                "estimator_variance": result["variance"],
                "runtime_seconds": result["runtime"],
                "beta": result.get("beta", ""),
                "n_strata": result.get("n_strata", ""),
                "qmc_method": result.get("qmc_method", ""),
                "payoff_evaluations": result["payoff_evaluations"],
                "pilot_evaluations": result["pilot_evaluations"],
                "qmc_replications": result.get("replications", 0),
                "qmc_points_per_replication": str(result.get("replication_path_counts", ())),
            }
        )
    return rows


def run_scenarios(base_parameters, scenarios, path_counts, **kwargs):
    """Run all selected methods for explicit parameter overrides per scenario."""
    rows = []
    for name, overrides in scenarios.items():
        parameters = {**base_parameters, **overrides}
        rows.extend(run_repeated_experiments(
            parameters, path_counts, scenario_id=name, **kwargs
        ))
    return rows


def run_seeded_convergence(
    parameters: Mapping[str, float],
    path_counts: Iterable[int],
    methods: Sequence[str] = ("standard",),
    option_type: str = "call",
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Run a reproducible single-seed convergence study from actual simulations.

    This is the data source for paper-style price/CI and direct method-comparison
    figures. Only the model inputs, path-count grid, and seed are configuration;
    every plotted estimate and confidence interval comes from ``OptionPricing``.
    """
    required = {"S0", "K", "r", "sigma", "T"}
    if set(parameters) != required:
        raise ValueError(f"parameters must contain exactly {sorted(required)}")
    counts = tuple(int(value) for value in path_counts)
    method_names = tuple(methods)
    _validate_grid(counts, 2, method_names, (option_type,))

    analytical = _analytical_price(parameters, option_type)
    rows: list[dict[str, Any]] = []
    for method in method_names:
        for n_paths in counts:
            result = price_once(
                OptionPricing(**parameters, n_paths=n_paths, seed=seed),
                method,
                option_type,
            )
            absolute_error = abs(result["price"] - analytical)
            rows.append(
                {
                    "option_type": option_type,
                    "method": method,
                    "n_paths": n_paths,
                    "seed": seed,
                    "analytical_price": analytical,
                    "mc_price": result["price"],
                    "std_error": result["std_error"],
                    "ci_lower": result["ci_lower"],
                    "ci_upper": result["ci_upper"],
                    "absolute_error": absolute_error,
                    "relative_error": (
                        absolute_error / abs(analytical)
                        if analytical != 0
                        else math.inf
                    ),
                    "runtime_seconds": result["runtime"],
                }
            )
    return rows


def run_repeated_sensitivity(
    base_parameters: Mapping[str, float],
    parameter_grids: Mapping[str, Iterable[float]],
    n_paths: int = 50_000,
    repetitions: int = 30,
    method: str = "control_variate",
    option_types: Sequence[str] = SUPPORTED_OPTION_TYPES,
    seed_start: int = 2_000,
    warmup: bool = True,
) -> list[dict[str, Any]]:
    """Run repeated call/put sensitivity experiments for selected parameters."""
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"method must be one of {SUPPORTED_METHODS}")
    if repetitions < 2:
        raise ValueError("repetitions must be at least 2")
    unknown = set(parameter_grids) - {"S0", "K", "r", "sigma", "T"}
    if unknown:
        raise ValueError(f"unsupported sensitivity parameters: {sorted(unknown)}")
    rows: list[dict[str, Any]] = []
    for option_type in option_types:
        for parameter_name, parameter_values in parameter_grids.items():
            values = tuple(float(value) for value in parameter_values)
            if not values:
                raise ValueError(f"grid for {parameter_name} cannot be empty")
            if warmup:
                warm_parameters = dict(base_parameters)
                warm_parameters[parameter_name] = values[0]
                price_once(
                    OptionPricing(
                        **warm_parameters,
                        n_paths=n_paths,
                        seed=seed_start - 1,
                    ),
                    method,
                    option_type,
                )
            for value in values:
                parameters = dict(base_parameters)
                parameters[parameter_name] = value
                analytical = _analytical_price(parameters, option_type)
                for repetition in range(repetitions):
                    seed = seed_start + repetition
                    result = price_once(
                        OptionPricing(
                            **parameters, n_paths=n_paths, seed=seed
                        ),
                        method,
                        option_type,
                    )
                    signed_error = result["price"] - analytical
                    rows.append(
                        {
                            "option_type": option_type,
                            "method": method,
                            "parameter": parameter_name,
                            "parameter_value": value,
                            "moneyness": parameters["K"] / parameters["S0"],
                            "n_paths": n_paths,
                            "repetition": repetition,
                            "seed": seed,
                            "analytical_price": analytical,
                            "mc_price": result["price"],
                            "signed_error": signed_error,
                            "absolute_error": abs(signed_error),
                            "reported_std_error": result["std_error"],
                            "ci_lower": result["ci_lower"],
                            "ci_upper": result["ci_upper"],
                            "ci_covers_exact": int(
                                result["ci_lower"]
                                <= analytical
                                <= result["ci_upper"]
                            ),
                            "runtime_seconds": result["runtime"],
                        }
                    )
    return rows


def _run_pricing_method(pricer, method, option_type, qmc_method="sobol"):
    return price_once(pricer, qmc_method if method == "quasi" else method, option_type)


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
    analytical_price = _analytical_price(dict(S0=S0, K=K, r=r, sigma=sigma, T=T), option_type)
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
            [_relative_error(price, analytical_price) for price in prices]
        ),
        "analytical_price": analytical_price,
        "method": method,
        "option_type": option_type,
    }


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
            _analytical_price(dict(S0=S0, K=strike, r=r, sigma=sigma, T=T), option_type)
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
                _relative_error(mc, exact)
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
            _analytical_price(parameters, option_type)
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
                _relative_error(mc, exact)
                for mc, exact in zip(mc_array, analytical_array)
            ]
        ),
        "method": method,
        "option_type": option_type,
    }
