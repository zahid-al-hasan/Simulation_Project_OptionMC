"""Repeated experiments and aggregation for reproducible method comparisons."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
import csv
import math
from pathlib import Path
from typing import Any

import numpy as np

from optionmc.models import BlackScholesAnalytical
from optionmc.pricing import OptionPricing


SUPPORTED_METHODS = (
    "standard",
    "antithetic",
    "control_variate",
    "stratified",
    "sobol",
    "halton",
)
SUPPORTED_OPTION_TYPES = ("call", "put")


def _analytical_price(parameters: Mapping[str, float], option_type: str) -> float:
    model = BlackScholesAnalytical(**parameters)
    return float(model.call_price() if option_type == "call" else model.put_price())


def _strata_for(n_paths: int, preferred: int = 10) -> int:
    """Return the largest useful stratum count up to ``preferred`` that divides N."""
    n_strata = min(preferred, n_paths)
    while n_strata > 1 and n_paths % n_strata:
        n_strata -= 1
    return n_strata


def price_once(pricer: OptionPricing, method: str, option_type: str) -> dict:
    """Dispatch one pricing run while preserving a common total path budget."""
    if method == "standard":
        return pricer.standard_mc(option_type)
    if method == "antithetic":
        return pricer.antithetic_mc(option_type)
    if method == "control_variate":
        return pricer.control_variate_mc(option_type)
    if method == "stratified":
        return pricer.stratified_mc(
            option_type, n_strata=_strata_for(pricer.n_paths)
        )
    if method in {"sobol", "halton"}:
        return pricer.quasi_mc(option_type, method=method)
    raise ValueError(f"method must be one of {SUPPORTED_METHODS}")


def _validate_grid(
    path_counts: Sequence[int],
    repetitions: int,
    methods: Sequence[str],
    option_types: Sequence[str],
) -> None:
    if repetitions < 2:
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


def run_repeated_experiments(
    parameters: Mapping[str, float],
    path_counts: Iterable[int],
    repetitions: int = 30,
    methods: Sequence[str] = SUPPORTED_METHODS,
    option_types: Sequence[str] = SUPPORTED_OPTION_TYPES,
    seed_start: int = 1_000,
    warmup: bool = True,
) -> list[dict[str, Any]]:
    """Run the full method, option-type, path-count, and seed experiment grid."""
    required = {"S0", "K", "r", "sigma", "T"}
    if set(parameters) != required:
        raise ValueError(f"parameters must contain exactly {sorted(required)}")
    counts = tuple(int(value) for value in path_counts)
    method_names = tuple(methods)
    option_names = tuple(option_types)
    _validate_grid(counts, repetitions, method_names, option_names)

    rows: list[dict[str, Any]] = []
    for option_type in option_names:
        analytical = _analytical_price(parameters, option_type)
        for method in method_names:
            if warmup:
                warm_pricer = OptionPricing(
                    **parameters, n_paths=counts[0], seed=seed_start - 1
                )
                price_once(warm_pricer, method, option_type)
            for n_paths in counts:
                for repetition in range(repetitions):
                    seed = seed_start + repetition
                    pricer = OptionPricing(
                        **parameters, n_paths=n_paths, seed=seed
                    )
                    result = price_once(pricer, method, option_type)
                    signed_error = result["price"] - analytical
                    absolute_error = abs(signed_error)
                    rows.append(
                        {
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
                        }
                    )
    return rows


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def aggregate_experiments(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate independent runs and calculate empirical comparison metrics."""
    if not rows:
        raise ValueError("rows cannot be empty")
    grouped: dict[tuple[str, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["option_type"]), str(row["method"]), int(row["n_paths"]))].append(row)

    summary: list[dict[str, Any]] = []
    for (option_type, method, n_paths), group in grouped.items():
        prices = np.asarray([float(row["mc_price"]) for row in group])
        runtimes = np.asarray([float(row["runtime_seconds"]) for row in group])
        analytical = float(group[0]["analytical_price"])
        empirical_variance = float(np.var(prices, ddof=1))
        repetitions = len(group)
        empirical_std = float(np.sqrt(empirical_variance))
        mean_price = float(np.mean(prices))
        mean_price_se = empirical_std / np.sqrt(repetitions)
        summary.append(
            {
                "option_type": option_type,
                "method": method,
                "n_paths": n_paths,
                "repetitions": repetitions,
                "analytical_price": analytical,
                "mean_price": mean_price,
                "bias": mean_price - analytical,
                "mean_absolute_error": _mean(group, "absolute_error"),
                "mean_relative_error": _mean(group, "relative_error"),
                "rmse": float(np.sqrt(_mean(group, "squared_error"))),
                "empirical_variance": empirical_variance,
                "empirical_std": empirical_std,
                "mean_price_ci_lower": mean_price - 1.96 * mean_price_se,
                "mean_price_ci_upper": mean_price + 1.96 * mean_price_se,
                "mean_reported_std_error": _mean(group, "reported_std_error"),
                "mean_estimator_variance": _mean(group, "estimator_variance"),
                "coverage_rate": _mean(group, "ci_covers_exact"),
                "median_runtime_seconds": float(np.median(runtimes)),
                "runtime_q1_seconds": float(np.percentile(runtimes, 25)),
                "runtime_q3_seconds": float(np.percentile(runtimes, 75)),
            }
        )

    lookup = {
        (row["option_type"], row["method"], row["n_paths"]): row
        for row in summary
    }
    for row in summary:
        standard = lookup.get((row["option_type"], "standard", row["n_paths"]))
        if standard is None:
            row["empirical_vrr"] = math.nan
            row["empirical_efficiency_ratio"] = math.nan
            continue
        denominator = row["empirical_variance"]
        row["empirical_vrr"] = (
            standard["empirical_variance"] / denominator
            if denominator > 0
            else math.inf
        )
        reduced_cost = row["rmse"] ** 2 * row["median_runtime_seconds"]
        standard_cost = (
            standard["rmse"] ** 2 * standard["median_runtime_seconds"]
        )
        row["empirical_efficiency_ratio"] = (
            standard_cost / reduced_cost if reduced_cost > 0 else math.inf
        )

    for option_type in {row["option_type"] for row in summary}:
        for method in {row["method"] for row in summary}:
            method_rows = sorted(
                (
                    row
                    for row in summary
                    if row["option_type"] == option_type and row["method"] == method
                ),
                key=lambda row: row["n_paths"],
            )
            valid = [row for row in method_rows if row["rmse"] > 0]
            slope = (
                float(
                    np.polyfit(
                        np.log([row["n_paths"] for row in valid]),
                        np.log([row["rmse"] for row in valid]),
                        1,
                    )[0]
                )
                if len(valid) >= 2
                else math.nan
            )
            for row in method_rows:
                row["rmse_convergence_slope"] = slope

    return sorted(
        summary,
        key=lambda row: (
            SUPPORTED_OPTION_TYPES.index(row["option_type"]),
            SUPPORTED_METHODS.index(row["method"]),
            row["n_paths"],
        ),
    )


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


def aggregate_sensitivity(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate repeated sensitivity runs into plot-ready statistics."""
    if not rows:
        raise ValueError("rows cannot be empty")
    grouped: dict[tuple[str, str, str, float], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["option_type"]),
            str(row["method"]),
            str(row["parameter"]),
            float(row["parameter_value"]),
        )
        grouped[key].append(row)
    summary = []
    for (option_type, method, parameter, value), group in grouped.items():
        prices = np.asarray([float(row["mc_price"]) for row in group])
        empirical_std = float(np.std(prices, ddof=1))
        mean_price = float(np.mean(prices))
        repetitions = len(group)
        margin = 1.96 * empirical_std / np.sqrt(repetitions)
        summary.append(
            {
                "option_type": option_type,
                "method": method,
                "parameter": parameter,
                "parameter_value": value,
                "moneyness": float(group[0]["moneyness"]),
                "n_paths": int(group[0]["n_paths"]),
                "repetitions": repetitions,
                "analytical_price": float(group[0]["analytical_price"]),
                "mean_price": mean_price,
                "empirical_std": empirical_std,
                "mean_price_ci_lower": mean_price - margin,
                "mean_price_ci_upper": mean_price + margin,
                "mean_absolute_error": _mean(group, "absolute_error"),
                "coverage_rate": _mean(group, "ci_covers_exact"),
                "median_runtime_seconds": float(
                    np.median([float(row["runtime_seconds"]) for row in group])
                ),
            }
        )
    return sorted(
        summary,
        key=lambda row: (
            row["parameter"],
            SUPPORTED_OPTION_TYPES.index(row["option_type"]),
            row["parameter_value"],
        ),
    )


def write_csv(rows: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    """Write dictionaries to CSV with a stable union of fields."""
    if not rows:
        raise ValueError("rows cannot be empty")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output


def read_csv(path: str | Path) -> list[dict[str, str]]:
    """Read a UTF-8 CSV file into dictionaries."""
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))
