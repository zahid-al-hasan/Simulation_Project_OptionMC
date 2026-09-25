"""Convergence analysis and performance metrics."""

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any
import math

import numpy as np
from scipy.stats import t as student_t

from optionmc.pricing import SUPPORTED_METHODS, SUPPORTED_OPTION_TYPES


def aggregate_experiments(rows):
    """Summarize each scenario separately, never pooling different contracts."""
    if not rows:
        raise ValueError("rows cannot be empty")
    scenarios = defaultdict(list)
    for row in rows:
        scenarios[row.get("scenario_id", "baseline")].append(row)
    summary = []
    for scenario_id, group in scenarios.items():
        parameters = {key: group[0][key] for key in ("S0", "K", "r", "sigma", "T") if key in group[0]}
        if any(any(row.get(key) != value for key, value in parameters.items()) for row in group):
            raise ValueError("a scenario_id must identify one set of option parameters")
        for row in _aggregate_one_scenario(group):
            row.update(scenario_id=scenario_id, **parameters)
            summary.append(row)
    return summary


def time_to_target(summary, targets=(0.1, 0.05, 0.01)):
    """Use the smallest TESTED budget meeting each repeated-run RMSE target.

    This is a grid estimate, not a first-passage stopping rule. Runtime is the
    median cost of one estimator at that budget, not the whole benchmark sweep.
    Unreached targets have no fabricated runtime or extrapolation.
    """
    targets = tuple(float(target) for target in targets)
    if not targets or any(not np.isfinite(t) or t <= 0 for t in targets):
        raise ValueError("targets must be finite and positive")
    groups = defaultdict(list)
    for row in summary:
        groups[(row.get("scenario_id", "baseline"), row["option_type"], row["method"])].append(row)
    results = []
    for (scenario, option, method), rows in groups.items():
        rows = sorted(rows, key=lambda row: row["n_paths"])
        for target in targets:
            chosen = next((row for row in rows if row["rmse"] <= target), None)
            result = dict(scenario_id=scenario, option_type=option, method=method,
                          target_rmse=target, reached=chosen is not None)
            for key in ("n_paths", "rmse", "median_runtime_seconds", "runtime_q1_seconds", "runtime_q3_seconds"):
                result[key] = chosen[key] if chosen else None
            results.append(result)
    return results


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


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def _aggregate_one_scenario(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate independent runs and calculate empirical comparison metrics."""
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
        if repetitions < 2:
            raise ValueError("at least two independent runs per configuration are required")
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
                "price_q025": float(np.quantile(prices, 0.025)),
                "price_q975": float(np.quantile(prices, 0.975)),
                "mean_price_ci_lower": mean_price - student_t.ppf(0.975, repetitions - 1) * mean_price_se,
                "mean_price_ci_upper": mean_price + student_t.ppf(0.975, repetitions - 1) * mean_price_se,
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
        if repetitions < 2:
            raise ValueError("at least two independent runs per configuration are required")
        margin = student_t.ppf(0.975, repetitions - 1) * empirical_std / np.sqrt(repetitions)
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
                "price_q025": float(np.quantile(prices, 0.025)),
                "price_q975": float(np.quantile(prices, 0.975)),
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
