"""Generate report figures and tables from repeated OptionMC experiments."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

from optionmc.experiments import (
    SUPPORTED_METHODS,
    run_seeded_convergence,
    write_csv,
)
from optionmc.models import GeometricBrownianMotion
from optionmc.samplers import StandardNormalSampler


METHOD_LABELS = {
    "standard": "Standard MC",
    "antithetic": "Antithetic",
    "control_variate": "Control variate",
    "stratified": "Stratified",
    "sobol": "Sobol QMC",
    "halton": "Halton QMC",
}

BASE_PARAMETERS = {"S0": 100, "K": 100, "r": 0.05, "sigma": 0.2, "T": 1.0}
PAPER_PATH_COUNTS = (100, 200, 500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate report-ready figures from repeated experiment CSVs."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("artifacts/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("report_figures"))
    return parser.parse_args()


def _read_numeric_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    numeric_fields = {
        "n_paths",
        "repetitions",
        "analytical_price",
        "mean_price",
        "bias",
        "mean_absolute_error",
        "mean_relative_error",
        "rmse",
        "empirical_variance",
        "empirical_std",
        "mean_price_ci_lower",
        "mean_price_ci_upper",
        "mean_reported_std_error",
        "mean_estimator_variance",
        "coverage_rate",
        "median_runtime_seconds",
        "runtime_q1_seconds",
        "runtime_q3_seconds",
        "empirical_vrr",
        "empirical_efficiency_ratio",
        "rmse_convergence_slope",
        "parameter_value",
        "moneyness",
    }
    for row in rows:
        for field in numeric_fields & row.keys():
            row[field] = float(row[field])
    return rows


def _method_rows(rows, option_type, method):
    return sorted(
        [
            row
            for row in rows
            if row["option_type"] == option_type and row["method"] == method
        ],
        key=lambda row: row["n_paths"],
    )


def _save(figure, path: Path):
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote {path}")


def plot_price_convergence(rows, option_type, output_dir):
    figure, axis = plt.subplots(figsize=(9, 5.5))
    for method in SUPPORTED_METHODS:
        selected = _method_rows(rows, option_type, method)
        x = np.asarray([row["n_paths"] for row in selected])
        y = np.asarray([row["mean_price"] for row in selected])
        lower = np.asarray([row["mean_price_ci_lower"] for row in selected])
        upper = np.asarray([row["mean_price_ci_upper"] for row in selected])
        axis.plot(x, y, marker="o", label=METHOD_LABELS[method])
        axis.fill_between(x, lower, upper, alpha=0.10)
    analytical = _method_rows(rows, option_type, "standard")[0]["analytical_price"]
    axis.axhline(analytical, color="black", linestyle="--", label="Black-Scholes")
    axis.set_xscale("log")
    axis.set_xlabel("Total terminal-price budget")
    axis.set_ylabel("Option price")
    axis.set_title(f"European {option_type} price convergence across 30 runs")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(ncol=2)
    _save(figure, output_dir / f"{option_type}_price_convergence.png")


def plot_rmse_convergence(rows, option_type, output_dir):
    figure, axis = plt.subplots(figsize=(9, 5.5))
    for method in SUPPORTED_METHODS:
        selected = _method_rows(rows, option_type, method)
        axis.loglog(
            [row["n_paths"] for row in selected],
            [row["rmse"] for row in selected],
            marker="o",
            label=METHOD_LABELS[method],
        )
    axis.set_xlabel("Total terminal-price budget")
    axis.set_ylabel("Root-mean-square pricing error")
    axis.set_title(f"European {option_type} RMSE convergence")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(ncol=2)
    _save(figure, output_dir / f"{option_type}_rmse_convergence.png")


def plot_variance_reduction(rows, option_type, output_dir):
    figure, axis = plt.subplots(figsize=(9, 5.5))
    for method in SUPPORTED_METHODS[1:]:
        selected = _method_rows(rows, option_type, method)
        axis.loglog(
            [row["n_paths"] for row in selected],
            [row["empirical_vrr"] for row in selected],
            marker="o",
            label=METHOD_LABELS[method],
        )
    axis.axhline(1.0, color="black", linestyle="--", label="No reduction")
    axis.set_xlabel("Total terminal-price budget")
    axis.set_ylabel("Empirical variance-reduction ratio")
    axis.set_title(f"European {option_type} variance reduction across 30 runs")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(ncol=2)
    _save(figure, output_dir / f"{option_type}_variance_reduction.png")


def _largest_budget_rows(rows, option_type):
    selected = [row for row in rows if row["option_type"] == option_type]
    largest = max(row["n_paths"] for row in selected)
    return [row for row in selected if row["n_paths"] == largest]


def plot_accuracy_runtime(rows, output_dir):
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for axis, option_type in zip(axes, ("call", "put")):
        selected = _largest_budget_rows(rows, option_type)
        for row in selected:
            axis.scatter(row["median_runtime_seconds"], row["rmse"], s=65)
            axis.annotate(
                METHOD_LABELS[row["method"]],
                (row["median_runtime_seconds"], row["rmse"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
            )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("Median pricing runtime (seconds)")
        axis.set_ylabel("RMSE")
        axis.set_title(f"European {option_type}")
        axis.grid(True, which="both", alpha=0.25)
    figure.suptitle("Accuracy and runtime at 100,000 paths")
    _save(figure, output_dir / "accuracy_runtime_tradeoff.png")


def plot_coverage(rows, output_dir):
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    for axis, option_type in zip(axes, ("call", "put")):
        selected = _largest_budget_rows(rows, option_type)
        methods = [METHOD_LABELS[row["method"]] for row in selected]
        coverage = [100 * row["coverage_rate"] for row in selected]
        axis.bar(methods, coverage)
        axis.axhline(95, color="black", linestyle="--", label="Nominal 95%")
        axis.set_ylim(0, 105)
        axis.set_title(f"European {option_type}")
        axis.tick_params(axis="x", rotation=35)
        axis.grid(True, axis="y", alpha=0.25)
    axes[0].set_ylabel("Coverage across repetitions (%)")
    axes[1].legend()
    figure.suptitle("Reported confidence-interval coverage at 100,000 paths")
    _save(figure, output_dir / "confidence_interval_coverage.png")


def _sensitivity_rows(rows, option_type, parameter):
    return sorted(
        [
            row
            for row in rows
            if row["option_type"] == option_type and row["parameter"] == parameter
        ],
        key=lambda row: row["parameter_value"],
    )


def plot_sensitivity(rows, parameter, xlabel, filename, output_dir, x_field="parameter_value"):
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=False)
    for axis, option_type in zip(axes, ("call", "put")):
        selected = _sensitivity_rows(rows, option_type, parameter)
        x = np.asarray([row[x_field] for row in selected])
        mean = np.asarray([row["mean_price"] for row in selected])
        lower = np.asarray([row["mean_price_ci_lower"] for row in selected])
        upper = np.asarray([row["mean_price_ci_upper"] for row in selected])
        analytical = np.asarray([row["analytical_price"] for row in selected])
        axis.plot(x, mean, marker="o", label="Control-variate MC mean")
        axis.fill_between(x, lower, upper, alpha=0.2, label="95% CI for mean")
        axis.plot(x, analytical, linestyle="--", color="black", label="Black-Scholes")
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Option price")
        axis.set_title(f"European {option_type}")
        axis.grid(True, alpha=0.25)
    axes[0].legend()
    figure.suptitle(f"Option-price sensitivity to {xlabel.lower()} across 30 runs")
    _save(figure, output_dir / filename)


def plot_distribution_validation(output_dir):
    S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.2, 1.0
    model = GeometricBrownianMotion(S0, r, sigma, T)
    terminal = model.simulate(StandardNormalSampler(100_000, seed=42))
    discount = np.exp(-r * T)
    calls = discount * np.maximum(terminal - K, 0.0)
    puts = discount * np.maximum(K - terminal, 0.0)
    log_returns = np.log(terminal / S0)

    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    x = np.linspace(np.min(terminal), np.max(terminal), 500)
    log_mean = np.log(S0) + (r - 0.5 * sigma**2) * T
    log_std = sigma * np.sqrt(T)
    density = np.exp(-0.5 * ((np.log(x) - log_mean) / log_std) ** 2) / (
        x * log_std * np.sqrt(2 * np.pi)
    )
    axes[0, 0].hist(terminal, bins=70, density=True, alpha=0.65, label="Simulation")
    axes[0, 0].plot(x, density, color="black", linewidth=2, label="GBM lognormal")
    axes[0, 0].set_title("Terminal stock prices")
    axes[0, 0].set_xlabel("Terminal price")
    axes[0, 0].set_ylabel("Density")
    axes[0, 0].legend()

    axes[0, 1].hist(calls, bins=70, color="tab:blue", alpha=0.75)
    axes[0, 1].set_title("Discounted call payoffs")
    axes[0, 1].set_xlabel("Discounted payoff")
    axes[0, 1].set_ylabel("Count")

    axes[1, 0].hist(puts, bins=70, color="tab:orange", alpha=0.75)
    axes[1, 0].set_title("Discounted put payoffs")
    axes[1, 0].set_xlabel("Discounted payoff")
    axes[1, 0].set_ylabel("Count")

    normal_x = np.linspace(np.min(log_returns), np.max(log_returns), 500)
    normal_density = np.exp(
        -0.5 * ((normal_x - (r - 0.5 * sigma**2) * T) / log_std) ** 2
    ) / (log_std * np.sqrt(2 * np.pi))
    axes[1, 1].hist(log_returns, bins=70, density=True, alpha=0.65, label="Simulation")
    axes[1, 1].plot(normal_x, normal_density, color="black", linewidth=2, label="Normal theory")
    axes[1, 1].set_title("Log returns")
    axes[1, 1].set_xlabel("log(S_T / S_0)")
    axes[1, 1].set_ylabel("Density")
    axes[1, 1].legend()

    for axis in axes.ravel():
        axis.grid(True, alpha=0.2)
    figure.suptitle("Distribution validation with 100,000 standard MC paths")
    _save(figure, output_dir / "distribution_validation.png")


def generate_paper_convergence_figures(output_dir):
    """Reproduce the paper's Figures 1 and 4 with fresh model simulations."""
    rows = run_seeded_convergence(
        BASE_PARAMETERS,
        PAPER_PATH_COUNTS,
        methods=("standard", "antithetic"),
        option_type="call",
        seed=42,
    )
    write_csv(rows, output_dir / "paper_figures_1_and_4_simulation_data.csv")
    by_method = {
        method: sorted(
            [row for row in rows if row["method"] == method],
            key=lambda row: row["n_paths"],
        )
        for method in ("standard", "antithetic")
    }

    standard = by_method["standard"]
    x = np.asarray([row["n_paths"] for row in standard])
    estimates = np.asarray([row["mc_price"] for row in standard])
    lower = np.asarray([row["ci_lower"] for row in standard])
    upper = np.asarray([row["ci_upper"] for row in standard])
    analytical = standard[0]["analytical_price"]

    figure, axis = plt.subplots(figsize=(10, 6))
    axis.plot(x, estimates, marker="o", linewidth=2, label="Monte Carlo estimate")
    axis.fill_between(x, lower, upper, alpha=0.25, label="Per-run 95% confidence interval")
    axis.axhline(
        analytical,
        color="tab:red",
        linestyle="--",
        linewidth=2,
        label=f"Black-Scholes (${analytical:.2f})",
    )
    axis.set_xscale("log")
    axis.set_xlabel("Number of terminal-price simulations")
    axis.set_ylabel("European call price")
    axis.yaxis.set_major_formatter(StrMethodFormatter("${x:,.2f}"))
    axis.set_title("Paper Figure 1 reproduction: simulated Monte Carlo convergence")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    _save(figure, output_dir / "paper_figure1_mc_confidence_convergence.png")

    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for method, color in (("standard", "tab:blue"), ("antithetic", "tab:orange")):
        selected = by_method[method]
        method_x = [row["n_paths"] for row in selected]
        label = METHOD_LABELS[method]
        axes[0].plot(
            method_x,
            [row["mc_price"] for row in selected],
            marker="o",
            color=color,
            label=label,
        )
        axes[1].loglog(
            method_x,
            [100 * row["relative_error"] for row in selected],
            marker="o",
            color=color,
            label=label,
        )
    axes[0].axhline(
        analytical,
        color="black",
        linestyle="--",
        label=f"Black-Scholes (${analytical:.2f})",
    )
    axes[0].set_xscale("log")
    axes[0].set_ylabel("European call price")
    axes[0].yaxis.set_major_formatter(StrMethodFormatter("${x:,.2f}"))
    axes[0].set_title("Price convergence")
    axes[1].set_ylabel("Absolute relative error (%)")
    axes[1].set_title("Error comparison")
    for axis in axes:
        axis.set_xlabel("Number of terminal-price simulations")
        axis.grid(True, which="both", alpha=0.25)
        axis.legend()
    figure.suptitle("Paper Figure 4 reproduction: standard vs antithetic Monte Carlo")
    _save(figure, output_dir / "paper_figure4_standard_vs_antithetic.png")


def plot_paper_sensitivity_composite(rows, output_dir):
    """Create the paper's four-panel sensitivity layout from repeated simulations."""
    specs = (
        ("sigma", "parameter_value", "Volatility", "Volatility (sigma)"),
        ("T", "parameter_value", "Time to maturity", "Years"),
        ("K", "parameter_value", "Strike price", "Strike price (K)"),
        ("K", "moneyness", "Moneyness", "Moneyness (K/S0)"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    for axis, (parameter, x_field, title, xlabel) in zip(axes.ravel(), specs):
        for option_type, color in (("call", "tab:blue"), ("put", "tab:orange")):
            selected = _sensitivity_rows(rows, option_type, parameter)
            x = [row[x_field] for row in selected]
            axis.plot(
                x,
                [row["mean_price"] for row in selected],
                marker="o",
                color=color,
                label=f"{option_type.title()} simulated mean",
            )
            axis.plot(
                x,
                [row["analytical_price"] for row in selected],
                linestyle="--",
                color=color,
                label=f"{option_type.title()} Black-Scholes",
            )
        if x_field == "moneyness":
            axis.axvline(1.0, color="gray", linestyle=":", label="At the money")
        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Option price")
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle(
        "Paper Figure 2 reproduction: simulated call/put sensitivity across 30 runs"
    )
    _save(figure, output_dir / "paper_figure2_sensitivity_composite.png")


def plot_paper_distribution_composite(output_dir):
    """Create the paper's distribution panels from one reproducible GBM simulation."""
    S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.2, 1.0
    terminal = GeometricBrownianMotion(S0, r, sigma, T).simulate(
        StandardNormalSampler(100_000, seed=42)
    )
    calls = np.maximum(terminal - K, 0.0)
    puts = np.maximum(K - terminal, 0.0)
    x = np.linspace(max(np.min(terminal), np.finfo(float).eps), np.max(terminal), 500)
    log_mean = np.log(S0) + (r - 0.5 * sigma**2) * T
    log_std = sigma * np.sqrt(T)
    theoretical_density = np.exp(
        -0.5 * ((np.log(x) - log_mean) / log_std) ** 2
    ) / (x * log_std * np.sqrt(2 * np.pi))

    histogram_rows = []
    for series_name, values in (("terminal_stock", terminal), ("call_payoff", calls), ("put_payoff", puts)):
        counts, edges = np.histogram(values, bins=70)
        for index, count in enumerate(counts):
            histogram_rows.append(
                {
                    "series": series_name,
                    "seed": 42,
                    "simulations": 100_000,
                    "bin_left": edges[index],
                    "bin_right": edges[index + 1],
                    "count": int(count),
                }
            )
    write_csv(histogram_rows, output_dir / "paper_figure3_histogram_data.csv")

    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].hist(terminal, bins=70, color="tab:blue", alpha=0.72)
    axes[0, 0].axvline(K, color="tab:red", linestyle="--", label=f"Strike K = {K:.0f}")
    axes[0, 0].set_title("Terminal stock-price distribution")
    axes[0, 0].set_xlabel("Terminal stock price")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].legend()

    axes[0, 1].hist(calls, bins=70, color="tab:green", alpha=0.72)
    axes[0, 1].set_title("Call-payoff distribution")
    axes[0, 1].set_xlabel("Call payoff at maturity")
    axes[0, 1].set_ylabel("Frequency")

    axes[1, 0].hist(puts, bins=70, color="tab:orange", alpha=0.72)
    axes[1, 0].set_title("Put-payoff distribution")
    axes[1, 0].set_xlabel("Put payoff at maturity")
    axes[1, 0].set_ylabel("Frequency")

    axes[1, 1].hist(
        terminal,
        bins=70,
        density=True,
        color="tab:blue",
        alpha=0.55,
        label="Empirical simulation",
    )
    axes[1, 1].plot(x, theoretical_density, color="tab:red", linewidth=2, label="Theoretical lognormal")
    axes[1, 1].set_title("Theoretical vs empirical terminal prices")
    axes[1, 1].set_xlabel("Terminal stock price")
    axes[1, 1].set_ylabel("Density")
    axes[1, 1].legend()

    for axis in axes.ravel():
        axis.grid(True, alpha=0.2)
    figure.suptitle("Paper Figure 3 reproduction: 100,000 simulated GBM terminal prices")
    _save(figure, output_dir / "paper_figure3_distribution_composite.png")


def plot_four_technique_comparison(rows, output_dir):
    """Compare the four proposed variance-reduction techniques in one chart."""
    techniques = ("antithetic", "control_variate", "stratified", "sobol")
    labels = ("Antithetic", "Control variate", "Stratified", "QMC (Sobol)")
    selected = {
        (option_type, row["method"]): row
        for option_type in ("call", "put")
        for row in _largest_budget_rows(rows, option_type)
        if row["method"] in techniques
    }
    comparison_rows = [selected[(option_type, method)] for option_type in ("call", "put") for method in techniques]
    write_csv(comparison_rows, output_dir / "four_technique_comparison_data.csv")

    x = np.arange(len(techniques))
    width = 0.36
    figure, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    metrics = (
        ("rmse", "RMSE (lower is better)"),
        ("empirical_vrr", "Empirical VRR (higher is better)"),
        ("median_runtime_seconds", "Median runtime, seconds (lower is better)"),
    )
    for axis, (metric, ylabel) in zip(axes, metrics):
        call_values = [selected[("call", method)][metric] for method in techniques]
        put_values = [selected[("put", method)][metric] for method in techniques]
        axis.bar(x - width / 2, call_values, width, label="Call")
        axis.bar(x + width / 2, put_values, width, label="Put")
        axis.set_yscale("log")
        axis.set_xticks(x, labels, rotation=25, ha="right")
        axis.set_ylabel(ylabel)
        axis.grid(True, axis="y", which="both", alpha=0.25)
        axis.legend()
        if metric == "empirical_vrr":
            axis.axhline(1.0, color="black", linestyle="--", linewidth=1, label="No reduction")
    largest_budget = int(selected[("call", techniques[0])]["n_paths"])
    repetitions = int(selected[("call", techniques[0])]["repetitions"])
    figure.suptitle(
        f"Four variance-reduction techniques at {largest_budget:,} paths ({repetitions} runs)"
    )
    _save(figure, output_dir / "four_technique_comparison.png")


def write_figure_provenance(output_dir):
    lines = [
        "# Figure data provenance",
        "",
        "No plotted estimate is copied from the paper or manually chosen to match it.",
        "",
        "| Output | Numerical source |",
        "| --- | --- |",
        "| Paper Figure 1 reproduction | Fresh `OptionPricing.standard_mc` runs for seed 42 at each configured path count; per-run confidence intervals come from simulated payoff variance. |",
        "| Paper Figure 2 reproduction | `sensitivity_summary.csv`, aggregating 30 control-variate simulations at each grid value; dashed curves are independently evaluated Black-Scholes formulas. |",
        "| Paper Figure 3 reproduction | 100,000 GBM terminal prices generated from seeded standard-normal draws; theoretical overlay is the GBM lognormal density. |",
        "| Paper Figure 4 reproduction | Fresh standard and antithetic simulations for seed 42 at equal terminal-price budgets; relative errors use the computed Black-Scholes value. |",
        "| Four-technique comparison | `experiment_summary.csv`, aggregating 30 outer seeds/scramble sets at 100,000 paths. QMC is represented by scrambled Sobol. |",
        "",
        "Fixed values such as the parameter set, path-count grid, sample size, and random seed are experimental configuration, not fitted graph values.",
    ]
    path = output_dir / "FIGURE_PROVENANCE.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {path}")


def write_final_tables(rows, output_dir):
    selected = _largest_budget_rows(rows, "call") + _largest_budget_rows(rows, "put")
    fields = [
        "option_type",
        "method",
        "n_paths",
        "analytical_price",
        "mean_price",
        "bias",
        "rmse",
        "empirical_std",
        "empirical_vrr",
        "median_runtime_seconds",
        "coverage_rate",
        "rmse_convergence_slope",
    ]
    csv_path = output_dir / "final_method_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in selected)

    md_path = output_dir / "final_method_comparison.md"
    lines = [
        "# Repeated experiment summary at 100,000 paths",
        "",
        "Each entry summarizes 30 independent seeds or randomized scrambles.",
        "",
        "| Option | Method | Mean price | RMSE | Empirical VRR | Median runtime (s) | 95% CI coverage | RMSE slope |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in selected:
        lines.append(
            f"| {row['option_type'].title()} | {METHOD_LABELS[row['method']]} "
            f"| {row['mean_price']:.6f} | {row['rmse']:.6f} "
            f"| {row['empirical_vrr']:.2f} | {row['median_runtime_seconds']:.6f} "
            f"| {100 * row['coverage_rate']:.1f}% "
            f"| {row['rmse_convergence_slope']:.3f} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    experiment_rows = _read_numeric_csv(args.data_dir / "experiment_summary.csv")
    sensitivity_rows = _read_numeric_csv(args.data_dir / "sensitivity_summary.csv")

    for option_type in ("call", "put"):
        plot_price_convergence(experiment_rows, option_type, args.output_dir)
        plot_rmse_convergence(experiment_rows, option_type, args.output_dir)
        plot_variance_reduction(experiment_rows, option_type, args.output_dir)
    plot_accuracy_runtime(experiment_rows, args.output_dir)
    plot_coverage(experiment_rows, args.output_dir)
    plot_sensitivity(
        sensitivity_rows,
        "sigma",
        "Volatility (sigma)",
        "volatility_sensitivity.png",
        args.output_dir,
    )
    plot_sensitivity(
        sensitivity_rows,
        "T",
        "Time to maturity (years)",
        "maturity_sensitivity.png",
        args.output_dir,
    )
    plot_sensitivity(
        sensitivity_rows,
        "K",
        "Strike price (K)",
        "strike_sensitivity.png",
        args.output_dir,
    )
    plot_sensitivity(
        sensitivity_rows,
        "K",
        "Moneyness (K/S0)",
        "moneyness_sensitivity.png",
        args.output_dir,
        x_field="moneyness",
    )
    plot_distribution_validation(args.output_dir)
    generate_paper_convergence_figures(args.output_dir)
    plot_paper_sensitivity_composite(sensitivity_rows, args.output_dir)
    plot_paper_distribution_composite(args.output_dir)
    plot_four_technique_comparison(experiment_rows, args.output_dir)
    write_figure_provenance(args.output_dir)
    write_final_tables(experiment_rows, args.output_dir)


if __name__ == "__main__":
    main()
