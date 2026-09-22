"""Generate report figures and tables from repeated OptionMC experiments."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from optionmc.experiments import SUPPORTED_METHODS
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate report-ready figures from repeated experiment CSVs."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("artifacts/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/report"))
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
    write_final_tables(experiment_rows, args.output_dir)


if __name__ == "__main__":
    main()
