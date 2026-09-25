"""Optional paper-style reproductions, separate from the main benchmark figures."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from optionmc.experiments import run_seeded_convergence
from scripts.results_io import write_csv
from optionmc.models import GeometricBrownianMotion
from optionmc.samplers import StandardNormalSampler
from scripts.plots import METHOD_LABELS
BASE_PARAMETERS = dict(S0=100, K=100, r=0.05, sigma=0.2, T=1.0)
PAPER_PATH_COUNTS = (100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000)

def _save(figure, path: Path):
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote {path}")


def _sensitivity_rows(rows, option_type, parameter):
    return sorted(
        [
            row
            for row in rows
            if row["option_type"] == option_type and row["parameter"] == parameter
        ],
        key=lambda row: row["parameter_value"],
    )


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
        "Paper Figure 2 reproduction: simulated call/put sensitivity across repeated runs"
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

