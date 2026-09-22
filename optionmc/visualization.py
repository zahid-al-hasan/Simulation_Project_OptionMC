"""Visualization tools for option-pricing results."""

import numpy as np
import matplotlib.pyplot as plt


def plot_convergence(
    n_paths_list,
    prices,
    analytical_price,
    title="MC Convergence",
    ci_lowers=None,
    ci_uppers=None,
):
    """Plot price estimates against path count with an optional confidence band."""
    paths = np.asarray(n_paths_list)
    estimates = np.asarray(prices)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(paths, estimates, marker="o", label="Monte Carlo")
    ax.axhline(analytical_price, color="black", linestyle="--", label="Black-Scholes")
    if ci_lowers is not None and ci_uppers is not None:
        ax.fill_between(paths, ci_lowers, ci_uppers, alpha=0.2, label="95% CI")
    ax.set_xscale("log")
    ax.set_xlabel("Simulation paths")
    ax.set_ylabel("Option price")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_variance_comparison(n_paths_list, results_dict, analytical_price):
    """Compare price convergence for multiple simulation methods."""
    paths = np.asarray(n_paths_list)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for method, results in results_dict.items():
        ax.plot(paths, results["prices"], marker="o", label=method.replace("_", " ").title())
    ax.axhline(analytical_price, color="black", linestyle="--", label="Black-Scholes")
    ax.set_xscale("log")
    ax.set_xlabel("Simulation paths")
    ax.set_ylabel("Option price")
    ax.set_title("Monte Carlo method comparison")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_error_convergence(n_paths_list, errors_dict):
    """Plot relative pricing error against path count on logarithmic axes."""
    paths = np.asarray(n_paths_list)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for method, errors in errors_dict.items():
        positive_errors = np.maximum(np.asarray(errors), np.finfo(float).tiny)
        ax.loglog(paths, positive_errors, marker="o", label=method.replace("_", " ").title())
    ax.set_xlabel("Simulation paths")
    ax.set_ylabel("Absolute relative error")
    ax.set_title("Error convergence")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_payoff_distribution(payoffs, option_type="call"):
    """Plot a histogram of simulated discounted option payoffs."""
    values = np.asarray(payoffs, dtype=float)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins="auto", density=True, alpha=0.75, edgecolor="white")
    ax.set_xlabel("Discounted payoff")
    ax.set_ylabel("Density")
    ax.set_title(f"European {option_type.lower()} payoff distribution")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig, ax


def plot_stock_price_distribution(S_T, title="Terminal Stock Price Distribution"):
    """Plot terminal stock prices with a fitted lognormal density."""
    prices = np.asarray(S_T, dtype=float)
    if prices.ndim != 1 or prices.size < 2 or np.any(prices <= 0):
        raise ValueError("S_T must contain at least two positive prices")
    log_prices = np.log(prices)
    log_mean = np.mean(log_prices)
    log_std = np.std(log_prices, ddof=1)
    x = np.linspace(np.min(prices), np.max(prices), 400)
    density = np.exp(-0.5 * ((np.log(x) - log_mean) / log_std) ** 2) / (
        x * log_std * np.sqrt(2.0 * np.pi)
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(prices, bins="auto", density=True, alpha=0.65, label="Simulation")
    ax.plot(x, density, color="black", linewidth=2, label="Fitted lognormal")
    ax.set_xlabel("Terminal stock price")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_parameter_sensitivity(param_range, mc_prices, analytical_prices, param_name):
    """Compare simulated and analytical prices as one parameter changes."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(param_range, mc_prices, marker="o", label="Monte Carlo")
    ax.plot(param_range, analytical_prices, linestyle="--", label="Black-Scholes")
    ax.set_xlabel(param_name)
    ax.set_ylabel("Option price")
    ax.set_title(f"Option price sensitivity to {param_name}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_efficiency_comparison(methods, errors, runtimes):
    """Plot each method's relative error against runtime."""
    method_names = list(methods)
    error_values = np.asarray(errors, dtype=float)
    runtime_values = np.asarray(runtimes, dtype=float)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(runtime_values, error_values, s=70)
    for method, runtime, error in zip(method_names, runtime_values, error_values):
        ax.annotate(method.replace("_", " ").title(), (runtime, error), xytext=(5, 5), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Runtime (seconds)")
    ax.set_ylabel("Absolute relative error")
    ax.set_title("Accuracy and runtime")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    return fig, ax
