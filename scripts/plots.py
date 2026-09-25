"""Visualization tools for option-pricing results."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from optionmc.models import BlackScholesAnalytical

METHOD_LABELS = {
    "standard": "Plain MC", "antithetic": "Antithetic",
    "control_variate": "Control variate", "stratified": "Stratified",
    "sobol": "Scrambled Sobol", "halton": "Scrambled Halton",
    "antithetic_control": "Antithetic + control",
}
METHOD_COLORS = dict(zip(METHOD_LABELS, plt.get_cmap("tab10").colors))


def plot_price_surface(rows, option_type, maturity, budget, method="antithetic_control"):
    """Exact price surface with saved repeated-run mean prices as points.

    The smooth grid is evaluated analytically, not interpolated from simulations.
    Spot, interest rate, maturity and evaluation budget are held fixed.
    """
    selected = [r for r in rows if r["option_type"] == option_type
                and r["T"] == maturity and r["n_paths"] == budget and r["method"] == method]
    spot, rate = selected[0]["S0"], selected[0]["r"]
    moneyness = [r["K"] / r["S0"] for r in selected]
    volatility = [r["sigma"] for r in selected]
    x, y = np.meshgrid(np.linspace(min(moneyness), max(moneyness), 40),
                       np.linspace(min(volatility), max(volatility), 40))
    z = np.empty_like(x)
    for index in np.ndindex(x.shape):
        model = BlackScholesAnalytical(spot, spot * x[index], rate, y[index], maturity)
        z[index] = model.call_price() if option_type == "call" else model.put_price()

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
    ax.plot_surface(x, y, z, color="#8ab6ce", alpha=0.6, linewidth=0,
                    antialiased=True, zorder=1)
    points = ax.scatter(moneyness, volatility, [r["mean_price"] for r in selected],
                        color=METHOD_COLORS[method], edgecolors="black", s=65,
                        depthshade=False, zorder=3, label=f"{METHOD_LABELS[method]}: mean estimate")
    ax.set_xlabel("Moneyness (K / S0)", labelpad=12)
    ax.set_ylabel("Volatility (sigma)", labelpad=12)
    # A figure-level label stays visible when Matplotlib crops a 3D PNG.
    fig.text(0.95, 0.5, "Option price", rotation=90, va="center", fontsize=11)
    ax.view_init(elev=25, azim=-55)
    ax.set_box_aspect((1.2, 1, 0.85))
    ax.set_title(f"European {option_type}: exact surface and simulated estimates\n"
                 f"S0={spot:g}, r={rate:.1%}, T={maturity:g} years, N={budget:,}", pad=22)
    ax.legend(handles=[Patch(facecolor="#8ab6ce", alpha=0.6, label="Black-Scholes surface"), points],
              loc="upper left", bbox_to_anchor=(0, 1.01), fontsize=9)
    repetitions = sorted({r["repetitions"] for r in selected})
    repeat_label = str(repetitions[0]) if len(repetitions) == 1 else "varying numbers of"
    fig.text(0.5, 0.04, f"Each point averages {repeat_label} independent runs at a tested scenario.\n"
             "Close agreement can hide small errors; use RMSE plots to compare accuracy.",
             ha="center", fontsize=10)
    fig.subplots_adjust(left=0.02, right=0.90, bottom=0.12, top=0.88)
    return fig, ax


def _method_groups(rows):
    for method in METHOD_LABELS:
        selected = sorted((row for row in rows if row["method"] == method),
                          key=lambda row: row["n_paths"])
        if selected:
            yield method, selected


def plot_benchmark(rows, metric="rmse"):
    """Plot one scenario and option type; input is repeated-run summary data.

    Price bands show empirical 2.5--97.5% quantiles of individual runs.
    Runtime bands show the interquartile range, not a confidence interval.
    """
    fields = {
        "price": ("n_paths", "mean_price", "Option price"),
        "rmse": ("n_paths", "rmse", "RMSE (price units; lower is better)"),
        "runtime": ("n_paths", "median_runtime_seconds", "Median runtime (seconds)"),
        "accuracy_runtime": ("median_runtime_seconds", "rmse", "RMSE (price units; lower is better)"),
        "variance": ("n_paths", "empirical_vrr", "Variance reduction versus plain MC"),
        "coverage": ("n_paths", "coverage_rate", "Fraction of 95% intervals covering exact price"),
    }
    xfield, yfield, ylabel = fields[metric]
    fig, ax = plt.subplots(figsize=(10, 6))
    for method, selected in _method_groups(rows):
        x = [row[xfield] for row in selected]
        ax.plot(x, [row[yfield] for row in selected], marker="o",
                color=METHOD_COLORS[method], label=METHOD_LABELS[method])
        band = {"price": ("price_q025", "price_q975"),
                "runtime": ("runtime_q1_seconds", "runtime_q3_seconds")}.get(metric)
        if band:
            ax.fill_between(x, [row[band[0]] for row in selected],
                            [row[band[1]] for row in selected],
                            color=METHOD_COLORS[method], alpha=0.12)
    if metric == "price":
        ax.axhline(rows[0]["analytical_price"], color="black", ls="--", label="Black-Scholes")
    if metric == "rmse":
        standard = sorted((r for r in rows if r["method"] == "standard"), key=lambda r: r["n_paths"])
        if standard:
            x = np.array([r["n_paths"] for r in standard])
            ax.plot(x, standard[0]["rmse"] * np.sqrt(x[0] / x), "k--", label="MC reference: N^(-1/2)")
    if metric == "coverage":
        ax.axhline(0.95, color="black", ls="--", label="Nominal 95%")
        ax.set_ylim(0, 1.03)
    if metric == "variance":
        ax.axhline(1, color="black", ls="--")
    ax.set_xscale("log")
    if metric not in {"price", "coverage"} and any(np.isfinite(r[yfield]) and r[yfield] > 0 for r in rows):
        ax.set_yscale("log")
    ax.set_xlabel("Median runtime per estimate (seconds)" if metric == "accuracy_runtime"
                  else "Total payoff evaluations (including pilot)")
    ax.set_ylabel(ylabel)
    subtitle = {"price": " — shaded: middle 95% of runs",
                "runtime": " — shaded: runtime interquartile range"}.get(metric, "")
    ax.set_title(f"{rows[0].get('scenario_id', 'baseline')} / {rows[0]['option_type']}{subtitle}")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    return fig, ax


def plot_target_times(rows):
    """One option/scenario/target; unreached methods have labels, not zero bars."""
    fig, ax = plt.subplots(figsize=(10, 6))
    rows = sorted(rows, key=lambda r: list(METHOD_LABELS).index(r["method"]))
    for index, row in enumerate(rows):
        if row["reached"]:
            value = row["median_runtime_seconds"] * 1000
            ax.barh(index, value, color=METHOD_COLORS[row["method"]])
            ax.errorbar(value, index, xerr=[[value - 1000 * row["runtime_q1_seconds"]],
                                          [1000 * row["runtime_q3_seconds"] - value]],
                        color="black", capsize=3)
            ax.annotate(f" N={row['n_paths']:,}",
                        (1000 * row["runtime_q3_seconds"], index),
                        xytext=(5, 0), textcoords="offset points", va="center", fontsize=8)
        else:
            ax.text(0.02, index, "Not reached within tested budgets",
                    transform=ax.get_yaxis_transform(), va="center", fontsize=9)
    ax.set_yticks(range(len(rows)), [METHOD_LABELS[r["method"]] for r in rows])
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.margins(x=0.25)
    if not any(row["reached"] for row in rows):
        ax.set_xticks([])
    ax.set_xlabel("Median runtime per estimate (ms); error bars show runtime IQR")
    ax.set_title(f"{rows[0]['option_type']}: time to RMSE ≤ {rows[0]['target_rmse']:g}\nSmallest tested budget meeting target; shorter is better")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    return fig, ax


def plot_error_boxplots(rows):
    """Compare signed errors at the largest supplied budget for one scenario/type."""
    budget = max(row["n_paths"] for row in rows)
    groups = list(_method_groups([r for r in rows if r["n_paths"] == budget]))
    fig, ax = plt.subplots(figsize=(10, 6))
    boxes = ax.boxplot([[r["signed_error"] for r in group] for _, group in groups],
                       patch_artist=True)
    for box, (method, _) in zip(boxes["boxes"], groups):
        box.set_facecolor(METHOD_COLORS[method])
    ax.set_xticks(range(1, len(groups) + 1), [METHOD_LABELS[m] for m, _ in groups], rotation=20, ha="right")
    ax.axhline(0, color="black", ls="--")
    ax.set_ylabel("Estimated price − Black-Scholes price")
    ax.set_title(f"{rows[0]['option_type']}: variability across runs at N={budget:,}")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    return fig, ax


def plot_scenario_heatmap(rows, option_type, maturity, budget):
    """One panel per method: RMSE improvement over plain MC on a K/sigma grid."""
    selected = [r for r in rows if r["option_type"] == option_type
                and r["T"] == maturity and r["n_paths"] == budget]
    methods = [m for m, _ in _method_groups(selected) if m != "standard"]
    strikes = sorted({r["K"] / r["S0"] for r in selected})
    sigmas = sorted({r["sigma"] for r in selected})
    baseline = {(r["K"] / r["S0"], r["sigma"]): r["rmse"]
                for r in selected if r["method"] == "standard"}
    columns = min(3, len(methods))
    nrows = (len(methods) + columns - 1) // columns
    fig, axes = plt.subplots(nrows, columns, figsize=(5 * columns, 4 * nrows),
                             squeeze=False, layout="constrained")
    for ax, method in zip(axes.ravel(), methods):
        grid = np.full((len(sigmas), len(strikes)), np.nan)
        for r in selected:
            key = (r["K"] / r["S0"], r["sigma"])
            if r["method"] == method and key in baseline and r["rmse"] > 0 and baseline[key] > 0:
                grid[sigmas.index(key[1]), strikes.index(key[0])] = np.log10(baseline[key] / r["rmse"])
        im = ax.imshow(np.ma.masked_invalid(grid), origin="lower", cmap="RdBu", vmin=-2, vmax=2, aspect="auto")
        for (y, x), value in np.ndenumerate(grid):
            if np.isfinite(value):
                ax.text(x, y, f"{10**value:.1f}×", ha="center", va="center", fontsize=10,
                        color="white" if abs(value) > 1.2 else "black")
        ax.set_xticks(range(len(strikes)), [f"{s:g}" for s in strikes])
        ax.set_yticks(range(len(sigmas)), [f"{s:g}" for s in sigmas])
        ax.set(xlabel="K / S0", ylabel="Volatility", title=METHOD_LABELS[method])
    for ax in axes.ravel()[len(methods):]:
        ax.set_visible(False)
    fig.colorbar(im, ax=axes.ravel().tolist(), label="log10(RMSE plain / method); positive is better", shrink=0.8)
    fig.suptitle(f"{option_type}: accuracy improvement, T={maturity:g}, N={budget:,}")
    return fig, axes


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
