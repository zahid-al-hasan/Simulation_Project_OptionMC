"""Smoke tests for plotting helpers."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from optionmc.visualization import (
    plot_convergence,
    plot_efficiency_comparison,
    plot_error_convergence,
    plot_parameter_sensitivity,
    plot_payoff_distribution,
    plot_stock_price_distribution,
    plot_variance_comparison,
)


def _assert_figure(result):
    figure, axis = result
    assert figure.axes[0] is axis
    plt.close(figure)


def test_all_plotting_functions_return_figures():
    paths = np.array([100, 1_000, 10_000])
    prices = np.array([10.1, 10.4, 10.45])
    exact = 10.4506
    results = {
        "standard": {"prices": prices},
        "control_variate": {"prices": np.array([10.3, 10.44, 10.451])},
    }
    _assert_figure(plot_convergence(paths, prices, exact))
    _assert_figure(plot_variance_comparison(paths, results, exact))
    _assert_figure(
        plot_error_convergence(
            paths, {"standard": [0.03, 0.01, 0.001], "quasi": [0.01, 0.001, 0.0001]}
        )
    )
    _assert_figure(plot_payoff_distribution(np.arange(20.0)))
    _assert_figure(plot_stock_price_distribution(np.exp(np.linspace(4.0, 5.0, 100))))
    _assert_figure(plot_parameter_sensitivity([0.1, 0.2], [8, 10], [8.1, 10.1], "sigma"))
    _assert_figure(plot_efficiency_comparison(["standard", "quasi"], [0.01, 0.001], [0.1, 0.2]))
