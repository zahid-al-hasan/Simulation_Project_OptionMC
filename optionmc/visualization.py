"""Visualization tools for option pricing results."""

import numpy as np
import matplotlib.pyplot as plt


def plot_convergence(n_paths_list, prices, analytical_price, title="MC Convergence"):
    """Plot option price estimates vs n_paths with confidence intervals."""
    # TODO
    pass


def plot_variance_comparison(n_paths_list, results_dict, analytical_price):
    """Compare multiple variance reduction methods on same plot."""
    # TODO
    pass


def plot_error_convergence(n_paths_list, errors_dict):
    """Plot log-relative-error vs log(n_paths) for each method."""
    # TODO
    pass


def plot_payoff_distribution(payoffs, option_type="call"):
    """Histogram of simulated option payoffs."""
    # TODO
    pass


def plot_stock_price_distribution(S_T, title="Terminal Stock Price Distribution"):
    """Histogram of terminal stock prices with lognormal overlay."""
    # TODO
    pass


def plot_parameter_sensitivity(param_range, mc_prices, analytical_prices, param_name):
    """Plot MC vs analytical price as a function of a parameter."""
    # TODO
    pass


def plot_efficiency_comparison(methods, errors, runtimes):
    """Bar chart or scatter of error vs runtime for each method."""
    # TODO
    pass
