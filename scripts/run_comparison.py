"""Main script: compare all variance reduction methods side-by-side."""

import numpy as np
from optionmc.models import BlackScholesAnalytical
from optionmc.pricing import OptionPricing
from optionmc.analytics import convergence_analysis, variance_reduction_ratio
from optionmc.visualization import (
    plot_convergence,
    plot_variance_comparison,
    plot_error_convergence,
    plot_efficiency_comparison,
)


def main():
    S0, K, r, sigma, T = 100, 100, 0.05, 0.2, 1.0
    n_paths_list = [100, 500, 1000, 5000, 10000, 50000, 100000]
    bs = BlackScholesAnalytical(S0, K, r, sigma, T)
    analytical_price = bs.call_price()

    methods = ["standard", "antithetic", "control_variate", "stratified", "quasi"]
    results = {}

    for method in methods:
        # TODO: run convergence_analysis for each method, store in results
        pass

    # TODO: generate comparison plots
    # TODO: print summary table of VRR and efficiency


if __name__ == "__main__":
    main()
