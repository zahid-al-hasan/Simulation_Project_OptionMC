"""Parameter sensitivity analysis script."""

import numpy as np
from optionmc.analytics import parameter_sensitivity
from optionmc.visualization import plot_parameter_sensitivity


def main():
    S0, K, r, sigma, T = 100, 100, 0.05, 0.2, 1.0

    # Volatility sensitivity
    sigma_range = np.linspace(0.05, 0.50, 20)
    # TODO: call parameter_sensitivity for sigma, plot results

    # Time-to-maturity sensitivity
    T_range = np.linspace(0.1, 2.0, 20)
    # TODO: call parameter_sensitivity for T, plot results

    # Moneyness analysis
    K_range = np.linspace(70, 130, 30)
    # TODO: sweep K, compare MC vs analytical


if __name__ == "__main__":
    main()
