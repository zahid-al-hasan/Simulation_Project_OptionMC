"""Run volatility, maturity, and moneyness sensitivity analyses."""

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

from optionmc.analytics import moneyness_analysis, parameter_sensitivity
from optionmc.visualization import plot_parameter_sensitivity


def _save_sensitivity(result, output_path):
    figure, _ = plot_parameter_sensitivity(
        result["param_range"],
        result["mc_prices"],
        result["analytical_prices"],
        result["param_name"],
    )
    figure.savefig(output_path, dpi=160)


def main():
    S0, K, r, sigma, T = 100, 100, 0.05, 0.2, 1.0
    n_paths = 50_000
    output_dir = Path("artifacts")
    output_dir.mkdir(exist_ok=True)

    volatility = parameter_sensitivity(
        S0,
        K,
        r,
        sigma,
        T,
        "sigma",
        np.linspace(0.05, 0.50, 20),
        n_paths,
        method="control_variate",
    )
    _save_sensitivity(volatility, output_dir / "volatility_sensitivity.png")

    maturity = parameter_sensitivity(
        S0,
        K,
        r,
        sigma,
        T,
        "T",
        np.linspace(0.1, 2.0, 20),
        n_paths,
        method="control_variate",
    )
    _save_sensitivity(maturity, output_dir / "maturity_sensitivity.png")

    moneyness = moneyness_analysis(
        S0,
        np.linspace(70, 130, 30),
        r,
        sigma,
        T,
        n_paths,
        method="control_variate",
    )
    moneyness_for_plot = {
        "param_name": "Strike (K)",
        "param_range": moneyness["strikes"],
        "mc_prices": moneyness["mc_prices"],
        "analytical_prices": moneyness["analytical_prices"],
    }
    _save_sensitivity(moneyness_for_plot, output_dir / "moneyness_sensitivity.png")
    print(f"Sensitivity plots written to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
