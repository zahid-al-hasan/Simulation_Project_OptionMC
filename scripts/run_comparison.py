"""Compare Monte Carlo variance-reduction methods side by side."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from optionmc.analytics import (
    convergence_analysis,
    efficiency_ratio,
    variance_reduction_ratio,
)
from optionmc.models import BlackScholesAnalytical
from optionmc.visualization import (
    plot_efficiency_comparison,
    plot_error_convergence,
    plot_variance_comparison,
)


def main():
    S0, K, r, sigma, T = 100, 100, 0.05, 0.2, 1.0
    n_paths_list = [100, 500, 1_000, 5_000, 10_000, 50_000, 100_000]
    analytical_price = BlackScholesAnalytical(S0, K, r, sigma, T).call_price()
    methods = ["standard", "antithetic", "control_variate", "stratified", "quasi"]
    results = {
        method: convergence_analysis(
            S0, K, r, sigma, T, n_paths_list, method=method, seed=42
        )
        for method in methods
    }

    output_dir = Path("artifacts")
    output_dir.mkdir(exist_ok=True)

    figure, _ = plot_variance_comparison(n_paths_list, results, analytical_price)
    figure.savefig(output_dir / "method_comparison.png", dpi=160)

    figure, _ = plot_error_convergence(
        n_paths_list,
        {method: result["relative_errors"] for method, result in results.items()},
    )
    figure.savefig(output_dir / "error_convergence.png", dpi=160)

    final_errors = [results[method]["relative_errors"][-1] for method in methods]
    final_runtimes = [results[method]["runtimes"][-1] for method in methods]
    figure, _ = plot_efficiency_comparison(methods, final_errors, final_runtimes)
    figure.savefig(output_dir / "efficiency_comparison.png", dpi=160)

    standard_variance = results["standard"]["variances"][-1]
    standard_error = results["standard"]["std_errors"][-1]
    standard_runtime = results["standard"]["runtimes"][-1]
    print(f"Black-Scholes call price: {analytical_price:.8f}")
    print(f"{'Method':<18} {'Price':>11} {'Std. error':>12} {'VRR':>10} {'Efficiency':>12}")
    for method in methods:
        result = results[method]
        variance_ratio = variance_reduction_ratio(
            standard_variance, result["variances"][-1]
        )
        efficiency = efficiency_ratio(
            standard_error,
            standard_runtime,
            result["std_errors"][-1],
            result["runtimes"][-1],
        )
        print(
            f"{method:<18} {result['prices'][-1]:>11.6f} "
            f"{result['std_errors'][-1]:>12.6f} {variance_ratio:>10.2f} "
            f"{efficiency:>12.2f}"
        )
    print(f"Plots written to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
