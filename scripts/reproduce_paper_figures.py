"""Optional simulations reproducing the base paper's four figure layouts."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import numpy as np
from optionmc.experiments import run_repeated_sensitivity
from optionmc.analytics import aggregate_sensitivity
from scripts.results_io import write_csv
from scripts.paper_figures import (
    BASE_PARAMETERS, generate_paper_convergence_figures,
    plot_paper_distribution_composite, plot_paper_sensitivity_composite,
)


def main():
    output = Path("artifacts/paper_reproductions")
    output.mkdir(parents=True, exist_ok=True)
    generate_paper_convergence_figures(output)
    plot_paper_distribution_composite(output)
    raw = run_repeated_sensitivity(BASE_PARAMETERS,
        {"sigma": np.linspace(0.05, 0.5, 20), "T": np.linspace(0.1, 2, 20),
         "K": np.linspace(70, 130, 30)}, n_paths=16384)
    write_csv(raw, output / "sensitivity_raw.csv")
    plot_paper_sensitivity_composite(aggregate_sensitivity(raw), output)
    print(f"Paper reproductions: {output}")


if __name__ == "__main__":
    main()
