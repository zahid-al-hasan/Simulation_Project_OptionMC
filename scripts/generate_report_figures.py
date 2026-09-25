"""Render saved experiment data; this module never runs a pricing simulation."""

import json
from pathlib import Path
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from optionmc.analytics import aggregate_experiments, time_to_target
from scripts.results_io import read_numeric_csv, write_csv
from scripts.plots import (
    plot_benchmark, plot_target_times, plot_error_boxplots, plot_scenario_heatmap,
    plot_price_surface, METHOD_LABELS,
)


def _save(figure, path):
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def generate_report(data_dir, output_dir, surface_method="antithetic_control"):
    """Recompute tables from raw data, then write baseline plots and grid heatmaps."""
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = read_numeric_csv(data_dir / "raw_experiment_results.csv")
    summary = aggregate_experiments(raw)
    metadata = json.loads((data_dir / "experiment_metadata.json").read_text())
    targets = time_to_target(summary, metadata["config"]["targets"])
    write_csv(summary, output_dir / "experiment_summary.csv")
    write_csv(targets, output_dir / "time_to_target.csv")
    for option in sorted({row["option_type"] for row in summary}):
        rows = [r for r in summary if r["option_type"] == option]
        for metric in ("price", "rmse", "runtime", "accuracy_runtime", "variance", "coverage"):
            if metric == "variance" and not any(r["method"] == "standard" for r in rows):
                continue
            figure, _ = plot_benchmark(rows, metric)
            _save(figure, output_dir / f"{option}_{metric}.png")
        figure, _ = plot_error_boxplots([r for r in raw if r["option_type"] == option])
        _save(figure, output_dir / f"{option}_error_boxplots.png")
        for target in metadata["config"]["targets"]:
            rows = [r for r in targets if r["option_type"] == option and r["target_rmse"] == target]
            figure, _ = plot_target_times(rows)
            _save(figure, output_dir / f"{option}_time_to_rmse_{target:g}.png")
    scenario_path = data_dir / "raw_scenario_results.csv"
    if metadata.get("scenarios_run") and scenario_path.exists():
        scenarios = aggregate_experiments(read_numeric_csv(scenario_path))
        write_csv(scenarios, output_dir / "scenario_summary.csv")
        available_methods = {r["method"] for r in scenarios}
        if surface_method not in available_methods:
            surface_method = next(m for m in METHOD_LABELS if m in available_methods)
        for option in sorted({r["option_type"] for r in scenarios}):
            for maturity in sorted({r["T"] for r in scenarios}):
                for budget in sorted({r["n_paths"] for r in scenarios}):
                    figure, _ = plot_price_surface(scenarios, option, maturity, budget, surface_method)
                    _save(figure, output_dir / f"{option}_surface_T{maturity:g}_N{budget}.png")
                    if "standard" in metadata["config"]["methods"] and len(metadata["config"]["methods"]) > 1:
                        figure, _ = plot_scenario_heatmap(scenarios, option, maturity, budget)
                        _save(figure, output_dir / f"{option}_heatmap_T{maturity:g}_N{budget}.png")
    lines = ["# Experiment figures", "", f"Source data: `{data_dir.resolve()}`", "",
             "- Price shading: empirical middle 95% of individual runs, not a confidence interval for the mean.",
             "- Runtime shading/error bars: interquartile range across timed runs.",
             "- Target time: median estimator runtime at the smallest tested budget meeting repeated-run RMSE.",
             "- Unreached targets are not extrapolated. Threshold selection has sampling uncertainty.",
             "- Pilot samples are included in the total evaluation budget and runtime.",
             "- Heatmap numbers: plain-MC RMSE divided by method RMSE; larger than one is better.",
             f"- 3D surfaces: Black-Scholes prices versus moneyness and volatility, with {surface_method} mean estimates overlaid. Each maturity and budget has a separate figure.",
             "- Surface points use saved scenario results; the smooth reference surface is evaluated analytically. Use RMSE plots to see small differences.",
             "- All comparisons are specific to this one-dimensional European-option model.", "",
             "## Baseline at the largest budget", "",
             "| Option | Method | N | RMSE | Median ms |", "|---|---|---:|---:|---:|"]
    for row in summary:
        if row["n_paths"] == max(r["n_paths"] for r in summary):
            lines.append(f"| {row['option_type']} | {row['method']} | {row['n_paths']} | {row['rmse']:.6g} | {1000*row['median_runtime_seconds']:.4g} |")
    (output_dir / "FIGURE_GUIDE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("artifacts/refactored_data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/refactored_figures"))
    parser.add_argument("--surface-method", choices=METHOD_LABELS, default="antithetic_control",
                        help="Method to overlay on the 3D reference surface (defaults to the hybrid).")
    args = parser.parse_args()
    generate_report(args.data_dir, args.output_dir, args.surface_method)
    print(f"Figures and interpretation guide: {args.output_dir}")


if __name__ == "__main__":
    main()
