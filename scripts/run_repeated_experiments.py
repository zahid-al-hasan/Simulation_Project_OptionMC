"""Run the configured benchmark and optional market-scenario grid."""
import argparse
from datetime import datetime, timezone
from itertools import product
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

from optionmc.analytics import aggregate_experiments, time_to_target
from optionmc.experiments import run_repeated_experiments, run_scenarios
from scripts.results_io import write_csv


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs/baseline.json")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/refactored_data"))
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--seed-start", type=int)
    parser.add_argument("--path-counts", help="Comma-separated budgets, each a power of two")
    parser.add_argument("--methods", help="Comma-separated method names")
    parser.add_argument("--option-types", help="call,put or either one")
    parser.add_argument("--scenarios", action="store_true", help="Also run the configured volatility/moneyness/maturity grid")
    parser.add_argument("--no-warmup", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = json.loads(args.config.read_text())
    for key in ("repetitions", "seed_start"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    for key in ("path_counts", "methods", "option_types"):
        if getattr(args, key):
            config[key] = [int(v) if key == "path_counts" else v.strip()
                           for v in getattr(args, key).split(",")]
    # Eight QMC scrambles each receive a power-of-two number of points.
    budgets = config["path_counts"] + (config["scenario_grid"]["path_counts"] if args.scenarios else [])
    if any(not isinstance(n, int) or n < 16 or n & (n - 1) for n in budgets):
        raise ValueError("benchmark budgets must be powers of two and at least 16")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    settings = {key: config[key] for key in ("repetitions", "methods", "option_types", "seed_start")}
    settings["warmup"] = not args.no_warmup
    print("Running baseline comparisons...", flush=True)
    raw = run_repeated_experiments(config["parameters"], config["path_counts"], **settings)
    summary = aggregate_experiments(raw)
    write_csv(raw, args.output_dir / "raw_experiment_results.csv")
    write_csv(summary, args.output_dir / "experiment_summary.csv")
    write_csv(time_to_target(summary, config["targets"]), args.output_dir / "time_to_target.csv")
    scenarios_run = args.scenarios
    if scenarios_run:
        grid = config["scenario_grid"]
        scenarios = {f"m{m:g}_sigma{s:g}_T{t:g}": {"K": config["parameters"]["S0"] * m, "sigma": s, "T": t}
                     for m, s, t in product(grid["moneyness"], grid["sigma"], grid["T"])}
        print(f"Running {len(scenarios)} market scenarios...", flush=True)
        scenario_settings = {**settings, "seed_start": config["seed_start"] + 100000}
        rows = run_scenarios(config["parameters"], scenarios, grid["path_counts"], **scenario_settings)
        write_csv(rows, args.output_dir / "raw_scenario_results.csv")
        write_csv(aggregate_experiments(rows), args.output_dir / "scenario_summary.csv")
    metadata = dict(created_utc=datetime.now(timezone.utc).isoformat(), config=config,
                    scenarios_run=scenarios_run, warmup=not args.no_warmup,
                    python=sys.version, platform=platform.platform(),
                    packages={name: importlib.metadata.version(name) for name in ("numpy", "scipy", "matplotlib")})
    (args.output_dir / "experiment_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Saved baseline ({len(raw)} runs), summaries and metadata to {args.output_dir}")


if __name__ == "__main__":
    main()
