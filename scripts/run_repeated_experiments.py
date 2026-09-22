"""Run reproducible repeated OptionMC experiments and save raw/summary data."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

import numpy as np

from optionmc.experiments import (
    SUPPORTED_METHODS,
    SUPPORTED_OPTION_TYPES,
    aggregate_experiments,
    aggregate_sensitivity,
    run_repeated_experiments,
    run_repeated_sensitivity,
    write_csv,
)


BASE_PARAMETERS = {"S0": 100, "K": 100, "r": 0.05, "sigma": 0.2, "T": 1.0}
DEFAULT_PATH_COUNTS = (100, 500, 1_000, 5_000, 10_000, 50_000, 100_000)


def _comma_separated(value: str, cast):
    try:
        return tuple(cast(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run repeated call/put experiments for all OptionMC methods."
    )
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--seed-start", type=int, default=1_000)
    parser.add_argument(
        "--path-counts",
        type=lambda value: _comma_separated(value, int),
        default=DEFAULT_PATH_COUNTS,
        help="Comma-separated total path budgets.",
    )
    parser.add_argument(
        "--methods",
        type=lambda value: _comma_separated(value, str),
        default=SUPPORTED_METHODS,
        help=f"Comma-separated methods from: {', '.join(SUPPORTED_METHODS)}",
    )
    parser.add_argument(
        "--option-types",
        type=lambda value: _comma_separated(value, str),
        default=SUPPORTED_OPTION_TYPES,
        help="Comma-separated option types: call,put",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/data"))
    parser.add_argument("--sensitivity-paths", type=int, default=50_000)
    parser.add_argument("--skip-sensitivity", action="store_true")
    parser.add_argument("--no-warmup", action="store_true")
    return parser.parse_args()


def _version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    warmup = not args.no_warmup

    expected_runs = (
        len(args.methods)
        * len(args.option_types)
        * len(args.path_counts)
        * args.repetitions
    )
    print(
        f"Running {expected_runs:,} pricing experiments "
        f"({args.repetitions} repetitions per configuration)..."
    )
    raw = run_repeated_experiments(
        BASE_PARAMETERS,
        args.path_counts,
        repetitions=args.repetitions,
        methods=args.methods,
        option_types=args.option_types,
        seed_start=args.seed_start,
        warmup=warmup,
    )
    summary = aggregate_experiments(raw)
    raw_path = write_csv(raw, args.output_dir / "raw_experiment_results.csv")
    summary_path = write_csv(summary, args.output_dir / "experiment_summary.csv")
    print(f"Wrote {len(raw):,} raw rows to {raw_path}")
    print(f"Wrote {len(summary):,} summary rows to {summary_path}")

    sensitivity_outputs = {}
    if not args.skip_sensitivity:
        sensitivity_grids = {
            "sigma": np.linspace(0.05, 0.50, 20),
            "T": np.linspace(0.1, 2.0, 20),
            "K": np.linspace(70, 130, 30),
        }
        sensitivity_runs = (
            len(args.option_types)
            * sum(len(values) for values in sensitivity_grids.values())
            * args.repetitions
        )
        print(
            f"Running {sensitivity_runs:,} repeated sensitivity experiments "
            "with the control-variate estimator..."
        )
        sensitivity_raw = run_repeated_sensitivity(
            BASE_PARAMETERS,
            sensitivity_grids,
            n_paths=args.sensitivity_paths,
            repetitions=args.repetitions,
            method="control_variate",
            option_types=args.option_types,
            seed_start=args.seed_start + 10_000,
            warmup=warmup,
        )
        sensitivity_summary = aggregate_sensitivity(sensitivity_raw)
        sensitivity_raw_path = write_csv(
            sensitivity_raw, args.output_dir / "raw_sensitivity_results.csv"
        )
        sensitivity_summary_path = write_csv(
            sensitivity_summary, args.output_dir / "sensitivity_summary.csv"
        )
        sensitivity_outputs = {
            "raw_sensitivity_results": str(sensitivity_raw_path),
            "sensitivity_summary": str(sensitivity_summary_path),
        }
        print(
            f"Wrote {len(sensitivity_raw):,} sensitivity rows to "
            f"{sensitivity_raw_path}"
        )

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "base_parameters": BASE_PARAMETERS,
        "path_counts": list(args.path_counts),
        "methods": list(args.methods),
        "option_types": list(args.option_types),
        "repetitions": args.repetitions,
        "seed_start": args.seed_start,
        "warmup": warmup,
        "sensitivity_paths": (
            None if args.skip_sensitivity else args.sensitivity_paths
        ),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            "optionmc": _version("optionmc"),
            "numpy": _version("numpy"),
            "scipy": _version("scipy"),
            "matplotlib": _version("matplotlib"),
        },
        "outputs": {
            "raw_experiment_results": str(raw_path),
            "experiment_summary": str(summary_path),
            **sensitivity_outputs,
        },
    }
    metadata_path = args.output_dir / "experiment_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote reproducibility metadata to {metadata_path}")


if __name__ == "__main__":
    main()
