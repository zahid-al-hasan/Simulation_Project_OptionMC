"""Tests for repeated experiments, aggregation, and CSV persistence."""

import numpy as np
import pytest

from optionmc.experiments import (
    aggregate_experiments,
    aggregate_sensitivity,
    read_csv,
    run_repeated_experiments,
    run_repeated_sensitivity,
    write_csv,
)


PARAMETERS = {"S0": 100, "K": 100, "r": 0.05, "sigma": 0.2, "T": 1.0}


def test_repeated_grid_is_complete_and_reproducible():
    arguments = dict(
        parameters=PARAMETERS,
        path_counts=[100, 200],
        repetitions=2,
        methods=["standard", "control_variate"],
        option_types=["call", "put"],
        seed_start=10,
        warmup=False,
    )
    first = run_repeated_experiments(**arguments)
    second = run_repeated_experiments(**arguments)
    assert len(first) == 16
    assert [row["mc_price"] for row in first] == [
        row["mc_price"] for row in second
    ]
    assert {row["option_type"] for row in first} == {"call", "put"}
    assert all(row["runtime_seconds"] >= 0 for row in first)
    assert all(row["ci_covers_exact"] in {0, 1} for row in first)


def test_aggregation_produces_empirical_metrics():
    raw = run_repeated_experiments(
        PARAMETERS,
        [200, 400],
        repetitions=3,
        methods=["standard", "antithetic"],
        option_types=["call"],
        seed_start=20,
        warmup=False,
    )
    summary = aggregate_experiments(raw)
    assert len(summary) == 4
    assert all(row["repetitions"] == 3 for row in summary)
    assert all(row["rmse"] >= 0 for row in summary)
    assert all(0 <= row["coverage_rate"] <= 1 for row in summary)
    standard = [row for row in summary if row["method"] == "standard"]
    assert all(row["empirical_vrr"] == pytest.approx(1.0) for row in standard)
    assert all(np.isfinite(row["rmse_convergence_slope"]) for row in summary)


def test_sensitivity_grid_and_aggregation():
    raw = run_repeated_sensitivity(
        PARAMETERS,
        {"sigma": [0.1, 0.2], "K": [90, 110]},
        n_paths=200,
        repetitions=2,
        method="control_variate",
        option_types=["call", "put"],
        seed_start=30,
        warmup=False,
    )
    assert len(raw) == 16
    summary = aggregate_sensitivity(raw)
    assert len(summary) == 8
    assert all(row["repetitions"] == 2 for row in summary)
    assert all(row["mean_absolute_error"] >= 0 for row in summary)


def test_csv_round_trip(tmp_path):
    rows = [{"method": "standard", "price": 10.5}, {"method": "sobol", "price": 10.4}]
    output = write_csv(rows, tmp_path / "results.csv")
    loaded = read_csv(output)
    assert loaded == [
        {"method": "standard", "price": "10.5"},
        {"method": "sobol", "price": "10.4"},
    ]


def test_invalid_experiment_configuration():
    with pytest.raises(ValueError):
        run_repeated_experiments(PARAMETERS, [101], methods=["antithetic"])
    with pytest.raises(ValueError):
        run_repeated_experiments(PARAMETERS, [100], repetitions=1)
