"""Statistical accounting and end-to-end checks for the comparison pipeline."""
import json

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest
from scipy.stats import t

from optionmc.pricing import OptionPricing, price_once, SUPPORTED_METHODS
from optionmc.experiments import run_repeated_experiments, run_scenarios
from optionmc.analytics import aggregate_experiments, time_to_target
from scripts.results_io import write_csv
from scripts.generate_report_figures import generate_report

PARAMETERS = dict(S0=100, K=100, r=0.05, sigma=0.2, T=1.0)


@pytest.mark.parametrize("method", ["control_variate", "antithetic_control"])
@pytest.mark.parametrize("option", ["call", "put"])
def test_pilot_budget_and_repeatability(method, option):
    first = price_once(OptionPricing(**PARAMETERS, n_paths=1024, seed=42), method, option)
    second = price_once(OptionPricing(**PARAMETERS, n_paths=1024, seed=42), method, option)
    assert first["price"] == second["price"]
    assert first["payoff_evaluations"] == 1024
    assert first["pilot_evaluations"] + first["production_evaluations"] == 1024
    unit = 2 if method == "antithetic_control" else 1
    assert first["effective_samples"] * unit == first["production_evaluations"]
    assert np.isfinite(first["std_error"])


def test_combined_estimator_matches_independent_manual_calculation():
    # Independently reconstruct the paired pilot and production estimator.
    seeds = np.random.SeedSequence(9).spawn(2)
    def paired(n, seed):
        z = np.random.default_rng(seed).normal(size=n)
        terminals = 100 * np.exp(0.03 + 0.2 * np.array([z, -z]))
        y = (np.maximum(terminals - 100, 0) * np.exp(-0.05)).mean(axis=0)
        x = (terminals * np.exp(-0.05)).mean(axis=0)
        return y, x
    yp, xp = paired(51, seeds[0])
    beta = np.cov(yp, xp, ddof=1)[0, 1] / np.var(xp, ddof=1)
    y, x = paired(461, seeds[1])
    adjusted = y - beta * (x - 100)
    result = OptionPricing(**PARAMETERS, n_paths=1024, seed=9).antithetic_control_mc()
    assert result["price"] == pytest.approx(adjusted.mean())
    assert result["variance"] == pytest.approx(adjusted.var(ddof=1) / len(adjusted))


@pytest.mark.parametrize("method", ["control_variate", "antithetic_control"])
def test_deterministic_control_has_zero_uncertainty(method):
    result = price_once(OptionPricing(**{**PARAMETERS, "sigma": 0}, n_paths=128, seed=3), method)
    assert result["price"] == pytest.approx(100 - 100 * np.exp(-0.05))
    assert result["std_error"] == pytest.approx(0, abs=1e-12)


def test_qmc_equal_power_two_scrambles_and_t_interval():
    result = OptionPricing(**PARAMETERS, n_paths=1024, seed=2).quasi_mc()
    assert result["replication_path_counts"] == (128,) * 8
    assert result["ci_upper"] - result["price"] == pytest.approx(t.ppf(0.975, 7) * result["std_error"])


def test_one_sample_per_stratum_is_rejected():
    with pytest.raises(ValueError, match="two samples"):
        OptionPricing(**PARAMETERS, n_paths=16).stratified_mc(n_strata=16)


def test_scenarios_are_not_pooled():
    rows = run_scenarios(PARAMETERS, {"atm": {}, "otm": {"K": 120}}, [128],
                         repetitions=3, methods=["standard", "antithetic_control"],
                         option_types=["call"], warmup=False)
    summary = aggregate_experiments(rows)
    assert len(summary) == 4
    assert {row["K"] for row in summary} == {100, 120}
    assert all(row["repetitions"] == 3 for row in summary)
    for row in rows:
        row["scenario_id"] = "same"
    with pytest.raises(ValueError, match="one set"):
        aggregate_experiments(rows)


def test_target_uses_smallest_qualifying_budget_and_marks_unreached():
    def row(n, error, seconds):
        return dict(scenario_id="baseline", method="standard", option_type="call",
                    n_paths=n, rmse=error, median_runtime_seconds=seconds,
                    runtime_q1_seconds=seconds * 0.8, runtime_q3_seconds=seconds * 1.2)
    # Non-monotone error: no smoothing, interpolation, or selecting lucky prices.
    result = time_to_target([row(256, .02, .2), row(64, .1, .1), row(128, .01, .15)], [.03, .001])
    assert result[0]["n_paths"] == 128
    assert result[0]["median_runtime_seconds"] == .15
    assert result[1]["reached"] is False
    assert result[1]["median_runtime_seconds"] is None


def test_saved_data_report_supports_method_and_option_subsets(tmp_path):
    raw = run_repeated_experiments(PARAMETERS, [64, 128], repetitions=3,
                                   methods=["control_variate", "antithetic_control"],
                                   option_types=["put"], warmup=False)
    data = tmp_path / "data"
    write_csv(raw, data / "raw_experiment_results.csv")
    (data / "experiment_metadata.json").write_text(json.dumps({"config": {"targets": [.1]}, "scenarios_run": False}))
    generate_report(data, tmp_path / "figures")
    assert (tmp_path / "figures/put_time_to_rmse_0.1.png").exists()
    assert (tmp_path / "figures/FIGURE_GUIDE.md").exists()


def test_fractional_budget_rejected_before_simulation():
    with pytest.raises(ValueError):
        run_repeated_experiments(PARAMETERS, [128.5], repetitions=2)
