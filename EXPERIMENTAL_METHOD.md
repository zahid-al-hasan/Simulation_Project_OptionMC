# OptionMC Repeated-Experiment Protocol

This document defines the reproducible experiment used for the final OptionMC comparison. It replaces conclusions based on a single random seed with measurements across independent seeds and randomized QMC scrambles.

## 1. Environment

Run from the project root in the existing virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pytest -q -p no:cacheprovider
```

The completed reference run used Python 3.13.2, NumPy 2.5.3, SciPy 1.18.1, and Matplotlib 3.11.2 on Windows 11. Exact package and platform information is saved in `artifacts/data/experiment_metadata.json` whenever the experiment is run.

## 2. Baseline problem

The fixed Black-Scholes inputs are:

| Parameter | Value |
| --- | ---: |
| Initial stock price, `S0` | 100 |
| Strike, `K` | 100 |
| Risk-free rate, `r` | 0.05 |
| Volatility, `sigma` | 0.20 |
| Maturity, `T` | 1 year |

The analytical reference prices are `10.4505835722` for the call and `5.5735260223` for the put.

All estimators receive the same total terminal-price budget. The path-count grid is `100, 500, 1,000, 5,000, 10,000, 50,000, 100,000`.

## 3. Methods and fair-budget rules

- **Standard MC:** independent standard-normal draws.
- **Antithetic:** `N/2` independent draws paired with their negatives, producing `N` terminal prices. Pair averages are the independent observations used for standard errors.
- **Control variate:** discounted terminal stock is the control and has known expectation `S0`. The coefficient is estimated from sample covariance.
- **Stratified:** ten equal-probability strata are used when possible. The standard error combines within-stratum variances.
- **Sobol and Halton QMC:** the total budget is split across eight independent randomized scrambles. QMC standard errors are calculated from the replicate means. Child scramble seeds are derived independently from each outer seed with `numpy.random.SeedSequence.spawn`.

The same outer seeds are reused across methods. This common-random-number design makes method comparisons less sensitive to unrelated seed selection while preserving independence between the 30 repetitions of each configuration.

## 4. Repeated comparison

For every option type, method, and path count, run 30 repetitions with outer seeds 1000 through 1029. Save every run, rather than only an aggregate, so results can be audited or re-aggregated.

```powershell
python scripts\run_repeated_experiments.py --repetitions 30 --output-dir artifacts\data
```

The command also runs repeated sensitivity studies unless `--skip-sensitivity` is supplied. Useful smaller checks are:

```powershell
python scripts\run_repeated_experiments.py --repetitions 3 --path-counts 100,1000 --skip-sensitivity --output-dir artifacts\smoke
python scripts\run_repeated_experiments.py --help
```

The full run creates:

- `raw_experiment_results.csv`: one row per pricing run (2,520 rows with the default configuration).
- `experiment_summary.csv`: one row per option-method-budget combination (84 rows).
- `raw_sensitivity_results.csv`: one row per repeated sensitivity run (4,200 rows).
- `sensitivity_summary.csv`: aggregated sensitivity results (140 rows).
- `experiment_metadata.json`: inputs, seeds, platform, package versions, and output paths.

## 5. Metrics

Let `V` be the Black-Scholes price and `V_hat_j` the estimate from repetition `j`.

- Bias: `mean(V_hat_j) - V`.
- Mean absolute error: `mean(abs(V_hat_j - V))`.
- RMSE: `sqrt(mean((V_hat_j - V)^2))`.
- Empirical estimator variance: sample variance of the 30 independent price estimates.
- Empirical variance-reduction ratio: standard-MC empirical variance divided by the compared method's empirical variance at the same option type and budget.
- Confidence-interval coverage: fraction of per-run reported 95% intervals containing the analytical price.
- Runtime: median elapsed pricing time, with first and third quartiles also retained.
- Convergence slope: least-squares slope of `log(RMSE)` against `log(N)` across the path-count grid.
- Empirical efficiency ratio: standard-MC `RMSE^2 * median runtime` divided by the corresponding value for the compared method.

RMSE and variance reduction are calculated from independent outer runs. They are more defensible than comparing the absolute error of one lucky seed.

## 6. Sensitivity study

The sensitivity study uses the control-variate estimator, 50,000 paths, 30 repetitions, and both call and put payoffs. It varies:

- Volatility from 0.05 to 0.50 over 20 values.
- Maturity from 0.1 to 2.0 years over 20 values.
- Strike from 70 to 130 over 30 values; this also gives moneyness `K/S0` from 0.70 to 1.30.

Seeds 11000 through 11029 are reused across grid values. This provides common random numbers for smoother comparisons while each repetition remains independent of the other repetitions.

## 7. Generate the report outputs

After the CSV files exist, generate figures and the final comparison table:

```powershell
python scripts\generate_report_figures.py --data-dir artifacts\data --output-dir artifacts\report
```

This produces price and RMSE convergence, variance reduction, accuracy-versus-runtime, confidence-interval coverage, distribution validation, and call/put sensitivity figures. It also writes `final_method_comparison.csv` and `final_method_comparison.md`.

## 8. Interpretation rules

- Treat timings as measurements for this machine and software environment, not universal constants.
- Treat QMC advantages as results for this smooth, one-dimensional European payoff problem; higher-dimensional path-dependent products may behave differently.
- Do not infer that every variance-reduction technique improves every payoff. The put antithetic result in this experiment is a concrete counterexample.
- Coverage based on 30 repetitions is a useful diagnostic but has substantial binomial uncertainty.
- The experiment changes the sampling estimator, not the Black-Scholes market model. European exercise, constant parameters, no dividends, and lognormal GBM remain scope limitations.
