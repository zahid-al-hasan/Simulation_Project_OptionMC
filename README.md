# OptionMC: understandable numerical comparisons

**Research question:** Which sampling method reaches a reliable European option
price with the least computational effort, and how does that change across
volatility, maturity, and moneyness?

The project compares plain Monte Carlo, antithetic variates, independent-pilot
control variates, stratified sampling, scrambled Sobol, scrambled Halton, and
**antithetic + control variates** against Black-Scholes.

## Start here

Run commands from the project root. Python 3.10 or newer is required.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -q
```

Use that interpreter in place of `python` below, or activate the environment.
The `-m scripts...` commands also work from the project root without an editable
package installation, provided the dependencies are available.

```powershell
# Quick end-to-end check: seven methods, calls and puts, three budgets.
python -m scripts.run_repeated_experiments --repetitions 5 --path-counts 256,1024,4096 --output-dir artifacts/quick_data
python -m scripts.generate_report_figures --data-dir artifacts/quick_data --output-dir artifacts/quick_figures

# Final baseline: 100 independent repetitions at each configured budget.
# --scenarios adds 27 combinations of moneyness, volatility and maturity.
python -m scripts.run_repeated_experiments --scenarios
python -m scripts.generate_report_figures
```

Edit [configs/baseline.json](configs/baseline.json) to change parameters, methods,
budgets, repetitions, accuracy targets, or the scenario grid. CLI options can
select a subset of methods or option types. Scenario comparisons use the separate
budget list in `scenario_grid`; the default is 4,096 evaluations per estimate.

## Read the code in this order

| File | What to understand |
|---|---|
| `optionmc/models.py` | Exact terminal GBM simulation and Black-Scholes prices |
| `optionmc/samplers.py` | Independent, stratified, and scrambled QMC inputs |
| `optionmc/variance_reduction.py` | Antithetic pairing and control corrections |
| `optionmc/pricing.py` | How each method turns inputs into one price estimate |
| `optionmc/experiments.py` | Independent repetitions, budgets, seeds, scenarios |
| `optionmc/analytics.py` | RMSE, bias, variance, coverage, target-accuracy tables |
| `scripts/plots.py` | Reusable graphs that return Matplotlib figures |
| `scripts/results_io.py` | CSV reading and writing |
| `scripts/generate_report_figures.py` | Generate reports and figures from saved CSVs |

The `optionmc/` package contains the numerical methods, experiment loops, and
statistics. All report, graph, and CSV generation lives in `scripts/`.
`price_once` in `pricing.py` is the single method dispatcher.
Import analysis functions from `analytics.py` and experiment functions from
`experiments.py` directly; there are no forwarding compatibility wrappers.

Input checks are kept where they clarify valid inputs or prevent invalid
numerical calculations. CSV columns use explicit types instead of try/except
conversion. Dependencies are listed once in `pyproject.toml`.

A typical calculation:

```python
from optionmc.pricing import OptionPricing, price_once

option = OptionPricing(S0=100, K=100, r=0.05, sigma=0.2, T=1,
                       n_paths=4096, seed=42)
result = price_once(option, "antithetic_control", "call")
print(result["price"], result["std_error"])
print(result["payoff_evaluations"], result["pilot_evaluations"])
```

## Numerical conventions

- A budget is the **total number of discounted-payoff evaluations**, including
  pilot samples. An antithetic pair costs two evaluations.
- GBM is sampled directly at maturity. There is no time-discretization error.
- Antithetic uncertainty uses independent pair averages.
- Control beta is fitted using an independent pilot: roughly 10% of observations,
  with a minimum of two pilot observations. For the hybrid, observations are pairs,
  and both the pilot payoff and stock control are averaged within each pair.
  Only production observations enter the final price. Pilot cost is included.
- A deterministic stock control uses beta zero.
- Stratification uses up to 16 equally probable strata, choosing a divisor of the
  budget and requiring at least two observations per stratum. Uncertainty combines
  within-stratum variances. The fixed stratum count is a design choice, not a claim
  that this is the best possible stratification scheme.
- QMC uses up to eight independent, equal-size scrambles. Benchmark budgets are
  powers of two (at least 16), so each Sobol scramble preserves its balance.
  Direct API calls may use other divisible budgets; Sobol warnings are visible.
  A Student-t interval is computed from scramble means. This is an approximation;
  repeated-run coverage provides an empirical check.
- A fixed seed reproduces estimates. Repetitions use different seeds. Methods and
  budgets may share a repetition seed, so comparisons can be paired; results
  across methods should not be treated as independent observations.
- Method/budget jobs are interleaved in a reproducibly shuffled order. Each method
  is warmed up first. Runtime includes sampling setup, pilot fitting, payoffs,
  and uncertainty calculations, but excludes imports, plotting, and file I/O.
  Timings describe this machine and software environment.

## The comparison figures

| Figure | Question |
|---|---|
| Price versus evaluation budget | Do methods approach the correct option price? |
| RMSE versus evaluation budget | How quickly does typical error decrease? |
| Signed-error boxplots | How stable are independent estimates? |
| Runtime versus budget | How does computation grow? |
| RMSE versus runtime | What accuracy do we get for our time? |
| Time to target RMSE | Which method reaches the same accuracy fastest? |
| Variance reduction | How much estimator variability is removed? |
| Confidence-interval coverage | Are the reported intervals credible? |
| Scenario heatmaps | Where does each method outperform plain MC? |
| 3D price surfaces | How does option price vary with moneyness and volatility? |

The 3D figures overlay saved mean simulation prices on a smooth Black-Scholes
surface, holding spot, interest rate, maturity and budget fixed. Calls and puts
have separate figures for each maturity. The default overlay is antithetic +
control variates; choose another method without rerunning experiments:

```powershell
python -m scripts.generate_report_figures --surface-method sobol
```

This replaces the surface figures with the selected overlay. These are static
PNG figures for the report. The surface explains pricing relationships; use RMSE
plots to compare errors too small to see on the price scale.

Price shading shows the empirical middle 95% of **individual runs**. It is not
uncertainty in the mean over all repetitions. Summary CSVs contain both quantities.
Runtime shading and timing-bar error bars show the interquartile range.

**Time to target** uses the smallest tested budget whose repeated-run RMSE meets
that target. The displayed time is the median time for one estimator at that
budget, not the duration of the whole benchmark. It does not stop a single run
when that run happens to be close to the exact price. Unreached targets are labeled;
no extrapolation is performed. The threshold decision itself has sampling
uncertainty, particularly with few repetitions or near the threshold. Refine the
budget grid and increase repetitions before claiming precise speedup factors.

The heatmaps compare all selected methods at equal budgets. Their annotations are
plain-MC RMSE divided by method RMSE; values above one favor the method. Separate
panels are generated for calls, puts, and each maturity.

## Saved data and reproducibility

The experiment command writes to `artifacts/refactored_data/` by default:

- `raw_experiment_results.csv`: one record per baseline run, with parameters,
  seed, price, errors, uncertainty, runtime, pilot cost and QMC replication details.
- `experiment_summary.csv`: repeated-run statistics per method and budget.
- `time_to_target.csv`: target reached, selected budget, RMSE and timing quartiles.
- `raw_scenario_results.csv` and `scenario_summary.csv`: when `--scenarios` is used.
- `experiment_metadata.json`: effective configuration, package versions and platform.

The figure command writes to `artifacts/refactored_figures/` by default.
It recomputes summaries from raw CSVs and writes PNGs, summary
CSVs, and `FIGURE_GUIDE.md`. It never reruns simulations. Use separate output folders
when preserving earlier experiments. Old result files created before this refactor
should be regenerated with the new pipeline.

`scenario_id` identifies a unique contract configuration. Aggregation rejects
mixed parameters under the same identifier rather than combining unlike results.

## Optional paper reproductions

```powershell
python -m scripts.reproduce_paper_figures
```

These simulations reproduce the four layouts in the base paper and save their
source data under `artifacts/paper_reproductions/`. They are separate from the
main repeated comparison. The paper-specific layout code is in
`scripts/paper_figures.py`.

Base paper: Herho, Kaban and Nugraha, *OptionMC: A Python Package for Monte Carlo
Pricing of European Options*, International Journal of Data Science, 6(2), 70-84.
[Published paper](https://ijods.org/index.php/ds/article/download/94/89).

The conclusions apply to constant-volatility, one-dimensional European-option
pricing. Black-Scholes is the exact model benchmark, not an observed market price.
The combined method is evaluated experimentally; it is not assumed to win.

## One-command run on Windows

```powershell
.\run_all.ps1                  # tests + quick benchmark + figures
.\run_all.ps1 -FullExperiments # tests + configured benchmark and scenario grid
```

The runner uses `.venv` when present, otherwise the active Python interpreter.
Install dependencies with the setup instructions first.
