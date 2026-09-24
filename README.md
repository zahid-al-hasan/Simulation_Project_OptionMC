# OptionMC

OptionMC prices European call and put options under the Black-Scholes assumptions. It provides plain Monte Carlo simulation and four variance-reduction approaches:

- Antithetic variates
- Discounted-stock control variates
- Stratified sampling
- Scrambled Sobol and Halton quasi-Monte Carlo

The package also includes analytical Black-Scholes prices, confidence intervals, convergence and sensitivity analysis, comparison metrics, plots, and tests.

## Setup

Python 3.10 or newer is required. Create an isolated virtual environment from the project directory.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

If PowerShell execution policy prevents activation, invoke the environment's interpreter directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

## Run the tests

```powershell
python -m pytest -q
```

The verified implementation has 52 passing tests.

## Basic usage

```python
from optionmc import BlackScholesAnalytical, OptionPricing

parameters = dict(S0=100, K=100, r=0.05, sigma=0.2, T=1.0)

exact = BlackScholesAnalytical(**parameters).call_price()
pricer = OptionPricing(**parameters, n_paths=100_000, seed=42)

standard = pricer.standard_mc("call")
antithetic = pricer.antithetic_mc("call")
controlled = pricer.control_variate_mc("call")
stratified = pricer.stratified_mc("call", n_strata=10)
sobol = pricer.quasi_mc("call", method="sobol")

print(exact)
print(standard)
```

Every pricing method returns at least:

- `price`
- `std_error`
- `ci_lower` and `ci_upper`
- `variance`
- `runtime`
- `n_paths`
- `method`

## Run the experiments

```powershell
python scripts\run_comparison.py
python scripts\run_sensitivity.py
```

The scripts write comparison and sensitivity plots to the ignored `artifacts/` directory.

For the defensible final comparison, run 30 independent repetitions for calls and puts, then generate the report figures and tables:

```powershell
python scripts\run_repeated_experiments.py --repetitions 30 --output-dir artifacts\data
python scripts\generate_report_figures.py --data-dir artifacts\data --output-dir report_figures
python -m pip install -e ".[report]"
python scripts\build_report_pdf.py
```

On Windows, the native all-in-one runner avoids requiring Bash or WSL:

```powershell
.\run_all.ps1                  # tests and quick demonstrations
.\run_all.ps1 -FullExperiments # also rebuild repeated data and report figures
```

The full run saves raw per-seed data, aggregate CSV files, package/platform metadata, and 18 simulation/formula-driven report figures. These include reproductions of all four paper figures and one combined comparison of antithetic, control-variate, stratified, and Sobol QMC results. The generated CSV files and [figure provenance](report_figures/FIGURE_PROVENANCE.md) make every plotted value auditable. See [EXPERIMENTAL_METHOD.md](EXPERIMENTAL_METHOD.md) for the exact design and interpretation rules.

## Numerical conventions

- Stock prices follow risk-neutral geometric Brownian motion with constant volatility and no dividends.
- Payoffs are discounted at the continuously compounded risk-free rate.
- Antithetic standard errors use independent pair averages.
- Stratified standard errors combine within-stratum sample variances.
- QMC standard errors use independent randomized scrambles as replications.
- A fixed seed makes pseudorandom and scrambled low-discrepancy runs reproducible.

## Project report

- [FINAL_PROJECT_REPORT.md](FINAL_PROJECT_REPORT.md) contains the completed methods, repeated results, discussion, and limitations.
- [EXPERIMENTAL_METHOD.md](EXPERIMENTAL_METHOD.md) contains the reproducible experiment protocol and commands.
- [PROPOSAL_IMPLEMENTATION_STATUS.md](PROPOSAL_IMPLEMENTATION_STATUS.md) cross-checks the proposal against the implementation.
