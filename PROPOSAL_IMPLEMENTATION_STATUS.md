# OptionMC Proposal and Implementation Status

This document converts `B1_G10_OptionMC_Proposal (1).pptx` into Markdown and verifies the proposal against the current repository implementation. The original code baseline was commit `1b53ffe`; the implementation results below describe the completed working tree verified on 22 September 2026.

## Project identification

- Course: CSE-402
- Project: Variance Reduction Techniques for Monte Carlo Option Pricing
- Section: B
- Subsection: B1
- Group: 10

### Group members

| Name | Student ID |
| --- | --- |
| Himel Sutradhar | 2105073 |
| Suprio Paul | 2105085 |
| Nafis Hussain | 2105086 |
| Zahid Al Hasan | 2105087 |
| Nayeem Uz Zaman | 2105090 |

## Base paper overview

The proposal names the following base paper:

> S. H. S. Herho, S. N. Kaban, and C. Nugraha, “OptionMC: Monte Carlo Pricing of European Options,” *International Journal of Data Science*, vol. 6, no. 2, pp. 70–84, 2025.

Domain: computational finance and numerical methods for derivative pricing.

According to the proposal, the paper:

- Prices European options with Monte Carlo simulation.
- Implements standard Monte Carlo and antithetic variates.
- Benchmarks simulated prices against the exact Black-Scholes formula.
- Studies convergence and sensitivity to volatility, time to expiry, and moneyness.

### Limitations identified in the proposal

- Plain Monte Carlo converges slowly because its error decreases at approximately \(1/\sqrt{N}\).
- The work covers European options only.
- It assumes geometric Brownian motion with constant volatility.
- It implements only one variance-reduction method: antithetic variates.

## Problem statement and proposed extension

The project asks whether additional variance-reduction methods can improve plain Monte Carlo. Halving standard Monte Carlo error generally requires approximately four times as many simulations.

The proposed comparison includes:

1. **Antithetic variates:** pair each normal draw with its negative so opposing outcomes cancel part of the sampling noise.
2. **Control variates:** correct option payoffs with a simulated quantity whose expectation is known.
3. **Stratified sampling:** divide the uniform domain into strata and sample from every stratum.
4. **Quasi-Monte Carlo:** use low-discrepancy Sobol or Halton points for more even domain coverage.

The implemented Sobol and Halton methods use scrambling. They are therefore randomized quasi-Monte Carlo methods. Fixed seeds preserve reproducibility.

## Expected methodology

The proposal defines this pipeline:

1. Simulate stock prices under geometric Brownian motion.
2. Apply the four variance-reduction techniques.
3. Calculate discounted option payoffs and validate prices against Black-Scholes.
4. Compare convergence, standard error, variance, and runtime.

### Evaluation metrics

- Pricing error relative to Black-Scholes as the number of paths increases.
- Standard error or estimator variance at a fixed path count.
- Runtime and the resulting accuracy-versus-cost trade-off.

## Cross-verification matrix

| Proposal requirement | Current status | Implementation evidence |
| --- | --- | --- |
| Isolated project setup | Implemented | [`.gitignore`](.gitignore) excludes `.venv`; dependencies are declared in [`pyproject.toml`](pyproject.toml), and the supported setuptools build backend is configured. |
| Standard-normal sampling | Implemented | [`StandardNormalSampler`](optionmc/samplers.py) uses a seeded NumPy generator. |
| Sobol sampling | Implemented | [`SobolSampler`](optionmc/samplers.py) produces scrambled low-discrepancy points and maps them through the normal inverse CDF. |
| Halton sampling | Implemented | [`HaltonSampler`](optionmc/samplers.py) provides the equivalent Halton workflow. |
| Geometric Brownian motion | Implemented | [`GeometricBrownianMotion`](optionmc/models.py) uses the risk-neutral drift and volatility-scaled diffusion term. |
| Black-Scholes benchmark | Implemented | [`BlackScholesAnalytical`](optionmc/models.py) prices calls and puts, including expiry and zero-volatility cases. |
| Discounted call and put payoffs | Implemented | [`OptionPricing`](optionmc/pricing.py) calculates and discounts both payoff types. |
| Standard Monte Carlo | Implemented | `standard_mc()` returns price, uncertainty, runtime, and variance statistics. |
| Antithetic variates | Implemented | The engine averages paired payoffs and estimates uncertainty from independent pair means. |
| Control variates | Implemented | Discounted terminal stock is used as the control with known expectation \(S_0\); beta is estimated from sample covariance. |
| Stratified sampling | Implemented | Uniform samples cover every equal-width stratum; the estimator combines within-stratum variance. |
| Quasi-Monte Carlo | Implemented | Sobol and Halton pricing use independent randomized scrambles to estimate QMC standard error. |
| Confidence intervals | Implemented | All pricing methods return normal-approximation 95% confidence intervals. |
| Convergence analysis | Implemented | [`convergence_analysis`](optionmc/analytics.py) evaluates prices, errors, uncertainty, variance, and runtime across path counts. |
| Relative error and VRR | Implemented | Relative error and estimator variance-reduction ratio functions include zero-denominator handling. |
| Runtime efficiency | Implemented | Efficiency compares error-squared runtime products. |
| Volatility sensitivity | Implemented | [`parameter_sensitivity`](optionmc/analytics.py) and [`run_sensitivity.py`](scripts/run_sensitivity.py) compare MC and analytical prices. |
| Time-to-expiry sensitivity | Implemented | The same sensitivity pipeline varies maturity. |
| Moneyness analysis | Implemented | [`moneyness_analysis`](optionmc/analytics.py) varies strike and reports \(K/S_0\), prices, errors, and uncertainty. |
| Comparison visualizations | Implemented | [`visualization.py`](optionmc/visualization.py) contains seven tested plotting functions. |
| End-to-end comparison | Implemented | [`run_comparison.py`](scripts/run_comparison.py) generates summary metrics and three comparison plots. |
| Repeated experimental evaluation | Implemented | [`experiments.py`](optionmc/experiments.py) runs and aggregates independent seeds/scrambles; [`run_repeated_experiments.py`](scripts/run_repeated_experiments.py) persists raw data and metadata. |
| Automated tests | Implemented | 51 tests cover samplers, formulas, pricing methods, analytics, repeated experiments, validation, and plots. |

## Numerical implementation

### Geometric Brownian motion

Terminal risk-neutral prices use:

\[
S_T = S_0 \exp\left((r-\tfrac{1}{2}\sigma^2)T + \sigma\sqrt{T}Z\right).
\]

The implementation generates terminal values rather than full time-stepped paths. Terminal values are sufficient for the European call and put payoffs in this project.

### Black-Scholes validation

For the standard parameters

\[
S_0=100,\quad K=100,\quad r=0.05,\quad \sigma=0.2,\quad T=1,
\]

the analytical values are:

- Call: `10.4505835722`
- Put: `5.5735260223`

Tests verify these values, put-call parity, payoff bounds, expiry values, and zero-volatility behavior.

### Uncertainty estimates

- Standard and control-variate methods use the sample variance of discounted estimator observations.
- Antithetic pricing treats each pair average as one independent estimator observation.
- Stratified pricing combines within-stratum variances using equal stratum weights.
- QMC pricing repeats independent scrambles and estimates uncertainty from the replicate means.

These distinctions avoid reporting an ordinary independent-sample error estimate for estimators whose observations are deliberately dependent.

## Verified comparison result

The end-to-end comparison ran successfully with 100,000 paths and seed 42. Runtime depends on the machine, so the stable comparison measures are price and standard error.

| Method | Estimated call price | Standard error | Variance-reduction ratio |
| --- | ---: | ---: | ---: |
| Standard | 10.420541 | 0.046770 | 1.00 |
| Antithetic | 10.467314 | 0.033077 | 2.00 |
| Control variate | 10.466844 | 0.017875 | 6.85 |
| Stratified | 10.453145 | 0.013036 | 12.87 |
| Scrambled Sobol QMC | 10.450934 | 0.000647 | 5220.50 |

The variance-reduction ratio in this legacy quick comparison comes from a single run. The final report therefore uses the repeated-run experiment described below for formal empirical conclusions.

## Repeated final-report experiment

The completed study runs all six estimators for calls and puts over seven path budgets with 30 independent outer seeds or randomized-scramble sets. It saves 2,520 raw comparison rows, 84 aggregate rows, 4,200 repeated sensitivity rows, 140 sensitivity aggregates, and complete environment metadata.

At 100,000 paths, scrambled Sobol QMC achieved the lowest RMSE for both the call (`0.000739`) and put (`0.000305`). Control variates and stratification materially improved conventional Monte Carlo. Antithetic sampling modestly improved the call but did not improve the put across these repetitions, showing why variance reduction must be measured for each payoff rather than assumed.

See [FINAL_PROJECT_REPORT.md](FINAL_PROJECT_REPORT.md) for results and [EXPERIMENTAL_METHOD.md](EXPERIMENTAL_METHOD.md) for the protocol and reproduction commands.

## Generated analyses

Running the scripts creates these ignored artifacts:

- `artifacts/method_comparison.png`
- `artifacts/error_convergence.png`
- `artifacts/efficiency_comparison.png`
- `artifacts/volatility_sensitivity.png`
- `artifacts/maturity_sensitivity.png`
- `artifacts/moneyness_sensitivity.png`

The comparison script prints the final prices, standard errors, variance-reduction ratios, and efficiency ratios. The sensitivity script compares Monte Carlo values directly with Black-Scholes values across volatility, maturity, and strike ranges.

The repeated workflow additionally creates raw and aggregate CSV files, environment metadata, call/put convergence and variance-reduction figures, accuracy-runtime and confidence-coverage figures, repeated sensitivity figures, distribution validation, and final comparison tables under `artifacts/data/` and `artifacts/report/`.

## Test and environment verification

The project was installed as an editable package into `.venv` with the optional development dependencies.

Verified command:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Verified result:

```text
51 passed
```

The quick and repeated experiment scripts completed successfully. The final report figures were inspected for readable labels, legends, scales, and data alignment.

## Remaining scope limitations

These are model boundaries rather than incomplete proposal requirements:

- European calls and puts only
- Constant risk-free rate and volatility
- No dividends
- Lognormal geometric Brownian motion
- No stochastic volatility, jumps, transaction costs, or early exercise
- Normal-approximation confidence intervals
- QMC performance conclusions depend on the number of independent scrambles

The current implementation satisfies the methodology and comparison scope described in the proposal while retaining the financial assumptions stated there.
