# Variance Reduction Techniques for Monte Carlo Option Pricing

## Abstract

This project reproduces the European option-pricing baseline in Herho, Kaban, and Nugraha's *OptionMC: A Python Package for Monte Carlo Pricing of European Options* and extends it with control variates, stratified sampling, and randomized Sobol and Halton quasi-Monte Carlo (QMC). Prices are validated against the Black-Scholes formula. A repeated experiment covering calls and puts, six estimators, seven path budgets, and 30 independent outer seeds measures bias, root-mean-square error (RMSE), estimator variance, confidence-interval coverage, and runtime. At 100,000 terminal-price evaluations, scrambled Sobol QMC produced the lowest RMSE for both the call (`0.000739`) and put (`0.000305`). Stratification and control variates also materially improved conventional Monte Carlo. Antithetic sampling provided only a modest call improvement and did not improve the put in this experiment. These results demonstrate that variance reduction is method- and payoff-dependent and that repeated runs are necessary for defensible comparisons.

## 1. Project objective

Plain Monte Carlo is flexible but converges slowly: its sampling error typically decreases in proportion to `N^(-1/2)`. Reducing the error by half therefore usually requires about four times as many paths. The project asks whether better sampling and estimation schemes can produce more accurate European option prices for the same terminal-price budget.

The base paper implements standard Monte Carlo and antithetic variates. Our proposed extension adds:

1. A discounted-stock control variate.
2. Equal-probability stratified sampling.
3. Scrambled Sobol and Halton QMC.
4. A reproducible, repeated comparison for calls and puts.

The extension concerns numerical efficiency. It does not replace the Black-Scholes/GBM market assumptions with a richer financial model.

## 2. Financial model and analytical benchmark

Under the risk-neutral geometric Brownian motion model, the terminal stock price is

```text
S_T = S_0 exp((r - sigma^2 / 2)T + sigma sqrt(T) Z),  Z ~ N(0,1).
```

European call and put values are estimated from discounted terminal payoffs:

```text
Call payoff = exp(-rT) max(S_T - K, 0)
Put payoff  = exp(-rT) max(K - S_T, 0)
```

Because these payoffs depend only on `S_T`, full time-stepped paths are unnecessary. For the baseline parameters `S0=100`, `K=100`, `r=0.05`, `sigma=0.20`, and `T=1`, the analytical Black-Scholes prices are:

- Call: `10.4505835722`
- Put: `5.5735260223`

These values provide the ground truth for error, RMSE, and interval-coverage calculations.

## 3. Implemented estimators

### 3.1 Standard Monte Carlo

The baseline draws independent standard normals, converts them to terminal GBM prices, discounts the option payoffs, and reports their sample mean. Its estimated standard error is the payoff sample standard deviation divided by `sqrt(N)`.

### 3.2 Antithetic variates

For each normal draw `Z`, the implementation also evaluates `-Z`. The two discounted payoffs are averaged, and each pair average is treated as one independent observation. The comparison counts both terminal prices, so `N/2` generated random draws still consume a total path budget of `N`.

### 3.3 Control variates

The discounted terminal stock `exp(-rT) S_T` has known risk-neutral expectation `S0`. The estimator adjusts each payoff by

```text
Y_adjusted = Y - beta (X - S0),
```

where `beta` is estimated from the sample covariance of the payoff `Y` and control `X`. This reduces variance when the payoff and discounted stock are strongly correlated.

### 3.4 Stratified sampling

The unit interval is divided into ten equal-probability strata, and samples are drawn within every stratum before inverse-normal transformation. The estimator averages the stratum means and calculates uncertainty from the within-stratum variances. This guarantees broader coverage of the input distribution than unrestricted random sampling.

### 3.5 Randomized quasi-Monte Carlo

Sobol and Halton sequences distribute points more evenly than independent pseudorandom draws. Scrambling randomizes each sequence so independent replicated estimates and standard errors can be obtained. Each outer run splits its total budget across eight independent scrambles. The implementation derives non-overlapping child scramble seeds from the outer seed using `SeedSequence.spawn`, making repeated QMC experiments reproducible and independent at the outer-run level.

## 4. Software implementation

The repository separates the work into testable modules:

- `optionmc/models.py`: risk-neutral GBM and analytical Black-Scholes prices.
- `optionmc/samplers.py`: pseudorandom, Sobol, and Halton normal samplers.
- `optionmc/variance_reduction.py`: antithetic, control-variate, and stratification components.
- `optionmc/pricing.py`: call/put pricing, uncertainty estimates, and timing.
- `optionmc/analytics.py`: convergence, error, efficiency, moneyness, and sensitivity analysis.
- `optionmc/experiments.py`: repeated runs, aggregation, empirical metrics, and CSV persistence.
- `scripts/run_repeated_experiments.py`: reproducible experiment command-line interface.
- `scripts/generate_report_figures.py`: report tables and visualizations.

The automated suite contains 51 tests covering input validation, distributions, formulas, payoff pricing, variance-reduction behavior, QMC reproducibility, analytics, experiment aggregation, CSV output, and plotting.

## 5. Experimental design

The principal comparison uses both calls and puts, all six methods, seven budgets from 100 to 100,000 terminal-price evaluations, and 30 outer seeds per configuration. This gives 2,520 individual pricing runs and 84 aggregated configurations. The same outer seeds are reused across methods for a paired comparison. Method warm-up occurs before timing, and runtime is summarized by the median rather than a single observation.

The study reports bias, mean absolute error, RMSE, empirical variance across independent runs, variance-reduction ratio (VRR), 95% interval coverage, median runtime, and log-log RMSE convergence slope. A separate control-variate sensitivity study performs 4,200 pricing runs over volatility, maturity, and strike/moneyness grids for both option types.

The complete protocol and reproduction commands are in [EXPERIMENTAL_METHOD.md](EXPERIMENTAL_METHOD.md).

## 6. Results

### 6.1 Final comparison at 100,000 paths

Each entry below summarizes 30 independent outer seeds or randomized-scramble sets. Empirical VRR is the standard-MC variance divided by the method's variance across those 30 price estimates.

| Option | Method | Mean price | RMSE | Empirical VRR | Median runtime (s) | 95% CI coverage | RMSE slope |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Call | Standard MC | 10.448868 | 0.048769 | 1.00 | 0.001986 | 93.3% | -0.471 |
| Call | Antithetic | 10.450472 | 0.040251 | 1.47 | 0.001197 | 83.3% | -0.456 |
| Call | Control variate | 10.448685 | 0.013339 | 13.63 | 0.003212 | 96.7% | -0.491 |
| Call | Stratified | 10.449920 | 0.012530 | 15.17 | 0.005013 | 96.7% | -0.495 |
| Call | Sobol QMC | 10.450442 | 0.000739 | 4513.00 | 0.005039 | 90.0% | -0.937 |
| Call | Halton QMC | 10.450135 | 0.001896 | 699.65 | 0.028239 | 90.0% | -0.882 |
| Put | Standard MC | 5.571610 | 0.022235 | 1.00 | 0.003360 | 100.0% | -0.491 |
| Put | Antithetic | 5.573767 | 0.024258 | 0.83 | 0.002116 | 86.7% | -0.465 |
| Put | Control variate | 5.571627 | 0.013339 | 2.82 | 0.005367 | 96.7% | -0.491 |
| Put | Stratified | 5.572373 | 0.006545 | 11.82 | 0.008806 | 96.7% | -0.486 |
| Put | Sobol QMC | 5.573356 | 0.000305 | 7679.25 | 0.010122 | 93.3% | -0.967 |
| Put | Halton QMC | 5.573655 | 0.000857 | 683.03 | 0.033170 | 86.7% | -0.911 |

The full-precision machine-readable table is in `artifacts/report/final_method_comparison.csv`.

### 6.2 Accuracy and convergence

Standard MC slopes (`-0.471` for calls and `-0.491` for puts) agree with the theoretical `-0.5` rate. The control-variate and stratified estimators preserve approximately the same asymptotic rate but lower the error constant. In contrast, the observed Sobol and Halton slopes are close to `-1` in this one-dimensional problem, giving a much faster empirical decrease in RMSE over the tested range.

![Call RMSE convergence](artifacts/report/call_rmse_convergence.png)

![Put RMSE convergence](artifacts/report/put_rmse_convergence.png)

At 100,000 paths, Sobol QMC reduced call RMSE by roughly 66 times and put RMSE by roughly 73 times relative to standard MC. Halton was also highly accurate but slower than Sobol in this implementation. Control variates and stratification provided substantial improvements with conventional random sampling.

![Accuracy and runtime](artifacts/report/accuracy_runtime_tradeoff.png)

### 6.3 Variance reduction is payoff-dependent

For the call, stratification and the control variate achieved empirical VRRs of `15.17` and `13.63`. For the put, their VRRs were `11.82` and `2.82`. Antithetic sampling achieved a modest call VRR of `1.47`, but its put VRR was `0.83`; a ratio below one means its estimates varied more than standard MC across these repetitions. Antithetic pairing is therefore not automatically beneficial for every payoff under a fixed budget.

![Call variance reduction](artifacts/report/call_variance_reduction.png)

![Put variance reduction](artifacts/report/put_variance_reduction.png)

The very large QMC VRRs reflect this smooth, one-dimensional integration problem. They should not be treated as universal multipliers for high-dimensional or path-dependent derivatives.

### 6.4 Confidence intervals

Control-variate and stratified intervals covered the analytical value in 96.7% of the 30 repetitions for both option types. Other methods ranged from 83.3% to 100%. With only 30 repetitions, each run changes observed coverage by 3.3 percentage points, so these values are diagnostics rather than precise calibration estimates. QMC intervals in particular are based on eight randomized replicate means per outer run.

![Confidence-interval coverage](artifacts/report/confidence_interval_coverage.png)

### 6.5 Sensitivity and distribution validation

Repeated control-variate simulations follow the Black-Scholes curves over the tested volatility, maturity, and strike ranges for both calls and puts. Call values increase with volatility and generally with maturity; put behavior follows the corresponding discounted payoff economics. Varying strike also demonstrates the expected call/put changes across moneyness from `K/S0=0.70` to `1.30`.

![Volatility sensitivity](artifacts/report/volatility_sensitivity.png)

![Maturity sensitivity](artifacts/report/maturity_sensitivity.png)

![Moneyness sensitivity](artifacts/report/moneyness_sensitivity.png)

The simulated terminal-price and log-return distributions also match the GBM lognormal and normal theoretical shapes.

![Distribution validation](artifacts/report/distribution_validation.png)

## 7. Discussion

The experiments support three main conclusions. First, repeated-run evidence is materially more reliable than a single-seed comparison: it exposes method variability, convergence rates, runtime dispersion, and interval coverage. Second, control variates and stratification are reliable improvements for this problem, but their benefit depends on payoff structure. Third, randomized QMC is especially effective here because a European terminal payoff reduces to a one-dimensional integral. Sobol outperformed Halton in both accuracy and runtime in this implementation.

Runtime alone does not determine the best estimator. At 100,000 paths, Sobol required more time than standard MC but reduced RMSE by orders of magnitude, placing it on the strongest accuracy-cost trade-off. Halton's accuracy was strong, but its generation cost was the highest among the tested methods.

The experiment also corrects a common reporting problem: a confidence interval from one simulation describes uncertainty within that run, while the empirical variance and RMSE across independent outer runs directly measure repeated estimator performance. Both perspectives are useful and are retained in the saved data.

## 8. Limitations

- Only European calls and puts are priced.
- The market model assumes constant volatility and interest rate, no dividends, and lognormal GBM prices.
- There is no early exercise, stochastic volatility, jump process, calibration, or transaction-cost model.
- Normal-approximation confidence intervals are used.
- Thirty outer repetitions are enough for a useful comparison but not a high-precision coverage study.
- Timings are machine- and software-version-specific.
- QMC performance is demonstrated for a one-dimensional terminal-payoff problem and may degrade as effective dimension increases.
- Sensitivity experiments use the control-variate estimator; they validate price behavior but do not compare every method at every parameter value.

## 9. Conclusion

The completed implementation fulfills the proposal's numerical scope and strengthens it with a reproducible experiment layer. Standard MC, antithetic variates, control variates, stratified sampling, and scrambled Sobol/Halton QMC all support European calls and puts and are validated against Black-Scholes. The repeated experiment shows that Sobol QMC is the most accurate method for the selected one-dimensional benchmark, while stratification and control variates offer substantial conventional Monte Carlo variance reduction. Antithetic results demonstrate why method performance must be measured rather than assumed.

All raw runs, aggregates, metadata, tables, and figures can be regenerated from the documented commands. The remaining limitations concern the financial model and study breadth, not missing proposal features.

## References

1. S. H. S. Herho, S. N. Kaban, and C. Nugraha, "OptionMC: A Python Package for Monte Carlo Pricing of European Options," *International Journal of Data Science*, vol. 6, no. 2, pp. 70-84, 2025.
2. F. Black and M. Scholes, "The Pricing of Options and Corporate Liabilities," *Journal of Political Economy*, vol. 81, no. 3, pp. 637-654, 1973.
