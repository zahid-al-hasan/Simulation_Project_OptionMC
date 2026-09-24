# Base-paper insights and OptionMC project progress

This note extracts the material in `Sandy.pdf` that matters to the four-slide `B1_G10_OptionMC_Proposal (1).pptx`, then checks the proposed work against the current code. It is a project-focused summary, not a transcription of the paper. Status checked on 24 September 2026 against the working tree; the 52 automated tests passed.

## Sources and an important title correction

- **Base paper:** Sandy H. S. Herho, Siti N. Kaban, and Cahya Nugraha, “OptionMC: A Python Package for Monte Carlo Pricing of European Options,” *International Journal of Data Science* 6(2), 70–84 (2025). Source: `C:\Users\USER\Downloads\Sandy.pdf`.
- **Proposal:** “Variance Reduction Techniques for Monte Carlo Option Pricing,” CSE-402, Section B, Subsection B1, Group 10. The original PowerPoint file is no longer at its previously supplied Downloads path, so the proposal details below are cross-checked against the existing [slide transcription and status note](PROPOSAL_IMPLEMENTATION_STATUS.md), not a fresh read of the deck.

The proposal slide abbreviates the paper title to “OptionMC: Monte Carlo Pricing of European Options.” Use the **full title above** in the final report and bibliography (paper PDF p. 1, printed p. 70).

## What the paper contributes to our project

1. **A reproducible pricing baseline.** Under risk-neutral geometric Brownian motion, draw independent standard normals and calculate terminal prices as

   \[
   S_T=S_0\exp\!\left[(r-\sigma^2/2)T+\sigma\sqrt{T}Z\right],\quad Z\sim N(0,1).
   \]

   Discount each European call payoff `max(S_T - K, 0)` or put payoff `max(K - S_T, 0)` by `exp(-rT)`, then average. The paper presents this method in Section 2.2.1 (PDF pp. 6–7, printed pp. 75–76). Since these payoffs depend only on maturity, simulating **terminal prices** is sufficient; full time-stepped paths are not required for this project.

2. **An analytical truth value.** The paper compares MC prices with closed-form Black–Scholes call and put prices (Section 2.1, PDF pp. 4–5). Its main demonstration uses `S0=100`, `K=100`, `r=0.05`, `sigma=0.20`, `T=1` year. The call benchmark is about **10.45** (Figure 1, PDF p. 10); our analytical implementation returns **10.4505835722**. Compare simulated prices against that benchmark using absolute/relative error, not just whether a plot “looks close.”

3. **Why variance reduction matters.** Ordinary MC standard error is proportional to `1/sqrt(N)`: halving it usually needs about four times as many independent paths. Report estimated standard error and confidence intervals alongside each point estimate (Section 2.2.1–2.2.2, PDF pp. 7–8). A small observed error from one run is not proof of a better method.

4. **The paper's antithetic benchmark.** For each normal draw `Z`, price both `Z` and `-Z` and average the paired payoffs. The pair average is the independent statistical observation used to estimate uncertainty. The paper defines the variance-reduction ratio as `Var(standard estimator) / Var(antithetic estimator)` (PDF pp. 7–8). It reports typical antithetic VRR of roughly 2–5 for its setting, but that is a paper-specific empirical claim, not a guaranteed result for every option or sample size.

5. **Which experiments matter.** The paper presents convergence against Black–Scholes with 95% confidence bands (Figure 1), sensitivity to volatility, maturity, strike, and moneyness for calls and puts (Figure 2), terminal-price and payoff distributions (Figure 3), and standard-vs-antithetic price/error convergence (Figure 4). See PDF pp. 9–13. These give useful **validation patterns** for our extension rather than target values that every random seed must reproduce exactly.

6. **Model boundaries.** The paper assumes European exercise, constant volatility and interest rate, lognormal GBM prices, and the usual idealized Black–Scholes market. It notes that smiles, jumps, and early exercise require richer models (PDF pp. 2 and 5). Our proposal extends **sampling/estimation efficiency**, not the market model or option types; these limitations remain.

## The proposal's contribution beyond the paper

The paper implements **standard MC plus antithetic variates**. The proposal keeps those as baselines and adds:

| Added method | Core idea | Comparison question |
| --- | --- | --- |
| Control variates | Use discounted terminal stock, whose expected value is `S0`, to correct option-payoff estimates | How much estimator variance falls after accounting for correlation with stock value? |
| Stratified sampling | Force samples into every equal-width region of the unit interval before inverse-normal mapping | Does broad coverage reduce error at a fixed path budget? |
| Sobol and Halton QMC | Replace independent uniform draws with scrambled low-discrepancy points | Does more even coverage improve accuracy per unit time? |

The proposal's expected pipeline is **simulate, estimate, validate, compare**. It calls for price error versus Black–Scholes over increasing `N`, standard error or variance at the same `N`, and runtime/efficiency. This is the key difference between reproducing the paper and delivering the proposed extension.

## Step-by-step: proposed work versus implementation

| Step | What the proposal requires | What is in the repository now | Status |
| --- | --- | --- | --- |
| 1. Set the financial model | Use risk-neutral GBM with European call/put payoffs | [`models.py`](optionmc/models.py) generates correctly volatility-scaled terminal prices; [`pricing.py`](optionmc/pricing.py) calculates discounted call and put payoffs | Implemented |
| 2. Set an exact benchmark | Compute Black–Scholes prices for validation | `BlackScholesAnalytical` handles calls, puts, expiry, and zero volatility; known-value and parity tests exist | Implemented |
| 3. Build the plain-MC baseline | Draw independent normals, estimate price, standard error, and CI | `StandardNormalSampler` and `OptionPricing.standard_mc()` do this | Implemented |
| 4. Reproduce the paper's variance reduction | Pair `Z` with `-Z` and compare fairly at the same total path count | `AntitheticVariates` and `antithetic_mc()` calculate pair-average prices and pair-based standard errors | Implemented |
| 5. Add control variates | Correct discounted payoffs using a known-mean control | `ControlVariates` estimates beta from covariance and uses discounted terminal stock as the control | Implemented |
| 6. Add stratification | Sample every stratum and calculate estimator uncertainty | `StratifiedSampling` and `stratified_mc()` provide equal-stratum sampling and a within-stratum variance estimate | Implemented for the experiment settings |
| 7. Add quasi-MC | Price using Sobol and Halton sequences | Both samplers and `quasi_mc()` are present; QMC error uses independent scrambles | Implemented |
| 8. Test convergence and sensitivity | Sweep path counts and market/option parameters; compare to Black–Scholes | [`analytics.py`](optionmc/analytics.py) provides convergence, relative error, moneyness, and parameter sensitivity; the scripts run these analyses | Implemented, with scope notes below |
| 9. Compare methods and present results | Plot price/error convergence and report variance, runtime, and efficiency | The quick scripts remain available; [`run_repeated_experiments.py`](scripts/run_repeated_experiments.py) and [`generate_report_figures.py`](scripts/generate_report_figures.py) now produce the repeated call/put evidence, CSV data, metadata, tables, and final figures | Implemented with 30-run evidence |
| 10. Validate the software | Run automated correctness and behavior checks | 52 tests cover formulas, distributions, pricing, analytics, repeated experiments, simulation-backed paper-figure data, CSV output, and plotting (`python -m pytest -q -p no:cacheprovider`) | Implemented; 52 passed on 24 September 2026 |

## Experimental-rigor work completed for the final project report

The earlier review identified five reporting gaps. They have now been addressed as follows:

1. **Independent repetitions:** every method/budget/option configuration now uses 30 independent outer seeds or randomized QMC scramble sets.
2. **Robust timing:** methods are warmed up, timed repeatedly, and summarized with medians and quartiles on the same machine.
3. **Empirical convergence:** RMSE across repetitions is plotted on log-log axes and fitted slopes are saved.
4. **Call and put coverage:** the comparison and sensitivity studies now report both option types.
5. **Persistent data:** raw CSV rows, aggregate CSVs, environment metadata, final tables, and figures are written below `artifacts/`.

The implementation and final conclusions are documented in [FINAL_PROJECT_REPORT.md](FINAL_PROJECT_REPORT.md). The exact run design is in [EXPERIMENTAL_METHOD.md](EXPERIMENTAL_METHOD.md).

These are reporting/experimental gaps, not a request to expand into American options, stochastic volatility, or jumps. The paper suggests such models as future work (PDF p. 13), but they are outside the four-slide proposal.

## Completed practical sequence

1. The paper's baseline parameters and equal terminal-price budgets were fixed in the experiment CLI.
2. Every method was run for calls and puts over seven path counts and 30 independent outer seeds/scramble sets.
3. Bias, RMSE, empirical variance, confidence-interval coverage, runtime, VRR, efficiency, and convergence slopes were aggregated.
4. Report-ready convergence, variance-reduction, accuracy-time, coverage, sensitivity, and distribution figures were generated. All four paper figures now have simulation/formula-driven counterparts, and the four proposed techniques are compared together in one chart.
5. [FINAL_PROJECT_REPORT.md](FINAL_PROJECT_REPORT.md) now distinguishes the base paper from our measured findings and states all retained model limitations.

## Source-reading cautions

- The PDF's extracted mathematical glyphs are occasionally corrupted; the GBM, payoff, and variance formulas above use standard notation corroborated by the paper's surrounding explanation and current code.
- The paper's specific plotted prices and relative errors come from its own simulation runs. They should not be treated as deterministic acceptance tests for our independent implementation.
- The paper mentions additional future methods, including **importance sampling**. Our proposal selects **control variates, stratification, and Sobol/Halton QMC** instead; importance sampling is not a missing proposal requirement.
