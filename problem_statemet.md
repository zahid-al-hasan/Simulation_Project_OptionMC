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