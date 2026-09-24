# Figure data provenance

No plotted estimate is copied from the paper or manually chosen to match it.

| Output | Numerical source |
| --- | --- |
| Paper Figure 1 reproduction | Fresh `OptionPricing.standard_mc` runs for seed 42 at each configured path count; per-run confidence intervals come from simulated payoff variance. |
| Paper Figure 2 reproduction | `sensitivity_summary.csv`, aggregating 30 control-variate simulations at each grid value; dashed curves are independently evaluated Black-Scholes formulas. |
| Paper Figure 3 reproduction | 100,000 GBM terminal prices generated from seeded standard-normal draws; theoretical overlay is the GBM lognormal density. |
| Paper Figure 4 reproduction | Fresh standard and antithetic simulations for seed 42 at equal terminal-price budgets; relative errors use the computed Black-Scholes value. |
| Four-technique comparison | `experiment_summary.csv`, aggregating 30 outer seeds/scramble sets at 100,000 paths. QMC is represented by scrambled Sobol. |

Fixed values such as the parameter set, path-count grid, sample size, and random seed are experimental configuration, not fitted graph values.
