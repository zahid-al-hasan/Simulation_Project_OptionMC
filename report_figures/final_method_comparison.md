# Repeated experiment summary at 100,000 paths

Each entry summarizes 30 independent seeds or randomized scrambles.

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
