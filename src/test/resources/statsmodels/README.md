# statsmodels Test Resources

This directory mirrors selected files from `reference/statsmodels/statsmodels` that are required by Java tests.

| Resource | Upstream source | Used by |
|---|---|---|
| `datasets/macrodata/macrodata.csv` | `statsmodels/datasets/macrodata/macrodata.csv` | Kalman statsmodels parity tests, `RegressorTest` |
| `tsa/statespace/tests/results/*.csv` | `statsmodels/tsa/statespace/tests/results/*.csv` | Kalman filter and simulation smoother parity tests |
| `tsa/statespace/tests/results/results_sarimax.py` | `statsmodels/tsa/statespace/tests/results/results_sarimax.py` | SARIMAX statsmodels parity tests |
| `tsa/statespace/tests/results/results_sarimax_coverage.csv` | `statsmodels/tsa/statespace/tests/results/results_sarimax_coverage.csv` | SARIMAX statsmodels parity tests |

Keep paths aligned with the upstream statsmodels tree when adding new fixtures.