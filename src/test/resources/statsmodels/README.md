# statsmodels Test Resources

This directory mirrors selected files from `reference/statsmodels/statsmodels` that are required by Java tests.

| Resource | Upstream source | Used by |
|---|---|---|
| `datasets/macrodata/macrodata.csv` | `statsmodels/datasets/macrodata/macrodata.csv` | Kalman statsmodels parity tests, `RegressorTest` |
| `datasets/sunspots/sunspots.csv` | `statsmodels/datasets/sunspots/sunspots.csv` | KPSS lag selection parity tests |
| `datasets/nile/nile.csv` | `statsmodels/datasets/nile/nile.csv` | KPSS lag selection parity tests |
| `datasets/randhie/randhie.csv` | `statsmodels/datasets/randhie/randhie.csv` | KPSS lag selection parity tests |
| `datasets/modechoice/modechoice.csv` | `statsmodels/datasets/modechoice/modechoice.csv` | KPSS lag selection parity tests |
| `tsa/vector_ar/tests/Matlab_results/test_coint.csv` | `statsmodels/tsa/vector_ar/tests/Matlab_results/test_coint.csv` | Johansen cointegration parity tests |
| `tsa/statespace/tests/results/*.csv` | `statsmodels/tsa/statespace/tests/results/*.csv` | Kalman filter and simulation smoother parity tests |
| `tsa/statespace/tests/results/results_sarimax.py` | `statsmodels/tsa/statespace/tests/results/results_sarimax.py` | SARIMAX statsmodels parity tests |
| `tsa/statespace/tests/results/results_sarimax_coverage.csv` | `statsmodels/tsa/statespace/tests/results/results_sarimax_coverage.csv` | SARIMAX statsmodels parity tests |

Keep paths aligned with the upstream statsmodels tree when adding new fixtures.