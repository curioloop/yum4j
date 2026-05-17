# Statsmodels Kalman Migration Ledger

This ledger tracks the yum4j migration status for relevant statsmodels
statespace Kalman tests. New numeric parity fixtures should live next to this
file and use the existing alignment manifest / CSV surface conventions where
possible.

## Status

Status categories:

- `covered-data-driven`: statsmodels-generated manifest / CSV fixtures drive the numeric parity assertions.
- `covered-by-public-varmax-api`: statsmodels-generated fixtures exercise the public `VARMAX` model API rather than only raw `KalmanSSM` matrices.
- `covered-hardcoded`: Java has representative or algebraic coverage, but not a statsmodels-generated fixture matrix yet.
- `partial-data-needed`: important coverage exists, but additional statsmodels-generated cases remain target work.
- `non-target`: behavior belongs to Python object lifecycle, pandas/index/table formatting, Cython internals, or high-level model APIs not exposed by yum4j.

| statsmodels source | status | yum4j coverage / target |
| --- | --- | --- |
| `test_kalman.py` | covered-data-driven | Core low-level filter likelihood, state, covariance, gain, missing-data, complex, and route surfaces are covered by the Kalman alignment manifest and CSVs. Python pickle/copy lifecycle checks are non-target. |
| `test_smoothing.py` | partial-data-needed | Core smoother, in-sample smoothed state autocovariance, and basic/revision news impact rows are data-driven. Remaining targets: wider missing / univariate / time-varying autocov-news matrices and out-of-sample autocovariance cases that are expressible by `KalmanSSM`. |
| `test_univariate.py` | covered-data-driven | Univariate/conventional route parity, missing patterns, singular fallback profiles, and observation transform coverage are represented by manifest cases and focused route tests. |
| `test_exact_diffuse_filtering.py` | partial-data-needed | Exact diffuse filter and smoother parity has data-driven coverage. Remaining targets are wider edge matrices such as common-level, restricted common-level, non-diagonal observation covariance, and irrelevant-state cases. |
| `test_collapsed.py` | covered-data-driven | Collapsed route parity, missing data fallback, smoother routes, and state autocovariance surfaces are covered by generated fixtures plus route smoke tests. |
| `test_chandrasekhar.py` | covered-data-driven | Chandrasekhar route parity, concentrated likelihood, univariate profiles, and unsupported-profile checks are covered. |
| `test_cfa_simulation_smoothing.py` | covered-data-driven | DFM all/partial/mixed missing, VARME, SARIMAXME, and UC-like posterior mean/cov surfaces are generated from statsmodels. Zero-variate draw semantics remain local Java behavior coverage. |
| `test_cfa_tvpvar.py` | covered-data-driven | TVP-VAR-like posterior mean/cov surfaces are generated from statsmodels. Inverse-Wishart / inverse-gamma checks stay model-specific non-target unless yum4j grows that API. |
| `test_simulation_smoothing.py` | covered-hardcoded | KFS simulation smoothing, intercept behavior, generated outputs, missing data, and zero-variate posterior semantics are covered by Java tests. Numeric draw parity is intentionally local because RNG stream compatibility is non-target. |
| `test_conserve_memory.py` | covered-hardcoded | Memory conservation likelihood, forecast availability, and prediction wrapper availability are covered by Java behavior tests. Exact Python memory flag behavior is non-target. |
| `test_prediction.py` | partial-data-needed | Basic predict / forecast / dynamic / memory behavior and information-set identities are covered. Remaining target: move SARIMAX concatenated prediction, `noPredicted`, exog/trend, and concentrated prediction numeric expectations into generated fixtures. |
| `test_representation.py` | partial-data-needed | Generic representation entry points and wrapper behavior are covered. Python slicing, pandas index, object lifecycle, and Cython checks are non-target; low-level time-varying matrix parity remains target work. |
| `test_initialization.py` | covered-hardcoded | Composite, nested, stationary, mixed, and invalid initialization matrices have representative Java coverage. Further expansion should be generator-backed if it becomes numeric parity rather than validation coverage. |
| `test_multivariate_switch_univariate.py` | covered-hardcoded | Full-route singular fallback and opt-in fallback telemetry are covered. Additional missing / initialization matrices should be added as generated route fixtures. |
| `test_weights.py` | partial-data-needed | `SmootherDiagnostics.computeSmoothedStateWeights` is now checked against generated statsmodels observation/state-intercept/prior weight rows for a bivariate fixture. Remaining target: SARIMAX exog/trend, univariate, singular/collapsed, and time-varying profiles. |
| `test_decompose.py` | partial-data-needed | `SmootherDiagnostics.getSmoothedDecomposition` is now checked against generated statsmodels data/intercept/prior contribution rows for a bivariate fixture. Remaining target: broader decomposition matrices from the statsmodels source parametrization. |
| `test_news.py` | partial-data-needed | Basic extended-sample and historical-revision news impacts are generated from statsmodels and checked against `SmootherDiagnostics.news`. Remaining target: grouped/mixed revisions and table/index formatting remains non-target. |
| `varmax.py` / VARMAX public surfaces | covered-by-public-varmax-api | `varmax_public_alignment.json` is generated by `tools/generate_statsmodels_varmax_fixtures.py` and validated through `StatsmodelsVARMAXAlignmentTest` using the public `VARMAX` API. Current data-driven cases cover VAR(1), VAR(2), VARX forecast with future exog, constant trend, diagonal/unstructured covariance, measurement error, VARMA(1,1), coefficient matrices, transform parity, filter likelihood/state/error/covariance surfaces, and public forecast means/covariances. Table formatting, pandas index plumbing, fixed-parameter UI, and statsmodels object lifecycle behavior remain non-target. |

## Non-Targets

- pandas date / PeriodIndex behavior.
- pickle, save, remove-data, and object lifecycle tests.
- Cython extension internals.
- DynamicFactorMQ and UnobservedComponents high-level model matrices until yum4j
  has matching public model families.
