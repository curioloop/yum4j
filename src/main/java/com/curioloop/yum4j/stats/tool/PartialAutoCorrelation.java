package com.curioloop.yum4j.stats.tool;

import com.curioloop.yum4j.linalg.Regressor;
import com.curioloop.yum4j.linalg.reg.OLS;
import com.curioloop.yum4j.math.Double2;
import com.curioloop.yum4j.stats.NormalDistribution;
import com.curioloop.yum4j.math.VectorOps;

/**
 * Partial autocorrelation function (PACF) estimator.
 *
 * <p>Mirrors {@code statsmodels.tsa.stattools.pacf} with four estimation methods:</p>
 * <ul>
 *   <li>{@link Method#YW_ADJUSTED} — Yule-Walker with unbiased autocovariance (default)</li>
 *   <li>{@link Method#YW_MLE} — Yule-Walker with biased autocovariance</li>
 *   <li>{@link Method#LD_ADJUSTED} — Levinson-Durbin with unbiased autocovariance</li>
 *   <li>{@link Method#LD_BIASED} — Levinson-Durbin with biased autocovariance</li>
 *   <li>{@link Method#BURG} — Burg's method (minimum-variance estimator)</li>
 *   <li>{@link Method#OLS} — OLS regression on lagged values (efficient)</li>
 *   <li>{@link Method#OLS_INEFFICIENT} — OLS with a single common sample</li>
 * </ul>
 *
 * <p>Confidence intervals use the asymptotic standard error {@code 1/√n} for all lags ≥ 1.</p>
 *
 * @param pacf    partial autocorrelations for lags 0, 1, …, nLags (length {@code nLags+1})
 * @param confInt confidence intervals as {@link Double2}{@code [nLags+1]} where each element
 *                carries {@code (lower, upper)} bounds; {@code null} if not requested
 */
public record PartialAutoCorrelation(double[] pacf, Double2[] confInt) {

    /** PACF estimation method. */
    public enum Method {
        /** Yule-Walker with adjusted (unbiased) autocovariance denominator. */
        YW_ADJUSTED,
        /** Yule-Walker with MLE (biased) autocovariance denominator. */
        YW_MLE,
        /** Levinson-Durbin with adjusted autocovariance. */
        LD_ADJUSTED,
        /** Levinson-Durbin with biased autocovariance. */
        LD_BIASED,
        /** Burg's partial autocorrelation estimator. */
        BURG,
        /** OLS regression on lags (efficient: uses nobs-lag observations per lag). */
        OLS,
        /** OLS regression on lags (inefficient: uses a single common sample). */
        OLS_INEFFICIENT
    }

    /**
     * Computes the PACF.
     *
     * @param x      input time series
     * @param nLags  number of lags; {@code null} → {@code min(10·log₁₀(n), n/2 - 1)}
     * @param method estimation method
     * @param alpha  confidence level (e.g. {@code 0.05} for 95 %); {@code null} to skip
     * @return PACF result
     */
    public static PartialAutoCorrelation compute(double[] x, Integer nLags, Method method, Double alpha) {
        int nobs = x.length;
        if (nobs < 4) throw new IllegalArgumentException("series too short for PACF");

        if (nLags == null) {
            nLags = Math.min((int)(10 * Math.log10(nobs)), nobs / 2 - 1);
        }
        nLags = Math.max(nLags, 1);
        if (nLags > nobs / 2) {
            throw new IllegalArgumentException(
                "nLags must be <= nobs/2 (" + (nobs / 2) + ")");
        }

        double[] pacf = switch (method) {
            case YW_ADJUSTED    -> pacfYuleWalker(x, nLags, true);
            case YW_MLE         -> pacfYuleWalker(x, nLags, false);
            case LD_ADJUSTED    -> pacfLevinsonDurbin(x, nLags, true);
            case LD_BIASED      -> pacfLevinsonDurbin(x, nLags, false);
            case BURG           -> pacfBurg(x, nLags);
            case OLS            -> pacfOls(x, nLags, true, false);
            case OLS_INEFFICIENT -> pacfOls(x, nLags, false, false);
        };

        Double2[] confInt = null;
        if (alpha != null) {
            double interval = new NormalDistribution().quantile(1.0 - alpha / 2.0)
                    / Math.sqrt(nobs);
            confInt = new Double2[pacf.length];
            confInt[0] = Double2.bound(pacf[0], pacf[0]); // lag-0 has no uncertainty
            for (int i = 1; i < pacf.length; i++) {
                confInt[i] = Double2.bound(pacf[i] - interval, pacf[i] + interval);
            }
        }
        return new PartialAutoCorrelation(pacf, confInt);
    }

    /** Convenience overload: Yule-Walker adjusted, no confidence intervals. */
    public static double[] compute(double[] x, int nLags) {
        return compute(x, nLags, Method.YW_ADJUSTED, null).pacf();
    }

    // ── Yule-Walker ───────────────────────────────────────────────────

    /**
     * Solves the Yule-Walker equations via a single Levinson-Durbin recursion.
     * The PACF at each lag is the reflection coefficient from that recursion.
     *
     * <p>Computes the full acovf once up to {@code nLags}; the recursion then
     * yields all lag PACF values in one pass.</p>
     */
    private static double[] pacfYuleWalker(double[] x, int nLags, boolean adjusted) {
        double[] acovFull = AutoCovariance.compute(x, adjusted, true, false,
                MissingPolicy.NONE, nLags);
        return LevinsonDurbin.compute(acovFull, nLags, true).pacf();
    }

    // ── Levinson-Durbin ───────────────────────────────────────────────

    private static double[] pacfLevinsonDurbin(double[] x, int nLags, boolean adjusted) {
        double[] acov = AutoCovariance.compute(x, adjusted, true, false,
                MissingPolicy.NONE, nLags);
        return LevinsonDurbin.compute(acov, nLags, true).pacf();
    }

    // ── Burg ──────────────────────────────────────────────────────────

    /**
     * Burg's partial autocorrelation estimator.
     *
     * <p>Minimises the sum of forward and backward prediction errors at each order.
     * Mirrors {@code statsmodels.tsa.stattools.pacf_burg}.</p>
     */
    static double[] pacfBurg(double[] x, int nLags) {
        int nobs = x.length;
        int p = Math.min(nLags, nobs - 1);
        double[] d    = new double[p + 1];
        double[] pacf = new double[p + 1];
        double[] u    = new double[nobs];
        double[] v    = new double[nobs];

        // Demean directly into u in reverse; v is an identical copy.
        // Avoids a separate xd buffer since xd is only used for d[0] = 2·Σxd²
        // and that equals 2·Σu² (reversal doesn't change the sum of squares).
        double mean = VectorOps.mean(x);
        for (int i = 0; i < nobs; i++) {
            double val = x[nobs - 1 - i] - mean;
            u[i] = val;
            v[i] = val;
        }

        d[0] = 2.0 * VectorOps.dot(u, 0, u, 0, nobs);
        d[1] = VectorOps.dot(u, 0, u, 0, nobs - 1) + VectorOps.dot(v, 1, v, 1, nobs - 1);
        pacf[1] = 2.0 / d[1] * VectorOps.dot(v, 1, u, 0, nobs - 1);

        // Use buffer-swap instead of System.arraycopy to avoid O(n) copies per iteration
        double[] tmpU = new double[nobs];
        double[] tmpV = new double[nobs];

        for (int i = 1; i < p; i++) {
            // Compute new u/v into tmp buffers
            for (int j = 1; j < nobs; j++) {
                tmpU[j] = u[j - 1] - pacf[i] * v[j];
                tmpV[j] = v[j]     - pacf[i] * u[j - 1];
            }
            // Swap: u ↔ tmpU, v ↔ tmpV
            double[] swp;
            swp = u; u = tmpU; tmpU = swp;
            swp = v; v = tmpV; tmpV = swp;

            d[i + 1] = (1.0 - pacf[i] * pacf[i]) * d[i]
                    - v[i] * v[i] - u[nobs - 1] * u[nobs - 1];
            pacf[i + 1] = 2.0 / d[i + 1] * VectorOps.dot(v, i + 1, u, i, nobs - i - 1);
        }
        pacf[0] = 1.0;
        return pacf;
    }

    // ── OLS ───────────────────────────────────────────────────────────

    /**
     * OLS-based PACF: regresses x[t] on x[t-1], …, x[t-k] for each lag k.
     * The PACF at lag k is the coefficient on x[t-k].
     *
     * <p>Mirrors {@code statsmodels.tsa.stattools.pacf_ols}.</p>
     */
    private static double[] pacfOls(double[] x, int nLags, boolean efficient, boolean adjusted) {
        int nobs = x.length;
        double[] pacf = new double[nLags + 1];
        pacf[0] = 1.0;

        if (efficient) {
            // For each lag k, use observations [k..nobs-1] with regressors [0..k-1]
            // Regressor expects row-major X: X[row * k + col]
            // Reused y/X buffers are refilled each iteration because OLS overwrites X.
            double[] y = new double[nobs - 1];
            double[] X = new double[(nobs - 1) * (nLags + 1)];
            OLS.Pool pool = new OLS.Pool();
            if (nLags > 0) pool.ensure(nobs - 1, nLags + 1,
                Regressor.Opts.QR, Regressor.Opts.HAS_CONST, Regressor.Opts.PARAMS);
            for (int k = 1; k <= nLags; k++) {
                int n = nobs - k;
                int cols = k + 1; // constant + k lags
                System.arraycopy(x, k, y, 0, n);
                for (int t = 0; t < n; t++) {
                    X[t * cols] = 1.0; // constant
                    for (int j = 1; j <= k; j++) {
                        X[t * cols + j] = x[k + t - j];
                    }
                }
                var ols = Regressor.ols(y, X, n, cols, pool,
                    Regressor.Opts.QR, Regressor.Opts.HAS_CONST, Regressor.Opts.PARAMS);
                double[] params = ols.params();
                pacf[k] = params[k]; // last lag coefficient
            }
        } else {
            // Demean once, use nobs - nLags observations for all lags
            double[] xd = x.clone();
            VectorOps.addConst(-VectorOps.mean(x), xd);

            int n = nobs - nLags;
            double[] yCopy = new double[n];
            double[] Xk = new double[n * nLags];
            OLS.Pool pool = new OLS.Pool();
            if (nLags > 0) pool.ensure(n, nLags,
                Regressor.Opts.QR, Regressor.Opts.NO_CONST, Regressor.Opts.PARAMS);
            for (int k = 1; k <= nLags; k++) {
                // Build lag matrix for lags 1..k, row-major: X[row * k + col]
                for (int t = 0; t < n; t++) {
                    for (int j = 1; j <= k; j++) {
                        Xk[t * k + (j - 1)] = xd[nLags + t - j];
                    }
                }
                System.arraycopy(xd, nLags, yCopy, 0, n);
                var ols = Regressor.ols(yCopy, Xk, n, k, pool,
                    Regressor.Opts.QR, Regressor.Opts.NO_CONST, Regressor.Opts.PARAMS);
                double[] params = ols.params();
                pacf[k] = params[k - 1];
            }
        }
        return pacf;
    }

}
