package com.curioloop.yum4j.stats.test;

import com.curioloop.yum4j.linalg.Regressor;
import com.curioloop.yum4j.math.VectorOps;

/**
 * Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test for stationarity.
 *
 * <p>Tests the null hypothesis that a time series is level-stationary ({@link Regression#CONSTANT})
 * or trend-stationary ({@link Regression#TREND}) against the alternative of a unit root.</p>
 *
 * <h3>Test statistic</h3>
 * <pre>{@code
 * η = (1/T²) * Σ_{t=1}^{T} S_t²  /  σ̂²
 * }</pre>
 * where {@code S_t = Σ_{s=1}^{t} ẽ_s} is the partial sum of residuals and
 * {@code σ̂²} is the Newey-West long-run variance estimator.
 *
 * <p>Mirrors {@code statsmodels.tsa.stattools.kpss} exactly, including the
 * Hobijn et al. (1998) automatic lag selection and the interpolated p-values
 * from Kwiatkowski et al. (1992) Table 1.</p>
 *
 * <p><b>Note:</b> The null hypothesis is <em>stationarity</em>. A small p-value
 * rejects stationarity (evidence of a unit root).</p>
 */
public record KPSS(double statistic, double pValue, int lags, double[] critValues) {

    /** Returns the critical value at the given significance level (0.10, 0.05, 0.025, 0.01). */
    public double critValue(double alpha) {
        return switch ((int) Math.round(alpha * 1000)) {
            case 100 -> critValues[0];
            case  50 -> critValues[1];
            case  25 -> critValues[2];
            case  10 -> critValues[3];
            default  -> throw new IllegalArgumentException("alpha must be 0.10, 0.05, 0.025, or 0.01");
        };
    }

    /** Deterministic component included in the auxiliary regression. */
    public enum Regression {
        /** Level-stationary null: demean the series. */
        CONSTANT,
        /** Trend-stationary null: detrend the series via OLS. */
        TREND
    }

    /** Lag selection strategy for the Newey-West variance estimator. */
    public enum LagMethod {
        /** Hobijn et al. (1998) data-dependent method (default). */
        AUTO,
        /** Legacy rule: {@code int(12 * (n/100)^(1/4))} from Schwert (1989). */
        LEGACY
    }

    /**
     * Runs the KPSS test.
     *
     * @param x          input time series
     * @param regression deterministic component ({@link Regression#CONSTANT} or {@link Regression#TREND})
     * @param lagMethod  lag selection method
     * @param nLags      explicit lag count; ignored when {@code lagMethod} is not {@code null};
     *                   pass {@code null} to use {@code lagMethod}
     * @return test result
     */
    public static KPSS test(double[] x, Regression regression, LagMethod lagMethod, Integer nLags) {
        int nobs = x.length;
        if (nobs < 4) throw new IllegalArgumentException("series too short for KPSS test");

        // ── Compute residuals ─────────────────────────────────────────
        double[] resids;
        double[] crit;
        if (regression == Regression.TREND) {
            // OLS detrend: y = β₀ + β₁·t + ε
            // Row-major X: X[row * 2 + col]
            double[] X = new double[nobs * 2];
            for (int i = 0; i < nobs; i++) {
                X[i * 2]     = 1.0;       // constant
                X[i * 2 + 1] = i + 1.0;  // trend
            }
            var ols = Regressor.ols(x, X, nobs, 2,
                Regressor.Opts.QR, Regressor.Opts.HAS_CONST, Regressor.Opts.FITNESS);
            resids = ols.residual(false);
            crit = new double[]{0.119, 0.146, 0.176, 0.216};
        } else {
            // Level: demean
            resids = x.clone();
            VectorOps.addConst(-VectorOps.mean(x), resids);
            crit = new double[]{0.347, 0.463, 0.574, 0.739};
        }

        // ── Determine lag count ───────────────────────────────────────
        int lag;
        if (nLags != null && lagMethod == null) {
            lag = nLags;
            if (lag >= nobs) throw new IllegalArgumentException("nLags must be < nobs");
        } else if (lagMethod == LagMethod.LEGACY) {
            lag = (int) Math.ceil(12.0 * Math.pow(nobs / 100.0, 0.25));
            lag = Math.min(lag, nobs - 1);
        } else {
            // AUTO: Hobijn et al. (1998)
            lag = kpssAutoLag(resids, nobs);
            lag = Math.min(lag, nobs - 1);
        }

        // ── Newey-West long-run variance σ̂² ──────────────────────────
        double sHat = sigmaEstKpss(resids, nobs, lag);

        // ── Test statistic η = (1/T²) Σ S_t² / σ̂² ───────────────────
        double cumSum = 0.0, eta = 0.0;
        for (double r : resids) {
            cumSum += r;
            eta += cumSum * cumSum;
        }
        eta /= ((double) nobs * nobs);
        double kpssStat = eta / sHat;

        // ── Interpolate p-value from Table 1 of Kwiatkowski et al. ───
        // crit values correspond to p = [0.10, 0.05, 0.025, 0.01]
        double[] pvals = {0.10, 0.05, 0.025, 0.01};
        double pValue = interpolate(kpssStat, crit, pvals);

        return new KPSS(kpssStat, pValue, lag, crit);
    }

    /**
     * Convenience overload: level-stationary null, automatic lag selection.
     */
    public static KPSS test(double[] x) {
        return test(x, Regression.CONSTANT, LagMethod.AUTO, null);
    }

    // ── Private helpers ───────────────────────────────────────────────

    /**
     * Newey-West variance estimator (Bartlett kernel).
     * Equation 10, p. 164 of Kwiatkowski et al. (1992).
     */
    private static double sigmaEstKpss(double[] resids, int nobs, int lags) {
        double sHat = VectorOps.dot(resids, 0, resids, 0, nobs);
        for (int i = 1; i <= lags; i++) {
            double prod = VectorOps.dot(resids, i, resids, 0, nobs - i);
            sHat += 2.0 * prod * (1.0 - (double) i / (lags + 1.0));
        }
        return sHat / nobs;
    }

    /**
     * Automatic lag selection via Hobijn et al. (1998).
     */
    private static int kpssAutoLag(double[] resids, int nobs) {
        int covLags = (int) Math.pow(nobs, 2.0 / 9.0);
        double s0 = VectorOps.dot(resids, 0, resids, 0, nobs) / nobs;
        double s1 = 0.0;
        for (int i = 1; i <= covLags; i++) {
            double prod = VectorOps.dot(resids, i, resids, 0, nobs - i) / (nobs / 2.0);
            s0 += prod;
            s1 += i * prod;
        }
        double sHat = s1 / s0;
        double gammaHat = 1.1447 * Math.pow(sHat * sHat, 1.0 / 3.0);
        return (int) (gammaHat * Math.pow(nobs, 1.0 / 3.0));
    }

    /**
     * Linear interpolation of {@code pvals} at {@code x} given the {@code xp} knots.
     * Returns boundary values when {@code x} is outside the range.
     */
    private static double interpolate(double x, double[] xp, double[] fp) {
        if (x <= xp[0]) return fp[0];
        if (x >= xp[xp.length - 1]) return fp[fp.length - 1];
        for (int i = 0; i < xp.length - 1; i++) {
            if (x >= xp[i] && x <= xp[i + 1]) {
                double t = (x - xp[i]) / (xp[i + 1] - xp[i]);
                return fp[i] + t * (fp[i + 1] - fp[i]);
            }
        }
        return fp[fp.length - 1];
    }
}
