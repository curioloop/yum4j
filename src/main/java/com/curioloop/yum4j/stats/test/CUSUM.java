package com.curioloop.yum4j.stats.test;

import com.curioloop.yum4j.stats.KolmogorovSmirnovDistribution;
import com.curioloop.yum4j.math.VectorOps;

/**
 * CUSUM test for structural change / parameter stability.
 *
 * <p>Tests whether the cumulative sum of (scaled) residuals is consistent with
 * parameter stability throughout the sample. A large deviation from zero indicates
 * a structural break.</p>
 *
 * <h3>Test statistic</h3>
 * <pre>{@code
 * B_t = Σ_{s=1}^{t} e_s / √(Σ e_s²)
 * sup_B = max_t |B_t|
 * }</pre>
 * <p>Under the null of no structural change, {@code B_t} is asymptotically a
 * standard Brownian Bridge, so {@code sup_B} follows the Kolmogorov-Smirnov
 * distribution (kstwobign).</p>
 *
 * <p>Mirrors {@code statsmodels.stats.diagnostic.breaks_cusumolsresid}.</p>
 *
 * @param statistic  {@code sup|B_t|}, the maximum absolute scaled cumulative residual
 * @param pValue     asymptotic p-value based on the Kolmogorov-Smirnov distribution
 * @param critValues tabulated critical values: {@code [0]=1% (1.63), [1]=5% (1.36), [2]=10% (1.22)}
 * @param cusum      the full scaled cumulative sum array {@code B_t} (length = nobs)
 */
public record CUSUM(double statistic, double pValue, double[] critValues, double[] cusum) {

    /**
     * Runs the CUSUM test.
     *
     * @param resid residuals from an OLS (or other) estimation
     * @param ddof  number of estimated parameters; used to correct the variance denominator.
     *              Pass {@code 0} for raw residuals with no correction.
     * @return test result
     */
    public static CUSUM test(double[] resid, int ddof) {
        int nobs = resid.length;
        if (nobs < 2) throw new IllegalArgumentException("resid must have at least 2 elements");

        double nobsSigma2 = VectorOps.dot(resid, 0, resid, 0, nobs);
        if (ddof > 0) {
            nobsSigma2 = nobsSigma2 / (nobs - ddof) * nobs;
        }

        double scale = Math.sqrt(nobsSigma2);
        // Fused pass: cumulative sum, scale, and max-abs in one sweep.
        // Avoids three separate traversals (cumSum → scal → maxAbs) over nobs elements.
        double[] b = new double[nobs];
        double cum = 0.0;
        double supB = 0.0;
        double invScale = 1.0 / scale;
        for (int t = 0; t < nobs; t++) {
            cum += resid[t];
            double bt = cum * invScale;
            b[t] = bt;
            double abs = Math.abs(bt);
            if (abs > supB) supB = abs;
        }

        double pValue = new KolmogorovSmirnovDistribution(1.0).ccdf(supB);
        double[] critValues = {1.6276, 1.3581, 1.2238};

        return new CUSUM(supB, pValue, critValues, b);
    }

    /** Convenience overload with no degrees-of-freedom correction. */
    public static CUSUM test(double[] resid) {
        return test(resid, 0);
    }
}
