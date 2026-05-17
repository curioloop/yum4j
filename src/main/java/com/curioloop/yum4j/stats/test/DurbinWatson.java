package com.curioloop.yum4j.stats.test;

import com.curioloop.yum4j.math.VectorOps;

/**
 * Durbin-Watson test statistic for first-order autocorrelation in residuals.
 *
 * <p>The statistic is</p>
 * <pre>{@code
 * DW = sum_{t=2}^{n} (e_t - e_{t-1})^2 / sum_{t=1}^{n} e_t^2
 * }</pre>
 * <p>Values near 2 indicate no first-order autocorrelation, values near 0
 * indicate positive autocorrelation, and values near 4 indicate negative
 * autocorrelation.</p>
 *
 * <p>Mirrors {@code statsmodels.stats.stattools.durbin_watson}.</p>
 */
public final class DurbinWatson {

    private DurbinWatson() {}

    /**
     * Computes the Durbin-Watson statistic.
     *
     * @param residual residual array with at least two observations
     * @return Durbin-Watson statistic in [0, 4] for typical residual series
     * @throws IllegalArgumentException if fewer than two residuals are provided
     */
    public static double test(double[] residual) {
        if (residual.length < 2) {
            throw new IllegalArgumentException("residual must contain at least 2 elements");
        }
        int n = residual.length;
        // Derive the two partial sums of squares from the full one in O(1):
        //   earlier = Σ_{t=0..n-2} e_t² = denominator - e[n-1]²
        //   later   = Σ_{t=1..n-1} e_t² = denominator - e[0]²
        // Numerator simplifies to:
        //   (later + earlier) - 2·cross
        //   = 2·denominator - e[0]² - e[n-1]² - 2·cross
        double denominator = VectorOps.dot(residual, 0, residual, 0, n);
        double cross = VectorOps.dot(residual, 1, residual, 0, n - 1);
        double first = residual[0], last = residual[n - 1];
        return (2.0 * denominator - first * first - last * last - 2.0 * cross) / denominator;
    }
}