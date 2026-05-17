package com.curioloop.yum4j.stats.test;

import com.curioloop.yum4j.stats.ChiSquareDistribution;
import com.curioloop.yum4j.math.VectorOps;

/**
 * Jarque-Bera normality test.
 *
 * <p>Tests whether a residual series is compatible with a normal distribution
 * using sample skewness and kurtosis:</p>
 * <pre>{@code
 * JB = n / 6 * (S^2 + (K - 3)^2 / 4)
 * }</pre>
 * <p>where {@code S} is sample skewness and {@code K} is sample kurtosis, not
 * excess kurtosis. The statistic is asymptotically chi-square(2) under the null
 * of normality.</p>
 *
 * <p>Mirrors {@code statsmodels.stats.stattools.jarque_bera}.</p>
 *
 * @param statistic JB test statistic
 * @param pValue p-value from chi-square(2)
 * @param skewness sample skewness
 * @param kurtosis sample kurtosis, not excess kurtosis
 */
public record JarqueBera(double statistic, double pValue, double skewness, double kurtosis) {

    /**
     * Runs the Jarque-Bera normality test.
     *
     * @param residual residual array with at least two observations
     * @return JB statistic, p-value, skewness, and kurtosis
     * @throws IllegalArgumentException if fewer than two residuals are provided
     */
    public static JarqueBera test(double[] residual) {
        if (residual.length < 2) {
            throw new IllegalArgumentException("residual must contain at least 2 elements");
        }
        int n = residual.length;
        // Single pass: compute mean, m2, m3, m4 in one sweep.
        // VectorOps.mean + sumSqDev would walk the array twice internally,
        // and m3/m4 require a third pass — fuse them all.
        double mean = VectorOps.mean(residual);
        double m2 = 0.0, m3 = 0.0, m4 = 0.0;
        for (double value : residual) {
            double centered = value - mean;
            double c2 = centered * centered;
            m2 += c2;
            m3 += c2 * centered;
            m4 += c2 * c2;
        }
        m2 /= n;
        m3 /= n;
        m4 /= n;

        double skewness;
        double kurtosis;
        double tol = mean * mean * 1e-15;
        if (m2 > tol) {
            skewness = m3 / Math.pow(m2, 1.5);
            kurtosis = m4 / (m2 * m2);
        } else {
            skewness = Double.NaN;
            kurtosis = Double.NaN;
        }

        double jbStat = (n / 6.0) * (skewness * skewness + 0.25 * (kurtosis - 3) * (kurtosis - 3));
        double pValue = new ChiSquareDistribution(2.0).ccdf(jbStat);
        return new JarqueBera(jbStat, pValue, skewness, kurtosis);
    }
}