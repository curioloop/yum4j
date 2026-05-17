package com.curioloop.yum4j.stats.test;

import com.curioloop.yum4j.stats.ChiSquareDistribution;
import com.curioloop.yum4j.math.VectorOps;

/**
 * D'Agostino-Pearson omnibus normality test.
 *
 * <p>Combines the skewness z-score from D'Agostino (1970) and the kurtosis
 * z-score from Anscombe and Glynn (1983) into a single chi-square statistic:</p>
 * <pre>{@code
 * K^2 = z_skew^2 + z_kurt^2
 * }</pre>
 * <p>Returns {@code NaN} statistic and p-value when fewer than 8 observations
 * are supplied, matching the valid-sample requirement in SciPy/statsmodels.</p>
 *
 * <p>Mirrors {@code scipy.stats.normaltest} and
 * {@code statsmodels.stats.stattools.omni_normtest}.</p>
 *
 * @param statistic K squared omnibus statistic
 * @param pValue p-value from chi-square(2)
 */
public record Omnibus(double statistic, double pValue) {

        /**
         * Runs the omnibus normality test.
         *
         * @param residual residual array; at least 8 observations are needed for a
         *                 finite statistic
         * @return omnibus statistic and p-value
         */
    public static Omnibus test(double[] residual) {
        int n = residual.length;
        if (n < 8) return new Omnibus(Double.NaN, Double.NaN);

        // Single pass: compute mean, m2, m3, m4 in one sweep (see JarqueBera).
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
        double skew = (m2 > 0) ? m3 / Math.pow(m2, 1.5) : 0.0;
        double kurt = (m2 > 0) ? m4 / (m2 * m2) : 3.0;

        // Skewness z-score transformation from D'Agostino (1970).
        double y = skew * Math.sqrt((double) (n + 1) * (n + 3) / (6.0 * (n - 2)));
        double b2s = 3.0 * (n * n + 27.0 * n - 70) * (n + 1) * (n + 3)
                / ((n - 2.0) * (n + 5) * (n + 7) * (n + 9));
        double w2 = -1.0 + Math.sqrt(2.0 * (b2s - 1.0));
        double delta = 1.0 / Math.sqrt(0.5 * Math.log(w2));
        double alpha = Math.sqrt(2.0 / (w2 - 1.0));
        double zSkew = delta * Math.log(y / alpha + Math.sqrt(y * y / (alpha * alpha) + 1.0));

        // Kurtosis z-score transformation from Anscombe and Glynn (1983).
        double eb2 = 3.0 * (n - 1.0) / (n + 1.0);
        double varb2 = 24.0 * n * (n - 2.0) * (n - 3.0)
                / ((n + 1.0) * (n + 1.0) * (n + 3.0) * (n + 5.0));
        double x = (kurt - eb2) / Math.sqrt(varb2);
        double sqrtB1 = 6.0 * (n * n - 5.0 * n + 2.0) / ((n + 7.0) * (n + 9.0))
                * Math.sqrt(6.0 * (n + 3.0) * (n + 5.0) / (n * (n - 2.0) * (n - 3.0)));
        double a = 6.0 + 8.0 / sqrtB1 * (2.0 / sqrtB1 + Math.sqrt(1.0 + 4.0 / (sqrtB1 * sqrtB1)));
        double zKurt = (1.0 - 2.0 / (9.0 * a))
                - Math.pow((1.0 - 2.0 / a) / (1.0 + x * Math.sqrt(2.0 / (a - 4.0))), 1.0 / 3.0);
        zKurt /= Math.sqrt(2.0 / (9.0 * a));

        double k2 = zSkew * zSkew + zKurt * zKurt;
        if (Double.isNaN(k2) || k2 < 0) return new Omnibus(Double.NaN, Double.NaN);
        return new Omnibus(k2, new ChiSquareDistribution(2.0).ccdf(k2));
    }
}