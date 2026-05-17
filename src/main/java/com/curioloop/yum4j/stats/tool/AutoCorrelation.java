package com.curioloop.yum4j.stats.tool;

import com.curioloop.yum4j.math.Double2;
import com.curioloop.yum4j.stats.ChiSquareDistribution;
import com.curioloop.yum4j.stats.NormalDistribution;
import com.curioloop.yum4j.math.VectorOps;

/**
 * Autocorrelation function (ACF) and Ljung-Box Q-statistic.
 *
 * <p>Mirrors {@code statsmodels.tsa.stattools.acf} and {@code q_stat} exactly.</p>
 *
 * <h3>ACF definition</h3>
 * <pre>{@code
 * acf[k] = acovf[k] / acovf[0]
 * }</pre>
 *
 * <h3>Bartlett confidence intervals</h3>
 * <p>When {@code bartlettConfInt=true} (default), the variance at lag {@code k} is</p>
 * <pre>{@code
 * var[0] = 0
 * var[1] = 1/n
 * var[k] = (1 + 2 * Σ_{j=1}^{k-1} acf[j]²) / n   for k >= 2
 * }</pre>
 * <p>Otherwise a flat {@code 1/n} is used for all lags.</p>
 *
 * <h3>Q-statistic (Ljung-Box)</h3>
 * <pre>{@code
 * Q[k] = n(n+2) * Σ_{j=1}^{k} acf[j]² / (n-j)
 * }</pre>
 *
 * @param acf      autocorrelation values for lags 0, 1, …, nLags (length {@code nLags+1})
 * @param confInt  confidence intervals as {@link Double2}{@code [nLags+1]} where each element
 *                 carries {@code (lower, upper)} bounds; {@code null} when not requested
 * @param qStat    Ljung-Box Q-statistics for lags 1, …, nLags; {@code null} when not requested
 * @param pValues  p-values for the Q-statistics; {@code null} when not requested
 */
public record AutoCorrelation(double[] acf, Double2[] confInt, double[] qStat, double[] pValues) {

    /**
     * Computes the ACF with all optional outputs.
     *
     * @param x               input time series
     * @param adjusted        if {@code true}, use unbiased autocovariance denominator {@code n-k}
     * @param nLags           number of lags (exclusive of lag 0); {@code null} → {@code min(10·log₁₀(n), n-1)}
     * @param qStat           if {@code true}, compute Ljung-Box Q-statistics and p-values
     * @param fft             if {@code true}, use FFT path in autocovariance computation
     * @param alpha           confidence level for intervals (e.g. {@code 0.05} for 95 %);
     *                        {@code null} to skip confidence interval computation
     * @param bartlettConfInt if {@code true}, use Bartlett's formula for confidence intervals
     * @param missing         strategy for handling NaN values
     * @return {@link AutoCorrelation} containing the requested outputs
     */
    public static AutoCorrelation compute(double[] x, boolean adjusted, Integer nLags,
                                          boolean qStat, boolean fft, Double alpha,
                                          boolean bartlettConfInt,
                                          MissingPolicy missing) {
        if (x.length == 0) {
            throw new IllegalArgumentException("input series x cannot be empty");
        }
        int nobs = x.length;
        if (nLags == null) {
            nLags = Math.min((int) (10 * Math.log10(nobs)), nobs - 1);
        }

        // Compute autocovariance (always demean for ACF)
        double[] avf = AutoCovariance.compute(x, adjusted, true, fft, missing, nLags);

        // Normalise to autocorrelation in-place; avf is not exposed elsewhere.
        VectorOps.scal(1.0 / avf[0], avf, 0, nLags + 1);
        double[] acf = avf;

        if (!qStat && alpha == null) {
            return new AutoCorrelation(acf, null, null, null);
        }

        // ── Confidence intervals ──────────────────────────────────────
        Double2[] confInt = null;
        if (alpha != null) {
            double ppf = new NormalDistribution().quantile(1.0 - alpha / 2.0);
            double invN = 1.0 / nobs;
            confInt = new Double2[acf.length];
            if (bartlettConfInt) {
                // var[0]=0, var[1]=1/n, var[k]=(1+2·Σ_{j=1}^{k-1} acf[j]²)/n for k>=2
                // Inlined: compute variance and confInt in one pass, no temp array.
                double interval0 = 0.0;
                confInt[0] = Double2.bound(acf[0] - interval0, acf[0] + interval0);
                if (acf.length > 1) {
                    double interval1 = Math.sqrt(invN) * ppf;
                    confInt[1] = Double2.bound(acf[1] - interval1, acf[1] + interval1);
                    double cumSumSq = 0.0;
                    for (int i = 2; i < acf.length; i++) {
                        cumSumSq += acf[i - 1] * acf[i - 1];
                        double interval = Math.sqrt((1.0 + 2.0 * cumSumSq) * invN) * ppf;
                        confInt[i] = Double2.bound(acf[i] - interval, acf[i] + interval);
                    }
                }
            } else {
                double interval = Math.sqrt(invN) * ppf;
                for (int i = 0; i < acf.length; i++)
                    confInt[i] = Double2.bound(acf[i] - interval, acf[i] + interval);
            }
        }

        // ── Q-statistic ───────────────────────────────────────────────
        double[] qStatVals = null;
        double[] pValues = null;
        if (qStat) {
            // acf[0] is lag-0 (always 1.0); Q-stat uses lags 1..nLags
            int m = nLags;
            qStatVals = new double[m];
            pValues   = new double[m];
            double cumSum = 0.0;
            for (int i = 1; i <= m; i++) {
                cumSum += acf[i] * acf[i] / (nobs - i);
                qStatVals[i - 1] = (double) nobs * (nobs + 2) * cumSum;
                pValues[i - 1] = new ChiSquareDistribution(i).ccdf(qStatVals[i - 1]);
            }
        }

        return new AutoCorrelation(acf, confInt, qStatVals, pValues);
    }

    /**
     * Convenience overload with sensible defaults:
     * {@code adjusted=false}, FFT enabled, Bartlett confidence intervals, no missing handling.
     *
     * @param x     input time series
     * @param nLags number of lags; {@code null} for automatic selection
     * @param alpha confidence level for intervals; {@code null} to skip
     * @return ACF result
     */
    public static AutoCorrelation compute(double[] x, Integer nLags, Double alpha) {
        return compute(x, false, nLags, false, true, alpha, true, MissingPolicy.NONE);
    }
}
