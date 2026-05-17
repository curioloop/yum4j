package com.curioloop.yum4j.stats.tool;

import com.curioloop.yum4j.math.Double2;
import com.curioloop.yum4j.stats.NormalDistribution;
import com.curioloop.yum4j.math.VectorOps;

import java.util.Arrays;

/**
 * Cross-covariance and cross-correlation functions between two time series.
 *
 * <p>Mirrors {@code statsmodels.tsa.stattools.ccovf} and {@code ccf}.</p>
 *
 * <h3>Cross-covariance</h3>
 * <pre>{@code
 * ccovf[k] = (1/d[k]) * Σ_{t=0}^{n-k-1} (x[t+k] - x̄)(y[t] - ȳ)
 * }</pre>
 * where {@code d[k] = n-k} when {@code adjusted=true}, or {@code d[k] = n} otherwise.
 *
 * <h3>Cross-correlation</h3>
 * <pre>{@code
 * ccf[k] = ccovf[k] / (std(x) * std(y))
 * }</pre>
 *
 * <p>The returned record holds cross-correlation values for lags 0, 1, …, nLags-1
 * and optional confidence intervals as {@link Double2}{@code []} where each element
 * carries {@code (lower, upper)} bounds.</p>
 *
 * @param ccf     cross-correlation values for lags 0, 1, …, nLags-1
 * @param confInt confidence intervals as {@link Double2}{@code [nLags]};
 *                {@code null} if not requested
 */
public record CrossCorrelation(double[] ccf, Double2[] confInt) {

    /**
     * Computes the cross-covariance function between {@code x} and {@code y}.
     *
     * @param x        first time series
     * @param y        second time series (must have the same length as {@code x})
     * @param adjusted if {@code true}, divide by {@code n-k}; otherwise divide by {@code n}
     * @param demean   if {@code true}, subtract the respective means before computing
     * @param fft      if {@code true}, use FFT-based computation
     * @return cross-covariance array for lags 0, 1, …, n-1
     */
    public static double[] ccovf(double[] x, double[] y, boolean adjusted, boolean demean, boolean fft) {
        return ccovf(x, y, adjusted, demean, fft, x.length);
    }

    private static double[] ccovf(double[] x, double[] y, boolean adjusted, boolean demean, boolean fft,
                                  int lagCount) {
        if (x.length != y.length)
            throw new IllegalArgumentException("x and y must have the same length");
        int n = x.length;
        if (n == 0)
            throw new IllegalArgumentException("x and y must not be empty");
        lagCount = Math.min(lagCount, n);

        double[] xo = x;
        double[] yo = y;
        if (demean) {
            xo = x.clone();
            VectorOps.addConst(-VectorOps.mean(x), xo);
            yo = y.clone();
            VectorOps.addConst(-VectorOps.mean(y), yo);
        }

        double[] ccov;
        if (fft) {
            // correlate(xo, yo, FULL)[n-1:] gives lags 0, 1, ..., n-1
            double[] full = Correlate.correlate(xo, yo, Correlate.Mode.FULL, true);
            ccov = Arrays.copyOfRange(full, n - 1, n - 1 + lagCount);
        } else {
            ccov = new double[lagCount];
            for (int k = 0; k < lagCount; k++) {
                ccov[k] = VectorOps.dot(xo, k, yo, 0, n - k);
            }
        }

        if (adjusted) {
            for (int k = 0; k < lagCount; k++) ccov[k] /= (n - k);
        } else {
            VectorOps.scal(1.0 / n, ccov);
        }
        return ccov;
    }

    /**
     * Computes the cross-correlation function between {@code x} and {@code y}.
     *
     * @param x        first time series
     * @param y        second time series
     * @param adjusted if {@code true}, use unbiased denominator {@code n-k}
     * @param fft      if {@code true}, use FFT-based computation
     * @param nLags    number of lags to return; {@code null} returns all {@code n} lags
     * @param alpha    confidence level (e.g. {@code 0.05} for 95 %); {@code null} to skip
     * @return cross-correlation result
     */
    public static CrossCorrelation compute(double[] x, double[] y, boolean adjusted, boolean fft,
                                           Integer nLags, Double alpha) {
        if (x.length != y.length)
            throw new IllegalArgumentException("x and y must have the same length");
        int n = x.length;
        int len = (nLags != null) ? Math.min(nLags, n) : n;

        double[] ccf = ccovf(x, y, adjusted, true, fft, len);

        // Normalise by std(x) * std(y)
        double norm = VectorOps.std(x) * VectorOps.std(y);
        VectorOps.scal(1.0 / norm, ccf);

        Double2[] confInt = null;
        if (alpha != null) {
            double interval = new NormalDistribution().quantile(1.0 - alpha / 2.0)
                    / Math.sqrt(n);
            confInt = new Double2[len];
            for (int k = 0; k < len; k++)
                confInt[k] = Double2.bound(ccf[k] - interval, ccf[k] + interval);
        }
        return new CrossCorrelation(ccf, confInt);
    }

    /**
     * Convenience overload with sensible defaults:
     * adjusted denominator, FFT enabled, no confidence intervals.
     */
    public static double[] compute(double[] x, double[] y, int nLags) {
        return compute(x, y, true, true, nLags, null).ccf();
    }
}
