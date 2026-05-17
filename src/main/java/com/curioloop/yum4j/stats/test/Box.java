package com.curioloop.yum4j.stats.test;

import com.curioloop.yum4j.stats.ChiSquareDistribution;
import com.curioloop.yum4j.stats.tool.AutoCorrelation;
import com.curioloop.yum4j.stats.tool.MissingPolicy;

import java.util.Arrays;

/**
 * Ljung-Box and Box-Pierce portmanteau tests for serial autocorrelation.
 *
 * <p>Mirrors {@code statsmodels.stats.diagnostic.acorr_ljungbox} exactly,
 * including automatic lag selection and seasonal period support.</p>
 *
 * <h3>Test statistics</h3>
 * <pre>{@code
 * Ljung-Box:  Q_LB(h) = n(n+2) * Σ_{k=1}^{h} ρ̂_k² / (n-k)
 * Box-Pierce: Q_BP(h) = n * Σ_{k=1}^{h} ρ̂_k²
 * }</pre>
 * <p>Both are compared against a χ²({@code h - adjustDF}) distribution.</p>
 *
 * <h3>Null hypothesis</h3>
 * <p>The data are independently distributed (no serial autocorrelation up to lag {@code h}).</p>
 *
 * @param lags          the lag values tested (1-based)
 * @param ljungBoxStat  Ljung-Box Q-statistics; {@code null} if not requested
 * @param ljungBoxPv    p-values for Ljung-Box; {@code null} if not requested
 * @param boxPierceStat Box-Pierce Q-statistics; {@code null} if not requested
 * @param boxPiercePv   p-values for Box-Pierce; {@code null} if not requested
 */
public record Box(int[] lags,
                  double[] ljungBoxStat, double[] ljungBoxPv,
                  double[] boxPierceStat, double[] boxPiercePv) {

    /** Selects which test statistics to compute. */
    public enum Statistic {
        /** Compute only the Ljung-Box statistic. */
        LJUNG_BOX,
        /** Compute only the Box-Pierce statistic. */
        BOX_PIERCE,
        /** Compute both statistics. */
        BOTH
    }

    /**
     * Runs the portmanteau test.
     *
     * @param x         input time series (demeaned internally by ACF)
     * @param statistic which statistics to compute
     * @param adjustDF  degrees-of-freedom adjustment (e.g. {@code p+q} for ARMA residuals)
     * @param autoLag   if {@code true}, automatically select the optimal lag using the
     *                  penalised Q-statistic criterion from Escanciano &amp; Lobato (2009)
     * @param period    seasonal period for default lag selection ({@code 0} = no seasonality)
     * @param lags      explicit lag values to test; ignored when {@code autoLag=true} or
     *                  when a single value is passed to set the maximum lag
     * @return test result
     * @throws IllegalArgumentException on invalid inputs
     */
    public static Box test(double[] x, Statistic statistic, int adjustDF,
                           boolean autoLag, int period, int... lags) {
        int nobs = x.length;
        if (nobs < 2) throw new IllegalArgumentException("data series is too short");
        if (adjustDF < 0) throw new IllegalArgumentException("adjustDF must be >= 0");
        if (period != 0 && period <= 1) throw new IllegalArgumentException("period must be >= 2");
        if (lags.length > 0) {
            for (int lag : lags) if (lag < 0) throw new IllegalArgumentException("lags must be >= 0");
        }

        boolean doLB = (statistic == Statistic.LJUNG_BOX || statistic == Statistic.BOTH);
        boolean doBP = (statistic == Statistic.BOX_PIERCE || statistic == Statistic.BOTH);

        // ── Determine the set of lags to test ─────────────────────────
        int maxLag;
        int[] testLags;

        if (autoLag) {
            maxLag = nobs - 1;
            double[] acf = AutoCorrelation.compute(x, false, maxLag, false, false, null, true,
                    MissingPolicy.NONE).acf();
            double maxAbsAcf = 0.0;
            for (int i = 1; i <= maxLag; i++) maxAbsAcf = Math.max(maxAbsAcf, Math.abs(acf[i]));
            double thresholdMetric = maxAbsAcf * Math.sqrt(nobs);
            double threshold = Math.sqrt(2.4 * Math.log(nobs));
            double penalty = thresholdMetric <= threshold ? Math.log(nobs) : 2.0;

            // Compute cumulative penalised Q-statistic to find optimal lag.
            double bestQ = Double.NEGATIVE_INFINITY;
            int optIdx = 0;
            if (doLB || statistic == Statistic.BOTH) {
                double cumSum = 0.0;
                for (int i = 1; i <= maxLag; i++) {
                    cumSum += acf[i] * acf[i] / (nobs - i);
                    double q = (double) nobs * (nobs + 2) * cumSum - i * penalty;
                    if (q > bestQ) {
                        bestQ = q;
                        optIdx = i - 1;
                    }
                }
            } else {
                double cumSum = 0.0;
                for (int i = 1; i <= maxLag; i++) {
                    cumSum += acf[i] * acf[i];
                    double q = nobs * cumSum - i * penalty;
                    if (q > bestQ) {
                        bestQ = q;
                        optIdx = i - 1;
                    }
                }
            }
            int optLag = Math.max(1, optIdx + 1); // 1-based
            testLags = range(1, optLag + 1);
            maxLag = optLag;
            // Reuse the already-computed ACF (no need to recompute)
            return accumulate(acf, nobs, maxLag, testLags, doLB, doBP, adjustDF);
        } else if (period != 0) {
            maxLag = Math.min(nobs / 5, period * 2);
            testLags = range(1, maxLag + 1);
        } else if (lags.length == 0) {
            maxLag = Math.min(nobs / 5, 10);
            testLags = range(1, maxLag + 1);
        } else if (lags.length == 1) {
            maxLag = lags[0];
            testLags = range(1, maxLag + 1);
        } else {
            testLags = Arrays.copyOf(lags, lags.length);
            Arrays.sort(testLags);
            maxLag = testLags[testLags.length - 1];
        }

        // ── Compute ACF up to maxLag ──────────────────────────────────
        double[] acf = AutoCorrelation.compute(x, false, maxLag, false, false, null, true,
                MissingPolicy.NONE).acf();

        return accumulate(acf, nobs, maxLag, testLags, doLB, doBP, adjustDF);
    }

    /** Accumulates LB/BP statistics from a pre-computed ACF array. */
    private static Box accumulate(double[] acf, int nobs, int maxLag,
                                  int[] testLags, boolean doLB, boolean doBP, int adjustDF) {
        // ── Accumulate statistics ─────────────────────────────────────
        double[] lbStat = doLB ? new double[testLags.length] : null;
        double[] lbPv   = doLB ? new double[testLags.length] : null;
        double[] bpStat = doBP ? new double[testLags.length] : null;
        double[] bpPv   = doBP ? new double[testLags.length] : null;

        double cumLB = 0.0, cumBP = 0.0;
        int lagIdx = 0;
        for (int i = 1; i <= maxLag && lagIdx < testLags.length; i++) {
            cumLB += acf[i] * acf[i] / (nobs - i);
            cumBP += acf[i] * acf[i];
            if (i == testLags[lagIdx]) {
                if (doLB) lbStat[lagIdx] = (double) nobs * (nobs + 2) * cumLB;
                if (doBP) bpStat[lagIdx] = nobs * cumBP;
                double adjDof = i - adjustDF;
                if (adjDof > 0) {
                    ChiSquareDistribution chi2 = new ChiSquareDistribution(adjDof);
                    if (doLB) lbPv[lagIdx] = chi2.ccdf(lbStat[lagIdx]);
                    if (doBP) bpPv[lagIdx] = chi2.ccdf(bpStat[lagIdx]);
                } else {
                    if (doLB) lbPv[lagIdx] = Double.NaN;
                    if (doBP) bpPv[lagIdx] = Double.NaN;
                }
                lagIdx++;
            }
        }

        return new Box(testLags, lbStat, lbPv, bpStat, bpPv);
    }

    /**
     * Convenience overload: Ljung-Box only, no degree-of-freedom adjustment,
     * no auto-lag, no seasonality.
     *
     * @param x    input time series
     * @param lags explicit lag values; if empty, defaults to {@code min(n/5, 10)}
     * @return test result
     */
    public static Box ljungBox(double[] x, int... lags) {
        return test(x, Statistic.LJUNG_BOX, 0, false, 0, lags);
    }

    // ── Utilities ─────────────────────────────────────────────────────

    /** Returns {@code [from, from+1, ..., to-1]}. */
    private static int[] range(int from, int to) {
        int[] a = new int[to - from];
        for (int i = 0; i < a.length; i++) a[i] = from + i;
        return a;
    }

}
