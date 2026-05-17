package com.curioloop.yum4j.stats.test;

import com.curioloop.yum4j.stats.ChiSquareDistribution;
import com.curioloop.yum4j.stats.FisherFDistribution;
import com.curioloop.yum4j.math.VectorOps;

/**
 * Break-in-variance heteroskedasticity test.
 *
 * <p>Compares the sum of squared residuals in the first and last subsets of a
 * residual series. The test statistic is</p>
 * <pre>{@code
 * H(h) = sum_{t=n-h+1}^{n} e_t^2 / sum_{t=1}^{h} e_t^2
 * }</pre>
 * <p>Under {@code useF=true}, {@code H(h)} is compared with an F distribution.
 * Under {@code useF=false}, {@code h * H(h)} is compared with a chi-square
 * distribution. Missing values are ignored inside the two tested subsets.</p>
 *
 * <p>Mirrors {@code statsmodels.tsa.stattools.breakvar_heteroskedasticity_test}.</p>
 *
 * @param statistic test statistic H(h) = SS_later / SS_earlier
 * @param pValue p-value under the chosen alternative
 */
public record BreakVar(double statistic, double pValue) {

    /** Direction of the variance change under the alternative hypothesis. */
    public enum Alternative {
        /** The later subset has larger variance than the earlier subset. */
        INCREASING,
        /** The earlier subset has larger variance than the later subset. */
        DECREASING,
        /** Variance changes in either direction. */
        TWO_SIDED
    }

    /**
     * Runs the break-in-variance test.
     *
     * @param residual residual array from a fitted model
     * @param subsetLen subset length; values in (0, 1] are treated as a fraction
     *                  of {@code residual.length}, while values greater than 1
     *                  are treated as an absolute observation count
     * @param alternative direction of the alternative hypothesis
     * @param useF if {@code true}, compare against an F distribution; otherwise
     *             use the chi-square approximation
     * @return test statistic and p-value
     * @throws IllegalArgumentException if the residual array is empty or either
     *                                  subset has too few non-missing observations
     */
    public static BreakVar test(double[] residual, double subsetLen, Alternative alternative, boolean useF) {
        if (residual.length == 0) {
            throw new IllegalArgumentException("residual array must not be empty");
        }
        int n = residual.length;
        int h = (subsetLen > 0 && subsetLen <= 1) ? (int) Math.round(n * subsetLen) : (int) subsetLen;

        Subset earlier = subset(residual, 0, h);
        Subset later = subset(residual, n - h, h);
        if (later.dof < 2 || earlier.dof < 2) {
            throw new IllegalArgumentException(
                    "subset of data has too few non-missing observations to calculate test statistic");
        }

        double stat = later.sumSquares / earlier.sumSquares;
        double pValue;

        if (useF) {
            FisherFDistribution f = new FisherFDistribution(later.dof, earlier.dof);
            pValue = switch (alternative) {
                case INCREASING -> f.ccdf(stat);
                case DECREASING -> {
                    stat = 1.0 / stat;
                    yield f.ccdf(stat);
                }
                case TWO_SIDED -> 2.0 * Math.min(f.cdf(stat), f.ccdf(stat));
            };
        } else {
            ChiSquareDistribution chi2 = new ChiSquareDistribution(earlier.dof);
            pValue = switch (alternative) {
                case INCREASING -> chi2.ccdf(later.dof * stat);
                case DECREASING -> {
                    stat = 1.0 / stat;
                    yield chi2.ccdf(later.dof * stat);
                }
                case TWO_SIDED -> {
                    double scaled = later.dof * stat;
                    yield 2.0 * Math.min(chi2.cdf(scaled), chi2.ccdf(scaled));
                }
            };
        }

        return new BreakVar(stat, pValue);
    }

    private static Subset subset(double[] residual, int offset, int length) {
        int dof = 0;
        boolean hasNaN = false;
        for (int i = 0; i < length; i++) {
            if (Double.isNaN(residual[offset + i])) {
                hasNaN = true;
            } else {
                dof++;
            }
        }
        if (!hasNaN) {
            return new Subset(dof, VectorOps.dot(residual, offset, residual, offset, length));
        }
        double sumSquares = 0.0;
        for (int i = 0; i < length; i++) {
            double value = residual[offset + i];
            if (!Double.isNaN(value)) sumSquares += value * value;
        }
        return new Subset(dof, sumSquares);
    }

    private record Subset(int dof, double sumSquares) {}
}