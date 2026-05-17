package com.curioloop.yum4j.stats;

import com.curioloop.yum4j.math.Double2;

import com.curioloop.yum4j.math.Gamma;

/** Boost-style hypergeometric distribution object with unified PDF/CDF/quantile access. */
public value record HypergeometricDistribution(long defective, long sampleCount, long total)
    implements DiscreteDistribution {

    public HypergeometricDistribution {
        if (defective < 0L) {
            throw new IllegalArgumentException("defective must be non-negative: " + defective);
        }
        if (sampleCount < 0L) {
            throw new IllegalArgumentException("sampleCount must be non-negative: " + sampleCount);
        }
        if (total < 0L) {
            throw new IllegalArgumentException("total must be non-negative: " + total);
        }
        if (defective > total) {
            throw new IllegalArgumentException("defective must be <= total: defective=" + defective + ", total=" + total);
        }
        if (sampleCount > total) {
            throw new IllegalArgumentException(
                "sampleCount must be <= total: sampleCount=" + sampleCount + ", total=" + total
            );
        }
    }

    @Override
    public Double2 range() {
        return Double2.bound(lowerSupport(), upperSupport());
    }

    @Override
    public Double2 support() {
        return range();
    }

    @Override
    public double pdf(double x) {
        validateX(x);
        if (Double.isInfinite(x) || !isIntegralPoint(x)) {
            return 0.0;
        }
        long value = (long) x;
        if (value < lowerSupport() || value > upperSupport()) {
            return 0.0;
        }
        return pdfAt(value);
    }

    /**
     * Direct-formula log-pmf:
     * {@code logBinom(K, x) + logBinom(N-K, n-x) - logBinom(N, n)}
     * where {@code K = defective, N = total, n = sampleCount}.
     *
     * <p>Avoids the {@code exp}/{@code log} round-trip in
     * {@link #pdfAt(long)} (which already computes the log first).
     */
    @Override
    public double logPdf(double x) {
        validateX(x);
        if (Double.isInfinite(x) || !isIntegralPoint(x)) {
            return Double.NEGATIVE_INFINITY;
        }
        long value = (long) x;
        if (value < lowerSupport() || value > upperSupport()) {
            return Double.NEGATIVE_INFINITY;
        }
        if (total == 0L) {
            return value == 0L ? 0.0 : Double.NEGATIVE_INFINITY;
        }
        return logBinomialCoefficient(defective, value)
            + logBinomialCoefficient(total - defective, sampleCount - value)
            - logBinomialCoefficient(total, sampleCount);
    }

    @Override
    public double cdf(double x) {
        validateX(x);
        long lower = lowerSupport();
        long upper = upperSupport();
        if (x < lower) {
            return 0.0;
        }
        if (Double.isInfinite(x) || x >= upper) {
            return 1.0;
        }
        return cdfAt((long) Math.floor(x));
    }

    @Override
    public double ccdf(double x) {
        validateX(x);
        long lower = lowerSupport();
        long upper = upperSupport();
        if (x < lower) {
            return 1.0;
        }
        if (Double.isInfinite(x) || x >= upper) {
            return 0.0;
        }
        return ccdfAt((long) Math.floor(x));
    }

    @Override
    public double quantile(double probability) {
        validateProbability(probability);
        return StatisticsQuantileSupport.discreteQuantile(this, probability);
    }

    @Override
    public double quantileUpperTail(double probability) {
        validateProbability(probability);
        return StatisticsQuantileSupport.discreteUpperTailQuantile(this, probability);
    }

    public double mode() {
        if (total == 0L) {
            return 0.0;
        }
        return Math.floor(((defective + 1.0) * (sampleCount + 1.0)) / (total + 2.0));
    }

    public double median() {
        return quantile(0.5);
    }

    @Override
    public double mean() {
        return (double) defective * sampleCount / total;
    }

    @Override
    public double variance() {
        double defectiveValue = defective;
        double sampleCountValue = sampleCount;
        double totalValue = total;
        return sampleCountValue * defectiveValue * (totalValue - defectiveValue) * (totalValue - sampleCountValue)
            / (totalValue * totalValue * (totalValue - 1.0));
    }

    public double skewness() {
        double defectiveValue = defective;
        double sampleCountValue = sampleCount;
        double totalValue = total;
        return (totalValue - 2.0 * defectiveValue) * Math.sqrt(totalValue - 1.0) * (totalValue - 2.0 * sampleCountValue)
            / (Math.sqrt(sampleCountValue * defectiveValue * (totalValue - defectiveValue) * (totalValue - sampleCountValue))
            * (totalValue - 2.0));
    }

    public double kurtosis() {
        return kurtosisExcess() + 3.0;
    }

    public double kurtosisExcess() {
        double defectiveValue = defective;
        double sampleCountValue = sampleCount;
        double sampleCountSquared = sampleCountValue * sampleCountValue;
        double totalValue = total;
        double totalSquared = totalValue * totalValue;
        return ((totalValue - 1.0) * totalSquared
            * ((3.0 * defectiveValue * (totalValue - defectiveValue)
            * (sampleCountSquared * (-totalValue)
            + (sampleCountValue - 2.0) * totalSquared
            + 6.0 * sampleCountValue * (totalValue - sampleCountValue))) / totalSquared
            - 6.0 * sampleCountValue * (totalValue - sampleCountValue)
            + totalValue * (totalValue + 1.0)))
            / (defectiveValue * sampleCountValue * (totalValue - 3.0) * (totalValue - 2.0)
            * (totalValue - defectiveValue) * (totalValue - sampleCountValue));
    }

    private double cdfAt(long x) {
        long lower = lowerSupport();
        long upper = upperSupport();
        if (x < lower) {
            return 0.0;
        }
        if (x >= upper) {
            return 1.0;
        }
        if (x - lower <= upper - x) {
            return clampProbability(lowerTailProbability(x));
        }
        return clampProbability(1.0 - upperTailProbability(x));
    }

    private double ccdfAt(long x) {
        long lower = lowerSupport();
        long upper = upperSupport();
        if (x < lower) {
            return 1.0;
        }
        if (x >= upper) {
            return 0.0;
        }
        if (x - lower <= upper - x) {
            return clampProbability(1.0 - lowerTailProbability(x));
        }
        return clampProbability(upperTailProbability(x));
    }

    private double lowerTailProbability(long upperInclusive) {
        long lower = lowerSupport();
        double term = pdfAt(lower);
        double sum = 0.0;
        double compensation = 0.0;
        double adjusted = term - compensation;
        double updated = sum + adjusted;
        compensation = (updated - sum) - adjusted;
        sum = updated;
        for (long value = lower; value < upperInclusive; value++) {
            term *= forwardRatio(value);
            adjusted = term - compensation;
            updated = sum + adjusted;
            compensation = (updated - sum) - adjusted;
            sum = updated;
        }
        return sum;
    }

    private double upperTailProbability(long lowerExclusive) {
        long upper = upperSupport();
        double term = pdfAt(upper);
        double sum = 0.0;
        double compensation = 0.0;
        for (long value = upper; value > lowerExclusive; value--) {
            double adjusted = term - compensation;
            double updated = sum + adjusted;
            compensation = (updated - sum) - adjusted;
            sum = updated;
            if (value == lowerSupport()) {
                break;
            }
            term *= backwardRatio(value);
        }
        return sum;
    }

    private double pdfAt(long x) {
        if (total == 0L) {
            return x == 0L ? 1.0 : 0.0;
        }
        double logProbability = logBinomialCoefficient(defective, x)
            + logBinomialCoefficient(total - defective, sampleCount - x)
            - logBinomialCoefficient(total, sampleCount);
        return Math.exp(logProbability);
    }

    private double forwardRatio(long x) {
        return ((double) (defective - x) * (sampleCount - x))
            / ((x + 1.0) * (total - defective - sampleCount + x + 1.0));
    }

    private double backwardRatio(long x) {
        return (x * (total - defective - sampleCount + x))
            / ((defective - x + 1.0) * (sampleCount - x + 1.0));
    }

    private long lowerSupport() {
        long threshold = total - defective;
        return sampleCount > threshold ? sampleCount - threshold : 0L;
    }

    private long upperSupport() {
        return Math.min(defective, sampleCount);
    }

    private static double logBinomialCoefficient(long n, long k) {
        if (k < 0L || k > n) {
            return Double.NEGATIVE_INFINITY;
        }
        return Gamma.lgamma(n + 1.0) - Gamma.lgamma(k + 1.0) - Gamma.lgamma(n - k + 1.0);
    }

    private static boolean isIntegralPoint(double x) {
        return x >= Long.MIN_VALUE && x <= Long.MAX_VALUE && x == Math.rint(x);
    }

    private static double clampProbability(double value) {
        if (value <= 0.0) {
            return 0.0;
        }
        if (value >= 1.0) {
            return 1.0;
        }
        return value;
    }

    private static void validateX(double x) {
        if (Double.isNaN(x)) {
            throw new IllegalArgumentException("x must not be NaN");
        }
    }

    private static void validateProbability(double probability) {
        if (Double.isNaN(probability) || probability < 0.0 || probability > 1.0) {
            throw new IllegalArgumentException("probability must be in [0, 1]: " + probability);
        }
    }
}