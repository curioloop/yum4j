package com.curioloop.yum4j.stats;

import com.curioloop.yum4j.math.Double2;

import com.curioloop.yum4j.math.Beta;
import com.curioloop.yum4j.math.Gamma;
import com.curioloop.yum4j.math.Normal;
import com.curioloop.yum4j.math.RootIterators;

import java.util.random.RandomGenerator;

/**
 * Boost-style Student's t distribution object with unified PDF/CDF/quantile access.
 */
public value record StudentsTDistribution(double degreesOfFreedom) implements ContinuousDistribution {

    private static final double LARGE_DEGREES_OF_FREEDOM = 1.0 / Math.ulp(1.0);
    private static final int DOUBLE_PRECISION_BITS = 53;
    private static final int ROOT_ITERATIONS = 256;
    private static final double MIN_DEGREES_OF_FREEDOM = 1.0e-12;
    private static final double DEFAULT_DF_HINT = 100.0;

    public StudentsTDistribution {
        validateDegreesOfFreedom(degreesOfFreedom);
    }

    @Override
    public Double2 range() {
        return Double2.bound(Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY);
    }

    @Override
    public Double2 support() {
        return Double2.bound(Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY);
    }

    public double pdf(double x) {
        if (Double.isNaN(x)) {
            throw new IllegalArgumentException("x must not be NaN");
        }
        return pdfKernel(degreesOfFreedom, x);
    }

    public double cdf(double x) {
        if (Double.isNaN(x)) {
            throw new IllegalArgumentException("x must not be NaN");
        }
        if (x == 0.0) {
            return 0.5;
        }
        if (x == Double.NEGATIVE_INFINITY) {
            return 0.0;
        }
        if (x == Double.POSITIVE_INFINITY) {
            return 1.0;
        }
        return cdfKernel(degreesOfFreedom, x);
    }

    public double ccdf(double x) {
        if (Double.isNaN(x)) {
            throw new IllegalArgumentException("x must not be NaN");
        }
        if (x == 0.0) {
            return 0.5;
        }
        if (x == Double.NEGATIVE_INFINITY) {
            return 1.0;
        }
        if (x == Double.POSITIVE_INFINITY) {
            return 0.0;
        }
        return ccdfKernel(degreesOfFreedom, x);
    }

    public double quantile(double probability) {
        validateProbability(probability, "probability");
        if ((probability == 0.0) || (probability == 1.0)) {
            throw new ArithmeticException("StudentsTDistribution.quantile overflow: probability=" + probability);
        }
        return quantileKernel(degreesOfFreedom, probability);
    }

    public double quantileUpperTail(double probability) {
        validateProbability(probability, "probability");
        if ((probability == 0.0) || (probability == 1.0)) {
            throw new ArithmeticException("StudentsTDistribution.quantileUpperTail overflow: probability=" + probability);
        }
        return quantileUpperTailKernel(degreesOfFreedom, probability);
    }

    public double mode() {
        return 0.0;
    }

    public double median() {
        return 0.0;
    }

    public double mean() {
        if (degreesOfFreedom <= 1.0 || Double.isNaN(degreesOfFreedom)) {
            throw new ArithmeticException("Mean is undefined for degrees of freedom <= 1: " + degreesOfFreedom);
        }
        return 0.0;
    }

    public double variance() {
        if (degreesOfFreedom <= 2.0 || Double.isNaN(degreesOfFreedom)) {
            throw new ArithmeticException("Variance is undefined for degrees of freedom <= 2: " + degreesOfFreedom);
        }
        if (usesNormalApproximation(degreesOfFreedom)) {
            return 1.0;
        }
        return degreesOfFreedom / (degreesOfFreedom - 2.0);
    }

    public double skewness() {
        if (degreesOfFreedom <= 3.0 || Double.isNaN(degreesOfFreedom)) {
            throw new ArithmeticException("Skewness is undefined for degrees of freedom <= 3: " + degreesOfFreedom);
        }
        return 0.0;
    }

    public double kurtosis() {
        if (degreesOfFreedom <= 4.0 || Double.isNaN(degreesOfFreedom)) {
            throw new ArithmeticException("Kurtosis is undefined for degrees of freedom <= 4: " + degreesOfFreedom);
        }
        if (usesNormalApproximation(degreesOfFreedom)) {
            return 3.0;
        }
        return 3.0 + 6.0 / (degreesOfFreedom - 4.0);
    }

    public double kurtosisExcess() {
        if (degreesOfFreedom <= 4.0 || Double.isNaN(degreesOfFreedom)) {
            throw new ArithmeticException("Kurtosis excess is undefined for degrees of freedom <= 4: " + degreesOfFreedom);
        }
        if (usesNormalApproximation(degreesOfFreedom)) {
            return 0.0;
        }
        return 6.0 / (degreesOfFreedom - 4.0);
    }

    public double entropy() {
        double vp1 = 0.5 * (degreesOfFreedom + 1.0);
        double vd2 = 0.5 * degreesOfFreedom;
        return vp1 * (Gamma.digamma(vp1) - Gamma.digamma(vd2))
            + Math.log(Math.sqrt(degreesOfFreedom) * Beta.beta(vd2, 0.5));
    }

    // ---- Batch overrides --------------------------------------

    /**
     * Direct-formula log-pdf for Student's t:
     *
     * <pre>
     * logPdf(x) = lgamma((v+1)/2) - 0.5*log(v*pi) - lgamma(v/2)
     *           - ((v+1)/2) * log(1 + x*x/v)
     * </pre>
     */
    @Override
    public double logPdf(double x) {
        if (Double.isNaN(x)) {
            throw new IllegalArgumentException("x must not be NaN");
        }
        double logConst = studentLogConst(degreesOfFreedom);
        double halfDfP1 = 0.5 * (degreesOfFreedom + 1.0);
        if (Double.isInfinite(x)) {
            return Double.NEGATIVE_INFINITY;
        }
        return logConst - halfDfP1 * Math.log1p(x * x / degreesOfFreedom);
    }

    @Override
    public void batch(Metric metric, double[] x, int xOff, int xStride, int n,
                            double[] out, int outOff) {
        if (metric == Metric.LOG_PDF) {
            if (n == 0) return;
            double logConst = studentLogConst(degreesOfFreedom);
            double halfDfP1 = 0.5 * (degreesOfFreedom + 1.0);
            double invDf = 1.0 / degreesOfFreedom;
            for (int i = 0; i < n; i++) {
                double v = x[xOff + i * xStride];
                if (Double.isInfinite(v)) {
                    out[outOff + i] = Double.NEGATIVE_INFINITY;
                } else {
                    out[outOff + i] = logConst - halfDfP1 * Math.log1p(v * v * invDf);
                }
            }
        } else {
            ContinuousDistribution.super.batch(metric, x, xOff, xStride, n, out, outOff);
        }
    }

    private static double studentLogConst(double df) {
        return Gamma.lgamma(0.5 * (df + 1.0))
            - 0.5 * Math.log(df * Math.PI)
            - Gamma.lgamma(0.5 * df);
    }

    // ---- Sampling overrides ---------------------------------------

    /**
     * Sample via {@code Z / sqrt(ChiSq(df) / df)}, where
     * {@code ChiSq(df) = Gamma(shape = df/2, scale = 2)}. The Gamma
     * kernel is the shared Marsaglia-Tsang path
     */
    @Override
    public double sample(RandomGenerator g) {
        double z = g.nextGaussian();
        double chi = Distribution.nextGamma(g, 0.5 * degreesOfFreedom) * 2.0;
        return z / Math.sqrt(chi / degreesOfFreedom);
    }

    @Override
    public void sample(RandomGenerator g, int n, double[] out, int off, int stride) {
        if (n == 0) return;
        double halfDf = 0.5 * degreesOfFreedom;
        for (int i = 0; i < n; i++) {
            double z = g.nextGaussian();
            double chi = Distribution.nextGamma(g, halfDf) * 2.0;
            out[off + i * stride] = z / Math.sqrt(chi / degreesOfFreedom);
        }
    }

    public static double findDegreesOfFreedom(double differenceFromMean,
                                              double alpha,
                                              double beta,
                                              double sampleStandardDeviation) {
        return findDegreesOfFreedom(differenceFromMean, alpha, beta, sampleStandardDeviation, DEFAULT_DF_HINT);
    }

    public static double findDegreesOfFreedom(double differenceFromMean,
                                              double alpha,
                                              double beta,
                                              double sampleStandardDeviation,
                                              double hint) {
        validateProbability(alpha, "alpha");
        validateProbability(beta, "beta");
        if (!Double.isFinite(differenceFromMean) || differenceFromMean == 0.0) {
            throw new IllegalArgumentException("differenceFromMean must be finite and non-zero: " + differenceFromMean);
        }
        if (!Double.isFinite(sampleStandardDeviation) || !(sampleStandardDeviation > 0.0)) {
            throw new IllegalArgumentException(
                    "sampleStandardDeviation must be finite and positive: " + sampleStandardDeviation
            );
        }

        double safeHint = hint > 0.0 && Double.isFinite(hint) ? hint : 1.0;
        double ratio = sampleStandardDeviation * sampleStandardDeviation
                / (differenceFromMean * differenceFromMean);

        return RootIterators.toms748SolveBounded(
            degrees -> sampleSizeFunction(degrees, alpha, beta, ratio),
            safeHint,
            2.0,
            false,
            MIN_DEGREES_OF_FREEDOM,
            DOUBLE_PRECISION_BITS,
            ROOT_ITERATIONS,
            "StudentsTDistribution.findDegreesOfFreedom"
        );
    }

    private static double sampleSizeFunction(double degreesOfFreedom,
                                             double alpha,
                                             double beta,
                                             double ratio) {
        if (degreesOfFreedom <= MIN_DEGREES_OF_FREEDOM) {
            return 1.0;
        }
        double qa = quantileUpperTailKernel(degreesOfFreedom, alpha);
        double qb = quantileUpperTailKernel(degreesOfFreedom, beta);
        double total = qa + qb;
        return total * total * ratio - (degreesOfFreedom + 1.0);
    }

    static double pdfKernel(double degreesOfFreedom, double x) {
        if (Double.isInfinite(x)) {
            return 0.0;
        }
        if (usesNormalApproximation(degreesOfFreedom)) {
            return Normal.pdf(x);
        }

        double base = x * x / degreesOfFreedom;
        double result;
        if (base < 0.125) {
            result = Math.exp(-Math.log1p(base) * (degreesOfFreedom + 1.0) * 0.5);
        } else {
            result = Math.pow(1.0 / (1.0 + base), (degreesOfFreedom + 1.0) * 0.5);
        }
        result /= Math.sqrt(degreesOfFreedom) * Beta.beta(0.5 * degreesOfFreedom, 0.5);
        return result;
    }

    static double cdfKernel(double degreesOfFreedom, double x) {
        if (x == 0.0) {
            return 0.5;
        }
        if (x == Double.NEGATIVE_INFINITY) {
            return 0.0;
        }
        if (x == Double.POSITIVE_INFINITY) {
            return 1.0;
        }
        if (usesNormalApproximation(degreesOfFreedom)) {
            return Normal.cdf(x);
        }

        double probability = halfTailProbability(degreesOfFreedom, Math.abs(x));
        return x > 0.0 ? 1.0 - probability : probability;
    }

    static double ccdfKernel(double degreesOfFreedom, double x) {
        return cdfKernel(degreesOfFreedom, -x);
    }

    static double quantileKernel(double degreesOfFreedom, double probability) {
        if (probability == 0.5) {
            return 0.0;
        }
        return StudentsTQuantileSupport.fastQuantile(degreesOfFreedom, probability);
    }

    static double quantileUpperTailKernel(double degreesOfFreedom, double probability) {
        return -quantileKernel(degreesOfFreedom, probability);
    }

    private static double halfTailProbability(double degreesOfFreedom, double absX) {
        double xSquared = absX * absX;
        double z;
        if (degreesOfFreedom > 2.0 * xSquared) {
            z = xSquared / (degreesOfFreedom + xSquared);
            return 0.5 * Beta.ibetac(0.5, 0.5 * degreesOfFreedom, z);
        }
        z = degreesOfFreedom / (degreesOfFreedom + xSquared);
        return 0.5 * Beta.ibeta(0.5 * degreesOfFreedom, 0.5, z);
    }

    private static boolean usesNormalApproximation(double degreesOfFreedom) {
        return Double.isInfinite(degreesOfFreedom) || degreesOfFreedom > LARGE_DEGREES_OF_FREEDOM;
    }

    private static void validateDegreesOfFreedom(double degreesOfFreedom) {
        if (Double.isNaN(degreesOfFreedom) || !(degreesOfFreedom > 0.0 || Double.isInfinite(degreesOfFreedom))) {
            throw new IllegalArgumentException("degreesOfFreedom must be positive or +infinity: " + degreesOfFreedom);
        }
    }

    private static void validateProbability(double probability, String name) {
        if (Double.isNaN(probability) || probability < 0.0 || probability > 1.0) {
            throw new IllegalArgumentException(name + " must be in [0, 1]: " + probability);
        }
    }
}