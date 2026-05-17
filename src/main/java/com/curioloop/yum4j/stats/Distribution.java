package com.curioloop.yum4j.stats;

import com.curioloop.yum4j.math.Double2;

import java.util.random.RandomGenerator;

/**
 * Unified contract for probability distributions: density evaluation,
 * batch computation, and random sampling.
 *
 * <p>Every concrete distribution in this package implements this
 * interface. The API is organised into three groups:
 *
 * <ul>
 *   <li><b>Single-point evaluation</b> — {@link #pdf}, {@link #logPdf},
 *       {@link #cdf}, {@link #ccdf}, {@link #quantile}.</li>
 *   <li><b>Batch evaluation</b> — {@link #batch(Metric, double[], int, int, int, double[], int)}
 *       evaluates any metric over a strided array. Distributions
 *       override this to hoist normalising constants out of the loop
 *       (10× speedup for Normal/Exponential).</li>
 *   <li><b>Sampling</b> — {@link #sample(RandomGenerator)} and
 *       {@link #sample(RandomGenerator, int, double[], int, int)} draw
 *       random variates. The batch form writes into a strided output
 *       buffer for direct column-major particle-buffer filling.</li>
 * </ul>
 *
 * <h3>Stride convention</h3>
 *
 * <p>Batch methods use a stride parameter for flexible memory layout:
 * <ul>
 *   <li>{@code collect} reads {@code x[xOff + i * xStride]} and writes
 *       {@code out[outOff + i]} (stride-1 output).</li>
 *   <li>{@code sample} writes {@code out[off + i * stride]} — stride
 *       on the output side for filling columns or rows of a matrix.</li>
 * </ul>
 *
 * <h3>Improper distributions</h3>
 *
 * <p>Distributions whose density is not normalisable (e.g.
 * {@link FlatNormalDistribution}) throw
 * {@link UnsupportedOperationException} from {@code sample},
 * {@code cdf}, and {@code quantile}.
 */
public interface Distribution {

    // ── Metric enum ───────────────────────────────────────────────────

    /**
     * Selects which scalar function to evaluate in batch via
     * {@link #batch}.
     */
    enum Metric {
        /** Log probability density (continuous) or log probability mass (discrete). */
        LOG_PDF,
        /** Probability density (continuous) or probability mass (discrete). */
        PDF,
        /** Cumulative distribution function. */
        CDF,
        /** Complementary CDF: {@code 1 - cdf(x)}. */
        CCDF,
        /** Quantile (inverse CDF). Input is a probability in [0, 1]. */
        QUANTILE
    }

    // ── Single-point evaluation ───────────────────────────────────────

    /**
     * Probability density function (continuous) or probability mass
     * function (discrete) evaluated at {@code x}.
     *
     * @param x evaluation point
     * @return {@code f(x) ≥ 0}
     */
    double pdf(double x);

    /**
     * Cumulative distribution function: {@code P(X ≤ x)}.
     *
     * @param x evaluation point
     * @return probability in [0, 1]
     */
    double cdf(double x);

    /**
     * Complementary CDF (survival function): {@code P(X > x) = 1 - cdf(x)}.
     *
     * <p>Computed directly rather than via subtraction to preserve
     * precision in the upper tail.
     *
     * @param x evaluation point
     * @return probability in [0, 1]
     */
    double ccdf(double x);

    /**
     * Quantile function (inverse CDF): the smallest {@code x} such that
     * {@code cdf(x) ≥ probability}.
     *
     * @param probability target probability in [0, 1]
     * @return quantile value
     */
    double quantile(double probability);

    /**
     * Upper-tail quantile: the largest {@code x} such that
     * {@code ccdf(x) ≥ probability}.
     *
     * <p>Equivalent to {@code quantile(1 - probability)} but computed
     * directly for upper-tail precision.
     *
     * @param probability target upper-tail probability in [0, 1]
     * @return quantile value
     */
    double quantileUpperTail(double probability);

    /**
     * Distribution mean {@code E[X]}.
     *
     * @return the expected value (may be {@code NaN} if undefined)
     */
    double mean();

    /**
     * Distribution variance {@code Var(X) = E[(X - μ)²]}.
     *
     * @return the variance (may be {@code NaN} or {@code +∞} if undefined)
     */
    double variance();

    /**
     * Theoretical range of the distribution: the interval
     * {@code [min, max]} outside which the PDF is identically zero.
     *
     * @return {@code Double2(lower, upper)}
     */
    Double2 range();

    /**
     * Support of the distribution: the closure of the set where
     * {@code pdf(x) > 0}. May coincide with {@link #range()} for
     * distributions with full-range support.
     *
     * @return {@code Double2(lower, upper)}
     */
    Double2 support();

    /**
     * Natural logarithm of the probability density (or mass) at {@code x}.
     *
     * <p>The default computes {@code Math.log(pdf(x))}. Concrete
     * distributions override with numerically-stable direct formulas
     * that avoid catastrophic cancellation for extreme inputs.
     *
     * @param x evaluation point
     * @return {@code log f(x)}, or {@link Double#NEGATIVE_INFINITY} when
     *         {@code pdf(x) == 0}
     */
    default double logPdf(double x) {
        double p = pdf(x);
        return p > 0.0 ? Math.log(p) : Double.NEGATIVE_INFINITY;
    }

    /**
     * Standard deviation: {@code √Var(X)}.
     *
     * @return the standard deviation
     * @throws ArithmeticException if variance is negative or NaN
     */
    default double standardDeviation() {
        double variance = variance();
        if (Double.isNaN(variance) || variance < 0.0) {
            throw new ArithmeticException("Standard deviation is undefined for variance: " + variance);
        }
        return Math.sqrt(variance);
    }

    // ── Batch evaluation ──────────────────────────────────────────────

    /**
     * Evaluate the given {@link Metric} over a strided input array,
     * writing results into {@code out[outOff..outOff+n)}.
     *
     * <p>Reads from {@code x[xOff + i * xStride]} for {@code i in [0, n)}
     * and writes to {@code out[outOff + i]}. Distributions override this
     * to provide optimised batch kernels that hoist normalising constants
     * out of the loop.
     *
     * @param metric   which function to evaluate
     * @param x        input array
     * @param xOff     first input index
     * @param xStride  stride between input elements
     * @param n        number of elements to evaluate
     * @param out      output array (length ≥ {@code outOff + n})
     * @param outOff   first output index
     */
    default void batch(Metric metric,
                         double[] x, int xOff, int xStride, int n,
                         double[] out, int outOff) {
        if (n == 0) return;
        switch (metric) {
            case LOG_PDF  -> { for (int i = 0; i < n; i++) out[outOff + i] = logPdf(x[xOff + i * xStride]); }
            case PDF      -> { for (int i = 0; i < n; i++) out[outOff + i] = pdf(x[xOff + i * xStride]); }
            case CDF      -> { for (int i = 0; i < n; i++) out[outOff + i] = cdf(x[xOff + i * xStride]); }
            case CCDF     -> { for (int i = 0; i < n; i++) out[outOff + i] = ccdf(x[xOff + i * xStride]); }
            case QUANTILE -> { for (int i = 0; i < n; i++) out[outOff + i] = quantile(x[xOff + i * xStride]); }
        }
    }

    // ── Sampling ──────────────────────────────────────────────────────

    /**
     * Draw a single random variate from this distribution.
     *
     * @param g random number generator
     * @return a sample from this distribution
     * @throws UnsupportedOperationException for improper distributions
     */
    double sample(RandomGenerator g);

    /**
     * Draw {@code n} random variates into a strided output buffer.
     *
     * <p>Writes {@code out[off + i * stride]} for {@code i in [0, n)}.
     * Implementations should hoist distribution-scoped constants so they
     * are not recomputed per element.
     *
     * @param g      random number generator
     * @param n      number of samples to draw
     * @param out    output array (length ≥ {@code off + (n-1) * stride + 1})
     * @param off    first output index
     * @param stride stride between output elements
     */
    default void sample(RandomGenerator g, int n, double[] out, int off, int stride) {
        if (n == 0) return;
        for (int i = 0; i < n; i++) {
            out[off + i * stride] = sample(g);
        }
    }

    // ── Gamma sampling utilities ──────────────────────────────────────

    /**
     * Draw a single sample from {@code Gamma(shape, scale = 1.0)}
     * using the Marsaglia-Tsang method with squeeze acceleration.
     *
     * <p>For {@code shape ≥ 1} this is a direct application of the
     * algorithm. For {@code shape < 1} it uses the Johnk boost:
     * {@code Gamma(shape) = Gamma(shape+1) · U^(1/shape)}.
     *
     * @param g     random generator
     * @param shape shape parameter (α), must be strictly positive
     * @return a sample from Gamma(shape, 1)
     */
    static double nextGamma(RandomGenerator g, double shape) {
        if (shape >= 1.0) {
            return marsagliaTsang(g, shape);
        }
        double y = marsagliaTsang(g, shape + 1.0);
        return y * Math.pow(g.nextDouble(), 1.0 / shape);
    }

    /**
     * Fill {@code out[off + i*stride]} for {@code i in [0, n)} with
     * iid {@code Gamma(shape, 1.0)} draws.
     *
     * <p>Pre-computes the Marsaglia-Tsang constants {@code d} and
     * {@code c} once, then reuses them across all {@code n} draws.
     *
     * @param g      random generator
     * @param shape  shape parameter (α), must be strictly positive
     * @param n      number of samples
     * @param out    output array
     * @param off    first output index
     * @param stride stride between output elements
     */
    static void nextGamma(RandomGenerator g, double shape, int n,
                          double[] out, int off, int stride) {
        if (n == 0) return;
        if (shape >= 1.0) {
            double d = shape - 1.0 / 3.0;
            double c = 1.0 / Math.sqrt(9.0 * d);
            for (int i = 0; i < n; i++) {
                out[off + i * stride] = marsagliaTsangCore(g, d, c);
            }
        } else {
            double shiftedShape = shape + 1.0;
            double d = shiftedShape - 1.0 / 3.0;
            double c = 1.0 / Math.sqrt(9.0 * d);
            double invShape = 1.0 / shape;
            for (int i = 0; i < n; i++) {
                double y = marsagliaTsangCore(g, d, c);
                out[off + i * stride] = y * Math.pow(g.nextDouble(), invShape);
            }
        }
    }

    // ── Marsaglia-Tsang core (private) ────────────────────────────────

    private static double marsagliaTsang(RandomGenerator g, double shape) {
        double d = shape - 1.0 / 3.0;
        double c = 1.0 / Math.sqrt(9.0 * d);
        return marsagliaTsangCore(g, d, c);
    }

    /**
     * Marsaglia-Tsang transformed rejection with squeeze.
     * Generates a cube of a shifted normal; the squeeze test
     * accepts ~95% of candidates without computing any logarithm.
     */
    private static double marsagliaTsangCore(RandomGenerator g, double d, double c) {
        for (;;) {
            double x, v;
            do {
                x = g.nextGaussian();
                v = 1.0 + c * x;
            } while (v <= 0.0);
            v = v * v * v;
            double u = g.nextDouble();
            double xSq = x * x;
            if (u < 1.0 - 0.0331 * xSq * xSq) {
                return d * v;
            }
            if (Math.log(u) < 0.5 * xSq + d * (1.0 - v + Math.log(v))) {
                return d * v;
            }
        }
    }
}
