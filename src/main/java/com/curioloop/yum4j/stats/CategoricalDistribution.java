package com.curioloop.yum4j.stats;

import com.curioloop.yum4j.math.Double2;

import java.util.random.RandomGenerator;

/**
 * Discrete distribution over the finite support {@code {0, 1, ..., K-1}}
 * with user-supplied non-negative weights that sum to one (unnormalised
 * inputs are accepted and normalised internally).
 *
 * <p>The distribution backs the categorical sampling step of the
 * mixture and resampling paths in {@code filter.particle}. Sampling is
 * O(1) via a {@link #aliasProb alias} table built at construction
 * (Walker, 1977).
 *
 * <h3>Lazy accessors</h3>
 *
 * <p>{@link #logProbs()} and the cumulative array used by
 * {@link #cdf(double)} / {@link #quantile(double)} are computed on the
 * first access and cached. This makes the category-count-independent
 * methods ({@link #pdf}, {@link #sample}, {@link #mean}, {@link #variance})
 * fully constant-time while keeping construction cheap.
 *
 * <p>This class is not a {@code value record} because those lazy caches
 * demand a mutable field.
 */
public final class CategoricalDistribution implements DiscreteDistribution {

    /** Normalised probabilities. */
    private final double[] probs;

    /**
     * Walker alias table: {@code prob[k]} is the probability of
     * accepting slot {@code k}; otherwise the draw is re-mapped to
     * {@code alias[k]}.
     */
    private final double[] aliasProb;

    private final int[] alias;

    // Lazy caches (volatile for safe publication under benign races).
    private volatile double[] logProbsCache;
    private volatile double[] cumulativeCache;

    /**
     * @param probs non-negative weights; at least one positive. The
     *              array is defensively copied and normalised so that
     *              {@link #probs()} sums to one.
     */
    public CategoricalDistribution(double[] probs) {
        if (probs == null) {
            throw new IllegalArgumentException("probs must not be null");
        }
        final int k = probs.length;
        if (k == 0) {
            throw new IllegalArgumentException("probs must be non-empty");
        }
        double sum = 0.0;
        for (int i = 0; i < k; i++) {
            double p = probs[i];
            if (Double.isNaN(p) || p < 0.0 || Double.isInfinite(p)) {
                throw new IllegalArgumentException(
                    "probs[" + i + "] must be finite and non-negative: " + p
                );
            }
            sum += p;
        }
        if (!(sum > 0.0)) {
            throw new IllegalArgumentException(
                "probs must have at least one positive weight; sum=" + sum
            );
        }
        double[] normalised = new double[k];
        double invSum = 1.0 / sum;
        for (int i = 0; i < k; i++) {
            normalised[i] = probs[i] * invSum;
        }
        this.probs = normalised;

        // Walker alias table. Scales probabilities by K and maintains
        // "small" (prob < 1) and "large" (prob >= 1) buckets. See
        // Vose (1991) for the stable variant used here.
        double[] aprob = new double[k];
        int[] aidx = new int[k];
        double[] scaled = new double[k];
        int[] small = new int[k];
        int[] large = new int[k];
        int sTop = 0, lTop = 0;
        for (int i = 0; i < k; i++) {
            scaled[i] = normalised[i] * k;
            if (scaled[i] < 1.0) {
                small[sTop++] = i;
            } else {
                large[lTop++] = i;
            }
        }
        while (sTop > 0 && lTop > 0) {
            int s = small[--sTop];
            int l = large[--lTop];
            aprob[s] = scaled[s];
            aidx[s] = l;
            scaled[l] = (scaled[l] + scaled[s]) - 1.0;
            if (scaled[l] < 1.0) {
                small[sTop++] = l;
            } else {
                large[lTop++] = l;
            }
        }
        // Remaining buckets have prob == 1 (modulo floating-point
        // drift); fix them up so the split < prob[k] test always
        // returns the slot index.
        while (lTop > 0) {
            int l = large[--lTop];
            aprob[l] = 1.0;
            aidx[l] = l;
        }
        while (sTop > 0) {
            int s = small[--sTop];
            aprob[s] = 1.0;
            aidx[s] = s;
        }
        this.aliasProb = aprob;
        this.alias = aidx;
    }

    /** Number of categories {@code K}. */
    public int categories() {
        return probs.length;
    }

    /** Defensive copy of the normalised probability vector. */
    public double[] probs() {
        return probs.clone();
    }

    /** Lazily cached element-wise {@code log(probs())}. */
    public double[] logProbs() {
        return logProbsInternal().clone();
    }

    private double[] logProbsInternal() {
        double[] cached = logProbsCache;
        if (cached != null) {
            return cached;
        }
        final int k = probs.length;
        double[] lp = new double[k];
        for (int i = 0; i < k; i++) {
            lp[i] = probs[i] > 0.0 ? Math.log(probs[i]) : Double.NEGATIVE_INFINITY;
        }
        logProbsCache = lp;
        return lp;
    }

    private double[] cumulativeInternal() {
        double[] cached = cumulativeCache;
        if (cached != null) {
            return cached;
        }
        final int k = probs.length;
        double[] cum = new double[k];
        double acc = 0.0;
        for (int i = 0; i < k; i++) {
            acc += probs[i];
            cum[i] = acc;
        }
        // Force the last entry to exactly 1.0 so quantile is closed.
        cum[k - 1] = 1.0;
        cumulativeCache = cum;
        return cum;
    }

    @Override
    public Double2 range() {
        return Double2.bound(0.0, probs.length - 1.0);
    }

    @Override
    public Double2 support() {
        return Double2.bound(0.0, probs.length - 1.0);
    }

    @Override
    public double pdf(double x) {
        if (Double.isNaN(x) || !isIntegralPoint(x)) {
            return 0.0;
        }
        int k = (int) x;
        if (k < 0 || k >= probs.length) {
            return 0.0;
        }
        return probs[k];
    }

    @Override
    public double cdf(double x) {
        if (Double.isNaN(x)) {
            throw new IllegalArgumentException("x must not be NaN");
        }
        if (x < 0.0) {
            return 0.0;
        }
        double[] cum = cumulativeInternal();
        int floor = x >= cum.length - 1 ? cum.length - 1 : (int) Math.floor(x);
        return cum[floor];
    }

    @Override
    public double ccdf(double x) {
        return 1.0 - cdf(x);
    }

    @Override
    public double quantile(double probability) {
        validateProbability(probability);
        double[] cum = cumulativeInternal();
        if (probability == 0.0) {
            // Find the smallest index with non-zero probability.
            for (int i = 0; i < probs.length; i++) {
                if (probs[i] > 0.0) return i;
            }
            return 0.0;
        }
        if (probability >= 1.0) {
            return probs.length - 1.0;
        }
        // Binary search for the first index whose cumulative >= probability.
        int lo = 0, hi = cum.length - 1;
        while (lo < hi) {
            int mid = (lo + hi) >>> 1;
            if (cum[mid] >= probability) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        return lo;
    }

    @Override
    public double quantileUpperTail(double probability) {
        return quantile(1.0 - probability);
    }

    @Override
    public double mean() {
        double m = 0.0;
        for (int i = 0; i < probs.length; i++) {
            m += i * probs[i];
        }
        return m;
    }

    @Override
    public double variance() {
        double m = mean();
        double v = 0.0;
        for (int i = 0; i < probs.length; i++) {
            double d = i - m;
            v += probs[i] * d * d;
        }
        return v;
    }

    // ---------------------------------------------------------------
    // Batch / Sampling specialisations
    // ---------------------------------------------------------------

    @Override
    public double logPdf(double x) {
        if (Double.isNaN(x) || !isIntegralPoint(x)) {
            return Double.NEGATIVE_INFINITY;
        }
        int k = (int) x;
        if (k < 0 || k >= probs.length) {
            return Double.NEGATIVE_INFINITY;
        }
        return logProbsInternal()[k];
    }

    @Override
    public void batch(Metric metric, double[] x, int xOff, int xStride, int n,
                            double[] out, int outOff) {
        if (metric == Metric.LOG_PDF) {
            if (n == 0) return;
            final double[] lp = logProbsInternal();
            final int K = lp.length;
            for (int i = 0; i < n; i++) {
                double v = x[xOff + i * xStride];
                double r;
                if (Double.isNaN(v) || !isIntegralPoint(v)) {
                    r = Double.NEGATIVE_INFINITY;
                } else {
                    int k = (int) v;
                    r = (k < 0 || k >= K) ? Double.NEGATIVE_INFINITY : lp[k];
                }
                out[outOff + i] = r;
            }
        } else {
            DiscreteDistribution.super.batch(metric, x, xOff, xStride, n, out, outOff);
        }
    }

    @Override
    public double sample(RandomGenerator g) {
        final int K = probs.length;
        double u = g.nextDouble();
        double scaled = u * K;
        int k = (int) scaled;
        if (k >= K) {
            // Floating-point guard: nextDouble() < 1.0 so this is
            // unreachable in practice, but defend against it anyway.
            k = K - 1;
        }
        double frac = scaled - k;
        return frac < aliasProb[k] ? k : alias[k];
    }

    @Override
    public void sample(RandomGenerator g, int n, double[] out, int off, int stride) {
        if (n == 0) return;
        final int K = probs.length;
        final double[] ap = aliasProb;
        final int[] ai = alias;
        for (int i = 0; i < n; i++) {
            double u = g.nextDouble();
            double scaled = u * K;
            int k = (int) scaled;
            if (k >= K) k = K - 1;
            double frac = scaled - k;
            int drawn = frac < ap[k] ? k : ai[k];
            out[off + i * stride] = drawn;
        }
    }

    // ---------------------------------------------------------------

    private static boolean isIntegralPoint(double x) {
        return x >= Integer.MIN_VALUE && x <= Integer.MAX_VALUE && x == Math.rint(x);
    }

    private static void validateProbability(double probability) {
        if (Double.isNaN(probability) || probability < 0.0 || probability > 1.0) {
            throw new IllegalArgumentException("probability must be in [0, 1]: " + probability);
        }
    }
}
