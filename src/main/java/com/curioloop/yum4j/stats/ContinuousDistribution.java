package com.curioloop.yum4j.stats;

/**
 * Marker interface for continuous distributions.
 *
 * <p>Provides default implementations of {@link #logPdf} and
 * {@link #sample} suitable for most continuous distributions.
 * Performance-critical distributions override these with direct
 * formulas and specialised sampling kernels.
 */
public interface ContinuousDistribution extends Distribution {

    @Override
    default double logPdf(double x) {
        double p = pdf(x);
        return p > 0.0 ? Math.log(p) : Double.NEGATIVE_INFINITY;
    }

    /**
     * Default sampling via inverse-CDF: draw {@code u ~ U(0, 1)},
     * return {@link #quantile(double) quantile(u)}.
     */
    @Override
    default double sample(java.util.random.RandomGenerator g) {
        return quantile(g.nextDouble());
    }
}
