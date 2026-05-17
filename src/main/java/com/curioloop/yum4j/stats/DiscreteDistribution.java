package com.curioloop.yum4j.stats;

/**
 * Marker interface for discrete distributions.
 *
 * <p>{@link #logPdf(double)} is interpreted as the log probability mass;
 * inputs not in the support return {@link Double#NEGATIVE_INFINITY}.
 * Batch sampling writes exact whole-number {@code double} values.
 */
public interface DiscreteDistribution extends Distribution {

    @Override
    default double logPdf(double x) {
        double p = pdf(x);
        return p > 0.0 ? Math.log(p) : Double.NEGATIVE_INFINITY;
    }

    @Override
    default double sample(java.util.random.RandomGenerator g) {
        return quantile(g.nextDouble());
    }
}
