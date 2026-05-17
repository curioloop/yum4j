package com.curioloop.yum4j.ssm.particle.dist;

import com.curioloop.yum4j.stats.Distribution;

import java.util.random.RandomGenerator;

/**
 * Product of independent univariate distributions acting as a
 * {@code d}-dimensional multivariate distribution. Each component
 * governs one dimension of the {@code (d, N)} column-major particle
 * buffer.
 *
 * <p>Every component must be a {@link Distribution} that also
 * implements {@link Distribution}. All concrete
 * {@code ContinuousDistribution} and {@code DiscreteDistribution}
 * records in the {@code com.curioloop.yum4j.stats} package satisfy
 * this out of the box; passing a distribution that does not will
 * raise {@link IllegalArgumentException} at construction.
 *
 * <p>Sampling delegates per-row to
 * {@link Distribution#sample(RandomGenerator, int, double[], int, int)}
 * with {@code stride = 1} since each row of the column-major buffer is
 * stride-1 contiguous. The log-pdf accumulates per-row log densities
 * into {@code out}; it requires a scratch row of {@code N} doubles
 * which can be a slice of the caller scratch or allocated internally.
 */
public record IndependentProductDistribution(Distribution... components) implements MultivariateDistribution {

    public IndependentProductDistribution(Distribution... components) {
        if (components == null || components.length == 0) {
            throw new IllegalArgumentException("components must be non-empty");
        }
        this.components = components.clone();
    }

    @Override
    public int dim() {
        return components.length;
    }

    /**
     * The components array (shared reference — do not mutate).
     */
    @Override
    public Distribution[] components() {
        return components;
    }

    @Override
    public void mean(double[] out) {
        for (int j = 0; j < components.length; j++) {
            out[j] = components[j].mean();
        }
    }

    @Override
    public void logPdfBatch(double[] x, int xOff, int N,
                            double[] out, int outOff,
                            double[] scratch) {
        if (N == 0) return;
        // Row scratch of length N: reuse a slice of caller scratch if it can
        // hold at least N doubles; otherwise allocate.
        double[] row = (scratch != null && scratch.length >= N) ? scratch : new double[N];

        // Zero the output range so per-row accumulation is clean.
        for (int n = 0; n < N; n++) {
            out[outOff + n] = 0.0;
        }

        for (int j = 0; j < components.length; j++) {
            // Read N stride-1 values starting at x[xOff + j*N]; write contiguous into row.
            components[j].batch(Distribution.Metric.LOG_PDF, x, xOff + j * N, 1, N, row, 0);
            for (int n = 0; n < N; n++) {
                out[outOff + n] += row[n];
            }
        }
    }

    @Override
    public void sample(RandomGenerator g, int N,
                       double[] x, int xOff,
                       double[] scratch) {
        if (N == 0) return;
        for (int j = 0; j < components.length; j++) {
            components[j].sample(g, N, x, xOff + j * N, 1);
        }
    }
}
