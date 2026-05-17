package com.curioloop.yum4j.ssm.particle.dist;

import java.util.random.RandomGenerator;

/**
 * Finite mixture of {@link MultivariateDistribution} components. All
 * components must share the same dimension.
 *
 * <p>Weights may be supplied unnormalised but must be non-negative and
 * finite; they are normalised to sum to one at construction. Sampling
 * uses a cumulative-weight inverse-CDF draw per particle (binary
 * search) since Mixture is not on the hot particle-filter path. Each
 * particle is written into the target buffer slot by delegating to the
 * selected component's {@code sample} call with {@code N = 1}.
 *
 * <p>{@link #logPdfBatch} computes a log-sum-exp over component
 * log-pdfs. The caller may supply a scratch of at least
 * {@code components.length * N + dim * N} doubles to avoid
 * allocation; otherwise the method allocates internally.
 */
public final class MixtureDistribution implements MultivariateDistribution {

    private final double[] weights;       // normalised
    private final double[] logWeights;    // log of normalised weights
    private final double[] cumWeights;    // cumulative sum, length K
    private final MultivariateDistribution[] components;
    private final int dim;

    public MixtureDistribution(double[] weights, MultivariateDistribution... components) {
        if (components == null || components.length == 0) {
            throw new IllegalArgumentException("components must be non-empty");
        }
        if (weights == null || weights.length != components.length) {
            throw new IllegalArgumentException(
                "weights length must equal components length (" + components.length
                    + "); got " + (weights == null ? "null" : weights.length));
        }
        int d = components[0].dim();
        for (int k = 0; k < components.length; k++) {
            if (components[k] == null) {
                throw new IllegalArgumentException("components[" + k + "] is null");
            }
            if (components[k].dim() != d) {
                throw new IllegalArgumentException(
                    "components must all have the same dim; components[0].dim()="
                        + d + " but components[" + k + "].dim()=" + components[k].dim());
            }
        }
        double sum = 0.0;
        for (int k = 0; k < weights.length; k++) {
            double w = weights[k];
            if (Double.isNaN(w) || w < 0.0 || Double.isInfinite(w)) {
                throw new IllegalArgumentException(
                    "weights must be finite non-negative; weights[" + k + "]=" + w);
            }
            sum += w;
        }
        if (!(sum > 0.0)) {
            throw new IllegalArgumentException("weights sum must be > 0");
        }
        this.components = components.clone();
        this.weights = new double[weights.length];
        this.logWeights = new double[weights.length];
        this.cumWeights = new double[weights.length];
        double cum = 0.0;
        for (int k = 0; k < weights.length; k++) {
            double w = weights[k] / sum;
            this.weights[k] = w;
            this.logWeights[k] = w > 0.0 ? Math.log(w) : Double.NEGATIVE_INFINITY;
            cum += w;
            this.cumWeights[k] = cum;
        }
        // Force the last cumulative weight to exactly 1.0 to guard against
        // floating-point rounding above 1 when the input sum was already
        // normalised.
        this.cumWeights[weights.length - 1] = 1.0;
        this.dim = d;
    }

    @Override
    public int dim() {
        return dim;
    }

    public double[] weights() {
        return weights;
    }

    public MultivariateDistribution[] components() {
        return components;
    }

    @Override
    public void mean(double[] out) {
        // Weighted combination of component means.
        double[] tmp = new double[dim];
        java.util.Arrays.fill(out, 0, dim, 0.0);
        for (int k = 0; k < components.length; k++) {
            components[k].mean(tmp);
            double w = weights[k];
            for (int j = 0; j < dim; j++) {
                out[j] += w * tmp[j];
            }
        }
    }

    @Override
    public void logPdfBatch(double[] x, int xOff, int N,
                            double[] out, int outOff,
                            double[] scratch) {
        if (N == 0) return;
        final int K = components.length;
        final int d = dim;
        final int compScratchSize = K * N;
        final int workSize = d * N;

        // compLogs holds component log-pdfs row-wise (row k at offset k*N);
        // componentWork is the (d, N) scratch passed into each component.
        // We can share the caller scratch across both segments when it is
        // long enough; otherwise allocate.
        double[] compLogs;
        double[] componentWork;
        if (scratch != null && scratch.length >= compScratchSize + workSize) {
            compLogs = scratch;
            // Allocate a dedicated component work buffer so it cannot alias
            // the log-pdf rows stored in `compLogs`. Mixture is deliberately
            // off the hottest particle-filter path; this small allocation is
            // an accepted cost for correctness.
            componentWork = new double[workSize];
        } else {
            compLogs = new double[compScratchSize];
            componentWork = new double[workSize];
        }

        // Fill row k with component k's log-pdf + log weight_k.
        for (int k = 0; k < K; k++) {
            components[k].logPdfBatch(x, xOff, N, compLogs, k * N, componentWork);
            double lw = logWeights[k];
            int rowOff = k * N;
            for (int n = 0; n < N; n++) {
                compLogs[rowOff + n] = compLogs[rowOff + n] + lw;
            }
        }

        // Column-wise log-sum-exp.
        for (int n = 0; n < N; n++) {
            double max = Double.NEGATIVE_INFINITY;
            for (int k = 0; k < K; k++) {
                double v = compLogs[k * N + n];
                if (v > max) max = v;
            }
            if (max == Double.NEGATIVE_INFINITY) {
                out[outOff + n] = Double.NEGATIVE_INFINITY;
                continue;
            }
            double s = 0.0;
            for (int k = 0; k < K; k++) {
                s += Math.exp(compLogs[k * N + n] - max);
            }
            out[outOff + n] = max + Math.log(s);
        }
    }

    @Override
    public void sample(RandomGenerator g, int N,
                       double[] x, int xOff,
                       double[] scratch) {
        if (N == 0) return;
        final int d = dim;
        // We draw one particle at a time and need a d-sized column scratch to
        // receive the component's batch(N=1) output before scattering it into
        // the stride-N slots of x. Reuse the caller scratch only if it is at
        // least d doubles long; otherwise allocate a tiny per-call buffer.
        double[] column = (scratch != null && scratch.length >= d) ? scratch : new double[d];

        for (int n = 0; n < N; n++) {
            int k = drawComponent(g);
            // Draw one column via the component's batch(N=1) form.
            components[k].sample(g, 1, column, 0, null);
            for (int j = 0; j < d; j++) {
                x[xOff + j * N + n] = column[j];
            }
        }
    }

    /**
     * Inverse-CDF component selection via a simple linear scan; binary
     * search would help for large K but Mixture is not hot here.
     */
    private int drawComponent(RandomGenerator g) {
        double u = g.nextDouble();
        double[] cum = cumWeights;
        for (int k = 0; k < cum.length; k++) {
            if (u < cum[k]) return k;
        }
        return cum.length - 1;
    }
}
