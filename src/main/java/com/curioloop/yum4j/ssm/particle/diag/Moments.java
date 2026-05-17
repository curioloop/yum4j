package com.curioloop.yum4j.ssm.particle.diag;

import com.curioloop.yum4j.ssm.particle.engine.RunState;
import com.curioloop.yum4j.ssm.particle.engine.Workspace;
import com.curioloop.yum4j.math.VectorOps;

import java.util.Arrays;

/**
 * Per-step weighted moments of the particle cloud.
 *
 * <p>After every step {@code t} this collector computes, for each
 * dimension {@code j in [0, d)}:
 *
 * <pre>
 *   mean[t][j] = Σ_n W_n · X[n * dim + j]
 *   var[t][j]  = Σ_n W_n · X[n * dim + j]^2 − mean[t][j]^2
 * </pre>
 *
 * where {@code W_n = exp(lw_n − max_k lw_k) / Σ_k exp(lw_k − max_k lw_k)}
 * is the normalised weight vector. Buffers are flat arrays of length
 * {@code T * dim} accessed as {@code [t * dim + j]}.
 *
 * <p>Access the outputs via {@link #means()} and {@link #variances()}.
 *
 * <p>The normalised-weight vector is computed once per step and reused
 * across all dimensions, reducing the number of {@code exp} calls from
 * {@code d·N} to {@code N}.
 *
 */
public final class Moments implements Collector {

    private double[] means;
    private double[] vars;
    private double[] Wscratch;
    private int dim;
    private int N;
    private int T;

    @Override
    public void attach(Workspace ws, RunState rs) {
        this.dim = ws.dim;
        this.N = ws.N;
        this.T = Math.max(1, rs.T);
        int len = T * dim;
        if (means == null || means.length != len) {
            means = new double[len];
            vars = new double[len];
        }
        if (Wscratch == null || Wscratch.length < N) {
            Wscratch = new double[N];
        }
    }

    @Override
    public void afterMutation(Workspace ws, RunState rs, int t) {
        if (t < 0 || t >= T) return;
        final double[] lw = ws.logW;
        final double[] W = Wscratch;

        // Compute normalised weights once.
        double m = VectorOps.max(lw, 0, N);
        if (m == Double.NEGATIVE_INFINITY) {
            double uniform = 1.0 / N;
            Arrays.fill(W, 0, N, uniform);
        } else {
            double sumW = 0.0;
            for (int n = 0; n < N; n++) {
                double w = Math.exp(lw[n] - m);
                W[n] = w;
                sumW += w;
            }
            VectorOps.scal(1.0 / sumW, W, 0, N);
        }

        final double[] X = ws.X;
        final int base = t * dim;
        for (int j = 0; j < dim; j++) {
            double acc1 = 0.0;
            double acc2 = 0.0;
            for (int n = 0; n < N; n++) {
                double w = W[n];
                double x = X[n * dim + j];
                acc1 += w * x;
                acc2 += w * x * x;
            }
            means[base + j] = acc1;
            vars[base + j] = acc2 - acc1 * acc1;
        }
    }

    /** Per-step weighted means, flat array of length {@code T * dim}. Access as {@code means[t * dim + j]}. {@code null} before attach. */
    public double[] means() { return means; }

    /** Per-step weighted variances, flat array of length {@code T * dim}. Access as {@code vars[t * dim + j]}. {@code null} before attach. */
    public double[] variances() { return vars; }

    /** Number of time steps. */
    public int T() { return T; }

    /** State dimension. */
    public int dim() { return dim; }
}
