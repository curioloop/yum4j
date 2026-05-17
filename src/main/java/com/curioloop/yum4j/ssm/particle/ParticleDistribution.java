package com.curioloop.yum4j.ssm.particle;

import java.util.Arrays;

/**
 * Lightweight immutable wrapper around the final-step particle snapshot
 * (states and log-weights) from a completed filter run.
 *
 * <p>All accessors return defensive copies — callers cannot mutate the
 * internal state. This record is the typed replacement for raw
 * {@code Workspace.X} / {@code Workspace.logW} access.
 *
 * @param <Y> observation type (carried for type consistency with the filter)
 */
public final class ParticleDistribution<Y> {

    private final int N;
    private final int dim;
    private final double[] particles;  // length N * dim
    private final double[] logWeights; // length N

    /**
     * Constructs a particle distribution from a snapshot of the final step.
     *
     * <p>The provided arrays are defensively copied — subsequent mutations
     * to the source arrays do not affect this distribution.
     *
     * @param N          number of particles
     * @param dim        state dimension per particle
     * @param particles  particle states, length {@code N * dim}
     * @param logWeights per-particle log-weights, length {@code N}
     * @throws IllegalArgumentException if array lengths do not match
     *         {@code N * dim} and {@code N} respectively
     */
    public ParticleDistribution(int N, int dim, double[] particles, double[] logWeights) {
        if (N <= 0) throw new IllegalArgumentException("N must be > 0: " + N);
        if (dim <= 0) throw new IllegalArgumentException("dim must be > 0: " + dim);
        if (particles == null) throw new IllegalArgumentException("particles must not be null");
        if (logWeights == null) throw new IllegalArgumentException("logWeights must not be null");
        if (particles.length != N * dim) {
            throw new IllegalArgumentException(
                "particles.length must be N*dim=" + (N * dim) + ", got " + particles.length);
        }
        if (logWeights.length != N) {
            throw new IllegalArgumentException(
                "logWeights.length must be N=" + N + ", got " + logWeights.length);
        }
        this.N = N;
        this.dim = dim;
        this.particles = Arrays.copyOf(particles, particles.length);
        this.logWeights = Arrays.copyOf(logWeights, logWeights.length);
    }

    /** Number of particles. */
    public int N() { return N; }

    /** State dimension per particle. */
    public int dim() { return dim; }

    /**
     * Returns a defensive copy of the particle states.
     * Layout: length {@code N * dim}, with particle {@code n} at
     * indices {@code [n*dim .. (n+1)*dim)} for row-major, or
     * dimension {@code j} of particle {@code n} at {@code j*N + n}
     * for column-major (matching the engine's internal layout).
     *
     * @return copy of particle states, length {@code N * dim}
     */
    public double[] particles() {
        return Arrays.copyOf(particles, particles.length);
    }

    /**
     * Returns a defensive copy of the per-particle log-weights.
     *
     * @return copy of log-weights, length {@code N}
     */
    public double[] logWeights() {
        return Arrays.copyOf(logWeights, logWeights.length);
    }

    /**
     * Computes the weighted mean of the particle distribution for a
     * 1-dimensional state space ({@code dim == 1}).
     *
     * <p>The mean is computed as:
     * <pre>
     *   E[X] = Σ_n w_n * X_n / Σ_n w_n
     * </pre>
     * where {@code w_n = exp(logW_n - max(logW))}.
     *
     * @return the weighted mean
     * @throws IllegalStateException if {@code dim != 1}
     */
    public double weightedMean() {
        if (dim != 1) {
            throw new IllegalStateException(
                "weightedMean() is only defined for dim=1, got dim=" + dim);
        }
        double maxLw = Double.NEGATIVE_INFINITY;
        for (int n = 0; n < N; n++) {
            if (logWeights[n] > maxLw) maxLw = logWeights[n];
        }
        if (maxLw == Double.NEGATIVE_INFINITY) {
            // All weights are -inf; return uniform average
            double sum = 0.0;
            for (int n = 0; n < N; n++) sum += particles[n];
            return sum / N;
        }
        double sumW = 0.0;
        double sumWX = 0.0;
        for (int n = 0; n < N; n++) {
            double w = Math.exp(logWeights[n] - maxLw);
            sumW += w;
            sumWX += w * particles[n];
        }
        return sumWX / sumW;
    }

    /**
     * Computes the weighted mean vector for a multi-dimensional state space.
     *
     * <p>Returns a length-{@code dim} array where element {@code j} is:
     * <pre>
     *   E[X_j] = Σ_n w_n * X_{j,n} / Σ_n w_n
     * </pre>
     *
     * <p>Assumes column-major layout: dimension {@code j} of particle
     * {@code n} is at index {@code j * N + n}.
     *
     * @return weighted mean vector, length {@code dim}
     */
    public double[] weightedMeanVector() {
        double maxLw = Double.NEGATIVE_INFINITY;
        for (int n = 0; n < N; n++) {
            if (logWeights[n] > maxLw) maxLw = logWeights[n];
        }

        double[] weights = new double[N];
        double sumW = 0.0;
        if (maxLw == Double.NEGATIVE_INFINITY) {
            // Uniform weights
            Arrays.fill(weights, 1.0);
            sumW = N;
        } else {
            for (int n = 0; n < N; n++) {
                weights[n] = Math.exp(logWeights[n] - maxLw);
                sumW += weights[n];
            }
        }

        double[] mean = new double[dim];
        for (int j = 0; j < dim; j++) {
            double acc = 0.0;
            int rowOff = j * N;
            for (int n = 0; n < N; n++) {
                acc += weights[n] * particles[rowOff + n];
            }
            mean[j] = acc / sumW;
        }
        return mean;
    }

    /**
     * Computes the normalised weights (summing to 1) from the log-weights.
     *
     * @return normalised weights, length {@code N}
     */
    public double[] normalisedWeights() {
        double maxLw = Double.NEGATIVE_INFINITY;
        for (int n = 0; n < N; n++) {
            if (logWeights[n] > maxLw) maxLw = logWeights[n];
        }
        double[] w = new double[N];
        if (maxLw == Double.NEGATIVE_INFINITY) {
            Arrays.fill(w, 1.0 / N);
            return w;
        }
        double sum = 0.0;
        for (int n = 0; n < N; n++) {
            w[n] = Math.exp(logWeights[n] - maxLw);
            sum += w[n];
        }
        for (int n = 0; n < N; n++) {
            w[n] /= sum;
        }
        return w;
    }
}
