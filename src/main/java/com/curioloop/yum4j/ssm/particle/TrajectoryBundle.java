package com.curioloop.yum4j.ssm.particle;

import com.curioloop.yum4j.ssm.particle.smooth.ParticleHistory;

import java.util.Arrays;

/**
 * Immutable wrapper around smoothed trajectory indices produced by
 * backward-sampling smoothers (FFBS, TwoFilter, FixedLag, Paris).
 *
 * <p>The internal array is laid out row-major: element
 * {@code indices[t * M + m]} is the particle index at time {@code t}
 * for trajectory {@code m}. To reconstruct actual state values, the
 * caller copies particles via {@code particleHistory.viewX(t, scratch, 0)}
 * and indexes into the scratch buffer for the
 * appropriate particle.
 *
 * <p>All accessors return defensive copies — callers cannot mutate the
 * internal state.
 *
 * @param <Y> observation type (carried for type consistency)
 */
public final class TrajectoryBundle<Y> {

    private final int T;
    private final int M;
    private final int[] indices;  // length T * M, row-major
    private final ParticleHistory history;

    /**
     * Constructs a trajectory bundle from smoother output.
     *
     * @param T       number of time steps
     * @param M       number of trajectories
     * @param indices trajectory indices, length {@code T * M}, row-major
     * @param history the particle history used for state reconstruction
     *                (may be null if state reconstruction is not needed)
     * @throws IllegalArgumentException if indices length does not match
     *         {@code T * M}
     */
    public TrajectoryBundle(int T, int M, int[] indices, ParticleHistory history) {
        if (T <= 0) throw new IllegalArgumentException("T must be > 0: " + T);
        if (M <= 0) throw new IllegalArgumentException("M must be > 0: " + M);
        if (indices == null) throw new IllegalArgumentException("indices must not be null");
        if (indices.length != T * M) {
            throw new IllegalArgumentException(
                "indices.length must be T*M=" + (T * M) + ", got " + indices.length);
        }
        this.T = T;
        this.M = M;
        this.indices = Arrays.copyOf(indices, indices.length);
        this.history = history;
    }

    /** Number of time steps. */
    public int T() { return T; }

    /** Number of trajectories. */
    public int M() { return M; }

    /**
     * Returns a defensive copy of the full trajectory index array.
     * Layout: row-major, element {@code [t * M + m]} is the particle
     * index at time {@code t} for trajectory {@code m}.
     *
     * @return copy of trajectory indices, length {@code T * M}
     */
    public int[] indices() {
        return Arrays.copyOf(indices, indices.length);
    }

    /**
     * Returns the particle index at time {@code t} for trajectory {@code m}.
     *
     * @param t time step in {@code [0, T)}
     * @param m trajectory index in {@code [0, M)}
     * @return particle index at (t, m)
     * @throws IndexOutOfBoundsException if t or m is out of range
     */
    public int index(int t, int m) {
        if (t < 0 || t >= T) throw new IndexOutOfBoundsException("t=" + t + " not in [0, " + T + ")");
        if (m < 0 || m >= M) throw new IndexOutOfBoundsException("m=" + m + " not in [0, " + M + ")");
        return indices[t * M + m];
    }

    /**
     * Returns the trajectory indices at time {@code t} for all M trajectories.
     *
     * @param t time step in {@code [0, T)}
     * @return defensive copy of indices at time t, length {@code M}
     * @throws IndexOutOfBoundsException if t is out of range
     */
    public int[] indicesAt(int t) {
        if (t < 0 || t >= T) throw new IndexOutOfBoundsException("t=" + t + " not in [0, " + T + ")");
        int[] row = new int[M];
        System.arraycopy(indices, t * M, row, 0, M);
        return row;
    }

    /**
     * Returns the particle history associated with this bundle, for
     * reconstructing actual state values from the trajectory indices.
     *
     * @return the particle history, or null if not available
     */
    public ParticleHistory particleHistory() {
        return history;
    }

    /**
     * Reconstructs the state trajectory for trajectory {@code m} at
     * time {@code t} using the associated particle history.
     *
     * <p>Returns a length-{@code dim} array containing the state of
     * the particle at the trajectory's index for time {@code t}.
     *
     * @param t   time step in {@code [0, T)}
     * @param m   trajectory index in {@code [0, M)}
     * @param dim state dimension per particle
     * @param N   number of particles
     * @return state vector at (t, m), length {@code dim}
     * @throws IllegalStateException if no particle history is available
     * @throws IndexOutOfBoundsException if t or m is out of range
     */
    public double[] stateAt(int t, int m, int dim, int N) {
        if (history == null) {
            throw new IllegalStateException("No particle history available for state reconstruction");
        }
        int idx = index(t, m);
        double[] Xt = new double[dim * N];
        history.viewX(t, Xt, 0);
        double[] state = new double[dim];
        for (int j = 0; j < dim; j++) {
            state[j] = Xt[j * N + idx];
        }
        return state;
    }
}
