package com.curioloop.yum4j.ssm.particle.smooth;

import com.curioloop.yum4j.ssm.particle.engine.Workspace;

/**
 * Storage policy for recording per-step particles, log-weights, and
 * ancestor indices during an SMC forward pass.
 *
 * <p>Implementations are attached to a {@link Workspace} via
 * {@link Workspace#history}. The engine calls
 * {@link #save(int, Workspace, boolean)} at the end of every step; offline
 * smoothers ({@code FFBS}, {@code TwoFilter}, {@code FixedLag},
 * {@code Paris}) read back through the zero-copy view methods
 * {@link #viewX(int, double[], int)},
 * {@link #viewLogW(int, double[], int)}, and
 * {@link #viewAncestors(int, int[], int)}, which copy directly into a
 * caller-supplied scratch buffer and avoid the allocation hotspot of
 * defensive-copy accessors.
 *
 * <p>Three concrete variants are permitted:
 *
 * <ul>
 *   <li>{@link Full} — store every {@code (X_t, logW_t, a_t)} triple for
 *       {@code t in [0, T)}; memory {@code O(T · d · N + T · N)}.</li>
 *   <li>{@link Rolling} — circular buffer of the last {@code L} steps;
 *       memory {@code O(L · d · N + L · N)}.</li>
 *   <li>{@link Partial} — retains only ancestor indices for all steps
 *       (for online additive smoothers like Paris); memory
 *       {@code O(T · N)}.</li>
 * </ul>
 *
 * <p>All implementations pre-allocate their buffers at construction time.
 * The {@link #save(int, Workspace, boolean)} method performs no heap allocation on
 * the hot path — it copies from the workspace into pre-allocated storage.
 * The view methods likewise perform no heap allocation: they issue a
 * single {@link System#arraycopy} from the backing store into the caller
 * buffer.
 *
 * @see Workspace
 */
public sealed interface ParticleHistory permits Full, Rolling, Partial {

    /**
     * Copies the particle states at time {@code t} into
     * {@code out[outOff .. outOff + dim * N)}.
     *
     * <p>Replaces the former allocating accessor {@code double[] X(int)}:
     * per-step smoothing loops can now reuse a single scratch buffer
     * across every {@code (t, m)} iteration without triggering any
     * {@code double[]} allocation.
     *
     * @param t      the time step
     * @param out    caller-supplied destination buffer
     * @param outOff offset into {@code out} at which to start writing
     * @throws IllegalArgumentException      if step {@code t} is not available
     * @throws UnsupportedOperationException if this history does not store particles
     * @throws IndexOutOfBoundsException     if {@code out.length - outOff < dim * N}
     */
    void viewX(int t, double[] out, int outOff);

    /**
     * Copies the log-weights at time {@code t} into
     * {@code out[outOff .. outOff + N)}.
     *
     * @param t      the time step
     * @param out    caller-supplied destination buffer
     * @param outOff offset into {@code out} at which to start writing
     * @throws IllegalArgumentException      if step {@code t} is not available
     * @throws UnsupportedOperationException if this history does not store log-weights
     * @throws IndexOutOfBoundsException     if {@code out.length - outOff < N}
     */
    void viewLogW(int t, double[] out, int outOff);

    /**
     * Copies the ancestor indices at time {@code t} into
     * {@code out[outOff .. outOff + N)}.
     *
     * @param t      the time step
     * @param out    caller-supplied destination buffer
     * @param outOff offset into {@code out} at which to start writing
     * @throws IllegalArgumentException  if step {@code t} is not available
     * @throws IndexOutOfBoundsException if {@code out.length - outOff < N}
     */
    void viewAncestors(int t, int[] out, int outOff);

    /**
     * Called by the engine to store the current step's data from the
     * workspace. This method performs no heap allocation — all buffers
     * are pre-allocated at construction time.
     *
     * <p>The {@code resampled} flag carries the engine's resample decision
     * for step {@code t} — {@code true} when the engine triggered
     * resampling (so {@code ws.a} contains a non-trivial permutation),
     * {@code false} when the step was uniformly weighted and ancestors
     * are the identity {@code (0, 1, …, N-1)}. Compressed history
     * variants use this flag to elide ancestor storage on identity
     * steps; the simple variants store the same backing data either
     * way and use the flag only to answer {@link #resampledAt(int)}.
     *
     * @param t         the time step being saved
     * @param ws        the workspace containing current particle state
     * @param resampled the engine's resample decision for step {@code t};
     *                  {@code true} iff resampling was triggered
     */
    void save(int t, Workspace ws, boolean resampled);

    /**
     * Returns whether the engine triggered resampling at step {@code t}.
     *
     * <p>When {@code true}, {@link #viewAncestors(int, int[], int)} at
     * step {@code t} returns the resample permutation. When {@code false},
     * the ancestor vector at step {@code t} is the identity
     * {@code (0, 1, …, N-1)} — smoothers may use this to skip a
     * {@code viewAncestors} copy and treat the lineage step as a no-op.
     *
     * @param t the time step to query
     * @return {@code true} iff the engine resampled at step {@code t}
     * @throws IllegalArgumentException if step {@code t} is not currently stored
     */
    boolean resampledAt(int t);

    /**
     * Returns the maximum number of steps this history can store.
     *
     * <p>For {@link Full} this equals {@code T}; for {@link Rolling}
     * it equals the lag {@code L}; for {@link Partial} it equals
     * {@code T}.
     *
     * @return the capacity in number of steps
     */
    int capacity();

    /**
     * Returns {@code true} if step {@code t} is currently available
     * for reading.
     *
     * @param t the time step to query
     * @return whether step t data is available
     */
    boolean hasStep(int t);

    /** Number of particles. */
    int N();

    /** State dimension per particle. */
    int dim();
}
