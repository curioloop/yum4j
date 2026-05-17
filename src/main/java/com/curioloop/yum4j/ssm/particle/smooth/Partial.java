package com.curioloop.yum4j.ssm.particle.smooth;

import com.curioloop.yum4j.ssm.particle.engine.Workspace;

/**
 * Partial history: retains only ancestor indices, compressed via
 * {@code Identity_Skip} and {@code Ancestor_Type_Narrowing}.
 *
 * <p>This implementation is designed for online additive smoothers
 * (e.g. Paris-style) that only need the ancestor lineage to trace
 * particle genealogies. It does <b>not</b> store particle states or
 * log-weights, saving significant memory compared to {@link Full}.
 *
 * <h2>Ancestor compression (Stage B of {@code particle-v2-mem})</h2>
 *
 * Ancestor storage is delegated to {@link AncestorIndex} (linear,
 * fail-fast pool), which implements the two lossless transforms
 * introduced by Stage B:
 * <ul>
 *   <li><b>Identity-step skip</b> — non-resampled steps consume zero
 *       slots; identity ancestors are synthesised at view time.</li>
 *   <li><b>Char vs int narrowing</b> — {@code char[]} backing when
 *       {@code N <= 65535}, else {@code int[]}.</li>
 * </ul>
 *
 * <p>{@code Partial} never stored {@code X} or {@code lw} to begin with,
 * so Stage B's lossless reduction lands entirely on the ancestor
 * footprint — this variant is the biggest beneficiary of compression
 * (≈ 60% total reduction at the gate workload, &gt; 99% on
 * char-applicable workloads where {@code N ≤ 65535}; see
 * {@code bench/mem-results.md}).
 *
 * <h2>{@code maxResampleRate} parameter</h2>
 *
 * The ancestor slot pool is pre-sized to
 * {@code ceil(maxResampleRate * T) * N} elements. The default rate is
 * {@code 0.6}; configure via the overload
 * {@link #Partial(int, int, int, double) Partial(N, dim, T, maxResampleRate)}.
 * The argument must satisfy {@code 0 < maxResampleRate <= 1}; otherwise
 * the constructor throws {@link IllegalArgumentException}. If the
 * engine's actual resample rate exceeds the configured bound,
 * {@link #save(int, Workspace, boolean)} throws
 * {@link IllegalStateException} — there is no silent grow, since a
 * hidden reallocation would violate the {@code Engine.step} 0 B/op
 * gate inherited from {@code particle-v2-perf}. On overflow, rebuild
 * the history with a higher {@code maxResampleRate} or switch to
 * {@link Rolling}.
 *
 * <h2>Memory footprint</h2>
 *
 * Approximate byte cost as a function of {@code (T, N, rate)} (no
 * {@code X} or {@code lw} terms):
 * <pre>
 *   F_partial(T, N, rate)
 *       ≈ ceil(rate · T) · N · width          // ancestor slot pool
 *       + bookkeeping
 *
 *       where width      = 2 (char[]) when N ≤ 65535
 *                          else 4 (int[])
 *             bookkeeping ≈ ceil(T/64) · 8    // resampledBits
 *                         + ceil(T/64) · 8    // hasAncestorSlot
 *                         + 4 · T             // stepToSlot
 * </pre>
 *
 * <p>All buffers are pre-allocated at construction time —
 * {@link #save(int, Workspace, boolean)} performs no heap allocation.
 *
 * <p>Calls to {@link #viewX(int, double[], int)} or
 * {@link #viewLogW(int, double[], int)} throw
 * {@link UnsupportedOperationException} since this history does not
 * retain those arrays.
 *
 * <h2>Cross-references</h2>
 *
 * Design lives at {@code .kiro/specs/particle-v2-mem/}; Stage B
 * (ancestor compression) has landed and Stage C (log-weight derivation)
 * is deferred to a follow-up spec — though it has no effect on this
 * variant since {@code Partial} never stored {@code lw}.
 *
 */
public final class Partial implements ParticleHistory {

    private final int N;
    private final int dim;
    private final int T;

    /** Compressed ancestor pool — identity-skip + char-narrowing. */
    private final AncestorIndex ancestors;

    private int savedCount;

    /**
     * Allocates a partial history for {@code T} steps of {@code N} particles.
     *
     * <p>The {@code maxResampleRate} argument bounds the number of
     * pre-allocated ancestor slots: at most {@code ceil(maxResampleRate * T)}
     * resampled steps can be saved before {@link #save(int, Workspace, boolean)}
     * throws {@link IllegalStateException}. There is no silent reallocation —
     * the caller must size for the workload's expected resample rate.
     *
     * @param N               number of particles (must be &gt; 0)
     * @param dim             state dimension per particle (must be &gt; 0)
     * @param T               total number of time steps (must be &gt; 0)
     * @param maxResampleRate upper bound on resampled-step fraction in
     *                        {@code (0, 1]}
     * @throws IllegalArgumentException if any parameter is out of range or
     *         if the required buffer exceeds addressable array size
     */
    public Partial(int N, int dim, int T, double maxResampleRate) {
        if (N <= 0) throw new IllegalArgumentException("N must be > 0: " + N);
        if (dim <= 0) throw new IllegalArgumentException("dim must be > 0: " + dim);
        if (T <= 0) throw new IllegalArgumentException("T must be > 0 for Partial history: " + T);
        this.N = N;
        this.dim = dim;
        this.T = T;
        this.ancestors = new AncestorIndex(N, T, maxResampleRate, /* circular */ false);
        this.savedCount = 0;
    }

    /**
     * Convenience overload using the default {@code maxResampleRate} of
     * {@value AncestorIndex#DEFAULT_MAX_RESAMPLE_RATE}.
     *
     * @param N   number of particles (must be &gt; 0)
     * @param dim state dimension per particle (must be &gt; 0)
     * @param T   total number of time steps (must be &gt; 0)
     */
    public Partial(int N, int dim, int T) {
        this(N, dim, T, AncestorIndex.DEFAULT_MAX_RESAMPLE_RATE);
    }

    /** Horizon (total number of steps this history can hold). */
    public int T() { return T; }

    @Override public int N() { return N; }
    @Override public int dim() { return dim; }
    @Override public int capacity() { return T; }

    @Override
    public boolean hasStep(int t) {
        return t >= 0 && t < savedCount;
    }

    @Override
    public void save(int t, Workspace ws, boolean resampled) {
        if (t < 0 || t >= T) {
            throw new IllegalArgumentException(
                "t out of range [0, T=" + T + "): " + t);
        }
        if (resampled) {
            try {
                ancestors.recordResample(t, ws.a, t);
            } catch (IllegalStateException e) {
                // Re-wrap to surface the variant name and the
                // "maxResampledSlots" keyword that the existing
                // overflow tests assert on.
                throw new IllegalStateException(
                    "Partial history exceeded configured maxResampledSlots="
                        + ancestors.maxResampledSlots() + " at t=" + t
                        + "; raise maxResampleRate or switch to Rolling", e);
            }
        } else {
            ancestors.recordIdentity(t);
        }
        savedCount = Math.max(savedCount, t + 1);
    }

    @Override
    public boolean resampledAt(int t) {
        requireStored(t);
        return ancestors.resampledAt(t);
    }

    /**
     * Not supported by Partial history — particle states are not retained.
     *
     * @throws UnsupportedOperationException always
     */
    @Override
    public void viewX(int t, double[] out, int outOff) {
        throw new UnsupportedOperationException(
            "Partial history does not store particle states; use Full or Rolling");
    }

    /**
     * Not supported by Partial history — log-weights are not retained.
     *
     * @throws UnsupportedOperationException always
     */
    @Override
    public void viewLogW(int t, double[] out, int outOff) {
        throw new UnsupportedOperationException(
            "Partial history does not store log-weights; use Full or Rolling");
    }

    @Override
    public void viewAncestors(int t, int[] out, int outOff) {
        requireStored(t);
        ancestors.read(t, out, outOff);
    }

    /**
     * Package-private access to the ancestor pool for diagnostics and
     * benchmarks (e.g. {@code HistoryFootprintBench} measuring the
     * deep retained size of the ancestor backing). Not part of the
     * public API surface.
     */
    AncestorIndex ancestors() { return ancestors; }

    private void requireStored(int t) {
        if (!hasStep(t)) {
            throw new IllegalArgumentException(
                "step " + t + " not stored (savedCount=" + savedCount + ", T=" + T + ")");
        }
    }
}
