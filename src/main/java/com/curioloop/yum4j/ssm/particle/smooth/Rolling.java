package com.curioloop.yum4j.ssm.particle.smooth;

import com.curioloop.yum4j.ssm.particle.engine.Workspace;

/**
 * Rolling-window history with ancestor compression: retains the last
 * {@code L} steps in a circular buffer for {@code X} and {@code logW},
 * and stores ancestor permutations only for steps where the engine
 * actually resampled.
 *
 * <h2>Ancestor compression (Stage B of {@code particle-v2-mem})</h2>
 *
 * Ancestor storage is delegated to {@link AncestorIndex} (circular,
 * wrap-on-overflow pool), which implements the two lossless transforms
 * introduced by Stage B (identity-skip + char/int narrowing) on a
 * {@code maxResampledSlots = ceil(maxResampleRate * L)}-sized pool
 * indexed circularly.
 *
 * <p>{@code X} (length {@code L*d*N}) and {@code lw} (length {@code L*N})
 * are indexed by {@code t mod L}. Ancestor slots are pooled separately:
 * only resampled steps consume a pool slot, and the pool wraps via
 * {@code nextSlot % maxResampledSlots}. The variant uses {@code t mod L}
 * as the {@link AncestorIndex} index and passes the absolute step
 * {@code t} as the wrap-invariant probe.
 *
 * <p>The {@code X} and {@code lw} backing arrays are unchanged in Stage
 * B: {@link #viewX} and {@link #viewLogW} are each a single
 * {@link System#arraycopy} from the stored backing. Stage C (log-weight
 * derivation) is deferred to a follow-up spec; see the
 * {@code particle-v2-mem} requirements.md R4 status block for the
 * algorithmic and cross-spec-API rationale.
 *
 * <h2>{@code maxResampleRate} parameter</h2>
 *
 * The default rate is {@code 0.6}; configure via the overload
 * {@link #Rolling(int, int, int, double) Rolling(L, N, dim, maxResampleRate)}.
 * The argument must satisfy {@code 0 < maxResampleRate <= 1}; otherwise
 * the constructor throws {@link IllegalArgumentException}. If the
 * engine's actual resample rate (over any window of length {@code L})
 * exceeds the configured bound, the circular pool wraps and overwrites
 * a still-queryable slot — see the wrap invariant below.
 *
 * <h2>Wrap invariant (correctness condition)</h2>
 *
 * <p>When {@code nextSlot} wraps and overwrites an ancestor pool slot
 * that was previously claimed by some step {@code t_old}, correctness
 * requires that {@code t_old} has fallen outside the current rolling
 * window {@code [oldest, newest]} — i.e. its data is dead and a query
 * for {@code viewAncestors(t_old, ...)} would already throw
 * "step outside window".
 *
 * <p>This invariant holds <em>iff</em> the engine's actual resample
 * rate (over any window of length {@code L}) does not exceed the
 * configured {@code maxResampleRate}. The math: between two saves
 * to the same ancestor pool slot there are exactly
 * {@code maxResampledSlots} resample steps. If the actual rate
 * {@code r ≤ maxResampleRate}, then the gap in absolute steps between
 * the two saves is at least {@code maxResampledSlots / r ≥ L}, so
 * {@code t_old ≤ t - L < oldest()}.
 *
 * <p>If the actual rate <em>exceeds</em> {@code maxResampleRate} the
 * wrap silently overwrites a still-queryable slot and
 * {@code viewAncestors(t_old, ...)} returns the wrong (newer)
 * permutation. The class does <strong>not</strong> guard this in
 * production (every save's slot lookup would be a hot-path branch);
 * users are expected to configure {@code maxResampleRate} ≥ the
 * worst-case observed rate, or switch to {@link Full} for a
 * fail-fast bounded forward pass. In <em>development builds</em>
 * (i.e. when the JVM is started with {@code -ea}) the invariant is
 * checked on every wrap by {@link AncestorIndex} and a violation
 * surfaces as {@link AssertionError}.
 *
 * <h2>Memory footprint</h2>
 *
 * Approximate byte cost as a function of {@code (L, N, dim, rate)}:
 * <pre>
 *   F_rolling(L, N, dim, rate)
 *       ≈ 8 · L · dim · N                     // X (particles)
 *       + 8 · L · N                           // lw (log-weights)
 *       + ceil(rate · L) · N · width          // ancestor slot pool
 *       + bookkeeping
 *
 *       where width      = 2 (char[]) when N ≤ 65535
 *                          else 4 (int[])
 *             bookkeeping ≈ ceil(L/64) · 8    // resampledBits
 *                         + ceil(L/64) · 8    // hasAncestorSlot
 *                         + 4 · L             // stepToSlot
 * </pre>
 *
 * <p>All buffers are pre-allocated at construction time;
 * {@link #save(int, Workspace, boolean)} performs no heap allocation.
 *
 * <p>Reads must target a step in the currently retained window
 * {@code [newest - L + 1, newest]}; a request for anything older
 * throws {@link IllegalArgumentException}.
 *
 * <h2>Cross-references</h2>
 *
 * Design lives at {@code .kiro/specs/particle-v2-mem/}; Stage B
 * (ancestor compression) has landed and Stage C (log-weight derivation)
 * is deferred. Achieved reduction at the gate workload is reported in
 * {@code bench/mem-results.md}.
 *
 */
public final class Rolling implements ParticleHistory {

    private final int L;
    private final int N;
    private final int dim;

    /** Particle states, length {@code L*d*N}. Indexed by {@code t mod L}. */
    private final double[] X;

    /** Log-weights, length {@code L*N}. Indexed by {@code t mod L}. */
    private final double[] lw;

    /** Compressed ancestor pool — identity-skip + char-narrowing + circular wrap. */
    private final AncestorIndex ancestors;

    private int newest = -1;             // most recent saved step; -1 before first save

    /**
     * Allocates a rolling history with window size {@code L}, ancestor
     * pool sized to a worst-case resample rate of {@code maxResampleRate}.
     *
     * @param L                window size / lag (must be {@code > 0})
     * @param N                particle count (must be {@code > 0})
     * @param dim              state dimension per particle (must be {@code > 0})
     * @param maxResampleRate  upper bound on the engine's observed
     *                         resample rate over any window of length
     *                         {@code L}; must be in {@code (0, 1]}
     * @throws IllegalArgumentException if any parameter is out of range
     *         or if any backing buffer would exceed
     *         {@code Integer.MAX_VALUE} elements
     */
    public Rolling(int L, int N, int dim, double maxResampleRate) {
        if (L <= 0) throw new IllegalArgumentException("L must be > 0: " + L);
        if (N <= 0) throw new IllegalArgumentException("N must be > 0: " + N);
        if (dim <= 0) throw new IllegalArgumentException("dim must be > 0: " + dim);
        long totalX = (long) L * dim * N;
        long totalLw = (long) L * N;
        if (totalX > Integer.MAX_VALUE || totalLw > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(
                "Rolling history exceeds addressable limit: L=" + L
                    + " dim=" + dim + " N=" + N);
        }
        this.L = L;
        this.N = N;
        this.dim = dim;
        this.X = new double[(int) totalX];
        this.lw = new double[(int) totalLw];
        this.ancestors = new AncestorIndex(N, L, maxResampleRate, /* circular */ true);
    }

    /**
     * Allocates a rolling history with the default
     * {@code maxResampleRate = }{@value AncestorIndex#DEFAULT_MAX_RESAMPLE_RATE}.
     * See {@link #Rolling(int, int, int, double)} for parameter semantics.
     */
    public Rolling(int L, int N, int dim) {
        this(L, N, dim, AncestorIndex.DEFAULT_MAX_RESAMPLE_RATE);
    }

    /** Window size (lag). */
    public int lag() { return L; }

    /** Most recently saved step, or {@code -1} before the first save. */
    public int newest() { return newest; }

    /** Oldest step currently retained in the window. */
    public int oldest() {
        if (newest < 0) return -1;
        return Math.max(0, newest - L + 1);
    }

    /** Pre-sized ancestor pool capacity. Useful for diagnostics. */
    public int maxResampledSlots() { return ancestors.maxResampledSlots(); }

    @Override public int N() { return N; }
    @Override public int dim() { return dim; }
    @Override public int capacity() { return L; }

    @Override
    public boolean hasStep(int t) {
        if (newest < 0) return false;
        return t >= oldest() && t <= newest;
    }

    @Override
    public void save(int t, Workspace ws, boolean resampled) {
        if (t < 0) {
            throw new IllegalArgumentException("t must be >= 0: " + t);
        }
        if (t <= newest) {
            throw new IllegalArgumentException(
                "Rolling.save expects strictly increasing t; got t=" + t
                    + " after newest=" + newest);
        }
        int slotT = t % L;
        int dN = dim * N;
        System.arraycopy(ws.X, 0, X, slotT * dN, dN);
        System.arraycopy(ws.logW, 0, lw, slotT * N, N);
        if (resampled) {
            ancestors.recordResample(slotT, ws.a, t);
        } else {
            ancestors.recordIdentity(slotT);
        }
        newest = t;
    }

    @Override
    public boolean resampledAt(int t) {
        requireStored(t);
        return ancestors.resampledAt(t % L);
    }

    @Override
    public void viewX(int t, double[] out, int outOff) {
        requireStored(t);
        int dN = dim * N;
        int slot = t % L;
        System.arraycopy(X, slot * dN, out, outOff, dN);
    }

    @Override
    public void viewLogW(int t, double[] out, int outOff) {
        requireStored(t);
        int slot = t % L;
        System.arraycopy(lw, slot * N, out, outOff, N);
    }

    @Override
    public void viewAncestors(int t, int[] out, int outOff) {
        requireStored(t);
        ancestors.read(t % L, out, outOff);
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
                "step " + t + " not in rolling window [" + oldest()
                    + ", " + newest + "]");
        }
    }
}
