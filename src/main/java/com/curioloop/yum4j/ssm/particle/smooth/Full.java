package com.curioloop.yum4j.ssm.particle.smooth;

import com.curioloop.yum4j.ssm.particle.engine.Workspace;

import java.lang.System.Logger;
import java.lang.System.Logger.Level;

/**
 * Full-history storage: retains {@code (X_t, logW_t, a_t)} for every
 * {@code t in [0, T)} in contiguous pre-allocated buffers.
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
 * <p>The {@code X} array (length {@code T*d*N}) and the {@code lw} array
 * (length {@code T*N}) are unchanged in Stage B: {@link #viewX} and
 * {@link #viewLogW} are each a single {@link System#arraycopy} from the
 * stored backing. Stage C (log-weight derivation) is deferred to a
 * follow-up spec; see the {@code particle-v2-mem} requirements.md R4
 * status block for the algorithmic and cross-spec-API rationale.
 *
 * <h2>{@code maxResampleRate} parameter</h2>
 *
 * The ancestor slot pool is pre-sized to
 * {@code ceil(maxResampleRate * T) * N} elements. The default rate is
 * {@code 0.6} (typical SMC ESS thresholds yield 30–50% resample rates;
 * 0.6 buys headroom without inflating the buffer). Configure via the
 * overload {@link #Full(int, int, int, double)
 * Full(N, dim, T, maxResampleRate)}. The argument must satisfy
 * {@code 0 < maxResampleRate <= 1}; otherwise the constructor throws
 * {@link IllegalArgumentException}. If the engine's actual resample
 * rate exceeds the configured bound, {@link #save(int, Workspace, boolean)}
 * throws {@link IllegalStateException} — there is no silent grow, since
 * a hidden reallocation would violate the {@code Engine.step} 0 B/op
 * gate inherited from {@code particle-v2-perf}. On overflow, rebuild
 * the history with a higher {@code maxResampleRate} or switch to
 * {@link Rolling} for an unbounded forward pass.
 *
 * <h2>Memory footprint</h2>
 *
 * Approximate byte cost as a function of {@code (T, N, dim, rate)}:
 * <pre>
 *   F_full(T, N, dim, rate)
 *       ≈ 8 · T · dim · N                     // X (particles)
 *       + 8 · T · N                           // lw (log-weights)
 *       + ceil(rate · T) · N · width          // ancestor slot pool
 *       + bookkeeping
 *
 *       where width      = 2 (char[]) when N ≤ 65535
 *                          else 4 (int[])
 *             bookkeeping ≈ ceil(T/64) · 8    // resampledBits
 *                         + ceil(T/64) · 8    // hasAncestorSlot
 *                         + 4 · T             // stepToSlot
 * </pre>
 *
 * Bookkeeping is &lt; 0.001% of the total at the gate workload
 * ({@code T = 10⁴}, {@code N = 10⁵}); see {@code bench/mem-results.md}.
 *
 * <h2>Soft-warning threshold</h2>
 *
 * The constructor logs a warning when the projected total exceeds
 * {@value #SOFT_WARN_BYTES} bytes (≈ 2 GiB) but does not refuse the
 * allocation — {@link Rolling} or {@link Partial} are advised at that
 * scale. The 2 GiB threshold predates Stage B and intentionally remains
 * unchanged: although ancestor compression has lowered the per-step
 * ancestor cost (e.g. from {@code 4·T·N} to {@code rate·N·width}, a
 * 60% reduction at the gate workload), the {@code X} and {@code lw}
 * arrays still dominate the footprint at scales relevant to the warning,
 * so the original guidance remains accurate.
 *
 * <h2>Cross-references</h2>
 *
 * Design lives at {@code .kiro/specs/particle-v2-mem/}; Stage B
 * (ancestor compression) has landed and Stage C (log-weight derivation)
 * is deferred — see the spec's requirements.md R4 status block and
 * design.md "Stage C — Log-weight derivation" section. Achieved Stage B
 * reduction is reported in {@code bench/mem-results.md}.
 *
 */
public final class Full implements ParticleHistory {

    private static final Logger LOG = System.getLogger(Full.class.getName());
    private static final long SOFT_WARN_BYTES = 2L * 1024L * 1024L * 1024L;

    private final int N;
    private final int dim;
    private final int T;

    /** Particle states, length {@code T*d*N}. Unchanged in Stage B. */
    private final double[] X;

    /** Log-weights, length {@code T*N}. Unchanged in Stage B (Stage C deferred). */
    private final double[] lw;

    /** Compressed ancestor pool — identity-skip + char-narrowing. */
    private final AncestorIndex ancestors;

    private int savedCount;

    /**
     * Allocates a full history for {@code T} steps of {@code N} particles
     * each with state dimension {@code dim}, using the default maximum
     * resample rate of {@value AncestorIndex#DEFAULT_MAX_RESAMPLE_RATE}.
     *
     * <p>Equivalent to {@link #Full(int, int, int, double)
     * Full(N, dim, T, 0.6)}.
     *
     * @param N   number of particles (must be {@code > 0})
     * @param dim state dimension per particle (must be {@code > 0})
     * @param T   total number of time steps (must be {@code > 0})
     * @throws IllegalArgumentException if any parameter is non-positive or
     *         if the required buffer exceeds addressable array size
     */
    public Full(int N, int dim, int T) {
        this(N, dim, T, AncestorIndex.DEFAULT_MAX_RESAMPLE_RATE);
    }

    /**
     * Allocates a full history with a configurable upper bound on the
     * resample rate. The ancestor slot pool is sized to hold
     * {@code ceil(maxResampleRate * T)} resampled steps; if the engine
     * exceeds that count, {@link #save(int, Workspace, boolean)} throws
     * {@link IllegalStateException}.
     *
     * @param N                number of particles (must be {@code > 0})
     * @param dim              state dimension per particle (must be {@code > 0})
     * @param T                total number of time steps (must be {@code > 0})
     * @param maxResampleRate  upper bound on the observed resample rate;
     *                         must satisfy {@code 0 < maxResampleRate <= 1}
     * @throws IllegalArgumentException if any parameter is out of range or
     *         if the required buffer exceeds addressable array size
     */
    public Full(int N, int dim, int T, double maxResampleRate) {
        if (N <= 0) throw new IllegalArgumentException("N must be > 0: " + N);
        if (dim <= 0) throw new IllegalArgumentException("dim must be > 0: " + dim);
        if (T <= 0) throw new IllegalArgumentException("T must be > 0 for Full history: " + T);
        long totalX = (long) T * dim * N;
        long totalLw = (long) T * N;
        if (totalX > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(
                "Full history X buffer exceeds addressable limit: T=" + T
                    + " dim=" + dim + " N=" + N + " (needs " + totalX + " doubles)");
        }
        if (totalLw > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(
                "Full history lw buffer exceeds addressable limit: T=" + T
                    + " N=" + N + " (needs " + totalLw + " doubles)");
        }

        // Allocate the ancestor pool first so we can include its byte cost in
        // the soft-warning threshold and surface its parameter validation
        // (e.g. maxResampleRate range) before allocating the much-larger
        // X / lw buffers.
        AncestorIndex anc = new AncestorIndex(N, T, maxResampleRate, /* circular */ false);
        long ancestorBytes = (long) anc.maxResampledSlots() * N * (anc.charBacked() ? 2L : 4L);
        long totalBytes = totalX * 8L + totalLw * 8L + ancestorBytes;
        if (totalBytes > SOFT_WARN_BYTES) {
            LOG.log(Level.WARNING,
                () -> "ParticleHistory.Full: storage " + totalBytes
                    + " bytes exceeds 2 GiB; consider Rolling or Partial");
        }

        this.N = N;
        this.dim = dim;
        this.T = T;
        this.X = new double[(int) totalX];
        this.lw = new double[(int) totalLw];
        this.ancestors = anc;
        this.savedCount = 0;
    }

    /** Horizon (total number of steps this history can hold). */
    public int T() { return T; }

    /** Pre-sized number of resampled-step slots in the ancestor pool. */
    public int maxResampledSlots() { return ancestors.maxResampledSlots(); }

    /** {@code true} iff the ancestor backing is {@code char[]} (i.e. {@code N <= 65535}). */
    public boolean charBacked() { return ancestors.charBacked(); }

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
        int dN = dim * N;
        System.arraycopy(ws.X, 0, X, t * dN, dN);
        System.arraycopy(ws.logW, 0, lw, t * N, N);
        if (resampled) {
            try {
                ancestors.recordResample(t, ws.a, t);
            } catch (IllegalStateException e) {
                // Re-wrap with the variant name + maxResampleRate hint that
                // existing tests assert on (keyword: "maxResampleRate").
                throw new IllegalStateException(
                    "Full: observed resample rate exceeds maxResampleRate"
                        + " at step t=" + t + "; rebuild with a higher"
                        + " maxResampleRate or use Rolling", e);
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

    @Override
    public void viewX(int t, double[] out, int outOff) {
        requireStored(t);
        int dN = dim * N;
        System.arraycopy(X, t * dN, out, outOff, dN);
    }

    @Override
    public void viewLogW(int t, double[] out, int outOff) {
        requireStored(t);
        System.arraycopy(lw, t * N, out, outOff, N);
    }

    @Override
    public void viewAncestors(int t, int[] out, int outOff) {
        requireStored(t);
        ancestors.read(t, out, outOff);
    }

    /**
     * Direct access to the backing particle array (for zero-copy interop
     * with smoothers that need contiguous access).
     */
    public double[] rawX() { return X; }

    /** Direct access to the backing log-weight array. */
    public double[] rawLw() { return lw; }

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
