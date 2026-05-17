package com.curioloop.yum4j.ssm.particle.engine;

import com.curioloop.yum4j.ssm.particle.HistoryMode;
import com.curioloop.yum4j.ssm.particle.kernel.RandomBatch;
import com.curioloop.yum4j.ssm.particle.smooth.Full;
import com.curioloop.yum4j.ssm.particle.smooth.Partial;
import com.curioloop.yum4j.ssm.particle.smooth.ParticleHistory;
import com.curioloop.yum4j.ssm.particle.smooth.Rolling;

import java.util.Arrays;

/**
 * Buffer-owning workspace for the particle-filter engine.
 *
 * <p>Holds the mutable {@code (N × dim)} particle buffers, per-particle
 * log-weights, ancestor indices, a precomputed identity permutation,
 * a unified scratch arena, optional history storage, and a batched RNG.
 *
 * <p>Shape fields ({@code N}, {@code dim}) are final and set at allocation
 * time. Buffer fields are mutable — the engine swaps {@code X} and
 * {@code Xprev} references on non-resample steps.
 *
 * <p>This class explicitly does <b>not</b> expose any {@code _OFF} constants
 * or {@code histX}/{@code histLw}/{@code histA} aliases (R7.3, R7.4).
 *
 * <p>The {@link #rng} field is <b>not</b> populated by {@link #allocate} —
 * callers (particle-filter builder, online-filter builder, PMMH,
 * SMC² rejuvenation loop) <em>must</em> set it explicitly before invoking
 * {@link Engine#init}, which fails fast with {@link IllegalStateException}
 * if {@code ws.rng == null} (R2.4, R2.5).
 *
 * <p>The {@link #slots} and {@link #slabRngs} arrays are lazily grown by
 * {@link Engine#makeCtx} to the maximum {@code chunks} value seen across
 * the workspace's lifetime; callers should not touch them directly.
 *
 * @see RunState
 * @see StepContext
 */
public final class Workspace {

    // ── Shape (immutable after allocation) ──────────────────────────────

    /** Number of particles. */
    public final int N;

    /** State dimension per particle. */
    public final int dim;

    // ── Particle buffers ────────────────────────────────────────────────

    /** Current particle buffer, length {@code N * dim}. */
    public double[] X;

    /** Previous particle buffer, length {@code N * dim}. */
    public double[] Xprev;

    /** Per-particle log-weights, length {@code N}. */
    public double[] logW;

    /** Ancestor indices, length {@code N}. */
    public int[] a;

    /** Precomputed identity permutation {@code [0, 1, ..., N-1]}, length {@code N}. */
    public int[] identity;

    /**
     * Unified scratch arena, length {@code 3 * N}.
     *
     * <p>Sized to cover the largest per-scheme requirement documented on
     * {@link com.curioloop.yum4j.ssm.particle.resample.Resample} (R7.4).
     * Under the engine's {@code M = N} contract the per-scheme minima are
     * {@code SYSTEMATIC: N + 1}, {@code STRATIFIED: 2N},
     * {@code MULTINOMIAL: 2N + 1}, {@code RESIDUAL: 2N},
     * {@code SSP: 3N - 1}, {@code KILLING: 2N}. A single {@code 3 * N}
     * allocation satisfies all of them.
     */
    public double[] scratch;

    /**
     * Optional particle history (null when {@link HistoryMode#NONE}).
     *
     * <p>Typed as {@link ParticleHistory}. Implementations are
     * {@link Full}, {@link Rolling}, or {@link Partial} depending on
     * the configured {@link HistoryMode}.
     */
    public ParticleHistory history;

    /**
     * Batched RNG view for the engine. Starts {@code null}; the caller
     * must assign a {@link RandomBatch} before any {@link Engine#init}
     * call. See the class Javadoc for the rationale (R2.4, R2.5).
     */
    public RandomBatch rng;

    /**
     * Pre-allocated per-slab step contexts, one per chunk index. The engine
     * mutates these in place across every step, eliminating per-step
     * {@link StepContext} allocation. Lazily sized to the maximum chunk count
     * seen by any {@link ParallelStrategy} the workspace has been used with.
     *
     * <p>Reusable across runs: {@link #resetBuffers()} does not clear the
     * slots; {@link Engine#makeCtx} overwrites their fields on every call.
     *
     * <p>See {@link #ensureSlots(int)} (R3.1, R3.2).
     */
    public StepContext<?>[] slots;

    /**
     * Lazy cache of per-slab {@link RandomBatch} instances for STRIPED
     * dispatch. Populated on first access under the same Workspace + master
     * RNG seed; {@link #resetBuffers()} nulls every entry so a subsequent
     * run whose master RNG has been re-seeded picks up fresh per-slab
     * streams via {@link RandomBatch#split(long)}.
     *
     * <p>See {@link #ensureSlabRngs(int)} (R2.1, R2.3).
     */
    public RandomBatch[] slabRngs;

    // ── Private constructor ─────────────────────────────────────────────

    private Workspace(int N, int dim) {
        this.N = N;
        this.dim = dim;
    }

    // ── Factory ─────────────────────────────────────────────────────────

    /**
     * Allocates a fully initialised workspace (except for {@link #rng},
     * which the caller must set before engine invocation — see the class
     * Javadoc and R2.4).
     *
     * @param N          number of particles (must be &gt; 0)
     * @param dim        state dimension per particle (must be &gt; 0)
     * @param mode       history retention mode
     * @param historyArg mode-specific argument (e.g. lag for {@link HistoryMode#ROLLING})
     * @return a new workspace with all buffers allocated and identity filled
     * @throws IllegalArgumentException if {@code N <= 0} or {@code dim <= 0}
     */
    public static Workspace allocate(int N, int dim, HistoryMode mode, int historyArg) {
        if (N <= 0) {
            throw new IllegalArgumentException("N must be > 0, got " + N);
        }
        if (dim <= 0) {
            throw new IllegalArgumentException("dim must be > 0, got " + dim);
        }

        Workspace ws = new Workspace(N, dim);

        ws.X = new double[N * dim];
        ws.Xprev = new double[N * dim];
        ws.logW = new double[N];
        ws.a = new int[N];
        ws.identity = new int[N];
        ws.scratch = new double[3 * N];

        // Fill identity permutation [0, 1, ..., N-1]
        for (int i = 0; i < N; i++) {
            ws.identity[i] = i;
        }

        // Copy identity into ancestor array as initial state
        System.arraycopy(ws.identity, 0, ws.a, 0, N);

        // History: allocate ParticleHistory based on mode + historyArg
        if (mode == HistoryMode.NONE) {
            ws.history = null;
        } else if (mode == HistoryMode.FULL) {
            ws.history = new Full(N, dim, historyArg > 0 ? historyArg : 1);
        } else if (mode == HistoryMode.ROLLING) {
            ws.history = new Rolling(historyArg, N, dim);
        } else if (mode == HistoryMode.PARTIAL) {
            ws.history = new Partial(N, dim, historyArg > 0 ? historyArg : 1);
        }

        // ws.rng stays null; the caller must set it before Engine.init.
        // ws.slots and ws.slabRngs stay null; Engine.makeCtx grows them
        // lazily on first dispatch.

        return ws;
    }

    // ── Buffer operations ───────────────────────────────────────────────

    /**
     * Swaps the {@code X} and {@code Xprev} references (no array copy).
     * Used on non-resample steps where the current particles become the
     * previous particles for the next step.
     */
    public void swapXandXprev() {
        double[] tmp = X;
        X = Xprev;
        Xprev = tmp;
    }

    /**
     * Resets mutable buffers to their initial state:
     * <ul>
     *   <li>{@code logW} is zeroed</li>
     *   <li>{@code a} is filled with the identity permutation</li>
     *   <li>every {@link #slabRngs} entry is nulled so a subsequent run
     *       whose {@link #rng} has been re-seeded re-splits via
     *       {@link RandomBatch#split(long)} on first access (R2.3)</li>
     * </ul>
     *
     * <p>{@link #slots} is <em>not</em> cleared — the slot entries are
     * reusable scratch containers; {@link Engine#makeCtx} overwrites their
     * fields on every dispatch (R3.2).
     */
    public void resetBuffers() {
        Arrays.fill(logW, 0.0);
        System.arraycopy(identity, 0, a, 0, N);
        if (slabRngs != null) {
            Arrays.fill(slabRngs, null);
        }
    }

    // ── Lazy growth for engine-side pools ───────────────────────────────

    /**
     * Ensures {@link #slots} has at least {@code chunks} entries. Called
     * by {@link Engine#makeCtx} before accessing {@code slots[slabIndex]}.
     * Existing entries are preserved; new entries are filled with fresh
     * no-arg {@link StepContext} instances whose fields the engine will
     * overwrite on every dispatch.
     *
     * @param chunks the minimum required size
     */
    public void ensureSlots(int chunks) {
        if (slots == null || slots.length < chunks) {
            StepContext<?>[] grown = new StepContext<?>[chunks];
            int existing = (slots == null) ? 0 : slots.length;
            if (existing > 0) {
                System.arraycopy(slots, 0, grown, 0, existing);
            }
            for (int i = existing; i < chunks; i++) {
                grown[i] = new StepContext<>();
            }
            slots = grown;
        }
    }

    /**
     * Ensures {@link #slabRngs} has at least {@code chunks} entries. Existing
     * entries are preserved; new entries are {@code null} and populated on
     * first access via {@link RandomBatch#split(long)} by the engine.
     *
     * @param chunks the minimum required size
     */
    public void ensureSlabRngs(int chunks) {
        if (slabRngs == null || slabRngs.length < chunks) {
            RandomBatch[] grown = new RandomBatch[chunks];
            int existing = (slabRngs == null) ? 0 : slabRngs.length;
            if (existing > 0) {
                System.arraycopy(slabRngs, 0, grown, 0, existing);
            }
            slabRngs = grown;
        }
    }
}
