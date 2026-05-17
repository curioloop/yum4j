package com.curioloop.yum4j.ssm.particle.smooth;

/**
 * Pre-allocated scratch buffers for the backward-sampling smoothers
 * ({@link FFBS}, {@link TwoFilter}, {@link FixedLag}, {@link Paris}).
 * Allocated once per smoothing shape and reused across every
 * {@code (t, m)} iteration so the inner smoothing loop performs no heap
 * allocation (R5.1).
 *
 * <p>A single workspace covers all four FFBS smoother variants —
 * {@code sampleON2}, {@code sampleMcmc}, {@code sampleReject},
 * {@code sampleHybrid} — as well as {@link TwoFilter#sample},
 * {@link FixedLag#compute}, {@link FixedLag#traceAncestors}, and
 * {@link Paris#smooth}. If any dimension of a subsequent smoothing call
 * exceeds the workspace's allocated shape,
 * {@link #validateShape(int, int, int, int)} throws
 * {@link IllegalArgumentException} naming the mismatched dimension
 * (R5.4).
 *
 * <p>The fields are {@code public final} to allow direct access from the
 * smoother kernels; callers must not reassign them and should treat the
 * array references as non-owned scratch.
 *
 */
public final class SmoothingWorkspace {

    /** Maximum number of smoothed trajectories this workspace supports. */
    public final int M;
    /** Maximum number of particles this workspace supports. */
    public final int N;
    /** Maximum state dimension this workspace supports. */
    public final int dim;
    /** Maximum horizon this workspace supports. */
    public final int T;

    // --------------------------------------------------------------
    // Backward-loop buffers reused across every (t, m) iteration.
    // --------------------------------------------------------------

    /** Particle states at time {@code t}, length {@code dim * N}. */
    public final double[] Xt;
    /** Log-weights at time {@code t}, length {@code N}. */
    public final double[] logWt;
    /** Scratch for {@code logPt} evaluations over all N particles, length {@code N}. */
    public final double[] logPtBuf;
    /** Backward weights buffer (forward logW + logPt), length {@code N}. */
    public final double[] lwBack;
    /** Cumulative-weight buffer for the inverse-CDF scan, length {@code N}. */
    public final double[] cums;
    /** Broadcast of a single {@code X_{t+1}^target} across all N columns, length {@code dim * N}. */
    public final double[] Xtp1Broadcast;
    /** Scratch buffer passed to {@code StepContext} on the full-N path, length {@code 2 * N}. */
    public final double[] scratchCtx;
    /** Ancestor indices at time {@code t+1}, length {@code N}. */
    public final int[] aTp1;

    // --------------------------------------------------------------
    // Single-particle evaluation buffers for rejection/MCMC variants.
    // --------------------------------------------------------------

    /** Single-particle {@code Xprev} slot, length {@code dim}. */
    public final double[] xPrev1;
    /** Single-particle {@code Xcur} slot, length {@code dim}. */
    public final double[] xCur1;
    /** Single-element {@code logPt} output, length {@code 1}. */
    public final double[] logPt1;
    /** Second single-element {@code logPt} output (MCMC current path), length {@code 1}. */
    public final double[] logPt2;
    /** Scratch buffer passed to {@code StepContext} on the single-particle path, length {@code 2}. */
    public final double[] scratch1;

    // --------------------------------------------------------------
    // Proposal buffers for rejection / MCMC variants.
    // --------------------------------------------------------------

    /** Per-trajectory proposal indices, length {@code M}. */
    public final int[] proposals;
    /** Resample scratch for proposal draws, length {@code max(N + M + 1, 2 * N)}. */
    public final double[] propScratch;
    /** Per-trajectory acceptance flags, length {@code M}. */
    public final boolean[] accepted;

    // --------------------------------------------------------------
    // Seed pass (multinomial draw from final-step weights).
    // --------------------------------------------------------------

    /** Buffer for the initial seed-row draw, length {@code M}. */
    public final int[] lastRow;
    /** Copy of the final-step log-weights for the seed draw, length {@code N}. */
    public final double[] lwBuf;
    /** Resample scratch for the seed draw, length {@code max(N + M + 1, 2 * N)}. */
    public final double[] seedScratch;

    /**
     * Allocates a workspace sized to support smoothing calls up to the
     * given shape. Buffers are allocated once; subsequent calls reuse
     * them.
     *
     * @param M   maximum number of smoothed trajectories
     * @param N   maximum number of particles
     * @param dim maximum state dimension
     * @param T   maximum horizon
     * @return a newly allocated workspace
     * @throws IllegalArgumentException if any dimension is {@code <= 0}
     */
    public static SmoothingWorkspace allocate(int M, int N, int dim, int T) {
        if (M <= 0) throw new IllegalArgumentException("M must be > 0: " + M);
        if (N <= 0) throw new IllegalArgumentException("N must be > 0: " + N);
        if (dim <= 0) throw new IllegalArgumentException("dim must be > 0: " + dim);
        if (T <= 0) throw new IllegalArgumentException("T must be > 0: " + T);
        return new SmoothingWorkspace(M, N, dim, T);
    }

    private SmoothingWorkspace(int M, int N, int dim, int T) {
        this.M = M;
        this.N = N;
        this.dim = dim;
        this.T = T;

        int resampleScratch = Math.max(N + M + 1, 2 * N);

        this.Xt = new double[dim * N];
        this.logWt = new double[N];
        this.logPtBuf = new double[N];
        this.lwBack = new double[N];
        this.cums = new double[N];
        this.Xtp1Broadcast = new double[dim * N];
        this.scratchCtx = new double[2 * N];
        this.aTp1 = new int[N];

        this.xPrev1 = new double[dim];
        this.xCur1 = new double[dim];
        this.logPt1 = new double[1];
        this.logPt2 = new double[1];
        this.scratch1 = new double[2];

        this.proposals = new int[M];
        this.propScratch = new double[resampleScratch];
        this.accepted = new boolean[M];

        this.lastRow = new int[M];
        this.lwBuf = new double[N];
        this.seedScratch = new double[resampleScratch];
    }

    /**
     * Validates that this workspace can satisfy a smoothing call with
     * the given shape. Throws {@link IllegalArgumentException} naming
     * the mismatched dimension (R5.4).
     *
     * @param reqM   requested number of smoothed trajectories
     * @param reqN   requested number of particles
     * @param reqDim requested state dimension
     * @param reqT   requested horizon
     * @throws IllegalArgumentException if any requested dimension exceeds
     *         the allocated capacity
     */
    public void validateShape(int reqM, int reqN, int reqDim, int reqT) {
        if (reqM > this.M) {
            throw new IllegalArgumentException(
                "SmoothingWorkspace: requested M=" + reqM
                    + " exceeds workspace.M=" + this.M);
        }
        if (reqN > this.N) {
            throw new IllegalArgumentException(
                "SmoothingWorkspace: requested N=" + reqN
                    + " exceeds workspace.N=" + this.N);
        }
        if (reqDim > this.dim) {
            throw new IllegalArgumentException(
                "SmoothingWorkspace: requested dim=" + reqDim
                    + " exceeds workspace.dim=" + this.dim);
        }
        if (reqT > this.T) {
            throw new IllegalArgumentException(
                "SmoothingWorkspace: requested T=" + reqT
                    + " exceeds workspace.T=" + this.T);
        }
    }
}
