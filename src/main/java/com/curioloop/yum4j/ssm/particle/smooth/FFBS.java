package com.curioloop.yum4j.ssm.particle.smooth;

import com.curioloop.yum4j.ssm.particle.engine.StepContext;
import com.curioloop.yum4j.ssm.particle.kernel.RandomBatch;
import com.curioloop.yum4j.ssm.particle.model.TransitionDensity;
import com.curioloop.yum4j.ssm.particle.resample.Resample;
import com.curioloop.yum4j.ssm.particle.resample.Scheme;

import java.util.random.RandomGenerator;

/**
 * Forward-Filtering Backward-Sampling (FFBS) offline smoother.
 *
 * <p>Takes a completed {@link ParticleHistory} (must be {@link Full}) and a
 * {@link TransitionDensity} trait, and produces {@code M} smoothed trajectories
 * by sampling backwards through the particle cloud.
 *
 * <p>The returned {@code int[T*M]} array is laid out row-major: element
 * {@code idx[t*M + m]} is the particle index at time {@code t} for trajectory
 * {@code m}. To reconstruct actual state values, the caller copies particles
 * into a reusable scratch via {@code history.viewX(t, scratch, 0)} and
 * indexes into the scratch buffer for the appropriate particle.
 *
 * <p>Four backward-sampling variants are provided:
 * <ul>
 *   <li>{@link #sampleON2} — textbook O(T·N·M); reweighs every ancestor.</li>
 *   <li>{@link #sampleReject} — pure rejection using {@link TransitionDensity#upperBound};
 *       may have unbounded runtime capped by {@code maxTrials}.</li>
 *   <li>{@link #sampleMcmc} — MCMC-based O(T·M·nSteps) using independent MH.</li>
 *   <li>{@link #sampleHybrid} — rejection with fallback to O(N²) for remaining particles.
 *       Recommended default.</li>
 * </ul>
 *
 * <p>Each variant offers two overloads: a convenience form that allocates a
 * fresh {@link SmoothingWorkspace} per call, and a workspace-accepting form
 * that reuses caller-supplied scratch across every {@code (t, m)} iteration.
 * Sampler inner loops (PMMH, SMC²) should call the workspace form so the
 * backward pass is allocation-free after the one-time workspace construction
 *
 */
public final class FFBS {

    private FFBS() {}

    // ------------------------------------------------------------------
    // ON² — textbook backward sampling
    // ------------------------------------------------------------------

    /**
     * Textbook O(T·N·M) backward sampling. Allocates a fresh
     * {@link SmoothingWorkspace} sized to {@code (M, N, dim, T)} and
     * delegates to {@link #sampleON2(ParticleHistory, TransitionDensity, int, RandomBatch, SmoothingWorkspace)}.
     */
    public static int[] sampleON2(ParticleHistory history, TransitionDensity<?> td,
                                  int M, RandomBatch rng) {
        validateArgs(history, M);
        Full full = requireFull(history);
        SmoothingWorkspace sw = SmoothingWorkspace.allocate(M, full.N(), full.dim(), full.T());
        return sampleON2Internal(full, td, M, rng, sw);
    }

    /**
     * Workspace-accepting overload of {@link #sampleON2(ParticleHistory, TransitionDensity, int, RandomBatch)}.
     *
     * <p>For each trajectory {@code m} and each time step {@code t = T-2 ... 0},
     * computes backward weights {@code w_t[n] * p(X_{t+1}^{idx[t+1,m]} | X_t^n)}
     * and draws from the resulting categorical distribution. All scratch lives
     * in {@code sw}; no heap allocation on the inner loop.
     *
     * @param history completed full particle history
     * @param td      transition density trait
     * @param M       number of smoothed trajectories to produce
     * @param rng     random number generator
     * @param sw      pre-allocated smoothing workspace
     * @return length {@code T * M} array of trajectory indices (row-major)
     */
    public static int[] sampleON2(ParticleHistory history, TransitionDensity<?> td,
                                  int M, RandomBatch rng, SmoothingWorkspace sw) {
        validateArgs(history, M);
        Full full = requireFull(history);
        sw.validateShape(M, full.N(), full.dim(), full.T());
        return sampleON2Internal(full, td, M, rng, sw);
    }

    private static int[] sampleON2Internal(Full full, TransitionDensity<?> td,
                                           int M, RandomBatch rng, SmoothingWorkspace sw) {
        int T = full.T();
        int N = full.N();
        int dim = full.dim();

        int[] idx = seedBackward(full, M, rng, sw);
        if (T == 1) return idx;

        double[] logWt = sw.logWt;
        double[] logPtBuf = sw.logPtBuf;
        double[] lwBack = sw.lwBack;
        double[] cums = sw.cums;
        double[] Xt = sw.Xt;
        double[] Xtp1Broadcast = sw.Xtp1Broadcast;
        double[] scratchCtx = sw.scratchCtx;

        for (int t = T - 2; t >= 0; t--) {
            // Load particles and weights at time t via zero-copy views
            full.viewX(t, Xt, 0);
            full.viewLogW(t, logWt, 0);

            int tOff = t * M;
            int tp1Off = (t + 1) * M;

            for (int m = 0; m < M; m++) {
                int tgtIdx = idx[tp1Off + m];

                // Broadcast X_{t+1}^target across all N columns
                broadcastParticle(full.rawX(), (t + 1) * dim * N, tgtIdx, dim, N, Xtp1Broadcast);

                // Compute log p(X_{t+1}^target | X_t^n) for all n
                @SuppressWarnings("unchecked")
                TransitionDensity<Object> tdCast = (TransitionDensity<Object>) td;
                StepContext<Object> ctx = new StepContext<>(
                        t + 1, null,
                        Xt, 0,              // Xprev = particles at time t
                        Xtp1Broadcast, 0,   // X = target particle broadcast
                        lwBack, 0,          // logW (unused by logPt)
                        N, dim, rng,
                        scratchCtx, 0
                );
                tdCast.logPt(ctx, logPtBuf, 0);

                // Backward weights: logW_t[n] + log p(X_{t+1} | X_t^n)
                for (int n = 0; n < N; n++) {
                    lwBack[n] = logWt[n] + logPtBuf[n];
                }

                // Draw from categorical(lwBack)
                idx[tOff + m] = multinomialOnce(lwBack, N, rng.asRandomGenerator(), cums);
            }
        }
        return idx;
    }

    // ------------------------------------------------------------------
    // Pure rejection sampling
    // ------------------------------------------------------------------

    /**
     * Pure rejection backward sampling using {@link TransitionDensity#upperBound}.
     * Allocates a fresh {@link SmoothingWorkspace} and delegates.
     */
    public static int[] sampleReject(ParticleHistory history, TransitionDensity<?> td,
                                     int M, int maxTrials, RandomBatch rng) {
        validateArgs(history, M);
        if (maxTrials <= 0) throw new IllegalArgumentException("maxTrials must be > 0: " + maxTrials);
        Full full = requireFull(history);
        SmoothingWorkspace sw = SmoothingWorkspace.allocate(M, full.N(), full.dim(), full.T());
        return rejectOrFallbackInternal(full, td, M, maxTrials, false, rng, sw);
    }

    /**
     * Workspace-accepting overload of {@link #sampleReject}.
     *
     * @param history   completed full particle history
     * @param td        transition density trait (must implement upperBound)
     * @param M         number of smoothed trajectories
     * @param maxTrials maximum rejection attempts per particle per time step
     * @param rng       random number generator
     * @param sw        pre-allocated smoothing workspace
     * @return length {@code T * M} array of trajectory indices (row-major)
     */
    public static int[] sampleReject(ParticleHistory history, TransitionDensity<?> td,
                                     int M, int maxTrials, RandomBatch rng,
                                     SmoothingWorkspace sw) {
        validateArgs(history, M);
        if (maxTrials <= 0) throw new IllegalArgumentException("maxTrials must be > 0: " + maxTrials);
        Full full = requireFull(history);
        sw.validateShape(M, full.N(), full.dim(), full.T());
        return rejectOrFallbackInternal(full, td, M, maxTrials, false, rng, sw);
    }

    // ------------------------------------------------------------------
    // MCMC backward sampling
    // ------------------------------------------------------------------

    /**
     * MCMC-based backward sampling using independent-proposal Metropolis–Hastings.
     * Allocates a fresh {@link SmoothingWorkspace} and delegates.
     */
    public static int[] sampleMcmc(ParticleHistory history, TransitionDensity<?> td,
                                   int M, int nSteps, RandomBatch rng) {
        validateArgs(history, M);
        if (nSteps <= 0) throw new IllegalArgumentException("nSteps must be > 0: " + nSteps);
        Full full = requireFull(history);
        SmoothingWorkspace sw = SmoothingWorkspace.allocate(M, full.N(), full.dim(), full.T());
        return sampleMcmcInternal(full, td, M, nSteps, rng, sw);
    }

    /**
     * Workspace-accepting overload of {@link #sampleMcmc}.
     *
     * @param history completed full particle history
     * @param td      transition density trait
     * @param M       number of smoothed trajectories
     * @param nSteps  number of MH iterations per (t, m) pair
     * @param rng     random number generator
     * @param sw      pre-allocated smoothing workspace
     * @return length {@code T * M} array of trajectory indices (row-major)
     */
    public static int[] sampleMcmc(ParticleHistory history, TransitionDensity<?> td,
                                   int M, int nSteps, RandomBatch rng,
                                   SmoothingWorkspace sw) {
        validateArgs(history, M);
        if (nSteps <= 0) throw new IllegalArgumentException("nSteps must be > 0: " + nSteps);
        Full full = requireFull(history);
        sw.validateShape(M, full.N(), full.dim(), full.T());
        return sampleMcmcInternal(full, td, M, nSteps, rng, sw);
    }

    private static int[] sampleMcmcInternal(Full full, TransitionDensity<?> td,
                                            int M, int nSteps, RandomBatch rng,
                                            SmoothingWorkspace sw) {
        int T = full.T();
        int N = full.N();
        int dim = full.dim();

        int[] idx = seedBackward(full, M, rng, sw);
        if (T == 1) return idx;

        double[] logWt = sw.logWt;
        double[] Xt = sw.Xt;
        int[] aTp1 = sw.aTp1;

        double[] xPrev1 = sw.xPrev1;
        double[] xCur1 = sw.xCur1;
        double[] logPt1 = sw.logPt1;
        double[] logPt2 = sw.logPt2;
        double[] scratch1 = sw.scratch1;

        int[] proposals = sw.proposals;
        double[] propScratch = sw.propScratch;

        for (int t = T - 2; t >= 0; t--) {
            full.viewX(t, Xt, 0);
            full.viewLogW(t, logWt, 0);

            int tOff = t * M;
            int tp1Off = (t + 1) * M;

            // Initialise: use forward ancestor index. When step (t+1) was
            // not resampled, ancestors are the identity permutation so
            // aTp1[k] == k and the assignment collapses to a direct copy
            // — skip the viewAncestors synthesis pass entirely.
            if (full.resampledAt(t + 1)) {
                full.viewAncestors(t + 1, aTp1, 0);
                for (int m = 0; m < M; m++) {
                    idx[tOff + m] = aTp1[idx[tp1Off + m]];
                }
            } else {
                for (int m = 0; m < M; m++) {
                    idx[tOff + m] = idx[tp1Off + m];
                }
            }

            @SuppressWarnings("unchecked")
            TransitionDensity<Object> tdCast = (TransitionDensity<Object>) td;

            for (int step = 0; step < nSteps; step++) {
                // Draw M proposals from forward weights
                Resample.apply(Scheme.MULTINOMIAL, logWt, N, M, proposals, rng, propScratch);

                for (int m = 0; m < M; m++) {
                    int tgtIdx = idx[tp1Off + m];
                    int currIdx = idx[tOff + m];
                    int propIdx = proposals[m];

                    // Extract target particle at t+1
                    extractParticle(full.rawX(), (t + 1) * dim * N, tgtIdx, dim, N, xCur1);

                    // Compute log p(X_{t+1}^target | X_t^prop)
                    extractParticle(Xt, 0, propIdx, dim, N, xPrev1);
                    StepContext<Object> ctxProp = new StepContext<>(
                            t + 1, null,
                            xPrev1, 0, xCur1, 0,
                            logPt1, 0, 1, dim, rng, scratch1, 0
                    );
                    tdCast.logPt(ctxProp, logPt1, 0);

                    // Compute log p(X_{t+1}^target | X_t^curr)
                    extractParticle(Xt, 0, currIdx, dim, N, xPrev1);
                    StepContext<Object> ctxCurr = new StepContext<>(
                            t + 1, null,
                            xPrev1, 0, xCur1, 0,
                            logPt2, 0, 1, dim, rng, scratch1, 0
                    );
                    tdCast.logPt(ctxCurr, logPt2, 0);

                    // MH acceptance
                    double logAlpha = logPt1[0] - logPt2[0];
                    if (Math.log(rng.asRandomGenerator().nextDouble()) < logAlpha) {
                        idx[tOff + m] = propIdx;
                    }
                }
            }
        }
        return idx;
    }

    // ------------------------------------------------------------------
    // Hybrid (rejection + ON² fallback) — recommended default
    // ------------------------------------------------------------------

    /**
     * Hybrid backward sampling: attempts rejection sampling with
     * {@code M} trials, then falls back to O(N²) for any remaining
     * un-accepted particles. Recommended default. Allocates a fresh
     * {@link SmoothingWorkspace} per call.
     */
    public static int[] sampleHybrid(ParticleHistory history, TransitionDensity<?> td,
                                     int M, RandomBatch rng) {
        validateArgs(history, M);
        Full full = requireFull(history);
        SmoothingWorkspace sw = SmoothingWorkspace.allocate(M, full.N(), full.dim(), full.T());
        return rejectOrFallbackInternal(full, td, M, M, true, rng, sw);
    }

    /**
     * Workspace-accepting overload of {@link #sampleHybrid}.
     *
     * @param history completed full particle history
     * @param td      transition density trait (must implement upperBound)
     * @param M       number of smoothed trajectories
     * @param rng     random number generator
     * @param sw      pre-allocated smoothing workspace
     * @return length {@code T * M} array of trajectory indices (row-major)
     */
    public static int[] sampleHybrid(ParticleHistory history, TransitionDensity<?> td,
                                     int M, RandomBatch rng, SmoothingWorkspace sw) {
        validateArgs(history, M);
        Full full = requireFull(history);
        sw.validateShape(M, full.N(), full.dim(), full.T());
        return rejectOrFallbackInternal(full, td, M, M, true, rng, sw);
    }

    // ------------------------------------------------------------------
    // Convenience entry point matching design spec signature
    // ------------------------------------------------------------------

    /**
     * Produces {@code M} smoothed trajectories using the hybrid
     * (rejection + fallback) strategy. This is the canonical entry
     * point matching the design document signature. Allocates a fresh
     * {@link SmoothingWorkspace} per call.
     */
    public static int[] sample(ParticleHistory history, TransitionDensity<?> td,
                               int M, RandomBatch rng) {
        return sampleHybrid(history, td, M, rng);
    }

    /**
     * Workspace-accepting overload of {@link #sample}. Dispatches to
     * {@link #sampleHybrid(ParticleHistory, TransitionDensity, int, RandomBatch, SmoothingWorkspace)}.
     */
    public static int[] sample(ParticleHistory history, TransitionDensity<?> td,
                               int M, RandomBatch rng, SmoothingWorkspace sw) {
        return sampleHybrid(history, td, M, rng, sw);
    }

    // ------------------------------------------------------------------
    // Internal: rejection or fallback implementation
    // ------------------------------------------------------------------

    private static int[] rejectOrFallbackInternal(Full full, TransitionDensity<?> td,
                                                  int M, int maxTrials, boolean fallbackToOn2,
                                                  RandomBatch rng, SmoothingWorkspace sw) {
        int T = full.T();
        int N = full.N();
        int dim = full.dim();

        int[] idx = seedBackward(full, M, rng, sw);
        if (T == 1) return idx;

        double[] logWt = sw.logWt;
        double[] Xt = sw.Xt;

        // Single-particle buffers for rejection
        double[] xPrev1 = sw.xPrev1;
        double[] xCur1 = sw.xCur1;
        double[] logPt1 = sw.logPt1;
        double[] scratch1 = sw.scratch1;

        // Proposal buffer for rejection
        int[] props = sw.proposals;
        double[] propScratch = sw.propScratch;
        boolean[] accepted = sw.accepted;

        // ON² fallback buffers
        double[] logPtBuf = sw.logPtBuf;
        double[] lwBack = sw.lwBack;
        double[] cums = sw.cums;
        double[] Xtp1Broadcast = sw.Xtp1Broadcast;
        double[] scratchCtx = sw.scratchCtx;

        @SuppressWarnings("unchecked")
        TransitionDensity<Object> tdCast = (TransitionDensity<Object>) td;

        for (int t = T - 2; t >= 0; t--) {
            full.viewX(t, Xt, 0);
            full.viewLogW(t, logWt, 0);

            int tOff = t * M;
            int tp1Off = (t + 1) * M;

            // Get upper bound for rejection
            double upb = td.upperBound(t + 1);

            // Reset acceptance flags
            int remaining = M;
            for (int i = 0; i < M; i++) accepted[i] = false;

            // Rejection phase
            if (!Double.isNaN(upb) && !Double.isInfinite(upb)) {
                int tries = 0;
                while (remaining > 0 && tries < maxTrials) {
                    tries++;
                    // Draw proposals from forward weights for remaining particles
                    Resample.apply(Scheme.MULTINOMIAL, logWt, N, remaining, props, rng, propScratch);

                    int writer = 0;
                    for (int m = 0; m < M; m++) {
                        if (accepted[m]) continue;
                        int tgtIdx = idx[tp1Off + m];
                        int propIdx = props[writer++];

                        // Extract target particle at t+1
                        extractParticle(full.rawX(), (t + 1) * dim * N, tgtIdx, dim, N, xCur1);
                        // Extract proposed particle at t
                        extractParticle(Xt, 0, propIdx, dim, N, xPrev1);

                        // Compute log p(X_{t+1}^target | X_t^prop)
                        StepContext<Object> ctx = new StepContext<>(
                                t + 1, null,
                                xPrev1, 0, xCur1, 0,
                                logPt1, 0, 1, dim, rng, scratch1, 0
                        );
                        tdCast.logPt(ctx, logPt1, 0);

                        // Accept with probability exp(logPt - upb)
                        double logAcceptProb = logPt1[0] - upb;
                        double logU = Math.log(rng.asRandomGenerator().nextDouble());
                        if (logU < logAcceptProb) {
                            idx[tOff + m] = propIdx;
                            accepted[m] = true;
                            remaining--;
                        }
                    }
                }
            }

            // Fallback to ON² for remaining particles
            if (remaining > 0) {
                if (!fallbackToOn2) {
                    throw new IllegalStateException(
                            "Rejection smoothing exhausted " + maxTrials
                                    + " trials at t=" + t + " with " + remaining
                                    + " particle(s) still un-sampled; use sampleHybrid or increase maxTrials.");
                }

                for (int m = 0; m < M; m++) {
                    if (accepted[m]) continue;
                    int tgtIdx = idx[tp1Off + m];

                    // Broadcast target particle across all N slots
                    broadcastParticle(full.rawX(), (t + 1) * dim * N, tgtIdx, dim, N, Xtp1Broadcast);

                    // Compute log p(X_{t+1}^target | X_t^n) for all n
                    StepContext<Object> ctx = new StepContext<>(
                            t + 1, null,
                            Xt, 0,
                            Xtp1Broadcast, 0,
                            lwBack, 0,
                            N, dim, rng,
                            scratchCtx, 0
                    );
                    tdCast.logPt(ctx, logPtBuf, 0);

                    // Backward weights
                    for (int n = 0; n < N; n++) {
                        lwBack[n] = logWt[n] + logPtBuf[n];
                    }

                    idx[tOff + m] = multinomialOnce(lwBack, N, rng.asRandomGenerator(), cums);
                }
            }
        }
        return idx;
    }

    // ------------------------------------------------------------------
    // Shared helpers
    // ------------------------------------------------------------------

    private static void validateArgs(ParticleHistory history, int M) {
        if (history == null) throw new IllegalArgumentException("history must not be null");
        if (M <= 0) throw new IllegalArgumentException("M must be > 0: " + M);
    }

    private static Full requireFull(ParticleHistory history) {
        if (!(history instanceof Full full)) {
            throw new IllegalArgumentException(
                    "FFBS requires a Full particle history, got: " + history.getClass().getSimpleName());
        }
        return full;
    }

    /**
     * Seeds the backward pass by drawing M indices from the final-step
     * log-weights via multinomial resampling. Uses {@code sw.lwBuf},
     * {@code sw.seedScratch}, and {@code sw.lastRow} for scratch.
     */
    private static int[] seedBackward(Full full, int M, RandomBatch rng, SmoothingWorkspace sw) {
        int T = full.T();
        int N = full.N();
        int[] idx = new int[T * M];

        // Copy final log-weights
        full.viewLogW(T - 1, sw.lwBuf, 0);

        // Multinomial draw of M indices from final weights
        Resample.apply(Scheme.MULTINOMIAL, sw.lwBuf, N, M, sw.lastRow, rng, sw.seedScratch);
        System.arraycopy(sw.lastRow, 0, idx, (T - 1) * M, M);
        return idx;
    }

    /**
     * Extracts a single particle from a (dim, N) buffer at the given index.
     * Layout: buffer[baseOff + j*N + particleIdx] for j in [0, dim).
     */
    private static void extractParticle(double[] buffer, int baseOff, int particleIdx,
                                        int dim, int N, double[] out) {
        for (int j = 0; j < dim; j++) {
            out[j] = buffer[baseOff + j * N + particleIdx];
        }
    }

    /**
     * Broadcasts a single particle from a (dim, N) buffer at the given index
     * across all N columns of the output buffer.
     * Input layout: buffer[baseOff + j*N + particleIdx] for j in [0, dim).
     * Output layout: out[j*N + n] = value for all n in [0, N).
     */
    private static void broadcastParticle(double[] buffer, int baseOff, int particleIdx,
                                          int dim, int N, double[] out) {
        for (int j = 0; j < dim; j++) {
            double v = buffer[baseOff + j * N + particleIdx];
            int rowOff = j * N;
            for (int n = 0; n < N; n++) {
                out[rowOff + n] = v;
            }
        }
    }

    /**
     * Draws a single index from a categorical distribution defined by
     * log-weights. Uses a cumulative-sum approach with no allocation.
     */
    private static int multinomialOnce(double[] lw, int N, RandomGenerator g, double[] cums) {
        // Find max for numerical stability
        double m = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < N; i++) {
            if (lw[i] > m) m = lw[i];
        }
        if (m == Double.NEGATIVE_INFINITY) {
            return 0; // all weights are -inf; return 0 as fallback
        }

        // Compute cumulative exp(lw - max)
        double cum = 0.0;
        for (int i = 0; i < N; i++) {
            cum += Math.exp(lw[i] - m);
            cums[i] = cum;
        }

        // Draw uniform and find index via inverse CDF
        double u = g.nextDouble() * cum;
        int n = 0;
        while (n < N - 1 && cums[n] < u) n++;
        return n;
    }
}
