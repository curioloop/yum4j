package com.curioloop.yum4j.ssm.particle.smooth;

import com.curioloop.yum4j.ssm.particle.engine.StepContext;
import com.curioloop.yum4j.ssm.particle.kernel.LogWeight;
import com.curioloop.yum4j.ssm.particle.kernel.RandomBatch;
import com.curioloop.yum4j.ssm.particle.model.TransitionDensity;
import com.curioloop.yum4j.ssm.particle.resample.Resample;
import com.curioloop.yum4j.ssm.particle.resample.Scheme;

/**
 * Two-filter smoothing.
 *
 * <p>Combines a forward filter's {@link Full} history with a backward
 * information filter's {@link Full} history to produce smoothed estimates.
 * At a given time {@code t} the smoothing distribution is obtained by
 * combining forward particles at {@code t}, information-filter particles
 * at {@code t + 1} (time-reversed to row {@code ti = T - 2 - t}), and
 * the transition density {@code p(x_{t+1} | x_t)}.
 *
 * <p>Two variants are provided:
 *
 * <ul>
 *   <li>{@link #smoothON2} — O(N²) double loop; computes exact smoothing
 *       weights for all (forward, backward) particle pairs.</li>
 *   <li>{@link #smoothON} — O(N) subsampling variant; draws N pairs
 *       independently from the forward and info weights, then reweights
 *       by the transition density.</li>
 * </ul>
 *
 * <p>Both methods return trajectory indices suitable for reconstructing
 * smoothed state estimates from the history.
 *
 * <p>Each entry point offers two overloads: a convenience form that
 * allocates fresh scratch per call, and a
 * {@link SmoothingWorkspace}-accepting form that reuses caller-supplied
 * scratch across every {@code (t, m)} iteration (R5.2, R5.3).
 *
 * @see Full
 * @see TransitionDensity
 */
public final class TwoFilter {

    private TwoFilter() {}

    /**
     * Test function on two particle states → one scalar.
     */
    @FunctionalInterface
    public interface Phi {
        /**
         * Evaluates the test function on a pair of particle states.
         *
         * @param xt   state at time t (length dim)
         * @param xtp1 state at time t+1 (length dim)
         * @return scalar value of the test function
         */
        double apply(double[] xt, double[] xtp1);
    }

    /**
     * Log auxiliary density {@code log γ(x_{t+1})} required for the
     * two-filter weighting correction.
     */
    @FunctionalInterface
    public interface LogGamma {
        /**
         * Evaluates {@code log γ} for N particles.
         *
         * @param X      particle buffer
         * @param xOff   offset into X
         * @param N      number of particles
         * @param out    output buffer for log-gamma values
         * @param outOff offset into out
         */
        void apply(double[] X, int xOff, int N, double[] out, int outOff);
    }

    /**
     * O(N²) two-filter smoothing estimate at time {@code t}. Allocates
     * its own scratch.
     */
    public static <Y> double smoothON2(Full forward, Full infoHist, int t,
                                       TransitionDensity<Y> td, Phi phi,
                                       LogGamma logGamma, RandomBatch rng) {
        validateShapes(forward, infoHist, t);
        SmoothingWorkspace sw = SmoothingWorkspace.allocate(
                Math.max(forward.N(), 1), forward.N(), forward.dim(), forward.T());
        return smoothON2Internal(forward, infoHist, t, td, phi, logGamma, rng, sw);
    }

    /**
     * Workspace-accepting overload of {@link #smoothON2}. The workspace
     * {@code sw.Xt}, {@code sw.Xtp1Broadcast}, {@code sw.logWt},
     * {@code sw.lwBack} (info weights), {@code sw.logPtBuf}, and
     * {@code sw.scratchCtx} are reused across the N² loop.
     *
     * @param forward  forward-filter history (Full)
     * @param infoHist information-filter history (reversed time axis)
     * @param t        target time in {@code [0, T - 2]}
     * @param td       transition density
     * @param phi      test function of {@code (x_t, x_{t+1})}
     * @param logGamma auxiliary density for two-filter correction
     * @param rng      random batch (used for building StepContext)
     * @param sw       pre-allocated smoothing workspace
     * @param <Y>      observation type
     * @return the smoothing estimate of {@code E[phi(X_t, X_{t+1})]}
     */
    public static <Y> double smoothON2(Full forward, Full infoHist, int t,
                                       TransitionDensity<Y> td, Phi phi,
                                       LogGamma logGamma, RandomBatch rng,
                                       SmoothingWorkspace sw) {
        validateShapes(forward, infoHist, t);
        sw.validateShape(1, forward.N(), forward.dim(), forward.T());
        return smoothON2Internal(forward, infoHist, t, td, phi, logGamma, rng, sw);
    }

    private static <Y> double smoothON2Internal(Full forward, Full infoHist, int t,
                                                TransitionDensity<Y> td, Phi phi,
                                                LogGamma logGamma, RandomBatch rng,
                                                SmoothingWorkspace sw) {
        int T = forward.T();
        int N = forward.N();
        int dim = forward.dim();
        int ti = T - 2 - t;

        // Load forward particles and weights at time t (views into workspace)
        double[] Xt = sw.Xt;
        double[] lwFwd = sw.logWt;
        forward.viewX(t, Xt, 0);
        forward.viewLogW(t, lwFwd, 0);

        // Load info-filter particles and weights at reversed time
        // Reuse sw.Xtp1Broadcast as the Xtp1 storage here (we never need
        // the broadcast form simultaneously on this path).
        double[] Xtp1 = sw.Xtp1Broadcast;
        double[] lwInfo = sw.lwBack;
        infoHist.viewX(ti, Xtp1, 0);
        infoHist.viewLogW(ti, lwInfo, 0);

        // Subtract logGamma from info weights (result written back into lwInfo)
        double[] logGammaBuf = sw.logPtBuf;
        logGamma.apply(Xtp1, 0, N, logGammaBuf, 0);
        for (int n = 0; n < N; n++) {
            lwInfo[n] -= logGammaBuf[n];
        }

        // logPt output reuses logPtBuf after logGamma is consumed
        double[] logPtBuf = sw.logPtBuf;
        // N²-pair broadcast buffer: allocate locally since sw.Xtp1Broadcast is
        // already used for Xtp1. This is a one-shot per smoothON2 call (not a
        // per-(t, m) hot loop), so the allocation is acceptable.
        double[] broadcast = new double[dim * N];
        double[] scratch = sw.scratchCtx;

        double sp = 0.0;
        double sw2 = 0.0;

        // Find max for numerical stability
        double maxLwFwd = Double.NEGATIVE_INFINITY;
        for (double v : lwFwd) if (v > maxLwFwd) maxLwFwd = v;
        double maxLwInfo = Double.NEGATIVE_INFINITY;
        for (double v : lwInfo) if (v > maxLwInfo) maxLwInfo = v;
        double shift = maxLwFwd + maxLwInfo;

        double[] xtSingle = sw.xPrev1;
        double[] xtp1Single = sw.xCur1;

        for (int n = 0; n < N; n++) {
            // Broadcast forward particle n across all N columns
            for (int j = 0; j < dim; j++) {
                double v = Xt[j * N + n];
                int rowOff = j * N;
                for (int k = 0; k < N; k++) {
                    broadcast[rowOff + k] = v;
                }
            }

            // Evaluate logPt(t+1, x_t[n], x_{t+1}[k]) for all k
            StepContext<Y> ctx = new StepContext<>(
                    t + 1, null,
                    broadcast, 0,
                    Xtp1, 0,
                    logPtBuf, 0,
                    N, dim, rng, scratch, 0
            );
            td.logPt(ctx, logPtBuf, 0);

            // Accumulate weighted phi values
            for (int k = 0; k < N; k++) {
                double lomega = lwFwd[n] + lwInfo[k] + logPtBuf[k] - shift;
                double omega = Math.exp(lomega);
                // Extract single particle states
                for (int j = 0; j < dim; j++) {
                    xtSingle[j] = Xt[j * N + n];
                    xtp1Single[j] = Xtp1[j * N + k];
                }
                sp += omega * phi.apply(xtSingle, xtp1Single);
                sw2 += omega;
            }
        }
        return (sw2 == 0.0) ? 0.0 : sp / sw2;
    }

    /**
     * O(N) two-filter smoother via subsampling. Allocates its own scratch.
     */
    public static <Y> double smoothON(Full forward, Full infoHist, int t,
                                      TransitionDensity<Y> td, Phi phi,
                                      LogGamma logGamma, RandomBatch rng,
                                      double[] modifForward, double[] modifInfo) {
        validateShapes(forward, infoHist, t);
        SmoothingWorkspace sw = SmoothingWorkspace.allocate(
                forward.N(), forward.N(), forward.dim(), forward.T());
        return smoothONInternal(forward, infoHist, t, td, phi, logGamma, rng,
                modifForward, modifInfo, sw);
    }

    /**
     * Workspace-accepting overload of {@link #smoothON}. Requires
     * {@code sw.M >= N} and {@code sw.N >= N}.
     *
     * @param forward      forward-filter history
     * @param infoHist     information-filter history (reversed time axis)
     * @param t            target time in {@code [0, T - 2]}
     * @param td           transition density
     * @param phi          test function of {@code (x_t, x_{t+1})}
     * @param logGamma     {@code log γ(x_{t+1})} auxiliary density
     * @param rng          random batch generator
     * @param modifForward optional additive modifier for forward log-weights (length N), or null
     * @param modifInfo    optional additive modifier for info log-weights (length N), or null
     * @param sw           pre-allocated smoothing workspace
     * @param <Y>          observation type
     * @return the smoothing estimate of {@code E[phi(X_t, X_{t+1})]}
     */
    public static <Y> double smoothON(Full forward, Full infoHist, int t,
                                      TransitionDensity<Y> td, Phi phi,
                                      LogGamma logGamma, RandomBatch rng,
                                      double[] modifForward, double[] modifInfo,
                                      SmoothingWorkspace sw) {
        validateShapes(forward, infoHist, t);
        sw.validateShape(forward.N(), forward.N(), forward.dim(), forward.T());
        return smoothONInternal(forward, infoHist, t, td, phi, logGamma, rng,
                modifForward, modifInfo, sw);
    }

    private static <Y> double smoothONInternal(Full forward, Full infoHist, int t,
                                               TransitionDensity<Y> td, Phi phi,
                                               LogGamma logGamma, RandomBatch rng,
                                               double[] modifForward, double[] modifInfo,
                                               SmoothingWorkspace sw) {
        int T = forward.T();
        int N = forward.N();
        int dim = forward.dim();
        int ti = T - 2 - t;

        // Load particles and log-weights via zero-copy views into workspace
        double[] Xt = sw.Xt;
        double[] Xtp1 = sw.Xtp1Broadcast;
        double[] lwFwd = sw.logWt;
        double[] lwInfo = sw.lwBack;
        forward.viewX(t, Xt, 0);
        infoHist.viewX(ti, Xtp1, 0);
        forward.viewLogW(t, lwFwd, 0);
        infoHist.viewLogW(ti, lwInfo, 0);

        // Subtract logGamma from info weights
        double[] logGammaBuf = sw.logPtBuf;
        logGamma.apply(Xtp1, 0, N, logGammaBuf, 0);
        for (int n = 0; n < N; n++) {
            lwInfo[n] -= logGammaBuf[n];
        }

        // Step 1: add modifiers to info weights and draw I ~ Multinomial(Winfo).
        // lwInfoMod and lwFwdMod are small (length N) one-shot buffers; allocate
        // locally since the workspace does not reserve a dedicated modifier slot.
        double[] lwInfoMod = new double[N];
        System.arraycopy(lwInfo, 0, lwInfoMod, 0, N);
        if (modifInfo != null) {
            for (int n = 0; n < N; n++) lwInfoMod[n] += modifInfo[n];
        }
        int[] I = new int[N];
        double[] scratch = sw.propScratch; // size max(N + M + 1, 2 * N), M >= N here
        Resample.apply(Scheme.MULTINOMIAL, lwInfoMod, N, N, I, rng, scratch);

        // Step 2: add modifiers to forward weights and draw J ~ Multinomial(W)
        double[] lwFwdMod = new double[N];
        System.arraycopy(lwFwd, 0, lwFwdMod, 0, N);
        if (modifForward != null) {
            for (int n = 0; n < N; n++) lwFwdMod[n] += modifForward[n];
        }
        int[] J = new int[N];
        Resample.apply(Scheme.MULTINOMIAL, lwFwdMod, N, N, J, rng, scratch);

        // Step 3: gather selected particles into contiguous buffers for batch logPt.
        // XtJ and Xtp1I are O(dim * N) buffers; allocate locally (workspace does not
        // reserve separate gather slots for this path).
        double[] XtJ = new double[dim * N];
        double[] Xtp1I = new double[dim * N];
        for (int n = 0; n < N; n++) {
            for (int j = 0; j < dim; j++) {
                XtJ[j * N + n] = Xt[j * N + J[n]];
                Xtp1I[j * N + n] = Xtp1[j * N + I[n]];
            }
        }

        // Step 4: compute log_omega[n] = logPt(t+1, XtJ[n], Xtp1I[n]) - modifiers
        double[] logOmega = sw.logPtBuf;
        double[] logPtScratch = sw.scratchCtx;
        StepContext<Y> ctx = new StepContext<>(
                t + 1, null,
                XtJ, 0,
                Xtp1I, 0,
                logOmega, 0,
                N, dim, rng, logPtScratch, 0
        );
        td.logPt(ctx, logOmega, 0);

        if (modifForward != null) {
            for (int n = 0; n < N; n++) logOmega[n] -= modifForward[J[n]];
        }
        if (modifInfo != null) {
            for (int n = 0; n < N; n++) logOmega[n] -= modifInfo[I[n]];
        }

        // Step 5: normalise omega weights and compute weighted average
        double logSumOmega = LogWeight.logSumExp(logOmega, 0, N);
        double[] xtSingle = sw.xPrev1;
        double[] xtp1Single = sw.xCur1;
        double result = 0.0;
        for (int n = 0; n < N; n++) {
            double wn = Math.exp(logOmega[n] - logSumOmega);
            for (int j = 0; j < dim; j++) {
                xtSingle[j] = XtJ[j * N + n];
                xtp1Single[j] = Xtp1I[j * N + n];
            }
            result += wn * phi.apply(xtSingle, xtp1Single);
        }
        return result;
    }

    /**
     * Draws {@code M} smoothed trajectory indices using the two-filter
     * approach. Allocates a fresh {@link SmoothingWorkspace} per call.
     */
    public static <Y> int[] sample(Full forward, Full infoHist,
                                   TransitionDensity<Y> td, LogGamma logGamma,
                                   int M, RandomBatch rng) {
        if (M <= 0) throw new IllegalArgumentException("M must be > 0: " + M);
        int T = forward.T();
        if (infoHist.T() != T) {
            throw new IllegalArgumentException(
                    "info-filter horizon must match forward horizon; forward T=" + T
                            + ", info T=" + infoHist.T());
        }
        SmoothingWorkspace sw = SmoothingWorkspace.allocate(M, forward.N(), forward.dim(), T);
        return sampleInternal(forward, infoHist, td, logGamma, M, rng, sw);
    }

    /**
     * Workspace-accepting overload of {@link #sample}.
     *
     * @param forward  forward-filter history (Full)
     * @param infoHist information-filter history (reversed time axis)
     * @param td       transition density
     * @param logGamma auxiliary density for two-filter correction
     * @param M        number of trajectories to sample
     * @param rng      random batch generator
     * @param sw       pre-allocated smoothing workspace
     * @param <Y>      observation type
     * @return flat {@code int[T * M]} array of trajectory indices
     */
    public static <Y> int[] sample(Full forward, Full infoHist,
                                   TransitionDensity<Y> td, LogGamma logGamma,
                                   int M, RandomBatch rng, SmoothingWorkspace sw) {
        if (M <= 0) throw new IllegalArgumentException("M must be > 0: " + M);
        int T = forward.T();
        if (infoHist.T() != T) {
            throw new IllegalArgumentException(
                    "info-filter horizon must match forward horizon; forward T=" + T
                            + ", info T=" + infoHist.T());
        }
        sw.validateShape(M, forward.N(), forward.dim(), T);
        return sampleInternal(forward, infoHist, td, logGamma, M, rng, sw);
    }

    private static <Y> int[] sampleInternal(Full forward, Full infoHist,
                                            TransitionDensity<Y> td, LogGamma logGamma,
                                            int M, RandomBatch rng,
                                            SmoothingWorkspace sw) {
        int T = forward.T();
        int N = forward.N();
        int dim = forward.dim();

        int[] idx = new int[T * M];

        // At the last time step T-1, sample from the forward weights directly
        forward.viewLogW(T - 1, sw.lwBuf, 0);
        Resample.apply(Scheme.MULTINOMIAL, sw.lwBuf, N, M, sw.lastRow, rng, sw.seedScratch);
        System.arraycopy(sw.lastRow, 0, idx, (T - 1) * M, M);

        if (T == 1) return idx;

        // For intermediate time steps t in [0, T-2], use two-filter weights.
        // Map workspace buffers to the local names used by the backward loop.
        double[] logPtBuf = sw.logPtBuf;
        double[] broadcast = sw.Xtp1Broadcast;
        double[] smoothW = sw.lwBack;
        double[] logGammaBuf = sw.cums;              // reuse cums for logGamma output
        double[] Xt = sw.Xt;
        double[] lwFwd = sw.logWt;
        // Xtp1 and lwInfo need their own slots distinct from Xt/logWt/broadcast.
        // The workspace doesn't reserve dedicated info-filter slots; allocate
        // the two small arrays once at the top of the backward loop.
        double[] Xtp1 = new double[dim * N];
        double[] lwInfo = new double[N];
        double[] lwInfoCorr = new double[N];

        for (int t = T - 2; t >= 0; t--) {
            int ti = T - 2 - t;

            forward.viewX(t, Xt, 0);
            forward.viewLogW(t, lwFwd, 0);
            infoHist.viewX(ti, Xtp1, 0);
            infoHist.viewLogW(ti, lwInfo, 0);

            // Subtract logGamma from info weights
            logGamma.apply(Xtp1, 0, N, logGammaBuf, 0);
            for (int n = 0; n < N; n++) {
                lwInfoCorr[n] = lwInfo[n] - logGammaBuf[n];
            }

            // For each trajectory m, compute smoothing weights for forward particles
            // conditioned on the backward particle at t+1
            for (int m = 0; m < M; m++) {
                int tgtIdx = idx[(t + 1) * M + m];

                // Broadcast backward target particle across all N columns
                for (int j = 0; j < dim; j++) {
                    double v = Xtp1[j * N + tgtIdx];
                    int rowOff = j * N;
                    for (int k = 0; k < N; k++) {
                        broadcast[rowOff + k] = v;
                    }
                }

                // Evaluate logPt(t+1, x_t[n], x_{t+1}[tgt]) for all forward particles n
                StepContext<Y> ctx = new StepContext<>(
                        t + 1, null,
                        Xt, 0,
                        broadcast, 0,
                        logPtBuf, 0,
                        N, dim, rng, sw.scratchCtx, 0
                );
                td.logPt(ctx, logPtBuf, 0);

                // Smoothing weight for forward particle n:
                // w_smooth[n] = lwFwd[n] + logPt(x_{t+1}[tgt] | x_t[n])
                for (int n = 0; n < N; n++) {
                    smoothW[n] = lwFwd[n] + logPtBuf[n];
                }

                // Draw one index from smoothing weights
                idx[t * M + m] = multinomialOnce(smoothW, N, rng);
            }
        }
        return idx;
    }

    private static void validateShapes(Full forward, Full infoHist, int t) {
        int T = forward.T();
        if (t < 0 || t >= T - 1) {
            throw new IllegalArgumentException(
                    "t must be in [0, T - 2]: t=" + t + " T=" + T);
        }
        if (infoHist.T() != T) {
            throw new IllegalArgumentException(
                    "info-filter horizon must match forward horizon; forward T=" + T
                            + ", info T=" + infoHist.T());
        }
    }

    /**
     * Draws a single index from log-weights using the inverse-CDF method.
     */
    private static int multinomialOnce(double[] lw, int N, RandomBatch rng) {
        double m = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < N; i++) {
            if (lw[i] > m) m = lw[i];
        }
        if (m == Double.NEGATIVE_INFINITY) {
            return 0;
        }
        double cum = 0.0;
        for (int i = 0; i < N; i++) {
            cum += Math.exp(lw[i] - m);
        }
        double u = rng.asRandomGenerator().nextDouble() * cum;
        double running = 0.0;
        for (int i = 0; i < N; i++) {
            running += Math.exp(lw[i] - m);
            if (running >= u) return i;
        }
        return N - 1;
    }
}
