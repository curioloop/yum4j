package com.curioloop.yum4j.ssm.particle.smooth;

import com.curioloop.yum4j.ssm.particle.engine.StepContext;
import com.curioloop.yum4j.ssm.particle.kernel.RandomBatch;
import com.curioloop.yum4j.ssm.particle.model.TransitionDensity;

import java.util.random.RandomGenerator;

/**
 * Paris-style online additive smoother (Olsson &amp; Westerborn 2017).
 *
 * <p>Computes smoothed additive functionals online without storing the
 * full particle history. At each time step, for each particle {@code n},
 * the smoother samples a backward ancestor using rejection sampling from
 * the transition density, then updates the additive functional.
 *
 * <p>The key idea: given the current particles {@code X_t} and the
 * previous particles {@code X_{t-1}} with weights {@code W_{t-1}}, for
 * each particle {@code n} at time {@code t}, sample a backward ancestor
 * {@code B_n} from the backward kernel:
 * <pre>
 *   P(B_n = k | X_t^n) ∝ W_{t-1}^k · p(X_t^n | X_{t-1}^k)
 * </pre>
 * using rejection sampling with the transition density upper bound
 * {@code M_t} such that {@code p(x_t | x_{t-1}) ≤ M_t} for all
 * {@code (x_t, x_{t-1})}. The acceptance probability is:
 * <pre>
 *   α = p(X_t^n | X_{t-1}^{prop}) / M_t
 * </pre>
 * where the proposal is drawn from the normalised weights {@code W_{t-1}}.
 *
 * <p>This smoother can work with {@link Partial} history (only ancestor
 * indices needed) or {@link Full} history. It requires the model to
 * implement {@link TransitionDensity#logPt} and
 * {@link TransitionDensity#upperBound}.
 *
 * <p>Each smoother entry offers two overloads: a convenience form that
 * allocates fresh scratch per call, and a
 * {@link SmoothingWorkspace}-accepting form that reuses the
 * history-read buffers. Paris-specific per-step buffers
 * ({@code prevPhi}, {@code newPhi}) are still allocated locally since
 * they are not shared with the other smoothers (R5.2, R5.3).
 *
 * @see TransitionDensity
 * @see Partial
 */
public final class Paris {

    private Paris() {}

    /**
     * Additive functional definition. Provides the per-step contribution
     * to the smoothed functional.
     */
    @FunctionalInterface
    public interface AdditiveFunction {
        /**
         * Evaluates the additive contribution at time {@code t} for
         * particle {@code n} given its state.
         *
         * @param X     particle state buffer
         * @param xOff  offset into X for particle n (= n * dim)
         * @param dim   state dimension
         * @param t     time index
         * @return the additive contribution ψ_t(X_t^n)
         */
        double evaluate(double[] X, int xOff, int dim, int t);
    }

    /**
     * Result of the Paris smoother containing per-step smoothing estimates.
     *
     * @param estimates per-step smoothing estimates (length T)
     * @param Phi       final per-particle additive functional values (length N)
     */
    public record Result(double[] estimates, double[] Phi) {}

    /**
     * Runs the Paris online additive smoother using a freshly allocated
     * {@link SmoothingWorkspace}. Delegates to
     * {@link #smooth(Full, TransitionDensity, AdditiveFunction, RandomBatch, int, SmoothingWorkspace)}.
     */
    public static Result smooth(Full history, TransitionDensity<?> td,
                                AdditiveFunction psi, RandomBatch rng,
                                int maxTrials) {
        if (history == null) throw new IllegalArgumentException("history must not be null");
        if (td == null) throw new IllegalArgumentException("td must not be null");
        if (psi == null) throw new IllegalArgumentException("psi must not be null");
        if (rng == null) throw new IllegalArgumentException("rng must not be null");

        SmoothingWorkspace sw = SmoothingWorkspace.allocate(
                Math.max(history.N(), 1), history.N(), history.dim(), history.capacity());
        return smoothInternal(history, td, psi, rng, maxTrials, sw);
    }

    /**
     * Workspace-accepting overload of
     * {@link #smooth(Full, TransitionDensity, AdditiveFunction, RandomBatch, int)}.
     *
     * <p>The workspace supplies {@code sw.Xt} / {@code sw.Xtp1Broadcast}
     * for the forward particle reads, {@code sw.logWt} / {@code sw.lwBack}
     * for the log-weight reads, {@code sw.logPtBuf} for logPt outputs,
     * and {@code sw.xPrev1} / {@code sw.xCur1} / {@code sw.logPt1} /
     * {@code sw.scratch1} for the per-proposal rejection evaluations.
     * The Paris-specific {@code Phi} / {@code prevPhi} / {@code estimates}
     * arrays are allocated once at the top of the call (they are not
     * shared with FFBS / TwoFilter / FixedLag).
     *
     * @param history   full particle history from a completed forward pass
     * @param td        transition density (must support logPt and upperBound)
     * @param psi       additive function to smooth
     * @param rng       random number generator
     * @param maxTrials maximum rejection trials per particle per step
     *                  (0 or negative means use N as default)
     * @param sw        pre-allocated smoothing workspace
     * @return smoothing result with per-step estimates and final Phi values
     */
    public static Result smooth(Full history, TransitionDensity<?> td,
                                AdditiveFunction psi, RandomBatch rng,
                                int maxTrials, SmoothingWorkspace sw) {
        if (history == null) throw new IllegalArgumentException("history must not be null");
        if (td == null) throw new IllegalArgumentException("td must not be null");
        if (psi == null) throw new IllegalArgumentException("psi must not be null");
        if (rng == null) throw new IllegalArgumentException("rng must not be null");
        sw.validateShape(1, history.N(), history.dim(), history.capacity());
        return smoothInternal(history, td, psi, rng, maxTrials, sw);
    }

    private static Result smoothInternal(Full history, TransitionDensity<?> td,
                                         AdditiveFunction psi, RandomBatch rng,
                                         int maxTrials, SmoothingWorkspace sw) {
        int N = history.N();
        int dim = history.dim();
        int T = history.capacity();
        int trials = maxTrials > 0 ? maxTrials : N;

        // Paris-specific (no suitable workspace slot): allocated once per call.
        double[] Phi = new double[N];
        double[] prevPhi = new double[N];
        double[] estimates = new double[T];

        // Workspace-backed buffers for backward sampling and history reads.
        double[] prevX = sw.Xt;
        double[] prevLw = sw.logWt;
        double[] curX = sw.Xtp1Broadcast;
        double[] curLw = sw.lwBack;
        double[] xPrev1 = sw.xPrev1;
        double[] xCur1 = sw.xCur1;
        double[] out1 = sw.logPt1;
        double[] lwBuf = sw.logPtBuf;
        // xSlice (dim * N) is reused from sw.scratchCtx when dim * N <= 2 * N,
        // i.e. dim <= 2. For higher dim, allocate locally once per call.
        double[] xSlice = (dim * N <= sw.scratchCtx.length) ? sw.scratchCtx : new double[dim * N];

        RandomGenerator g = rng.asRandomGenerator();

        // t = 0: initialise Phi with psi_0
        history.viewX(0, curX, 0);
        history.viewLogW(0, curLw, 0);

        for (int n = 0; n < N; n++) {
            Phi[n] = psi.evaluate(curX, n * dim, dim, 0);
        }
        estimates[0] = weightedAverage(curLw, Phi, N);

        // t = 1 .. T-1: backward sampling + update
        for (int t = 1; t < T; t++) {
            // Snapshot previous state
            System.arraycopy(curX, 0, prevX, 0, dim * N);
            System.arraycopy(curLw, 0, prevLw, 0, N);
            System.arraycopy(Phi, 0, prevPhi, 0, N);

            // Load current step
            history.viewX(t, curX, 0);
            history.viewLogW(t, curLw, 0);

            // Get upper bound for rejection sampling
            double upb;
            try {
                upb = td.upperBound(t);
            } catch (UnsupportedOperationException e) {
                upb = Double.NaN;
            }

            // For each particle n at time t, sample backward ancestor
            for (int n = 0; n < N; n++) {
                double psiN = psi.evaluate(curX, n * dim, dim, t);
                boolean accepted = false;

                if (!Double.isNaN(upb)) {
                    // Rejection sampling: propose from prevLw, accept with
                    // prob exp(logPt(t, xProp, x_n) - upb)
                    for (int trial = 0; trial < trials; trial++) {
                        int prop = sampleFromWeights(prevLw, N, g);
                        double logPt = computeLogPt(td, t, prevX, prop, curX, n, dim, xPrev1, xCur1, out1);
                        double lpr = logPt - upb;
                        double lu = Math.log(g.nextDouble());
                        if (lu < lpr) {
                            Phi[n] = prevPhi[prop] + psiN;
                            accepted = true;
                            break;
                        }
                    }
                }

                if (!accepted) {
                    // Exact O(N) fallback: compute full backward weights
                    // Broadcast current particle n across all N slots
                    for (int k = 0; k < N; k++) {
                        System.arraycopy(curX, n * dim, xSlice, k * dim, dim);
                    }

                    @SuppressWarnings("unchecked")
                    TransitionDensity<Object> tdObj = (TransitionDensity<Object>) td;
                    StepContext<Object> ctx = new StepContext<>(
                        t, null,
                        prevX, 0,
                        xSlice, 0,
                        lwBuf, 0,
                        N, dim,
                        rng,
                        null, 0
                    );
                    tdObj.logPt(ctx, lwBuf, 0);

                    // Add prevLw to get backward weights
                    for (int k = 0; k < N; k++) {
                        lwBuf[k] += prevLw[k];
                    }

                    // Compute weighted sum of prevPhi
                    double m = max(lwBuf, N);
                    if (m == Double.NEGATIVE_INFINITY) {
                        Phi[n] = psiN;
                    } else {
                        double sumW = 0.0;
                        double acc = 0.0;
                        for (int k = 0; k < N; k++) {
                            double w = Math.exp(lwBuf[k] - m);
                            sumW += w;
                            acc += w * prevPhi[k];
                        }
                        Phi[n] = acc / sumW + psiN;
                    }
                }
            }

            estimates[t] = weightedAverage(curLw, Phi, N);
        }

        return new Result(estimates, Phi);
    }

    /**
     * Runs the Paris smoother with default maxTrials = N.
     *
     * @see #smooth(Full, TransitionDensity, AdditiveFunction, RandomBatch, int)
     */
    public static Result smooth(Full history, TransitionDensity<?> td,
                                AdditiveFunction psi, RandomBatch rng) {
        return smooth(history, td, psi, rng, 0);
    }

    /**
     * Workspace-accepting default-maxTrials overload.
     *
     * @see #smooth(Full, TransitionDensity, AdditiveFunction, RandomBatch, int, SmoothingWorkspace)
     */
    public static Result smooth(Full history, TransitionDensity<?> td,
                                AdditiveFunction psi, RandomBatch rng,
                                SmoothingWorkspace sw) {
        return smooth(history, td, psi, rng, 0, sw);
    }

    /**
     * Samples a single backward ancestor index for a given target particle
     * using the Paris rejection kernel.
     *
     * <p>This is the core backward sampling kernel exposed for use by
     * other smoothers or collectors that need Paris-style backward sampling.
     *
     * @param prevX     previous-step particle states (dim * N)
     * @param prevLw    previous-step log-weights (length N)
     * @param curX      current particle state (length dim, single particle)
     * @param td        transition density
     * @param t         current time index
     * @param N         number of particles
     * @param dim       state dimension
     * @param maxTrials maximum rejection attempts
     * @param rng       random number generator
     * @return sampled ancestor index in [0, N), or -1 if rejection failed
     */
    public static int backwardSampleOne(double[] prevX, double[] prevLw,
                                        double[] curX,
                                        TransitionDensity<?> td,
                                        int t, int N, int dim,
                                        int maxTrials, RandomGenerator rng) {
        double upb;
        try {
            upb = td.upperBound(t);
        } catch (UnsupportedOperationException e) {
            return -1; // No upper bound available
        }

        double[] xPrev1 = new double[dim];
        double[] xCur1 = new double[dim];
        double[] out1 = new double[1];

        for (int trial = 0; trial < maxTrials; trial++) {
            int prop = sampleFromWeights(prevLw, N, rng);
            System.arraycopy(prevX, prop * dim, xPrev1, 0, dim);
            System.arraycopy(curX, 0, xCur1, 0, dim);

            @SuppressWarnings("unchecked")
            TransitionDensity<Object> tdObj = (TransitionDensity<Object>) td;
            StepContext<Object> ctx = new StepContext<>(
                t, null,
                xPrev1, 0,
                xCur1, 0,
                out1, 0,
                1, dim,
                null,
                null, 0
            );
            tdObj.logPt(ctx, out1, 0);

            double lpr = out1[0] - upb;
            double lu = Math.log(rng.nextDouble());
            if (lu < lpr) {
                return prop;
            }
        }
        return -1; // Rejection failed
    }

    /**
     * Samples a backward ancestor with exact O(N) fallback when rejection fails.
     *
     * @param prevX     previous-step particle states (dim * N)
     * @param prevLw    previous-step log-weights (length N)
     * @param curX      current particle state (length dim, single particle)
     * @param td        transition density
     * @param t         current time index
     * @param N         number of particles
     * @param dim       state dimension
     * @param maxTrials maximum rejection attempts before fallback
     * @param rng       random number generator
     * @return sampled ancestor index in [0, N)
     */
    public static int backwardSampleOneExact(double[] prevX, double[] prevLw,
                                             double[] curX,
                                             TransitionDensity<?> td,
                                             int t, int N, int dim,
                                             int maxTrials, RandomGenerator rng) {
        int result = backwardSampleOne(prevX, prevLw, curX, td, t, N, dim, maxTrials, rng);
        if (result >= 0) {
            return result;
        }

        // Exact O(N) fallback: compute full backward weights
        double[] lwBuf = new double[N];
        double[] xSlice = new double[dim * N];

        // Broadcast curX across all N slots
        for (int k = 0; k < N; k++) {
            System.arraycopy(curX, 0, xSlice, k * dim, dim);
        }

        @SuppressWarnings("unchecked")
        TransitionDensity<Object> tdObj = (TransitionDensity<Object>) td;
        StepContext<Object> ctx = new StepContext<>(
            t, null,
            prevX, 0,
            xSlice, 0,
            lwBuf, 0,
            N, dim,
            null,
            null, 0
        );
        tdObj.logPt(ctx, lwBuf, 0);

        // Add prevLw to get backward weights
        for (int k = 0; k < N; k++) {
            lwBuf[k] += prevLw[k];
        }

        return sampleFromWeights(lwBuf, N, rng);
    }

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    /**
     * Compute logPt(t, prevX[prop*dim..], curX[n*dim..]) for a single pair.
     */
    private static double computeLogPt(TransitionDensity<?> td, int t,
                                       double[] prevX, int prop,
                                       double[] curX, int n, int dim,
                                       double[] xPrev1, double[] xCur1,
                                       double[] out1) {
        System.arraycopy(prevX, prop * dim, xPrev1, 0, dim);
        System.arraycopy(curX, n * dim, xCur1, 0, dim);

        @SuppressWarnings("unchecked")
        TransitionDensity<Object> tdObj = (TransitionDensity<Object>) td;
        StepContext<Object> ctx = new StepContext<>(
            t, null,
            xPrev1, 0,
            xCur1, 0,
            out1, 0,
            1, dim,
            null,
            null, 0
        );
        tdObj.logPt(ctx, out1, 0);
        return out1[0];
    }

    /**
     * Sample a single index from log-weights using the inverse-CDF method.
     */
    private static int sampleFromWeights(double[] lw, int N, RandomGenerator rng) {
        double m = max(lw, N);
        if (m == Double.NEGATIVE_INFINITY) {
            return rng.nextInt(N);
        }
        double u = rng.nextDouble();
        double total = 0.0;
        for (int k = 0; k < N; k++) {
            total += Math.exp(lw[k] - m);
        }
        double target = u * total;
        double cumSum = 0.0;
        for (int k = 0; k < N; k++) {
            cumSum += Math.exp(lw[k] - m);
            if (cumSum >= target) return k;
        }
        return N - 1;
    }

    /**
     * Compute weighted average of values using log-weights.
     */
    private static double weightedAverage(double[] lw, double[] values, int N) {
        double m = max(lw, N);
        if (m == Double.NEGATIVE_INFINITY) {
            return 0.0;
        }
        double sumW = 0.0;
        double acc = 0.0;
        for (int k = 0; k < N; k++) {
            double w = Math.exp(lw[k] - m);
            sumW += w;
            acc += w * values[k];
        }
        return sumW > 0.0 ? acc / sumW : 0.0;
    }

    /**
     * Find max of array.
     */
    private static double max(double[] x, int N) {
        double m = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < N; i++) {
            if (x[i] > m) m = x[i];
        }
        return m;
    }
}
