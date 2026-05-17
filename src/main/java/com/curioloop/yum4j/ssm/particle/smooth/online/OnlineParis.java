package com.curioloop.yum4j.ssm.particle.smooth.online;

import com.curioloop.yum4j.ssm.particle.diag.Collector;
import com.curioloop.yum4j.ssm.particle.engine.RunState;
import com.curioloop.yum4j.ssm.particle.engine.StepContext;
import com.curioloop.yum4j.ssm.particle.engine.Workspace;
import com.curioloop.yum4j.ssm.particle.model.TransitionDensity;
import com.curioloop.yum4j.math.VectorOps;

import java.util.random.RandomGenerator;

/**
 * Paris online smoothing collector (Olsson &amp; Westerborn 2017).
 *
 * <p>Implements hybrid rejection/exact backward sampling for online
 * smoothing. At each step, for each particle {@code n}, the collector
 * attempts to sample the backward ancestor using rejection sampling
 * with the ancestor-weight distribution as the proposal. If rejection
 * fails after {@code maxTrials} attempts, it falls back to the exact
 * O(N) method.
 *
 * <p>The model must implement {@link TransitionDensity#logPt} and
 * {@link TransitionDensity#upperBound}.
 *
 */
public final class OnlineParis implements Collector {

    /** Same contract as {@link OnlineSmoothNaive.AddFunc}. */
    public interface AddFunc {
        void psi0(Workspace ws, RunState rs, double[] out);
        void psi(Workspace ws, RunState rs, int t, double[] out);
    }

    private final AddFunc psi;
    private final TransitionDensity<?> td;
    private final int maxTrials;

    private double[] Phi;
    private double[] prevPhi;
    private double[] prevX;
    private double[] prevLw;
    private double[] estimates;
    private double[] psiBuf;
    private double[] lwBuf;
    private double[] xSlice;
    private int N;
    private int dim;
    private int T;

    /** Pre-allocated (dim) buffer for single-pair logPt: previous particle. */
    private double[] xPrev1;
    /** Pre-allocated (dim) buffer for single-pair logPt: current particle. */
    private double[] xCur1;
    /** Pre-allocated output buffer for single-pair logPt. */
    private double[] out1;

    /**
     * @param psi       additive function
     * @param td        transition density (must support logPt and upperBound)
     * @param maxTrials maximum rejection trials per particle (default: N)
     */
    public OnlineParis(AddFunc psi, TransitionDensity<?> td, int maxTrials) {
        if (psi == null) throw new IllegalArgumentException("psi must not be null");
        if (td == null) throw new IllegalArgumentException("td must not be null");
        if (maxTrials <= 0) throw new IllegalArgumentException("maxTrials must be > 0");
        this.psi = psi;
        this.td = td;
        this.maxTrials = maxTrials;
    }

    /**
     * Constructor with default maxTrials = N (set at attach time).
     */
    public OnlineParis(AddFunc psi, TransitionDensity<?> td) {
        if (psi == null) throw new IllegalArgumentException("psi must not be null");
        if (td == null) throw new IllegalArgumentException("td must not be null");
        this.psi = psi;
        this.td = td;
        this.maxTrials = -1; // sentinel: use N
    }

    @Override
    public void attach(Workspace ws, RunState rs) {
        this.N = ws.N;
        this.dim = ws.dim;
        this.T = Math.max(1, rs.T);
        this.Phi = new double[N];
        this.prevPhi = new double[N];
        this.prevX = new double[dim * N];
        this.prevLw = new double[N];
        this.estimates = new double[T];
        this.psiBuf = new double[N];
        this.lwBuf = new double[N];
        this.xSlice = new double[dim * N];
        this.xPrev1 = new double[dim];
        this.xCur1 = new double[dim];
        this.out1 = new double[1];
    }

    @Override
    public void afterMutation(Workspace ws, RunState rs, int t) {
        if (t == 0) {
            psi.psi0(ws, rs, Phi);
        } else {
            System.arraycopy(Phi, 0, prevPhi, 0, N);
            psi.psi(ws, rs, t, psiBuf);

            int trials = maxTrials > 0 ? maxTrials : N;
            double upb;
            try {
                upb = td.upperBound(t);
            } catch (UnsupportedOperationException e) {
                // No upper bound: fall back to exact for all particles.
                upb = Double.NaN;
            }

            RandomGenerator rng = ws.rng.asRandomGenerator();
            double[] X = ws.X;

            // For each target particle n, attempt rejection sampling.
            for (int n = 0; n < N; n++) {
                boolean accepted = false;

                if (!Double.isNaN(upb)) {
                    // Rejection sampling: propose from prevLw, accept with
                    // prob exp(logPt(t, xProp, x_n) - upb).
                    for (int trial = 0; trial < trials; trial++) {
                        int prop = sampleFromWeights(prevLw, rng);
                        double logPt = computeLogPt(t, prop, n, X);
                        double lpr = logPt - upb;
                        double lu = Math.log(rng.nextDouble());
                        if (lu < lpr) {
                            Phi[n] = prevPhi[prop] + psiBuf[n];
                            accepted = true;
                            break;
                        }
                    }
                }

                if (!accepted) {
                    // Exact O(N) fallback: compute full backward weights.
                    for (int k = 0; k < N; k++) {
                        System.arraycopy(X, n * dim, xSlice, k * dim, dim);
                    }

                    @SuppressWarnings("unchecked")
                    TransitionDensity<Object> tdObj = (TransitionDensity<Object>) td;
                    StepContext<Object> ctx = new StepContext<>(
                        t, null,
                        prevX, 0,
                        xSlice, 0,
                        lwBuf, 0,
                        N, dim,
                        ws.rng,
                        ws.scratch, 0
                    );
                    tdObj.logPt(ctx, lwBuf, 0);
                    VectorOps.add(lwBuf, 0, prevLw, 0, N);

                    // Normalise and compute weighted sum of prevPhi.
                    double m = VectorOps.max(lwBuf, 0, N);
                    if (m == Double.NEGATIVE_INFINITY) {
                        Phi[n] = psiBuf[n];
                    } else {
                        double sumW = 0.0;
                        double acc = 0.0;
                        for (int k = 0; k < N; k++) {
                            double w = Math.exp(lwBuf[k] - m);
                            sumW += w;
                            acc += w * prevPhi[k];
                        }
                        Phi[n] = acc / sumW + psiBuf[n];
                    }
                }
            }
        }

        if (t >= 0 && t < T) {
            estimates[t] = OnlineSmoothNaive.weightedAverage(ws.logW, Phi);
        }

        // Snapshot state for the next step.
        System.arraycopy(ws.X, 0, prevX, 0, dim * N);
        System.arraycopy(ws.logW, 0, prevLw, 0, N);
    }

    /** Per-step smoothing estimate; length {@code T}. */
    public double[] estimates() { return estimates; }

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    /**
     * Compute logPt(t, prevX[prop*dim..], X[n*dim..]) for a single pair.
     * Uses pre-allocated xPrev1, xCur1, out1 buffers.
     */
    private double computeLogPt(int t, int prop, int n, double[] X) {
        System.arraycopy(prevX, prop * dim, xPrev1, 0, dim);
        System.arraycopy(X, n * dim, xCur1, 0, dim);

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
    private int sampleFromWeights(double[] lw, RandomGenerator rng) {
        double m = VectorOps.max(lw, 0, N);
        if (m == Double.NEGATIVE_INFINITY) {
            return rng.nextInt(N);
        }
        double u = rng.nextDouble();
        double cumSum = 0.0;
        double total = 0.0;
        for (int k = 0; k < N; k++) {
            total += Math.exp(lw[k] - m);
        }
        double target = u * total;
        for (int k = 0; k < N; k++) {
            cumSum += Math.exp(lw[k] - m);
            if (cumSum >= target) return k;
        }
        return N - 1;
    }
}
