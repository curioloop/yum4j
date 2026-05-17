package com.curioloop.yum4j.ssm.particle.smooth.online;

import com.curioloop.yum4j.ssm.particle.diag.Collector;
import com.curioloop.yum4j.ssm.particle.engine.RunState;
import com.curioloop.yum4j.ssm.particle.engine.Workspace;
import com.curioloop.yum4j.math.VectorOps;

/**
 * Naive online smoothing collector.
 *
 * <p>Tracks, per particle, a running additive statistic along its
 * ancestor chain. At step {@code t = 0} this collector expects the
 * caller to supply an {@link AddFunc#psi0(Workspace, RunState, double[]) psi_0}
 * initialisation, and at step {@code t >= 1} an
 * {@link AddFunc#psi(Workspace, RunState, int, double[]) psi_t} increment.
 * The per-particle summary after step {@code t} is
 *
 * <pre>
 *   Phi_t[n] = Phi_{t-1}[a_t[n]] + psi_t(X_{t-1}[a_t[n]], X_t[n]).
 * </pre>
 *
 * <p>After every step the collector records the weighted average
 * {@code Σ_n W_n · Phi_t[n]} into {@link #estimates()}. Per the
 * reference {@code Online_smooth_naive}, this simple "naive" variant
 * has O(N) per-step cost but its Monte Carlo variance grows
 * quickly with {@code t} because the ancestor chains coalesce.
 *
 */
public final class OnlineSmoothNaive implements Collector {

    /** Additive function contract. */
    public interface AddFunc {
        /** psi_0(x_0); fills one value per particle into {@code out}. */
        void psi0(Workspace ws, RunState rs, double[] out);

        /** psi_t(x_{t-1}, x_t); fills one value per particle into {@code out}. */
        void psi(Workspace ws, RunState rs, int t, double[] out);
    }

    private final AddFunc psi;
    private double[] Phi;        // per-particle running sum
    private double[] scratch;    // reshuffle buffer
    private double[] incr;       // psi increment buffer (reused each step)
    private double[] estimates;  // per-step weighted average
    private int N;
    private int T;

    public OnlineSmoothNaive(AddFunc psi) {
        if (psi == null) throw new IllegalArgumentException("psi must not be null");
        this.psi = psi;
    }

    @Override
    public void attach(Workspace ws, RunState rs) {
        this.N = ws.N;
        this.T = Math.max(1, rs.T);
        this.Phi = new double[N];
        this.scratch = new double[N];
        this.incr = new double[N];
        this.estimates = new double[T];
    }

    @Override
    public void afterMutation(Workspace ws, RunState rs, int t) {
        if (t == 0) {
            psi.psi0(ws, rs, Phi);
        } else {
            // Phi[n] = Phi_prev[a[n]] + psi(t, Xprev, X)
            int[] a = ws.a;
            // Capture previous Phi before mutating.
            System.arraycopy(Phi, 0, scratch, 0, N);
            psi.psi(ws, rs, t, incr);
            for (int n = 0; n < N; n++) {
                Phi[n] = scratch[a[n]] + incr[n];
            }
        }
        if (t >= 0 && t < T) {
            estimates[t] = weightedAverage(ws.logW, Phi);
        }
    }

    /** Per-step smoothing estimate; length {@code T}. Null before attach. */
    public double[] estimates() { return estimates; }

    /** Latest per-particle running sum; length {@code N}. Null before attach. */
    public double[] runningSums() { return Phi; }

    static double weightedAverage(double[] lw, double[] phi) {
        int N = lw.length;
        double m = VectorOps.max(lw, 0, N);
        double sum = 0.0, sumW = 0.0;
        if (m == Double.NEGATIVE_INFINITY) {
            return VectorOps.mean(phi, 0, phi.length);
        }
        for (int n = 0; n < N; n++) {
            double w = Math.exp(lw[n] - m);
            sumW += w;
            sum += w * phi[n];
        }
        return sum / sumW;
    }
}
