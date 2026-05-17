package com.curioloop.yum4j.ssm.particle.smooth.online;

import com.curioloop.yum4j.ssm.particle.diag.Collector;
import com.curioloop.yum4j.ssm.particle.engine.RunState;
import com.curioloop.yum4j.ssm.particle.engine.Workspace;
import com.curioloop.yum4j.ssm.particle.model.TransitionDensity;
import com.curioloop.yum4j.ssm.particle.engine.StepContext;
import com.curioloop.yum4j.ssm.particle.kernel.RandomBatch;
import com.curioloop.yum4j.math.VectorOps;

/**
 * O(N²) online smoothing collector.
 *
 * <p>Implements the reference {@code Online_smooth_ON2} kernel: at
 * every step {@code t ≥ 1} the collector draws the single-particle
 * ancestor distribution from the {@code logPt(t, Xprev, x_t^{(n)})}
 * kernel of the supplied {@link TransitionDensity}. This is the textbook
 * unbiased O(N²) estimator; it is substantially more expensive than
 * {@link OnlineSmoothNaive} but much more accurate.
 *
 * <p>The model must implement {@link TransitionDensity#logPt}. If it
 * does not, the first call to {@link #afterMutation} will surface the
 * underlying {@link UnsupportedOperationException}.
 *
 */
public final class OnlineSmoothON2 implements Collector {

    /** Same contract as {@link OnlineSmoothNaive.AddFunc}. */
    public interface AddFunc {
        void psi0(Workspace ws, RunState rs, double[] out);
        void psi(Workspace ws, RunState rs, int t, double[] out);
    }

    private final AddFunc psi;
    private final TransitionDensity<?> td;
    private double[] Phi;
    private double[] prevPhi;
    private double[] prevX;
    private double[] prevLw;
    private double[] estimates;
    private double[] lwBuf;
    private double[] psiBuf;
    private double[] xSlice;
    private int N;
    private int dim;
    private int T;

    public OnlineSmoothON2(AddFunc psi, TransitionDensity<?> td) {
        if (psi == null) throw new IllegalArgumentException("psi must not be null");
        if (td == null) throw new IllegalArgumentException("td must not be null");
        this.psi = psi;
        this.td = td;
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
        this.lwBuf = new double[N];
        this.psiBuf = new double[N];
        this.xSlice = new double[dim * N];
    }

    @Override
    public void afterMutation(Workspace ws, RunState rs, int t) {
        if (t == 0) {
            psi.psi0(ws, rs, Phi);
        } else {
            System.arraycopy(Phi, 0, prevPhi, 0, N);
            double[] X = ws.X;
            psi.psi(ws, rs, t, psiBuf);

            for (int n = 0; n < N; n++) {
                // Broadcast X[n*dim .. n*dim+dim) across xSlice so logPt
                // produces N values (one per ancestor k) against the same target.
                for (int k = 0; k < N; k++) {
                    System.arraycopy(X, n * dim, xSlice, k * dim, dim);
                }

                // Build a StepContext for the logPt call with prevX as Xprev
                // and xSlice as X (N copies of target particle n).
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

                // Combine with prevLw.
                VectorOps.add(lwBuf, 0, prevLw, 0, N);
                // Normalise via max-shift.
                double m = VectorOps.max(lwBuf, 0, N);
                if (m == Double.NEGATIVE_INFINITY) {
                    Phi[n] = psiBuf[n];
                    continue;
                }
                double sumW = 0.0;
                double acc = 0.0;
                for (int k = 0; k < N; k++) {
                    double w = Math.exp(lwBuf[k] - m);
                    sumW += w;
                    acc += w * (prevPhi[k] + psiBuf[n]);
                }
                Phi[n] = acc / sumW;
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
}
