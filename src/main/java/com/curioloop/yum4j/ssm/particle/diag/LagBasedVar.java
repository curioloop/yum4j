package com.curioloop.yum4j.ssm.particle.diag;

import com.curioloop.yum4j.ssm.particle.engine.RunState;
import com.curioloop.yum4j.ssm.particle.engine.Workspace;
import com.curioloop.yum4j.ssm.particle.kernel.LogWeight;
import com.curioloop.yum4j.ssm.particle.smooth.Rolling;

import java.util.Arrays;

/**
 * Olsson &amp; Douc lag-based variance estimator.
 *
 * <p>Computes variance estimates for lags 0, 1, ..., k simultaneously
 * at each time step using the ancestor indices at time {@code t − l}
 * from the {@link Rolling} history as eve variables.
 *
 * <p>Requires {@code store_history = ROLLING(k)} with {@code k ≥ 1}.
 * Throws {@link IllegalStateException} at attach time if this is not
 * satisfied.
 *
 * <p>At lag 0, produces the same estimate as the standard
 * {@link VarEstimator} (Chan &amp; Lai).
 *
 */
public final class LagBasedVar implements Collector {

    /** Maximum lag (= rolling window size k). */
    private int maxLag;

    /** Number of particles. */
    private int N;

    /** Variance estimates: flat array of length T * (maxLag+1), accessed as [t * (maxLag+1) + lag]. */
    private double[] varEstimates;

    /** Number of time steps. */
    private int T;

    /** Reference to the Rolling history. */
    private Rolling rolling;

    /** Eve variables for lag 0 (standard). */
    private int[] B0;

    /** Pre-allocated scratch: normalised weights. */
    private double[] W;

    /** Pre-allocated scratch: test function values. */
    private double[] phiVals;

    /** Pre-allocated scratch: group sums. */
    private double[] groupSums;

    /** Pre-allocated scratch: group usage flags. */
    private boolean[] groupUsed;

    /** Pre-allocated scratch: new B after resample. */
    private int[] newB;

    /** Pre-allocated scratch: ancestors buffer. */
    private int[] ancestors;

    /** Pre-allocated scratch: eve variables for lag computation. */
    private int[] eve;

    /** Pre-allocated scratch: new eve after tracing. */
    private int[] newEve;

    @Override
    public void attach(Workspace ws, RunState rs) {
        if (!(ws.history instanceof Rolling r)) {
            throw new IllegalStateException(
                "LagBasedVar requires ROLLING history mode, got " +
                    (ws.history == null ? "NONE" : ws.history.getClass().getSimpleName()));
        }
        this.rolling = r;
        this.maxLag = r.lag();
        this.N = ws.N;
        this.B0 = new int[N];
        for (int n = 0; n < N; n++) {
            B0[n] = n;
        }
        this.T = Math.max(1, rs.T);
        varEstimates = new double[T * (maxLag + 1)];
        // Pre-allocate scratch buffers
        W = new double[N];
        phiVals = new double[N];
        groupSums = new double[N];
        groupUsed = new boolean[N];
        newB = new int[N];
        ancestors = new int[N];
        eve = new int[N];
        newEve = new int[N];
    }

    @Override
    public void afterResample(Workspace ws, RunState rs, int t) {
        // Update lag-0 eve variables
        int[] a = ws.a;
        for (int n = 0; n < N; n++) {
            newB[n] = B0[a[n]];
        }
        System.arraycopy(newB, 0, B0, 0, N);
    }

    @Override
    public void afterMutation(Workspace ws, RunState rs, int t) {
        if (t < 0 || t >= T) return;

        int stride = maxLag + 1;

        // Compute normalised weights
        double[] lw = ws.logW;
        double logSum = LogWeight.logSumExp(lw, 0, N);
        for (int n = 0; n < N; n++) {
            W[n] = Math.exp(lw[n] - logSum);
        }

        // Test function values (first dimension of X)
        int dim = ws.dim;
        for (int n = 0; n < N; n++) {
            phiVals[n] = ws.X[n * dim];
        }

        // Lag 0: use B0 (standard eve variables)
        varEstimates[t * stride] = computeVarWithEve(W, phiVals, B0);

        // Lags 1..maxLag: use ancestors from history
        for (int lag = 1; lag <= maxLag; lag++) {
            int tLag = t - lag;
            if (tLag < 0 || !rolling.hasStep(tLag + 1)) {
                // Not enough history yet
                varEstimates[t * stride + lag] = varEstimates[t * stride];
                continue;
            }

            // Compute eve variables at this lag
            computeLagEveVariables(t, lag);
            varEstimates[t * stride + lag] = computeVarWithEve(W, phiVals, eve);
        }
    }

    /**
     * Compute eve variables at a given lag by tracing ancestry backwards.
     * Writes result into the pre-allocated {@code eve} buffer.
     */
    private void computeLagEveVariables(int t, int lag) {
        for (int n = 0; n < N; n++) {
            eve[n] = n;
        }

        // Trace backwards through the rolling history
        for (int step = t; step > t - lag && step > 0; step--) {
            if (rolling.hasStep(step)) {
                rolling.viewAncestors(step, ancestors, 0);
                for (int n = 0; n < N; n++) {
                    newEve[n] = ancestors[eve[n]];
                }
                System.arraycopy(newEve, 0, eve, 0, N);
            }
        }
    }

    /**
     * Compute variance estimate using given eve variables.
     */
    private double computeVarWithEve(double[] W, double[] phiVals, int[] eveVars) {
        double totalWeightedPhi = 0.0;
        Arrays.fill(groupSums, 0, N, 0.0);
        Arrays.fill(groupUsed, 0, N, false);
        int distinctGroups = 0;

        for (int n = 0; n < N; n++) {
            double wPhi = W[n] * phiVals[n];
            totalWeightedPhi += wPhi;
            int b = eveVars[n];
            if (!groupUsed[b]) {
                groupUsed[b] = true;
                distinctGroups++;
            }
            groupSums[b] += wPhi;
        }

        if (distinctGroups <= 1) return 0.0;

        double sumSqGroups = 0.0;
        for (int b = 0; b < N; b++) {
            if (groupUsed[b]) {
                sumSqGroups += groupSums[b] * groupSums[b];
            }
        }

        return Math.max(0.0, sumSqGroups - totalWeightedPhi * totalWeightedPhi);
    }

    /**
     * Get the variance estimates as a flat array of length {@code T * (maxLag+1)}.
     * Access as {@code estimates[t * (maxLag+1) + lag]}.
     */
    public double[] estimates() {
        return varEstimates;
    }

    /** Maximum lag. */
    public int maxLag() { return maxLag; }

    /** Number of time steps. */
    public int T() { return T; }
}
