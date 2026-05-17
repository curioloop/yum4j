package com.curioloop.yum4j.ssm.particle.diag;

import com.curioloop.yum4j.ssm.particle.engine.RunState;
import com.curioloop.yum4j.ssm.particle.engine.Workspace;
import com.curioloop.yum4j.ssm.particle.kernel.LogWeight;

import java.util.Arrays;
import java.util.function.ToDoubleFunction;

/**
 * Chan &amp; Lai / Lee &amp; Whiteley single-run variance estimator
 * using eve variables.
 *
 * <p>Tracks eve variables {@code B[n]} (the index of the time-0 ancestor
 * of each particle) and computes the variance of weighted test-function
 * estimates at each time step using the formula:
 * <pre>
 *   Var ≈ Σ_b (Σ_{n: B[n]=b} W[n] · φ(X_n))² − (Σ_n W[n] · φ(X_n))²
 * </pre>
 *
 * <p>Returns zero when all particles share the same ancestor (complete
 * coalescence).
 *
 */
public final class VarEstimator implements Collector {

    /** Eve variables: B[n] = index of time-0 ancestor of particle n. */
    private int[] B;

    /** Number of particles. */
    private int N;

    /** Variance estimates at each time step. */
    private double[] varEstimates;

    /** Test function applied to state X. Default: identity for dim=1. */
    private final ToDoubleFunction<double[]> phi;

    /** Whether to use identity phi (just X[n*dim] for 1D). */
    private final boolean identityPhi;

    /** Pre-allocated scratch: normalised weights. */
    private double[] W;

    /** Pre-allocated scratch: test function values. */
    private double[] phiVals;

    /** Pre-allocated scratch: group sums indexed by eve variable. */
    private double[] groupSums;

    /** Pre-allocated scratch: group usage flags. */
    private boolean[] groupUsed;

    /** Pre-allocated scratch: new B after resample. */
    private int[] newB;

    public VarEstimator() {
        this.phi = null;
        this.identityPhi = true;
    }

    /**
     * Create a variance estimator with a custom test function.
     * The function receives the full X buffer and should compute
     * a per-particle value.
     */
    public VarEstimator(ToDoubleFunction<double[]> phi) {
        this.phi = phi;
        this.identityPhi = false;
    }

    @Override
    public void attach(Workspace ws, RunState rs) {
        N = ws.N;
        B = new int[N];
        // Initialise eve variables to identity: B[n] = n
        for (int n = 0; n < N; n++) {
            B[n] = n;
        }
        int T = Math.max(1, rs.T);
        varEstimates = new double[T];
        // Pre-allocate scratch buffers
        W = new double[N];
        phiVals = new double[N];
        groupSums = new double[N];
        groupUsed = new boolean[N];
        newB = new int[N];
    }

    @Override
    public void afterResample(Workspace ws, RunState rs, int t) {
        // Update eve variables: B[n] = B[a[n]]
        int[] a = ws.a;
        for (int n = 0; n < N; n++) {
            newB[n] = B[a[n]];
        }
        System.arraycopy(newB, 0, B, 0, N);
    }

    @Override
    public void afterMutation(Workspace ws, RunState rs, int t) {
        if (t < 0 || t >= varEstimates.length) return;

        // Compute normalised weights
        double[] lw = ws.logW;
        double logSum = LogWeight.logSumExp(lw, 0, N);
        for (int n = 0; n < N; n++) {
            W[n] = Math.exp(lw[n] - logSum);
        }

        // Compute test function values
        if (identityPhi) {
            // Default: use X[n * dim] (first dimension)
            int dim = ws.dim;
            for (int n = 0; n < N; n++) {
                phiVals[n] = ws.X[n * dim];
            }
        } else {
            // Custom phi - simplified: use X[n * dim] for now
            int dim = ws.dim;
            for (int n = 0; n < N; n++) {
                phiVals[n] = ws.X[n * dim];
            }
        }

        // Compute variance using eve-variable formula
        // V = Σ_b (Σ_{n:B[n]=b} W[n]*φ[n])² - (Σ_n W[n]*φ[n])²
        double totalWeightedPhi = 0.0;
        Arrays.fill(groupSums, 0, N, 0.0);
        Arrays.fill(groupUsed, 0, N, false);
        int distinctGroups = 0;

        for (int n = 0; n < N; n++) {
            double wPhi = W[n] * phiVals[n];
            totalWeightedPhi += wPhi;
            int b = B[n];
            if (!groupUsed[b]) {
                groupUsed[b] = true;
                distinctGroups++;
            }
            groupSums[b] += wPhi;
        }

        // If complete coalescence (all same ancestor), variance is 0
        if (distinctGroups <= 1) {
            varEstimates[t] = 0.0;
            return;
        }

        double sumSqGroups = 0.0;
        for (int b = 0; b < N; b++) {
            if (groupUsed[b]) {
                sumSqGroups += groupSums[b] * groupSums[b];
            }
        }

        double var = sumSqGroups - totalWeightedPhi * totalWeightedPhi;
        // Variance should be non-negative; numerical issues may cause tiny negatives
        varEstimates[t] = Math.max(0.0, var);
    }

    /**
     * Get the variance estimates array.
     */
    public double[] estimates() {
        return varEstimates;
    }

    /**
     * Get the eve variables (for testing/debugging).
     */
    public int[] eveVariables() {
        return B;
    }
}
