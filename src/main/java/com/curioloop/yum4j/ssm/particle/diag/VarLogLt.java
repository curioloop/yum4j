package com.curioloop.yum4j.ssm.particle.diag;

import com.curioloop.yum4j.ssm.particle.engine.RunState;
import com.curioloop.yum4j.ssm.particle.engine.Workspace;
import com.curioloop.yum4j.ssm.particle.kernel.LogWeight;

import java.util.Arrays;

/**
 * Variance of log normalising constant estimator.
 *
 * <p>Computes a cumulative estimate of {@code Var(log L_t)} at each time
 * step using the delta method and the eve-variable variance of the
 * normalising constant increments.
 *
 */
public final class VarLogLt implements Collector {

    /** Eve variables: B[n] = index of time-0 ancestor of particle n. */
    private int[] B;

    /** Number of particles. */
    private int N;

    /** Cumulative variance estimates of log L_t. */
    private double[] varLogLt;

    /** Reusable buffer for eve-variable update. */
    private int[] newB;

    /** Reusable buffer for normalised weights. */
    private double[] W;

    /** Reusable buffer for group sums. */
    private double[] groupSums;

    /** Reusable buffer for group-used flags. */
    private boolean[] groupUsed;

    @Override
    public void attach(Workspace ws, RunState rs) {
        N = ws.N;
        B = new int[N];
        for (int n = 0; n < N; n++) {
            B[n] = n;
        }
        int T = Math.max(1, rs.T);
        varLogLt = new double[T];
        newB = new int[N];
        W = new double[N];
        groupSums = new double[N];
        groupUsed = new boolean[N];
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
    public void afterReweight(Workspace ws, RunState rs, int t) {
        if (t < 0 || t >= varLogLt.length) return;

        // Compute normalised weights
        double[] lw = ws.logW;
        double logSum = LogWeight.logSumExp(lw, 0, N);
        for (int n = 0; n < N; n++) {
            W[n] = Math.exp(lw[n] - logSum);
        }

        // Compute variance of the normalising constant increment
        // using eve-variable formula on the weights themselves
        Arrays.fill(groupSums, 0, N, 0.0);
        Arrays.fill(groupUsed, 0, N, false);
        int distinctGroups = 0;

        for (int n = 0; n < N; n++) {
            int b = B[n];
            if (!groupUsed[b]) {
                groupUsed[b] = true;
                distinctGroups++;
            }
            groupSums[b] += W[n];
        }

        if (distinctGroups <= 1) {
            // Complete coalescence: no variance contribution
            varLogLt[t] = t > 0 ? varLogLt[t - 1] : 0.0;
            return;
        }

        // Var(L_t/L_{t-1}) ≈ Σ_b (groupSum_b)² - 1
        double sumSqGroups = 0.0;
        for (int b = 0; b < N; b++) {
            if (groupUsed[b]) {
                sumSqGroups += groupSums[b] * groupSums[b];
            }
        }
        double varIncrement = Math.max(0.0, sumSqGroups - 1.0);

        // Cumulative: Var(log L_t) ≈ Var(log L_{t-1}) + varIncrement
        // (delta method approximation)
        double prev = t > 0 ? varLogLt[t - 1] : 0.0;
        varLogLt[t] = prev + varIncrement;
    }

    /**
     * Get the cumulative variance estimates of log L_t.
     */
    public double[] estimates() {
        return varLogLt;
    }
}
