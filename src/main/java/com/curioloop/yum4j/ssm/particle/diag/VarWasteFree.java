package com.curioloop.yum4j.ssm.particle.diag;

import com.curioloop.yum4j.ssm.particle.engine.RunState;
import com.curioloop.yum4j.ssm.particle.engine.Workspace;
import com.curioloop.yum4j.ssm.particle.kernel.LogWeight;

/**
 * Waste-free MCMC variance estimator.
 *
 * <p>Reshapes the weighted test-function values into a {@code (P, M)}
 * matrix (P = len_chain, M = N_resampled) and applies the initial-sequence
 * estimator of Geyer (1992) to estimate the variance of the weighted mean.
 *
 * <p>This collector is designed for waste-free SMC runs where the total
 * number of particles is {@code N = M * P}.
 *
 */
public final class VarWasteFree implements Collector {

    /** Chain length P (len_chain). */
    private final int lenChain;

    /** Number of particles N (= M * P). */
    private int N;

    /** Variance estimates at each time step. */
    private double[] varEstimates;

    /** Variance of log L_t estimates. */
    private double[] varLogLtEstimates;

    /** Reusable buffer for normalised weights. */
    private double[] W;

    /** Reusable buffer for weighted test-function values. */
    private double[] weightedPhi;

    /** Reusable buffer for reshaped log-weights. */
    private double[] lwReshaped;

    public VarWasteFree() {
        this.lenChain = 1;
    }

    public VarWasteFree(int lenChain) {
        this.lenChain = lenChain;
    }

    @Override
    public void attach(Workspace ws, RunState rs) {
        N = ws.N;
        int T = Math.max(1, rs.T);
        varEstimates = new double[T];
        varLogLtEstimates = new double[T];
        W = new double[N];
        weightedPhi = new double[N];
        lwReshaped = new double[N];
    }

    @Override
    public void afterMutation(Workspace ws, RunState rs, int t) {
        if (t < 0 || t >= varEstimates.length) return;
        if (lenChain <= 1) {
            varEstimates[t] = 0.0;
            varLogLtEstimates[t] = 0.0;
            return;
        }

        int M = N / lenChain;
        if (M <= 0) {
            varEstimates[t] = 0.0;
            varLogLtEstimates[t] = 0.0;
            return;
        }

        // Compute normalised weights
        double[] lw = ws.logW;
        double logSum = LogWeight.logSumExp(lw, 0, N);
        for (int n = 0; n < N; n++) {
            W[n] = Math.exp(lw[n] - logSum);
        }

        // Test function values (first dimension of X)
        // Reshape into (P, M): values[p*M + m] = W[p*M + m] * X[(p*M + m) * dim]
        int dim = ws.dim;
        for (int n = 0; n < N; n++) {
            weightedPhi[n] = W[n] * ws.X[n * dim];
        }

        // Apply initial-sequence estimator on the (P, M) matrix
        varEstimates[t] = MCMCVariance.initialSequence(weightedPhi, lenChain, M);

        // Variance of log-weights (for VarLogLtWF)
        System.arraycopy(lw, 0, lwReshaped, 0, N);
        varLogLtEstimates[t] = MCMCVariance.initialSequence(lwReshaped, lenChain, M);
    }

    /**
     * Get the variance estimates for the test function.
     */
    public double[] estimates() {
        return varEstimates;
    }

    /**
     * Get the variance of log L_t estimates.
     */
    public double[] varLogLtEstimates() {
        return varLogLtEstimates;
    }
}
