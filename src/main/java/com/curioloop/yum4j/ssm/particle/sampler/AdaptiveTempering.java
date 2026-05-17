package com.curioloop.yum4j.ssm.particle.sampler;

import com.curioloop.yum4j.ssm.particle.sampler.moves.MCMCSequence;
import com.curioloop.yum4j.ssm.particle.engine.RunState;
import com.curioloop.yum4j.optim.root.BrentqSolver;

/**
 * Adaptive tempering SMC sampler that chooses exponents adaptively
 * to maintain ESS ≈ {@code essRmin * N}.
 *
 * <p>At each step, the next exponent γ_{t+1} is found by solving
 * {@code ESS(exp((γ_{t+1} − γ_t) · llik)) = essRmin · N} via Brent's
 * method. When γ = 1.0 satisfies the ESS threshold, the exponent is
 * set to 1.0 and the algorithm terminates.
 *
 * @see Tempering
 */
public final class AdaptiveTempering extends Tempering {

    /** Current exponent (starts at 0, ends at 1). */
    private double currentExponent = 0.0;

    /** Target ESS fraction for exponent selection. */
    private final double essTarget;

    /** Whether the algorithm has reached exponent 1.0. */
    private boolean finished = false;

    public AdaptiveTempering(StaticModel model, MCMCSequence move,
                             int N, int lenChain, double essRmin) {
        // Use a dummy exponents array; we compute adaptively
        super(model, move, N, lenChain, essRmin, new double[] { 0.0, 1.0 });
        this.essTarget = essRmin;
    }

    @Override
    public int horizon() {
        // Open-ended: use a large horizon, termination via done()
        return Integer.MAX_VALUE;
    }

    @Override
    public boolean done(RunState rs) {
        return finished;
    }

    @Override
    protected double getDelta(int t) {
        int N = theta.N;
        int llikOff = theta.llikOff();
        double targetEss = essTarget * N;

        // Pre-compute max(llik) once — reused by all computeEss calls
        // during Brent iterations (valid because delta > 0 preserves
        // the argmax).
        double maxLlik = Double.NEGATIVE_INFINITY;
        for (int n = 0; n < N; n++) {
            double v = theta.arena[llikOff + n];
            if (v > maxLlik) maxLlik = v;
        }

        // Check if jumping to 1.0 satisfies the ESS threshold
        double deltaToOne = 1.0 - currentExponent;
        double essAtOne = computeEss(theta.arena, llikOff, N, deltaToOne, maxLlik);

        if (essAtOne >= targetEss) {
            currentExponent = 1.0;
            finished = true;
            return deltaToOne;
        }

        // Brent for the right delta
        double delta = solveDelta(theta.arena, llikOff, N, targetEss, deltaToOne, maxLlik);
        currentExponent += delta;
        if (currentExponent >= 1.0 - 1e-12) {
            currentExponent = 1.0;
            finished = true;
        }
        return delta;
    }

    @Override
    protected double getCurrentExponent(int t) {
        return currentExponent;
    }

    /**
     * Compute ESS for a given delta applied to the log-likelihood values.
     * ESS of weights exp(delta * llik[n]).
     *
     * <p>Since delta > 0, max(delta * llik) = delta * max(llik), so the
     * caller can pass the pre-computed max(llik) to avoid a redundant
     * pass over the array on each Brent iteration.
     */
    private static double computeEss(double[] arena, int llikOff, int N,
                                     double delta, double maxLlik) {
        double maxLw = delta * maxLlik;
        if (maxLw == Double.NEGATIVE_INFINITY) return 0.0;
        double s1 = 0.0, s2 = 0.0;
        for (int n = 0; n < N; n++) {
            double e = Math.exp(delta * arena[llikOff + n] - maxLw);
            s1 += e;
            s2 += e * e;
        }
        return (s1 * s1) / s2;
    }

    /**
     * Solve for delta such that ESS(exp(delta * llik)) = targetEss
     * using Brent's method.
     */
    private static double solveDelta(double[] arena, int llikOff, int N,
                                     double targetEss, double maxDelta, double maxLlik) {
        double lo = 1e-12;
        double hi = maxDelta;
        var result = BrentqSolver.solve(
            d -> computeEss(arena, llikOff, N, d, maxLlik) - targetEss,
            lo, hi,
            BrentqSolver.DEFAULT_XTOL, BrentqSolver.DEFAULT_RTOL,
            BrentqSolver.DEFAULT_MAXITER);
        return result.root();
    }
}
