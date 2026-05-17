package com.curioloop.yum4j.ssm.particle.sampler;
import com.curioloop.yum4j.ssm.particle.sampler.moves.MCMCSequence;

import com.curioloop.yum4j.ssm.particle.engine.StepContext;

import java.util.function.Consumer;
import java.util.random.RandomGenerator;

/**
 * Fixed-exponent tempering SMC sampler.
 *
 * <p>The target at step t is {@code π_t(θ) ∝ prior(θ) · L(θ)^{γ_t}} where
 * γ_t increases from 0 to 1. The log-weight increment is
 * {@code logG(t) = (γ_t − γ_{t-1}) · llik(θ)}.
 *
 * <p>Operates directly on the engine's {@code Workspace.X} buffer via
 * {@link FKSMCSampler}'s arena aliasing — there is no side-car copy.
 *
 * <p>A path-sampling estimate of log Z is accumulated and exposed via
 * {@link ThetaParticles#pathSamplingLogZ}.
 *
 * @see FKSMCSampler
 * @see AdaptiveTempering
 */
public class Tempering extends FKSMCSampler {

    /** Exponent schedule γ_0, γ_1, ..., γ_T (length T+1). */
    protected double[] exponents;

    /** Differences δ_t = γ_t − γ_{t-1}. */
    protected double[] deltas;

    /** Path-sampling log Z accumulator. */
    protected double pathSamplingLogZ = 0.0;

    public Tempering(StaticModel model, MCMCSequence move,
                     int N, int lenChain, double essRmin,
                     double[] exponents) {
        super(model, move, N, lenChain, essRmin);
        this.exponents = exponents;
        this.deltas = new double[exponents.length - 1];
        for (int i = 0; i < deltas.length; i++) {
            deltas[i] = exponents[i + 1] - exponents[i];
        }
    }

    @Override
    public int horizon() {
        return deltas.length;
    }

    @Override
    protected void initializeParticles(int N) {
        int lpriorOff = theta.lpriorOff();
        int lpostOff = theta.lpostOff();
        int llikOff = theta.llikOff();
        // Fused lprior + full loglik evaluation.
        model.loglikAndLogprior(theta.arena, 0, N, -1,
            theta.arena, llikOff,
            theta.arena, lpriorOff);
        System.arraycopy(theta.arena, lpriorOff, theta.arena, lpostOff, N);

        theta.appendExponent(exponents[0]);
    }

    @Override
    protected void doLogG(int t, double[] Xprev, double[] X, int N,
                          double[] logW, int lwOff) {
        // Alias theta to the engine's X buffer directly.
        aliasTheta(X, N);

        double delta = getDelta(t);

        int llikOff = theta.llikOff();
        int lpostOff = theta.lpostOff();

        // logG = delta * llik; lpost += delta * llik; mean(llik) accumulator
        // Fused loop: avoid three passes over llik.
        double meanLlik = 0.0;
        for (int n = 0; n < N; n++) {
            double ll = theta.arena[llikOff + n];
            logW[lwOff + n] = delta * ll;
            theta.arena[lpostOff + n] += delta * ll;
            meanLlik += ll;
        }
        meanLlik /= N;
        pathSamplingLogZ += delta * meanLlik;

        // Record exponent and expose path-sampling logZ on the container.
        double currentExp = getCurrentExponent(t);
        theta.appendExponent(currentExp);
        theta.pathSamplingLogZ = pathSamplingLogZ;

        rsFlag = true; // Tempering always resamples
    }

    /**
     * Get the exponent delta for step t. Subclasses (AdaptiveTempering)
     * override this to compute adaptively.
     */
    protected double getDelta(int t) {
        if (t < deltas.length) {
            return deltas[t];
        }
        return 0.0;
    }

    /**
     * Get the current exponent after step t.
     */
    protected double getCurrentExponent(int t) {
        if (t < exponents.length - 1) {
            return exponents[t + 1];
        }
        return 1.0;
    }

    @Override
    protected void doMutation(int t, double[] Xprev, double[] X, int N, RandomGenerator g) {
        currentT = t;

        // Alias theta to the current X buffer. Scalars (lpost/llik/lprior)
        // are already propagated by gather since dim() returns d+3.
        aliasTheta(X, N);

        // Calibrate and apply MCMC moves.
        move.calibrate(uniformWeights(N), theta);
        Consumer<ThetaParticles> target = this::evaluateTarget;
        ThetaParticles moved = move.apply(theta, target, g);

        if (moved != theta) {
            if (moved.N == theta.N) {
                System.arraycopy(moved.arena, 0, theta.arena, 0, theta.arenaLength());
                theta.copySharedFrom(moved);
            } else {
                theta = moved;
            }
        }
    }

    @Override
    protected void evaluateTarget(ThetaParticles x) {
        int N = x.N;
        int d = x.dim;
        int lpriorOff = x.lpriorOff();
        int lpostOff = x.lpostOff();
        int llikOff = x.llikOff();
        double currentExp = getCurrentExponent(currentT - 1);
        ensureEvalScratch(d * N);
        // Fused lprior + loglik evaluation — one pass over the data.
        model.loglikAndLogprior(x.arena, 0, N, -1,
            x.arena, llikOff,
            x.arena, lpriorOff);
        for (int n = 0; n < N; n++) {
            x.arena[lpostOff + n] = x.arena[lpriorOff + n] + currentExp * x.arena[llikOff + n];
        }
    }
}
