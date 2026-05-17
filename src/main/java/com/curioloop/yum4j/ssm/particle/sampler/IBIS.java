package com.curioloop.yum4j.ssm.particle.sampler;
import com.curioloop.yum4j.ssm.particle.sampler.moves.MCMCSequence;

import com.curioloop.yum4j.ssm.particle.engine.StepContext;

import java.util.Arrays;
import java.util.function.Consumer;
import java.util.random.RandomGenerator;

/**
 * IBIS (Iterated Batch Importance Sampling) SMC sampler.
 *
 * <p>At each time step t, the target is the posterior of θ given data
 * Y_{0:t}. The log-weight increment is {@code logG(t) = model.logpyt(θ, t)},
 * which is accumulated into {@code lpost}. When resampling fires, particles
 * are moved using the MCMC sequence targeting the posterior given Y_{0:t-1}.
 *
 * <p>Operates directly on the engine's {@code Workspace.X} buffer via
 * {@link FKSMCSampler}'s arena aliasing — there is no side-car copy.
 *
 * @see FKSMCSampler
 * @see StaticModel
 */
public final class IBIS extends FKSMCSampler {

    public IBIS(StaticModel model, MCMCSequence move,
                int N, int lenChain, double essRmin) {
        super(model, move, N, lenChain, essRmin);
    }

    public IBIS(StaticModel model, MCMCSequence move, int N, int lenChain) {
        this(model, move, N, lenChain, 0.5);
    }

    @Override
    protected void initializeParticles(int N) {
        super.initializeParticles(N);
        // For IBIS, llik starts at 0 (no data incorporated yet)
        int llikOff = theta.llikOff();
        Arrays.fill(theta.arena, llikOff, llikOff + N, 0.0);
    }

    @Override
    protected void doLogG(int t, double[] Xprev, double[] X, int N,
                          double[] logW, int lwOff) {
        // Alias theta directly to the engine's X buffer.
        aliasTheta(X, N);

        // logG = model.logpyt(theta, t)
        model.logpyt(theta.arena, 0, N, t, logW, lwOff);

        // Accumulate into llik and lpost
        int llikOff = theta.llikOff();
        int lpostOff = theta.lpostOff();
        for (int n = 0; n < N; n++) {
            double v = logW[lwOff + n];
            if (Double.isNaN(v)) v = Double.NEGATIVE_INFINITY;
            theta.arena[llikOff + n] += v;
            theta.arena[lpostOff + n] += v;
        }

        rsFlag = true; // IBIS always moves on resample; engine decides when to resample
    }

    @Override
    protected void doMutation(int t, double[] Xprev, double[] X, int N, RandomGenerator g) {
        currentT = t;

        // Alias theta to the current X buffer. The engine's gather
        // already propagated scalars (lpost, llik, lprior) alongside theta
        // because dim() returns d+3 — no recomputation needed.
        aliasTheta(X, N);

        // Apply MCMC moves targeting posterior given Y_{0:t-1}.
        move.calibrate(uniformWeights(N), theta);
        Consumer<ThetaParticles> target = this::evaluateTarget;
        ThetaParticles moved = move.apply(theta, target, g);

        if (moved != theta) {
            if (moved.N == theta.N) {
                // Standard (non-waste-free): MCMC wrote into its own
                // buffer — mirror back into the aliased arena.
                System.arraycopy(moved.arena, 0, theta.arena, 0, theta.arenaLength());
                theta.copySharedFrom(moved);
            } else {
                // Waste-free expanded (N·P) — swap container.
                theta = moved;
            }
        }
    }

    @Override
    protected void evaluateTarget(ThetaParticles x) {
        int N = x.N;
        int d = x.dim;
        int lpostOff = x.lpostOff();
        int llikOff = x.llikOff();
        int lpriorOff = x.lpriorOff();
        // Target is posterior given Y_{0:t-1} where t = currentT
        int targetT = currentT - 1;
        ensureEvalScratch(d * N);
        if (targetT < 0) {
            // Only prior
            model.prior().logPdfBatch(x.arena, 0, N, x.arena, lpostOff, evalScratch);
            System.arraycopy(x.arena, lpostOff, x.arena, lpriorOff, N);
            Arrays.fill(x.arena, llikOff, llikOff + N, 0.0);
        } else {
            // Fused loglik + logprior in a single pass over the data,
            // then lpost = llik + lprior. Avoids the redundant logpyt
            // loop that separate loglik + logpost would perform.
            model.loglikAndLogprior(x.arena, 0, N, targetT,
                x.arena, llikOff,
                x.arena, lpriorOff);
            for (int n = 0; n < N; n++) {
                x.arena[lpostOff + n] = x.arena[lpriorOff + n] + x.arena[llikOff + n];
            }
        }
    }
}
