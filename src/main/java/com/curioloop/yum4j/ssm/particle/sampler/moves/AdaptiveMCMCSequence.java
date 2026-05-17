package com.curioloop.yum4j.ssm.particle.sampler.moves;
import com.curioloop.yum4j.ssm.particle.sampler.ThetaParticles;

import java.util.function.Consumer;
import java.util.random.RandomGenerator;

/**
 * Standard/adaptive MCMC sequence: keeps only the final state.
 *
 * <p>Given N input particles and {@code lenChain = P}, this sequence applies
 * up to {@code P − 1} MCMC steps and returns only the final
 * {@link ThetaParticles} of size N.
 *
 * <p>When {@code adaptive = true}, the sequence stops early when the mean
 * displacement between consecutive iterations changes by less than
 * {@code deltaDist} fraction of the previous displacement:
 * {@code |dist − prevDist| < deltaDist × prevDist}.
 *
 * <p>Per-step acceptance rates are recorded on the input particles via
 * {@link ThetaParticles#appendAccRates(double[], int)}.
 *
 * <h3>Zero-copy fast path</h3>
 * <p>When {@code adaptive = false} (the default for standard SMC samplers),
 * MCMC steps run directly on the input buffer — no intermediate {@code xout}
 * copy is needed. This pairs with {@link FKSMCSampler}'s arena aliasing so
 * that the MCMC kernel mutates {@code Workspace.X} in place.
 *
 * <p>When {@code adaptive = true}, displacement from the initial state must
 * be measured so an internal {@code xout} buffer is copied once per call and
 * grown lazily.
 *
 * @see MCMCSequence
 * @see ArrayMCMC
 */
public final class AdaptiveMCMCSequence implements MCMCSequence {

    private final ArrayMCMC mcmc;
    private final int lenChain;
    private final boolean adaptive;
    private final double deltaDist;

    /** Pre-allocated scratch buffer for MCMC steps. */
    private double[] scratch;

    /** Pre-allocated working ThetaParticles reused across apply calls (adaptive path only). */
    private ThetaParticles xout;

    /** Reusable per-iteration acceptance-rate buffer, sized {@code >= lenChain - 1}. */
    private double[] accRatesScratch;

    /**
     * Constructs an adaptive MCMC sequence.
     *
     * @param mcmc      the underlying MCMC kernel
     * @param lenChain  total chain length P (max number of MCMC steps = P − 1)
     * @param adaptive  whether to enable early stopping based on displacement
     * @param deltaDist relative displacement threshold for early stopping
     */
    public AdaptiveMCMCSequence(ArrayMCMC mcmc, int lenChain, boolean adaptive, double deltaDist) {
        if (lenChain < 1) {
            throw new IllegalArgumentException("lenChain must be >= 1, got " + lenChain);
        }
        this.mcmc = mcmc;
        this.lenChain = lenChain;
        this.adaptive = adaptive;
        this.deltaDist = deltaDist;
    }

    /**
     * Constructs a non-adaptive MCMC sequence (standard SMC).
     *
     * @param mcmc     the underlying MCMC kernel
     * @param lenChain total chain length P (number of MCMC steps = P − 1)
     */
    public AdaptiveMCMCSequence(ArrayMCMC mcmc, int lenChain) {
        this(mcmc, lenChain, false, 0.1);
    }

    @Override
    public void calibrate(double[] W, ThetaParticles x) {
        mcmc.calibrate(W, x);
    }

    /**
     * Apply up to P−1 MCMC steps and return the final state.
     *
     * <p>In non-adaptive mode, MCMC steps are run directly on the input
     * buffer and the same {@code x} is returned (zero copy).
     *
     * <p>In adaptive mode, the input is copied into a reusable working
     * buffer first so mean-displacement can be measured against the
     * initial state; MCMC steps are run on the working buffer and the
     * working buffer is returned.
     *
     * <p>Acceptance rates are recorded onto the returned particles via
     * {@link ThetaParticles#appendAccRates(double[], int)}.
     *
     * @param x      input particles (N particles)
     * @param target target density evaluator
     * @param g      random number generator
     * @return final N particles after MCMC moves
     */
    @Override
    public ThetaParticles apply(ThetaParticles x, Consumer<ThetaParticles> target,
                                RandomGenerator g) {
        int N = x.N;
        int dim = x.dim;
        int nSteps = lenChain - 1;

        int scratchSize = N * dim;
        if (scratch == null || scratch.length < scratchSize) {
            scratch = new double[scratchSize];
        }
        if (accRatesScratch == null || accRatesScratch.length < Math.max(1, nSteps)) {
            accRatesScratch = new double[Math.max(1, nSteps)];
        }

        ThetaParticles workBuf;
        if (adaptive) {
            // Adaptive mode needs the initial state preserved for
            // displacement measurement — copy into xout once.
            if (xout == null || xout.N != N || xout.dim != dim) {
                xout = ThetaParticles.allocate(dim, N);
            }
            System.arraycopy(x.arena, 0, xout.arena, 0, x.arenaLength());
            xout.copySharedFrom(x);
            workBuf = xout;
        } else {
            // Non-adaptive: operate directly on the input buffer.
            workBuf = x;
        }

        int written = 0;
        double dist = 0.0;

        for (int step = 0; step < nSteps; step++) {
            double ar = mcmc.step(workBuf, target, g, scratch);
            accRatesScratch[written++] = ar;

            if (adaptive) {
                double prevDist = dist;
                dist = meanDisplacement(workBuf.arena, x.arena, N, dim);
                if (Math.abs(dist - prevDist) < deltaDist * prevDist) {
                    break;
                }
            }
        }

        workBuf.appendAccRates(accRatesScratch, written);
        return workBuf;
    }

    /**
     * Compute mean Euclidean displacement between two theta buffers.
     *
     * <p>For each particle n, computes ‖a[:,n] − b[:,n]‖₂ (Euclidean norm
     * across dimensions), then returns the mean over all N particles.
     * Both a and b are arena buffers with theta at offset 0.
     */
    private static double meanDisplacement(double[] a, double[] b, int N, int dim) {
        double sum = 0.0;
        for (int n = 0; n < N; n++) {
            double normSq = 0.0;
            for (int j = 0; j < dim; j++) {
                double diff = a[j * N + n] - b[j * N + n];
                normSq += diff * diff;
            }
            sum += Math.sqrt(normSq);
        }
        return sum / N;
    }
}
