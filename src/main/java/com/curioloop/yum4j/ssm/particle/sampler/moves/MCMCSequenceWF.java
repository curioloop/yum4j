package com.curioloop.yum4j.ssm.particle.sampler.moves;
import com.curioloop.yum4j.ssm.particle.sampler.ThetaParticles;

import java.util.function.Consumer;
import java.util.random.RandomGenerator;

/**
 * Waste-free MCMC sequence: keeps all intermediate states.
 *
 * <p>Given N input particles and {@code lenChain = P}, this sequence applies
 * {@code P − 1} MCMC steps and returns a concatenated {@link ThetaParticles}
 * of size {@code N · P} containing all intermediate states (including the
 * initial input as the first N particles).
 *
 * <p>Per-step acceptance rates are recorded on the returned particles via
 * {@link ThetaParticles#appendAccRates(double[], int)}.
 *
 * <h3>Allocation profile</h3>
 * <p>The result buffer of size {@code N · P} and the intermediate working
 * buffer are pre-allocated on the first call and reused across invocations
 * when dimensions match.
 *
 * @see MCMCSequence
 * @see ArrayMCMC
 */
public final class MCMCSequenceWF implements MCMCSequence {

    private final ArrayMCMC mcmc;
    private final int lenChain;

    /** Pre-allocated scratch buffer for MCMC steps. */
    private double[] scratch;

    /** Pre-allocated result buffer of size N·P reused across apply calls. */
    private ThetaParticles result;

    /** Pre-allocated working ThetaParticles of size N reused across apply calls. */
    private ThetaParticles working;

    /** Reusable per-call acceptance-rate buffer, sized {@code >= lenChain - 1}. */
    private double[] accRatesScratch;

    /**
     * Constructs a waste-free MCMC sequence.
     *
     * @param mcmc     the underlying MCMC kernel
     * @param lenChain total chain length P (number of MCMC steps = P − 1)
     */
    public MCMCSequenceWF(ArrayMCMC mcmc, int lenChain) {
        if (lenChain < 1) {
            throw new IllegalArgumentException("lenChain must be >= 1, got " + lenChain);
        }
        this.mcmc = mcmc;
        this.lenChain = lenChain;
    }

    @Override
    public void calibrate(double[] W, ThetaParticles x) {
        mcmc.calibrate(W, x);
    }

    /**
     * Apply P−1 MCMC steps and return all intermediate states concatenated.
     *
     * <p>The returned ThetaParticles has size N·P. The first N particles are
     * the input, followed by the state after each successive MCMC step.
     * Acceptance rates are recorded via
     * {@link ThetaParticles#appendAccRates(double[], int)}.
     *
     * @param x      input particles (N particles)
     * @param target target density evaluator
     * @param g      random number generator
     * @return concatenated N·P particles with all intermediate states
     */
    @Override
    public ThetaParticles apply(ThetaParticles x, Consumer<ThetaParticles> target,
                                RandomGenerator g) {
        int N = x.N;
        int dim = x.dim;
        int nSteps = lenChain - 1;
        int totalN = N * lenChain;

        int scratchSize = N * dim;
        if (scratch == null || scratch.length < scratchSize) {
            scratch = new double[scratchSize];
        }
        if (accRatesScratch == null || accRatesScratch.length < Math.max(1, nSteps)) {
            accRatesScratch = new double[Math.max(1, nSteps)];
        }

        if (result == null || result.N != totalN || result.dim != dim) {
            result = ThetaParticles.allocate(dim, totalN);
        }
        if (working == null || working.N != N || working.dim != dim) {
            working = ThetaParticles.allocate(dim, N);
        }

        // Copy input x into slots [0, N) of the result and into working.
        copyIntoResult(result, x, 0);
        copyFromResult(working, result, 0);

        // Apply P-1 MCMC steps.
        for (int step = 1; step <= nSteps; step++) {
            double ar = mcmc.step(working, target, g, scratch);
            accRatesScratch[step - 1] = ar;
            copyIntoResult(result, working, step * N);
        }

        // Propagate shared state (strongly-typed) from input to result.
        result.copySharedFrom(x);
        result.appendAccRates(accRatesScratch, nSteps);

        return result;
    }

    /**
     * Copy particles from src into the result buffer at the given particle offset.
     * Handles the column-major layout correctly: for result with totalN particles,
     * particle n of src goes to position (offset + n) in result.
     */
    private static void copyIntoResult(ThetaParticles result, ThetaParticles src, int offset) {
        int srcN = src.N;
        int dim = src.dim;
        int totalN = result.N;

        // Copy theta: column-major (dim, totalN) layout
        // src.arena[j * srcN + n] → result.arena[j * totalN + offset + n]
        for (int j = 0; j < dim; j++) {
            System.arraycopy(src.arena, j * srcN, result.arena, j * totalN + offset, srcN);
        }

        // Copy scalar arrays
        int srcLpost = src.lpostOff();
        int srcLlik = src.llikOff();
        int srcLprior = src.lpriorOff();
        int dstLpost = result.lpostOff();
        int dstLlik = result.llikOff();
        int dstLprior = result.lpriorOff();
        System.arraycopy(src.arena, srcLpost, result.arena, dstLpost + offset, srcN);
        System.arraycopy(src.arena, srcLlik, result.arena, dstLlik + offset, srcN);
        System.arraycopy(src.arena, srcLprior, result.arena, dstLprior + offset, srcN);
    }

    /**
     * Copy particles from the result buffer at the given offset into dst.
     * Inverse of {@link #copyIntoResult}.
     */
    private static void copyFromResult(ThetaParticles dst, ThetaParticles result, int offset) {
        int dstN = dst.N;
        int dim = dst.dim;
        int totalN = result.N;

        for (int j = 0; j < dim; j++) {
            System.arraycopy(result.arena, j * totalN + offset, dst.arena, j * dstN, dstN);
        }
        int srcLpost = result.lpostOff();
        int srcLlik = result.llikOff();
        int srcLprior = result.lpriorOff();
        int dstLpost = dst.lpostOff();
        int dstLlik = dst.llikOff();
        int dstLprior = dst.lpriorOff();
        System.arraycopy(result.arena, srcLpost + offset, dst.arena, dstLpost, dstN);
        System.arraycopy(result.arena, srcLlik + offset, dst.arena, dstLlik, dstN);
        System.arraycopy(result.arena, srcLprior + offset, dst.arena, dstLprior, dstN);
    }
}
