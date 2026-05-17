package com.curioloop.yum4j.ssm.particle.sampler.moves;
import com.curioloop.yum4j.ssm.particle.sampler.ThetaParticles;

import java.util.function.Consumer;
import java.util.random.RandomGenerator;

/**
 * Interface for MCMC repetition strategies within SMC samplers.
 *
 * <p>A {@code MCMCSequence} wraps an {@link ArrayMCMC} kernel and defines
 * how many times it is applied and what is returned:
 * <ul>
 *   <li>{@link MCMCSequenceWF} — waste-free: keeps all intermediate states,
 *       returning N·P particles.</li>
 *   <li>{@link AdaptiveMCMCSequence} — standard/adaptive: keeps only the
 *       final state, returning N particles.</li>
 * </ul>
 *
 * <p>Usage within an SMC sampler:
 * <ol>
 *   <li>{@link #calibrate(double[], ThetaParticles)} is called before
 *       resampling to tune the underlying MCMC kernel.</li>
 *   <li>{@link #apply(ThetaParticles, Consumer, RandomGenerator)} is called
 *       after resampling to move particles.</li>
 * </ol>
 *
 * @see ArrayMCMC
 * @see MCMCSequenceWF
 * @see AdaptiveMCMCSequence
 */
public interface MCMCSequence {

    /**
     * Calibrate the underlying MCMC kernel to the current weighted sample.
     *
     * @param W normalised weights, length N
     * @param x current theta-particles
     */
    void calibrate(double[] W, ThetaParticles x);

    /**
     * Apply the MCMC sequence to the given particles.
     *
     * <p>The returned {@link ThetaParticles} may be larger than the input
     * (waste-free) or the same size (standard/adaptive). Acceptance rates
     * are recorded in {@code result.shared["acc_rates"]} as a list of lists.
     *
     * @param x      input theta-particles (N particles)
     * @param target consumer that evaluates the target density, filling
     *               {@code lpost} (and optionally {@code llik}, {@code lprior})
     * @param g      random number generator
     * @return the moved particles (N·P for waste-free, N for standard)
     */
    ThetaParticles apply(ThetaParticles x, Consumer<ThetaParticles> target,
                         RandomGenerator g);
}
