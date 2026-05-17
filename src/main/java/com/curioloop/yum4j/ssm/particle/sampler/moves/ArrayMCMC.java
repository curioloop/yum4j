package com.curioloop.yum4j.ssm.particle.sampler.moves;
import com.curioloop.yum4j.ssm.particle.sampler.ThetaParticles;

import java.util.function.Consumer;
import java.util.random.RandomGenerator;

/**
 * Interface for array-level MCMC kernels applied to all N particles
 * simultaneously.
 *
 * <p>Implementations (e.g. {@code ArrayRandomWalk}, {@code ArrayIndepMetropolis})
 * operate on the entire {@link ThetaParticles} buffer in a single call,
 * proposing and accepting/rejecting independently for each particle.
 *
 * <p>The typical usage within an SMC sampler is:
 * <ol>
 *   <li>{@link #calibrate(double[], ThetaParticles)} — tune the kernel to the
 *       current weighted sample (called before resampling).</li>
 *   <li>{@link #step(ThetaParticles, Consumer, RandomGenerator, double[])} —
 *       apply one MCMC step in-place (called after resampling).</li>
 * </ol>
 *
 * @see MCMCSequence
 */
public interface ArrayMCMC {

    /**
     * Tune the kernel to the current weighted sample.
     *
     * <p>Typically computes the weighted covariance and its Cholesky factor
     * for use in subsequent {@link #step} calls.
     *
     * @param W normalised weights, length N (summing to 1)
     * @param x current theta-particles
     */
    void calibrate(double[] W, ThetaParticles x);

    /**
     * Apply one MCMC step in-place to all particles.
     *
     * <p>For each particle n, proposes a new value, evaluates the target
     * density via the {@code target} consumer, and accepts/rejects
     * independently via the Metropolis-Hastings ratio.
     *
     * @param x       current theta-particles (modified in-place on acceptance)
     * @param target  consumer that fills {@code x.lpost} (and optionally
     *                {@code x.llik}, {@code x.lprior}) for the given particles
     * @param g       random number generator
     * @param scratch pre-allocated scratch buffer (length ≥ N × dim)
     * @return mean acceptance probability across all N particles
     */
    double step(ThetaParticles x, Consumer<ThetaParticles> target,
                RandomGenerator g, double[] scratch);
}
