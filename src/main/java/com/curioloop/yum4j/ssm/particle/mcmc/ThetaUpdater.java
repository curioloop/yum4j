package com.curioloop.yum4j.ssm.particle.mcmc;

import com.curioloop.yum4j.ssm.particle.Particle;
import com.curioloop.yum4j.ssm.particle.SamplingMethod;

/**
 * Functional interface for the theta update step in Particle Gibbs.
 *
 * <p>The updater is invoked once per iteration after CSMC has produced
 * a fresh state trajectory; it should mutate {@code thetaCurr} in place
 * to the next theta sample and may record diagnostics on
 * {@code chain}.
 *
 * @see Particle#sample(SamplingMethod) Particle.sample(SamplingMethod.PARTICLE_GIBBS)
 */
@FunctionalInterface
public interface ThetaUpdater {

    /**
     * Update {@code thetaCurr} in place given the current state
     * trajectory.
     *
     * @param chain      the chain being built (callee may write
     *                   diagnostics into it)
     * @param iter       current iteration index in {@code [0, niter)}
     * @param trajectory current state trajectory, row-major
     *                   {@code (T, stateDim)}
     * @param thetaCurr  current theta vector (length
     *                   {@code chain.dim}); modified in place
     */
    void update(MCMCChain chain, int iter, double[] trajectory, double[] thetaCurr);
}
