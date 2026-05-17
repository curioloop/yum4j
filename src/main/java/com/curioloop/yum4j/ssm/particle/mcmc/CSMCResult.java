package com.curioloop.yum4j.ssm.particle.mcmc;

/**
 * Result of a Conditional SMC run. The trajectory array is a defensive
 * copy captured at construction time; the record is safely shareable.
 *
 * @param trajectory    sampled state trajectory in row-major
 *                      {@code (T, stateDim)} layout:
 *                      {@code trajectory[t * stateDim + j]} is the
 *                      {@code j}-th component of the state at step {@code t}
 * @param logLikelihood log marginal likelihood estimate from the
 *                      conditional run
 * @param T             time horizon
 * @param stateDim      state dimension
 */
public record CSMCResult(
    double[] trajectory,
    double logLikelihood,
    int T,
    int stateDim
) {}
