package com.curioloop.yum4j.ssm.particle.sampler;

/**
 * Result of an SMC² run. Defensive copies of all backing arrays are
 * captured at construction time; the record is safely shareable.
 *
 * @param finalTheta final θ-particle arena (flat copy of
 *                   {@link ThetaParticles#arena})
 * @param dimTheta   parameter dimension
 * @param N          number of θ-particles
 * @param logLt      per-step log marginal likelihood series
 * @param ess        per-step ESS series
 * @param resampled  per-step resampling flags
 * @param T          number of time steps processed
 */
public record SMC2Result(
    double[] finalTheta,
    int dimTheta,
    int N,
    double[] logLt,
    double[] ess,
    boolean[] resampled,
    int T
) {}
