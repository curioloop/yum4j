package com.curioloop.yum4j.ssm.particle.sampler;

/**
 * Result of an IBIS run. Defensive copies of all backing arrays are
 * captured at construction time; the record is safely shareable.
 *
 * @param finalTheta final θ-particle arena (length {@code (dimTheta + 3) * N},
 *                   layout {@code [theta : lpost : llik : lprior]}; see
 *                   {@link ThetaParticles})
 * @param dimTheta   parameter dimension
 * @param N          number of θ-particles
 * @param logLt      per-step log marginal likelihood series (length {@code T})
 * @param ess        per-step ESS series (length {@code T})
 * @param resampled  per-step resampling flags (length {@code T})
 * @param T          number of time steps processed
 */
public record IBISResult(
    double[] finalTheta,
    int dimTheta,
    int N,
    double[] logLt,
    double[] ess,
    boolean[] resampled,
    int T
) {}
