package com.curioloop.yum4j.ssm.particle.sampler;

/**
 * Result of a tempering run (fixed schedule or adaptive). Defensive
 * copies of all backing arrays are captured at construction time; the
 * record is safely shareable.
 *
 * @param finalTheta       final θ-particle arena
 * @param dimTheta         parameter dimension
 * @param N                number of θ-particles
 * @param logLt            per-step log marginal likelihood series
 * @param ess              per-step ESS series
 * @param resampled        per-step resampling flags
 * @param exponents        exponent schedule visited during the run
 * @param pathSamplingLogZ path-sampling log-evidence estimate
 * @param T                number of tempering steps
 */
public record TemperingResult(
    double[] finalTheta,
    int dimTheta,
    int N,
    double[] logLt,
    double[] ess,
    boolean[] resampled,
    double[] exponents,
    double pathSamplingLogZ,
    int T
) {}
