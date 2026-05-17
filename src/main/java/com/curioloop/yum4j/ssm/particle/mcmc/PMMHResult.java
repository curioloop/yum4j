package com.curioloop.yum4j.ssm.particle.mcmc;

/**
 * Result of a PMMH (Particle Marginal Metropolis-Hastings) run.
 * Defensive copies of the chain arrays are captured at construction
 * time; the record is safely shareable.
 *
 * @param chain    parameter samples flattened column-major:
 *                 {@code chain[j * niter + i]} is the {@code j}-th
 *                 component of the {@code i}-th iterate
 * @param dim      parameter dimension
 * @param niter    number of MCMC iterations
 * @param logPost  log-posterior at each iteration (length {@code niter})
 * @param accRate  overall acceptance rate
 */
public record PMMHResult(
    double[] chain,
    int dim,
    int niter,
    double[] logPost,
    double accRate
) {}
