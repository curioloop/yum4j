package com.curioloop.yum4j.ssm.particle.sampler.nested;

import java.util.List;

/**
 * Result container for nested sampling algorithms.
 *
 * @param logEvidence     final log-evidence estimate (log Z)
 * @param logEvidencePath log Z estimate at each iteration
 * @param logWeights      log weight of each dead point
 * @param deadPoints      the dead points (theta values, each of length dim)
 * @param iterations      total number of iterations
 */
public record NestedSamplingResult(
    double logEvidence,
    List<Double> logEvidencePath,
    List<Double> logWeights,
    List<double[]> deadPoints,
    int iterations
) {}
