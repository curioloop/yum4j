package com.curioloop.yum4j.ssm.particle.dist;

import java.util.random.RandomGenerator;

/**
 * Sealed interface for multivariate distributions over
 * column-major {@code (d, N)} buffers.
 *
 * <p>All implementations operate in place on {@code double[]} buffers of
 * length {@code d * N}, with element {@code X[j, n]} stored at
 * {@code X[j * N + n]}. The leading dimension is {@code N}; the
 * dimension axis has stride {@code N} between successive elements of a
 * column. Single-point evaluation is supported by passing {@code N = 1}.
 *
 * <p>Batch methods accept an optional caller-supplied {@code scratch}
 * buffer of size at least {@code dim() * N} doubles. When non-null,
 * implementations must not allocate any temporary arrays on the hot
 * path. When null, implementations may allocate internally; this is
 * the ergonomic default and is suitable for one-shot evaluations.
 *
 * <p>Implementations are limited to the four types in this subpackage;
 * custom multivariate distributions are out of scope for the current
 * particle-filter spec.
 */
public sealed interface MultivariateDistribution
    permits MvNormalDistribution, MvStudentDistribution,
            IndependentProductDistribution, MixtureDistribution {

    /** Number of dimensions. */
    int dim();

    /**
     * Copy the distribution's mean into {@code out[0..dim()-1]}.
     */
    void mean(double[] out);

    /**
     * Batch log-pdf of {@code N} column vectors packed as
     * {@code (d, N)} column-major. Writes one scalar per particle into
     * {@code out[outOff..outOff+N-1]}.
     *
     * @param x       input buffer of length at least {@code xOff + dim() * N}
     * @param xOff    offset of the first particle in {@code x}
     * @param N       number of particles
     * @param out     output buffer (length at least {@code outOff + N})
     * @param outOff  output offset
     * @param scratch optional caller scratch (length at least {@code dim() * N}); non-null
     *                enables zero-allocation on the hot path
     */
    void logPdfBatch(double[] x, int xOff, int N,
                     double[] out, int outOff,
                     double[] scratch);

    /**
     * Batch sample of {@code N} column vectors written as
     * {@code (d, N)} column-major into {@code x}.
     *
     * @param g       random generator
     * @param N       number of particles to draw
     * @param x       output buffer (length at least {@code xOff + dim() * N})
     * @param xOff    offset of the first particle in {@code x}
     * @param scratch optional caller scratch (length at least {@code dim() * N})
     */
    void sample(RandomGenerator g, int N,
                double[] x, int xOff,
                double[] scratch);
}
