package com.curioloop.yum4j.ssm.particle.kernel;

import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

/**
 * Scalar fallback {@link RandomBatch} implementation used when the
 * hardware does not support SIMD with {@code LANES > 1}.
 *
 * <p>Delegates all generation to the underlying {@code L64X128MixRandom}
 * generator. Gaussian draws use the standard scalar Box–Muller transform
 * with the same zero-patch semantics as the SIMD path.
 *
 * <p>This class is <b>not thread-safe</b>.
 */
final class ScalarRandomBatch implements RandomBatch {

    private static final String ALGORITHM = "L64X128MixRandom";
    private static final double TWO_PI = 2.0 * Math.PI;

    private final long masterSeed;
    private final RandomGenerator rng;

    // Box–Muller spare cache
    private boolean hasSpare;
    private double spare;

    ScalarRandomBatch(long seed) {
        this.masterSeed = seed;
        this.rng = RandomGeneratorFactory.of(ALGORITHM).create(seed);
    }

    @Override
    public void nextUniforms(double[] out, int off, int N) {
        for (int i = 0; i < N; i++) {
            out[off + i] = rng.nextDouble();
        }
    }

    @Override
    public void nextGaussians(double[] out, int off, int N) {
        for (int i = 0; i < N; i++) {
            out[off + i] = nextGaussianScalar();
        }
    }

    @Override
    public void nextGaussiansAffineInto(double[] out, int off, int N,
                                        double mu, double sigma) {
        for (int i = 0; i < N; i++) {
            out[off + i] = Math.fma(nextGaussianScalar(), sigma, mu);
        }
    }

    @Override
    public void nextGaussiansAffineInto(double[] out, int off, int N,
                                        double[] mu, int muOff, double sigma) {
        for (int i = 0; i < N; i++) {
            out[off + i] = Math.fma(nextGaussianScalar(), sigma, mu[muOff + i]);
        }
    }

    @Override
    public void addAffineGaussianInto(double[] out, int off, int N, double sigma) {
        for (int i = 0; i < N; i++) {
            out[off + i] = Math.fma(nextGaussianScalar(), sigma, out[off + i]);
        }
    }

    @Override
    public void nextExponentials(double[] out, int off, int N, double lambda) {
        final double invLambda = 1.0 / lambda;
        for (int i = 0; i < N; i++) {
            double u = rng.nextDouble();
            while (u == 0.0) {
                u = rng.nextDouble();
            }
            out[off + i] = -Math.log(u) * invLambda;
        }
    }

    @Override
    public RandomGenerator asRandomGenerator() {
        return rng;
    }

    @Override
    public RandomBatch split(long key) {
        long subSeed = DefaultRandomBatch.mixSeed(masterSeed, key);
        return new ScalarRandomBatch(subSeed);
    }

    // ------------------------------------------------------------------
    // Scalar Box–Muller
    // ------------------------------------------------------------------

    private double nextGaussianScalar() {
        if (hasSpare) {
            hasSpare = false;
            return spare;
        }
        double u1 = rng.nextDouble();
        while (u1 == 0.0) {
            u1 = rng.nextDouble();
        }
        double u2 = rng.nextDouble();

        double r = Math.sqrt(-2.0 * Math.log(u1));
        double angle = TWO_PI * u2;
        spare = r * Math.sin(angle);
        hasSpare = true;
        return r * Math.cos(angle);
    }
}
