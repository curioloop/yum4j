package com.curioloop.yum4j.ssm.particle.kernel;

import java.util.random.RandomGenerator;

/**
 * Batched random number generator for the particle-filter engine.
 *
 * <p>Fills caller-owned {@code double[]} buffers with uniform, Gaussian
 * (via SIMD Box–Muller), and exponential draws. Designed to replace
 * per-particle {@code Random.nextGaussian()} calls with vectorised
 * batch generation.
 *
 * <p>Implementations are <b>not thread-safe</b>. Use {@link #split(long)}
 * to derive per-slab generators for parallel execution.
 *
 * <p>All hot-path methods are allocation-free (R4.7).
 */
public interface RandomBatch {

    /**
     * Writes {@code N} uniform {@code [0, 1)} draws into
     * {@code out[off .. off+N)}.
     */
    void nextUniforms(double[] out, int off, int N);

    /**
     * Writes {@code N} standard normal draws into
     * {@code out[off .. off+N)} using SIMD Box–Muller.
     */
    void nextGaussians(double[] out, int off, int N);

    /**
     * Writes {@code sigma * g + mu} for {@code N} standard normal draws
     * {@code g} into {@code out[off .. off+N)} in a single SIMD pass
     * that fuses the Box–Muller final multiplication with the affine
     * FMA (R6.1).
     *
     * <p>The default implementation falls back to a two-pass form
     * ({@link #nextGaussians} followed by a scalar affine pass); SIMD
     * implementations should override with a fused kernel.
     */
    default void nextGaussiansAffineInto(double[] out, int off, int N,
                                         double mu, double sigma) {
        nextGaussians(out, off, N);
        for (int i = 0; i < N; i++) {
            out[off + i] = out[off + i] * sigma + mu;
        }
    }

    /**
     * Variant with per-particle mean: writes {@code sigma * g + mu[muOff+i]}
     * for {@code N} standard normal draws {@code g} into
     * {@code out[off .. off+N)} in a single fused pass (R6.5).
     *
     * <p>The default implementation is the two-pass fallback; SIMD
     * implementations override for performance.
     */
    default void nextGaussiansAffineInto(double[] out, int off, int N,
                                         double[] mu, int muOff, double sigma) {
        nextGaussians(out, off, N);
        for (int i = 0; i < N; i++) {
            out[off + i] = out[off + i] * sigma + mu[muOff + i];
        }
    }

    /**
     * Adds {@code sigma * g} to each element of {@code out[off .. off+N)}
     * in a single read-modify-write pass, where {@code g} are standard
     * normal draws (R6.4).
     *
     * <p>This is the "Phase 2" of the single-write AR(1) advance used by
     * {@link ParticleOps#arOneInto}: the caller first fills {@code out}
     * with {@code c + rho * Xprev[i]}, then calls this method to add the
     * Gaussian innovation in place. SIMD implementations override with a
     * Box–Muller kernel that FMA-adds directly into {@code out} with no
     * intermediate scratch.
     *
     * <p>The default implementation is a simple scalar loop using a
     * temporary scratch array of size {@code N}.
     */
    default void addAffineGaussianInto(double[] out, int off, int N, double sigma) {
        double[] scratch = new double[N];
        nextGaussians(scratch, 0, N);
        for (int i = 0; i < N; i++) {
            out[off + i] = out[off + i] + sigma * scratch[i];
        }
    }

    /**
     * Writes {@code N} exponential draws with rate {@code lambda} into
     * {@code out[off .. off+N)}.
     *
     * <p>Each draw is {@code -ln(U) / lambda} where {@code U ~ U(0,1)}.
     */
    void nextExponentials(double[] out, int off, int N, double lambda);

    /**
     * Returns a {@link RandomGenerator} view for escape-hatch scalar use.
     * The returned generator shares state with this batch — interleaving
     * calls will advance the same underlying stream.
     */
    RandomGenerator asRandomGenerator();

    /**
     * Returns a fresh sub-generator seeded deterministically from
     * {@code (masterSeed, key)}. Used by STRIPED parallel slabs to
     * derive per-slab RNGs that produce bitwise-identical results
     * regardless of thread scheduling.
     */
    RandomBatch split(long key);

    /**
     * Creates a {@code RandomBatch} backed by {@code L64X128MixRandom}.
     * Automatically selects the SIMD implementation when the hardware
     * supports {@code SPECIES_256} (4 double lanes); falls back to
     * scalar otherwise.
     */
    static RandomBatch of(long seed) {
        if (DefaultRandomBatch.SIMD_SUPPORTED) {
            return new DefaultRandomBatch(seed);
        }
        return new ScalarRandomBatch(seed);
    }
}
