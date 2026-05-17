package com.curioloop.yum4j.ssm.particle.kernel;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * SIMD-vectorised kernel primitives for the particle-filter engine.
 *
 * <p>All methods write into caller-owned buffers using
 * {@link DoubleVector#SPECIES_256} for vectorised loops with a scalar
 * tail for the remainder. Zero allocation on the hot path (R11).
 *
 * <p>These primitives replace per-particle scalar loops in model
 * implementations. The critical path is {@link #arOneInto} which fuses
 * the AR(1) transition into a single SIMD kernel:
 * {@code out = c + rho*Xprev + sigma*N(0,1)}.
 *
 * <p><b>Thread safety:</b> All methods are stateless and thread-safe.
 * The {@link RandomBatch} parameter is the only mutable dependency.
 */
public final class ParticleOps {

    private ParticleOps() {}

    // ------------------------------------------------------------------
    // SIMD configuration — capped at 256 bits to avoid AVX-512
    // frequency throttling.
    // ------------------------------------------------------------------

    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_256;
    private static final int LANES = SPECIES.length(); // 4

    // ------------------------------------------------------------------
    // Gaussian draws
    // ------------------------------------------------------------------

    /**
     * Writes {@code N} independent {@code Normal(mu, sigma)} draws into
     * {@code out[off .. off+N)}.
     *
     * <p>Implementation: delegates to
     * {@link RandomBatch#nextGaussiansAffineInto(double[], int, int, double, double)},
     * which fuses the Box–Muller tail with the scale-and-shift FMA into
     * a single SIMD pass over {@code out} (R6.3).
     *
     * @param out   output buffer
     * @param off   offset into {@code out}
     * @param N     number of draws
     * @param mu    scalar mean
     * @param sigma standard deviation (must be &gt; 0)
     * @param rng   batched RNG
     */
    public static void gaussianInto(double[] out, int off, int N,
                                    double mu, double sigma, RandomBatch rng) {
        rng.nextGaussiansAffineInto(out, off, N, mu, sigma);
    }

    /**
     * Writes {@code N} independent {@code Normal(mu[n], sigma)} draws into
     * {@code out[off .. off+N)} with per-particle mean from
     * {@code mu[muOff .. muOff+N)}.
     *
     * <p>Implementation: delegates to
     * {@link RandomBatch#nextGaussiansAffineInto(double[], int, int, double[], int, double)}
     * for a single fused SIMD pass (R6.5).
     *
     * @param out   output buffer
     * @param off   offset into {@code out}
     * @param N     number of draws
     * @param mu    per-particle mean buffer
     * @param muOff offset into {@code mu}
     * @param sigma standard deviation (must be &gt; 0)
     * @param rng   batched RNG
     */
    public static void gaussianInto(double[] out, int off, int N,
                                    double[] mu, int muOff, double sigma,
                                    RandomBatch rng) {
        rng.nextGaussiansAffineInto(out, off, N, mu, muOff, sigma);
    }

    // ------------------------------------------------------------------
    // AR(1) one-step — critical path
    // ------------------------------------------------------------------

    /**
     * Fused AR(1) transition kernel: writes
     * {@code out[off+n] = c + rho * Xprev[xpOff+n] + sigma * N(0,1)}
     * for {@code n ∈ [0, N)}.
     *
     * <p>This is the critical path for Gaussian SSMs. The implementation
     * is a single-write pass over {@code out} (R6.4):
     * <ol>
     *   <li><b>Phase 1 (SIMD):</b> write {@code out[i] = c + rho * Xprev[i]}
     *       via FMA. {@code Xprev} is read once; {@code out} is written once.</li>
     *   <li><b>Phase 2 (SIMD):</b> read {@code out[i]} as the base,
     *       write {@code out[i] = base + sigma * g[i]} via the fused
     *       Box–Muller + FMA kernel
     *       {@link RandomBatch#addAffineGaussianInto}.</li>
     * </ol>
     * Total memory traffic through the particle buffer: one {@code Xprev}
     * read pass, one {@code out} write pass (Phase 1), one {@code out}
     * read-modify-write pass (Phase 2). The previous implementation
     * required an additional {@code out} read-modify-write.
     *
     * @param out   output buffer
     * @param off   offset into {@code out}
     * @param N     number of particles
     * @param c     intercept constant
     * @param rho   autoregressive coefficient
     * @param Xprev previous state buffer
     * @param xpOff offset into {@code Xprev}
     * @param sigma innovation standard deviation (must be &gt; 0)
     * @param rng   batched RNG
     */
    public static void arOneInto(double[] out, int off, int N,
                                 double c, double rho,
                                 double[] Xprev, int xpOff,
                                 double sigma, RandomBatch rng) {
        // Phase 1: out[i] = c + rho * Xprev[i]  via SIMD FMA
        final DoubleVector rhoVec = DoubleVector.broadcast(SPECIES, rho);
        final DoubleVector cVec = DoubleVector.broadcast(SPECIES, c);

        int i = 0;
        final int bound = SPECIES.loopBound(N);
        for (; i < bound; i += LANES) {
            DoubleVector xp = DoubleVector.fromArray(SPECIES, Xprev, xpOff + i);
            xp.fma(rhoVec, cVec).intoArray(out, off + i);
        }
        // Scalar tail for phase 1
        for (; i < N; i++) {
            out[off + i] = Math.fma(rho, Xprev[xpOff + i], c);
        }

        // Phase 2: out[i] = out[i] + sigma * g[i]  via fused Box–Muller + FMA
        rng.addAffineGaussianInto(out, off, N, sigma);
    }

    // ------------------------------------------------------------------
    // Gaussian log-pdf
    // ------------------------------------------------------------------

    /** Precomputed constant: -0.5 * log(2π) */
    private static final double HALF_LOG_2PI = 0.5 * Math.log(2.0 * Math.PI);

    /**
     * Writes the Gaussian log-pdf with scalar mean into
     * {@code out[off .. off+N)}:
     * <pre>
     *   out[off+n] = -0.5 * ((x[xOff+n] - mu) / sigma)² + logNorm
     * </pre>
     * where {@code logNorm = -log(sigma) - 0.5*log(2π)} is hoisted out.
     *
     * @param out   output buffer for log-pdf values
     * @param off   offset into {@code out}
     * @param N     number of evaluations
     * @param x     input values
     * @param xOff  offset into {@code x}
     * @param mu    scalar mean
     * @param sigma standard deviation (must be &gt; 0)
     */
    public static void gaussianLogPdfInto(double[] out, int off, int N,
                                          double[] x, int xOff,
                                          double mu, double sigma) {
        final double invSigma = 1.0 / sigma;
        final double logNorm = -Math.log(sigma) - HALF_LOG_2PI;

        final DoubleVector muVec = DoubleVector.broadcast(SPECIES, mu);
        final DoubleVector invSigmaVec = DoubleVector.broadcast(SPECIES, invSigma);
        final DoubleVector logNormVec = DoubleVector.broadcast(SPECIES, logNorm);
        final DoubleVector negHalf = DoubleVector.broadcast(SPECIES, -0.5);

        int i = 0;
        final int bound = SPECIES.loopBound(N);
        for (; i < bound; i += LANES) {
            DoubleVector xv = DoubleVector.fromArray(SPECIES, x, xOff + i);
            // z = (x - mu) / sigma
            DoubleVector z = xv.sub(muVec).mul(invSigmaVec);
            // result = -0.5 * z² + logNorm = z² * (-0.5) + logNorm
            z.fma(z.mul(negHalf), logNormVec).intoArray(out, off + i);
        }
        // Scalar tail
        for (; i < N; i++) {
            double z = (x[xOff + i] - mu) * invSigma;
            out[off + i] = -0.5 * z * z + logNorm;
        }
    }

    /**
     * Writes the Gaussian log-pdf with per-particle mean into
     * {@code out[off .. off+N)}:
     * <pre>
     *   out[off+n] = -0.5 * ((x[xOff+n] - mu[muOff+n]) / sigma)² + logNorm
     * </pre>
     * where {@code logNorm = -log(sigma) - 0.5*log(2π)} is hoisted out.
     *
     * @param out   output buffer for log-pdf values
     * @param off   offset into {@code out}
     * @param N     number of evaluations
     * @param x     input values
     * @param xOff  offset into {@code x}
     * @param mu    per-particle mean buffer
     * @param muOff offset into {@code mu}
     * @param sigma standard deviation (must be &gt; 0)
     */
    public static void gaussianLogPdfInto(double[] out, int off, int N,
                                          double[] x, int xOff,
                                          double[] mu, int muOff,
                                          double sigma) {
        final double invSigma = 1.0 / sigma;
        final double logNorm = -Math.log(sigma) - HALF_LOG_2PI;

        final DoubleVector invSigmaVec = DoubleVector.broadcast(SPECIES, invSigma);
        final DoubleVector logNormVec = DoubleVector.broadcast(SPECIES, logNorm);
        final DoubleVector negHalf = DoubleVector.broadcast(SPECIES, -0.5);

        int i = 0;
        final int bound = SPECIES.loopBound(N);
        for (; i < bound; i += LANES) {
            DoubleVector xv = DoubleVector.fromArray(SPECIES, x, xOff + i);
            DoubleVector mv = DoubleVector.fromArray(SPECIES, mu, muOff + i);
            // z = (x - mu) / sigma
            DoubleVector z = xv.sub(mv).mul(invSigmaVec);
            // result = -0.5 * z² + logNorm
            z.fma(z.mul(negHalf), logNormVec).intoArray(out, off + i);
        }
        // Scalar tail
        for (; i < N; i++) {
            double z = (x[xOff + i] - mu[muOff + i]) * invSigma;
            out[off + i] = -0.5 * z * z + logNorm;
        }
    }

    // ------------------------------------------------------------------
    // Elementwise transcendentals
    // ------------------------------------------------------------------

    /**
     * Elementwise exponential: {@code out[off+n] = exp(x[xOff+n])}
     * for {@code n ∈ [0, N)}.
     *
     * @param out  output buffer
     * @param off  offset into {@code out}
     * @param N    number of elements
     * @param x    input buffer
     * @param xOff offset into {@code x}
     */
    public static void expInto(double[] out, int off, int N,
                               double[] x, int xOff) {
        int i = 0;
        final int bound = SPECIES.loopBound(N);
        for (; i < bound; i += LANES) {
            DoubleVector v = DoubleVector.fromArray(SPECIES, x, xOff + i);
            v.lanewise(VectorOperators.EXP).intoArray(out, off + i);
        }
        // Scalar tail
        for (; i < N; i++) {
            out[off + i] = Math.exp(x[xOff + i]);
        }
    }

    /**
     * Elementwise natural logarithm: {@code out[off+n] = log(x[xOff+n])}
     * for {@code n ∈ [0, N)}.
     *
     * @param out  output buffer
     * @param off  offset into {@code out}
     * @param N    number of elements
     * @param x    input buffer
     * @param xOff offset into {@code x}
     */
    public static void logInto(double[] out, int off, int N,
                               double[] x, int xOff) {
        int i = 0;
        final int bound = SPECIES.loopBound(N);
        for (; i < bound; i += LANES) {
            DoubleVector v = DoubleVector.fromArray(SPECIES, x, xOff + i);
            v.lanewise(VectorOperators.LOG).intoArray(out, off + i);
        }
        // Scalar tail
        for (; i < N; i++) {
            out[off + i] = Math.log(x[xOff + i]);
        }
    }
}
