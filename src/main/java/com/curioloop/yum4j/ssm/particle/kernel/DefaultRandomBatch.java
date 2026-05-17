package com.curioloop.yum4j.ssm.particle.kernel;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

/**
 * SIMD-accelerated {@link RandomBatch} implementation using
 * {@code L64X128MixRandom} as the underlying generator and a
 * vectorised Box–Muller transform for Gaussian draws.
 *
 * <p>The Box–Muller kernel uses {@code SPECIES_256} (4 doubles per
 * lane-group) and produces 2×LANES = 8 Gaussians per iteration.
 * A scalar patch handles the rare {@code u1 == 0.0} case (rate
 * bounded by 2⁻⁵² per draw) to avoid {@code log(0) = -Infinity}.
 *
 * <p>This class is <b>not thread-safe</b>.
 */
final class DefaultRandomBatch implements RandomBatch {

    private static final String ALGORITHM = "L64X128MixRandom";
    private static final String SIMD_PROPERTY = "yum4j.vector";
    private static final boolean SIMD_ENABLED =
        !"false".equalsIgnoreCase(System.getProperty(SIMD_PROPERTY, "true"));

    static final VectorSpecies<Double> SPECIES = preferredSpecies256Cap();
    static final int LANES = SPECIES.length();
    static final boolean SIMD_SUPPORTED = detectSimd();

    private static final double TWO_PI = 2.0 * Math.PI;

    private final long masterSeed;
    private final RandomGenerator rng;

    /** Reusable buffer for uniform draws fed into Box–Muller. Length = 2*LANES. */
    private final double[] uniformBuf;

    DefaultRandomBatch(long seed) {
        this.masterSeed = seed;
        this.rng = RandomGeneratorFactory.of(ALGORITHM).create(seed);
        this.uniformBuf = new double[2 * LANES];
    }

    // ------------------------------------------------------------------
    // RandomBatch interface
    // ------------------------------------------------------------------

    @Override
    public void nextUniforms(double[] out, int off, int N) {
        for (int i = 0; i < N; i++) {
            out[off + i] = rng.nextDouble();
        }
    }

    @Override
    public void nextGaussians(double[] out, int off, int N) {
        final int blockSize = 2 * LANES; // produces 2*LANES gaussians per block
        int i = 0;

        // SIMD blocks
        final int simdBound = (N / blockSize) * blockSize;
        for (; i < simdBound; i += blockSize) {
            simdBoxMuller(out, off + i);
        }

        // Scalar tail
        for (; i < N; i++) {
            out[off + i] = scalarGaussian();
        }
    }

    @Override
    public void nextGaussiansAffineInto(double[] out, int off, int N,
                                        double mu, double sigma) {
        final int blockSize = 2 * LANES;
        int i = 0;

        final DoubleVector sigmaVec = DoubleVector.broadcast(SPECIES, sigma);
        final DoubleVector muVec = DoubleVector.broadcast(SPECIES, mu);

        final int simdBound = (N / blockSize) * blockSize;
        for (; i < simdBound; i += blockSize) {
            simdBoxMullerAffine(out, off + i, sigmaVec, muVec);
        }

        // Scalar tail — fused affine inline
        for (; i < N; i++) {
            out[off + i] = Math.fma(scalarGaussian(), sigma, mu);
        }
    }

    @Override
    public void nextGaussiansAffineInto(double[] out, int off, int N,
                                        double[] mu, int muOff, double sigma) {
        final int blockSize = 2 * LANES;
        int i = 0;

        final DoubleVector sigmaVec = DoubleVector.broadcast(SPECIES, sigma);

        final int simdBound = (N / blockSize) * blockSize;
        for (; i < simdBound; i += blockSize) {
            // Load per-particle mean in two LANES-sized chunks matching the
            // Box–Muller (u1=cos, u2=sin) output ordering.
            DoubleVector muVec1 = DoubleVector.fromArray(SPECIES, mu, muOff + i);
            DoubleVector muVec2 = DoubleVector.fromArray(SPECIES, mu, muOff + i + LANES);
            simdBoxMullerAffineVecMean(out, off + i, sigmaVec, muVec1, muVec2);
        }

        // Scalar tail — fused affine inline, per-particle mean
        for (; i < N; i++) {
            out[off + i] = Math.fma(scalarGaussian(), sigma, mu[muOff + i]);
        }
    }

    @Override
    public void addAffineGaussianInto(double[] out, int off, int N, double sigma) {
        final int blockSize = 2 * LANES;
        int i = 0;

        final DoubleVector sigmaVec = DoubleVector.broadcast(SPECIES, sigma);

        final int simdBound = (N / blockSize) * blockSize;
        for (; i < simdBound; i += blockSize) {
            simdBoxMullerAddFma(out, off + i, sigmaVec);
        }

        // Scalar tail — read-modify-write with fused arithmetic
        for (; i < N; i++) {
            out[off + i] = Math.fma(scalarGaussian(), sigma, out[off + i]);
        }
    }

    @Override
    public void nextExponentials(double[] out, int off, int N, double lambda) {
        final double invLambda = 1.0 / lambda;
        for (int i = 0; i < N; i++) {
            double u = rng.nextDouble();
            // Guard against u == 0.0 (same as Box–Muller patch)
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
        // Deterministic sub-seed from (masterSeed, key) using a simple
        // mixing function. The split produces a fresh L64X128MixRandom
        // whose stream is independent of the parent.
        long subSeed = mixSeed(masterSeed, key);
        return new DefaultRandomBatch(subSeed);
    }

    // ------------------------------------------------------------------
    // SIMD Box–Muller kernel
    // ------------------------------------------------------------------

    /**
     * Produces {@code 2*LANES} Gaussians into {@code out[off..off+2*LANES)}.
     * Uses the standard Box–Muller transform:
     * <pre>
     *   g1 = sqrt(-2 ln u1) * cos(2π u2)
     *   g2 = sqrt(-2 ln u1) * sin(2π u2)
     * </pre>
     */
    private void simdBoxMuller(double[] out, int off) {
        // Fill uniform buffer: first LANES for u1, next LANES for u2
        for (int k = 0; k < uniformBuf.length; k++) {
            uniformBuf[k] = rng.nextDouble();
        }

        // Load u1 and u2 as SIMD vectors
        DoubleVector u1 = DoubleVector.fromArray(SPECIES, uniformBuf, 0);
        DoubleVector u2 = DoubleVector.fromArray(SPECIES, uniformBuf, LANES);

        // Scalar patch: detect u1 == 0.0 lanes and resample
        VectorMask<Double> zeroMask = u1.eq(0.0);
        if (zeroMask.anyTrue()) {
            u1 = patchZeroes(u1, zeroMask);
        }

        // r = sqrt(-2 * ln(u1))
        DoubleVector r = u1.lanewise(VectorOperators.LOG)
                           .neg()
                           .mul(2.0)
                           .lanewise(VectorOperators.SQRT);

        // arg = 2π * u2
        DoubleVector arg = u2.mul(TWO_PI);

        // g1 = r * cos(arg), g2 = r * sin(arg)
        DoubleVector g1 = r.mul(arg.lanewise(VectorOperators.COS));
        DoubleVector g2 = r.mul(arg.lanewise(VectorOperators.SIN));

        // Store results
        g1.intoArray(out, off);
        g2.intoArray(out, off + LANES);
    }

    /**
     * Fused Box–Muller + scalar-mean affine: produces {@code 2*LANES}
     * values {@code sigma * g + mu} into {@code out[off..off+2*LANES)}
     * in a single SIMD pass (R6.2). Equivalent to
     * {@link #simdBoxMuller} followed by {@code * sigma + mu}, collapsed
     * via FMA into one arithmetic chain.
     */
    private void simdBoxMullerAffine(double[] out, int off,
                                     DoubleVector sigmaVec, DoubleVector muVec) {
        for (int k = 0; k < uniformBuf.length; k++) {
            uniformBuf[k] = rng.nextDouble();
        }

        DoubleVector u1 = DoubleVector.fromArray(SPECIES, uniformBuf, 0);
        DoubleVector u2 = DoubleVector.fromArray(SPECIES, uniformBuf, LANES);

        VectorMask<Double> zeroMask = u1.eq(0.0);
        if (zeroMask.anyTrue()) {
            u1 = patchZeroes(u1, zeroMask);
        }

        DoubleVector r = u1.lanewise(VectorOperators.LOG)
                           .neg()
                           .mul(2.0)
                           .lanewise(VectorOperators.SQRT);
        DoubleVector arg = u2.mul(TWO_PI);

        DoubleVector g1 = r.mul(arg.lanewise(VectorOperators.COS));
        DoubleVector g2 = r.mul(arg.lanewise(VectorOperators.SIN));

        // Fused affine: out = sigma * g + mu  (single FMA per lane)
        g1.fma(sigmaVec, muVec).intoArray(out, off);
        g2.fma(sigmaVec, muVec).intoArray(out, off + LANES);
    }

    /**
     * Fused Box–Muller + per-particle-mean affine (R6.5). Writes
     * {@code sigma * g + mu[i]} for {@code 2*LANES} particles into
     * {@code out[off..off+2*LANES)}.
     */
    private void simdBoxMullerAffineVecMean(double[] out, int off,
                                            DoubleVector sigmaVec,
                                            DoubleVector muVec1,
                                            DoubleVector muVec2) {
        for (int k = 0; k < uniformBuf.length; k++) {
            uniformBuf[k] = rng.nextDouble();
        }

        DoubleVector u1 = DoubleVector.fromArray(SPECIES, uniformBuf, 0);
        DoubleVector u2 = DoubleVector.fromArray(SPECIES, uniformBuf, LANES);

        VectorMask<Double> zeroMask = u1.eq(0.0);
        if (zeroMask.anyTrue()) {
            u1 = patchZeroes(u1, zeroMask);
        }

        DoubleVector r = u1.lanewise(VectorOperators.LOG)
                           .neg()
                           .mul(2.0)
                           .lanewise(VectorOperators.SQRT);
        DoubleVector arg = u2.mul(TWO_PI);

        DoubleVector g1 = r.mul(arg.lanewise(VectorOperators.COS));
        DoubleVector g2 = r.mul(arg.lanewise(VectorOperators.SIN));

        g1.fma(sigmaVec, muVec1).intoArray(out, off);
        g2.fma(sigmaVec, muVec2).intoArray(out, off + LANES);
    }

    /**
     * Fused Box–Muller + read-modify-write accumulate (R6.4). Reads
     * {@code 2*LANES} base values from {@code out}, writes
     * {@code out[i] = base[i] + sigma * g[i]} in one SIMD pass.
     */
    private void simdBoxMullerAddFma(double[] out, int off, DoubleVector sigmaVec) {
        for (int k = 0; k < uniformBuf.length; k++) {
            uniformBuf[k] = rng.nextDouble();
        }

        DoubleVector u1 = DoubleVector.fromArray(SPECIES, uniformBuf, 0);
        DoubleVector u2 = DoubleVector.fromArray(SPECIES, uniformBuf, LANES);

        VectorMask<Double> zeroMask = u1.eq(0.0);
        if (zeroMask.anyTrue()) {
            u1 = patchZeroes(u1, zeroMask);
        }

        DoubleVector r = u1.lanewise(VectorOperators.LOG)
                           .neg()
                           .mul(2.0)
                           .lanewise(VectorOperators.SQRT);
        DoubleVector arg = u2.mul(TWO_PI);

        DoubleVector g1 = r.mul(arg.lanewise(VectorOperators.COS));
        DoubleVector g2 = r.mul(arg.lanewise(VectorOperators.SIN));

        // Load base[i..i+LANES) from out, accumulate via FMA: out = base + sigma * g
        DoubleVector base1 = DoubleVector.fromArray(SPECIES, out, off);
        DoubleVector base2 = DoubleVector.fromArray(SPECIES, out, off + LANES);

        g1.fma(sigmaVec, base1).intoArray(out, off);
        g2.fma(sigmaVec, base2).intoArray(out, off + LANES);
    }

    /**
     * Replaces zero lanes in {@code u1} with fresh non-zero uniform draws.
     * This is the scalar patch — expected rate bounded by 2⁻⁵² per draw.
     * Reuses {@code uniformBuf[0..LANES)} as scratch to avoid allocation.
     */
    private DoubleVector patchZeroes(DoubleVector u1, VectorMask<Double> zeroMask) {
        // Extract to the first LANES slots of uniformBuf (safe: we've
        // already consumed those values), patch zero lanes, reload.
        u1.intoArray(uniformBuf, 0);
        for (int k = 0; k < LANES; k++) {
            if (zeroMask.laneIsSet(k)) {
                double v;
                do {
                    v = rng.nextDouble();
                } while (v == 0.0);
                uniformBuf[k] = v;
            }
        }
        return DoubleVector.fromArray(SPECIES, uniformBuf, 0);
    }

    /**
     * Scalar Gaussian using Box–Muller for the tail elements.
     * Caches the second value from each pair.
     */
    private boolean hasSpare;
    private double spare;

    private double scalarGaussian() {
        if (hasSpare) {
            hasSpare = false;
            return spare;
        }
        double u1, u2;
        u1 = rng.nextDouble();
        while (u1 == 0.0) {
            u1 = rng.nextDouble();
        }
        u2 = rng.nextDouble();

        double r = Math.sqrt(-2.0 * Math.log(u1));
        double angle = TWO_PI * u2;
        spare = r * Math.sin(angle);
        hasSpare = true;
        return r * Math.cos(angle);
    }

    // ------------------------------------------------------------------
    // Seed mixing
    // ------------------------------------------------------------------

    /**
     * Mixes master seed and key into a deterministic sub-seed.
     * Uses a variant of SplitMix64's mixing function.
     */
    static long mixSeed(long masterSeed, long key) {
        long z = masterSeed + key * 0x9E3779B97F4A7C15L;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    // ------------------------------------------------------------------
    // SIMD detection
    // ------------------------------------------------------------------

    private static VectorSpecies<Double> preferredSpecies256Cap() {
        VectorSpecies<Double> preferred = DoubleVector.SPECIES_PREFERRED;
        if (preferred.length() >= DoubleVector.SPECIES_256.length()) {
            return DoubleVector.SPECIES_256;
        }
        return preferred;
    }

    private static boolean detectSimd() {
        if (!SIMD_ENABLED) return false;
        if (LANES <= 1) return false;
        try {
            // Probe: attempt a trivial SIMD operation to confirm linkage
            double[] probe = new double[LANES * 2];
            for (int i = 0; i < probe.length; i++) probe[i] = (i + 1) * 0.1;
            DoubleVector v = DoubleVector.fromArray(SPECIES, probe, 0);
            v.lanewise(VectorOperators.LOG).intoArray(probe, 0);
            return true;
        } catch (LinkageError e) {
            return false;
        }
    }
}
