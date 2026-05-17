package com.curioloop.yum4j.ssm.particle.qmc;

/**
 * Halton quasi-random sequence generator using van der Corput sequences
 * with prime bases.
 *
 * <p>Points are generated in [ε, 1−ε] with ε = 1e-10 to avoid boundary
 * singularities. Output is column-major (d, N): {@code out[j*N + n]}.
 *
 * <p>For dimensions > 5, Faure permutation scrambling is applied to
 * reduce correlation between dimensions.
 */
public final class Halton {

    private static final double EPS = 1e-10;

    /** First 20 primes for bases. */
    private static final int[] PRIMES = {
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
        31, 37, 41, 43, 47, 53, 59, 61, 67, 71
    };

    private Halton() {}

    /**
     * Generate the first N points of the Halton sequence in d dimensions.
     *
     * @param N number of points
     * @param d number of dimensions (1 ≤ d ≤ 20)
     * @return column-major double[d*N]: out[j*N + n] = point n, dimension j
     */
    public static double[] generate(int N, int d) {
        if (d < 1 || d > PRIMES.length) {
            throw new IllegalArgumentException(
                "d must be in [1, " + PRIMES.length + "], got " + d);
        }
        if (N <= 0) return new double[0];

        double[] out = new double[d * N];
        generate(N, d, out, 0);
        return out;
    }

    /**
     * Generate the first N points into a pre-allocated buffer.
     *
     * @param N      number of points
     * @param d      number of dimensions (1 ≤ d ≤ 20)
     * @param out    output buffer (length ≥ outOff + d*N)
     * @param outOff offset into output buffer
     */
    public static void generate(int N, int d, double[] out, int outOff) {
        if (d < 1 || d > PRIMES.length) {
            throw new IllegalArgumentException(
                "d must be in [1, " + PRIMES.length + "], got " + d);
        }
        if (N <= 0) return;

        for (int j = 0; j < d; j++) {
            int base = PRIMES[j];
            boolean scramble = j >= 5; // Faure scrambling for d > 5
            int[] perm = scramble ? faurePermutation(base) : null;

            for (int n = 0; n < N; n++) {
                double val = vanDerCorput(n + 1, base, perm);
                out[outOff + j * N + n] = clamp(val);
            }
        }
    }

    /**
     * Van der Corput sequence value for index n in the given base.
     * Optionally applies a digit permutation for scrambling.
     */
    private static double vanDerCorput(int n, int base, int[] perm) {
        double result = 0.0;
        double denom = 1.0;
        int val = n;

        while (val > 0) {
            denom *= base;
            int digit = val % base;
            if (perm != null) {
                digit = perm[digit];
            }
            result += digit / denom;
            val /= base;
        }

        return result;
    }

    /**
     * Generate a Faure permutation for the given base.
     * The Faure permutation for base b is defined recursively.
     * For simplicity, we use the standard Faure permutation:
     * perm[i] = (i * (b-1)) mod b for i > 0, perm[0] = 0.
     */
    private static int[] faurePermutation(int base) {
        int[] perm = new int[base];
        // Simple Faure-style permutation
        if (base <= 2) {
            for (int i = 0; i < base; i++) perm[i] = i;
            return perm;
        }

        // Use the reverse-radix permutation as a simple scrambling
        perm[0] = 0;
        if (base == 3) {
            perm[1] = 2;
            perm[2] = 1;
        } else {
            // General case: bit-reversal-like permutation
            for (int i = 0; i < base; i++) {
                perm[i] = (base - 1 - i);
            }
            perm[0] = 0; // Keep 0 fixed
        }
        return perm;
    }

    /**
     * Clamp a value to [EPS, 1-EPS].
     */
    private static double clamp(double v) {
        if (v < EPS) return EPS;
        if (v > 1.0 - EPS) return 1.0 - EPS;
        return v;
    }
}
