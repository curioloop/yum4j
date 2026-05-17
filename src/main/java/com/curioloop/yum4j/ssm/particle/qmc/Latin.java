package com.curioloop.yum4j.ssm.particle.qmc;

import java.util.random.RandomGenerator;

/**
 * Latin Hypercube Sampling (LHS) generator.
 *
 * <p>Generates a stratified random sample of N points in [0,1]^d where
 * each dimension is divided into N equal strata and exactly one point
 * falls in each stratum per dimension.
 *
 * <p>Points are clamped to [ε, 1−ε] with ε = 1e-10. Output is
 * column-major (d, N): {@code out[j*N + n]}.
 */
public final class Latin {

    private static final double EPS = 1e-10;

    private Latin() {}

    /**
     * Generate a Latin Hypercube Sample of N points in d dimensions.
     *
     * @param N number of points
     * @param d number of dimensions
     * @param g random number generator
     * @return column-major double[d*N]: out[j*N + n] = point n, dimension j
     */
    public static double[] generate(int N, int d, RandomGenerator g) {
        if (N <= 0 || d <= 0) return new double[0];

        double[] out = new double[d * N];
        generate(N, d, out, 0, g);
        return out;
    }

    /**
     * Generate a Latin Hypercube Sample into a pre-allocated buffer.
     *
     * @param N      number of points
     * @param d      number of dimensions
     * @param out    output buffer (length ≥ outOff + d*N)
     * @param outOff offset into output buffer
     * @param g      random number generator
     */
    public static void generate(int N, int d, double[] out, int outOff,
                                RandomGenerator g) {
        if (N <= 0 || d <= 0) return;

        int[] perm = new int[N];

        for (int j = 0; j < d; j++) {
            // Generate a random permutation of 0..N-1
            for (int n = 0; n < N; n++) {
                perm[n] = n;
            }
            // Fisher-Yates shuffle
            for (int n = N - 1; n > 0; n--) {
                int k = g.nextInt(n + 1);
                int tmp = perm[n];
                perm[n] = perm[k];
                perm[k] = tmp;
            }

            // point[n, j] = (perm[n] + U_nj) / N
            for (int n = 0; n < N; n++) {
                double val = (perm[n] + g.nextDouble()) / N;
                out[outOff + j * N + n] = clamp(val);
            }
        }
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
