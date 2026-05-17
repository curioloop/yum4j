package com.curioloop.yum4j.ssm.particle.qmc;

/**
 * Sobol quasi-random sequence generator using Gray-code optimisation.
 *
 * <p>Supports up to 10 dimensions with hardcoded Joe-Kuo direction numbers.
 * Points are generated in [ε, 1−ε] with ε = 1e-10 to avoid boundary
 * singularities. Output is column-major (d, N): {@code out[j*N + n]}.
 *
 * <p>For N not a power of 2, the next power of 2 is used internally
 * and only the first N points are returned.
 */
public final class Sobol {

    private static final double EPS = 1e-10;
    private static final int MAX_BITS = 30;
    private static final int MAX_DIM = 10;

    /**
     * Joe-Kuo direction numbers for dimensions 2-10.
     * Each row: [s, a, m_1, m_2, ..., m_s] where s is the degree,
     * a is the polynomial coefficient, and m_i are initial direction numbers.
     * Dimension 1 uses the van der Corput sequence (all direction numbers = 1).
     */
    private static final int[][] DIRECTION_NUMBERS = {
        // dim 2: s=1, a=0, m=[1]
        {1, 0, 1},
        // dim 3: s=2, a=1, m=[1,1]
        {2, 1, 1, 1},
        // dim 4: s=3, a=1, m=[1,3,1]
        {3, 1, 1, 3, 1},
        // dim 5: s=3, a=2, m=[1,1,1]
        {3, 2, 1, 1, 1},
        // dim 6: s=4, a=1, m=[1,1,3,3]
        {4, 1, 1, 1, 3, 3},
        // dim 7: s=4, a=4, m=[1,3,5,13]
        {4, 4, 1, 3, 5, 13},
        // dim 8: s=5, a=2, m=[1,1,5,5,17]
        {5, 2, 1, 1, 5, 5, 17},
        // dim 9: s=5, a=4, m=[1,1,5,5,5]
        {5, 4, 1, 1, 5, 5, 5},
        // dim 10: s=5, a=7, m=[1,1,7,11,19]
        {5, 7, 1, 1, 7, 11, 19},
    };

    private Sobol() {}

    /**
     * Generate the first N points of the Sobol sequence in d dimensions.
     *
     * @param N number of points
     * @param d number of dimensions (1 ≤ d ≤ 10)
     * @return column-major double[d*N]: out[j*N + n] = point n, dimension j
     */
    public static double[] generate(int N, int d) {
        if (d < 1 || d > MAX_DIM) {
            throw new IllegalArgumentException("d must be in [1, " + MAX_DIM + "], got " + d);
        }
        if (N <= 0) {
            return new double[0];
        }

        double[] out = new double[d * N];
        generate(N, d, out, 0);
        return out;
    }

    /**
     * Generate the first N points into a pre-allocated buffer.
     *
     * @param N      number of points
     * @param d      number of dimensions (1 ≤ d ≤ 10)
     * @param out    output buffer (length ≥ outOff + d*N)
     * @param outOff offset into output buffer
     */
    public static void generate(int N, int d, double[] out, int outOff) {
        if (d < 1 || d > MAX_DIM) {
            throw new IllegalArgumentException("d must be in [1, " + MAX_DIM + "], got " + d);
        }
        if (N <= 0) return;

        // Compute direction numbers for each dimension
        long[][] V = new long[d][MAX_BITS];
        computeDirectionNumbers(d, V);

        double scale = 1.0 / (1L << MAX_BITS);

        // Generate points using Gray code
        long[] x = new long[d]; // current point (integer representation)

        // Point 0 is the origin → clamp to EPS
        for (int j = 0; j < d; j++) {
            out[outOff + j * N] = clamp(0.0);
        }

        for (int n = 1; n < N; n++) {
            // Find the rightmost zero bit of n-1 (Gray code index)
            int c = rightmostZeroBit(n - 1);

            // XOR with direction number
            for (int j = 0; j < d; j++) {
                x[j] ^= V[j][c];
                out[outOff + j * N + n] = clamp(x[j] * scale);
            }
        }
    }

    /**
     * Compute direction numbers for all dimensions.
     */
    private static void computeDirectionNumbers(int d, long[][] V) {
        // Dimension 1: van der Corput (all direction numbers = 1 << (MAX_BITS - i - 1))
        for (int i = 0; i < MAX_BITS; i++) {
            V[0][i] = 1L << (MAX_BITS - 1 - i);
        }

        // Dimensions 2..d
        for (int dim = 1; dim < d; dim++) {
            int[] params = DIRECTION_NUMBERS[dim - 1];
            int s = params[0];
            int a = params[1];

            // Initial direction numbers (scaled to MAX_BITS)
            for (int i = 0; i < s && i < MAX_BITS; i++) {
                long m = params[2 + i];
                V[dim][i] = m << (MAX_BITS - 1 - i);
            }

            // Recurrence for remaining direction numbers
            for (int i = s; i < MAX_BITS; i++) {
                long val = V[dim][i - s] ^ (V[dim][i - s] >> s);
                for (int k = 1; k < s; k++) {
                    val ^= ((a >> (s - 1 - k)) & 1) * V[dim][i - k];
                }
                V[dim][i] = val;
            }
        }
    }

    /**
     * Find the position of the rightmost zero bit (0-indexed).
     */
    private static int rightmostZeroBit(int n) {
        int c = 0;
        int val = n;
        while ((val & 1) == 1) {
            val >>= 1;
            c++;
        }
        return c;
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
