package com.curioloop.yum4j.ssm.particle.dist;

import com.curioloop.yum4j.linalg.blas.BLAS;
import com.curioloop.yum4j.math.VectorOps;

import java.util.random.RandomGenerator;

/**
 * Multivariate normal distribution on {@code (d, N)} column-major
 * particle buffers.
 *
 * <p>Use {@link #of(double[], double[])} to construct; it Cholesky-factors
 * the covariance and precomputes the log-normaliser.
 *
 * @param dim            number of dimensions
 * @param mean           mean vector (length d)
 * @param covariance     covariance matrix, row-major d×d
 * @param lowerCholesky  lower Cholesky factor L where Σ = LLᵀ, row-major d×d
 * @param logNormaliser  precomputed {@code -0.5·(d·log(2π) + log|Σ|)}
 */
public record MvNormalDistribution(
    int dim,
    double[] mean,
    double[] covariance,
    double[] lowerCholesky,
    double logNormaliser
) implements MultivariateDistribution {

    private static final double LOG_2PI = Math.log(2.0 * Math.PI);

    /**
     * Construct a multivariate normal {@code N(mean, cov)}.
     *
     * @param mean mean vector of length {@code d}
     * @param cov  covariance matrix, row-major {@code d*d}; must be positive definite
     * @throws IllegalArgumentException if {@code cov} is not positive definite
     */
    public static MvNormalDistribution of(double[] mean, double[] cov) {
        if (mean == null || cov == null) {
            throw new IllegalArgumentException("mean and cov must not be null");
        }
        if (mean.length == 0) {
            throw new IllegalArgumentException("mean must be non-empty");
        }
        int d = mean.length;
        if (cov.length != d * d) {
            throw new IllegalArgumentException(
                "cov length must be d*d (" + (d * d) + "); got " + cov.length);
        }

        double[] meanCopy = mean.clone();
        double[] covCopy = cov.clone();

        // Cholesky factorisation: L such that Σ = LLᵀ.
        double[] L = cov.clone();
        int info = BLAS.dpotrf(BLAS.Uplo.Lower, d, L, 0, d);
        if (info != 0) {
            throw new IllegalArgumentException(
                "covariance not positive definite; smallest problematic leading minor: " + info);
        }
        // Zero the strict upper triangle.
        for (int j = 0; j < d; j++) {
            int rowOff = j * d;
            for (int k = j + 1; k < d; k++) {
                L[rowOff + k] = 0.0;
            }
        }

        // logNormaliser = -0.5 * (d*log(2π) + log|Σ|); log|Σ| = 2 * Σ log L[j,j].
        double logDet2 = 0.0;
        for (int j = 0; j < d; j++) {
            logDet2 += Math.log(L[j * d + j]);
        }
        double logNorm = -0.5 * d * LOG_2PI - logDet2;

        return new MvNormalDistribution(d, meanCopy, covCopy, L, logNorm);
    }

    @Override
    public void mean(double[] out) {
        System.arraycopy(mean, 0, out, 0, dim);
    }

    /**
     * Single-point log-pdf convenience wrapper.
     */
    public double logPdf(double[] x) {
        if (x == null || x.length != dim) {
            throw new IllegalArgumentException(
                "x must have length " + dim + "; got " + (x == null ? "null" : x.length));
        }
        double[] out = new double[1];
        logPdfBatch(x, 0, 1, out, 0, null);
        return out[0];
    }

    @Override
    public void logPdfBatch(double[] x, int xOff, int N,
                            double[] out, int outOff,
                            double[] scratch) {
        if (N == 0) return;
        final int d = dim;
        double[] work = (scratch != null && scratch.length >= d * N) ? scratch : new double[d * N];

        // 1. work := x - μ
        for (int j = 0; j < d; j++) {
            double muj = mean[j];
            int rowOff = j * N;
            for (int n = 0; n < N; n++) {
                work[rowOff + n] = x[xOff + rowOff + n] - muj;
            }
        }

        // 2. work := L⁻¹ · work
        BLAS.dtrsm(
            BLAS.Side.Left, BLAS.Uplo.Lower, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit,
            d, N, 1.0,
            lowerCholesky, 0, d,
            work, 0, N
        );

        // 3. out[n] = logNormaliser - 0.5 * ||work[:, n]||²
        final double logC = logNormaliser;
        for (int n = 0; n < N; n++) {
            double ss = 0.0;
            for (int j = 0; j < d; j++) {
                double v = work[j * N + n];
                ss += v * v;
            }
            out[outOff + n] = logC - 0.5 * ss;
        }
    }

    @Override
    public void sample(RandomGenerator g, int N,
                       double[] x, int xOff,
                       double[] scratch) {
        if (N == 0) return;
        final int d = dim;

        // Step 1: fill X with iid N(0, 1) draws.
        for (int j = 0; j < d; j++) {
            int rowOff = xOff + j * N;
            for (int n = 0; n < N; n++) {
                x[rowOff + n] = g.nextGaussian();
            }
        }

        // Step 2: X := L · X
        applyLowerCholeskyBlas(lowerCholesky, d, x, xOff, N);

        // Step 3: X += μ
        addMean(mean, d, x, xOff, N);
    }

    // ── Shared helpers ────────────────────────────────────────────────

    /** {@code X := L · X} via BLAS dtrmm on a (d, N) column-major buffer. */
    static void applyLowerCholeskyBlas(double[] L, int d, double[] x, int xOff, int N) {
        BLAS.dtrmm(
            BLAS.Side.Left, BLAS.Uplo.Lower, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit,
            d, N, 1.0,
            L, 0, d,
            x, xOff, N
        );
    }

    /** Add {@code μ[j]} to each row of a (d, N) column-major buffer. */
    static void addMean(double[] mean, int d, double[] x, int xOff, int N) {
        for (int j = 0; j < d; j++) {
            VectorOps.addConst(mean[j], x, xOff + j * N, N);
        }
    }
}
