package com.curioloop.yum4j.ssm.particle.dist;

import com.curioloop.yum4j.linalg.blas.BLAS;
import com.curioloop.yum4j.math.Gamma;
import com.curioloop.yum4j.stats.Distribution;

import java.util.random.RandomGenerator;

/**
 * Multivariate Student's t distribution on {@code (d, N)} column-major
 * particle buffers.
 *
 * <p>Use {@link #of(double[], double[], double)} to construct; it
 * Cholesky-factors the scale matrix and precomputes the log-normaliser.
 *
 * @param dim            number of dimensions
 * @param df             degrees of freedom (> 0)
 * @param mean           mean vector (length d)
 * @param covariance     scale matrix, row-major d×d
 * @param lowerCholesky  lower Cholesky factor L where Σ = LLᵀ, row-major d×d
 * @param logNormaliser  precomputed normalising constant
 */
public record MvStudentDistribution(
    int dim,
    double df,
    double[] mean,
    double[] covariance,
    double[] lowerCholesky,
    double logNormaliser
) implements MultivariateDistribution {

    /**
     * Construct a multivariate Student's t distribution.
     *
     * @param mean mean vector of length {@code d}
     * @param cov  scale matrix, row-major {@code d*d}; must be positive definite
     * @param df   degrees of freedom, must be > 0
     */
    public static MvStudentDistribution of(double[] mean, double[] cov, double df) {
        if (mean == null || cov == null) {
            throw new IllegalArgumentException("mean and cov must not be null");
        }
        if (mean.length == 0) {
            throw new IllegalArgumentException("mean must be non-empty");
        }
        if (!(df > 0.0) || Double.isNaN(df)) {
            throw new IllegalArgumentException("df must be > 0: " + df);
        }
        int d = mean.length;
        if (cov.length != d * d) {
            throw new IllegalArgumentException(
                "cov length must be d*d (" + (d * d) + "); got " + cov.length);
        }

        double[] meanCopy = mean.clone();
        double[] covCopy = cov.clone();

        double[] L = cov.clone();
        int info = BLAS.dpotrf(BLAS.Uplo.Lower, d, L, 0, d);
        if (info != 0) {
            throw new IllegalArgumentException(
                "covariance not positive definite; smallest problematic leading minor: " + info);
        }
        for (int j = 0; j < d; j++) {
            int rowOff = j * d;
            for (int k = j + 1; k < d; k++) {
                L[rowOff + k] = 0.0;
            }
        }

        double logDetHalf = 0.0;
        for (int j = 0; j < d; j++) {
            logDetHalf += Math.log(L[j * d + j]);
        }
        double logNorm =
            Gamma.lgamma(0.5 * (df + d))
                - Gamma.lgamma(0.5 * df)
                - 0.5 * d * Math.log(df * Math.PI)
                - logDetHalf;

        return new MvStudentDistribution(d, df, meanCopy, covCopy, L, logNorm);
    }

    @Override
    public void mean(double[] out) {
        System.arraycopy(mean, 0, out, 0, dim);
    }

    /**
     * Single-point log-pdf convenience.
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

        // work := x - μ
        for (int j = 0; j < d; j++) {
            double muj = mean[j];
            int rowOff = j * N;
            for (int n = 0; n < N; n++) {
                work[rowOff + n] = x[xOff + rowOff + n] - muj;
            }
        }

        // work := L⁻¹ · (x - μ)
        BLAS.dtrsm(
            BLAS.Side.Left, BLAS.Uplo.Lower, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit,
            d, N, 1.0,
            lowerCholesky, 0, d,
            work, 0, N
        );

        final double logC = logNormaliser;
        final double halfSum = 0.5 * (df + d);
        final double invDf = 1.0 / df;
        for (int n = 0; n < N; n++) {
            double m = 0.0;
            for (int j = 0; j < d; j++) {
                double v = work[j * N + n];
                m += v * v;
            }
            out[outOff + n] = logC - halfSum * Math.log1p(m * invDf);
        }
    }

    @Override
    public void sample(RandomGenerator g, int N,
                       double[] x, int xOff,
                       double[] scratch) {
        if (N == 0) return;
        final int d = dim;

        // Step 1: fill x with iid N(0, 1).
        for (int j = 0; j < d; j++) {
            int rowOff = xOff + j * N;
            for (int n = 0; n < N; n++) {
                x[rowOff + n] = g.nextGaussian();
            }
        }

        // Step 2: x := L · x
        MvNormalDistribution.applyLowerCholeskyBlas(lowerCholesky, d, x, xOff, N);

        // Step 3: scale by sqrt(df / ChiSq(df)) and add μ.
        final double halfDf = 0.5 * df;
        for (int n = 0; n < N; n++) {
            double w = Distribution.nextGamma(g, halfDf) * 2.0;
            double s = Math.sqrt(df / w);
            for (int j = 0; j < d; j++) {
                x[xOff + j * N + n] = mean[j] + x[xOff + j * N + n] * s;
            }
        }
    }
}
