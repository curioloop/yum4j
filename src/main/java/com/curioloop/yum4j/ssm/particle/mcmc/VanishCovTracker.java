package com.curioloop.yum4j.ssm.particle.mcmc;

import com.curioloop.yum4j.linalg.blas.BLAS;

import java.util.Arrays;

/**
 * Recursive covariance estimator with vanishing adaptation rate.
 *
 * <p>Tracks a running mean {@code μ} and covariance {@code Σ} of a
 * stream of d-dimensional vectors using the exponential decay
 * {@code γ = (t+1)^{−α}}. After each update, an in-place Cholesky
 * decomposition is attempted; if the updated covariance is not positive
 * definite the previous Cholesky factor is retained unchanged.
 *
 * <p>All operations are allocation-free after construction: the mean,
 * covariance, and Cholesky arrays are pre-allocated and updated in place.
 */
public final class VanishCovTracker {

    private final double alpha;
    private final int dim;
    private int t;
    private final double[] mu;       // length dim
    private final double[] sigma;    // dim×dim row-major
    private final double[] L;        // dim×dim lower-triangular Cholesky
    private final double[] Lsave;    // backup of L before Cholesky attempt

    /**
     * Create a new tracker.
     *
     * @param alpha  decay exponent (typical default 0.6)
     * @param dim    parameter dimension
     * @param sigma0 initial covariance estimate (dim×dim row-major), or
     *               {@code null} to use the identity matrix
     */
    public VanishCovTracker(double alpha, int dim, double[] sigma0) {
        this.alpha = alpha;
        this.dim = dim;
        this.t = 0;
        this.mu = new double[dim];
        this.sigma = new double[dim * dim];
        this.L = new double[dim * dim];
        this.Lsave = new double[dim * dim];

        // Initialise sigma to sigma0 (or identity)
        if (sigma0 != null) {
            System.arraycopy(sigma0, 0, sigma, 0, dim * dim);
        } else {
            for (int i = 0; i < dim; i++) {
                sigma[i * dim + i] = 1.0;
            }
        }

        // Compute Cholesky of initial sigma into L
        boolean ok = choleskyDecompose(sigma, L);
        if (!ok) {
            // sigma0 was not PD; fall back to identity Cholesky (= identity)
            Arrays.fill(L, 0.0);
            for (int i = 0; i < dim; i++) {
                L[i * dim + i] = 1.0;
            }
        }
    }

    /**
     * Update the running mean and covariance with a new observation.
     *
     * <p>Increments the internal counter, computes the decay factor
     * {@code γ = (t+1)^{−α}}, and updates μ and Σ in place. Then
     * attempts an in-place Cholesky decomposition of the updated Σ
     * into L. If the decomposition fails (Σ is not positive definite),
     * L is left unchanged (restored from backup).
     *
     * @param v    array containing the new observation vector
     * @param vOff offset into {@code v} where the vector starts
     */
    public void update(double[] v, int vOff) {
        t++;
        double gamma = Math.pow(t + 1, -alpha);
        double oneMinusGamma = 1.0 - gamma;

        // Update mean: mu[j] = (1 - γ) * mu[j] + γ * v[vOff + j]
        for (int j = 0; j < dim; j++) {
            mu[j] = oneMinusGamma * mu[j] + gamma * v[vOff + j];
        }

        // Update covariance: sigma[i*dim+j] = (1-γ)*sigma[i*dim+j] + γ*(v[i]-mu[i])*(v[j]-mu[j])
        for (int i = 0; i < dim; i++) {
            double di = v[vOff + i] - mu[i];
            for (int j = 0; j < dim; j++) {
                double dj = v[vOff + j] - mu[j];
                sigma[i * dim + j] = oneMinusGamma * sigma[i * dim + j] + gamma * di * dj;
            }
        }

        // Save current L before attempting decomposition
        System.arraycopy(L, 0, Lsave, 0, dim * dim);

        // Attempt Cholesky decomposition of sigma into L.
        // If it fails, restore L from the saved copy.
        if (!choleskyDecompose(sigma, L)) {
            System.arraycopy(Lsave, 0, L, 0, dim * dim);
        }
    }

    /**
     * Returns the current lower-triangular Cholesky factor L such that
     * {@code L·Lᵀ = Σ}.
     *
     * <p>The returned array reference is the internal buffer — the
     * caller must not modify it.
     *
     * @return dim×dim row-major lower-triangular Cholesky factor
     */
    public double[] choleskyL() {
        return L;
    }

    /**
     * Returns the current running mean.
     *
     * <p>The returned array reference is the internal buffer — the
     * caller must not modify it.
     *
     * @return length-dim mean vector
     */
    public double[] mean() {
        return mu;
    }

    /**
     * Returns the current covariance estimate.
     *
     * <p>The returned array reference is the internal buffer — the
     * caller must not modify it.
     *
     * @return dim×dim row-major covariance matrix
     */
    public double[] covariance() {
        return sigma;
    }

    /**
     * In-place Cholesky decomposition of a symmetric positive-definite
     * matrix into a lower-triangular factor.
     *
     * <p>If the matrix is not positive definite (a diagonal element
     * becomes non-positive during decomposition), the method returns
     * {@code false} and the contents of {@code dst} are indeterminate.
     *
     * <p>Both {@code src} and {@code dst} are dim×dim row-major. Only
     * the lower triangle of {@code dst} is written; upper elements are
     * set to zero.
     *
     * <p>For {@code dim >= 8}, delegates to {@link BLAS#dpotrf} which is
     * significantly faster than the hand-written loop for larger dimensions.
     *
     * @param src the symmetric PD matrix to decompose (not modified)
     * @param dst output buffer for the lower-triangular factor
     * @return {@code true} if decomposition succeeded
     */
    private boolean choleskyDecompose(double[] src, double[] dst) {
        if (dim >= 8) {
            // Use BLAS dpotrf for larger dimensions (row-major, lower triangle)
            System.arraycopy(src, 0, dst, 0, dim * dim);
            int info = BLAS.dpotrf(BLAS.Uplo.Lower, dim, dst, 0, dim);
            if (info != 0) return false;
            // Zero upper triangle (dpotrf leaves it dirty)
            for (int i = 0; i < dim; i++)
                for (int j = i + 1; j < dim; j++)
                    dst[i * dim + j] = 0.0;
            return true;
        }

        // Hand-written Cholesky for small dimensions (d < 8)
        Arrays.fill(dst, 0.0);

        for (int i = 0; i < dim; i++) {
            for (int j = 0; j <= i; j++) {
                double sum = src[i * dim + j];
                for (int k = 0; k < j; k++) {
                    sum -= dst[i * dim + k] * dst[j * dim + k];
                }
                if (i == j) {
                    if (sum <= 0.0) {
                        return false;
                    }
                    dst[i * dim + j] = Math.sqrt(sum);
                } else {
                    dst[i * dim + j] = sum / dst[j * dim + j];
                }
            }
        }
        return true;
    }
}
