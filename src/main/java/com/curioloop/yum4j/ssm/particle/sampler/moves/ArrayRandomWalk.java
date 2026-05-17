package com.curioloop.yum4j.ssm.particle.sampler.moves;
import com.curioloop.yum4j.ssm.particle.sampler.ThetaParticles;

import java.util.Arrays;
import java.util.function.Consumer;
import java.util.random.RandomGenerator;

/**
 * Gaussian random-walk Metropolis kernel operating on all N particles
 * simultaneously.
 *
 * <p>At calibration time, computes the weighted covariance of the particle
 * cloud and stores its Cholesky factor scaled by {@code 2.38 / √d} (the
 * optimal scaling for Gaussian targets in dimension d). At step time,
 * proposes {@code θ* = θ + L·z} where {@code z ~ N(0, I_d)} and
 * accepts/rejects each particle independently via the Metropolis ratio.
 *
 * <p>The implementation is allocation-free after the first call to
 * {@link #calibrate}: all internal buffers (proposal particles, Cholesky
 * factor, scratch vectors) are pre-allocated and reused across steps.
 *
 * <p>Hand-written Cholesky decomposition supports {@code d ≤ 20}; for
 * larger dimensions an external BLAS path would be needed (not yet
 * implemented).
 *
 * @see ArrayMCMC
 */
public final class ArrayRandomWalk implements ArrayMCMC {

    /** Lower-triangular Cholesky factor L (dim×dim, row-major). */
    private double[] L;

    /** Parameter dimension. */
    private int dim;

    /** Pre-allocated proposal ThetaParticles (reused across steps). */
    private ThetaParticles xProp;

    /** Scratch vector for z ~ N(0, I_d) during proposal generation. */
    private double[] zBuf;

    /** Pre-allocated weighted mean buffer (length dim). */
    private double[] mu;

    /** Pre-allocated scaled covariance buffer (length dim*dim). */
    private double[] scaledCov;

    /**
     * {@inheritDoc}
     *
     * <p>Computes the weighted mean μ and weighted covariance Σ of
     * {@code x.arena[0..dim*N)} (theta region), then stores the Cholesky
     * factor of {@code (2.38/√d)² · Σ}.
     *
     * <p>Weighted mean: {@code μ[j] = Σ_n W[n] * arena[j*N + n]}.
     * <p>Weighted covariance:
     * {@code Σ[i*dim+j] = Σ_n W[n] * (arena[i*N+n] - μ[i]) * (arena[j*N+n] - μ[j])}.
     */
    @Override
    public void calibrate(double[] W, ThetaParticles x) {
        int N = x.N;
        int d = x.dim;
        this.dim = d;
        double[] arena = x.arena;

        // Allocate or resize internal buffers if needed
        if (L == null || L.length < d * d) {
            L = new double[d * d];
        }
        if (zBuf == null || zBuf.length < d) {
            zBuf = new double[d];
        }
        if (xProp == null || xProp.N != N || xProp.dim != d) {
            xProp = ThetaParticles.allocate(d, N);
        }
        if (mu == null || mu.length < d) {
            mu = new double[d];
        }
        if (scaledCov == null || scaledCov.length < d * d) {
            scaledCov = new double[d * d];
        }

        // Compute weighted mean μ
        Arrays.fill(mu, 0, d, 0.0);
        for (int j = 0; j < d; j++) {
            double sum = 0.0;
            int base = j * N;
            for (int n = 0; n < N; n++) {
                sum += W[n] * arena[base + n];
            }
            mu[j] = sum;
        }

        // Compute weighted covariance Σ (symmetric, store full d×d row-major)
        // Then scale by (2.38/√d)² and Cholesky-factor in one pass.
        double scale = 2.38 / Math.sqrt(d);
        double scale2 = scale * scale;

        // We compute scale² · Σ directly into the pre-allocated scaledCov buffer, then Cholesky
        Arrays.fill(scaledCov, 0, d * d, 0.0);
        for (int i = 0; i < d; i++) {
            int baseI = i * N;
            for (int j = 0; j <= i; j++) {
                int baseJ = j * N;
                double sum = 0.0;
                for (int n = 0; n < N; n++) {
                    sum += W[n] * (arena[baseI + n] - mu[i]) * (arena[baseJ + n] - mu[j]);
                }
                double val = scale2 * sum;
                scaledCov[i * d + j] = val;
                scaledCov[j * d + i] = val; // symmetric
            }
        }

        // Cholesky decomposition of scaledCov into L (row-major, lower-triangular)
        if (!choleskyDecompose(scaledCov, L, d)) {
            // Fallback: if covariance is not PD (degenerate sample),
            // use scale * identity as the Cholesky factor
            Arrays.fill(L, 0, d * d, 0.0);
            for (int i = 0; i < d; i++) {
                L[i * d + i] = scale;
            }
        }
    }

    /**
     * {@inheritDoc}
     *
     * <p>For each particle n:
     * <ol>
     *   <li>Draw {@code z[j] ~ N(0,1)} for j = 0..dim-1</li>
     *   <li>Compute proposal {@code θ*[j] = θ[j*N+n] + Σ_k L[j*dim+k] * z[k]}</li>
     *   <li>Write proposal into pre-allocated {@code xProp}</li>
     * </ol>
     * Then evaluate the target density on all proposals at once, and
     * accept/reject each particle independently:
     * accept if {@code log(U) < xProp.arena[lpostOff+n] - x.arena[lpostOff+n]}.
     *
     * <p>No allocation occurs in this method.
     */
    @Override
    public double step(ThetaParticles x, Consumer<ThetaParticles> target,
                       RandomGenerator g, double[] scratch) {
        int N = x.N;
        int d = dim;
        double[] arena = x.arena;
        double[] propArena = xProp.arena;

        // Generate proposals for all particles
        for (int n = 0; n < N; n++) {
            // Draw z ~ N(0, I_d)
            for (int j = 0; j < d; j++) {
                zBuf[j] = g.nextGaussian();
            }
            // Compute θ*[j] = θ[j*N+n] + Σ_k L[j*d+k] * z[k]
            for (int j = 0; j < d; j++) {
                double shift = 0.0;
                int rowOff = j * d;
                for (int k = 0; k <= j; k++) { // L is lower-triangular
                    shift += L[rowOff + k] * zBuf[k];
                }
                propArena[j * N + n] = arena[j * N + n] + shift;
            }
        }

        // Evaluate target density on all proposals (fills xProp lpost)
        target.accept(xProp);

        // Accept/reject each particle independently
        int accepted = 0;
        int lpostOff = x.lpostOff();
        int llikOff = x.llikOff();
        int lpriorOff = x.lpriorOff();
        int propLpostOff = xProp.lpostOff();
        int propLlikOff = xProp.llikOff();
        int propLpriorOff = xProp.lpriorOff();
        for (int n = 0; n < N; n++) {
            double logAlpha = propArena[propLpostOff + n] - arena[lpostOff + n];
            if (Math.log(g.nextDouble()) < logAlpha) {
                // Accept: copy proposal into x
                for (int j = 0; j < d; j++) {
                    arena[j * N + n] = propArena[j * N + n];
                }
                arena[lpostOff + n] = propArena[propLpostOff + n];
                arena[llikOff + n] = propArena[propLlikOff + n];
                arena[lpriorOff + n] = propArena[propLpriorOff + n];
                accepted++;
            }
        }

        return (double) accepted / N;
    }

    /**
     * In-place Cholesky decomposition of a symmetric positive-definite
     * matrix into a lower-triangular factor.
     *
     * <p>Both {@code src} and {@code dst} are dim×dim row-major. Only
     * the lower triangle of {@code dst} is written; upper elements are
     * set to zero.
     *
     * @param src the symmetric PD matrix to decompose (not modified)
     * @param dst output buffer for the lower-triangular factor
     * @param d   matrix dimension
     * @return {@code true} if decomposition succeeded
     */
    private static boolean choleskyDecompose(double[] src, double[] dst, int d) {
        Arrays.fill(dst, 0, d * d, 0.0);

        for (int i = 0; i < d; i++) {
            for (int j = 0; j <= i; j++) {
                double sum = src[i * d + j];
                for (int k = 0; k < j; k++) {
                    sum -= dst[i * d + k] * dst[j * d + k];
                }
                if (i == j) {
                    if (sum <= 0.0) {
                        return false;
                    }
                    dst[i * d + j] = Math.sqrt(sum);
                } else {
                    dst[i * d + j] = sum / dst[j * d + j];
                }
            }
        }
        return true;
    }
}
