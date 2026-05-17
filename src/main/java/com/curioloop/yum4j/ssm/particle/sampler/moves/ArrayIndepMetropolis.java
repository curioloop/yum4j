package com.curioloop.yum4j.ssm.particle.sampler.moves;
import com.curioloop.yum4j.ssm.particle.sampler.ThetaParticles;

import com.curioloop.yum4j.linalg.blas.BLAS;

import java.util.Arrays;
import java.util.function.Consumer;
import java.util.random.RandomGenerator;

/**
 * Independent Metropolis kernel operating on all N particles simultaneously.
 *
 * <p>At calibration time, computes the weighted mean μ and weighted covariance
 * Σ of the particle cloud, then stores the Cholesky factor of
 * {@code scale² · Σ}. At step time, proposes {@code θ* = μ + L·z} where
 * {@code z ~ N(0, I_d)} — note the proposal is centred at μ, NOT at the
 * current θ — and accepts/rejects each particle independently via the
 * Metropolis-Hastings ratio including the proposal density ratio.
 *
 * <p>The acceptance probability is:
 * <pre>
 *   log α = [lpost(θ*) + logQ(θ)] − [lpost(θ) + logQ(θ*)]
 * </pre>
 * where {@code logQ(v)} is the log-density of the proposal distribution
 * evaluated at v. Since the normalising constants cancel, the ratio simplifies
 * to:
 * <pre>
 *   log α = lpost(θ*) − lpost(θ) + 0.5 * (‖L⁻¹(θ* − μ)‖² − ‖L⁻¹(θ − μ)‖²)
 * </pre>
 *
 * <p>The implementation is allocation-free after the first call to
 * {@link #calibrate}: all internal buffers are pre-allocated and reused.
 *
 * @see ArrayMCMC
 */
public final class ArrayIndepMetropolis implements ArrayMCMC {

    /** User-supplied scale multiplier on the covariance. */
    private final double scale;

    /** Lower-triangular Cholesky factor L (dim×dim, row-major). */
    private double[] L;

    /** Weighted mean μ (length dim). */
    private double[] mu;

    /** Parameter dimension. */
    private int dim;

    /** Pre-allocated proposal ThetaParticles (reused across steps). */
    private ThetaParticles xProp;

    /** Scratch vector for z ~ N(0, I_d) during proposal generation. */
    private double[] zBuf;

    /** Scratch vector for forward substitution (length dim). */
    private double[] linvBuf;

    /**
     * Creates an independent Metropolis kernel with the given scale factor.
     *
     * @param scale multiplier applied to the weighted covariance before
     *              Cholesky factorisation; controls the spread of the
     *              independent proposal relative to the particle cloud
     */
    public ArrayIndepMetropolis(double scale) {
        this.scale = scale;
    }

    /**
     * {@inheritDoc}
     *
     * <p>Computes the weighted mean μ and weighted covariance Σ of
     * the theta region of {@code x.arena}, then stores the Cholesky factor of
     * {@code scale² · Σ}. Also stores μ for use in {@link #step}.
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
        if (mu == null || mu.length < d) {
            mu = new double[d];
        }
        if (zBuf == null || zBuf.length < d) {
            zBuf = new double[d];
        }
        if (linvBuf == null || linvBuf.length < d) {
            linvBuf = new double[d];
        }
        if (xProp == null || xProp.N != N || xProp.dim != d) {
            xProp = ThetaParticles.allocate(d, N);
        }

        // Compute weighted mean μ
        for (int j = 0; j < d; j++) {
            double sum = 0.0;
            int base = j * N;
            for (int n = 0; n < N; n++) {
                sum += W[n] * arena[base + n];
            }
            mu[j] = sum;
        }

        // Compute weighted covariance Σ scaled by scale²
        double scale2 = scale * scale;
        double[] scaledCov = new double[d * d]; // temporary for calibration
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
     *   <li>Compute proposal {@code θ*[j] = μ[j] + Σ_k L[j*dim+k] * z[k]}</li>
     *   <li>Compute {@code ‖L⁻¹(θ* − μ)‖²} and {@code ‖L⁻¹(θ − μ)‖²}
     *       via forward substitution</li>
     *   <li>Accept if {@code log(U) < lpost(θ*) − lpost(θ)
     *       + 0.5 * (‖L⁻¹(θ* − μ)‖² − ‖L⁻¹(θ − μ)‖²)}</li>
     * </ol>
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

        // Generate proposals for all particles and compute ‖L⁻¹(θ* − μ)‖²
        // We store ‖L⁻¹(θ* − μ)‖² in scratch[n] for later use
        for (int n = 0; n < N; n++) {
            // Draw z ~ N(0, I_d)
            for (int j = 0; j < d; j++) {
                zBuf[j] = g.nextGaussian();
            }
            // Compute θ*[j] = μ[j] + Σ_k L[j*d+k] * z[k]
            for (int j = 0; j < d; j++) {
                double shift = 0.0;
                int rowOff = j * d;
                for (int k = 0; k <= j; k++) { // L is lower-triangular
                    shift += L[rowOff + k] * zBuf[k];
                }
                propArena[j * N + n] = mu[j] + shift;
            }

            // ‖L⁻¹(θ* − μ)‖² = ‖z‖² (since θ* − μ = L·z, so L⁻¹(θ* − μ) = z)
            double normSqProp = 0.0;
            for (int j = 0; j < d; j++) {
                normSqProp += zBuf[j] * zBuf[j];
            }
            scratch[n] = normSqProp;
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
            // Compute ‖L⁻¹(θ − μ)‖² for current particle via forward substitution
            // Solve L·w = (θ − μ), then ‖w‖²
            double normSqCurr = forwardSubNormSq(arena, n, N, d);

            double normSqProp = scratch[n];

            // log α = lpost(θ*) − lpost(θ) + 0.5 * (normSqProp − normSqCurr)
            // Wait — the formula is:
            // log α = lpost(θ*) + logQ(θ) − lpost(θ) − logQ(θ*)
            // logQ(v) = −0.5 * ‖L⁻¹(v − μ)‖² − const
            // logQ(θ) − logQ(θ*) = −0.5 * normSqCurr + 0.5 * normSqProp
            //                     = 0.5 * (normSqProp − normSqCurr)
            double logAlpha = propArena[propLpostOff + n] - arena[lpostOff + n]
                    + 0.5 * (normSqProp - normSqCurr);

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
     * Computes {@code ‖L⁻¹(θ_n − μ)‖²} via forward substitution.
     *
     * <p>Solves {@code L·w = (θ_n − μ)} for w, then returns {@code ‖w‖²}.
     * Uses the pre-allocated {@code linvBuf} as scratch.
     *
     * <p>For {@code d >= 8}, delegates to {@link BLAS#dtrsl} and
     * {@link BLAS#ddot} which are faster for larger dimensions.
     *
     * @param arena  the arena buffer (theta at offset 0, column-major)
     * @param n      particle index
     * @param N      number of particles
     * @param d      parameter dimension
     * @return the squared norm ‖L⁻¹(θ_n − μ)‖²
     */
    private double forwardSubNormSq(double[] arena, int n, int N, int d) {
        // Build RHS: linvBuf[j] = arena[j*N + n] - mu[j]
        for (int j = 0; j < d; j++) {
            linvBuf[j] = arena[j * N + n] - mu[j];
        }

        if (d >= 8) {
            // Use BLAS triangular solve + dot product.
            // L is row-major lower-triangular. dtrsl expects column-major.
            // Row-major L interpreted as column-major is L^T (upper-triangular).
            // To solve L*w = b, call dtrsl with Uplo.Upper, Trans.Trans.
            BLAS.dtrsl(L, 0, d, d, linvBuf, 0, BLAS.Uplo.Upper, BLAS.Trans.Trans);
            return BLAS.ddot(d, linvBuf, 0, 1, linvBuf, 0, 1);
        }

        // Hand-written forward substitution for small dimensions (d < 8)
        double normSq = 0.0;
        for (int i = 0; i < d; i++) {
            double rhs = linvBuf[i];
            int rowOff = i * d;
            for (int k = 0; k < i; k++) {
                rhs -= L[rowOff + k] * linvBuf[k];
            }
            double wi = rhs / L[rowOff + i];
            linvBuf[i] = wi;
            normSq += wi * wi;
        }
        return normSq;
    }

    /**
     * In-place Cholesky decomposition of a symmetric positive-definite
     * matrix into a lower-triangular factor.
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
