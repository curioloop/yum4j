/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.linalg.reg;

import com.curioloop.yum4j.linalg.blas.BLAS;
import com.curioloop.yum4j.math.VectorOps;

import static java.lang.Math.*;

/**
 * Generalized Least Squares (GLS) linear regression.
 *
 * <p>Solves the linear model with correlated/heteroskedastic errors:
 * <pre>  y = Xβ + ε,  Var(ε) = σ²·Σ</pre>
 * where Σ is a known positive-definite covariance structure.
 *
 * <p>GLS whitens the system via Cholesky decomposition Σ = LLᵀ:
 * <pre>  ỹ = L⁻¹y,  X̃ = L⁻¹X</pre>
 * then delegates to the OLS solver on (ỹ, X̃).
 *
 * <p>Sigma can be specified as:
 * <ul>
 *   <li>Scalar (length 1): uniform variance σ²·I</li>
 *   <li>Diagonal vector (length n): heteroskedastic but uncorrelated</li>
 *   <li>Full matrix (length n×n): arbitrary correlation structure
 *       — <b>overwritten in-place</b> with the Cholesky factor L</li>
 * </ul>
 *
 * <p>Data layout: same as {@link OLS} — X is row-major n×k.
 * <b>X and sigma (full-matrix case) are overwritten in-place.</b> y is never modified.
 */
public class GLS extends OLS {

    /** Sigma structure type. */
    public enum Sigma { SCALAR, DIAGONAL, FULL }

    private final double[] sigma; // after fit: holds L (Cholesky factor) for FULL case
    private final Sigma sigmaType;
    private double cholLogDet;

    // =========================================================================
    // Pool
    // =========================================================================

    /**
     * Reusable workspace for GLS computations.
     */
    public static class Pool extends OLS.Pool {
        public double[] yWhiten; // ỹ = L⁻¹y
        public double[] iota;    // L⁻¹1 for centered TSS

        public Pool() {}

        public Pool ensureWhiten(int n) {
            if (yWhiten == null || yWhiten.length < n) yWhiten = new double[n];
            return this;
        }

        public Pool ensureIota(int n) {
            if (iota == null || iota.length < n) iota = new double[n];
            return this;
        }
    }

    // =========================================================================
    // Constructors
    // =========================================================================

    public GLS(double[] y, double[] X, double[] sigma, int n, int k, boolean useQR, int kConst) {
        this(y, X, sigma, n, k, useQR, kConst, OUTPUT_FULL);
    }

    public GLS(double[] y, double[] X, double[] sigma, int n, int k, boolean useQR, int kConst, int outputMask) {
        super(y, X, n, k, useQR, kConst, outputMask);
        if (sigma == null) throw new NullPointerException("sigma must not be null");
        this.sigma = sigma;
        int len = sigma.length;
        if (len == 1) {
            if (sigma[0] <= 0) throw new IllegalArgumentException("Scalar sigma must be positive");
            sigmaType = Sigma.SCALAR;
        } else if (len == n) {
            for (int i = 0; i < n; i++) {
                if (sigma[i] <= 0) throw new IllegalArgumentException("All diagonal sigma elements must be positive");
            }
            sigmaType = Sigma.DIAGONAL;
        } else if (len == n * n) {
            sigmaType = Sigma.FULL;
        } else {
            throw new IllegalArgumentException("Sigma length must be 1, n, or n*n (got " + len + " for n=" + n + ")");
        }
    }

    /**
     * Total sum of squares for GLS: (y - μ̂)ᵀ Σ⁻¹ (y - μ̂).
     *
     * <p>Uses the Pool-backed whitened y and a Pool-backed whitened ones vector
     * to compute the GLS-weighted mean, matching statsmodels exactly.</p>
     */
    @Override
    public double tss(boolean centered) {
        double[] yWhiten = pool().yWhiten;
        if (!centered) {
            return BLAS.ddot(nObs, yWhiten, 0, 1, yWhiten, 0, 1);
        }
        // Compute whitened ones vector: ĩota = L⁻¹·1
        double[] iota = pool().ensureIota(nObs).iota;
        switch (sigmaType) {
            case SCALAR -> {
                double invSqrt = 1.0 / sqrt(sigma[0]);
                java.util.Arrays.fill(iota, invSqrt);
            }
            case DIAGONAL -> {
                for (int i = 0; i < nObs; i++) iota[i] = 1.0 / sqrt(sigma[i]);
            }
            case FULL -> {
                // sigma now holds L (Cholesky factor) after fit
                java.util.Arrays.fill(iota, 1.0);
                BLAS.dtrsm(BLAS.Side.Left, BLAS.Uplo.Lower, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit,
                           nObs, 1, 1.0, sigma, 0, nObs, iota, 0, 1);
            }
        }
        // GLS-weighted mean: μ̂ = (ỹ · ĩota) / (ĩota · ĩota)
        double iotaDotIota = BLAS.ddot(nObs, iota, 0, 1, iota, 0, 1);
        double yDotIota = BLAS.ddot(nObs, yWhiten, 0, 1, iota, 0, 1);
        double meanGls = yDotIota / iotaDotIota;
        // TSS = Σ(ỹᵢ - μ̂ · ĩotaᵢ)²
        double s = 0;
        for (int i = 0; i < nObs; i++) {
            double d = yWhiten[i] - meanGls * iota[i];
            s += d * d;
        }
        return s;
    }

    @Override
    public GLS fit() {
        return fit(new Pool());
    }

    @Override
    public GLS.Pool alloc() {
        return new GLS.Pool();
    }

    @Override
    public GLS fit(OLS.Pool ws) {
        Pool p = (ws instanceof Pool pp) ? pp : new Pool();
        pool = p;
        switch (sigmaType) {
            case SCALAR   -> fitScalar(p);
            case DIAGONAL -> fitDiagonal(p);
            case FULL     -> fitFull(p);
        }
        return this;
    }

    public GLS fit(Pool ws) {
        return fit((OLS.Pool) ws);
    }

    @Override
    public GLS.Pool pool() {
        return (GLS.Pool) pool;
    }

    // =========================================================================
    // Whitening paths
    // =========================================================================

    private void fitScalar(Pool ws) {
        if (needsParams()) ws.ensureParams(nParams);
        if (needsFitness()) ws.ensureFitness(nObs);
        if (needsCovariance()) ws.ensureCovariance(nParams);
        if (kConst < 0) kConst = detectConst(X, nObs, nParams, ws);
        cholLogDet = nObs * log(sigma[0]);
        double invSqrt = 1.0 / sqrt(sigma[0]);
        double[] yWhiten = ws.ensureWhiten(nObs).yWhiten;
        VectorOps.scalTo(yWhiten, 0, invSqrt, y, 0, nObs);
        BLAS.dscal(nObs * nParams, invSqrt, X, 0, 1);
        if (useQR) {
            if (needsFitness()) solveQR(yWhiten, X, ws);
            else                solveQRParamsOnly(yWhiten, X, ws);
        } else {
            if (needsFitness()) solveSVD(yWhiten, X, ws);
            else                solveSVDParamsOnly(yWhiten, X, ws);
        }
    }

    private void fitDiagonal(Pool ws) {
        if (needsParams()) ws.ensureParams(nParams);
        if (needsFitness()) ws.ensureFitness(nObs);
        if (needsCovariance()) ws.ensureCovariance(nParams);
        if (kConst < 0) kConst = detectConst(X, nObs, nParams, ws);
        cholLogDet = 0.0;
        double[] yWhiten = ws.ensureWhiten(nObs).yWhiten;
        for (int i = 0; i < nObs; i++) {
            double invSqrt = 1.0 / sqrt(sigma[i]);
            cholLogDet += log(sigma[i]);
            yWhiten[i] = y[i] * invSqrt;
            for (int j = 0; j < nParams; j++) X[i * nParams + j] *= invSqrt;
        }
        if (useQR) {
            if (needsFitness()) solveQR(yWhiten, X, ws);
            else                solveQRParamsOnly(yWhiten, X, ws);
        } else {
            if (needsFitness()) solveSVD(yWhiten, X, ws);
            else                solveSVDParamsOnly(yWhiten, X, ws);
        }
    }

    private void fitFull(Pool ws) {
        if (needsParams()) ws.ensureParams(nParams);
        if (needsFitness()) ws.ensureFitness(nObs);
        if (needsCovariance()) ws.ensureCovariance(nParams);
        int n = nObs, k = nParams;

        // Detect constant BEFORE whitening
        if (kConst < 0) kConst = detectConst(X, n, k, ws);

        // Cholesky factorization Σ = LLᵀ — directly on sigma (overwrites it with L)
        int info = BLAS.dpotrf(BLAS.Uplo.Lower, n, sigma, 0, n);
        if (info > 0) throw new IllegalArgumentException(
                "Sigma must be symmetric positive-definite (dpotrf info=" + info + ")");

        // log|Σ| = 2·Σ log(Lᵢᵢ)
        cholLogDet = 0.0;
        for (int i = 0; i < n; i++) cholLogDet += 2.0 * log(sigma[i * n + i]);

        // ỹ = L⁻¹y (y is NOT modified)
        double[] yWhiten = ws.ensureWhiten(n).yWhiten;
        System.arraycopy(y, 0, yWhiten, 0, n);
        BLAS.dtrsm(BLAS.Side.Left, BLAS.Uplo.Lower, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit,
                   n, 1, 1.0, sigma, 0, n, yWhiten, 0, 1);

        // X̃ = L⁻¹X (in-place)
        BLAS.dtrsm(BLAS.Side.Left, BLAS.Uplo.Lower, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit,
                   n, k, 1.0, sigma, 0, n, X, 0, k);

        if (useQR) {
            if (needsFitness()) solveQR(yWhiten, X, ws);
            else                solveQRParamsOnly(yWhiten, X, ws);
        } else {
            if (needsFitness()) solveSVD(yWhiten, X, ws);
            else                solveSVDParamsOnly(yWhiten, X, ws);
        }
    }

    // =========================================================================
    // Log-likelihood override
    // =========================================================================

    @Override
    public double logLike() {
        return super.logLike() - 0.5 * cholLogDet;
    }
}
