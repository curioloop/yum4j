/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.linalg;

import com.curioloop.yum4j.linalg.reg.GLS;
import com.curioloop.yum4j.linalg.reg.OLS;
import com.curioloop.yum4j.linalg.reg.WLS;

/**
 * Facade for ordinary, weighted, and generalized least squares regression.
 *
 * <p>Solves the linear model y = Xβ + ε, with optional per-observation weights.
 *
 * <pre>{@code
 * // OLS with SVD solver (X is overwritten in-place; clone if needed)
 * OLS ols = Regressor.ols(y, X, n, k, Regressor.Opts.PINV, Regressor.Opts.INFERENCE);
 *
 * // OLS with QR solver
 * OLS ols = Regressor.ols(y, X, n, k, Regressor.Opts.QR, Regressor.Opts.INFERENCE);
 *
 * // WLS (X is overwritten in-place; y is never modified)
 * WLS wls = Regressor.wls(y, X, weights, n, k, Regressor.Opts.PINV, Regressor.Opts.INFERENCE);
 *
 * // GLS (X is overwritten in-place; full sigma is overwritten by Cholesky)
 * GLS gls = Regressor.gls(y, X, sigma, n, k, Regressor.Opts.PINV, Regressor.Opts.INFERENCE);
 *
 * // Workspace reuse across multiple fits
 * OLS.Pool ws = new OLS.Pool();
 * for (double[] Xi : series) {
 *     OLS r = Regressor.ols(y, Xi.clone(), n, k, ws, Regressor.Opts.PINV, Regressor.Opts.INFERENCE);
 * }
 * }</pre>
 *
 * <p>Data layout: X is row-major n×k, each row is one observation.
 * <b>OLS, WLS, and GLS overwrite X in-place</b>; y is never modified.
 * WLS and GLS keep the whitened endogenous vector in reusable Pool buffers.
 */
public final class Regressor {

    private Regressor() {}

    // =========================================================================
    // Opts
    // =========================================================================

    /** Algorithm options for least squares fitting. */
    public enum Opts {
        /** Use QR factorization (faster when X is full rank). */
        QR,
        /** Use SVD/pinv solver (numerically robust, handles rank-deficient X). */
        PINV,
        /**
         * Declare that X contains a constant column (intercept); sets kConst = 1.
         * Skips automatic detection. Equivalent to statsmodels hasconst=True.
         */
        HAS_CONST,
        /**
         * Declare that X has no constant column; sets kConst = 0.
         * Skips automatic detection. Equivalent to statsmodels hasconst=False.
         */
        NO_CONST,
        /** Request parameter estimates. */
        PARAMS,
        /** Request fitted values, residuals, and fit-quality statistics. */
        FITNESS,
        /** Request parameter estimates plus fitted values, residuals, and fit-quality statistics. */
        ESTIMATION,
        /** Request estimation outputs plus covariance and inference diagnostics. */
        INFERENCE
    }

    // =========================================================================
    // OLS
    // =========================================================================

    /**
     * Fits an ordinary least squares model.
     *
     * <p><b>Warning:</b> X is overwritten in-place. Clone X before calling if
     * the original data must be preserved.
     *
     * @param y    endogenous vector (length >= n, not modified)
     * @param X    exogenous matrix, row-major n×k (<b>overwritten in-place</b>)
     * @param n    number of observations
     * @param k    number of regressors
     * @param opts one or more {@link Opts} values; {@code QR} or {@code PINV} is required, plus exactly one output opt
     * @return fitted OLS result
     */
    public static OLS ols(double[] y, double[] X, int n, int k, Opts... opts) {
        return ols(y, X, n, k, (OLS.Pool) null, opts);
    }

    /**
     * Fits an ordinary least squares model with workspace reuse.
     *
     * @param y    endogenous vector (length >= n, not modified)
     * @param X    exogenous matrix, row-major n×k (<b>overwritten in-place</b>)
     * @param n    number of observations
     * @param k    number of regressors
     * @param ws   reusable workspace (may be null; {@link Pool} is accepted)
     * @param opts one or more {@link Opts} values; {@code QR} or {@code PINV} is required, plus exactly one output opt
     * @return fitted OLS result
     */
    public static OLS ols(double[] y, double[] X, int n, int k, OLS.Pool ws, Opts... opts) {
        boolean hasQR = contains(opts, Opts.QR), hasPINV = contains(opts, Opts.PINV);
        if (!hasQR && !hasPINV) throw new IllegalArgumentException("Must specify Opts.QR or Opts.PINV");
        if (hasQR && hasPINV)   throw new IllegalArgumentException("Cannot specify both Opts.QR and Opts.PINV");
        int outputMask = requiredOutputMask(opts);
        int kConst = contains(opts, Opts.HAS_CONST) ? 1 : contains(opts, Opts.NO_CONST) ? 0 : -1;
        if (ws == null) ws = new OLS.Pool();
        return new OLS(y, X, n, k, hasQR, kConst, outputMask).fit(ws);
    }

    // =========================================================================
    // WLS
    // =========================================================================

    /**
     * Fits a weighted least squares model.
     *
     * @param y       endogenous vector (length >= n, not modified)
     * @param X       exogenous matrix, row-major n×k (<b>overwritten in-place</b>)
     * @param weights observation weights (length >= n, all positive, not modified)
     * @param n       number of observations
     * @param k       number of regressors
     * @param opts    one or more {@link Opts} values; {@code QR} or {@code PINV} is required, plus exactly one output opt
     * @return fitted WLS result
     */
    public static WLS wls(double[] y, double[] X, double[] weights, int n, int k, Opts... opts) {
        return wls(y, X, weights, n, k, (WLS.Pool) null, opts);
    }

    /**
     * Fits a weighted least squares model with workspace reuse.
     *
     * @param y       endogenous vector (length >= n, not modified)
     * @param X       exogenous matrix, row-major n×k (<b>overwritten in-place</b>)
     * @param weights observation weights (length >= n, all positive, not modified)
     * @param n       number of observations
     * @param k       number of regressors
     * @param ws      reusable workspace (may be null; {@link Pool} is accepted)
     * @param opts    one or more {@link Opts} values; {@code QR} or {@code PINV} is required, plus exactly one output opt
     * @return fitted WLS result
     */
    public static WLS wls(double[] y, double[] X, double[] weights, int n, int k, WLS.Pool ws, Opts... opts) {
        boolean hasQR = contains(opts, Opts.QR), hasPINV = contains(opts, Opts.PINV);
        if (!hasQR && !hasPINV) throw new IllegalArgumentException("Must specify Opts.QR or Opts.PINV");
        if (hasQR && hasPINV)   throw new IllegalArgumentException("Cannot specify both Opts.QR and Opts.PINV");
        int outputMask = requiredOutputMask(opts);
        int kConst = contains(opts, Opts.HAS_CONST) ? 1 : contains(opts, Opts.NO_CONST) ? 0 : -1;
        if (ws == null) ws = new WLS.Pool();
        return new WLS(y, X, weights, n, k, hasQR, kConst, outputMask).fit(ws);
    }

    // =========================================================================
    // GLS
    // =========================================================================

    /**
     * Fits a generalized least squares model.
     *
     * <p>Sigma specifies the error covariance structure:
     * <ul>
     *   <li>Length 1: scalar (uniform variance)</li>
     *   <li>Length n: diagonal (heteroskedastic, uncorrelated)</li>
     *   <li>Length n×n: full covariance matrix (correlated errors)</li>
     * </ul>
     *
     * <p><b>Warning:</b> X is overwritten in-place; full-matrix sigma is overwritten by its Cholesky factor.
     *
     * @param y     endogenous vector (length >= n, not modified)
     * @param X     exogenous matrix, row-major n×k (<b>overwritten in-place</b>)
     * @param sigma error covariance structure (length 1, n, or n×n)
     * @param n     number of observations
     * @param k     number of regressors
     * @param opts  one or more {@link Opts} values; {@code QR} or {@code PINV} is required, plus exactly one output opt
     * @return fitted GLS result
     */
    public static GLS gls(double[] y, double[] X, double[] sigma, int n, int k, Opts... opts) {
        return gls(y, X, sigma, n, k, (GLS.Pool) null, opts);
    }

    /**
     * Fits a generalized least squares model with workspace reuse.
     *
     * @param y     endogenous vector (length >= n, not modified)
     * @param X     exogenous matrix, row-major n×k (<b>overwritten in-place</b>)
     * @param sigma error covariance structure (length 1, n, or n×n)
     * @param n     number of observations
     * @param k     number of regressors
     * @param ws    reusable workspace (may be null)
     * @param opts  one or more {@link Opts} values; {@code QR} or {@code PINV} is required, plus exactly one output opt
     * @return fitted GLS result
     */
    public static GLS gls(double[] y, double[] X, double[] sigma, int n, int k, GLS.Pool ws, Opts... opts) {
        boolean hasQR = contains(opts, Opts.QR), hasPINV = contains(opts, Opts.PINV);
        if (!hasQR && !hasPINV) throw new IllegalArgumentException("Must specify Opts.QR or Opts.PINV");
        if (hasQR && hasPINV)   throw new IllegalArgumentException("Cannot specify both Opts.QR and Opts.PINV");
        int outputMask = requiredOutputMask(opts);
        int kConst = contains(opts, Opts.HAS_CONST) ? 1 : contains(opts, Opts.NO_CONST) ? 0 : -1;
        if (ws == null) ws = new GLS.Pool();
        return new GLS(y, X, sigma, n, k, hasQR, kConst, outputMask).fit(ws);
    }

    // =========================================================================
    // Helper
    // =========================================================================

    private static boolean contains(Opts[] opts, Opts target) {
        if (opts == null) return false;
        for (Opts o : opts) if (o == target) return true;
        return false;
    }

    private static int outputOptCount(Opts[] opts) {
        int count = 0;
        if (contains(opts, Opts.PARAMS)) count++;
        if (contains(opts, Opts.FITNESS)) count++;
        if (contains(opts, Opts.ESTIMATION)) count++;
        if (contains(opts, Opts.INFERENCE)) count++;
        return count;
    }

    private static int requiredOutputMask(Opts[] opts) {
        int count = outputOptCount(opts);
        if (count == 0) {
            throw new IllegalArgumentException("Must specify one of Opts.PARAMS, Opts.FITNESS, Opts.ESTIMATION, or Opts.INFERENCE");
        }
        if (count > 1) throw new IllegalArgumentException("Output opts are mutually exclusive");
        return outputMask(opts);
    }

    private static int outputMask(Opts[] opts) {
        if (contains(opts, Opts.INFERENCE)) return OLS.OUTPUT_FULL;
        if (contains(opts, Opts.ESTIMATION)) return OLS.OUTPUT_PARAMS | OLS.OUTPUT_FITNESS;
        int mask = 0;
        if (contains(opts, Opts.PARAMS)) mask |= OLS.OUTPUT_PARAMS;
        if (contains(opts, Opts.FITNESS)) mask |= OLS.OUTPUT_FITNESS;
        return OLS.normalizeOutputMask(mask);
    }
}
