/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.linalg.reg;

import com.curioloop.yum4j.linalg.Regression;
import com.curioloop.yum4j.linalg.Regressor;
import com.curioloop.yum4j.linalg.blas.BLAS;
import com.curioloop.yum4j.math.VectorOps;

import static java.lang.Math.*;

/**
 * Ordinary Least Squares (OLS) linear regression.
 *
 * <p>Solves the linear model:
 * <pre>  y = Xβ + ε,  ε ~ N(0, σ²I)</pre>
 *
 * <p>Parameter estimation:
 * <pre>  β̂ = (XᵀX)⁻¹Xᵀy</pre>
 *
 * <p>Two solvers are supported:
 * <ul>
 *   <li>SVD — via pseudoinverse X⁺ = VΣ⁺Uᵀ (numerically robust, handles rank-deficient X)</li>
 *   <li>QR  — via QR factorization X = QR, then β̂ = R⁻¹Qᵀy (faster when X is full rank)</li>
 * </ul>
 *
 * <p>Data layout: X is row-major n×k, each row is one observation.
 * <b>X is overwritten in-place by the solver.</b> y is never modified.
 * Fitted values and residuals are computed inside the solver and cached in {@link Pool}.
 */
public class OLS extends Regression {

    private static final double PINV_RCOND = 1e-15;

    public static final int OUTPUT_PARAMS = 1;
    public static final int OUTPUT_FITNESS = 1 << 1;
    public static final int OUTPUT_COVARIANCE = 1 << 2;
    public static final int OUTPUT_FULL = OUTPUT_PARAMS | OUTPUT_FITNESS | OUTPUT_COVARIANCE;

    final int nObs;
    final int nParams;
    int kConst;           // resolved lazily in fit(); -1 = not yet computed
    final double[] y;
    final double[] X;
    final boolean useQR;
    final int outputMask;

    @Override public    int      nObs()       { return nObs; }
    @Override public    int      nParams()    { return nParams; }
    @Override public    int      kConst()     { return kConst; }
    @Override protected int      dfModel()    { return rank - kConst; }
    @Override protected int      dfResidual() { return nObs - rank; }
    @Override public    double[] endog()      { return y; }
    @Override public    double[] weights()    { return null; }

    // =========================================================================
    // Pool
    // =========================================================================

    /**
     * Reusable workspace for OLS computations.
     *
     * <p>All buffers grow on demand and are reused across fits.
     * No per-fit heap allocations after the first call.
     *
     * <ul>
     *   <li>{@code work}        — LAPACK floating-point scratch</li>
     *   <li>{@code ipiv}        — pivot indices (QR path)</li>
     *   <li>{@code beta}        — β̂ output, length k</li>
     *   <li>{@code unscaledCov} — (XᵀX)⁻¹, length k×k</li>
     *   <li>{@code fitted}      — ŷ = Xβ̂, length n (computed inside solver)</li>
     *   <li>{@code residual}    — e = y − ŷ, length n (computed inside solver)</li>
     * </ul>
     */
    public static class Pool {

        public double[] work;        // LAPACK floating-point scratch
        public int[]    ipiv;        // pivot indices for dgetrf/dgetri (QR path)
        public double[] beta;        // pre-allocated β̂ output (length k)
        public double[] unscaledCov; // pre-allocated (XᵀX)⁻¹ output (length k×k)
        public double[] fitted;      // ŷ = Xβ̂, length n
        public double[] residual;    // e = y − ŷ, length n

        public Pool() {}

        /**
         * Pre-allocates buffers for the requested solver/output options.
         * Empty options keep the historical behavior and allocate all result buffers.
         */
        public Pool ensure(int n, int k, Regressor.Opts... opts) {
            if (n < 1 || k < 1) throw new IllegalArgumentException("n and k must be >= 1");
            int mask = outputMask(opts);
            if ((mask & OUTPUT_PARAMS) != 0) ensureParams(k);
            if ((mask & OUTPUT_FITNESS) != 0) ensureFitness(n);
            if ((mask & OUTPUT_COVARIANCE) != 0) ensureCovariance(k);

            boolean qr = contains(opts, Regressor.Opts.QR);
            boolean pinv = contains(opts, Regressor.Opts.PINV);
            if (qr && pinv) throw new IllegalArgumentException("Cannot specify both Opts.QR and Opts.PINV");
            if (qr) ensureQR(n, k, mask);
            else if (pinv) ensureSVD(n, k, mask);
            return this;
        }

        Pool ensureParams(int k) {
            if (beta == null || beta.length < k) beta = new double[k];
            return this;
        }

        Pool ensureFitness(int n) {
            if (fitted   == null || fitted.length   < n) fitted   = new double[n];
            if (residual == null || residual.length < n) residual = new double[n];
            return this;
        }

        Pool ensureCovariance(int k) {
            if (unscaledCov == null || unscaledCov.length < k*k) unscaledCov = new double[k*k];
            return this;
        }

        Pool ensureWork(int size) {
            if (work == null || work.length < size) work = new double[size];
            return this;
        }

        Pool ensureIpiv(int size) {
            if (ipiv == null || ipiv.length < size) ipiv = new int[size];
            return this;
        }

        private Pool ensureQR(int n, int k, int mask) {
            int t = min(n, k);
            double[] wq = queryWork(this);
            BLAS.dgeqrf(n, k, null, 0, k, wq, 0, wq, 0, -1);
            int lwork = (int) wq[0];
            boolean fitness = (mask & OUTPUT_FITNESS) != 0;
            if (fitness) {
                BLAS.dorgqr(n, k, k, null, 0, k, wq, 0, wq, 0, -1);
                lwork = max(lwork, max((int) wq[0], k));
                boolean needR = (mask & (OUTPUT_PARAMS | OUTPUT_COVARIANCE)) != 0;
                int size = lwork + (needR ? k*k : 0) + t;
                if ((mask & OUTPUT_COVARIANCE) != 0) size = max(size, k + svdWorkSize(k, k, this));
                ensureWork(size);
                if ((mask & OUTPUT_COVARIANCE) != 0) ensureIpiv(k);
            } else {
                lwork = max(lwork, k);
                ensureWork(lwork + n + t);
            }
            return this;
        }

        private Pool ensureSVD(int n, int k, int mask) {
            int r = min(n, k);
            boolean overwriteU = n >= k;
            boolean needVT = (mask & (OUTPUT_PARAMS | OUTPUT_COVARIANCE)) != 0;
            int lwork;
            if (overwriteU) {
                lwork = svdOverwriteUWorkSize(n, k);
            } else {
                double[] wq = queryWork(this);
                BLAS.dgesvd('S', needVT ? 'S' : 'N', n, k, null, 0, k,
                        wq, 0, null, 0, r, null, 0, needVT ? k : 1, wq, 0, -1);
                lwork = (int) wq[0];
            }
            int vtSize = needVT ? r*k : 0;
            ensureWork(lwork + (overwriteU ? 0 : n*r) + vtSize + r + r);
            return this;
        }

    }

    /** Workspace bound to this result after {@code fit()}. */
    protected Pool pool;

    // =========================================================================
    // Constructors
    // =========================================================================

    public OLS(double[] y, double[] X, int n, int k, boolean useQR, int kConst) {
        this(y, X, n, k, useQR, kConst, OUTPUT_FULL);
    }

    public OLS(double[] y, double[] X, int n, int k, boolean useQR, int kConst, int outputMask) {
        if (n < 1 || k < 1) throw new IllegalArgumentException("n and k must be >= 1");
        if (y.length < n)   throw new IllegalArgumentException("y too short");
        if (X.length < n*k) throw new IllegalArgumentException("X too short");
        this.nObs    = n;
        this.nParams = k;
        this.y       = y;
        this.X       = X;
        this.useQR   = useQR;
        this.kConst  = kConst;  // -1 = auto-detect, 0 = no const, 1 = has const
        this.outputMask = normalizeOutputMask(outputMask);
    }



    // =========================================================================
    // fit
    // =========================================================================

    public OLS fit() {
        return fit(new Pool());
    }

    /**
     * Pre-allocates a reusable {@link Pool} for this model.
     *
     * <p>Pass the returned pool to repeated {@code fit()} calls to reuse solver buffers.</p>
     */
    public Pool alloc() {
        return new Pool();
    }

    /**
     * Fits the model.
     *
     * <p><b>X is overwritten in-place.</b> The solver operates directly on X,
     * computes fitted values and residuals internally, and caches them in {@code ws}.
     */
    public OLS fit(Pool ws) {
        if (ws == null) return fit();
        pool = ws;
        if (needsParams()) {
            ws.ensureParams(nParams);
        }
        if (needsFitness()) ws.ensureFitness(nObs);
        if (needsCovariance()) ws.ensureCovariance(nParams);
        if (kConst < 0) kConst = detectConst(X, nObs, nParams, ws);
        if (useQR) {
            if (needsFitness()) solveQR(y, X, ws);
            else                solveQRParamsOnly(y, X, ws);
        } else {
            if (needsFitness()) solveSVD(y, X, ws);
            else                solveSVDParamsOnly(y, X, ws);
        }
        return this;
    }

    /** Returns the workspace pool bound to this result, or {@code null} before fit. */
    public Pool pool() {
        return pool;
    }

    public static int normalizeOutputMask(int mask) {
        mask &= OUTPUT_FULL;
        if (mask == 0) return OUTPUT_FULL;
        if ((mask & OUTPUT_COVARIANCE) != 0) mask |= OUTPUT_PARAMS | OUTPUT_FITNESS;
        return mask;
    }

    protected final boolean needsParams() {
        return (outputMask & OUTPUT_PARAMS) != 0;
    }

    protected final boolean needsFitness() {
        return (outputMask & OUTPUT_FITNESS) != 0;
    }

    protected final boolean needsCovariance() {
        return (outputMask & OUTPUT_COVARIANCE) != 0;
    }

    // =========================================================================
    // SVD solver
    // =========================================================================
    //
    // Decompose X = UΣVᵀ, then:
    //   β̂          = VΣ⁺Uᵀy
    //   fitted      = U·(Uᵀy)  (projection onto column space of X)
    //   unscaledCov = VΣ⁺²Vᵀ  (= (XᵀX)⁻¹ when full rank)
    //
    // Memory layout in ws.work:
    //   [lwork | U:n×r | VT:r×k | S:r | tmp:r]

    void solveSVD(double[] endo, double[] exog, Pool ws) {
        int n = nObs, k = nParams;
        int r = min(n, k);
        boolean needParams = needsParams();
        boolean needCovariance = needsCovariance();
        boolean needVT = needParams || needCovariance;

        boolean overwriteU = n >= k;
        int lwork;
        if (overwriteU) {
            lwork = svdOverwriteUWorkSize(n, k);
        } else {
            double[] wq = queryWork(ws);
            BLAS.dgesvd('S', needVT ? 'S' : 'N', n, k, exog, 0, k,
                    wq, 0, null, 0, r, null, 0, needVT ? k : 1, wq, 0, -1);
            lwork = (int) wq[0];
        }

        // work layout: [lwork | U:n×r (wide fallback only) | VT:r×k | S:r | tmp:r]
        int offU = overwriteU ? -1 : lwork;
        int offVT = lwork + (overwriteU ? 0 : n*r);
        int vtSize = needVT ? r*k : 0;
        int offS = offVT + vtSize;
        int offTmp = offS + r;
        ws.ensureWork(lwork + (overwriteU ? 0 : n*r) + vtSize + r + r);

        BLAS.dgesvd(overwriteU ? 'O' : 'S', needVT ? 'S' : 'N', n, k, exog, 0, k,
                    ws.work, offS, overwriteU ? null : ws.work, overwriteU ? 0 : offU, overwriteU ? 1 : r,
                    needVT ? ws.work : null, needVT ? offVT : 0, needVT ? k : 1, ws.work, 0, lwork);

        double[] u = overwriteU ? exog : ws.work;
        int uOff = overwriteU ? 0 : offU;
        int ldu = overwriteU ? k : r;

        // rank / cond from Σ
        double cutoff = pinvCutoff(ws.work[offS], n, k);
        rank = 0;
        for (int i = 0; i < r; i++) if (ws.work[offS + i] > cutoff) rank++;
        cond = sqrt((ws.work[offS] * ws.work[offS]) / (ws.work[offS + r - 1] * ws.work[offS + r - 1]));

        // tmp = Uᵀy  (used for both fitted and β̂)
        BLAS.dgemv(BLAS.Trans.Trans, n, r, 1.0, u, uOff, ldu, endo, 0, 1, 0.0, ws.work, offTmp, 1);

        // fitted = U·tmp,  residual = endo - fitted
        // zero out components beyond rank before projecting
        for (int i = rank; i < r; i++) ws.work[offTmp + i] = 0.0;
        BLAS.dgemv(BLAS.Trans.NoTrans, n, r, 1.0, u, uOff, ldu, ws.work, offTmp, 1, 0.0, ws.fitted, 0, 1);
        VectorOps.axpyTo(ws.residual, 0, -1.0, ws.fitted, 0, endo, 0, n);
        setFittedResidualCache(ws);

        if (!needParams) {
            beta = null;
            unscaledCov = null;
            return;
        }

        for (int i = 0; i < r; i++) {
            double inv = (ws.work[offS + i] > cutoff) ? 1.0 / ws.work[offS + i] : 0.0;
            ws.work[offS + i] = inv;
            ws.work[offTmp + i] *= inv;
        }

        // β̂ = Vᵀᵀ(Σ⁺·(Uᵀy))
        beta = ws.beta;
        BLAS.dgemv(BLAS.Trans.Trans, r, k, 1.0, ws.work, offVT, k, ws.work, offTmp, 1, 0.0, beta, 0, 1);

        if (!needCovariance) {
            unscaledCov = null;
            return;
        }

        // unscaledCov = (VΣ⁺)ᵀ(VΣ⁺) via dsyrk
        unscaledCov = ws.unscaledCov;
        for (int i = 0; i < r; i++) {
            double si = ws.work[offS + i];
            for (int j = 0; j < k; j++) ws.work[offVT + i*k + j] *= si;
        }
        BLAS.dsyrk(BLAS.Uplo.Upper, BLAS.Trans.Trans, k, r,
                   1.0, ws.work, offVT, k, 0.0, unscaledCov, 0, k);
        for (int i = 0; i < k; i++)
            for (int j = i + 1; j < k; j++) unscaledCov[j*k + i] = unscaledCov[i*k + j];
    }

    void solveSVDParamsOnly(double[] endo, double[] exog, Pool ws) {
        int n = nObs, k = nParams;
        int r = min(n, k);

        boolean overwriteU = n >= k;
        int lwork;
        if (overwriteU) {
            lwork = svdOverwriteUWorkSize(n, k);
        } else {
            double[] wq = queryWork(ws);
            BLAS.dgesvd('S', 'S', n, k, exog, 0, k, wq, 0, null, 0, r, null, 0, k, wq, 0, -1);
            lwork = (int) wq[0];
        }

        int offU = overwriteU ? -1 : lwork;
        int offVT = lwork + (overwriteU ? 0 : n*r);
        int offS = offVT + r*k;
        int offTmp = offS + r;
        ws.ensureWork(lwork + (overwriteU ? 0 : n*r) + r*k + r + r);

        BLAS.dgesvd(overwriteU ? 'O' : 'S', 'S', n, k, exog, 0, k,
                ws.work, offS, overwriteU ? null : ws.work, overwriteU ? 0 : offU, overwriteU ? 1 : r,
                ws.work, offVT, k, ws.work, 0, lwork);

        double[] u = overwriteU ? exog : ws.work;
        int uOff = overwriteU ? 0 : offU;
        int ldu = overwriteU ? k : r;

        double cutoff = pinvCutoff(ws.work[offS], n, k);
        rank = 0;
        for (int i = 0; i < r; i++) if (ws.work[offS + i] > cutoff) rank++;
        cond = sqrt((ws.work[offS] * ws.work[offS]) / (ws.work[offS + r - 1] * ws.work[offS + r - 1]));

        BLAS.dgemv(BLAS.Trans.Trans, n, r, 1.0, u, uOff, ldu, endo, 0, 1, 0.0, ws.work, offTmp, 1);
        for (int i = 0; i < r; i++) {
            ws.work[offTmp + i] *= (ws.work[offS + i] > cutoff) ? 1.0 / ws.work[offS + i] : 0.0;
        }

        beta = ws.beta;
        BLAS.dgemv(BLAS.Trans.Trans, r, k, 1.0, ws.work, offVT, k, ws.work, offTmp, 1, 0.0, beta, 0, 1);
        unscaledCov = null;
    }

    // =========================================================================
    // QR solver
    // =========================================================================
    //
    // Decompose X = QR, then:
    //   β̂          = R⁻¹Qᵀy
    //   fitted      = Q·(Qᵀy)  (projection onto column space of X)
    //   unscaledCov = (RᵀR)⁻¹
    //
    // Memory layout in ws.work:
    //   [lwork | R:k×k | tau:t]

    void solveQR(double[] endo, double[] exog, Pool ws) {
        int n = nObs, k = nParams;
        int t = min(n, k);
        boolean needParams = needsParams();
        boolean needCovariance = needsCovariance();
        boolean needR = needParams || needCovariance;

        double[] wq = queryWork(ws);
        BLAS.dgeqrf(n, k, exog, 0, k, wq, 0, wq, 0, -1);
        int lwork = (int) wq[0];
        BLAS.dorgqr(n, k, k, exog, 0, k, wq, 0, wq, 0, -1);
        int orgqrWork = (int) wq[0];
        int svdRWork = needCovariance ? svdWorkSize(k, k, ws) : 0;
        lwork = max(lwork, max(orgqrWork, k));

        // work layout: [lwork | R:k×k | tau:t]
        int offR = lwork, offTau = lwork + (needR ? k*k : 0);
        ws.ensureWork(lwork + (needR ? k*k : 0) + t);
        if (needCovariance) ws.ensureIpiv(k);

        // Step 1: QR factorization
        BLAS.dgeqrf(n, k, exog, 0, k, ws.work, offTau, ws.work, 0, lwork);

        // Step 2: save R
        if (needR) {
            java.util.Arrays.fill(ws.work, offR, offR + k*k, 0.0);
            for (int i = 0; i < k; i++)
                System.arraycopy(exog, i*k + i, ws.work, offR + i*k + i, k - i);
        }

        // Step 3: expand Q in-place (exog now holds Q)
        BLAS.dorgqr(n, k, k, exog, 0, k, ws.work, offTau, ws.work, 0, lwork);

        // Step 4: Qᵀy
        beta = needParams ? ws.beta : null;
        double[] qty = needParams ? beta : ws.work;
        BLAS.dgemv(BLAS.Trans.Trans, n, k, 1.0, exog, 0, k, endo, 0, 1, 0.0, qty, 0, 1);

        // Step 5: fitted = Q·(Qᵀy),  residual = endo - fitted
        BLAS.dgemv(BLAS.Trans.NoTrans, n, k, 1.0, exog, 0, k, qty, 0, 1, 0.0, ws.fitted, 0, 1);
        VectorOps.axpyTo(ws.residual, 0, -1.0, ws.fitted, 0, endo, 0, n);
        setFittedResidualCache(ws);

        // Step 6: β̂ = R⁻¹(Qᵀy)
        if (needParams) {
            BLAS.dtrtrs(BLAS.Uplo.Upper, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit, k, 1,
                    ws.work, offR, k, beta, 0, 1);
        }

        if (!needCovariance) {
            rank = k;
            cond = Double.NaN;
            unscaledCov = null;
            return;
        }

        // Step 7: rank/cond via SVD of R — use ws.unscaledCov as temp
        unscaledCov = ws.unscaledCov;
        java.util.Arrays.fill(unscaledCov, 0, k*k, 0.0);
        for (int i = 0; i < k; i++)
            System.arraycopy(ws.work, offR + i*k + i, unscaledCov, i*k + i, k - i);
        BLAS.dgesvd('N', 'N', k, k, unscaledCov, 0, k, ws.work, 0,
                    null, 0, 1, null, 0, 1, ws.work, k, max(lwork - k, svdRWork));
        rank = 0;
        for (int i = 0; i < k; i++) if (ws.work[i] > ws.work[0] * 0x1p-53) rank++;
        cond = sqrt((ws.work[0] * ws.work[0]) / (ws.work[k - 1] * ws.work[k - 1]));

        // Step 8: unscaledCov = (RᵀR)⁻¹ via LU
        BLAS.dgemm(BLAS.Trans.Trans, BLAS.Trans.NoTrans, k, k, k,
               1.0, ws.work, offR, k, ws.work, offR, k, 0.0, unscaledCov, 0, k);
        BLAS.dgetrf(k, k, unscaledCov, 0, k, ws.ipiv, 0);
        BLAS.dgetri(k, unscaledCov, 0, k, ws.ipiv, 0, ws.work, 0, lwork);
    }

    void solveQRParamsOnly(double[] endo, double[] exog, Pool ws) {
        int n = nObs, k = nParams;
        int t = min(n, k);

        double[] wq = queryWork(ws);
        BLAS.dgeqrf(n, k, exog, 0, k, wq, 0, wq, 0, -1);
        int lwork = max((int) wq[0], k);

        int offQty = lwork;
        int offTau = offQty + n;
        ws.ensureWork(lwork + n + t);

        BLAS.dgeqrf(n, k, exog, 0, k, ws.work, offTau, ws.work, 0, lwork);

        System.arraycopy(endo, 0, ws.work, offQty, n);
        BLAS.dorm2r(BLAS.Side.Left, BLAS.Trans.Trans, n, 1, t,
                exog, 0, k, ws.work, offTau,
                ws.work, offQty, 1, ws.work, 0);

        beta = ws.beta;
        System.arraycopy(ws.work, offQty, beta, 0, k);
        BLAS.dtrtrs(BLAS.Uplo.Upper, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit, k, 1,
                    exog, 0, k, beta, 0, 1);

        rank = k;
        cond = Double.NaN;
        unscaledCov = null;
    }

    // =========================================================================
    // Constant detection
    // =========================================================================

    /**
     * Detects whether X contains a constant term (intercept column).
     */
    static int detectConst(double[] X, int n, int k, Pool ws) {
        // Fast path: single-pass O(nk) scan.
        // reuse work[0..k) as base — fast path does no LAPACK so work[0..k) is free
        ws.ensureWork(k);
        double[] base = ws.work;
        System.arraycopy(X, 0, base, 0, k);
        int remaining = k;
        for (int i = 1; i < n && remaining > 0; i++) {
            for (int j = 0; j < k; j++) {
                if (!Double.isNaN(base[j]) && X[i*k + j] != base[j]) {
                    base[j] = Double.NaN;
                    remaining--;
                }
            }
        }
        int constCount = 0;
        boolean hasExplicitOne = false;
        for (int j = 0; j < k; j++) {
            if (!Double.isNaN(base[j])) {
                constCount++;
                if (base[j] == 1.0) hasExplicitOne = true;
            }
        }
        if (constCount == 1) return 1;
        if (constCount > 1 && hasExplicitOne) return 1;

        // Slow path: compare rank(X) vs rank([1|X])
        // work layout: [ lwork | S: min(n,k1) | A: n×k1 ]
        int k1 = k + 1;
        int lwork = svdWorkSize(n, k1, ws);
        int offS  = lwork;
        int offA  = lwork + min(n, k1);
        ws.ensureWork(offA + n * k1);

        // first call: rank(X) — copy X into work[offA], padded to k1 columns
        for (int i = 0; i < n; i++)
            System.arraycopy(X, i * k, ws.work, offA + i * k, k);
        int orgRank = numericalRank(ws.work, offA, n, k, offS, lwork, ws);

        // second call: rank([1|X]) — prepend ones column
        for (int i = 0; i < n; i++) {
            ws.work[offA + i * k1] = 1.0;
            System.arraycopy(X, i * k, ws.work, offA + i * k1 + 1, k);
        }
        int augRank = numericalRank(ws.work, offA, n, k1, offS, lwork, ws);

        return augRank == orgRank ? 1 : 0;
    }

    private static double pinvCutoff(double largestSingularValue, int rows, int columns) {
        return largestSingularValue * max(PINV_RCOND, max(rows, columns) * 0x1p-52);
    }

    /** Returns the LAPACK lwork estimate for dgesvd('N','N', m, n). */
    static int svdWorkSize(int m, int n, Pool ws) {
        double[] wq = queryWork(ws);
        BLAS.dgesvd('N', 'N', m, n, null, 0, n, wq, 0, null, 0, 1, null, 0, 1, wq, 0, -1);
        return (int) wq[0];
    }

    private static int outputMask(Regressor.Opts[] opts) {
        int count = 0;
        int mask = 0;
        if (contains(opts, Regressor.Opts.PARAMS)) {
            count++;
            mask |= OUTPUT_PARAMS;
        }
        if (contains(opts, Regressor.Opts.FITNESS)) {
            count++;
            mask |= OUTPUT_FITNESS;
        }
        if (contains(opts, Regressor.Opts.ESTIMATION)) {
            count++;
            mask |= OUTPUT_PARAMS | OUTPUT_FITNESS;
        }
        if (contains(opts, Regressor.Opts.INFERENCE)) {
            count++;
            mask |= OUTPUT_FULL;
        }
        if (count > 1) throw new IllegalArgumentException("Output opts are mutually exclusive");
        return normalizeOutputMask(mask);
    }

    private static boolean contains(Regressor.Opts[] opts, Regressor.Opts target) {
        if (opts == null) return false;
        for (Regressor.Opts opt : opts) if (opt == target) return true;
        return false;
    }

    private static double[] queryWork(Pool ws) {
        ws.ensureWork(1);
        return ws.work;
    }

    /** Returns the minimum legal DGESVD work size for tall jobU='O', jobVT='S'. */
    static int svdOverwriteUWorkSize(int m, int n) {
        return max(3 * n + m, 5 * n);
    }

    /** Computes numerical rank of A (m×n) stored in ws.work[offA..], using ws.work[offS..] for singular values. */
    private static int numericalRank(double[] work, int offA, int m, int n, int offS, int lwork, Pool ws) {
        int minmn = min(m, n);
        BLAS.dgesvd('N', 'N', m, n, work, offA, n,
                    work, offS, null, 0, 1, null, 0, 1,
                    work, 0, lwork);
        int rank = 0;
        for (int i = 0; i < minmn; i++) if (work[offS + i] > work[offS] * 0x1p-53) rank++;
        return rank;
    }
}
