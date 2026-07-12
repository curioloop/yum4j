package com.curioloop.yum4j.optim.trf;

import com.curioloop.yum4j.linalg.blas.BLAS;

import java.util.Arrays;

/**
 * Shared MINPACK-style Levenberg-Marquardt linear algebra helpers.
 *
 * <p>This class keeps the QR, {@code qrsolv}, {@code lmpar}, and norm helpers
 * that were previously maintained in {@code TRFCore}. Method-level Javadocs
 * intentionally mirror the original TRF helper comments; overloads with explicit
 * offsets add only the extra indexing contract needed by callers that store
 * matrices and scratch vectors in larger reusable workspaces.</p>
 *
 * <p>All matrices are row-major. For an {@code m x n} matrix, element
 * {@code A[i,j]} is stored at {@code a[aOff + i*n + j]}.</p>
 */
public final class LMCore {

    private static final double EPSMCH = Math.ulp(1.0);
    private static final int QRFAC_TRAILING_SCALAR_MIN_N = 8;
    private static final int QRFAC_TRAILING_BLAS_MIN_N = 64;

    private LMCore() {}

    /** Euclidean norm of {@code a[off..off+n-1]}. */
    public static double enorm(int n, double[] a, int off) {
        return BLAS.dnrm2(n, a, off, 1);
    }

    /**
     * Column-pivoted QR factorization: A·P = Q·R (unblocked Householder).
     *
     * <p>Given m × n matrix A, computes the factorization:</p>
     * <pre>
     *   A·P = Q·R
     * </pre>
     * <p>where:</p>
     * <ul>
     *   <li>P is an n × n column permutation matrix (stored as ipvt)</li>
     *   <li>Q is an m × m orthogonal matrix (stored implicitly in lower triangle of A)</li>
     *   <li>R is an m × n upper triangular matrix (stored in upper triangle of A)</li>
     * </ul>
     *
     * <p>Column pivoting ensures |R₁₁| ≥ |R₂₂| ≥ ··· ≥ |Rₖₖ| which improves
     * numerical stability and enables rank determination.</p>
     *
     * <p>Extracted unchanged from the previous {@code TRFCore.qrfac} helper.</p>
     *
     * @param m      number of rows in A
     * @param n      number of columns in A
     * @param a      matrix A (row-major m×n), modified in place to store Q and R
     * @param ipvt   output: column permutation P (length n)
     * @param rdiag  output: diagonal of R (length n)
     * @param acnorm output: column norms of original A (length n)
     * @param wa     scratch array (length n)
     * @param tmp    scratch array of length n (reuses ws.qtf before applyQtToVec)
     * @param tmpOff offset into tmp[]
     */
    public static void qrfac(int m, int n, double[] a, int[] ipvt,
                             double[] rdiag, int rdiagOff,
                             double[] acnorm, int acnormOff,
                             double[] wa, int waOff,
                             double[] tmp, int tmpOff) {
        qrfac(m, n, a, 0, ipvt, 0, rdiag, rdiagOff, acnorm, acnormOff, wa, waOff, tmp, tmpOff);
    }

    /**
     * Column-pivoted QR factorization with explicit matrix and pivot offsets.
     *
     * <p>This overload performs the same algorithm described by
     * {@link #qrfac(int, int, double[], int[], double[], int, double[], int, double[], int, double[], int)},
     * with A and {@code ipvt} stored at caller-supplied offsets.</p>
     *
     * @param m         number of rows in A
     * @param n         number of columns in A
     * @param a         matrix A (row-major m×n), modified in place to store Q and R
     * @param aOff      offset of A inside {@code a}
     * @param ipvt      output: column permutation P
     * @param ipvtOff   offset into {@code ipvt}
     * @param rdiag     output: diagonal of R
     * @param rdiagOff  offset into {@code rdiag}
     * @param acnorm    output: column norms of original A
     * @param acnormOff offset into {@code acnorm}
     * @param wa        scratch array
     * @param waOff     offset into {@code wa}
     * @param tmp       scratch array
     * @param tmpOff    offset into {@code tmp}
     */
    public static void qrfac(int m, int n, double[] a, int aOff, int[] ipvt, int ipvtOff,
                             double[] rdiag, int rdiagOff,
                             double[] acnorm, int acnormOff,
                             double[] wa, int waOff,
                             double[] tmp, int tmpOff) {
        for (int j = 0; j < n; j++) { acnorm[acnormOff+j] = 0.0; rdiag[rdiagOff+j] = 0.0; }
        for (int i = 0; i < m; i++) {
            int rowBase = aOff + i * n;
            for (int j = 0; j < n; j++) { double v = a[rowBase + j]; acnorm[acnormOff+j] += v * v; }
        }
        for (int j = 0; j < n; j++) {
            acnorm[acnormOff+j] = Math.sqrt(acnorm[acnormOff+j]);
            rdiag[rdiagOff+j]   = acnorm[acnormOff+j];
            wa[waOff+j]         = rdiag[rdiagOff+j];
            ipvt[ipvtOff+j]     = j;
        }

        int minmn = Math.min(m, n);
        for (int j = 0; j < minmn; j++) {
            int kmax = j;
            for (int k = j + 1; k < n; k++) if (rdiag[rdiagOff+k] > rdiag[rdiagOff+kmax]) kmax = k;
            if (kmax != j) {
                for (int i = 0; i < m; i++) {
                    int base = aOff + i * n;
                    double t = a[base + j]; a[base + j] = a[base + kmax]; a[base + kmax] = t;
                }
                rdiag[rdiagOff+kmax] = rdiag[rdiagOff+j]; wa[waOff+kmax] = wa[waOff+j];
                int itmp = ipvt[ipvtOff+j]; ipvt[ipvtOff+j] = ipvt[ipvtOff+kmax]; ipvt[ipvtOff+kmax] = itmp;
            }

            double ajnorm = 0.0;
            for (int i = j; i < m; i++) { double v = a[aOff + i * n + j]; ajnorm += v * v; }
            ajnorm = Math.sqrt(ajnorm);
            if (ajnorm == 0.0) { rdiag[rdiagOff+j] = 0.0; continue; }
            if (a[aOff + j * n + j] < 0.0) ajnorm = -ajnorm;
            BLAS.dscal(m - j, 1.0 / ajnorm, a, aOff + j * n + j, n);
            a[aOff + j * n + j] += 1.0;

            int nk = n - j - 1;
            if (nk > 0) {
                int tmpBase = tmpOff + j + 1;
                Arrays.fill(tmp, tmpBase, tmpOff + n, 0.0);
                if (nk >= QRFAC_TRAILING_SCALAR_MIN_N && nk < QRFAC_TRAILING_BLAS_MIN_N) {
                    for (int i = j; i < m; i++) {
                        double vij = a[aOff + i * n + j];
                        int rowBase = aOff + i * n + j + 1;
                        for (int k = 0; k < nk; k++) {
                            tmp[tmpBase + k] = Math.fma(vij, a[rowBase + k], tmp[tmpBase + k]);
                        }
                    }
                    double invAjj = 1.0 / a[aOff + j * n + j];
                    for (int k = 0; k < nk; k++) tmp[tmpBase + k] *= invAjj;
                    for (int i = j; i < m; i++) {
                        double vij = a[aOff + i * n + j];
                        int rowBase = aOff + i * n + j + 1;
                        for (int k = 0; k < nk; k++) {
                            a[rowBase + k] = Math.fma(-vij, tmp[tmpBase + k], a[rowBase + k]);
                        }
                    }
                } else {
                    for (int i = j; i < m; i++) {
                        double vij = a[aOff + i * n + j];
                        int rowBase = aOff + i * n + j + 1;
                        BLAS.daxpy(nk, vij, a, rowBase, 1, tmp, tmpBase, 1);
                    }
                    BLAS.dscal(nk, 1.0 / a[aOff + j * n + j], tmp, tmpBase, 1);
                    for (int i = j; i < m; i++) {
                        double vij = a[aOff + i * n + j];
                        int rowBase = aOff + i * n + j + 1;
                        BLAS.daxpy(nk, -vij, tmp, tmpBase, 1, a, rowBase, 1);
                    }
                }
                for (int k = j + 1; k < n; k++) {
                    if (rdiag[rdiagOff+k] != 0.0) {
                        double t = a[aOff + j * n + k] / rdiag[rdiagOff+k];
                        rdiag[rdiagOff+k] *= Math.sqrt(Math.max(0.0, 1.0 - t * t));
                        if (0.05 * (rdiag[rdiagOff+k] / wa[waOff+k]) * (rdiag[rdiagOff+k] / wa[waOff+k]) <= EPSMCH) {
                            double s2 = 0.0;
                            for (int i = j + 1; i < m; i++) { double v = a[aOff + i * n + k]; s2 += v * v; }
                            rdiag[rdiagOff+k] = Math.sqrt(s2); wa[waOff+k] = rdiag[rdiagOff+k];
                        }
                    }
                }
            }
            rdiag[rdiagOff+j] = -ajnorm;
        }
    }

    /**
     * Applies Qᵀ to vector b in-place, then restores the R diagonal.
     *
     * <p>Given the Householder factors stored in the lower triangle of fjac
     * (as produced by qrfac), applies the accumulated orthogonal transformation:</p>
     * <pre>
     *   b ← Qᵀb
     * </pre>
     * <p>After application, restores the diagonal of R from rdiag so that
     * fjac contains the upper triangular R in its upper triangle.</p>
     *
     * @param fjac  Jacobian factor (row-major m×n), lower triangle holds Householder vectors
     * @param m     number of rows
     * @param n     number of columns
     * @param b     vector to transform in place (length m)
     * @param rdiag diagonal of R (length n), restored into fjac diagonal on exit
     */
    public static void applyQtToVec(double[] fjac, int m, int n, double[] b, double[] rdiag, int rdiagOff) {
        applyQtToVec(fjac, 0, m, n, b, 0, rdiag, rdiagOff);
    }

    /**
     * Applies Qᵀ to an offset vector using an offset row-major QR factor.
     *
     * <p>The diagonal of R is restored from {@code rdiag} before returning, matching
     * the original MINPACK workflow used by {@link #qrsolv} and {@link #lmpar}.</p>
     *
     * @param fjac     Jacobian factor (row-major m×n), lower triangle holds Householder vectors
     * @param fjacOff  offset of {@code fjac} inside the backing array
     * @param m        number of rows
     * @param n        number of columns
     * @param b        vector to transform in place
     * @param bOff     offset into {@code b}
     * @param rdiag    diagonal of R
     * @param rdiagOff offset into {@code rdiag}
     */
    public static void applyQtToVec(double[] fjac, int fjacOff, int m, int n,
                                    double[] b, int bOff, double[] rdiag, int rdiagOff) {
        int minmn = Math.min(m, n);
        for (int j = 0; j < minmn; j++) {
            int diagonal = fjacOff + j * n + j;
            if (fjac[diagonal] == 0.0) { fjac[diagonal] = rdiag[rdiagOff+j]; continue; }
            double sum = 0.0;
            for (int i = j; i < m; i++) sum += fjac[fjacOff + i * n + j] * b[bOff + i];
            double tmp = -sum / fjac[diagonal];
            BLAS.daxpy(m - j, tmp, fjac, diagonal, n, b, bOff + j, 1);
            fjac[diagonal] = rdiag[rdiagOff+j];
        }
    }

    /**
     * Solves the augmented least-squares system via Givens rotations (qrsolv).
     *
     * <p>Given the QR factorization A·P = Q·R, solves the augmented system:</p>
     * <pre>
     *   min ‖[ R  ] x - [ Qᵀb ]‖₂
     *       ‖[ D  ]     [  0  ]‖
     * </pre>
     * <p>where D = diag(d₁, ..., dₙ) is a diagonal matrix. This is equivalent to
     * solving the normal equations (RᵀR + DᵀD)x = Rᵀ(Qᵀb).</p>
     *
     * <p>The algorithm uses Givens rotations to zero out the diagonal elements of D
     * one at a time, updating R and Qᵀb accordingly. The resulting upper triangular
     * system is then solved by back-substitution.</p>
     *
     * @param n      number of variables
     * @param r      upper triangular R (n×n, row-major), modified in place
     * @param ipvt   column permutation from qrfac (length n)
     * @param diag   diagonal scaling D (length n)
     * @param qtb    Qᵀb vector (length n)
     * @param x      output: solution vector (length n)
     * @param sdiag  output: diagonal of the modified R after Givens rotations (length n)
     * @param wa     scratch array (length n)
     */
    public static void qrsolv(int n, double[] r, int rOff, int[] ipvt, double[] diag,
                              double[] qtb, int qtbOff, double[] x, int xOff, double[] sdiag, int sdiagOff,
                              double[] wa) {
        qrsolv(n, r, rOff, ipvt, 0, diag, 0, qtb, qtbOff, x, xOff, sdiag, sdiagOff, wa, 0);
    }

    /**
     * Solves the augmented least-squares system via Givens rotations with full offset support.
     *
     * <p>This overload performs the same algorithm described by
     * {@link #qrsolv(int, double[], int, int[], double[], double[], int, double[], int, double[], int, double[])},
     * with all mutable arrays stored at caller-supplied offsets.</p>
     *
     * @param n        number of variables
     * @param r        upper triangular R (n×n, row-major), modified in place
     * @param rOff     offset of R inside {@code r}
     * @param ipvt     column permutation from qrfac
     * @param ipvtOff  offset into {@code ipvt}
     * @param diag     diagonal scaling D
     * @param diagOff  offset into {@code diag}
     * @param qtb      Qᵀb vector
     * @param qtbOff   offset into {@code qtb}
     * @param x        output solution vector
     * @param xOff     offset into {@code x}
     * @param sdiag    output: diagonal of the modified R after Givens rotations
     * @param sdiagOff offset into {@code sdiag}
     * @param wa       scratch array
     * @param waOff    offset into {@code wa}
     */
    public static void qrsolv(int n, double[] r, int rOff, int[] ipvt, int ipvtOff,
                              double[] diag, int diagOff,
                              double[] qtb, int qtbOff, double[] x, int xOff, double[] sdiag, int sdiagOff,
                              double[] wa, int waOff) {
        for (int j = 0; j < n; j++) {
            for (int i = j; i < n; i++) r[rOff+i*n+j] = r[rOff+j*n+i];
            x[xOff+j] = r[rOff+j*n+j];
            wa[waOff+j] = qtb[qtbOff+j];
        }
        for (int j = 0; j < n; j++) {
            int l = ipvt[ipvtOff+j];
            if (diag[diagOff+l] == 0.0) { sdiag[sdiagOff+j] = r[rOff+j*n+j]; r[rOff+j*n+j] = x[xOff+j]; continue; }
            for (int k = j; k < n; k++) sdiag[sdiagOff+k] = 0.0;
            sdiag[sdiagOff+j] = diag[diagOff+l];
            double qtbpj = 0.0;
            for (int k = j; k < n; k++) {
                if (sdiag[sdiagOff+k] == 0.0) continue;
                double cos, sin;
                if (Math.abs(r[rOff+k*n+k]) >= Math.abs(sdiag[sdiagOff+k])) {
                    double tan = sdiag[sdiagOff+k] / r[rOff+k*n+k];
                    cos = 0.5 / Math.sqrt(0.25 + 0.25*tan*tan);
                    sin = cos * tan;
                } else {
                    double cotan = r[rOff+k*n+k] / sdiag[sdiagOff+k];
                    sin = 0.5 / Math.sqrt(0.25 + 0.25*cotan*cotan);
                    cos = sin * cotan;
                }
                r[rOff+k*n+k] = cos*r[rOff+k*n+k] + sin*sdiag[sdiagOff+k];
                double temp = cos*wa[waOff+k] + sin*qtbpj;
                qtbpj = -sin*wa[waOff+k] + cos*qtbpj;
                wa[waOff+k] = temp;
                for (int i = k+1; i < n; i++) {
                    temp              =  cos*r[rOff+i*n+k] + sin*sdiag[sdiagOff+i];
                    sdiag[sdiagOff+i] = -sin*r[rOff+i*n+k] + cos*sdiag[sdiagOff+i];
                    r[rOff+i*n+k]     = temp;
                }
            }
            sdiag[sdiagOff+j] = r[rOff+j*n+j];
            r[rOff+j*n+j] = x[xOff+j];
        }
        int nsing = n;
        for (int j = 0; j < n; j++) {
            if (sdiag[sdiagOff+j] == 0.0 && nsing == n) nsing = j;
            if (nsing < n) wa[waOff+j] = 0.0;
        }
        for (int k = 0; k < nsing; k++) {
            int j = nsing - 1 - k;
            double sum = 0.0;
            for (int i = j+1; i < nsing; i++) sum += r[rOff+i*n+j] * wa[waOff+i];
            wa[waOff+j] = (wa[waOff+j] - sum) / sdiag[sdiagOff+j];
        }
        for (int j = 0; j < n; j++) x[xOff+ipvt[ipvtOff+j]] = wa[waOff+j];
    }

    /**
     * Finds the Levenberg-Marquardt parameter λ such that ‖D·p(λ)‖₂ ≈ Δ (lmpar).
     *
     * <p>Given the QR factorization A·P = Q·R, finds λ ≥ 0 such that the solution p
     * to the augmented system:</p>
     * <pre>
     *   (RᵀR + λDᵀD)p = -Rᵀ(Qᵀf)
     * </pre>
     * <p>satisfies ‖D·p‖₂ ≈ Δ (the trust-region radius).</p>
     *
     * <p>The algorithm uses a Newton iteration on the secular equation:</p>
     * <pre>
     *   φ(λ) = 1/‖D·p(λ)‖₂ - 1/Δ = 0
     * </pre>
     * <p>starting from a bracket [λₗ, λᵤ] determined by the Gershgorin bounds.</p>
     *
     * <p>Special cases:</p>
     * <ul>
     *   <li>If ‖D·p(0)‖₂ ≤ Δ (unconstrained solution is feasible), returns λ = 0</li>
     *   <li>If rank(R) &lt; n (singular), uses the minimum-norm solution</li>
     * </ul>
     *
     * @param n        number of variables
     * @param r        upper triangular R (n×n, row-major), modified in place by qrsolv
     * @param rOff     offset of R inside {@code r}
     * @param ipvt     column permutation from qrfac (length n)
     * @param diag     diagonal scaling D (length n)
     * @param diagOff  offset into {@code diag}
     * @param qtb      Qᵀf vector (length n)
     * @param qtbOff   offset into {@code qtb}
     * @param delta    trust-region radius Δ
     * @param par      initial estimate of λ (updated on return)
     * @param x        output: solution p(λ) (length n, starting at xOff)
     * @param xOff     offset into x[]
     * @param sdiag    scratch: modified diagonal from qrsolv (length n)
     * @param sdiagOff offset into {@code sdiag}
     * @param wa1      scratch array (length n)
     * @param wa2      scratch array (length n)
     * @return         updated Levenberg-Marquardt parameter λ
     */
    public static double lmpar(int n, double[] r, int rOff, int[] ipvt,
                               double[] diag, int diagOff,
                               double[] qtb, int qtbOff,
                               double delta, double par,
                               double[] x, int xOff,
                               double[] sdiag, int sdiagOff,
                               double[] wa1,
                               double[] wa2) {
        return lmpar(n, r, rOff, ipvt, 0, diag, diagOff, qtb, qtbOff,
            delta, par, x, xOff, sdiag, sdiagOff, wa1, 0, wa2, 0);
    }

    /**
     * Finds the Levenberg-Marquardt parameter λ such that ‖D·p(λ)‖₂ ≈ Δ (lmpar).
     *
     * <p>This overload performs the same algorithm described by
     * {@link #lmpar(int, double[], int, int[], double[], int, double[], int, double, double, double[], int,
     * double[], int, double[], double[])}, with {@code ipvt}, {@code wa1}, and {@code wa2} stored at caller-supplied
     * offsets.</p>
     *
     * @param n        number of variables
     * @param r        upper triangular R (n×n, row-major), modified in place by qrsolv
     * @param rOff     offset of R inside {@code r}
     * @param ipvt     column permutation from qrfac
     * @param ipvtOff  offset into {@code ipvt}
     * @param diag     diagonal scaling D
     * @param diagOff  offset into {@code diag}
     * @param qtb      Qᵀf vector
     * @param qtbOff   offset into {@code qtb}
     * @param delta    trust-region radius Δ
     * @param par      initial estimate of λ (updated on return)
     * @param x        output: solution p(λ)
     * @param xOff     offset into {@code x}
     * @param sdiag    scratch: modified diagonal from qrsolv
     * @param sdiagOff offset into {@code sdiag}
     * @param wa1      scratch array
     * @param wa1Off   offset into {@code wa1}
     * @param wa2      scratch array
     * @param wa2Off   offset into {@code wa2}
     * @return         updated Levenberg-Marquardt parameter λ
     */
    public static double lmpar(int n, double[] r, int rOff, int[] ipvt, int ipvtOff,
                               double[] diag, int diagOff,
                               double[] qtb, int qtbOff,
                               double delta, double par,
                               double[] x, int xOff,
                               double[] sdiag, int sdiagOff,
                               double[] wa1, int wa1Off,
                               double[] wa2, int wa2Off) {
        final double dwarf = Double.MIN_VALUE;
        final double p1 = 0.1, p001 = 0.001;

        int nsing = n;
        System.arraycopy(qtb, qtbOff, wa1, wa1Off, n);
        for (int j = 0; j < n; j++) {
            if (r[rOff+j*n+j] == 0.0 && nsing == n) nsing = j;
            if (nsing < n) wa1[wa1Off+j] = 0.0;
        }
        for (int k = 0; k < nsing; k++) {
            int j = nsing - 1 - k;
            wa1[wa1Off+j] /= r[rOff+j*n+j];
            double tmp = wa1[wa1Off+j];
            for (int i = 0; i < j; i++) wa1[wa1Off+i] -= r[rOff+i*n+j] * tmp;
        }
        for (int j = 0; j < n; j++) x[xOff+ipvt[ipvtOff+j]] = wa1[wa1Off+j];

        int iter = 0;
        for (int j = 0; j < n; j++) wa2[wa2Off+j] = diag[diagOff+j] * x[xOff+j];
        double dxnorm = enorm(n, wa2, wa2Off);
        double fp = dxnorm - delta;
        if (fp <= p1 * delta) { if (iter == 0) par = 0.0; return par; }

        double parl = 0.0;
        if (nsing >= n) {
            for (int j = 0; j < n; j++) {
                int l = ipvt[ipvtOff+j];
                wa1[wa1Off+j] = diag[diagOff+l] * (wa2[wa2Off+l] / dxnorm);
            }
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int i = 0; i < j; i++) sum += r[rOff+i*n+j] * wa1[wa1Off+i];
                wa1[wa1Off+j] = (wa1[wa1Off+j] - sum) / r[rOff+j*n+j];
            }
            double tmp = enorm(n, wa1, wa1Off);
            parl = ((fp / delta) / tmp) / tmp;
        }

        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int i = 0; i <= j; i++) sum += r[rOff+i*n+j] * qtb[qtbOff+i];
            int l = ipvt[ipvtOff+j];
            wa1[wa1Off+j] = sum / diag[diagOff+l];
        }
        double gnorm = enorm(n, wa1, wa1Off);
        double paru = gnorm / delta;
        if (paru == 0.0) paru = dwarf / Math.min(delta, p1);

        par = Math.max(par, parl);
        par = Math.min(par, paru);
        if (par == 0.0) par = gnorm / dxnorm;

        for (iter = 1; iter <= 10; iter++) {
            if (par == 0.0) par = Math.max(dwarf, p001 * paru);
            double sqrtPar = Math.sqrt(par);
            for (int j = 0; j < n; j++) wa1[wa1Off+j] = sqrtPar * diag[diagOff+j];
            qrsolv(n, r, rOff, ipvt, ipvtOff, wa1, wa1Off, qtb, qtbOff, x, xOff, sdiag, sdiagOff, wa2, wa2Off);
            for (int j = 0; j < n; j++) wa2[wa2Off+j] = diag[diagOff+j] * x[xOff+j];
            dxnorm = enorm(n, wa2, wa2Off);
            double fpOld = fp;
            fp = dxnorm - delta;

            if (Math.abs(fp) <= p1 * delta
                || (parl == 0.0 && fp <= fpOld && fpOld < 0.0)
                || iter == 10) break;

            for (int j = 0; j < n; j++) {
                int l = ipvt[ipvtOff+j];
                wa1[wa1Off+j] = diag[diagOff+l] * (wa2[wa2Off+l] / dxnorm);
            }
            for (int j = 0; j < n; j++) {
                wa1[wa1Off+j] /= sdiag[sdiagOff+j];
                double tmp = wa1[wa1Off+j];
                for (int i = j+1; i < n; i++) wa1[wa1Off+i] -= r[rOff+i*n+j] * tmp;
            }
            double tmp = enorm(n, wa1, wa1Off);
            double parc = ((fp / delta) / tmp) / tmp;
            if (fp > 0.0) parl = Math.max(parl, par);
            if (fp < 0.0) paru = Math.min(paru, par);
            par = Math.max(parl, par + parc);
        }
        if (iter == 0) par = 0.0;
        return par;
    }
}
