/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.linalg.blas.cmplx;

import com.curioloop.yum4j.linalg.blas.BLAS;
import com.curioloop.yum4j.linalg.blas.Ilaenv;

import static java.lang.Math.max;
import static java.lang.Math.min;

interface Zgetr {

    static int zgetf2(int m, int n, double[] A, int aOff, int lda, int[] ipiv, int ipivOff) {
        if (m < 0 || n < 0 || lda < max(1, n)) return -1;

        int mn = min(m, n);
        if (mn == 0) return 0;

        int info = 0;

        for (int j = 0; j < mn; j++) {
            int jp = j + ZLAS.izamax(m - j, A, (aOff + j * lda + j) * 2, lda);
            ipiv[ipivOff + j] = jp + 1;

            if (A[(aOff + jp * lda + j) * 2] == 0 && A[(aOff + jp * lda + j) * 2 + 1] == 0) {
                if (info == 0) info = j + 1;
            } else {
                if (jp != j) {
                    Zswap.zswap(n, A, (aOff + j * lda) * 2, 1, A, (aOff + jp * lda) * 2, 1);
                }
                if (j < m - 1) {
                    double ajRe = A[(aOff + j * lda + j) * 2];
                    double ajIm = A[(aOff + j * lda + j) * 2 + 1];
                    double denom = ajRe * ajRe + ajIm * ajIm;
                    for (int i = j + 1; i < m; i++) {
                        double aiRe = A[(aOff + i * lda + j) * 2];
                        double aiIm = A[(aOff + i * lda + j) * 2 + 1];
                        A[(aOff + i * lda + j) * 2] = (aiRe * ajRe + aiIm * ajIm) / denom;
                        A[(aOff + i * lda + j) * 2 + 1] = (aiIm * ajRe - aiRe * ajIm) / denom;
                    }
                }
            }

            if (j < mn - 1) {
                Zgerc.zgerc(m - j - 1, n - j - 1, -1.0, 0.0,
                        A, (aOff + (j + 1) * lda + j) * 2, lda,
                        A, (aOff + j * lda + j + 1) * 2, 1,
                        A, (aOff + (j + 1) * lda + j + 1) * 2, lda);
            }
        }

        return info;
    }

    static boolean zgetri(int n, double[] A, int aOff, int lda, int[] ipiv,
                          double[] work, int workOff, int lwork) {
        if (n == 0) {
            work[workOff] = 1;
            return true;
        }

        int nb = Ilaenv.ilaenv(1, "ZGETRI", " ", n, -1, -1, -1);
        if (lwork == -1) {
            work[workOff] = n * nb * 2;
            return true;
        }

        if (Ztrtri.ztrtri(BLAS.Uplo.Upper, BLAS.Diag.NonUnit, n, A, aOff, lda) != 0) {
            return false;
        }

        int nbmin = 2;
        int ldwork = nb * 2;

        if (nb >= nbmin && n > nb) {
            int iws = max(n * nb * 2, 2);
            if (lwork < iws) {
                nb = lwork / (n * 2);
                nbmin = 2;
            }
            ldwork = nb * 2;
        }

        if (nb < nbmin || n <= nb) {
            for (int j = n - 1; j >= 0; j--) {
                for (int i = j + 1; i < n; i++) {
                    work[workOff + i * 2] = A[(aOff + i * lda + j) * 2];
                    work[workOff + i * 2 + 1] = A[(aOff + i * lda + j) * 2 + 1];
                    A[(aOff + i * lda + j) * 2] = 0;
                    A[(aOff + i * lda + j) * 2 + 1] = 0;
                }
                if (j < n - 1) {
                    for (int i = 0; i < n; i++) {
                        double sumRe = 0, sumIm = 0;
                        for (int k = j + 1; k < n; k++) {
                            double aRe = A[(aOff + i * lda + k) * 2];
                            double aIm = A[(aOff + i * lda + k) * 2 + 1];
                            double wRe = work[workOff + k * 2];
                            double wIm = work[workOff + k * 2 + 1];
                            sumRe += aRe * wRe - aIm * wIm;
                            sumIm += aRe * wIm + aIm * wRe;
                        }
                        A[(aOff + i * lda + j) * 2] -= sumRe;
                        A[(aOff + i * lda + j) * 2 + 1] -= sumIm;
                    }
                }
            }
        } else {
            int nn = ((n - 1) / nb) * nb;
            for (int j = nn; j >= 0; j -= nb) {
                int jb = min(nb, n - j);

                for (int jj = j; jj < j + jb; jj++) {
                    for (int i = jj + 1; i < n; i++) {
                        work[workOff + i * ldwork + (jj - j) * 2] = A[(aOff + i * lda + jj) * 2];
                        work[workOff + i * ldwork + (jj - j) * 2 + 1] = A[(aOff + i * lda + jj) * 2 + 1];
                        A[(aOff + i * lda + jj) * 2] = 0;
                        A[(aOff + i * lda + jj) * 2 + 1] = 0;
                    }
                }

                if (j + jb < n) {
                    Zgemm.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, n, jb, n - j - jb,
                            -1.0, 0.0,
                            A, aOff + j + jb, lda,
                            work, workOff / 2 + (j + jb) * (ldwork / 2), ldwork / 2,
                            1.0, 0.0,
                            A, aOff + j, lda);
                }
                Ztrsm.ztrsm(BLAS.Side.Right, BLAS.Uplo.Lower, BLAS.Trans.NoTrans, BLAS.Diag.Unit,
                        n, jb, 1.0, 0.0,
                        work, workOff + j * ldwork, ldwork / 2,
                        A, (aOff + j) * 2, lda);
            }
        }

        for (int j = n - 2; j >= 0; j--) {
            int jp = ipiv[j] - 1;
            if (jp != j) {
                Zswap.zswap(n, A, (aOff + j) * 2, lda, A, (aOff + jp) * 2, lda);
            }
        }

        return true;
    }

    static double zgecon(char norm, int n, double[] A, int aOff, int lda, double anorm,
                         double[] work, int[] iwork) {
        if (n == 0) return 1.0;
        if (anorm == 0) return 0.0;
        if (Double.isNaN(anorm)) return anorm;
        if (Double.isInfinite(anorm)) return 0.0;

        double smlnum = BLAS.dlamch('S');
        double rcond = 0.0;
        double ainvnm = 0.0;
        boolean normin = false;
        boolean onenrm = norm == '1' || norm == 'O' || norm == 'o';
        int kase1 = onenrm ? 1 : 2;

        int[] isave = new int[3];
        int[] kase = {0};

        while (true) {
            ainvnm = Zlacn2.zlacn2(n, work, n * 2, work, 0, iwork, 0, ainvnm, kase, 0, isave, 0);

            if (kase[0] == 0) {
                if (ainvnm != 0) {
                    rcond = (1.0 / ainvnm) / anorm;
                }
                return rcond;
            }

            double sl, su;
            if (kase[0] == kase1) {
                sl = Zlatrs.zlatrs(BLAS.Uplo.Lower, BLAS.Trans.NoTrans, BLAS.Diag.Unit, normin, n, A, aOff, lda, work, 0, work, 4 * n);
                su = Zlatrs.zlatrs(BLAS.Uplo.Upper, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit, normin, n, A, aOff, lda, work, 0, work, 5 * n);
            } else {
                su = Zlatrs.zlatrs(BLAS.Uplo.Upper, BLAS.Trans.Conj, BLAS.Diag.NonUnit, normin, n, A, aOff, lda, work, 0, work, 5 * n);
                sl = Zlatrs.zlatrs(BLAS.Uplo.Lower, BLAS.Trans.Conj, BLAS.Diag.Unit, normin, n, A, aOff, lda, work, 0, work, 4 * n);
            }

            double scale = sl * su;
            normin = true;

            if (scale != 1) {
                double xmax = Izamax.izamaxAbs(n, work, 0, 1);
                if (scale == 0 || scale < xmax * smlnum) {
                    return rcond;
                }
                Zdrscl.zdrscl(n, scale, work, 0, 1);
            }
        }
    }

    static int zgetrf(int m, int n, double[] A, int aOff, int lda, int[] ipiv) {
        if (m < 0) return -1;
        if (n < 0) return -2;
        if (A == null) return -3;
        if (aOff < 0) return -4;
        if (lda < Math.max(1, n)) return -5;
        if (ipiv == null) return -6;
        if (ipiv.length < Math.min(m, n)) return -7;

        if (m == 0 || n == 0) {
            return 0;
        }

        int info = 0;
        int minMN = Math.min(m, n);

        for (int j = 0; j < minMN; j++) {
            int pivot = j;
            double maxAbs = 0.0;

            for (int i = j; i < m; i++) {
                double re = A[aOff + (i * lda + j) * 2];
                double im = A[aOff + (i * lda + j) * 2 + 1];
                double absVal = Math.hypot(re, im);
                if (absVal > maxAbs) {
                    maxAbs = absVal;
                    pivot = i;
                }
            }

            ipiv[j] = pivot + 1;

            if (pivot != j) {
                for (int k = 0; k < n; k++) {
                    int pos1 = aOff + (j * lda + k) * 2;
                    int pos2 = aOff + (pivot * lda + k) * 2;
                    double tempRe = A[pos1];
                    A[pos1] = A[pos2];
                    A[pos2] = tempRe;
                    double tempIm = A[pos1 + 1];
                    A[pos1 + 1] = A[pos2 + 1];
                    A[pos2 + 1] = tempIm;
                }
            }

            if (j < m && (A[aOff + (j * lda + j) * 2] != 0.0 || A[aOff + (j * lda + j) * 2 + 1] != 0.0)) {
                for (int i = j + 1; i < m; i++) {
                    double ajjRe = A[aOff + (j * lda + j) * 2];
                    double ajjIm = A[aOff + (j * lda + j) * 2 + 1];
                    double aijRe = A[aOff + (i * lda + j) * 2];
                    double aijIm = A[aOff + (i * lda + j) * 2 + 1];

                    double denom = ajjRe * ajjRe + ajjIm * ajjIm;
                    double multiplierRe = (aijRe * ajjRe + aijIm * ajjIm) / denom;
                    double multiplierIm = (aijIm * ajjRe - aijRe * ajjIm) / denom;

                    A[aOff + (i * lda + j) * 2] = multiplierRe;
                    A[aOff + (i * lda + j) * 2 + 1] = multiplierIm;

                    for (int k = j + 1; k < n; k++) {
                        double akjRe = A[aOff + (j * lda + k) * 2];
                        double akjIm = A[aOff + (j * lda + k) * 2 + 1];
                        double aikRe = A[aOff + (i * lda + k) * 2];
                        double aikIm = A[aOff + (i * lda + k) * 2 + 1];

                        double tempRe = multiplierRe * akjRe - multiplierIm * akjIm;
                        double tempIm = multiplierRe * akjIm + multiplierIm * akjRe;
                        A[aOff + (i * lda + k) * 2] = aikRe - tempRe;
                        A[aOff + (i * lda + k) * 2 + 1] = aikIm - tempIm;
                    }
                }
            } else if (j < m && info == 0) {
                info = j + 1;
            }
        }

        return info;
    }

    static void zgetrs(BLAS.Trans trans, int n, int nrhs, double[] A, int aOff, int lda,
                       int[] ipiv, double[] B, int bOff, int ldb) {
        if (n == 0 || nrhs == 0) return;

        if (trans == BLAS.Trans.NoTrans) {
            for (int i = 0; i < n; i++) {
                int j = ipiv[i] - 1;
                if (j != i) {
                    int rowI = bOff + i * ldb * 2;
                    int rowJ = bOff + j * ldb * 2;
                    for (int k = 0; k < nrhs; k++) {
                        int pI = rowI + k * 2, pJ = rowJ + k * 2;
                        double tmpRe = B[pI], tmpIm = B[pI + 1];
                        B[pI] = B[pJ]; B[pI + 1] = B[pJ + 1];
                        B[pJ] = tmpRe; B[pJ + 1] = tmpIm;
                    }
                }
            }
            Ztrsm.ztrsm(BLAS.Side.Left, BLAS.Uplo.Lower, BLAS.Trans.NoTrans, BLAS.Diag.Unit,
                    n, nrhs, 1.0, 0.0, A, aOff * 2, lda, B, bOff, ldb);
            Ztrsm.ztrsm(BLAS.Side.Left, BLAS.Uplo.Upper, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit,
                    n, nrhs, 1.0, 0.0, A, aOff * 2, lda, B, bOff, ldb);
        } else if (trans == BLAS.Trans.Trans) {
            Ztrsm.ztrsm(BLAS.Side.Left, BLAS.Uplo.Upper, BLAS.Trans.Trans, BLAS.Diag.NonUnit,
                    n, nrhs, 1.0, 0.0, A, aOff * 2, lda, B, bOff, ldb);
            Ztrsm.ztrsm(BLAS.Side.Left, BLAS.Uplo.Lower, BLAS.Trans.Trans, BLAS.Diag.Unit,
                    n, nrhs, 1.0, 0.0, A, aOff * 2, lda, B, bOff, ldb);
            for (int i = n - 1; i >= 0; i--) {
                int j = ipiv[i] - 1;
                if (j != i) {
                    int rowI = bOff + i * ldb * 2;
                    int rowJ = bOff + j * ldb * 2;
                    for (int k = 0; k < nrhs; k++) {
                        int pI = rowI + k * 2, pJ = rowJ + k * 2;
                        double tmpRe = B[pI], tmpIm = B[pI + 1];
                        B[pI] = B[pJ]; B[pI + 1] = B[pJ + 1];
                        B[pJ] = tmpRe; B[pJ + 1] = tmpIm;
                    }
                }
            }
        } else {
            Ztrsm.ztrsm(BLAS.Side.Left, BLAS.Uplo.Upper, BLAS.Trans.Conj, BLAS.Diag.NonUnit,
                    n, nrhs, 1.0, 0.0, A, aOff * 2, lda, B, bOff, ldb);
            Ztrsm.ztrsm(BLAS.Side.Left, BLAS.Uplo.Lower, BLAS.Trans.Conj, BLAS.Diag.Unit,
                    n, nrhs, 1.0, 0.0, A, aOff * 2, lda, B, bOff, ldb);
            for (int i = n - 1; i >= 0; i--) {
                int j = ipiv[i] - 1;
                if (j != i) {
                    int rowI = bOff + i * ldb * 2;
                    int rowJ = bOff + j * ldb * 2;
                    for (int k = 0; k < nrhs; k++) {
                        int pI = rowI + k * 2, pJ = rowJ + k * 2;
                        double tmpRe = B[pI], tmpIm = B[pI + 1];
                        B[pI] = B[pJ]; B[pI + 1] = B[pJ + 1];
                        B[pJ] = tmpRe; B[pJ + 1] = tmpIm;
                    }
                }
            }
        }
    }
}
