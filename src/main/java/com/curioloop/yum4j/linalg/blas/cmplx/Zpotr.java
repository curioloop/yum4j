/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.linalg.blas.cmplx;

import com.curioloop.yum4j.linalg.blas.BLAS;
import com.curioloop.yum4j.linalg.blas.Ilaenv;

import static java.lang.Math.*;

interface Zpotr {

    static boolean zpotf2(BLAS.Uplo uplo, int n, double[] A, int aOff, int lda) {
        if (n < 0 || lda < max(1, n)) return false;
        if (n == 0) return true;
        boolean lower = uplo == BLAS.Uplo.Lower;
        return lower ? zpotf2Lower(n, A, aOff, lda) : zpotf2Upper(n, A, aOff, lda);
    }

    static boolean zpotf2Lower(int n, double[] A, int aOff, int lda) {
        for (int j = 0; j < n; j++) {
            int rowJ = aOff + j * lda;
            double ajjRe = A[(rowJ + j) * 2];
            if (j > 0) {
                for (int k = 0; k < j; k++) {
                    double re = A[(rowJ + k) * 2];
                    double im = A[(rowJ + k) * 2 + 1];
                    ajjRe -= re * re + im * im;
                }
            }
            if (ajjRe <= 0 || Double.isNaN(ajjRe)) {
                A[(rowJ + j) * 2] = ajjRe;
                return false;
            }
            double sqrtAjj = sqrt(ajjRe);
            A[(rowJ + j) * 2] = sqrtAjj;
            A[(rowJ + j) * 2 + 1] = 0;
            if (j < n - 1) {
                for (int i = j + 1; i < n; i++) {
                    double aijRe = A[(aOff + i * lda + j) * 2];
                    double aijIm = A[(aOff + i * lda + j) * 2 + 1];
                    if (j > 0) {
                        for (int k = 0; k < j; k++) {
                            double akjRe = A[(rowJ + k) * 2];
                            double akjIm = A[(rowJ + k) * 2 + 1];
                            double akiRe = A[(aOff + i * lda + k) * 2];
                            double akiIm = A[(aOff + i * lda + k) * 2 + 1];
                            aijRe -= akiRe * akjRe + akiIm * akjIm;
                            aijIm -= akiIm * akjRe - akiRe * akjIm;
                        }
                    }
                    A[(aOff + i * lda + j) * 2] = aijRe / sqrtAjj;
                    A[(aOff + i * lda + j) * 2 + 1] = aijIm / sqrtAjj;
                }
            }
        }
        return true;
    }

    static boolean zpotf2Upper(int n, double[] A, int aOff, int lda) {
        for (int j = 0; j < n; j++) {
            int rowJ = aOff + j * lda;
            double ajjRe = A[(rowJ + j) * 2];
            if (j > 0) {
                for (int k = 0; k < j; k++) {
                    double re = A[(aOff + k * lda + j) * 2];
                    double im = A[(aOff + k * lda + j) * 2 + 1];
                    ajjRe -= re * re + im * im;
                }
            }
            if (ajjRe <= 0 || Double.isNaN(ajjRe)) {
                A[(rowJ + j) * 2] = ajjRe;
                return false;
            }
            double sqrtAjj = sqrt(ajjRe);
            A[(rowJ + j) * 2] = sqrtAjj;
            A[(rowJ + j) * 2 + 1] = 0;
            if (j < n - 1) {
                for (int i = j + 1; i < n; i++) {
                    double aijRe = A[(rowJ + i) * 2];
                    double aijIm = A[(rowJ + i) * 2 + 1];
                    if (j > 0) {
                        for (int k = 0; k < j; k++) {
                            double akjRe = A[(aOff + k * lda + j) * 2];
                            double akjIm = A[(aOff + k * lda + j) * 2 + 1];
                            double akiRe = A[(aOff + k * lda + i) * 2];
                            double akiIm = A[(aOff + k * lda + i) * 2 + 1];
                            aijRe -= akjRe * akiRe + akjIm * akiIm;
                            aijIm -= akjRe * akiIm - akjIm * akiRe;
                        }
                    }
                    A[(rowJ + i) * 2] = aijRe / sqrtAjj;
                    A[(rowJ + i) * 2 + 1] = aijIm / sqrtAjj;
                }
            }
        }
        return true;
    }

    static int zpotrf(BLAS.Uplo uplo, int n, double[] A, int aOff, int lda) {
        if (uplo == null) return -1;
        if (n < 0) return -2;
        if (A == null) return -3;
        if (aOff < 0) return -4;
        if (lda < max(1, n)) return -5;
        if (n == 0) return 0;

        boolean lower = uplo == BLAS.Uplo.Lower;
        int nb = Ilaenv.ilaenv(1, "ZPOTRF", String.valueOf(uplo.code), n, -1, -1, -1);

        if (nb <= 1 || n <= nb) {
            return zpotf2(uplo, n, A, aOff, lda) ? 0 : 1;
        }

        return lower ? zpotrfLower(n, A, aOff, lda, nb) : zpotrfUpper(n, A, aOff, lda, nb);
    }

    static int zpotrfLower(int n, double[] A, int aOff, int lda, int nb) {
        for (int j = 0; j < n; j += nb) {
            int jb = min(nb, n - j);

            Zherk.zherk(BLAS.Uplo.Lower, BLAS.Trans.NoTrans, jb, j, -1.0,
                    A, aOff + j * lda, lda,
                    1.0, A, aOff + j * lda + j, lda);

            if (!zpotf2Lower(jb, A, aOff + j * lda + j, lda)) return j + 1;

            if (j + jb < n) {
                Zgemm.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.Conj, n - j - jb, jb, j,
                        -1.0, 0.0,
                        A, aOff + (j + jb) * lda, lda,
                        A, aOff + j * lda, lda,
                        1.0, 0.0,
                        A, aOff + (j + jb) * lda + j, lda);

                Ztrsm.ztrsm(BLAS.Side.Right, BLAS.Uplo.Lower, BLAS.Trans.Conj, BLAS.Diag.NonUnit,
                        n - j - jb, jb, 1.0, 0.0,
                        A, (aOff + j * lda + j) * 2, lda,
                        A, (aOff + (j + jb) * lda + j) * 2, lda);
            }
        }
        return 0;
    }

    static int zpotrfUpper(int n, double[] A, int aOff, int lda, int nb) {
        for (int j = 0; j < n; j += nb) {
            int jb = min(nb, n - j);

            Zherk.zherk(BLAS.Uplo.Upper, BLAS.Trans.Conj, jb, j, -1.0,
                    A, aOff + j, lda,
                    1.0, A, aOff + j * lda + j, lda);

            if (!zpotf2Upper(jb, A, aOff + j * lda + j, lda)) return j + 1;

            if (j + jb < n) {
                Zgemm.zgemm(BLAS.Trans.Conj, BLAS.Trans.NoTrans, jb, n - j - jb, j,
                        -1.0, 0.0,
                        A, aOff + j, lda,
                        A, aOff + j + jb, lda,
                        1.0, 0.0,
                        A, aOff + j * lda + j + jb, lda);

                Ztrsm.ztrsm(BLAS.Side.Left, BLAS.Uplo.Upper, BLAS.Trans.Conj, BLAS.Diag.NonUnit,
                        jb, n - j - jb, 1.0, 0.0,
                        A, (aOff + j * lda + j) * 2, lda,
                        A, (aOff + j * lda + j + jb) * 2, lda);
            }
        }
        return 0;
    }

    static void zpotrs(BLAS.Uplo uplo, int n, int nrhs, double[] A, int aOff, int lda,
                       double[] B, int bOff, int ldb) {
        if (n == 0 || nrhs == 0) return;

        boolean lower = uplo == BLAS.Uplo.Lower;
        if (lower) {
            Ztrsm.ztrsm(BLAS.Side.Left, BLAS.Uplo.Lower, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit,
                    n, nrhs, 1.0, 0.0, A, aOff * 2, lda, B, bOff * 2, ldb);
            Ztrsm.ztrsm(BLAS.Side.Left, BLAS.Uplo.Lower, BLAS.Trans.Conj, BLAS.Diag.NonUnit,
                    n, nrhs, 1.0, 0.0, A, aOff * 2, lda, B, bOff * 2, ldb);
        } else {
            Ztrsm.ztrsm(BLAS.Side.Left, BLAS.Uplo.Upper, BLAS.Trans.Conj, BLAS.Diag.NonUnit,
                    n, nrhs, 1.0, 0.0, A, aOff * 2, lda, B, bOff * 2, ldb);
            Ztrsm.ztrsm(BLAS.Side.Left, BLAS.Uplo.Upper, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit,
                    n, nrhs, 1.0, 0.0, A, aOff * 2, lda, B, bOff * 2, ldb);
        }
    }

    static boolean zpotri(BLAS.Uplo uplo, int n, double[] A, int aOff, int lda) {
        if (n == 0) return true;

        if (Ztrtri.ztrtri(uplo, BLAS.Diag.NonUnit, n, A, aOff, lda) != 0) {
            return false;
        }

        Zlauum.zlauum(uplo, n, A, aOff, lda);
        return true;
    }

    static double zpocon(BLAS.Uplo uplo, int n, double[] A, int aOff, int lda, double anorm,
                         double[] work, int[] iwork) {
        if (n == 0) return 1.0;
        if (anorm == 0) return 0.0;

        boolean upper = uplo == BLAS.Uplo.Upper;
        double smlnum = BLAS.dlamch('S');
        double rcond = 0.0;
        double ainvnm = 0.0;
        boolean normin = false;

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
            if (upper) {
                sl = Zlatrs.zlatrs(BLAS.Uplo.Upper, BLAS.Trans.Conj, BLAS.Diag.NonUnit, normin, n, A, aOff, lda, work, 0, work, 4 * n);
                normin = true;
                su = Zlatrs.zlatrs(BLAS.Uplo.Upper, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit, normin, n, A, aOff, lda, work, 0, work, 4 * n);
            } else {
                sl = Zlatrs.zlatrs(BLAS.Uplo.Lower, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit, normin, n, A, aOff, lda, work, 0, work, 4 * n);
                normin = true;
                su = Zlatrs.zlatrs(BLAS.Uplo.Lower, BLAS.Trans.Conj, BLAS.Diag.NonUnit, normin, n, A, aOff, lda, work, 0, work, 4 * n);
            }

            double scale = sl * su;

            if (scale != 1) {
                double xmax = Izamax.izamaxAbs(n, work, 0, 1);
                if (scale == 0 || scale < xmax * smlnum) {
                    return rcond;
                }
                Zdrscl.zdrscl(n, scale, work, 0, 1);
            }
        }
    }
}
