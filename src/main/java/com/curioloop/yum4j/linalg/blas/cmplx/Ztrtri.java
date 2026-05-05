/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.linalg.blas.cmplx;

import com.curioloop.yum4j.linalg.blas.BLAS;

import static java.lang.Math.*;

interface Ztrtri {

    int BLOCK_SIZE = 64;

    static int ztrtri(BLAS.Uplo uplo, BLAS.Diag diag, int n, double[] A, int aOff, int lda) {
        boolean upper = uplo == BLAS.Uplo.Upper;
        boolean unit = diag == BLAS.Diag.Unit;

        if (n == 0) return 0;

        if (!unit) {
            for (int i = 0; i < n; i++) {
                int dp = (aOff + i * lda + i) * 2;
                if (A[dp] == 0 && A[dp + 1] == 0) return i + 1;
            }
        }

        if (BLOCK_SIZE <= 1 || n <= BLOCK_SIZE) {
            ztrti2(uplo, diag, n, A, aOff, lda);
            return 0;
        }

        if (upper) {
            for (int j = 0; j < n; j += BLOCK_SIZE) {
                int jb = min(BLOCK_SIZE, n - j);

                Ztrmm.ztrmm(BLAS.Side.Left, BLAS.Uplo.Upper, BLAS.Trans.NoTrans, diag,
                        j, jb, 1.0, 0.0,
                        A, aOff * 2, lda,
                        A, (aOff + j) * 2, lda);
                Ztrsm.ztrsm(BLAS.Side.Right, BLAS.Uplo.Upper, BLAS.Trans.NoTrans, diag,
                        j, jb, -1.0, 0.0,
                        A, (aOff + j * lda + j) * 2, lda,
                        A, (aOff + j) * 2, lda);
                ztrti2(uplo, diag, jb, A, aOff + j * lda + j, lda);
            }
        } else {
            int nn = ((n - 1) / BLOCK_SIZE) * BLOCK_SIZE;
            for (int j = nn; j >= 0; j -= BLOCK_SIZE) {
                int jb = min(BLOCK_SIZE, n - j);

                if (j + jb <= n - 1) {
                    Ztrmm.ztrmm(BLAS.Side.Left, BLAS.Uplo.Lower, BLAS.Trans.NoTrans, diag,
                            n - j - jb, jb, 1.0, 0.0,
                            A, (aOff + (j + jb) * lda + j + jb) * 2, lda,
                            A, (aOff + (j + jb) * lda + j) * 2, lda);
                    Ztrsm.ztrsm(BLAS.Side.Right, BLAS.Uplo.Lower, BLAS.Trans.NoTrans, diag,
                            n - j - jb, jb, -1.0, 0.0,
                            A, (aOff + j * lda + j) * 2, lda,
                            A, (aOff + (j + jb) * lda + j) * 2, lda);
                }
                ztrti2(uplo, diag, jb, A, aOff + j * lda + j, lda);
            }
        }

        return 0;
    }

    static void ztrti2(BLAS.Uplo uplo, BLAS.Diag diag, int n, double[] A, int aOff, int lda) {
        boolean upper = uplo == BLAS.Uplo.Upper;
        boolean unit = diag == BLAS.Diag.Unit;

        if (upper) {
            for (int j = 0; j < n; j++) {
                int rowJ = aOff + j * lda;
                if (!unit) {
                    int dp = (rowJ + j) * 2;
                    double re = A[dp], im = A[dp + 1];
                    double denom = re * re + im * im;
                    A[dp] = re / denom;
                    A[dp + 1] = -im / denom;
                }
                double ajjRe = unit ? -1.0 : -A[(rowJ + j) * 2];
                double ajjIm = unit ? 0.0 : -A[(rowJ + j) * 2 + 1];

                if (j > 0) {
                    Ztrmv.ztrmv(BLAS.Uplo.Upper, BLAS.Trans.NoTrans, diag, j,
                            A, aOff * 2, lda, A, (aOff + j) * 2, lda);
                    Zscal.zscal(j, ajjRe, ajjIm, A, (aOff + j) * 2, lda);
                }
            }
        } else {
            for (int j = n - 1; j >= 0; j--) {
                int rowJ = aOff + j * lda;
                if (!unit) {
                    int dp = (rowJ + j) * 2;
                    double re = A[dp], im = A[dp + 1];
                    double denom = re * re + im * im;
                    A[dp] = re / denom;
                    A[dp + 1] = -im / denom;
                }
                double ajjRe = unit ? -1.0 : -A[(rowJ + j) * 2];
                double ajjIm = unit ? 0.0 : -A[(rowJ + j) * 2 + 1];

                if (j < n - 1) {
                    Ztrmv.ztrmv(BLAS.Uplo.Lower, BLAS.Trans.NoTrans, diag, n - j - 1,
                            A, (aOff + (j + 1) * lda + j + 1) * 2, lda,
                            A, (aOff + (j + 1) * lda + j) * 2, lda);
                    Zscal.zscal(n - j - 1, ajjRe, ajjIm, A, (aOff + (j + 1) * lda + j) * 2, lda);
                }
            }
        }
    }

    static double ztrcon(char norm, BLAS.Uplo uplo, BLAS.Diag diag, int n,
                         double[] A, int aOff, int lda,
                         double[] work, int[] iwork) {
        if (n == 0) return 1.0;

        double smlnum = BLAS.dlamch('S') * n;
        double rcond = 0.0;

        double anorm = zlantr(norm, uplo, diag, n, A, aOff, lda);
        if (anorm <= 0) return rcond;

        double ainvnm = 0.0;
        boolean normin = false;
        int kase1 = (norm == '1' || norm == 'O' || norm == 'o') ? 1 : 2;

        int[] isave = new int[3];
        int[] kase = {0};

        while (true) {
            ainvnm = Zlacn2.zlacn2(n, work, n * 2, work, 0, iwork, 0, ainvnm, kase, 0, isave, 0);

            if (kase[0] == 0) {
                if (ainvnm != 0) {
                    rcond = (1.0 / anorm) / ainvnm;
                }
                return rcond;
            }

            double scale;
            if (kase[0] == kase1) {
                scale = Zlatrs.zlatrs(uplo, BLAS.Trans.Conj, diag, normin, n, A, aOff, lda, work, 0, work, n * 2);
            } else {
                scale = Zlatrs.zlatrs(uplo, BLAS.Trans.NoTrans, diag, normin, n, A, aOff, lda, work, 0, work, n * 2);
            }

            normin = true;

            if (scale != 1) {
                double xnorm = Izamax.izamaxAbs(n, work, 0, 1);
                if (scale == 0 || scale < xnorm * smlnum) {
                    return rcond;
                }
                Zdrscl.zdrscl(n, scale, work, 0, 1);
            }
        }
    }

    static double zlantr(char norm, BLAS.Uplo uplo, BLAS.Diag diag, int n,
                         double[] A, int aOff, int lda) {
        double value = 0;
        boolean upper = uplo == BLAS.Uplo.Upper;
        boolean unit = diag == BLAS.Diag.Unit;

        if (norm == '1' || norm == 'O' || norm == 'o') {
            for (int j = 0; j < n; j++) {
                double sum = 0;
                if (upper) {
                    if (unit) {
                        sum = 1;
                        for (int i = 0; i < j; i++) {
                            int p = (aOff + i * lda + j) * 2;
                            sum += Math.hypot(A[p], A[p + 1]);
                        }
                    } else {
                        for (int i = 0; i <= j; i++) {
                            int p = (aOff + i * lda + j) * 2;
                            sum += Math.hypot(A[p], A[p + 1]);
                        }
                    }
                } else {
                    if (unit) {
                        sum = 1;
                        for (int i = j + 1; i < n; i++) {
                            int p = (aOff + i * lda + j) * 2;
                            sum += Math.hypot(A[p], A[p + 1]);
                        }
                    } else {
                        for (int i = j; i < n; i++) {
                            int p = (aOff + i * lda + j) * 2;
                            sum += Math.hypot(A[p], A[p + 1]);
                        }
                    }
                }
                value = max(value, sum);
            }
        } else if (norm == 'I') {
            if (upper) {
                for (int i = 0; i < n; i++) {
                    double sum = unit ? 1 : 0;
                    if (!unit) {
                        int dp = (aOff + i * lda + i) * 2;
                        sum = Math.hypot(A[dp], A[dp + 1]);
                    }
                    for (int j = i + 1; j < n; j++) {
                        int p = (aOff + i * lda + j) * 2;
                        sum += Math.hypot(A[p], A[p + 1]);
                    }
                    value = max(value, sum);
                }
            } else {
                for (int i = 0; i < n; i++) {
                    double sum = 0;
                    for (int j = 0; j < i; j++) {
                        int p = (aOff + i * lda + j) * 2;
                        sum += Math.hypot(A[p], A[p + 1]);
                    }
                    if (unit) {
                        sum += 1;
                    } else {
                        int dp = (aOff + i * lda + i) * 2;
                        sum += Math.hypot(A[dp], A[dp + 1]);
                    }
                    value = max(value, sum);
                }
            }
        } else if (norm == 'F') {
            double sum = 0;
            if (upper) {
                if (unit) {
                    sum = n;
                    for (int i = 0; i < n; i++) {
                        for (int j = i + 1; j < n; j++) {
                            int p = (aOff + i * lda + j) * 2;
                            sum += A[p] * A[p] + A[p + 1] * A[p + 1];
                        }
                    }
                } else {
                    for (int i = 0; i < n; i++) {
                        int dp = (aOff + i * lda + i) * 2;
                        sum += A[dp] * A[dp] + A[dp + 1] * A[dp + 1];
                        for (int j = i + 1; j < n; j++) {
                            int p = (aOff + i * lda + j) * 2;
                            sum += A[p] * A[p] + A[p + 1] * A[p + 1];
                        }
                    }
                }
            } else {
                if (unit) {
                    sum = n;
                    for (int i = 0; i < n; i++) {
                        for (int j = 0; j < i; j++) {
                            int p = (aOff + i * lda + j) * 2;
                            sum += A[p] * A[p] + A[p + 1] * A[p + 1];
                        }
                    }
                } else {
                    for (int i = 0; i < n; i++) {
                        for (int j = 0; j <= i; j++) {
                            int p = (aOff + i * lda + j) * 2;
                            sum += A[p] * A[p] + A[p + 1] * A[p + 1];
                        }
                    }
                }
            }
            value = sqrt(sum);
        }

        return value;
    }
}
