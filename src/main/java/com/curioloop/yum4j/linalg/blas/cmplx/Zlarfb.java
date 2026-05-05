/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.linalg.blas.cmplx;
import com.curioloop.yum4j.linalg.blas.BLAS;
interface Zlarfb {

    static void zlarfb(char side, char trans, char direct, char storev,
                       int m, int n, int k,
                       double[] V, int vOff, int ldv,
                       double[] T, int tOff, int ldt,
                       double[] C, int cOff, int ldc,
                       double[] work, int wOff, int ldwork) {
        boolean left = side == 'L' || side == 'l';
        boolean notrans = trans == 'N' || trans == 'n';
        boolean forward = direct == 'F' || direct == 'f';
        boolean colwise = storev == 'C' || storev == 'c';

        if (colwise) {
            if (forward) {
                if (left) {
                    if (notrans) {
                        zlarfbLeftForwardColWise(V, vOff, ldv, T, tOff, ldt, C, cOff, ldc, work, wOff, ldwork, m, n, k);
                    } else {
                        zlarfbLeftConjTransForwardColWise(V, vOff, ldv, T, tOff, ldt, C, cOff, ldc, work, wOff, ldwork, m, n, k);
                    }
                } else {
                    if (notrans) {
                        zlarfbRightForwardColWise(V, vOff, ldv, T, tOff, ldt, C, cOff, ldc, work, wOff, ldwork, m, n, k);
                    } else {
                        zlarfbRightConjTransForwardColWise(V, vOff, ldv, T, tOff, ldt, C, cOff, ldc, work, wOff, ldwork, m, n, k);
                    }
                }
            } else {
                if (left) {
                    if (notrans) {
                        zlarfbLeftBackwardColWise(V, vOff, ldv, T, tOff, ldt, C, cOff, ldc, work, wOff, ldwork, m, n, k);
                    } else {
                        zlarfbLeftConjTransBackwardColWise(V, vOff, ldv, T, tOff, ldt, C, cOff, ldc, work, wOff, ldwork, m, n, k);
                    }
                } else {
                    if (notrans) {
                        zlarfbRightBackwardColWise(V, vOff, ldv, T, tOff, ldt, C, cOff, ldc, work, wOff, ldwork, m, n, k);
                    } else {
                        zlarfbRightConjTransBackwardColWise(V, vOff, ldv, T, tOff, ldt, C, cOff, ldc, work, wOff, ldwork, m, n, k);
                    }
                }
            }
        } else {
            if (forward) {
                if (left) {
                    if (notrans) {
                        zlarfbLeftForwardRowWise(V, vOff, ldv, T, tOff, ldt, C, cOff, ldc, work, wOff, ldwork, m, n, k);
                    } else {
                        zlarfbLeftConjTransForwardRowWise(V, vOff, ldv, T, tOff, ldt, C, cOff, ldc, work, wOff, ldwork, m, n, k);
                    }
                } else {
                    if (notrans) {
                        zlarfbRightForwardRowWise(V, vOff, ldv, T, tOff, ldt, C, cOff, ldc, work, wOff, ldwork, m, n, k);
                    } else {
                        zlarfbRightConjTransForwardRowWise(V, vOff, ldv, T, tOff, ldt, C, cOff, ldc, work, wOff, ldwork, m, n, k);
                    }
                }
            } else {
                if (left) {
                    if (notrans) {
                        zlarfbLeftBackwardRowWise(V, vOff, ldv, T, tOff, ldt, C, cOff, ldc, work, wOff, ldwork, m, n, k);
                    } else {
                        zlarfbLeftConjTransBackwardRowWise(V, vOff, ldv, T, tOff, ldt, C, cOff, ldc, work, wOff, ldwork, m, n, k);
                    }
                } else {
                    if (notrans) {
                        zlarfbRightBackwardRowWise(V, vOff, ldv, T, tOff, ldt, C, cOff, ldc, work, wOff, ldwork, m, n, k);
                    } else {
                        zlarfbRightConjTransBackwardRowWise(V, vOff, ldv, T, tOff, ldt, C, cOff, ldc, work, wOff, ldwork, m, n, k);
                    }
                }
            }
        }
    }

    // ==================== ColumnWise + Forward ====================

    static void zlarfbLeftForwardColWise(double[] V, int vOff, int ldv,
                                          double[] T, int tOff, int ldt,
                                          double[] C, int cOff, int ldc,
                                          double[] work, int wOff, int ldwork, int m, int n, int k) {
        if (m <= 0 || n <= 0 || k <= 0) return;

        for (int j = 0; j < k; j++) {
            for (int i = 0; i < n; i++) {
                int cPos = (cOff + j * ldc + i) * 2;
                int wPos = (wOff + i * ldwork + j) * 2;
                work[wPos] = C[cPos];
                work[wPos + 1] = C[cPos + 1];
            }
            Zlacgv.zlacgv(n, work, wOff + j, ldwork);
        }

        Ztrmm.ztrmm(BLAS.Side.Right, BLAS.Uplo.Lower, BLAS.Trans.NoTrans, BLAS.Diag.Unit, n, k, 1.0, 0.0, V, vOff * 2, ldv, work, wOff * 2, ldwork);

        if (m > k) {
            Zgemm.zgemm(BLAS.Trans.Conj, BLAS.Trans.NoTrans, n, k, m - k, 1.0, 0.0, C, cOff + k * ldc, ldc, V, vOff + k * ldv, ldv, 1.0, 0.0, work, wOff, ldwork);
        }

        Ztrmm.ztrmm(BLAS.Side.Right, BLAS.Uplo.Upper, BLAS.Trans.Conj, BLAS.Diag.NonUnit, n, k, 1.0, 0.0, T, tOff * 2, ldt, work, wOff * 2, ldwork);

        if (m > k) {
            Zgemm.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.Conj, m - k, n, k, -1.0, 0.0, V, vOff + k * ldv, ldv, work, wOff, ldwork, 1.0, 0.0, C, cOff + k * ldc, ldc);
        }

        Ztrmm.ztrmm(BLAS.Side.Right, BLAS.Uplo.Lower, BLAS.Trans.Conj, BLAS.Diag.Unit, n, k, 1.0, 0.0, V, vOff * 2, ldv, work, wOff * 2, ldwork);

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < k; j++) {
                int cPos = (cOff + j * ldc + i) * 2;
                int wPos = (wOff + i * ldwork + j) * 2;
                C[cPos] -= work[wPos];
                C[cPos + 1] -= -work[wPos + 1];
            }
        }
    }

    static void zlarfbLeftConjTransForwardColWise(double[] V, int vOff, int ldv,
                                                   double[] T, int tOff, int ldt,
                                                   double[] C, int cOff, int ldc,
                                                   double[] work, int wOff, int ldwork, int m, int n, int k) {
        if (m <= 0 || n <= 0 || k <= 0) return;

        for (int j = 0; j < k; j++) {
            for (int i = 0; i < n; i++) {
                int cPos = (cOff + j * ldc + i) * 2;
                int wPos = (wOff + i * ldwork + j) * 2;
                work[wPos] = C[cPos];
                work[wPos + 1] = C[cPos + 1];
            }
            Zlacgv.zlacgv(n, work, wOff + j, ldwork);
        }

        Ztrmm.ztrmm(BLAS.Side.Right, BLAS.Uplo.Lower, BLAS.Trans.NoTrans, BLAS.Diag.Unit, n, k, 1.0, 0.0, V, vOff * 2, ldv, work, wOff * 2, ldwork);

        if (m > k) {
            Zgemm.zgemm(BLAS.Trans.Conj, BLAS.Trans.NoTrans, n, k, m - k, 1.0, 0.0, C, cOff + k * ldc, ldc, V, vOff + k * ldv, ldv, 1.0, 0.0, work, wOff, ldwork);
        }

        Ztrmm.ztrmm(BLAS.Side.Right, BLAS.Uplo.Upper, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit, n, k, 1.0, 0.0, T, tOff * 2, ldt, work, wOff * 2, ldwork);

        if (m > k) {
            Zgemm.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.Conj, m - k, n, k, -1.0, 0.0, V, vOff + k * ldv, ldv, work, wOff, ldwork, 1.0, 0.0, C, cOff + k * ldc, ldc);
        }

        Ztrmm.ztrmm(BLAS.Side.Right, BLAS.Uplo.Lower, BLAS.Trans.Conj, BLAS.Diag.Unit, n, k, 1.0, 0.0, V, vOff * 2, ldv, work, wOff * 2, ldwork);

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < k; j++) {
                int cPos = (cOff + j * ldc + i) * 2;
                int wPos = (wOff + i * ldwork + j) * 2;
                C[cPos] -= work[wPos];
                C[cPos + 1] -= -work[wPos + 1];
            }
        }
    }

    static void zlarfbRightForwardColWise(double[] V, int vOff, int ldv,
                                           double[] T, int tOff, int ldt,
                                           double[] C, int cOff, int ldc,
                                           double[] work, int wOff, int ldwork, int m, int n, int k) {
        if (m <= 0 || n <= 0 || k <= 0) return;

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < k; j++) {
                int cPos = (cOff + i * ldc + j) * 2;
                int wPos = (wOff + i * ldwork + j) * 2;
                work[wPos] = C[cPos];
                work[wPos + 1] = C[cPos + 1];
            }
        }

        Ztrmm.ztrmm(BLAS.Side.Right, BLAS.Uplo.Lower, BLAS.Trans.NoTrans, BLAS.Diag.Unit, m, k, 1.0, 0.0, V, vOff * 2, ldv, work, wOff * 2, ldwork);

        if (n > k) {
            Zgemm.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, m, k, n - k, 1.0, 0.0, C, cOff + k, ldc, V, vOff + k * ldv, ldv, 1.0, 0.0, work, wOff, ldwork);
        }

        Ztrmm.ztrmm(BLAS.Side.Right, BLAS.Uplo.Upper, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit, m, k, 1.0, 0.0, T, tOff * 2, ldt, work, wOff * 2, ldwork);

        if (n > k) {
            Zgemm.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.Conj, m, n - k, k, -1.0, 0.0, work, wOff, ldwork, V, vOff + k * ldv, ldv, 1.0, 0.0, C, cOff + k, ldc);
        }

        Ztrmm.ztrmm(BLAS.Side.Right, BLAS.Uplo.Lower, BLAS.Trans.Conj, BLAS.Diag.Unit, m, k, 1.0, 0.0, V, vOff * 2, ldv, work, wOff * 2, ldwork);

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < k; j++) {
                int cPos = (cOff + i * ldc + j) * 2;
                int wPos = (wOff + i * ldwork + j) * 2;
                C[cPos] -= work[wPos];
                C[cPos + 1] -= work[wPos + 1];
            }
        }
    }

    static void zlarfbRightConjTransForwardColWise(double[] V, int vOff, int ldv,
                                                    double[] T, int tOff, int ldt,
                                                    double[] C, int cOff, int ldc,
                                                    double[] work, int wOff, int ldwork, int m, int n, int k) {
        if (m <= 0 || n <= 0 || k <= 0) return;

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < k; j++) {
                int cPos = (cOff + i * ldc + j) * 2;
                int wPos = (wOff + i * ldwork + j) * 2;
                work[wPos] = C[cPos];
                work[wPos + 1] = C[cPos + 1];
            }
        }

        Ztrmm.ztrmm(BLAS.Side.Right, BLAS.Uplo.Lower, BLAS.Trans.NoTrans, BLAS.Diag.Unit, m, k, 1.0, 0.0, V, vOff * 2, ldv, work, wOff * 2, ldwork);

        if (n > k) {
            Zgemm.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, m, k, n - k, 1.0, 0.0, C, cOff + k, ldc, V, vOff + k * ldv, ldv, 1.0, 0.0, work, wOff, ldwork);
        }

        Ztrmm.ztrmm(BLAS.Side.Right, BLAS.Uplo.Upper, BLAS.Trans.Conj, BLAS.Diag.NonUnit, m, k, 1.0, 0.0, T, tOff * 2, ldt, work, wOff * 2, ldwork);

        if (n > k) {
            Zgemm.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.Conj, m, n - k, k, -1.0, 0.0, work, wOff, ldwork, V, vOff + k * ldv, ldv, 1.0, 0.0, C, cOff + k, ldc);
        }

        Ztrmm.ztrmm(BLAS.Side.Right, BLAS.Uplo.Lower, BLAS.Trans.Conj, BLAS.Diag.Unit, m, k, 1.0, 0.0, V, vOff * 2, ldv, work, wOff * 2, ldwork);

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < k; j++) {
                int cPos = (cOff + i * ldc + j) * 2;
                int wPos = (wOff + i * ldwork + j) * 2;
                C[cPos] -= work[wPos];
                C[cPos + 1] -= work[wPos + 1];
            }
        }
    }

    // ==================== ColumnWise + Backward ====================

    static void zlarfbLeftBackwardColWise(double[] V, int vOff, int ldv,
                                           double[] T, int tOff, int ldt,
                                           double[] C, int cOff, int ldc,
                                           double[] work, int wOff, int ldwork, int m, int n, int k) {
        if (m <= 0 || n <= 0 || k <= 0) return;

        int v2Off = vOff + (m - k) * ldv;

        for (int j = 0; j < k; j++) {
            for (int i = 0; i < n; i++) {
                int cPos = (cOff + (m - k + j) * ldc + i) * 2;
                int wPos = (wOff + i * ldwork + j) * 2;
                work[wPos] = C[cPos];
                work[wPos + 1] = C[cPos + 1];
            }
            Zlacgv.zlacgv(n, work, wOff + j, ldwork);
        }

        Ztrmm.ztrmm(BLAS.Side.Right, BLAS.Uplo.Upper, BLAS.Trans.NoTrans, BLAS.Diag.Unit, n, k, 1.0, 0.0, V, v2Off * 2, ldv, work, wOff * 2, ldwork);

        if (m > k) {
            Zgemm.zgemm(BLAS.Trans.Conj, BLAS.Trans.NoTrans, n, k, m - k, 1.0, 0.0, C, cOff, ldc, V, vOff, ldv, 1.0, 0.0, work, wOff, ldwork);
        }

        Ztrmm.ztrmm(BLAS.Side.Right, BLAS.Uplo.Lower, BLAS.Trans.Conj, BLAS.Diag.NonUnit, n, k, 1.0, 0.0, T, tOff * 2, ldt, work, wOff * 2, ldwork);

        if (m > k) {
            Zgemm.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.Conj, m - k, n, k, -1.0, 0.0, V, vOff, ldv, work, wOff, ldwork, 1.0, 0.0, C, cOff, ldc);
        }

        Ztrmm.ztrmm(BLAS.Side.Right, BLAS.Uplo.Upper, BLAS.Trans.Conj, BLAS.Diag.Unit, n, k, 1.0, 0.0, V, v2Off * 2, ldv, work, wOff * 2, ldwork);

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < k; j++) {
                int cPos = (cOff + (m - k + j) * ldc + i) * 2;
                int wPos = (wOff + i * ldwork + j) * 2;
                C[cPos] -= work[wPos];
                C[cPos + 1] -= -work[wPos + 1];
            }
        }
    }

    static void zlarfbLeftConjTransBackwardColWise(double[] V, int vOff, int ldv,
                                                    double[] T, int tOff, int ldt,
                                                    double[] C, int cOff, int ldc,
                                                    double[] work, int wOff, int ldwork, int m, int n, int k) {
        if (m <= 0 || n <= 0 || k <= 0) return;

        int v2Off = vOff + (m - k) * ldv;

        for (int j = 0; j < k; j++) {
            for (int i = 0; i < n; i++) {
                int cPos = (cOff + (m - k + j) * ldc + i) * 2;
                int wPos = (wOff + i * ldwork + j) * 2;
                work[wPos] = C[cPos];
                work[wPos + 1] = C[cPos + 1];
            }
            Zlacgv.zlacgv(n, work, wOff + j, ldwork);
        }

        Ztrmm.ztrmm(BLAS.Side.Right, BLAS.Uplo.Upper, BLAS.Trans.NoTrans, BLAS.Diag.Unit, n, k, 1.0, 0.0, V, v2Off * 2, ldv, work, wOff * 2, ldwork);

        if (m > k) {
            Zgemm.zgemm(BLAS.Trans.Conj, BLAS.Trans.NoTrans, n, k, m - k, 1.0, 0.0, C, cOff, ldc, V, vOff, ldv, 1.0, 0.0, work, wOff, ldwork);
        }

        Ztrmm.ztrmm(BLAS.Side.Right, BLAS.Uplo.Lower, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit, n, k, 1.0, 0.0, T, tOff * 2, ldt, work, wOff * 2, ldwork);

        if (m > k) {
            Zgemm.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.Conj, m - k, n, k, -1.0, 0.0, V, vOff, ldv, work, wOff, ldwork, 1.0, 0.0, C, cOff, ldc);
        }

        Ztrmm.ztrmm(BLAS.Side.Right, BLAS.Uplo.Upper, BLAS.Trans.Conj, BLAS.Diag.Unit, n, k, 1.0, 0.0, V, v2Off * 2, ldv, work, wOff * 2, ldwork);

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < k; j++) {
                int cPos = (cOff + (m - k + j) * ldc + i) * 2;
                int wPos = (wOff + i * ldwork + j) * 2;
                C[cPos] -= work[wPos];
                C[cPos + 1] -= -work[wPos + 1];
            }
        }
    }

    static void zlarfbRightBackwardColWise(double[] V, int vOff, int ldv,
                                            double[] T, int tOff, int ldt,
                                            double[] C, int cOff, int ldc,
                                            double[] work, int wOff, int ldwork, int m, int n, int k) {
        if (m <= 0 || n <= 0 || k <= 0) return;

        int v2Off = vOff + (n - k) * ldv;

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < k; j++) {
                int cPos = (cOff + i * ldc + (n - k + j)) * 2;
                int wPos = (wOff + i * ldwork + j) * 2;
                work[wPos] = C[cPos];
                work[wPos + 1] = C[cPos + 1];
            }
        }

        Ztrmm.ztrmm(BLAS.Side.Right, BLAS.Uplo.Upper, BLAS.Trans.NoTrans, BLAS.Diag.Unit, m, k, 1.0, 0.0, V, v2Off * 2, ldv, work, wOff * 2, ldwork);

        if (n > k) {
            Zgemm.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, m, k, n - k, 1.0, 0.0, C, cOff, ldc, V, vOff, ldv, 1.0, 0.0, work, wOff, ldwork);
        }

        Ztrmm.ztrmm(BLAS.Side.Right, BLAS.Uplo.Lower, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit, m, k, 1.0, 0.0, T, tOff * 2, ldt, work, wOff * 2, ldwork);

        if (n > k) {
            Zgemm.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.Conj, m, n - k, k, -1.0, 0.0, work, wOff, ldwork, V, vOff, ldv, 1.0, 0.0, C, cOff, ldc);
        }

        Ztrmm.ztrmm(BLAS.Side.Right, BLAS.Uplo.Upper, BLAS.Trans.Conj, BLAS.Diag.Unit, m, k, 1.0, 0.0, V, v2Off * 2, ldv, work, wOff * 2, ldwork);

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < k; j++) {
                int cPos = (cOff + i * ldc + (n - k + j)) * 2;
                int wPos = (wOff + i * ldwork + j) * 2;
                C[cPos] -= work[wPos];
                C[cPos + 1] -= work[wPos + 1];
            }
        }
    }

    static void zlarfbRightConjTransBackwardColWise(double[] V, int vOff, int ldv,
                                                     double[] T, int tOff, int ldt,
                                                     double[] C, int cOff, int ldc,
                                                     double[] work, int wOff, int ldwork, int m, int n, int k) {
        if (m <= 0 || n <= 0 || k <= 0) return;

        int v2Off = vOff + (n - k) * ldv;

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < k; j++) {
                int cPos = (cOff + i * ldc + (n - k + j)) * 2;
                int wPos = (wOff + i * ldwork + j) * 2;
                work[wPos] = C[cPos];
                work[wPos + 1] = C[cPos + 1];
            }
        }

        Ztrmm.ztrmm(BLAS.Side.Right, BLAS.Uplo.Upper, BLAS.Trans.NoTrans, BLAS.Diag.Unit, m, k, 1.0, 0.0, V, v2Off * 2, ldv, work, wOff * 2, ldwork);

        if (n > k) {
            Zgemm.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, m, k, n - k, 1.0, 0.0, C, cOff, ldc, V, vOff, ldv, 1.0, 0.0, work, wOff, ldwork);
        }

        Ztrmm.ztrmm(BLAS.Side.Right, BLAS.Uplo.Lower, BLAS.Trans.Conj, BLAS.Diag.NonUnit, m, k, 1.0, 0.0, T, tOff * 2, ldt, work, wOff * 2, ldwork);

        if (n > k) {
            Zgemm.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.Conj, m, n - k, k, -1.0, 0.0, work, wOff, ldwork, V, vOff, ldv, 1.0, 0.0, C, cOff, ldc);
        }

        Ztrmm.ztrmm(BLAS.Side.Right, BLAS.Uplo.Upper, BLAS.Trans.Conj, BLAS.Diag.Unit, m, k, 1.0, 0.0, V, v2Off * 2, ldv, work, wOff * 2, ldwork);

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < k; j++) {
                int cPos = (cOff + i * ldc + (n - k + j)) * 2;
                int wPos = (wOff + i * ldwork + j) * 2;
                C[cPos] -= work[wPos];
                C[cPos + 1] -= work[wPos + 1];
            }
        }
    }

    // ==================== RowWise + Forward ====================

    static void zlarfbLeftForwardRowWise(double[] V, int vOff, int ldv,
                                          double[] T, int tOff, int ldt,
                                          double[] C, int cOff, int ldc,
                                          double[] work, int wOff, int ldwork, int m, int n, int k) {
        if (m <= 0 || n <= 0 || k <= 0) return;

        for (int j = 0; j < k; j++) {
            for (int i = 0; i < n; i++) {
                int cPos = (cOff + j * ldc + i) * 2;
                int wPos = (wOff + i * ldwork + j) * 2;
                work[wPos] = C[cPos];
                work[wPos + 1] = C[cPos + 1];
            }
            Zlacgv.zlacgv(n, work, wOff + j, ldwork);
        }

        Ztrmm.ztrmm(BLAS.Side.Right, BLAS.Uplo.Upper, BLAS.Trans.Conj, BLAS.Diag.Unit, n, k, 1.0, 0.0, V, vOff * 2, ldv, work, wOff * 2, ldwork);

        if (m > k) {
            Zgemm.zgemm(BLAS.Trans.Conj, BLAS.Trans.Conj, n, k, m - k, 1.0, 0.0, C, cOff + k * ldc, ldc, V, vOff + k, ldv, 1.0, 0.0, work, wOff, ldwork);
        }

        Ztrmm.ztrmm(BLAS.Side.Right, BLAS.Uplo.Upper, BLAS.Trans.Conj, BLAS.Diag.NonUnit, n, k, 1.0, 0.0, T, tOff * 2, ldt, work, wOff * 2, ldwork);

        if (m > k) {
            Zgemm.zgemm(BLAS.Trans.Conj, BLAS.Trans.Conj, m - k, n, k, -1.0, 0.0, V, vOff + k, ldv, work, wOff, ldwork, 1.0, 0.0, C, cOff + k * ldc, ldc);
        }

        Ztrmm.ztrmm(BLAS.Side.Right, BLAS.Uplo.Upper, BLAS.Trans.NoTrans, BLAS.Diag.Unit, n, k, 1.0, 0.0, V, vOff * 2, ldv, work, wOff * 2, ldwork);

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < k; j++) {
                int cPos = (cOff + j * ldc + i) * 2;
                int wPos = (wOff + i * ldwork + j) * 2;
                C[cPos] -= work[wPos];
                C[cPos + 1] -= -work[wPos + 1];
            }
        }
    }

    static void zlarfbLeftConjTransForwardRowWise(double[] V, int vOff, int ldv,
                                                   double[] T, int tOff, int ldt,
                                                   double[] C, int cOff, int ldc,
                                                   double[] work, int wOff, int ldwork, int m, int n, int k) {
        if (m <= 0 || n <= 0 || k <= 0) return;

        for (int j = 0; j < k; j++) {
            for (int i = 0; i < n; i++) {
                int cPos = (cOff + j * ldc + i) * 2;
                int wPos = (wOff + i * ldwork + j) * 2;
                work[wPos] = C[cPos];
                work[wPos + 1] = C[cPos + 1];
            }
            Zlacgv.zlacgv(n, work, wOff + j, ldwork);
        }

        Ztrmm.ztrmm(BLAS.Side.Right, BLAS.Uplo.Upper, BLAS.Trans.Conj, BLAS.Diag.Unit, n, k, 1.0, 0.0, V, vOff * 2, ldv, work, wOff * 2, ldwork);

        if (m > k) {
            Zgemm.zgemm(BLAS.Trans.Conj, BLAS.Trans.Conj, n, k, m - k, 1.0, 0.0, C, cOff + k * ldc, ldc, V, vOff + k, ldv, 1.0, 0.0, work, wOff, ldwork);
        }

        Ztrmm.ztrmm(BLAS.Side.Right, BLAS.Uplo.Upper, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit, n, k, 1.0, 0.0, T, tOff * 2, ldt, work, wOff * 2, ldwork);

        if (m > k) {
            Zgemm.zgemm(BLAS.Trans.Conj, BLAS.Trans.Conj, m - k, n, k, -1.0, 0.0, V, vOff + k, ldv, work, wOff, ldwork, 1.0, 0.0, C, cOff + k * ldc, ldc);
        }

        Ztrmm.ztrmm(BLAS.Side.Right, BLAS.Uplo.Upper, BLAS.Trans.NoTrans, BLAS.Diag.Unit, n, k, 1.0, 0.0, V, vOff * 2, ldv, work, wOff * 2, ldwork);

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < k; j++) {
                int cPos = (cOff + j * ldc + i) * 2;
                int wPos = (wOff + i * ldwork + j) * 2;
                C[cPos] -= work[wPos];
                C[cPos + 1] -= -work[wPos + 1];
            }
        }
    }

    static void zlarfbRightForwardRowWise(double[] V, int vOff, int ldv,
                                           double[] T, int tOff, int ldt,
                                           double[] C, int cOff, int ldc,
                                           double[] work, int wOff, int ldwork, int m, int n, int k) {
        if (m <= 0 || n <= 0 || k <= 0) return;

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < k; j++) {
                int cPos = (cOff + i * ldc + j) * 2;
                int wPos = (wOff + i * ldwork + j) * 2;
                work[wPos] = C[cPos];
                work[wPos + 1] = C[cPos + 1];
            }
        }

        Ztrmm.ztrmm(BLAS.Side.Right, BLAS.Uplo.Upper, BLAS.Trans.Conj, BLAS.Diag.Unit, m, k, 1.0, 0.0, V, vOff * 2, ldv, work, wOff * 2, ldwork);

        if (n > k) {
            Zgemm.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.Conj, m, k, n - k, 1.0, 0.0, C, cOff + k, ldc, V, vOff + k, ldv, 1.0, 0.0, work, wOff, ldwork);
        }

        Ztrmm.ztrmm(BLAS.Side.Right, BLAS.Uplo.Upper, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit, m, k, 1.0, 0.0, T, tOff * 2, ldt, work, wOff * 2, ldwork);

        if (n > k) {
            Zgemm.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, m, n - k, k, -1.0, 0.0, work, wOff, ldwork, V, vOff + k, ldv, 1.0, 0.0, C, cOff + k, ldc);
        }

        Ztrmm.ztrmm(BLAS.Side.Right, BLAS.Uplo.Upper, BLAS.Trans.NoTrans, BLAS.Diag.Unit, m, k, 1.0, 0.0, V, vOff * 2, ldv, work, wOff * 2, ldwork);

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < k; j++) {
                int cPos = (cOff + i * ldc + j) * 2;
                int wPos = (wOff + i * ldwork + j) * 2;
                C[cPos] -= work[wPos];
                C[cPos + 1] -= work[wPos + 1];
            }
        }
    }

    static void zlarfbRightConjTransForwardRowWise(double[] V, int vOff, int ldv,
                                                    double[] T, int tOff, int ldt,
                                                    double[] C, int cOff, int ldc,
                                                    double[] work, int wOff, int ldwork, int m, int n, int k) {
        if (m <= 0 || n <= 0 || k <= 0) return;

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < k; j++) {
                int cPos = (cOff + i * ldc + j) * 2;
                int wPos = (wOff + i * ldwork + j) * 2;
                work[wPos] = C[cPos];
                work[wPos + 1] = C[cPos + 1];
            }
        }

        Ztrmm.ztrmm(BLAS.Side.Right, BLAS.Uplo.Upper, BLAS.Trans.Conj, BLAS.Diag.Unit, m, k, 1.0, 0.0, V, vOff * 2, ldv, work, wOff * 2, ldwork);

        if (n > k) {
            Zgemm.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.Conj, m, k, n - k, 1.0, 0.0, C, cOff + k, ldc, V, vOff + k, ldv, 1.0, 0.0, work, wOff, ldwork);
        }

        Ztrmm.ztrmm(BLAS.Side.Right, BLAS.Uplo.Upper, BLAS.Trans.Conj, BLAS.Diag.NonUnit, m, k, 1.0, 0.0, T, tOff * 2, ldt, work, wOff * 2, ldwork);

        if (n > k) {
            Zgemm.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, m, n - k, k, -1.0, 0.0, work, wOff, ldwork, V, vOff + k, ldv, 1.0, 0.0, C, cOff + k, ldc);
        }

        Ztrmm.ztrmm(BLAS.Side.Right, BLAS.Uplo.Upper, BLAS.Trans.NoTrans, BLAS.Diag.Unit, m, k, 1.0, 0.0, V, vOff * 2, ldv, work, wOff * 2, ldwork);

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < k; j++) {
                int cPos = (cOff + i * ldc + j) * 2;
                int wPos = (wOff + i * ldwork + j) * 2;
                C[cPos] -= work[wPos];
                C[cPos + 1] -= work[wPos + 1];
            }
        }
    }

    // ==================== RowWise + Backward ====================

    static void zlarfbLeftBackwardRowWise(double[] V, int vOff, int ldv,
                                           double[] T, int tOff, int ldt,
                                           double[] C, int cOff, int ldc,
                                           double[] work, int wOff, int ldwork, int m, int n, int k) {
        if (m <= 0 || n <= 0 || k <= 0) return;

        int v2Off = vOff + (m - k);

        for (int j = 0; j < k; j++) {
            for (int i = 0; i < n; i++) {
                int cPos = (cOff + (m - k + j) * ldc + i) * 2;
                int wPos = (wOff + i * ldwork + j) * 2;
                work[wPos] = C[cPos];
                work[wPos + 1] = C[cPos + 1];
            }
            Zlacgv.zlacgv(n, work, wOff + j, ldwork);
        }

        Ztrmm.ztrmm(BLAS.Side.Right, BLAS.Uplo.Lower, BLAS.Trans.Conj, BLAS.Diag.Unit, n, k, 1.0, 0.0, V, v2Off * 2, ldv, work, wOff * 2, ldwork);

        if (m > k) {
            Zgemm.zgemm(BLAS.Trans.Conj, BLAS.Trans.Conj, n, k, m - k, 1.0, 0.0, C, cOff, ldc, V, vOff, ldv, 1.0, 0.0, work, wOff, ldwork);
        }

        Ztrmm.ztrmm(BLAS.Side.Right, BLAS.Uplo.Lower, BLAS.Trans.Conj, BLAS.Diag.NonUnit, n, k, 1.0, 0.0, T, tOff * 2, ldt, work, wOff * 2, ldwork);

        if (m > k) {
            Zgemm.zgemm(BLAS.Trans.Conj, BLAS.Trans.Conj, m - k, n, k, -1.0, 0.0, V, vOff, ldv, work, wOff, ldwork, 1.0, 0.0, C, cOff, ldc);
        }

        Ztrmm.ztrmm(BLAS.Side.Right, BLAS.Uplo.Lower, BLAS.Trans.NoTrans, BLAS.Diag.Unit, n, k, 1.0, 0.0, V, v2Off * 2, ldv, work, wOff * 2, ldwork);

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < k; j++) {
                int cPos = (cOff + (m - k + j) * ldc + i) * 2;
                int wPos = (wOff + i * ldwork + j) * 2;
                C[cPos] -= work[wPos];
                C[cPos + 1] -= -work[wPos + 1];
            }
        }
    }

    static void zlarfbLeftConjTransBackwardRowWise(double[] V, int vOff, int ldv,
                                                    double[] T, int tOff, int ldt,
                                                    double[] C, int cOff, int ldc,
                                                    double[] work, int wOff, int ldwork, int m, int n, int k) {
        if (m <= 0 || n <= 0 || k <= 0) return;

        int v2Off = vOff + (m - k);

        for (int j = 0; j < k; j++) {
            for (int i = 0; i < n; i++) {
                int cPos = (cOff + (m - k + j) * ldc + i) * 2;
                int wPos = (wOff + i * ldwork + j) * 2;
                work[wPos] = C[cPos];
                work[wPos + 1] = C[cPos + 1];
            }
            Zlacgv.zlacgv(n, work, wOff + j, ldwork);
        }

        Ztrmm.ztrmm(BLAS.Side.Right, BLAS.Uplo.Lower, BLAS.Trans.Conj, BLAS.Diag.Unit, n, k, 1.0, 0.0, V, v2Off * 2, ldv, work, wOff * 2, ldwork);

        if (m > k) {
            Zgemm.zgemm(BLAS.Trans.Conj, BLAS.Trans.Conj, n, k, m - k, 1.0, 0.0, C, cOff, ldc, V, vOff, ldv, 1.0, 0.0, work, wOff, ldwork);
        }

        Ztrmm.ztrmm(BLAS.Side.Right, BLAS.Uplo.Lower, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit, n, k, 1.0, 0.0, T, tOff * 2, ldt, work, wOff * 2, ldwork);

        if (m > k) {
            Zgemm.zgemm(BLAS.Trans.Conj, BLAS.Trans.Conj, m - k, n, k, -1.0, 0.0, V, vOff, ldv, work, wOff, ldwork, 1.0, 0.0, C, cOff, ldc);
        }

        Ztrmm.ztrmm(BLAS.Side.Right, BLAS.Uplo.Lower, BLAS.Trans.NoTrans, BLAS.Diag.Unit, n, k, 1.0, 0.0, V, v2Off * 2, ldv, work, wOff * 2, ldwork);

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < k; j++) {
                int cPos = (cOff + (m - k + j) * ldc + i) * 2;
                int wPos = (wOff + i * ldwork + j) * 2;
                C[cPos] -= work[wPos];
                C[cPos + 1] -= -work[wPos + 1];
            }
        }
    }

    static void zlarfbRightBackwardRowWise(double[] V, int vOff, int ldv,
                                            double[] T, int tOff, int ldt,
                                            double[] C, int cOff, int ldc,
                                            double[] work, int wOff, int ldwork, int m, int n, int k) {
        if (m <= 0 || n <= 0 || k <= 0) return;

        for (int j = 0; j < k; j++) {
            for (int i = 0; i < m; i++) {
                int cPos = (cOff + i * ldc + (n - k + j)) * 2;
                int wPos = (wOff + i * ldwork + j) * 2;
                work[wPos] = C[cPos];
                work[wPos + 1] = C[cPos + 1];
            }
        }

        Ztrmm.ztrmm(BLAS.Side.Right, BLAS.Uplo.Lower, BLAS.Trans.Conj, BLAS.Diag.Unit, m, k, 1.0, 0.0, V, (vOff + (n - k)) * 2, ldv, work, wOff * 2, ldwork);

        if (n > k) {
            Zgemm.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.Conj, m, k, n - k, 1.0, 0.0, C, cOff, ldc, V, vOff, ldv, 1.0, 0.0, work, wOff, ldwork);
        }

        Ztrmm.ztrmm(BLAS.Side.Right, BLAS.Uplo.Lower, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit, m, k, 1.0, 0.0, T, tOff * 2, ldt, work, wOff * 2, ldwork);

        if (n > k) {
            Zgemm.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, m, n - k, k, -1.0, 0.0, work, wOff, ldwork, V, vOff, ldv, 1.0, 0.0, C, cOff, ldc);
        }

        Ztrmm.ztrmm(BLAS.Side.Right, BLAS.Uplo.Lower, BLAS.Trans.NoTrans, BLAS.Diag.Unit, m, k, 1.0, 0.0, V, (vOff + (n - k)) * 2, ldv, work, wOff * 2, ldwork);

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < k; j++) {
                int cPos = (cOff + i * ldc + (n - k + j)) * 2;
                int wPos = (wOff + i * ldwork + j) * 2;
                C[cPos] -= work[wPos];
                C[cPos + 1] -= work[wPos + 1];
            }
        }
    }

    static void zlarfbRightConjTransBackwardRowWise(double[] V, int vOff, int ldv,
                                                     double[] T, int tOff, int ldt,
                                                     double[] C, int cOff, int ldc,
                                                     double[] work, int wOff, int ldwork, int m, int n, int k) {
        if (m <= 0 || n <= 0 || k <= 0) return;

        int v2Off = vOff + (n - k);

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < k; j++) {
                int cPos = (cOff + i * ldc + (n - k + j)) * 2;
                int wPos = (wOff + i * ldwork + j) * 2;
                work[wPos] = C[cPos];
                work[wPos + 1] = C[cPos + 1];
            }
        }

        Ztrmm.ztrmm(BLAS.Side.Right, BLAS.Uplo.Lower, BLAS.Trans.Conj, BLAS.Diag.NonUnit, m, k, 1.0, 0.0, V, v2Off * 2, ldv, work, wOff * 2, ldwork);

        if (n > k) {
            Zgemm.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.Conj, m, k, n - k, 1.0, 0.0, C, cOff, ldc, V, vOff, ldv, 1.0, 0.0, work, wOff, ldwork);
        }

        Ztrmm.ztrmm(BLAS.Side.Right, BLAS.Uplo.Lower, BLAS.Trans.Conj, BLAS.Diag.NonUnit, m, k, 1.0, 0.0, T, tOff * 2, ldt, work, wOff * 2, ldwork);

        if (n > k) {
            Zgemm.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, m, n - k, k, -1.0, 0.0, work, wOff, ldwork, V, vOff, ldv, 1.0, 0.0, C, cOff, ldc);
        }

        Ztrmm.ztrmm(BLAS.Side.Right, BLAS.Uplo.Lower, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit, m, k, 1.0, 0.0, V, v2Off * 2, ldv, work, wOff * 2, ldwork);

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < k; j++) {
                int cPos = (cOff + i * ldc + (n - k + j)) * 2;
                int wPos = (wOff + i * ldwork + j) * 2;
                C[cPos] -= work[wPos];
                C[cPos + 1] -= work[wPos + 1];
            }
        }
    }

}
