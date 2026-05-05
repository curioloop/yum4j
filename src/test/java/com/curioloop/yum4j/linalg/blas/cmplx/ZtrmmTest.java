/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.linalg.blas.cmplx;

import com.curioloop.yum4j.linalg.blas.BLAS;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class ZtrmmTest {

    private static final double TOL = 1e-11;
    private static final int[] SIZES = {1, 2, 3, 4, 8, 16};

    @Test
    void testAllParameterCombinations() {
        long seed = 2026042506L;
        for (BLAS.Side side : new BLAS.Side[]{BLAS.Side.Left, BLAS.Side.Right}) {
            for (BLAS.Uplo uplo : new BLAS.Uplo[]{BLAS.Uplo.Upper, BLAS.Uplo.Lower}) {
                for (BLAS.Trans trans : new BLAS.Trans[]{BLAS.Trans.NoTrans, BLAS.Trans.Trans, BLAS.Trans.Conj}) {
                    for (BLAS.Diag diag : new BLAS.Diag[]{BLAS.Diag.Unit, BLAS.Diag.NonUnit}) {
                        for (int m : SIZES) {
                            for (int n : SIZES) {
                                int tri = side == BLAS.Side.Left ? m : n;
                                double[] a = ZBlasTestSupport.randomTriangularMatrix(tri, uplo, diag == BLAS.Diag.Unit, seed++);
                                double[] actual = ZBlasTestSupport.randomComplexMatrix(m, n, seed++);
                                double[] expected = refZtrmm(side, uplo, trans, diag, m, n,
                                    -0.75, 0.5, a, 0, tri, actual, 0, n);

                                Ztrmm.ztrmm(side, uplo, trans, diag, m, n,
                                    -0.75, 0.5, a, 0, tri, actual, 0, n);

                                ZBlasTestSupport.assertArrayClose(
                                    side + " " + uplo + " " + trans + " " + diag + " m=" + m + " n=" + n,
                                    expected, actual, TOL);
                            }
                        }
                    }
                }
            }
        }
    }

    @Test
    void testOffsetParity() {
        int m = 5;
        int n = 4;
        int lda = n;
        int ldb = n;
        int aOff = 6;
        int bOff = 8;
        double[] directA = ZBlasTestSupport.randomTriangularMatrix(n, BLAS.Uplo.Upper, false, 2026042601L);
        double[] directB = ZBlasTestSupport.randomComplexMatrix(m, n, 2026042602L);
        double[] directResult = directB.clone();

        double[] offsetA = new double[aOff + n * n * 2 + 6];
        double[] offsetB = new double[bOff + m * n * 2 + 8];
        ZBlasTestSupport.embedMatrix(directA, n, n, 0, lda, offsetA, aOff, lda);
        ZBlasTestSupport.embedMatrix(directB, m, n, 0, ldb, offsetB, bOff, ldb);

        Ztrmm.ztrmm(BLAS.Side.Right, BLAS.Uplo.Upper, BLAS.Trans.Conj, BLAS.Diag.NonUnit,
            m, n, 1.0, 0.0, directA, 0, lda, directResult, 0, ldb);
        Ztrmm.ztrmm(BLAS.Side.Right, BLAS.Uplo.Upper, BLAS.Trans.Conj, BLAS.Diag.NonUnit,
            m, n, 1.0, 0.0, offsetA, aOff, lda, offsetB, bOff, ldb);

        assertMatrixRegionClose(directResult, 0, ldb, offsetB, bOff, ldb, m, n);
    }

    @Test
    void testAlphaZeroClearsMatrix() {
        int m = 4;
        int n = 3;
        double[] a = ZBlasTestSupport.randomTriangularMatrix(n, BLAS.Uplo.Lower, false, 2026042603L);
        double[] b = ZBlasTestSupport.randomComplexMatrix(m, n, 2026042604L);

        Ztrmm.ztrmm(BLAS.Side.Right, BLAS.Uplo.Lower, BLAS.Trans.Conj, BLAS.Diag.NonUnit,
            m, n, 0.0, 0.0, a, 0, n, b, 0, n);

        for (double value : b) {
            assertEquals(0.0, value, TOL);
        }
    }

    private static double[] refZtrmm(BLAS.Side side, BLAS.Uplo uplo, BLAS.Trans trans, BLAS.Diag diag,
                                     int m, int n,
                                     double alphaRe, double alphaIm,
                                     double[] a, int aOff, int lda,
                                     double[] b, int bOff, int ldb) {
        double[] result = b.clone();
        double[] input = new double[m * n * 2];
        for (int row = 0; row < m; row++) {
            for (int col = 0; col < n; col++) {
                int src = ZBlasTestSupport.matrixIndex(bOff, ldb, row, col);
                int dst = (row * n + col) * 2;
                input[dst] = b[src];
                input[dst + 1] = b[src + 1];
            }
        }

        for (int row = 0; row < m; row++) {
            for (int col = 0; col < n; col++) {
                double sumRe = 0.0;
                double sumIm = 0.0;
                if (side == BLAS.Side.Left) {
                    for (int p = 0; p < m; p++) {
                        double[] aVal = triangularOpValue(uplo, trans, diag, a, aOff, lda, row, p);
                        if (aVal[0] == 0.0 && aVal[1] == 0.0) {
                            continue;
                        }
                        int inputPos = (p * n + col) * 2;
                        double bRe = input[inputPos];
                        double bIm = input[inputPos + 1];
                        sumRe = Math.fma(aVal[0], bRe, Math.fma(-aVal[1], bIm, sumRe));
                        sumIm = Math.fma(aVal[0], bIm, Math.fma(aVal[1], bRe, sumIm));
                    }
                } else {
                    for (int p = 0; p < n; p++) {
                        double[] aVal = triangularOpValue(uplo, trans, diag, a, aOff, lda, p, col);
                        if (aVal[0] == 0.0 && aVal[1] == 0.0) {
                            continue;
                        }
                        int inputPos = (row * n + p) * 2;
                        double bRe = input[inputPos];
                        double bIm = input[inputPos + 1];
                        sumRe = Math.fma(bRe, aVal[0], Math.fma(-bIm, aVal[1], sumRe));
                        sumIm = Math.fma(bRe, aVal[1], Math.fma(bIm, aVal[0], sumIm));
                    }
                }

                int dst = ZBlasTestSupport.matrixIndex(bOff, ldb, row, col);
                result[dst] = alphaRe * sumRe - alphaIm * sumIm;
                result[dst + 1] = alphaRe * sumIm + alphaIm * sumRe;
            }
        }
        return result;
    }

    private static double[] triangularOpValue(BLAS.Uplo uplo, BLAS.Trans trans, BLAS.Diag diag,
                                              double[] a, int aOff, int lda,
                                              int row, int col) {
        if (row == col && diag == BLAS.Diag.Unit) {
            return new double[]{1.0, 0.0};
        }

        int srcRow = trans == BLAS.Trans.NoTrans ? row : col;
        int srcCol = trans == BLAS.Trans.NoTrans ? col : row;
        boolean stored = uplo == BLAS.Uplo.Upper ? srcCol >= srcRow : srcCol <= srcRow;
        if (!stored) {
            return new double[]{0.0, 0.0};
        }

        int pos = ZBlasTestSupport.matrixIndex(aOff, lda, srcRow, srcCol);
        double re = a[pos];
        double im = a[pos + 1];
        if (trans == BLAS.Trans.Conj) {
            im = -im;
        }
        return new double[]{re, im};
    }

    private static void assertMatrixRegionClose(double[] expected, int expectedOff, int expectedLd,
                                                double[] actual, int actualOff, int actualLd,
                                                int rows, int cols) {
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                int e = ZBlasTestSupport.matrixIndex(expectedOff, expectedLd, row, col);
                int a = ZBlasTestSupport.matrixIndex(actualOff, actualLd, row, col);
                assertEquals(expected[e], actual[a], TOL);
                assertEquals(expected[e + 1], actual[a + 1], TOL);
            }
        }
    }
}