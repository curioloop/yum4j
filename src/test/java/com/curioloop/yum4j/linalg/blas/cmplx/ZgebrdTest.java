/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.linalg.blas.cmplx;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ZgebrdTest {

    private static final double TOL = 1e-10;

    @Test
    void testBasicSquareMatrix() {
        double[] A = {
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0, 17.0, 18.0
        };
        double[] d = new double[3];
        double[] e = new double[2];
        double[] tauQ = new double[3 * 2];
        double[] tauP = new double[3 * 2];
        double[] work = new double[3 * 2];
        int lwork = 3;

        int info = Zgebrd.zgebrd(3, 3, A, 0, 3, d, 0, e, 0, tauQ, 0, tauP, 0, work, 0, lwork);

        assertEquals(0, info);

        for (int i = 0; i < 3; i++) {
            assertTrue(Double.isFinite(d[i]), "d[" + i + "] should be finite");
        }
        for (int i = 0; i < 2; i++) {
            assertTrue(Double.isFinite(e[i]), "e[" + i + "] should be finite");
        }
    }

    @Test
    void testRectangularMatrixMoreRows() {
        int m = 4, n = 3;
        double[] A = new double[m * n * 2];
        for (int i = 0; i < m * n; i++) {
            A[i * 2] = i + 1;
            A[i * 2 + 1] = i + 2;
        }
        double[] d = new double[3];
        double[] e = new double[2];
        double[] tauQ = new double[3 * 2];
        double[] tauP = new double[3 * 2];
        double[] work = new double[4 * 2];
        int lwork = 4;

        int info = Zgebrd.zgebrd(m, n, A, 0, n, d, 0, e, 0, tauQ, 0, tauP, 0, work, 0, lwork);

        assertEquals(0, info);

        for (int i = 0; i < 3; i++) {
            assertTrue(Double.isFinite(d[i]), "d[" + i + "] should be finite");
        }
    }

    @Test
    void testRectangularMatrixMoreColumns() {
        int m = 3, n = 4;
        double[] A = new double[m * n * 2];
        for (int i = 0; i < m * n; i++) {
            A[i * 2] = i + 1;
            A[i * 2 + 1] = i + 2;
        }
        double[] d = new double[3];
        double[] e = new double[2];
        double[] tauQ = new double[3 * 2];
        double[] tauP = new double[3 * 2];
        double[] work = new double[4 * 2];
        int lwork = 4;

        int info = Zgebrd.zgebrd(m, n, A, 0, n, d, 0, e, 0, tauQ, 0, tauP, 0, work, 0, lwork);

        assertEquals(0, info);

        for (int i = 0; i < 3; i++) {
            assertTrue(Double.isFinite(d[i]), "d[" + i + "] should be finite");
        }
    }

    @Test
    void testWorkspaceQuery() {
        double[] A = {
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0
        };
        double[] d = new double[2];
        double[] e = new double[1];
        double[] tauQ = new double[2 * 2];
        double[] tauP = new double[2 * 2];
        double[] work = new double[1];
        int lwork = -1;

        int info = Zgebrd.zgebrd(2, 2, A, 0, 2, d, 0, e, 0, tauQ, 0, tauP, 0, work, 0, lwork);

        assertEquals(0, info);
        assertTrue(work[0] > 0);

        lwork = (int) work[0];
        work = new double[lwork];
        info = Zgebrd.zgebrd(2, 2, A, 0, 2, d, 0, e, 0, tauQ, 0, tauP, 0, work, 0, lwork);
        assertEquals(0, info);
    }

    @Test
    void testZeroMatrix() {
        double[] A = new double[3 * 3 * 2];
        double[] d = new double[3];
        double[] e = new double[2];
        double[] tauQ = new double[3 * 2];
        double[] tauP = new double[3 * 2];
        double[] work = new double[3 * 2];
        int lwork = 3;

        int info = Zgebrd.zgebrd(3, 3, A, 0, 3, d, 0, e, 0, tauQ, 0, tauP, 0, work, 0, lwork);

        assertEquals(0, info);

        for (int i = 0; i < 3; i++) {
            assertEquals(0.0, d[i], TOL);
        }
    }

    @Test
    void testIdentityMatrix() {
        int n = 3;
        double[] A = new double[n * n * 2];
        for (int i = 0; i < n; i++) {
            A[(i * n + i) * 2] = 1.0;
        }
        double[] d = new double[3];
        double[] e = new double[2];
        double[] tauQ = new double[3 * 2];
        double[] tauP = new double[3 * 2];
        double[] work = new double[3 * 2];
        int lwork = 3;

        int info = Zgebrd.zgebrd(3, 3, A, 0, 3, d, 0, e, 0, tauQ, 0, tauP, 0, work, 0, lwork);

        assertEquals(0, info);

        for (int i = 0; i < 3; i++) {
            assertEquals(1.0, Math.abs(d[i]), TOL, "d[" + i + "] should be 1 or -1");
        }
        for (int i = 0; i < 2; i++) {
            assertEquals(0.0, e[i], TOL, "e[" + i + "] should be 0");
        }
    }

    @Test
    void testEmpty() {
        double[] A = new double[0];
        double[] d = new double[0];
        double[] e = new double[0];
        double[] tauQ = new double[0];
        double[] tauP = new double[0];
        double[] work = new double[0];
        int lwork = 0;

        int info = Zgebrd.zgebrd(0, 0, A, 0, 1, d, 0, e, 0, tauQ, 0, tauP, 0, work, 0, lwork);

        assertEquals(0, info);
    }

    @Test
    void testInvalidInput() {
        double[] A = {1.0, 2.0, 3.0, 4.0};
        double[] d = new double[2];
        double[] e = new double[1];
        double[] tauQ = new double[2 * 2];
        double[] tauP = new double[2 * 2];
        double[] work = new double[2 * 2];
        int lwork = 2;

        int info = Zgebrd.zgebrd(-1, 2, A, 0, 2, d, 0, e, 0, tauQ, 0, tauP, 0, work, 0, lwork);
        assertEquals(-1, info);

        info = Zgebrd.zgebrd(2, -1, A, 0, 2, d, 0, e, 0, tauQ, 0, tauP, 0, work, 0, lwork);
        assertEquals(-2, info);

        info = Zgebrd.zgebrd(2, 2, A, 0, 1, d, 0, e, 0, tauQ, 0, tauP, 0, work, 0, lwork);
        assertEquals(-5, info);

        info = Zgebrd.zgebrd(2, 2, A, 0, 2, d, 0, e, 0, tauQ, 0, tauP, 0, work, 0, -2);
        assertEquals(-10, info);
    }

    @Test
    void testWithSmallMatrix() {
        double[] A = {1.0, 2.0};
        double[] d = new double[1];
        double[] e = new double[0];
        double[] tauQ = new double[1 * 2];
        double[] tauP = new double[1 * 2];
        double[] work = new double[1 * 2];
        int lwork = 1;

        int info = Zgebrd.zgebrd(1, 1, A, 0, 1, d, 0, e, 0, tauQ, 0, tauP, 0, work, 0, lwork);

        assertEquals(0, info);

        double expectedBeta = -Math.sqrt(1.0 * 1.0 + 2.0 * 2.0);
        assertEquals(expectedBeta, d[0], TOL);
    }

    @Test
    void testWithVector() {
        double[] A = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        double[] d = new double[1];
        double[] e = new double[0];
        double[] tauQ = new double[1 * 2];
        double[] tauP = new double[3 * 2];
        double[] work = new double[3 * 2];
        int lwork = 3;

        int info = Zgebrd.zgebrd(1, 3, A, 0, 3, d, 0, e, 0, tauQ, 0, tauP, 0, work, 0, lwork);

        assertEquals(0, info);

        assertTrue(Double.isFinite(d[0]), "d[0] should be finite");
        assertEquals(0.0, tauQ[0], TOL, "tauQ should be 0 for m=1");
        assertEquals(0.0, tauQ[1], TOL);
    }

    @Test
    void testRandomMatrix() {
        int m = 6, n = 6;
        double[] A = new double[m * n * 2];
        for (int i = 0; i < m * n; i++) {
            A[i * 2] = Math.random() * 10 - 5;
            A[i * 2 + 1] = Math.random() * 10 - 5;
        }
        int minmn = Math.min(m, n);
        double[] d = new double[minmn];
        double[] e = new double[minmn - 1];
        double[] tauQ = new double[minmn * 2];
        double[] tauP = new double[minmn * 2];
        double[] work = new double[Math.max(m, n) * 2];
        int lwork = Math.max(m, n);

        int info = Zgebrd.zgebrd(m, n, A, 0, n, d, 0, e, 0, tauQ, 0, tauP, 0, work, 0, lwork);

        assertEquals(0, info);

        for (int i = 0; i < minmn; i++) {
            assertTrue(Double.isFinite(d[i]), "d[" + i + "] should be finite");
        }
        for (int i = 0; i < minmn - 1; i++) {
            assertTrue(Double.isFinite(e[i]), "e[" + i + "] should be finite");
        }
    }

    @Test
    void testTallMatrix() {
        int m = 5, n = 3;
        double[] A = new double[m * n * 2];
        for (int i = 0; i < m * n; i++) {
            A[i * 2] = Math.random() * 10 - 5;
            A[i * 2 + 1] = Math.random() * 10 - 5;
        }
        int minmn = Math.min(m, n);
        double[] d = new double[minmn];
        double[] e = new double[minmn - 1];
        double[] tauQ = new double[minmn * 2];
        double[] tauP = new double[minmn * 2];
        double[] work = new double[Math.max(m, n) * 2];
        int lwork = Math.max(m, n);

        int info = Zgebrd.zgebrd(m, n, A, 0, n, d, 0, e, 0, tauQ, 0, tauP, 0, work, 0, lwork);

        assertEquals(0, info);

        for (int i = 0; i < minmn; i++) {
            assertTrue(Double.isFinite(d[i]), "d[" + i + "] should be finite");
        }
    }

    @Test
    void testWideMatrix() {
        int m = 3, n = 5;
        double[] A = new double[m * n * 2];
        for (int i = 0; i < m * n; i++) {
            A[i * 2] = Math.random() * 10 - 5;
            A[i * 2 + 1] = Math.random() * 10 - 5;
        }
        int minmn = Math.min(m, n);
        double[] d = new double[minmn];
        double[] e = new double[minmn - 1];
        double[] tauQ = new double[minmn * 2];
        double[] tauP = new double[minmn * 2];
        double[] work = new double[Math.max(m, n) * 2];
        int lwork = Math.max(m, n);

        int info = Zgebrd.zgebrd(m, n, A, 0, n, d, 0, e, 0, tauQ, 0, tauP, 0, work, 0, lwork);

        assertEquals(0, info);

        for (int i = 0; i < minmn; i++) {
            assertTrue(Double.isFinite(d[i]), "d[" + i + "] should be finite");
        }
    }

    @Test
    void testOffsetSubmatrixMatchesDirectReductionTall() {
        int m = 4;
        int n = 3;
        int lda = 5;
        int aOff = 6;
        int lwork = Math.max(m, n);
        int minmn = Math.min(m, n);

        double[] direct = {
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
                19.0, 20.0, 21.0, 22.0, 23.0, 24.0
        };
        double[] padded = new double[(aOff + (m - 1) * lda + n + 1) * 2];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                int src = (i * n + j) * 2;
                int dst = (aOff + i * lda + j) * 2;
                padded[dst] = direct[src];
                padded[dst + 1] = direct[src + 1];
            }
        }

        double[] dDirect = new double[minmn];
        double[] dOffset = new double[minmn];
        double[] eDirect = new double[minmn - 1];
        double[] eOffset = new double[minmn - 1];
        double[] tauQDirect = new double[minmn * 2];
        double[] tauQOffset = new double[minmn * 2];
        double[] tauPDirect = new double[minmn * 2];
        double[] tauPOffset = new double[minmn * 2];
        double[] workDirect = new double[lwork * 2];
        double[] workOffset = new double[lwork * 2];

        assertEquals(0, Zgebrd.zgebrd(m, n, direct, 0, n, dDirect, 0, eDirect, 0, tauQDirect, 0, tauPDirect, 0, workDirect, 0, lwork));
        assertEquals(0, Zgebrd.zgebrd(m, n, padded, aOff, lda, dOffset, 0, eOffset, 0, tauQOffset, 0, tauPOffset, 0, workOffset, 0, lwork));

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                int src = (i * n + j) * 2;
                int dst = (aOff + i * lda + j) * 2;
                assertEquals(direct[src], padded[dst], TOL);
                assertEquals(direct[src + 1], padded[dst + 1], TOL);
            }
        }

        for (int i = 0; i < minmn; i++) {
            assertEquals(dDirect[i], dOffset[i], TOL);
            assertEquals(tauQDirect[i * 2], tauQOffset[i * 2], TOL);
            assertEquals(tauQDirect[i * 2 + 1], tauQOffset[i * 2 + 1], TOL);
            assertEquals(tauPDirect[i * 2], tauPOffset[i * 2], TOL);
            assertEquals(tauPDirect[i * 2 + 1], tauPOffset[i * 2 + 1], TOL);
        }
        for (int i = 0; i < minmn - 1; i++) {
            assertEquals(eDirect[i], eOffset[i], TOL);
        }
    }

    @Test
    void testOffsetSubmatrixMatchesDirectReductionWide() {
        int m = 3;
        int n = 4;
        int lda = 6;
        int aOff = 5;
        int lwork = Math.max(m, n);
        int minmn = Math.min(m, n);

        double[] direct = {
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
                17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0
        };
        double[] padded = new double[(aOff + (m - 1) * lda + n + 1) * 2];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                int src = (i * n + j) * 2;
                int dst = (aOff + i * lda + j) * 2;
                padded[dst] = direct[src];
                padded[dst + 1] = direct[src + 1];
            }
        }

        double[] dDirect = new double[minmn];
        double[] dOffset = new double[minmn];
        double[] eDirect = new double[minmn - 1];
        double[] eOffset = new double[minmn - 1];
        double[] tauQDirect = new double[minmn * 2];
        double[] tauQOffset = new double[minmn * 2];
        double[] tauPDirect = new double[minmn * 2];
        double[] tauPOffset = new double[minmn * 2];
        double[] workDirect = new double[lwork * 2];
        double[] workOffset = new double[lwork * 2];

        assertEquals(0, Zgebrd.zgebrd(m, n, direct, 0, n, dDirect, 0, eDirect, 0, tauQDirect, 0, tauPDirect, 0, workDirect, 0, lwork));
        assertEquals(0, Zgebrd.zgebrd(m, n, padded, aOff, lda, dOffset, 0, eOffset, 0, tauQOffset, 0, tauPOffset, 0, workOffset, 0, lwork));

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                int src = (i * n + j) * 2;
                int dst = (aOff + i * lda + j) * 2;
                assertEquals(direct[src], padded[dst], TOL);
                assertEquals(direct[src + 1], padded[dst + 1], TOL);
            }
        }

        for (int i = 0; i < minmn; i++) {
            assertEquals(dDirect[i], dOffset[i], TOL);
            assertEquals(tauQDirect[i * 2], tauQOffset[i * 2], TOL);
            assertEquals(tauQDirect[i * 2 + 1], tauQOffset[i * 2 + 1], TOL);
            assertEquals(tauPDirect[i * 2], tauPOffset[i * 2], TOL);
            assertEquals(tauPDirect[i * 2 + 1], tauPOffset[i * 2 + 1], TOL);
        }
        for (int i = 0; i < minmn - 1; i++) {
            assertEquals(eDirect[i], eOffset[i], TOL);
        }
    }
}
