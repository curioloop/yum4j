/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.linalg.blas.cmplx;

import com.curioloop.yum4j.linalg.blas.BLAS;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ZgelqfTest {

    private static final double TOL = 1e-10;
    private static final double SCIPY_TOL = 1e-9;

    static final double[] LQF_A_SQ = {
            2.140937441302040e-01, 5.150476863060478e-01, -1.245738778711988e+00, 3.852731490654721e+00,
            1.731809258511820e-01, 5.708905106931670e-01, 3.853173797288368e-01, 1.135565640180599e+00,
            -8.838574362011330e-01, 9.540017634932023e-01, 1.537251059455279e-01, 6.513912513057980e-01,
            5.820871844599990e-02, -3.152692446403456e-01, -1.142970297830623e+00, 7.589692204932672e-01,
            3.577873603482833e-01, -7.728252145375718e-01, 5.607845263682344e-01, -2.368186067400089e-01,
            1.083051243175277e+00, -4.853635478291035e-01, 1.053802052034903e+00, 8.187413938632256e-02,
            -1.377669367957091e+00, 2.314658566673509e+00, -9.378250399151228e-01, -1.867265192591748e+00,
            5.150352672086598e-01, 6.862601903745136e-01, 5.137859509122088e-01, -1.612715871189652e+00
    };

    static final double[] LQF_A_MN = {
            2.140937441302040e-01, 5.150476863060478e-01, -1.245738778711988e+00, 3.852731490654721e+00,
            1.731809258511820e-01, 5.708905106931670e-01, 3.853173797288368e-01, 1.135565640180599e+00,
            -8.838574362011330e-01, 9.540017634932023e-01, 1.537251059455279e-01, 6.513912513057980e-01,
            5.820871844599990e-02, -3.152692446403456e-01,
            3.577873603482833e-01, -7.728252145375718e-01, 5.607845263682344e-01, -2.368186067400089e-01,
            1.083051243175277e+00, -4.853635478291035e-01,
            -1.377669367957091e+00, 2.314658566673509e+00, -9.378250399151228e-01, -1.867265192591748e+00,
            5.150352672086598e-01, 6.862601903745136e-01
    };

    static final double[] LQF_A_NM = {
            2.140937441302040e-01, 5.150476863060478e-01, -1.245738778711988e+00, 3.852731490654721e+00,
            1.731809258511820e-01, 5.708905106931670e-01,
            -8.838574362011330e-01, 9.540017634932023e-01, 1.537251059455279e-01, 6.513912513057980e-01,
            5.820871844599990e-02, -3.152692446403456e-01,
            3.577873603482833e-01, -7.728252145375718e-01, 5.607845263682344e-01, -2.368186067400089e-01,
            1.083051243175277e+00, -4.853635478291035e-01,
            -1.377669367957091e+00, 2.314658566673509e+00, -9.378250399151228e-01, -1.867265192591748e+00,
            5.150352672086598e-01, 6.862601903745136e-01
    };

    @Test
    void testBasicSquareMatrix() {
        double[] A = {
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0, 19.0, 20.0
        };
        double[] tau = new double[3 * 2];
        double[] work = new double[3 * 2];
        int lwork = 3;

        int info = Zgelq.zgelqf(3, 3, A, 0, 3, tau, 0, work, 0, lwork);

        assertEquals(0, info);

        assertTrue(Math.abs(A[0 * 3 * 2 + 0 * 2 + 0]) > TOL || Math.abs(A[0 * 3 * 2 + 0 * 2 + 1]) > TOL);
        assertTrue(Math.abs(A[1 * 3 * 2 + 0 * 2 + 0]) > TOL || Math.abs(A[1 * 3 * 2 + 0 * 2 + 1]) > TOL);
        assertTrue(Math.abs(A[1 * 3 * 2 + 1 * 2 + 0]) > TOL || Math.abs(A[1 * 3 * 2 + 1 * 2 + 1]) > TOL);
        assertTrue(Math.abs(A[2 * 3 * 2 + 0 * 2 + 0]) > TOL || Math.abs(A[2 * 3 * 2 + 0 * 2 + 1]) > TOL);
        assertTrue(Math.abs(A[2 * 3 * 2 + 1 * 2 + 0]) > TOL || Math.abs(A[2 * 3 * 2 + 1 * 2 + 1]) > TOL);
        assertTrue(Math.abs(A[2 * 3 * 2 + 2 * 2 + 0]) > TOL || Math.abs(A[2 * 3 * 2 + 2 * 2 + 1]) > TOL);

        for (int i = 0; i < 3; i++) {
            assertTrue(Double.isFinite(tau[i * 2]) && Double.isFinite(tau[i * 2 + 1]));
        }
    }

    @Test
    void testRectangularMatrixMoreRows() {
        double[] A = {
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0
        };
        double[] tau = new double[2 * 2];
        double[] work = new double[2 * 2];
        int lwork = 2;

        int info = Zgelq.zgelqf(2, 3, A, 0, 3, tau, 0, work, 0, lwork);

        assertEquals(0, info);

        assertTrue(Math.abs(A[0 * 3 * 2 + 0 * 2 + 0]) > TOL || Math.abs(A[0 * 3 * 2 + 0 * 2 + 1]) > TOL);
        assertTrue(Math.abs(A[1 * 3 * 2 + 0 * 2 + 0]) > TOL || Math.abs(A[1 * 3 * 2 + 0 * 2 + 1]) > TOL);
        assertTrue(Math.abs(A[1 * 3 * 2 + 1 * 2 + 0]) > TOL || Math.abs(A[1 * 3 * 2 + 1 * 2 + 1]) > TOL);

        for (int i = 0; i < 2; i++) {
            assertTrue(Math.abs(tau[i * 2]) > TOL || Math.abs(tau[i * 2 + 1]) > TOL);
        }
    }

    @Test
    void testRectangularMatrixMoreColumns() {
        double[] A = {
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0
        };
        double[] tau = new double[3 * 2];
        double[] work = new double[3 * 2];
        int lwork = 3;

        int info = Zgelq.zgelqf(3, 2, A, 0, 2, tau, 0, work, 0, lwork);

        assertEquals(0, info);

        assertTrue(Math.abs(A[0 * 2 * 2 + 0 * 2 + 0]) > TOL || Math.abs(A[0 * 2 * 2 + 0 * 2 + 1]) > TOL);
        assertTrue(Math.abs(A[1 * 2 * 2 + 0 * 2 + 0]) > TOL || Math.abs(A[1 * 2 * 2 + 0 * 2 + 1]) > TOL);
        assertTrue(Math.abs(A[1 * 2 * 2 + 1 * 2 + 0]) > TOL || Math.abs(A[1 * 2 * 2 + 1 * 2 + 1]) > TOL);
        assertTrue(Math.abs(A[2 * 2 * 2 + 0 * 2 + 0]) > TOL || Math.abs(A[2 * 2 * 2 + 0 * 2 + 1]) > TOL);
        assertTrue(Math.abs(A[2 * 2 * 2 + 1 * 2 + 0]) > TOL || Math.abs(A[2 * 2 * 2 + 1 * 2 + 1]) > TOL);

        for (int i = 0; i < 2; i++) {
            assertTrue(Double.isFinite(tau[i * 2]) && Double.isFinite(tau[i * 2 + 1]));
        }
    }

    @Test
    void testWorkspaceQuery() {
        double[] A = {
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0
        };
        double[] tau = new double[2 * 2];
        double[] work = new double[1];
        int lwork = -1;

        int info = Zgelq.zgelqf(2, 2, A, 0, 2, tau, 0, work, 0, lwork);

        assertEquals(0, info);
        assertTrue(work[0] > 0);

        lwork = (int) work[0];
        work = new double[lwork * 2];
        info = Zgelq.zgelqf(2, 2, A, 0, 2, tau, 0, work, 0, lwork);
        assertEquals(0, info);
    }

    @Test
    void testZeroMatrix() {
        double[] A = {
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        };
        double[] tau = new double[3 * 2];
        double[] work = new double[3 * 2];
        int lwork = 3;

        int info = Zgelq.zgelqf(3, 3, A, 0, 3, tau, 0, work, 0, lwork);

        assertEquals(0, info);

        for (int i = 0; i < A.length; i++) {
            assertTrue(Math.abs(A[i]) < TOL);
        }

        for (int i = 0; i < 3; i++) {
            assertTrue(Math.abs(tau[i * 2]) < TOL && Math.abs(tau[i * 2 + 1]) < TOL);
        }
    }

    @Test
    void testIdentityMatrix() {
        double[] A = {
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 1.0, 0.0
        };
        double[] tau = new double[3 * 2];
        double[] work = new double[3 * 2];
        int lwork = 3;

        int info = Zgelq.zgelqf(3, 3, A, 0, 3, tau, 0, work, 0, lwork);

        assertEquals(0, info);

        assertTrue(Math.abs(A[0 * 3 * 2 + 0 * 2 + 0]) > TOL);
        assertTrue(Math.abs(A[1 * 3 * 2 + 1 * 2 + 0]) > TOL);
        assertTrue(Math.abs(A[2 * 3 * 2 + 2 * 2 + 0]) > TOL);

        assertTrue(Math.abs(A[0 * 3 * 2 + 1 * 2 + 0]) < TOL && Math.abs(A[0 * 3 * 2 + 1 * 2 + 1]) < TOL);
        assertTrue(Math.abs(A[0 * 3 * 2 + 2 * 2 + 0]) < TOL && Math.abs(A[0 * 3 * 2 + 2 * 2 + 1]) < TOL);
        assertTrue(Math.abs(A[1 * 3 * 2 + 0 * 2 + 0]) < TOL && Math.abs(A[1 * 3 * 2 + 0 * 2 + 1]) < TOL);
        assertTrue(Math.abs(A[1 * 3 * 2 + 2 * 2 + 0]) < TOL && Math.abs(A[1 * 3 * 2 + 2 * 2 + 1]) < TOL);
        assertTrue(Math.abs(A[2 * 3 * 2 + 0 * 2 + 0]) < TOL && Math.abs(A[2 * 3 * 2 + 0 * 2 + 1]) < TOL);
        assertTrue(Math.abs(A[2 * 3 * 2 + 1 * 2 + 0]) < TOL && Math.abs(A[2 * 3 * 2 + 1 * 2 + 1]) < TOL);
    }

    @Test
    void testEmpty() {
        double[] A = new double[0];
        double[] tau = new double[0];
        double[] work = new double[0];
        int lwork = 0;

        int info = Zgelq.zgelqf(0, 0, A, 0, 1, tau, 0, work, 0, lwork);

        assertEquals(0, info);
    }

    @Test
    void testInvalidInput() {
        double[] A = {1.0, 2.0, 3.0, 4.0};
        double[] tau = new double[2 * 2];
        double[] work = new double[2 * 2];
        int lwork = 2;

        int info = Zgelq.zgelqf(-1, 2, A, 0, 2, tau, 0, work, 0, lwork);
        assertEquals(-1, info);

        info = Zgelq.zgelqf(2, -1, A, 0, 2, tau, 0, work, 0, lwork);
        assertEquals(-2, info);

        info = Zgelq.zgelqf(2, 2, A, 0, 1, tau, 0, work, 0, lwork);
        assertEquals(-5, info);

        info = Zgelq.zgelqf(2, 2, A, 0, 2, tau, 0, work, 0, -2);
        assertEquals(-10, info);
    }

    @Test
    void testWithSmallMatrix() {
        double[] A = {1.0, 2.0};
        double[] tau = new double[1 * 2];
        double[] work = new double[1 * 2];
        int lwork = 1;

        int info = Zgelq.zgelqf(1, 1, A, 0, 1, tau, 0, work, 0, lwork);

        assertEquals(0, info);

        double expectedBeta = -Math.sqrt(1.0 * 1.0 + 2.0 * 2.0);
        assertEquals(expectedBeta, A[0], TOL);
        assertEquals(0.0, A[1], TOL);
        assertTrue(Math.abs(tau[0]) > TOL || Math.abs(tau[1]) > TOL);
    }

    @Test
    void testWithVector() {
        double[] A = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        double[] tau = new double[1 * 2];
        double[] work = new double[1 * 2];
        int lwork = 1;

        int info = Zgelq.zgelqf(1, 3, A, 0, 3, tau, 0, work, 0, lwork);

        assertEquals(0, info);

        assertTrue(Math.abs(tau[0]) > TOL || Math.abs(tau[1]) > TOL);
    }

    @Test
    void testOffsetSubmatrixMatchesDirectFactorization() {
        int m = 3;
        int n = 3;
        int lda = 5;
        int aOff = 6;
        int lwork = m;

        double[] direct = {
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                13.0, 14.0, 15.0, 16.0, 19.0, 20.0
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

        double[] tauDirect = new double[Math.min(m, n) * 2];
        double[] tauOffset = new double[Math.min(m, n) * 2];
        double[] workDirect = new double[lwork * 2];
        double[] workOffset = new double[lwork * 2];

        assertEquals(0, Zgelq.zgelqf(m, n, direct, 0, n, tauDirect, 0, workDirect, 0, lwork));
        assertEquals(0, Zgelq.zgelqf(m, n, padded, aOff, lda, tauOffset, 0, workOffset, 0, lwork));

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                int src = (i * n + j) * 2;
                int dst = (aOff + i * lda + j) * 2;
                assertEquals(direct[src], padded[dst], TOL);
                assertEquals(direct[src + 1], padded[dst + 1], TOL);
            }
        }

        for (int i = 0; i < tauDirect.length; i++) {
            assertEquals(tauDirect[i], tauOffset[i], TOL);
        }
    }

    @Test
    void testSquareLQScipy() {
        int m = 4, n = 4;
        int k = Math.min(m, n);
        double[] A = LQF_A_SQ.clone();
        double[] Aorig = LQF_A_SQ.clone();
        double[] tau = new double[k * 2];

        double[] work = new double[2];
        int info = Zgelq.zgelqf(m, n, A, 0, n, tau, 0, work, 0, -1);
        assertEquals(0, info);
        int lwork = Math.max((int) work[0], n * 2);
        work = new double[lwork];
        info = Zgelq.zgelqf(m, n, A, 0, n, tau, 0, work, 0, lwork);
        assertEquals(0, info);

        double[] L = A.clone();
        for (int i = 0; i < m; i++) {
            for (int j = i + 1; j < n; j++) {
                L[i * n * 2 + j * 2] = 0.0;
                L[i * n * 2 + j * 2 + 1] = 0.0;
            }
        }

        double[] Q = A.clone();
        work = new double[2];
        info = Zgelq.zorglq(m, n, k, Q, 0, n, tau, 0, work, 0, -1);
        assertEquals(0, info);
        lwork = Math.max((int) work[0], n * 2);
        work = new double[lwork];
        info = Zgelq.zorglq(m, n, k, Q, 0, n, tau, 0, work, 0, lwork);
        assertEquals(0, info);

        double[] LQ = new double[m * n * 2];
        ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, m, n, n, 1.0, 0.0,
                L, 0, n, Q, 0, n, 0.0, 0.0, LQ, 0, n);

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                int idx = i * n * 2 + j * 2;
                assertEquals(Aorig[idx], LQ[idx], SCIPY_TOL);
                assertEquals(Aorig[idx + 1], LQ[idx + 1], SCIPY_TOL);
            }
        }

        for (int i = 0; i < m; i++) {
            for (int j = i + 1; j < n; j++) {
                int idx = i * n * 2 + j * 2;
                assertEquals(0.0, L[idx], SCIPY_TOL);
                assertEquals(0.0, L[idx + 1], SCIPY_TOL);
            }
        }

        double[] QHQ = new double[n * n * 2];
        ZLAS.zgemm(BLAS.Trans.Conj, BLAS.Trans.NoTrans, n, n, n, 1.0, 0.0,
                Q, 0, n, Q, 0, n, 0.0, 0.0, QHQ, 0, n);

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                int idx = i * n * 2 + j * 2;
                if (i == j) {
                    assertEquals(1.0, QHQ[idx], SCIPY_TOL);
                    assertEquals(0.0, QHQ[idx + 1], SCIPY_TOL);
                } else {
                    assertEquals(0.0, QHQ[idx], SCIPY_TOL);
                    assertEquals(0.0, QHQ[idx + 1], SCIPY_TOL);
                }
            }
        }
    }

    @Test
    void testRectangularMltNScipy() {
        int m = 3, n = 4;
        int k = Math.min(m, n);
        double[] A = LQF_A_MN.clone();
        double[] Aorig = LQF_A_MN.clone();
        double[] tau = new double[k * 2];

        double[] work = new double[2];
        int info = Zgelq.zgelqf(m, n, A, 0, n, tau, 0, work, 0, -1);
        assertEquals(0, info);
        int lwork = Math.max((int) work[0], n * 2);
        work = new double[lwork];
        info = Zgelq.zgelqf(m, n, A, 0, n, tau, 0, work, 0, lwork);
        assertEquals(0, info);

        double[] L = A.clone();
        for (int i = 0; i < m; i++) {
            for (int j = i + 1; j < n; j++) {
                L[i * n * 2 + j * 2] = 0.0;
                L[i * n * 2 + j * 2 + 1] = 0.0;
            }
        }

        double[] Q = new double[n * n * 2];
        for (int i = 0; i < m; i++) {
            System.arraycopy(A, i * n * 2, Q, i * n * 2, n * 2);
        }

        work = new double[2];
        info = Zgelq.zorglq(n, n, k, Q, 0, n, tau, 0, work, 0, -1);
        assertEquals(0, info);
        lwork = Math.max((int) work[0], n * 2);
        work = new double[lwork];
        info = Zgelq.zorglq(n, n, k, Q, 0, n, tau, 0, work, 0, lwork);
        assertEquals(0, info);

        double[] LQ = new double[m * n * 2];
        ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, m, n, n, 1.0, 0.0,
                L, 0, n, Q, 0, n, 0.0, 0.0, LQ, 0, n);

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                int idx = i * n * 2 + j * 2;
                assertEquals(Aorig[idx], LQ[idx], SCIPY_TOL);
                assertEquals(Aorig[idx + 1], LQ[idx + 1], SCIPY_TOL);
            }
        }
    }

    @Test
    void testRectangularMgtNScipy() {
        int m = 4, n = 3;
        int k = Math.min(m, n);
        double[] A = LQF_A_NM.clone();
        double[] Aorig = LQF_A_NM.clone();
        double[] tau = new double[k * 2];

        double[] work = new double[2];
        int info = Zgelq.zgelqf(m, n, A, 0, n, tau, 0, work, 0, -1);
        assertEquals(0, info);
        int lwork = Math.max((int) work[0], m * 2);
        work = new double[lwork];
        info = Zgelq.zgelqf(m, n, A, 0, n, tau, 0, work, 0, lwork);
        assertEquals(0, info);

        double[] L = A.clone();
        for (int i = 0; i < m; i++) {
            for (int j = i + 1; j < n; j++) {
                L[i * n * 2 + j * 2] = 0.0;
                L[i * n * 2 + j * 2 + 1] = 0.0;
            }
        }

        double[] QforOrg = new double[n * n * 2];
        for (int i = 0; i < n; i++) {
            System.arraycopy(A, i * n * 2, QforOrg, i * n * 2, n * 2);
        }

        work = new double[2];
        info = Zgelq.zorglq(n, n, k, QforOrg, 0, n, tau, 0, work, 0, -1);
        assertEquals(0, info);
        lwork = Math.max((int) work[0], n * 2);
        work = new double[lwork];
        info = Zgelq.zorglq(n, n, k, QforOrg, 0, n, tau, 0, work, 0, lwork);
        assertEquals(0, info);

        double[] LQ = new double[m * n * 2];
        ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, m, n, n, 1.0, 0.0,
                L, 0, n, QforOrg, 0, n, 0.0, 0.0, LQ, 0, n);

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                int idx = i * n * 2 + j * 2;
                assertEquals(Aorig[idx], LQ[idx], SCIPY_TOL);
                assertEquals(Aorig[idx + 1], LQ[idx + 1], SCIPY_TOL);
            }
        }
    }

}
