package com.curioloop.yum4j.linalg.blas.cmplx;

import com.curioloop.yum4j.linalg.blas.BLAS;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class ZgehrdTest {

    private static final double TOL = 1e-9;

    static final double[] GEHRD_A = {
            2.140937441302040e-01, 5.150476863060478e-01, -1.245738778711988e+00, 3.852731490654721e+00,
            1.731809258511820e-01, 5.708905106931670e-01, 3.853173797288368e-01, 1.135565640180599e+00,
            -8.838574362011330e-01, 9.540017634932023e-01, 1.537251059455279e-01, 6.513912513057980e-01,
            5.820871844599990e-02, -3.152692446403456e-01, -1.142970297830623e+00, 7.589692204932672e-01,
            3.577873603482833e-01, -7.728252145375718e-01, 5.607845263682344e-01, -2.368186067400089e-01,
            1.083051243175277e+00, -4.853635478291035e-01, 1.053802052034903e+00, 8.187413938632256e-02,
            -1.377669367957091e+00, 2.314658566673509e+00, -9.378250399151228e-01, -1.867265192591748e+00,
            5.150352672086598e-01, 6.862601903745136e-01, 5.137859509122088e-01, -1.612715871189652e+00
    };

    @Test
    void testHessenbergAndQ() {
        int n = 4;
        double[] A = GEHRD_A.clone();
        double[] Acopy = GEHRD_A.clone();
        double[] tau = new double[(n - 1) * 2];

        double[] work = new double[2];
        int info = Zgehrd.zgehrd(n, 0, n - 1, A, 0, n, tau, 0, work, 0, -1);
        assertEquals(0, info);

        int lwork = (int) work[0];
        work = new double[lwork];

        info = Zgehrd.zgehrd(n, 0, n - 1, A, 0, n, tau, 0, work, 0, lwork);
        assertEquals(0, info);

        double[] H = A.clone();

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < i - 1; j++) {
                H[i * n * 2 + j * 2] = 0.0;
                H[i * n * 2 + j * 2 + 1] = 0.0;
            }
        }

        work = new double[2];
        info = Zgees.zorghr(n, 0, n - 1, A, 0, n, tau, 0, work, 0, -1);
        assertEquals(0, info);

        lwork = (int) work[0];
        work = new double[lwork];

        info = Zgees.zorghr(n, 0, n - 1, A, 0, n, tau, 0, work, 0, lwork);
        assertEquals(0, info);

        for (int i = 2; i < n; i++) {
            for (int j = 0; j <= i - 2; j++) {
                assertEquals(0.0, H[i * n * 2 + j * 2], TOL);
                assertEquals(0.0, H[i * n * 2 + j * 2 + 1], TOL);
            }
        }

        double[] AQ = new double[n * n * 2];
        ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, n, n, n, 1.0, 0.0,
                Acopy, 0, n, A, 0, n, 0.0, 0.0, AQ, 0, n);

        double[] QHAQ = new double[n * n * 2];
        ZLAS.zgemm(BLAS.Trans.Conj, BLAS.Trans.NoTrans, n, n, n, 1.0, 0.0,
                A, 0, n, AQ, 0, n, 0.0, 0.0, QHAQ, 0, n);

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                int idx = i * n * 2 + j * 2;
                assertEquals(H[idx], QHAQ[idx], TOL);
                assertEquals(H[idx + 1], QHAQ[idx + 1], TOL);
            }
        }

        double[] QHQ = new double[n * n * 2];
        ZLAS.zgemm(BLAS.Trans.Conj, BLAS.Trans.NoTrans, n, n, n, 1.0, 0.0,
                A, 0, n, A, 0, n, 0.0, 0.0, QHQ, 0, n);

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                int idx = i * n * 2 + j * 2;
                if (i == j) {
                    assertEquals(1.0, QHQ[idx], TOL);
                    assertEquals(0.0, QHQ[idx + 1], TOL);
                } else {
                    assertEquals(0.0, QHQ[idx], TOL);
                    assertEquals(0.0, QHQ[idx + 1], TOL);
                }
            }
        }
    }

    @Test
    void testOffsetSubmatrixMatchesDirectReduction() {
        int n = 4;
        int ilo = 0;
        int ihi = n - 1;
        int lda = 6;
        int aOff = 5;

        double[] direct = GEHRD_A.clone();
        double[] padded = new double[(aOff + (n - 1) * lda + n + 1) * 2];
        copyMatrix(direct, n, padded, aOff, lda);

        double[] tauDirect = new double[(n - 1) * 2];
        double[] tauOffset = new double[(n - 1) * 2];

        double[] query = new double[1];
        assertEquals(0, Zgehrd.zgehrd(n, ilo, ihi, direct.clone(), 0, n, tauDirect.clone(), 0, query, 0, -1));
        int lwork = (int) query[0];
        double[] workDirect = new double[lwork];
        double[] workOffset = new double[lwork];

        assertEquals(0, Zgehrd.zgehrd(n, ilo, ihi, direct, 0, n, tauDirect, 0, workDirect, 0, lwork));
        assertEquals(0, Zgehrd.zgehrd(n, ilo, ihi, padded, aOff, lda, tauOffset, 0, workOffset, 0, lwork));

        assertSubmatrixEquals(direct, n, padded, aOff, lda, n, TOL);
        assertVectorEquals(tauDirect, tauOffset, TOL);
    }

    @Test
    void testOffsetSubmatrixMatchesDirectQGeneration() {
        int n = 4;
        int ilo = 0;
        int ihi = n - 1;
        int lda = 6;
        int aOff = 5;

        double[] directFactorized = GEHRD_A.clone();
        double[] tauDirect = new double[(n - 1) * 2];
        double[] query = new double[1];
        assertEquals(0, Zgehrd.zgehrd(n, ilo, ihi, directFactorized.clone(), 0, n, tauDirect.clone(), 0, query, 0, -1));
        int gehrdLwork = (int) query[0];
        assertEquals(0, Zgehrd.zgehrd(n, ilo, ihi, directFactorized, 0, n, tauDirect, 0, new double[gehrdLwork], 0, gehrdLwork));

        double[] offsetFactorized = new double[(aOff + (n - 1) * lda + n + 1) * 2];
        copyMatrix(directFactorized, n, offsetFactorized, aOff, lda);

        double[] directQ = directFactorized.clone();
        double[] offsetTau = tauDirect.clone();
        query[0] = 0.0;
        assertEquals(0, Zgees.zorghr(n, ilo, ihi, directQ.clone(), 0, n, tauDirect.clone(), 0, query, 0, -1));
        int lwork = (int) query[0];
        double[] workDirect = new double[lwork];
        double[] workOffset = new double[lwork];

        assertEquals(0, Zgees.zorghr(n, ilo, ihi, directQ, 0, n, tauDirect, 0, workDirect, 0, lwork));
        assertEquals(0, Zgees.zorghr(n, ilo, ihi, offsetFactorized, aOff, lda, offsetTau, 0, workOffset, 0, lwork));

        assertSubmatrixEquals(directQ, n, offsetFactorized, aOff, lda, n, TOL);
    }

    private static void copyMatrix(double[] src, int n, double[] dst, int aOff, int lda) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                int srcPos = (i * n + j) * 2;
                int dstPos = (aOff + i * lda + j) * 2;
                dst[dstPos] = src[srcPos];
                dst[dstPos + 1] = src[srcPos + 1];
            }
        }
    }

    private static void assertSubmatrixEquals(double[] expected, int expectedLda, double[] actual, int aOff, int actualLda, int n, double tol) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                int expectedPos = (i * expectedLda + j) * 2;
                int actualPos = (aOff + i * actualLda + j) * 2;
                assertEquals(expected[expectedPos], actual[actualPos], tol);
                assertEquals(expected[expectedPos + 1], actual[actualPos + 1], tol);
            }
        }
    }

    private static void assertVectorEquals(double[] expected, double[] actual, double tol) {
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], actual[i], tol);
        }
    }
}
