package com.curioloop.yum4j.linalg.blas.cmplx;

import com.curioloop.yum4j.linalg.blas.BLAS;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertEquals;

class ZgeesTest {

    private static final double TOL = 1e-9;

    static final double[] GEES_A = {
            2.140937441302040e-01, 5.150476863060478e-01, -1.245738778711988e+00, 3.852731490654721e+00,
            1.731809258511820e-01, 5.708905106931670e-01, 3.853173797288368e-01, 1.135565640180599e+00,
            -8.838574362011330e-01, 9.540017634932023e-01, 1.537251059455279e-01, 6.513912513057980e-01,
            5.820871844599990e-02, -3.152692446403456e-01, -1.142970297830623e+00, 7.589692204932672e-01,
            3.577873603482833e-01, -7.728252145375718e-01, 5.607845263682344e-01, -2.368186067400089e-01,
            1.083051243175277e+00, -4.853635478291035e-01, 1.053802052034903e+00, 8.187413938632256e-02,
            -1.377669367957091e+00, 2.314658566673509e+00, -9.378250399151228e-01, -1.867265192591748e+00,
            5.150352672086598e-01, 6.862601903745136e-01, 5.137859509122088e-01, -1.612715871189652e+00
    };

    static final double[] SCIPY_EIGS = {
            -1.475937890268199e+00, 2.820827449149002e+00,
            -4.054034923498664e-01, -2.858381626063451e+00,
            1.433684619520953e+00, -8.367833532861877e-01,
            2.412312807260335e+00, -5.730295120627463e-02
    };

    @Test
    void testEigenvalues() {
        int n = 4;
        double[] A = GEES_A.clone();
        double[] w = new double[n * 2];
        double[] vs = new double[n * n * 2];
        double[] rwork = new double[n];
        boolean[] bwork = new boolean[n];
        double[] work = new double[2];

        int info = Zgees.zgees('V', 'N', null, n, A, 0, n, w, 0, vs, 0, n, work, 0, -1, rwork, bwork);
        assertEquals(0, info);

        int lwork = (int) work[0];
        work = new double[lwork];

        info = Zgees.zgees('V', 'N', null, n, A, 0, n, w, 0, vs, 0, n, work, 0, lwork, rwork, bwork);
        assertEquals(0, info);

        double[] refWr = new double[n];
        double[] refWi = new double[n];
        for (int i = 0; i < n; i++) {
            refWr[i] = SCIPY_EIGS[i * 2];
            refWi[i] = SCIPY_EIGS[i * 2 + 1];
        }

        double[] wr = new double[n];
        double[] wi = new double[n];
        for (int i = 0; i < n; i++) {
            wr[i] = w[i * 2];
            wi[i] = w[i * 2 + 1];
        }

        sortEigenvalues(refWr, refWi);
        sortEigenvalues(wr, wi);

        for (int i = 0; i < n; i++) {
            assertEquals(refWr[i], wr[i], TOL);
            assertEquals(refWi[i], wi[i], TOL);
        }
    }

    @Test
    void testSchurReconstruction() {
        int n = 4;
        double[] A = GEES_A.clone();
        double[] Aorig = GEES_A.clone();
        double[] w = new double[n * 2];
        double[] vs = new double[n * n * 2];
        double[] rwork = new double[n];
        boolean[] bwork = new boolean[n];
        double[] work = new double[2];

        int info = Zgees.zgees('V', 'N', null, n, A, 0, n, w, 0, vs, 0, n, work, 0, -1, rwork, bwork);
        assertEquals(0, info);

        int lwork = (int) work[0];
        work = new double[lwork];

        info = Zgees.zgees('V', 'N', null, n, A, 0, n, w, 0, vs, 0, n, work, 0, lwork, rwork, bwork);
        assertEquals(0, info);

        double[] T = A.clone();
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < i; j++) {
                T[i * n * 2 + j * 2] = 0.0;
                T[i * n * 2 + j * 2 + 1] = 0.0;
            }
        }

        double[] ZT = new double[n * n * 2];
        ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, n, n, n, 1.0, 0.0,
                vs, 0, n, T, 0, n, 0.0, 0.0, ZT, 0, n);

        double[] ZTZH = new double[n * n * 2];
        ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.Conj, n, n, n, 1.0, 0.0,
                ZT, 0, n, vs, 0, n, 0.0, 0.0, ZTZH, 0, n);

        for (int i = 0; i < n * n * 2; i++) {
            assertEquals(Aorig[i], ZTZH[i], TOL);
        }
    }

    @Test
    void testZUnitary() {
        int n = 4;
        double[] A = GEES_A.clone();
        double[] w = new double[n * 2];
        double[] vs = new double[n * n * 2];
        double[] rwork = new double[n];
        boolean[] bwork = new boolean[n];
        double[] work = new double[2];

        int info = Zgees.zgees('V', 'N', null, n, A, 0, n, w, 0, vs, 0, n, work, 0, -1, rwork, bwork);
        assertEquals(0, info);

        int lwork = (int) work[0];
        work = new double[lwork];

        info = Zgees.zgees('V', 'N', null, n, A, 0, n, w, 0, vs, 0, n, work, 0, lwork, rwork, bwork);
        assertEquals(0, info);

        double[] ZHZ = new double[n * n * 2];
        ZLAS.zgemm(BLAS.Trans.Conj, BLAS.Trans.NoTrans, n, n, n, 1.0, 0.0,
                vs, 0, n, vs, 0, n, 0.0, 0.0, ZHZ, 0, n);

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                int idx = (i * n + j) * 2;
                if (i == j) {
                    assertEquals(1.0, ZHZ[idx], TOL);
                    assertEquals(0.0, ZHZ[idx + 1], TOL);
                } else {
                    assertEquals(0.0, ZHZ[idx], TOL);
                    assertEquals(0.0, ZHZ[idx + 1], TOL);
                }
            }
        }
    }

    @Test
    void testEigenvaluesNoSchurVectorsScipy() {
        int n = 4;
        double[] A = GEES_A.clone();
        double[] w = new double[n * 2];
        double[] rwork = new double[n];
        boolean[] bwork = new boolean[n];
        double[] work = new double[2];

        int info = Zgees.zgees('N', 'N', null, n, A, 0, n, w, 0, null, 0, 1, work, 0, -1, rwork, bwork);
        assertEquals(0, info);

        int lwork = (int) work[0];
        work = new double[lwork];

        info = Zgees.zgees('N', 'N', null, n, A, 0, n, w, 0, null, 0, 1, work, 0, lwork, rwork, bwork);
        assertEquals(0, info);

        double[] refWr = new double[n];
        double[] refWi = new double[n];
        for (int i = 0; i < n; i++) {
            refWr[i] = SCIPY_EIGS[i * 2];
            refWi[i] = SCIPY_EIGS[i * 2 + 1];
        }

        double[] wr = new double[n];
        double[] wi = new double[n];
        for (int i = 0; i < n; i++) {
            wr[i] = w[i * 2];
            wi[i] = w[i * 2 + 1];
        }

        sortEigenvalues(refWr, refWi);
        sortEigenvalues(wr, wi);

        for (int i = 0; i < n; i++) {
            assertEquals(refWr[i], wr[i], TOL);
            assertEquals(refWi[i], wi[i], TOL);
        }
    }

    @Test
    void testOffsetSubmatrixMatchesDirectSchurVectors() {
        assertOffsetSchurMatchesDirect('N');
    }

    @Test
    void testOffsetSubmatrixMatchesDirectSortedSchurVectors() {
        assertOffsetSchurMatchesDirect('S');
    }

    private static void assertOffsetSchurMatchesDirect(char sort) {
        int n = 4;
        int lda = 6;
        int ldvs = 5;
        int aOff = 5;
        int vsOff = 4;
        Zselect select = sort == 'S' ? (wr, wi) -> wr > 0.0 : null;

        double[] directA = GEES_A.clone();
        double[] offsetA = new double[(aOff + (n - 1) * lda + n + 1) * 2];
        copyMatrix(directA, n, n, n, offsetA, aOff, lda);

        double[] directW = new double[n * 2];
        double[] offsetW = new double[n * 2];
        double[] directVs = new double[n * n * 2];
        double[] offsetVs = new double[(vsOff + (n - 1) * ldvs + n + 1) * 2];
        double[] directRwork = new double[n];
        double[] offsetRwork = new double[n];
        boolean[] directBwork = new boolean[n];
        boolean[] offsetBwork = new boolean[n];
        double[] query = new double[2];

        assertEquals(0, Zgees.zgees('V', sort, select, n, directA, 0, n, directW, 0, directVs, 0, n, query, 0, -1, directRwork, directBwork));
        int lwork = (int) query[0];
        double[] directWork = new double[lwork];
        double[] offsetWork = new double[lwork];

        assertEquals(0, Zgees.zgees('V', sort, select, n, directA, 0, n, directW, 0, directVs, 0, n, directWork, 0, lwork, directRwork, directBwork));
        assertEquals(0, Zgees.zgees('V', sort, select, n, offsetA, aOff, lda, offsetW, 0, offsetVs, vsOff, ldvs, offsetWork, 0, lwork, offsetRwork, offsetBwork));

        assertSubmatrixEquals(directA, n, offsetA, aOff, lda, n);
        assertSubmatrixEquals(directVs, n, offsetVs, vsOff, ldvs, n);
        assertVectorEquals(directW, offsetW);
        assertEquals(directWork[0], offsetWork[0], TOL);
        for (int i = 0; i < n; i++) {
            assertEquals(directBwork[i] ? 1.0 : 0.0, offsetBwork[i] ? 1.0 : 0.0, TOL);
        }
    }

    private static void copyMatrix(double[] src, int rows, int cols, int srcLda, double[] dst, int aOff, int dstLda) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                int srcPos = (i * srcLda + j) * 2;
                int dstPos = (aOff + i * dstLda + j) * 2;
                dst[dstPos] = src[srcPos];
                dst[dstPos + 1] = src[srcPos + 1];
            }
        }
    }

    private static void assertSubmatrixEquals(double[] expected, int expectedLda, double[] actual, int aOff, int actualLda, int n) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                int expectedPos = (i * expectedLda + j) * 2;
                int actualPos = (aOff + i * actualLda + j) * 2;
                assertEquals(expected[expectedPos], actual[actualPos], TOL);
                assertEquals(expected[expectedPos + 1], actual[actualPos + 1], TOL);
            }
        }
    }

    private static void assertVectorEquals(double[] expected, double[] actual) {
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], actual[i], TOL);
        }
    }

    private static void sortEigenvalues(double[] wr, double[] wi) {
        int n = wr.length;
        Integer[] idx = new Integer[n];
        for (int i = 0; i < n; i++) idx[i] = i;
        Arrays.sort(idx, (a, b) -> {
            int c = Double.compare(wr[a], wr[b]);
            return c != 0 ? c : Double.compare(wi[a], wi[b]);
        });
        double[] tmpWr = new double[n];
        double[] tmpWi = new double[n];
        for (int i = 0; i < n; i++) {
            tmpWr[i] = wr[idx[i]];
            tmpWi[i] = wi[idx[i]];
        }
        System.arraycopy(tmpWr, 0, wr, 0, n);
        System.arraycopy(tmpWi, 0, wi, 0, n);
    }
}
