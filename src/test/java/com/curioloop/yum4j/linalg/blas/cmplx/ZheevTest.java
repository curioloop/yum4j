package com.curioloop.yum4j.linalg.blas.cmplx;

import com.curioloop.yum4j.linalg.blas.BLAS;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class ZheevTest {

    private static final double TOL = 1e-10;

    static final double[] HEEV_A = {
            1.993027301752824e-01, 0.0, -1.565779367842304e+00, 2.598813061604124e-01,
            -2.333916723402370e+00, -5.569259892995911e-01, 2.960229094101567e-01, -3.186231124229033e+00,
            -1.565779367842304e+00, -2.598813061604124e-01, 9.471848612703632e-01, 0.0,
            -1.241485750439479e+00, 1.823915318487310e-02, 2.857077159299968e+00, -1.273847114148117e-01,
            -2.333916723402370e+00, 5.569259892995911e-01, -1.241485750439479e+00, -1.823915318487310e-02,
            1.627034434739340e+00, 0.0, -2.838347550995183e+00, 4.769520791762808e-01,
            2.960229094101567e-01, 3.186231124229033e+00, 2.857077159299968e+00, 1.273847114148117e-01,
            -2.838347550995183e+00, -4.769520791762808e-01, 3.692677170646084e-01, 0.0
    };

    static final double[] HEEV_W = {
            -4.952574276322900e+00, -1.350776623340627e+00, 3.020946904973698e+00, 6.425193737939441e+00
    };

    @Test
    void testSquareMatrixEigenvaluesOnly() {
        int n = 3;
        double[] A = {
            3.0, 0.0, 1.0, -1.0, 0.0, 0.0,
            1.0, 1.0, 2.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 1.0, 0.0, 4.0, 0.0
        };
        double[] w = new double[n];
        double[] work = new double[1];

        int info = ZLAS.zheev('N', 'L', n, A, n, w, 0, work, 0, -1);
        assertEquals(0, info);

        int lwork = (int) work[0];
        work = new double[lwork];

        info = ZLAS.zheev('N', 'L', n, A, n, w, 0, work, 0, lwork);
        assertEquals(0, info);

        for (int i = 0; i < n-1; i++) {
            assertTrue(w[i] <= w[i+1], "Eigenvalues should be sorted in ascending order");
            assertTrue(Double.isFinite(w[i]), "Eigenvalue should be finite");
        }
        assertTrue(Double.isFinite(w[n-1]), "Eigenvalue should be finite");

        double trace = 3.0 + 2.0 + 4.0;
        double sumW = w[0] + w[1] + w[2];
        assertEquals(trace, sumW, TOL, "Trace should equal sum of eigenvalues");
    }

    @Test
    void testSquareMatrixWithEigenvectors() {
        int n = 2;
        double[] A = {
            2.0, 0.0, 1.0, -1.0,
            1.0, 1.0, 2.0, 0.0
        };
        double[] w = new double[n];
        double[] work = new double[1];

        int info = ZLAS.zheev('V', 'L', n, A, n, w, 0, work, 0, -1);
        assertEquals(0, info);

        int lwork = (int) work[0];
        work = new double[lwork];

        info = ZLAS.zheev('V', 'L', n, A, n, w, 0, work, 0, lwork);
        assertEquals(0, info);

        assertEquals(2 - Math.sqrt(2), w[0], TOL);
        assertEquals(2 + Math.sqrt(2), w[1], TOL);

        for (int i = 0; i < n; i++) {
            double norm = 0.0;
            for (int j = 0; j < n; j++) {
                int idx = j * n * 2 + i * 2;
                norm += A[idx] * A[idx] + A[idx + 1] * A[idx + 1];
            }
            norm = Math.sqrt(norm);
            assertEquals(1.0, norm, TOL, "Eigenvector " + i + " should be normalized");
        }
    }

    @Test
    void testUpperTriangleInput() {
        int n = 2;
        double[] A = {
            2.0, 0.0, 1.0, -1.0,
            0.0, 0.0, 2.0, 0.0
        };
        double[] w = new double[n];
        double[] work = new double[1];

        int info = ZLAS.zheev('N', 'U', n, A, n, w, 0, work, 0, -1);
        assertEquals(0, info);

        int lwork = (int) work[0];
        work = new double[lwork];

        info = ZLAS.zheev('N', 'U', n, A, n, w, 0, work, 0, lwork);
        assertEquals(0, info);

        for (int i = 0; i < n; i++) {
            assertTrue(Double.isFinite(w[i]), "Eigenvalue should be finite");
        }
    }

    @Test
    void testZeroMatrix() {
        int n = 3;
        double[] A = new double[n * n * 2];
        double[] w = new double[n];
        double[] work = new double[1];

        int info = ZLAS.zheev('N', 'L', n, A, n, w, 0, work, 0, -1);
        assertEquals(0, info);

        int lwork = (int) work[0];
        work = new double[lwork];

        info = ZLAS.zheev('N', 'L', n, A, n, w, 0, work, 0, lwork);
        assertEquals(0, info);

        for (int i = 0; i < n; i++) {
            assertEquals(0.0, w[i], TOL, "Eigenvalue should be zero");
        }
    }

    @Test
    void testIdentityMatrix() {
        int n = 3;
        double[] A = new double[n * n * 2];
        for (int i = 0; i < n; i++) {
            A[i * n * 2 + i * 2] = 1.0;
        }
        double[] w = new double[n];
        double[] work = new double[1];

        int info = ZLAS.zheev('N', 'L', n, A, n, w, 0, work, 0, -1);
        assertEquals(0, info);

        int lwork = (int) work[0];
        work = new double[lwork];

        info = ZLAS.zheev('N', 'L', n, A, n, w, 0, work, 0, lwork);
        assertEquals(0, info);

        for (int i = 0; i < n; i++) {
            assertEquals(1.0, w[i], TOL, "Eigenvalue should be 1");
        }
    }

    @Test
    void testEmptyMatrix() {
        int n = 0;
        double[] A = new double[0];
        double[] w = new double[0];
        double[] work = new double[0];

        int info = ZLAS.zheev('N', 'L', n, A, n, w, 0, work, 0, 0);
        assertEquals(0, info);
    }

    @Test
    void testSmallMatrix() {
        int n = 1;
        double[] A = {5.0, 0.0};
        double[] w = new double[1];
        double[] work = new double[1];

        int info = ZLAS.zheev('N', 'L', n, A, n, w, 0, work, 0, -1);
        assertEquals(0, info);

        int lwork = (int) work[0];
        work = new double[lwork];

        info = ZLAS.zheev('N', 'L', n, A, n, w, 0, work, 0, lwork);
        assertEquals(0, info);

        assertEquals(5.0, w[0], TOL);
    }

    @Test
    void testInvalidInputs() {
        int n = 2;
        double[] A = {2.0, 0.0, 1.0, -1.0, 1.0, 1.0, 2.0, 0.0};
        double[] w = new double[2];
        double[] work = new double[1];

        int info = ZLAS.zheev('X', 'L', n, A, n, w, 0, work, 0, -1);
        assertEquals(-1, info);

        info = ZLAS.zheev('N', 'X', n, A, n, w, 0, work, 0, -1);
        assertEquals(-2, info);

        info = ZLAS.zheev('N', 'L', -1, A, n, w, 0, work, 0, -1);
        assertEquals(-3, info);

        info = ZLAS.zheev('N', 'L', n, A, n-1, w, 0, work, 0, -1);
        assertEquals(-5, info);
    }

    @Test
    void testEigenvaluesOnlyScipy() {
        int n = 4;
        double[] A = HEEV_A.clone();
        double[] w = new double[n];
        double[] work = new double[1];

        int info = ZLAS.zheev('N', 'L', n, A, n, w, 0, work, 0, -1);
        assertEquals(0, info);

        int lwork = (int) work[0];
        work = new double[lwork];

        info = ZLAS.zheev('N', 'L', n, A, n, w, 0, work, 0, lwork);
        assertEquals(0, info);

        for (int i = 0; i < n; i++) {
            assertEquals(HEEV_W[i], w[i], TOL);
        }
    }

    @Test
    void testEigenvectorsScipy() {
        int n = 4;
        double[] A = HEEV_A.clone();
        double[] Aorig = HEEV_A.clone();
        double[] w = new double[n];
        double[] work = new double[1];

        int info = ZLAS.zheev('V', 'L', n, A, n, w, 0, work, 0, -1);
        assertEquals(0, info);

        int lwork = (int) work[0];
        work = new double[lwork];

        info = ZLAS.zheev('V', 'L', n, A, n, w, 0, work, 0, lwork);
        assertEquals(0, info);

        for (int i = 0; i < n; i++) {
            assertEquals(HEEV_W[i], w[i], TOL);
        }

        double[] AV = new double[n * n * 2];
        ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, n, n, n, 1.0, 0.0,
                Aorig, 0, n, A, 0, n, 0.0, 0.0, AV, 0, n);

        double[] VL = new double[n * n * 2];
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < n; i++) {
                int idx = i * n * 2 + j * 2;
                VL[idx] = A[idx] * w[j];
                VL[idx + 1] = A[idx + 1] * w[j];
            }
        }

        for (int i = 0; i < n * n * 2; i++) {
            assertEquals(VL[i], AV[i], TOL);
        }

        double[] VHV = new double[n * n * 2];
        ZLAS.zgemm(BLAS.Trans.Conj, BLAS.Trans.NoTrans, n, n, n, 1.0, 0.0,
                A, 0, n, A, 0, n, 0.0, 0.0, VHV, 0, n);

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                int idx = i * n * 2 + j * 2;
                if (i == j) {
                    assertEquals(1.0, VHV[idx], TOL);
                    assertEquals(0.0, VHV[idx + 1], TOL);
                } else {
                    assertEquals(0.0, VHV[idx], TOL);
                    assertEquals(0.0, VHV[idx + 1], TOL);
                }
            }
        }
    }

    @Test
    void testZhetrdDecomposition() {
        int n = 4;
        double[] A = HEEV_A.clone();
        double[] w = new double[n];
        int tauSize = 2 * (n - 1);
        int eSize = n - 1;
        int workStart = tauSize + eSize;
        if (workStart % 2 != 0) workStart++;
        double[] work = new double[100];

        int info = Zhetrd.zhetrd(BLAS.Uplo.Lower, n, A, 0, n, w, 0, work, tauSize, work, 0, work, workStart, 100 - workStart);
        assertEquals(0, info);

        for (int i = 0; i < n; i++) {
            assertTrue(Double.isFinite(w[i]));
        }
        for (int i = 0; i < n - 1; i++) {
            assertTrue(Double.isFinite(work[tauSize + i]));
        }
        for (int i = 0; i < tauSize; i++) {
            assertTrue(Double.isFinite(work[i]));
        }
    }

    @Test
    void testZorgtrOrthogonality() {
        int n = 4;
        double[] A = HEEV_A.clone();
        double[] w = new double[n];
        int tauSize = 2 * (n - 1);
        int eSize = n - 1;
        int workStart = tauSize + eSize;
        if (workStart % 2 != 0) workStart++;
        double[] work = new double[100];

        int info = Zhetrd.zhetrd(BLAS.Uplo.Lower, n, A, 0, n, w, 0, work, tauSize, work, 0, work, workStart, 100 - workStart);
        assertEquals(0, info);

        double[] tauCopy = new double[tauSize];
        System.arraycopy(work, 0, tauCopy, 0, tauSize);

        info = Zorgtr.zorgtr(BLAS.Uplo.Lower, n, A, 0, n, tauCopy, 0, work, workStart, 100 - workStart);
        assertEquals(0, info);

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

        double[] Aorig = HEEV_A.clone();
        double[] QA = new double[n * n * 2];
        ZLAS.zgemm(BLAS.Trans.Conj, BLAS.Trans.NoTrans, n, n, n, 1.0, 0.0,
                A, 0, n, Aorig, 0, n, 0.0, 0.0, QA, 0, n);
        double[] QHAQ = new double[n * n * 2];
        ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, n, n, n, 1.0, 0.0,
                QA, 0, n, A, 0, n, 0.0, 0.0, QHAQ, 0, n);
        for (int i = 2; i < n; i++) {
            for (int j = 0; j < i - 1; j++) {
                int idx = i * n * 2 + j * 2;
                assertEquals(0.0, QHAQ[idx], TOL);
                assertEquals(0.0, QHAQ[idx + 1], TOL);
            }
        }
    }

    @Test
    void testZsteqrEigenvectors() {
        int n = 4;
        double[] A = HEEV_A.clone();
        double[] w = new double[n];
        int tauSize = 2 * (n - 1);
        int eSize = n - 1;
        int workStart = tauSize + eSize;
        if (workStart % 2 != 0) workStart++;
        double[] work = new double[100];

        int info = Zhetrd.zhetrd(BLAS.Uplo.Lower, n, A, 0, n, w, 0, work, tauSize, work, 0, work, workStart, 100 - workStart);
        assertEquals(0, info);

        double[] tauCopy = new double[tauSize];
        System.arraycopy(work, 0, tauCopy, 0, tauSize);
        double[] eCopy = new double[eSize];
        System.arraycopy(work, tauSize, eCopy, 0, eSize);

        info = Zorgtr.zorgtr(BLAS.Uplo.Lower, n, A, 0, n, tauCopy, 0, work, workStart, 100 - workStart);
        assertEquals(0, info);

        info = Zsteqr.zsteqr('V', n, w, 0, eCopy, 0, A, 0, n, work, 0);
        assertEquals(0, info);

        for (int i = 0; i < n; i++) {
            assertEquals(HEEV_W[i], w[i], TOL);
        }

        double[] VHV = new double[n * n * 2];
        ZLAS.zgemm(BLAS.Trans.Conj, BLAS.Trans.NoTrans, n, n, n, 1.0, 0.0,
                A, 0, n, A, 0, n, 0.0, 0.0, VHV, 0, n);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                int idx = i * n * 2 + j * 2;
                if (i == j) {
                    assertEquals(1.0, VHV[idx], TOL);
                    assertEquals(0.0, VHV[idx + 1], TOL);
                } else {
                    assertEquals(0.0, VHV[idx], TOL);
                    assertEquals(0.0, VHV[idx + 1], TOL);
                }
            }
        }

        double[] Aorig = HEEV_A.clone();
        double[] AV = new double[n * n * 2];
        ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, n, n, n, 1.0, 0.0,
                Aorig, 0, n, A, 0, n, 0.0, 0.0, AV, 0, n);
        double[] VL = new double[n * n * 2];
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < n; i++) {
                int idx = i * n * 2 + j * 2;
                VL[idx] = A[idx] * w[j];
                VL[idx + 1] = A[idx + 1] * w[j];
            }
        }
        for (int i = 0; i < n * n * 2; i++) {
            assertEquals(VL[i], AV[i], TOL);
        }
    }

    @Test
    void testEigenvaluesOnlyUpperScipy() {
        int n = 4;
        double[] A = HEEV_A.clone();
        double[] w = new double[n];
        double[] work = new double[1];

        int info = ZLAS.zheev('N', 'U', n, A, n, w, 0, work, 0, -1);
        assertEquals(0, info);

        int lwork = (int) work[0];
        work = new double[lwork];

        info = ZLAS.zheev('N', 'U', n, A, n, w, 0, work, 0, lwork);
        assertEquals(0, info);

        for (int i = 0; i < n; i++) {
            assertEquals(HEEV_W[i], w[i], TOL);
        }
    }

    @Test
    void testEigenvectorsUpperScipy() {
        int n = 4;
        double[] A = HEEV_A.clone();
        double[] Aorig = HEEV_A.clone();
        double[] w = new double[n];
        double[] work = new double[1];

        int info = ZLAS.zheev('V', 'U', n, A, n, w, 0, work, 0, -1);
        assertEquals(0, info);

        int lwork = (int) work[0];
        work = new double[lwork];

        info = ZLAS.zheev('V', 'U', n, A, n, w, 0, work, 0, lwork);
        assertEquals(0, info);

        for (int i = 0; i < n; i++) {
            assertEquals(HEEV_W[i], w[i], TOL);
        }

        double[] AV = new double[n * n * 2];
        ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, n, n, n, 1.0, 0.0,
                Aorig, 0, n, A, 0, n, 0.0, 0.0, AV, 0, n);

        double[] VL = new double[n * n * 2];
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < n; i++) {
                int idx = i * n * 2 + j * 2;
                VL[idx] = A[idx] * w[j];
                VL[idx + 1] = A[idx + 1] * w[j];
            }
        }

        for (int i = 0; i < n * n * 2; i++) {
            assertEquals(VL[i], AV[i], TOL);
        }

        double[] VHV = new double[n * n * 2];
        ZLAS.zgemm(BLAS.Trans.Conj, BLAS.Trans.NoTrans, n, n, n, 1.0, 0.0,
                A, 0, n, A, 0, n, 0.0, 0.0, VHV, 0, n);

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                int idx = i * n * 2 + j * 2;
                if (i == j) {
                    assertEquals(1.0, VHV[idx], TOL);
                    assertEquals(0.0, VHV[idx + 1], TOL);
                } else {
                    assertEquals(0.0, VHV[idx], TOL);
                    assertEquals(0.0, VHV[idx + 1], TOL);
                }
            }
        }
    }

    @Test
    void testZorgtrShift() {
        int n = 4;
        double[] A = HEEV_A.clone();
        double[] w = new double[n];
        int tauSize = 2 * (n - 1);
        int eSize = n - 1;
        int workStart = tauSize + eSize;
        if (workStart % 2 != 0) workStart++;
        double[] work = new double[256];

        int info = Zhetrd.zhetrd(BLAS.Uplo.Lower, n, A, 0, n, w, 0, work, tauSize, work, 0, work, workStart, 256 - workStart);
        assertEquals(0, info);

        double[] tauCopy = new double[tauSize];
        System.arraycopy(work, 0, tauCopy, 0, tauSize);

        info = Zorgtr.zorgtr(BLAS.Uplo.Lower, n, A, 0, n, tauCopy, 0, work, workStart, 256 - workStart);
        assertEquals(0, info);

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
    void testZhetrd2x2WithZsterf() {
        int n = 2;
        double[] A = {2.0, 0.0, 1.0, -1.0, 1.0, 1.0, 2.0, 0.0};
        double[] d = new double[2];
        double[] e = new double[1];
        double[] tau = new double[2];
        double[] work = new double[100];

        int info = Zhetrd.zhetrd(BLAS.Uplo.Lower, n, A, 0, n, d, 0, e, 0, tau, 0, work, 0, 100);
        assertEquals(0, info);

        assertEquals(2.0, d[0], TOL);
        assertEquals(2.0, d[1], TOL);
        assertEquals(-Math.sqrt(2), e[0], TOL);

        boolean ok = Zsterf.zsterf(n, d, 0, e, 0);
        assertTrue(ok);
        assertEquals(2 - Math.sqrt(2), d[0], TOL);
        assertEquals(2 + Math.sqrt(2), d[1], TOL);
    }

    @Test
    void testZhetrd3x3WithZsterf() {
        int n = 3;
        double[] A = {
            3, 0,   1, -1,   0, 0,
            1, 1,   2, 0,    1, 0,
            0, 0,   1, 0,    4, 0
        };
        double[] d = new double[3];
        double[] e = new double[2];
        double[] tau = new double[4];
        double[] work = new double[100];

        int info = Zhetrd.zhetrd(BLAS.Uplo.Lower, n, A, 0, n, d, 0, e, 0, tau, 0, work, 0, 100);
        assertEquals(0, info);

        assertEquals(9.0, d[0] + d[1] + d[2], TOL);

        boolean ok = Zsterf.zsterf(n, d, 0, e, 0);
        assertTrue(ok);
        for (int i = 0; i < n; i++) {
            assertTrue(Double.isFinite(d[i]));
        }
        for (int i = 0; i < n - 1; i++) {
            assertTrue(d[i] <= d[i + 1] + TOL);
        }
    }

    @Test
    void testZsteqrRawOffsetMatchesDirectEigenvectors() {
        int n = 4;
        double[] factorized = HEEV_A.clone();
        double[] dBase = new double[n];
        int tauSize = 2 * (n - 1);
        int eSize = n - 1;
        int workStart = tauSize + eSize;
        if (workStart % 2 != 0) workStart++;
        double[] prepWork = new double[256];

        int info = Zhetrd.zhetrd(BLAS.Uplo.Lower, n, factorized, 0, n, dBase, 0, prepWork, tauSize, prepWork, 0, prepWork, workStart, 256 - workStart);
        assertEquals(0, info);

        double[] tau = new double[tauSize];
        System.arraycopy(prepWork, 0, tau, 0, tauSize);
        double[] eBase = new double[eSize];
        System.arraycopy(prepWork, tauSize, eBase, 0, eSize);

        double[] directZ = factorized.clone();
        double[] tauDirect = tau.clone();
        double[] qWork = new double[256];
        info = Zorgtr.zorgtr(BLAS.Uplo.Lower, n, directZ, 0, n, tauDirect, 0, qWork, workStart, 256 - workStart);
        assertEquals(0, info);

        double[] directD = dBase.clone();
        double[] directE = eBase.clone();
        double[] directWork = new double[256];
        info = Zsteqr.zsteqr('V', n, directD, 0, directE, 0, directZ, 0, n, directWork, 0);
        assertEquals(0, info);

        int ldz = 6;
        int zOff = 10;
        double[] offsetZ = new double[zOff + ldz * n * 2];
        copyMatrixToRawOffset(factorized, n, n, n, offsetZ, zOff, ldz);
        double[] tauOffset = tau.clone();
        double[] offsetPrepWork = new double[256];
        info = Zorgtr.zorgtr(BLAS.Uplo.Lower, n, offsetZ, zOff / 2, ldz, tauOffset, 0, offsetPrepWork, workStart, 256 - workStart);
        assertEquals(0, info);

        double[] offsetD = dBase.clone();
        double[] offsetE = eBase.clone();
        double[] offsetWork = new double[256];
        info = Zsteqr.zsteqr('V', n, offsetD, 0, offsetE, 0, offsetZ, zOff, ldz, offsetWork, 0);
        assertEquals(0, info);

        assertRealVectorEquals(directD, offsetD);
        assertRawSubmatrixEquals(directZ, n, n, n, offsetZ, zOff, ldz);
    }

    private static void copyMatrixToRawOffset(double[] src, int rows, int cols, int srcLda, double[] dst, int aOff, int dstLda) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                int srcPos = (i * srcLda + j) * 2;
                int dstPos = aOff + (i * dstLda + j) * 2;
                dst[dstPos] = src[srcPos];
                dst[dstPos + 1] = src[srcPos + 1];
            }
        }
    }

    private static void assertRawSubmatrixEquals(double[] expected, int rows, int cols, int expectedLda, double[] actual, int aOff, int actualLda) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                int expectedPos = (i * expectedLda + j) * 2;
                int actualPos = aOff + (i * actualLda + j) * 2;
                assertEquals(expected[expectedPos], actual[actualPos], TOL);
                assertEquals(expected[expectedPos + 1], actual[actualPos + 1], TOL);
            }
        }
    }

    private static void assertRealVectorEquals(double[] expected, double[] actual) {
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], actual[i], TOL);
        }
    }
}
