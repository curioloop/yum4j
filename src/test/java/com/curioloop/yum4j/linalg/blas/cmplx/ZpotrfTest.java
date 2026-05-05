package com.curioloop.yum4j.linalg.blas.cmplx;

import com.curioloop.yum4j.linalg.blas.BLAS;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class ZpotrfTest {

    private static final double TOL = 1e-12;
    private static final double SCIPY_TOL = 1e-9;

    static final double[] POTRF_A = {
            9.219996595802913e+00, 0.0, 9.815626522969335e-01, -5.683027906619609e-01,
            -2.564169407442167e+00, 3.132568457770329e+00, -2.789281219825326e+00, 1.440373093635805e+00,
            9.815626522969335e-01, 5.683027906619609e-01, 8.758816910867061e+00, 0.0,
            -4.159422224877828e+00, -1.401127976863429e+00, 3.885737575222743e-01, 1.937239566313883e+00,
            -2.564169407442167e+00, -3.132568457770329e+00, -4.159422224877828e+00, 1.401127976863429e+00,
            1.613814715149200e+01, 0.0, 4.543745742690927e+00, 2.257519683910790e+00,
            -2.789281219825326e+00, -1.440373093635805e+00, 3.885737575222743e-01, -1.937239566313883e+00,
            4.543745742690927e+00, -2.257519683910790e+00, 1.841307079254466e+01, 0.0
    };

    @Test
    void testUpperTriangular() {
        int n = 3;
        double[] A = {
            4.0, 0.0, 1.0, -1.0, 2.0, 0.0,
            1.0, 1.0, 3.0, 0.0, 0.0, -1.0,
            2.0, 0.0, 0.0, 1.0, 5.0, 0.0
        };
        double[] Aorig = A.clone();

        int info = ZLAS.zpotrf(BLAS.Uplo.Upper, n, A, 0, n);
        assertEquals(0, info);

        verifyUpperCholesky(A, Aorig, n);
    }

    @Test
    void testLowerTriangular() {
        int n = 3;
        double[] A = {
            4.0, 0.0, 1.0, 1.0, 2.0, 0.0,
            1.0, -1.0, 3.0, 0.0, 0.0, 1.0,
            2.0, 0.0, 0.0, -1.0, 5.0, 0.0
        };
        double[] Aorig = A.clone();

        int info = ZLAS.zpotrf(BLAS.Uplo.Lower, n, A, 0, n);
        assertEquals(0, info);

        verifyLowerCholesky(A, Aorig, n);
    }

    @Test
    void testIdentityMatrix() {
        int n = 3;
        double[] A = new double[n * n * 2];
        for (int i = 0; i < n; i++) {
            A[i * n * 2 + i * 2] = 1.0;
        }
        double[] Aorig = A.clone();

        int info = ZLAS.zpotrf(BLAS.Uplo.Upper, n, A, 0, n);
        assertEquals(0, info);

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                int idx = i * n * 2 + j * 2;
                if (i == j) {
                    assertEquals(1.0, A[idx], TOL, "Diagonal element should be 1");
                    assertEquals(0.0, A[idx + 1], TOL, "Imaginary part should be 0");
                } else if (i < j) {
                    assertEquals(0.0, A[idx], TOL, "Upper triangular element should be 0");
                    assertEquals(0.0, A[idx + 1], TOL, "Imaginary part should be 0");
                }
            }
        }
    }

    @Test
    void testEmptyMatrix() {
        int n = 0;
        double[] A = new double[0];

        int info = ZLAS.zpotrf(BLAS.Uplo.Upper, n, A, 0, 1);
        assertEquals(0, info);
    }

    @Test
    void testSmallMatrix() {
        int n = 1;
        double[] A = {4.0, 0.0};
        double[] Aorig = A.clone();

        int info = ZLAS.zpotrf(BLAS.Uplo.Upper, n, A, 0, n);
        assertEquals(0, info);

        assertEquals(2.0, A[0], TOL, "Cholesky factor should be 2");
        assertEquals(0.0, A[1], TOL, "Imaginary part should be 0");
    }

    @Test
    void testInvalidInputs() {
        int n = 2;
        double[] A = {2.0, 0.0, 1.0, -1.0, 1.0, 1.0, 2.0, 0.0};

        int info = ZLAS.zpotrf(null, n, A, 0, n);
        assertEquals(-1, info);

        info = ZLAS.zpotrf(BLAS.Uplo.Upper, -1, A, 0, n);
        assertEquals(-2, info);

        info = ZLAS.zpotrf(BLAS.Uplo.Upper, n, A, 0, n-1);
        assertEquals(-5, info);
    }

    @Test
    void testCholeskyLowerScipy() {
        int n = 4;
        double[] A = POTRF_A.clone();
        double[] Aorig = POTRF_A.clone();

        int info = ZLAS.zpotrf(BLAS.Uplo.Lower, n, A, 0, n);
        assertEquals(0, info);

        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                A[i * n * 2 + j * 2] = 0.0;
                A[i * n * 2 + j * 2 + 1] = 0.0;
            }
        }

        double[] LLH = new double[n * n * 2];
        ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.Conj, n, n, n, 1.0, 0.0,
                A, 0, n, A, 0, n, 0.0, 0.0, LLH, 0, n);

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                int idx = i * n * 2 + j * 2;
                assertEquals(Aorig[idx], LLH[idx], SCIPY_TOL);
                assertEquals(Aorig[idx + 1], LLH[idx + 1], SCIPY_TOL);
            }
        }
    }

    @Test
    void testCholeskyUpperScipy() {
        int n = 4;
        double[] A = POTRF_A.clone();
        double[] Aorig = POTRF_A.clone();

        int info = ZLAS.zpotrf(BLAS.Uplo.Upper, n, A, 0, n);
        assertEquals(0, info);

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < i; j++) {
                A[i * n * 2 + j * 2] = 0.0;
                A[i * n * 2 + j * 2 + 1] = 0.0;
            }
        }

        double[] UHU = new double[n * n * 2];
        ZLAS.zgemm(BLAS.Trans.Conj, BLAS.Trans.NoTrans, n, n, n, 1.0, 0.0,
                A, 0, n, A, 0, n, 0.0, 0.0, UHU, 0, n);

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                int idx = i * n * 2 + j * 2;
                assertEquals(Aorig[idx], UHU[idx], SCIPY_TOL);
                assertEquals(Aorig[idx + 1], UHU[idx + 1], SCIPY_TOL);
            }
        }
    }

    private void verifyUpperCholesky(double[] U, double[] Aorig, int n) {
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                double sumR = 0.0;
                double sumI = 0.0;
                for (int k = 0; k < n; k++) {
                    if (k <= i && k <= j) {
                        double ukir = U[k * n * 2 + i * 2];
                        double ukii = U[k * n * 2 + i * 2 + 1];
                        double ukjr = U[k * n * 2 + j * 2];
                        double ukji = U[k * n * 2 + j * 2 + 1];
                        sumR += ukir * ukjr + ukii * ukji;
                        sumI += ukir * ukji - ukii * ukjr;
                    }
                }
                int idx = i * n * 2 + j * 2;
                assertEquals(Aorig[idx], sumR, TOL, "Mismatch at (" + i + "," + j + ") real");
                if (i != j) {
                    assertEquals(Aorig[idx + 1], sumI, TOL, "Mismatch at (" + i + "," + j + ") imag");
                }
            }
        }
    }

    private void verifyLowerCholesky(double[] L, double[] Aorig, int n) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= i; j++) {
                double sumR = 0.0;
                double sumI = 0.0;
                for (int k = 0; k < n; k++) {
                    if (k <= i && k <= j) {
                        double likr = L[i * n * 2 + k * 2];
                        double liki = L[i * n * 2 + k * 2 + 1];
                        double ljkr = L[j * n * 2 + k * 2];
                        double ljki = L[j * n * 2 + k * 2 + 1];
                        sumR += likr * ljkr + liki * ljki;
                        sumI += liki * ljkr - likr * ljki;
                    }
                }
                int idx = i * n * 2 + j * 2;
                assertEquals(Aorig[idx], sumR, TOL, "Mismatch at (" + i + "," + j + ") real");
                if (i != j) {
                    assertEquals(Aorig[idx + 1], sumI, TOL, "Mismatch at (" + i + "," + j + ") imag");
                }
            }
        }
    }

}
