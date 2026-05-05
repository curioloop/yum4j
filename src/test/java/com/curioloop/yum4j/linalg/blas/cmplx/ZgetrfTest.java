package com.curioloop.yum4j.linalg.blas.cmplx;

import com.curioloop.yum4j.linalg.blas.BLAS;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class ZgetrfTest {

    private static final double TOL = 1e-12;
    private static final double SCIPY_TOL = 1e-9;

    static final double[] GETRF_A = {
            1.266911149186623e+00, 1.163163752154960e+00, -7.076694656187807e-01, 1.023306101958705e-02,
            4.438194281462284e-01, -9.815086510479509e-01, 7.746340534293368e-01, 4.621034742632708e-01,
            -9.269304715780829e-01, 1.990596955734700e-01, -5.952535606180008e-02, -6.002168771587947e-01,
            -3.241267340069073e+00, 6.980208499001886e-02, -1.024387641334290e+00, -3.853135968617599e-01,
            -2.525681513931603e-01, 1.135173452512480e-01, -1.247783181964849e+00, 6.621306745210463e-01,
            1.632411303931635e+00, 1.586016816145352e+00, -1.430141377960633e+00, -1.237815498826849e+00,
            -4.400444866969838e-01, 2.133033374656267e+00, 1.307405772860913e-01, -1.952087799522502e+00,
            1.441273289066116e+00, -1.517850950355832e-01, -1.435862151179439e+00, 5.883172064845762e-01
    };

    @Test
    void testSquareMatrix() {
        int m = 3;
        int n = 3;
        double[] A = {
            2.0, 0.0, 1.0, -1.0, 0.0, 0.0,
            1.0, 1.0, 3.0, 0.0, 1.0, 1.0,
            0.0, 0.0, 1.0, 1.0, 2.0, 0.0
        };
        double[] Aorig = A.clone();
        int[] ipiv = new int[Math.min(m, n)];

        int info = ZLAS.zgetrf(m, n, A, 0, n, ipiv);
        assertEquals(0, info);

        verifyLUDecomposition(A, Aorig, ipiv, m, n);
    }

    @Test
    void testRectangularMatrixMoreRows() {
        int m = 4;
        int n = 3;
        double[] A = {
            2.0, 0.0, 1.0, -1.0, 0.0, 0.0,
            1.0, 1.0, 3.0, 0.0, 1.0, 1.0,
            0.0, 0.0, 1.0, 1.0, 2.0, 0.0,
            3.0, 1.0, 2.0, 0.0, 1.0, -1.0
        };
        double[] Aorig = A.clone();
        int[] ipiv = new int[Math.min(m, n)];

        int info = ZLAS.zgetrf(m, n, A, 0, n, ipiv);
        assertEquals(0, info);

        verifyLUDecomposition(A, Aorig, ipiv, m, n);
    }

    @Test
    void testRectangularMatrixMoreColumns() {
        int m = 3;
        int n = 4;
        double[] A = {
            2.0, 0.0, 1.0, -1.0, 0.0, 0.0, 1.0, 1.0,
            1.0, 1.0, 3.0, 0.0, 1.0, 1.0, 2.0, 0.0,
            0.0, 0.0, 1.0, 1.0, 2.0, 0.0, 3.0, 1.0
        };
        double[] Aorig = A.clone();
        int[] ipiv = new int[Math.min(m, n)];

        int info = ZLAS.zgetrf(m, n, A, 0, n, ipiv);
        assertEquals(0, info);

        verifyLUDecomposition(A, Aorig, ipiv, m, n);
    }

    @Test
    void testZeroMatrix() {
        int m = 3;
        int n = 3;
        double[] A = new double[m * n * 2];
        int[] ipiv = new int[Math.min(m, n)];

        int info = ZLAS.zgetrf(m, n, A, 0, n, ipiv);
        assertEquals(1, info);

        for (int i = 0; i < m * n * 2; i++) {
            assertEquals(0.0, A[i], TOL, "Element " + i + " should be zero");
        }
    }

    @Test
    void testIdentityMatrix() {
        int m = 3;
        int n = 3;
        double[] A = new double[m * n * 2];
        for (int i = 0; i < m; i++) {
            A[i * n * 2 + i * 2] = 1.0;
        }
        double[] Aorig = A.clone();
        int[] ipiv = new int[Math.min(m, n)];

        int info = ZLAS.zgetrf(m, n, A, 0, n, ipiv);
        assertEquals(0, info);

        verifyLUDecomposition(A, Aorig, ipiv, m, n);
    }

    @Test
    void testEmptyMatrix() {
        int m = 0;
        int n = 0;
        double[] A = new double[0];
        int[] ipiv = new int[0];

        int info = ZLAS.zgetrf(m, n, A, 0, 1, ipiv);
        assertEquals(0, info);
    }

    @Test
    void testSmallMatrix() {
        int m = 1;
        int n = 1;
        double[] A = {5.0, 3.0};
        double[] Aorig = A.clone();
        int[] ipiv = new int[1];

        int info = ZLAS.zgetrf(m, n, A, 0, n, ipiv);
        assertEquals(0, info);

        verifyLUDecomposition(A, Aorig, ipiv, m, n);
    }

    @Test
    void testInvalidInputs() {
        int m = -1;
        int n = 2;
        double[] A = {1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0};
        int[] ipiv = new int[1];

        int info = ZLAS.zgetrf(m, n, A, 0, n, ipiv);
        assertEquals(-1, info);

        m = 2;
        n = -1;
        info = ZLAS.zgetrf(m, n, A, 0, n, ipiv);
        assertEquals(-2, info);

        m = 2;
        n = 2;
        info = ZLAS.zgetrf(m, n, A, 0, 1, ipiv);
        assertEquals(-5, info);
    }

    @Test
    void testLUDecompositionScipy() {
        int n = 4;
        double[] A = GETRF_A.clone();
        double[] Aorig = GETRF_A.clone();
        int[] ipiv = new int[n];

        int info = ZLAS.zgetrf(n, n, A, 0, n, ipiv);
        assertEquals(0, info);

        double[] L = new double[n * n * 2];
        double[] U = new double[n * n * 2];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                int idx = i * n * 2 + j * 2;
                if (i > j) {
                    L[idx] = A[idx];
                    L[idx + 1] = A[idx + 1];
                } else if (i == j) {
                    L[idx] = 1.0;
                    L[idx + 1] = 0.0;
                    U[idx] = A[idx];
                    U[idx + 1] = A[idx + 1];
                } else {
                    U[idx] = A[idx];
                    U[idx + 1] = A[idx + 1];
                }
            }
        }

        double[] LU = new double[n * n * 2];
        ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, n, n, n, 1.0, 0.0,
                L, 0, n, U, 0, n, 0.0, 0.0, LU, 0, n);

        double[] PmA = Aorig.clone();
        for (int k = 0; k < n; k++) {
            int p = ipiv[k] - 1;
            if (p != k) {
                for (int j = 0; j < n; j++) {
                    double tr = PmA[k * n * 2 + j * 2];
                    double ti = PmA[k * n * 2 + j * 2 + 1];
                    PmA[k * n * 2 + j * 2] = PmA[p * n * 2 + j * 2];
                    PmA[k * n * 2 + j * 2 + 1] = PmA[p * n * 2 + j * 2 + 1];
                    PmA[p * n * 2 + j * 2] = tr;
                    PmA[p * n * 2 + j * 2 + 1] = ti;
                }
            }
        }

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                int idx = i * n * 2 + j * 2;
                assertEquals(PmA[idx], LU[idx], SCIPY_TOL);
                assertEquals(PmA[idx + 1], LU[idx + 1], SCIPY_TOL);
            }
        }
    }

    private void verifyLUDecomposition(double[] A, double[] Aorig, int[] ipiv, int m, int n) {
        double[] L = new double[m * n * 2];
        double[] U = new double[m * n * 2];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                int idx = i * n * 2 + j * 2;
                if (i > j) {
                    L[idx] = A[idx];
                    L[idx + 1] = A[idx + 1];
                } else if (i == j) {
                    L[idx] = 1.0;
                    L[idx + 1] = 0.0;
                    U[idx] = A[idx];
                    U[idx + 1] = A[idx + 1];
                } else {
                    U[idx] = A[idx];
                    U[idx + 1] = A[idx + 1];
                }
            }
        }

        double[] LU = new double[m * n * 2];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double sumR = 0, sumI = 0;
                for (int k = 0; k < Math.min(m, n); k++) {
                    double lr = L[i * n * 2 + k * 2];
                    double li = L[i * n * 2 + k * 2 + 1];
                    double ur = U[k * n * 2 + j * 2];
                    double ui = U[k * n * 2 + j * 2 + 1];
                    sumR += lr * ur - li * ui;
                    sumI += lr * ui + li * ur;
                }
                LU[i * n * 2 + j * 2] = sumR;
                LU[i * n * 2 + j * 2 + 1] = sumI;
            }
        }

        double[] PmA = Aorig.clone();
        for (int k = 0; k < Math.min(m, n); k++) {
            int p = ipiv[k] - 1;
            if (p != k) {
                for (int j = 0; j < n; j++) {
                    double tr = PmA[k * n * 2 + j * 2];
                    double ti = PmA[k * n * 2 + j * 2 + 1];
                    PmA[k * n * 2 + j * 2] = PmA[p * n * 2 + j * 2];
                    PmA[k * n * 2 + j * 2 + 1] = PmA[p * n * 2 + j * 2 + 1];
                    PmA[p * n * 2 + j * 2] = tr;
                    PmA[p * n * 2 + j * 2 + 1] = ti;
                }
            }
        }

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                int idx = i * n * 2 + j * 2;
                assertEquals(PmA[idx], LU[idx], TOL, "Mismatch at (" + i + "," + j + ") real");
                assertEquals(PmA[idx + 1], LU[idx + 1], TOL, "Mismatch at (" + i + "," + j + ") imag");
            }
        }
    }

}
