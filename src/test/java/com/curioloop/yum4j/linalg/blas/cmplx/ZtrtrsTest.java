package com.curioloop.yum4j.linalg.blas.cmplx;

import com.curioloop.yum4j.linalg.blas.BLAS;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class ZtrtrsTest {

    private static final double TOL = 1e-12;

    @Test
    void testUpperNoTransNonUnit() {
        // Test upper triangular matrix with no transpose and non-unit diagonal
        int n = 3;
        int nrhs = 1;
        double[] A = {
            2.0, 0.0, 1.0, -1.0, 3.0, 2.0,
            0.0, 0.0, 1.0, 0.0, 2.0, -1.0,
            0.0, 0.0, 0.0, 0.0, 3.0, 0.0
        };
        double[] B = {9.0, 3.0, 5.0, -1.0, 3.0, 0.0};
        double[] B_original = B.clone();

        int info = ZLAS.ztrtrs(BLAS.Uplo.Upper, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit,
                n, nrhs, A, 0, n, B, 0, nrhs);
        assertEquals(0, info);

        // Verify A * X = B_original
        verifySolution(A, B, B_original, n, nrhs);
    }

    @Test
    void testLowerNoTransNonUnit() {
        // Test lower triangular matrix with no transpose and non-unit diagonal
        int n = 3;
        int nrhs = 1;
        double[] A = {
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0,  // A[0][0], A[0][1], A[0][2]
            1.0, 1.0, 1.0, 0.0, 0.0, 0.0,  // A[1][0], A[1][1], A[1][2]
            2.0, -1.0, 3.0, 0.0, 1.0, 1.0   // A[2][0], A[2][1], A[2][2]
        };
        double[] B = {1.0, 0.0, 3.0, 1.0, 8.0, 2.0};
        double[] B_original = B.clone();

        int info = ZLAS.ztrtrs(BLAS.Uplo.Lower, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit,
                n, nrhs, A, 0, n, B, 0, nrhs);
        assertEquals(0, info);

        // Verify A * X = B_original
        verifySolution(A, B, B_original, n, nrhs);
    }

    @Test
    void testUpperConjTransNonUnit() {
        // Test upper triangular matrix with conjugate transpose and non-unit diagonal
        int n = 2;
        int nrhs = 1;
        double[] A = {
            2.0, 0.0, 1.0, -1.0,
            0.0, 0.0, 1.0, 0.0
        };
        double[] B = {2.0, 0.0, 2.0, 2.0};
        double[] B_original = B.clone();

        int info = ZLAS.ztrtrs(BLAS.Uplo.Upper, BLAS.Trans.Conj, BLAS.Diag.NonUnit,
                n, nrhs, A, 0, n, B, 0, nrhs);
        assertEquals(0, info);

        // Verify A^H * X = B_original
        verifyConjTransSolution(A, B, B_original, n, nrhs);
    }

    @Test
    void testLowerConjTransNonUnit() {
        // Test lower triangular matrix with conjugate transpose and non-unit diagonal
        int n = 2;
        int nrhs = 1;
        double[] A = {
            1.0, 0.0, 0.0, 0.0,
            1.0, 1.0, 2.0, 0.0
        };
        double[] B = {1.0, 0.0, 4.0, 2.0};
        double[] B_original = B.clone();

        int info = ZLAS.ztrtrs(BLAS.Uplo.Lower, BLAS.Trans.Conj, BLAS.Diag.NonUnit,
                n, nrhs, A, 0, n, B, 0, nrhs);
        assertEquals(0, info);

        // Verify A^H * X = B_original
        verifyConjTransSolution(A, B, B_original, n, nrhs);
    }

    @Test
    void testUpperTransNonUnit() {
        // Test upper triangular matrix with transpose and non-unit diagonal
        int n = 2;
        int nrhs = 1;
        double[] A = {
            2.0, 0.0, 1.0, -1.0,
            0.0, 0.0, 1.0, 0.0
        };
        double[] B = {2.0, 0.0, 2.0, -2.0};
        double[] B_original = B.clone();

        int info = ZLAS.ztrtrs(BLAS.Uplo.Upper, BLAS.Trans.Trans, BLAS.Diag.NonUnit,
                n, nrhs, A, 0, n, B, 0, nrhs);
        assertEquals(0, info);

        // Verify A^T * X = B_original
        verifyTransSolution(A, B, B_original, n, nrhs);
    }

    @Test
    void testLowerTransNonUnit() {
        // Test lower triangular matrix with transpose and non-unit diagonal
        int n = 2;
        int nrhs = 1;
        double[] A = {
            1.0, 0.0, 0.0, 0.0,
            1.0, 1.0, 2.0, 0.0
        };
        double[] B = {1.0, 0.0, 4.0, 0.0};
        double[] B_original = B.clone();

        int info = ZLAS.ztrtrs(BLAS.Uplo.Lower, BLAS.Trans.Trans, BLAS.Diag.NonUnit,
                n, nrhs, A, 0, n, B, 0, nrhs);
        assertEquals(0, info);

        // Verify A^T * X = B_original
        verifyTransSolution(A, B, B_original, n, nrhs);
    }

    @Test
    void testUpperUnitDiagonal() {
        // Test upper triangular matrix with unit diagonal
        int n = 2;
        int nrhs = 1;
        double[] A = {
            1.0, 0.0, 1.0, -1.0, // Note: diagonal is unit, so only off-diagonal elements
            0.0, 0.0, 1.0, 0.0
        };
        double[] B = {2.0, 0.0, 1.0, 0.0};
        double[] B_original = B.clone();

        int info = ZLAS.ztrtrs(BLAS.Uplo.Upper, BLAS.Trans.NoTrans, BLAS.Diag.Unit,
                n, nrhs, A, 0, n, B, 0, nrhs);
        assertEquals(0, info);

        // Verify A * X = B_original (A has unit diagonal)
        verifySolution(A, B, B_original, n, nrhs, true);
    }

    @Test
    void testLowerUnitDiagonal() {
        // Test lower triangular matrix with unit diagonal
        int n = 2;
        int nrhs = 1;
        double[] A = {
            1.0, 0.0, 0.0, 0.0, // Note: diagonal is unit, so only off-diagonal elements
            1.0, 1.0, 0.0, 0.0
        };
        double[] B = {1.0, 0.0, 2.0, 1.0};
        double[] B_original = B.clone();

        int info = ZLAS.ztrtrs(BLAS.Uplo.Lower, BLAS.Trans.NoTrans, BLAS.Diag.Unit,
                n, nrhs, A, 0, n, B, 0, nrhs);
        assertEquals(0, info);

        // Verify A * X = B_original (A has unit diagonal)
        verifySolution(A, B, B_original, n, nrhs, true);
    }

    @Test
    void testMultipleRightHandSides() {
        // Test with multiple right-hand sides
        int n = 2;
        int nrhs = 2;
        double[] A = {
            2.0, 0.0, 1.0, -1.0,
            0.0, 0.0, 1.0, 0.0
        };
        double[] B = {
            3.0, 1.0, 5.0, 2.0,  // First RHS
            1.0, 0.0, 2.0, 1.0   // Second RHS
        };
        double[] B_original = B.clone();

        int info = ZLAS.ztrtrs(BLAS.Uplo.Upper, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit,
                n, nrhs, A, 0, n, B, 0, nrhs);
        assertEquals(0, info);

        // Verify A * X = B_original for each RHS
        for (int i = 0; i < nrhs; i++) {
            double[] X = new double[n * 2];
            double[] B_col = new double[n * 2];
            for (int j = 0; j < n; j++) {
                X[j * 2] = B[j * nrhs * 2 + i * 2];
                X[j * 2 + 1] = B[j * nrhs * 2 + i * 2 + 1];
                B_col[j * 2] = B_original[j * nrhs * 2 + i * 2];
                B_col[j * 2 + 1] = B_original[j * nrhs * 2 + i * 2 + 1];
            }
            verifyMatrixVectorMult(A, X, B_col, n, false);
        }
    }

    @Test
    void testIdentityMatrix() {
        // Test with identity matrix
        int n = 3;
        int nrhs = 1;
        double[] A = new double[n * n * 2];
        for (int i = 0; i < n; i++) {
            A[i * n * 2 + i * 2] = 1.0; // Real part of diagonal
        }
        double[] B = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        double[] B_original = B.clone();

        int info = ZLAS.ztrtrs(BLAS.Uplo.Upper, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit,
                n, nrhs, A, 0, n, B, 0, nrhs);
        assertEquals(0, info);

        // Solution should be the same as B_original
        for (int i = 0; i < n * 2; i++) {
            assertEquals(B_original[i], B[i], TOL, "Solution should be same as B for identity matrix");
        }
    }

    @Test
    void testEmptyMatrix() {
        // Test with zero-sized matrix
        int n = 0;
        int nrhs = 0;
        double[] A = new double[0];
        double[] B = new double[0];

        int info = ZLAS.ztrtrs(BLAS.Uplo.Upper, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit,
                n, nrhs, A, 0, 1, B, 0, 1);
        assertEquals(0, info);
    }

    @Test
    void testSmallMatrix() {
        // Test a 1x1 matrix
        int n = 1;
        int nrhs = 1;
        double[] A = {2.0, 3.0}; // 2+3i
        double[] B = {4.0, 6.0}; // 4+6i
        double[] B_original = B.clone();

        int info = ZLAS.ztrtrs(BLAS.Uplo.Upper, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit,
                n, nrhs, A, 0, n, B, 0, nrhs);
        assertEquals(0, info);

        // Verify A * X = B_original
        verifySolution(A, B, B_original, n, nrhs);
    }

    @Test
    void testInvalidInputs() {
        // Test with invalid uplo
        int n = 2;
        int nrhs = 1;
        double[] A = {2.0, 0.0, 1.0, -1.0, 0.0, 0.0, 1.0, 0.0};
        double[] B = {3.0, 1.0, 1.0, 0.0};

        int info = ZLAS.ztrtrs(null, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit,
                n, nrhs, A, 0, n, B, 0, nrhs);
        assertEquals(-1, info);

        // Test with invalid trans
        info = ZLAS.ztrtrs(BLAS.Uplo.Upper, null, BLAS.Diag.NonUnit,
                n, nrhs, A, 0, n, B, 0, nrhs);
        assertEquals(-2, info);

        // Test with invalid diag
        info = ZLAS.ztrtrs(BLAS.Uplo.Upper, BLAS.Trans.NoTrans, null,
                n, nrhs, A, 0, n, B, 0, nrhs);
        assertEquals(-3, info);

        // Test with invalid n
        info = ZLAS.ztrtrs(BLAS.Uplo.Upper, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit,
                -1, nrhs, A, 0, n, B, 0, nrhs);
        assertEquals(-4, info);

        // Test with invalid nrhs
        info = ZLAS.ztrtrs(BLAS.Uplo.Upper, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit,
                n, -1, A, 0, n, B, 0, nrhs);
        assertEquals(-5, info);

        // Test with invalid lda
        info = ZLAS.ztrtrs(BLAS.Uplo.Upper, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit,
                n, nrhs, A, 0, n-1, B, 0, nrhs);
        assertEquals(-8, info);

        // Test with invalid ldb
        info = ZLAS.ztrtrs(BLAS.Uplo.Upper, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit,
                n, nrhs, A, 0, n, B, 0, nrhs-1);
        assertEquals(-11, info);
    }

    /**
     * Helper method to verify A * X = B
     */
    private void verifySolution(double[] A, double[] X, double[] B, int n, int nrhs) {
        verifySolution(A, X, B, n, nrhs, false);
    }

    /**
     * Helper method to verify A * X = B with optional unit diagonal
     */
    private void verifySolution(double[] A, double[] X, double[] B, int n, int nrhs, boolean unitDiag) {
        for (int i = 0; i < nrhs; i++) {
            double[] X_col = new double[n * 2];
            double[] B_col = new double[n * 2];
            for (int j = 0; j < n; j++) {
                X_col[j * 2] = X[j * nrhs * 2 + i * 2];
                X_col[j * 2 + 1] = X[j * nrhs * 2 + i * 2 + 1];
                B_col[j * 2] = B[j * nrhs * 2 + i * 2];
                B_col[j * 2 + 1] = B[j * nrhs * 2 + i * 2 + 1];
            }
            verifyMatrixVectorMult(A, X_col, B_col, n, unitDiag);
        }
    }

    /**
     * Helper method to verify A^H * X = B
     */
    private void verifyConjTransSolution(double[] A, double[] X, double[] B, int n, int nrhs) {
        for (int i = 0; i < nrhs; i++) {
            double[] X_col = new double[n * 2];
            double[] B_col = new double[n * 2];
            for (int j = 0; j < n; j++) {
                X_col[j * 2] = X[j * nrhs * 2 + i * 2];
                X_col[j * 2 + 1] = X[j * nrhs * 2 + i * 2 + 1];
                B_col[j * 2] = B[j * nrhs * 2 + i * 2];
                B_col[j * 2 + 1] = B[j * nrhs * 2 + i * 2 + 1];
            }
            verifyConjTransMatrixVectorMult(A, X_col, B_col, n);
        }
    }

    /**
     * Helper method to verify A^T * X = B
     */
    private void verifyTransSolution(double[] A, double[] X, double[] B, int n, int nrhs) {
        for (int i = 0; i < nrhs; i++) {
            double[] X_col = new double[n * 2];
            double[] B_col = new double[n * 2];
            for (int j = 0; j < n; j++) {
                X_col[j * 2] = X[j * nrhs * 2 + i * 2];
                X_col[j * 2 + 1] = X[j * nrhs * 2 + i * 2 + 1];
                B_col[j * 2] = B[j * nrhs * 2 + i * 2];
                B_col[j * 2 + 1] = B[j * nrhs * 2 + i * 2 + 1];
            }
            verifyTransMatrixVectorMult(A, X_col, B_col, n);
        }
    }

    /**
     * Helper method to compute A * x and compare to b
     */
    private void verifyMatrixVectorMult(double[] A, double[] x, double[] b, int n, boolean unitDiag) {
        double[] Ax = new double[n * 2];
        for (int i = 0; i < n; i++) {
            double sumR = 0.0;
            double sumI = 0.0;
            for (int j = 0; j < n; j++) {
                if (i == j && unitDiag) {
                    // Unit diagonal, contribution is x[j]
                    sumR += x[j * 2];
                    sumI += x[j * 2 + 1];
                } else {
                    double ar = A[i * n * 2 + j * 2];
                    double ai = A[i * n * 2 + j * 2 + 1];
                    double xr = x[j * 2];
                    double xi = x[j * 2 + 1];
                    sumR += ar * xr - ai * xi;
                    sumI += ar * xi + ai * xr;
                }
            }
            Ax[i * 2] = sumR;
            Ax[i * 2 + 1] = sumI;
        }

        for (int i = 0; i < n * 2; i++) {
            assertEquals(b[i], Ax[i], TOL, "A*x mismatch at index " + i);
        }
    }

    /**
     * Helper method to compute A^H * x and compare to b
     */
    private void verifyConjTransMatrixVectorMult(double[] A, double[] x, double[] b, int n) {
        double[] AHx = new double[n * 2];
        for (int i = 0; i < n; i++) {
            double sumR = 0.0;
            double sumI = 0.0;
            for (int j = 0; j < n; j++) {
                double ar = A[j * n * 2 + i * 2]; // Transpose
                double ai = -A[j * n * 2 + i * 2 + 1]; // Conjugate
                double xr = x[j * 2];
                double xi = x[j * 2 + 1];
                sumR += ar * xr - ai * xi;
                sumI += ar * xi + ai * xr;
            }
            AHx[i * 2] = sumR;
            AHx[i * 2 + 1] = sumI;
        }

        for (int i = 0; i < n * 2; i++) {
            assertEquals(b[i], AHx[i], TOL, "A^H*x mismatch at index " + i);
        }
    }

    /**
     * Helper method to compute A^T * x and compare to b
     */
    private void verifyTransMatrixVectorMult(double[] A, double[] x, double[] b, int n) {
        double[] ATx = new double[n * 2];
        for (int i = 0; i < n; i++) {
            double sumR = 0.0;
            double sumI = 0.0;
            for (int j = 0; j < n; j++) {
                double ar = A[j * n * 2 + i * 2]; // Transpose
                double ai = A[j * n * 2 + i * 2 + 1]; // No conjugate
                double xr = x[j * 2];
                double xi = x[j * 2 + 1];
                sumR += ar * xr - ai * xi;
                sumI += ar * xi + ai * xr;
            }
            ATx[i * 2] = sumR;
            ATx[i * 2 + 1] = sumI;
        }

        for (int i = 0; i < n * 2; i++) {
            assertEquals(b[i], ATx[i], TOL, "A^T*x mismatch at index " + i);
        }
    }

}

