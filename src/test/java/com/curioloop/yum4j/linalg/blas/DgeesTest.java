/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.linalg.blas;

import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

class DgeesTest {

    private static final double TOL = 1e-10;

    @Test
    void testSimpleMatrix() {
        double[] A = {
            1, 2, 3,
            4, 5, 6,
            7, 8, 9
        };
        int n = 3;

        double[] wr = new double[n];
        double[] wi = new double[n];
        double[] vs = new double[n * n];
        double[] work = new double[100];
        boolean[] bwork = new boolean[n];

        int info = Dgees.dgees('V', 'N', null, n, A, n, wr, wi, vs, n, work, 0, work.length, bwork);

        assertEquals(0, info);
        
        double[] expectedReal = {16.116843969807043, -1.1168439698070427, 0};
        double[] expectedImag = {0, 0, 0};

        for (int i = 0; i < n; i++) {
            assertEquals(expectedReal[i], wr[i], 1e-8, "Real eigenvalue mismatch at " + i);
            assertEquals(expectedImag[i], wi[i], 1e-8, "Imag eigenvalue mismatch at " + i);
        }
    }

    @Test
    void testDiagonalMatrix() {
        double[] A = {
            1, 0, 0,
            0, 2, 0,
            0, 0, 3
        };
        int n = 3;

        double[] wr = new double[n];
        double[] wi = new double[n];
        double[] vs = new double[n * n];
        double[] work = new double[100];
        boolean[] bwork = new boolean[n];

        int info = Dgees.dgees('V', 'N', null, n, A, n, wr, wi, vs, n, work, 0, work.length, bwork);

        assertEquals(0, info);

        double[] expected = {1, 2, 3};
        for (int i = 0; i < n; i++) {
            assertEquals(expected[i], wr[i], TOL);
            assertEquals(0, wi[i], TOL);
        }
    }

    @Test
    void testComplexEigenvalues() {
        double[] A = {
            0, -1,
            1, 0
        };
        int n = 2;

        double[] wr = new double[n];
        double[] wi = new double[n];
        double[] vs = new double[n * n];
        double[] work = new double[100];
        boolean[] bwork = new boolean[n];

        int info = Dgees.dgees('V', 'N', null, n, A, n, wr, wi, vs, n, work, 0, work.length, bwork);

        assertEquals(0, info);

        assertEquals(0, wr[0], TOL);
        assertEquals(1, wi[0], TOL);
        assertEquals(0, wr[1], TOL);
        assertEquals(-1, wi[1], TOL);
    }

    @Test
    void testNoVectors() {
        double[] A = {
            1, 2,
            3, 4
        };
        int n = 2;

        double[] wr = new double[n];
        double[] wi = new double[n];
        double[] work = new double[100];
        boolean[] bwork = new boolean[n];

        int info = Dgees.dgees('N', 'N', null,  n, A, n, wr, wi, null, n, work, 0, work.length, bwork);

        assertEquals(0, info);

        double trace = 5;
        double det = -2;
        double discriminant = trace * trace - 4 * det;
        double sqrtDisc = Math.sqrt(discriminant);

        double e1 = (trace + sqrtDisc) / 2;
        double e2 = (trace - sqrtDisc) / 2;

        boolean[] used = new boolean[n];
        for (int i = 0; i < n; i++) {
            boolean found = false;
            for (int j = 0; j < 2; j++) {
                if (!used[j]) {
                    double expected = (j == 0) ? e1 : e2;
                    if (Math.abs(wr[i] - expected) < TOL && Math.abs(wi[i]) < TOL) {
                        used[j] = true;
                        found = true;
                        break;
                    }
                }
            }
            assertTrue(found, "Unexpected eigenvalue " + wr[i]);
        }
    }

    @Test
    void testWithSort() {
        double[] A = {
            3, 1, 0,
            0, 2, 0,
            0, 0, 1
        };
        int n = 3;

        double[] wr = new double[n];
        double[] wi = new double[n];
        double[] vs = new double[n * n];
        double[] work = new double[100];
        boolean[] bwork = new boolean[n];

        Select select = (r, i) -> r > 1.5;

        int info = Dgees.dgees('V', 'S', select, n, A, n, wr, wi, vs, n, work, 0, work.length, bwork);

        assertEquals(0, info);
        int sdim = (int) work[0];
        assertEquals(2, sdim);
    }

    @Test
    void testOffsetEigenvalueOutputUsesSharedBacking() {
        double[] A = {
            2, 0,
            0, -1
        };
        int n = 2;
        int wrOff = 1;
        int wiOff = wrOff + n;

        double[] eigenvalues = {123.0, 0.0, 0.0, 0.0, 0.0, 456.0};
        double[] vs = new double[n * n];
        double[] work = new double[100];
        boolean[] bwork = new boolean[n];

        int info = Dgees.dgees('V', 'S', (wr, wi) -> wr < 0.0, n, A, n,
            eigenvalues, wrOff, eigenvalues, wiOff,
            vs, n, work, 0, work.length, bwork);

        assertEquals(0, info);
        assertEquals(123.0, eigenvalues[0], 0.0);
        assertEquals(456.0, eigenvalues[eigenvalues.length - 1], 0.0);
        assertEquals(-1.0, eigenvalues[wrOff], TOL);
        assertEquals(2.0, eigenvalues[wrOff + 1], TOL);
        assertEquals(0.0, eigenvalues[wiOff], TOL);
        assertEquals(0.0, eigenvalues[wiOff + 1], TOL);
        assertEquals(1.0, Math.abs(vs[0]), TOL);
    }

    @Test
    void testOffsetMatrixAndVectorStorageUsePaddedBacking() {
        int n = 3;
        int lda = 5;
        int aOff = 4;
        double sentinel = -777.0;

        double[] paddedA = new double[aOff + lda * n + 6];
        Arrays.fill(paddedA, sentinel);
        writeMatrix(paddedA, aOff, lda, n, new double[] {
            0, 0, 0,
            1, 2, 0,
            0, 4, 3
        });

        int ldvs = 4;
        int vsOff = 3;
        double[] paddedVs = new double[vsOff + ldvs * n + 5];
        Arrays.fill(paddedVs, sentinel);

        double[] wr = new double[n];
        double[] wi = new double[n];
        double[] work = new double[256];
        boolean[] bwork = new boolean[n];

        int info = Dgees.dgees('V', 'N', null, n,
            paddedA, aOff, lda,
            wr, 0, wi, 0,
            paddedVs, vsOff, ldvs,
            work, 0, work.length, bwork);

        assertEquals(0, info);
        assertPaddingUnchanged(paddedA, aOff, lda, n, n, sentinel);
        assertPaddingUnchanged(paddedVs, vsOff, ldvs, n, n, sentinel);

        double[] wrSorted = wr.clone();
        Arrays.sort(wrSorted);
        assertArrayEquals(new double[] {0.0, 2.0, 3.0}, wrSorted, TOL);
        for (double imag : wi) {
            assertEquals(0.0, imag, TOL);
        }

        assertQuasiTriangular(paddedA, aOff, lda, n);
        assertOrthogonal(paddedVs, vsOff, ldvs, n);
    }

    @Test
    void testWorkOffsetUsesOnlyActiveSlice() {
        int n = 3;
        double sentinel = -321.0;
        double[] a = {
            0.0, 0.0, 0.0,
            1.0, 2.0, 0.0,
            0.0, 4.0, 3.0
        };
        double[] wr = new double[n];
        double[] wi = new double[n];
        double[] work = new double[300];
        Arrays.fill(work, sentinel);
        int workOff = 17;
        int lwork = 200;

        int info = Dgees.dgees('N', 'N', null, n,
            a, 0, n,
            wr, 0, wi, 0,
            null, 0, n,
            work, workOff, lwork, null);

        assertEquals(0, info);
        for (int i = 0; i < workOff; i++) {
            assertEquals(sentinel, work[i], 0.0, "Unexpected write before work slice at " + i);
        }
        for (int i = workOff + lwork; i < work.length; i++) {
            assertEquals(sentinel, work[i], 0.0, "Unexpected write after work slice at " + i);
        }
    }

    @Test
    void testRandomMatrix() {
        Random rnd = new Random(42);
        int n = 10;

        double[] A = new double[n * n];
        for (int i = 0; i < n * n; i++) {
            A[i] = rnd.nextGaussian();
        }

        double[] wr = new double[n];
        double[] wi = new double[n];
        double[] vs = new double[n * n];
        double[] work = new double[n * 10];
        boolean[] bwork = new boolean[n];

        int info = Dgees.dgees('V', 'N', null, n, A, n, wr, wi, vs, n, work, 0, work.length, bwork);

        assertEquals(0, info);

        for (int i = 0; i < n; i++) {
            if (wi[i] != 0) {
                assertTrue(i < n - 1, "Complex eigenvalue at last position");
                assertEquals(wr[i], wr[i + 1], TOL, "Conjugate pair real parts differ");
                assertEquals(wi[i], -wi[i + 1], TOL, "Conjugate pair imag parts not opposite");
                assertTrue(wi[i] > 0, "First of conjugate pair should have positive imag");
                i++;
            }
        }
    }

    @Test
    void testWorkspaceQuery() {
        double[] A = new double[4];
        double[] wr = new double[2];
        double[] wi = new double[2];
        double[] work = new double[1];

        int info = Dgees.dgees('V', 'N', null, 2, A, 2, wr, wi, null, 2, work, 0, -1, null);

        assertEquals(0, info);
        assertTrue(work[0] > 0);
    }

    private static void writeMatrix(double[] backing, int off, int ld, int n, double[] values) {
        for (int row = 0; row < n; row++) {
            System.arraycopy(values, row * n, backing, off + row * ld, n);
        }
    }

    private static void assertPaddingUnchanged(double[] backing, int off, int ld, int rows, int cols, double sentinel) {
        for (int i = 0; i < off; i++) {
            assertEquals(sentinel, backing[i], 0.0, "Unexpected write before active block at " + i);
        }
        for (int row = 0; row < rows; row++) {
            for (int col = cols; col < ld; col++) {
                int index = off + row * ld + col;
                assertEquals(sentinel, backing[index], 0.0, "Unexpected write in row padding at " + index);
            }
        }
        int end = off + rows * ld;
        for (int i = end; i < backing.length; i++) {
            assertEquals(sentinel, backing[i], 0.0, "Unexpected write after active block at " + i);
        }
    }

    private static void assertQuasiTriangular(double[] A, int aOff, int lda, int n) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < i - 1; j++) {
                assertEquals(0.0, A[aOff + i * lda + j], TOL, "Schur form violation at (" + i + "," + j + ")");
            }
        }
    }

    private static void assertOrthogonal(double[] z, int zOff, int ldz, int n) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double dot = 0.0;
                for (int k = 0; k < n; k++) {
                    dot += z[zOff + i * ldz + k] * z[zOff + j * ldz + k];
                }
                assertEquals(i == j ? 1.0 : 0.0, dot, 1e-8, "Non-orthogonal vectors at (" + i + "," + j + ")");
            }
        }
    }

    @Test
    void testZeroMatrix() {
        double[] A = new double[4];
        int n = 2;

        double[] wr = new double[n];
        double[] wi = new double[n];
        double[] vs = new double[n * n];
        double[] work = new double[100];
        boolean[] bwork = new boolean[n];

        int info = Dgees.dgees('V', 'N', null, n, A, n, wr, wi, vs, n, work, 0, work.length, bwork);

        assertEquals(0, info);
        assertEquals(0, wr[0], TOL);
        assertEquals(0, wr[1], TOL);
    }

    @Test
    void testIdentityMatrix() {
        double[] A = {
            1, 0, 0,
            0, 1, 0,
            0, 0, 1
        };
        int n = 3;

        double[] wr = new double[n];
        double[] wi = new double[n];
        double[] vs = new double[n * n];
        double[] work = new double[100];
        boolean[] bwork = new boolean[n];

        int info = Dgees.dgees('V', 'N', null, n, A, n, wr, wi, vs, n, work, 0, work.length, bwork);

        assertEquals(0, info);
        for (int i = 0; i < n; i++) {
            assertEquals(1, wr[i], TOL);
            assertEquals(0, wi[i], TOL);
        }
    }

    @Test
    void testSchurForm() {
        double[] A = {
            4, 1, 0,
            0, 3, 1,
            0, 0, 2
        };
        int n = 3;

        double[] wr = new double[n];
        double[] wi = new double[n];
        double[] vs = new double[n * n];
        double[] work = new double[100];
        boolean[] bwork = new boolean[n];

        int info = Dgees.dgees('V', 'N', null, n, A, n, wr, wi, vs, n, work, 0, work.length, bwork);

        assertEquals(0, info);

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < i - 1; j++) {
                assertEquals(0, A[i * n + j], TOL, "Schur form violation at (" + i + "," + j + ")");
            }
        }
    }
}
