package com.curioloop.yum4j.linalg.blas.cmplx;

import com.curioloop.yum4j.linalg.blas.BLAS;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class ZgeevTest {

    private static final double TOL = 1e-14;

    static final double[] GEEV_A = {
            -9.746816702273214e-01, 1.300189187790702e-02, 7.870846037424520e-01, 1.453534077157317e+00,
            1.158595579007404e+00, -2.646568332379561e-01, -8.206823183517105e-01, 2.720169166589619e+00,
            9.633761292443218e-01, 6.256673477650062e-01, 4.127809269364983e-01, -8.571575564162826e-01,
            8.220601599944900e-01, -1.070892498061112e+00, 1.896792982653947e+00, 4.824724152431853e-01,
            -2.453881160028705e-01, -2.234627853258508e-01, -7.537361643574896e-01, 7.140004940920919e-01,
            -8.895144296255233e-01, 4.732376245735448e-01, -8.158102849654383e-01, -7.282891265687277e-02,
            -7.710170941410412e-02, -8.467937180684050e-01, 3.411519748166435e-01, -1.514847224685865e+00,
            2.766907993300191e-01, -4.465149520670211e-01, 8.271832490360238e-01, 8.563987943234723e-01
    };

    static final double[] GEEV_W = {
            -1.047838700925767e+00, -1.332565642356848e+00,
            2.566410679031423e+00, 4.772885942553791e-01,
            -8.261099671932326e-01, 5.678764176737480e-01,
            -1.316693934792749e+00, 7.728813847863629e-01
    };

    @Test
    void testSquareMatrixEigenvaluesOnly() {
        int n = 3;
        double[] A = {
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0, 17.0, 18.0
        };
        double[] wr = new double[n];
        double[] wi = new double[n];
        double[] work = new double[1];

        int info = ZLAS.zgeev('N', 'N', n, A, n, wr, wi, null, 0, null, 0, work, 0, -1);
        assertEquals(0, info);

        int lwork = (int) work[0];
        work = new double[lwork];

        info = ZLAS.zgeev('N', 'N', n, A, n, wr, wi, null, 0, null, 0, work, 0, lwork);
        assertEquals(0, info);

        for (int i = 0; i < n; i++) {
            assertTrue(Double.isFinite(wr[i]));
            assertTrue(Double.isFinite(wi[i]));
        }
    }

    @Test
    void testSquareMatrixWithEigenvectors() {
        int n = 2;
        double[] A = {
            1.0, 1.0, 0.0, 1.0,
            0.0, 0.0, 0.0, 0.0
        };
        double[] wr = new double[n];
        double[] wi = new double[n];
        double[] vl = new double[n * n * 2];
        double[] vr = new double[n * n * 2];
        double[] work = new double[1];

        int info = ZLAS.zgeev('V', 'V', n, A, n, wr, wi, vl, n, vr, n, work, 0, -1);
        assertEquals(0, info);

        int lwork = (int) work[0];
        work = new double[lwork];

        info = ZLAS.zgeev('V', 'V', n, A, n, wr, wi, vl, n, vr, n, work, 0, lwork);
        assertEquals(0, info);

        assertEquals(1.0, wr[0], TOL);
        assertEquals(1.0, wi[0], TOL);
        assertEquals(0.0, wr[1], TOL);
        assertEquals(0.0, wi[1], TOL);

        for (int i = 0; i < n; i++) {
            double norm = 0.0;
            for (int j = 0; j < n; j++) {
                int idx = j * n * 2 + i * 2;
                norm += vr[idx] * vr[idx] + vr[idx + 1] * vr[idx + 1];
            }
            norm = Math.sqrt(norm);
            assertEquals(1.0, norm, TOL, "Right eigenvector " + i + " should be normalized");
        }

        for (int i = 0; i < n; i++) {
            double norm = 0.0;
            for (int j = 0; j < n; j++) {
                int idx = j * n * 2 + i * 2;
                norm += vl[idx] * vl[idx] + vl[idx + 1] * vl[idx + 1];
            }
            norm = Math.sqrt(norm);
            assertEquals(1.0, norm, TOL, "Left eigenvector " + i + " should be normalized");
        }
    }

    @Test
    void testLeftEigenvectorsOnly() {
        int n = 2;
        double[] A = {
            1.0, 1.0, 0.0, 1.0,
            0.0, 0.0, 0.0, 0.0
        };
        double[] wr = new double[n];
        double[] wi = new double[n];
        double[] vl = new double[n * n * 2];
        double[] work = new double[1];

        int info = ZLAS.zgeev('V', 'N', n, A, n, wr, wi, vl, n, null, 0, work, 0, -1);
        assertEquals(0, info);

        int lwork = (int) work[0];
        work = new double[lwork];

        info = ZLAS.zgeev('V', 'N', n, A, n, wr, wi, vl, n, null, 0, work, 0, lwork);
        assertEquals(0, info);

        assertEquals(1.0, wr[0], TOL);
        assertEquals(1.0, wi[0], TOL);
        assertEquals(0.0, wr[1], TOL);
        assertEquals(0.0, wi[1], TOL);

        for (int i = 0; i < n; i++) {
            double norm = 0.0;
            for (int j = 0; j < n; j++) {
                int idx = j * n * 2 + i * 2;
                norm += vl[idx] * vl[idx] + vl[idx + 1] * vl[idx + 1];
            }
            norm = Math.sqrt(norm);
            assertEquals(1.0, norm, TOL, "Left eigenvector " + i + " should be normalized");
        }
    }

    @Test
    void testRightEigenvectorsOnly() {
        int n = 2;
        double[] A = {
            1.0, 1.0, 0.0, 1.0,
            0.0, 0.0, 0.0, 0.0
        };
        double[] wr = new double[n];
        double[] wi = new double[n];
        double[] vr = new double[n * n * 2];
        double[] work = new double[1];

        int info = ZLAS.zgeev('N', 'V', n, A, n, wr, wi, null, 0, vr, n, work, 0, -1);
        assertEquals(0, info);

        int lwork = (int) work[0];
        work = new double[lwork];

        info = ZLAS.zgeev('N', 'V', n, A, n, wr, wi, null, 0, vr, n, work, 0, lwork);
        assertEquals(0, info);

        assertEquals(1.0, wr[0], TOL);
        assertEquals(1.0, wi[0], TOL);
        assertEquals(0.0, wr[1], TOL);
        assertEquals(0.0, wi[1], TOL);

        for (int i = 0; i < n; i++) {
            double norm = 0.0;
            for (int j = 0; j < n; j++) {
                int idx = j * n * 2 + i * 2;
                norm += vr[idx] * vr[idx] + vr[idx + 1] * vr[idx + 1];
            }
            norm = Math.sqrt(norm);
            assertEquals(1.0, norm, TOL, "Right eigenvector " + i + " should be normalized");
        }
    }

    @Test
    void testZeroMatrix() {
        int n = 3;
        double[] A = new double[n * n * 2];
        double[] wr = new double[n];
        double[] wi = new double[n];
        double[] work = new double[1];

        int info = ZLAS.zgeev('N', 'N', n, A, n, wr, wi, null, 0, null, 0, work, 0, -1);
        assertEquals(0, info);

        int lwork = (int) work[0];
        work = new double[lwork];

        info = ZLAS.zgeev('N', 'N', n, A, n, wr, wi, null, 0, null, 0, work, 0, lwork);
        assertEquals(0, info);

        for (int i = 0; i < n; i++) {
            assertEquals(0.0, wr[i], TOL, "Eigenvalue real part should be zero");
            assertEquals(0.0, wi[i], TOL, "Eigenvalue imaginary part should be zero");
        }
    }

    @Test
    void testIdentityMatrix() {
        int n = 3;
        double[] A = new double[n * n * 2];
        for (int i = 0; i < n; i++) {
            A[i * n * 2 + i * 2] = 1.0;
        }
        double[] wr = new double[n];
        double[] wi = new double[n];
        double[] work = new double[1];

        int info = ZLAS.zgeev('N', 'N', n, A, n, wr, wi, null, 0, null, 0, work, 0, -1);
        assertEquals(0, info);

        int lwork = (int) work[0];
        work = new double[lwork];

        info = ZLAS.zgeev('N', 'N', n, A, n, wr, wi, null, 0, null, 0, work, 0, lwork);
        assertEquals(0, info);

        for (int i = 0; i < n; i++) {
            assertEquals(1.0, wr[i], TOL, "Eigenvalue real part should be 1");
            assertEquals(0.0, wi[i], TOL, "Eigenvalue imaginary part should be zero");
        }
    }

    @Test
    void testEmptyMatrix() {
        int n = 0;
        double[] A = new double[0];
        double[] wr = new double[0];
        double[] wi = new double[0];
        double[] work = new double[0];

        int info = ZLAS.zgeev('N', 'N', n, A, n, wr, wi, null, 0, null, 0, work, 0, 0);
        assertEquals(0, info);
    }

    @Test
    void testSmallMatrix() {
        int n = 1;
        double[] A = {2.0, 3.0};
        double[] wr = new double[1];
        double[] wi = new double[1];
        double[] work = new double[1];

        int info = ZLAS.zgeev('N', 'N', n, A, n, wr, wi, null, 0, null, 0, work, 0, -1);
        assertEquals(0, info);

        int lwork = (int) work[0];
        work = new double[lwork];

        info = ZLAS.zgeev('N', 'N', n, A, n, wr, wi, null, 0, null, 0, work, 0, lwork);
        assertEquals(0, info);

        assertEquals(2.0, wr[0], TOL);
        assertEquals(3.0, wi[0], TOL);
    }

    @Test
    void testInvalidInputs() {
        int n = 2;
        double[] A = {1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0};
        double[] wr = new double[2];
        double[] wi = new double[2];
        double[] work = new double[1];

        int info = ZLAS.zgeev('X', 'N', n, A, n, wr, wi, null, 0, null, 0, work, 0, -1);
        assertEquals(-1, info);

        info = ZLAS.zgeev('N', 'X', n, A, n, wr, wi, null, 0, null, 0, work, 0, -1);
        assertEquals(-2, info);

        info = ZLAS.zgeev('N', 'N', -1, A, n, wr, wi, null, 0, null, 0, work, 0, -1);
        assertEquals(-3, info);

        info = ZLAS.zgeev('N', 'N', n, A, n-1, wr, wi, null, 0, null, 0, work, 0, -1);
        assertEquals(-5, info);

        double[] vl = new double[n * n * 2];
        info = ZLAS.zgeev('V', 'N', n, A, n, wr, wi, vl, n-1, null, 0, work, 0, -1);
        assertEquals(-9, info);

        double[] vr = new double[n * n * 2];
        info = ZLAS.zgeev('N', 'V', n, A, n, wr, wi, null, 0, vr, n-1, work, 0, -1);
        assertEquals(-11, info);
    }

    @Test
    void testEigenvaluesScipy() {
        int n = 4;
        double[] A = GEEV_A.clone();
        double[] wr = new double[n];
        double[] wi = new double[n];
        double[] vl = new double[n * n * 2];
        double[] vr = new double[n * n * 2];
        double[] work = new double[1];

        int info = ZLAS.zgeev('N', 'N', n, A, n, wr, wi, vl, n, vr, n, work, 0, -1);
        assertEquals(0, info);

        int lwork = (int) work[0];
        work = new double[lwork];

        info = ZLAS.zgeev('N', 'N', n, A, n, wr, wi, vl, n, vr, n, work, 0, lwork);
        assertEquals(0, info);

        double[] refWr = new double[n];
        double[] refWi = new double[n];
        for (int i = 0; i < n; i++) {
            refWr[i] = GEEV_W[i * 2];
            refWi[i] = GEEV_W[i * 2 + 1];
        }

        sortEigenvalues(refWr, refWi);
        sortEigenvalues(wr, wi);

        for (int i = 0; i < n; i++) {
            assertEquals(refWr[i], wr[i], 1e-10);
            assertEquals(refWi[i], wi[i], 1e-10);
        }
    }

    @Test
    void testRightEigenvectorsScipy() {
        int n = 4;
        double[] A_orig = GEEV_A.clone();
        double[] A = GEEV_A.clone();
        double[] wr = new double[n];
        double[] wi = new double[n];
        double[] vr = new double[n * n * 2];
        double[] work = new double[1];

        int info = ZLAS.zgeev('N', 'V', n, A, n, wr, wi, null, 1, vr, n, work, 0, -1);
        assertEquals(0, info);

        int lwork = (int) work[0];
        work = new double[lwork];

        info = ZLAS.zgeev('N', 'V', n, A, n, wr, wi, null, 1, vr, n, work, 0, lwork);
        assertEquals(0, info);

        double[] refWr = new double[n];
        double[] refWi = new double[n];
        for (int i = 0; i < n; i++) {
            refWr[i] = GEEV_W[i * 2];
            refWi[i] = GEEV_W[i * 2 + 1];
        }
        sortEigenvalues(refWr, refWi);

        double[] sortedWr = wr.clone();
        double[] sortedWi = wi.clone();
        sortEigenvalues(sortedWr, sortedWi);

        for (int i = 0; i < n; i++) {
            assertEquals(refWr[i], sortedWr[i], 1e-10);
            assertEquals(refWi[i], sortedWi[i], 1e-10);
        }

        for (int j = 0; j < n; j++) {
            double[] result = new double[n * 2];
            ZLAS.zgemv(BLAS.Trans.NoTrans, n, n, 1, 0, A_orig, 0, n, vr, j * 2, n, 0, 0, result, 0, 1);
            for (int i = 0; i < n; i++) {
                double vr_re = vr[(i * n + j) * 2];
                double vr_im = vr[(i * n + j) * 2 + 1];
                double expected_re = wr[j] * vr_re - wi[j] * vr_im;
                double expected_im = wr[j] * vr_im + wi[j] * vr_re;
                assertEquals(expected_re, result[i * 2], 1e-10);
                assertEquals(expected_im, result[i * 2 + 1], 1e-10);
            }
        }
    }

    @Test
    void testLeftEigenvectorsScipy() {
        int n = 4;
        double[] A_orig = GEEV_A.clone();
        double[] A = GEEV_A.clone();
        double[] wr = new double[n];
        double[] wi = new double[n];
        double[] vl = new double[n * n * 2];
        double[] work = new double[1];

        int info = ZLAS.zgeev('V', 'N', n, A, n, wr, wi, vl, n, null, 1, work, 0, -1);
        assertEquals(0, info);

        int lwork = (int) work[0];
        work = new double[lwork];

        info = ZLAS.zgeev('V', 'N', n, A, n, wr, wi, vl, n, null, 1, work, 0, lwork);
        assertEquals(0, info);

        double[] refWr = new double[n];
        double[] refWi = new double[n];
        for (int i = 0; i < n; i++) {
            refWr[i] = GEEV_W[i * 2];
            refWi[i] = GEEV_W[i * 2 + 1];
        }
        sortEigenvalues(refWr, refWi);

        double[] sortedWr = wr.clone();
        double[] sortedWi = wi.clone();
        sortEigenvalues(sortedWr, sortedWi);

        for (int i = 0; i < n; i++) {
            assertEquals(refWr[i], sortedWr[i], 1e-10);
            assertEquals(refWi[i], sortedWi[i], 1e-10);
        }

        for (int j = 0; j < n; j++) {
            double[] result = new double[n * 2];
            ZLAS.zgemv(BLAS.Trans.Conj, n, n, 1, 0, A_orig, 0, n, vl, j * 2, n, 0, 0, result, 0, 1);
            for (int i = 0; i < n; i++) {
                double vl_re = vl[(i * n + j) * 2];
                double vl_im = vl[(i * n + j) * 2 + 1];
                double expected_re = wr[j] * vl_re + wi[j] * vl_im;
                double expected_im = wr[j] * vl_im - wi[j] * vl_re;
                assertEquals(expected_re, result[i * 2], 1e-10);
                assertEquals(expected_im, result[i * 2 + 1], 1e-10);
            }
        }
    }

    @Test
    void testBothEigenvectorsScipy() {
        int n = 4;
        double[] A_orig = GEEV_A.clone();
        double[] A = GEEV_A.clone();
        double[] wr = new double[n];
        double[] wi = new double[n];
        double[] vl = new double[n * n * 2];
        double[] vr = new double[n * n * 2];
        double[] work = new double[1];

        int info = ZLAS.zgeev('V', 'V', n, A, n, wr, wi, vl, n, vr, n, work, 0, -1);
        assertEquals(0, info);

        int lwork = (int) work[0];
        work = new double[lwork];

        info = ZLAS.zgeev('V', 'V', n, A, n, wr, wi, vl, n, vr, n, work, 0, lwork);
        assertEquals(0, info);

        double[] refWr = new double[n];
        double[] refWi = new double[n];
        for (int i = 0; i < n; i++) {
            refWr[i] = GEEV_W[i * 2];
            refWi[i] = GEEV_W[i * 2 + 1];
        }
        sortEigenvalues(refWr, refWi);

        double[] sortedWr = wr.clone();
        double[] sortedWi = wi.clone();
        sortEigenvalues(sortedWr, sortedWi);

        for (int i = 0; i < n; i++) {
            assertEquals(refWr[i], sortedWr[i], 1e-10);
            assertEquals(refWi[i], sortedWi[i], 1e-10);
        }

        for (int j = 0; j < n; j++) {
            double[] result = new double[n * 2];
            ZLAS.zgemv(BLAS.Trans.NoTrans, n, n, 1, 0, A_orig, 0, n, vr, j * 2, n, 0, 0, result, 0, 1);
            for (int i = 0; i < n; i++) {
                double vr_re = vr[(i * n + j) * 2];
                double vr_im = vr[(i * n + j) * 2 + 1];
                double expected_re = wr[j] * vr_re - wi[j] * vr_im;
                double expected_im = wr[j] * vr_im + wi[j] * vr_re;
                assertEquals(expected_re, result[i * 2], 1e-10);
                assertEquals(expected_im, result[i * 2 + 1], 1e-10);
            }
        }

        for (int j = 0; j < n; j++) {
            double[] result = new double[n * 2];
            ZLAS.zgemv(BLAS.Trans.Conj, n, n, 1, 0, A_orig, 0, n, vl, j * 2, n, 0, 0, result, 0, 1);
            for (int i = 0; i < n; i++) {
                double vl_re = vl[(i * n + j) * 2];
                double vl_im = vl[(i * n + j) * 2 + 1];
                double expected_re = wr[j] * vl_re + wi[j] * vl_im;
                double expected_im = wr[j] * vl_im - wi[j] * vl_re;
                assertEquals(expected_re, result[i * 2], 1e-10);
                assertEquals(expected_im, result[i * 2 + 1], 1e-10);
            }
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
