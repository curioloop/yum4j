/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.linalg.mat;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

class SVDReconstructionTest {

    private static final double EPSILON = 1e-10;

    @Test
    void testSVDReconstruction2x2() {
        double[] A = {
            1.0, 2.0,
            3.0, 4.0
        };
        int m = 2, n = 2;
        int lda = n;

        double[] d = new double[Math.min(m, n)];
        double[] e = new double[Math.min(m, n)];
        double[] tauQ = new double[Math.min(m, n)];
        double[] tauP = new double[Math.min(m, n)];
        double[] work = new double[Math.max(m, n) * 10];

        double[] AWork = A.clone();
        com.curioloop.yum4j.linalg.blas.BLAS.dgebd2(m, n, AWork, 0, lda, d, 0, e, 0, tauQ, 0, tauP, 0, work, 0);

        double[] Q = new double[m * m];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                Q[i * m + j] = AWork[i * lda + j];
            }
        }

        com.curioloop.yum4j.linalg.blas.BLAS.dorgbr('Q', m, m, n, Q, 0, m, tauQ, 0, work, 0, work.length);

        double[] PT = new double[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                PT[i * n + j] = AWork[i * lda + j];
            }
        }

        com.curioloop.yum4j.linalg.blas.BLAS.dorgbr('P', n, n, n, PT, 0, n, tauP, 0, work, 0, work.length);

        double[] S = d.clone();
        double[] E = e.clone();

        int ncvt = n;
        int nru = m;
        int ldvt = n;
        int ldu = m;

        boolean ok = com.curioloop.yum4j.linalg.blas.BLAS.dbdsqr(com.curioloop.yum4j.linalg.blas.BLAS.Uplo.Upper, Math.min(m, n), ncvt, nru, 0, S, 0, E, 0, PT, 0, ldvt, Q, 0, ldu, null, 0, 0, work, 0);
        assertThat(ok).isTrue();

        double[] reconstructed = new double[m * n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0;
                for (int k = 0; k < Math.min(m, n); k++) {
                    sum += Q[i * m + k] * S[k] * PT[k * n + j];
                }
                reconstructed[i * n + j] = sum;
            }
        }

        for (int i = 0; i < m * n; i++) {
            assertThat(reconstructed[i]).isCloseTo(A[i], org.assertj.core.data.Offset.offset(EPSILON));
        }
    }

    @Test
    void testSVDReconstruction3x2() {
        double[] A = {
            1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0
        };
        int m = 3, n = 2;
        int lda = n;

        double[] d = new double[Math.min(m, n)];
        double[] e = new double[Math.min(m, n)];
        double[] tauQ = new double[Math.min(m, n)];
        double[] tauP = new double[Math.min(m, n)];
        double[] work = new double[Math.max(m, n) * 10];

        double[] AWork = A.clone();
        com.curioloop.yum4j.linalg.blas.BLAS.dgebd2(m, n, AWork, 0, lda, d, 0, e, 0, tauQ, 0, tauP, 0, work, 0);

        double[] Q = new double[m * n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                Q[i * n + j] = AWork[i * lda + j];
            }
        }

        com.curioloop.yum4j.linalg.blas.BLAS.dorgbr('Q', m, n, n, Q, 0, n, tauQ, 0, work, 0, work.length);

        double[] PT = new double[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                PT[i * n + j] = AWork[i * lda + j];
            }
        }

        com.curioloop.yum4j.linalg.blas.BLAS.dorgbr('P', n, n, n, PT, 0, n, tauP, 0, work, 0, work.length);

        double[] S = d.clone();
        double[] E = e.clone();

        int ncvt = n;
        int nru = m;
        int ldvt = n;
        int ldu = n;

        boolean ok = com.curioloop.yum4j.linalg.blas.BLAS.dbdsqr(com.curioloop.yum4j.linalg.blas.BLAS.Uplo.Upper, Math.min(m, n), ncvt, nru, 0, S, 0, E, 0, PT, 0, ldvt, Q, 0, ldu, null, 0, 0, work, 0);
        assertThat(ok).isTrue();

        double[] reconstructed = new double[m * n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0;
                for (int k = 0; k < Math.min(m, n); k++) {
                    sum += Q[i * n + k] * S[k] * PT[k * n + j];
                }
                reconstructed[i * n + j] = sum;
            }
        }

        for (int i = 0; i < m * n; i++) {
            assertThat(reconstructed[i]).isCloseTo(A[i], org.assertj.core.data.Offset.offset(EPSILON));
        }
    }
}
