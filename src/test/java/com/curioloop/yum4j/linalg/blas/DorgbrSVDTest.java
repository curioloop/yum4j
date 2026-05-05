package com.curioloop.yum4j.linalg.blas;

import org.junit.jupiter.api.Test;
import static org.assertj.core.api.Assertions.assertThat;

class DorgbrSVDTest {

    @Test
    void testDorgbrPInSVD() {
        int m = 4, n = 3, minMN = 3;
        
        double[] A = {
            0.8147, 0.9134, 0.9,
            0.9058, 0.6324, 0.9,
            0.1270, 0.0975, 0.1,
            1.6, 2.8, -3.5
        };
        
        double[] S = new double[minMN];
        double[] e = new double[minMN];
        double[] tauQ = new double[minMN];
        double[] tauP = new double[minMN];
        double[] work = new double[4 * minMN];
        
        BLAS.dgebd2(m, n, A, 0, n, S, 0, e, 0, tauQ, 0, tauP, 0, work, 0);

        double[] VT = new double[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                VT[i * n + j] = A[i * n + j];
            }
        }
        
        BLAS.dorgbr('P', n, n, n, VT, 0, n, tauP, 0, work, 0, work.length);

        double[] VTVTt = new double[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0;
                for (int l = 0; l < n; l++) {
                    sum += VT[i * n + l] * VT[j * n + l];
                }
                VTVTt[i * n + j] = sum;
            }
        }

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double expected = (i == j) ? 1.0 : 0.0;
                assertThat(VTVTt[i * n + j]).isCloseTo(expected, org.assertj.core.data.Offset.offset(1e-10));
            }
        }
    }
}
