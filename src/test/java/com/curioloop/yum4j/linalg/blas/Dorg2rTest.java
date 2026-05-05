package com.curioloop.yum4j.linalg.blas;

import org.junit.jupiter.api.Test;
import static org.assertj.core.api.Assertions.assertThat;

class Dorg2rTest {

    @Test
    void testDorg2rWithDgeqr2() {
        int m = 4, n = 3, k = 3;
        
        double[] A = {
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
            10.0, 11.0, 12.0
        };
        double[] tau = new double[k];
        double[] work = new double[n];
        
        Dgeqr.dgeqr2(m, n, A, 0, n, tau, 0, work, 0);
        
        Dgeqr.dorg2r(m, n, k, A, 0, n, tau, 0, work, 0);

        double[] QtQ = new double[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0;
                for (int l = 0; l < m; l++) {
                    sum += A[l * n + i] * A[l * n + j];
                }
                QtQ[i * n + j] = sum;
            }
        }

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double expected = (i == j) ? 1.0 : 0.0;
                assertThat(QtQ[i * n + j]).isCloseTo(expected, org.assertj.core.data.Offset.offset(1e-10));
            }
        }
    }
}
