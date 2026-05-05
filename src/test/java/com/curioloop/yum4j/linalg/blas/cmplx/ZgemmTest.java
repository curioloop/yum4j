/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.linalg.blas.cmplx;

import com.curioloop.yum4j.linalg.blas.BLAS;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class ZgemmTest {

    private static final double TOL = 1e-10;

    @Test
    void testBasic() {
        // A = [[1+2i, 3+4i], [5+6i, 7+8i]]
        double[] A = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
        // B = [[9+10i, 11+12i], [13+14i, 15+16i]]
        double[] B = {9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0};
        // C = [[0+0i, 0+0i], [0+0i, 0+0i]]
        double[] C = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

        // alpha = 1.0 + 0.0i, beta = 0.0 + 0.0i
        Zgemm.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, 2, 2, 2,
                1.0, 0.0, A, 0, 2, B, 0, 2, 0.0, 0.0, C, 0, 2);

        // Expected: C = A * B
        // C[0][0] = (1+2i)(9+10i) + (3+4i)(13+14i) = (-11+28i) + (-17+94i) = -28+122i
        assertEquals(-28.0, C[0], TOL);
        assertEquals(122.0, C[1], TOL);

        // C[0][1] = (1+2i)(11+12i) + (3+4i)(15+16i) = (-13+34i) + (-19+108i) = -32+142i
        assertEquals(-32.0, C[2], TOL);
        assertEquals(142.0, C[3], TOL);

        // C[1][0] = (5+6i)(9+10i) + (7+8i)(13+14i) = (-15+104i) + (-21+202i) = -36+306i
        assertEquals(-36.0, C[4], TOL);
        assertEquals(306.0, C[5], TOL);

        // C[1][1] = (5+6i)(11+12i) + (7+8i)(15+16i) = (-17+126i) + (-23+232i) = -40+358i
        assertEquals(-40.0, C[6], TOL);
        assertEquals(358.0, C[7], TOL);
    }

    @Test
    void testWithAlpha() {
        // A = [[1+0i, 0+0i], [0+0i, 1+0i]]
        double[] A = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0};
        // B = [[2+3i, 4+5i], [6+7i, 8+9i]]
        double[] B = {2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
        // C = [[0+0i, 0+0i], [0+0i, 0+0i]]
        double[] C = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

        // alpha = 2.0 + 0.0i, beta = 0.0 + 0.0i
        Zgemm.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, 2, 2, 2,
                2.0, 0.0, A, 0, 2, B, 0, 2, 0.0, 0.0, C, 0, 2);

        // Expected: C = 2 * A * B = 2 * B
        assertEquals(4.0, C[0], TOL);  // 2*2
        assertEquals(6.0, C[1], TOL);  // 2*3
        assertEquals(8.0, C[2], TOL);  // 2*4
        assertEquals(10.0, C[3], TOL); // 2*5
        assertEquals(12.0, C[4], TOL); // 2*6
        assertEquals(14.0, C[5], TOL); // 2*7
        assertEquals(16.0, C[6], TOL); // 2*8
        assertEquals(18.0, C[7], TOL); // 2*9
    }

    @Test
    void testWithBeta() {
        // A = [[1+0i, 0+0i], [0+0i, 1+0i]]
        double[] A = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0};
        // B = [[1+0i, 1+0i], [1+0i, 1+0i]]
        double[] B = {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0};
        // C = [[1+2i, 3+4i], [5+6i, 7+8i]]
        double[] C = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};

        // alpha = 1.0 + 0.0i, beta = 2.0 + 0.0i
        Zgemm.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, 2, 2, 2,
                1.0, 0.0, A, 0, 2, B, 0, 2, 2.0, 0.0, C, 0, 2);

        // Expected: C = 1*A*B + 2*C = B + 2*C
        assertEquals(1.0 + 2*1.0, C[0], TOL);  // 1 + 2*1
        assertEquals(0.0 + 2*2.0, C[1], TOL);  // 0 + 2*2
        assertEquals(1.0 + 2*3.0, C[2], TOL);  // 1 + 2*3
        assertEquals(0.0 + 2*4.0, C[3], TOL);  // 0 + 2*4
        assertEquals(1.0 + 2*5.0, C[4], TOL);  // 1 + 2*5
        assertEquals(0.0 + 2*6.0, C[5], TOL);  // 0 + 2*6
        assertEquals(1.0 + 2*7.0, C[6], TOL);  // 1 + 2*7
        assertEquals(0.0 + 2*8.0, C[7], TOL);  // 0 + 2*8
    }

    @Test
    void testTransposeA() {
        // A = [[1+2i, 3+4i], [5+6i, 7+8i]]
        double[] A = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
        // B = [[9+10i, 11+12i], [13+14i, 15+16i]]
        double[] B = {9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0};
        // C = [[0+0i, 0+0i], [0+0i, 0+0i]]
        double[] C = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

        // alpha = 1.0 + 0.0i, beta = 0.0 + 0.0i, transA = Trans
        Zgemm.zgemm(BLAS.Trans.Trans, BLAS.Trans.NoTrans, 2, 2, 2,
                1.0, 0.0, A, 0, 2, B, 0, 2, 0.0, 0.0, C, 0, 2);

        // Expected: C = A^T * B
        // C[0][0] = (1+2i)(9+10i) + (5+6i)(13+14i) = (-11+28i) + (-19+148i) = -30+176i
        assertEquals(-30.0, C[0], TOL);
        assertEquals(176.0, C[1], TOL);

        // C[0][1] = (1+2i)(11+12i) + (5+6i)(15+16i) = (-13+34i) + (-21+170i) = -34+204i
        assertEquals(-34.0, C[2], TOL);
        assertEquals(204.0, C[3], TOL);

        // C[1][0] = (3+4i)(9+10i) + (7+8i)(13+14i) = (-13+66i) + (-21+202i) = -34+268i
        assertEquals(-34.0, C[4], TOL);
        assertEquals(268.0, C[5], TOL);

        // C[1][1] = (3+4i)(11+12i) + (7+8i)(15+16i) = (-15+80i) + (-23+232i) = -38+312i
        assertEquals(-38.0, C[6], TOL);
        assertEquals(312.0, C[7], TOL);
    }

    @Test
    void testConjTransposeA() {
        // A = [[1+2i, 3+4i], [5+6i, 7+8i]]
        double[] A = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
        // B = [[9+10i, 11+12i], [13+14i, 15+16i]]
        double[] B = {9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0};
        // C = [[0+0i, 0+0i], [0+0i, 0+0i]]
        double[] C = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

        // alpha = 1.0 + 0.0i, beta = 0.0 + 0.0i, transA = Conj
        Zgemm.zgemm(BLAS.Trans.Conj, BLAS.Trans.NoTrans, 2, 2, 2,
                1.0, 0.0, A, 0, 2, B, 0, 2, 0.0, 0.0, C, 0, 2);

        // Expected: C = A^H * B
        // C[0][0] = (1-2i)(9+10i) + (5-6i)(13+14i) = (29-8i) + (149-8i) = 178-16i
        assertEquals(178.0, C[0], TOL);
        assertEquals(-16.0, C[1], TOL);

        // C[0][1] = (1-2i)(11+12i) + (5-6i)(15+16i) = (35-10i) + (171-10i) = 206-20i
        assertEquals(206.0, C[2], TOL);
        assertEquals(-20.0, C[3], TOL);

        // C[1][0] = (3-4i)(9+10i) + (7-8i)(13+14i) = (67-6i) + (203-6i) = 270-12i
        assertEquals(270.0, C[4], TOL);
        assertEquals(-12.0, C[5], TOL);

        // C[1][1] = (3-4i)(11+12i) + (7-8i)(15+16i) = (71-10i) + (243-6i) = 314-16i
        assertEquals(314.0, C[6], TOL);
        assertEquals(-16.0, C[7], TOL);
    }

    @Test
    void testEmpty() {
        // Test with zero-sized matrices
        Zgemm.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, 0, 0, 0,
                1.0, 0.0, new double[0], 0, 0, new double[0], 0, 0, 0.0, 0.0, new double[0], 0, 0);
    }

    @Test
    void testNoTransConjThinKMatchesReference() {
        int m = 96;
        int n = 24;
        int k = 32;
        int lda = 37;
        int ldb = 35;
        int ldc = 29;
        int aStart = 10;
        int bStart = 14;
        int cStart = 8;

        double[] compactA = ZBlasTestSupport.randomComplexMatrix(m, k, 211L);
        double[] compactB = ZBlasTestSupport.randomComplexMatrix(n, k, 223L);
        double[] compactC = ZBlasTestSupport.randomComplexMatrix(m, n, 227L);

        double[] expectedA = new double[aStart + (m - 1) * lda * 2 + k * 2 + 8];
        double[] actualA = expectedA.clone();
        ZBlasTestSupport.embedMatrix(compactA, m, k, 0, k, expectedA, aStart, lda);
        ZBlasTestSupport.embedMatrix(compactA, m, k, 0, k, actualA, aStart, lda);

        double[] expectedB = new double[bStart + (n - 1) * ldb * 2 + k * 2 + 8];
        double[] actualB = expectedB.clone();
        ZBlasTestSupport.embedMatrix(compactB, n, k, 0, k, expectedB, bStart, ldb);
        ZBlasTestSupport.embedMatrix(compactB, n, k, 0, k, actualB, bStart, ldb);

        double[] expectedC = new double[cStart + (m - 1) * ldc * 2 + n * 2 + 8];
        double[] actualC = expectedC.clone();
        ZBlasTestSupport.embedMatrix(compactC, m, n, 0, n, expectedC, cStart, ldc);
        ZBlasTestSupport.embedMatrix(compactC, m, n, 0, n, actualC, cStart, ldc);

        ZBlasTestSupport.refZgemm(BLAS.Trans.NoTrans, BLAS.Trans.Conj, m, n, k, -0.625, 0.375,
            expectedA, aStart, lda, expectedB, bStart, ldb, 0.5, -0.25, expectedC, cStart, ldc);
        Zgemm.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.Conj, m, n, k,
            -0.625, 0.375, actualA, aStart / 2, lda, actualB, bStart / 2, ldb, 0.5, -0.25, actualC, cStart / 2, ldc);

        ZBlasTestSupport.assertArrayClose("Zgemm NT thin-k A unchanged", expectedA, actualA, TOL);
        ZBlasTestSupport.assertArrayClose("Zgemm NT thin-k B unchanged", expectedB, actualB, TOL);
        ZBlasTestSupport.assertArrayClose("Zgemm NT thin-k result", expectedC, actualC, TOL);
    }

    @Test
    void testNoTransConjThinKOffsetWorkspaceMatchesReference() {
        int m = 72;
        int n = 16;
        int k = 24;
        int lda = 28;
        int ldb = 30;
        int ldc = 21;
        int aStart = 6;
        int bStart = 12;
        int cStart = 10;

        double[] compactA = ZBlasTestSupport.randomComplexMatrix(m, k, 229L);
        double[] compactB = ZBlasTestSupport.randomComplexMatrix(n, k, 233L);
        double[] compactC = ZBlasTestSupport.randomComplexMatrix(m, n, 239L);

        double[] expectedA = new double[aStart + (m - 1) * lda * 2 + k * 2 + 8];
        double[] actualA = expectedA.clone();
        ZBlasTestSupport.embedMatrix(compactA, m, k, 0, k, expectedA, aStart, lda);
        ZBlasTestSupport.embedMatrix(compactA, m, k, 0, k, actualA, aStart, lda);

        double[] expectedB = new double[bStart + (n - 1) * ldb * 2 + k * 2 + 8];
        double[] actualB = expectedB.clone();
        ZBlasTestSupport.embedMatrix(compactB, n, k, 0, k, expectedB, bStart, ldb);
        ZBlasTestSupport.embedMatrix(compactB, n, k, 0, k, actualB, bStart, ldb);

        double[] expectedC = new double[cStart + (m - 1) * ldc * 2 + n * 2 + 8];
        double[] actualC = expectedC.clone();
        ZBlasTestSupport.embedMatrix(compactC, m, n, 0, n, expectedC, cStart, ldc);
        ZBlasTestSupport.embedMatrix(compactC, m, n, 0, n, actualC, cStart, ldc);

        ZBlasTestSupport.refZgemm(BLAS.Trans.NoTrans, BLAS.Trans.Conj, m, n, k, -1.0, 0.0,
            expectedA, aStart, lda, expectedB, bStart, ldb, 1.0, 0.0, expectedC, cStart, ldc);
        Zgemm.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.Conj, m, n, k,
            -1.0, 0.0, actualA, aStart / 2, lda, actualB, bStart / 2, ldb, 1.0, 0.0, actualC, cStart / 2, ldc);

        ZBlasTestSupport.assertArrayClose("Zgemm NT replay-like A unchanged", expectedA, actualA, TOL);
        ZBlasTestSupport.assertArrayClose("Zgemm NT replay-like B unchanged", expectedB, actualB, TOL);
        ZBlasTestSupport.assertArrayClose("Zgemm NT replay-like result", expectedC, actualC, TOL);
    }

}