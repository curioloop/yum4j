/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.linalg.blas.cmplx;

import com.curioloop.yum4j.linalg.blas.BLAS;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;

class ZormlqTest {

    @Test
    void testLeftModesMatchExplicitReferenceApplication() {
        assertMatchesExplicitApplication('L', 'N', 5, 3, 4, 20260429L);
        assertMatchesExplicitApplication('L', 'C', 5, 3, 4, 20260430L);
    }

    @Test
    void testRightModesMatchExplicitReferenceApplication() {
        assertMatchesExplicitApplication('R', 'N', 3, 5, 4, 20260501L);
        assertMatchesExplicitApplication('R', 'C', 3, 5, 4, 20260502L);
    }

    @Test
    void testWorkspaceQueryProvidesExecutableSize() {
        int m = 3;
        int n = 5;
        int k = 4;
        int lda = n;
        int ldc = n;

        Random random = new Random(20260503L);
        double[] factorInput = randomComplexMatrix(random, k, lda);
        double[] factorized = factorInput.clone();
        double[] tau = new double[k * 2];
        double[] factorQuery = new double[2];
        assertEquals(0, Zgelq.zgelqf(k, n, factorized.clone(), 0, lda, tau.clone(), 0, factorQuery, 0, -1));
        int factorLwork = Math.max(1, (int) Math.ceil(factorQuery[0]));
        double[] factorWork = new double[factorLwork * 2];
        assertEquals(0, Zgelq.zgelqf(k, n, factorized, 0, lda, tau, 0, factorWork, 0, factorLwork));

        double[] c = randomComplexMatrix(random, m, ldc);
        double[] query = new double[2];
        assertEquals(0, Zgelq.zormlq('R', 'C', m, n, k, factorized.clone(), 0, lda, tau.clone(), 0, c.clone(), 0, ldc, query, 0, -1));
        int lwork = Math.max(1, (int) Math.ceil(query[0]));
        double[] work = new double[lwork * 2];
        assertEquals(0, Zgelq.zormlq('R', 'C', m, n, k, factorized.clone(), 0, lda, tau.clone(), 0, c.clone(), 0, ldc, work, 0, lwork));
    }

    @Test
    void testBlockedRightWorkspaceQueryProvidesExecutableSize() {
        assertBlockedWorkspaceQueryExecutes('R', 'C', 32, 96, 64, 20260504L);
    }

    @Test
    void testBlockedLeftWorkspaceQueryProvidesExecutableSize() {
        assertBlockedWorkspaceQueryExecutes('L', 'N', 96, 32, 64, 20260505L);
    }

    private static void assertMatchesExplicitApplication(char side, char trans, int m, int n, int k, long seed) {
        boolean left = side == 'L';
        int qCols = left ? m : n;
        int lda = qCols;
        int ldc = n;

        Random random = new Random(seed);
        double[] factorizedA = randomComplexMatrix(random, k, lda);
        double[] tau = new double[k * 2];
        double[] factorQuery = new double[2];
        assertEquals(0, Zgelq.zgelqf(k, qCols, factorizedA.clone(), 0, lda, tau.clone(), 0, factorQuery, 0, -1));
        int factorLwork = Math.max(1, (int) Math.ceil(factorQuery[0]));
        double[] factorWork = new double[factorLwork * 2];
        assertEquals(0, Zgelq.zgelqf(k, qCols, factorizedA, 0, lda, tau, 0, factorWork, 0, factorLwork));

        double[] baseC = randomComplexMatrix(random, m, ldc);
        double[] expected = baseC.clone();
        double[] actual = baseC.clone();
        applyReferenceUnml2(side, trans, m, n, k, factorizedA.clone(), lda, tau.clone(), expected, ldc);

        double[] query = new double[2];
        assertEquals(0, Zgelq.zormlq(side, trans, m, n, k, factorizedA.clone(), 0, lda, tau.clone(), 0, actual.clone(), 0, ldc, query, 0, -1));
        int lwork = Math.max(1, (int) Math.ceil(query[0]));
        double[] work = new double[lwork * 2];
        assertEquals(0, Zgelq.zormlq(side, trans, m, n, k, factorizedA.clone(), 0, lda, tau.clone(), 0, actual, 0, ldc, work, 0, lwork));
        ZBlasTestSupport.assertArrayClose("zormlq explicit " + side + trans, expected, actual, 1e-10);
    }

    private static void applyReferenceUnml2(char side, char trans, int m, int n, int k,
                                            double[] a, int lda, double[] tau,
                                            double[] c, int ldc) {
        boolean left = side == 'L';
        boolean notrans = trans == 'N';
        int start;
        int end;
        int step;

        if (left) {
            start = notrans ? 0 : k - 1;
            end = notrans ? k : -1;
            step = notrans ? 1 : -1;
        } else {
            start = notrans ? k - 1 : 0;
            end = notrans ? -1 : k;
            step = notrans ? -1 : 1;
        }

        double[] tauValue = new double[2];
        double[] work = new double[(left ? n : m) * 2];

        for (int i = start; i != end; i += step) {
            int aiiPos = (i * lda + i) * 2;
            double savedRe = a[aiiPos];
            double savedIm = a[aiiPos + 1];
            a[aiiPos] = 1.0;
            a[aiiPos + 1] = 0.0;

            tauValue[0] = tau[i * 2];
            tauValue[1] = notrans ? -tau[i * 2 + 1] : tau[i * 2 + 1];

            if (left) {
                Zlarf.zlarf(BLAS.Side.Left, m - i, n, a, aiiPos, 1, tauValue, 0, c, i * ldc * 2, ldc, work, 0);
            } else {
                Zlarf.zlarf(BLAS.Side.Right, m, n - i, a, aiiPos, 1, tauValue, 0, c, i * 2, ldc, work, 0);
            }

            a[aiiPos] = savedRe;
            a[aiiPos + 1] = savedIm;
        }
    }

    private static double[] randomComplexMatrix(Random random, int rows, int cols) {
        double[] matrix = new double[rows * cols * 2];
        for (int i = 0; i < matrix.length; i++) {
            matrix[i] = random.nextDouble() - 0.5;
        }
        return matrix;
    }

    private static void assertBlockedWorkspaceQueryExecutes(char side, char trans, int m, int n, int k, long seed) {
        boolean left = side == 'L';
        int qCols = left ? m : n;
        int lda = qCols;
        int ldc = n;

        Random random = new Random(seed);
        double[] factorizedA = randomComplexMatrix(random, k, lda);
        double[] tau = new double[k * 2];
        double[] factorQuery = new double[2];
        assertEquals(0, Zgelq.zgelqf(k, qCols, factorizedA.clone(), 0, lda, tau.clone(), 0, factorQuery, 0, -1));
        int factorLwork = Math.max(1, (int) Math.ceil(factorQuery[0]));
        double[] factorWork = new double[factorLwork * 2];
        assertEquals(0, Zgelq.zgelqf(k, qCols, factorizedA, 0, lda, tau, 0, factorWork, 0, factorLwork));

        double[] c = randomComplexMatrix(random, m, ldc);
        double[] query = new double[2];
        assertEquals(0, Zgelq.zormlq(side, trans, m, n, k, factorizedA.clone(), 0, lda, tau.clone(), 0, c.clone(), 0, ldc, query, 0, -1));
        int lwork = Math.max(1, (int) Math.ceil(query[0]));
        double[] work = new double[lwork * 2];
        assertEquals(0, Zgelq.zormlq(side, trans, m, n, k, factorizedA.clone(), 0, lda, tau.clone(), 0, c.clone(), 0, ldc, work, 0, lwork));
    }
}