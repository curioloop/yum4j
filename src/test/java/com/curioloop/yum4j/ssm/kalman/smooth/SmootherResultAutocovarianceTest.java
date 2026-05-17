package com.curioloop.yum4j.ssm.kalman.smooth;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class SmootherResultAutocovarianceTest {

    private static final double TOL = 1e-12;

    @Test
    void embeddedLaggedAutocovarianceMatchesStatsmodelsBackwardWindowSemantics() {
        SmootherResult result = fixtureResult();

        double[] all = result.smoothedStateAutocovariance(2, 2);
        assertAllNaN(slice(all, 0, 4));
        assertAllNaN(slice(all, 4, 8));
        assertArrayEquals(expectedBlock(result, 2, 0, 2, 2), slice(all, 8, 12), TOL);
        assertArrayEquals(expectedBlock(result, 5, 0, 2, 2), slice(all, 20, 24), TOL);

        double[] at = result.smoothedStateAutocovarianceAt(2, 2, 4);
        assertArrayEquals(expectedBlock(result, 4, 0, 2, 2), at, TOL);

        double[] range = result.smoothedStateAutocovarianceRange(1, 2, 3, 5);
        assertArrayEquals(expectedBlock(result, 3, 0, 1, 2), slice(range, 0, 4), TOL);
        assertArrayEquals(expectedBlock(result, 4, 0, 1, 2), slice(range, 4, 8), TOL);
    }

    @Test
    void embeddedLaggedAutocovarianceMatchesStatsmodelsForwardWindowSemantics() {
        SmootherResult result = fixtureResult();

        double[] all = result.smoothedStateAutocovariance(-2, 2);
        assertArrayEquals(expectedBlock(result, 2, 2, 0, 2), slice(all, 0, 4), TOL);
        assertArrayEquals(expectedBlock(result, 5, 2, 0, 2), slice(all, 12, 16), TOL);
        assertAllNaN(slice(all, 16, 20));
        assertAllNaN(slice(all, 20, 24));

        double[] at = result.smoothedStateAutocovarianceAt(-1, 2, 4);
        assertArrayEquals(expectedBlock(result, 5, 1, 0, 2), at, TOL);

        double[] range = result.smoothedStateAutocovarianceRange(-1, 2, 1, 3);
        assertArrayEquals(expectedBlock(result, 2, 1, 0, 2), slice(range, 0, 4), TOL);
        assertArrayEquals(expectedBlock(result, 3, 1, 0, 2), slice(range, 4, 8), TOL);
    }

    @Test
    void embeddedLaggedAutocovarianceRejectsInvalidStatsmodelsStyleRequests() {
        SmootherResult result = fixtureResult();
        assertThrows(IllegalArgumentException.class,
            () -> result.smoothedStateAutocovariance(1, 2, 1, 1, null));
        assertThrows(IllegalArgumentException.class,
            () -> result.smoothedStateAutocovarianceAt(1, 2, -1));
        assertThrows(IllegalArgumentException.class,
            () -> result.smoothedStateAutocovarianceRange(1, 2, -1, 2));
        assertThrows(IllegalArgumentException.class,
            () -> result.smoothedStateAutocovarianceRange(1, 2, 5, 4));
        assertThrows(IllegalArgumentException.class,
            () -> result.smoothedStateAutocovariance(4, 2));

        SmootherResult stateOnly = new SmootherResult(1, 8, 1, 6,
            true, false, false, false, false);
        assertThrows(IllegalStateException.class,
            () -> stateOnly.smoothedStateAutocovariance(1, 2));
    }

    private static SmootherResult fixtureResult() {
        SmootherResult result = new SmootherResult(1, 8, 1, 6,
            false, true, false, false, false);
        for (int timeIndex = 0; timeIndex < result.nobs; timeIndex++) {
            int offset = result.smoothedStateCovOffset(timeIndex);
            for (int rowIndex = 0; rowIndex < result.kStates; rowIndex++) {
                for (int colIndex = 0; colIndex < result.kStates; colIndex++) {
                    result.smoothedStateCov[offset + rowIndex * result.kStates + colIndex] =
                        100.0 * timeIndex + 10.0 * rowIndex + colIndex;
                }
            }
        }
        return result;
    }

    private static double[] expectedBlock(SmootherResult result,
                                          int timeIndex,
                                          int rowBlock,
                                          int colBlock,
                                          int blockSize) {
        double[] values = new double[blockSize * blockSize];
        int outputOffset = 0;
        int covarianceOffset = result.smoothedStateCovOffset(timeIndex);
        int rowStart = rowBlock * blockSize;
        int colStart = colBlock * blockSize;
        for (int rowIndex = 0; rowIndex < blockSize; rowIndex++) {
            for (int colIndex = 0; colIndex < blockSize; colIndex++) {
                values[outputOffset++] = result.smoothedStateCov[
                    covarianceOffset + (rowStart + rowIndex) * result.kStates + colStart + colIndex];
            }
        }
        return values;
    }

    private static double[] slice(double[] values, int startInclusive, int endExclusive) {
        double[] slice = new double[endExclusive - startInclusive];
        System.arraycopy(values, startInclusive, slice, 0, slice.length);
        return slice;
    }

    private static void assertAllNaN(double[] values) {
        for (double value : values) {
            assertTrue(Double.isNaN(value));
        }
    }
}
