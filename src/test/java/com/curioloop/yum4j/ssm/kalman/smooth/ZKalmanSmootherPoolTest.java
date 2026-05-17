package com.curioloop.yum4j.ssm.kalman.smooth;

import com.curioloop.yum4j.ssm.kalman.filter.ZKalmanFilterTest;

import com.curioloop.yum4j.ssm.kalman.filter.KalmanEngine;

import com.curioloop.yum4j.ssm.kalman.arena.ZSmootherScratchLayout;
import com.curioloop.yum4j.ssm.kalman.filter.ZFilterResult;
import com.curioloop.yum4j.ssm.kalman.filter.FilterOptions;
import com.curioloop.yum4j.ssm.kalman.init.InitialState;
import com.curioloop.yum4j.ssm.kalman.model.ModelFixture;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ZKalmanSmootherPoolTest {

    private static final double TOL = 1e-10;

        private static long expectedZScratchDoubleCount(int kEndog,
                                                        int kStates,
                                                        int kPosdef,
                                                        boolean needObservationFactor,
                                                        boolean needForecastFactor,
                                                        boolean needDisturbanceCovariance,
                                                        boolean needDiffuseScratch,
                                                        boolean needBorrowedSmoothingError) {
                return ZSmootherScratchLayout.create(
                                kEndog,
                                kStates,
                                kPosdef,
                                needObservationFactor,
                                needForecastFactor,
                                false,
                                needDisturbanceCovariance,
                                false,
                                needDiffuseScratch,
                                needBorrowedSmoothingError)
                                .totalLength();
        }

        private static long expectedZScratchDoubleCount(ModelFixture rep,
                                                        boolean needObservationFactor,
                                                        boolean needForecastFactor,
                                                        boolean needDisturbanceCovariance,
                                                        boolean needDiffuseScratch,
                                                        boolean needBorrowedSmoothingError) {
                return expectedZScratchDoubleCount(
                                rep.kEndog,
                                rep.kStates,
                                rep.kPosdef,
                                needObservationFactor,
                                needForecastFactor,
                                needDisturbanceCovariance,
                                needDiffuseScratch,
                                needBorrowedSmoothingError);
        }

        private static long expectedZBaseRuntimeScratchDoubleCount(ModelFixture rep) {
                return expectedZScratchDoubleCount(rep, false, false, false, false, false);
        }

        private static long expectedZDiffuseSmootherScratchDoubleCount(ModelFixture rep) {
                return expectedZScratchDoubleCount(rep, false, false, false, true, false)
                                - expectedZBaseRuntimeScratchDoubleCount(rep);
        }

        private static long expectedZSmootherResultDoubleCount(ModelFixture rep) {
                long kEndog = rep.kEndog;
                long kStates = rep.kStates;
                long kPosdef = rep.kPosdef;
                long nobs = rep.nobs;
                return 2L * kStates * nobs
                                + 2L * kStates * kStates * nobs
                                + 2L * kEndog * nobs
                                + 2L * kPosdef * nobs
                                + 2L * kEndog * kEndog * nobs
                                + 2L * kPosdef * kPosdef * nobs
                                + 2L * kStates * (nobs + 1)
                                + 2L * kStates * kStates * (nobs + 1)
                                + 2L * kEndog * nobs
                                + 2L * kStates * kStates * nobs;
        }

        private static long expectedZSmootherResultDoubleCount(ModelFixture rep, SmootherOptions spec) {
                long kEndog = rep.kEndog;
                long kStates = rep.kStates;
                long kPosdef = rep.kPosdef;
                long nobs = rep.nobs;
                long total = 0L;
                if (spec.includes(SmootherOptions.Surface.STATE)) {
                        total += 2L * kStates * nobs;
                }
                if (spec.includes(SmootherOptions.Surface.STATE_COVARIANCE)) {
                        total += 2L * kStates * kStates * nobs;
                }
                if (spec.includes(SmootherOptions.Surface.DISTURBANCE)) {
                        total += 2L * kEndog * nobs;
                        total += 2L * kPosdef * nobs;
                }
                if (spec.includes(SmootherOptions.Surface.DISTURBANCE_COVARIANCE)) {
                        total += 2L * kEndog * kEndog * nobs;
                        total += 2L * kPosdef * kPosdef * nobs;
                }
                if (spec.includes(SmootherOptions.Surface.STATE)
                                && spec.includes(SmootherOptions.Surface.STATE_COVARIANCE)
                                && spec.includes(SmootherOptions.Surface.DISTURBANCE)
                                && spec.includes(SmootherOptions.Surface.DISTURBANCE_COVARIANCE)) {
                        total += 2L * kStates * (nobs + 1);
                        total += 2L * kStates * kStates * (nobs + 1);
                        total += 2L * kEndog * nobs;
                        total += 2L * kStates * kStates * nobs;
                }
                return total;
        }

        private static long expectedZDiffuseSmootherResultDoubleCount(ModelFixture rep) {
                long kStates = rep.kStates;
                long nobs = rep.nobs;
                return 2L * kStates * (nobs + 1)
                                + 4L * kStates * kStates * (nobs + 1);
        }

    private static double[] activeSlice(double[] values, int base, int length) {
        return Arrays.copyOfRange(values, base, base + length);
    }

    private static double[] nullableActiveSlice(double[] values, int base, int length) {
        return values == null ? null : Arrays.copyOfRange(values, base, base + length);
    }

    private static void assertNullableArrayEquals(double[] expected, double[] actual) {
        assertEquals(expected == null, actual == null);
        if (expected != null) {
            assertArrayEquals(expected, actual, TOL);
        }
    }

    private static ModelFixture buildComplexBivariate(double[] y) {
        int nobs = y.length / 2;
        ModelFixture rep = ModelFixture.complex(2, 2, 1, nobs);
        double[] Z = {1.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0};
        double[] d = {0.0, 0.0, 0.0, 0.0};
        double[] H = {1.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0};
        double[] T = {0.9, 0.0, 0.1, 0.0,
                0.0, 0.0, 0.9, 0.0};
        double[] c = {0.0, 0.0, 0.0, 0.0};
        double[] R = {1.0, 0.0, 0.0, 0.0};
        double[] Q = {1.0, 0.0};
        for (int t = 0; t < nobs; t++) {
                rep.setDesign(Z, t);
                rep.setObsIntercept(d, t);
                rep.setObsCov(H, t);
                rep.setTransition(T, t);
                rep.setStateIntercept(c, t);
                rep.setSelection(R, t);
                rep.setStateCov(Q, t);
                rep.setEndog(new double[]{y[t * 2], 0.0, y[t * 2 + 1], 0.0}, t);
        }
        return rep;
    }

    private static void assertOutputsMatch(ZSmootherResult expected, ZSmootherResult actual) {
        assertArrayEquals(activeSlice(expected.smoothedState, expected.smoothedStateBase(), expected.smoothedStateLength()),
                activeSlice(actual.smoothedState, actual.smoothedStateBase(), actual.smoothedStateLength()), TOL);
        assertArrayEquals(activeSlice(expected.smoothedStateCov, expected.smoothedStateCovBase(), expected.smoothedStateCovLength()),
                activeSlice(actual.smoothedStateCov, actual.smoothedStateCovBase(), actual.smoothedStateCovLength()), TOL);
        assertArrayEquals(activeSlice(expected.smoothedObsDisturbance, expected.smoothedObsDisturbanceBase(), expected.smoothedObsDisturbanceLength()),
                activeSlice(actual.smoothedObsDisturbance, actual.smoothedObsDisturbanceBase(), actual.smoothedObsDisturbanceLength()), TOL);
        assertArrayEquals(activeSlice(expected.smoothedStateDisturbance, expected.smoothedStateDisturbanceBase(), expected.smoothedStateDisturbanceLength()),
                activeSlice(actual.smoothedStateDisturbance, actual.smoothedStateDisturbanceBase(), actual.smoothedStateDisturbanceLength()), TOL);
        assertArrayEquals(activeSlice(expected.smoothedObsDisturbanceCov, expected.smoothedObsDisturbanceCovBase(), expected.smoothedObsDisturbanceCovLength()),
                activeSlice(actual.smoothedObsDisturbanceCov, actual.smoothedObsDisturbanceCovBase(), actual.smoothedObsDisturbanceCovLength()), TOL);
        assertArrayEquals(activeSlice(expected.smoothedStateDisturbanceCov, expected.smoothedStateDisturbanceCovBase(), expected.smoothedStateDisturbanceCovLength()),
                activeSlice(actual.smoothedStateDisturbanceCov, actual.smoothedStateDisturbanceCovBase(), actual.smoothedStateDisturbanceCovLength()), TOL);
        assertArrayEquals(activeSlice(expected.scaledSmoothedEstimator, expected.scaledSmoothedEstimatorBase(), expected.scaledSmoothedEstimatorLength()),
                activeSlice(actual.scaledSmoothedEstimator, actual.scaledSmoothedEstimatorBase(), actual.scaledSmoothedEstimatorLength()), TOL);
        assertArrayEquals(activeSlice(expected.scaledSmoothedEstimatorCov, expected.scaledSmoothedEstimatorCovBase(), expected.scaledSmoothedEstimatorCovLength()),
                activeSlice(actual.scaledSmoothedEstimatorCov, actual.scaledSmoothedEstimatorCovBase(), actual.scaledSmoothedEstimatorCovLength()), TOL);
        assertArrayEquals(activeSlice(expected.smoothingError, expected.smoothingErrorBase(), expected.smoothingErrorLength()),
                activeSlice(actual.smoothingError, actual.smoothingErrorBase(), actual.smoothingErrorLength()), TOL);
        assertArrayEquals(activeSlice(expected.innovationsTransition, expected.innovationsTransitionBase(), expected.innovationsTransitionLength()),
                activeSlice(actual.innovationsTransition, actual.innovationsTransitionBase(), actual.innovationsTransitionLength()), TOL);
        assertNullableArrayEquals(
                nullableActiveSlice(expected.scaledSmoothedDiffuseEstimator, expected.scaledSmoothedDiffuseEstimatorBase(), expected.scaledSmoothedDiffuseEstimatorLength()),
                nullableActiveSlice(actual.scaledSmoothedDiffuseEstimator, actual.scaledSmoothedDiffuseEstimatorBase(), actual.scaledSmoothedDiffuseEstimatorLength()));
        assertNullableArrayEquals(
                nullableActiveSlice(expected.scaledSmoothedDiffuse1EstimatorCov, expected.scaledSmoothedDiffuse1EstimatorCovBase(), expected.scaledSmoothedDiffuse1EstimatorCovLength()),
                nullableActiveSlice(actual.scaledSmoothedDiffuse1EstimatorCov, actual.scaledSmoothedDiffuse1EstimatorCovBase(), actual.scaledSmoothedDiffuse1EstimatorCovLength()));
        assertNullableArrayEquals(
                nullableActiveSlice(expected.scaledSmoothedDiffuse2EstimatorCov, expected.scaledSmoothedDiffuse2EstimatorCovBase(), expected.scaledSmoothedDiffuse2EstimatorCovLength()),
                nullableActiveSlice(actual.scaledSmoothedDiffuse2EstimatorCov, actual.scaledSmoothedDiffuse2EstimatorCovBase(), actual.scaledSmoothedDiffuse2EstimatorCovLength()));
    }

        private static ZFilterResult filterForSmoothing(ModelFixture rep) {
                return KalmanEngine.filterComplex(rep, InitialState.approximateDiffuse(),
                                SmootherOptions.classical().requiredFilterOptions());
        }

        private static ZFilterResult filterForSmoothing(ModelFixture rep, FilterOptions spec) {
                return KalmanEngine.filterComplex(rep, InitialState.approximateDiffuse(), spec);
        }

        private static ZFilterResult diffuseFilterForSmoothing(ModelFixture rep, FilterOptions spec) {
                return KalmanEngine.filterComplex(rep, InitialState.diffuse(), spec);
        }

    @Test
    void testPoolReuseReturnsBorrowedResultByDefault() {
        double[] y = {1.0, 0.5, -0.3, 0.8, 0.2};
        ModelFixture rep = ZKalmanSmootherTest.buildZAR1(0.5, 0.0, 1.0, y);
        ZFilterResult fr = KalmanEngine.filterComplex(rep, InitialState.approximateDiffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        ZKalmanSmoother.Pool pool = new ZKalmanSmoother.Pool();
        ZSmootherResult expected = SmootherEngine.smoothComplex(rep, fr, com.curioloop.yum4j.ssm.kalman.smooth.SmootherOptions.conventional());

        ZSmootherResult r1 = SmootherEngine.smoothComplexBorrowed(rep, fr, pool, com.curioloop.yum4j.ssm.kalman.smooth.SmootherOptions.conventional());
        ZSmootherResult r2 = SmootherEngine.smoothComplexBorrowed(rep, fr, pool, com.curioloop.yum4j.ssm.kalman.smooth.SmootherOptions.conventional());

        assertSame(r1, r2);
        assertOutputsMatch(expected, r1);
    }

    @Test
    void testClonePreservesBorrowedSnapshot() {
        ModelFixture rep1 = ZKalmanSmootherTest.buildZAR1(0.5, 0.0, 1.0, new double[]{1.0, 0.5, -0.3, 0.8, 0.2});
        ModelFixture rep2 = ZKalmanSmootherTest.buildZAR1(0.5, 0.0, 1.0, new double[]{-0.2, 0.1, 0.4, -0.6, 0.9});
        ZFilterResult fr1 = KalmanEngine.filterComplex(rep1, InitialState.approximateDiffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        ZFilterResult fr2 = KalmanEngine.filterComplex(rep2, InitialState.approximateDiffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        ZKalmanSmoother.Pool pool = new ZKalmanSmoother.Pool();
        ZSmootherResult expected = SmootherEngine.smoothComplex(rep1, fr1, com.curioloop.yum4j.ssm.kalman.smooth.SmootherOptions.conventional());

        ZSmootherResult borrowed = SmootherEngine.smoothComplexBorrowed(rep1, fr1, pool, com.curioloop.yum4j.ssm.kalman.smooth.SmootherOptions.conventional());
        ZSmootherResult snapshot = borrowed.clone();

        SmootherEngine.smoothComplexBorrowed(rep2, fr2, pool, com.curioloop.yum4j.ssm.kalman.smooth.SmootherOptions.conventional());

        assertNotSame(snapshot, borrowed);
        assertOutputsMatch(expected, snapshot);
    }

    @Test
    void testStateOnlyBorrowedResultUsesLogicalSuppression() {
        double[] y = {1.0, 0.5, -0.3, 0.8, 0.2};
        ModelFixture rep = ZKalmanSmootherTest.buildZAR1(0.5, 0.0, 1.0, y);
        ZFilterResult fr = KalmanEngine.filterComplex(rep, InitialState.approximateDiffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        ZKalmanSmoother.Pool pool = new ZKalmanSmoother.Pool();
        SmootherOptions stateOnly = SmootherOptions.conventional().only(SmootherOptions.Surface.STATE);
        long fullResultDoubles = expectedZSmootherResultDoubleCount(rep, SmootherOptions.conventional());
        long stateOnlyResultDoubles = expectedZSmootherResultDoubleCount(rep, stateOnly);
        ZSmootherResult expected = SmootherEngine.smoothComplex(rep, fr, stateOnly);

        ZSmootherResult full = SmootherEngine.smoothComplexBorrowed(rep, fr, pool, SmootherOptions.conventional());
        assertEquals(fullResultDoubles, pool.retainedResultDoubleCount());
        ZSmootherResult stateOnlyResult = SmootherEngine.smoothComplexBorrowed(rep, fr, pool, stateOnly);

        assertSame(full, stateOnlyResult);
        assertTrue(stateOnlyResultDoubles < fullResultDoubles);
        assertEquals(fullResultDoubles, pool.retainedResultDoubleCount());
        assertArrayEquals(activeSlice(expected.smoothedState, expected.smoothedStateBase(), expected.smoothedStateLength()),
                activeSlice(stateOnlyResult.smoothedState, stateOnlyResult.smoothedStateBase(), stateOnlyResult.smoothedStateLength()), TOL);
        assertEquals(0, stateOnlyResult.smoothedStateCovLength());
        assertEquals(0, stateOnlyResult.smoothedObsDisturbanceLength());
        assertEquals(0, stateOnlyResult.smoothedStateDisturbanceLength());
        assertEquals(0, stateOnlyResult.smoothedObsDisturbanceCovLength());
        assertEquals(0, stateOnlyResult.smoothedStateDisturbanceCovLength());
        assertEquals(0, stateOnlyResult.scaledSmoothedEstimatorLength());
        assertEquals(0, stateOnlyResult.scaledSmoothedEstimatorCovLength());
        assertEquals(0, stateOnlyResult.smoothingErrorLength());
        assertEquals(0, stateOnlyResult.innovationsTransitionLength());
    }

    @Test
    void testDiffuseBorrowedResultMatchesStandalone() {
        double[] y = {10.2394, 1.0, 1.0, 1.0, 1.0, 1.0};
        ModelFixture rep = ZKalmanFilterTest.buildZExactDiffuseLocalLevel(y, 1.993, 8.253);
        ZFilterResult fr = KalmanEngine.filterComplex(rep, InitialState.diffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        ZKalmanSmoother.Pool pool = new ZKalmanSmoother.Pool();
        ZSmootherResult expected = SmootherEngine.smoothComplex(rep, fr, com.curioloop.yum4j.ssm.kalman.smooth.SmootherOptions.conventional());
        ZSmootherResult borrowed = SmootherEngine.smoothComplexBorrowed(rep, fr, pool, com.curioloop.yum4j.ssm.kalman.smooth.SmootherOptions.conventional());

        assertOutputsMatch(expected, borrowed);
    }

    @Test
    void testClassicalAndAlternativeZSmoothingRequireFilteredHistoryOnNonDiffusePath() {
        double[] y = {1.0, 0.5, -0.3, 0.8, 0.2};
        ModelFixture rep = ZKalmanSmootherTest.buildZAR1(0.5, 0.0, 1.0, y);
        FilterOptions compactSpec = SmootherOptions.conventional().requiredFilterOptions();
        ZFilterResult expectedFilter = filterForSmoothing(rep);
        ZFilterResult compactFilter = filterForSmoothing(rep, compactSpec);

        ZSmootherResult expected = SmootherEngine.smoothComplex(rep, expectedFilter, SmootherOptions.conventional());
        ZSmootherResult conventional = assertDoesNotThrow(
                () -> SmootherEngine.smoothComplex(rep, compactFilter, SmootherOptions.conventional()));
        assertThrows(IllegalArgumentException.class,
                () -> SmootherEngine.smoothComplex(rep, compactFilter, SmootherOptions.classical()));
        assertThrows(IllegalArgumentException.class,
                () -> SmootherEngine.smoothComplex(rep, compactFilter, SmootherOptions.alternative()));

        assertOutputsMatch(expected, conventional);
    }

    @Test
    void testZConventionalSmoothingStillRequiresKalmanGainAndForecastCovariance() {
        double[] y = {1.0, 0.5, -0.3, 0.8, 0.2};
        ModelFixture rep = ZKalmanSmootherTest.buildZAR1(0.5, 0.0, 1.0, y);
        FilterOptions noGain = FilterOptions.defaults().without(
                FilterOptions.Surface.LIKELIHOOD,
                FilterOptions.Surface.KALMAN_GAIN);
        FilterOptions noForecast = FilterOptions.defaults().without(
                FilterOptions.Surface.LIKELIHOOD,
                FilterOptions.Surface.FORECAST_MEAN, FilterOptions.Surface.FORECAST_ERROR, FilterOptions.Surface.FORECAST_COVARIANCE, FilterOptions.Surface.STANDARDIZED_FORECAST_ERROR, FilterOptions.Surface.FORECAST_ERROR_DIFFUSE_COVARIANCE);

        assertThrows(IllegalArgumentException.class,
                () -> SmootherEngine.smoothComplex(rep, filterForSmoothing(rep, noGain), SmootherOptions.conventional()));
        assertThrows(IllegalArgumentException.class,
                () -> SmootherEngine.smoothComplex(rep, filterForSmoothing(rep, noForecast), SmootherOptions.conventional()));
    }

    @Test
    void testDiffuseZSmoothingDoesNotRequireFilteredHistoryForAnyMethod() {
        double[] y = {10.2394, 1.0, 1.0, 1.0, 1.0, 1.0};
        ModelFixture rep = ZKalmanFilterTest.buildZExactDiffuseLocalLevel(y, 1.993, 8.253);
        FilterOptions compactSpec = SmootherOptions.conventional().requiredFilterOptions();
        ZFilterResult compactFilter = diffuseFilterForSmoothing(rep, compactSpec);

        assertDoesNotThrow(() -> SmootherEngine.smoothComplex(rep, compactFilter, SmootherOptions.conventional()));
        assertDoesNotThrow(() -> SmootherEngine.smoothComplex(rep, compactFilter, SmootherOptions.classical()));
        assertDoesNotThrow(() -> SmootherEngine.smoothComplex(rep, compactFilter, SmootherOptions.alternative()));
    }

        @Test
        void testPoolsCanShareScratchWithExactRetainedCounts() {
                double[] y = {10.2394, 0.0, 6.123123, 1.0, 1.0, 1.0};
                ModelFixture rep = ZKalmanFilterTest.buildZExactDiffuseLocalLinearTrend(y, 1.993, 8.253, 2.334);
                ZKalmanSmoother.Pool primary = new ZKalmanSmoother.Pool();
                ZKalmanSmoother.Pool shared = new ZKalmanSmoother.Pool();
                long expectedScratchDoubles = 0L;
                long expectedResultDoubles = expectedZSmootherResultDoubleCount(rep);

                primary.reserve(rep, SmootherOptions.conventional());
                shared.shareScratchFrom(primary);
                shared.reserve(rep, SmootherOptions.conventional());

                assertTrue(primary.sharesScratchWith(shared));
                assertNotSame(primary.retainedResult().smoothedState, shared.retainedResult().smoothedState);
                assertEquals(expectedScratchDoubles, primary.retainedScratchDoubleCount());
                assertEquals(expectedScratchDoubles, shared.retainedScratchDoubleCount());
                assertEquals(expectedResultDoubles, primary.retainedResultDoubleCount());
                assertEquals(expectedResultDoubles, shared.retainedResultDoubleCount());
                assertEquals((expectedResultDoubles) * Double.BYTES,
                        primary.retainedTotalByteCount());
                assertEquals((expectedResultDoubles) * Double.BYTES,
                        shared.retainedTotalByteCount());
        }

        @Test
        void testDiffusePoolQuantifiesAdditionalComplexSmootherRetention() {
                double[] y = {10.2394, 0.0, 6.123123, 1.0, 1.0, 1.0};
                ModelFixture rep = ZKalmanFilterTest.buildZExactDiffuseLocalLinearTrend(y, 1.993, 8.253, 2.334);
                ZFilterResult fr = KalmanEngine.filterComplex(rep, InitialState.diffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
                ZKalmanSmoother.Pool pool = new ZKalmanSmoother.Pool();
                long expectedScratchDoubles = expectedZBaseRuntimeScratchDoubleCount(rep);
                long expectedResultDoubles = expectedZSmootherResultDoubleCount(rep);
                long expectedDiffuseScratchDoubles = expectedZDiffuseSmootherScratchDoubleCount(rep);
                long expectedDiffuseResultDoubles = expectedZDiffuseSmootherResultDoubleCount(rep);

                pool.reserve(rep, SmootherOptions.conventional());

                ZSmootherResult result = SmootherEngine.smoothComplexBorrowed(rep, fr, pool, com.curioloop.yum4j.ssm.kalman.smooth.SmootherOptions.conventional());

                assertTrue(result.scaledSmoothedDiffuseEstimatorLength() > 0);
                assertEquals(expectedScratchDoubles + expectedDiffuseScratchDoubles,
                        pool.retainedScratchDoubleCount());
                assertEquals(expectedResultDoubles + expectedDiffuseResultDoubles,
                        pool.retainedResultDoubleCount());
                ZSmootherResult retained = pool.retainedResult();
                assertNotNull(retained);
                assertEquals(expectedDiffuseResultDoubles,
                        (long) retained.scaledSmoothedDiffuseEstimatorLength()
                                + retained.scaledSmoothedDiffuse1EstimatorCovLength()
                                + retained.scaledSmoothedDiffuse2EstimatorCovLength());
                assertEquals((expectedScratchDoubles + expectedDiffuseScratchDoubles
                        + expectedResultDoubles + expectedDiffuseResultDoubles) * Double.BYTES,
                        pool.retainedTotalByteCount());
        }

        @Test
        void testReleaseRetainedScratchClearsSharedComplexSmootherScratchOnly() {
                double[] y = {10.2394, 0.0, 6.123123, 1.0, 1.0, 1.0};
                ModelFixture rep = ZKalmanFilterTest.buildZExactDiffuseLocalLinearTrend(y, 1.993, 8.253, 2.334);
                ZFilterResult fr = KalmanEngine.filterComplex(rep, InitialState.diffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
                ZKalmanSmoother.Pool primary = new ZKalmanSmoother.Pool();
                ZKalmanSmoother.Pool shared = new ZKalmanSmoother.Pool();

                primary.reserve(rep, SmootherOptions.conventional());
                shared.shareScratchFrom(primary);
                shared.reserve(rep, SmootherOptions.conventional());

                SmootherEngine.smoothComplex(rep, fr, primary, SmootherOptions.conventional());

                assertTrue(primary.retainedScratchDoubleCount() > 0);
                assertTrue(shared.retainedScratchDoubleCount() > 0);
                assertNotNull(primary.retainedResult());
                assertNotNull(shared.retainedResult());

                primary.releaseRetainedScratch();
                shared.releaseRetainedScratch();

                assertEquals(0L, primary.retainedScratchDoubleCount());
                assertEquals(0L, shared.retainedScratchDoubleCount());
                assertTrue(!primary.sharesScratchWith(shared));
                assertNotNull(primary.retainedResult());
                assertNotNull(shared.retainedResult());
        }

        @Test
        void testReleaseRetainedScratchClearsSharedComplexSmootherAliasesOnPeers() {
                double[] y = {10.2394, 0.0, 6.123123, 1.0, 1.0, 1.0};
                ModelFixture rep = ZKalmanFilterTest.buildZExactDiffuseLocalLinearTrend(y, 1.993, 8.253, 2.334);
                ZFilterResult fr = KalmanEngine.filterComplex(rep, InitialState.diffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
                ZKalmanSmoother.Pool primary = new ZKalmanSmoother.Pool();
                ZKalmanSmoother.Pool shared = new ZKalmanSmoother.Pool();

                primary.reserve(rep, SmootherOptions.conventional());
                shared.shareScratchFrom(primary);

                SmootherEngine.smoothComplex(rep, fr, primary, SmootherOptions.conventional());

                assertTrue(shared.retainedScratchDoubleCount() > 0);

                primary.releaseRetainedScratch();

                assertEquals(0L, primary.retainedScratchDoubleCount());
                assertEquals(0L, shared.retainedScratchDoubleCount());
                assertTrue(!primary.sharesScratchWith(shared));
        }

        @Test
        void testReleaseRetainedResultsClearsComplexSmootherResultAliases() {
                double[] y = {10.2394, 0.0, 6.123123, 1.0, 1.0, 1.0};
                ModelFixture rep = ZKalmanFilterTest.buildZExactDiffuseLocalLinearTrend(y, 1.993, 8.253, 2.334);
                ZFilterResult fr = KalmanEngine.filterComplex(rep, InitialState.diffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
                ZKalmanSmoother.Pool pool = new ZKalmanSmoother.Pool();

                SmootherEngine.smoothComplexBorrowed(rep, fr, pool, SmootherOptions.conventional());

                assertNotNull(pool.retainedResult());

                pool.releaseRetainedResults();

                assertNull(pool.retainedResult());
        }

        @Test
        void testPartialMissingComplexObservationScratchUsesUnifiedWorstCaseLayout() {
                ModelFixture noMissing = buildComplexBivariate(new double[]{1.0, 0.5, -0.3, 0.8, 0.2, -0.1});
                ModelFixture missing = buildComplexBivariate(new double[]{1.0, 0.5, -0.3, 0.8, 0.2, -0.1});
                missing.setMissing(new boolean[]{false, true}, 0);
                missing.setMissing(new boolean[]{false, true}, 2);

                ZFilterResult noMissingFr = KalmanEngine.filterComplex(noMissing, InitialState.approximateDiffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
                ZFilterResult missingFr = KalmanEngine.filterComplex(missing, InitialState.approximateDiffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
                ZKalmanSmoother.Pool noMissingPool = new ZKalmanSmoother.Pool();
                ZKalmanSmoother.Pool missingPool = new ZKalmanSmoother.Pool();
                SmootherOptions stateOnly = SmootherOptions.conventional().only(SmootherOptions.Surface.STATE);
                long expectedScratch = expectedZScratchDoubleCount(missing, true, true, false, false, false);

                noMissingPool.reserve(noMissing, stateOnly);
                missingPool.reserve(missing, stateOnly);

                assertEquals(noMissingPool.retainedScratchDoubleCount(), missingPool.retainedScratchDoubleCount());

                SmootherEngine.smoothComplex(noMissing, noMissingFr, noMissingPool, stateOnly);
                SmootherEngine.smoothComplex(missing, missingFr, missingPool, stateOnly);

                assertEquals(expectedScratch, noMissingPool.retainedScratchDoubleCount());
                assertEquals(expectedScratch, missingPool.retainedScratchDoubleCount());
        }

        @Test
        void testOptionalComplexScratchAllocatesLazily() {
                double[] y = {1.0, 0.5, -0.3, 0.8, 0.2};
                ModelFixture rep = ZKalmanSmootherTest.buildZAR1(0.5, 0.0, 1.0, y);
                ZFilterResult fr = KalmanEngine.filterComplex(rep, InitialState.approximateDiffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
                ZKalmanSmoother.Pool pool = new ZKalmanSmoother.Pool();
                SmootherOptions stateOnly = SmootherOptions.conventional().only(SmootherOptions.Surface.STATE);
                long expectedStateOnlyScratch = expectedZScratchDoubleCount(rep, true, true, false, false, false);
                long expectedConventionalScratch = expectedZScratchDoubleCount(rep, true, true, true, false, false);

                pool.reserve(rep, stateOnly);
                long retainedBefore = pool.retainedScratchDoubleCount();

                SmootherEngine.smoothComplexBorrowed(rep, fr, pool, stateOnly);

                assertEquals(retainedBefore + expectedStateOnlyScratch,
                                pool.retainedScratchDoubleCount());

                SmootherEngine.smoothComplexBorrowed(rep, fr, pool, SmootherOptions.conventional());

                assertEquals(retainedBefore + expectedConventionalScratch,
                                pool.retainedScratchDoubleCount());
        }

                @Test
                void testReservePreservesRetainedComplexOptionalScratchUntilExplicitRelease() {
                        ModelFixture rep = buildComplexBivariate(new double[]{1.0, 0.5, -0.3, 0.8, 0.2, -0.1});
                        rep.setMissing(new boolean[]{false, true}, 0);
                        rep.setMissing(new boolean[]{false, true}, 2);
                        ZFilterResult fr = KalmanEngine.filterComplex(rep, InitialState.approximateDiffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
                        ZKalmanSmoother.Pool pool = new ZKalmanSmoother.Pool();
                        long expectedBaseScratch = 0L;

                        SmootherEngine.smoothComplexBorrowed(rep, fr, pool, SmootherOptions.conventional());

                        assertTrue(pool.retainedScratchDoubleCount() > expectedZBaseRuntimeScratchDoubleCount(rep));

                        pool.reserve(rep, SmootherOptions.conventional());

                        assertTrue(pool.retainedScratchDoubleCount() > expectedZBaseRuntimeScratchDoubleCount(rep));

                        pool.releaseRetainedScratch();

                        assertEquals(expectedBaseScratch, pool.retainedScratchDoubleCount());
                }
}