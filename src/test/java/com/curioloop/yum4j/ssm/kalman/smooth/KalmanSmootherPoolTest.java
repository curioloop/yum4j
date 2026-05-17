package com.curioloop.yum4j.ssm.kalman.smooth;

import com.curioloop.yum4j.ssm.kalman.KalmanModelFixtures;

import com.curioloop.yum4j.ssm.kalman.filter.KalmanEngine;

import com.curioloop.yum4j.ssm.kalman.arena.SmootherScratchLayout;
import com.curioloop.yum4j.ssm.kalman.filter.FilterResult;
import com.curioloop.yum4j.ssm.kalman.filter.FilterOptions;
import com.curioloop.yum4j.ssm.kalman.init.InitialState;
import com.curioloop.yum4j.ssm.kalman.model.ModelFixture;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class KalmanSmootherPoolTest {

    private static final double TOL = 1e-6;

    private static long expectedRealScratchDoubleCount(int kEndog,
                                                       int kStates,
                                                       boolean needObservationFactor,
                                                       boolean needForecastFactor,
                                                       boolean needTransitionGain,
                                                       boolean needDisturbanceCovariance,
                                                       boolean needFactor,
                                                       boolean needDiffuse,
                                                       boolean needBorrowedSmoothingError) {
        return SmootherScratchLayout.create(
            kEndog,
            kStates,
            needObservationFactor,
            needForecastFactor,
            needTransitionGain,
            needDisturbanceCovariance,
            needFactor,
            needDiffuse,
            needBorrowedSmoothingError).totalLength();
    }

    private static long expectedRealBaseRuntimeScratchDoubleCount(int kEndog, int kStates) {
        return expectedRealScratchDoubleCount(kEndog, kStates,
            false, false, false, false, false, false, false);
    }

    private static long expectedRealStateOnlyScratchDoubleCount(int kEndog, int kStates) {
        return expectedRealScratchDoubleCount(kEndog, kStates,
            true, true, false, false, false, false, false);
    }

    private static long expectedRealFullScratchDoubleCount(int kEndog, int kStates) {
        return expectedRealScratchDoubleCount(kEndog, kStates,
            true, true, false, true, false, false, false);
    }

    private static long expectedRealBorrowedDisturbanceScratchDoubleCount(int kEndog, int kStates) {
        return expectedRealScratchDoubleCount(kEndog, kStates,
            true, true, false, false, false, false, true);
    }

    private static long expectedRealDiffuseScratchDoubleCount(int kEndog, int kStates) {
        return expectedRealScratchDoubleCount(kEndog, kStates,
            false, false, false, false, false, true, false);
    }

    private static ModelFixture buildAR1() {
        return buildAR1(new double[]{1.0, 0.5, -0.3, 0.8, 0.2});
    }

    private static ModelFixture buildAR1(double[] y) {
        return KalmanModelFixtures.scalarAr1(0.5, 0.7, 1.0, 1.0, y);
    }

    private static ModelFixture buildLocalLevel(double[] y) {
        return KalmanModelFixtures.localLevel(0.25, 0.5, y);
    }

    private static void assertNullableArrayEquals(double[] expected, double[] actual) {
        assertEquals(expected == null, actual == null);
        if (expected != null) {
            assertArrayEquals(expected, actual, TOL);
        }
    }

    private static double[] activeSlice(double[] values, int base, int length) {
        return Arrays.copyOfRange(values, base, base + length);
    }

    private static double[] nullableActiveSlice(double[] values, int base, int length) {
        return values == null ? null : Arrays.copyOfRange(values, base, base + length);
    }

    private static ModelFixture buildBivariate(double[] y) {
        return KalmanModelFixtures.bivariate2State(y);
    }

    private static FilterResult filterForSmoothing(ModelFixture rep) {
        return KalmanEngine.filter(rep, InitialState.approximateDiffuse(),
                SmootherOptions.classical().requiredFilterOptions());
    }

    private static FilterResult filterForSmoothing(ModelFixture rep, FilterOptions spec) {
        return KalmanEngine.filter(rep, InitialState.approximateDiffuse(), spec);
    }

    private static FilterResult diffuseFilterForSmoothing(ModelFixture rep, FilterOptions spec) {
        return KalmanEngine.filter(rep, InitialState.diffuse(), spec);
    }

    private static FilterResult univariateFilterForSmoothing(ModelFixture rep, FilterOptions spec) {
        return KalmanEngine.filterUnivariate(rep, InitialState.approximateDiffuse(), spec);
    }

    private static SmootherOptions stateOnly(SmootherOptions spec) {
        return spec.without(
                SmootherOptions.Surface.STATE_COVARIANCE,
                SmootherOptions.Surface.DISTURBANCE,
                SmootherOptions.Surface.DISTURBANCE_COVARIANCE);
    }

    private static void assertFullOutputsMatch(SmootherResult expected, SmootherResult actual) {
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
        assertNullableArrayEquals(nullableActiveSlice(expected.scaledSmoothedDiffuseEstimator,
                expected.scaledSmoothedDiffuseEstimatorBase(), expected.scaledSmoothedDiffuseEstimatorLength()),
            nullableActiveSlice(actual.scaledSmoothedDiffuseEstimator,
                actual.scaledSmoothedDiffuseEstimatorBase(), actual.scaledSmoothedDiffuseEstimatorLength()));
        assertNullableArrayEquals(nullableActiveSlice(expected.scaledSmoothedDiffuse1EstimatorCov,
                expected.scaledSmoothedDiffuse1EstimatorCovBase(), expected.scaledSmoothedDiffuse1EstimatorCovLength()),
            nullableActiveSlice(actual.scaledSmoothedDiffuse1EstimatorCov,
                actual.scaledSmoothedDiffuse1EstimatorCovBase(), actual.scaledSmoothedDiffuse1EstimatorCovLength()));
        assertNullableArrayEquals(nullableActiveSlice(expected.scaledSmoothedDiffuse2EstimatorCov,
                expected.scaledSmoothedDiffuse2EstimatorCovBase(), expected.scaledSmoothedDiffuse2EstimatorCovLength()),
            nullableActiveSlice(actual.scaledSmoothedDiffuse2EstimatorCov,
                actual.scaledSmoothedDiffuse2EstimatorCovBase(), actual.scaledSmoothedDiffuse2EstimatorCovLength()));
    }

    private static void assertBorrowedMethodMatchesExpected(SmootherOptions methodSpec) {
        ModelFixture rep = buildAR1(new double[]{1.0, Double.NaN, -0.3, 0.8, 0.2});
        FilterResult fr = filterForSmoothing(rep);
        KalmanSmoother.Pool pool = new KalmanSmoother.Pool();
        SmootherOptions stateOnlySpec = stateOnly(methodSpec);

        pool.reserve(rep, methodSpec);

        SmootherResult expectedFull = SmootherEngine.smooth(rep, fr, methodSpec);
        SmootherResult pooledFull = SmootherEngine.smoothBorrowed(rep, fr, pool, methodSpec);

        assertFullOutputsMatch(expectedFull, pooledFull);

        SmootherResult expectedStateOnly = SmootherEngine.smooth(rep, fr, stateOnlySpec);
        SmootherResult pooledStateOnly = SmootherEngine.smoothBorrowed(rep, fr, pool, stateOnlySpec);

        assertSame(pooledFull, pooledStateOnly);
        assertEquals(rep.nobs, pooledStateOnly.smoothedStateLength());
        assertEquals(0, pooledStateOnly.smoothedStateCovLength());
        assertEquals(0, pooledStateOnly.smoothedObsDisturbanceLength());
        assertEquals(0, pooledStateOnly.smoothedStateDisturbanceLength());
        assertEquals(0, pooledStateOnly.smoothedObsDisturbanceCovLength());
        assertEquals(0, pooledStateOnly.smoothedStateDisturbanceCovLength());
        assertEquals(0, pooledStateOnly.scaledSmoothedEstimatorLength());
        assertEquals(0, pooledStateOnly.scaledSmoothedEstimatorCovLength());
        assertEquals(0, pooledStateOnly.smoothingErrorLength());
        assertEquals(0, pooledStateOnly.innovationsTransitionLength());
        assertArrayEquals(activeSlice(expectedStateOnly.smoothedState, expectedStateOnly.smoothedStateBase(), expectedStateOnly.smoothedStateLength()),
            activeSlice(pooledStateOnly.smoothedState, pooledStateOnly.smoothedStateBase(), pooledStateOnly.smoothedStateLength()), TOL);
    }

    @Test
    void testPoolReuseReturnsBorrowedResultByDefault() {
        ModelFixture rep = buildAR1();
        FilterResult fr = filterForSmoothing(rep);
        KalmanSmoother.Pool pool = new KalmanSmoother.Pool();

        SmootherResult r1 = SmootherEngine.smoothBorrowed(rep, fr, pool, com.curioloop.yum4j.ssm.kalman.smooth.SmootherOptions.conventional());
        SmootherResult r2 = SmootherEngine.smoothBorrowed(rep, fr, pool, com.curioloop.yum4j.ssm.kalman.smooth.SmootherOptions.conventional());

        assertSame(r1, r2);
    }

    @Test
    void testClonePreservesBorrowedResultSnapshot() {
        ModelFixture rep1 = buildAR1(new double[]{1.0, 0.5, -0.3, 0.8, 0.2});
        ModelFixture rep2 = buildAR1(new double[]{-2.0, -1.0, 1.5, 0.4, -0.2});
        FilterResult fr1 = filterForSmoothing(rep1);
        FilterResult fr2 = filterForSmoothing(rep2);
        KalmanSmoother.Pool pool = new KalmanSmoother.Pool();
        SmootherResult expected = SmootherEngine.smooth(rep1, fr1, com.curioloop.yum4j.ssm.kalman.smooth.SmootherOptions.conventional());

        SmootherResult borrowed = SmootherEngine.smoothBorrowed(rep1, fr1, pool, com.curioloop.yum4j.ssm.kalman.smooth.SmootherOptions.conventional());
        SmootherResult snapshot = borrowed.clone();

        SmootherEngine.smoothBorrowed(rep2, fr2, pool, com.curioloop.yum4j.ssm.kalman.smooth.SmootherOptions.conventional());

        assertNotSame(snapshot, borrowed);
        for (int t = 0; t < rep1.nobs; t++) {
            assertEquals(expected.smoothedState(0, t), snapshot.smoothedState(0, t), TOL);
        }
    }

    @Test
    void testPoolReuseInvalidatesPreviousResult() {
        ModelFixture rep1 = buildAR1(new double[]{1.0, 0.5, -0.3, 0.8, 0.2});
        ModelFixture rep2 = buildAR1(new double[]{-2.0, -1.0, 1.5, 0.4, -0.2});
        FilterResult fr1 = filterForSmoothing(rep1);
        FilterResult fr2 = filterForSmoothing(rep2);
        KalmanSmoother.Pool pool = new KalmanSmoother.Pool();

        pool.reserve(rep1, SmootherOptions.conventional());

        SmootherResult r1 = SmootherEngine.smoothBorrowed(rep1, fr1, pool, SmootherOptions.conventional());
        double firstSmoothedState = r1.smoothedState(0, 0);

        SmootherResult r2 = SmootherEngine.smoothBorrowed(rep2, fr2, pool, SmootherOptions.conventional());

        assertSame(r1, r2);
        assertNotEquals(firstSmoothedState, r1.smoothedState(0, 0), TOL);
        assertEquals(r2.smoothedState(0, 0), r1.smoothedState(0, 0), TOL);
    }

    @Test
    void testPoolStateOnlyMask() {
        ModelFixture rep = buildAR1(new double[]{1.0, Double.NaN, -0.3, 0.8, 0.2});
        FilterResult fr = filterForSmoothing(rep);
        KalmanSmoother.Pool pool = new KalmanSmoother.Pool();
        SmootherOptions pooledStateOnly = stateOnly(SmootherOptions.conventional());

        pool.reserve(rep, pooledStateOnly);

        SmootherResult expected = SmootherEngine.smooth(rep, fr,
            stateOnly(SmootherOptions.conventional()));
        SmootherResult result = SmootherEngine.smoothBorrowed(rep, fr, pool, pooledStateOnly);

        assertEquals(rep.nobs, result.smoothedStateLength());
        assertEquals(0, result.smoothedStateCovLength());
        assertEquals(0, result.smoothedObsDisturbanceLength());
        assertEquals(0, result.smoothedStateDisturbanceLength());
        assertEquals(0, result.smoothedObsDisturbanceCovLength());
        assertEquals(0, result.smoothedStateDisturbanceCovLength());
        assertEquals(0, result.scaledSmoothedEstimatorLength());
        assertEquals(0, result.scaledSmoothedEstimatorCovLength());
        assertEquals(0, result.smoothingErrorLength());
        assertEquals(0, result.innovationsTransitionLength());
        for (int t = 0; t < rep.nobs; t++) {
            assertEquals(expected.smoothedState(0, t), result.smoothedState(0, t), TOL);
        }
    }

    @Test
    void testPoolSupportsDiffuseSmoothing() {
        ModelFixture rep1 = buildLocalLevel(new double[]{1.0, 2.0, 3.0, 4.0});
        ModelFixture rep2 = buildLocalLevel(new double[]{-1.5, 0.25, 2.0, -0.75});
        FilterResult fr1 = KalmanEngine.filter(rep1, InitialState.diffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        FilterResult fr2 = KalmanEngine.filter(rep2, InitialState.diffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        KalmanSmoother.Pool pool = new KalmanSmoother.Pool();
        SmootherOptions fullSpec = SmootherOptions.conventional();
        SmootherOptions stateOnlySpec = stateOnly(fullSpec);

        assertTrue(fr1.nobsDiffuse > 0);
        assertTrue(fr2.nobsDiffuse > 0);

        pool.reserve(rep1, fullSpec);

        SmootherResult expectedFull1 = SmootherEngine.smooth(rep1, fr1, fullSpec);
        SmootherResult borrowedFull1 = SmootherEngine.smoothBorrowed(rep1, fr1, pool, fullSpec);

        assertFullOutputsMatch(expectedFull1, borrowedFull1);

        SmootherResult expectedFull2 = SmootherEngine.smooth(rep2, fr2, fullSpec);
        SmootherResult borrowedFull2 = SmootherEngine.smoothBorrowed(rep2, fr2, pool, fullSpec);

        assertSame(borrowedFull1, borrowedFull2);
        assertFullOutputsMatch(expectedFull2, borrowedFull2);

        SmootherResult expectedStateOnly = SmootherEngine.smooth(rep2, fr2, stateOnlySpec);
        SmootherResult borrowedStateOnly = SmootherEngine.smoothBorrowed(rep2, fr2, pool, stateOnlySpec);

        assertSame(borrowedFull2, borrowedStateOnly);
        assertArrayEquals(activeSlice(expectedStateOnly.smoothedState, expectedStateOnly.smoothedStateBase(), expectedStateOnly.smoothedStateLength()),
            activeSlice(borrowedStateOnly.smoothedState, borrowedStateOnly.smoothedStateBase(), borrowedStateOnly.smoothedStateLength()), TOL);
        assertEquals(0, borrowedStateOnly.smoothedStateCovLength());
        assertEquals(0, borrowedStateOnly.smoothedObsDisturbanceLength());
        assertEquals(0, borrowedStateOnly.smoothedStateDisturbanceLength());
        assertEquals(0, borrowedStateOnly.smoothedObsDisturbanceCovLength());
        assertEquals(0, borrowedStateOnly.smoothedStateDisturbanceCovLength());
        assertEquals(0, borrowedStateOnly.scaledSmoothedEstimatorLength());
        assertEquals(0, borrowedStateOnly.scaledSmoothedEstimatorCovLength());
        assertEquals(0, borrowedStateOnly.smoothingErrorLength());
        assertEquals(0, borrowedStateOnly.innovationsTransitionLength());
        assertEquals(0, borrowedStateOnly.scaledSmoothedDiffuseEstimatorLength());
        assertEquals(0, borrowedStateOnly.scaledSmoothedDiffuse1EstimatorCovLength());
        assertEquals(0, borrowedStateOnly.scaledSmoothedDiffuse2EstimatorCovLength());
    }

    @Test
    void testPoolSupportsClassicalMethod() {
        assertBorrowedMethodMatchesExpected(SmootherOptions.classical());
    }

    @Test
    void testPoolSupportsAlternativeMethod() {
        assertBorrowedMethodMatchesExpected(SmootherOptions.alternative());
    }

    @Test
    void testPoolSupportsUnivariateSmoothing() {
        ModelFixture rep = buildBivariate(new double[]{1.0, 0.5, Double.NaN, Double.NaN, 0.2, -0.1});
        FilterResult fr = KalmanEngine.filterUnivariate(rep, InitialState.approximateDiffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        KalmanSmoother.Pool pool = new KalmanSmoother.Pool();
        SmootherOptions stateOnlySpec = stateOnly(SmootherOptions.alternative());

        pool.reserve(rep, SmootherOptions.alternative());

        SmootherResult expectedFull = SmootherEngine.smooth(rep, fr, SmootherOptions.alternative());
        SmootherResult borrowedFull = SmootherEngine.smoothBorrowed(rep, fr, pool, SmootherOptions.alternative());

        assertFullOutputsMatch(expectedFull, borrowedFull);

        SmootherResult expectedStateOnly = SmootherEngine.smooth(rep, fr, stateOnlySpec);
        SmootherResult borrowedStateOnly = SmootherEngine.smoothBorrowed(rep, fr, pool, stateOnlySpec);

        assertSame(borrowedFull, borrowedStateOnly);
        assertArrayEquals(activeSlice(expectedStateOnly.smoothedState, expectedStateOnly.smoothedStateBase(), expectedStateOnly.smoothedStateLength()),
            activeSlice(borrowedStateOnly.smoothedState, borrowedStateOnly.smoothedStateBase(), borrowedStateOnly.smoothedStateLength()), TOL);
        assertEquals(0, borrowedStateOnly.smoothedStateCovLength());
        assertEquals(0, borrowedStateOnly.smoothedObsDisturbanceLength());
        assertEquals(0, borrowedStateOnly.smoothedStateDisturbanceLength());
        assertEquals(0, borrowedStateOnly.smoothedObsDisturbanceCovLength());
        assertEquals(0, borrowedStateOnly.smoothedStateDisturbanceCovLength());
        assertEquals(0, borrowedStateOnly.scaledSmoothedEstimatorLength());
        assertEquals(0, borrowedStateOnly.scaledSmoothedEstimatorCovLength());
        assertEquals(0, borrowedStateOnly.smoothingErrorLength());
        assertEquals(0, borrowedStateOnly.innovationsTransitionLength());
    }

        @Test
        void testConventionalSmoothingDoesNotRequireFilteredHistory() {
        ModelFixture rep = buildAR1();
        FilterOptions compactSpec = SmootherOptions.conventional().requiredFilterOptions();
        FilterResult expectedFilter = filterForSmoothing(rep);
        FilterResult compactFilter = filterForSmoothing(rep, compactSpec);

        SmootherResult expected = SmootherEngine.smooth(rep, expectedFilter, SmootherOptions.conventional());
        SmootherResult actual = assertDoesNotThrow(
            () -> SmootherEngine.smooth(rep, compactFilter, SmootherOptions.conventional()));

        assertFullOutputsMatch(expected, actual);
        }

        @Test
        void testConventionalSmoothingStillRequiresKalmanGainAndForecastCovariance() {
        ModelFixture rep = buildAR1();
        FilterOptions noGain = FilterOptions.defaults().without(
            FilterOptions.Surface.LIKELIHOOD,
            FilterOptions.Surface.KALMAN_GAIN);
        FilterOptions noForecast = FilterOptions.defaults().without(
            FilterOptions.Surface.LIKELIHOOD,
            FilterOptions.Surface.FORECAST_MEAN, FilterOptions.Surface.FORECAST_ERROR, FilterOptions.Surface.FORECAST_COVARIANCE, FilterOptions.Surface.STANDARDIZED_FORECAST_ERROR, FilterOptions.Surface.FORECAST_ERROR_DIFFUSE_COVARIANCE);

        assertThrows(IllegalArgumentException.class,
            () -> SmootherEngine.smooth(rep, filterForSmoothing(rep, noGain), SmootherOptions.conventional()));
        assertThrows(IllegalArgumentException.class,
            () -> SmootherEngine.smooth(rep, filterForSmoothing(rep, noForecast), SmootherOptions.conventional()));
        }

        @Test
        void testClassicalAndAlternativeSmoothingRequireFilteredHistoryOnNonDiffusePath() {
        ModelFixture rep = buildAR1();
        FilterOptions compactSpec = SmootherOptions.conventional().requiredFilterOptions();
        FilterResult compactFilter = filterForSmoothing(rep, compactSpec);

        assertThrows(IllegalArgumentException.class,
            () -> SmootherEngine.smooth(rep, compactFilter, SmootherOptions.classical()));
        assertThrows(IllegalArgumentException.class,
            () -> SmootherEngine.smooth(rep, compactFilter, SmootherOptions.alternative()));
        }

        @Test
        void testDiffuseSmoothingDoesNotRequireFilteredHistoryForAnyMethod() {
        ModelFixture rep = buildLocalLevel(new double[]{1.0, 2.0, 3.0, 4.0});
        FilterOptions compactSpec = SmootherOptions.conventional().requiredFilterOptions();
        FilterResult compactFilter = diffuseFilterForSmoothing(rep, compactSpec);

        SmootherResult conventional = assertDoesNotThrow(
            () -> SmootherEngine.smooth(rep, compactFilter, SmootherOptions.conventional()));
        SmootherResult classical = assertDoesNotThrow(
            () -> SmootherEngine.smooth(rep, compactFilter, SmootherOptions.classical()));
        SmootherResult alternative = assertDoesNotThrow(
            () -> SmootherEngine.smooth(rep, compactFilter, SmootherOptions.alternative()));

        assertFullOutputsMatch(conventional, classical);
        assertFullOutputsMatch(conventional, alternative);
        }

        @Test
        void testUnivariateSmoothingDoesNotRequireFilteredHistory() {
        ModelFixture rep = buildBivariate(new double[]{1.0, 0.5, Double.NaN, Double.NaN, 0.2, -0.1});
        FilterOptions compactSpec = SmootherOptions.conventional().requiredFilterOptions();
        FilterResult expectedFilter = KalmanEngine.filterUnivariate(rep, InitialState.approximateDiffuse(), com.curioloop.yum4j.ssm.kalman.filter.FilterOptions.defaults());
        FilterResult compactFilter = univariateFilterForSmoothing(rep, compactSpec);

        SmootherResult expected = SmootherEngine.smooth(rep, expectedFilter, SmootherOptions.alternative());
        SmootherResult actual = assertDoesNotThrow(
            () -> SmootherEngine.smooth(rep, compactFilter, SmootherOptions.alternative()));

        assertFullOutputsMatch(expected, actual);
        }

    @Test
    void testPoolReserveSupportsClassicalMethod() {
        ModelFixture rep = buildAR1();
        KalmanSmoother.Pool pool = new KalmanSmoother.Pool();

        pool.reserve(rep, SmootherOptions.classical());

        SmootherResult retained = pool.retainedResult();
        assertNotNull(retained);
        assertTrue(retained.smoothedState.length >= rep.nobs);
    }

    @Test
    void testPoolsCanShareScratchWhileKeepingSeparateResults() {
        ModelFixture rep = buildAR1();
        KalmanSmoother.Pool primary = new KalmanSmoother.Pool();
        KalmanSmoother.Pool shared = new KalmanSmoother.Pool();
        long expectedScratch = 0L;

        primary.reserve(rep, SmootherOptions.conventional());
        shared.shareScratchFrom(primary);
        shared.reserve(rep, SmootherOptions.conventional());

        assertTrue(primary.sharesScratchWith(shared));
        assertNotSame(primary.retainedResult().smoothedState, shared.retainedResult().smoothedState);
        assertEquals(expectedScratch, primary.retainedScratchDoubleCount());
        assertEquals(expectedScratch, shared.retainedScratchDoubleCount());
    }

    @Test
    void testReleaseRetainedScratchClearsSharedRealSmootherScratchOnly() {
        ModelFixture rep = buildAR1();
        FilterResult fr = filterForSmoothing(rep);
        KalmanSmoother.Pool primary = new KalmanSmoother.Pool();
        KalmanSmoother.Pool shared = new KalmanSmoother.Pool();

        primary.reserve(rep, SmootherOptions.conventional());
        shared.shareScratchFrom(primary);
        shared.reserve(rep, SmootherOptions.conventional());

        SmootherEngine.smoothBorrowed(rep, fr, primary, SmootherOptions.conventional());

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
    void testReleaseRetainedScratchClearsSharedRealSmootherAliasesOnPeers() {
        ModelFixture rep = buildAR1();
        FilterResult fr = filterForSmoothing(rep);
        KalmanSmoother.Pool primary = new KalmanSmoother.Pool();
        KalmanSmoother.Pool shared = new KalmanSmoother.Pool();

        primary.reserve(rep, SmootherOptions.conventional());
        shared.shareScratchFrom(primary);

        SmootherEngine.smoothBorrowed(rep, fr, primary, SmootherOptions.conventional());

        assertTrue(shared.retainedScratchDoubleCount() > 0);

        primary.releaseRetainedScratch();

        assertEquals(0L, primary.retainedScratchDoubleCount());
        assertEquals(0L, shared.retainedScratchDoubleCount());
        assertTrue(!primary.sharesScratchWith(shared));
    }

    @Test
    void testReleaseRetainedResultsClearsRealSmootherResultAliases() {
        ModelFixture rep = buildLocalLevel(new double[]{1.0, 2.0, 3.0, 4.0});
        FilterResult fr = diffuseFilterForSmoothing(rep, SmootherOptions.conventional().requiredFilterOptions());
        KalmanSmoother.Pool pool = new KalmanSmoother.Pool();

        SmootherEngine.smoothBorrowed(rep, fr, pool, SmootherOptions.conventional());

        SmootherResult retained = pool.retainedResult();
        assertNotNull(retained);
        assertTrue(retained.smoothedState.length > 0);
        assertNotNull(retained.scaledSmoothedDiffuseEstimator);

        pool.releaseRetainedResults();

        assertNull(pool.retainedResult());
    }

    @Test
    void testPartialMissingRealSelectionScratchReusesUnifiedObservationScratch() {
        ModelFixture noMissing = buildBivariate(new double[]{1.0, 0.5, -0.3, 0.8, 0.2, -0.1});
        ModelFixture missing = buildBivariate(new double[]{1.0, 0.5, -0.3, 0.8, 0.2, -0.1});
        missing.setMissing(new boolean[]{false, true}, 0);
        missing.setMissing(new boolean[]{false, true}, 2);

        FilterResult noMissingFr = filterForSmoothing(noMissing);
        FilterResult missingFr = filterForSmoothing(missing);
        KalmanSmoother.Pool noMissingPool = new KalmanSmoother.Pool();
        KalmanSmoother.Pool missingPool = new KalmanSmoother.Pool();
        SmootherOptions stateOnly = stateOnly(SmootherOptions.conventional());
        long expectedUnifiedScratch = expectedRealStateOnlyScratchDoubleCount(missing.kEndog, missing.kStates);

        noMissingPool.reserve(noMissing, stateOnly);
        missingPool.reserve(missing, stateOnly);

        assertEquals(noMissingPool.retainedScratchDoubleCount(), missingPool.retainedScratchDoubleCount());

        SmootherEngine.smoothBorrowed(noMissing, noMissingFr, noMissingPool, stateOnly);
        SmootherEngine.smoothBorrowed(missing, missingFr, missingPool, stateOnly);

        assertEquals(expectedUnifiedScratch, noMissingPool.retainedScratchDoubleCount());
        assertEquals(noMissingPool.retainedScratchDoubleCount(), missingPool.retainedScratchDoubleCount());
    }

    @Test
    void testOptionalRealScratchAllocatesLazily() {
        ModelFixture rep = buildAR1();
        FilterResult fr = filterForSmoothing(rep);
        KalmanSmoother.Pool pool = new KalmanSmoother.Pool();
        SmootherOptions stateOnly = stateOnly(SmootherOptions.conventional());
        long expectedStateOnlyScratch = expectedRealStateOnlyScratchDoubleCount(rep.kEndog, rep.kStates);
        long expectedFullScratch = expectedRealFullScratchDoubleCount(rep.kEndog, rep.kStates);

        pool.reserve(rep, stateOnly);
        long retainedBefore = pool.retainedScratchDoubleCount();

        SmootherEngine.smoothBorrowed(rep, fr, pool, stateOnly);

        assertEquals(retainedBefore + expectedStateOnlyScratch,
            pool.retainedScratchDoubleCount());

        SmootherEngine.smoothBorrowed(rep, fr, pool, SmootherOptions.conventional());

        assertEquals(retainedBefore + expectedFullScratch,
            pool.retainedScratchDoubleCount());
    }

    @Test
    void testReservePreservesRetainedRealOptionalScratchUntilExplicitRelease() {
        ModelFixture rep = buildBivariate(new double[]{1.0, 0.5, -0.3, 0.8, 0.2, -0.1});
        rep.setMissing(new boolean[]{false, true}, 0);
        rep.setMissing(new boolean[]{false, true}, 2);
        FilterResult fr = filterForSmoothing(rep);
        KalmanSmoother.Pool pool = new KalmanSmoother.Pool();
        long expectedBaseScratch = 0L;

        SmootherEngine.smoothBorrowed(rep, fr, pool, SmootherOptions.conventional());

        assertTrue(pool.retainedScratchDoubleCount() > expectedRealBaseRuntimeScratchDoubleCount(rep.kEndog, rep.kStates));

        pool.reserve(rep, SmootherOptions.conventional());

        assertTrue(pool.retainedScratchDoubleCount() > expectedRealBaseRuntimeScratchDoubleCount(rep.kEndog, rep.kStates));

        pool.releaseRetainedScratch();

        assertEquals(expectedBaseScratch, pool.retainedScratchDoubleCount());
    }

    @Test
    void testBorrowedRealDisturbanceOnlyScratchAllocatesSmoothingErrorLazily() {
        ModelFixture rep = buildAR1();
        FilterResult fr = filterForSmoothing(rep);
        KalmanSmoother.Pool pool = new KalmanSmoother.Pool();
        SmootherOptions disturbanceOnly = SmootherOptions.conventional().only(SmootherOptions.Surface.DISTURBANCE);
        long expectedBorrowedScratch = expectedRealBorrowedDisturbanceScratchDoubleCount(rep.kEndog, rep.kStates);

        pool.reserve(rep, disturbanceOnly);
        long retainedBefore = pool.retainedScratchDoubleCount();

        SmootherEngine.smoothBorrowed(rep, fr, pool, disturbanceOnly);

        assertEquals(retainedBefore + expectedBorrowedScratch,
            pool.retainedScratchDoubleCount());
    }

    @Test
    void testDiffuseRealScratchDoesNotAllocateUnusedGainOrPooledErrorScratch() {
        ModelFixture rep = buildLocalLevel(new double[]{1.0, 2.0, 3.0, 4.0});
        FilterResult fr = diffuseFilterForSmoothing(rep, SmootherOptions.conventional().requiredFilterOptions());
        KalmanSmoother.Pool pool = new KalmanSmoother.Pool();
        SmootherOptions stateOnly = stateOnly(SmootherOptions.conventional());
        long expectedDiffuseScratch = expectedRealDiffuseScratchDoubleCount(rep.kEndog, rep.kStates);

        assertTrue(fr.nobsDiffuse > 0);

        pool.reserve(rep, stateOnly);
        long retainedBefore = pool.retainedScratchDoubleCount();

        SmootherEngine.smoothBorrowed(rep, fr, pool, stateOnly);

        assertEquals(retainedBefore + expectedDiffuseScratch,
            pool.retainedScratchDoubleCount());
    }
}