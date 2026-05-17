package com.curioloop.yum4j.ssm.kalman;

import com.curioloop.yum4j.ssm.kalman.init.InitialState;
import com.curioloop.yum4j.ssm.kalman.filter.FilterOptions;
import com.curioloop.yum4j.ssm.kalman.filter.FilterResult;
import com.curioloop.yum4j.ssm.kalman.filter.KalmanEngine;
import com.curioloop.yum4j.ssm.kalman.model.ModelFixture;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class InitialStateTest {

    private static final double TOL = 1e-10;

    private static double[] diagonal(double... values) {
        double[] matrix = new double[values.length * values.length];
        for (int i = 0; i < values.length; i++) {
            matrix[i * values.length + i] = values[i];
        }
        return matrix;
    }

    private static double[] filledDiagonal(int dimension, double value) {
        double[] matrix = new double[dimension * dimension];
        for (int i = 0; i < dimension; i++) {
            matrix[i * dimension + i] = value;
        }
        return matrix;
    }

    private static ModelFixture buildSimpleModel(int kStates) {
        ModelFixture rep = new ModelFixture(1, kStates, kStates, 1);
        rep.setDesign(new double[kStates], 0);
        rep.setObsIntercept(new double[]{0.0}, 0);
        rep.setObsCov(new double[]{1.0}, 0);
        rep.setTransition(new double[kStates * kStates], 0);
        rep.setStateIntercept(new double[kStates], 0);
        rep.setSelection(filledDiagonal(kStates, 1.0), 0);
        rep.setStateCov(filledDiagonal(kStates, 1.0), 0);
        rep.setEndog(new double[]{0.0}, 0);
        return rep;
    }

    private static ModelFixture buildStationaryBlockModel() {
        ModelFixture rep = new ModelFixture(1, 2, 2, 1);
        rep.setDesign(new double[]{0.0, 0.0}, 0);
        rep.setObsIntercept(new double[]{0.0}, 0);
        rep.setObsCov(new double[]{1.0}, 0);
        rep.setTransition(new double[]{0.5, 0.0, 0.0, 0.2}, 0);
        rep.setStateIntercept(new double[]{1.0, -1.0}, 0);
        rep.setSelection(new double[]{1.0, 0.0, 0.0, 1.0}, 0);
        rep.setStateCov(diagonal(1.0, 4.0), 0);
        rep.setEndog(new double[]{0.0}, 0);
        return rep;
    }

    private static ModelFixture buildNestedCompositeStationaryModel() {
        ModelFixture rep = new ModelFixture(1, 3, 3, 1);
        rep.setDesign(new double[]{0.0, 0.0, 0.0}, 0);
        rep.setObsIntercept(new double[]{0.0}, 0);
        rep.setObsCov(new double[]{1.0}, 0);
        rep.setTransition(new double[]{0.2, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.7}, 0);
        rep.setStateIntercept(new double[]{0.0, 1.0, 0.0}, 0);
        rep.setSelection(diagonal(1.0, 1.0, 1.0), 0);
        rep.setStateCov(diagonal(1.0, 4.0, 9.0), 0);
        rep.setEndog(new double[]{0.0}, 0);
        return rep;
    }

    private static ModelFixture buildMixedInitializationModel(int nobs) {
        ModelFixture rep = new ModelFixture(1, 3, 3, nobs);
        for (int timeIndex = 0; timeIndex < nobs; timeIndex++) {
            rep.setDesign(new double[]{1.0, 0.25, -0.1}, timeIndex);
            rep.setObsIntercept(new double[]{0.0}, timeIndex);
            rep.setObsCov(new double[]{0.8}, timeIndex);
            rep.setTransition(new double[]{0.5, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.7}, timeIndex);
            rep.setStateIntercept(new double[]{0.5, -1.0, 0.3}, timeIndex);
            rep.setSelection(diagonal(1.0, 1.0, 1.0), timeIndex);
            rep.setStateCov(diagonal(1.0, 4.0, 9.0), timeIndex);
            rep.setEndog(new double[]{1.0 + 0.15 * timeIndex}, timeIndex);
        }
        return rep;
    }

    @Test
    void testKnownResolveFillsMissingStateOrCovariance() {
        ModelFixture rep = buildSimpleModel(2);

        InitialState meanOnly = InitialState.known(new double[]{1.0, -2.0}, null).resolve(rep);
        assertArrayEquals(new double[]{1.0, -2.0}, meanOnly.initialState(), TOL);
        assertArrayEquals(new double[4], meanOnly.initialStateCov(), TOL);
        assertNull(meanOnly.initialDiffuseStateCov());

        InitialState covarianceOnly = InitialState.known(null, diagonal(3.0, 4.0)).resolve(rep);
        assertArrayEquals(new double[]{0.0, 0.0}, covarianceOnly.initialState(), TOL);
        assertArrayEquals(diagonal(3.0, 4.0), covarianceOnly.initialStateCov(), TOL);
        assertNull(covarianceOnly.initialDiffuseStateCov());
    }

    @Test
    void testApproximateDiffuseResolveUsesCustomVariance() {
        InitialState resolved = InitialState.approximateDiffuse(new double[]{1.5, -0.5}, 25.0)
            .resolve(buildSimpleModel(2));

        assertArrayEquals(new double[]{1.5, -0.5}, resolved.initialState(), TOL);
        assertArrayEquals(diagonal(25.0, 25.0), resolved.initialStateCov(), TOL);
        assertNull(resolved.initialDiffuseStateCov());
    }

    @Test
    void testCompositeResolveSupportsNestedFiniteAndDiffuseBlocks() {
        InitialState nested = new InitialState(2)
            .set(0, 1, InitialState.known(new double[]{3.0}, new double[]{4.0}))
            .set(1, 2, InitialState.approximateDiffuse(1, new double[]{-2.0}, 25.0));
        InitialState root = new InitialState(3)
            .set(0, 2, nested)
            .set(2, 3, InitialState.diffuse(1));

        InitialState resolved = root.resolve(buildSimpleModel(3));

        assertArrayEquals(new double[]{3.0, -2.0, 0.0}, resolved.initialState(), TOL);
        assertArrayEquals(diagonal(4.0, 25.0, 0.0), resolved.initialStateCov(), TOL);
        assertArrayEquals(diagonal(0.0, 0.0, 1.0), resolved.initialDiffuseStateCov(), TOL);
        assertTrue(resolved.hasDiffuseComponent());
    }

    @Test
    void testCompositeResolveSolvesStationarySubBlockIndependently() {
        InitialState root = new InitialState(2)
            .set(0, 1, InitialState.stationary(1))
            .set(1, 2, InitialState.known(new double[]{9.0}, new double[]{7.0}));

        InitialState resolved = root.resolve(buildStationaryBlockModel());

        assertArrayEquals(new double[]{2.0, 9.0}, resolved.initialState(), TOL);
        assertArrayEquals(diagonal(4.0 / 3.0, 7.0), resolved.initialStateCov(), TOL);
        assertNull(resolved.initialDiffuseStateCov());
    }

    @Test
    void testStatsmodelsMixedBasicInitializationMatrixResolves() {
        InitialState root = new InitialState(3)
            .set(0, 1, InitialState.known(new double[]{1.2}, null))
            .set(1, 2, InitialState.approximateDiffuse(1, new double[]{-0.2}, 1e10))
            .set(2, 3, InitialState.diffuse(1));

        InitialState resolved = root.resolve(buildMixedInitializationModel(1));

        assertArrayEquals(new double[]{1.2, -0.2, 0.0}, resolved.initialState(), TOL);
        assertArrayEquals(diagonal(0.0, 1e10, 0.0), resolved.initialStateCov(), TOL);
        assertArrayEquals(diagonal(0.0, 0.0, 1.0), resolved.initialDiffuseStateCov(), TOL);
        assertTrue(resolved.hasDiffuseComponent());

        InitialState knownBlock = new InitialState(3)
            .set(0, 2, InitialState.known(new double[]{1.2, -0.2}, diagonal(1.0, 4.2)))
            .set(2, 3, InitialState.approximateDiffuse(1));
        InitialState knownResolved = knownBlock.resolve(buildMixedInitializationModel(1));

        assertArrayEquals(new double[]{1.2, -0.2, 0.0}, knownResolved.initialState(), TOL);
        assertArrayEquals(diagonal(1.0, 4.2, 1e6), knownResolved.initialStateCov(), TOL);
        assertNull(knownResolved.initialDiffuseStateCov());
    }

    @Test
    void testStatsmodelsMixedStationaryInitializationMatrixResolves() {
        InitialState root = new InitialState(3)
            .set(0, 1, InitialState.known(new double[]{1.2}, new double[]{2.0}))
            .set(1, 3, InitialState.stationary(2));

        InitialState resolved = root.resolve(buildMixedInitializationModel(1));

        assertArrayEquals(new double[]{1.2, -1.25, 1.0}, resolved.initialState(), TOL);
        assertArrayEquals(diagonal(2.0, 4.0 / 0.96, 9.0 / 0.51), resolved.initialStateCov(), TOL);
        assertNull(resolved.initialDiffuseStateCov());

        InitialState individual = new InitialState(3)
            .set(0, 1, InitialState.known(new double[]{1.2}, new double[]{2.0}))
            .set(1, 2, InitialState.stationary(1))
            .set(2, 3, InitialState.stationary(1));
        InitialState individualResolved = individual.resolve(buildMixedInitializationModel(1));

        assertArrayEquals(resolved.initialState(), individualResolved.initialState(), TOL);
        assertArrayEquals(resolved.initialStateCov(), individualResolved.initialStateCov(), TOL);
    }

    @Test
    void testNearUnitStationaryInitializationRestoresLyapunovFallbackInput() {
        ModelFixture rep = new ModelFixture(1, 3, 1, 1);
        rep.setDesign(new double[]{1.0, 0.0, 0.0}, 0);
        rep.setObsIntercept(new double[]{0.0}, 0);
        rep.setObsCov(new double[]{0.0}, 0);
        rep.setTransition(new double[]{
            1.600047, -0.3624836, -0.2377553,
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0
        }, 0);
        rep.setStateIntercept(new double[]{0.0, 0.0, 0.0}, 0);
        rep.setSelection(new double[]{1.0, 0.0, 0.0}, 0);
        rep.setStateCov(new double[]{0.56482755281001}, 0);
        rep.setEndog(new double[]{0.0}, 0);

        InitialState resolved = InitialState.stationary(rep);

        assertArrayEquals(new double[]{0.0, 0.0, 0.0}, resolved.initialState(), TOL);
        assertArrayEquals(new double[]{
            9082.39470997301, 9081.606910961304, 9079.57858637325,
            9081.606910961304, 9082.394709973009, 9081.606910961302,
            9079.57858637325, 9081.606910961302, 9082.394709973009
        }, resolved.initialStateCov(), 1e-6);
    }

    @Test
    void testResolveIntoMatchesResolveForNestedCompositeDirectWrite() {
        InitialState nested = new InitialState(2)
            .set(0, 1, InitialState.known(new double[]{3.0}, new double[]{4.0}))
            .set(1, 2, InitialState.stationary(1));
        InitialState root = new InitialState(3)
            .set(0, 2, nested)
            .set(2, 3, InitialState.diffuse(1));

        ModelFixture rep = buildNestedCompositeStationaryModel();
        InitialState resolved = root.resolve(rep);

        assertArrayEquals(new double[]{3.0, 2.0, 0.0}, resolved.initialState(), TOL);
        assertArrayEquals(diagonal(4.0, 16.0 / 3.0, 0.0), resolved.initialStateCov(), TOL);
        assertArrayEquals(diagonal(0.0, 0.0, 1.0), resolved.initialDiffuseStateCov(), TOL);

        double sentinel = -123.0;
        double[] state = new double[8];
        double[] covariance = new double[16];
        double[] diffuse = new double[16];
        InitialState.StationaryWorkspace stationaryWorkspace = new InitialState.StationaryWorkspace();
        Arrays.fill(state, sentinel);
        Arrays.fill(covariance, sentinel);
        Arrays.fill(diffuse, sentinel);

        boolean hasDiffuse = root.resolveInto(rep, 0, stationaryWorkspace,
            state, 2,
            covariance, 3,
            diffuse, 4);

        assertTrue(hasDiffuse);
        assertEquals(InitialState.requiredStationaryBackingLength(rep)
                + InitialState.requiredStationaryPivotLength(rep) * ((long) Integer.BYTES) / Double.BYTES,
            stationaryWorkspace.retainedDoubleCount());
        assertEquals(sentinel, state[1], 0.0);
        assertEquals(sentinel, state[5], 0.0);
        assertArrayEquals(resolved.initialState(), Arrays.copyOfRange(state, 2, 5), TOL);
        assertArrayEquals(resolved.initialStateCov(), Arrays.copyOfRange(covariance, 3, 12), TOL);
        assertArrayEquals(resolved.initialDiffuseStateCov(), Arrays.copyOfRange(diffuse, 4, 13), TOL);
        assertEquals(sentinel, covariance[2], 0.0);
        assertEquals(sentinel, covariance[12], 0.0);
        assertEquals(sentinel, diffuse[3], 0.0);
        assertEquals(sentinel, diffuse[13], 0.0);
    }

    @Test
    void testCompositeRejectsOverlapsAndResolveRejectsGaps() {
        InitialState overlap = new InitialState(2)
            .set(0, 1, InitialState.diffuse(1));
        assertThrows(IllegalArgumentException.class,
            () -> overlap.set(0, 2, InitialState.diffuse(2)));

        InitialState gap = new InitialState(3)
            .set(0, 1, InitialState.diffuse(1))
            .set(2, 3, InitialState.diffuse(1));
        assertThrows(IllegalArgumentException.class,
            () -> gap.resolve(buildSimpleModel(3)));
    }

    @Test
    void testStatsmodelsInvalidInitializationMatrixRejectsBadRequests() {
        assertThrows(IllegalArgumentException.class, () -> InitialState.known(null, null));
        assertThrows(IllegalArgumentException.class,
            () -> InitialState.known(2, new double[]{1.0}, diagonal(1.0, 2.0)).resolve(buildSimpleModel(2)));
        assertThrows(IllegalArgumentException.class,
            () -> InitialState.known(2, new double[]{1.0, 2.0}, new double[]{1.0}).resolve(buildSimpleModel(2)));
        assertThrows(IllegalArgumentException.class, () -> InitialState.approximateDiffuse(2, -1.0));
        assertThrows(IllegalArgumentException.class, () -> new InitialState(-1));

        InitialState root = new InitialState(2).set(0, 1, InitialState.diffuse(1));
        assertThrows(IllegalArgumentException.class,
            () -> root.set(1, 2, InitialState.known(2, new double[]{1.0, 2.0}, diagonal(1.0, 1.0))));
        assertThrows(IllegalArgumentException.class, () -> root.unset(1, 2));
        assertThrows(IndexOutOfBoundsException.class, () -> root.resolve(buildSimpleModel(2), 1));
    }

    @Test
    void testMixedInitializationSmokeMatchesResolvedPayloadInFilter() {
        ModelFixture model = buildMixedInitializationModel(6);
        InitialState mixed = new InitialState(3)
            .set(0, 1, InitialState.known(new double[]{1.2}, new double[]{2.0}))
            .set(1, 2, InitialState.stationary(1))
            .set(2, 3, InitialState.approximateDiffuse(1, new double[]{0.4}, 25.0));
        InitialState resolved = mixed.resolve(model);
        FilterOptions options = FilterOptions.defaults();

        FilterResult direct = KalmanEngine.filter(model, resolved, options);
        FilterResult viaComposite = KalmanEngine.filter(model, mixed, options);

        assertEquals(direct.logLikelihood(), viaComposite.logLikelihood(), TOL);
        assertArrayEquals(direct.logLikelihoodObs, viaComposite.logLikelihoodObs, TOL);
        assertArrayEquals(resolved.initialState(), Arrays.copyOfRange(viaComposite.predictedState, 0, 3), TOL);
        assertArrayEquals(resolved.initialStateCov(), Arrays.copyOfRange(viaComposite.predictedStateCov, 0, 9), TOL);
    }
}