package com.curioloop.yum4j.kalman;

import com.curioloop.yum4j.kalman.init.StateInitialization;
import com.curioloop.yum4j.kalman.model.ModelFixture;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class StateInitializationTest {

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

    @Test
    void testKnownResolveFillsMissingStateOrCovariance() {
        ModelFixture rep = buildSimpleModel(2);

        StateInitialization meanOnly = StateInitialization.known(new double[]{1.0, -2.0}, null).resolve(rep);
        assertArrayEquals(new double[]{1.0, -2.0}, meanOnly.initialState(), TOL);
        assertArrayEquals(new double[4], meanOnly.initialStateCov(), TOL);
        assertNull(meanOnly.initialDiffuseStateCov());

        StateInitialization covarianceOnly = StateInitialization.known(null, diagonal(3.0, 4.0)).resolve(rep);
        assertArrayEquals(new double[]{0.0, 0.0}, covarianceOnly.initialState(), TOL);
        assertArrayEquals(diagonal(3.0, 4.0), covarianceOnly.initialStateCov(), TOL);
        assertNull(covarianceOnly.initialDiffuseStateCov());
    }

    @Test
    void testApproximateDiffuseResolveUsesCustomVariance() {
        StateInitialization resolved = StateInitialization.approximateDiffuse(new double[]{1.5, -0.5}, 25.0)
            .resolve(buildSimpleModel(2));

        assertArrayEquals(new double[]{1.5, -0.5}, resolved.initialState(), TOL);
        assertArrayEquals(diagonal(25.0, 25.0), resolved.initialStateCov(), TOL);
        assertNull(resolved.initialDiffuseStateCov());
    }

    @Test
    void testCompositeResolveSupportsNestedFiniteAndDiffuseBlocks() {
        StateInitialization nested = new StateInitialization(2)
            .set(0, 1, StateInitialization.known(new double[]{3.0}, new double[]{4.0}))
            .set(1, 2, StateInitialization.approximateDiffuse(1, new double[]{-2.0}, 25.0));
        StateInitialization root = new StateInitialization(3)
            .set(0, 2, nested)
            .set(2, 3, StateInitialization.diffuse(1));

        StateInitialization resolved = root.resolve(buildSimpleModel(3));

        assertArrayEquals(new double[]{3.0, -2.0, 0.0}, resolved.initialState(), TOL);
        assertArrayEquals(diagonal(4.0, 25.0, 0.0), resolved.initialStateCov(), TOL);
        assertArrayEquals(diagonal(0.0, 0.0, 1.0), resolved.initialDiffuseStateCov(), TOL);
        assertTrue(resolved.hasDiffuseComponent());
    }

    @Test
    void testCompositeResolveSolvesStationarySubBlockIndependently() {
        StateInitialization root = new StateInitialization(2)
            .set(0, 1, StateInitialization.stationary(1))
            .set(1, 2, StateInitialization.known(new double[]{9.0}, new double[]{7.0}));

        StateInitialization resolved = root.resolve(buildStationaryBlockModel());

        assertArrayEquals(new double[]{2.0, 9.0}, resolved.initialState(), TOL);
        assertArrayEquals(diagonal(4.0 / 3.0, 7.0), resolved.initialStateCov(), TOL);
        assertNull(resolved.initialDiffuseStateCov());
    }

    @Test
    void testResolveIntoMatchesResolveForNestedCompositeDirectWrite() {
        StateInitialization nested = new StateInitialization(2)
            .set(0, 1, StateInitialization.known(new double[]{3.0}, new double[]{4.0}))
            .set(1, 2, StateInitialization.stationary(1));
        StateInitialization root = new StateInitialization(3)
            .set(0, 2, nested)
            .set(2, 3, StateInitialization.diffuse(1));

        ModelFixture rep = buildNestedCompositeStationaryModel();
        StateInitialization resolved = root.resolve(rep);

        assertArrayEquals(new double[]{3.0, 2.0, 0.0}, resolved.initialState(), TOL);
        assertArrayEquals(diagonal(4.0, 16.0 / 3.0, 0.0), resolved.initialStateCov(), TOL);
        assertArrayEquals(diagonal(0.0, 0.0, 1.0), resolved.initialDiffuseStateCov(), TOL);

        double sentinel = -123.0;
        double[] state = new double[8];
        double[] covariance = new double[16];
        double[] diffuse = new double[16];
        StateInitialization.StationaryWorkspace stationaryWorkspace = new StateInitialization.StationaryWorkspace();
        Arrays.fill(state, sentinel);
        Arrays.fill(covariance, sentinel);
        Arrays.fill(diffuse, sentinel);

        boolean hasDiffuse = root.resolveInto(rep, 0, stationaryWorkspace,
            state, 2,
            covariance, 3,
            diffuse, 4);

        assertTrue(hasDiffuse);
        assertEquals(StateInitialization.requiredStationaryBackingLength(rep)
                + StateInitialization.requiredStationaryPivotLength(rep) * ((long) Integer.BYTES) / Double.BYTES,
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
        StateInitialization overlap = new StateInitialization(2)
            .set(0, 1, StateInitialization.diffuse(1));
        assertThrows(IllegalArgumentException.class,
            () -> overlap.set(0, 2, StateInitialization.diffuse(2)));

        StateInitialization gap = new StateInitialization(3)
            .set(0, 1, StateInitialization.diffuse(1))
            .set(2, 3, StateInitialization.diffuse(1));
        assertThrows(IllegalArgumentException.class,
            () -> gap.resolve(buildSimpleModel(3)));
    }
}