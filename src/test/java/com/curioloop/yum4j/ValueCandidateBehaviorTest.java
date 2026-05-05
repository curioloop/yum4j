package com.curioloop.yum4j;

import com.curioloop.yum4j.fft.FftWorkspace;
import com.curioloop.yum4j.kalman.filter.FilterResultShape;
import com.curioloop.yum4j.kalman.filter.FilterSpec;
import com.curioloop.yum4j.kalman.smooth.SimulationSmootherSpec;
import com.curioloop.yum4j.kalman.smooth.SmootherResultShape;
import com.curioloop.yum4j.kalman.smooth.SmootherSpec;
import com.curioloop.yum4j.tsa.sarimax.ARIMAOrder;
import com.curioloop.yum4j.tsa.sarimax.SeasonalOrder;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ValueCandidateBehaviorTest {

    @Test
    void testArimaOrderHelpersAndValidation() {
        ARIMAOrder zero = ARIMAOrder.of(0, 0, 0);
        ARIMAOrder order = ARIMAOrder.of(2, 1, 3);

        assertTrue(zero.isZero());
        assertFalse(order.isZero());
        assertEquals(new ARIMAOrder(2, 1, 3), order);

        assertThrows(IllegalArgumentException.class, () -> ARIMAOrder.of(-1, 0, 0));
        assertThrows(IllegalArgumentException.class, () -> ARIMAOrder.of(0, -1, 0));
        assertThrows(IllegalArgumentException.class, () -> ARIMAOrder.of(0, 0, -1));
    }

    @Test
    void testSeasonalOrderHelpersAndValidation() {
        SeasonalOrder none = SeasonalOrder.none();
        SeasonalOrder order = SeasonalOrder.of(1, 0, 1, 12);

        assertTrue(none.isZero());
        assertFalse(order.isZero());
        assertEquals(new SeasonalOrder(1, 0, 1, 12), order);

        assertThrows(IllegalArgumentException.class, () -> SeasonalOrder.of(-1, 0, 0, 0));
        assertThrows(IllegalArgumentException.class, () -> SeasonalOrder.of(0, -1, 0, 0));
        assertThrows(IllegalArgumentException.class, () -> SeasonalOrder.of(0, 0, -1, 0));
        assertThrows(IllegalArgumentException.class, () -> SeasonalOrder.of(1, 0, 0, 1));
        assertThrows(IllegalArgumentException.class, () -> SeasonalOrder.of(0, 0, 0, -1));
    }

    @Test
    void testFilterSpecProjectionAndCanonicalization() {
        FilterSpec forecastOnly = FilterSpec.full().without(
            FilterSpec.Storage.PREDICTED_STATE,
            FilterSpec.Storage.FILTERED_STATE,
            FilterSpec.Storage.KALMAN_GAIN,
            FilterSpec.Storage.LIKELIHOOD);

        FilterResultShape withoutMissing = forecastOnly.resultShape(false);
        FilterResultShape withMissing = forecastOnly.resultShape(true);

        assertSame(FilterSpec.defaults(), FilterSpec.full());
        assertTrue(withoutMissing.storesForecast());
        assertFalse(withoutMissing.storesPredictedState());
        assertFalse(withoutMissing.storesFilteredState());
        assertFalse(withoutMissing.storesKalmanGain());
        assertFalse(withoutMissing.storesLikelihood());
        assertTrue(withMissing.storesPredictedState());
        assertSame(withoutMissing, forecastOnly.resultShape(false));
        assertSame(withMissing, forecastOnly.resultShape(true));
        assertEquals(withoutMissing.layoutOptions(), FilterResultShape.of(true, false, false, false, false).layoutOptions());
    }

    @Test
    void testSmootherSpecProjectionAndCanonicalization() {
        SmootherSpec stateOnly = SmootherSpec.conventional().stateOnly();
        SmootherSpec noCovariances = SmootherSpec.classical().withoutCovariances();
        SmootherResultShape shape = stateOnly.resultShape();

        assertSame(FilterSpec.conventionalSmoothing(), stateOnly.requiredFilterSpec());
        assertSame(noCovariances,
            SmootherSpec.classical().without(
                SmootherSpec.Output.STATE_COVARIANCE,
                SmootherSpec.Output.DISTURBANCE_COVARIANCE));
        assertTrue(shape.storesState());
        assertFalse(shape.storesStateCovariance());
        assertFalse(shape.storesDisturbance());
        assertFalse(shape.storesDisturbanceCovariance());
        assertFalse(shape.storesAuxiliary());
        assertSame(shape, stateOnly.resultShape());
    }

    @Test
    void testSimulationSmootherSpecProjection() {
        SimulationSmootherSpec spec = SimulationSmootherSpec.all()
            .withUnivariateFilter()
            .withGeneratedOutputs();
        SmootherSpec required = spec.requiredSmootherSpec();

        assertTrue(spec.includes(SimulationSmootherSpec.Output.STATE));
        assertTrue(spec.includes(SimulationSmootherSpec.Output.DISTURBANCE));
        assertTrue(spec.usesUnivariateFilter());
        assertTrue(spec.storesGeneratedOutputs());
        assertSame(FilterSpec.conventionalSmoothing(), spec.requiredFilterSpec());
        assertTrue(required.includes(SmootherSpec.Output.STATE));
        assertTrue(required.includes(SmootherSpec.Output.DISTURBANCE));
        assertFalse(required.includes(SmootherSpec.Output.STATE_COVARIANCE));
        assertFalse(required.includes(SmootherSpec.Output.DISTURBANCE_COVARIANCE));
    }

    @Test
    void testShapeFactoriesAreCanonicalForSameMask() {
        FilterResultShape filterShape = FilterResultShape.of(true, false, true, false, true);
        SmootherResultShape smootherShape = SmootherResultShape.of(true, false, true, false, false);

        assertSame(filterShape, FilterResultShape.of(true, false, true, false, true));
        assertTrue(filterShape.storesForecast());
        assertFalse(filterShape.storesPredictedState());
        assertTrue(filterShape.storesFilteredState());
        assertFalse(filterShape.storesKalmanGain());
        assertTrue(filterShape.storesLikelihood());

        assertSame(smootherShape, SmootherResultShape.of(true, false, true, false, false));
        assertTrue(smootherShape.storesState());
        assertFalse(smootherShape.storesStateCovariance());
        assertTrue(smootherShape.storesDisturbance());
        assertFalse(smootherShape.storesDisturbanceCovariance());
        assertFalse(smootherShape.storesAuxiliary());
    }

    @Test
    void testFftRequirementCompositionAndValidation() {
        FftWorkspace.Requirement base = FftWorkspace.Requirement.doubles(3, 4);
        FftWorkspace.Requirement extra = new FftWorkspace.Requirement(5, 2);
        FftWorkspace.Requirement combined = base.plus(extra);
        FftWorkspace.Requirement shared = base.max(extra);

        assertSame(FftWorkspace.Requirement.empty(), FftWorkspace.Requirement.empty());
        assertEquals(7, base.totalDoubles());
        assertEquals(0, base.intSlots());
        assertEquals(12, combined.totalDoubles());
        assertEquals(2, combined.intSlots());
        assertEquals(7, shared.totalDoubles());
        assertEquals(2, shared.intSlots());
        assertEquals(7, base.withIntSlots(9).totalDoubles());
        assertEquals(9, base.withIntSlots(9).intSlots());

        assertThrows(IllegalArgumentException.class, () -> new FftWorkspace.Requirement(-1, 0));
        assertThrows(IllegalArgumentException.class, () -> new FftWorkspace.Requirement(0, -1));
        assertThrows(IllegalArgumentException.class, () -> FftWorkspace.Requirement.doubles(1, -1));
    }
}