package com.curioloop.yum4j.ssm.kalman.init;

import com.curioloop.yum4j.ssm.kalman.model.KalmanSSM;
import org.junit.jupiter.api.Test;

import java.lang.reflect.Method;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ComplexStationarySolverTest {

    private static final double TOL = 1e-9;

    @Test
    void testRealifiedCovarianceFallbackMatchesDirectSolve() throws Exception {
        KalmanSSM model = buildStableComplexModel();

        double[] direct = ComplexStationarySolver.solveCov(model, 0);
        assertNotNull(direct);

        double[] fallback = new double[2 * model.stateCount() * model.stateCount()];
        Method method = ComplexStationarySolver.class.getDeclaredMethod(
            "solveCovViaRealEmbedding",
            KalmanSSM.class,
            int.class,
            double[].class,
            int.class,
            int.class);
        method.setAccessible(true);

        boolean ok = (boolean) method.invoke(null, model, 0, fallback, 0, model.stateCount());
        assertTrue(ok);
        assertArrayEquals(direct, fallback, TOL);
    }

    @Test
    void testRealifiedMeanFallbackMatchesDirectSolve() throws Exception {
        KalmanSSM model = buildStableComplexModel();

        double[] direct = ComplexStationarySolver.solveMean(model, 0);
        assertNotNull(direct);

        double[] fallback = new double[2 * model.stateCount()];
        Method method = ComplexStationarySolver.class.getDeclaredMethod(
            "solveMeanViaRealEmbedding",
            KalmanSSM.class,
            int.class,
            double[].class,
            int.class);
        method.setAccessible(true);

        boolean ok = (boolean) method.invoke(null, model, 0, fallback, 0);
        assertTrue(ok);
        assertArrayEquals(direct, fallback, TOL);
    }

    private static KalmanSSM buildStableComplexModel() {
        return KalmanSSM.complexBuilder(1, 2, 2, 1)
            .design(new double[]{1.0, 0.0, 0.0, 0.0}, false)
            .obsIntercept(new double[]{0.0, 0.0}, false)
            .obsCov(new double[]{0.0, 0.0}, false)
            .transition(new double[]{
                0.45, 0.10, 0.05, -0.02,
                0.00, 0.00, 0.30, 0.20
            }, false)
            .stateIntercept(new double[]{0.25, -0.15, -0.10, 0.35}, false)
            .selection(new double[]{
                1.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0
            }, false)
            .stateCovariance(new double[]{
                0.70, 0.00, 0.10, 0.05,
                0.10, -0.05, 0.60, 0.00
            }, false)
            .endog(new double[]{0.0, 0.0})
            .allObserved()
            .build();
    }
}