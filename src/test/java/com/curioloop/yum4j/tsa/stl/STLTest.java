package com.curioloop.yum4j.tsa.stl;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class STLTest {

    @Test
    void stlDecomposesMonthlySeasonality() {
        int period = 12;
        double[] seasonalPattern = {-1.6, -1.1, -0.4, 0.3, 1.1, 1.7, 1.4, 0.7, 0.0, -0.5, -0.9, -0.7};
        double[] values = new double[period * 12];
        for (int i = 0; i < values.length; i++) {
            double trend = 20.0 + 0.04 * i + 0.0002 * i * i;
            values[i] = trend + seasonalPattern[i % period];
        }

        STLResult result = STL.builder(period).robust(false).build().fit(values);

        assertEquals(values.length, result.length());
        assertReconstruction(values, result);
        assertTrue(rms(result.residual()) < 0.35);
    }

    @Test
    void robustStlDownweightsLargeOutlier() {
        int period = 12;
        double[] values = new double[period * 14];
        for (int i = 0; i < values.length; i++) {
            values[i] = 5.0 + 0.02 * i + Math.sin(2.0 * Math.PI * i / period);
        }
        values[73] += 80.0;

        STLResult result = STL.builder(period).robust(true).build().fit(values);

        assertTrue(result.weight(73) < 0.2);
        for (double weight : result.weights()) {
            assertTrue(weight >= 0.0 && weight <= 1.0);
        }
        assertReconstruction(values, result);
    }

    @Test
    void stlAcceptsExplicitFitIterationsAndJumpControls() {
        int period = 12;
        double[] values = new double[period * 10];
        for (int i = 0; i < values.length; i++) {
            values[i] = 3.0 + 0.05 * i + Math.cos(2.0 * Math.PI * i / period);
        }

        STL stl = STL.builder(period)
            .seasonal(9, 0, 2)
            .trend(17, 1, 2)
            .lowPass(13, 1, 2)
            .build();
        STLResult result = stl.fit(values, 3, 1);

        assertReconstruction(values, result);
        assertThrows(IllegalArgumentException.class, () -> stl.fit(values, 0, 0));
        assertThrows(IllegalArgumentException.class, () -> stl.fit(values, 1, -1));
    }

    @Test
    void mstlSeparatesMultipleSeasonalComponents() {
        int n = 720;
        double[] values = new double[n];
        for (int i = 0; i < n; i++) {
            double t = i + 1.0;
            double trend = 100.0 + 0.01 * t + 0.00002 * t * t;
            double daily = 4.0 * Math.sin(2.0 * Math.PI * t / 24.0);
            double weekly = 7.0 * Math.cos(2.0 * Math.PI * t / (24.0 * 7.0));
            values[i] = trend + daily + weekly;
        }

        MSTL mstl = MSTL.builder(24, 24 * 7).iterations(2).build();
        MSTLResult result = mstl.fit(values);

        assertArrayEquals(new int[]{24, 24 * 7}, mstl.periods());
        assertEquals(2, result.seasonalCount());
        assertEquals(n, result.length());
        assertMultiSeasonReconstruction(result);
        assertTrue(rms(result.residual()) < 1.5);
    }

    @Test
    void mstlDropsPeriodsAtLeastHalfTheSeriesLength() {
        double[] values = new double[120];
        for (int i = 0; i < values.length; i++) {
            values[i] = 10.0 + 0.01 * i + Math.sin(2.0 * Math.PI * i / 12.0);
        }

        MSTLResult result = MSTL.builder(12, 96).windows(11, 15).build().fit(values);

        assertEquals(1, result.seasonalCount());
        assertMultiSeasonReconstruction(result);
    }

    @Test
    void mstlAppliesFiniteBoxCoxLambda() {
        double[] values = new double[96];
        for (int i = 0; i < values.length; i++) {
            values[i] = 20.0 + 0.05 * i + 0.5 * Math.sin(2.0 * Math.PI * i / 12.0);
        }

        MSTLResult result = MSTL.builder(12).lambda(0.0).build().fit(values);

        for (int i = 0; i < values.length; i++) {
            assertEquals(Math.log(values[i]), result.observed(i), 1e-12);
        }
        assertMultiSeasonReconstruction(result);

        double[] invalid = values.clone();
        invalid[7] = -1.0;
        assertThrows(IllegalArgumentException.class,
            () -> MSTL.builder(12).lambda(0.5).build().fit(invalid));
    }

    @Test
    void validatesConfiguration() {
        assertThrows(IllegalArgumentException.class, () -> STL.builder(1).build());
        assertThrows(IllegalArgumentException.class, () -> STL.builder(12).trend(STLSmooth.of(11, 1, 1)).build());
        assertThrows(IllegalArgumentException.class, () -> STLSmooth.of(4, 1, 1));
        assertThrows(IllegalArgumentException.class, () -> MSTL.builder(24).windows(11, 15).build());
        assertThrows(IllegalArgumentException.class, () -> MSTL.builder(24).innerIterations(0));
        assertThrows(IllegalArgumentException.class, () -> MSTL.builder(24).outerIterations(-1));
        assertThrows(IllegalArgumentException.class, () -> MSTL.builder(24).lambda(Double.POSITIVE_INFINITY));
    }

    private static void assertReconstruction(double[] values, STLResult result) {
        assertArrayEquals(values, result.observed(), 0.0);
        for (int i = 0; i < values.length; i++) {
            double actual = result.seasonal(i) + result.trend(i) + result.residual(i);
            assertEquals(result.observed(i), actual, 1e-9);
        }
    }

    private static void assertMultiSeasonReconstruction(MSTLResult result) {
        for (int i = 0; i < result.length(); i++) {
            double actual = result.trend(i) + result.residual(i);
            for (int component = 0; component < result.seasonalCount(); component++) {
                actual += result.seasonal(component, i);
            }
            assertEquals(result.observed(i), actual, 1e-8);
        }
    }

    private static double rms(double[] values) {
        double sum = 0.0;
        for (double value : values) {
            sum += value * value;
        }
        return Math.sqrt(sum / values.length);
    }
}
