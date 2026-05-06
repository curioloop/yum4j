package com.curioloop.yum4j.math;

import org.junit.jupiter.api.Test;

import java.util.function.DoubleUnaryOperator;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class Toms748Test {

    private static final double FULL_PRECISION_EPSILON = Math.max(Math.scalb(1.0, 1 - 53), 4.0 * Math.ulp(1.0));

    @Test
    void solveWithEndpointValuesFindsSqrtTwo() {
        DoubleUnaryOperator function = x -> x * x - 2.0;

        Double2 interval = Toms748.solve(
            function,
            1.0,
            2.0,
            function.applyAsDouble(1.0),
            function.applyAsDouble(2.0),
            FULL_PRECISION_EPSILON,
            50,
            "sqrt2"
        );

        assertEquals(Math.sqrt(2.0), midpoint(interval), 1.0e-12);
    }

    @Test
    void solveWithZeroIterationsReturnsInitialBracket() {
        DoubleUnaryOperator function = x -> x * x - 2.0;

        Double2 interval = Toms748.solve(
            function,
            1.0,
            2.0,
            -1.0,
            2.0,
            FULL_PRECISION_EPSILON,
            0,
            "sqrt2"
        );

        assertEquals(1.0, interval._1());
        assertEquals(2.0, interval._2());
    }

    @Test
    void toms748SolveFindsIncreasingRoot() {
        double root = RootIterators.toms748Solve(
            x -> x * x - 2.0,
            1.0,
            2.0,
            true,
            53,
            64,
            "sqrt2"
        );

        assertEquals(Math.sqrt(2.0), root, 1.0e-12);
    }

    @Test
    void toms748SolveHandlesNegativeGuessForIncreasingFunction() {
        double root = RootIterators.toms748Solve(
            x -> x + 2.0,
            -1.0,
            2.0,
            true,
            53,
            64,
            "negative-root"
        );

        assertEquals(-2.0, root, 1.0e-12);
    }

    @Test
    void toms748SolveBoundedFindsPositiveRoot() {
        double root = RootIterators.toms748SolveBounded(
            x -> x - 0.25,
            0.8,
            2.0,
            true,
            0.1,
            53,
            64,
            "positive-root"
        );

        assertEquals(0.25, root, 1.0e-12);
        assertTrue(root >= 0.1);
    }

    @Test
    void toms748Solve01FindsUnitIntervalRoot() {
        double root = RootIterators.toms748Solve01(
            x -> x - 0.8,
            0.2,
            2.0,
            true,
            53,
            64,
            "unit-root"
        );

        assertEquals(0.8, root, 1.0e-12);
        assertTrue(root >= 0.0);
        assertTrue(root <= 1.0);
    }

    @Test
    void equalFloorToleranceTerminatesOnSharedIntegerBucket() {
        DoubleUnaryOperator function = x -> x - 10.25;

        Double2 interval = Toms748.solve(
            function,
            10.0,
            11.0,
            function.applyAsDouble(10.0),
            function.applyAsDouble(11.0),
            Toms748.Tolerance.EQUAL_FLOOR,
            32,
            "equal-floor"
        );

        assertEquals(Math.floor(interval._1()), Math.floor(interval._2()));
        assertTrue(interval._1() <= 10.25 && interval._2() >= 10.25);
    }

    private static double midpoint(Double2 interval) {
        return 0.5 * (interval._1() + interval._2());
    }
}