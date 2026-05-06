package com.curioloop.yum4j.math;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class MinimaTest {

    @Test
    void findsQuadraticMinimum() {
        Minima result = Minima.brentFindMinima(
            x -> (x - 1.75) * (x - 1.75) + 2.0,
            -2.0,
            5.0,
            40,
            100
        );

        assertEquals(1.75, result.point(), 2.0e-8);
        assertEquals(2.0, result.value(), 1.0e-12);
        assertTrue(result.converged());
    }

    @Test
    void reportsConsumedIterations() {
        Minima result = Minima.brentFindMinima(
            x -> Math.exp(x) - 3.0 * x,
            -1.0,
            2.0,
            32,
            32
        );

        assertTrue(result.iterations() > 0);
        assertTrue(result.iterations() <= 32);
        assertEquals(Math.log(3.0), result.point(), 2.0e-8);
        assertEquals(3.0 - 3.0 * Math.log(3.0), result.value(), 1.0e-10);
        assertTrue(result.converged());
    }

    @Test
    void returnsNonConvergedResultWhenIterationBudgetIsExhausted() {
        Minima result = Minima.brentFindMinima(
            x -> (x - 1.75) * (x - 1.75) + 2.0,
            -2.0,
            5.0,
            40,
            1
        );

        assertEquals(1, result.iterations());
        assertFalse(result.converged());
    }
}