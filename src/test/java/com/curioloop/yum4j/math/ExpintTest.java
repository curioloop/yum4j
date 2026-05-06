package com.curioloop.yum4j.math;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ExpintTest {

    @Test
    void eiMatchesBoostRepresentativeValues() {
        assertClose(-6.3532793397275915136, Expint.expint(1.0 / 1024.0), 5.0e-14);
        assertClose(-1.3732085249429833378, Expint.expint(0.125), 5.0e-14);
        assertClose(0.45421990486317357992, Expint.expint(0.5), 5.0e-14);
        assertClose(1.8951178163559367555, Expint.expint(1.0), 5.0e-14);
        assertClose(1.7276319560291180520e20, Expint.expint(50.5), 5.0e-13);

        assertClose(-6.3552324648310718026, Expint.expint(-1.0 / 1024.0), 5.0e-14);
        assertClose(-1.6234256405841687915, Expint.expint(-0.125), 5.0e-14);
        assertClose(-0.55977359477616081175, Expint.expint(-0.5), 5.0e-14);
        assertClose(-0.21938393439552027368, Expint.expint(-1.0), 5.0e-14);
        assertClose(-2.2723713293221935044e-24, Expint.expint(-50.5), 5.0e-13);
    }

    @Test
    void indexedSurfaceMatchesBoostRepresentativeValues() {
        assertClose(7.0599752206767632229, Expint.expint(0, 0.125), 5.0e-14);
        assertClose(1.6234256405841687915, Expint.expint(1, 0.125), 5.0e-14);
        assertClose(0.67956869751157430393, Expint.expint(2, 0.125), 5.0e-14);
        assertClose(0.13097731169586484778, Expint.expint(5, 0.5), 5.0e-14);
        assertClose(0.038529924425495155396, Expint.expint(5, 1.5), 5.0e-14);
        assertClose(2.6900194251201629501e-24, Expint.expint(22, 50.0), 5.0e-13);
    }

    @Test
    void indexedSurfaceSatisfiesRecurrenceAwayFromSingularities() {
        int[] orders = {1, 2, 5, 22};
        double[] points = {0.125, 0.5, 1.5, 4.5};

        for (int n : orders) {
            for (double z : points) {
                double left = n * Expint.expint(n + 1, z);
                double right = Math.exp(-z) - z * Expint.expint(n, z);
                assertClose(right, left, 5.0e-13);
            }
        }
    }

    @Test
    void boundaryBehaviorMatchesBoostRealSurface() {
        assertTrue(Double.isInfinite(Expint.expint(0.0)) && Expint.expint(0.0) < 0.0);
        assertEquals(Double.POSITIVE_INFINITY, Expint.expint(Double.POSITIVE_INFINITY));
        assertEquals(0.0, Expint.expint(Double.NEGATIVE_INFINITY), 0.0);
        assertEquals(0.0, Expint.expint(-Double.MAX_VALUE), 0.0);
        assertEquals(Double.POSITIVE_INFINITY, Expint.expint(2.0 * Math.log(Double.MAX_VALUE)));
        assertEquals(Double.POSITIVE_INFINITY, Expint.expint(Math.log(Double.MAX_VALUE) + 38.0));

        assertThrows(ArithmeticException.class, () -> Expint.expint(0, 0.0));
        assertThrows(ArithmeticException.class, () -> Expint.expint(1, 0.0));
        assertEquals(1.0, Expint.expint(2, 0.0), 0.0);
        assertEquals(0.5, Expint.expint(3, 0.0), 0.0);
        assertEquals(0.0, Expint.expint(2, Double.POSITIVE_INFINITY), 0.0);
    }

    @Test
    void invalidInputsAreRejected() {
        assertThrows(IllegalArgumentException.class, () -> Expint.expint(Double.NaN));
        assertThrows(IllegalArgumentException.class, () -> Expint.expint(-1, 1.0));
        assertThrows(IllegalArgumentException.class, () -> Expint.expint(1, Double.NaN));
        assertThrows(IllegalArgumentException.class, () -> Expint.expint(1, -1.0));
        assertThrows(IllegalArgumentException.class, () -> Expint.expint(2, Double.NEGATIVE_INFINITY));
    }

    private static void assertClose(double expected, double actual, double tolerance) {
        double scale = Math.max(1.0, Math.abs(expected));
        double error = Math.abs(expected - actual);
        assertTrue(error <= tolerance * scale,
            "mismatch: expected=" + expected + ", actual=" + actual + ", error=" + error);
    }
}