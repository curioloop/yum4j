package com.curioloop.yum4j.math;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class OwensTTest {

    @Test
    void preservesSymmetryAndSpecialRelations() {
        double h = 0.75;
        double a = 0.5;

        assertClose(OwensT.value(h, a), OwensT.value(-h, a), 0.0);
        assertClose(-OwensT.value(h, a), OwensT.value(h, -a), 0.0);
        assertClose(Math.atan(a) / (2.0 * Math.PI), OwensT.value(0.0, a), 5.0e-15);
        assertClose(0.5 * Normal.cdf(h) * Normal.ccdf(h), OwensT.value(h, 1.0), 5.0e-15);
    }

    @Test
    void handlesInfiniteArgumentsViaOwensLimitingRelations() {
        double h = 0.5;
        assertClose(0.5 * Normal.ccdf(h), OwensT.value(h, Double.POSITIVE_INFINITY), 5.0e-15);
        assertClose(-0.5 * Normal.ccdf(h), OwensT.value(h, Double.NEGATIVE_INFINITY), 5.0e-15);
        assertEquals(0.0, OwensT.value(Double.POSITIVE_INFINITY, 2.0), 0.0);
        assertEquals(0.0, OwensT.value(Double.NEGATIVE_INFINITY, -2.0), 0.0);
    }

    @Test
    void nanInputsPropagate() {
        assertTrue(Double.isNaN(OwensT.value(Double.NaN, 0.5)));
        assertTrue(Double.isNaN(OwensT.value(0.5, Double.NaN)));
    }

    @Test
    void argumentTransformsMatchClosedRelationsForLargeA() {
        double hSmall = 0.5;
        double aLarge = 2.0;
        double expectedSmall = 0.25
            - (Normal.cdf(hSmall) - 0.5) * (Normal.cdf(aLarge * hSmall) - 0.5)
            - OwensT.value(aLarge * hSmall, 1.0 / aLarge);
        assertClose(expectedSmall, OwensT.value(hSmall, aLarge), 5.0e-15);

        double hLarge = 0.96875;
        double aVeryLarge = 7.0;
        double normH = Normal.ccdf(hLarge);
        double normAh = Normal.ccdf(aVeryLarge * hLarge);
        double expectedLarge = 0.5 * (normH + normAh)
            - normH * normAh
            - OwensT.value(aVeryLarge * hLarge, 1.0 / aVeryLarge);
        assertClose(expectedLarge, OwensT.value(hLarge, aVeryLarge), 5.0e-15);
    }

    private static void assertClose(double expected, double actual, double tolerance) {
        double scale = Math.max(1.0, Math.abs(expected));
        double error = Math.abs(actual - expected);
        assertTrue(error <= tolerance * scale,
            "mismatch: expected=" + expected + ", actual=" + actual + ", error=" + error);
    }
}