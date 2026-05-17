package com.curioloop.yum4j.tsa.diagnostics;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DiagnosticSupportTest {

    @Test
    void extractsFiniteValuesAndColumns() {
        assertArrayEquals(new double[]{1.0, 2.0},
            DiagnosticSupport.finiteValues(new double[]{Double.NaN, 1.0, Double.POSITIVE_INFINITY, 2.0}));

        double[][] columns = DiagnosticSupport.finiteColumns(new double[][]{
            {1.0, Double.NaN},
            {2.0, 3.0},
            {Double.NaN, 4.0}
        }, 2);
        assertArrayEquals(new double[]{1.0, 2.0}, columns[0]);
        assertArrayEquals(new double[]{3.0, 4.0}, columns[1]);
    }

    @Test
    void computesInstantaneousCausalityFromFinitePairs() {
        double[][] residuals = {
            {1.0, 2.0},
            {2.0, 1.0},
            {3.0, 5.0},
            {Double.NaN, 3.0}
        };
        InstantaneousCausality result = DiagnosticSupport.instantaneousCausality(residuals, 0, 1, 2);
        assertEquals(0, result.caused());
        assertEquals(1, result.causing());
        assertEquals(1, result.degreesOfFreedom());
        assertTrue(Double.isFinite(result.statistic()));
        assertTrue(result.pValue() >= 0.0 && result.pValue() <= 1.0);
        assertArrayEquals(new double[]{1.0, 2.0, 2.0, 1.0, 3.0, 5.0},
            DiagnosticSupport.finitePairMatrix(residuals, 0, 1, 2));
        assertEquals(3, DiagnosticSupport.usablePairCount(residuals, 0, 1, 2));
        assertThrows(IllegalArgumentException.class, () -> DiagnosticSupport.instantaneousCausality(residuals, 0, 0, 2));
        assertThrows(IllegalArgumentException.class, () -> DiagnosticSupport.usablePairCount(residuals, 0, 2, 2));
    }
}