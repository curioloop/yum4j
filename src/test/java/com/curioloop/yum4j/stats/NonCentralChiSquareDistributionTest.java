package com.curioloop.yum4j.stats;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class NonCentralChiSquareDistributionTest {

    @Test
    void accessorsMatchBoostRangeAndSupport() {
        NonCentralChiSquareDistribution distribution = new NonCentralChiSquareDistribution(3.5, 2.0);

        assertEquals(0.0, distribution.range().lower(), 0.0);
        assertEquals(Double.MAX_VALUE, distribution.range().upper(), 0.0);
        assertEquals(0.0, distribution.support().lower(), 0.0);
        assertEquals(Double.MAX_VALUE, distribution.support().upper(), 0.0);
    }

    @Test
    void statisticsMatchClosedFormFormulas() {
        NonCentralChiSquareDistribution distribution = new NonCentralChiSquareDistribution(3.5, 2.0);

        assertEquals(5.5, distribution.mean(), 1.0e-15);
        assertEquals(15.0, distribution.variance(), 1.0e-15);
        assertEquals(1.3082077080522831, distribution.skewness(), 1.0e-12);
        assertEquals(5.453333333333333, distribution.kurtosis(), 1.0e-12);
        assertEquals(2.453333333333333, distribution.kurtosisExcess(), 1.0e-12);
        assertEquals(distribution.quantile(0.5), distribution.median(), 1.0e-12);
    }

    @Test
    void modeIsZeroWhenDensityDivergesAtOriginAndOtherwiseLocallyMaximal() {
        assertEquals(0.0, NonCentralChiSquareDistribution.mode(1.5, 0.0), 0.0);

        NonCentralChiSquareDistribution distribution = new NonCentralChiSquareDistribution(3.5, 2.0);
        double mode = distribution.mode();
        double step = Math.max(1.0e-6, mode * 1.0e-6);

        assertTrue(mode >= 0.0);
        assertTrue(distribution.pdf(mode) >= distribution.pdf(Math.max(0.0, mode - step)));
        assertTrue(distribution.pdf(mode) >= distribution.pdf(mode + step));
    }

    @Test
    void centralCasePdfMatchesClosedFormAtOriginAndInterior() {
        assertEquals(0.5, NonCentralChiSquareDistribution.pdf(2.0, 0.0, 0.0), 1.0e-15);
        assertEquals(0.18393972058572117, NonCentralChiSquareDistribution.pdf(4.0, 0.0, 2.0), 1.0e-15);
    }

    @Test
    void nonCentralPdfAtOriginMatchesBoostBoundarySemantics() {
        assertEquals(0.0, NonCentralChiSquareDistribution.pdf(3.5, 2.0, 0.0), 0.0);
    }

    @Test
    void finderHelpersReturnSelfConsistentSolutions() {
        double degreesOfFreedom = NonCentralChiSquareDistribution.findDegreesOfFreedom(2.0, 4.0, 0.42424000248778682);
        assertEquals(3.5, degreesOfFreedom, 1.0e-8);

        double degreesOfFreedomUpperTail = NonCentralChiSquareDistribution.findDegreesOfFreedomUpperTail(2.0, 4.0, 0.5757599975122132);
        assertEquals(3.5, degreesOfFreedomUpperTail, 1.0e-8);

        double nonCentrality = NonCentralChiSquareDistribution.findNonCentrality(3.5, 4.0, 0.42424000248778682);
        assertEquals(2.0, nonCentrality, 1.0e-8);

        double nonCentralityUpperTail = NonCentralChiSquareDistribution.findNonCentralityUpperTail(3.5, 4.0, 0.5757599975122132);
        assertEquals(2.0, nonCentralityUpperTail, 1.0e-8);
    }

    @Test
    void invalidArgumentsAreRejected() {
        assertThrows(IllegalArgumentException.class, () -> NonCentralChiSquareDistribution.pdf(0.0, 1.0, 1.0));
        assertThrows(IllegalArgumentException.class, () -> NonCentralChiSquareDistribution.cdf(3.5, 2.0, Double.POSITIVE_INFINITY));
        assertThrows(IllegalArgumentException.class, () -> NonCentralChiSquareDistribution.pdf(3.5, 2.0, Double.POSITIVE_INFINITY));
        assertThrows(IllegalArgumentException.class, () -> NonCentralChiSquareDistribution.findDegreesOfFreedom(1.0, 1.0, 0.0));
        assertThrows(IllegalArgumentException.class, () -> NonCentralChiSquareDistribution.findDegreesOfFreedomUpperTail(1.0, 1.0, 1.0));
        assertThrows(IllegalArgumentException.class, () -> NonCentralChiSquareDistribution.findNonCentrality(1.0, 1.0, 1.0));
        assertThrows(IllegalArgumentException.class, () -> NonCentralChiSquareDistribution.findNonCentralityUpperTail(1.0, 1.0, 0.0));
        assertThrows(ArithmeticException.class, () -> NonCentralChiSquareDistribution.findNonCentrality(10.0, 0.01, 0.99));
        assertThrows(ArithmeticException.class, () -> NonCentralChiSquareDistribution.findNonCentralityUpperTail(10.0, 0.01, 0.01));
    }

    @Test
    void pdfIsNonNegativeAcrossRepresentativeCases() {
        for (double value : new double[]{0.0, 1.0, 4.0, 20.0, 55.0}) {
            assertTrue(NonCentralChiSquareDistribution.pdf(3.5, 2.0, value) >= 0.0);
            assertTrue(NonCentralChiSquareDistribution.pdf(3.5, 50.0, value) >= 0.0);
        }
    }
}