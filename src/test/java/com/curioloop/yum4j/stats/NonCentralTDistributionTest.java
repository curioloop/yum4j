package com.curioloop.yum4j.stats;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class NonCentralTDistributionTest {

    @Test
    void zeroNonCentralityReducesToStudentsTDistribution() {
        NonCentralTDistribution distribution = new NonCentralTDistribution(6.0, 0.0);
        StudentsTDistribution student = new StudentsTDistribution(6.0);

        assertEquals(student.pdf(0.75), distribution.pdf(0.75), 1.0e-15);
        assertEquals(student.cdf(-0.4), distribution.cdf(-0.4), 1.0e-15);
        assertEquals(student.ccdf(1.1), distribution.ccdf(1.1), 1.0e-15);
        assertEquals(student.quantile(0.2), distribution.quantile(0.2), 1.0e-12);
        assertEquals(student.quantileUpperTail(0.3), distribution.quantileUpperTail(0.3), 1.0e-12);
        assertEquals(0.0, distribution.mode(), 1.0e-12);
        assertEquals(0.0, distribution.mean(), 1.0e-15);
        assertEquals(1.5, distribution.variance(), 1.0e-15);
        assertEquals(0.0, distribution.skewness(), 1.0e-15);
        assertEquals(4.0, distribution.kurtosis(), 1.0e-12);
        assertEquals(1.0, distribution.kurtosisExcess(), 1.0e-12);
    }

    @Test
    void infiniteDegreesOfFreedomReducesToShiftedNormalDistribution() {
        NonCentralTDistribution distribution = new NonCentralTDistribution(Double.POSITIVE_INFINITY, 1.25);
        NormalDistribution normal = new NormalDistribution(1.25, 1.0);

        assertEquals(normal.pdf(0.9), distribution.pdf(0.9), 1.0e-15);
        assertEquals(normal.cdf(0.9), distribution.cdf(0.9), 1.0e-15);
        assertEquals(normal.quantile(0.75), distribution.quantile(0.75), 1.0e-12);
        assertEquals(normal.quantileUpperTail(0.25), distribution.quantileUpperTail(0.25), 1.0e-12);
        assertEquals(1.25, distribution.mode(), 1.0e-12);
        assertEquals(1.25, distribution.mean(), 1.0e-15);
        assertEquals(1.0, distribution.variance(), 1.0e-15);
        assertEquals(0.0, distribution.skewness(), 1.0e-15);
        assertEquals(4.0, distribution.kurtosis(), 1.0e-15);
        assertEquals(1.0, distribution.kurtosisExcess(), 1.0e-15);
    }

    @Test
    void representativeBoostCaseMatchesReferenceValues() {
        NonCentralTDistribution distribution = new NonCentralTDistribution(7.0, 1.5);

        assertEquals(0.35602739430288594, distribution.pdf(1.2), 1.0e-12);
        assertEquals(0.3716903605396073, distribution.cdf(1.2), 1.0e-12);
        assertEquals(0.62830963946039264, distribution.ccdf(1.2), 1.0e-12);
        assertEquals(1.3647784864425734, distribution.mode(), 5.0e-8);
        assertEquals(1.6888032982579013, distribution.mean(), 1.0e-12);
        assertEquals(1.6979434197932337, distribution.variance(), 1.0e-12);
        assertEquals(0.94772110981577706, distribution.skewness(), 1.0e-11);
        assertEquals(6.6190796481837113, distribution.kurtosis(), 1.0e-11);
        assertEquals(3.6190796481837113, distribution.kurtosisExcess(), 1.0e-11);
        assertEquals(1.2, distribution.quantile(distribution.cdf(1.2)), 1.0e-8);
        assertEquals(1.2, distribution.quantileUpperTail(distribution.ccdf(1.2)), 1.0e-8);
        assertEquals(1.0, distribution.cdf(1.2) + distribution.ccdf(1.2), 1.0e-12);
    }

    @Test
    void invalidArgumentsAndUndefinedMomentsAreRejected() {
        assertThrows(IllegalArgumentException.class, () -> new NonCentralTDistribution(0.0, 1.0));
        assertThrows(IllegalArgumentException.class, () -> new NonCentralTDistribution(Double.NaN, 1.0));
        assertThrows(IllegalArgumentException.class, () -> new NonCentralTDistribution(5.0, Double.NaN));
        assertThrows(IllegalArgumentException.class, () -> new NonCentralTDistribution(5.0, 1.0).cdf(Double.NaN));
        assertThrows(IllegalArgumentException.class, () -> new NonCentralTDistribution(5.0, 1.0).quantile(-0.1));
        assertThrows(ArithmeticException.class, () -> new NonCentralTDistribution(1.0, 1.0).mean());
        assertThrows(ArithmeticException.class, () -> new NonCentralTDistribution(2.0, 1.0).variance());
        assertThrows(ArithmeticException.class, () -> new NonCentralTDistribution(3.0, 1.0).skewness());
        assertThrows(ArithmeticException.class, () -> new NonCentralTDistribution(4.0, 1.0).kurtosisExcess());
        assertThrows(ArithmeticException.class, () -> new NonCentralTDistribution(4.0, 1.0).kurtosis());
    }
}