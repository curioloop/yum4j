package com.curioloop.yum4j.stats;

import com.curioloop.yum4j.math.Beta;
import com.curioloop.yum4j.math.Gamma;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class StudentsTDistributionTest {

    @Test
    void accessorsAndStatisticsMatchBoostDefinitions() {
        StudentsTDistribution distribution = new StudentsTDistribution(6.0);

        assertEquals(6.0, distribution.degreesOfFreedom());
        assertEquals(0.0, distribution.mode());
        assertEquals(0.0, distribution.median());
        assertEquals(0.0, distribution.mean());
        assertEquals(1.5, distribution.variance(), 1.0e-15);
        assertEquals(0.0, distribution.skewness(), 1.0e-15);
        assertEquals(6.0, distribution.kurtosis(), 1.0e-15);
        assertEquals(3.0, distribution.kurtosisExcess(), 1.0e-15);
        double vp1 = 3.5;
        double vd2 = 3.0;
        double expectedEntropy = vp1 * (Gamma.digamma(vp1) - Gamma.digamma(vd2))
            + Math.log(Math.sqrt(6.0) * Beta.beta(vd2, 0.5));
        assertEquals(expectedEntropy, distribution.entropy(), 1.0e-15);
    }

    @Test
    void undefinedMomentsRaise() {
        assertThrows(ArithmeticException.class, () -> new StudentsTDistribution(1.0).mean());
        assertThrows(ArithmeticException.class, () -> new StudentsTDistribution(2.0).variance());
        assertThrows(ArithmeticException.class, () -> new StudentsTDistribution(3.0).skewness());
        assertThrows(ArithmeticException.class, () -> new StudentsTDistribution(4.0).kurtosis());
        assertThrows(ArithmeticException.class, () -> new StudentsTDistribution(4.0).kurtosisExcess());
    }

    @Test
    void justBelowMomentThresholdsStillRaise() {
        assertThrows(ArithmeticException.class, () -> new StudentsTDistribution(Math.nextDown(1.0)).mean());
        assertThrows(ArithmeticException.class, () -> new StudentsTDistribution(Math.nextDown(2.0)).variance());
        assertThrows(ArithmeticException.class, () -> new StudentsTDistribution(Math.nextDown(3.0)).skewness());
        assertThrows(ArithmeticException.class, () -> new StudentsTDistribution(Math.nextDown(4.0)).kurtosis());
        assertThrows(ArithmeticException.class, () -> new StudentsTDistribution(Math.nextDown(4.0)).kurtosisExcess());
    }

    @Test
    void justAboveMomentThresholdsUseBoostFormulas() {
        double meanDf = Math.nextUp(1.0);
        double varianceDf = Math.nextUp(2.0);
        double skewnessDf = Math.nextUp(3.0);
        double kurtosisDf = Math.nextUp(4.0);

        assertEquals(0.0, new StudentsTDistribution(meanDf).mean(), 0.0);
        assertEquals(varianceDf / (varianceDf - 2.0), new StudentsTDistribution(varianceDf).variance(), 0.0);
        assertEquals(0.0, new StudentsTDistribution(skewnessDf).skewness(), 0.0);
        assertEquals(6.0 / (kurtosisDf - 4.0) + 3.0, new StudentsTDistribution(kurtosisDf).kurtosis(), 0.0);
        assertEquals(6.0 / (kurtosisDf - 4.0), new StudentsTDistribution(kurtosisDf).kurtosisExcess(), 0.0);
    }

    @Test
    void infiniteDegreesOfFreedomUsesNormalLimitStatistics() {
        StudentsTDistribution distribution = new StudentsTDistribution(Double.POSITIVE_INFINITY);

        assertEquals(0.0, distribution.mean(), 0.0);
        assertEquals(1.0, distribution.variance(), 0.0);
        assertEquals(0.0, distribution.skewness(), 0.0);
        assertEquals(3.0, distribution.kurtosis(), 0.0);
        assertEquals(0.0, distribution.kurtosisExcess(), 0.0);
    }

    @Test
    void invalidInputsAreRejected() {
        assertThrows(IllegalArgumentException.class, () -> new StudentsTDistribution(0.0));
        assertThrows(IllegalArgumentException.class, () -> new StudentsTDistribution(Double.NaN));
        assertThrows(IllegalArgumentException.class, () -> new StudentsTDistribution(2.0).cdf(Double.NaN));
        assertThrows(ArithmeticException.class, () -> new StudentsTDistribution(2.0).quantile(0.0));
        assertThrows(ArithmeticException.class, () -> new StudentsTDistribution(2.0).quantile(1.0));
        assertThrows(ArithmeticException.class, () -> new StudentsTDistribution(2.0).quantileUpperTail(0.0));
        assertThrows(ArithmeticException.class, () -> new StudentsTDistribution(2.0).quantileUpperTail(1.0));
        assertThrows(IllegalArgumentException.class,
            () -> StudentsTDistribution.findDegreesOfFreedom(0.0, 0.05, 0.1, 1.0, 30.0));
    }

    @Test
    void findDegreesOfFreedomDefaultHintMatchesExplicitDefault() {
        double differenceFromMean = 0.5;
        double alpha = 0.05;
        double beta = 0.10;
        double sampleStandardDeviation = 1.2;

        double withDefault = StudentsTDistribution.findDegreesOfFreedom(
            differenceFromMean,
            alpha,
            beta,
            sampleStandardDeviation
        );
        double withExplicit = StudentsTDistribution.findDegreesOfFreedom(
            differenceFromMean,
            alpha,
            beta,
            sampleStandardDeviation,
            100.0
        );

        assertEquals(withExplicit, withDefault, 1.0e-12);
    }

    @Test
    void findDegreesOfFreedomSatisfiesBoostEquation() {
        double differenceFromMean = 0.5;
        double alpha = 0.05;
        double beta = 0.10;
        double sampleStandardDeviation = 1.2;

        double result = StudentsTDistribution.findDegreesOfFreedom(
            differenceFromMean,
            alpha,
            beta,
            sampleStandardDeviation,
            30.0
        );

        StudentsTDistribution distribution = new StudentsTDistribution(result);
        double qa = distribution.quantileUpperTail(alpha);
        double qb = distribution.quantileUpperTail(beta);
        double ratio = sampleStandardDeviation * sampleStandardDeviation / (differenceFromMean * differenceFromMean);
        double residual = (qa + qb) * (qa + qb) * ratio - (result + 1.0);

        assertTrue(result > 0.0);
        assertEquals(0.0, residual, 1.0e-8);
    }

    @Test
    void quantileRoundTripsThroughDfLessThanThreeBodySeriesBranch() {
        StudentsTDistribution distribution = new StudentsTDistribution(2.5);

        double probability = 0.7;
        double quantile = distribution.quantile(probability);

        assertEquals(probability, distribution.cdf(quantile), 1.0e-12);
        assertEquals(quantile, distribution.quantileUpperTail(1.0 - probability), 1.0e-12);
    }
}