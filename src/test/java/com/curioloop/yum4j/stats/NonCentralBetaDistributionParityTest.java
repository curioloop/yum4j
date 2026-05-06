package com.curioloop.yum4j.stats;

import com.curioloop.yum4j.math.ParityFixtureLoader;
import org.junit.jupiter.api.Test;

import java.util.Collections;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class NonCentralBetaDistributionParityTest {

    private static final String FIXTURE_PATH = "boost/math/non-central-beta-distribution-boost-parity-fixture.json";

    @Test
    void boostReferenceValuesMatch() {
        NonCentralBetaDistributionParityFixture fixture =
            ParityFixtureLoader.load(FIXTURE_PATH, NonCentralBetaDistributionParityFixture.class);
        for (NonCentralBetaDistributionCase testCase : fixture.cases) {
            assertCase(testCase);
        }
    }

    private static void assertCase(NonCentralBetaDistributionCase testCase) {
        NonCentralBetaDistribution distribution = new NonCentralBetaDistribution(
            testCase.alpha,
            testCase.beta,
            testCase.nonCentrality
        );

        assertClose(testCase.name + " pdf", testCase.pdf, distribution.pdf(testCase.x), testCase.tolerance);
        assertClose(testCase.name + " cdf", testCase.cdf, distribution.cdf(testCase.x), testCase.tolerance);
        assertClose(testCase.name + " ccdf", testCase.ccdf, distribution.ccdf(testCase.x), testCase.tolerance);
        assertClose(testCase.name + " complement cdf", testCase.ccdf,
            Distributions.cdf(Distributions.complement(distribution, testCase.x)), testCase.tolerance);
        assertClose(testCase.name + " quantile", testCase.quantile,
            distribution.quantile(testCase.cdf), testCase.quantileTolerance);
        assertClose(testCase.name + " upper-tail quantile", testCase.upperTailQuantile,
            distribution.quantileUpperTail(testCase.ccdf), testCase.quantileTolerance);
        assertClose(testCase.name + " complement quantile", testCase.upperTailQuantile,
            Distributions.quantile(Distributions.complement(distribution, testCase.ccdf)), testCase.quantileTolerance);
        assertClose(testCase.name + " median", testCase.median, distribution.median(), testCase.quantileTolerance);
        assertClose(testCase.name + " mean", testCase.mean, distribution.mean(), testCase.tolerance);
        assertClose(testCase.name + " variance", testCase.variance, distribution.variance(), testCase.tolerance);
        assertClose(testCase.name + " mode", testCase.mode, distribution.mode(), testCase.modeTolerance);
        assertThrows(ArithmeticException.class, distribution::skewness, testCase.name + " skewness should be undefined");
        assertThrows(ArithmeticException.class, distribution::kurtosisExcess, testCase.name + " kurtosis excess should be undefined");
        assertThrows(ArithmeticException.class, distribution::kurtosis, testCase.name + " kurtosis should be undefined");
    }

    private static void assertClose(String label, double expected, double actual, double tolerance) {
        double scale = Math.max(1.0, Math.abs(expected));
        double error = Math.abs(actual - expected);
        assertTrue(error <= tolerance * scale,
            label + " mismatch: expected=" + expected + ", actual=" + actual + ", error=" + error);
    }

    public static final class NonCentralBetaDistributionParityFixture {
        public FixtureMeta meta;
        public List<NonCentralBetaDistributionCase> cases = Collections.emptyList();
    }

    public static final class FixtureMeta {
        public String module;
        public String source;
    }

    public static final class NonCentralBetaDistributionCase {
        public String name;
        public double alpha;
        public double beta;
        public double nonCentrality;
        public double x;
        public double pdf;
        public double cdf;
        public double ccdf;
        public double median;
        public double mean;
        public double variance;
        public double mode;
        public double quantile;
        public double upperTailQuantile;
        public double tolerance;
        public double quantileTolerance;
        public double modeTolerance;
    }
}