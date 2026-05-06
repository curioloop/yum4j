package com.curioloop.yum4j.stats;

import com.curioloop.yum4j.math.ParityFixtureLoader;
import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.TestFactory;

import java.util.Collections;
import java.util.List;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.DynamicTest.dynamicTest;

class BernoulliDistributionParityTest {

    private static final String FIXTURE_PATH = "boost/math/bernoulli-distribution-boost-parity-fixture.json";

    @TestFactory
    Stream<DynamicTest> boostReferenceValuesMatch() {
        BernoulliDistributionParityFixture fixture = ParityFixtureLoader.load(FIXTURE_PATH, BernoulliDistributionParityFixture.class);
        return fixture.cases.stream().map(testCase -> dynamicTest(testCase.name, () -> assertCase(testCase)));
    }

    private static void assertCase(BernoulliDistributionCase testCase) {
        BernoulliDistribution distribution = new BernoulliDistribution(testCase.successProbability);

        assertClose(testCase.name + " pdf", testCase.pdf, distribution.pdf(testCase.x), testCase.tolerance);
        assertClose(testCase.name + " cdf", testCase.cdf, distribution.cdf(testCase.x), testCase.tolerance);
        assertClose(testCase.name + " ccdf", testCase.ccdf, distribution.ccdf(testCase.x), testCase.tolerance);
        assertClose(testCase.name + " complement cdf", testCase.ccdf,
            Distributions.cdf(Distributions.complement(distribution, testCase.x)), testCase.tolerance);
        assertClose(testCase.name + " quantile", testCase.quantile,
            distribution.quantile(testCase.quantileProbability), testCase.quantileTolerance);
        assertClose(testCase.name + " upper-tail quantile", testCase.upperTailQuantile,
            distribution.quantileUpperTail(testCase.upperTailProbability), testCase.quantileTolerance);
        assertClose(testCase.name + " complement quantile", testCase.upperTailQuantile,
            Distributions.quantile(Distributions.complement(distribution, testCase.upperTailProbability)), testCase.quantileTolerance);
        assertClose(testCase.name + " cdf roundtrip quantile", testCase.x,
            distribution.quantile(distribution.cdf(testCase.x)), testCase.quantileTolerance);
        assertClose(testCase.name + " median", testCase.median, distribution.median(), testCase.quantileTolerance);
        assertClose(testCase.name + " mean", testCase.mean, distribution.mean(), testCase.tolerance);
        assertClose(testCase.name + " variance", testCase.variance, distribution.variance(), testCase.tolerance);
        assertFiniteOrSignedInfinity(
            testCase.name + " skewness",
            testCase.skewnessFinite,
            testCase.skewness,
            testCase.skewnessPositiveInfinity,
            distribution::skewness,
            testCase.tolerance
        );
        assertFiniteOrSignedInfinity(
            testCase.name + " kurtosis excess",
            testCase.kurtosisExcessFinite,
            testCase.kurtosisExcess,
            testCase.kurtosisExcessPositiveInfinity,
            distribution::kurtosisExcess,
            testCase.tolerance
        );
        assertFiniteOrSignedInfinity(
            testCase.name + " kurtosis",
            testCase.kurtosisFinite,
            testCase.kurtosis,
            testCase.kurtosisPositiveInfinity,
            distribution::kurtosis,
            testCase.tolerance
        );
        assertClose(testCase.name + " mode", testCase.mode, distribution.mode(), testCase.quantileTolerance);
    }

    private static void assertClose(String label, double expected, double actual, double tolerance) {
        double scale = Math.max(1.0, Math.abs(expected));
        double error = Math.abs(actual - expected);
        assertTrue(error <= tolerance * scale,
            label + " mismatch: expected=" + expected + ", actual=" + actual + ", error=" + error);
    }

    private static void assertFiniteOrSignedInfinity(String label,
                                                     boolean finite,
                                                     double expected,
                                                     boolean positiveInfinity,
                                                     StatisticSupplier supplier,
                                                     double tolerance) {
        double actual = supplier.getAsDouble();
        if (finite) {
            assertClose(label, expected, actual, tolerance);
            return;
        }
        assertTrue(
            Double.isInfinite(actual) && (positiveInfinity ? actual > 0.0 : actual < 0.0),
            label + " mismatch: expected=" + (positiveInfinity ? "+Infinity" : "-Infinity") + ", actual=" + actual
        );
    }

    @FunctionalInterface
    private interface StatisticSupplier {
        double getAsDouble();
    }

    public static final class BernoulliDistributionParityFixture {
        public FixtureMeta meta;
        public List<BernoulliDistributionCase> cases = Collections.emptyList();
    }

    public static final class FixtureMeta {
        public String module;
        public String source;
    }

    public static final class BernoulliDistributionCase {
        public String name;
        public double successProbability;
        public double x;
        public double pdf;
        public double cdf;
        public double ccdf;
        public double quantileProbability;
        public double quantile;
        public double upperTailProbability;
        public double upperTailQuantile;
        public double median;
        public double mean;
        public double variance;
        public boolean skewnessFinite;
        public double skewness;
        public boolean skewnessPositiveInfinity;
        public boolean kurtosisFinite;
        public double kurtosis;
        public boolean kurtosisPositiveInfinity;
        public boolean kurtosisExcessFinite;
        public double kurtosisExcess;
        public boolean kurtosisExcessPositiveInfinity;
        public double mode;
        public double tolerance;
        public double quantileTolerance;
    }
}