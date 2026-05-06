package com.curioloop.yum4j.stats;

import com.curioloop.yum4j.math.ParityFixtureLoader;
import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.TestFactory;

import java.util.Collections;
import java.util.List;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.DynamicTest.dynamicTest;

class SkewNormalDistributionParityTest {

    private static final String FIXTURE_PATH = "boost/math/skew-normal-distribution-boost-parity-fixture.json";

    @TestFactory
    Stream<DynamicTest> boostReferenceValuesMatch() {
        SkewNormalDistributionParityFixture fixture = ParityFixtureLoader.load(FIXTURE_PATH, SkewNormalDistributionParityFixture.class);
        return fixture.cases.stream().map(testCase -> dynamicTest(testCase.name, () -> assertCase(testCase)));
    }

    private static void assertCase(SkewNormalDistributionCase testCase) {
        SkewNormalDistribution distribution = new SkewNormalDistribution(testCase.location, testCase.scale, testCase.shape);

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
        assertClose(testCase.name + " median", testCase.median, distribution.median(), testCase.quantileTolerance);
        assertClose(testCase.name + " mean", testCase.mean, distribution.mean(), testCase.tolerance);
        assertClose(testCase.name + " variance", testCase.variance, distribution.variance(), testCase.tolerance);
        assertClose(testCase.name + " skewness", testCase.skewness, distribution.skewness(), testCase.tolerance);
        assertClose(testCase.name + " kurtosis", testCase.kurtosis, distribution.kurtosis(), testCase.tolerance);
        assertClose(
            testCase.name + " kurtosis excess",
            testCase.kurtosisExcess,
            distribution.kurtosisExcess(),
            testCase.tolerance
        );
        assertClose(testCase.name + " mode", testCase.mode, distribution.mode(), testCase.modeTolerance);
    }

    private static void assertClose(String label, double expected, double actual, double tolerance) {
        double scale = Math.max(1.0, Math.abs(expected));
        double error = Math.abs(actual - expected);
        assertTrue(error <= tolerance * scale,
            label + " mismatch: expected=" + expected + ", actual=" + actual + ", error=" + error);
    }

    public static final class SkewNormalDistributionParityFixture {
        public FixtureMeta meta;
        public List<SkewNormalDistributionCase> cases = Collections.emptyList();
    }

    public static final class FixtureMeta {
        public String module;
        public String source;
    }

    public static final class SkewNormalDistributionCase {
        public String name;
        public double location;
        public double scale;
        public double shape;
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
        public double skewness;
        public double kurtosis;
        public double kurtosisExcess;
        public double mode;
        public double tolerance;
        public double quantileTolerance;
        public double modeTolerance;
    }
}