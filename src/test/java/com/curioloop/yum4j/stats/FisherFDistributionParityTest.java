package com.curioloop.yum4j.stats;

import com.curioloop.yum4j.math.ParityFixtureLoader;
import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.TestFactory;

import java.util.Collections;
import java.util.List;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.DynamicTest.dynamicTest;

class FisherFDistributionParityTest {

    private static final String FIXTURE_PATH = "boost/math/fisher-f-distribution-boost-parity-fixture.json";

    @TestFactory
    Stream<DynamicTest> boostReferenceValuesMatch() {
        FisherFDistributionParityFixture fixture = ParityFixtureLoader.load(FIXTURE_PATH, FisherFDistributionParityFixture.class);
        return fixture.cases.stream().map(testCase -> dynamicTest(testCase.name, () -> assertCase(testCase)));
    }

    private static void assertCase(FisherFDistributionCase testCase) {
        FisherFDistribution distribution = new FisherFDistribution(testCase.degreesOfFreedom1, testCase.degreesOfFreedom2);

        assertClose(testCase.name + " pdf", testCase.pdf, distribution.pdf(testCase.x), testCase.tolerance);
        assertClose(testCase.name + " cdf", testCase.cdf, distribution.cdf(testCase.x), testCase.tolerance);
        assertClose(testCase.name + " ccdf", testCase.ccdf, distribution.ccdf(testCase.x), testCase.tolerance);
        assertClose(testCase.name + " complement cdf", testCase.ccdf,
            Distributions.cdf(Distributions.complement(distribution, testCase.x)), testCase.tolerance);
        assertClose(testCase.name + " quantile", testCase.x, distribution.quantile(testCase.cdf), testCase.quantileTolerance);
        assertClose(testCase.name + " upper-tail quantile", testCase.x, distribution.quantileUpperTail(testCase.ccdf), testCase.quantileTolerance);
        assertClose(testCase.name + " complement quantile", testCase.x,
            Distributions.quantile(Distributions.complement(distribution, testCase.ccdf)), testCase.quantileTolerance);
        assertClose(testCase.name + " median", testCase.median, distribution.median(), testCase.quantileTolerance);
        assertClose(testCase.name + " mean", testCase.mean, distribution.mean(), testCase.tolerance);
        assertClose(testCase.name + " variance", testCase.variance, distribution.variance(), testCase.tolerance);
        assertClose(testCase.name + " skewness", testCase.skewness, distribution.skewness(), testCase.tolerance);
        assertClose(testCase.name + " kurtosis", testCase.kurtosis, distribution.kurtosis(), testCase.tolerance);
        assertClose(testCase.name + " kurtosis excess", testCase.kurtosisExcess, distribution.kurtosisExcess(), testCase.tolerance);
        assertClose(testCase.name + " mode", testCase.mode, distribution.mode(), testCase.tolerance);
    }

    private static void assertClose(String label, double expected, double actual, double tolerance) {
        double scale = Math.max(1.0, Math.abs(expected));
        double error = Math.abs(actual - expected);
        assertTrue(error <= tolerance * scale,
            label + " mismatch: expected=" + expected + ", actual=" + actual + ", error=" + error);
    }

    public static final class FisherFDistributionParityFixture {
        public FixtureMeta meta;
        public List<FisherFDistributionCase> cases = Collections.emptyList();
    }

    public static final class FixtureMeta {
        public String module;
        public String source;
    }

    public static final class FisherFDistributionCase {
        public String name;
        public double degreesOfFreedom1;
        public double degreesOfFreedom2;
        public double x;
        public double pdf;
        public double cdf;
        public double ccdf;
        public double median;
        public double mean;
        public double variance;
        public double skewness;
        public double kurtosis;
        public double kurtosisExcess;
        public double mode;
        public double tolerance;
        public double quantileTolerance;
    }
}