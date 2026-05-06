package com.curioloop.yum4j.stats;

import com.curioloop.yum4j.math.ParityFixtureLoader;
import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.TestFactory;

import java.util.Collections;
import java.util.List;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.DynamicTest.dynamicTest;

class CauchyDistributionParityTest {

    private static final String FIXTURE_PATH = "boost/math/cauchy-distribution-boost-parity-fixture.json";

    @TestFactory
    Stream<DynamicTest> boostReferenceValuesMatch() {
        CauchyDistributionParityFixture fixture = ParityFixtureLoader.load(FIXTURE_PATH, CauchyDistributionParityFixture.class);
        return fixture.cases.stream().map(testCase -> dynamicTest(testCase.name, () -> assertCase(testCase)));
    }

    private static void assertCase(CauchyDistributionCase testCase) {
        CauchyDistribution distribution = new CauchyDistribution(testCase.location, testCase.scale);

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
        assertClose(testCase.name + " median", testCase.median, distribution.median(), testCase.tolerance);
        assertClose(testCase.name + " mode", testCase.mode, distribution.mode(), testCase.tolerance);
        assertThrows(ArithmeticException.class, distribution::mean, testCase.name + " mean should be undefined");
        assertThrows(ArithmeticException.class, distribution::variance, testCase.name + " variance should be undefined");
        assertThrows(ArithmeticException.class, distribution::skewness, testCase.name + " skewness should be undefined");
        assertThrows(ArithmeticException.class, distribution::kurtosis, testCase.name + " kurtosis should be undefined");
        assertThrows(ArithmeticException.class, distribution::kurtosisExcess, testCase.name + " kurtosis excess should be undefined");
    }

    private static void assertClose(String label, double expected, double actual, double tolerance) {
        double scale = Math.max(1.0, Math.abs(expected));
        double error = Math.abs(actual - expected);
        assertTrue(error <= tolerance * scale,
            label + " mismatch: expected=" + expected + ", actual=" + actual + ", error=" + error);
    }

    public static final class CauchyDistributionParityFixture {
        public FixtureMeta meta;
        public List<CauchyDistributionCase> cases = Collections.emptyList();
    }

    public static final class FixtureMeta {
        public String module;
        public String source;
    }

    public static final class CauchyDistributionCase {
        public String name;
        public double location;
        public double scale;
        public double x;
        public double pdf;
        public double cdf;
        public double ccdf;
        public double quantileProbability;
        public double quantile;
        public double upperTailProbability;
        public double upperTailQuantile;
        public double median;
        public double mode;
        public double tolerance;
        public double quantileTolerance;
    }
}