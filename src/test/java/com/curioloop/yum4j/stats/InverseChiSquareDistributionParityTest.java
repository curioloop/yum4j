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

class InverseChiSquareDistributionParityTest {

    private static final String FIXTURE_PATH = "boost/math/inverse-chi-square-distribution-boost-parity-fixture.json";

    @TestFactory
    Stream<DynamicTest> boostReferenceValuesMatch() {
        InverseChiSquareDistributionParityFixture fixture = ParityFixtureLoader.load(FIXTURE_PATH, InverseChiSquareDistributionParityFixture.class);
        return fixture.cases.stream().map(testCase -> dynamicTest(testCase.name, () -> assertCase(testCase)));
    }

    private static void assertCase(InverseChiSquareDistributionCase testCase) {
        InverseChiSquareDistribution distribution = new InverseChiSquareDistribution(testCase.degreesOfFreedom, testCase.scale);

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
        assertClose(testCase.name + " mode", testCase.mode, distribution.mode(), testCase.quantileTolerance);
        assertOptionalMoment(
            testCase.name + " mean",
            testCase.meanDefined,
            testCase.mean,
            distribution::mean,
            testCase.tolerance
        );
        assertOptionalMoment(
            testCase.name + " variance",
            testCase.varianceDefined,
            testCase.variance,
            distribution::variance,
            testCase.tolerance
        );
        assertOptionalMoment(
            testCase.name + " skewness",
            testCase.skewnessDefined,
            testCase.skewness,
            distribution::skewness,
            testCase.tolerance
        );
        assertOptionalMoment(
            testCase.name + " kurtosis",
            testCase.kurtosisDefined,
            testCase.kurtosis,
            distribution::kurtosis,
            testCase.tolerance
        );
        assertOptionalMoment(
            testCase.name + " kurtosis excess",
            testCase.kurtosisDefined,
            testCase.kurtosisExcess,
            distribution::kurtosisExcess,
            testCase.tolerance
        );
    }

    private static void assertClose(String label, double expected, double actual, double tolerance) {
        double scale = Math.max(1.0, Math.abs(expected));
        double error = Math.abs(actual - expected);
        assertTrue(error <= tolerance * scale,
            label + " mismatch: expected=" + expected + ", actual=" + actual + ", error=" + error);
    }

    private static void assertOptionalMoment(String label,
                                             boolean defined,
                                             double expected,
                                             StatisticSupplier supplier,
                                             double tolerance) {
        if (defined) {
            assertClose(label, expected, supplier.getAsDouble(), tolerance);
        } else {
            assertThrows(ArithmeticException.class, supplier::getAsDouble, label + " should be undefined");
        }
    }

    @FunctionalInterface
    private interface StatisticSupplier {
        double getAsDouble();
    }

    public static final class InverseChiSquareDistributionParityFixture {
        public FixtureMeta meta;
        public List<InverseChiSquareDistributionCase> cases = Collections.emptyList();
    }

    public static final class FixtureMeta {
        public String module;
        public String source;
    }

    public static final class InverseChiSquareDistributionCase {
        public String name;
        public double degreesOfFreedom;
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
        public boolean meanDefined;
        public double mean;
        public boolean varianceDefined;
        public double variance;
        public boolean skewnessDefined;
        public double skewness;
        public boolean kurtosisDefined;
        public double kurtosis;
        public double kurtosisExcess;
        public double tolerance;
        public double quantileTolerance;
    }
}