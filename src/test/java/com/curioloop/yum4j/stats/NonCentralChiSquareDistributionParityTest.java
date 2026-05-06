package com.curioloop.yum4j.stats;

import com.curioloop.yum4j.math.ParityFixtureLoader;
import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.TestFactory;

import java.util.Collections;
import java.util.List;
import java.util.stream.Stream;

import static org.junit.jupiter.api.DynamicTest.dynamicTest;
import static org.junit.jupiter.api.Assertions.assertTrue;

class NonCentralChiSquareDistributionParityTest {

    private static final String FIXTURE_PATH = "boost/math/non-central-chi-square-distribution-boost-parity-fixture.json";

    @TestFactory
    Stream<DynamicTest> boostReferenceValuesMatch() {
        NonCentralChiSquareDistributionParityFixture fixture = ParityFixtureLoader.load(
            FIXTURE_PATH,
            NonCentralChiSquareDistributionParityFixture.class
        );
        return fixture.cases.stream().map(testCase -> dynamicTest(testCase.name, () -> assertCase(testCase)));
    }

    private static void assertCase(NonCentralChiSquareDistributionCase testCase) {
        NonCentralChiSquareDistribution distribution = new NonCentralChiSquareDistribution(
            testCase.degreesOfFreedom,
            testCase.nonCentrality
        );
        double actualPdf = NonCentralChiSquareDistribution.pdf(
            testCase.degreesOfFreedom,
            testCase.nonCentrality,
            testCase.x
        );
        assertClose(testCase.name + " pdf", testCase.pdf, actualPdf, Math.max(testCase.tolerance, 1.0e-10));

        double actualMode = distribution.mode();
        assertClose(testCase.name + " mode", testCase.mode, actualMode, testCase.quantileTolerance);
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

        double actualCdf = NonCentralChiSquareDistribution.cdf(
            testCase.degreesOfFreedom,
            testCase.nonCentrality,
            testCase.x
        );
        assertClose(testCase.name + " cdf", testCase.cdf, actualCdf, testCase.tolerance);

        double actualCcdf = NonCentralChiSquareDistribution.ccdf(
            testCase.degreesOfFreedom,
            testCase.nonCentrality,
            testCase.x
        );
        assertClose(testCase.name + " ccdf", testCase.ccdf, actualCcdf, testCase.tolerance);
        assertClose(
            testCase.name + " complement cdf",
            testCase.ccdf,
            Distributions.cdf(Distributions.complement(distribution, testCase.x)),
            testCase.tolerance
        );

        double actualQuantile = NonCentralChiSquareDistribution.quantile(
            testCase.degreesOfFreedom,
            testCase.nonCentrality,
            testCase.quantileProbability
        );
        assertClose(testCase.name + " quantile", testCase.quantile, actualQuantile, testCase.quantileTolerance);

        double actualUpperTailQuantile = NonCentralChiSquareDistribution.quantileUpperTail(
            testCase.degreesOfFreedom,
            testCase.nonCentrality,
            testCase.upperTailProbability
        );
        assertClose(
            testCase.name + " upper-tail quantile",
            testCase.upperTailQuantile,
            actualUpperTailQuantile,
            testCase.quantileTolerance
        );
        assertClose(
            testCase.name + " complement quantile",
            testCase.upperTailQuantile,
            Distributions.quantile(Distributions.complement(distribution, testCase.upperTailProbability)),
            testCase.quantileTolerance
        );

        double actualDirectQuantile = NonCentralChiSquareDistribution.quantile(
            testCase.degreesOfFreedom,
            testCase.nonCentrality,
            testCase.directQuantileProbability
        );
        assertClose(
            testCase.name + " direct quantile",
            testCase.directQuantile,
            actualDirectQuantile,
            testCase.quantileTolerance
        );

        double actualDirectUpperTailQuantile = NonCentralChiSquareDistribution.quantileUpperTail(
            testCase.degreesOfFreedom,
            testCase.nonCentrality,
            testCase.directUpperTailProbability
        );
        assertClose(
            testCase.name + " direct upper-tail quantile",
            testCase.directUpperTailQuantile,
            actualDirectUpperTailQuantile,
            testCase.quantileTolerance
        );
    }

    private static void assertClose(String label, double expected, double actual, double tolerance) {
        double scale = Math.max(1.0, Math.abs(expected));
        double error = Math.abs(actual - expected);
        assertTrue(error <= tolerance * scale,
            label + " mismatch: expected=" + expected + ", actual=" + actual + ", error=" + error);
    }

    public static final class NonCentralChiSquareDistributionParityFixture {
        public FixtureMeta meta;
        public List<NonCentralChiSquareDistributionCase> cases = Collections.emptyList();
    }

    public static final class FixtureMeta {
        public String module;
        public String source;
    }

    public static final class NonCentralChiSquareDistributionCase {
        public String name;
        public double degreesOfFreedom;
        public double nonCentrality;
        public double x;
        public double pdf;
        public double cdf;
        public double ccdf;
        public double mode;
        public double median;
        public double mean;
        public double variance;
        public double skewness;
        public double kurtosis;
        public double kurtosisExcess;
        public double quantileProbability;
        public double quantile;
        public double upperTailProbability;
        public double upperTailQuantile;
        public double directQuantileProbability;
        public double directQuantile;
        public double directUpperTailProbability;
        public double directUpperTailQuantile;
        public double tolerance;
        public double quantileTolerance;
    }
}