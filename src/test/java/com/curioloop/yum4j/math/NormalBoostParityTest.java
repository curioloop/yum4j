package com.curioloop.yum4j.math;

import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestFactory;

import java.util.Collections;
import java.util.List;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.DynamicTest.dynamicTest;

class NormalBoostParityTest {

    private static final String FIXTURE_PATH = "boost/math/normal-boost-parity-fixture.json";

    private static void assertClose(double expected, double actual, double relativeTolerance, double absoluteTolerance) {
        double tolerance = Math.max(absoluteTolerance, Math.abs(expected) * relativeTolerance);
        assertEquals(expected, actual, tolerance);
    }

    @TestFactory
    Stream<DynamicTest> helperSurfaceMatchesBoostReferenceValues() {
        NormalBoostParityFixture fixture = ParityFixtureLoader.load(FIXTURE_PATH, NormalBoostParityFixture.class);
        return fixture.cases.stream().map(testCase -> dynamicTest(testCase.name, () -> assertCase(testCase)));
    }

    private static void assertCase(NormalParityCase testCase) {
        assertClose(testCase.pdf, Normal.pdf(testCase.x), testCase.relativeTolerance, testCase.absoluteTolerance);
        assertClose(testCase.logPdf, Normal.logPdf(testCase.x), testCase.relativeTolerance, testCase.absoluteTolerance);
        assertClose(testCase.cdf, Normal.cdf(testCase.x), testCase.relativeTolerance, testCase.absoluteTolerance);
        assertClose(testCase.ccdf, Normal.ccdf(testCase.x), testCase.relativeTolerance, testCase.absoluteTolerance);
        assertClose(testCase.inv, Normal.inv(testCase.probability), testCase.relativeTolerance, testCase.absoluteTolerance);
        assertClose(
            testCase.invUpperTail,
            Normal.invUpperTail(testCase.probability),
            testCase.relativeTolerance,
            testCase.absoluteTolerance
        );
        assertClose(testCase.erfc, Normal.erfc(testCase.erfcArgument), testCase.relativeTolerance, testCase.absoluteTolerance);
        assertClose(
            testCase.erfInv,
            Normal.erfInv(testCase.erfInvArgument),
            testCase.relativeTolerance,
            testCase.absoluteTolerance
        );
        assertClose(
            testCase.erfcInv,
            Normal.erfcInv(testCase.erfcInvArgument),
            testCase.relativeTolerance,
            testCase.absoluteTolerance
        );
    }

    @Test
    void inverseUsesBoostStyleDomainAndOverflowSemantics() {
        assertThrows(IllegalArgumentException.class, () -> Normal.inv(Double.NaN));
        assertThrows(IllegalArgumentException.class, () -> Normal.inv(-1.0e-12));
        assertThrows(IllegalArgumentException.class, () -> Normal.inv(1.0 + 1.0e-12));
        assertThrows(ArithmeticException.class, () -> Normal.inv(0.0));
        assertThrows(ArithmeticException.class, () -> Normal.inv(1.0));
    }

    @Test
    void upperTailInverseUsesBoostStyleDomainAndOverflowSemantics() {
        assertThrows(IllegalArgumentException.class, () -> Normal.invUpperTail(Double.NaN));
        assertThrows(IllegalArgumentException.class, () -> Normal.invUpperTail(-1.0e-12));
        assertThrows(IllegalArgumentException.class, () -> Normal.invUpperTail(1.0 + 1.0e-12));
        assertThrows(ArithmeticException.class, () -> Normal.invUpperTail(0.0));
        assertThrows(ArithmeticException.class, () -> Normal.invUpperTail(1.0));
    }

    @Test
    void inverseErrorFunctionsUseBoostStyleDomainAndOverflowSemantics() {
        assertThrows(IllegalArgumentException.class, () -> Normal.erfInv(Double.NaN));
        assertThrows(IllegalArgumentException.class, () -> Normal.erfInv(-1.0 - 1.0e-12));
        assertThrows(IllegalArgumentException.class, () -> Normal.erfInv(1.0 + 1.0e-12));
        assertThrows(ArithmeticException.class, () -> Normal.erfInv(-1.0));
        assertThrows(ArithmeticException.class, () -> Normal.erfInv(1.0));

        assertThrows(IllegalArgumentException.class, () -> Normal.erfcInv(Double.NaN));
        assertThrows(IllegalArgumentException.class, () -> Normal.erfcInv(-1.0e-12));
        assertThrows(IllegalArgumentException.class, () -> Normal.erfcInv(2.0 + 1.0e-12));
        assertThrows(ArithmeticException.class, () -> Normal.erfcInv(0.0));
        assertThrows(ArithmeticException.class, () -> Normal.erfcInv(2.0));
    }

    @Test
    void cdfAndInverseRemainMutualInversesOnInteriorPoints() {
        double[] probabilities = {1.0e-12, 1.0e-8, 0.025, 0.2, 0.5, 0.8, 0.975, 1.0 - 1.0e-12};
        for (double probability : probabilities) {
            assertClose(probability, Normal.cdf(Normal.inv(probability)), 1.0e-13, 1.0e-15);
        }
    }

    @Test
    void ccdfAndUpperTailInverseRemainMutualInversesOnInteriorPoints() {
        double[] probabilities = {1.0e-12, 1.0e-8, 0.025, 0.2, 0.5, 0.8, 0.975, 1.0 - 1.0e-12};
        for (double probability : probabilities) {
            assertClose(probability, Normal.ccdf(Normal.invUpperTail(probability)), 1.0e-13, 1.0e-15);
        }
    }

    @Test
    void errorFunctionsRemainMutualInversesOnInteriorPoints() {
        double[] erfValues = {-0.95, -0.5, -0.125, 0.0, 0.125, 0.5, 0.95};
        for (double erfValue : erfValues) {
            assertClose(erfValue, Normal.erf(Normal.erfInv(erfValue)), 1.0e-14, 1.0e-15);
        }

        double[] erfcValues = {0.05, 0.25, 0.5, 1.0, 1.5, 1.75, 1.95};
        for (double erfcValue : erfcValues) {
            assertClose(erfcValue, Normal.erfc(Normal.erfcInv(erfcValue)), 1.0e-14, 1.0e-15);
        }
    }

    public static final class NormalBoostParityFixture {
        public FixtureMeta meta;
        public List<NormalParityCase> cases = Collections.emptyList();
    }

    public static final class FixtureMeta {
        public String module;
        public String source;
    }

    public static final class NormalParityCase {
        public String name;
        public double x;
        public double pdf;
        public double logPdf;
        public double cdf;
        public double ccdf;
        public double probability;
        public double inv;
        public double invUpperTail;
        public double erfcArgument;
        public double erfc;
        public double erfInvArgument;
        public double erfInv;
        public double erfcInvArgument;
        public double erfcInv;
        public double relativeTolerance;
        public double absoluteTolerance;
    }
}