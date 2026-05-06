package com.curioloop.yum4j.math;

import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.TestFactory;

import java.util.Collections;
import java.util.List;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.DynamicTest.dynamicTest;

class GammaPolygammaBoostParityTest {

    private static final String FIXTURE_PATH = "boost/math/gamma-polygamma-boost-parity-fixture.json";

    @TestFactory
    Stream<DynamicTest> gammaPolygammaSurfaceMatchesBoostReferenceValues() {
        GammaPolygammaBoostParityFixture fixture = ParityFixtureLoader.load(FIXTURE_PATH, GammaPolygammaBoostParityFixture.class);
        return fixture.cases.stream().map(testCase -> dynamicTest(testCase.name, () -> assertCase(testCase)));
    }

    private static void assertCase(GammaPolygammaParityCase testCase) {
        assertClose(testCase.name + " polygamma",
            testCase.expected,
            Gamma.polygamma(testCase.order, testCase.x),
            testCase.tolerance);

        if (testCase.order == 0) {
            assertClose(testCase.name + " digamma",
                testCase.expected,
                Gamma.digamma(testCase.x),
                testCase.tolerance);
        }
        if (testCase.order == 1) {
            assertClose(testCase.name + " trigamma",
                testCase.expected,
                Gamma.trigamma(testCase.x),
                testCase.tolerance);
        }
    }

    private static void assertClose(String label, double expected, double actual, double tolerance) {
        double scale = Math.max(1.0, Math.abs(expected));
        double error = Math.abs(actual - expected);
        assertTrue(error <= tolerance * scale,
            label + " mismatch: expected=" + expected + ", actual=" + actual + ", error=" + error);
    }

    public static final class GammaPolygammaBoostParityFixture {
        public FixtureMeta meta;
        public List<GammaPolygammaParityCase> cases = Collections.emptyList();
    }

    public static final class FixtureMeta {
        public String module;
        public String source;
    }

    public static final class GammaPolygammaParityCase {
        public String name;
        public String branch;
        public int order;
        public double x;
        public double expected;
        public double tolerance;
    }
}