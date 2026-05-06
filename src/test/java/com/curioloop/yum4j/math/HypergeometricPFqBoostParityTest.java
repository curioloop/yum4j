package com.curioloop.yum4j.math;

import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.TestFactory;

import java.util.Collections;
import java.util.List;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.DynamicTest.dynamicTest;

class HypergeometricPFqBoostParityTest {

    private static final String FIXTURE_PATH = "boost/math/hypergeometric-pfq-boost-parity-fixture.json";

    @TestFactory
    Stream<DynamicTest> generalizedHypergeometricSurfaceMatchesBoostReferenceValues() {
        HypergeometricPFqBoostParityFixture fixture = ParityFixtureLoader.load(FIXTURE_PATH, HypergeometricPFqBoostParityFixture.class);
        return fixture.cases.stream().map(testCase -> dynamicTest(testCase.name, () -> assertCase(testCase)));
    }

    private static void assertCase(HypergeometricPFqParityCase testCase) {
        double[] absErrorHolder = new double[1];
        double actual = Hypergeometric.pFq(testCase.upper, testCase.lower, testCase.z, absErrorHolder);
        assertClose(testCase.name, testCase.expected, actual, testCase.tolerance);
        assertClose(testCase.name + " absError", testCase.expectedAbsError, absErrorHolder[0], 5.0e-12);
    }

    private static void assertClose(String label, double expected, double actual, double tolerance) {
        double scale = Math.max(1.0, Math.abs(expected));
        double error = Math.abs(actual - expected);
        assertTrue(error <= tolerance * scale,
            label + " mismatch: expected=" + expected + ", actual=" + actual + ", error=" + error);
    }

    public static final class HypergeometricPFqBoostParityFixture {
        public FixtureMeta meta;
        public List<HypergeometricPFqParityCase> cases = Collections.emptyList();
    }

    public static final class FixtureMeta {
        public String module;
        public String source;
    }

    public static final class HypergeometricPFqParityCase {
        public String name;
        public double[] upper = new double[0];
        public double[] lower = new double[0];
        public double z;
        public double expected;
        public double expectedAbsError;
        public double tolerance;
    }
}
