package com.curioloop.yum4j.math;

import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.TestFactory;

import java.util.Collections;
import java.util.List;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.DynamicTest.dynamicTest;

class CarlsonIntegralsBoostParityTest {

    private static final String FIXTURE_PATH = "boost/math/carlson-symmetric-integrals-boost-parity-fixture.json";

    @TestFactory
    Stream<DynamicTest> carlsonSymmetricIntegralsMatchBoostReferenceValues() {
        CarlsonBoostParityFixture fixture = ParityFixtureLoader.load(FIXTURE_PATH, CarlsonBoostParityFixture.class);
        return fixture.cases.stream().map(testCase -> dynamicTest(testCase.name, () -> assertCase(testCase)));
    }

    private static void assertCase(CarlsonParityCase testCase) {
        double actual = switch (testCase.function) {
            case "rc" -> CarlsonIntegrals.rc(testCase.x, testCase.y);
            case "rf" -> CarlsonIntegrals.rf(testCase.x, testCase.y, testCase.z);
            case "rd" -> CarlsonIntegrals.rd(testCase.x, testCase.y, testCase.z);
            case "rj" -> CarlsonIntegrals.rj(testCase.x, testCase.y, testCase.z, testCase.p);
            case "rg" -> CarlsonIntegrals.rg(testCase.x, testCase.y, testCase.z);
            default -> throw new IllegalArgumentException("Unknown Carlson function: " + testCase.function);
        };
        assertClose(testCase.name, testCase.expected, actual, testCase.tolerance);
    }

    private static void assertClose(String label, double expected, double actual, double tolerance) {
        double scale = Math.max(1.0, Math.abs(expected));
        double error = Math.abs(actual - expected);
        assertTrue(error <= tolerance * scale,
            label + " mismatch: expected=" + expected + ", actual=" + actual + ", error=" + error);
    }

    public static final class CarlsonBoostParityFixture {
        public FixtureMeta meta;
        public List<CarlsonParityCase> cases = Collections.emptyList();
    }

    public static final class FixtureMeta {
        public String module;
        public String source;
    }

    public static final class CarlsonParityCase {
        public String name;
        public String function;
        public double x;
        public double y;
        public double z;
        public double p;
        public double expected;
        public double tolerance;
    }
}