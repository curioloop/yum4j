package com.curioloop.yum4j.math;

import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.TestFactory;

import java.util.Collections;
import java.util.List;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.DynamicTest.dynamicTest;

class BesselZeroBoostParityTest {

    private static final String FIXTURE_PATH = "boost/math/bessel-zero-boost-parity-fixture.json";

    @TestFactory
    Stream<DynamicTest> cylindricalBesselZeroSurfaceMatchesBoostReferenceValues() {
        BesselZeroBoostParityFixture fixture = ParityFixtureLoader.load(FIXTURE_PATH, BesselZeroBoostParityFixture.class);
        return fixture.cases.stream().map(testCase -> dynamicTest(testCase.name, () -> assertCase(testCase)));
    }

    private static void assertCase(BesselZeroParityCase testCase) {
        double actual = switch (testCase.family) {
            case "j" -> Bessel.jZero(testCase.nu, testCase.rank);
            case "y" -> Bessel.yZero(testCase.nu, testCase.rank);
            default -> throw new IllegalArgumentException("Unknown Bessel zero family: " + testCase.family);
        };
        assertClose(testCase.name + " zero", testCase.expectedZero, actual, testCase.tolerance);
    }

    private static void assertClose(String label, double expected, double actual, double tolerance) {
        double scale = Math.max(1.0, Math.abs(expected));
        double error = Math.abs(actual - expected);
        assertTrue(error <= tolerance * scale,
            label + " mismatch: expected=" + expected + ", actual=" + actual + ", error=" + error);
    }

    public static final class BesselZeroBoostParityFixture {
        public FixtureMeta meta;
        public List<BesselZeroParityCase> cases = Collections.emptyList();
    }

    public static final class FixtureMeta {
        public String module;
        public String source;
    }

    public static final class BesselZeroParityCase {
        public String name;
        public String branch;
        public String family;
        public double nu;
        public int rank;
        public double expectedZero;
        public double tolerance;
    }
}