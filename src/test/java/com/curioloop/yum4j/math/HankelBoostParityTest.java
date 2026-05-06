package com.curioloop.yum4j.math;

import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.TestFactory;

import java.util.Collections;
import java.util.List;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.DynamicTest.dynamicTest;

class HankelBoostParityTest {

    private static final String FIXTURE_PATH = "boost/math/hankel-boost-parity-fixture.json";

    @TestFactory
    Stream<DynamicTest> hankelSurfaceMatchesBoostReferenceValues() {
        HankelBoostParityFixture fixture = ParityFixtureLoader.load(FIXTURE_PATH, HankelBoostParityFixture.class);
        return fixture.cases.stream().map(testCase -> dynamicTest(testCase.name, () -> assertCase(testCase)));
    }

    private static void assertCase(HankelParityCase testCase) {
        Complex actual = switch (testCase.function) {
            case "cylHankel1" -> Hankel.cylHankel1(testCase.nu, testCase.x);
            case "cylHankel2" -> Hankel.cylHankel2(testCase.nu, testCase.x);
            case "sphHankel1" -> Hankel.sphHankel1(testCase.nu, testCase.x);
            case "sphHankel2" -> Hankel.sphHankel2(testCase.nu, testCase.x);
            default -> throw new IllegalArgumentException("Unsupported Hankel function: " + testCase.function);
        };
        assertClose(testCase.name + " real", testCase.expectedReal, actual.real(), testCase.tolerance);
        assertClose(testCase.name + " imag", testCase.expectedImag, actual.imag(), testCase.tolerance);
    }

    private static void assertClose(String label, double expected, double actual, double tolerance) {
        double scale = Math.max(1.0, Math.abs(expected));
        double error = Math.abs(actual - expected);
        assertTrue(error <= tolerance * scale,
            label + " mismatch: expected=" + expected + ", actual=" + actual + ", error=" + error);
    }

    public static final class HankelBoostParityFixture {
        public FixtureMeta meta;
        public List<HankelParityCase> cases = Collections.emptyList();
    }

    public static final class FixtureMeta {
        public String module;
        public String source;
    }

    public static final class HankelParityCase {
        public String name;
        public String function;
        public String branch;
        public double nu;
        public double x;
        public double expectedReal;
        public double expectedImag;
        public double tolerance;
    }
}