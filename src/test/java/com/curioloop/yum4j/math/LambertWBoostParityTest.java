package com.curioloop.yum4j.math;

import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.TestFactory;

import java.util.Collections;
import java.util.List;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.DynamicTest.dynamicTest;

class LambertWBoostParityTest {

    private static final String FIXTURE_PATH = "boost/math/lambert-w-boost-parity-fixture.json";

    @TestFactory
    Stream<DynamicTest> lambertWSurfaceMatchesBoostReferenceValues() {
        LambertWBoostParityFixture fixture = ParityFixtureLoader.load(FIXTURE_PATH, LambertWBoostParityFixture.class);
        return fixture.cases.stream().map(testCase -> dynamicTest(testCase.name, () -> assertCase(testCase)));
    }

    private static void assertCase(LambertWParityCase testCase) {
        double actual;
        switch (testCase.function) {
            case "w0" -> actual = LambertW.w0(testCase.z);
            case "wm1" -> actual = LambertW.wm1(testCase.z);
            case "w0Prime" -> actual = LambertW.w0Prime(testCase.z);
            case "wm1Prime" -> actual = LambertW.wm1Prime(testCase.z);
            default -> throw new IllegalArgumentException("Unsupported Lambert W fixture case: " + testCase.function);
        }

        assertClose(testCase.name, testCase.expected, actual, testCase.tolerance);
    }

    private static void assertClose(String label, double expected, double actual, double tolerance) {
        double scale = Math.max(1.0, Math.abs(expected));
        double error = Math.abs(actual - expected);
        assertTrue(error <= tolerance * scale,
            label + " mismatch: expected=" + expected + ", actual=" + actual + ", error=" + error);
    }

    public static final class LambertWBoostParityFixture {
        public FixtureMeta meta;
        public List<LambertWParityCase> cases = Collections.emptyList();
    }

    public static final class FixtureMeta {
        public String module;
        public String source;
    }

    public static final class LambertWParityCase {
        public String name;
        public String function;
        public double z;
        public double expected;
        public double tolerance;
    }
}