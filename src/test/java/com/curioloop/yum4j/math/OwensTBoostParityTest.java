package com.curioloop.yum4j.math;

import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.TestFactory;

import java.util.Collections;
import java.util.List;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.DynamicTest.dynamicTest;

class OwensTBoostParityTest {

    private static final String FIXTURE_PATH = "boost/math/owens-t-boost-parity-fixture.json";

    @TestFactory
    Stream<DynamicTest> owensTSurfaceMatchesBoostReferenceValues() {
        OwensTBoostParityFixture fixture = ParityFixtureLoader.load(FIXTURE_PATH, OwensTBoostParityFixture.class);
        return fixture.cases.stream().map(testCase -> dynamicTest(testCase.name, () -> assertCase(testCase)));
    }

    private static void assertCase(OwensTParityCase testCase) {
        assertClose(testCase.name, testCase.expected, OwensT.value(testCase.h, testCase.a), testCase.tolerance);
    }

    private static void assertClose(String label, double expected, double actual, double tolerance) {
        double scale = Math.max(1.0, Math.abs(expected));
        double error = Math.abs(actual - expected);
        assertTrue(error <= tolerance * scale,
            label + " mismatch: expected=" + expected + ", actual=" + actual + ", error=" + error);
    }

    public static final class OwensTBoostParityFixture {
        public FixtureMeta meta;
        public List<OwensTParityCase> cases = Collections.emptyList();
    }

    public static final class FixtureMeta {
        public String module;
        public String source;
    }

    public static final class OwensTParityCase {
        public String name;
        public double h;
        public double a;
        public double expected;
        public double tolerance;
    }
}