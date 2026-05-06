package com.curioloop.yum4j.math;

import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.TestFactory;

import java.util.Collections;
import java.util.List;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.DynamicTest.dynamicTest;

class BesselBoostParityTest {

    private static final String FIXTURE_PATH = "boost/math/bessel-boost-parity-fixture.json";

    @TestFactory
    Stream<DynamicTest> realBesselSurfaceMatchesBoostReferenceValues() {
        BesselBoostParityFixture fixture = ParityFixtureLoader.load(FIXTURE_PATH, BesselBoostParityFixture.class);
        return fixture.cases.stream().map(testCase -> dynamicTest(testCase.name, () -> assertCase(testCase)));
    }

    private static void assertCase(BesselParityCase testCase) {
        assertClose(testCase.name + " J", testCase.expectedJ, Bessel.j(testCase.nu, testCase.x), testCase.tolerance);
        assertClose(testCase.name + " Y", testCase.expectedY, Bessel.y(testCase.nu, testCase.x), testCase.tolerance);
        assertClose(testCase.name + " I", testCase.expectedI, Bessel.i(testCase.nu, testCase.x), testCase.tolerance);
        assertClose(testCase.name + " K", testCase.expectedK, Bessel.k(testCase.nu, testCase.x), testCase.tolerance);
        assertClose(testCase.name + " weighted I", testCase.expectedWeightedI, Bessel.iExponentiallyWeighted(testCase.nu, testCase.x), testCase.tolerance);
        assertClose(testCase.name + " weighted K", testCase.expectedWeightedK, Bessel.kExponentiallyWeighted(testCase.nu, testCase.x), testCase.tolerance);
    }

    private static void assertClose(String label, double expected, double actual, double tolerance) {
        double scale = Math.max(1.0, Math.abs(expected));
        double error = Math.abs(actual - expected);
        assertTrue(error <= tolerance * scale,
            label + " mismatch: expected=" + expected + ", actual=" + actual + ", error=" + error);
    }

    public static final class BesselBoostParityFixture {
        public FixtureMeta meta;
        public List<BesselParityCase> cases = Collections.emptyList();
    }

    public static final class FixtureMeta {
        public String module;
        public String source;
    }

    public static final class BesselParityCase {
        public String name;
        public String branch;
        public double nu;
        public double x;
        public double expectedJ;
        public double expectedY;
        public double expectedI;
        public double expectedK;
        public double expectedWeightedI;
        public double expectedWeightedK;
        public double tolerance;
    }
}