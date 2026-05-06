package com.curioloop.yum4j.math;

import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.TestFactory;

import java.util.Collections;
import java.util.List;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.DynamicTest.dynamicTest;

class BesselPrimeBoostParityTest {

    private static final String FIXTURE_PATH = "boost/math/bessel-prime-boost-parity-fixture.json";

    @TestFactory
    Stream<DynamicTest> realBesselPrimeSurfaceMatchesBoostReferenceValues() {
        BesselPrimeBoostParityFixture fixture = ParityFixtureLoader.load(FIXTURE_PATH, BesselPrimeBoostParityFixture.class);
        return fixture.cases.stream().map(testCase -> dynamicTest(testCase.name, () -> assertCase(testCase)));
    }

    private static void assertCase(BesselPrimeParityCase testCase) {
        assertClose(testCase.name + " J'", testCase.expectedJPrime, Bessel.jPrime(testCase.nu, testCase.x), testCase.tolerance);
        assertClose(testCase.name + " Y'", testCase.expectedYPrime, Bessel.yPrime(testCase.nu, testCase.x), testCase.tolerance);
        assertClose(testCase.name + " I'", testCase.expectedIPrime, Bessel.iPrime(testCase.nu, testCase.x), testCase.tolerance);
        assertClose(testCase.name + " K'", testCase.expectedKPrime, Bessel.kPrime(testCase.nu, testCase.x), testCase.tolerance);
    }

    private static void assertClose(String label, double expected, double actual, double tolerance) {
        double scale = Math.max(1.0, Math.abs(expected));
        double error = Math.abs(actual - expected);
        assertTrue(error <= tolerance * scale,
            label + " mismatch: expected=" + expected + ", actual=" + actual + ", error=" + error);
    }

    public static final class BesselPrimeBoostParityFixture {
        public FixtureMeta meta;
        public List<BesselPrimeParityCase> cases = Collections.emptyList();
    }

    public static final class FixtureMeta {
        public String module;
        public String source;
    }

    public static final class BesselPrimeParityCase {
        public String name;
        public String branch;
        public double nu;
        public double x;
        public double expectedJPrime;
        public double expectedYPrime;
        public double expectedIPrime;
        public double expectedKPrime;
        public double tolerance;
    }
}