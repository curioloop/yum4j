package com.curioloop.yum4j.math;

import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.TestFactory;

import java.util.Collections;
import java.util.List;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.DynamicTest.dynamicTest;

class SphericalBesselBoostParityTest {

    private static final String FIXTURE_PATH = "boost/math/spherical-bessel-boost-parity-fixture.json";

    @TestFactory
    Stream<DynamicTest> sphericalBesselSurfaceMatchesBoostReferenceValues() {
        SphericalBesselBoostParityFixture fixture = ParityFixtureLoader.load(FIXTURE_PATH, SphericalBesselBoostParityFixture.class);
        return fixture.cases.stream().map(testCase -> dynamicTest(testCase.name, () -> assertCase(testCase)));
    }

    private static void assertCase(SphericalBesselParityCase testCase) {
        assertClose(testCase.name + " sph j", testCase.expectedSphBessel, Bessel.sphBessel(testCase.n, testCase.x), testCase.tolerance);
        assertClose(testCase.name + " sph y", testCase.expectedSphNeumann, Bessel.sphNeumann(testCase.n, testCase.x), testCase.tolerance);
        assertClose(testCase.name + " sph j'", testCase.expectedSphBesselPrime, Bessel.sphBesselPrime(testCase.n, testCase.x), testCase.tolerance);
        assertClose(testCase.name + " sph y'", testCase.expectedSphNeumannPrime, Bessel.sphNeumannPrime(testCase.n, testCase.x), testCase.tolerance);
    }

    private static void assertClose(String label, double expected, double actual, double tolerance) {
        double scale = Math.max(1.0, Math.abs(expected));
        double error = Math.abs(actual - expected);
        assertTrue(error <= tolerance * scale,
            label + " mismatch: expected=" + expected + ", actual=" + actual + ", error=" + error);
    }

    public static final class SphericalBesselBoostParityFixture {
        public FixtureMeta meta;
        public List<SphericalBesselParityCase> cases = Collections.emptyList();
    }

    public static final class FixtureMeta {
        public String module;
        public String source;
    }

    public static final class SphericalBesselParityCase {
        public String name;
        public String branch;
        public int n;
        public double x;
        public double expectedSphBessel;
        public double expectedSphNeumann;
        public double expectedSphBesselPrime;
        public double expectedSphNeumannPrime;
        public double tolerance;
    }
}