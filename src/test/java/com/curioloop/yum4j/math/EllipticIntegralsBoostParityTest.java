package com.curioloop.yum4j.math;

import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.TestFactory;

import java.util.Collections;
import java.util.List;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.DynamicTest.dynamicTest;

class EllipticIntegralsBoostParityTest {

    private static final String FIXTURE_PATH = "boost/math/elliptic-integrals-boost-parity-fixture.json";

    @TestFactory
    Stream<DynamicTest> ellipticIntegralSurfaceMatchesBoostReferenceValues() {
        EllipticIntegralsBoostParityFixture fixture = ParityFixtureLoader.load(
            FIXTURE_PATH,
            EllipticIntegralsBoostParityFixture.class
        );
        return fixture.cases.stream().map(testCase -> dynamicTest(testCase.name, () -> assertCase(testCase)));
    }

    private static void assertCase(EllipticIntegralParityCase testCase) {
        double actual;
        switch (testCase.function) {
            case "completeFirstKind" -> actual = EllipticIntegrals.completeFirstKind(testCase.k);
            case "incompleteFirstKind" -> actual = EllipticIntegrals.incompleteFirstKind(testCase.k, requirePhi(testCase));
            case "completeSecondKind" -> actual = EllipticIntegrals.completeSecondKind(testCase.k);
            case "incompleteSecondKind" -> actual = EllipticIntegrals.incompleteSecondKind(testCase.k, requirePhi(testCase));
            case "completeThirdKind" -> actual = EllipticIntegrals.completeThirdKind(testCase.k, requireNu(testCase));
            case "incompleteThirdKind" -> actual = EllipticIntegrals.incompleteThirdKind(testCase.k, requireNu(testCase), requirePhi(testCase));
            default -> throw new IllegalArgumentException("Unsupported elliptic-integral fixture case: " + testCase.function);
        }

        assertClose(testCase.name, testCase.expected, actual, testCase.tolerance);
    }

    private static double requirePhi(EllipticIntegralParityCase testCase) {
        if (testCase.phi == null) {
            throw new IllegalArgumentException("Fixture case is missing phi: " + testCase.name);
        }
        return testCase.phi;
    }

    private static double requireNu(EllipticIntegralParityCase testCase) {
        if (testCase.nu == null) {
            throw new IllegalArgumentException("Fixture case is missing nu: " + testCase.name);
        }
        return testCase.nu;
    }

    private static void assertClose(String label, double expected, double actual, double tolerance) {
        double scale = Math.max(1.0, Math.abs(expected));
        double error = Math.abs(actual - expected);
        assertTrue(error <= tolerance * scale,
            label + " mismatch: expected=" + expected + ", actual=" + actual + ", error=" + error);
    }

    public static final class EllipticIntegralsBoostParityFixture {
        public FixtureMeta meta;
        public List<EllipticIntegralParityCase> cases = Collections.emptyList();
    }

    public static final class FixtureMeta {
        public String module;
        public String source;
    }

    public static final class EllipticIntegralParityCase {
        public String name;
        public String function;
        public double k;
        public Double nu;
        public Double phi;
        public double expected;
        public double tolerance;
    }
}