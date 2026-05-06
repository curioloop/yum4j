package com.curioloop.yum4j.math;

import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.TestFactory;

import java.util.Collections;
import java.util.List;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.DynamicTest.dynamicTest;

class JacobiEllipticBoostParityTest {

    private static final String FIXTURE_PATH = "boost/math/jacobi-elliptic-boost-parity-fixture.json";

    @TestFactory
    Stream<DynamicTest> jacobiEllipticSurfaceMatchesBoostReferenceValues() {
        JacobiEllipticBoostParityFixture fixture = ParityFixtureLoader.load(FIXTURE_PATH, JacobiEllipticBoostParityFixture.class);
        return fixture.cases.stream().map(testCase -> dynamicTest(testCase.name, () -> assertCase(testCase)));
    }

    private static void assertCase(JacobiEllipticParityCase testCase) {
        double actual;
        switch (testCase.function) {
            case "sn" -> actual = JacobiElliptic.sn(testCase.k, testCase.theta);
            case "cn" -> actual = JacobiElliptic.cn(testCase.k, testCase.theta);
            case "dn" -> actual = JacobiElliptic.dn(testCase.k, testCase.theta);
            case "cd" -> actual = JacobiElliptic.cd(testCase.k, testCase.theta);
            case "dc" -> actual = JacobiElliptic.dc(testCase.k, testCase.theta);
            case "ns" -> actual = JacobiElliptic.ns(testCase.k, testCase.theta);
            case "sd" -> actual = JacobiElliptic.sd(testCase.k, testCase.theta);
            case "ds" -> actual = JacobiElliptic.ds(testCase.k, testCase.theta);
            case "nc" -> actual = JacobiElliptic.nc(testCase.k, testCase.theta);
            case "nd" -> actual = JacobiElliptic.nd(testCase.k, testCase.theta);
            case "sc" -> actual = JacobiElliptic.sc(testCase.k, testCase.theta);
            case "cs" -> actual = JacobiElliptic.cs(testCase.k, testCase.theta);
            default -> throw new IllegalArgumentException("Unsupported Jacobi elliptic fixture case: " + testCase.function);
        }

        assertClose(testCase.name, testCase.expected, actual, testCase.tolerance);
    }

    private static void assertClose(String label, double expected, double actual, double tolerance) {
        double scale = Math.max(1.0, Math.abs(expected));
        double error = Math.abs(actual - expected);
        assertTrue(error <= tolerance * scale,
            label + " mismatch: expected=" + expected + ", actual=" + actual + ", error=" + error);
    }

    public static final class JacobiEllipticBoostParityFixture {
        public FixtureMeta meta;
        public List<JacobiEllipticParityCase> cases = Collections.emptyList();
    }

    public static final class FixtureMeta {
        public String module;
        public String source;
    }

    public static final class JacobiEllipticParityCase {
        public String name;
        public String function;
        public double k;
        public double theta;
        public double expected;
        public double tolerance;
    }
}