package com.curioloop.yum4j.math;

import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.TestFactory;

import java.util.Collections;
import java.util.List;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.DynamicTest.dynamicTest;

class JacobiThetaBoostParityTest {

    private static final String FIXTURE_PATH = "boost/math/jacobi-theta-boost-parity-fixture.json";

    @TestFactory
    Stream<DynamicTest> jacobiThetaSurfaceMatchesBoostReferenceValues() {
        JacobiThetaBoostParityFixture fixture = ParityFixtureLoader.load(FIXTURE_PATH, JacobiThetaBoostParityFixture.class);
        return fixture.cases.stream().map(testCase -> dynamicTest(testCase.name, () -> assertCase(testCase)));
    }

    private static void assertCase(JacobiThetaParityCase testCase) {
        double actual;
        switch (testCase.function) {
            case "theta1" -> actual = JacobiTheta.theta1(testCase.z, requireQ(testCase));
            case "theta2" -> actual = JacobiTheta.theta2(testCase.z, requireQ(testCase));
            case "theta3" -> actual = JacobiTheta.theta3(testCase.z, requireQ(testCase));
            case "theta4" -> actual = JacobiTheta.theta4(testCase.z, requireQ(testCase));
            case "theta1Tau" -> actual = JacobiTheta.theta1Tau(testCase.z, requireTau(testCase));
            case "theta2Tau" -> actual = JacobiTheta.theta2Tau(testCase.z, requireTau(testCase));
            case "theta3Tau" -> actual = JacobiTheta.theta3Tau(testCase.z, requireTau(testCase));
            case "theta4Tau" -> actual = JacobiTheta.theta4Tau(testCase.z, requireTau(testCase));
            case "theta3MinusOne" -> actual = JacobiTheta.theta3MinusOne(testCase.z, requireQ(testCase));
            case "theta4MinusOne" -> actual = JacobiTheta.theta4MinusOne(testCase.z, requireQ(testCase));
            case "theta3MinusOneTau" -> actual = JacobiTheta.theta3MinusOneTau(testCase.z, requireTau(testCase));
            case "theta4MinusOneTau" -> actual = JacobiTheta.theta4MinusOneTau(testCase.z, requireTau(testCase));
            default -> throw new IllegalArgumentException("Unsupported Jacobi theta fixture case: " + testCase.function);
        }

        assertClose(testCase.name, testCase.expected, actual, testCase.tolerance);
    }

    private static double requireQ(JacobiThetaParityCase testCase) {
        if (testCase.q == null) {
            throw new IllegalArgumentException("Fixture case is missing q: " + testCase.name);
        }
        return testCase.q;
    }

    private static double requireTau(JacobiThetaParityCase testCase) {
        if (testCase.tau == null) {
            throw new IllegalArgumentException("Fixture case is missing tau: " + testCase.name);
        }
        return testCase.tau;
    }

    private static void assertClose(String label, double expected, double actual, double tolerance) {
        double scale = Math.max(1.0, Math.abs(expected));
        double error = Math.abs(actual - expected);
        assertTrue(error <= tolerance * scale,
            label + " mismatch: expected=" + expected + ", actual=" + actual + ", error=" + error);
    }

    public static final class JacobiThetaBoostParityFixture {
        public FixtureMeta meta;
        public List<JacobiThetaParityCase> cases = Collections.emptyList();
    }

    public static final class FixtureMeta {
        public String module;
        public String source;
    }

    public static final class JacobiThetaParityCase {
        public String name;
        public String function;
        public double z;
        public Double q;
        public Double tau;
        public double expected;
        public double tolerance;
    }
}
