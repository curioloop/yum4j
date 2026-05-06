package com.curioloop.yum4j.math;

import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.TestFactory;

import java.util.Collections;
import java.util.List;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.DynamicTest.dynamicTest;

class HypergeometricBoostParityTest {

    private static final String FIXTURE_PATH = "boost/math/hypergeometric-boost-parity-fixture.json";

    @TestFactory
    Stream<DynamicTest> hypergeometricSurfaceMatchesBoostReferenceValues() {
        HypergeometricBoostParityFixture fixture = ParityFixtureLoader.load(FIXTURE_PATH, HypergeometricBoostParityFixture.class);
        return fixture.cases.stream().map(testCase -> dynamicTest(testCase.name, () -> assertCase(testCase)));
    }

    private static void assertCase(HypergeometricParityCase testCase) {
        double actual;
        Integer actualSign = null;
        switch (testCase.function) {
            case "oneFZero" -> actual = Hypergeometric.oneFZero(testCase.a, testCase.z);
            case "zeroFOne" -> actual = Hypergeometric.zeroFOne(requireB(testCase), testCase.z);
            case "oneFOne" -> actual = Hypergeometric.oneFOne(testCase.a, requireB(testCase), testCase.z);
            case "oneFOneRegularized" -> actual = Hypergeometric.oneFOneRegularized(testCase.a, requireB(testCase), testCase.z);
            case "logOneFOne" -> actual = Hypergeometric.logOneFOne(testCase.a, requireB(testCase), testCase.z);
            case "logOneFOneSigned" -> {
                DoubleI signed = Hypergeometric.logOneFOneSigned(testCase.a, requireB(testCase), testCase.z);
                actual = signed.value();
                actualSign = signed.sign();
            }
            case "twoFZero" -> actual = Hypergeometric.twoFZero(testCase.a, requireB(testCase), testCase.z);
            default -> throw new IllegalArgumentException("Unsupported hypergeometric fixture case: " + testCase.function);
        }

        assertClose(testCase.name, testCase.expected, actual, testCase.tolerance);
        if (testCase.expectedSign != null) {
            assertTrue(actualSign != null && actualSign.equals(testCase.expectedSign),
                testCase.name + " sign mismatch: expected=" + testCase.expectedSign + ", actual=" + actualSign);
        }
    }

    private static double requireB(HypergeometricParityCase testCase) {
        if (testCase.b == null) {
            throw new IllegalArgumentException("Fixture case is missing b: " + testCase.name);
        }
        return testCase.b;
    }

    private static void assertClose(String label, double expected, double actual, double tolerance) {
        double scale = Math.max(1.0, Math.abs(expected));
        double error = Math.abs(actual - expected);
        assertTrue(error <= tolerance * scale,
            label + " mismatch: expected=" + expected + ", actual=" + actual + ", error=" + error);
    }

    public static final class HypergeometricBoostParityFixture {
        public FixtureMeta meta;
        public List<HypergeometricParityCase> cases = Collections.emptyList();
    }

    public static final class FixtureMeta {
        public String module;
        public String source;
    }

    public static final class HypergeometricParityCase {
        public String name;
        public String function;
        public double a;
        public Double b;
        public Double c;
        public double z;
        public double expected;
        public double tolerance;
        public Integer expectedSign;
    }
}