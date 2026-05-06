package com.curioloop.yum4j.math;

import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.TestFactory;

import java.util.Collections;
import java.util.List;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.DynamicTest.dynamicTest;

class AiryBoostParityTest {

    private static final String FIXTURE_PATH = "boost/math/airy-boost-parity-fixture.json";

    @TestFactory
    Stream<DynamicTest> airySurfaceMatchesBoostReferenceValues() {
        AiryBoostParityFixture fixture = ParityFixtureLoader.load(FIXTURE_PATH, AiryBoostParityFixture.class);
        return fixture.cases.stream().map(testCase -> dynamicTest(testCase.name, () -> assertCase(testCase)));
    }

    private static void assertCase(AiryParityCase testCase) {
        double actual;
        switch (testCase.function) {
            case "ai" -> actual = Airy.ai(testCase.x);
            case "bi" -> actual = Airy.bi(testCase.x);
            case "aiPrime" -> actual = Airy.aiPrime(testCase.x);
            case "biPrime" -> actual = Airy.biPrime(testCase.x);
            case "aiZero" -> actual = Airy.aiZero(testCase.rank);
            case "biZero" -> actual = Airy.biZero(testCase.rank);
            default -> throw new IllegalArgumentException("Unsupported Airy fixture case: " + testCase.function);
        }

        assertClose(testCase.name, testCase.expected, actual, testCase.tolerance);
    }

    private static void assertClose(String label, double expected, double actual, double tolerance) {
        double scale = Math.max(1.0, Math.abs(expected));
        double error = Math.abs(actual - expected);
        assertTrue(error <= tolerance * scale,
            label + " mismatch: expected=" + expected + ", actual=" + actual + ", error=" + error);
    }

    public static final class AiryBoostParityFixture {
        public FixtureMeta meta;
        public List<AiryParityCase> cases = Collections.emptyList();
    }

    public static final class FixtureMeta {
        public String module;
        public String source;
    }

    public static final class AiryParityCase {
        public String name;
        public String function;
        public int rank;
        public double x;
        public double expected;
        public double tolerance;
    }
}