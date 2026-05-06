package com.curioloop.yum4j.math;

import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.TestFactory;

import java.util.Collections;
import java.util.List;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.DynamicTest.dynamicTest;

class HeumanLambdaBoostParityTest {

    private static final String FIXTURE_PATH = "boost/math/heuman-lambda-boost-parity-fixture.json";

    @TestFactory
    Stream<DynamicTest> heumanLambdaMatchesBoostReferenceValues() {
        HeumanLambdaBoostParityFixture fixture = ParityFixtureLoader.load(FIXTURE_PATH, HeumanLambdaBoostParityFixture.class);
        return fixture.cases.stream().map(testCase -> dynamicTest(testCase.name, () -> assertCase(testCase)));
    }

    private static void assertCase(HeumanLambdaParityCase testCase) {
        double actual = HeumanLambda.lambda(testCase.k, testCase.phi);
        assertClose(testCase.name, testCase.expected, actual, testCase.tolerance);
    }

    private static void assertClose(String label, double expected, double actual, double tolerance) {
        double scale = Math.max(1.0, Math.abs(expected));
        double error = Math.abs(actual - expected);
        assertTrue(error <= tolerance * scale,
            label + " mismatch: expected=" + expected + ", actual=" + actual + ", error=" + error);
    }

    public static final class HeumanLambdaBoostParityFixture {
        public FixtureMeta meta;
        public List<HeumanLambdaParityCase> cases = Collections.emptyList();
    }

    public static final class FixtureMeta {
        public String module;
        public String source;
    }

    public static final class HeumanLambdaParityCase {
        public String name;
        public String branch;
        public double k;
        public double phi;
        public double expected;
        public double tolerance;
    }
}