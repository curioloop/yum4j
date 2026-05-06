package com.curioloop.yum4j.math;

import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.TestFactory;

import java.util.Collections;
import java.util.List;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.DynamicTest.dynamicTest;

class ComplexStdlibGridParityTest {

    private static final String FIXTURE_PATH = "math/complex-stdlib-parity-fixture.json";

    @TestFactory
    Stream<DynamicTest> directComplexFunctionsMatchStdComplexGrid() {
        Fixture fixture = ParityFixtureLoader.load(FIXTURE_PATH, Fixture.class);
        Stream<DynamicTest> complexCases = fixture.complexCases.stream()
            .map(testCase -> dynamicTest(testCase.name, () -> assertComplexCase(testCase)));
        Stream<DynamicTest> realCases = fixture.realCases.stream()
            .map(testCase -> dynamicTest(testCase.name, () -> assertRealCase(testCase)));
        return Stream.concat(complexCases, realCases);
    }

    private static void assertComplexCase(ComplexCase testCase) {
        Complex input = new Complex(testCase.inputReal, testCase.inputImag);
        Complex actual = switch (testCase.operation) {
            case "sin" -> input.sin();
            case "cos" -> input.cos();
            case "tan" -> input.tan();
            case "sinh" -> input.sinh();
            case "cosh" -> input.cosh();
            case "tanh" -> input.tanh();
            case "exp" -> input.exp();
            case "log" -> input.log();
            case "sqrt" -> input.sqrt();
            case "recip" -> input.recip();
            case "div" -> input.div(new Complex(testCase.otherReal, testCase.otherImag));
            default -> throw new IllegalArgumentException("Unsupported complex operation: " + testCase.operation);
        };

        assertDoubleClose(
            testCase.name + " real",
            testCase.expectedReal,
            actual.real(),
            testCase.relativeTolerance,
            testCase.absoluteTolerance
        );
        assertDoubleClose(
            testCase.name + " imag",
            testCase.expectedImag,
            actual.imag(),
            testCase.relativeTolerance,
            testCase.absoluteTolerance
        );
    }

    private static void assertRealCase(RealCase testCase) {
        Complex input = new Complex(testCase.inputReal, testCase.inputImag);
        double actual = switch (testCase.operation) {
            case "arg" -> input.arg();
            default -> throw new IllegalArgumentException("Unsupported real operation: " + testCase.operation);
        };

        assertDoubleClose(
            testCase.name,
            testCase.expected,
            actual,
            testCase.relativeTolerance,
            testCase.absoluteTolerance
        );
    }

    private static void assertDoubleClose(String label,
                                          double expected,
                                          double actual,
                                          double relativeTolerance,
                                          double absoluteTolerance) {
        if (Double.isNaN(expected)) {
            assertTrue(Double.isNaN(actual), label + " mismatch: expected NaN, actual=" + actual);
            return;
        }
        if (Double.isInfinite(expected)) {
            assertEquals(expected, actual, 0.0, label + " infinite mismatch");
            return;
        }
        if (expected == 0.0 && actual == 0.0) {
            assertEquals(
                Double.doubleToRawLongBits(expected),
                Double.doubleToRawLongBits(actual),
                label + " signed zero mismatch"
            );
            return;
        }

        double tolerance = Math.max(absoluteTolerance, Math.abs(expected) * relativeTolerance);
        assertTrue(
            Math.abs(expected - actual) <= tolerance,
            label + " mismatch: expected=" + expected + ", actual=" + actual + ", tolerance=" + tolerance
        );
    }

    public static final class Fixture {
        public Meta meta;
        public List<ComplexCase> complexCases = Collections.emptyList();
        public List<RealCase> realCases = Collections.emptyList();
    }

    public static final class Meta {
        public String module;
        public String source;
        public String compiler;
        public String generatedAt;
    }

    public static final class ComplexCase {
        public String name;
        public String operation;
        public double inputReal;
        public double inputImag;
        public Double otherReal;
        public Double otherImag;
        public double expectedReal;
        public double expectedImag;
        public double relativeTolerance;
        public double absoluteTolerance;
    }

    public static final class RealCase {
        public String name;
        public String operation;
        public double inputReal;
        public double inputImag;
        public double expected;
        public double relativeTolerance;
        public double absoluteTolerance;
    }
}