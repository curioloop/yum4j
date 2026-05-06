package com.curioloop.yum4j.math;

import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.TestFactory;

import java.util.Collections;
import java.util.List;
import java.util.stream.Stream;

import static org.junit.jupiter.api.DynamicTest.dynamicTest;
import static org.junit.jupiter.api.Assertions.assertTrue;

class BetaBoostParityTest {

    private static final String FIXTURE_PATH = "boost/math/beta-boost-parity-fixture.json";

    @TestFactory
    Stream<DynamicTest> boostReferenceValuesMatchAcrossBetaBranches() {
        BetaBoostParityFixture fixture = ParityFixtureLoader.load(FIXTURE_PATH, BetaBoostParityFixture.class);
        return fixture.cases.stream().map(testCase -> dynamicTest(testCase.name, () -> assertCase(testCase)));
    }

    private static void assertCase(BetaParityCase testCase) {
        assertClose(testCase.name + " complete beta",
            testCase.completeBeta,
            Beta.beta(testCase.a, testCase.b),
            testCase.tolerance);
        assertClose(testCase.name + " lower incomplete beta",
            testCase.lowerBeta,
            Beta.beta(testCase.a, testCase.b, testCase.x),
            testCase.tolerance);
        assertClose(testCase.name + " upper incomplete beta",
            testCase.upperBeta,
            Beta.betac(testCase.a, testCase.b, testCase.x),
            testCase.tolerance);
        assertClose(testCase.name + " regularized lower",
            testCase.regularizedLower,
            Beta.ibeta(testCase.a, testCase.b, testCase.x),
            testCase.tolerance);
        assertClose(testCase.name + " regularized upper",
            testCase.regularizedUpper,
            Beta.ibetac(testCase.a, testCase.b, testCase.x),
            testCase.tolerance);
        assertClose(testCase.name + " derivative",
            testCase.derivative,
            Beta.ibetaDerivative(testCase.a, testCase.b, testCase.x),
            testCase.tolerance);

        assertClose(testCase.name + " ibetaInv",
            testCase.inverseXFromP,
            Beta.ibetaInv(testCase.a, testCase.b, testCase.regularizedLower),
            testCase.inverseXTolerance);
        Double2 inverseFromP = Beta.ibetaInvComplement(testCase.a, testCase.b, testCase.regularizedLower);
        assertClose(testCase.name + " ibetaInv pair x",
            testCase.inverseXFromP,
            inverseFromP._1(),
            testCase.inverseXTolerance);
        assertClose(testCase.name + " ibetaInv pair y",
            testCase.inverseYFromP,
            inverseFromP._2(),
            testCase.inverseXTolerance);
        assertClose(testCase.name + " ibetacInv",
            testCase.inverseXFromQ,
            Beta.ibetacInv(testCase.a, testCase.b, testCase.regularizedUpper),
            testCase.inverseXTolerance);
        Double2 inverseFromQ = Beta.ibetacInvComplement(testCase.a, testCase.b, testCase.regularizedUpper);
        assertClose(testCase.name + " ibetacInv pair x",
            testCase.inverseXFromQ,
            inverseFromQ._1(),
            testCase.inverseXTolerance);
        assertClose(testCase.name + " ibetacInv pair y",
            testCase.inverseYFromQ,
            inverseFromQ._2(),
            testCase.inverseXTolerance);

        assertClose(testCase.name + " ibetaInva",
            testCase.inverseAFromP,
            Beta.ibetaInva(testCase.b, testCase.x, testCase.regularizedLower),
            testCase.inverseParameterTolerance);
        assertClose(testCase.name + " ibetacInva",
            testCase.inverseAFromQ,
            Beta.ibetacInva(testCase.b, testCase.x, testCase.regularizedUpper),
            testCase.inverseParameterTolerance);
        assertClose(testCase.name + " ibetaInvb",
            testCase.inverseBFromP,
            Beta.ibetaInvb(testCase.a, testCase.x, testCase.regularizedLower),
            testCase.inverseParameterTolerance);
        assertClose(testCase.name + " ibetacInvb",
            testCase.inverseBFromQ,
            Beta.ibetacInvb(testCase.a, testCase.x, testCase.regularizedUpper),
            testCase.inverseParameterTolerance);

        assertClose(testCase.name + " complete partition",
            testCase.completeBeta,
            Beta.beta(testCase.a, testCase.b, testCase.x) + Beta.betac(testCase.a, testCase.b, testCase.x),
            testCase.tolerance);
        assertClose(testCase.name + " regularized partition",
            1.0,
            Beta.ibeta(testCase.a, testCase.b, testCase.x) + Beta.ibetac(testCase.a, testCase.b, testCase.x),
            testCase.tolerance);
    }

    private static void assertClose(String label, double expected, double actual, double tolerance) {
        double scale = Math.max(1.0, Math.abs(expected));
        double error = Math.abs(actual - expected);
        assertTrue(error <= tolerance * scale,
            label + " mismatch: expected=" + expected + ", actual=" + actual + ", error=" + error);
    }

    public static final class BetaBoostParityFixture {
        public FixtureMeta meta;
        public List<BetaParityCase> cases = Collections.emptyList();
    }

    public static final class FixtureMeta {
        public String module;
        public String source;
    }

    public static final class BetaParityCase {
        public String name;
        public String branch;
        public double a;
        public double b;
        public double x;
        public double completeBeta;
        public double lowerBeta;
        public double upperBeta;
        public double regularizedLower;
        public double regularizedUpper;
        public double derivative;
        public double inverseXFromP;
        public double inverseYFromP;
        public double inverseXFromQ;
        public double inverseYFromQ;
        public double inverseAFromP;
        public double inverseAFromQ;
        public double inverseBFromP;
        public double inverseBFromQ;
        public double tolerance;
        public double inverseXTolerance;
        public double inverseParameterTolerance;
    }
}