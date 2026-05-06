package com.curioloop.yum4j.math;

import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestFactory;

import java.util.Collections;
import java.util.List;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.DynamicTest.dynamicTest;

class GammaBoostParityTest {

    private static final String FIXTURE_PATH = "boost/math/gamma-boost-parity-fixture.json";

    @TestFactory
    Stream<DynamicTest> gammaFamilyMatchesBoostReferenceValuesAcrossEvaluationBranches() {
        GammaBoostParityFixture fixture = ParityFixtureLoader.load(FIXTURE_PATH, GammaBoostParityFixture.class);
        return fixture.cases.stream().map(testCase -> dynamicTest(testCase.name, () -> assertCase(testCase)));
    }

    @Test
    void derivativeMatchesBoostBoundarySemanticsAtZero() {
        assertTrue(Double.isInfinite(Gamma.gammaPDerivative(0.75, 0.0)));
        assertEquals(1.0, Gamma.gammaPDerivative(1.0, 0.0), 0.0);
        assertEquals(0.0, Gamma.gammaPDerivative(2.0, 0.0), 0.0);
    }

    @TestFactory
    Stream<DynamicTest> inverseRegularizedGammaMatchesBoostReferencePoints() {
        GammaBoostParityFixture fixture = ParityFixtureLoader.load(FIXTURE_PATH, GammaBoostParityFixture.class);
        return fixture.cases.stream()
            .filter(testCase -> testCase.inverseXFromP != null || testCase.inverseXFromQ != null)
            .flatMap(testCase -> Stream.of(
                testCase.inverseXFromP == null ? null : dynamicTest(testCase.name + " P inverse", () ->
                    assertInverseClose(
                        testCase.name + " P inverse",
                        testCase.a,
                        testCase.inverseXFromP,
                        testCase.expectedP,
                        testCase.expectedDerivative,
                        false,
                        Gamma.gammaPInv(testCase.a, testCase.expectedP)
                    )
                ),
                testCase.inverseXFromQ == null ? null : dynamicTest(testCase.name + " Q inverse", () ->
                    assertInverseClose(
                        testCase.name + " Q inverse",
                        testCase.a,
                        testCase.inverseXFromQ,
                        testCase.expectedQ,
                        testCase.expectedDerivative,
                        true,
                        Gamma.gammaQInv(testCase.a, testCase.expectedQ)
                    )
                )
            ))
            .filter(dynamicTest -> dynamicTest != null);
    }

    @TestFactory
    Stream<DynamicTest> inverseRegularizedGammaOnShapeMatchesBoostReferencePoints() {
        GammaBoostParityFixture fixture = ParityFixtureLoader.load(FIXTURE_PATH, GammaBoostParityFixture.class);
        return fixture.cases.stream()
            .filter(testCase -> testCase.inverseAFromP != null || testCase.inverseAFromQ != null)
            .flatMap(testCase -> Stream.of(
                testCase.inverseAFromP == null ? null : dynamicTest(testCase.name + " P inverse-on-a", () ->
                    assertInverseShapeClose(
                        testCase.name + " P inverse-on-a",
                        testCase.x,
                        testCase.inverseAFromP,
                        testCase.expectedP,
                        false,
                        Gamma.gammaPInva(testCase.x, testCase.expectedP)
                    )
                ),
                testCase.inverseAFromQ == null ? null : dynamicTest(testCase.name + " Q inverse-on-a", () ->
                    assertInverseShapeClose(
                        testCase.name + " Q inverse-on-a",
                        testCase.x,
                        testCase.inverseAFromQ,
                        testCase.expectedQ,
                        true,
                        Gamma.gammaQInva(testCase.x, testCase.expectedQ)
                    )
                )
            ))
            .filter(dynamicTest -> dynamicTest != null);
    }

    @Test
    void inverseRegularizedGammaHonorsProbabilityBoundaries() {
        assertEquals(0.0, Gamma.gammaPInv(2.0, 0.0), 0.0);
        assertEquals(0.0, Gamma.gammaQInv(2.0, 1.0), 0.0);
        assertEquals(Double.POSITIVE_INFINITY, Gamma.gammaPInv(2.0, 1.0));
        assertEquals(Double.POSITIVE_INFINITY, Gamma.gammaQInv(2.0, 0.0));
    }

    @Test
    void inverseRegularizedGammaRejectsInvalidInputs() {
        assertThrows(IllegalArgumentException.class, () -> Gamma.gammaPInv(0.0, 0.5));
        assertThrows(IllegalArgumentException.class, () -> Gamma.gammaQInv(1.0, -0.1));
    }

    @Test
    void completeAndIncompleteGammaMatchBoostReferenceValues() {
        assertClose("tgamma small_series", 4.5908437119988026, Gamma.tgamma(0.2), 5.0e-15);
        assertClose("tgammaLower small_series", 3.9129462935281603, Gamma.tgammaLower(0.2, 0.4), 5.0e-15);
        assertClose("tgamma upper small_series", 0.67789741847064211, Gamma.tgamma(0.2, 0.4), 5.0e-15);
        assertClose("lgamma small_series", 1.5240638224307843, Gamma.lgamma(0.2), 5.0e-15);

        assertClose("tgamma half_integer_q", 3.3233509704478417, Gamma.tgamma(3.5), 5.0e-15);
        assertClose("tgammaLower half_integer_q", 0.73187696325676854, Gamma.tgammaLower(3.5, 2.0), 5.0e-15);
        assertClose("tgamma upper half_integer_q", 2.5914740071910738, Gamma.tgamma(3.5, 2.0), 5.0e-15);
        assertClose("lgamma half_integer_q", 1.2009736023470743, Gamma.lgamma(3.5), 5.0e-15);

        assertClose("tgamma tiny_x", 24.0, Gamma.tgamma(5.0), 5.0e-15);
        assertClose("tgammaLower tiny_x", 1.9999999998333335e-51, Gamma.tgammaLower(5.0, 1.0e-10), 5.0e-14);
        assertClose("tgamma upper tiny_x", 24.0, Gamma.tgamma(5.0, 1.0e-10), 5.0e-15);
        assertClose("lgamma tiny_x", 3.1780538303479458, Gamma.lgamma(5.0), 5.0e-15);

        assertTrue(Double.isInfinite(Gamma.tgamma(400.0)));
        assertTrue(Double.isInfinite(Gamma.tgammaLower(400.0, 400.0)));
        assertTrue(Double.isInfinite(Gamma.tgamma(400.0, 400.0)));
        assertClose("lgamma temme_center", 1994.5092334361334, Gamma.lgamma(400.0), 2.0e-13);

        assertClose("tgamma large_x_q", 2.0397882081197444e+46, Gamma.tgamma(40.0), 2.0e-14);
        assertClose("tgammaLower large_x_q", 2.0397882081197444e+46, Gamma.tgammaLower(40.0, 2000.0), 2.0e-14);
        assertClose("tgamma upper large_x_q", 0.0, Gamma.tgamma(40.0, 2000.0), 0.0);
        assertClose("lgamma large_x_q", 106.63176026064346, Gamma.lgamma(40.0), 5.0e-15);

        assertClose("tgamma cev_gamma_p", 1.2254167024651779, Gamma.tgamma(0.75), 5.0e-15);
        assertClose("tgammaLower cev_gamma_p", 1.2254166951674714, Gamma.tgammaLower(0.75, 18.0), 5.0e-15);
        assertClose("tgamma upper cev_gamma_p", 7.2977063926238379e-09, Gamma.tgamma(0.75, 18.0), 5.0e-15);
        assertClose("lgamma cev_gamma_p", 0.20328095143129535, Gamma.lgamma(0.75), 5.0e-15);
    }

    @Test
    void incompleteGammaHonorsBoostStyleBoundaries() {
        assertEquals(0.0, Gamma.tgammaLower(2.0, 0.0), 0.0);
        assertEquals(0.0, Gamma.tgamma(2.0, Double.POSITIVE_INFINITY), 0.0);
        assertEquals(Gamma.tgamma(2.0), Gamma.tgammaLower(2.0, Double.POSITIVE_INFINITY), 0.0);
        assertEquals(Gamma.tgamma(2.0), Gamma.tgamma(2.0, 0.0), 0.0);
    }

    @Test
    void gammaSurfaceRejectsInvalidInputs() {
        assertThrows(ArithmeticException.class, () -> Gamma.tgamma(0.0));
        assertThrows(ArithmeticException.class, () -> Gamma.lgamma(0.0));
        assertThrows(IllegalArgumentException.class, () -> Gamma.tgammaLower(0.0, 1.0));
        assertThrows(IllegalArgumentException.class, () -> Gamma.tgamma(1.0, -1.0));
    }

    private static void assertCase(GammaParityCase testCase) {
        assertClose(testCase.name + " P",
            testCase.expectedP,
            Gamma.gammaP(testCase.a, testCase.x),
            testCase.tolerance);
        assertClose(testCase.name + " Q",
            testCase.expectedQ,
            Gamma.gammaQ(testCase.a, testCase.x),
            testCase.tolerance);
        assertClose(testCase.name + " derivative",
            testCase.expectedDerivative,
            Gamma.gammaPDerivative(testCase.a, testCase.x),
            testCase.tolerance);
    }

    private static void assertClose(String label, double expected, double actual, double tolerance) {
        double scale = Math.max(1.0, Math.abs(expected));
        double error = Math.abs(actual - expected);
        assertTrue(error <= tolerance * scale,
            label + " mismatch: expected=" + expected + ", actual=" + actual + ", error=" + error);
    }

    private static void assertInverseClose(String label,
                                           double a,
                                           double expectedX,
                                           double targetProbability,
                                           double expectedDerivative,
                                           boolean complement,
                                           double actualX) {
        double scale = Math.max(1.0, Math.abs(expectedX));
        double conditioningTolerance = Math.max(Math.ulp(targetProbability), Double.MIN_VALUE)
            / Math.max(expectedDerivative, Double.MIN_NORMAL);
        double tolerance = Math.max(1.0e-10 * scale, 2.0 * conditioningTolerance);
        double error = Math.abs(actualX - expectedX);
        assertTrue(error <= tolerance,
            label + " mismatch: expected=" + expectedX + ", actual=" + actualX + ", error=" + error
                + ", tolerance=" + tolerance);

        double roundTrip = complement ? Gamma.gammaQ(a, actualX) : Gamma.gammaP(a, actualX);
        double probabilityTolerance = Math.max(32.0 * Math.ulp(targetProbability), 2.0e-14 * Math.abs(targetProbability));
        double probabilityError = Math.abs(roundTrip - targetProbability);
        assertTrue(probabilityError <= probabilityTolerance,
            label + " roundtrip mismatch: target=" + targetProbability + ", actual=" + roundTrip
                + ", error=" + probabilityError + ", tolerance=" + probabilityTolerance);
    }

    private static void assertInverseShapeClose(String label,
                                                double x,
                                                double expectedA,
                                                double targetProbability,
                                                boolean complement,
                                                double actualA) {
        double scale = Math.max(1.0, Math.abs(expectedA));
        double tolerance = Math.max(1.0e-7 * scale, 32.0 * Math.ulp(expectedA));
        double error = Math.abs(actualA - expectedA);
        assertTrue(error <= tolerance,
            label + " mismatch: expected=" + expectedA + ", actual=" + actualA + ", error=" + error
                + ", tolerance=" + tolerance);

        double roundTrip = complement ? Gamma.gammaQ(actualA, x) : Gamma.gammaP(actualA, x);
        double probabilityTolerance = Math.max(64.0 * Math.ulp(targetProbability), 2.0e-12 * Math.max(1.0, Math.abs(targetProbability)));
        double probabilityError = Math.abs(roundTrip - targetProbability);
        assertTrue(probabilityError <= probabilityTolerance,
            label + " roundtrip mismatch: target=" + targetProbability + ", actual=" + roundTrip
                + ", error=" + probabilityError + ", tolerance=" + probabilityTolerance);
    }

    public static final class GammaBoostParityFixture {
        public FixtureMeta meta;
        public List<GammaParityCase> cases = Collections.emptyList();
    }

    public static final class FixtureMeta {
        public String module;
        public String source;
    }

    public static final class GammaParityCase {
        public String name;
        public String branch;
        public double a;
        public double x;
        public double expectedP;
        public double expectedQ;
        public double expectedDerivative;
        public Double inverseXFromP;
        public Double inverseXFromQ;
        public Double inverseAFromP;
        public Double inverseAFromQ;
        public double tolerance;
    }
}
