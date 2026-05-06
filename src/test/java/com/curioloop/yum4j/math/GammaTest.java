package com.curioloop.yum4j.math;

import org.junit.jupiter.api.Test;

import java.lang.reflect.Method;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class GammaTest {

    private static final double EULER_MASCHERONI = 0.5772156649015329;

    @Test
    void digammaMatchesKnownConstants() {
        assertEquals(-EULER_MASCHERONI, Gamma.digamma(1.0), 5.0e-15);
        assertEquals(-EULER_MASCHERONI - 2.0 * Math.log(2.0), Gamma.digamma(0.5), 5.0e-15);
    }

    @Test
    void trigammaMatchesKnownConstants() {
        assertEquals(Math.PI * Math.PI / 6.0, Gamma.trigamma(1.0), 5.0e-15);
        assertEquals(Math.PI * Math.PI / 2.0, Gamma.trigamma(0.5), 5.0e-15);
    }

    @Test
    void polygammaAliasesRemainConsistentWithOrderSpecificEntryPoints() {
        assertEquals(Gamma.digamma(2.5), Gamma.polygamma(0, 2.5), 0.0);
        assertEquals(Gamma.trigamma(2.5), Gamma.polygamma(1, 2.5), 0.0);
    }

    @Test
    void polygammaRecurrenceMatchesOrderSpecificIdentity() {
        double x = 2.75;
        for (int order = 0; order <= 5; order++) {
            double left = Gamma.polygamma(order, x + 1.0);
            double right = Gamma.polygamma(order, x) + recurrenceTerm(order, x);
            assertClose("order " + order, right, left, 5.0e-14);
        }
    }

    @Test
    void polygammaRejectsInvalidInputs() {
        assertThrows(ArithmeticException.class, () -> Gamma.digamma(0.0));
        assertThrows(ArithmeticException.class, () -> Gamma.trigamma(0.0));
        assertThrows(IllegalArgumentException.class, () -> Gamma.polygamma(-1, 1.0));
        assertThrows(ArithmeticException.class, () -> Gamma.polygamma(2, 0.0));
        assertThrows(ArithmeticException.class, () -> Gamma.polygamma(2, -1.0));
        assertThrows(ArithmeticException.class, () -> Gamma.polygamma(2, -2.0));
    }

    @Test
    void polygammaNegativeNonIntegerMatchesRecurrenceIdentity() {
        double x = -0.25;
        for (int order = 2; order <= 5; order++) {
            double left = Gamma.polygamma(order, x + 1.0);
            double right = Gamma.polygamma(order, x) + recurrenceTerm(order, x);
            assertClose("negative order " + order, right, left, 1.0e-12);
        }
    }

    @Test
    void sinPiAndCosPiPreserveExactSpecialPoints() {
        assertEquals(0.0, Commons.sinPi(1.0), 0.0);
        assertEquals(0.0, Commons.sinPi(2.0), 0.0);
        assertEquals(1.0, Commons.sinPi(0.5), 0.0);
        assertEquals(-1.0, Commons.sinPi(-0.5), 0.0);

        assertEquals(0.0, Commons.cosPi(0.5), 0.0);
        assertEquals(0.0, Commons.cosPi(1.5), 0.0);
        assertEquals(-1.0, Commons.cosPi(1.0), 0.0);
        assertEquals(1.0, Commons.cosPi(2.0), 0.0);
    }

    @Test
    void sinPiAndCosPiReduceLargeArgumentsLikeBoost() {
        double quarterOffset = Math.scalb(1.0, 50) + 0.25;
        double eighthOffset = Math.scalb(1.0, 49) + 0.125;

        double rootHalf = Math.sqrt(0.5);
        assertClose("sinPi quarter-reduced", rootHalf, Commons.sinPi(quarterOffset), 5.0e-16);
        assertClose("cosPi quarter-reduced", rootHalf, Commons.cosPi(quarterOffset), 5.0e-16);

        double sinPiOverEight = 0.5 * Math.sqrt(2.0 - Math.sqrt(2.0));
        double cosPiOverEight = 0.5 * Math.sqrt(2.0 + Math.sqrt(2.0));
        assertClose("sinPi eighth-reduced", sinPiOverEight, Commons.sinPi(eighthOffset), 5.0e-16);
        assertClose("cosPi eighth-reduced", cosPiOverEight, Commons.cosPi(eighthOffset), 5.0e-16);
    }

    @Test
    void inverseRegularizedGammaRoundTripsTinyUpperTailWithoutCancellation() {
        double a = 0.75;
        double x = 40.0;
        double q = Gamma.gammaQ(a, x);

        assertTrue(q > 0.0 && q < Math.ulp(1.0));

        double actual = Gamma.gammaQInv(a, q);
        assertClose("tiny upper tail inverse", x, actual, 5.0e-12);
        assertClose("tiny upper tail roundtrip", q, Gamma.gammaQ(a, actual), 5.0e-13);
    }

    @Test
    void inverseGammaInitialGuessStaysNearBoostSeedRegions() throws Exception {
        assertInitialGuessNearRoot(0.2, 80.0);
        assertInitialGuessNearRoot(10.0, 0.001);
        assertInitialGuessNearRoot(800.0, 800.0);
    }

    @Test
    void gammaDirectFamiliesMatchReflectionAndRatioIdentities() {
        double negative = -0.25;
        double expectedLogAbs = Math.log(Math.PI)
            - Math.log(Math.abs(Math.sin(Math.PI * negative)))
            - Gamma.lgamma(1.0 - negative);
        DoubleI signedLog = Gamma.lgammaSigned(negative);
        assertClose("lgamma negative reflection", expectedLogAbs, signedLog.value(), 5.0e-15);
        assertEquals(-1, signedLog.sign());
        assertClose("lgamma negative alias", expectedLogAbs, Gamma.lgamma(negative), 5.0e-15);
        assertClose("tgamma negative reflection", -4.0 * Gamma.tgamma(0.75), Gamma.tgamma(negative), 5.0e-15);

        assertClose("tgamma1pm1 tiny positive", Math.expm1(Gamma.lgamma(1.0 + 1.0e-10)), Gamma.tgamma1pm1(1.0e-10), 5.0e-15);
        assertClose("tgamma1pm1 negative", Gamma.tgamma(0.75) - 1.0, Gamma.tgamma1pm1(-0.25), 5.0e-15);
        assertClose("tgammaRatio factorial", 12.0, Gamma.tgammaRatio(5.0, 3.0), 5.0e-15);
        assertClose("tgammaDeltaRatio factorial", 1.0 / 30.0, Gamma.tgammaDeltaRatio(5.0, 2.0), 5.0e-15);
        assertClose("tgammaDeltaRatio large", 1.0 / 400.0, Gamma.tgammaDeltaRatio(400.0, 1.0), 5.0e-15);
    }

    @Test
    void gammaPositiveIntegersUseExactFactorialFastPaths() {
        assertEquals(362880.0, Gamma.tgamma(10.0), 0.0);
        assertEquals(Math.log(362880.0), Gamma.lgamma(10.0), 0.0);
        assertEquals(362879.0, Gamma.tgamma1pm1(9.0), 0.0);
    }

    @Test
    void incompleteGammaFamiliesStayFiniteAcrossModerateAndExtremeCases() {
        double a = 3.5;
        double x = 2.0;
        assertClose("incomplete partition moderate", Gamma.tgamma(a), Gamma.tgammaLower(a, x) + Gamma.tgamma(a, x), 5.0e-15);

        double largeA = 400.0;
        double largeX = 400.0;
        double p = Gamma.gammaP(largeA, largeX);
        double q = Gamma.gammaQ(largeA, largeX);
        assertTrue(Double.isFinite(p));
        assertTrue(Double.isFinite(q));
        assertClose("regularized partition large", 1.0, p + q, 5.0e-15);

        assertEquals(0.0, Gamma.tgamma(40.0, 2000.0), 0.0);
    }

    @Test
    void inverseRegularizedGammaOnShapeRoundTripsRepresentativeCases() {
        assertInverseShapeRoundTrip(0.2, 0.4, 5.0e-8);
        assertInverseShapeRoundTrip(3.5, 2.0, 5.0e-8);
        assertInverseShapeRoundTrip(400.0, 400.0, 5.0e-6);
        assertInverseShapeRoundTrip(0.75, 18.0, 5.0e-7);
    }

    @Test
    void inverseRegularizedGammaOnShapeHonorsBoundariesAndValidation() {
        assertEquals(Double.MIN_NORMAL, Gamma.gammaPInva(2.0, 1.0), 0.0);
        assertEquals(Double.MIN_NORMAL, Gamma.gammaQInva(2.0, 0.0), 0.0);

        assertThrows(ArithmeticException.class, () -> Gamma.gammaPInva(2.0, 0.0));
        assertThrows(ArithmeticException.class, () -> Gamma.gammaQInva(2.0, 1.0));
        assertThrows(IllegalArgumentException.class, () -> Gamma.gammaPInva(0.0, 0.5));
        assertThrows(IllegalArgumentException.class, () -> Gamma.gammaQInva(2.0, -0.1));
        assertThrows(IllegalArgumentException.class, () -> Gamma.lgammaSigned(Double.NaN));
    }

    private static void assertInitialGuessNearRoot(double a, double x) throws Exception {
        double p = Gamma.gammaP(a, x);
        double q = Gamma.gammaQ(a, x);

        Method method = Gamma.class.getDeclaredMethod("findInverseGamma", double.class, double.class, double.class);
        method.setAccessible(true);
        double guessValue = ((Number) method.invoke(null, a, p, q)).doubleValue();

        double scale = Math.max(1.0, Math.abs(x));
        assertTrue(Math.abs(guessValue - x) <= 1.0e-8 * scale,
            "expected initial guess near root for a=" + a + ", x=" + x + ", guess=" + guessValue);
    }

    private static void assertInverseShapeRoundTrip(double a, double x, double tolerance) {
        double p = Gamma.gammaP(a, x);
        if (p > 0.0 && p < 1.0) {
            assertClose("shape inverse P", a, Gamma.gammaPInva(x, p), tolerance);
        }

        double q = Gamma.gammaQ(a, x);
        if (q > 0.0 && q < 1.0) {
            assertClose("shape inverse Q", a, Gamma.gammaQInva(x, q), tolerance);
        }
    }

    private static double recurrenceTerm(int order, double x) {
        if (order == 0) {
            return 1.0 / x;
        }
        double sign = ((order & 1) == 0) ? 1.0 : -1.0;
        double factorial = 1.0;
        for (int i = 2; i <= order; i++) {
            factorial *= i;
        }
        return sign * factorial / Math.pow(x, order + 1);
    }

    private static void assertClose(String label, double expected, double actual, double tolerance) {
        double scale = Math.max(1.0, Math.abs(expected));
        double error = Math.abs(actual - expected);
        assertTrue(error <= tolerance * scale,
            label + " mismatch: expected=" + expected + ", actual=" + actual + ", error=" + error);
    }
}