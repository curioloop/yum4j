package com.curioloop.yum4j.math;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class HypergeometricTest {

    @Test
    void oneFZeroMatchesClosedFormAndPolynomialCases() {
        assertClose(Math.pow(1.0 - 0.25, -0.75), Hypergeometric.oneFZero(0.75, 0.25), 1.0e-15);
        assertClose(Math.pow(1.0 - 1.5, 3.0), Hypergeometric.oneFZero(-3.0, 1.5), 1.0e-15);
        assertClose(Math.pow(1.0 - 1.5, -3.0), Hypergeometric.oneFZero(3.0, 1.5), 1.0e-15);
    }

    @Test
    void zeroFOneMatchesElementaryBoundaries() {
        assertEquals(1.0, Hypergeometric.zeroFOne(1.5, 0.0), 0.0);
        assertClose(Math.cosh(2.0), Hypergeometric.zeroFOne(0.5, 1.0), 5.0e-13);
    }

    @Test
    void oneFOnePreservesElementaryIdentities() {
        assertEquals(1.0, Hypergeometric.oneFOne(0.75, 2.0, 0.0), 0.0);
        assertClose(Math.exp(0.75), Hypergeometric.oneFOne(1.75, 1.75, 0.75), 5.0e-13);
        assertEquals(1.0, Hypergeometric.oneFOne(0.0, 2.0, 3.0), 0.0);
    }

    @Test
    void oneFOneRegularizedAndLogMatchTheBaseFunctionOnModerateInputs() {
        double a = 0.75;
        double b = 1.5;
        double z = 0.5;
        double base = Hypergeometric.oneFOne(a, b, z);
        assertClose(base / Gamma.tgamma(b), Hypergeometric.oneFOneRegularized(a, b, z), 5.0e-13);
        assertClose(Math.log(Math.abs(base)), Hypergeometric.logOneFOne(a, b, z), 5.0e-13);

        DoubleI signedLog = Hypergeometric.logOneFOneSigned(-40.5, 5.5, 10.0);
        double signedValue = Hypergeometric.oneFOne(-40.5, 5.5, 10.0);
        assertEquals(-1, signedLog.sign());
        assertClose(Math.log(Math.abs(signedValue)), signedLog.value(), 5.0e-13);
    }

    @Test
    void twoFZeroMatchesExplicitPolynomialAndSymmetry() {
        double z = 0.8;
        double expected = 1.0
            + (-3.0 * 1.5) * z
            + (((-3.0) * (-2.0)) * (1.5 * 2.5) / 2.0) * z * z
            + (((-3.0) * (-2.0) * (-1.0)) * (1.5 * 2.5 * 3.5) / 6.0) * z * z * z;
        assertClose(expected, Hypergeometric.twoFZero(-3.0, 1.5, z), 5.0e-13);
        assertClose(Hypergeometric.twoFZero(-3.0, 1.5, -0.75), Hypergeometric.twoFZero(1.5, -3.0, -0.75), 5.0e-13);
    }

    @Test
    void twoFZeroMatchesAlternatingContinuedFractionCandidate() {
        double expected = explicitTwoFZero(-4.0, -7.25, -0.75);
        assertClose(expected, Hypergeometric.twoFZero(-4.0, -7.25, -0.75), 5.0e-13);
        assertClose(expected, Hypergeometric.twoFZero(-7.25, -4.0, -0.75), 5.0e-13);
    }

    @Test
    void twoFZeroMatchesHermiteReductionCandidate() {
        double expected = explicitTwoFZero(-3.0, -2.5, -0.5);
        assertClose(expected, Hypergeometric.twoFZero(-3.0, -2.5, -0.5), 5.0e-13);
        assertClose(expected, Hypergeometric.twoFZero(-2.5, -3.0, -0.5), 5.0e-13);
    }

    @Test
    void twoFZeroMatchesLaguerreReductionCandidate() {
        double expected = explicitTwoFZero(-4.0, -2.0, 1.25);
        assertClose(expected, Hypergeometric.twoFZero(-4.0, -2.0, 1.25), 5.0e-13);
        assertClose(expected, Hypergeometric.twoFZero(-2.0, -4.0, 1.25), 5.0e-13);
    }

    @Test
    void pFqMatchesBoostCheckedSeriesFrontEnd() {
        assertClose(Math.exp(0.75), Hypergeometric.pFq(new double[0], new double[0], 0.75), 5.0e-13);
        assertClose(Hypergeometric.oneFZero(0.75, -2.0), Hypergeometric.pFq(new double[] {0.75}, new double[0], -2.0), 5.0e-13);
        assertClose(Hypergeometric.oneFZero(3.0, 1.5), Hypergeometric.pFq(new double[] {3.0}, new double[0], 1.5), 5.0e-13);
        assertClose(Hypergeometric.oneFOne(0.75, 1.5, 0.5), Hypergeometric.pFq(new double[] {0.75}, new double[] {1.5}, 0.5), 5.0e-13);
        assertClose(1.1324769673616009, Hypergeometric.pFq(new double[] {0.5, 1.25}, new double[] {1.75}, 0.3), 5.0e-13);
        assertClose(1.5255001145864491, Hypergeometric.pFq(new double[] {0.5, 0.75}, new double[] {2.0}, 1.0), 5.0e-12);
    }

    @Test
    void pfqDebugRouteRecognizesDelegatedShapesWithoutWideningTheSurface() {
        assertEquals("delegate-one-f-zero", Hypergeometric.debugPFqRoute(new double[] {0.75}, new double[0], 0.25));
        assertEquals("one-f-zero-exterior-special-case", Hypergeometric.debugPFqRoute(new double[] {3.0}, new double[0], 1.5));
        assertEquals("checked-series", Hypergeometric.debugPFqRoute(new double[] {-3.0}, new double[0], 1.0));
        assertEquals("delegate-zero-f-one", Hypergeometric.debugPFqRoute(new double[0], new double[] {1.5}, -10.0));
        assertEquals("delegate-one-f-one", Hypergeometric.debugPFqRoute(new double[] {1.25}, new double[] {5.0}, 80.0));
        assertEquals("checked-series", Hypergeometric.debugPFqRoute(new double[] {0.5, 1.25}, new double[] {1.75}, 0.3));
    }

    @Test
    void pfqDelegatedShapesMatchDirectNamedFamilies() {
        assertClose(Hypergeometric.oneFZero(0.75, 0.25), Hypergeometric.pFq(new double[] {0.75}, new double[0], 0.25), 5.0e-13);
        assertClose(Hypergeometric.zeroFOne(1.5, -10.0), Hypergeometric.pFq(new double[0], new double[] {1.5}, -10.0), 5.0e-13);
        assertClose(Hypergeometric.oneFOne(1.25, 5.0, 80.0), Hypergeometric.pFq(new double[] {1.25}, new double[] {5.0}, 80.0), 5.0e-12);
    }

    @Test
    void pfqAbsErrorOverloadKeepsDelegatedValueOnNamedShapes() {
        double[] absErrorHolder = new double[1];
        double actual = Hypergeometric.pFq(new double[] {0.75}, new double[] {1.5}, 0.5, absErrorHolder);
        assertClose(Hypergeometric.oneFOne(0.75, 1.5, 0.5), actual, 5.0e-13);
        assertTrue(absErrorHolder[0] >= 0.0);
    }

    @Test
    void pFqRejectsNonPublicNamedFamilyWidening() {
        assertThrows(IllegalArgumentException.class,
            () -> Hypergeometric.pFq(new double[] {-3.0, 1.5}, new double[0], 0.8));
        assertThrows(IllegalArgumentException.class,
            () -> Hypergeometric.pFq(new double[] {0.5, 0.75}, new double[] {2.0}, 1.5));
        assertThrows(IllegalArgumentException.class,
            () -> Hypergeometric.pFq(new double[] {-2.0, 1.5, 0.75}, new double[] {2.25, 1.25}, 1.5));
        assertThrows(IllegalArgumentException.class,
            () -> Hypergeometric.pFq(new double[] {0.75}, new double[0], 1.5));
        assertThrows(IllegalArgumentException.class,
            () -> Hypergeometric.pFq(new double[] {0.5, 0.75, 1.25}, new double[] {2.0, 2.5}, 1.05));
    }

    @Test
    void oneFOneAllowsFiniteNegativeIntegerLowerParameterBranch() {
        double z = 1.25;
        double expected = 1.0 + (2.0 / 3.0) * z + (z * z) / 6.0;
        assertClose(expected, Hypergeometric.oneFOne(-2.0, -3.0, z), 5.0e-13);
    }

    @Test
    void oneFOneAllowsEqualLowerPoleBoundary() {
        double z = 0.5;
        double expected = 1.0 + z + 0.5 * z * z + (z * z * z) / 6.0;
        assertClose(expected, Hypergeometric.oneFOne(-3.0, -3.0, z), 5.0e-13);
    }

    @Test
    void oneFOneMatchesBoostEqualNegativeIntegerExactCase() {
        double expected = Math.exp(0.5) * Gamma.gammaQ(31.0, 0.5);
        assertClose(expected, Hypergeometric.oneFOne(-30.0, -30.0, 0.5), 5.0e-13);
    }

    @Test
    void oneFOneBypassesKummerWhenLowerParameterIsNegativeInteger() {
        assertClose(329.0 / 945.0, Hypergeometric.oneFOne(-5.0, -10.0, -2.0), 5.0e-13);
        assertClose(13.0 / 105.0, Hypergeometric.oneFOne(-3.0, -7.0, -4.0), 5.0e-13);
    }

    @Test
    void oneFOneUsesKummerTieCaseForLargeNegativeArgument() {
        assertClose(0.0023899049564020686, Hypergeometric.oneFOne(2.75, 5.5, -30.0), 5.0e-13);
    }

    @Test
    void oneFOneUsesKummerForLargeNegativeGeneralBranch() {
        assertClose(0.02172870194649509, Hypergeometric.oneFOne(1.25, 5.0, -80.0), 5.0e-13);
    }

    @Test
    void oneFOneUsesExpm1IdentityForNearZeroOneTwoCase() {
        assertClose(1.0000000050000002, Hypergeometric.oneFOne(1.0, 2.0, 1.0e-8), 5.0e-15);
    }

    @Test
    void oneFOneMatchesLargePositiveAsymptoticBoostPoints() {
        assertClose(1.0583322436247727e29, Hypergeometric.oneFOne(1.25, 5.0, 80.0), 5.0e-12);
        assertClose(3.4769137460969137e81, Hypergeometric.oneFOne(2.5, 5.5, 200.0), 5.0e-12);
    }

    @Test
    void oneFOneMatchesAdditionalExactBoostBranches() {
        assertClose(0.99999999999969236, Hypergeometric.oneFOne(-1.0, 3.25, 1.0e-12), 5.0e-15);
        assertClose(4.3664867586881123e9, Hypergeometric.oneFOne(3.5, 2.5, 20.0), 5.0e-12);
    }

    @Test
    void oneFOneUsesNegativeARecurrencesInDivergentPositiveZRegions() {
        assertClose(-0.0026388002957836898, Hypergeometric.oneFOne(-25.25, 4.5, 20.0), 5.0e-12);
        assertClose(6.0087466519592357e-4, Hypergeometric.oneFOne(-40.5, 5.5, 15.0), 5.0e-12);
        assertClose(1.8019495599509887e-13, Hypergeometric.oneFOne(-80.5, 120.0, 30.0), 5.0e-12);
    }

    @Test
    void oneFOneMatchesAuxiliaryBoostSelectorRegions() {
        assertClose(0.71607543610687319, Hypergeometric.oneFOne(0.3, 20.0, -40.0), 5.0e-13);
        assertClose(1.0015026245531711, Hypergeometric.oneFOne(0.75, 500.0, 1.0), 5.0e-13);
    }

    @Test
    void oneFOneMatchesNegativeBBoostRegions() {
        assertClose(1.0011898953675544, Hypergeometric.oneFOne(0.01, -20.5, -40.0), 5.0e-12);
        assertClose(1.0072994558266608, Hypergeometric.oneFOne(0.01, -80.5, -120.0), 5.0e-12);
        assertClose(0.9966772438212735, Hypergeometric.oneFOne(0.01, -50.5, 20.0), 5.0e-12);
        assertClose(-1.2015602740428710e10, Hypergeometric.oneFOne(0.01, -80.5, 30.0), 1.0e-10);
        assertClose(-2.9505422636153340e8, Hypergeometric.oneFOne(0.01, -120.5, 40.0), 1.0e-10);
        assertClose(2.4949526375748996, Hypergeometric.oneFOne(-20.5, -55.25, 2.5), 5.0e-12);
        assertClose(3152.1712608095422, Hypergeometric.oneFOne(-20.5, -55.25, 25.0), 5.0e-12);
        assertClose(0.40911331919730348, Hypergeometric.oneFOne(15.0, -40.25, 2.5), 5.0e-12);
        assertClose(-1.210744548371988e42, Hypergeometric.oneFOne(15.0, -40.25, 30.0), 5.0e-12);
        assertClose(-1.8258728431168369e112, Hypergeometric.oneFOne(40.0, -120.5, 80.0), 5.0e-12);
    }

    @Test
    void oneFOneMatchesPositiveBTricomiBoostRegion() {
        assertClose(-0.00063141067047430628, Hypergeometric.oneFOne(-40.5, 5.5, 10.0), 5.0e-12);
    }

    @Test
    void oneFOneMatchesLargeAbzBoostRegions() {
        assertClose(8.9640818622317433e7, Hypergeometric.oneFOne(15.0, 30.0, 30.0), 5.0e-12);
        assertClose(4.3673607232628889e7, Hypergeometric.oneFOne(20.0, 40.0, 30.0), 5.0e-12);
        assertClose(1.623809437259031e40, Hypergeometric.oneFOne(60.0, 90.0, 120.0), 5.0e-12);
        assertClose(7.2707649104688672e50, Hypergeometric.oneFOne(40.0, 65.0, 150.0), 5.0e-12);
    }

    @Test
    void oneFOneMatchesDoubleUpperBesselIdentity() {
        double a = 2.75;
        double z = 4.0;
        double expected = Math.exp(0.5 * z) * Gamma.tgamma(a + 0.5) * Math.pow(0.25 * z, 0.5 - a) * Bessel.i(a - 0.5, 0.5 * z);
        assertClose(expected, Hypergeometric.oneFOne(a, 2.0 * a, z), 5.0e-13);
    }

    @Test
    void terminatedPFqAllowsFiniteNegativeIntegerLowerParameters() {
        double expected = 1.0 + 0.15625 + 0.025634765625;
        assertClose(expected,
            Hypergeometric.pFq(new double[] {-2.0, 0.75, 1.25}, new double[] {-3.0, 2.0}, 0.5),
            5.0e-13);
    }

    @Test
    void pfqSupportsDiskEdgeJustInsideUnitRadius() {
        assertClose(1.1339871075659018,
            Hypergeometric.pFq(new double[] {0.5, 0.75, 1.25}, new double[] {2.0, 2.5}, 0.95),
            5.0e-13);
    }

    @Test
    void pfqSupportsConvergentUnitCircleBoundary() {
        double[] upper = {0.5, 0.75, 1.25};
        double[] lower = {2.0, 2.5};
        assertTrue(Double.isFinite(Hypergeometric.pFq(upper, lower, 1.0)));
        assertTrue(Double.isFinite(Hypergeometric.pFq(upper, lower, -1.0)));
    }

    @Test
    void pfqMatchesBoostNegativeLowerCrossoverCases() {
        assertClose(Math.exp(0.5), Hypergeometric.pFq(new double[] {-30.0}, new double[] {-30.0}, 0.5), 5.0e-13);
        assertClose(-1.2015602740428710e10,
            Hypergeometric.pFq(new double[] {0.01}, new double[] {-80.5}, 30.0),
            1.0e-10);
        assertClose(-5.104205209027346,
            Hypergeometric.pFq(new double[] {0.75, 1.25}, new double[] {-2.5, 2.5}, 2.0),
            5.0e-12);
    }

    @Test
    void pfqRejectsMoreThanFiveLowerParametersByDefault() {
        assertThrows(IllegalArgumentException.class,
            () -> Hypergeometric.pFq(new double[] {0.75}, new double[] {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, 0.5));
    }

    @Test
    void terminatedPFqRejectsOutsideBoostPublicConvergenceBoundary() {
        assertThrows(IllegalArgumentException.class,
            () -> Hypergeometric.pFq(new double[] {-2.0, 1.5, 0.75}, new double[] {2.25, 1.25}, 1.5));
    }

    @Test
    void terminatedOneFOneMatchesExplicitPolynomial() {
        double z = 2.0;
        double expected = 1.0 - (3.0 / 1.5) * z + (6.0 / (1.5 * 2.5)) * 0.5 * z * z - (6.0 / (1.5 * 2.5 * 3.5)) * (z * z * z / 6.0);
        assertClose(expected, Hypergeometric.oneFOne(-3.0, 1.5, z), 5.0e-13);
    }

    @Test
    void invalidInputsAreRejected() {
        assertThrows(IllegalArgumentException.class, () -> Hypergeometric.oneFZero(0.75, 1.0));
        assertThrows(IllegalArgumentException.class, () -> Hypergeometric.oneFZero(-3.0, 1.0));
        assertThrows(IllegalArgumentException.class, () -> Hypergeometric.zeroFOne(0.0, 0.5));
        assertThrows(IllegalArgumentException.class, () -> Hypergeometric.oneFOne(0.75, -2.0, 0.5));
        assertThrows(IllegalArgumentException.class, () -> Hypergeometric.oneFOne(-4.0, -3.0, 0.5));
        assertThrows(IllegalArgumentException.class, () -> Hypergeometric.twoFZero(0.75, 1.25, -0.5));
        assertThrows(IllegalArgumentException.class, () -> Hypergeometric.pFq(new double[] {0.5, 0.75, 1.25}, new double[] {2.0}, 0.5));
        assertThrows(IllegalArgumentException.class, () -> Hypergeometric.pFq(new double[] {0.5, 0.75, 2.25}, new double[] {1.5, 2.0}, 1.0));
        assertThrows(IllegalArgumentException.class, () -> Hypergeometric.pFq(new double[] {0.5, 0.75, 3.5}, new double[] {1.5, 2.0}, -1.0));
        assertThrows(IllegalArgumentException.class, () -> Hypergeometric.pFq(new double[] {-4.0, 0.75, 1.25}, new double[] {-3.0, 2.0}, 0.5));
        assertThrows(IllegalArgumentException.class, () -> Hypergeometric.pFq(null, new double[0], 0.5));
    }

    private static double explicitTwoFZero(double a, double b, double z) {
        int degree = terminatingDegreeForTest(a, b);
        if (degree < 0) {
            throw new IllegalArgumentException("explicitTwoFZero requires a terminating branch");
        }

        double sum = 1.0;
        double term = 1.0;
        for (int n = 0; n < degree; n++) {
            term *= ((a + n) * (b + n) * z) / (n + 1.0);
            sum += term;
        }
        return sum;
    }

    private static int terminatingDegreeForTest(double a, double b) {
        int degreeA = terminatingDegreeForTest(a);
        int degreeB = terminatingDegreeForTest(b);
        if (degreeA < 0) {
            return degreeB;
        }
        if (degreeB < 0) {
            return degreeA;
        }
        return Math.min(degreeA, degreeB);
    }

    private static int terminatingDegreeForTest(double value) {
        if (value > 0.0) {
            return -1;
        }
        double rounded = Math.rint(value);
        if (Math.abs(value - rounded) > 1.0e-12 * Math.max(1.0, Math.abs(value))) {
            return -1;
        }
        return (int) Math.rint(-value);
    }

    private static void assertClose(double expected, double actual, double tolerance) {
        double scale = Math.max(1.0, Math.abs(expected));
        double error = Math.abs(actual - expected);
        assertTrue(error <= tolerance * scale,
            "mismatch: expected=" + expected + ", actual=" + actual + ", error=" + error);
    }
}