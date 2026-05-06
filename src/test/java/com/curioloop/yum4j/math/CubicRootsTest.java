package com.curioloop.yum4j.math;

import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.SplittableRandom;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class CubicRootsTest {

    @Test
    void zeroPolynomialReturnsTripleZero() {
        assertRoots(CubicRoots.solveReal(0.0, 0.0, 0.0, 0.0), 0.0, 0.0, 0.0, 0.0);
    }

    @Test
    void monomialWithZeroConstantReturnsTripleZero() {
        assertRoots(CubicRoots.solveReal(1.0, 0.0, 0.0, 0.0), 0.0, 0.0, 0.0, 0.0);
    }

    @Test
    void quadraticFallbackPreservesNaNPadding() {
        assertRoots(CubicRoots.solveReal(0.0, 1.0, -5.0, 6.0), 2.0, 3.0, Double.NaN, 1.0e-12);
    }

    @Test
    void linearFallbackPreservesNaNPadding() {
        assertRoots(CubicRoots.solveReal(0.0, 0.0, 2.0, -4.0), 2.0, Double.NaN, Double.NaN, 1.0e-12);
    }

    @Test
    void degeneratePolynomialWithoutSolutionsReturnsNaNTriple() {
        assertRoots(CubicRoots.solveReal(0.0, 0.0, 0.0, 5.0), Double.NaN, Double.NaN, Double.NaN, 0.0);
    }

    @Test
    void quadraticFallbackWithNoRealRootsReturnsNaNTriple() {
        assertRoots(CubicRoots.solveReal(0.0, 1.0, 0.0, 1.0), Double.NaN, Double.NaN, Double.NaN, 0.0);
    }

    @Test
    void solvesCanonicalSingleRealRootCasesWithNaNPadding() {
        assertRoots(CubicRoots.solveReal(1.0, 0.0, 0.0, 1.0), -1.0, Double.NaN, Double.NaN, 0.0);
        assertRoots(CubicRoots.solveReal(1.0, 0.0, 0.0, -1.0), 1.0, Double.NaN, Double.NaN, 0.0);
        assertRoots(CubicRoots.solveReal(1.0, 0.0, 0.0, -2.0), Math.cbrt(2.0), Double.NaN, Double.NaN, 1.0e-12);
        assertRoots(CubicRoots.solveReal(1.0, 0.0, 0.0, -8.0), 2.0, Double.NaN, Double.NaN, 0.0);
    }

    @Test
    void dZeroShortcutKeepsZeroRootInSortedTriple() {
        assertRoots(CubicRoots.solveReal(1.0, -3.0, 2.0, 0.0), 0.0, 1.0, 2.0, 1.0e-12);
    }

    @Test
    void solvesThreeDistinctRoots() {
        assertRoots(CubicRoots.solveReal(1.0, -6.0, 11.0, -6.0), 1.0, 2.0, 3.0, 1.0e-12);
    }

    @Test
    void preservesDoubleRoots() {
        assertRoots(CubicRoots.solveReal(1.0, 0.0, -3.0, -2.0), -1.0, -1.0, 2.0, 1.0e-12);
    }

    @Test
    void singleRealRootRetainsNaNPadding() {
        assertRoots(CubicRoots.solveReal(1.0, -3.0, 1.0, -3.0), 3.0, Double.NaN, Double.NaN, 1.0e-12);
    }

    @Test
    void issue757CaseReturnsOnlyTheRealRoot() {
        assertRoots(CubicRoots.solveReal(1.0, 3.0, 4.0, 2.0), -1.0, Double.NaN, Double.NaN, 1.0e-12);
    }

    @Test
    void residualMatchesBoostReferenceFormula() {
        Double2 residual = CubicRoots.cubicRootResidual(1.0, -6.0, 11.0, -6.0, 2.0);
        assertEquals(0.0, residual._1(), 0.0);
        assertEquals(referenceExpectedResidual(1.0, -6.0, 11.0, -6.0, 2.0), residual._2(), 0.0);
    }

    @Test
    void conditionNumberMatchesReferenceFormula() {
        assertEquals(30.0, CubicRoots.cubicRootConditionNumber(1.0, -6.0, 11.0, -6.0, 2.0), 0.0);
    }

    @Test
    void conditionNumberIsInfiniteAtZeroRoot() {
        assertTrue(Double.isInfinite(CubicRoots.cubicRootConditionNumber(1.0, 0.0, 0.0, 0.0, 0.0)));
    }

    @Test
    void helperMethodsPropagateNaNRootSlots() {
        Double2 residual = CubicRoots.cubicRootResidual(1.0, 3.0, 4.0, 2.0, Double.NaN);
        assertTrue(Double.isNaN(residual._1()));
        assertTrue(Double.isNaN(residual._2()));
        assertTrue(Double.isNaN(CubicRoots.cubicRootConditionNumber(1.0, 3.0, 4.0, 2.0, Double.NaN)));
    }

    @Test
    void illConditionedCaseMatchesBoostExpectations() {
        Double3 roots = CubicRoots.solveReal(1.0, 10000.0, 200.0, 1.0);
        assertClose(-9999.97999997, roots._1(), 1.0e-12);
        if (Double.isNaN(roots._2())) {
            assertTrue(Double.isNaN(roots._3()));
            return;
        }
        assertClose(-0.0100100150263001, roots._2(), 1.01e-5);
        assertClose(-0.0099900149737999, roots._3(), 1.01e-5);
    }

    @Test
    void solvesDeterministicThreeRealRootStressCases() {
        SplittableRandom random = new SplittableRandom(12345);
        for (int trial = 0; trial < 10; trial++) {
            double[] expected = generateSeparatedRoots(random);
            double c2 = -(expected[0] + expected[1] + expected[2]);
            double c3 = expected[0] * expected[1] + expected[0] * expected[2] + expected[1] * expected[2];
            double c4 = -expected[0] * expected[1] * expected[2];
            Double3 roots = CubicRoots.solveReal(1.0, c2, c3, c4);
            assertClose(expected[0], roots._1(), 1.0e-11);
            assertClose(expected[1], roots._2(), 1.0e-11);
            assertClose(expected[2], roots._3(), 1.0e-11);
            assertResidualWithinExpected(1.0, c2, c3, c4, roots._1());
            assertResidualWithinExpected(1.0, c2, c3, c4, roots._2());
            assertResidualWithinExpected(1.0, c2, c3, c4, roots._3());
        }
    }

    private static double[] generateSeparatedRoots(SplittableRandom random) {
        double[] roots = new double[3];
        int size = 0;
        while (size < roots.length) {
            double candidate = random.nextDouble(-2.0, 2.0);
            boolean separated = true;
            for (int index = 0; index < size; index++) {
                if (Math.abs(candidate - roots[index]) <= 0.05) {
                    separated = false;
                    break;
                }
            }
            if (separated) {
                roots[size++] = candidate;
            }
        }
        Arrays.sort(roots);
        return roots;
    }

    private static void assertRoots(Double3 actual,
                                    double expected1,
                                    double expected2,
                                    double expected3,
                                    double tolerance) {
        assertRoot(expected1, actual._1(), tolerance);
        assertRoot(expected2, actual._2(), tolerance);
        assertRoot(expected3, actual._3(), tolerance);
    }

    private static void assertRoot(double expected, double actual, double tolerance) {
        if (Double.isNaN(expected)) {
            assertTrue(Double.isNaN(actual), "Expected NaN but was " + actual);
            return;
        }
        assertClose(expected, actual, tolerance);
    }

    private static void assertResidualWithinExpected(double c1, double c2, double c3, double c4, double root) {
        Double2 residual = CubicRoots.cubicRootResidual(c1, c2, c3, c4, root);
        assertTrue(Math.abs(residual._1()) <= residual._2(),
            "Residual exceeds expected bound for root=" + root + ": residual=" + residual._1() + ", expected=" + residual._2());
    }

    private static double referenceExpectedResidual(double c1, double c2, double c3, double c4, double root) {
        double absRoot = Math.abs(root);
        double expectedResidual = Math.fma(4.0 * Math.abs(c1), absRoot, 3.0 * Math.abs(c2));
        expectedResidual = Math.fma(expectedResidual, absRoot, 2.0 * Math.abs(c3));
        expectedResidual = Math.fma(expectedResidual, absRoot, Math.abs(c4));
        return expectedResidual * Math.ulp(1.0);
    }

    private static void assertClose(double expected, double actual, double tolerance) {
        double scale = Math.max(1.0, Math.abs(expected));
        assertEquals(expected, actual, tolerance * scale);
    }
}