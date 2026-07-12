package com.curioloop.yum4j.stats.tool;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class CorrelationTest {

    private static final double[] WILKINSON_X = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    private static final double[] WILKINSON_BIG = {99999991.0, 99999992.0, 99999993.0, 99999994.0, 99999995.0, 99999996.0, 99999997.0, 99999998.0, 99999999.0};
    private static final double[] WILKINSON_LITTLE = {0.99999991, 0.99999992, 0.99999993, 0.99999994, 0.99999995, 0.99999996, 0.99999997, 0.99999998, 0.99999999};
    private static final double[] WILKINSON_HUGE = {1e12, 2e12, 3e12, 4e12, 5e12, 6e12, 7e12, 8e12, 9e12};
    private static final double[] WILKINSON_TINY = {1e-12, 2e-12, 3e-12, 4e-12, 5e-12, 6e-12, 7e-12, 8e-12, 9e-12};
    private static final double[] WILKINSON_ROUND = {0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5};

    @ParameterizedTest(name = "{0}")
    @MethodSource("pearsonCases")
    void pearsonMatchesScipyReferenceCases(PearsonCase testCase) {
        Correlation.Result actual = Correlation.pearson(testCase.left, testCase.right, testCase.alternative);
        assertClose(testCase.expectedStatistic, actual.statistic(), testCase.statTolerance, testCase.name + " statistic");
        assertClose(testCase.expectedPValue, actual.pValue(), testCase.pTolerance, testCase.name + " pValue");
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("wilkinsonScalePairs")
    void pearsonWilkinsonScalePairsRemainPerfectlyCorrelated(WilkinsonPair pair) {
        Correlation.Result actual = Correlation.pearson(pair.left, pair.right);
        assertEquals(1.0, actual.statistic(), 2e-14, pair.name);
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("spearmanCases")
    void spearmanMatchesScipyReferenceCases(SpearmanCase testCase) {
        Correlation.Result actual = Correlation.spearman(testCase.left, testCase.right,
                testCase.alternative, testCase.missingPolicy);
        assertClose(testCase.expectedStatistic, actual.statistic(), testCase.statTolerance, testCase.name + " statistic");
        if (testCase.expectedPValue != null) {
            assertClose(testCase.expectedPValue, actual.pValue(), testCase.pTolerance, testCase.name + " pValue");
        }
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("wilkinsonScalePairs")
    void spearmanWilkinsonScalePairsRemainPerfectlyCorrelated(WilkinsonPair pair) {
        Correlation.Result actual = Correlation.spearman(pair.left, pair.right);
        assertEquals(1.0, actual.statistic(), 2e-14, pair.name);
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("kendallExactNoTieCases")
    void kendallNoTieExactCasesMatchScipyTables(KendallExactCase testCase) {
        Correlation.Result actual = Correlation.kendall(testCase.left, testCase.right,
                testCase.variant, Correlation.KendallMethod.AUTO,
                Correlation.Alternative.TWO_SIDED, MissingPolicy.RAISE);
        assertClose(testCase.expectedStatistic, actual.statistic(), 1e-11, testCase.name + " statistic");
        assertClose(testCase.expectedPValue, actual.pValue(), 1e-11, testCase.name + " pValue");
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("kendallTieCases")
    void kendallTieAndVariantCasesMatchScipyTables(KendallCase testCase) {
        Correlation.Result actual = Correlation.kendall(testCase.left, testCase.right,
                testCase.variant, testCase.method, testCase.alternative, testCase.missingPolicy);
        assertClose(testCase.expectedStatistic, actual.statistic(), testCase.statTolerance, testCase.name + " statistic");
        if (testCase.expectedPValue != null) {
            assertClose(testCase.expectedPValue, actual.pValue(), testCase.pTolerance, testCase.name + " pValue");
        }
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("kendallAlternativeCases")
    void kendallAlternativePValuesMatchScipySplitting(KendallCase testCase) {
        Correlation.Result actual = Correlation.kendall(testCase.left, testCase.right,
                testCase.variant, testCase.method, testCase.alternative, testCase.missingPolicy);
        assertClose(testCase.expectedStatistic, actual.statistic(), testCase.statTolerance, testCase.name + " statistic");
        assertClose(testCase.expectedPValue, actual.pValue(), testCase.pTolerance, testCase.name + " pValue");
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("degenerateKendallCases")
    void kendallDegenerateInputsReturnNan(KendallCase testCase) {
        Correlation.Result actual = Correlation.kendall(testCase.left, testCase.right,
                testCase.variant, testCase.method, testCase.alternative, testCase.missingPolicy);
        assertTrue(Double.isNaN(actual.statistic()), testCase.name + " statistic");
        assertTrue(Double.isNaN(actual.pValue()), testCase.name + " pValue");
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("matrixCases")
    void matrixOverloadsAreSymmetricAndMatchScalarPairwise(MatrixCase testCase) {
        Correlation.MatrixResult actual = matrix(testCase.kind, testCase.data, testCase.observations,
                testCase.variables, Correlation.Alternative.TWO_SIDED, MissingPolicy.RAISE);
        for (int variable = 0; variable < testCase.variables; variable++) {
            assertEquals(1.0, actual.statistic(variable, variable), 0.0, testCase.name + " diag statistic");
            assertEquals(0.0, actual.pValue(variable, variable), 0.0, testCase.name + " diag pValue");
        }
        for (int row = 0; row < testCase.variables; row++) {
            for (int column = row + 1; column < testCase.variables; column++) {
                Correlation.Result expected = scalar(testCase.kind,
                        column(testCase.data, testCase.observations, testCase.variables, row),
                        column(testCase.data, testCase.observations, testCase.variables, column));
                assertClose(expected.statistic(), actual.statistic(row, column), 2e-14,
                        testCase.name + " statistic[" + row + "," + column + "]");
                assertClose(expected.pValue(), actual.pValue(row, column), 2e-13,
                        testCase.name + " pValue[" + row + "," + column + "]");
                assertEquals(actual.statistic(row, column), actual.statistic(column, row), 0.0,
                        testCase.name + " statistic symmetry");
                assertEquals(actual.pValue(row, column), actual.pValue(column, row), 0.0,
                        testCase.name + " pValue symmetry");
            }
        }
    }

    @Test
    void spearmanThreeColumnMatrixMatchesScipyReference() {
        double[] data = {
                0.0, -0.0, 0.0,
                1.0, -1.0, 1.0,
                2.0, -2.0, 2.0,
                3.0, -3.0, 3.0,
                4.0, -4.0, 5.0,
                5.0, -5.0, 4.0
        };
        Correlation.MatrixResult actual = Correlation.spearman(data, 6, 3);
        double[] expectedStatistic = {
                1.0, -1.0, 0.94285714,
                -1.0, 1.0, -0.94285714,
                0.94285714, -0.94285714, 1.0
        };
        double[] expectedPValue = {
                0.0, 0.0, 0.00480466472,
                0.0, 0.0, 0.00480466472,
                0.00480466472, 0.00480466472, 0.0
        };
        assertArrayClose(expectedStatistic, actual.statistic(), 5e-9, "spearman 3-column statistic");
        assertArrayClose(expectedPValue, actual.pValue(), 5e-11, "spearman 3-column pValue");
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("matrixKinds")
    void matrixMissingPolicyIsValidatedOnDiagonalOnlyInputs(CorrelationKind kind) {
        assertThrows(IllegalArgumentException.class, () ->
                matrix(kind, new double[]{Double.NaN, 2.0}, 2, 1,
                        Correlation.Alternative.TWO_SIDED, MissingPolicy.RAISE));
        assertThrows(IllegalArgumentException.class, () ->
                matrix(kind, new double[]{1.0, 2.0}, 2, 1,
                        Correlation.Alternative.TWO_SIDED, MissingPolicy.CONSERVATIVE));
    }

    @Test
    void pairwiseDropMissingUsesOnlyFinitePairs() {
        double[] left = {1.0, 2.0, Double.NaN, 4.0, 5.0};
        double[] right = {5.0, 4.0, 100.0, 2.0, 1.0};
        Correlation.Result actual = Correlation.pearson(left, right,
                Correlation.Alternative.TWO_SIDED, MissingPolicy.DROP);
        Correlation.Result expected = Correlation.pearson(
                new double[]{1.0, 2.0, 4.0, 5.0}, new double[]{5.0, 4.0, 2.0, 1.0});
        assertEquals(expected.statistic(), actual.statistic(), 0.0);
        assertEquals(expected.pValue(), actual.pValue(), 0.0);
    }

    @Test
    void kendallDropMissingMatchesCompressedInputsWhenNanIsInSecondArgument() {
        double[] left = {1.0, 2.0, 3.0, 4.0};
        double[] right = {Double.NaN, 2.4, 3.4, 3.4};
        Correlation.Result actual = Correlation.kendall(left, right,
                Correlation.KendallVariant.B, Correlation.KendallMethod.AUTO,
                Correlation.Alternative.TWO_SIDED, MissingPolicy.DROP);
        Correlation.Result expected = Correlation.kendall(
                new double[]{2.0, 3.0, 4.0}, new double[]{2.4, 3.4, 3.4});
        assertEquals(expected.statistic(), actual.statistic(), 0.0);
        assertEquals(expected.pValue(), actual.pValue(), 0.0);
    }

    @Test
    void invalidInputsFailFast() {
        assertThrows(IllegalArgumentException.class, () ->
                Correlation.pearson(new double[]{1.0, 2.0}, new double[]{1.0}));
        assertThrows(IllegalArgumentException.class, () ->
                Correlation.pearson(new double[]{1.0, Double.POSITIVE_INFINITY}, new double[]{1.0, 2.0}));
        assertThrows(IllegalArgumentException.class, () ->
                Correlation.pearson(new double[]{1.0, Double.NaN}, new double[]{1.0, 2.0}));
        assertThrows(IllegalArgumentException.class, () ->
                Correlation.spearman(new double[]{1.0, Double.NaN}, new double[]{1.0, 2.0}));
        assertThrows(IllegalArgumentException.class, () ->
                Correlation.kendall(new double[]{1.0, Double.NaN}, new double[]{1.0, 2.0}));
        assertThrows(IllegalArgumentException.class, () ->
                Correlation.kendall(new double[]{1.0, 2.0, 2.0}, new double[]{1.0, 2.0, 3.0},
                        Correlation.KendallVariant.B, Correlation.KendallMethod.EXACT,
                        Correlation.Alternative.TWO_SIDED, MissingPolicy.RAISE));
        assertThrows(IllegalArgumentException.class, () -> Correlation.pearson(new double[]{1.0}, 1, 2));
        assertThrows(IllegalArgumentException.class, () -> Correlation.pearson(new double[]{1.0, 2.0}, 1, 2));
    }

    static Stream<PearsonCase> pearsonCases() {
        List<PearsonCase> cases = new ArrayList<>();
        cases.add(new PearsonCase("fixture scipy-doc-pearson-mixed", a(1, 2, 3, 4, 5, 6, 7),
                a(10, 9, 2.5, 6, 4, 3, 2), Correlation.Alternative.TWO_SIDED,
                -0.8285038835884277, 0.021280260007523352, 2e-14, 2e-13));
        cases.add(new PearsonCase("fixture spearman-doc-ties", a(1, 2, 3, 4, 5), a(5, 6, 7, 8, 7),
                Correlation.Alternative.TWO_SIDED, 0.8320502943378436, 0.0805095732984986, 2e-14, 2e-13));
        cases.add(new PearsonCase("fixture kendall-doc-ties", a(12, 2, 1, 12, 2), a(1, 4, 7, 1, 0),
                Correlation.Alternative.TWO_SIDED, -0.5566399985665696, 0.3297879804263832, 2e-14, 2e-13));
        cases.add(new PearsonCase("scipy basic sqrt3-over-2", a(-1, 0, 1), a(0, 0, 3),
                Correlation.Alternative.TWO_SIDED, Math.sqrt(3.0) / 2.0, 1.0 / 3.0, 2e-15, 2e-15));
        cases.add(new PearsonCase("scipy exact positive one", a(0, 1, 2), a(0, 1, 2),
                Correlation.Alternative.TWO_SIDED, 1.0, 0.0, 2e-15, 1e-14));
        cases.add(new PearsonCase("scipy exact negative one", a(0, 1, 2), a(0, -1, -2),
                Correlation.Alternative.TWO_SIDED, -1.0, 0.0, 2e-15, 1e-14));
        cases.add(new PearsonCase("scipy n2 positive", a(1, 2), a(3, 5),
                Correlation.Alternative.TWO_SIDED, 1.0, 1.0, 0.0, 0.0));
        cases.add(new PearsonCase("scipy n2 negative", a(2, 1), a(3, 5),
                Correlation.Alternative.TWO_SIDED, -1.0, 1.0, 0.0, 0.0));
        cases.add(new PearsonCase("scipy n2 constant", a(0.667, 0.667), a(0.123, 0.456),
                Correlation.Alternative.TWO_SIDED, Double.NaN, Double.NaN, 0.0, 0.0));
        cases.add(new PearsonCase("scipy very small values", a(0.004434375, 0.004756007, 0.003911996, 0.0038005, 0.003409971),
                a(2.48e-188, 7.41e-181, 4.09e-208, 2.08e-223, 2.66e-245),
                Correlation.Alternative.TWO_SIDED, 0.7272930540750450, 0.1637805429533202, 2e-15, 2e-14));
        cases.add(new PearsonCase("scipy very large values", a(0, 0, 0, 1e90, 1e90, 1e90, 1e90),
                a(0, 1e90, 2e90, 3e90, 4e90, 5e90, 6e90),
                Correlation.Alternative.TWO_SIDED, 0.8660254037844386, 0.011724811003954638, 2e-15, 2e-14));
        cases.add(new PearsonCase("scipy extremely large values", a(2.3e200, 4.5e200, 6.7e200, 8e200),
                a(1.2e199, 5.5e200, 3.3e201, 1.0e200),
                Correlation.Alternative.TWO_SIDED, 0.351312332103289, 0.648687667896711, 2e-15, 2e-14));
        cases.add(new PearsonCase("R alternative positive two-sided", a(1, 2, 3, 4), a(0, 1, 0.5, 1),
                Correlation.Alternative.TWO_SIDED, 0.6741998624632421, 0.325800137536, 2e-15, 1e-12));
        cases.add(new PearsonCase("R alternative positive less", a(1, 2, 3, 4), a(0, 1, 0.5, 1),
                Correlation.Alternative.LESS, 0.6741998624632421, 0.8370999312316, 2e-15, 1e-12));
        cases.add(new PearsonCase("R alternative positive greater", a(1, 2, 3, 4), a(0, 1, 0.5, 1),
                Correlation.Alternative.GREATER, 0.6741998624632421, 0.1629000687684, 2e-15, 1e-12));
        cases.add(new PearsonCase("R alternative negative less", a(1, 2, 3, 4), a(0, -1, -0.5, -1),
                Correlation.Alternative.LESS, -0.6741998624632421, 0.1629000687684, 2e-15, 1e-12));
        cases.add(new PearsonCase("R alternative negative greater", a(1, 2, 3, 4), a(0, -1, -0.5, -1),
                Correlation.Alternative.GREATER, -0.6741998624632421, 0.8370999312316, 2e-15, 1e-12));
        cases.add(new PearsonCase("gh17795 negative perfect greater", range(10), negRange(10),
                Correlation.Alternative.GREATER, -1.0, 1.0, 2e-15, 1e-20));
        cases.add(new PearsonCase("gh17795 negative perfect less", range(10), negRange(10),
                Correlation.Alternative.LESS, -1.0, 0.0, 2e-15, 1e-20));
        return cases.stream();
    }

    static Stream<WilkinsonPair> wilkinsonScalePairs() {
        Object[][] values = {
                {"X", WILKINSON_X},
                {"BIG", WILKINSON_BIG},
                {"LITTLE", WILKINSON_LITTLE},
                {"HUGE", WILKINSON_HUGE},
                {"TINY", WILKINSON_TINY},
                {"ROUND", WILKINSON_ROUND}
        };
        List<WilkinsonPair> cases = new ArrayList<>();
        for (int left = 0; left < values.length; left++) {
            for (int right = left; right < values.length; right++) {
                cases.add(new WilkinsonPair(values[left][0] + " vs " + values[right][0],
                        (double[]) values[left][1], (double[]) values[right][1]));
            }
        }
        return cases.stream();
    }

    static Stream<SpearmanCase> spearmanCases() {
        double[] rCaseX = a(2.0, 47.4, 42.0, 10.8, 60.1, 1.7, 64.0, 63.1, 1.0, 1.4, 7.9, 0.3, 3.9, 0.3, 6.7);
        double[] rCaseY = a(22.6, 8.3, 44.4, 11.9, 24.6, 0.6, 5.7, 41.6, 0.0, 0.6, 6.7, 3.8, 1.0, 1.2, 1.4);
        return Stream.of(
                new SpearmanCase("fixture scipy-doc-pearson-mixed", a(1, 2, 3, 4, 5, 6, 7),
                        a(10, 9, 2.5, 6, 4, 3, 2), Correlation.Alternative.TWO_SIDED, MissingPolicy.RAISE,
                        -0.7857142857142859, 0.03623846267982714, 2e-14, 2e-13),
                new SpearmanCase("R cross-check ties", a(1, 2, 3, 4, 5), a(5, 6, 7, 8, 7),
                        Correlation.Alternative.TWO_SIDED, MissingPolicy.RAISE,
                        0.82078268166812329, 0.088587005313543798, 2e-14, 2e-13),
                new SpearmanCase("mstats tie example", a(5.05, 6.75, 3.21, 2.66), a(1.65, 2.64, 2.64, 6.95),
                        Correlation.Alternative.TWO_SIDED, MissingPolicy.RAISE,
                        -0.6324555, null, 5e-8, 0.0),
                new SpearmanCase("mstats longer tie example", rCaseX, rCaseY,
                        Correlation.Alternative.TWO_SIDED, MissingPolicy.RAISE,
                        0.6887298747763864, 0.004519192910756, 2e-14, 2e-13),
                new SpearmanCase("mstats longer tie greater", rCaseX, rCaseY,
                        Correlation.Alternative.GREATER, MissingPolicy.RAISE,
                        0.6887298747763864, 0.002259596455378, 2e-14, 2e-13),
                new SpearmanCase("mstats longer tie less", rCaseX, rCaseY,
                        Correlation.Alternative.LESS, MissingPolicy.RAISE,
                        0.6887298747763864, 0.9977404035446, 2e-14, 2e-13),
                new SpearmanCase("rank tie equals Pearson on ranks", a(1, 2, 3, 4), a(1, 2, 2, 3),
                        Correlation.Alternative.TWO_SIDED, MissingPolicy.RAISE,
                        0.9486832980505138, null, 2e-15, 0.0),
                new SpearmanCase("drop missing paired value", a(1, 2, 3, 4), a(8, 7, 6, Double.NaN),
                        Correlation.Alternative.TWO_SIDED, MissingPolicy.DROP,
                        -1.0, 0.0, 0.0, 0.0),
                new SpearmanCase("constant left returns nan", a(1, 1, 1), a(0, 1, 2),
                        Correlation.Alternative.TWO_SIDED, MissingPolicy.RAISE,
                        Double.NaN, Double.NaN, 0.0, 0.0),
                new SpearmanCase("constant right returns nan", a(2, 0, 2), a(2, 2, 2),
                        Correlation.Alternative.TWO_SIDED, MissingPolicy.RAISE,
                        Double.NaN, Double.NaN, 0.0, 0.0),
                new SpearmanCase("gh6061 overflow regression", spearmanOverflowX(), spearmanOverflowY(),
                        Correlation.Alternative.TWO_SIDED, MissingPolicy.RAISE,
                        0.998, null, 2e-15, 0.0)
        );
    }

    static Stream<KendallExactCase> kendallExactNoTieCases() {
        List<KendallExactCase> cases = new ArrayList<>();
        Object[][] base = {
                {"balanced n8", a(5, 2, 1, 3, 6, 4, 7, 8), a(5, 2, 6, 3, 1, 8, 7, 4), 0.0, 1.0},
                {"balanced n9", a(0, 5, 2, 1, 3, 6, 4, 7, 8), a(5, 2, 0, 6, 3, 1, 8, 7, 4), 0.0, 1.0},
                {"negative small n7", a(5, 2, 1, 3, 6, 4, 7), a(5, 2, 6, 3, 1, 7, 4), -0.14285714286, 0.77261904762},
                {"positive small n7", a(2, 1, 3, 6, 4, 7, 8), a(2, 6, 3, 1, 8, 7, 4), 0.047619047619, 1.0},
                {"perfect n10", range(10), range(10), 1.0, 5.511463844797e-07},
                {"one swap n10", range(10), a(0, 2, 1, 3, 4, 5, 6, 7, 8, 9), 0.9555555555555556, 5.511463844797e-06},
                {"two swaps n10", range(10), a(0, 2, 1, 3, 4, 6, 5, 7, 8, 9), 0.9111111111111111, 2.976190476190e-05},
                {"reverse n10", range(10), reverseRange(10), -1.0, 5.511463844797e-07},
                {"reverse one swap n10", range(10), a(9, 7, 8, 6, 5, 4, 3, 2, 1, 0), -0.9555555555555556, 5.511463844797e-06},
                {"reverse two swaps n10", range(10), a(9, 7, 8, 6, 5, 3, 4, 2, 1, 0), -0.9111111111111111, 2.976190476190e-05}
        };
        for (Object[] row : base) {
            for (Correlation.KendallVariant variant : Correlation.KendallVariant.values()) {
                cases.add(new KendallExactCase(row[0] + " tau-" + variant.name().toLowerCase(),
                        (double[]) row[1], (double[]) row[2], variant, (double) row[3], (double) row[4]));
            }
        }
        return cases.stream();
    }

    static Stream<KendallCase> kendallTieCases() {
        return Stream.of(
                new KendallCase("Kendall 1970 tau-b ties", a(1, 2, 2, 4, 4, 6, 6, 8, 9, 9),
                        a(1, 2, 4, 4, 4, 4, 8, 8, 8, 10), Correlation.KendallVariant.B,
                        Correlation.KendallMethod.AUTO, Correlation.Alternative.TWO_SIDED, MissingPolicy.RAISE,
                        0.85895569, null, 5e-9, 0.0),
                new KendallCase("Kendall 1970 tau-c ties", a(1, 2, 2, 4, 4, 6, 6, 8, 9, 9),
                        a(1, 2, 4, 4, 4, 4, 8, 8, 8, 10), Correlation.KendallVariant.C,
                        Correlation.KendallMethod.AUTO, Correlation.Alternative.TWO_SIDED, MissingPolicy.RAISE,
                        0.825, null, 5e-14, 0.0),
                new KendallCase("R tau-b with ties", a(12, 2, 1, 12, 2), a(1, 4, 7, 1, 0),
                        Correlation.KendallVariant.B, Correlation.KendallMethod.AUTO, Correlation.Alternative.TWO_SIDED,
                        MissingPolicy.RAISE, -0.47140452079103173, 0.28274545993277478, 2e-14, 2e-13),
                new KendallCase("perfect with ties tau-b", a(1, 1, 2), a(1, 1, 2),
                        Correlation.KendallVariant.B, Correlation.KendallMethod.AUTO, Correlation.Alternative.TWO_SIDED,
                        MissingPolicy.RAISE, 1.0, null, 0.0, 0.0),
                new KendallCase("perfect with ties tau-c", a(1, 1, 2), a(1, 1, 2),
                        Correlation.KendallVariant.C, Correlation.KendallMethod.AUTO, Correlation.Alternative.TWO_SIDED,
                        MissingPolicy.RAISE, 0.88888888, null, 1e-8, 0.0),
                new KendallCase("DROP missing exact perfect", withTrailingNaN(range(9)), withTrailingNaN(range(9)),
                        Correlation.KendallVariant.B, Correlation.KendallMethod.EXACT, Correlation.Alternative.TWO_SIDED,
                        MissingPolicy.DROP, 1.0, 5.511463844797e-6, 0.0, 1e-15),
                new KendallCase("DROP missing asymptotic perfect", withTrailingNaN(range(9)), withTrailingNaN(range(9)),
                        Correlation.KendallVariant.B, Correlation.KendallMethod.ASYMPTOTIC, Correlation.Alternative.TWO_SIDED,
                        MissingPolicy.DROP, 1.0, 0.00017455009626808976, 0.0, 1e-10)
        );
    }

    static Stream<KendallCase> kendallAlternativeCases() {
        double[] left = a(1, 2, 3, 4, 5);
        double[] positive = a(5, 6, 7, 8, 7);
        double[] negative = a(7, 8, 7, 6, 5);
        double statistic = 0.7378647873726218;
        double pValue = 0.07697417298126676;
        return Stream.of(
                new KendallCase("positive less", left, positive, Correlation.KendallVariant.B,
                        Correlation.KendallMethod.AUTO, Correlation.Alternative.LESS, MissingPolicy.RAISE,
                        statistic, 1.0 - pValue / 2.0, 2e-14, 2e-13),
                new KendallCase("positive greater", left, positive, Correlation.KendallVariant.B,
                        Correlation.KendallMethod.AUTO, Correlation.Alternative.GREATER, MissingPolicy.RAISE,
                        statistic, pValue / 2.0, 2e-14, 2e-13),
                new KendallCase("negative less", left, negative, Correlation.KendallVariant.B,
                        Correlation.KendallMethod.AUTO, Correlation.Alternative.LESS, MissingPolicy.RAISE,
                        -statistic, pValue / 2.0, 2e-14, 2e-13),
                new KendallCase("negative greater", left, negative, Correlation.KendallVariant.B,
                        Correlation.KendallMethod.AUTO, Correlation.Alternative.GREATER, MissingPolicy.RAISE,
                        -statistic, 1.0 - pValue / 2.0, 2e-14, 2e-13)
        );
    }

    static Stream<KendallCase> degenerateKendallCases() {
        return Stream.of(
                degenerate("all ties both tau-b", a(2, 2, 2), a(2, 2, 2), Correlation.KendallVariant.B),
                degenerate("all ties right tau-b", a(2, 0, 2), a(2, 2, 2), Correlation.KendallVariant.B),
                degenerate("all ties left tau-b", a(2, 2, 2), a(2, 0, 2), Correlation.KendallVariant.B),
                degenerate("all ties both tau-c", a(2, 2, 2), a(2, 2, 2), Correlation.KendallVariant.C),
                degenerate("all ties right tau-c", a(2, 0, 2), a(2, 2, 2), Correlation.KendallVariant.C),
                degenerate("all ties left tau-c", a(2, 2, 2), a(2, 0, 2), Correlation.KendallVariant.C)
        );
    }

    static Stream<MatrixCase> matrixCases() {
        double[] monotoneReverse = {
                1.0, 3.0,
                2.0, 2.0,
                3.0, 1.0,
                4.0, 0.0
        };
        double[] threeColumns = {
                1.0, 6.0, 1.0,
                2.0, 5.0, 3.0,
                3.0, 4.0, 2.0,
                4.0, 3.0, 6.0,
                5.0, 2.0, 4.0,
                6.0, 1.0, 5.0
        };
        List<MatrixCase> cases = new ArrayList<>();
        for (CorrelationKind kind : CorrelationKind.values()) {
            cases.add(new MatrixCase(kind + " two-column reverse", kind, monotoneReverse, 4, 2));
            cases.add(new MatrixCase(kind + " three-column mixed", kind, threeColumns, 6, 3));
        }
        return cases.stream();
    }

    static Stream<CorrelationKind> matrixKinds() {
        return Stream.of(CorrelationKind.values());
    }

    private static KendallCase degenerate(String name, double[] left, double[] right, Correlation.KendallVariant variant) {
        return new KendallCase(name, left, right, variant, Correlation.KendallMethod.AUTO,
                Correlation.Alternative.TWO_SIDED, MissingPolicy.RAISE, Double.NaN, Double.NaN, 0.0, 0.0);
    }

    private static Correlation.MatrixResult matrix(CorrelationKind kind, double[] data, int observations, int variables,
                                                   Correlation.Alternative alternative, MissingPolicy missingPolicy) {
        return switch (kind) {
            case PEARSON -> Correlation.pearson(data, observations, variables, alternative, missingPolicy);
            case SPEARMAN -> Correlation.spearman(data, observations, variables, alternative, missingPolicy);
            case KENDALL -> Correlation.kendall(data, observations, variables, alternative, missingPolicy);
        };
    }

    private static Correlation.Result scalar(CorrelationKind kind, double[] left, double[] right) {
        return switch (kind) {
            case PEARSON -> Correlation.pearson(left, right);
            case SPEARMAN -> Correlation.spearman(left, right);
            case KENDALL -> Correlation.kendall(left, right);
        };
    }

    private static void assertArrayClose(double[] expected, double[] actual, double tolerance, String message) {
        assertEquals(expected.length, actual.length, message + " length");
        for (int index = 0; index < expected.length; index++) {
            assertClose(expected[index], actual[index], tolerance, message + " index " + index);
        }
    }

    private static void assertClose(double expected, double actual, double tolerance, String message) {
        if (Double.isNaN(expected)) {
            assertTrue(Double.isNaN(actual), message);
        } else {
            assertEquals(expected, actual, tolerance, message);
        }
    }

    private static double[] column(double[] data, int observations, int variables, int column) {
        double[] out = new double[observations];
        for (int row = 0; row < observations; row++) out[row] = data[row * variables + column];
        return out;
    }

    private static double[] a(double... values) {
        return values;
    }

    private static double[] range(int length) {
        double[] out = new double[length];
        for (int index = 0; index < length; index++) out[index] = index;
        return out;
    }

    private static double[] negRange(int length) {
        double[] out = new double[length];
        for (int index = 0; index < length; index++) out[index] = -index;
        return out;
    }

    private static double[] reverseRange(int length) {
        double[] out = new double[length];
        for (int index = 0; index < length; index++) out[index] = length - 1.0 - index;
        return out;
    }

    private static double[] withTrailingNaN(double[] values) {
        double[] out = new double[values.length + 1];
        System.arraycopy(values, 0, out, 0, values.length);
        out[values.length] = Double.NaN;
        return out;
    }

    private static double[] spearmanOverflowX() {
        return range(2000);
    }

    private static double[] spearmanOverflowY() {
        double[] y = range(2000);
        swap(y, 0, 9);
        swap(y, 10, 434);
        swap(y, 435, 1509);
        return y;
    }

    private static void swap(double[] values, int left, int right) {
        double tmp = values[left];
        values[left] = values[right];
        values[right] = tmp;
    }

    private enum CorrelationKind {
        PEARSON,
        SPEARMAN,
        KENDALL
    }

    private record PearsonCase(String name, double[] left, double[] right,
                               Correlation.Alternative alternative,
                               double expectedStatistic, double expectedPValue,
                               double statTolerance, double pTolerance) {
        @Override
        public String toString() {
            return name;
        }
    }

    private record SpearmanCase(String name, double[] left, double[] right,
                                Correlation.Alternative alternative, MissingPolicy missingPolicy,
                                double expectedStatistic, Double expectedPValue,
                                double statTolerance, double pTolerance) {
        @Override
        public String toString() {
            return name;
        }
    }

    private record KendallExactCase(String name, double[] left, double[] right,
                                    Correlation.KendallVariant variant,
                                    double expectedStatistic, double expectedPValue) {
        @Override
        public String toString() {
            return name;
        }
    }

    private record KendallCase(String name, double[] left, double[] right,
                               Correlation.KendallVariant variant, Correlation.KendallMethod method,
                               Correlation.Alternative alternative, MissingPolicy missingPolicy,
                               double expectedStatistic, Double expectedPValue,
                               double statTolerance, double pTolerance) {
        @Override
        public String toString() {
            return name;
        }
    }

    private record MatrixCase(String name, CorrelationKind kind, double[] data, int observations, int variables) {
        @Override
        public String toString() {
            return name;
        }
    }

    private record WilkinsonPair(String name, double[] left, double[] right) {
        @Override
        public String toString() {
            return name;
        }
    }
}
