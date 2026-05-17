package com.curioloop.yum4j.optim;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.util.Arrays;
import java.util.concurrent.atomic.AtomicInteger;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

class NumericalHessianTest {

    private static final Univariate.Objective QUADRATIC = (x, n) ->
            x[0] * x[0]
                    + 2.0 * x[1] * x[1]
                    + 3.0 * x[2] * x[2]
                    + 0.5 * x[0] * x[1]
                    - 0.25 * x[1] * x[2];

    private static final double[] EXPECTED_HESSIAN = {
            2.0, 0.5, 0.0,
            0.5, 4.0, -0.25,
            0.0, -0.25, 6.0
    };

    @ParameterizedTest
    @EnumSource(NumericalHessian.class)
    @DisplayName("All Hessian methods should match analytical quadratic Hessian")
    void testAllMethodsMatchQuadraticHessian(NumericalHessian method) {
        double[] point = {1.25, -0.75, 0.5};

        double[] actual = compute(method, QUADRATIC, point);

        assertArrayCloseTo(actual, EXPECTED_HESSIAN, method == NumericalHessian.CENTRAL ? 1e-8 : 1e-4);
    }

    @ParameterizedTest
    @EnumSource(NumericalHessian.class)
    @DisplayName("All Hessian methods should produce symmetric matrices")
    void testAllMethodsProduceSymmetricMatrix(NumericalHessian method) {
        double[] point = {0.3, -1.2, 2.1};

        double[] actual = compute(method, QUADRATIC, point);

        int n = point.length;

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                assertThat(actual[i * n + j]).isCloseTo(actual[j * n + i], within(1e-12));
            }
        }
    }

    @Test
    @DisplayName("Enum constants should expose named Hessian methods")
    void testEnumConstants() {
        assertThat(NumericalHessian.values()).containsExactly(
                NumericalHessian.FORWARD,
                NumericalHessian.AVERAGE,
                NumericalHessian.CENTRAL);
    }

    @Test
    @DisplayName("Mutable compute should restore input point")
    void testMutableComputeRestoresInputPoint() {
        double[] point = {1.25, -0.75, 0.5, 99.0};
        double[] before = point.clone();
        double[] actual = new double[9];

        NumericalHessian.CENTRAL.compute(QUADRATIC, point, 3, actual);

        assertThat(point).containsExactly(before);
        assertArrayCloseTo(actual, EXPECTED_HESSIAN, 1e-7);
    }

    @Test
    @DisplayName("Mutable compute should reuse caller-provided Hessian buffer")
    void testMutableComputeReusesHessianBuffer() {
        double[] point = {1.25, -0.75, 0.5};
        double[] before = point.clone();
        double[] actual = new double[9];

        NumericalHessian.AVERAGE.compute(QUADRATIC, point, point.length, actual);

        assertThat(point).containsExactly(before);
        assertArrayCloseTo(actual, EXPECTED_HESSIAN, 1e-4);
    }

    @Test
    @DisplayName("AVERAGE should handle one-dimensional objectives")
    void testHess2OneDimensionalObjective() {
        Univariate.Objective objective = (x, n) -> 1.5 * x[0] * x[0] - 0.25 * x[0];

        double[] actual = compute(NumericalHessian.AVERAGE, objective, new double[]{2.0});

        assertArrayCloseTo(actual, new double[]{3.0}, 1e-4);
    }

    @Test
    @DisplayName("AVERAGE should handle two-dimensional objectives")
    void testHess2TwoDimensionalObjective() {
        Univariate.Objective objective = (x, n) -> 1.5 * x[0] * x[0]
                - 0.25 * x[0] * x[1]
                + 2.0 * x[1] * x[1];

        double[] actual = compute(NumericalHessian.AVERAGE, objective, new double[]{2.0, -1.0});

        assertArrayCloseTo(actual, new double[]{3.0, -0.25, -0.25, 4.0}, 1e-4);
    }

    @Test
    @DisplayName("Statsmodels methods should use expected function evaluation counts")
    void testEvaluationCounts() {
        double[] point = {1.0, 2.0, 3.0};

        assertThat(countEvaluations(NumericalHessian.FORWARD, point)).isEqualTo(10);
        assertThat(countEvaluations(NumericalHessian.AVERAGE, point)).isEqualTo(19);
        assertThat(countEvaluations(NumericalHessian.CENTRAL, point)).isEqualTo(24);
    }

    @Test
    @DisplayName("Invalid inputs should fail fast")
    void testInvalidInputs() {
        double[] point = {1.0, 2.0};
        double[] hessian = new double[4];

        assertThatThrownBy(() -> NumericalHessian.CENTRAL.compute(null, point, point.length, hessian))
                .isInstanceOf(NullPointerException.class);
        assertThatThrownBy(() -> NumericalHessian.CENTRAL.compute(QUADRATIC, point, 3, hessian))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("n out of range");
        assertThatThrownBy(() -> NumericalHessian.CENTRAL.compute(QUADRATIC, point, 2, new double[3]))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("hessian output too small");
    }

    @Test
    @DisplayName("Mutable compute should restore input when objective throws")
    void testMutableComputeRestoresInputWhenObjectiveThrows() {
        double[] point = {1.0, 2.0};
        double[] before = point.clone();
        double[] hessian = new double[4];
        Univariate.Objective throwing = (x, n) -> {
            if (!Arrays.equals(x, before)) {
                throw new IllegalStateException("shifted evaluation failed");
            }
            return 0.0;
        };

        assertThatThrownBy(() -> NumericalHessian.CENTRAL.compute(throwing, point, 2, hessian))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("shifted evaluation failed");
        assertThat(point).containsExactly(before);
    }

    private static int countEvaluations(NumericalHessian method, double[] point) {
        AtomicInteger count = new AtomicInteger();
        Univariate.Objective counted = (x, n) -> {
            count.incrementAndGet();
            return Arrays.stream(x, 0, n).map(value -> value * value).sum();
        };
        compute(method, counted, point);
        return count.get();
    }

    private static double[] compute(NumericalHessian method, Univariate.Objective objective, double[] point) {
        double[] hessian = new double[point.length * point.length];
        method.compute(objective, point, point.length, hessian);
        return hessian;
    }

    private static void assertArrayCloseTo(double[] actual, double[] expected, double tolerance) {
        assertThat(actual).hasSize(expected.length);
        for (int index = 0; index < expected.length; index++) {
            assertThat(actual[index]).isCloseTo(expected[index], within(tolerance));
        }
    }
}