/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.kalman.mle;

import com.curioloop.yum4j.optim.Univariate;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
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

    @Test
    @DisplayName("CENTRAL Hessian should match analytical quadratic Hessian")
    void testCentralMatchesQuadraticHessian() {
        double[] point = {1.25, -0.75, 0.5};

        double[] actual = NumericalHessian.central(QUADRATIC, point);

        for (int index = 0; index < EXPECTED_HESSIAN.length; index++) {
            assertThat(actual[index]).isCloseTo(EXPECTED_HESSIAN[index], within(1e-8));
        }
    }

    @Test
    @DisplayName("CENTRAL Hessian should be symmetric")
    void testCentralProducesSymmetricMatrix() {
        double[] point = {0.3, -1.2, 2.1};

        double[] actual = NumericalHessian.central(QUADRATIC, point);

        assertThat(actual[1]).isCloseTo(actual[3], within(1e-12));
        assertThat(actual[2]).isCloseTo(actual[6], within(1e-12));
        assertThat(actual[5]).isCloseTo(actual[7], within(1e-12));
    }
}