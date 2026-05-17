package com.curioloop.yum4j.ssm.kalman.filter;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ForecastErrorSolverTest {

    private static final double TOL = 1e-12;

    @Test
    void choleskyAndLuSolveAgreeForPositiveDefiniteCovariance() {
        ForecastErrorSolver.Result cholesky = solve(new ForecastErrorSolver.Method[] { ForecastErrorSolver.Method.CHOLESKY_SOLVE },
            new double[] { 4.0, 1.0, 1.0, 3.0 },
            new double[] { 2.0, -1.0 });
        ForecastErrorSolver.Result lu = solve(new ForecastErrorSolver.Method[] { ForecastErrorSolver.Method.LU_SOLVE },
            new double[] { 4.0, 1.0, 1.0, 3.0 },
            new double[] { 2.0, -1.0 });

        assertEquals(ForecastErrorSolver.Method.CHOLESKY_SOLVE, cholesky.method);
        assertEquals(ForecastErrorSolver.Method.LU_SOLVE, lu.method);
        assertEquals(Math.log(11.0), cholesky.logDeterminant, TOL);
        assertEquals(20.0 / 11.0, cholesky.quadraticForm, TOL);
        assertEquals(cholesky.logDeterminant, lu.logDeterminant, TOL);
        assertEquals(cholesky.quadraticForm, lu.quadraticForm, TOL);
    }

    @Test
    void autoInversionUsesCholeskyAndCanFallThroughToLu() {
        ForecastErrorSolver.Result auto = solve(new ForecastErrorSolver.Method[] { ForecastErrorSolver.Method.AUTO },
            new double[] { 4.0, 1.0, 1.0, 3.0 },
            new double[] { 2.0, -1.0 });
        ForecastErrorSolver.Result autoLu = solve(new ForecastErrorSolver.Method[] { ForecastErrorSolver.Method.AUTO, ForecastErrorSolver.Method.LU_SOLVE },
            new double[] { -1.0, 0.0, 0.0, -2.0 },
            new double[] { 1.0, 2.0 });

        assertEquals(ForecastErrorSolver.Method.CHOLESKY_SOLVE, auto.method);
        assertEquals(ForecastErrorSolver.Method.LU_SOLVE, autoLu.method);
        assertEquals(Math.log(2.0), autoLu.logDeterminant, TOL);
        assertEquals(-3.0, autoLu.quadraticForm, TOL);
    }

    @Test
    void choleskyFailureFallsBackToLuWhenRequested() {
        ForecastErrorSolver.Result result = solve(new ForecastErrorSolver.Method[] { ForecastErrorSolver.Method.CHOLESKY_SOLVE, ForecastErrorSolver.Method.LU_SOLVE },
            new double[] { -1.0, 0.0, 0.0, -2.0 },
            new double[] { 1.0, 2.0 });

        assertEquals(ForecastErrorSolver.Method.LU_SOLVE, result.method);
        assertEquals(Math.log(2.0), result.logDeterminant, TOL);
        assertEquals(-3.0, result.quadraticForm, TOL);
    }

    @Test
    void singularCovarianceThrowsAfterConfiguredInversionsFail() {
        assertThrows(KalmanFilter.SingularForecastException.class, () -> solve(
            new ForecastErrorSolver.Method[] { ForecastErrorSolver.Method.CHOLESKY_SOLVE, ForecastErrorSolver.Method.LU_SOLVE },
            new double[] { 1.0, 1.0, 1.0, 1.0 },
            new double[] { 1.0, 0.0 }));
    }

    @Test
    void standardizationUsesCholeskyWhitening() {
        double[] standardized = new double[2];
        boolean ok = ForecastErrorSolver.standardizeCholesky(
            new double[] { 4.0, 1.0, 1.0, 3.0 }, 0, 2,
            new double[] { 2.0, -1.0 }, 0,
            standardized, 0, 2,
            new double[4], 0.0);

        assertTrue(ok);
        assertEquals(1.0, standardized[0], TOL);
        assertEquals(-1.5 / Math.sqrt(2.75), standardized[1], TOL);
    }

    @Test
    void standardizationRejectsUnusableVariance() {
        assertTrue(Double.isNaN(ForecastErrorSolver.standardizeScalar(1.0, 0.0, 0.0)));
        assertFalse(ForecastErrorSolver.standardizeCholesky(
            new double[] { 1.0, 1.0, 1.0, 1.0 }, 0, 2,
            new double[] { 1.0, 0.0 }, 0,
            new double[2], 0, 2,
            new double[4], 0.0));
    }

    private static ForecastErrorSolver.Result solve(ForecastErrorSolver.Method[] sequence, double[] covariance, double[] error) {
        ForecastErrorSolver.Result result = new ForecastErrorSolver.Result();
        double[] factor = new double[4];
        double[] stateRhs = { 1.0, 2.0, 3.0, 4.0 };
        double[] errorRhs = error.clone();
        ForecastErrorSolver.solveMultivariate(sequence,
            factor, 0, 2,
            covariance, 0,
            stateRhs, 0, 2, 2,
            errorRhs, 0,
            error, 0,
            new int[2],
            0.0,
            result);
        return result;
    }
}
