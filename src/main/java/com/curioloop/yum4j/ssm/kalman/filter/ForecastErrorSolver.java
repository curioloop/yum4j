package com.curioloop.yum4j.ssm.kalman.filter;

import com.curioloop.yum4j.linalg.blas.BLAS;

final class ForecastErrorSolver {

    enum Method {
        AUTO,
        CHOLESKY_SOLVE,
        LU_SOLVE,
        UNIVARIATE
    }

    static final class Result {
        double logDeterminant;
        double quadraticForm;
        Method method;

        void clear() {
            logDeterminant = 0.0;
            quadraticForm = 0.0;
            method = null;
        }
    }

    private ForecastErrorSolver() {
    }

    static void solveScalar(double error,
                            double variance,
                            double[] predictedState,
                            int predictedStateOffset,
                            double[] predictedStateCov,
                            int predictedStateCovOffset,
                            double[] pzt,
                            int pztOffset,
                            int stateCount,
                            double[] transition,
                            int transitionOffset,
                            int transitionStride,
                            double[] filteredState,
                            int filteredStateOffset,
                            double[] filteredStateCov,
                            int filteredStateCovOffset,
                            double[] kalmanGain,
                            int kalmanGainOffset,
                            boolean storeGain,
                            double tolerance,
                            Result result) {
        result.clear();
        if (!usableVariance(variance, tolerance)) {
            BLAS.dcopy(stateCount, predictedState, predictedStateOffset, 1,
                filteredState, filteredStateOffset, 1);
            BLAS.dcopy(stateCount * stateCount, predictedStateCov, predictedStateCovOffset, 1,
                filteredStateCov, filteredStateCovOffset, 1);
            if (storeGain) {
                for (int row = 0; row < stateCount; row++) {
                    kalmanGain[kalmanGainOffset + row] = 0.0;
                }
            }
            result.method = Method.UNIVARIATE;
            return;
        }

        double varianceInverse = 1.0 / variance;
        result.logDeterminant = Math.log(variance);
        result.quadraticForm = error * varianceInverse * error;
        result.method = Method.UNIVARIATE;

        BLAS.dcopy(stateCount, predictedState, predictedStateOffset, 1,
            filteredState, filteredStateOffset, 1);
        BLAS.daxpy(stateCount, error * varianceInverse,
            pzt, pztOffset, 1, filteredState, filteredStateOffset, 1);

        BLAS.dcopy(stateCount * stateCount, predictedStateCov, predictedStateCovOffset, 1,
            filteredStateCov, filteredStateCovOffset, 1);
        BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.Trans, stateCount, stateCount, 1,
            -varianceInverse, pzt, pztOffset, 1,
            pzt, pztOffset, 1,
            1.0, filteredStateCov, filteredStateCovOffset, stateCount);

        if (storeGain) {
            BLAS.dscal(stateCount, varianceInverse, pzt, pztOffset, 1);
            BLAS.dgemv(BLAS.Trans.NoTrans, stateCount, stateCount, 1.0,
                transition, transitionOffset, transitionStride,
                pzt, pztOffset, 1,
                0.0, kalmanGain, kalmanGainOffset, 1);
        }
    }

    static void solveMultivariate(Method[] sequence,
                                  double[] covariance,
                                  int covarianceOffset,
                                  int dimension,
                                  double[] originalCovariance,
                                  int originalCovarianceOffset,
                                  double[] stateRhs,
                                  int stateRhsOffset,
                                  int stateRhsCount,
                                  int stateRhsStride,
                                  double[] errorRhs,
                                  int errorRhsOffset,
                                  double[] originalError,
                                  int originalErrorOffset,
                                  int[] pivots,
                                  double tolerance,
                                  Result result) {
        solveMultivariate(sequence,
            covariance,
            covarianceOffset,
            dimension,
            originalCovariance,
            originalCovarianceOffset,
            stateRhs,
            stateRhsOffset,
            stateRhsCount,
            stateRhsStride,
            errorRhs,
            errorRhsOffset,
            originalError,
            originalErrorOffset,
            pivots,
            0,
            tolerance,
            result);
    }

    static void solveMultivariate(Method[] sequence,
                                  double[] covariance,
                                  int covarianceOffset,
                                  int dimension,
                                  double[] originalCovariance,
                                  int originalCovarianceOffset,
                                  double[] stateRhs,
                                  int stateRhsOffset,
                                  int stateRhsCount,
                                  int stateRhsStride,
                                  double[] errorRhs,
                                  int errorRhsOffset,
                                  double[] originalError,
                                  int originalErrorOffset,
                                  int[] pivots,
                                  int pivotOffset,
                                  double tolerance,
                                  Result result) {
        result.clear();
        boolean defaultSequence = sequence == null || sequence.length == 0;
        int attemptCount = defaultSequence ? 1 : sequence.length;
        for (int attempt = 0; attempt < attemptCount; attempt++) {
            Method method = defaultSequence ? Method.AUTO : sequence[attempt];
            Method actual = method == Method.AUTO ? Method.CHOLESKY_SOLVE : method;
            BLAS.dcopy(dimension * dimension,
                originalCovariance, originalCovarianceOffset, 1,
                covariance, covarianceOffset, 1);
            if (actual == Method.CHOLESKY_SOLVE) {
                if (solveWithCholesky(covariance, covarianceOffset, dimension,
                    stateRhs, stateRhsOffset, stateRhsCount, stateRhsStride,
                    errorRhs, errorRhsOffset,
                    originalError, originalErrorOffset,
                    tolerance, result)) {
                    result.method = Method.CHOLESKY_SOLVE;
                    return;
                }
            } else if (actual == Method.LU_SOLVE) {
                if (pivots == null || pivots.length < pivotOffset + dimension) {
                    throw new IllegalArgumentException("LU forecast-error solve requires pivot workspace of length " + dimension);
                }
                if (solveWithLu(covariance, covarianceOffset, dimension,
                    stateRhs, stateRhsOffset, stateRhsCount, stateRhsStride,
                    errorRhs, errorRhsOffset,
                    originalError, originalErrorOffset,
                    pivots, pivotOffset,
                    tolerance, result)) {
                    result.method = Method.LU_SOLVE;
                    return;
                }
            } else if (actual == Method.UNIVARIATE) {
                throw new KalmanFilter.SingularForecastException("Forecast-error covariance is singular for conventional solving; rerun with univariate filtering");
            } else {
                throw new UnsupportedOperationException(actual + " forecast-error solving is not implemented");
            }
        }
        throw new KalmanFilter.SingularForecastException("Forecast-error covariance is singular for inversion sequence");
    }

    static double standardizeScalar(double error, double variance, double tolerance) {
        return usableVariance(variance, tolerance) && Double.isFinite(error)
            ? error / Math.sqrt(variance)
            : Double.NaN;
    }

    static boolean standardizeCholesky(double[] covariance,
                                       int covarianceOffset,
                                       int covarianceStride,
                                       double[] error,
                                       int errorOffset,
                                       double[] standardized,
                                       int standardizedOffset,
                                       int dimension,
                                       double[] factor,
                                       double tolerance) {
        return standardizeCholesky(covariance,
            covarianceOffset,
            covarianceStride,
            error,
            errorOffset,
            standardized,
            standardizedOffset,
            dimension,
            factor,
            0,
            tolerance);
    }

    static boolean standardizeCholesky(double[] covariance,
                                       int covarianceOffset,
                                       int covarianceStride,
                                       double[] error,
                                       int errorOffset,
                                       double[] standardized,
                                       int standardizedOffset,
                                       int dimension,
                                       double[] factor,
                                       int factorOffset,
                                       double tolerance) {
        for (int row = 0; row < dimension; row++) {
            System.arraycopy(covariance,
                covarianceOffset + row * covarianceStride,
                factor,
                factorOffset + row * dimension,
                dimension);
        }
        if (!choleskyLowerInPlace(factor, factorOffset, dimension, tolerance)) {
            return false;
        }
        for (int row = 0; row < dimension; row++) {
            double value = error[errorOffset + row];
            if (!Double.isFinite(value)) {
                return false;
            }
            for (int col = 0; col < row; col++) {
                value -= factor[factorOffset + row * dimension + col] * standardized[standardizedOffset + col];
            }
            standardized[standardizedOffset + row] = value / factor[factorOffset + row * dimension + row];
        }
        return true;
    }

    static boolean choleskyLowerInPlace(double[] matrix, int dimension, double tolerance) {
        return choleskyLowerInPlace(matrix, 0, dimension, tolerance);
    }

    static boolean choleskyLowerInPlace(double[] matrix, int offset, int dimension, double tolerance) {
        if (BLAS.dpotrf(BLAS.Uplo.Lower, dimension, matrix, offset, dimension) != 0) {
            return false;
        }
        for (int row = 0; row < dimension; row++) {
            double diagonal = matrix[offset + row * dimension + row];
            if (!usableVariance(diagonal * diagonal, tolerance)) {
                return false;
            }
            for (int col = row + 1; col < dimension; col++) {
                matrix[offset + row * dimension + col] = 0.0;
            }
        }
        return true;
    }

    private static boolean solveWithCholesky(double[] covariance,
                                             int covarianceOffset,
                                             int dimension,
                                             double[] stateRhs,
                                             int stateRhsOffset,
                                             int stateRhsCount,
                                             int stateRhsStride,
                                             double[] errorRhs,
                                             int errorRhsOffset,
                                             double[] originalError,
                                             int originalErrorOffset,
                                             double tolerance,
                                             Result result) {
        if (BLAS.dpotrf(BLAS.Uplo.Lower, dimension, covariance, covarianceOffset, dimension) != 0) {
            return false;
        }
        double logDeterminant = 0.0;
        for (int row = 0; row < dimension; row++) {
            double diagonal = covariance[covarianceOffset + row * dimension + row];
            if (!usableVariance(diagonal * diagonal, tolerance)) {
                return false;
            }
            logDeterminant += Math.log(diagonal);
        }
        BLAS.dpotrs(BLAS.Uplo.Lower, dimension, stateRhsCount,
            covariance, covarianceOffset, dimension,
            stateRhs, stateRhsOffset, stateRhsStride);
        BLAS.dpotrs(BLAS.Uplo.Lower, dimension, 1,
            covariance, covarianceOffset, dimension,
            errorRhs, errorRhsOffset, 1);
        result.logDeterminant = 2.0 * logDeterminant;
        result.quadraticForm = BLAS.ddot(dimension,
            originalError, originalErrorOffset, 1,
            errorRhs, errorRhsOffset, 1);
        return Double.isFinite(result.logDeterminant) && Double.isFinite(result.quadraticForm);
    }

    private static boolean solveWithLu(double[] covariance,
                                       int covarianceOffset,
                                       int dimension,
                                       double[] stateRhs,
                                       int stateRhsOffset,
                                       int stateRhsCount,
                                       int stateRhsStride,
                                       double[] errorRhs,
                                       int errorRhsOffset,
                                       double[] originalError,
                                       int originalErrorOffset,
                                       int[] pivots,
                                       int pivotOffset,
                                       double tolerance,
                                       Result result) {
        if (BLAS.dgetrf(dimension, dimension, covariance, covarianceOffset, dimension, pivots, pivotOffset) != 0) {
            return false;
        }
        double logAbsDeterminant = 0.0;
        int sign = 1;
        for (int row = 0; row < dimension; row++) {
            if (pivots[pivotOffset + row] != row) {
                sign = -sign;
            }
            double diagonal = covariance[covarianceOffset + row * dimension + row];
            if (!usableVariance(Math.abs(diagonal), tolerance)) {
                return false;
            }
            if (diagonal < 0.0) {
                sign = -sign;
            }
            logAbsDeterminant += Math.log(Math.abs(diagonal));
        }
        if (sign <= 0 || !Double.isFinite(logAbsDeterminant)) {
            return false;
        }
        BLAS.dgetrs(BLAS.Trans.NoTrans, dimension, stateRhsCount,
            covariance, covarianceOffset, dimension,
            pivots, pivotOffset,
            stateRhs, stateRhsOffset, stateRhsStride);
        BLAS.dgetrs(BLAS.Trans.NoTrans, dimension, 1,
            covariance, covarianceOffset, dimension,
            pivots, pivotOffset,
            errorRhs, errorRhsOffset, 1);
        result.logDeterminant = logAbsDeterminant;
        result.quadraticForm = BLAS.ddot(dimension,
            originalError, originalErrorOffset, 1,
            errorRhs, errorRhsOffset, 1);
        return Double.isFinite(result.quadraticForm);
    }

    private static boolean usableVariance(double value, double tolerance) {
        return value > tolerance && Double.isFinite(value);
    }
}