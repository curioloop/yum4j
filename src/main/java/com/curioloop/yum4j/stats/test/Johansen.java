package com.curioloop.yum4j.stats.test;

import com.curioloop.yum4j.linalg.Decomposer;
import com.curioloop.yum4j.linalg.Regressor;
import com.curioloop.yum4j.linalg.blas.BLAS;
import com.curioloop.yum4j.linalg.mat.GEVD;
import com.curioloop.yum4j.linalg.mat.GGEVD;
import com.curioloop.yum4j.linalg.reg.OLS;
import com.curioloop.yum4j.math.VectorOps;
import java.util.Arrays;

import static com.curioloop.yum4j.linalg.Regressor.Opts.FITNESS;
import static com.curioloop.yum4j.linalg.Regressor.Opts.NO_CONST;
import static com.curioloop.yum4j.linalg.Regressor.Opts.PINV;

/**
 * Johansen cointegration test for a row-major multivariate time-series matrix.
 *
 * <p>This mirrors {@code statsmodels.tsa.vector_ar.vecm.coint_johansen}. The
 * deterministic order follows statsmodels: {@code -1} no deterministic term,
 * {@code 0} constant, and {@code 1} linear trend.</p>
 *
 * @param rkt residuals for lagged levels, row-major rows x equations
 * @param r0t residuals for differenced data, row-major rows x equations
 * @param rows effective observation count after differencing and lag trimming
 * @param equations number of endogenous variables
 * @param eigenvalues sorted eigenvalues in descending order
 * @param eigenvectors row-major equations x equations matrix; columns are eigenvectors
 * @param traceStatistic trace statistics for ranks 0..equations-1
 * @param maxEigenStatistic maximum-eigenvalue statistics for ranks 0..equations-1
 * @param traceCriticalValues row-major equations x 3 critical values at 90%, 95%, 99%
 * @param maxEigenCriticalValues row-major equations x 3 critical values at 90%, 95%, 99%
 * @param sortIndex original ascending eigenvalue indices after descending sort
 */
public record Johansen(double[] rkt, double[] r0t, int rows, int equations,
                       double[] eigenvalues, double[] eigenvectors,
                       double[] traceStatistic, double[] maxEigenStatistic,
                       double[] traceCriticalValues, double[] maxEigenCriticalValues,
                       int[] sortIndex) {

    public enum RankMethod {
        TRACE,
        MAX_EIGEN
    }

    public record Rank(int rank, int equations, double[] testStatistics,
                       double[] criticalValues, RankMethod method, double significance) {
    }

    public static Johansen test(double[] endog, int observations, int equations,
                                int detOrder, int kArDiff) {
        validateInput(endog, observations, equations, detOrder, kArDiff);

        double[] detrended = detrend(endog, observations, equations, detOrder);
        double[] differences = diff(detrended, observations, equations);
        int differenceRows = observations - 1;
        int rows = differenceRows - kArDiff;
        if (rows <= equations) {
            throw new IllegalArgumentException("insufficient observations for Johansen test");
        }

        int filterOrder = detOrder > -1 ? 0 : detOrder;
        double[] laggedDifferences = lagMatrix(differences, differenceRows, equations, kArDiff);
        if (kArDiff > 0) laggedDifferences = detrend(laggedDifferences, rows, equations * kArDiff, filterOrder);

        double[] trimmedDifferences = sliceRows(differences, differenceRows, equations, kArDiff, rows);
        trimmedDifferences = detrend(trimmedDifferences, rows, equations, filterOrder);
        double[] r0t = residualize(trimmedDifferences, rows, equations, laggedDifferences, equations * kArDiff);

        double[] laggedLevels = sliceRows(detrended, observations, equations, 1, rows);
        laggedLevels = detrend(laggedLevels, rows, equations, filterOrder);
        double[] rkt = residualize(laggedLevels, rows, equations, laggedDifferences, equations * kArDiff);

        double[] skk = crossProduct(rkt, rkt, rows, equations, equations, 1.0 / rows);
        double[] sk0 = crossProduct(rkt, r0t, rows, equations, equations, 1.0 / rows);
        double[] s00 = crossProduct(r0t, r0t, rows, equations, equations, 1.0 / rows);
        double[] sig = sigMatrix(sk0, s00, equations);
        symmetrize(sig, equations);
        symmetrize(skk, equations);

        EigenResult eig = generalizedEigen(sig, skk, equations);
        sortDescending(eig.eigenvalues, eig.eigenvectors, eig.sortIndex, equations);
        normalizeEigenvectorSign(eig.eigenvectors);

        double[] traceStatistic = new double[equations];
        double[] maxEigenStatistic = new double[equations];
        double[] traceCriticalValues = new double[equations * 3];
        double[] maxEigenCriticalValues = new double[equations * 3];
        for (int rank = 0; rank < equations; rank++) {
            double trace = 0.0;
            for (int index = rank; index < equations; index++) trace += Math.log(1.0 - eig.eigenvalues[index]);
            traceStatistic[rank] = -rows * trace;
            maxEigenStatistic[rank] = -rows * Math.log(1.0 - eig.eigenvalues[rank]);
            System.arraycopy(JohansenCriticalValues.trace(equations - rank, detOrder), 0,
                    traceCriticalValues, rank * 3, 3);
            System.arraycopy(JohansenCriticalValues.maxEigen(equations - rank, detOrder), 0,
                    maxEigenCriticalValues, rank * 3, 3);
        }

        return new Johansen(rkt, r0t, rows, equations, eig.eigenvalues, eig.eigenvectors,
                traceStatistic, maxEigenStatistic, traceCriticalValues, maxEigenCriticalValues, eig.sortIndex);
    }

    public static Rank selectRank(double[] endog, int observations, int equations,
                                  int detOrder, int kArDiff, RankMethod method, double significance) {
        RankMethod resolvedMethod = method == null ? RankMethod.TRACE : method;
        int criticalIndex = criticalIndex(significance);
        Johansen result = test(endog, observations, equations, detOrder, kArDiff);
        double[] statistics = resolvedMethod == RankMethod.TRACE ? result.traceStatistic : result.maxEigenStatistic;
        double[] criticalMatrix = resolvedMethod == RankMethod.TRACE ? result.traceCriticalValues : result.maxEigenCriticalValues;

        int rank = 0;
        while (rank < equations && statistics[rank] >= criticalMatrix[rank * 3 + criticalIndex]) rank++;
        int outputLength = Math.min(rank + 1, equations);
        double[] selectedStatistics = Arrays.copyOf(statistics, outputLength);
        double[] selectedCriticalValues = new double[outputLength];
        for (int index = 0; index < outputLength; index++) {
            selectedCriticalValues[index] = criticalMatrix[index * 3 + criticalIndex];
        }
        return new Rank(rank, equations, selectedStatistics, selectedCriticalValues, resolvedMethod, significance);
    }

    public double traceCriticalValue(int rank, int criticalIndex) {
        return traceCriticalValues[rank * 3 + criticalIndex];
    }

    public double maxEigenCriticalValue(int rank, int criticalIndex) {
        return maxEigenCriticalValues[rank * 3 + criticalIndex];
    }

    private static void validateInput(double[] endog, int observations, int equations, int detOrder, int kArDiff) {
        TestSupport.checkMatrix(endog, observations, equations, "endog");
        if (equations < 2) throw new IllegalArgumentException("Johansen test requires at least two variables");
        if (detOrder < -1 || detOrder > 1) throw new IllegalArgumentException("detOrder must be -1, 0, or 1");
        if (kArDiff < 0) throw new IllegalArgumentException("kArDiff must be non-negative");
        if (observations - 1 - kArDiff <= 0) throw new IllegalArgumentException("kArDiff leaves no observations");
    }

    private static int criticalIndex(double significance) {
        if (significance == 0.10) return 0;
        if (significance == 0.05) return 1;
        if (significance == 0.01) return 2;
        throw new IllegalArgumentException("significance must be 0.10, 0.05, or 0.01");
    }

    private static double[] detrend(double[] matrix, int rows, int columns, int order) {
        if (order == -1 || columns == 0) return matrix.clone();
        if (order < -1) throw new IllegalArgumentException("unsupported trend order");
        int trendColumns = order + 1;
        double[] trend = new double[rows * trendColumns];
        for (int row = 0; row < rows; row++) {
            double location = rows == 1 ? 0.0 : -1.0 + 2.0 * row / (rows - 1.0);
            for (int column = 0; column < trendColumns; column++) {
                int power = order - column;
                trend[row * trendColumns + column] = power == 0 ? 1.0 : Math.pow(location, power);
            }
        }
        double[] residual = new double[rows * columns];
        var pool = new OLS.Pool();
        for (int column = 0; column < columns; column++) {
            double[] response = column(matrix, rows, columns, column);
            var ols = Regressor.ols(response, trend.clone(), rows, trendColumns, pool, PINV, FITNESS);
            double[] columnResidual = ols.residual(false);
            BLAS.dcopy(rows, columnResidual, 0, 1, residual, column, columns);
        }
        return residual;
    }

    private static double[] diff(double[] matrix, int rows, int columns) {
        double[] out = new double[(rows - 1) * columns];
        for (int row = 0; row < rows - 1; row++) {
            VectorOps.axpyTo(out, row * columns, -1.0,
                    matrix, row * columns, matrix, (row + 1) * columns, columns);
        }
        return out;
    }

    private static double[] lagMatrix(double[] matrix, int rows, int columns, int lagCount) {
        int outputRows = rows - lagCount;
        int outputColumns = columns * lagCount;
        double[] out = new double[outputRows * outputColumns];
        for (int row = 0; row < outputRows; row++) {
            int sourceRow = row + lagCount;
            for (int lag = 1; lag <= lagCount; lag++) {
                System.arraycopy(matrix, (sourceRow - lag) * columns,
                        out, row * outputColumns + (lag - 1) * columns, columns);
            }
        }
        return out;
    }

    private static double[] residualize(double[] response, int rows, int responseColumns,
                                        double[] regressors, int regressorColumns) {
        if (regressorColumns == 0) return response.clone();
        double[] residual = new double[rows * responseColumns];
        var pool = new OLS.Pool();
        for (int column = 0; column < responseColumns; column++) {
            double[] responseColumn = column(response, rows, responseColumns, column);
            var ols = Regressor.ols(responseColumn, regressors.clone(), rows, regressorColumns, pool,
                    PINV, FITNESS, NO_CONST);
            double[] columnResidual = ols.residual(false);
            BLAS.dcopy(rows, columnResidual, 0, 1, residual, column, responseColumns);
        }
        return residual;
    }

    private static double[] sigMatrix(double[] sk0, double[] s00, int size) {
        double[] s00Inverse = Decomposer.lu(s00.clone(), size).inverse(null);
        double[] temp = multiply(sk0, s00Inverse, size, size, size);
        return multiplyTransposeRight(temp, sk0, size, size, size);
    }

    private static EigenResult generalizedEigen(double[] sig, double[] skk, int size) {
        double[] a = sig.clone();
        double[] b = skk.clone();
        GEVD gevd = Decomposer.gevd(a, b, size, GEVD.Opts.WANT_V);
        if (gevd.ok()) {
            double[] eigenvalues = Arrays.copyOf(gevd.eigenvalues(), size);
            double[] eigenvectors = gevd.toV().data.clone();
            int[] sortIndex = new int[size];
            for (int index = 0; index < size; index++) sortIndex[index] = index;
            return new EigenResult(eigenvalues, eigenvectors, sortIndex);
        }

        double[] work = new double[size];
        GGEVD ggevd = Decomposer.ggevd(sig.clone(), skk.clone(), size, GGEVD.Opts.WANT_VR);
        if (!ggevd.ok()) throw new ArithmeticException("Johansen generalized eigenvalue decomposition failed");
        double[] complexEigenvalues = ggevd.eigenvalues();
        double[] eigenvalues = new double[size];
        for (int index = 0; index < size; index++) {
            if (Math.abs(complexEigenvalues[index * 2 + 1]) > 1e-10) {
                throw new ArithmeticException("Johansen eigenvalues must be real");
            }
            eigenvalues[index] = complexEigenvalues[index * 2];
        }
        double[] eigenvectors = ggevd.toVR().data.clone();
        normalizeColumnsWithMetric(eigenvectors, skk, work, size);
        int[] sortIndex = new int[size];
        for (int index = 0; index < size; index++) sortIndex[index] = index;
        return new EigenResult(eigenvalues, eigenvectors, sortIndex);
    }

    private static void sortDescending(double[] eigenvalues, double[] eigenvectors, int[] sortIndex, int size) {
        int[] indices = new int[size];
        for (int index = 0; index < size; index++) indices[index] = index;
        for (int index = 0; index < size - 1; index++) {
            int best = index;
            for (int candidate = index + 1; candidate < size; candidate++) {
                if (eigenvalues[indices[candidate]] > eigenvalues[indices[best]]) best = candidate;
            }
            int swap = indices[index];
            indices[index] = indices[best];
            indices[best] = swap;
        }
        double[] sortedValues = new double[size];
        double[] sortedVectors = new double[size * size];
        for (int column = 0; column < size; column++) {
            int source = indices[column];
            sortedValues[column] = eigenvalues[source];
            for (int row = 0; row < size; row++) sortedVectors[row * size + column] = eigenvectors[row * size + source];
        }
        System.arraycopy(sortedValues, 0, eigenvalues, 0, size);
        System.arraycopy(sortedVectors, 0, eigenvectors, 0, size * size);
        System.arraycopy(indices, 0, sortIndex, 0, size);
    }

    private static void normalizeEigenvectorSign(double[] eigenvectors) {
        for (double value : eigenvectors) {
            if (value != 0.0) {
                if (value < 0.0) {
                    VectorOps.scal(-1.0, eigenvectors);
                }
                return;
            }
        }
    }

    private static void normalizeColumnsWithMetric(double[] eigenvectors, double[] metric,
                                                   double[] work, int size) {
        for (int column = 0; column < size; column++) {
            BLAS.dgemv(BLAS.Trans.NoTrans, size, size, 1.0,
                    metric, 0, size, eigenvectors, column, size, 0.0, work, 0, 1);
            double normSquared = BLAS.ddot(size, eigenvectors, column, size, work, 0, 1);
            double scale = Math.sqrt(Math.abs(normSquared));
            if (scale != 0.0) {
                BLAS.dscal(size, 1.0 / scale, eigenvectors, column, size);
            }
        }
    }

    private static double[] crossProduct(double[] left, double[] right, int rows,
                                         int leftColumns, int rightColumns, double scale) {
        double[] out = new double[leftColumns * rightColumns];
        BLAS.dgemm(BLAS.Trans.Trans, BLAS.Trans.NoTrans, leftColumns, rightColumns, rows,
                scale, left, 0, leftColumns, right, 0, rightColumns, 0.0, out, 0, rightColumns);
        return out;
    }

    private static double[] multiply(double[] left, double[] right, int rows, int shared, int columns) {
        double[] out = new double[rows * columns];
        BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, rows, columns, shared,
                1.0, left, 0, shared, right, 0, columns, 0.0, out, 0, columns);
        return out;
    }

    private static double[] multiplyTransposeRight(double[] left, double[] right, int rows, int shared, int rightRows) {
        double[] out = new double[rows * rightRows];
        BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.Trans, rows, rightRows, shared,
                1.0, left, 0, shared, right, 0, shared, 0.0, out, 0, rightRows);
        return out;
    }

    private static double[] sliceRows(double[] matrix, int rows, int columns, int startRow, int rowCount) {
        if (startRow < 0 || rowCount < 0 || startRow + rowCount > rows) {
            throw new IllegalArgumentException("row slice out of range");
        }
        double[] out = new double[rowCount * columns];
        for (int row = 0; row < rowCount; row++) {
            System.arraycopy(matrix, (startRow + row) * columns, out, row * columns, columns);
        }
        return out;
    }

    private static double[] column(double[] matrix, int rows, int columns, int column) {
        double[] out = new double[rows];
        BLAS.dcopy(rows, matrix, column, columns, out, 0, 1);
        return out;
    }

    private static void symmetrize(double[] matrix, int size) {
        for (int row = 0; row < size; row++) {
            for (int column = row + 1; column < size; column++) {
                double value = 0.5 * (matrix[row * size + column] + matrix[column * size + row]);
                matrix[row * size + column] = value;
                matrix[column * size + row] = value;
            }
        }
    }

    private record EigenResult(double[] eigenvalues, double[] eigenvectors, int[] sortIndex) {
    }
}