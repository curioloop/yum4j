package com.curioloop.yum4j.stats.test;

import com.curioloop.yum4j.linalg.blas.BLAS;
import com.curioloop.yum4j.math.VectorOps;

import java.util.Arrays;

final class TestSupport {

    private TestSupport() {}

    static void checkMatrix(double[] x, int n, int k, String name) {
        if (n <= 0) throw new IllegalArgumentException("n must be positive");
        if (k <= 0) throw new IllegalArgumentException("k must be positive");
        if (x.length != n * k) throw new IllegalArgumentException(name + " length must equal n * k");
        if (VectorOps.hasInf(x, 0, n * k)) throw new IllegalArgumentException(name + " contains NaN or infinite values");
    }

    static void checkVector(double[] x, String name) {
        if (x.length == 0) throw new IllegalArgumentException(name + " must not be empty");
        if (VectorOps.hasInf(x)) throw new IllegalArgumentException(name + " contains NaN or infinite values");
    }

    static void checkHetExog(double[] x, int n, int k, String testName) {
        checkMatrix(x, n, k, "exog");
        if (k < 2 || !hasNonZeroConstant(x, n, k)) {
            throw new IllegalArgumentException(testName + " test requires exog to have at least two columns where one is a constant");
        }
    }

    static boolean hasNonZeroConstant(double[] x, int n, int k) {
        for (int j = 0; j < k; j++) {
            double first = x[j];
            if (first == 0.0) continue;
            boolean constant = true;
            for (int i = 1; i < n; i++) {
                if (x[i * k + j] != first) {
                    constant = false;
                    break;
                }
            }
            if (constant) return true;
        }
        return false;
    }

    static double restrictedF(double restrictedSsr, double unrestrictedSsr, int restrictions, int n, int unrestrictedRank) {
        int dfDenom = n - unrestrictedRank;
        if (restrictions <= 0 || dfDenom <= 0 || unrestrictedSsr <= 0.0) return Double.NaN;
        return ((restrictedSsr - unrestrictedSsr) / restrictions) / (unrestrictedSsr / dfDenom);
    }

    static double quadraticFormInverse(double[] matrix, int matrixK, int start, double[] vector, int vectorStart, int n) {
        if (n == 1) {
            double a = matrix[start * matrixK + start];
            if (a == 0.0) return Double.NaN;
            double v = vector[vectorStart];
            return v * (v / a);
        }
        double[] a = new double[n * n];
        double[] b = new double[n];
        for (int i = 0; i < n; i++) {
            b[i] = vector[vectorStart + i];
            System.arraycopy(matrix, (start + i) * matrixK + start, a, i * n, n);
        }
        return quadraticFormInverseInPlace(a, b, vector, vectorStart, n);
    }

    private static double quadraticFormInverseInPlace(double[] a, double[] b, double[] vector, int vectorStart, int n) {
        for (int col = 0; col < n; col++) {
            int colBase = col * n;
            int pivot = col;
            double pivotAbs = Math.abs(a[colBase + col]);
            for (int row = col + 1; row < n; row++) {
                double candidate = Math.abs(a[row * n + col]);
                if (candidate > pivotAbs) {
                    pivot = row;
                    pivotAbs = candidate;
                }
            }
            if (pivotAbs == 0.0) return Double.NaN;
            if (pivot != col) {
                int pivotBase = pivot * n;
                for (int j = col; j < n; j++) {
                    double tmp = a[colBase + j];
                    a[colBase + j] = a[pivotBase + j];
                    a[pivotBase + j] = tmp;
                }
                double tmp = b[col];
                b[col] = b[pivot];
                b[pivot] = tmp;
            }
            double diag = a[colBase + col];
            double rhs = b[col];
            for (int row = col + 1; row < n; row++) {
                int rowBase = row * n;
                double factor = a[rowBase + col] / diag;
                if (factor == 0.0) continue;
                a[rowBase + col] = 0.0;
                for (int j = col + 1; j < n; j++) a[rowBase + j] -= factor * a[colBase + j];
                b[row] -= factor * rhs;
            }
        }
        // In-place back-substitution: b[row] is read (post-elimination RHS) and then
        // overwritten with the solution component. Entries b[j] for j>row already hold
        // their solution values from earlier iterations, so the recurrence still works
        // and we avoid a separate O(n) solution buffer.
        for (int row = n - 1; row >= 0; row--) {
            int rowBase = row * n;
            double sum = b[row];
            for (int j = row + 1; j < n; j++) sum -= a[rowBase + j] * b[j];
            b[row] = sum / a[rowBase + row];
        }
        return VectorOps.dot(vector, vectorStart, b, 0, n);
    }

    static double[] addTrend(double[] x, int n, int k, int trendCols, boolean prepend) {
        checkMatrix(x, n, k, "x");
        if (trendCols < 0 || trendCols > 3) throw new IllegalArgumentException("unsupported trend column count: " + trendCols);
        if (trendCols == 0) return Arrays.copyOf(x, x.length);
        double[] out = new double[n * (k + trendCols)];
        for (int i = 0; i < n; i++) {
            int base = i * (k + trendCols);
            int xOffset = prepend ? trendCols : 0;
            int trendOffset = prepend ? 0 : k;
            BLAS.dcopy(k, x, i * k, 1, out, base + xOffset, 1);
            out[base + trendOffset] = 1.0;
            if (trendCols >= 2) out[base + trendOffset + 1] = i + 1.0;
            if (trendCols >= 3) {
                double t = i + 1.0;
                out[base + trendOffset + 2] = t * t;
            }
        }
        return out;
    }

    static void copyFirstColumns(double[] x, int n, int k, int columns, double[] out) {
        if (columns < 1 || columns > k) throw new IllegalArgumentException("columns out of range");
        if (out.length < n * columns) throw new IllegalArgumentException("out length must be at least n * columns");
        for (int i = 0; i < n; i++) {
            BLAS.dcopy(columns, x, i * k, 1, out, i * columns, 1);
        }
    }

}
