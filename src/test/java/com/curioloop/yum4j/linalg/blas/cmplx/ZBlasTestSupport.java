/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.linalg.blas.cmplx;

import com.curioloop.yum4j.linalg.blas.BLAS;

import java.util.Random;

final class ZBlasTestSupport {

    private ZBlasTestSupport() {
    }

    static double[] randomComplexVector(int n, long seed) {
        Random random = new Random(seed);
        double[] vector = new double[n * 2];
        for (int i = 0; i < n; i++) {
            int index = i * 2;
            vector[index] = random.nextDouble() * 2.0 - 1.0;
            vector[index + 1] = random.nextDouble() * 2.0 - 1.0;
        }
        return vector;
    }

    static double[] randomComplexMatrix(int rows, int cols, long seed) {
        Random random = new Random(seed);
        double[] matrix = new double[rows * cols * 2];
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                int index = matrixIndex(0, cols, row, col);
                matrix[index] = random.nextDouble() * 2.0 - 1.0;
                matrix[index + 1] = random.nextDouble() * 2.0 - 1.0;
            }
        }
        return matrix;
    }

    static double[] randomHermitianMatrix(int n, long seed) {
        Random random = new Random(seed);
        double[] matrix = new double[n * n * 2];
        for (int i = 0; i < n; i++) {
            int diag = matrixIndex(0, n, i, i);
            matrix[diag] = random.nextDouble() * 2.0 - 1.0;
            matrix[diag + 1] = 0.0;
            for (int j = i + 1; j < n; j++) {
                double real = random.nextDouble() * 2.0 - 1.0;
                double imag = random.nextDouble() * 2.0 - 1.0;
                int upper = matrixIndex(0, n, i, j);
                int lower = matrixIndex(0, n, j, i);
                matrix[upper] = real;
                matrix[upper + 1] = imag;
                matrix[lower] = real;
                matrix[lower + 1] = -imag;
            }
        }
        return matrix;
    }

    static double[] randomSymmetricMatrix(int n, long seed) {
        Random random = new Random(seed);
        double[] matrix = new double[n * n * 2];
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                double real = random.nextDouble() * 2.0 - 1.0;
                double imag = random.nextDouble() * 2.0 - 1.0;
                int upper = matrixIndex(0, n, i, j);
                int lower = matrixIndex(0, n, j, i);
                matrix[upper] = real;
                matrix[upper + 1] = imag;
                matrix[lower] = real;
                matrix[lower + 1] = imag;
            }
        }
        return matrix;
    }

    static double[] randomTriangularMatrix(int n, BLAS.Uplo uplo, boolean unitDiag, long seed) {
        Random random = new Random(seed);
        double[] matrix = new double[n * n * 2];
        boolean upper = uplo == BLAS.Uplo.Upper;
        for (int row = 0; row < n; row++) {
            for (int col = 0; col < n; col++) {
                boolean stored = upper ? col >= row : col <= row;
                if (!stored) {
                    continue;
                }
                int index = matrixIndex(0, n, row, col);
                if (row == col) {
                    matrix[index] = unitDiag ? 1.0 : random.nextDouble() + 0.5;
                    matrix[index + 1] = unitDiag ? 0.0 : random.nextDouble() * 2.0 - 1.0;
                } else {
                    matrix[index] = random.nextDouble() * 2.0 - 1.0;
                    matrix[index + 1] = random.nextDouble() * 2.0 - 1.0;
                }
            }
        }
        return matrix;
    }

    static void fillUsedComplexVector(Random random, double[] vector, int start, int inc, int n) {
        for (int i = 0; i < n; i++) {
            int index = vectorIndex(start, inc, i);
            vector[index] = random.nextDouble() * 2.0 - 1.0;
            vector[index + 1] = random.nextDouble() * 2.0 - 1.0;
        }
    }

    static void embedMatrix(double[] src, int rows, int cols, int srcStart, int srcLd,
                            double[] dst, int dstStart, int dstLd) {
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                int srcIndex = matrixIndex(srcStart, srcLd, row, col);
                int dstIndex = matrixIndex(dstStart, dstLd, row, col);
                dst[dstIndex] = src[srcIndex];
                dst[dstIndex + 1] = src[srcIndex + 1];
            }
        }
    }

    static void refZhemv(BLAS.Uplo uplo, int n, double alphaRe, double alphaIm,
                         double[] a, int aStart, int lda,
                         double[] x, int xStart, int incX,
                         double betaRe, double betaIm,
                         double[] y, int yStart, int incY) {
        boolean upper = uplo == BLAS.Uplo.Upper;
        for (int i = 0; i < n; i++) {
            double sumRe = 0.0;
            double sumIm = 0.0;
            for (int j = 0; j < n; j++) {
                double aRe;
                double aIm;
                if (upper) {
                    if (j < i) {
                        int index = matrixIndex(aStart, lda, j, i);
                        aRe = a[index];
                        aIm = -a[index + 1];
                    } else {
                        int index = matrixIndex(aStart, lda, i, j);
                        aRe = a[index];
                        aIm = j == i ? 0.0 : a[index + 1];
                    }
                } else {
                    if (j > i) {
                        int index = matrixIndex(aStart, lda, j, i);
                        aRe = a[index];
                        aIm = -a[index + 1];
                    } else {
                        int index = matrixIndex(aStart, lda, i, j);
                        aRe = a[index];
                        aIm = j == i ? 0.0 : a[index + 1];
                    }
                }
                int xIndex = vectorIndex(xStart, incX, j);
                double xRe = x[xIndex];
                double xIm = x[xIndex + 1];
                sumRe += aRe * xRe - aIm * xIm;
                sumIm += aRe * xIm + aIm * xRe;
            }

            int yIndex = vectorIndex(yStart, incY, i);
            double yRe = y[yIndex];
            double yIm = y[yIndex + 1];
            y[yIndex] = betaRe * yRe - betaIm * yIm + alphaRe * sumRe - alphaIm * sumIm;
            y[yIndex + 1] = betaRe * yIm + betaIm * yRe + alphaRe * sumIm + alphaIm * sumRe;
        }
    }

    static void refZsymv(BLAS.Uplo uplo, int n, double alphaRe, double alphaIm,
                         double[] a, int aStart, int lda,
                         double[] x, int xStart, int incX,
                         double betaRe, double betaIm,
                         double[] y, int yStart, int incY) {
        boolean upper = uplo == BLAS.Uplo.Upper;
        for (int i = 0; i < n; i++) {
            double sumRe = 0.0;
            double sumIm = 0.0;
            for (int j = 0; j < n; j++) {
                int index;
                if (upper) {
                    index = j < i ? matrixIndex(aStart, lda, j, i) : matrixIndex(aStart, lda, i, j);
                } else {
                    index = j > i ? matrixIndex(aStart, lda, j, i) : matrixIndex(aStart, lda, i, j);
                }
                double aRe = a[index];
                double aIm = a[index + 1];
                int xIndex = vectorIndex(xStart, incX, j);
                double xRe = x[xIndex];
                double xIm = x[xIndex + 1];
                sumRe += aRe * xRe - aIm * xIm;
                sumIm += aRe * xIm + aIm * xRe;
            }

            int yIndex = vectorIndex(yStart, incY, i);
            double yRe = y[yIndex];
            double yIm = y[yIndex + 1];
            y[yIndex] = betaRe * yRe - betaIm * yIm + alphaRe * sumRe - alphaIm * sumIm;
            y[yIndex + 1] = betaRe * yIm + betaIm * yRe + alphaRe * sumIm + alphaIm * sumRe;
        }
    }

    static void refZgemv(BLAS.Trans trans, int m, int n, double alphaRe, double alphaIm,
                         double[] a, int aStart, int lda,
                         double[] x, int xStart, int incX,
                         double betaRe, double betaIm,
                         double[] y, int yStart, int incY) {
        boolean noTrans = trans == BLAS.Trans.NoTrans;
        boolean conjTrans = trans == BLAS.Trans.Conj;
        int outLen = noTrans ? m : n;
        int sumLen = noTrans ? n : m;
        for (int out = 0; out < outLen; out++) {
            double sumRe = 0.0;
            double sumIm = 0.0;
            for (int p = 0; p < sumLen; p++) {
                int aIndex;
                if (noTrans) {
                    aIndex = matrixIndex(aStart, lda, out, p);
                } else {
                    aIndex = matrixIndex(aStart, lda, p, out);
                }
                double aRe = a[aIndex];
                double aIm = a[aIndex + 1];
                if (conjTrans) {
                    aIm = -aIm;
                }
                int xIndex = vectorIndex(xStart, incX, p);
                double xRe = x[xIndex];
                double xIm = x[xIndex + 1];
                sumRe += aRe * xRe - aIm * xIm;
                sumIm += aRe * xIm + aIm * xRe;
            }

            int yIndex = vectorIndex(yStart, incY, out);
            double yRe = y[yIndex];
            double yIm = y[yIndex + 1];
            y[yIndex] = betaRe * yRe - betaIm * yIm + alphaRe * sumRe - alphaIm * sumIm;
            y[yIndex + 1] = betaRe * yIm + betaIm * yRe + alphaRe * sumIm + alphaIm * sumRe;
        }
    }

    static void refZgemm(BLAS.Trans transA, BLAS.Trans transB, int m, int n, int k,
                         double alphaRe, double alphaIm,
                         double[] a, int aStart, int lda,
                         double[] b, int bStart, int ldb,
                         double betaRe, double betaIm,
                         double[] c, int cStart, int ldc) {
        boolean transAFlag = transA == BLAS.Trans.Trans || transA == BLAS.Trans.Conj;
        boolean conjAFlag = transA == BLAS.Trans.Conj;
        boolean transBFlag = transB == BLAS.Trans.Trans || transB == BLAS.Trans.Conj;
        boolean conjBFlag = transB == BLAS.Trans.Conj;

        for (int row = 0; row < m; row++) {
            for (int col = 0; col < n; col++) {
                double sumRe = 0.0;
                double sumIm = 0.0;

                for (int p = 0; p < k; p++) {
                    int aIndex = transAFlag
                        ? matrixIndex(aStart, lda, p, row)
                        : matrixIndex(aStart, lda, row, p);
                    double aRe = a[aIndex];
                    double aIm = a[aIndex + 1];
                    if (conjAFlag) {
                        aIm = -aIm;
                    }

                    int bIndex = transBFlag
                        ? matrixIndex(bStart, ldb, col, p)
                        : matrixIndex(bStart, ldb, p, col);
                    double bRe = b[bIndex];
                    double bIm = b[bIndex + 1];
                    if (conjBFlag) {
                        bIm = -bIm;
                    }

                    sumRe = Math.fma(aRe, bRe, Math.fma(-aIm, bIm, sumRe));
                    sumIm = Math.fma(aRe, bIm, Math.fma(aIm, bRe, sumIm));
                }

                int cIndex = matrixIndex(cStart, ldc, row, col);
                double cRe = c[cIndex];
                double cIm = c[cIndex + 1];
                c[cIndex] = betaRe * cRe - betaIm * cIm + alphaRe * sumRe - alphaIm * sumIm;
                c[cIndex + 1] = betaRe * cIm + betaIm * cRe + alphaRe * sumIm + alphaIm * sumRe;
            }
        }
    }

    static void refZtrmv(BLAS.Uplo uplo, BLAS.Trans trans, BLAS.Diag diag,
                         int n, double[] a, int aStart, int lda,
                         double[] x, int xStart, int incX) {
        boolean upper = uplo == BLAS.Uplo.Upper;
        boolean noTrans = trans == BLAS.Trans.NoTrans;
        boolean conjTrans = trans == BLAS.Trans.Conj;
        boolean unitDiag = diag == BLAS.Diag.Unit;

        double[] input = new double[n * 2];
        for (int i = 0; i < n; i++) {
            int xIndex = vectorIndex(xStart, incX, i);
            int inputIndex = i * 2;
            input[inputIndex] = x[xIndex];
            input[inputIndex + 1] = x[xIndex + 1];
        }

        for (int i = 0; i < n; i++) {
            double sumRe = 0.0;
            double sumIm = 0.0;
            int jStart;
            int jEnd;
            if (noTrans) {
                if (upper) {
                    jStart = i;
                    jEnd = n;
                } else {
                    jStart = 0;
                    jEnd = i + 1;
                }
            } else {
                if (upper) {
                    jStart = 0;
                    jEnd = i + 1;
                } else {
                    jStart = i;
                    jEnd = n;
                }
            }

            for (int j = jStart; j < jEnd; j++) {
                double aRe;
                double aIm;
                if (i == j && unitDiag) {
                    aRe = 1.0;
                    aIm = 0.0;
                } else if (noTrans) {
                    int aIndex = matrixIndex(aStart, lda, i, j);
                    aRe = a[aIndex];
                    aIm = a[aIndex + 1];
                } else {
                    int aIndex = matrixIndex(aStart, lda, j, i);
                    aRe = a[aIndex];
                    aIm = a[aIndex + 1];
                    if (conjTrans) {
                        aIm = -aIm;
                    }
                }
                double xRe = input[j * 2];
                double xIm = input[j * 2 + 1];
                sumRe += aRe * xRe - aIm * xIm;
                sumIm += aRe * xIm + aIm * xRe;
            }

            int xIndex = vectorIndex(xStart, incX, i);
            x[xIndex] = sumRe;
            x[xIndex + 1] = sumIm;
        }
    }

    static void assertArrayClose(String label, double[] expected, double[] actual, double tolerance) {
        if (expected.length != actual.length) {
            throw new IllegalStateException(label + " length mismatch: expected=" + expected.length + ", actual=" + actual.length);
        }

        for (int i = 0; i < expected.length; i++) {
            double e = expected[i];
            double a = actual[i];
            double scale = Math.max(1.0, Math.max(Math.abs(e), Math.abs(a)));
            if (Math.abs(e - a) > tolerance * scale) {
                throw new IllegalStateException(label + " mismatch at index " + i + ": expected=" + e + ", actual=" + a);
            }
        }
    }

    static int matrixIndex(int start, int lda, int row, int col) {
        return start + (row * lda + col) * 2;
    }

    static int vectorIndex(int start, int inc, int index) {
        return start + index * inc * 2;
    }
}