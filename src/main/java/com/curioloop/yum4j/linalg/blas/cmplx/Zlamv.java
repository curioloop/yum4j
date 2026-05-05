/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.linalg.blas.cmplx;

interface Zlamv {

    static void zcopy(int n, double[] x, int xOff, int incX,
                      double[] y, int yOff, int incY) {
        if (n <= 0) return;

        if (incX == 1 && incY == 1) {
            System.arraycopy(x, xOff * 2, y, yOff * 2, n * 2);
        } else {
            int ix = xOff * 2 + ((incX > 0) ? 0 : (n - 1) * (-incX) * 2);
            int iy = yOff * 2 + ((incY > 0) ? 0 : (n - 1) * (-incY) * 2);
            for (int i = 0; i < n; i++) {
                y[iy] = x[ix];
                y[iy + 1] = x[ix + 1];
                ix += incX * 2;
                iy += incY * 2;
            }
        }
    }

    static void zset(int n, double real, double imag, double[] x, int xOff, int incX) {
        if (n <= 0) return;

        if (incX == 1) {
            int idx = xOff * 2;
            for (int i = 0; i < n; i++) {
                x[idx] = real;
                x[idx + 1] = imag;
                idx += 2;
            }
        } else {
            int ix = xOff * 2 + ((incX > 0) ? 0 : (n - 1) * (-incX) * 2);
            for (int i = 0; i < n; i++) {
                x[ix] = real;
                x[ix + 1] = imag;
                ix += incX * 2;
            }
        }
    }

    static void zzero(int n, double[] x, int xOff, int incX) {
        zset(n, 0.0, 0.0, x, xOff, incX);
    }

    static void zswap(int n, double[] x, int xOff, int incX,
                      double[] y, int yOff, int incY) {
        if (n <= 0) return;

        if (incX == 1 && incY == 1) {
            int k = 0;
            for (; k + 3 < n; k += 4) {
                int i0 = xOff * 2 + k * 2, j0 = yOff * 2 + k * 2;
                double t0 = x[i0];     x[i0] = y[j0];         y[j0] = t0;
                double t1 = x[i0 + 1]; x[i0 + 1] = y[j0 + 1]; y[j0 + 1] = t1;
                double t2 = x[i0 + 2]; x[i0 + 2] = y[j0 + 2]; y[j0 + 2] = t2;
                double t3 = x[i0 + 3]; x[i0 + 3] = y[j0 + 3]; y[j0 + 3] = t3;
                double t4 = x[i0 + 4]; x[i0 + 4] = y[j0 + 4]; y[j0 + 4] = t4;
                double t5 = x[i0 + 5]; x[i0 + 5] = y[j0 + 5]; y[j0 + 5] = t5;
                double t6 = x[i0 + 6]; x[i0 + 6] = y[j0 + 6]; y[j0 + 6] = t6;
                double t7 = x[i0 + 7]; x[i0 + 7] = y[j0 + 7]; y[j0 + 7] = t7;
            }
            for (; k < n; k++) {
                int i = xOff * 2 + k * 2, j = yOff * 2 + k * 2;
                double tempRe = x[i];
                double tempIm = x[i + 1];
                x[i] = y[j];
                x[i + 1] = y[j + 1];
                y[j] = tempRe;
                y[j + 1] = tempIm;
            }
        } else {
            int ix = xOff * 2 + ((incX > 0) ? 0 : (n - 1) * (-incX) * 2);
            int iy = yOff * 2 + ((incY > 0) ? 0 : (n - 1) * (-incY) * 2);
            for (int i = 0; i < n; i++) {
                double tempRe = x[ix];
                double tempIm = x[ix + 1];
                x[ix] = y[iy];
                x[ix + 1] = y[iy + 1];
                y[iy] = tempRe;
                y[iy + 1] = tempIm;
                ix += incX * 2;
                iy += incY * 2;
            }
        }
    }

    static void zlacpy(char uplo, int m, int n, double[] A, int aOff, int lda,
                       double[] B, int bOff, int ldb) {
        if (m == 0 || n == 0) return;

        char u = Character.toUpperCase(uplo);

        switch (u) {
            case 'U':
                for (int i = 0; i < m; i++) {
                    System.arraycopy(A, (aOff + i * lda + i) * 2, B, (bOff + i * ldb + i) * 2, (n - i) * 2);
                }
                break;
            case 'L':
                for (int i = 0; i < m; i++) {
                    int maxJ = Math.min(i + 1, n);
                    System.arraycopy(A, (aOff + i * lda) * 2, B, (bOff + i * ldb) * 2, maxJ * 2);
                }
                break;
            default:
                for (int i = 0; i < m; i++) {
                    System.arraycopy(A, (aOff + i * lda) * 2, B, (bOff + i * ldb) * 2, n * 2);
                }
                break;
        }
    }

    static void zlaset(char uplo, int m, int n, double alphaRe, double alphaIm, double betaRe, double betaIm,
                       double[] A, int aOff, int lda) {
        if (m == 0 || n == 0) return;

        char u = Character.toUpperCase(uplo);
        int minMN = Math.min(m, n);

        switch (u) {
            case 'U':
                for (int i = 0; i < m; i++) {
                    int rowOff = (aOff + i * lda) * 2;
                    for (int j = i + 1; j < n; j++) {
                        int idx = rowOff + j * 2;
                        A[idx] = alphaRe;
                        A[idx + 1] = alphaIm;
                    }
                }
                break;
            case 'L':
                for (int i = 0; i < m; i++) {
                    int rowOff = (aOff + i * lda) * 2;
                    for (int j = 0; j < Math.min(i, n); j++) {
                        int idx = rowOff + j * 2;
                        A[idx] = alphaRe;
                        A[idx + 1] = alphaIm;
                    }
                }
                break;
            default:
                for (int i = 0; i < m; i++) {
                    int rowOff = (aOff + i * lda) * 2;
                    for (int j = 0; j < n; j++) {
                        int idx = rowOff + j * 2;
                        A[idx] = alphaRe;
                        A[idx + 1] = alphaIm;
                    }
                }
                break;
        }

        for (int i = 0; i < minMN; i++) {
            int idx = (aOff + i * lda + i) * 2;
            A[idx] = betaRe;
            A[idx + 1] = betaIm;
        }
    }

    static void zlascl(char kind, int kl, int ku, double cfrom, double cto,
                       int m, int n, double[] a, int aOff, int lda) {
        if (m == 0 || n == 0) return;
        if (cfrom == 0.0) throw new ArithmeticException("cfrom is zero");

        char k = Character.toUpperCase(kind);
        double mul = cto / cfrom;

        switch (k) {
            case 'U':
                for (int i = 0, rowOff = aOff * 2; i < m; i++, rowOff += lda * 2) {
                    for (int j = i; j < n; j++) {
                        int idx = rowOff + j * 2;
                        a[idx] *= mul;
                        a[idx + 1] *= mul;
                    }
                }
                break;
            case 'L':
                for (int i = 0, rowOff = aOff * 2; i < m; i++, rowOff += lda * 2) {
                    int jEnd = Math.min(i, n - 1);
                    for (int j = 0; j <= jEnd; j++) {
                        int idx = rowOff + j * 2;
                        a[idx] *= mul;
                        a[idx + 1] *= mul;
                    }
                }
                break;
            case 'G':
                for (int i = 0, rowOff = aOff * 2; i < m; i++, rowOff += lda * 2) {
                    for (int j = 0; j < n; j++) {
                        int idx = rowOff + j * 2;
                        a[idx] *= mul;
                        a[idx + 1] *= mul;
                    }
                }
                break;
            default:
                throw new IllegalArgumentException("unknown matrix type: " + kind);
        }
    }
}
