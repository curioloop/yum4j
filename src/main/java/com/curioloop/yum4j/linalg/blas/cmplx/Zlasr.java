/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.linalg.blas.cmplx;

interface Zlasr {

    static void zlasr(char side, char pivot, char direct, int m, int n, double[] c, int cOff, double[] s, int sOff, double[] A, int aOff, int lda) {
        if (m <= 0 || n <= 0) return;

        if (side == 'L') {
            if (pivot == 'V') {
                if (direct == 'F') {
                    for (int j = 0; j < m - 1; j++) {
                        double ctemp = c[cOff + j];
                        double stemp = s[sOff + j];
                        if (ctemp != 1.0 || stemp != 0.0) {
                            for (int i = 0; i < n; i++) {
                                int offAprev = aOff + j * lda * 2 + i * 2;
                                int offA = aOff + (j + 1) * lda * 2 + i * 2;
                                double origA1Re = A[offA];
                                double origA1Im = A[offA + 1];
                                double origA0Re = A[offAprev];
                                double origA0Im = A[offAprev + 1];
                                A[offA] = ctemp * origA1Re - stemp * origA0Re;
                                A[offA + 1] = ctemp * origA1Im - stemp * origA0Im;
                                A[offAprev] = stemp * origA1Re + ctemp * origA0Re;
                                A[offAprev + 1] = stemp * origA1Im + ctemp * origA0Im;
                            }
                        }
                    }
                } else {
                    for (int j = m - 2; j >= 0; j--) {
                        double ctemp = c[cOff + j];
                        double stemp = s[sOff + j];
                        if (ctemp != 1.0 || stemp != 0.0) {
                            for (int i = 0; i < n; i++) {
                                int offAprev = aOff + j * lda * 2 + i * 2;
                                int offA = aOff + (j + 1) * lda * 2 + i * 2;
                                double origA1Re = A[offA];
                                double origA1Im = A[offA + 1];
                                double origA0Re = A[offAprev];
                                double origA0Im = A[offAprev + 1];
                                A[offA] = ctemp * origA1Re - stemp * origA0Re;
                                A[offA + 1] = ctemp * origA1Im - stemp * origA0Im;
                                A[offAprev] = stemp * origA1Re + ctemp * origA0Re;
                                A[offAprev + 1] = stemp * origA1Im + ctemp * origA0Im;
                            }
                        }
                    }
                }
            } else if (pivot == 'T') {
                if (direct == 'F') {
                    for (int j = 1; j < m; j++) {
                        double ctemp = c[cOff + j - 1];
                        double stemp = s[sOff + j - 1];
                        if (ctemp != 1.0 || stemp != 0.0) {
                            for (int i = 0; i < n; i++) {
                                int offAj = aOff + j * lda * 2 + i * 2;
                                int offA1 = aOff + i * 2;
                                double origAjRe = A[offAj];
                                double origAjIm = A[offAj + 1];
                                double origA1Re = A[offA1];
                                double origA1Im = A[offA1 + 1];
                                A[offAj] = ctemp * origAjRe - stemp * origA1Re;
                                A[offAj + 1] = ctemp * origAjIm - stemp * origA1Im;
                                A[offA1] = stemp * origAjRe + ctemp * origA1Re;
                                A[offA1 + 1] = stemp * origAjIm + ctemp * origA1Im;
                            }
                        }
                    }
                } else {
                    for (int j = m - 1; j >= 1; j--) {
                        double ctemp = c[cOff + j - 1];
                        double stemp = s[sOff + j - 1];
                        if (ctemp != 1.0 || stemp != 0.0) {
                            for (int i = 0; i < n; i++) {
                                int offAj = aOff + j * lda * 2 + i * 2;
                                int offA1 = aOff + i * 2;
                                double origAjRe = A[offAj];
                                double origAjIm = A[offAj + 1];
                                double origA1Re = A[offA1];
                                double origA1Im = A[offA1 + 1];
                                A[offAj] = ctemp * origAjRe - stemp * origA1Re;
                                A[offAj + 1] = ctemp * origAjIm - stemp * origA1Im;
                                A[offA1] = stemp * origAjRe + ctemp * origA1Re;
                                A[offA1 + 1] = stemp * origAjIm + ctemp * origA1Im;
                            }
                        }
                    }
                }
            } else if (pivot == 'B') {
                if (direct == 'F') {
                    for (int j = 0; j < m - 1; j++) {
                        double ctemp = c[cOff + j];
                        double stemp = s[sOff + j];
                        if (ctemp != 1.0 || stemp != 0.0) {
                            for (int i = 0; i < n; i++) {
                                int offAj = aOff + j * lda * 2 + i * 2;
                                int offAm = aOff + (m - 1) * lda * 2 + i * 2;
                                double origAjRe = A[offAj];
                                double origAjIm = A[offAj + 1];
                                double origAmRe = A[offAm];
                                double origAmIm = A[offAm + 1];
                                A[offAj] = stemp * origAmRe + ctemp * origAjRe;
                                A[offAj + 1] = stemp * origAmIm + ctemp * origAjIm;
                                A[offAm] = ctemp * origAmRe - stemp * origAjRe;
                                A[offAm + 1] = ctemp * origAmIm - stemp * origAjIm;
                            }
                        }
                    }
                } else {
                    for (int j = m - 2; j >= 0; j--) {
                        double ctemp = c[cOff + j];
                        double stemp = s[sOff + j];
                        if (ctemp != 1.0 || stemp != 0.0) {
                            for (int i = 0; i < n; i++) {
                                int offAj = aOff + j * lda * 2 + i * 2;
                                int offAm = aOff + (m - 1) * lda * 2 + i * 2;
                                double origAjRe = A[offAj];
                                double origAjIm = A[offAj + 1];
                                double origAmRe = A[offAm];
                                double origAmIm = A[offAm + 1];
                                A[offAj] = stemp * origAmRe + ctemp * origAjRe;
                                A[offAj + 1] = stemp * origAmIm + ctemp * origAjIm;
                                A[offAm] = ctemp * origAmRe - stemp * origAjRe;
                                A[offAm + 1] = ctemp * origAmIm - stemp * origAjIm;
                            }
                        }
                    }
                }
            }
        } else if (side == 'R') {
            if (pivot == 'V') {
                if (direct == 'F') {
                    for (int j = 0; j < n - 1; j++) {
                        double ctemp = c[cOff + j];
                        double stemp = s[sOff + j];
                        if (ctemp != 1.0 || stemp != 0.0) {
                            for (int i = 0; i < m; i++) {
                                int offAprev = aOff + i * lda * 2 + j * 2;
                                int offA = aOff + i * lda * 2 + (j + 1) * 2;
                                double origA1Re = A[offA];
                                double origA1Im = A[offA + 1];
                                double origA0Re = A[offAprev];
                                double origA0Im = A[offAprev + 1];
                                A[offA] = ctemp * origA1Re - stemp * origA0Re;
                                A[offA + 1] = ctemp * origA1Im - stemp * origA0Im;
                                A[offAprev] = stemp * origA1Re + ctemp * origA0Re;
                                A[offAprev + 1] = stemp * origA1Im + ctemp * origA0Im;
                            }
                        }
                    }
                } else {
                    for (int j = n - 2; j >= 0; j--) {
                        double ctemp = c[cOff + j];
                        double stemp = s[sOff + j];
                        if (ctemp != 1.0 || stemp != 0.0) {
                            for (int i = 0; i < m; i++) {
                                int offAprev = aOff + i * lda * 2 + j * 2;
                                int offA = aOff + i * lda * 2 + (j + 1) * 2;
                                double origA1Re = A[offA];
                                double origA1Im = A[offA + 1];
                                double origA0Re = A[offAprev];
                                double origA0Im = A[offAprev + 1];
                                A[offA] = ctemp * origA1Re - stemp * origA0Re;
                                A[offA + 1] = ctemp * origA1Im - stemp * origA0Im;
                                A[offAprev] = stemp * origA1Re + ctemp * origA0Re;
                                A[offAprev + 1] = stemp * origA1Im + ctemp * origA0Im;
                            }
                        }
                    }
                }
            } else if (pivot == 'T') {
                if (direct == 'F') {
                    for (int j = 1; j < n; j++) {
                        double ctemp = c[cOff + j - 1];
                        double stemp = s[sOff + j - 1];
                        if (ctemp != 1.0 || stemp != 0.0) {
                            for (int i = 0; i < m; i++) {
                                int offAj = aOff + i * lda * 2 + j * 2;
                                int offA1 = aOff + i * lda * 2;
                                double origAjRe = A[offAj];
                                double origAjIm = A[offAj + 1];
                                double origA1Re = A[offA1];
                                double origA1Im = A[offA1 + 1];
                                A[offAj] = ctemp * origAjRe - stemp * origA1Re;
                                A[offAj + 1] = ctemp * origAjIm - stemp * origA1Im;
                                A[offA1] = stemp * origAjRe + ctemp * origA1Re;
                                A[offA1 + 1] = stemp * origAjIm + ctemp * origA1Im;
                            }
                        }
                    }
                } else {
                    for (int j = n - 1; j >= 1; j--) {
                        double ctemp = c[cOff + j - 1];
                        double stemp = s[sOff + j - 1];
                        if (ctemp != 1.0 || stemp != 0.0) {
                            for (int i = 0; i < m; i++) {
                                int offAj = aOff + i * lda * 2 + j * 2;
                                int offA1 = aOff + i * lda * 2;
                                double origAjRe = A[offAj];
                                double origAjIm = A[offAj + 1];
                                double origA1Re = A[offA1];
                                double origA1Im = A[offA1 + 1];
                                A[offAj] = ctemp * origAjRe - stemp * origA1Re;
                                A[offAj + 1] = ctemp * origAjIm - stemp * origA1Im;
                                A[offA1] = stemp * origAjRe + ctemp * origA1Re;
                                A[offA1 + 1] = stemp * origAjIm + ctemp * origA1Im;
                            }
                        }
                    }
                }
            } else if (pivot == 'B') {
                if (direct == 'F') {
                    for (int j = 0; j < n - 1; j++) {
                        double ctemp = c[cOff + j];
                        double stemp = s[sOff + j];
                        if (ctemp != 1.0 || stemp != 0.0) {
                            for (int i = 0; i < m; i++) {
                                int offAj = aOff + i * lda * 2 + j * 2;
                                int offAn = aOff + i * lda * 2 + (n - 1) * 2;
                                double origAjRe = A[offAj];
                                double origAjIm = A[offAj + 1];
                                double origAnRe = A[offAn];
                                double origAnIm = A[offAn + 1];
                                A[offAj] = stemp * origAnRe + ctemp * origAjRe;
                                A[offAj + 1] = stemp * origAnIm + ctemp * origAjIm;
                                A[offAn] = ctemp * origAnRe - stemp * origAjRe;
                                A[offAn + 1] = ctemp * origAnIm - stemp * origAjIm;
                            }
                        }
                    }
                } else {
                    for (int j = n - 2; j >= 0; j--) {
                        double ctemp = c[cOff + j];
                        double stemp = s[sOff + j];
                        if (ctemp != 1.0 || stemp != 0.0) {
                            for (int i = 0; i < m; i++) {
                                int offAj = aOff + i * lda * 2 + j * 2;
                                int offAn = aOff + i * lda * 2 + (n - 1) * 2;
                                double origAjRe = A[offAj];
                                double origAjIm = A[offAj + 1];
                                double origAnRe = A[offAn];
                                double origAnIm = A[offAn + 1];
                                A[offAj] = stemp * origAnRe + ctemp * origAjRe;
                                A[offAj + 1] = stemp * origAnIm + ctemp * origAjIm;
                                A[offAn] = ctemp * origAnRe - stemp * origAjRe;
                                A[offAn + 1] = ctemp * origAnIm - stemp * origAjIm;
                            }
                        }
                    }
                }
            }
        }
    }
}
