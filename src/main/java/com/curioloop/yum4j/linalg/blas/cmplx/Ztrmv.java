package com.curioloop.yum4j.linalg.blas.cmplx;

import com.curioloop.yum4j.linalg.blas.BLAS;
interface Ztrmv {

    int COLUMN_VIEW_MIN_N = 16;

    public static void ztrmv(BLAS.Uplo uplo, BLAS.Trans trans, BLAS.Diag diag,
                            int n, double[] A, int aStart, int lda,
                            double[] x, int xStart, int incX) {
        if (n <= 0) return;

        boolean upper = (uplo == BLAS.Uplo.Upper);
        boolean noTrans = (trans == BLAS.Trans.NoTrans);
        boolean conjTrans = (trans == BLAS.Trans.Conj);
        boolean unitDiag = (diag == BLAS.Diag.Unit);

        if (noTrans) {
            if (upper) {
                if (A == x && incX == lda && incX > 1
                    && ztrmvUpperNoTransAdjacentColumnView(n, A, aStart, lda, x, xStart, incX, unitDiag)) {
                    return;
                }
                ztrmvUpperNoTrans(n, A, aStart, lda, x, xStart, incX, unitDiag);
            } else {
                if (A == x && incX == lda && incX > 1
                    && ztrmvLowerNoTransAdjacentColumnView(n, A, aStart, lda, x, xStart, incX, unitDiag)) {
                    return;
                }
                ztrmvLowerNoTrans(n, A, aStart, lda, x, xStart, incX, unitDiag);
            }
        } else if (conjTrans) {
            if (upper) {
                ztrmvUpperConjTrans(n, A, aStart, lda, x, xStart, incX, unitDiag);
            } else {
                ztrmvLowerConjTrans(n, A, aStart, lda, x, xStart, incX, unitDiag);
            }
        } else {
            if (upper) {
                ztrmvUpperTrans(n, A, aStart, lda, x, xStart, incX, unitDiag);
            } else {
                ztrmvLowerTrans(n, A, aStart, lda, x, xStart, incX, unitDiag);
            }
        }
    }

    static boolean ztrmvUpperNoTransAdjacentColumnView(int n,
                                                       double[] A, int aStart, int lda,
                                                       double[] x, int xStart, int incX,
                                                       boolean unitDiag) {
        return ztrmvNoTransAdjacentColumnView(n, A, aStart, lda, x, xStart, incX, unitDiag, n, true);
    }

    static boolean ztrmvLowerNoTransAdjacentColumnView(int n,
                                                       double[] A, int aStart, int lda,
                                                       double[] x, int xStart, int incX,
                                                       boolean unitDiag) {
        return ztrmvNoTransAdjacentColumnView(n, A, aStart, lda, x, xStart, incX, unitDiag, -1, false);
    }

    static boolean ztrmvNoTransAdjacentColumnView(int n,
                                                  double[] A, int aStart, int lda,
                                                  double[] x, int xStart, int incX,
                                                  boolean unitDiag,
                                                  int vectorColOffset,
                                                  boolean upper) {
        if (!isSameArrayAdjacentColumnView(n, A, aStart, lda, x, xStart, incX, vectorColOffset)) {
            return false;
        }

        if (upper) {
            ztrmvUpperNoTransAdjacentColumn(n, A, aStart, lda, vectorColOffset, unitDiag);
        } else {
            ztrmvLowerNoTransAdjacentColumn(n, A, aStart, lda, vectorColOffset, unitDiag);
        }
        return true;
    }

    static boolean isSameArrayAdjacentColumnView(int n,
                                                 double[] A, int aStart, int lda,
                                                 double[] x, int xStart, int incX,
                                                 int vectorColOffset) {
        return n >= COLUMN_VIEW_MIN_N
            && A == x
            && incX == lda
            && incX > 1
            && xStart == aStart + vectorColOffset * 2;
    }

    static void ztrmvUpperNoTransAdjacentColumn(int n, double[] A, int aStart, int lda,
                                                int vectorColOffset, boolean unitDiag) {
        int lda2 = lda * 2;
        int columnDelta = vectorColOffset * 2;
        int i = 0;
        for (; i + 3 < n; i += 4) {
            int row0 = aStart + i * lda2;
            int row1 = row0 + lda2;
            int row2 = row1 + lda2;
            int row3 = row2 + lda2;

            int x0Off = row0 + columnDelta;
            int x1Off = row1 + columnDelta;
            int x2Off = row2 + columnDelta;
            int x3Off = row3 + columnDelta;

            double x0Re = A[x0Off], x0Im = A[x0Off + 1];
            double x1Re = A[x1Off], x1Im = A[x1Off + 1];
            double x2Re = A[x2Off], x2Im = A[x2Off + 1];
            double x3Re = A[x3Off], x3Im = A[x3Off + 1];

            double s0Re, s0Im;
            double s1Re, s1Im;
            double s2Re, s2Im;
            double s3Re, s3Im;
            if (unitDiag) {
                s0Re = x0Re;
                s0Im = x0Im;
                s1Re = x1Re;
                s1Im = x1Im;
                s2Re = x2Re;
                s2Im = x2Im;
                s3Re = x3Re;
                s3Im = x3Im;
            } else {
                int a00 = row0 + i * 2;
                int a11 = row1 + (i + 1) * 2;
                int a22 = row2 + (i + 2) * 2;
                int a33 = row3 + (i + 3) * 2;

                double a00Re = A[a00], a00Im = A[a00 + 1];
                double a11Re = A[a11], a11Im = A[a11 + 1];
                double a22Re = A[a22], a22Im = A[a22 + 1];
                double a33Re = A[a33], a33Im = A[a33 + 1];

                s0Re = a00Re * x0Re - a00Im * x0Im;
                s0Im = a00Re * x0Im + a00Im * x0Re;
                s1Re = a11Re * x1Re - a11Im * x1Im;
                s1Im = a11Re * x1Im + a11Im * x1Re;
                s2Re = a22Re * x2Re - a22Im * x2Im;
                s2Im = a22Re * x2Im + a22Im * x2Re;
                s3Re = a33Re * x3Re - a33Im * x3Im;
                s3Im = a33Re * x3Im + a33Im * x3Re;
            }

            int a01 = row0 + (i + 1) * 2;
            double a01Re = A[a01], a01Im = A[a01 + 1];
            s0Re = Math.fma(a01Re, x1Re, Math.fma(-a01Im, x1Im, s0Re));
            s0Im = Math.fma(a01Re, x1Im, Math.fma(a01Im, x1Re, s0Im));

            int a02 = row0 + (i + 2) * 2;
            double a02Re = A[a02], a02Im = A[a02 + 1];
            s0Re = Math.fma(a02Re, x2Re, Math.fma(-a02Im, x2Im, s0Re));
            s0Im = Math.fma(a02Re, x2Im, Math.fma(a02Im, x2Re, s0Im));

            int a03 = row0 + (i + 3) * 2;
            double a03Re = A[a03], a03Im = A[a03 + 1];
            s0Re = Math.fma(a03Re, x3Re, Math.fma(-a03Im, x3Im, s0Re));
            s0Im = Math.fma(a03Re, x3Im, Math.fma(a03Im, x3Re, s0Im));

            int a12 = row1 + (i + 2) * 2;
            double a12Re = A[a12], a12Im = A[a12 + 1];
            s1Re = Math.fma(a12Re, x2Re, Math.fma(-a12Im, x2Im, s1Re));
            s1Im = Math.fma(a12Re, x2Im, Math.fma(a12Im, x2Re, s1Im));

            int a13 = row1 + (i + 3) * 2;
            double a13Re = A[a13], a13Im = A[a13 + 1];
            s1Re = Math.fma(a13Re, x3Re, Math.fma(-a13Im, x3Im, s1Re));
            s1Im = Math.fma(a13Re, x3Im, Math.fma(a13Im, x3Re, s1Im));

            int a23 = row2 + (i + 3) * 2;
            double a23Re = A[a23], a23Im = A[a23 + 1];
            s2Re = Math.fma(a23Re, x3Re, Math.fma(-a23Im, x3Im, s2Re));
            s2Im = Math.fma(a23Re, x3Im, Math.fma(a23Im, x3Re, s2Im));

            for (int j = i + 4; j < n; j++) {
                int xjOff = aStart + j * lda2 + columnDelta;
                double xjRe = A[xjOff], xjIm = A[xjOff + 1];

                int a0 = row0 + j * 2;
                double a0Re = A[a0], a0Im = A[a0 + 1];
                s0Re = Math.fma(a0Re, xjRe, Math.fma(-a0Im, xjIm, s0Re));
                s0Im = Math.fma(a0Re, xjIm, Math.fma(a0Im, xjRe, s0Im));

                int a1 = row1 + j * 2;
                double a1Re = A[a1], a1Im = A[a1 + 1];
                s1Re = Math.fma(a1Re, xjRe, Math.fma(-a1Im, xjIm, s1Re));
                s1Im = Math.fma(a1Re, xjIm, Math.fma(a1Im, xjRe, s1Im));

                int a2 = row2 + j * 2;
                double a2Re = A[a2], a2Im = A[a2 + 1];
                s2Re = Math.fma(a2Re, xjRe, Math.fma(-a2Im, xjIm, s2Re));
                s2Im = Math.fma(a2Re, xjIm, Math.fma(a2Im, xjRe, s2Im));

                int a3 = row3 + j * 2;
                double a3Re = A[a3], a3Im = A[a3 + 1];
                s3Re = Math.fma(a3Re, xjRe, Math.fma(-a3Im, xjIm, s3Re));
                s3Im = Math.fma(a3Re, xjIm, Math.fma(a3Im, xjRe, s3Im));
            }

            A[x0Off] = s0Re;
            A[x0Off + 1] = s0Im;
            A[x1Off] = s1Re;
            A[x1Off + 1] = s1Im;
            A[x2Off] = s2Re;
            A[x2Off + 1] = s2Im;
            A[x3Off] = s3Re;
            A[x3Off + 1] = s3Im;
        }

        for (; i < n; i++) {
            int rowOff = aStart + i * lda2;
            int xiOff = rowOff + columnDelta;
            double xiRe = A[xiOff], xiIm = A[xiOff + 1];
            double tmpRe;
            double tmpIm;
            if (unitDiag) {
                tmpRe = xiRe;
                tmpIm = xiIm;
            } else {
                int aPos = rowOff + i * 2;
                double aRe = A[aPos], aIm = A[aPos + 1];
                tmpRe = aRe * xiRe - aIm * xiIm;
                tmpIm = aRe * xiIm + aIm * xiRe;
            }
            for (int j = i + 1; j < n; j++) {
                int aPos = rowOff + j * 2;
                int xjOff = aStart + j * lda2 + columnDelta;
                double aRe = A[aPos], aIm = A[aPos + 1];
                double xjRe = A[xjOff], xjIm = A[xjOff + 1];
                tmpRe = Math.fma(aRe, xjRe, Math.fma(-aIm, xjIm, tmpRe));
                tmpIm = Math.fma(aRe, xjIm, Math.fma(aIm, xjRe, tmpIm));
            }
            A[xiOff] = tmpRe;
            A[xiOff + 1] = tmpIm;
        }
    }

    static void ztrmvLowerNoTransAdjacentColumn(int n, double[] A, int aStart, int lda,
                                                int vectorColOffset, boolean unitDiag) {
        int lda2 = lda * 2;
        int columnDelta = vectorColOffset * 2;
        int rem = n & 3;
        for (int i = n - 1; i >= rem + 3; i -= 4) {
            int base = i - 3;
            int row0 = aStart + base * lda2;
            int row1 = row0 + lda2;
            int row2 = row1 + lda2;
            int row3 = row2 + lda2;

            int x0Off = row0 + columnDelta;
            int x1Off = row1 + columnDelta;
            int x2Off = row2 + columnDelta;
            int x3Off = row3 + columnDelta;

            double x0Re = A[x0Off], x0Im = A[x0Off + 1];
            double x1Re = A[x1Off], x1Im = A[x1Off + 1];
            double x2Re = A[x2Off], x2Im = A[x2Off + 1];
            double x3Re = A[x3Off], x3Im = A[x3Off + 1];

            double s0Re, s0Im;
            double s1Re, s1Im;
            double s2Re, s2Im;
            double s3Re, s3Im;
            if (unitDiag) {
                s0Re = x0Re;
                s0Im = x0Im;
                s1Re = x1Re;
                s1Im = x1Im;
                s2Re = x2Re;
                s2Im = x2Im;
                s3Re = x3Re;
                s3Im = x3Im;
            } else {
                int a00 = row0 + base * 2;
                int a11 = row1 + (base + 1) * 2;
                int a22 = row2 + (base + 2) * 2;
                int a33 = row3 + (base + 3) * 2;

                double a00Re = A[a00], a00Im = A[a00 + 1];
                double a11Re = A[a11], a11Im = A[a11 + 1];
                double a22Re = A[a22], a22Im = A[a22 + 1];
                double a33Re = A[a33], a33Im = A[a33 + 1];

                s0Re = a00Re * x0Re - a00Im * x0Im;
                s0Im = a00Re * x0Im + a00Im * x0Re;
                s1Re = a11Re * x1Re - a11Im * x1Im;
                s1Im = a11Re * x1Im + a11Im * x1Re;
                s2Re = a22Re * x2Re - a22Im * x2Im;
                s2Im = a22Re * x2Im + a22Im * x2Re;
                s3Re = a33Re * x3Re - a33Im * x3Im;
                s3Im = a33Re * x3Im + a33Im * x3Re;
            }

            for (int j = 0; j < base; j++) {
                int xjOff = aStart + j * lda2 + columnDelta;
                double xjRe = A[xjOff], xjIm = A[xjOff + 1];

                int a0 = row0 + j * 2;
                double a0Re = A[a0], a0Im = A[a0 + 1];
                s0Re = Math.fma(a0Re, xjRe, Math.fma(-a0Im, xjIm, s0Re));
                s0Im = Math.fma(a0Re, xjIm, Math.fma(a0Im, xjRe, s0Im));

                int a1 = row1 + j * 2;
                double a1Re = A[a1], a1Im = A[a1 + 1];
                s1Re = Math.fma(a1Re, xjRe, Math.fma(-a1Im, xjIm, s1Re));
                s1Im = Math.fma(a1Re, xjIm, Math.fma(a1Im, xjRe, s1Im));

                int a2 = row2 + j * 2;
                double a2Re = A[a2], a2Im = A[a2 + 1];
                s2Re = Math.fma(a2Re, xjRe, Math.fma(-a2Im, xjIm, s2Re));
                s2Im = Math.fma(a2Re, xjIm, Math.fma(a2Im, xjRe, s2Im));

                int a3 = row3 + j * 2;
                double a3Re = A[a3], a3Im = A[a3 + 1];
                s3Re = Math.fma(a3Re, xjRe, Math.fma(-a3Im, xjIm, s3Re));
                s3Im = Math.fma(a3Re, xjIm, Math.fma(a3Im, xjRe, s3Im));
            }

            int a10 = row1 + base * 2;
            double a10Re = A[a10], a10Im = A[a10 + 1];
            s1Re = Math.fma(a10Re, x0Re, Math.fma(-a10Im, x0Im, s1Re));
            s1Im = Math.fma(a10Re, x0Im, Math.fma(a10Im, x0Re, s1Im));

            int a20 = row2 + base * 2;
            double a20Re = A[a20], a20Im = A[a20 + 1];
            s2Re = Math.fma(a20Re, x0Re, Math.fma(-a20Im, x0Im, s2Re));
            s2Im = Math.fma(a20Re, x0Im, Math.fma(a20Im, x0Re, s2Im));

            int a21 = row2 + (base + 1) * 2;
            double a21Re = A[a21], a21Im = A[a21 + 1];
            s2Re = Math.fma(a21Re, x1Re, Math.fma(-a21Im, x1Im, s2Re));
            s2Im = Math.fma(a21Re, x1Im, Math.fma(a21Im, x1Re, s2Im));

            int a30 = row3 + base * 2;
            double a30Re = A[a30], a30Im = A[a30 + 1];
            s3Re = Math.fma(a30Re, x0Re, Math.fma(-a30Im, x0Im, s3Re));
            s3Im = Math.fma(a30Re, x0Im, Math.fma(a30Im, x0Re, s3Im));

            int a31 = row3 + (base + 1) * 2;
            double a31Re = A[a31], a31Im = A[a31 + 1];
            s3Re = Math.fma(a31Re, x1Re, Math.fma(-a31Im, x1Im, s3Re));
            s3Im = Math.fma(a31Re, x1Im, Math.fma(a31Im, x1Re, s3Im));

            int a32 = row3 + (base + 2) * 2;
            double a32Re = A[a32], a32Im = A[a32 + 1];
            s3Re = Math.fma(a32Re, x2Re, Math.fma(-a32Im, x2Im, s3Re));
            s3Im = Math.fma(a32Re, x2Im, Math.fma(a32Im, x2Re, s3Im));

            A[x0Off] = s0Re;
            A[x0Off + 1] = s0Im;
            A[x1Off] = s1Re;
            A[x1Off + 1] = s1Im;
            A[x2Off] = s2Re;
            A[x2Off + 1] = s2Im;
            A[x3Off] = s3Re;
            A[x3Off + 1] = s3Im;
        }

        for (int i = rem - 1; i >= 0; i--) {
            int rowOff = aStart + i * lda2;
            int xiOff = rowOff + columnDelta;
            double xiRe = A[xiOff], xiIm = A[xiOff + 1];
            double tmpRe;
            double tmpIm;
            if (unitDiag) {
                tmpRe = xiRe;
                tmpIm = xiIm;
            } else {
                int aPos = rowOff + i * 2;
                double aRe = A[aPos], aIm = A[aPos + 1];
                tmpRe = aRe * xiRe - aIm * xiIm;
                tmpIm = aRe * xiIm + aIm * xiRe;
            }
            for (int j = 0; j < i; j++) {
                int aPos = rowOff + j * 2;
                int xjOff = aStart + j * lda2 + columnDelta;
                double aRe = A[aPos], aIm = A[aPos + 1];
                double xjRe = A[xjOff], xjIm = A[xjOff + 1];
                tmpRe = Math.fma(aRe, xjRe, Math.fma(-aIm, xjIm, tmpRe));
                tmpIm = Math.fma(aRe, xjIm, Math.fma(aIm, xjRe, tmpIm));
            }
            A[xiOff] = tmpRe;
            A[xiOff + 1] = tmpIm;
        }
    }

    static void ztrmvUpperNoTrans(int n, double[] A, int aStart, int lda,
                                  double[] x, int xStart, int incX, boolean unitDiag) {
        if (incX == 1) {
            for (int i = 0; i < n; i++) {
                int rowOff = aStart + i * lda * 2;
                int xiOff = xStart + i * 2;
                double tmpRe, tmpIm;
                if (unitDiag) {
                    tmpRe = x[xiOff];
                    tmpIm = x[xiOff + 1];
                } else {
                    int aPos = rowOff + i * 2;
                    double aRe = A[aPos], aIm = A[aPos + 1];
                    double xiRe = x[xiOff], xiIm = x[xiOff + 1];
                    tmpRe = aRe * xiRe - aIm * xiIm;
                    tmpIm = aRe * xiIm + aIm * xiRe;
                }

                int j = i + 1;
                int j2 = i + 1 + ((n - i - 1) / 2) * 2;

                for (; j < j2; j += 2) {
                    int ap0 = rowOff + j * 2;
                    double a0Re = A[ap0], a0Im = A[ap0 + 1];
                    double a1Re = A[ap0 + 2], a1Im = A[ap0 + 3];

                    int xp0 = xStart + j * 2;
                    double x0Re = x[xp0], x0Im = x[xp0 + 1];
                    double x1Re = x[xp0 + 2], x1Im = x[xp0 + 3];

                    tmpRe = Math.fma(a0Re, x0Re, Math.fma(-a0Im, x0Im, tmpRe));
                    tmpIm = Math.fma(a0Re, x0Im, Math.fma(a0Im, x0Re, tmpIm));
                    tmpRe = Math.fma(a1Re, x1Re, Math.fma(-a1Im, x1Im, tmpRe));
                    tmpIm = Math.fma(a1Re, x1Im, Math.fma(a1Im, x1Re, tmpIm));
                }

                for (; j < n; j++) {
                    int aPos = rowOff + j * 2;
                    int xjOff = xStart + j * 2;
                    double aRe = A[aPos], aIm = A[aPos + 1];
                    double xjRe = x[xjOff], xjIm = x[xjOff + 1];
                    tmpRe = Math.fma(aRe, xjRe, Math.fma(-aIm, xjIm, tmpRe));
                    tmpIm = Math.fma(aRe, xjIm, Math.fma(aIm, xjRe, tmpIm));
                }

                x[xiOff] = tmpRe;
                x[xiOff + 1] = tmpIm;
            }
        } else {
            for (int i = 0; i < n; i++) {
                int xPos = xStart + i * incX * 2;
                int aRow = aStart + i * lda * 2;
                double tmpRe, tmpIm;
                if (unitDiag) {
                    tmpRe = x[xPos];
                    tmpIm = x[xPos + 1];
                } else {
                    int aPos = aRow + i * 2;
                    double aRe = A[aPos], aIm = A[aPos + 1];
                    double xiRe = x[xPos], xiIm = x[xPos + 1];
                    tmpRe = aRe * xiRe - aIm * xiIm;
                    tmpIm = aRe * xiIm + aIm * xiRe;
                }
                for (int j = i + 1; j < n; j++) {
                    int aPos = aRow + j * 2;
                    int xjPos = xStart + j * incX * 2;
                    double aRe = A[aPos], aIm = A[aPos + 1];
                    double xjRe = x[xjPos], xjIm = x[xjPos + 1];
                    tmpRe = Math.fma(aRe, xjRe, Math.fma(-aIm, xjIm, tmpRe));
                    tmpIm = Math.fma(aRe, xjIm, Math.fma(aIm, xjRe, tmpIm));
                }
                x[xPos] = tmpRe;
                x[xPos + 1] = tmpIm;
            }
        }
    }

    static void ztrmvLowerNoTrans(int n, double[] A, int aStart, int lda,
                                  double[] x, int xStart, int incX, boolean unitDiag) {
        if (incX == 1) {
            for (int i = n - 1; i >= 0; i--) {
                int rowOff = aStart + i * lda * 2;
                int xiOff = xStart + i * 2;
                double tmpRe, tmpIm;
                if (unitDiag) {
                    tmpRe = x[xiOff];
                    tmpIm = x[xiOff + 1];
                } else {
                    int aPos = rowOff + i * 2;
                    double aRe = A[aPos], aIm = A[aPos + 1];
                    double xiRe = x[xiOff], xiIm = x[xiOff + 1];
                    tmpRe = aRe * xiRe - aIm * xiIm;
                    tmpIm = aRe * xiIm + aIm * xiRe;
                }

                int j2 = (i / 2) * 2;
                int j = 0;

                for (; j < j2; j += 2) {
                    int ap0 = rowOff + j * 2;
                    double a0Re = A[ap0], a0Im = A[ap0 + 1];
                    double a1Re = A[ap0 + 2], a1Im = A[ap0 + 3];

                    int xp0 = xStart + j * 2;
                    double x0Re = x[xp0], x0Im = x[xp0 + 1];
                    double x1Re = x[xp0 + 2], x1Im = x[xp0 + 3];

                    tmpRe = Math.fma(a0Re, x0Re, Math.fma(-a0Im, x0Im, tmpRe));
                    tmpIm = Math.fma(a0Re, x0Im, Math.fma(a0Im, x0Re, tmpIm));
                    tmpRe = Math.fma(a1Re, x1Re, Math.fma(-a1Im, x1Im, tmpRe));
                    tmpIm = Math.fma(a1Re, x1Im, Math.fma(a1Im, x1Re, tmpIm));
                }

                for (; j < i; j++) {
                    int aPos = rowOff + j * 2;
                    int xjOff = xStart + j * 2;
                    double aRe = A[aPos], aIm = A[aPos + 1];
                    double xjRe = x[xjOff], xjIm = x[xjOff + 1];
                    tmpRe = Math.fma(aRe, xjRe, Math.fma(-aIm, xjIm, tmpRe));
                    tmpIm = Math.fma(aRe, xjIm, Math.fma(aIm, xjRe, tmpIm));
                }

                x[xiOff] = tmpRe;
                x[xiOff + 1] = tmpIm;
            }
        } else {
            for (int i = n - 1; i >= 0; i--) {
                int xPos = xStart + i * incX * 2;
                int aRow = aStart + i * lda * 2;
                double tmpRe, tmpIm;
                if (unitDiag) {
                    tmpRe = x[xPos];
                    tmpIm = x[xPos + 1];
                } else {
                    int aPos = aRow + i * 2;
                    double aRe = A[aPos], aIm = A[aPos + 1];
                    double xiRe = x[xPos], xiIm = x[xPos + 1];
                    tmpRe = aRe * xiRe - aIm * xiIm;
                    tmpIm = aRe * xiIm + aIm * xiRe;
                }
                for (int j = 0; j < i; j++) {
                    int aPos = aRow + j * 2;
                    int xjPos = xStart + j * incX * 2;
                    double aRe = A[aPos], aIm = A[aPos + 1];
                    double xjRe = x[xjPos], xjIm = x[xjPos + 1];
                    tmpRe = Math.fma(aRe, xjRe, Math.fma(-aIm, xjIm, tmpRe));
                    tmpIm = Math.fma(aRe, xjIm, Math.fma(aIm, xjRe, tmpIm));
                }
                x[xPos] = tmpRe;
                x[xPos + 1] = tmpIm;
            }
        }
    }

    static void ztrmvUpperConjTrans(int n, double[] A, int aStart, int lda,
                                    double[] x, int xStart, int incX, boolean unitDiag) {
        if (incX == 1) {
            for (int i = n - 1; i >= 0; i--) {
                int rowOff = aStart + i * lda * 2;
                double xiRe = x[xStart + i * 2];
                double xiIm = x[xStart + i * 2 + 1];
                for (int j = i + 1; j < n; j++) {
                    int aPos = rowOff + j * 2;
                    double aRe = A[aPos], aIm = A[aPos + 1];
                    int xjOff = xStart + j * 2;
                    x[xjOff]     = Math.fma(aRe, xiRe, Math.fma( aIm, xiIm, x[xjOff]));
                    x[xjOff + 1] = Math.fma(aRe, xiIm, Math.fma(-aIm, xiRe, x[xjOff + 1]));
                }
                if (!unitDiag) {
                    int aPos = rowOff + i * 2;
                    double aRe = A[aPos], aIm = A[aPos + 1];
                    double newXiRe = aRe * xiRe + aIm * xiIm;
                    double newXiIm = aRe * xiIm - aIm * xiRe;
                    x[xStart + i * 2] = newXiRe;
                    x[xStart + i * 2 + 1] = newXiIm;
                }
            }
        } else {
            for (int i = n - 1; i >= 0; i--) {
                int xiOff = xStart + i * incX * 2;
                double xiRe = x[xiOff];
                double xiIm = x[xiOff + 1];
                int rowOff = aStart + i * lda * 2;
                for (int j = i + 1; j < n; j++) {
                    int aPos = rowOff + j * 2;
                    double aRe = A[aPos], aIm = A[aPos + 1];
                    int xjOff = xStart + j * incX * 2;
                    x[xjOff]     = Math.fma(aRe, xiRe, Math.fma( aIm, xiIm, x[xjOff]));
                    x[xjOff + 1] = Math.fma(aRe, xiIm, Math.fma(-aIm, xiRe, x[xjOff + 1]));
                }
                if (!unitDiag) {
                    int aPos = rowOff + i * 2;
                    double aRe = A[aPos], aIm = A[aPos + 1];
                    double newXiRe = aRe * xiRe + aIm * xiIm;
                    double newXiIm = aRe * xiIm - aIm * xiRe;
                    x[xiOff] = newXiRe;
                    x[xiOff + 1] = newXiIm;
                }
            }
        }
    }

    static void ztrmvLowerConjTrans(int n, double[] A, int aStart, int lda,
                                    double[] x, int xStart, int incX, boolean unitDiag) {
        if (incX == 1) {
            for (int i = 0; i < n; i++) {
                int rowOff = aStart + i * lda * 2;
                double xiRe = x[xStart + i * 2];
                double xiIm = x[xStart + i * 2 + 1];
                for (int j = 0; j < i; j++) {
                    int aPos = rowOff + j * 2;
                    double aRe = A[aPos], aIm = A[aPos + 1];
                    int xjOff = xStart + j * 2;
                    x[xjOff]     = Math.fma(aRe, xiRe, Math.fma( aIm, xiIm, x[xjOff]));
                    x[xjOff + 1] = Math.fma(aRe, xiIm, Math.fma(-aIm, xiRe, x[xjOff + 1]));
                }
                if (!unitDiag) {
                    int aPos = rowOff + i * 2;
                    double aRe = A[aPos], aIm = A[aPos + 1];
                    double newXiRe = aRe * xiRe + aIm * xiIm;
                    double newXiIm = aRe * xiIm - aIm * xiRe;
                    x[xStart + i * 2] = newXiRe;
                    x[xStart + i * 2 + 1] = newXiIm;
                }
            }
        } else {
            for (int i = 0; i < n; i++) {
                int xiOff = xStart + i * incX * 2;
                double xiRe = x[xiOff];
                double xiIm = x[xiOff + 1];
                int rowOff = aStart + i * lda * 2;
                for (int j = 0; j < i; j++) {
                    int aPos = rowOff + j * 2;
                    double aRe = A[aPos], aIm = A[aPos + 1];
                    int xjOff = xStart + j * incX * 2;
                    x[xjOff]     = Math.fma(aRe, xiRe, Math.fma( aIm, xiIm, x[xjOff]));
                    x[xjOff + 1] = Math.fma(aRe, xiIm, Math.fma(-aIm, xiRe, x[xjOff + 1]));
                }
                if (!unitDiag) {
                    int aPos = rowOff + i * 2;
                    double aRe = A[aPos], aIm = A[aPos + 1];
                    double newXiRe = aRe * xiRe + aIm * xiIm;
                    double newXiIm = aRe * xiIm - aIm * xiRe;
                    x[xiOff] = newXiRe;
                    x[xiOff + 1] = newXiIm;
                }
            }
        }
    }

    static void ztrmvUpperTrans(int n, double[] A, int aStart, int lda,
                                double[] x, int xStart, int incX, boolean unitDiag) {
        if (incX == 1) {
            for (int i = n - 1; i >= 0; i--) {
                int rowOff = aStart + i * lda * 2;
                double xiRe = x[xStart + i * 2];
                double xiIm = x[xStart + i * 2 + 1];
                for (int j = i + 1; j < n; j++) {
                    int aPos = rowOff + j * 2;
                    double aRe = A[aPos], aIm = A[aPos + 1];
                    int xjOff = xStart + j * 2;
                    x[xjOff]     = Math.fma(aRe, xiRe, Math.fma(-aIm, xiIm, x[xjOff]));
                    x[xjOff + 1] = Math.fma(aRe, xiIm, Math.fma( aIm, xiRe, x[xjOff + 1]));
                }
                if (!unitDiag) {
                    int aPos = rowOff + i * 2;
                    double aRe = A[aPos], aIm = A[aPos + 1];
                    double newXiRe = aRe * xiRe - aIm * xiIm;
                    double newXiIm = aRe * xiIm + aIm * xiRe;
                    x[xStart + i * 2] = newXiRe;
                    x[xStart + i * 2 + 1] = newXiIm;
                }
            }
        } else {
            for (int i = n - 1; i >= 0; i--) {
                int xiOff = xStart + i * incX * 2;
                double xiRe = x[xiOff];
                double xiIm = x[xiOff + 1];
                int rowOff = aStart + i * lda * 2;
                for (int j = i + 1; j < n; j++) {
                    int aPos = rowOff + j * 2;
                    double aRe = A[aPos], aIm = A[aPos + 1];
                    int xjOff = xStart + j * incX * 2;
                    x[xjOff]     = Math.fma(aRe, xiRe, Math.fma(-aIm, xiIm, x[xjOff]));
                    x[xjOff + 1] = Math.fma(aRe, xiIm, Math.fma( aIm, xiRe, x[xjOff + 1]));
                }
                if (!unitDiag) {
                    int aPos = rowOff + i * 2;
                    double aRe = A[aPos], aIm = A[aPos + 1];
                    double newXiRe = aRe * xiRe - aIm * xiIm;
                    double newXiIm = aRe * xiIm + aIm * xiRe;
                    x[xiOff] = newXiRe;
                    x[xiOff + 1] = newXiIm;
                }
            }
        }
    }

    static void ztrmvLowerTrans(int n, double[] A, int aStart, int lda,
                                double[] x, int xStart, int incX, boolean unitDiag) {
        if (incX == 1) {
            for (int i = 0; i < n; i++) {
                int rowOff = aStart + i * lda * 2;
                double xiRe = x[xStart + i * 2];
                double xiIm = x[xStart + i * 2 + 1];
                for (int j = 0; j < i; j++) {
                    int aPos = rowOff + j * 2;
                    double aRe = A[aPos], aIm = A[aPos + 1];
                    int xjOff = xStart + j * 2;
                    x[xjOff]     = Math.fma(aRe, xiRe, Math.fma(-aIm, xiIm, x[xjOff]));
                    x[xjOff + 1] = Math.fma(aRe, xiIm, Math.fma( aIm, xiRe, x[xjOff + 1]));
                }
                if (!unitDiag) {
                    int aPos = rowOff + i * 2;
                    double aRe = A[aPos], aIm = A[aPos + 1];
                    double newXiRe = aRe * xiRe - aIm * xiIm;
                    double newXiIm = aRe * xiIm + aIm * xiRe;
                    x[xStart + i * 2] = newXiRe;
                    x[xStart + i * 2 + 1] = newXiIm;
                }
            }
        } else {
            for (int i = 0; i < n; i++) {
                int xiOff = xStart + i * incX * 2;
                double xiRe = x[xiOff];
                double xiIm = x[xiOff + 1];
                int rowOff = aStart + i * lda * 2;
                for (int j = 0; j < i; j++) {
                    int aPos = rowOff + j * 2;
                    double aRe = A[aPos], aIm = A[aPos + 1];
                    int xjOff = xStart + j * incX * 2;
                    x[xjOff]     = Math.fma(aRe, xiRe, Math.fma(-aIm, xiIm, x[xjOff]));
                    x[xjOff + 1] = Math.fma(aRe, xiIm, Math.fma( aIm, xiRe, x[xjOff + 1]));
                }
                if (!unitDiag) {
                    int aPos = rowOff + i * 2;
                    double aRe = A[aPos], aIm = A[aPos + 1];
                    double newXiRe = aRe * xiRe - aIm * xiIm;
                    double newXiIm = aRe * xiIm + aIm * xiRe;
                    x[xiOff] = newXiRe;
                    x[xiOff + 1] = newXiIm;
                }
            }
        }
    }
}
