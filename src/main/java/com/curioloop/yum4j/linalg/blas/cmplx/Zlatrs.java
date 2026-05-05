/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.linalg.blas.cmplx;

import com.curioloop.yum4j.linalg.blas.BLAS;

import static java.lang.Math.max;
import static java.lang.Math.min;

interface Zlatrs {

    static double zlatrs(BLAS.Uplo uplo, BLAS.Trans trans, BLAS.Diag diag, boolean normin,
                         int n, double[] A, int aOff, int lda,
                         double[] x, int xOff,
                         double[] cnorm, int cnormOff) {
        boolean upper = uplo == BLAS.Uplo.Upper;
        boolean notrans = trans == BLAS.Trans.NoTrans;
        boolean nounit = diag == BLAS.Diag.NonUnit;
        if (n == 0) return 1.0;

        double smlnum = BLAS.dlamch('S') / BLAS.dlamch('P');
        double bignum = 1.0 / smlnum;
        double scale = 1.0;

        if (!normin) {
            if (upper) {
                cnorm[cnormOff] = 0;
                for (int j = 1; j < n; j++) {
                    double sum = 0;
                    for (int i = 0; i < j; i++) {
                        double re = A[(aOff + i * lda + j) * 2];
                        double im = A[(aOff + i * lda + j) * 2 + 1];
                        sum += cabs1(re, im);
                    }
                    cnorm[cnormOff + j] = sum;
                }
            } else {
                for (int j = 0; j < n - 1; j++) {
                    double sum = 0;
                    for (int i = j + 1; i < n; i++) {
                        double re = A[(aOff + i * lda + j) * 2];
                        double im = A[(aOff + i * lda + j) * 2 + 1];
                        sum += cabs1(re, im);
                    }
                    cnorm[cnormOff + j] = sum;
                }
                cnorm[cnormOff + n - 1] = 0;
            }
        }

        double tmax = cnorm[cnormOff];
        for (int i = 1; i < n; i++) {
            if (cnorm[cnormOff + i] > tmax) {
                tmax = cnorm[cnormOff + i];
            }
        }

        double tscal = 1.0;
        if (tmax > bignum * 0.5) {
            if (tmax <= Double.MAX_VALUE) {
                tscal = 0.5 / (smlnum * tmax);
                for (int j = 0; j < n; j++) {
                    cnorm[cnormOff + j] *= tscal;
                }
            } else {
                tmax = 0;
                if (upper) {
                    for (int j = 1; j < n; j++) {
                        for (int i = 0; i < j; i++) {
                            double re = A[(aOff + i * lda + j) * 2];
                            double im = A[(aOff + i * lda + j) * 2 + 1];
                            tmax = max(tmax, max(Math.abs(re), Math.abs(im)));
                        }
                    }
                } else {
                    for (int j = 0; j < n - 1; j++) {
                        for (int i = j + 1; i < n; i++) {
                            double re = A[(aOff + i * lda + j) * 2];
                            double im = A[(aOff + i * lda + j) * 2 + 1];
                            tmax = max(tmax, max(Math.abs(re), Math.abs(im)));
                        }
                    }
                }
                if (tmax <= Double.MAX_VALUE) {
                    tscal = 1.0 / (smlnum * tmax);
                    for (int j = 0; j < n; j++) {
                        if (cnorm[cnormOff + j] <= Double.MAX_VALUE) {
                            cnorm[cnormOff + j] *= tscal;
                        } else {
                            cnorm[cnormOff + j] = 0;
                            if (upper) {
                                for (int i = 0; i < j; i++) {
                                    double re = A[(aOff + i * lda + j) * 2];
                                    double im = A[(aOff + i * lda + j) * 2 + 1];
                                    cnorm[cnormOff + j] += tscal * cabs1(re, im);
                                }
                            } else {
                                for (int i = j + 1; i < n; i++) {
                                    double re = A[(aOff + i * lda + j) * 2];
                                    double im = A[(aOff + i * lda + j) * 2 + 1];
                                    cnorm[cnormOff + j] += tscal * cabs1(re, im);
                                }
                            }
                        }
                    }
                } else {
                    Ztrsv.ztrsv(uplo, trans, diag, n, A, aOff * 2, lda, x, xOff, 1);
                    return scale;
                }
            }
        }

        int jfirst, jlast, jinc;
        if (notrans) {
            if (upper) {
                jfirst = n - 1;
                jlast = -1;
                jinc = -1;
            } else {
                jfirst = 0;
                jlast = n;
                jinc = 1;
            }
        } else {
            if (upper) {
                jfirst = 0;
                jlast = n;
                jinc = 1;
            } else {
                jfirst = n - 1;
                jlast = -1;
                jinc = -1;
            }
        }

        double xmax = izamaxAbs1(n, x, xOff, 1);
        double xbnd = xmax;

        double grow = 0;
        if (tscal == 1.0) {
            if (nounit) {
                grow = 0.5 / max(xbnd, smlnum);
                xbnd = grow;
                for (int j = jfirst; j != jlast; j += jinc) {
                    if (grow <= smlnum) break;
                    int dp = (aOff + j * lda + j) * 2;
                    double tjj = cabs1(A[dp], A[dp + 1]);
                    if (tjj >= smlnum) {
                        xbnd = min(xbnd, min(1.0, tjj) * grow);
                    } else {
                        xbnd = 0.0;
                    }
                    if (tjj + cnorm[cnormOff + j] >= smlnum) {
                        grow *= tjj / (tjj + cnorm[cnormOff + j]);
                    } else {
                        grow = 0;
                    }
                }
                grow = xbnd;
            } else {
                grow = min(1.0, 0.5 / max(xbnd, smlnum));
                for (int j = jfirst; j != jlast; j += jinc) {
                    if (grow <= smlnum) break;
                    grow *= 1.0 / (1.0 + cnorm[cnormOff + j]);
                }
            }
        }

        if (grow * tscal > smlnum) {
            Ztrsv.ztrsv(uplo, trans, diag, n, A, aOff * 2, lda, x, xOff, 1);
            if (tscal != 1.0) {
                for (int j = 0; j < n; j++) {
                    cnorm[cnormOff + j] /= tscal;
                }
            }
            return scale;
        }

        if (xmax > bignum * 0.5) {
            scale = (bignum * 0.5) / xmax;
            Zdscal.zdscal(n, scale, x, xOff, 1);
            xmax = bignum;
        } else {
            xmax *= 2.0;
        }

        if (notrans) {
            for (int j = jfirst; j != jlast; j += jinc) {
                double xj = zabs(x, xOff, j);
                int diagPos = (aOff + j * lda + j) * 2;
                double tjjsRe = nounit ? A[diagPos] * tscal : tscal;
                double tjjsIm = nounit ? A[diagPos + 1] * tscal : 0.0;
                double tjj = cabs1(tjjsRe, tjjsIm);

                if (tjj > smlnum) {
                    if (tjj < 1 && xj > tjj * bignum) {
                        double rec = 1.0 / xj;
                        Zdscal.zdscal(n, rec, x, xOff, 1);
                        scale *= rec;
                        xmax *= rec;
                    }
                    zdiv(x, xOff, j, tjjsRe, tjjsIm);
                    xj = zabs(x, xOff, j);
                } else if (tjj > 0) {
                    if (xj > tjj * bignum) {
                        double rec = (tjj * bignum) / xj;
                        if (cnorm[cnormOff + j] > 1) rec /= cnorm[cnormOff + j];
                        Zdscal.zdscal(n, rec, x, xOff, 1);
                        scale *= rec;
                        xmax *= rec;
                    }
                    zdiv(x, xOff, j, tjjsRe, tjjsIm);
                    xj = zabs(x, xOff, j);
                } else {
                    for (int i = 0; i < n; i++) {
                        x[xOff + i * 2] = 0;
                        x[xOff + i * 2 + 1] = 0;
                    }
                    x[xOff + j * 2] = 1;
                    x[xOff + j * 2 + 1] = 0;
                    xj = 1;
                    scale = 0;
                    xmax = 0;
                }

                if (xj > 1) {
                    double rec = 1.0 / xj;
                    if (cnorm[cnormOff + j] > (bignum - xmax) * rec) {
                        rec *= 0.5;
                        Zdscal.zdscal(n, rec, x, xOff, 1);
                        scale *= rec;
                    }
                } else if (xj * cnorm[cnormOff + j] > bignum - xmax) {
                    Zdscal.zdscal(n, 0.5, x, xOff, 1);
                    scale *= 0.5;
                }

                double xjRe = x[xOff + j * 2];
                double xjIm = x[xOff + j * 2 + 1];

                if (upper) {
                    if (j > 0) {
                        for (int i = 0; i < j; i++) {
                            int aPos = (aOff + i * lda + j) * 2;
                            double aRe = A[aPos];
                            double aIm = A[aPos + 1];
                            x[xOff + i * 2] -= tscal * (aRe * xjRe - aIm * xjIm);
                            x[xOff + i * 2 + 1] -= tscal * (aRe * xjIm + aIm * xjRe);
                        }
                        xmax = izamaxAbs1(j, x, xOff, 1);
                    }
                } else {
                    if (j < n - 1) {
                        for (int i = j + 1; i < n; i++) {
                            int aPos = (aOff + i * lda + j) * 2;
                            double aRe = A[aPos];
                            double aIm = A[aPos + 1];
                            x[xOff + i * 2] -= tscal * (aRe * xjRe - aIm * xjIm);
                            x[xOff + i * 2 + 1] -= tscal * (aRe * xjIm + aIm * xjRe);
                        }
                        // Skip one complex entry in the interleaved [re, im] buffer.
                        xmax = izamaxAbs1(n - j - 1, x, xOff + (j + 1) * 2, 1);
                    }
                }
            }
        } else {
            boolean conjTrans = trans == BLAS.Trans.Conj;
            for (int j = jfirst; j != jlast; j += jinc) {
                double xj = zabs(x, xOff, j);
                double uscal = tscal;
                double rec = 1.0 / max(xmax, 1.0);

                if (cnorm[cnormOff + j] > (bignum - xj) * rec) {
                    rec *= 0.5;
                    int diagPos = (aOff + j * lda + j) * 2;
                    double tjjsRe = nounit ? A[diagPos] * tscal : tscal;
                    double tjjsIm = nounit ? A[diagPos + 1] * tscal : 0.0;
                    double tjj = Math.hypot(tjjsRe, tjjsIm);
                    if (tjj > 1) {
                        rec = min(1.0, rec * tjj);
                        uscal /= tjj;
                    }
                    if (rec < 1) {
                        Zdscal.zdscal(n, rec, x, xOff, 1);
                        scale *= rec;
                        xmax *= rec;
                    }
                }

                double sumjRe = 0, sumjIm = 0;
                if (uscal == tscal) {
                    if (upper) {
                        for (int i = 0; i < j; i++) {
                            int aPos = (aOff + i * lda + j) * 2;
                            double aRe = A[aPos];
                            double aIm = A[aPos + 1];
                            double xiRe = x[xOff + i * 2];
                            double xiIm = x[xOff + i * 2 + 1];
                            if (conjTrans) {
                                sumjRe += aRe * xiRe + aIm * xiIm;
                                sumjIm += aRe * xiIm - aIm * xiRe;
                            } else {
                                sumjRe += aRe * xiRe - aIm * xiIm;
                                sumjIm += aRe * xiIm + aIm * xiRe;
                            }
                        }
                    } else if (j < n - 1) {
                        for (int i = j + 1; i < n; i++) {
                            int aPos = (aOff + i * lda + j) * 2;
                            double aRe = A[aPos];
                            double aIm = A[aPos + 1];
                            double xiRe = x[xOff + i * 2];
                            double xiIm = x[xOff + i * 2 + 1];
                            if (conjTrans) {
                                sumjRe += aRe * xiRe + aIm * xiIm;
                                sumjIm += aRe * xiIm - aIm * xiRe;
                            } else {
                                sumjRe += aRe * xiRe - aIm * xiIm;
                                sumjIm += aRe * xiIm + aIm * xiRe;
                            }
                        }
                    }
                } else {
                    if (upper) {
                        for (int i = 0; i < j; i++) {
                            int aPos = (aOff + i * lda + j) * 2;
                            double aRe = A[aPos] * uscal;
                            double aIm = A[aPos + 1] * uscal;
                            double xiRe = x[xOff + i * 2];
                            double xiIm = x[xOff + i * 2 + 1];
                            if (conjTrans) {
                                sumjRe += aRe * xiRe + aIm * xiIm;
                                sumjIm += aRe * xiIm - aIm * xiRe;
                            } else {
                                sumjRe += aRe * xiRe - aIm * xiIm;
                                sumjIm += aRe * xiIm + aIm * xiRe;
                            }
                        }
                    } else if (j < n - 1) {
                        for (int i = j + 1; i < n; i++) {
                            int aPos = (aOff + i * lda + j) * 2;
                            double aRe = A[aPos] * uscal;
                            double aIm = A[aPos + 1] * uscal;
                            double xiRe = x[xOff + i * 2];
                            double xiIm = x[xOff + i * 2 + 1];
                            if (conjTrans) {
                                sumjRe += aRe * xiRe + aIm * xiIm;
                                sumjIm += aRe * xiIm - aIm * xiRe;
                            } else {
                                sumjRe += aRe * xiRe - aIm * xiIm;
                                sumjIm += aRe * xiIm + aIm * xiRe;
                            }
                        }
                    }
                }

                if (uscal == tscal) {
                    x[xOff + j * 2] -= sumjRe;
                    x[xOff + j * 2 + 1] -= sumjIm;
                    xj = zabs(x, xOff, j);
                    int diagPos = (aOff + j * lda + j) * 2;
                    double tjjsRe = nounit ? A[diagPos] * tscal : tscal;
                    double tjjsIm = nounit ? A[diagPos + 1] * tscal : 0.0;
                    double tjj = Math.hypot(tjjsRe, tjjsIm);

                    if (tjj > smlnum) {
                        if (tjj < 1 && xj > tjj * bignum) {
                            rec = 1.0 / xj;
                            Zdscal.zdscal(n, rec, x, xOff, 1);
                            scale *= rec;
                            xmax *= rec;
                        }
                        zdiv(x, xOff, j, tjjsRe, tjjsIm);
                    } else if (tjj > 0) {
                        if (xj > tjj * bignum) {
                            rec = (tjj * bignum) / xj;
                            Zdscal.zdscal(n, rec, x, xOff, 1);
                            scale *= rec;
                            xmax *= rec;
                        }
                        zdiv(x, xOff, j, tjjsRe, tjjsIm);
                    } else {
                        for (int i = 0; i < n; i++) {
                            x[xOff + i * 2] = 0;
                            x[xOff + i * 2 + 1] = 0;
                        }
                        x[xOff + j * 2] = 1;
                        x[xOff + j * 2 + 1] = 0;
                        scale = 0;
                        xmax = 0;
                    }
                } else {
                    int diagPos = (aOff + j * lda + j) * 2;
                    double tjjsRe = nounit ? A[diagPos] * tscal : tscal;
                    double tjjsIm = nounit ? A[diagPos + 1] * tscal : 0.0;
                    double xRe = x[xOff + j * 2];
                    double xIm = x[xOff + j * 2 + 1];
                    double denom = tjjsRe * tjjsRe + tjjsIm * tjjsIm;
                    x[xOff + j * 2] = (xRe * tjjsRe + xIm * tjjsIm) / denom - sumjRe;
                    x[xOff + j * 2 + 1] = (xIm * tjjsRe - xRe * tjjsIm) / denom - sumjIm;
                }

                xmax = max(xmax, zabs(x, xOff, j));
            }
        }

        scale /= tscal;
        if (tscal != 1.0) {
            for (int j = 0; j < n; j++) {
                cnorm[cnormOff + j] /= tscal;
            }
        }
        return scale;
    }

    static double zabs(double[] x, int xOff, int i) {
        double re = x[xOff + i * 2];
        double im = x[xOff + i * 2 + 1];
        return cabs1(re, im);
    }

    static double izamaxAbs1(int n, double[] x, int xOff, int incX) {
        double xmax = 0.0;
        for (int i = 0; i < n; i++) {
            int p = xOff + i * incX * 2;
            xmax = max(xmax, cabs1(x[p], x[p + 1]));
        }
        return xmax;
    }

    static double cabs1(double re, double im) {
        return Math.abs(re) + Math.abs(im);
    }

    static void zdiv(double[] x, int xOff, int j, double dRe, double dIm) {
        double xRe = x[xOff + j * 2];
        double xIm = x[xOff + j * 2 + 1];
        double denom = dRe * dRe + dIm * dIm;
        x[xOff + j * 2] = (xRe * dRe + xIm * dIm) / denom;
        x[xOff + j * 2 + 1] = (xIm * dRe - xRe * dIm) / denom;
    }
}
