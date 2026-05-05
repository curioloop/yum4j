/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.linalg.blas.cmplx;

import com.curioloop.yum4j.linalg.blas.BLAS;

import static java.lang.Math.max;

interface Ztrsyl {

    static double ztrsyl(char trana, char tranb, int isgn, int m, int n,
                         double[] a, int aOff, int lda,
                         double[] b, int bOff, int ldb,
                         double[] c, int cOff, int ldc,
                         boolean[] okOut) {
        double scale = 1.0;
        boolean ok = true;

        if (m == 0 || n == 0) {
            if (okOut != null) okOut[0] = ok;
            return scale;
        }

        double eps = BLAS.dlamch('P');
        double smlnum = BLAS.dlamch('S') * (m * n) / eps;
        double bignum = 1.0 / smlnum;
        double sgn = isgn;

        double smin = max(smlnum, eps * Zlange.zlange('M', m, m, a, aOff, lda, null));
        smin = max(smin, eps * Zlange.zlange('M', n, n, b, bOff, ldb, null));

        boolean notrna = trana == 'N' || trana == 'n';
        boolean notrnb = tranb == 'N' || tranb == 'n';
        boolean conja = trana == 'C' || trana == 'c';
        boolean conjb = tranb == 'C' || tranb == 'c';

        if (notrna && notrnb) {
            for (int l = 0; l < n; l++) {
                for (int k = m - 1; k >= 0; k--) {
                    double sumlRe = 0, sumlIm = 0;
                    for (int j = k + 1; j < m; j++) {
                        int aIdx = aOff + (k * lda + j) * 2;
                        int cIdx = cOff + (j * ldc + l) * 2;
                        double are = a[aIdx], aim = a[aIdx + 1];
                        double cre = c[cIdx], cim = c[cIdx + 1];
                        sumlRe += are * cre - aim * cim;
                        sumlIm += are * cim + aim * cre;
                    }
                    double sumrRe = 0, sumrIm = 0;
                    for (int j = 0; j < l; j++) {
                        int cIdx = cOff + (k * ldc + j) * 2;
                        int bIdx = bOff + (j * ldb + l) * 2;
                        double cre = c[cIdx], cim = c[cIdx + 1];
                        double bre = b[bIdx], bim = b[bIdx + 1];
                        sumrRe += cre * bre - cim * bim;
                        sumrIm += cre * bim + cim * bre;
                    }
                    int cIdx = cOff + (k * ldc + l) * 2;
                    double vecRe = c[cIdx] - (sumlRe + sgn * sumrRe);
                    double vecIm = c[cIdx + 1] - (sumlIm + sgn * sumrIm);
                    int aDiag = aOff + (k * lda + k) * 2;
                    int bDiag = bOff + (l * ldb + l) * 2;
                    double a11Re = a[aDiag] + sgn * b[bDiag];
                    double a11Im = a[aDiag + 1] + sgn * b[bDiag + 1];
                    double scaloc = 1.0;
                    double da11 = Math.hypot(a11Re, a11Im);
                    if (da11 <= smin) {
                        a11Re = smin;
                        a11Im = 0;
                        da11 = smin;
                        ok = false;
                    }
                    double db = Math.hypot(vecRe, vecIm);
                    if (da11 < 1 && db > 1 && db > bignum * da11) {
                        scaloc = 1.0 / db;
                    }
                    double vecSRe = vecRe * scaloc;
                    double vecSIm = vecIm * scaloc;
                    double denom = a11Re * a11Re + a11Im * a11Im;
                    double xRe = (vecSRe * a11Re + vecSIm * a11Im) / denom;
                    double xIm = (vecSIm * a11Re - vecSRe * a11Im) / denom;
                    if (scaloc != 1.0) {
                        for (int j = 0; j < n; j++) {
                            Zdscal.zdscal(m, scaloc, c, cOff + j * 2, ldc);
                        }
                        scale *= scaloc;
                    }
                    c[cIdx] = xRe;
                    c[cIdx + 1] = xIm;
                }
            }
        } else if (!notrna && notrnb) {
            for (int l = 0; l < n; l++) {
                for (int k = 0; k < m; k++) {
                    double sumlRe = 0, sumlIm = 0;
                    for (int j = 0; j < k; j++) {
                        int aIdx = aOff + (j * lda + k) * 2;
                        int cIdx = cOff + (j * ldc + l) * 2;
                        double are = a[aIdx], aim = a[aIdx + 1];
                        double cre = c[cIdx], cim = c[cIdx + 1];
                        if (conja) {
                            sumlRe += are * cre + aim * cim;
                            sumlIm += are * cim - aim * cre;
                        } else {
                            sumlRe += are * cre - aim * cim;
                            sumlIm += are * cim + aim * cre;
                        }
                    }
                    double sumrRe = 0, sumrIm = 0;
                    for (int j = 0; j < l; j++) {
                        int cIdx = cOff + (k * ldc + j) * 2;
                        int bIdx = bOff + (j * ldb + l) * 2;
                        double cre = c[cIdx], cim = c[cIdx + 1];
                        double bre = b[bIdx], bim = b[bIdx + 1];
                        sumrRe += cre * bre - cim * bim;
                        sumrIm += cre * bim + cim * bre;
                    }
                    int cIdx = cOff + (k * ldc + l) * 2;
                    double vecRe = c[cIdx] - (sumlRe + sgn * sumrRe);
                    double vecIm = c[cIdx + 1] - (sumlIm + sgn * sumrIm);
                    int aDiag = aOff + (k * lda + k) * 2;
                    int bDiag = bOff + (l * ldb + l) * 2;
                    double a11Re, a11Im;
                    if (conja) {
                        a11Re = a[aDiag] + sgn * b[bDiag];
                        a11Im = -a[aDiag + 1] + sgn * b[bDiag + 1];
                    } else {
                        a11Re = a[aDiag] + sgn * b[bDiag];
                        a11Im = a[aDiag + 1] + sgn * b[bDiag + 1];
                    }
                    double scaloc = 1.0;
                    double da11 = Math.hypot(a11Re, a11Im);
                    if (da11 <= smin) {
                        a11Re = smin;
                        a11Im = 0;
                        da11 = smin;
                        ok = false;
                    }
                    double db = Math.hypot(vecRe, vecIm);
                    if (da11 < 1 && db > 1 && db > bignum * da11) {
                        scaloc = 1.0 / db;
                    }
                    double vecSRe = vecRe * scaloc;
                    double vecSIm = vecIm * scaloc;
                    double denom = a11Re * a11Re + a11Im * a11Im;
                    double xRe = (vecSRe * a11Re + vecSIm * a11Im) / denom;
                    double xIm = (vecSIm * a11Re - vecSRe * a11Im) / denom;
                    if (scaloc != 1.0) {
                        for (int j = 0; j < n; j++) {
                            Zdscal.zdscal(m, scaloc, c, cOff + j * 2, ldc);
                        }
                        scale *= scaloc;
                    }
                    c[cIdx] = xRe;
                    c[cIdx + 1] = xIm;
                }
            }
        } else if (!notrna && !notrnb) {
            for (int l = n - 1; l >= 0; l--) {
                for (int k = 0; k < m; k++) {
                    double sumlRe = 0, sumlIm = 0;
                    for (int j = 0; j < k; j++) {
                        int aIdx = aOff + (j * lda + k) * 2;
                        int cIdx = cOff + (j * ldc + l) * 2;
                        double are = a[aIdx], aim = a[aIdx + 1];
                        double cre = c[cIdx], cim = c[cIdx + 1];
                        if (conja) {
                            sumlRe += are * cre + aim * cim;
                            sumlIm += are * cim - aim * cre;
                        } else {
                            sumlRe += are * cre - aim * cim;
                            sumlIm += are * cim + aim * cre;
                        }
                    }
                    double sumrRe = 0, sumrIm = 0;
                    for (int j = l + 1; j < n; j++) {
                        int cIdx = cOff + (k * ldc + j) * 2;
                        int bIdx = bOff + (l * ldb + j) * 2;
                        double cre = c[cIdx], cim = c[cIdx + 1];
                        double bre = b[bIdx], bim = b[bIdx + 1];
                        if (conjb) {
                            sumrRe += cre * bre + cim * bim;
                            sumrIm += cim * bre - cre * bim;
                        } else {
                            sumrRe += cre * bre - cim * bim;
                            sumrIm += cre * bim + cim * bre;
                        }
                    }
                    int cIdx = cOff + (k * ldc + l) * 2;
                    double vecRe = c[cIdx] - (sumlRe + sgn * sumrRe);
                    double vecIm = c[cIdx + 1] - (sumlIm + sgn * sumrIm);
                    int aDiag = aOff + (k * lda + k) * 2;
                    int bDiag = bOff + (l * ldb + l) * 2;
                    double a11Re, a11Im;
                    if (conja) {
                        a11Re = a[aDiag];
                        a11Im = -a[aDiag + 1];
                    } else {
                        a11Re = a[aDiag];
                        a11Im = a[aDiag + 1];
                    }
                    if (conjb) {
                        a11Re += sgn * b[bDiag];
                        a11Im += sgn * (-b[bDiag + 1]);
                    } else {
                        a11Re += sgn * b[bDiag];
                        a11Im += sgn * b[bDiag + 1];
                    }
                    double scaloc = 1.0;
                    double da11 = Math.hypot(a11Re, a11Im);
                    if (da11 <= smin) {
                        a11Re = smin;
                        a11Im = 0;
                        da11 = smin;
                        ok = false;
                    }
                    double db = Math.hypot(vecRe, vecIm);
                    if (da11 < 1 && db > 1 && db > bignum * da11) {
                        scaloc = 1.0 / db;
                    }
                    double vecSRe = vecRe * scaloc;
                    double vecSIm = vecIm * scaloc;
                    double denom = a11Re * a11Re + a11Im * a11Im;
                    double xRe = (vecSRe * a11Re + vecSIm * a11Im) / denom;
                    double xIm = (vecSIm * a11Re - vecSRe * a11Im) / denom;
                    if (scaloc != 1.0) {
                        for (int j = 0; j < n; j++) {
                            Zdscal.zdscal(m, scaloc, c, cOff + j * 2, ldc);
                        }
                        scale *= scaloc;
                    }
                    c[cIdx] = xRe;
                    c[cIdx + 1] = xIm;
                }
            }
        } else {
            for (int l = n - 1; l >= 0; l--) {
                for (int k = m - 1; k >= 0; k--) {
                    double sumlRe = 0, sumlIm = 0;
                    for (int j = k + 1; j < m; j++) {
                        int aIdx = aOff + (k * lda + j) * 2;
                        int cIdx = cOff + (j * ldc + l) * 2;
                        double are = a[aIdx], aim = a[aIdx + 1];
                        double cre = c[cIdx], cim = c[cIdx + 1];
                        sumlRe += are * cre - aim * cim;
                        sumlIm += are * cim + aim * cre;
                    }
                    double sumrRe = 0, sumrIm = 0;
                    for (int j = l + 1; j < n; j++) {
                        int cIdx = cOff + (k * ldc + j) * 2;
                        int bIdx = bOff + (l * ldb + j) * 2;
                        double cre = c[cIdx], cim = c[cIdx + 1];
                        double bre = b[bIdx], bim = b[bIdx + 1];
                        if (conjb) {
                            sumrRe += cre * bre + cim * bim;
                            sumrIm += cim * bre - cre * bim;
                        } else {
                            sumrRe += cre * bre - cim * bim;
                            sumrIm += cre * bim + cim * bre;
                        }
                    }
                    int cIdx = cOff + (k * ldc + l) * 2;
                    double vecRe = c[cIdx] - (sumlRe + sgn * sumrRe);
                    double vecIm = c[cIdx + 1] - (sumlIm + sgn * sumrIm);
                    int aDiag = aOff + (k * lda + k) * 2;
                    int bDiag = bOff + (l * ldb + l) * 2;
                    double a11Re = a[aDiag];
                    double a11Im = a[aDiag + 1];
                    if (conjb) {
                        a11Re += sgn * b[bDiag];
                        a11Im += sgn * (-b[bDiag + 1]);
                    } else {
                        a11Re += sgn * b[bDiag];
                        a11Im += sgn * b[bDiag + 1];
                    }
                    double scaloc = 1.0;
                    double da11 = Math.hypot(a11Re, a11Im);
                    if (da11 <= smin) {
                        a11Re = smin;
                        a11Im = 0;
                        da11 = smin;
                        ok = false;
                    }
                    double db = Math.hypot(vecRe, vecIm);
                    if (da11 < 1 && db > 1 && db > bignum * da11) {
                        scaloc = 1.0 / db;
                    }
                    double vecSRe = vecRe * scaloc;
                    double vecSIm = vecIm * scaloc;
                    double denom = a11Re * a11Re + a11Im * a11Im;
                    double xRe = (vecSRe * a11Re + vecSIm * a11Im) / denom;
                    double xIm = (vecSIm * a11Re - vecSRe * a11Im) / denom;
                    if (scaloc != 1.0) {
                        for (int j = 0; j < n; j++) {
                            Zdscal.zdscal(m, scaloc, c, cOff + j * 2, ldc);
                        }
                        scale *= scaloc;
                    }
                    c[cIdx] = xRe;
                    c[cIdx + 1] = xIm;
                }
            }
        }

        if (okOut != null) okOut[0] = ok;
        return scale;
    }
}
