package com.curioloop.yum4j.linalg.blas.cmplx;

import com.curioloop.yum4j.linalg.blas.BLAS;

interface Zlahqr {

    static int zlahqr(boolean wantt, boolean wantz, int n, int ilo, int ihi,
                      double[] H, int hOff, int ldh, double[] wr, double[] wi,
                      int iloz, int ihiz, double[] Z, int zOff, int ldz,
                      double[] work, int wOff) {
        if (n == 0) return 0;

        if (ilo == ihi) {
            int p = hOff + ilo * ldh * 2 + ilo * 2;
            wr[ilo] = H[p];
            wi[ilo] = H[p + 1];
            return 0;
        }

        double dat1 = 0.75;
        int kexsh = 10;
        double rzero = 0.0;
        double safmin = BLAS.safmin();
        double ulp = BLAS.eps();
        int nh = ihi - ilo + 1;
        double smlnum = safmin * ((double) nh / ulp);

        int i1, i2;
        if (wantt) { i1 = 0; i2 = n - 1; }
        else { i1 = ilo; i2 = ihi; }

        int itmax = 30 * Math.max(10, nh);
        int kdefl = 0;

        for (int j = ilo; j <= ihi - 3; j++) {
            setH(H, hOff, ldh, j + 2, j, 0, 0);
            if (j + 3 <= ihi) setH(H, hOff, ldh, j + 3, j, 0, 0);
        }
        if (ilo <= ihi - 2) setH(H, hOff, ldh, ihi, ihi - 2, 0, 0);

        int jlo = wantt ? 0 : ilo;
        int jhi = wantt ? n - 1 : ihi;
        for (int ii = ilo + 1; ii <= ihi; ii++) {
            double subIm = getI(H, hOff, ldh, ii, ii - 1);
            if (subIm != rzero) {
                double subRe = getR(H, hOff, ldh, ii, ii - 1);
                double cabs1Sub = Math.abs(subRe) + Math.abs(subIm);
                double scRe = subRe / cabs1Sub, scIm = subIm / cabs1Sub;
                double absSc = Math.hypot(scRe, scIm);
                scRe = scRe / absSc;
                scIm = -scIm / absSc;
                double absSub = Math.hypot(subRe, subIm);
                setH(H, hOff, ldh, ii, ii - 1, absSub, 0);
                for (int j = ii; j <= jhi; j++) {
                    double hRe = getR(H, hOff, ldh, ii, j);
                    double hIm = getI(H, hOff, ldh, ii, j);
                    setH(H, hOff, ldh, ii, j, scRe * hRe - scIm * hIm, scRe * hIm + scIm * hRe);
                }
                int jEnd = Math.min(jhi, ii + 1);
                for (int j = jlo; j <= jEnd; j++) {
                    double hRe = getR(H, hOff, ldh, j, ii);
                    double hIm = getI(H, hOff, ldh, j, ii);
                    setH(H, hOff, ldh, j, ii, scRe * hRe + scIm * hIm, scRe * hIm - scIm * hRe);
                }
                if (wantz) {
                    for (int j = iloz; j <= ihiz; j++) {
                        double zRe = Z[zOff + j * ldz * 2 + ii * 2];
                        double zIm = Z[zOff + j * ldz * 2 + ii * 2 + 1];
                        Z[zOff + j * ldz * 2 + ii * 2] = scRe * zRe + scIm * zIm;
                        Z[zOff + j * ldz * 2 + ii * 2 + 1] = scRe * zIm - scIm * zRe;
                    }
                }
            }
        }

        double[] scratch = new double[6];
        int vOff = 0;
        int tauOff = 4;

        int i = ihi;
        while (i >= ilo) {
            int l = 0;
            int its;
            for (its = 0; its <= itmax; its++) {
                int k;
                for (k = i; k >= ilo + 1; k--) {
                    double subRe = getR(H, hOff, ldh, k, k - 1);
                    if (Math.abs(subRe) <= smlnum) break;
                    double tst = cabs1(getR(H, hOff, ldh, k - 1, k - 1), getI(H, hOff, ldh, k - 1, k - 1))
                               + cabs1(getR(H, hOff, ldh, k, k), getI(H, hOff, ldh, k, k));
                    if (tst == rzero) {
                        if (k - 2 >= ilo) tst += Math.abs(getR(H, hOff, ldh, k - 1, k - 2));
                        if (k + 1 <= ihi) tst += Math.abs(getR(H, hOff, ldh, k + 1, k));
                    }
                    if (Math.abs(subRe) <= ulp * tst) {
                        double ab = Math.max(Math.abs(subRe), cabs1(getR(H, hOff, ldh, k - 1, k), getI(H, hOff, ldh, k - 1, k)));
                        double ba = Math.min(Math.abs(subRe), cabs1(getR(H, hOff, ldh, k - 1, k), getI(H, hOff, ldh, k - 1, k)));
                        double hKKRe = getR(H, hOff, ldh, k, k), hKKIm = getI(H, hOff, ldh, k, k);
                        double hKm1Re = getR(H, hOff, ldh, k - 1, k - 1), hKm1Im = getI(H, hOff, ldh, k - 1, k - 1);
                        double aa = Math.max(cabs1(hKKRe, hKKIm), cabs1(hKm1Re - hKKRe, hKm1Im - hKKIm));
                        double bb = Math.min(cabs1(hKKRe, hKKIm), cabs1(hKm1Re - hKKRe, hKm1Im - hKKIm));
                        double s = aa + ab;
                        if (ba * (ab / s) <= Math.max(smlnum, ulp * (bb * (aa / s)))) break;
                    }
                }
                l = k;
                if (l > ilo) setH(H, hOff, ldh, l, l - 1, 0, 0);
                if (l >= i) break;
                kdefl++;
                if (!wantt) { i1 = l; i2 = i; }

                double tRe, tIm, s = 0;
                if (kdefl % (2 * kexsh) == 0) {
                    s = dat1 * Math.abs(getR(H, hOff, ldh, i, i - 1));
                    tRe = s + getR(H, hOff, ldh, i, i);
                    tIm = getI(H, hOff, ldh, i, i);
                } else if (kdefl % kexsh == 0) {
                    s = dat1 * Math.abs(getR(H, hOff, ldh, l + 1, l));
                    tRe = s + getR(H, hOff, ldh, l, l);
                    tIm = getI(H, hOff, ldh, l, l);
                } else {
                    tRe = getR(H, hOff, ldh, i, i);
                    tIm = getI(H, hOff, ldh, i, i);
                    double hIm1 = getR(H, hOff, ldh, i, i - 1);
                    double hI1m1Re = getR(H, hOff, ldh, i - 1, i);
                    double hI1m1Im = getI(H, hOff, ldh, i - 1, i);
                    double sqrtHIm1 = hIm1 >= 0 ? Math.sqrt(hIm1) : 0;
                    csqrt(hI1m1Re, hI1m1Im, scratch, 0);
                    double uRe = scratch[0] * sqrtHIm1;
                    double uIm = scratch[1] * sqrtHIm1;
                    s = cabs1(uRe, uIm);
                    if (s != rzero) {
                        double hI1I1Re = getR(H, hOff, ldh, i - 1, i - 1);
                        double hI1I1Im = getI(H, hOff, ldh, i - 1, i - 1);
                        double xRe = 0.5 * (hI1I1Re - tRe);
                        double xIm = 0.5 * (hI1I1Im - tIm);
                        double sx = cabs1(xRe, xIm);
                        s = Math.max(s, cabs1(xRe, xIm));
                        double xDivSRe = xRe / s, xDivSIm = xIm / s;
                        double uDivSRe = uRe / s, uDivSIm = uIm / s;
                        double sqRe = xDivSRe * xDivSRe - xDivSIm * xDivSIm + uDivSRe * uDivSRe - uDivSIm * uDivSIm;
                        double sqIm = 2 * xDivSRe * xDivSIm + 2 * uDivSRe * uDivSIm;
                        csqrt(sqRe, sqIm, scratch, 0);
                        double yRe = s * scratch[0], yIm = s * scratch[1];
                        if (sx > rzero) {
                            if ((xRe / sx) * yRe + (xIm / sx) * yIm < rzero) {
                                yRe = -yRe; yIm = -yIm;
                            }
                        }
                        zladiv(uRe, uIm, xRe + yRe, xIm + yIm, scratch, 0);
                        tRe -= (uRe * scratch[0] - uIm * scratch[1]);
                        tIm -= (uRe * scratch[1] + uIm * scratch[0]);
                    }
                }

                int m;
                double v0Re = 0, v0Im = 0, v1 = 0;
                for (m = i - 1; m >= l + 1; m--) {
                    double h11Re = getR(H, hOff, ldh, m, m), h11Im = getI(H, hOff, ldh, m, m);
                    double h22Re = getR(H, hOff, ldh, m + 1, m + 1), h22Im = getI(H, hOff, ldh, m + 1, m + 1);
                    double h11sRe = h11Re - tRe, h11sIm = h11Im - tIm;
                    double h21 = getR(H, hOff, ldh, m + 1, m);
                    s = cabs1(h11sRe, h11sIm) + Math.abs(h21);
                    h11sRe /= s; h11sIm /= s; h21 /= s;
                    v0Re = h11sRe; v0Im = h11sIm; v1 = h21;
                    double h10 = getR(H, hOff, ldh, m, m - 1);
                    if (Math.abs(h10) * Math.abs(h21) <= ulp * cabs1(h11sRe, h11sIm) * (cabs1(h11Re, h11Im) + cabs1(h22Re, h22Im)))
                        break;
                }
                if (m < l + 1) {
                    double h11Re = getR(H, hOff, ldh, l, l), h11Im = getI(H, hOff, ldh, l, l);
                    double h11sRe = h11Re - tRe, h11sIm = h11Im - tIm;
                    double h21 = getR(H, hOff, ldh, l + 1, l);
                    s = cabs1(h11sRe, h11sIm) + Math.abs(h21);
                    h11sRe /= s; h11sIm /= s; h21 /= s;
                    v0Re = h11sRe; v0Im = h11sIm; v1 = h21;
                    m = l;
                }

                for (int kk = m; kk <= i - 1; kk++) {
                    if (kk > m) {
                        scratch[vOff] = getR(H, hOff, ldh, kk, kk - 1);
                        scratch[vOff + 1] = getI(H, hOff, ldh, kk, kk - 1);
                        scratch[vOff + 2] = getR(H, hOff, ldh, kk + 1, kk - 1);
                        scratch[vOff + 3] = getI(H, hOff, ldh, kk + 1, kk - 1);
                    } else {
                        scratch[vOff] = v0Re; scratch[vOff + 1] = v0Im; scratch[vOff + 2] = v1; scratch[vOff + 3] = 0;
                    }

                    Zlarfg.zlarfg(2, scratch, vOff, scratch, vOff + 1, 1, scratch, tauOff);
                    double t1Re = scratch[tauOff], t1Im = scratch[tauOff + 1];
                    double v2Re = scratch[vOff + 2], v2Im = scratch[vOff + 3];
                    double t2 = t1Re * v2Re - t1Im * v2Im;

                    if (kk > m) {
                        setH(H, hOff, ldh, kk, kk - 1, scratch[vOff], scratch[vOff + 1]);
                        setH(H, hOff, ldh, kk + 1, kk - 1, 0, 0);
                    }

                    for (int j = kk; j <= i2; j++) {
                        double hKJRe = getR(H, hOff, ldh, kk, j), hKJIm = getI(H, hOff, ldh, kk, j);
                        double hKp1JRe = getR(H, hOff, ldh, kk + 1, j), hKp1JIm = getI(H, hOff, ldh, kk + 1, j);
                        double sumRe = t1Re * hKJRe + t1Im * hKJIm + t2 * hKp1JRe;
                        double sumIm = t1Re * hKJIm - t1Im * hKJRe + t2 * hKp1JIm;
                        setH(H, hOff, ldh, kk, j, hKJRe - sumRe, hKJIm - sumIm);
                        double svRe = sumRe * v2Re - sumIm * v2Im;
                        double svIm = sumRe * v2Im + sumIm * v2Re;
                        setH(H, hOff, ldh, kk + 1, j, hKp1JRe - svRe, hKp1JIm - svIm);
                    }

                    int jEnd = Math.min(kk + 2, i);
                    for (int j = i1; j <= jEnd; j++) {
                        double hJKRe = getR(H, hOff, ldh, j, kk), hJKIm = getI(H, hOff, ldh, j, kk);
                        double hJKp1Re = getR(H, hOff, ldh, j, kk + 1), hJKp1Im = getI(H, hOff, ldh, j, kk + 1);
                        double sumRe = t1Re * hJKRe - t1Im * hJKIm + t2 * hJKp1Re;
                        double sumIm = t1Re * hJKIm + t1Im * hJKRe + t2 * hJKp1Im;
                        setH(H, hOff, ldh, j, kk, hJKRe - sumRe, hJKIm - sumIm);
                        double svcvRe = sumRe * v2Re + sumIm * v2Im;
                        double svcvIm = sumIm * v2Re - sumRe * v2Im;
                        setH(H, hOff, ldh, j, kk + 1, hJKp1Re - svcvRe, hJKp1Im - svcvIm);
                    }

                    if (wantz) {
                        for (int j = iloz; j <= ihiz; j++) {
                            double zJKRe = Z[zOff + j * ldz * 2 + kk * 2];
                            double zJKIm = Z[zOff + j * ldz * 2 + kk * 2 + 1];
                            double zJKp1Re = Z[zOff + j * ldz * 2 + (kk + 1) * 2];
                            double zJKp1Im = Z[zOff + j * ldz * 2 + (kk + 1) * 2 + 1];
                            double sumRe = t1Re * zJKRe - t1Im * zJKIm + t2 * zJKp1Re;
                            double sumIm = t1Re * zJKIm + t1Im * zJKRe + t2 * zJKp1Im;
                            Z[zOff + j * ldz * 2 + kk * 2] = zJKRe - sumRe;
                            Z[zOff + j * ldz * 2 + kk * 2 + 1] = zJKIm - sumIm;
                            double svcvRe = sumRe * v2Re + sumIm * v2Im;
                            double svcvIm = sumIm * v2Re - sumRe * v2Im;
                            Z[zOff + j * ldz * 2 + (kk + 1) * 2] = zJKp1Re - svcvRe;
                            Z[zOff + j * ldz * 2 + (kk + 1) * 2 + 1] = zJKp1Im - svcvIm;
                        }
                    }

                    if (kk == m && m > l) {
                        double tempRe = 1.0 - t1Re, tempIm = -t1Im;
                        double absTemp = Math.hypot(tempRe, tempIm);
                        tempRe /= absTemp; tempIm /= absTemp;
                        double hMp1MRe = getR(H, hOff, ldh, m + 1, m), hMp1MIm = getI(H, hOff, ldh, m + 1, m);
                        setH(H, hOff, ldh, m + 1, m,
                             hMp1MRe * tempRe + hMp1MIm * tempIm,
                             -hMp1MRe * tempIm + hMp1MIm * tempRe);
                        if (m + 2 <= i) {
                            double hMp2Re = getR(H, hOff, ldh, m + 2, m + 1), hMp2Im = getI(H, hOff, ldh, m + 2, m + 1);
                            setH(H, hOff, ldh, m + 2, m + 1,
                                 hMp2Re * tempRe - hMp2Im * tempIm,
                                 hMp2Re * tempIm + hMp2Im * tempRe);
                        }
                        for (int j = m; j <= i; j++) {
                            if (j != m + 1) {
                                if (i2 > j) {
                                    for (int jj = j + 1; jj <= i2; jj++) {
                                        double hRe = getR(H, hOff, ldh, j, jj), hIm = getI(H, hOff, ldh, j, jj);
                                        setH(H, hOff, ldh, j, jj, hRe * tempRe - hIm * tempIm, hRe * tempIm + hIm * tempRe);
                                    }
                                }
                                for (int jj = i1; jj <= j - 1; jj++) {
                                    double hRe = getR(H, hOff, ldh, jj, j), hIm = getI(H, hOff, ldh, jj, j);
                                    setH(H, hOff, ldh, jj, j, hRe * tempRe + hIm * tempIm, -hRe * tempIm + hIm * tempRe);
                                }
                                if (wantz) {
                                    for (int jj = iloz; jj <= ihiz; jj++) {
                                        double zRe = Z[zOff + jj * ldz * 2 + j * 2];
                                        double zIm = Z[zOff + jj * ldz * 2 + j * 2 + 1];
                                        Z[zOff + jj * ldz * 2 + j * 2] = zRe * tempRe + zIm * tempIm;
                                        Z[zOff + jj * ldz * 2 + j * 2 + 1] = -zRe * tempIm + zIm * tempRe;
                                    }
                                }
                            }
                        }
                    }
                }

                double subRe = getR(H, hOff, ldh, i, i - 1);
                double subIm = getI(H, hOff, ldh, i, i - 1);
                if (subIm != rzero) {
                    double rtemp = Math.hypot(subRe, subIm);
                    setH(H, hOff, ldh, i, i - 1, rtemp, 0);
                    double tempRe = subRe / rtemp, tempIm = subIm / rtemp;
                    if (i2 > i) {
                        for (int j = i + 1; j <= i2; j++) {
                            double hRe = getR(H, hOff, ldh, i, j), hIm = getI(H, hOff, ldh, i, j);
                            setH(H, hOff, ldh, i, j, hRe * tempRe + hIm * tempIm, -hRe * tempIm + hIm * tempRe);
                        }
                    }
                    for (int j = i1; j <= i - 1; j++) {
                        double hRe = getR(H, hOff, ldh, j, i), hIm = getI(H, hOff, ldh, j, i);
                        setH(H, hOff, ldh, j, i, hRe * tempRe - hIm * tempIm, hRe * tempIm + hIm * tempRe);
                    }
                    if (wantz) {
                        for (int j = iloz; j <= ihiz; j++) {
                            double zRe = Z[zOff + j * ldz * 2 + i * 2];
                            double zIm = Z[zOff + j * ldz * 2 + i * 2 + 1];
                            Z[zOff + j * ldz * 2 + i * 2] = zRe * tempRe - zIm * tempIm;
                            Z[zOff + j * ldz * 2 + i * 2 + 1] = zRe * tempIm + zIm * tempRe;
                        }
                    }
                }
            }

            if (its > itmax) return i + 1;

            wr[i] = getR(H, hOff, ldh, i, i);
            wi[i] = getI(H, hOff, ldh, i, i);
            kdefl = 0;
            i = l - 1;
        }

        return 0;
    }

    static double cabs1(double re, double im) { return Math.abs(re) + Math.abs(im); }

    static double getR(double[] H, int hOff, int ldh, int row, int col) {
        return H[hOff + row * ldh * 2 + col * 2];
    }

    static double getI(double[] H, int hOff, int ldh, int row, int col) {
        return H[hOff + row * ldh * 2 + col * 2 + 1];
    }

    static void setH(double[] H, int hOff, int ldh, int row, int col, double re, double im) {
        int p = hOff + row * ldh * 2 + col * 2;
        H[p] = re; H[p + 1] = im;
    }

    static void csqrt(double re, double im, double[] out, int off) {
        double absZ = Math.hypot(re, im);
        if (absZ == 0.0) { out[off] = 0; out[off + 1] = 0; return; }
        double sqrtAbs = Math.sqrt(absZ);
        out[off] = sqrtAbs * Math.sqrt((re + absZ) / (2 * absZ));
        out[off + 1] = (im >= 0 ? 1 : -1) * sqrtAbs * Math.sqrt((-re + absZ) / (2 * absZ));
    }

    static void zladiv(double xr, double xi, double yr, double yi, double[] out, int off) {
        if (Math.abs(yr) >= Math.abs(yi)) {
            double r = yi / yr;
            double d = yr + r * yi;
            out[off] = (xr + xi * r) / d;
            out[off + 1] = (xi - xr * r) / d;
        } else {
            double r = yr / yi;
            double d = yi + r * yr;
            out[off] = (xr * r + xi) / d;
            out[off + 1] = (xi * r - xr) / d;
        }
    }
}
