/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.linalg.blas.cmplx;

interface Zlartg {

    double SAFMIN = 0x1p-1022;
    double SAFMAX = 1.0 / SAFMIN;

    static void zlartg(double fRe, double fIm, double gRe, double gIm,
                       double[] out, int off) {
        double cs;
        double snRe, snIm;
        double rRe, rIm;

        if (gRe == 0.0 && gIm == 0.0) {
            cs = 1.0;
            snRe = 0.0;
            snIm = 0.0;
            rRe = fRe;
            rIm = fIm;
        } else if (fRe == 0.0 && fIm == 0.0) {
            cs = 0.0;
            double gAbs = Math.hypot(gRe, gIm);
            if (gAbs == 0.0) {
                snRe = 0.0;
                snIm = 0.0;
                rRe = 0.0;
                rIm = 0.0;
            } else {
                snRe = gRe / gAbs;
                snIm = -gIm / gAbs;
                rRe = gAbs;
                rIm = 0.0;
            }
        } else {
            double f1 = Math.max(Math.abs(fRe), Math.abs(fIm));
            double g1 = Math.max(Math.abs(gRe), Math.abs(gIm));
            double rtmin = Math.sqrt(SAFMIN);
            double rtmax = Math.sqrt(SAFMAX / 4.0);

            if (f1 > rtmin && f1 < rtmax && g1 > rtmin && g1 < rtmax) {
                double f2 = fRe * fRe + fIm * fIm;
                double g2 = gRe * gRe + gIm * gIm;
                double h2 = f2 + g2;

                if (f2 >= h2 * SAFMIN) {
                    cs = Math.sqrt(f2 / h2);
                    double rAbs = Math.sqrt(f2) / cs;
                    rRe = fRe / cs;
                    rIm = fIm / cs;
                    double denom;
                    if (f2 > rtmin && h2 < rtmax * 2.0) {
                        denom = Math.sqrt(f2 * h2);
                    } else {
                        denom = h2 * rAbs;
                    }
                    snRe = (fRe * gRe + fIm * gIm) / denom;
                    snIm = (fRe * gIm - fIm * gRe) / denom;
                } else {
                    double d = Math.sqrt(f2 * h2);
                    cs = f2 / d;
                    if (cs >= SAFMIN) {
                        rRe = fRe / cs;
                        rIm = fIm / cs;
                    } else {
                        rRe = fRe * (h2 / d);
                        rIm = fIm * (h2 / d);
                    }
                    snRe = (fRe * gRe + fIm * gIm) / d;
                    snIm = (fRe * gIm - fIm * gRe) / d;
                }
            } else {
                double u = Math.min(SAFMAX, Math.max(SAFMIN, Math.max(f1, g1)));
                double gsRe = gRe / u;
                double gsIm = gIm / u;
                double g2 = gsRe * gsRe + gsIm * gsIm;

                double fsRe, fsIm, w, h2;
                if (f1 / u < rtmin) {
                    double v = Math.min(SAFMAX, Math.max(SAFMIN, f1));
                    w = v / u;
                    fsRe = fRe / v;
                    fsIm = fIm / v;
                    double f2 = fsRe * fsRe + fsIm * fsIm;
                    h2 = f2 * w * w + g2;
                } else {
                    w = 1.0;
                    fsRe = fRe / u;
                    fsIm = fIm / u;
                    double f2 = fsRe * fsRe + fsIm * fsIm;
                    h2 = f2 + g2;
                }

                double f2 = fsRe * fsRe + fsIm * fsIm;
                if (f2 >= h2 * SAFMIN) {
                    cs = Math.sqrt(f2 / h2);
                    rRe = fsRe / cs;
                    rIm = fsIm / cs;
                    double denom;
                    if (f2 > rtmin && h2 < rtmax * 2.0) {
                        denom = Math.sqrt(f2 * h2);
                    } else {
                        denom = h2 * Math.sqrt(f2) / cs;
                    }
                    snRe = (fsRe * gsRe + fsIm * gsIm) / denom;
                    snIm = (fsRe * gsIm - fsIm * gsRe) / denom;
                } else {
                    double d = Math.sqrt(f2 * h2);
                    cs = f2 / d;
                    if (cs >= SAFMIN) {
                        rRe = fsRe / cs;
                        rIm = fsIm / cs;
                    } else {
                        rRe = fsRe * (h2 / d);
                        rIm = fsIm * (h2 / d);
                    }
                    snRe = (fsRe * gsRe + fsIm * gsIm) / d;
                    snIm = (fsRe * gsIm - fsIm * gsRe) / d;
                }

                cs *= w;
                rRe *= u;
                rIm *= u;
            }
        }

        out[off] = cs;
        out[off + 1] = snRe;
        out[off + 2] = snIm;
        out[off + 3] = rRe;
        out[off + 4] = rIm;
    }
}
