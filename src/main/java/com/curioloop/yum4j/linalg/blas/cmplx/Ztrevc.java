/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.linalg.blas.cmplx;

import com.curioloop.yum4j.linalg.blas.BLAS;

interface Ztrevc {

    static void ztrevc(char side, char howmny, int n,
                       double[] T, int tOff, int ldt,
                       double[] VL, int vlOff, int ldvl,
                       double[] VR, int vrOff, int ldvr,
                       int mm, double[] work, int workOff) {
        boolean bothv = side == 'B';
        boolean rightv = side == 'R' || bothv;
        boolean leftv = side == 'L' || bothv;
        boolean allMulQ = howmny == 'B';

        double ulp = BLAS.eps();
        double smlnum = BLAS.safmin();

        double[] Qr = null;
        double[] Ql = null;

        if (rightv && allMulQ) {
            Qr = new double[n * n * 2];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    int dIdx = (i * n + j) * 2;
                    int sIdx = (i * ldvr + j) * 2;
                    Qr[dIdx] = VR[vrOff + sIdx];
                    Qr[dIdx + 1] = VR[vrOff + sIdx + 1];
                }
            }
        }

        if (leftv && allMulQ) {
            Ql = new double[n * n * 2];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    int dIdx = (i * n + j) * 2;
                    int sIdx = (i * ldvl + j) * 2;
                    Ql[dIdx] = VL[vlOff + sIdx];
                    Ql[dIdx + 1] = VL[vlOff + sIdx + 1];
                }
            }
        }

        if (rightv) {
            for (int j = 0; j < n; j++) {
                double smin = Math.max(ulp * Math.hypot(T[tOff + (j * ldt + j) * 2], T[tOff + (j * ldt + j) * 2 + 1]),
                        smlnum);

                double[] v = new double[n * 2];
                v[j * 2] = 1.0;
                v[j * 2 + 1] = 0.0;

                for (int i = j - 1; i >= 0; i--) {
                    double tr_re = T[tOff + (i * ldt + i) * 2];
                    double tr_im = T[tOff + (i * ldt + i) * 2 + 1];
                    double tj_re = T[tOff + (j * ldt + j) * 2];
                    double tj_im = T[tOff + (j * ldt + j) * 2 + 1];

                    double denom_re = tr_re - tj_re;
                    double denom_im = tr_im - tj_im;
                    double denom_abs = Math.hypot(denom_re, denom_im);

                    double sum_re = 0.0, sum_im = 0.0;
                    for (int k = i + 1; k <= j; k++) {
                        double t_ik_re = T[tOff + (i * ldt + k) * 2];
                        double t_ik_im = T[tOff + (i * ldt + k) * 2 + 1];
                        sum_re += t_ik_re * v[k * 2] - t_ik_im * v[k * 2 + 1];
                        sum_im += t_ik_re * v[k * 2 + 1] + t_ik_im * v[k * 2];
                    }

                    if (denom_abs < smin) {
                        denom_re = smin;
                        denom_im = 0.0;
                    }

                    double inv_denom = 1.0 / (denom_re * denom_re + denom_im * denom_im);
                    v[i * 2] = -(sum_re * denom_re + sum_im * denom_im) * inv_denom;
                    v[i * 2 + 1] = -(sum_im * denom_re - sum_re * denom_im) * inv_denom;
                }

                if (allMulQ && Qr != null) {
                    for (int i = 0; i < n; i++) {
                        double sum_re = 0.0, sum_im = 0.0;
                        for (int k = 0; k < n; k++) {
                            double q_ik_re = Qr[(i * n + k) * 2];
                            double q_ik_im = Qr[(i * n + k) * 2 + 1];
                            sum_re += q_ik_re * v[k * 2] - q_ik_im * v[k * 2 + 1];
                            sum_im += q_ik_re * v[k * 2 + 1] + q_ik_im * v[k * 2];
                        }
                        work[workOff + i * 2] = sum_re;
                        work[workOff + i * 2 + 1] = sum_im;
                    }
                    for (int i = 0; i < n; i++) {
                        VR[vrOff + (i * ldvr + j) * 2] = work[workOff + i * 2];
                        VR[vrOff + (i * ldvr + j) * 2 + 1] = work[workOff + i * 2 + 1];
                    }
                } else {
                    for (int i = 0; i < n; i++) {
                        VR[vrOff + (i * ldvr + j) * 2] = v[i * 2];
                        VR[vrOff + (i * ldvr + j) * 2 + 1] = v[i * 2 + 1];
                    }
                }

                double norm = 0.0;
                for (int i = 0; i < n; i++) {
                    double re = VR[vrOff + (i * ldvr + j) * 2];
                    double im = VR[vrOff + (i * ldvr + j) * 2 + 1];
                    norm += re * re + im * im;
                }
                norm = Math.sqrt(norm);
                if (norm > 0) {
                    double remax = 1.0 / norm;
                    for (int i = 0; i < n; i++) {
                        VR[vrOff + (i * ldvr + j) * 2] *= remax;
                        VR[vrOff + (i * ldvr + j) * 2 + 1] *= remax;
                    }
                }
            }
        }

        if (leftv) {
            for (int j = 0; j < n; j++) {
                double smin = Math.max(ulp * Math.hypot(T[tOff + (j * ldt + j) * 2], T[tOff + (j * ldt + j) * 2 + 1]),
                        smlnum);

                double[] v = new double[n * 2];
                v[j * 2] = 1.0;
                v[j * 2 + 1] = 0.0;

                for (int i = j + 1; i < n; i++) {
                    double ti_re = T[tOff + (i * ldt + i) * 2];
                    double ti_im = T[tOff + (i * ldt + i) * 2 + 1];
                    double tj_re = T[tOff + (j * ldt + j) * 2];
                    double tj_im = T[tOff + (j * ldt + j) * 2 + 1];

                    double denom_re = ti_re - tj_re;
                    double denom_im = ti_im - tj_im;
                    double denom_abs = Math.hypot(denom_re, denom_im);

                    double sum_re = 0.0, sum_im = 0.0;
                    for (int k = j; k < i; k++) {
                        double t_ki_re = T[tOff + (k * ldt + i) * 2];
                        double t_ki_im = T[tOff + (k * ldt + i) * 2 + 1];
                        sum_re += v[k * 2] * t_ki_re - v[k * 2 + 1] * t_ki_im;
                        sum_im += v[k * 2] * t_ki_im + v[k * 2 + 1] * t_ki_re;
                    }

                    if (denom_abs < smin) {
                        denom_re = smin;
                        denom_im = 0.0;
                    }

                    double inv_denom = 1.0 / (denom_re * denom_re + denom_im * denom_im);
                    v[i * 2] = -(sum_re * denom_re + sum_im * denom_im) * inv_denom;
                    v[i * 2 + 1] = -(sum_im * denom_re - sum_re * denom_im) * inv_denom;
                }

                if (allMulQ && Ql != null) {
                    // The forward substitution accumulates entries of y^H; convert back to y before applying Q.
                    for (int i = 0; i < n; i++) {
                        v[i * 2 + 1] = -v[i * 2 + 1];
                    }
                    for (int i = 0; i < n; i++) {
                        double sum_re = 0.0, sum_im = 0.0;
                        for (int k = 0; k < n; k++) {
                            double ql_ik_re = Ql[(i * n + k) * 2];
                            double ql_ik_im = Ql[(i * n + k) * 2 + 1];
                            sum_re += ql_ik_re * v[k * 2] - ql_ik_im * v[k * 2 + 1];
                            sum_im += ql_ik_re * v[k * 2 + 1] + ql_ik_im * v[k * 2];
                        }
                        work[workOff + i * 2] = sum_re;
                        work[workOff + i * 2 + 1] = sum_im;
                    }
                    for (int i = 0; i < n; i++) {
                        VL[vlOff + (i * ldvl + j) * 2] = work[workOff + i * 2];
                        VL[vlOff + (i * ldvl + j) * 2 + 1] = work[workOff + i * 2 + 1];
                    }
                } else {
                    for (int i = 0; i < n; i++) {
                        VL[vlOff + (i * ldvl + j) * 2] = v[i * 2];
                        VL[vlOff + (i * ldvl + j) * 2 + 1] = -v[i * 2 + 1];
                    }
                }

                double norm = 0.0;
                for (int i = 0; i < n; i++) {
                    double re = VL[vlOff + (i * ldvl + j) * 2];
                    double im = VL[vlOff + (i * ldvl + j) * 2 + 1];
                    norm += re * re + im * im;
                }
                norm = Math.sqrt(norm);
                if (norm > 0) {
                    double remax = 1.0 / norm;
                    for (int i = 0; i < n; i++) {
                        VL[vlOff + (i * ldvl + j) * 2] *= remax;
                        VL[vlOff + (i * ldvl + j) * 2 + 1] *= remax;
                    }
                }
            }
        }
    }
}
