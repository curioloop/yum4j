/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.linalg.blas.cmplx;

import com.curioloop.yum4j.linalg.blas.BLAS;
import com.curioloop.yum4j.linalg.blas.Dlas2;
import com.curioloop.yum4j.linalg.blas.Dlasq1;

import static java.lang.Math.sqrt;

interface Zbdsqr {

    static int zbdsqr(char uplo, int n, int ncvt, int nru, int ncc, double[] d, int dOff, double[] e, int eOff, double[] vt, int vtOff, int ldvt, double[] u, int uOff, int ldu, double[] c, int cOff, int ldc, double[] work, int workOff) {
        char cuplo = Character.toUpperCase(uplo);
        int info = 0;
        boolean lower = cuplo == 'L';

        if (cuplo != 'U' && !lower) {
            return -1;
        }
        if (n < 0) return -2;
        if (ncvt < 0) return -3;
        if (nru < 0) return -4;
        if (ncc < 0) return -5;
        if ((ncvt == 0 && ldvt < 1) || (ncvt > 0 && ldvt < max(1, n))) return -8;
        if (ldu < max(1, nru)) return -10;
        if ((ncc == 0 && ldc < 1) || (ncc > 0 && ldc < max(1, n))) return -12;

        if (n == 0) return 0;
        if (n == 1) {
            if (ncvt > 0 && d[dOff] < 0) {
                for (int i = 0; i < ncvt; i++) {
                    int off = vtOff + i * 2;
                    vt[off] = -vt[off];
                    vt[off + 1] = -vt[off + 1];
                }
                d[dOff] = -d[dOff];
            }
            return 0;
        }

        boolean rotate = (ncvt > 0) || (nru > 0) || (ncc > 0);

        if (!rotate) {
            info = Dlasq1.dlasq1(n, d, dOff, e, eOff, work, workOff);
            if (info != 2) return info;
            info = 0;
        }

        int nm1 = n - 1;
        int nm12 = nm1 + nm1;
        int nm13 = nm12 + nm1;
        int idir = 0;
        double[] tmp = new double[6];

        double eps = BLAS.eps();
        double unfl = BLAS.dlamch('S');

        if (lower) {
            for (int i = 0; i < n - 1; i++) {
            BLAS.dlartg(d[dOff + i], e[eOff + i], tmp, 0);
                double cs = tmp[0];
                double sn = tmp[1];
                double r = tmp[2];
                d[dOff + i] = r;
                e[eOff + i] = sn * d[dOff + i + 1];
                d[dOff + i + 1] = cs * d[dOff + i + 1];
                work[workOff + i] = cs;
                work[workOff + nm1 + i] = sn;
            }

            if (nru > 0) {
                Zlasr.zlasr('R', 'V', 'F', nru, n, work, workOff, work, workOff + n - 1, u, uOff, ldu);
            }
            if (ncc > 0) {
                Zlasr.zlasr('L', 'V', 'F', n, ncc, work, workOff, work, workOff + n - 1, c, cOff, ldc);
            }
        }

        double tolmul = Math.max(10.0, Math.min(100.0, Math.pow(eps, -0.125)));
        double tol = tolmul * eps;

        double smax = 0.0;
        for (int i = 0; i < n; i++) {
            smax = Math.max(smax, Math.abs(d[dOff + i]));
        }
        for (int i = 0; i < n - 1; i++) {
            smax = Math.max(smax, Math.abs(e[eOff + i]));
        }

        double smin = 0.0;
        double thresh = 0.0;
        if (tol >= 0) {
            double sminoA = Math.abs(d[dOff]);
            if (sminoA != 0) {
                double mu = sminoA;
                for (int i = 1; i < n; i++) {
                    mu = Math.abs(d[dOff + i]) * (mu / (mu + Math.abs(e[eOff + i - 1])));
                    sminoA = Math.min(sminoA, mu);
                    if (sminoA == 0) break;
                }
            }
            sminoA = sminoA / sqrt(n);
            thresh = Math.max(tol * sminoA, 6.0 * (n * (n * unfl)));
        } else {
            thresh = Math.max(Math.abs(tol) * smax, 6.0 * (n * (n * unfl)));
        }

        int maxitdivn = 6 * n;
        int iterdivn = 0;
        int iter = -1;
        int oldll = -1;
        int oldm = -1;

        int m = n;

        while (true) {
            if (m <= 1) break;
            if (iter >= n) {
                iter -= n;
                iterdivn++;
                if (iterdivn >= maxitdivn) {
                    info = 0;
                    for (int i = 0; i < n - 1; i++) {
                        if (e[eOff + i] != 0) info++;
                    }
                    return info;
                }
            }

            if (tol < 0 && Math.abs(d[dOff + m - 1]) <= thresh) {
                d[dOff + m - 1] = 0;
            }
            smax = Math.abs(d[dOff + m - 1]);
            boolean foundZero = false;
            int ll = 0;
            for (int lll = 1; lll < m; lll++) {
                ll = m - lll - 1;
                double abss = Math.abs(d[dOff + ll]);
                double abse = Math.abs(e[eOff + ll]);
                if (tol < 0 && abss <= thresh) {
                    d[dOff + ll] = 0;
                }
                if (abse <= thresh) {
                    e[eOff + ll] = 0;
                    foundZero = true;
                    break;
                }
                smin = Math.min(smin, abss);
                smax = Math.max(Math.max(smax, abss), abse);
            }

            if (foundZero) {
                if (ll == m - 2) {
                    m--;
                    continue;
                }
                ll++;
            } else {
                ll = 0;
            }

            if (ll == m - 2) {
                Dlas2.dlasv2(d[dOff + m - 2], e[eOff + m - 2], d[dOff + m - 1], tmp, 0);
                double sigmn = tmp[0];
                double sigmx = tmp[1];
                double sinr = tmp[2];
                double cosr = tmp[3];
                double sinl = tmp[4];
                double cosl = tmp[5];
                d[dOff + m - 2] = sigmx;
                e[eOff + m - 2] = 0;
                d[dOff + m - 1] = sigmn;

                if (ncvt > 0) {
                    Zrot.zrot(ncvt, vt, vtOff + (m - 2) * ldvt * 2, 1, vt, vtOff + (m - 1) * ldvt * 2, 1, cosr, sinr, 0);
                }
                if (nru > 0) {
                    Zrot.zrot(nru, u, uOff + (m - 2) * 2, ldu, u, uOff + (m - 1) * 2, ldu, cosl, sinl, 0);
                }
                if (ncc > 0) {
                    Zrot.zrot(ncc, c, cOff + (m - 2) * ldc * 2, 1, c, cOff + (m - 1) * ldc * 2, 1, cosl, sinl, 0);
                }
                m -= 2;
                continue;
            }

            if (ll > oldm || m < oldll) {
                if (Math.abs(d[dOff + ll]) >= Math.abs(d[dOff + m - 1])) {
                    idir = 1;
                } else {
                    idir = 2;
                }
            }

            if (idir == 1) {
                if (Math.abs(e[eOff + m - 2]) <= Math.abs(tol) * Math.abs(d[dOff + m - 1]) || (tol < 0 && Math.abs(e[eOff + m - 2]) <= thresh)) {
                    e[eOff + m - 2] = 0;
                    continue;
                }

                if (tol >= 0) {
                    double mu = Math.abs(d[dOff + ll]);
                    double localSmin = mu;
                    for (int lll = ll; lll < m - 1; lll++) {
                        if (Math.abs(e[eOff + lll]) <= tol * mu) {
                            e[eOff + lll] = 0;
                            continue;
                        }
                        mu = Math.abs(d[dOff + lll + 1]) * (mu / (mu + Math.abs(e[eOff + lll])));
                        localSmin = Math.min(localSmin, mu);
                    }
                    smin = localSmin;
                }
            } else {
                if (Math.abs(e[eOff + ll]) <= Math.abs(tol) * Math.abs(d[dOff + ll]) || (tol < 0 && Math.abs(e[eOff + ll]) <= thresh)) {
                    e[eOff + ll] = 0;
                    continue;
                }

                if (tol >= 0) {
                    double mu = Math.abs(d[dOff + m - 1]);
                    double localSmin = mu;
                    for (int lll = m - 2; lll >= ll; lll--) {
                        if (Math.abs(e[eOff + lll]) <= tol * mu) {
                            e[eOff + lll] = 0;
                            continue;
                        }
                        mu = Math.abs(d[dOff + lll]) * (mu / (mu + Math.abs(e[eOff + lll])));
                        localSmin = Math.min(localSmin, mu);
                    }
                    smin = localSmin;
                }
            }

            oldll = ll;
            oldm = m;

            double shift = 0.0;
            if (tol >= 0 && n * tol * (smin / smax) <= Math.max(eps, 0.01 * tol)) {
                shift = 0.0;
            } else {
                if (idir == 1) {
                    Dlas2.dlas2(d[dOff + m - 2], e[eOff + m - 2], d[dOff + m - 1], tmp, 0);
                    shift = tmp[0];
                } else {
                    Dlas2.dlas2(d[dOff + ll], e[eOff + ll], d[dOff + ll + 1], tmp, 0);
                    shift = tmp[0];
                }

                double sll = idir == 1 ? Math.abs(d[dOff + ll]) : Math.abs(d[dOff + m - 1]);
                if (sll > 0 && Math.pow(shift / sll, 2) < eps) {
                    shift = 0.0;
                }
            }

            iter += m - ll;

            if (shift == 0) {
                if (idir == 1) {
                    double cs = 1.0;
                    double oldcs = 1.0;
                    double oldsn = 0.0;
                    for (int i = ll; i < m - 1; i++) {
                        BLAS.dlartg(d[dOff + i] * cs, e[eOff + i], tmp, 0);
                        double cs_new = tmp[0];
                        double sn = tmp[1];
                        double r = tmp[2];
                        if (i > ll) {
                            e[eOff + i - 1] = sn * r;
                        }
                        BLAS.dlartg(oldcs * r, d[dOff + i + 1] * sn, tmp, 0);
                        double oldcs_new = tmp[0];
                        oldsn = tmp[1];
                        d[dOff + i] = tmp[2];
                        work[workOff + i - ll] = cs_new;
                        work[workOff + i - ll + nm1] = sn;
                        work[workOff + i - ll + nm12] = oldcs_new;
                        work[workOff + i - ll + nm13] = oldsn;
                        cs = cs_new;
                        oldcs = oldcs_new;
                    }
                    double h = d[dOff + m - 1] * cs;
                    d[dOff + m - 1] = h * oldcs;
                    e[eOff + m - 2] = h * oldsn;

                    if (ncvt > 0) {
                        Zlasr.zlasr('L', 'V', 'F', m - ll, ncvt, work, workOff, work, workOff + nm1, vt, vtOff + ll * ldvt * 2, ldvt);
                    }
                    if (nru > 0) {
                        Zlasr.zlasr('R', 'V', 'F', nru, m - ll, work, workOff + nm12, work, workOff + nm13, u, uOff + ll * 2, ldu);
                    }
                    if (ncc > 0) {
                        Zlasr.zlasr('L', 'V', 'F', m - ll, ncc, work, workOff + nm12, work, workOff + nm13, c, cOff + ll * ldc * 2, ldc);
                    }

                    if (Math.abs(e[eOff + m - 2]) <= thresh) {
                        e[eOff + m - 2] = 0;
                    }
                } else {
                    double cs = 1.0;
                    double oldcs = 1.0;
                    double oldsn = 0.0;
                    for (int i = m - 1; i >= ll + 1; i--) {
                        BLAS.dlartg(d[dOff + i] * cs, e[eOff + i - 1], tmp, 0);
                        double cs_new = tmp[0];
                        double sn = tmp[1];
                        double r = tmp[2];
                        if (i < m - 1) {
                            e[eOff + i] = sn * r;
                        }
                        BLAS.dlartg(oldcs * r, d[dOff + i - 1] * sn, tmp, 0);
                        double oldcs_new = tmp[0];
                        oldsn = tmp[1];
                        d[dOff + i] = tmp[2];
                        work[workOff + i - ll - 1] = cs_new;
                        work[workOff + i - ll - 1 + nm1] = -sn;
                        work[workOff + i - ll - 1 + nm12] = oldcs_new;
                        work[workOff + i - ll - 1 + nm13] = -oldsn;
                        cs = cs_new;
                        oldcs = oldcs_new;
                    }
                    double h = d[dOff + ll] * cs;
                    d[dOff + ll] = h * oldcs;
                    e[eOff + ll] = h * oldsn;

                    if (ncvt > 0) {
                        Zlasr.zlasr('L', 'V', 'B', m - ll, ncvt, work, workOff + nm12, work, workOff + nm13, vt, vtOff + ll * ldvt * 2, ldvt);
                    }
                    if (nru > 0) {
                        Zlasr.zlasr('R', 'V', 'B', nru, m - ll, work, workOff, work, workOff + nm1, u, uOff + ll * 2, ldu);
                    }
                    if (ncc > 0) {
                        Zlasr.zlasr('L', 'V', 'B', m - ll, ncc, work, workOff, work, workOff + nm1, c, cOff + ll * ldc * 2, ldc);
                    }

                    if (Math.abs(e[eOff + ll]) <= thresh) {
                        e[eOff + ll] = 0;
                    }
                }
            } else {
                if (idir == 1) {
                    double f = (Math.abs(d[dOff + ll]) - shift) * (signum(d[dOff + ll]) + shift / d[dOff + ll]);
                    double g = e[eOff + ll];
                    for (int i = ll; i < m - 1; i++) {
                        BLAS.dlartg(f, g, tmp, 0);
                        double cosr = tmp[0];
                        double sinr = tmp[1];
                        double r = tmp[2];
                        if (i > ll) {
                            e[eOff + i - 1] = r;
                        }
                        f = cosr * d[dOff + i] + sinr * e[eOff + i];
                        e[eOff + i] = cosr * e[eOff + i] - sinr * d[dOff + i];
                        g = sinr * d[dOff + i + 1];
                        d[dOff + i + 1] = cosr * d[dOff + i + 1];
                        BLAS.dlartg(f, g, tmp, 0);
                        double cosl = tmp[0];
                        double sinl = tmp[1];
                        d[dOff + i] = tmp[2];
                        f = cosl * e[eOff + i] + sinl * d[dOff + i + 1];
                        d[dOff + i + 1] = cosl * d[dOff + i + 1] - sinl * e[eOff + i];
                        if (i < m - 2) {
                            g = sinl * e[eOff + i + 1];
                            e[eOff + i + 1] = cosl * e[eOff + i + 1];
                        }
                        work[workOff + i - ll] = cosr;
                        work[workOff + i - ll + nm1] = sinr;
                        work[workOff + i - ll + nm12] = cosl;
                        work[workOff + i - ll + nm13] = sinl;
                    }
                    e[eOff + m - 2] = f;

                    if (ncvt > 0) {
                        Zlasr.zlasr('L', 'V', 'F', m - ll, ncvt, work, workOff, work, workOff + nm1, vt, vtOff + ll * ldvt * 2, ldvt);
                    }
                    if (nru > 0) {
                        Zlasr.zlasr('R', 'V', 'F', nru, m - ll, work, workOff + nm12, work, workOff + nm13, u, uOff + ll * 2, ldu);
                    }
                    if (ncc > 0) {
                        Zlasr.zlasr('L', 'V', 'F', m - ll, ncc, work, workOff + nm12, work, workOff + nm13, c, cOff + ll * ldc * 2, ldc);
                    }

                    if (Math.abs(e[eOff + m - 2]) <= thresh) {
                        e[eOff + m - 2] = 0;
                    }
                } else {
                    double f = (Math.abs(d[dOff + m - 1]) - shift) * (signum(d[dOff + m - 1]) + shift / d[dOff + m - 1]);
                    double g = e[eOff + m - 2];
                    for (int i = m - 1; i >= ll + 1; i--) {
                        BLAS.dlartg(f, g, tmp, 0);
                        double cosr = tmp[0];
                        double sinr = tmp[1];
                        double r = tmp[2];
                        if (i < m - 1) {
                            e[eOff + i] = r;
                        }
                        f = cosr * d[dOff + i] + sinr * e[eOff + i - 1];
                        e[eOff + i - 1] = cosr * e[eOff + i - 1] - sinr * d[dOff + i];
                        g = sinr * d[dOff + i - 1];
                        d[dOff + i - 1] = cosr * d[dOff + i - 1];
                        BLAS.dlartg(f, g, tmp, 0);
                        double cosl = tmp[0];
                        double sinl = tmp[1];
                        d[dOff + i] = tmp[2];
                        f = cosl * e[eOff + i - 1] + sinl * d[dOff + i - 1];
                        d[dOff + i - 1] = cosl * d[dOff + i - 1] - sinl * e[eOff + i - 1];
                        if (i > ll + 1) {
                            g = sinl * e[eOff + i - 2];
                            e[eOff + i - 2] = cosl * e[eOff + i - 2];
                        }
                        work[workOff + i - ll - 1] = cosr;
                        work[workOff + i - ll - 1 + nm1] = -sinr;
                        work[workOff + i - ll - 1 + nm12] = cosl;
                        work[workOff + i - ll - 1 + nm13] = -sinl;
                    }
                    e[eOff + ll] = f;

                    if (Math.abs(e[eOff + ll]) <= thresh) {
                        e[eOff + ll] = 0;
                    }

                    if (ncvt > 0) {
                        Zlasr.zlasr('L', 'V', 'B', m - ll, ncvt, work, workOff + nm12, work, workOff + nm13, vt, vtOff + ll * ldvt * 2, ldvt);
                    }
                    if (nru > 0) {
                        Zlasr.zlasr('R', 'V', 'B', nru, m - ll, work, workOff, work, workOff + nm1, u, uOff + ll * 2, ldu);
                    }
                    if (ncc > 0) {
                        Zlasr.zlasr('L', 'V', 'B', m - ll, ncc, work, workOff, work, workOff + nm1, c, cOff + ll * ldc * 2, ldc);
                    }
                }
            }
        }

        for (int i = 0; i < n; i++) {
            if (d[dOff + i] == 0) {
                d[dOff + i] = 0;
            } else if (d[dOff + i] < 0) {
                d[dOff + i] = -d[dOff + i];
                if (ncvt > 0) {
                    for (int j = 0; j < ncvt; j++) {
                        int off = vtOff + i * ldvt * 2 + j * 2;
                        vt[off] = -vt[off];
                        vt[off + 1] = -vt[off + 1];
                    }
                }
            }
        }

        for (int i = 0; i < n - 1; i++) {
            int isub = 0;
            smin = d[dOff];
            for (int j = 1; j < n - i; j++) {
                if (d[dOff + j] <= smin) {
                    isub = j;
                    smin = d[dOff + j];
                }
            }
            if (isub != n - i - 1) {
                double temp = d[dOff + isub];
                d[dOff + isub] = d[dOff + n - i - 1];
                d[dOff + n - i - 1] = temp;

                if (ncvt > 0) {
                    for (int j = 0; j < ncvt; j++) {
                        int off1 = vtOff + isub * ldvt * 2 + j * 2;
                        int off2 = vtOff + (n - i - 1) * ldvt * 2 + j * 2;
                        double tre = vt[off1];
                        double tim = vt[off1 + 1];
                        vt[off1] = vt[off2];
                        vt[off1 + 1] = vt[off2 + 1];
                        vt[off2] = tre;
                        vt[off2 + 1] = tim;
                    }
                }
                if (nru > 0) {
                    for (int j = 0; j < nru; j++) {
                        int off1 = uOff + j * ldu * 2 + isub * 2;
                        int off2 = uOff + j * ldu * 2 + (n - i - 1) * 2;
                        double tre = u[off1];
                        double tim = u[off1 + 1];
                        u[off1] = u[off2];
                        u[off1 + 1] = u[off2 + 1];
                        u[off2] = tre;
                        u[off2 + 1] = tim;
                    }
                }
                if (ncc > 0) {
                    for (int j = 0; j < ncc; j++) {
                        int off1 = cOff + isub * ldc * 2 + j * 2;
                        int off2 = cOff + (n - i - 1) * ldc * 2 + j * 2;
                        double tre = c[off1];
                        double tim = c[off1 + 1];
                        c[off1] = c[off2];
                        c[off1 + 1] = c[off2 + 1];
                        c[off2] = tre;
                        c[off2 + 1] = tim;
                    }
                }
            }
        }

        return info;
    }

    static int max(int a, int b) {
        return a > b ? a : b;
    }

    static double max(double a, double b, double c) {
        return Math.max(Math.max(a, b), c);
    }

    static double signum(double x) {
        return x >= 0 ? 1.0 : -1.0;
    }
}
