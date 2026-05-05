package com.curioloop.yum4j.linalg.blas.cmplx;

import com.curioloop.yum4j.linalg.blas.*;

interface Zsteqr {

    static final double DLAMCH_E = 0x1p-53;
    static final double DLAMCH_S = 0x1p-1022;
    static final int MAXIT = 30;

    static int zsteqr(char compz, int n, double[] d, int dOff, double[] e, int eOff,
                      double[] Z, int zOff, int ldz, double[] work, int wOff) {
        if (n < 0) return -2;
        if (n == 0) return 0;

        int icompz;
        char cz = Character.toUpperCase(compz);
        if (cz == 'N') icompz = 0;
        else if (cz == 'V') icompz = 1;
        else if (cz == 'I') icompz = 2;
        else return -1;

        if (ldz < 1 || (icompz > 0 && ldz < n)) return -6;

        if (n == 1) {
            if (icompz == 2 && Z != null) {
                Z[zOff] = 1.0;
                Z[zOff + 1] = 0.0;
            }
            return 0;
        }

        double eps = DLAMCH_E;
        double eps2 = eps * eps;
        double safmin = DLAMCH_S;
        double safmax = 1.0 / safmin;
        double ssfmax = Math.sqrt(safmax) / 3.0;
        double ssfmin = Math.sqrt(safmin) / eps2;

        if (icompz == 2) {
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    int pos = zOff + i * ldz * 2 + j * 2;
                    Z[pos] = (i == j) ? 1.0 : 0.0;
                    Z[pos + 1] = 0.0;
                }
            }
        }

        int nmaxit = n * MAXIT;
        int jtot = 0;
        double[] temp = {0, 0, 0, 0};

        int l1 = 0;
        int nm1 = n - 1;

        while (true) {
            if (l1 > n - 1) {
                if (icompz == 0) {
                    BLAS.dlasrt('I', n, d, dOff);
                } else {
                    for (int ii = 1; ii < n; ii++) {
                        int i = ii - 1;
                        int k = i;
                        double p = d[dOff + i];
                        for (int j = ii; j < n; j++) {
                            if (d[dOff + j] < p) {
                                k = j;
                                p = d[dOff + j];
                            }
                        }
                        if (k != i) {
                            d[dOff + k] = d[dOff + i];
                            d[dOff + i] = p;
                            if (Z != null) {
                                for (int row = 0; row < n; row++) {
                                    int posI = zOff + row * ldz * 2 + i * 2;
                                    int posK = zOff + row * ldz * 2 + k * 2;
                                    double tr = Z[posI];
                                    double ti = Z[posI + 1];
                                    Z[posI] = Z[posK];
                                    Z[posI + 1] = Z[posK + 1];
                                    Z[posK] = tr;
                                    Z[posK + 1] = ti;
                                }
                            }
                        }
                    }
                }
                return 0;
            }

            if (l1 > 0) {
                e[eOff + l1 - 1] = 0;
            }

            int m = n - 1;
            if (l1 <= nm1) {
                for (m = l1; m < nm1; m++) {
                    double test = Math.abs(e[eOff + m]);
                    if (test == 0.0) break;
                    if (test <= (Math.sqrt(Math.abs(d[dOff + m])) * Math.sqrt(Math.abs(d[dOff + m + 1]))) * eps) {
                        e[eOff + m] = 0.0;
                        break;
                    }
                }
            }

            int l = l1;
            int lsv = l;
            int lend = m;
            int lendsv = lend;
            l1 = m + 1;
            if (lend == l) continue;

            int len = lend - l + 1;
            double anorm = BLAS.dlanst(BLAS.Norm.MaxAbs, len, d, dOff + l, e, eOff + l);
            int iscale = 0;
            if (anorm == 0.0) continue;
            if (anorm > ssfmax) {
                iscale = 1;
                BLAS.dlascl('G', anorm, ssfmax, len, d, dOff + l, 1);
                BLAS.dlascl('G', anorm, ssfmax, len - 1, e, eOff + l, 1);
            } else if (anorm < ssfmin) {
                iscale = 2;
                BLAS.dlascl('G', anorm, ssfmin, len, d, dOff + l, 1);
                BLAS.dlascl('G', anorm, ssfmin, len - 1, e, eOff + l, 1);
            }

            if (Math.abs(d[dOff + lend]) < Math.abs(d[dOff + l])) {
                lend = lsv;
                l = lendsv;
            }

            if (lend > l) {
                while (true) {
                    if (l != lend) {
                        for (m = l; m < lend; m++) {
                            double v = Math.abs(e[eOff + m]);
                            if (v * v <= (eps2 * Math.abs(d[dOff + m])) * Math.abs(d[dOff + m + 1]) + safmin) {
                                break;
                            }
                        }
                    } else {
                        m = lend;
                    }

                    if (m < lend) {
                        e[eOff + m] = 0.0;
                    }

                    double p = d[dOff + l];
                    if (m == l) {
                        l++;
                        if (l > lend) break;
                        continue;
                    }

                    if (m == l + 1) {
                        if (icompz > 0) {
                            Dlae2.dlaev2(d[dOff + l], e[eOff + l], d[dOff + l + 1], temp, 0);
                            d[dOff + l] = temp[0];
                            d[dOff + l + 1] = temp[1];
                            work[wOff + l] = temp[2];
                            work[wOff + n - 1 + l] = temp[3];
                            applyRotationRight(n, 2, work, wOff + l, work, wOff + n - 1 + l, Z, zOff, l, ldz, 'B');
                        } else {
                            Dlae2.dlae2(d[dOff + l], e[eOff + l], d[dOff + l + 1], d, dOff + l);
                        }
                        e[eOff + l] = 0.0;
                        l += 2;
                        if (l > lend) break;
                        continue;
                    }

                    if (jtot >= nmaxit) break;
                    jtot++;

                    double g = (d[dOff + l + 1] - p) / (2.0 * e[eOff + l]);
                    double r = Math.hypot(g, 1.0);
                    g = d[dOff + m] - p + e[eOff + l] / (g + Math.copySign(r, g));
                    double s = 1.0;
                    double c = 1.0;
                    p = 0.0;

                    for (int i = m - 1; i >= l; i--) {
                        double f = s * e[eOff + i];
                        double b = c * e[eOff + i];
                        BLAS.dlartg(g, f, temp, 0);
                        c = temp[0];
                        s = temp[1];
                        r = temp[2];
                        if (i != m - 1) {
                            e[eOff + i + 1] = r;
                        }
                        g = d[dOff + i + 1] - p;
                        r = (d[dOff + i] - g) * s + 2.0 * c * b;
                        p = s * r;
                        d[dOff + i + 1] = g + p;
                        g = c * r - b;

                        if (icompz > 0) {
                            work[wOff + i] = c;
                            work[wOff + n - 1 + i] = -s;
                        }
                    }

                    if (icompz > 0) {
                        int mm = m - l + 1;
                        applyRotationRight(n, mm, work, wOff + l, work, wOff + n - 1 + l, Z, zOff, l, ldz, 'B');
                    }
                    d[dOff + l] -= p;
                    e[eOff + l] = g;
                }
            } else {
                while (true) {
                    if (l != lend) {
                        for (m = l; m > lend; m--) {
                            double v = Math.abs(e[eOff + m - 1]);
                            if (v * v <= (eps2 * Math.abs(d[dOff + m]) * Math.abs(d[dOff + m - 1]) + safmin)) {
                                break;
                            }
                        }
                    } else {
                        m = lend;
                    }

                    if (m > lend) {
                        e[eOff + m - 1] = 0.0;
                    }

                    double p = d[dOff + l];
                    if (m == l) {
                        l--;
                        if (l < lend) break;
                        continue;
                    }

                    if (m == l - 1) {
                        if (icompz > 0) {
                            Dlae2.dlaev2(d[dOff + l - 1], e[eOff + l - 1], d[dOff + l], temp, 0);
                            d[dOff + l - 1] = temp[0];
                            d[dOff + l] = temp[1];
                            work[wOff + m] = temp[2];
                            work[wOff + n - 1 + m] = temp[3];
                            applyRotationRight(n, 2, work, wOff + m, work, wOff + n - 1 + m, Z, zOff, l - 1, ldz, 'F');
                        } else {
                            Dlae2.dlae2(d[dOff + l - 1], e[eOff + l - 1], d[dOff + l], d, dOff + l - 1);
                        }
                        e[eOff + l - 1] = 0.0;
                        l -= 2;
                        if (l < lend) break;
                        continue;
                    }

                    if (jtot >= nmaxit) break;
                    jtot++;

                    double g = (d[dOff + l - 1] - p) / (2.0 * e[eOff + l - 1]);
                    double r = Math.hypot(g, 1.0);
                    g = d[dOff + m] - p + e[eOff + l - 1] / (g + Math.copySign(r, g));
                    double s = 1.0;
                    double c = 1.0;
                    p = 0.0;

                    for (int i = m; i < l; i++) {
                        double f = s * e[eOff + i];
                        double b = c * e[eOff + i];
                        BLAS.dlartg(g, f, temp, 0);
                        c = temp[0];
                        s = temp[1];
                        r = temp[2];
                        if (i != m) {
                            e[eOff + i - 1] = r;
                        }
                        g = d[dOff + i] - p;
                        r = (d[dOff + i + 1] - g) * s + 2.0 * c * b;
                        p = s * r;
                        d[dOff + i] = g + p;
                        g = c * r - b;

                        if (icompz > 0) {
                            work[wOff + i] = c;
                            work[wOff + n - 1 + i] = s;
                        }
                    }

                    if (icompz > 0) {
                        int mm = l - m + 1;
                        applyRotationRight(n, mm, work, wOff + m, work, wOff + n - 1 + m, Z, zOff, m, ldz, 'F');
                    }
                    d[dOff + l] -= p;
                    e[eOff + l - 1] = g;
                }
            }

            switch (iscale) {
                case 1:
                    BLAS.dlascl('G', ssfmax, anorm, lendsv - lsv + 1, d, dOff + lsv, 1);
                    BLAS.dlascl('G', ssfmax, anorm, lendsv - lsv, e, eOff + lsv, 1);
                    break;
                case 2:
                    BLAS.dlascl('G', ssfmin, anorm, lendsv - lsv + 1, d, dOff + lsv, 1);
                    BLAS.dlascl('G', ssfmin, anorm, lendsv - lsv, e, eOff + lsv, 1);
                    break;
            }

            if (jtot >= nmaxit) break;
        }

        for (int i = 0; i < n - 1; i++) {
            if (e[eOff + i] != 0.0) return i + 1;
        }
        return 0;
    }

    static void applyRotationRight(int n, int mm, double[] c, int cOff, double[] s, int sOff,
                                   double[] Z, int zOff, int zColStart, int ldz, char dir) {
        for (int row = 0; row < n; row++) {
            int zBase = zOff + row * ldz * 2 + zColStart * 2;
            if (dir == 'F') {
                for (int j = 0; j < mm - 1; j++) {
                    double cj = c[cOff + j];
                    double sj = s[sOff + j];
                    int pos1 = zBase + j * 2;
                    int pos2 = zBase + (j + 1) * 2;
                    double z1r = Z[pos1], z1i = Z[pos1 + 1];
                    double z2r = Z[pos2], z2i = Z[pos2 + 1];
                    Z[pos1] = cj * z1r + sj * z2r;
                    Z[pos1 + 1] = cj * z1i + sj * z2i;
                    Z[pos2] = -sj * z1r + cj * z2r;
                    Z[pos2 + 1] = -sj * z1i + cj * z2i;
                }
            } else {
                for (int j = mm - 2; j >= 0; j--) {
                    double cj = c[cOff + j];
                    double sj = s[sOff + j];
                    int pos1 = zBase + j * 2;
                    int pos2 = zBase + (j + 1) * 2;
                    double z1r = Z[pos1], z1i = Z[pos1 + 1];
                    double z2r = Z[pos2], z2i = Z[pos2 + 1];
                    Z[pos1] = cj * z1r + sj * z2r;
                    Z[pos1 + 1] = cj * z1i + sj * z2i;
                    Z[pos2] = -sj * z1r + cj * z2r;
                    Z[pos2 + 1] = -sj * z1i + cj * z2i;
                }
            }
        }
    }
}
