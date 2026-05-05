/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.linalg.blas.cmplx;

import com.curioloop.yum4j.linalg.blas.BLAS;

interface Zgebal {

    static void zgebal(char job, int n, double[] A, int aOff, int lda, double[] scale, int scaleOff, double[] out, int outOff) {
        if (n == 0) {
            out[outOff] = 0;
            out[outOff + 1] = 0;
            return;
        }

        char jobUpper = Character.toUpperCase(job);

        int k = 0;
        int l = n - 1;

        if (jobUpper == 'N') {
            for (int i = 0; i < n; i++) {
                scale[scaleOff + i] = 1.0;
            }
            out[outOff] = k;
            out[outOff + 1] = l;
            return;
        }

        if (jobUpper != 'S') {
            boolean noConv = true;
            while (noConv) {
                noConv = false;
                for (int i = l; i >= 0; i--) {
                    boolean canSwap = true;
                    for (int j = 0; j <= l; j++) {
                        if (i != j) {
                            int idx = aOff + (i * lda + j) * 2;
                            if (A[idx] != 0 || A[idx + 1] != 0) {
                                canSwap = false;
                                break;
                            }
                        }
                    }
                    if (canSwap) {
                        scale[scaleOff + l] = i;
                        if (i != l) {
                            Zswap.zswap(n, A, aOff + i * 2, lda, A, aOff + l * 2, lda);
                            Zswap.zswap(l + 1, A, aOff + i * lda * 2, 1, A, aOff + l * lda * 2, 1);
                        }
                        noConv = true;
                        if (l == 0) {
                            scale[scaleOff] = 1;
                            out[outOff] = 0;
                            out[outOff + 1] = 0;
                            return;
                        }
                        l--;
                        break;
                    }
                }
            }

            noConv = true;
            while (noConv) {
                noConv = false;
                for (int j = k; j <= l; j++) {
                    boolean canSwap = true;
                    for (int i = k; i <= l; i++) {
                        if (i != j) {
                            int idx = aOff + (i * lda + j) * 2;
                            if (A[idx] != 0 || A[idx + 1] != 0) {
                                canSwap = false;
                                break;
                            }
                        }
                    }
                    if (canSwap) {
                        scale[scaleOff + k] = j;
                        if (j != k) {
                            Zswap.zswap(l + 1, A, aOff + j * lda * 2, 1, A, aOff + k * lda * 2, 1);
                            Zswap.zswap(n - k, A, aOff + k * lda * 2 + j * 2, lda, A, aOff + k * lda * 2 + k * 2, lda);
                        }
                        noConv = true;
                        k++;
                        break;
                    }
                }
            }
        }

        for (int i = k; i <= l; i++) {
            scale[scaleOff + i] = 1.0;
        }

        if (jobUpper == 'P') {
            out[outOff] = k;
            out[outOff + 1] = l;
            return;
        }

        final double sclfac = 2.0;
        final double factor = 0.95;
        double sfmin1 = BLAS.dlamch('S') / BLAS.dlamch('P');
        double sfmax1 = 1.0 / sfmin1;
        double sfmin2 = sfmin1 * sclfac;
        double sfmax2 = 1.0 / sfmin2;

        boolean noConv = true;
        while (noConv) {
            noConv = false;
            for (int i = k; i <= l; i++) {
                double c = ZLAS.dznrm2(l - k + 1, A, aOff + k * lda * 2 + i * 2, lda);
                double r = ZLAS.dznrm2(l - k + 1, A, aOff + i * lda * 2 + k * 2, 1);

                int ica = Izamax.izamax(l + 1, A, aOff + i * 2, lda);
                double ca = Math.hypot(A[aOff + ica * lda * 2 + i * 2], A[aOff + ica * lda * 2 + i * 2 + 1]);

                int ira = Izamax.izamax(n - k, A, aOff + i * lda * 2 + k * 2, 1);
                double ra = Math.hypot(A[aOff + i * lda * 2 + (k + ira) * 2], A[aOff + i * lda * 2 + (k + ira) * 2 + 1]);

                if (c == 0 || r == 0) continue;

                if (Double.isNaN(c + ca + r + ra)) {
                    throw new RuntimeException("lapack: NaN");
                }

                double g = r / sclfac;
                double f = 1.0;
                double s = c + r;

                while (c < g && Math.max(f, Math.max(c, ca)) < sfmax2 && Math.min(r, Math.min(g, ra)) > sfmin2) {
                    f *= sclfac;
                    c *= sclfac;
                    ca *= sclfac;
                    g /= sclfac;
                    r /= sclfac;
                    ra /= sclfac;
                }

                g = c / sclfac;
                while (g >= r && Math.max(r, ra) < sfmax2 && Math.min(Math.min(f, c), Math.min(g, ca)) > sfmin2) {
                    f /= sclfac;
                    c /= sclfac;
                    g /= sclfac;
                    ca /= sclfac;
                    r *= sclfac;
                    ra *= sclfac;
                }

                if (c + r >= factor * s) continue;
                if (f < 1 && scale[scaleOff + i] < 1 && f * scale[scaleOff + i] <= sfmin1) continue;
                if (f > 1 && scale[scaleOff + i] > 1 && scale[scaleOff + i] >= sfmax1 / f) continue;

                g = 1.0 / f;
                scale[scaleOff + i] *= f;
                noConv = true;
                Zdscal.zdscal(n - k, g, A, aOff + i * lda * 2 + k * 2, 1);
                Zdscal.zdscal(l + 1, f, A, aOff + i * 2, lda);
            }
        }

        out[outOff] = k;
        out[outOff + 1] = l;
    }
}
