package com.curioloop.yum4j.linalg.blas.cmplx;
import com.curioloop.yum4j.linalg.blas.BLAS;
import com.curioloop.yum4j.linalg.blas.Ilaenv;
interface Zhetrd {

    static int zhetrd(BLAS.Uplo uplo, int n, double[] A, int aOff, int lda,
                      double[] d, int dOff, double[] e, int eOff,
                      double[] tau, int tOff, double[] work, int wOff, int lwork) {
        if (n == 0) return 0;

        int nb = Ilaenv.ilaenv(1, "ZHETRD", String.valueOf(uplo), n, 0, 0, -1);
        int lworkopt = Math.max(1, n * nb * 2);

        if (lwork == -1) {
            work[wOff] = lworkopt;
            return 0;
        }

        if (lda < Math.max(1, n)) return -5;
        if (lwork < Math.max(1, n * 2)) return -9;

        boolean upper = (uplo == BLAS.Uplo.Upper);
        double[] dotOut = new double[2];

        if (upper) {
            for (int i = n - 2; i >= 0; i--) {
                int alphaPos = (aOff + i * lda + i + 1) * 2;

                if (i > 0) {
                    Zlarfg.zlarfg(i + 1, A, alphaPos, A, aOff + i + 1, lda, tau, tOff + i * 2);
                } else {
                    Zlarfg.zlarfg(1, A, alphaPos, null, 0, 1, tau, tOff + i * 2);
                }

                double taur = tau[tOff + i * 2];
                double taui = tau[tOff + i * 2 + 1];
                e[eOff + i] = A[alphaPos];

                if (taur != 0.0 || taui != 0.0) {
                    A[alphaPos] = 1.0;
                    A[alphaPos + 1] = 0.0;

                    Zhemv.zhemv(BLAS.Uplo.Upper, i + 1, taur, taui,
                            A, aOff * 2, lda,
                            A, (aOff + i + 1) * 2, lda,
                            0.0, 0.0,
                            work, wOff, 1);

                        Zdot.zdotc(i + 1, work, wOff / 2, 1, A, aOff + i + 1, lda, dotOut, 0);
                    double alphaRe = -0.5 * (taur * dotOut[0] - taui * dotOut[1]);
                    double alphaIm = -0.5 * (taur * dotOut[1] + taui * dotOut[0]);

                    Zaxpy.zaxpy(i + 1, alphaRe, alphaIm,
                            A, aOff + i + 1, lda,
                            work, wOff / 2, 1);

                    Zher2.zher2(BLAS.Uplo.Upper, i + 1, -1.0, 0.0,
                            A, (aOff + i + 1) * 2, lda,
                            work, wOff, 1,
                            A, aOff * 2, lda);

                    A[alphaPos] = e[eOff + i];
                    A[alphaPos + 1] = 0.0;
                }

                d[dOff + i + 1] = A[(aOff + (i + 1) * lda + i + 1) * 2];
            }

            d[dOff] = A[aOff * 2];
        } else {
            for (int i = 0; i < n - 1; i++) {
                int m = n - i - 1;

                int alphaPos = (aOff + (i + 1) * lda + i) * 2;

                if (m > 1) {
                    Zlarfg.zlarfg(m, A, alphaPos, A, aOff + (i + 2) * lda + i, lda, tau, tOff + i * 2);
                } else {
                    Zlarfg.zlarfg(1, A, alphaPos, null, 0, 1, tau, tOff + i * 2);
                }

                double taur = tau[tOff + i * 2];
                double taui = tau[tOff + i * 2 + 1];
                e[eOff + i] = A[alphaPos];

                if (taur != 0.0 || taui != 0.0) {
                    A[alphaPos] = 1.0;
                    A[alphaPos + 1] = 0.0;

                        int subStart = (aOff + (i + 1) * lda + i + 1) * 2;

                    Zhemv.zhemv(BLAS.Uplo.Lower, m, taur, taui,
                            A, subStart, lda,
                            A, (aOff + (i + 1) * lda + i) * 2, lda,
                            0.0, 0.0,
                            work, wOff, 1);

                        Zdot.zdotc(m, work, wOff / 2, 1, A, aOff + (i + 1) * lda + i, lda, dotOut, 0);
                    double alphaRe = -0.5 * (taur * dotOut[0] - taui * dotOut[1]);
                    double alphaIm = -0.5 * (taur * dotOut[1] + taui * dotOut[0]);

                    Zaxpy.zaxpy(m, alphaRe, alphaIm,
                            A, aOff + (i + 1) * lda + i, lda,
                            work, wOff / 2, 1);

                    Zher2.zher2(BLAS.Uplo.Lower, m, -1.0, 0.0,
                            A, (aOff + (i + 1) * lda + i) * 2, lda,
                            work, wOff, 1,
                            A, subStart, lda);

                    A[alphaPos] = e[eOff + i];
                    A[alphaPos + 1] = 0.0;
                }

                d[dOff + i] = A[(aOff + i * lda + i) * 2];
            }

            d[dOff + n - 1] = A[(aOff + (n - 1) * lda + (n - 1)) * 2];
        }

        work[wOff] = lworkopt;
        return 0;
    }
}
