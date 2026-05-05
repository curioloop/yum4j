package com.curioloop.yum4j.linalg.blas.cmplx;
import com.curioloop.yum4j.linalg.blas.BLAS;
import com.curioloop.yum4j.linalg.blas.Ilaenv;
interface Zgehrd {

    static int zgehrd(int n, int ilo, int ihi, double[] A, int aOff, int lda,
                     double[] tau, int tOff, double[] work, int wOff, int lwork) {
        if (n == 0) {
            return 0;
        }

        int nb = Ilaenv.ilaenv(1, "ZGEHRD", "", n, ilo + 1, ihi + 1, -1);
        int lworkopt = Math.max(1, n * nb * 2);

        if (lwork == -1) {
            work[wOff] = lworkopt;
            return 0;
        }

        if (ilo < 0 || ilo > Math.max(0, n - 1)) {
            return -2;
        }
        if (ihi < Math.min(ilo, n - 1) || ihi > n - 1) {
            return -3;
        }
        if (lda < Math.max(1, n)) {
            return -5;
        }
        if (lwork < Math.max(1, n * 2)) {
            return -8;
        }

        for (int i = ilo; i <= ihi - 1; i++) {
            int alphaPos = (aOff + (i + 1) * lda + i) * 2;

            Zlarfg.zlarfg(ihi - i, A, alphaPos, A, aOff + Math.min(i + 2, n - 1) * lda + i, lda, tau, tOff + i * 2);

            double betaRe = A[alphaPos];
            double betaIm = A[alphaPos + 1];

            A[alphaPos] = 1.0;
            A[alphaPos + 1] = 0.0;

            Zlarf1f.zlarf1f(BLAS.Side.Right, ihi + 1, ihi - i, A, alphaPos, lda, tau, tOff + i * 2,
                           A, (aOff + i + 1) * 2, lda, work, wOff);

            tau[tOff + i * 2 + 1] = -tau[tOff + i * 2 + 1];
            Zlarf1f.zlarf1f(BLAS.Side.Left, ihi - i, n - i - 1, A, alphaPos, lda, tau, tOff + i * 2,
                           A, (aOff + (i + 1) * lda + i + 1) * 2, lda, work, wOff);
            tau[tOff + i * 2 + 1] = -tau[tOff + i * 2 + 1];

            A[alphaPos] = betaRe;
            A[alphaPos + 1] = betaIm;
        }

        work[wOff] = lworkopt;
        return 0;
    }
}
