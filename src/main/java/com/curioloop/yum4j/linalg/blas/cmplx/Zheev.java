package com.curioloop.yum4j.linalg.blas.cmplx;
import com.curioloop.yum4j.linalg.blas.BLAS;
import com.curioloop.yum4j.linalg.blas.Ilaenv;
interface Zheev {

    static int zheev(char jobz, char uplo, int n, double[] A, int lda,
                    double[] w, int wOff, double[] work, int workOff, int lwork) {
        boolean wantz = (jobz == 'V' || jobz == 'v');
        boolean upper = (uplo == 'U' || uplo == 'u');

        if (!wantz && jobz != 'N' && jobz != 'n') {
            return -1;
        }
        if (!upper && uplo != 'L' && uplo != 'l') {
            return -2;
        }
        if (n < 0) {
            return -3;
        }

        if (n == 0) {
            if (lwork >= 1) work[workOff] = 1;
            return 0;
        }

        if (lda < Math.max(1, n)) {
            return -5;
        }

        if (lwork == -1) {
            int nb = Ilaenv.ilaenv(1, "ZHETRD", String.valueOf(uplo), n, -1, -1, -1);
            if (nb < 1) nb = 1;
            int lwkopt = Math.max(1, (nb + 1) * n) * 2 + 3 * (n - 1) + 1;
            work[workOff] = lwkopt;
            return 0;
        }

        if (lwork < Math.max(1, 3 * n - 2)) {
            return -9;
        }

        if (n == 2) {
            return compute2x2(wantz, A, lda, w, wOff);
        }

        int tauSize = 2 * (n - 1);
        int eSize = n - 1;
        int workStart = tauSize + eSize;
        if (workStart % 2 != 0) workStart++;
        int llwork = lwork - workStart;
        int info = Zhetrd.zhetrd(upper ? BLAS.Uplo.Upper : BLAS.Uplo.Lower, n, A, 0, lda,
                w, wOff, work, workOff + tauSize, work, workOff, work, workOff + workStart, llwork);

        if (info != 0) return info;

        if (!wantz) {
            boolean ok = Zsterf.zsterf(n, w, wOff, work, workOff + tauSize);
            if (!ok) return 1;
        } else {
            info = Zorgtr.zorgtr(upper ? BLAS.Uplo.Upper : BLAS.Uplo.Lower, n, A, 0, lda,
                               work, workOff, work, workOff + workStart, llwork);
            if (info != 0) return info;

            info = Zsteqr.zsteqr('V', n, w, wOff, work, workOff + tauSize, A, 0, lda, work, workOff + workStart);
        }

        return info;
    }

    static int compute2x2(boolean wantz, double[] A, int lda, double[] w, int wOff) {
        double a = A[0];
        double d = A[2 * lda + 2];
        double b_re = A[2];
        double b_im = A[3];

        double b_mag_sq = b_re * b_re + b_im * b_im;
        double delta = (a - d) / 2;
        double mu = Math.sqrt(delta * delta + b_mag_sq);

        w[wOff] = (a + d) / 2 - mu;
        w[wOff + 1] = (a + d) / 2 + mu;

        if (wantz) {
            if (mu == 0) {
                A[0] = 1.0; A[1] = 0.0;
                A[2] = 0.0; A[3] = 0.0;
                A[2 * lda] = 0.0; A[2 * lda + 1] = 0.0;
                A[2 * lda + 2] = 1.0; A[2 * lda + 3] = 0.0;
            } else if (delta >= 0) {
                double norm = Math.sqrt(2 * mu * (mu + delta));
                double s = (mu + delta) / norm;
                A[0] = -b_re / norm; A[1] = -b_im / norm;
                A[2] = s; A[3] = 0.0;
                A[2 * lda] = s; A[2 * lda + 1] = 0.0;
                A[2 * lda + 2] = b_re / norm; A[2 * lda + 3] = -b_im / norm;
            } else {
                double norm = Math.sqrt(2 * mu * (mu - delta));
                double s = (mu - delta) / norm;
                A[0] = s; A[1] = 0.0;
                A[2] = b_re / norm; A[3] = b_im / norm;
                A[2 * lda] = -b_re / norm; A[2 * lda + 1] = b_im / norm;
                A[2 * lda + 2] = s; A[2 * lda + 3] = 0.0;
            }
        }

        return 0;
    }
}
