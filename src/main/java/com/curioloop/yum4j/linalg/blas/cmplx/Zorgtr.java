package com.curioloop.yum4j.linalg.blas.cmplx;
import com.curioloop.yum4j.linalg.blas.BLAS;
import com.curioloop.yum4j.linalg.blas.Ilaenv;
/**
 * LAPACK ZORGTR: Generates a complex unitary matrix Q from a triangular factor of the
 * orthogonal factorization determined by ZSYTRD or ZHETRD.
 * 
 * Based on LAPACK.
 */
interface Zorgtr {

    /**
     * ZORGTR generates a complex unitary matrix Q which is defined as the
     * product of n-1 elementary reflectors of order n, as returned by
     * ZSYTRD or ZHETRD: 
     * 
     *    Q = H(1) H(2) . . . H(n-1).
     *
     * @param uplo  'U' for upper triangle, 'L' for lower triangle
     * @param n     order of the matrix Q
     * @param A     complex matrix (n x n, row-major, interleaved format), overwritten on exit
     * @param lda   leading dimension of A
     * @param tau   scalar factors of the elementary reflectors (input, length n-1)
     * @param tOff  offset into tau
     * @param work  workspace (length lwork)
     * @param wOff  offset into work
     * @param lwork length of workspace (>= n-1, or -1 for query)
     * @return 0 on success, non-zero on failure
     */
    static int zorgtr(BLAS.Uplo uplo, int n, double[] A, int aOff, int lda, 
                     double[] tau, int tOff, double[] work, int wOff, int lwork) {
        if (n == 0) {
            work[wOff] = 1;
            return 0;
        }

        boolean upper = (uplo == BLAS.Uplo.Upper);
        int nb, lworkopt;

        if (upper) {
            nb = Ilaenv.ilaenv(1, "ZUNGQL", "", n-1, n-1, n-1, -1);
        } else {
            nb = Ilaenv.ilaenv(1, "ZUNGQR", "", n-1, n-1, n-1, -1);
        }
        lworkopt = Math.max(1, (n-1) * nb * 2); // Complex workspace

        // Query mode: return optimal size
        if (lwork == -1) {
            work[wOff] = lworkopt;
            return 0;
        }

        // Validate dimensions
        if (lda < Math.max(1, n)) {
            return -4;
        }
        if (lwork < Math.max(1, n-1)) {
            return -7;
        }

        if (upper) {
            // Q was determined by a call to ZHETRD with UPLO = 'U'
            // Shift the vectors which define the elementary reflectors one
            // column to the left, and set the last row and column of Q to
            // those of the unit matrix
            for (int j = 0; j < n - 1; j++) {
                for (int i = 0; i < j; i++) {
                    int pos1 = (aOff + i * lda + j) * 2;
                    int pos2 = (aOff + i * lda + j + 1) * 2;
                    A[pos1] = A[pos2];
                    A[pos1 + 1] = A[pos2 + 1];
                }
                int pos = (aOff + (n - 1) * lda + j) * 2;
                A[pos] = 0.0;
                A[pos + 1] = 0.0;
            }
            for (int i = 0; i < n - 1; i++) {
                int pos = (aOff + i * lda + n - 1) * 2;
                A[pos] = 0.0;
                A[pos + 1] = 0.0;
            }
            int pos = (aOff + (n - 1) * lda + n - 1) * 2;
            A[pos] = 1.0;
            A[pos + 1] = 0.0;

            // Generate Q(1:n-1,1:n-1)
            if (n > 1) {
                Zungql.zungql(n - 1, n - 1, n - 1, A, aOff, lda, tau, tOff, work, wOff, lwork);
            }
        } else {
            // Q was determined by a call to ZHETRD with UPLO = 'L'
            // Shift the vectors which define the elementary reflectors one
            // column to the right, and set the first row and column of Q to
            // those of the unit matrix
            for (int j = n - 1; j >= 1; j--) {
                int pos = (aOff + j) * 2;
                A[pos] = 0.0;
                A[pos + 1] = 0.0;
                for (int i = j + 1; i < n; i++) {
                    int pos1 = (aOff + i * lda + j) * 2;
                    int pos2 = (aOff + i * lda + j - 1) * 2;
                    A[pos1] = A[pos2];
                    A[pos1 + 1] = A[pos2 + 1];
                }
            }
            int pos = aOff * 2;
            A[pos] = 1.0;
            A[pos + 1] = 0.0;
            for (int i = 1; i < n; i++) {
                pos = (aOff + i * lda) * 2;
                A[pos] = 0.0;
                A[pos + 1] = 0.0;
            }

            // Generate Q(2:n,2:n)
            if (n > 1) {
                Zgeqr.zungqr(n - 1, n - 1, n - 1, A, aOff + lda + 1, lda, tau, tOff, work, wOff, lwork);
            }
        }

        work[wOff] = lworkopt;
        return 0;
    }
}