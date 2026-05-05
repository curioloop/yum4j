package com.curioloop.yum4j.linalg.blas.cmplx;

import com.curioloop.yum4j.linalg.blas.BLAS;

/**
 * LAPACK ZLASCL: Multiplies a complex matrix A by the real scalar CTO/CFROM.
 * 
 * Based on LAPACK.
 */
interface Zlascl {

    /**
     * ZLASCL multiplies a complex matrix A by the real scalar CTO/CFROM.
     * TYPE specifies that A is a general, upper triangular, lower triangular,
     * or symmetric matrix.
     *
     * @param type  'G' for general matrix, 'L' for lower triangular, 'U' for upper triangular, 'H' for symmetric/Hermitian upper, 'A' for symmetric/Hermitian lower
     * @param kl    lower bandwidth (for band matrices)
     * @param ku    upper bandwidth (for band matrices)
     * @param cfrom scalar factor
     * @param cto   scalar factor
     * @param m     number of rows of the matrix A
     * @param n     number of columns of the matrix A
     * @param A     complex matrix (m x n, row-major, interleaved format), overwritten on exit
     * @param aOff  offset into A
     * @param lda   leading dimension of A
     * @return 0 on success, non-zero on failure
     */
    static int zlascl(char type, int kl, int ku, double cfrom, double cto, 
                     int m, int n, double[] A, int aOff, int lda) {
        if (m == 0 || n == 0) {
            return 0;
        }

        if (cfrom == 0.0) {
            // Set all elements to zero
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    int pos = aOff + i * lda * 2 + j * 2;
                    A[pos] = 0.0;
                    A[pos + 1] = 0.0;
                }
            }
            return 0;
        }

        double ratio = cto / cfrom;
        boolean done = false;

        // Compute machine constants
        double safmin = BLAS.safmin();
        double bignum = 1.0 / safmin;
        double smlnum = BLAS.eps() * bignum;

        while (!done) {
            double cfrom1 = cfrom * smlnum;
            double cto1 = cto / bignum;

            if (Math.abs(cfrom1) > Math.abs(cto) && cto != 0.0) {
                // Scale by smlnum
                for (int i = 0; i < m; i++) {
                    for (int j = 0; j < n; j++) {
                        int pos = aOff + i * lda * 2 + j * 2;
                        A[pos] *= smlnum;
                        A[pos + 1] *= smlnum;
                    }
                }
                cfrom = cfrom1;
                done = false;
            } else if (Math.abs(cto1) > Math.abs(cfrom)) {
                // Scale by bignum
                for (int i = 0; i < m; i++) {
                    for (int j = 0; j < n; j++) {
                        int pos = aOff + i * lda * 2 + j * 2;
                        A[pos] *= bignum;
                        A[pos + 1] *= bignum;
                    }
                }
                cto = cto1;
                done = false;
            } else {
                // Scale by ratio
                for (int i = 0; i < m; i++) {
                    for (int j = 0; j < n; j++) {
                        int pos = aOff + i * lda * 2 + j * 2;
                        A[pos] *= ratio;
                        A[pos + 1] *= ratio;
                    }
                }
                done = true;
            }
        }

        return 0;
    }
}
