package com.curioloop.yum4j.linalg.blas.cmplx;

/**
 * LAPACK ZLANGE: Computes the value of the one norm, Frobenius norm, infinity norm,
 * or the largest absolute value of any element of a complex matrix.
 * 
 * Based on LAPACK.
 */
interface Zlange {

    /**
     * ZLANGE returns the value of the one norm, Frobenius norm, infinity norm,
     * or the largest absolute value of any element of a complex matrix A.
     *
     * @param norm  'M' or 'm' (maximum absolute value), '1' or 'O' (one norm),
     *              'I' or 'i' (infinity norm), 'F' or 'f' (Frobenius norm)
     * @param m     number of rows of the matrix A
     * @param n     number of columns of the matrix A
     * @param A     complex matrix (m x n, row-major, interleaved format)
     * @param lda   leading dimension of A
     * @param work  workspace for norm '1' or 'I' (length m or n)
     * @return      the specified norm of A
     */
    static double zlange(char norm, int m, int n, double[] A, int aOff, int lda, double[] work) {
        if (m == 0 || n == 0) {
            return 0.0;
        }

        double value = 0.0;

        switch (Character.toUpperCase(norm)) {
            case 'M':// Maximum absolute value
                for (int i = 0; i < m; i++) {
                    for (int j = 0; j < n; j++) {
                        int pos = aOff + i * lda * 2 + j * 2;
                        if (pos + 1 < A.length) {
                            double re = A[pos];
                            double im = A[pos + 1];
                            double absVal = Math.hypot(re, im);
                            if (absVal > value) {
                                value = absVal;
                            }
                        }
                    }
                }
                break;

            case '1':// One norm (maximum column sum)
                for (int j = 0; j < n; j++) {
                    double sum = 0.0;
                    for (int i = 0; i < m; i++) {
                        int pos = aOff + i * lda * 2 + j * 2;
                        if (pos + 1 < A.length) {
                            double re = A[pos];
                            double im = A[pos + 1];
                            sum += Math.hypot(re, im);
                        }
                    }
                    if (sum > value) {
                        value = sum;
                    }
                }
                break;

            case 'I':// Infinity norm (maximum row sum)
                for (int i = 0; i < m; i++) {
                    double sum = 0.0;
                    for (int j = 0; j < n; j++) {
                        int pos = aOff + i * lda * 2 + j * 2;
                        if (pos + 1 < A.length) {
                            double re = A[pos];
                            double im = A[pos + 1];
                            sum += Math.hypot(re, im);
                        }
                    }
                    if (sum > value) {
                        value = sum;
                    }
                }
                break;

            case 'F':// Frobenius norm
                double sumSquares = 0.0;
                for (int i = 0; i < m; i++) {
                    for (int j = 0; j < n; j++) {
                        int pos = aOff + i * lda * 2 + j * 2;
                        if (pos + 1 < A.length) {
                            double re = A[pos];
                            double im = A[pos + 1];
                            sumSquares += re * re + im * im;
                        }
                    }
                }
                value = Math.sqrt(sumSquares);
                break;

            default:
                throw new IllegalArgumentException("Invalid norm specifier: " + norm);
        }

        return value;
    }
}
