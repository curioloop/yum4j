package com.curioloop.yum4j.linalg.blas.cmplx;
import com.curioloop.yum4j.linalg.blas.BLAS;
/**
 * LAPACK ZLANSY: Computes the value of the one norm, or the Frobenius norm,
 * or the infinity norm, or the element of largest absolute value of a complex
 * symmetric matrix.
 * 
 * Based on LAPACK.
 */
interface Zlansy {

    /**
     * ZLANSY returns the value of the one norm, or the Frobenius norm,
     * or the infinity norm, or the element of largest absolute value of a complex
     * symmetric matrix A.
     *
     * @param norm  'M' or 'm' (maximum absolute value), '1' or 'O' (one norm),
     *              'I' or 'i' (infinity norm), 'F' or 'f' (Frobenius norm)
     * @param uplo  'U' for upper triangle, 'L' for lower triangle
     * @param n     order of the matrix A
     * @param A     complex symmetric matrix (n x n, row-major, interleaved format)
     * @param lda   leading dimension of A
     * @param work  workspace for norm '1' or 'I' (length n)
     * @return      the specified norm of A
     */
    static double zlansy(char norm, BLAS.Uplo uplo, int n, double[] A, int lda, double[] work) {
        if (n == 0) {
            return 0.0;
        }

        double value = 0.0;
        boolean upper = (uplo == BLAS.Uplo.Upper);

        switch (Character.toUpperCase(norm)) {
            case 'M':// Maximum absolute value
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < n; j++) {
                        if ((upper && j >= i) || (!upper && j <= i)) {
                            double re = A[i * lda * 2 + j * 2];
                            double im = A[i * lda * 2 + j * 2 + 1];
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
                    for (int i = 0; i < n; i++) {
                        if ((upper && j >= i) || (!upper && j <= i)) {
                            double re = A[i * lda * 2 + j * 2];
                            double im = A[i * lda * 2 + j * 2 + 1];
                            sum += Math.hypot(re, im);
                        }
                    }
                    if (sum > value) {
                        value = sum;
                    }
                }
                break;

            case 'I':// Infinity norm (maximum row sum)
                for (int i = 0; i < n; i++) {
                    double sum = 0.0;
                    for (int j = 0; j < n; j++) {
                        if ((upper && j >= i) || (!upper && j <= i)) {
                            double re = A[i * lda * 2 + j * 2];
                            double im = A[i * lda * 2 + j * 2 + 1];
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
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < n; j++) {
                        if ((upper && j >= i) || (!upper && j <= i)) {
                            double re = A[i * lda * 2 + j * 2];
                            double im = A[i * lda * 2 + j * 2 + 1];
                            if (i == j) {
                                sumSquares += re * re + im * im;
                            } else {
                                sumSquares += 2 * (re * re + im * im);
                            }
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
