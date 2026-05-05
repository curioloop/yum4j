package com.curioloop.yum4j.linalg.blas.cmplx;

/**
 * ZCOPY copies a complex vector x to a complex vector y.
 * y[i] = x[i] for i = 0, 1, ..., n-1
 */
interface Zcopy {

    /**
     * Copies a complex vector x to a complex vector y.
     *
     * @param n      Number of elements in vector x and y
     * @param x      Complex vector x stored in interleaved format [re, im, re, im, ...]
     * @param xStart Starting index of x
     * @param incX   Increment for x
     * @param y      Complex vector y stored in interleaved format [re, im, re, im, ...]
     * @param yStart Starting index of y
     * @param incY   Increment for y
     */
    public static void zcopy(int n, double[] x, int xStart, int incX, double[] y, int yStart, int incY) {
        if (n <= 0) return;
        if (incX == 1 && incY == 1) {
            System.arraycopy(x, xStart, y, yStart, n * 2);
        } else {
            // Non-unit stride case
            for (int i = 0; i < n; i++) {
                int xPos = xStart + i * incX * 2;
                int yPos = yStart + i * incY * 2;
                y[yPos] = x[xPos];       // real part
                y[yPos + 1] = x[xPos + 1]; // imaginary part
            }
        }
    }

    /**
     * Copies a complex vector x to a complex vector y with unit increments.
     *
     * @param n      Number of elements in vector x and y
     * @param x      Complex vector x stored in interleaved format [re, im, re, im, ...]
     * @param y      Complex vector y stored in interleaved format [re, im, re, im, ...]
     */
    public static void zcopy(int n, double[] x, double[] y) {
        zcopy(n, x, 0, 1, y, 0, 1);
    }
}
