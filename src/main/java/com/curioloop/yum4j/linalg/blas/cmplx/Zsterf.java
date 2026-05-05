package com.curioloop.yum4j.linalg.blas.cmplx;

import com.curioloop.yum4j.linalg.blas.BLAS;

/**
 * LAPACK ZSTERF: Computes all eigenvalues of a real symmetric tridiagonal matrix
 * using the Pal-Walker-Kahan variant of the QL or QR algorithm.
 *
    * <p>This is the complex-precision wrapper that delegates to the real BLAS facade,
 * since the tridiagonal matrix produced by ZHETRD/ZSYTRD has real diagonal
 * and real off-diagonal elements.</p>
 */
interface Zsterf {

    /**
     * ZSTERF computes all eigenvalues of a real symmetric tridiagonal matrix
     * using the Pal-Walker-Kahan variant of the QL or QR algorithm.
     *
     * @param n     order of the matrix
     * @param d     diagonal elements of the matrix (input/output, length n)
     * @param dOff  offset into d
     * @param e     off-diagonal elements of the matrix (input/output, length n-1)
     * @param eOff  offset into e
     * @return true on success, false on failure
     */
    static boolean zsterf(int n, double[] d, int dOff, double[] e, int eOff) {
        return BLAS.dsterf(n, d, dOff, e, eOff);
    }
}
