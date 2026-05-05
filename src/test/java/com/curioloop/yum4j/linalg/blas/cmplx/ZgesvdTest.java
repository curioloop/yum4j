/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.linalg.blas.cmplx;

import com.curioloop.yum4j.linalg.blas.BLAS;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ZgesvdTest {

    private static final double TOL = 1e-10;

    static final double[] GESVD_A_SQ = {
        4.967141530112327e-01, -1.012831120334424e+00, -1.382643011711847e-01, 3.142473325952739e-01,
        6.476885381006925e-01, -9.080240755212111e-01, 1.523029856408025e+00, -1.412303701335292e+00,
        -2.341533747233360e-01, 1.465648768921554e+00, -2.341369569491805e-01, -2.257763004865357e-01,
        1.579212815507391e+00, 6.752820468792384e-02, 7.674347291529088e-01, -1.424748186213457e+00,
        -4.694743859349521e-01, -5.443827245251827e-01, 5.425600435859647e-01, 1.109225897098661e-01,
        -4.634176928124623e-01, -1.150993577422303e+00, -4.657297535702569e-01, 3.756980183456720e-01,
        2.419622715660341e-01, -6.006386899188050e-01, -1.913280244657798e+00, -2.916937497932768e-01,
        -1.724917832513033e+00, -6.017066122293969e-01, -5.622875292409727e-01, 1.852278184508938e+00
    };

    static final double[] GESVD_S_SQ = {
        4.293222506007609e+00, 2.561566085693732e+00, 1.599728945163163e+00, 8.903850115002069e-01
    };

    static final double[] GESVD_A_MN = {
        -1.349722473793391e-02, -8.392175232226385e-01, -1.057710928955900e+00, -3.092123758512146e-01,
        8.225449121031890e-01, 3.312634314035640e-01, -1.220843649971022e+00, 9.755451271223592e-01,
        2.088635950047554e-01, -4.791742378452900e-01, -1.959670123879776e+00, -1.856589766638171e-01,
        -1.328186048898431e+00, -1.106334974006028e+00, 1.968612358691235e-01, -1.196206624080671e+00,
        7.384665799954104e-01, 8.125258223941980e-01, 1.713682811899705e-01, 1.356240028570823e+00,
        -1.156482823882405e-01, -7.201012158033385e-02, -3.011036955892888e-01, 1.003532897892024e+00,
        -1.478521990367427e+00, 3.616360250476341e-01, -7.198442083947086e-01, -6.451197546051243e-01,
        -4.606387709597875e-01, 3.613956055084139e-01, 1.057122226218916e+00, 1.538036566465969e+00,
        3.436182895684614e-01, -3.582603910995153e-02, -1.763040155362734e+00, 1.564643655814006e+00,
        3.240839693947950e-01, -2.619745104089744e+00, -3.850822804163165e-01, 8.219025043752238e-01,
        -6.769220003059587e-01, 8.704706823817134e-02, 6.116762888408679e-01, -2.990073504658678e-01,
        1.030999522495951e+00, 9.176077653550230e-02, 9.312801191161986e-01, -1.987568914600893e+00
    };

    static final double[] GESVD_S_MN = {
        4.621784017764585e+00, 3.594026191505454e+00, 2.775372398262553e+00, 1.823776357464790e+00
    };

    static final double[] GESVD_A_NM = {
        -2.196718878375119e-01, -1.612857116660091e-01, 3.571125715117464e-01, 4.040508568145384e-01,
        1.477894044741516e+00, 1.886185901210530e+00, -5.182702182736474e-01, 1.745778128318390e-01,
        -8.084936028931876e-01, 2.575503907227640e-01, -5.017570435845365e-01, -7.444591576616710e-02,
        9.154021177020741e-01, -1.918771215299041e+00, 3.287511096596845e-01, -2.651387544921688e-02,
        -5.297602037670388e-01, 6.023020994102644e-02, 5.132674331133561e-01, 2.463242112485286e+00,
        9.707754934804039e-02, -1.923609647811225e-01, 9.686449905328892e-01, 3.015473423336125e-01,
        -7.020530938773524e-01, -3.471176970524331e-02, -3.276621465977682e-01, -1.168678037619532e+00,
        -3.921081531321576e-01, 1.142822814515021e+00, -1.463514948132119e+00, 7.519330326867741e-01,
        2.961202770645761e-01, 7.910319470430469e-01, 2.610552721798893e-01, -9.093874547947389e-01,
        5.113456642460900e-03, 1.402794310936099e+00, -2.345871333751474e-01, -1.401851062792281e+00,
        -1.415370742050414e+00, 5.868570938002703e-01, -4.206453227653590e-01, 2.190455625809979e+00,
        -3.427145165267695e-01, -9.905363251306883e-01, -8.022772692216189e-01, -5.662977296027719e-01
    };

    static final double[] GESVD_S_NM = {
        4.363085822067004e+00, 3.402453008751189e+00, 2.816641727501545e+00, 1.507510313248954e+00
    };

    @Test
    void testBasicSquareMatrix() {
        double[] A = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
        double[] Aorig = A.clone();
        double[] s = new double[2];
        double[] u = new double[2 * 2 * 2];
        double[] vt = new double[2 * 2 * 2];
        double[] work = new double[256];
        int lwork = work.length;

        int info = Zgesvd.zgesvd('A', 'A', 2, 2, A, 0, 2, s, 0, u, 0, 2, vt, 0, 2, work, 0, lwork);
        assertEquals(0, info);
        assertTrue(s[0] >= s[1]);
        assertTrue(s[0] > 0 && s[1] > 0);
        checkUnitary(u, 2);
        checkUnitary(vt, 2);
        checkSVD(Aorig, u, s, vt, 2, 2);
    }

    @Test
    void testOnlySingularValues() {
        double[] A = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
        double[] s = new double[2];
        double[] work = new double[256];
        int lwork = work.length;

        int info = Zgesvd.zgesvd('N', 'N', 2, 2, A, 0, 2, s, 0, null, 0, 1, null, 0, 1, work, 0, lwork);
        assertEquals(0, info);
        assertTrue(s[0] >= s[1]);
        assertTrue(s[0] > 0 && s[1] > 0);
    }

    @Test
    void testRectangularMatrixMoreRows() {
        double[] A = {
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0
        };
        double[] Aorig = A.clone();
        double[] s = new double[2];
        double[] u = new double[2 * 2 * 2];
        double[] vt = new double[3 * 3 * 2];
        double[] work = new double[256];
        int lwork = work.length;

        int info = Zgesvd.zgesvd('A', 'A', 2, 3, A, 0, 3, s, 0, u, 0, 2, vt, 0, 3, work, 0, lwork);
        assertEquals(0, info);
        assertTrue(s[0] >= s[1]);
        assertTrue(s[0] > 0 && s[1] > 0);
        checkUnitary(u, 2);
        checkUnitary(vt, 3);
        checkSVD(Aorig, u, s, vt, 2, 3);
    }

    @Test
    void testRectangularMatrixMoreColumns() {
        double[] A = {
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0
        };
        double[] Aorig = A.clone();
        double[] s = new double[2];
        double[] u = new double[3 * 3 * 2];
        double[] vt = new double[2 * 2 * 2];
        double[] work = new double[256];
        int lwork = work.length;

        int info = Zgesvd.zgesvd('A', 'A', 3, 2, A, 0, 2, s, 0, u, 0, 3, vt, 0, 2, work, 0, lwork);
        assertEquals(0, info);
        assertTrue(s[0] >= s[1]);
        assertTrue(s[0] > 0 && s[1] > 0);
        checkUnitary(u, 3);
        checkUnitary(vt, 2);
        checkSVD(Aorig, u, s, vt, 3, 2);
    }

    @Test
    void testWorkspaceQuery() {
        double[] A = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
        double[] s = new double[2];
        double[] u = new double[2 * 2 * 2];
        double[] vt = new double[2 * 2 * 2];
        double[] work = new double[1];

        int info = Zgesvd.zgesvd('A', 'A', 2, 2, A, 0, 2, s, 0, u, 0, 2, vt, 0, 2, work, 0, -1);
        assertEquals(0, info);
        assertTrue(work[0] > 0);

        int lwork = (int) Math.ceil(work[0]);
        work = new double[Math.max(1, 2 * lwork)];
        A = new double[]{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
        info = Zgesvd.zgesvd('A', 'A', 2, 2, A, 0, 2, s, 0, u, 0, 2, vt, 0, 2, work, 0, lwork);
        assertEquals(0, info);
    }

    @Test
    void testZeroMatrix() {
        double[] A = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        double[] s = new double[2];
        double[] u = new double[2 * 2 * 2];
        double[] vt = new double[2 * 2 * 2];
        double[] work = new double[256];
        int lwork = work.length;

        int info = Zgesvd.zgesvd('A', 'A', 2, 2, A, 0, 2, s, 0, u, 0, 2, vt, 0, 2, work, 0, lwork);
        assertEquals(0, info);
        assertTrue(s[0] < TOL);
        assertTrue(s[1] < TOL);
        checkUnitary(u, 2);
        checkUnitary(vt, 2);
    }

    @Test
    void testIdentityMatrix() {
        double[] A = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0};
        double[] Aorig = A.clone();
        double[] s = new double[2];
        double[] u = new double[2 * 2 * 2];
        double[] vt = new double[2 * 2 * 2];
        double[] work = new double[256];
        int lwork = work.length;

        int info = Zgesvd.zgesvd('A', 'A', 2, 2, A, 0, 2, s, 0, u, 0, 2, vt, 0, 2, work, 0, lwork);
        assertEquals(0, info);
        assertEquals(1.0, s[0], TOL);
        assertEquals(1.0, s[1], TOL);
        checkUnitary(u, 2);
        checkUnitary(vt, 2);
    }

    @Test
    void testEmpty() {
        double[] A = new double[0];
        double[] s = new double[0];
        double[] work = new double[1];

        int info = Zgesvd.zgesvd('N', 'N', 0, 0, A, 0, 1, s, 0, null, 0, 1, null, 0, 1, work, 0, 0);
        assertEquals(0, info);
    }

    @Test
    void testInvalidInput() {
        double[] A = {1.0, 2.0, 3.0, 4.0};
        double[] s = new double[2];
        double[] work = new double[256];
        int lwork = work.length;

        int info = Zgesvd.zgesvd('X', 'N', 2, 2, A, 0, 2, s, 0, null, 0, 2, null, 0, 2, work, 0, lwork);
        assertEquals(-1, info);

        info = Zgesvd.zgesvd('N', 'X', 2, 2, A, 0, 2, s, 0, null, 0, 2, null, 0, 2, work, 0, lwork);
        assertEquals(-2, info);

        info = Zgesvd.zgesvd('N', 'N', -1, 2, A, 0, 2, s, 0, null, 0, 2, null, 0, 2, work, 0, lwork);
        assertEquals(-3, info);

        info = Zgesvd.zgesvd('N', 'N', 2, -1, A, 0, 2, s, 0, null, 0, 2, null, 0, 2, work, 0, lwork);
        assertEquals(-4, info);

        info = Zgesvd.zgesvd('N', 'N', 2, 2, A, 0, 1, s, 0, null, 0, 2, null, 0, 2, work, 0, lwork);
        assertEquals(-6, info);

        info = Zgesvd.zgesvd('N', 'N', 2, 2, A, 0, 2, s, 0, null, 0, 2, null, 0, 2, work, 0, -2);
        assertEquals(-13, info);
    }

    @Test
    void testWithSmallMatrix() {
        double[] A = {1.0, 2.0};
        double[] Aorig = A.clone();
        double[] s = new double[1];
        double[] u = new double[1 * 1 * 2];
        double[] vt = new double[1 * 1 * 2];
        double[] work = new double[256];
        int lwork = work.length;

        int info = Zgesvd.zgesvd('A', 'A', 1, 1, A, 0, 1, s, 0, u, 0, 1, vt, 0, 1, work, 0, lwork);
        assertEquals(0, info);
        assertEquals(Math.sqrt(5.0), s[0], TOL);
    }

    @Test
    void testWithVector() {
        double[] A = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        double[] Aorig = A.clone();
        double[] s = new double[1];
        double[] u = new double[1 * 1 * 2];
        double[] vt = new double[3 * 3 * 2];
        double[] work = new double[256];
        int lwork = work.length;

        int info = Zgesvd.zgesvd('A', 'A', 1, 3, A, 0, 3, s, 0, u, 0, 1, vt, 0, 3, work, 0, lwork);
        assertEquals(0, info);

        double expectedNorm = Math.sqrt(1.0*1.0 + 2.0*2.0 + 3.0*3.0 + 4.0*4.0 + 5.0*5.0 + 6.0*6.0);
        assertEquals(expectedNorm, s[0], TOL);
        checkUnitary(u, 1);
        checkUnitary(vt, 3);
    }

    @Test
    void testSquareSVDScipy() {
        int m = 4, n = 4;
        double[] A = GESVD_A_SQ.clone();
        double[] Aorig = GESVD_A_SQ.clone();
        double[] s = new double[n];
        double[] u = new double[m * m * 2];
        double[] vt = new double[n * n * 2];
        double[] work = new double[4096];
        int lwork = work.length;

        int info = Zgesvd.zgesvd('A', 'A', m, n, A, 0, n, s, 0, u, 0, m, vt, 0, n, work, 0, lwork);
        assertEquals(0, info);

        for (int i = 0; i < GESVD_S_SQ.length; i++) {
            assertEquals(GESVD_S_SQ[i], s[i], 1e-10);
        }

        verifyUnitaryUHxU(u, m, 1e-9);
        verifyUnitaryVTHxVT(vt, n, 1e-9);
        verifyReconstruction(Aorig, u, s, vt, m, n, 1e-9);
    }

    @Test
    void testRectangularMgtNScipy() {
        int m = 6, n = 4;
        double[] A = GESVD_A_MN.clone();
        double[] Aorig = GESVD_A_MN.clone();
        double[] s = new double[n];
        double[] u = new double[m * m * 2];
        double[] vt = new double[n * n * 2];
        double[] work = new double[4096];
        int lwork = work.length;

        int info = Zgesvd.zgesvd('A', 'A', m, n, A, 0, n, s, 0, u, 0, m, vt, 0, n, work, 0, lwork);
        assertEquals(0, info);

        for (int i = 0; i < GESVD_S_MN.length; i++) {
            assertEquals(GESVD_S_MN[i], s[i], 1e-10);
        }

        verifyReconstruction(Aorig, u, s, vt, m, n, 1e-9);
        verifyUnitaryUHxU(u, m, 1e-9);
        verifyUnitaryVTxVTH(vt, n, 1e-9);
    }

    @Test
    void testRectangularNgtMScipy() {
        int m = 4, n = 6;
        double[] A = GESVD_A_NM.clone();
        double[] Aorig = GESVD_A_NM.clone();
        double[] s = new double[m];
        double[] u = new double[m * m * 2];
        double[] vt = new double[n * n * 2];
        double[] work = new double[4096];
        int lwork = work.length;

        int info = Zgesvd.zgesvd('A', 'A', m, n, A, 0, n, s, 0, u, 0, m, vt, 0, n, work, 0, lwork);
        assertEquals(0, info);

        for (int i = 0; i < GESVD_S_NM.length; i++) {
            assertEquals(GESVD_S_NM[i], s[i], 1e-10);
        }

        verifyReconstruction(Aorig, u, s, vt, m, n, 1e-9);
        verifyUnitaryUHxU(u, m, 1e-9);
        verifyUnitaryVTxVTH(vt, n, 1e-9);
    }

    @Test
    void testZorglqOrthogonality() {
        int m = 3, n = 3;
        double[] A = GESVD_A_SQ.clone();
        double[] tau = new double[m * 2];
        double[] work = new double[4096];

        int info = Zgelq.zgelqf(m, n, A, 0, n, tau, 0, work, 0, 4096);
        assertEquals(0, info);

        info = Zgelq.zorglq(m, n, m, A, 0, n, tau, 0, work, 0, 4096);
        assertEquals(0, info);

        double[] qqh = new double[m * n * 2];
        Zgemm.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.Conj, m, m, n, 1.0, 0.0, A, 0, n, A, 0, n, 0.0, 0.0, qqh, 0, m);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
                int idx = (i * m + j) * 2;
                if (i == j) {
                    assertEquals(1.0, qqh[idx], TOL);
                    assertEquals(0.0, qqh[idx + 1], TOL);
                } else {
                    assertEquals(0.0, qqh[idx], TOL);
                    assertEquals(0.0, qqh[idx + 1], TOL);
                }
            }
        }
    }

    @Test
    void testZorgbrPWithGebrd() {
        int m = 4, n = 4;
        int lda = n;

        double[] A = GESVD_A_SQ.clone();
        double[] d = new double[n];
        double[] e = new double[n];
        double[] tauq = new double[n * 2];
        double[] taup = new double[n * 2];
        double[] work = new double[4096];

        Zgebrd.zgebrd(m, n, A, 0, lda, d, 0, e, 0, tauq, 0, taup, 0, work, 0, 4096);

        double[] vt = new double[lda * n * 2];
        Zlacpy.zlacpy('U', n, n, A, 0, lda, vt, 0, lda);

        int info = Zorgbr.zorgbr('P', n, n, n, vt, 0, lda, taup, 0, work, 0, 4096);
        assertEquals(0, info);

        double[] vthvt = new double[n * n * 2];
        Zgemm.zgemm(BLAS.Trans.Conj, BLAS.Trans.NoTrans, n, n, n, 1.0, 0.0, vt, 0, lda, vt, 0, lda, 0.0, 0.0, vthvt, 0, n);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                int idx = (i * n + j) * 2;
                if (i == j) {
                    assertEquals(1.0, vthvt[idx], TOL);
                    assertEquals(0.0, vthvt[idx + 1], TOL);
                } else {
                    assertEquals(0.0, vthvt[idx], TOL);
                    assertEquals(0.0, vthvt[idx + 1], TOL);
                }
            }
        }

        double[] vtvth = new double[n * n * 2];
        Zgemm.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.Conj, n, n, n, 1.0, 0.0, vt, 0, lda, vt, 0, lda, 0.0, 0.0, vtvth, 0, n);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                int idx = (i * n + j) * 2;
                if (i == j) {
                    assertEquals(1.0, vtvth[idx], TOL);
                    assertEquals(0.0, vtvth[idx + 1], TOL);
                } else {
                    assertEquals(0.0, vtvth[idx], TOL);
                    assertEquals(0.0, vtvth[idx + 1], TOL);
                }
            }
        }
    }

    private void checkUnitary(double[] matrix, int n) {
        double[] product = new double[n * n * 2];
        Zgemm.zgemm(BLAS.Trans.Conj, BLAS.Trans.NoTrans, n, n, n, 1.0, 0.0,
                matrix, 0, n, matrix, 0, n, 0.0, 0.0, product, 0, n);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                int index = i * n * 2 + j * 2;
                if (i == j) {
                    assertEquals(1.0, product[index], TOL);
                    assertEquals(0.0, product[index + 1], TOL);
                } else {
                    assertEquals(0.0, product[index], TOL);
                    assertEquals(0.0, product[index + 1], TOL);
                }
            }
        }
    }

    private void checkSVD(double[] Aorig, double[] U, double[] S, double[] VT, int m, int n) {
        int minMN = Math.min(m, n);
        double[] S_matrix = new double[m * n * 2];
        for (int i = 0; i < minMN; i++) {
            S_matrix[i * n * 2 + i * 2] = S[i];
        }
        double[] US = new double[m * n * 2];
        Zgemm.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, m, n, minMN, 1.0, 0.0,
                U, 0, m, S_matrix, 0, n, 0.0, 0.0, US, 0, n);
        double[] USVt = new double[m * n * 2];
        Zgemm.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, m, n, n, 1.0, 0.0,
                US, 0, n, VT, 0, n, 0.0, 0.0, USVt, 0, n);
        for (int i = 0; i < m * n * 2; i++) {
            assertEquals(Aorig[i], USVt[i], 1e-8);
        }
    }

    private void verifyReconstruction(double[] Aorig, double[] U, double[] S, double[] VT,
                                      int m, int n, double tol) {
        int minMN = Math.min(m, n);
        double[] sigma = new double[m * n * 2];
        for (int i = 0; i < minMN; i++) {
            sigma[i * n * 2 + i * 2] = S[i];
        }
        double[] US = new double[m * n * 2];
        Zgemm.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, m, n, minMN, 1.0, 0.0,
                U, 0, m, sigma, 0, n, 0.0, 0.0, US, 0, n);
        double[] USVT = new double[m * n * 2];
        Zgemm.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, m, n, n, 1.0, 0.0,
                US, 0, n, VT, 0, n, 0.0, 0.0, USVT, 0, n);
        for (int i = 0; i < m * n * 2; i++) {
            assertEquals(Aorig[i], USVT[i], tol);
        }
    }

    private void verifyUnitaryUHxU(double[] U, int m, double tol) {
        double[] UHU = new double[m * m * 2];
        Zgemm.zgemm(BLAS.Trans.Conj, BLAS.Trans.NoTrans, m, m, m, 1.0, 0.0,
                U, 0, m, U, 0, m, 0.0, 0.0, UHU, 0, m);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
                int idx = (i * m + j) * 2;
                if (i == j) {
                    assertEquals(1.0, UHU[idx], tol);
                    assertEquals(0.0, UHU[idx + 1], tol);
                } else {
                    assertEquals(0.0, UHU[idx], tol);
                    assertEquals(0.0, UHU[idx + 1], tol);
                }
            }
        }
    }

    private void verifyUnitaryVTHxVT(double[] VT, int n, double tol) {
        double[] VTHV = new double[n * n * 2];
        Zgemm.zgemm(BLAS.Trans.Conj, BLAS.Trans.NoTrans, n, n, n, 1.0, 0.0,
                VT, 0, n, VT, 0, n, 0.0, 0.0, VTHV, 0, n);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                int idx = (i * n + j) * 2;
                if (i == j) {
                    assertEquals(1.0, VTHV[idx], tol);
                    assertEquals(0.0, VTHV[idx + 1], tol);
                } else {
                    assertEquals(0.0, VTHV[idx], tol);
                    assertEquals(0.0, VTHV[idx + 1], tol);
                }
            }
        }
    }

    private void verifyUnitaryVTxVTH(double[] VT, int n, double tol) {
        double[] VTVH = new double[n * n * 2];
        Zgemm.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.Conj, n, n, n, 1.0, 0.0,
                VT, 0, n, VT, 0, n, 0.0, 0.0, VTVH, 0, n);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                int idx = (i * n + j) * 2;
                if (i == j) {
                    assertEquals(1.0, VTVH[idx], tol);
                    assertEquals(0.0, VTVH[idx + 1], tol);
                } else {
                    assertEquals(0.0, VTVH[idx], tol);
                    assertEquals(0.0, VTVH[idx + 1], tol);
                }
            }
        }
    }

    private void verifyEconomyUnitaryUHxU(double[] U, int m, int k, double tol) {
        double[] UHU = new double[k * k * 2];
        Zgemm.zgemm(BLAS.Trans.Conj, BLAS.Trans.NoTrans, k, k, m, 1.0, 0.0,
                U, 0, k, U, 0, k, 0.0, 0.0, UHU, 0, k);
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < k; j++) {
                int idx = (i * k + j) * 2;
                if (i == j) {
                    assertEquals(1.0, UHU[idx], tol);
                    assertEquals(0.0, UHU[idx + 1], tol);
                } else {
                    assertEquals(0.0, UHU[idx], tol);
                    assertEquals(0.0, UHU[idx + 1], tol);
                }
            }
        }
    }

    private void verifyEconomyUnitaryVTxVTH(double[] VT, int k, int n, double tol) {
        double[] VTVH = new double[k * k * 2];
        Zgemm.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.Conj, k, k, n, 1.0, 0.0,
                VT, 0, n, VT, 0, n, 0.0, 0.0, VTVH, 0, k);
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < k; j++) {
                int idx = (i * k + j) * 2;
                if (i == j) {
                    assertEquals(1.0, VTVH[idx], tol);
                    assertEquals(0.0, VTVH[idx + 1], tol);
                } else {
                    assertEquals(0.0, VTVH[idx], tol);
                    assertEquals(0.0, VTVH[idx + 1], tol);
                }
            }
        }
    }

    private void verifyEconomyReconstruction(double[] Aorig, double[] U, double[] S, double[] VT,
                                             int m, int n, double tol) {
        int k = Math.min(m, n);
        double[] sigma = new double[k * k * 2];
        for (int i = 0; i < k; i++) {
            sigma[i * k * 2 + i * 2] = S[i];
        }
        double[] US = new double[m * k * 2];
        Zgemm.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, m, k, k, 1.0, 0.0,
                U, 0, k, sigma, 0, k, 0.0, 0.0, US, 0, k);
        double[] USVT = new double[m * n * 2];
        Zgemm.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, m, n, k, 1.0, 0.0,
                US, 0, k, VT, 0, n, 0.0, 0.0, USVT, 0, n);
        for (int i = 0; i < m * n * 2; i++) {
            assertEquals(Aorig[i], USVT[i], tol);
        }
    }

    @Test
    void testSquareSVDEconomyScipy() {
        int m = 4, n = 4;
        double[] A = GESVD_A_SQ.clone();
        double[] Aorig = GESVD_A_SQ.clone();
        double[] s = new double[4];
        double[] u = new double[4 * 4 * 2];
        double[] vt = new double[4 * 4 * 2];
        double[] work = new double[4096];

        int info = Zgesvd.zgesvd('S', 'S', 4, 4, A, 0, 4, s, 0, u, 0, 4, vt, 0, 4, work, 0, 4096);
        assertEquals(0, info);

        for (int i = 0; i < GESVD_S_SQ.length; i++) {
            assertEquals(GESVD_S_SQ[i], s[i], 1e-10);
        }

        verifyUnitaryUHxU(u, m, 1e-9);
        verifyUnitaryVTxVTH(vt, n, 1e-9);
        verifyReconstruction(Aorig, u, s, vt, m, n, 1e-9);
    }

    @Test
    void testRectangularSVDEconomyScipy() {
        {
            int m = 6, n = 4, k = Math.min(m, n);
            double[] A = GESVD_A_MN.clone();
            double[] Aorig = GESVD_A_MN.clone();
            double[] s = new double[k];
            double[] u = new double[m * k * 2];
            double[] vt = new double[k * n * 2];
            double[] work = new double[4096];

            int info = Zgesvd.zgesvd('S', 'S', m, n, A, 0, n, s, 0, u, 0, k, vt, 0, n, work, 0, 4096);
            assertEquals(0, info);

            for (int i = 0; i < GESVD_S_MN.length; i++) {
                assertEquals(GESVD_S_MN[i], s[i], 1e-10);
            }

            verifyEconomyUnitaryUHxU(u, m, k, 1e-9);
            verifyEconomyUnitaryVTxVTH(vt, k, n, 1e-9);
            verifyEconomyReconstruction(Aorig, u, s, vt, m, n, 1e-9);
        }
        {
            int m = 4, n = 6, k = Math.min(m, n);
            double[] A = GESVD_A_NM.clone();
            double[] Aorig = GESVD_A_NM.clone();
            double[] s = new double[k];
            double[] u = new double[m * k * 2];
            double[] vt = new double[k * n * 2];
            double[] work = new double[4096];

            int info = Zgesvd.zgesvd('S', 'S', m, n, A, 0, n, s, 0, u, 0, k, vt, 0, n, work, 0, 4096);
            assertEquals(0, info);

            for (int i = 0; i < GESVD_S_NM.length; i++) {
                assertEquals(GESVD_S_NM[i], s[i], 1e-10);
            }

            verifyEconomyUnitaryUHxU(u, m, k, 1e-9);
            verifyEconomyUnitaryVTxVTH(vt, k, n, 1e-9);
            verifyEconomyReconstruction(Aorig, u, s, vt, m, n, 1e-9);
        }
    }

    @Test
    void testFullWorkspaceQueryProvidesExecutableSizeForTallMatrix() {
        int m = 48;
        int n = 32;
        int k = Math.min(m, n);
        double[] a = randomComplexMatrix(m, n, 2026042501L);
        double[] s = new double[k];
        double[] u = new double[m * m * 2];
        double[] vt = new double[n * n * 2];
        double[] query = new double[1];

        int info = Zgesvd.zgesvd('A', 'A', m, n, a.clone(), 0, n, s, 0, u, 0, m, vt, 0, n, query, 0, -1);
        assertEquals(0, info);

        int lwork = (int) Math.ceil(query[0]);
        double[] work = new double[Math.max(1, 2 * lwork)];
        info = Zgesvd.zgesvd('A', 'A', m, n, a.clone(), 0, n, s, 0, u, 0, m, vt, 0, n, work, 0, lwork);
        assertEquals(0, info);
    }

    @Test
    void testSquareSVDOverwriteScipy() {
        {
            int m = 4, n = 4;
            double[] A = GESVD_A_SQ.clone();
            double[] s = new double[4];
            double[] work = new double[4096];

            int info = Zgesvd.zgesvd('O', 'N', m, n, A, 0, n, s, 0, null, 0, m, null, 0, n, work, 0, 4096);
            assertEquals(0, info);

            for (int i = 0; i < GESVD_S_SQ.length; i++) {
                assertEquals(GESVD_S_SQ[i], s[i], 1e-10);
            }

            verifyUnitaryUHxU(A, m, 1e-9);
        }
        {
            int m = 4, n = 4;
            double[] A = GESVD_A_SQ.clone();
            double[] s = new double[4];
            double[] work = new double[4096];

            int info = Zgesvd.zgesvd('N', 'O', m, n, A, 0, n, s, 0, null, 0, m, null, 0, n, work, 0, 4096);
            assertEquals(0, info);

            for (int i = 0; i < GESVD_S_SQ.length; i++) {
                assertEquals(GESVD_S_SQ[i], s[i], 1e-10);
            }

            verifyUnitaryVTHxVT(A, n, 1e-9);
        }
    }

    @Test
    void testSVDNoUFullVTScipy() {
        int n = 4;
        double[] A = GESVD_A_SQ.clone();
        double[] s = new double[4];
        double[] vt = new double[4 * 4 * 2];
        double[] work = new double[4096];

        int info = Zgesvd.zgesvd('N', 'A', 4, 4, A, 0, 4, s, 0, null, 0, 4, vt, 0, 4, work, 0, 4096);
        assertEquals(0, info);

        for (int i = 0; i < GESVD_S_SQ.length; i++) {
            assertEquals(GESVD_S_SQ[i], s[i], 1e-10);
        }

        verifyUnitaryVTHxVT(vt, n, 1e-9);
    }

    @Test
    void testSVDFullUNoVTScipy() {
        int m = 4;
        double[] A = GESVD_A_SQ.clone();
        double[] s = new double[4];
        double[] u = new double[4 * 4 * 2];
        double[] work = new double[4096];

        int info = Zgesvd.zgesvd('A', 'N', 4, 4, A, 0, 4, s, 0, u, 0, 4, null, 0, 4, work, 0, 4096);
        assertEquals(0, info);

        for (int i = 0; i < GESVD_S_SQ.length; i++) {
            assertEquals(GESVD_S_SQ[i], s[i], 1e-10);
        }

        verifyUnitaryUHxU(u, m, 1e-9);
    }

    @Test
    void testOffsetSubmatrixMatchesDirectFullSvd() {
        int m = 4;
        int n = 4;
        int minmn = Math.min(m, n);
        int lda = 6;
        int ldu = 6;
        int ldvt = 7;
        int aOff = 5;
        int uOff = 4;
        int vtOff = 6;

        double[] directA = GESVD_A_SQ.clone();
        double[] offsetA = new double[(aOff + (m - 1) * lda + n + 1) * 2];
        copyMatrixToOffset(directA, m, n, n, offsetA, aOff, lda);

        double[] directS = new double[minmn];
        double[] offsetS = new double[minmn];
        double[] directU = new double[m * m * 2];
        double[] offsetU = new double[(uOff + (m - 1) * ldu + m + 1) * 2];
        double[] directVT = new double[n * n * 2];
        double[] offsetVT = new double[(vtOff + (n - 1) * ldvt + n + 1) * 2];
        double[] query = new double[1];

        assertEquals(0, Zgesvd.zgesvd('A', 'A', m, n, directA, 0, n, directS, 0, directU, 0, m, directVT, 0, n, query, 0, -1));
        int lwork = (int) Math.ceil(query[0]);
        double[] directWork = new double[Math.max(1, 2 * lwork)];
        double[] offsetWork = new double[Math.max(1, 2 * lwork)];

        assertEquals(0, Zgesvd.zgesvd('A', 'A', m, n, directA, 0, n, directS, 0, directU, 0, m, directVT, 0, n, directWork, 0, lwork));
        assertEquals(0, Zgesvd.zgesvd('A', 'A', m, n, offsetA, aOff, lda, offsetS, 0, offsetU, uOff, ldu, offsetVT, vtOff, ldvt, offsetWork, 0, lwork));

        assertRealVectorEquals(directS, offsetS);
        assertOffsetSubmatrixEquals(directU, m, m, m, offsetU, uOff, ldu);
        assertOffsetSubmatrixEquals(directVT, n, n, n, offsetVT, vtOff, ldvt);
    }

    private static void copyMatrixToOffset(double[] src, int rows, int cols, int srcLda, double[] dst, int aOff, int dstLda) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                int srcPos = (i * srcLda + j) * 2;
                int dstPos = (aOff + i * dstLda + j) * 2;
                dst[dstPos] = src[srcPos];
                dst[dstPos + 1] = src[srcPos + 1];
            }
        }
    }

    private static void assertOffsetSubmatrixEquals(double[] expected, int rows, int cols, int expectedLda, double[] actual, int aOff, int actualLda) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                int expectedPos = (i * expectedLda + j) * 2;
                int actualPos = (aOff + i * actualLda + j) * 2;
                assertEquals(expected[expectedPos], actual[actualPos], TOL);
                assertEquals(expected[expectedPos + 1], actual[actualPos + 1], TOL);
            }
        }
    }

    private static void assertRealVectorEquals(double[] expected, double[] actual) {
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], actual[i], TOL);
        }
    }

    private static double[] randomComplexMatrix(int rows, int cols, long seed) {
        java.util.Random random = new java.util.Random(seed);
        double[] data = new double[rows * cols * 2];
        for (int i = 0; i < data.length; i++) {
            data[i] = random.nextDouble() * 2.0 - 1.0;
        }
        return data;
    }
}
