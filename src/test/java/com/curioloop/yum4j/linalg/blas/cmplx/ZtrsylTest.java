/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.yum4j.linalg.blas.cmplx;

import com.curioloop.yum4j.linalg.blas.BLAS;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ZtrsylTest {

    private static final double EPSILON = 1e-10;

    static final double[] TRSYL_A = {
            8.125258223941980e-01, -2.196718878375119e-01, 1.356240028570823e+00, 3.571125715117464e-01, -7.201012158033385e-02, 1.477894044741516e+00, 1.003532897892024e+00, -5.182702182736474e-01,
            0.000000000000000e+00, 0.000000000000000e+00, -6.451197546051243e-01, -5.017570435845365e-01, 3.613956055084139e-01, 9.154021177020741e-01, 1.538036566465969e+00, 3.287511096596845e-01,
            0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, -2.619745104089744e+00, 9.707754934804039e-02, 8.219025043752238e-01, 9.686449905328892e-01,
            0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, -1.987568914600893e+00, -1.463514948132119e+00
    };

    static final double[] TRSYL_B = {
            2.961202770645761e-01, 6.023020994102644e-02, 2.610552721798893e-01, 2.463242112485286e+00, 5.113456642460900e-03, -1.923609647811225e-01, -2.345871333751474e-01, 3.015473423336125e-01,
            0.000000000000000e+00, 0.000000000000000e+00, -4.206453227653590e-01, -1.168678037619532e+00, -3.427145165267695e-01, 1.142822814515021e+00, -8.022772692216189e-01, 7.519330326867741e-01,
            0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 1.886185901210530e+00, 1.402794310936099e+00, 1.745778128318390e-01, -1.401851062792281e+00,
            0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, -2.651387544921688e-02, -5.662977296027719e-01
    };

    static final double[] TRSYL_C = {
            9.965136508764122e-02, 2.598827942484235e-01, -5.034756541161992e-01, 7.818228717773100e-01, -1.550663431066133e+00, -1.236950710878082e+00, 6.856297480602733e-02, -1.320456613084276e+00,
            -1.062303713726105e+00, 5.219415656168976e-01, 4.735924306351816e-01, 2.969846732331861e-01, -9.194242342338034e-01, 2.504928503458766e-01, 1.549934405017540e+00, 3.464482094969759e-01,
            -7.832532923362371e-01, -6.800247215784908e-01, -3.220615162056756e-01, 2.322536971610036e-01, 8.135172173696698e-01, 2.930724732986812e-01, -1.230864316433955e+00, -7.143514180263678e-01,
            2.274599346041294e-01, 1.865774511144757e+00, 1.307142754282428e+00, 4.738329209117875e-01, -1.607483234561228e+00, -1.191303497202649e+00, 1.846338585323042e-01, 6.565536086338297e-01
    };

    static final double[] TRSYL_X_NN_p1 = {
            1.066135734008878e+00, 3.305774453548225e+00, 3.738779902453628e-01, 5.474552301436885e+00, 1.003253808571531e+01, 1.808330279553913e+00, -5.043375915345035e+00, -1.337509973447997e+00,
            -1.243712383723022e+00, -2.274237200281469e+00, -7.375431634549138e-01, -3.124402357825717e+00, -1.235849962695148e+01, 5.886305561451620e+00, 1.084279283436509e+01, 1.915624668527965e+00,
            3.720844811940001e-01, -1.490502429568623e-01, 2.233925771045123e-01, -1.191941078348358e-01, -4.391860625629428e+00, 1.377969414317480e+01, 7.506887356737111e+00, -2.389604624572401e+00,
            -6.217021284442789e-01, -5.872773080741295e-01, -4.507176992233770e-01, -4.036898549504294e-01, 1.934545559964887e+01, -2.401948585289297e+00, -6.815008427763484e+00, -7.163002858602504e+00
    };

    static final double[] TRSYL_X_NN_m1 = {
            -1.012036339804363e+00, 2.946852833012107e+00, -5.073111669945713e-01, 7.183154066688129e+00, 4.046084607287931e+00, -1.496431488805605e+00, 1.656521365673698e+01, 1.139497745736931e+00,
            -1.619971127347668e-03, -1.272590756007318e+00, -2.574863882658436e+00, -7.054093051607337e+00, -1.650664492271436e+00, 1.520024015878020e+00, -1.832151885539687e+01, -1.115131286467188e+01,
            3.161168297785921e-01, -5.737641072937361e-02, -2.721035153641659e-01, -9.589345827355482e-01, -2.973402415416581e-01, 2.297652719462620e-01, -4.678479130821457e-01, -1.199629717533674e-02,
            -4.461187177925614e-01, -5.193365539018918e-01, -1.434175889584520e+00, 7.552946626995285e-01, 7.182355877705821e-01, 2.445475711868731e-01, -2.443988684959072e-01, 1.134057008956137e+00
    };

    static final double[] TRSYL_X_TN = {
            5.503464932958501e-02, 2.423294604212581e-01, -3.740439954606472e-01, 1.625339089345368e-01, -5.800854732031366e-01, -2.153711379857940e-02, 1.136196184761197e+00, -9.986735604566827e-01,
            9.153311658085651e-01, -1.655516396739687e+00, 1.575796297503817e+00, -9.569556555286938e-01, -7.589035952480686e-01, -6.233516049340413e-01, 1.140040020584458e+00, 1.577195479025257e+00,
            9.428325533137323e-01, 4.870943571885954e-01, 4.592166966556605e-01, 7.424635554041602e-01, 7.083744789916818e-01, -1.247148357695227e+00, -3.047667098820038e-01, 1.107583980915700e+00,
            2.301364069676059e-02, -1.546083572574946e+00, 7.977619834637827e-01, -1.022498286033662e+00, 2.281338819810852e+01, -2.093232650852756e+00, -6.805965686462590e+00, -7.732388340482158e+00
    };

    static final double[] TRSYL_X_CN = {
            1.401361792249309e-01, 1.990341045861420e-01, -3.648040474508717e-01, 9.817622163401503e-02, -5.605885416481668e-01, 5.524007289115514e-02, 7.865806678493825e-01, -1.890310769950050e+00,
            1.443288556764074e+00, 1.458632082766670e+00, -1.151295362001704e+00, 4.382033015687487e+00, 2.124764250338437e+00, -7.925888055420479e-01, -8.949607208032955e+00, -1.491687934858180e+01,
            1.255836553655602e+00, -1.642840879752907e-01, 2.050276013646990e+00, 1.115895493862281e+00, -8.780285598813240e-01, -1.537999428407561e+00, -7.969660728012512e+00, 3.704275071165439e+00,
            1.449608890982403e+00, 6.106524019560492e-01, -6.309285880836039e-01, 3.749308490381794e+00, 1.247868568974414e+00, -7.877534888337461e-01, -6.810467056522515e+00, -1.120505776085591e+01
    };

    static final double[] TRSYL_X_NT = {
            1.210766941962929e+01, 1.441313748528525e+01, 1.471126316682130e+00, 6.250461037697581e+00, 6.091122046586520e+00, 5.985997393519455e+00, 3.300126645023313e+00, -7.467803780513551e-01,
            -7.757385289642951e+00, -1.482863358862232e+01, -3.902765729485999e+00, -3.509767690672951e+00, -1.150141705073086e+01, 4.884275797685350e-01, -8.322971832355482e-01, 1.103639667310026e+00,
            -1.663892929442060e+00, -4.521884311185872e+00, -4.416051575205050e+00, -5.432187020685567e-01, -7.863347163250519e+00, 9.613238070526341e+00, 4.564100108382291e-01, 7.671854805161302e-02,
            -4.632310245534343e+00, 3.528209669264052e+00, 1.242978739456142e+00, 5.650034079822338e+00, 1.658831897247378e+01, 4.498341683979702e+00, -2.084649283275331e-01, -1.158884110447618e-01
    };

    static final double[] TRSYL_X_NC = {
            5.099308396826955e-01, 1.459042391649775e+01, 1.230685550287711e+00, -9.273280711215422e+00, 6.142683634090184e-01, -1.890138182063759e+00, 4.434163651548801e+00, -1.840177867601073e+00,
            1.160285861628566e+01, -9.884363133381251e+00, -7.855061894136419e-01, 3.559264583667249e+00, -2.304666889794979e+00, 4.327632245098479e-01, -2.660384122598713e+00, -6.985059051656726e-01,
            -3.519232888335935e-01, 1.885063502594512e-01, 7.516642067344974e-02, -5.298308855159868e-01, 2.536253302184369e-02, 1.376503127158195e-01, 4.326371885504220e-01, 2.321518320688557e-01,
            3.662584246984494e-02, 2.899241048021438e-01, -9.088423015836393e-01, -6.949554758318686e-03, 3.275758703046882e-01, -6.535602132095349e-01, -1.976619358201753e-01, -2.379286088613086e-01
    };

    @Test
    @DisplayName("ztrsyl: 1x1 complex diagonal")
    void test1x1Diagonal() {
        double[] a = {2, 0};
        double[] b = {3, 0};
        double[] c = {10, 0};

        boolean[] okOut = new boolean[1];
        double scale = Ztrsyl.ztrsyl('N', 'N', 1, 1, 1,
                a, 0, 1, b, 0, 1, c, 0, 1, okOut);

        assertTrue(okOut[0]);
        assertEquals(1.0, scale, EPSILON);
        assertEquals(2.0, c[0], EPSILON);
        assertEquals(0.0, c[1], EPSILON);
    }

    @Test
    @DisplayName("ztrsyl: 2x2 upper triangular complex matrices")
    void test2x2UpperTriangular() {
        double[] a = {1, 1, 2, 0, 0, 0, 3, -1};
        double[] b = {2, 0, 1, 1, 0, 0, 4, 0};
        double[] c = {1, 0, 0, 0, 0, 0, 1, 0};

        double[] cOrig = c.clone();

        boolean[] okOut = new boolean[1];
        double scale = Ztrsyl.ztrsyl('N', 'N', 1, 2, 2,
                a, 0, 2, b, 0, 2, c, 0, 2, okOut);

        assertTrue(okOut[0]);
        verifyResidual('N', 'N', 1, 2, 2, a, 2, b, 2, c, 2, cOrig, scale);
    }

    @Test
    @DisplayName("ztrsyl: A*X - X*B = C (isgn=-1)")
    void testIsgnMinus1() {
        double[] a = {1, 1, 2, 0, 0, 0, 3, -1};
        double[] b = {2, 0, 1, 1, 0, 0, 4, 0};
        double[] c = {1, 0, 0, 0, 0, 0, 1, 0};

        double[] cOrig = c.clone();

        boolean[] okOut = new boolean[1];
        double scale = Ztrsyl.ztrsyl('N', 'N', -1, 2, 2,
                a, 0, 2, b, 0, 2, c, 0, 2, okOut);

        assertTrue(okOut[0]);
        verifyResidual('N', 'N', -1, 2, 2, a, 2, b, 2, c, 2, cOrig, scale);
    }

    @Test
    @DisplayName("ztrsyl: trana='C' (conjugate transpose of A)")
    void testTranaC() {
        double[] a = {1, 1, 2, 0, 0, 0, 3, -1};
        double[] b = {2, 0, 1, 1, 0, 0, 4, 0};
        double[] c = {1, 0, 0, 0, 0, 0, 1, 0};

        double[] cOrig = c.clone();

        boolean[] okOut = new boolean[1];
        double scale = Ztrsyl.ztrsyl('C', 'N', 1, 2, 2,
                a, 0, 2, b, 0, 2, c, 0, 2, okOut);

        assertTrue(okOut[0]);
        verifyResidual('C', 'N', 1, 2, 2, a, 2, b, 2, c, 2, cOrig, scale);
    }

    @Test
    @DisplayName("ztrsyl: tranb='C' (conjugate transpose of B)")
    void testTranbC() {
        double[] a = {1, 1, 2, 0, 0, 0, 3, -1};
        double[] b = {2, 0, 1, 1, 0, 0, 4, 0};
        double[] c = {1, 0, 0, 0, 0, 0, 1, 0};

        double[] cOrig = c.clone();

        boolean[] okOut = new boolean[1];
        double scale = Ztrsyl.ztrsyl('N', 'C', 1, 2, 2,
                a, 0, 2, b, 0, 2, c, 0, 2, okOut);

        assertTrue(okOut[0]);
        verifyResidual('N', 'C', 1, 2, 2, a, 2, b, 2, c, 2, cOrig, scale);
    }

    @Test
    @DisplayName("ztrsyl: m=0 or n=0 boundary case")
    void testM0N0() {
        double[] a0 = {};
        double[] b2 = {1, 0, 0, 0, 0, 0, 1, 0};
        double[] c0 = {};

        boolean[] okOut1 = new boolean[1];
        double scale1 = Ztrsyl.ztrsyl('N', 'N', 1, 0, 2,
                a0, 0, 1, b2, 0, 2, c0, 0, 1, okOut1);

        assertEquals(1.0, scale1, EPSILON);
        assertTrue(okOut1[0]);

        double[] a2 = {1, 0, 0, 0, 0, 0, 1, 0};
        double[] b0 = {};

        boolean[] okOut2 = new boolean[1];
        double scale2 = Ztrsyl.ztrsyl('N', 'N', 1, 2, 0,
                a2, 0, 2, b0, 0, 1, c0, 0, 2, okOut2);

        assertEquals(1.0, scale2, EPSILON);
        assertTrue(okOut2[0]);
    }

    @Test
    void testNNIsgnP1Scipy() {
        int n = 4;
        double[] C_work = TRSYL_C.clone();
        double scale = Ztrsyl.ztrsyl('N', 'N', 1, n, n, TRSYL_A, 0, n, TRSYL_B, 0, n, C_work, 0, n, null);
        double[] diff = new double[n * n * 2];
        for (int i = 0; i < n * n * 2; i++) diff[i] = C_work[i] - TRSYL_X_NN_p1[i];
        assertTrue(zmaxNorm(diff, n) / zmaxNorm(TRSYL_X_NN_p1, n) < EPSILON);
        verifyResidual('N', 'N', 1, n, TRSYL_A, TRSYL_B, C_work, TRSYL_C, scale);
    }

    @Test
    void testNNIsgnM1Scipy() {
        int n = 4;
        double[] C_work = TRSYL_C.clone();
        double scale = Ztrsyl.ztrsyl('N', 'N', -1, n, n, TRSYL_A, 0, n, TRSYL_B, 0, n, C_work, 0, n, null);
        double[] diff = new double[n * n * 2];
        for (int i = 0; i < n * n * 2; i++) diff[i] = C_work[i] - TRSYL_X_NN_m1[i];
        assertTrue(zmaxNorm(diff, n) / zmaxNorm(TRSYL_X_NN_m1, n) < EPSILON);
        verifyResidual('N', 'N', -1, n, TRSYL_A, TRSYL_B, C_work, TRSYL_C, scale);
    }

    @Test
    void testTNScipy() {
        int n = 4;
        double[] C_work = TRSYL_C.clone();
        double scale = Ztrsyl.ztrsyl('T', 'N', 1, n, n, TRSYL_A, 0, n, TRSYL_B, 0, n, C_work, 0, n, null);
        double[] diff = new double[n * n * 2];
        for (int i = 0; i < n * n * 2; i++) diff[i] = C_work[i] - TRSYL_X_TN[i];
        assertTrue(zmaxNorm(diff, n) / zmaxNorm(TRSYL_X_TN, n) < EPSILON);
        verifyResidual('T', 'N', 1, n, TRSYL_A, TRSYL_B, C_work, TRSYL_C, scale);
    }

    @Test
    void testCNScipy() {
        int n = 4;
        double[] C_work = TRSYL_C.clone();
        double scale = Ztrsyl.ztrsyl('C', 'N', 1, n, n, TRSYL_A, 0, n, TRSYL_B, 0, n, C_work, 0, n, null);
        double[] diff = new double[n * n * 2];
        for (int i = 0; i < n * n * 2; i++) diff[i] = C_work[i] - TRSYL_X_CN[i];
        assertTrue(zmaxNorm(diff, n) / zmaxNorm(TRSYL_X_CN, n) < EPSILON);
        verifyResidual('C', 'N', 1, n, TRSYL_A, TRSYL_B, C_work, TRSYL_C, scale);
    }

    @Test
    void testNTScipy() {
        int n = 4;
        double[] C_work = TRSYL_C.clone();
        double scale = Ztrsyl.ztrsyl('N', 'T', 1, n, n, TRSYL_A, 0, n, TRSYL_B, 0, n, C_work, 0, n, null);
        double[] diff = new double[n * n * 2];
        for (int i = 0; i < n * n * 2; i++) diff[i] = C_work[i] - TRSYL_X_NT[i];
        assertTrue(zmaxNorm(diff, n) / zmaxNorm(TRSYL_X_NT, n) < EPSILON);
        verifyResidual('N', 'T', 1, n, TRSYL_A, TRSYL_B, C_work, TRSYL_C, scale);
    }

    @Test
    void testNCScipy() {
        int n = 4;
        double[] C_work = TRSYL_C.clone();
        double scale = Ztrsyl.ztrsyl('N', 'C', 1, n, n, TRSYL_A, 0, n, TRSYL_B, 0, n, C_work, 0, n, null);
        double[] diff = new double[n * n * 2];
        for (int i = 0; i < n * n * 2; i++) diff[i] = C_work[i] - TRSYL_X_NC[i];
        assertTrue(zmaxNorm(diff, n) / zmaxNorm(TRSYL_X_NC, n) < EPSILON);
        verifyResidual('N', 'C', 1, n, TRSYL_A, TRSYL_B, C_work, TRSYL_C, scale);
    }

    @Test
    void testTTScipy() {
        int n = 4;
        double[] C_work = TRSYL_C.clone();
        double scale = Ztrsyl.ztrsyl('T', 'T', 1, n, n, TRSYL_A, 0, n, TRSYL_B, 0, n, C_work, 0, n, null);
        assertTrue(scale > 0);
        verifyResidual('T', 'T', 1, n, TRSYL_A, TRSYL_B, C_work, TRSYL_C, scale);
    }

    @Test
    void testTCScipy() {
        int n = 4;
        double[] C_work = TRSYL_C.clone();
        double scale = Ztrsyl.ztrsyl('T', 'C', 1, n, n, TRSYL_A, 0, n, TRSYL_B, 0, n, C_work, 0, n, null);
        assertTrue(scale > 0);
        verifyResidual('T', 'C', 1, n, TRSYL_A, TRSYL_B, C_work, TRSYL_C, scale);
    }

    @Test
    void testCTScipy() {
        int n = 4;
        double[] C_work = TRSYL_C.clone();
        double scale = Ztrsyl.ztrsyl('C', 'T', 1, n, n, TRSYL_A, 0, n, TRSYL_B, 0, n, C_work, 0, n, null);
        assertTrue(scale > 0);
        verifyResidual('C', 'T', 1, n, TRSYL_A, TRSYL_B, C_work, TRSYL_C, scale);
    }

    @Test
    void testCCScipy() {
        int n = 4;
        double[] C_work = TRSYL_C.clone();
        double scale = Ztrsyl.ztrsyl('C', 'C', 1, n, n, TRSYL_A, 0, n, TRSYL_B, 0, n, C_work, 0, n, null);
        assertTrue(scale > 0);
        verifyResidual('C', 'C', 1, n, TRSYL_A, TRSYL_B, C_work, TRSYL_C, scale);
    }

    @Test
    void testTNIsgnM1Scipy() {
        int n = 4;
        double[] C_work = TRSYL_C.clone();
        double scale = Ztrsyl.ztrsyl('T', 'N', -1, n, n, TRSYL_A, 0, n, TRSYL_B, 0, n, C_work, 0, n, null);
        assertTrue(scale > 0);
        verifyResidual('T', 'N', -1, n, TRSYL_A, TRSYL_B, C_work, TRSYL_C, scale);
    }

    @Test
    void testCNIsgnM1Scipy() {
        int n = 4;
        double[] C_work = TRSYL_C.clone();
        double scale = Ztrsyl.ztrsyl('C', 'N', -1, n, n, TRSYL_A, 0, n, TRSYL_B, 0, n, C_work, 0, n, null);
        assertTrue(scale > 0);
        verifyResidual('C', 'N', -1, n, TRSYL_A, TRSYL_B, C_work, TRSYL_C, scale);
    }

    @Test
    void testNTIsgnM1Scipy() {
        int n = 4;
        double[] C_work = TRSYL_C.clone();
        double scale = Ztrsyl.ztrsyl('N', 'T', -1, n, n, TRSYL_A, 0, n, TRSYL_B, 0, n, C_work, 0, n, null);
        assertTrue(scale > 0);
        verifyResidual('N', 'T', -1, n, TRSYL_A, TRSYL_B, C_work, TRSYL_C, scale);
    }

    @Test
    void testNCIsgnM1Scipy() {
        int n = 4;
        double[] C_work = TRSYL_C.clone();
        double scale = Ztrsyl.ztrsyl('N', 'C', -1, n, n, TRSYL_A, 0, n, TRSYL_B, 0, n, C_work, 0, n, null);
        assertTrue(scale > 0);
        verifyResidual('N', 'C', -1, n, TRSYL_A, TRSYL_B, C_work, TRSYL_C, scale);
    }

    @Test
    void testRawOffsetMatchesDirectNNIsgnP1() {
        assertRawOffsetMatchesDirect('N', 'N', 1);
    }

    @Test
    void testRawOffsetMatchesDirectCNIsgnM1() {
        assertRawOffsetMatchesDirect('C', 'N', -1);
    }

    private void assertRawOffsetMatchesDirect(char trana, char tranb, int isgn) {
        int n = 4;
        int lda = 6;
        int ldb = 7;
        int ldc = 8;
        int aOff = 6;
        int bOff = 8;
        int cOff = 10;

        double[] directA = TRSYL_A.clone();
        double[] directB = TRSYL_B.clone();
        double[] directC = TRSYL_C.clone();
        double[] offsetA = new double[aOff + lda * n * 2];
        double[] offsetB = new double[bOff + ldb * n * 2];
        double[] offsetC = new double[cOff + ldc * n * 2];
        copyMatrixToRawOffset(directA, n, n, n, offsetA, aOff, lda);
        copyMatrixToRawOffset(directB, n, n, n, offsetB, bOff, ldb);
        copyMatrixToRawOffset(directC, n, n, n, offsetC, cOff, ldc);

        boolean[] directOk = new boolean[1];
        boolean[] offsetOk = new boolean[1];

        double directScale = Ztrsyl.ztrsyl(trana, tranb, isgn, n, n,
                directA, 0, n, directB, 0, n, directC, 0, n, directOk);
        double offsetScale = Ztrsyl.ztrsyl(trana, tranb, isgn, n, n,
                offsetA, aOff, lda, offsetB, bOff, ldb, offsetC, cOff, ldc, offsetOk);

        assertEquals(directOk[0], offsetOk[0]);
        assertEquals(directScale, offsetScale, EPSILON);
        assertRawSubmatrixEquals(directC, n, n, n, offsetC, cOff, ldc);
    }

    private void verifyResidual(char trana, char tranb, int isgn, int m, int n,
                                double[] a, int lda, double[] b, int ldb,
                                double[] x, int ldc, double[] cOrig, double scale) {
        BLAS.Trans transA = (trana == 'N' || trana == 'n') ? BLAS.Trans.NoTrans : BLAS.Trans.Conj;
        BLAS.Trans transB = (tranb == 'N' || tranb == 'n') ? BLAS.Trans.NoTrans : BLAS.Trans.Conj;

        double[] r = new double[m * n * 2];

        ZLAS.zgemm(transA, BLAS.Trans.NoTrans, m, n, m, 1, 0,
                a, 0, lda, x, 0, ldc, 0, 0, r, 0, n);
        ZLAS.zgemm(BLAS.Trans.NoTrans, transB, m, n, n, isgn, 0,
                x, 0, ldc, b, 0, ldb, 1, 0, r, 0, n);

        double normR = 0, normC = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                int rIdx = (i * n + j) * 2;
                int cIdx = (i * ldc + j) * 2;
                r[rIdx] -= scale * cOrig[cIdx];
                r[rIdx + 1] -= scale * cOrig[cIdx + 1];
                double absR = Math.sqrt(r[rIdx] * r[rIdx] + r[rIdx + 1] * r[rIdx + 1]);
                double absC = Math.sqrt(cOrig[cIdx] * cOrig[cIdx] + cOrig[cIdx + 1] * cOrig[cIdx + 1]);
                normR = Math.max(normR, absR);
                normC = Math.max(normC, absC);
            }
        }

        assertTrue(normR / (normC + EPSILON) < EPSILON,
                "Residual too large: " + normR / (normC + EPSILON));
    }

    private static double zmaxNorm(double[] M, int n) {
        double norm = 0;
        for (int i = 0; i < n * n; i++) {
            norm = Math.max(norm, Math.hypot(M[i * 2], M[i * 2 + 1]));
        }
        return norm;
    }

    private static BLAS.Trans toTrans(char tr) {
        if (tr == 'N') return BLAS.Trans.NoTrans;
        if (tr == 'T') return BLAS.Trans.Trans;
        return BLAS.Trans.Conj;
    }

    private static void copyMatrixToRawOffset(double[] src, int rows, int cols, int srcLda, double[] dst, int rawOff, int dstLda) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                int srcPos = (i * srcLda + j) * 2;
                int dstPos = rawOff + (i * dstLda + j) * 2;
                dst[dstPos] = src[srcPos];
                dst[dstPos + 1] = src[srcPos + 1];
            }
        }
    }

    private static void assertRawSubmatrixEquals(double[] expected, int rows, int cols, int expectedLda, double[] actual, int rawOff, int actualLda) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                int expectedPos = (i * expectedLda + j) * 2;
                int actualPos = rawOff + (i * actualLda + j) * 2;
                assertEquals(expected[expectedPos], actual[actualPos], EPSILON);
                assertEquals(expected[expectedPos + 1], actual[actualPos + 1], EPSILON);
            }
        }
    }

    private void verifyResidual(char trana, char tranb, int isgn, int n,
                                double[] A, double[] B, double[] X, double[] C, double scale) {
        double[] r = new double[n * n * 2];
        ZLAS.zgemm(toTrans(trana), BLAS.Trans.NoTrans, n, n, n, 1, 0,
                A, 0, n, X, 0, n, 0, 0, r, 0, n);
        ZLAS.zgemm(BLAS.Trans.NoTrans, toTrans(tranb), n, n, n, isgn, 0,
                X, 0, n, B, 0, n, 1, 0, r, 0, n);
        for (int i = 0; i < n * n; i++) {
            r[i * 2] -= scale * C[i * 2];
            r[i * 2 + 1] -= scale * C[i * 2 + 1];
        }
        double normR = zmaxNorm(r, n);
        double normC = zmaxNorm(C, n);
        assertTrue(normR / normC < EPSILON, "Residual too large: " + normR / normC);
    }
}
