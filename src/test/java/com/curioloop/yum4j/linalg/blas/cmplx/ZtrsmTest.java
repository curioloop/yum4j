package com.curioloop.yum4j.linalg.blas.cmplx;

import com.curioloop.yum4j.linalg.blas.BLAS;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertTrue;

public class ZtrsmTest {

    private static final double TOL = 1e-10;

    static final double[] TRSM_A_UPPER = {
            2.449770784502578e-01, -1.510640841669740e-01, 1.414555069896746e+00, -1.024898415434741e+00, -8.313440625510439e-02, 1.779597445791622e+00, 1.251510875447925e+00, -4.855949821024690e-01,
            0.000000000000000e+00, 0.000000000000000e+00, -1.686819866346470e-01, -7.417515526549459e-01, 3.510968530541845e-01, 9.519860452638837e-01, 1.549933897930296e+00, 3.287511096596845e-01,
            0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, -2.619745104089744e+00, 9.707754934804039e-02, 8.219025043752238e-01, 9.686449905328892e-01,
            0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, -1.987568914600893e+00, -1.463514948132119e+00
    };

    static final double[] TRSM_A_LOWER = {
            2.449770784502578e-01, -1.510640841669740e-01, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00,
            1.414555069896746e+00, -1.024898415434741e+00, -1.686819866346470e-01, -7.417515526549459e-01, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00,
            -8.313440625510439e-02, 1.779597445791622e+00, 3.510968530541845e-01, 9.519860452638837e-01, -2.619745104089744e+00, 9.707754934804039e-02, 0.000000000000000e+00, 0.000000000000000e+00,
            1.251510875447925e+00, -4.855949821024690e-01, 1.549933897930296e+00, 3.287511096596845e-01, 8.219025043752238e-01, 9.686449905328892e-01, -1.987568914600893e+00, -1.463514948132119e+00
    };

    static final double[] TRSM_B = {
            -9.746816702273214e-01, 1.300189187790702e-02, 7.870846037424520e-01, 1.453534077157317e+00, 1.158595579007404e+00, -2.646568332379561e-01, -8.206823183517105e-01, 2.720169166589619e+00,
            9.633761292443218e-01, 6.256673477650062e-01, 4.127809269364983e-01, -8.571575564162826e-01, 8.220601599944900e-01, -1.070892498061112e+00, 1.896792982653947e+00, 4.824724152431853e-01,
            -2.453881160028705e-01, -2.234627853258508e-01, -7.537361643574896e-01, 7.140004940920919e-01, -8.895144296255233e-01, 4.732376245735448e-01, -8.158102849654383e-01, -7.282891265687277e-02,
            -7.710170941410412e-02, -8.467937180684050e-01, 3.411519748166435e-01, -1.514847224685865e+00, 2.766907993300191e-01, -4.465149520670211e-01, 8.271832490360238e-01, 8.563987943234723e-01
    };

    static final double[] TRSM_X_LU_N = {
            -1.113725450634664e+00, -9.742791702258998e+00, -1.586496649887645e+01, -1.524097796622059e+00, -7.588388656841918e+00, -8.655497308502804e+00, 2.388965268291222e+00, -1.229759027198001e+01,
            -2.719410403919895e-01, 1.169432757376966e+00, 2.493022052237730e+00, 7.876112627368521e-01, 1.889421495128379e+00, 1.343109894964340e+00, -1.505788211516814e+00, 2.865881296933697e+00,
            6.070932352019718e-02, 2.529249124642435e-01, 1.536590352121922e-01, 7.308750418560305e-03, 2.700600186987893e-01, -9.779594784976341e-02, 1.981848194806854e-01, -1.660177214996384e-01,
            2.285735780504517e-01, 2.577384190708126e-01, 2.526036033377453e-01, 5.761601858615242e-01, 1.699508723270254e-02, 2.121397576505256e-01, -4.755893839538786e-01, -8.068481075345607e-02
    };

    static final double[] TRSM_X_LU_T = {
            -2.906275433540883e+00, -1.739068599895263e+00, -3.230444816254097e-01, 5.734143240183219e+00, 3.909131609138459e+00, 1.330216505340856e+00, -7.387875583869132e+00, 6.548067752790167e+00,
            -2.136044504049052e+00, 8.758349311039462e+00, 1.338037927323621e+01, -3.707629888929291e+00, 4.187773639113359e-01, -8.089391044664273e+00, 1.931802025085552e+01, 1.199168048049759e+01,
            -2.045681355884359e+00, -1.511981651549205e+00, -5.926821762795649e-01, 3.669466465199041e+00, 2.248893938078742e+00, 1.584002892622698e+00, -5.790007555688959e+00, 3.213924092136745e+00,
            -1.189776292957267e+00, 5.771499799184859e+00, 8.898528612080888e+00, -1.550131239603527e+00, 8.067323606747112e-01, -4.974769022516099e+00, 1.156744508762065e+01, 8.033517086458399e+00
    };

    static final double[] TRSM_X_LU_C = {
            -2.858852493934415e+00, 1.815973267608829e+00, 4.978557731672618e+00, 2.863340592463692e+00, 2.943825549301971e+00, -2.895630678094778e+00, 2.533635786776050e+00, 9.541414289183122e+00,
            -7.371728630407324e-01, -9.092290721738218e+00, -1.175439328306026e+01, 7.654563455186228e+00, 1.849664020663121e+00, 8.086084169619262e+00, -2.237021436170092e+01, -5.821833207840221e+00,
            -1.944416145165380e+00, 1.091084125313513e+00, 3.333992602310219e+00, 1.428377303675252e+00, 1.401334499491373e+00, -1.728881660425732e+00, 1.794760472784720e+00, 5.286251299722490e+00,
            -7.109395623576740e-01, -5.222007892769752e+00, -7.399557027299331e+00, 5.212105216769707e+00, 1.425054945662556e+00, 4.771732328145379e+00, -1.371466990272734e+01, -3.431048343697639e+00
    };

    static final double[] TRSM_X_LL_N = {
            -2.906275433540883e+00, -1.739068599895263e+00, -3.230444816254097e-01, 5.734143240183219e+00, 3.909131609138459e+00, 1.330216505340856e+00, -7.387875583869132e+00, 6.548067752790167e+00,
            -2.136044504049052e+00, 8.758349311039462e+00, 1.338037927323621e+01, -3.707629888929291e+00, 4.187773639113359e-01, -8.089391044664273e+00, 1.931802025085552e+01, 1.199168048049759e+01,
            -2.045681355884359e+00, -1.511981651549205e+00, -5.926821762795649e-01, 3.669466465199041e+00, 2.248893938078742e+00, 1.584002892622698e+00, -5.790007555688959e+00, 3.213924092136745e+00,
            -1.189776292957267e+00, 5.771499799184859e+00, 8.898528612080888e+00, -1.550131239603527e+00, 8.067323606747112e-01, -4.974769022516099e+00, 1.156744508762065e+01, 8.033517086458399e+00
    };

    static final double[] TRSM_X_LL_T = {
            -1.113725450634664e+00, -9.742791702258998e+00, -1.586496649887645e+01, -1.524097796622059e+00, -7.588388656841918e+00, -8.655497308502804e+00, 2.388965268291222e+00, -1.229759027198001e+01,
            -2.719410403919895e-01, 1.169432757376966e+00, 2.493022052237730e+00, 7.876112627368521e-01, 1.889421495128379e+00, 1.343109894964340e+00, -1.505788211516814e+00, 2.865881296933697e+00,
            6.070932352019718e-02, 2.529249124642435e-01, 1.536590352121922e-01, 7.308750418560305e-03, 2.700600186987893e-01, -9.779594784976341e-02, 1.981848194806854e-01, -1.660177214996384e-01,
            2.285735780504517e-01, 2.577384190708126e-01, 2.526036033377453e-01, 5.761601858615242e-01, 1.699508723270254e-02, 2.121397576505256e-01, -4.755893839538786e-01, -8.068481075345607e-02
    };

    static final double[] TRSM_X_LL_C = {
            -3.320535153576031e+00, 8.944486338902095e+00, 1.967875590631417e+01, 7.526395480876563e+00, 1.400063386145027e+01, 5.247615609589573e+00, -3.836536104283345e+00, 3.093569959445369e+01,
            -2.721441886975776e-02, -1.155446840317152e+00, -2.031324473040094e+00, -7.388198479198791e-01, -1.517166926620335e+00, -1.117942938346911e+00, 1.077737837818273e+00, -3.227442689555561e+00,
            1.555523796957333e-01, 2.379316621602024e-01, 2.918609470060483e-01, 2.168142034225681e-02, 3.033724848798417e-01, -9.399839985715207e-02, 1.107081337129727e-01, -1.025854549814051e-01,
            -1.782658764931379e-01, 2.947816997814503e-01, -4.752001172825375e-01, 4.122547619201768e-01, -1.975316572014469e-01, 7.920450851637767e-02, -6.413519987099223e-02, -4.781024753684702e-01
    };

    static final double[] TRSM_X_RU_N = {
            -2.906275433540883e+00, -1.739068599895263e+00, -3.145869542889526e+00, 8.291035399843457e+00, -2.531121941171424e+00, -1.943834018193206e+00, -2.710221119290305e+00, 4.149812099878362e+00,
            1.708102815844518e+00, 3.607277631417348e+00, 7.057420295453530e+00, -6.080281203994117e+00, 2.178700466848792e-01, 3.212406939782010e+00, 3.661843433578557e+00, -3.224597193133446e+00,
            -3.181932839834537e-01, -1.108390891411610e+00, -2.749671436069678e+00, 4.968426223721305e-01, 6.006495513036252e-01, -1.271973950719048e+00, -1.342162028544999e+00, 1.041202646123773e-01,
            1.316268675738442e+00, -2.644952744760026e+00, -4.930384237158822e+00, 4.831244557966954e-01, 8.332921131552853e-01, -5.474996997341751e-01, -3.578027176868297e+00, -4.233744351693880e-02
    };

    static final double[] TRSM_X_RU_T = {
            2.420917869791166e+01, 1.547203280605251e+00, -4.742856590658452e+00, -1.193206050755055e-01, -1.478893369729380e-01, -3.873405973966549e-01, -3.857070700855182e-01, -1.084581816551546e+00,
            2.435071801205151e+00, -1.004931218147234e+01, 6.051653351585440e-02, 2.815705667210968e+00, -6.622158946401242e-01, 2.061502244119641e-01, -7.347155936054349e-01, 2.982512124582940e-01,
            1.168197432513313e+00, 9.128596582182475e+00, -1.362323027160376e-01, -2.094074589330508e+00, 4.963370179210901e-01, -1.114026468062113e-01, 2.836471625026400e-01, -1.722168963030569e-01,
            -4.199042123846250e+00, -1.068479226421099e+01, 9.721003817411398e-01, 1.694760038199256e+00, -2.235475571073904e-01, -3.900334392114932e-02, -4.755893839538786e-01, -8.068481075345607e-02
    };

    static final double[] TRSM_X_RU_C = {
            -2.587766158746906e+01, 6.328202639719014e+00, 3.257652689282401e+00, -9.493073213428658e-01, -4.247988366347699e-01, -4.404091605286903e-01, 9.211898947691658e-01, -6.902875042207981e-01,
            1.058220929221170e+01, 9.126179261600798e+00, -1.475366011294264e+00, -1.317670310504960e+00, -6.824032314567217e-01, 4.276788781793374e-01, -5.029131958531060e-01, -6.130571805766933e-01,
            -8.894205062885307e+00, -7.342607896795776e+00, 1.623747038582421e+00, 1.211387546905893e+00, 4.905816184036770e-01, -2.218231351738948e-01, 2.486567339388664e-01, 2.197366624733177e-01,
            7.048282336909415e+00, 2.260649841108324e+00, -1.600167241616768e+00, -5.132032767621312e-01, -3.004672153479040e-01, 5.529323863791676e-02, -6.413519987099223e-02, -4.781024753684702e-01
    };

    static final double[] TRSM_X_RL_N = {
            2.420917869791166e+01, 1.547203280605251e+00, -4.742856590658452e+00, -1.193206050755055e-01, -1.478893369729380e-01, -3.873405973966549e-01, -3.857070700855182e-01, -1.084581816551546e+00,
            2.435071801205151e+00, -1.004931218147234e+01, 6.051653351585440e-02, 2.815705667210968e+00, -6.622158946401242e-01, 2.061502244119641e-01, -7.347155936054349e-01, 2.982512124582940e-01,
            1.168197432513313e+00, 9.128596582182475e+00, -1.362323027160376e-01, -2.094074589330508e+00, 4.963370179210901e-01, -1.114026468062113e-01, 2.836471625026400e-01, -1.722168963030569e-01,
            -4.199042123846250e+00, -1.068479226421099e+01, 9.721003817411398e-01, 1.694760038199256e+00, -2.235475571073904e-01, -3.900334392114932e-02, -4.755893839538786e-01, -8.068481075345607e-02
    };

    static final double[] TRSM_X_RL_T = {
            -2.906275433540883e+00, -1.739068599895263e+00, -3.145869542889526e+00, 8.291035399843457e+00, -2.531121941171424e+00, -1.943834018193206e+00, -2.710221119290305e+00, 4.149812099878362e+00,
            1.708102815844518e+00, 3.607277631417348e+00, 7.057420295453530e+00, -6.080281203994117e+00, 2.178700466848792e-01, 3.212406939782010e+00, 3.661843433578557e+00, -3.224597193133446e+00,
            -3.181932839834537e-01, -1.108390891411610e+00, -2.749671436069678e+00, 4.968426223721305e-01, 6.006495513036252e-01, -1.271973950719048e+00, -1.342162028544999e+00, 1.041202646123773e-01,
            1.316268675738442e+00, -2.644952744760026e+00, -4.930384237158822e+00, 4.831244557966954e-01, 8.332921131552853e-01, -5.474996997341751e-01, -3.578027176868297e+00, -4.233744351693880e-02
    };

    static final double[] TRSM_X_RL_C = {
            -2.858852493934415e+00, 1.815973267608829e+00, 3.754336749620439e-01, -9.107639736688553e+00, -2.350724368045345e+00, 7.154989460954081e-01, 7.950199394448139e-01, -6.061050603436378e+00,
            3.990154160788843e+00, 9.347145424892611e-02, -5.013342283677567e+00, 8.063850544309579e+00, 1.901047880999226e+00, 5.273524601752448e-01, -3.435971446233155e+00, 4.670067170918415e+00,
            -1.133248683021787e+00, -2.133653124708496e-01, 2.607153635062363e+00, -1.443082371445577e+00, 3.533544451938858e-02, -5.461748924083473e-01, 1.875778929171775e+00, -7.930222722285417e-01,
            -1.772316256991989e+00, -2.363732923565125e+00, 4.548631791713675e+00, -1.608189476056210e+00, -1.643128399879799e+00, -3.581727878154159e-01, 2.724236991485319e+00, -1.700083303753467e+00
    };

    private static double zmaxNorm(double[] M, int n) {
        double norm = 0;
        for (int i = 0; i < n * n; i++) {
            norm = Math.max(norm, Math.hypot(M[i * 2], M[i * 2 + 1]));
        }
        return norm;
    }

    private void verifyLeftResidual(BLAS.Trans trans, int n, double[] A, double[] X, double[] B) {
        double[] r = new double[n * n * 2];
        ZLAS.zgemm(trans, BLAS.Trans.NoTrans, n, n, n, 1, 0,
                A, 0, n, X, 0, n, 0, 0, r, 0, n);
        for (int i = 0; i < n * n * 2; i++) r[i] -= B[i];
        assertTrue(zmaxNorm(r, n) / zmaxNorm(B, n) < TOL);
    }

    private void verifyRightResidual(BLAS.Trans trans, int n, double[] A, double[] X, double[] B) {
        double[] r = new double[n * n * 2];
        ZLAS.zgemm(BLAS.Trans.NoTrans, trans, n, n, n, 1, 0,
                X, 0, n, A, 0, n, 0, 0, r, 0, n);
        for (int i = 0; i < n * n * 2; i++) r[i] -= B[i];
        assertTrue(zmaxNorm(r, n) / zmaxNorm(B, n) < TOL);
    }

    @Test
    void testLeftUpperNoTrans() {
        int n = 4;
        double[] work = TRSM_B.clone();
        Ztrsm.ztrsm(BLAS.Side.Left, BLAS.Uplo.Upper, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit,
                n, n, 1, 0, TRSM_A_UPPER, 0, n, work, 0, n);
        double[] diff = new double[n * n * 2];
        for (int i = 0; i < n * n * 2; i++) diff[i] = work[i] - TRSM_X_LU_N[i];
        assertTrue(zmaxNorm(diff, n) / zmaxNorm(TRSM_X_LU_N, n) < TOL);
        verifyLeftResidual(BLAS.Trans.NoTrans, n, TRSM_A_UPPER, work, TRSM_B);
    }

    @Test
    void testLeftUpperTrans() {
        int n = 4;
        double[] work = TRSM_B.clone();
        Ztrsm.ztrsm(BLAS.Side.Left, BLAS.Uplo.Upper, BLAS.Trans.Trans, BLAS.Diag.NonUnit,
                n, n, 1, 0, TRSM_A_UPPER, 0, n, work, 0, n);
        double[] diff = new double[n * n * 2];
        for (int i = 0; i < n * n * 2; i++) diff[i] = work[i] - TRSM_X_LU_T[i];
        assertTrue(zmaxNorm(diff, n) / zmaxNorm(TRSM_X_LU_T, n) < TOL);
        verifyLeftResidual(BLAS.Trans.Trans, n, TRSM_A_UPPER, work, TRSM_B);
    }

    @Test
    void testLeftUpperConj() {
        int n = 4;
        double[] work = TRSM_B.clone();
        Ztrsm.ztrsm(BLAS.Side.Left, BLAS.Uplo.Upper, BLAS.Trans.Conj, BLAS.Diag.NonUnit,
                n, n, 1, 0, TRSM_A_UPPER, 0, n, work, 0, n);
        double[] diff = new double[n * n * 2];
        for (int i = 0; i < n * n * 2; i++) diff[i] = work[i] - TRSM_X_LU_C[i];
        assertTrue(zmaxNorm(diff, n) / zmaxNorm(TRSM_X_LU_C, n) < TOL);
        verifyLeftResidual(BLAS.Trans.Conj, n, TRSM_A_UPPER, work, TRSM_B);
    }

    @Test
    void testLeftLowerNoTrans() {
        int n = 4;
        double[] work = TRSM_B.clone();
        Ztrsm.ztrsm(BLAS.Side.Left, BLAS.Uplo.Lower, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit,
                n, n, 1, 0, TRSM_A_LOWER, 0, n, work, 0, n);
        double[] diff = new double[n * n * 2];
        for (int i = 0; i < n * n * 2; i++) diff[i] = work[i] - TRSM_X_LL_N[i];
        assertTrue(zmaxNorm(diff, n) / zmaxNorm(TRSM_X_LL_N, n) < TOL);
        verifyLeftResidual(BLAS.Trans.NoTrans, n, TRSM_A_LOWER, work, TRSM_B);
    }

    @Test
    void testLeftLowerTrans() {
        int n = 4;
        double[] work = TRSM_B.clone();
        Ztrsm.ztrsm(BLAS.Side.Left, BLAS.Uplo.Lower, BLAS.Trans.Trans, BLAS.Diag.NonUnit,
                n, n, 1, 0, TRSM_A_LOWER, 0, n, work, 0, n);
        double[] diff = new double[n * n * 2];
        for (int i = 0; i < n * n * 2; i++) diff[i] = work[i] - TRSM_X_LL_T[i];
        assertTrue(zmaxNorm(diff, n) / zmaxNorm(TRSM_X_LL_T, n) < TOL);
        verifyLeftResidual(BLAS.Trans.Trans, n, TRSM_A_LOWER, work, TRSM_B);
    }

    @Test
    void testLeftLowerConj() {
        int n = 4;
        double[] work = TRSM_B.clone();
        Ztrsm.ztrsm(BLAS.Side.Left, BLAS.Uplo.Lower, BLAS.Trans.Conj, BLAS.Diag.NonUnit,
                n, n, 1, 0, TRSM_A_LOWER, 0, n, work, 0, n);
        double[] diff = new double[n * n * 2];
        for (int i = 0; i < n * n * 2; i++) diff[i] = work[i] - TRSM_X_LL_C[i];
        assertTrue(zmaxNorm(diff, n) / zmaxNorm(TRSM_X_LL_C, n) < TOL);
        verifyLeftResidual(BLAS.Trans.Conj, n, TRSM_A_LOWER, work, TRSM_B);
    }

    @Test
    void testRightUpperNoTrans() {
        int n = 4;
        double[] work = TRSM_B.clone();
        Ztrsm.ztrsm(BLAS.Side.Right, BLAS.Uplo.Upper, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit,
                n, n, 1, 0, TRSM_A_UPPER, 0, n, work, 0, n);
        double[] diff = new double[n * n * 2];
        for (int i = 0; i < n * n * 2; i++) diff[i] = work[i] - TRSM_X_RU_N[i];
        assertTrue(zmaxNorm(diff, n) / zmaxNorm(TRSM_X_RU_N, n) < TOL);
        verifyRightResidual(BLAS.Trans.NoTrans, n, TRSM_A_UPPER, work, TRSM_B);
    }

    @Test
    void testRightUpperTrans() {
        int n = 4;
        double[] work = TRSM_B.clone();
        Ztrsm.ztrsm(BLAS.Side.Right, BLAS.Uplo.Upper, BLAS.Trans.Trans, BLAS.Diag.NonUnit,
                n, n, 1, 0, TRSM_A_UPPER, 0, n, work, 0, n);
        double[] diff = new double[n * n * 2];
        for (int i = 0; i < n * n * 2; i++) diff[i] = work[i] - TRSM_X_RU_T[i];
        assertTrue(zmaxNorm(diff, n) / zmaxNorm(TRSM_X_RU_T, n) < TOL);
        verifyRightResidual(BLAS.Trans.Trans, n, TRSM_A_UPPER, work, TRSM_B);
    }

    @Test
    void testRightUpperConj() {
        int n = 4;
        double[] work = TRSM_B.clone();
        Ztrsm.ztrsm(BLAS.Side.Right, BLAS.Uplo.Upper, BLAS.Trans.Conj, BLAS.Diag.NonUnit,
                n, n, 1, 0, TRSM_A_UPPER, 0, n, work, 0, n);
        double[] diff = new double[n * n * 2];
        for (int i = 0; i < n * n * 2; i++) diff[i] = work[i] - TRSM_X_RU_C[i];
        assertTrue(zmaxNorm(diff, n) / zmaxNorm(TRSM_X_RU_C, n) < TOL);
        verifyRightResidual(BLAS.Trans.Conj, n, TRSM_A_UPPER, work, TRSM_B);
    }

    @Test
    void testRightLowerNoTrans() {
        int n = 4;
        double[] work = TRSM_B.clone();
        Ztrsm.ztrsm(BLAS.Side.Right, BLAS.Uplo.Lower, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit,
                n, n, 1, 0, TRSM_A_LOWER, 0, n, work, 0, n);
        double[] diff = new double[n * n * 2];
        for (int i = 0; i < n * n * 2; i++) diff[i] = work[i] - TRSM_X_RL_N[i];
        assertTrue(zmaxNorm(diff, n) / zmaxNorm(TRSM_X_RL_N, n) < TOL);
        verifyRightResidual(BLAS.Trans.NoTrans, n, TRSM_A_LOWER, work, TRSM_B);
    }

    @Test
    void testRightLowerTrans() {
        int n = 4;
        double[] work = TRSM_B.clone();
        Ztrsm.ztrsm(BLAS.Side.Right, BLAS.Uplo.Lower, BLAS.Trans.Trans, BLAS.Diag.NonUnit,
                n, n, 1, 0, TRSM_A_LOWER, 0, n, work, 0, n);
        double[] diff = new double[n * n * 2];
        for (int i = 0; i < n * n * 2; i++) diff[i] = work[i] - TRSM_X_RL_T[i];
        assertTrue(zmaxNorm(diff, n) / zmaxNorm(TRSM_X_RL_T, n) < TOL);
        verifyRightResidual(BLAS.Trans.Trans, n, TRSM_A_LOWER, work, TRSM_B);
    }

    @Test
    void testRightLowerConj() {
        int n = 4;
        double[] work = TRSM_B.clone();
        Ztrsm.ztrsm(BLAS.Side.Right, BLAS.Uplo.Lower, BLAS.Trans.Conj, BLAS.Diag.NonUnit,
                n, n, 1, 0, TRSM_A_LOWER, 0, n, work, 0, n);
        double[] diff = new double[n * n * 2];
        for (int i = 0; i < n * n * 2; i++) diff[i] = work[i] - TRSM_X_RL_C[i];
        assertTrue(zmaxNorm(diff, n) / zmaxNorm(TRSM_X_RL_C, n) < TOL);
        verifyRightResidual(BLAS.Trans.Conj, n, TRSM_A_LOWER, work, TRSM_B);
    }

    @Test
    void testLeftUpperNoTransUnit() {
        int n = 4;
        double[] A = TRSM_A_UPPER.clone();
        for (int i = 0; i < n; i++) {
            A[i * n * 2 + i * 2] = 1.0;
            A[i * n * 2 + i * 2 + 1] = 0.0;
        }
        double[] work = TRSM_B.clone();
        Ztrsm.ztrsm(BLAS.Side.Left, BLAS.Uplo.Upper, BLAS.Trans.NoTrans, BLAS.Diag.Unit,
                n, n, 1, 0, A, 0, n, work, 0, n);
        verifyLeftResidual(BLAS.Trans.NoTrans, n, A, work, TRSM_B);
    }

    @Test
    void testLeftUpperTransUnit() {
        int n = 4;
        double[] A = TRSM_A_UPPER.clone();
        for (int i = 0; i < n; i++) {
            A[i * n * 2 + i * 2] = 1.0;
            A[i * n * 2 + i * 2 + 1] = 0.0;
        }
        double[] work = TRSM_B.clone();
        Ztrsm.ztrsm(BLAS.Side.Left, BLAS.Uplo.Upper, BLAS.Trans.Trans, BLAS.Diag.Unit,
                n, n, 1, 0, A, 0, n, work, 0, n);
        verifyLeftResidual(BLAS.Trans.Trans, n, A, work, TRSM_B);
    }

    @Test
    void testLeftUpperConjUnit() {
        int n = 4;
        double[] A = TRSM_A_UPPER.clone();
        for (int i = 0; i < n; i++) {
            A[i * n * 2 + i * 2] = 1.0;
            A[i * n * 2 + i * 2 + 1] = 0.0;
        }
        double[] work = TRSM_B.clone();
        Ztrsm.ztrsm(BLAS.Side.Left, BLAS.Uplo.Upper, BLAS.Trans.Conj, BLAS.Diag.Unit,
                n, n, 1, 0, A, 0, n, work, 0, n);
        verifyLeftResidual(BLAS.Trans.Conj, n, A, work, TRSM_B);
    }

    @Test
    void testLeftLowerNoTransUnit() {
        int n = 4;
        double[] A = TRSM_A_LOWER.clone();
        for (int i = 0; i < n; i++) {
            A[i * n * 2 + i * 2] = 1.0;
            A[i * n * 2 + i * 2 + 1] = 0.0;
        }
        double[] work = TRSM_B.clone();
        Ztrsm.ztrsm(BLAS.Side.Left, BLAS.Uplo.Lower, BLAS.Trans.NoTrans, BLAS.Diag.Unit,
                n, n, 1, 0, A, 0, n, work, 0, n);
        verifyLeftResidual(BLAS.Trans.NoTrans, n, A, work, TRSM_B);
    }

    @Test
    void testLeftLowerTransUnit() {
        int n = 4;
        double[] A = TRSM_A_LOWER.clone();
        for (int i = 0; i < n; i++) {
            A[i * n * 2 + i * 2] = 1.0;
            A[i * n * 2 + i * 2 + 1] = 0.0;
        }
        double[] work = TRSM_B.clone();
        Ztrsm.ztrsm(BLAS.Side.Left, BLAS.Uplo.Lower, BLAS.Trans.Trans, BLAS.Diag.Unit,
                n, n, 1, 0, A, 0, n, work, 0, n);
        verifyLeftResidual(BLAS.Trans.Trans, n, A, work, TRSM_B);
    }

    @Test
    void testLeftLowerConjUnit() {
        int n = 4;
        double[] A = TRSM_A_LOWER.clone();
        for (int i = 0; i < n; i++) {
            A[i * n * 2 + i * 2] = 1.0;
            A[i * n * 2 + i * 2 + 1] = 0.0;
        }
        double[] work = TRSM_B.clone();
        Ztrsm.ztrsm(BLAS.Side.Left, BLAS.Uplo.Lower, BLAS.Trans.Conj, BLAS.Diag.Unit,
                n, n, 1, 0, A, 0, n, work, 0, n);
        verifyLeftResidual(BLAS.Trans.Conj, n, A, work, TRSM_B);
    }

    @Test
    void testRightUpperNoTransUnit() {
        int n = 4;
        double[] A = TRSM_A_UPPER.clone();
        for (int i = 0; i < n; i++) {
            A[i * n * 2 + i * 2] = 1.0;
            A[i * n * 2 + i * 2 + 1] = 0.0;
        }
        double[] work = TRSM_B.clone();
        Ztrsm.ztrsm(BLAS.Side.Right, BLAS.Uplo.Upper, BLAS.Trans.NoTrans, BLAS.Diag.Unit,
                n, n, 1, 0, A, 0, n, work, 0, n);
        verifyRightResidual(BLAS.Trans.NoTrans, n, A, work, TRSM_B);
    }

    @Test
    void testRightUpperTransUnit() {
        int n = 4;
        double[] A = TRSM_A_UPPER.clone();
        for (int i = 0; i < n; i++) {
            A[i * n * 2 + i * 2] = 1.0;
            A[i * n * 2 + i * 2 + 1] = 0.0;
        }
        double[] work = TRSM_B.clone();
        Ztrsm.ztrsm(BLAS.Side.Right, BLAS.Uplo.Upper, BLAS.Trans.Trans, BLAS.Diag.Unit,
                n, n, 1, 0, A, 0, n, work, 0, n);
        verifyRightResidual(BLAS.Trans.Trans, n, A, work, TRSM_B);
    }

    @Test
    void testRightUpperConjUnit() {
        int n = 4;
        double[] A = TRSM_A_UPPER.clone();
        for (int i = 0; i < n; i++) {
            A[i * n * 2 + i * 2] = 1.0;
            A[i * n * 2 + i * 2 + 1] = 0.0;
        }
        double[] work = TRSM_B.clone();
        Ztrsm.ztrsm(BLAS.Side.Right, BLAS.Uplo.Upper, BLAS.Trans.Conj, BLAS.Diag.Unit,
                n, n, 1, 0, A, 0, n, work, 0, n);
        verifyRightResidual(BLAS.Trans.Conj, n, A, work, TRSM_B);
    }

    @Test
    void testRightLowerNoTransUnit() {
        int n = 4;
        double[] A = TRSM_A_LOWER.clone();
        for (int i = 0; i < n; i++) {
            A[i * n * 2 + i * 2] = 1.0;
            A[i * n * 2 + i * 2 + 1] = 0.0;
        }
        double[] work = TRSM_B.clone();
        Ztrsm.ztrsm(BLAS.Side.Right, BLAS.Uplo.Lower, BLAS.Trans.NoTrans, BLAS.Diag.Unit,
                n, n, 1, 0, A, 0, n, work, 0, n);
        verifyRightResidual(BLAS.Trans.NoTrans, n, A, work, TRSM_B);
    }

    @Test
    void testRightLowerTransUnit() {
        int n = 4;
        double[] A = TRSM_A_LOWER.clone();
        for (int i = 0; i < n; i++) {
            A[i * n * 2 + i * 2] = 1.0;
            A[i * n * 2 + i * 2 + 1] = 0.0;
        }
        double[] work = TRSM_B.clone();
        Ztrsm.ztrsm(BLAS.Side.Right, BLAS.Uplo.Lower, BLAS.Trans.Trans, BLAS.Diag.Unit,
                n, n, 1, 0, A, 0, n, work, 0, n);
        verifyRightResidual(BLAS.Trans.Trans, n, A, work, TRSM_B);
    }

    @Test
    void testRightLowerConjUnit() {
        int n = 4;
        double[] A = TRSM_A_LOWER.clone();
        for (int i = 0; i < n; i++) {
            A[i * n * 2 + i * 2] = 1.0;
            A[i * n * 2 + i * 2 + 1] = 0.0;
        }
        double[] work = TRSM_B.clone();
        Ztrsm.ztrsm(BLAS.Side.Right, BLAS.Uplo.Lower, BLAS.Trans.Conj, BLAS.Diag.Unit,
                n, n, 1, 0, A, 0, n, work, 0, n);
        verifyRightResidual(BLAS.Trans.Conj, n, A, work, TRSM_B);
    }
}
