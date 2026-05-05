package com.curioloop.yum4j.kalman;

import com.curioloop.yum4j.kalman.init.*;
import com.curioloop.yum4j.kalman.model.*;
import com.curioloop.yum4j.linalg.blas.BLAS;
import com.curioloop.yum4j.linalg.blas.cmplx.ZLAS;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class ZStationaryInitTest {

    private static final double TOL = 1e-10;

        private static StateInitialization realStationaryInitialization(double[] transition, int kStates,
                                                                                                                                        double[] selection, int kPosdef,
                                                                                                                                        double[] stateCov, double[] stateIntercept) {
                ModelFixture rep = new ModelFixture(1, kStates, kPosdef, 1);
                rep.setTransition(transition, 0);
                rep.setSelection(selection, 0);
                rep.setStateCov(stateCov, 0);
                rep.setStateIntercept(stateIntercept == null ? new double[kStates] : stateIntercept, 0);
                return StateInitialization.stationary(rep);
        }

        private static StateInitialization complexStationaryInitialization(double[] transition, int kStates,
                                                                                                                                           double[] selection, int kPosdef,
                                                                                                                                           double[] stateCov, double[] stateIntercept) {
                ModelFixture rep = ModelFixture.complex(1, kStates, kPosdef, 1);
                rep.setTransition(transition, 0);
                rep.setSelection(selection, 0);
                rep.setStateCov(stateCov, 0);
                rep.setStateIntercept(stateIntercept == null ? new double[2 * kStates] : stateIntercept, 0);
                return StateInitialization.stationary(rep);
        }

    @Test
    void testRealDegradation() {
        int kStates = 1, kPosdef = 1;
        double phi = 0.5;

        double[] rT = {phi};
        double[] rR = {1.0};
        double[] rQ = {1.0};
        double[] rc = {0.0};

        StateInitialization realResult = realStationaryInitialization(rT, kStates, rR, kPosdef, rQ, rc);
        assertNotNull(realResult);

        double[] zT = {phi, 0.0};
        double[] zR = {1.0, 0.0};
        double[] zQ = {1.0, 0.0};
        double[] zc = {0.0, 0.0};

        StateInitialization zResult = complexStationaryInitialization(zT, kStates, zR, kPosdef, zQ, zc);
        assertNotNull(zResult);

        assertEquals(realResult.initialState()[0], zResult.initialState()[0], TOL);
        assertEquals(0.0, zResult.initialState()[1], TOL);

        assertEquals(realResult.initialStateCov()[0], zResult.initialStateCov()[0], TOL);
        assertEquals(0.0, zResult.initialStateCov()[1], TOL);
    }

    @Test
    void testComplexUnivariate() {
        int kStates = 2, kPosdef = 2;

        double[] T = {
                0.5, 0.1, 0.0, 0.0,
                0.0, 0.0, 0.3, 0.05
        };
        double[] R = {
                1.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0
        };
        double[] Q = {
                10.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 5.0, 0.0
        };

        StateInitialization result = complexStationaryInitialization(T, kStates, R, kPosdef, Q, null);
        assertNotNull(result);

        double[] P = result.initialStateCov();

        double[] TPT = new double[kStates * 2 * kStates];
        double[] TP = new double[kStates * 2 * kStates];
        ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                1.0, 0.0, T, 0, kStates, P, 0, kStates,
                0.0, 0.0, TP, 0, kStates);
        ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.Conj, kStates, kStates, kStates,
                1.0, 0.0, TP, 0, kStates, T, 0, kStates,
                0.0, 0.0, TPT, 0, kStates);

        double[] RQ = new double[kStates * 2 * kPosdef];
        double[] RQR = new double[kStates * 2 * kStates];
        ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kPosdef, kPosdef,
                1.0, 0.0, R, 0, kPosdef, Q, 0, kPosdef,
                0.0, 0.0, RQ, 0, kPosdef);
        ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.Conj, kStates, kStates, kPosdef,
                1.0, 0.0, RQ, 0, kPosdef, R, 0, kPosdef,
                0.0, 0.0, RQR, 0, kStates);

        for (int i = 0; i < kStates * 2 * kStates; i++) {
            double residual = P[i] - TPT[i] - RQR[i];
            assertEquals(0.0, residual, TOL);
        }
    }

    @Test
    void testComplexMultivariate() {
        int kStates = 2, kPosdef = 2;

        double[] T = {
                0.4, -0.1, -0.5, 0.0,
                1.0, 0.0, 0.0, 0.0
        };
        double[] R = {
                1.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0
        };
        double[] Q = {
                10.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 5.0, 0.0
        };

        StateInitialization result = complexStationaryInitialization(T, kStates, R, kPosdef, Q, null);
        assertNotNull(result);

        double[] P = result.initialStateCov();

        double[] TPT = new double[kStates * 2 * kStates];
        double[] TP = new double[kStates * 2 * kStates];
        ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kStates, kStates,
                1.0, 0.0, T, 0, kStates, P, 0, kStates,
                0.0, 0.0, TP, 0, kStates);
        ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.Conj, kStates, kStates, kStates,
                1.0, 0.0, TP, 0, kStates, T, 0, kStates,
                0.0, 0.0, TPT, 0, kStates);

        double[] RQ = new double[kStates * 2 * kPosdef];
        double[] RQR = new double[kStates * 2 * kStates];
        ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kStates, kPosdef, kPosdef,
                1.0, 0.0, R, 0, kPosdef, Q, 0, kPosdef,
                0.0, 0.0, RQ, 0, kPosdef);
        ZLAS.zgemm(BLAS.Trans.NoTrans, BLAS.Trans.Conj, kStates, kStates, kPosdef,
                1.0, 0.0, RQ, 0, kPosdef, R, 0, kPosdef,
                0.0, 0.0, RQR, 0, kStates);

        for (int i = 0; i < kStates * 2 * kStates; i++) {
            double residual = P[i] - TPT[i] - RQR[i];
            assertEquals(0.0, residual, TOL);
        }

                StateInitialization repeated = complexStationaryInitialization(T, kStates, R, kPosdef, Q, null);
                for (int i = 0; i < result.initialState().length; i++) {
                        assertEquals(result.initialState()[i], repeated.initialState()[i], TOL);
                }
                for (int i = 0; i < P.length; i++) {
                        assertEquals(P[i], repeated.initialStateCov()[i], TOL);
                }
    }

    @Test
    void testStationaryInitializationRejectsUnitModulus() {
        assertThrows(IllegalArgumentException.class,
                () -> realStationaryInitialization(
                        new double[]{1.0}, 1,
                        new double[]{1.0}, 1,
                        new double[]{1.0}, new double[]{0.0}));

        assertThrows(IllegalArgumentException.class,
                () -> complexStationaryInitialization(
                        new double[]{0.0, 1.0}, 1,
                        new double[]{1.0, 0.0}, 1,
                        new double[]{1.0, 0.0}, new double[]{0.0, 0.0}));
    }
}
