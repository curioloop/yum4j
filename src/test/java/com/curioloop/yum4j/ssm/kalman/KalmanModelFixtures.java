package com.curioloop.yum4j.ssm.kalman;

import com.curioloop.yum4j.fixtures.StatsmodelsResources;
import com.curioloop.yum4j.ssm.kalman.model.ModelFixture;

public final class KalmanModelFixtures {

    private static final double[] DEFAULT_AR1_Y = {1.0, 0.5, -0.3, 0.8, 0.2};
    private static final double[] DEFAULT_LOCAL_LEVEL_Y = {1.0, 2.0, 1.5, 3.0, 2.5};
    private static final double[] DEFAULT_BIVARIATE_Y = {1.0, 0.5, -0.3, 0.8, 0.2, -0.1};

    private KalmanModelFixtures() {
    }

    public static ModelFixture defaultScalarAr1() {
        return scalarAr1(0.5, 0.5, 1.0, 1.0, DEFAULT_AR1_Y);
    }

    public static ModelFixture scalarAr1(double design, double transition, double obsVariance,
                                  double stateVariance, double[] y) {
        int nobs = y.length;
        ModelFixture rep = new ModelFixture(1, 1, 1, nobs);
        double[] z = {design};
        double[] d = {0.0};
        double[] h = {obsVariance};
        double[] tmat = {transition};
        double[] c = {0.0};
        double[] r = {1.0};
        double[] q = {stateVariance};
        for (int t = 0; t < nobs; t++) {
            rep.setDesign(z, t);
            rep.setObsIntercept(d, t);
            rep.setObsCov(h, t);
            rep.setTransition(tmat, t);
            rep.setStateIntercept(c, t);
            rep.setSelection(r, t);
            rep.setStateCov(q, t);
            rep.setEndog(new double[]{y[t]}, t);
        }
        return rep;
    }

    public static ModelFixture defaultLocalLevel() {
        return localLevel(1.0, 0.5, DEFAULT_LOCAL_LEVEL_Y);
    }

    public static ModelFixture localLevel(double obsVariance, double stateVariance, double[] y) {
        return scalarAr1(1.0, 1.0, obsVariance, stateVariance, y);
    }

    public static ModelFixture defaultBivariate2State() {
        return bivariate2State(DEFAULT_BIVARIATE_Y);
    }

    public static ModelFixture bivariate2State(double[] y) {
        int nobs = y.length / 2;
        ModelFixture rep = new ModelFixture(2, 2, 1, nobs);
        double[] z = {1.0, 0.0, 0.0, 1.0};
        double[] d = {0.0, 0.0};
        double[] h = {1.0, 0.0, 0.0, 1.0};
        double[] tmat = {0.9, 0.1, 0.0, 0.9};
        double[] c = {0.0, 0.0};
        double[] r = {1.0, 0.0};
        double[] q = {1.0};
        for (int t = 0; t < nobs; t++) {
            rep.setDesign(z, t);
            rep.setObsIntercept(d, t);
            rep.setObsCov(h, t);
            rep.setTransition(tmat, t);
            rep.setStateIntercept(c, t);
            rep.setSelection(r, t);
            rep.setStateCov(q, t);
            rep.setEndog(new double[]{y[t * 2], y[t * 2 + 1]}, t);
        }
        return rep;
    }

    public static ModelFixture statsmodelsAr1(double phi, double sigma2, double[] y) {
        return scalarAr1(1.0, phi, sigma2, 1.0, y);
    }

    public static ModelFixture statsmodelsAr3(double phi1, double phi2, double phi3,
                                       double sigma2Obs, double sigma2State,
                                       double[] y) {
        int nobs = y.length;
        ModelFixture rep = new ModelFixture(1, 3, 1, nobs);
        double[] z = {1.0, 0.0, 0.0};
        double[] h = {sigma2Obs};
        double[] tmat = {phi1, phi2, phi3, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0};
        double[] r = {1.0, 0.0, 0.0};
        double[] q = {sigma2State};
        for (int t = 0; t < nobs; t++) {
            rep.setDesign(z, t);
            rep.setObsCov(h, t);
            rep.setTransition(tmat, t);
            rep.setSelection(r, t);
            rep.setStateCov(q, t);
            rep.setEndog(new double[]{y[t]}, t);
        }
        return rep;
    }

    public static ModelFixture statsmodelsBivariateAr1(double[] y) {
        int nobs = y.length / 2;
        ModelFixture rep = new ModelFixture(2, 4, 2, nobs);
        double[] z = {1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0};
        double[] h = {0.1, 0.0, 0.0, 0.1};
        double[] tmat = {0.9, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.7, 0.0, 0.0, 0.0, 0.0, 0.6};
        double[] r = {1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0};
        double[] q = {1.0, 0.0, 0.0, 1.0};
        for (int t = 0; t < nobs; t++) {
            rep.setDesign(z, t);
            rep.setObsCov(h, t);
            rep.setTransition(tmat, t);
            rep.setSelection(r, t);
            rep.setStateCov(q, t);
            rep.setEndog(new double[]{y[t * 2], y[t * 2 + 1]}, t);
        }
        return rep;
    }

    public static ModelFixture statsmodelsMultivariateMissingModel() {
        double[][] raw = StatsmodelsResources.readMacrodata();
        int nobs = 202;
        int kEndog = 3;
        int kStates = 3;
        int kPosdef = 3;

        double[] realgdp = new double[nobs];
        double[] realcons = new double[nobs];
        double[] realinv = new double[nobs];
        for (int t = 0; t < nobs; t++) {
            realgdp[t] = raw[t + 1][2] - raw[t][2];
            realcons[t] = raw[t + 1][3] - raw[t][3];
            realinv[t] = raw[t + 1][4] - raw[t][4];
        }

        ModelFixture rep = new ModelFixture(kEndog, kStates, kPosdef, nobs);
        double[] z = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
        double[] h = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
        double[] tmat = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
        double[] r = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
        double[] q = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
        for (int t = 0; t < nobs; t++) {
            rep.setDesign(z, t);
            rep.setObsCov(h, t);
            rep.setTransition(tmat, t);
            rep.setSelection(r, t);
            rep.setStateCov(q, t);
            rep.setEndog(new double[]{realgdp[t], realcons[t], realinv[t]}, t);
        }

        boolean[][] missing = new boolean[nobs][kEndog];
        for (int t = 0; t < 50; t++) {
            missing[t][0] = true;
        }
        for (int t = 19; t < 70; t++) {
            missing[t][1] = true;
        }
        for (int t = 39; t < 90; t++) {
            missing[t][2] = true;
        }
        for (int t = 119; t < Math.min(130, nobs); t++) {
            missing[t][0] = true;
            missing[t][2] = true;
        }
        for (int t = 0; t < nobs; t++) {
            rep.setMissing(missing[t], t);
        }
        return rep;
    }

    public static ModelFixture signalOnlyAr1(double phi, double sigma, double[] y) {
        return scalarAr1(1.0, phi, 0.0, sigma * sigma, y);
    }

    public static ModelFixture complexSignalOnlyAr1(double phiRe, double phiIm, double sigma, double[] y) {
        int nobs = y.length;
        ModelFixture rep = ModelFixture.complex(1, 1, 1, nobs);
        double[] z = {1.0, 0.0};
        double[] d = {0.0, 0.0};
        double[] h = {0.0, 0.0};
        double[] tmat = {phiRe, phiIm};
        double[] c = {0.0, 0.0};
        double[] r = {1.0, 0.0};
        double[] q = {sigma * sigma, 0.0};
        for (int t = 0; t < nobs; t++) {
            rep.setDesign(z, t);
            rep.setObsIntercept(d, t);
            rep.setObsCov(h, t);
            rep.setTransition(tmat, t);
            rep.setStateIntercept(c, t);
            rep.setSelection(r, t);
            rep.setStateCov(q, t);
            rep.setEndog(new double[]{y[t], 0.0}, t);
        }
        return rep;
    }
}