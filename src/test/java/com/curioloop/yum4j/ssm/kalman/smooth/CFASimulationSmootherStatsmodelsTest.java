package com.curioloop.yum4j.ssm.kalman.smooth;

import com.curioloop.yum4j.ssm.kalman.init.InitialState;
import com.curioloop.yum4j.ssm.kalman.model.ModelFixture;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class CFASimulationSmootherStatsmodelsTest {

    private static final double TOL = 1e-8;

    @Test
    void dynamicFactorPosteriorMomentsMatchSmootherForAllMissingPrefix() {
        assertPosteriorMomentsMatchSmoother(dynamicFactorModel(MissingPattern.ALL));
    }

    @Test
    void dynamicFactorPosteriorMomentsMatchSmootherForPartialMissingPrefix() {
        assertPosteriorMomentsMatchSmoother(dynamicFactorModel(MissingPattern.PARTIAL));
    }

    @Test
    void dynamicFactorPosteriorMomentsMatchSmootherForMixedMissingBlocks() {
        assertPosteriorMomentsMatchSmoother(dynamicFactorModel(MissingPattern.MIXED));
    }

    @Test
    void varMeasurementErrorPosteriorMomentsMatchSmootherForMixedMissingBlocks() {
        assertPosteriorMomentsMatchSmoother(varMeasurementErrorModel(MissingPattern.MIXED));
    }

    @Test
    void sarimaxMeasurementErrorPosteriorMomentsMatchSmootherForMixedMissingBlocks() {
        assertPosteriorMomentsMatchSmoother(sarimaxMeasurementErrorModel(MissingPattern.MIXED));
    }

    @Test
    void unobservedComponentsPosteriorMomentsMatchSmootherForMixedMissingBlocks() {
        assertPosteriorMomentsMatchSmoother(unobservedComponentsModel(MissingPattern.MIXED));
    }

    @Test
    void tvpVarPosteriorMomentsMatchSmootherForPartialMissingBlocks() {
        assertPosteriorMomentsMatchSmoother(tvpVarLikeModel(MissingPattern.PARTIAL));
    }

    @Test
    void cfaZeroVariatesReturnPosteriorMeanForDfmModel() {
        ModelFixture model = dynamicFactorModel(MissingPattern.MIXED);
        InitialState initialState = knownInitial(model.kStates);
        CFASimulationSmoother.PosteriorMoments moments = CFASimulationSmoother.posteriorMoments(model, initialState);
        SimulationSmootherResult result = SimulationSmootherEngine.simulate(model, initialState,
            null, new double[model.nobs * model.kStates], null, SimulationSmootherOptions.stateOnly()
                .withMethod(SimulationSmootherOptions.Method.CFA));

        for (int t = 0; t < model.nobs; t++) {
            for (int i = 0; i < model.kStates; i++) {
                assertEquals(moments.mean(i, t), result.simulatedState(i, t), TOL,
                    "zero-variance CFA draw t=" + t + " i=" + i);
            }
        }
    }

    @Test
    void cfaZeroVariatesReturnPosteriorMeanForTvpVarLikeModel() {
        ModelFixture model = tvpVarLikeModel(MissingPattern.MIXED);
        InitialState initialState = knownInitial(model.kStates);
        CFASimulationSmoother.PosteriorMoments moments = CFASimulationSmoother.posteriorMoments(model, initialState);
        SimulationSmootherResult result = SimulationSmootherEngine.simulate(model, initialState,
            null, new double[model.nobs * model.kStates], null, SimulationSmootherOptions.stateOnly()
                .withMethod(SimulationSmootherOptions.Method.CFA));

        for (int t = 0; t < model.nobs; t++) {
            for (int i = 0; i < model.kStates; i++) {
                assertEquals(moments.mean(i, t), result.simulatedState(i, t), TOL,
                    "zero-variance TVP-VAR-like CFA draw t=" + t + " i=" + i);
            }
        }
    }

    private static void assertPosteriorMomentsMatchSmoother(ModelFixture model) {
        InitialState initialState = knownInitial(model.kStates);
        CFASimulationSmoother.PosteriorMoments moments = CFASimulationSmoother.posteriorMoments(model, initialState);
        SmootherResult smoother = SmootherEngine.smooth(model, initialState, SmootherOptions.conventional());

        assertEquals(model.kStates, moments.kStates());
        assertEquals(model.nobs, moments.nobs());
        for (int t = 0; t < model.nobs; t++) {
            for (int i = 0; i < model.kStates; i++) {
                assertEquals(smoother.smoothedState(i, t), moments.mean(i, t), TOL,
                    "posterior mean t=" + t + " i=" + i);
                for (int j = 0; j < model.kStates; j++) {
                    assertEquals(smoother.smoothedStateCov(i, j, t), moments.covariance(i, j, t), TOL,
                        "posterior covariance t=" + t + " i=" + i + " j=" + j);
                }
            }
        }
    }

    private static ModelFixture dynamicFactorModel(MissingPattern missingPattern) {
        int nobs = 18;
        ModelFixture model = new ModelFixture(3, 1, 1, nobs);
        double[] design = {0.5, 1.0, -0.7};
        double[] obsCov = {0.8, 0.0, 0.0, 0.0, 1.1, 0.0, 0.0, 0.0, 0.9};
        for (int t = 0; t < nobs; t++) {
            model.setDesign(design, t);
            model.setObsCov(obsCov, t);
            model.setTransition(new double[]{0.72}, t);
            model.setSelection(new double[]{1.0}, t);
            model.setStateCov(new double[]{0.65}, t);
            model.setEndog(new double[]{
                Math.sin(0.13 * t) + 0.04 * t,
                Math.cos(0.11 * t) - 0.02 * t,
                0.5 * Math.sin(0.19 * t) + 0.03 * t
            }, t);
            applyMissing(model, t, missingPattern);
        }
        return model;
    }

    private static ModelFixture varMeasurementErrorModel(MissingPattern missingPattern) {
        int nobs = 20;
        ModelFixture model = new ModelFixture(2, 2, 2, nobs);
        double[] design = {1.0, 0.0, 0.0, 1.0};
        double[] obsCov = {0.45, 0.08, 0.08, 0.55};
        double[] transition = {0.58, 0.14, -0.10, 0.48};
        double[] selection = {1.0, 0.0, 0.0, 1.0};
        double[] stateCov = {0.75, 0.10, 0.10, 0.70};
        for (int t = 0; t < nobs; t++) {
            model.setDesign(design, t);
            model.setObsCov(obsCov, t);
            model.setTransition(transition, t);
            model.setSelection(selection, t);
            model.setStateCov(stateCov, t);
            model.setEndog(new double[]{Math.sin(0.09 * t) + 0.02 * t, Math.cos(0.07 * t) - 0.03 * t}, t);
            applyMissing(model, t, missingPattern);
        }
        return model;
    }

    private static ModelFixture sarimaxMeasurementErrorModel(MissingPattern missingPattern) {
        int nobs = 22;
        ModelFixture model = new ModelFixture(1, 1, 1, nobs);
        for (int t = 0; t < nobs; t++) {
            model.setDesign(new double[]{1.0}, t);
            model.setObsCov(new double[]{0.35}, t);
            model.setTransition(new double[]{0.64}, t);
            model.setSelection(new double[]{1.0}, t);
            model.setStateCov(new double[]{0.42}, t);
            model.setEndog(new double[]{0.2 + Math.sin(0.17 * t) + 0.03 * t}, t);
            applyMissing(model, t, missingPattern);
        }
        return model;
    }

    private static ModelFixture unobservedComponentsModel(MissingPattern missingPattern) {
        int nobs = 24;
        ModelFixture model = new ModelFixture(1, 2, 2, nobs);
        double[] transition = {1.0, 1.0, 0.0, 1.0};
        double[] selection = {1.0, 0.0, 0.0, 1.0};
        double[] stateCov = {0.20, 0.03, 0.03, 0.08};
        for (int t = 0; t < nobs; t++) {
            model.setDesign(new double[]{1.0, 0.0}, t);
            model.setObsCov(new double[]{0.55}, t);
            model.setTransition(transition, t);
            model.setSelection(selection, t);
            model.setStateCov(stateCov, t);
            model.setEndog(new double[]{1.0 + 0.08 * t + 0.4 * Math.sin(0.12 * t)}, t);
            applyMissing(model, t, missingPattern);
        }
        return model;
    }

    private static ModelFixture tvpVarLikeModel(MissingPattern missingPattern) {
        int nobs = 18;
        ModelFixture model = new ModelFixture(2, 4, 4, nobs);
        double[] transition = {
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0
        };
        double[] selection = transition;
        double[] stateCov = {
            0.08, 0.01, 0.00, 0.00,
            0.01, 0.06, 0.00, 0.00,
            0.00, 0.00, 0.07, 0.01,
            0.00, 0.00, 0.01, 0.05
        };
        double[] obsCov = {0.45, 0.06, 0.06, 0.50};
        for (int t = 0; t < nobs; t++) {
            double laggedFirst = 1.0 + 0.2 * Math.sin(0.19 * t);
            double laggedSecond = -0.4 + 0.1 * Math.cos(0.23 * t);
            model.setDesign(new double[]{
                laggedFirst, laggedSecond, 0.0, 0.0,
                0.0, 0.0, laggedFirst, laggedSecond
            }, t);
            model.setObsCov(obsCov, t);
            model.setTransition(transition, t);
            model.setSelection(selection, t);
            model.setStateCov(stateCov, t);
            model.setEndog(new double[]{
                0.6 * Math.sin(0.11 * t) + 0.04 * t,
                0.5 * Math.cos(0.13 * t) - 0.02 * t
            }, t);
            applyMissing(model, t, missingPattern);
        }
        return model;
    }

    private static void applyMissing(ModelFixture model, int t, MissingPattern missingPattern) {
        switch (missingPattern) {
            case NONE -> { }
            case ALL -> {
                if (t < 5) {
                    boolean[] mask = new boolean[model.kEndog];
                    for (int i = 0; i < model.kEndog; i++) {
                        mask[i] = true;
                    }
                    model.setMissing(mask, t);
                }
            }
            case PARTIAL -> {
                if (t < 6) {
                    boolean[] mask = new boolean[model.kEndog];
                    mask[0] = true;
                    model.setMissing(mask, t);
                }
            }
            case MIXED -> {
                boolean[] mask = new boolean[model.kEndog];
                mask[0] = t < 6 || (t >= 12 && t < 15);
                if (model.kEndog > 1) {
                    mask[1] = t >= 4 && t < 10;
                }
                if (model.kEndog > 2) {
                    mask[2] = (t >= 8 && t < 13) || t >= model.nobs - 3;
                }
                model.setMissing(mask, t);
            }
        }
    }

    private static InitialState knownInitial(int kStates) {
        double[] state = new double[kStates];
        double[] covariance = new double[kStates * kStates];
        for (int i = 0; i < kStates; i++) {
            state[i] = 0.1 * (i + 1);
            covariance[i * kStates + i] = 1.0 + 0.25 * i;
        }
        return InitialState.known(state, covariance);
    }

    private enum MissingPattern {
        NONE,
        ALL,
        PARTIAL,
        MIXED
    }
}