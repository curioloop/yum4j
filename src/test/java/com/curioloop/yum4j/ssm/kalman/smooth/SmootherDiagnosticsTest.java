package com.curioloop.yum4j.ssm.kalman.smooth;

import com.curioloop.yum4j.ssm.kalman.init.InitialState;
import com.curioloop.yum4j.ssm.kalman.model.KalmanSSM;
import com.curioloop.yum4j.ssm.kalman.model.ModelFixture;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class SmootherDiagnosticsTest {

    private static final double TOL = 1e-10;

    @Test
    void smoothedStateWeightsMatchManualObservationAndInterceptPerturbations() {
        ModelFixture model = diagnosticsModel();
        InitialState initialState = knownInitial();
        SmootherOptions options = SmootherOptions.conventional();
        SmootherResult baseline = SmootherEngine.smooth(model, initialState, options);

        SmootherDiagnostics.SmoothedStateWeights weights =
            SmootherDiagnostics.computeSmoothedStateWeights(model, initialState, options);

        KalmanSSM observationPerturbation = KalmanSSM.copyOf(model);
        observationPerturbation.endogData()[observationPerturbation.endogOffset(2) + 1] += 1.0;
        SmootherResult observationShifted = SmootherEngine.smooth(observationPerturbation, initialState, options);
        for (int targetTime = 0; targetTime < model.nobs; targetTime++) {
            for (int stateIndex = 0; stateIndex < model.kStates; stateIndex++) {
                assertEquals(observationShifted.smoothedState(stateIndex, targetTime)
                        - baseline.smoothedState(stateIndex, targetTime),
                    weights.observationWeight(targetTime, 2, stateIndex, 1), TOL);
                assertTrue(Double.isNaN(weights.observationWeight(targetTime, 1, stateIndex, 0)));
            }
        }

        KalmanSSM interceptPerturbation = KalmanSSM.copyOf(model);
        interceptPerturbation.stateInterceptData()[interceptPerturbation.stateInterceptOffset(3)] += 1.0;
        SmootherResult interceptShifted = SmootherEngine.smooth(interceptPerturbation, initialState, options);
        for (int targetTime = 0; targetTime < model.nobs; targetTime++) {
            for (int stateIndex = 0; stateIndex < model.kStates; stateIndex++) {
                assertEquals(interceptShifted.smoothedState(stateIndex, targetTime)
                        - baseline.smoothedState(stateIndex, targetTime),
                    weights.stateInterceptWeight(targetTime, 3, stateIndex, 0), TOL);
            }
        }

        InitialState shiftedInitial = InitialState.known(new double[]{0.3, 0.6}, new double[]{1.4, 0.2, 0.2, 1.8});
        SmootherResult priorShifted = SmootherEngine.smooth(model, shiftedInitial, options);
        for (int targetTime = 0; targetTime < model.nobs; targetTime++) {
            for (int stateIndex = 0; stateIndex < model.kStates; stateIndex++) {
                assertEquals(priorShifted.smoothedState(stateIndex, targetTime)
                        - baseline.smoothedState(stateIndex, targetTime),
                    weights.priorWeight(targetTime, stateIndex, 1), TOL);
            }
        }
    }

    @Test
    void smoothedDecompositionReconstructsStateSignalAndForecast() {
        ModelFixture model = diagnosticsModel();
        InitialState initialState = knownInitial();
        SmootherResult smoother = SmootherEngine.smooth(model, initialState, SmootherOptions.conventional());

        SmootherDiagnostics.SmoothedDecomposition decomposition =
            SmootherDiagnostics.getSmoothedDecomposition(model, initialState, SmootherOptions.conventional());

        assertArrayEquals(flattenSmoothedState(smoother), decomposition.reconstructedSmoothedState(), TOL);
        assertArrayEquals(smoothedSignal(model, smoother, false), decomposition.reconstructedSmoothedSignal(model), TOL);
        assertArrayEquals(smoothedSignal(model, smoother, true), decomposition.reconstructedSmoothedForecast(model), TOL);
    }

    @Test
    void smoothedDiagnosticsRejectDiffuseInitialization() {
        assertThrows(UnsupportedOperationException.class,
            () -> SmootherDiagnostics.computeSmoothedStateWeights(
                diagnosticsModel(), InitialState.diffuse(2), SmootherOptions.conventional()));
    }

    @Test
    void newsImpactSeparatesNewObservationImpactWhenUpdatedSampleExtendsPrevious() {
        ModelFixture previousModel = diagnosticsModel(5);
        ModelFixture updatedModel = diagnosticsModel(7);
        InitialState initialState = knownInitial();
        SmootherResult previous = SmootherEngine.smooth(previousModel, initialState, SmootherOptions.conventional());
        SmootherResult updated = SmootherEngine.smooth(updatedModel, initialState, SmootherOptions.conventional());

        SmootherDiagnostics.NewsImpact news = SmootherDiagnostics.news(
            previousModel, initialState, updatedModel, initialState, SmootherOptions.conventional());

        assertEquals(previousModel.nobs, news.targetCount());
        for (int targetTime = 0; targetTime < previousModel.nobs; targetTime++) {
            for (int stateIndex = 0; stateIndex < previousModel.kStates; stateIndex++) {
                double expected = updated.smoothedState(stateIndex, targetTime) - previous.smoothedState(stateIndex, targetTime);
                assertEquals(expected, news.smoothedStateImpact(stateIndex, targetTime), TOL);
                assertEquals(0.0, news.revisionImpact(stateIndex, targetTime), TOL);
                assertEquals(expected, news.newObservationImpact(stateIndex, targetTime), TOL);
            }
            for (int observationIndex = 0; observationIndex < previousModel.kEndog; observationIndex++) {
                double expectedSignal = 0.0;
                int designOffset = updatedModel.designOffset(targetTime);
                for (int stateIndex = 0; stateIndex < previousModel.kStates; stateIndex++) {
                    expectedSignal += updatedModel.designData()[designOffset + observationIndex * previousModel.kStates + stateIndex]
                        * news.smoothedStateImpact(stateIndex, targetTime);
                }
                assertEquals(expectedSignal, news.smoothedSignalImpact(observationIndex, targetTime), TOL);
            }
        }
    }

    @Test
    void newsImpactSeparatesHistoricalRevisionImpact() {
        ModelFixture previousModel = diagnosticsModel(5);
        KalmanSSM updatedModel = KalmanSSM.copyOf(previousModel);
        updatedModel.endogData()[updatedModel.endogOffset(2) + 1] += 0.7;
        InitialState initialState = knownInitial();
        SmootherResult previous = SmootherEngine.smooth(previousModel, initialState, SmootherOptions.conventional());
        SmootherResult updated = SmootherEngine.smooth(updatedModel, initialState, SmootherOptions.conventional());

        SmootherDiagnostics.NewsImpact news = SmootherDiagnostics.news(
            previousModel, initialState, updatedModel, initialState, SmootherOptions.conventional());

        for (int targetTime = 0; targetTime < previousModel.nobs; targetTime++) {
            for (int stateIndex = 0; stateIndex < previousModel.kStates; stateIndex++) {
                double expected = updated.smoothedState(stateIndex, targetTime) - previous.smoothedState(stateIndex, targetTime);
                assertEquals(expected, news.smoothedStateImpact(stateIndex, targetTime), TOL);
                assertEquals(expected, news.revisionImpact(stateIndex, targetTime), TOL);
                assertEquals(0.0, news.newObservationImpact(stateIndex, targetTime), TOL);
            }
        }
    }

    @Test
    void newsImpactRejectsInvalidModelPairs() {
        InitialState initialState = knownInitial();
        assertThrows(IllegalArgumentException.class,
            () -> SmootherDiagnostics.news(diagnosticsModel(5), initialState,
                diagnosticsModel(4), initialState, SmootherOptions.conventional()));
        assertThrows(IllegalArgumentException.class,
            () -> SmootherDiagnostics.news(diagnosticsModel(5), initialState,
                new ModelFixture(1, 2, 2, 5), initialState, SmootherOptions.conventional()));
    }

    private static ModelFixture diagnosticsModel() {
        return diagnosticsModel(5);
    }

    private static ModelFixture diagnosticsModel(int nobs) {
        ModelFixture model = new ModelFixture(2, 2, 2, nobs);
        double[] transition = {0.7, 0.1, -0.2, 0.6};
        double[] selection = {1.0, 0.0, 0.0, 1.0};
        double[] stateCov = {0.4, 0.05, 0.05, 0.35};
        double[] obsCov = {0.8, 0.1, 0.1, 0.7};
        for (int timeIndex = 0; timeIndex < model.nobs; timeIndex++) {
            model.setDesign(new double[]{1.0, 0.2, -0.1, 0.9}, timeIndex);
            model.setObsIntercept(new double[]{0.1 + 0.02 * timeIndex, -0.2 + 0.01 * timeIndex}, timeIndex);
            model.setObsCov(obsCov, timeIndex);
            model.setTransition(transition, timeIndex);
            model.setStateIntercept(new double[]{0.05, -0.03}, timeIndex);
            model.setSelection(selection, timeIndex);
            model.setStateCov(stateCov, timeIndex);
            model.setEndog(new double[]{1.0 + 0.2 * timeIndex, -0.4 + 0.15 * timeIndex}, timeIndex);
        }
        model.setMissing(new boolean[]{true, false}, 1);
        if (nobs > 4) {
            model.setMissing(new boolean[]{false, true}, 4);
        }
        return model;
    }

    private static InitialState knownInitial() {
        return InitialState.known(new double[]{0.3, -0.4}, new double[]{1.4, 0.2, 0.2, 1.8});
    }

    private static double[] flattenSmoothedState(SmootherResult smoother) {
        double[] values = new double[smoother.nobs * smoother.kStates];
        for (int timeIndex = 0; timeIndex < smoother.nobs; timeIndex++) {
            for (int stateIndex = 0; stateIndex < smoother.kStates; stateIndex++) {
                values[timeIndex * smoother.kStates + stateIndex] = smoother.smoothedState(stateIndex, timeIndex);
            }
        }
        return values;
    }

    private static double[] smoothedSignal(ModelFixture model, SmootherResult smoother, boolean includeObservationIntercept) {
        double[] values = new double[model.nobs * model.kEndog];
        for (int timeIndex = 0; timeIndex < model.nobs; timeIndex++) {
            int designOffset = model.designOffset(timeIndex);
            int interceptOffset = model.obsInterceptOffset(timeIndex);
            for (int observationIndex = 0; observationIndex < model.kEndog; observationIndex++) {
                double value = includeObservationIntercept ? model.obsInterceptData()[interceptOffset + observationIndex] : 0.0;
                for (int stateIndex = 0; stateIndex < model.kStates; stateIndex++) {
                    value += model.designData()[designOffset + observationIndex * model.kStates + stateIndex]
                        * smoother.smoothedState(stateIndex, timeIndex);
                }
                values[timeIndex * model.kEndog + observationIndex] = value;
            }
        }
        return values;
    }
}
