package com.curioloop.yum4j.ssm.kalman.smooth;

import com.curioloop.yum4j.ssm.kalman.init.InitialState;
import com.curioloop.yum4j.ssm.kalman.model.KalmanSSM;

import java.util.Arrays;
import java.util.Objects;

public final class SmootherDiagnostics {

    private SmootherDiagnostics() {
    }

    public record SmoothedStateWeights(int kEndog,
                                       int kStates,
                                       int nobs,
                                       double[] observationWeights,
                                       double[] stateInterceptWeights,
                                       double[] priorWeights) {
        public double observationWeight(int targetTime, int sourceTime, int stateIndex, int observationIndex) {
            return observationWeights[observationOffset(targetTime, sourceTime, stateIndex, observationIndex)];
        }

        public double stateInterceptWeight(int targetTime, int sourceTime, int stateTo, int stateFrom) {
            return stateInterceptWeights[stateInterceptOffset(targetTime, sourceTime, stateTo, stateFrom)];
        }

        public double priorWeight(int targetTime, int stateTo, int stateFrom) {
            return priorWeights[priorOffset(targetTime, stateTo, stateFrom)];
        }

        int observationOffset(int targetTime, int sourceTime, int stateIndex, int observationIndex) {
            return (((targetTime * nobs + sourceTime) * kStates + stateIndex) * kEndog) + observationIndex;
        }

        int stateInterceptOffset(int targetTime, int sourceTime, int stateTo, int stateFrom) {
            return (((targetTime * nobs + sourceTime) * kStates + stateTo) * kStates) + stateFrom;
        }

        int priorOffset(int targetTime, int stateTo, int stateFrom) {
            return ((targetTime * kStates + stateTo) * kStates) + stateFrom;
        }
    }

    public record SmoothedDecomposition(int kEndog,
                                        int kStates,
                                        int nobs,
                                        double[] dataContribution,
                                        double[] observationInterceptContribution,
                                        double[] stateInterceptContribution,
                                        double[] priorContribution) {
        public double[] reconstructedSmoothedState() {
            double[] values = new double[nobs * kStates];
            for (int targetTime = 0; targetTime < nobs; targetTime++) {
                for (int stateIndex = 0; stateIndex < kStates; stateIndex++) {
                    double total = 0.0;
                    for (int sourceTime = 0; sourceTime < nobs; sourceTime++) {
                        for (int observationIndex = 0; observationIndex < kEndog; observationIndex++) {
                            total += dataContribution[observationOffset(targetTime, sourceTime, stateIndex, observationIndex)];
                            total += observationInterceptContribution[observationOffset(targetTime, sourceTime, stateIndex, observationIndex)];
                        }
                        for (int stateFrom = 0; stateFrom < kStates; stateFrom++) {
                            total += stateInterceptContribution[stateInterceptOffset(targetTime, sourceTime, stateIndex, stateFrom)];
                        }
                    }
                    for (int stateFrom = 0; stateFrom < kStates; stateFrom++) {
                        total += priorContribution[priorOffset(targetTime, stateIndex, stateFrom)];
                    }
                    values[targetTime * kStates + stateIndex] = total;
                }
            }
            return values;
        }

        public double[] reconstructedSmoothedSignal(KalmanSSM model) {
            Objects.requireNonNull(model, "model");
            double[] state = reconstructedSmoothedState();
            double[] signal = new double[nobs * kEndog];
            for (int timeIndex = 0; timeIndex < nobs; timeIndex++) {
                int designOffset = model.designOffset(timeIndex);
                for (int observationIndex = 0; observationIndex < kEndog; observationIndex++) {
                    double value = 0.0;
                    for (int stateIndex = 0; stateIndex < kStates; stateIndex++) {
                        value += model.designData()[designOffset + observationIndex * kStates + stateIndex]
                            * state[timeIndex * kStates + stateIndex];
                    }
                    signal[timeIndex * kEndog + observationIndex] = value;
                }
            }
            return signal;
        }

        public double[] reconstructedSmoothedForecast(KalmanSSM model) {
            double[] forecast = reconstructedSmoothedSignal(model);
            for (int timeIndex = 0; timeIndex < nobs; timeIndex++) {
                int interceptOffset = model.obsInterceptOffset(timeIndex);
                for (int observationIndex = 0; observationIndex < kEndog; observationIndex++) {
                    forecast[timeIndex * kEndog + observationIndex] += model.obsInterceptData()[interceptOffset + observationIndex];
                }
            }
            return forecast;
        }

        int observationOffset(int targetTime, int sourceTime, int stateIndex, int observationIndex) {
            return (((targetTime * nobs + sourceTime) * kStates + stateIndex) * kEndog) + observationIndex;
        }

        int stateInterceptOffset(int targetTime, int sourceTime, int stateTo, int stateFrom) {
            return (((targetTime * nobs + sourceTime) * kStates + stateTo) * kStates) + stateFrom;
        }

        int priorOffset(int targetTime, int stateTo, int stateFrom) {
            return ((targetTime * kStates + stateTo) * kStates) + stateFrom;
        }
    }

    public record NewsImpact(int kEndog,
                             int kStates,
                             int targetCount,
                             double[] smoothedStateImpact,
                             double[] smoothedSignalImpact,
                             double[] revisionImpact,
                             double[] newObservationImpact) {
        public double smoothedStateImpact(int stateIndex, int targetTime) {
            return smoothedStateImpact[targetTime * kStates + stateIndex];
        }

        public double smoothedSignalImpact(int observationIndex, int targetTime) {
            return smoothedSignalImpact[targetTime * kEndog + observationIndex];
        }

        public double revisionImpact(int stateIndex, int targetTime) {
            return revisionImpact[targetTime * kStates + stateIndex];
        }

        public double newObservationImpact(int stateIndex, int targetTime) {
            return newObservationImpact[targetTime * kStates + stateIndex];
        }
    }

    public static SmoothedStateWeights computeSmoothedStateWeights(KalmanSSM model,
                                                                   InitialState initialState,
                                                                   SmootherOptions options) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(initialState, "initialState");
        SmootherOptions smootherOptions = stateOptions(options);
        InitialState resolvedInitial = finiteInitial(initialState, model);
        SmootherResult baseline = SmootherEngine.smooth(model, resolvedInitial, smootherOptions);
        int kEndog = model.observationDimension();
        int kStates = model.stateCount();
        int nobs = model.observationCount();
        double[] observationWeights = new double[nobs * nobs * kStates * kEndog];
        double[] stateInterceptWeights = new double[nobs * nobs * kStates * kStates];
        double[] priorWeights = new double[nobs * kStates * kStates];

        for (int sourceTime = 0; sourceTime < nobs; sourceTime++) {
            for (int observationIndex = 0; observationIndex < kEndog; observationIndex++) {
                if (model.isMissing(observationIndex, sourceTime)) {
                    fillObservationColumn(observationWeights, nobs, kStates, kEndog, sourceTime, observationIndex, Double.NaN);
                    continue;
                }
                KalmanSSM perturbed = KalmanSSM.copyOf(model);
                perturbed.endogData()[perturbed.endogOffset(sourceTime) + observationIndex] += 1.0;
                SmootherResult shifted = SmootherEngine.smooth(perturbed, resolvedInitial, smootherOptions);
                writeObservationDifference(baseline, shifted, observationWeights, sourceTime, observationIndex);
            }
        }

        for (int sourceTime = 0; sourceTime < nobs; sourceTime++) {
            for (int stateFrom = 0; stateFrom < kStates; stateFrom++) {
                KalmanSSM perturbed = KalmanSSM.copyOf(model);
                perturbed.stateInterceptData()[perturbed.stateInterceptOffset(sourceTime) + stateFrom] += 1.0;
                SmootherResult shifted = SmootherEngine.smooth(perturbed, resolvedInitial, smootherOptions);
                writeStateInterceptDifference(baseline, shifted, stateInterceptWeights, sourceTime, stateFrom);
            }
        }

        for (int stateFrom = 0; stateFrom < kStates; stateFrom++) {
            double[] shiftedMean = Arrays.copyOf(resolvedInitial.initialState(), kStates);
            shiftedMean[stateFrom] += 1.0;
            InitialState shiftedInitial = InitialState.known(shiftedMean, resolvedInitial.initialStateCov());
            SmootherResult shifted = SmootherEngine.smooth(model, shiftedInitial, smootherOptions);
            writePriorDifference(baseline, shifted, priorWeights, stateFrom);
        }

        return new SmoothedStateWeights(kEndog, kStates, nobs, observationWeights, stateInterceptWeights, priorWeights);
    }

    public static SmoothedDecomposition getSmoothedDecomposition(KalmanSSM model,
                                                                InitialState initialState,
                                                                SmootherOptions options) {
        Objects.requireNonNull(model, "model");
        InitialState resolvedInitial = finiteInitial(initialState, model);
        SmoothedStateWeights weights = computeSmoothedStateWeights(model, resolvedInitial, options);
        int kEndog = model.observationDimension();
        int kStates = model.stateCount();
        int nobs = model.observationCount();
        double[] dataContribution = new double[nobs * nobs * kStates * kEndog];
        double[] observationInterceptContribution = new double[dataContribution.length];
        double[] stateInterceptContribution = new double[nobs * nobs * kStates * kStates];
        double[] priorContribution = new double[nobs * kStates * kStates];

        for (int targetTime = 0; targetTime < nobs; targetTime++) {
            for (int sourceTime = 0; sourceTime < nobs; sourceTime++) {
                int endogOffset = model.endogOffset(sourceTime);
                int obsInterceptOffset = model.obsInterceptOffset(sourceTime);
                int stateInterceptOffset = model.stateInterceptOffset(sourceTime);
                for (int stateIndex = 0; stateIndex < kStates; stateIndex++) {
                    for (int observationIndex = 0; observationIndex < kEndog; observationIndex++) {
                        int offset = weights.observationOffset(targetTime, sourceTime, stateIndex, observationIndex);
                        double weight = weights.observationWeights()[offset];
                        if (!Double.isNaN(weight)) {
                            dataContribution[offset] = weight * model.endogData()[endogOffset + observationIndex];
                            observationInterceptContribution[offset] = -weight * model.obsInterceptData()[obsInterceptOffset + observationIndex];
                        }
                    }
                    for (int stateFrom = 0; stateFrom < kStates; stateFrom++) {
                        int offset = weights.stateInterceptOffset(targetTime, sourceTime, stateIndex, stateFrom);
                        stateInterceptContribution[offset] = weights.stateInterceptWeights()[offset]
                            * model.stateInterceptData()[stateInterceptOffset + stateFrom];
                    }
                }
            }
            for (int stateIndex = 0; stateIndex < kStates; stateIndex++) {
                for (int stateFrom = 0; stateFrom < kStates; stateFrom++) {
                    int offset = weights.priorOffset(targetTime, stateIndex, stateFrom);
                    priorContribution[offset] = weights.priorWeights()[offset] * resolvedInitial.initialState()[stateFrom];
                }
            }
        }

        return new SmoothedDecomposition(kEndog, kStates, nobs,
            dataContribution, observationInterceptContribution, stateInterceptContribution, priorContribution);
    }

    public static NewsImpact news(KalmanSSM previousModel,
                                  InitialState previousInitialState,
                                  KalmanSSM updatedModel,
                                  InitialState updatedInitialState,
                                  SmootherOptions options) {
        Objects.requireNonNull(previousModel, "previousModel");
        Objects.requireNonNull(updatedModel, "updatedModel");
        validateNewsModels(previousModel, updatedModel);
        SmootherOptions smootherOptions = stateOptions(options);
        InitialState previousInitial = finiteInitial(previousInitialState, previousModel);
        InitialState updatedInitial = finiteInitial(updatedInitialState, updatedModel);
        SmootherResult previous = SmootherEngine.smooth(previousModel, previousInitial, smootherOptions);
        SmootherResult updated = SmootherEngine.smooth(updatedModel, updatedInitial, smootherOptions);
        SmoothedStateWeights weights = computeSmoothedStateWeights(updatedModel, updatedInitial, smootherOptions);

        int kEndog = previousModel.observationDimension();
        int kStates = previousModel.stateCount();
        int targetCount = previousModel.observationCount();
        double[] smoothedStateImpact = new double[targetCount * kStates];
        double[] revisionImpact = new double[targetCount * kStates];
        double[] newObservationImpact = new double[targetCount * kStates];
        for (int targetTime = 0; targetTime < targetCount; targetTime++) {
            for (int stateIndex = 0; stateIndex < kStates; stateIndex++) {
                int targetOffset = targetTime * kStates + stateIndex;
                smoothedStateImpact[targetOffset] = updated.smoothedState(stateIndex, targetTime)
                    - previous.smoothedState(stateIndex, targetTime);
                revisionImpact[targetOffset] = revisionContribution(previousModel, updatedModel, weights,
                    targetTime, stateIndex);
                newObservationImpact[targetOffset] = smoothedStateImpact[targetOffset] - revisionImpact[targetOffset];
            }
        }
        double[] smoothedSignalImpact = signalImpact(updatedModel, smoothedStateImpact, targetCount, kEndog, kStates);
        return new NewsImpact(kEndog, kStates, targetCount,
            smoothedStateImpact, smoothedSignalImpact, revisionImpact, newObservationImpact);
    }

    private static InitialState finiteInitial(InitialState initialState, KalmanSSM model) {
        InitialState resolved = initialState.resolve(model);
        if (resolved.initialDiffuseStateCov() != null) {
            throw new UnsupportedOperationException("smoothed-state diagnostics require finite initialization");
        }
        return resolved;
    }

    private static void validateNewsModels(KalmanSSM previousModel, KalmanSSM updatedModel) {
        if (updatedModel.observationCount() < previousModel.observationCount()) {
            throw new IllegalArgumentException("updated model must contain at least as many observations as previous model");
        }
        if (updatedModel.observationDimension() != previousModel.observationDimension()
                || updatedModel.stateCount() != previousModel.stateCount()
                || updatedModel.stateDisturbanceCount() != previousModel.stateDisturbanceCount()) {
            throw new IllegalArgumentException("previous and updated models must have matching dimensions");
        }
    }

    private static double revisionContribution(KalmanSSM previousModel,
                                               KalmanSSM updatedModel,
                                               SmoothedStateWeights weights,
                                               int targetTime,
                                               int stateIndex) {
        double value = 0.0;
        int previousNobs = previousModel.observationCount();
        int kEndog = previousModel.observationDimension();
        for (int sourceTime = 0; sourceTime < previousNobs; sourceTime++) {
            int previousEndogOffset = previousModel.endogOffset(sourceTime);
            int updatedEndogOffset = updatedModel.endogOffset(sourceTime);
            int previousInterceptOffset = previousModel.obsInterceptOffset(sourceTime);
            int updatedInterceptOffset = updatedModel.obsInterceptOffset(sourceTime);
            for (int observationIndex = 0; observationIndex < kEndog; observationIndex++) {
                boolean previousObserved = !previousModel.isMissing(observationIndex, sourceTime);
                boolean updatedObserved = !updatedModel.isMissing(observationIndex, sourceTime);
                if (!previousObserved || !updatedObserved) {
                    continue;
                }
                double previousInnovationInput = previousModel.endogData()[previousEndogOffset + observationIndex]
                    - previousModel.obsInterceptData()[previousInterceptOffset + observationIndex];
                double updatedInnovationInput = updatedModel.endogData()[updatedEndogOffset + observationIndex]
                    - updatedModel.obsInterceptData()[updatedInterceptOffset + observationIndex];
                double delta = updatedInnovationInput - previousInnovationInput;
                if (delta != 0.0) {
                    value += weights.observationWeight(targetTime, sourceTime, stateIndex, observationIndex) * delta;
                }
            }
        }
        return value;
    }

    private static double[] signalImpact(KalmanSSM model,
                                         double[] stateImpact,
                                         int targetCount,
                                         int kEndog,
                                         int kStates) {
        double[] values = new double[targetCount * kEndog];
        for (int targetTime = 0; targetTime < targetCount; targetTime++) {
            int designOffset = model.designOffset(targetTime);
            for (int observationIndex = 0; observationIndex < kEndog; observationIndex++) {
                double value = 0.0;
                for (int stateIndex = 0; stateIndex < kStates; stateIndex++) {
                    value += model.designData()[designOffset + observationIndex * kStates + stateIndex]
                        * stateImpact[targetTime * kStates + stateIndex];
                }
                values[targetTime * kEndog + observationIndex] = value;
            }
        }
        return values;
    }

    private static SmootherOptions stateOptions(SmootherOptions options) {
        SmootherOptions smootherOptions = options == null ? SmootherOptions.conventional() : options;
        return smootherOptions.includes(SmootherOptions.Surface.STATE)
            ? smootherOptions
            : smootherOptions.with(SmootherOptions.Surface.STATE);
    }

    private static void fillObservationColumn(double[] weights,
                                              int nobs,
                                              int kStates,
                                              int kEndog,
                                              int sourceTime,
                                              int observationIndex,
                                              double value) {
        for (int targetTime = 0; targetTime < nobs; targetTime++) {
            for (int stateIndex = 0; stateIndex < kStates; stateIndex++) {
                int offset = (((targetTime * nobs + sourceTime) * kStates + stateIndex) * kEndog) + observationIndex;
                weights[offset] = value;
            }
        }
    }

    private static void writeObservationDifference(SmootherResult baseline,
                                                   SmootherResult shifted,
                                                   double[] weights,
                                                   int sourceTime,
                                                   int observationIndex) {
        int nobs = baseline.nobs;
        int kStates = baseline.kStates;
        int kEndog = baseline.kEndog;
        for (int targetTime = 0; targetTime < nobs; targetTime++) {
            for (int stateIndex = 0; stateIndex < kStates; stateIndex++) {
                int offset = (((targetTime * nobs + sourceTime) * kStates + stateIndex) * kEndog) + observationIndex;
                weights[offset] = shifted.smoothedState(stateIndex, targetTime) - baseline.smoothedState(stateIndex, targetTime);
            }
        }
    }

    private static void writeStateInterceptDifference(SmootherResult baseline,
                                                      SmootherResult shifted,
                                                      double[] weights,
                                                      int sourceTime,
                                                      int stateFrom) {
        int nobs = baseline.nobs;
        int kStates = baseline.kStates;
        for (int targetTime = 0; targetTime < nobs; targetTime++) {
            for (int stateTo = 0; stateTo < kStates; stateTo++) {
                int offset = (((targetTime * nobs + sourceTime) * kStates + stateTo) * kStates) + stateFrom;
                weights[offset] = shifted.smoothedState(stateTo, targetTime) - baseline.smoothedState(stateTo, targetTime);
            }
        }
    }

    private static void writePriorDifference(SmootherResult baseline,
                                             SmootherResult shifted,
                                             double[] weights,
                                             int stateFrom) {
        int kStates = baseline.kStates;
        for (int targetTime = 0; targetTime < baseline.nobs; targetTime++) {
            for (int stateTo = 0; stateTo < kStates; stateTo++) {
                int offset = ((targetTime * kStates + stateTo) * kStates) + stateFrom;
                weights[offset] = shifted.smoothedState(stateTo, targetTime) - baseline.smoothedState(stateTo, targetTime);
            }
        }
    }
}
