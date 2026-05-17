package com.curioloop.yum4j.ssm.kalman.arena;

public value record SmootherResultLayout(int smoothedStateBase,
                                   int smoothedStateLength,
                                   int smoothedStateCovBase,
                                   int smoothedStateCovLength,
                                   int smoothedStateAutocovarianceBase,
                                   int smoothedStateAutocovarianceLength,
                                   int smoothedObsDisturbanceBase,
                                   int smoothedObsDisturbanceLength,
                                   int smoothedStateDisturbanceBase,
                                   int smoothedStateDisturbanceLength,
                                   int smoothedObsDisturbanceCovBase,
                                   int smoothedObsDisturbanceCovLength,
                                   int smoothedStateDisturbanceCovBase,
                                   int smoothedStateDisturbanceCovLength,
                                   int scaledSmoothedEstimatorBase,
                                   int scaledSmoothedEstimatorLength,
                                   int scaledSmoothedEstimatorCovBase,
                                   int scaledSmoothedEstimatorCovLength,
                                   int smoothingErrorBase,
                                   int smoothingErrorLength,
                                   int innovationsTransitionBase,
                                   int innovationsTransitionLength,
                                   int totalLength) {

    public static SmootherResultLayout create(int scalarWidth,
                                              int kEndog,
                                              int kStates,
                                              int kPosdef,
                                              int nobs,
                                              boolean storeState,
                                              boolean storeStateCovariance,
                                              boolean storeDisturbance,
                                              boolean storeDisturbanceCovariance,
                                              boolean storeAuxiliary) {
                            return create(scalarWidth,
                                kEndog,
                                kStates,
                                kPosdef,
                                nobs,
                                storeState,
                                storeStateCovariance,
                                false,
                                storeDisturbance,
                                storeDisturbanceCovariance,
                                storeAuxiliary);
                            }

                            public static SmootherResultLayout create(int scalarWidth,
                                                  int kEndog,
                                                  int kStates,
                                                  int kPosdef,
                                                  int nobs,
                                                  boolean storeState,
                                                  boolean storeStateCovariance,
                                                  boolean storeStateAutocovariance,
                                                  boolean storeDisturbance,
                                                  boolean storeDisturbanceCovariance,
                                                  boolean storeAuxiliary) {
        int offset = 0;

        int smoothedStateBase = offset;
        int smoothedStateLength = storeState ? scalarWidth * kStates * nobs : 0;
        offset += smoothedStateLength;
        int smoothedStateCovBase = offset;
        int smoothedStateCovLength = storeStateCovariance ? scalarWidth * kStates * kStates * nobs : 0;
        offset += smoothedStateCovLength;
                            int smoothedStateAutocovarianceBase = offset;
                            int smoothedStateAutocovarianceLength = storeStateAutocovariance ? scalarWidth * kStates * kStates * nobs : 0;
                            offset += smoothedStateAutocovarianceLength;
        int smoothedObsDisturbanceBase = offset;
        int smoothedObsDisturbanceLength = storeDisturbance ? scalarWidth * kEndog * nobs : 0;
        offset += smoothedObsDisturbanceLength;
        int smoothedStateDisturbanceBase = offset;
        int smoothedStateDisturbanceLength = storeDisturbance ? scalarWidth * kPosdef * nobs : 0;
        offset += smoothedStateDisturbanceLength;
        int smoothedObsDisturbanceCovBase = offset;
        int smoothedObsDisturbanceCovLength = storeDisturbanceCovariance ? scalarWidth * kEndog * kEndog * nobs : 0;
        offset += smoothedObsDisturbanceCovLength;
        int smoothedStateDisturbanceCovBase = offset;
        int smoothedStateDisturbanceCovLength = storeDisturbanceCovariance ? scalarWidth * kPosdef * kPosdef * nobs : 0;
        offset += smoothedStateDisturbanceCovLength;
        int scaledSmoothedEstimatorBase = offset;
        int scaledSmoothedEstimatorLength = storeAuxiliary ? scalarWidth * kStates * (nobs + 1) : 0;
        offset += scaledSmoothedEstimatorLength;
        int scaledSmoothedEstimatorCovBase = offset;
        int scaledSmoothedEstimatorCovLength = storeAuxiliary ? scalarWidth * kStates * kStates * (nobs + 1) : 0;
        offset += scaledSmoothedEstimatorCovLength;
        int smoothingErrorBase = offset;
        int smoothingErrorLength = storeAuxiliary ? scalarWidth * kEndog * nobs : 0;
        offset += smoothingErrorLength;
        int innovationsTransitionBase = offset;
        int innovationsTransitionLength = storeAuxiliary ? scalarWidth * kStates * kStates * nobs : 0;
        offset += innovationsTransitionLength;

        return new SmootherResultLayout(
            smoothedStateBase,
            smoothedStateLength,
            smoothedStateCovBase,
            smoothedStateCovLength,
            smoothedStateAutocovarianceBase,
            smoothedStateAutocovarianceLength,
            smoothedObsDisturbanceBase,
            smoothedObsDisturbanceLength,
            smoothedStateDisturbanceBase,
            smoothedStateDisturbanceLength,
            smoothedObsDisturbanceCovBase,
            smoothedObsDisturbanceCovLength,
            smoothedStateDisturbanceCovBase,
            smoothedStateDisturbanceCovLength,
            scaledSmoothedEstimatorBase,
            scaledSmoothedEstimatorLength,
            scaledSmoothedEstimatorCovBase,
            scaledSmoothedEstimatorCovLength,
            smoothingErrorBase,
            smoothingErrorLength,
            innovationsTransitionBase,
            innovationsTransitionLength,
            offset);
    }
}