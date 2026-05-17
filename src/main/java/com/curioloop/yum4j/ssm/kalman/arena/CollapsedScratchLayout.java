package com.curioloop.yum4j.ssm.kalman.arena;

public value record CollapsedScratchLayout(int logLikelihoodAdjustmentBase,
                                           int designBase,
                                           int obsCovBase,
                                           int obsCovFactorBase,
                                           int hInvZBase,
                                           int informationBase,
                                           int collapseFactorBase,
                                           int collapsedDesignBase,
                                           int collapsedObsCovBase,
                                           int collapsedEndogBase,
                                           int zHyBase,
                                           int reconstructedBase,
                                           int residualBase,
                                           int solvedResidualBase,
                                           int observedEndogBase,
                                           int transitionBase,
                                           int stateInterceptBase,
                                           int selectionBase,
                                           int stateCovBase,
                                           int forecastMeanBase,
                                           int forecastErrorBase,
                                           int forecastErrorCovBase,
                                           int forecastErrorDiffuseCovBase,
                                           int factorBase,
                                           int zPBase,
                                           int solvedErrorBase,
                                           int covarianceWorkBase,
                                           int selectedErrorBase,
                                           int selectedStandardizedBase,
                                           int selectedCovarianceBase,
                                           int totalLength) {

    public static CollapsedScratchLayout create(int kEndog,
                                                int kStates,
                                                int kPosdef,
                                                int nobs) {
        return create(kEndog, kStates, kPosdef, nobs, true, true, true, true, true);
    }

    public static CollapsedScratchLayout create(int kEndog,
                                                int kStates,
                                                int kPosdef,
                                                int nobs,
                                                boolean needForecast,
                                                boolean needForecastCovariance,
                                                boolean needForecastDiffuseCovariance,
                                                boolean needStandardizedForecastError,
                                                boolean needKalmanGain) {
        needForecast |= needStandardizedForecastError || needKalmanGain;
        needForecastCovariance |= needStandardizedForecastError || needKalmanGain;
        int keKe = kEndog * kEndog;
        int ksKs = kStates * kStates;
        int offset = 0;

        int logLikelihoodAdjustmentBase = -1;

        int phaseBase = offset;

        int collapseOffset = phaseBase;
        int designBase = collapseOffset;
        collapseOffset += kEndog * kStates;
        int obsCovBase = collapseOffset;
        collapseOffset += keKe;
        int obsCovFactorBase = collapseOffset;
        collapseOffset += keKe;
        int hInvZBase = collapseOffset;
        collapseOffset += kEndog * kStates;
        int informationBase = collapseOffset;
        collapseOffset += ksKs;
        int collapseFactorBase = collapseOffset;
        collapseOffset += ksKs;
        int collapsedDesignBase = collapseOffset;
        collapseOffset += ksKs;
        int collapsedObsCovBase = collapseOffset;
        collapseOffset += ksKs;
        int collapsedEndogBase = collapseOffset;
        collapseOffset += kStates;
        int zHyBase = collapseOffset;
        collapseOffset += kStates;
        int reconstructedBase = collapseOffset;
        collapseOffset += kStates;
        int residualBase = collapseOffset;
        collapseOffset += kEndog;
        int solvedResidualBase = collapseOffset;
        collapseOffset += kEndog;
        int observedEndogBase = collapseOffset;
        collapseOffset += kEndog;

        int reconstructionOffset = phaseBase;
        int forecastMeanBase = -1;
        int forecastErrorBase = -1;
        if (needForecast) {
            forecastMeanBase = reconstructionOffset;
            reconstructionOffset += kEndog;
            forecastErrorBase = reconstructionOffset;
            reconstructionOffset += kEndog;
        }
        int forecastErrorCovBase = -1;
        if (needForecastCovariance) {
            forecastErrorCovBase = reconstructionOffset;
            reconstructionOffset += keKe;
        }
        boolean needSharedMatrix = needForecastDiffuseCovariance || needStandardizedForecastError || needKalmanGain;
        int sharedMatrixBase = -1;
        if (needSharedMatrix) {
            sharedMatrixBase = reconstructionOffset;
            reconstructionOffset += keKe;
        }
        int forecastErrorDiffuseCovBase = needForecastDiffuseCovariance ? sharedMatrixBase : -1;
        int factorBase = needStandardizedForecastError ? sharedMatrixBase : -1;
        int covarianceWorkBase = needKalmanGain ? sharedMatrixBase : -1;
        int zPBase = -1;
        if (needKalmanGain) {
            zPBase = reconstructionOffset;
            reconstructionOffset += kEndog * kStates;
        }
        boolean needSharedVector = needStandardizedForecastError || needKalmanGain;
        int sharedVectorBase = -1;
        if (needSharedVector) {
            sharedVectorBase = reconstructionOffset;
            reconstructionOffset += kEndog;
        }
        int solvedErrorBase = needKalmanGain ? sharedVectorBase : -1;
        int selectedErrorBase = needSharedVector ? forecastMeanBase : -1;
        int selectedStandardizedBase = needStandardizedForecastError ? sharedVectorBase : -1;
        int selectedCovarianceBase = -1;
        if (needSharedVector) {
            selectedCovarianceBase = reconstructionOffset;
            reconstructionOffset += keKe;
        }

        offset = Math.max(collapseOffset, reconstructionOffset);

        int transitionBase = -1;
        int stateInterceptBase = -1;
        int selectionBase = -1;
        int stateCovBase = -1;

        return new CollapsedScratchLayout(
            logLikelihoodAdjustmentBase,
            designBase,
            obsCovBase,
            obsCovFactorBase,
            hInvZBase,
            informationBase,
            collapseFactorBase,
            collapsedDesignBase,
            collapsedObsCovBase,
            collapsedEndogBase,
            zHyBase,
            reconstructedBase,
            residualBase,
            solvedResidualBase,
            observedEndogBase,
            transitionBase,
            stateInterceptBase,
            selectionBase,
            stateCovBase,
            forecastMeanBase,
            forecastErrorBase,
            forecastErrorCovBase,
            forecastErrorDiffuseCovBase,
            factorBase,
            zPBase,
            solvedErrorBase,
            covarianceWorkBase,
            selectedErrorBase,
            selectedStandardizedBase,
            selectedCovarianceBase,
            offset);
    }
}
