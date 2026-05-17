package com.curioloop.yum4j.ssm.kalman.arena;

public value record FilterResultLayout(int predictedStateBase,
                                 int predictedStateLength,
                                 int predictedStateCovBase,
                                 int predictedStateCovLength,
                                 int filteredStateBase,
                                 int filteredStateLength,
                                 int filteredStateCovBase,
                                 int filteredStateCovLength,
                                 int forecastBase,
                                 int forecastLength,
                                 int forecastErrorBase,
                                 int forecastErrorLength,
                                 int forecastErrorCovBase,
                                 int forecastErrorCovLength,
                                 int standardizedForecastErrorBase,
                                 int standardizedForecastErrorLength,
                                 int kalmanGainBase,
                                 int kalmanGainLength,
                                 int logLikelihoodObsBase,
                                 int logLikelihoodObsLength,
                                 int totalLength) {

    public static FilterResultLayout create(int scalarWidth,
                                            int kEndog,
                                            int kStates,
                                            int nobs,
                                            boolean storeForecast,
                                            boolean storePredicted,
                                            boolean storeFiltered,
                                            boolean storeGain,
                                            boolean storeLikelihood) {
        return create(scalarWidth,
            kEndog,
            kStates,
            nobs,
            storeForecast,
            storeForecast,
            storeForecast,
            false,
            storePredicted,
            storeFiltered,
            storeGain,
            storeLikelihood);
    }

    public static FilterResultLayout create(int scalarWidth,
                                            int kEndog,
                                            int kStates,
                                            int nobs,
                                            boolean storeForecastMean,
                                            boolean storeForecastError,
                                            boolean storeForecastCovariance,
                                            boolean storeStandardizedForecastError,
                                            boolean storePredicted,
                                            boolean storeFiltered,
                                            boolean storeGain,
                                            boolean storeLikelihood) {
        return create(scalarWidth,
            kEndog,
            kStates,
            nobs,
            storeForecastMean,
            storeForecastError,
            storeForecastCovariance,
            storeStandardizedForecastError,
            storePredicted,
            storePredicted,
            storeFiltered,
            storeFiltered,
            storeGain,
            storeLikelihood);
    }

    public static FilterResultLayout create(int scalarWidth,
                                            int kEndog,
                                            int kStates,
                                            int nobs,
                                            boolean storeForecastMean,
                                            boolean storeForecastError,
                                            boolean storeForecastCovariance,
                                            boolean storeStandardizedForecastError,
                                            boolean storePredictedState,
                                            boolean storePredictedStateCovariance,
                                            boolean storeFilteredState,
                                            boolean storeFilteredStateCovariance,
                                            boolean storeGain,
                                            boolean storeLikelihood) {
        int offset = 0;

        int predictedStateBase = offset;
        int predictedStateLength = storePredictedState ? scalarWidth * kStates * (nobs + 1) : 0;
        offset += predictedStateLength;
        int predictedStateCovBase = offset;
        int predictedStateCovLength = storePredictedStateCovariance ? scalarWidth * kStates * kStates * (nobs + 1) : 0;
        offset += predictedStateCovLength;
        int filteredStateBase = offset;
        int filteredStateLength = storeFilteredState ? scalarWidth * kStates * nobs : 0;
        offset += filteredStateLength;
        int filteredStateCovBase = offset;
        int filteredStateCovLength = storeFilteredStateCovariance ? scalarWidth * kStates * kStates * nobs : 0;
        offset += filteredStateCovLength;
        int forecastBase = offset;
        int forecastLength = storeForecastMean ? scalarWidth * kEndog * nobs : 0;
        offset += forecastLength;
        int forecastErrorBase = offset;
        int forecastErrorLength = storeForecastError ? scalarWidth * kEndog * nobs : 0;
        offset += forecastErrorLength;
        int forecastErrorCovBase = offset;
        int forecastErrorCovLength = storeForecastCovariance ? scalarWidth * kEndog * kEndog * nobs : 0;
        offset += forecastErrorCovLength;
        int standardizedForecastErrorBase = offset;
        int standardizedForecastErrorLength = storeStandardizedForecastError ? scalarWidth * kEndog * nobs : 0;
        offset += standardizedForecastErrorLength;
        int kalmanGainBase = offset;
        int kalmanGainLength = storeGain ? scalarWidth * kStates * kEndog * nobs : 0;
        offset += kalmanGainLength;
        int logLikelihoodObsBase = offset;
        int logLikelihoodObsLength = storeLikelihood ? nobs : 0;
        offset += logLikelihoodObsLength;

        return new FilterResultLayout(
            predictedStateBase,
            predictedStateLength,
            predictedStateCovBase,
            predictedStateCovLength,
            filteredStateBase,
            filteredStateLength,
            filteredStateCovBase,
            filteredStateCovLength,
            forecastBase,
            forecastLength,
            forecastErrorBase,
            forecastErrorLength,
            forecastErrorCovBase,
            forecastErrorCovLength,
            standardizedForecastErrorBase,
            standardizedForecastErrorLength,
            kalmanGainBase,
            kalmanGainLength,
            logLikelihoodObsBase,
            logLikelihoodObsLength,
            offset);
    }
}