package com.curioloop.yum4j.kalman.arena;

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
        int offset = 0;

        int predictedStateBase = offset;
        int predictedStateLength = storePredicted ? scalarWidth * kStates * (nobs + 1) : 0;
        offset += predictedStateLength;
        int predictedStateCovBase = offset;
        int predictedStateCovLength = storePredicted ? scalarWidth * kStates * kStates * (nobs + 1) : 0;
        offset += predictedStateCovLength;
        int filteredStateBase = offset;
        int filteredStateLength = storeFiltered ? scalarWidth * kStates * nobs : 0;
        offset += filteredStateLength;
        int filteredStateCovBase = offset;
        int filteredStateCovLength = storeFiltered ? scalarWidth * kStates * kStates * nobs : 0;
        offset += filteredStateCovLength;
        int forecastBase = offset;
        int forecastLength = storeForecast ? scalarWidth * kEndog * nobs : 0;
        offset += forecastLength;
        int forecastErrorBase = offset;
        int forecastErrorLength = storeForecast ? scalarWidth * kEndog * nobs : 0;
        offset += forecastErrorLength;
        int forecastErrorCovBase = offset;
        int forecastErrorCovLength = storeForecast ? scalarWidth * kEndog * kEndog * nobs : 0;
        offset += forecastErrorCovLength;
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
            kalmanGainBase,
            kalmanGainLength,
            logLikelihoodObsBase,
            logLikelihoodObsLength,
            offset);
    }
}