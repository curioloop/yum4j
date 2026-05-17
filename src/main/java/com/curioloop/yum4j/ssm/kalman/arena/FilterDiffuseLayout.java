package com.curioloop.yum4j.ssm.kalman.arena;

public value record FilterDiffuseLayout(int predictedDiffuseStateCovBase,
                                  int predictedDiffuseStateCovLength,
                                  int forecastErrorDiffuseCovBase,
                                  int forecastErrorDiffuseCovLength,
                                  int totalLength) {

    public static FilterDiffuseLayout create(int scalarWidth,
                                             int kEndog,
                                             int kStates,
                                             int nobs,
                                             boolean storePredicted,
                                             boolean storeForecast) {
        int offset = 0;

        int predictedDiffuseStateCovBase = offset;
        int predictedDiffuseStateCovLength = storePredicted ? scalarWidth * kStates * kStates * (nobs + 1) : 0;
        offset += predictedDiffuseStateCovLength;
        int forecastErrorDiffuseCovBase = offset;
        int forecastErrorDiffuseCovLength = storeForecast ? scalarWidth * kEndog * kEndog * nobs : 0;
        offset += forecastErrorDiffuseCovLength;

        return new FilterDiffuseLayout(
            predictedDiffuseStateCovBase,
            predictedDiffuseStateCovLength,
            forecastErrorDiffuseCovBase,
            forecastErrorDiffuseCovLength,
            offset);
    }
}